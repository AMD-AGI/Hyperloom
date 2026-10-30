# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Turning the kernels a trace shows into the candidate set the orchestrator acts on."""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Any

from hyperloom.common import kernel_source_contract as _KSC
from hyperloom.common.io import JsonFormat, atomic_write_json

from ._harness_discovery import find_benchmark_files, find_repo_root
from ._io_utils import append_log
from ._kernel_classification import (
    classify_patchability,
    extract_shape,
    extract_source_file,
    is_kernel_event,
    is_runtime_generated_kernel,
    is_vendor_dispatch_wrapper,
    source_type_for,
)
from ._kernel_roofline import _resolve_flydsl_source_fallback
from ._trace_inputs import open_json
from ._trace_shapes import (
    _FUSED_MOE_KERNEL_MARKER,
    resolve_fused_moe_invocation_cases_from_csv,
    resolve_fused_moe_shapes_from_csv,
    resolve_invocation_cases_from_csv,
    resolve_raw_arg_spec_from_csv,
    resolve_shapes_from_csv_for_op,
)
from ._vendor_operator_playbooks import match_vendor_operator_playbook, resolve_kernel_anchor_path
from .tracelens_skill_runner import normalize_upstream_category

log = logging.getLogger(__name__)

# Kernel-name markers per GEAK category, in priority order: the first category with a marker in the name wins.
_NAME_CATEGORY_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "GEMM",
        (
            "gemm",
            "matmul",
            "rocblas",
            "hipblas",
            "cijk",
            "sgemm",
            "hgemm",
            # PyTorch op-name variants.
            "::mm",
            "::addmm",
            "::bmm",
        ),
    ),
    ("SDPA", ("attention", "attn", "fmha", "paged_attention", "flash")),
    ("LayerNorm", ("rmsnorm", "layernorm", "norm_kernel")),
    ("Activation", ("act_and_mul", "silu", "gelu", "activation")),
    ("MoE", ("moe", "topk", "expert")),
    ("Softmax", ("softmax",)),
    ("Embedding", ("embed",)),
    ("Communication", ("reduce", "all_reduce", "all_gather")),
    ("Triton", ("triton",)),
    ("Elementwise", ("elementwise", "binary")),
)


def derive_kernel_category(candidate: dict[str, Any]) -> str:
    """Map a candidate to its GEAK-facing kernel category.

    Priority: explicit TraceLens category (normalized), then a kernel-name
    heuristic, then ``unknown``.

    Args:
        candidate: The hot-kernel candidate dict.

    Returns:
        The GEAK-facing kernel category string.
    """
    cat = (candidate.get("tracelens_category") or "").strip()
    if cat:
        return normalize_upstream_category(cat)
    if candidate.get("source_type") == "flydsl":
        return "FlyDSL"
    name = str(candidate.get("name") or "").lower()
    for category, markers in _NAME_CATEGORY_MARKERS:
        if any(marker in name for marker in markers):
            return category
    return "unknown"


def is_multigpu_kernel(name: str, source_file: str) -> bool:
    """Heuristic: kernel is a multi-GPU collective if name/source hints it.

    Args:
        name (str): Kernel symbol/name.
        source_file (str): Resolved source-file path (may be empty).

    Returns:
        bool: ``True`` when the name/source contains a collective /
            distributed marker.
    """
    blob = f"{name} {source_file}".lower()
    return any(
        tag in blob
        for tag in (
            "all_reduce",
            "allreduce",
            "all_gather",
            "allgather",
            "reduce_scatter",
            "reducescatter",
            "all_to_all",
            "alltoall",
            "broadcast",
            "p2p",
            "send_recv",
            "cross_device",
            "rank_signal",
            "ranksignals",
            "/dist/",
            "dist/",
            "communicator",
        )
    )


def analyze_trace_files(
    trace_files: list[Path],
    top_k: int,
) -> list[dict[str, Any]]:
    """Aggregate GPU kernels across raw trace files into top-K candidates.

    Sums per-kernel duration and call counts across all events, takes the
    top ``top_k`` by duration, then runs :func:`_finalize_candidates`. This
    is the dry-run / test-only raw-trace path (production reads candidates
    from the orchestrator's analysis.json).

    Args:
        trace_files (list[Path]): Trace files (optionally gzipped) to scan.
        top_k (int): Number of hottest kernels to keep.

    Returns:
        list[dict[str, Any]]: Finalized hot-kernel candidate dicts.
    """
    aggregates: dict[str, dict[str, Any]] = {}
    total_dur = 0.0

    for trace_file in trace_files:
        try:
            payload = open_json(trace_file)
        except ValueError:
            continue

        if isinstance(payload.get("kernels"), list):
            events = payload["kernels"]
        else:
            events = payload.get("traceEvents", [])
        if not isinstance(events, list):
            continue

        for event in events:
            if not isinstance(event, dict) or not is_kernel_event(event):
                continue
            name = str(event.get("kernel_name") or event.get("name") or "unknown_kernel")
            dur = float(event.get("dur") or event.get("duration_us") or event.get("duration") or 0)
            if dur <= 0:
                continue
            total_dur += dur
            item = aggregates.setdefault(
                name,
                {
                    "name": name,
                    "duration_us": 0.0,
                    "call_count": 0,
                    "source_file": "",
                    "source_type": "unknown",
                    "shapes": [],
                },
            )
            item["duration_us"] += dur
            item["call_count"] += 1
            if not item.get("_extracted_source_checked"):
                item["source_file"] = extract_source_file(event)
                item["_extracted_source_checked"] = True
            shape = extract_shape(event)
            if shape and shape not in item["shapes"]:
                item["shapes"].append(shape)

    candidates = sorted(aggregates.values(), key=lambda x: x["duration_us"], reverse=True)
    top = candidates[:top_k]
    # This raw-trace path never runs the analysis.json reader, so it fills its own
    # per-kernel GPU share and records that any source came straight from the
    # trace event, not TraceLens' resolver.
    denom = total_dur or 1.0
    for item in top:
        item["gpu_pct"] = round(item["duration_us"] / denom * 100.0, 3)
        if item.get("source_file"):
            item["source_resolution_method"] = "trace_kernel_file"
    return _finalize_candidates(top)


def load_op_category_map(
    perf_report_csv_dir: Path | str,
) -> dict[str, str]:
    """Read a ``{name: raw op category}`` map from the unified perf summary.

    Args:
        perf_report_csv_dir: Directory containing
            ``unified_perf_summary.csv``.

    Returns:
        A mapping of kernel name to its first non-empty TraceLens op category,
        or ``{}`` when the CSV is absent or unreadable.
    """
    csv_path = Path(perf_report_csv_dir) / "unified_perf_summary.csv"
    if not csv_path.is_file():
        return {}
    out: dict[str, str] = {}
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = str(row.get("name") or "").strip()
                cat = str(row.get("op category") or "").strip()
                if name and cat and name not in out:
                    out[name] = cat
    except (OSError, csv.Error):
        return {}
    return out


def _is_fused_moe_candidate(item: dict[str, Any]) -> bool:
    """Detect the Triton fused-MoE expert-kernel candidate.

    Matches by op name or the ``moe_fused`` category.

    Args:
        item: The candidate dict to test.

    Returns:
        ``True`` when the candidate is the fused-MoE expert kernel.
    """
    name = str(item.get("name") or "").lower()
    if _FUSED_MOE_KERNEL_MARKER in name:
        return True
    cat = str(item.get("tracelens_category") or "").strip().lower()
    return cat == "moe_fused" and "moe" in name


def _stamp_candidate_metadata(item: dict[str, Any], op_cat_map: dict[str, str] | None) -> None:
    """Stamp routing, backend, and category metadata onto a finalized candidate."""
    # source_file was resolved to a real path (or left empty) by the reader's
    # resolve_kernel_source call, so the only thing left to note is whether the
    # resolved path is present on this host.
    source_file = str(item.get("source_file") or "").strip()
    if source_file and not os.path.isfile(source_file):
        # Path-shaped but absent: keep it (the resolution may target the serving
        # container) and leave a breadcrumb for triage.
        item["source_file_missing_on_disk"] = True
    reusable, skip_reason = classify_patchability(item)
    item["reusable_native_kernel"] = reusable
    item["skip_reason"] = skip_reason
    playbook = match_vendor_operator_playbook(item)
    if playbook is not None:
        item["patch_strategy"] = "vendor_playbook"
        item["vendor_operator_playbook"] = playbook
        item["vendor_playbook_role"] = playbook.get("role", "")
        # Coordinator kernel handlers gate dispatch on a non-empty, path-shaped
        # source_file before they will send work to any backend; a vendor
        # playbook candidate has no rewritable device source, so point that
        # field at the task bundle's anchor file instead of leaving it
        # empty (which would otherwise fall through as "missing_native_source").
        #
        # The anchor also overrides whatever the grep tier guessed, and does so
        # whether or not the guess was right. A registry match is a curated
        # statement that this operator is tuned through a task bundle, so the
        # device source is not the file a backend should be handed even when
        # the guess found it -- and the guess can just as easily be a same-word
        # collision (``mori::EpDispatchCombineOp::dispatch`` reduces to the
        # keyword "dispatch" and lands on an unrelated vendor header).
        # Unconditional, and safe to be: the registry refuses an entry with no
        # ``kernel_anchor``, so ``resolve_kernel_anchor_path`` cannot come back
        # empty for a playbook that matched. Guarding it here instead cost two
        # commits and produced a real deadlock -- the guard's marker gated on
        # ``source_file``, so a row with neither anchor nor path read as
        # anchor-backed, went into ``protected_ids``, and the row most in need of
        # the review was the one refused it. The shape it defended against does
        # not occur in the data; the shape it created did.
        anchor = resolve_kernel_anchor_path(playbook)
        if source_file and source_file != anchor:
            # Keep the displaced path for triage: an unconditional override is
            # only auditable if the value it replaced is still recorded.
            item["source_file_superseded_by_playbook"] = source_file
        item["source_file"] = anchor
    item["benchmark_files"] = find_benchmark_files(
        item["name"], item.get("kernel_repo", ""), item.get("source_file", "")
    )
    # The nccl-summary lane identifies collectives from TraceLens' own nccl_summary table plus a resolved device
    # symbol, which outranks a name guess. Re-deriving over it unresolves rows the lane already resolved:
    # small_collective, EpDispatchIntraNodeKernel_bf16 and ncclDevKernel_Generic_1 all carry is_multigpu=True from
    # _nccl_summary_candidates and all read False by name. That flip reaches the invocation contract built below,
    # which gates kind == "collective" and world_size on this flag -- so the rows that most need a rank count would
    # be published as ordinary single-rank work.
    authoritative = bool(item.get("is_multigpu")) and str(item.get("candidate_source") or "") == "nccl_summary"
    if not authoritative:
        item["is_multigpu"] = is_multigpu_kernel(item["name"], item.get("source_file", ""))
    item["num_gpus_recommended"] = 2 if item["is_multigpu"] else 1
    item["recommended_backends"] = recommend_backends(item)
    item["optimization_notes"] = build_notes(item)
    if op_cat_map and not str(item.get("tracelens_category") or "").strip():
        csv_cat = op_cat_map.get(str(item.get("name") or ""))
        if csv_cat:
            item["tracelens_category"] = csv_cat
    item["kernel_category"] = derive_kernel_category(item)
    item.setdefault("source_path", item.get("source_file", ""))


def _recover_csv_invocations(item: dict[str, Any], perf_report_csv_dir: Path | str | None) -> None:
    """Fill a candidate's missing shapes, invocation cases and raw arg spec from ``ops_unique_args.csv``.

    The fused-MoE expert kernel's trace event carries no Input Dims, so it takes the sidecar's fused-MoE rows
    instead of an exact-name match.
    """
    had_observed_shapes = bool(item.get("shapes"))
    operation = str(item.get("name") or "")
    if _is_fused_moe_candidate(item):
        fused_moe_shapes = resolve_fused_moe_shapes_from_csv(perf_report_csv_dir)
        invocation_cases = resolve_fused_moe_invocation_cases_from_csv(perf_report_csv_dir)
        if not item.get("shapes") and fused_moe_shapes:
            item["shapes"] = fused_moe_shapes
            item["shape_provenance"] = "torch_trace"
    else:
        invocation_cases = resolve_invocation_cases_from_csv(perf_report_csv_dir, operation)
    if not item.get("shapes"):
        csv_shapes = resolve_shapes_from_csv_for_op(perf_report_csv_dir, operation)
        if csv_shapes:
            item["shapes"] = csv_shapes
            item["shape_provenance"] = "torch_trace"
    if invocation_cases:
        if had_observed_shapes:
            invocation_cases = [
                {
                    "operation": operation,
                    "input_shapes": (item.get("input_shapes") or item.get("shapes") or []),
                    "input_dtypes": item.get("input_dtypes") or [],
                    "raw_arg_spec": item.get("raw_arg_spec") or {},
                },
                *invocation_cases,
            ]
        item["invocation_cases"] = invocation_cases
    # Mark trace-extracted shapes so the dispatch-time validator knows their provenance.
    if item.get("shapes"):
        item.setdefault("shape_provenance", "torch_trace")
    # Forward the full ordered arg metadata so GEAK's harness builder can
    # reconstruct the exact call signature.
    if not item.get("raw_arg_spec"):
        raw_spec = (
            invocation_cases[0].get("raw_arg_spec")
            if invocation_cases
            else resolve_raw_arg_spec_from_csv(perf_report_csv_dir, operation)
        )
        if raw_spec:
            item["raw_arg_spec"] = raw_spec


def _finalize_candidates(
    top: list[dict[str, Any]],
    *,
    perf_report_csv_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Apply host-only post-processing to reader-resolved candidate rows.

    Recovers CSV shapes, classifies source type, recommends backends, gates
    patchability, and attaches notes; mutates ``top`` in place. Source paths were
    already resolved by the ``analysis.json`` reader through TraceLens.

    Args:
        top: The candidate rows to finalize (mutated in place).
        perf_report_csv_dir: Optional CSV directory used to populate each
            item's ``tracelens_category`` and recover missing shapes.

    Returns:
        The finalized candidate list (the same ``top`` object).
    """
    op_cat_map = load_op_category_map(perf_report_csv_dir) if perf_report_csv_dir is not None else {}
    # Denominator for the duration-derived gpu_pct fallback below.
    sum_dur = sum(float(it.get("duration_us") or 0.0) for it in top) or 1.0
    for idx, item in enumerate(top, 1):
        item.pop("_extracted_source_checked", None)
        item.setdefault("source_file", "")
        item.setdefault("source_type", "unknown")
        item.setdefault("shapes", [])
        _recover_csv_invocations(item, perf_report_csv_dir)
        item["kernel_id"] = f"k{idx:03d}"
        item["duration_us"] = round(item["duration_us"], 3)
        # Keep TraceLens' authoritative pct_e2e when present, including a real
        # 0.0; derive from the duration share only when it is absent/None (e.g.
        # collectives and null-pct_e2e members).
        if item.get("gpu_pct") is None:
            item["gpu_pct"] = round(float(item.get("duration_us") or 0.0) / sum_dur * 100.0, 3)
        # source_file was already resolved by the analysis.json reader through
        # TraceLens' resolve_kernel_source; classify its type host-side and record
        # the repo it lives in for the invocation spec.
        item["kernel_repo"] = find_repo_root(item.get("source_file", ""))
        item["source_type"] = source_type_for(item["name"], item.get("source_file", ""))
        # FlyDSL pseudo-ops carry no real source_file; inject the real FlyDSL
        # MoE kernel source before the patchability gate.
        if item["source_type"] == "flydsl":
            _sf = str(item.get("source_file") or "").strip()
            if not os.path.isfile(_sf):
                _fb = _resolve_flydsl_source_fallback()
                if _fb:
                    item["source_file"] = _fb
                    item["kernel_repo"] = find_repo_root(_fb) or item["kernel_repo"]
                    item["flydsl_source_from_fallback"] = True
        # Downgrade thin dispatch wrappers to vendor_binary so recommend_backends drops them.
        if item["source_type"] != "vendor_binary" and is_vendor_dispatch_wrapper(
            item["name"], item.get("source_file", "")
        ):
            item["source_type"] = "vendor_binary"
            item["vendor_dispatch_wrapper"] = True
        item["runtime_generated_kernel"] = is_runtime_generated_kernel(item["name"], item.get("source_file", ""))
        _stamp_candidate_metadata(item, op_cat_map)
    _apply_vendor_operator_playbook_grouping(top)
    return top


def _apply_vendor_operator_playbook_grouping(top: list[dict[str, Any]]) -> None:
    """Sum GPU share across a vendor playbook's sibling roles; mutates ``top``.

    mori's dispatch and combine are two separate kernel launches under the
    hood, but this deliberately invokes them as **one** Forge task/session
    (see KernelForge PR #88): each is gated on the *sum* of dispatch's +
    combine's ``gpu_pct``, since together they are one logical round trip.
    Each member keeps its own candidate entry (so either one can be picked as
    the ``--kernel-id`` the orchestrator dispatches); the vendor-playbook
    dispatch path de-duplicates so only one forge-loop session actually
    runs per group per analysis session.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in top:
        if not isinstance(item, dict) or item.get("patch_strategy") != "vendor_playbook":
            continue
        playbook = item.get("vendor_operator_playbook")
        group_id = str(playbook.get("id") or "") if isinstance(playbook, dict) else ""
        if not group_id:
            continue
        groups.setdefault(group_id, []).append(item)
    for group_id, members in groups.items():
        aggregate = round(sum(float(m.get("gpu_pct") or 0.0) for m in members), 3)
        member_ids = [str(m.get("kernel_id") or "") for m in members]
        floor: float | None = None
        for member in members:
            playbook = member.get("vendor_operator_playbook")
            if isinstance(playbook, dict) and playbook.get("min_gpu_pct_floor") is not None:
                try:
                    floor = float(playbook["min_gpu_pct_floor"])
                except (TypeError, ValueError):
                    floor = None
                break
        for member in members:
            member["vendor_playbook_group_id"] = group_id
            # Namespaced (not bare "aggregate_gpu_pct"): that name is already
            # an existing task_group-level concept (the sum of GPU% across a
            # task_group's rows -- see tracelens_skill_runner.py),
            # stamped on task_group dicts, not candidate
            # rows. Candidate rows don't carry it today, so reusing the name
            # here is currently harmless, but any future code that flattens
            # a task_group onto its candidate rows would silently change
            # every ordinary group's gating behavior the moment it did (PR
            # #1191 review finding #7).
            member["vendor_playbook_aggregate_gpu_pct"] = aggregate
            member["vendor_playbook_group_kernel_ids"] = list(member_ids)
            if floor is not None:
                # Consumed by effective_hot_kernel_min_gpu_pct() in the
                # orchestrator's hot-kernel gate (kernel_decisions.py /
                # request_handlers.py) so a heavier forge-loop-session
                # playbook isn't dispatched below its own floor even when
                # HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT is loosened elsewhere.
                member["vendor_playbook_min_gpu_pct_floor"] = floor


def _candidate_resolution_method(item: dict[str, Any]) -> str:
    """Map a finalized candidate onto the artifact's method vocabulary.

    The reader always stamps one of TraceLens' ``resolve_kernel_source`` methods;
    an unrecognized value degrades to ``unresolved`` rather than failing the write.
    """
    stamped = str(item.get("source_resolution_method") or "").strip()
    if stamped in _KSC.KNOWN_METHODS:
        return stamped
    return _KSC.METHOD_UNRESOLVED


def build_source_resolution_entries(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project finalized candidates onto source-resolution entries."""
    entries: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        method = _candidate_resolution_method(item)
        line = item.get("source_line")
        entry = _KSC.SourceResolutionEntry(
            kernel_id=str(item.get("kernel_id") or ""),
            name=str(item.get("name") or ""),
            gpu_pct=float(item.get("gpu_pct") or 0.0),
            source_file=str(item.get("source_file") or ""),
            source_line=int(line) if isinstance(line, (int, float)) else None,
            source_function=str(item.get("source_function") or ""),
            method=method,
            reason=str(item.get("source_resolution_reason") or ""),
            # Classify from the routing gate's own verdict, so the artifact
            # reports why a kernel is undispatchable rather than only how many.
            reason_class=_KSC.classify_skip_reason(
                reusable=item.get("reusable_native_kernel"),
                skip_reason=item.get("skip_reason"),
                source_file=item.get("source_file"),
            ),
        ).to_dict()
        entries.append(entry)
    return entries


def write_source_resolution_artifact(
    candidates: list[dict[str, Any]],
    out_path: Path | str,
    *,
    framework: str = "",
    model_name: str = "",
    log_path: Path | str | None = None,
) -> Path | None:
    """Write the source-resolution artifact next to the candidate report.

    One file answering "where does each hot kernel live, and how do we know",
    so a reviewer does not have to reconstruct it from candidate internals.
    Failure is non-fatal: the artifact is for humans and the review tier, and
    the candidates themselves already carry the same fields.
    """
    try:
        entries = build_source_resolution_entries(candidates)
        doc = _KSC.make_document(
            entries,
            generated_by="tracelens_analysis",
            model_name=model_name,
            framework=framework,
        )
        problems = _KSC.validate_document(doc)
        if problems:
            log.warning(
                "source-resolution artifact failed its own contract (%d issue(s)): %s",
                len(problems),
                "; ".join(problems[:3]),
            )
            return None
        path = Path(out_path)
        atomic_write_json(path, doc, fmt=JsonFormat(trailing_newline=True))
        summary = _KSC.summarize_resolution(doc.get("entries") or [])
        line = f"source resolution: {summary['located']}/{summary['total']} kernel(s) located -> {path.name}"
        # A count alone cannot say whether the unlocated kernels were worth
        # chasing, so report the GPU share behind each class next to it.
        if summary["undispatchable_gpu_pct"] > 0.0:
            line += (
                f"; {summary['undispatchable_gpu_pct']:.1f}% GPU undispatchable"
                f" ({summary['recoverable']} recoverable, {summary['unsalvageable']} unsalvageable)"
            )
        for reason_class, count in sorted(summary["by_class"].items()):
            line += f"; {reason_class}={count}"
        log.info("%s", line)
        if log_path:
            append_log(log_path, line)
        return path
    except (OSError, TypeError, ValueError, AttributeError, KeyError) as exc:
        # Reporting aid: a write or projection failure must not fail the run,
        # but it is logged with its type so a silent gap is not mistaken for
        # "every kernel resolved".
        log.warning("could not write source-resolution artifact: %r", exc)
        return None


def recommend_backends(candidate: dict[str, Any]) -> list[str]:
    """Recommend a backend ladder for a reusable native kernel.

    Recommends the forge backend.

    Args:
        candidate: The hot-kernel candidate dict.

    Returns:
        The recommended backend names, or ``[]`` for unresolved source,
        non-reusable, vendor-binary, or runtime-generated kernels.
    """
    source_type = candidate.get("source_type")
    if not candidate.get("source_file"):
        return []
    if not candidate.get("reusable_native_kernel", classify_patchability(candidate)[0]):
        return []
    if source_type == "vendor_binary":
        return []
    if source_type == "runtime_generated":
        return []
    return ["forge"]


def build_notes(candidate: dict[str, Any]) -> str:
    """Build a short human-readable optimization note for a candidate.

    Args:
        candidate (dict[str, Any]): A finalized hot-kernel candidate row.

    Returns:
        str: A note describing whether/why the candidate is routable
            (resolved source vs. an explanation of why it was skipped).
    """
    if not candidate.get("source_file"):
        return "source file not resolved; backend dispatch will be skipped"
    if candidate.get(
        "runtime_generated_kernel",
        is_runtime_generated_kernel(str(candidate.get("name") or ""), str(candidate.get("source_file") or "")),
    ):
        return "runtime-generated torch.compile/Inductor kernel; not reusable, kernel-opt disabled"
    if not candidate.get("reusable_native_kernel", classify_patchability(candidate)[0]):
        return "not a reusable native source; kernel-opt disabled"
    return f"resolved source: {candidate['source_file']}"
