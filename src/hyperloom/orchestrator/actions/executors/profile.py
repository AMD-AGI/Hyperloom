# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real ``profile`` ActionRunner — Magpie run with torch profiler on."""

from __future__ import annotations

import contextlib
import gzip
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from hyperloom.orchestrator.trace_analysis._capture_shapes import is_capture_fragment
from hyperloom.orchestrator.trace_analysis._trace_rank import (
    select_primary_trace,
    trace_rank as _trace_rank,
)
from hyperloom.common.io import atomic_write_json, safe_mtime
from hyperloom.common.profile_args import sanitize_profile_server_args as _sanitize_profile_server_args
from hyperloom.common.timeutil import now_iso
from hyperloom.inference_optimizer.session.paths import asset_root, mn_profile_trace_root
from hyperloom.inference_optimizer import framework_registry
from ._inferencex_patcher import (
    benchmark_serving_path_in,
    ensure_benchmark_lib_patched,
    ensure_benchmark_lib_eval_dest_patched,
    ensure_benchmark_serving_patched,
)
from .benchmark_result import round_succeeded
from .baseline import BenchmarkRunExecutor


log = logging.getLogger(__name__)


# Leading bytes of a trace to sample for sentinel substrings.
_TRACE_INSPECT_BYTES = 2_000_000

# Cap for the confirmation streaming scan used when the leading-window sample finds zero of a sentinel.
_TRACE_CONFIRM_BYTES = 64_000_000

# Min fraction of ``cpu_op`` events carrying ``Input Dims`` for a healthy ``capture_traces/`` file (Deval ref 99.97%;
# gated low to avoid false-positives).
_INPUT_DIMS_FRACTION_FLOOR = 0.90

# Kineto puts the annotation category in ``cat`` and the label the framework wrote in ``name``, so a marker keyed on
# ``"name": "user_annotation"`` looks for a label no producer emits.
_USER_ANNOTATION_MARKER = '"user_annotation"'


def _trace_contains(path: Path, substring: str) -> bool:
    """Stream-decompress ``path`` for ``substring``, reading at most :data:`_TRACE_CONFIRM_BYTES`."""
    read = 0
    carry = ""
    chunk_size = 4_000_000
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            while read < _TRACE_CONFIRM_BYTES:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                read += len(chunk)
                if substring in (carry + chunk):
                    return True
                # Carry tail to catch a sentinel split across the chunk boundary.
                carry = chunk[-(len(substring)) :]
    except (OSError, EOFError, UnicodeDecodeError) as e:
        log.debug("_trace_contains: cannot stream %s: %s", path, e)
        return False
    return False


def _sample_trace_text(path: Path) -> str | None:
    """Read up to ``_TRACE_INSPECT_BYTES`` of decompressed text from a gzipped trace."""
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            return fh.read(_TRACE_INSPECT_BYTES)
    except (OSError, EOFError, UnicodeDecodeError) as e:
        # Best-effort: a malformed sample must not fail the profile path.
        log.debug(
            "_validate_trace_structure: cannot sample %s: %s",
            path,
            e,
        )
        return None


# Structured verdict ids for the post-profile trace validation.
CHECK_INSTRUMENTATION_PREFLIGHT = "instrumentation_preflight"
CHECK_CAPTURE_TRACES_PRESENT = "capture_traces_present"
CHECK_CAPTURE_INPUT_DIMS = "capture_input_dims"
CHECK_STEP_ANNOTATIONS = "step_annotations"
CHECK_SPLIT_CHUNK_ANNOTATIONS = "split_chunk_annotations"
CHECK_SGLANG_SHAPE_PROFILER = "sglang_shape_profiler"
CHECK_STEADY_STATE_SPLIT_NAMING = "steady_state_split_naming"
CHECK_TRACE_HAS_OPS = "trace_has_ops"
CHECK_GRAPH_LAUNCH_COVERAGE = "graph_launch_coverage"
CHECK_RANK_SHAPE = "rank_shape"


def _check_row(
    check_id: str,
    *,
    status: str,
    skip_reason: str | None = None,
    **detail: Any,
) -> dict[str, Any]:
    """Build one structured check row."""
    return {"check_id": check_id, "status": status, "skip_reason": skip_reason, "detail": detail}


def _instrumentation_preflight_row(bench: Any, patchers: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """State, before the run, whether the annotations the trace checks look for can land at all.

    When the TraceLens runtime patch is unavailable the env layer turns ``detailed_annotations`` and
    ``shape_discovery`` off, which makes checks 3 and 5 certain to fail. Recording that decision here gives the
    post-hoc failures their cause. This only reports it -- the run proceeds unchanged, because a trace without
    annotations is still a trace.

    ``patchers`` carries each patcher's own outcome. The env block records what the patch results *caused*, which
    is not the same as which patcher ran and what it returned: a successful patch records its outcome too, so
    "instrumentation was fine" and "nobody looked" are different records.
    """
    envs = (bench or {}).get("envs") if isinstance(bench, dict) else None
    if not isinstance(envs, dict):
        return _check_row(
            CHECK_INSTRUMENTATION_PREFLIGHT,
            status="skipped",
            skip_reason="materialized config carries no benchmark.envs block",
        )
    degraded = str(envs.get("HYPERLOOM_PROFILE_DEGRADED_REASON") or "")
    patchers = dict(patchers or {})
    try:
        extra_body = json.loads(str(envs.get("PROFILE_EXTRA_BODY") or "{}"))
    except (TypeError, ValueError):
        extra_body = {}
    if not isinstance(extra_body, dict):
        extra_body = {}
    failed_patchers = sorted(name for name, ok in patchers.items() if ok is False)
    return _check_row(
        CHECK_INSTRUMENTATION_PREFLIGHT,
        status="failed" if (degraded or failed_patchers) else "passed",
        degraded_reason=degraded,
        tracelens_patch_status=str(envs.get("HYPERLOOM_TRACELENS_PATCH_STATUS") or ""),
        detailed_annotations=extra_body.get("detailed_annotations"),
        shape_discovery=extra_body.get("shape_discovery"),
        # The sglang flag the patched build exposes; absent means capture-time shapes were never requested.
        shape_discovery_flag_present=(
            "shape-discovery-for-cuda-graph-profile" in str(envs.get("EXTRA_SGLANG_ARGS") or "")
        ),
        patchers=patchers,
        failed_patchers=failed_patchers,
        # Named so a reader knows which post-hoc checks this predicts rather than having to rediscover the link.
        predicts_failure_of=[CHECK_STEP_ANNOTATIONS, CHECK_SGLANG_SHAPE_PROFILER] if degraded else [],
    )


def _probe_check_rows(certificate: dict[str, Any]) -> list[dict[str, Any]]:
    """Express the probe's measurements in the shared check vocabulary."""
    inventory = certificate.get("trace_dir_level") or {}
    thresholds = (certificate.get("verdict") or {}).get("thresholds_effective") or {}
    ranks = [row for row in (certificate.get("rank_level") or []) if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []

    for rank in ranks:
        density = rank.get("density") or {}
        measured = {
            "rank": rank.get("rank"),
            "graph_mode": density.get("graph_mode"),
            "graph_launch_count": density.get("graph_launch_count"),
            "graph_launches_with_kernels": density.get("graph_launches_with_kernels"),
            "graph_launch_coverage": density.get("graph_launch_coverage"),
            "coverage_max": thresholds.get("graph_launch_coverage_max"),
        }
        under_recorded = density.get("graph_under_recorded")
        if under_recorded is None:
            # Nothing was measured, which is not the same as measured and fine.
            rows.append(
                _check_row(
                    CHECK_GRAPH_LAUNCH_COVERAGE,
                    status="skipped",
                    skip_reason="the probe reached no coverage measurement for this rank",
                    **measured,
                )
            )
        elif not density.get("graph_mode"):
            # An eager capture has no graph launches, so the coverage denominator is zero.
            rows.append(
                _check_row(
                    CHECK_GRAPH_LAUNCH_COVERAGE,
                    status="skipped",
                    skip_reason="capture recorded no CUDA graph launches",
                    **measured,
                )
            )
        else:
            rows.append(
                _check_row(
                    CHECK_GRAPH_LAUNCH_COVERAGE,
                    status="failed" if under_recorded else "passed",
                    **measured,
                )
            )

    rank_count = inventory.get("rank_count")
    certified = [rank.get("rank") for rank in ranks]
    shape = {
        "rank_count": rank_count,
        "certified_rank_count": len(ranks),
        "certified_ranks": certified,
    }
    if isinstance(rank_count, int) and rank_count > len(ranks):
        # The probe certifies the file the live resolver would open, so on a tensor-parallel capture the other ranks
        # are unmeasured rather than measured and equal.
        rows.append(
            _check_row(
                CHECK_RANK_SHAPE,
                status="skipped",
                skip_reason="only the resolved rank was certified; the remaining ranks are unmeasured",
                **shape,
            )
        )
    else:
        rows.append(_check_row(CHECK_RANK_SHAPE, status="passed", **shape))
    return rows


@dataclass(frozen=True)
class _TraceProbe:
    """One self-certification probe over a trace dir: where it ran, and its certificate or why it produced none."""

    trace_dir: Path
    framework: str
    certificate: dict[str, Any] | None = None
    error: str = ""


def _build_trace_validate(
    health: dict[str, Any],
    probe: _TraceProbe,
    *,
    preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the profile-stage trace validation block.

    ``preflight`` leads the check list because it was known before the run: when it failed, the trace checks that
    follow are consequences rather than independent findings.
    """
    checks = [row for row in (health.get("checks") or []) if isinstance(row, dict)]
    if isinstance(preflight, dict):
        checks = [preflight] + checks
    certificate = probe.certificate or {}
    if certificate:
        checks = checks + _probe_check_rows(certificate)
        probe_status = "ok"
    else:
        probe_status = "failed" if probe.error else "skipped"
    return {
        "schema_version": certificate.get("schema_version"),
        "probe_version": certificate.get("probe_version"),
        "probe_status": probe_status,
        "probe_error": str(probe.error or ""),
        "checked_at": now_iso(timespec="seconds"),
        "trace_dir": str(probe.trace_dir),
        "framework": str(probe.framework or ""),
        "verdict": certificate.get("verdict") or {},
        "trace_dir_level": certificate.get("trace_dir_level") or {},
        "rank_level": certificate.get("rank_level") or [],
        "checks": checks,
    }


def _write_trace_certificate(trace_dir: Path, validate: dict[str, Any]) -> str:
    """Write the full certificate beside the trace it describes, returning the path (empty when unwritable).

    Deliberately written *outside* ``trace_dir``: the resolver's ``_trace_candidates`` rglobs that directory for
    anything ending in ``_TRACE_EXTS``, which includes a bare ``.json``, so a certificate stored among the traces
    becomes a trace candidate itself. That is not hypothetical -- it makes ``require_single_rank`` resolution
    return nothing (a second unranked candidate) and lets an 82 KB certificate outrank a small real trace in the
    size fallback. A subdirectory would not help, the scan is recursive. The name carries ``trace_dir``'s so the
    ``torch_trace`` and ``capture_traces`` certificates of one workspace do not collide.
    """
    target = trace_dir.parent / f"{trace_dir.name}.selfcert.json"
    try:
        atomic_write_json(target, validate)
    except OSError as exc:
        log.warning("profile_executor: cannot write trace certificate to %s: %s", target, exc)
        return ""
    return str(target)


def _certify_trace_dir(trace_dir: Path, framework: str) -> dict[str, Any]:
    """Run the capture-time self-certification probe over a profile trace."""
    from hyperloom.orchestrator.trace_analysis import trace_selfcert

    return trace_selfcert.certify_trace_dir(
        trace_dir,
        framework=framework or trace_selfcert.read_framework(trace_dir),
    )


def _capture_trace_files(trace_dir: Path) -> tuple[Path | None, list[Path]]:
    capture_dirs = (
        trace_dir / "capture_traces",
        trace_dir / "graph_capture_profile",
        trace_dir / "graph_capture",
    )
    capture = next((path for path in capture_dirs if path.is_dir()), None)
    if capture is None:
        return None, []
    return capture, sorted(p for p in capture.iterdir() if p.is_file())


def _main_trace_and_text(trace_dir: Path) -> tuple[Path | None, str | None]:
    main_traces = sorted(
        (p for p in trace_dir.glob("*.trace.json.gz") if p.is_file()),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    if not main_traces:
        return None, None
    return main_traces[0], _sample_trace_text(main_traces[0])


def _check_capture_traces(capture: Path | None, capture_files: list[Path]) -> tuple[dict, str]:
    if capture is None:
        return _check_row(CHECK_CAPTURE_TRACES_PRESENT, status="failed", capture_dir_present=False, file_count=0), (
            "[1] graph-capture subdirectory missing — graph capture "
            "didn't fire. Verify EXTRA_VLLM_ARGS / EXTRA_SGLANG_ARGS "
            "include the TraceLens flag and the server-side patch landed."
        )
    row = _check_row(
        CHECK_CAPTURE_TRACES_PRESENT,
        status="passed" if capture_files else "failed",
        capture_dir_present=True,
        capture_dir=str(capture),
        file_count=len(capture_files),
    )
    if capture_files:
        return row, ""
    return row, f"[1] {capture.name}/ exists but is empty — graph capture path fired but produced no files."


def _check_capture_input_dims(capture_files: list[Path]) -> tuple[dict, str]:
    # Sample the heaviest capture file; gate the cpu_op-with-Input-Dims fraction at _INPUT_DIMS_FRACTION_FLOOR.
    if not capture_files:
        return _check_row(CHECK_CAPTURE_INPUT_DIMS, status="skipped", skip_reason="no capture file to sample"), ""
    target = max(capture_files, key=lambda p: p.stat().st_size)
    text = _sample_trace_text(target)
    if text is None:
        return _check_row(
            CHECK_CAPTURE_INPUT_DIMS,
            status="skipped",
            skip_reason="capture file could not be sampled",
            sampled_file=target.name,
        ), ""
    # ``cpu_op`` is the event's category; its ``name`` is the operator (``aten::mm``). Matching on the
    # name key found nothing in any real trace, so this check reported "no cpu_op events" on captures
    # that were fully instrumented, and the advisory below rationalised the miss as an SGLang naming
    # quirk. Verified against a real sglang capture: 0 hits for the name form, 2915 for this one.
    cpu_op_count = text.count('"cat": "cpu_op"')
    input_dims_count = text.count('"Input Dims"')
    if not cpu_op_count:
        # Zero cpu_op leaves no fraction to judge, and on ROCm/SGLang it is an event-naming difference rather
        # than a capture failure -- which is what the advisory says.
        return _check_row(
            CHECK_CAPTURE_INPUT_DIMS,
            status="skipped",
            skip_reason="no literal cpu_op events to measure the Input Dims fraction against",
            sampled_file=target.name,
            cpu_op_count=cpu_op_count,
            input_dims_count=input_dims_count,
            input_dims_fraction=None,
            floor=_INPUT_DIMS_FRACTION_FLOOR,
        ), (
            f"[2] capture file {target.name} has no literal "
            f"'cpu_op' events in the first "
            f"{_TRACE_INSPECT_BYTES // 1_000_000} MB — on ROCm/SGLang "
            "this is often just an event-naming difference (kernels "
            "logged under 'sglang_profiler::*'); cross-check Check "
            "[5] (kernel_shape_profiler) and the server log before "
            "treating it as a capture regression."
        )
    fraction = input_dims_count / cpu_op_count
    row = _check_row(
        CHECK_CAPTURE_INPUT_DIMS,
        status="passed" if fraction >= _INPUT_DIMS_FRACTION_FLOOR else "failed",
        sampled_file=target.name,
        cpu_op_count=cpu_op_count,
        input_dims_count=input_dims_count,
        input_dims_fraction=fraction,
        floor=_INPUT_DIMS_FRACTION_FLOOR,
    )
    if fraction >= _INPUT_DIMS_FRACTION_FLOOR:
        return row, ""
    return row, (
        f"[2] capture file {target.name}: only {100.0 * fraction:.1f}% of "
        f"cpu_op events carry 'Input Dims' (expected ≥ "
        f"{int(_INPUT_DIMS_FRACTION_FLOOR * 100)}%). Shape-"
        "discovery instrumentation may not be fully active — "
        "verify TraceLens server patch and capture flag."
    )


def _check_step_annotations(main_trace: Path | None, main_text: str | None, scriptable: bool) -> tuple[dict, str]:
    # ``execute_*`` annotations are InferenceX per-step writes when detailed_annotations is honoured (distinct
    # from check 5).
    if main_trace is None:
        return _check_row(
            CHECK_STEP_ANNOTATIONS, status="skipped", skip_reason="no *.trace.json.gz in the trace dir"
        ), ""
    if main_text is None:
        return _check_row(
            CHECK_STEP_ANNOTATIONS,
            status="skipped",
            skip_reason="main trace could not be sampled",
            sampled_file=main_trace.name,
        ), ""
    user_ann_count = main_text.count(_USER_ANNOTATION_MARKER)
    execute_count = main_text.count('"execute_')
    # ``execute_*`` labels are the real health signal; ``user_annotation`` presence is profiler-version-dependent.
    confirmed_absent = (
        not scriptable
        and execute_count == 0
        and user_ann_count == 0
        and not (_trace_contains(main_trace, '"execute_') or _trace_contains(main_trace, _USER_ANNOTATION_MARKER))
    )
    row = _check_row(
        CHECK_STEP_ANNOTATIONS,
        status="failed" if confirmed_absent else "passed",
        sampled_file=main_trace.name,
        execute_annotation_count=execute_count,
        user_annotation_count=user_ann_count,
        confirmed_absent=confirmed_absent,
    )
    if not confirmed_absent:
        return row, ""
    return row, (
        f"[3] main trace {main_trace.name} has no "
        "execute_* / user_annotation events — InferenceX "
        "per-step annotations didn't fire. Verify "
        "detailed_annotations reached the framework "
        "(PROFILE_EXTRA_BODY consumed; see #210)."
    )


# Checks 4 and 6: the per-step split chunks carry annotations and are named for the steady state.
def _check_split_chunks(split: Path, scriptable: bool) -> list[tuple[dict, str]]:
    if not split.is_dir() or scriptable:
        skip_reason = "scriptable framework has no per-step split" if scriptable else "no trace_split/ directory"
        return [
            (_check_row(CHECK_SPLIT_CHUNK_ANNOTATIONS, status="skipped", skip_reason=skip_reason), ""),
            (_check_row(CHECK_STEADY_STATE_SPLIT_NAMING, status="skipped", skip_reason=skip_reason), ""),
        ]
    split_files = sorted(p for p in split.iterdir() if p.is_file())
    return [_check_split_chunk_annotations(split_files), _check_steady_state_naming(split_files)]


def _check_split_chunk_annotations(split_files: list[Path]) -> tuple[dict, str]:
    empty_splits = [sp.name for sp in split_files if _split_chunk_is_empty(sp)]
    chunk_row = _check_row(
        CHECK_SPLIT_CHUNK_ANNOTATIONS,
        status="failed" if (split_files and empty_splits) else "passed",
        split_file_count=len(split_files),
        empty_chunk_count=len(empty_splits),
        empty_chunk_samples=empty_splits[:3],
    )
    if not (split_files and empty_splits):
        return chunk_row, ""
    return chunk_row, (
        f"[4] {len(empty_splits)} trace_split/ file(s) have NO "
        "execute_* user_annotations: "
        f"{', '.join(empty_splits[:3])}"
        + (f" (and {len(empty_splits) - 3} more)" if len(empty_splits) > 3 else "")
        + " — splitter ran but the chunks are empty. Likely the "
        "trace lacks the per-step annotations needed for splitting "
        "(see check [3])."
    )


def _check_steady_state_naming(split_files: list[Path]) -> tuple[dict, str]:
    names = [p.name for p in split_files]
    has_extend = any("_extend_" in n or "extend_only_" in n for n in names)
    has_decode = any("_decode_" in n or "decode_only_" in n for n in names)
    has_steady_state = any("steady_state" in n for n in names)
    unnamed = (has_extend or has_decode) and not has_steady_state
    naming_row = _check_row(
        CHECK_STEADY_STATE_SPLIT_NAMING,
        status="failed" if unnamed else "passed",
        has_extend=has_extend,
        has_decode=has_decode,
        has_steady_state=has_steady_state,
    )
    if not unnamed:
        return naming_row, ""
    return naming_row, (
        "[6] trace_split/ has _extend_* / _decode_* files but NO "
        "_steady_state_* — profile_by_stage=True leaked through, "
        "PROFILE_EXTRA_BODY env was not consumed by the framework. "
        "Confirm _inferencex_patcher patched Magpie's bundled "
        "InferenceX (#210; check $MAGPIE_PATH/InferenceX/utils/"
        "bench_serving/benchmark_serving.py)."
    )


def _split_chunk_is_empty(chunk: Path) -> bool:
    if not chunk.name.endswith(".json.gz"):
        return False
    text = _sample_trace_text(chunk)
    return text is not None and text.count('"execute_') == 0


def _check_trace_has_ops(main_trace: Path | None) -> tuple[dict, str]:
    # A metadata-only trace (no ``cpu_op`` / ``kernel`` events) means the profiler active window never recorded
    # real execution; flag it so roofline re-profiles rather than caching an empty snapshot.
    if main_trace is None:
        return _check_row(CHECK_TRACE_HAS_OPS, status="skipped", skip_reason="no *.trace.json.gz in the trace dir"), ""
    has_ops = _trace_contains(main_trace, '"cat": "cpu_op"') or _trace_contains(main_trace, '"cat": "kernel"')
    row = _check_row(
        CHECK_TRACE_HAS_OPS,
        status="passed" if has_ops else "failed",
        sampled_file=main_trace.name,
        has_ops=has_ops,
    )
    if has_ops:
        return row, ""
    return row, (
        f"[7] main trace {main_trace.name} has NO cpu_op / kernel "
        "events — the torch-profiler active window recorded nothing "
        "(metadata-only trace). On xDiT/diffusion this is the "
        "torch.profiler.schedule repeat=0 discard (the active window is "
        "dropped when the schedule restarts after the last active "
        "step). The trace is unusable for roofline; re-profile needed."
    )


def _check_sglang_shape_profiler(framework: str, main_trace: Path | None, main_text: str | None) -> tuple[dict, str]:
    if framework.lower() != "sglang":
        return _check_row(
            CHECK_SGLANG_SHAPE_PROFILER,
            status="skipped",
            skip_reason=f"framework is {framework or 'unset'}, not sglang",
        ), ""
    if main_text is None or main_trace is None:
        return _check_row(
            CHECK_SGLANG_SHAPE_PROFILER, status="skipped", skip_reason="main trace could not be sampled"
        ), ""
    # Shape markers left by either mechanism: the sglang_profiler:: op
    # namespace, or the kernel_shape_profiler frame (when with_stack is on).
    present = any(marker in main_text for marker in ("sglang_profiler::", "kernel_shape_profiler"))
    row = _check_row(
        CHECK_SGLANG_SHAPE_PROFILER,
        status="passed" if present else "failed",
        sampled_file=main_trace.name,
        sampled_bytes=_TRACE_INSPECT_BYTES,
    )
    if present:
        return row, ""
    return row, (
        f"[5] sglang main trace ({main_trace.name}, sampled "
        f"first {_TRACE_INSPECT_BYTES // 1_000_000} MB) lacks "
        "kernel-shape events — shape discovery didn't reach the live "
        "SGLang. For SGLang < 0.5.18 verify the _server_patcher "
        "git-apply succeeded; for >= 0.5.18 verify the kernel_shape_tool "
        "is on the server PYTHONPATH and TRACELENS_SHAPE_DISCOVERY=1 "
        "(check log warnings)."
    )


def _validate_trace_structure(
    trace_dir: Path,
    framework: str,
) -> dict[str, Any]:
    """Post-profile sanity check on the produced trace structure."""
    # A roofline-composite ctx carries no framework, and an empty name resolves to the serving default — which fires
    # every serving-only check below against a scriptable trace. $FRAMEWORK is the session-wide lock, so fall back to
    # it.
    framework = str(framework or os.environ.get("FRAMEWORK", "") or "")
    # Scriptable image frameworks (xDiT diffusion) produce a plain torch-profiler trace, so checks 1-6
    # (LLM/serving-specific) would emit spurious warnings; only check 7 (zero_ops) is meaningful and always runs.
    from hyperloom.inference_optimizer import framework_registry as _fw_reg

    scriptable = _fw_reg.is_scriptable(framework)
    capture_files: list[Path] = []
    if scriptable:
        skip = "scriptable framework writes a plain torch-profiler trace"
        capture_result = (_check_row(CHECK_CAPTURE_TRACES_PRESENT, status="skipped", skip_reason=skip), "")
    else:
        capture, capture_files = _capture_trace_files(trace_dir)
        capture_result = _check_capture_traces(capture, capture_files)
    main_trace, main_text = _main_trace_and_text(trace_dir)

    results = [
        capture_result,
        _check_capture_input_dims(capture_files),
        _check_step_annotations(main_trace, main_text, scriptable),
        *_check_split_chunks(trace_dir / "trace_split", scriptable),
        _check_trace_has_ops(main_trace),
        _check_sglang_shape_profiler(framework, main_trace, main_text),
    ]
    # Structured mirror of ``issues``: same findings, but with the measured values attached so a consumer can
    # compare attempts instead of diffing prose.
    checks = [row for row, _issue in results]
    issues = [issue for _row, issue in results if issue]
    detail = {row["check_id"]: row["detail"] for row in checks}
    if issues:
        for issue in issues:
            log.warning("trace structure check: %s", issue)
        log.warning(
            "trace structure check: %d issue(s) detected — TraceLens "
            "downstream analysis may be degraded. See per-issue messages "
            "above for the actionable check.",
            len(issues),
        )
    return {
        "issues": issues,
        "per_kernel_attribution_degraded": bool(detail[CHECK_STEP_ANNOTATIONS].get("confirmed_absent")),
        "capture_traces_present": bool(capture_files),
        "zero_ops": not detail[CHECK_TRACE_HAS_OPS].get("has_ops", True),
        "checks": checks,
    }


# Fold the composite's base overrides into the caller's profile params.
def _merge_profile_params(params: dict[str, Any]) -> None:
    from hyperloom.inference_optimizer.grid_server_args import merge_server_args

    merged_args = merge_server_args(
        _sanitize_profile_server_args(str(params.get("base_extra_args") or "").strip()),
        _sanitize_profile_server_args(str(params.get("extra_server_args") or "")),
    )
    if merged_args:
        params["extra_server_args"] = merged_args
    else:
        params.pop("extra_server_args", None)
    merged_envs: dict[str, Any] = {}
    for envs in (params.get("base_extra_envs"), params.get("extra_envs")):
        if isinstance(envs, dict):
            merged_envs.update(envs)
    if merged_envs:
        params["extra_envs"] = merged_envs
    else:
        params.pop("extra_envs", None)
    for base_key, key in (("base_remove_args", "remove_args"), ("base_unset_envs", "unset_envs")):
        raw = params.get(base_key)
        if raw and key not in params:
            params[key] = [raw] if isinstance(raw, str) else list(raw or [])
    if str(params.get("base_args_mode") or "").strip().lower() == "replace":
        params.setdefault("args_mode", "replace")


def _mn_round_trace_root() -> str:
    """Return the shared torch-trace base dir for multi-node, or ''."""
    from ._multi_node_env import is_multi_node

    if not is_multi_node():
        return ""
    provisioned = os.environ.get("HYPERLOOM_MN_PROFILE_TRACE_DIR", "").strip()
    if provisioned:
        return provisioned
    scoped = mn_profile_trace_root() / f"default-{os.getpid()}" / "torch_trace"
    try:
        scoped.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning(
            "cannot mkdir multi-node profile fallback dir %s: %s; downstream readers may FileNotFoundError",
            scoped,
            exc,
        )
    return str(scoped)


# Restarts the multi-node server for this profiling round; returns the refusal result when it failed.
async def _restart_round_server(params: dict[str, Any], round_trace_root: str) -> dict[str, Any] | None:
    from ._multi_node_server_lifecycle import RoundServer, ServerRestartFailed, restart_server_for_round

    try:
        await restart_server_for_round(
            RoundServer.from_env(
                str(params.get("extra_server_args") or ""),
                str(params.get("model_path") or "").strip() or os.environ.get("MODEL_PATH") or None,
            ),
            torch_profiler_dir=round_trace_root,
        )
    except ServerRestartFailed as exc:
        return {
            "status": "failed",
            "error_class": "mn_server_restart_failed",
            "error": str(exc),
            "trace_dir": round_trace_root,
        }
    return None


# Runs the body with a warmup-then-bounded Infera engine profiling window around it.
@contextlib.asynccontextmanager
async def _infera_profile_window(round_trace_root: str):
    import asyncio as _asyncio

    from ._multi_node_server_lifecycle import trigger_infera_engine_profile

    prof_body: dict[str, Any] = {}
    try:
        parsed = json.loads(os.environ.get("PROFILE_EXTRA_BODY") or "{}")
    except (ValueError, TypeError):
        parsed = {}
    if isinstance(parsed, dict):
        prof_body = {k: v for (k, v) in parsed.items() if k in ("output_dir",)}
    if round_trace_root:
        prof_body.setdefault("output_dir", round_trace_root)
    warmup_s = float(os.environ.get("HYPERLOOM_MN_PROFILE_WARMUP_S", "60") or 60)
    window_s = float(os.environ.get("HYPERLOOM_MN_PROFILE_WINDOW_S", "8") or 8)
    started = {"v": False}

    async def _bounded_profile_window() -> None:
        await _asyncio.sleep(warmup_s)
        await trigger_infera_engine_profile("start", prof_body)
        started["v"] = True
        await _asyncio.sleep(window_s)
        await trigger_infera_engine_profile("stop")
        started["v"] = False

    prof_task = _asyncio.create_task(_bounded_profile_window())
    try:
        yield
    finally:
        if not prof_task.done():
            prof_task.cancel()
            try:
                await prof_task
            except (_asyncio.CancelledError, OSError, RuntimeError, ValueError) as exc:
                log.debug("profile_executor: bounded profile task ended during cleanup: %r", exc)
        if started.get("v"):
            await trigger_infera_engine_profile("stop")


# Attaches this round's multi-node trace files, selecting the heaviest as the main trace.
def _record_mn_round_traces(result: dict[str, Any], round_trace_root: str, task_started_unix: float) -> None:
    trace_dir = Path(round_trace_root)
    if not trace_dir.is_dir():
        result["trace_dir"] = None
        result["trace_files"] = []
        log.warning(
            "profile_executor: round trace dir %s does not exist after magpie completed; check sglang server logs for torch profiler errors",
            round_trace_root,
        )
        return
    all_files = sorted(trace_dir.glob("*.trace.json.gz"))
    trace_files = [p for p in all_files if safe_mtime(p) >= task_started_unix]
    result["trace_dir"] = str(trace_dir)
    result["trace_files"] = [str(p) for p in trace_files]
    if trace_files:
        main_trace = max(trace_files, key=_trace_size_bytes)
        result["main_trace_path"] = str(main_trace)
        log.info(
            "profile_executor: multi-node main trace selected: %s (%d candidate traces this round)",
            main_trace.name,
            len(trace_files),
        )
    elif all_files:
        log.warning(
            "profile_executor: multi-node trace dir %s has %d historical trace(s) but none with mtime >= %.0f (this round's start); sglang may have skipped /start_profile or the trace flush is lagging",
            trace_dir,
            len(all_files),
            task_started_unix,
        )
    else:
        log.warning(
            "profile_executor: multi-node trace dir %s exists but no .trace.json.gz files found yet (server pods may still be flushing)",
            trace_dir,
        )


def _first_trace_dir_with(workspace: Path, finder: Any) -> tuple[Path | None, list[Path]]:
    for trace_dir in _candidate_trace_dirs(workspace):
        if not trace_dir.is_dir():
            continue
        files = finder(trace_dir)
        if files:
            return trace_dir, files
    return None, []


def _record_trace_validation(result: dict[str, Any], trace_dir: Path, framework: str, preflight: Any) -> None:
    health = _validate_trace_structure(trace_dir, framework)
    result["trace_health"] = health
    certificate: dict[str, Any] = {}
    probe_error = ""
    try:
        certificate = _certify_trace_dir(trace_dir, framework)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        probe_error = f"{type(exc).__name__}: {exc}"
        log.debug("profile_executor: trace self-certification failed: %s", probe_error)
    result["trace_validate"] = _build_trace_validate(
        health,
        _TraceProbe(trace_dir, framework, certificate=certificate, error=probe_error),
        preflight=preflight,
    )
    result["trace_validate_path"] = _write_trace_certificate(trace_dir, result["trace_validate"])


def _record_missing_traces(result: dict[str, Any], workspace: Path, workspace_str: str) -> None:
    candidates = _candidate_trace_dirs(workspace)
    result["trace_dir"] = None
    result["trace_files"] = []
    result["status"] = "failed"
    result["error_class"] = "no_trace_files"
    probed = ", ".join(str(p) for p in candidates)
    result["error"] = f"no .trace.json.gz or capture sidecar under {workspace_str} (probed: {probed})"
    existing_empty = [d for d in candidates if d.is_dir()]
    if existing_empty:
        log.warning(
            "profile_executor: trace dirs exist but no .trace.json.gz or capture sidecar (bs_*_rank*.json.gz) files in %s",
            ", ".join(str(p) for p in existing_empty),
        )
    else:
        log.warning(
            "profile_executor: workspace=%s has no trace dir (checked: %s)",
            workspace_str,
            probed,
        )


# Attaches the single-node trace dir this run produced, and certifies it.
def _record_workspace_traces(result: dict[str, Any], workspace_str: str, framework: str, preflight: Any) -> None:
    workspace = Path(workspace_str)
    trace_dir, trace_files = _first_trace_dir_with(workspace, _trace_files_for_dir)
    capture_only = False
    if trace_dir is None:
        trace_dir, trace_files = _first_trace_dir_with(workspace, _capture_sidecar_traces_for_dir)
        capture_only = trace_dir is not None
    if trace_dir is None:
        _record_missing_traces(result, workspace, workspace_str)
        return
    result["trace_dir"] = str(trace_dir)
    result["trace_files"] = [str(p) for p in trace_files]
    main_trace = trace_dir
    if capture_only:
        result["profile_trace_selection_reason"] = "capture_only_fallback"
        log.info(
            "profile_executor: no *.trace.json.gz; falling back to %d SGLang capture sidecar(s) in %s (#575)",
            len(trace_files),
            trace_dir,
        )
    else:
        main_trace = _preferred_main_trace_path(trace_dir, trace_files)
        result["profile_trace_selection_reason"] = (
            "merged_trace_preferred" if main_trace.name.startswith("merged-") else "trace_dir_preferred"
        )
    result["main_trace_path"] = str(main_trace)
    try:
        _record_trace_validation(result, trace_dir, framework, preflight)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        log.debug("profile_executor: trace structure validator failed: %s", exc)


PROFILE_DEFAULT_TIMEOUT_SEC = 14400  # 4 h wall cap


def _trace_size_bytes(path: Path) -> int:
    """Size in bytes, or 0 when it cannot be read."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _is_split_chunk(path: Path, root: Path) -> bool:
    """True when ``path`` is steady-state splitter output below ``root``."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    return any(part == "trace_split" for part in relative.parts)


def _trace_files_for_dir(trace_dir: Path) -> list[Path]:
    """Return annotated ``*.trace.json.gz`` files under ``trace_dir``."""
    candidates = [
        p
        for p in trace_dir.rglob("*.trace.json.gz")
        if not is_capture_fragment(p, trace_dir) and not _is_split_chunk(p, trace_dir)
    ]
    return sorted(candidates, key=lambda p: (-_trace_size_bytes(p), str(p)))


def _capture_sidecar_traces_for_dir(trace_dir: Path) -> list[Path]:
    """Return CUDA-graph capture sidecars under ``trace_dir`` (fallback only)."""
    return sorted(p for p in trace_dir.rglob("*.json.gz") if is_capture_fragment(p, trace_dir))


def _preferred_main_trace_path(trace_dir: Path, trace_files: list[Path]) -> Path:
    """Trace path to pass downstream to TraceLens: the first ``merged-*`` trace, else ``trace_dir``."""
    merged = sorted(p for p in trace_files if p.name.startswith("merged-"))
    return merged[0] if merged else trace_dir


def _candidate_trace_dirs(workspace: Path) -> list[Path]:
    """Trace directories to probe for a Magpie profile workspace."""
    return [
        workspace / "torch_trace",
        workspace / "capture_traces",
        workspace.parent / "capture_traces",
    ]


def _default_profile_config() -> Path:
    """Resolve the shipped profile YAML for ``$FRAMEWORK``, or for the default framework when it is unset."""
    fw = os.environ.get("FRAMEWORK") or framework_registry.DEFAULT_FRAMEWORK
    return asset_root() / "assets" / "configs" / framework_registry.shipped_config_name("profile", fw)


class ProfileExecutor(BenchmarkRunExecutor):
    """Benchmark round with the torch profiler on; extracts and certifies the trace_dir."""

    runs_accuracy_eval = False
    default_timeout_sec = PROFILE_DEFAULT_TIMEOUT_SEC

    def __init__(
        self,
        *,
        magpie_python: str | None = None,
        default_config_path: Path | str | None = None,
        session_dir: Path | str | None = None,
    ):
        """Initialize the profile executor with profile-specific defaults."""
        super().__init__(
            magpie_python=magpie_python,
            default_config_path=default_config_path,
            session_dir=session_dir,
        )
        # Set by ``_after_materialize_config`` once the probe is armed; read after the run to merge per-rank reports.
        self._host_probe_dir: str = ""
        # Non-empty only when arming failed, and then says why (an empty probe dir cannot tell never-asked from failed).
        self._host_probe_status: str = ""
        # Set by ``_after_materialize_config`` from the config the run will actually execute, so the post-hoc trace
        # checks ship alongside the pre-run statement of whether their subject could have been produced.
        self._instrumentation_preflight: dict[str, Any] | None = None

    def _resolve_default_config(self) -> Path:
        """Use the session workload for native AgentX diagnostics."""
        from hyperloom.common.benchmark_config import benchmark_config_path

        return benchmark_config_path() or _default_profile_config()

    def _inject_host_probe(self, config_path: Path, output_dir: Path) -> str:
        """Put the profile start-up shim on the materialized config's ``PYTHONPATH`` and arm the host probe.

        The shim always goes in, since it keeps GPU events in the traces of spawned engines; the
        probe's environment only when the probe is enabled. Returns the probe's report directory,
        or ``""`` when the probe is not armed.
        """
        from . import _framework_rewrite_evidence as _evidence
        from hyperloom.inference_optimizer.runtime_environment import prepend_launch_path

        asset_dir = _evidence.probe_asset_dir()
        if not asset_dir.is_dir():
            log.warning(
                "profile_executor: start-up shim assets missing at %s; host-side rewrite evidence disabled",
                asset_dir,
            )
            return ""
        probe_env = self._host_probe_env(output_dir) if _evidence.probe_enabled() else {}
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            log.warning("profile_executor: cannot read %s to arm the start-up shim: %s", config_path, exc)
            return ""
        bench = cfg.get("benchmark") if isinstance(cfg, dict) else None
        if not isinstance(bench, dict):
            return ""
        envs = bench.setdefault("envs", {})
        if not isinstance(envs, dict):
            return ""
        # First on the path even when already present: Python imports only the first sitecustomize, and the
        # shim's chains to the next one.
        prepend_launch_path(envs, "PYTHONPATH", str(asset_dir))
        for key, value in probe_env.items():
            envs[key] = value
        try:
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        except OSError as exc:
            log.warning("profile_executor: cannot write %s after arming the start-up shim: %s", config_path, exc)
            return ""
        if not probe_env:
            return ""
        log.info(
            "profile_executor: host-side rewrite evidence probe armed (deep=%s), reports -> %s",
            bool(probe_env.get("HYPERLOOM_HOST_PROBE_DEEP")),
            probe_env["HYPERLOOM_HOST_PROBE_DIR"],
        )
        return probe_env["HYPERLOOM_HOST_PROBE_DIR"]

    def _host_probe_env(self, output_dir: Path) -> dict[str, str]:
        """Return the environment that arms the host probe, or ``{}`` when its report directory cannot be made."""
        from . import _framework_rewrite_evidence as _evidence

        probe_dir = output_dir / _evidence.PROBE_SUBDIR
        try:
            probe_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.warning("profile_executor: cannot create host-probe dir %s: %s", probe_dir, exc)
            return {}

        from hyperloom.inference_optimizer.framework_paths import resolve_kernel_search_roots

        return _evidence.build_probe_env(
            probe_dir=probe_dir,
            source_roots=list(resolve_kernel_search_roots()),
            deep=_evidence.deep_probe_enabled(),
        )

    def _after_materialize_config(
        self,
        config_path: Path,
        output_dir: Path,
    ) -> dict[str, Any] | None:
        """Arm the host probe, then patch the InferenceX checkout Magpie will execute."""
        from .magpie_agentx import enable_profile, native_config

        if native_config(config_path):
            enable_profile(config_path)
            return None
        try:
            self._host_probe_dir = self._inject_host_probe(config_path, output_dir)
            self._host_probe_status = ""
        except (OSError, RuntimeError, ValueError, yaml.YAMLError) as exc:
            log.warning("profile_executor: host-probe injection failed: %s", exc, exc_info=True)
            self._host_probe_dir = ""
            self._host_probe_status = f"probe_injection_failed: {exc}"
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError, yaml.YAMLError) as exc:
            return {
                "status": "failed",
                "error_class": "profile_config_unreadable",
                "error": f"cannot read materialized profile config {config_path}: {exc}",
            }
        bench = cfg.get("benchmark") if isinstance(cfg, dict) else {}
        patchers: dict[str, Any] = {}

        def _note_instrumentation() -> None:
            """Re-state the instrumentation report. Called before every exit below, so no path leaves it unset."""
            self._instrumentation_preflight = _instrumentation_preflight_row(bench, patchers)
            detail = self._instrumentation_preflight["detail"]
            if self._instrumentation_preflight["status"] != "failed":
                return
            log.warning(
                "profile_executor: instrumentation preflight — degraded_reason=%r failed_patchers=%s; "
                "annotation-dependent trace checks (%s, %s) cannot pass for this run",
                detail.get("degraded_reason"),
                detail.get("failed_patchers"),
                CHECK_STEP_ANNOTATIONS,
                CHECK_SGLANG_SHAPE_PROFILER,
            )

        _note_instrumentation()
        framework = ""
        if isinstance(bench, dict):
            framework = str(bench.get("framework") or "").strip().lower()
        from hyperloom.inference_optimizer import framework_registry

        if framework_registry.is_scriptable(framework):
            return None
        inferencex_path = ""
        if isinstance(bench, dict):
            inferencex_path = str(bench.get("inferencex_path") or "").strip()
        if not inferencex_path:
            inferencex_path = os.environ.get("INFERENCEX_PATH", "").strip()
        if not inferencex_path:
            log.warning(
                "profile_executor: no benchmark.inferencex_path / "
                "INFERENCEX_PATH configured; skipping InferenceX profile "
                "patch validation"
            )
            patchers["inferencex"] = None
            _note_instrumentation()
            return None

        ix_root = Path(inferencex_path)
        lib_ok = ensure_benchmark_lib_patched(ix_root)
        patchers["benchmark_lib"] = lib_ok
        patchers["benchmark_lib_eval_dest"] = ensure_benchmark_lib_eval_dest_patched(ix_root)
        serving_ok = ensure_benchmark_serving_patched(ix_root)
        patchers["benchmark_serving"] = serving_ok
        lib_path = ix_root / "benchmarks" / "benchmark_lib.sh"
        # path as a forwarding shim, which never carries the sentinel however well the patch landed.
        # Scoped to ``ix_root`` like ``lib_path`` above: this gate speaks for the tree Magpie runs.
        serving_path = benchmark_serving_path_in(ix_root)

        def _contains(path: Path, needle: str) -> bool:
            """Check whether ``needle`` appears in ``path``'s text."""
            try:
                return needle in path.read_text(encoding="utf-8")
            except OSError:
                return False

        lib_valid = _contains(lib_path, "${NUM_PROMPTS:-$max_concurrency}")
        serving_valid = _contains(serving_path, "PROFILE_EXTRA_BODY")
        # The sentinels are a separate fact from the patcher's return: a patcher can report success against a
        # checkout whose anchors have since moved, and only reading the file back tells them apart.
        patchers["benchmark_lib_sentinel"] = lib_valid
        patchers["benchmark_serving_sentinel"] = serving_valid
        patchers["inferencex_path"] = str(ix_root)
        _note_instrumentation()
        if not (lib_ok and serving_ok and lib_valid and serving_valid):
            return {
                "status": "failed",
                "error_class": "profile_inferencex_patch_failed",
                "error": (
                    "profile requires InferenceX to honour NUM_PROMPTS and "
                    "PROFILE_EXTRA_BODY, but the checkout Magpie will use is "
                    f"not patched: inferencex_path={ix_root}, "
                    f"benchmark_lib_ok={lib_ok}/{lib_valid}, "
                    f"benchmark_serving_ok={serving_ok}/{serving_valid}"
                ),
                "inferencex_path": str(ix_root),
                "benchmark_lib": str(lib_path),
                "benchmark_serving": str(serving_path),
            }
        return None

    def drain_instrumentation_report(self) -> dict[str, Any] | None:
        """Take the instrumentation report this executor produced, clearing it.

        Draining rather than reading: the caller runs several profile attempts against this module-level singleton,
        and an attempt that dies before materializing a config must report nothing rather than inherit the
        previous attempt's report. Exposed because the report has to survive the paths where no result dict does
        -- an executor that raises still patched, or still failed to.
        """
        report, self._instrumentation_preflight = self._instrumentation_preflight, None
        return report

    def _collect_rewrite_evidence(self, result: dict[str, Any]) -> None:
        """Merge the per-rank host-probe reports onto ``result``."""
        probe_dir = str(self._host_probe_dir or "").strip()
        if not probe_dir:
            result["framework_rewrite_evidence_status"] = self._host_probe_status or "probe_not_armed"
            return
        from . import _framework_rewrite_evidence as _evidence

        try:
            out_path = Path(probe_dir).parent / _evidence.EVIDENCE_FILENAME
            document = _evidence.aggregate_probe_dir(probe_dir, out_path)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            log.warning(
                "profile_executor: rewrite-evidence aggregation failed for %s: %s",
                probe_dir,
                exc,
                exc_info=True,
            )
            result["framework_rewrite_evidence_status"] = f"aggregation_failed: {exc}"
            return
        candidates = document.get("candidates") or []
        if not candidates:
            log.info(
                "profile_executor: host probe produced no rewrite candidates (ranks_merged=%s); see %s for why",
                document.get("ranks_merged"),
                out_path,
            )
            result["framework_rewrite_evidence_status"] = "no_candidates"
            return
        result["framework_rewrite_evidence"] = str(out_path)
        result["framework_rewrite_candidate_count"] = len(candidates)
        result["framework_rewrite_evidence_status"] = "ok"
        log.info(
            "profile_executor: %d host-side rewrite candidate(s) from %s rank(s) -> %s",
            len(candidates),
            document.get("ranks_merged"),
            out_path,
        )

    async def __call__(self, ctx) -> dict[str, Any]:
        """Run the profiling action for the given context."""
        params = ctx.task.params or {}
        _merge_profile_params(params)
        extra = getattr(ctx, "extra", None) or {}
        from hyperloom.common.benchmark_config import configured_benchmark, is_magpie_agentx

        if is_magpie_agentx(configured_benchmark()):
            return await self._run_native_profile(ctx)
        if not (params.get("output_dir") or extra.get("workspace")):
            output_dir = self._resolve_workspace(ctx, "profile")
            output_dir.mkdir(parents=True, exist_ok=True)
            extra["workspace"] = str(output_dir)
        import time as _time

        task_started_unix = _time.time()
        from ._multi_node_env import log_mn_banner

        round_trace_root = _mn_round_trace_root()
        log_mn_banner("profile_executor", log, trace_dir=round_trace_root)
        if round_trace_root:
            refusal = await _restart_round_server(params, round_trace_root)
            if refusal is not None:
                return refusal
            extra["mn_round_restarted"] = True
        async with _infera_profile_window(round_trace_root):
            result = await super().__call__(ctx)
        self._collect_rewrite_evidence(result)
        workspace_str = result.get("workspace")
        if round_trace_root:
            _record_mn_round_traces(result, round_trace_root, task_started_unix)
        elif workspace_str:
            _record_workspace_traces(
                result,
                workspace_str,
                str(getattr(ctx, "framework", "") or extra.get("framework") or ""),
                self._instrumentation_preflight,
            )
        if result.get("main_trace_path"):
            result["trace_input_ready"] = True
        return result

    _measurement_succeeded = staticmethod(round_succeeded)

    async def _run_native_profile(self, ctx) -> dict[str, Any]:
        """Collect Magpie-owned captures and certify each time window separately."""
        import asyncio

        result = await super().__call__(ctx)
        if result.get("status") != "succeeded":
            return result
        capture = result.get("agentx_metrics", {}).get("profile_capture", {})
        if capture.get("status") != "complete":
            return {
                **result,
                "status": "failed",
                "error_class": "profile_capture_failed",
                "trace_capture": capture,
                "trace_input_ready": False,
            }
        result["trace_capture"] = capture
        result["trace_capture_status"] = capture["status"]
        result["trace_input_ready"] = False
        result["profile_captures"] = []
        for window in capture.get("profiles", [capture]):
            mapped = await asyncio.to_thread(_map_native_capture, window, result)
            result["profile_captures"].append(mapped)
        primary = next((window for window in result["profile_captures"] if window.get("trace_input_ready")), None)
        if primary is None:
            result.update(status="failed", error_class="primary_rank_trace_missing")
        else:
            result.update(primary)
        return result


profile_executor = ProfileExecutor()


__all__ = [
    "PROFILE_DEFAULT_TIMEOUT_SEC",
    "ProfileExecutor",
    "profile_executor",
]


def _map_native_capture(capture: dict[str, Any], measurement: dict[str, Any]) -> dict[str, Any]:
    """Select and check one capture, without combining ranks from different windows."""
    files = [Path(name) for name in capture.get("trace_files", [])]
    result: dict[str, Any] = {"capture_id": capture["capture_id"], "trace_input_ready": False}
    if not files or not all(path.is_file() for path in files):
        return result
    trace_dir = files[0].parent
    main = select_primary_trace(files, file_size=_trace_size_bytes, tensor_parallel_size=measurement.get("tp"))
    result.update(trace_dir=str(trace_dir), trace_files=[str(path) for path in files])
    if main is None:
        return result
    result.update(main_trace_path=str(main), primary_rank=_trace_rank(main), trace_input_ready=True)
    framework = measurement["framework"]
    health = _validate_trace_structure(trace_dir, framework)
    certificate = _certify_trace_dir(trace_dir, framework)
    result["trace_health"] = health
    result["trace_validate"] = _build_trace_validate(health, _TraceProbe(trace_dir, framework, certificate=certificate))
    result["trace_validate_path"] = _write_trace_certificate(trace_dir, result["trace_validate"])
    return result
