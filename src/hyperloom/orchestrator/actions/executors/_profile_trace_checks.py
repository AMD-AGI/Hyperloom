# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a finished profile produced: the structural checks over its trace dir, and the certificate they build."""

from __future__ import annotations

import gzip
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NamedTuple

from hyperloom.common.io import atomic_write_json
from hyperloom.common.timeutil import now_iso

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

# Where a framework writes its CUDA-graph capture trace, newest convention first.
_CAPTURE_DIR_NAMES = ("capture_traces", "graph_capture_profile", "graph_capture")


def _trace_contains(path: Path, substring: str, max_bytes: int | None = None) -> bool:
    """Stream-decompress ``path`` for ``substring``, reading at most ``max_bytes`` (default :data:`_TRACE_CONFIRM_BYTES`)."""
    if not substring:
        return False
    if max_bytes is None:
        try:
            max_bytes = int(
                os.environ.get(
                    "INFERENCE_OPTIMIZER_TRACE_CONFIRM_BYTES",
                    _TRACE_CONFIRM_BYTES,
                )
            )
        except (TypeError, ValueError):
            max_bytes = _TRACE_CONFIRM_BYTES
    read = 0
    carry = ""
    chunk_size = 4_000_000
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            while read < max_bytes:
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


def _count_substring_occurrences(text: str, substring: str) -> int:
    """Count non-overlapping ``substring`` occurrences as a cheap lower-bound event count (avoids full JSON parsing)."""
    if not substring:
        return 0
    return text.count(substring)


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
        "chunk_level": [],
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


class _TraceCheck(NamedTuple):
    """One check's structured verdict, with the operator-facing advisory it raises (empty when it found nothing)."""

    row: dict[str, Any]
    issue: str = ""


@dataclass(frozen=True)
class _TraceSample:
    """What the post-profile checks measure: the files a profile produced, and who produced them."""

    framework: str
    #: Scriptable image frameworks (xDiT diffusion) write a plain torch-profiler trace, so the LLM/serving-specific
    #: checks would only emit spurious warnings against one.
    scriptable: bool
    capture_dir: Path | None
    capture_files: list[Path]
    main_traces: list[Path]
    main_text: str | None
    #: True only when ``trace_split/`` exists *and* this framework splits per step; otherwise the split checks skip.
    split_scanned: bool
    split_files: list[Path]


def _capture_dir(trace_dir: Path) -> Path | None:
    """The graph-capture subdirectory this profile wrote, or None when graph capture never fired."""
    return next((trace_dir / name for name in _CAPTURE_DIR_NAMES if (trace_dir / name).is_dir()), None)


def _sample_trace_dir(trace_dir: Path, framework: str) -> _TraceSample:
    """Read the files and leading text that the structure checks below measure."""
    from hyperloom.inference_optimizer import framework_registry as _fw_reg

    scriptable = _fw_reg.is_scriptable(framework)
    capture = _capture_dir(trace_dir)
    main_traces = sorted(
        (p for p in trace_dir.glob("*.trace.json.gz") if p.is_file()),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    split = trace_dir / "trace_split"
    split_scanned = split.is_dir() and not scriptable
    return _TraceSample(
        framework=framework,
        scriptable=scriptable,
        capture_dir=capture,
        capture_files=sorted(p for p in capture.iterdir() if p.is_file()) if capture is not None else [],
        main_traces=main_traces,
        main_text=_sample_trace_text(main_traces[0]) if main_traces else None,
        split_scanned=split_scanned,
        split_files=sorted(p for p in split.iterdir() if p.is_file()) if split_scanned else [],
    )


def _check_capture_traces_present(sample: _TraceSample) -> _TraceCheck:
    """Verdict on whether the graph-capture subdirectory exists and holds files."""
    if sample.scriptable:
        return _TraceCheck(
            _check_row(
                CHECK_CAPTURE_TRACES_PRESENT,
                status="skipped",
                skip_reason="scriptable framework writes a plain torch-profiler trace",
            )
        )
    if sample.capture_dir is None:
        return _TraceCheck(
            _check_row(CHECK_CAPTURE_TRACES_PRESENT, status="failed", capture_dir_present=False, file_count=0),
            "[1] graph-capture subdirectory missing — graph capture "
            "didn't fire. Verify EXTRA_VLLM_ARGS / EXTRA_SGLANG_ARGS "
            "include the TraceLens flag and the server-side patch landed.",
        )
    row = _check_row(
        CHECK_CAPTURE_TRACES_PRESENT,
        status="passed" if sample.capture_files else "failed",
        capture_dir_present=True,
        capture_dir=str(sample.capture_dir),
        file_count=len(sample.capture_files),
    )
    if sample.capture_files:
        return _TraceCheck(row)
    return _TraceCheck(
        row,
        f"[1] {sample.capture_dir.name}/ exists but is empty — graph capture path fired but produced no files.",
    )


def _no_cpu_op_verdict(sampled_file: str, input_dims_count: int) -> _TraceCheck:
    """Verdict for a capture whose sample holds no literal ``cpu_op`` event, so there is no fraction to judge."""
    return _TraceCheck(
        # On ROCm/SGLang this is an event-naming difference rather than a capture failure.
        _check_row(
            CHECK_CAPTURE_INPUT_DIMS,
            status="skipped",
            skip_reason="no literal cpu_op events to measure the Input Dims fraction against",
            sampled_file=sampled_file,
            cpu_op_count=0,
            input_dims_count=input_dims_count,
            input_dims_fraction=None,
            floor=_INPUT_DIMS_FRACTION_FLOOR,
        ),
        f"[2] capture file {sampled_file} has no literal "
        f"'cpu_op' events in the first "
        f"{_TRACE_INSPECT_BYTES // 1_000_000} MB — on ROCm/SGLang "
        "this is often just an event-naming difference (kernels "
        "logged under 'sglang_profiler::*'); cross-check Check "
        "[5] (kernel_shape_profiler) and the server log before "
        "treating it as a capture regression.",
    )


def _check_capture_input_dims(sample: _TraceSample) -> _TraceCheck:
    """Verdict on how many of the heaviest capture file's ``cpu_op`` events carry ``Input Dims``."""
    if not sample.capture_files:
        return _TraceCheck(
            _check_row(CHECK_CAPTURE_INPUT_DIMS, status="skipped", skip_reason="no capture file to sample")
        )
    target = max(sample.capture_files, key=lambda p: p.stat().st_size)
    text = _sample_trace_text(target)
    if text is None:
        return _TraceCheck(
            _check_row(
                CHECK_CAPTURE_INPUT_DIMS,
                status="skipped",
                skip_reason="capture file could not be sampled",
                sampled_file=target.name,
            )
        )
    # ``cpu_op`` is the event's category; its ``name`` is the operator (``aten::mm``). Matching on the
    # name key found nothing in any real trace, so this check reported "no cpu_op events" on captures
    # that were fully instrumented, and the advisory below rationalised the miss as an SGLang naming
    # quirk. Verified against a real sglang capture: 0 hits for the name form, 2915 for this one.
    cpu_op_count = _count_substring_occurrences(text, '"cat": "cpu_op"')
    input_dims_count = _count_substring_occurrences(text, '"Input Dims"')
    if cpu_op_count == 0:
        return _no_cpu_op_verdict(target.name, input_dims_count)
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
        return _TraceCheck(row)
    pct = 100.0 * input_dims_count / cpu_op_count
    return _TraceCheck(
        row,
        f"[2] capture file {target.name}: only {pct:.1f}% of "
        f"cpu_op events carry 'Input Dims' (expected ≥ "
        f"{int(_INPUT_DIMS_FRACTION_FLOOR * 100)}%). Shape-"
        "discovery instrumentation may not be fully active — "
        "verify TraceLens server patch and capture flag.",
    )


def _check_step_annotations(sample: _TraceSample) -> _TraceCheck:
    """Verdict on whether the main trace carries InferenceX's per-step ``execute_*`` / ``user_annotation`` events."""
    if not sample.main_traces:
        return _TraceCheck(
            _check_row(
                CHECK_STEP_ANNOTATIONS,
                status="skipped",
                skip_reason="no *.trace.json.gz in the trace dir",
            )
        )
    main = sample.main_traces[0]
    if sample.main_text is None:
        return _TraceCheck(
            _check_row(
                CHECK_STEP_ANNOTATIONS,
                status="skipped",
                skip_reason="main trace could not be sampled",
                sampled_file=main.name,
            )
        )
    user_ann_count = _count_substring_occurrences(sample.main_text, _USER_ANNOTATION_MARKER)
    execute_count = _count_substring_occurrences(sample.main_text, '"execute_')
    # ``execute_*`` labels are the real health signal; ``user_annotation`` presence is profiler-version-dependent.
    confirmed_absent = (
        not sample.scriptable
        and execute_count == 0
        and user_ann_count == 0
        and not (_trace_contains(main, '"execute_') or _trace_contains(main, _USER_ANNOTATION_MARKER))
    )
    row = _check_row(
        CHECK_STEP_ANNOTATIONS,
        status="failed" if confirmed_absent else "passed",
        sampled_file=main.name,
        execute_annotation_count=execute_count,
        user_annotation_count=user_ann_count,
        confirmed_absent=confirmed_absent,
    )
    if not confirmed_absent:
        return _TraceCheck(row)
    return _TraceCheck(
        row,
        f"[3] main trace {main.name} has no "
        "execute_* / user_annotation events — InferenceX "
        "per-step annotations didn't fire. Verify "
        "detailed_annotations reached the framework "
        "(PROFILE_EXTRA_BODY consumed; see #210).",
    )


def _no_split_reason(sample: _TraceSample) -> str:
    """Why ``trace_split/`` was not examined."""
    return "scriptable framework has no per-step split" if sample.scriptable else "no trace_split/ directory"


def _chunks_without_step_annotations(split_files: list[Path]) -> list[str]:
    """Names of the ``trace_split/`` chunks carrying no ``execute_*`` user annotation."""
    empty: list[str] = []
    for chunk in split_files:
        if not chunk.name.endswith(".json.gz"):
            continue
        text = _sample_trace_text(chunk)
        if text is None:
            continue
        if _count_substring_occurrences(text, '"execute_') == 0:
            empty.append(chunk.name)
    return empty


def _check_split_chunk_annotations(sample: _TraceSample) -> _TraceCheck:
    """Verdict on whether the steady-state splitter's chunks came out with per-step annotations in them."""
    if not sample.split_scanned:
        return _TraceCheck(
            _check_row(
                CHECK_SPLIT_CHUNK_ANNOTATIONS,
                status="skipped",
                skip_reason=_no_split_reason(sample),
            )
        )
    empty_splits = _chunks_without_step_annotations(sample.split_files)
    produced_empty_chunks = bool(sample.split_files and empty_splits)
    row = _check_row(
        CHECK_SPLIT_CHUNK_ANNOTATIONS,
        status="failed" if produced_empty_chunks else "passed",
        split_file_count=len(sample.split_files),
        empty_chunk_count=len(empty_splits),
        empty_chunk_samples=empty_splits[:3],
    )
    if not produced_empty_chunks:
        return _TraceCheck(row)
    return _TraceCheck(
        row,
        f"[4] {len(empty_splits)} trace_split/ file(s) have NO "
        "execute_* user_annotations: "
        f"{', '.join(empty_splits[:3])}"
        + (f" (and {len(empty_splits) - 3} more)" if len(empty_splits) > 3 else "")
        + " — splitter ran but the chunks are empty. Likely the "
        "trace lacks the per-step annotations needed for splitting "
        "(see check [3]).",
    )


def _check_steady_state_split_naming(sample: _TraceSample) -> _TraceCheck:
    """Verdict on whether ``trace_split/`` holds steady-state chunks rather than raw extend/decode stages."""
    if not sample.split_scanned:
        return _TraceCheck(
            _check_row(
                CHECK_STEADY_STATE_SPLIT_NAMING,
                status="skipped",
                skip_reason=_no_split_reason(sample),
            )
        )
    names = [p.name for p in sample.split_files]
    has_extend = any("_extend_" in n or "extend_only_" in n for n in names)
    has_decode = any("_decode_" in n or "decode_only_" in n for n in names)
    has_steady_state = any("steady_state" in n for n in names)
    split_by_stage = (has_extend or has_decode) and not has_steady_state
    row = _check_row(
        CHECK_STEADY_STATE_SPLIT_NAMING,
        status="failed" if split_by_stage else "passed",
        has_extend=has_extend,
        has_decode=has_decode,
        has_steady_state=has_steady_state,
    )
    if not split_by_stage:
        return _TraceCheck(row)
    return _TraceCheck(
        row,
        "[6] trace_split/ has _extend_* / _decode_* files but NO "
        "_steady_state_* — profile_by_stage=True leaked through, "
        "PROFILE_EXTRA_BODY env was not consumed by the framework. "
        "Confirm _inferencex_patcher patched Magpie's bundled "
        "InferenceX (#210; check $MAGPIE_PATH/InferenceX/utils/"
        "bench_serving/benchmark_serving.py).",
    )


def _check_trace_has_ops(sample: _TraceSample) -> _TraceCheck:
    """Verdict on whether the torch-profiler active window recorded any real execution.

    A metadata-only trace (no ``cpu_op`` / ``kernel`` events) is unusable for roofline, so flag it and let the
    caller re-profile rather than cache an empty snapshot.
    """
    if not sample.main_traces:
        return _TraceCheck(
            _check_row(
                CHECK_TRACE_HAS_OPS,
                status="skipped",
                skip_reason="no *.trace.json.gz in the trace dir",
            )
        )
    main = sample.main_traces[0]
    has_ops = _trace_contains(main, '"cat": "cpu_op"') or _trace_contains(main, '"cat": "kernel"')
    row = _check_row(
        CHECK_TRACE_HAS_OPS,
        status="passed" if has_ops else "failed",
        sampled_file=main.name,
        has_ops=has_ops,
    )
    if has_ops:
        return _TraceCheck(row)
    return _TraceCheck(
        row,
        f"[7] main trace {main.name} has NO cpu_op / kernel "
        "events — the torch-profiler active window recorded nothing "
        "(metadata-only trace). On xDiT/diffusion this is the "
        "torch.profiler.schedule repeat=0 discard (the active window is "
        "dropped when the schedule restarts after the last active "
        "step). The trace is unusable for roofline; re-profile needed.",
    )


def _check_sglang_shape_profiler(sample: _TraceSample) -> _TraceCheck:
    """Verdict on whether sglang's kernel-shape markers reached the main trace."""
    if sample.framework.lower() != "sglang":
        return _TraceCheck(
            _check_row(
                CHECK_SGLANG_SHAPE_PROFILER,
                status="skipped",
                skip_reason=f"framework is {sample.framework or 'unset'}, not sglang",
            )
        )
    if sample.main_text is None:
        return _TraceCheck(
            _check_row(
                CHECK_SGLANG_SHAPE_PROFILER,
                status="skipped",
                skip_reason="main trace could not be sampled",
            )
        )
    # Shape markers left by either mechanism: the sglang_profiler:: op
    # namespace, or the kernel_shape_profiler frame (when with_stack is on).
    shape_present = any(m in sample.main_text for m in ("sglang_profiler::", "kernel_shape_profiler"))
    row = _check_row(
        CHECK_SGLANG_SHAPE_PROFILER,
        status="passed" if shape_present else "failed",
        sampled_file=sample.main_traces[0].name,
        sampled_bytes=_TRACE_INSPECT_BYTES,
    )
    if shape_present:
        return _TraceCheck(row)
    return _TraceCheck(
        row,
        f"[5] sglang main trace ({sample.main_traces[0].name}, sampled "
        f"first {_TRACE_INSPECT_BYTES // 1_000_000} MB) lacks "
        "kernel-shape events — shape discovery didn't reach the live "
        "SGLang. For SGLang < 0.5.18 verify the _server_patcher "
        "git-apply succeeded; for >= 0.5.18 verify the kernel_shape_tool "
        "is on the server PYTHONPATH and TRACELENS_SHAPE_DISCOVERY=1 "
        "(check log warnings).",
    )


def _check_status(checks: list[dict[str, Any]], check_id: str) -> str:
    """The status recorded for ``check_id``."""
    return next((row["status"] for row in checks if row["check_id"] == check_id), "skipped")


def _log_trace_issues(issues: list[str]) -> None:
    """Warn once per finding, then state what the findings together cost downstream."""
    if not issues:
        return
    for issue in issues:
        log.warning("trace structure check: %s", issue)
    log.warning(
        "trace structure check: %d issue(s) detected — TraceLens "
        "downstream analysis may be degraded. See per-issue messages "
        "above for the actionable check.",
        len(issues),
    )


def _validate_trace_structure(
    trace_dir: Path,
    framework: str,
) -> dict[str, Any]:
    """Post-profile sanity check on the produced trace structure."""
    # A roofline-composite ctx carries no framework, and an empty name resolves to the serving default — which fires
    # every serving-only check below against a scriptable trace. $FRAMEWORK is the session-wide lock, so fall back to
    # it.
    sample = _sample_trace_dir(trace_dir, str(framework or os.environ.get("FRAMEWORK", "") or ""))
    verdicts = [
        _check_capture_traces_present(sample),
        _check_capture_input_dims(sample),
        _check_step_annotations(sample),
        _check_split_chunk_annotations(sample),
        _check_steady_state_split_naming(sample),
        _check_trace_has_ops(sample),
        _check_sglang_shape_profiler(sample),
    ]
    checks = [verdict.row for verdict in verdicts]
    issues = [verdict.issue for verdict in verdicts if verdict.issue]
    _log_trace_issues(issues)
    return {
        "issues": issues,
        "per_kernel_attribution_degraded": _check_status(checks, CHECK_STEP_ANNOTATIONS) == "failed",
        "capture_traces_present": _check_status(checks, CHECK_CAPTURE_TRACES_PRESENT) == "passed",
        "zero_ops": _check_status(checks, CHECK_TRACE_HAS_OPS) == "failed",
        "checks": checks,
    }
