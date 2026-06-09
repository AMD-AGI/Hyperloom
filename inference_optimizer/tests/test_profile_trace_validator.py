# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``profile._validate_trace_structure`` (#210 / Deval's check_torch_trace.py)."""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path

from inference_optimizer.orchestrator.action_executors.profile import (
    _validate_trace_structure,
)


def _write_minimal_sglang_trace(
    path: Path,
    *,
    with_kernel_shape_profiler: bool,
    with_user_annotation: bool = True,
    with_execute_star: bool = True,
) -> None:
    """Write a tiny gzipped JSON trace blob; each flag toggles one validator signal."""
    events: list[dict] = [
        {"name": "cpu_op", "ph": "X", "ts": 0, "dur": 1, "args": {"Input Dims": [[1, 2, 3]]}},
    ]
    if with_user_annotation:
        events.append({
            "name": "user_annotation",
            "ph": "i",
            "args": {
                "label": (
                    "execute_16384_context_16(sq16384sk16384)_generation_0(sq0sk0)"
                    if with_execute_star
                    else "some_other_annotation_label"
                ),
            },
        })
    if with_kernel_shape_profiler:
        events.append({
            "name": "python_function",
            "ph": "i",
            "args": {"frame": "kernel_shape_profiler"},
        })
    payload = {"schemaVersion": 1, "traceEvents": events}
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _write_capture_file(
    path: Path, *, cpu_op_count: int, with_input_dims_fraction: float = 1.0,
) -> None:
    """Synthesize a capture file with a given fraction of ``Input Dims`` events (check [2])."""
    events: list[dict] = []
    threshold = int(round(cpu_op_count * with_input_dims_fraction))
    for i in range(cpu_op_count):
        ev = {"name": "cpu_op", "ph": "X", "ts": i, "dur": 1, "args": {}}
        if i < threshold:
            ev["args"]["Input Dims"] = [[1, 2, 3]]
        events.append(ev)
    payload = {"schemaVersion": 1, "traceEvents": events}
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _write_split_file(path: Path, *, with_execute_star: bool = True) -> None:
    """Synthesize a ``trace_split`` file with/without ``execute_*`` annotations (check [4])."""
    events: list[dict] = [
        {"name": "cpu_op", "ph": "X", "ts": 0, "dur": 1},
    ]
    if with_execute_star:
        events.append({
            "name": "user_annotation",
            "ph": "i",
            "args": {"label": "execute_32_context_1(sq32sk1025)_generation_0(sq0sk0)"},
        })
    payload = {"schemaVersion": 1, "traceEvents": events}
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _build_healthy_layout(tmp_path: Path) -> Path:
    """Build a complete reference layout that satisfies all 6 checks."""
    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()
    capture = trace_dir / "capture_traces"
    capture.mkdir()
    _write_capture_file(
        capture / "bs_104_rank0.json.gz",
        cpu_op_count=200, with_input_dims_fraction=0.99,
    )
    split = trace_dir / "trace_split"
    split.mkdir()
    _write_split_file(
        split / "decode_only_steady_state_chunk0.json.gz",
        with_execute_star=True,
    )
    _write_split_file(
        split / "mixed_steady_state_chunk0.json.gz",
        with_execute_star=True,
    )
    main_trace = trace_dir / "1776409856.2485812-TP-0.trace.json.gz"
    _write_minimal_sglang_trace(
        main_trace,
        with_kernel_shape_profiler=True,
        with_user_annotation=True,
        with_execute_star=True,
    )
    return trace_dir


def test_validator_no_warnings_on_healthy_layout(tmp_path, caplog):
    """The reference healthy layout (#210) must not trip any of the 6 checks."""
    trace_dir = _build_healthy_layout(tmp_path)
    caplog.set_level(logging.WARNING)
    _validate_trace_structure(trace_dir, "sglang")
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == [], (
        f"healthy layout produced unexpected warnings: "
        f"{[r.getMessage() for r in warnings]}"
    )


# Check [6] (Hyperloom-specific #210 smoking-gun)
def test_validator_warns_on_extend_decode_files_without_steady_state(
    tmp_path, caplog,
):
    """#210 symptom: ``_extend_*`` / ``_decode_*`` split files signal
    ``profile_by_stage=True`` leaked through."""
    trace_dir = _build_healthy_layout(tmp_path)
    split = trace_dir / "trace_split"
    for p in list(split.iterdir()):
        p.unlink()
    _write_split_file(
        split / "_extend_step_0_TP-0.trace.json.gz", with_execute_star=True,
    )
    _write_split_file(
        split / "_decode_step_0_TP-0.trace.json.gz", with_execute_star=True,
    )

    caplog.set_level(logging.WARNING)
    _validate_trace_structure(trace_dir, "sglang")
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("profile_by_stage=True leaked through" in m for m in msgs), msgs
    assert any("$MAGPIE_DIR/InferenceX" in m for m in msgs), (
        "warning message should point operators at the #210 fix surface"
    )


# Check [1] capture_traces/ presence
def test_validator_warns_when_capture_traces_missing(tmp_path, caplog):
    """A missing ``capture_traces/`` means graph capture didn't fire."""
    trace_dir = _build_healthy_layout(tmp_path)
    capture = trace_dir / "capture_traces"
    for p in list(capture.iterdir()):
        p.unlink()
    capture.rmdir()

    caplog.set_level(logging.WARNING)
    _validate_trace_structure(trace_dir, "sglang")
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("capture_traces/" in m and "missing" in m for m in msgs), msgs


# Check [2] (Deval) capture file has cpu_op + Input Dims
def test_validator_warns_when_capture_file_lacks_input_dims(tmp_path, caplog):
    """A capture file with too few ``Input Dims`` events trips the 90% floor."""
    trace_dir = _build_healthy_layout(tmp_path)
    capture = trace_dir / "capture_traces"
    bad = capture / "bs_104_rank0.json.gz"
    bad.unlink()
    _write_capture_file(bad, cpu_op_count=200, with_input_dims_fraction=0.5)

    caplog.set_level(logging.WARNING)
    _validate_trace_structure(trace_dir, "sglang")
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "Input Dims" in m and "Shape-discovery instrumentation" in m for m in msgs
    ), msgs


def test_validator_warns_when_capture_file_has_no_cpu_op(tmp_path, caplog):
    """A capture file with zero ``cpu_op`` events is a distinct failure mode."""
    trace_dir = _build_healthy_layout(tmp_path)
    capture = trace_dir / "capture_traces"
    bad = capture / "bs_104_rank0.json.gz"
    bad.unlink()
    _write_capture_file(bad, cpu_op_count=0)

    caplog.set_level(logging.WARNING)
    _validate_trace_structure(trace_dir, "sglang")
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    # Wording softened: zero literal ``cpu_op`` events on ROCm/SGLang is
    # often just an event-naming difference, not a capture regression.
    assert any("no literal 'cpu_op' events" in m for m in msgs), msgs


# Check [3] (Deval) main trace has user_annotation + execute_*
def test_validator_no_warning_when_execute_star_present_without_user_annotation(
    tmp_path, caplog,
):
    """``execute_*`` present without a ``user_annotation`` wrapper is healthy — Check [3] must not warn."""
    trace_dir = _build_healthy_layout(tmp_path)
    main = next(trace_dir.glob("*.trace.json.gz"))
    main.unlink()
    # execute_* under a non-user_annotation event (ROCm/SGLang 0.5.11 shape).
    payload = {
        "schemaVersion": 1,
        "traceEvents": [
            {"name": "cpu_op", "ph": "X", "ts": 0, "dur": 1},
            {
                "name": "kernel",
                "ph": "X",
                "args": {
                    "label": "execute_16384_context_16(sq16384sk16384)"
                },
            },
            {"name": "python_function", "ph": "i",
             "args": {"frame": "kernel_shape_profiler"}},
        ],
    }
    with gzip.open(main, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)

    caplog.set_level(logging.WARNING)
    _validate_trace_structure(trace_dir, "sglang")
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("[3]" in m for m in msgs), (
        "execute_* present → Check [3] must not warn even without the "
        f"user_annotation wrapper; got: {msgs}"
    )


def test_validator_warns_when_main_trace_lacks_all_annotations(
    tmp_path, caplog,
):
    """Neither ``execute_*`` nor ``user_annotation`` anywhere → Check [3] warns."""
    trace_dir = _build_healthy_layout(tmp_path)
    main = next(trace_dir.glob("*.trace.json.gz"))
    main.unlink()
    _write_minimal_sglang_trace(
        main,
        with_kernel_shape_profiler=True,
        with_user_annotation=False,
        with_execute_star=False,
    )

    caplog.set_level(logging.WARNING)
    _validate_trace_structure(trace_dir, "sglang")
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "[3]" in m and "per-step annotations didn't fire" in m for m in msgs
    ), msgs
    assert any("PROFILE_EXTRA_BODY" in m for m in msgs), (
        "should point operators at the #210 fix path"
    )


# Check [4] (Deval) per-file execute_* in trace_split/
def test_validator_warns_when_split_file_lacks_execute_star(
    tmp_path, caplog,
):
    """A split chunk with no ``execute_*`` annotations warns, naming the file."""
    trace_dir = _build_healthy_layout(tmp_path)
    split = trace_dir / "trace_split"
    bad = split / "decode_only_steady_state_chunk0.json.gz"
    bad.unlink()
    _write_split_file(bad, with_execute_star=False)

    caplog.set_level(logging.WARNING)
    _validate_trace_structure(trace_dir, "sglang")
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "trace_split/ file(s) have NO execute_* user_annotations" in m
        for m in msgs
    ), msgs
    assert any(
        "decode_only_steady_state_chunk0.json.gz" in m for m in msgs
    ), "warning should name the offending file"


# Check [5] (Deval) sglang kernel_shape_profiler presence
def test_validator_warns_when_kernel_shape_profiler_absent_in_sglang(
    tmp_path, caplog,
):
    """A sglang trace lacking ``kernel_shape_profiler`` warns (PR #207 patch missing)."""
    trace_dir = _build_healthy_layout(tmp_path)
    main = next(trace_dir.glob("*.trace.json.gz"))
    main.unlink()
    _write_minimal_sglang_trace(
        main,
        with_kernel_shape_profiler=False,
    )

    caplog.set_level(logging.WARNING)
    _validate_trace_structure(trace_dir, "sglang")
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("kernel_shape_profiler" in m for m in msgs), msgs
    assert any("PR #207" in m for m in msgs), (
        "warning should point at the server-side patcher PR for diagnosis"
    )


def test_validator_skips_kernel_shape_check_for_non_sglang(tmp_path, caplog):
    """The ``kernel_shape_profiler`` check is sglang-specific and must not fire for vLLM."""
    trace_dir = _build_healthy_layout(tmp_path)
    main = next(trace_dir.glob("*.trace.json.gz"))
    main.unlink()
    _write_minimal_sglang_trace(
        main,
        with_kernel_shape_profiler=False,
    )

    caplog.set_level(logging.WARNING)
    _validate_trace_structure(trace_dir, "vllm")
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("kernel_shape_profiler" in m for m in msgs), (
        f"vLLM run should NOT produce kernel_shape_profiler warnings, got: {msgs}"
    )


# Defensive: validator is best-effort
def test_validator_never_raises_even_on_unreadable_trace(tmp_path, caplog):
    """A malformed/truncated trace must log and continue, never raise."""
    trace_dir = tmp_path / "torch_trace"
    trace_dir.mkdir()
    (trace_dir / "capture_traces").mkdir()
    # Truncated gzip header — gzip.open() succeeds, .read() raises.
    (trace_dir / "capture_traces" / "x.json.gz").write_bytes(b"\x1f\x8b")
    (trace_dir / "trace_split").mkdir()
    (trace_dir / "trace_split" / "decode_only_steady_state.json.gz").write_bytes(b"\x1f\x8b")
    (trace_dir / "broken.trace.json.gz").write_bytes(b"\x1f\x8b")
    # Should not raise:
    _validate_trace_structure(trace_dir, "sglang")


# ---------------------------------------------------------------------------
# Issue #431: structured trace_health return (basis for the eager-mode
# re-profile fallback when CUDA-graph folding zeroes out hot kernels)
# ---------------------------------------------------------------------------
def test_trace_health_healthy_layout(tmp_path):
    """Healthy layout: per-kernel attribution intact, capture traces
    present, no issues — so the kernel pipeline keeps the cuda-graph
    profile and does NOT trigger an eager re-profile."""
    trace_dir = _build_healthy_layout(tmp_path)
    health = _validate_trace_structure(trace_dir, "sglang")
    assert health["per_kernel_attribution_degraded"] is False
    assert health["capture_traces_present"] is True
    assert health["issues"] == []


def test_trace_health_flags_degraded_attribution_cuda_graph(tmp_path):
    """#431 core symptom: the main trace carries NO execute_* /
    user_annotation events (per-kernel device activity folded into
    hipGraphLaunch wrappers under cuda-graph capture). trace_health must
    flag ``per_kernel_attribution_degraded=True`` so the kernel pipeline
    can route to an eager-mode re-profile instead of silently producing
    hot_kernels=0."""
    trace_dir = _build_healthy_layout(tmp_path)
    # Overwrite the main trace with one carrying no annotations.
    for p in trace_dir.glob("*.trace.json.gz"):
        p.unlink()
    _write_minimal_sglang_trace(
        trace_dir / "1776409856.2485812-TP-0.trace.json.gz",
        with_kernel_shape_profiler=True,
        with_user_annotation=False,
        with_execute_star=False,
    )
    health = _validate_trace_structure(trace_dir, "sglang")
    assert health["per_kernel_attribution_degraded"] is True
    # capture_traces/ is still intact — a capture-fold fallback (#431
    # proper fix) would have data to mine even though the live trace lost
    # per-kernel attribution.
    assert health["capture_traces_present"] is True
    assert any("execute_* / user_annotation" in m for m in health["issues"]), health["issues"]


def test_trace_health_capture_traces_absent(tmp_path):
    """When ``capture_traces/`` is empty, ``capture_traces_present`` must
    be False (a capture-fold fallback would have nothing to mine)."""
    trace_dir = _build_healthy_layout(tmp_path)
    capture = trace_dir / "capture_traces"
    for p in list(capture.iterdir()):
        p.unlink()
    health = _validate_trace_structure(trace_dir, "sglang")
    assert health["capture_traces_present"] is False


def test_trace_health_return_shape_backward_compatible(tmp_path):
    """Validator now returns a structured dict (issue #431) while staying
    backward compatible: existing callers that ignore the return value are
    unaffected (it never raises)."""
    trace_dir = _build_healthy_layout(tmp_path)
    health = _validate_trace_structure(trace_dir, "sglang")
    assert isinstance(health, dict)
    assert {"issues", "per_kernel_attribution_degraded",
            "capture_traces_present"} <= set(health)
    assert isinstance(health["issues"], list)
