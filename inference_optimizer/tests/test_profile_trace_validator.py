# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``profile._validate_trace_structure`` (#210 / Deval's
``check_torch_trace.py`` guidance).

The validator's contract:

* Read-only inspection of the trace folder; never mutates anything.
* Logs WARNINGs (does NOT raise) so a partial / silently-degraded
  profile run still completes — operators see actionable messages
  pointing at the specific symptom (PROFILE_EXTRA_BODY leaked,
  shape-discovery patch missed, etc.).
* No false positives on the healthy reference layout from Deval's
  comment 1.
* Detects the smoking-gun symptom: ``trace_split/`` carrying
  ``_extend_*`` / ``_decode_*`` files instead of ``_steady_state_*``
  (proves ``profile_by_stage=True`` leaked through).
* Detects the sglang-specific symptom: main rank-0 trace lacking
  ``kernel_shape_profiler`` substring (proves the server-side patch
  from PR #207 didn't reach the deployed SGLang).
"""

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
    """Write a tiny gzipped JSON trace blob carrying enough event
    structure to satisfy the validator's substring-based content
    checks. The validator counts:

    * ``"name": "cpu_op"`` events
    * ``"name": "user_annotation"`` events
    * ``"execute_`` substrings (per-step InferenceX annotations)
    * ``kernel_shape_profiler`` substring (sglang-specific marker)

    Each flag here lets a test selectively turn off the corresponding
    signal to assert the validator's per-check warnings independently.
    """
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
    """Synthesize a ``capture_traces/<file>.json.gz`` carrying
    ``cpu_op_count`` events; ``with_input_dims_fraction`` of them
    have an ``Input Dims`` arg field (rest don't). Used to test
    Deval check [2]: capture file shape-discovery instrumentation."""
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
    """Synthesize a ``trace_split/<file>.json.gz`` carrying (or not)
    ``execute_*`` user_annotations. Used to test Deval check [4]."""
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
    """Build a complete reference layout that satisfies all 6 checks.
    Used by tests that need a 'no false positives' baseline AND by
    targeted-symptom tests as a starting point to selectively
    invalidate one piece."""
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
    """Reference healthy layout from Deval's example output (issue
    #210, comment 1) MUST NOT trip any of the 6 checks:

    * [1] ``capture_traces/`` exists with files
    * [2] capture file has ``cpu_op`` + ``Input Dims`` ≥ floor
    * [3] main trace has ``user_annotation`` + ``execute_*``
    * [4] every ``trace_split/`` file has ``execute_*``
    * [5] (sglang) main trace has ``kernel_shape_profiler``
    * [6] ``trace_split/`` has only ``_steady_state_*`` (no _extend/_decode)
    """
    trace_dir = _build_healthy_layout(tmp_path)
    caplog.set_level(logging.WARNING)
    _validate_trace_structure(trace_dir, "sglang")
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == [], (
        f"healthy layout produced unexpected warnings: "
        f"{[r.getMessage() for r in warnings]}"
    )


# ---------------------------------------------------------------------------
# Check [6] (Hyperloom-specific #210 smoking-gun)
# ---------------------------------------------------------------------------
def test_validator_warns_on_extend_decode_files_without_steady_state(
    tmp_path, caplog,
):
    """The exact #210 / mohbasit-comment-3 symptom: ``trace_split/``
    carries ``_extend_*`` / ``_decode_*`` files (per-stage splits)
    instead of ``_steady_state_*`` files. This is the visible signal
    that ``profile_by_stage=True`` leaked through and PROFILE_EXTRA_BODY
    didn't reach the framework."""
    trace_dir = _build_healthy_layout(tmp_path)
    # Replace the healthy steady_state split files with the smoking-gun
    # _extend_* / _decode_* layout.
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


# ---------------------------------------------------------------------------
# Check [1] capture_traces/ presence
# ---------------------------------------------------------------------------
def test_validator_warns_when_capture_traces_missing(tmp_path, caplog):
    """When ``capture_traces/`` doesn't exist at all, graph capture
    didn't fire — typically because the server-side patch from PR
    #207 didn't land. The validator should call this out explicitly."""
    trace_dir = _build_healthy_layout(tmp_path)
    # Remove the healthy capture_traces/ subtree to trigger check [1].
    capture = trace_dir / "capture_traces"
    for p in list(capture.iterdir()):
        p.unlink()
    capture.rmdir()

    caplog.set_level(logging.WARNING)
    _validate_trace_structure(trace_dir, "sglang")
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("capture_traces/" in m and "missing" in m for m in msgs), msgs


# ---------------------------------------------------------------------------
# Check [2] (Deval) capture file has cpu_op + Input Dims
# ---------------------------------------------------------------------------
def test_validator_warns_when_capture_file_lacks_input_dims(tmp_path, caplog):
    """Capture file with cpu_op events but missing the ``Input Dims``
    field on most events → shape-discovery instrumentation isn't
    fully active. Healthy reference is 99.97% (Deval); validator
    floors at 90%."""
    trace_dir = _build_healthy_layout(tmp_path)
    capture = trace_dir / "capture_traces"
    bad = capture / "bs_104_rank0.json.gz"
    bad.unlink()
    # Only 50% of cpu_op events have Input Dims — well below the 90%
    # floor.
    _write_capture_file(bad, cpu_op_count=200, with_input_dims_fraction=0.5)

    caplog.set_level(logging.WARNING)
    _validate_trace_structure(trace_dir, "sglang")
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "Input Dims" in m and "Shape-discovery instrumentation" in m for m in msgs
    ), msgs


def test_validator_warns_when_capture_file_has_no_cpu_op(tmp_path, caplog):
    """Capture file present but contains zero ``cpu_op`` events →
    graph capture wrote files but they don't carry kernel-level
    events. Distinct failure mode from missing capture_traces/
    entirely."""
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


# ---------------------------------------------------------------------------
# Check [3] (Deval) main trace has user_annotation + execute_*
# ---------------------------------------------------------------------------
def test_validator_no_warning_when_execute_star_present_without_user_annotation(
    tmp_path, caplog,
):
    """Regression guard for the profiler-version false positive: some
    torch / SGLang builds (e.g. sglang 0.5.11 on ROCm) emit the
    ``execute_*`` per-step annotation labels WITHOUT wrapping them in a
    literal ``"name": "user_annotation"`` event. That trace is healthy —
    the splitter and roofline analysis key on ``execute_*`` — so Check
    [3] MUST NOT warn just because the ``user_annotation`` wrapper is
    absent."""
    trace_dir = _build_healthy_layout(tmp_path)
    main = next(trace_dir.glob("*.trace.json.gz"))
    main.unlink()
    # execute_* label present, but emitted under a non-user_annotation
    # event name (mimics the ROCm/SGLang 0.5.11 profiler shape).
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
    """Genuine absence: neither ``execute_*`` labels NOR
    ``user_annotation`` events anywhere in the trace → InferenceX
    per-step annotations really didn't fire. This is the only case
    Check [3] should warn on."""
    trace_dir = _build_healthy_layout(tmp_path)
    main = next(trace_dir.glob("*.trace.json.gz"))
    main.unlink()
    _write_minimal_sglang_trace(
        main,
        with_kernel_shape_profiler=True,
        with_user_annotation=False,   # no user_annotation wrapper
        with_execute_star=False,      # and no execute_* labels at all
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


# ---------------------------------------------------------------------------
# Check [4] (Deval) per-file execute_* in trace_split/
# ---------------------------------------------------------------------------
def test_validator_warns_when_split_file_lacks_execute_star(
    tmp_path, caplog,
):
    """Splitter ran but a chunk has no ``execute_*`` annotations →
    the chunk is empty (or worse, the source trace was already
    missing execute_* labels). Validator names the offending
    file(s) so operators can grep them."""
    trace_dir = _build_healthy_layout(tmp_path)
    split = trace_dir / "trace_split"
    bad = split / "decode_only_steady_state_chunk0.json.gz"
    bad.unlink()
    _write_split_file(bad, with_execute_star=False)  # empty split

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


# ---------------------------------------------------------------------------
# Check [5] (Deval) sglang kernel_shape_profiler presence
# ---------------------------------------------------------------------------
def test_validator_warns_when_kernel_shape_profiler_absent_in_sglang(
    tmp_path, caplog,
):
    """sglang trace lacking the ``kernel_shape_profiler`` substring →
    server-side shape-discovery patch didn't land. Pointer to PR #207
    in the message so operators know where to look."""
    trace_dir = _build_healthy_layout(tmp_path)
    main = next(trace_dir.glob("*.trace.json.gz"))
    main.unlink()
    _write_minimal_sglang_trace(
        main,
        with_kernel_shape_profiler=False,  # smoking-gun for check [5]
    )

    caplog.set_level(logging.WARNING)
    _validate_trace_structure(trace_dir, "sglang")
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("kernel_shape_profiler" in m for m in msgs), msgs
    assert any("PR #207" in m for m in msgs), (
        "warning should point at the server-side patcher PR for diagnosis"
    )


def test_validator_skips_kernel_shape_check_for_non_sglang(tmp_path, caplog):
    """vLLM traces don't carry ``kernel_shape_profiler`` events — that
    sentinel is sglang-specific. The validator must NOT fire that
    warning when ``framework != "sglang"``, otherwise vLLM runs would
    log a false-positive on every profile."""
    trace_dir = _build_healthy_layout(tmp_path)
    main = next(trace_dir.glob("*.trace.json.gz"))
    main.unlink()
    _write_minimal_sglang_trace(
        main,
        with_kernel_shape_profiler=False,  # would fire on sglang…
    )

    caplog.set_level(logging.WARNING)
    _validate_trace_structure(trace_dir, "vllm")  # …but framework=vllm
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("kernel_shape_profiler" in m for m in msgs), (
        f"vLLM run should NOT produce kernel_shape_profiler warnings, got: {msgs}"
    )


# ---------------------------------------------------------------------------
# Defensive: validator is best-effort
# ---------------------------------------------------------------------------
def test_validator_never_raises_even_on_unreadable_trace(tmp_path, caplog):
    """Validator is best-effort — if the gzipped trace is malformed /
    truncated, the check must continue and log a debug message, never
    raise (would fail the profile post-execution path otherwise)."""
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
