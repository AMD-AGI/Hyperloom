# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A serving failure only accuses the kernel when the GPU actually faulted."""

from __future__ import annotations

from types import SimpleNamespace

from kernelforge.workflows.forge_fusion import serving_gate
from kernelforge.workflows.forge_fusion.validate import (
    SMOKE_STAGE_STARTUP_CRASH,
    SmokeVerdict,
    _explicit_fatal_error,
    _is_hard_gpu_fault,
    _serving_crash_reason,
)

# The failure that prompted this: an install that needs an env var it was not given.
AITER_LOG = """
(EngineCore pid=367907) INFO 08-15 15:25:16 [core.py:114] Initializing a V1 LLM engine
(EngineCore pid=367907) ERROR 08-15 15:32:33 [core.py:1231] EngineCore failed to start.
(EngineCore pid=367907) ERROR 08-15 15:32:33 [core.py:1231]     raise RuntimeError(
(EngineCore pid=367907) ERROR 08-15 15:32:33 [core.py:1231] RuntimeError: Sparse attention indexer ROCm path is only supported on AITER. Please enable aiter with VLLM_ROCM_USE_AITER=1
(APIServer pid=367617) RuntimeError: Engine core initialization failed. See root cause above.
"""

FAULT_LOG = """
[rank0] Memory access fault by GPU node-1 on address 0x7f0000000000
"""


def test_engine_start_failure_is_reported_not_guessed_at() -> None:
    reason = _serving_crash_reason(AITER_LOG)

    assert "VLLM_ROCM_USE_AITER" in reason
    assert "no explicit GPU-fault line" not in reason


def test_a_fault_marker_outranks_a_later_exception_line() -> None:
    mixed = FAULT_LOG + "\nRuntimeError: some later noise\n"

    assert "Memory access fault" in _serving_crash_reason(mixed)


def test_silence_stays_silence() -> None:
    reason = _serving_crash_reason("nothing interesting here\n")

    assert reason == "server exited unexpectedly (no explicit GPU-fault line)"


def test_prefixes_are_stripped_from_the_reported_line() -> None:
    assert _explicit_fatal_error(AITER_LOG).startswith("RuntimeError:")


def test_the_root_cause_outranks_the_wrapper_that_points_at_it() -> None:
    # The API server wraps the engine's failure, so the last line says least.
    assert "root cause above" not in _serving_crash_reason(AITER_LOG)


def _gate_verdict(monkeypatch, tmp_path, server_log: str):
    """Run the serving gate against a failed boot and return the resulting verdict."""
    monkeypatch.setattr(serving_gate, "_serving_check_enabled", lambda: True)
    verdict = SmokeVerdict(
        ok=False,
        reason=_serving_crash_reason(server_log),
        stage=SMOKE_STAGE_STARTUP_CRASH,
        blames_kernel=_is_hard_gpu_fault(server_log),
    )
    monkeypatch.setattr(serving_gate, "serving_smoke_verdict", lambda *a, **k: verdict)
    vr = SimpleNamespace(note="", kept=True, correctness_passed=True, kernel_speedup=1.5)
    result = SimpleNamespace(
        kept=True,
        best=vr,
        best_recipe=SimpleNamespace(
            env_flag="X_FUSED",
            pattern_id="llm:x",
            source_file="/s.py",
        ),
        termination_reason="",
    )

    serving_gate.apply_serving_gate(
        result, serving_gate.ServingSession(framework="vllm", model_path="/m", gpu="0", isl=8, osl=8), tmp_path
    )
    return result


def test_the_gate_files_the_cuda_graph_lesson_for_a_real_fault(monkeypatch, tmp_path) -> None:
    result = _gate_verdict(monkeypatch, tmp_path, FAULT_LOG)

    assert result.termination_reason == "serving_crash"
    assert "NOT CUDA-graph-capture safe" in result.best.note


def test_the_gate_does_not_send_the_author_after_a_server_that_never_started(monkeypatch, tmp_path) -> None:
    """The wiring, not the classifier: a gate that never asks repeats the bug."""
    result = _gate_verdict(monkeypatch, tmp_path, AITER_LOG)

    assert result.termination_reason == "serving_unconfirmed"
    assert result.kept is True
    assert result.best.kernel_speedup == 1.5
    assert "defer e2e" in result.best.note.lower()
    assert "Re-author" not in result.best.note


def test_a_real_fault_is_not_exportable(monkeypatch, tmp_path) -> None:
    result = _gate_verdict(monkeypatch, tmp_path, FAULT_LOG)

    assert result.kept is False
    assert result.best.kernel_speedup is None


def test_an_env_boot_miss_stays_exportable_for_e2e(monkeypatch, tmp_path) -> None:
    result = _gate_verdict(monkeypatch, tmp_path, AITER_LOG)

    assert result.kept is True
    assert result.best.kernel_speedup == 1.5
