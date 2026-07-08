# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""WS2: ``--gpu-specialist-capacity`` defaults to the whole machine.

Env wins when set (including ``0`` as an explicit disable escape hatch);
otherwise the default is the visible GPU count on the launch host.
"""

from __future__ import annotations

from hyperloom.inference_optimizer.cli import _default_gpu_specialist_capacity


def test_env_value_wins(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_GPU_SPECIALIST_CAPACITY", "4")
    assert _default_gpu_specialist_capacity() == 4


def test_env_zero_disables(monkeypatch):
    # Explicit disable escape hatch must be honoured (not overridden by detect).
    monkeypatch.setenv("INFERENCE_OPTIMIZER_GPU_SPECIALIST_CAPACITY", "0")
    assert _default_gpu_specialist_capacity() == 0


def test_env_unset_falls_back_to_detected_whole_machine(monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_GPU_SPECIALIST_CAPACITY", raising=False)
    monkeypatch.setattr(
        "hyperloom.orchestrator.policy.gate.detect_gpu_count",
        lambda: 8,
    )
    assert _default_gpu_specialist_capacity() == 8


def test_env_blank_falls_back_to_detected(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_GPU_SPECIALIST_CAPACITY", "  ")
    monkeypatch.setattr(
        "hyperloom.orchestrator.policy.gate.detect_gpu_count",
        lambda: 8,
    )
    assert _default_gpu_specialist_capacity() == 8


def test_env_garbage_falls_back_to_detected(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_GPU_SPECIALIST_CAPACITY", "notanint")
    monkeypatch.setattr(
        "hyperloom.orchestrator.policy.gate.detect_gpu_count",
        lambda: 2,
    )
    assert _default_gpu_specialist_capacity() == 2
