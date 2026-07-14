# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""WS2: ``--gpu-specialist-capacity`` defaults to the whole machine.

The default is the visible GPU count on the launch host; operators disable
or resize the pool with the explicit CLI flag.
"""

from __future__ import annotations

from hyperloom.inference_optimizer.cli.parser import _default_gpu_specialist_capacity


def test_env_value_is_ignored(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_GPU_SPECIALIST_CAPACITY", "4")
    monkeypatch.setattr(
        "hyperloom.orchestrator.policy.gate.detect_gpu_count",
        lambda: 8,
    )
    assert _default_gpu_specialist_capacity() == 8


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
