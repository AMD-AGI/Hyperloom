# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for GPU-specialist device-pool resolution (mask-scoping)."""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.gpu_pool import resolve_gpu_specialist_devices


_MASK_VARS = (
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES",
)


@pytest.fixture(autouse=True)
def _clean_masks(monkeypatch) -> None:
    for var in _MASK_VARS:
        monkeypatch.delenv(var, raising=False)


def test_non_positive_capacity_disables(monkeypatch) -> None:
    assert resolve_gpu_specialist_devices(0) == []
    assert resolve_gpu_specialist_devices(-3) == []


def test_no_mask_falls_back_to_range(monkeypatch) -> None:
    assert resolve_gpu_specialist_devices(4) == [0, 1, 2, 3]


def test_explicit_pool_wins_and_caps(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES", "2;5;6")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2,3")
    assert resolve_gpu_specialist_devices(2) == [2, 5]


def test_rocr_mask_scopes_pool_to_absolute_ids(monkeypatch) -> None:
    # A ROCR-pinned run must keep specialists on the masked cards (absolute ids),
    # never range(capacity) which would point at cards outside the mask.
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5,6,7")
    assert resolve_gpu_specialist_devices(4) == [4, 5, 6, 7]


def test_rocr_mask_capped_to_capacity(monkeypatch) -> None:
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5,6,7")
    assert resolve_gpu_specialist_devices(2) == [4, 5]


def test_rocr_wins_over_hip(monkeypatch) -> None:
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1,2,3")
    assert resolve_gpu_specialist_devices(4) == [4, 5]


def test_hip_used_when_rocr_unset(monkeypatch) -> None:
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1,3")
    assert resolve_gpu_specialist_devices(8) == [1, 3]


def test_empty_mask_yields_empty_pool(monkeypatch) -> None:
    # An explicitly empty mask means "no visible GPUs", not whole-machine.
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "")
    assert resolve_gpu_specialist_devices(4) == []
