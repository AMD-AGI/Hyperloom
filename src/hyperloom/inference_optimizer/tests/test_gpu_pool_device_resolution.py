# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for GPU-specialist device-pool resolution (mask-scoping)."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.bus.gpu_pool import (
    SpecialistGpuPool,
    resolve_gpu_specialist_devices,
    resolve_whole_machine_devices,
)
from hyperloom.orchestrator.bus.storage.connection import SqliteConnection


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
    # A ROCR-pinned run keeps specialists on the masked cards (absolute ids).
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


# Serving holds the first ``serving_tp`` cards; they are carved off the
# specialist pool so a specialist never co-resides on a serving card.
def test_serving_tp_carves_whole_machine_pool(monkeypatch) -> None:
    # No mask + 8-card box, TP=4 serving → specialist pool {4,5,6,7}.
    assert resolve_gpu_specialist_devices(8, serving_tp=4) == [4, 5, 6, 7]


def test_serving_tp_carves_rocr_mask_pool(monkeypatch) -> None:
    # A ROCR-pinned 8-card mask with TP=4 serving carves the serving cards off
    # the front and keeps the specialist on the remaining masked (absolute) ids.
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    assert resolve_gpu_specialist_devices(8, serving_tp=4) == [4, 5, 6, 7]


def test_serving_tp_zero_preserves_legacy_whole_pool(monkeypatch) -> None:
    # serving_tp=0 (the default) preserves the whole-pool behaviour.
    assert resolve_gpu_specialist_devices(8, serving_tp=0) == [0, 1, 2, 3, 4, 5, 6, 7]
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2,3")
    assert resolve_gpu_specialist_devices(4, serving_tp=0) == [0, 1, 2, 3]


def test_serving_tp_claims_whole_pool_yields_empty(monkeypatch) -> None:
    # serving_tp >= pool size leaves no free cards for a specialist.
    assert resolve_gpu_specialist_devices(4, serving_tp=4) == []
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2,3")
    assert resolve_gpu_specialist_devices(4, serving_tp=8) == []


def test_explicit_operator_pool_ignores_serving_tp(monkeypatch) -> None:
    # The operator pool is already carved; serving_tp must NOT subtract again.
    monkeypatch.setenv("INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES", "4;5;6;7")
    assert resolve_gpu_specialist_devices(4, serving_tp=4) == [4, 5, 6, 7]


def test_explicit_pool_all_invalid_fails_closed(monkeypatch) -> None:
    """An explicit pool that parses to empty must return [] (fail closed), not fall through."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES", "gpu4,gpu5")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    result = resolve_gpu_specialist_devices(4)
    assert result == [], f"expected [] (fail closed), got {result!r}"


def test_whole_machine_explicit_pool_all_invalid_fails_closed(monkeypatch) -> None:
    """resolve_whole_machine_devices also fails closed on an unusable explicit pool."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES", "-1,-2")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2,3")
    result = resolve_whole_machine_devices()
    assert result == [], f"expected [] (fail closed), got {result!r}"


def test_empty_string_env_falls_back_to_mask(monkeypatch) -> None:
    """An empty INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES is treated as unset."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES", "")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2,3")
    assert resolve_gpu_specialist_devices(4) == [0, 1, 2, 3]
    assert resolve_whole_machine_devices() == [0, 1, 2, 3]


def test_blank_string_env_falls_back_to_mask(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES", "   ")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5")
    assert resolve_gpu_specialist_devices(2) == [4, 5]


@pytest.mark.asyncio
async def test_try_acquire_same_holder_task_is_idempotent(tmp_path) -> None:
    """Repeated try_acquire for the same holder_id + task_id returns the existing lease."""
    db = SqliteConnection(tmp_path / "test.db")
    pool = SpecialistGpuPool(db, gpu_ids=[0, 1, 2, 3])

    lease_a = await pool.try_acquire(count=2, holder_id="h1", task_id="t1")
    assert lease_a is not None
    assert len(lease_a.gpu_ids) == 2

    lease_b = await pool.try_acquire(count=2, holder_id="h1", task_id="t1")
    assert lease_b is not None
    assert set(lease_b.gpu_ids) == set(lease_a.gpu_ids), "idempotent re-acquire must return the same GPU ids"

    lease_c = await pool.try_acquire(count=2, holder_id="h2", task_id="t2")
    assert lease_c is not None, "the remaining 2 GPUs must still be available for a different holder"
    assert set(lease_c.gpu_ids).isdisjoint(set(lease_a.gpu_ids))

    db.close()


@pytest.mark.asyncio
async def test_try_acquire_reallocates_when_count_mismatches(tmp_path) -> None:
    """When a re-acquire requests more GPUs the stale lease is released and a fresh one granted."""
    db = SqliteConnection(tmp_path / "test.db")
    pool = SpecialistGpuPool(db, gpu_ids=[0, 1, 2, 3])

    lease_a = await pool.try_acquire(count=1, holder_id="h1", task_id="t1")
    assert lease_a is not None and len(lease_a.gpu_ids) == 1

    lease_b = await pool.try_acquire(count=2, holder_id="h1", task_id="t1")
    assert lease_b is not None, "count mismatch must trigger reallocation"
    assert len(lease_b.gpu_ids) == 2

    db.close()


@pytest.mark.asyncio
async def test_try_acquire_reallocates_when_pool_shrinks(tmp_path) -> None:
    """When the pool no longer contains the existing lease ids a fresh lease is granted."""
    db = SqliteConnection(tmp_path / "test.db")
    full_pool = SpecialistGpuPool(db, gpu_ids=[0, 1, 2, 3])

    lease_a = await full_pool.try_acquire(count=2, holder_id="h1", task_id="t1")
    assert lease_a is not None

    small_pool = SpecialistGpuPool(db, gpu_ids=[2, 3])
    lease_b = await small_pool.try_acquire(count=2, holder_id="h1", task_id="t1")
    assert lease_b is not None
    assert set(lease_b.gpu_ids) <= {2, 3}, "re-acquired ids must all be in the current pool"

    db.close()
