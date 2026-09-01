# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

"""Regression tests for the GPU-preflight failure seen on Kimi-K3.

Session ``Kimi-K3/20260830T162217Z-8e8fbee2`` burned five roofline attempts and
produced no usable trace. Three of the five failed for a reason the retry loop
could not have recovered from as written: three profile attempts inside three
minutes were all refused by vLLM with ``Free memory on device cuda:0
(84.11/287.98 GiB) on startup is less than desired GPU memory utilization`` --
an ``explore`` variant server orphaned by a dead driver was still holding
~204 GiB per card. The retry loop changed nothing between attempts, so all
three were guaranteed to fail.

Hermetic: no GPU, no subprocess, no network.
"""

import asyncio
from pathlib import Path
from unittest.mock import patch


from hyperloom.orchestrator.actions.executors import roofline as rf
from hyperloom.orchestrator.actions.executors.baseline import (
    _is_cuda_graph_capture_failure,
    _is_insufficient_gpu_memory,
)


# --------------------------------------------------------------------------
# _is_insufficient_gpu_memory -- classify "somebody else holds the VRAM"
# --------------------------------------------------------------------------

# Verbatim from the failing session's server.log.
_VLLM_REFUSAL = (
    "ValueError: Free memory on device cuda:0 (84.11/287.98 GiB) on startup is "
    "less than desired GPU memory utilization (0.95, 273.59 GiB). Decrease GPU "
    "memory utilization or reduce GPU memory used by other processes."
)
_SGLANG_REFUSAL = "Not enough memory. Please try to increase --mem-fraction-static."


def test_detects_vllm_startup_refusal():
    assert _is_insufficient_gpu_memory(_VLLM_REFUSAL)


def test_detects_sglang_startup_refusal():
    assert _is_insufficient_gpu_memory(_SGLANG_REFUSAL)


def test_scans_every_blob_like_the_cuda_graph_classifier():
    assert _is_insufficient_gpu_memory("", "unrelated noise", _VLLM_REFUSAL)


def test_ignores_a_mid_run_workload_oom():
    # A genuine OOM is NOT recoverable by reaping a squatter; reclaiming and
    # retrying would just burn attempts.
    assert not _is_insufficient_gpu_memory("torch.OutOfMemoryError: HIP out of memory. Tried to allocate 2.00 GiB")


def test_ignores_an_empty_blob():
    assert not _is_insufficient_gpu_memory("", "")


def test_disjoint_from_the_cuda_graph_classifier():
    # The two retry adaptations must not both fire on the same failure.
    assert not _is_cuda_graph_capture_failure(_VLLM_REFUSAL)
    assert not _is_insufficient_gpu_memory("Capture cuda graph failed")


# --------------------------------------------------------------------------
# _reclaim_gpus_for_retry -- actually change something between attempts
# --------------------------------------------------------------------------


def _patch_reclaim(*, reaped, killed, probe=None):
    """Patch the three blocking helpers plus the settle sleep."""
    slept: list[float] = []

    async def fake_sleep(secs):
        slept.append(secs)

    return (
        slept,
        patch(
            "hyperloom.orchestrator.actions.executors._server_lifecycle.reap_orphaned_servers",
            return_value=reaped,
        ),
        patch(
            "hyperloom.orchestrator.actions.executors.recover.kill_stale_gpu_owners",
            return_value=killed,
        ),
        patch(
            "hyperloom.orchestrator.actions.executors.recover.probe_gpu_free_mb",
            return_value=probe if probe is not None else [{"gpu_id": 0, "free_mb": 280000.0}],
        ),
        patch.object(rf.asyncio, "sleep", new=fake_sleep),
    )


def test_reclaim_reaps_orphans_and_settles_before_retry(tmp_path):
    slept, p_reap, p_kill, p_probe, p_sleep = _patch_reclaim(reaped=[30933], killed=[])
    with p_reap as reap, p_kill as kill, p_probe as probe, p_sleep:
        asyncio.run(rf._reclaim_gpus_for_retry(tmp_path, attempt=1))
    reap.assert_called_once_with(Path(tmp_path))
    kill.assert_called_once_with()
    probe.assert_called_once_with()
    # A SIGKILLed server's VRAM is not returned instantly, so the retry waits.
    assert slept == [rf._GPU_RECLAIM_SETTLE_S]


def test_reclaim_falls_back_to_killing_stale_owners(tmp_path):
    # The squatter's pidfile is already gone -- recover's soft cleanup is the
    # only thing left that can free the cards.
    slept, p_reap, p_kill, p_probe, p_sleep = _patch_reclaim(
        reaped=[], killed=[{"pid": 30933, "cmd": "VLLM::EngineCore", "signal": "KILL"}]
    )
    with p_reap, p_kill as kill, p_probe, p_sleep:
        asyncio.run(rf._reclaim_gpus_for_retry(tmp_path, attempt=2))
    kill.assert_called_once_with()
    assert slept == [rf._GPU_RECLAIM_SETTLE_S]


def test_reclaim_does_not_settle_when_nothing_was_reclaimed(tmp_path):
    # The VRAM belongs to something outside this session: sleeping 20 s would
    # only delay a failure that is already certain.
    slept, p_reap, p_kill, p_probe, p_sleep = _patch_reclaim(reaped=[], killed=[])
    with p_reap, p_kill, p_probe as probe, p_sleep:
        asyncio.run(rf._reclaim_gpus_for_retry(tmp_path, attempt=1))
    assert slept == []
    probe.assert_not_called()


def test_reclaim_never_raises_when_the_helpers_blow_up(tmp_path):
    # Reclaiming is an optimisation on the retry path; a failure here must not
    # mask the underlying profile error.
    with (
        patch(
            "hyperloom.orchestrator.actions.executors._server_lifecycle.reap_orphaned_servers",
            side_effect=OSError("proc gone"),
        ),
        patch(
            "hyperloom.orchestrator.actions.executors.recover.kill_stale_gpu_owners",
            side_effect=RuntimeError("rocm-smi missing"),
        ),
    ):
        asyncio.run(rf._reclaim_gpus_for_retry(tmp_path, attempt=3))
