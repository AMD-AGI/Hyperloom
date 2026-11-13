# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The GEMM e2e validator runs its fused-MoE probe and its coverage replay off the event loop."""

from __future__ import annotations

import threading

import pytest

from hyperloom.orchestrator.kernel import request_handlers
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.phases.kernel import KernelPhase
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.mark.asyncio
async def test_the_fused_moe_probe_and_the_coverage_replay_run_off_the_event_loop(tmp_path, monkeypatch):
    """Both read a whole server.log (~17MB on the fleet); run on the loop, they stall every other coroutine."""
    threads: dict[str, list[int]] = {}

    def _probe(self):
        threads.setdefault("fused_moe_probe", []).append(threading.get_ident())
        return False

    def _coverage(self, tuner_name, envs):
        threads.setdefault("coverage_replay", []).append(threading.get_ident())

    async def _integrate(payload, *, session_dir):
        return {"decision": "REVERT", "new_tput": 90.0, "gain_pct": -10.0}

    monkeypatch.setattr(KernelPhase, "_runtime_uses_aiter_fused_moe", _probe)
    monkeypatch.setattr(KernelPhase, "_gemm_tuned_config_coverage", _coverage)
    monkeypatch.setattr(request_handlers, "integrate_handler", _integrate)
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(baseline_tput=100.0, framework="vllm")
    result = {"candidates": [{"tuner": "vllm_moe_triton", "env": {"VLLM_TUNED_CONFIG_FOLDER": str(tmp_path)}}]}

    await coord.phase_kernel._validate_gemm_tuning_e2e(result)

    loop_thread = threading.get_ident()
    assert {helper: [ident == loop_thread for ident in idents] for helper, idents in threads.items()} == {
        "fused_moe_probe": [False],
        "coverage_replay": [False],
    }
