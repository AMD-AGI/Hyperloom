# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit coverage for the unified GEMM-tuning result handling on Coordinator.

Exercises ``_promote_gemm_tuning_keep`` guard rails plus the forge/geak
promote branches, and the ``_handle_gemm_tuning_result`` routing that keeps
forge results on the per-tuner E2E path while GEAK results promote inline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.shared_state import SharedState


def _coord(tmp_path: Path, **state_kwargs) -> Coordinator:
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(**state_kwargs)
    return coord


class TestPromoteGemmTuningKeep:
    def test_ignores_non_dict(self, tmp_path):
        coord = _coord(tmp_path, baseline_tput=100.0)
        coord._promote_gemm_tuning_keep("not-a-dict")  # type: ignore[arg-type]
        assert coord.shared_state.optimization_stack == []

    def test_ignores_non_ok_status(self, tmp_path):
        coord = _coord(tmp_path, baseline_tput=100.0)
        coord._promote_gemm_tuning_keep({"status": "failed", "decision": "KEEP"})
        assert coord.shared_state.optimization_stack == []

    def test_ignores_non_keep_decision(self, tmp_path):
        coord = _coord(tmp_path, baseline_tput=100.0)
        coord._promote_gemm_tuning_keep({"status": "ok", "decision": "REVERT"})
        assert coord.shared_state.optimization_stack == []

    def test_ignores_unparseable_speedup(self, tmp_path):
        coord = _coord(tmp_path, baseline_tput=100.0)
        coord._promote_gemm_tuning_keep(
            {"status": "ok", "decision": "KEEP", "best_speedup": object()}
        )
        assert coord.shared_state.optimization_stack == []

    def test_ignores_low_speedup_or_baseline(self, tmp_path):
        coord = _coord(tmp_path, baseline_tput=0.0)
        coord._promote_gemm_tuning_keep(
            {"status": "ok", "decision": "KEEP", "best_speedup": 1.5}
        )
        assert coord.shared_state.optimization_stack == []

        coord2 = _coord(tmp_path, baseline_tput=100.0)
        coord2._promote_gemm_tuning_keep(
            {"status": "ok", "decision": "KEEP", "best_speedup": 1.0}
        )
        assert coord2.shared_state.optimization_stack == []

    def test_forge_backend_records_stack_and_current_best(self, tmp_path):
        coord = _coord(tmp_path, baseline_tput=100.0)
        coord._promote_gemm_tuning_keep(
            {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.25,
                "backend": "forge",
                "extra_envs": {"AITER_CONFIG": "/cfg/tuned.json"},
                "workspace": str(tmp_path),
            }
        )
        stack = coord.shared_state.optimization_stack
        assert len(stack) == 1
        assert stack[0]["variant_name"] == "forge_gemm_tuned"
        assert stack[0]["backend"] == "forge"
        assert coord.shared_state.current_best["engine"] == "forge"
        assert coord.shared_state.cumulative_gain == pytest.approx(25.0)
        assert coord.shared_state.cumulative_gain_validated == pytest.approx(25.0)

    def test_geak_backend_uses_tuned_file(self, tmp_path):
        coord = _coord(tmp_path, baseline_tput=200.0)
        coord._promote_gemm_tuning_keep(
            {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.1,
                "backend": "geak",
                "tuned_file": "/tuned/gemm.csv",
            }
        )
        stack = coord.shared_state.optimization_stack
        assert len(stack) == 1
        assert stack[0]["variant_name"] == "a8w8_blockscale_tuned_gemm"
        envs = coord.shared_state.current_best["extra_envs"]
        assert envs["AITER_CONFIG_GEMM_A8W8_BLOCKSCALE"] == "/tuned/gemm.csv"

    def test_forge_keep_dedupes_same_tuned_file(self, tmp_path):
        coord = _coord(tmp_path, baseline_tput=100.0)
        result = {
            "status": "ok",
            "decision": "KEEP",
            "best_speedup": 1.2,
            "backend": "forge",
            "artifacts": {"cfg": "/cfg/tuned.json"},
            "extra_envs": {"AITER_CONFIG": "/cfg/tuned.json"},
        }
        coord._promote_gemm_tuning_keep(result)
        coord._promote_gemm_tuning_keep(result)
        assert len(coord.shared_state.optimization_stack) == 1


class TestHandleGemmTuningResult:
    @pytest.mark.asyncio
    async def test_forge_requires_e2e_routes_to_validator(self, tmp_path):
        coord = _coord(tmp_path, baseline_tput=100.0)
        called: dict[str, object] = {}

        async def _fake_validate(result):
            called["result"] = result

        coord._validate_forge_gemm_tuning_e2e = _fake_validate  # type: ignore[assignment]

        await coord._handle_gemm_tuning_result(
            {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.3,
                "backend": "forge",
                "requires_e2e_validation": True,
                "extra_envs": {"AITER_CONFIG": "/cfg/tuned.json"},
            }
        )

        assert "result" in called
        # Validator owns promotion; inline promote must not have run.
        assert coord.shared_state.optimization_stack == []

    @pytest.mark.asyncio
    async def test_non_forge_routes_to_inline_promote(self, tmp_path):
        coord = _coord(tmp_path, baseline_tput=100.0)

        await coord._handle_gemm_tuning_result(
            {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.4,
                "backend": "geak",
                "tuned_file": "/tuned/gemm.csv",
            }
        )

        assert len(coord.shared_state.optimization_stack) == 1
        assert coord.shared_state.current_best["engine"] == "geak"
