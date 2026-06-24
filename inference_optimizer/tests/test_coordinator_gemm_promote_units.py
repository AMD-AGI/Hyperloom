# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit coverage for the unified GEMM-tuning result handling on Coordinator.

Exercises ``_promote_gemm_tuning_keep`` guard rails plus the forge/geak
promote branches, and the ``_handle_gemm_tuning_result`` routing that keeps
forge results on the per-tuner E2E path while GEAK results promote inline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import inference_optimizer.orchestrator.action_executors.explore as explore_mod
import inference_optimizer.orchestrator.kernel_request_handlers as krh_mod
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.shared_state import SharedState


def _make_integrate(responses):
    """Return an async ``integrate_handler`` double yielding queued responses."""
    calls: list[dict] = []

    async def _fake(payload, *, session_dir):
        calls.append(payload)
        idx = len(calls) - 1
        return responses[idx] if idx < len(responses) else responses[-1]

    _fake.calls = calls
    return _fake


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


class TestPromoteInjectsCkBlockscaleEnv:
    """PR2: an fp8 block-scale a8w8 tuner KEEP on sglang+fp8 also activates the
    CK backend switch by injecting ``SGLANG_FP8_BLOCKSCALE_CK_MAX_M=256``
    (attributed to gemm_tuning). It must NOT fire for MoE / bf16 / other tuner
    envs, and must never clobber an operator-set value."""

    def test_injects_for_fp8_blockscale_sglang_keep(self, tmp_path):
        coord = _coord(
            tmp_path,
            baseline_tput=100.0,
            framework="sglang",
            precision="fp8",
        )
        coord._promote_gemm_tuning_keep(
            {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.2,
                "backend": "geak",
                "tuned_file": "/tuned/gemm.csv",
            }
        )
        envs = coord.shared_state.current_best["extra_envs"]
        assert envs["AITER_CONFIG_GEMM_A8W8_BLOCKSCALE"] == "/tuned/gemm.csv"
        assert envs["SGLANG_FP8_BLOCKSCALE_CK_MAX_M"] == "256"
        # Flows into the stacked entry too (shared extra_envs object).
        stack_envs = coord.shared_state.optimization_stack[0]["extra_envs"]
        assert stack_envs["SGLANG_FP8_BLOCKSCALE_CK_MAX_M"] == "256"

    def test_does_not_inject_for_moe_forge_tuner(self, tmp_path):
        coord = _coord(
            tmp_path,
            baseline_tput=100.0,
            framework="sglang",
            precision="fp8",
        )
        coord._promote_gemm_tuning_keep(
            {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.2,
                "backend": "forge",
                "extra_envs": {"AITER_CONFIG_FMOE": "/fmoe.json"},
            }
        )
        envs = coord.shared_state.current_best["extra_envs"]
        assert "SGLANG_FP8_BLOCKSCALE_CK_MAX_M" not in envs

    def test_does_not_inject_for_bf16_precision(self, tmp_path):
        coord = _coord(
            tmp_path,
            baseline_tput=100.0,
            framework="sglang",
            precision="bf16",
        )
        coord._promote_gemm_tuning_keep(
            {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.2,
                "backend": "geak",
                "tuned_file": "/tuned/gemm.csv",
            }
        )
        envs = coord.shared_state.current_best["extra_envs"]
        assert "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE" in envs
        assert "SGLANG_FP8_BLOCKSCALE_CK_MAX_M" not in envs

    def test_does_not_inject_for_non_sglang_framework(self, tmp_path):
        coord = _coord(
            tmp_path,
            baseline_tput=100.0,
            framework="vllm",
            precision="fp8",
        )
        coord._promote_gemm_tuning_keep(
            {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.2,
                "backend": "geak",
                "tuned_file": "/tuned/gemm.csv",
            }
        )
        envs = coord.shared_state.current_best["extra_envs"]
        assert "SGLANG_FP8_BLOCKSCALE_CK_MAX_M" not in envs

    def test_respects_preset_value_setdefault(self, tmp_path):
        coord = _coord(
            tmp_path,
            baseline_tput=100.0,
            framework="sglang",
            precision="fp8",
        )
        coord._promote_gemm_tuning_keep(
            {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.2,
                "backend": "forge",
                "extra_envs": {
                    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": "/tuned/gemm.csv",
                    "SGLANG_FP8_BLOCKSCALE_CK_MAX_M": "512",
                },
            }
        )
        envs = coord.shared_state.current_best["extra_envs"]
        assert envs["SGLANG_FP8_BLOCKSCALE_CK_MAX_M"] == "512"


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
    async def test_forge_e2e_rewrites_latest_attempt_history(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        fake = _make_integrate([{"decision": "REVERT", "new_tput": 90.0, "gain_pct": -10.0}])
        monkeypatch.setattr(krh_mod, "integrate_handler", fake)

        await coord._handle_gemm_tuning_result(
            {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.5,
                "backend": "forge",
                "engine": "forge",
                "requires_e2e_validation": True,
                "recommended_env": {"AITER_DENSE": "/dense.json"},
                "extra_envs": {"AITER_DENSE": "/dense.json"},
                "tuners_run": [
                    {
                        "status": "ok",
                        "improved_shapes": 3,
                        "tuner": "dense_gemm",
                        "env_var": "AITER_DENSE",
                        "env_value": "/dense.json",
                    }
                ],
            }
        )

        attempts = coord.shared_state.gemm_tuning_attempts
        assert len(attempts) == 1
        assert attempts[0]["engine"] == "forge"
        assert attempts[0]["e2e_validated"] is True
        assert attempts[0]["decision"] == "REVERT"
        assert attempts[0]["best_speedup"] == 1.5
        assert coord.shared_state.last_gemm_tuning["decision"] == "REVERT"

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


class TestValidateForgeGemmTuningE2E:
    @pytest.mark.asyncio
    async def test_no_candidates_returns_early(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        fake = _make_integrate([{"decision": "KEEP", "new_tput": 200.0, "gain_pct": 100.0}])
        monkeypatch.setattr(krh_mod, "integrate_handler", fake)

        result = {
            "recommended_env": {"X": "1"},
            "extra_envs": {"X": "1"},
            "requires_e2e_validation": True,
            "tuners_run": [
                "not-a-dict",
                {"status": "failed", "improved_shapes": 5, "env_var": "A", "env_value": "1"},
                {"status": "ok", "improved_shapes": 0, "env_var": "B", "env_value": "2"},
                {"status": "ok", "improved_shapes": 3, "env_var": "", "env_value": ""},
            ],
        }
        await coord._validate_forge_gemm_tuning_e2e(result)

        assert fake.calls == []
        # No candidates means the result is left untouched (no rewrite).
        assert result["requires_e2e_validation"] is True
        assert "e2e_validated" not in result
        assert coord.shared_state.optimization_stack == []

    @pytest.mark.asyncio
    async def test_keep_stacks_envs_and_rewrites_result(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        fake = _make_integrate([
            {"decision": "KEEP", "new_tput": 120.0, "gain_pct": 20.0},
            {"decision": "KEEP", "new_tput": 132.0, "gain_pct": 10.0},
        ])
        monkeypatch.setattr(krh_mod, "integrate_handler", fake)

        result = {
            "workspace": str(tmp_path),
            "recommended_env": {"AITER_CONFIG_FMOE": "/fmoe.json", "AITER_DENSE": "/dense.json"},
            "extra_envs": {"AITER_CONFIG_FMOE": "/fmoe.json", "AITER_DENSE": "/dense.json"},
            "requires_e2e_validation": True,
            "tuners_run": [
                {
                    "status": "ok", "improved_shapes": 5, "tuner": "fmoe_ck",
                    "env_var": "AITER_CONFIG_FMOE", "env_value": "/fmoe.json",
                    "best_micro_speedup": 1.2,
                },
                {
                    "status": "ok", "improved_shapes": 3, "tuner": "dense_gemm",
                    "env_var": "AITER_DENSE", "env_value": "/dense.json",
                    "best_micro_speedup": 1.1,
                },
            ],
        }
        await coord._validate_forge_gemm_tuning_e2e(result)

        # fmoe_ck on sglang carries the aiter MoE runner arg; dense does not.
        assert fake.calls[0]["extra_server_args"] == "--moe-runner-backend aiter"
        assert fake.calls[1]["extra_server_args"] == ""
        # First integrate sees only its own env; the second sees the stacked set.
        assert fake.calls[0]["extra_envs"] == {"AITER_CONFIG_FMOE": "/fmoe.json"}
        assert fake.calls[1]["extra_envs"] == {
            "AITER_CONFIG_FMOE": "/fmoe.json",
            "AITER_DENSE": "/dense.json",
        }
        # base_tput advances after the first KEEP.
        assert fake.calls[1]["base_tput"] == pytest.approx(120.0)

        assert len(coord.shared_state.optimization_stack) == 2
        cb = coord.shared_state.current_best
        assert cb["engine"] == "forge"
        assert cb["tput"] == pytest.approx(132.0)
        assert cb["extra_server_args"] == "--moe-runner-backend aiter"
        assert coord.shared_state.cumulative_gain == pytest.approx(32.0)
        assert coord.shared_state.cumulative_gain_validated == pytest.approx(32.0)

        # Result rewritten to the E2E-validated outcome.
        assert result["e2e_validated"] is True
        assert result["requires_e2e_validation"] is False
        assert result["decision"] == "KEEP"
        assert result["status"] == "complete"
        assert result["e2e_gain_pct"] == pytest.approx(32.0)
        assert result["recommended_env"] == {
            "AITER_CONFIG_FMOE": "/fmoe.json",
            "AITER_DENSE": "/dense.json",
        }
        # Raw (pre-validation) envs are preserved for debugging.
        assert result["recommended_env_raw"] == {
            "AITER_CONFIG_FMOE": "/fmoe.json",
            "AITER_DENSE": "/dense.json",
        }

    @pytest.mark.asyncio
    async def test_keep_only_when_tput_improves(self, tmp_path, monkeypatch):
        # decision==KEEP but new_tput not above running_tput → treated as REVERT.
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        fake = _make_integrate([{"decision": "KEEP", "new_tput": 100.0, "gain_pct": 0.0}])
        monkeypatch.setattr(krh_mod, "integrate_handler", fake)

        result = {
            "recommended_env": {"X": "1"},
            "extra_envs": {"X": "1"},
            "requires_e2e_validation": True,
            "tuners_run": [
                {
                    "status": "ok", "improved_shapes": 2, "tuner": "dense",
                    "env_var": "X", "env_value": "1", "best_micro_speedup": 1.05,
                },
            ],
        }
        await coord._validate_forge_gemm_tuning_e2e(result)

        assert coord.shared_state.optimization_stack == []
        assert result["decision"] == "REVERT"
        assert result["micro_decision"] == "candidate_no_e2e_gain"

    @pytest.mark.asyncio
    async def test_all_revert_resets_and_marks_no_gain(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path, baseline_tput=100.0, framework="vllm")
        fake = _make_integrate([{"decision": "REVERT", "new_tput": 90.0, "gain_pct": -10.0}])
        monkeypatch.setattr(krh_mod, "integrate_handler", fake)

        result = {
            "recommended_env": {"X": "1"},
            "extra_envs": {"X": "1"},
            "requires_e2e_validation": True,
            "tuners_run": [
                {
                    "status": "ok", "improved_shapes": 4, "tuner": "dense",
                    "env_var": "X", "env_value": "1", "best_micro_speedup": 1.05,
                },
            ],
        }
        await coord._validate_forge_gemm_tuning_e2e(result)

        assert coord.shared_state.optimization_stack == []
        assert result["decision"] == "REVERT"
        assert result["micro_decision"] == "candidate_no_e2e_gain"
        assert result["e2e_gain_pct"] == 0.0
        assert result["recommended_env"] == {}
        assert result["requires_e2e_validation"] is False

    @pytest.mark.asyncio
    async def test_integrate_exception_reverts_tuner(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")

        async def _boom(payload, *, session_dir):
            raise RuntimeError("integrate crashed")

        monkeypatch.setattr(krh_mod, "integrate_handler", _boom)

        result = {
            "recommended_env": {"X": "1"},
            "extra_envs": {"X": "1"},
            "requires_e2e_validation": True,
            "tuners_run": [
                {
                    "status": "ok", "improved_shapes": 2, "tuner": "dense",
                    "env_var": "X", "env_value": "1", "best_micro_speedup": 1.1,
                },
            ],
        }
        await coord._validate_forge_gemm_tuning_e2e(result)

        assert result["decision"] == "REVERT"
        reverted = result["e2e_results"]["reverted"]
        assert len(reverted) == 1
        assert reverted[0]["reason"].startswith("RuntimeError")

    @pytest.mark.asyncio
    async def test_timeout_fallback_when_explore_helper_raises(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")

        def _raise(**kwargs):
            raise ValueError("no runtime budget")

        monkeypatch.setattr(explore_mod, "_compute_explore_variant_timeout", _raise)

        captured: dict[str, object] = {}

        async def _fake(payload, *, session_dir):
            captured["budget"] = payload["budget_minutes"]
            return {"decision": "KEEP", "new_tput": 150.0, "gain_pct": 50.0}

        monkeypatch.setattr(krh_mod, "integrate_handler", _fake)

        result = {
            "recommended_env": {"X": "1"},
            "extra_envs": {"X": "1"},
            "requires_e2e_validation": True,
            "tuners_run": [
                {
                    "status": "ok", "improved_shapes": 2, "tuner": "dense",
                    "env_var": "X", "env_value": "1", "best_micro_speedup": 1.1,
                },
            ],
        }
        await coord._validate_forge_gemm_tuning_e2e(result)

        # Fallback budget is 15 * 60 sec → 15 minutes.
        assert captured["budget"] == 15
