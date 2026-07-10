# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit coverage for the unified GEMM-tuning result handling on Coordinator.

Exercises ``_promote_gemm_tuning_keep`` guard rails plus the forge/geak
promote branches, and the ``_handle_gemm_tuning_result`` routing that keeps
forge results on the per-tuner E2E path while GEAK results promote inline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import hyperloom.inference_optimizer.model_config_utils as mcu_mod
import hyperloom.orchestrator.actions.executors.explore as explore_mod
import hyperloom.orchestrator.kernel.request_handlers as krh_mod
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.phases.kernel import KernelPhase
from hyperloom.orchestrator.state.shared_state import SharedState


def _journal_entries(session_dir: Path) -> list[dict]:
    """Read the optimization_journal entries written under ``session_dir``."""
    path = session_dir / "reports" / "optimization_journal.json"
    if not path.exists():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")).get("entries") or [])


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

    def test_geak_keep_writes_gemm_tuning_journal_event(self, tmp_path):
        # The adopted GEMM-tuning run must surface as a phase_timeline event:
        # a KIND_GEMM_TUNING KEEP journal row carrying the serving throughput
        # and the originating task_id (for token attribution).
        coord = _coord(tmp_path, baseline_tput=200.0)
        coord._promote_gemm_tuning_keep(
            {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.1,
                "backend": "geak",
                "tuned_file": "/tuned/gemm.csv",
                "task_id": "kernel_entry_gemm_tuning",
            }
        )
        rows = [e for e in _journal_entries(tmp_path) if e.get("kind") == "gemm_tuning"]
        assert len(rows) == 1
        row = rows[0]
        assert row["outcome"] == "KEEP"
        assert row["variant_name"] == "a8w8_blockscale_tuned_gemm"
        assert row["throughput_after"] == pytest.approx(220.0)
        assert row["task_id"] == "kernel_entry_gemm_tuning"
        assert row["provenance"] == "gemm_tuning:geak"

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


class TestPromoteFusionIntegrateKeep:
    def test_records_patch_envs_and_current_best(self, tmp_path):
        coord = _coord(
            tmp_path,
            baseline_tput=100.0,
            current_best={
                "action": "replay_warm_recipe",
                "tput": 120.0,
                "extra_envs": {"SGLANG_USE_AITER": "1"},
                "extra_server_args": "--moe-runner-backend aiter",
            },
        )

        KernelPhase(coord)._promote_fusion_integrate_keep(
            {
                "patch": "/tmp/fusion.patch",
                "source_file": "/repo/model.py",
                "kernel_speedup": 3.05,
                "best_pattern": "llm:fused_a+llm:fused_b",
            },
            {
                "status": "ok",
                "decision": "KEEP",
                "new_tput": 180.0,
                "gain_pct": 80.0,
                "workspace": "/tmp/run",
                "extra_server_args": "--moe-runner-backend aiter",
            },
            extra_envs={"SGLANG_USE_AITER": "1", "ZAYA_FUSED_HYBRID_RESIDUAL": "1"},
        )

        stack = coord.shared_state.optimization_stack
        assert len(stack) == 1
        assert stack[0]["action"] == "fusion"
        assert stack[0]["patch_path"] == "/tmp/fusion.patch"
        assert stack[0]["extra_envs"]["SGLANG_USE_AITER"] == "1"
        assert stack[0]["extra_envs"]["ZAYA_FUSED_HYBRID_RESIDUAL"] == "1"
        assert stack[0]["kernel_speedup"] == 3.05
        assert coord.shared_state.current_best["action"] == "fusion"
        assert coord.shared_state.current_best["tput"] == 180.0
        assert coord.shared_state.cumulative_gain_validated == 80.0
        assert coord.shared_state.cumulative_gain_validated_stack_len == 1


class TestBf16DenseFallback:
    def test_fallback_predicate_requires_forge_sglang_fp8_no_candidate(self, tmp_path):
        coord = _coord(tmp_path, framework="sglang")

        assert coord._should_run_bf16_dense_gemm_fallback(
            {
                "backend": "forge",
                "precision": "fp8",
                "framework": "sglang",
                "micro_decision": "no_improvement",
                "tuners_run": [
                    {
                        "status": "no_improvement",
                        "tuner": "a8w8",
                        "improved_shapes": 0,
                    }
                ],
            }
        )

    def test_fallback_predicate_skips_existing_candidate(self, tmp_path):
        coord = _coord(tmp_path, framework="sglang")

        assert not coord._should_run_bf16_dense_gemm_fallback(
            {
                "backend": "forge",
                "precision": "fp8",
                "framework": "sglang",
                "micro_decision": "candidate",
                "recommended_env": {"AITER_CONFIG_GEMM_A8W8": "/tmp/tuned.csv"},
                "tuners_run": [
                    {
                        "status": "ok",
                        "tuner": "a8w8",
                        "improved_shapes": 4,
                        "env_var": "AITER_CONFIG_GEMM_A8W8",
                        "env_value": "/tmp/tuned.csv",
                    }
                ],
            }
        )

    def test_fallback_predicate_skips_candidate_reverted_by_e2e(self, tmp_path):
        coord = _coord(tmp_path, framework="sglang")

        assert not coord._should_run_bf16_dense_gemm_fallback(
            {
                "backend": "forge",
                "precision": "fp8",
                "framework": "sglang",
                "micro_decision": "candidate_no_e2e_gain",
                "e2e_validated": True,
                "e2e_results": {"kept": [], "reverted": [{"tuner": "a8w8"}]},
            }
        )

    def test_fallback_pending_resumes_terminal_fp8_no_candidate(
        self, tmp_path, monkeypatch
    ):
        coord = _coord(
            tmp_path,
            framework="sglang",
            precision="fp8",
            last_gemm_tuning={
                "status": "ok",
                "decision": "REVERT",
                "backend": "forge",
                "precision": "fp8",
                "framework": "sglang",
                "micro_decision": "no_improvement",
                "tuners_run": [{"status": "no_improvement", "tuner": "a8w8"}],
            },
        )
        monkeypatch.setattr(krh_mod, "_resolve_gemm_tuning_backend", lambda _p: "forge")

        assert coord._bf16_dense_gemm_fallback_pending() is True
        assert coord._gemm_tuning_required_before_kernel_opt() is True

        coord.shared_state.gemm_tuning_attempts.append(
            {
                "status": "complete",
                "decision": "REVERT",
                "backend": "forge",
                "precision": "bf16",
                "workspace": str(
                    tmp_path / "runs/gemm_tuning/kernel_entry_gemm_tuning_bf16_fallback"
                ),
                "tuners_run": [{"tuner": "sglang_dense_bf16"}],
            }
        )

        assert coord._bf16_dense_gemm_fallback_pending() is False
        assert coord._gemm_tuning_required_before_kernel_opt() is False

    @pytest.mark.asyncio
    async def test_kernel_entry_runs_bf16_dense_fallback_after_fp8_no_improvement(
        self, tmp_path, monkeypatch
    ):
        coord = _coord(tmp_path, framework="sglang")
        coord.bus = type(
            "Bus",
            (),
            {"append_and_seq": staticmethod(lambda *_args, **_kwargs: None)},
        )()

        async def _append_and_seq(*_args, **_kwargs):
            return None

        coord.bus.append_and_seq = _append_and_seq
        coord.phase_machine._kernel_enabled = lambda: True
        coord.phase_kernel._perfskills_enabled = lambda: False
        coord._gemm_tuning_required_before_kernel_opt = lambda: True
        coord.phase_machine._record_phase_entry_evidence = lambda **_kwargs: None
        coord.phase_kernel._should_continue_kernel_after_gemm = lambda: False

        async def _noop(*_args, **_kwargs):
            return None

        coord.phase_kernel._maybe_reprofile_for_kernel = _noop

        calls: list[dict] = []
        responses = [
            {
                "status": "ok",
                "decision": "REVERT",
                "backend": "forge",
                "engine": "forge",
                "precision": "fp8",
                "framework": "sglang",
                "micro_decision": "no_improvement",
                "tuners_run": [
                    {
                        "status": "no_improvement",
                        "tuner": "a8w8",
                        "improved_shapes": 0,
                    }
                ],
            },
            {
                "status": "ok",
                "decision": "REVERT",
                "backend": "forge",
                "engine": "forge",
                "precision": "bf16",
                "framework": "sglang",
                "micro_decision": "no_improvement",
            },
        ]

        async def _fake_run_gemm(payload, *, session_dir):
            assert session_dir == tmp_path
            calls.append(payload)
            return dict(responses[len(calls) - 1])

        monkeypatch.setattr(krh_mod, "run_gemm_tuning_handler", _fake_run_gemm)

        await coord._on_enter_kernel(from_phase="EXPLORE")

        assert [c["task_id"] for c in calls] == [
            "kernel_entry_gemm_tuning",
            "kernel_entry_gemm_tuning_bf16_fallback",
        ]
        assert calls[1]["precision"] == "bf16"
        assert calls[1]["tuner"] == "sglang_dense_bf16"
        assert coord.shared_state.last_gemm_tuning["precision"] == "bf16"

    @pytest.mark.asyncio
    async def test_kernel_entry_resumes_pending_bf16_fallback_without_rerunning_fp8(
        self, tmp_path, monkeypatch
    ):
        coord = _coord(
            tmp_path,
            framework="sglang",
            precision="fp8",
            last_gemm_tuning={
                "status": "ok",
                "decision": "REVERT",
                "backend": "forge",
                "precision": "fp8",
                "framework": "sglang",
                "micro_decision": "no_improvement",
                "tuners_run": [{"status": "no_improvement", "tuner": "a8w8"}],
            },
        )
        coord.bus = type("Bus", (), {})()

        async def _append_and_seq(*_args, **_kwargs):
            return None

        async def _noop(*_args, **_kwargs):
            return None

        coord.bus.append_and_seq = _append_and_seq
        coord.phase_machine._kernel_enabled = lambda: True
        coord.phase_kernel._perfskills_enabled = lambda: False
        coord.phase_machine._record_phase_entry_evidence = lambda **_kwargs: None
        coord.phase_kernel._should_continue_kernel_after_gemm = lambda: False
        coord.phase_kernel._maybe_reprofile_for_kernel = _noop
        monkeypatch.setattr(krh_mod, "_resolve_gemm_tuning_backend", lambda _p: "forge")

        calls: list[dict] = []

        async def _fake_run_gemm(payload, *, session_dir):
            assert session_dir == tmp_path
            calls.append(payload)
            return {
                "status": "ok",
                "decision": "REVERT",
                "backend": "forge",
                "engine": "forge",
                "precision": "bf16",
                "framework": "sglang",
                "micro_decision": "no_improvement",
            }

        monkeypatch.setattr(krh_mod, "run_gemm_tuning_handler", _fake_run_gemm)

        await coord._on_enter_kernel(from_phase="EXPLORE")

        assert [c["task_id"] for c in calls] == [
            "kernel_entry_gemm_tuning_bf16_fallback"
        ]
        assert calls[0]["precision"] == "bf16"
        assert coord.shared_state.last_gemm_tuning["task_id"] == (
            "kernel_entry_gemm_tuning_bf16_fallback"
        )
        assert coord._bf16_dense_gemm_fallback_pending() is False

    @pytest.mark.asyncio
    async def test_bf16_fallback_failure_is_recorded_as_attempt(self, tmp_path):
        coord = _coord(tmp_path, framework="sglang")

        async def _raise(*_args, **_kwargs):
            raise RuntimeError("fallback boom")

        result = await coord._run_bf16_dense_gemm_fallback(_raise)

        assert result["status"] == "failed"
        assert result["decision"] == "REVERT"
        assert result["task_id"] == "kernel_entry_gemm_tuning_bf16_fallback"
        assert result["source"] == "fp8_no_improvement_bf16_fallback"
        assert result["precision"] == "bf16"
        assert coord._is_bf16_dense_gemm_fallback_attempt(result) is True


def _eligible_coord(tmp_path, monkeypatch, **overrides):
    """Coordinator wired for a CK-switch-eligible forge workload.

    forge + sglang + fp8 + gfx942 (mi300x) + block-scale fp8. The block-scale
    probe is forced to ``True`` unless overridden, so the helper does not depend
    on a real model config on disk.
    """
    kwargs = dict(
        baseline_tput=100.0,
        framework="sglang",
        precision="fp8",
        gpu_type="mi300x",
        model_path="/models/blockscale-fp8",
    )
    kwargs.update(overrides)
    coord = _coord(tmp_path, **kwargs)
    monkeypatch.setattr(mcu_mod, "_fp8_is_block_scale", lambda _p: True)
    return coord


class TestCkBlockscaleSwitchEligible:
    """``_ck_blockscale_switch_eligible`` gates the standalone CK backend switch
    to forge + sglang + fp8 (any signal) + gfx942 + block-scale checkpoints."""

    def test_eligible_for_forge_sglang_fp8_mi300x_blockscale(self, tmp_path, monkeypatch):
        coord = _eligible_coord(tmp_path, monkeypatch)
        assert coord._ck_blockscale_switch_eligible({"backend": "forge"}) is True

    def test_not_eligible_non_forge_backend(self, tmp_path, monkeypatch):
        coord = _eligible_coord(tmp_path, monkeypatch)
        assert coord._ck_blockscale_switch_eligible({"backend": "geak"}) is False

    def test_not_eligible_non_sglang(self, tmp_path, monkeypatch):
        coord = _eligible_coord(tmp_path, monkeypatch, framework="vllm")
        assert coord._ck_blockscale_switch_eligible({"backend": "forge"}) is False

    def test_not_eligible_non_fp8(self, tmp_path, monkeypatch):
        # Non-fp8 session precision and no runtime fp8 signal -> not eligible.
        coord = _eligible_coord(tmp_path, monkeypatch, precision="bf16")
        monkeypatch.setattr(
            krh_mod, "_resolve_forge_precision_and_quant", lambda _s, _p: ("bf16", "auto")
        )
        assert coord._ck_blockscale_switch_eligible({"backend": "forge"}) is False

    def test_not_eligible_non_gfx942_gpu(self, tmp_path, monkeypatch):
        # mi355x is a known AMD type but NOT in _GFX942_GPU_TYPES.
        coord = _eligible_coord(tmp_path, monkeypatch, gpu_type="mi355x")
        assert coord._ck_blockscale_switch_eligible({"backend": "forge"}) is False

    def test_not_eligible_non_block_scale_fp8(self, tmp_path, monkeypatch):
        # Per-tensor / static / per-channel-per-token fp8: no weight_block_size,
        # so the positive block-scale probe declines (Bug #1).
        coord = _eligible_coord(tmp_path, monkeypatch)
        monkeypatch.setattr(mcu_mod, "_fp8_is_block_scale", lambda _p: False)
        assert coord._ck_blockscale_switch_eligible({"backend": "forge"}) is False

    def test_eligible_for_runtime_fp8_via_result_precision(self, tmp_path, monkeypatch):
        # Session precision is bf16, but the forge result envelope stamps the
        # runtime-resolved precision fp8 (Bug #2, signal #2).
        coord = _eligible_coord(tmp_path, monkeypatch, precision="bf16")
        monkeypatch.setattr(
            krh_mod, "_resolve_forge_precision_and_quant", lambda _s, _p: ("bf16", "auto")
        )
        assert (
            coord._ck_blockscale_switch_eligible({"backend": "forge", "precision": "fp8"})
            is True
        )

    def test_eligible_for_runtime_fp8_via_quantization_arg(self, tmp_path, monkeypatch):
        # Session precision bf16 and no result precision, but runtime
        # --quantization fp8 is resolved from server args (Bug #2, signal #3).
        coord = _eligible_coord(tmp_path, monkeypatch, precision="bf16")
        monkeypatch.setattr(
            krh_mod, "_resolve_forge_precision_and_quant", lambda _s, _p: ("fp8", "auto")
        )
        assert coord._ck_blockscale_switch_eligible({"backend": "forge"}) is True

    def test_not_eligible_per_token_fp8(self, tmp_path, monkeypatch):
        # Per-channel/per-token fp8 carries no weight_block_size -> declined.
        coord = _eligible_coord(tmp_path, monkeypatch)
        monkeypatch.setattr(mcu_mod, "_fp8_is_block_scale", lambda _p: False)
        assert coord._ck_blockscale_switch_eligible({"backend": "forge"}) is False

    def test_non_dict_result_is_not_eligible(self, tmp_path, monkeypatch):
        coord = _eligible_coord(tmp_path, monkeypatch)
        assert coord._ck_blockscale_switch_eligible("nope") is False  # type: ignore[arg-type]


class TestPromoteInjectsCkBlockscaleEnv:
    """Inline-promote safety net: an eligible forge result that reaches
    ``_promote_gemm_tuning_keep`` (without the validator) injects
    ``SGLANG_FP8_BLOCKSCALE_CK_MAX_M=256`` (attributed to gemm_tuning). The
    primary forge path is the E2E validator; the GEAK path is no longer gated on
    the a8w8 table env and never injects the un-validated CK switch."""

    def test_injects_for_forge_eligible_keep(self, tmp_path, monkeypatch):
        coord = _eligible_coord(tmp_path, monkeypatch)
        coord._promote_gemm_tuning_keep(
            {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.2,
                "backend": "forge",
                "extra_envs": {"AITER_CONFIG": "/cfg/tuned.json"},
            }
        )
        envs = coord.shared_state.current_best["extra_envs"]
        assert envs["SGLANG_FP8_BLOCKSCALE_CK_MAX_M"] == "256"
        # Flows into the stacked entry too (shared extra_envs object).
        stack_envs = coord.shared_state.optimization_stack[0]["extra_envs"]
        assert stack_envs["SGLANG_FP8_BLOCKSCALE_CK_MAX_M"] == "256"

    def test_does_not_inject_for_geak_backend(self, tmp_path, monkeypatch):
        # GEAK is not forge: the helper rejects it, so the un-validated CK
        # switch is never stamped on the inline (non-E2E) promote path.
        coord = _eligible_coord(tmp_path, monkeypatch)
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
        assert "SGLANG_FP8_BLOCKSCALE_CK_MAX_M" not in envs

    def test_does_not_inject_for_bf16_precision(self, tmp_path, monkeypatch):
        coord = _eligible_coord(tmp_path, monkeypatch, precision="bf16")
        coord._promote_gemm_tuning_keep(
            {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.2,
                "backend": "forge",
                "extra_envs": {"AITER_CONFIG": "/cfg/tuned.json"},
            }
        )
        envs = coord.shared_state.current_best["extra_envs"]
        assert "SGLANG_FP8_BLOCKSCALE_CK_MAX_M" not in envs

    def test_does_not_inject_for_non_sglang_framework(self, tmp_path, monkeypatch):
        coord = _eligible_coord(tmp_path, monkeypatch, framework="vllm")
        coord._promote_gemm_tuning_keep(
            {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.2,
                "backend": "forge",
                "extra_envs": {"AITER_CONFIG": "/cfg/tuned.json"},
            }
        )
        envs = coord.shared_state.current_best["extra_envs"]
        assert "SGLANG_FP8_BLOCKSCALE_CK_MAX_M" not in envs

    def test_does_not_inject_for_non_gfx942_gpu(self, tmp_path, monkeypatch):
        coord = _eligible_coord(tmp_path, monkeypatch, gpu_type="mi355x")
        coord._promote_gemm_tuning_keep(
            {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.2,
                "backend": "forge",
                "extra_envs": {"AITER_CONFIG": "/cfg/tuned.json"},
            }
        )
        envs = coord.shared_state.current_best["extra_envs"]
        assert "SGLANG_FP8_BLOCKSCALE_CK_MAX_M" not in envs

    def test_does_not_inject_for_non_block_scale_fp8(self, tmp_path, monkeypatch):
        coord = _eligible_coord(tmp_path, monkeypatch)
        monkeypatch.setattr(mcu_mod, "_fp8_is_block_scale", lambda _p: False)
        coord._promote_gemm_tuning_keep(
            {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.2,
                "backend": "forge",
                "extra_envs": {"AITER_CONFIG": "/cfg/tuned.json"},
            }
        )
        envs = coord.shared_state.current_best["extra_envs"]
        assert "SGLANG_FP8_BLOCKSCALE_CK_MAX_M" not in envs

    def test_respects_preset_value_setdefault(self, tmp_path, monkeypatch):
        coord = _eligible_coord(tmp_path, monkeypatch)
        coord._promote_gemm_tuning_keep(
            {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.2,
                "backend": "forge",
                "extra_envs": {
                    "AITER_CONFIG": "/cfg/tuned.json",
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

        coord.phase_kernel._validate_forge_gemm_tuning_e2e = _fake_validate  # type: ignore[assignment]

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
    async def test_forge_no_improvement_but_ck_eligible_routes_to_validator(
        self, tmp_path, monkeypatch
    ):
        # a8w8 table tuner reported no_improvement (decision REVERT, no
        # requires_e2e_validation), but the CK block-scale switch is eligible →
        # must still route to the E2E validator, not inline promote.
        coord = _eligible_coord(tmp_path, monkeypatch)
        called: dict[str, object] = {}

        async def _fake_validate(result):
            called["result"] = result

        coord.phase_kernel._validate_forge_gemm_tuning_e2e = _fake_validate  # type: ignore[assignment]

        await coord._handle_gemm_tuning_result(
            {
                "status": "complete",
                "decision": "REVERT",
                "micro_decision": "no_improvement",
                "backend": "forge",
                "requires_e2e_validation": False,
            }
        )

        assert "result" in called
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
        # Each kept tuner also lands as a gemm_tuning KEEP journal event with
        # its serving throughput + per-tuner task_id.
        gj = [e for e in _journal_entries(tmp_path) if e.get("kind") == "gemm_tuning"]
        assert [e["throughput_after"] for e in gj] == pytest.approx([120.0, 132.0])
        assert {e["task_id"] for e in gj} == {
            "gemm_tune_e2e_fmoe_ck",
            "gemm_tune_e2e_dense_gemm",
        }
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
    async def test_injects_synthetic_ck_candidate_when_eligible_no_table_candidates(
        self, tmp_path, monkeypatch
    ):
        # No table candidates (all no_improvement), but CK switch is eligible:
        # the synthetic CK candidate is injected and E2E-validated, landing in
        # the optimization_stack attributed to gemm_tuning.
        coord = _eligible_coord(tmp_path, monkeypatch)
        fake = _make_integrate([{"decision": "KEEP", "new_tput": 209.0, "gain_pct": 109.0}])
        monkeypatch.setattr(krh_mod, "integrate_handler", fake)

        result = {
            "workspace": str(tmp_path),
            "backend": "forge",
            "recommended_env": {},
            "extra_envs": {},
            "tuners_run": [
                {
                    "status": "ok", "improved_shapes": 0, "tuner": "a8w8_blockscale",
                    "env_var": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE", "env_value": "/t.csv",
                },
            ],
        }
        await coord._validate_forge_gemm_tuning_e2e(result)

        # The synthetic candidate flows through the existing integrate loop with
        # the same payload shape as a table candidate.
        assert len(fake.calls) == 1
        assert fake.calls[0]["extra_envs"] == {"SGLANG_FP8_BLOCKSCALE_CK_MAX_M": "256"}
        assert fake.calls[0]["extra_server_args"] == ""

        stack = coord.shared_state.optimization_stack
        assert len(stack) == 1
        assert stack[0]["action"] == "gemm_tuning"
        assert stack[0]["variant_name"] == "forge_ck_blockscale_backend_switch"
        assert stack[0]["extra_envs"]["SGLANG_FP8_BLOCKSCALE_CK_MAX_M"] == "256"
        # Result rewritten to the E2E-validated KEEP outcome.
        assert result["e2e_validated"] is True
        assert result["decision"] == "KEEP"
        assert result["recommended_env"] == {"SGLANG_FP8_BLOCKSCALE_CK_MAX_M": "256"}

    @pytest.mark.asyncio
    async def test_no_synthetic_ck_candidate_when_not_eligible(self, tmp_path, monkeypatch):
        # Not eligible (vllm framework): no table candidates → early return,
        # no synthetic CK candidate, integrate never called.
        coord = _eligible_coord(tmp_path, monkeypatch, framework="vllm")
        fake = _make_integrate([{"decision": "KEEP", "new_tput": 209.0, "gain_pct": 109.0}])
        monkeypatch.setattr(krh_mod, "integrate_handler", fake)

        result = {
            "backend": "forge",
            "recommended_env": {},
            "extra_envs": {},
            "requires_e2e_validation": True,
            "tuners_run": [
                {
                    "status": "ok", "improved_shapes": 0, "tuner": "a8w8_blockscale",
                    "env_var": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE", "env_value": "/t.csv",
                },
            ],
        }
        await coord._validate_forge_gemm_tuning_e2e(result)

        assert fake.calls == []
        assert coord.shared_state.optimization_stack == []

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
