# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the recipe loop (no GPU, no forge-loop — fakes only)."""

from __future__ import annotations

from kernelforge.fusion.loop import (
    FusionExperienceLedger,
    LoopConfig,
    run_fusion_loop,
)
from kernelforge.fusion.models import Recipe, ValidationResult


def _recipe(pattern_id="residual_add_rmsnorm", env_flag="LFM2_FUSED_RESIDUAL", **over) -> Recipe:
    base = dict(
        pattern_id=pattern_id,
        description="Fold residual-add into RMSNorm.",
        env_flag=env_flag,
        source_file="/sgl/models/lfm2.py",
        source_hints=["+ residual"],
        fusion_math="y, residual = norm(x + residual)",
        eager_reference_hint="Import RMSNorm; compare.",
        shapes={"hidden_size": 2048, "T": 16},
        matched_categories=["rmsnorm"],
        trigger_share=0.3,
    )
    base.update(over)
    return Recipe(**base)


def _vr(*, correct=True, kept=False, speedup=None, note="", max_abs_err=None) -> ValidationResult:
    return ValidationResult(
        correctness_passed=correct,
        max_abs_err=max_abs_err,
        rtol=2e-2,
        kernel_speedup=speedup,
        eager_us=None,
        fused_us=None,
        kept=kept,
        note=note,
    )


class _ScriptedCampaign:
    """Returns a scripted result per recipe, recording the experience it saw."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = []  # list of (recipe.pattern_id, experience)

    def __call__(self, recipe, experience):
        result = self._results[min(len(self.calls), len(self._results) - 1)]
        self.calls.append((recipe.pattern_id, experience))
        return result


class TestEarlyExit:
    def test_stops_at_the_first_recipe_that_keeps(self, tmp_path):
        campaign = _ScriptedCampaign([_vr(kept=True, speedup=1.2, note="KEPT")])
        res = run_fusion_loop(
            [_recipe(), _recipe(pattern_id="swiglu", env_flag="LFM2_FUSED_SWIGLU")],
            framework="sglang",
            campaign_fn=campaign,
            config=LoopConfig(output_dir=str(tmp_path)),
        )
        assert res.kept is True
        assert res.best.kernel_speedup == 1.2
        assert res.best_recipe.pattern_id == "residual_add_rmsnorm"
        assert res.termination_reason == "kept"
        assert len(res.history) == 1
        assert len(campaign.calls) == 1


class TestExperienceInjection:
    def test_a_failed_recipe_teaches_the_next_one(self, tmp_path):
        campaign = _ScriptedCampaign(
            [
                _vr(correct=False, note="PARITY FAILED: min SNR=12 dB | LESSON: accumulate in fp32", max_abs_err=0.5),
                _vr(kept=True, speedup=1.1, note="KEPT"),
            ]
        )
        res = run_fusion_loop(
            [_recipe(), _recipe(pattern_id="swiglu", env_flag="LFM2_FUSED_SWIGLU")],
            framework="sglang",
            campaign_fn=campaign,
            config=LoopConfig(output_dir=str(tmp_path)),
        )
        assert res.kept is True
        assert campaign.calls[0][1] == ""
        second = campaign.calls[1][1]
        assert "Known constraints" in second
        assert "recipe 1" in second

    def test_compile_failure_distills_the_cuda_constraint(self, tmp_path):
        campaign = _ScriptedCampaign(
            [
                _vr(correct=False, note="COMPILE FAILED (module import): cuda_bf16.h | LESSON: author Triton"),
                _vr(kept=True, speedup=1.05, note="KEPT"),
            ]
        )
        run_fusion_loop(
            [_recipe(), _recipe(pattern_id="swiglu", env_flag="F2")],
            framework="sglang",
            campaign_fn=campaign,
            config=LoopConfig(output_dir=str(tmp_path)),
        )
        assert "CUDA-only" in campaign.calls[1][1]


class TestBounds:
    def test_outer_bound_limits_recipes(self, tmp_path):
        campaign = _ScriptedCampaign([_vr(correct=False, note="PARITY FAILED")])
        recipes = [_recipe(pattern_id=f"p{i}", env_flag=f"F{i}") for i in range(5)]
        res = run_fusion_loop(
            recipes,
            framework="sglang",
            campaign_fn=campaign,
            config=LoopConfig(max_recipes=2, output_dir=str(tmp_path)),
        )
        assert {h.recipe_index for h in res.history} == {0, 1}
        assert len(res.history) == 2

    def test_one_campaign_per_recipe(self, tmp_path):
        campaign = _ScriptedCampaign([_vr(correct=True, kept=False, speedup=1.0)])
        res = run_fusion_loop(
            [_recipe(), _recipe(pattern_id="swiglu", env_flag="F2")],
            framework="sglang",
            campaign_fn=campaign,
            config=LoopConfig(output_dir=str(tmp_path)),
        )
        # Repeated authoring belongs to the campaign, so the loop never retries.
        assert len(campaign.calls) == 2
        assert len(res.history) == 2

    def test_already_satisfied_recipe_skipped(self, tmp_path):
        campaign = _ScriptedCampaign([_vr(kept=True, speedup=1.2)])
        res = run_fusion_loop(
            [_recipe(already_satisfied=True)],
            framework="sglang",
            campaign_fn=campaign,
            config=LoopConfig(output_dir=str(tmp_path)),
        )
        assert res.kept is False
        assert res.history == []
        assert campaign.calls == []


class TestRobustnessAndOutputs:
    def test_campaign_exception_costs_one_recipe(self, tmp_path):
        def boom(recipe, experience):
            raise RuntimeError("campaign crashed")

        res = run_fusion_loop(
            [_recipe(), _recipe(pattern_id="swiglu", env_flag="F2")],
            framework="sglang",
            campaign_fn=boom,
            config=LoopConfig(output_dir=str(tmp_path)),
        )
        assert res.kept is False
        assert len(res.history) == 2
        assert all("campaign crashed" in h.note for h in res.history)

    def test_best_near_miss_reported_when_nothing_kept(self, tmp_path):
        campaign = _ScriptedCampaign(
            [
                _vr(correct=True, kept=False, speedup=1.01),
                _vr(correct=True, kept=False, speedup=1.02),
            ]
        )
        res = run_fusion_loop(
            [_recipe(), _recipe(pattern_id="swiglu", env_flag="F2")],
            framework="sglang",
            campaign_fn=campaign,
            config=LoopConfig(output_dir=str(tmp_path)),
        )
        assert res.kept is False
        assert res.best is not None
        assert res.best.kernel_speedup == 1.02

    def test_ledger_persisted_to_output_dir(self, tmp_path):
        campaign = _ScriptedCampaign([_vr(correct=False, note="PARITY FAILED: snr low")])
        run_fusion_loop(
            [_recipe()],
            framework="sglang",
            campaign_fn=campaign,
            config=LoopConfig(output_dir=str(tmp_path)),
        )
        ledger = tmp_path / "fusion_experience.md"
        assert ledger.is_file()
        assert "experience ledger" in ledger.read_text().lower()

    def test_result_to_dict_shape(self, tmp_path):
        campaign = _ScriptedCampaign([_vr(kept=True, speedup=1.2)])
        res = run_fusion_loop(
            [_recipe()],
            framework="sglang",
            campaign_fn=campaign,
            config=LoopConfig(output_dir=str(tmp_path)),
        )
        d = res.to_dict()
        assert d["kept"] is True
        assert d["best_pattern"] == "residual_add_rmsnorm"
        assert isinstance(d["history"], list) and d["history"][0]["kept"] is True


class TestLedgerUnit:
    def test_no_output_dir_does_not_write(self):
        ledger = FusionExperienceLedger(None)
        ledger.record(label="r1", outcome="PARITY FAILED", error_text="snr low")
        assert ledger.path is None
        assert "Recent attempts" in ledger.render_for_prompt()

    def test_constraints_deduped(self):
        ledger = FusionExperienceLedger(None)
        for _ in range(3):
            ledger.record(label="x", outcome="COMPILE FAILED", error_text="cuda_bf16.h missing")
        rendered = ledger.render_for_prompt()
        assert rendered.count("CUDA-only") == 1
