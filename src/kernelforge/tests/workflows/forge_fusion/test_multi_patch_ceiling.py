# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One lane ceiling has to cover both multi-patch pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import kernelforge.workflows.forge_fusion.pipeline as cli_module
from kernelforge.workflows.forge_fusion.loop import LoopResult
from kernelforge.workflows.forge_fusion.models import CompilePassOutcome, Recipe


def _recipe(pattern_id: str) -> Recipe:
    return Recipe(
        pattern_id=pattern_id,
        description=pattern_id,
        env_flag=f"FLAG_{pattern_id.upper()}",
        source_file=f"/repo/{pattern_id}.py",
        source_hints=[],
        fusion_math="",
        eager_reference_hint="",
        shapes=[],
        matched_categories=[],
        trigger_share=0.0,
    )


@pytest.fixture
def spied(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Record what each pipeline was asked to do, without running either."""
    seen: dict[str, Any] = {"autoloop_ceiling": None, "claims": []}

    def _fake_autoloop(recipes, run, plan, campaign, publish=None):
        seen["autoloop_ceiling"] = plan.max_recipes
        seen["autoloop_recipes"] = [r.pattern_id for r in recipes]
        return LoopResult(kept=False, best=None, best_recipe=None, termination_reason="faked")

    def _fake_claim(claim, *_args):
        seen["claims"].append(claim.pattern_id)
        return CompilePassOutcome(flag=claim.env_flag, kept=False), None

    monkeypatch.setattr(cli_module, "_run_fusion_autoloop", _fake_autoloop)
    monkeypatch.setattr(cli_module, "_run_single_compile_pass_claim", _fake_claim)
    return seen


def _nominate(tmp_path: Path, *, authored: list[str], claims: list[str], max_recipes: int) -> None:
    cli_module._run_multi_patch_nomination(
        cli_module._NominationSplit(
            claims=[_recipe(name) for name in claims],
            deferred=[_recipe(name) for name in authored],
            top_recipe=None,
            fuse_all_confirmed=False,
            multi_patch=True,
        ),
        cli_module._FusionRun(
            out=tmp_path,
            repo_root="/repo",
            pristine_dir="",
            framework_root="/repo",
            repo_scope=False,
            session=cli_module.ServingSession("sglang", "/model", "0", 512, 128),
            author=cli_module.AuthorAgent(backend=lambda *_a, **_k: None, gpu="0", model="m", max_turns=10),
            gpu_arch="gfx942",
            target_speedup=1.03,
        ),
        cli_module._LoopPlan(author=True, max_recipes=max_recipes),
        cli_module.CampaignRuntime(
            gpu="0", gpu_target="gfx942", model="m", agent_backend="claude", agent_sandbox_mode="workspace-write"
        ),
    )


def test_the_ceiling_is_shared_between_claims_and_authored_recipes(tmp_path, spied):
    """Five claims spend the whole ceiling, so the autoloop is not run at all."""
    _nominate(tmp_path, authored=["a1", "a2"], claims=["c1", "c2", "c3", "c4", "c5"], max_recipes=3)

    assert spied["claims"] == ["c1", "c2", "c3"]
    assert spied["autoloop_ceiling"] is None


def test_claims_take_the_whole_ceiling_when_nothing_was_authored(tmp_path, spied):
    _nominate(tmp_path, authored=[], claims=["c1", "c2", "c3", "c4"], max_recipes=3)

    assert spied["autoloop_ceiling"] is None, "the autoloop is not run without authored recipes"
    assert spied["claims"] == ["c1", "c2", "c3"]


def test_no_ceiling_withholds_neither_kind(tmp_path, spied):
    """Zero means none could be derived, never \"run nothing\"."""
    _nominate(tmp_path, authored=["a1", "a2"], claims=["c1", "c2", "c3"], max_recipes=0)

    assert spied["autoloop_recipes"] == ["a1", "a2"]
    assert spied["autoloop_ceiling"] == 2
    assert spied["claims"] == ["c1", "c2", "c3"]


def test_a_ceiling_wider_than_both_kinds_withholds_nothing(tmp_path, spied):
    _nominate(tmp_path, authored=["a1"], claims=["c1"], max_recipes=9)

    assert spied["autoloop_ceiling"] == 1
    assert spied["claims"] == ["c1"]
