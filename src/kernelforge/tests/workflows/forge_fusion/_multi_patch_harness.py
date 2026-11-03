# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Drive one multi-patch nomination with both fusion pipelines spied, so only the lane split runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import kernelforge.workflows.forge_fusion.pipeline as cli_module
from kernelforge.workflows.forge_fusion.loop import LoopResult
from kernelforge.workflows.forge_fusion.models import CompilePassOutcome, Recipe


def recipe(pattern_id: str) -> Recipe:
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


def spy_pipelines(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Record what each pipeline was asked to do, without running either."""
    seen: dict[str, Any] = {"autoloop_ceiling": None, "autoloop_recipes": [], "claims": []}

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


def nominate(tmp_path: Path, *, authored: list[str], claims: list[str], max_recipes: int) -> tuple[Any, ...]:
    return cli_module._run_multi_patch_nomination(
        cli_module._NominationSplit(
            claims=[recipe(name) for name in claims],
            deferred=[recipe(name) for name in authored],
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
