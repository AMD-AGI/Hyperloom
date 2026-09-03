# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One lane ceiling has to cover both multi-patch pipelines.

``_run_multi_patch_nomination`` runs authored recipes through the kernel autoloop
and compile-pass claims through their own serving A/B. Each costs a full
validation, so a ceiling that bounds only the autoloop lets the claims push the
round past the share that paid for it.

Both pipelines are faked: what is pinned is how many targets each was handed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import kernelforge.fusion.command as cli_module
from kernelforge.fusion.loop import LoopResult
from kernelforge.fusion.models import CompilePassOutcome, Recipe


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

    def _fake_autoloop(recipes, **kwargs):
        seen["autoloop_ceiling"] = kwargs.get("max_recipes")
        seen["autoloop_recipes"] = [r.pattern_id for r in recipes]
        return LoopResult(kept=False, best=None, best_recipe=None, termination_reason="faked")

    def _fake_claim(claim, **kwargs):
        seen["claims"].append(claim.pattern_id)
        return CompilePassOutcome(flag=claim.env_flag, kept=False), None

    monkeypatch.setattr(cli_module, "_run_fusion_autoloop", _fake_autoloop)
    monkeypatch.setattr(cli_module, "_run_single_compile_pass_claim", _fake_claim)
    return seen


def _nominate(tmp_path: Path, *, authored: list[str], claims: list[str], max_recipes: int) -> None:
    cli_module._run_multi_patch_nomination(
        claims=[_recipe(name) for name in claims],
        authored=[_recipe(name) for name in authored],
        framework="sglang",
        framework_root="/repo",
        out=tmp_path,
        repo_root="/repo",
        author=True,
        gpu="0",
        llm_model="m",
        target_speedup=1.03,
        model_path="/model",
        run_arch="gfx942",
        agent_backend="claude",
        agent_sandbox_mode="workspace-write",
        server_extra="",
        ab_isl=512,
        ab_osl=128,
        max_turns=10,
        max_recipes=max_recipes,
        pristine_dir="",
        tp=1,
        block_size=0,
        max_model_len=0,
        agent_factory=lambda *_a, **_k: None,
    )


def test_the_ceiling_is_shared_between_claims_and_authored_recipes(tmp_path, spied):
    """Two authored recipes spend two of three, leaving one claim funded."""
    _nominate(tmp_path, authored=["a1", "a2"], claims=["c1", "c2", "c3", "c4", "c5"], max_recipes=3)

    assert spied["autoloop_ceiling"] == 2
    assert spied["claims"] == ["c1"]


def test_authored_recipes_are_funded_before_claims(tmp_path, spied):
    """They run first, so the shared ceiling spends on them first."""
    _nominate(tmp_path, authored=["a1", "a2", "a3", "a4"], claims=["c1", "c2"], max_recipes=3)

    assert spied["autoloop_ceiling"] == 3
    assert spied["claims"] == []


def test_claims_take_the_whole_ceiling_when_nothing_was_authored(tmp_path, spied):
    _nominate(tmp_path, authored=[], claims=["c1", "c2", "c3", "c4"], max_recipes=3)

    assert spied["autoloop_ceiling"] is None, "the autoloop is not run without authored recipes"
    assert spied["claims"] == ["c1", "c2", "c3"]


def test_no_ceiling_withholds_neither_kind(tmp_path, spied):
    """Zero means none could be derived, never "run nothing".

    The autoloop is handed the discovered count, which withholds nothing, and
    every claim still runs.
    """
    _nominate(tmp_path, authored=["a1", "a2"], claims=["c1", "c2", "c3"], max_recipes=0)

    assert spied["autoloop_recipes"] == ["a1", "a2"]
    assert spied["autoloop_ceiling"] == 2
    assert spied["claims"] == ["c1", "c2", "c3"]


def test_a_ceiling_wider_than_both_kinds_withholds_nothing(tmp_path, spied):
    _nominate(tmp_path, authored=["a1"], claims=["c1"], max_recipes=9)

    assert spied["autoloop_ceiling"] == 1
    assert spied["claims"] == ["c1"]
