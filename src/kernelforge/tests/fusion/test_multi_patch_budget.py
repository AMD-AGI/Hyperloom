# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shared lane budget has to fund compile-pass claims before authoring.

``rank_recipes`` puts a ``compile_pass`` first because it is a deterministic
one-line flip that hands the work to a vendor-tuned kernel -- the cheapest
certain win in the round. A budget that funds the LLM authoring loop first
spends the whole lane ceiling on the expensive half and withholds every claim,
which is the ordering ``rank_recipes`` exists to prevent.

Both pipelines are faked: what is pinned is which targets each was handed, and
what the round reports about the targets it withheld.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import kernelforge.fusion.command as cli_module
from kernelforge.fusion.loop import LoopResult
from kernelforge.fusion.models import CompilePassOutcome, Recipe

#: Every real Hyperloom session caps the fusion lane at three targets
#: (``FUSION_MAX_TARGETS``), so this is the mainline ceiling, not an edge case.
LANE_CEILING = 3


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
    seen: dict[str, Any] = {"autoloop_ceiling": None, "autoloop_recipes": [], "claims": []}

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


def _nominate(tmp_path: Path, *, authored: list[str], claims: list[str], max_recipes: int) -> tuple[Any, ...]:
    return cli_module._run_multi_patch_nomination(
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


def test_a_full_authoring_slate_does_not_withhold_the_claim(tmp_path, spied):
    """Three authored recipes must not consume the ceiling ahead of a claim."""
    _nominate(tmp_path, authored=["a1", "a2", "a3"], claims=["c1"], max_recipes=LANE_CEILING)

    assert spied["claims"] == ["c1"]


def test_a_funded_claim_reports_its_verdict_on_the_compile_pass_slot(tmp_path, spied):
    """A null ``compile_pass`` reads as "no claim ran", so the claim must fill it."""
    outcome = _nominate(tmp_path, authored=["a1", "a2", "a3"], claims=["c1"], max_recipes=LANE_CEILING)[1]

    assert outcome is not None
    assert outcome.flag == "FLAG_C1"


def test_authored_recipes_take_what_the_claims_leave(tmp_path, spied):
    """The remainder still funds the autoloop, ranked order preserved."""
    _nominate(tmp_path, authored=["a1", "a2", "a3"], claims=["c1"], max_recipes=LANE_CEILING)

    assert spied["autoloop_recipes"] == ["a1", "a2"]
    assert spied["autoloop_ceiling"] == 2


def test_an_exhausted_ceiling_runs_no_authored_recipe(tmp_path, spied):
    """Zero remaining means the loop is skipped, never re-read as uncapped."""
    _nominate(tmp_path, authored=["a1", "a2"], claims=["c1", "c2", "c3"], max_recipes=LANE_CEILING)

    assert spied["claims"] == ["c1", "c2", "c3"]
    assert spied["autoloop_recipes"] == []
    assert spied["autoloop_ceiling"] is None


def test_the_round_reports_how_many_targets_it_withheld(tmp_path, spied):
    """Starvation is otherwise indistinguishable from "everything ran and failed"."""
    _patches, _outcome, _loop, withheld = _nominate(
        tmp_path, authored=["a1", "a2"], claims=["c1", "c2", "c3", "c4"], max_recipes=LANE_CEILING
    )

    assert withheld == 3


def test_an_uncapped_round_withholds_nothing(tmp_path, spied):
    """A non-positive ceiling means none was derived, never "run nothing"."""
    _patches, _outcome, _loop, withheld = _nominate(
        tmp_path, authored=["a1", "a2"], claims=["c1", "c2", "c3"], max_recipes=0
    )

    assert spied["claims"] == ["c1", "c2", "c3"]
    assert spied["autoloop_recipes"] == ["a1", "a2"]
    assert spied["autoloop_ceiling"] == 2
    assert withheld == 0
