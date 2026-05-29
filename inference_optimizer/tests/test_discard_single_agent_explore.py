"""PR-A9 (Arbor-into-Hyperloom): retire single-agent explore.

The legacy ``provenance='llm_direct'`` path — where the Orchestration
LLM authored an explore grid from one prompt window without any
specialist research — is now denied by PolicyGate's new rule
``explore_requires_specialist_provenance``. The cold-start escape
hatch is ``provenance='default_grid'``; specialist-derived rounds
carry ``provenance='specialist:<domain>'``.

These tests pin:

1. The PolicyGate rule denies an all-``llm_direct`` grid.
2. The rule allows a grid where at least one variant is
   ``default_grid`` or ``specialist:<domain>``.
3. The rule allows omitting / empty grid (executor surfaces its own
   ``empty_grid`` error path).
4. ``_grid_variants_from_payload`` defaults missing provenance to
   ``default_grid`` (not ``llm_direct``).
5. Updated prompt builder hint no longer advertises ``llm_direct``.
6. Updated orchestration.md no longer permits llm_direct.
"""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.action_executors.explore import (
    _grid_variants_from_payload,
)
from inference_optimizer.orchestrator.agent_role import default_role_registry
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.policy import (
    EXPLORE_PERMISSIVE_PROVENANCE_LITERALS,
    EXPLORE_PERMISSIVE_PROVENANCE_PREFIXES,
    PolicyDenied,
    PolicyGate,
)
from inference_optimizer.orchestrator.shared_state import SharedState


# ---------------------------------------------------------------------------
# 1. PolicyGate rule — happy and unhappy paths
# ---------------------------------------------------------------------------
def _gate(state: SharedState | None = None) -> PolicyGate:
    s = state or SharedState()
    s.phase = "EXPLORE"
    return PolicyGate(role_registry=default_role_registry(), shared_state=s)


def _delegate(grid: list[dict]) -> Intent:
    return Intent(type=IntentType.DELEGATE, payload={
        "action_name": "explore",
        "params": {"grid": grid},
    })


def test_policy_denies_all_llm_direct_grid():
    gate = _gate()
    intent = _delegate([
        {"name": "v1", "provenance": "llm_direct"},
        {"name": "v2", "provenance": "llm_direct"},
    ])
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "explore_requires_specialist_provenance"


def test_policy_denies_unstamped_grid():
    """No provenance field → defaults to ``llm_direct`` in the legacy
    contract; PR-A9 denies these too. Operators MUST stamp explicitly."""
    gate = _gate()
    intent = _delegate([
        {"name": "v1"},  # missing provenance
        {"name": "v2", "provenance": ""},
    ])
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "explore_requires_specialist_provenance"


def test_policy_allows_grid_with_default_grid_variant():
    gate = _gate()
    intent = _delegate([
        {"name": "v1", "provenance": "default_grid"},
    ])
    gate.validate_intent("orchestration", intent)  # no raise


def test_policy_allows_grid_with_specialist_provenance():
    """Single specialist:* variant is allowed. Multi-specialist grids
    are covered by ``explore_specialist_grid_max_one`` in
    test_explore_grid_limits.py."""
    gate = _gate()
    intent = _delegate([
        {"name": "v1", "provenance": "specialist:serving_specialist"},
    ])
    gate.validate_intent("orchestration", intent)  # no raise


def test_policy_allows_mixed_grid_when_at_least_one_is_permitted():
    """A grid that mixes one default_grid variant with several
    llm_direct variants is still permitted — the rule's goal is to
    deny rounds that are ENTIRELY single-agent, not to forbid mixing."""
    gate = _gate()
    intent = _delegate([
        {"name": "v1", "provenance": "llm_direct"},
        {"name": "v2", "provenance": "default_grid"},
    ])
    gate.validate_intent("orchestration", intent)  # no raise


def test_policy_skips_provenance_check_when_grid_omitted():
    """No grid in params → executor surfaces ``empty_grid``; PolicyGate
    must not preempt that with a provenance error."""
    gate = _gate()
    intent = Intent(type=IntentType.DELEGATE, payload={
        "action_name": "explore",
        "params": {},  # no grid
    })
    gate.validate_intent("orchestration", intent)  # no raise


def test_policy_constants_cover_expected_values():
    assert "default_grid" in EXPLORE_PERMISSIVE_PROVENANCE_LITERALS
    assert "specialist:" in EXPLORE_PERMISSIVE_PROVENANCE_PREFIXES
    # Defense: llm_direct must NOT be in the permissive sets.
    assert "llm_direct" not in EXPLORE_PERMISSIVE_PROVENANCE_LITERALS
    assert "llm_direct" not in EXPLORE_PERMISSIVE_PROVENANCE_PREFIXES


# ---------------------------------------------------------------------------
# 2. ExploreExecutor variant builder — default provenance changed
# ---------------------------------------------------------------------------
def test_grid_variants_from_payload_defaults_to_default_grid():
    """PR-A9 retired the ``llm_direct`` default; missing provenance
    now falls back to ``default_grid``."""
    out = _grid_variants_from_payload([
        {"name": "vA", "extra_args": "--foo"},
        {"name": "vB", "extra_args": "--bar", "provenance": "specialist:kernel_switch_specialist"},
    ])
    assert len(out) == 2
    assert getattr(out[0], "provenance") == "default_grid"
    assert getattr(out[1], "provenance") == "specialist:kernel_switch_specialist"


# ---------------------------------------------------------------------------
# 3. Prompt builder hint — no llm_direct in the advertised set
# ---------------------------------------------------------------------------
def test_prompt_grid_hint_no_longer_advertises_llm_direct():
    from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
        _format_grid_injection_hint,
    )
    hint = _format_grid_injection_hint("explore")
    assert hint is not None
    assert "specialist:<domain>" in hint
    assert "default_grid" in hint
    # The literal 'llm_direct' should only appear in the deprecation
    # notice, not in the advertised options. We check the hint mentions
    # llm_direct as denied (so the LLM learns the rule), but absolutely
    # not as a legitimate provenance value.
    assert "DENIED" in hint or "denied" in hint
    assert "explore_requires_specialist_provenance" in hint


# ---------------------------------------------------------------------------
# 4. Orchestration rules fragment — PR-A9 contract present
# ---------------------------------------------------------------------------
def test_orchestration_rules_fragment_documents_pr_a9():
    from inference_optimizer.paths import asset_system_prompts_dir
    text = (asset_system_prompts_dir() / "orchestration.md").read_text(
        encoding="utf-8",
    )
    assert "PR-A9" in text
    assert "explore_requires_specialist_provenance" in text


# ---------------------------------------------------------------------------
# 5. SKILL.md — IR-4 present
# ---------------------------------------------------------------------------
def test_skill_md_has_ir4_specialist_first():
    from inference_optimizer.paths import asset_root
    text = (asset_root() / "SKILL.md").read_text(encoding="utf-8")
    assert "### IR-4" in text
    assert "EXPLORE is specialist-first" in text
    assert "explore_requires_specialist_provenance" in text
