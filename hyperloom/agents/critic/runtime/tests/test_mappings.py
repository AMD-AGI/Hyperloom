"""Tests for :mod:`runtime.importance_mapping` and
:mod:`runtime.category_mapping` — the small enum/score lookup tables
that gate Critic verdicts and KB-draft kinds.
"""

from __future__ import annotations

import pytest

from hyperloom.agents.critic.runtime.category_mapping import (
    CATEGORY_TO_KIND,
    KB_KINDS,
    filter_supported_categories,
    map_category_to_kind,
)
from hyperloom.agents.critic.runtime.errors import RuntimeAdapterError
from hyperloom.agents.critic.runtime.importance_mapping import (
    CRITIC_IMPORTANCE_CEILING,
    cap_importance,
    importance_for_kb_draft,
    importance_for_verdict,
)


# ---------------------------------------------------------------------------
# runtime.importance_mapping
# ---------------------------------------------------------------------------


def test_high_with_measurement_scores_above_default():
    score = importance_for_verdict(
        verdict="reject", confidence="high", has_measurement=True
    )
    assert score == 0.7


def test_high_without_measurement_drops_back_to_low():
    score = importance_for_verdict(
        verdict="reject", confidence="high", has_measurement=False
    )
    assert score == 0.4


def test_advise_clamped_low_regardless_of_confidence():
    assert importance_for_verdict(verdict="advise", confidence="high") == 0.4


def test_kb_draft_high_confidence_promoted():
    assert importance_for_kb_draft(confidence=0.85) == 0.6


def test_kb_draft_default_when_unknown():
    assert importance_for_kb_draft(confidence=None) == 0.5


def test_cap_importance_enforces_critic_ceiling():
    assert cap_importance(0.99) == CRITIC_IMPORTANCE_CEILING
    assert cap_importance(0.5) == 0.5
    assert cap_importance(-1.0) == 0.0


# ---------------------------------------------------------------------------
# runtime.category_mapping
# ---------------------------------------------------------------------------


def test_kb_kinds_set_matches_contract():
    assert KB_KINDS == frozenset({"pitfall", "technique", "params_catalog", "model_profile"})


def test_kernel_opt_maps_to_technique():
    assert map_category_to_kind("kernel_optimization") == "technique"


def test_pitfall_categories_collapse_correctly():
    assert map_category_to_kind("crash_recovery") == "pitfall"
    assert map_category_to_kind("benchmark_methodology") == "pitfall"
    assert map_category_to_kind("architecture_constraint") == "pitfall"


def test_unknown_category_raises():
    with pytest.raises(RuntimeAdapterError):
        map_category_to_kind("not_a_thing")


def test_filter_supported_partitions_lists():
    supported, rejected = filter_supported_categories(
        ["kernel_optimization", "definitely_not_a_thing", "server_params"]
    )
    assert supported == ["kernel_optimization", "server_params"]
    assert rejected == ["definitely_not_a_thing"]


def test_full_table_targets_known_kinds():
    for category, kind in CATEGORY_TO_KIND.items():
        assert kind in KB_KINDS, (category, kind)
