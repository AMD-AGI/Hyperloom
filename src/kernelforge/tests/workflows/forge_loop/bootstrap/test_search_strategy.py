# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""--search-strategy and --relay-mode: the loop a campaign runs, and the shapes it refuses."""

from __future__ import annotations

from dataclasses import replace

import pytest

from kernelforge.workflows.forge_loop.bootstrap import run as run_module
from kernelforge.workflows.forge_loop.bootstrap.errors import ForgeLoopUsageError
from kernelforge.workflows.forge_loop.bootstrap.request import LoopShape
from kernelforge.workflows.forge_loop.bootstrap.search_strategy import (
    loop_class_for,
    require_relay_mode_for,
    requested_relay_mode,
    requested_strategy,
)
from kernelforge.workflows.forge_loop.campaign.campaign_config import CampaignConfig, RelayMode, SearchStrategy
from kernelforge.workflows.forge_loop.iteration.runner import IterationLoop
from kernelforge.workflows.forge_loop.search.agent_relay import AgentProfile
from kernelforge.workflows.forge_loop.search.search_actors import SearchActorBundle
from kernelforge.workflows.forge_loop.tree.search_loop import TreeSearchLoop

SHAPE = LoopShape(nproc_per_node=1, lanes=3, merge_stacking=True, bench_repeat=1, commit_new_paths=())


def test_a_fresh_campaign_searches_linearly_unless_asked():
    assert requested_strategy(SHAPE) is SearchStrategy.LINEAR
    assert requested_strategy(replace(SHAPE, search_strategy="tree")) is SearchStrategy.TREE


@pytest.mark.parametrize(("strategy", "loop_class"), [("linear", IterationLoop), ("tree", TreeSearchLoop)])
def test_the_campaign_strategy_picks_the_loop(strategy, loop_class):
    assert loop_class_for(SHAPE, CampaignConfig(search_strategy=strategy)) is loop_class


def test_a_tree_campaign_needs_two_lanes():
    with pytest.raises(ForgeLoopUsageError, match="--lanes 2 or more"):
        loop_class_for(replace(SHAPE, lanes=1), CampaignConfig(search_strategy="tree"))


def test_a_resume_may_only_repeat_the_campaign_strategy():
    tree_campaign = CampaignConfig(search_strategy="tree")

    assert loop_class_for(replace(SHAPE, search_strategy="tree"), tree_campaign) is TreeSearchLoop
    with pytest.raises(ForgeLoopUsageError, match="may only repeat it"):
        loop_class_for(replace(SHAPE, search_strategy="linear"), tree_campaign)


def test_a_tree_search_on_one_lane_is_refused_before_anything_is_set_up():
    with pytest.raises(ForgeLoopUsageError, match="--lanes 2 or more"):
        run_module._reject_incompatible_loop_options(
            producer="", loop_shape=replace(SHAPE, lanes=1, search_strategy="tree"), resume=False
        )
    run_module._reject_incompatible_loop_options(producer="", loop_shape=replace(SHAPE, lanes=1), resume=False)


def test_a_fresh_campaign_relays_by_handoff_unless_asked():
    assert requested_relay_mode(SHAPE) is RelayMode.HANDOFF
    assert requested_relay_mode(replace(SHAPE, relay_mode="split")) is RelayMode.SPLIT
    assert requested_relay_mode(replace(SHAPE, relay_mode="off")) is RelayMode.OFF


def test_either_search_strategy_may_run_with_its_relay_off():
    require_relay_mode_for(SearchStrategy.LINEAR, RelayMode.OFF)
    require_relay_mode_for(SearchStrategy.TREE, RelayMode.OFF)
    off_tree = CampaignConfig(search_strategy="tree", relay_mode="off")

    assert loop_class_for(replace(SHAPE, search_strategy="tree", relay_mode="off"), off_tree) is TreeSearchLoop
    with pytest.raises(ForgeLoopUsageError, match="relays with --relay-mode off"):
        loop_class_for(replace(SHAPE, relay_mode="handoff"), off_tree)


def test_a_split_relay_needs_a_tree_search():
    with pytest.raises(ForgeLoopUsageError, match="--search-strategy tree"):
        require_relay_mode_for(SearchStrategy.LINEAR, RelayMode.SPLIT)
    require_relay_mode_for(SearchStrategy.TREE, RelayMode.SPLIT)
    require_relay_mode_for(SearchStrategy.LINEAR, RelayMode.HANDOFF)


def test_a_resume_may_only_repeat_the_campaign_relay_mode():
    split_campaign = CampaignConfig(search_strategy="tree", relay_mode="split")

    assert loop_class_for(replace(SHAPE, relay_mode="split"), split_campaign) is TreeSearchLoop
    assert loop_class_for(SHAPE, split_campaign) is TreeSearchLoop
    with pytest.raises(ForgeLoopUsageError, match="--relay-mode split; a resume may only repeat it"):
        loop_class_for(replace(SHAPE, relay_mode="handoff"), split_campaign)


def test_a_split_relay_of_a_fresh_linear_search_is_refused_before_anything_is_set_up():
    with pytest.raises(ForgeLoopUsageError, match="--search-strategy tree"):
        run_module._reject_incompatible_loop_options(
            producer="", loop_shape=replace(SHAPE, relay_mode="split"), resume=False
        )
    # A resume that names no strategy is checked once the strategy its campaign was created with is read.
    run_module._reject_incompatible_loop_options(
        producer="", loop_shape=replace(SHAPE, relay_mode="split"), resume=True
    )


def test_a_campaign_without_relay_plans_and_selects_with_its_start_actor():
    """Without relay no actor bundle reaches the iterations, so the loop's own agents must carry the tree's."""
    planning, selector = object(), object()
    bundle = SearchActorBundle(
        AgentProfile("claude", "claude", "claude-opus-5"),
        agent_fn=None,
        orchestration_factory=planning,
        slot_selector=selector,
    )

    agents = run_module._loop_agents(bundle, lanes=2, analysis_service=None, search_actors=None)

    assert agents.orchestration_factory is planning
    assert agents.slot_selector is selector
