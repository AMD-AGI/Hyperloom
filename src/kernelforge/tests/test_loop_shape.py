# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""--search-strategy: the loop it runs, and the shapes it refuses."""

from __future__ import annotations

from dataclasses import replace

import click
import pytest

from kernelforge.cli.loop_shape import LoopShape, campaign_loop_inputs, loop_class_for
from kernelforge.loop.campaign_config import CampaignConfig
from kernelforge.loop.runner import IterationLoop
from kernelforge.loop.tree.search_loop import TreeSearchLoop

SHAPE = LoopShape(
    nproc_per_node=1, lanes=3, merge_stacking=True, bench_repeat=1, commit_new_paths=(), search_strategy=None
)


def test_a_fresh_campaign_searches_linearly_unless_asked():
    assert campaign_loop_inputs(SHAPE)["search_strategy"] == "linear"
    assert campaign_loop_inputs(replace(SHAPE, search_strategy="tree"))["search_strategy"] == "tree"


@pytest.mark.parametrize(("strategy", "loop_class"), [("linear", IterationLoop), ("tree", TreeSearchLoop)])
def test_the_campaign_strategy_picks_the_loop(strategy, loop_class):
    assert loop_class_for(SHAPE, CampaignConfig(search_strategy=strategy)) is loop_class


def test_a_tree_campaign_needs_two_lanes():
    with pytest.raises(click.UsageError, match="--lanes 2 or more"):
        loop_class_for(replace(SHAPE, lanes=1), CampaignConfig(search_strategy="tree"))


def test_a_resume_may_only_repeat_the_campaign_strategy():
    tree_campaign = CampaignConfig(search_strategy="tree")

    assert loop_class_for(replace(SHAPE, search_strategy="tree"), tree_campaign) is TreeSearchLoop
    with pytest.raises(click.UsageError, match="may only repeat it"):
        loop_class_for(replace(SHAPE, search_strategy="linear"), tree_campaign)
