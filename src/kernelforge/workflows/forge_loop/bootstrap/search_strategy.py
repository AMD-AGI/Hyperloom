# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which iteration loop a campaign's search strategy runs, and the loop shapes a strategy refuses."""

from __future__ import annotations

from kernelforge.workflows.forge_loop.bootstrap.errors import ForgeLoopUsageError
from kernelforge.workflows.forge_loop.bootstrap.request import LoopShape
from kernelforge.workflows.forge_loop.campaign.campaign_config import CampaignConfig, SearchStrategy


def require_lanes_for(strategy: SearchStrategy, lanes: int) -> None:
    """Refuse a lane count the search strategy cannot run with."""
    if strategy is SearchStrategy.TREE and lanes < 2:
        raise ForgeLoopUsageError("--search-strategy tree expands several nodes each round and needs --lanes 2 or more")


def requested_strategy(shape: LoopShape) -> SearchStrategy:
    """The strategy a fresh campaign is created with: the one this invocation named, else linear."""
    return SearchStrategy(shape.search_strategy or SearchStrategy.LINEAR.value)


def loop_class_for(shape: LoopShape, campaign: CampaignConfig) -> type:
    """The iteration loop the campaign's search strategy runs, refusing a shape that strategy cannot run with."""
    strategy = SearchStrategy(campaign.search_strategy)
    if shape.search_strategy is not None and SearchStrategy(shape.search_strategy) is not strategy:
        raise ForgeLoopUsageError(
            f"this campaign searches with --search-strategy {strategy.value}; a resume may only repeat it"
        )
    require_lanes_for(strategy, shape.lanes)
    if strategy is SearchStrategy.LINEAR:
        from kernelforge.workflows.forge_loop.iteration.runner import IterationLoop

        return IterationLoop
    from kernelforge.workflows.forge_loop.tree.search_loop import TreeSearchLoop

    return TreeSearchLoop
