# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which iteration loop a campaign's search strategy runs, and the loop shapes and relay modes a strategy refuses."""

from __future__ import annotations

from kernelforge.workflows.forge_loop.bootstrap.errors import ForgeLoopUsageError
from kernelforge.workflows.forge_loop.bootstrap.request import LoopShape
from kernelforge.workflows.forge_loop.campaign.campaign_config import CampaignConfig, RelayMode, SearchStrategy


def require_lanes_for(strategy: SearchStrategy, lanes: int) -> None:
    """Refuse a lane count the search strategy cannot run with."""
    if strategy is SearchStrategy.TREE and lanes < 2:
        raise ForgeLoopUsageError("--search-strategy tree expands several nodes each round and needs --lanes 2 or more")


def require_relay_mode_for(strategy: SearchStrategy, mode: RelayMode) -> None:
    """Refuse a relay mode the search strategy cannot run with."""
    if mode is RelayMode.SPLIT and strategy is not SearchStrategy.TREE:
        raise ForgeLoopUsageError(
            "--relay-mode split runs the slots of a tree round on several Agent profiles at once and needs "
            "--search-strategy tree"
        )


def requested_strategy(shape: LoopShape) -> SearchStrategy:
    """The strategy a fresh campaign is created with: the one this invocation named, else linear."""
    return SearchStrategy(shape.search_strategy or SearchStrategy.LINEAR.value)


def requested_relay_mode(shape: LoopShape) -> RelayMode:
    """The relay mode a fresh campaign is created with: the one this invocation named, else handoff."""
    return RelayMode(shape.relay_mode or RelayMode.HANDOFF.value)


def loop_class_for(shape: LoopShape, campaign: CampaignConfig) -> type:
    """The iteration loop the campaign's search strategy runs, refusing a shape the campaign cannot run with."""
    strategy = SearchStrategy(campaign.search_strategy)
    if shape.search_strategy is not None and SearchStrategy(shape.search_strategy) is not strategy:
        raise ForgeLoopUsageError(
            f"this campaign searches with --search-strategy {strategy.value}; a resume may only repeat it"
        )
    mode = RelayMode(campaign.relay_mode)
    if shape.relay_mode is not None and RelayMode(shape.relay_mode) is not mode:
        raise ForgeLoopUsageError(f"this campaign relays with --relay-mode {mode.value}; a resume may only repeat it")
    require_lanes_for(strategy, shape.lanes)
    require_relay_mode_for(strategy, mode)
    if strategy is SearchStrategy.LINEAR:
        from kernelforge.workflows.forge_loop.iteration.runner import IterationLoop

        return IterationLoop
    from kernelforge.workflows.forge_loop.tree.search_loop import TreeSearchLoop

    return TreeSearchLoop
