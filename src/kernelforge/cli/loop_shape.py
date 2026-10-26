# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How a forge-loop campaign searches and measures: its loop-shape options, and the loop its strategy runs."""

from __future__ import annotations

from dataclasses import dataclass

import click

from kernelforge.loop.campaign_config import CampaignConfig, SearchStrategy
from kernelforge.option_groups import option_group


@dataclass(frozen=True)
class LoopShape:
    """How the loop measures, how many candidates each round explores, and how it chooses what they build on."""

    nproc_per_node: int
    lanes: int
    merge_stacking: bool
    bench_repeat: int
    commit_new_paths: tuple[str, ...]
    # None when the invocation did not say: a fresh campaign then searches linearly, a resumed one as it always has.
    search_strategy: str | None


def campaign_loop_inputs(shape: LoopShape) -> dict[str, object]:
    """The loop-shape fields a fresh campaign snapshots into its immutable configuration."""
    return {
        "nproc_per_node": shape.nproc_per_node,
        "bench_repeat": shape.bench_repeat,
        "commit_new_paths": tuple(shape.commit_new_paths),
        "search_strategy": shape.search_strategy or SearchStrategy.LINEAR.value,
    }


def loop_class_for(shape: LoopShape, campaign: CampaignConfig) -> type:
    """The iteration loop the campaign's search strategy runs, refusing a shape that strategy cannot run."""
    strategy = SearchStrategy(campaign.search_strategy)
    if shape.search_strategy is not None and SearchStrategy(shape.search_strategy) is not strategy:
        raise click.UsageError(
            f"this campaign searches with --search-strategy {strategy.value}; a resume may only repeat it"
        )
    if strategy is SearchStrategy.LINEAR:
        from kernelforge.loop.runner import IterationLoop

        return IterationLoop
    if shape.lanes < 2:
        raise click.UsageError("--search-strategy tree expands several nodes each round and needs --lanes 2 or more")
    from kernelforge.loop.tree.search_loop import TreeSearchLoop

    return TreeSearchLoop


LOOP_SHAPE_OPTIONS = option_group(
    "loop_shape",
    LoopShape,
    click.option(
        "--nproc-per-node",
        default=1,
        type=click.IntRange(min=1),
        help="Ranks the driver self-launches via torchrun (collective tasks "
        "such as all-reduce). >1 profiles EVERY rank in its own "
        "rocprofv3 session, because wrapping the driver would only "
        "profile the launcher process, which runs no kernel. Default 1 "
        "(single-GPU, unchanged behavior).",
    ),
    click.option(
        "--lanes",
        default=3,
        type=click.IntRange(min=1, max=8),
        help="Implementer lanes per round. Above 1 the round's analysis is "
        "partitioned into that many non-overlapping plans, each run "
        "concurrently in its own workspace copy, and each candidate is "
        "measured on its own. Default 3: the lanes of a round run "
        "concurrently, so a lane costs a session rather than a share "
        "of the round's wall clock, and three is what the three "
        "specialist analyses can be divided into. The partition "
        "returns fewer when the evidence supports fewer. Above 1 "
        "needs a provider that declares session_env and is refused on "
        "one that does not; a provider without stop_hooks runs and is "
        "warned, because it can waste a round but not misreport one. "
        "Under --search-strategy tree it is how many nodes a round may expand, and must be 2 or more.",
    ),
    click.option(
        "--merge-stacking/--no-merge-stacking",
        default=True,
        help="Once consecutive iterations stop producing a new best, spend "
        "one iteration measuring two archived rejected gains applied "
        "together, chosen for winning on different cases. Costs a "
        "measurement but no Implementer session. Default on; this "
        "applies at every --lanes setting, so turn it off to compare "
        "against a run that predates it. A tree campaign stacks its own nodes instead.",
    ),
    click.option(
        "--bench-repeat",
        default=1,
        type=click.IntRange(min=1),
        help="How many times each bench repeats its measurement in-process, "
        "reporting the per-case median. Default 1 (single shot). >1 "
        "shrinks run-to-run spread. Requires a driver that accepts "
        "--repeat; the flag is omitted entirely when this is 1.",
    ),
    click.option(
        "--commit-new-path",
        "commit_new_paths",
        multiple=True,
        help="Workspace-relative path or glob naming a file the agent may "
        "CREATE and still have committed with a KEEP (repeatable, e.g. "
        "--commit-new-path configs/*.json). Untracked files are "
        "otherwise never staged and never removed by a REVERT. A '*' "
        "does not cross a directory separator and '**' is rejected; "
        "name each level instead. Protected measurement paths are "
        "never admitted however they are spelled. Immutable per "
        "campaign: it is snapshotted into campaign_config.json and "
        "read back on --resume.",
    ),
    click.option(
        "--search-strategy",
        type=click.Choice([item.value for item in SearchStrategy]),
        default=None,
        help="How each iteration chooses what it builds on. 'linear' (the default for a fresh campaign) builds every "
        "round on the current best. 'tree' keeps every measured correct candidate as a node of a search tree, and "
        "each round a read-only selector chooses up to --lanes nodes to expand -- continuing a route, opening a new "
        "one, or carrying gains from one node onto another -- each planned and implemented in its own checkout of "
        "its node; the round's best node that clears the KEEP bar is published. Immutable per campaign: "
        "snapshotted into campaign_config.json, and a --resume may only repeat it.",
    ),
)


__all__ = ["LOOP_SHAPE_OPTIONS", "LoopShape", "campaign_loop_inputs", "loop_class_for"]
