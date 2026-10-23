# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``forge-loop`` options that shape the loop: measurement, search strategy, lanes and what a KEEP may ship."""

from __future__ import annotations

from dataclasses import dataclass, replace

import click

from kernelforge.loop.search_tree import SearchStrategy
from kernelforge.option_groups import option_group

# The three specialist analyses are what a linear round's lanes are divided from.
LINEAR_DEFAULT_LANES = 3


@dataclass(frozen=True)
class LoopShape:
    """How the loop measures, how it searches, and how many candidates each round explores.

    ``lanes`` and ``merge_stacking`` are ``None`` until ``settle_loop_shape`` resolves them for the campaign's
    search strategy, so an explicit value can be told apart from the default.
    """

    nproc_per_node: int
    search_strategy: str | None
    lanes: int | None
    merge_stacking: bool | None
    bench_repeat: int
    commit_new_paths: tuple[str, ...]


def resolve_lanes(strategy: SearchStrategy, lanes: int | None) -> int:
    """The lanes per round a campaign searching with ``strategy`` runs; ``None`` takes the strategy's default."""
    if strategy is SearchStrategy.LINEAR:
        return LINEAR_DEFAULT_LANES if lanes is None else lanes
    if lanes not in (None, 1):
        raise click.ClickException(f"--search-strategy tree runs one lane per round; drop --lanes {lanes}")
    return 1


def settle_loop_shape(shape: LoopShape, strategy: SearchStrategy) -> LoopShape:
    """The loop shape a campaign searching with ``strategy`` runs, refusing what that strategy cannot honour."""
    if shape.search_strategy and SearchStrategy(shape.search_strategy) is not strategy:
        raise click.ClickException(
            f"this campaign searches with --search-strategy {strategy.value}; the strategy cannot change on resume"
        )
    lanes = resolve_lanes(strategy, shape.lanes)
    if strategy is SearchStrategy.LINEAR:
        return replace(shape, lanes=lanes, merge_stacking=shape.merge_stacking is not False)
    if shape.merge_stacking:
        raise click.ClickException("--search-strategy tree does not stack rejected candidates; drop --merge-stacking")
    return replace(shape, lanes=lanes, merge_stacking=False)


loop_shape_options = option_group(
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
        "--search-strategy",
        default=None,
        type=click.Choice([strategy.value for strategy in SearchStrategy]),
        help="Fresh campaign: how each iteration chooses the commit it starts "
        "from (stored immutably in the campaign config; on --resume it may "
        "only repeat the stored value). 'linear' (default) always builds on "
        "the current best. 'tree' keeps every KEEP as a node of a search "
        "tree on git branches: before each iteration a read-only start "
        "selector picks any node to build on, a candidate is kept when it "
        "beats the node it started from, and the published best is the "
        "highest-scoring node. Tree campaigns run one lane, no merge "
        "stacking and no Supervisor.",
    ),
    click.option(
        "--lanes",
        default=None,
        type=click.IntRange(min=1, max=8),
        help="Implementer lanes per round. Above 1 the round's analysis is "
        "partitioned into that many non-overlapping plans, each run "
        "concurrently in its own workspace copy, and each candidate is "
        "measured on its own. Default 3 for a linear campaign: the lanes "
        "of a round run concurrently, so a lane costs a session rather "
        "than a share of the round's wall clock, and three is what the "
        "three specialist analyses can be divided into. The partition "
        "returns fewer when the evidence supports fewer. Above 1 "
        "needs a provider that declares session_env and is refused on "
        "one that does not; a provider without stop_hooks runs and is "
        "warned, because it can waste a round but not misreport one. "
        "A tree campaign runs 1 and refuses any other value.",
    ),
    click.option(
        "--merge-stacking/--no-merge-stacking",
        default=None,
        help="Once consecutive iterations stop producing a new best, spend "
        "one iteration measuring two archived rejected gains applied "
        "together, chosen for winning on different cases. Costs a "
        "measurement but no Implementer session. Default on for a linear "
        "campaign; this applies at every --lanes setting, so turn it off "
        "to compare against a run that predates it. A tree campaign runs "
        "without it and refuses --merge-stacking.",
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
)


__all__ = ["LINEAR_DEFAULT_LANES", "LoopShape", "loop_shape_options", "resolve_lanes", "settle_loop_shape"]
