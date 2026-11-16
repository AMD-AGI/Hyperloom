# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Each component agent's share of the session's directions: a given map's split before EXPLORE, a made one's in it.

The user names one directions map with ``--agent-directions``: the open directions of a component agent's dossier map,
or a list written by hand, in the form its ``component.plan`` reads. Two component agents given the whole map would
work the same directions. Split, each works its own. Before the agent branches open, each agent source is fetched
once, and the map is split by the agent's adapter (``split_directions``,
:mod:`~hyperloom_multibranch.specialist.component_agent`) into as many disjoint shares as there are component agents,
one each, in the order the user named them. Any other agent gets none: only a component agent reads a plan.

Without a map, two or more component agents split the map of the dossier pass
(:mod:`~hyperloom_multibranch.specialist.dossier_pass`) a mapping branch runs in EXPLORE
(:mod:`~hyperloom_multibranch.specialist.map_session`), which the session passes here as the map once the pass ends and
opens a share branch for each share. The dossier pass is the default, and it runs on a branch, so no other branch waits
for it. When the pass gives none, they split one the first generates from the root's shares, PRELUDE's profile, with
no GPU time (its adapter's ``generate_directions``): one direction per op family, ranked by its share. Those shares are
the root's op shares (:mod:`~hyperloom_multibranch.scheduling.root_shares`), kernel names where the toolbox ranked
kernels, and ``shares_source`` says where they came from. A single component agent gets none and maps its own
directions, which its dossier measures better.

A map that cannot be split, or generated, leaves every branch without a share, as without a map, and the log says why:
a source that cannot be fetched, no component agent among the agents, fewer independent directions than component
agents, or a root with no shares a component agent can use.

A session that stops after its map (``--stop-after-map``) maps for one component agent too, splits nothing and keeps the
map and the dossier beside it for a person to review (:func:`keep_map`); the run after the review is given them back.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Mapping
from pathlib import Path

from hyperloom_common.io import atomic_write_json
from hyperloom_multibranch.contracts.blackboard.records import OpShares
from hyperloom_multibranch.specialist.agent_onboarding import (
    ROOT_PROFILE_SOURCE,
    AttemptFailed,
    fetch_source,
    root_profile,
)
from hyperloom_multibranch.specialist.component_agent import SplitFailed, adapter_for, agent_of

logger = logging.getLogger(__name__)

SOURCES_DIR = "sources"
SHARES_DIR = "shares"
#: The map generated from the root's op shares, beside the profile it was generated from.
GENERATED_MAP = "generated-directions.json"
#: The fewest component agents a generated map is split for; one component agent maps its own directions.
GENERATED_FOR = 2
#: The ``PRELUDE`` decisions of the session's agent map, as the lifecycle records them: the root's shares as it waited
#: for and read them, a source that could not be fetched, the dossier pass skipped, started and ended, and the shares
#: left unopened because each would have less than ``--agent-share-minutes`` before its cutoff.
ROOT_SHARES_READ = "agent map: root shares read"
AGENT_SOURCES_UNREAD = "agent map: an agent source cannot be fetched"
DOSSIER_SKIPPED = "agent map: no dossier pass"
DOSSIER_STARTED = "agent map: dossier pass started"
DOSSIER_ENDED = "agent map: dossier pass ended"
SHARES_TOO_LATE = "agent map: too little time left for the shares"
#: The files a session that stops after its map keeps in its ``MB-agent-map`` directory: the map, the dossier under its
#: own name beside it, and what made them, read by whoever reviews it.
KEPT_MAP = "directions.json"
KEPT_DOSSIER = "dossier"
KEPT_SUMMARY = "agent-map.json"
#: The ``PRELUDE`` decision of a session that stops after its map, naming the summary :func:`keep_map` wrote, and what
#: made the map it kept.
MAP_KEPT = "agent map: kept for review"
MADE_BY_PASS = "dossier pass"
MADE_BY_SHARES = "generated from the root's shares"


async def component_agents(agents: tuple[str, ...], work_dir: Path) -> dict[int, Path]:
    """The component agents among ``agents``: each one's index and its checkout, each distinct source fetched once under
    ``work_dir``; raise :class:`~hyperloom_multibranch.specialist.agent_onboarding.AttemptFailed` or ``OSError`` when a
    source cannot be fetched."""

    checkouts = await _checkouts(agents, work_dir / SOURCES_DIR)
    return {index: path for index, agent in enumerate(agents) if (path := checkouts[agent]) is not None}


async def agent_plans(
    agents: tuple[str, ...],
    map_file: str,
    work_dir: Path,
    root_shares: OpShares | None = None,
    shares_source: str = ROOT_PROFILE_SOURCE,
) -> tuple[str, ...]:
    """Each of ``agents``' share of the directions in ``map_file``, or, without one, of a map generated from
    ``root_shares`` (which came from ``shares_source``) for two or more component agents, in order: the absolute path
    of a component agent's share, ``""`` for any other agent, and every one ``""`` when there is no map or it cannot be
    split. ``work_dir`` keeps the fetched sources, a generated map and the shares."""

    none = ("",) * len(agents)
    generating = not map_file and root_shares is not None and len(agents) >= GENERATED_FOR
    if not agents or not (map_file or generating):
        return none
    try:
        adapted = await component_agents(agents, work_dir)
    except (AttemptFailed, OSError) as exc:
        logger.error("the agents' directions are not split: an agent source cannot be fetched: %s", exc)
        return none
    if generating and len(adapted) < GENERATED_FOR:
        logger.info("no directions map is generated: %d component agents, each maps its own", len(adapted))
        return none
    if not adapted:
        logger.error("the agents' directions are not split: no agent named is a component agent")
        return none
    checkout = next(iter(adapted.values()))
    try:
        given = Path(map_file) if map_file else await generated_map(checkout, root_shares, shares_source, work_dir)
        shares = await _split(checkout, given, len(adapted), work_dir)
    except SplitFailed as exc:
        logger.error("the agents' directions are not split: %s", exc)
        return none
    plans = list(none)
    for index, share in zip(adapted, shares):
        plans[index] = str(share.absolute())
    logger.info("directions of %s split into %d shares, one per component agent", map_file or "the root", len(shares))
    return tuple(plans)


async def _split(checkout: Path, map_file: Path, shares: int, work_dir: Path) -> tuple[Path, ...]:
    """``map_file`` split by the component agent's checkout ``checkout`` into ``shares`` plans under ``work_dir``."""

    out = work_dir / SHARES_DIR
    out.mkdir(parents=True, exist_ok=True)
    return await agent_of(checkout).split_directions(checkout, map_file, shares, out)


async def generated_map(checkout: Path, root_shares: OpShares | None, source: str, work_dir: Path) -> Path:
    """The map the component agent's checkout ``checkout`` generates in ``work_dir`` from ``root_shares``, which came
    from ``source``; raise :class:`~hyperloom_multibranch.specialist.component_agent.SplitFailed` when there are none
    to generate it from or the agent cannot."""

    profile = root_profile(root_shares, work_dir, source)
    if profile is None:
        raise SplitFailed("the root has no shares to generate a map from")
    return await agent_of(checkout).generate_directions(checkout, profile, work_dir / GENERATED_MAP)


def keep_map(target: Path, map_file: Path, dossier: Path | None, made: Mapping[str, object]) -> Path:
    """Copy ``map_file`` and ``dossier``, when there is one, into ``target`` and write beside them ``made``, what made
    them, with the paths of both copies; the path of that summary."""

    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(map_file, target / KEPT_MAP)
    kept_dossier = ""
    if dossier is not None and dossier.is_dir():
        copied = target / KEPT_DOSSIER / dossier.name
        shutil.copytree(dossier, copied, dirs_exist_ok=True)
        kept_dossier = str(copied)
    summary = target / KEPT_SUMMARY
    atomic_write_json(summary, {**made, "map": str(target / KEPT_MAP), "dossier": kept_dossier})
    return summary


async def _checkouts(agents: tuple[str, ...], root: Path) -> dict[str, Path | None]:
    """Each distinct source of ``agents`` fetched once under ``root``, ``None`` for one that is no component agent."""

    checkouts: dict[str, Path | None] = {}
    for index, source in enumerate(dict.fromkeys(agents)):
        path = root / str(index)
        await fetch_source(source, path)
        checkouts[source] = path if adapter_for(path) is not None else None
    return checkouts
