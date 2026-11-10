# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Arbor's map as the scheduler deals it: each direction of the session's agent map, the share it was split into, and
the Arbor branch that share went to.

A session's agent map is split into one share per Arbor agent (:mod:`agent_plans`), and a share branch works the open
directions of its share. The scheduler keeps the map on the blackboard
(:meth:`~hyperloom_multibranch.contracts.blackboard.Blackboard.write_agent_map`), each direction with the branch its
share was dealt to, and that record is what the open areas, the specialists' briefs, the menu and the judge read: a
direction is held while its branch runs, and the ones the map closed (``done``, ``killed``, ``below_floor``,
``plan``) are its verdicts, each with why. What stays here is which share each direction is in, which the record does
not carry, so that the branch a share is dealt to can be named as its directions' holder.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from hyperloom_multibranch.contracts.blackboard.records import MAP_OPEN, BlackboardSnapshot, MapDirection
from hyperloom_multibranch.scheduling.review import ENCODING, directions

_TITLE_CHARS = 160
_WHY_CHARS = 300
#: The status Arbor's master gives a direction a report closed without one.
_MAP_DONE = "done"
#: The judge's rule for the map, the one the specialists' brief binds them to.
JUDGE_RULE = (
    "Arbor's map: the directions the session's Arbor branches work, and those the map set aside. Open no branch on a "
    "direction a running Arbor branch holds, nor focus one on its ops; it is free again once that branch stops. Open "
    "none on a set-aside direction unless its focus answers the reason given, and say how."
)


@dataclass(frozen=True)
class SharedMap:
    """The session's map, its ``directions`` in the map's order, and ``shares``, the plan file of the share each
    direction was split into, by its id; a direction in no share is not in it."""

    directions: tuple[MapDirection, ...]
    shares: Mapping[str, str]

    @classmethod
    def read(cls, map_file: Path, plans: Sequence[str]) -> SharedMap:
        """The map in ``map_file`` and the share of ``plans``, plan files, each direction is in, no holder yet; an empty
        map when the file cannot be read, and only directions with an id, the first of each."""

        shares = {str(entry.get("id", "")): plan for plan in plans if plan for entry in directions(Path(plan))}
        listed: dict[str, MapDirection] = {}
        for entry in directions(map_file):
            found = MapDirection(
                id=str(entry.get("id") or ""),
                title=" ".join(str(entry.get("title") or "").split())[:_TITLE_CHARS],
                status=str(entry.get("status") or MAP_OPEN),
                why=" ".join(str(entry.get("why") or "").split())[:_WHY_CHARS],
                ops=tuple(str(op) for op in entry.get("ops") or () if str(op)),
            )
            if found.id:
                listed.setdefault(found.id, found)
        return cls(tuple(listed.values()), {key: plan for key, plan in shares.items() if key in listed})

    def folded(self, report: Path, order: Path | None) -> SharedMap:
        """The map once a worker's ``report`` is folded in and Arbor's master dealt ``order`` (``None`` for none): each
        open direction the report closed takes its status and why, and the order's directions move to that share, whose
        holder the branch it is dealt to becomes; every other holder is kept."""

        closed = _closed_in(report)
        folded = tuple(
            replace(
                item,
                status=str(closed[item.id].get("status") or _MAP_DONE),
                why=" ".join(str(closed[item.id].get("why") or "").split())[:_WHY_CHARS] or item.why,
            )
            if item.open and item.id in closed
            else item
            for item in self.directions
        )
        listed = {item.id for item in self.directions}
        moved = {str(entry.get("id") or ""): str(order) for entry in directions(order)} if order else {}
        return SharedMap(folded, {**self.shares, **{key: plan for key, plan in moved.items() if key in listed}})

    def dealt(self, plan: str, branch: str) -> SharedMap:
        """The map with the directions of the share ``plan`` held by ``branch``, the branch it was dealt to."""

        held = tuple(
            replace(item, holder=branch) if self.shares.get(item.id) == plan else item for item in self.directions
        )
        return replace(self, directions=held)


def _closed_in(report: Path) -> dict[str, Mapping[str, Any]]:
    """The directions an Arbor worker report (``arbor.worker_report/1``) closed, by id, as Arbor's master reads them:
    its ``directions.closed``, each with its ``status`` and ``why``; none when the report cannot be read."""

    try:
        document = json.loads(report.read_text(encoding=ENCODING))
    except (OSError, ValueError):
        return {}
    reported = document.get("directions") if isinstance(document, Mapping) else None
    listed = reported.get("closed") if isinstance(reported, Mapping) else None
    entries = listed if isinstance(listed, list) else []
    return {str(entry["id"]): entry for entry in entries if isinstance(entry, Mapping) and entry.get("id")}


def render_map(snapshot: BlackboardSnapshot) -> str:
    """The map ``snapshot`` keeps, for the judge's prompt: each direction held by its running Arbor branch, open, or
    set aside with why, as the specialists' brief words them; ``""`` for a session without a map."""

    if not snapshot.agent_map:
        return ""
    holders = snapshot.map_holders()
    lines = [JUDGE_RULE]
    for item in snapshot.agent_map:
        if not item.open:
            lines += [f"- {item.id} (set aside, {item.status}): {item.title}", f"  why: {item.why or 'not given'}"]
            continue
        holder = holders.get(item.id)
        lines.append(
            f"- {item.id} ({f'held by {holder}' if holder else 'open, held by no running branch'}): {item.title}"
        )
    return "\n".join(lines)


__all__ = ["JUDGE_RULE", "SharedMap", "render_map"]
