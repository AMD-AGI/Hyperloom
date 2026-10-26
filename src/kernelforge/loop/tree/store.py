# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Atomic, fail-closed persistence of ``forge_experiments/search_tree.json``."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from enum import Enum
from pathlib import Path
from typing import Any

from kernelforge.loop.tree.model import (
    Attempt,
    AttemptOutcome,
    Direction,
    DirectionStatus,
    NodeSource,
    Round,
    SearchTree,
    Slot,
    SlotStatus,
    SlotType,
    TreeNode,
)
from kernelforge.tools.durable_io import atomic_write_text

SEARCH_TREE_FILE = "search_tree.json"


class SearchTreeStore:
    """The one file a tree campaign's search state lives in."""

    def __init__(self, experiments_dir: str | Path):
        self.path = Path(experiments_dir) / SEARCH_TREE_FILE

    def exists(self) -> bool:
        return self.path.is_file()

    def load(self) -> SearchTree:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid search tree {self.path}: {error}") from error
        return tree_from_dict(payload)

    def save(self, tree: SearchTree) -> None:
        atomic_write_text(self.path, json.dumps(tree_to_dict(tree), indent=2, sort_keys=True) + "\n")


def tree_to_dict(tree: SearchTree) -> dict[str, Any]:
    return _plain(asdict(tree))


def tree_from_dict(payload: object) -> SearchTree:
    """The tree a payload this module wrote describes; anything else is refused."""
    data = _exact(payload, SearchTree, "search tree")
    tree = SearchTree(
        root_branch=str(data["root_branch"]),
        nodes=[_node(item) for item in _list(data["nodes"], "nodes")],
        published_best=str(data["published_best"]),
        published_head=str(data["published_head"]),
        directions=[_direction(item) for item in _list(data["directions"], "directions")],
        attempts=[_attempt(item) for item in _list(data["attempts"], "attempts")],
        round=None if data["round"] is None else _round(data["round"]),
        next_direction_seq=int(data["next_direction_seq"]),
    )
    check_consistency(tree)
    return tree


def check_consistency(tree: SearchTree) -> None:
    """Refuse a tree whose references do not close over its own nodes and directions."""
    if not tree.nodes or not tree.root.is_root:
        raise ValueError("the search tree must start with its root node")
    seen: set[str] = set()
    known_directions = {item.direction_id for item in tree.directions}
    for node in tree.nodes:
        if node.commit in seen:
            raise ValueError(f"the search tree holds {node.commit} twice")
        if not node.is_root and (node.parent not in seen or node.direction not in known_directions):
            raise ValueError(f"node {node.commit} names a parent or direction recorded after it, or never")
        seen.add(node.commit)
    for direction in tree.directions:
        if direction.origin not in seen:
            raise ValueError(f"direction {direction.direction_id} opens at {direction.origin}, which is not a node")
    if tree.published_best not in seen:
        raise ValueError(f"the published best {tree.published_best} is not a node")
    for slot in tree.round.slots if tree.round is not None else ():
        if slot.start not in seen:
            raise ValueError(f"slot {slot.index} starts at {slot.start}, which is not a node")


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _exact(payload: object, model: type, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    expected = {item.name for item in fields(model)}
    if set(payload) != expected:
        raise ValueError(f"{label} fields do not match: {sorted(set(payload) ^ expected)}")
    return payload


def _list(value: object, label: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"search tree {label} must be a list")
    return value


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _node(payload: object) -> TreeNode:
    data = _exact(payload, TreeNode, "search tree node")
    return TreeNode(
        commit=str(data["commit"]),
        parent=str(data["parent"]),
        direction=str(data["direction"]),
        slot_type=None if data["slot_type"] is None else SlotType(data["slot_type"]),
        iteration=int(data["iteration"]),
        slot=int(data["slot"]),
        samples=tuple(float(item) for item in data["samples"]),
        wall_ms=_optional_float(data["wall_ms"]),
        case_times={str(key): float(value) for key, value in dict(data["case_times"]).items()},
        case_spreads={str(key): float(value) for key, value in dict(data["case_spreads"]).items()},
        snr_db=_optional_float(data["snr_db"]),
        plan=str(data["plan"]),
        source=NodeSource(data["source"]),
        provenance=tuple(str(item) for item in data["provenance"]),
    )


def _direction(payload: object) -> Direction:
    data = _exact(payload, Direction, "search tree direction")
    return Direction(
        direction_id=str(data["direction_id"]),
        name=str(data["name"]),
        hypothesis=str(data["hypothesis"]),
        origin=str(data["origin"]),
        branch=str(data["branch"]),
        expected_ceiling=str(data["expected_ceiling"]),
        failure_criterion=str(data["failure_criterion"]),
        created_iteration=int(data["created_iteration"]),
        status=DirectionStatus(data["status"]),
        status_reason=str(data["status_reason"]),
        hint_rebuttals=tuple(str(item) for item in data["hint_rebuttals"]),
    )


def _attempt(payload: object) -> Attempt:
    data = _exact(payload, Attempt, "search tree attempt")
    return Attempt(
        iteration=int(data["iteration"]),
        slot=int(data["slot"]),
        slot_type=SlotType(data["slot_type"]),
        start=str(data["start"]),
        direction=str(data["direction"]),
        outcome=AttemptOutcome(data["outcome"]),
        score=_optional_float(data["score"]),
        node=str(data["node"]),
        archive_path=str(data["archive_path"]),
        reason=str(data["reason"]),
    )


def _slot(payload: object) -> Slot:
    data = _exact(payload, Slot, "search tree slot")
    return Slot(
        index=int(data["index"]),
        slot_type=SlotType(data["slot_type"]),
        start=str(data["start"]),
        reason=str(data["reason"]),
        direction=str(data["direction"]),
        source=str(data["source"]),
        hint=str(data["hint"]),
        gains=str(data["gains"]),
        status=SlotStatus(data["status"]),
    )


def _round(payload: object) -> Round:
    data = _exact(payload, Round, "search tree round")
    return Round(
        iteration=int(data["iteration"]),
        slots=tuple(_slot(item) for item in _list(data["slots"], "round slots")),
        remeasure=tuple(str(item) for item in data["remeasure"]),
        refresh_analysis=tuple(str(item) for item in data["refresh_analysis"]),
    )


__all__ = ["SEARCH_TREE_FILE", "SearchTreeStore", "check_consistency", "tree_from_dict", "tree_to_dict"]
