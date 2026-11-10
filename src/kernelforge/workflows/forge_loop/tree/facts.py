# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The facts a selector decides a round from: the tree as measured, with no score of the framework's own."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from kernelforge.workflows.forge_loop.iteration.merge_candidates import cases_beating_reference
from kernelforge.workflows.forge_loop.tree.model import Attempt, Direction, MergeFailure, SearchTree, TreeNode

_PLAN_CHARS = 110
# How much of a direction's own record (hypothesis, ceiling, criterion, rebuttal) one line carries.
_RECORD_CHARS = 300
_COMMIT_CHARS = 12
# Attempts listed under a node; older ones are counted.
_ATTEMPTS_PER_NODE = 6
LESSON_FILENAME = "lesson.md"
LESSON_MISSING_FILENAME = "lesson_missing.txt"


@dataclass(frozen=True)
class RoundOutlook:
    """How much campaign is left for the rounds still to come."""

    lanes: int
    remaining_sec: float
    # None until a round of this campaign has finished and given its cost.
    rounds_left: int | None
    # The model each slot position of this round is planned and implemented on, in the order the selector names its
    # slots; empty when every slot runs on one model.
    slot_models: tuple[str, ...] = ()


@dataclass(frozen=True)
class FactSources:
    """Where the detail behind each fact can be read."""

    archive_dir: str
    selections_dir: str


def render_facts(
    tree: SearchTree,
    *,
    outlook: RoundOutlook,
    sources: FactSources,
    merge_failures: tuple[MergeFailure, ...] = (),
    models: Mapping[str, str] | None = None,
) -> str:
    """Every fact the selector is given, in the order it reads them; ``models`` names each Agent profile's model."""
    return "\n".join(
        [
            *_sources(sources),
            *_standing(tree, outlook),
            *_directions(tree),
            *_nodes(tree, models or {}),
            *_merge_failures(merge_failures),
            *_previous_selections(tree),
        ]
    )


def _short(commit: str) -> str:
    return commit[:_COMMIT_CHARS]


def _one_line(text: str, limit: int = _PLAN_CHARS) -> str:
    return " ".join(str(text or "").split())[:limit]


def render_line_history(tree: SearchTree, start: str) -> str:
    """The line ending at ``start`` as its planners read it: the path from the root, then its attempts, oldest first."""
    line = tree.line(start)
    attempts = [entry for attempt in line.attempts for entry in _line_attempt(attempt)]
    return "\n".join(
        [
            f"# The line of search-tree node {start}",
            "",
            f"A records directory holds the change.diff, measurement.json and {LESSON_FILENAME} (the Implementer's own "
            f"record of what it tried and found, or {LESSON_MISSING_FILENAME} saying why there is none) of what made "
            "that node or attempt.",
            "",
            "## The path from the root",
            *(entry for node in line.nodes for entry in _path_node(tree, node)),
            "",
            "## Attempts from this path",
            *(attempts or ["- none yet"]),
            "",
        ]
    )


def _path_node(tree: SearchTree, node: TreeNode) -> list[str]:
    head = (
        f"- {node.commit} [{node.direction or '-'}] {node.source.value} in iteration {node.iteration}: "
        f"{node.score:.6f}x over {len(node.samples)} sample(s){_versus_parent(tree, node)}"
    )
    records = [f"  records: {node.records}"] if node.records else []
    return [head, *_what_it_did(node, ""), *records]


def _line_attempt(attempt: Attempt) -> list[str]:
    lines = [
        f"- {_attempt_summary(attempt, {})}",
        f"  from {attempt.start}: {_one_line(attempt.reason, _RECORD_CHARS)}",
    ]
    if attempt.plan_path:
        lines.append(f"  plan: {attempt.plan_path}")
    if attempt.archive_path:
        lines.append(f"  records: {attempt.archive_path}")
    return lines


def _sources(sources: FactSources) -> list[str]:
    return [
        "## Where the detail lives",
        f"- {sources.archive_dir}/iter_NNN/slot_KK/: one slot's change.diff, measurement.json (per-case timings of "
        f"every run) and {LESSON_FILENAME} (its Implementer's own record of what it tried and found, or "
        f"{LESSON_MISSING_FILENAME} saying why there is none); start_<commit>/slot_KK.md is the plan it ran "
        "(start_<commit>_<profile>/ when the start's slots ran on several profiles).",
        f"- {sources.selections_dir}/iter_NNN.md: what you were shown and answered in an earlier round.",
        "- Every node is a commit: `git show <commit>`, `git diff <parent> <commit>`, `git log --oneline <branch>`.",
        "",
    ]


def _standing(tree: SearchTree, outlook: RoundOutlook) -> list[str]:
    published = tree.require_node(tree.published_best)
    measured = tree.best_measured()
    rounds = (
        f"about {outlook.rounds_left} round(s) at the cost rounds have had so far"
        if outlook.rounds_left is not None
        else "no round has finished yet to price the rest from"
    )
    return [
        "## Standing",
        f"- Published best: {published.commit}, score {published.score:.6f}x over {len(published.samples)} samples.",
        f"- Highest measured node: {measured.commit}, score {measured.score:.6f}x.",
        f"- Lanes this round: {outlook.lanes}. Campaign time left: {outlook.remaining_sec / 60:.0f} min, {rounds}.",
        f"- {len(tree.nodes)} node(s), {len(tree.directions)} direction(s), {len(tree.attempts)} finished attempt(s).",
        *_slot_models(outlook.slot_models),
        "",
    ]


def _slot_models(models: tuple[str, ...]) -> list[str]:
    if not models:
        return []
    order = ", ".join(f"slot {position} on {model}" for position, model in enumerate(models, start=1))
    return [
        f"- Each slot of this round is planned and implemented on its own model, by the position you name it at: "
        f"{order}."
    ]


def _directions(tree: SearchTree) -> list[str]:
    lines = ["## Directions"]
    if not tree.directions:
        lines.append("- none yet: every first slot opens one with BRANCH")
    for direction in tree.directions:
        lines.extend(_direction(tree, direction))
    return [*lines, ""]


def _direction(tree: SearchTree, direction: Direction) -> list[str]:
    best = tree.direction_best(direction.direction_id)
    attempts = [item for item in tree.attempts if item.direction == direction.direction_id]
    status = direction.status.value + (f" ({direction.status_reason})" if direction.status_reason else "")
    standing = (
        f"best {best.commit[:_COMMIT_CHARS]} at {best.score:.6f}x from iteration {best.iteration}"
        if best is not None
        else "no node yet"
    )
    lines = [
        f'- {direction.direction_id} "{_one_line(direction.name)}" [{status}], branch {direction.branch}, '
        f"opened at {_short(direction.origin)} in iteration {direction.created_iteration}; {standing}; "
        f"{len(tree.direction_nodes(direction.direction_id))} node(s), {len(attempts)} attempt(s).",
        f"  hypothesis: {_one_line(direction.hypothesis, _RECORD_CHARS)}",
        f"  expected ceiling: {_one_line(direction.expected_ceiling, _RECORD_CHARS)}; "
        f"fails if: {_one_line(direction.failure_criterion, _RECORD_CHARS)}",
    ]
    lines.extend(f"  planners refuted a hint: {_one_line(item, _RECORD_CHARS)}" for item in direction.hint_rebuttals)
    return lines


def _nodes(tree: SearchTree, models: Mapping[str, str]) -> list[str]:
    published = tree.require_node(tree.published_best)
    lines = ["## Nodes, depth first"]
    stack = [tree.root]
    while stack:
        node = stack.pop()
        depth = len(tree.ancestors(node.commit)) - 1
        lines.extend(_node(tree, node, published, "  " * depth, models))
        stack.extend(reversed(tree.children(node.commit)))
    return [*lines, ""]


def _node(tree: SearchTree, node: TreeNode, published: TreeNode, indent: str, models: Mapping[str, str]) -> list[str]:
    marks = [mark for mark, on in (("PUBLISHED", node.commit == published.commit), ("root", node.is_root)) if on]
    head = (
        f"{indent}- {node.commit} [{', '.join([node.direction or '-', *marks])}] "
        f"{node.source.value} in iteration {node.iteration}{_model(node.agent_profile_id, models, 'by')}: "
        f"{node.score:.6f}x over {len(node.samples)} sample(s)"
        f"{_versus_parent(tree, node)}, {_percent(node.score, published.score)} vs published"
    )
    lines = [head, *_what_it_did(node, indent)]
    lines.extend(_case_standing(node, published, indent))
    opened = tree.directions_from(node.commit)
    if opened:
        lines.append(f"{indent}  directions opened here: {', '.join(item.direction_id for item in opened)}")
    lines.extend(_attempts(tree.attempts_from(node.commit), indent, models))
    return lines


def _versus_parent(tree: SearchTree, node: TreeNode) -> str:
    parent = tree.node(node.parent)
    return f", {_percent(node.score, parent.score)} vs parent" if parent is not None else ""


def _what_it_did(node: TreeNode, indent: str) -> list[str]:
    """The plan a node was made by, and the approach and cases its lane was given."""
    lines = []
    if node.plan:
        lines.append(f"{indent}  plan: {_one_line(node.plan)}")
    if node.approach or node.focus_cases:
        aimed = ", ".join(node.focus_cases) or "every case"
        lines.append(f"{indent}  approach: {_one_line(node.approach, _RECORD_CHARS) or '-'}; aimed at: {aimed}")
    return lines


def _model(profile_id: str, models: Mapping[str, str], preposition: str) -> str:
    return f" {preposition} {models.get(profile_id, profile_id)}" if profile_id else ""


def _case_standing(node: TreeNode, published: TreeNode, indent: str) -> list[str]:
    if node.commit == published.commit:
        return []
    wins = cases_beating_reference(node.case_times, published.case_times, node.case_spreads)
    losses = cases_beating_reference(published.case_times, node.case_times, published.case_spreads)
    if not wins and not losses:
        return [f"{indent}  vs published: no case differs beyond noise"]
    return [
        f"{indent}  beats published beyond noise on: {', '.join(sorted(wins)) or 'none'}; "
        f"loses on: {', '.join(sorted(losses)) or 'none'}"
    ]


def _attempts(attempts: list[Attempt], indent: str, models: Mapping[str, str]) -> list[str]:
    if not attempts:
        return [f"{indent}  attempts from here: none"]
    lines = [f"{indent}  attempts from here: {len(attempts)}"]
    shown = attempts[-_ATTEMPTS_PER_NODE:]
    if len(shown) < len(attempts):
        lines.append(f"{indent}    ({len(attempts) - len(shown)} older attempt(s) not listed)")
    lines.extend(f"{indent}    {_attempt_summary(attempt, models)}" for attempt in shown)
    return lines


def _attempt_summary(attempt: Attempt, models: Mapping[str, str]) -> str:
    score = f" {attempt.score:.6f}x" if attempt.score is not None else ""
    child = f" -> {_short(attempt.node)}" if attempt.node else ""
    return (
        f"iter {attempt.iteration} slot {attempt.slot} {attempt.slot_type.value} "
        f"{attempt.direction or 'new'}{_model(attempt.agent_profile_id, models, 'on')} "
        f"{attempt.outcome.value}{score}{child}{_detail(attempt)}"
    )


def _detail(attempt: Attempt) -> str:
    return f": {_one_line(attempt.detail, _RECORD_CHARS)}" if attempt.detail else ""


def _merge_failures(failures: tuple[MergeFailure, ...]) -> list[str]:
    if not failures:
        return []
    lines = ["## Merges of the last round that did not land (candidates for PORT)"]
    lines.extend(
        f"- {_short(item.source)} onto {_short(item.base)}: {_one_line(item.reason, _RECORD_CHARS)}"
        for item in failures
    )
    return [*lines, ""]


def _previous_selections(tree: SearchTree) -> list[str]:
    by_iteration: dict[int, list[Attempt]] = {}
    for attempt in tree.attempts:
        by_iteration.setdefault(attempt.iteration, []).append(attempt)
    if not by_iteration:
        return []
    lines = ["## Your earlier rounds"]
    for iteration in sorted(by_iteration):
        slots = "; ".join(
            f"{item.slot_type.value} at {_short(item.start)} ({_one_line(item.reason, 80)}) -> "
            f"{item.outcome.value}{_detail(item)}"
            for item in by_iteration[iteration]
        )
        lines.append(f"- iteration {iteration}: {slots}")
    return lines


def _percent(value: float, reference: float) -> str:
    if not reference:
        return "n/a"
    return f"{(value / reference - 1.0) * 100:+.2f}%"


__all__ = [
    "LESSON_FILENAME",
    "LESSON_MISSING_FILENAME",
    "FactSources",
    "RoundOutlook",
    "render_facts",
    "render_line_history",
]
