# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The facts a selector decides a round from: the tree as measured, with no score of the framework's own."""

from __future__ import annotations

from dataclasses import dataclass

from kernelforge.loop.merge_candidates import cases_beating_reference
from kernelforge.loop.tree.model import Attempt, Direction, SearchTree, TreeNode

_PLAN_CHARS = 110
# How much of a direction's own record (hypothesis, ceiling, criterion, rebuttal) one line carries.
_RECORD_CHARS = 300
_COMMIT_CHARS = 12
# Attempts listed under a node; older ones are counted.
_ATTEMPTS_PER_NODE = 6


@dataclass(frozen=True)
class RoundOutlook:
    """How much campaign is left for the rounds still to come."""

    lanes: int
    remaining_sec: float
    rounds_left: int


@dataclass(frozen=True)
class FactSources:
    """Where the detail behind each fact can be read."""

    archive_dir: str
    lessons_dir: str
    selections_dir: str


def render_facts(
    tree: SearchTree,
    *,
    outlook: RoundOutlook,
    sources: FactSources,
    compose_failures: tuple[tuple[str, str], ...] = (),
) -> str:
    """Every fact the selector is given, in the order it reads them."""
    return "\n".join(
        [
            *_sources(sources),
            *_standing(tree, outlook),
            *_directions(tree),
            *_nodes(tree),
            *_compose_failures(compose_failures),
            *_previous_selections(tree),
        ]
    )


def _short(commit: str) -> str:
    return commit[:_COMMIT_CHARS]


def _one_line(text: str, limit: int = _PLAN_CHARS) -> str:
    return " ".join(str(text or "").split())[:limit]


def _sources(sources: FactSources) -> list[str]:
    return [
        "## Where the detail lives",
        f"- {sources.archive_dir}/iter_NNN/: an attempt's kernel, change.diff, meta.json (per-case measurements).",
        f"- {sources.lessons_dir}/iter_NNN.md: what that attempt's Implementer session tried and measured, including "
        "its section of locally validated results.",
        f"- {sources.selections_dir}/iter_NNN.md: what you were shown and answered in an earlier round.",
        "- Every node is a commit: `git show <commit>`, `git diff <parent> <commit>`, `git log --oneline <branch>`.",
        "",
    ]


def _standing(tree: SearchTree, outlook: RoundOutlook) -> list[str]:
    published = tree.require_node(tree.published_best)
    measured = tree.best_measured()
    return [
        "## Standing",
        f"- Published best: {published.commit}, score {published.score:.6f}x over {len(published.samples)} samples.",
        f"- Highest measured node: {measured.commit}, score {measured.score:.6f}x.",
        f"- Lanes this round: {outlook.lanes}. Campaign time left: {outlook.remaining_sec / 60:.0f} min, "
        f"about {outlook.rounds_left} round(s) at the cost rounds have had so far.",
        f"- {len(tree.nodes)} node(s), {len(tree.directions)} direction(s), {len(tree.attempts)} finished attempt(s).",
        "",
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


def _nodes(tree: SearchTree) -> list[str]:
    published = tree.require_node(tree.published_best)
    lines = ["## Nodes, depth first"]
    stack = [tree.root]
    while stack:
        node = stack.pop()
        depth = len(tree.ancestors(node.commit)) - 1
        lines.extend(_node(tree, node, published, "  " * depth))
        stack.extend(reversed(tree.children(node.commit)))
    return [*lines, ""]


def _node(tree: SearchTree, node: TreeNode, published: TreeNode, indent: str) -> list[str]:
    parent = tree.node(node.parent)
    versus_parent = f", {_percent(node.score, parent.score)} vs parent" if parent is not None else ""
    marks = [mark for mark, on in (("PUBLISHED", node.commit == published.commit), ("root", node.is_root)) if on]
    head = (
        f"{indent}- {node.commit} [{', '.join([node.direction or '-', *marks])}] "
        f"{node.source.value} in iteration {node.iteration}: {node.score:.6f}x over {len(node.samples)} sample(s)"
        f"{versus_parent}, {_percent(node.score, published.score)} vs published"
    )
    lines = [head]
    if node.plan:
        lines.append(f"{indent}  plan: {_one_line(node.plan)}")
    lines.extend(_case_standing(node, published, indent))
    opened = tree.directions_from(node.commit)
    if opened:
        lines.append(f"{indent}  directions opened here: {', '.join(item.direction_id for item in opened)}")
    lines.extend(_attempts(tree.attempts_from(node.commit), indent))
    return lines


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


def _attempts(attempts: list[Attempt], indent: str) -> list[str]:
    if not attempts:
        return [f"{indent}  attempts from here: none"]
    lines = [f"{indent}  attempts from here: {len(attempts)}"]
    shown = attempts[-_ATTEMPTS_PER_NODE:]
    if len(shown) < len(attempts):
        lines.append(f"{indent}    ({len(attempts) - len(shown)} older attempt(s) not listed)")
    for attempt in shown:
        score = f" {attempt.score:.6f}x" if attempt.score is not None else ""
        child = f" -> {_short(attempt.node)}" if attempt.node else ""
        lines.append(
            f"{indent}    iter {attempt.iteration} slot {attempt.slot} {attempt.slot_type.value} "
            f"{attempt.direction or 'new'} {attempt.outcome.value}{score}{child}"
        )
    return lines


def _compose_failures(failures: tuple[tuple[str, str], ...]) -> list[str]:
    if not failures:
        return []
    lines = ["## Gains that did not stack mechanically (candidates for PORT)"]
    lines.extend(f"- {_short(source)} onto {_short(target)}: the diff did not apply" for source, target in failures)
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
            f"{item.slot_type.value} at {_short(item.start)} ({_one_line(item.reason, 80)}) -> {item.outcome.value}"
            for item in by_iteration[iteration]
        )
        lines.append(f"- iteration {iteration}: {slots}")
    return lines


def _percent(value: float, reference: float) -> str:
    if not reference:
        return "n/a"
    return f"{(value / reference - 1.0) * 100:+.2f}%"


__all__ = ["FactSources", "RoundOutlook", "render_facts"]
