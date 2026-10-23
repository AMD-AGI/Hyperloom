# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The search tree a tree-strategy campaign keeps over its KEEP commits."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path

from kernelforge.tools.durable_io import atomic_write_text


class SearchStrategy(str, Enum):
    """How a campaign chooses the commit each iteration starts from."""

    LINEAR = "linear"
    TREE = "tree"


ROOT_SOURCE_BASELINE = "baseline"
NODE_SOURCE_ITERATION = "iteration"
KEEP_DECISION = "KEEP"

# How many attempts per node the tree view spells out; older ones are counted, not listed.
VIEW_ATTEMPTS_PER_NODE = 8
_VIEW_PLAN_CHARS = 90
_VIEW_COMMIT_CHARS = 12


@dataclass(frozen=True)
class TreeNode:
    """One validated and measured kernel version: a KEEP commit, or the root the campaign started from."""

    commit: str
    parent: str
    branch: str
    depth: int
    iteration: int
    mean_case_speedup: float
    wall_ms: float | None
    case_times: dict[str, float]
    plan: str
    source: str


@dataclass(frozen=True)
class TreeBranch:
    """One git branch of the tree, the commit it was created at and the commit it now ends on."""

    name: str
    head: str
    start_commit: str
    created_iteration: int


@dataclass(frozen=True)
class TreeSeat:
    """The start an iteration was given, recorded before anything is planned on it."""

    iteration: int
    start_commit: str
    branch: str
    selector_reason: str
    selector_fallback: str


@dataclass(frozen=True)
class TreeAttempt:
    """One finished iteration: where it started, what it was judged, and the node it produced."""

    iteration: int
    start_commit: str
    branch: str
    decision: str
    plan: str
    mean_case_speedup: float | None
    child_commit: str
    selector_reason: str
    selector_fallback: str


def _rebuild(model, payload: object, label: str):
    """One dataclass from exactly the fields this module writes."""
    if not isinstance(payload, dict):
        raise ValueError(f"search tree {label} must be an object")
    expected = {item.name for item in fields(model)}
    if set(payload) != expected:
        raise ValueError(f"search tree {label} fields do not match: {sorted(set(payload) ^ expected)}")
    return model(**payload)


@dataclass
class SearchTree:
    """Every node, branch and attempt of one tree-strategy campaign."""

    root_branch: str
    nodes: list[TreeNode]
    branches: list[TreeBranch]
    attempts: list[TreeAttempt] = field(default_factory=list)
    current_branch: str = ""
    next_branch_seq: int = 1
    seat: TreeSeat | None = None

    @classmethod
    def create(cls, root: TreeNode) -> "SearchTree":
        """A tree holding only the root, on the campaign's own branch."""
        branch = TreeBranch(root.branch, root.commit, root.commit, 0)
        return cls(root_branch=root.branch, nodes=[root], branches=[branch], current_branch=root.branch)

    def to_dict(self) -> dict:
        payload = asdict(self)
        # Derived, written for readers of the file; ``from_dict`` refuses a copy that disagrees with the nodes.
        payload["best_commit"] = self.global_best().commit
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "SearchTree":
        if not isinstance(payload, dict):
            raise ValueError("search tree must be a JSON object")
        expected = {item.name for item in fields(cls)} | {"best_commit"}
        if set(payload) != expected:
            raise ValueError(f"search tree fields do not match: {sorted(set(payload) ^ expected)}")
        seat = payload["seat"]
        tree = cls(
            root_branch=str(payload["root_branch"]),
            nodes=[_rebuild(TreeNode, item, "node") for item in payload["nodes"]],
            branches=[_rebuild(TreeBranch, item, "branch") for item in payload["branches"]],
            attempts=[_rebuild(TreeAttempt, item, "attempt") for item in payload["attempts"]],
            current_branch=str(payload["current_branch"]),
            next_branch_seq=int(payload["next_branch_seq"]),
            seat=None if seat is None else _rebuild(TreeSeat, seat, "seat"),
        )
        if not tree.nodes or tree.nodes[0].parent:
            raise ValueError("search tree must start with its root node")
        if tree.branch(tree.current_branch) is None:
            raise ValueError(f"search tree current branch is unknown: {tree.current_branch}")
        if payload["best_commit"] != tree.global_best().commit:
            raise ValueError("search tree best_commit does not match its nodes")
        return tree

    @property
    def root(self) -> TreeNode:
        return self.nodes[0]

    def node(self, commit: str) -> TreeNode | None:
        return next((item for item in self.nodes if item.commit == commit), None)

    def branch(self, name: str) -> TreeBranch | None:
        return next((item for item in self.branches if item.name == name), None)

    def branch_headed_by(self, commit: str) -> str:
        """The branch whose HEAD is ``commit``, or \"\" when it heads none."""
        return next((item.name for item in self.branches if item.head == commit), "")

    def resolve(self, token: str) -> str:
        """The node a full commit or an unambiguous prefix of at least seven characters names, or \"\"."""
        candidate = token.strip().lower()
        if len(candidate) < 7:
            return ""
        matches = [item.commit for item in self.nodes if item.commit.startswith(candidate)]
        return matches[0] if len(matches) == 1 else ""

    def ancestors(self, commit: str) -> list[TreeNode]:
        """The path from the root to ``commit``, both included."""
        path: list[TreeNode] = []
        current = self.node(commit)
        while current is not None:
            path.append(current)
            current = self.node(current.parent) if current.parent else None
        return list(reversed(path))

    def global_best(self) -> TreeNode:
        """The highest-scoring node; the earliest one wins a tie."""
        return max(self.nodes, key=lambda item: (item.mean_case_speedup, -item.iteration))

    def attempts_from(self, commit: str) -> list[TreeAttempt]:
        return [item for item in self.attempts if item.start_commit == commit]

    def branch_stall(self, name: str) -> int:
        """Attempts on ``name`` since it last produced a node."""
        streak = 0
        for attempt in reversed(self.attempts):
            if attempt.branch != name:
                continue
            if attempt.decision == KEEP_DECISION:
                break
            streak += 1
        return streak

    def history_scope(self, start_commit: str) -> frozenset[int]:
        """Iterations whose history belongs to the line ending at ``start_commit``.

        The nodes on the path, the failed attempts made from any of them, and every attempt made from the start
        itself; a KEEP made from an ancestor that is not on the path belongs to another branch.
        """
        path = self.ancestors(start_commit)
        on_path = {item.commit for item in path}
        scope = {item.iteration for item in path if item.iteration > 0}
        for attempt in self.attempts:
            if attempt.start_commit == start_commit or (
                attempt.start_commit in on_path and attempt.decision != KEEP_DECISION
            ):
                scope.add(attempt.iteration)
        return frozenset(scope)

    def allocate_branch(self, start_commit: str, iteration: int) -> str:
        """Name and record a new branch starting at ``start_commit``."""
        name = f"{self.root_branch}-b{self.next_branch_seq:03d}"
        self.next_branch_seq += 1
        self.branches.append(TreeBranch(name, start_commit, start_commit, iteration))
        return name

    def open_seat(self, seat: TreeSeat) -> None:
        if self.branch(seat.branch) is None:
            raise ValueError(f"seat names an unknown branch: {seat.branch}")
        self.seat = seat
        self.current_branch = seat.branch

    def close_seat(self, attempt: TreeAttempt, node: TreeNode | None) -> None:
        """Record the seated iteration's attempt and, for a KEEP, the node it added on the seat's branch."""
        if self.seat is None or self.seat.iteration != attempt.iteration:
            raise ValueError(f"iteration {attempt.iteration} does not hold the seat")
        if node is not None:
            if self.node(node.commit) is not None:
                raise ValueError(f"search tree already holds {node.commit}")
            self.nodes.append(node)
            self.branches = [
                TreeBranch(item.name, node.commit, item.start_commit, item.created_iteration)
                if item.name == node.branch
                else item
                for item in self.branches
            ]
        self.attempts.append(attempt)
        self.seat = None


class SearchTreeStore:
    """Atomic store for ``forge_experiments/search_tree.json``."""

    def __init__(self, workspace_dir: str):
        self.path = Path(workspace_dir) / "forge_experiments" / "search_tree.json"

    def load(self) -> SearchTree | None:
        if not self.path.is_file():
            return None
        try:
            payload = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid search tree: {self.path}: {error}") from error
        return SearchTree.from_dict(payload)

    def save(self, tree: SearchTree) -> None:
        atomic_write_text(self.path, json.dumps(tree.to_dict(), indent=2, sort_keys=True) + "\n")


def _short(commit: str) -> str:
    return commit[:_VIEW_COMMIT_CHARS]


def _one_line(text: str) -> str:
    return " ".join(str(text or "").split())[:_VIEW_PLAN_CHARS]


def _render_attempts(tree: SearchTree, node: TreeNode, indent: str) -> list[str]:
    attempts = tree.attempts_from(node.commit)
    if not attempts:
        return [f"{indent}  attempts from here: none"]
    lines = [f"{indent}  attempts from here: {len(attempts)}"]
    shown = attempts[-VIEW_ATTEMPTS_PER_NODE:]
    if len(shown) < len(attempts):
        lines.append(f"{indent}    ({len(attempts) - len(shown)} older attempt(s) not listed)")
    for attempt in shown:
        score = f" {attempt.mean_case_speedup:.6f}x" if attempt.mean_case_speedup is not None else ""
        child = f" -> {_short(attempt.child_commit)}" if attempt.child_commit else ""
        lines.append(
            f'{indent}    iter {attempt.iteration} {attempt.decision}{score}{child}: "{_one_line(attempt.plan)}"'
        )
    return lines


def _render_node(tree: SearchTree, node: TreeNode, best: TreeNode) -> list[str]:
    indent = "  " * node.depth
    marks = [f"branch {node.branch}"]
    if tree.branch_headed_by(node.commit):
        marks.append("HEAD")
    if node.commit == best.commit:
        marks.append("GLOBAL BEST")
    parent = tree.node(node.parent)
    delta = ""
    if parent is not None and parent.mean_case_speedup:
        delta = f" ({(node.mean_case_speedup / parent.mean_case_speedup - 1.0) * 100:+.2f}% vs parent)"
    origin = "root" if node.iteration == 0 else f"iter {node.iteration}"
    line = (
        f"{indent}- {node.commit} [{', '.join(marks)}] {origin}, depth {node.depth}, "
        f"mean case speedup {node.mean_case_speedup:.6f}x{delta}"
    )
    if node.plan:
        line += f': "{_one_line(node.plan)}"'
    return [line, *_render_attempts(tree, node, indent)]


def _render_sources(archive_dir: str, lessons_dir: str) -> list[str]:
    return [
        "## Where the detail lives",
        f"- {archive_dir}/iter_NNN/: one attempt's full kernel, change.diff, meta.json (per-case measurements) and "
        "validation.txt, for the iteration number NNN shown below; attempts that changed nothing have none.",
        f"- {lessons_dir}/iter_NNN.md: what that attempt's Implementer session tried, including abandoned directions.",
        "- Every node is a git commit and every branch a git branch: `git show <commit>`, "
        "`git diff <parent> <commit>`, `git log --oneline <branch>`.",
        "",
    ]


def render_tree_view(tree: SearchTree, *, archive_dir: str, lessons_dir: str) -> str:
    """The tree as the start selector reads it: where detail lives, every node depth-first, its attempts, branches."""
    best = tree.global_best()
    children: dict[str, list[TreeNode]] = {}
    for item in tree.nodes[1:]:
        children.setdefault(item.parent, []).append(item)
    lines = [
        *_render_sources(archive_dir, lessons_dir),
        "## Search tree",
        f"Root branch: {tree.root_branch}. Global best: {best.commit} on {best.branch}, "
        f"mean case speedup {best.mean_case_speedup:.6f}x.",
        f"{len(tree.nodes)} node(s), {len(tree.branches)} branch(es), {len(tree.attempts)} finished attempt(s).",
        "",
    ]
    stack = [tree.root]
    while stack:
        node = stack.pop()
        lines.extend(_render_node(tree, node, best))
        stack.extend(reversed(children.get(node.commit, [])))
    lines.extend(["", "## Branches"])
    for branch in tree.branches:
        lines.append(
            f"- {branch.name}: HEAD {_short(branch.head)}, created at {_short(branch.start_commit)} "
            f"(iteration {branch.created_iteration}), {tree.branch_stall(branch.name)} attempt(s) since its last node"
        )
    return "\n".join(lines)


__all__ = [
    "KEEP_DECISION",
    "NODE_SOURCE_ITERATION",
    "ROOT_SOURCE_BASELINE",
    "SearchStrategy",
    "SearchTree",
    "SearchTreeStore",
    "TreeAttempt",
    "TreeBranch",
    "TreeNode",
    "TreeSeat",
    "render_tree_view",
]
