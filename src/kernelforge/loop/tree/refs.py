# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The git references that keep every tree node reachable and every direction readable."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kernelforge.loop.tree.model import SearchTree
from kernelforge.tools.git import git

NODE_REF_PREFIX = "refs/forge/nodes/"


def node_ref(commit: str) -> str:
    return f"{NODE_REF_PREFIX}{commit}"


def pin_node(repo: str | Path, commit: str) -> None:
    """Keep ``commit`` reachable whatever branch moves past it."""
    git("update-ref", node_ref(commit), commit, cwd=repo)


def point_direction_branch(repo: str | Path, branch: str, commit: str) -> None:
    """Move a direction's branch to its best node; no worktree ever has a direction branch checked out."""
    git("update-ref", f"refs/heads/{branch}", commit, cwd=repo)


def commit_exists(repo: str | Path, commit: str) -> bool:
    result = git("cat-file", "-e", f"{commit}^{{commit}}", cwd=repo, check=False)
    return result.returncode == 0


def pinned_nodes(repo: str | Path) -> set[str]:
    """Every commit a node ref holds."""
    listed = git("for-each-ref", "--format=%(objectname)", NODE_REF_PREFIX, cwd=repo).stdout
    return {line.strip() for line in listed.splitlines() if line.strip()}


@dataclass(frozen=True)
class RefRepair:
    """What reconciling the tree with the repository found."""

    repinned: tuple[str, ...]
    lost: tuple[str, ...]


def reconcile_node_refs(repo: str | Path, tree: SearchTree) -> RefRepair:
    """Re-pin every node whose ref is missing; name the nodes whose commit no longer exists."""
    pinned = pinned_nodes(repo)
    repinned: list[str] = []
    lost: list[str] = []
    for node in tree.nodes:
        if node.commit in pinned:
            continue
        if commit_exists(repo, node.commit):
            pin_node(repo, node.commit)
            repinned.append(node.commit)
        else:
            lost.append(node.commit)
    return RefRepair(tuple(repinned), tuple(lost))


__all__ = [
    "NODE_REF_PREFIX",
    "RefRepair",
    "commit_exists",
    "node_ref",
    "pin_node",
    "pinned_nodes",
    "point_direction_branch",
    "reconcile_node_refs",
]
