# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The declared deliverable a specialist round hands back.

A round names what it changed -- tree, target paths, env and arg layers, setup
commands, whole-file artifacts -- rather than leaving the harness to infer it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class DeliverableRefused(ValueError):
    """A declared deliverable cannot be trusted and must not be integrated."""


@dataclass(frozen=True)
class Artifact:
    """One whole-file deliverable.

    Attributes:
        target: Path within ``tree_id`` the file installs to.
        tree_id: Tree the target belongs to, not necessarily the round's own.
        source: Absolute path of the authored file, where it was validated.
        kind: Free-form artifact kind label.
        description: Free-form human description.
    """

    target: str
    tree_id: str
    source: str
    kind: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "target": self.target,
            "tree_id": self.tree_id,
            "source": self.source,
            "kind": self.kind,
            "description": self.description,
        }


@dataclass(frozen=True)
class Deliverable:
    """Everything one round declares it produced.

    Attributes:
        tree_id: The round's primary tree, the one its patches apply to.
        targets: Tree-relative paths in the primary tree the round touches;
            they scope the harvest pathspec and the baseline manifest.
        patches: Absolute paths of the unified diffs the round authored.
        artifacts: Whole-file deliverables, each with its own tree.
        envs: Environment layer the round validated with.
        server_args: Server-arg fragment the round validated with.
        setup_commands: Ordered setup commands the round ran.
    """

    tree_id: str
    targets: tuple[str, ...] = ()
    patches: tuple[str, ...] = ()
    artifacts: tuple[Artifact, ...] = ()
    envs: Mapping[str, str] = field(default_factory=dict)
    server_args: str = ""
    setup_commands: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict, in the declared shape only."""
        return {
            "tree_id": self.tree_id,
            "targets": list(self.targets),
            "patches": list(self.patches),
            "artifacts": [a.to_dict() for a in self.artifacts],
            "envs": dict(self.envs),
            "server_args": self.server_args,
            "setup_commands": list(self.setup_commands),
        }


def _clean_seq(values: Any) -> tuple[str, ...]:
    """Return a de-duplicated tuple of non-empty strings, order preserved."""
    if not isinstance(values, (list, tuple)):
        return ()
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in out:
            out.append(text)
    return tuple(out)


def parse_deliverable(payload: Mapping[str, Any], *, default_tree_id: str) -> Deliverable:
    """Read a round's declared deliverable out of its ``specialist_done`` payload.

    A ``deliverable`` object is read when the round emits one, otherwise the
    flat keys (``patches_written``, ``extra_envs``, ``extra_server_args``,
    ``setup_commands``). Artifacts are not read here: the install resolves each
    declared entry against the workspace sandbox and the allowlisted roots.

    Args:
        payload: The parsed ``specialist_done`` content.
        default_tree_id: Tree id to attribute anything that does not name one.

    Returns:
        Deliverable: The declared deliverable, with no artifacts attached yet.
    """
    declared = payload.get("deliverable")
    source: Mapping[str, Any] = declared if isinstance(declared, Mapping) else payload
    patches = source.get("patches")
    if not isinstance(patches, (list, tuple)):
        patches = payload.get("patches_written")
    envs = source.get("envs")
    if not isinstance(envs, Mapping):
        envs = payload.get("extra_envs")
    server_args = source.get("server_args")
    if server_args is None:
        server_args = payload.get("extra_server_args")
    setup = source.get("setup_commands")
    if not isinstance(setup, (list, tuple)):
        setup = payload.get("setup_commands")

    return Deliverable(
        tree_id=str(source.get("tree_id", "")).strip() or default_tree_id,
        targets=_clean_seq(source.get("targets")),
        patches=_clean_seq(patches),
        envs={str(k): str(v) for k, v in envs.items() if str(k).strip()} if isinstance(envs, Mapping) else {},
        server_args=str(server_args).strip() if isinstance(server_args, str) else "",
        setup_commands=_clean_seq(setup),
    )


__all__ = [
    "Artifact",
    "Deliverable",
    "DeliverableRefused",
    "parse_deliverable",
]
