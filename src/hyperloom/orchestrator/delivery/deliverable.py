# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The declared deliverable a specialist round hands back, and its frozen digests.

A round names what it changed -- tree, target paths, env and arg layers, setup
commands, whole-file artifacts -- rather than leaving the harness to infer it.
Digests are frozen where the work was validated, before transport, and any
digest the specialist supplies is discarded.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.session.paths import is_path_within
from hyperloom.orchestrator.delivery.manifest import ABSENT, TreeBaseline, file_digest
from hyperloom.orchestrator.source_snapshot import _safe_rel

#: Recorded in place of a pre-image hash when the target did not exist before
#: the round. Distinct from an empty hash, which reads as "not computed".
NO_PRE_IMAGE = "absent"


class DeliverableRefused(ValueError):
    """A declared deliverable cannot be trusted and must not be integrated."""


@dataclass(frozen=True)
class Artifact:
    """One whole-file deliverable.

    Attributes:
        target: Path within ``tree_id`` the file installs to.
        tree_id: Tree the target belongs to, not necessarily the round's own.
        source: Absolute path of the authored file, where it was validated.
        source_sha256: Frozen hash of ``source``, empty until
            :func:`freeze_digests` runs.
        pre_image_sha256: Frozen hash of what ``target`` held before the round,
            or :data:`NO_PRE_IMAGE` when it held nothing.
        kind: Free-form artifact kind label.
        description: Free-form human description.
    """

    target: str
    tree_id: str
    source: str
    source_sha256: str = ABSENT
    pre_image_sha256: str = ABSENT
    kind: str = ""
    description: str = ""

    @property
    def frozen(self) -> bool:
        """Whether both digests have been computed and can be checked against."""
        return bool(self.source_sha256) and bool(self.pre_image_sha256)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "target": self.target,
            "tree_id": self.tree_id,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "pre_image_sha256": self.pre_image_sha256,
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
    declared entry against the workspace sandbox and the allowlisted roots, and
    :func:`freeze_digests` hashes what that resolved to.

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


def freeze_digests(
    deliverable: Deliverable,
    *,
    baselines: Mapping[str, TreeBaseline],
    validated_roots: Sequence[Path | str] = (),
) -> Deliverable:
    """Compute and freeze every artifact digest at the validation site.

    The source hash comes from the file where the work was validated, the
    pre-image hash from that target's tree baseline rather than the live tree.

    Args:
        deliverable: The parsed deliverable.
        baselines: Pre-round baselines by tree id.
        validated_roots: Directories a source file may legitimately live in,
            being the round's worktree and workspace.

    Returns:
        Deliverable: The same deliverable with digests frozen.

    Raises:
        DeliverableRefused: When an artifact's source is unreadable, or lies
            outside ``validated_roots`` with no recorded pre-image.
    """
    roots = [Path(r) for r in validated_roots]
    frozen: list[Artifact] = []
    for artifact in deliverable.artifacts:
        source = Path(artifact.source)
        digest = file_digest(source)
        if not digest:
            raise DeliverableRefused(f"artifact source is unreadable: {artifact.source}")

        baseline = baselines.get(artifact.tree_id)
        rel = _safe_rel(artifact.target)
        entry = baseline.entry(rel) if baseline is not None and rel is not None else None
        if entry is None:
            if not any(is_path_within(source, root) for root in roots):
                raise DeliverableRefused(
                    f"artifact {artifact.target} was authored in place with no recorded pre-image; "
                    "its target has no baseline entry to check the install against"
                )
            pre_image = NO_PRE_IMAGE
        else:
            pre_image = entry.sha256 if entry.existed else NO_PRE_IMAGE

        frozen.append(
            Artifact(
                target=artifact.target,
                tree_id=artifact.tree_id,
                source=artifact.source,
                source_sha256=digest,
                pre_image_sha256=pre_image,
                kind=artifact.kind,
                description=artifact.description,
            )
        )
    return Deliverable(
        tree_id=deliverable.tree_id,
        targets=deliverable.targets,
        patches=deliverable.patches,
        artifacts=tuple(frozen),
        envs=dict(deliverable.envs),
        server_args=deliverable.server_args,
        setup_commands=deliverable.setup_commands,
    )


def mismatched_recorded_artifacts(records: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    """Return recorded artifacts whose source no longer matches its frozen digest.

    Args:
        records: Accepted artifact records, each with ``target``, ``source``
            and, once frozen, ``source_sha256``. A record carrying no digest
            predates freezing and is skipped.

    Returns:
        tuple[str, ...]: The targets that no longer match, sorted.
    """
    failed: list[str] = []
    for record in records:
        expected = str(record.get("source_sha256", ""))
        if not expected:
            continue
        if file_digest(Path(str(record["source"]))) != expected:
            failed.append(str(record["target"]))
    return tuple(sorted(failed))


__all__ = [
    "NO_PRE_IMAGE",
    "Artifact",
    "Deliverable",
    "DeliverableRefused",
    "freeze_digests",
    "parse_deliverable",
    "mismatched_recorded_artifacts",
]
