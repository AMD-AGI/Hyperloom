# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``MB-manifest.json``: what a session records about itself before its first phase.

The session writes the document once, as its first act, and reads it back wherever a later run over the same directory,
or CLOSE's breakdown, needs what that session ran on; the frontier policy it chose is the one field a running session
rewrites, and a resumed session may record the Claw session it was started without (:func:`kept_claw_session_id`).
Readers go through :func:`read_manifest` and writers through :meth:`SessionManifest.write`, so the document
has one shape and one owner.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hyperloom_common.io import WriteOptions, atomic_write_json
from hyperloom_common.jsonio import read_json_strict
from hyperloom_multibranch.blackboard.codec import decode, encode
from hyperloom_multibranch.blackboard.knowledge.breakdown import claw_session_id
from hyperloom_multibranch.contracts.build import MB_DISTRIBUTION, CodeVersion, code_version
from hyperloom_multibranch.scheduling.launch_isolation import LaunchIsolation
from hyperloom_multibranch.scheduling.settings import RunSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageCheckSummary:
    """:meth:`~hyperloom.common.outside_facts.image_check.ImageCheckReport.summary` as the manifest records it, in
    the image check's own shape, which the manifest carries and does not read."""

    fields: Mapping[str, object]

    def to_document(self) -> dict[str, object]:
        return dict(self.fields)

    @classmethod
    def from_document(cls, document: object) -> ImageCheckSummary:
        if not isinstance(document, Mapping):
            raise TypeError(f"expected an image check summary, got {document!r}")
        return cls(dict(document))


#: The summary a manifest without one reads as.
NO_IMAGE_CHECK = ImageCheckSummary({})


@dataclass(frozen=True)
class SessionManifest:
    """The manifest document: the build the session ran on, the rule pack it read, the window it was given, the
    settings it ran under, the metric it optimized, the policy versions it scored with, what the image check found and
    how its agents were isolated.

    ``rule_pack`` is :func:`hyperloom.common.outside_facts.rule_pack.rule_pack_identity`'s version and digest, so a session's
    prompts can be told apart from another's when the pack moves under a single build. ``image_check`` is the
    :meth:`~hyperloom.common.outside_facts.image_check.ImageCheckReport.summary` of the check the session ran before its
    first phase: its mode, the pack digest, the counts and the facts that failed; a manifest without it reads as
    empty, so a session directory an earlier build wrote still reads back far enough to be refused for its build.
    ``agent_isolation`` is how the session's peers launched its specialists, agent and dev sandboxes and tests
    (:mod:`~hyperloom_multibranch.scheduling.launch_isolation`), and why a launch on its host was not isolated; ``None``
    in a manifest an earlier build wrote. ``claw_session_id`` is the Claw session that hosts the session, from
    ``CLAW_SESSION_ID`` at its start (:func:`~hyperloom_multibranch.blackboard.knowledge.breakdown.claw_session_id`), so
    a collector of running sessions can tell whose a directory is before CLOSE writes the breakdown; ``None`` for a run
    Claw did not host and in a manifest an earlier build wrote."""

    code_version: CodeVersion
    rule_pack: dict[str, str]
    session_id: str
    started_at: float
    deadline: float
    settings: RunSettings
    metric: str
    policy_versions: dict[str, str]
    image_check: ImageCheckSummary = NO_IMAGE_CHECK
    agent_isolation: LaunchIsolation | None = None
    claw_session_id: str | None = None

    def write(self, path: Path) -> None:
        """Write the document to ``path``, replacing whatever is there; ``664``, as a later leg of a hosted run, maybe
        another account of the group, reads it back to take the session over."""

        atomic_write_json(path, encode(self), options=WriteOptions(shared=True))


def read_manifest(path: Path) -> SessionManifest:
    """The manifest at ``path``; raises ``OSError`` where there is none, and ``ValueError``, ``TypeError`` or
    ``KeyError`` where what is there is not one."""

    return decode(SessionManifest, read_json_strict(path, require_dict=True))


def kept_claw_session_id(recorded: str | None, environ: Mapping[str, str], where: str) -> str | None:
    """The Claw session a session that ``recorded`` one in ``where`` keeps when it runs again under ``environ``: the
    recorded one, with a warning when ``environ`` names another, as a resumed session does not change its identity;
    ``environ``'s (:func:`~hyperloom_multibranch.blackboard.knowledge.breakdown.claw_session_id`) when it recorded
    none."""

    given = claw_session_id(environ)
    if not recorded:
        return given
    if given is not None and given != recorded:
        logger.warning(
            "%s records the Claw session %r and this run is given %r; the session keeps %r",
            where,
            recorded,
            given,
            recorded,
        )
    return recorded


__all__ = [
    "MB_DISTRIBUTION",
    "CodeVersion",
    "ImageCheckSummary",
    "SessionManifest",
    "code_version",
    "kept_claw_session_id",
    "read_manifest",
]
