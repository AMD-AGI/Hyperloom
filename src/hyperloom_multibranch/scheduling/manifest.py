# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``MB-manifest.json``: what a session records about itself before its first phase.

The session writes the document once, as its first act, and reads it back wherever a later run over the same directory,
or CLOSE's breakdown, needs what that session ran on; the frontier policy it chose is the one field a running session
rewrites. Readers go through :func:`read_manifest` and writers through :meth:`SessionManifest.write`, so the document
has one shape and one owner.
"""

from __future__ import annotations

import importlib.metadata
import os
from dataclasses import dataclass
from pathlib import Path

from hyperloom.common.io import atomic_write_json
from hyperloom.common.jsonio import read_json_strict
from hyperloom.common.provenance import detect_code_revision
from hyperloom_multibranch.notebook.codec import decode, encode
from hyperloom_multibranch.scheduling.settings import RunSettings, Workload

#: The distribution this package ships in (pyproject.toml ``[project] name``).
MB_DISTRIBUTION = "hyperloom-inference_optimizer"


@dataclass(frozen=True)
class CodeVersion:
    """The installed package version and the source revision :func:`detect_code_revision` reports: the source
    tree's commit, else the revision baked into the image; empty where there is none."""

    package: str
    git_sha: str


def code_version() -> CodeVersion:
    """This scheduler's :class:`CodeVersion`. Blocks on git."""

    try:
        package = importlib.metadata.version(MB_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        package = ""
    return CodeVersion(package, detect_code_revision(os.environ))


@dataclass(frozen=True)
class SessionManifest:
    """The manifest document: the build the session ran on, the rule pack it read, the window it was given, the
    settings it ran under, the metric it optimized and the policy versions it scored with.

    ``rule_pack`` is :func:`hyperloom.common.outside_facts.rule_pack.rule_pack_identity`'s version and digest, so a session's
    prompts can be told apart from another's when the pack moves under a single build."""

    code_version: CodeVersion
    rule_pack: dict[str, str]
    session_id: str
    started_at: float
    deadline: float
    settings: RunSettings
    metric: str
    policy_versions: dict[str, str]

    def write(self, path: Path) -> None:
        """Write the document to ``path``, replacing whatever is there."""

        atomic_write_json(path, encode(self))


def read_manifest(path: Path) -> SessionManifest:
    """The manifest at ``path``; raises ``OSError`` where there is none, and ``ValueError``, ``TypeError`` or
    ``KeyError`` where what is there is not one."""

    return decode(SessionManifest, read_json_strict(path, require_dict=True))


def read_workload(path: Path) -> Workload:
    """The workload the session at ``path`` ran, read on its own: a PRELUDE is reused across builds, and the rest of
    an earlier manifest need not decode for this one to be the same workload. Raises as :func:`read_manifest` does."""

    return decode(Workload, read_json_strict(path, require_dict=True)["settings"]["workload"])


__all__ = ["MB_DISTRIBUTION", "CodeVersion", "SessionManifest", "code_version", "read_manifest", "read_workload"]
