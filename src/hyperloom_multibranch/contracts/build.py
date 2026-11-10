# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The build a multi-branch process runs: what a session's manifest records and what a resume compares with.

It sits with the contracts because both the scheduler (the manifest) and the toolbox (the baseline's environment record)
name it, and neither may import the other.
"""

from __future__ import annotations

import importlib.metadata
import os
from dataclasses import dataclass

from hyperloom.common.provenance import detect_code_revision

#: The distribution this package ships in (pyproject.toml ``[project] name``).
MB_DISTRIBUTION = "hyperloom-inference_optimizer"


@dataclass(frozen=True)
class CodeVersion:
    """The installed package version and the source revision :func:`detect_code_revision` reports: the source
    tree's commit, else the revision baked into the image; empty where there is none."""

    package: str
    git_sha: str


def code_version() -> CodeVersion:
    """This process's :class:`CodeVersion`. Blocks on git."""

    try:
        package = importlib.metadata.version(MB_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        package = ""
    return CodeVersion(package, detect_code_revision(os.environ))


__all__ = ["MB_DISTRIBUTION", "CodeVersion", "code_version"]
