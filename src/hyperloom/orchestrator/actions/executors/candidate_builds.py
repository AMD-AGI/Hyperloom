# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What integration asks of a session whose framework server is built from source: the build of a candidate's tree, and
making a build the one every later round runs.

The builds themselves are the measurement layer's (:mod:`hyperloom.measure.runner.native.session_builds`); this module
declares what an integration needs of them, so the entry point that opens a session hands them in
(:func:`~hyperloom.inference_optimizer.native_builds.start_candidate_builds`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

__all__ = ["CandidateBuilds", "TreeBuild", "TreeBuildStatus"]


class TreeBuildStatus(Enum):
    """What a build of a tree came to: built, the candidate's compile error, or the machine's failure."""

    BUILT = "built"
    COMPILE_ERROR = "compile_error"
    INFRA = "infra"


@dataclass(frozen=True)
class TreeBuild:
    """One build of a tree: what it came to, its key (none for a tree no build may take), whether another request built
    it first, the compile's failure or the machine's reason, and the directory of its executables once it is built."""

    status: TreeBuildStatus
    key: str = ""
    reused: bool = False
    detail: str = ""
    bin: Path | None = None

    def to_record(self) -> dict[str, Any]:
        """The build as a round's record names it."""
        return {"status": self.status.value, "key": self.key, "reused": self.reused, "detail": self.detail[-2000:]}


class CandidateBuilds(Protocol):
    """A session's builds of its framework's server, from the framework checkout ``checkout``."""

    checkout: Path

    def build_tree(self, paths: Sequence[str] = ()) -> TreeBuild:
        """The build of the framework checkout's ``HEAD`` with ``paths`` as its working tree has them."""

    def accept(self, build: TreeBuild) -> None:
        """Make ``build`` the one every later round of the session runs."""
