# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a round's archive actually holds, recorded one copy at a time.

The archive collector drops ``runs/`` wholesale and retains ``reports/``, so a
deliverable is resolvable by a later reader only once its copy lands under
``reports/``. A record is therefore appended by the copy that made it, never
derived ahead of the write: a path named before the copy exists is a path every
consumer resolves to nothing, which is worse than naming none at all.

Each record carries the role the file plays, because a reader that wants the
launch config a bench started from should not have to recognise it by filename,
and because a round applies any number of patches under names its specialist
chose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: A unified diff the round authored -- applied, or attempted and left behind.
#: The one repeating role: a round applies any number of patches.
ROLE_PATCH = "patch"

#: The ``specialist_done`` payload the round's specialist handed back.
ROLE_SPECIALIST_RESULT = "specialist_result"

#: The prompt the specialist was dispatched with.
ROLE_PROMPT = "prompt"

#: The materialized launch config the round's bench was started from.
ROLE_LAUNCH_CONFIG = "launch_config"

#: A tail of the server's own log, where a failed boot writes its traceback.
ROLE_SERVER_LOG = "server_log"


@dataclass(frozen=True)
class ArchivedFile:
    """One copy that landed in the archive, and what it is.

    Attributes:
        path: Session-relative POSIX path of the copy, not of the original.
        role: What the file is to a reader; one of the ``ROLE_*`` constants.
    """

    path: str
    role: str

    def to_dict(self) -> dict[str, str]:
        """Serialize to a JSON-safe dict."""
        return {"path": self.path, "role": self.role}


class RoundArchive:
    """The copies one round's archive holds, in the order they landed.

    A collector, not a plan: :meth:`record` is called by the copy, so the
    archive never names a file the copy refused or never attempted.
    """

    def __init__(self, session_dir: Path | str) -> None:
        """Open an empty archive record rooted at ``session_dir``.

        Args:
            session_dir: The session root every recorded path is relative to.
        """
        self._root = Path(session_dir)
        self._files: list[ArchivedFile] = []

    def record(self, role: str, dest: Path) -> None:
        """Record a copy that has already landed at ``dest``.

        Args:
            role: One of the ``ROLE_*`` constants.
            dest: Absolute path of the copy, under the archive root.
        """
        self._files.append(ArchivedFile(path=dest.relative_to(self._root).as_posix(), role=role))

    def path_for(self, role: str) -> str:
        """Return the session-relative path recorded for a single-valued role.

        Args:
            role: One of the single-valued ``ROLE_*`` constants;
                :data:`ROLE_PATCH` repeats and wants :meth:`paths_for`.

        Returns:
            str: The recorded path, ``""`` when no such copy landed.
        """
        return next((f.path for f in self._files if f.role == role), "")

    def paths_for(self, role: str) -> tuple[str, ...]:
        """Return every session-relative path recorded for ``role``, in order.

        Args:
            role: One of the ``ROLE_*`` constants.

        Returns:
            tuple[str, ...]: The recorded paths, empty when none landed.
        """
        return tuple(f.path for f in self._files if f.role == role)

    def to_list(self) -> list[dict[str, str]]:
        """Serialize every record to a JSON-safe ``{"path", "role"}`` list."""
        return [f.to_dict() for f in self._files]


__all__ = [
    "ROLE_LAUNCH_CONFIG",
    "ROLE_PATCH",
    "ROLE_PROMPT",
    "ROLE_SERVER_LOG",
    "ROLE_SPECIALIST_RESULT",
    "ArchivedFile",
    "RoundArchive",
]
