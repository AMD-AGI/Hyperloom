# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``native_builds`` socket: a session's builds of a framework server that is built from its source."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

NATIVE_BUILDS_SOCKET = "native_builds"


@runtime_checkable
class NativeBuilds(Protocol):
    """Builds the server of a session's framework from its checkout.

    What a caller asks of the builds is declared by the layer that uses them, which the kernel must not import, so
    they cross this socket as ``object``.
    """

    def open_builds(self, session_dir: Path, framework: str) -> object | None:
        """The builds of ``framework``'s server in the session at ``session_dir``, the accepted one first on this
        process's ``PATH``; ``None`` when ``framework`` builds none."""
