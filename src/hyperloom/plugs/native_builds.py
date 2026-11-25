# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``native_builds`` socket: a session's builds of a framework server that is built from its source, and the
option list of the server so built."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

NATIVE_BUILDS_SOCKET = "native_builds"


@runtime_checkable
class NativeBuilds(Protocol):
    """Builds the server of a session's framework from its checkout.

    What a caller asks of the builds and of the option list is declared by the layer that uses them, which the kernel
    must not import, so they cross this socket as ``object``.
    """

    def open_builds(self, session_dir: Path, framework: str) -> object | None:
        """The builds of ``framework``'s server in the session at ``session_dir``, the accepted one first on this
        process's ``PATH``; ``None`` when ``framework`` builds none."""

    def record_options(self, session_dir: Path, framework: str) -> Path | None:
        """Keep the options the server first on this process's ``PATH`` takes in the session at ``session_dir``,
        unless it keeps them already; the file, or ``None`` when ``framework`` builds no server."""

    def options(self, session_dir: Path) -> object | None:
        """The option list the session at ``session_dir`` keeps; ``None`` when it keeps none."""
