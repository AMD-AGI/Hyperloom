# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The implementation of the ``native_builds`` socket (:mod:`hyperloom.plugs.native_builds`): a single-branch
session's native builds (:mod:`~hyperloom.measure.runner.native.session_builds`) and the option list it keeps of the
server it built (:mod:`~hyperloom.measure.runner.native.session_options`).

Every run starts this plug, so the engine is imported by the call that needs it: a session whose server is the image's
never loads it.
"""

from __future__ import annotations

import os
from pathlib import Path

from hyperloom.plugs.socket import PlugContext

__all__ = ["SessionNativeBuildsPlug", "create"]


class SessionNativeBuildsPlug:
    """The session's builds and option list, on this process's environment."""

    def open_builds(self, session_dir: Path, framework: str) -> object | None:
        """As :func:`~hyperloom.measure.runner.native.session_builds.start_native_builds`."""
        from hyperloom.measure.runner.native.session_builds import start_native_builds

        return start_native_builds(session_dir, framework, os.environ)

    def record_options(self, session_dir: Path, framework: str) -> Path | None:
        """As :func:`~hyperloom.measure.runner.native.session_options.record_session_options`."""
        from hyperloom.measure.runner.native.session_options import record_session_options

        return record_session_options(session_dir, framework, os.environ)

    def options(self, session_dir: Path) -> object | None:
        """As :func:`~hyperloom.measure.runner.native.session_options.session_options`."""
        from hyperloom.measure.runner.native.session_options import session_options

        return session_options(session_dir)


def create(context: PlugContext) -> SessionNativeBuildsPlug:
    """Factory the built-in registration points at."""
    return SessionNativeBuildsPlug()
