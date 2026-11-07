# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``inference_suite`` -- resolves a serving framework's base launch and runs one inference round on it.

It builds only on ``hyperloom_common`` and third-party libraries; ``.importlinter`` holds the
contracts that keep it so, and names the modules other packages may import.

A host resolves a session's base in two steps: :func:`fetch_sources` reads every source once, the only step that
reaches the network or runs a source's own code, and :func:`resolve_base` reads that :class:`SourceSnapshot` alone.
The operator's own launch file enters both as the :class:`UserFile` :func:`parse_user_file` reads from its bytes;
:func:`fetch_response` reads one at a URL.

A runner runs one round with :func:`run_round`, and reaps a server a round left running with :func:`stop`. The round
names the staged tools it runs by :func:`manifest_sha256`, the digest of the tool cache's manifest. A runner that runs
the round in another process runs ``python -m inference_suite run``, which exits :data:`EXIT_REFUSED` when it refuses
the config before the round starts.
"""

from __future__ import annotations

from inference_suite.cli import EXIT_REFUSED
from inference_suite.contracts.base_launch import LaunchQuery
from inference_suite.contracts.layout import RunDirError
from inference_suite.recipe.resolve import Miss, SourceSnapshot, resolve_base
from inference_suite.recipe.sources import SourceFetchError, fetch_response, fetch_sources
from inference_suite.recipe.user_file import UserFile, UserFileError, parse_user_file
from inference_suite.run import RoundResult, run_round
from inference_suite.runtime.server import stop
from inference_suite.staging import manifest_sha256

__all__ = [
    "EXIT_REFUSED",
    "LaunchQuery",
    "Miss",
    "RoundResult",
    "RunDirError",
    "SourceFetchError",
    "SourceSnapshot",
    "UserFile",
    "UserFileError",
    "fetch_response",
    "fetch_sources",
    "manifest_sha256",
    "parse_user_file",
    "resolve_base",
    "run_round",
    "stop",
]
