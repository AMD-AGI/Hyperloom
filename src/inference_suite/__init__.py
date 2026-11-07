# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``inference_suite`` -- resolves a serving framework's base launch and runs one inference round on it.

It builds only on ``hyperloom_common`` and third-party libraries; ``.importlinter`` holds the
contracts that keep it so, and names the modules other packages may import.

A host resolves a session's base in two steps: :func:`fetch_sources` reads every source once, the only step that
reaches the network or runs a source's own code, and :func:`resolve_base` reads that :class:`SourceSnapshot` alone.
The operator's own launch file enters both as the :class:`UserFile` :func:`parse_user_file` reads from its bytes;
:func:`fetch_response` reads one at a URL. A scriptable session's base is its script, which :func:`scriptable_base`
freezes with its content.

A runner runs one round with :func:`run_round`, and reaps a server a round left running with :func:`stop`. The round
names the staged tools it runs by :func:`manifest_sha256`, the digest of the tool cache's manifest;
:func:`verified_manifest_sha256` gives it only for a cache that verifies against the pins. Two hosts compare their
stagings by :func:`normalised_manifest`, a :class:`NormalisedManifest` that leaves every path out, and a host that
shares stagings finds the one of its pins under :func:`staging_key` (``python -m inference_suite publish``). ``install.sh
<tool-cache-dir>``, shipped in this package, is the published entry point that stages those tools for a runner or a
local setup: it runs ``python -m inference_suite stage`` under ``$PYTHON``, the interpreter rounds run under, prints
the manifest's sha256, stages nothing into a cache that already verifies, and exits 2 with ``staging_missing`` on a
damaged cache or a failed staging. A runner that runs
the round in another process runs ``python -m inference_suite run``, which exits :data:`EXIT_REFUSED` when it refuses
the config before the round starts. :func:`inferencex_root` is the staged InferenceX in a tool cache, and
:data:`CLIENT_FILE` its benchmark client under that root. A round's ``launch.json`` names the GPUs its server ran on
under :data:`GPU_MASK_ENV`.
"""

from __future__ import annotations

from inference_suite.cli import EXIT_REFUSED
from inference_suite.compose import GPU_MASK_ENV
from inference_suite.contracts.base_launch import LaunchQuery
from inference_suite.contracts.layout import RunDirError
from inference_suite.recipe.resolve import Miss, SourceSnapshot, resolve_base
from inference_suite.recipe.sources import SourceFetchError, fetch_response, fetch_sources
from inference_suite.recipe.user_file import UserFile, UserFileError, parse_user_file
from inference_suite.load.scriptable import scriptable_base
from inference_suite.run import RoundResult, run_round
from inference_suite.runtime.server import stop
from inference_suite.pins import INFERENCEX_CLIENT_COMMIT
from inference_suite.contracts.normalised import NormalisedManifest
from inference_suite.staging import (
    CLIENT_FILE,
    StagingError,
    inferencex_root,
    manifest_sha256,
    normalised_manifest,
    staging_key,
    verified_manifest_sha256,
)

__all__ = [
    "CLIENT_FILE",
    "EXIT_REFUSED",
    "GPU_MASK_ENV",
    "INFERENCEX_CLIENT_COMMIT",
    "LaunchQuery",
    "Miss",
    "NormalisedManifest",
    "RoundResult",
    "RunDirError",
    "SourceFetchError",
    "SourceSnapshot",
    "StagingError",
    "UserFile",
    "UserFileError",
    "fetch_response",
    "fetch_sources",
    "inferencex_root",
    "manifest_sha256",
    "normalised_manifest",
    "parse_user_file",
    "resolve_base",
    "run_round",
    "scriptable_base",
    "staging_key",
    "stop",
    "verified_manifest_sha256",
]
