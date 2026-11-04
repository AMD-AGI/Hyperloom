# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A temporary directory of the test's own, so what the server writes below ``tempfile.gettempdir()`` is the test's.

Newer aiter writes the merged view of a variable that lists several tuned tables to
``tempfile.gettempdir()/aiter_configs`` (``AITER_CONFIG.update_config_files`` in ``aiter/jit/core.py``), where the
image's aiter writes it to the literal ``/tmp/aiter_configs`` that :mod:`hyperloom.measure.aiter_merge_hook` redirects. Local
sandboxes share ``/tmp``, so two tests that give one variable different tables would overwrite each other's merged file
there; the hook finds no literal to redirect in that aiter. A server whose environment names no temporary directory
(``TMPDIR``, ``TEMP`` or ``TMP``) is therefore given one of the test's own for the test, removed afterwards.

The directory is made beside the runner's own temporary files, not below the test's work directory: that can be deep
in a shared file system, and a server makes its Unix sockets below ``TMPDIR`` (a socket's path is at most 107 bytes).
It goes into the runner's own environment, which the server inherits, not into the sheet's ``env_overrides``: a sheet
may not set ``TMPDIR`` (``hyperloom.common.env_safety``). The compiled-code cache's key leaves these variables out
(``measure.jit.key``), so no key moves.
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from collections.abc import Iterator, MutableMapping
from pathlib import Path

from hyperloom.measure.runner.own_jit import point_configs, server_environments
from hyperloom.measure.socket import TestCall

__all__ = ["OWN_TMP_PREFIX", "TEMP_VARIABLES", "own_tmpdir"]

#: What ``tempfile.gettempdir()`` reads, in its order.
TEMP_VARIABLES = ("TMPDIR", "TEMP", "TMP")
OWN_TMP_PREFIX = "hl-test-tmp-"


@contextlib.contextmanager
def own_tmpdir(call: TestCall, environ: MutableMapping[str, str]) -> Iterator[Path | None]:
    """Set ``TMPDIR`` in ``environ`` (what the server inherits) to a new directory of the test's own while the test
    runs, then put ``environ`` back and remove the directory; yield it, or ``None`` when the environment of a point's
    server already names a temporary directory (that one is left as it is)."""
    servers = server_environments(call, point_configs(call), environ)
    if any((server.get(name) or "").strip() for server in servers for name in TEMP_VARIABLES):
        yield None
        return
    # Made before TMPDIR is set: this also fixes the runner's own tempfile directory (tempfile caches it on first use), so
    # the runner never puts a file of its own in the directory that is removed.
    directory = Path(tempfile.mkdtemp(prefix=OWN_TMP_PREFIX))
    before = environ.get(TEMP_VARIABLES[0])
    environ[TEMP_VARIABLES[0]] = str(directory)
    try:
        yield directory
    finally:
        if before is None:
            environ.pop(TEMP_VARIABLES[0], None)
        else:
            environ[TEMP_VARIABLES[0]] = before
        shutil.rmtree(directory, ignore_errors=True)
