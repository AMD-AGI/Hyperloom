# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run a store's writer and reader as one unprivileged account, so a mode that shuts its owner out cannot pass a test
only because the test runs as root (root reads past every permission bit)."""

from __future__ import annotations

import os
import sys
import traceback
from collections.abc import Callable
from pathlib import Path

#: The account a store is written and read as when the tests run as root: ``nobody``, which holds no privilege.
UNPRIVILEGED = 65534


def as_store_owner(directory: Path, body: Callable[[Path], None]) -> int:
    """Call ``body`` with a directory it owns, as an account that cannot read past a permission bit; return that uid.

    Run as root, ``body`` runs in a child that has dropped to :data:`UNPRIVILEGED` (no supplementary groups) inside
    ``directory``, which is handed to it, and gets the relative path ``.``, so the parents it cannot enter do not matter.
    Otherwise it runs here, as the account the tests already run as.
    """
    if os.geteuid() != 0:
        body(directory)
        return os.geteuid()
    os.chown(directory, UNPRIVILEGED, UNPRIVILEGED)
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child reports through its exit status
        status = 1
        try:
            os.chdir(directory)
            os.setgroups([])
            os.setgid(UNPRIVILEGED)
            os.setuid(UNPRIVILEGED)
            body(Path("."))
            status = 0
        except Exception:  # noqa: BLE001 - the child's outermost handler; its traceback goes to captured stderr
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
        finally:
            os._exit(status)
    _, wait_status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(wait_status) == 0, "the store's own account could not write and read it back"
    return UNPRIVILEGED
