# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where Hyperloom puts things in a session directory, as the Inspector needs to know it.

Spelled here, not imported: their sources of truth, :mod:`hyperloom.inference_optimizer.session.paths` and the grid
runner's ``hyperloom.orchestrator.actions.executors.grid._grid_base``, sit in packages that are to import the
Inspector (the optimizer's command line, to offer it as a subcommand), so importing them back would make a cycle the
architecture contracts (``.importlinter``) refuse. The contract tests (``tests/test_contracts_*.py``) hold every name here equal
to Hyperloom's.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

#: Top-level session subdirectories that belong to the application alone (``paths.PRIVATE_SESSION_SUBDIRS``): never
#: listed or served.
PRIVATE_SESSION_SUBDIRS: tuple[str, ...] = ("archive",)

#: The session's optimization state (``session_paths.state_path``).
STATE_FILE = "state.json"
#: Files whose presence means a directory holds a session that was started (``paths.SESSION_FILES``).
SESSION_FILES: tuple[str, ...] = ("manifest.json", STATE_FILE)

_DB_NAME = "coordinator.db"
#: The coordinator database every account of the group opens (``paths.shared_db_path_for``).
SHARED_DB = f"storage/{_DB_NAME}"
#: Written once a leg keeps the database outside the session directory (``paths.ledger_mark_path``); from then on one
#: found at :data:`SHARED_DB` is stale.
LEDGER_MARK = "runtime/ledger_private.json"
#: The directory an operator names for an account's private databases (``paths.ENV_LEDGER_DIR``), else one below
#: ``TMPDIR`` with this prefix and the account's uid (``paths.PRIVATE_DB_DIR_PREFIX``).
ENV_LEDGER_DIR = "HYPERLOOM_LEDGER_DIR"
_PRIVATE_DB_DIR_PREFIX = "hyperloom-ledger-"
_LEDGER_KEY_NAME_MAX = 40
_LEDGER_KEY_HASH_LEN = 16

#: The longest variant name a grid slot's directory keeps (``_grid_base._safe``).
_VARIANT_NAME_MAX = 60


def variant_dir_name(name: str) -> str:
    """The name the grid runner gives variant *name* in its slot's directory, ``v<NN>_<this>`` (``_grid_base._safe``):
    every character but a letter, a digit, ``-``, ``_`` and ``.`` made ``_``, and cut short."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:_VARIANT_NAME_MAX]


def _recorded_session_id(session_dir: Path) -> str:
    """The session id the manifest records, else the state file; ``""`` when neither is readable
    (``paths._recorded_session_id``)."""
    for name in SESSION_FILES:
        try:
            recorded = json.loads((session_dir / name).read_text(encoding="utf-8")).get("session_id")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(recorded, str) and recorded.strip():
            return recorded.strip()
    return ""


def private_db_path(session_dir: Path) -> Path:
    """The coordinator database this account keeps for the session outside it (``paths.private_db_path_for``).

    Raises:
        ValueError: When :data:`ENV_LEDGER_DIR` is not an absolute path.
    """
    named = os.environ.get(ENV_LEDGER_DIR, "").strip()
    if named and not os.path.isabs(named):
        raise ValueError(f"{ENV_LEDGER_DIR} must be an absolute path, not {named!r}")
    ledgers = (
        Path(named) if named else Path(os.environ.get("TMPDIR") or "/tmp") / f"{_PRIVATE_DB_DIR_PREFIX}{os.geteuid()}"
    )
    ident = _recorded_session_id(session_dir) or Path(os.path.realpath(session_dir)).name or "session"
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", ident)[:_LEDGER_KEY_NAME_MAX]
    key = f"{name}-{hashlib.sha256(ident.encode()).hexdigest()[:_LEDGER_KEY_HASH_LEN]}"
    return ledgers / key / _DB_NAME


def _is_own_file(path: Path) -> bool:
    """A regular file of this account's, the link at the end not followed (``paths.private_db_exists``)."""
    try:
        info = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid()


def coordinator_db(session_dir: Path) -> tuple[Path, str | None]:
    """The coordinator database to read for the session, as ``paths.db_path_for`` finds it, and why its tasks cannot
    be read, or ``None`` when they can.

    It is this account's own outside the session directory once that exists, else the one in it. When the session says
    it keeps its database outside (:data:`LEDGER_MARK`) and this account has none (another account's run, or a run in
    another container, whose ``TMPDIR`` is its own), the one left in the session is stale: a reason is returned, and the
    path is not to be read.
    """
    try:
        private: Path | None = private_db_path(session_dir)
    except ValueError as exc:
        private, unusable = None, str(exc)
    else:
        unusable = f"this account has none at {private}"
    if private is not None and _is_own_file(private):
        return private, None
    if not (session_dir / LEDGER_MARK).is_file():
        return session_dir / SHARED_DB, None
    reason = (
        "tasks not shown: the session keeps its coordinator database in a private ledger outside the session "
        f"directory, and {unusable}; set {ENV_LEDGER_DIR} to the ledger directory of the account that ran it"
    )
    return private or session_dir / SHARED_DB, reason
