# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Who may read what the operator's launch pins end up in: the accounts of the run's group, unless they hold a credential.

``state.json`` keeps the operator's ``--extra-env`` and ``--server-args`` in plaintext because a resume re-exports them
from there, and the message bus stores the launches that carry them (refused argv, launch evidence), so those two
fields decide the mode of ``state.json`` and of the coordinator database (see :mod:`hyperloom.common.session_files`);
the rest is the run's own record.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperloom.common.session_files import OPERATOR_PINS, pins_hold_credentials, shared_options
from hyperloom.inference_optimizer.session.paths import private_db_exists, private_db_path_for, shared_db_path_for
from hyperloom.inference_optimizer.session.resume_guard import require_own_ledger
from hyperloom_common.io import atomic_write_json
from hyperloom_common.shared_storage import SHARED_FILE_MODE

_PRIVATE_FILE_MODE = 0o600

log = logging.getLogger(__name__)


def write_state(path: Path, document: Mapping[str, Any]) -> None:
    """Atomically write ``document`` as ``state.json``: ``664``, or ``600`` when the operator's pins hold a credential."""
    atomic_write_json(path, document, options=shared_options({name: document.get(name) for name in OPERATOR_PINS}))


def database_file_mode(state: Any) -> int:
    """The mode of the session's coordinator database: ``664``, or ``600`` when the operator's pins hold a credential,
    as ``state.json`` is: the bus stores them in the launches it carries.

    The pins are those ``state`` records and those this run exports: a resume with ``--reset-state`` records none, and
    still launches with the ones it exports.
    """
    return _PRIVATE_FILE_MODE if pins_hold_credentials(_recorded(state)) else SHARED_FILE_MODE


def _recorded(state: Any) -> dict[str, Any]:
    """The fields of ``state`` that decide whether a credential session has a ledger to resume: the operator's pins and
    the marks of an earlier leg's work (a counted tick, an integrate window left open, a stack)."""
    return {
        **{name: getattr(state, name, None) for name in OPERATOR_PINS},
        "tick": getattr(state, "tick", 0),
        "pending_integrate": getattr(state, "pending_integrate", None),
        "optimization_stack": getattr(state, "optimization_stack", None),
    }


def database_target(session_dir: Path, state: Any) -> tuple[Path, int]:
    """Where this account opens the session's coordinator database, and the most its mode may be.

    With no credential in the pins it is the database in the session directory (``664``), so the next account of the
    group opens the same one. With one, and from then on once this account has made it, it is the account's own, outside
    the session directory (:func:`private_db_path_for`), ``600``: a database in a directory the group can write is
    reached by path, which another account of the group can rename entries in.

    A session that holds a credential and has run before cannot be resumed by an account that has no database of its own
    where none is left in the session directory either (another account's resume, or this account's in a new container,
    since the default is below the container's ``TMPDIR``): :func:`require_own_ledger` refuses it, naming
    ``HYPERLOOM_LEDGER_DIR``, because the earlier legs' leases and tasks cannot be inspected and work they left running
    would go unseen. A first leg, and a session that a credential-free leg left a database for in the session directory
    (which the resume guard inspects), start an empty ledger; the second logs a warning that the earlier database is not
    read. ``state.json`` still carries the optimization state.
    """
    file_mode = database_file_mode(state)
    private = private_db_path_for(session_dir)
    made = private_db_exists(session_dir)
    if file_mode == SHARED_FILE_MODE and not made:
        return shared_db_path_for(session_dir), file_mode
    if not made:
        require_own_ledger(session_dir, _recorded(state))
    earlier = shared_db_path_for(session_dir)
    if not made and earlier.exists():
        log.warning(
            "the session holds a credential, so this account keeps its own coordinator database (%s); the one an "
            "earlier leg left at %s is not read",
            private,
            earlier,
        )
    return private, _PRIVATE_FILE_MODE
