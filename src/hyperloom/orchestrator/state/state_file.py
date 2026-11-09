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
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import holds_credentials, operator_extra_env
from hyperloom.common.session_files import shared_options
from hyperloom.inference_optimizer.session.paths import private_db_path_for, shared_db_path_for
from hyperloom_common.io import atomic_write_json
from hyperloom_common.shared_storage import SHARED_FILE_MODE

_OPERATOR_PINS = ("operator_extra_env", "operator_server_args")
#: Where a run exports the operator's ``--server-args`` (the pins of ``--extra-env`` are exported by name and as a blob).
_SERVER_ARGS_ENV = "INFERENCE_OPTIMIZER_SERVER_ARGS"
_PRIVATE_FILE_MODE = 0o600

log = logging.getLogger(__name__)


def write_state(path: Path, document: Mapping[str, Any]) -> None:
    """Atomically write ``document`` as ``state.json``: ``664``, or ``600`` when the operator's pins hold a credential."""
    atomic_write_json(path, document, options=shared_options({name: document.get(name) for name in _OPERATOR_PINS}))


def database_file_mode(state: Any) -> int:
    """The mode of the session's coordinator database: ``664``, or ``600`` when the operator's pins hold a credential,
    as ``state.json`` is: the bus stores them in the launches it carries.

    The pins are those ``state`` records and those this run exports: a resume with ``--reset-state`` records none, and
    still launches with the ones it exports.
    """
    recorded = {name: getattr(state, name, None) for name in _OPERATOR_PINS}
    exported = {
        "operator_extra_env": operator_extra_env(),
        "operator_server_args": os.environ.get(_SERVER_ARGS_ENV, ""),
    }
    return _PRIVATE_FILE_MODE if holds_credentials([recorded, exported]) else SHARED_FILE_MODE


def database_target(session_dir: Path, state: Any) -> tuple[Path, int]:
    """Where this account opens the session's coordinator database, and the most its mode may be.

    With no credential in the pins it is the database in the session directory (``664``), so the next account of the
    group opens the same one. With one, and from then on once this account has made it, it is the account's own, outside
    the session directory (:func:`private_db_path_for`), ``600``: a database in a directory the group can write is
    reached by path, which another account of the group can rename entries in. A session resumed that way by an account
    that has no database of its own starts with an empty one: what the earlier legs' database recorded (the bus events,
    the task and lease records, the cursors) is not carried over, and a warning says so when it is in the session
    directory; one another account kept outside the session directory cannot be seen at all. ``state.json`` still
    carries the optimization state.
    """
    file_mode = database_file_mode(state)
    private = private_db_path_for(session_dir)
    if file_mode == SHARED_FILE_MODE and not private.exists():
        return shared_db_path_for(session_dir), file_mode
    earlier = shared_db_path_for(session_dir)
    if not private.exists() and earlier.exists():
        log.warning(
            "the session holds a credential, so this account keeps its own coordinator database (%s) and starts it "
            "empty: what the database an earlier leg left at %s recorded (bus events, task and lease records, "
            "cursors) is not carried over; check that no work of an earlier leg is still running",
            private,
            earlier,
        )
    return private, _PRIVATE_FILE_MODE
