# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Who may read what the operator's launch pins end up in: the accounts of the run's group, unless they hold a credential.

``state.json`` keeps the operator's ``--extra-env`` and ``--server-args`` in plaintext because a resume re-exports them
from there, and the message bus stores the launches that carry them (refused argv, launch evidence), so those two
fields decide the mode of ``state.json`` and of the coordinator database (see :mod:`hyperloom.common.session_files`);
the rest is the run's own record.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import holds_credentials, operator_extra_env
from hyperloom.common.session_files import shared_options
from hyperloom_common.io import atomic_write_json
from hyperloom_common.shared_storage import SHARED_FILE_MODE

_OPERATOR_PINS = ("operator_extra_env", "operator_server_args")
#: Where a run exports the operator's ``--server-args`` (the pins of ``--extra-env`` are exported by name and as a blob).
_SERVER_ARGS_ENV = "INFERENCE_OPTIMIZER_SERVER_ARGS"
_PRIVATE_FILE_MODE = 0o600


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
