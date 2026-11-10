# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Who may read what a leg writes into a session: the accounts of the run's group, unless it holds a credential.

The next leg of a run may be another account of the group on shared storage, so a session file is written ``664``
(:data:`~hyperloom_common.shared_storage.SHARED_FILE_MODE`). A document that holds a credential (see
:func:`~hyperloom.common.env_safety.holds_credentials`) is written owner-only instead, and another account fails
reading it rather than being handed the credential.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from hyperloom.common.env_safety import holds_credentials, operator_extra_env
from hyperloom_common.io import WriteOptions

#: The fields of ``state.json`` that keep the operator's launch pins in plaintext.
OPERATOR_PINS = ("operator_extra_env", "operator_server_args")
#: Where a run exports the operator's ``--server-args`` (the pins of ``--extra-env`` are exported by name and as a blob).
SERVER_ARGS_ENV = "INFERENCE_OPTIMIZER_SERVER_ARGS"


def shared_options(document: object) -> WriteOptions:
    """The write options of ``document``: ``664`` below shared storage, or ``600`` when it holds a credential."""
    return WriteOptions(make_parents=True, shared=not holds_credentials(document))


def pins_hold_credentials(state: Mapping[str, Any]) -> bool:
    """Whether the operator's pins hold a credential: those ``state`` records (a mapping with the fields of
    :data:`OPERATOR_PINS`, as ``state.json`` is) and those this run exports (a resume with ``--reset-state`` records
    none, and still launches with the ones it exports)."""
    recorded = {name: state.get(name) for name in OPERATOR_PINS}
    exported = {
        "operator_extra_env": operator_extra_env(),
        "operator_server_args": os.environ.get(SERVER_ARGS_ENV, ""),
    }
    return holds_credentials([recorded, exported])
