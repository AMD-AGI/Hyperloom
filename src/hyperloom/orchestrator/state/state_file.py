# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Who may read ``state.json``: the accounts of the run's group, unless the operator's launch pins hold a credential.

``state.json`` keeps the operator's ``--extra-env`` and ``--server-args`` in plaintext because a resume re-exports them
from there, so those two fields decide (see :mod:`hyperloom.common.session_files`); the rest is the run's own record.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperloom.common.session_files import shared_options
from hyperloom_common.io import atomic_write_json

_OPERATOR_PINS = ("operator_extra_env", "operator_server_args")


def write_state(path: Path, document: Mapping[str, Any]) -> None:
    """Atomically write ``document`` as ``state.json``: ``664``, or ``600`` when the operator's pins hold a credential."""
    atomic_write_json(path, document, options=shared_options({name: document.get(name) for name in _OPERATOR_PINS}))
