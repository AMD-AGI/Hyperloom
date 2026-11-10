# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Who may read ``state.json``: the accounts of the run's group, unless the operator's launch pins hold a credential.

``state.json`` keeps the operator's ``--extra-env`` and ``--server-args`` in plaintext because a resume re-exports them
from there, and the launches recorded in it carry them, so those pins decide its mode (see
:mod:`hyperloom.common.session_files`); the rest is the run's own record. The next leg of a run may be another account
of the group on shared storage, which must read it back.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperloom.common.session_files import OPERATOR_PINS, pins_hold_credentials
from hyperloom_common.io import WriteOptions, atomic_write_json


def write_state(path: Path, document: Mapping[str, Any]) -> None:
    """Atomically write ``document`` as ``state.json``: ``664`` below shared storage, or ``600`` when the operator's
    pins (those it records and those this run exports) hold a credential."""
    pins = {name: document.get(name) for name in OPERATOR_PINS}
    atomic_write_json(path, document, options=WriteOptions(make_parents=True, shared=not pins_hold_credentials(pins)))
