# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Who may read ``state.json``: the accounts of the run's group, unless the operator's launch pins hold a credential.

A hosted run's next leg may be another account of the group, and a resume reads ``state.json``, so the file is written
``664`` (:data:`~hyperloom_common.shared_storage.SHARED_FILE_MODE`). It keeps the operator's ``--extra-env`` and
``--server-args`` in plaintext because a resume re-exports them from there; when one of them is a credential the file
stays owner-only, and another account's resume fails reading it rather than being handed the credential.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import holds_credentials
from hyperloom_common.io import WriteOptions, atomic_write_json

_OPERATOR_PINS = ("operator_extra_env", "operator_server_args")


def write_state(path: Path, document: Mapping[str, Any]) -> None:
    """Atomically write ``document`` as ``state.json``: ``664``, or ``600`` when the operator's pins hold a credential."""
    pins = {name: document.get(name) for name in _OPERATOR_PINS}
    atomic_write_json(path, document, options=WriteOptions(make_parents=True, shared=not holds_credentials(pins)))
