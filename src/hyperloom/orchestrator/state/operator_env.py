# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Keep the operator's ``--extra-env`` pins out of the group-readable ``state.json``.

``state.json`` is group-readable (the retest runner reads it) but the pins may be credentials. The full values live
in an owner-only sidecar that ``--resume`` reads; ``state.json`` carries them masked. A ``state.json`` written before
the sidecar existed still holds them itself and is read as is; the next save moves them to the sidecar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import redact_secret_env_values
from hyperloom_common.io import WriteOptions, atomic_write_json
from hyperloom_common.jsonio import read_json_strict

#: Owner-only file beside state.json holding the pins in full (never group-readable).
EXTRA_ENV_SIDECAR = "operator_extra_env.json"


def read_state_json(session_dir: Path, path: Path) -> dict[str, Any]:
    """The state dict at ``path`` with the pins restored from the sidecar (an old session keeps its own)."""
    doc = read_json_strict(path, require_dict=True)
    sidecar = Path(session_dir) / EXTRA_ENV_SIDECAR
    if sidecar.is_file():
        doc["operator_extra_env"] = {str(k): str(v) for k, v in read_json_strict(sidecar, require_dict=True).items()}
    return doc


def write_state_json(session_dir: Path, path: Path, doc: dict[str, Any]) -> None:
    """Write ``doc`` (a state dict) to ``path`` group-readable, its pins masked, the pins to the sidecar."""
    pins = dict(doc.get("operator_extra_env") or {})
    sidecar = Path(session_dir) / EXTRA_ENV_SIDECAR
    if pins:
        atomic_write_json(sidecar, pins, options=WriteOptions(make_parents=True, mode=0o600))
    else:
        sidecar.unlink(missing_ok=True)
    atomic_write_json(
        path,
        {**doc, "operator_extra_env": redact_secret_env_values(pins)},
        options=WriteOptions(make_parents=True, group_read=True),
    )
