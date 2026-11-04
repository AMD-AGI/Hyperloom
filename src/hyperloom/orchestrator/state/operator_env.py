# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Keep the operator's ``--extra-env`` pins and ``--server-args`` out of the group-readable ``state.json``.

``state.json`` is group-readable (the retest runner reads that file directly, not through ``SharedState``) but the
pins may be credentials, and so may the launch evidence recorded from them. The full operator inputs live in an
owner-only sidecar that ``--resume`` reads through ``SharedState.load_or_init``; ``state.json`` carries every
recorded launch secret masked (consumers of the live state compare the real values, so the masking is applied to
the written document, not where the values are recorded). A ``state.json`` written before the sidecar existed still
holds them itself and is read as is; the next save moves them to the sidecar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import redact_secret_env_values
from hyperloom.common.launch_masking import mask_launch_secrets
from hyperloom_common.io import WriteOptions, atomic_write_json
from hyperloom_common.jsonio import read_json_strict

#: Owner-only file beside state.json holding the operator's launch inputs in full (never group-readable).
EXTRA_ENV_SIDECAR = "operator_extra_env.json"
_SIDECAR_KEYS = ("operator_extra_env", "operator_server_args")


def read_state_json(session_dir: Path, path: Path) -> dict[str, Any]:
    """The state dict at ``path`` with the operator's launch inputs restored from the sidecar (an old session keeps
    its own)."""
    doc = read_json_strict(path, require_dict=True)
    sidecar = Path(session_dir) / EXTRA_ENV_SIDECAR
    if sidecar.is_file():
        doc.update(read_json_strict(sidecar, require_dict=True))
    return doc


def write_state_json(session_dir: Path, path: Path, doc: dict[str, Any]) -> None:
    """Write ``doc`` (a state dict) to ``path`` group-readable with every recorded launch secret masked, and the
    operator's pins and ``--server-args`` in full to the sidecar."""
    kept = {key: doc[key] for key in _SIDECAR_KEYS if doc.get(key)}
    sidecar = Path(session_dir) / EXTRA_ENV_SIDECAR
    if kept:
        atomic_write_json(sidecar, kept, options=WriteOptions(make_parents=True, mode=0o600))
    else:
        sidecar.unlink(missing_ok=True)
    atomic_write_json(
        path,
        mask_launch_secrets({**doc, "operator_extra_env": redact_secret_env_values(doc.get("operator_extra_env"))}),
        options=WriteOptions(make_parents=True, group_read=True),
    )
