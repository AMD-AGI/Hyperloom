# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The workflow-evaluation contract the bound session exports against.

A session keeps the contract identity its manifest was stamped with, so a session started under an
older contract and resumed on newer code still exports against the older schema. A frozen fact that
only a newer contract declares has to stay out of that session's records.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from hyperloom.inference_optimizer.breakdown.workflow_contract import (
    CURRENT_WORKFLOW_CONTRACT_VERSION,
    contract_declares,
    manifest_contract_version,
)
from hyperloom.inference_optimizer.session.session_binding import bound_session_or_none
from hyperloom.inference_optimizer.session.session_paths import manifest_path

# The stamp is written once, with a fresh manifest, and never rewritten, so a successful read holds
# for the life of the session. Only successful reads are kept; a failed read is retried next time.
_STAMPED_VERSIONS: dict[Path, str] = {}


def _session_contract_version(session_dir: Path) -> str:
    """Contract stamped on the session's manifest.

    A missing manifest, or one that cannot be read or parsed right now (EMFILE, EIO, ESTALE, EACCES,
    a torn write), answers with the current contract: the stamp is only ever absent from sessions
    that predate it and whose manifest is readable, and every session this code starts carries the
    current stamp. Falling back to "declares nothing" would drop facts the current contract
    requires.
    """
    key = Path(session_dir)
    cached = _STAMPED_VERSIONS.get(key)
    if cached is not None:
        return cached
    try:
        manifest = json.loads(manifest_path(key).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return CURRENT_WORKFLOW_CONTRACT_VERSION
    if not isinstance(manifest, Mapping):
        return CURRENT_WORKFLOW_CONTRACT_VERSION
    version = manifest_contract_version(manifest)
    _STAMPED_VERSIONS[key] = version
    return version


def bound_session_declares(definition: str, field: str) -> bool:
    """Whether the bound session's contract schema declares ``field`` on ``$defs[definition]``.

    Nothing bound means no recorded identity to honour, so the current contract answers, as it does
    when the bound session's manifest cannot be read. A readable identity that names no published
    contract declares nothing.
    """
    session = bound_session_or_none()
    version = CURRENT_WORKFLOW_CONTRACT_VERSION if session is None else _session_contract_version(session)
    try:
        return contract_declares(version, definition, field)
    except KeyError:
        return False


__all__ = ["bound_session_declares"]
