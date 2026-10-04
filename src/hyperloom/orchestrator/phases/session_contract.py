# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The workflow-evaluation contract the bound session exports against.

A session keeps the contract identity its manifest was stamped with, so a session started under an
older contract and resumed on newer code still exports against the older schema. A frozen fact that
only a newer contract declares has to stay out of that session's records.
"""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.inference_optimizer.breakdown.workflow_contract import (
    CURRENT_WORKFLOW_CONTRACT_VERSION,
    contract_declares,
    manifest_contract_version,
)
from hyperloom.inference_optimizer.session.session_binding import bound_session_or_none
from hyperloom.inference_optimizer.session.session_paths import manifest_path


def _session_contract_version(session_dir: Path) -> str | None:
    """Contract stamped on the session's manifest, or ``None`` when the manifest cannot be read."""
    path = manifest_path(session_dir)
    if not path.exists():
        return CURRENT_WORKFLOW_CONTRACT_VERSION
    try:
        return manifest_contract_version(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, AttributeError):
        return None


def bound_session_declares(definition: str, field: str) -> bool:
    """Whether the bound session's contract schema declares ``field`` on ``$defs[definition]``.

    Nothing bound means no recorded identity to honour, so the current contract answers. An identity
    that cannot be read, or names no published contract, declares nothing: leaving a fact out is the
    shape every contract version accepts.
    """
    session = bound_session_or_none()
    version = CURRENT_WORKFLOW_CONTRACT_VERSION if session is None else _session_contract_version(session)
    if version is None:
        return False
    try:
        return contract_declares(version, definition, field)
    except KeyError:
        return False


__all__ = ["bound_session_declares"]
