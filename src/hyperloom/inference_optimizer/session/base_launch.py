# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session's frozen base launch, ``base_launch.json``.

A fresh launch resolves the base once and writes it here; every later read, a resume's included, takes it from this
file and never resolves it again. Reading refuses a file whose content no longer matches its recorded digest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hyperloom.common.session_files import shared_options
from hyperloom_common.io import atomic_write_json
from inference_suite.contracts import BaseLaunch

BASE_LAUNCH_FILE = "base_launch.json"


def base_launch_path(session_dir: Path) -> Path:
    return Path(session_dir) / BASE_LAUNCH_FILE


def write_base_launch(session_dir: Path, base: BaseLaunch) -> Path:
    """Freeze ``base`` as the session's base launch, ``664`` below shared storage unless it holds a credential;
    returns the file written."""
    path = base_launch_path(session_dir)
    document = base.to_document()
    atomic_write_json(path, document, options=shared_options(document))  # a later leg may be another account
    return path


def read_base_launch(session_dir: Path) -> BaseLaunch | None:
    """The session's frozen base, or ``None`` for a session that froze none (a scriptable or AgentX workload)."""
    try:
        text = base_launch_path(session_dir).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return BaseLaunch.from_document(json.loads(text))


def shared_state_fields(base: BaseLaunch | None) -> dict[str, Any]:
    """The ``SharedState`` fields a fresh session seeds from its frozen ``base``: the digest naming the frozen file."""
    if base is None:
        return {}
    return {"base_launch_digest": base.digest()}


__all__ = ["BASE_LAUNCH_FILE", "base_launch_path", "read_base_launch", "shared_state_fields", "write_base_launch"]
