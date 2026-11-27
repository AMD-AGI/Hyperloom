# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The persisted state of the session ``--resume-from`` names, read before the resume block loads it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _resumed_session_state(args: argparse.Namespace) -> dict[str, Any] | None:
    """The persisted ``state.json`` of the session ``--resume-from`` names, or ``None``.

    Preflight and the launch-shape export run before the resume block loads the session, so anything they decide from
    a session-pinned setting has to read it here. ``None`` (no session named, or no readable state) leaves the decision
    to the flags; the resume block reports a missing or unreadable state itself.
    """
    raw = str(getattr(args, "resume_from", "") or "").strip()
    if not raw:
        return None
    try:
        state = json.loads((Path(raw).expanduser() / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) else None
