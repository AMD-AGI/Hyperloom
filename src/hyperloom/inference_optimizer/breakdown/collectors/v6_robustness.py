# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The robustness agent's own account, for SBD V6.

``robustness`` is a top-level V6 key rather than a timeline event: the agent
watches the whole session from the side and its turns do not belong to any one
phase or macro cycle. Like ``close``, it therefore keeps a fixed place in the
payload so a reader can ask "did the agent ever speak" and get an answer.

There is no projection behind this collector, and that is deliberate. The
section it replaces rebuilt each turn at export time from two files nothing
writes (see :mod:`..recorder.robustness_out`), so every row it produced was
blank. A fallback here could only reproduce that. What the agent raised is
known only at the moment it raises it, so a session with no fragments reports
no turns rather than a row per workdir it happens to find on disk.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = ["collect_v6_robustness"]


def _rows(value: Any) -> list[dict[str, Any]]:
    return [dict(r) for r in value if isinstance(r, Mapping)] if isinstance(value, list) else []


def collect_v6_robustness(recorded: Any = None) -> dict[str, Any]:
    """Put the recorded robustness turns on the wire.

    Args:
        recorded (Any): The assembled ``robustness`` view, when present.

    Returns:
        dict[str, Any]: ``{"turns": [...]}``. Always a full object: an empty
        ``turns`` says the agent never completed a turn, which is itself the
        answer a reader is after.
    """
    view = recorded if isinstance(recorded, Mapping) else {}
    return {"turns": _rows(view.get("turns"))}
