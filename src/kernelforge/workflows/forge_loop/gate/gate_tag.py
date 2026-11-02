# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The in-session gate's outcome tag that opened an Implementer session's rationale in earlier runs.

The Implementer now records only the RATIONALE line it declares, but a run archived before that still carries the
tag in front of it, and ``kernelforge export-runs`` strips it to read the agent's own text back.
"""

from __future__ import annotations

import re

_OPENING = "[gate edits="
_TAG = re.compile(rf"^{re.escape(_OPENING)}[^\]]*\]\s*")


def strip_gate_outcome_tag(rationale: str) -> str:
    """``rationale`` without the gate outcome tag an earlier run put in front of it."""
    return _TAG.sub("", rationale, count=1)
