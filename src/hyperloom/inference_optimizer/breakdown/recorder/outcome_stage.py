# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Author-time workflow stage milestones."""

from __future__ import annotations

from pathlib import Path

from .assembler import assemble_parts
from .recorder import recorder_for

PHASE_STAGES = {
    "PRELUDE": "prelude",
    "ENABLEMENT": "enablement",
    "FRAMEWORK_AGENT": "framework_agent",
    "KERNEL_AGENT": "kernel_agent",
    "SWEEP": "conc_sweep",
    "CLOSE": "close",
}
_STAGE_ORDER = (
    "install",
    "model_gate",
    "prelude",
    "warm_start",
    "baseline",
    "warm_replay",
    "profile",
    "roofline",
    "enablement",
    "framework_agent",
    "kernel_agent",
    "kernel",
    "conc_sweep",
    "close",
)
EVENT_STAGES = frozenset(
    {"warm_start", "baseline", "warm_replay", "roofline", "enablement", "framework_agent", "kernel", "conc_sweep"}
)


def record_stage_reached(session_dir: Path, stage: str) -> None:
    """Persist an authored milestone without downgrading it on a phase reloop."""
    recorded = assemble_parts(session_dir, only_sections=("outcome",)).get("outcome") or {}
    previous = recorded.get("stage_reached_recorded")
    if stage in _STAGE_ORDER and previous in _STAGE_ORDER:
        if _STAGE_ORDER.index(stage) <= _STAGE_ORDER.index(previous):
            return
    recorder_for(session_dir, producer="workflow_outcome").record_upsert_singleton(
        "outcome", {"stage_reached_recorded": stage}
    )
