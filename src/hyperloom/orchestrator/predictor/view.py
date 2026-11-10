# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The slice of session state the predictor reads, as a protocol ``SharedState`` meets.

``SharedState`` sits in the orchestrator's import cycle through the phases that
drive it, so the predictor names what it uses instead of importing it, even for
type checking, which the cycle analysis counts too.
"""

from __future__ import annotations

from typing import Any, Protocol


class SessionView(Protocol):
    """What the predictor reads of a session, and the ledgers it appends to."""

    session_id: str
    macro_cycle: int
    framework: str
    framework_version: str
    model_name: str
    model_class: str
    model_info: dict
    gpu_type: str
    precision: str
    tp: int
    ep: int
    isl: int
    osl: int
    conc: int
    max_model_len: int
    baseline_tput: float
    baseline_accuracy: float
    cumulative_gain_validated: float
    current_best: dict[str, Any]
    optimization_stack: list[dict[str, Any]]
    roofline_snapshots: list[dict[str, Any]]
    last_trace_analyze: dict[str, Any]
    auto_roofline_pending_task_id: str
    phase_history: list[dict[str, Any]]
    specialist_rounds: list[dict[str, Any]]
    predictor_asked_keys: list[str]

    def record_specialist_round(self, entry: dict[str, Any]) -> dict[str, Any] | None:
        """Append one round to ``specialist_rounds``, stamping ``completed_at``; returns the stored row."""

    def benched_fingerprints(self) -> set[str]:
        """Content fingerprints of every variant an explore round has benched."""

    def untested_proposal_rows(self) -> list[dict[str, Any]]:
        """Executable proposals from this cycle that no explore round has benched, in queue order."""
