# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic working-memory aggregation for FRAMEWORK candidate selection.

``_build_framework_working_memory`` folds the "already tried this session"
ledger into the shape the selection path reads, capped and most-recent-first.
"""

from __future__ import annotations

from typing import Any

from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.phases.framework import FrameworkPhase


class _StateStub:
    def __init__(self) -> None:
        self.framework_agent_phase_progress: list[dict[str, Any]] = []
        self.framework_agent_batches: list[dict[str, Any]] = []
        self.research_scout_seen_pr_ids: list[str] = []
        # Workload context read by the ranker prompt.
        self.model = "test-model"
        self.model_path = ""
        self.framework = "sglang"
        self.gpu_type = "MI300X"
        self.precision = "fp8"
        self.tp = 4
        self.best_throughput = 0.0
        self.baseline_throughput = 0.0


class _MemCoord:
    _LOCAL_EXPLORE_KIND = FrameworkPhase._LOCAL_EXPLORE_KIND
    _FRAMEWORK_KEEP_STATUSES = Coordinator._FRAMEWORK_KEEP_STATUSES
    _FRAMEWORK_TRIED_MEMORY_CAP = Coordinator._FRAMEWORK_TRIED_MEMORY_CAP
    _framework_candidate_key = staticmethod(Coordinator._framework_candidate_key)
    _framework_processed_candidate_keys = Coordinator._framework_processed_candidate_keys
    _framework_known_candidate_ids = Coordinator._framework_known_candidate_ids
    _unprocessed_framework_agent_candidates = Coordinator._unprocessed_framework_agent_candidates
    _build_framework_working_memory = Coordinator._build_framework_working_memory

    def __init__(self) -> None:
        self.shared_state = _StateStub()


def test_build_working_memory_aggregates_tried_excluded_learnings():
    coord = _MemCoord()
    st = coord.shared_state
    st.framework_agent_phase_progress = [
        {"candidate_id": "PR:723", "status": "reverted", "gain_pct": 0.0, "rationale": "throughput == baseline"},
        {"candidate_id": "PR:1015", "status": "critic_denied", "rationale": "does not address mem-bw bottleneck"},
        {"candidate_id": "PR:900", "status": "kept", "gain_pct": 6.1},
    ]
    st.framework_agent_batches = [
        {
            "batch_id": "b1",
            "candidates": [
                {"candidate_id": "PR:723"},
                {"candidate_id": "PR:1015"},
                {"candidate_id": "PR:900"},
                {"candidate_id": "PR:2000"},
            ],
        },
    ]
    mem = coord._build_framework_working_memory()

    refs = {t["ref"] for t in mem["tried_and_why"]}
    assert refs == {"PR:723", "PR:1015", "PR:900"}
    revert = next(t for t in mem["tried_and_why"] if t["ref"] == "PR:723")
    assert revert["status"] == "reverted"
    assert revert["gain_pct"] == 0.0
    assert "baseline" in revert["why"]
    # excluded_refs = known ids ∪ processed keys.
    assert {"PR:723", "PR:1015", "PR:900", "PR:2000"} <= set(mem["excluded_refs"])
    # Learnings come from the denial rows in the progress ledger, which is the
    # only place a Critic rejection is recorded.
    assert mem["learnings"] == ["does not address mem-bw bottleneck"]
    # pending = unprocessed candidate in the latest batch.
    assert mem["pending"] == ["PR:2000"]


def test_build_working_memory_empty_when_no_progress():
    coord = _MemCoord()
    mem = coord._build_framework_working_memory()
    assert mem["tried_and_why"] == []
    assert mem["learnings"] == []


def test_build_working_memory_caps_tried_rows():
    coord = _MemCoord()
    cap = coord._FRAMEWORK_TRIED_MEMORY_CAP
    coord.shared_state.framework_agent_phase_progress = [
        {"candidate_id": f"PR:{i}", "status": "reverted"} for i in range(cap + 5)
    ]
    mem = coord._build_framework_working_memory()
    assert len(mem["tried_and_why"]) == cap
    # Most-recent kept (last cap entries).
    assert mem["tried_and_why"][-1]["ref"] == f"PR:{cap + 4}"
