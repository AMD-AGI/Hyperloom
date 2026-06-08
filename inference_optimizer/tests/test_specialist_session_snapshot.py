# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""``§ 5d. SESSION SNAPSHOT`` plumbing tests for the ``session_steward``
specialist.

The session_steward specialist used to claim "everything is in
$SESSION_DIR/state.json which the Coordinator pre-warms below" in its
focus block, but no inline rendering of those SharedState fields
existed. The steward either had to spend a turn on ``Bash cat
state.json`` to recover the data, or invent numbers. This module
locks in the new digest plumbing:

* ``Coordinator._build_session_snapshot()`` extracts the panoramic
  state digest in a stable shape.
* ``Coordinator._warm_specialist_params()`` only populates
  ``session_snapshot`` on ``session_steward_specialist`` tasks —
  other specialists' prompts stay focused.
* ``_section_session_snapshot()`` renders the digest as a fenced
  JSON block, or emits nothing when the dict is empty.
* ``build_specialist_prompts`` injects § 5d between § 5c (pitfalls)
  and § 6 (PR feed), and only when the snapshot is non-empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.specialist_domains import (
    get_domain,
)
from inference_optimizer.orchestrator.system_prompts.specialist_prompt_builder import (
    SpecialistPromptInputs,
    _section_session_snapshot,
    build_specialist_prompts,
)


# ---------------------------------------------------------------------------
# Coordinator helpers
# ---------------------------------------------------------------------------
@dataclass
class _FullStewardState:
    """A SharedState double rich enough to exercise every snapshot field."""

    # Fields _build_session_snapshot reads:
    phase: str = "EXPLORE"
    tick: int = 42
    optimization_stack: list[dict[str, Any]] = field(default_factory=lambda: [
        {"name": "a"}, {"name": "b"}, {"name": "c"},
    ])
    cumulative_gain: float = 18.7
    cumulative_gain_validated: float = 14.2
    gain_per_stack_entry: list[float | None] = field(
        default_factory=lambda: [8.0, 5.0, None, 1.5, 0.2],
    )
    explore_search: dict[str, Any] = field(default_factory=lambda: {
        "rejected": [
            {"reason": "gain_below_threshold"},
            {"reason": "gain_below_threshold"},
            {"reason": "stack_unstable"},
            {"reason": "gain_below_threshold"},
        ],
    })
    specialist_domain_empty_streak: dict[str, int] = field(default_factory=lambda: {
        "serving_specialist": 3,
        "kernel_switch_specialist": 1,
    })
    gaps: list[dict[str, Any]] = field(default_factory=lambda: [
        {"canonical_id": "gap.attention.softmax", "summary": "..."},
        {"canonical_id": "gap.kv_cache.layout", "summary": "..."},
        {"canonical_id": "gap.moe.expert_routing", "summary": "..."},
    ])
    policy_denial_history: list[dict[str, Any]] = field(default_factory=lambda: [
        {"rule": "kb_write_unauthorized", "reason": "specialist tried propose_point"},
        {"rule": "kb_write_unauthorized", "reason": "specialist tried propose_point"},
        {"rule": "tool_whitelist_role",   "reason": "system_specialist used CDP"},
    ])
    steward_continuation_used: bool = False
    # Fields _warm_specialist_params touches but we don't care here.
    gpu_type: str = ""
    tp: int = 0
    precision: str = ""
    conc: int = 0
    isl: int = 0
    osl: int = 0
    max_model_len: int = 0
    warm_start_recipe: dict[str, Any] = field(default_factory=dict)
    warm_start_pitfalls: list[dict[str, Any]] = field(default_factory=list)
    warm_start_lessons: list[dict[str, Any]] = field(default_factory=list)
    warm_replay_outcome: dict[str, Any] = field(default_factory=dict)
    last_trace_analyze: dict[str, Any] = field(default_factory=dict)

    def find_gap(self, _cid: str):
        return None


def _make_coord(tmp_path: Path, *, state: _FullStewardState) -> Coordinator:
    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = state
    c.knowledge_plane = None
    return c


# ---------------------------------------------------------------------------
# _build_session_snapshot — output shape
# ---------------------------------------------------------------------------
def test_build_session_snapshot_returns_all_documented_fields(tmp_path: Path):
    """Every key the focus prompt references must be present; missing
    keys would force the prompt to render ``null`` / cause the LLM
    to halt with "missing field"."""
    coord = _make_coord(tmp_path, state=_FullStewardState())
    snap = coord._build_session_snapshot()
    # All documented keys present (locks the contract).
    for key in (
        "phase", "tick",
        "optimization_stack_len",
        "cumulative_gain_pct", "cumulative_gain_validated_pct",
        "gain_per_stack_entry_tail",
        "rejected_counts",
        "specialist_empty_streak",
        "gaps_count", "gaps_top5_canonical_ids",
        "policy_denial_history_tail",
        "steward_continuation_used",
        # GAP 1 — warm-replay context so steward can distinguish
        # cumulative_gain that came from KB recipe inheritance vs
        # this session's own EXPLORE work.
        "warm_replay_status",
        "warm_replay_actual_gain_pct",
    ):
        assert key in snap, f"missing key: {key}"


def test_build_session_snapshot_exposes_warm_replay_outcome(tmp_path: Path):
    """GAP 1 — when warm-replay reproduced the KB best_config, the
    snapshot must surface the status + actual_gain_pct so the steward
    can attribute the cumulative_gain to inheritance vs. session work."""
    state = _FullStewardState()
    state.warm_replay_outcome = {
        "status": "reproduced",
        "actual_gain_pct": 23.5,
        "expected_gain_pct": 25.0,
        "warm_recipe_tier": "exact",
    }
    coord = _make_coord(tmp_path, state=state)
    snap = coord._build_session_snapshot()
    assert snap["warm_replay_status"] == "reproduced"
    assert snap["warm_replay_actual_gain_pct"] == 23.5


def test_build_session_snapshot_warm_replay_empty_when_not_attempted(
    tmp_path: Path,
):
    """No warm-replay → snapshot still has the keys (stable contract)
    but with empty / zero values so the prompt template doesn't branch."""
    coord = _make_coord(tmp_path, state=_FullStewardState())
    snap = coord._build_session_snapshot()
    assert snap["warm_replay_status"] == ""
    assert snap["warm_replay_actual_gain_pct"] == 0.0


def test_build_session_snapshot_aggregates_rejected_reasons(tmp_path: Path):
    """``rejected_counts`` must group ``explore_search.rejected`` by
    reason — a long tail of one kind is the primary plateau signal."""
    coord = _make_coord(tmp_path, state=_FullStewardState())
    snap = coord._build_session_snapshot()
    assert snap["rejected_counts"]["gain_below_threshold"] == 3
    assert snap["rejected_counts"]["stack_unstable"] == 1


def test_build_session_snapshot_coerces_none_gain_to_zero(tmp_path: Path):
    """The gain ledger uses ``None`` for seeded / resumed entries.
    Snapshot must coerce to 0.0 so the rendered JSON doesn't leak
    ``null`` tokens (LLM-friendly invariant)."""
    coord = _make_coord(tmp_path, state=_FullStewardState())
    snap = coord._build_session_snapshot()
    assert None not in snap["gain_per_stack_entry_tail"]
    # The 3rd entry in the fixture is None — must be 0.0 after coercion.
    assert snap["gain_per_stack_entry_tail"][2] == 0.0


def test_build_session_snapshot_truncates_gaps_to_top5(tmp_path: Path):
    """Long gaps lists could flood the prompt; only top-5 canonical_ids
    surface (the steward references at most one in
    ``next_gap_canonical_id`` anyway)."""
    state = _FullStewardState()
    state.gaps = [
        {"canonical_id": f"gap.{i}"} for i in range(20)
    ]
    coord = _make_coord(tmp_path, state=state)
    snap = coord._build_session_snapshot()
    assert snap["gaps_count"] == 20
    assert len(snap["gaps_top5_canonical_ids"]) == 5
    assert snap["gaps_top5_canonical_ids"][0] == "gap.0"


def test_build_session_snapshot_truncates_denials_to_last10(tmp_path: Path):
    """Policy denial history can grow unbounded; only the last 10
    survive into the snapshot, with reason text capped per row to
    keep prompt tokens predictable."""
    state = _FullStewardState()
    state.policy_denial_history = [
        {"rule": "r", "reason": "x" * 500} for _ in range(50)
    ]
    coord = _make_coord(tmp_path, state=state)
    snap = coord._build_session_snapshot()
    tail = snap["policy_denial_history_tail"]
    assert len(tail) == 10
    # Reason text capped to 120 chars per row.
    assert all(len(d["reason"]) <= 120 for d in tail)


# ---------------------------------------------------------------------------
# _warm_specialist_params — domain gating
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_warmer_populates_session_snapshot_only_for_session_steward(
    tmp_path: Path,
):
    """Other specialists must not see § 5d — their prompt would
    explode with irrelevant state, and the snapshot is shaped for
    steward-specific signals only."""
    coord = _make_coord(tmp_path, state=_FullStewardState())
    # Non-steward domain.
    params_serving: dict[str, Any] = {"domain": "serving_specialist"}
    await coord._warm_specialist_params(params_serving)
    assert "session_snapshot" not in params_serving
    # Steward domain.
    params_steward: dict[str, Any] = {"domain": "session_steward_specialist"}
    await coord._warm_specialist_params(params_steward)
    assert "session_snapshot" in params_steward
    assert params_steward["session_snapshot"]["optimization_stack_len"] == 3


# ---------------------------------------------------------------------------
# Prompt section + assembler
# ---------------------------------------------------------------------------
def _make_inp(snapshot: dict[str, Any] | None = None) -> SpecialistPromptInputs:
    return SpecialistPromptInputs(
        task_id="t-1",
        domain=get_domain("session_steward_specialist"),
        session_snapshot=snapshot or {},
    )


def test_section_session_snapshot_empty_returns_no_section():
    """Non-steward specialists pass an empty dict → section is
    skipped entirely (no ``## 5d.`` header at all)."""
    rows = _section_session_snapshot(_make_inp({}))
    assert rows == []


def test_section_session_snapshot_renders_fenced_json():
    snap = {
        "phase": "EXPLORE",
        "tick": 17,
        "optimization_stack_len": 4,
        "cumulative_gain_pct": 22.5,
    }
    rows = _section_session_snapshot(_make_inp(snap))
    text = "\n".join(rows)
    assert "## 5d. SESSION SNAPSHOT" in text
    assert "```json" in text
    # Stable JSON ordering so the prompt diff is reviewable.
    assert "\"cumulative_gain_pct\": 22.5" in text
    assert "\"optimization_stack_len\": 4" in text


def test_build_specialist_prompts_injects_5d_only_when_snapshot_present():
    """End-to-end: § 5d appears between § 5c (pitfalls) and § 6 (PR
    feed) when ``session_snapshot`` is populated; not at all otherwise."""
    # With snapshot — section appears in order.
    inp_with = _make_inp({"phase": "EXPLORE", "tick": 1, "optimization_stack_len": 0})
    _sys, user = build_specialist_prompts(inp_with)
    pitfalls_idx = user.index("## 5c. KNOWN PITFALLS")
    snapshot_idx = user.index("## 5d. SESSION SNAPSHOT")
    pr_idx = user.index("## 6. PR FEED")
    assert pitfalls_idx < snapshot_idx < pr_idx
    # Without snapshot — section absent.
    inp_without = _make_inp(None)
    _sys, user2 = build_specialist_prompts(inp_without)
    assert "## 5d. SESSION SNAPSHOT" not in user2
