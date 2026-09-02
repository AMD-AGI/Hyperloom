# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""structured gaps[] ledger tests.

Covers the SharedState ``gaps`` field + write helpers, the PolicyGate lock,
Coordinator ``_refresh_gaps`` extraction, the ``to_gaps_summary`` rendering,
and specialist-param warmup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.protocol.intent import (
    Intent,
    IntentType,
)
from hyperloom.orchestrator.policy.gate import (
    CORE_STATE_FIELDS,
    PolicyDenied,
    PolicyGate,
)
from hyperloom.orchestrator.state.shared_state import (
    SharedState,
    _GAPS_ATTEMPTS_HISTORY,
    _GAPS_MAX_ENTRIES,
)
from hyperloom.orchestrator.roles.agent_role import default_role_registry


# 1. Field surface
def test_gaps_field_exists_default_empty_list():
    """SharedState exposes a ``gaps: list[dict]`` field defaulting to ``[]``."""
    s = SharedState()
    assert hasattr(s, "gaps")
    assert s.gaps == []
    assert isinstance(s.gaps, list)


def test_gaps_field_roundtrip_through_state_json(tmp_path):
    """The field survives save → load (Inv-10.1 fact-layer survival)."""
    sd = tmp_path / "session"
    sd.mkdir()
    s = SharedState()
    s.gaps = [
        {
            "canonical_id": "issue.moe.routing",
            "symptom": "MoE routing overhead dominates",
            "layer": "kernel_agent",
            "severity": "high",
            "domain_hint": "kernel_switch_specialist",
            "source": "baseline",
            "first_seen_ts": "2025-01-01T00:00:00+00:00",
            "last_updated_ts": "2025-01-01T00:00:00+00:00",
            "attempts": [],
        },
    ]
    s.save(sd)
    loaded = SharedState.load_or_init(sd)
    assert len(loaded.gaps) == 1
    assert loaded.gaps[0]["canonical_id"] == "issue.moe.routing"
    assert loaded.gaps[0]["layer"] == "kernel_agent"


# 2. PolicyGate lock (Inv-1 / Inv-10.2)
def test_core_state_fields_includes_gaps():
    """``CORE_STATE_FIELDS`` MUST contain ``gaps`` so the LLM can't fabricate entries via ``update_state``."""
    assert "gaps" in CORE_STATE_FIELDS


def test_policy_gate_rejects_update_state_for_gaps():
    """Orchestration cannot mutate gaps[] via ``update_state`` (rule='state_field')."""
    gate = PolicyGate(role_registry=default_role_registry())
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent(
            "orchestration",
            Intent(
                type=IntentType.UPDATE_STATE,
                payload={"changes": {"gaps": [{"canonical_id": "fake"}]}},
            ),
        )
    assert exc.value.rule == "state_field"


# 3. SharedState helpers
def test_upsert_gap_inserts_then_merges_by_canonical_id():
    s = SharedState()
    e1 = s.upsert_gap(
        {
            "canonical_id": "issue.attn.headdim",
            "symptom": "attention head dim mismatch",
            "layer": "kernel_agent",
            "severity": "medium",
        }
    )
    assert len(s.gaps) == 1
    assert e1["canonical_id"] == "issue.attn.headdim"
    assert e1["first_seen_ts"]
    assert e1["last_updated_ts"]
    # Re-upserting with the same canonical_id MUST NOT duplicate.
    e2 = s.upsert_gap(
        {
            "canonical_id": "issue.attn.headdim",
            "symptom": "(refined) attention head dim mismatch on prefill",
            "severity": "high",  # override
        }
    )
    assert len(s.gaps) == 1
    assert e2["severity"] == "high"
    assert "prefill" in e2["symptom"]
    # ``first_seen_ts`` preserved.
    assert e2["first_seen_ts"] == e1["first_seen_ts"]


def test_upsert_gap_drops_blank_canonical_id():
    s = SharedState()
    out = s.upsert_gap({"symptom": "no id"})
    assert out == {}
    assert s.gaps == []


def test_append_gap_attempt_caps_at_history_limit():
    s = SharedState()
    s.upsert_gap(
        {
            "canonical_id": "issue.x",
            "symptom": "x",
            "layer": "framework",
        }
    )
    for i in range(_GAPS_ATTEMPTS_HISTORY + 5):
        s.append_gap_attempt(
            "issue.x",
            {
                "action": "explore",
                "variant_name": f"v{i}",
                "outcome": "REVERT",
            },
        )
    gap = s.find_gap("issue.x")
    assert gap is not None
    assert len(gap["attempts"]) == _GAPS_ATTEMPTS_HISTORY
    # Cap retains the newest rows (insertion order).
    assert gap["attempts"][-1]["variant_name"] == (f"v{_GAPS_ATTEMPTS_HISTORY + 4}")


def test_append_gap_attempt_returns_none_for_unknown_gap():
    s = SharedState()
    assert s.append_gap_attempt("issue.nope", {"action": "explore"}) is None


def test_upsert_gap_enforces_global_entries_cap():
    s = SharedState()
    for i in range(_GAPS_MAX_ENTRIES + 5):
        s.upsert_gap(
            {
                "canonical_id": f"issue.cap.{i}",
                "symptom": f"gap {i}",
                "layer": "framework",
            }
        )
    assert len(s.gaps) == _GAPS_MAX_ENTRIES
    # Most recently upserted MUST be retained.
    assert s.find_gap(f"issue.cap.{_GAPS_MAX_ENTRIES + 4}") is not None


# 4. Prompt rendering (to_gaps_summary)
def test_to_gaps_summary_empty_returns_empty_string():
    """Cold-start sessions skip the whole block; the header is added only when the body is non-empty."""
    assert SharedState().to_gaps_summary() == ""


def test_to_gaps_summary_renders_canonical_id_and_layer():
    s = SharedState()
    s.upsert_gap(
        {
            "canonical_id": "issue.moe.routing",
            "symptom": "MoE routing dominates",
            "layer": "kernel_agent",
            "severity": "high",
        }
    )
    s.append_gap_attempt(
        "issue.moe.routing",
        {
            "action": "explore",
            "variant_name": "moe_v1",
            "outcome": "REVERT",
        },
    )
    out = s.to_gaps_summary()
    assert "issue.moe.routing" in out
    assert "kernel_agent/high" in out
    assert "MoE routing dominates" in out
    assert "attempts=1" in out
    assert "last=explore:REVERT" in out


def test_to_gaps_summary_caps_entries_at_max():
    s = SharedState()
    for i in range(15):
        s.upsert_gap(
            {
                "canonical_id": f"issue.c{i}",
                "symptom": f"gap {i}",
                "layer": "framework",
            }
        )
    out = s.to_gaps_summary(max_entries=5)
    # Only 5 rows + 1 elision marker.
    rendered_rows = [ln for ln in out.splitlines() if ln.startswith("  - issue.c")]
    assert len(rendered_rows) == 5
    assert "older gaps elided" in out


# 5. Coordinator helpers (_refresh_gaps / _extract_* / _warm_specialist_params)
@dataclass
class _StubTask:
    task_id: str
    kind: str = "explore"
    params: dict[str, Any] = field(default_factory=dict)


@pytest.fixture
def coord(tmp_path: Path):
    """Coordinator stand-in via ``Coordinator.__new__`` (skips the full constructor)."""
    from hyperloom.orchestrator.loop.coordinator import Coordinator

    c = Coordinator.__new__(Coordinator)
    c.session_dir = tmp_path
    c.shared_state = SharedState()
    c.shared_state.session_id = "test-session"
    c.shared_state.model_name = "llama-3.1-70B"
    c.shared_state.gpu_type = "mi300x"
    c.knowledge_plane = None
    return c


@pytest.mark.asyncio
async def test_refresh_gaps_no_op_until_baseline(coord):
    """Before baseline, _refresh_gaps keeps gaps[] empty (extractors gate on baseline_tput > 0)."""
    await coord._refresh_gaps(reason="baseline_done")
    assert coord.shared_state.gaps == []


@pytest.mark.asyncio
async def test_refresh_gaps_seeds_throughput_gap_from_baseline(coord):
    """After baseline + non-zero target_gap_pct, the extractor emits a `throughput_below_target` gap row anchored to the workload id."""
    s = coord.shared_state
    s.baseline_tput = 1000.0
    s.target_gap_pct = 12.0
    await coord._refresh_gaps(reason="baseline_done")
    matches = [g for g in s.gaps if g["canonical_id"].endswith("#throughput_below_target")]
    assert matches, f"missing throughput gap in {s.gaps!r}"
    gap = matches[0]
    assert gap["layer"] == "framework"
    assert gap["severity"] == "high"  # 12% gap → high severity
    assert gap["domain_hint"] == "serving_specialist"
    assert gap["source"] == "baseline"


@pytest.mark.asyncio
async def test_refresh_gaps_emits_baseline_unstable_gap(coord):
    s = coord.shared_state
    s.baseline_tput = 800.0
    s.baseline_failure_streak = 2
    await coord._refresh_gaps(reason="baseline_done")
    instab = [g for g in s.gaps if g["canonical_id"].endswith("#baseline_unstable")]
    assert instab, "missing baseline_unstable gap"
    assert instab[0]["layer"] == "system"
    assert instab[0]["severity"] == "high"  # streak ≥ 2


@pytest.mark.asyncio
async def test_refresh_gaps_dedupes_recurring_failures(coord):
    """The attempts extractor folds repeated (action, error_class) failures into a single gap row, one attempt per failure."""
    s = coord.shared_state
    s.baseline_tput = 700.0
    s.last_action_failures = [
        {"action": "explore", "error_class": "no_report", "ts": "2025-01-01T00:00:00+00:00"},
        {"action": "explore", "error_class": "no_report", "ts": "2025-01-01T00:01:00+00:00"},
        {"action": "integrate", "error_class": "compile_failure", "ts": "2025-01-01T00:02:00+00:00"},
    ]
    await coord._refresh_gaps(reason="explore_round")
    by_id = {g["canonical_id"]: g for g in s.gaps}
    explore_gaps = [g for cid, g in by_id.items() if "#fail:explore:no_report" in cid]
    kernel_gaps = [g for cid, g in by_id.items() if "#fail:integrate:compile_failure" in cid]
    assert len(explore_gaps) == 1
    assert len(explore_gaps[0]["attempts"]) == 2
    assert explore_gaps[0]["layer"] == "framework"
    assert kernel_gaps and kernel_gaps[0]["layer"] == "kernel_agent"


@pytest.mark.asyncio
async def test_refresh_gaps_emits_explore_plateau_after_streak(coord):
    s = coord.shared_state
    s.baseline_tput = 1000.0
    s.params_no_promote_streak = 4
    s.explore_search = {
        "winners_history": [
            {"variant_name": "v1", "gain_pct": 0.0},
            {"variant_name": "v2", "gain_pct": 0.0},
        ]
    }
    await coord._refresh_gaps(reason="explore_round")
    plateau = [g for g in s.gaps if g["canonical_id"].endswith("#explore_plateau")]
    assert plateau, "explore_plateau gap missing"
    assert plateau[0]["domain_hint"] == "serving_specialist"


@pytest.mark.asyncio
async def test_record_explore_round_gaps_appends_attempts(coord):
    s = coord.shared_state
    s.baseline_tput = 900.0
    s.upsert_gap(
        {
            "canonical_id": "issue.fp8.kv",
            "symptom": "fp8 kv cache test",
            "layer": "framework",
        }
    )
    task = _StubTask(
        task_id="explore-1",
        kind="explore",
        params={"gap_canonical_id": "issue.fp8.kv"},
    )
    coord._record_explore_round_gaps(
        task=task,
        result={
            "per_variant_outcomes": [
                {"variant_name": "kv_a", "outcome": "KEEP", "gain_pct": 1.2},
                {"variant_name": "kv_b", "outcome": "REVERT", "gain_pct": -0.3},
            ],
        },
    )
    gap = s.find_gap("issue.fp8.kv")
    assert gap is not None
    attempts = gap["attempts"]
    assert [a["variant_name"] for a in attempts] == ["kv_a", "kv_b"]
    assert attempts[0]["outcome"] == "KEEP"
    assert attempts[1]["outcome"] == "REVERT"


@pytest.mark.asyncio
async def test_record_explore_round_gaps_falls_back_to_anchor(coord):
    """Without a ``gap_canonical_id`` the attempts are still recorded under the workload anchor."""
    s = coord.shared_state
    s.baseline_tput = 900.0
    task = _StubTask(
        task_id="explore-fallback",
        kind="explore",
        params={},
    )
    coord._record_explore_round_gaps(
        task=task,
        result={
            "per_variant_outcomes": [
                {"variant_name": "v1", "outcome": "REVERT"},
            ]
        },
    )
    anchor = coord._workload_canonical_id()
    gap = s.find_gap(anchor)
    assert gap is not None
    assert any(a["variant_name"] == "v1" for a in gap["attempts"])


# 6. specialist warmup — gap fields flow into task.params
@pytest.mark.asyncio
async def test_warm_specialist_params_pulls_gap_symptom_and_layer(coord):
    s = coord.shared_state
    s.baseline_tput = 1000.0
    s.upsert_gap(
        {
            "canonical_id": "issue.moe.routing",
            "symptom": "MoE routing overhead",
            "layer": "kernel_agent",
            "severity": "high",
            "domain_hint": "kernel_switch_specialist",
        }
    )
    s.append_gap_attempt(
        "issue.moe.routing",
        {
            "action": "explore",
            "variant_name": "moe_x",
            "outcome": "REVERT",
        },
    )
    params: dict[str, Any] = {
        "domain": "kernel_switch_specialist",
        "gap_canonical_id": "issue.moe.routing",
    }
    await coord._warm_specialist_params(params)
    assert params["gap_symptom"] == "MoE routing overhead"
    assert params["gap_layer"] == "kernel_agent"
    evidence = params["gap_evidence"]
    assert isinstance(evidence, dict)
    assert evidence["severity"] == "high"
    assert len(evidence["recent_attempts"]) == 1


@pytest.mark.asyncio
async def test_warm_specialist_params_uses_domain_hint_when_domain_missing(coord):
    """When the LLM omits ``domain`` the gap's ``domain_hint`` fills in (PolicyGate still validates routing)."""
    s = coord.shared_state
    s.baseline_tput = 1000.0
    s.upsert_gap(
        {
            "canonical_id": "issue.collective.allreduce",
            "symptom": "all-reduce stall",
            "layer": "comm",
            "domain_hint": "comm_specialist",
        }
    )
    params: dict[str, Any] = {"gap_canonical_id": "issue.collective.allreduce"}
    await coord._warm_specialist_params(params)
    assert params.get("domain") == "comm_specialist"


@pytest.mark.asyncio
async def test_warm_specialist_params_noop_when_gap_unknown(coord):
    """Unknown ``gap_canonical_id`` must not clobber existing params."""
    params: dict[str, Any] = {
        "domain": "serving_specialist",
        "gap_canonical_id": "issue.unknown",
        "gap_symptom": "preset",
    }
    await coord._warm_specialist_params(params)
    assert params["domain"] == "serving_specialist"
    assert params["gap_symptom"] == "preset"


# 7. Recipe KB traverse fallback (defensive)
class _StubKnowledgePlane:
    """Minimal KnowledgePlane double returning one issue_node row from ``recipe_kb_traverse_issues``."""

    def __init__(self, rows: list[dict[str, Any]] | None = None, raises: Exception | None = None):
        self._rows = rows or []
        self._raises = raises
        self.pr_monitor_enabled = False

    def recipe_kb_traverse_issues(self, *, model_class: str, gpu_type: str):
        if self._raises is not None:
            raise self._raises
        return list(self._rows)


@pytest.mark.asyncio
async def test_refresh_gaps_merges_recipe_kb_traverse_rows(coord):
    coord.knowledge_plane = _StubKnowledgePlane(
        rows=[
            {
                "canonical_id": "issue.kb.fp8_kv_prior",
                "symptom": "prior session refuted fp8 kv at bs=256",
                "layer": "kernel_agent",
                "severity": "medium",
                "domain_hint": "kernel_switch_specialist",
            },
        ]
    )
    coord.shared_state.baseline_tput = 900.0
    await coord._refresh_gaps(reason="recipe_kb_refresh")
    found = coord.shared_state.find_gap("issue.kb.fp8_kv_prior")
    assert found is not None
    assert found["source"] == "recipe_kb"


@pytest.mark.asyncio
async def test_refresh_gaps_absorbs_recipe_kb_traverse_exception(coord):
    """Recipe KB outages must NOT crash the refresh (best-effort facets can't block the calling path)."""
    coord.knowledge_plane = _StubKnowledgePlane(
        raises=RuntimeError("traverse down"),
    )
    coord.shared_state.baseline_tput = 900.0
    # Must not raise.
    await coord._refresh_gaps(reason="recipe_kb_refresh")


@pytest.mark.asyncio
async def test_record_explore_round_gaps_carries_failure_artifacts(coord):
    """When a per_variant_outcome carries failure_id/fingerprint/workspace, they flow into the attempt."""
    s = coord.shared_state
    s.baseline_tput = 900.0
    s.upsert_gap(
        {
            "canonical_id": "issue.fp8.kv2",
            "symptom": "fp8 kv test with artifacts",
            "layer": "framework",
        }
    )
    task = _StubTask(
        task_id="explore-artifacts",
        kind="explore",
        params={"gap_canonical_id": "issue.fp8.kv2"},
    )
    coord._record_explore_round_gaps(
        task=task,
        result={
            "per_variant_outcomes": [
                {
                    "variant_name": "kv_failed",
                    "outcome": "FAILED",
                    "failure_id": "fail.explore-artifacts.abc123",
                    "fingerprint": "abc123456789",
                    "stage": "warmup",
                    "workspace": "/runs/v00_kv_failed",
                    "server_log_path": "/runs/v00_kv_failed/server.log",
                },
            ],
        },
    )
    gap = s.find_gap("issue.fp8.kv2")
    assert gap is not None
    attempt = gap["attempts"][0]
    assert attempt.get("failure_id") == "fail.explore-artifacts.abc123"
    assert attempt.get("fingerprint") == "abc123456789"
    assert attempt.get("stage") == "warmup"
    assert attempt.get("workspace") == "/runs/v00_kv_failed"
    assert attempt.get("server_log_path") == "/runs/v00_kv_failed/server.log"
