# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""structured gaps[] ledger tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.baseline_comparison import research_hints
from hyperloom.orchestrator.knowledge.recipe_journal import recipe_attrs_from_state
from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentResult
from hyperloom.orchestrator.state.gaps import _GAPS_ATTEMPTS_HISTORY, _GAPS_MAX_ENTRIES
from hyperloom.orchestrator.state.objective import TargetGainObjective
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import Task

from .conftest import make_coordinator


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


# 2. SharedState helpers
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


def test_the_cap_drops_an_untried_gap_before_one_that_holds_attempts():
    """The oldest row survives a flood of newer ones when it is the one something was tried against."""
    s = SharedState()
    s.upsert_gap({"canonical_id": "issue.tried", "symptom": "measured"})
    s.append_gap_attempt("issue.tried", {"action": "explore", "variant_name": "v1", "outcome": "REVERT"})
    s.upsert_gap({"canonical_id": "issue.untried", "symptom": "never attempted"})
    for i in range(_GAPS_MAX_ENTRIES):
        s.upsert_gap({"canonical_id": f"issue.flood.{i}", "symptom": f"gap {i}"})

    assert len(s.gaps) == _GAPS_MAX_ENTRIES
    assert [a["variant_name"] for a in s.find_gap("issue.tried")["attempts"]] == ["v1"]
    assert s.find_gap("issue.untried") is None
    assert s.find_gap(f"issue.flood.{_GAPS_MAX_ENTRIES - 1}") is not None


def test_upsert_gap_does_not_write_attempts():
    """Attempts reach a row through ``append_gap_attempt``; an upsert neither seeds nor replaces them."""
    s = SharedState()
    seeded = s.upsert_gap({"canonical_id": "issue.x", "attempts": [{"action": "explore", "outcome": "REVERT"}]})
    assert seeded["attempts"] == []

    s.append_gap_attempt("issue.x", {"action": "explore", "variant_name": "v1", "outcome": "KEEP"})
    again = s.upsert_gap({"canonical_id": "issue.x", "symptom": "refined", "attempts": []})
    assert [a["variant_name"] for a in again["attempts"]] == ["v1"]


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
    out = s.to_gaps_summary()
    # Only 10 rows + 1 elision marker.
    rendered_rows = [ln for ln in out.splitlines() if ln.startswith("  - issue.c")]
    assert len(rendered_rows) == 10
    assert "older gaps elided" in out


def test_to_gaps_summary_lists_attempt_rows_only_when_asked():
    s = SharedState()
    s.upsert_gap({"canonical_id": "g1", "symptom": "test", "layer": "framework", "severity": "medium"})
    s.append_gap_attempt("g1", {"action": "explore", "outcome": "FAILED", "error_class": "x", "failure_id": "fid1"})

    count_only = s.to_gaps_summary(max_attempts=0)
    assert "attempts=1" in count_only
    assert "fid1" not in count_only

    with_rows = s.to_gaps_summary(max_attempts=5)
    assert "fid1" in with_rows
    assert "attempt:" in with_rows


# 5. Coordinator helpers (record_explore_round_gaps / warm_specialist_params)
@dataclass
class _StubTask:
    task_id: str
    kind: str = "explore"
    params: dict[str, Any] = field(default_factory=dict)


@pytest.fixture
def coord(tmp_path: Path):
    return make_coordinator(
        tmp_path,
        shared_state_overrides={"session_id": "test-session", "model_name": "llama-3.1-70B", "gpu_type": "mi300x"},
    )


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
    coord.gap_refresh.record_explore_round_gaps(
        workload_id=coord.recipe_journal.workload_canonical_id(),
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
    coord.gap_refresh.record_explore_round_gaps(
        workload_id=coord.recipe_journal.workload_canonical_id(),
        task=task,
        result={
            "per_variant_outcomes": [
                {"variant_name": "v1", "outcome": "REVERT"},
            ]
        },
    )
    anchor = coord.recipe_journal.workload_canonical_id()
    gap = s.find_gap(anchor)
    assert gap is not None
    assert any(a["variant_name"] == "v1" for a in gap["attempts"])


@pytest.mark.parametrize(
    ("framework", "domain"),
    [("custom", "framework_rewrite_specialist"), ("sglang", "serving_specialist")],
)
def test_the_explore_anchor_gap_names_the_authoring_domain_of_the_framework(coord, framework, domain):
    """A scriptable workload's framework-layer gap routes to the rewrite specialist, not back to the serving surface."""
    coord.shared_state.framework = framework
    anchor = coord.recipe_journal.workload_canonical_id()
    coord.gap_refresh.record_explore_round_gaps(
        workload_id=anchor,
        task=_StubTask(task_id="explore-anchor"),
        result={"per_variant_outcomes": [{"variant_name": "v1", "outcome": "REVERT"}]},
    )
    gap = coord.shared_state.find_gap(anchor)
    assert (gap["layer"], gap["domain_hint"]) == ("framework", domain)


def test_seeding_more_hints_than_the_ledger_holds_keeps_the_gap_with_measured_attempts(coord, tmp_path):
    """Research hints fill the ledger to its cap and still leave the explore outcomes already on record."""
    anchor = coord.recipe_journal.workload_canonical_id()
    coord.gap_refresh.record_explore_round_gaps(
        workload_id=anchor,
        task=_StubTask(task_id="explore-measured"),
        result={"per_variant_outcomes": [{"variant_name": "kv_b", "outcome": "REVERT"}]},
    )
    research_hints.append_hints(
        tmp_path,
        [{"what": f"prior {i}", "source": f"https://example.test/{i}"} for i in range(_GAPS_MAX_ENTRIES + 10)],
    )

    for _ in range(2):
        coord.gap_refresh.seed_gaps_from_research_hints()
        assert len(coord.shared_state.gaps) == _GAPS_MAX_ENTRIES
        assert [a["variant_name"] for a in coord.shared_state.find_gap(anchor)["attempts"]] == ["kv_b"]
    assert sum(g["layer"] == "research_hint" for g in coord.shared_state.gaps) == _GAPS_MAX_ENTRIES - 1


async def test_a_closed_specialist_round_derives_no_gap_from_the_objective_or_the_failure_log(coord):
    """The objective shortfall, a baseline crash streak and a failed action are shown elsewhere, not as gap rows."""
    s = coord.shared_state
    s.baseline_tput = 1000.0
    s.baseline_failure_streak = 2
    coord._current_objective = TargetGainObjective(target_gain_pct=12.0)
    s.record_action_failure(action="roofline", task_id="t-1", result={"error_class": "trace_analyze_failed"})

    await coord.specialist_dispatch.record_specialist_result(
        task=_StubTask(task_id="spec-1", kind="specialist"),
        done_payload={"domain": "serving_specialist", "gap_canonical_id": "gap.own.idea", "proposal_set": []},
        source="specialist:spec-1",
    )

    assert coord.target_gap_pct() > 0
    assert [row["task_id"] for row in s.specialist_rounds] == ["spec-1"]
    assert s.gaps == []


async def test_an_explore_round_publishes_only_its_measured_loser_as_reverted(coord):
    """A variant that measured nothing reaches the recipe once, by its error class; a measured loser as reverted."""
    s = coord.shared_state
    s.baseline_tput = 900.0
    s.current_best = {"action": "baseline", "tput": 900.0, "extra_server_args": "", "extra_envs": {}}
    task = Task(task_id="t-explore", kind="explore", state="running", params={}, idempotency_key="explore-t")
    outcomes = [
        {"variant_name": "boot_dead", "outcome": "FAILED", "error_class": "server_init_dead", "fingerprint": "aa"},
        {"variant_name": "slower", "outcome": "REVERT", "reason": "below_bar", "fingerprint": "bb"},
    ]

    await coord.dispatcher.reap_dispatched_task(
        task,
        SubAgentResult(
            task_id=task.task_id,
            state="succeeded",
            result={"status": "succeeded", "round_id": "r1", "per_variant_outcomes": outcomes},
        ),
    )

    (gap,) = s.gaps
    assert gap["canonical_id"] == coord.recipe_journal.workload_canonical_id()
    assert [(a["variant_name"], a["outcome"]) for a in gap["attempts"]] == [
        ("boot_dead", "FAILED"),
        ("slower", "REVERT"),
    ]
    assert [(row["variant_name"], row["error_class"]) for row in s.last_action_failures] == [
        ("boot_dead", "server_init_dead")
    ]
    assert s.recipe_sediment_enabled
    assert recipe_attrs_from_state(s, "test-session")["what_failed"] == [
        {"name": "explore", "reason": "server_init_dead"},
        {"name": "slower", "reason": "reverted", "gain_pct": None},
    ]


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
    await coord.specialist_dispatch.warm_specialist_params(params)
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
    await coord.specialist_dispatch.warm_specialist_params(params)
    assert params.get("domain") == "comm_specialist"


@pytest.mark.asyncio
async def test_warm_specialist_params_noop_when_gap_unknown(coord):
    """Unknown ``gap_canonical_id`` must not clobber existing params."""
    params: dict[str, Any] = {
        "domain": "serving_specialist",
        "gap_canonical_id": "issue.unknown",
        "gap_symptom": "preset",
    }
    await coord.specialist_dispatch.warm_specialist_params(params)
    assert params["domain"] == "serving_specialist"
    assert params["gap_symptom"] == "preset"


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
    coord.gap_refresh.record_explore_round_gaps(
        workload_id=coord.recipe_journal.workload_canonical_id(),
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
