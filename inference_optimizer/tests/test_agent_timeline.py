# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit coverage for the unified ``agent_timeline`` builder and Langfuse reader.

The builder is a pure projection that merges the three decision histories that
already live in a breakdown (``decision_trace`` / ``specialist_runs`` /
``critic_robustness``) onto one ``ts`` axis. These tests pin:

* the merge picks all three actors and sorts by timestamp (unparseable last);
* per-actor field projection (outcome/gain, domain/findings, verdict/topic),
  including specialist schema drift (``domain`` vs ``domains``);
* partial / missing sections degrade gracefully (warning + ``degraded``);
* the Langfuse reader resolves trace ids, extracts the breakdown from a
  trace ``output`` (dict or JSON-string), and the upload-path bridge injects a
  ``degraded`` empty timeline when the trace cannot be fetched.
"""

from __future__ import annotations

import json

import pytest

from inference_optimizer.breakdown.agent_timeline import (
    AGENT_TIMELINE_SCHEMA,
    build_agent_timeline,
    empty_timeline,
    enrich_breakdown_with_langfuse_timeline,
)
from inference_optimizer.orchestrator.trace import langfuse_reader
from inference_optimizer.orchestrator.trace.langfuse_mapping import derive_trace_id


def _full_breakdown() -> dict:
    """A breakdown carrying one event of each actor at known timestamps."""
    return {
        "session": {"claw_session_id": "claw-xyz"},
        "decision_trace": {
            "decision_trace": [
                {
                    "ts": "2026-06-17T07:35:59Z",
                    "phase": "PRELUDE",
                    "tick": 1,
                    "tokens": {"total": 100},
                    "decision": {
                        "outcome": "KEEP",
                        "change": "target_analysis",
                        "component": "orchestration",
                        "operation_kind": "other",
                        "gain_pct": None,
                        "task_id": "t-1",
                    },
                }
            ]
        },
        "specialist_runs": [
            {
                "completed_at": "2026-06-17T07:52:07.359619+00:00",
                "domain": "research_scout_specialist",
                "confidence": 0.6,
                "new_findings": ["DENSE Qwen2.5-1.5B; NO MoE"],
                "gap_canonical_id": "gap.research_scout.round0",
            }
        ],
        "critic_robustness": {
            "critic_iterations": [
                {
                    "iter": 0,
                    "ts": "2026-06-17T08:00:00Z",
                    "verdict": "approve",
                    "topic": "kernel_opt:k001",
                    "summary": "looks good",
                }
            ]
        },
    }


def test_merge_orders_by_timestamp_and_counts_actors():
    tl = build_agent_timeline(_full_breakdown())
    assert tl["schema"] == AGENT_TIMELINE_SCHEMA
    assert tl["source"] == "local"
    assert tl["degraded"] is False
    actors = [e["actor"] for e in tl["events"]]
    # 07:35 orchestrator -> 07:52 specialist -> 08:00 critic
    assert actors == ["orchestrator", "specialist", "critic"]
    assert [e["seq"] for e in tl["events"]] == [0, 1, 2]
    assert tl["counts"] == {"orchestrator": 1, "specialist": 1, "critic": 1}


def test_actor_field_projection():
    tl = build_agent_timeline(_full_breakdown())
    by_actor = {e["actor"]: e for e in tl["events"]}

    orch = by_actor["orchestrator"]
    assert orch["kind"] == "decision"
    assert orch["title"] == "KEEP target_analysis"
    assert orch["detail"]["outcome"] == "KEEP"
    assert orch["detail"]["tokens"] == {"total": 100}
    assert orch["source"] == {"section": "decision_trace.decision_trace", "index": 0}

    spec = by_actor["specialist"]
    assert spec["kind"] == "proposal"
    assert spec["title"] == "research_scout_specialist (conf 0.6)"
    assert spec["detail"]["new_findings"] == ["DENSE Qwen2.5-1.5B; NO MoE"]

    crit = by_actor["critic"]
    assert crit["kind"] == "review"
    assert crit["title"] == "verdict approve: kernel_opt:k001"
    assert crit["detail"]["verdict"] == "approve"


def test_specialist_domains_list_fallback():
    bd = {"specialist_runs": [{"completed_at": "2026-06-17T07:00:00Z", "domains": ["a", "b"]}]}
    tl = build_agent_timeline(bd)
    ev = tl["events"][0]
    assert ev["detail"]["domain"] == "a"
    assert ev["detail"]["domains"] == ["a", "b"]
    assert ev["title"] == "a"


def test_unparseable_timestamp_sorts_last_without_crashing():
    bd = {
        "critic_robustness": {
            "critic_iterations": [
                {"iter": 1, "ts": "not-a-date", "verdict": "reject"},
                {"iter": 0, "ts": "2026-06-17T07:00:00Z", "verdict": "approve"},
            ]
        }
    }
    tl = build_agent_timeline(bd)
    assert [e["detail"]["iter"] for e in tl["events"]] == [0, 1]


def test_missing_sections_degrade_with_warnings():
    tl = build_agent_timeline({})
    assert tl["events"] == []
    assert tl["degraded"] is True
    assert len(tl["warnings"]) == 3  # one per missing section
    assert tl["counts"] == {"orchestrator": 0, "specialist": 0, "critic": 0}


def test_langfuse_source_stamps_trace_id_on_events():
    tl = build_agent_timeline(_full_breakdown(), source="langfuse", trace_id="abc123")
    assert tl["source"] == "langfuse"
    assert tl["trace_id"] == "abc123"
    assert all(e["source"]["trace_id"] == "abc123" for e in tl["events"])


def test_empty_timeline_is_well_formed():
    tl = empty_timeline(reason="boom")
    assert tl["schema"] == AGENT_TIMELINE_SCHEMA
    assert tl["events"] == []
    assert tl["degraded"] is True
    assert tl["warnings"] == ["boom"]


def test_trace_seed_falls_back_to_session_dir_basename():
    from inference_optimizer.breakdown.agent_timeline import _trace_seed_from_breakdown

    # No claw/session id -> match the writer's correlation_seed fallback
    # (the session-dir basename).
    bd = {"session": {"session_dir": "/data/model/20260617_073000/"}}
    tid, seed = _trace_seed_from_breakdown(bd)
    assert tid is None
    assert seed == "20260617_073000"


def test_resolve_trace_id_prefers_explicit_then_seed():
    assert langfuse_reader.resolve_trace_id(trace_id="  fixed  ") == "fixed"
    assert langfuse_reader.resolve_trace_id(seed="claw-xyz") == derive_trace_id("claw-xyz")
    assert langfuse_reader.resolve_trace_id() is None


def test_coerce_output_handles_dict_and_json_string():
    assert langfuse_reader._coerce_output({"a": 1}) == {"a": 1}
    assert langfuse_reader._coerce_output(json.dumps({"a": 1})) == {"a": 1}
    assert langfuse_reader._coerce_output("not json") is None
    assert langfuse_reader._coerce_output(None) is None


def test_breakdown_recovered_from_session_breakdown_observation():
    # The writer attaches the breakdown as a `session_breakdown` span output,
    # not the trace root output, so recovery must look there.
    obs = [
        {"name": "kernel", "output": {"unrelated": 1}},
        {"name": "session_breakdown", "output": {"schema_version": "v3", "ok": True}},
    ]
    assert langfuse_reader._breakdown_from_observations(obs) == {"schema_version": "v3", "ok": True}
    assert langfuse_reader._breakdown_from_observations([{"name": "kernel"}]) is None


def test_fetch_session_breakdown_reads_embedded_observation(monkeypatch):
    trace = {
        "output": None,
        "observations": [{"name": "session_breakdown", "output": {"schema_version": "v3"}}],
    }
    monkeypatch.setattr(langfuse_reader, "fetch_trace", lambda *a, **kw: trace)
    bd = langfuse_reader.fetch_session_breakdown(trace_id="t1")
    assert bd == {"schema_version": "v3"}


def test_enrich_injects_built_timeline_when_fetch_succeeds(monkeypatch):
    target = {"session": {"claw_session_id": "claw-xyz"}}

    def fake_fetch(*, trace_id, credentials=None, timeout=30.0):
        assert trace_id == derive_trace_id("claw-xyz")
        return _full_breakdown()

    monkeypatch.setattr(
        "inference_optimizer.orchestrator.trace.langfuse_reader.fetch_session_breakdown",
        fake_fetch,
    )
    out = enrich_breakdown_with_langfuse_timeline(target)
    tl = out["agent_timeline"]
    assert tl["source"] == "langfuse"
    assert tl["trace_id"] == derive_trace_id("claw-xyz")
    assert tl["counts"] == {"orchestrator": 1, "specialist": 1, "critic": 1}


def test_enrich_injects_degraded_timeline_when_fetch_fails(monkeypatch):
    monkeypatch.setattr(
        "inference_optimizer.orchestrator.trace.langfuse_reader.fetch_session_breakdown",
        lambda **kw: None,
    )
    out = enrich_breakdown_with_langfuse_timeline({"session": {"claw_session_id": "claw-xyz"}})
    tl = out["agent_timeline"]
    assert tl["degraded"] is True
    assert tl["events"] == []


def test_enrich_handles_no_trace_seed():
    out = enrich_breakdown_with_langfuse_timeline({})
    assert out["agent_timeline"]["degraded"] is True


def test_enrich_keeps_populated_local_timeline_on_fetch_failure(monkeypatch):
    # A failed Langfuse re-derive must not clobber an already-populated
    # (locally built) timeline with a degraded empty one.
    local = build_agent_timeline(_full_breakdown(), source="local")
    target = {"session": {"claw_session_id": "claw-xyz"}, "agent_timeline": local}
    monkeypatch.setattr(
        "inference_optimizer.orchestrator.trace.langfuse_reader.fetch_session_breakdown",
        lambda **kw: None,
    )
    out = enrich_breakdown_with_langfuse_timeline(target)
    assert out["agent_timeline"] is local
    assert out["agent_timeline"]["counts"] == {"orchestrator": 1, "specialist": 1, "critic": 1}


def test_enrich_success_does_not_downgrade_populated_local_timeline(monkeypatch):
    # Successful fetch but the recovered breakdown has no decision sources ->
    # the built langfuse timeline is empty; keep the populated local one.
    local = build_agent_timeline(_full_breakdown(), source="local")
    target = {"session": {"claw_session_id": "claw-xyz"}, "agent_timeline": local}
    monkeypatch.setattr(
        "inference_optimizer.orchestrator.trace.langfuse_reader.fetch_session_breakdown",
        lambda **kw: {},  # recovered, but empty -> no events
    )
    out = enrich_breakdown_with_langfuse_timeline(target)
    assert out["agent_timeline"] is local
