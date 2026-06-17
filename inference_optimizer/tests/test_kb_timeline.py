# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit coverage for the KB decision timeline builder and observation reader.

``kb_timeline`` merges KB *read/use* decisions onto one ``ts`` axis:
recipe-snapshot reads, warm-start, warm-replay, and the critic's per-iteration
``kb_assess`` / ``kb_priors`` reads. Write-side KB activity is out of scope.
These tests pin:

* the breakdown path (warm-start/replay from ``kb_provenance`` + recipe reads
  from the audit tail + critic assess/priors) merges and sorts by ``ts``;
* the Langfuse observation path projects ``kb:recipe_snapshot`` /
  ``kb_assess:iter_N`` / ``kb_priors:iter_N`` spans and is preferred when given;
* partial / missing sources degrade gracefully;
* ``fetch_observations`` paginates and filters by name prefix;
* the upload-path bridge injects a ``degraded`` empty timeline on fetch failure.
"""

from __future__ import annotations

import pytest

from inference_optimizer.breakdown.kb_timeline import (
    KB_TIMELINE_SCHEMA,
    build_kb_timeline,
    empty_kb_timeline,
    enrich_breakdown_with_langfuse_kb_timeline,
)
from inference_optimizer.orchestrator.trace import langfuse_reader
from inference_optimizer.orchestrator.trace.langfuse_mapping import derive_trace_id


def _breakdown() -> dict:
    """A breakdown with warm-start/replay + one recipe read + critic KB reads."""
    return {
        "session": {"claw_session_id": "claw-xyz"},
        "kb_provenance": {
            "cortex_session_id": "cortex-1",
            "warm_start_ts": "2026-06-17T07:30:00Z",
            "warm_start_recipe_seen": True,
            "warm_start_recipe_tier": "exact",
            "warm_start_pitfall_count": 2,
            "warm_start_lesson_count": 0,
            "warm_history_injected": True,
            "warm_replay_attempted": True,
            "warm_replay": {"status": "promoted", "actual_gain_pct": 12.0, "reason": "ok"},
            "recipe_snapshot_reads": {
                "count": 1,
                "hits": 1,
                "tail": [
                    {
                        "ts": "2026-06-17T07:31:00Z",
                        "method": "get_recipe",
                        "remote": "gbrain",
                        "resolution": "remote",
                        "hit": True,
                        "request": {"canonical_id": "cid-1"},
                        "result": {"canonical_id": "cid-1"},
                    }
                ],
            },
        },
        "critic_robustness": {
            "critic_iterations": [
                {
                    "iter": 5,
                    "ts": "2026-06-17T08:00:00Z",
                    "kb_assess": {"configured": True, "mode": "injected", "verdict_count": 1, "referenced_in_verdict": True},
                    "kb_priors": {"configured": True, "mode": "per_proposal", "prior_count": 3, "referenced_in_verdict": False},
                }
            ]
        },
    }


def test_breakdown_path_merges_and_sorts():
    tl = build_kb_timeline(_breakdown())
    assert tl["schema"] == KB_TIMELINE_SCHEMA
    assert tl["source"] == "local"
    # 07:30 warm_start, 07:30 warm_replay (anchored), 07:31 recipe_read, 08:00 critic x2
    cats = [e["category"] for e in tl["events"]]
    assert cats[0] == "warm_start"
    assert "recipe_read" in cats
    assert cats[-2:] == ["critic_assess", "critic_priors"]
    assert tl["counts"] == {
        "recipe_read": 1,
        "warm_start": 1,
        "warm_replay": 1,
        "critic_assess": 1,
        "critic_priors": 1,
    }
    assert tl["degraded"] is False
    assert [e["seq"] for e in tl["events"]] == list(range(len(tl["events"])))


def test_recipe_read_detail_fields():
    tl = build_kb_timeline(_breakdown())
    read = next(e for e in tl["events"] if e["category"] == "recipe_read")
    assert read["detail"]["remote"] == "gbrain"
    assert read["detail"]["resolution"] == "remote"
    assert read["detail"]["hit"] is True
    assert read["detail"]["canonical_id"] == "cid-1"
    assert read["source"] == {"section": "kb_provenance.recipe_snapshot_reads.tail", "index": 0}


def test_critic_kb_use_flag_in_title():
    tl = build_kb_timeline(_breakdown())
    assess = next(e for e in tl["events"] if e["category"] == "critic_assess")
    priors = next(e for e in tl["events"] if e["category"] == "critic_priors")
    assert "used" in assess["title"]          # referenced_in_verdict True
    assert "not used" in priors["title"]       # referenced_in_verdict False


def test_observations_path_preferred_for_reads():
    observations = [
        {
            "id": "o1",
            "name": "kb:recipe_snapshot:search",
            "startTime": "2026-06-17T07:31:00Z",
            "metadata": {"method": "search", "remote": "cortex", "resolution": "remote", "hit": False},
        },
        {
            "id": "o2",
            "name": "kb_assess:iter_7",
            "startTime": "2026-06-17T07:45:00Z",
            "metadata": {"iter": 7, "verdict_count": 2, "referenced_in_verdict": True},
        },
        {"id": "o3", "name": "kernel", "startTime": "2026-06-17T07:50:00Z", "metadata": {}},
    ]
    tl = build_kb_timeline(_breakdown(), observations=observations, source="langfuse", trace_id="t1")
    cats = sorted(e["category"] for e in tl["events"])
    # warm_start + warm_replay (from kbp) + recipe_read + critic_assess (from spans);
    # the breakdown tail/critic_iterations are NOT used when observations given.
    assert cats == ["critic_assess", "recipe_read", "warm_replay", "warm_start"]
    read = next(e for e in tl["events"] if e["category"] == "recipe_read")
    assert read["source"] == {"span": "kb:recipe_snapshot:search", "observation_id": "o1", "trace_id": "t1"}
    assert tl["trace_id"] == "t1"


def test_missing_sections_degrade():
    tl = build_kb_timeline({})
    assert tl["events"] == []
    assert tl["degraded"] is True
    assert tl["warnings"]
    assert tl["counts"] == {c: 0 for c in ("recipe_read", "warm_start", "warm_replay", "critic_assess", "critic_priors")}


def test_empty_kb_timeline_well_formed():
    tl = empty_kb_timeline(reason="boom")
    assert tl["schema"] == KB_TIMELINE_SCHEMA
    assert tl["events"] == []
    assert tl["degraded"] is True
    assert tl["warnings"] == ["boom"]


def test_fetch_observations_paginates_and_filters(monkeypatch):
    pages = {
        1: {"data": [{"name": "kb:recipe_snapshot:get_recipe", "id": "a"}, {"name": "kernel", "id": "b"}], "meta": {"totalPages": 2}},
        2: {"data": [{"name": "kb_assess:iter_1", "id": "c"}], "meta": {"totalPages": 2}},
    }

    def fake_get_json(url, creds, timeout):
        page = 2 if "page=2" in url else 1
        return pages[page]

    monkeypatch.setattr(langfuse_reader, "_get_json", fake_get_json)
    creds = {"host": "https://h", "public_key": "pk", "secret_key": "sk"}
    allobs = langfuse_reader.fetch_observations("t1", credentials=creds)
    assert [o["id"] for o in allobs] == ["a", "b", "c"]
    kb_only = langfuse_reader.fetch_observations("t1", name_prefix="kb:recipe_snapshot", credentials=creds)
    assert [o["id"] for o in kb_only] == ["a"]


def test_enrich_injects_timeline_on_success(monkeypatch):
    bd = {"session": {"claw_session_id": "claw-xyz"}}
    monkeypatch.setattr(
        "inference_optimizer.orchestrator.trace.langfuse_reader.fetch_session_breakdown",
        lambda **kw: _breakdown(),
    )
    monkeypatch.setattr(
        "inference_optimizer.orchestrator.trace.langfuse_reader.fetch_observations",
        lambda *a, **kw: [],
    )
    out = enrich_breakdown_with_langfuse_kb_timeline(bd)
    tl = out["kb_timeline"]
    assert tl["source"] == "langfuse"
    assert tl["trace_id"] == derive_trace_id("claw-xyz")
    assert tl["counts"]["warm_start"] == 1


def test_enrich_injects_degraded_on_fetch_failure(monkeypatch):
    monkeypatch.setattr(
        "inference_optimizer.orchestrator.trace.langfuse_reader.fetch_session_breakdown",
        lambda **kw: None,
    )
    out = enrich_breakdown_with_langfuse_kb_timeline({"session": {"claw_session_id": "claw-xyz"}})
    assert out["kb_timeline"]["degraded"] is True
    assert out["kb_timeline"]["events"] == []
