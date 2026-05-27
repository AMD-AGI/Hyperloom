"""Tests for the direct-fact-write surface on ``CortexKBClient``.

Covers the new methods added for the KEEP / REVERT / CLOSE fact-write
contract (kg-usage-guide §3.2 / §3.4 / §3.5):

* ``propose_lesson`` / ``propose_pitfall`` shape registered ``lesson`` /
  ``pitfall`` points with the right canonical_id hashing.
* ``update_recipe`` wraps ``propose_point`` with the recipe canonical_id
  derived from (model, hardware).
* ``propose_edge`` posts to /v1/edges/propose with ``attrs.relation``
  honoured and falls back to NDJSON on transport failure.
* ``lesson_canonical_id`` / ``pitfall_canonical_id`` are stable under
  citation reorder and prefix-disjoint.

All HTTP is mocked via ``respx`` — no real Cortex service required.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from inference_optimizer.cortex_kb_client import (
    CortexKBClient,
    lesson_canonical_id,
    pitfall_canonical_id,
    recipe_canonical_id,
)
from inference_optimizer.paths import make_session_dir
from inference_optimizer.session_paths import cortex_pending_ndjson


KB_URL = "http://kb-test.local"


# ===========================================================================
# fixtures
# ===========================================================================
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("CORTEX_KB_URL", KB_URL)
    monkeypatch.delenv("KB_SERVICE_TOKEN", raising=False)
    monkeypatch.delenv("CORTEX_KB_SMOKE", raising=False)
    return make_session_dir()


# ===========================================================================
# canonical id helpers
# ===========================================================================
def test_lesson_canonical_id_is_stable_under_citation_reorder():
    a = lesson_canonical_id("X improves Y", ["c1", "c2", "c3"])
    b = lesson_canonical_id("X improves Y", ["c3", "c1", "c2"])
    assert a == b
    assert a.startswith("lesson:")
    assert len(a) == len("lesson:") + 16


def test_pitfall_canonical_id_is_disjoint_from_lesson():
    """Same text under both kinds must NOT collide."""
    statement = "Z crashes on gfx942"
    cit = ["c1"]
    lesson = lesson_canonical_id(statement, cit)
    pitfall = pitfall_canonical_id(statement, cit)
    assert lesson != pitfall
    assert pitfall.startswith("pitfall:")


def test_lesson_canonical_id_empty_citations_is_stable():
    a = lesson_canonical_id("X works", None)
    b = lesson_canonical_id("X works", [])
    assert a == b


# ===========================================================================
# propose_lesson / propose_pitfall — wrap propose_point with the right kind
# ===========================================================================
def test_propose_lesson_posts_with_kind_lesson(session_dir):
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    with respx.mock(base_url=KB_URL) as router:
        route = router.post("/v1/points/propose").mock(
            return_value=httpx.Response(200, json={
                "proposal_id": 11, "status": "auto_accepted", "point_id": 11,
            }),
        )
        out = client.propose_lesson(
            statement="X improves Y by 12%",
            measured_impact="gain_pct=12.0",
            applicable_models=["m"],
            applicable_hardware=["mi300x"],
            cited_citation_ids=["exp:36:0001"],
            evidence=["log:task-1"],
        )
    assert out["status"] == "auto_accepted"
    body = json.loads(route.calls.last.request.content)
    assert body["kind"] == "lesson"
    assert body["canonical_id"].startswith("lesson:")
    assert body["attrs"]["statement"] == "X improves Y by 12%"
    assert body["attrs"]["measured_impact"] == "gain_pct=12.0"
    assert body["attrs"]["applicable_models"] == ["m"]
    assert body["attrs"]["applicable_hardware"] == ["mi300x"]
    assert body["attrs"]["cited_citation_ids"] == ["exp:36:0001"]
    assert body["authority"] == "EXPERIENTIAL"


def test_propose_pitfall_posts_with_kind_pitfall_and_severity(session_dir):
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    with respx.mock(base_url=KB_URL) as router:
        route = router.post("/v1/points/propose").mock(
            return_value=httpx.Response(200, json={
                "proposal_id": 22, "status": "auto_accepted", "point_id": 22,
            }),
        )
        out = client.propose_pitfall(
            description="Z crashes on gfx942",
            severity="crash",
            applicable_models=["m"],
            applicable_hardware=["mi300x"],
            cited_citation_ids=["exp:36:0001"],
            evidence=["log:task-2"],
        )
    assert out["status"] == "auto_accepted"
    body = json.loads(route.calls.last.request.content)
    assert body["kind"] == "pitfall"
    assert body["canonical_id"].startswith("pitfall:")
    assert body["attrs"]["severity"] == "crash"
    assert body["attrs"]["description"] == "Z crashes on gfx942"


# ===========================================================================
# update_recipe — merges fact fields into the recipe anchor
# ===========================================================================
def test_update_recipe_uses_recipe_canonical_id(session_dir):
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    with respx.mock(base_url=KB_URL) as router:
        route = router.post("/v1/points/propose").mock(
            return_value=httpx.Response(200, json={
                "proposal_id": 33, "status": "auto_accepted", "point_id": 33,
            }),
        )
        out = client.update_recipe(
            model="DeepSeek-R1",
            hardware="MI300X",
            best_config={"extra_sglang_args": "--attention-backend AITER"},
            best_throughput=875.0,
            what_worked=[{"name": "attn-aiter", "gain_pct": 12.0}],
            what_failed=[{"name": "fp4bmm", "reason": "crash"}],
            stack_fingerprint={"sha": "abc123"},
            last_profiled="2026-05-26T08:00:00Z",
            sessions=[{"session_id": "sid-1", "gain_pct": 44.9}],
        )
    assert out["status"] == "auto_accepted"
    body = json.loads(route.calls.last.request.content)
    assert body["kind"] == "recipe"
    assert body["canonical_id"] == recipe_canonical_id("DeepSeek-R1", "MI300X")
    attrs = body["attrs"]
    assert attrs["best_config"]["extra_sglang_args"] == "--attention-backend AITER"
    assert attrs["best_throughput"] == 875.0
    assert attrs["what_worked"][0]["name"] == "attn-aiter"
    assert attrs["what_failed"][0]["reason"] == "crash"
    assert attrs["stack_fingerprint"]["sha"] == "abc123"
    assert attrs["last_profiled"] == "2026-05-26T08:00:00Z"
    assert attrs["sessions"][0]["session_id"] == "sid-1"


def test_update_recipe_omits_unset_fields_from_attrs(session_dir):
    """Caller passing only model+hardware should not erase existing
    optional fields on the server side. Our wrapper achieves this by
    omitting unset keys from the propose body so KB's shallow new-wins
    merge keeps them intact."""
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    with respx.mock(base_url=KB_URL) as router:
        route = router.post("/v1/points/propose").mock(
            return_value=httpx.Response(200, json={
                "proposal_id": 44, "status": "auto_accepted", "point_id": 44,
            }),
        )
        client.update_recipe(model="m", hardware="h")
    body = json.loads(route.calls.last.request.content)
    attrs = body["attrs"]
    assert attrs == {"model": "m", "hardware": "h"}


# ===========================================================================
# propose_edge — direct edge writes
# ===========================================================================
def test_propose_edge_resolves_canonical_ids_and_carries_relation(session_dir):
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    with respx.mock(base_url=KB_URL) as router:
        router.post("/v1/points/query").mock(side_effect=[
            httpx.Response(200, json={"points": [{"id": 10, "canonical_id": "lesson:abc"}]}),
            httpx.Response(200, json={"points": [{"id": 20, "canonical_id": "exp:42:0001"}]}),
        ])
        edge_route = router.post("/v1/edges/propose").mock(
            return_value=httpx.Response(200, json={
                "edge_id": 99, "status": "auto_accepted",
            }),
        )
        out = client.propose_edge(
            from_canonical_id="lesson:abc",
            to_canonical_id="exp:42:0001",
            edge_type="empirical",
            relation="cites",
            authority="EXPERIENTIAL",
            evidence=["log:task-1"],
        )
    assert out["status"] == "auto_accepted"
    assert out["edge_id"] == "99"
    body = json.loads(edge_route.calls.last.request.content)
    assert body["from_point"] == 10
    assert body["to_point"] == 20
    assert body["edge_type"] == "empirical"
    assert body["attrs"]["relation"] == "cites"
    assert body["authority"] == "EXPERIENTIAL"
    assert body["evidence_refs"][0]["kind"] == "log"


def test_propose_edge_falls_back_to_ndjson_on_http_failure(session_dir):
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    with respx.mock(base_url=KB_URL) as router:
        router.post("/v1/points/query").mock(
            return_value=httpx.Response(500, json={"detail": "boom"}),
        )
        out = client.propose_edge(
            from_canonical_id="lesson:abc",
            to_canonical_id="exp:42:0001",
            edge_type="empirical",
            relation="cites",
        )
    assert out["status"] == "queued"
    pending = cortex_pending_ndjson(session_dir).read_text(encoding="utf-8")
    rows = [json.loads(line) for line in pending.splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["op"] == "propose_edge"
    assert rows[0]["payload"]["from_canonical_id"] == "lesson:abc"
    assert rows[0]["payload"]["to_canonical_id"] == "exp:42:0001"
    assert rows[0]["payload"]["edge_type"] == "empirical"
    assert rows[0]["payload"]["relation"] == "cites"


def test_flush_one_does_not_duplicate_row_on_transient_failure(session_dir):
    """When ``_flush_one`` re-runs an enqueued op (e.g. propose_edge)
    and the KB is still unreachable, the row must NOT be silently
    re-enqueued. Otherwise every drain duplicates the pending entry
    and permanent errors loop forever.

    Regression test for the ``sync-with-fallback`` re-enqueue bug:
    previously ``_flush_one`` called ``propose_edge(prefer_sync=True)``
    which internally caught CortexKBError and ``_enqueue()``-d a
    duplicate, then returned ``queued`` (a non-exception), causing
    ``_flush_one`` to mis-classify as ``ok``.
    """
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    pending_path = cortex_pending_ndjson(session_dir)
    # Seed one propose_edge row while KB is down.
    with respx.mock(base_url=KB_URL) as router:
        router.post("/v1/points/query").mock(
            return_value=httpx.Response(500, json={"detail": "boom"}),
        )
        client.propose_edge(
            from_canonical_id="lesson:abc",
            to_canonical_id="exp:42:0001",
            edge_type="empirical",
            relation="cites",
        )
    rows_before = pending_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows_before) == 1, rows_before
    # Drain while KB is still down — the row should be classified as
    # transient (NOT_FOUND), counted in ``remaining`` exactly once,
    # and the file must still hold exactly one line (no duplicate).
    with respx.mock(base_url=KB_URL) as router:
        router.post("/v1/points/query").mock(
            return_value=httpx.Response(500, json={"detail": "still boom"}),
        )
        report = client.drain_pending(timeout_sec=5.0)
    assert report["drained"] == 0
    assert report["dead_letter"] == 0
    assert report["remaining"] == 1
    rows_after = pending_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows_after) == 1, rows_after
    # attempts counter has incremented from 0 to 1.
    envelope = json.loads(rows_after[0])
    assert envelope["attempts"] == 1


def test_flush_one_not_found_is_transient_not_permanent(session_dir):
    """``CortexKBError(category="business", code="NOT_FOUND")`` from
    ``_resolve_point_id`` must be classified as transient — the row
    that registers the missing ``canonical_id`` may still be ahead of
    this one in the NDJSON queue. Previously NOT_FOUND was treated
    as a permanent business error and dead-lettered immediately.
    """
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    pending_path = cortex_pending_ndjson(session_dir)
    with respx.mock(base_url=KB_URL) as router:
        router.post("/v1/points/query").mock(
            return_value=httpx.Response(500, json={"detail": "boom"}),
        )
        client.propose_edge(
            from_canonical_id="lesson:abc",
            to_canonical_id="exp:42:0001",
            edge_type="empirical",
            relation="cites",
        )
    # Now KB is reachable but the point genuinely does not exist yet
    # (empty ``points`` array → business / NOT_FOUND).
    with respx.mock(base_url=KB_URL) as router:
        router.post("/v1/points/query").mock(
            return_value=httpx.Response(200, json={"points": []}),
        )
        report = client.drain_pending(timeout_sec=5.0)
    assert report["dead_letter"] == 0, "NOT_FOUND must NOT be dead-lettered"
    assert report["remaining"] == 1


def test_drain_pending_attempts_exhausted_becomes_dead_letter(session_dir):
    """After ``MAX_FLUSH_ATTEMPTS`` consecutive transient failures the
    row must be dead-lettered so the queue doesn't grow unbounded
    when a dependency never resolves."""
    from inference_optimizer import cortex_kb_constants as C
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    pending_path = cortex_pending_ndjson(session_dir)
    # Seed one row.
    with respx.mock(base_url=KB_URL) as router:
        router.post("/v1/points/query").mock(
            return_value=httpx.Response(500, json={"detail": "boom"}),
        )
        client.propose_edge(
            from_canonical_id="lesson:abc",
            to_canonical_id="exp:42:0001",
            edge_type="empirical",
            relation="cites",
        )
    # Drain MAX_FLUSH_ATTEMPTS times while KB stays down. The last
    # iteration must dead-letter the row instead of restoring it.
    last_report = None
    for i in range(C.MAX_FLUSH_ATTEMPTS):
        with respx.mock(base_url=KB_URL) as router:
            router.post("/v1/points/query").mock(
                return_value=httpx.Response(500, json={"detail": "boom"}),
            )
            last_report = client.drain_pending(timeout_sec=5.0)
    assert last_report is not None
    assert last_report["dead_letter"] == 1, last_report
    assert last_report["remaining"] == 0, last_report
    # The pending file is now empty / gone.
    assert (
        not pending_path.exists()
        or pending_path.read_text(encoding="utf-8").strip() == ""
    )


# ===========================================================================
# find_recipe_with_fallback — _has_real_config + workload-shape T2 tier
# ===========================================================================
def test_find_recipe_recognises_hyperloom_best_config_dict_shape(session_dir):
    """Regression: ``_has_real_config`` must recognise the nested
    ``best_config`` dict written by ``update_recipe`` (post T2/T3
    retirement). Previously only the flat Arbor-shape
    ``best_config_args`` / ``best_config_envs`` was recognised, so
    every hyperloom-written recipe was silently filtered out and the
    fallback ladder returned ``miss`` no matter how many sessions
    had populated KB.
    """
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    canonical = "recipe:deepseek-r1-0528:mi300x"
    with respx.mock(base_url=KB_URL) as router:
        router.post("/v1/points/query").mock(
            return_value=httpx.Response(200, json={
                "points": [{
                    "id": 1,
                    "canonical_id": canonical,
                    "kind": "recipe",
                    "attrs": {
                        "model":    "DeepSeek-R1-0528",
                        "hardware": "mi300x",
                        "best_config": {
                            "extra_sglang_args": "--attention-backend AITER",
                            "name": "attn-aiter",
                        },
                        "best_throughput": 875.0,
                    },
                }],
            }),
        )
        point, tier, conf = client.find_recipe_with_fallback(
            workload="DeepSeek-R1-0528",
            hw="mi300x",
        )
    assert tier == "T1_exact", f"expected T1_exact, got {tier}"
    assert conf == 0.85
    assert point["attrs"]["best_config"]["name"] == "attn-aiter"


def test_find_recipe_t2_same_shape_prefers_matching_precision_tp(session_dir):
    """A KB row with same family + same hw + same precision + same tp
    must be returned as T2_same_shape (conf=0.70), preferred over a
    bare same-family match. Same-precision-different-tp must NOT
    match T2 because the dual filter is strict-AND in the KB query.
    """
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    with respx.mock(base_url=KB_URL) as router:
        # T1 exact lookup → empty
        # T2 same-shape lookup (filter precision=fp8 + tp=8) → match
        router.post("/v1/points/query").mock(side_effect=[
            httpx.Response(200, json={"points": []}),  # T1
            httpx.Response(200, json={
                "points": [{
                    "id": 11,
                    "canonical_id": "recipe:deepseek-v3:mi300x",
                    "kind": "recipe",
                    "attrs": {
                        "model":     "DeepSeek-V3",
                        "hardware":  "mi300x",
                        "precision": "fp8",
                        "tp":        8,
                        "best_config": {"extra_sglang_args": "--xxx"},
                    },
                    "confidence": 0.9,
                }],
            }),
        ])
        point, tier, conf = client.find_recipe_with_fallback(
            workload="DeepSeek-R1-0528",
            hw="mi300x",
            precision="fp8",
            tp=8,
        )
    assert tier == "T2_same_shape"
    assert conf == 0.70
    assert point["canonical_id"] == "recipe:deepseek-v3:mi300x"


def test_find_recipe_falls_through_to_t3_when_shape_misses(session_dir):
    """When the workload-shape T2 query returns nothing, the ladder
    must fall through to T3 same-family."""
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    with respx.mock(base_url=KB_URL) as router:
        router.post("/v1/points/query").mock(side_effect=[
            httpx.Response(200, json={"points": []}),  # T1 miss
            httpx.Response(200, json={"points": []}),  # T2 miss
            httpx.Response(200, json={
                "points": [{
                    "id": 22,
                    "canonical_id": "recipe:deepseek-v3:mi300x",
                    "kind": "recipe",
                    "attrs": {
                        "model":    "DeepSeek-V3",
                        "hardware": "mi300x",
                        "best_config": {"extra_sglang_args": "--y"},
                    },
                    "confidence": 0.5,
                }],
            }),  # T3 hit
        ])
        _point, tier, conf = client.find_recipe_with_fallback(
            workload="DeepSeek-R1",
            hw="mi300x",
            precision="bf16",
            tp=4,
        )
    assert tier == "T3_same_family"
    assert conf == 0.55


def test_find_recipe_t2_includes_framework_in_filter(session_dir):
    """T2 same-shape MUST filter by framework — a sglang session must
    not pick up a vLLM recipe (best_config args are framework-
    specific and would crash the server)."""
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    with respx.mock(base_url=KB_URL) as router:
        # T1 miss → T2 fires
        route = router.post("/v1/points/query").mock(side_effect=[
            httpx.Response(200, json={"points": []}),  # T1
            httpx.Response(200, json={"points": []}),  # T2 (asserted below)
            httpx.Response(200, json={"points": []}),  # T3
            httpx.Response(200, json={"points": []}),  # T4
            httpx.Response(200, json={"points": []}),  # T5
            httpx.Response(200, json={"points": []}),  # T6
        ])
        client.find_recipe_with_fallback(
            workload="DeepSeek-R1",
            hw="mi300x",
            framework="sglang",
            precision="fp8",
            tp=8,
        )
    # T2 query (2nd call) must carry framework=sglang in attrs_filter.
    t2_body = json.loads(route.calls[1].request.content)
    assert t2_body["attrs_filter"]["framework"] == "sglang"
    assert t2_body["attrs_filter"]["precision"] == "fp8"
    assert t2_body["attrs_filter"]["tp"] == 8


def test_find_recipe_t3_includes_framework_in_filter(session_dir):
    """T3 same-family MUST also filter by framework."""
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    with respx.mock(base_url=KB_URL) as router:
        route = router.post("/v1/points/query").mock(side_effect=[
            httpx.Response(200, json={"points": []}),  # T1
            # No precision/tp ⇒ T2 skipped
            httpx.Response(200, json={"points": []}),  # T3
            httpx.Response(200, json={"points": []}),  # T4
            httpx.Response(200, json={"points": []}),  # T5
            httpx.Response(200, json={"points": []}),  # T6
        ])
        client.find_recipe_with_fallback(
            workload="DeepSeek-R1",
            hw="mi300x",
            framework="sglang",
            model_class="moe_mla",
        )
    # 2nd call is T3 (T2 skipped due to no precision/tp).
    t3_body = json.loads(route.calls[1].request.content)
    assert t3_body["attrs_filter"]["framework"] == "sglang"
    assert t3_body["attrs_filter"]["hardware"] == "mi300x"
    # 3rd call is T4 (model_class) — must also have framework filter.
    t4_body = json.loads(route.calls[2].request.content)
    assert t4_body["attrs_filter"]["framework"] == "sglang"
    assert t4_body["attrs_filter"]["model_class"] == "moe_mla"


def test_find_recipe_t2_includes_ep_in_filter(session_dir):
    """T2 same-shape MUST include ``ep`` in the filter when supplied.
    EP=1 (TP-shared experts) vs EP=TP (rank-local experts) produce
    very different best_configs (``--enable-expert-parallel`` flag
    + MoE schedule changes) — picking one for the other crashes the
    server or silently drops throughput."""
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    with respx.mock(base_url=KB_URL) as router:
        route = router.post("/v1/points/query").mock(side_effect=[
            httpx.Response(200, json={"points": []}),  # T1
            httpx.Response(200, json={"points": []}),  # T2
            httpx.Response(200, json={"points": []}),  # T3
            httpx.Response(200, json={"points": []}),  # T4
            httpx.Response(200, json={"points": []}),  # T5
            httpx.Response(200, json={"points": []}),  # T6
        ])
        client.find_recipe_with_fallback(
            workload="DeepSeek-R1", hw="mi300x",
            framework="sglang",
            precision="fp8", tp=8, ep=8,
        )
    t2_body = json.loads(route.calls[1].request.content)
    assert t2_body["attrs_filter"]["ep"] == 8


def test_find_recipe_t2_fires_with_only_ep_when_precision_tp_missing(session_dir):
    """``ep`` alone is enough to trigger T2 same-shape (the OR guard
    is precision OR tp OR ep)."""
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    with respx.mock(base_url=KB_URL) as router:
        route = router.post("/v1/points/query").mock(side_effect=[
            httpx.Response(200, json={"points": []}),  # T1
            httpx.Response(200, json={"points": []}),  # T2 (must fire)
            httpx.Response(200, json={"points": []}),  # T3
            httpx.Response(200, json={"points": []}),  # T5 (T4 skipped — no model_class)
            httpx.Response(200, json={"points": []}),  # T6
        ])
        client.find_recipe_with_fallback(
            workload="DeepSeek-R1", hw="mi300x",
            framework="sglang",
            ep=4,  # only ep — model_class omitted so T4 is skipped
        )
    # The 2nd call must be T2 (not T3). With the old "precision OR tp"
    # guard ep alone would have skipped T2 and the 2nd call would be
    # T3 (which has no precision / tp / ep keys).
    t2_body = json.loads(route.calls[1].request.content)
    assert t2_body["attrs_filter"]["ep"] == 4
    assert "precision" not in t2_body["attrs_filter"]
    assert "tp" not in t2_body["attrs_filter"]


def test_find_recipe_no_framework_falls_back_to_unfiltered(session_dir):
    """When the caller doesn't pin a framework, the fallback ladder
    must NOT add a framework filter (avoids accidentally filtering
    out otherwise-valid same-family recipes during e.g. tests)."""
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    with respx.mock(base_url=KB_URL) as router:
        route = router.post("/v1/points/query").mock(side_effect=[
            httpx.Response(200, json={"points": []}),
        ] * 6)
        client.find_recipe_with_fallback(
            workload="DeepSeek-R1",
            hw="mi300x",
            # framework omitted
        )
    # T3 query (2nd call: T1 miss → T2 skipped → T3) — attrs_filter
    # must NOT contain 'framework' key.
    t3_body = json.loads(route.calls[1].request.content)
    assert "framework" not in t3_body["attrs_filter"]


# ===========================================================================
# read_recipe_exact + sessions[] read-modify-write
# ===========================================================================
def test_read_recipe_exact_returns_point_dict(session_dir):
    """``read_recipe_exact`` queries the (model, hw) anchor and
    returns the parsed point dict (or ``{}`` on miss)."""
    from inference_optimizer.cortex_kb_client import recipe_canonical_id
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    expected_cid = recipe_canonical_id("DeepSeek-R1", "MI300X")
    with respx.mock(base_url=KB_URL) as router:
        route = router.post("/v1/points/query").mock(
            return_value=httpx.Response(200, json={
                "points": [{
                    "id": 99,
                    "canonical_id": expected_cid,
                    "kind": "recipe",
                    "attrs": {
                        "model": "DeepSeek-R1",
                        "hardware": "mi300x",
                        "sessions": [
                            {"session_id": "A", "gain_pct": 12.0},
                            {"session_id": "B", "gain_pct": 18.5},
                        ],
                    },
                }],
            }),
        )
        point = client.read_recipe_exact(model="DeepSeek-R1", hardware="MI300X")
    assert point["id"] == 99
    assert point["canonical_id"] == expected_cid
    body = json.loads(route.calls.last.request.content)
    assert body["canonical_id"] == expected_cid
    assert body["kind"] == "recipe"


def test_read_recipe_exact_returns_empty_dict_on_miss(session_dir):
    """When the anchor doesn't exist yet, ``read_recipe_exact`` returns
    ``{}`` — caller treats it as 'no prior sessions'."""
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    with respx.mock(base_url=KB_URL) as router:
        router.post("/v1/points/query").mock(
            return_value=httpx.Response(200, json={"points": []}),
        )
        point = client.read_recipe_exact(model="NewModel", hardware="mi300x")
    assert point == {}


def test_read_recipe_exact_returns_empty_dict_on_http_failure(session_dir):
    """Network / 5xx failure → ``{}`` so the caller can proceed with
    a clean write (last-writer-wins is preferable to crashing CLOSE)."""
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    with respx.mock(base_url=KB_URL) as router:
        router.post("/v1/points/query").mock(
            return_value=httpx.Response(500, json={"detail": "boom"}),
        )
        point = client.read_recipe_exact(model="M", hardware="h")
    assert point == {}


def test_update_recipe_with_extra_workload_attrs_lands_them_flat(session_dir):
    """The workload-shape tags coordinator hoists into ``extra_attrs``
    must land as top-level recipe attrs (so the KB ``attrs_filter``
    can match them), NOT nested under a ``workload`` sub-dict."""
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    with respx.mock(base_url=KB_URL) as router:
        route = router.post("/v1/points/propose").mock(
            return_value=httpx.Response(200, json={
                "proposal_id": 1, "status": "auto_accepted", "point_id": 1,
            }),
        )
        client.update_recipe(
            model="DeepSeek-R1",
            hardware="MI300X",
            best_config={"extra_sglang_args": "--x"},
            extra_attrs={
                "precision":         "fp8",
                "tp":                8,
                "isl":               1024,
                "framework_version": "0.5.11",
            },
        )
    body = json.loads(route.calls.last.request.content)
    attrs = body["attrs"]
    assert attrs["precision"] == "fp8"
    assert attrs["tp"] == 8
    assert attrs["isl"] == 1024
    assert attrs["framework_version"] == "0.5.11"
    # And the canonical_id is still keyed by (model, hardware) only —
    # workload-shape tags are searchable but NOT part of identity.
    assert body["canonical_id"] == "recipe:deepseek-r1:mi300x"


# ===========================================================================
# Coordinator.cortex_finalize_recipe_and_journal — sessions[] read-modify-write
# ===========================================================================
@pytest.fixture
def _coord_with_kb_stub(session_dir):
    """Minimal Coordinator stub for the finalize-recipe path.

    Uses ``Coordinator.__new__`` so we skip the heavy __init__ and
    wire only the attributes the helper reads.
    """
    from dataclasses import dataclass, field
    from inference_optimizer.orchestrator.coordinator import Coordinator

    @dataclass
    class _StubSharedState:
        model_name: str = "DeepSeek-R1"
        gpu_type: str = "MI300X"
        framework: str = "sglang"
        cortex_session_id: str = "session-NEW"
        current_best: dict = field(default_factory=lambda: {
            "tput": 875.0, "extra_sglang_args": "--attention-backend AITER",
        })
        optimization_stack: list = field(default_factory=list)
        gain_per_stack_entry: list = field(default_factory=list)
        last_action_failures: list = field(default_factory=list)
        cumulative_gain: float = 12.5
        cumulative_gain_validated: float = 12.5
        cumulative_gain_validated_stack_len: int = 1
        cumulative_gain_validated_ts: str = "2026-05-27T00:00:00Z"
        stack_fingerprint: str = "abc123"
        phase: str = "CLOSE"
        tick: int = 5
        baseline_tput: float = 700.0
        precision: str = "bf16"
        tp: int = 8
        ep: int = 0
        conc: int = 64
        isl: int = 1024
        osl: int = 1024
        max_model_len: int = 0

        def save(self, *args, **kwargs):
            pass

    class _StubKB:
        enabled = True

        def __init__(self, prior_sessions):
            self._prior_sessions = prior_sessions
            self.read_calls = 0
            self.update_recipe_calls = []
            self.read_recipe_calls = []

        def read_recipe_exact(self, *, model, hardware):
            self.read_calls += 1
            self.read_recipe_calls.append((model, hardware))
            return {
                "id": 42,
                "canonical_id": f"recipe:{model.lower()}:{hardware.lower()}",
                "kind": "recipe",
                "attrs": {
                    "model": model,
                    "hardware": hardware,
                    "sessions": list(self._prior_sessions),
                },
            }

        def update_recipe(self, **kwargs):
            self.update_recipe_calls.append(kwargs)
            return {"status": "auto_accepted", "point_id": "42"}

    def _build(prior_sessions=None, my_session_id="session-NEW"):
        coord = Coordinator.__new__(Coordinator)
        coord.session_dir = session_dir
        coord.shared_state = _StubSharedState(cortex_session_id=my_session_id)
        coord.cortex_kb = _StubKB(prior_sessions or [])
        coord._fact_writes_enabled = True
        return coord
    return _build


def test_finalize_recipe_appends_new_session_keeping_history(_coord_with_kb_stub):
    """update_recipe must include prior sessions[] entries alongside the
    new one — otherwise the KB shallow-merge overwrites and loses history."""
    coord = _coord_with_kb_stub(
        prior_sessions=[
            {"session_id": "session-A", "gain_pct": 8.0, "stack_len": 1},
            {"session_id": "session-B", "gain_pct": 15.0, "stack_len": 2},
        ],
        my_session_id="session-NEW",
    )
    coord.cortex_finalize_recipe_and_journal()
    assert len(coord.cortex_kb.update_recipe_calls) == 1
    written_sessions = coord.cortex_kb.update_recipe_calls[0]["sessions"]
    ids = [s["session_id"] for s in written_sessions]
    assert ids == ["session-A", "session-B", "session-NEW"]


def test_finalize_recipe_dedup_resume_replays_own_session_id(
    _coord_with_kb_stub,
):
    """Resume / retry of the SAME session must not duplicate the session
    in sessions[] — the new entry supersedes the prior copy carrying
    the latest gain_pct / stack_len."""
    coord = _coord_with_kb_stub(
        prior_sessions=[
            {"session_id": "session-A", "gain_pct": 8.0, "stack_len": 1},
            {"session_id": "session-NEW", "gain_pct": 5.0, "stack_len": 1},
        ],
        my_session_id="session-NEW",
    )
    coord.cortex_finalize_recipe_and_journal()
    written_sessions = coord.cortex_kb.update_recipe_calls[0]["sessions"]
    ids = [s["session_id"] for s in written_sessions]
    assert ids == ["session-A", "session-NEW"]
    # Latest copy wins — gain_pct reflects current cumulative_gain (12.5),
    # not the stale 5.0 from the prior entry.
    new_entry = next(s for s in written_sessions if s["session_id"] == "session-NEW")
    assert new_entry["gain_pct"] == 12.5


def test_finalize_recipe_empty_anchor_only_writes_own_entry(
    _coord_with_kb_stub,
):
    """First-time write (no prior anchor) → sessions[] is just our entry."""
    coord = _coord_with_kb_stub(prior_sessions=[], my_session_id="session-NEW")
    coord.cortex_finalize_recipe_and_journal()
    written_sessions = coord.cortex_kb.update_recipe_calls[0]["sessions"]
    assert len(written_sessions) == 1
    assert written_sessions[0]["session_id"] == "session-NEW"


def test_finalize_recipe_prefers_shared_state_ep_over_env(
    _coord_with_kb_stub, monkeypatch,
):
    """CLOSE-time finalize must prefer ``SharedState.ep`` over the
    ``EP`` env var (resume-safety mirror of the T0 logic). When
    SharedState has ep=8 but env is empty / different, the recipe
    write must carry the SharedState value."""
    monkeypatch.delenv("EP", raising=False)
    coord = _coord_with_kb_stub()
    coord.shared_state.ep = 8
    coord.cortex_finalize_recipe_and_journal()
    extra = coord.cortex_kb.update_recipe_calls[0]["extra_attrs"]
    assert extra["ep"] == 8


def test_finalize_recipe_falls_back_to_env_ep_when_shared_state_unset(
    _coord_with_kb_stub, monkeypatch,
):
    monkeypatch.setenv("EP", "4")
    coord = _coord_with_kb_stub()
    coord.shared_state.ep = 0  # not seeded
    coord.cortex_finalize_recipe_and_journal()
    extra = coord.cortex_kb.update_recipe_calls[0]["extra_attrs"]
    assert extra["ep"] == 4


def test_finalize_recipe_includes_workload_tags_in_extra_attrs(
    _coord_with_kb_stub,
):
    """The workload-shape tags (precision/tp/conc/isl/osl + ep/pp from
    env) AND the framework / model_class CLOSE-time backstops must be
    hoisted into ``extra_attrs`` so they land flat on the recipe
    anchor (NOT nested under ``workload``). Framework / model_class
    are written here as a backstop in case T0 backfill was skipped
    (KB unreachable at PRELUDE) — without them the future fallback
    queries' framework filter would never match this recipe."""
    import os as _os
    coord = _coord_with_kb_stub()
    # The stub sets framework="sglang" + model_class="" by default;
    # set model_class so the assertion exercises that branch too.
    coord.shared_state.model_class = "moe_mla"
    prev_ep = _os.environ.get("EP")
    _os.environ["EP"] = "4"
    try:
        coord.cortex_finalize_recipe_and_journal()
    finally:
        if prev_ep is None:
            _os.environ.pop("EP", None)
        else:
            _os.environ["EP"] = prev_ep
    extra = coord.cortex_kb.update_recipe_calls[0]["extra_attrs"]
    # CLOSE-time backstop tags (framework + model_class).
    assert extra["framework"] == "sglang"
    assert extra["model_class"] == "moe_mla"
    # Workload-shape tags.
    assert extra["precision"] == "bf16"
    assert extra["tp"] == 8
    assert extra["conc"] == 64
    assert extra["isl"] == 1024
    assert extra["osl"] == 1024
    assert extra["ep"] == 4


# ===========================================================================
# Coordinator._record_fact_per_task / _record_fact_per_variant — direct unit tests
# ===========================================================================
@pytest.fixture
def _coord_for_fact_writes(session_dir):
    """Coordinator stub with KB call-recording so we can assert the
    exact ``propose_lesson`` / ``propose_pitfall`` calls per outcome."""
    from dataclasses import dataclass, field
    from inference_optimizer.orchestrator.coordinator import Coordinator

    @dataclass
    class _StubSharedState:
        model_name: str = "DeepSeek-R1"
        gpu_type: str = "MI300X"
        framework: str = "sglang"
        cortex_session_id: str = "session-X"
        tick: int = 7
        phase: str = "EXPLORE"
        baseline_tput: float = 700.0

        def save(self, *args, **kwargs):
            pass

    class _StubKB:
        enabled = True

        def __init__(self):
            self.lesson_calls: list[dict] = []
            self.pitfall_calls: list[dict] = []

        def propose_lesson(self, **kwargs):
            self.lesson_calls.append(kwargs)
            return {"status": "auto_accepted"}

        def propose_pitfall(self, **kwargs):
            self.pitfall_calls.append(kwargs)
            return {"status": "auto_accepted"}

    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = session_dir
    coord.shared_state = _StubSharedState()
    coord.cortex_kb = _StubKB()
    coord._fact_writes_enabled = True
    return coord


class _Task:
    """Minimal Task stub — _record_fact_per_task only reads .kind / .task_id."""
    def __init__(self, task_id: str = "task-1", kind: str = "kernel_opt"):
        self.task_id = task_id
        self.kind = kind


def test_record_fact_per_task_keep_writes_lesson(_coord_for_fact_writes):
    """KEEP with gain > 0 → exactly one propose_lesson call, no pitfall."""
    coord = _coord_for_fact_writes
    coord._record_fact_per_task(
        task=_Task("t-1", "kernel_opt"),
        source_session_id="session-X",
        result_dict={"gain_pct": 12.3, "output_throughput": 875.0},
        kept=True,
    )
    assert len(coord.cortex_kb.lesson_calls) == 1
    assert len(coord.cortex_kb.pitfall_calls) == 0
    lesson = coord.cortex_kb.lesson_calls[0]
    assert "+12.30%" in lesson["statement"]
    assert "DeepSeek-R1" in lesson["statement"]
    assert "mi300x" in lesson["statement"].lower()
    assert lesson["source_session_id"] == "session-X"
    assert lesson["source_task_id"] == "t-1"
    assert lesson["applicable_models"] == ["DeepSeek-R1"]
    assert lesson["applicable_hardware"] == ["MI300X"]


def test_record_fact_per_task_keep_zero_gain_skips_lesson(_coord_for_fact_writes):
    """KEEP with gain_pct == 0 → no lesson (avoids noise)."""
    coord = _coord_for_fact_writes
    coord._record_fact_per_task(
        task=_Task("t-2", "kernel_opt"),
        source_session_id="session-X",
        result_dict={"gain_pct": 0.0, "output_throughput": 700.0},
        kept=True,
    )
    assert coord.cortex_kb.lesson_calls == []
    assert coord.cortex_kb.pitfall_calls == []


def test_record_fact_per_task_revert_crash_writes_pitfall(_coord_for_fact_writes):
    """REVERT with error_class=crash → pitfall (severity=crash)."""
    coord = _coord_for_fact_writes
    coord._record_fact_per_task(
        task=_Task("t-3", "kernel_opt"),
        source_session_id="session-X",
        result_dict={
            "gain_pct": None,
            "error_class": "crash",
            "reason": "segfault",
        },
        kept=False,
    )
    assert len(coord.cortex_kb.lesson_calls) == 0
    assert len(coord.cortex_kb.pitfall_calls) == 1
    pitfall = coord.cortex_kb.pitfall_calls[0]
    assert pitfall["severity"] == "crash"
    assert pitfall["source_session_id"] == "session-X"


def test_record_fact_per_task_revert_minor_drop_skips_pitfall(
    _coord_for_fact_writes,
):
    """REVERT with gain_pct = -1% (above -5% threshold) → no pitfall
    (noise control — Threshold-B contract)."""
    coord = _coord_for_fact_writes
    coord._record_fact_per_task(
        task=_Task("t-4", "kernel_opt"),
        source_session_id="session-X",
        result_dict={"gain_pct": -1.0, "output_throughput": 693.0},
        kept=False,
    )
    assert coord.cortex_kb.lesson_calls == []
    assert coord.cortex_kb.pitfall_calls == []


def test_record_fact_per_task_revert_large_regression_writes_pitfall(
    _coord_for_fact_writes,
):
    """REVERT with gain_pct = -8% (below -5% threshold) → pitfall."""
    coord = _coord_for_fact_writes
    coord._record_fact_per_task(
        task=_Task("t-5", "kernel_opt"),
        source_session_id="session-X",
        result_dict={"gain_pct": -8.0, "output_throughput": 644.0},
        kept=False,
    )
    assert len(coord.cortex_kb.pitfall_calls) == 1
    assert coord.cortex_kb.pitfall_calls[0]["severity"] == "regress"


def test_record_fact_per_task_no_fact_writes_flag_skips_kb(
    _coord_for_fact_writes,
):
    """--no-fact-writes (``_fact_writes_enabled=False``) skips KB calls
    but still writes journal — independent gates."""
    coord = _coord_for_fact_writes
    coord._fact_writes_enabled = False
    coord._record_fact_per_task(
        task=_Task("t-6", "kernel_opt"),
        source_session_id="session-X",
        result_dict={"gain_pct": 15.0, "output_throughput": 805.0},
        kept=True,
    )
    assert coord.cortex_kb.lesson_calls == []
    # journal still got the row.
    j = coord._ensure_journal()
    assert any(e.task_id == "t-6" for e in j.entries)


def test_record_fact_per_variant_keep_writes_lesson_with_variant_name(
    _coord_for_fact_writes,
):
    """Explore-grid KEEP variant → lesson tagged with source_variant_name."""
    coord = _coord_for_fact_writes
    coord._record_fact_per_variant(
        task=_Task("t-7", "explore"),
        source_session_id="session-X",
        variant_outcome={
            "variant_name": "attn_aiter",
            "outcome": "KEEP",
            "variant": {
                "name": "attn_aiter",
                "extra_sglang_args": "--attention-backend AITER",
            },
            "metrics": {"gain_pct": 9.0, "output_throughput": 763.0},
        },
    )
    assert len(coord.cortex_kb.lesson_calls) == 1
    lesson = coord.cortex_kb.lesson_calls[0]
    assert lesson["source_variant_name"] == "attn_aiter"
    assert "--attention-backend AITER" in lesson["statement"]


def test_record_fact_per_variant_skipped_dedup_writes_nothing(
    _coord_for_fact_writes,
):
    """SKIPPED_DEDUP variant must NOT write a journal entry OR a
    KB lesson / pitfall (it's a no-op — the variant didn't actually
    run, so there's nothing to attribute)."""
    coord = _coord_for_fact_writes
    coord._record_fact_per_variant(
        task=_Task("t-8", "explore"),
        source_session_id="session-X",
        variant_outcome={
            "variant_name": "dedup_target",
            "outcome": "SKIPPED_DEDUP",
        },
    )
    assert coord.cortex_kb.lesson_calls == []
    assert coord.cortex_kb.pitfall_calls == []
    j = coord._ensure_journal()
    assert not any(e.variant_name == "dedup_target" for e in j.entries)


# ===========================================================================
# client.lessons() — T0 reader symmetric with traps()
# ===========================================================================
def test_lessons_query_filters_by_model_hardware_framework(session_dir):
    """``lessons()`` posts an ``attrs_filter`` with the three discriminators
    that match what ``propose_lesson`` writes (applicable_models /
    applicable_hardware / framework). Critical for ensuring a sglang
    session doesn't pick up vLLM-only lesson statements."""
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    with respx.mock(base_url=KB_URL) as router:
        route = router.post("/v1/points/query").mock(
            return_value=httpx.Response(200, json={"points": []}),
        )
        client.lessons(model="DeepSeek-R1", hardware="mi300x", framework="sglang")
    body = json.loads(route.calls.last.request.content)
    assert body["kind"] == "lesson"
    assert body["attrs_filter"]["applicable_models"] == "DeepSeek-R1"
    assert body["attrs_filter"]["applicable_hardware"] == "mi300x"
    assert body["attrs_filter"]["framework"] == "sglang"


def test_lessons_returns_points_sorted_by_confidence_desc(session_dir):
    """Higher-confidence lessons must surface first so the specialist
    prompt's truncated list shows the most authoritative ones."""
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    with respx.mock(base_url=KB_URL) as router:
        router.post("/v1/points/query").mock(
            return_value=httpx.Response(200, json={
                "points": [
                    {"id": 1, "canonical_id": "lesson:a", "kind": "lesson",
                     "attrs": {"statement": "low-conf"}, "confidence": 0.3},
                    {"id": 2, "canonical_id": "lesson:b", "kind": "lesson",
                     "attrs": {"statement": "high-conf"}, "confidence": 0.95},
                    {"id": 3, "canonical_id": "lesson:c", "kind": "lesson",
                     "attrs": {"statement": "mid-conf"}, "confidence": 0.6},
                ],
            }),
        )
        out = client.lessons(model="m", hardware="h")
    ids = [p["canonical_id"] for p in out]
    assert ids == ["lesson:b", "lesson:c", "lesson:a"]


def test_lessons_returns_empty_on_disabled_or_failure(session_dir):
    """Both disabled-client and HTTP-failure paths return ``[]`` — the
    warm-start surface is best-effort, never crash PRELUDE."""
    # disabled
    disabled = CortexKBClient(session_dir=session_dir, enabled=False, kb_url=KB_URL)
    assert disabled.lessons(model="m", hardware="h") == []
    # HTTP failure
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    with respx.mock(base_url=KB_URL) as router:
        router.post("/v1/points/query").mock(
            return_value=httpx.Response(500, json={"detail": "boom"}),
        )
        assert client.lessons(model="m", hardware="h") == []


def test_propose_edge_drain_replays_to_edge_endpoint(session_dir):
    """End-to-end fallback contract: when ``propose_edge`` sync failed
    and the row was enqueued, a later :meth:`drain_pending` must
    successfully replay it against ``/v1/edges/propose`` — NOT route
    it to dead-letter.

    Regression test: ``_flush_one`` previously had no ``propose_edge``
    branch and fell through to ``return "permanent"``, silently
    dropping every queued edge write the next time the KB recovered.
    """
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    # Step 1 — enqueue with KB down.
    with respx.mock(base_url=KB_URL) as router:
        router.post("/v1/points/query").mock(
            return_value=httpx.Response(500, json={"detail": "boom"}),
        )
        client.propose_edge(
            from_canonical_id="lesson:abc",
            to_canonical_id="exp:42:0001",
            edge_type="empirical",
            relation="cites",
            evidence=["log:task-1"],
        )
    pending_path = cortex_pending_ndjson(session_dir)
    assert pending_path.exists()
    # Step 2 — KB recovers; drain should replay the row against the
    # real edges endpoint and clear the pending file.
    with respx.mock(base_url=KB_URL) as router:
        router.post("/v1/points/query").mock(side_effect=[
            httpx.Response(200, json={"points": [{"id": 10, "canonical_id": "lesson:abc"}]}),
            httpx.Response(200, json={"points": [{"id": 20, "canonical_id": "exp:42:0001"}]}),
        ])
        edge_route = router.post("/v1/edges/propose").mock(
            return_value=httpx.Response(200, json={
                "edge_id": 99, "status": "auto_accepted",
            }),
        )
        report = client.drain_pending(timeout_sec=5.0)
    assert report["drained"] == 1
    assert report["remaining"] == 0
    assert report["dead_letter"] == 0
    assert edge_route.called
    body = json.loads(edge_route.calls.last.request.content)
    assert body["from_point"] == 10
    assert body["to_point"] == 20
    assert body["edge_type"] == "empirical"
    assert body["attrs"]["relation"] == "cites"
    # Pending file is consumed (drain removes the snapshot when all
    # rows succeed and leftover_lines is empty).
    assert (
        not pending_path.exists()
        or pending_path.read_text(encoding="utf-8").strip() == ""
    )


# ===========================================================================
# disabled client — fact writes are no-ops
# ===========================================================================
def test_disabled_client_skips_fact_writes(session_dir):
    client = CortexKBClient(session_dir=session_dir, enabled=False, kb_url=KB_URL)
    assert client.propose_lesson(
        statement="X", measured_impact="y",
    )["status"] == "skip_disabled"
    assert client.propose_pitfall(
        description="Z", severity="crash",
    )["status"] == "skip_disabled"
    assert client.update_recipe(
        model="m", hardware="h",
    )["status"] == "skip_disabled"
    assert client.propose_edge(
        from_canonical_id="a", to_canonical_id="b", edge_type="empirical",
    )["status"] == "skip_disabled"
    assert not cortex_pending_ndjson(session_dir).exists()
