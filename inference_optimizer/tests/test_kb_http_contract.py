"""Contract tests for the Cortex KB HTTP client.

Locks the wire shape against the schema captured in
``cortex-kb-http-branch-b-2026-05-20.md`` (= primus-cortex
``f48a785`` / image ``kbsg-edeb3d1``). Each test asserts:

* request body field names match the documented Pydantic models;
* ``extra="forbid"`` fields don't appear (no rogue ``task`` /
  ``evidence``);
* enum literals come from :mod:`cortex_kb_constants`.

When the KB backend bumps schema, regen ``openapi.json`` (out of
band) and update :mod:`cortex_kb_constants` + this test in lockstep.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from inference_optimizer import cortex_kb_constants as C
from inference_optimizer.cortex_kb_client import CortexKBClient
from inference_optimizer.paths import make_session_dir


KB_URL = "http://kb-test.local"


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("CORTEX_KB_URL", KB_URL)
    monkeypatch.delenv("KB_SERVICE_TOKEN", raising=False)
    monkeypatch.delenv("CORTEX_KB_SMOKE", raising=False)
    return make_session_dir()


@pytest.fixture
def client(session_dir) -> CortexKBClient:
    return CortexKBClient(session_dir=session_dir, kb_url=KB_URL)


# ---------------------------------------------------------------------------
# §1.1 — POST /v1/sessions/begin
# ---------------------------------------------------------------------------
def test_session_begin_body_matches_schema(client):
    """Required: goal + initiator. ``task`` must be absent (extra=forbid).
    ``thinking_style`` and ``attrs`` are optional.
    """
    with respx.mock(base_url=KB_URL) as router:
        route = router.post(C.PATH_BEGIN).mock(
            return_value=httpx.Response(200, json={
                "session_id": 1, "thinking_style": "recommendation", "lens_schedule": [],
            }),
        )
        client.session_begin(
            workload="w", hw="mi300x",
            stack_fingerprint={"rocm": "7.2.0"},
            extra_attrs={"framework": "sglang"},
        )
    body = json.loads(route.calls.last.request.content)
    required = {C.F_GOAL, C.F_INITIATOR}
    allowed = required | {C.F_THINKING_STYLE, C.F_ATTRS}
    assert required.issubset(body.keys())
    assert set(body.keys()).issubset(allowed), f"extra fields: {set(body)-allowed}"
    assert body[C.F_GOAL] in {
        C.GOAL_FIND_ROOT_CAUSE, C.GOAL_FIND_MIGRATION_PLAN,
        C.GOAL_FIND_RECOMMENDATION, C.GOAL_VERIFY_HYPOTHESIS,
        C.GOAL_EXPLORE_DOMAIN, C.GOAL_AUDIT_CONSISTENCY,
    }


# ---------------------------------------------------------------------------
# §1.2 — POST /v1/sessions/{sid}/hypothesize
# ---------------------------------------------------------------------------
def test_hypothesize_body_matches_schema(client):
    """Required: from_point (int), to_point (int), edge_type, reason."""
    with respx.mock(base_url=KB_URL) as router:
        router.post(C.PATH_QUERY_POINT).mock(side_effect=[
            httpx.Response(200, json={"points": [{"id": 100, "canonical_id": "x"}]}),
            httpx.Response(200, json={"points": [{"id": 200, "canonical_id": "y"}]}),
        ])
        hyp = router.post(C.PATH_HYPOTHESIZE.format(session_id="9")).mock(
            return_value=httpx.Response(200, json={"tentative_edge_id": 7}),
        )
        client.hypothesize(
            sid="9", from_canonical="x", to_canonical="y",
            edge_type=C.EDGE_HYPOTHETICAL, reason="why",
        )
    body = json.loads(hyp.calls.last.request.content)
    required = {C.F_FROM_POINT, C.F_TO_POINT, C.F_EDGE_TYPE, C.F_REASON}
    allowed = required | {C.F_ATTRS}
    assert required.issubset(body.keys())
    assert set(body.keys()).issubset(allowed)
    assert isinstance(body[C.F_FROM_POINT], int) and body[C.F_FROM_POINT] == 100
    assert isinstance(body[C.F_TO_POINT], int) and body[C.F_TO_POINT] == 200
    assert body[C.F_EDGE_TYPE] in {
        C.EDGE_STRUCTURAL, C.EDGE_CAUSAL, C.EDGE_INVESTIGATION,
        C.EDGE_EVOLUTIONARY, C.EDGE_EMPIRICAL, C.EDGE_HYPOTHETICAL,
        C.EDGE_NEGATION,
    }


# ---------------------------------------------------------------------------
# §1.3 — POST /v1/sessions/{sid}/verify
# ---------------------------------------------------------------------------
def test_verify_body_matches_schema(client):
    """Required: tentative_edge_id (int), outcome, evidence_refs,
    promoted_authority. (verify only fires via _flush_one — exercise
    that path directly.)
    """
    with respx.mock(base_url=KB_URL) as router:
        v_route = router.post(C.PATH_VERIFY.format(session_id="9")).mock(
            return_value=httpx.Response(200, json={
                "outcome": "confirmed", "promoted_edge_id": 42, "negation_edge_id": None,
            }),
        )
        client._verify_sync(
            sid="9", edge_id="7", outcome=C.OUTCOME_CONFIRMED,
            evidence=["log:proof"], promote_authority=C.AUTHORITY_EXPERIENTIAL,
        )
    body = json.loads(v_route.calls.last.request.content)
    required = {C.F_TENTATIVE_EDGE_ID, C.F_OUTCOME, C.F_EVIDENCE_REFS, C.F_PROMOTED_AUTHORITY}
    assert required.issubset(body.keys())
    assert isinstance(body[C.F_TENTATIVE_EDGE_ID], int)
    assert body[C.F_OUTCOME] in {C.OUTCOME_CONFIRMED, C.OUTCOME_REFUTED}
    assert body[C.F_PROMOTED_AUTHORITY] in {
        C.AUTHORITY_AUTHORITATIVE, C.AUTHORITY_EXPERIENTIAL,
        C.AUTHORITY_INFERRED, C.AUTHORITY_HYPOTHESIZED,
    }
    for ev in body[C.F_EVIDENCE_REFS]:
        assert ev[C.F_EV_KIND] in {
            C.EV_KIND_URL, C.EV_KIND_COMMIT, C.EV_KIND_PROFILE_FILE,
            C.EV_KIND_LOG, C.EV_KIND_POINT_ID, C.EV_KIND_EDGE_ID,
        }


# ---------------------------------------------------------------------------
# §1.4 — POST /v1/sessions/{sid}/commit
# ---------------------------------------------------------------------------
def test_commit_body_is_empty(client):
    with respx.mock(base_url=KB_URL) as router:
        route = router.post(C.PATH_COMMIT.format(session_id="9")).mock(
            return_value=httpx.Response(200, json={
                "status": "committed", "promoted_edges": [], "derived_summary_id": None,
            }),
        )
        client.session_commit("9")
    body = json.loads(route.calls.last.request.content)
    assert body == {}


# ---------------------------------------------------------------------------
# §1.5 — POST /v1/sessions/{sid}/abort
# ---------------------------------------------------------------------------
def test_abort_body_is_empty(client):
    with respx.mock(base_url=KB_URL) as router:
        route = router.post(C.PATH_ABORT.format(session_id="9")).mock(
            return_value=httpx.Response(200, json={
                "status": "aborted", "trace_preserved": True,
            }),
        )
        client.session_abort("9", reason="testing")
    body = json.loads(route.calls.last.request.content)
    assert body == {}


# ---------------------------------------------------------------------------
# §1.6 — POST /v1/points/propose
# ---------------------------------------------------------------------------
def test_propose_point_body_matches_schema(client):
    """Required: canonical_id, kind, authority, evidence_refs (non-empty),
    provenance. ``evidence`` (legacy name) must NOT be present.
    """
    with respx.mock(base_url=KB_URL) as router:
        route = router.post(C.PATH_PROPOSE_POINT).mock(
            return_value=httpx.Response(200, json={
                "proposal_id": 5, "status": "auto_accepted", "point_id": 5,
            }),
        )
        client.propose_point(
            canonical_id="recipe:foo:mi300x",
            kind=C.KIND_RECIPE,
            authority=C.AUTHORITY_EXPERIENTIAL,
            attrs={"model": "foo"},
            evidence=["log:hyperloom"],
        )
    body = json.loads(route.calls.last.request.content)
    required = {
        C.F_CANONICAL_ID, C.F_KIND, C.F_AUTHORITY,
        C.F_EVIDENCE_REFS, C.F_PROVENANCE,
    }
    allowed = required | {C.F_ATTRS, C.F_ENTITY_TYPE}
    assert required.issubset(body.keys())
    assert "evidence" not in body
    assert set(body.keys()).issubset(allowed), f"extra fields: {set(body)-allowed}"
    assert body[C.F_AUTHORITY] in {
        C.AUTHORITY_AUTHORITATIVE, C.AUTHORITY_EXPERIENTIAL,
        C.AUTHORITY_INFERRED, C.AUTHORITY_HYPOTHESIZED,
    }
    assert len(body[C.F_EVIDENCE_REFS]) >= 1
    prov = body[C.F_PROVENANCE]
    assert prov[C.F_PV_SOURCE] in {
        C.SOURCE_OFFLINE_INGEST, C.SOURCE_AGENT_OBSERVATION,
        C.SOURCE_CASCADE_DERIVED, C.SOURCE_CO_OCCURRENCE,
        C.SOURCE_ANALOGY_SEED,
    }
    assert prov[C.F_PV_GENERATOR]
    assert prov[C.F_PV_GENERATED_AT]


# ---------------------------------------------------------------------------
# §1.7 — POST /v1/points/query  (used by _resolve_point_id + find_recipe)
# ---------------------------------------------------------------------------
def test_query_body_uses_stable_fields(client):
    """``find_recipe`` issues a query keyed by canonical_id + kind."""
    with respx.mock(base_url=KB_URL) as router:
        route = router.post(C.PATH_QUERY_POINT).mock(
            return_value=httpx.Response(200, json={"points": []}),
        )
        client.find_recipe(workload="m", hw="mi300x")
    body = json.loads(route.calls.last.request.content)
    stable_subset = {
        C.F_CANONICAL_ID, C.F_KIND, C.F_NEIGHBOR_PREVIEW, C.F_LIMIT,
    }
    assert set(body.keys()).issubset(stable_subset)
    assert body[C.F_LIMIT] <= 10_000
    assert body[C.F_KIND] == C.KIND_RECIPE


# ---------------------------------------------------------------------------
# Cross-cutting: NDJSON envelope shape is unchanged
# ---------------------------------------------------------------------------
def test_ndjson_envelope_keeps_legacy_shape(client, session_dir):
    """Flusher / collectors / breakdown consumers depend on the
    envelope shape. Lock it down: ``{op, payload, created_at,
    idempotency_key, attempts}``.
    """
    from inference_optimizer.cortex_kb_client import _ndjson_envelope
    env = _ndjson_envelope(op="propose_point", payload={"x": 1}, idempotency_key="k")
    assert set(env.keys()) == {"op", "payload", "created_at", "idempotency_key", "attempts"}
    assert env["attempts"] == 0
    assert env["op"] == "propose_point"
