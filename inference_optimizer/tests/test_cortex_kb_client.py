"""v0.8 M1 — Cortex KB integration tests (KB_design §3.13 M1).

Covers the HTTP-based write surface:

* Session-path helpers create the ``runtime/cortex/`` skeleton.
* CortexKBClient canonical_id derivations + NDJSON envelope contract.
* HTTP failures degrade to NDJSON enqueue.
* T0 fail-fast vs ``--degraded-kb`` bypass.
* breakdown.collect_kb_provenance returns the documented stable shape.

The tests use ``respx`` to mock ``CORTEX_KB_URL`` so no real Cortex
service is contacted.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from inference_optimizer.cortex_kb_client import (
    CortexKBClient,
    recipe_canonical_id,
)
from inference_optimizer.paths import make_session_dir
from inference_optimizer.session_paths import (
    cortex_audit_jsonl,
    cortex_dir,
    cortex_pending_ndjson,
)


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
# session_paths / skeleton
# ===========================================================================
def test_make_session_dir_creates_runtime_cortex(session_dir):
    cortex_root = cortex_dir(session_dir)
    assert cortex_root.exists()
    assert cortex_root.is_dir()
    assert cortex_pending_ndjson(session_dir).parent == cortex_root


def test_canonical_id_derivations_are_idempotent():
    # PR-A10: slugs are all-lowercase and path-style inputs get basenamed,
    # so the canonical_id matches the KB corpus convention regardless of
    # CLI casing or whether the operator passed a full model path.
    a = recipe_canonical_id("meta-llama/Llama-3.1-8B-Instruct", "mi300x")
    b = recipe_canonical_id("meta-llama/Llama-3.1-8B-Instruct", "MI300x")
    assert a == b
    assert a == "recipe:llama-3.1-8b-instruct:mi300x"
    c = recipe_canonical_id("/wekafs/models/DeepSeek-R1-0528", "MI300X")
    assert c == "recipe:deepseek-r1-0528:mi300x"
    assert recipe_canonical_id("", "") == "recipe:unknown_model:unknown_hw"


# ===========================================================================
# CortexKBClient — sync success + NDJSON fallback
# ===========================================================================
def test_disabled_client_skips_all_writes(session_dir):
    client = CortexKBClient(session_dir=session_dir, enabled=False, kb_url=KB_URL)
    assert client.propose_point(canonical_id="recipe:foo:bar", kind="recipe")[
        "status"
    ] == "skip_disabled"
    assert client.update_recipe(model="m", hardware="h")["status"] == "skip_disabled"
    assert not cortex_pending_ndjson(session_dir).exists()


def test_propose_point_uses_evidence_refs_and_provenance(session_dir):
    """Schema §1.6: ``evidence`` renamed to ``evidence_refs``; provenance required."""
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    with respx.mock(base_url=KB_URL) as router:
        route = router.post("/v1/points/propose").mock(
            return_value=httpx.Response(200, json={
                "proposal_id": 7, "status": "auto_accepted", "point_id": 7,
            }),
        )
        res = client.propose_point(
            canonical_id="recipe:foo:bar",
            kind="recipe",
            authority="EXPERIENTIAL",
            evidence=["log:hyperloom-session-x"],
        )
    assert res["status"] == "auto_accepted"
    assert res["point_id"] == "7"
    body = json.loads(route.calls.last.request.content)
    assert "evidence" not in body
    assert isinstance(body["evidence_refs"], list)
    assert body["evidence_refs"][0]["kind"] == "log"
    assert "provenance" in body
    assert body["provenance"]["source"] in (
        "agent_observation", "offline_ingest", "cascade_derived",
        "co_occurrence", "analogy_seed",
    )
    assert body["authority"] == "EXPERIENTIAL"


def test_drain_pending_empty_queue_is_no_op(session_dir):
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    out = client.drain_pending(timeout_sec=1.0)
    assert out["drained"] == 0
    assert out["remaining"] == 0
    assert out["dead_letter"] == 0


def test_parse_kb_error_business_vs_validation(session_dir):
    from inference_optimizer.cortex_kb_client import parse_kb_error
    business = httpx.Response(400, json={
        "detail": {"error": {
            "code": "INVALID_INPUT",
            "message": "kind unknown",
            "details": {"kind": "junk"},
        }},
    })
    validation = httpx.Response(422, json={
        "detail": [
            {"type": "missing", "loc": ["body", "authority"], "msg": "Field required"},
        ],
    })
    unknown = httpx.Response(503, text="<html>nginx 503</html>")
    cat, code, msg, details = parse_kb_error(business)
    assert cat == "business"
    assert code == "INVALID_INPUT"
    assert details == {"kind": "junk"}
    cat, code, msg, _ = parse_kb_error(validation)
    assert cat == "validation"
    assert "authority" in msg
    cat, _, _, _ = parse_kb_error(unknown)
    assert cat == "unknown"


def test_smoke_env_tags_attrs_and_provenance(session_dir, monkeypatch):
    monkeypatch.setenv("CORTEX_KB_SMOKE", "1")
    client = CortexKBClient(session_dir=session_dir, kb_url=KB_URL)
    with respx.mock(base_url=KB_URL) as router:
        route = router.post("/v1/points/propose").mock(
            return_value=httpx.Response(200, json={
                "proposal_id": 1, "status": "auto_accepted", "point_id": 1,
            }),
        )
        client.propose_point(canonical_id="x", kind="recipe", attrs={"a": 1})
    body = json.loads(route.calls.last.request.content)
    assert body["attrs"].get("kbsg_smoke") is True
    assert body["provenance"]["generator"].startswith("hyperloom-smoke")


# ===========================================================================
# breakdown.kb_provenance
# ===========================================================================
def test_kb_provenance_emits_stable_shape(session_dir):
    cortex_pending_ndjson(session_dir).write_text("", encoding="utf-8")
    cortex_audit_jsonl(session_dir).write_text(
        json.dumps({"ts": "now", "op": "propose_point", "status": "ok"}) + "\n",
        encoding="utf-8",
    )

    from inference_optimizer.breakdown.collectors import collect_kb_provenance
    warnings: list[str] = []
    state = {
        "cortex_session_id": "42",
        "warm_start_ts": "2026-05-19T00:00:00+00:00",
        "warm_start_recipe": {"raw": "recipe_node row 1"},
        "warm_start_pitfalls": [{"raw": "trap line"}],
    }
    manifest = {"stack_fingerprint": {"rocm": "7.2.0", "sglang": "0.4.10"}}
    out = collect_kb_provenance(session_dir, state, manifest, warnings)
    assert out["cortex_session_id"] == "42"
    assert out["warm_start_recipe_seen"] is True
    assert out["warm_start_pitfall_count"] == 1
    assert out["stack_fingerprint"]["rocm"] == "7.2.0"
    assert out["queue"]["pending_lines"] == 0
    assert out["audit_status_counts"]["ok"] == 1


def test_kb_provenance_no_cortex_session_still_succeeds(session_dir):
    from inference_optimizer.breakdown.collectors import collect_kb_provenance
    out = collect_kb_provenance(
        session_dir, state={}, manifest={}, warnings=[],
    )
    assert out["cortex_session_id"] == ""
    assert out["pending_edges"] == []
    assert out["queue"]["pending_lines"] == 0


# ===========================================================================
# SharedState additions
# ===========================================================================
def test_shared_state_has_v08_m1_cortex_fields():
    from inference_optimizer.orchestrator.shared_state import SharedState
    s = SharedState()
    assert s.cortex_session_id == ""
    assert s.cortex_session_summary == {}
    assert s.warm_start_recipe == {}
    assert s.warm_start_pitfalls == []
    assert s.warm_start_ts == ""


def test_policy_gate_core_state_fields_includes_cortex():
    from inference_optimizer.orchestrator.policy import CORE_STATE_FIELDS
    assert "cortex_session_id" in CORE_STATE_FIELDS
    assert "cortex_session_summary" in CORE_STATE_FIELDS
    assert "warm_start_recipe" in CORE_STATE_FIELDS


# ===========================================================================
# manifest stack_fingerprint
# ===========================================================================
def test_manifest_includes_stack_fingerprint(session_dir, monkeypatch):
    monkeypatch.setenv("ROCM_VERSION", "7.2.0-test")
    monkeypatch.setenv("SGLANG_VERSION", "0.4.10-test")
    monkeypatch.setenv("AITER_COMMIT", "721f045")
    monkeypatch.delenv("VLLM_VERSION", raising=False)
    from inference_optimizer.manifest import build_manifest
    manifest = build_manifest(session_dir, args=None, session_id="abc")
    assert manifest["stack_fingerprint"]["rocm"] == "7.2.0-test"
    assert manifest["stack_fingerprint"]["sglang"] == "0.4.10-test"
    assert manifest["stack_fingerprint"]["aiter"] == "721f045"
    assert manifest["stack_fingerprint"]["vllm"] in ("unknown",) or \
           manifest["stack_fingerprint"]["vllm"]
