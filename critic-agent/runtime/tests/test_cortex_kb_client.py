# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for :class:`runtime.cortex_kb_client.CortexKBClient`.

The transport is exercised by stubbing ``urllib.request.urlopen`` with a
queue of canned responses, capturing each outgoing request so the
scoped-article -> cortex ``/v1`` mapping can be asserted.
"""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any

import pytest

from runtime.cortex_kb_client import CortexKBClient
from runtime.errors import (
    KBNotFoundError,
    KBTransportError,
    KBValidationError,
)


class _FakeResp:
    def __init__(self, body: dict[str, Any], status: int = 200):
        self._raw = json.dumps(body).encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class _Transport:
    """Records requests and replays a scripted sequence of responses."""

    def __init__(self, responses: list[Any]):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, req: Any, timeout: float | None = None) -> Any:
        self.calls.append(
            {
                "url": req.full_url,
                "body": json.loads(req.data.decode("utf-8")),
            }
        )
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _http_error(code: int, body: str = "boom") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://kb/v1",
        code=code,
        msg="err",
        hdrs=None,
        fp=io.BytesIO(body.encode("utf-8")),
    )


@pytest.fixture()
def make_client(monkeypatch):
    def _make(responses: list[Any]) -> tuple[CortexKBClient, _Transport]:
        transport = _Transport(responses)
        monkeypatch.setattr(
            "runtime.cortex_kb_client.urllib.request.urlopen", transport
        )
        client = CortexKBClient(
            base_url="http://kb-service.test/",
            retry_max=2,
            backoff_base=0.0,
            sleep_fn=lambda _s: None,
        )
        return client, transport

    return _make


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------
def test_list_builds_scope_attrs_filter_and_namespaced_kind(make_client):
    client, transport = make_client([_FakeResp({"points": []})])
    client.list(scope_filter={"model": " Qwen3 ", "framework": "SGLang"}, kind="pitfall")

    body = transport.calls[0]["body"]
    assert transport.calls[0]["url"] == "http://kb-service.test/v1/points/query"
    assert body["kind"] == "critic_pitfall"
    assert body["neighbor_preview"] is False
    # scope values are normalised (trim + lowercase) under the reserved key.
    assert body["attrs_filter"]["_critic_scope"] == {
        "model": "qwen3",
        "framework": "sglang",
    }


def test_list_maps_points_to_entries_sorted_desc_and_limited(make_client):
    points = [
        {
            "id": 1,
            "canonical_id": "critic.pitfall.old.abc",
            "kind": "critic_pitfall",
            "is_active": True,
            "created_at": "t1",
            "attrs": {
                "_critic_scope": {"model": "qwen3"},
                "_critic_kind": "pitfall",
                "_critic_slug": "old",
                "_critic_importance": 0.4,
                "_critic_summary": "older",
                "_critic_metadata": {"topic": "a"},
                "_critic_updated_at": 100.0,
            },
        },
        {
            "id": 2,
            "canonical_id": "critic.pitfall.new.def",
            "kind": "critic_pitfall",
            "is_active": True,
            "created_at": "t2",
            "attrs": {
                "_critic_scope": {"model": "qwen3"},
                "_critic_kind": "pitfall",
                "_critic_slug": "new",
                "_critic_importance": 0.9,
                "_critic_summary": "newer",
                "_critic_metadata": {"topic": "b"},
                "_critic_updated_at": 200.0,
            },
        },
    ]
    client, _ = make_client([_FakeResp({"points": points})])
    out = client.list(scope_filter={"model": "qwen3"}, limit=1)

    assert out["count"] == 1
    entry = out["entries"][0]
    # newest (updated_at=200) first, limited to 1
    assert entry["slug"] == "new"
    assert entry["kind"] == "pitfall"
    assert entry["importance"] == 0.9
    assert entry["summary"] == "newer"
    assert entry["scope"] == {"model": "qwen3"}
    assert entry["deleted"] is False


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------
def test_upsert_builds_propose_envelope(make_client):
    client, transport = make_client(
        [_FakeResp({"proposal_id": 7, "status": "auto_accepted", "point_id": 7})]
    )
    res = client.upsert(
        {
            "scope": {"model": "Qwen3", "framework": "sglang"},
            "kind": "technique",
            "slug": "use-fp8",
            "importance": 0.7,
            "summary": "prefer fp8",
            "metadata": {"source_session": "sess-1"},
        }
    )

    body = transport.calls[0]["body"]
    assert transport.calls[0]["url"] == "http://kb-service.test/v1/points/propose"
    assert body["kind"] == "critic_technique"
    assert body["authority"] == "EXPERIENTIAL"
    assert body["canonical_id"].startswith("critic.technique.use-fp8.")
    assert body["evidence_refs"][0]["kind"] == "log"
    assert body["evidence_refs"][0]["ref"] == "critic-session:sess-1"
    assert body["provenance"]["source"] == "agent_observation"
    assert body["provenance"]["generator"] == "critic-agent"
    # reserved attrs round-trip
    assert body["attrs"]["_critic_scope"] == {"model": "qwen3", "framework": "sglang"}
    assert body["attrs"]["_critic_importance"] == 0.7
    assert res["created"] is True
    assert res["row"]["slug"] == "use-fp8"


def test_upsert_canonical_id_is_deterministic(make_client):
    client, transport = make_client(
        [
            _FakeResp({"status": "auto_accepted", "point_id": 1}),
            _FakeResp({"status": "auto_accepted", "point_id": 1}),
        ]
    )
    payload = {
        "scope": {"model": "qwen3"},
        "kind": "pitfall",
        "slug": "oom",
        "importance": 0.5,
    }
    client.upsert(dict(payload))
    client.upsert(dict(payload))
    assert (
        transport.calls[0]["body"]["canonical_id"]
        == transport.calls[1]["body"]["canonical_id"]
    )


def test_upsert_missing_field_raises_validation(make_client):
    client, _ = make_client([])
    with pytest.raises(KBValidationError):
        client.upsert({"scope": {}, "kind": "pitfall", "slug": "x"})  # no importance


# ---------------------------------------------------------------------------
# batch_insert
# ---------------------------------------------------------------------------
def test_batch_insert_builds_bulk_request(make_client):
    client, transport = make_client(
        [_FakeResp({"accepted": {"points": [11, 12]}, "rejected": {"points": []}})]
    )
    items = [
        {"scope": {"model": "qwen3"}, "kind": "pitfall", "slug": "a", "importance": 0.5},
        {"scope": {"model": "qwen3"}, "kind": "technique", "slug": "b", "importance": 0.6},
    ]
    res = client.batch_insert(items, on_conflict="upsert")

    body = transport.calls[0]["body"]
    assert transport.calls[0]["url"] == "http://kb-service.test/v1/bulk/ingest"
    assert body["pipeline_id"] == "critic-kb"
    assert "batch_id" in body
    assert [p["kind"] for p in body["points"]] == ["critic_pitfall", "critic_technique"]
    assert res["count"] == 2


def test_batch_insert_rejected_points_raise(make_client):
    client, _ = make_client(
        [
            _FakeResp(
                {
                    "accepted": {"points": []},
                    "rejected": {
                        "points": [
                            {"request_index": 0, "code": "INVALID_INPUT", "message": "bad"}
                        ]
                    },
                }
            )
        ]
    )
    with pytest.raises(KBValidationError):
        client.batch_insert(
            [{"scope": {}, "kind": "pitfall", "slug": "a", "importance": 0.1}]
        )


def test_batch_insert_bad_on_conflict_raises(make_client):
    client, _ = make_client([])
    with pytest.raises(KBValidationError):
        client.batch_insert([], on_conflict="merge")


# ---------------------------------------------------------------------------
# add_edges
# ---------------------------------------------------------------------------
def test_add_edges_maps_contradicts_to_negate(make_client):
    client, transport = make_client([_FakeResp({"edge": {"id": 99}})])
    out = client.add_edges([{"kind": "contradicts", "from_id": 5, "to_id": 6}])

    body = transport.calls[0]["body"]
    assert transport.calls[0]["url"] == "http://kb-service.test/v1/edges/negate"
    assert body["from_point"] == 5
    assert body["to_point"] == 6
    assert body["authority"] == "EXPERIENTIAL"
    assert out["added"][0]["kind"] == "negation"


def test_add_edges_rejects_non_contradicts(make_client):
    client, _ = make_client([])
    with pytest.raises(KBValidationError):
        client.add_edges([{"kind": "supports", "from_id": 1, "to_id": 2}])


def test_add_edges_rejects_non_int_ids(make_client):
    client, _ = make_client([])
    with pytest.raises(KBValidationError):
        client.add_edges([{"kind": "contradicts", "from_id": "kb_x", "to_id": "kb_y"}])


# ---------------------------------------------------------------------------
# transport / error mapping
# ---------------------------------------------------------------------------
def test_404_maps_to_not_found(make_client):
    client, _ = make_client([_http_error(404)])
    with pytest.raises(KBNotFoundError):
        client.list(scope_filter={"model": "qwen3"})


def test_400_maps_to_validation(make_client):
    client, _ = make_client([_http_error(422)])
    with pytest.raises(KBValidationError):
        client.list(scope_filter={"model": "qwen3"})


def test_5xx_retries_then_transport_error(make_client):
    # retry_max=2 -> 3 attempts total, all 503
    client, transport = make_client([_http_error(503), _http_error(503), _http_error(503)])
    with pytest.raises(KBTransportError):
        client.list(scope_filter={"model": "qwen3"})
    assert len(transport.calls) == 3


def test_5xx_then_success_recovers(make_client):
    client, transport = make_client([_http_error(503), _FakeResp({"points": []})])
    out = client.list(scope_filter={"model": "qwen3"})
    assert out["count"] == 0
    assert len(transport.calls) == 2


# ---------------------------------------------------------------------------
# CLI resolver wiring
# ---------------------------------------------------------------------------
def test_cli_resolves_cortex_client(monkeypatch):
    from runtime.cli import _resolve_kb_client

    monkeypatch.setenv("CRITIC_KB_CLIENT_MODE", "cortex")
    monkeypatch.setenv("KB_BASE_URL", "http://kb-service.test")
    client = _resolve_kb_client()
    assert isinstance(client, CortexKBClient)


def test_cli_cortex_mode_requires_base_url(monkeypatch):
    from runtime.cli import _resolve_kb_client
    from runtime.errors import RuntimeAdapterError

    monkeypatch.setenv("CRITIC_KB_CLIENT_MODE", "cortex")
    monkeypatch.delenv("KB_BASE_URL", raising=False)
    with pytest.raises(RuntimeAdapterError):
        _resolve_kb_client()
