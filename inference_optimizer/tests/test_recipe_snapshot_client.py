"""recipe-snapshot v2 HTTP client tests.

Covers Phase 1 of the cortex-KB → recipe-snapshot cutover:

* ``put_recipe`` 200 happy path: wire body matches the v2 spec,
  ``provenance`` is auto-stamped, evidence refs normalised, response
  carries ``{status, canonical_id, version, created}``.
* ``put_recipe`` 5xx → NDJSON enqueue (foreground client falls
  through within the 2.5 s side-channel budget).
* ``put_recipe`` 422 → :class:`RecipeSnapshotError` with
  ``category="validation"`` AND NDJSON enqueue (the row is still
  parked so an offline schema fix can replay it).
* ``get_recipe`` 200 / 404 / ``?version=`` paths.
* ``get_history`` empty-array contract on unknown id.
* ``health()`` smoke probe.
* ``drain_pending`` replays NDJSON, separates flushed / retained /
  dead-letter buckets.
* ``--degraded-kb`` disabled client short-circuits every entrypoint.
* Foreground vs background timeout / retry profile resolution.
* Env-var override precedence.

The tests use ``respx`` to mock the kb-service base URL so no real
network call is made.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from inference_optimizer.recipe_snapshot_client import (
    RecipeSnapshotClient,
    RecipeSnapshotError,
    parse_error_envelope,
)
from inference_optimizer.recipe_snapshot_constants import (
    AUTHORITY_EXPERIENTIAL,
    DEFAULT_HTTP_TIMEOUT_SEC,
    DEFAULT_KB_URL,
    DEFAULT_RETRY_ATTEMPTS,
    FOREGROUND_HTTP_TIMEOUT_SEC,
    FOREGROUND_RETRY_ATTEMPTS,
    OP_PUT_RECIPE,
    PATH_HEALTH,
    PATH_RECIPE_HISTORY_TPL,
    PATH_RECIPE_TPL,
    format_recipe_path,
    recipe_canonical_id,
)
from inference_optimizer.session_paths import (
    recipe_snapshot_audit_jsonl,
    recipe_snapshot_dead_letter_ndjson,
    recipe_snapshot_dir,
    recipe_snapshot_flushed_ndjson,
    recipe_snapshot_pending_ndjson,
)


KB_URL = "http://kb-test.local"


def _cid(
    *,
    model: str = "m",
    hardware: str = "mi300x",
    framework: str = "sglang",
    framework_version: str = "0.4.5",
    precision: str = "fp8",
) -> str:
    """Test-only convenience wrapper around :func:`recipe_canonical_id`.

    Centralises the 5-tuple call so the bulk of the suite can override
    just the dimension(s) it cares about (most often ``model`` to
    distinguish two recipes in the same test). Detailed canonical_id
    semantics — slug normalisation, default fall-through, mhfvp
    ordering — are tested in ``test_canonical_id_5tuple.py``.
    """
    return recipe_canonical_id(
        model=model,
        hardware=hardware,
        framework=framework,
        framework_version=framework_version,
        precision=precision,
    )


# ===========================================================================
# fixtures
# ===========================================================================
@pytest.fixture
def session_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Use ``tmp_path`` directly as the session_dir; we don't need the
    # full make_session_dir skeleton for the client unit tests.
    monkeypatch.setenv("CORTEX_KB_URL", KB_URL)
    monkeypatch.delenv("KB_SERVICE_TOKEN", raising=False)
    monkeypatch.delenv("CORTEX_KB_SMOKE", raising=False)
    monkeypatch.delenv("CORTEX_KB_HTTP_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("CORTEX_KB_RETRY_ATTEMPTS", raising=False)
    monkeypatch.delenv("CORTEX_KB_MAX_CONCURRENCY", raising=False)
    return tmp_path


@pytest.fixture
def client(session_dir: Path) -> RecipeSnapshotClient:
    # Foreground=False so retries don't slow the test down on the
    # error-path cases (background uses ``retry_attempts=3``; we
    # override to 1 below where the test needs deterministic timing).
    return RecipeSnapshotClient(
        session_dir=session_dir,
        kb_url=KB_URL,
        foreground=False,
        retry_attempts=1,
    )


# ===========================================================================
# path helpers
# ===========================================================================
# Detailed canonical_id semantics (slug rules, default fall-through,
# mhfvp ordering, auto-detect helper) live in
# ``test_canonical_id_5tuple.py``. This module covers behavioural
# tests of the HTTP client only; the cid is just an opaque identifier
# from this file's point of view.
def test_format_recipe_path_preserves_colons():
    cid = "inference:qwen3-30b-a3b:sglang:mi355x"
    path = format_recipe_path(PATH_RECIPE_TPL, cid)
    assert path == "/recipe-snapshot/recipes/" + cid


def test_format_recipe_path_keeps_slashes_in_hf_stem():
    """HF stems carry ``/`` (``Qwen/Qwen3-30B-A3B``). The server
    accepts them as ``:path``-typed; the helper must NOT
    percent-encode."""
    cid = "Qwen/Qwen3-30B-A3B:sglang:mi355x"
    path = format_recipe_path(PATH_RECIPE_TPL, cid)
    assert "/" in path.split("/recipes/", 1)[1]


# ===========================================================================
# parse_error_envelope
# ===========================================================================
def test_parse_error_envelope_business_shape():
    resp = httpx.Response(
        404,
        json={
            "detail": {
                "error": {
                    "code": "NOT_FOUND",
                    "message": "recipe canonical_id 'x' not found",
                    "details": {"canonical_id": "x"},
                },
            },
        },
    )
    category, code, message, details = parse_error_envelope(resp)
    assert category == "business"
    assert code == "NOT_FOUND"
    assert "not found" in message
    assert details == {"canonical_id": "x"}


def test_parse_error_envelope_validation_shape():
    resp = httpx.Response(
        422,
        json={
            "detail": [
                {"loc": ["body", "authority"], "msg": "Field required",
                 "type": "missing"},
            ],
        },
    )
    category, code, message, _details = parse_error_envelope(resp)
    assert category == "validation"
    assert code == "VALIDATION_ERROR"
    assert "body.authority" in message


def test_parse_error_envelope_unknown_shape():
    resp = httpx.Response(502, text="<html>upstream timeout</html>")
    category, _code, message, _details = parse_error_envelope(resp)
    assert category == "unknown"
    assert "upstream" in message


# ===========================================================================
# put_recipe — happy path
# ===========================================================================
@respx.mock
def test_put_recipe_happy_path_writes_v2_body_shape(client: RecipeSnapshotClient):
    cid = _cid(model="test-model")
    route = respx.put(f"{KB_URL}{format_recipe_path(PATH_RECIPE_TPL, cid)}").mock(
        return_value=httpx.Response(
            200, json={"canonical_id": cid, "version": 1, "created": True},
        ),
    )
    result = client.put_recipe(
        canonical_id=cid,
        labels={"model": "test-model", "hardware": "mi300x", "framework": "sglang"},
        body={"best_config": {"tp": "1"}, "stack_fingerprint": {"rocm": "7.2"}},
        metrics={"throughput": 5178.78, "e2el_ms": 12646.7},
        lessons=[{"statement": "fp8 helps decode"}],
        evidence=["log:hyperloom-session-uuid"],
    )
    assert result["status"] == "ok"
    assert result["canonical_id"] == cid
    assert result["version"] == 1
    assert result["created"] is True

    # Inspect the wire body the client actually sent.
    request = route.calls[-1].request
    sent = json.loads(request.content)
    # v2 strict-required.
    assert sent["authority"] == AUTHORITY_EXPERIENTIAL
    assert "provenance" in sent
    assert sent["provenance"]["source"] and sent["provenance"]["generator"]
    assert sent["provenance"]["generated_at"]
    # Caller-defined opaque blobs, passed through verbatim.
    assert sent["labels"] == {
        "model": "test-model", "hardware": "mi300x", "framework": "sglang",
    }
    assert sent["body"] == {
        "best_config": {"tp": "1"}, "stack_fingerprint": {"rocm": "7.2"},
    }
    assert sent["metrics"] == {"throughput": 5178.78, "e2el_ms": 12646.7}
    assert sent["lessons"] == [{"statement": "fp8 helps decode"}]
    # Evidence refs normalised: ``"log:..."`` string → dict shape.
    assert sent["evidence_refs"] == [
        {"kind": "log", "ref": "hyperloom-session-uuid"},
    ]
    # Empty arrays / dicts NOT emitted (terse audit log invariant).
    assert "findings" not in sent
    assert "pitfalls" not in sent


@respx.mock
def test_put_recipe_omits_optional_collections_when_empty(client: RecipeSnapshotClient):
    """Sending ``[]`` is harmless on v2 but cluttery in the audit
    log; the client omits empty list / dict fields entirely."""
    cid = _cid()
    respx.put(f"{KB_URL}{format_recipe_path(PATH_RECIPE_TPL, cid)}").mock(
        return_value=httpx.Response(
            200, json={"canonical_id": cid, "version": 2, "created": False},
        ),
    )
    client.put_recipe(canonical_id=cid)
    sent = json.loads(respx.calls[-1].request.content)
    for absent in (
        "labels", "body", "metrics",
        "findings", "failures", "pitfalls", "lessons", "gaps",
        "evidence_refs",
    ):
        assert absent not in sent, f"empty {absent!r} should not be on the wire"


@respx.mock
def test_put_recipe_smoke_mode_stamps_smoke_generator(session_dir: Path):
    cid = _cid()
    respx.put(f"{KB_URL}{format_recipe_path(PATH_RECIPE_TPL, cid)}").mock(
        return_value=httpx.Response(
            200, json={"canonical_id": cid, "version": 1, "created": True},
        ),
    )
    smoke_client = RecipeSnapshotClient(
        session_dir=session_dir, kb_url=KB_URL, smoke=True,
        foreground=False, retry_attempts=1,
    )
    smoke_client.put_recipe(canonical_id=cid, labels={"x": 1})
    sent = json.loads(respx.calls[-1].request.content)
    assert sent["provenance"]["generator"].startswith("hyperloom-smoke")


# ===========================================================================
# put_recipe — failure → NDJSON enqueue
# ===========================================================================
@respx.mock
def test_put_recipe_transient_failure_enqueues_to_ndjson(
    client: RecipeSnapshotClient, session_dir: Path,
):
    cid = _cid()
    respx.put(f"{KB_URL}{format_recipe_path(PATH_RECIPE_TPL, cid)}").mock(
        return_value=httpx.Response(503, text="pool not ready"),
    )
    result = client.put_recipe(
        canonical_id=cid,
        labels={"model": "m"},
        body={"best_config": {}},
    )
    assert result["status"] == "queued"
    assert result["canonical_id"] == cid

    # NDJSON queue gained exactly one row, with the full PUT body
    # preserved so the flusher can replay verbatim.
    pending = recipe_snapshot_pending_ndjson(session_dir)
    assert pending.exists()
    rows = [json.loads(l) for l in pending.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["op"] == OP_PUT_RECIPE
    assert row["payload"]["canonical_id"] == cid
    wire = row["payload"]["wire_body"]
    assert wire["authority"] == AUTHORITY_EXPERIENTIAL
    assert wire["labels"] == {"model": "m"}
    assert wire["body"] == {"best_config": {}}


@respx.mock
def test_put_recipe_validation_failure_raises_and_does_not_enqueue(
    client: RecipeSnapshotClient, session_dir: Path,
):
    """422 validation rejects re-issuing the same body would hit the
    same wall — but Phase 1 still enqueues so an offline schema fix
    can re-emit. This mirrors the legacy cortex client's behaviour;
    flusher dead-letters business / validation errors on the first
    retry attempt so the queue won't spin forever."""
    cid = _cid()
    respx.put(f"{KB_URL}{format_recipe_path(PATH_RECIPE_TPL, cid)}").mock(
        return_value=httpx.Response(
            422,
            json={
                "detail": [
                    {"loc": ["body", "authority"], "msg": "Field required",
                     "type": "missing"},
                ],
            },
        ),
    )
    result = client.put_recipe(canonical_id=cid, labels={"m": 1})
    assert result["status"] == "queued"
    rows = [
        json.loads(l)
        for l in recipe_snapshot_pending_ndjson(session_dir).read_text().splitlines()
        if l.strip()
    ]
    assert len(rows) == 1


@respx.mock
def test_put_recipe_propagates_when_enqueue_disabled(
    client: RecipeSnapshotClient,
):
    """``drain_pending`` re-issues queued rows with
    ``_enqueue_on_failure=False`` — a persistent failure MUST raise
    so the drain loop can dead-letter the row instead of duplicating
    it on the queue forever."""
    cid = _cid()
    respx.put(f"{KB_URL}{format_recipe_path(PATH_RECIPE_TPL, cid)}").mock(
        return_value=httpx.Response(503, text="x"),
    )
    with pytest.raises(RecipeSnapshotError) as exc_info:
        client.put_recipe(
            canonical_id=cid, _enqueue_on_failure=False,
        )
    assert exc_info.value.category == "transport"


# ===========================================================================
# get_recipe / get_history
# ===========================================================================
@respx.mock
def test_get_recipe_200(client: RecipeSnapshotClient):
    cid = _cid()
    recipe = {
        "canonical_id": cid, "version": 7,
        "labels": {"model": "m"}, "body": {"best_config": {}},
        "metrics": {"throughput": 100.0},
        "lessons": [], "pitfalls": [], "findings": [],
        "failures": [], "gaps": [],
        "authority": "EXPERIENTIAL", "confidence": 0.85,
        "evidence_refs": [], "provenance": {
            "source": "x", "generator": "y", "generated_at": "z",
        },
        "created_at": "2026-05-28T00:00:00+00:00",
        "updated_at": "2026-05-28T00:00:00+00:00",
    }
    respx.get(f"{KB_URL}{format_recipe_path(PATH_RECIPE_TPL, cid)}").mock(
        return_value=httpx.Response(200, json=recipe),
    )
    got = client.get_recipe(canonical_id=cid)
    assert got is not None
    assert got["canonical_id"] == cid
    assert got["version"] == 7
    assert got["metrics"] == {"throughput": 100.0}


@respx.mock
def test_get_recipe_404_returns_none(client: RecipeSnapshotClient):
    cid = _cid(model="absent")
    respx.get(f"{KB_URL}{format_recipe_path(PATH_RECIPE_TPL, cid)}").mock(
        return_value=httpx.Response(
            404,
            json={
                "detail": {
                    "error": {
                        "code": "NOT_FOUND",
                        "message": f"recipe canonical_id '{cid}' not found",
                        "details": {"canonical_id": cid},
                    },
                },
            },
        ),
    )
    assert client.get_recipe(canonical_id=cid) is None


@respx.mock
def test_get_recipe_with_explicit_version_forwards_query_param(
    client: RecipeSnapshotClient,
):
    cid = _cid()
    route = respx.get(f"{KB_URL}{format_recipe_path(PATH_RECIPE_TPL, cid)}").mock(
        return_value=httpx.Response(
            200, json={"canonical_id": cid, "version": 3},
        ),
    )
    client.get_recipe(canonical_id=cid, version=3)
    sent = route.calls[-1].request
    assert sent.url.params["version"] == "3"


@respx.mock
def test_get_recipe_500_raises(client: RecipeSnapshotClient):
    cid = _cid()
    respx.get(f"{KB_URL}{format_recipe_path(PATH_RECIPE_TPL, cid)}").mock(
        return_value=httpx.Response(503, text="pool"),
    )
    with pytest.raises(RecipeSnapshotError) as exc_info:
        client.get_recipe(canonical_id=cid)
    assert exc_info.value.category == "transport"


@respx.mock
def test_get_history_returns_archives(client: RecipeSnapshotClient):
    cid = _cid()
    respx.get(
        f"{KB_URL}{format_recipe_path(PATH_RECIPE_HISTORY_TPL, cid)}",
    ).mock(
        return_value=httpx.Response(200, json={
            "canonical_id": cid,
            "history": [
                {"version": 1, "archived_at": "2026-05-27T00:00:00+00:00",
                 "replaced_by": {"source": "x", "generator": "y", "generated_at": "z"},
                 "snapshot": {"canonical_id": cid, "version": 1}},
                {"version": 2, "archived_at": "2026-05-27T01:00:00+00:00",
                 "replaced_by": {"source": "x", "generator": "y", "generated_at": "z"},
                 "snapshot": {"canonical_id": cid, "version": 2}},
            ],
        }),
    )
    history = client.get_history(canonical_id=cid)
    assert len(history) == 2
    assert history[0]["version"] == 1
    assert history[1]["version"] == 2


@respx.mock
def test_get_history_empty_for_unknown_id(client: RecipeSnapshotClient):
    """Spec: history endpoint returns ``{canonical_id, history: []}``
    for unknown id (NOT 404). The client must surface that as an
    empty list, not raise."""
    cid = _cid(model="absent")
    respx.get(
        f"{KB_URL}{format_recipe_path(PATH_RECIPE_HISTORY_TPL, cid)}",
    ).mock(
        return_value=httpx.Response(
            200, json={"canonical_id": cid, "history": []},
        ),
    )
    assert client.get_history(canonical_id=cid) == []


# ===========================================================================
# health()
# ===========================================================================
@respx.mock
def test_health_ok(client: RecipeSnapshotClient):
    respx.get(f"{KB_URL}{PATH_HEALTH}").mock(
        return_value=httpx.Response(200, json={"status": "ok"}),
    )
    assert client.health() is True


@respx.mock
def test_health_bad_body_treated_as_unhealthy(client: RecipeSnapshotClient):
    respx.get(f"{KB_URL}{PATH_HEALTH}").mock(
        return_value=httpx.Response(200, json={"status": "degraded"}),
    )
    assert client.health() is False


@respx.mock
def test_health_500_treated_as_unhealthy(client: RecipeSnapshotClient):
    respx.get(f"{KB_URL}{PATH_HEALTH}").mock(
        return_value=httpx.Response(503, text="pool"),
    )
    assert client.health() is False


def test_health_disabled_short_circuits(session_dir: Path):
    c = RecipeSnapshotClient(
        session_dir=session_dir, kb_url=KB_URL, enabled=False,
    )
    assert c.health() is False


# ===========================================================================
# drain_pending
# ===========================================================================
@respx.mock
def test_drain_pending_flushes_recoverable_rows(
    client: RecipeSnapshotClient, session_dir: Path,
):
    # Seed two queued rows.
    cid_a = _cid(model="a")
    cid_b = _cid(model="b")
    respx.put(f"{KB_URL}{format_recipe_path(PATH_RECIPE_TPL, cid_a)}").mock(
        return_value=httpx.Response(503, text="seed"),
    )
    respx.put(f"{KB_URL}{format_recipe_path(PATH_RECIPE_TPL, cid_b)}").mock(
        return_value=httpx.Response(503, text="seed"),
    )
    client.put_recipe(canonical_id=cid_a, labels={"x": 1})
    client.put_recipe(canonical_id=cid_b, labels={"x": 2})
    assert len(
        [l for l in recipe_snapshot_pending_ndjson(session_dir).read_text().splitlines()
         if l.strip()]
    ) == 2

    # Now the server recovered for A but is still down for B.
    respx.put(f"{KB_URL}{format_recipe_path(PATH_RECIPE_TPL, cid_a)}").mock(
        return_value=httpx.Response(
            200, json={"canonical_id": cid_a, "version": 1, "created": True},
        ),
    )
    # cid_b still 503 (transient → kept in queue with attempts+1).

    summary = client.drain_pending()
    assert summary["pending"] == 2
    assert summary["flushed"] == 1
    assert summary["dead_lettered"] == 0
    assert summary["remaining"] == 1

    # The retained row carries attempts=1.
    retained_rows = [
        json.loads(l)
        for l in recipe_snapshot_pending_ndjson(session_dir).read_text().splitlines()
        if l.strip()
    ]
    assert len(retained_rows) == 1
    assert retained_rows[0]["payload"]["canonical_id"] == cid_b
    assert retained_rows[0]["attempts"] == 1

    # Flushed row landed in the ``.flushed.ndjson`` audit file.
    flushed_rows = [
        json.loads(l)
        for l in recipe_snapshot_flushed_ndjson(session_dir).read_text().splitlines()
        if l.strip()
    ]
    assert len(flushed_rows) == 1
    assert flushed_rows[0]["payload"]["canonical_id"] == cid_a


@respx.mock
def test_drain_pending_dead_letters_on_business_error(
    client: RecipeSnapshotClient, session_dir: Path,
):
    """422 / business errors short-circuit to dead-letter on the
    first drain attempt — retrying with the same body cannot help."""
    cid = _cid()
    respx.put(f"{KB_URL}{format_recipe_path(PATH_RECIPE_TPL, cid)}").mock(
        return_value=httpx.Response(503, text="seed"),
    )
    client.put_recipe(canonical_id=cid, labels={"x": 1})
    respx.put(f"{KB_URL}{format_recipe_path(PATH_RECIPE_TPL, cid)}").mock(
        return_value=httpx.Response(422, json={
            "detail": [
                {"loc": ["body", "authority"], "msg": "Field required",
                 "type": "missing"},
            ],
        }),
    )
    summary = client.drain_pending()
    assert summary["flushed"] == 0
    assert summary["dead_lettered"] == 1
    assert summary["remaining"] == 0
    dead_rows = [
        json.loads(l)
        for l in recipe_snapshot_dead_letter_ndjson(session_dir).read_text().splitlines()
        if l.strip()
    ]
    assert len(dead_rows) == 1
    assert "validation" in dead_rows[0]["last_error"]


def test_drain_pending_empty_queue_returns_zeros(client: RecipeSnapshotClient):
    summary = client.drain_pending()
    assert summary == {
        "pending": 0, "flushed": 0, "dead_lettered": 0, "remaining": 0,
    }


# ===========================================================================
# disabled client short-circuits
# ===========================================================================
def test_disabled_client_short_circuits_all_writes(
    session_dir: Path,
):
    c = RecipeSnapshotClient(
        session_dir=session_dir, kb_url=KB_URL, enabled=False,
    )
    cid = _cid()
    assert c.put_recipe(canonical_id=cid, labels={"x": 1}) == {
        "status": "skip_disabled", "canonical_id": cid,
    }
    assert c.get_recipe(canonical_id=cid) is None
    assert c.get_history(canonical_id=cid) == []
    # No NDJSON / audit on a disabled client — nothing should be written.
    assert not recipe_snapshot_pending_ndjson(session_dir).exists()


# ===========================================================================
# timeout / retry profile resolution
# ===========================================================================
def test_foreground_profile_defaults(session_dir: Path):
    c = RecipeSnapshotClient(
        session_dir=session_dir, kb_url=KB_URL, foreground=True,
    )
    assert c.timeout_sec == FOREGROUND_HTTP_TIMEOUT_SEC
    assert c.retry_attempts == FOREGROUND_RETRY_ATTEMPTS


def test_background_profile_defaults(session_dir: Path):
    c = RecipeSnapshotClient(
        session_dir=session_dir, kb_url=KB_URL, foreground=False,
    )
    assert c.timeout_sec == DEFAULT_HTTP_TIMEOUT_SEC
    assert c.retry_attempts == DEFAULT_RETRY_ATTEMPTS


def test_env_overrides_take_precedence(
    session_dir: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("CORTEX_KB_HTTP_TIMEOUT_SEC", "7.5")
    monkeypatch.setenv("CORTEX_KB_RETRY_ATTEMPTS", "5")
    monkeypatch.setenv("CORTEX_KB_MAX_CONCURRENCY", "16")
    monkeypatch.setenv("KB_SERVICE_TOKEN", "test-token-123")
    c = RecipeSnapshotClient(
        session_dir=session_dir, kb_url=KB_URL, foreground=True,
    )
    assert c.timeout_sec == 7.5
    assert c.retry_attempts == 5
    assert c.max_connections == 16
    assert c.token == "test-token-123"


def test_explicit_kb_url_overrides_env(
    session_dir: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("CORTEX_KB_URL", "http://env-url.local")
    c = RecipeSnapshotClient(
        session_dir=session_dir, kb_url="http://explicit.local",
    )
    assert c.kb_url == "http://explicit.local"


def test_kb_url_falls_back_to_env(
    session_dir: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("CORTEX_KB_URL", "http://env-url.local")
    c = RecipeSnapshotClient(session_dir=session_dir)
    assert c.kb_url == "http://env-url.local"


def test_kb_url_default_when_no_env(
    session_dir: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("CORTEX_KB_URL", raising=False)
    c = RecipeSnapshotClient(session_dir=session_dir)
    assert c.kb_url == DEFAULT_KB_URL


# ===========================================================================
# audit log
# ===========================================================================
@respx.mock
def test_put_recipe_writes_audit_row(
    client: RecipeSnapshotClient, session_dir: Path,
):
    cid = _cid()
    respx.put(f"{KB_URL}{format_recipe_path(PATH_RECIPE_TPL, cid)}").mock(
        return_value=httpx.Response(
            200, json={"canonical_id": cid, "version": 1, "created": True},
        ),
    )
    client.put_recipe(canonical_id=cid, labels={"x": 1})
    audit_path = recipe_snapshot_audit_jsonl(session_dir)
    rows = [json.loads(l) for l in audit_path.read_text().splitlines() if l.strip()]
    # One HTTP row + one put_recipe outcome row.
    ops = [r["op"] for r in rows]
    assert "http" in ops
    assert "put_recipe" in ops
    put_row = next(r for r in rows if r["op"] == "put_recipe")
    assert put_row["status"] == "ok"
    assert put_row["canonical_id"] == cid


def test_session_dir_skeleton_created(session_dir: Path):
    _ = RecipeSnapshotClient(session_dir=session_dir, kb_url=KB_URL)
    assert recipe_snapshot_dir(session_dir).exists()
