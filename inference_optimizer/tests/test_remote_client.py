# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Read-only tests for :class:`RemoteRecipeClient` — HTTP transport + read-method surface (writes go local-only)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from inference_optimizer.recipe_kb import (
    RemoteRecipeClient,
    RemoteRecipeClientError,
    recipe_canonical_id,
)
from inference_optimizer.recipe_snapshot_constants import (
    DEFAULT_HTTP_TIMEOUT_SEC,
    DEFAULT_RETRY_ATTEMPTS,
    FOREGROUND_HTTP_TIMEOUT_SEC,
    FOREGROUND_RETRY_ATTEMPTS,
    PATH_HEALTH,
    PATH_RECIPE_HISTORY_TPL,
    PATH_RECIPE_TPL,
    PATH_RECIPES_LIST,
    PATH_RECIPES_SEARCH,
    format_recipe_path,
)


KB_URL = "http://kb-test.local"


def _cid(model: str = "m") -> str:
    return recipe_canonical_id(
        model=model,
        hardware="mi300x",
        framework="sglang",
        framework_version="0.4.5",
        precision="fp8",
    )


@pytest.fixture
def env_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wipe the env vars the client consults so tests opt into each one explicitly."""
    for key in (
        "CORTEX_KB_URL",
        "CORTEX_KB_HTTP_TIMEOUT_SEC",
        "CORTEX_KB_RETRY_ATTEMPTS",
        "CORTEX_KB_MAX_CONCURRENCY",
        "KB_SERVICE_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def client(env_clean: None) -> RemoteRecipeClient:
    return RemoteRecipeClient(
        kb_url=KB_URL,
        foreground=False,
        retry_attempts=1,
    )


# Health
def test_health_returns_true_on_status_ok(client: RemoteRecipeClient) -> None:
    with respx.mock(base_url=KB_URL) as mock:
        mock.get(PATH_HEALTH).mock(
            return_value=httpx.Response(200, json={"status": "ok"}),
        )
        assert client.health() is True


def test_health_returns_false_on_500(client: RemoteRecipeClient) -> None:
    with respx.mock(base_url=KB_URL) as mock:
        mock.get(PATH_HEALTH).mock(
            return_value=httpx.Response(503, json={"detail": "warming up"}),
        )
        assert client.health() is False


def test_health_returns_false_when_disabled(env_clean: None) -> None:
    c = RemoteRecipeClient(kb_url=KB_URL, enabled=False)
    assert c.health() is False


def test_health_returns_false_on_unexpected_body(client: RemoteRecipeClient) -> None:
    with respx.mock(base_url=KB_URL) as mock:
        mock.get(PATH_HEALTH).mock(
            return_value=httpx.Response(200, json={"status": "warming"}),
        )
        assert client.health() is False


# get_recipe
def test_get_recipe_returns_dict_on_200(client: RemoteRecipeClient) -> None:
    cid = _cid()
    expected = {"canonical_id": cid, "version": 3, "metrics": {"x": 1.0}}
    with respx.mock(base_url=KB_URL) as mock:
        mock.get(format_recipe_path(PATH_RECIPE_TPL, cid)).mock(
            return_value=httpx.Response(200, json=expected),
        )
        assert client.get_recipe(canonical_id=cid) == expected


def test_get_recipe_returns_none_on_404(client: RemoteRecipeClient) -> None:
    """A missing recipe is normal — None is the dispatcher's "fall through to local" trigger."""
    cid = _cid(model="absent")
    with respx.mock(base_url=KB_URL) as mock:
        mock.get(format_recipe_path(PATH_RECIPE_TPL, cid)).mock(
            return_value=httpx.Response(
                404,
                json={
                    "detail": {
                        "error": {
                            "code": "NOT_FOUND",
                            "message": "missing",
                            "details": {"canonical_id": cid},
                        },
                    },
                },
            ),
        )
        assert client.get_recipe(canonical_id=cid) is None


def test_get_recipe_with_version_forwards_query_param(
    client: RemoteRecipeClient,
) -> None:
    cid = _cid()
    with respx.mock(base_url=KB_URL) as mock:
        route = mock.get(format_recipe_path(PATH_RECIPE_TPL, cid)).mock(
            return_value=httpx.Response(200, json={"canonical_id": cid, "version": 2}),
        )
        client.get_recipe(canonical_id=cid, version=2)
        assert route.called
        assert route.calls.last.request.url.params["version"] == "2"


def test_get_recipe_500_raises_transport_error(
    client: RemoteRecipeClient,
) -> None:
    cid = _cid()
    with respx.mock(base_url=KB_URL) as mock:
        mock.get(format_recipe_path(PATH_RECIPE_TPL, cid)).mock(
            return_value=httpx.Response(503, json={"detail": "warming up"}),
        )
        with pytest.raises(RemoteRecipeClientError) as ei:
            client.get_recipe(canonical_id=cid)
    assert ei.value.category == "transport"


def test_get_recipe_disabled_short_circuits(env_clean: None) -> None:
    c = RemoteRecipeClient(kb_url=KB_URL, enabled=False)
    # No respx mock — disabled client must not issue any request.
    assert c.get_recipe(canonical_id=_cid()) is None


def test_get_recipe_rejects_empty_canonical_id(client: RemoteRecipeClient) -> None:
    with pytest.raises(ValueError):
        client.get_recipe(canonical_id="")


# get_history
def test_get_history_returns_archives(client: RemoteRecipeClient) -> None:
    cid = _cid()
    payload = {
        "canonical_id": cid,
        "history": [
            {"canonical_id": cid, "version": 1, "snapshot": {"version": 1}},
            {"canonical_id": cid, "version": 2, "snapshot": {"version": 2}},
        ],
    }
    with respx.mock(base_url=KB_URL) as mock:
        mock.get(format_recipe_path(PATH_RECIPE_HISTORY_TPL, cid)).mock(
            return_value=httpx.Response(200, json=payload),
        )
        assert client.get_history(canonical_id=cid) == payload["history"]


def test_get_history_empty_for_unknown_id(client: RemoteRecipeClient) -> None:
    """Spec: unknown id returns ``{"history": []}`` (no 404)."""
    cid = _cid(model="absent")
    with respx.mock(base_url=KB_URL) as mock:
        mock.get(format_recipe_path(PATH_RECIPE_HISTORY_TPL, cid)).mock(
            return_value=httpx.Response(
                200, json={"canonical_id": cid, "history": []},
            ),
        )
        assert client.get_history(canonical_id=cid) == []


def test_get_history_disabled_returns_empty(env_clean: None) -> None:
    c = RemoteRecipeClient(kb_url=KB_URL, enabled=False)
    assert c.get_history(canonical_id=_cid()) == []


# list_recent / search
def test_list_recent_forwards_limit(client: RemoteRecipeClient) -> None:
    with respx.mock(base_url=KB_URL) as mock:
        route = mock.get(PATH_RECIPES_LIST).mock(
            return_value=httpx.Response(200, json={"recipes": []}),
        )
        client.list_recent(limit=25)
        assert route.calls.last.request.url.params["limit"] == "25"


def test_search_posts_full_body(client: RemoteRecipeClient) -> None:
    """Body must match the server's POST /recipes/search shape."""
    with respx.mock(base_url=KB_URL) as mock:
        route = mock.post(PATH_RECIPES_SEARCH).mock(
            return_value=httpx.Response(
                200, json={"recipes": [{"canonical_id": _cid(), "version": 1}]},
            ),
        )
        client.search(
            label_match={"hardware": "mi300x"},
            metric_filters={"throughput": {"min": 10000}},
            updated_since="2026-05-01T00:00:00Z",
            order_by="updated_at DESC",
            limit=20,
        )
        sent: dict[str, Any] = json.loads(route.calls.last.request.content)
        assert sent["label_match"]    == {"hardware": "mi300x"}
        assert sent["metric_filters"] == {"throughput": {"min": 10000}}
        assert sent["updated_since"]  == "2026-05-01T00:00:00Z"
        assert sent["order_by"]       == "updated_at DESC"
        assert sent["limit"]          == 20


def test_search_omits_optional_fields_when_unset(
    client: RemoteRecipeClient,
) -> None:
    with respx.mock(base_url=KB_URL) as mock:
        route = mock.post(PATH_RECIPES_SEARCH).mock(
            return_value=httpx.Response(200, json={"recipes": []}),
        )
        client.search()
        sent: dict[str, Any] = json.loads(route.calls.last.request.content)
        assert "label_match"    not in sent
        assert "metric_filters" not in sent
        assert "updated_since"  not in sent
        assert sent["order_by"] == "updated_at DESC"
        assert sent["limit"]    == 50


def test_search_returns_empty_list_when_disabled(env_clean: None) -> None:
    c = RemoteRecipeClient(kb_url=KB_URL, enabled=False)
    assert c.search() == []


# Attempts
def test_list_attempts_returns_attempts_array(
    client: RemoteRecipeClient,
) -> None:
    cid = _cid()
    with respx.mock(base_url=KB_URL) as mock:
        mock.get(
            format_recipe_path(
                "/recipe-snapshot/recipes/{canonical_id}/attempts",
                cid,
            ),
        ).mock(
            return_value=httpx.Response(
                200,
                json={"attempts": [
                    {"id": 1, "outcome": "kept"},
                    {"id": 2, "outcome": "reverted"},
                ]},
            ),
        )
        rows = client.list_attempts(canonical_id=cid)
    assert [r["id"] for r in rows] == [1, 2]


def test_list_session_attempts_uses_session_path(
    client: RemoteRecipeClient,
) -> None:
    with respx.mock(base_url=KB_URL) as mock:
        mock.get(
            "/recipe-snapshot/sessions/sess-1/attempts",
        ).mock(
            return_value=httpx.Response(
                200, json={"attempts": [{"id": 7, "outcome": "kept"}]},
            ),
        )
        rows = client.list_session_attempts(session_id="sess-1")
    assert rows[0]["id"] == 7


def test_session_summary_returns_dict(client: RemoteRecipeClient) -> None:
    with respx.mock(base_url=KB_URL) as mock:
        mock.get(
            "/recipe-snapshot/sessions/sess-1/summary",
        ).mock(
            return_value=httpx.Response(
                200,
                json={
                    "session_id": "sess-1",
                    "total_attempts": 4,
                    "kept": 1, "reverted": 2, "failed": 1, "skipped": 0,
                },
            ),
        )
        summary = client.session_summary(session_id="sess-1")
    assert summary is not None
    assert summary["total_attempts"] == 4


# Configuration / env
def test_kb_url_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORTEX_KB_URL", "http://env-url.example")
    c = RemoteRecipeClient()
    assert c.kb_url == "http://env-url.example"


def test_no_url_no_env_forces_local_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """No URL and no ``$CORTEX_KB_URL`` → client forces ``enabled=False`` (local-only, never a silent remote connect)."""
    monkeypatch.delenv("CORTEX_KB_URL", raising=False)
    c = RemoteRecipeClient()
    assert c.enabled is False
    assert c.kb_url is None


def test_explicit_kb_url_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORTEX_KB_URL", "http://env-url.example")
    c = RemoteRecipeClient(kb_url="http://explicit.example")
    assert c.kb_url == "http://explicit.example"


def test_env_overrides_take_precedence_for_timeout_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CORTEX_KB_HTTP_TIMEOUT_SEC", "7.5")
    monkeypatch.setenv("CORTEX_KB_RETRY_ATTEMPTS", "9")
    c = RemoteRecipeClient(kb_url=KB_URL)
    assert c.timeout_sec == 7.5
    assert c.retry_attempts == 9


def test_foreground_profile_defaults(env_clean: None) -> None:
    c = RemoteRecipeClient(kb_url=KB_URL, foreground=True)
    assert c.timeout_sec    == FOREGROUND_HTTP_TIMEOUT_SEC
    assert c.retry_attempts == FOREGROUND_RETRY_ATTEMPTS


def test_background_profile_defaults(env_clean: None) -> None:
    c = RemoteRecipeClient(kb_url=KB_URL, foreground=False)
    assert c.timeout_sec    == DEFAULT_HTTP_TIMEOUT_SEC
    assert c.retry_attempts == DEFAULT_RETRY_ATTEMPTS


def test_token_added_to_header_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_SERVICE_TOKEN", "tok-xyz")
    c = RemoteRecipeClient(kb_url=KB_URL, foreground=False, retry_attempts=1)
    with respx.mock(base_url=KB_URL) as mock:
        route = mock.get(PATH_HEALTH).mock(
            return_value=httpx.Response(200, json={"status": "ok"}),
        )
        c.health()
        auth = route.calls.last.request.headers.get("Authorization")
        assert auth == "Bearer tok-xyz"


def test_no_write_methods_present() -> None:
    """Defence-in-depth: the read-only contract is enforced at the class-attribute level too."""
    c = RemoteRecipeClient(kb_url=KB_URL, enabled=False)
    for forbidden in (
        "put_recipe",
        "append_attempt",
        "delete_recipe",
        "drain_pending",
        "_enqueue",
    ):
        assert not hasattr(c, forbidden), (
            f"RemoteRecipeClient must not expose {forbidden!r} — "
            "writes go local-only via LocalRecipeStore."
        )
