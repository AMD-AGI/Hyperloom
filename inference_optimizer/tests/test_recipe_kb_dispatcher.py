"""Tests for :class:`RecipeKB` — the local-write / remote-read dispatcher.

Covers the contract documented at the top of ``recipe_kb/dispatcher.py``:

* writes (put_recipe / append_attempt / delete_recipe) NEVER touch
  the remote — verified with a spy ``RemoteRecipeClient`` that
  raises if any read method is called during a write;
* reads (get_recipe / get_history / search / list_recent /
  list_attempts / list_session_attempts):
    * remote-first when configured AND enabled;
    * remote-success result returned verbatim;
    * remote 404 / empty falls THROUGH to local (so a freshly-
      bootstrapped remote can't shadow our authoritative local
      writes);
    * remote raise → local fallback + ``on_remote_failure`` callback;
* ``remote=None`` → local-only mode (no fallback ladder, no
  network calls);
* ``remote.enabled=False`` → equivalent to ``remote=None`` (the
  client itself short-circuits, dispatcher never even tries).

Also covers integration with the actual :class:`RemoteRecipeClient`
+ :class:`LocalRecipeStore` so the dispatcher's contract is tested
end-to-end at least once (no all-mocks unit testing).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from inference_optimizer.recipe_kb import (
    LocalRecipeStore,
    RecipeKB,
    RemoteRecipeClient,
    RemoteRecipeClientError,
    recipe_canonical_id,
)
from inference_optimizer.recipe_snapshot_constants import (
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


# ===========================================================================
# Fixtures
# ===========================================================================
@pytest.fixture
def env_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "CORTEX_KB_URL",
        "CORTEX_KB_HTTP_TIMEOUT_SEC",
        "CORTEX_KB_RETRY_ATTEMPTS",
        "CORTEX_KB_MAX_CONCURRENCY",
        "KB_SERVICE_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def local_store(tmp_path: Path) -> LocalRecipeStore:
    return LocalRecipeStore(root=tmp_path / "kb")


@pytest.fixture
def remote(env_clean: None) -> RemoteRecipeClient:
    return RemoteRecipeClient(
        kb_url=KB_URL, foreground=False, retry_attempts=1,
    )


@pytest.fixture
def kb(local_store: LocalRecipeStore, remote: RemoteRecipeClient) -> RecipeKB:
    return RecipeKB(local=local_store, remote=remote)


# ===========================================================================
# Writes — local-only
# ===========================================================================
class _ReadOnlyRemoteSpy:
    """A ``RemoteRecipeClient``-shaped sentinel that raises if any
    read method is invoked. Used to assert writes never touch remote."""

    enabled = True

    def _boom(self, *_a: Any, **_k: Any) -> Any:
        raise AssertionError(
            "remote read method invoked during a write — "
            "writes must be local-only",
        )

    health = _boom
    get_recipe = _boom
    get_history = _boom
    list_recent = _boom
    search = _boom
    list_attempts = _boom
    list_session_attempts = _boom
    session_summary = _boom

    def close(self) -> None:
        pass


def test_put_recipe_never_touches_remote(
    local_store: LocalRecipeStore,
) -> None:
    kb = RecipeKB(local=local_store, remote=_ReadOnlyRemoteSpy())  # type: ignore[arg-type]
    cid = _cid()
    out = kb.put_recipe(canonical_id=cid, best_throughput=1.0)
    assert out["created"] is True
    assert local_store.get_recipe(canonical_id=cid) is not None


def test_append_attempt_never_touches_remote(
    local_store: LocalRecipeStore,
) -> None:
    kb = RecipeKB(local=local_store, remote=_ReadOnlyRemoteSpy())  # type: ignore[arg-type]
    cid = _cid()
    out = kb.append_attempt(canonical_id=cid, session_id="s", outcome="kept")
    assert out["id"] == 1


def test_delete_recipe_never_touches_remote(
    local_store: LocalRecipeStore,
) -> None:
    kb = RecipeKB(local=local_store, remote=_ReadOnlyRemoteSpy())  # type: ignore[arg-type]
    cid = _cid()
    local_store.put_recipe(canonical_id=cid)
    assert kb.delete_recipe(canonical_id=cid) is True


# ===========================================================================
# Reads — remote-first, local fallback (real remote via respx)
# ===========================================================================
def test_get_recipe_returns_remote_when_remote_hits(kb: RecipeKB) -> None:
    """Remote answers 200 with a v2-shape recipe — dispatcher must
    return it after translating to arbor shape (top-level
    ``model``/``hardware``/``best_throughput`` etc.)."""
    cid = _cid()
    # Simulate the central kb-service v2 wire shape: identity goes
    # in ``labels``, throughput in ``metrics``, best_config in body.
    v2_payload = {
        "canonical_id": cid,
        "version":      5,
        "labels":       {"model": "remote-model", "hardware": "mi300x"},
        "body":         {"best_config": {"tp": "16"}},
        "metrics":      {"throughput": 30000.0},
    }
    with respx.mock(base_url=KB_URL) as mock:
        mock.get(format_recipe_path(PATH_RECIPE_TPL, cid)).mock(
            return_value=httpx.Response(200, json=v2_payload),
        )
        out = kb.get_recipe(canonical_id=cid)
    assert out is not None
    # Translated arbor shape — top-level identity + best_config /
    # best_throughput pulled out of v2's nested body / metrics.
    assert out["canonical_id"]     == cid
    assert out["version"]          == 5
    assert out["model"]            == "remote-model"
    assert out["hardware"]         == "mi300x"
    assert out["best_config"]      == {"tp": "16"}
    assert out["best_throughput"]  == 30000.0


def test_get_recipe_falls_through_to_local_on_remote_404(
    kb: RecipeKB, local_store: LocalRecipeStore,
) -> None:
    """Remote 404 must NOT shadow a local write — the dispatcher's
    invariant is "local is authoritative for writes"."""
    cid = _cid()
    local_store.put_recipe(
        canonical_id=cid, model="local-marker",
    )
    with respx.mock(base_url=KB_URL) as mock:
        mock.get(format_recipe_path(PATH_RECIPE_TPL, cid)).mock(
            return_value=httpx.Response(
                404,
                json={"detail": {"error": {"code": "NOT_FOUND",
                                            "message": "missing",
                                            "details": {}}}},
            ),
        )
        out = kb.get_recipe(canonical_id=cid)
    assert out is not None
    assert out["model"] == "local-marker"


def test_get_recipe_falls_through_to_local_on_transport_error(
    kb: RecipeKB, local_store: LocalRecipeStore,
) -> None:
    cid = _cid()
    local_store.put_recipe(canonical_id=cid, model="local-marker")
    failures: list[tuple[str, Exception]] = []
    kb.on_remote_failure = lambda m, exc: failures.append((m, exc))
    with respx.mock(base_url=KB_URL) as mock:
        mock.get(format_recipe_path(PATH_RECIPE_TPL, cid)).mock(
            return_value=httpx.Response(503, json={"detail": "warming up"}),
        )
        out = kb.get_recipe(canonical_id=cid)
    assert out is not None
    assert out["model"] == "local-marker"
    assert len(failures) == 1
    assert failures[0][0] == "get_recipe"
    assert isinstance(failures[0][1], RemoteRecipeClientError)


def test_get_recipe_returns_none_when_neither_has_it(
    kb: RecipeKB,
) -> None:
    cid = _cid(model="never-seen")
    with respx.mock(base_url=KB_URL) as mock:
        mock.get(format_recipe_path(PATH_RECIPE_TPL, cid)).mock(
            return_value=httpx.Response(
                404,
                json={"detail": {"error": {"code": "NOT_FOUND",
                                            "message": "missing",
                                            "details": {}}}},
            ),
        )
        assert kb.get_recipe(canonical_id=cid) is None


def test_get_history_falls_through_when_remote_returns_empty(
    kb: RecipeKB, local_store: LocalRecipeStore,
) -> None:
    """Remote returning [] for an unknown id is a normal-shape
    response, but if the local store has archived rows we must
    surface them (writes go local; the central can be stale)."""
    cid = _cid()
    local_store.put_recipe(canonical_id=cid)
    local_store.put_recipe(canonical_id=cid)  # creates v1 history
    with respx.mock(base_url=KB_URL) as mock:
        mock.get(format_recipe_path(PATH_RECIPE_HISTORY_TPL, cid)).mock(
            return_value=httpx.Response(
                200, json={"canonical_id": cid, "history": []},
            ),
        )
        rows = kb.get_history(canonical_id=cid)
    assert len(rows) == 1
    assert rows[0]["version"] == 1


def test_search_falls_through_to_local_on_remote_failure(
    kb: RecipeKB, local_store: LocalRecipeStore,
) -> None:
    cid = _cid()
    local_store.put_recipe(
        canonical_id=cid,
        extras={"task": "pretrain"},
        best_throughput=10000.0,
    )
    with respx.mock(base_url=KB_URL) as mock:
        mock.post(PATH_RECIPES_SEARCH).mock(
            return_value=httpx.Response(503, json={"detail": "down"}),
        )
        rows = kb.search(label_match={"task": "pretrain"})
    assert len(rows) == 1
    assert rows[0]["canonical_id"] == cid


def test_list_recent_remote_first_then_local(
    kb: RecipeKB, local_store: LocalRecipeStore,
) -> None:
    cid_local = _cid(model="local-only")
    local_store.put_recipe(canonical_id=cid_local)
    with respx.mock(base_url=KB_URL) as mock:
        mock.get(PATH_RECIPES_LIST).mock(
            return_value=httpx.Response(
                200,
                json={"recipes": [{"canonical_id": "remote:cid:1", "version": 1}]},
            ),
        )
        rows = kb.list_recent()
    # Remote rows win when remote is reachable AND non-empty.
    assert [r["canonical_id"] for r in rows] == ["remote:cid:1"]


def test_list_attempts_remote_then_local(
    kb: RecipeKB, local_store: LocalRecipeStore,
) -> None:
    cid = _cid()
    local_store.append_attempt(
        canonical_id=cid, session_id="s", outcome="kept",
    )
    with respx.mock(base_url=KB_URL) as mock:
        mock.get(
            format_recipe_path(
                "/recipe-snapshot/recipes/{canonical_id}/attempts", cid,
            ),
        ).mock(
            return_value=httpx.Response(503, json={"detail": "down"}),
        )
        rows = kb.list_attempts(canonical_id=cid)
    # Remote 503 → fall through to local; local has 1 row.
    assert len(rows) == 1
    assert rows[0]["outcome"] == "kept"


def test_list_session_attempts_falls_through(
    kb: RecipeKB, local_store: LocalRecipeStore,
) -> None:
    cid = _cid()
    local_store.append_attempt(
        canonical_id=cid, session_id="sess-7", outcome="kept",
    )
    with respx.mock(base_url=KB_URL) as mock:
        mock.get("/recipe-snapshot/sessions/sess-7/attempts").mock(
            return_value=httpx.Response(503, json={"detail": "down"}),
        )
        rows = kb.list_session_attempts(session_id="sess-7")
    assert len(rows) == 1
    assert rows[0]["session_id"] == "sess-7"


# ===========================================================================
# remote=None / remote disabled — local-only mode
# ===========================================================================
def test_no_remote_means_local_only_for_reads(
    local_store: LocalRecipeStore,
) -> None:
    """``--degraded-kb`` / no ``--cortex-kb-url`` path: dispatcher
    constructed with ``remote=None`` reads only from the local
    store and never makes a network call."""
    kb = RecipeKB(local=local_store, remote=None)
    cid = _cid()
    local_store.put_recipe(canonical_id=cid, model="local-marker")
    out = kb.get_recipe(canonical_id=cid)
    assert out is not None
    assert out["model"] == "local-marker"


def test_disabled_remote_treated_as_no_remote(
    local_store: LocalRecipeStore, env_clean: None,
) -> None:
    """A ``remote.enabled=False`` client behaves identically to
    ``remote=None`` from the dispatcher's perspective."""
    disabled = RemoteRecipeClient(kb_url=KB_URL, enabled=False)
    kb = RecipeKB(local=local_store, remote=disabled)
    cid = _cid()
    local_store.put_recipe(canonical_id=cid, model="local-marker")
    out = kb.get_recipe(canonical_id=cid)
    assert out is not None
    assert out["model"] == "local-marker"


# ===========================================================================
# Lifecycle
# ===========================================================================
def test_close_releases_remote_transport(
    local_store: LocalRecipeStore, remote: RemoteRecipeClient,
) -> None:
    """``RecipeKB.close`` must call ``remote.close`` (free up
    socket / keepalive connection pool)."""
    closed_marker: list[bool] = []
    remote.close = lambda: closed_marker.append(True)  # type: ignore[method-assign]
    kb = RecipeKB(local=local_store, remote=remote)
    kb.close()
    assert closed_marker == [True]


def test_close_with_no_remote_is_noop(local_store: LocalRecipeStore) -> None:
    kb = RecipeKB(local=local_store, remote=None)
    kb.close()


def test_close_swallows_remote_close_error(
    local_store: LocalRecipeStore, remote: RemoteRecipeClient,
) -> None:
    def _boom() -> None:
        raise RuntimeError("close failed")
    remote.close = _boom  # type: ignore[method-assign]
    kb = RecipeKB(local=local_store, remote=remote)
    kb.close()
