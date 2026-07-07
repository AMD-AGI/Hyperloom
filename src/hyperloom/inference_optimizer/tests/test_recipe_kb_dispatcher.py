# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for :class:`RecipeKB` — the local-write / remote-read dispatcher.

Covers: writes never touch the remote; reads are remote-first with local
fallback (empty/raise all fall through to authoritative local writes);
``remote=None`` and ``remote.enabled=False`` are local-only. The remote is
duck-typed (gbrain is the only backend); these tests drive a fake remote so
the dispatcher's routing/translation/audit logic is exercised without network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.inference_optimizer.recipe_kb import (
    LocalRecipeStore,
    RecipeKB,
    RemoteRecipeClientError,
    recipe_canonical_id,
)
from hyperloom.inference_optimizer.recipe_kb.dispatcher import _v2_to_arbor


def test_v2_to_arbor_reads_legacy_framework_label() -> None:
    """v2 rows whose ``labels`` predate the framework->framework_name rename
    store the serving framework under the legacy ``framework`` key. The arbor
    projection must still surface it as ``framework_name``."""
    v2 = {
        "canonical_id": "inference:m:mi300x:sglang:unknown_model_type:unknown_arch:0.4.5:fp8",
        "labels": {"model": "m", "hardware": "mi300x", "framework": "sglang"},
        "body": {},
        "metrics": {},
    }
    arbor = _v2_to_arbor(v2)
    assert arbor["framework_name"] == "sglang"


def _cid(model: str = "m") -> str:
    return recipe_canonical_id(
        model=model,
        hardware="mi300x",
        framework_name="sglang",
        framework_version="0.4.5",
        precision="fp8",
    )


# Fixtures
@pytest.fixture
def local_store(tmp_path: Path) -> LocalRecipeStore:
    return LocalRecipeStore(root=tmp_path / "kb")


# Name matches the real client so ``RecipeKB._remote_label`` resolves "gbrain".
class GbrainRemoteRecipeClient:
    """Duck-typed read remote that returns canned rows / raises on demand.

    ``get_row`` drives the exact-cid fast path; ``search_rows`` drives the
    label-match search. ``raise_exc`` makes every read raise (transport error).
    """

    enabled = True

    def __init__(
        self,
        *,
        get_row: dict[str, Any] | None = None,
        search_rows: list[dict[str, Any]] | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self._get_row = get_row
        self._search_rows = search_rows or []
        self._raise = raise_exc
        self.last_search_kwargs: dict[str, Any] | None = None
        self.search_calls: list[dict[str, Any]] = []
        self.closed = False

    def get_recipe(self, *, canonical_id: str, version: int | None = None) -> dict[str, Any] | None:
        if self._raise is not None:
            raise self._raise
        return self._get_row

    def search(
        self,
        *,
        label_match: dict[str, Any] | None = None,
        metric_filters: dict[str, Any] | None = None,
        updated_since: str | None = None,
        order_by: str = "updated_at DESC",
        limit: int = 50,
        prefer: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.last_search_kwargs = {
            "label_match": label_match,
            "metric_filters": metric_filters,
            "updated_since": updated_since,
            "order_by": order_by,
            "limit": limit,
            "prefer": prefer,
        }
        self.search_calls.append(self.last_search_kwargs)
        if self._raise is not None:
            raise self._raise
        return list(self._search_rows)

    def close(self) -> None:
        self.closed = True


# Writes — local-only
class _ReadOnlyRemoteSpy:
    """A remote-shaped sentinel that raises if any read method is invoked."""

    enabled = True

    def _boom(self, *_a: Any, **_k: Any) -> Any:
        raise AssertionError(
            "remote read method invoked during a write — writes must be local-only",
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


def test_put_recipe_never_touches_remote(local_store: LocalRecipeStore) -> None:
    kb = RecipeKB(local=local_store, remote=_ReadOnlyRemoteSpy())  # type: ignore[arg-type]
    cid = _cid()
    out = kb.put_recipe(canonical_id=cid, best_throughput=1.0)
    assert out["created"] is True
    assert local_store.get_recipe(canonical_id=cid) is not None


def test_append_attempt_never_touches_remote(local_store: LocalRecipeStore) -> None:
    kb = RecipeKB(local=local_store, remote=_ReadOnlyRemoteSpy())  # type: ignore[arg-type]
    cid = _cid()
    out = kb.append_attempt(canonical_id=cid, session_id="s", outcome="kept")
    assert out["id"] == 1


def test_delete_recipe_never_touches_remote(local_store: LocalRecipeStore) -> None:
    kb = RecipeKB(local=local_store, remote=_ReadOnlyRemoteSpy())  # type: ignore[arg-type]
    cid = _cid()
    local_store.put_recipe(canonical_id=cid)
    assert kb.delete_recipe(canonical_id=cid) is True


# Reads — remote-first, local fallback (fake remote)
def test_get_recipe_returns_remote_when_remote_hits(local_store: LocalRecipeStore) -> None:
    """A remote hit is returned, translated from the nested v2 envelope to arbor."""
    cid = _cid()
    v2_payload = {
        "canonical_id": cid,
        "version": 5,
        "labels": {"model": "remote-model", "hardware": "mi300x"},
        "body": {"best_config": {"tp": "16"}},
        "metrics": {"throughput": 30000.0},
    }
    kb = RecipeKB(local=local_store, remote=GbrainRemoteRecipeClient(search_rows=[v2_payload]))
    out = kb.get_recipe(canonical_id=cid)
    assert out is not None
    assert out["canonical_id"] == cid
    assert out["version"] == 5
    assert out["model"] == "remote-model"
    assert out["hardware"] == "mi300x"
    assert out["best_config"] == {"tp": "16"}
    assert out["best_throughput"] == 30000.0


def test_v2_framework_version_label_is_read(local_store: LocalRecipeStore) -> None:
    """A remote row's ``framework_version`` label is surfaced; the row ``version`` stays independent."""
    cid = _cid()
    v2_payload = {
        "canonical_id": cid,
        "version": 7,
        "labels": {
            "model": "remote-model",
            "hardware": "mi300x",
            "framework_name": "sglang",
            "framework_version": "0.4.5",
            "precision": "fp8",
        },
        "body": {},
        "metrics": {},
    }
    kb = RecipeKB(local=local_store, remote=GbrainRemoteRecipeClient(search_rows=[v2_payload]))
    out = kb.get_recipe(canonical_id=cid)
    assert out is not None
    assert out["framework_version"] == "0.4.5"
    assert out["version"] == 7


def test_v2_legacy_version_label_is_not_read(local_store: LocalRecipeStore) -> None:
    """The legacy ``labels.version`` key is no longer read; a row with only it yields empty ``framework_version``."""
    cid = _cid()
    v2_payload = {
        "canonical_id": cid,
        "version": 7,
        "labels": {
            "model": "remote-model",
            "hardware": "mi300x",
            "framework_name": "sglang",
            "version": "0.4.5",
            "precision": "fp8",
        },
        "body": {},
        "metrics": {},
    }
    kb = RecipeKB(local=local_store, remote=GbrainRemoteRecipeClient(search_rows=[v2_payload]))
    out = kb.get_recipe(canonical_id=cid)
    assert out is not None
    assert out["framework_version"] == ""
    assert out["version"] == 7


def test_get_recipe_falls_through_to_local_when_remote_empty(local_store: LocalRecipeStore) -> None:
    """Remote search finding no match must NOT shadow an authoritative local write."""
    cid = _cid()
    local_store.put_recipe(canonical_id=cid, model="local-marker")
    kb = RecipeKB(local=local_store, remote=GbrainRemoteRecipeClient(search_rows=[]))
    out = kb.get_recipe(canonical_id=cid)
    assert out is not None
    assert out["model"] == "local-marker"


def test_get_recipe_falls_through_to_local_on_transport_error(local_store: LocalRecipeStore) -> None:
    cid = _cid()
    local_store.put_recipe(canonical_id=cid, model="local-marker")
    failures: list[tuple[str, Exception]] = []
    remote = GbrainRemoteRecipeClient(raise_exc=RemoteRecipeClientError("down", category="transport"))
    kb = RecipeKB(local=local_store, remote=remote)
    kb.on_remote_failure = lambda m, exc: failures.append((m, exc))
    out = kb.get_recipe(canonical_id=cid)
    assert out is not None
    assert out["model"] == "local-marker"
    assert len(failures) == 1
    assert failures[0][0] == "get_recipe"
    assert isinstance(failures[0][1], RemoteRecipeClientError)


def test_get_recipe_returns_none_when_neither_has_it(local_store: LocalRecipeStore) -> None:
    cid = _cid(model="never-seen")
    kb = RecipeKB(local=local_store, remote=GbrainRemoteRecipeClient(search_rows=[]))
    assert kb.get_recipe(canonical_id=cid) is None


def test_get_recipe_remote_sends_5tuple_label_match(local_store: LocalRecipeStore) -> None:
    """The label-match search receives the full 7-tuple decoded from the cid."""
    cid = _cid()
    remote = GbrainRemoteRecipeClient(search_rows=[])  # fast path miss -> search
    kb = RecipeKB(local=local_store, remote=remote)
    kb.get_recipe(canonical_id=cid)
    label_match = remote.search_calls[0]["label_match"] if remote.search_calls else {}
    assert set(label_match) == {
        "model",
        "hardware",
        "framework_name",
        "framework_version",
        "precision",
        "model_type",
        "architectures",
    }
    assert all(label_match.values()), "all cid segments must be present"


def test_get_recipe_remote_falls_back_without_framework_version(
    local_store: LocalRecipeStore,
) -> None:
    """Remote get_recipe tolerates framework-version drift after exact misses."""
    requested = recipe_canonical_id(
        model="test-model",
        hardware="mi300x",
        framework_name="sglang",
        model_type="qwen3",
        architectures="qwen3forcausallm",
        framework_version="0.5.12",
        precision="fp8",
    )
    sibling = recipe_canonical_id(
        model="test-model",
        hardware="mi300x",
        framework_name="sglang",
        model_type="qwen3",
        architectures="qwen3forcausallm",
        framework_version="0.5.11",
        precision="fp8",
    )
    row = {
        "canonical_id": sibling,
        "version": 4,
        "labels": {
            "model": "test-model",
            "hardware": "mi300x",
            "framework_name": "sglang",
            "framework_version": "0.5.11",
            "precision": "fp8",
            "model_type": "qwen3",
            "architectures": "qwen3forcausallm",
        },
        "body": {"best_config": {"tp": "8"}},
        "metrics": {"throughput": 12345.0},
    }

    class _VersionDriftRemote(GbrainRemoteRecipeClient):
        """Remote fake that only returns a row for relaxed label matches."""

        def search(self, **kwargs: Any) -> list[dict[str, Any]]:
            super().search(**kwargs)
            labels = kwargs.get("label_match") or {}
            if "framework_version" in labels:
                return []
            return [row]

    remote = _VersionDriftRemote()
    kb = RecipeKB(local=local_store, remote=remote)

    out = kb.get_recipe(canonical_id=requested)

    assert out is not None
    assert out["canonical_id"] == sibling
    assert out["best_config"] == {"tp": "8"}
    assert len(remote.search_calls) == 2
    assert remote.search_calls[0]["label_match"]["framework_version"] == "0.5.12"
    assert "framework_version" not in remote.search_calls[1]["label_match"]


def test_get_recipe_remote_exact_search_hit_skips_version_fallback(
    local_store: LocalRecipeStore,
) -> None:
    """A full label-match hit must not issue the relaxed version fallback."""
    cid = recipe_canonical_id(
        model="test-model",
        hardware="mi300x",
        framework_name="sglang",
        model_type="qwen3",
        architectures="qwen3forcausallm",
        framework_version="0.5.12",
        precision="fp8",
    )
    row = {"canonical_id": cid, "version": 3, "labels": {"model": "test-model"}, "body": {}, "metrics": {}}
    remote = GbrainRemoteRecipeClient(search_rows=[row])
    kb = RecipeKB(local=local_store, remote=remote)

    out = kb.get_recipe(canonical_id=cid)

    assert out is not None
    assert out["canonical_id"] == cid
    assert len(remote.search_calls) == 1


def test_get_history_is_local_only(local_store: LocalRecipeStore) -> None:
    """get_history is LOCAL only; the dispatcher must not touch the remote."""
    cid = _cid()
    local_store.put_recipe(canonical_id=cid)
    local_store.put_recipe(canonical_id=cid)
    kb = RecipeKB(local=local_store, remote=_ReadOnlyRemoteSpy())  # type: ignore[arg-type]
    rows = kb.get_history(canonical_id=cid)
    assert len(rows) == 1
    assert rows[0]["version"] == 1


def test_search_falls_through_to_local_on_remote_failure(local_store: LocalRecipeStore) -> None:
    cid = _cid()
    local_store.put_recipe(
        canonical_id=cid,
        extras={"task": "pretrain"},
        best_throughput=10000.0,
    )
    remote = GbrainRemoteRecipeClient(raise_exc=RemoteRecipeClientError("down", category="transport"))
    kb = RecipeKB(local=local_store, remote=remote)
    rows = kb.search(label_match={"task": "pretrain"})
    assert len(rows) == 1
    assert rows[0]["canonical_id"] == cid


def test_list_recent_is_local_only(local_store: LocalRecipeStore) -> None:
    """list_recent is LOCAL only."""
    cid_local = _cid(model="local-only")
    local_store.put_recipe(canonical_id=cid_local)
    kb = RecipeKB(local=local_store, remote=_ReadOnlyRemoteSpy())  # type: ignore[arg-type]
    rows = kb.list_recent()
    assert [r["canonical_id"] for r in rows] == [cid_local]


def test_list_attempts_is_local_only(local_store: LocalRecipeStore) -> None:
    """list_attempts is LOCAL only."""
    cid = _cid()
    local_store.append_attempt(canonical_id=cid, session_id="s", outcome="kept")
    kb = RecipeKB(local=local_store, remote=_ReadOnlyRemoteSpy())  # type: ignore[arg-type]
    rows = kb.list_attempts(canonical_id=cid)
    assert len(rows) == 1
    assert rows[0]["outcome"] == "kept"


def test_list_session_attempts_is_local_only(local_store: LocalRecipeStore) -> None:
    """list_session_attempts is LOCAL only."""
    cid = _cid()
    local_store.append_attempt(canonical_id=cid, session_id="sess-7", outcome="kept")
    kb = RecipeKB(local=local_store, remote=_ReadOnlyRemoteSpy())  # type: ignore[arg-type]
    rows = kb.list_session_attempts(session_id="sess-7")
    assert len(rows) == 1
    assert rows[0]["session_id"] == "sess-7"


# remote=None / remote disabled — local-only mode
def test_no_remote_means_local_only_for_reads(local_store: LocalRecipeStore) -> None:
    """``remote=None`` reads only from the local store and never makes a network call."""
    kb = RecipeKB(local=local_store, remote=None)
    cid = _cid()
    local_store.put_recipe(canonical_id=cid, model="local-marker")
    out = kb.get_recipe(canonical_id=cid)
    assert out is not None
    assert out["model"] == "local-marker"


def test_disabled_remote_treated_as_no_remote(local_store: LocalRecipeStore) -> None:
    """A ``remote.enabled=False`` client behaves identically to ``remote=None``."""
    disabled = GbrainRemoteRecipeClient(search_rows=[{"canonical_id": "x"}])
    disabled.enabled = False
    kb = RecipeKB(local=local_store, remote=disabled)
    cid = _cid()
    local_store.put_recipe(canonical_id=cid, model="local-marker")
    out = kb.get_recipe(canonical_id=cid)
    assert out is not None
    assert out["model"] == "local-marker"


# Audit hook — recipe-snapshot read trace
def test_audit_hook_emitted_on_remote_hit(local_store: LocalRecipeStore) -> None:
    cid = _cid()
    events: list[dict[str, Any]] = []
    v2_payload = {
        "canonical_id": cid,
        "version": 5,
        "labels": {"model": "remote-model", "hardware": "mi300x"},
        "body": {"best_config": {"tp": "16"}},
        "metrics": {"throughput": 30000.0},
    }
    kb = RecipeKB(local=local_store, remote=GbrainRemoteRecipeClient(search_rows=[v2_payload]))
    kb.audit_hook = lambda e: events.append(e)
    kb.get_recipe(canonical_id=cid, prefer={"tp": 8})
    assert len(events) == 1
    ev = events[0]
    assert ev["method"] == "get_recipe"
    assert ev["remote"] == "gbrain"
    assert ev["resolution"] == "remote"
    assert ev["hit"] is True
    assert ev["request"]["canonical_id"] == cid
    assert ev["request"]["prefer_keys"] == ["tp"]
    assert ev["result"]["exact"] is True
    assert ev["result"]["best_throughput"] == 30000.0


def test_audit_hook_emitted_on_local_fallthrough(local_store: LocalRecipeStore) -> None:
    cid = _cid()
    local_store.put_recipe(canonical_id=cid, model="local-marker")
    events: list[dict[str, Any]] = []
    remote = GbrainRemoteRecipeClient(raise_exc=RemoteRecipeClientError("down", category="transport"))
    kb = RecipeKB(local=local_store, remote=remote)
    kb.audit_hook = lambda e: events.append(e)
    kb.get_recipe(canonical_id=cid)
    assert len(events) == 1
    ev = events[0]
    assert ev["resolution"] == "remote_error"
    assert ev["hit"] is True  # served from local


def test_audit_hook_no_remote_records_local(local_store: LocalRecipeStore) -> None:
    kb = RecipeKB(local=local_store, remote=None)
    cid = _cid()
    events: list[dict[str, Any]] = []
    kb.audit_hook = lambda e: events.append(e)
    kb.get_recipe(canonical_id=cid)
    assert len(events) == 1
    assert events[0]["remote"] == "none"
    assert events[0]["resolution"] == "local"
    assert events[0]["hit"] is False


def test_audit_hook_never_raises_into_caller(local_store: LocalRecipeStore) -> None:
    cid = _cid(model="never-seen")

    def _boom(_e: dict[str, Any]) -> None:
        raise RuntimeError("audit sink down")

    kb = RecipeKB(local=local_store, remote=GbrainRemoteRecipeClient(search_rows=[]))
    kb.audit_hook = _boom
    # Must not raise despite the failing hook.
    assert kb.get_recipe(canonical_id=cid) is None


# Lifecycle
def test_close_releases_remote_transport(local_store: LocalRecipeStore) -> None:
    """``RecipeKB.close`` must call ``remote.close``."""
    remote = GbrainRemoteRecipeClient()
    kb = RecipeKB(local=local_store, remote=remote)
    kb.close()
    assert remote.closed is True


def test_close_with_no_remote_is_noop(local_store: LocalRecipeStore) -> None:
    kb = RecipeKB(local=local_store, remote=None)
    kb.close()


def test_close_swallows_remote_close_error(local_store: LocalRecipeStore) -> None:
    remote = GbrainRemoteRecipeClient()

    def _boom() -> None:
        raise RuntimeError("close failed")

    remote.close = _boom  # type: ignore[method-assign]
    kb = RecipeKB(local=local_store, remote=remote)
    kb.close()
