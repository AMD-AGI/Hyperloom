# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Local filesystem graph backend and KG factory integration tests."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.knowledge.recipe_kb.kg_client import KGClient, _entity
from hyperloom.orchestrator.knowledge.recipe_kb.local_graph_store import (
    LocalGraphStore,
    LocalGraphStoreError,
)
from hyperloom.orchestrator.knowledge.recipe_kb_t0 import _enhance_warm_start_with_kg


def _put(store: LocalGraphStore, slug: str, content: str | None = None) -> None:
    result = store.call(
        "put_page",
        {"slug": slug, "content": content or f"# {slug}\n\nKnowledge-graph entity node.\n"},
    )
    assert result["status"] in {"created", "updated"}


def _process_add(root: str, index: int) -> None:
    store = LocalGraphStore(root)
    store.call(
        "add_link",
        {
            "from": "hub",
            "to": f"process-{index}",
            "link_type": "improves",
            "context": f'{{"index":"{index}"}}',
        },
    )


def test_page_roundtrip_and_durable_layout(tmp_path: Path) -> None:
    root = tmp_path / "kg"
    store = LocalGraphStore(root)
    content = "# Entity\n\nA local graph page.\n"
    _put(store, "family/entity", content)

    page = store.call("get_page", {"slug": "family/entity"})
    assert page["slug"] == "family/entity"
    assert page["content"] == content
    assert store.call("list_pages", {"limit": 10}) == [{"slug": "family/entity"}]
    assert (root / "pages" / "family" / "entity.md").read_text(encoding="utf-8") == content
    assert (root / ".lock").is_file()


def test_list_pages_is_bounded_slug_only_and_never_reads_bodies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = LocalGraphStore(tmp_path / "kg")
    for slug in ("c", "a", "b"):
        _put(store, slug)

    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *_args, **_kwargs: pytest.fail("list_pages must not read page bodies"),
    )

    assert store.call("list_pages", {"offset": 1, "limit": 1}) == [{"slug": "b"}]


def test_edges_have_consistent_outbound_and_inbound_indexes(tmp_path: Path) -> None:
    store = LocalGraphStore(tmp_path / "kg")
    _put(store, "patch")
    _put(store, "arch")
    result = store.call(
        "add_link",
        {"from": "patch", "to": "arch", "link_type": "improves", "context": '{"gain":"+8%"}'},
    )

    assert result == {"status": "ok"}
    outbound = store.call("get_links", {"slug": "patch"})
    inbound = store.call("get_backlinks", {"slug": "arch"})
    assert outbound == inbound
    assert outbound[0]["from_slug"] == "patch"
    assert outbound[0]["to_slug"] == "arch"


def test_add_link_idempotently_replaces_context(tmp_path: Path) -> None:
    store = LocalGraphStore(tmp_path / "kg")
    _put(store, "patch")
    _put(store, "arch")
    args = {"from": "patch", "to": "arch", "link_type": "improves", "context": '{"gain":"+1%"}'}
    store.call("add_link", args)
    store.call("add_link", {**args, "context": '{"gain":"+9%"}'})

    outbound = store.call("get_links", {"slug": "patch"})
    inbound = store.call("get_backlinks", {"slug": "arch"})
    assert len(outbound) == len(inbound) == 1
    assert outbound[0]["context"] == inbound[0]["context"] == '{"gain":"+9%"}'


def test_traverse_and_search(tmp_path: Path) -> None:
    store = LocalGraphStore(tmp_path / "kg")
    for slug in ("a", "b", "c"):
        _put(store, slug, f"# {slug}\n\n{'Needle' if slug == 'b' else 'other'}\n")
    store.call("add_link", {"from": "a", "to": "b", "link_type": "uses_arch"})
    store.call("add_link", {"from": "b", "to": "c", "link_type": "variant_of"})

    traversed = store.call("traverse_graph", {"slug": "a", "depth": 2, "direction": "out"})
    assert [(edge["to_slug"], edge["depth"]) for edge in traversed] == [("b", 1), ("c", 2)]
    assert store.call("search", {"query": "needle", "limit": 10})[0]["slug"] == "b"


@pytest.mark.parametrize(
    "slug",
    ("../escape", "a/../../escape", "/absolute", "a//b", "a\\b", ".", "..", "a/$bad"),
)
def test_unsafe_slugs_are_rejected(tmp_path: Path, slug: str) -> None:
    store = LocalGraphStore(tmp_path / "kg")
    with pytest.raises(ValueError, match="unsafe slug"):
        store.call("get_page", {"slug": slug})


@pytest.mark.parametrize("raw", ("=", "/", ":", "a=b", "a:b", "-leading"))
def test_kg_entities_normalize_to_local_graph_slugs(tmp_path: Path, raw: str) -> None:
    slug = _entity(raw)
    store = LocalGraphStore(tmp_path / "kg")
    _put(store, slug)
    assert store.call("get_page", {"slug": slug})["slug"] == slug


def test_persists_across_instances(tmp_path: Path) -> None:
    root = tmp_path / "kg"
    first = LocalGraphStore(root)
    _put(first, "a")
    _put(first, "b")
    first.call("add_link", {"from": "a", "to": "b", "link_type": "improves", "context": "first"})

    second = LocalGraphStore(root)
    assert second.call("get_page", {"slug": "a"})["slug"] == "a"
    assert second.call("get_links", {"slug": "a"})[0]["context"] == "first"


def test_thread_locking_prevents_lost_edges(tmp_path: Path) -> None:
    root = tmp_path / "kg"
    store = LocalGraphStore(root)
    _put(store, "hub")
    for index in range(24):
        _put(store, f"thread-{index}")

    def add(index: int) -> None:
        LocalGraphStore(root).call(
            "add_link",
            {"from": "hub", "to": f"thread-{index}", "link_type": "improves"},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(add, range(24)))
    assert len(store.call("get_links", {"slug": "hub"})) == 24


def test_process_locking_prevents_lost_edges(tmp_path: Path) -> None:
    root = tmp_path / "kg"
    store = LocalGraphStore(root)
    _put(store, "hub")
    for index in range(8):
        _put(store, f"process-{index}")

    context = multiprocessing.get_context("fork")
    processes = [context.Process(target=_process_add, args=(str(root), index)) for index in range(8)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert len(store.call("get_links", {"slug": "hub"})) == 8


def test_failed_edge_commit_rolls_back_both_indexes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "kg"
    store = LocalGraphStore(root)
    _put(store, "a")
    _put(store, "b")
    original = store._atomic_write
    failed = False

    def fail_inbound_once(path: Path, content: str) -> None:
        nonlocal failed
        if path == store._edge_path("inbound", "b") and not failed:
            failed = True
            raise OSError("injected interruption")
        original(path, content)

    monkeypatch.setattr(store, "_atomic_write", fail_inbound_once)
    with pytest.raises(LocalGraphStoreError, match="injected interruption"):
        store.call("add_link", {"from": "a", "to": "b", "link_type": "improves"})
    assert store.call("get_links", {"slug": "a"}) == []
    assert store.call("get_backlinks", {"slug": "b"}) == []
    assert not (root / ".edge-transaction.json").exists()
    assert not list(root.rglob("*.tmp"))


def test_hard_interruption_recovers_both_indexes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "kg"
    store = LocalGraphStore(root)
    _put(store, "a")
    _put(store, "b")
    original = store._atomic_write
    interrupted = False

    def interrupt_inbound_once(path: Path, content: str) -> None:
        nonlocal interrupted
        if path == store._edge_path("inbound", "b") and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        original(path, content)

    monkeypatch.setattr(store, "_atomic_write", interrupt_inbound_once)
    with pytest.raises(KeyboardInterrupt):
        store.call("add_link", {"from": "a", "to": "b", "link_type": "improves"})
    assert (root / ".edge-transaction.json").is_file()
    recovered = LocalGraphStore(root)
    assert recovered.call("get_links", {"slug": "a"}) == recovered.call("get_backlinks", {"slug": "b"})
    assert len(recovered.call("get_links", {"slug": "a"})) == 1
    assert not (root / ".edge-transaction.json").exists()
    assert not list(root.rglob("*.tmp"))


def test_local_factory_ignores_ambient_gbrain_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.orchestrator.knowledge.recipe_kb import kg_client as kgmod

    class ForbiddenGbrain:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("local mode constructed GBrain")

    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "local")
    monkeypatch.setenv("KNOWLEDGE_LOCAL_ROOT", str(tmp_path))
    monkeypatch.setenv("GBRAIN_BASE_URL", "https://ambient.invalid")
    monkeypatch.setenv("GBRAIN_TOKEN", "ambient-secret")
    monkeypatch.setattr(kgmod, "_GbrainMcp", ForbiddenGbrain)

    client = kgmod.build_kg_client_from_env()
    assert client is not None
    assert isinstance(client._mcp, LocalGraphStore)
    assert client._mcp.root == (tmp_path / "hyperloom" / "kg").resolve()


def test_remote_factory_constructs_gbrain_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hyperloom.orchestrator.knowledge.recipe_kb import kg_client as kgmod

    calls: list[tuple[str, str, float]] = []

    class FakeGbrain:
        def __init__(self, base_url: str, token: str, timeout_sec: float) -> None:
            calls.append((base_url, token, timeout_sec))

    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    monkeypatch.setenv("KNOWLEDGE_LOCAL_ROOT", str(tmp_path))
    monkeypatch.setenv("KB_STORE_URL", "https://kb.example")
    monkeypatch.setenv("KB_STORE_TOKEN", "kb-secret")
    monkeypatch.setenv("GBRAIN_BASE_URL", "https://gbrain.example")
    monkeypatch.setenv("GBRAIN_TOKEN", "secret")
    monkeypatch.setattr(kgmod, "_GbrainMcp", FakeGbrain)

    client = kgmod.build_kg_client_from_env()
    assert client is not None
    assert calls == [("https://gbrain.example", "secret", 2.0)]


def test_get_kg_client_singleton_and_reset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.orchestrator.knowledge.recipe_kb import kg_client as kgmod

    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "local")
    monkeypatch.setenv("KNOWLEDGE_LOCAL_ROOT", str(tmp_path))
    kgmod.reset_kg_client()
    first = kgmod.get_kg_client()
    second = kgmod.get_kg_client()
    assert first is second
    kgmod.reset_kg_client()
    assert kgmod.get_kg_client() is not first
    kgmod.reset_kg_client()


def test_get_kg_client_caches_build_failure_and_logs_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from hyperloom.orchestrator.knowledge.recipe_kb import kg_client as kgmod

    calls = 0

    def fail_build() -> None:
        nonlocal calls
        calls += 1
        raise ValueError("bad KG configuration")

    kgmod.reset_kg_client()
    monkeypatch.setattr(kgmod, "build_kg_client_from_env", fail_build)
    with caplog.at_level("WARNING"):
        assert kgmod.get_kg_client() is None
        assert kgmod.get_kg_client() is None
    assert calls == 1
    assert caplog.text.count("knowledge graph is unavailable") == 1

    kgmod.reset_kg_client()
    assert kgmod.get_kg_client() is None
    assert calls == 2
    kgmod.reset_kg_client()


def test_t0_enhancement_reads_local_native_edges(tmp_path: Path) -> None:
    kg = KGClient(LocalGraphStore(tmp_path / "kg"))
    assert kg.emit_fact(
        subject="bad_patch",
        predicate="REVERTED_ON",
        object="qwen3",
        properties={"error": "regression", "confidence": "0.9"},
    )
    assert kg.emit_fact(
        subject="good_knob",
        predicate="IMPROVES",
        object="qwen3",
        properties={"gain": "+12%", "confidence": "0.95", "hw": "mi300x", "fw": "sglang"},
    )
    context: dict[str, Any] = {}
    _enhance_warm_start_with_kg(
        context,
        model_architectures=["qwen3"],
        hardware="mi300x",
        framework="sglang",
        kg_client=kg,
    )
    assert context["blocked_patches"][0]["patch_file"] == "bad_patch"
    assert context["recommended_knobs"][0]["knob"] == "good_knob"


def test_empty_first_run_graph_leaves_t0_context_unchanged(tmp_path: Path) -> None:
    kg = KGClient(LocalGraphStore(tmp_path / "kg"))
    context: dict[str, Any] = {"recipe_result": "still-available"}
    _enhance_warm_start_with_kg(
        context,
        model_architectures=["qwen3"],
        hardware="mi300x",
        framework="sglang",
        kg_client=kg,
    )
    assert context == {"recipe_result": "still-available"}


def test_framework_emit_fact_shape_writes_local_edge(tmp_path: Path) -> None:
    kg = KGClient(LocalGraphStore(tmp_path / "kg"))
    assert kg.emit_fact_safe(
        subject="framework_patch.py",
        predicate="IMPROVES",
        object="Qwen3",
        properties={"gain": "+5.0%", "hw": "mi300x", "fw": "sglang"},
    )
    facts = kg.query_facts_safe(subject="framework_patch.py", predicate="IMPROVES")
    assert len(facts) == 1
    assert facts[0].object == "qwen3"
    assert facts[0].gain == pytest.approx(5.0)
