# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 1 local/remote KnowledgePlane contract tests (no live calls)."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import threading
import time
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.cli import kb as cli_kb
from hyperloom.orchestrator.knowledge.config import (
    KnowledgeConfig,
    KnowledgeStoreMode,
)
from hyperloom.orchestrator.knowledge.knowledge_plane import KnowledgePlane
from hyperloom.orchestrator.knowledge.recipe_kb import LocalRecipeStore, RecipeKB
from hyperloom.orchestrator.knowledge.recipe_kb.gbrain_remote_client import (
    GbrainRemoteError,
    GbrainRemoteRecipeClient,
)
from hyperloom.orchestrator.knowledge.recipe_kb.gbrain_store import (
    GbrainRecipeLockError,
    GbrainRecipeStore,
)
from hyperloom.orchestrator.knowledge.recipe_kb.replay_bundle import (
    refresh_bundle_digest,
)


def _args(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(degraded_kb=False, local_kb_root=None, **kwargs)


def _legacy_recipe(root: Path, *, model: str = "model") -> Path:
    recipe_dir = root / model / "mi300x" / "sglang" / "qwen3" / "qwen3" / "1.0" / "fp8"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "recipe.json").write_text('{"canonical_id":"legacy"}', encoding="utf-8")
    (recipe_dir / "attempts.ndjson").write_text('{"id":1}\n', encoding="utf-8")
    (recipe_dir / "metadata.json").write_text('{"safe":true}', encoding="utf-8")
    (recipe_dir / ".lock").write_text("live lock", encoding="utf-8")
    (recipe_dir / ".recipe.json.partial.tmp").write_text("temporary", encoding="utf-8")
    history = recipe_dir / "history"
    history.mkdir()
    (history / "v1.json").write_text('{"version":1}', encoding="utf-8")
    return recipe_dir


def _clear_root_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "KNOWLEDGE_STORE_MODE",
        "KNOWLEDGE_LOCAL_ROOT",
        "HYPERLOOM_LOCAL_KB_ROOT",
        "USER_DATA_PATH",
    ):
        monkeypatch.delenv(name, raising=False)


def test_config_defaults_local_under_user_data_path() -> None:
    config = KnowledgeConfig.from_env({"USER_DATA_PATH": "/data/user"})
    assert config.mode is KnowledgeStoreMode.LOCAL
    assert config.local_root == "/data/user/knowledge"


def test_config_default_cache_and_explicit_root(monkeypatch) -> None:
    monkeypatch.setenv("HOME", "/home/tester")
    assert KnowledgeConfig.from_env({}).local_root.endswith(".cache/hyperloom/knowledge")
    config = KnowledgeConfig.from_env(
        {
            "KNOWLEDGE_STORE_MODE": "local",
            "KNOWLEDGE_LOCAL_ROOT": "relative/root",
        }
    )
    assert config.local_root == "relative/root"


@pytest.mark.parametrize("mode", ["REMOTE", "hybrid", ""])
def test_config_rejects_unknown_or_incomplete_mode(mode: str) -> None:
    env = {"KNOWLEDGE_STORE_MODE": mode}
    if mode == "":
        assert KnowledgeConfig.from_env(env).mode is KnowledgeStoreMode.LOCAL
    else:
        with pytest.raises(ValueError, match="invalid KNOWLEDGE_STORE_MODE"):
            KnowledgeConfig.from_env(env)


def test_remote_requires_both_credentials() -> None:
    with pytest.raises(ValueError, match="GBRAIN_TOKEN"):
        KnowledgeConfig.from_env(
            {
                "KNOWLEDGE_STORE_MODE": "remote",
                "GBRAIN_BASE_URL": "https://gbrain.test",
            }
        )


def test_local_mode_ignores_ambient_gbrain(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "local")
    monkeypatch.setenv("KNOWLEDGE_LOCAL_ROOT", str(tmp_path / "knowledge"))
    monkeypatch.setenv("GBRAIN_BASE_URL", "https://ambient.invalid")
    monkeypatch.setenv("GBRAIN_TOKEN", "ambient-secret")

    monkeypatch.setattr(
        GbrainRecipeStore,
        "from_credentials",
        classmethod(lambda cls, **kwargs: pytest.fail("local mode constructed GBrain")),
    )
    recipe_kb = cli_kb._build_recipe_kb_dispatcher(_args())
    assert isinstance(recipe_kb.local, LocalRecipeStore)
    assert recipe_kb.remote is None


def test_user_data_legacy_recipe_corpus_migrates_once(monkeypatch, tmp_path: Path) -> None:
    _clear_root_selection(monkeypatch)
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    source = tmp_path / "kb"
    legacy_dir = _legacy_recipe(source)

    kb = cli_kb._build_recipe_kb_dispatcher(_args())

    destination = tmp_path / "knowledge"
    relative = legacy_dir.relative_to(source)
    migrated = destination / relative
    assert kb.local.root == destination
    assert (migrated / "recipe.json").is_file()
    assert (migrated / "history" / "v1.json").is_file()
    assert (migrated / "attempts.ndjson").is_file()
    assert (migrated / "metadata.json").is_file()
    assert not (migrated / ".lock").exists()
    assert not (migrated / ".recipe.json.partial.tmp").exists()
    marker = destination / cli_kb._RECIPE_MIGRATION_MARKER
    assert marker.is_file()

    shutil.rmtree(migrated)
    _legacy_recipe(source, model="second")
    assert cli_kb._migrate_legacy_recipe_kb_once(destination=destination, source=source) is False
    assert not list(destination.rglob("recipe.json"))


def test_workspace_legacy_source_is_injectable(monkeypatch, tmp_path: Path) -> None:
    _clear_root_selection(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    source = tmp_path / "workspace-kb"
    _legacy_recipe(source)
    monkeypatch.setattr(cli_kb, "_LEGACY_WORKSPACE_KB_ROOT", source)

    kb = cli_kb._build_recipe_kb_dispatcher(_args())

    assert kb.local.root == tmp_path / "home" / ".cache" / "hyperloom" / "knowledge"
    assert len(list(kb.local.root.rglob("recipe.json"))) == 1


def test_existing_destination_recipe_skips_legacy_migration(monkeypatch, tmp_path: Path) -> None:
    destination = tmp_path / "knowledge"
    source = tmp_path / "kb"
    existing = _legacy_recipe(destination, model="existing")
    _legacy_recipe(source, model="legacy")

    assert cli_kb._migrate_legacy_recipe_kb_once(destination=destination, source=source) is False
    assert existing.joinpath("recipe.json").is_file()
    assert not (destination / cli_kb._RECIPE_MIGRATION_MARKER).exists()
    assert len(list(destination.rglob("recipe.json"))) == 1


def test_absent_legacy_source_does_not_create_destination(tmp_path: Path) -> None:
    destination = tmp_path / "knowledge"

    assert (
        cli_kb._migrate_legacy_recipe_kb_once(
            destination=destination,
            source=tmp_path / "missing-kb",
        )
        is False
    )
    assert not destination.exists()


@pytest.mark.parametrize("selection", ["flag", "compat-env", "knowledge-env"])
def test_explicit_local_root_skips_migration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    selection: str,
) -> None:
    _clear_root_selection(monkeypatch)
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    _legacy_recipe(tmp_path / "kb")
    args = _args()
    selected = tmp_path / "selected"
    if selection == "flag":
        args.local_kb_root = str(selected)
    elif selection == "compat-env":
        monkeypatch.setenv("HYPERLOOM_LOCAL_KB_ROOT", str(selected))
    else:
        monkeypatch.setenv("KNOWLEDGE_LOCAL_ROOT", str(selected))
    monkeypatch.setattr(
        cli_kb,
        "_migrate_legacy_recipe_kb_once",
        lambda **_kwargs: pytest.fail("explicit root must not migrate"),
    )

    kb = cli_kb._build_recipe_kb_dispatcher(args)

    assert kb.local.root == selected


def test_legacy_migration_failure_aborts_startup(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "kb"
    destination = tmp_path / "knowledge"
    _legacy_recipe(source)
    monkeypatch.setattr(
        cli_kb,
        "_copy_recipe_corpus",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("injected copy failure")),
    )

    with pytest.raises(RuntimeError, match="migration.*failed.*injected copy failure"):
        cli_kb._migrate_legacy_recipe_kb_once(destination=destination, source=source)
    assert not (destination / cli_kb._RECIPE_MIGRATION_MARKER).exists()


def test_legacy_migration_marker_failure_rolls_back_copied_corpus(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "kb"
    destination = tmp_path / "knowledge"
    _legacy_recipe(source)
    monkeypatch.setattr(
        cli_kb,
        "atomic_write_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("marker fsync failed")),
    )

    with pytest.raises(RuntimeError, match="marker fsync failed"):
        cli_kb._migrate_legacy_recipe_kb_once(destination=destination, source=source)
    assert not list(destination.rglob("recipe.json"))
    assert not (destination / cli_kb._RECIPE_MIGRATION_MARKER).exists()


def test_legacy_migration_copy_interrupt_rolls_back_partial_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "kb"
    destination = tmp_path / "knowledge"
    _legacy_recipe(source)
    monkeypatch.setattr(
        cli_kb.shutil,
        "copyfileobj",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        cli_kb._migrate_legacy_recipe_kb_once(destination=destination, source=source)
    assert not list(destination.rglob("recipe.json"))
    assert not list(destination.rglob("attempts.ndjson"))
    assert not (destination / cli_kb._RECIPE_MIGRATION_MARKER).exists()


def test_legacy_migration_marker_interrupt_rolls_back_completed_copy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "kb"
    destination = tmp_path / "knowledge"
    _legacy_recipe(source)
    monkeypatch.setattr(
        cli_kb,
        "atomic_write_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        cli_kb._migrate_legacy_recipe_kb_once(destination=destination, source=source)
    assert not list(destination.rglob("recipe.json"))
    assert not (destination / cli_kb._RECIPE_MIGRATION_MARKER).exists()


def test_remote_build_does_not_create_local_root(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "must-not-exist"
    captured: dict = {}
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    monkeypatch.setenv("KNOWLEDGE_LOCAL_ROOT", str(root))
    monkeypatch.setenv("GBRAIN_BASE_URL", "https://gbrain.test")
    monkeypatch.setenv("GBRAIN_TOKEN", "token")

    class _RemoteStore:
        backend_name = "gbrain"

    monkeypatch.setattr(
        GbrainRecipeStore,
        "from_credentials",
        classmethod(lambda cls, **kwargs: captured.update(kwargs) or _RemoteStore()),
    )
    recipe_kb = cli_kb._build_recipe_kb_dispatcher(_args())
    assert recipe_kb.mode == "remote"
    assert captured["lock_root"] == root / ".remote-locks" / "recipes"
    assert not root.exists()


class _PageMcp:
    def __init__(self, *, fail_put: bool = False) -> None:
        self.frontmatter: dict = {}
        self.fail_put = fail_put

    def call(self, tool: str, arguments: dict):
        if tool == "put_page":
            if self.fail_put:
                raise RuntimeError("write failed")
            import yaml

            content = arguments["content"]
            self.frontmatter = yaml.safe_load(content.split("---", 2)[1])
            return {"ok": True}
        if tool == "get_page":
            return {"frontmatter": self.frontmatter} if self.frontmatter else {}
        if tool in {"search", "list_pages"}:
            return []
        raise AssertionError(tool)


def _remote_store(mcp: _PageMcp, lock_root: Path) -> GbrainRecipeStore:
    client = GbrainRemoteRecipeClient(
        base_url="https://gbrain.test",
        token="token",
        enabled=True,
    )
    client._mcp = mcp
    return GbrainRecipeStore(client=client, mcp=mcp, lock_root=lock_root)


def _remote_kb(
    mcp: _PageMcp,
    lock_root: Path,
    audits: list[dict] | None = None,
) -> RecipeKB:
    store = _remote_store(mcp, lock_root)
    return RecipeKB(
        local=store,
        mode="remote",
        backend_name="gbrain",
        audit_hook=(audits.append if audits is not None else None),
    )


def test_direct_remote_put_read_round_trip_and_no_local_files(tmp_path: Path) -> None:
    lock_root = tmp_path / ".remote-locks" / "recipes"
    kb = _remote_kb(_PageMcp(), lock_root)
    cid = "inference:model:mi300x:sglang:llama:llama:1.0:fp8"
    result = kb.put_recipe(
        canonical_id=cid,
        model="model",
        hardware="mi300x",
        framework_name="sglang",
        framework_version="1.0",
        precision="fp8",
        lessons=[{"statement": "keep it", "measured_impact": "1%"}],
        sessions=[{"session_id": "s1", "gain_pct": 1.0}],
        extras={"model_type": "llama", "architectures": ["Llama"]},
    )
    row = kb.get_recipe(canonical_id=cid)
    assert result["created"] is True
    assert row is not None
    assert row["lessons"][0]["statement"] == "keep it"
    assert row["sessions"][0]["session_id"] == "s1"
    assert result["write_safety"]["latest_read"] == "inside_lock"
    assert [path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()] == [
        Path(".remote-locks/recipes") / f"{hashlib.sha256(cid.encode('utf-8')).hexdigest()}.lock"
    ]
    assert not list(tmp_path.rglob("recipe.json"))
    assert not list(tmp_path.rglob("history"))
    assert not list(tmp_path.rglob("attempts.ndjson"))


def test_remote_failure_is_observable_and_audited(tmp_path: Path) -> None:
    audits: list[dict] = []
    kb = _remote_kb(_PageMcp(fail_put=True), tmp_path / "locks", audits)
    with pytest.raises(RuntimeError, match="write failed"):
        kb.put_recipe(
            canonical_id="inference:m:h:f:mt:a:v:p",
            model="m",
            hardware="h",
        )
    assert audits[-1]["success"] is False
    assert audits[-1]["mode"] == "remote"
    assert audits[-1]["backend"] == "gbrain"


class _FailingSelectedRemoteStore:
    backend_name = "gbrain"
    enabled = True

    def __init__(self) -> None:
        self.closed = False

    def get_recipe(self, **_kwargs):
        raise GbrainRemoteError("injected selected-store read failure")

    def search(self, **_kwargs):
        raise GbrainRemoteError("injected selected-store search failure")

    def close(self) -> None:
        self.closed = True


def test_selected_remote_read_failures_degrade_and_emit_audit() -> None:
    audits: list[dict] = []
    failures: list[tuple[str, Exception]] = []
    store = _FailingSelectedRemoteStore()
    kb = RecipeKB(
        local=store,
        mode="remote",
        backend_name="gbrain",
        audit_hook=audits.append,
        on_remote_failure=lambda method, exc: failures.append((method, exc)),
    )

    assert kb.get_recipe(canonical_id="inference:m:h:f:mt:a:v:p") is None
    assert kb.search(label_match={"model": "m"}) == []

    assert [method for method, _exc in failures] == ["get_recipe", "search"]
    assert [event["resolution"] for event in audits] == [
        "remote_error",
        "remote_error",
    ]
    assert all(event["mode"] == "remote" and event["hit"] is False for event in audits)


def test_close_releases_selected_remote_store() -> None:
    store = _FailingSelectedRemoteStore()
    kb = RecipeKB(local=store, mode="remote", backend_name="gbrain")

    kb.close()

    assert store.closed is True


def test_remote_store_champion_merge_is_monotonic_and_paired(tmp_path: Path) -> None:
    mcp = _PageMcp()
    store = _remote_store(mcp, tmp_path / "locks")
    cid = "inference:m:h:f:mt:a:v:p"
    bundle_a = refresh_bundle_digest(
        {
            "schema_version": 1,
            "replayable": True,
            "config": {"argv": ["--champion"], "extra_envs": {}},
            "source_artifacts": [],
            "measurement": {"optimized_throughput": 100.0},
        }
    )
    store.put_recipe(
        canonical_id=cid,
        best_config={"extra_server_args": "--champion"},
        best_throughput=100.0,
        extras={"replay_bundle": bundle_a},
    )

    store.put_recipe(
        canonical_id=cid,
        best_config={"extra_server_args": "--equal"},
        best_throughput=100.0,
    )
    row = store.get_recipe(canonical_id=cid)
    assert row is not None
    assert row["best_config"] == {"extra_server_args": "--champion"}

    rejected = store.put_recipe(
        canonical_id=cid,
        best_config={"extra_server_args": "--unbound-higher"},
        best_throughput=110.0,
    )
    row = store.get_recipe(canonical_id=cid)
    assert row is not None
    assert row["best_config"] == {"extra_server_args": "--champion"}
    assert row["best_throughput"] == 100.0
    assert row["replay_bundle"]["bundle_sha256"] == bundle_a["bundle_sha256"]
    assert rejected["write_safety"]["champion"] == "incoming_rejected_unbound_bundle"

    lower = store.put_recipe(
        canonical_id=cid,
        best_config={"extra_server_args": "--late-lower"},
        best_throughput=90.0,
        extras={
            "replay_bundle": refresh_bundle_digest(
                {
                    **bundle_a,
                    "measurement": {"optimized_throughput": 90.0},
                }
            )
        },
    )
    row = store.get_recipe(canonical_id=cid)
    assert row is not None
    assert row["best_config"] == {"extra_server_args": "--champion"}
    assert row["best_throughput"] == 100.0
    assert row["replay_bundle"]["bundle_sha256"] == bundle_a["bundle_sha256"]
    assert lower["write_safety"]["champion"] == "latest_preserved"

    store.put_recipe(canonical_id=cid, best_config={}, best_throughput=200.0)
    row = store.get_recipe(canonical_id=cid)
    assert row is not None
    assert row["best_config"] == {"extra_server_args": "--champion"}
    assert row["best_throughput"] == 100.0

    higher = store.put_recipe(
        canonical_id=cid,
        best_config={"extra_server_args": "--higher"},
        best_throughput=101.0,
        extras={
            "replay_bundle": refresh_bundle_digest(
                {
                    **bundle_a,
                    "config": {"argv": ["--higher"], "extra_envs": {}},
                    "measurement": {"optimized_throughput": 101.0},
                }
            )
        },
    )
    row = store.get_recipe(canonical_id=cid)
    assert row is not None
    assert row["best_config"] == {"extra_server_args": "--higher"}
    assert row["best_throughput"] == 101.0
    assert row["replay_bundle"]["bundle_sha256"] != bundle_a["bundle_sha256"]
    assert higher["write_safety"]["champion"] == "incoming_higher_throughput"

    empty_cid = "inference:m:h:f:mt:a:v:fp16"
    store.put_recipe(canonical_id=empty_cid, best_config={}, best_throughput=500.0)
    filled = store.put_recipe(
        canonical_id=empty_cid,
        best_config={"extra_server_args": "--first-real-config"},
        best_throughput=5.0,
    )
    row = store.get_recipe(canonical_id=empty_cid)
    assert row is not None
    assert row["best_config"] == {"extra_server_args": "--first-real-config"}
    assert row["best_throughput"] == 5.0
    assert filled["write_safety"]["champion"] == "incoming_filled_empty"


def test_local_store_rejects_champion_advance_with_stale_bundle(tmp_path: Path) -> None:
    store = LocalRecipeStore(tmp_path / "local")
    cid = "inference:m:h:f:mt:a:v:p"
    bundle = refresh_bundle_digest(
        {
            "schema_version": 1,
            "replayable": True,
            "config": {"argv": ["--old"], "extra_envs": {}},
            "source_artifacts": [],
            "measurement": {"optimized_throughput": 100.0},
        }
    )
    store.put_recipe(
        canonical_id=cid,
        best_config={"extra_server_args": "--old"},
        best_throughput=100.0,
        extras={"replay_bundle": bundle},
    )
    store.put_recipe(
        canonical_id=cid,
        best_config={"extra_server_args": "--new"},
        best_throughput=200.0,
        # This is the exact stale-extras shape produced by a mid-session amend.
        extras={"replay_bundle": bundle},
    )
    row = store.get_recipe(canonical_id=cid)
    assert row is not None
    assert row["best_config"] == {"extra_server_args": "--old"}
    assert row["best_throughput"] == 100.0
    assert row["replay_bundle"]["bundle_sha256"] == bundle["bundle_sha256"]


def test_remote_store_stale_collections_sessions_and_mappings_merge(tmp_path: Path) -> None:
    store = _remote_store(_PageMcp(), tmp_path / "locks")
    cid = "inference:m:h:f:mt:a:v:p"
    worked_a = {"description": "latest", "measured_impact": "1%"}
    worked_b = {"description": "incoming", "measured_impact": "2%"}
    store.put_recipe(
        canonical_id=cid,
        model="model",
        hardware="gpu",
        framework_name="framework",
        framework_version="1",
        precision="fp8",
        best_config={"extra_server_args": "--best"},
        best_throughput=10.0,
        what_worked=[worked_a],
        what_failed=[{"description": "failure-a", "reason": "reason-a"}],
        remaining_gaps=[{"description": "gap-a", "metrics": "metric-a"}],
        prs_tested=[{"repo": "repo", "number": 1, "outcome": "kept"}],
        pitfalls=[{"description": "pitfall-a", "severity": "high"}],
        lessons=[{"statement": "lesson-a", "measured_impact": "1%"}],
        last_profiled="2026-08-01T00:00:00Z",
        sessions=[{"session_id": "s1", "gain_pct": 2.0}],
        stack_fingerprint={"rocm_version": "6.3"},
        evidence_refs=[{"url": "a", "kind": "run"}],
        extras={"model_type": "llama", "architectures": ["Llama"], "custom": "latest"},
    )
    result = store.put_recipe(
        canonical_id=cid,
        what_worked=[worked_a, worked_b],
        what_failed=[{"description": "failure-b", "reason": "reason-b"}],
        remaining_gaps=[{"description": "gap-b", "metrics": "metric-b"}],
        prs_tested=[{"repo": "repo", "number": 2, "outcome": "reverted"}],
        pitfalls=[{"description": "pitfall-b", "severity": "low"}],
        lessons=[{"statement": "lesson-b", "measured_impact": "2%"}],
        sessions=[
            {"session_id": "s1", "gain_pct": 3.0},
            {"session_id": "s2", "gain_pct": 4.0},
        ],
        stack_fingerprint={"rocm_version": "", "aiter_commit": "abc"},
        evidence_refs=[
            {"kind": "run", "url": "a"},
            {"url": "b", "kind": "run"},
        ],
        extras={"custom": "", "new_extra": "incoming"},
    )
    row = store.get_recipe(canonical_id=cid)
    assert row is not None
    assert result["version"] == 2
    assert row["created_at"]
    assert row["model"] == "model"
    assert row["what_worked"] == [worked_a, worked_b]
    assert [item["description"] for item in row["what_failed"]] == ["failure-a", "failure-b"]
    assert [item["description"] for item in row["remaining_gaps"]] == ["gap-a", "gap-b"]
    assert [item["number"] for item in row["prs_tested"]] == [1, 2]
    assert [item["description"] for item in row["pitfalls"]] == ["pitfall-a", "pitfall-b"]
    assert [item["statement"] for item in row["lessons"]] == ["lesson-a", "lesson-b"]
    assert [(item["session_id"], item["gain_pct"]) for item in row["sessions"]] == [
        ("s1", 3.0),
        ("s2", 4.0),
    ]
    assert row["stack_fingerprint"]["rocm_version"] == "6.3"
    assert row["stack_fingerprint"]["aiter_commit"] == "abc"
    assert row["evidence_refs"] == [
        {"url": "a", "kind": "run"},
        {"url": "b", "kind": "run"},
    ]
    assert row["custom"] == "latest"
    assert row["new_extra"] == "incoming"
    assert row["last_profiled"] == "2026-08-01T00:00:00Z"


class _BlockingPageMcp(_PageMcp):
    def __init__(self) -> None:
        super().__init__()
        self.first_put_entered = threading.Event()
        self.release_first_put = threading.Event()
        self.get_count = 0
        self._blocked_once = False
        self._guard = threading.Lock()

    def call(self, tool: str, arguments: dict):
        if tool == "get_page":
            with self._guard:
                self.get_count += 1
        if tool == "put_page":
            with self._guard:
                should_block = not self._blocked_once
                if should_block:
                    self._blocked_once = True
            if should_block:
                self.first_put_entered.set()
                if not self.release_first_put.wait(timeout=3):
                    raise TimeoutError("test did not release first put")
        return super().call(tool, arguments)


def test_two_remote_store_instances_serialize_latest_read_and_put(tmp_path: Path) -> None:
    mcp = _BlockingPageMcp()
    lock_root = tmp_path / "locks"
    first = _remote_store(mcp, lock_root)
    second = _remote_store(mcp, lock_root)
    cid = "inference:m:h:f:mt:a:v:p"
    errors: list[Exception] = []

    def _write(store: GbrainRecipeStore, statement: str) -> None:
        try:
            store.put_recipe(
                canonical_id=cid,
                lessons=[{"statement": statement, "measured_impact": ""}],
            )
        except Exception as exc:  # noqa: BLE001 - surfaced in test thread
            errors.append(exc)

    first_thread = threading.Thread(target=_write, args=(first, "first"))
    second_thread = threading.Thread(target=_write, args=(second, "second"))
    first_thread.start()
    assert mcp.first_put_entered.wait(timeout=2)
    reads_before_second_writer = mcp.get_count
    second_thread.start()
    time.sleep(0.1)
    assert second_thread.is_alive()
    assert mcp.get_count == reads_before_second_writer
    mcp.release_first_put.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert errors == []
    row = first.get_recipe(canonical_id=cid)
    assert row is not None
    assert [item["statement"] for item in row["lessons"]] == ["first", "second"]
    assert row["version"] == 2


def test_remote_put_exception_releases_lock(tmp_path: Path) -> None:
    mcp = _PageMcp(fail_put=True)
    lock_root = tmp_path / "locks"
    first = _remote_store(mcp, lock_root)
    second = _remote_store(mcp, lock_root)
    cid = "inference:m:h:f:mt:a:v:p"
    with pytest.raises(RuntimeError, match="write failed"):
        first.put_recipe(canonical_id=cid)
    mcp.fail_put = False
    assert second.put_recipe(canonical_id=cid)["version"] == 1


def test_remote_store_rejects_missing_or_unavailable_locking(monkeypatch, tmp_path: Path) -> None:
    mcp = _PageMcp()
    client = GbrainRemoteRecipeClient(
        base_url="https://gbrain.test",
        token="token",
        enabled=True,
    )
    client._mcp = mcp
    with pytest.raises(ValueError, match="requires lock_root"):
        GbrainRecipeStore(client=client, mcp=mcp)

    store_module = sys.modules[GbrainRecipeStore.__module__]
    monkeypatch.setattr(store_module, "_fcntl", None)
    with pytest.raises(GbrainRecipeLockError, match="requires POSIX fcntl"):
        GbrainRecipeStore(client=client, mcp=mcp, lock_root=tmp_path / "locks")


def test_remote_store_reports_unsafe_lock_filesystem_without_path(tmp_path: Path) -> None:
    lock_root = tmp_path / "not-a-directory"
    lock_root.write_text("unsafe", encoding="utf-8")
    store = _remote_store(_PageMcp(), lock_root)
    with pytest.raises(GbrainRecipeLockError) as exc_info:
        store.put_recipe(canonical_id="inference:m:h:f:mt:a:v:p")
    assert "shared read-write POSIX-lock-capable filesystem" in str(exc_info.value)
    assert str(tmp_path) not in str(exc_info.value)


def test_remote_write_audit_includes_bounded_lock_merge_provenance(tmp_path: Path) -> None:
    audits: list[dict] = []
    kb = _remote_kb(_PageMcp(), tmp_path / "locks", audits)
    kb.put_recipe(canonical_id="inference:m:h:f:mt:a:v:p")
    safety = audits[-1]["result"]["write_safety"]
    assert safety == {
        "lock": "thread+posix-flock",
        "latest_read": "inside_lock",
        "merge": "latest_then_incoming",
        "champion": "incoming_new",
    }
    assert str(tmp_path) not in str(audits[-1])


def test_remote_t0_warm_replay_seed_parity(tmp_path: Path) -> None:
    from hyperloom.inference_optimizer.tests.test_recipe_kb_t0_anchor import (
        _FakeSharedState,
    )
    from hyperloom.orchestrator.knowledge.recipe_kb import recipe_canonical_id
    from hyperloom.orchestrator.knowledge.recipe_kb_t0 import run_t0_anchor

    kb = _remote_kb(_PageMcp(), tmp_path / "locks")
    state = _FakeSharedState()
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    run_t0_anchor(
        kb,
        state,
        workload="model",
        hw="mi300x",
        extra_attrs={"framework_name": "sglang"},
        session_dir=session_dir,
    )
    cid = recipe_canonical_id(
        model="model",
        hardware="mi300x",
        framework_name="sglang",
        framework_version=state.framework_version,
        precision=state.precision,
    )
    assert kb.get_recipe(canonical_id=cid) is not None
    assert state.warm_start_context["status"] == "seed_only"
    assert state.warm_start_context["match"]["source"] == "gbrain"


def test_plane_owns_injected_recipe_and_typed_results(tmp_path: Path) -> None:
    config = KnowledgeConfig.from_env(
        {
            "KNOWLEDGE_STORE_MODE": "local",
            "KNOWLEDGE_LOCAL_ROOT": str(tmp_path),
        }
    )
    recipe_kb = RecipeKB(
        local=LocalRecipeStore(tmp_path),
        mode="local",
        backend_name="local-json",
    )
    plane = KnowledgePlane.from_clients(recipe_kb=recipe_kb, config=config)
    assert plane.recipe_kb is recipe_kb
    assert plane.status["recipe"]["backend"] == "local-json"
    result = plane.read_recipe(canonical_id="inference:m:h:f:mt:a:v:p")
    assert result.hit is False
    assert result.backend == "local-json"


def test_coordinator_recipe_compatibility_alias_comes_from_plane(tmp_path: Path) -> None:
    from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.roles import (
        MockBackend,
        MockCriticBackend,
        MockRobustnessBackend,
        ScriptedPlan,
    )

    intent = Intent(
        type=IntentType.SEND_MESSAGE,
        payload={"topic": "heartbeat", "body_md": "ok"},
    )
    plan = ScriptedPlan(turns=[], default_intent=intent)
    backends = {
        "orchestration": MockBackend(plan, name="o"),
        "kernel_agent": MockBackend(plan, name="k"),
        "critic": MockCriticBackend(),
        "robustness": MockRobustnessBackend(),
    }
    config = KnowledgeConfig.from_env(
        {
            "KNOWLEDGE_STORE_MODE": "local",
            "KNOWLEDGE_LOCAL_ROOT": str(tmp_path / "knowledge"),
        }
    )
    recipe_kb = RecipeKB(
        local=LocalRecipeStore(tmp_path / "knowledge"),
        mode="local",
        backend_name="local-json",
    )
    plane = KnowledgePlane.from_clients(recipe_kb=recipe_kb, config=config)
    coordinator = Coordinator(
        tmp_path / "session",
        backends=backends,
        recipe_kb=None,
        knowledge_plane=plane,
    )
    assert coordinator.recipe_kb is plane.recipe_kb
