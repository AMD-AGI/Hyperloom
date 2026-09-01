# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 1 local/remote KnowledgePlane contract tests."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.cli import kb as cli_kb
from hyperloom.orchestrator.knowledge.config import (
    KnowledgeConfig,
    KnowledgeStoreMode,
)
from hyperloom.orchestrator.knowledge.knowledge_plane import KnowledgePlane
from hyperloom.orchestrator.knowledge.recipe_kb import (
    LocalRecipeStore,
    RecipeKB,
)


def _args(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(degraded_kb=False, local_kb_root=None, **kwargs)


def _legacy_recipe(root: Path, *, model: str = "model") -> Path:
    recipe_dir = root / model / "mi300x" / "sglang" / "qwen3" / "qwen3" / "1.0" / "fp8"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "recipe.json").write_text(
        '{"canonical_id":"legacy"}',
        encoding="utf-8",
    )
    (recipe_dir / "attempts.ndjson").write_text('{"id":1}\n', encoding="utf-8")
    (recipe_dir / "metadata.json").write_text('{"safe":true}', encoding="utf-8")
    (recipe_dir / ".lock").write_text("live lock", encoding="utf-8")
    (recipe_dir / ".recipe.json.partial.tmp").write_text(
        "temporary",
        encoding="utf-8",
    )
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
    assert config.backend == "local-json"


@pytest.mark.parametrize("mode", ["REMOTE", "hybrid"])
def test_config_rejects_unknown_mode(mode: str) -> None:
    with pytest.raises(ValueError, match="invalid KNOWLEDGE_STORE_MODE"):
        KnowledgeConfig.from_env({"KNOWLEDGE_STORE_MODE": mode})


@pytest.mark.parametrize(
    ("env", "missing"),
    [
        (
            {
                "KNOWLEDGE_STORE_MODE": "remote",
                "KB_STORE_URL": "https://kb.test",
            },
            "KB_STORE_TOKEN",
        ),
        (
            {
                "KNOWLEDGE_STORE_MODE": "remote",
                "KB_STORE_TOKEN": "token",
            },
            "KB_STORE_URL",
        ),
    ],
)
def test_remote_requires_kb_store_credentials(
    env: dict[str, str],
    missing: str,
) -> None:
    with pytest.raises(ValueError, match=missing):
        KnowledgeConfig.from_env(env)


def test_old_gbrain_credentials_do_not_satisfy_recipe_remote() -> None:
    with pytest.raises(ValueError, match="KB_STORE_URL"):
        KnowledgeConfig.from_env(
            {
                "KNOWLEDGE_STORE_MODE": "remote",
                "GBRAIN_BASE_URL": "https://gbrain.test",
                "GBRAIN_TOKEN": "legacy-token",
            }
        )


def test_remote_config_uses_kb_store_backend_and_keeps_optional_gbrain() -> None:
    config = KnowledgeConfig.from_env(
        {
            "KNOWLEDGE_STORE_MODE": "remote",
            "KB_STORE_URL": "https://kb.test",
            "KB_STORE_TOKEN": "kb-token",
            "GBRAIN_BASE_URL": "https://gbrain.test",
            "GBRAIN_TOKEN": "gbrain-token",
        }
    )
    assert config.backend == "kb-store"
    assert config.kb_store_url == "https://kb.test"


def test_a_remote_child_gets_only_kb_store_credentials() -> None:
    remote = KnowledgeConfig.from_env(
        {
            "KNOWLEDGE_STORE_MODE": "remote",
            "KB_STORE_URL": "https://kb.test",
            "KB_STORE_TOKEN": "kb-token",
            "GBRAIN_BASE_URL": "https://gbrain.test",
            "GBRAIN_TOKEN": "gbrain-token",
        }
    )
    child: dict[str, str] = {}
    remote.apply_to_child_env(child)
    assert child["KB_STORE_URL"] == "https://kb.test"
    assert child["KB_STORE_TOKEN"] == "kb-token"
    assert "GBRAIN_BASE_URL" not in child
    assert "GBRAIN_TOKEN" not in child
    assert child["KERNELFORGE_GBRAIN_ENABLED"] == "false"


def test_a_remote_child_without_gbrain_is_told_so_rather_than_left_guessing() -> None:
    remote = KnowledgeConfig.from_env(
        {
            "KNOWLEDGE_STORE_MODE": "remote",
            "KB_STORE_URL": "https://kb.test",
            "KB_STORE_TOKEN": "kb-token",
        }
    )
    child: dict[str, str] = {
        "KERNELFORGE_GBRAIN_ENABLED": "true",
        "GBRAIN_BASE_URL": "https://ambient.invalid",
        "GBRAIN_TOKEN": "ambient-secret",
    }
    remote.apply_to_child_env(child)
    assert child["KB_STORE_URL"] == "https://kb.test"
    assert child["KERNELFORGE_GBRAIN_ENABLED"] == "false"
    assert "GBRAIN_BASE_URL" not in child
    assert "GBRAIN_TOKEN" not in child


def test_a_local_child_keeps_pr_service_url_without_recipe_token() -> None:
    local = KnowledgeConfig.from_env(
        {
            "KNOWLEDGE_STORE_MODE": "local",
            "KB_STORE_URL": "https://kb.test",
            "KB_STORE_TOKEN": "must-not-cross",
            "GBRAIN_BASE_URL": "https://gbrain.test",
            "GBRAIN_TOKEN": "gbrain-token",
        }
    )
    child: dict[str, str] = {
        "KERNELFORGE_GBRAIN_ENABLED": "true",
        "KB_STORE_URL": "https://kb.test",
        "KB_STORE_TOKEN": "kb-token",
        "GBRAIN_BASE_URL": "https://gbrain.test",
        "GBRAIN_TOKEN": "gbrain-token",
    }
    local.apply_to_child_env(child)
    assert child["KB_STORE_URL"] == "https://kb.test"
    assert "KB_STORE_TOKEN" not in child
    assert "GBRAIN_BASE_URL" not in child
    assert "GBRAIN_TOKEN" not in child
    assert child["KERNELFORGE_GBRAIN_ENABLED"] == "false"


def test_a_kernelforge_child_never_stages_into_the_inference_draft() -> None:
    config = KnowledgeConfig.from_env(
        {
            "KNOWLEDGE_STORE_MODE": "remote",
            "KB_STORE_URL": "https://kb.test",
            "KB_STORE_TOKEN": "kb-token",
        }
    )
    child = {
        "KB_DRAFT_DIR": "/session/runtime/kb_draft",
        "KB_WARM_START_DIR": "/session/runtime/remote_recipe",
    }
    config.apply_to_child_env(child)
    assert "KB_DRAFT_DIR" not in child
    assert "KB_WARM_START_DIR" not in child


def test_local_dispatcher_ignores_ambient_remote_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "local")
    monkeypatch.setenv("KNOWLEDGE_LOCAL_ROOT", str(tmp_path / "knowledge"))
    monkeypatch.setenv("KB_STORE_URL", "https://ambient.invalid")
    monkeypatch.setenv("KB_STORE_TOKEN", "ambient-token")
    recipe_kb = cli_kb._build_recipe_kb_dispatcher(_args())
    assert isinstance(recipe_kb.local, LocalRecipeStore)
    assert recipe_kb.mode == "local"


def test_user_data_legacy_recipe_corpus_migrates_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_root_selection(monkeypatch)
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    source = tmp_path / "kb"
    legacy_dir = _legacy_recipe(source)

    kb = cli_kb._build_recipe_kb_dispatcher(_args())

    destination = tmp_path / "knowledge"
    migrated = destination / legacy_dir.relative_to(source)
    assert kb.local.root == destination
    assert (migrated / "recipe.json").is_file()
    assert (migrated / "history" / "v1.json").is_file()
    assert (migrated / "attempts.ndjson").is_file()
    assert (migrated / "metadata.json").is_file()
    assert not (migrated / ".lock").exists()
    assert not (migrated / ".recipe.json.partial.tmp").exists()
    assert (destination / cli_kb._RECIPE_MIGRATION_MARKER).is_file()

    shutil.rmtree(migrated)
    _legacy_recipe(source, model="second")
    assert (
        cli_kb._migrate_legacy_recipe_kb_once(
            destination=destination,
            source=source,
        )
        is False
    )
    assert not list(destination.rglob("recipe.json"))


def test_workspace_legacy_source_is_injectable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _clear_root_selection(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    source = tmp_path / "workspace-kb"
    _legacy_recipe(source)
    monkeypatch.setattr(cli_kb, "_LEGACY_WORKSPACE_KB_ROOT", source)

    kb = cli_kb._build_recipe_kb_dispatcher(_args())

    assert kb.local.root == (tmp_path / "home" / ".cache" / "hyperloom" / "knowledge")
    assert len(list(kb.local.root.rglob("recipe.json"))) == 1


def test_existing_destination_recipe_skips_legacy_migration(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "knowledge"
    source = tmp_path / "kb"
    existing = _legacy_recipe(destination, model="existing")
    _legacy_recipe(source, model="legacy")

    assert (
        cli_kb._migrate_legacy_recipe_kb_once(
            destination=destination,
            source=source,
        )
        is False
    )
    assert existing.joinpath("recipe.json").is_file()
    assert not (destination / cli_kb._RECIPE_MIGRATION_MARKER).exists()
    assert len(list(destination.rglob("recipe.json"))) == 1


def test_absent_legacy_source_does_not_create_destination(
    tmp_path: Path,
) -> None:
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


def test_legacy_migration_failure_aborts_startup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "kb"
    destination = tmp_path / "knowledge"
    _legacy_recipe(source)
    monkeypatch.setattr(
        cli_kb,
        "_copy_recipe_corpus",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("injected copy failure")),
    )

    with pytest.raises(RuntimeError, match="migration.*failed.*copy failure"):
        cli_kb._migrate_legacy_recipe_kb_once(
            destination=destination,
            source=source,
        )
    assert not (destination / cli_kb._RECIPE_MIGRATION_MARKER).exists()


def test_legacy_migration_marker_failure_rolls_back_corpus(
    monkeypatch: pytest.MonkeyPatch,
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
        cli_kb._migrate_legacy_recipe_kb_once(
            destination=destination,
            source=source,
        )
    assert not list(destination.rglob("recipe.json"))
    assert not (destination / cli_kb._RECIPE_MIGRATION_MARKER).exists()


def test_legacy_migration_interrupt_rolls_back_partial_file(
    monkeypatch: pytest.MonkeyPatch,
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
        cli_kb._migrate_legacy_recipe_kb_once(
            destination=destination,
            source=source,
        )
    assert not list(destination.rglob("recipe.json"))
    assert not list(destination.rglob("attempts.ndjson"))
    assert not (destination / cli_kb._RECIPE_MIGRATION_MARKER).exists()


def test_remote_dispatcher_is_none_without_creating_local_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "must-not-exist"
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    monkeypatch.setenv("KNOWLEDGE_LOCAL_ROOT", str(root))
    monkeypatch.setenv("KB_STORE_URL", "https://kb.test")
    monkeypatch.setenv("KB_STORE_TOKEN", "token")
    assert cli_kb._build_recipe_kb_dispatcher(_args()) is None
    assert not root.exists()


def test_remote_plane_reports_close_writer_enabled() -> None:
    config = KnowledgeConfig.from_env(
        {
            "KNOWLEDGE_STORE_MODE": "remote",
            "KB_STORE_URL": "https://kb.test",
            "KB_STORE_TOKEN": "token",
        }
    )
    plane = KnowledgePlane.from_clients(recipe_kb=None, config=config)
    assert plane.status["recipe"]["enabled"] is True
    assert plane.status["recipe"]["read_enabled"] is False


def test_plane_owns_injected_local_recipe(
    tmp_path: Path,
) -> None:
    config = KnowledgeConfig.from_env(
        {
            "KNOWLEDGE_STORE_MODE": "local",
            "KNOWLEDGE_LOCAL_ROOT": str(tmp_path),
        }
    )
    recipe_kb = RecipeKB(local=LocalRecipeStore(tmp_path))
    plane = KnowledgePlane.from_clients(recipe_kb=recipe_kb, config=config)
    assert plane.recipe_kb is recipe_kb
    assert plane.status["recipe"]["backend"] == "local-json"
