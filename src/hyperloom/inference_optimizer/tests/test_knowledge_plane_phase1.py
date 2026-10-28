# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 1 local/remote KnowledgePlane contract tests."""

from __future__ import annotations

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


def test_config_defaults_local_under_user_data_path() -> None:
    config = KnowledgeConfig.from_env({"USER_DATA_PATH": "/data/user"})
    assert config.mode is KnowledgeStoreMode.LOCAL
    assert config.local_root == "/data/user/knowledge"


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


def test_remote_config_accepts_kb_store_credentials_and_keeps_optional_gbrain() -> None:
    config = KnowledgeConfig.from_env(
        {
            "KNOWLEDGE_STORE_MODE": "remote",
            "KB_STORE_URL": "https://kb.test",
            "KB_STORE_TOKEN": "kb-token",
            "GBRAIN_BASE_URL": "https://gbrain.test",
            "GBRAIN_TOKEN": "gbrain-token",
        }
    )
    assert config.mode is KnowledgeStoreMode.REMOTE


def test_local_dispatcher_ignores_ambient_remote_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "local")
    monkeypatch.setenv("KNOWLEDGE_LOCAL_ROOT", str(tmp_path / "knowledge"))
    monkeypatch.setenv("KB_STORE_URL", "https://ambient.invalid")
    monkeypatch.setenv("KB_STORE_TOKEN", "ambient-token")
    recipe_kb = cli_kb._build_recipe_kb_dispatcher()
    assert isinstance(recipe_kb.local, LocalRecipeStore)
    assert recipe_kb.mode == "local"
    assert recipe_kb.backend_name == "local-json"


def test_remote_dispatcher_is_none_without_creating_local_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "must-not-exist"
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    monkeypatch.setenv("KNOWLEDGE_LOCAL_ROOT", str(root))
    monkeypatch.setenv("KB_STORE_URL", "https://kb.test")
    monkeypatch.setenv("KB_STORE_TOKEN", "token")
    assert cli_kb._build_recipe_kb_dispatcher() is None
    assert not root.exists()


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
