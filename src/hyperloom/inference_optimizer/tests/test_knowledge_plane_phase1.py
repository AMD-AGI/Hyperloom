# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 1 local/remote KnowledgePlane contract tests."""

from __future__ import annotations

import argparse
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
    assert config.gbrain_base_url == "https://gbrain.test"


def test_apply_child_env_routes_recipe_credentials_by_mode() -> None:
    remote = KnowledgeConfig.from_env(
        {
            "KNOWLEDGE_STORE_MODE": "remote",
            "KB_STORE_URL": "https://kb.test",
            "KB_STORE_TOKEN": "kb-token",
            "GBRAIN_BASE_URL": "https://gbrain.test",
            "GBRAIN_TOKEN": "gbrain-token",
        }
    )
    child: dict[str, str] = {"KERNELFORGE_GBRAIN_ENABLED": "true"}
    remote.apply_to_child_env(child)
    assert child["KB_STORE_URL"] == "https://kb.test"
    assert child["KB_STORE_TOKEN"] == "kb-token"
    assert child["KERNELFORGE_GBRAIN_ENABLED"] == "false"
    assert "GBRAIN_BASE_URL" not in child
    assert "GBRAIN_TOKEN" not in child

    local = KnowledgeConfig.from_env(
        {
            "KNOWLEDGE_STORE_MODE": "local",
            "GBRAIN_BASE_URL": "https://gbrain.test",
            "GBRAIN_TOKEN": "gbrain-token",
        }
    )
    local.apply_to_child_env(child)
    assert "KB_STORE_URL" not in child
    assert "KB_STORE_TOKEN" not in child
    assert child["KERNELFORGE_GBRAIN_ENABLED"] == "false"


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


def test_plane_owns_injected_local_recipe_and_typed_results(
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
    result = plane.read_recipe(canonical_id="inference:m:h:f:mt:a:v:p")
    assert result.hit is False
    assert result.backend == "local-json"
