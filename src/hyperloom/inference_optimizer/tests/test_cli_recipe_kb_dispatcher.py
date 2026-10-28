# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the recipe-snapshot KB dispatcher bootstrap helpers in cli."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inference_optimizer.cli.kb import _build_recipe_kb_dispatcher
from hyperloom.orchestrator.knowledge.config import KnowledgeConfig
from hyperloom.orchestrator.knowledge.recipe_kb import (
    LocalRecipeStore,
    RecipeKB,
)


@pytest.fixture
def env_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wipe the env vars these helpers consult so each test's precedence tier is explicit."""
    for key in (
        "USER_DATA_PATH",
        "GBRAIN_BASE_URL",
        "GBRAIN_TOKEN",
        "KB_STORE_URL",
        "KB_STORE_TOKEN",
        "KNOWLEDGE_LOCAL_ROOT",
        "KNOWLEDGE_STORE_MODE",
    ):
        monkeypatch.delenv(key, raising=False)


def test_default_local_root_is_user_data_path_knowledge(
    env_clean: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Shared default: ``$USER_DATA_PATH/knowledge``."""
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    assert Path(KnowledgeConfig.from_env().local_root) == tmp_path / "knowledge"


def test_default_local_root_falls_back_to_user_cache(
    env_clean: None,
) -> None:
    """Shared fallback: the user's Hyperloom cache."""
    assert Path(KnowledgeConfig.from_env().local_root).as_posix().endswith("/.cache/hyperloom/knowledge")


def test_knowledge_local_root_selects_the_store_root(
    env_clean: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``KNOWLEDGE_LOCAL_ROOT`` wins over the ``USER_DATA_PATH`` default and is not created by resolution."""
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "user"))
    monkeypatch.setenv("KNOWLEDGE_LOCAL_ROOT", str(tmp_path / "explicit"))
    assert Path(KnowledgeConfig.from_env().local_root) == tmp_path / "explicit"
    assert not (tmp_path / "explicit").exists()


def test_build_dispatcher_returns_recipe_kb(
    env_clean: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("KNOWLEDGE_LOCAL_ROOT", str(tmp_path))
    kb = _build_recipe_kb_dispatcher()
    assert isinstance(kb, RecipeKB)
    assert isinstance(kb.local, LocalRecipeStore)
    assert kb.local.root == tmp_path


def test_build_dispatcher_is_local_without_remote_credentials(
    env_clean: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Local mode wires only the local store."""
    monkeypatch.setenv("KNOWLEDGE_LOCAL_ROOT", str(tmp_path))
    kb = _build_recipe_kb_dispatcher()
    assert kb.mode == "local"


def test_build_dispatcher_remote_returns_none(
    env_clean: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Remote mode uses the KB Store CLOSE writer, not a dispatcher."""
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    monkeypatch.setenv("KB_STORE_URL", "https://kb.test")
    monkeypatch.setenv("KB_STORE_TOKEN", "token")
    monkeypatch.setenv("KNOWLEDGE_LOCAL_ROOT", str(tmp_path))
    kb = _build_recipe_kb_dispatcher()
    assert kb is None


def test_build_dispatcher_idempotent(
    env_clean: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Two builds produce equivalent but distinct dispatchers (no shared mutable state)."""
    monkeypatch.setenv("KNOWLEDGE_LOCAL_ROOT", str(tmp_path))
    a = _build_recipe_kb_dispatcher()
    b = _build_recipe_kb_dispatcher()
    assert a is not b
    assert a.local.root == b.local.root
    assert a.mode == b.mode == "local"
