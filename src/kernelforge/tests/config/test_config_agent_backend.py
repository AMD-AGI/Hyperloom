# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Forge follows the platform's harness selection unless told otherwise."""

from __future__ import annotations

import pytest

from hyperloom.common import llm_config
from kernelforge.config import Config, resolve_agent_model

_ENV_KEYS = (
    "FORGE_AGENT_BACKEND",
    "FORGE_AGENT_FALLBACK_PROVIDER",
    llm_config.AGENT_BACKEND_ENV,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(llm_config, "deepseek_harness_sdk_installed", lambda: True)


def test_the_platform_harness_selection_reaches_forge(monkeypatch: pytest.MonkeyPatch) -> None:
    """A live dsh run authored its kernel on Claude: Forge read only FORGE_AGENT_BACKEND."""
    monkeypatch.setenv(llm_config.AGENT_BACKEND_ENV, "deepseek")

    assert Config.from_env().agent_backend == "deepseek"


def test_a_named_harness_does_not_fall_back_to_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    """A silent Claude fallback would make a failed harness run read as a working one."""
    monkeypatch.setenv(llm_config.AGENT_BACKEND_ENV, "deepseek")

    assert Config.from_env().agent_fallback_provider == ""


def test_forge_own_selection_still_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(llm_config.AGENT_BACKEND_ENV, "deepseek")
    monkeypatch.setenv("FORGE_AGENT_BACKEND", "codex")

    assert Config.from_env().agent_backend == "codex"


def test_an_explicit_fallback_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(llm_config.AGENT_BACKEND_ENV, "deepseek")
    monkeypatch.setenv("FORGE_AGENT_FALLBACK_PROVIDER", "codex")

    assert Config.from_env().agent_fallback_provider == "codex"


def test_nothing_named_keeps_the_default_ladder() -> None:
    config = Config.from_env()

    assert config.agent_backend == "auto"
    assert config.agent_fallback_provider == "claude"


def test_forge_loop_runs_the_model_the_harness_resolver_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """forge-fuse and forge-loop must not run different models under one ``DSH_MODEL``."""
    monkeypatch.setenv("DSH_MODEL", "deepseek-v4-pro")

    assert resolve_agent_model("deepseek") == "deepseek-v4-pro"
    assert resolve_agent_model("deepseek") == llm_config.resolve_forge_llm_model("deepseek")
