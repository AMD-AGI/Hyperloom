"""KernelForge child environment contract for Phase 1 KnowledgePlane."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKENDS_DIR = Path(__file__).resolve().parent.parent / "tools" / "backends"
sys.path.insert(0, str(_BACKENDS_DIR))
import forge_submit  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_config_cache() -> None:
    forge_submit._reset_knowledge_config_cache()
    yield
    forge_submit._reset_knowledge_config_cache()


def _avoid_unrelated_fellow_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    import _llm_stability_env

    monkeypatch.setattr(_llm_stability_env, "apply_llm_stability_env", lambda env: None)
    monkeypatch.setattr(forge_submit.shutil, "which", lambda _name: None)


def test_local_child_env_strips_remote_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _avoid_unrelated_fellow_setup(monkeypatch)
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "local")
    monkeypatch.setenv("KNOWLEDGE_LOCAL_ROOT", "/data/knowledge")
    monkeypatch.setenv("GBRAIN_BASE_URL", "https://ambient.invalid")
    monkeypatch.setenv("GBRAIN_TOKEN", "secret")
    monkeypatch.setenv("KB_STORE_URL", "https://ambient-kb.invalid")
    monkeypatch.setenv("KB_STORE_TOKEN", "kb-secret")
    env = {
        "KNOWLEDGE_STORE_MODE": "local",
        "KNOWLEDGE_LOCAL_ROOT": "/data/knowledge",
        "GBRAIN_BASE_URL": "https://ambient.invalid",
        "GBRAIN_TOKEN": "secret",
        "KB_STORE_URL": "https://ambient-kb.invalid",
        "KB_STORE_TOKEN": "kb-secret",
        "KERNELFORGE_GBRAIN_ENABLED": "true",
    }
    forge_submit._apply_fellow_env(env)
    assert env["KNOWLEDGE_STORE_MODE"] == "local"
    assert env["KNOWLEDGE_LOCAL_ROOT"] == "/data/knowledge"
    assert env["KERNELFORGE_GBRAIN_ENABLED"] == "false"
    assert "GBRAIN_BASE_URL" not in env
    assert "GBRAIN_TOKEN" not in env
    assert "KB_STORE_URL" not in env
    assert "KB_STORE_TOKEN" not in env


def test_remote_child_env_forwards_kb_store_alone_when_gbrain_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _avoid_unrelated_fellow_setup(monkeypatch)
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    monkeypatch.setenv("KNOWLEDGE_LOCAL_ROOT", "unchanged/root")
    monkeypatch.setenv("KB_STORE_URL", "https://kb.test")
    monkeypatch.setenv("KB_STORE_TOKEN", "token")
    monkeypatch.delenv("GBRAIN_BASE_URL", raising=False)
    monkeypatch.delenv("GBRAIN_TOKEN", raising=False)
    env = {
        "KNOWLEDGE_STORE_MODE": "remote",
        "KNOWLEDGE_LOCAL_ROOT": "unchanged/root",
        "KB_STORE_URL": "https://kb.test",
        "KB_STORE_TOKEN": "token",
        "KERNELFORGE_GBRAIN_ENABLED": "true",
    }
    forge_submit._apply_fellow_env(env)
    assert env["KNOWLEDGE_STORE_MODE"] == "remote"
    assert env["KNOWLEDGE_LOCAL_ROOT"] == "unchanged/root"
    assert env["KB_STORE_URL"] == "https://kb.test"
    assert env["KB_STORE_TOKEN"] == "token"
    assert env["KERNELFORGE_GBRAIN_ENABLED"] == "false"


def test_remote_child_env_strips_parent_gbrain_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _avoid_unrelated_fellow_setup(monkeypatch)
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    monkeypatch.setenv("KNOWLEDGE_LOCAL_ROOT", "unchanged/root")
    monkeypatch.setenv("KB_STORE_URL", "https://kb.test")
    monkeypatch.setenv("KB_STORE_TOKEN", "token")
    monkeypatch.setenv("GBRAIN_BASE_URL", "https://gbrain.test")
    monkeypatch.setenv("GBRAIN_TOKEN", "gbrain-token")
    env = {
        "KNOWLEDGE_STORE_MODE": "remote",
        "KNOWLEDGE_LOCAL_ROOT": "unchanged/root",
        "KB_STORE_URL": "https://kb.test",
        "KB_STORE_TOKEN": "token",
        "GBRAIN_BASE_URL": "https://gbrain.test",
        "GBRAIN_TOKEN": "gbrain-token",
    }
    forge_submit._apply_fellow_env(env)
    assert env["KB_STORE_URL"] == "https://kb.test"
    assert "GBRAIN_BASE_URL" not in env
    assert "GBRAIN_TOKEN" not in env
    assert env["KERNELFORGE_GBRAIN_ENABLED"] == "false"


def test_remote_child_env_missing_kb_store_credentials_degrades_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _avoid_unrelated_fellow_setup(monkeypatch)
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "remote")
    monkeypatch.setenv("KNOWLEDGE_LOCAL_ROOT", "/unused")
    monkeypatch.setenv("KB_STORE_URL", "https://kb.test")
    monkeypatch.delenv("KB_STORE_TOKEN", raising=False)
    env = {
        "KNOWLEDGE_STORE_MODE": "remote",
        "KNOWLEDGE_LOCAL_ROOT": "/unused",
        "KB_STORE_URL": "https://kb.test",
    }
    with caplog.at_level("WARNING"):
        forge_submit._apply_fellow_env(env)
        forge_submit._apply_fellow_env(env)
    assert env["KNOWLEDGE_STORE_MODE"] == "local"
    assert env["KERNELFORGE_GBRAIN_ENABLED"] == "false"
    assert "KB_STORE_URL" not in env
    assert "KB_STORE_TOKEN" not in env
    assert caplog.text.count("Forge knowledge configuration is invalid") == 1


def test_malformed_mode_hot_path_is_cached_without_mutating_process_env(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _avoid_unrelated_fellow_setup(monkeypatch)
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "malformed")
    monkeypatch.setenv("GBRAIN_TOKEN", "ambient-secret")
    process_mode = dict(forge_submit.os.environ)
    first = {"KNOWLEDGE_STORE_MODE": "malformed", "GBRAIN_TOKEN": "child-secret"}
    second = dict(first)

    with caplog.at_level("WARNING"):
        forge_submit._apply_fellow_env(first)
        forge_submit._apply_fellow_env(second)

    assert first["KNOWLEDGE_STORE_MODE"] == second["KNOWLEDGE_STORE_MODE"] == "local"
    assert "GBRAIN_TOKEN" not in first
    assert "GBRAIN_TOKEN" not in second
    assert dict(forge_submit.os.environ) == process_mode
    assert caplog.text.count("Forge knowledge configuration is invalid") == 1


def test_unset_mode_defaults_local_and_uses_user_data_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _avoid_unrelated_fellow_setup(monkeypatch)
    monkeypatch.delenv("KNOWLEDGE_STORE_MODE", raising=False)
    monkeypatch.delenv("KNOWLEDGE_LOCAL_ROOT", raising=False)
    monkeypatch.setenv("USER_DATA_PATH", "/data/user")
    monkeypatch.setenv("GBRAIN_BASE_URL", "https://ambient.invalid")
    monkeypatch.setenv("GBRAIN_TOKEN", "secret")
    env = {
        "USER_DATA_PATH": "/data/user",
        "GBRAIN_BASE_URL": "https://ambient.invalid",
        "GBRAIN_TOKEN": "secret",
    }
    forge_submit._apply_fellow_env(env)
    assert env["KNOWLEDGE_STORE_MODE"] == "local"
    assert env["KNOWLEDGE_LOCAL_ROOT"] == "/data/user/knowledge"
    assert "GBRAIN_BASE_URL" not in env
    assert "GBRAIN_TOKEN" not in env


def test_child_env_cannot_seed_process_config_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _avoid_unrelated_fellow_setup(monkeypatch)
    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "local")
    monkeypatch.setenv("KNOWLEDGE_LOCAL_ROOT", "/process/knowledge")
    monkeypatch.delenv("GBRAIN_BASE_URL", raising=False)
    monkeypatch.delenv("GBRAIN_TOKEN", raising=False)
    child = {
        "KNOWLEDGE_STORE_MODE": "remote",
        "KNOWLEDGE_LOCAL_ROOT": "/child/knowledge",
        "KB_STORE_URL": "https://child.invalid",
        "KB_STORE_TOKEN": "child-secret",
    }

    forge_submit._apply_fellow_env(child)
    cached = forge_submit._knowledge_config_for_forge()

    assert cached.mode.value == "local"
    assert str(cached.local_root) == "/process/knowledge"
    assert child["KNOWLEDGE_STORE_MODE"] == "local"
    assert child["KNOWLEDGE_LOCAL_ROOT"] == "/process/knowledge"
    assert "KB_STORE_URL" not in child
    assert "KB_STORE_TOKEN" not in child
