# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every workspace runs its own loopback Experience KB service, configured by setup and started by each launch."""

from __future__ import annotations

import logging
import socket
from pathlib import Path
from typing import Any

import pytest

from hyperloom.common.llm_config import DEFAULT_CLAUDE_MODEL
from hyperloom.inference_optimizer import cli, experience_collect, experience_kb_service
from hyperloom_kb import LocalService, LocalServiceError


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _env_values(env_file: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in env_file.read_text(encoding="utf-8").splitlines() if "=" in line)


def test_init_env_points_a_new_workspace_at_the_local_service(tmp_path: Path, capsys) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("USER_DATA_PATH=/data\n", encoding="utf-8")

    assert experience_kb_service.main(["init-env", "--env-file", str(env_file)]) == 0

    values = _env_values(env_file)
    assert values["USER_DATA_PATH"] == "/data"
    assert values["HYPERLOOM_KB_URL"] == "http://127.0.0.1:8787"
    assert len(values["HYPERLOOM_KB_TOKEN"]) >= 32
    output = capsys.readouterr().out
    assert output == "HYPERLOOM_KB_URL: written\nHYPERLOOM_KB_TOKEN: written\n"
    assert values["HYPERLOOM_KB_TOKEN"] not in output


def test_init_env_keeps_configured_values_and_replaces_placeholders(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# HYPERLOOM_KB_URL=http://commented.example\n"
        'export HYPERLOOM_KB_URL="http://127.0.0.1:9901"\n'
        "HYPERLOOM_KB_TOKEN=<PLEASE_FILL_IN>\n",
        encoding="utf-8",
    )

    status = experience_kb_service.init_env(env_file)

    assert status == {"HYPERLOOM_KB_URL": "kept", "HYPERLOOM_KB_TOKEN": "written"}
    lines = env_file.read_text(encoding="utf-8").splitlines()
    assert lines[:2] == [
        "# HYPERLOOM_KB_URL=http://commented.example",
        'export HYPERLOOM_KB_URL="http://127.0.0.1:9901"',
    ]
    assert lines[2].startswith("HYPERLOOM_KB_TOKEN=") and "<PLEASE_FILL_IN>" not in lines[2]
    written = env_file.read_text(encoding="utf-8")

    assert experience_kb_service.init_env(env_file) == {"HYPERLOOM_KB_URL": "kept", "HYPERLOOM_KB_TOKEN": "kept"}
    assert env_file.read_text(encoding="utf-8") == written


@pytest.fixture
def started(monkeypatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def ensure_local_service(config, home, *, env):
        calls.append({"url": config.base_url, "home": home, "env": env})
        return LocalService({"status": "ok", "experience_count": 3})

    monkeypatch.setattr(experience_kb_service, "ensure_local_service", ensure_local_service)
    for key in ("HYPERLOOM_KB_URL", "LOCAL_KB_PLANNER_MODEL", "CLAUDE_MODEL", "ANTHROPIC_MODEL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HYPERLOOM_KB_TOKEN", "workspace-token")
    return calls


def test_only_a_loopback_url_is_a_service_this_workspace_runs(monkeypatch, started) -> None:
    assert experience_kb_service.ensure_service() is None
    monkeypatch.setenv("HYPERLOOM_KB_URL", "https://kb.example")
    assert experience_kb_service.ensure_service() is None
    assert started == []


def test_the_service_keeps_its_data_under_user_data_path(monkeypatch, tmp_path: Path, started) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("HYPERLOOM_KB_URL", "http://127.0.0.1:8787")

    experience_kb_service.ensure_service()
    monkeypatch.setenv("CLAUDE_MODEL", "claude-opus-4-8")
    experience_kb_service.ensure_service()

    assert [call["home"] for call in started] == [tmp_path / "experience-kb"] * 2
    assert started[0]["env"]["LOCAL_KB_PLANNER_MODEL"] == DEFAULT_CLAUDE_MODEL
    assert "LOCAL_KB_PLANNER_MODEL" not in started[1]["env"]


def test_ensure_reports_a_service_that_cannot_serve(monkeypatch, capsys) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise LocalServiceError("http://127.0.0.1:8787 is listening but did not answer as this Experience service")

    monkeypatch.setattr(experience_kb_service, "ensure_local_service", refuse)
    monkeypatch.setenv("HYPERLOOM_KB_URL", "http://127.0.0.1:8787")
    monkeypatch.setenv("HYPERLOOM_KB_TOKEN", "workspace-token")

    assert experience_kb_service.main(["ensure"]) == 1
    captured = capsys.readouterr()
    assert "did not answer as this Experience service" in captured.err
    assert "workspace-token" not in captured.err + captured.out


def test_a_launch_continues_when_the_service_cannot_serve(monkeypatch, caplog) -> None:
    validated: list[bool] = []

    def refuse() -> Any:
        raise LocalServiceError("Experience service exited with status 1")

    monkeypatch.setattr(experience_kb_service, "ensure_service", refuse)
    monkeypatch.setattr(experience_collect, "validate_config", lambda: validated.append(True))

    with caplog.at_level(logging.WARNING):
        cli._start_experience_kb()

    assert validated == [True]
    assert "Experience writes are spooled" in caplog.text


def test_setup_then_launch_start_one_service_for_the_workspace(monkeypatch, tmp_path: Path, capsys) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"HYPERLOOM_KB_URL=http://127.0.0.1:{_free_port()}\n", encoding="utf-8")
    experience_kb_service.init_env(env_file)
    for key, value in _env_values(env_file).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "data"))
    for key in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(key, raising=False)

    service = experience_kb_service.ensure_service()
    assert service is not None and service.process is not None
    try:
        assert experience_kb_service.main(["ensure"]) == 0
        assert "already running" in capsys.readouterr().out
        assert (tmp_path / "data" / "experience-kb" / "service.log").is_file()
    finally:
        service.process.terminate()
        service.process.wait(timeout=10)
