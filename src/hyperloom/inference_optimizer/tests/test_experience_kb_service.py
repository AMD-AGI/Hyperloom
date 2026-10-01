# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every workspace runs its own loopback Experience KB service, configured by setup and started by each launch."""

from __future__ import annotations

import json
import logging
import socket
from pathlib import Path
from typing import Any, cast

import pytest

import hyperloom
from hyperloom.common.llm_config import DEFAULT_CLAUDE_MODEL
from hyperloom.inference_optimizer import cli, experience_collect, experience_kb_service
from hyperloom_kb import (
    GLOBAL_TOKEN_ENV,
    GLOBAL_URL_ENV,
    ExperienceDeclaration,
    LocalService,
    LocalServiceError,
)

_PACKAGE = Path(hyperloom.__file__).parent


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
    assert values["HYPERLOOM_KB_URL"] == experience_kb_service.local_url(tmp_path)
    assert len(values["HYPERLOOM_KB_TOKEN"]) >= 32
    output = capsys.readouterr().out
    assert output == "HYPERLOOM_KB_URL: written\nHYPERLOOM_KB_TOKEN: written\n"
    assert values["HYPERLOOM_KB_TOKEN"] not in output


def test_each_workspace_gets_its_own_local_port_and_keeps_it(tmp_path: Path) -> None:
    first, second = tmp_path / "alice" / "workspace", tmp_path / "bob" / "workspace"
    urls = {}
    for workspace in (first, second):
        workspace.mkdir(parents=True)
        experience_kb_service.init_env(workspace / ".env")
        urls[workspace] = _env_values(workspace / ".env")["HYPERLOOM_KB_URL"]

    ports = {int(url.rsplit(":", 1)[1]) for url in urls.values()}
    assert all(url.startswith("http://127.0.0.1:") for url in urls.values())
    assert len(ports) == 2 and ports <= set(experience_kb_service.LOCAL_PORTS)
    assert experience_kb_service.local_url(first) == urls[first]

    port = int(urls[first].rsplit(":", 1)[1])
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", port))
        assert experience_kb_service.local_url(first) != urls[first]


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

    def ensure_local_service(config, home, *, env, restart):
        calls.append({"url": config.base_url, "home": home, "env": env, "restart": restart})
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


def test_ensure_says_when_it_restarted_a_stale_service(monkeypatch, capsys) -> None:
    def restart(*_args: Any, **_kwargs: Any) -> LocalService:
        return LocalService({"experience_count": 14}, cast(Any, object()), restarted=True)

    monkeypatch.setattr(experience_kb_service, "ensure_local_service", restart)
    monkeypatch.setenv("HYPERLOOM_KB_URL", "http://127.0.0.1:8787")
    monkeypatch.setenv("HYPERLOOM_KB_TOKEN", "workspace-token")

    assert experience_kb_service.main(["ensure"]) == 0
    assert "Experience KB service restarted at http://127.0.0.1:8787: 14 Experiences" in capsys.readouterr().out


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


def test_a_launch_continues_when_the_workspace_names_a_service_without_its_token(monkeypatch, caplog) -> None:
    monkeypatch.setenv("HYPERLOOM_KB_URL", f"http://127.0.0.1:{_free_port()}")
    monkeypatch.delenv("HYPERLOOM_KB_TOKEN", raising=False)

    with caplog.at_level(logging.WARNING):
        cli._start_experience_kb()

    assert "HYPERLOOM_KB_TOKEN must be configured" in caplog.text
    assert "the run continues without them" in caplog.text


def test_a_launch_continues_when_the_packaged_mapping_cannot_load(monkeypatch, caplog) -> None:
    monkeypatch.setenv("HYPERLOOM_KB_URL", f"http://127.0.0.1:{_free_port()}")
    monkeypatch.setenv("HYPERLOOM_KB_TOKEN", "local-token")
    monkeypatch.setattr(experience_kb_service, "ensure_service", lambda: None)
    monkeypatch.setattr(experience_collect, "MAPPING", "no-such-mapping")

    with caplog.at_level(logging.WARNING):
        cli._start_experience_kb()

    assert "the run continues without them" in caplog.text


def _push_report(**overrides: Any) -> dict[str, Any]:
    return {
        "status": "completed",
        "global_url": "https://global.example",
        "created": 2,
        "unchanged": 0,
        "skipped": 0,
        "held_back": 1,
        "rejected": [],
        **overrides,
    }


def _opt_into_auto_push(monkeypatch) -> None:
    monkeypatch.setenv("HYPERLOOM_KB_AUTO_PUSH", "1")
    monkeypatch.setenv("HYPERLOOM_GLOBAL_KB_URL", "https://global.example")
    monkeypatch.setenv("HYPERLOOM_GLOBAL_KB_TOKEN", "global-token")


def test_auto_push_is_off_until_the_workspace_opts_in(monkeypatch, caplog) -> None:
    pushed: list[str] = []
    monkeypatch.setattr(experience_kb_service, "sync_with_global", lambda direction: pushed.append(direction))
    monkeypatch.delenv("HYPERLOOM_KB_AUTO_PUSH", raising=False)
    experience_kb_service.auto_push()
    assert pushed == []

    monkeypatch.setattr(experience_kb_service, "sync_with_global", lambda direction: _push_report())
    _opt_into_auto_push(monkeypatch)
    with caplog.at_level(logging.INFO):
        experience_kb_service.auto_push()
    assert "push with https://global.example: 2 created, 0 unchanged, 0 skipped, 1 held_back, 0 rejected" in caplog.text


@pytest.mark.parametrize(
    "outcome",
    [
        LocalServiceError("Experience service exited with status 1"),
        _push_report(status="incomplete", created=1, error="global KB went away"),
    ],
)
def test_a_failed_auto_push_is_logged_and_never_fails_the_run(monkeypatch, caplog, outcome: Any) -> None:
    def push(_direction: str) -> Any:
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(experience_kb_service, "sync_with_global", push)
    _opt_into_auto_push(monkeypatch)

    with caplog.at_level(logging.WARNING):
        experience_kb_service.auto_push()

    assert "auto push" in caplog.text
    assert "the next push" in caplog.text


def test_push_without_a_global_kb_explains_what_is_missing(monkeypatch, capsys) -> None:
    monkeypatch.delenv("HYPERLOOM_GLOBAL_KB_URL", raising=False)

    assert experience_kb_service.main(["push"]) == 1
    assert "HYPERLOOM_GLOBAL_KB_URL is not configured" in capsys.readouterr().err


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


def test_a_push_never_stops_the_service_a_run_may_be_reading_from(monkeypatch, tmp_path: Path, caplog, capsys) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"HYPERLOOM_KB_URL=http://127.0.0.1:{_free_port()}\n", encoding="utf-8")
    experience_kb_service.init_env(env_file)
    for key, value in _env_values(env_file).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "data"))
    for key in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", GLOBAL_URL_ENV, GLOBAL_TOKEN_ENV):
        monkeypatch.delenv(key, raising=False)
    launched = experience_kb_service.ensure_service()
    relaunched: LocalService | None = None
    assert launched is not None and launched.process is not None
    try:
        monkeypatch.setenv(GLOBAL_URL_ENV, "https://global.example")
        monkeypatch.setenv(GLOBAL_TOKEN_ENV, "global-token")
        with caplog.at_level(logging.WARNING):
            assert experience_kb_service.main(["push"]) == 1

        assert launched.process.poll() is None
        assert "runs with other settings" in caplog.text
        assert "started without a global Experience KB" in capsys.readouterr().err

        relaunched = experience_kb_service.ensure_service()
        assert relaunched is not None and relaunched.restarted
        assert launched.process.wait(timeout=10) is not None
    finally:
        for service in (launched, relaunched):
            if service is not None and service.process is not None and service.process.poll() is None:
                service.process.terminate()
                service.process.wait(timeout=10)


def test_workspace_labels_restores_and_exclusions_act_on_the_schema_its_runs_write(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"HYPERLOOM_KB_URL=http://127.0.0.1:{_free_port()}\n", encoding="utf-8")
    experience_kb_service.init_env(env_file)
    for key, value in _env_values(env_file).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "data"))
    for key in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", GLOBAL_URL_ENV, GLOBAL_TOKEN_ENV):
        monkeypatch.delenv(key, raising=False)
    service = experience_kb_service.ensure_service()
    assert service is not None and service.process is not None
    try:
        assert experience_kb_service.main(["label", "--name", "before tuning"]) == 0
        label = json.loads(capsys.readouterr().out)
        assert experience_kb_service.main(["labels"]) == 0
        labels = json.loads(capsys.readouterr().out)
        assert experience_kb_service.main(["restore", label["label_id"]]) == 0
        restored = json.loads(capsys.readouterr().out)
        assert experience_kb_service.main(["exclude", "exp-" + "0" * 32, "--reason", "noisy node"]) == 1
        refused = capsys.readouterr().err
        assert experience_kb_service.main(["export"]) == 0
        exported = json.loads(capsys.readouterr().out)
        assert service.process.poll() is None
    finally:
        service.process.terminate()
        service.process.wait(timeout=10)

    assert label["schema_ref"] == labels["schema_ref"] == experience_collect.mapping_schema_ref()
    assert (
        ExperienceDeclaration.from_dict(exported["declaration"]).schema_ref == experience_collect.mapping_schema_ref()
    )
    assert [entry["label_id"] for entry in labels["labels"]] == [label["label_id"]]
    assert (restored["restored"]["label_id"], restored["saved"]) == (label["label_id"], None)
    assert "Experience KB exclude failed" in refused and "404" in refused


def test_skills_describe_the_service_by_the_commands_and_variables_it_reads() -> None:
    setup = (_PACKAGE / "skills/hyperloom-setup/SKILL.md").read_text(encoding="utf-8")
    optimizer = (_PACKAGE / "inference_optimizer/SKILL.md").read_text(encoding="utf-8")
    global_kb = (_PACKAGE / "skills/hyperloom-global-kb/SKILL.md").read_text(encoding="utf-8")

    for name in ("HYPERLOOM_KB_URL", "HYPERLOOM_KB_TOKEN"):
        assert name in setup
        assert name in optimizer
    # Every workspace gets its local service: setup generates its .env entries and starts it; nobody opts out.
    assert "hyperloom.inference_optimizer.experience_kb_service init-env" in setup
    assert "hyperloom.inference_optimizer.experience_kb_service ensure" in setup
    assert "No Experience KB" not in setup
    assert "pip install your_package.whl --target ." in setup
    assert "hyperloom_kb-" not in setup
    assert "[kb]" not in setup
    assert "experience_kb_injections" in optimizer
    for name in (GLOBAL_URL_ENV, GLOBAL_TOKEN_ENV, experience_kb_service.AUTO_PUSH_ENV):
        for text in (setup, optimizer, global_kb):
            assert name in text
    for command in ("push", "pull"):
        assert f"hyperloom.inference_optimizer.experience_kb_service {command}" in setup
        assert f"hyperloom.inference_optimizer.experience_kb_service {command}" in optimizer
    # Labels, restores, and exclusions are the hyperloom-kb skill's commands, run through the workspace entry point.
    assert "hyperloom.inference_optimizer.experience_kb_service labels" in setup
    assert "experience_kb_service restore <label_id>" in optimizer
    for text in (setup, optimizer, global_kb):
        assert "`hyperloom-kb` skill" in " ".join(text.split())
    assert "python3 -m hyperloom_kb --host 0.0.0.0" in global_kb
    for text in (setup, optimizer, global_kb):
        assert "HYPERLOOM_FLEET_KB" not in text
        assert "HYPERLOOM_KB_ENABLE" not in text
        assert "HYPERLOOM_KB_DECL" not in text
        assert "Hyperloom-KB.git" not in text
        assert "trust_state" not in text
        assert "unverified" not in text
        assert "slack" not in text.lower()
