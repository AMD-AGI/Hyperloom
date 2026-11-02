# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A forge-loop run collected into the Experience KB its environment names: as it ends, and again from the CLI."""

from __future__ import annotations

import os
import socket
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from hyperloom_kb import (
    ExperienceHTTPService,
    HTTPServiceConfig,
    LLMQueryPlanner,
    PlannerConfiguration,
    RemoteClient,
    RemoteConfig,
    create_http_server,
)
from hyperloom_kb.collect import load_mapping
from hyperloom_kb.database import SqliteDatabase
from hyperloom_kb import workspace as kb_workspace
from hyperloom.inference_optimizer import experience_kb_service
from kernelforge.cli import main
from kernelforge.workflows.forge_loop.campaign.experiments_layout import ExperimentsLayout
from kernelforge.workflows.forge_loop.export import collect as run_collect
from kernelforge.tests.workflows.forge_loop.export.test_run_export import (
    KB_RECORD,
    WORKSPACE,
    _loop_log,
    _plain_run,
    _record_written,
)

TOKEN = "kernelforge-collection-test-token"
# The real one, which the test suite's conftest stands in for in every test that does not test it.
_USE_WORKSPACE_KB = run_collect.use_workspace_kb


@pytest.fixture(autouse=True)
def _the_real_workspace_kb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_collect, "use_workspace_kb", _USE_WORKSPACE_KB)


class _UnusedPlanner:
    model = "unused-planner"

    def complete(self, **_kwargs: Any) -> str:
        raise AssertionError("collection never plans a read")


@pytest.fixture
def kb_url(tmp_path: Path) -> Iterator[str]:
    """A local Experience KB serving the kernelforge-v1 declaration, as a workspace's own service does."""

    app = ExperienceHTTPService(
        HTTPServiceConfig(tmp_path / "service", TOKEN),
        load_mapping(run_collect.MAPPING).declaration,
        LLMQueryPlanner(_UnusedPlanner(), PlannerConfiguration.create("unused-planner")),
        database=SqliteDatabase(tmp_path / "kb.sqlite3"),
    )
    server = create_http_server(app, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host!s}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _stored(url: str) -> set[str]:
    client = RemoteClient(RemoteConfig(url, TOKEN))
    return {item["experience_id"] for item in client.list_experiences().items}


def test_a_run_is_collected_as_it_ends_and_collecting_it_again_writes_nothing_new(
    tmp_path: Path, kb_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """The token reaches collection from its file, and a later export with the run's log gives the same Experience."""

    root = tmp_path / "runs"
    layout = _plain_run(root)
    _record_written(layout, 1, KB_RECORD)
    token_file = tmp_path / "token"
    token_file.write_text(TOKEN + "\n", encoding="utf-8")
    env = {"HYPERLOOM_KB_URL": kb_url, "HYPERLOOM_KB_TOKEN_FILE": str(token_file)}

    run_collect.collect_run(layout.root, env)

    ended = capsys.readouterr().out
    assert "collected 1 of 3 candidates" in ended and "(1 created)" in ended
    assert "2 skipped, 0 failed" in ended
    assert (layout.experience_collection / run_collect.RECEIPT).is_file()
    [experience] = _stored(kb_url)

    _loop_log(root / "forge-loop.log")
    document = tmp_path / "later" / "runs.json"
    result = CliRunner(env=env).invoke(main, ["export-runs", str(root), "-o", str(document), "--collect"])

    assert result.exit_code == 0, result.output
    assert "(1 unchanged)" in result.output
    assert _stored(kb_url) == {experience}


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_a_workspace_with_no_kb_starts_the_one_it_writes_and_the_run_collects_into_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    started: list[Any] = []

    def recording(*args: Any, **kwargs: Any) -> Any:
        service = real_ensure(*args, **kwargs)
        started.append(service)
        return service

    real_ensure = run_collect.ensure_local_service
    monkeypatch.setattr(run_collect, "ensure_local_service", recording)
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "REPO_ROOT": str(tmp_path),
        "USER_DATA_PATH": str(tmp_path / "user-data"),
    }
    try:
        note = run_collect.use_workspace_kb(env)
        launch = run_collect.prepare_collection(env, note=note)
        layout = _plain_run(tmp_path / "runs")
        run_collect.collect_run(layout.root, env)
    finally:
        for service in started:
            if service.process is not None:
                service.process.terminate()
                service.process.wait(timeout=10)

    home = tmp_path / "user-data" / "experience-kb"
    assert launch[0] == note
    assert launch[1] == f"  Experience KB service: started at {env['HYPERLOOM_KB_URL']} from {home}"
    assert "(1 created)" in capsys.readouterr().out


def test_the_launch_starts_the_local_kb_a_loopback_url_names_and_the_run_collects_into_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With wiki or without, the run's local KB serves from the home a Hyperloom workspace keeps it in."""

    started: list[Any] = []

    def recording(*args: Any, **kwargs: Any) -> Any:
        service = real_ensure(*args, **kwargs)
        started.append(service)
        return service

    real_ensure = run_collect.ensure_local_service
    monkeypatch.setattr(run_collect, "ensure_local_service", recording)
    url = f"http://127.0.0.1:{_free_port()}"
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "HYPERLOOM_KB_URL": url,
        "HYPERLOOM_KB_TOKEN": TOKEN,
        "USER_DATA_PATH": str(tmp_path / "user-data"),
    }
    try:
        launch = run_collect.prepare_collection(env)
        layout = _plain_run(tmp_path / "runs")
        run_collect.collect_run(layout.root, env)
        again = run_collect.start_local_kb(env)
    finally:
        for service in started:
            if service.process is not None:
                service.process.terminate()
                service.process.wait(timeout=10)

    home = tmp_path / "user-data" / "experience-kb"
    assert launch[0] == f"  Experience KB service: started at {url} from {home}"
    assert launch[1].startswith(f"  Experience KB collection: {url} (")
    assert "(1 created)" in capsys.readouterr().out
    assert again == f"  Experience KB service: serving at {url} from {home}"
    assert [service.process is not None for service in started] == [True, False]


def test_a_kb_elsewhere_is_not_started_and_a_home_can_be_named(tmp_path: Path) -> None:
    env = {"HYPERLOOM_KB_URL": "https://kb.example.invalid", "HYPERLOOM_KB_TOKEN": TOKEN}

    assert run_collect.start_local_kb(env) == ""
    assert kb_workspace.service_home({"HYPERLOOM_KB_HOME": str(tmp_path / "kb")}) == tmp_path / "kb"
    assert kb_workspace.service_home({"USER_DATA_PATH": str(tmp_path)}) == tmp_path / "experience-kb"


def test_a_run_takes_the_kb_its_workspace_env_names_when_its_environment_names_none(tmp_path: Path) -> None:
    """As an optimize run does: the .env fills only what the environment leaves unset."""

    (tmp_path / ".env").write_text(
        f"OTHER=kept\nHYPERLOOM_KB_URL=http://127.0.0.1:21000\nHYPERLOOM_KB_TOKEN={TOKEN}\n"
        f"USER_DATA_PATH={tmp_path / 'user-data'}\nANTHROPIC_API_KEY=not-a-kb-setting\n",
        encoding="utf-8",
    )
    env = {"REPO_ROOT": str(tmp_path), "USER_DATA_PATH": str(tmp_path / "chosen")}

    assert run_collect.use_workspace_kb(env) == ""

    assert env == {
        "REPO_ROOT": str(tmp_path),
        "USER_DATA_PATH": str(tmp_path / "chosen"),
        "HYPERLOOM_KB_URL": "http://127.0.0.1:21000",
        "HYPERLOOM_KB_TOKEN": TOKEN,
    }


def test_a_workspace_with_no_kb_gets_the_one_setup_would_write(tmp_path: Path) -> None:
    """Written once, beside what the .env already holds, and taken as it is by every run after."""

    env_file = tmp_path / ".env"
    env_file.write_text("OTHER=kept\n", encoding="utf-8")
    env = {"REPO_ROOT": str(tmp_path)}

    line = run_collect.use_workspace_kb(env)

    assert (
        line
        == f"  Experience KB: the workspace's local KB, whose URL and token were written to {env_file} as setup does"
    )
    written = env_file.read_text(encoding="utf-8")
    assert written.startswith("OTHER=kept\n")
    assert env["HYPERLOOM_KB_URL"].startswith("http://127.0.0.1:") and env["HYPERLOOM_KB_TOKEN"]
    later = {"REPO_ROOT": str(tmp_path)}
    assert run_collect.use_workspace_kb(later) == ""
    assert (later["HYPERLOOM_KB_URL"], later["HYPERLOOM_KB_TOKEN"]) == (
        env["HYPERLOOM_KB_URL"],
        env["HYPERLOOM_KB_TOKEN"],
    )
    assert env_file.read_text(encoding="utf-8") == written


def test_a_run_given_a_kb_url_leaves_the_workspace_env_alone(tmp_path: Path) -> None:
    env = {"REPO_ROOT": str(tmp_path), "HYPERLOOM_KB_URL": "https://kb.example.invalid", "HYPERLOOM_KB_TOKEN": TOKEN}

    assert run_collect.use_workspace_kb(env) == ""
    assert not (tmp_path / ".env").exists()


def test_an_optimize_run_and_a_forge_loop_run_serve_one_kb_from_one_home(tmp_path: Path, monkeypatch) -> None:
    """The workspace has one Experience KB: both products name the same home for the same settings."""

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "user-data"))
    monkeypatch.delenv("HYPERLOOM_KB_HOME", raising=False)

    assert (
        kb_workspace.service_home() == experience_kb_service.service_home() == tmp_path / "user-data" / "experience-kb"
    )


def test_a_run_is_not_collected_without_a_kb_url(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    layout = _plain_run(tmp_path)

    run_collect.collect_run(layout.root, {"HYPERLOOM_KB_TOKEN": TOKEN})

    assert capsys.readouterr().out == ""
    assert not layout.experience_collection.exists()


def test_an_unreachable_kb_keeps_the_writes_for_the_next_collection_and_never_fails_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def offline(*_args: Any, **_kwargs: Any) -> Any:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    layout = _plain_run(tmp_path)

    run_collect.collect_run(layout.root, {"HYPERLOOM_KB_URL": "http://kb.invalid", "HYPERLOOM_KB_TOKEN": TOKEN})

    assert "(1 spooled)" in capsys.readouterr().out
    assert list((layout.experience_collection / run_collect.SPOOL).glob("spool-*.json"))


def test_a_kb_without_a_token_is_reported_and_the_run_goes_on(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = _plain_run(tmp_path)
    env = {"HYPERLOOM_KB_URL": "http://kb.invalid"}

    run_collect.collect_run(layout.root, env)

    ended = capsys.readouterr().out
    assert "[experience-kb] not collected (RemoteClientError: HYPERLOOM_KB_TOKEN must be configured)" in ended
    assert f"kernelforge export-runs {layout.root} -o {layout.experience_collection / run_collect.DOCUMENT}" in ended
    assert run_collect.collection_target(env).startswith("  Experience KB collection: off, HYPERLOOM_KB_URL is set")


def test_the_launch_line_names_the_kb_and_the_schema_the_run_is_collected_under() -> None:
    env = {"HYPERLOOM_KB_URL": "http://127.0.0.1:28788", "HYPERLOOM_KB_TOKEN": TOKEN}
    schema_ref = load_mapping(run_collect.MAPPING).declaration.schema_ref

    assert run_collect.collection_target({}) == ""
    assert run_collect.collection_target(env) == (
        f"  Experience KB collection: http://127.0.0.1:28788 ({schema_ref}) as the run ends"
    )
    assert TOKEN not in run_collect.collection_target(env)


@pytest.mark.parametrize(
    ("switch", "pushed"),
    [
        pytest.param(None, "", id="not-asked"),
        pytest.param("1", "pushed to the global KB: 1 created, 0 unchanged, 0 rejected", id="asked"),
        pytest.param("sometimes", "push skipped", id="unreadable-switch"),
    ],
)
def test_the_kb_is_asked_to_push_to_its_global_kb_only_when_the_run_asks(
    monkeypatch: pytest.MonkeyPatch, switch: str | None, pushed: str
) -> None:
    monkeypatch.setattr(RemoteClient, "push", lambda _self: {"created": 1, "unchanged": 0, "rejected": []})
    env = {"HYPERLOOM_KB_URL": "http://127.0.0.1:28788", "HYPERLOOM_KB_TOKEN": TOKEN}
    if switch is not None:
        env[run_collect.AUTO_PUSH_ENV] = switch

    assert run_collect.push_to_global(env).startswith(pushed)


def test_export_runs_collect_needs_a_kb(tmp_path: Path) -> None:
    _plain_run(tmp_path / "runs")

    result = CliRunner(env={"HYPERLOOM_KB_URL": ""}).invoke(
        main, ["export-runs", str(tmp_path / "runs"), "-o", str(tmp_path / "runs.json"), "--collect"]
    )

    assert result.exit_code != 0
    assert "--collect needs the Experience KB HYPERLOOM_KB_URL names" in result.output


def test_the_document_and_receipt_are_kept_in_the_runs_experience_kb_directory(tmp_path: Path) -> None:
    assert ExperimentsLayout.of(tmp_path / WORKSPACE).experience_collection == (
        tmp_path / WORKSPACE / "forge_experiments" / "experience_kb"
    )
