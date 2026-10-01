# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``hyperloom-kb`` operates whatever service ``HYPERLOOM_KB_URL`` names: JSON out, and exit 1 when it did not work."""

from __future__ import annotations

import argparse
import json
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

import hyperloom_kb
from hyperloom_kb import (
    Change,
    Experience,
    ExperienceDeclaration,
    ExperienceHTTPService,
    ExperienceStatus,
    FieldDeclaration,
    HTTPServiceConfig,
    ObjectiveDeclaration,
    ObjectiveDirection,
    Outcome,
    Provenance,
    RemoteClient,
    RemoteConfig,
    create_http_server,
    derive_experience_id,
)
from hyperloom_kb.cli import add_commands, main, run_command

TOKEN = "cli-token"
NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)
SCHEMA = ExperienceDeclaration(
    identity=(FieldDeclaration("model", "Model."),),
    baseline_identity=(FieldDeclaration("config", "Baseline."),),
    change_identity=(FieldDeclaration("knob", "Knob."),),
    objectives=(ObjectiveDeclaration("throughput@v1", ObjectiveDirection.HIGHER_IS_BETTER, "Throughput."),),
    decisions=("keep", "revert"),
)


def _experience(seq: int, run_id: str = "cli-run") -> Experience:
    return Experience(
        id=derive_experience_id("cli-test", run_id, seq),
        run_id=run_id,
        seq=seq,
        created_at=NOW,
        completed_at=NOW,
        identity={"model": "qwen3"},
        objective="throughput@v1",
        baseline_identity={"config": "default"},
        baseline_value=100.0,
        provenance=Provenance("cli-test", "1"),
        schema_ref=SCHEMA.schema_ref,
        status=ExperienceStatus.COMPLETE,
        reasoning=f"Try knob {seq}.",
        change=Change({"knob": f"knob_{seq}"}, f"Set knob {seq}.", kind="config"),
        outcome=Outcome("keep", 110.0 + seq),
        reflection="Measured.",
    )


@contextmanager
def _serving(home: Path, global_url: str | None = None) -> Iterator[RemoteClient]:
    global_kb = None if global_url is None else RemoteClient(RemoteConfig(global_url, TOKEN))
    app = ExperienceHTTPService(HTTPServiceConfig(home, TOKEN), SCHEMA, None, global_kb=global_kb)
    server = create_http_server(app, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    thread.start()
    try:
        yield RemoteClient(RemoteConfig(f"http://127.0.0.1:{server.server_address[1]}", TOKEN))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def kb(monkeypatch, capsys) -> Any:
    """Run ``hyperloom-kb`` against a client's service: ``(exit status, parsed stdout or stderr text)``."""

    def run(client: RemoteClient, *argv: str) -> tuple[int, Any]:
        monkeypatch.setenv("HYPERLOOM_KB_URL", client.config.base_url)
        monkeypatch.setenv("HYPERLOOM_KB_TOKEN", TOKEN)
        status = main(list(argv))
        out, err = capsys.readouterr()
        return status, json.loads(out) if out else err

    return run


def _closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_label_exclude_and_restore_a_state_from_the_command_line(tmp_path: Path, kb) -> None:
    with _serving(tmp_path / "kb") as client:
        for seq in range(3):
            client.write(_experience(seq), declaration=SCHEMA)
        _, three = kb(client, "label", "--name", "three tried knobs")
        for seq in (3, 4):
            client.write(_experience(seq))
        _, labels = kb(client, "labels")
        _, excluded = kb(client, "exclude", _experience(0).id, "--reason", "measured on a noisy node")
        _, listed = kb(client, "list")
        _, everything = kb(client, "list", "--include-excluded")
        status, restored = kb(client, "restore", three["label_id"])
        _, exclusions = kb(client, "exclusions", "--schema", SCHEMA.schema_ref)
        _, health = kb(client, "health")

    assert (three["name"], three["member_count"]) == ("three tried knobs", 3)
    assert (labels["current_label_id"], labels["modified"]) == (three["label_id"], True)
    assert excluded == {"experience_id": _experience(0).id, "status": "excluded"}
    assert (len(listed["items"]), len(everything["items"])) == (4, 5)
    assert status == 0
    assert (restored["saved"]["reason"], restored["saved"]["member_count"], restored["saved"]["excluded_count"]) == (
        "before_restore",
        5,
        1,
    )
    assert exclusions["exclusions"] == []
    assert [entry["action"] for entry in exclusions["history"]] == ["exclude"]
    assert health["experience_count"] == 3


def test_pull_names_its_schema_and_reports_the_label_that_undoes_it(tmp_path: Path, kb) -> None:
    with _serving(tmp_path / "global") as global_client:
        for seq in range(2):
            global_client.write(_experience(seq, run_id="teammate-run"), declaration=SCHEMA)
        with _serving(tmp_path / "local", global_client.config.base_url) as local:
            local.write(_experience(0), declaration=SCHEMA)
            status, pulled = kb(local, "pull", "--schema", SCHEMA.schema_ref)
            _, exported = kb(local, "export", "--schema", SCHEMA.schema_ref)

    assert status == 0
    assert (pulled["status"], pulled["created"], pulled["saved"]["reason"]) == ("completed", 2, "before_pull")
    # Export names what was written here, never what was pulled.
    assert [item["experience"]["id"] for item in exported["items"]] == [_experience(0).id]
    assert exported["declaration"] == SCHEMA.to_dict()


def test_a_sync_that_did_not_finish_and_a_refused_request_exit_1(tmp_path: Path, kb) -> None:
    with _serving(tmp_path / "local", f"http://127.0.0.1:{_closed_port()}") as local:
        pull_status, pulled = kb(local, "pull", "--schema", SCHEMA.schema_ref)
        restore_status, restore_error = kb(local, "restore", "label-unknown")
    with _serving(tmp_path / "alone") as alone:
        push_status, push_error = kb(alone, "push")

    assert (pull_status, pulled["status"]) == (1, "incomplete")
    assert restore_status == 1 and "Experience KB restore failed" in restore_error and "404" in restore_error
    assert push_status == 1 and "started without a global Experience KB" in push_error


def test_a_tool_embeds_every_command_with_its_own_schema_and_its_own_versions(tmp_path: Path, capsys) -> None:
    parser = argparse.ArgumentParser(prog="tool")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("push").set_defaults(own=True)
    add_commands(commands, schema_ref=SCHEMA.schema_ref)

    assert set(commands.choices) == {
        *("health", "push", "pull", "labels", "label", "restore", "exclude", "include", "exclusions"),
        *("list", "export"),
    }
    assert parser.parse_args(["push"]).own and not hasattr(parser.parse_args(["push"]), "run")
    for name in ("pull", "labels", "label", "exclusions", "list", "export"):
        assert parser.parse_args([name]).schema_ref == SCHEMA.schema_ref
    with _serving(tmp_path / "kb") as client:
        client.write(_experience(0), declaration=SCHEMA)
        status = run_command(client, parser.parse_args(["export"]))
    exported = json.loads(capsys.readouterr().out)

    assert status == 0
    assert exported["declaration"] == SCHEMA.to_dict()
    assert exported["state"]
    assert [item["experience"]["id"] for item in exported["items"]] == [_experience(0).id]
    # Without a tool's schema, a pull has none to fall back on.
    with pytest.raises(SystemExit):
        main(["pull"])


def test_the_skill_describes_every_command_by_how_it_is_run() -> None:
    skill = (Path(hyperloom_kb.__file__).parent / "skills/hyperloom-kb/SKILL.md").read_text(encoding="utf-8")
    commands = argparse.ArgumentParser().add_subparsers(dest="command")
    add_commands(commands)

    assert [name for name in commands.choices if f"hyperloom-kb {name}" not in skill] == []
    for text in ("HYPERLOOM_KB_URL", "HYPERLOOM_KB_TOKEN", "python3 -m hyperloom_kb.cli", "add_commands"):
        assert text in skill


def test_without_a_service_to_talk_to_it_exits_2(monkeypatch, capsys) -> None:
    monkeypatch.delenv("HYPERLOOM_KB_URL", raising=False)
    assert main(["health"]) == 2
    assert "HYPERLOOM_KB_URL must be configured" in capsys.readouterr().err

    monkeypatch.setenv("HYPERLOOM_KB_URL", "http://127.0.0.1:1")
    monkeypatch.delenv("HYPERLOOM_KB_TOKEN", raising=False)
    assert main(["health"]) == 2
    assert "HYPERLOOM_KB_TOKEN must be configured" in capsys.readouterr().err
