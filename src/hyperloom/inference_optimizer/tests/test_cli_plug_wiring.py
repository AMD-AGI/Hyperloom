# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The plug flags of ``optimize``: a run that selects nothing is unchanged, one that selects something is recorded."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

import hyperloom.inference_optimizer.cli as ocli
from hyperloom.inference_optimizer.cli.bootstrap import prepare_plug_host, running_plug_host
from hyperloom.inference_optimizer.cli.parser import _build_parser
from hyperloom.plugs import PlugError, active
from hyperloom.plugs.events_local import LocalProgressEvents


def _args(*extra: str):
    return _build_parser().parse_args(["optimize", "--model", "/models/m", *extra])


@pytest.fixture(autouse=True)
def _no_plug_env(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_MANIFEST", raising=False)
    monkeypatch.delenv("HYPERLOOM_MANIFEST_OVERRIDES", raising=False)


def test_the_plug_flags_default_to_nothing_selected() -> None:
    args = _args()
    assert (args.plug_profile, args.plug_manifest, args.plug, args.plug_config) == (None, None, [], [])


def test_the_plug_flags_parse() -> None:
    args = _args("--profile", "hosted-gpu", "--manifest", "m.json", "--plug", "events=a", "--plug-config", "events.k=1")
    assert (args.plug_profile, args.plug_manifest, args.plug, args.plug_config) == (
        "hosted-gpu",
        "m.json",
        ["events=a"],
        ["events.k=1"],
    )


def test_profile_osl_still_parses_as_its_own_flag() -> None:
    assert _args("--profile-osl", "128").profile_osl == 128


def test_a_run_that_selects_nothing_writes_and_prints_nothing(tmp_path: Path, capsys) -> None:
    host = prepare_plug_host(_args(), tmp_path)
    assert list(tmp_path.iterdir()) == []
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")
    assert {k: v["impl"] for k, v in host.describe()["sockets"].items()} == {"events": "local"}


def test_a_run_that_selects_nothing_starts_only_the_local_events_plug(tmp_path: Path) -> None:
    host = prepare_plug_host(_args(), tmp_path)
    with running_plug_host(host):
        assert active() is host
        assert isinstance(host.get("events"), LocalProgressEvents)
    with pytest.raises(PlugError, match="socket 'events' is not started"):
        host.get("events")


def test_a_selection_is_recorded_in_the_session_and_announced(tmp_path: Path, capsys) -> None:
    host = prepare_plug_host(_args("--plug-config", "events.level=debug"), tmp_path)
    recorded = json.loads((tmp_path / "plugs.resolved.json").read_text())
    assert recorded["sockets"]["events"] == {"impl": "local", "config": {"level": "debug"}}
    assert recorded["layers"]["events"] == "cli"
    assert recorded["fingerprint"] == host.fingerprint()
    assert f"fingerprint={host.fingerprint()}" in capsys.readouterr().out


def test_a_resume_reads_the_recorded_selection_back(tmp_path: Path) -> None:
    prepare_plug_host(_args("--plug-config", "events.level=debug"), tmp_path)
    resumed = prepare_plug_host(_args(), tmp_path)
    events = resumed.describe()["sockets"]["events"]
    assert (events["config"], events["layer"]) == ({"level": "debug"}, "resolved")


def test_the_manifest_and_override_environment_variables_are_layers(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"schema": "hyperloom.plugs/v1", "sockets": {"events": {"config": {"a": 1}}}}))
    overrides = tmp_path / "o.json"
    overrides.write_text(json.dumps({"schema": "hyperloom.plugs/v1", "sockets": {"events": {"config": {"b": 2}}}}))
    monkeypatch.setenv("HYPERLOOM_MANIFEST", str(manifest))
    monkeypatch.setenv("HYPERLOOM_MANIFEST_OVERRIDES", str(overrides))
    session = tmp_path / "session"
    session.mkdir()
    events = prepare_plug_host(_args(), session).describe()["sockets"]["events"]
    assert (events["config"], events["layer"]) == ({"a": 1, "b": 2}, "overrides")


def test_a_profile_naming_an_unset_variable_is_refused(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.delenv("HYPERLOOM_EVENTS_DIR", raising=False)
    with pytest.raises(SystemExit) as exc:
        prepare_plug_host(_args("--profile", "hosted-cpu"), tmp_path)
    assert exc.value.code == 2
    assert "environment variable HYPERLOOM_EVENTS_DIR is not set" in capsys.readouterr().err


def test_a_profile_whose_implementations_are_not_installed_is_refused_before_anything_starts(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    monkeypatch.setenv("HYPERLOOM_EVENTS_DIR", str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        prepare_plug_host(_args("--profile", "hosted-cpu"), tmp_path)
    assert exc.value.code == 2
    assert (
        "plug configuration refused: socket 'events' has no implementation 'hosted'; available: local"
        in capsys.readouterr().err
    )


def test_an_undefined_socket_is_refused_with_exit_2(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        prepare_plug_host(_args("--plug", "warp=x"), tmp_path)
    assert exc.value.code == 2
    assert "cli: unknown socket 'warp'" in capsys.readouterr().err


def test_a_damaged_resolved_file_stops_the_resume(tmp_path: Path, capsys) -> None:
    (tmp_path / "plugs.resolved.json").write_text("{oops")
    with pytest.raises(SystemExit) as exc:
        prepare_plug_host(_args(), tmp_path)
    assert exc.value.code == 2
    assert "fix or remove the file to resume" in capsys.readouterr().err


def test_a_resumed_file_naming_a_missing_implementation_stops_the_resume(tmp_path: Path, capsys) -> None:
    (tmp_path / "plugs.resolved.json").write_text(
        json.dumps({"schema": "hyperloom.plugs/v1", "sockets": {"events": {"impl": "ghost"}}})
    )
    with pytest.raises(SystemExit):
        prepare_plug_host(_args(), tmp_path)
    assert "socket 'events' has no implementation 'ghost'" in capsys.readouterr().err


def test_optimize_resolves_the_plugs_after_the_session_opens_and_runs_the_coordinator_inside_the_host() -> None:
    tree = ast.parse(inspect.getsource(ocli._run_optimize))
    first_line = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            first_line.setdefault(ast.unparse(node.func), node.lineno)
    session_opened = max(first_line["_seed_fresh_session"], first_line["_resume_session"])
    assert session_opened < first_line["prepare_plug_host"] < first_line["_bootstrap_recipe_kb"]
    withs = [n for n in ast.walk(tree) if isinstance(n, ast.With) and "running_plug_host" in ast.unparse(n.items[0])]
    assert len(withs) == 1
    assert any("coordinator.run" in ast.unparse(c) for c in ast.walk(withs[0]) if isinstance(c, ast.Call))
