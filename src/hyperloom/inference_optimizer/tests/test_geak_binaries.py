# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the preflight gate on the GEAK binaries pinned in the environment."""

from __future__ import annotations

import argparse
import json

import pytest

from hyperloom.inference_optimizer.cli import geak_binaries, preflight

NODE, CODEX, CLAUDE = geak_binaries.GEAK_BINARY_ENVS

_REGISTRY = {
    "agents": {
        "claude": {"bin": "claude", "bin_env": CLAUDE},
        "codex": {"bin": "codex", "bin_env": CODEX},
    },
    "profiles": {"claude": {"agent": "claude"}, "codex-gpt56": {"agent": "codex", "model": "openai_gpt56"}},
}


@pytest.fixture(autouse=True)
def _no_pinned_binaries(monkeypatch):
    for name in (*geak_binaries.GEAK_BINARY_ENVS, "GEAK_AGENT_PROFILE", "GEAK_AGENT_AUTO"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEAK_AGENT_BACKEND", "codex")


@pytest.fixture(autouse=True)
def _geak_commands_on_path(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for command in ("node", "codex"):
        _executable(bin_dir, command)
    monkeypatch.setenv("PATH", str(bin_dir))


@pytest.fixture(autouse=True)
def geak_registry(tmp_path, monkeypatch):
    """A minimal GEAK checkout; returns the path of its registry.json."""
    interface = tmp_path / "geak" / "interface"
    (interface / "runtime" / "engine").mkdir(parents=True)
    (interface / "run_e2e.py").write_text("", encoding="utf-8")
    registry = interface / "runtime" / "engine" / "registry.json"
    registry.write_text(json.dumps(_REGISTRY), encoding="utf-8")
    monkeypatch.setenv("GEAK_ROOT", str(tmp_path / "geak"))
    for name in ("GEAK_E2E_RUNNER", "HYPERLOOM_CACHE_DIR"):
        monkeypatch.delenv(name, raising=False)
    return registry


def _executable(tmp_path, name):
    path = tmp_path / name
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_unset_binaries_pass_when_the_default_commands_are_on_path(tmp_path, monkeypatch, capsys):
    _executable(tmp_path, "node")
    _executable(tmp_path, "codex")
    monkeypatch.setenv("PATH", str(tmp_path))

    assert geak_binaries.check_geak_binaries() == {"status": "applied", "skip_reason": None, "target": "GEAK"}
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("name, default", [("GEAK_NODE_BIN", "node"), ("GEAK_CODEX_BIN", "codex")])
def test_an_unset_binary_whose_default_command_is_missing_refuses_to_start(
    tmp_path, monkeypatch, capsys, name, default
):
    for command in ("node", "codex"):
        if command != default:
            _executable(tmp_path, command)
    monkeypatch.setenv("PATH", str(tmp_path))

    with pytest.raises(SystemExit) as excinfo:
        geak_binaries.check_geak_binaries()

    assert excinfo.value.code == 2
    assert f"{name}={default} " in capsys.readouterr().err


def test_pinned_executables_pass(tmp_path, monkeypatch):
    for name in geak_binaries.GEAK_BINARY_ENVS:
        monkeypatch.setenv(name, str(_executable(tmp_path, name.lower())))

    assert geak_binaries.check_geak_binaries()["status"] == "applied"


def test_a_command_name_found_on_path_passes(tmp_path, monkeypatch):
    _executable(tmp_path, "node")
    _executable(tmp_path, "codex")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("GEAK_NODE_BIN", "node")

    assert geak_binaries.check_geak_binaries()["status"] == "applied"


def test_a_command_name_missing_from_path_refuses_to_start(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("GEAK_NODE_BIN", "node")

    with pytest.raises(SystemExit) as excinfo:
        geak_binaries.check_geak_binaries()

    assert excinfo.value.code == 2


@pytest.mark.parametrize("name", geak_binaries.GEAK_BINARY_ENVS)
def test_a_pinned_binary_that_is_missing_refuses_to_start(tmp_path, monkeypatch, capsys, name):
    monkeypatch.setenv(name, str(tmp_path / "gone" / "node"))

    with pytest.raises(SystemExit) as excinfo:
        geak_binaries.check_geak_binaries()

    assert excinfo.value.code == 2
    assert f"{name}={tmp_path / 'gone' / 'node'}" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["", " node "])
def test_a_node_pin_geak_cannot_run_as_is_refuses_to_start(tmp_path, monkeypatch, value):
    _executable(tmp_path, "node")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("GEAK_NODE_BIN", value)

    with pytest.raises(SystemExit) as excinfo:
        geak_binaries.check_geak_binaries()

    assert excinfo.value.code == 2


@pytest.mark.parametrize("name", ["GEAK_CLAUDE_BIN", "GEAK_CODEX_BIN"])
def test_an_empty_agent_pin_falls_back_to_geak_default(monkeypatch, name):
    monkeypatch.setenv(name, "")

    assert geak_binaries.check_geak_binaries()["status"] == "applied"


def test_a_padded_claude_pin_passes_because_geak_strips_it(tmp_path, monkeypatch):
    monkeypatch.setenv("GEAK_CLAUDE_BIN", f"  {_executable(tmp_path, 'claude')}  ")

    assert geak_binaries.check_geak_binaries()["status"] == "applied"


def test_a_pinned_directory_or_non_executable_refuses_to_start(tmp_path, monkeypatch):
    plain = tmp_path / "node"
    plain.write_text("", encoding="utf-8")
    monkeypatch.setenv("GEAK_NODE_BIN", str(tmp_path))
    with pytest.raises(SystemExit):
        geak_binaries.check_geak_binaries()
    monkeypatch.setenv("GEAK_NODE_BIN", str(plain))
    with pytest.raises(SystemExit):
        geak_binaries.check_geak_binaries()


@pytest.mark.parametrize(
    ("env", "openai_only", "required"),
    [
        ({}, False, []),
        ({}, True, [NODE, CODEX]),
        ({"GEAK_AGENT_AUTO": "0"}, True, []),
        ({"GEAK_AGENT_BACKEND": "claude"}, True, [NODE, CLAUDE]),
        ({"GEAK_AGENT_BACKEND": "codex"}, False, [NODE, CODEX]),
        ({"GEAK_AGENT_PROFILE": "codex-gpt56"}, False, [NODE, CODEX]),
        ({"GEAK_AGENT_PROFILE": "claude"}, False, [NODE, CLAUDE]),
        ({"GEAK_AGENT_PROFILE": "claude"}, True, [NODE, CLAUDE]),
        ({"GEAK_AGENT_BACKEND": "claude", "GEAK_AGENT_PROFILE": "codex-gpt56"}, False, [NODE, CLAUDE]),
        ({"GEAK_AGENT_BACKEND": "codex", "GEAK_AGENT_PROFILE": "claude"}, False, [NODE, CODEX]),
    ],
)
def test_only_the_binaries_of_the_backend_geak_selects_are_required(
    tmp_path, monkeypatch, capsys, env, openai_only, required
):
    """With no backend selected GEAK keeps its native Claude path, which runs neither Node nor Codex."""
    monkeypatch.delenv("GEAK_AGENT_BACKEND")
    monkeypatch.setattr(geak_binaries, "is_openai_only", lambda: openai_only)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("PATH", str(tmp_path))

    if not required:
        assert geak_binaries.check_geak_binaries()["status"] == "applied"
        return
    with pytest.raises(SystemExit):
        geak_binaries.check_geak_binaries()
    err = capsys.readouterr().err
    assert [name for name in geak_binaries.GEAK_BINARY_ENVS if f"{name}=" in err] == required


@pytest.mark.parametrize(
    ("env", "command"),
    [({"GEAK_AGENT_PROFILE": "codex-gpt56"}, "codex"), ({"GEAK_AGENT_BACKEND": "claude"}, "claude")],
)
def test_the_selected_agents_cli_on_path_passes(tmp_path, monkeypatch, env, command):
    monkeypatch.delenv("GEAK_AGENT_BACKEND")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    _executable(tmp_path, "node")
    _executable(tmp_path, command)
    monkeypatch.setenv("PATH", str(tmp_path))

    assert geak_binaries.check_geak_binaries()["status"] == "applied"


def test_a_pinned_cli_of_the_selected_agent_replaces_its_default_command(tmp_path, monkeypatch):
    monkeypatch.delenv("GEAK_AGENT_BACKEND")
    monkeypatch.setenv("GEAK_AGENT_PROFILE", "codex-gpt56")
    monkeypatch.setenv(CODEX, str(_executable(tmp_path, "my-codex")))
    _executable(tmp_path, "node")
    monkeypatch.setenv("PATH", str(tmp_path))

    assert geak_binaries.check_geak_binaries()["status"] == "applied"


@pytest.mark.parametrize("content", [None, "not json", json.dumps({"agents": {}, "profiles": {}}), json.dumps([])])
@pytest.mark.parametrize("env", [{"GEAK_AGENT_PROFILE": "codex-gpt56"}, {"GEAK_AGENT_BACKEND": "claude"}])
def test_a_registry_that_cannot_name_the_agent_refuses_to_start(monkeypatch, capsys, geak_registry, env, content):
    monkeypatch.delenv("GEAK_AGENT_BACKEND")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    if content is None:
        geak_registry.unlink()
    else:
        geak_registry.write_text(content, encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        geak_binaries.check_geak_binaries()

    assert excinfo.value.code == 2
    assert next(iter(env.values())) in capsys.readouterr().err


def test_a_profile_the_registry_does_not_define_refuses_to_start(monkeypatch, capsys):
    monkeypatch.delenv("GEAK_AGENT_BACKEND")
    monkeypatch.setenv("GEAK_AGENT_PROFILE", "no-such-profile")

    with pytest.raises(SystemExit) as excinfo:
        geak_binaries.check_geak_binaries()

    assert excinfo.value.code == 2
    assert "no-such-profile" in capsys.readouterr().err


def test_a_missing_geak_checkout_refuses_to_start_when_an_agent_is_selected(tmp_path, monkeypatch):
    monkeypatch.setenv("GEAK_ROOT", str(tmp_path / "absent"))

    with pytest.raises(SystemExit) as excinfo:
        geak_binaries.check_geak_binaries()

    assert excinfo.value.code == 2


def test_the_registry_is_not_read_when_no_agent_is_selected(tmp_path, monkeypatch):
    monkeypatch.delenv("GEAK_AGENT_BACKEND")
    monkeypatch.setattr(geak_binaries, "is_openai_only", lambda: False)
    monkeypatch.setenv("GEAK_ROOT", str(tmp_path / "absent"))

    assert geak_binaries.check_geak_binaries()["status"] == "applied"


def _checked_steps(monkeypatch, *, no_kernel: bool) -> list[str]:
    steps: list[str] = []
    monkeypatch.setattr(preflight, "_run_install_step", lambda _event, *, step_id, **_k: steps.append(step_id))
    monkeypatch.setattr(preflight, "_record_install_step", lambda *_a, **_k: None)
    preflight._check_kernel_and_tracelens_clis(None, argparse.Namespace(no_kernel=no_kernel, enable_roofline=True))
    return steps


def test_geak_binaries_are_checked_when_geak_is_the_kernel_backend(monkeypatch):
    monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER", raising=False)

    assert "check_geak_binaries" in _checked_steps(monkeypatch, no_kernel=False)


def test_forge_and_no_kernel_runs_skip_the_geak_check(monkeypatch):
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")
    assert "check_geak_binaries" not in _checked_steps(monkeypatch, no_kernel=False)

    monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER")
    assert "check_geak_binaries" not in _checked_steps(monkeypatch, no_kernel=True)
