# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The hosted entry: session directory, resume, exit.json, the environment a child sees, the multibranch hand-over."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom.hosted import context, entry, host_info
from hyperloom.hosted.host_info import Var
from hyperloom.hosted.tests.test_campaign import FULL, MINIMAL

_SECRETS = {name: f"sentinel-secret-{index}" for index, name in enumerate(host_info.SECRET_VARS)}
_IMAGE = "registry.example/sglang:v0.5.4-rocm"


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A complete host environment in ``os.environ`` and the paths it names."""
    state, work = tmp_path / "state", tmp_path / "work"
    state.mkdir()
    work.mkdir()
    spec = tmp_path / "campaign.json"
    spec.write_text(json.dumps({**MINIMAL, "framework": {"name": "vllm", "image": _IMAGE}}))
    values = {
        Var.INPUT_FILE: str(spec),
        Var.RUN_ID: "run-1",
        Var.RUN_CREATED_AT: "2026-10-23T08:30:05Z",
        Var.STATE_DIR: str(state),
        Var.GRANT_LIMITS: json.dumps({"max_gpus": 8, "gpu_hours": 12}),
        Var.SANDBOX_TEMPLATES: json.dumps({"runner": {"1": "r1"}}),
        Var.E2B_API_URL: "https://api.sandbox.example.test",
        Var.MCP_PORT: "8765",
        "USER_DATA_PATH": str(work),
        "ANTHROPIC_API_KEY": "gateway-key",
        "ANTHROPIC_BASE_URL": "https://gateway.example.test",
        **_SECRETS,
    }
    for name in (*(v for k, v in vars(Var).items() if not k.startswith("_")), entry.ENV_IMAGE):
        monkeypatch.delenv(name, raising=False)

    def inject():
        """What the host does at every (re)start of the application: the same environment again."""
        context.clear()
        for name, value in values.items():
            monkeypatch.setenv(name, value)

    inject()
    yield type("Run", (), {"state": state, "work": work, "spec": spec, "restart": staticmethod(inject)})
    context.clear()


class FakeOptimize:
    """Stands in for ``cli.main``: records what it was called with and what the process looked like then."""

    def __init__(self, monkeypatch, *, returncode=0, writes_session=True, stop_reason=None, effect=None):
        self.calls: list[list[str]] = []
        self.environments: list[dict[str, str]] = []
        self.runs: list[context.HostedRun | None] = []
        self._opts = (returncode, writes_session, stop_reason, effect)
        monkeypatch.setattr(entry, "_optimize_main", self)

    def __call__(self, argv):
        returncode, writes_session, stop_reason, effect = self._opts
        self.calls.append(list(argv))
        self.environments.append(dict(os.environ))
        self.runs.append(context.current())
        session = Path(argv[1].partition("=")[2] if argv[1].startswith("--resume-from") else self._session_dir(argv))
        if writes_session:  # what a real fresh session leaves behind
            (session / "manifest.json").write_text("{}")
            state = {"stop_reason": stop_reason} if stop_reason else {}
            (session / "state.json").write_text(json.dumps(state))
        if effect is not None:
            effect(argv)
        return returncode

    @staticmethod
    def _session_dir(argv):
        return next(a for a in argv if a.startswith("--session-dir=")).partition("=")[2]


def _exit(run):
    return json.loads((run.state / "exit.json").read_text())


def _session_json(run):
    return json.loads((run.state / "session.json").read_text())["session_dir"]


# -- a fresh run -------------------------------------------------------------------------------------------------
def test_a_fresh_run_gets_its_session_dir_the_plug_profile_and_a_private_archive(run, monkeypatch):
    fake = FakeOptimize(monkeypatch, stop_reason="target_reached")
    assert entry.main(["run"]) == 0
    (argv,) = fake.calls
    session = _session_json(run)
    assert session.startswith(str(run.work / "m") + "/20261023T083005Z-")
    assert argv[0] == "optimize" and f"--session-dir={session}" in argv and "--plug-profile=hosted-gpu" in argv
    assert not [a for a in argv if a.startswith("--resume-from")]
    assert _exit(run) == {"final": True, "returncode": 0, "stop_reason": "target_reached"}
    archive = Path(session) / "archive"
    assert stat.S_IMODE(archive.stat().st_mode) == 0o700 and archive.stat().st_uid == os.getuid()
    assert context.current().archive_root == archive


def test_session_json_is_written_before_optimize_starts(run, monkeypatch):
    seen = []
    FakeOptimize(monkeypatch, effect=lambda argv: None)
    monkeypatch.setattr(entry, "_optimize_main", lambda argv: seen.append((run.state / "session.json").exists()) or 0)
    entry.main(["run"])
    assert seen == [True]


# -- resume ------------------------------------------------------------------------------------------------------
def test_a_second_start_of_the_same_run_resumes_the_same_directory(run, monkeypatch):
    fake = FakeOptimize(monkeypatch, returncode=1)  # first leg leaves a session behind, then the process dies
    entry.main(["run"])
    first = _session_json(run)
    run.restart()
    entry.main(["run"])
    assert [a for a in fake.calls[1] if a.startswith("--session-dir")] == []
    assert fake.calls[1] == ["optimize", f"--resume-from={first}", "--plug-profile=hosted-gpu"]
    assert _session_json(run) == first


def test_a_different_run_never_reuses_another_runs_directory(run, monkeypatch):
    fake = FakeOptimize(monkeypatch)
    entry.main(["run"])
    first = _session_json(run)
    run.restart()
    monkeypatch.setenv(Var.RUN_ID, "run-2")
    entry.main(["run"])
    assert _session_json(run) != first
    assert not [a for a in fake.calls[1] if a.startswith("--resume-from")]


@pytest.mark.parametrize("present", [("manifest.json",), ("state.json",), ()])
def test_a_directory_with_only_part_of_a_session_is_started_fresh(run, monkeypatch, present):
    fake = FakeOptimize(monkeypatch, writes_session=False)
    entry.main(["run"])
    session = Path(_session_json(run))
    for name in present:
        (session / name).write_text("{}")
    run.restart()
    entry.main(["run"])
    assert any(a.startswith("--session-dir=") for a in fake.calls[1])
    assert not any(a.startswith("--resume-from") for a in fake.calls[1])


# -- exit.json ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("returncode", [1, 2, 130])
def test_a_normal_end_with_a_failing_return_code_is_still_final(run, monkeypatch, returncode):
    FakeOptimize(monkeypatch, returncode=returncode, stop_reason="budget_exhausted")
    assert entry.main(["run"]) == returncode
    assert _exit(run) == {"final": True, "returncode": returncode, "stop_reason": "budget_exhausted"}


def test_a_system_exit_from_optimize_is_a_normal_end(run, monkeypatch):
    def refuse(argv):
        raise SystemExit(2)

    monkeypatch.setattr(entry, "_optimize_main", refuse)
    assert entry.main(["run"]) == 2
    assert _exit(run) == {"final": True, "returncode": 2, "stop_reason": None}


def test_a_crash_leaves_no_final(run, monkeypatch):
    def crash(argv):
        raise RuntimeError("boom")

    monkeypatch.setattr(entry, "_optimize_main", crash)
    with pytest.raises(RuntimeError):
        entry.main(["run"])
    assert not (run.state / "exit.json").exists()
    assert (run.state / "session.json").exists()


def test_a_watchdog_restart_ends_a_leg_not_the_run(run, monkeypatch):
    FakeOptimize(monkeypatch, returncode=1, stop_reason="supervisor_restart_requested")
    entry.main(["run"])
    assert _exit(run)["final"] is False


def _break_spec(run, doc):
    run.spec.write_text(json.dumps(doc))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r, m: _break_spec(r, {**MINIMAL, "budget": {}}), "budget.max_hours is required"),
        (lambda r, m: r.spec.write_text("{not json"), "is not valid JSON"),
        (
            lambda r, m: _break_spec(r, {**MINIMAL, "placement": {"profile": "hosted-cpu"}}),
            "does not match profile 'hosted-cpu'",
        ),
        (lambda r, m: m.delenv(Var.RUN_ID), Var.RUN_ID),
        (lambda r, m: m.setenv(Var.MCP_PORT, "99999"), Var.MCP_PORT),
        (lambda r, m: m.setenv("USER_DATA_PATH", "relative"), "USER_DATA_PATH must be set to an absolute path"),
    ],
)
def test_a_refused_request_is_final_with_return_code_2_and_never_reaches_optimize(run, monkeypatch, mutate, message):
    fake = FakeOptimize(monkeypatch)
    mutate(run, monkeypatch)
    assert entry.main(["run"]) == 2
    written = _exit(run)
    assert (written["final"], written["returncode"]) == (True, 2) and message in written["error"]
    assert fake.calls == []
    assert not any(secret in json.dumps(written) for secret in _SECRETS.values())


def test_a_refusal_the_state_directory_cannot_hold_is_still_return_code_2(run, monkeypatch, capsys):
    monkeypatch.delenv(Var.STATE_DIR)
    assert entry.main(["run"]) == 2
    assert Var.STATE_DIR in capsys.readouterr().err


def test_anything_but_run_is_a_usage_error_that_writes_nothing(run, capsys):
    assert entry.main(["walk"]) == 2 and entry.main([]) == 2
    assert not (run.state / "exit.json").exists()


# -- what optimize and its children see --------------------------------------------------------------------------
def test_no_host_only_input_reaches_optimize_or_a_real_child_process(run, monkeypatch):
    def spawn_child(argv):
        out = subprocess.run(
            [sys.executable, "-c", "import json, os; print(json.dumps(dict(os.environ)))"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        seen.append(json.loads(out))

    seen: list[dict[str, str]] = []
    fake = FakeOptimize(monkeypatch, effect=spawn_child)
    entry.main(["run"])
    for environment in (fake.environments[0], seen[0]):
        assert not set(host_info.HOST_ONLY_VARS) & set(environment)
        assert not [v for v in environment.values() if "sentinel-secret" in v]
        assert environment["USER_DATA_PATH"] == str(run.work)
        assert environment["ANTHROPIC_API_KEY"] == "gateway-key"  # the model gateway is not a host-only input
        assert environment[entry.ENV_IMAGE] == _IMAGE


def test_the_host_values_travel_in_code_to_the_plug_host(run, monkeypatch):
    fake = FakeOptimize(monkeypatch)
    entry.main(["run"])
    (hosted,) = fake.runs
    assert hosted.plug_lock == {"measurement": {"impl": "local"}}
    assert hosted.host_info.sandbox_keys.measurement.reveal() == _SECRETS[Var.SANDBOX_KEY_MEASUREMENT]
    assert hosted.host_info.mcp_token.reveal() == _SECRETS[Var.MCP_TOKEN]
    assert hosted.host_info.sandbox_templates == {"runner": {"1": "r1"}}
    assert dataclass_dict(hosted.host_info.grant_limits) == {
        "max_sandboxes": None,
        "max_gpus": 8,
        "gpu_hours": 12,
        "max_runtime_sec": None,
    }


def dataclass_dict(value):
    import dataclasses

    return dataclasses.asdict(value)


def test_the_grant_limits_are_only_logged(run, monkeypatch, capsys):
    FakeOptimize(monkeypatch)
    entry.main(["run"])
    assert "grant limits" in capsys.readouterr().out


# -- the multibranch hand-over -----------------------------------------------------------------------------------
def _multibranch(run, **placement):
    doc = {**FULL, "flow": "multibranch", "placement": {"profile": "hosted-gpu", "measurement": "hosted", **placement}}
    doc["multibranch"] = ["opaque", {"never": "validated"}, 3]
    run.spec.write_text(json.dumps(doc))


def test_multibranch_execs_into_its_entry_with_the_environment_unchanged(run, monkeypatch):
    _multibranch(run)
    before = dict(os.environ)
    calls = []
    monkeypatch.setattr(entry.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(entry.os, "execvp", lambda file, argv: calls.append((file, argv, dict(os.environ))))
    fake = FakeOptimize(monkeypatch)
    entry.main(["run"])
    assert [(c[0], c[1]) for c in calls] == [("hyperloom-multibranch", ["hyperloom-multibranch", "hosted", "run"])]
    assert calls[0][2] == before and Var.RUN_ID in before
    assert fake.calls == [] and not (run.state / "session.json").exists() and not (run.state / "exit.json").exists()
    assert os.environ.get(entry.ENV_IMAGE) is None


def test_multibranch_without_its_command_is_final_with_return_code_2(run, monkeypatch):
    _multibranch(run)
    monkeypatch.setattr(entry.shutil, "which", lambda name: None)
    assert entry.main(["run"]) == 2
    written = _exit(run)
    assert (written["final"], written["returncode"]) == (True, 2) and "hyperloom-multibranch" in written["error"]


def test_multibranch_is_still_refused_when_the_shared_part_of_the_spec_is_wrong(run, monkeypatch):
    _multibranch(run)
    run.spec.write_text(json.dumps({**json.loads(run.spec.read_text()), "budget": {}}))
    monkeypatch.setattr(entry.os, "execvp", lambda file, argv: pytest.fail("must not exec"))
    assert entry.main(["run"]) == 2
    assert _exit(run)["final"] is True
