# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The hosted entry: session directory, resume, exit.json, the environment a child sees, the multibranch hand-over."""

from __future__ import annotations

import json
import logging
import os
import socket
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from hyperloom_common import bwrap
from hyperloom_common.bwrap import BwrapProbe
from hyperloom_common.tests._bwrap_host import usable_bwrap

from hyperloom_common.tests.shared_accounts import theirs
from hyperloom.common import agent_isolation, confinement
from hyperloom.common.agent_isolation import HOST_LAYER_ENV, NO_LAYER_BWRAP, RAY_ACTOR_PATH, IsolationRefused
from hyperloom.common.env_safety import RunAs
from hyperloom.hosted import entry, host_info
from hyperloom.hosted.host_info import Var
from hyperloom.hosted.tests.test_campaign import FULL, MINIMAL
from hyperloom.inference_optimizer.cli import bootstrap
from hyperloom.inference_optimizer.cli.bootstrap import set_hosted_plug_inputs
from hyperloom.plugs.manifest import LOCKABLE_SOCKETS, load_builtin_profile

INSTALL_REAL = Path(entry.INSTALL_SCRIPT)
_SECRETS = {name: f"sentinel-secret-{index}" for index, name in enumerate(host_info.SECRET_VARS)}
_IMAGE = "registry.example/sglang:v0.5.4-rocm"
_LAYER_BWRAP_VERSION = (0, 13, 1)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Isolation:
    """Stands in for the bubblewrap probe the entry's isolation check runs, and records what it was asked about and
    whether the run had started anything then. Answers ``problems`` (none: a working bubblewrap) for any path."""

    def __init__(self) -> None:
        self.asked: list[str | None] = []
        self.state: Path | None = None
        self.started_then: list[bool] = []
        self.problems: tuple[str, ...] = ()
        self.transient = False

    def probe(self, path: str | None) -> BwrapProbe:
        self.asked.append(path)
        if self.state is not None:
            self.started_then.append(any(self.state.iterdir()))
        return BwrapProbe(path, _LAYER_BWRAP_VERSION, self.problems, transient=self.transient)


@pytest.fixture(autouse=True)
def isolation(monkeypatch) -> Isolation:
    """A process that is not yet marked hosted, with no specialist account configured and a registry of private paths
    of its own, whose isolation check asks :class:`Isolation` instead of a real bubblewrap."""
    monkeypatch.setattr(agent_isolation, "_STATE", agent_isolation._Process())
    monkeypatch.setattr(confinement, "PRIVATE_PATHS", confinement.PrivatePaths())
    monkeypatch.delenv("HYPERLOOM_SPECIALIST_USER", raising=False)
    stand_in = Isolation()
    monkeypatch.setattr(agent_isolation, "bwrap_probe", stand_in.probe)
    return stand_in


@pytest.fixture
def run(tmp_path, monkeypatch, isolation):
    """A complete host environment in ``os.environ`` and the paths it names, with the hosting layer's directory in
    ``HYPERLOOM_HOST_LAYER``."""
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
        Var.MCP_PORT: str(_free_port()),  # a fixed port collides with whatever else listens on this host
        HOST_LAYER_ENV: str(tmp_path / "layer"),
        "USER_DATA_PATH": str(work),
        "ANTHROPIC_API_KEY": "gateway-key",
        "ANTHROPIC_BASE_URL": "https://gateway.example.test",
        **_SECRETS,
    }
    for name in (*(v for k, v in vars(Var).items() if not k.startswith("_")), entry.ENV_IMAGE, entry.ENV_RUNTIME_DIR):
        # Set first so that monkeypatch restores it: what the entry sets in os.environ must not outlive the test.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    container_tmp = tmp_path / "container-tmp"
    container_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(container_tmp))  # the container's own disk

    def inject():
        """What the host does at every (re)start of the application: the same environment again."""
        set_hosted_plug_inputs(None)
        for name, value in values.items():
            monkeypatch.setenv(name, value)

    monkeypatch.setattr(sys, "path", list(sys.path))  # the entry edits it
    installer = tmp_path / "install.sh"
    installer.write_text("#!/bin/bash\necho stub installer ran\n")
    monkeypatch.setattr(entry, "INSTALL_SCRIPT", installer)  # the real installer is never run by a unit test
    inject()
    isolation.state = state
    yield type(
        "Run",
        (),
        {
            "state": state,
            "work": work,
            "spec": spec,
            "restart": staticmethod(inject),
            "installer": installer,
            "container_tmp": container_tmp,
            "layer": tmp_path / "layer",
        },
    )
    set_hosted_plug_inputs(None)


class FakeOptimize:
    """Stands in for ``cli.main``: records what it was called with and what the process looked like then."""

    def __init__(self, monkeypatch, *, returncode=0, writes_session=True, stop_reason=None, effect=None):
        self.calls: list[list[str]] = []
        self.environments: list[dict[str, str]] = []
        self.runs: list[bootstrap.HostedPlugInputs | None] = []
        self._opts = (returncode, writes_session, stop_reason, effect)
        monkeypatch.setattr(entry, "_optimize_main", self)

    def __call__(self, argv):
        returncode, writes_session, stop_reason, effect = self._opts
        self.calls.append(list(argv))
        self.environments.append(dict(os.environ))
        self.runs.append(bootstrap._hosted_inputs)
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
def test_a_fresh_run_gets_its_session_dir_the_plug_profile_and_a_group_shared_archive(run, monkeypatch):
    run.work.chmod(0o2775)  # storage set up to be shared: setgid, group-writable, closed to others
    fake = FakeOptimize(monkeypatch, stop_reason="target_reached")
    assert entry.main(["run"]) == 0
    (argv,) = fake.calls
    session = _session_json(run)
    assert session.startswith(str(run.work / "m") + "/20261023T083005Z-")
    assert argv[0] == "optimize" and f"--session-dir={session}" in argv and "--plug-profile=hosted-gpu" in argv
    assert not [a for a in argv if a.startswith("--resume-from")]
    assert _exit(run) == {"final": True, "returncode": 0, "stop_reason": "target_reached"}
    archive = Path(session) / "archive"
    assert stat.S_IMODE(archive.stat().st_mode) == 0o2770 and archive.stat().st_uid == os.getuid()
    assert [stat.S_IMODE(path.stat().st_mode) for path in (Path(session).parent, Path(session))] == [0o2775, 0o2775]


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
    from hyperloom.inference_optimizer.cli.parser import _build_parser

    FakeOptimize(monkeypatch, writes_session=False)
    entry.main(["run"])
    session = Path(_session_json(run))
    for name in present:
        (session / name).write_text("{}")
    run.restart()
    parsed = []
    monkeypatch.setattr(entry, "_optimize_main", lambda argv: parsed.append(_build_parser().parse_args(argv)) or 0)
    assert entry.main(["run"]) == 0  # the real parser refuses a directory that still holds a session file
    assert _exit(run)["returncode"] == 0
    assert parsed[0].session_dir == session and getattr(parsed[0], "resume_from", None) is None


# -- plug wiring ------------------------------------------------------------------------------------------------
def _start_plug_host(profile):
    """Stands in for ``cli.main``: runs the real plug resolution and host start the way ``optimize`` does."""
    import argparse

    from hyperloom.inference_optimizer.cli import bootstrap
    from hyperloom.plugs import progress

    def probe(argv):
        manifest = bootstrap.resolve_plug_manifest(argparse.Namespace(plug_profile=profile))
        host = bootstrap.open_plug_host(
            manifest,
            Path(
                json.loads((bootstrap._hosted_inputs.host_info.state_dir / "session.json").read_text())["session_dir"]
            ),
        )
        with bootstrap.running_plug_host(host):
            progress.warn("probe", "first event")
        return 0

    return probe


def test_a_hosted_run_starts_the_events_implementation_from_the_host_inputs(run, monkeypatch):
    monkeypatch.setattr(entry, "_optimize_main", _start_plug_host("hosted-gpu"))
    assert entry.main(["run"]) == 0
    (segment,) = (run.state / "events").glob("*.jsonl")
    (event,) = [json.loads(line) for line in segment.read_text().splitlines()]
    assert event["source"] == "hyperloom/run-1" and event["data"] == {"code": "probe", "message": "first event"}
    assert Var.EVENTS_DIR not in os.environ


def test_outside_a_hosted_run_the_hosted_events_implementation_has_no_host_inputs(run):
    import argparse

    from hyperloom.inference_optimizer.cli import bootstrap
    from hyperloom.plugs.errors import PlugError

    manifest = bootstrap.resolve_plug_manifest(argparse.Namespace(plug_profile="hosted-gpu"))
    host = bootstrap.open_plug_host(manifest, run.work)
    with pytest.raises(PlugError, match="needs host inputs"), bootstrap.running_plug_host(host):
        pass


def test_the_plug_inputs_do_not_outlive_the_run(run, monkeypatch):
    from hyperloom.inference_optimizer.cli import bootstrap

    seen = []
    monkeypatch.setattr(entry, "_optimize_main", lambda argv: seen.append(bootstrap._hosted_inputs) or 0)
    entry.main(["run"])
    assert seen[0] is not None and bootstrap._hosted_inputs is None

    def crash(argv):
        raise RuntimeError("boom")

    monkeypatch.setattr(entry, "_optimize_main", crash)
    run.restart()
    with pytest.raises(RuntimeError):
        entry.main(["run"])
    assert bootstrap._hosted_inputs is None


# -- the lock follows the profile --------------------------------------------------------------------------------
def _spec(measurement="local"):
    from hyperloom.hosted.campaign import parse_campaign

    return parse_campaign({**MINIMAL, "placement": {"profile": "hosted-gpu", "measurement": measurement}})


def test_the_lock_is_the_lockable_sockets_the_profile_declares(monkeypatch):
    monkeypatch.setattr(entry, "load_builtin_profile", lambda name: {"sockets": {"events": {"impl": "hosted"}}})
    assert entry._plug_lock(_spec()) == {"events": {"impl": "hosted"}}


def test_a_profile_declaring_several_sockets_locks_each_lockable_one(monkeypatch):
    sockets = {
        "measurement": {"impl": "local"},
        "agent_runner": {"impl": "local"},  # not lockable: left to the profile layer
        "events": {"impl": "hosted"},
        "gpu_token": {"impl": "none"},
    }
    monkeypatch.setattr(entry, "load_builtin_profile", lambda name: {"sockets": sockets})
    assert entry._plug_lock(_spec()) == {
        "measurement": {"impl": "local"},
        "events": {"impl": "hosted"},
        "gpu_token": {"impl": "none"},
    }


def test_a_declared_measurement_must_agree_with_the_spec(monkeypatch):
    monkeypatch.setattr(entry, "load_builtin_profile", lambda name: {"sockets": {"measurement": {"impl": "hosted"}}})
    with pytest.raises(entry.Refused, match="does not match profile"):
        entry._plug_lock(_spec())


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
        (
            lambda r, m: m.setenv("HYPERLOOM_LEDGER_DIR", "relative/ledgers"),
            "HYPERLOOM_LEDGER_DIR must be an absolute path",
        ),
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


# -- the shared storage, which each leg may write as another account ---------------------------------------------
def test_a_user_data_path_this_account_cannot_write_ends_the_run_finally_naming_it(run, monkeypatch):
    fake = FakeOptimize(monkeypatch)
    run.work.rmdir()
    run.work.write_text("")  # what no account can create a file in
    assert entry.main(["run"]) == 2
    error = _exit(run)["error"]
    assert f"cannot write the shared storage: USER_DATA_PATH {run.work}: ENOTDIR" in error
    assert fake.calls == [] and not (run.state / "session.json").exists()


def test_a_state_dir_this_account_cannot_write_is_said_on_stderr_only(run, monkeypatch, capsys):
    FakeOptimize(monkeypatch)
    probe = entry.writable_problems
    monkeypatch.setattr(
        entry,
        "writable_problems",
        lambda dirs: [f"{name} {path}: EACCES" for name, path in dirs.items() if name == Var.STATE_DIR] or probe(dirs),
    )
    assert entry.main(["run"]) == 2
    assert f"{Var.STATE_DIR} {run.state}: EACCES" in capsys.readouterr().err
    assert list(run.state.iterdir()) == []


def test_a_session_directory_this_account_cannot_write_ends_the_run_finally_naming_it(run, monkeypatch):
    fake = FakeOptimize(monkeypatch)
    (run.work / "m").write_text("")  # where the model's directory of sessions goes
    assert entry.main(["run"]) == 2
    assert "cannot write the session directory " + str(run.work / "m") in _exit(run)["error"]
    assert fake.calls == []


def test_an_archive_an_earlier_leg_left_unusable_ends_the_run_finally_naming_it(run, monkeypatch):
    fake = FakeOptimize(monkeypatch)
    entry.main(["run"])
    archive = Path(_session_json(run)) / "archive"
    archive.rmdir()
    archive.symlink_to(run.work)
    run.restart()
    assert entry.main(["run"]) == 2
    assert _exit(run) == {
        "final": True,
        "returncode": 2,
        "error": f"the archive {archive} is not a directory (a link to one is refused too)",
    }
    assert len(fake.calls) == 1


def test_a_link_another_account_put_on_the_way_to_the_session_ends_the_run_finally_naming_it(run, monkeypatch):
    fake = FakeOptimize(monkeypatch)
    entry.main(["run"])
    sessions = Path(_session_json(run)).parent
    sessions.rename(sessions.with_name("moved"))
    sessions.symlink_to(sessions.with_name("moved"))
    theirs(sessions, monkeypatch)
    run.restart()
    assert entry.main(["run"]) == 2
    error = _exit(run)["error"]
    assert "is not followed past a directory another account can write or owns" in error and str(sessions) in error
    assert len(fake.calls) == 1


def test_a_key_directory_without_setgid_is_warned_about_and_one_with_it_is_not(run, monkeypatch, capsys):
    FakeOptimize(monkeypatch)
    entry.main(["run"])
    err = capsys.readouterr().err
    assert f"WARNING: USER_DATA_PATH {run.work} is not setgid" in err and f"chmod g+s {run.state}" in err
    for key in (run.work, run.state):
        key.chmod(0o2775)
    run.restart()
    entry.main(["run"])
    assert "is not setgid" not in capsys.readouterr().err


def test_the_runtime_dir_is_one_of_the_legs_own_on_the_container_disk_unless_the_host_names_one(run, monkeypatch):
    run.installer.write_text(f'#!/bin/bash\necho "$HYPERLOOM_RUNTIME_DIR" >> {run.work}/seen\n')
    fake = FakeOptimize(monkeypatch)
    entry.main(["run"])
    runtime = Path(fake.environments[0][entry.ENV_RUNTIME_DIR])
    assert runtime.parent == run.container_tmp and stat.S_IMODE(runtime.stat().st_mode) == 0o700
    assert (run.work / "seen").read_text().split() == [str(runtime)]
    run.restart()
    monkeypatch.setenv(entry.ENV_RUNTIME_DIR, "/host/chose/this")
    entry.main(["run"])
    assert fake.environments[1][entry.ENV_RUNTIME_DIR] == "/host/chose/this"


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
    declared = load_builtin_profile("hosted-gpu")["sockets"]
    assert hosted.lock == {n: {"impl": v["impl"]} for n, v in declared.items() if n in LOCKABLE_SOCKETS}
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


def test_multibranch_execs_into_its_entry_with_the_environment_unchanged_but_for_path(run, monkeypatch, isolation):
    _multibranch(run)
    before = dict(os.environ)
    calls = []
    monkeypatch.setattr(entry.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(entry.os, "execvp", lambda file, argv: calls.append((file, argv, dict(os.environ))))
    fake = FakeOptimize(monkeypatch)
    entry.main(["run"])
    assert [(c[0], c[1]) for c in calls] == [("hyperloom-multibranch", ["hyperloom-multibranch", "hosted", "run"])]
    bin_dir = str(Path(sys.executable).parent)
    path = calls[0][2].pop("PATH")
    assert path.split(os.pathsep)[0] == bin_dir  # the installer's interpreter resolves the command
    before.pop("PATH", None)
    runtime = calls[0][2].pop(entry.ENV_RUNTIME_DIR)
    assert Path(runtime).parent == run.container_tmp
    assert calls[0][2] == before and Var.RUN_ID in before
    assert fake.calls == [] and not (run.state / "session.json").exists() and not (run.state / "exit.json").exists()
    assert os.environ.get(entry.ENV_IMAGE) is None
    assert isolation.asked == []  # the entry hands over to this command before its isolation check


def _multibranch_order(run, monkeypatch, *, exit_code=0):
    _multibranch(run)
    _recording_installer(run, None, exit_code=exit_code)

    def fake_exec(file, argv):
        with (run.work / "order").open("a") as order:
            order.write("exec\n")

    monkeypatch.setattr(entry.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(entry.os, "execvp", fake_exec)


def test_multibranch_runs_the_installer_before_the_exec_without_host_secrets(run, monkeypatch):
    _multibranch_order(run, monkeypatch)
    run.installer.write_text(
        run.installer.read_text().replace(
            "exit 0", f'echo "${{HYPERLOOM_MCP_TOKEN:-unset}}" >> {run.work}/order\nexit 0'
        )
    )
    entry.main(["run"])
    assert _order(run) == ["installer 0", "unset", "exec"]


def test_multibranch_installer_failure_stops_before_the_exec(run, monkeypatch):
    _multibranch_order(run, monkeypatch, exit_code=7)
    assert entry.main(["run"]) == 2
    written = _exit(run)
    assert (written["final"], written["returncode"]) == (True, 2) and "install.sh failed" in written["error"]
    assert _order(run) == ["installer 0"]


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


# -- the MCP service ---------------------------------------------------------------------------------------------
def _status_call(port: int, token: str) -> dict:
    import http.client

    body = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "get_status", "arguments": {}}}
    )
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request("POST", "/mcp", body=body, headers={"X-Hyperloom-Token": token})
    reply = json.loads(connection.getresponse().read())
    connection.close()
    return json.loads(reply["result"]["content"][0]["text"])


def test_the_host_tools_are_served_while_optimize_runs_and_stopped_after(run, monkeypatch):
    port = _free_port()
    monkeypatch.setenv(Var.MCP_PORT, str(port))
    monkeypatch.setenv(Var.MCP_HOST, "127.0.0.1")
    answers = []

    def during(argv):
        session = Path(next(a for a in argv if a.startswith("--session-dir=")).partition("=")[2])
        (session / "state.json").write_text(json.dumps({"session_id": "s", "phase": "PRELUDE", "stop_reason": ""}))
        answers.append(_status_call(port, _SECRETS[Var.MCP_TOKEN]))

    FakeOptimize(monkeypatch, effect=during)
    assert entry.main(["run"]) == 0
    assert answers[0]["phase"] == "PRELUDE" and answers[0]["session_id"] == "s"
    with pytest.raises(OSError):
        _status_call(port, _SECRETS[Var.MCP_TOKEN])  # the server went down with the run


def test_no_port_means_no_server(run, monkeypatch):
    monkeypatch.delenv(Var.MCP_PORT)
    started = []
    monkeypatch.setattr(entry.McpServer, "start", lambda self: started.append(1))
    FakeOptimize(monkeypatch)
    assert entry.main(["run"]) == 0 and started == []


def test_a_port_that_cannot_be_bound_refuses_the_run(run, monkeypatch):
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        monkeypatch.setenv(Var.MCP_PORT, str(taken.getsockname()[1]))
        monkeypatch.setenv(Var.MCP_HOST, "127.0.0.1")
        fake = FakeOptimize(monkeypatch)
        assert entry.main(["run"]) == entry.RETURNCODE_REFUSED
    assert fake.calls == [] and _exit(run)["final"] is True and "MCP server cannot start" in _exit(run)["error"]


# -- stop requests and the host token ---------------------------------------------------------------------------
def _stop_call(port: int, token: str, reason: str = "operator asked") -> None:
    import http.client

    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "request_stop", "arguments": {"reason": reason}},
        }
    )
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request("POST", "/mcp", body=body, headers={"X-Hyperloom-Token": token})
    assert json.loads(connection.getresponse().read())["result"]["isError"] is False
    connection.close()


def test_a_stop_requested_while_no_coordinator_runs_neither_kills_the_run_nor_is_lost(run, monkeypatch):
    """request_stop with the real SIGTERM, before any SignalDrain exists: the process survives, and the restart ends."""
    port = _free_port()
    monkeypatch.setenv(Var.MCP_PORT, str(port))
    monkeypatch.setenv(Var.MCP_HOST, "127.0.0.1")
    first = FakeOptimize(
        monkeypatch,
        returncode=1,
        stop_reason="supervisor_restart_requested",
        effect=lambda argv: _stop_call(port, _SECRETS[Var.MCP_TOKEN]),
    )
    entry.main(["run"])
    session = Path(_session_json(run))
    assert json.loads((session / "hosted" / "control" / "stop.json").read_text())["reason"] == "operator asked"
    assert _exit(run)["final"] is False  # this leg ended as a watchdog leg: the host would restart it
    run.restart()
    second = FakeOptimize(monkeypatch)
    assert entry.main(["run"]) == 0
    assert second.calls == [] and len(first.calls) == 1  # the restart does not start optimize again
    assert _exit(run) == {"final": True, "returncode": 0, "stop_reason": "signal"}


def test_a_sigterm_without_a_stop_request_keeps_its_default_meaning(tmp_path):
    import signal
    import subprocess
    import sys

    code = (
        "import os, signal, sys; from pathlib import Path; from hyperloom.hosted.entry import _stop_request_guard\n"
        f"with _stop_request_guard(Path({str(tmp_path)!r})):\n"
        "    os.kill(os.getpid(), signal.SIGTERM)\n"
        "    sys.exit(0)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], check=False, timeout=60)
    assert done.returncode == -signal.SIGTERM


def test_the_coordinator_reads_the_stop_request_file(tmp_path):
    import asyncio
    from types import SimpleNamespace

    from hyperloom.orchestrator.loop.coordinator import Coordinator

    stand_in = SimpleNamespace(_stop=asyncio.Event(), session_dir=tmp_path, _signals=None)
    assert Coordinator.stop_requested(stand_in) is False
    (tmp_path / "hosted" / "control").mkdir(parents=True)
    (tmp_path / "hosted" / "control" / "stop.json").write_text("{}")
    assert Coordinator.stop_requested(stand_in) is True


def test_a_host_token_shorter_than_the_server_minimum_refuses_the_run(run, monkeypatch):
    from hyperloom.hosted.mcp_server import MIN_TOKEN_LENGTH

    assert host_info.MIN_MCP_TOKEN_LENGTH == MIN_TOKEN_LENGTH
    monkeypatch.setenv(Var.MCP_TOKEN, "x" * (MIN_TOKEN_LENGTH - 1))
    fake = FakeOptimize(monkeypatch)
    assert entry.main(["run"]) == entry.RETURNCODE_REFUSED
    assert fake.calls == [] and _exit(run)["final"] is True
    assert Var.MCP_TOKEN in _exit(run)["error"] and "x" * MIN_TOKEN_LENGTH not in _exit(run)["error"]


# -- agent isolation ---------------------------------------------------------------------------------------------
def test_the_layers_bubblewrap_is_probed_by_path_before_anything_runs_and_marks_the_run_hosted(
    run, monkeypatch, isolation
):
    fake = FakeOptimize(monkeypatch)
    assert entry.main(["run"]) == 0
    assert isolation.asked == [str(run.layer / "bin" / "bwrap")]
    assert isolation.started_then == [False]  # no session.json, no install.log yet
    assert len(fake.calls) == 1 and agent_isolation.isolation_required()


@pytest.mark.parametrize(
    ("layer", "said"),
    [
        (None, NO_LAYER_BWRAP),
        (" \t\n", NO_LAYER_BWRAP),
        ("layer", "bubblewrap at layer/bin/bwrap is not an absolute path"),
    ],
    ids=["unset", "blank", "relative"],
)
def test_a_run_whose_layer_names_no_bubblewrap_ends_finally_without_probing_one(
    run, monkeypatch, isolation, layer, said
):
    """No bubblewrap on PATH stands in for the layer's: the framework image may carry its own, maybe setuid."""
    if layer is None:
        monkeypatch.delenv(HOST_LAYER_ENV)
    else:
        monkeypatch.setenv(HOST_LAYER_ENV, layer)
    fake = FakeOptimize(monkeypatch)
    assert entry.main(["run"]) == entry.RETURNCODE_REFUSED
    written = _exit(run)
    assert (written["final"], written["returncode"]) == (True, entry.RETURNCODE_REFUSED)
    assert written["error"].startswith("agents cannot be isolated on this host: ") and said in written["error"]
    assert isolation.asked == [] and fake.calls == []


def test_a_layer_named_with_whitespace_around_it_is_probed_at_the_path_inside_it(run, monkeypatch, isolation):
    monkeypatch.setenv(HOST_LAYER_ENV, f"  {run.layer}\t\n")
    fake = FakeOptimize(monkeypatch)
    assert entry.main(["run"]) == 0 and len(fake.calls) == 1
    assert isolation.asked == [str(run.layer / "bin" / "bwrap")]


@pytest.mark.parametrize(
    ("problems", "warned"),
    [(("bwrap: setting up uid map: Permission denied",), True), ((), False)],
    ids=["layer-bubblewrap-denied", "layer-bubblewrap-confines"],
)
def test_a_run_whose_specialists_have_an_account_goes_on_and_warns_once_where_the_ray_path_would_refuse(
    run, monkeypatch, isolation, caplog, problems, warned
):
    """The account isolates every specialist but those of the Ray GPU actor path, which only bubblewrap can confine:
    start-up probes the layer's bubblewrap too, and says so where it cannot, without ending the run."""
    account = RunAs(user="hl-specialist", uid=2000, gid=2000, groups=(2000,), home="/home/hl", drop_tool="/bin/setpriv")
    monkeypatch.setenv("HYPERLOOM_SPECIALIST_USER", account.user)
    monkeypatch.setattr(agent_isolation, "resolve_run_as", lambda name: account if name == account.user else None)
    monkeypatch.setattr(agent_isolation, "missing_launch_capabilities", list)
    isolation.problems = problems
    fake = FakeOptimize(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=agent_isolation.__name__):
        assert entry.main(["run"]) == 0
    bwrap = run.layer / "bin" / "bwrap"
    assert len(fake.calls) == 1 and isolation.asked == [str(bwrap)] and isolation.started_then == [False]
    said = [record.getMessage() for record in caplog.records if record.name == agent_isolation.__name__]
    assert said == (
        [
            f"specialists run as the account {account.user}, but {RAY_ACTOR_PATH} and bubblewrap cannot confine one"
            f" here, so that path refuses its specialists: bubblewrap ({bwrap}, 0.13.1): {problems[0]}"
        ]
        if warned
        else []
    )


def test_a_host_where_no_tier_can_isolate_an_agent_ends_the_run_finally_naming_both(run, monkeypatch, isolation):
    isolation.problems = ("bwrap: setting up uid map: Permission denied",)
    _recording_installer(run, None)
    fake = FakeOptimize(monkeypatch)
    assert entry.main(["run"]) == entry.RETURNCODE_REFUSED
    bwrap = run.layer / "bin" / "bwrap"
    assert _exit(run) == {
        "final": True,
        "returncode": entry.RETURNCODE_REFUSED,
        "error": "agents cannot be isolated on this host: running them as an account: HYPERLOOM_SPECIALIST_USER is"
        f" not set; bubblewrap ({bwrap}, 0.13.1): bwrap: setting up uid map: Permission denied",
    }
    assert fake.calls == [] and not (run.work / "order").exists()  # neither the installer nor optimize ran
    assert sorted(path.name for path in run.state.iterdir()) == [entry.EXIT_FILE]  # no session either


def test_a_probe_that_gave_no_answer_ends_the_leg_without_exit_json_so_the_host_starts_it_again(
    run, monkeypatch, isolation
):
    isolation.problems, isolation.transient = ("no answer in 10 s",), True
    fake = FakeOptimize(monkeypatch)
    with pytest.raises(IsolationRefused) as raised:
        entry.main(["run"])
    assert raised.value.transient and "no answer in 10 s" in str(raised.value)
    assert fake.calls == [] and not list(run.state.iterdir())


def test_the_state_directory_is_masked_wherever_an_agent_sandbox_would_show_it(run, monkeypatch):
    """The entry takes HYPERLOOM_STATE_DIR out of the environment, where a launch looks for the app's own trees, so it
    registers the directory itself: a sandbox that binds a tree holding it masks it."""
    FakeOptimize(monkeypatch)
    assert entry.main(["run"]) == 0
    assert Var.STATE_DIR not in os.environ
    shown = run.state.parent
    masked = confinement.credential_paths({}, allowed=(shown,))
    policy = confinement.Confinement(readable=(shown,), masked=masked).policy()
    assert str(run.state) in policy.masked, policy.masked


def _image_bwrap(tmp_path: Path, monkeypatch) -> None:
    """Put first on ``PATH`` a ``bwrap`` that fails every probe, as a framework image's own might, and have the entry
    probe for real."""
    image_bin = tmp_path / "image-bin"
    image_bin.mkdir()
    (image_bin / "bwrap").write_text("#!/bin/sh\necho 'not the layer bubblewrap' >&2\nexit 1\n")
    (image_bin / "bwrap").chmod(0o755)
    monkeypatch.setenv("PATH", f"{image_bin}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(agent_isolation, "bwrap_probe", bwrap.bwrap_probe)
    monkeypatch.setattr(bwrap, "_PROBES", {})


@pytest.mark.agent_isolation
def test_a_hosted_run_isolates_its_agents_with_a_real_layer_bubblewrap_and_never_the_one_on_path(
    run, monkeypatch, tmp_path
):
    real = usable_bwrap()
    (run.layer / "bin").mkdir(parents=True)
    (run.layer / "bin" / "bwrap").symlink_to(str(real.path))
    _image_bwrap(tmp_path, monkeypatch)
    fake = FakeOptimize(monkeypatch)
    assert entry.main(["run"]) == 0 and len(fake.calls) == 1
    report = agent_isolation.agent_isolation_report()
    assert (report.specialists, report.sandboxes, report.required) == ("bwrap", "bwrap", True)
    assert report.bwrap == f"{'.'.join(map(str, real.version or ()))} {run.layer / 'bin' / 'bwrap'}"


def test_a_hosted_run_whose_layer_has_no_bubblewrap_is_refused_naming_it(run, monkeypatch, tmp_path):
    _image_bwrap(tmp_path, monkeypatch)
    fake = FakeOptimize(monkeypatch)
    assert entry.main(["run"]) == entry.RETURNCODE_REFUSED and fake.calls == []
    error = _exit(run)["error"]
    assert f"bubblewrap at {run.layer / 'bin' / 'bwrap'}: No such file or directory" in error, error
    assert "not the layer bubblewrap" not in error


# -- the installer -----------------------------------------------------------------------------------------------
def _recording_installer(run, events, *, exit_code=0):
    run.installer.write_text(
        f'#!/bin/bash\necho "installer $#" >> {run.work}/order\necho install-output\necho install-error >&2\nexit {exit_code}\n'
    )


def _record_optimize(run, monkeypatch):
    def fake(argv):
        with (run.work / "order").open("a") as order:
            order.write("optimize\n")
        return 0

    monkeypatch.setattr(entry, "_optimize_main", fake)


def _order(run):
    return (run.work / "order").read_text().split("\n")[:-1]


def test_the_installer_runs_before_optimize_on_a_fresh_leg_and_on_a_resume_leg(run, monkeypatch):
    _recording_installer(run, None)
    _record_optimize(run, monkeypatch)
    entry.main(["run"])
    assert _order(run) == ["installer 0", "optimize"]  # no flags, like the legacy launchers
    session = Path(_session_json(run))
    for name in entry.SESSION_FILES:
        (session / name).write_text("{}")
    run.restart()
    entry.main(["run"])
    assert _order(run) == ["installer 0", "optimize", "installer 0", "optimize"]


def test_the_installer_output_goes_to_a_log_in_the_state_dir(run, monkeypatch):
    _recording_installer(run, None)
    _record_optimize(run, monkeypatch)
    entry.main(["run"])
    assert set((run.state / entry.INSTALL_LOG).read_text().split()) == {"install-output", "install-error"}


def test_each_legs_installer_log_is_a_new_file_renamed_over_the_last_never_one_opened_in_place(run, monkeypatch):
    """The previous leg's log may be another account's, and a member of the group may have put a link there."""
    run.state.chmod(0o2775)  # storage set up to be shared: setgid, group-writable, closed to others
    _recording_installer(run, None)
    _record_optimize(run, monkeypatch)
    entry.main(["run"])
    log = run.state / entry.INSTALL_LOG
    assert stat.S_IMODE(log.stat().st_mode) == 0o664
    (run.state.parent / "first-leg-log").hardlink_to(log)
    first = log.read_text()
    outside = run.state.parent / "outside"
    outside.write_text("kept")
    log.unlink()
    log.symlink_to(outside)

    run.restart()
    entry.main(["run"])

    assert outside.read_text() == "kept" and not log.is_symlink()
    assert (run.state.parent / "first-leg-log").read_text() == first
    assert set(log.read_text().split()) == {"install-output", "install-error"}
    assert stat.S_IMODE(log.stat().st_mode) == 0o664
    assert [path.name for path in run.state.iterdir() if path.name.startswith(".")] == []


def test_exit_and_session_json_are_readable_by_the_group_and_replaced_never_rewritten(run, monkeypatch):
    run.state.chmod(0o2775)  # storage set up to be shared: setgid, group-writable, closed to others
    FakeOptimize(monkeypatch)
    entry.main(["run"])
    for name in (entry.EXIT_FILE, entry.SESSION_FILE):
        (run.state.parent / f"earlier-{name}").hardlink_to(run.state / name)
    run.restart()
    entry.main(["run"])
    for name in (entry.EXIT_FILE, entry.SESSION_FILE):
        assert stat.S_IMODE((run.state / name).stat().st_mode) == 0o664
        assert not (run.state / name).samefile(run.state.parent / f"earlier-{name}")


def test_a_failing_installer_ends_the_run_finally_and_optimize_never_starts(run, monkeypatch, capsys):
    _recording_installer(run, None, exit_code=7)
    _record_optimize(run, monkeypatch)
    assert entry.main(["run"]) == 2
    assert _order(run) == ["installer 0"]
    log = run.state / entry.INSTALL_LOG
    assert _exit(run) == {
        "final": True,
        "returncode": 2,
        "error": f"install.sh failed with exit code 7; see {log}",
    }
    assert "install-error" in log.read_text()


def test_the_installer_sees_no_host_only_input(run, monkeypatch):
    run.installer.write_text(f"#!/bin/bash\nenv > {run.work}/seen\n")
    FakeOptimize(monkeypatch)
    entry.main(["run"])
    seen = (run.work / "seen").read_text()
    assert "sentinel-secret" not in seen
    leaked = [name for name in host_info.HOST_ONLY_VARS if f"\n{name}=" in "\n" + seen]
    assert leaked == [Var.STATE_DIR]


def test_the_installer_is_told_the_state_dir_so_it_installs_the_hosted_extra(run, monkeypatch):
    run.installer.write_text(f'#!/bin/bash\necho "$HYPERLOOM_STATE_DIR" > {run.work}/seen\n')
    FakeOptimize(monkeypatch)
    entry.main(["run"])
    assert (run.work / "seen").read_text().strip() == str(run.state)


def test_a_missing_installer_ends_the_run_finally(run, monkeypatch):
    run.installer.unlink()
    monkeypatch.setattr(entry, "INSTALL_SCRIPT", run.installer)
    FakeOptimize(monkeypatch)
    monkeypatch.setattr(
        entry.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError(2, "No such file"))
    )
    assert entry.main(["run"]) == 2
    assert _exit(run)["final"] is True
    assert "install.sh" in _exit(run)["error"]


def test_the_installer_the_entry_names_is_the_packaged_asset_beside_the_optimizer():
    assert INSTALL_REAL.is_file()
    assert INSTALL_REAL.parent.parts[-3:] == ("hyperloom", "inference_optimizer", "assets")


def test_the_installer_installs_into_the_running_venv_and_optimize_finds_its_scripts(run, monkeypatch, tmp_path):
    venv = tmp_path / "state-venv"
    monkeypatch.setattr(sys, "prefix", str(venv))
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    monkeypatch.setattr(sys, "executable", str(venv / "bin" / "python"))
    monkeypatch.setenv("PATH", f"/usr/bin{os.pathsep}{venv}/bin")  # the venv is on PATH, but not first
    run.installer.write_text(
        "#!/bin/bash\n"
        f'printf "%s\\n" "$PYTHON" "$VIRTUAL_ENV" "$INFERENCE_OPTIMIZER_FORCE_PYTHON" "${{PATH%%:*}}" > {run.work}/installer-env\n'
    )
    fake = FakeOptimize(monkeypatch)
    entry.main(["run"])
    assert (run.work / "installer-env").read_text().split("\n")[:-1] == [
        str(venv / "bin" / "python"),
        str(venv),
        "1",
        str(venv / "bin"),
    ]
    assert fake.environments[0]["PATH"].split(os.pathsep)[0] == str(venv / "bin")
    assert fake.environments[0]["PATH"].split(os.pathsep).count(str(venv / "bin")) == 1


def test_a_package_the_working_directory_shadows_as_a_namespace_still_resolves_for_optimize(run, monkeypatch, tmp_path):
    """The shape of the framework image: a directory without ``__init__.py`` in the working directory, the real package
    reached by an editable install's finder, which the import system consults after ``sys.path``."""
    import importlib.abc
    import importlib.util

    cwd, real = tmp_path / "cwd", tmp_path / "real" / "shadowed_pkg"
    (cwd / "shadowed_pkg").mkdir(parents=True)
    real.mkdir(parents=True)
    (real / "__init__.py").write_text("")

    class EditableFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "shadowed_pkg":
                return importlib.util.spec_from_file_location(
                    name, real / "__init__.py", submodule_search_locations=[str(real)]
                )
            return None

    finder = EditableFinder()
    monkeypatch.setattr(sys, "meta_path", [*sys.meta_path, finder])
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(sys, "path", ["", str(cwd), *sys.path])  # what ``python -m`` leaves
    monkeypatch.delitem(sys.modules, "shadowed_pkg", raising=False)
    assert importlib.util.find_spec("shadowed_pkg").origin is None  # the defect, without the entry's guard
    seen = []

    def optimize(argv):
        sys.modules.pop("shadowed_pkg", None)
        importlib.invalidate_caches()
        seen.append(importlib.util.find_spec("shadowed_pkg").origin)
        return 0

    monkeypatch.setattr(entry, "_optimize_main", optimize)
    entry.main(["run"])
    assert seen == [str(real / "__init__.py")]


# -- a leg in a new container ------------------------------------------------------------------------------------
_THIS = "0b0b0b0b-2222-4222-8222-000000000002:4026531002"
_GONE = "0a0a0a0a-1111-4111-8111-000000000001:4026531001"


def _resume_rig(run, monkeypatch, *, scope_of_session):
    """A session that a first leg started and left a running task in, resumed by an entry that runs in ``_THIS``.

    The fake ``optimize`` runs the real resume preflight, as ``optimize --resume-from`` does.
    """
    from hyperloom.inference_optimizer.session import resume_guard
    from hyperloom.inference_optimizer.tests.test_resume_scope_gone import _session

    monkeypatch.setattr(entry, "local_owner_scope", lambda: _THIS)
    FakeOptimize(monkeypatch)
    entry.main(["run"])
    session = Path(_session_json(run))
    _session(session, {"explore-1": ["benchmark_lane", "profile_lane"]}, scope=scope_of_session)
    confirmations = []
    real = resume_guard.confirm_scope_gone
    monkeypatch.setattr(
        entry, "confirm_scope_gone", lambda *a, **k: confirmations.append(k["owner_scope"]) or real(*a, **k)
    )
    preflight = []

    def optimize(argv):
        resume_guard.ensure_resume_safe(session, owner_scope=entry.local_owner_scope())
        preflight.append(argv[1])
        return 0

    monkeypatch.setattr(entry, "_optimize_main", optimize)
    run.restart()
    return session, confirmations, preflight


def test_a_resume_in_the_same_container_confirms_nothing(run, monkeypatch):
    _session_dir, confirmations, preflight = _resume_rig(run, monkeypatch, scope_of_session=_THIS)
    assert entry.main(["run"]) == 0
    assert confirmations == [] and len(preflight) == 1


def test_a_resume_in_a_new_container_confirms_the_old_one_gone_before_the_preflight(run, monkeypatch, capsys):
    from hyperloom.inference_optimizer.session.resume_guard import ResumeBlocked, ensure_resume_safe

    session, confirmations, preflight = _resume_rig(run, monkeypatch, scope_of_session=_GONE)
    with pytest.raises(ResumeBlocked):  # without the confirmation the real preflight refuses
        ensure_resume_safe(session, owner_scope=_THIS)
    assert entry.main(["run"]) == 0
    assert confirmations == [_GONE] and len(preflight) == 1
    assert not (run.state / "exit.json").exists() or _exit(run)["returncode"] == 0
    assert "marked interrupted" in capsys.readouterr().out


def test_a_confirmation_that_is_refused_ends_the_run_finally_with_the_reason(run, monkeypatch):
    from hyperloom.inference_optimizer.session.paths import db_path_for
    from hyperloom.inference_optimizer.tests.test_resume_scope_gone import FUTURE, _lease
    import sqlite3
    from contextlib import closing

    session, confirmations, preflight = _resume_rig(run, monkeypatch, scope_of_session=_GONE)
    with closing(sqlite3.connect(db_path_for(session))) as db, db:
        _lease(db, "other_lane", "explore-1", "0c0c0c0c-3333-4333-8333-000000000003:1", FUTURE)  # a live other owner
    assert entry.main(["run"]) == 2
    assert preflight == []
    document = _exit(run)
    assert document["final"] is True and document["returncode"] == 2
    assert "previous container's work cannot be confirmed gone" in document["error"]


# -- where the results go ----------------------------------------------------------------------------------------
def test_the_session_package_is_written_into_the_archive_not_the_container(run, monkeypatch):
    from hyperloom.inference_optimizer.breakdown.session_package import PACKAGE_SUBDIR, package_session_artifacts

    monkeypatch.setenv("HYPERLOOM_SESSION_PACKAGE_DEST", "/workspace")  # a host or image default must not win
    packaged = []

    def optimize(argv):
        session = Path(_session_json(run))
        (session / "state.json").write_text("{}")
        packaged.append(package_session_artifacts(session, session_id="s1"))
        return 0

    monkeypatch.setattr(entry, "_optimize_main", optimize)
    entry.main(["run"])
    archive = Path(_session_json(run)) / "archive"
    assert packaged == [archive / PACKAGE_SUBDIR / "s1.zip"] and packaged[0].is_file()


def test_write_exit_is_public_and_writes_the_documented_document(tmp_path):
    from hyperloom.hosted.entry import write_exit

    write_exit(tmp_path, final=True, returncode=3, stop_reason="x")

    assert json.loads((tmp_path / "exit.json").read_text()) == {"final": True, "returncode": 3, "stop_reason": "x"}


# -- a resume that cannot be inspected is final, whatever the previous leg ended on ---------------------------------
_CREDENTIAL_SESSION = {
    "operator_extra_env": {"HF_TOKEN": "hf_live_value_123"},
    "tick": 3,
    "stop_reason": "supervisor_restart_requested",  # what a watchdog restart left: the host starts the run again
}


def _a_credential_session_that_ran_in_another_container(run, monkeypatch, tmp_path):
    """A first leg leaves a session; it is a credential session that ran on a ledger in the first container's TMPDIR, and
    its last ``stop_reason`` is the watchdog's. The next leg starts in a new container, with a TMPDIR of its own."""
    monkeypatch.delenv("HYPERLOOM_LEDGER_DIR", raising=False)
    first = tmp_path / "tmp-container-1"
    first.mkdir(mode=0o700)
    monkeypatch.setenv("TMPDIR", str(first))
    FakeOptimize(monkeypatch)  # the first leg: leaves a session behind
    assert entry.main(["run"]) == 0
    session = Path(_session_json(run))
    (session / "state.json").write_text(json.dumps(_CREDENTIAL_SESSION))
    from hyperloom.inference_optimizer.session.paths import ledger_mark_path

    ledger_mark_path(session).parent.mkdir(parents=True, exist_ok=True)
    ledger_mark_path(session).write_text(
        '{"ledger_private": true}'
    )  # what the first leg left when it opened its ledger
    run.restart()
    second = tmp_path / "tmp-container-2"
    second.mkdir(mode=0o700)
    monkeypatch.setenv("TMPDIR", str(second))
    (run.state / "install.log").unlink(missing_ok=True)
    return session


def test_a_resume_with_no_ledger_is_final_with_the_remedy_even_after_a_watchdog_restart(run, monkeypatch, tmp_path):
    session = _a_credential_session_that_ran_in_another_container(run, monkeypatch, tmp_path)
    fake = FakeOptimize(monkeypatch)
    assert entry.main(["run"]) == 2
    written = _exit(run)
    assert (written["final"], written["returncode"]) == (True, 2), written
    assert "HYPERLOOM_LEDGER_DIR" in written["error"] and "start a new session" in written["error"]
    assert "stop_reason" not in written, "not the previous leg's"
    assert fake.calls == [] and not (run.state / "install.log").exists(), "nothing was installed or started"

    run.restart()  # the host starts it again, in another container
    assert entry.main(["run"]) == 2
    assert _exit(run) == written, "the same final answer, nothing was run"
    assert fake.calls == [] and json.loads((session / "state.json").read_text()) == _CREDENTIAL_SESSION


def test_a_resume_the_guard_of_optimize_would_refuse_is_final_too(run, monkeypatch, tmp_path):
    """A worker whose exit is not proven (a held GPU lease in the session's database): ``optimize`` would exit 2, which
    after a watchdog restart ``exit.json`` records as not final, and the host would start the run again for ever."""
    from hyperloom.inference_optimizer.session.paths import shared_db_path_for
    from hyperloom.orchestrator.bus.storage.connection import SqliteConnection

    FakeOptimize(monkeypatch)
    assert entry.main(["run"]) == 0
    session = Path(_session_json(run))
    (session / "state.json").write_text(json.dumps({"stop_reason": "supervisor_restart_requested", "tick": 3}))
    database = SqliteConnection(shared_db_path_for(session))
    database.raw.execute("INSERT INTO gpu_leases (gpu_id, holder_id, task_id) VALUES (0, 'worker', 'ti-live')")
    database.raw.close()
    run.restart()
    fake = FakeOptimize(monkeypatch)
    assert entry.main(["run"]) == 2
    written = _exit(run)
    assert (written["final"], written["returncode"]) == (True, 2) and "worker exit is unproven" in written["error"]
    assert fake.calls == []
