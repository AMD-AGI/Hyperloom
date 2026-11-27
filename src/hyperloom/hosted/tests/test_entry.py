# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The hosted entry: the checks before the hand-over, exit.json of a refusal, the installer, the multibranch hand-over."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
from pathlib import Path

import pytest
from hyperloom.common import agent_isolation
from hyperloom.common.agent_isolation import HOST_LAYER_ENV
from hyperloom.hosted import entry, host_info
from hyperloom.hosted.host_info import Var
from hyperloom.hosted.tests.test_campaign import FULL, MINIMAL

INSTALL_REAL = Path(entry.INSTALL_SCRIPT)
_SECRETS = {name: f"sentinel-secret-{index}" for index, name in enumerate(host_info.SECRET_VARS)}
_IMAGE = "registry.example/sglang:v0.5.4-rocm"


class Exec:
    """Stands in for ``os.execvp``: records each hand-over and the environment it would have run in, and returns."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], dict[str, str]]] = []

    def __call__(self, file: str, argv: list[str]) -> None:
        self.calls.append((file, list(argv), dict(os.environ)))


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A complete host environment in ``os.environ`` and the paths it names, with the hosting layer's directory in
    ``HYPERLOOM_HOST_LAYER``; the hand-over to the multibranch command is recorded, not made."""
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
        Var.SANDBOX_TEMPLATES: json.dumps({"measurement": {"1": "r1"}}),
        Var.E2B_API_URL: "https://api.sandbox.example.test",
        HOST_LAYER_ENV: str(tmp_path / "layer"),
        "USER_DATA_PATH": str(work),
        "ANTHROPIC_API_KEY": "gateway-key",
        **_SECRETS,
    }
    for name in (*(v for k, v in vars(Var).items() if not k.startswith("_")), entry.ENV_RUNTIME_DIR):
        # Set first so that monkeypatch restores it: what the entry sets in os.environ must not outlive the test.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))  # the entry edits it
    container_tmp = tmp_path / "container-tmp"
    container_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(container_tmp))  # the container's own disk

    def inject():
        """What the host does at every (re)start of the application: the same environment again."""
        for name, value in values.items():
            monkeypatch.setenv(name, value)

    installer = tmp_path / "install.sh"
    installer.write_text("#!/bin/bash\necho stub installer ran\n")
    monkeypatch.setattr(entry, "INSTALL_SCRIPT", installer)  # the real installer is never run by a unit test
    handed = Exec()
    monkeypatch.setattr(entry.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(entry.os, "execvp", handed)
    probed: list[object] = []
    monkeypatch.setattr(agent_isolation, "bwrap_probe", lambda path: probed.append(path))
    inject()
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
            "handed": handed,
            "probed": probed,
        },
    )


def _exit(run):
    return json.loads((run.state / "exit.json").read_text())


def _break_spec(run, doc):
    run.spec.write_text(json.dumps(doc))


# -- the hand-over -----------------------------------------------------------------------------------------------
def test_a_spec_without_a_flow_execs_into_the_multibranch_command_with_the_environment_unchanged_but_for_path(run):
    before = dict(os.environ)
    assert entry.main(["run"]) == 0
    assert [(c[0], c[1]) for c in run.handed.calls] == [
        ("hyperloom-multibranch", ["hyperloom-multibranch", "hosted", "run"])
    ]
    environment = run.handed.calls[0][2]
    bin_dir = str(Path(sys.executable).parent)
    path = environment.pop("PATH")
    assert path.split(os.pathsep)[0] == bin_dir  # the installer's interpreter resolves the command
    before.pop("PATH", None)
    runtime = environment.pop(entry.ENV_RUNTIME_DIR)
    assert Path(runtime).parent == run.container_tmp
    assert environment == before and Var.RUN_ID in before  # the command reads the host inputs itself
    assert not (run.state / "session.json").exists() and not (run.state / "exit.json").exists()
    assert run.probed == []  # the command checks the isolation of its specialists itself


def test_a_spec_naming_the_multibranch_flow_is_handed_over_with_its_section_unread(run):
    doc = {**FULL, "flow": "multibranch", "placement": {"profile": "hosted-gpu", "measurement": "hosted"}}
    doc["multibranch"] = ["opaque", {"never": "validated"}, 3]
    _break_spec(run, doc)
    assert entry.main(["run"]) == 0
    assert len(run.handed.calls) == 1


def test_a_sweep_spec_is_handed_over_like_any_other(run):
    _break_spec(run, {**MINIMAL, "flow": "conc_sweep", "conc_sweep": {"session_dir": "/nfs/s"}})
    assert entry.main(["run"]) == 0
    assert len(run.handed.calls) == 1


def test_a_spec_naming_the_retired_single_flow_is_refused_finally_naming_why(run):
    _break_spec(run, {**MINIMAL, "flow": "single"})
    assert entry.main(["run"]) == 2
    assert _exit(run) == {
        "final": True,
        "returncode": 2,
        "error": "campaign spec refused: flow 'single' is retired; leave flow out or set it to 'multibranch'",
    }
    assert run.handed.calls == [] and not (run.state / entry.INSTALL_LOG).exists()


def test_an_unchanged_single_spec_of_1_2_0_without_flow_or_multibranch_is_refused_finally_naming_why(run):
    doc = {key: value for key, value in MINIMAL.items() if key != "multibranch"}
    _break_spec(run, doc)
    assert entry.main(["run"]) == 2
    assert _exit(run) == {
        "final": True,
        "returncode": 2,
        "error": "campaign spec refused: a spec with neither flow nor multibranch ran the single flow, which is"
        " retired; set flow to 'multibranch' and add a multibranch section with workload, and give workload.conc"
        " and workload.precision",
    }
    assert run.handed.calls == [] and not (run.state / entry.INSTALL_LOG).exists()


def test_the_installer_runs_before_the_exec_without_host_secrets(run):
    run.installer.write_text(
        f'#!/bin/bash\necho "installer $#" >> {run.work}/order\necho "${{HYPERLOOM_SANDBOX_KEY_DEV:-unset}}" >> {run.work}/order\n'
    )
    entry.main(["run"])
    assert (run.work / "order").read_text().split("\n")[:-1] == ["installer 0", "unset"]  # no flags, like the launchers
    assert len(run.handed.calls) == 1


def test_without_its_command_the_run_is_final_with_return_code_2(run, monkeypatch):
    monkeypatch.setattr(entry.shutil, "which", lambda name: None)
    assert entry.main(["run"]) == 2
    written = _exit(run)
    assert (written["final"], written["returncode"]) == (True, 2) and "hyperloom-multibranch" in written["error"]


def test_a_command_that_cannot_start_is_final_with_return_code_2(run, monkeypatch):
    def refuse(file, argv):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(entry.os, "execvp", refuse)
    assert entry.main(["run"]) == 2
    assert _exit(run)["error"] == "could not start hyperloom-multibranch: Permission denied"


# -- refusals ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r, m: _break_spec(r, {**MINIMAL, "budget": {}}), "budget.max_hours is required"),
        (lambda r, m: r.spec.write_text("{not json"), "is not valid JSON"),
        (lambda r, m: m.delenv(Var.RUN_ID), Var.RUN_ID),
        (lambda r, m: m.setenv(Var.E2B_API_URL, "not a url"), Var.E2B_API_URL),
        (lambda r, m: m.setenv("USER_DATA_PATH", "relative"), "USER_DATA_PATH must be set to an absolute path"),
        (
            lambda r, m: m.setenv("HYPERLOOM_LEDGER_DIR", "relative/ledgers"),
            "HYPERLOOM_LEDGER_DIR must be an absolute path",
        ),
    ],
)
def test_a_refused_request_is_final_with_return_code_2_and_is_never_handed_over(run, monkeypatch, mutate, message):
    mutate(run, monkeypatch)
    assert entry.main(["run"]) == 2
    written = _exit(run)
    assert (written["final"], written["returncode"]) == (True, 2) and message in written["error"]
    assert run.handed.calls == []
    assert not any(secret in json.dumps(written) for secret in _SECRETS.values())


def test_a_refusal_the_state_directory_cannot_hold_is_still_return_code_2(run, monkeypatch, capsys):
    monkeypatch.delenv(Var.STATE_DIR)
    assert entry.main(["run"]) == 2
    assert Var.STATE_DIR in capsys.readouterr().err


def test_anything_but_run_is_a_usage_error_that_writes_nothing(run, capsys):
    assert entry.main(["walk"]) == 2 and entry.main([]) == 2
    assert not (run.state / "exit.json").exists()


def test_a_refusals_exit_json_is_readable_by_the_group_and_replaced_never_rewritten(run):
    run.state.chmod(0o2775)  # storage set up to be shared: setgid, group-writable, closed to others
    _break_spec(run, {**MINIMAL, "flow": "single"})
    entry.main(["run"])
    (run.state.parent / "earlier-exit.json").hardlink_to(run.state / entry.EXIT_FILE)
    run.restart()
    entry.main(["run"])
    assert stat.S_IMODE((run.state / entry.EXIT_FILE).stat().st_mode) == 0o664
    assert not (run.state / entry.EXIT_FILE).samefile(run.state.parent / "earlier-exit.json")


def test_write_exit_is_public_and_writes_the_documented_document(tmp_path):
    from hyperloom.hosted.entry import write_exit

    write_exit(tmp_path, final=True, returncode=3, stop_reason="x")

    assert json.loads((tmp_path / "exit.json").read_text()) == {"final": True, "returncode": 3, "stop_reason": "x"}


# -- the shared storage, which each leg may write as another account ---------------------------------------------
def test_a_user_data_path_this_account_cannot_write_ends_the_run_finally_naming_it(run):
    run.work.rmdir()
    run.work.write_text("")  # what no account can create a file in
    assert entry.main(["run"]) == 2
    assert f"cannot write the shared storage: USER_DATA_PATH {run.work}: ENOTDIR" in _exit(run)["error"]
    assert run.handed.calls == []


def test_a_state_dir_this_account_cannot_write_is_said_on_stderr_only(run, monkeypatch, capsys):
    probe = entry.writable_problems
    monkeypatch.setattr(
        entry,
        "writable_problems",
        lambda dirs: [f"{name} {path}: EACCES" for name, path in dirs.items() if name == Var.STATE_DIR] or probe(dirs),
    )
    assert entry.main(["run"]) == 2
    assert f"{Var.STATE_DIR} {run.state}: EACCES" in capsys.readouterr().err
    assert list(run.state.iterdir()) == []


def test_a_key_directory_without_setgid_is_warned_about_and_one_with_it_is_not(run, capsys):
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
    entry.main(["run"])
    runtime = Path(run.handed.calls[0][2][entry.ENV_RUNTIME_DIR])
    assert runtime.parent == run.container_tmp and stat.S_IMODE(runtime.stat().st_mode) == 0o700
    assert (run.work / "seen").read_text().split() == [str(runtime)]
    run.restart()
    monkeypatch.setenv(entry.ENV_RUNTIME_DIR, "/host/chose/this")
    entry.main(["run"])
    assert run.handed.calls[1][2][entry.ENV_RUNTIME_DIR] == "/host/chose/this"


# -- the installer -----------------------------------------------------------------------------------------------
def _recording_installer(run, *, exit_code=0):
    run.installer.write_text(
        f'#!/bin/bash\necho "installer $#" >> {run.work}/order\necho install-output\necho install-error >&2\nexit {exit_code}\n'
    )


def _order(run):
    return (run.work / "order").read_text().split("\n")[:-1]


def test_the_installer_runs_on_every_leg(run):
    _recording_installer(run)
    entry.main(["run"])
    run.restart()
    entry.main(["run"])
    assert _order(run) == ["installer 0", "installer 0"]
    assert len(run.handed.calls) == 2


def test_the_installer_output_goes_to_a_log_in_the_state_dir(run):
    _recording_installer(run)
    entry.main(["run"])
    assert set((run.state / entry.INSTALL_LOG).read_text().split()) == {"install-output", "install-error"}


def test_each_legs_installer_log_is_a_new_file_renamed_over_the_last_never_one_opened_in_place(run):
    """The previous leg's log may be another account's, and a member of the group may have put a link there."""
    run.state.chmod(0o2775)  # storage set up to be shared: setgid, group-writable, closed to others
    _recording_installer(run)
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


def test_a_failing_installer_ends_the_run_finally_and_the_command_never_starts(run):
    _recording_installer(run, exit_code=7)
    assert entry.main(["run"]) == 2
    assert _order(run) == ["installer 0"]
    log = run.state / entry.INSTALL_LOG
    assert _exit(run) == {
        "final": True,
        "returncode": 2,
        "error": f"install.sh failed with exit code 7; see {log}",
    }
    assert "install-error" in log.read_text()
    assert run.handed.calls == []


def test_the_installer_sees_no_host_only_input(run):
    run.installer.write_text(f"#!/bin/bash\nenv > {run.work}/seen\n")
    entry.main(["run"])
    seen = (run.work / "seen").read_text()
    assert "sentinel-secret" not in seen
    leaked = [name for name in host_info.HOST_ONLY_VARS if f"\n{name}=" in "\n" + seen]
    assert leaked == [Var.STATE_DIR]


def test_the_installer_is_told_the_state_dir_so_it_installs_the_hosted_extra(run):
    run.installer.write_text(f'#!/bin/bash\necho "$HYPERLOOM_STATE_DIR" > {run.work}/seen\n')
    entry.main(["run"])
    assert (run.work / "seen").read_text().strip() == str(run.state)


def test_a_missing_installer_ends_the_run_finally(run, monkeypatch):
    run.installer.unlink()
    monkeypatch.setattr(
        entry.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError(2, "No such file"))
    )
    assert entry.main(["run"]) == 2
    assert _exit(run)["final"] is True
    assert "install.sh" in _exit(run)["error"]


def test_the_installer_the_entry_names_is_the_packaged_asset_beside_the_optimizer():
    assert INSTALL_REAL.is_file()
    assert INSTALL_REAL.parent.parts[-3:] == ("hyperloom", "inference_optimizer", "assets")


def test_the_installer_installs_into_the_running_venv_and_the_command_finds_its_scripts(run, monkeypatch, tmp_path):
    venv = tmp_path / "state-venv"
    monkeypatch.setattr(sys, "prefix", str(venv))
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    monkeypatch.setattr(sys, "executable", str(venv / "bin" / "python"))
    monkeypatch.setenv("PATH", f"/usr/bin{os.pathsep}{venv}/bin")  # the venv is on PATH, but not first
    run.installer.write_text(
        "#!/bin/bash\n"
        f'printf "%s\\n" "$PYTHON" "$VIRTUAL_ENV" "$INFERENCE_OPTIMIZER_FORCE_PYTHON" "${{PATH%%:*}}" > {run.work}/installer-env\n'
    )
    entry.main(["run"])
    assert (run.work / "installer-env").read_text().split("\n")[:-1] == [
        str(venv / "bin" / "python"),
        str(venv),
        "1",
        str(venv / "bin"),
    ]
    path = run.handed.calls[0][2]["PATH"].split(os.pathsep)
    assert path[0] == str(venv / "bin") and path.count(str(venv / "bin")) == 1
