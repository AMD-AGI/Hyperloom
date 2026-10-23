# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner's start-up script: the venv it builds sees the framework image, installs only what the lock pins, and a
step that fails leaves the application a ``status.json`` instead of a runner that never comes.

The framework image is a temporary base venv (``bootstrap_rig``): its ``python3`` is a venv, as ``/opt/venv`` is, and
holds a module the runner needs that only it has.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

import pytest

from hyperloom.measure.runner.bootstrap import BOOTSTRAP_SCRIPT, runner_start_command
from hyperloom.measure.runner.tests.bootstrap_rig import (
    Rig,
    copy_source,
    lock_text,
    make_rig,
    read_probe,
    run_script,
)

_SPEC = b'{"id": 7}'
_START = runner_start_command("--spec-wait-sec", "20")
_GIT = ("git", "-c", "user.name=t", "-c", "user.email=t@example.invalid")


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Rig:
    return make_rig(tmp_path_factory.mktemp("rig") / "rig")


def _run(rig: Rig, work: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    (work / "in").mkdir(parents=True, exist_ok=True)
    (work / "in" / "spec.json").write_bytes(_SPEC)
    return run_script(_START, rig, work, env)


def _status(work: Path) -> dict[str, str]:
    return json.loads((work / "out" / "status.json").read_text())


def _venv_has(work: Path, module: str) -> bool:
    return any((work / "scratch" / "hl-venv").rglob(f"{module}.py"))


def test_the_runner_sees_the_base_and_what_the_lock_installs_comes_first(rig: Rig, tmp_path: Path) -> None:
    done = _run(rig, tmp_path)
    assert done.returncode == 0, done.stderr
    probe = read_probe(tmp_path / "out")
    venv = tmp_path / "scratch" / "hl-venv"
    assert probe["executable"] == str(venv / "bin" / "python")
    # Only the base has this module: the .pth is what makes it importable in the new venv.
    assert probe["base_only"] == str(rig.base_site / "base_only_pkg.py")
    # A pin the base already satisfies is left alone, never installed a second time.
    assert probe["already"] == str(rig.base_site / "already_pkg.py")
    assert not _venv_has(tmp_path, "already_pkg")
    # A pin the base does not satisfy is installed into the venv, and the venv's copy wins over the base's older one.
    # Its declared dependency is in no lock: nothing is resolved.
    assert probe["lock_origin"] == "lock"
    assert _venv_has(tmp_path, "lock_pkg")
    assert probe["runner_file"] == str(rig.source / "src/hyperloom/measure/runner/__main__.py")
    assert (tmp_path / "out" / "status.json").read_bytes() == _SPEC


def test_the_runner_gets_its_directories_and_its_own_arguments(rig: Rig, tmp_path: Path) -> None:
    assert _run(rig, tmp_path).returncode == 0
    assert read_probe(tmp_path / "out")["argv"] == [
        "run",
        "--in",
        str(tmp_path / "in"),
        "--out",
        str(tmp_path / "out"),
        "--scratch",
        str(tmp_path / "scratch"),
        "--spec-wait-sec",
        "20",
    ]


@pytest.mark.parametrize(
    ("lock", "reason"),
    [
        ("hash mismatch", "DO NOT MATCH THE HASHES"),
        ("no hash", "Hashes are required in --require-hashes mode"),
    ],
)
def test_a_lock_that_does_not_verify_installs_nothing_and_leaves_a_status(
    rig: Rig, tmp_path: Path, lock: str, reason: str
) -> None:
    if lock == "hash mismatch":
        text = lock_text(lock_pkg=("1.0", "0" * 64), already_pkg=("1.0", rig.already_pkg_sha256))
    else:
        text = "lock_pkg==1.0\nalready_pkg==1.0\n"  # pip checks hashes by itself only when some pin has one
    source = copy_source(rig, tmp_path / "source", text)
    done = _run(rig, tmp_path, {"HYPERLOOM_SOURCE_DIR": str(source)})
    assert done.returncode == 2
    assert reason in done.stderr
    status = _status(tmp_path)
    assert (status["state"], status["error_class"]) == ("failed", "runner_start_failed")
    assert "step lock" in status["message"]
    assert not _venv_has(tmp_path, "lock_pkg")
    assert not (tmp_path / "out" / "probe.json").exists()


def _git_repo(source: Path, repo: Path) -> str:
    shutil.copytree(source, repo)
    for args in (("init", "-q"), ("add", "-A"), ("commit", "-q", "-m", "runner")):
        subprocess.run([*_GIT, "-C", str(repo), *args], check=True, capture_output=True)
    return subprocess.run(
        [*_GIT, "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def test_a_pinned_commit_is_fetched_and_named_as_the_code_revision(rig: Rig, tmp_path: Path) -> None:
    sha = _git_repo(rig.source, tmp_path / "repo")
    env = {
        "HYPERLOOM_SOURCE_REPO": f"file://{tmp_path / 'repo'}",
        "HYPERLOOM_SOURCE_SHA": sha,
        "HYPERLOOM_SOURCE_DIR": str(tmp_path / "ignored"),
    }
    done = _run(rig, tmp_path, env)
    assert done.returncode == 0, done.stderr
    probe = read_probe(tmp_path / "out")
    assert probe["revision"] == sha
    assert probe["runner_file"] == str(tmp_path / "scratch" / "hl-src" / "src/hyperloom/measure/runner/__main__.py")
    head = subprocess.run(
        ["git", "-C", str(tmp_path / "scratch" / "hl-src"), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert head == sha


def test_a_commit_the_repository_does_not_have_ends_the_start(rig: Rig, tmp_path: Path) -> None:
    _git_repo(rig.source, tmp_path / "repo")
    env = {"HYPERLOOM_SOURCE_REPO": f"file://{tmp_path / 'repo'}", "HYPERLOOM_SOURCE_SHA": "1" * 40}
    done = _run(rig, tmp_path, env)
    assert done.returncode == 2
    assert "step git-fetch" in _status(tmp_path)["message"]
    assert not (tmp_path / "scratch" / "hl-venv").exists()


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"HYPERLOOM_SOURCE_DIR": ""}, "neither HYPERLOOM_SOURCE_SHA nor HYPERLOOM_SOURCE_DIR is set"),
        ({"HYPERLOOM_SOURCE_SHA": "abc123", "HYPERLOOM_SOURCE_REPO": "r"}, "not a full commit hash"),
        ({"HYPERLOOM_SOURCE_SHA": "A" * 40, "HYPERLOOM_SOURCE_REPO": "r"}, "not a lowercase hexadecimal"),
        ({"HYPERLOOM_SOURCE_SHA": "a" * 40}, "HYPERLOOM_SOURCE_REPO is not set"),
        ({"HYPERLOOM_SOURCE_DIR": "/nonexistent"}, "no requirements/runner.lock"),
    ],
)
def test_a_source_that_is_not_a_pinned_commit_or_a_checkout_is_refused(
    rig: Rig, tmp_path: Path, env: dict[str, str], message: str
) -> None:
    done = _run(rig, tmp_path, env)
    assert done.returncode == 2
    status = _status(tmp_path)
    assert "step source" in status["message"]
    assert message in status["message"]
    assert not (tmp_path / "scratch").exists()


def test_a_runner_that_does_not_import_ends_the_start_with_a_status(rig: Rig, tmp_path: Path) -> None:
    source = copy_source(rig, tmp_path / "source")
    main = source / "src/hyperloom/measure/runner/__main__.py"
    main.write_text("import module_that_is_not_there\n" + main.read_text())
    done = _run(rig, tmp_path, {"HYPERLOOM_SOURCE_DIR": str(source)})
    assert done.returncode == 2
    assert "ModuleNotFoundError" in done.stderr
    assert "step import" in _status(tmp_path)["message"]
    assert not (tmp_path / "out" / "probe.json").exists()


def _write_script(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_the_status_of_a_failed_step_is_json_whatever_the_step_printed(rig: Rig, tmp_path: Path) -> None:
    noisy = _write_script(
        tmp_path / "noisy-python",
        "#!/bin/sh\nprintf 'quote \" backslash \\\\ caf\\303\\251 tab\\t end\\n' >&2\nexit 3\n",
    )
    done = _run(rig, tmp_path, {"HYPERLOOM_BASE_PYTHON": str(noisy)})
    assert done.returncode == 2
    message = _status(tmp_path)["message"]
    assert message.startswith("runner bootstrap, step venv, exit 3: quote  backslash  caf tab")
    assert message.endswith("end")


def test_the_steps_run_without_the_signing_key_and_the_runner_keeps_it(rig: Rig, tmp_path: Path) -> None:
    seen = tmp_path / "seen"
    spy = _write_script(
        tmp_path / "spy-python",
        f'#!/bin/sh\necho "${{HYPERLOOM_SIGNING_KEY_FILE-unset}}" >> {seen}\nexec {rig.base_python} "$@"\n',
    )
    done = _run(rig, tmp_path, {"HYPERLOOM_BASE_PYTHON": str(spy), "HYPERLOOM_SIGNING_KEY_FILE": "/run/key"})
    assert done.returncode == 0, done.stderr
    assert seen.read_text().split() == ["unset", "unset"]
    assert read_probe(tmp_path / "out")["signing_key_env"] == "/run/key"


def test_the_script_is_posix_sh_and_the_command_carries_the_runner_arguments() -> None:
    assert BOOTSTRAP_SCRIPT.startswith("#!/bin/sh\n")
    subprocess.run(["/bin/sh", "-n", "-c", BOOTSTRAP_SCRIPT], check=True)
    command = runner_start_command("--spec-wait-sec", "30")
    assert command[:3] == ("/bin/sh", "-c", BOOTSTRAP_SCRIPT)
    assert command[4:] == ("--spec-wait-sec", "30")


def _date_shim(directory: Path, body: str) -> dict[str, str]:
    """A ``PATH`` that finds a ``date`` of the test's own first: the clock of an image whose ``date`` is not GNU's."""
    directory.mkdir()
    shim = directory / "date"
    shim.write_text(f"#!/bin/sh\n{body}\n")
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    return {"PATH": f"{directory}:{os.environ['PATH']}"}


def test_the_runner_is_told_the_unix_times_the_script_began_and_ended(rig: Rig, tmp_path: Path) -> None:
    before = time.time()
    done = _run(rig, tmp_path)
    after = time.time()

    assert done.returncode == 0, done.stderr
    probe = read_probe(tmp_path / "out")
    began, ended = float(str(probe["bootstrap_started"])), float(str(probe["bootstrap_ended"]))
    assert before - 1.0 <= began <= ended <= after, "the script ran between this test's two readings of the clock"
    assert ended - began > 0.1, "the venv, the lock install and the import check are the script's time"


def test_an_image_whose_date_has_no_sub_second_form_still_says_the_seconds_the_script_began_and_ended(
    rig: Rig, tmp_path: Path
) -> None:
    path = _date_shim(tmp_path / "bin", 'case "$*" in "+%s.%N") echo "1700000000.%N";; "+%s") echo 1700000000;; esac')

    assert _run(rig, tmp_path, path).returncode == 0

    probe = read_probe(tmp_path / "out")
    assert (probe["bootstrap_started"], probe["bootstrap_ended"]) == ("1700000000", "1700000000")


def test_an_image_whose_date_cannot_say_leaves_both_times_empty_and_the_runner_starts_all_the_same(
    rig: Rig, tmp_path: Path
) -> None:
    done = _run(rig, tmp_path, _date_shim(tmp_path / "bin", "exit 1"))

    assert done.returncode == 0, done.stderr
    probe = read_probe(tmp_path / "out")
    assert (probe["bootstrap_started"], probe["bootstrap_ended"]) == ("", "")
