# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner's start-up script: it installs only from the runner cache and touches no network, the venv it builds sees
the framework image, and a step that fails leaves the application a ``status.json`` that names what is wrong instead of
a runner that never comes.

The framework image is a temporary base venv (``bootstrap_rig``): its ``python3`` is a venv, as ``/opt/venv`` is, and
holds a module the runner needs that only it has. Every run's environment names a :class:`Tripwire` as the package
index, the find-links page and the proxy, and every test checks that nothing connected to it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from hyperloom.measure.runner.bootstrap import BOOTSTRAP_SCRIPT, runner_start_command
from hyperloom.measure.runner.cache_layout import MANIFEST_NAME, REQUIREMENTS_FILE
from hyperloom.measure.runner.tests.bootstrap_rig import (
    CacheRef,
    Rig,
    commit_variant,
    copy_cache,
    read_probe,
    run_script,
    tripwire,
)

_SPEC = b'{"id": 7}'
_START = runner_start_command("--spec-wait-sec", "20")
_LOCK_WHEEL = "lock_pkg-1.0-py3-none-any.whl"


def _run(
    rig: Rig, work: Path, env: dict[str, str] | None = None, cache: CacheRef | None = None
) -> subprocess.CompletedProcess[str]:
    """One run of the script with every network setting aimed at a tripwire that must stay quiet."""
    (work / "in").mkdir(parents=True, exist_ok=True)
    (work / "in" / "spec.json").write_bytes(_SPEC)
    with tripwire() as wire:
        done = run_script(_START, rig, work, {**wire.env, **(env or {})}, cache)
    assert wire.connections == 0, "the script tried to reach the network"
    return done


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
    # A pin the base does not satisfy is installed into the venv from the cache's wheels, and the venv's copy wins over
    # the base's older one. Its declared dependency is in no lock: nothing is resolved.
    assert probe["lock_origin"] == "lock"
    assert _venv_has(tmp_path, "lock_pkg")
    # A pin the index served only as a source distribution was built into a wheel by the builder, and is installed.
    assert probe["sdist_origin"] == "sdist" and _venv_has(tmp_path, "sdist_pkg")
    # The runner runs from the cache entry's source, and the commit is the cache entry's.
    assert probe["runner_file"] == str(rig.source / "src/hyperloom/measure/runner/__main__.py")
    assert probe["revision"] == rig.cache.sha
    assert (tmp_path / "out" / "status.json").read_bytes() == _SPEC
    # What pip installed from is the manifest's wheels, each pinned to the hash the manifest records.
    pins = (tmp_path / "scratch" / REQUIREMENTS_FILE).read_text()
    assert "lock-pkg==1.0" in pins and f"--hash=sha256:{rig.lock_pkg_sha256}" in pins


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


def _delete_wheel(entry: Path) -> None:
    (entry / "wheels" / _LOCK_WHEEL).unlink()


def _truncate_wheel(entry: Path) -> None:
    wheel = entry / "wheels" / _LOCK_WHEEL
    wheel.write_bytes(wheel.read_bytes()[:-8])


def _delete_manifest(entry: Path) -> None:
    (entry / MANIFEST_NAME).unlink()


def _edit_source(entry: Path) -> None:
    main = entry / "source" / "src/hyperloom/measure/runner/__main__.py"
    main.write_text(main.read_text() + "\n# edited after the build\n")


def _remove_wheels_dir(entry: Path) -> None:
    for wheel in (entry / "wheels").iterdir():
        wheel.unlink()
    (entry / "wheels").rmdir()


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (_delete_wheel, f"missing wheel {_LOCK_WHEEL} for lock-pkg==1.0"),
        (_truncate_wheel, f"wheel {_LOCK_WHEEL} for lock-pkg==1.0 is "),
        (_delete_manifest, "has no manifest: an unfinished build"),
        (_edit_source, "is not what the manifest recorded"),
        (_remove_wheels_dir, "lacks its source or wheels directory"),
    ],
)
def test_a_cache_entry_that_is_not_whole_is_refused_and_says_what_is_wrong(
    rig: Rig, tmp_path: Path, damage: Callable[[Path], None], message: str
) -> None:
    cache = copy_cache(rig, tmp_path / "cache")
    damage(cache.root / cache.sha)
    done = _run(rig, tmp_path, cache=cache)
    assert done.returncode == 2
    status = _status(tmp_path)
    assert (status["state"], status["error_class"]) == ("failed", "runner_start_failed")
    assert "step cache" in status["message"]
    assert message in status["message"]
    assert not (tmp_path / "scratch").exists()
    assert not (tmp_path / "out" / "probe.json").exists()


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"HYPERLOOM_SOURCE_SHA": ""}, "HYPERLOOM_SOURCE_SHA is not set"),
        ({"HYPERLOOM_SOURCE_SHA": "abc123"}, "is not a full lowercase commit hash"),
        ({"HYPERLOOM_SOURCE_SHA": "A" * 40}, "is not a full lowercase commit hash"),
        ({"HYPERLOOM_SOURCE_SHA": "../" + "a" * 37}, "is not a full lowercase commit hash"),
        ({"HYPERLOOM_SOURCE_SHA": "a" * 40}, f"has no entry for commit {'a' * 40}"),
        ({"HYPERLOOM_RUNNER_CACHE": "/nonexistent/cache"}, "the runner cache /nonexistent/cache does not exist"),
    ],
)
def test_a_cache_or_commit_that_is_not_there_ends_the_start_naming_it(
    rig: Rig, tmp_path: Path, env: dict[str, str], message: str
) -> None:
    done = _run(rig, tmp_path, env)
    assert done.returncode == 2
    status = _status(tmp_path)
    assert "step cache" in status["message"]
    assert message in status["message"]
    assert not (tmp_path / "scratch").exists()


def test_without_a_cache_directory_in_the_environment_the_documented_default_is_read(rig: Rig, tmp_path: Path) -> None:
    done = _run(rig, tmp_path, {"HYPERLOOM_RUNNER_CACHE": ""})
    assert done.returncode == 2
    assert "the runner cache /runner-cache does not exist" in _status(tmp_path)["message"]


def test_a_manifest_that_is_not_the_one_pinned_is_refused(rig: Rig, tmp_path: Path) -> None:
    pinned = {"HYPERLOOM_RUNNER_CACHE_MANIFEST_SHA256": "0" * 64}
    done = _run(rig, tmp_path, pinned)
    assert done.returncode == 2
    assert "is not the one pinned by HYPERLOOM_RUNNER_CACHE_MANIFEST_SHA256" in _status(tmp_path)["message"]


def test_the_manifest_that_is_pinned_is_accepted(rig: Rig, tmp_path: Path) -> None:
    digest = hashlib.sha256((rig.source.parent / MANIFEST_NAME).read_bytes()).hexdigest()
    done = _run(rig, tmp_path, {"HYPERLOOM_RUNNER_CACHE_MANIFEST_SHA256": digest})
    assert done.returncode == 0, done.stderr


def test_a_wheel_whose_content_is_not_its_hash_installs_nothing_and_leaves_a_status(rig: Rig, tmp_path: Path) -> None:
    cache = copy_cache(rig, tmp_path / "cache")
    wheel = cache.root / cache.sha / "wheels" / _LOCK_WHEEL
    damaged = bytearray(wheel.read_bytes())
    damaged[len(damaged) // 2] ^= 0xFF
    wheel.write_bytes(bytes(damaged))
    done = _run(rig, tmp_path, cache=cache)
    assert done.returncode == 2
    assert "DO NOT MATCH THE HASHES" in done.stderr
    status = _status(tmp_path)
    assert (status["state"], status["error_class"]) == ("failed", "runner_start_failed")
    assert "step lock" in status["message"]
    assert not _venv_has(tmp_path, "lock_pkg")
    assert not (tmp_path / "out" / "probe.json").exists()


def test_a_runner_that_does_not_import_ends_the_start_with_a_status(rig: Rig, tmp_path: Path) -> None:
    def break_the_runner(repo: Path) -> None:
        main = repo / "src/hyperloom/measure/runner/__main__.py"
        main.write_text("import module_that_is_not_there\n" + main.read_text())

    (tmp_path / "variant").mkdir()
    cache = commit_variant(rig, tmp_path / "variant", break_the_runner)
    done = _run(rig, tmp_path, cache=cache)
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
    assert message.startswith("runner bootstrap, step cache, exit 3: quote  backslash  caf tab")
    assert message.endswith("end")


def test_the_steps_run_without_the_signing_key_and_the_runner_keeps_it(rig: Rig, tmp_path: Path) -> None:
    seen = tmp_path / "seen"
    spy = _write_script(
        tmp_path / "spy-python",
        f'#!/bin/sh\necho "${{HYPERLOOM_SIGNING_KEY_FILE-unset}}" >> {seen}\nexec {rig.base_python} "$@"\n',
    )
    done = _run(rig, tmp_path, {"HYPERLOOM_BASE_PYTHON": str(spy), "HYPERLOOM_SIGNING_KEY_FILE": "/run/key"})
    assert done.returncode == 0, done.stderr
    assert seen.read_text().split() == ["unset", "unset", "unset"]
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


def test_the_script_has_no_network_path_in_its_text() -> None:
    text = BOOTSTRAP_SCRIPT.replace("\\\n", " ")
    # No git, no source repository: the source comes from the cache.
    assert "SOURCE_REPO" not in text and not re.search(r"^\s*step git", text, re.MULTILINE)
    # Every pip run is told to use no index and no pip configuration of the environment.
    pips = [line for line in text.splitlines() if " -m pip " in line]
    assert pips and all("--no-index" in line and "--isolated" in line for line in pips)
    # The embedded checker is the cache module's own text, and the heredoc that carries it ends only at its end.
    assert text.count("\nPY\n") == 2
