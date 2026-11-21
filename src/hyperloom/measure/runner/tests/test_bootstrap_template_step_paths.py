# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where the start step writes when the image's root is read-only.

The step's defaults (``/runner-boot.log``, ``/in``, ``/out``, ``/scratch``) sit at the image root. On a read-only image
the boot log moves (``HYPERLOOM_RUNNER_BOOT_LOG``, else the out directory, else ``TMPDIR``) and the start goes on; a
directory the step cannot make or write ends the start with a failure that names it, as only the platform can bind it.
A path below a missing directory or below a regular file stands in for one the account cannot write, which holds for
root too.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

import pytest

from hyperloom.measure.runner.bootstrap import BOOT_LOG_ENV, runner_template_step

_SHA = "0" * 40
_WAIT_SEC = 20.0
#: The runner the stand-in entry starts: it says it started, in the step's TMPDIR, and ends.
_STUB = 'touch "$TMPDIR/started"\n'


def _sandbox(root: Path, bootstrap: str = _STUB) -> tuple[dict[str, str], dict[str, object]]:
    """The sandbox's environment and the step's paths below ``root``, with a cache entry whose script is ``bootstrap``."""
    for name in ("tmp", "work"):
        (root / name).mkdir()
    entry = root / "runner-cache" / _SHA
    entry.mkdir(parents=True)
    (entry / "bootstrap.sh").write_text(bootstrap)
    manifest = f'{{\n  "bootstrap_sha256": "{hashlib.sha256(bootstrap.encode()).hexdigest()}",\n  "schema": 1\n}}\n'
    (entry / "manifest").write_text(manifest)
    env = {
        "PATH": os.environ["PATH"],
        "TMPDIR": str(root / "tmp"),
        "HYPERLOOM_OUT_DIR": str(root / "out"),
        "HYPERLOOM_RUNNER_CACHE": str(root / "runner-cache"),
        "HYPERLOOM_SOURCE_SHA": _SHA,
        "HYPERLOOM_RUNNER_CACHE_MANIFEST_SHA256": hashlib.sha256(manifest.encode()).hexdigest(),
    }
    paths: dict[str, object] = {
        "dirs": (str(root / "in"), str(root / "out"), str(root / "scratch")),
        "boot_log": str(root / "image-root" / "runner-boot.log"),  # its directory does not exist: a read-only root
    }
    return env, paths


def _run(step: str, env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/sh", "-c", step], env=dict(env), timeout=10, check=False, capture_output=True, text=True
    )


def _wait_for(path: Path) -> bool:
    deadline = time.monotonic() + _WAIT_SEC
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    return path.exists()


def _boot_log_of_a_start(log: Path) -> str:
    """The boot log of a start that ran the checked script: the two ``sha256sum -c`` lines."""
    deadline = time.monotonic() + _WAIT_SEC
    while log.exists() and log.read_text().count(": OK") < 2 and time.monotonic() < deadline:
        time.sleep(0.1)
    return log.read_text() if log.exists() else ""


def _status(out: Path) -> dict[str, str]:
    assert _wait_for(out / "status.json")
    return json.loads((out / "status.json").read_text())


def test_the_boot_log_goes_to_the_out_directory_when_the_image_root_is_read_only(tmp_path: Path) -> None:
    env, paths = _sandbox(tmp_path)
    assert _run(runner_template_step(**paths), env).returncode == 0  # type: ignore[arg-type]
    assert _wait_for(tmp_path / "tmp" / "started")
    assert _boot_log_of_a_start(tmp_path / "out" / "runner-boot.log").count(": OK") == 2
    assert not (tmp_path / "image-root").exists()
    assert not (tmp_path / "tmp" / "runner-boot.log").exists()


def test_the_boot_log_goes_to_tmpdir_when_the_out_directory_cannot_take_it_either(tmp_path: Path) -> None:
    env, paths = _sandbox(tmp_path)
    env["HYPERLOOM_OUT_DIR"] = str(tmp_path / "missing" / "out")
    assert _run(runner_template_step(**paths), env).returncode == 0  # type: ignore[arg-type]
    assert _wait_for(tmp_path / "tmp" / "started")
    assert _boot_log_of_a_start(tmp_path / "tmp" / "runner-boot.log").count(": OK") == 2
    assert not (tmp_path / "out" / "runner-boot.log").exists()


def test_the_template_boot_log_is_kept_where_it_can_be_written(tmp_path: Path) -> None:
    env, paths = _sandbox(tmp_path)
    paths["boot_log"] = str(tmp_path / "work" / "runner-boot.log")
    assert _run(runner_template_step(**paths), env).returncode == 0  # type: ignore[arg-type]
    assert _wait_for(tmp_path / "tmp" / "started")
    assert _boot_log_of_a_start(tmp_path / "work" / "runner-boot.log").count(": OK") == 2
    assert not (tmp_path / "out" / "runner-boot.log").exists()


def test_the_sandbox_variable_names_the_boot_log(tmp_path: Path) -> None:
    env, paths = _sandbox(tmp_path)
    paths["boot_log"] = str(tmp_path / "work" / "runner-boot.log")
    env[BOOT_LOG_ENV] = str(tmp_path / "work" / "named.log")
    assert BOOT_LOG_ENV == "HYPERLOOM_RUNNER_BOOT_LOG"
    assert _run(runner_template_step(**paths), env).returncode == 0  # type: ignore[arg-type]
    assert _wait_for(tmp_path / "tmp" / "started")
    assert _boot_log_of_a_start(tmp_path / "work" / "named.log").count(": OK") == 2
    assert not (tmp_path / "work" / "runner-boot.log").exists()


def test_a_boot_log_the_variable_names_and_the_step_cannot_write_ends_the_start(tmp_path: Path) -> None:
    env, paths = _sandbox(tmp_path)
    env[BOOT_LOG_ENV] = str(tmp_path / "missing" / "named.log")
    ran = _run(runner_template_step(**paths), env)  # type: ignore[arg-type]
    assert ran.returncode == 1
    reason = f"cannot write the boot log {tmp_path}/missing/named.log that HYPERLOOM_RUNNER_BOOT_LOG names"
    assert reason in ran.stderr
    status = _status(tmp_path / "out")
    assert status["error_class"] == "runner_start_failed" and reason in status["message"]
    assert not (tmp_path / "out" / "runner-boot.log").exists()
    time.sleep(0.5)
    assert not (tmp_path / "tmp" / "started").exists()


@pytest.mark.parametrize("which", [0, 2])
def test_a_directory_the_step_cannot_make_ends_the_start_naming_it(tmp_path: Path, which: int) -> None:
    env, paths = _sandbox(tmp_path)
    (tmp_path / "image-file").write_text("")
    dirs = list(paths["dirs"])  # type: ignore[call-overload]
    dirs[which] = str(tmp_path / "image-file" / "dir")
    paths["dirs"] = tuple(dirs)
    ran = _run(runner_template_step(**paths), env)  # type: ignore[arg-type]
    assert ran.returncode == 1
    reason = f"cannot make or write the directory {dirs[which]} "
    assert reason in ran.stderr and f"the platform must bind {dirs[which]} writable" in ran.stderr
    status = _status(tmp_path / "out")
    assert status["error_class"] == "runner_start_failed" and reason in status["message"]
    time.sleep(0.5)
    assert not (tmp_path / "tmp" / "started").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes a directory whatever its mode")
def test_a_directory_that_exists_but_is_not_writable_ends_the_start_naming_it(tmp_path: Path) -> None:
    env, paths = _sandbox(tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o555)
    try:
        ran = _run(runner_template_step(**paths), env)  # type: ignore[arg-type]
    finally:
        scratch.chmod(0o755)
    assert ran.returncode == 1
    assert f"cannot make or write the directory {scratch} " in ran.stderr
    assert f"cannot make or write the directory {scratch} " in _status(tmp_path / "out")["message"]
