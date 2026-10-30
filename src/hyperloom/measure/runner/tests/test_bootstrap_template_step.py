# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The start step of a measurement template: one line that starts the checked runner in the background and returns.

A template's steps run before the sandbox service starts its agent, so a step that ran the runner in the foreground would
keep the sandbox from ever turning ready and the spec from ever arriving. The line is pinned to the step a joint run
measured working, and run here with ``/bin/sh`` against the rig's stand-in cache entry.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import signal
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from hyperloom.measure.runner.bootstrap import TEMPLATE_SPEC_WAIT_SEC, runner_start_shell_line, runner_template_step
from hyperloom.measure.runner.cache_layout import MANIFEST_NAME, MANIFEST_PIN_ENV
from hyperloom.measure.runner.tests.bootstrap_rig import PROBE_FILE, Rig, read_probe

#: The step of the joint run whose sandboxes turned ready (a key copy as the template's own preparation).
_MEASURED_STEP = (
    'mkdir -p /app/in /app/out /app/work/scratch && (umask 077 && cp "$HYPERLOOM_SIGNING_KEY_FILE" /app/w'
    "ork/hl-sign.pem) && (HYPERLOOM_SIGNING_KEY_FILE=/app/work/hl-sign.pem nohup sh -c 'e=\"${HYPERLOOM_RU"
    'NNER_CACHE:-/runner-cache}/${HYPERLOOM_SOURCE_SHA:-}" d="${TMPDIR:-/tmp}/hl-start.$$" o="${HYPERLOOM'
    '_OUT_DIR:-/out}"; if [ -n "${HYPERLOOM_SOURCE_SHA:-}" ] && [ -n "${HYPERLOOM_RUNNER_CACHE_MANIFEST_S'
    'HA256:-}" ] && mkdir -m 700 "$d" && cp "$e/manifest" "$e/bootstrap.sh" "$d/" && printf \'"\'"\'%s  %s\\n'
    '\'"\'"\' "$HYPERLOOM_RUNNER_CACHE_MANIFEST_SHA256" "$d/manifest" | sha256sum -c - >&2 && h=$(sed -n \'"\''
    '"\'s/^  "bootstrap_sha256": "\\([0-9a-f]\\{64\\}\\)",*$/\\1/p\'"\'"\' "$d/manifest") && [ -n "$h" ] && printf'
    ' \'"\'"\'%s  %s\\n\'"\'"\' "$h" "$d/bootstrap.sh" | sha256sum -c - >&2; then exec sh "$d/bootstrap.sh" --sp'
    'ec-wait-sec 1800; fi; echo "hyperloom-runner-bootstrap: bootstrap.sh or manifest of $e is missing or'
    ' not the one HYPERLOOM_RUNNER_CACHE_MANIFEST_SHA256 pins" >&2; mkdir -p "$o" && printf \'"\'"\'%s\\n\'"\'"'
    '\' \'"\'"\'{"error_class": "runner_start_failed", "message": "runner bootstrap, step verify, exit 1: boo'
    "tstrap.sh or manifest of the cache entry is missing or not the one HYPERLOOM_RUNNER_CACHE_MANIFEST_S"
    'HA256 pins, or that or HYPERLOOM_SOURCE_SHA is not set", "state": "failed"}\'"\'"\' >"$o/.status.json.t'
    'mp" && mv "$o/.status.json.tmp" "$o/status.json"; exit 2\' >/app/work/runner-boot.log 2>&1 &)'
)
#: The SHA-256 of that step as the run's file held it, with its final newline.
_MEASURED_SHA256 = "a9f37e7ea9b19f7c5ae6c2d79d9c1a0a27872dd1b96bb75ab1bf7b03c8edf36c"
_KEY_COPY = '(umask 077 && cp "$HYPERLOOM_SIGNING_KEY_FILE" /app/work/hl-sign.pem)'
_APP_PATHS = {"dirs": ("/app/in", "/app/out", "/app/work/scratch"), "boot_log": "/app/work/runner-boot.log"}
#: How long the step may take to return: it makes three directories and forks; the runner's start is not waited for.
_RETURNS_WITHIN_SEC = 3.0
#: How long the background runner may take to reach the point where it waits for the spec (a venv and a pip install).
_RUNNER_UP_SEC = 150.0
_SPEC = b'{"id": 7}'


def test_the_step_is_the_one_the_joint_run_measured() -> None:
    assert hashlib.sha256(f"{_MEASURED_STEP}\n".encode()).hexdigest() == _MEASURED_SHA256
    step = runner_template_step(
        "--spec-wait-sec",
        "1800",
        pre=_KEY_COPY,
        env={"HYPERLOOM_SIGNING_KEY_FILE": "/app/work/hl-sign.pem"},
        **_APP_PATHS,
    )
    assert step == _MEASURED_STEP
    subprocess.run(["/bin/sh", "-n", "-c", step], check=True)


def test_the_runner_waits_the_template_default_for_its_spec_unless_the_arguments_say() -> None:
    assert TEMPLATE_SPEC_WAIT_SEC == 1800
    assert runner_template_step() == runner_template_step("--spec-wait-sec", "1800")
    assert f"nohup sh -c {shlex.quote(runner_start_shell_line('--spec-wait-sec', '1800'))} >" in runner_template_step()
    for named in (("--spec-wait-sec", "30"), ("--spec-wait-sec=30",)):
        assert runner_template_step(*named).count("--spec-wait-sec") == 1


def test_the_step_without_preparation_or_environment_starts_the_runner_straight_after_the_directories() -> None:
    step = runner_template_step()
    assert step.startswith("mkdir -p /in /out /scratch && (nohup sh -c ")
    assert step.endswith(" >/runner-boot.log 2>&1 &)")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pre": "true\ntrue"}, "pre is one line of sh"),
        ({"env": {"A-B": "1"}}, "not shell variable names: 'A-B'"),
    ],
)
def test_a_step_that_could_not_be_one_line_of_sh_is_refused(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        runner_template_step(**kwargs)  # type: ignore[arg-type]


@pytest.fixture
def started() -> Iterator[list[int]]:
    """The process groups a test started (each led by the shell it ran); what is still alive of them is killed after."""
    groups: list[int] = []
    yield groups
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _wait_for(path: Path, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    return path.exists()


def test_the_step_returns_while_the_runner_goes_on_in_the_background(
    rig: Rig, tmp_path: Path, started: list[int]
) -> None:
    work = tmp_path / "work"
    dirs = {"in_dir": str(work / "in"), "out_dir": str(work / "out"), "scratch_dir": str(work / "scratch")}
    pin = hashlib.sha256((rig.cache.root / rig.cache.sha / MANIFEST_NAME).read_bytes()).hexdigest()
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "TMPDIR": str(tmp_path),
        "HYPERLOOM_IN_DIR": dirs["in_dir"],
        "HYPERLOOM_OUT_DIR": dirs["out_dir"],
        MANIFEST_PIN_ENV: pin,
        **rig.sandbox_env(scratch=dirs["scratch_dir"]),
    }
    began = time.monotonic()
    shell = subprocess.Popen(
        [
            "/bin/sh",
            "-c",
            runner_template_step(pre="true", dirs=tuple(dirs.values()), boot_log=str(work / "runner-boot.log")),
        ],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    started.append(shell.pid)
    try:
        returned = shell.wait(timeout=_RETURNS_WITHIN_SEC)
    except subprocess.TimeoutExpired:
        pytest.fail(f"the step did not return within {_RETURNS_WITHIN_SEC} s: it runs the runner in the foreground")
    assert returned == 0 and time.monotonic() - began < _RETURNS_WITHIN_SEC
    assert all(Path(directory).is_dir() for directory in dirs.values())
    assert _wait_for(work / "out" / PROBE_FILE, _RUNNER_UP_SEC), (work / "runner-boot.log").read_text()
    os.killpg(shell.pid, 0)  # the runner, started by the step that has returned, is still alive and waiting
    assert read_probe(work / "out")["argv"][-2:] == ["--spec-wait-sec", str(TEMPLATE_SPEC_WAIT_SEC)]
    (work / "in" / "spec.json").write_bytes(_SPEC)
    assert _wait_for(work / "out" / "status.json", 30.0)
    assert (work / "out" / "status.json").read_bytes() == _SPEC
    assert "bootstrap: cache" in (work / "runner-boot.log").read_text()
