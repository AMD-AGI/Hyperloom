# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The local runner template: the command names the sandbox's own directories, never the global ones."""

from __future__ import annotations

import subprocess
import sys

import pytest

from hyperloom.measure.local_templates import local_runner_command, local_runner_template


def _expanded(command: tuple[str, ...], root: str) -> str:
    """What the shell would run, with ``exec`` replaced by ``printf`` so nothing starts."""
    script = command[2].replace('exec "$0"', 'printf "%s\\n" "$0"')
    done = subprocess.run(
        [command[0], "-c", script, *command[3:]],
        env={"HYPERLOOM_SANDBOX_ROOT": root, "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout


def test_the_command_runs_the_runner_in_the_sandbox_directories() -> None:
    command = local_runner_command("/opt/py", spec_wait_sec=30)
    printed = _expanded(command, "/sb/1")
    assert printed.split() == [
        "/opt/py",
        "-m",
        "hyperloom.measure.runner",
        "run",
        "--in",
        "/sb/1/in",
        "--out",
        "/sb/1/out",
        "--scratch",
        "/sb/1/work/scratch",
        "--spec-wait-sec",
        "30",
    ]


def test_no_global_directory_is_named() -> None:
    script = local_runner_command()[2]
    for forbidden in (" /in", " /out", " /scratch", '"/in', '"/out', '"/scratch'):
        assert forbidden not in script


def test_the_default_python_is_this_interpreter() -> None:
    assert local_runner_command()[3] == sys.executable


def test_the_template_is_a_sealed_signing_measurement_template() -> None:
    template = local_runner_template("r", gpus=2, envs={"A": "b"})
    assert (template.role, template.sealed, template.signing, template.gpus) == ("measurement", True, True, 2)
    assert template.envs == {"A": "b"}
    assert template.command == local_runner_command()


def test_a_non_positive_wait_is_refused() -> None:
    with pytest.raises(ValueError, match="spec_wait_sec"):
        local_runner_command(spec_wait_sec=0)
