# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner's start-up script inside a sandbox, through the socket: what the runner sees there, and the failure the
application reads when the install does not verify.

The contract suite's sealed template runs the same script, so every sealed test of that suite exercises it too; these
tests look at what the script did.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from hyperloom.measure.runner.bootstrap import runner_start_command
from hyperloom.measure.runner.tests.bootstrap_rig import PROBE_FILE, Rig, copy_source, lock_text
from hyperloom.sandbox.e2b import E2BSandboxProvider
from hyperloom.sandbox.socket import Sandbox
from hyperloom.sandbox.tests.e2b_standin import StandIn, StandInTemplate, interpreter_binds

_SPEC = b'{"id": 3}'


@pytest.fixture
def provider(e2b_envd: Path, bootstrap_rig: Rig, tmp_path: Path) -> Iterator[E2BSandboxProvider]:
    unverified = copy_source(
        bootstrap_rig,
        bootstrap_rig.root / f"source-{tmp_path.name}",
        lock_text(lock_pkg=("1.0", "0" * 64)),
    )
    binds = (*interpreter_binds(), str(bootstrap_rig.root))
    templates = [
        StandInTemplate(
            "good",
            "measurement",
            sealed=True,
            start=runner_start_command(),
            binds=binds,
            envs=bootstrap_rig.sandbox_env(),
        ),
        StandInTemplate(
            "unverified",
            "measurement",
            sealed=True,
            start=runner_start_command(),
            binds=binds,
            envs=bootstrap_rig.sandbox_env(unverified),
        ),
    ]
    with StandIn(e2b_envd, tmp_path, templates) as standin:
        yield E2BSandboxProvider(standin.keys, api_url=standin.api_url, sandbox_url=standin.sandbox_url)


def _wait_for(sandbox: Sandbox, path: str, seconds: float = 60.0) -> bytes:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            return sandbox.read_file(path)
        except FileNotFoundError:
            time.sleep(0.1)
    raise AssertionError(f"{path} did not appear within {seconds}s")


def test_the_runner_in_a_sandbox_runs_on_the_venv_the_script_built(
    provider: E2BSandboxProvider, bootstrap_rig: Rig
) -> None:
    sandbox = provider.create("good", role="measurement", timeout_sec=120, metadata={})
    sandbox.write_file("/in/spec.json", _SPEC)
    assert _wait_for(sandbox, "/out/status.json") == _SPEC
    probe = json.loads(sandbox.read_file(f"/out/{PROBE_FILE}"))
    assert probe["executable"] == "/work/scratch/hl-venv/bin/python"
    assert probe["base_only"] == str(bootstrap_rig.base_site / "base_only_pkg.py")
    assert probe["lock_origin"] == "lock"


def test_a_lock_that_does_not_verify_leaves_a_status_and_no_runner(provider: E2BSandboxProvider) -> None:
    sandbox = provider.create("unverified", role="measurement", timeout_sec=120, metadata={})
    status = json.loads(_wait_for(sandbox, "/out/status.json"))
    assert (status["state"], status["error_class"]) == ("failed", "runner_start_failed")
    assert "step lock" in status["message"]
    with pytest.raises(FileNotFoundError):
        sandbox.read_file(f"/out/{PROBE_FILE}")
