# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner's start-up script inside a sandbox, through the socket: with no network and the runner cache mounted
read-only, what the runner sees there, and the failures the application reads when the cache does not verify.

The measurement role's sandboxes have no egress and the template mounts the cache of source and wheels read-only; the
stand-in reproduces both (``without_network`` for the start command, a read-only bind for the cache). The contract
suite's sealed template runs the same script, so every sealed test of that suite exercises it too; these tests look at
what the script did.
"""

from __future__ import annotations

import json
import shutil
import socket
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from hyperloom.measure.runner.bootstrap import runner_start_command
from hyperloom.measure.runner.cache_layout import MANIFEST_NAME
from hyperloom.measure.runner.tests.bootstrap_rig import PROBE_FILE, Rig, copy_cache
from hyperloom.sandbox.e2b import E2BSandboxProvider
from hyperloom.sandbox.socket import Sandbox
from hyperloom.sandbox.tests.e2b_standin import StandIn, StandInTemplate, interpreter_binds, without_network

_SPEC = b'{"id": 3}'
#: Connects to the host's listener from inside the sandbox and writes what happened to ``/out/net``.
_CONNECT = (
    "import socket, sys\n"
    "try:\n"
    "    socket.create_connection(('127.0.0.1', int(sys.argv[1])), timeout=3).close()\n"
    "    answer = 'reachable'\n"
    "except OSError as exc:\n"
    "    answer = 'unreachable: ' + type(exc).__name__\n"
    "open('/out/net', 'w').write(answer)\n"
)


def _connect_start(port: int) -> tuple[str, ...]:
    return (sys.executable, "-c", _CONNECT, str(port))


@pytest.fixture
def listener() -> Iterator[int]:
    """A listening TCP port on the host's loopback, which a sandbox with a network can connect to."""
    with socket.create_server(("127.0.0.1", 0)) as server:
        yield server.getsockname()[1]


@pytest.fixture
def provider(e2b_envd: Path, bootstrap_rig: Rig, tmp_path: Path, listener: int) -> Iterator[E2BSandboxProvider]:
    broken = bootstrap_rig.root / f"cache-{tmp_path.name}"
    damaged = copy_cache(bootstrap_rig, broken)
    (damaged.root / damaged.sha / MANIFEST_NAME).unlink()
    binds = (*interpreter_binds(), str(bootstrap_rig.root))
    templates = [
        StandInTemplate(
            "good",
            "measurement",
            sealed=True,
            start=without_network(runner_start_command()),
            binds=binds,
            envs=bootstrap_rig.sandbox_env(),
        ),
        StandInTemplate(
            "unfinished",
            "measurement",
            sealed=True,
            start=without_network(runner_start_command()),
            binds=binds,
            envs=bootstrap_rig.sandbox_env(damaged),
        ),
        StandInTemplate(
            "offline", "measurement", sealed=True, start=without_network(_connect_start(listener)), binds=binds
        ),
        StandInTemplate("online", "measurement", sealed=True, start=_connect_start(listener), binds=binds),
        StandInTemplate("read-only", "measurement", sealed=True, binds=binds),
    ]
    with StandIn(e2b_envd, tmp_path, templates) as standin:
        yield E2BSandboxProvider(standin.keys, api_url=standin.api_url, sandbox_url=standin.sandbox_url)
    shutil.rmtree(broken, ignore_errors=True)


def _wait_for(sandbox: Sandbox, path: str, seconds: float = 60.0) -> bytes:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            return sandbox.read_file(path)
        except FileNotFoundError:
            time.sleep(0.1)
    raise AssertionError(f"{path} did not appear within {seconds}s")


def test_the_runner_in_a_sandbox_without_network_runs_on_the_venv_the_script_built_from_the_cache(
    provider: E2BSandboxProvider, bootstrap_rig: Rig
) -> None:
    sandbox = provider.create("good", role="measurement", timeout_sec=120, metadata={})
    sandbox.write_file("/in/spec.json", _SPEC)
    assert _wait_for(sandbox, "/out/status.json") == _SPEC
    probe = json.loads(sandbox.read_file(f"/out/{PROBE_FILE}"))
    assert probe["executable"] == "/work/scratch/hl-venv/bin/python"
    assert probe["base_only"] == str(bootstrap_rig.base_site / "base_only_pkg.py")
    assert probe["lock_origin"] == "lock"
    assert probe["runner_file"] == str(bootstrap_rig.source / "src/hyperloom/measure/runner/__main__.py")
    assert probe["revision"] == bootstrap_rig.cache.sha


def test_the_stand_in_blocks_the_network_of_the_start_command_and_only_that(
    provider: E2BSandboxProvider,
) -> None:
    online = provider.create("online", role="measurement", timeout_sec=60, metadata={})
    assert _wait_for(online, "/out/net") == b"reachable"
    offline = provider.create("offline", role="measurement", timeout_sec=60, metadata={})
    assert _wait_for(offline, "/out/net").decode().startswith("unreachable: ")
    assert offline.run("echo still-served").stdout == "still-served\n"


def test_a_cache_that_is_not_whole_leaves_a_status_and_no_runner(provider: E2BSandboxProvider) -> None:
    sandbox = provider.create("unfinished", role="measurement", timeout_sec=120, metadata={})
    status = json.loads(_wait_for(sandbox, "/out/status.json"))
    assert (status["state"], status["error_class"]) == ("failed", "runner_start_failed")
    assert "step cache" in status["message"] and "has no manifest: an unfinished build" in status["message"]
    with pytest.raises(FileNotFoundError):
        sandbox.read_file(f"/out/{PROBE_FILE}")


def test_the_cache_is_mounted_read_only(provider: E2BSandboxProvider, bootstrap_rig: Rig) -> None:
    sandbox = provider.create("read-only", role="measurement", timeout_sec=60, metadata={})
    entry = bootstrap_rig.cache.root / bootstrap_rig.cache.sha
    assert sandbox.run(f"ls {entry}").stdout.split() == ["manifest", "source", "wheels"]
    written = sandbox.run(f"touch {entry}/source/new-file")
    assert written.exit_code != 0 and "Read-only file system" in written.stderr
