# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A local agent sandbox under real bubblewrap: what its commands reach, a server one command starts and a later one
reads and stops (by ``pkill``, and by its PID file as Arbor does), and the kills and the stop that still reach every
process of it. The trees live outside ``/tmp``: before bubblewrap 0.12 nothing may be bound below the sandbox's own
``/tmp``, which every command shares."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from hyperloom.common.agent_isolation import BwrapIsolation
from hyperloom.measure.compile import PackageBase, UnsupportedComponentError
from hyperloom_common.bwrap import BwrapProbe
from hyperloom_common.tests._bwrap_host import unavailable, usable_bwrap
from hyperloom_multibranch.specialist import agent_sandbox
from hyperloom_multibranch.specialist.agent_sandbox import (
    LocalAgentSandbox,
    SandboxPaths,
    installed_frameworks,
    read_tail,
)
from hyperloom_multibranch.specialist.agent_watch import MARKER, PROCESS_MARKER, sandbox_pids

pytestmark = pytest.mark.agent_isolation

SECRET = "app-secret-canary-value"
PYTHON = shlex.quote(sys.executable)
#: A server: listens on loopback, says where in the sandbox's /tmp, then serves until it is stopped.
SERVER = r"""
import os, socket, sys
listener = socket.socket()
listener.bind(("127.0.0.1", 0))
listener.listen()
open("/tmp/server.port", "w").write(str(listener.getsockname()[1]))
open("/tmp/server.log", "w").write("started")
print("listening", flush=True)
while True:
    connection, _ = listener.accept()
    connection.sendall(b"reply")
    connection.close()
"""
CLIENT = r"""
import socket
print(socket.create_connection(("127.0.0.1", int(open("/tmp/server.port").read())), 5).recv(16).decode())
"""
#: Arbor's stop: the PID its file names is still its server (the start time ``/proc`` gives, after the boot time),
#: then that process group is signalled.
PID_FILE_SERVER = r"""
import os, sys, time
os.setsid()
open(sys.argv[1], "w").write(str(os.getpid()))
print("serving", flush=True)
time.sleep(1000)
"""
PID_FILE_STOP = r"""
import os, signal, sys, time
pid = int(open(sys.argv[1]).read())
started = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[19])
boot = next(int(line.split()[1]) for line in open("/proc/stat") if line.startswith("btime"))
assert boot > 0 and started > 0 and b"pid-file-server" in open(f"/proc/{pid}/cmdline", "rb").read()
os.killpg(pid, signal.SIGTERM)
while os.path.exists(f"/proc/{pid}"):
    time.sleep(0.05)
print("stopped", pid)
"""
#: Run inside: each named read (``ok:<text>`` or the errno's name), each named write, the environment, and what of the
#: test's process and of the app's environment it sees.
REACH = r"""
import errno, json, os, sys
def outcome(action):
    try:
        return "ok:" + (action() or "")
    except OSError as exc:
        return errno.errorcode.get(exc.errno, str(exc.errno))
def write(path):
    with open(path, "w") as handle:
        handle.write("from inside")
args = json.loads(sys.argv[1])
seen = {name: outcome(lambda path=path: open(path).read()) for name, path in args["reads"].items()}
seen.update({name: outcome(lambda path=path: write(path)) for name, path in args["writes"].items()})
seen["env"] = dict(os.environ)
seen["test_pid"] = outcome(lambda: os.kill(args["test_pid"], 0))
pids = [int(entry) for entry in os.listdir("/proc") if entry.isdigit()]
seen["test_pid_listed"] = args["test_pid"] in pids
leaked = []
for pid in pids:
    try:
        if args["secret"].encode() in open(f"/proc/{pid}/environ", "rb").read():
            leaked.append(pid)
    except OSError:
        pass
seen["leaked"] = leaked
print(json.dumps(seen))
"""


#: A session that starts a command in a confined sandbox, says its holder and scratch, then runs until it is killed.
OWNER = r"""
import asyncio, json, os, sys
from pathlib import Path
from hyperloom.common.agent_isolation import BwrapIsolation
from hyperloom_common.bwrap import bwrap_probe
from hyperloom_multibranch.specialist.agent_sandbox import LocalAgentSandbox, SandboxPaths

area, out, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]

async def main():
    sandbox = LocalAgentSandbox(
        cards=(0,), python_path=(), root=area / "lease", base_env=dict(os.environ),
        isolation=BwrapIsolation(bwrap_probe()), paths=SandboxPaths(writable=(area / "agent",)),
    )
    await sandbox.start(f"exec -a {name} sleep 1000", area, {})
    confined = sandbox._confined
    out.write_text(json.dumps([confined.holder.pid, confined.scratch.tmp, confined.scratch.shm]))
    while True:
        await asyncio.sleep(1)

asyncio.run(main())
"""


@pytest.fixture(scope="module")
def probe() -> BwrapProbe:
    return usable_bwrap()


@pytest.fixture
def area() -> Iterator[Path]:
    """A directory outside ``/tmp``, removed afterwards."""
    if not os.access("/var/tmp", os.W_OK):
        unavailable("no writable /var/tmp to hold a sandbox's trees outside /tmp")
    made = Path(tempfile.mkdtemp(prefix="hl-agent-sandbox-", dir="/var/tmp")).resolve()
    yield made
    shutil.rmtree(made, ignore_errors=True)


@pytest.fixture
def frameworks(area: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Path]]:
    """An installed aiter and vllm of the test's own, in a tree the sandbox reads."""
    site = area / "site-packages"
    found = {name: site / name for name in ("aiter", "vllm")}
    for path in found.values():
        path.mkdir(parents=True)
        (path / "__init__.py").write_text("")

    def locate(layout):
        if layout.import_name not in found:
            raise UnsupportedComponentError(f"{layout.import_name} is not installed")
        return PackageBase(layout.import_name, found[layout.import_name], layout.import_name, "0.0.0")

    monkeypatch.setattr(agent_sandbox, "locate_installed_package", locate)
    installed_frameworks.cache_clear()
    yield found
    installed_frameworks.cache_clear()


def _sandbox(area: Path, probe: BwrapProbe, **base: str) -> LocalAgentSandbox:
    agent, worktree = area / "agent", area / "worktree"
    agent.mkdir(exist_ok=True)
    worktree.mkdir(exist_ok=True)
    paths = SandboxPaths(writable=(agent,), readable=(worktree,), home=agent / "sandbox" / "home")
    return LocalAgentSandbox(
        cards=(0,),
        python_path=(),
        root=area / "lease",
        base_env={**os.environ, **base},
        grace_seconds=5.0,
        poll_seconds=0.05,
        isolation=BwrapIsolation(probe),
        paths=paths,
        forwarded={"ANTHROPIC_BASE_URL": "http://127.0.0.1:1/gateway"},
    )


async def _until(predicate: Callable[[], object], seconds: float = 20.0) -> None:
    until = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < until, "timed out waiting"
        await asyncio.sleep(0.05)


def _named(name: str) -> list[int]:
    """The processes of this host whose argv[0] is ``name``."""
    found = []
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                if Path(f"/proc/{entry}/cmdline").read_bytes().split(b"\0")[0] == name.encode():
                    found.append(int(entry))
            except OSError:
                continue
    return found


def _environ(pid: int) -> bytes:
    try:
        return Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return b""


async def test_a_server_one_command_started_is_read_and_stopped_by_a_later_one(area: Path, probe: BwrapProbe) -> None:
    sandbox = _sandbox(area, probe)
    try:
        server = await sandbox.start(f"exec {PYTHON} -I -S -c {shlex.quote(SERVER)} hl-test-server", area, {})
        await _until(lambda: "listening" in read_tail(server.log))
        later = await sandbox.run(
            f"cat /tmp/server.log; echo; {PYTHON} -I -S -c {shlex.quote(CLIENT)}; "
            "pgrep -f hl-test-server >/dev/null && echo seen; pkill -f hl-test-server; echo pkill=$?",
            area,
            {},
            60,
        )
        await _until(lambda: server.returncode is not None)
    finally:
        await sandbox.stop()

    assert later.returncode == 0, later.output
    assert later.output.split() == ["started", "reply", "seen", "pkill=0"], later.output
    assert server.returncode == -signal.SIGTERM


async def test_a_server_is_stopped_by_its_pid_file_in_the_sandbox_s_own_pid_namespace(
    area: Path, probe: BwrapProbe
) -> None:
    sandbox = _sandbox(area, probe)
    pid_file = area / "agent" / "server.pid"
    try:
        server = await sandbox.start(
            f"exec {PYTHON} -I -S -c {shlex.quote(PID_FILE_SERVER)} {pid_file} pid-file-server", area, {}
        )
        await _until(lambda: "serving" in read_tail(server.log))
        stop = await sandbox.run(f"{PYTHON} -I -S -c {shlex.quote(PID_FILE_STOP)} {pid_file}", area, {}, 60)
        await _until(lambda: server.returncode is not None)
    finally:
        await sandbox.stop()

    assert stop.returncode == 0, stop.output
    assert stop.output.split()[0] == "stopped" and int(stop.output.split()[1]) == int(pid_file.read_text())
    assert int(pid_file.read_text()) < 100, "the PID file names the server's pid in the sandbox's own namespace"
    assert server.returncode == -signal.SIGTERM


async def test_a_confined_command_reaches_its_trees_and_frameworks_and_nothing_of_the_app(
    area: Path, probe: BwrapProbe, frameworks: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    app_home, state, hf, base = area / "app-home", area / "state", area / "hf", area / "base"
    for path, text in (
        (app_home / ".claude" / ".credentials.json", "SECRET-LOGIN"),
        (state / "exit.json", "{}"),
        (hf / "token", "SECRET-HF"),
        (hf / "hub" / "model.txt", "weights"),
        (base / "third-party" / "header.h", "header"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    monkeypatch.setenv("HOME", str(app_home))
    monkeypatch.setenv("HYPERLOOM_STATE_DIR", str(state))
    sandbox = _sandbox(area, probe, HF_HOME=str(hf), HYPERLOOM_TEST_APP_VALUE=SECRET, GH_TOKEN=SECRET)
    worktree, agent = area / "worktree", area / "agent"
    (worktree / "source.py").write_text("source")
    (worktree / "3rdparty").symlink_to(base / "third-party")
    sandbox = LocalAgentSandbox(
        cards=sandbox.cards,
        python_path=(),
        root=sandbox.root,
        base_env=sandbox.base_env,
        isolation=sandbox.isolation,
        paths=SandboxPaths(sandbox.paths.writable, (worktree, base / "third-party"), sandbox.paths.home),
        forwarded=sandbox.forwarded,
    )
    home = agent / "sandbox" / "home"
    args = {
        "reads": {
            "app_login": str(app_home / ".claude" / ".credentials.json"),
            "state": str(state / "exit.json"),
            "hf_token": str(hf / "token"),
            "hf_model": str(hf / "hub" / "model.txt"),
            "worktree": str(worktree / "source.py"),
            "worktree_link": str(worktree / "3rdparty" / "header.h"),
        },
        "writes": {
            "agent": str(agent / "written"),
            "lease": str(sandbox.root / "written"),
            "home": str(home / ".claude" / "written"),
            "tmp": "/tmp/written",
            "shm": "/dev/shm/written",
            "worktree_write": str(worktree / "written"),
            "aiter": str(frameworks["aiter"] / "jit-module"),
            "vllm": str(frameworks["vllm"] / "patched.py"),
        },
        "test_pid": os.getpid(),
        "secret": SECRET,
    }
    check = agent / "reach.py"
    check.write_text(REACH)
    try:
        done = await sandbox.run(f"{PYTHON} -I -S {check} {shlex.quote(json.dumps(args))}", agent, {}, 60)
    finally:
        await sandbox.stop()
    assert done.returncode == 0, done.output
    seen = json.loads(done.output.strip().splitlines()[-1])
    env = seen.pop("env")

    assert seen == {
        "app_login": "ENOENT",
        "state": "ENOENT",
        "hf_token": "EACCES",
        "hf_model": "ok:weights",
        "worktree": "ok:source",
        "worktree_link": "ok:header",
        "agent": "ok:",
        "lease": "ok:",
        "home": "ok:",
        "tmp": "ok:",
        "shm": "ok:",
        "worktree_write": "EROFS",
        "aiter": "ok:",
        "vllm": "EROFS",
        "test_pid": "ESRCH",
        "test_pid_listed": False,
        "leaked": [],
    }
    assert (agent / "written").read_text() == "from inside" and not (worktree / "written").exists()
    assert SECRET not in json.dumps(env) and "GH_TOKEN" not in env and "HYPERLOOM_TEST_APP_VALUE" not in env
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:1/gateway" and env["HF_HOME"] == str(hf)
    assert (env["HOME"], env["CLAUDE_CONFIG_DIR"], env["TMPDIR"]) == (str(home), str(home / ".claude"), "/tmp")
    assert env["AITER_JIT_DIR"].startswith(f"{home}/") and env["TRITON_CACHE_DIR"].startswith(f"{home}/")
    assert env["FLYDSL_RUNTIME_CACHE_DIR"].startswith(f"{home}/") and env["PIP_USER"] == "1"
    assert env["PATH"].split(os.pathsep)[:2] == [str(sandbox.root / "bin"), str(home / ".local" / "bin")]
    assert env[MARKER] == str(sandbox.root) and env[PROCESS_MARKER].startswith(f"{sandbox.root}/")


async def test_the_kills_and_the_stop_reach_every_process_and_never_bubblewrap_s_own(
    area: Path, probe: BwrapProbe
) -> None:
    sandbox = _sandbox(area, probe)
    root = str(sandbox.root)
    # Each process is named apart from any other of this host (an argv[0] of its own), so none is taken for another's.
    name = f"hl-test-{os.getpid()}-{time.monotonic_ns()}"
    server, detached_name, beside = (f"{name}-{part}" for part in ("server", "detached", "beside"))
    first = await sandbox.start(
        f"(setsid env -i /bin/bash -c 'exec -a {name}-detached /bin/sleep 1000' &); "
        f"echo started; exec -a {name}-server sleep 1000",
        area,
        {},
    )
    second = await sandbox.start(f"echo started; exec -a {name}-beside sleep 1000", area, {})
    try:
        await _until(lambda: _named(detached_name) and _named(beside) and _named(server))
        holder = sandbox._confined.holder if sandbox._confined else None
        assert holder is not None
        bwrap_marked = MARKER.encode() in _environ(first.popen.pid)
        detached = _named(detached_name)
        await first.kill()
        after_kill = (_named(server), _named(detached_name), _named(beside))
        scratch = sandbox._confined.scratch if sandbox._confined else None
    finally:
        await sandbox.stop()

    assert not bwrap_marked, "bubblewrap's own process carries the sandbox's marker"
    assert detached and MARKER.encode() not in _environ(detached[0]), "the detached process carries no marker"
    assert after_kill[:2] == ([], []) and after_kill[2], "a command's kill ended its processes alone"
    assert not _named(beside) and sandbox_pids(root) == (set(), set())
    assert not os.path.exists(f"/proc/{holder.pid}"), "the stop ended the holder"
    assert scratch is not None and not os.path.exists(scratch.tmp) and not os.path.exists(scratch.shm)
    assert second.returncode is not None


async def test_a_command_that_ended_leaving_nothing_retires_its_launcher_and_bubblewrap(
    area: Path, probe: BwrapProbe
) -> None:
    sandbox = _sandbox(area, probe)
    try:
        done = await sandbox.run("echo ran", area, {}, 60)
        launchers = sandbox_pids(str(sandbox.root))[1]
        commands = agent_sandbox._launched.get(str(sandbox.root), [])
        ended = [popen.poll() is not None for popen in commands]
    finally:
        await sandbox.stop()

    assert done.output.strip() == "ran" and done.returncode == 0
    assert launchers == set(), "the launcher of a command that left nothing outlived it"
    assert ended and all(ended), "bubblewrap's process of a retired command still runs"


async def test_a_holder_that_ended_is_started_again_for_the_next_command(area: Path, probe: BwrapProbe) -> None:
    sandbox = _sandbox(area, probe)
    try:
        first = await sandbox.run("echo one > /tmp/kept; echo one", area, {}, 60)
        holder = sandbox._confined.holder if sandbox._confined else None
        assert holder is not None
        os.kill(holder.pid, signal.SIGKILL)
        await _until(lambda: not os.path.exists(f"/proc/{holder.pid}/ns"))
        second = await sandbox.run("cat /tmp/kept", area, {}, 60)
        again = sandbox._confined.holder if sandbox._confined else None
    finally:
        await sandbox.stop()

    assert (first.output.strip(), second.output.strip()) == ("one", "one"), second.output
    assert again is not None and again is not holder


def test_the_session_s_watch_ends_a_confined_sandbox_and_its_holder_once_the_session_is_killed(
    area: Path, probe: BwrapProbe
) -> None:
    (area / "agent").mkdir()
    script, out = area / "owner.py", area / "owner.json"
    script.write_text(OWNER)
    name = f"hl-test-{os.getpid()}-{time.monotonic_ns()}-session"
    owner = subprocess.Popen([sys.executable, str(script), str(area), str(out), name], env=dict(os.environ))
    holder, tmp, shm = 0, "", ""
    try:
        _wait(lambda: out.exists() and _named(name), 60, "the session's command did not start")
        holder, tmp, shm = json.loads(out.read_text())
        owner.kill()
        owner.wait()
        _wait(lambda: not _named(name), 60, "the command outlived the killed session")
        _wait(lambda: not os.path.exists(f"/proc/{holder}/ns"), 10, "the holder outlived the killed session")
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait()
        if holder and os.path.exists(f"/proc/{holder}/ns"):
            os.kill(holder, signal.SIGKILL)
        for path in (tmp, shm):
            if path:
                shutil.rmtree(path, ignore_errors=True)


def _wait(predicate: Callable[[], object], seconds: float, what: str) -> None:
    until = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < until, what
        time.sleep(0.1)
