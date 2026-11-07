# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A local agent sandbox under real bubblewrap: what its commands reach, a server one command starts and a later one
reads and stops (by ``pkill``, and by its PID file as Arbor does), the kills and the stop that still reach every
process of it, and where pip installs in it, in a venv laid out as the image's: for Arbor, which reads ``ARBOR_SITE``,
never for a server. The trees live outside ``/tmp``: before bubblewrap 0.12 nothing may be bound below the sandbox's
own ``/tmp``, which every command shares."""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable, Iterator, Mapping
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
from hyperloom_multibranch.specialist.arbor_adapter import ARBOR_PYTHON, INSTALL, SITE_ENV

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


#: Run inside: each named read (``ok:<text>`` or the errno's name) at start, and again once the trigger file exists,
#: which the test makes when the app has replaced and made its logins.
LOGIN_WATCH = r"""
import errno, json, os, sys, time
def outcome(path):
    try:
        return "ok:" + open(path).read()
    except OSError as exc:
        return errno.errorcode.get(exc.errno, str(exc.errno))
paths, trigger = json.loads(sys.argv[1]), sys.argv[2]
print("before " + json.dumps({name: outcome(path) for name, path in paths.items()}), flush=True)
while not os.path.exists(trigger):
    time.sleep(0.05)
print("after " + json.dumps({name: outcome(path) for name, path in paths.items()}), flush=True)
"""


#: Arbor's end of a turn (its ``arbor/execute/turns.py``, F9): SIGTERM every process of its own uid, itself aside,
#: whose environment carries ``ARBOR_TURN=<tag>``, then wait for them; print the pids found, whether each also carries
#: the variable its command was given, and those still running.
TURN_END = r"""
import json, os, signal, sys, time
from pathlib import Path
want = ("ARBOR_TURN=" + sys.argv[1]).encode()
uid, me = os.getuid(), os.getpid()
def running(pid):
    try:
        return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False
found, given = [], []
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit() or int(entry.name) == me:
        continue
    try:
        if entry.stat().st_uid != uid:
            continue
        env = (entry / "environ").read_bytes().split(b"\0")
    except OSError:
        continue
    if want in env:
        found.append(int(entry.name))
        given.append(b"HL_TEST_GIVEN=given" in env)
for pid in found:
    os.kill(pid, signal.SIGTERM)
until = time.monotonic() + 20
while any(map(running, found)) and time.monotonic() < until:
    time.sleep(0.05)
print(json.dumps({"found": found, "given": given, "running": [pid for pid in found if running(pid)]}))
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


def _sandbox(area: Path, probe: BwrapProbe, *, readable: tuple[Path, ...] = (), **base: str) -> LocalAgentSandbox:
    agent, worktree = area / "agent", area / "worktree"
    agent.mkdir(exist_ok=True)
    worktree.mkdir(exist_ok=True)
    paths = SandboxPaths(writable=(agent,), readable=(worktree, *readable), home=agent / "sandbox" / "home")
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


async def test_what_a_turn_left_running_is_found_by_its_tag_and_stopped_by_a_later_command_of_its_sandbox_alone(
    area: Path, probe: BwrapProbe
) -> None:
    """Arbor's F9: a process an agent turn left running, detached, carries the tag a process inside set for its
    children, and the variables its command was given; a later command of the sandbox (Arbor, at the turn's end)
    finds it among the processes of its own uid by its environment, and stops it. Another sandbox's Arbor, whose own
    leftover carries the same tag, finds and stops its own alone."""
    (area / "other").mkdir()
    ours, theirs = _sandbox(area, probe), _sandbox(area / "other", probe)
    name = f"hl-test-{os.getpid()}-{time.monotonic_ns()}"
    leave = "(ARBOR_TURN=t1 setsid bash -c 'exec -a {} sleep 600' </dev/null >/dev/null 2>&1 &); echo left"
    for sandbox in (ours, theirs):
        (sandbox.paths.writable[0] / "turn_end.py").write_text(TURN_END)
    try:
        left = [
            await sandbox.run(leave.format(f"{name}-{part}"), sandbox.paths.writable[0], {"HL_TEST_GIVEN": "given"}, 60)
            for sandbox, part in ((ours, "ours"), (theirs, "theirs"))
        ]
        await _until(lambda: _named(f"{name}-ours") and _named(f"{name}-theirs"))
        tagged = b"ARBOR_TURN=t1" in _environ(_named(f"{name}-ours")[0]).split(b"\0")
        end = f"{PYTHON} -I -S turn_end.py t1"
        their_end = await theirs.run(end, theirs.paths.writable[0], {}, 60)
        after_theirs = (_named(f"{name}-ours"), _named(f"{name}-theirs"))
        our_end = await ours.run(end, ours.paths.writable[0], {}, 60)
        after_ours = _named(f"{name}-ours")
    finally:
        await ours.stop()
        await theirs.stop()

    assert [done.output.strip() for done in left] == ["left", "left"], left
    assert tagged, "the leftover does not carry the tag its shell set for it"
    assert their_end.returncode == 0 and our_end.returncode == 0, (their_end.output, our_end.output)
    theirs_seen, ours_seen = (json.loads(done.output.strip().splitlines()[-1]) for done in (their_end, our_end))
    assert len(theirs_seen["found"]) == 1 and theirs_seen["given"] == [True] and theirs_seen["running"] == []
    assert after_theirs[0] and not after_theirs[1], "the other sandbox's Arbor stopped a process of this one"
    assert len(ours_seen["found"]) == 1 and ours_seen["given"] == [True] and ours_seen["running"] == []
    assert not after_ours, "the turn's leftover outlived the turn's end"


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
        "hf_token": "ENOENT",
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
    assert "AITER_JIT_DIR" not in env and "FLYDSL_RUNTIME_CACHE_DIR" not in env, "aiter's own, in its writable tree"
    assert env["TRITON_CACHE_DIR"].startswith(f"{home}/") and env["PIP_TARGET"] == str(home / "site-packages")
    assert "PIP_USER" not in env and SITE_ENV not in env and str(home) not in env.get("PYTHONPATH", "")
    assert env["PATH"].split(os.pathsep)[0] == str(sandbox.root / "bin")
    assert not [entry for entry in env["PATH"].split(os.pathsep) if entry.startswith(f"{home}/")], "a home's bin"
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


# ---- where pip installs, in a venv laid out as the image's -----------------------------------------------------------

#: What an inference server started in the sandbox sees: whether the SDK imports, which hl_shadow (the image's, or the
#: one the SDK's install brought), its import path and ``ARBOR_SITE``.
SERVER_SEES = r"""
import json, os, sys
try:
    import claude_agent_sdk
    sdk = "imported"
except ImportError:
    sdk = "absent"
import hl_shadow
print(json.dumps({"sdk": sdk, "shadow": hl_shadow.VALUE, "path": sys.path, "site": os.environ.get("ARBOR_SITE")}))
"""
#: An Arbor that reads ``ARBOR_SITE``: the directory on its own import path, at its import.
ARBOR_INIT = r"""
import os, sys
_site = os.environ.get("ARBOR_SITE")
if _site and _site not in sys.path:
    sys.path.insert(0, _site)
"""
#: Arbor's driver: imports the SDK, starts a server from its shell as its server turn does, says what each saw.
ARBOR_MAIN = r"""
import json, subprocess, sys
import claude_agent_sdk, hl_shadow
server = subprocess.run(["bash", "-c", "python3 server_sees.py"], capture_output=True, text=True)
if server.returncode:
    sys.exit(server.stderr)
print(json.dumps({"sdk": claude_agent_sdk.SHADOW, "shadow": hl_shadow.VALUE, "server": json.loads(server.stdout)}))
"""


@dataclasses.dataclass(frozen=True)
class Image:
    """A venv laid out as the image's ``/opt/venv`` and the wheels an index would serve."""

    venv: Path
    wheels: Path

    def index(self) -> dict[str, str]:
        """What a command is given for pip to find :attr:`wheels` alone, the index's stand-in."""
        return {"PIP_NO_INDEX": "1", "PIP_FIND_LINKS": str(self.wheels), "PIP_DISABLE_PIP_VERSION_CHECK": "1"}


@pytest.fixture(scope="module")
def image() -> Iterator[Image]:
    """``python3 -m venv`` without system site packages, so with pip and without a user site, holding hl-shadow 1.0
    (the image's own packages); and wheels of PyYAML, of claude-agent-sdk, which needs hl-shadow 2 or later, and of
    hl-shadow 2.0, all built here."""
    if not os.access("/var/tmp", os.W_OK):
        unavailable("no writable /var/tmp to hold an image's venv outside /tmp")
    made = Path(tempfile.mkdtemp(prefix="hl-image-", dir="/var/tmp")).resolve()
    venv, wheels, own = made / "venv", made / "wheels", made / "own"
    wheels.mkdir()
    own.mkdir()
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, capture_output=True)
    shadow = _wheel(own, "hl-shadow", "1.0", {"hl_shadow/__init__.py": 'VALUE = "image"\n'})
    pip = [str(venv / "bin" / "python3"), "-m", "pip", "install", "--no-index", "--quiet", str(shadow)]
    environ = {name: value for name, value in os.environ.items() if not name.startswith("PIP_")}
    subprocess.run(pip, check=True, capture_output=True, env={**environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
    _wheel(wheels, "PyYAML", "6.0", {"yaml/__init__.py": ""})
    sdk = {"claude_agent_sdk/__init__.py": "import hl_shadow\nSHADOW = hl_shadow.VALUE\n"}
    _wheel(wheels, "claude-agent-sdk", "1.0", sdk, requires=("hl-shadow>=2",))
    _wheel(wheels, "hl-shadow", "2.0", {"hl_shadow/__init__.py": 'VALUE = "target"\n'})
    yield Image(venv, wheels)
    shutil.rmtree(made, ignore_errors=True)


def _wheel(directory: Path, name: str, version: str, files: Mapping[str, str], requires: tuple[str, ...] = ()) -> Path:
    """A pure-Python wheel of ``files``, written here: no index, no build backend."""
    dist = name.replace("-", "_")
    info = f"{dist}-{version}.dist-info"
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    contents = {
        **files,
        f"{info}/METADATA": metadata + "".join(f"Requires-Dist: {requirement}\n" for requirement in requires),
        f"{info}/WHEEL": "Wheel-Version: 1.0\nGenerator: hyperloom-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = [f"{member},sha256={_digest(text)},{len(text.encode())}" for member, text in contents.items()]
    wheel = directory / f"{dist}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for member, text in contents.items():
            archive.writestr(member, text)
        archive.writestr(f"{info}/RECORD", "\n".join([*record, f"{info}/RECORD,,"]) + "\n")
    return wheel


def _digest(text: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).rstrip(b"=").decode()


def _image_sandbox(area: Path, probe: BwrapProbe, image: Image) -> LocalAgentSandbox:
    """A sandbox whose ``python3`` is ``image``'s venv, bound read-only whole, as the image's ``/opt`` is."""
    agent = area / "agent"
    agent.mkdir(exist_ok=True)
    return LocalAgentSandbox(
        cards=(0,),
        python_path=(),
        root=area / "lease",
        base_env={**os.environ, "PATH": f"{image.venv / 'bin'}{os.pathsep}{os.environ['PATH']}"},
        grace_seconds=5.0,
        poll_seconds=0.05,
        isolation=BwrapIsolation(probe),
        paths=SandboxPaths(writable=(agent,), readable=(image.venv, image.wheels), home=agent / "sandbox" / "home"),
    )


def _arbor(agent: Path, init: str) -> Path:
    """An Arbor checkout in ``agent`` whose package's import runs ``init``."""
    checkout = agent / "arbor-checkout"
    for name, text in (("arbor/__init__.py", init), ("arbor/__main__.py", ARBOR_MAIN), ("server_sees.py", SERVER_SEES)):
        (checkout / name).parent.mkdir(parents=True, exist_ok=True)
        (checkout / name).write_text(text)
    return checkout


async def test_what_pip_installs_in_a_confined_sandbox_imports_for_arbor_alone_never_for_a_server(
    area: Path, probe: BwrapProbe, image: Image
) -> None:
    """Arbor's install puts the SDK, and every package its install brings, in the home's site directory, out of the
    image's venv; Arbor's commands, given it as ARBOR_SITE, import them; a server started from Arbor's shell, or from a
    plain shell of the sandbox, imports the image's packages, and its import path has no site directory."""
    sandbox = _image_sandbox(area, probe, image)
    checkout = _arbor(area / "agent", ARBOR_INIT)
    site = area / "agent" / "sandbox" / "home" / "site-packages"
    as_arbor = (
        'import os, sys; sys.path.insert(0, os.environ["ARBOR_SITE"]); '
        "import claude_agent_sdk; print(claude_agent_sdk.SHADOW)"
    )
    try:
        # INSTALL[0] is the Python 3.11 gate, not where the SDK goes.
        installed = [await sandbox.run(command, checkout, image.index(), 300) for command in INSTALL[1:]]
        driver = await sandbox.run(f"{ARBOR_PYTHON} -m arbor", checkout, {}, 120)
        simulated = await sandbox.run(f"{ARBOR_PYTHON} -c {shlex.quote(as_arbor)}", area, {}, 60)
        plain = await sandbox.run("python3 server_sees.py", checkout, {}, 60)
    finally:
        await sandbox.stop()

    assert [done.returncode for done in installed] == [0, 0, 0], [done.output for done in installed]
    packages = sorted(entry.name for entry in site.iterdir() if not entry.name.endswith(".dist-info"))
    assert packages == ["claude_agent_sdk", "hl_shadow", "yaml"]
    assert not list(image.venv.glob("lib/python*/site-packages/claude_agent_sdk")), "the install wrote the venv"
    assert driver.returncode == 0, driver.output
    seen = json.loads(driver.output.strip().splitlines()[-1])
    server = seen.pop("server")
    assert seen == {"sdk": "target", "shadow": "target"}, "Arbor imports the SDK and what its install brought"
    assert (simulated.returncode, simulated.output.strip()) == (0, "target"), simulated.output
    assert plain.returncode == 0, plain.output
    alone = json.loads(plain.output.strip().splitlines()[-1])
    for sees in (server, alone):
        assert (sees["sdk"], sees["shadow"]) == ("absent", "image"), sees
        assert not [entry for entry in sees["path"] if entry.startswith(str(site))], sees["path"]
    assert (server["site"], alone["site"]) == (str(site), None), "ARBOR_SITE is Arbor's commands' alone"


async def test_an_arbor_that_does_not_read_arbor_site_fails_its_install_saying_why(
    area: Path, probe: BwrapProbe, image: Image
) -> None:
    sandbox = _image_sandbox(area, probe, image)
    checkout = _arbor(area / "agent", "")
    site = area / "agent" / "sandbox" / "home" / "site-packages"
    try:
        installed = [await sandbox.run(command, checkout, image.index(), 300) for command in INSTALL[1:]]
    finally:
        await sandbox.stop()

    assert [done.returncode for done in installed] == [0, 0, 1], [done.output for done in installed]
    assert (site / "claude_agent_sdk").is_dir() and "No module named 'yaml'" in installed[-1].output
    said = f"this Arbor does not read ARBOR_SITE, so it cannot import what pip installed in {site}: pin Arbor 545a850"
    assert said in installed[-1].output, installed[-1].output


async def test_the_image_s_venv_takes_neither_a_user_install_nor_one_into_itself(
    area: Path, probe: BwrapProbe, image: Image
) -> None:
    """Why a confined pip installs to a target of its own: the image's venv refuses the user site the sandbox once
    named (``PIP_USER=1``), and is read-only."""
    sandbox = _image_sandbox(area, probe, image)
    install = "python3 -m pip install claude-agent-sdk"
    try:
        user = await sandbox.run(f"env -u PIP_TARGET PIP_USER=1 {install}", area, image.index(), 300)
        into = await sandbox.run(f"env -u PIP_TARGET {install}", area, image.index(), 300)
    finally:
        await sandbox.stop()

    assert user.returncode != 0, user.output
    assert "Can not perform a '--user' install. User site-packages are not visible in this virtualenv." in user.output
    assert into.returncode != 0 and "Read-only file system" in into.output, into.output


@pytest.mark.parametrize("named", [False, True], ids=["searched", "named"])
@pytest.mark.parametrize("logged_in", [True, False], ids=["replaced", "made"])
async def test_a_login_the_app_replaces_or_makes_while_a_command_runs_never_shows_in_it(
    area: Path, probe: BwrapProbe, logged_in: bool, named: bool
) -> None:
    """The app's Hugging Face home, which a sandbox reads for its hub cache, holds its token beside the cache, or no
    login yet. While a command runs in the sandbox, the app writes the token by rename (as a login does) and
    ``stored_tokens`` too: the command sees neither, before or after, and still reads the cache and the file beside
    it. That holds whether the home is reached through ``HF_HOME`` (:func:`toolchain_paths` binds what is beside its
    logins) or named for the sandbox to read whole (:meth:`Confinement.policy` binds it entry by entry without
    them)."""
    hf = area / "hf"
    token = hf / "token"
    (hf / "hub").mkdir(parents=True)
    (hf / "hub" / "model.txt").write_text("weights")
    (hf / "note").write_text("beside")
    if logged_in:
        token.write_text("SECRET-HF")
    sandbox = _sandbox(area, probe, readable=(hf,) if named else (), HF_HOME=str(hf))
    agent = area / "agent"
    trigger, watch = agent / "go", agent / "watch.py"
    watch.write_text(LOGIN_WATCH)
    paths = {
        "token": str(token),
        "stored": str(hf / "stored_tokens"),
        "model": str(hf / "hub" / "model.txt"),
        "note": str(hf / "note"),
    }
    try:
        command = await sandbox.start(f"{PYTHON} -I -S {watch} {shlex.quote(json.dumps(paths))} {trigger}", agent, {})
        await _until(lambda: "before " in read_tail(command.log))
        for path, text in ((token, "NEW-SECRET-HF"), (hf / "stored_tokens", "NEW-SECRET-STORED")):
            fresh = path.with_name(path.name + ".lock")
            fresh.write_text(text)
            os.replace(fresh, path)
        trigger.write_text("")
        await _until(lambda: command.returncode is not None)
    finally:
        await sandbox.stop()

    assert command.returncode == 0, read_tail(command.log)
    seen = {
        line.split(" ", 1)[0]: json.loads(line.split(" ", 1)[1])
        for line in read_tail(command.log).splitlines()
        if line.startswith(("before ", "after "))
    }
    expected = {"token": "ENOENT", "stored": "ENOENT", "model": "ok:weights", "note": "ok:beside"}
    assert seen == {"before": expected, "after": expected}
    assert token.read_text() == "NEW-SECRET-HF"
