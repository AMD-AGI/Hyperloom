# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Local test sandboxes confined by their provider's plan, under real bubblewrap: what a test's server-shaped process
reaches and what it never sees (a masked credential, another sandbox's key, the app's state, its processes), that its
server answers the app on loopback, that a kill reaps the whole tree after its grace and leaves the cards free, that an
unconfined sandbox's commands share one holder, and that a hosted run launches its layer's bubblewrap.

The trees live outside ``/tmp`` unless a test says otherwise: before bubblewrap 0.12 nothing may be bound below a
sandbox's own ``/tmp``."""

from __future__ import annotations

import json
import os
import select
import shlex
import shutil
import signal
import socket
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from hyperloom.common import agent_isolation
from hyperloom.common import proctree
from hyperloom.common.agent_isolation import require_agent_isolation
from hyperloom.sandbox import confined, local_sandbox
from hyperloom.sandbox.card_occupancy import SANDBOX_TAG_ENV
from hyperloom.sandbox.local import LocalSandboxProvider, LocalTemplate, ProcessUsers, RunnerPlan
from hyperloom.sandbox.reaper import tag_in_use
from hyperloom.sandbox.socket import RunSettings, SandboxUnavailable
from hyperloom_common.bwrap import BwrapProbe
from hyperloom_common.tests._bwrap_host import unavailable, usable_bwrap

pytestmark = pytest.mark.agent_isolation

SECRET = "app-secret-canary-value"
#: A test's server: listens on loopback, waits for the request the app writes to ``/in``, reports what it reaches to
#: ``/out/seen.json``, then answers on its port until it is stopped.
SERVER = r"""
import errno, json, os, socket, time
root = os.environ["HYPERLOOM_SANDBOX_ROOT"]
listener = socket.socket()
listener.bind(("127.0.0.1", 0))
listener.listen()
while True:
    try:
        args = json.load(open(os.path.join(root, "in", "request.json")))
        break
    except (OSError, ValueError):
        time.sleep(0.05)
def outcome(action):
    try:
        return "ok:" + (action() or "")
    except OSError as exc:
        return errno.errorcode.get(exc.errno, str(exc.errno))
def write(path):
    with open(path, "w") as handle:
        handle.write("from inside")
seen = {name: outcome(lambda path=path: open(path).read()[:16]) for name, path in args["reads"].items()}
seen.update({name: outcome(lambda path=path: write(path)) for name, path in args["writes"].items()})
seen["sandboxes"] = outcome(lambda: ",".join(os.listdir(args["sandboxes"])))
seen["app_pid"] = outcome(lambda: os.kill(args["app_pid"], 0))
pids = [int(entry) for entry in os.listdir("/proc") if entry.isdigit()]
seen["app_pid_listed"] = args["app_pid"] in pids
seen["processes"] = len(pids)
leaked = []
for pid in pids:
    try:
        if args["secret"].encode() in open(f"/proc/{pid}/environ", "rb").read():
            leaked.append(pid)
    except OSError:
        pass
seen["leaked"] = leaked
seen["devices"] = outcome(lambda: ",".join(sorted(os.listdir("/dev/dri"))))
seen["env"] = dict(os.environ)
seen["port"] = listener.getsockname()[1]
with open(os.path.join(root, "out", "seen.tmp"), "w") as handle:
    json.dump(seen, handle)
os.rename(os.path.join(root, "out", "seen.tmp"), os.path.join(root, "out", "seen.json"))
while True:
    connection, _ = listener.accept()
    connection.sendall(b"reply")
    connection.close()
"""
#: A runner that leaves a child, a detached daemon and one that dropped its environment, and stops slowly on SIGTERM.
FAMILY = r"""
trap 'sleep 0.5; echo stopped > "$HYPERLOOM_SANDBOX_ROOT/out/stopped"; exit 0' TERM
(exec -a "$1-child" sleep 1000) &
(setsid bash -c "exec -a '$1-daemon' sleep 1000" &)
(env -i /bin/bash -c "exec -a '$1-stripped' sleep 1000" &)
echo ready > "$HYPERLOOM_SANDBOX_ROOT/out/ready"
wait
"""
SERVE = r"""
import os, socket
listener = socket.socket()
listener.bind(("127.0.0.1", 0))
listener.listen()
open("/tmp/server.port", "w").write(str(listener.getsockname()[1]))
open("/tmp/server.log", "w").write("started")
while True:
    connection, _ = listener.accept()
    connection.sendall(b"reply")
    connection.close()
"""
CLIENT = r"""
import socket
print(socket.create_connection(("127.0.0.1", int(open("/tmp/server.port").read())), 5).recv(16).decode())
"""


@pytest.fixture(scope="module")
def probe() -> BwrapProbe:
    return usable_bwrap()


@pytest.fixture(autouse=True)
def fresh_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_isolation, "_STATE", agent_isolation._Process())
    monkeypatch.setattr(confined, "_said", set())


@pytest.fixture
def area() -> Iterator[Path]:
    if not os.access("/var/tmp", os.W_OK):
        unavailable("no writable /var/tmp to hold the sandboxes' trees outside /tmp")
    made = Path(tempfile.mkdtemp(prefix="hl-test-sandbox-", dir="/var/tmp")).resolve()
    yield made
    shutil.rmtree(made, ignore_errors=True)


@pytest.fixture
def providers() -> Iterator[list[LocalSandboxProvider]]:
    made: list[LocalSandboxProvider] = []
    yield made
    for provider in made:
        provider.close()


def _wait(predicate: Callable[[], object], seconds: float = 30.0) -> None:
    until = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < until, "timed out waiting"
        time.sleep(0.05)


def _read(sandbox: object, path: str) -> bytes:
    found: list[bytes] = []

    def present() -> bool:
        try:
            found.append(sandbox.read_file(path))  # type: ignore[attr-defined]
        except FileNotFoundError:
            return False
        return True

    _wait(present)
    return found[0]


def _named(name: str) -> list[int]:
    """The running processes of this host whose argv[0] is ``name``."""
    found = []
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                if Path(f"/proc/{entry}/cmdline").read_bytes().split(b"\0")[0] == name.encode():
                    found.append(int(entry))
            except OSError:
                continue
    return [pid for pid in found if proctree.running(pid)]


def _with_argument(argument: str) -> list[int]:
    """The running processes of this host one of whose arguments is ``argument``."""
    found = []
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                if argument.encode() in Path(f"/proc/{entry}/cmdline").read_bytes().split(b"\0"):
                    found.append(int(entry))
            except OSError:
                continue
    return [pid for pid in found if proctree.running(pid)]


def _bash(command: str) -> str:
    """``command`` run by bash (the sandbox's shell is ``sh``), which names a process with ``exec -a``."""
    return f"exec /bin/bash -c {shlex.quote(command)}"


def _environ(pid: int) -> bytes:
    try:
        return Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return b""


@pytest.mark.parametrize("below_tmp", [False, True], ids=["sandboxes-off-tmp", "sandboxes-below-tmp"])
def test_a_test_s_server_reaches_its_own_tree_and_key_and_nothing_of_the_app_or_another_sandbox(
    area: Path,
    probe: BwrapProbe,
    providers: list[LocalSandboxProvider],
    monkeypatch: pytest.MonkeyPatch,
    below_tmp: bool,
) -> None:
    hf, model, state, app_home = area / "hf", area / "model", area / "state", area / "app-home"
    for path, text in (
        (hf / "token", "SECRET-HF-TOKEN"),
        (hf / "hub" / "model.txt", "hub-weights"),
        (model / "weights", "weights"),
        (state / "exit.json", "{}"),
        (app_home / ".claude" / ".credentials.json", "SECRET-LOGIN"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    server = area / "server.py"
    server.write_text(SERVER)
    monkeypatch.setenv("HF_HOME", str(hf))
    monkeypatch.setenv("HOME", str(app_home))
    monkeypatch.setenv("HYPERLOOM_STATE_DIR", str(state))
    monkeypatch.setenv("HYPERLOOM_TEST_APP_VALUE", SECRET)
    base = Path(tempfile.mkdtemp(prefix="hl-test-sandboxes-", dir="/tmp")) if below_tmp else area / "sandboxes"
    template = LocalTemplate(
        "runner",
        "measurement",
        command=(sys.executable, "-I", "-S", str(server)),
        gpus=1,
        sealed=True,
        signing=True,
    )
    plan = RunnerPlan(readable=(hf, model, server))
    try:
        provider = LocalSandboxProvider([template], gpus=(0, 1), base_dir=base, users=ProcessUsers(plan=plan))
        providers.append(provider)
        other = provider.create("runner", role="measurement", timeout_sec=120, metadata={})
        mine = provider.create("runner", role="measurement", timeout_sec=120, metadata={})
        other.write_file("/in/other-out", b"other's output")
        other_key = base / other.sandbox_id / "private" / "signing_key.pem"
        mine_root = base / mine.sandbox_id / "fs"
        request = {
            "reads": {
                "own_key": str(base / mine.sandbox_id / "private" / "signing_key.pem"),
                "other_key": str(other_key),
                "other_fs": str(base / other.sandbox_id / "fs" / "in" / "other-out"),
                "state": str(state / "exit.json"),
                "app_login": str(app_home / ".claude" / ".credentials.json"),
                "hf_token": str(hf / "token"),
                "hub_model": str(hf / "hub" / "model.txt"),
                "model": str(model / "weights"),
            },
            "writes": {
                "out": str(mine_root / "out" / "written"),
                "work": str(mine_root / "work" / "written"),
                "tmp": "/tmp/written",
                "shm": "/dev/shm/written",
                "model_write": str(model / "written"),
                "private_write": str(base / mine.sandbox_id / "private" / "written"),
            },
            "sandboxes": str(base),
            "app_pid": os.getpid(),
            "secret": SECRET,
        }
        mine.write_file("/in/request.json", json.dumps(request).encode())
        seen = json.loads(_read(mine, "/out/seen.json"))
        with socket.create_connection(("127.0.0.1", seen["port"]), 5) as connection:
            answer = connection.recv(16)
    finally:
        for made in providers:
            made.close()
        if below_tmp:
            shutil.rmtree(base, ignore_errors=True)

    env = seen.pop("env")
    processes = seen.pop("processes")
    devices = seen.pop("devices")
    seen.pop("port")
    assert mine._confined is not None and other._confined is not None
    assert seen == {
        "own_key": "ok:-----BEGIN PRIVA",
        "other_key": "ENOENT",
        "other_fs": "ENOENT",
        "state": "ENOENT",
        "app_login": "ENOENT",
        "hf_token": "ENOENT",  # a login file in a bound tree is left out of it, not masked
        "hub_model": "ok:hub-weights",
        "model": "ok:weights",
        "out": "ok:",
        "work": "ok:",
        "tmp": "ok:",
        "shm": "ok:",
        "model_write": "EROFS",
        "private_write": "ok:",
        "sandboxes": f"ok:{mine.sandbox_id}",
        "app_pid": "ESRCH",
        "app_pid_listed": False,
        "leaked": [],
    }
    assert processes < 10, "the server sees its own sandbox's processes only"
    host_devices = ",".join(sorted(os.listdir("/dev/dri"))) if os.path.isdir("/dev/dri") else None
    assert devices == (f"ok:{host_devices}" if host_devices is not None else "ENOENT")
    assert answer == b"reply", "the app's client reaches the server on loopback"
    assert not (model / "written").exists()
    assert env["HOME"] == str(mine_root / "work") and env[SANDBOX_TAG_ENV] == mine.sandbox_id
    assert SECRET not in json.dumps(env) and "HF_HOME" not in env


@pytest.mark.parametrize("sealed", [True, False], ids=["sealed-runner", "commands-in-a-holder"])
def test_a_kill_reaps_the_whole_tree_after_its_grace_and_frees_the_cards(
    area: Path, probe: BwrapProbe, providers: list[LocalSandboxProvider], monkeypatch: pytest.MonkeyPatch, sealed: bool
) -> None:
    """Sealed, the runner has a pid namespace of its own; else its commands share a holder's, which the kill ends only
    after the reap, so that a command's TERM handler still runs to its end."""
    name = f"hl-test-{os.getpid()}-{time.monotonic_ns()}"
    script = area / "family.sh"
    script.write_text(FAMILY)
    template = (
        LocalTemplate(
            "runner", "measurement", command=("/bin/bash", str(script), name), gpus=1, sealed=True, signing=True
        )
        if sealed
        else LocalTemplate("dev", "dev", gpus=1)
    )
    provider = LocalSandboxProvider(
        [template], gpus=(0,), base_dir=area / "sandboxes", users=ProcessUsers(plan=RunnerPlan(readable=(script,)))
    )
    providers.append(provider)
    kept = area / "kept-out"
    real_rmtree = shutil.rmtree

    def keep_out(path: object, *args: object, **kwargs: object) -> None:
        out = Path(str(path)) / "fs" / "out"
        if out.is_dir() and not kept.exists():
            shutil.copytree(out, kept)
        real_rmtree(path, *args, **kwargs)  # type: ignore[arg-type]

    sandbox = provider.create(template.name, role=template.role, timeout_sec=120, metadata={})
    if not sealed:
        sandbox.run(f"exec /bin/bash {shlex.quote(str(script))} {name}", background=True)
    _read(sandbox, "/out/ready")
    family = [f"{name}-{part}" for part in ("child", "daemon", "stripped")]
    _wait(lambda: all(_named(member) for member in family))
    bwraps = [popen.pid for popen in sandbox._procs]
    child = _named(family[0])[0]
    before = (
        tag_in_use(sandbox.sandbox_id),
        f"{SANDBOX_TAG_ENV}={sandbox.sandbox_id}".encode() in _environ(child).split(b"\0"),
        [pid for pid in bwraps if f"{SANDBOX_TAG_ENV}=".encode() in _environ(pid)],
        SANDBOX_TAG_ENV.encode() in _environ(_named(family[2])[0]),
    )
    scratch = sandbox._confined.scratch if sandbox._confined else None
    holder = sandbox._confined.holder if sandbox._confined else None
    monkeypatch.setattr(local_sandbox.shutil, "rmtree", keep_out)
    sandbox.kill()
    monkeypatch.setattr(local_sandbox.shutil, "rmtree", real_rmtree)

    assert before == (True, True, [], False), "the tag is set inside only, and the stripped daemon dropped it"
    assert [member for member in family if _named(member)] == []
    assert not [pid for pid in bwraps if proctree.running(pid)], "bubblewrap's own processes ended"
    assert not tag_in_use(sandbox.sandbox_id)
    assert (kept / "stopped").read_text().strip() == "stopped", "the runner's TERM handler ran to its end"
    assert provider.pool.free_cards() == (0,) and not provider.quarantined
    assert scratch is not None and not os.path.exists(scratch.tmp) and not os.path.exists(scratch.shm)
    assert (holder is None) == sealed and (holder is None or not os.path.exists(f"/proc/{holder.pid}/ns"))


def test_an_unsealed_sandbox_s_commands_share_one_holder_and_its_kill_ends_it(
    area: Path, probe: BwrapProbe, providers: list[LocalSandboxProvider]
) -> None:
    name = f"hl-test-{os.getpid()}-{time.monotonic_ns()}"
    serve = area / "serve.py"
    serve.write_text(SERVE)
    client = area / "client.py"
    client.write_text(CLIENT)
    python = shlex.quote(sys.executable)
    provider = LocalSandboxProvider(
        [LocalTemplate("dev", "dev")],
        base_dir=area / "sandboxes",
        users=ProcessUsers(plan=RunnerPlan(readable=(serve, client))),
    )
    providers.append(provider)
    sandbox = provider.create("dev", role="dev", timeout_sec=120, metadata={})
    # Named by an argument: a Python before 3.11 renamed with ``exec -a`` cannot find its own prefix.
    sandbox.run(f"exec {python} -I -S {serve} {name}-server", background=True)
    _wait(lambda: _with_argument(f"{name}-server"))
    # The server's command line alone: not the launcher's, nor the shell's of the command that names it.
    server = shlex.quote(f"^[^ ]+ -I -S [^ ]+ {name}-serve[r]$")
    later = sandbox.run(
        f"cat /tmp/server.log; echo; {python} -I -S {client}; pgrep -f {server} >/dev/null && echo seen; "
        f"ls {sandbox.private} >/dev/null 2>&1 || echo no-private",
        settings=RunSettings(timeout_sec=60),
    )
    with pytest.raises(SandboxUnavailable, match="timed out"):
        sandbox.run(_bash(f"exec -a {name}-slow sleep 1000"), settings=RunSettings(timeout_sec=1))
    after_timeout = (_named(f"{name}-slow"), bool(_with_argument(f"{name}-server")))
    # The pattern does not match the command's own shell and launcher, whose command lines carry it.
    stopped = sandbox.run(f"pkill -f {server}; echo pkill=$?", settings=RunSettings(timeout_sec=60))
    _wait(lambda: not _with_argument(f"{name}-server"))
    sandbox.run(_bash(f"exec -a {name}-left sleep 1000"), background=True)
    _wait(lambda: _named(f"{name}-left"))
    holder = sandbox._confined.holder if sandbox._confined else None
    assert holder is not None
    sandbox.kill()

    assert later.exit_code == 0 and later.stdout.split() == ["started", "reply", "seen", "no-private"], later
    assert after_timeout == ([], True), "a command's timeout killed its tree alone"
    assert stopped.stdout.split() == ["pkill=0"], "a later command stops what an earlier one started"
    assert not _named(f"{name}-left") and not os.path.exists(f"/proc/{holder.pid}/ns"), "the kill ended the holder"


def test_a_command_after_its_holder_ended_gets_a_new_holder(
    area: Path, probe: BwrapProbe, providers: list[LocalSandboxProvider]
) -> None:
    provider = LocalSandboxProvider(
        [LocalTemplate("dev", "dev")], base_dir=area / "sandboxes", users=ProcessUsers(plan=RunnerPlan())
    )
    providers.append(provider)
    sandbox = provider.create("dev", role="dev", timeout_sec=120, metadata={})
    first = sandbox.run("echo first", settings=RunSettings(timeout_sec=60))
    ended = sandbox._confined.holder if sandbox._confined else None
    assert ended is not None
    signal.pidfd_send_signal(ended.pidfd, signal.SIGKILL)  # the holder's pid 1, which this test's sandbox started
    _wait(lambda: select.select([ended.pidfd], [], [], 0)[0])
    later = sandbox.run("echo again", settings=RunSettings(timeout_sec=60))
    holder = sandbox._confined.holder if sandbox._confined else None
    sandbox.kill()

    assert (first.exit_code, first.stdout.strip(), later.exit_code, later.stdout.strip()) == (0, "first", 0, "again")
    assert holder is not None and holder is not ended and holder.pid != ended.pid


def test_a_hosted_run_confines_its_tests_with_its_layer_bubblewrap(
    area: Path, probe: BwrapProbe, providers: list[LocalSandboxProvider], monkeypatch: pytest.MonkeyPatch
) -> None:
    layer = area / "layer" / "bin"
    layer.mkdir(parents=True)
    assert probe.path is not None
    shutil.copy2(probe.path, layer / "bwrap")
    try:
        require_agent_isolation(bwrap=str(layer / "bwrap"))
    except agent_isolation.IsolationRefused as refused:
        pytest.fail(f"the layer's bubblewrap cannot sandbox an agent here: {refused}")
    template = LocalTemplate("runner", "measurement", command=("/bin/sh", "-c", "sleep 600"), sealed=True, signing=True)
    provider = LocalSandboxProvider([template], base_dir=area / "sandboxes", users=ProcessUsers(plan=RunnerPlan()))
    providers.append(provider)
    sandbox = provider.create("runner", role="measurement", timeout_sec=60, metadata={})
    launched = [Path(f"/proc/{popen.pid}/exe").resolve() for popen in sandbox._procs]

    assert sandbox._confined is not None and launched == [(layer / "bwrap").resolve()], launched
