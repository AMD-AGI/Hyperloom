# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real bubblewrap sandboxes launched through the adapter: what an agent inside can and cannot reach, how it ends, and
the holder whose namespaces a sandbox's commands share.

Skipped where there is no bubblewrap or no user namespace for it; failed instead where
``HYPERLOOM_REQUIRE_ISOLATION_TESTS=1`` (CI).
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom_common import bwrap
from hyperloom_common.bwrap import (
    SYSTEM_ROOTS,
    BwrapPolicy,
    BwrapSetupError,
    BwrapStatus,
    Holder,
    Scratch,
    interpreter_roots,
)
from hyperloom_common.tests._bwrap_host import usable_bwrap

pytestmark = pytest.mark.agent_isolation

#: What a masked credential holds; it must never reach the sandbox's output.
CANARY = "canary-canary-canary"
TIMEOUT = 60
#: Imported by every check below: ``outcome(action)`` is ``"ok"`` or the errno's name.
PRELUDE = """
import errno, json, os, sys
def outcome(action):
    try:
        action()
    except OSError as exc:
        return errno.errorcode.get(exc.errno, str(exc.errno))
    return "ok"
"""


@pytest.fixture(scope="module")
def launcher() -> tuple[str, ...]:
    return usable_bwrap().launcher


@pytest.fixture
def scratch() -> Iterator[Scratch]:
    made = Scratch.create()
    yield made
    made.remove()


def policy(*read_only: str, **changes: object) -> BwrapPolicy:
    """An agent-shaped policy that can run ``sys.executable -I -S``, plus ``read_only`` and ``changes``."""
    return BwrapPolicy(read_only=(*SYSTEM_ROOTS, *interpreter_roots(), *read_only), **changes)  # type: ignore[arg-type]


def python(code: str, *args: object) -> list[str]:
    return [sys.executable, "-I", "-S", "-c", PRELUDE + code, *map(str, args)]


def run(
    launcher: Sequence[str], sandbox: BwrapPolicy, command: Sequence[str], **popen: object
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        sandbox.argv(launcher, command),
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        check=False,
        stdin=subprocess.DEVNULL,
        **popen,  # type: ignore[arg-type]
    )


def report(done: subprocess.CompletedProcess[str]) -> dict[str, object]:
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def marked(name: str, value: str) -> list[int]:
    """The processes whose environment carries ``name=value``, as the app's reaper finds them."""
    needle = f"{name}={value}".encode()
    found = []
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                with open(f"/proc/{entry}/environ", "rb") as handle:
                    if needle in handle.read().split(b"\0"):
                        found.append(int(entry))
            except OSError:
                continue
    return found


def wait_for(predicate: Callable[[], object], timeout: float = 10.0) -> object:
    """``predicate()`` once it is true, polled every 50 ms; its last value at ``timeout``."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    return predicate()


# ---- what the agent reaches --------------------------------------------------------------------------------------


def test_masks_hide_credentials_inside_a_bound_tree(launcher: tuple[str, ...], tmp_path: Path) -> None:
    hf = tmp_path / "hf"
    (hf / "stored_tokens").mkdir(parents=True)
    (hf / "stored_tokens" / "default").write_text(CANARY)
    (hf / "token").write_text(CANARY)
    (hf / "config.json").write_text("{}")
    check = """
hf = sys.argv[1]
print(json.dumps({
    "config": outcome(lambda: open(os.path.join(hf, "config.json")).read()),
    "token": outcome(lambda: print(open(os.path.join(hf, "token")).read())),
    "stored_list": outcome(lambda: print(os.listdir(os.path.join(hf, "stored_tokens")))),
    "stored_read": outcome(lambda: print(open(os.path.join(hf, "stored_tokens", "default")).read())),
    "stored_create": outcome(lambda: open(os.path.join(hf, "stored_tokens", "new"), "w").close()),
}))
"""
    masked = run(launcher, policy(str(hf), masked=(str(hf / "stored_tokens"), str(hf / "token"))), python(check, hf))
    unmasked = run(launcher, policy(str(hf)), python(check, hf))

    assert report(masked) == {
        "config": "ok",
        "token": "EACCES",
        "stored_list": "EACCES",
        "stored_read": "EACCES",
        "stored_create": "EACCES",
    }
    assert CANARY not in masked.stdout + masked.stderr
    assert report(unmasked)["token"] == "ok" and CANARY in unmasked.stdout


def test_read_only_binds_refuse_writes_and_metadata(launcher: tuple[str, ...], tmp_path: Path) -> None:
    tree, work = tmp_path / "tree", tmp_path / "wt"
    tree.mkdir()
    work.mkdir()
    (tree / "file").write_text("kept")
    (tree / "file").chmod(0o644)
    check = """
tree, work = sys.argv[1:]
target = os.path.join(tree, "file")
print(json.dumps({
    "create": outcome(lambda: open(os.path.join(tree, "new"), "w").close()),
    "write": outcome(lambda: open(target, "a").close()),
    "chmod": outcome(lambda: os.chmod(target, 0o777)),
    "truncate": outcome(lambda: os.truncate(target, 0)),
    "utime": outcome(lambda: os.utime(target, (0, 0))),
    "rename": outcome(lambda: os.rename(target, target + ".moved")),
    "worktree": outcome(lambda: open(os.path.join(work, "edit"), "w").write("landed")),
}))
"""
    result = report(run(launcher, policy(str(tree), writable=(str(work),)), python(check, tree, work)))

    assert result == {
        "create": "EROFS",
        "write": "EROFS",
        "chmod": "EROFS",
        "truncate": "EROFS",
        "utime": "EROFS",
        "rename": "EROFS",
        "worktree": "ok",
    }
    assert stat.S_IMODE((tree / "file").stat().st_mode) == 0o644 and (tree / "file").read_text() == "kept"
    assert (work / "edit").read_text() == "landed"


def test_the_app_is_out_of_reach(launcher: tuple[str, ...], tmp_path: Path) -> None:
    state = tmp_path / "app-state"
    state.mkdir()
    (state / "exit.json").write_text("{}")
    tree = tmp_path / "tree"
    (tree / "secret").mkdir(parents=True)
    host_tmp = tempfile.NamedTemporaryFile(prefix="hl-app-", dir=tempfile.gettempdir(), delete=False)
    host_tmp.close()
    # A process of the app's that started after the sandbox's pid 1 and 2 were taken in this pid namespace, so the
    # checks below cannot hit the sandbox's own pids (where the test runs as pid 1 of a container).
    app = subprocess.Popen(["sleep", "60"])
    check = """
import ctypes
app, state, host_tmp, tree, secret = int(sys.argv[1]), *sys.argv[2:]
libc = ctypes.CDLL(None, use_errno=True)
def umount(path):
    if libc.umount2(path.encode(), 0) != 0:
        raise OSError(ctypes.get_errno(), "umount2")
status = dict(line.split(":", 1) for line in open("/proc/self/status").read().splitlines())
print(json.dumps({
    "state": outcome(lambda: open(os.path.join(state, "exit.json")).close()),
    "host_tmp": outcome(lambda: open(host_tmp).close()),
    "pids": sorted(int(name) for name in os.listdir("/proc") if name.isdigit()),
    "app_environ": outcome(lambda: open(f"/proc/{app}/environ", "rb").close()),
    "app_signal": outcome(lambda: os.kill(app, 0)),
    "NoNewPrivs": status["NoNewPrivs"].strip(),
    "CapEff": status["CapEff"].strip(),
    "CapBnd": status["CapBnd"].strip(),
    "umount_mask": outcome(lambda: umount(secret)),
    "umount_proc": outcome(lambda: umount("/proc")),
    "umount_ro": outcome(lambda: umount(tree)),
}))
"""
    try:
        sandbox = policy(str(tree), masked=(str(tree / "secret"),), app_only=(str(state),))
        assert sandbox.problems(usable_bwrap().version or (0, 0, 0)) == []
        result = report(run(launcher, sandbox, python(check, app.pid, state, host_tmp.name, tree, tree / "secret")))
    finally:
        os.unlink(host_tmp.name)
        app.kill()
        app.wait()

    assert app.pid not in (1, 2)
    assert result == {
        "state": "ENOENT",
        "pids": [1, 2],
        "host_tmp": "ENOENT",
        "app_environ": "ENOENT",
        "app_signal": "ESRCH",
        "NoNewPrivs": "1",
        "CapEff": "0000000000000000",
        "CapBnd": "0000000000000000",
        "umount_mask": "EPERM",
        "umount_proc": "EPERM",
        "umount_ro": "EPERM",
    }


def test_loopback_is_shared_unless_the_sandbox_has_its_own_network(launcher: tuple[str, ...]) -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    check = """
import socket, subprocess
port = int(sys.argv[1])
print(json.dumps({
    "loopback": outcome(lambda: socket.create_connection(("127.0.0.1", port), 5).close()),
    "devnull": outcome(lambda: open("/dev/null", "w").write("x")),
    "devnull_child": subprocess.run([sys.executable, "-I", "-S", "-c", "pass"], stdout=subprocess.DEVNULL).returncode,
    "cpus": outcome(lambda: os.listdir("/sys/devices/system/cpu")),
    "sys_class": outcome(lambda: os.listdir("/sys/class")),
}))
"""
    try:
        agent = report(run(launcher, policy(), python(check, port)))
        compile_shape = report(
            run(launcher, policy(own_network=True, sys_paths=("/sys/devices/system/cpu",)), python(check, port))
        )
    finally:
        listener.close()

    assert agent == {"loopback": "ok", "devnull": "ok", "devnull_child": 0, "cpus": "ok", "sys_class": "ok"}
    assert compile_shape == {
        "loopback": "ECONNREFUSED",
        "devnull": "ok",
        "devnull_child": 0,
        "cpus": "ok",
        "sys_class": "ENOENT",
    }


def test_the_command_gets_its_environment_its_marks_and_its_descriptors(launcher: tuple[str, ...]) -> None:
    read_end, write_end = os.pipe()
    check = """
os.write(int(sys.argv[1]), b"through the pipe")
print(json.dumps(dict(os.environ)))
"""
    try:
        done = run(
            launcher,
            policy(marks=(("HYPERLOOM_TEST_MARK", "m1"),)),
            python(check, write_end),
            env={"PATH": "/usr/bin:/bin", "GIVEN": "1"},
            pass_fds=(write_end,),
        )
        os.close(write_end)
        piped = os.read(read_end, 100)
    finally:
        os.close(read_end)

    inside = report(done)
    # bubblewrap sets PWD; Python's C-locale coercion (PEP 538) sets LC_CTYPE in its own environment.
    pwd, ctype = inside.pop("PWD"), inside.pop("LC_CTYPE", "C.UTF-8")
    assert (pwd, ctype) == ("/", "C.UTF-8")
    assert inside == {"PATH": "/usr/bin:/bin", "GIVEN": "1", "HYPERLOOM_TEST_MARK": "m1"}
    assert piped == b"through the pipe"


def test_the_status_tells_a_failed_setup_from_a_command_that_ran(launcher: tuple[str, ...], tmp_path: Path) -> None:
    def launched(sandbox: BwrapPolicy, command: list[str]) -> tuple[subprocess.CompletedProcess[str], BwrapStatus]:
        read_end, write_end = os.pipe()
        with os.fdopen(read_end, "rb") as status:
            try:
                done = subprocess.run(
                    sandbox.argv(launcher, command, status_fd=write_end),
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT,
                    check=False,
                    pass_fds=(write_end,),
                )
            finally:
                os.close(write_end)
            return done, BwrapStatus.parse(status.read())

    exited, exited_status = launched(policy(), python("sys.exit(7)"))
    killed, killed_status = launched(policy(), python("os.kill(os.getpid(), 9)"))
    unset, unset_status = launched(policy(writable=(str(tmp_path / "missing"),)), python("pass"))

    assert (exited.returncode, exited_status.exit_code) == (7, 7) and exited_status.child_pid
    assert (killed.returncode, killed_status.exit_code) == (137, 137)
    assert unset.returncode == 1 and unset_status.child_pid and unset_status.exit_code is None
    assert f"bwrap: Can't find source path {tmp_path / 'missing'}" in unset.stderr


# ---- how it ends -------------------------------------------------------------------------------------------------


def test_a_term_to_the_marked_processes_stops_the_agent_gracefully(launcher: tuple[str, ...]) -> None:
    token = f"graceful-{os.getpid()}-{time.monotonic_ns()}"
    agent = subprocess.Popen(
        policy(marks=(("HYPERLOOM_TEST_MARK", token),)).argv(
            launcher, ["sh", "-c", 'trap "echo TERM-HANDLED; exit 0" TERM; sleep 1000 & wait']
        ),
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        found = wait_for(
            lambda: len(marked("HYPERLOOM_TEST_MARK", token)) >= 2 and marked("HYPERLOOM_TEST_MARK", token)
        )
        names = {pid: Path(f"/proc/{pid}/comm").read_text().strip() for pid in found}
        assert agent.pid not in found and "bwrap" not in names.values(), names
        for pid in found:
            os.kill(pid, signal.SIGTERM)
        out, _ = agent.communicate(timeout=TIMEOUT)
    finally:
        if agent.poll() is None:
            agent.kill()
            agent.wait()

    assert agent.returncode == 0 and "TERM-HANDLED" in out
    assert wait_for(lambda: not marked("HYPERLOOM_TEST_MARK", token), 5)


PARENT = """
import json, subprocess, sys, time
child = subprocess.Popen(json.loads(sys.argv[1]), stdin=subprocess.DEVNULL, pass_fds=(int(sys.argv[2]),))
print(child.pid, flush=True)
time.sleep(1000)
"""


@pytest.mark.parametrize("die_with_parent", [True, False])
def test_the_sandbox_outlives_its_bwrap_process_unless_it_dies_with_its_parent(
    launcher: tuple[str, ...], die_with_parent: bool
) -> None:
    token = f"parent-{die_with_parent}-{os.getpid()}-{time.monotonic_ns()}"
    sandbox = policy(die_with_parent=die_with_parent, marks=(("HYPERLOOM_TEST_MARK", token),))
    read_end, write_end = os.pipe()
    argv = sandbox.argv(launcher, ["sleep", "1000"], status_fd=write_end)
    parent = subprocess.Popen(
        [sys.executable, "-c", PARENT, json.dumps(argv), str(write_end)],
        stdout=subprocess.PIPE,
        text=True,
        pass_fds=(write_end,),
    )
    os.close(write_end)
    sandbox_init = -1
    try:
        bwrap_pid = int(parent.stdout.readline()) if parent.stdout else 0
        assert wait_for(lambda: marked("HYPERLOOM_TEST_MARK", token))
        with os.fdopen(read_end, "rb") as status_file:
            read_end = -1
            sandbox_init = os.pidfd_open(BwrapStatus.parse(status_file.readline()).child_pid or 0)
        parent.kill()
        parent.wait()
        if not die_with_parent:
            os.kill(bwrap_pid, signal.SIGKILL)
        ended = wait_for(lambda: not marked("HYPERLOOM_TEST_MARK", token), 3)
    finally:
        if read_end >= 0:
            os.close(read_end)
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if sandbox_init >= 0:
            with contextlib.suppress(ProcessLookupError):
                signal.pidfd_send_signal(sandbox_init, signal.SIGKILL)
            os.close(sandbox_init)

    if die_with_parent:
        assert ended, "the sandbox survived the death of the thread that launched it"
    else:
        assert not ended, "killing bubblewrap's own process ended the sandbox"
    assert wait_for(lambda: not marked("HYPERLOOM_TEST_MARK", token), 5), "SIGKILL to its pid 1 did not end it"


# ---- one holder, many commands -----------------------------------------------------------------------------------


def test_joined_commands_share_pids_tmp_and_dev_shm_but_not_sysv_ipc(
    launcher: tuple[str, ...], scratch: Scratch
) -> None:
    sleep = shutil.which("sleep", path="/usr/bin:/bin")
    key = 0x484C0000 + os.getpid() % 0xFFFF
    sandbox = policy(tmp=scratch.tmp, shm=scratch.shm)
    first = f"""
import ctypes
libc = ctypes.CDLL(None, use_errno=True)
segment = libc.shmget({key}, 4096, 0o1600)
open("/tmp/server.log", "w").write("started")
open("/dev/shm/hl-posix", "w").write("shared")
print(json.dumps({{"sysv_own": libc.shmget({key}, 0, 0) == segment >= 0}}), flush=True)
os.execv({sleep!r}, ["sleep", "1000"])
"""
    second = f"""
import ctypes
libc = ctypes.CDLL(None, use_errno=True)
servers = [int(pid) for pid in os.listdir("/proc") if pid.isdigit() and open(f"/proc/{{pid}}/cmdline", "rb").read() == b"sleep\\x001000\\x00"]
for pid in servers:
    os.kill(pid, 15)
print(json.dumps({{
    "server_log": open("/tmp/server.log").read(),
    "posix_shm": open("/dev/shm/hl-posix").read(),
    "sysv_segment": libc.shmget({key}, 0, 0),
    "killed": len(servers),
}}))
"""
    holder = Holder.start(launcher, sandbox)
    try:
        with holder.joined(sandbox) as joined:
            one = subprocess.Popen(
                joined.argv(launcher, python(first)), stdout=subprocess.PIPE, text=True, pass_fds=joined.join
            )
        first_line = one.stdout.readline() if one.stdout else ""
        assert json.loads(first_line) == {"sysv_own": True}
        with holder.joined(sandbox) as joined:
            two = run(launcher, joined, python(second), pass_fds=joined.join)
        one_rc = one.wait(timeout=TIMEOUT)
        token = f"holder-{os.getpid()}-{time.monotonic_ns()}"
        with holder.joined(replace(sandbox, marks=(("HYPERLOOM_TEST_MARK", token),))) as joined:
            three = subprocess.Popen(joined.argv(launcher, ["sleep", "3000"]), pass_fds=joined.join)
        started = wait_for(lambda: marked("HYPERLOOM_TEST_MARK", token))
    finally:
        holder.stop()
        _remove_sysv_segment(key)
    three_rc = three.wait(timeout=TIMEOUT)

    assert report(two) == {"server_log": "started", "posix_shm": "shared", "sysv_segment": -1, "killed": 1}
    assert one_rc == 128 + signal.SIGTERM
    assert started and three_rc == 128 + signal.SIGKILL
    assert wait_for(lambda: not marked("HYPERLOOM_TEST_MARK", token), 5)
    with pytest.raises(ProcessLookupError), holder.joined(sandbox):
        pass
    holder.pid = os.getpid()  # the holder's pid, reused by a live process
    with pytest.raises(ProcessLookupError), holder.joined(sandbox):
        pass


def _remove_sysv_segment(key: int) -> None:
    """Remove the segment ``key`` where a command left it in this IPC namespace (only when IPC was not unshared)."""
    libc = ctypes.CDLL(None, use_errno=True)
    segment = libc.shmget(key, 0, 0)
    if segment >= 0:
        libc.shmctl(segment, 0, None)  # IPC_RMID


def test_a_holder_whose_setup_fails_says_why(launcher: tuple[str, ...], tmp_path: Path) -> None:
    with pytest.raises(BwrapSetupError, match="Can't find source path"):
        Holder.start(launcher, policy(writable=(str(tmp_path / "missing"),)))


# ---- hosts the probe refuses, and the unbindable-root launcher ---------------------------------------------------


def test_the_probe_refuses_where_a_user_cannot_make_a_user_namespace(tmp_path: Path) -> None:
    restricted = Path("/proc/sys/kernel/apparmor_restrict_unprivileged_userns")
    setpriv, python3 = shutil.which("setpriv"), shutil.which("python3", path="/usr/bin:/bin")
    if (
        os.geteuid() != 0
        or not setpriv
        or not python3
        or not restricted.exists()
        or restricted.read_text().strip() != "1"
    ):
        pytest.skip("needs root (to become another user) on a host that restricts unprivileged user namespaces")
    drop = [setpriv, "--reuid", "65534", "--regid", "65534", "--clear-groups", "--"]
    if subprocess.run([*drop, "true"], capture_output=True, check=False, timeout=TIMEOUT).returncode != 0:
        pytest.skip("uid 65534 is not mapped in this user namespace")
    shared = Path(tempfile.mkdtemp(prefix="hl-bwrap-nobody-"))
    try:
        shared.chmod(0o755)
        shutil.copy2(bwrap.__file__, shared / "bwrap.py")
        shutil.copy2(os.path.realpath(usable_bwrap().path or ""), shared / "bwrap")
        env = {"PATH": f"{shared}:/usr/bin:/bin", "TMPDIR": "/tmp"}
        done = subprocess.run(
            [*drop, python3, "-I", str(shared / "bwrap.py"), "probe"],
            capture_output=True,
            text=True,
            env=env,
            timeout=TIMEOUT,
            check=False,
        )
    finally:
        shutil.rmtree(shared)

    assert done.returncode == 1, done.stdout + done.stderr
    assert "uid map: Permission denied" in done.stdout


def test_a_launch_runs_under_unshare_mount(launcher: tuple[str, ...]) -> None:
    unshare = shutil.which("unshare")
    if os.geteuid() != 0 or unshare is None:
        pytest.skip("unshare --mount needs CAP_SYS_ADMIN, which only root has here")
    private = (unshare, "--mount", "--", launcher[-1])

    done = run(private, policy(), python('print(json.dumps({"pid": os.getpid()}))'))

    assert report(done) == {"pid": 2}
