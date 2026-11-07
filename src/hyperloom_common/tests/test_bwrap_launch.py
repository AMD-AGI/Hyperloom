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
    NESTED_SAFE,
    SYSTEM_ROOTS,
    BwrapPolicy,
    BwrapPolicyError,
    BwrapProbe,
    BwrapSetupError,
    BwrapStatus,
    Holder,
    Scratch,
    interpreter_roots,
)
from hyperloom_common.tests._bwrap_host import as_owner, usable_bwrap

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
def probe() -> BwrapProbe:
    return usable_bwrap()


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
    probe: BwrapProbe, sandbox: BwrapPolicy, command: Sequence[str], **popen: object
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        sandbox.argv(probe, command),
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


def test_masks_hide_credentials_inside_a_bound_tree(probe: BwrapProbe, tmp_path: Path) -> None:
    hf = tmp_path / "hf"
    (hf / "stored_tokens").mkdir(parents=True)
    (hf / "stored_tokens" / "default").write_text(CANARY)
    (hf / "config.json").write_text("{}")
    check = """
hf = sys.argv[1]
print(json.dumps({
    "config": outcome(lambda: open(os.path.join(hf, "config.json")).read()),
    "stored_list": outcome(lambda: print(os.listdir(os.path.join(hf, "stored_tokens")))),
    "stored_read": outcome(lambda: print(open(os.path.join(hf, "stored_tokens", "default")).read())),
    "stored_create": outcome(lambda: open(os.path.join(hf, "stored_tokens", "new"), "w").close()),
}))
"""
    masked = run(probe, policy(str(hf), masked=(str(hf / "stored_tokens"),)), python(check, hf))
    unmasked = run(probe, policy(str(hf)), python(check, hf))

    assert report(masked) == {
        "config": "ok",
        "stored_list": "EACCES",
        "stored_read": "EACCES",
        "stored_create": "EACCES",
    }
    assert CANARY not in masked.stdout + masked.stderr
    assert report(unmasked)["stored_read"] == "ok" and CANARY in unmasked.stdout


#: Reads the two credentials, waits for a line on stdin (the app changes them meanwhile), then reads them again.
READ_TWICE = """
def read(path):
    try:
        print(open(path).read())
    except OSError as exc:
        return errno.errorcode[exc.errno]
    return "read"
credentials, token = sys.argv[1:]
print(json.dumps([read(credentials), read(token)]), flush=True)
sys.stdin.readline()
print(json.dumps([read(credentials), read(token)]), flush=True)
"""


def test_a_masked_directory_holds_while_the_app_replaces_and_creates_credentials_in_it(
    probe: BwrapProbe, tmp_path: Path
) -> None:
    # The app refreshes a login by writing a new file and renaming it over the old one (git's lockfile, an atomic
    # save), or logs in during the task (HF's token): a mount over each file would be gone with the rename, and an
    # absent one would have none. A mask over the directory holding them stays.
    home = tmp_path / "home"
    tool = home / ".config" / "tool"
    tool.mkdir(parents=True)
    credentials, token = tool / "credentials.json", tool / "token"
    credentials.write_text("before-" + CANARY)
    files = policy(str(home), masked=(str(credentials), str(token)))
    assert files.problems(probe.version or (0, 0, 0)) == [
        f"{path} is masked below a bound tree but is not a directory: a mount over a file goes when the app replaces "
        "the file, and none covers one made later; mask the directory that holds it, or bind only what is needed "
        "beside it"
        for path in (credentials, token)
    ]
    sandbox = policy(str(home), masked=(str(tool),))

    with subprocess.Popen(
        sandbox.argv(probe, python(READ_TWICE, credentials, token)),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as agent:
        before = agent.stdout.readline() if agent.stdout else ""
        fresh = tool / "credentials.json.lock"
        fresh.write_text("after-" + CANARY)
        os.replace(fresh, credentials)
        token.write_text("login-" + CANARY)
        out, err = agent.communicate("go\n", timeout=TIMEOUT)

    assert agent.returncode == 0, err
    assert json.loads(before) == ["EACCES", "EACCES"]
    assert json.loads(out.strip().splitlines()[-1]) == ["EACCES", "EACCES"]
    assert CANARY not in before + out + err
    assert credentials.read_text() == "after-" + CANARY and token.read_text() == "login-" + CANARY


def test_read_only_binds_refuse_writes_and_metadata(probe: BwrapProbe, tmp_path: Path) -> None:
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
    result = report(run(probe, policy(str(tree), writable=(str(work),)), python(check, tree, work)))

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


def test_the_app_is_out_of_reach(probe: BwrapProbe, tmp_path: Path) -> None:
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
        result = report(run(probe, sandbox, python(check, app.pid, state, host_tmp.name, tree, tree / "secret")))
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


def test_loopback_is_shared_unless_the_sandbox_has_its_own_network(probe: BwrapProbe) -> None:
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
        agent = report(run(probe, policy(), python(check, port)))
        compile_shape = report(
            run(probe, policy(own_network=True, sys_paths=("/sys/devices/system/cpu",)), python(check, port))
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


def test_the_command_gets_its_environment_its_marks_and_its_descriptors(probe: BwrapProbe) -> None:
    read_end, write_end = os.pipe()
    check = """
os.write(int(sys.argv[1]), b"through the pipe")
print(json.dumps(dict(os.environ)))
"""
    try:
        done = run(
            probe,
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


def test_the_status_tells_a_failed_setup_from_a_command_that_ran(probe: BwrapProbe, tmp_path: Path) -> None:
    def launched(sandbox: BwrapPolicy, command: list[str]) -> tuple[subprocess.CompletedProcess[str], BwrapStatus]:
        read_end, write_end = os.pipe()
        with os.fdopen(read_end, "rb") as status:
            try:
                done = subprocess.run(
                    sandbox.argv(probe, command, status_fd=write_end),
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


def test_a_term_to_the_marked_processes_stops_the_agent_gracefully(probe: BwrapProbe) -> None:
    token = f"graceful-{os.getpid()}-{time.monotonic_ns()}"
    agent = subprocess.Popen(
        policy(marks=(("HYPERLOOM_TEST_MARK", token),)).argv(
            probe, ["sh", "-c", 'trap "echo TERM-HANDLED; exit 0" TERM; sleep 1000 & wait']
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
    probe: BwrapProbe, die_with_parent: bool
) -> None:
    token = f"parent-{die_with_parent}-{os.getpid()}-{time.monotonic_ns()}"
    sandbox = policy(die_with_parent=die_with_parent, marks=(("HYPERLOOM_TEST_MARK", token),))
    read_end, write_end = os.pipe()
    argv = sandbox.argv(probe, ["sleep", "1000"], status_fd=write_end)
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


def test_scratch_the_agent_locked_is_removed_from_the_host() -> None:
    # The agent maps to the owning uid, so it can chmod 000 its /tmp and /dev/shm and a directory in them.
    made = Scratch.create()
    lock = "mkdir -p /tmp/d/inner /dev/shm/x && echo y > /tmp/d/inner/f && chmod 000 /tmp/d/inner /tmp/d /tmp /dev/shm"
    done = run(usable_bwrap(), BwrapPolicy(read_only=SYSTEM_ROOTS, tmp=made.tmp, shm=made.shm), ["/bin/sh", "-c", lock])

    removed = as_owner(
        "import sys; from hyperloom_common.bwrap import Scratch; Scratch(*sys.argv[1:]).remove()", made.tmp, made.shm
    )

    left = [path for path in (made.tmp, made.shm) if os.path.lexists(path)]
    for path in left:
        subprocess.run(["chmod", "-R", "u+rwx", path], check=False)
        shutil.rmtree(path)
    assert done.returncode == 0, done.stderr
    assert removed.returncode == 0 and left == [], removed.stderr


def test_joined_commands_share_pids_tmp_and_dev_shm_but_not_sysv_ipc(probe: BwrapProbe, scratch: Scratch) -> None:
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
    holder = Holder.start(probe, sandbox)
    try:
        with holder.joined(sandbox) as joined:
            one = subprocess.Popen(
                joined.argv(probe, python(first)), stdout=subprocess.PIPE, text=True, pass_fds=joined.join
            )
        first_line = one.stdout.readline() if one.stdout else ""
        assert json.loads(first_line) == {"sysv_own": True}
        with holder.joined(sandbox) as joined:
            two = run(probe, joined, python(second), pass_fds=joined.join)
        one_rc = one.wait(timeout=TIMEOUT)
        token = f"holder-{os.getpid()}-{time.monotonic_ns()}"
        with holder.joined(replace(sandbox, marks=(("HYPERLOOM_TEST_MARK", token),))) as joined:
            three = subprocess.Popen(joined.argv(probe, ["sleep", "3000"]), pass_fds=joined.join)
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


def test_a_holder_whose_pid_1_ended_is_not_joined_before_bubblewrap_reaps_it(probe: BwrapProbe) -> None:
    sandbox = policy()
    holder = Holder.start(probe, sandbox)
    try:
        # bubblewrap, stopped, cannot reap the pid 1 killed here: that number still opens the dead holder's
        # namespaces, so only the pidfd tells that it has ended.
        os.kill(holder.process.pid, signal.SIGSTOP)
        os.kill(holder.pid, signal.SIGKILL)
        assert wait_for(lambda: _state(holder.pid) == "Z")
        os.close(os.open(f"/proc/{holder.pid}/ns/user", os.O_RDONLY))
        with pytest.raises(ProcessLookupError), holder.joined(sandbox):
            pass
    finally:
        os.kill(holder.process.pid, signal.SIGCONT)
        holder.stop()


def _state(pid: int) -> str:
    """The state letter of process ``pid`` (``Z`` for a zombie); empty when there is none."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return ""


def test_a_symlink_a_launch_left_in_a_shared_scratch_does_not_carry_a_later_mount_out(
    probe: BwrapProbe, scratch: Scratch
) -> None:
    # Two launches given the same Scratch, as a sandbox's commands are. The first leaves a link at /tmp/<carried> to a
    # host directory, through bubblewrap's view of the old root; the second binds a file below /tmp. Before 0.12
    # bubblewrap follows that link while it makes the mount point and creates the file on the host, so the adapter
    # refuses the launch there; from 0.12 bubblewrap refuses to follow it.
    host = Path(tempfile.mkdtemp(prefix="hl-host-", dir="/tmp"))
    carried = Path(tempfile.mkdtemp(prefix="hl-carried-", dir="/tmp"))
    (carried / "ca.pem").write_text("CA\n")
    sandbox = policy(tmp=scratch.tmp, shm=scratch.shm)
    plant = f"os.symlink({'/oldroot' + str(host)!r}, {str(carried)!r}); print(json.dumps(os.listdir('/tmp')))"
    later = replace(sandbox, read_only=(*sandbox.read_only, str(carried / "ca.pem")))
    refused: tuple[str, ...] = ()
    done = None
    try:
        planted = report(run(probe, sandbox, python(plant)))
        try:
            done = run(probe, later, ["true"])
        except BwrapPolicyError as refusal:
            refused = refusal.problems
        landed = os.listdir(host)
    finally:
        shutil.rmtree(host)
        shutil.rmtree(carried)

    assert planted == [carried.name]
    assert landed == [], "a mount below the shared /tmp created a file on the host"
    if (probe.version or (0, 0, 0)) < NESTED_SAFE:
        assert refused == tuple(later.problems(probe.version or (0, 0, 0))) != ()
        assert refused[0].startswith(f"{carried / 'ca.pem'} would be mounted below the agent-writable /tmp")
    else:
        assert done is not None and done.returncode == 1, done
        assert done.stderr.startswith("bwrap: "), done.stderr


def _remove_sysv_segment(key: int) -> None:
    """Remove the segment ``key`` where a command left it in this IPC namespace (only when IPC was not unshared)."""
    libc = ctypes.CDLL(None, use_errno=True)
    segment = libc.shmget(key, 0, 0)
    if segment >= 0:
        libc.shmctl(segment, 0, None)  # IPC_RMID


def test_a_joined_command_reads_nothing_of_the_apps_environment_from_the_holder(
    probe: BwrapProbe, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = f"holder-environ-{os.getpid()}-{time.monotonic_ns()}"
    monkeypatch.setenv("HYPERLOOM_TEST_APP_VALUE", token)
    sandbox = policy()
    read_all = """
seen = {}
for pid in sorted(int(entry) for entry in os.listdir("/proc") if entry.isdigit() and int(entry) != os.getpid()):
    try:
        seen[pid] = open(f"/proc/{pid}/environ", "rb").read().decode(errors="replace")
    except OSError as exc:
        seen[pid] = errno.errorcode.get(exc.errno, str(exc.errno))
print(json.dumps(seen))
"""
    holder = Holder.start(probe, sandbox)
    try:
        with holder.joined(sandbox) as joined:
            seen = report(run(probe, joined, python(read_all), pass_fds=joined.join, env={"PATH": "/usr/bin:/bin"}))
    finally:
        holder.stop()

    assert sorted(seen) == ["1", "2"], seen  # the holder's pid 1 and its process, both read
    assert all(token not in environ for environ in seen.values()), seen


def test_a_holder_whose_setup_fails_says_why(probe: BwrapProbe, tmp_path: Path) -> None:
    with pytest.raises(BwrapSetupError, match="Can't find source path"):
        Holder.start(probe, policy(writable=(str(tmp_path / "missing"),)))


# ---- hosts the probe refuses, and the unbindable-root prefix -----------------------------------------------------


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


def test_a_launch_runs_under_unshare_mount(probe: BwrapProbe) -> None:
    unshare = shutil.which("unshare")
    if os.geteuid() != 0 or unshare is None:
        pytest.skip("unshare --mount needs CAP_SYS_ADMIN, which only root has here")
    private = replace(probe, prefix=(unshare, "--mount", "--"))
    sandbox = policy()

    done = run(private, sandbox, python('print(json.dumps({"pid": os.getpid()}))'))
    # unshare execs bubblewrap, so the holder's pid 1 is still the child of the process it started.
    holder = Holder.start(private, sandbox)
    try:
        with holder.joined(sandbox) as joined:
            pids = 'print(json.dumps(sorted(int(name) for name in os.listdir("/proc") if name.isdigit())))'
            inside = run(private, joined, python(pids), pass_fds=joined.join)
    finally:
        holder.stop()

    assert report(done) == {"pid": 2}
    seen = report(inside)
    assert seen[:2] == [1, 2] and len(seen) > 2, seen  # the holder's pid 1 and its process, then this command's
