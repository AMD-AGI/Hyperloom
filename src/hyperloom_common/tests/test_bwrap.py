# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The bubblewrap adapter without namespaces: the argv a policy becomes, the policies it refuses, and the probe read
against a stand-in ``bwrap`` (real sandboxes: ``test_bwrap_launch.py``)."""

from __future__ import annotations

import json
import os
import signal
import stat
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from hyperloom_common import bwrap
from hyperloom_common.bwrap import (
    BASE_DEV_NODES,
    SYSTEM_ROOTS,
    BwrapPolicy,
    BwrapProbe,
    BwrapStatus,
    Scratch,
    bwrap_probe,
    interpreter_roots,
    main,
    probe_bwrap,
)

#: What the canary printed on this pod's bubblewrap 0.6.1 to 0.13.1 (``python3 -I bwrap.py probe``).
CANARY_OK = {
    "mask_dir": "EACCES",
    "mask_file": "EACCES",
    "ro_read": "ok",
    "ro_create": "EROFS",
    "ro_chmod": "EROFS",
    "rw_write": "ok",
    "umount_mask": "EPERM",
    "pid_namespace": "own",
    "NoNewPrivs": "1",
    "CapEff": "0000000000000000",
}
UID_MAP_DENIED = "bwrap: setting up uid map: Permission denied"
UNBINDABLE = (
    "bwrap: Can't bind mount /oldroot/usr on /newroot/usr: Unable to mount source on destination: Invalid argument"
)
DEV_NULL_MASK = ["--ro-bind", "/dev/null"]


@pytest.fixture(autouse=True)
def _fresh_probe_cache():
    bwrap._PROBES.clear()
    yield
    bwrap._PROBES.clear()


class FakeBwrap:
    """A ``bwrap`` stand-in in its own directory: answers ``--version``, and for a launch records its argv, writes
    ``stdout``/``stderr`` and exits ``code``. With ``unbindable`` it fails every launch that is not under the fake
    ``unshare --mount`` beside it."""

    def __init__(self, directory: Path, *, version: str = "0.13.1", unshare: bool = True) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "bwrap"
        self.version = version
        self.configure()
        if unshare:
            script = directory / "unshare"
            script.write_text(
                '#!/bin/sh\n[ "$1" = --mount ] || exit 9\nshift\n[ "$1" = -- ] && shift\nFAKE_PRIVATE=1 exec "$@"\n'
            )
            script.chmod(0o755)

    def configure(
        self,
        *,
        stdout: str = json.dumps(CANARY_OK),
        stderr: str = "",
        code: int = 0,
        sleep: float = 0,
        unbindable: str = "",
    ) -> None:
        (self.directory / "stdout").write_text(stdout + "\n")
        (self.directory / "stderr").write_text(stderr + "\n" if stderr else "")
        trap = f'if [ -z "$FAKE_PRIVATE" ]; then echo "{unbindable}" >&2; exit 1; fi\n' if unbindable else ""
        pause = f"/bin/sleep {sleep}\n" if sleep else ""
        self.path.write_text(
            "#!/bin/sh\n"
            f'if [ "$1" = --version ]; then echo "bubblewrap {self.version}"; exit 0; fi\n'
            f'/bin/echo "${{FAKE_PRIVATE:-plain}}" >> "{self.directory}/runs"\n'
            f'printf "%s\\n" "$@" > "{self.directory}/argv"\n'
            f"{pause}{trap}"
            f'/bin/cat "{self.directory}/stdout"\n/bin/cat "{self.directory}/stderr" >&2\nexit {code}\n'
        )
        self.path.chmod(0o755)

    @property
    def runs(self) -> list[str]:
        path = self.directory / "runs"
        return path.read_text().split() if path.exists() else []

    @property
    def argv(self) -> list[str]:
        return (self.directory / "argv").read_text().splitlines()


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeBwrap:
    stand_in = FakeBwrap(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(stand_in.directory))
    return stand_in


# ---- the argv ------------------------------------------------------------------------------------------------------


def test_argv_mounts_by_depth_with_the_masks_last(tmp_path: Path) -> None:
    models = tmp_path / "models"
    (models / "stored_tokens").mkdir(parents=True)
    (models / "token").write_text("hf_secret")
    work = tmp_path / "work"
    work.mkdir()
    policy = BwrapPolicy(
        read_only=("/usr", str(models)),
        writable=(str(work),),
        masked=(str(models / "stored_tokens"), str(models / "token"), str(models / "absent")),
        devices=("/dev/kfd",),
        tmp="/var/tmp/hl-agent-a",
        shm="/dev/shm/hl-agent-b",
        marks=(("HYPERLOOM_REAP_MARK", "m1"),),
        chdir=str(work),
    )

    argv = policy.argv(["/opt/bin/bwrap"], ["claude", "--print"], status_fd=9)

    nodes = [arg for node in (*BASE_DEV_NODES, "/dev/kfd") for arg in ("--dev-bind-try", node, node)]
    links = ["/proc/self/fd", "/dev/fd", "/proc/self/fd/0", "/dev/stdin", "/proc/self/fd/1", "/dev/stdout"]
    links += ["/proc/self/fd/2", "/dev/stderr"]
    stored = str(models / "stored_tokens")
    assert argv == [
        "/opt/bin/bwrap",
        *("--unshare-user", "--unshare-pid", "--unshare-ipc", "--new-session", "--cap-drop", "ALL"),
        "--die-with-parent",
        *("--ro-bind-try", "/usr", "/usr", "--tmpfs", "/dev", "--proc", "/proc", "--ro-bind-try", "/sys", "/sys"),
        *("--bind", "/var/tmp/hl-agent-a", "/tmp"),
        *nodes,
        *(arg for index in range(0, len(links), 2) for arg in ("--symlink", links[index], links[index + 1])),
        *("--bind", "/dev/shm/hl-agent-b", "/dev/shm"),
        *("--ro-bind-try", str(models), str(models), "--bind", str(work), str(work)),
        *("--perms", "0000", "--tmpfs", stored, "--remount-ro", stored),
        *DEV_NULL_MASK,
        str(models / "token"),
        *("--chdir", str(work), "--setenv", "HYPERLOOM_REAP_MARK", "m1", "--json-status-fd", "9"),
        *("--", "claude", "--print"),
    ]
    assert "--dev" not in argv and "--ro-bind" not in argv[: argv.index("--perms")]


def test_a_bind_below_tmp_comes_after_the_tmp_mount() -> None:
    policy = BwrapPolicy(read_only=(), writable=("/tmp/pytest-of-ci/wt",), tmp="/scratch/hl-agent-a")

    argv = policy.argv(["bwrap"], ["true"])

    tmp = argv.index("/tmp")
    assert argv[tmp - 2 : tmp + 1] == ["--bind", "/scratch/hl-agent-a", "/tmp"]
    assert argv.index("/tmp/pytest-of-ci/wt") > tmp


def test_argv_flags_follow_the_policy() -> None:
    joined = BwrapPolicy(read_only=(), die_with_parent=False, own_network=True, join=(7, 8)).argv(["bwrap"], ["true"])

    assert joined[1:9] == ["--userns", "7", "--pidns", "8", "--unshare-ipc", "--new-session", "--cap-drop", "ALL"]
    assert joined[9] == "--unshare-net"
    assert "--unshare-user" not in joined and "--unshare-pid" not in joined and "--die-with-parent" not in joined
    assert joined[joined.index("/tmp") - 1] == "--tmpfs" and joined[joined.index("/dev/shm") - 1] == "--tmpfs"


def test_a_symlinked_root_stays_a_symlink_when_its_target_is_bound(tmp_path: Path) -> None:
    (tmp_path / "usr" / "bin").mkdir(parents=True)
    (tmp_path / "bin").symlink_to("usr/bin")
    (tmp_path / "elsewhere").symlink_to("/nonexistent/target")
    usr, link, elsewhere = (str(tmp_path / name) for name in ("usr", "bin", "elsewhere"))

    argv = BwrapPolicy(read_only=(usr, link, elsewhere)).argv(["bwrap"], ["true"])

    assert argv[argv.index(link) - 2 : argv.index(link) + 1] == ["--symlink", f"{usr}/bin", link]
    assert argv[argv.index(elsewhere) - 1 : argv.index(elsewhere) + 2] == ["--ro-bind-try", elsewhere, elsewhere]


def test_a_writable_path_a_writable_bind_already_shows_is_left_out() -> None:
    policy = BwrapPolicy(
        read_only=("/work/ro",),
        writable=("/work", "/work/cache", "/work/ro/rw", "/work"),
    )

    argv = policy.argv(["bwrap"], ["true"])

    assert [argv[index + 1] for index, arg in enumerate(argv) if arg == "--bind"] == ["/work", "/work/ro/rw"]


# ---- what a policy refuses -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"read_only": ("usr",)}, "'usr' is not an absolute path"),
        ({"tmp": "scratch"}, "'scratch' is not an absolute path"),
        ({"read_only": ("/proc",)}, "/proc is mounted by the sandbox itself, so it cannot be bound or masked"),
        ({"writable": ("/proc/1",)}, "/proc/1 is mounted by the sandbox itself, so it cannot be bound or masked"),
        ({"read_only": ("/dev/dri",)}, "/dev/dri is mounted by the sandbox itself, so it cannot be bound or masked"),
        ({"masked": ("/",)}, "/ is mounted by the sandbox itself, so it cannot be bound or masked"),
        ({"read_only": ("/tmp",)}, "/tmp is mounted by the sandbox itself, so it cannot be bound or masked"),
        ({"devices": ("/srv/kfd",)}, "device /srv/kfd is not under /dev"),
        ({"sys_paths": ("/proc",)}, "/proc is not under /sys"),
    ],
)
def test_a_policy_with_a_misplaced_path_is_refused(changes: dict[str, object], problem: str) -> None:
    policy = BwrapPolicy(**{"read_only": (), **changes})  # type: ignore[arg-type]

    assert policy.problems((0, 13, 1)) == [problem]


def test_nothing_may_be_bound_at_or_above_an_app_only_path(tmp_path: Path) -> None:
    state = tmp_path / "app" / "state"
    state.mkdir(parents=True)
    (tmp_path / "link").symlink_to(tmp_path / "app")
    app = str(state)

    assert BwrapPolicy(read_only=(str(tmp_path / "app"),), app_only=(app,)).problems((0, 13, 1)) == [
        f"{tmp_path / 'app'} would expose {app}, which only the app may reach"
    ]
    assert BwrapPolicy(read_only=(), writable=(app,), app_only=(app,)).problems((0, 13, 1)) != []
    assert BwrapPolicy(read_only=(str(tmp_path / "link"),), app_only=(app,)).problems((0, 13, 1)) != []
    assert BwrapPolicy(read_only=(), tmp=str(tmp_path), app_only=(app,)).problems((0, 13, 1)) != []
    assert BwrapPolicy(read_only=(str(tmp_path / "app2"),), app_only=(app,)).problems((0, 13, 1)) == []


def test_a_read_only_bind_or_mask_below_a_writable_bind_needs_0_12() -> None:
    policy = BwrapPolicy(read_only=("/ws/.git",), writable=("/ws", "/ws/.home"), masked=("/ws/.home/token",))

    assert policy.problems((0, 11, 1)) == [
        "/ws/.git would be mounted below the agent-writable /ws, which needs bubblewrap 0.12.0 (this is 0.11.1)",
        "/ws/.home/token would be mounted below the agent-writable /ws, which needs bubblewrap 0.12.0 (this is 0.11.1)",
    ]
    assert policy.problems((0, 12, 0)) == []


def test_below_a_scratch_earlier_commands_shared_anything_needs_0_12() -> None:
    joined = BwrapPolicy(
        read_only=("/tmp/hyperloom-host/ca-bundle.pem",),
        writable=("/tmp/pytest-of-ci/ws",),
        tmp="/var/tmp/hl-agent-a",
        shm="/dev/shm/hl-agent-b",
        join=(7, 8),
    )

    assert joined.problems((0, 9, 0)) == [
        f"{path} would be mounted below the agent-writable /tmp, which needs bubblewrap 0.12.0 (this is 0.9.0)"
        for path in ("/tmp/hyperloom-host/ca-bundle.pem", "/tmp/pytest-of-ci/ws")
    ]
    assert joined.problems((0, 13, 1)) == []
    assert BwrapPolicy(read_only=joined.read_only, writable=joined.writable, join=(7, 8)).problems((0, 9, 0)) == []


def test_writable_below_writable_and_anything_below_the_scratch_mounts_pass_on_any_version() -> None:
    policy = BwrapPolicy(
        read_only=("/tmp/hyperloom-host/ca-bundle.pem", "/srv/models"),
        writable=("/ws", "/ws/.home", "/ws/.cache/ccache"),
        tmp="/var/tmp/hl-agent-a",
        shm="/dev/shm/hl-agent-b",
    )

    assert policy.problems((0, 5, 0)) == []


# ---- scratch, status ---------------------------------------------------------------------------------------------


def test_scratch_dirs_are_private_and_removable() -> None:
    first, second = Scratch.create(), Scratch.create()
    try:
        for path in (first.tmp, first.shm):
            assert stat.S_IMODE(os.stat(path).st_mode) == 0o700
            assert os.path.basename(path).startswith("hl-agent-")
        assert os.path.dirname(first.tmp) == tempfile.gettempdir()
        if os.access("/dev/shm", os.W_OK | os.X_OK):
            assert os.path.dirname(first.shm) == "/dev/shm"
        assert {first.tmp, first.shm}.isdisjoint({second.tmp, second.shm})
        locked = Path(first.tmp) / "left" / "locked"
        locked.mkdir(parents=True)
        (locked / "file").write_text("x")
        locked.chmod(0)
    finally:
        first.remove()
        second.remove()

    assert not any(os.path.exists(path) for path in (first.tmp, first.shm, second.tmp, second.shm))


def test_status_tells_a_command_that_ran_from_a_setup_that_failed() -> None:
    ran = '{ "child-pid": 2813551, "ipc-namespace": 4026534613, "pid-namespace": 4026534614 }\n{ "exit-code": 7 }\n'
    failed = '{ "child-pid": 2927880, "mnt-namespace": 4026534605, "pid-namespace": 4026535221 }\n'

    assert BwrapStatus.parse(ran) == BwrapStatus(child_pid=2813551, exit_code=7)
    assert BwrapStatus.parse(failed.encode()) == BwrapStatus(child_pid=2927880, exit_code=None)
    assert BwrapStatus.parse("") == BwrapStatus(child_pid=None, exit_code=None)


def test_interpreter_roots_are_outside_the_system_roots() -> None:
    roots = interpreter_roots()

    assert not any(root == system or root.startswith(system + "/") for root in roots for system in SYSTEM_ROOTS)
    if not any(sys.base_prefix == system or sys.base_prefix.startswith(system + "/") for system in SYSTEM_ROOTS):
        assert sys.base_prefix in roots


# ---- the probe ---------------------------------------------------------------------------------------------------


def test_probe_passes_a_canary_that_holds(fake: FakeBwrap) -> None:
    found = probe_bwrap()

    assert found == BwrapProbe(str(fake.path), (0, 13, 1), ())
    assert found.ok and found.launcher == (str(fake.path),)
    argv = fake.argv
    assert argv[argv.index("/usr") - 1] == "--ro-bind-try"  # a bind from the root mount, where an unbindable root fails
    assert "--die-with-parent" not in argv  # the probe may run on any thread


@pytest.mark.parametrize(
    ("configure", "problem"),
    [
        ({"stderr": UID_MAP_DENIED, "code": 1}, UID_MAP_DENIED),
        (
            {"stdout": json.dumps({**CANARY_OK, "CapEff": "000001ffffffffff"})},
            "the sandbox showed CapEff 000001ffffffffff",
        ),
        ({"stdout": json.dumps({**CANARY_OK, "mask_file": "ok"})}, "the sandbox showed mask_file ok"),
        (
            {"stdout": json.dumps({**CANARY_OK, "pid_namespace": "the app's"})},
            "the sandbox showed pid_namespace the app's",
        ),
        ({"stdout": "Traceback"}, "the sandbox printed no result: 'Traceback'"),
        ({"code": 1}, "bubblewrap exited 1"),
    ],
)
def test_probe_names_what_failed(fake: FakeBwrap, configure: dict[str, object], problem: str) -> None:
    fake.configure(**configure)  # type: ignore[arg-type]

    found = probe_bwrap()

    assert not found.ok and found.problems == (problem,) and not found.transient
    with pytest.raises(RuntimeError, match="cannot run a sandbox here"):
        _ = found.launcher


def test_probe_refuses_before_running_a_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert probe_bwrap().problems == ("bubblewrap: not found on PATH",)

    old = FakeBwrap(tmp_path / "old", version="0.4.0")
    assert probe_bwrap(str(old.path)).problems == ("bubblewrap 0.4.0 is older than 0.5.0 (needs --perms)",)
    assert old.runs == []

    mute = FakeBwrap(tmp_path / "mute", version="")
    mute.path.write_text('#!/bin/sh\necho "usage: bwrap [OPTIONS...]" >&2\n')
    assert probe_bwrap(str(mute.path)).problems == (
        f"bubblewrap at {mute.path} did not say its version: usage: bwrap [OPTIONS...]",
    )

    setuid = FakeBwrap(tmp_path / "setuid")
    setuid.path.chmod(0o4755)
    assert probe_bwrap(str(setuid.path)).problems == (
        f"bubblewrap at {setuid.path} is setuid; only the unprivileged build is supported",
    )
    assert setuid.runs == []


def test_probe_that_gets_no_answer_is_transient_and_kills_the_sandbox(fake: FakeBwrap) -> None:
    # The stand-in reports a sandbox pid 1 on the status descriptor, as bubblewrap does once it has cloned, then hangs
    # (bash: dash cannot redirect to a descriptor above 9).
    fake.path.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = --version ]; then echo "bubblewrap 0.13.1"; exit 0; fi\n'
        'fd=""; prev=""\n'
        'for arg in "$@"; do [ "$prev" = --json-status-fd ] && fd="$arg"; prev="$arg"; done\n'
        "/bin/sleep 300 &\n"
        f'echo $! > "{fake.directory}/sandbox"\n'
        'echo "{ \\"child-pid\\": $! }" >&"$fd"\n'
        "exec /bin/sleep 300\n"
    )

    found = probe_bwrap(timeout=0.5)

    assert found.transient and not found.ok
    assert found.problems == (f"bubblewrap at {fake.path}: no answer in 0.5 s",)
    sandbox = int((fake.directory / "sandbox").read_text())
    gone = _gone(sandbox)
    if not gone:
        os.kill(sandbox, signal.SIGKILL)
    assert gone, "the sandbox's pid 1 outlived the probe that gave up on it"


def _gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except OSError:
            return True
        if state in ("Z", "X"):
            return True
        time.sleep(0.05)
    return False


def test_an_unbindable_root_runs_every_launch_under_unshare_mount(fake: FakeBwrap) -> None:
    fake.configure(unbindable=UNBINDABLE)

    found = probe_bwrap()

    assert found.ok and fake.runs == ["plain", "1"]
    unshare = str(fake.directory / "unshare")
    assert found.prefix == (unshare, "--mount", "--")
    assert found.launcher == (unshare, "--mount", "--", str(fake.path))
    assert found.notes == (
        f"binds from the root mount fail here ({UNBINDABLE}): every launch runs under unshare --mount, which gives "
        "it a private copy of the mount tree",
    )
    assert main(["probe"]) == 0


def test_an_unbindable_root_that_unshare_does_not_fix_is_refused(
    fake: FakeBwrap, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake.configure(unbindable=UNBINDABLE, stderr=UID_MAP_DENIED, code=1)
    assert probe_bwrap().problems == (UNBINDABLE, f"under unshare --mount: {UID_MAP_DENIED}")

    bare = FakeBwrap(tmp_path / "bare", unshare=False)
    bare.configure(unbindable=UNBINDABLE)
    monkeypatch.setenv("PATH", str(bare.directory))
    assert probe_bwrap().problems == (UNBINDABLE, "unshare (util-linux) is not on PATH to give a launch its own mounts")


def test_the_probe_runs_once_per_key(fake: FakeBwrap, monkeypatch: pytest.MonkeyPatch) -> None:
    fake.configure(sleep=0.3)
    found = []
    threads = [threading.Thread(target=lambda: found.append(bwrap_probe())) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(found) == 6 and all(probe is found[0] and probe.ok for probe in found)
    assert fake.runs == ["plain"]
    assert bwrap_probe() is found[0] and fake.runs == ["plain"]

    info = fake.path.stat()
    os.utime(fake.path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
    assert bwrap_probe().ok and fake.runs == ["plain", "plain"]

    transient = BwrapProbe(str(fake.path), (0, 13, 1), ("no answer",), transient=True)
    calls = []
    monkeypatch.setattr(bwrap, "probe_bwrap", lambda path: calls.append(path) or transient)
    bwrap._PROBES.clear()
    assert bwrap_probe() is transient and bwrap_probe() is transient and len(calls) == 2


def test_main_prints_the_probe(fake: FakeBwrap, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["probe"]) == 0
    assert capsys.readouterr().out == f"ok bubblewrap 0.13.1 {fake.path}\n"

    fake.configure(stderr=UID_MAP_DENIED, code=1)
    assert main(["probe"]) == 1
    assert capsys.readouterr().out == f"{UID_MAP_DENIED}\n"
