# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The bubblewrap adapter without namespaces: the argv a policy becomes, the policies it refuses, and the probe read
against a stand-in ``bwrap`` (real sandboxes: ``test_bwrap_launch.py``)."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import signal
import stat
import subprocess
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
    BwrapPolicyError,
    BwrapProbe,
    BwrapSetupError,
    BwrapStatus,
    Holder,
    Scratch,
    bwrap_probe,
    interpreter_roots,
    main,
    probe_bwrap,
)
from hyperloom_common.tests._bwrap_host import as_owner

#: What the canary printed on this pod's bubblewrap 0.6.1 to 0.13.1 (``python3 -I bwrap.py probe``).
CANARY_OK = {
    "mask_dir": "EACCES",
    "mask_file": "EACCES",
    "ro_read": "ok",
    "ro_create": "EROFS",
    "ro_chmod": "EROFS",
    "rw_write": "ok",
    "umount_mask": "EPERM",
    "dev_null": "ok",
    "pid_namespace": "own",
    "NoNewPrivs": "1",
    "CapEff": "0000000000000000",
}
UID_MAP_DENIED = "bwrap: setting up uid map: Permission denied"
UNBINDABLE = (
    "bwrap: Can't bind mount /oldroot/usr on /newroot/usr: Unable to mount source on destination: Invalid argument"
)
#: What the layer's bwrap printed on the target pool (root in a user namespace mapping one outside uid, apptainer's
#: unbindable root): ``bwrap --unshare-user --uid 0 --gid 0 --unshare-pid --unshare-net --cap-drop ALL --ro-bind / /
#: --proc /proc --dev /dev true``.
FIELD_UNBINDABLE = "bwrap: Can't bind mount / on /: Invalid argument"
#: The same with ``--dev /dev`` under ``unshare -U -r -m``: apptainer's ``--containall`` /dev has no ``/dev/full``.
FIELD_NO_DEV_FULL = "bwrap: Can't open source /dev/full: No such file or directory"


def directories(monkeypatch: pytest.MonkeyPatch, *paths: str) -> None:
    """Make ``paths`` read as directories, the only thing a mask below a bound tree may be: the policies here name
    paths no host has."""
    real = bwrap._is_directory
    monkeypatch.setattr(bwrap, "_is_directory", lambda path: path in paths or real(path))


@pytest.fixture(autouse=True)
def _fresh_probe_cache():
    bwrap._PROBES.clear()
    yield
    bwrap._PROBES.clear()


UNSHARE_DENIED = "unshare: unshare failed: Operation not permitted"
#: ``unshare --help`` of the static util-linux 2.37.2 build a hosting layer ships (its option list; the probe reads
#: options, never the version).
UNSHARE_HELP = """
Usage:
 unshare [options] [<program> [<argument>...]]

Run a program with some namespaces unshared from the parent.

Options:
 -m, --mount[=<file>]      unshare mounts namespace
 -u, --uts[=<file>]        unshare UTS namespace (hostname etc)
 -i, --ipc[=<file>]        unshare System V IPC namespace
 -n, --net[=<file>]        unshare network namespace
 -p, --pid[=<file>]        unshare pid namespace
 -U, --user[=<file>]       unshare user namespace
 -C, --cgroup[=<file>]     unshare cgroup namespace
 -T, --time[=<file>]       unshare time namespace

 -f, --fork                fork before launching <program>
 --map-user=<uid>|<name>   map current user to uid (implies --user)
 --map-group=<gid>|<name>  map current group to gid (implies --user)
 -r, --map-root-user       map current user to root (implies --user)
 -c, --map-current-user    map current user to itself (implies --user)

 --kill-child[=<signame>]  when dying, kill the forked child (implies --fork)
                             defaults to SIGKILL
 --mount-proc[=<dir>]      mount proc filesystem first (implies --mount)
 --propagation slave|shared|private|unchanged
                           modify mount propagation in mount namespace
 --setgroups allow|deny    control the setgroups syscall in user namespaces
 --keep-caps               retain capabilities granted in user namespaces

 -h, --help                display this help
 -V, --version             display version

For more details see unshare(1).
"""
USER_MOUNTS = ("--user", "--map-current-user", "--mount", "--propagation", "private", "--")
ROOT_MOUNTS = ("--mount", "--propagation", "private", "--")


class FakeBwrap:
    """A ``bwrap`` stand-in in its own directory: answers ``--version``, and for a launch records its argv, writes
    ``stdout``/``stderr`` and exits ``code``. With ``unbindable`` it fails every launch that is not under the fake
    ``unshare`` beside it, which runs it under ``--mount`` (marked ``1`` in :attr:`runs`) or ``--user
    --map-current-user --mount --propagation private`` (``user``) only where :meth:`configure_unshare` allows that
    tier, and fails as ``unshare`` does without the privilege otherwise."""

    def __init__(self, directory: Path, *, version: str = "0.13.1", unshare: bool = True) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "bwrap"
        self.version = version
        self.configure()
        if unshare:
            self.configure_unshare()
            script = directory / "unshare"
            script.write_text(
                "#!/bin/sh\n"
                f'if [ "$1" = --help ]; then /bin/cat "{directory}/unshare-help"; exit 0; fi\n'
                'flags=""; while [ "$1" != -- ]; do flags="$flags $1"; shift; done; shift\n'
                'case "$flags" in\n'
                '  " --mount --propagation private"|" --mount") mode=1 ;;\n'
                '  " --user --map-current-user --mount --propagation private") mode=user ;;\n'
                "  *) exit 9 ;;\n"
                "esac\n"
                f'echo "$flags" >> "{directory}/unshare-runs"\n'
                f'case " $(/bin/cat "{directory}/unshare-allows") " in\n'
                f'  *" $mode "*) FAKE_PRIVATE=$mode exec "$@" ;;\n'
                f'  *) echo "{UNSHARE_DENIED}" >&2; exit 1 ;;\n'
                "esac\n"
            )
            script.chmod(0o755)

    def configure_unshare(self, *, allows: tuple[str, ...] = ("1",), help_text: str = UNSHARE_HELP) -> None:
        """Which tiers the fake ``unshare`` runs: ``1`` (``--mount``, as root) and ``user`` (``--user ...``)."""
        (self.directory / "unshare-allows").write_text(" ".join(allows) + "\n")
        (self.directory / "unshare-help").write_text(help_text)

    def configure(
        self,
        *,
        stdout: str = json.dumps(CANARY_OK),
        stderr: str = "",
        code: int = 0,
        sleep: float = 0,
        unbindable: str = "",
        dev_full_missing: bool = False,
    ) -> None:
        (self.directory / "stdout").write_text(stdout + "\n")
        (self.directory / "stderr").write_text(stderr + "\n" if stderr else "")
        trap = f'if [ -z "$FAKE_PRIVATE" ]; then echo "{unbindable}" >&2; exit 1; fi\n' if unbindable else ""
        pause = f"/bin/sleep {sleep}\n" if sleep else ""
        if dev_full_missing:  # bubblewrap's --dev opens each standard node; this host has no /dev/full
            trap += f'for arg in "$@"; do [ "$arg" = --dev ] && {{ echo "{FIELD_NO_DEV_FULL}" >&2; exit 1; }}; done\n'

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
    def unshare_runs(self) -> list[str]:
        """The flags of each launch the fake ``unshare`` started."""
        path = self.directory / "unshare-runs"
        return [line.strip() for line in path.read_text().splitlines()] if path.exists() else []

    @property
    def runs(self) -> list[str]:
        path = self.directory / "runs"
        return path.read_text().split() if path.exists() else []

    @property
    def argv(self) -> list[str]:
        return (self.directory / "argv").read_text().splitlines()


HOST_DEV_NOTES = bwrap._dev_notes


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeBwrap:
    """A stand-in ``bwrap`` first on ``PATH``, on a host whose ``/dev`` has every standard node (the tests that read
    the note on an absent node put the real :func:`bwrap._dev_notes` back)."""
    stand_in = FakeBwrap(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(stand_in.directory))
    monkeypatch.setattr(bwrap, "_dev_notes", lambda: ())
    return stand_in


def host_dev(monkeypatch: pytest.MonkeyPatch, *nodes: str) -> None:
    """A host ``/dev`` holding only ``nodes`` (other paths are read as they are), read by the real note."""
    exists = os.path.exists
    monkeypatch.setattr(
        bwrap.os.path, "exists", lambda path: path in nodes if path.startswith("/dev/") else exists(path)
    )
    monkeypatch.setattr(bwrap, "_dev_notes", HOST_DEV_NOTES)


def probed(path: str = "bwrap", version: tuple[int, int, int] = (0, 13, 1)) -> BwrapProbe:
    """A probe that passed, of bubblewrap ``version`` at ``path``."""
    return BwrapProbe(path, version, ())


# ---- the argv ------------------------------------------------------------------------------------------------------


def test_argv_mounts_by_depth_with_the_masks_last(tmp_path: Path) -> None:
    models = tmp_path / "models"
    (models / "stored_tokens").mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    policy = BwrapPolicy(
        read_only=("/usr", str(models)),
        writable=(str(work),),
        masked=(str(models / "stored_tokens"),),
        devices=("/dev/kfd",),
        tmp="/var/tmp/hl-agent-a",
        shm="/dev/shm/hl-agent-b",
        marks=(("HYPERLOOM_REAP_MARK", "m1"),),
        chdir=str(work),
    )

    argv = policy.argv(probed("/opt/bin/bwrap"), ["claude", "--print"], status_fd=9)

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
        *("--remount-ro", "/"),
        *("--chdir", str(work), "--setenv", "HYPERLOOM_REAP_MARK", "m1", "--json-status-fd", "9"),
        *("--", "claude", "--print"),
    ]
    assert "--dev" not in argv and "--ro-bind" not in argv


def test_below_a_bound_tree_only_a_directory_is_masked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A mount over a file goes with the file's first rename on the host, and an absent path has none to cover what the
    # app makes there later: the agent would read a credential the app refreshed or logged into during the task.
    home = tmp_path / "home"
    tool = home / ".config" / "tool"
    tool.mkdir(parents=True)
    (tool / "credentials.json").write_text("{}")
    (home / "link").symlink_to(tool)
    outside = tmp_path / "outside"
    outside.write_text("x")
    file, absent, link = str(tool / "credentials.json"), str(tool / "token"), str(home / "link")
    policy = BwrapPolicy(read_only=(str(home),), masked=(str(tool), file, absent, link, str(outside)))
    # Outside every bound tree (a fresh tmpfs /tmp here) nothing shows, so a file or absent path there is left out.
    elsewhere = BwrapPolicy(read_only=(str(home),), masked=(str(tool), str(outside), str(tmp_path / "absent")))
    shared = BwrapPolicy(read_only=(), masked=("/tmp/hl-key",), tmp="/var/tmp/hl-agent-a")

    assert policy.problems((0, 13, 1)) == [bwrap._not_a_directory(path) for path in (file, absent, link)]
    assert shared.problems((0, 13, 1)) == [bwrap._not_a_directory("/tmp/hl-key")]
    assert "is masked below a bound tree but is not a directory" in bwrap._not_a_directory(file)
    with pytest.raises(BwrapPolicyError) as refused:
        policy.argv(probed(), ["true"])
    assert refused.value.problems == tuple(policy.problems((0, 13, 1)))
    assert elsewhere.problems((0, 13, 1)) == []
    masks = [arg for arg in elsewhere.argv(probed(), ["true"]) if arg in (str(outside), str(tmp_path / "absent"))]
    assert masks == [] and "--ro-bind" not in elsewhere.argv(probed(), ["true"])

    # Checked again as each mount is made, after the policy's problems were read.
    monkeypatch.setattr(BwrapPolicy, "_mask_problems", lambda self: [])
    with pytest.raises(BwrapPolicyError, match="credentials.json is masked below a bound tree"):
        policy.argv(probed(), ["true"])


def test_a_bind_below_tmp_comes_after_the_tmp_mount() -> None:
    policy = BwrapPolicy(read_only=(), writable=("/tmp/pytest-of-ci/wt",), tmp="/scratch/hl-agent-a")

    argv = policy.argv(probed(), ["true"])

    tmp = argv.index("/tmp")
    assert argv[tmp - 2 : tmp + 1] == ["--bind", "/scratch/hl-agent-a", "/tmp"]
    assert argv.index("/tmp/pytest-of-ci/wt") > tmp


def test_argv_flags_follow_the_policy() -> None:
    joined = BwrapPolicy(read_only=(), die_with_parent=False, own_network=True, join=(7, 8)).argv(probed(), ["true"])

    assert joined[1:9] == ["--userns", "7", "--pidns", "8", "--unshare-ipc", "--new-session", "--cap-drop", "ALL"]
    assert joined[9] == "--unshare-net"
    assert "--unshare-user" not in joined and "--unshare-pid" not in joined and "--die-with-parent" not in joined
    assert joined[joined.index("/tmp") - 1] == "--tmpfs" and joined[joined.index("/dev/shm") - 1] == "--tmpfs"


def test_a_symlinked_root_stays_a_symlink_when_its_target_is_bound(tmp_path: Path) -> None:
    (tmp_path / "usr" / "bin").mkdir(parents=True)
    (tmp_path / "bin").symlink_to("usr/bin")
    (tmp_path / "elsewhere").symlink_to("/nonexistent/target")
    usr, link, elsewhere = (str(tmp_path / name) for name in ("usr", "bin", "elsewhere"))

    argv = BwrapPolicy(read_only=(usr, link, elsewhere)).argv(probed(), ["true"])

    assert argv[argv.index(link) - 2 : argv.index(link) + 1] == ["--symlink", f"{usr}/bin", link]
    assert argv[argv.index(elsewhere) - 1 : argv.index(elsewhere) + 2] == ["--ro-bind-try", elsewhere, elsewhere]


def test_a_writable_path_a_writable_bind_already_shows_is_left_out() -> None:
    policy = BwrapPolicy(
        read_only=("/work/ro",),
        writable=("/work", "/work/cache", "/work/ro/rw", "/work"),
    )

    argv = policy.argv(probed(), ["true"])

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


def test_a_read_only_bind_or_mask_below_a_writable_bind_needs_0_12(monkeypatch: pytest.MonkeyPatch) -> None:
    directories(monkeypatch, "/ws/.home/token")
    policy = BwrapPolicy(read_only=("/ws/.git",), writable=("/ws", "/ws/.home"), masked=("/ws/.home/token",))

    assert policy.problems((0, 11, 1)) == [
        "/ws/.git would be mounted below the agent-writable /ws, which needs bubblewrap 0.12.0 (this is 0.11.1)",
        "/ws/.home/token would be mounted below the agent-writable /ws, which needs bubblewrap 0.12.0 (this is 0.11.1)",
    ]
    assert policy.problems((0, 12, 0)) == []


@pytest.mark.parametrize("join", [None, (7, 8)])
def test_below_a_bound_scratch_anything_needs_0_12(
    join: tuple[int, int] | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A Scratch an earlier launch or command used may hold a link it left, whether or not this launch joins a holder.
    directories(monkeypatch, "/tmp/hl-key")
    policy = BwrapPolicy(
        read_only=("/tmp/hyperloom-host/ca-bundle.pem",),
        writable=("/tmp/pytest-of-ci/ws",),
        masked=("/tmp/hl-key",),
        devices=("/dev/shm/hl-node",),
        tmp="/var/tmp/hl-agent-a",
        shm="/dev/shm/hl-agent-b",
        join=join,
    )

    assert policy.problems((0, 11, 1)) == [
        f"{path} would be mounted below the agent-writable {parent}, which needs bubblewrap 0.12.0 (this is 0.11.1)"
        for path, parent in (
            ("/tmp/hyperloom-host/ca-bundle.pem", "/tmp"),
            ("/tmp/hl-key", "/tmp"),
            ("/tmp/pytest-of-ci/ws", "/tmp"),
            ("/dev/shm/hl-node", "/dev/shm"),
        )
    ]
    assert policy.problems((0, 12, 0)) == []


def test_writable_below_writable_and_anything_below_a_fresh_tmp_pass_on_any_version() -> None:
    policy = BwrapPolicy(
        read_only=("/tmp/hyperloom-host/ca-bundle.pem", "/srv/models"),
        writable=("/ws", "/ws/.home", "/ws/.cache/ccache", "/tmp/pytest-of-ci/ws"),
        masked=("/tmp/hl-key",),
        devices=("/dev/shm/hl-node",),
    )

    assert policy.problems((0, 5, 0)) == []


def test_a_launch_the_probed_bwrap_must_not_make_is_refused_before_it_starts(fake: FakeBwrap) -> None:
    policy = BwrapPolicy(read_only=("/ws/.git",), writable=("/ws",), tmp="/var/tmp/hl-agent-a")
    old = probed(str(fake.path), (0, 11, 1))

    with pytest.raises(BwrapPolicyError) as refused:
        policy.argv(old, ["true"])
    assert refused.value.problems == tuple(policy.problems((0, 11, 1))) != ()
    with pytest.raises(BwrapPolicyError):
        Holder.start(old, policy)
    with pytest.raises(BwrapPolicyError):
        Holder.start(old, BwrapPolicy(read_only=("/tmp/pytest-of-ci/ca.pem",), tmp="/var/tmp/hl-agent-a"))
    assert fake.runs == []
    assert policy.argv(probed(str(fake.path), (0, 12, 0)), ["true"])[0] == str(fake.path)
    with pytest.raises(RuntimeError, match="no answer"):
        policy.argv(BwrapProbe(str(fake.path), (0, 13, 1), ("no answer",), transient=True), ["true"])


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


def test_scratch_an_agent_locked_is_removed_by_its_owner() -> None:
    # The sandbox maps to the owning uid, so an agent can chmod 000 its /tmp and /dev/shm themselves and a directory
    # in them; root here is made to honour the modes, as the plain user CI and a deployment run as does.
    made = Scratch.create()
    for root in (made.tmp, made.shm):
        locked = Path(root) / "locked"
        (locked / "inner").mkdir(parents=True)
        (locked / "inner" / "file").write_text("x")
        (locked / "inner").chmod(0)
        locked.chmod(0)
        os.chmod(root, 0)

    done = as_owner(
        "import sys; from hyperloom_common.bwrap import Scratch; Scratch(*sys.argv[1:]).remove()", made.tmp, made.shm
    )

    left = [path for path in (made.tmp, made.shm) if os.path.lexists(path)]
    for path in left:
        subprocess.run(["chmod", "-R", "u+rwx", path], check=False)
        shutil.rmtree(path)
    assert done.returncode == 0 and left == [], done.stderr


def test_scratch_removes_dev_shm_when_tmp_fails_and_names_each_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    made = Scratch.create()
    remove = bwrap._remove_tree
    refused = {made.tmp}

    def failing(path: str) -> None:
        if path in refused:
            raise PermissionError(13, "Permission denied", path)
        remove(path)

    monkeypatch.setattr(bwrap, "_remove_tree", failing)
    again = Scratch(made.tmp, tempfile.mkdtemp(prefix="hl-agent-"))
    try:
        with pytest.raises(PermissionError) as one:
            made.remove()
        shm_left = os.path.lexists(made.shm)
        refused.add(again.shm)
        with pytest.raises(OSError, match=f"scratch not removed: .*{made.tmp}.*; .*{again.shm}"):
            again.remove()
    finally:
        for path in (made.tmp, made.shm, again.shm):
            if os.path.lexists(path):
                remove(path)

    assert one.value.filename == made.tmp and not shm_left


def _descend(root: str, name: str, levels: int) -> None:
    """Make ``levels`` nested directories ``name`` under ``root``, each relative to the last: no path is spelled out."""
    current = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for _ in range(levels):
            os.mkdir(name, dir_fd=current)
            below = os.open(name, os.O_RDONLY | os.O_DIRECTORY, dir_fd=current)
            os.close(current)
            current = below
        os.close(os.open("f", os.O_WRONLY | os.O_CREAT, 0o600, dir_fd=current))
        os.mkfifo("pipe", dir_fd=current)
        os.symlink("/", "up", dir_fd=current)
        os.mkdir("locked", 0, dir_fd=current)
    finally:
        os.close(current)


@pytest.mark.parametrize(
    ("name", "levels"), [("d", 1500), ("y" * 200, 30)], ids=["deeper-than-recursion", "longer-than-path-max"]
)
def test_scratch_removes_a_tree_too_deep_or_too_long_for_a_path(name: str, levels: int) -> None:
    made = Scratch.create()
    _descend(made.tmp, name, levels)
    os.mkdir(os.path.join(made.shm, "keep"))

    try:
        made.remove()
    finally:
        left = [path for path in (made.tmp, made.shm) if os.path.lexists(path)]
        for path in left:
            subprocess.run(["rm", "-rf", "--one-file-system", path], check=False)

    assert left == []


def test_scratch_removes_dev_shm_whatever_stops_tmp(monkeypatch: pytest.MonkeyPatch) -> None:
    made = Scratch.create()
    remove = bwrap._remove_tree

    def failing(path: str) -> None:
        if path == made.tmp:
            raise RecursionError("maximum recursion depth exceeded")
        remove(path)

    monkeypatch.setattr(bwrap, "_remove_tree", failing)
    try:
        with pytest.raises(RecursionError):
            made.remove()
        shm_left = os.path.lexists(made.shm)
    finally:
        for path in (made.tmp, made.shm):
            if os.path.lexists(path):
                remove(path)

    assert not shm_left


def test_remove_tree_refuses_what_is_not_a_directory(tmp_path: Path) -> None:
    (tmp_path / "file").write_text("x")
    (tmp_path / "link").symlink_to(tmp_path / "kept")
    (tmp_path / "kept").mkdir()

    for name in ("file", "link"):
        with pytest.raises(NotADirectoryError):
            bwrap._remove_tree(str(tmp_path / name))
    with pytest.raises(FileNotFoundError):
        bwrap._remove_tree(str(tmp_path / "absent"))

    assert (tmp_path / "file").exists() and (tmp_path / "kept").is_dir()


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
    assert found.prefix == (unshare, *ROOT_MOUNTS)
    assert found.launcher == (unshare, *ROOT_MOUNTS, str(fake.path))
    assert found.notes == (
        f"binds from the root mount fail here ({UNBINDABLE}): every launch runs under unshare --mount, which gives "
        "it a private copy of the mount tree",
    )
    assert main(["probe"]) == 0


def test_an_unbindable_root_that_unshare_does_not_fix_is_refused(
    fake: FakeBwrap, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake.configure(unbindable=UNBINDABLE, stderr=UID_MAP_DENIED, code=1)
    fake.configure_unshare(allows=("1", "user"))
    assert probe_bwrap().problems == (
        UNBINDABLE,
        f"under unshare --mount: {UID_MAP_DENIED}",
        f"under unshare --user --mount: {UID_MAP_DENIED}",
    )
    assert fake.runs == ["plain", "1", "user"]

    bare = FakeBwrap(tmp_path / "bare", unshare=False)
    bare.configure(unbindable=UNBINDABLE)
    monkeypatch.setenv("PATH", str(bare.directory))
    assert probe_bwrap().problems == (UNBINDABLE, "unshare (util-linux) is not on PATH to give a launch its own mounts")


def test_root_keeps_unshare_mount_where_both_tiers_hold(fake: FakeBwrap) -> None:
    fake.configure(unbindable=UNBINDABLE)
    fake.configure_unshare(allows=("1", "user"))

    found = probe_bwrap()

    assert found.ok and found.joins and fake.runs == ["plain", "1"]
    assert found.prefix == (str(fake.directory / "unshare"), *ROOT_MOUNTS)


def test_a_uid_without_cap_sys_admin_runs_every_launch_under_unshare_user(
    fake: FakeBwrap, capsys: pytest.CaptureFixture[str]
) -> None:
    # As any uid but root: unshare --mount is refused, a user namespace of its own is not.
    fake.configure(unbindable=UNBINDABLE)
    fake.configure_unshare(allows=("user",))

    found = probe_bwrap()

    unshare = str(fake.directory / "unshare")
    assert found.ok and fake.runs == ["plain", "user"], "unshare --mount never started bubblewrap"
    assert found.prefix == (unshare, *USER_MOUNTS)
    assert found.launcher == (unshare, *USER_MOUNTS, str(fake.path))
    assert not found.joins, "each launch has a user namespace of its own, from which no other can be joined"
    assert found.notes == (
        f"binds from the root mount fail here ({UNBINDABLE}) and unshare --mount does not run (under unshare --mount: "
        f"{UNSHARE_DENIED}): every launch runs under unshare --user --map-current-user --mount --propagation private, "
        "which gives it a private copy of the mount tree in a user namespace of its own, so no launch joins another "
        "sandbox's namespaces",
    )
    assert main(["probe"]) == 0
    assert "unshare --user --map-current-user" in capsys.readouterr().out


def test_a_launch_that_joins_a_holder_is_refused_under_unshare_user(tmp_path: Path) -> None:
    alone = BwrapProbe("/opt/bin/bwrap", (0, 13, 1), (), prefix=("/opt/bin/unshare", *USER_MOUNTS), joins=False)
    policy = BwrapPolicy(read_only=("/usr",))

    assert policy.argv(alone, ["true"])[: len(USER_MOUNTS) + 2] == ["/opt/bin/unshare", *USER_MOUNTS, "/opt/bin/bwrap"]
    with pytest.raises(BwrapPolicyError) as refused:
        dataclasses.replace(policy, join=(7, 8)).argv(alone, ["true"])
    assert refused.value.problems == (bwrap.NO_JOIN,)
    assert dataclasses.replace(policy, join=(7, 8)).argv(probed("/opt/bin/bwrap"), ["true"])[1:3] == ["--userns", "7"]


def test_an_unshare_without_the_user_namespace_flags_is_not_tried(fake: FakeBwrap) -> None:
    # An unshare whose help lists no --map-current-user (busybox, an old util-linux): the tier is refused before any
    # launch under it. Only the option list counts, never a version.
    fake.configure(unbindable=UNBINDABLE)
    fake.configure_unshare(
        allows=("user",), help_text="unshare from util-linux 2.40\nOptions:\n -m, --mount\n --propagation x\n"
    )
    unshare = fake.directory / "unshare"

    found = probe_bwrap()

    assert found.problems == (
        UNBINDABLE,
        f"under unshare --mount: {UNSHARE_DENIED}",
        f"unshare at {unshare} does not list --user, --map-current-user in its --help, which unshare --user needs here",
    )
    assert fake.runs == ["plain"]


def test_the_field_unbindable_root_runs_under_the_layer_unshare_2_37_2(fake: FakeBwrap, tmp_path: Path) -> None:
    # The target pool: the layer's bwrap fails every bind from the root mount, and its static unshare from
    # util-linux 2.37.2 lists --map-current-user and --propagation, so the user-namespace tier is taken.
    layer = FakeBwrap(tmp_path / "layer")
    layer.configure(unbindable=FIELD_UNBINDABLE)
    layer.configure_unshare(allows=("user",))

    found = bwrap_probe(str(layer.path))

    assert found.ok and layer.runs == ["plain", "user"] and not found.joins
    assert found.prefix == (str(layer.directory / "unshare"), *USER_MOUNTS)
    assert layer.unshare_runs == ["--mount --propagation private", " ".join(USER_MOUNTS[:-1])]
    assert found.notes[0].startswith(f"binds from the root mount fail here ({FIELD_UNBINDABLE}) and unshare --mount")


def test_unshare_mount_asks_for_private_propagation_where_the_unshare_lists_it(fake: FakeBwrap) -> None:
    fake.configure(unbindable=FIELD_UNBINDABLE)

    assert probe_bwrap().prefix == (str(fake.directory / "unshare"), *ROOT_MOUNTS)
    assert fake.unshare_runs == ["--mount --propagation private"]

    bwrap._PROBES.clear()
    fake.configure_unshare(help_text="Usage: unshare [-m] PROG\n")  # lists no --propagation: its default stands
    assert probe_bwrap().prefix == (str(fake.directory / "unshare"), "--mount", "--")


def test_a_sandbox_dev_is_built_from_the_nodes_the_host_has_never_bwraps_dev(
    fake: FakeBwrap, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The target pool's /dev (apptainer --containall) has no /dev/full: bubblewrap's --dev fails there, so each
    # standard node is bound onto a tmpfs where the host has it, and the probe says which are absent.
    fake.configure(unbindable=FIELD_UNBINDABLE, dev_full_missing=True)
    fake.configure_unshare(allows=("user",))
    host_dev(monkeypatch, *(node for node in BASE_DEV_NODES if node != "/dev/full"))

    found = probe_bwrap()

    assert found.ok, found.problems
    argv = fake.argv
    pairs = [argv[i : i + 2] for i in range(len(argv))]
    assert "--dev" not in argv and ["--dev-bind", "/dev"] not in pairs and ["--tmpfs", "/dev"] in pairs
    assert all(argv[argv.index(node) - 1] == "--dev-bind-try" for node in BASE_DEV_NODES)
    assert found.notes[-1] == (
        "the host's /dev has no /dev/full: each sandbox's /dev is a tmpfs with only the nodes the host has (/dev/null, "
        "/dev/zero, /dev/random, /dev/urandom, /dev/tty) bound into it, never bubblewrap's --dev or the host's whole "
        "/dev"
    )


def test_the_dev_note_names_the_standard_nodes_the_host_lacks(monkeypatch: pytest.MonkeyPatch) -> None:
    host_dev(monkeypatch, *BASE_DEV_NODES)
    assert bwrap._dev_notes() == ()

    host_dev(monkeypatch, "/dev/null", "/dev/tty")
    assert bwrap._dev_notes() == (
        "the host's /dev has no /dev/zero, /dev/full, /dev/random, /dev/urandom: each sandbox's /dev is a tmpfs with "
        "only the nodes the host has (/dev/null, /dev/tty) bound into it, never bubblewrap's --dev or the host's whole "
        "/dev",
    )

    host_dev(monkeypatch)
    assert bwrap._dev_notes()[0].startswith(f"the host's /dev has no {', '.join(BASE_DEV_NODES)}: ")
    assert "the host has (none) bound" in bwrap._dev_notes()[0]


def test_a_sandbox_whose_dev_null_does_not_work_is_refused(fake: FakeBwrap) -> None:
    fake.configure(stdout=json.dumps({**CANARY_OK, "dev_null": "ENOENT"}))

    assert probe_bwrap().problems == ("the sandbox showed dev_null ENOENT",)


def test_an_unbindable_root_where_neither_unshare_tier_runs_is_refused(fake: FakeBwrap) -> None:
    # A node that lets this uid make no user namespace: both tiers fail, and the run is refused, never unconfined.
    fake.configure(unbindable=UNBINDABLE)
    fake.configure_unshare(allows=())

    assert probe_bwrap().problems == (
        UNBINDABLE,
        f"under unshare --mount: {UNSHARE_DENIED}",
        f"under unshare --user --mount: {UNSHARE_DENIED}",
    )
    assert fake.runs == ["plain"]


def test_a_layer_bubblewrap_takes_unshare_user_only_from_beside_it(fake: FakeBwrap, tmp_path: Path) -> None:
    layer = FakeBwrap(tmp_path / "layer")
    layer.configure(unbindable=UNBINDABLE)
    layer.configure_unshare(allows=("user",))
    fake.configure_unshare(allows=("1", "user"))

    found = bwrap_probe(str(layer.path))

    assert found.ok and found.prefix == (str(layer.directory / "unshare"), *USER_MOUNTS) and not found.joins
    assert layer.runs == ["plain", "user"] and fake.runs == []


def test_a_bubblewrap_given_by_path_retries_only_with_the_unshare_beside_it(
    fake: FakeBwrap, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A hosted run names its layer's bubblewrap; an unshare from PATH is the image's, which the layer does not vouch
    # for. The fake on PATH has an unshare beside it.
    layer = FakeBwrap(tmp_path / "layer")
    layer.configure(unbindable=UNBINDABLE)
    bare = FakeBwrap(tmp_path / "bare", unshare=False)
    bare.configure(unbindable=UNBINDABLE)

    beside = probe_bwrap(str(layer.path))
    refused = bwrap_probe(str(bare.path))

    assert beside.ok and beside.prefix == (str(layer.directory / "unshare"), *ROOT_MOUNTS)
    assert layer.runs == ["plain", "1"]
    assert refused.problems == (
        UNBINDABLE,
        f"no unshare beside {bare.path} to give a launch its own mounts, and none is taken from PATH for a bubblewrap "
        "named by path: the host must make the root mount private first (mount --make-rprivate /), or the layer "
        "must carry a static unshare (util-linux; its --help must list --user, --map-current-user, --mount and "
        "--propagation) as bin/unshare",
    )
    assert bare.runs == ["plain"] and fake.runs == []
    assert main(["probe", "--bwrap", str(bare.path)]) == 1
    assert "no unshare beside" in capsys.readouterr().out

    # The same binary looked up on PATH may use the unshare there.
    (fake.directory / "bwrap").unlink()
    (fake.directory / "bwrap").symlink_to(bare.path)
    assert bwrap_probe().ok and bwrap_probe().prefix == (str(fake.directory / "unshare"), *ROOT_MOUNTS)


def test_the_probe_of_a_bubblewrap_named_by_path_is_not_the_one_found_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The same binary: found on PATH it may use the unshare there, named by path only one beside it.
    bare = FakeBwrap(tmp_path / "bare", unshare=False)
    bare.configure(unbindable=UNBINDABLE)
    other = FakeBwrap(tmp_path / "other")
    monkeypatch.setenv("PATH", f"{bare.directory}{os.pathsep}{other.directory}")

    found = bwrap_probe()
    named = bwrap_probe(str(bare.path))

    assert found.ok and found.prefix == (str(other.directory / "unshare"), *ROOT_MOUNTS)
    assert not named.ok and "no unshare beside" in named.problems[1]
    assert bwrap_probe() is found


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
    monkeypatch.setattr(bwrap, "_probe", lambda path, unshare: calls.append(path) or transient)
    bwrap._PROBES.clear()
    assert bwrap_probe() is transient and bwrap_probe() is transient and len(calls) == 2


def test_main_prints_the_probe(fake: FakeBwrap, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["probe"]) == 0
    assert capsys.readouterr().out == f"ok bubblewrap 0.13.1 {fake.path}\n"

    fake.configure(stderr=UID_MAP_DENIED, code=1)
    assert main(["probe"]) == 1
    assert capsys.readouterr().out == f"{UID_MAP_DENIED}\n"


def test_the_probe_runs_the_bubblewrap_it_is_given(
    fake: FakeBwrap, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layer = FakeBwrap(tmp_path / "layer", version="0.13.1")

    assert bwrap_probe(str(layer.path)).path == str(layer.path) and layer.runs == ["plain"] and fake.runs == []
    assert bwrap_probe(str(layer.path)) is bwrap_probe(str(layer.path)) and layer.runs == ["plain"]
    assert bwrap_probe().path == str(fake.path) and fake.runs == ["plain"] and layer.runs == ["plain"]
    assert probe_bwrap("bin/bwrap").problems == ("bubblewrap at bin/bwrap is not an absolute path",)

    assert main(["probe", "--bwrap", str(layer.path)]) == 0
    assert capsys.readouterr().out == f"ok bubblewrap 0.13.1 {layer.path}\n"
    assert layer.runs == ["plain", "plain"] and fake.runs == ["plain"]


def test_a_canary_tree_where_the_sandbox_mounts_its_own_is_a_problem(
    fake: FakeBwrap, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not os.access("/dev/shm", os.W_OK | os.X_OK):
        pytest.skip("no writable /dev/shm to put the canary's tree in")
    monkeypatch.setattr(tempfile, "tempdir", "/dev/shm")

    problems = probe_bwrap().problems

    assert problems and all(problem.startswith("the canary's policy: /dev/shm/hl-bwrap-probe-") for problem in problems)
    assert "is mounted by the sandbox itself" in problems[0] and fake.runs == []


# ---- the holder ----------------------------------------------------------------------------------------------------


def test_a_holder_whose_setup_failed_never_signals_the_process_its_pid_now_names(fake: FakeBwrap) -> None:
    # The stand-in names a pid 1 that is not its child, as bubblewrap's would be once reaped and the number reused,
    # then fails its setup (bash: dash cannot redirect to a descriptor above 9).
    decoy = subprocess.Popen([sys.executable, "-I", "-S", "-c", "import time; time.sleep(60)"])
    fake.path.write_text(
        "#!/bin/bash\n"
        'fd=""; prev=""\n'
        'for arg in "$@"; do [ "$prev" = --json-status-fd ] && fd="$arg"; prev="$arg"; done\n'
        f'echo "{{ \\"child-pid\\": {decoy.pid} }}" >&"$fd"\n'
        'echo "bwrap: Can\'t find source path /missing: No such file or directory" >&2\n'
        "exit 1\n"
    )
    try:
        with pytest.raises(BwrapSetupError, match="^bwrap: Can't find source path /missing"):
            Holder.start(probed(str(fake.path)), BwrapPolicy(read_only=()))
        with pytest.raises(subprocess.TimeoutExpired):
            decoy.wait(timeout=0.5)
    finally:
        decoy.kill()
        decoy.wait()
