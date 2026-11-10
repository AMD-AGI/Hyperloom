# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Local sandbox provider: GPU pool, signing identity, reaping, timeouts, path safety."""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hyperloom_common import proctree
from hyperloom.sandbox import local_sandbox, reaper
from hyperloom.sandbox.card_occupancy import SANDBOX_TAG_ENV
from hyperloom.sandbox.gpu_pool import GpuPool
from hyperloom.sandbox.local import LocalSandboxProvider, LocalTemplate, ProcessUsers
from hyperloom.sandbox.socket import (
    IDEMPOTENCY_KEY,
    SIGNING_PUBLIC_KEY,
    CreateOptions,
    RunSettings,
    SandboxDenied,
    SandboxGone,
    SandboxUnavailable,
)

_ENV_DUMP = (
    "import json,os,time\n"
    "root=os.environ['HYPERLOOM_SANDBOX_ROOT']\n"
    "info=dict(os.environ); info['_cwd']=os.getcwd()\n"
    "json.dump(info, open(root+'/out/env.json.tmp','w'))\n"
    "os.rename(root+'/out/env.json.tmp', root+'/out/env.json')\n"
    "time.sleep(600)\n"
)

# Spawns a child (which spawns a grandchild) and a detached daemon, then records pids.
_FAMILY = (
    "import os,subprocess,sys,time\n"
    "kid=subprocess.Popen([sys.executable,'-c','import subprocess,sys,time;"
    'g=subprocess.Popen([sys.executable,"-c","import time;time.sleep(600)"]);'
    "print(g.pid,flush=True);time.sleep(600)'],stdout=subprocess.PIPE,text=True)\n"
    "grand=int(kid.stdout.readline())\n"
    "daemon=subprocess.Popen([sys.executable,'-c','import time;time.sleep(600)'],start_new_session=True)\n"
    "ready=os.environ.get('FAMILY_OUT','pids.json')\n"
    "open(ready+'.tmp','w').write('%d %d %d %d' % (os.getpid(),kid.pid,grand,daemon.pid))\n"
    "os.rename(ready+'.tmp',ready)\n"
    "time.sleep(600)\n"
)


def _wait(predicate, seconds: float = 20.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("condition not met in time")


def _read_when_present(sb, path: str) -> bytes:
    def _try():
        try:
            return sb.read_file(path)
        except FileNotFoundError:
            return None

    return _wait(_try)


def _start_family(sb) -> None:
    sb.write_file("/work/family.py", _FAMILY.encode())
    sb.run(f"{sys.executable} family.py", background=True)


def _family_pids(sb) -> list[int]:
    return [int(x) for x in _read_when_present(sb, "/work/pids.json").split()]


@pytest.fixture
def make():
    providers: list[LocalSandboxProvider] = []

    def _make(templates, **kw) -> LocalSandboxProvider:
        provider = LocalSandboxProvider(templates, hostname="node-a", **kw)
        providers.append(provider)
        return provider

    yield _make
    for provider in providers:
        provider.close()


def _runner(code: str, **kw) -> LocalTemplate:
    kw.setdefault("sealed", True)
    return LocalTemplate("runner", "measurement", command=(sys.executable, "-c", code), **kw)


def _create(provider, template: str, role: str, **kw):
    options = CreateOptions(envs=kw.pop("envs", None), wait_sec=kw.pop("wait_sec", 60))
    return provider.create(
        template, role=role, timeout_sec=kw.pop("timeout_sec", 120), metadata=kw.pop("metadata", {}), options=options
    )


# -- template validation -------------------------------------------------------


def test_template_validation_rejects_unsafe_combinations() -> None:
    with pytest.raises(ValueError, match="measurement template must be sealed"):
        LocalTemplate("t", "measurement", sealed=False)
    with pytest.raises(ValueError, match="only a sealed template may carry a signing identity"):
        LocalTemplate("t", "dev", signing=True)
    with pytest.raises(ValueError, match="platform variables"):
        LocalTemplate("t", "dev", envs={"HYPERLOOM_SIGNING_KEY_FILE": "/x"})
    with pytest.raises(ValueError, match="unknown role"):
        LocalTemplate("t", "root")  # type: ignore[arg-type]


def test_template_from_mapping_is_strict() -> None:
    tpl = LocalTemplate.from_mapping(
        "r", {"role": "measurement", "sealed": True, "signing": True, "gpus": 2, "command": ["x"]}
    )
    assert (tpl.gpus, tpl.command, tpl.signing) == (2, ("x",), True)
    with pytest.raises(ValueError, match="unknown keys"):
        LocalTemplate.from_mapping("r", {"role": "dev", "sudo": True})
    with pytest.raises(ValueError, match="role is required"):
        LocalTemplate.from_mapping("r", {})
    with pytest.raises(ValueError, match="argv list"):
        LocalTemplate.from_mapping("r", {"role": "dev", "command": "echo hi"})


# -- signing identity and the runner's environment ------------------------------


def test_runner_environment_and_signing_identity(make, monkeypatch) -> None:
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization

    monkeypatch.setenv("SOME_CONTROL_PLANE_TOKEN", "must-not-leak")
    provider = make([_runner(_ENV_DUMP, gpus=2, signing=True)], gpus=[4, 5, 6, 7])
    sb = _create(provider, "runner", "measurement")
    env = json.loads(_read_when_present(sb, "/out/env.json"))
    assert env["ROCR_VISIBLE_DEVICES"] == "4,5"
    assert env["NODE_NAME"] == "node-a"
    assert env["E2B_SANDBOX_ID"] == sb.sandbox_id
    assert "SOME_CONTROL_PLANE_TOKEN" not in env
    assert env["_cwd"] == os.path.join(env["HYPERLOOM_SANDBOX_ROOT"], "work")

    key_file = Path(env["HYPERLOOM_SIGNING_KEY_FILE"])
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(key_file.parent.stat().st_mode) == 0o700
    root = Path(env["HYPERLOOM_SANDBOX_ROOT"]).resolve()
    assert root not in key_file.resolve().parents  # outside what the file API maps

    private = serialization.load_pem_private_key(key_file.read_bytes(), password=None)
    public = serialization.load_pem_public_key(sb.info().metadata[SIGNING_PUBLIC_KEY].encode())
    public.verify(private.sign(b"payload"), b"payload")  # raises if the pair does not match
    assert "PRIVATE" not in json.dumps(dict(sb.info().metadata))


def test_a_symlink_cannot_lead_the_file_api_to_the_private_key(make, tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    code = (
        "import os,time\n"
        "root=os.environ['HYPERLOOM_SANDBOX_ROOT']\n"
        "os.symlink(os.environ['HYPERLOOM_SIGNING_KEY_FILE'], root+'/out/stolen')\n"
        "open(root+'/out/ready','w').write('1')\n"
        "time.sleep(600)\n"
    )
    provider = make([_runner(code, signing=True)])
    sb = _create(provider, "runner", "measurement")
    _read_when_present(sb, "/out/ready")
    with pytest.raises(SandboxDenied, match="resolves outside the sandbox"):
        sb.read_file("/out/stolen")
    with pytest.raises(SandboxDenied, match="resolves outside the sandbox"):
        sb.copy_out("/out/stolen", tmp_path / "x")


def test_a_dev_symlink_cannot_escape_the_root(make) -> None:
    provider = make([LocalTemplate("dev", "dev")])
    sb = _create(provider, "dev", "dev")
    sb.run("ln -s /etc/passwd pw")
    with pytest.raises(SandboxDenied, match="resolves outside the sandbox"):
        sb.read_file("/work/pw")
    sb.run("ln -s /tmp tmpdir")
    with pytest.raises(SandboxDenied, match="resolves outside the sandbox"):
        sb.write_file("/work/tmpdir/escape.txt", b"x")


@pytest.mark.parametrize("verb", ["read_file", "copy_out", "list_dir", "write_file"])
def test_a_path_swapped_for_a_link_out_after_its_gate_is_never_followed(
    make, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verb: str
) -> None:
    """A sandbox's process that puts a link to an app's file where the gated path was, in the moment between the gate
    and the open, gets a refusal: no name is resolved twice, so nothing of the app's is read, listed or written."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "credentials.json").write_text("SECRET")
    provider = make([LocalTemplate("dev", "dev")])
    sb = _create(provider, "dev", "dev")
    sb.run("mkdir -p out/files && echo harmless > out/files/a.log")
    real_gate = local_sandbox.LocalSandbox._gate

    files = sb.fs / "work" / "out" / "files"
    gated, target = (files, outside) if verb == "list_dir" else (files / "a.log", outside / "credentials.json")

    def swapped(self, path: str, *, write: bool):
        if gated.is_symlink():  # what any gate (a second one too) checks is the sandbox's own file
            gated.unlink()
            gated.mkdir() if verb == "list_dir" else gated.write_text("harmless\n")
        found = real_gate(self, path, write=write)
        (self.fs / "work" / "swap").symlink_to(target)
        if gated.is_dir():
            shutil.rmtree(gated)
        os.replace(self.fs / "work" / "swap", gated)
        return found

    monkeypatch.setattr(local_sandbox.LocalSandbox, "_gate", swapped)
    named = f"/work/{gated.relative_to(sb.fs / 'work')}"
    with pytest.raises(SandboxDenied, match="resolves outside the sandbox"):
        if verb == "copy_out":
            sb.copy_out(named, tmp_path / "saved.log")
        elif verb == "write_file":
            sb.write_file(named, b"chosen by the app")
        else:
            getattr(sb, verb)(named)
    assert (outside / "credentials.json").read_text() == "SECRET" and sorted(os.listdir(outside)) == [
        "credentials.json"
    ]
    assert not (tmp_path / "saved.log").exists() or b"SECRET" not in (tmp_path / "saved.log").read_bytes()


def test_a_pipe_in_the_tree_is_refused_without_blocking_and_links_inside_the_root_are_followed(make) -> None:
    provider = make([LocalTemplate("dev", "dev")])
    sb = _create(provider, "dev", "dev")
    sb.run(
        "mkfifo pipe && echo real > real.txt && mkdir dir && ln -s real.txt alias && ln -s dir via && "
        'ln -s "$HYPERLOOM_SANDBOX_ROOT/work/real.txt" absolute'
    )
    outcome: list[object] = []

    def read_pipe() -> None:
        try:
            outcome.append(sb.read_file("/work/pipe"))
        except SandboxDenied as exc:
            outcome.append(exc)

    reader = threading.Thread(target=read_pipe, daemon=True)
    reader.start()
    reader.join(10)
    if reader.is_alive():
        os.close(os.open(sb.fs / "work" / "pipe", os.O_WRONLY | os.O_NONBLOCK))  # lets the blocked read end
    assert [type(found) for found in outcome] == [SandboxDenied] and "not a regular file" in str(outcome[0])
    with pytest.raises((SandboxDenied, OSError)):  # no reader: ENXIO; a reader: not a regular file
        sb.write_file("/work/pipe", b"x")
    sb.write_file("/work/via/new/file.txt", b"through a link inside")
    assert sb.read_file("/work/alias") == b"real\n" and sb.read_file("/work/absolute") == b"real\n"
    assert sb.read_file("/work/dir/new/file.txt") == b"through a link inside"
    with pytest.raises(IsADirectoryError):
        sb.read_file("/work/dir")


def test_path_traversal_is_confined_to_the_root(make) -> None:
    provider = make([LocalTemplate("dev", "dev")])
    sb = _create(provider, "dev", "dev")
    sb.write_file("/work/../../../../escape.txt", b"x")  # collapses to /escape.txt inside the root
    assert "escape.txt" in sb.list_dir("/")
    assert not Path("/escape.txt").exists()


def test_platform_names_in_envs_are_refused(make) -> None:
    provider = make([LocalTemplate("dev", "dev")])
    with pytest.raises(SandboxDenied, match="platform variables"):
        _create(provider, "dev", "dev", envs={"HYPERLOOM_SIGNING_KEY_FILE": "/x"})
    sb = _create(provider, "dev", "dev")
    with pytest.raises(SandboxDenied, match="platform variables"):
        sb.run("true", settings=RunSettings(envs={"E2B_SANDBOX_ID": "spoof"}))


def test_template_envs_and_create_envs_reach_commands(make) -> None:
    provider = make([LocalTemplate("dev", "dev", envs={"A": "1", "B": "tpl"})])
    sb = _create(provider, "dev", "dev", envs={"B": "create"})
    assert (
        sb.run('echo "$A $B $C $NODE_NAME"', settings=RunSettings(envs={"C": "call"})).stdout
        == "1 create call node-a\n"
    )


# -- GPU pool ---------------------------------------------------------------------


def test_pool_leases_disjoint_cards_and_rejects_foreign_release() -> None:
    pool = GpuPool([3, 1, 2, 0])
    a, b = pool.acquire(2, 1), pool.acquire(2, 1)
    assert (a, b) == ((0, 1), (2, 3))
    with pytest.raises(ValueError, match="not leased"):
        pool.release([9])
    pool.release(a)
    with pytest.raises(ValueError, match="not leased"):
        pool.release(a)
    assert pool.free_cards() == (0, 1)
    with pytest.raises(ValueError, match="duplicate"):
        GpuPool([1, 1])


def test_eight_cards_at_tp2_allow_exactly_four_concurrent_sandboxes(make) -> None:
    provider = make([_runner(_ENV_DUMP, gpus=2)], gpus=range(8))
    boxes = [_create(provider, "runner", "measurement") for _ in range(4)]
    seen = [json.loads(_read_when_present(b, "/out/env.json"))["ROCR_VISIBLE_DEVICES"] for b in boxes]
    cards = [c for s in seen for c in s.split(",")]
    assert sorted(cards) == [str(i) for i in range(8)]  # disjoint, all used
    with pytest.raises(SandboxUnavailable, match="no 2 free GPUs"):
        provider.create("runner", role="measurement", timeout_sec=60, metadata={}, options=CreateOptions(wait_sec=0.2))


def test_create_waits_for_cards_then_succeeds_when_they_come_back(make) -> None:
    provider = make([_runner(_ENV_DUMP, gpus=2)], gpus=[0, 1])
    first = _create(provider, "runner", "measurement")
    got: list = []
    worker = threading.Thread(target=lambda: got.append(_create(provider, "runner", "measurement", wait_sec=30)))
    worker.start()
    time.sleep(0.5)
    assert not got  # still waiting: the cards are leased
    first.kill()
    worker.join(30)
    assert got and json.loads(_read_when_present(got[0], "/out/env.json"))["ROCR_VISIBLE_DEVICES"] == "0,1"


def test_create_gives_up_after_wait_sec(make) -> None:
    provider = make([_runner(_ENV_DUMP, gpus=1)], gpus=[0])
    _create(provider, "runner", "measurement")
    started = time.monotonic()
    with pytest.raises(SandboxUnavailable, match="no 1 free GPUs within 0.3s"):
        provider.create("runner", role="measurement", timeout_sec=60, metadata={}, options=CreateOptions(wait_sec=0.3))
    assert 0.25 <= time.monotonic() - started < 5


def test_a_template_larger_than_the_pool_fails_at_once(make) -> None:
    provider = make([_runner(_ENV_DUMP, gpus=3)], gpus=[0, 1])
    started = time.monotonic()
    with pytest.raises(SandboxUnavailable, match="pool owns 2"):
        provider.create("runner", role="measurement", timeout_sec=60, metadata={}, options=CreateOptions(wait_sec=30))
    assert time.monotonic() - started < 5


def test_cards_return_when_the_timeout_reaps_the_sandbox(make) -> None:
    provider = make([_runner(_ENV_DUMP, gpus=2)], gpus=[0, 1])
    _create(provider, "runner", "measurement", timeout_sec=1)
    again = _create(provider, "runner", "measurement", wait_sec=20)  # waits for the expiry
    assert json.loads(_read_when_present(again, "/out/env.json"))["ROCR_VISIBLE_DEVICES"] == "0,1"


def test_cards_stay_leased_when_reaping_cannot_be_confirmed(make, monkeypatch) -> None:
    provider = make([LocalTemplate("dev", "dev", gpus=1)], gpus=[0, 1])
    sb = _create(provider, "dev", "dev")
    monkeypatch.setattr(local_sandbox, "reap_sandbox_processes", lambda *a, **k: reaper.Leftovers(unseen=True))
    with pytest.raises(SandboxUnavailable, match="its GPUs stay leased"):
        sb.kill()
    assert provider.quarantined == {0}
    assert provider.pool.free_cards() == (1,)
    with pytest.raises(SandboxGone):
        sb.info()


def test_a_failed_create_returns_its_cards_and_repeats_its_error(make) -> None:
    broken = LocalTemplate("broken", "measurement", command=("/nonexistent/runner",), gpus=1, sealed=True)
    provider = make([broken], gpus=[0])
    meta = {IDEMPOTENCY_KEY: "attempt-1"}
    with pytest.raises(SandboxUnavailable, match="could not create sandbox") as first:
        _create(provider, "broken", "measurement", metadata=meta)
    assert provider.pool.free_cards() == (0,)
    with pytest.raises(SandboxUnavailable) as again:
        _create(provider, "broken", "measurement", metadata=meta)
    assert again.value is first.value
    assert provider.find({}, role="measurement") == []


# -- process-tree reaping ---------------------------------------------------------


def _assert_all_gone(pids: list[int]) -> None:
    alive = [p for p in pids if proctree.running(p)]
    assert not alive, f"processes survived: {alive}"


def test_kill_reaps_children_grandchildren_and_a_detached_daemon(make) -> None:
    provider = make([LocalTemplate("dev", "dev")])
    sb = _create(provider, "dev", "dev")
    _start_family(sb)
    pids = _family_pids(sb)
    assert len(pids) == 4 and all(proctree.running(p) for p in pids)  # leader, child, grandchild, daemon
    sb.kill()
    _assert_all_gone(pids)


def test_kill_reaps_the_sealed_runners_whole_tree(make) -> None:
    code = _FAMILY.replace(
        "os.environ.get('FAMILY_OUT','pids.json')", "os.environ['HYPERLOOM_SANDBOX_ROOT']+'/out/pids.json'"
    )
    provider = make([_runner(code)])
    sb = _create(provider, "runner", "measurement")
    pids = [int(x) for x in _read_when_present(sb, "/out/pids.json").split()]
    assert all(proctree.running(p) for p in pids)
    sb.kill()
    _assert_all_gone(pids)


def test_timeout_reaps_the_whole_tree(make) -> None:
    provider = make([LocalTemplate("dev", "dev")])
    sb = _create(provider, "dev", "dev", timeout_sec=2)
    _start_family(sb)
    pids = _family_pids(sb)
    _wait(lambda: not any(proctree.running(p) for p in pids), seconds=30)
    with pytest.raises(SandboxGone):
        sb.info()


def test_killing_one_sandbox_leaves_its_neighbour_running(make) -> None:
    provider = make([LocalTemplate("dev", "dev")])
    victim, bystander = _create(provider, "dev", "dev"), _create(provider, "dev", "dev")
    for sb in (victim, bystander):
        _start_family(sb)
    mine, theirs = _family_pids(victim), _family_pids(bystander)
    victim.kill()
    _assert_all_gone(mine)
    assert all(proctree.running(p) for p in theirs)


def test_a_command_timeout_kills_its_tree_and_raises(make) -> None:
    provider = make([LocalTemplate("dev", "dev")])
    sb = _create(provider, "dev", "dev")
    with pytest.raises(SandboxUnavailable, match="command timed out after 0.5s"):
        sb.run("sleep 300 & echo $! > pid; wait", settings=RunSettings(timeout_sec=0.5))
    pid = int(sb.read_file("/work/pid"))
    _wait(lambda: not proctree.running(pid), seconds=10)
    assert sb.info().sandbox_id  # the sandbox itself is still usable
    assert sb.run("echo ok").stdout == "ok\n"


def _swallow_signals(monkeypatch) -> list[tuple[str, int, int]]:
    """Record every signal ``os.kill`` and ``os.killpg`` would send and send none; a probe (signal 0) still runs.

    For a check of what a kill sends when nothing it may signal is left: a freed pid, or a process found under it,
    may already be another process, so nothing reaches it even on code that would signal it.
    """
    sent: list[tuple[str, int, int]] = []
    real = {"kill": os.kill, "killpg": os.killpg}

    def _spy(name):
        def _send(pid: int, sig: int) -> None:
            if sig == 0:
                real[name](pid, sig)
            else:
                sent.append((name, pid, sig))

        return _send

    monkeypatch.setattr(os, "kill", _spy("kill"))
    monkeypatch.setattr(os, "killpg", _spy("killpg"))
    return sent


def test_kill_after_a_command_timeout_never_signals_the_launcher_the_timeout_reaped(make, monkeypatch) -> None:
    provider = make([LocalTemplate("dev", "dev")])
    sb = _create(provider, "dev", "dev")
    timed_out: list[int] = []
    kill_launchers = local_sandbox.kill_launchers

    def _recording(procs, **kw):
        timed_out.extend(proc.pid for proc in procs)
        return kill_launchers(procs, **kw)

    monkeypatch.setattr(local_sandbox, "kill_launchers", _recording)
    with pytest.raises(SandboxUnavailable, match="command timed out"):
        sb.run("sleep 30", settings=RunSettings(timeout_sec=0.5))
    assert len(timed_out) == 1
    handed: list[int] = []
    reap = local_sandbox.reap_sandbox_processes

    def _reap(procs, sandbox_id):
        handed.extend(proc.pid for proc in procs)
        return reap(procs, sandbox_id)

    monkeypatch.setattr(local_sandbox, "reap_sandbox_processes", _reap)
    signalled = _swallow_signals(monkeypatch)
    sb.kill()
    assert timed_out[0] not in handed
    assert signalled == []


def _ends_nothing(procs, **kw) -> reaper.Leftovers:
    """A timeout's reap stuck past its windows: it ends nothing and confirms nothing."""
    return reaper.Leftovers(unseen=True)


def _ends_the_launcher_unconfirmed(procs, **kw) -> reaper.Leftovers:
    """A timeout's reap that waits the launcher but cannot confirm its tree, as when a descendant outlives SIGKILL."""
    reaper.kill_launchers(procs, **kw)
    return reaper.Leftovers(unseen=True)


@pytest.mark.parametrize("timeout_reap", [_ends_nothing, _ends_the_launcher_unconfirmed])
def test_a_command_timeout_whose_reap_is_unconfirmed_keeps_the_cards_leased(make, monkeypatch, timeout_reap) -> None:
    provider = make([LocalTemplate("dev", "dev", gpus=1)], gpus=[0, 1])
    sb = _create(provider, "dev", "dev")
    monkeypatch.setattr(local_sandbox, "kill_launchers", timeout_reap)
    with pytest.raises(SandboxUnavailable, match="command timed out"):
        # The job drops the sandbox's tag, so only the launcher's tree leads kill() to it.
        sb.run(f"env -u {SANDBOX_TAG_ENV} sleep 300 & echo $! > pid; wait", settings=RunSettings(timeout_sec=0.5))
    pid = int(_read_when_present(sb, "/work/pid"))
    job = proctree.proc_identity(pid)
    try:
        with pytest.raises(SandboxUnavailable, match="its GPUs stay leased"):
            sb.kill()
        _assert_all_gone([pid])
        assert provider.quarantined == {0}
        assert provider.pool.free_cards() == (1,)
    finally:
        if job is not None:
            proctree.signal_processes([(pid, job[1])], signal.SIGKILL)


def test_kill_launchers_skips_a_launcher_that_was_already_waited(monkeypatch) -> None:
    launcher = subprocess.Popen([sys.executable, "-c", "pass"])
    launcher.wait()
    signalled = _swallow_signals(monkeypatch)
    assert reaper.kill_launchers([launcher], grace_sec=0.1) == reaper.Leftovers()
    assert signalled == []


def test_a_sweep_that_cannot_list_proc_is_not_clean(monkeypatch) -> None:
    iterdir = Path.iterdir

    def _iterdir(path: Path):
        if str(path) == "/proc":
            raise PermissionError(13, "Permission denied", "/proc")
        return iterdir(path)

    monkeypatch.setattr(Path, "iterdir", _iterdir)
    assert reaper.sweep_tagged("sandbox-that-never-ran") == reaper.Leftovers(unseen=True)


def test_a_sweep_kills_a_tagged_process_outside_any_launcher(monkeypatch) -> None:
    monkeypatch.setenv("E2B_SANDBOX_ID", "sweep-me")
    tagged = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(600)"])
    try:
        _wait(lambda: proctree.marked_pids("E2B_SANDBOX_ID", "sweep-me"))
        assert reaper.sweep_tagged("sweep-me") == reaper.Leftovers()
        assert tagged.wait(timeout=10) == -9
    finally:
        tagged.kill()
        tagged.wait()


# -- exited processes, survivors and zombies ------------------------------------------

#: Forks a child that exits at once and is never waited, so it stays a zombie carrying the sandbox's tag.
_LEAVES_A_ZOMBIE = (
    "import os,time\n"
    "child=os.fork()\n"
    "if child==0:\n"
    "    os._exit(0)\n"
    "open('zombie.tmp','w').write(str(child))\n"
    "os.rename('zombie.tmp','zombie')\n"
    "time.sleep(600)\n"
)

#: Leaves the launcher's process group and session, so only the tree walk reaches it once it has dropped the tag.
_DETACHES = (
    "import os,time\n"
    "os.setsid()\n"
    "open('pid.tmp','w').write(str(os.getpid()))\n"
    "os.rename('pid.tmp','pid')\n"
    "time.sleep(600)\n"
)

#: Kills a sandbox while it is the child subreaper of everything it starts, as a session is, and prints the zombies it
#: is left with.
_KILL_AS_SUBREAPER = r"""
import ctypes, json, os, sys, time
from pathlib import Path
from hyperloom.sandbox.local import LocalSandboxProvider, LocalTemplate

def zombie_children():
    me, found = os.getpid(), []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[-1].split()
        except OSError:
            continue
        if fields[0] == "Z" and int(fields[1]) == me:
            found.append(int(entry.name))
    return found

assert ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0  # PR_SET_CHILD_SUBREAPER
family, base = sys.argv[1], Path(sys.argv[2])
with LocalSandboxProvider([LocalTemplate("dev", "dev")], hostname="node-a", base_dir=base) as provider:
    sandbox = provider.create("dev", role="dev", timeout_sec=120, metadata={})
    sandbox.write_file("/work/family.py", family.encode())
    sandbox.run(f"{sys.executable} family.py", background=True)
    deadline = time.monotonic() + 30
    while True:
        try:
            pids = [int(pid) for pid in sandbox.read_file("/work/pids.json").split()]
            break
        except FileNotFoundError:
            assert time.monotonic() < deadline, "the family did not start"
            time.sleep(0.05)
    sandbox.kill()
    print(json.dumps({"family": pids, "zombies": zombie_children()}))
"""


def _state(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[-1].split()[0]
    except OSError:
        return ""


def _immortal(monkeypatch, pid: int) -> None:
    """``pid`` takes every signal and does not die, as a process does while the kernel still tears down its memory."""
    real_kill, real_killpg = os.kill, os.killpg

    def _kill(target: int, sig: int) -> None:
        if target != pid or sig == 0:
            real_kill(target, sig)

    def _killpg(pgid: int, sig: int) -> None:
        if pgid != pid or sig == 0:
            real_killpg(pgid, sig)

    monkeypatch.setattr(os, "kill", _kill)
    monkeypatch.setattr(os, "killpg", _killpg)


def _detached_immortal(sb, monkeypatch, *, tagged: bool = False) -> proctree.ProcessId:
    """Start a process in ``sb`` that leaves its session, and the sandbox's tag unless ``tagged``; make it immortal."""
    sb.write_file("/work/detach.py", _DETACHES.encode())
    keeps = "" if tagged else f"env -u {SANDBOX_TAG_ENV} "
    sb.run(f"{keeps}{sys.executable} detach.py", background=True)
    pid = int(_read_when_present(sb, "/work/pid"))
    identity = proctree.proc_identity(pid)
    assert identity is not None
    _immortal(monkeypatch, pid)
    return pid, identity[1]


def _mortal(monkeypatch) -> None:
    """Give the real signals back: the process the test made immortal dies of the next SIGKILL sent to it."""
    monkeypatch.setattr(os, "kill", _REAL_KILL)
    monkeypatch.setattr(os, "killpg", _REAL_KILLPG)


def _end(monkeypatch, process: proctree.ProcessId) -> None:
    """Give the real signals back and kill ``process``, the one the test made immortal."""
    _mortal(monkeypatch)
    proctree.signal_processes([process], signal.SIGKILL)
    _wait(lambda: not proctree.running(process[0]), seconds=10)


_REAL_KILL, _REAL_KILLPG = os.kill, os.killpg


def test_a_zombie_with_the_sandbox_tag_does_not_keep_the_cards(make) -> None:
    provider = make([LocalTemplate("dev", "dev", gpus=1)], gpus=[0])
    sb = _create(provider, "dev", "dev")
    sb.write_file("/work/zombie.py", _LEAVES_A_ZOMBIE.encode())
    sb.run(f"{sys.executable} zombie.py", background=True)
    zombie = int(_read_when_present(sb, "/work/zombie"))
    _wait(lambda: _state(zombie) == "Z")
    sb.kill()
    assert provider.quarantined == set()
    assert provider.pool.free_cards() == (0,)


def test_a_tagged_zombie_outside_any_launcher_is_not_waited_out(monkeypatch) -> None:
    monkeypatch.setenv(SANDBOX_TAG_ENV, "a-zombie")
    zombie = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        _wait(lambda: _state(zombie.pid) == "Z")
        started = time.monotonic()
        assert reaper.reap_sandbox_processes([], "a-zombie") == reaper.Leftovers()
        assert time.monotonic() - started < reaper._WINDOW_SEC, "a zombie is gone, not a process to wait out"
    finally:
        zombie.wait()


def test_a_live_process_that_dropped_the_tag_keeps_the_cards_and_is_named(make, monkeypatch) -> None:
    monkeypatch.setattr(reaper, "_WINDOW_SEC", 0.5)
    provider = make([LocalTemplate("dev", "dev", gpus=1)], gpus=[0, 1])
    sb = _create(provider, "dev", "dev")
    survivor = _detached_immortal(sb, monkeypatch)
    try:
        named = rf"outlived SIGKILL \(pid {survivor[0]} \(.*, state S\)\); its GPUs stay"
        with pytest.raises(SandboxUnavailable, match=named):
            sb.kill()
        assert proctree.running(survivor[0]), "the test's survivor is still there"
        assert provider.quarantined == {0}
        assert provider.pool.free_cards() == (1,)
    finally:
        _end(monkeypatch, survivor)


def test_a_live_tagged_process_outside_any_launcher_is_left_and_named(monkeypatch) -> None:
    monkeypatch.setattr(reaper, "_WINDOW_SEC", 0.3)
    monkeypatch.setenv(SANDBOX_TAG_ENV, "a-survivor")
    survivor = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    try:
        _wait(lambda: proctree.marked_pids(SANDBOX_TAG_ENV, "a-survivor"))
        identity = proctree.proc_identity(survivor.pid)
        assert identity is not None
        _immortal(monkeypatch, survivor.pid)
        left = reaper.reap_sandbox_processes([], "a-survivor")
        assert left == reaper.Leftovers(((survivor.pid, identity[1]),))
        assert reaper.wait_exited(left) == left, "a survivor still running is still left"
        _end(monkeypatch, (survivor.pid, identity[1]))
        assert reaper.wait_exited(left) == reaper.Leftovers()
    finally:
        survivor.kill()
        survivor.wait()


def test_a_command_timeout_survivor_that_has_exited_by_the_kill_does_not_keep_the_cards(make, monkeypatch) -> None:
    provider = make([LocalTemplate("dev", "dev", gpus=1)], gpus=[0, 1])
    sb = _create(provider, "dev", "dev")
    survivor = subprocess.Popen([sys.executable, "-c", "pass"])
    survivor.wait()
    gone = (survivor.pid, 1)

    def _left_one_behind(procs, **kw) -> reaper.Leftovers:
        reaper.kill_launchers(procs, **kw)
        return reaper.Leftovers((gone,))

    monkeypatch.setattr(local_sandbox, "kill_launchers", _left_one_behind)
    with pytest.raises(SandboxUnavailable, match="command timed out"):
        sb.run("sleep 30", settings=RunSettings(timeout_sec=0.5))
    sb.kill()
    assert provider.quarantined == set()
    assert provider.pool.free_cards() == (0, 1)


@pytest.mark.parametrize("dies_in_the_sweep", [True, False], ids=["dies-in-the-sweep", "outlives-the-sweep"])
def test_a_survivor_the_launchers_reap_names_is_looked_at_again_after_the_tag_sweep(
    make, monkeypatch, dies_in_the_sweep: bool
) -> None:
    monkeypatch.setattr(reaper, "_WINDOW_SEC", 0.5)
    provider = make([LocalTemplate("dev", "dev", gpus=1)], gpus=[0, 1])
    sb = _create(provider, "dev", "dev")
    survivor = _detached_immortal(sb, monkeypatch, tagged=True)
    named: list[reaper.Leftovers] = []
    kill_launchers, sweep_tagged = reaper.kill_launchers, reaper.sweep_tagged

    def _kill_launchers(procs, **kw) -> reaper.Leftovers:
        named.append(kill_launchers(procs, **kw))
        return named[-1]

    def _sweep(sandbox_id: str) -> reaper.Leftovers:
        if dies_in_the_sweep:
            _mortal(monkeypatch)  # the sweep's SIGKILL now ends it
        return sweep_tagged(sandbox_id)

    monkeypatch.setattr(reaper, "kill_launchers", _kill_launchers)
    monkeypatch.setattr(reaper, "sweep_tagged", _sweep)
    try:
        if dies_in_the_sweep:
            sb.kill()
            assert not proctree.running(survivor[0])
            assert provider.quarantined == set()
            assert provider.pool.free_cards() == (0, 1)
        else:
            with pytest.raises(SandboxUnavailable, match=rf"outlived SIGKILL \(pid {survivor[0]} \(.*, state S\)\)"):
                sb.kill()
            assert provider.quarantined == {0}
            assert provider.pool.free_cards() == (1,)
        assert named == [reaper.Leftovers((survivor,))], "the launchers' reap named the survivor before the sweep"
    finally:
        _end(monkeypatch, survivor)


def test_a_launcher_that_outlives_its_wait_is_collected_by_its_popen_alone(make, monkeypatch) -> None:
    """A launcher named because its wait timed out is this process's own child: once it exits its ``Popen`` collects it,
    with the status SIGKILL left, and the wait for the zombies of what a reap named never takes it."""
    monkeypatch.setattr(reaper, "_WINDOW_SEC", 0.5)
    provider = make([LocalTemplate("dev", "dev", gpus=1)], gpus=[0, 1])
    sb = _create(provider, "dev", "dev")
    sb.run("sleep 600", background=True)
    launchers: list[proctree.ProcessId] = []
    popens: list[subprocess.Popen[bytes]] = []
    named: list[reaper.Leftovers] = []
    waited: list[proctree.ProcessId] = []
    kill_launchers, sweep_tagged = reaper.kill_launchers, reaper.sweep_tagged
    reap_exited_children = proctree.reap_exited_children

    def _kill_launchers(procs, **kw) -> reaper.Leftovers:
        popens.extend(procs)
        for proc in procs:
            identity = proctree.proc_identity(proc.pid)
            assert identity is not None
            launchers.append((proc.pid, identity[1]))
        # Its wait times out, as while the kernel still tears it down; stopped, it cannot end once its tree has.
        os.kill(procs[0].pid, signal.SIGSTOP)
        _immortal(monkeypatch, procs[0].pid)
        named.append(kill_launchers(procs, **kw))
        return named[-1]

    def _sweep(sandbox_id: str) -> reaper.Leftovers:
        _mortal(monkeypatch)  # the launcher carries the tag, and the sweep's SIGKILL now ends it
        return sweep_tagged(sandbox_id)

    def _reap_exited_children(processes) -> int:
        processes = list(processes)
        waited.extend(processes)
        return reap_exited_children(processes)

    monkeypatch.setattr(reaper, "kill_launchers", _kill_launchers)
    monkeypatch.setattr(reaper, "sweep_tagged", _sweep)
    monkeypatch.setattr(proctree, "reap_exited_children", _reap_exited_children)
    try:
        sb.kill()
        [launcher] = popens
        assert [left.running for left in named] == [tuple(launchers)], "the launchers' reap named it"
        assert launcher.returncode == -signal.SIGKILL, "its Popen collected it, with its own exit status"
        assert launcher.pid not in {pid for pid, _ in waited}
        assert provider.quarantined == set()
        assert provider.pool.free_cards() == (0, 1)
    finally:
        _mortal(monkeypatch)
        for proc in popens:
            proc.kill()
            proc.wait()


def test_a_named_survivor_that_dies_is_waited_when_it_is_this_process_s_child(make, monkeypatch) -> None:
    """A survivor re-parented to this process (a child subreaper) leaves no zombie once it dies after it was named."""
    monkeypatch.setattr(reaper, "_WINDOW_SEC", 0.5)
    provider = make([LocalTemplate("dev", "dev", gpus=1)], gpus=[0])
    sb = _create(provider, "dev", "dev")
    # Spawned without a Popen, so nothing but the reap waits for it, as for an orphan this process adopted.
    pid = os.posix_spawn(
        sys.executable,
        [sys.executable, "-c", "import time; time.sleep(600)"],
        {**os.environ, SANDBOX_TAG_ENV: sb.sandbox_id},
    )
    try:
        _wait(lambda: pid in proctree.marked_pids(SANDBOX_TAG_ENV, sb.sandbox_id))
        identity = proctree.proc_identity(pid)
        assert identity is not None
        survivor = (pid, identity[1])
        _immortal(monkeypatch, pid)
        sweep_tagged = reaper.sweep_tagged
        named: list[reaper.Leftovers] = []

        def _sweep_then_die(sandbox_id: str) -> reaper.Leftovers:
            named.append(sweep_tagged(sandbox_id))
            _mortal(monkeypatch)
            proctree.signal_processes([survivor], signal.SIGKILL)
            _wait(lambda: _state(pid) == "Z")
            return named[-1]

        monkeypatch.setattr(reaper, "sweep_tagged", _sweep_then_die)
        sb.kill()
        assert named == [reaper.Leftovers((survivor,))], "the sweep named it while it still ran"
        assert provider.quarantined == set()
        assert _state(pid) == "", "its zombie was waited"
    finally:
        _mortal(monkeypatch)
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
        with contextlib.suppress(ChildProcessError):
            os.waitpid(pid, 0)


@pytest.mark.parametrize("reused", [True, False], ids=["pid-reused", "same-process"])
def test_a_named_pid_now_held_by_another_process_is_not_the_survivor(make, monkeypatch, reused: bool) -> None:
    provider = make([LocalTemplate("dev", "dev", gpus=1)], gpus=[0, 1])
    sb = _create(provider, "dev", "dev")
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    try:
        identity = _wait(lambda: proctree.proc_identity(holder.pid))
        # Reused: the survivor the sweep named under this pid started a tick earlier, died, and the pid went to holder.
        named = (holder.pid, identity[1] - 1 if reused else identity[1])
        sweep_tagged = reaper.sweep_tagged
        monkeypatch.setattr(
            reaper, "sweep_tagged", lambda sandbox_id: sweep_tagged(sandbox_id) | reaper.Leftovers((named,))
        )
        if reused:
            sb.kill()
            assert provider.quarantined == set()
            assert provider.pool.free_cards() == (0, 1)
        else:
            with pytest.raises(SandboxUnavailable, match=rf"outlived SIGKILL \(pid {holder.pid} "):
                sb.kill()
            assert provider.quarantined == {0}
        assert holder.poll() is None, "the reap leaves whatever runs under the pid alone"
    finally:
        holder.kill()
        holder.wait()


def test_a_process_first_found_as_the_window_ends_is_signalled_before_it_is_named(monkeypatch) -> None:
    """Every process a reap names was sent the signal first: after SIGKILL none of them can fork another."""
    sent: list[proctree.ProcessId] = []

    def _send(processes, sig):
        sent.extend(processes)
        return list(processes)

    monkeypatch.setattr(proctree, "signal_processes", _send)
    window = 0.2
    late = time.monotonic() + window

    def _find() -> list[proctree.ProcessId]:
        return [(10, 1), (11, 1)] if time.monotonic() >= late else [(10, 1)]

    left = reaper._signal_until_gone(_find, signal.SIGKILL, window)
    assert left == reaper.Leftovers(((10, 1), (11, 1)))
    assert (11, 1) in sent


def test_a_tree_that_keeps_forking_past_the_window_is_unseen(monkeypatch) -> None:
    monkeypatch.setattr(proctree, "signal_processes", lambda processes, sig: list(processes))
    fresh = itertools.count(100)
    left = reaper._signal_until_gone(lambda: [(next(fresh), 1)], signal.SIGKILL, 0.1)
    assert left.unseen and not left.clean


def test_withheld_cards_come_back_once_what_outlived_the_reap_is_gone(make, monkeypatch) -> None:
    monkeypatch.setattr(reaper, "_WINDOW_SEC", 0.5)
    provider = make([LocalTemplate("dev", "dev", gpus=1)], gpus=[0, 1])
    sb = _create(provider, "dev", "dev")
    survivor = _detached_immortal(sb, monkeypatch)
    try:
        with pytest.raises(SandboxUnavailable, match="its GPUs stay leased"):
            sb.kill()
        assert provider.release_freed_cards() == (), "a process that dropped the tag still holds the cards"
        assert provider.quarantined == {0}
    finally:
        _end(monkeypatch, survivor)
    assert provider.release_freed_cards() == (0,)
    assert provider.quarantined == set()
    assert provider.pool.free_cards() == (0, 1)
    assert provider.release_freed_cards() == (), "cards come back once"


def test_withheld_cards_of_a_launcher_that_outlived_its_wait_come_back_through_its_popen(make, monkeypatch) -> None:
    """A launcher still there after the kill keeps the cards; once it exits they come back, and its ``Popen`` collects
    it with the status SIGKILL left, never the wait for the zombies of what a reap named."""
    monkeypatch.setattr(reaper, "_WINDOW_SEC", 0.5)
    provider = make([LocalTemplate("dev", "dev", gpus=1)], gpus=[0, 1])
    sb = _create(provider, "dev", "dev")
    sb.run("sleep 600", background=True)
    popens: list[subprocess.Popen[bytes]] = []
    waited: list[proctree.ProcessId] = []
    kill_launchers, reap_exited_children = reaper.kill_launchers, proctree.reap_exited_children

    def _kill_launchers(procs, **kw) -> reaper.Leftovers:
        popens.extend(procs)
        # It outlives its wait and the sweep, as while the kernel tears it down; stopped, it cannot end once its tree has.
        os.kill(procs[0].pid, signal.SIGSTOP)
        _immortal(monkeypatch, procs[0].pid)
        return kill_launchers(procs, **kw)

    def _reap_exited_children(processes) -> int:
        processes = list(processes)
        waited.extend(processes)
        return reap_exited_children(processes)

    monkeypatch.setattr(reaper, "kill_launchers", _kill_launchers)
    monkeypatch.setattr(proctree, "reap_exited_children", _reap_exited_children)
    try:
        with pytest.raises(SandboxUnavailable, match="its GPUs stay leased"):
            sb.kill()
        [launcher] = popens
        assert provider.release_freed_cards() == (), "the launcher still runs"
        _mortal(monkeypatch)
        os.kill(launcher.pid, signal.SIGKILL)
        _wait(lambda: _state(launcher.pid) == "Z")
        assert provider.release_freed_cards() == (0,)
        assert launcher.returncode == -signal.SIGKILL, "its Popen collected it, with its own exit status"
        assert launcher.pid not in {pid for pid, _ in waited}
        assert provider.pool.free_cards() == (0, 1)
    finally:
        _mortal(monkeypatch)
        for proc in popens:
            proc.kill()
            proc.wait()


def test_withheld_cards_stay_while_a_process_with_the_sandbox_tag_runs(make, monkeypatch) -> None:
    provider = make([LocalTemplate("dev", "dev", gpus=1)], gpus=[0])
    sb = _create(provider, "dev", "dev")
    exited = subprocess.Popen([sys.executable, "-c", "pass"])
    exited.wait()
    monkeypatch.setattr(local_sandbox, "reap_sandbox_processes", lambda *a, **k: reaper.Leftovers(((exited.pid, 1),)))
    with pytest.raises(SandboxUnavailable, match="its GPUs stay leased"):
        sb.kill()
    tagged = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(600)"], env={**os.environ, SANDBOX_TAG_ENV: sb.sandbox_id}
    )
    try:
        _wait(lambda: proctree.marked_pids(SANDBOX_TAG_ENV, sb.sandbox_id))
        assert provider.release_freed_cards() == ()
    finally:
        tagged.kill()
        tagged.wait()
    assert provider.release_freed_cards() == (0,)


def test_cards_a_reap_could_not_look_after_are_never_given_back(make, monkeypatch) -> None:
    provider = make([LocalTemplate("dev", "dev", gpus=1)], gpus=[0])
    sb = _create(provider, "dev", "dev")
    monkeypatch.setattr(local_sandbox, "reap_sandbox_processes", lambda *a, **k: reaper.Leftovers(unseen=True))
    with pytest.raises(SandboxUnavailable, match="what else is left is unknown"):
        sb.kill()
    assert provider.release_freed_cards() == ()
    assert provider.quarantined == {0}
    assert provider.pool.free_cards() == ()


def test_a_kill_in_a_child_subreaper_leaves_no_zombie(tmp_path: Path) -> None:
    script = tmp_path / "kill_as_subreaper.py"
    script.write_text(_KILL_AS_SUBREAPER)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    done = subprocess.run(
        [sys.executable, str(script), _FAMILY, str(tmp_path / "sandboxes")],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert len(report["family"]) == 4
    assert report["zombies"] == [], "every launcher and every process the sandbox orphaned is waited"


# -- lifecycle ---------------------------------------------------------------------


def test_kill_removes_the_sandbox_directory_and_the_key(make) -> None:
    pytest.importorskip("cryptography")
    provider = make([_runner(_ENV_DUMP, signing=True)])
    sb = _create(provider, "runner", "measurement")
    key_file = Path(json.loads(_read_when_present(sb, "/out/env.json"))["HYPERLOOM_SIGNING_KEY_FILE"])
    assert key_file.exists()
    sb.kill()
    assert not key_file.exists()
    assert not key_file.parent.parent.exists()


def test_close_kills_every_sandbox_and_removes_the_base_dir() -> None:
    provider = LocalSandboxProvider([LocalTemplate("dev", "dev")])
    boxes = [_create(provider, "dev", "dev") for _ in range(2)]
    base = boxes[0].directory.parent
    provider.close()
    assert all(b.ended for b in boxes)
    assert not base.exists()


def test_no_launcher_outlives_the_close_of_its_provider() -> None:
    provider = LocalSandboxProvider([LocalTemplate("dev", "dev")])
    sb = _create(provider, "dev", "dev")
    sb.run("sleep 300", background=True)
    assert sb.run("true").exit_code == 0
    launchers = [proc.pid for proc in sb._procs]
    assert len(launchers) == 2
    provider.close()
    _assert_all_gone(launchers)


def test_a_second_kill_waits_for_the_first_to_finish(make) -> None:
    provider = make([LocalTemplate("dev", "dev", gpus=1)], gpus=[0])
    sb = _create(provider, "dev", "dev")
    sb.run("sleep 300", background=True)
    t = threading.Thread(target=sb.kill)
    t.start()
    sb.kill()
    assert provider.pool.free_cards() == (0,)  # by the time either returns, the cards are back
    t.join()


def test_a_command_killed_mid_flight_reports_the_sandbox_gone(make) -> None:
    provider = make([LocalTemplate("dev", "dev")])
    sb = _create(provider, "dev", "dev")
    threading.Timer(0.5, sb.kill).start()
    with pytest.raises(SandboxGone):
        sb.run("sleep 300")


def test_kill_reaps_an_env_stripped_daemon_that_outlived_its_parent_and_left_the_tree(make) -> None:
    provider = make([LocalTemplate("dev", "dev")])
    sb = _create(provider, "dev", "dev")
    sb.write_file(
        "/work/orphan.py",
        (
            "import subprocess,sys\n"
            "d=subprocess.Popen([sys.executable,'-c','import time;time.sleep(600)'],start_new_session=True,env={})\n"
            "open('orphan.pid','w').write(str(d.pid))\n"
        ).encode(),
    )
    sb.run(f"{sys.executable} orphan.py")  # the parent exits; the daemon is re-parented away
    pid = int(sb.read_file("/work/orphan.pid"))
    assert proctree.running(pid)
    assert proctree.proc_identity(pid)[0] != os.getpid()
    sb.kill()
    _assert_all_gone([pid])


def test_a_foreground_command_that_leaves_a_background_job_returns_and_the_job_is_reaped(make) -> None:
    provider = make([LocalTemplate("dev", "dev")])
    sb = _create(provider, "dev", "dev")
    started = time.monotonic()
    assert sb.run("sleep 300 & echo $! > pid").exit_code == 0
    assert time.monotonic() - started < 10
    pid = int(sb.read_file("/work/pid"))
    assert proctree.running(pid)
    sb.kill()
    _assert_all_gone([pid])


def _marked_pids(marker: str) -> list[int]:
    found = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                if marker.encode() in (entry / "cmdline").read_bytes() and proctree.running(int(entry.name)):
                    found.append(int(entry.name))
            except OSError:
                continue
    return found


def test_killing_right_after_create_leaves_no_runner_behind(make) -> None:
    import uuid

    for _ in range(5):
        marker = f"marker-{uuid.uuid4().hex}"
        code = f"import time  # {marker}\ntime.sleep(600)\n"
        # env -i drops the sandbox tag, so only the process tree can find the runner.
        template = LocalTemplate(
            "runner", "measurement", command=("/usr/bin/env", "-i", sys.executable, "-c", code), sealed=True
        )
        provider = make([template])
        sb = _create(provider, "runner", "measurement")
        started = time.monotonic()
        sb.kill()  # the launcher may not have forked the runner yet
        # A runner forked after the scan would sit out the whole SIGTERM grace (2s) unseen.
        assert time.monotonic() - started < 1.8
        assert _marked_pids(marker) == []


_ROOT_ONLY = pytest.mark.skipif(os.geteuid() != 0, reason="separate accounts need root")
_SYSTEM_PYTHON = "/usr/bin/python3"


@_ROOT_ONLY
@pytest.mark.skipif(not os.access(_SYSTEM_PYTHON, os.X_OK), reason="needs a python3 every account can run")
def test_separate_accounts_keep_the_signing_key_from_other_sandboxes(make) -> None:
    pytest.importorskip("cryptography")
    runner_code = "head -c 10 $HYPERLOOM_SIGNING_KEY_FILE > $HYPERLOOM_SANDBOX_ROOT/out/keyhead; id -u > $HYPERLOOM_SANDBOX_ROOT/out/uid; sleep 600"
    templates = [
        LocalTemplate("runner", "measurement", command=("/bin/sh", "-c", runner_code), sealed=True, signing=True),
        LocalTemplate("dev", "dev"),
    ]
    users = ProcessUsers(_SYSTEM_PYTHON, sealed_user=(60001, 60001), unsealed_user=(60002, 60002))
    provider = make(templates, users=users)
    runner = _create(provider, "runner", "measurement")
    assert _read_when_present(runner, "/out/keyhead") == b"-----BEGIN"  # the runner reads its own key
    assert _read_when_present(runner, "/out/uid").strip() == b"60001"
    key_file = next((provider._base / runner.sandbox_id / "private").glob("*.pem"))
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600 and key_file.stat().st_uid == 60001

    dev = _create(provider, "dev", "dev")
    assert dev.run("id -u").stdout.strip() == "60002"
    stolen = dev.run(f"cat {key_file}")
    assert stolen.exit_code != 0 and "Permission denied" in stolen.stderr
    assert dev.run(f"ls {key_file.parent}").exit_code != 0
    runner.kill()
    dev.kill()


def test_separate_accounts_need_root(make, monkeypatch) -> None:
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    with pytest.raises(ValueError, match="needs root"):
        LocalSandboxProvider([LocalTemplate("dev", "dev")], users=ProcessUsers(unsealed_user=(1, 1)))


def test_idempotency_key_is_released_when_create_fails_unexpectedly(make) -> None:
    provider = make([LocalTemplate("dev", "dev")])
    meta = {IDEMPOTENCY_KEY: "k"}
    with pytest.raises(ValueError, match="timeout_sec must be positive"):
        provider.create("dev", role="dev", timeout_sec=0, metadata=meta)
    assert _create(provider, "dev", "dev", metadata=meta, wait_sec=1).info().template == "dev"


def test_node_name_prefers_node_name_env_then_hostname(monkeypatch) -> None:
    import socket

    from hyperloom.sandbox.node import node_name

    monkeypatch.setenv("NODE_NAME", "k8s-node-7")
    assert node_name() == "k8s-node-7"
    monkeypatch.setenv("NODE_NAME", "")
    assert node_name() == socket.gethostname()
    monkeypatch.delenv("NODE_NAME")
    assert node_name() == socket.gethostname()


def test_provider_reports_the_shared_node_name(monkeypatch) -> None:
    monkeypatch.setenv("NODE_NAME", "k8s-node-7")
    with LocalSandboxProvider([LocalTemplate("dev", "dev")]) as provider:
        info = _create(provider, "dev", "dev").info()
        assert info.metadata["agent-sandbox.io/node-name"] == "k8s-node-7"
        assert _create(provider, "dev", "dev").run("echo $NODE_NAME").stdout == "k8s-node-7\n"
    monkeypatch.delenv("NODE_NAME")
    import socket

    with LocalSandboxProvider([LocalTemplate("dev", "dev")]) as provider:
        assert _create(provider, "dev", "dev").info().metadata["agent-sandbox.io/node-name"] == socket.gethostname()
