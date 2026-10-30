# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Local sandbox provider: GPU pool, signing identity, reaping, timeouts, path safety."""

from __future__ import annotations

import json
import os
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

from hyperloom.common import proctree
from hyperloom.sandbox import local_sandbox
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
    monkeypatch.setattr(local_sandbox, "reap_sandbox_processes", lambda *a, **k: False)
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
