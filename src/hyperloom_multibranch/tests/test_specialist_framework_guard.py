# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from hyperloom.measure.compile import PackageBase, UnsupportedComponentError
from hyperloom_multibranch.contracts.toolbox import BranchLease
from hyperloom_multibranch.specialist import agent_sandbox, framework_guard
from hyperloom_multibranch.specialist.agent_sandbox import (
    GUARDED_NOTE,
    FrameworkProtection,
    LocalAgentSandbox,
    framework_protection,
)
from hyperloom_multibranch.specialist.agent_sessions import LocalAgentSandboxes
from hyperloom_multibranch.specialist.framework_guard import FrameworkGuard, FrameworksUnguarded, GuardedPackage

ORIGINAL = "def forward(x):\n    return x\n"


@pytest.fixture(autouse=True)
def _local_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "local-tmp"))
    (tmp_path / "local-tmp").mkdir()
    framework_protection.cache_clear()
    yield
    framework_protection.cache_clear()


def _vllm(tmp_path: Path) -> Path:
    """A fake installed vllm, with a bytecode cache Python wrote beside it."""
    package = tmp_path / "site-packages" / "vllm"
    (package / "models" / "__pycache__").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "models" / "qwen3.py").write_text(ORIGINAL)
    (package / "models" / "__pycache__" / "qwen3.cpython-312.pyc").write_bytes(b"\0")
    return package


def _installed(monkeypatch: pytest.MonkeyPatch, vllm: Path, *, namespaces: bool = False) -> None:
    def locate(layout):
        if layout.import_name != "vllm":
            raise UnsupportedComponentError(f"{layout.import_name} is not installed")
        return PackageBase("vllm", vllm, "vllm", "0.0.0")

    monkeypatch.setattr(agent_sandbox, "locate_installed_package", locate)
    monkeypatch.setattr(agent_sandbox, "mount_namespaces", lambda: namespaces)


def _lease(tmp_path: Path, name: str) -> BranchLease:
    return BranchLease(name, name, (0,), tmp_path / "leases" / name, {}, 0)


def _sandbox(tmp_path: Path, branch: str) -> LocalAgentSandbox:
    sandbox = LocalAgentSandboxes(None).sandbox(branch, None, tmp_path, _lease(tmp_path, branch))
    sandbox.poll_seconds = 0.02
    sandbox.grace_seconds = 1.0
    return sandbox


async def _until(predicate: Callable[[], object], seconds: float = 10.0) -> None:
    until = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < until, "timed out waiting"
        await asyncio.sleep(0.02)


def test_a_command_writing_the_installed_framework_without_mount_namespaces_is_undone_and_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    vllm = _vllm(tmp_path)
    _installed(monkeypatch, vllm)
    fuse = f"echo FUSED >> {vllm}/models/qwen3.py; echo pass > {vllm}/models/qwen3_fused.py; rm {vllm}/__init__.py"
    with caplog.at_level(logging.ERROR):
        sandbox = _sandbox(tmp_path, "b1")
        done = asyncio.run(sandbox.run(fuse, tmp_path, {}, 10))

    assert (vllm / "models" / "qwen3.py").read_text() == ORIGINAL
    assert not (vllm / "models" / "qwen3_fused.py").exists() and (vllm / "__init__.py").exists()
    assert (vllm / "models" / "__pycache__" / "qwen3.cpython-312.pyc").exists()
    assert GUARDED_NOTE in done.output
    for said in (done.output, caplog.text):
        assert f"changed {vllm}/models/qwen3.py" in said and f"added {vllm}/models/qwen3_fused.py" in said
        assert f"removed {vllm}/__init__.py" in said and f"b1: `{fuse}`" in said
    # The degraded mode is an error of the session's log, not a warning.
    assert any(r.levelno == logging.ERROR and "cannot be mounted read-only" in r.message for r in caplog.records)


def test_a_host_with_mount_namespaces_mounts_the_frameworks_read_only_and_guards_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vllm = _vllm(tmp_path)
    _installed(monkeypatch, vllm, namespaces=True)
    assert framework_protection() == FrameworkProtection(read_only=(vllm,))


def test_a_host_with_no_framework_installed_has_nothing_to_protect(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(layout):
        raise UnsupportedComponentError(f"{layout.import_name} is not installed")

    monkeypatch.setattr(agent_sandbox, "locate_installed_package", missing)
    assert framework_protection() == FrameworkProtection()


def test_a_started_process_that_writes_the_framework_is_undone_once_it_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vllm = _vllm(tmp_path)
    _installed(monkeypatch, vllm)

    async def scenario() -> str:
        process = await _sandbox(tmp_path, "b1").start(f"sleep 0.3; echo FUSED >> {vllm}/models/qwen3.py", tmp_path, {})
        await _until(lambda: (vllm / "models" / "qwen3.py").read_text() == ORIGINAL and process.returncode == 0)
        await _until(lambda: "restored" in process.log.read_text())
        return await process.output()

    output = asyncio.run(scenario())
    assert f"changed {vllm}/models/qwen3.py" in output


def test_the_next_command_of_any_sandbox_restores_what_a_running_one_wrote_and_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vllm = _vllm(tmp_path)
    _installed(monkeypatch, vllm)
    writer = f"echo FUSED >> {vllm}/models/qwen3.py; touch {tmp_path}/written; sleep 60"

    async def scenario() -> str:
        forge = await _sandbox(tmp_path, "b1").start(writer, tmp_path, {})
        try:
            await _until(lambda: (tmp_path / "written").exists())
            done = await _sandbox(tmp_path, "b2").run(f"cat {vllm}/models/qwen3.py", tmp_path, {}, 10)
        finally:
            await forge.kill()
        return done.output

    output = asyncio.run(scenario())
    assert "before this command:" in output and f"b1: `{writer}`" in output
    assert "FUSED" not in output.split("Write into a copy")[-1], "the command reads the restored file"


def test_a_host_whose_frameworks_cannot_be_copied_refuses_every_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vllm = _vllm(tmp_path)
    _installed(monkeypatch, vllm)

    def no_disk(**_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(tempfile, "mkdtemp", no_disk)
    sandbox = _sandbox(tmp_path, "b1")
    with pytest.raises(FrameworksUnguarded, match="No space left"):
        asyncio.run(sandbox.run(f"touch {tmp_path}/ran", tmp_path, {}, 10))
    with pytest.raises(FrameworksUnguarded):
        asyncio.run(sandbox.start(f"touch {tmp_path}/ran", tmp_path, {}))
    assert not (tmp_path / "ran").exists()


def test_a_check_restores_every_kind_of_change_and_reports_only_real_ones(tmp_path: Path) -> None:
    vllm = _vllm(tmp_path)
    (vllm / "build").mkdir()
    (vllm / "tool.sh").write_text("#!/bin/sh\n")
    (vllm / "tool.sh").chmod(0o755)
    (vllm / "latest").symlink_to("models")
    guard = FrameworkGuard((GuardedPackage("vllm", vllm, ("build",)),))
    assert guard.check() is None

    os.utime(vllm / "__init__.py")
    (vllm / "models" / "__pycache__" / "fused.cpython-312.pyc").write_bytes(b"\1")
    (vllm / "build" / "kernel.so").write_bytes(b"\1")
    assert guard.check() is None, "a touch, a bytecode cache and a generated build are no change"

    (vllm / "tool.sh").chmod(0o644)
    (vllm / "latest").unlink()
    (vllm / "latest").symlink_to("elsewhere")
    (vllm / "kernels" / "fused").mkdir(parents=True)
    (vllm / "kernels" / "fused" / "op.py").write_text("")
    (vllm / "models" / "qwen3.py").unlink()
    (vllm / "models" / "qwen3.py").mkdir()
    restored = guard.check()

    assert restored is not None and restored.suspects == ()
    assert set(restored.paths) == {
        f"changed {vllm}/tool.sh",
        f"changed {vllm}/latest",
        f"added {vllm}/kernels/fused/op.py",
        f"added {vllm}/kernels/fused/",
        f"added {vllm}/kernels/",
        f"added {vllm}/models/qwen3.py/",
        f"removed {vllm}/models/qwen3.py",
    }
    assert (vllm / "models" / "qwen3.py").read_text() == ORIGINAL and not (vllm / "kernels").exists()
    assert os.access(vllm / "tool.sh", os.X_OK) and os.readlink(vllm / "latest") == "models"
    assert (vllm / "build" / "kernel.so").exists()
    assert guard.check() is None


def test_concurrent_checks_restore_once_and_name_every_command_that_ran(tmp_path: Path) -> None:
    vllm = _vllm(tmp_path)
    guard = FrameworkGuard((GuardedPackage("vllm", vllm),))
    assert guard.check() is None
    running = guard.started("b1", "forge-fuse")
    guard.ended(guard.started("b2", "make"))
    (vllm / "models" / "qwen3.py").write_text("FUSED\n")
    found = []

    def check() -> None:
        found.append(guard.check())

    threads = [threading.Thread(target=check) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    restored = [result for result in found if result is not None]
    assert len(restored) == 1 and restored[0].suspects == ("b1: `forge-fuse`", "b2: `make`")
    assert (vllm / "models" / "qwen3.py").read_text() == ORIGINAL
    guard.ended(running)
    (vllm / "models" / "qwen3.py").write_text("AGAIN\n")
    assert guard.check().suspects == ("b1: `forge-fuse`",), "an ended command is named once more, then forgotten"
    (vllm / "models" / "qwen3.py").write_text("THIRD\n")
    assert guard.check().suspects == ()


def _guard(sandbox: LocalAgentSandbox, patrol_seconds: float) -> FrameworkGuard:
    guard = sandbox.frameworks.guard
    assert guard is not None
    guard.patrol_seconds = patrol_seconds
    return guard


def _logs(sandbox: LocalAgentSandbox) -> str:
    return "".join(log.read_text() for log in sorted((sandbox.root / "logs").glob("*.log")))


def test_what_a_command_left_running_writes_is_undone_and_named_with_no_further_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vllm = _vllm(tmp_path)
    _installed(monkeypatch, vllm)
    detached = f"(sleep 0.3; echo FUSED >> {vllm}/models/qwen3.py) >/dev/null 2>&1 &"

    async def scenario(sandbox: LocalAgentSandbox) -> None:
        done = await sandbox.run(detached, tmp_path, {}, 10)
        assert done.returncode == 0
        await _until(lambda: f"changed {vllm}/models/qwen3.py" in _logs(sandbox))
        await _until(lambda: not sandbox.alive())

    sandbox = _sandbox(tmp_path, "b1")
    _guard(sandbox, 3600.0)
    asyncio.run(scenario(sandbox))
    assert (vllm / "models" / "qwen3.py").read_text() == ORIGINAL
    assert f"b1: `{detached}`" in _logs(sandbox)


def test_a_process_still_running_has_what_it_wrote_undone_while_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vllm = _vllm(tmp_path)
    _installed(monkeypatch, vllm)
    agent = f"echo FUSED >> {vllm}/models/qwen3.py; sleep 60"

    async def scenario(sandbox: LocalAgentSandbox) -> None:
        process = await sandbox.start(agent, tmp_path, {})
        try:
            await _until(lambda: f"changed {vllm}/models/qwen3.py" in _logs(sandbox))
            assert process.returncode is None
            assert (vllm / "models" / "qwen3.py").read_text() == ORIGINAL
        finally:
            await process.kill()

    sandbox = _sandbox(tmp_path, "b1")
    _guard(sandbox, 0.1)
    asyncio.run(scenario(sandbox))
    assert f"b1: `{agent}`" in _logs(sandbox)


def test_stopping_a_sandbox_undoes_what_its_processes_wrote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vllm = _vllm(tmp_path)
    _installed(monkeypatch, vllm)

    async def scenario(sandbox: LocalAgentSandbox) -> None:
        writer = f"(sleep 0.3; echo FUSED >> {vllm}/models/qwen3.py; sleep 60) >/dev/null 2>&1 &"
        await sandbox.run(writer, tmp_path, {}, 10)
        await _until(lambda: "FUSED" in (vllm / "models" / "qwen3.py").read_text())
        await sandbox.stop()
        assert (vllm / "models" / "qwen3.py").read_text() == ORIGINAL

    sandbox = _sandbox(tmp_path, "b1")
    _guard(sandbox, 3600.0)
    asyncio.run(scenario(sandbox))


def test_the_frameworks_are_restored_when_the_process_exits(tmp_path: Path) -> None:
    vllm = _vllm(tmp_path)
    script = (
        "import sys, tempfile\n"
        "from pathlib import Path\n"
        "from hyperloom_multibranch.specialist.framework_guard import FrameworkGuard, GuardedPackage\n"
        "tempfile.tempdir = sys.argv[2]\n"
        "guard = FrameworkGuard((GuardedPackage('vllm', Path(sys.argv[1])),))\n"
        "guard.check()\n"
        "(Path(sys.argv[1]) / 'models' / 'qwen3.py').write_text('FUSED\\n')\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [str(Path(__file__).parents[2]), os.environ.get("PYTHONPATH")])),
    }
    done = subprocess.run(
        [sys.executable, "-c", script, str(vllm), str(tmp_path / "local-tmp")], env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert (vllm / "models" / "qwen3.py").read_text() == ORIGINAL
    assert "restored as installed" in done.stderr
    assert not list((tmp_path / "local-tmp").glob("hyperloom-frameworks-*")), "the pristine copy is removed after"


def test_a_check_restores_around_files_a_running_writer_removes_meanwhile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vllm = _vllm(tmp_path)
    (vllm / "models" / "llama.py").write_text(ORIGINAL)
    guard = FrameworkGuard((GuardedPackage("vllm", vllm),))
    assert guard.check() is None
    (vllm / "tmpdir0" / "deep").mkdir(parents=True)
    (vllm / "tmpdir0" / "deep" / "x.py").write_text("")
    (vllm / "a_tmp.py").write_text("")
    (vllm / "models" / "qwen3.py").write_text("FUSED\n")
    walk = framework_guard._Tree.walk

    def racing(root: Path, generated: tuple[str, ...]) -> framework_guard._Tree:
        tree = walk(root, generated)
        # A writer's tmp+rename and cleanup land between the walk and the restore.
        shutil.rmtree(vllm / "tmpdir0")
        (vllm / "a_tmp.py").unlink()
        (vllm / "models" / "llama.py").unlink()
        return tree

    monkeypatch.setattr(framework_guard._Tree, "walk", staticmethod(racing))
    restored = guard.check()
    assert restored is not None and f"changed {vllm}/models/qwen3.py" in restored.paths
    assert (vllm / "models" / "qwen3.py").read_text() == ORIGINAL
    assert (vllm / "models" / "llama.py").read_text() == ORIGINAL


def test_a_walk_survives_a_directory_removed_under_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vllm = _vllm(tmp_path)
    (vllm / "tmpdir0").mkdir()
    real = os.walk

    def walk(top, onerror=None, **kwargs):
        for entry in real(top, onerror=onerror, **kwargs):
            yield entry
            if (vllm / "tmpdir0").exists():
                shutil.rmtree(vllm / "tmpdir0")

    monkeypatch.setattr(framework_guard.os, "walk", walk)
    tree = framework_guard._Tree.walk(vllm, ())
    assert "models/qwen3.py" in tree.files
