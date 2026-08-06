# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the shared aiter JIT lock-sweep helpers.

Exercises directory resolution (arg / env / dynamic / fallbacks), the stale
lock sweep (fresh vs stale, unreadable, delete errors), compiler-liveness
detection (psutil missing / process match / cmdline match / enumeration
error), and the liveness-gated sweep dispatch.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from hyperloom.orchestrator.actions.executors import _aiter_jit as aj


# ---------------------------------------------------------------------------
# _resolve_lock_sweep_dir
# ---------------------------------------------------------------------------


def test_resolve_dir_trusts_explicit_arg(tmp_path):
    assert aj._resolve_lock_sweep_dir(tmp_path) == tmp_path


def test_resolve_dir_uses_env_override(tmp_path, monkeypatch):
    build = tmp_path / "build"
    build.mkdir()
    monkeypatch.setenv("INFERENCE_OPTIMIZER_AITER_JIT_DIR", str(tmp_path))
    # Env override adds both <dir> and <dir>/build.
    resolved = aj._resolve_lock_sweep_dir(None)
    assert resolved in (tmp_path, build)


def test_resolve_dir_none_when_nothing_exists(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_AITER_JIT_DIR", "/nonexistent/aiter/xyz")

    def _no_aiter(name):
        raise ImportError("no aiter")

    monkeypatch.setattr(aj.importlib.util, "find_spec", _no_aiter)
    # Fallbacks are absolute system paths unlikely to exist in CI sandbox.
    resolved = aj._resolve_lock_sweep_dir(None)
    assert resolved is None or resolved.is_dir()


# ---------------------------------------------------------------------------
# _resolve_aiter_jit_dir_dynamic
# ---------------------------------------------------------------------------


def test_resolve_dynamic_returns_empty_when_missing(monkeypatch):
    monkeypatch.setattr(aj.importlib.util, "find_spec", lambda name: None)
    assert aj._resolve_aiter_jit_dir_dynamic() == []


def test_resolve_dynamic_importerror_returns_empty(monkeypatch):
    def _boom(name):
        raise ValueError("bad spec")

    monkeypatch.setattr(aj.importlib.util, "find_spec", _boom)
    assert aj._resolve_aiter_jit_dir_dynamic() == []


def test_resolve_dynamic_returns_paths(monkeypatch, tmp_path):
    fake_origin = tmp_path / "aiter" / "__init__.py"
    fake_origin.parent.mkdir(parents=True)
    fake_origin.write_text("", encoding="utf-8")

    class _Spec:
        origin = str(fake_origin)

    monkeypatch.setattr(aj.importlib.util, "find_spec", lambda name: _Spec())
    paths = aj._resolve_aiter_jit_dir_dynamic()
    assert paths == [
        str(tmp_path / "aiter" / "jit"),
        str(tmp_path / "aiter" / "jit" / "build"),
    ]


# ---------------------------------------------------------------------------
# clean_stale_aiter_locks
# ---------------------------------------------------------------------------


def test_clean_no_dir_returns_zero_stats():
    stats = aj.clean_stale_aiter_locks(Path("/definitely/not/here/aiter"))
    assert stats["scanned"] == 0
    assert stats["deleted"] == 0


def test_clean_unresolvable_dir_returns_empty_stats(monkeypatch):
    # Force resolution to yield no trees so the empty-list early return is hit.
    monkeypatch.setattr(aj, "_resolve_lock_sweep_dirs", lambda d: [])
    stats = aj.clean_stale_aiter_locks(None)
    assert stats["dir"] is None
    assert stats["scanned"] == 0


def test_clean_deletes_stale_lock(tmp_path):
    lock = tmp_path / "lock"
    lock.write_text("", encoding="utf-8")
    old = time.time() - 3600
    os.utime(lock, (old, old))

    stats = aj.clean_stale_aiter_locks(tmp_path, stale_minutes=5)
    assert stats["scanned"] == 1
    assert stats["deleted"] == 1
    assert not lock.exists()


def test_clean_skips_fresh_lock(tmp_path):
    lock = tmp_path / ".ninja_lock"
    lock.write_text("", encoding="utf-8")

    stats = aj.clean_stale_aiter_locks(tmp_path, stale_minutes=5)
    assert stats["scanned"] == 1
    assert stats["skipped_fresh"] == 1
    assert lock.exists()


def test_clean_matches_lock_prefix(tmp_path):
    lock = tmp_path / "lock_moduleA"
    lock.write_text("", encoding="utf-8")
    stats = aj.clean_stale_aiter_locks(tmp_path, stale_minutes=0)
    assert stats["deleted"] == 1


def test_clean_ignores_non_lock_files(tmp_path):
    (tmp_path / "kernel.so").write_text("", encoding="utf-8")
    stats = aj.clean_stale_aiter_locks(tmp_path, stale_minutes=0)
    assert stats["scanned"] == 0


def test_clean_counts_unlink_error(tmp_path, monkeypatch):
    lock = tmp_path / "lock"
    lock.write_text("", encoding="utf-8")

    def _boom(self, *a, **k):
        raise OSError("cannot unlink")

    monkeypatch.setattr(Path, "unlink", _boom)
    stats = aj.clean_stale_aiter_locks(tmp_path, stale_minutes=0)
    assert stats["errors"] == 1


def test_clean_counts_stat_error(tmp_path, monkeypatch):
    lock = tmp_path / "lock"
    lock.write_text("", encoding="utf-8")

    real_stat = Path.stat

    def _stat(self, *a, **k):
        if self.name == "lock":
            raise OSError("stat failed")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", _stat)
    stats = aj.clean_stale_aiter_locks(tmp_path, stale_minutes=0)
    assert stats["errors"] == 1


def test_clean_walk_oserror(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise OSError("walk failed")

    monkeypatch.setattr(aj.os, "walk", _boom)
    stats = aj.clean_stale_aiter_locks(tmp_path, stale_minutes=0)
    assert stats["errors"] == 1


def test_auto_resolution_sweeps_cpp_and_jit_build_trees(tmp_path, monkeypatch):
    aiter_root = tmp_path / "cpp"
    jit_root = tmp_path / "jit"
    cpp_lock = aiter_root / "build" / "pa_ragged" / "lock"
    jit_lock = jit_root / "build" / "lock_module"
    cpp_lock.parent.mkdir(parents=True)
    jit_lock.parent.mkdir(parents=True)
    cpp_lock.write_text("", encoding="utf-8")
    jit_lock.write_text("", encoding="utf-8")
    monkeypatch.setenv("AITER_ROOT_DIR", str(aiter_root))
    monkeypatch.setenv("AITER_JIT_DIR", str(jit_root))
    monkeypatch.setattr(aj, "AITER_CPP_BUILD_PROBE_PATHS", ())
    monkeypatch.setattr(aj, "AITER_JIT_PROBE_PATHS", ())
    monkeypatch.setattr(aj.importlib.util, "find_spec", lambda _name: None)

    stats = aj.clean_stale_aiter_locks(stale_minutes=0)

    assert stats["deleted"] == 2
    assert str(aiter_root / "build") in stats["dirs"]
    assert str(jit_root / "build") in stats["dirs"]
    assert not cpp_lock.exists()
    assert not jit_lock.exists()


def test_find_aiter_baton_wait_returns_bounded_evidence(tmp_path):
    server_log = tmp_path / "warmup" / "server.log"
    server_log.parent.mkdir()
    server_log.write_text(
        "model loaded\n"
        "[aiter] waiting for baton release at "
        "/root/.aiter/build/pa_ragged/lock\n",
        encoding="utf-8",
    )

    evidence = aj.find_aiter_baton_wait(tmp_path)

    assert evidence is not None
    assert evidence["log_path"] == str(server_log)
    assert "waiting for baton release" in evidence["excerpt"]


# ---------------------------------------------------------------------------
# _any_live_compiler
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, info):
        self.info = info


def _install_fake_psutil(monkeypatch, procs, iter_raises=False):
    import types

    fake = types.ModuleType("psutil")

    class NoSuchProcess(Exception):
        pass

    class AccessDenied(Exception):
        pass

    class ZombieProcess(Exception):
        pass

    fake.NoSuchProcess = NoSuchProcess
    fake.AccessDenied = AccessDenied
    fake.ZombieProcess = ZombieProcess

    def _process_iter(fields):
        if iter_raises:
            raise RuntimeError("enum blew up")
        return iter(procs)

    fake.process_iter = _process_iter
    monkeypatch.setitem(__import__("sys").modules, "psutil", fake)
    return fake


def test_any_live_compiler_psutil_missing(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "psutil", None)
    assert aj._any_live_compiler() is None


def test_any_live_compiler_name_match(monkeypatch):
    _install_fake_psutil(monkeypatch, [_FakeProc({"name": "ninja", "cmdline": []})])
    assert aj._any_live_compiler() is True


def test_any_live_compiler_cmdline_match(monkeypatch):
    _install_fake_psutil(
        monkeypatch,
        [_FakeProc({"name": "sh", "cmdline": ["/usr/bin/hipcc", "-c", "x.cpp"]})],
    )
    assert aj._any_live_compiler() is True


def test_live_compiler_filter_ignores_unrelated_build(monkeypatch, tmp_path):
    _install_fake_psutil(
        monkeypatch,
        [
            _FakeProc(
                {
                    "name": "hipcc",
                    "cmdline": ["hipcc", "-c", "/dev/null", "-o", "/dev/null"],
                    "cwd": "/tmp/unrelated",
                }
            )
        ],
    )

    assert aj._any_live_compiler([tmp_path / "aiter" / "build"]) is False


def test_live_compiler_filter_matches_build_output(monkeypatch, tmp_path):
    build_dir = tmp_path / "aiter" / "build"
    _install_fake_psutil(
        monkeypatch,
        [
            _FakeProc(
                {
                    "name": "hipcc",
                    "cmdline": [
                        "hipcc",
                        "-c",
                        "/src/attention.cu",
                        "-o",
                        str(build_dir / "pa_ragged" / "attention.o"),
                    ],
                    "cwd": "/src",
                }
            )
        ],
    )

    assert aj._any_live_compiler([build_dir]) is True


def test_any_live_compiler_none_alive(monkeypatch):
    _install_fake_psutil(
        monkeypatch,
        [_FakeProc({"name": "python", "cmdline": ["python", "run.py"]})],
    )
    assert aj._any_live_compiler() is False


def test_any_live_compiler_enum_error(monkeypatch):
    _install_fake_psutil(monkeypatch, [], iter_raises=True)
    assert aj._any_live_compiler() is None


class _RaisingProc:
    """A proc whose ``.info`` access raises a per-process psutil error."""

    def __init__(self, exc):
        self._exc = exc

    @property
    def info(self):
        raise self._exc


def test_any_live_compiler_skips_dead_process(monkeypatch):
    fake = _install_fake_psutil(monkeypatch, [])
    procs = [
        _RaisingProc(fake.NoSuchProcess()),
        _FakeProc({"name": "ninja", "cmdline": []}),
    ]
    # Raising proc first so the per-proc except runs before the real match.
    import sys

    def _iter(fields):
        return iter(procs)

    fake.process_iter = _iter
    monkeypatch.setitem(sys.modules, "psutil", fake)
    assert aj._any_live_compiler() is True


# ---------------------------------------------------------------------------
# sweep_stale_aiter_locks_if_dead
# ---------------------------------------------------------------------------


def test_sweep_skips_when_compiler_alive(monkeypatch):
    monkeypatch.setattr(aj, "_any_live_compiler", lambda *_args: True)
    stats = aj.sweep_stale_aiter_locks_if_dead(Path("/whatever"))
    assert stats["skipped_live"] is True
    assert stats["compiler_alive"] is True


def test_sweep_unknown_liveness_uses_mtime_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(aj, "_any_live_compiler", lambda *_args: None)
    stats = aj.sweep_stale_aiter_locks_if_dead(tmp_path)
    assert stats["compiler_alive"] is None


def test_sweep_dead_compiler_keeps_fresh_ownerless_lock(tmp_path, monkeypatch):
    lock = tmp_path / "lock"
    lock.write_text("", encoding="utf-8")
    monkeypatch.setattr(aj, "_any_live_compiler", lambda *_args: False)
    stats = aj.sweep_stale_aiter_locks_if_dead(tmp_path)
    assert stats["compiler_alive"] is False
    assert stats["deleted"] == 0
    assert stats["skipped_fresh"] == 1
    assert lock.exists()
