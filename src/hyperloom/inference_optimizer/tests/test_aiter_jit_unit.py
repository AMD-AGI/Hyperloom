# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
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
    # Force resolution to fail so the resolved-is-None early return is hit.
    monkeypatch.setattr(aj, "_resolve_lock_sweep_dir", lambda d: None)
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
    monkeypatch.setattr(aj, "_any_live_compiler", lambda: True)
    stats = aj.sweep_stale_aiter_locks_if_dead(Path("/whatever"))
    assert stats["skipped_live"] is True
    assert stats["compiler_alive"] is True


def test_sweep_unknown_liveness_uses_mtime_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(aj, "_any_live_compiler", lambda: None)
    stats = aj.sweep_stale_aiter_locks_if_dead(tmp_path)
    assert stats["compiler_alive"] is None


def test_sweep_dead_compiler_sweeps_with_zero_gate(tmp_path, monkeypatch):
    lock = tmp_path / "lock"
    lock.write_text("", encoding="utf-8")  # fresh, but dead compiler → deleted
    monkeypatch.setattr(aj, "_any_live_compiler", lambda: False)
    stats = aj.sweep_stale_aiter_locks_if_dead(tmp_path)
    assert stats["compiler_alive"] is False
    assert stats["deleted"] == 1
    assert not lock.exists()
