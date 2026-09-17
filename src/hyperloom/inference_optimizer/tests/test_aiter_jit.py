# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the shared aiter JIT lock-cleanup helpers (``_aiter_jit``)."""

import os
import time

import pytest

try:
    import psutil
except ModuleNotFoundError:  # optional runtime dependency
    psutil = None

from hyperloom.orchestrator.actions.executors import _aiter_jit
from hyperloom.orchestrator.actions.executors import baseline


# Only tests that drive psutil directly require it; sweep / _resolve_timeout tests monkeypatch ``_any_live_compiler``
# and run regardless.
requires_psutil = pytest.mark.skipif(psutil is None, reason="psutil not installed (optional runtime dependency)")


# fixtures / helpers
def _make_aiter_tree(root):
    """Build a jit/build/ layout with a stale + a fresh lock."""
    stale_mtime = time.time() - 30 * 60
    (root / "module_moe" / "build").mkdir(parents=True)

    stale_lock = root / "lock_module_moe"
    fresh_lock = root / "module_moe" / "build" / "lock"
    ninja_lock = root / "module_moe" / "build" / ".ninja_lock"
    non_lock = root / "module_moe" / "build" / "compile_commands.json"

    for p, content in (
        (stale_lock, "x"),
        (fresh_lock, "x"),
        (ninja_lock, "x"),
        (non_lock, "{}"),
    ):
        p.write_text(content)
    for p in (stale_lock, ninja_lock):
        os.utime(p, (stale_mtime, stale_mtime))
    return {
        "stale_lock": stale_lock,
        "fresh_lock": fresh_lock,
        "ninja_lock": ninja_lock,
        "non_lock": non_lock,
    }


class _FakeProc:
    """Minimal psutil.Process stand-in exposing the ``.info`` dict."""

    def __init__(self, name="", cmdline=None):
        self.info = {"name": name, "cmdline": cmdline or []}


def _patch_process_iter(monkeypatch, procs):
    monkeypatch.setattr(
        psutil,
        "process_iter",
        lambda attrs=None: iter(procs),
    )


# _any_live_compiler
@requires_psutil
def test_any_live_compiler_true_on_name_match(monkeypatch):
    _patch_process_iter(
        monkeypatch,
        [
            _FakeProc(name="bash"),
            _FakeProc(name="hipcc"),
        ],
    )
    assert _aiter_jit._any_live_compiler() is True


@requires_psutil
def test_any_live_compiler_false_when_no_compiler(monkeypatch):
    _patch_process_iter(
        monkeypatch,
        [
            _FakeProc(name="bash"),
            _FakeProc(name="python", cmdline=["python", "serve.py"]),
        ],
    )
    assert _aiter_jit._any_live_compiler() is False


@requires_psutil
def test_any_live_compiler_matches_cmdline_when_name_is_wrapper(monkeypatch):
    # ``name`` may surface as the wrapper (perl) while cmdline's first token is hipcc.
    _patch_process_iter(
        monkeypatch,
        [
            _FakeProc(name="perl", cmdline=["/opt/rocm/bin/hipcc", "-c", "x.cu"]),
        ],
    )
    assert _aiter_jit._any_live_compiler() is True


@requires_psutil
def test_any_live_compiler_none_on_enumeration_error(monkeypatch):
    def _boom(attrs=None):
        raise psutil.Error("boom")

    monkeypatch.setattr(psutil, "process_iter", _boom)
    assert _aiter_jit._any_live_compiler() is None


@requires_psutil
def test_any_live_compiler_skips_dead_procs(monkeypatch):
    class _RaisingProc:
        @property
        def info(self):
            raise psutil.NoSuchProcess(pid=1)

    _patch_process_iter(monkeypatch, [_RaisingProc(), _FakeProc(name="ninja")])
    assert _aiter_jit._any_live_compiler() is True


# sweep_stale_aiter_locks_if_dead
def test_sweep_skips_when_compiler_alive(monkeypatch, tmp_path):
    layout = _make_aiter_tree(tmp_path)
    monkeypatch.setattr(_aiter_jit, "_any_live_compiler", lambda *_args: True)
    stats = _aiter_jit.sweep_stale_aiter_locks_if_dead(aiter_jit_dir=tmp_path)
    assert stats["skipped_live"] is True
    assert stats["deleted"] == 0
    assert layout["stale_lock"].exists()
    assert layout["fresh_lock"].exists()


def test_sweep_deletes_stale_locks_when_dead(monkeypatch, tmp_path):
    layout = _make_aiter_tree(tmp_path)
    monkeypatch.setattr(_aiter_jit, "_any_live_compiler", lambda *_args: False)
    stats = _aiter_jit.sweep_stale_aiter_locks_if_dead(aiter_jit_dir=tmp_path)
    assert stats["compiler_alive"] is False
    assert stats["deleted"] == 2
    assert stats["skipped_fresh"] == 1
    assert not layout["stale_lock"].exists()
    assert layout["fresh_lock"].exists()
    assert not layout["ninja_lock"].exists()
    assert layout["non_lock"].exists()


def test_sweep_unknown_falls_back_to_mtime_gate(monkeypatch, tmp_path):
    layout = _make_aiter_tree(tmp_path)
    monkeypatch.setattr(_aiter_jit, "_any_live_compiler", lambda *_args: None)
    stats = _aiter_jit.sweep_stale_aiter_locks_if_dead(aiter_jit_dir=tmp_path)
    assert stats["compiler_alive"] is None
    # mtime gate (5 min) ⇒ the 30-min-old locks go; the fresh lock survives.
    assert stats["deleted"] == 2
    assert stats["skipped_fresh"] == 1
    assert not layout["stale_lock"].exists()
    assert not layout["ninja_lock"].exists()
    assert layout["fresh_lock"].exists()


def test_benchmark_timeout_does_not_probe_or_expand_for_jit(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", "7800")
    exe = baseline.BaselineExecutor()
    assert exe._resolve_timeout({}) == 7800
