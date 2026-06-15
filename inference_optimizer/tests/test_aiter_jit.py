# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the shared aiter JIT lock-cleanup helpers (``_aiter_jit``).

Covers the liveness-gated sweep that reaps orphaned JIT locks before a cold
start: compiler-process detection, the live/dead/unknown branching, and the
``_resolve_timeout`` integration that only sweeps on the COLD path.
"""

import os
import time

import psutil

from inference_optimizer.orchestrator.action_executors import _aiter_jit
from inference_optimizer.orchestrator.action_executors import baseline


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
def _make_aiter_tree(root):
    """Build a jit/build/ layout with a stale + a fresh lock."""
    stale_mtime = time.time() - 30 * 60  # 30 min ago
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
    # fresh_lock keeps its just-now mtime.
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
        psutil, "process_iter",
        lambda attrs=None: iter(procs),
    )


# ---------------------------------------------------------------------------
# _any_live_compiler
# ---------------------------------------------------------------------------
def test_any_live_compiler_true_on_name_match(monkeypatch):
    _patch_process_iter(monkeypatch, [
        _FakeProc(name="bash"),
        _FakeProc(name="hipcc"),
    ])
    assert _aiter_jit._any_live_compiler() is True


def test_any_live_compiler_false_when_no_compiler(monkeypatch):
    _patch_process_iter(monkeypatch, [
        _FakeProc(name="bash"),
        _FakeProc(name="python", cmdline=["python", "serve.py"]),
    ])
    assert _aiter_jit._any_live_compiler() is False


def test_any_live_compiler_matches_cmdline_when_name_is_wrapper(monkeypatch):
    # hipcc is a perl/bash wrapper, so ``name`` may surface as perl while the
    # cmdline's first token is the real hipcc path.
    _patch_process_iter(monkeypatch, [
        _FakeProc(name="perl", cmdline=["/opt/rocm/bin/hipcc", "-c", "x.cu"]),
    ])
    assert _aiter_jit._any_live_compiler() is True


def test_any_live_compiler_none_on_enumeration_error(monkeypatch):
    def _boom(attrs=None):
        raise psutil.Error("boom")

    monkeypatch.setattr(psutil, "process_iter", _boom)
    assert _aiter_jit._any_live_compiler() is None


def test_any_live_compiler_skips_dead_procs(monkeypatch):
    class _RaisingProc:
        @property
        def info(self):
            raise psutil.NoSuchProcess(pid=1)

    _patch_process_iter(monkeypatch, [_RaisingProc(), _FakeProc(name="ninja")])
    assert _aiter_jit._any_live_compiler() is True


# ---------------------------------------------------------------------------
# sweep_stale_aiter_locks_if_dead
# ---------------------------------------------------------------------------
def test_sweep_skips_when_compiler_alive(monkeypatch, tmp_path):
    layout = _make_aiter_tree(tmp_path)
    monkeypatch.setattr(_aiter_jit, "_any_live_compiler", lambda: True)
    stats = _aiter_jit.sweep_stale_aiter_locks_if_dead(aiter_jit_dir=tmp_path)
    assert stats["skipped_live"] is True
    assert stats["deleted"] == 0
    # Nothing touched.
    assert layout["stale_lock"].exists()
    assert layout["fresh_lock"].exists()


def test_sweep_deletes_all_locks_when_dead(monkeypatch, tmp_path):
    layout = _make_aiter_tree(tmp_path)
    monkeypatch.setattr(_aiter_jit, "_any_live_compiler", lambda: False)
    stats = _aiter_jit.sweep_stale_aiter_locks_if_dead(aiter_jit_dir=tmp_path)
    assert stats["compiler_alive"] is False
    # stale_minutes=0 ⇒ even the fresh lock is reaped (liveness proves orphan).
    assert stats["deleted"] == 3
    assert stats["skipped_fresh"] == 0
    assert not layout["stale_lock"].exists()
    assert not layout["fresh_lock"].exists()
    assert not layout["ninja_lock"].exists()
    assert layout["non_lock"].exists()


def test_sweep_unknown_falls_back_to_mtime_gate(monkeypatch, tmp_path):
    layout = _make_aiter_tree(tmp_path)
    monkeypatch.setattr(_aiter_jit, "_any_live_compiler", lambda: None)
    stats = _aiter_jit.sweep_stale_aiter_locks_if_dead(aiter_jit_dir=tmp_path)
    assert stats["compiler_alive"] is None
    # mtime gate (5 min) ⇒ only the >30-min-old locks go; fresh lock survives.
    assert stats["deleted"] == 2
    assert stats["skipped_fresh"] == 1
    assert not layout["stale_lock"].exists()
    assert not layout["ninja_lock"].exists()
    assert layout["fresh_lock"].exists()


# ---------------------------------------------------------------------------
# _resolve_timeout integration
# ---------------------------------------------------------------------------
def _cold_probe(*_a, **_k):
    return {
        "path": "/fake/jit",
        "kernel_count": 1,
        "size_mb": 0,
        "is_cold": True,
        "probe_status": "found",
    }


def _warm_probe(*_a, **_k):
    return {
        "path": "/fake/jit",
        "kernel_count": 999,
        "size_mb": 100,
        "is_cold": False,
        "probe_status": "found",
    }


def test_resolve_timeout_sweeps_on_cold(monkeypatch):
    calls = {"sweep": 0, "probe": 0}

    def _probe(*_a, **_k):
        calls["probe"] += 1
        return _cold_probe()

    def _sweep(*_a, **_k):
        calls["sweep"] += 1
        return {"deleted": 2, "dir": "/fake/jit", "compiler_alive": False}

    monkeypatch.setattr(baseline, "_probe_aiter_jit_cache", _probe)
    monkeypatch.setattr(baseline, "sweep_stale_aiter_locks_if_dead", _sweep)

    exe = baseline.BaselineExecutor()
    timeout = exe._resolve_timeout({})
    assert timeout == baseline.BASELINE_COLD_START_TIMEOUT_SEC
    assert calls["sweep"] == 1
    # Re-probe after a deleting sweep ⇒ probe called twice.
    assert calls["probe"] == 2


def test_resolve_timeout_no_sweep_on_warm(monkeypatch):
    calls = {"sweep": 0}

    monkeypatch.setattr(baseline, "_probe_aiter_jit_cache", _warm_probe)
    monkeypatch.setattr(
        baseline, "sweep_stale_aiter_locks_if_dead",
        lambda *a, **k: calls.__setitem__("sweep", calls["sweep"] + 1) or {},
    )

    exe = baseline.BaselineExecutor()
    timeout = exe._resolve_timeout({})
    assert timeout == exe.default_timeout_sec
    assert calls["sweep"] == 0


def test_resolve_timeout_skips_reprobe_when_sweep_live(monkeypatch):
    calls = {"probe": 0}

    def _probe(*_a, **_k):
        calls["probe"] += 1
        return _cold_probe()

    monkeypatch.setattr(baseline, "_probe_aiter_jit_cache", _probe)
    monkeypatch.setattr(
        baseline, "sweep_stale_aiter_locks_if_dead",
        lambda *a, **k: {"skipped_live": True, "deleted": 0},
    )

    exe = baseline.BaselineExecutor()
    timeout = exe._resolve_timeout({})
    assert timeout == baseline.BASELINE_COLD_START_TIMEOUT_SEC
    # skipped_live ⇒ no re-probe ⇒ probe called once only.
    assert calls["probe"] == 1
