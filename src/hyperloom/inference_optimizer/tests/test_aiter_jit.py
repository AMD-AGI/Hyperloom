# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the shared aiter JIT lock-cleanup helpers (``_aiter_jit``)."""

import os
import sys
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

try:
    import psutil
except ModuleNotFoundError:  # optional runtime dependency
    psutil = None

from hyperloom.orchestrator.actions.executors import _aiter_jit
from hyperloom.orchestrator.actions.executors import baseline


# Only tests that drive psutil directly require it; sweep tests replace the liveness probe.
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


@requires_psutil
@pytest.mark.parametrize("name", ["python", "hipcc"])
@pytest.mark.parametrize("process_state", ["zombie", "denied", "unknown-status"])
def test_sweep_distinguishes_zombies_from_unreadable_live_processes(monkeypatch, tmp_path, name, process_state):
    class ProcessAttrs:
        """Drive psutil's real exception-to-None conversion without an OS process."""

        def oneshot(self):
            return nullcontext()

        def name(self):
            return name

        def status(self):
            if process_state == "unknown-status":
                raise psutil.AccessDenied(pid=123)
            return psutil.STATUS_ZOMBIE if process_state == "zombie" else psutil.STATUS_RUNNING

        def cmdline(self):
            if process_state == "denied":
                raise psutil.AccessDenied(pid=123)
            raise psutil.ZombieProcess(pid=123)

        cwd = cmdline

    def process_iter(attrs):
        info = psutil.Process.as_dict(ProcessAttrs(), attrs=attrs)
        assert info["cmdline"] is None
        assert info["cwd"] is None
        return iter([SimpleNamespace(info=info)])

    monkeypatch.setattr(psutil, "process_iter", process_iter)
    layout = _make_aiter_tree(tmp_path)
    stats = _aiter_jit.sweep_stale_aiter_locks_if_dead(aiter_jit_dir=tmp_path)

    if process_state == "zombie":
        assert stats["compiler_alive"] is False
        assert stats["deleted"] == 2
        assert not layout["stale_lock"].exists()
        assert not layout["ninja_lock"].exists()
    else:
        assert stats["compiler_alive"] is None
        assert stats["deleted"] == 0
        assert stats["errors"] == 1
        assert layout["stale_lock"].exists()
        assert layout["ninja_lock"].exists()
    assert layout["fresh_lock"].exists()
    assert layout["non_lock"].exists()


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


def test_sweep_unknown_liveness_preserves_all_locks(monkeypatch, tmp_path):
    layout = _make_aiter_tree(tmp_path)
    monkeypatch.setattr(_aiter_jit, "_any_live_compiler", lambda *_args: None)
    stats = _aiter_jit.sweep_stale_aiter_locks_if_dead(aiter_jit_dir=tmp_path)
    assert stats["compiler_alive"] is None
    assert stats["deleted"] == 0
    assert stats["scanned"] == 0
    assert stats["errors"] == 1
    assert all(path.exists() for path in layout.values())


@pytest.mark.asyncio
async def test_integrate_does_not_retry_when_compiler_probe_is_unknown(monkeypatch, tmp_path):
    from hyperloom.orchestrator.kernel import request_handlers

    layout = _make_aiter_tree(tmp_path / "jit")
    monkeypatch.setattr(_aiter_jit, "_resolve_lock_sweep_dirs", lambda *_args: [tmp_path / "jit"])
    monkeypatch.setattr(_aiter_jit, "_any_live_compiler", lambda *_args: None)
    (tmp_path / "server.log").write_text(
        f"[aiter] waiting for baton release at {layout['stale_lock']}\n",
        encoding="utf-8",
    )
    calls = 0

    async def executor(_ctx):
        nonlocal calls
        calls += 1
        return {"status": "failed", "error_class": "timeout"}

    result = await request_handlers._run_integrate_rebaseline_with_lock_retry(
        executor, object(), workspace=tmp_path, reason="unknown compiler probe"
    )

    assert calls == 1
    assert result["stale_jit_lock"]["retry_attempted"] is False
    assert all(path.exists() for path in layout.values())


def test_benchmark_timeout_does_not_probe_or_expand_for_jit(monkeypatch):
    def unexpected_probe():
        pytest.fail("benchmark timeout resolution must not probe the JIT cache")

    monkeypatch.setattr(_aiter_jit, "probe_aiter_jit_cache", unexpected_probe)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", "7800")
    exe = baseline.BaselineExecutor()
    assert exe._resolve_timeout({}) == 7800


@pytest.fixture
def baseline_launch(monkeypatch, tmp_path):
    from hyperloom.orchestrator.actions.executors import _multi_node_server_lifecycle
    from hyperloom.orchestrator.actions.executors.profile import ProfileExecutor

    config = tmp_path / "baseline.yaml"
    config.write_text("benchmark:\n  framework: sglang\n", encoding="utf-8")
    jit_dir = tmp_path / "jit"
    jit_dir.mkdir()
    monkeypatch.setattr(_aiter_jit, "_resolve_lock_sweep_dirs", lambda *_args: [jit_dir])
    monkeypatch.setattr(baseline.BenchmarkRunExecutor, "_preflight_server_argv", lambda *args, **kwargs: None)
    monkeypatch.setattr(baseline, "build_benchmark_command", lambda **kwargs: ["fake-benchmark"])

    async def no_restart(**kwargs):
        return None

    monkeypatch.setattr(_multi_node_server_lifecycle, "restart_server_for_round", no_restart)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC", "7800")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SILENCE_TIMEOUT_SEC", "600")
    state = SimpleNamespace()
    baseline_executor = baseline.BaselineExecutor(
        magpie_python=sys.executable, session_dir=tmp_path, shared_state=state
    )
    profile_executor = ProfileExecutor(magpie_python=sys.executable, session_dir=tmp_path)
    profile_executor.shared_state = state

    async def launch(*, profile=False, ready=False, extra=None):
        executor = profile_executor if profile else baseline_executor
        return await executor._run_single_benchmark(
            config_path=config,
            output_dir=tmp_path / "round",
            timeout_sec=123,
            override_result_dir=None,
            resolved_model="fixture-model",
            materialized_config_path=config,
            inferencex_path="",
            effective_extra_server_args="",
            params={},
            ctx=SimpleNamespace(extra=extra or {}),
            server_already_ready=ready,
        )

    return launch, jit_dir


class _LaunchObserved(BaseException):
    """Stop the fake launch before any subprocess or result harvesting."""


@pytest.mark.asyncio
@pytest.mark.parametrize("compiler_alive", [False, True, None])
async def test_baseline_sweeps_before_every_server_launch(monkeypatch, baseline_launch, compiler_alive):
    launch, jit_dir = baseline_launch
    monkeypatch.setattr(_aiter_jit, "_any_live_compiler", lambda *_args: compiler_alive)

    for attempt in range(2):
        layout = _make_aiter_tree(jit_dir / str(attempt))

        def observe_launch(*args, **kwargs):
            assert layout["stale_lock"].exists() is (compiler_alive is not False)
            assert layout["ninja_lock"].exists() is (compiler_alive is not False)
            assert layout["fresh_lock"].exists()
            assert layout["non_lock"].exists()
            assert kwargs["timeout"] == 7800
            raise _LaunchObserved

        monkeypatch.setattr(baseline, "run_with_session_kill", observe_launch)
        with pytest.raises(_LaunchObserved):
            await launch()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["profile", "ready", "context-ready", "multi-node"])
async def test_baseline_does_not_sweep_for_profile_or_client_round(monkeypatch, baseline_launch, mode):
    launch, jit_dir = baseline_launch
    layout = _make_aiter_tree(jit_dir)

    def unexpected_scan(*args):
        pytest.fail("a profile or client-only round must not sweep AITER locks")

    def observe_launch(*args, **kwargs):
        assert all(path.exists() for path in layout.values())
        if mode == "profile":
            assert kwargs["timeout"] == 123
        raise _LaunchObserved

    if mode == "multi-node":
        monkeypatch.setenv("INFERENCE_OPTIMIZER_NODES", "2")
    monkeypatch.setattr(_aiter_jit, "_any_live_compiler", unexpected_scan)
    monkeypatch.setattr(baseline, "run_with_session_kill", observe_launch)
    with pytest.raises(_LaunchObserved):
        await launch(
            profile=mode == "profile",
            ready=mode == "ready",
            extra={"server_already_ready": mode == "context-ready", "mn_round_restarted": mode == "multi-node"},
        )


@pytest.mark.asyncio
async def test_baseline_does_not_sweep_when_preflight_refuses_launch(monkeypatch, baseline_launch):
    launch, jit_dir = baseline_launch
    layout = _make_aiter_tree(jit_dir)
    refusal = {"status": "failed", "error_class": "argv_invalid"}
    monkeypatch.setattr(baseline.BaselineExecutor, "_preflight_server_argv", lambda *args, **kwargs: refusal)

    def unexpected_scan(*args):
        pytest.fail("a refused launch must not sweep AITER locks")

    monkeypatch.setattr(_aiter_jit, "_any_live_compiler", unexpected_scan)
    assert await launch() == refusal
    assert all(path.exists() for path in layout.values())
