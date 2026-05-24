"""v0.8 KB_gaps/Dead-E — Cortex KB flusher lifecycle tests.

Covers the missing piece that previously left ``scripts.cortex_kb_flusher``
unwired: the cli now spawns the daemon after T0 anchoring and cleans it
up in its ``finally`` block. The lightweight tests below exercise the
spawn helper / stop helper / breakdown collector contract without
actually launching the real daemon (we stub :class:`subprocess.Popen`
so the suite stays hermetic).

Acceptance criteria covered (Dead-E §6):

* ``--degraded-kb``: spawn is skipped, marker reason=``cortex_disabled``.
* ``--no-kb-flusher``: spawn is skipped, marker reason=``flag_disabled``.
* Fresh session: spawn fires, pid path holds the launched pid, marker
  reason=``spawned``, breakdown surfaces ``flusher_status.alive``.
* Stale pid: prior pid file pointing at a dead process is overwritten
  (spawn fires anyway).
* Live prior pid: spawn is skipped, marker reason starts with
  ``prior_alive_pid=``.
* ``_stop_kb_flusher`` SIGTERMs the proc and removes the pid file.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer import cli as cli_module
from inference_optimizer.breakdown.collectors import collect_kb_provenance
from inference_optimizer.paths import make_session_dir
from inference_optimizer.session_paths import (
    cortex_flusher_pid,
    cortex_flusher_status_json,
)


# ---------------------------------------------------------------------------
# fixtures + helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _ns(**kwargs: Any) -> argparse.Namespace:
    base = dict(
        cortex_enabled=True,
        kb_flusher_enabled=True,
        kb_flusher_interval_sec=5.0,
        kb_flusher_batch_size=50,
        cortex_kb_url=None,
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


class _FakePopen:
    """Minimal subprocess.Popen stand-in for hermetic spawn tests.

    Defaults emulate a live long-running daemon (``poll() -> None`` until
    ``send_signal()`` is called); pass ``immediate_exit_code`` for a
    'daemon already gone' scenario.
    """

    def __init__(
        self,
        cmd: list[str],
        *,
        immediate_exit_code: int | None = None,
        pid: int = 99999,
        **_: Any,
    ) -> None:
        self.args = cmd
        self.pid = pid
        self._exit_code: int | None = immediate_exit_code
        self.signals_received: list[int] = []
        self.killed = False
        self.waited = False

    def poll(self) -> int | None:
        return self._exit_code

    def send_signal(self, signum: int) -> None:
        self.signals_received.append(signum)
        if self._exit_code is None:
            self._exit_code = 0

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        return 0 if self._exit_code is None else self._exit_code

    def kill(self) -> None:
        self.killed = True
        if self._exit_code is None:
            self._exit_code = -9


# ---------------------------------------------------------------------------
# _maybe_spawn_kb_flusher
# ---------------------------------------------------------------------------
def test_spawn_skipped_when_cortex_disabled(session_dir):
    args = _ns(cortex_enabled=False)
    proc, pid_path = cli_module._maybe_spawn_kb_flusher(
        args, session_dir=session_dir,
    )
    assert proc is None
    assert pid_path == cortex_flusher_pid(session_dir)
    marker = json.loads(
        cortex_flusher_status_json(session_dir).read_text(encoding="utf-8")
    )
    assert marker["enabled"] is False
    assert marker["spawned"] is False
    assert marker["reason"] == "cortex_disabled"


def test_spawn_skipped_when_flag_disabled(session_dir):
    args = _ns(kb_flusher_enabled=False)
    proc, _ = cli_module._maybe_spawn_kb_flusher(
        args, session_dir=session_dir,
    )
    assert proc is None
    marker = json.loads(
        cortex_flusher_status_json(session_dir).read_text(encoding="utf-8")
    )
    assert marker["enabled"] is False
    assert marker["reason"] == "flag_disabled"


def test_spawn_fires_for_fresh_session(session_dir, monkeypatch):
    captured: dict[str, Any] = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return _FakePopen(cmd, pid=12345)

    monkeypatch.setattr(cli_module.subprocess, "Popen", fake_popen)
    args = _ns(cortex_kb_url="http://kb-mock")
    proc, pid_path = cli_module._maybe_spawn_kb_flusher(
        args, session_dir=session_dir,
    )
    assert isinstance(proc, _FakePopen)
    assert proc.pid == 12345
    cmd = captured["cmd"]
    assert "inference_optimizer.scripts.cortex_kb_flusher" in cmd
    assert "--session-dir" in cmd and str(session_dir) in cmd
    assert "--interval-sec" in cmd and "5.0" in cmd
    # --batch-size flag was removed from the daemon; cli no longer forwards.
    assert "--batch-size" not in cmd
    assert "--cortex-kb-url" in cmd and "http://kb-mock" in cmd
    marker = json.loads(
        cortex_flusher_status_json(session_dir).read_text(encoding="utf-8")
    )
    assert marker["enabled"] is True
    assert marker["spawned"] is True
    assert marker["pid"] == 12345
    assert marker["reason"] == "spawned"
    assert marker["cortex_kb_url"] == "http://kb-mock"


def test_spawn_skipped_when_prior_pid_alive(session_dir, monkeypatch):
    # Write own pid into the pid file — os.kill(pid, 0) on ourselves
    # always succeeds without delivering a signal, so this exercises
    # the "prior alive" branch.
    pid_path = cortex_flusher_pid(session_dir)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    prior = os.getpid()
    pid_path.write_text(f"{prior}\n", encoding="utf-8")

    def fake_popen(*_a, **_kw):  # pragma: no cover — must not be called
        raise AssertionError("Popen must not run when prior daemon is alive")

    monkeypatch.setattr(cli_module.subprocess, "Popen", fake_popen)
    args = _ns()
    proc, _ = cli_module._maybe_spawn_kb_flusher(
        args, session_dir=session_dir,
    )
    assert proc is None
    marker = json.loads(
        cortex_flusher_status_json(session_dir).read_text(encoding="utf-8")
    )
    assert marker["spawned"] is False
    assert marker["pid"] == prior
    assert marker["reason"].startswith("prior_alive_pid=")


def test_spawn_overwrites_stale_pid_file(session_dir, monkeypatch):
    """A pid file pointing at a definitely-dead pid must not block spawn."""
    pid_path = cortex_flusher_pid(session_dir)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text("1\n", encoding="utf-8")  # init won't accept signals from us

    # Probe whether kill(1, 0) is permitted in this sandbox. Some CI
    # containers run as root and would falsely report pid 1 alive; in
    # that case skip the stale-detection branch under test.
    try:
        os.kill(1, 0)
        kill_one_works = True
    except (PermissionError, ProcessLookupError, OSError):
        kill_one_works = False
    if kill_one_works:
        pytest.skip("pid=1 reported alive in this sandbox; stale path untestable")

    spawn_called: dict[str, int] = {"n": 0}

    def fake_popen(cmd, **_kw):
        spawn_called["n"] += 1
        return _FakePopen(cmd, pid=22222)

    monkeypatch.setattr(cli_module.subprocess, "Popen", fake_popen)
    args = _ns()
    proc, _ = cli_module._maybe_spawn_kb_flusher(
        args, session_dir=session_dir,
    )
    assert spawn_called["n"] == 1
    assert isinstance(proc, _FakePopen)
    marker = json.loads(
        cortex_flusher_status_json(session_dir).read_text(encoding="utf-8")
    )
    assert marker["spawned"] is True
    assert marker["pid"] == 22222


# ---------------------------------------------------------------------------
# _stop_kb_flusher
# ---------------------------------------------------------------------------
def test_stop_kb_flusher_sigterm_and_cleanup(session_dir):
    pid_path = cortex_flusher_pid(session_dir)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text("12345\n", encoding="utf-8")
    proc = _FakePopen(["x"], pid=12345)
    cli_module._stop_kb_flusher(proc, pid_path, grace_sec=0.5)
    assert signal.SIGTERM in proc.signals_received
    assert proc.waited is True
    assert not pid_path.exists()


def test_stop_kb_flusher_none_is_noop(session_dir):
    pid_path = cortex_flusher_pid(session_dir)
    cli_module._stop_kb_flusher(None, pid_path)  # must not raise


def test_stop_kb_flusher_kills_after_timeout(session_dir, monkeypatch):
    pid_path = cortex_flusher_pid(session_dir)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text("12345\n", encoding="utf-8")

    class _StubborneProc(_FakePopen):
        def wait(self, timeout=None):
            if not self.signals_received or self.killed is False:
                # First wait (after SIGTERM) hangs until kill is called.
                if not self.killed:
                    raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)
            return 0

        def send_signal(self, signum):
            self.signals_received.append(signum)
            # do NOT auto-exit on SIGTERM

    proc = _StubborneProc(["x"], pid=12345)
    cli_module._stop_kb_flusher(proc, pid_path, grace_sec=0.01)
    assert proc.killed is True
    assert not pid_path.exists()


# ---------------------------------------------------------------------------
# collect_kb_provenance.flusher_status
# ---------------------------------------------------------------------------
def test_collect_kb_provenance_flusher_status_absent(session_dir):
    warnings: list[str] = []
    out = collect_kb_provenance(
        session_dir=session_dir,
        state={},
        manifest={},
        warnings=warnings,
    )
    fs = out["flusher_status"]
    assert fs["enabled"] is False
    assert fs["spawned"] is False
    assert fs["alive"] is False
    assert fs["reason"] == "no_marker"


def test_collect_kb_provenance_flusher_status_alive(session_dir):
    """When the boot marker + a live pid both exist, flusher_status
    reflects ``enabled/spawned/alive=True`` and the breakdown.warnings
    list does NOT get a ``kb_flusher:*`` entry.
    """
    cortex_flusher_status_json(session_dir).parent.mkdir(parents=True, exist_ok=True)
    cortex_flusher_status_json(session_dir).write_text(
        json.dumps({
            "enabled":       True,
            "spawned":       True,
            "pid":           os.getpid(),
            "cortex_kb_url": "http://kb",
            "interval_sec":  5.0,
            "batch_size":    50,
            "reason":        "spawned",
            "ts":            "2026-05-20T00:00:00+00:00",
            "pid_path":      str(cortex_flusher_pid(session_dir)),
        }),
        encoding="utf-8",
    )
    cortex_flusher_pid(session_dir).write_text(
        f"{os.getpid()}\n", encoding="utf-8",
    )

    warnings: list[str] = []
    out = collect_kb_provenance(
        session_dir=session_dir,
        state={},
        manifest={},
        warnings=warnings,
    )
    fs = out["flusher_status"]
    assert fs["enabled"] is True
    assert fs["spawned"] is True
    assert fs["alive"] is True
    assert fs["pid"] == os.getpid()
    assert "kb_flusher:disabled" not in warnings
    assert "kb_flusher:not_alive" not in warnings


def test_collect_kb_provenance_flusher_status_disabled_emits_warning(session_dir):
    cortex_flusher_status_json(session_dir).parent.mkdir(parents=True, exist_ok=True)
    cortex_flusher_status_json(session_dir).write_text(
        json.dumps({
            "enabled": False,
            "spawned": False,
            "pid":     None,
            "reason":  "cortex_disabled",
            "ts":      "2026-05-20T00:00:00+00:00",
        }),
        encoding="utf-8",
    )
    warnings: list[str] = []
    out = collect_kb_provenance(
        session_dir=session_dir,
        state={},
        manifest={},
        warnings=warnings,
    )
    fs = out["flusher_status"]
    assert fs["enabled"] is False
    assert "kb_flusher:disabled" in warnings


# ---------------------------------------------------------------------------
# CLI flag plumbing
# ---------------------------------------------------------------------------
def test_cli_parser_exposes_no_kb_flusher_flag():
    parser = cli_module._build_parser()
    args = parser.parse_args(["optimize", "--model", "/x", "--no-kb-flusher"])
    assert args.kb_flusher_enabled is False
    args2 = parser.parse_args(["optimize", "--model", "/x"])
    assert args2.kb_flusher_enabled is True
    assert args2.kb_flusher_interval_sec == 5.0
    assert args2.kb_flusher_batch_size == 50
