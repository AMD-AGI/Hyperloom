"""Tests for _kill_process_group's pgid-direct reap logic."""

from __future__ import annotations

import os
import signal
from types import SimpleNamespace

from kernelforge.loop import task_preparer


def _proc(pid):
    killed = {"called": False}

    def kill():
        killed["called"] = True

    return SimpleNamespace(pid=pid, kill=kill), killed


def test_none_pid_is_noop(monkeypatch):
    calls = []
    monkeypatch.setattr(os, "killpg", lambda *a: calls.append(a))
    proc, killed = _proc(None)
    task_preparer._kill_process_group(proc)
    assert calls == []
    assert killed["called"] is False


def test_happy_path_signals_pgid_once(monkeypatch):
    """Under start_new_session pid == pgid, so getpgid returns pid; killpg fires"""
    calls = []
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    proc, killed = _proc(123)
    task_preparer._kill_process_group(proc)
    assert calls == [(123, signal.SIGKILL)]
    assert killed["called"] is False


def test_never_calls_getpgid(monkeypatch):
    """Regression for the PID-reuse leak: the original pid (== pgid at creation"""
    calls = []

    def forbidden(pid):
        raise AssertionError("getpgid must not be called")

    monkeypatch.setattr(os, "getpgid", forbidden)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: calls.append(pgid))
    proc, killed = _proc(123)
    task_preparer._kill_process_group(proc)
    assert calls == [123]
    assert killed["called"] is False


def test_killpg_processlookup_falls_back_to_proc_kill(monkeypatch):
    """killpg 404s -> signalled stays False -> proc.kill() fallback fires."""

    def dead(pgid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(os, "killpg", dead)
    proc, killed = _proc(555)
    task_preparer._kill_process_group(proc)
    assert killed["called"] is True


def test_killpg_generic_exception_falls_back_to_proc_kill(monkeypatch):
    def broken(pgid, sig):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(os, "killpg", broken)
    proc, killed = _proc(999)
    task_preparer._kill_process_group(proc)
    assert killed["called"] is True


def test_permission_error_continues_then_fallback(monkeypatch):
    def denied(pgid, sig):
        raise PermissionError

    monkeypatch.setattr(os, "killpg", denied)
    proc, killed = _proc(321)
    task_preparer._kill_process_group(proc)
    assert killed["called"] is True
