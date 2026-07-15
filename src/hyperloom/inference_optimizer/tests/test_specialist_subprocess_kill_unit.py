# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for specialist_subprocess process teardown: the SIGTERM/SIGKILL
``_kill`` ladder."""

from __future__ import annotations


from hyperloom.orchestrator.specialists import subprocess_ as ss
from hyperloom.orchestrator.specialists.subprocess_ import (
    SpecialistSubprocessDispatcher,
)


class _FakeProc:
    def __init__(self, poll_seq):
        self._seq = list(poll_seq)
        self._last = self._seq[-1] if self._seq else None
        self.pid = 4242
        self.terminated = False
        self.killed = False

    def poll(self):
        if self._seq:
            return self._seq.pop(0)
        return self._last

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def test_kill_already_exited():
    proc = _FakeProc([0])
    SpecialistSubprocessDispatcher._kill(proc)
    assert proc.terminated is False and proc.killed is False


def test_kill_sigterm_then_exits(monkeypatch):
    # SIGTERM via killpg succeeds, then exits within grace
    proc = _FakeProc([None, None, 0])
    monkeypatch.setattr(ss.os, "getpgid", lambda pid: pid)
    sent = []
    monkeypatch.setattr(ss.os, "killpg", lambda pgid, sig: sent.append(sig))
    monkeypatch.setattr(ss.time, "sleep", lambda s: None)
    SpecialistSubprocessDispatcher._kill(proc)
    assert ss.signal.SIGTERM in sent


def test_kill_killpg_fails_then_terminate_then_sigkill(monkeypatch):
    # killpg raises -> proc.terminate(); SIGKILL path also raises -> proc.kill()
    proc = _FakeProc([None])  # poll always None

    def _getpgid(pid):
        raise ProcessLookupError("gone")

    monkeypatch.setattr(ss.os, "getpgid", _getpgid)
    monkeypatch.setattr(ss.time, "sleep", lambda s: None)
    SpecialistSubprocessDispatcher._kill(proc)
    assert proc.terminated is True
    assert proc.killed is True


def test_kill_sigkill_via_killpg(monkeypatch):
    # getpgid+killpg work -> reaches SIGKILL killpg branch
    proc = _FakeProc([None])
    monkeypatch.setattr(ss.os, "getpgid", lambda pid: pid)
    sent = []
    monkeypatch.setattr(ss.os, "killpg", lambda pgid, sig: sent.append(sig))
    monkeypatch.setattr(ss.time, "sleep", lambda s: None)
    SpecialistSubprocessDispatcher._kill(proc)
    assert ss.signal.SIGKILL in sent


def test_kill_terminate_and_kill_raise(monkeypatch):
    # terminate() and kill() both raise; must be swallowed
    class _RaisingProc(_FakeProc):
        def terminate(self):
            raise RuntimeError("term boom")

        def kill(self):
            raise RuntimeError("kill boom")

    proc = _RaisingProc([None])
    monkeypatch.setattr(ss.os, "getpgid", lambda pid: (_ for _ in ()).throw(ProcessLookupError("x")))
    monkeypatch.setattr(ss.time, "sleep", lambda s: None)
    SpecialistSubprocessDispatcher._kill(proc)  # must not raise
