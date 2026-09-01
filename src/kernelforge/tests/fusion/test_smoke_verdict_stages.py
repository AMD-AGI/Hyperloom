# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The smoke knows which stage failed; nobody should re-derive it from a message.

``classify_serving_smoke_failure`` read the reason string back and matched
substrings, which cannot tell a boot-time HIP OOM from a fused-kernel fault
(both carry "HIP error") nor an HTTP probe error from a crashed scheduler (both
were spelled "decode probe failed"). The verdict carries the stage the smoke was
in and whether the kernel is implicated, decided where the evidence is.
"""

from __future__ import annotations

from kernelforge.fusion import validate
from kernelforge.fusion.validate import (
    SMOKE_STAGE_BOOT_TIMEOUT,
    SMOKE_STAGE_DECODE_BENCH,
    SMOKE_STAGE_DECODE_CRASH,
    SMOKE_STAGE_DECODE_HANG,
    SMOKE_STAGE_DECODE_PROBE,
    SMOKE_STAGE_HARNESS_ERROR,
    SMOKE_STAGE_STARTUP_CRASH,
    serving_smoke_verdict,
)


class _Proc:
    def __init__(self, stdout="", stderr="", rc=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = rc


class _FakeServer:
    def __init__(self, poll_seq):
        self._poll = iter(poll_seq)
        self.returncode = 1
        self.pid = 4242

    def poll(self):
        try:
            return next(self._poll)
        except StopIteration:
            return None


def _patch_smoke(monkeypatch, tmp_path, *, tail, poll_seq, run=None):
    monkeypatch.setattr(validate, "_runtime_dir", lambda kind: tmp_path)
    monkeypatch.setattr(validate.os, "killpg", lambda *a, **k: None)
    monkeypatch.setattr(validate.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(validate.subprocess, "run", run or (lambda *a, **k: _Proc()))
    monkeypatch.setattr(validate.subprocess, "Popen", lambda *a, **k: _FakeServer(poll_seq))
    monkeypatch.setattr(validate, "_tail_text", tail if callable(tail) else (lambda *a, **k: tail))
    monkeypatch.setattr(validate, "_full_log_text", lambda *a, **k: "")
    import time

    monkeypatch.setattr(time, "sleep", lambda *_: None)


def test_a_boot_time_oom_is_the_environment_not_the_kernel(monkeypatch, tmp_path):
    """Repro: "HIP error: out of memory" during boot used to blame the kernel."""
    _patch_smoke(
        monkeypatch,
        tmp_path,
        tail="Initializing a V1 LLM engine\nRuntimeError: HIP error: out of memory\n",
        poll_seq=[0],
    )

    verdict = serving_smoke_verdict("/m", {"F": "1"}, framework="vllm", timeout_s=5, log_path=str(tmp_path / "a.log"))

    assert verdict.ok is False
    assert verdict.stage == SMOKE_STAGE_STARTUP_CRASH
    assert verdict.blames_kernel is False


def test_a_boot_time_gpu_fault_does_blame_the_kernel(monkeypatch, tmp_path):
    """CUDA-graph capture happens at boot, so a fault there IS the kernel."""
    _patch_smoke(
        monkeypatch,
        tmp_path,
        tail="capturing graphs\nMemory access fault by GPU node-1 on address 0x7f00\n",
        poll_seq=[None, None, 1],
    )

    verdict = serving_smoke_verdict("/m", {"F": "1"}, framework="vllm", timeout_s=5, log_path=str(tmp_path / "b.log"))

    assert verdict.stage == SMOKE_STAGE_STARTUP_CRASH
    assert verdict.blames_kernel is True


def test_gpu_fault_detection_is_case_insensitive(monkeypatch, tmp_path):
    """ROCm logs do not use stable capitalization for hardware faults."""
    _patch_smoke(
        monkeypatch,
        tmp_path,
        tail="capturing graphs\nMEMORY ACCESS FAULT by GPU node-1\n",
        poll_seq=[None, None, 1],
    )

    verdict = serving_smoke_verdict(
        "/m", {"F": "1"}, framework="vllm", timeout_s=5, log_path=str(tmp_path / "case.log")
    )

    assert verdict.stage == SMOKE_STAGE_STARTUP_CRASH
    assert verdict.blames_kernel is True
    assert "MEMORY ACCESS FAULT" in verdict.reason


def test_a_boot_timeout_is_never_the_kernel(monkeypatch, tmp_path):
    _patch_smoke(monkeypatch, tmp_path, tail="still loading weights\n", poll_seq=[None, None])

    # timeout_s=0 makes the readiness deadline expire immediately.
    verdict = serving_smoke_verdict("/m", {"F": "1"}, framework="vllm", timeout_s=0, log_path=str(tmp_path / "c.log"))

    assert verdict.stage == SMOKE_STAGE_BOOT_TIMEOUT
    assert verdict.blames_kernel is False


def test_a_probe_transport_error_is_not_the_kernel(monkeypatch, tmp_path):
    """The server is up and unfaulted; the probe could not reach it."""
    _patch_smoke(monkeypatch, tmp_path, tail="Application startup complete.\n", poll_seq=[None, None, None])
    monkeypatch.setattr(
        validate,
        "_vllm_decode_probe",
        lambda *a, **k: (False, "/v1/models probe error: OSError: boom"),
    )

    verdict = serving_smoke_verdict("/m", {"F": "1"}, framework="vllm", timeout_s=5, log_path=str(tmp_path / "d.log"))

    assert verdict.stage == SMOKE_STAGE_DECODE_PROBE
    assert verdict.blames_kernel is False


def test_a_fault_during_decode_blames_the_kernel(monkeypatch, tmp_path):
    ready = "Application startup complete.\n"
    _patch_smoke(
        monkeypatch,
        tmp_path,
        tail=ready + "HSA_STATUS_ERROR_EXCEPTION hardware exception\n",
        poll_seq=[None, None, None],
    )
    monkeypatch.setattr(validate, "_vllm_decode_probe", lambda *a, **k: (True, "ok"))

    verdict = serving_smoke_verdict("/m", {"F": "1"}, framework="vllm", timeout_s=5, log_path=str(tmp_path / "e.log"))

    assert verdict.stage == SMOKE_STAGE_DECODE_CRASH
    assert verdict.blames_kernel is True


def test_an_oom_death_during_decode_is_still_the_environment(monkeypatch, tmp_path):
    """Running out of KV memory mid-run says nothing about kernel correctness."""
    ready = "Application startup complete.\n"
    _patch_smoke(
        monkeypatch,
        tmp_path,
        tail=ready + "RuntimeError: HIP error: out of memory\n",
        poll_seq=[None, None, 1],
    )
    monkeypatch.setattr(validate, "_vllm_decode_probe", lambda *a, **k: (False, "no tokens"))

    verdict = serving_smoke_verdict("/m", {"F": "1"}, framework="vllm", timeout_s=5, log_path=str(tmp_path / "f.log"))

    assert verdict.stage == SMOKE_STAGE_DECODE_CRASH
    assert verdict.blames_kernel is False


def test_a_decode_hang_blames_the_kernel(monkeypatch, tmp_path):
    """A ready server that stops answering is the fused kernel's problem."""
    import subprocess as _sp

    def boom(cmd, *a, **k):
        if any("bench_serving" in str(c) for c in cmd):
            raise _sp.TimeoutExpired(cmd=cmd, timeout=5)
        return _Proc()

    _patch_smoke(
        monkeypatch,
        tmp_path,
        tail="The server is fired up and ready to roll!\n",
        poll_seq=[None, None, None],
        run=boom,
    )

    verdict = serving_smoke_verdict("/m", {"F": "1"}, framework="sglang", timeout_s=5, log_path=str(tmp_path / "g.log"))

    assert verdict.stage == SMOKE_STAGE_DECODE_HANG
    assert verdict.blames_kernel is True


def test_a_bench_that_could_not_run_is_not_the_kernel(monkeypatch, tmp_path):
    def run(cmd, *a, **k):
        if any("bench_serving" in str(c) for c in cmd):
            return _Proc(stderr="ModuleNotFoundError: No module named 'sglang.bench_serving'", rc=1)
        return _Proc()

    _patch_smoke(
        monkeypatch,
        tmp_path,
        tail="The server is fired up and ready to roll!\n",
        poll_seq=[None, None, None],
        run=run,
    )

    verdict = serving_smoke_verdict("/m", {"F": "1"}, framework="sglang", timeout_s=5, log_path=str(tmp_path / "h.log"))

    assert verdict.stage == SMOKE_STAGE_DECODE_BENCH
    assert verdict.blames_kernel is False


def test_a_harness_error_is_not_the_kernel(monkeypatch, tmp_path):
    monkeypatch.setattr(validate, "_runtime_dir", lambda kind: tmp_path)
    monkeypatch.setattr(validate.subprocess, "run", lambda *a, **k: _Proc())

    def boom(*a, **k):
        raise RuntimeError("popen exploded")

    monkeypatch.setattr(validate.subprocess, "Popen", boom)
    import time

    monkeypatch.setattr(time, "sleep", lambda *_: None)

    verdict = serving_smoke_verdict("/m", {"F": "1"}, timeout_s=1, log_path=str(tmp_path / "i.log"))

    assert verdict.stage == SMOKE_STAGE_HARNESS_ERROR
    assert verdict.blames_kernel is False


def test_serving_smoke_still_returns_the_two_tuple_callers_expect(monkeypatch, tmp_path):
    """The compile-pass A/B unpacks ``(ok, reason)``; keep that shape."""
    _patch_smoke(
        monkeypatch,
        tmp_path,
        tail="Application startup complete.\n",
        poll_seq=[None, None, None],
    )
    monkeypatch.setattr(validate, "_vllm_decode_probe", lambda *a, **k: (True, "ok"))

    ok, reason = validate.serving_smoke(
        "/m", {"F": "1"}, framework="vllm", timeout_s=5, log_path=str(tmp_path / "j.log")
    )

    assert ok is True
    assert "survives" in reason
