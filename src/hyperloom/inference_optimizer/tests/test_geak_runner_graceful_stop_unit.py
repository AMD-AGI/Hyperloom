# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Behavioral tests for geak_runner's graceful-stop / flush contract."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
import textwrap
import time
from pathlib import Path

import pytest

from hyperloom.orchestrator.kernel import geak_runner as psr


def _write_fake_runner(tmp_path: Path, body: str) -> Path:
    """A fake run_e2e.py: argv = <handoff> <result>. Body decides behavior."""
    f = tmp_path / "fake_run_e2e.py"
    f.write_text(textwrap.dedent(body), encoding="utf-8")
    return f


def _handoff() -> dict:
    return {
        "schema_version": 1,
        "model_path": "/m",
        "framework": "vllm",
        "tp": 1,
        "workload": {"isl": 8, "osl": 8, "conc": 1},
        "exp_root": "/tmp/x",
    }


def test_resolve_runner_falls_back_to_cache_dir(tmp_path, monkeypatch):
    geak_root = tmp_path / "cache" / "GEAK"
    runner = geak_root / "interface" / "run_e2e.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("# fake\n", encoding="utf-8")
    monkeypatch.delenv("GEAK_E2E_RUNNER", raising=False)
    monkeypatch.delenv("GEAK_ROOT", raising=False)
    monkeypatch.setenv("HYPERLOOM_CACHE_DIR", str(tmp_path / "cache"))

    assert psr._resolve_runner() == str(runner)


def test_resolve_runner_prefers_newest_pinned_geak_in_cache(tmp_path, monkeypatch):
    import os as _os

    cache = tmp_path / "cache"
    old_runner = cache / "GEAK@1111111" / "interface" / "run_e2e.py"
    new_runner = cache / "GEAK@2222222" / "interface" / "run_e2e.py"
    for r in (old_runner, new_runner):
        r.parent.mkdir(parents=True)
        r.write_text("# fake\n", encoding="utf-8")
    _os.utime(old_runner.parent.parent, (1_000_000, 1_000_000))
    _os.utime(new_runner.parent.parent, (2_000_000, 2_000_000))
    monkeypatch.delenv("GEAK_E2E_RUNNER", raising=False)
    monkeypatch.delenv("GEAK_ROOT", raising=False)
    monkeypatch.setenv("HYPERLOOM_CACHE_DIR", str(cache))

    assert psr._resolve_runner() == str(new_runner)


def test_inner_timeout_is_reduced_by_flush_grace(tmp_path, monkeypatch):
    """run_e2e must receive GEAK_E2E_TIMEOUT_S = timeout_s - flush_grace."""
    runner = _write_fake_runner(
        tmp_path,
        """
        import json, os, sys
        # Echo the inner budget we were handed so the test can assert on it.
        result = {"status": "ok", "throughput_speedup": 1.16,
                  "final_throughput_tok_s": 535.4,
                  "inner_budget": os.environ.get("GEAK_E2E_TIMEOUT_S")}
        with open(sys.argv[2], "w") as fh:
            json.dump(result, fh)
        sys.exit(0)
    """,
    )
    monkeypatch.setenv("GEAK_E2E_RUNNER", str(runner))
    monkeypatch.setenv("GEAK_FLUSH_GRACE_S", "180")

    out = psr.call_geak(_handoff(), tmp_path / "out", timeout_s=600)

    assert out["status"] == "ok"
    assert out["inner_budget"] == "420"
    assert out["returncode"] == 0


def test_call_geak_uses_current_python_when_path_is_polluted(tmp_path, monkeypatch):
    """GEAK must run in Hyperloom's Python, not a framework venv from PATH."""
    fake_vllm_bin = tmp_path / "vllm-venv" / "bin"
    fake_vllm_bin.mkdir(parents=True)
    fake_python = fake_vllm_bin / "python3"
    fake_python.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    fake_python.chmod(0o755)

    runner = _write_fake_runner(
        tmp_path,
        """
        import json, sys
        with open(sys.argv[2], "w") as fh:
            json.dump({"status": "ok", "python": sys.executable}, fh)
        sys.exit(0)
    """,
    )
    monkeypatch.setenv("GEAK_E2E_RUNNER", str(runner))
    monkeypatch.setenv(
        "PATH",
        str(fake_vllm_bin) + os.pathsep + os.environ.get("PATH", ""),
    )

    out = psr.call_geak(_handoff(), tmp_path / "out", timeout_s=600)

    assert out["status"] == "ok"
    assert out["python"] == sys.executable


def test_sigterm_grace_lets_child_flush_result(tmp_path, monkeypatch):
    """On the hard-timeout path, SIGTERM gives the child time to flush; the flushed result.json is then read back (not discarded as no_result_json)."""
    runner = _write_fake_runner(
        tmp_path,
        """
        import json, signal, sys, time
        handoff, result_path = sys.argv[1], sys.argv[2]
        def _flush(signum, frame):
            with open(result_path, "w") as fh:
                json.dump({"status": "ok", "throughput_speedup": 1.16,
                           "flushed_on_term": True}, fh)
            sys.exit(0)
        signal.signal(signal.SIGTERM, _flush)
        # Outlive the outer hard timeout so the SIGTERM path triggers.
        time.sleep(60)
    """,
    )
    monkeypatch.setenv("GEAK_E2E_RUNNER", str(runner))
    monkeypatch.setenv("GEAK_FLUSH_GRACE_S", "5")

    out = psr.call_geak(_handoff(), tmp_path / "out", timeout_s=2)

    assert out["status"] == "ok"
    assert out["flushed_on_term"] is True
    rp = json.loads((tmp_path / "out" / "result.json").read_text())
    assert rp["flushed_on_term"] is True


def test_failed_run_writes_its_error_last_on_stderr(tmp_path, monkeypatch, capsys):
    """The kernel phase logs only the tail of the runner's stderr, so the error must end it."""
    handoff = tmp_path / "handoff.json"
    handoff.write_text(json.dumps(_handoff()), encoding="utf-8")
    monkeypatch.setattr(
        psr,
        "call_geak",
        lambda *_a, **_k: {
            "status": "error",
            "error": "runtime ERROR: R9600D serving policy is unvalidated",
            "stderr_tail": "x" * 5000,
            "result_path": str(tmp_path / "result.json"),
        },
    )

    assert psr.main([str(handoff), str(tmp_path / "out"), "--timeout-s", "60"]) == 1
    err = capsys.readouterr().err
    assert err.rstrip().endswith("runtime ERROR: R9600D serving policy is unvalidated")


def test_timed_out_run_fails_and_writes_its_error_on_stderr(tmp_path, monkeypatch, capsys):
    handoff = tmp_path / "handoff.json"
    handoff.write_text(json.dumps(_handoff()), encoding="utf-8")
    monkeypatch.setattr(
        psr,
        "call_geak",
        lambda *_a, **_k: {"status": "timeout", "error": "workflow exceeded 3600s", "stderr_tail": ""},
    )

    assert psr.main([str(handoff), str(tmp_path / "out"), "--timeout-s", "60"]) == 1
    assert capsys.readouterr().err.rstrip().endswith("workflow exceeded 3600s")


def test_failed_run_redacts_credentials_on_stderr(tmp_path, monkeypatch, capsys):
    handoff = tmp_path / "handoff.json"
    handoff.write_text(json.dumps(_handoff()), encoding="utf-8")
    monkeypatch.setattr(
        psr,
        "call_geak",
        lambda *_a, **_k: {
            "status": "error",
            "error": "runtime (node) failed (rc=1): Authorization: Bearer abcdef0123456789",
            "stderr_tail": "Ocp-Apim-Subscription-Key: s3cr3tvalue",
        },
    )

    assert psr.main([str(handoff), str(tmp_path / "out"), "--timeout-s", "60"]) == 1
    err = capsys.readouterr().err
    assert "abcdef0123456789" not in err
    assert "s3cr3tvalue" not in err
    assert "[REDACTED]" in err


def test_successful_run_writes_nothing_on_stderr(tmp_path, monkeypatch, capsys):
    handoff = tmp_path / "handoff.json"
    handoff.write_text(json.dumps(_handoff()), encoding="utf-8")
    monkeypatch.setattr(
        psr,
        "call_geak",
        lambda *_a, **_k: {"status": "ok", "throughput_speedup": 1.1, "stderr_tail": "noise", "error": ""},
    )

    assert psr.main([str(handoff), str(tmp_path / "out"), "--timeout-s", "60"]) == 0
    assert capsys.readouterr().err == ""


def test_sigkill_escalation_when_child_ignores_sigterm(tmp_path, monkeypatch):
    """A child that ignores SIGTERM and never flushes is SIGKILLed; the runner reports a no-result error rather than hanging forever."""
    runner = _write_fake_runner(
        tmp_path,
        """
        import signal, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # refuse to die politely
        time.sleep(120)
    """,
    )
    monkeypatch.setenv("GEAK_E2E_RUNNER", str(runner))
    monkeypatch.setenv("GEAK_FLUSH_GRACE_S", "2")

    out = psr.call_geak(_handoff(), tmp_path / "out", timeout_s=2)

    assert out["status"] == "error"
    assert "no parseable result.json" in out["error"]
    assert out["returncode"] == -1
    assert not (tmp_path / "out" / "result.json").is_file()


def test_sigkill_drain_is_bounded_when_an_escaped_descendant_holds_the_pipes(tmp_path, monkeypatch):
    """A descendant that left the process group keeps stdout open past SIGKILL; the runner must still return."""
    pid_file = tmp_path / "escaped.pid"
    runner = _write_fake_runner(
        tmp_path,
        f"""
        import signal, subprocess, sys, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        escaped = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
        open({str(pid_file)!r}, "w").write(str(escaped.pid))
        print("partial output", flush=True)
        time.sleep(120)
    """,
    )
    monkeypatch.setenv("GEAK_E2E_RUNNER", str(runner))
    monkeypatch.setenv("GEAK_FLUSH_GRACE_S", "1")

    started = time.monotonic()
    try:
        out = psr.call_geak(_handoff(), tmp_path / "out", timeout_s=2)
    finally:
        if pid_file.is_file():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pid_file.read_text()), signal.SIGKILL)

    assert time.monotonic() - started < 30
    assert out["status"] == "error"
    assert out["returncode"] == -1
    assert "partial output" in out["stdout_tail"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
