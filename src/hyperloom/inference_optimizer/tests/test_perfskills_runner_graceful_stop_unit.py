# Copyright Advanced Micro Devices, Inc. All rights reserved.
"""Behavioral tests for perfskills_runner's graceful-stop / flush contract.

Regression guard for the bug where the runner's inner soft deadline
(PERFSKILLS_E2E_TIMEOUT_S, consumed by run_e2e's anyio.fail_after) and the outer
hard subprocess kill used the SAME value, so result.json was SIGKILLed mid-flush
-> "no_result_json" and the measured win was lost.

These drive the REAL call_perfskills() against a fake runner script so the
process-group signalling + soft/hard split are exercised end to end.
"""
from __future__ import annotations

import importlib.util
import json
import textwrap
from pathlib import Path

import pytest

_RUNNER_PY = (
    Path(__file__).resolve().parents[4]
    / "kernel-agent" / "tools" / "backends" / "perfskills_runner.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("perfskills_runner", _RUNNER_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


psr = _load_module()


def _write_fake_runner(tmp_path: Path, body: str) -> Path:
    """A fake run_e2e.py: argv = <handoff> <result>. Body decides behavior."""
    f = tmp_path / "fake_run_e2e.py"
    f.write_text(textwrap.dedent(body), encoding="utf-8")
    return f


def _handoff() -> dict:
    return {"schema_version": 1, "model_path": "/m", "framework": "vllm",
            "tp": 1, "workload": {"isl": 8, "osl": 8, "conc": 1},
            "exp_root": "/tmp/x"}


def test_inner_timeout_is_reduced_by_flush_grace(tmp_path, monkeypatch):
    """run_e2e must receive PERFSKILLS_E2E_TIMEOUT_S = timeout_s - flush_grace."""
    runner = _write_fake_runner(tmp_path, """
        import json, os, sys
        # Echo the inner budget we were handed so the test can assert on it.
        result = {"status": "ok", "throughput_speedup": 1.16,
                  "final_throughput_tok_s": 535.4,
                  "inner_budget": os.environ.get("PERFSKILLS_E2E_TIMEOUT_S")}
        with open(sys.argv[2], "w") as fh:
            json.dump(result, fh)
        sys.exit(0)
    """)
    monkeypatch.setenv("PERFSKILLS_E2E_RUNNER", str(runner))
    monkeypatch.setenv("PERFSKILLS_FLUSH_GRACE_S", "180")

    out = psr.call_perfskills(_handoff(), tmp_path / "out", timeout_s=600)

    assert out["status"] == "ok"
    assert out["inner_budget"] == "420"  # 600 - 180
    assert out["returncode"] == 0


def test_sigterm_grace_lets_child_flush_result(tmp_path, monkeypatch):
    """On the hard-timeout path, SIGTERM gives the child time to flush; the
    flushed result.json is then read back (not discarded as no_result_json)."""
    runner = _write_fake_runner(tmp_path, """
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
    """)
    monkeypatch.setenv("PERFSKILLS_E2E_RUNNER", str(runner))
    monkeypatch.setenv("PERFSKILLS_FLUSH_GRACE_S", "5")

    # timeout_s small so the outer communicate() times out quickly and SIGTERMs.
    out = psr.call_perfskills(_handoff(), tmp_path / "out", timeout_s=2)

    assert out["status"] == "ok"
    assert out["flushed_on_term"] is True
    rp = json.loads((tmp_path / "out" / "result.json").read_text())
    assert rp["flushed_on_term"] is True


def test_sigkill_escalation_when_child_ignores_sigterm(tmp_path, monkeypatch):
    """A child that ignores SIGTERM and never flushes is SIGKILLed; the runner
    reports a no-result error rather than hanging forever."""
    runner = _write_fake_runner(tmp_path, """
        import signal, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # refuse to die politely
        time.sleep(120)
    """)
    monkeypatch.setenv("PERFSKILLS_E2E_RUNNER", str(runner))
    monkeypatch.setenv("PERFSKILLS_FLUSH_GRACE_S", "2")

    out = psr.call_perfskills(_handoff(), tmp_path / "out", timeout_s=2)

    assert out["status"] == "error"
    assert "no parseable result.json" in out["error"]
    assert out["returncode"] == -1
    assert not (tmp_path / "out" / "result.json").is_file()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
