# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Behavioral tests for geak_runner's graceful-stop / flush contract.

Drive the real call_geak() against a fake runner script so the process-group
signalling and soft/hard timeout split are exercised end to end.
"""
from __future__ import annotations

import importlib.util
import json
import textwrap
from pathlib import Path

import pytest

_RUNNER_PY = (
    Path(__file__).resolve().parents[4]
    / "src" / "hyperloom" / "agents" / "kernel" / "tools" / "backends" / "geak_runner.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("geak_runner", _RUNNER_PY)
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


def test_resolve_runner_falls_back_to_open_source_root(tmp_path, monkeypatch):
    geak_root = tmp_path / "open-source" / "GEAK"
    runner = geak_root / "interface" / "run_e2e.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("# fake\n", encoding="utf-8")
    monkeypatch.delenv("GEAK_E2E_RUNNER", raising=False)
    monkeypatch.delenv("GEAK_ROOT", raising=False)
    monkeypatch.setenv("HYPERLOOM_OPEN_SOURCE_ROOT", str(tmp_path / "open-source"))

    assert psr._resolve_runner() == str(runner)


def test_inner_timeout_is_reduced_by_flush_grace(tmp_path, monkeypatch):
    """run_e2e must receive GEAK_E2E_TIMEOUT_S = timeout_s - flush_grace."""
    runner = _write_fake_runner(tmp_path, """
        import json, os, sys
        # Echo the inner budget we were handed so the test can assert on it.
        result = {"status": "ok", "throughput_speedup": 1.16,
                  "final_throughput_tok_s": 535.4,
                  "inner_budget": os.environ.get("GEAK_E2E_TIMEOUT_S")}
        with open(sys.argv[2], "w") as fh:
            json.dump(result, fh)
        sys.exit(0)
    """)
    monkeypatch.setenv("GEAK_E2E_RUNNER", str(runner))
    monkeypatch.setenv("GEAK_FLUSH_GRACE_S", "180")

    out = psr.call_geak(_handoff(), tmp_path / "out", timeout_s=600)

    assert out["status"] == "ok"
    assert out["inner_budget"] == "420"
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
    monkeypatch.setenv("GEAK_E2E_RUNNER", str(runner))
    monkeypatch.setenv("GEAK_FLUSH_GRACE_S", "5")

    out = psr.call_geak(_handoff(), tmp_path / "out", timeout_s=2)

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
    monkeypatch.setenv("GEAK_E2E_RUNNER", str(runner))
    monkeypatch.setenv("GEAK_FLUSH_GRACE_S", "2")

    out = psr.call_geak(_handoff(), tmp_path / "out", timeout_s=2)

    assert out["status"] == "error"
    assert "no parseable result.json" in out["error"]
    assert out["returncode"] == -1
    assert not (tmp_path / "out" / "result.json").is_file()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
