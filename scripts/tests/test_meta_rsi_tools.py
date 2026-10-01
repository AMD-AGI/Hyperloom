# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests for the Meta RSI analysis scripts the driver calls."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from meta_rsi.compare_ab import compare
from meta_rsi.rsi.pipeline import StepFailed
from meta_rsi.rsi.steps.check import failed_tests
from meta_rsi.rsi.steps.data import scenario_args

SCRIPTS = Path(__file__).resolve().parents[1] / "meta_rsi"
PHASES = [{"to_phase": p} for p in ("PRELUDE", "FRAMEWORK_AGENT", "KERNEL_AGENT", "SWEEP", "CLOSE")]


@pytest.mark.parametrize("script", ["targets.py", "build_plan.py", "an_ledger.py", "select_scenario.py"])
def test_scripts_name_the_missing_round_variable_instead_of_guessing_a_path(script):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PULSE_")}
    proc = subprocess.run([sys.executable, str(SCRIPTS / script)], cwd=SCRIPTS, env=env, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "is not set" in proc.stderr and "PULSE_" in proc.stderr


class TestScenarioArgs:
    MANIFEST = {
        "model_path": "/models/Qwen",
        "framework": "vllm",
        "gpu_type": "mi300x",
        "tp": 2,
        "ep": 1,
        "workload": {"conc": 64, "isl": 8192, "osl": 1024},
        "objective": {"kind": "gain_pct", "value": 300.0},
    }

    def test_a_manifest_becomes_the_optimize_arguments(self):
        args = scenario_args(self.MANIFEST)
        assert args[:4] == ["--model", "/models/Qwen", "--framework", "vllm"]
        assert args[args.index("--isl") + 1] == "8192" and args[-2:] == ["--target-gain", "300.0"]
        assert "--ep" not in args

    def test_expert_parallelism_is_carried_when_used(self):
        assert scenario_args({**self.MANIFEST, "ep": 8})[-4:-2] == ["--ep", "8"]

    def test_a_manifest_without_the_workload_is_rejected(self):
        with pytest.raises(StepFailed, match="workload|conc"):
            scenario_args({**self.MANIFEST, "workload": {}})


def _arm(root: Path, gain: float, rows: list[dict], stop: str = "sweep_done") -> Path:
    sess = root / "Qwen" / "20261001T000000Z-abc"
    (sess / "reports/trace").mkdir(parents=True)
    state = {"cumulative_gain_validated": gain, "stop_reason": stop, "phase_history": PHASES}
    (sess / "state.json").write_text(json.dumps(state))
    (sess / "reports/trace/llm_calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def _row(component: str, model: str, cache_read: int, output: int) -> dict:
    return {"component": component, "model": model, "cache_read_input_tokens": cache_read, "output_tokens": output}


class TestCompare:
    def test_cheaper_arm_within_the_gain_band_meets_the_rule(self, tmp_path):
        a = _arm(tmp_path / "A", 9.0, [_row("orchestration", "claude-opus-5", 1_000_000, 10_000)])
        b = _arm(tmp_path / "B", 8.0, [_row("orchestration", "claude-opus-5", 500_000, 5_000)])
        result = compare(a, b)
        v = result["verdict"]
        assert v["effect_kept"] and v["savings"] and v["pass"]
        assert v["opus_weighted_saving_pct"] == 50.0
        assert result["A"]["furthest_phase"] == "SWEEP" and result["B"]["router"] == {}

    def test_a_gain_below_the_band_or_a_new_failing_stop_breaks_the_rule(self, tmp_path):
        a = _arm(tmp_path / "A", 9.15, [_row("orchestration", "claude-opus-5", 100, 10)])
        low = _arm(tmp_path / "B", 4.85, [_row("orchestration", "claude-opus-5", 10, 1)])
        failing = _arm(tmp_path / "C", 9.0, [_row("orchestration", "claude-opus-5", 10, 1)], stop="baseline_failed")
        assert compare(a, low)["verdict"]["gain_ok"] is False
        assert compare(a, failing)["verdict"]["new_failure"] is True

    def test_router_requests_after_the_later_arms_last_call_are_not_its_own(self, tmp_path):
        a = _arm(tmp_path / "A", 5.0, [_row("orchestration", "claude-opus-5", 10, 1)])
        b_row = {**_row("specialist", "glm-5-3", 10, 1), "ts": "2026-10-01T02:00:00Z"}
        b = _arm(tmp_path / "B", 5.0, [b_row])
        (b / "Qwen" / "20261001T000000Z-abc").rename(b / "Qwen" / "20261001T010000Z-abc")
        log = tmp_path / "router.jsonl"
        stamps = ("2026-10-01T00:30:00Z", "2026-10-01T01:30:00Z", "2026-10-01T02:00:30Z", "2026-10-01T05:00:00Z")
        log.write_text("".join(json.dumps({"ts": ts, "route": "infera", "status": 200}) + "\n" for ts in stamps))
        result = compare(a, b, log)
        assert result["A"]["router"]["infera_requests"] == 1
        assert result["B"]["router"]["infera_requests"] == 2

    def test_self_hosted_glm_tokens_are_counted_but_not_priced(self, tmp_path):
        a = _arm(tmp_path / "A", 5.0, [_row("specialist", "claude-opus-5", 1000, 100)])
        b = _arm(tmp_path / "B", 5.0, [_row("specialist", "glm-5-3", 1000, 100)])
        result = compare(a, b)
        assert result["B"]["cost_usd"] == 0.0 and result["B"]["glm_raw_tokens"] == 1100


def test_failed_tests_come_from_the_short_test_summary():
    output = "\n".join(
        [
            "..F.E",
            "=========================== short test summary info ============================",
            "FAILED tests/test_a.py::test_bad[1-2] - assert 1 == 2",
            "ERROR tests/test_b.py::test_err - RuntimeError: boom",
            "ERROR tests/test_c.py",
            "1 failed, 3 passed, 2 errors in 0.12s",
        ]
    )
    assert failed_tests(output) == {"tests/test_a.py::test_bad[1-2]", "tests/test_b.py::test_err", "tests/test_c.py"}
    assert failed_tests("4 passed in 0.1s") == set()
