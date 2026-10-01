# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests for the Meta RSI analysis scripts the driver calls."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from meta_rsi import pulse
from meta_rsi.compare_ab import compare
from meta_rsi.rsi.config import parse_config
from meta_rsi.rsi.pipeline import RoundContext, StepFailed
from meta_rsi.rsi.state import RoundState
from meta_rsi.rsi.steps import data
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


class TestPulseExits:
    @pytest.mark.parametrize(
        ("raised", "status"),
        [(None, 0), (pulse.Deferred("timed out"), pulse.EXIT_PARTIAL), (pulse.AuthError("401"), pulse.EXIT_STOPPED)],
    )
    def test_census_tells_complete_partial_and_stopped_runs_apart(self, tmp_path, monkeypatch, raised, status):
        class Api:
            class outage:
                pauses = 0

            def ls(self, name):
                if raised:
                    raise raised
                return {"archive_status": "ok", "files": []}

        monkeypatch.setattr(pulse, "Pulse", Api)
        targets = tmp_path / "targets.txt"
        targets.write_text("a\nb\n")
        args = argparse.Namespace(targets=str(targets), out=str(tmp_path / "ls.jsonl.gz"), jobs=2)
        assert pulse.cmd_census(args) == status

    @pytest.mark.parametrize(
        ("codes", "status", "tiers_run"), [("0 0 0", 0, 3), ("0 3 0", 3, 3), ("0 2 0", 2, 2), ("1", 1, 1)]
    )
    def test_fetch_all_passes_the_fetch_status_on_and_stops_at_a_tier_that_stopped(
        self, tmp_path, codes, status, tiers_run
    ):
        stub = tmp_path / "bin" / "python3"
        stub.parent.mkdir()
        stub.write_text(
            '#!/usr/bin/env bash\nn=$(wc -l < "$CALLS")\necho "$*" >> "$CALLS"\ncodes=($STUB_CODES)\nexit "${codes[$n]:-0}"\n'
        )
        stub.chmod(0o755)
        calls = tmp_path / "calls.txt"
        calls.write_text("")
        env = {k: v for k, v in os.environ.items() if k not in ("TIERS", "PULSE_KEY_FILE")}
        env.update(
            PATH=f"{stub.parent}:{os.environ['PATH']}",
            PULSE_ROUND_DIR=str(tmp_path),
            PULSE_BUNDLES=str(tmp_path / "bundles"),
            PULSE_API_KEY="ak-test",
            CALLS=str(calls),
            STUB_CODES=codes,
        )
        proc = subprocess.run(["bash", str(SCRIPTS / "fetch_all.sh")], env=env, capture_output=True, text=True)
        assert proc.returncode == status, proc.stdout + proc.stderr
        assert len(calls.read_text().splitlines()) == tiers_run


class ScriptRunner:
    """Answers the fetch step's commands: Pulse census runs and fetch_all.sh exit as told, the rest succeed."""

    def __init__(self, census: int, bundles: int):
        self.census, self.bundles = census, bundles

    def run(self, cmd, check=True, **_kwargs):
        if cmd[-1].endswith("fetch_all.sh"):
            code = self.bundles
        else:
            code = self.census if cmd[2:3] in (["census"], ["census-retry"]) else 0
        if check and code:
            raise StepFailed(f"{cmd[1]} exited {code}")
        return subprocess.CompletedProcess(cmd, code, "", "boom\n" if code else "")


@pytest.mark.parametrize(
    ("census", "bundles", "outcome"),
    [(0, 0, False), (3, 0, True), (0, 3, True), (2, 0, "pulse.py census exited 2"), (0, 2, "fetch_all.sh exited 2")],
)
def test_fetch_goes_on_past_deferred_items_and_stops_on_a_stopped_run(
    rsi_config_dict, monkeypatch, census, bundles, outcome
):
    monkeypatch.setenv("PULSE_API_KEY", "ak-test")
    cfg = parse_config(rsi_config_dict)
    ctx = RoundContext(config=cfg, state=RoundState.load(cfg.round_dir), log=lambda _m: None)
    ctx.runner = ScriptRunner(census, bundles)
    cfg.round_dir.mkdir(parents=True)
    (cfg.round_dir / "targets_recent.txt").write_text("s1\ns2\n")
    if isinstance(outcome, bool):
        assert data.fetch(ctx) == {"recent_sessions": 2, "partial": outcome}
    else:
        with pytest.raises(StepFailed, match=outcome):
            data.fetch(ctx)


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
