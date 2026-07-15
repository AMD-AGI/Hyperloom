# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the bypass analysis layer (pure functions, no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.orchestrator.actions.executors import bypass_analysis as ba


def test_parse_server_log_throughput_sglang_and_vllm():
    text = "\n".join([
        "Decode batch. #running-req: 4 gen throughput (token/s): 100.0",
        "Decode batch. #running-req: 4 gen throughput (token/s): 200.5",
        "Avg generation throughput: 300 tokens/s",
        "irrelevant line",
        "gen throughput (token/s): 0",  # non-positive dropped
    ])
    samples = ba.parse_server_log_throughput(text)
    assert samples == [100.0, 200.5, 300.0]


def test_steady_state_mean_drops_warmup():
    samples = [10.0, 10.0, 100.0, 100.0, 100.0]
    # 20% warmup -> drop 1 leading sample -> mean of [10,100,100,100]
    assert ba.steady_state_mean(samples, warmup_skip_frac=0.2) == 77.5


def test_steady_state_mean_empty_is_none():
    assert ba.steady_state_mean([]) is None
    assert ba.steady_state_mean([0.0, -1.0]) is None


def test_classify_failure_priority():
    assert ba.classify_failure("HIP out of memory while allocating") == "oom"
    assert ba.classify_failure("Failed to capture cuda graph") == "cuda_graph_capture"
    assert ba.classify_failure("bind: Address already in use") == "port_conflict"
    assert ba.classify_failure("watchdog timeout on detokenizer") == "detokenizer_stall"
    assert ba.classify_failure("Traceback (most recent call last):") == "server_init_dead"
    assert ba.classify_failure("nothing notable") is None
    assert ba.classify_failure("") is None


def test_estimate_steady_state_from_log(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(
        "gen throughput (token/s): 50\ngen throughput (token/s): 150\n",
        encoding="utf-8",
    )
    out = ba.estimate_steady_state_from_log(log, warmup_skip_frac=0.0)
    assert out["sample_count"] == 2
    assert out["steady_state_output_throughput"] == 100.0


def test_estimate_steady_state_missing_log(tmp_path):
    out = ba.estimate_steady_state_from_log(tmp_path / "nope.log")
    assert out["sample_count"] == 0
    assert out["steady_state_output_throughput"] is None


def test_summarize_eval(tmp_path):
    eval_dir = tmp_path / "lm_eval" / "run1"
    eval_dir.mkdir(parents=True)
    (eval_dir / "results_2026.json").write_text(
        json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.83}}}),
        encoding="utf-8",
    )
    out = ba.summarize_eval(tmp_path)
    assert out is not None
    assert out["task"] == "gsm8k"
    assert out["metric"] == "exact_match,strict-match"
    assert out["accuracy"] == 0.83


def test_summarize_eval_none_when_missing(tmp_path):
    assert ba.summarize_eval(tmp_path) is None


def test_build_analysis_success_with_eval(tmp_path):
    (tmp_path / "server.log").write_text(
        "gen throughput (token/s): 100\ngen throughput (token/s): 100\n",
        encoding="utf-8",
    )
    eval_dir = tmp_path / "lm_eval"
    eval_dir.mkdir()
    (eval_dir / "results.json").write_text(
        json.dumps({"results": {"gsm8k": {"acc,none": 0.7}}}), encoding="utf-8"
    )
    analysis = ba.build_analysis(
        workspace=tmp_path,
        server_log=tmp_path / "server.log",
        success=True,
        run_eval=True,
    )
    assert analysis["throughput"]["steady_state_output_throughput"] == 100.0
    assert "failure_root_cause" not in analysis
    assert analysis["eval_summary"]["accuracy"] == 0.7


def test_build_analysis_failure_attribution(tmp_path):
    (tmp_path / "server.log").write_text("CUDA out of memory\n", encoding="utf-8")
    analysis = ba.build_analysis(
        workspace=tmp_path,
        server_log=tmp_path / "server.log",
        success=False,
        stderr_text="",
        run_eval=False,
    )
    assert analysis["failure_root_cause"] == "oom"
    assert "eval_summary" not in analysis