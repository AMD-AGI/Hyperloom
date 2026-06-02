"""Unit tests for the competitor target-gap helpers and tpot derivation.

Covers:

* :func:`research_hints.load_competitor_target` — source filtering + fail-soft.
* :func:`research_hints.gap_analysis` — throughput / tpot / interactivity gaps
  and ``primary_gap`` selection.
* :func:`research_hints.full_gap_summary` — advisory block + latency hint.
* tpot derive fallback in ``benchmark_result._derive_tpot_if_missing``.
"""

from __future__ import annotations

import json
from pathlib import Path

from inference_optimizer.orchestrator import research_hints


def _write(session_dir: Path, payload: dict) -> None:
    from inference_optimizer import session_paths
    path = session_paths.competitor_target_json(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_competitor_target_keeps_only_sourced_rows(tmp_path):
    _write(tmp_path, {
        "gpu": "b300", "model": "m",
        "per_conc": [
            {"conc": 64, "tput_per_gpu": 100.0, "source": "https://pr/1"},
            {"conc": 128, "tput_per_gpu": 200.0},  # sourceless -> dropped
        ],
    })
    target = research_hints.load_competitor_target(tmp_path)
    assert target is not None
    assert len(target["per_conc"]) == 1
    assert target["per_conc"][0]["conc"] == 64


def test_load_competitor_target_missing_returns_none(tmp_path):
    assert research_hints.load_competitor_target(tmp_path) is None


def test_load_competitor_target_all_sourceless_returns_none(tmp_path):
    _write(tmp_path, {
        "gpu": "b300", "model": "m",
        "per_conc": [{"conc": 64, "tput_per_gpu": 100.0}],
    })
    assert research_hints.load_competitor_target(tmp_path) is None


def test_gap_analysis_latency_primary():
    target = {
        "per_conc": [
            {"conc": 64, "tput_per_gpu": 100.0, "tpot_ms": 10.0,
             "source": "s"},
        ],
    }
    gap = research_hints.gap_analysis(
        target, our_tput_per_gpu=95.0, our_tpot_ms=20.0, conc=64,
    )
    assert gap is not None
    # ours 95 vs target 100 -> +5% throughput gap.
    assert round(gap["throughput_gap_pct"], 1) == 5.0
    # ours 20ms vs target 10ms -> 2x ratio.
    assert round(gap["tpot_ratio"], 2) == 2.0
    # latency ratio (100%) far exceeds throughput gap (5%) -> latency.
    assert gap["primary_gap"] == "latency"
    assert gap["source"] == "s"


def test_gap_analysis_throughput_primary():
    target = {
        "per_conc": [
            {"conc": 64, "tput_per_gpu": 200.0, "tpot_ms": 10.0,
             "source": "s"},
        ],
    }
    gap = research_hints.gap_analysis(
        target, our_tput_per_gpu=100.0, our_tpot_ms=10.5, conc=64,
    )
    assert gap is not None
    assert gap["primary_gap"] == "throughput"


def test_gap_analysis_none_when_no_target():
    assert research_hints.gap_analysis(
        None, our_tput_per_gpu=1.0, our_tpot_ms=1.0,
    ) is None


def test_full_gap_summary_emits_latency_hint():
    gap = {
        "throughput_gap_pct": 5.0,
        "tpot_ratio": 2.0,
        "interactivity_gap_pct": None,
        "primary_gap": "latency",
        "source": "https://pr/1",
    }
    text = research_hints.full_gap_summary(gap)
    assert "External target gap" in text
    assert "TPOT ratio" in text
    assert "Priority: TPOT" in text
    assert "https://pr/1" in text


def test_full_gap_summary_no_hint_below_threshold():
    gap = {
        "throughput_gap_pct": 20.0,
        "tpot_ratio": 1.1,
        "interactivity_gap_pct": None,
        "primary_gap": "throughput",
        "source": "s",
    }
    text = research_hints.full_gap_summary(gap)
    assert "Priority: TPOT" not in text


def test_full_gap_summary_empty_on_none():
    assert research_hints.full_gap_summary(None) == ""


def test_derive_tpot_from_e2el_ttft():
    from inference_optimizer.orchestrator.action_executors import benchmark_result
    measurement = {"ttft_mean_ms": 100.0, "e2el_mean_ms": 1090.0}
    report = {"config": {"osl": 100}}
    benchmark_result._derive_tpot_if_missing(measurement, report)
    # (1090 - 100) / (100 - 1) = 10.0
    assert round(measurement["tpot_mean_ms"], 2) == 10.0


def test_derive_tpot_skipped_when_present():
    from inference_optimizer.orchestrator.action_executors import benchmark_result
    measurement = {"ttft_mean_ms": 100.0, "e2el_mean_ms": 1090.0,
                   "tpot_mean_ms": 7.0}
    benchmark_result._derive_tpot_if_missing(measurement, {"osl": 100})
    assert measurement["tpot_mean_ms"] == 7.0


def test_derive_tpot_skipped_without_osl():
    from inference_optimizer.orchestrator.action_executors import benchmark_result
    measurement = {"ttft_mean_ms": 100.0, "e2el_mean_ms": 1090.0}
    benchmark_result._derive_tpot_if_missing(measurement, {})
    assert measurement.get("tpot_mean_ms") is None
