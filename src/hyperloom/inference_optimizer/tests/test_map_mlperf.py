# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""MLPerf harness -> InferenceX schema, read against upstream's real output.

The fixtures are one smoke run's ``result_summary.json`` and ``scores.json``
from ``inference-endpoint``, with only the per-turn lists trimmed.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

from hyperloom.common.perf_metric import (
    GRADED_INTVTY,
    GRADED_INTVTY_P50,
    GRADED_TOTAL,
    perf_snapshot_from_mapping,
)
from hyperloom.inference_optimizer.agentx.deploy import deploy_agentx_assets
from hyperloom.inference_optimizer.agentx.mapping import MlperfReportError, map_mlperf

_FIXTURES = Path(__file__).parent / "fixtures" / "mlperf_agentic_smoke"


def _report() -> dict:
    return json.loads((_FIXTURES / "result_summary.json").read_text(encoding="utf-8"))


def _scores() -> dict:
    return json.loads((_FIXTURES / "scores.json").read_text(encoding="utf-8"))


def _map(report=None, scores=None, **kw):
    return map_mlperf(
        _report() if report is None else report,
        corpus=kw.pop("corpus", "agentic_combined_v6"),
        scores=scores,
    )


def test_upstream_percentile_keys_carry_real_latencies():
    """Upstream keys percentiles as ``"50.0"``; a reader looking up ``"50"`` wrote 0.0 for all of them."""
    mapped = _map()
    assert mapped["median_ttft_ms"] == pytest.approx(3325.72024)
    assert mapped["p90_ttft_ms"] == pytest.approx(8220.389053)
    assert mapped["p99_ttft_ms"] == pytest.approx(42970.906161)
    assert mapped["median_tpot_ms"] == pytest.approx(76.69808934675615)
    assert mapped["p90_tpot_ms"] == pytest.approx(1012.8065783333334)
    assert mapped["median_e2el_ms"] == pytest.approx(22622.874268)
    assert mapped["p99_e2el_ms"] == pytest.approx(277600.946081)
    assert mapped["std_ttft_ms"] == pytest.approx(9251.136500507198)
    assert mapped["mean_e2el_ms"] == pytest.approx(43923.38362456977)


def test_throughput_and_counts_come_from_the_report():
    mapped = _map()
    assert mapped["output_throughput"] == pytest.approx(78.98510974994119)
    assert mapped["request_throughput"] == pytest.approx(0.25808204553552216)
    assert mapped["duration"] == pytest.approx(666.454730096)
    assert mapped["completed"] == 172
    assert mapped["total_output_tokens"] == 52640
    assert mapped["request_error_rate"] == 0.0
    assert mapped["submission_valid"] is True
    assert mapped["osl_distribution"]["p50"] == 30


def test_the_corpus_is_recorded():
    assert _map(corpus="agentic_combined_v6")["corpus_loader"] == "agentic_combined_v6"


def test_interactivity_axes_invert_the_tpot_percentiles():
    """Not aiperf's OSL/E2EL axis, but the only per-user rate this harness supports grading on.

    ``events.jsonl`` carries reply text rather than per-token deltas, and ``output_sequence_lengths`` has a zero
    median because a tool-call-only turn emits no text, so a P10 of OSL/E2EL is 0.0 and would be read as unmeasured.
    """
    mapped = _map()
    assert mapped["e2e_norm_intvty_p50"] == pytest.approx(1000.0 / 76.69808934675615)
    assert mapped["e2e_norm_intvty_p90"] == pytest.approx(1000.0 / 1012.8065783333334)
    # The slow tail must be the lower rate, the orientation the aiperf P10 carries.
    assert mapped["e2e_norm_intvty_p90"] < mapped["e2e_norm_intvty_p50"]


def test_the_mapped_result_resolves_as_a_grading_anchor():
    """The axes grading needs, together: a missing one discards the whole anchor and degrades the session."""
    snapshot = perf_snapshot_from_mapping(_map())
    assert snapshot is not None
    assert snapshot[GRADED_INTVTY] > 0
    assert snapshot[GRADED_INTVTY_P50] > 0
    assert snapshot[GRADED_TOTAL] > 0


def test_a_tpot_series_with_no_samples_writes_no_interactivity_axis():
    """Absent beats a measured zero: ``perf_snapshot_from_mapping`` reads 0.0 as unmeasured."""
    report = _report()
    report["tpot"] = {}
    mapped = _map(report)
    assert not [key for key in mapped if key.startswith("e2e_norm_intvty")]
    assert perf_snapshot_from_mapping(mapped) is None


def test_the_total_guard_carries_the_output_rate():
    """The harness measures no input-token series, so grading's total guard is the output rate."""
    mapped = _map()
    assert mapped["total_token_throughput"] == mapped["output_throughput"]


def test_inline_accuracy_and_unscored_turns():
    mapped = _map(scores=_scores())
    assert mapped["accuracy_score"] == pytest.approx(0.7178)
    assert mapped["accuracy_missing_turns"] == 0


def test_no_scores_file_leaves_accuracy_absent():
    assert "accuracy_score" not in _map()


def test_an_incomplete_run_is_not_a_valid_measurement():
    report = _report()
    report["complete"] = False
    report["state"] = "interrupted"
    mapped = _map(report)
    assert mapped["submission_valid"] is False
    assert mapped["submission_invalid_reasons"] == ["interrupted", "incomplete_run"]


def test_failed_samples_are_an_error_rate_over_issued():
    report = _report()
    report["n_samples_failed"] = 43
    assert _map(report)["request_error_rate"] == pytest.approx(25.0)


def test_a_series_with_no_samples_is_absent_not_zero():
    report = _report()
    report["ttft"] = {}
    mapped = _map(report)
    assert "median_ttft_ms" not in mapped
    assert mapped["median_tpot_ms"] > 0


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda r: r.pop("n_samples_completed"), id="missing-counter"),
        pytest.param(lambda r: r["ttft"]["percentiles"].pop("50.0"), id="missing-p50"),
        pytest.param(lambda r: r["ttft"].pop("std_dev"), id="std-renamed"),
        pytest.param(lambda r: r.__setitem__("tps", "78.9"), id="non-numeric-tps"),
    ],
)
def test_a_schema_mismatch_raises_instead_of_mapping_zeros(mutate):
    report = copy.deepcopy(_report())
    mutate(report)
    with pytest.raises(MlperfReportError):
        _map(report)


def test_a_scores_file_without_turns_raises():
    with pytest.raises(MlperfReportError):
        _map(scores={"score": 0.7})


def test_deployed_map_mlperf_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "agentx_mapping", raising=False)
    monkeypatch.setenv("AGENTIC_DATASET_PATH", "/data/agentic_combined_v6.jsonl")
    deployed = deploy_agentx_assets(tmp_path / "benchmarks")
    mapper = next(path for path in deployed if path.name == "map_mlperf.py")
    dst = tmp_path / "inferencex_result.json"
    from importlib.machinery import SourceFileLoader

    mod = SourceFileLoader("map_mlperf_cli", str(mapper)).load_module()
    mod.main(str(_FIXTURES / "result_summary.json"), str(dst), str(_FIXTURES / "scores.json"))
    out = json.loads(dst.read_text(encoding="utf-8"))
    assert out["corpus_loader"] == "agentic_combined_v6"
    assert out["accuracy_score"] == pytest.approx(0.7178)
    assert out["median_ttft_ms"] == pytest.approx(3325.72024)
