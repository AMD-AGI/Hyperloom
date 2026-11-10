# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stock anchor selection across the warmup and measured baseline rounds."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.actions.executors.benchmark_run.rounds import _BenchmarkRoundsMixin


class _RoundSequence(_BenchmarkRoundsMixin):
    def __init__(self, receipts: list[dict]):
        self.receipts = receipts
        self.calls = 0

    async def _run_guarded_round(self, *_args, **_kwargs):
        receipt = self.receipts[self.calls]
        self.calls += 1
        return receipt

    def _cold_anchor_when_hot_pass_unaffordable(self, *_args, **_kwargs):
        return None


def _run_rounds(tmp_path: Path, *, warmup: float, measured: float, kind="baseline", exempt=False):
    receipts = []
    for slot, throughput in (("warmup_round", warmup), ("measure_round", measured)):
        directory = tmp_path / slot
        directory.mkdir()
        (directory / "receipt.txt").write_text(str(throughput), encoding="utf-8")
        receipts.append(
            {
                "status": "succeeded",
                "valid_measurement": True,
                "output_throughput": throughput,
                "output_dir": str(directory),
                "workspace": str(directory),
            }
        )
        if slot == "warmup_round":
            receipts[-1]["accuracy"] = 0.96
    rounds = _RoundSequence(receipts)
    attempt = SimpleNamespace(
        ctx=SimpleNamespace(task=SimpleNamespace(kind=kind)),
        params={"quality_ref_exempt": exempt},
    )
    result = asyncio.run(
        rounds._run_cold_start_guarded_rounds(attempt, tmp_path, {"port": 8888}, [], defer_accuracy=False)
    )
    assert rounds.calls == 2
    return result, receipts


def test_stock_baseline_retains_the_higher_anchor_without_failing_or_dropping_a_round(tmp_path):
    result, receipts = _run_rounds(tmp_path, warmup=544.38, measured=333.71)

    assert result["status"] == "succeeded"
    assert result["valid_measurement"] is True
    assert result["output_throughput"] == pytest.approx(544.38)
    assert result["warmup_round_tput"] == pytest.approx(544.38)
    assert result["accuracy"] == pytest.approx(0.96)
    assert result["workspace"] == receipts[0]["workspace"]
    rejected = result["measure_round_rejected"]
    assert rejected["reason"] == "downward_drift"
    assert rejected["output_throughput"] == pytest.approx(333.71)
    assert rejected["delta_pct"] == pytest.approx(-38.7, abs=0.01)
    assert rejected["output_dir"] == receipts[1]["output_dir"]
    assert rejected["workspace"] == receipts[1]["workspace"]
    assert "baseline_downward_drift_anchor" in result["nonfatal_warnings"]
    assert not result.get("measure_round_dropped")
    assert "baseline_downward_drift_anchor" not in receipts[0].get("nonfatal_warnings", [])
    assert all((Path(receipt["output_dir"]) / "receipt.txt").is_file() for receipt in receipts)


def test_quality_ref_exempt_candidate_retains_its_actual_measured_result(tmp_path):
    result, receipts = _run_rounds(tmp_path, warmup=544.38, measured=333.71, exempt=True)

    assert result["status"] == "succeeded"
    assert result["output_throughput"] == pytest.approx(333.71)
    assert result["workspace"] == receipts[1]["workspace"]
    assert "measure_round_rejected" not in result


@pytest.mark.parametrize(
    ("warmup", "measured", "expected"),
    [(400.0, 300.0, 300.0), (400.0, 299.0, 400.0), (400.0, 390.0, 390.0), (334.0, 333.71, 333.71)],
)
def test_only_a_drop_exceeding_twenty_five_percent_changes_the_stock_anchor(tmp_path, warmup, measured, expected):
    result, _receipts = _run_rounds(tmp_path, warmup=warmup, measured=measured)

    assert result["output_throughput"] == pytest.approx(expected)
    assert ("measure_round_rejected" in result) is (expected != measured)


@pytest.mark.parametrize("kind", ["replay_warm_recipe", "profile"])
def test_non_stock_rounds_keep_their_measured_result(tmp_path, kind):
    result, _receipts = _run_rounds(tmp_path, warmup=544.38, measured=333.71, kind=kind)

    assert result["output_throughput"] == pytest.approx(333.71)
    assert "measure_round_rejected" not in result
