# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A row removed for being wrong is not a shape the run failed to reach.

``_build_result`` compared the surviving row count against ``n_expected``, but
``drop_inaccurate_rows`` had already deleted rows from the artifact by then. A
batch that tuned every shape and then had two of them removed by aiter's own
accuracy check therefore reported ``partial_output`` with a warning blaming the
grouped batch budget -- pointing the reader at ``--timeout`` when the timeout
was never the problem and a backend was computing wrong answers.
"""

from __future__ import annotations

from pathlib import Path

from kernelforge.gemm_tune.model_analyzer import ModelProfile
from kernelforge.gemm_tune.tuners.base import TuneContext
from kernelforge.gemm_tune.tuners.sglang_dense_bf16 import SglangDenseBf16Tuner


def _tuner(tmp_path: Path) -> SglangDenseBf16Tuner:
    return SglangDenseBf16Tuner(
        TuneContext(
            profile=ModelProfile(model_path="/fake", hidden_size=4096, intermediate_size=11008),
            framework="sglang",
            precision="bf16",
            quant_type="none",
            gpu_type="mi355x",
            tp=1,
            conc=8,
            tokens=[8],
            mp=1,
            output_dir=tmp_path,
            iters=20,
            warmup=5,
            min_improvement_pct=1.0,
            timeout_s=3600,
        )
    )


def _rows(n: int, *, improved: bool = True) -> list[dict]:
    return [
        {
            "M": 16 * (i + 1),
            "N": 1536,
            "K": 7168,
            "improved": improved,
            "speedup": 1.2 if improved else 1.0,
            "us": 8.0,
        }
        for i in range(n)
    ]


def _dropped(n: int) -> list[dict]:
    return [
        {
            "M": 4096 + i,
            "N": 1536,
            "K": 7168,
            "libtype": "flydsl",
            "splitK": 7,
            "us": 8.116,
            "err_ratio": 0.0202,
        }
        for i in range(n)
    ]


def _build(tmp_path, *, kept: int, dropped: int, expected: int, improved: bool = True):
    t = _tuner(tmp_path)
    t._dropped_inaccurate = _dropped(dropped)
    csv_path = tmp_path / "tuned_dense_bf16.csv"
    csv_path.write_text("header\n", encoding="utf-8")
    return t._build_result(
        _rows(kept, improved=improved),
        expected,
        csv_path,
        batch_timeout=900,
        rc=0,
    )


class TestAccuracyFilteringIsNotBudgetExhaustion:
    def test_a_complete_batch_with_dropped_rows_is_not_partial(self, tmp_path, caplog):
        # 8 tuned, 2 of them removed as wrong: every expected shape was reached.
        result = _build(tmp_path, kept=6, dropped=2, expected=8)

        assert result.status == "ok"
        # The old text sent the reader to --timeout; the new one names the
        # accuracy check and says outright that the budget was not the cause.
        assert "likely exhausted" not in caplog.text
        assert "accuracy filtering, not budget exhaustion" in caplog.text

    def test_a_complete_batch_with_dropped_rows_and_no_win_is_no_improvement(self, tmp_path):
        result = _build(tmp_path, kept=6, dropped=2, expected=8, improved=False)

        assert result.status == "no_improvement"

    def test_a_genuinely_short_batch_is_still_partial(self, tmp_path, caplog):
        # 6 written + 2 dropped = 8 reached, out of 20 asked for.
        result = _build(tmp_path, kept=6, dropped=2, expected=20)

        assert result.status == "partial_output"
        assert "reached 8 of 20" in caplog.text
        assert "budget" in caplog.text
        # The drops are reported, but not as the cause of the shortfall.
        assert "Separately, 2 of those rows were dropped" in caplog.text

    def test_a_short_batch_with_no_drops_reads_the_same_as_before(self, tmp_path, caplog):
        result = _build(tmp_path, kept=6, dropped=0, expected=20)

        assert result.status == "partial_output"
        assert "Separately" not in caplog.text

    def test_a_clean_complete_batch_is_untouched(self, tmp_path, caplog):
        result = _build(tmp_path, kept=8, dropped=0, expected=8)

        assert result.status == "ok"
        assert "accuracy filtering" not in caplog.text


class TestTheServializedCountsAgree:
    def test_filtered_rows_are_not_counted_as_missing(self, tmp_path):
        d = _build(tmp_path, kept=6, dropped=2, expected=8).to_dict()

        assert d["filtered_shapes"] == 2
        assert d["missing_shapes"] == 0
        assert d["total_shapes"] == 6
        assert d["expected_shapes"] == 8

    def test_a_real_shortfall_still_shows_as_missing(self, tmp_path):
        d = _build(tmp_path, kept=6, dropped=2, expected=20).to_dict()

        assert d["filtered_shapes"] == 2
        assert d["missing_shapes"] == 12


class TestPartialOutputStillReachesE2E:
    """Guard against "fixing" this by refusing to integrate a partial run.

    ``partial_output`` is deliberately on the E2E path (see
    ``phases/kernel.py`` and ``test_partial_output_still_reaches_e2e``): rows
    that were written are real tuning results. This accounting fix must not
    quietly become a new gate.
    """

    def test_partial_output_still_carries_a_deployable_artifact(self, tmp_path):
        result = _build(tmp_path, kept=6, dropped=2, expected=20)

        assert result.status == "partial_output"
        assert result.env_var and result.env_value
        assert result.artifact_path
