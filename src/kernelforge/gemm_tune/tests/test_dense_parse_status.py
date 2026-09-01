# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for aiter dense tuner stdout parsing + strict status separation (WP-3).

Format B fixtures are the *real* comparison table observed from the aiter
a8w8_blockscale tuner on a Qwen3-14B FP8 manifest run (the sample that exposed
forge's parser/status gap: rc==0 + 4/4 UPDATE but reported no_improvement).
"""

from __future__ import annotations

from types import SimpleNamespace

from kernelforge.gemm_tune.report import build_report
from kernelforge.gemm_tune.tuners.base import TuneResult
from kernelforge.gemm_tune.tuners._aiter_dense_common import (
    _parse_candidate_csv,
    _parse_tuner_stdout,
    _summarize_shape_results,
)

# Real candidate CSV header (best config per shape; `us` at index 8).
_CANDIDATE_HEADER = "gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio"

# Verbatim from a real run's tune.log (aiter --compare "Would update" block).
_REAL_AITER_TABLE = """
--- Would update (4 shapes) ---
Shape                                    |    Pre(us) |   Post(us) |   Improve |             Action
(8192, 5120, 5120)                       |    1037.74 |     269.71 |    74.01% |             UPDATE
(8192, 5120, 17408)                      |    3502.51 |     746.71 |    78.68% |             UPDATE
(8192, 7168, 5120)                       |    1446.45 |     336.56 |    76.73% |             UPDATE
(8192, 34816, 5120)                      |    6955.79 |    1464.96 |    78.94% |             UPDATE
Re-run with --update_improved to apply.
"""


class TestParser:
    def test_parse_format_b_real_aiter_table(self):
        rows = _parse_tuner_stdout(_REAL_AITER_TABLE, "")
        assert len(rows) == 4
        first = rows[0]
        assert (first["M"], first["N"], first["K"]) == (8192, 5120, 5120)
        assert first["default_us"] == 1037.74
        assert first["tuned_us"] == 269.71
        assert first["speedup"] == round(1037.74 / 269.71, 4)  # ~3.85x
        assert first["improved"] is True
        # gate_up shape parsed too
        assert any((r["M"], r["N"], r["K"]) == (8192, 34816, 5120) for r in rows)
        assert all(r["improved"] for r in rows)

    def test_parse_format_a_legacy(self):
        line = "shape M=1024 N=5120 K=5120 default: 200.0 us tuned: 100.0 us speedup: 2.0x Would update"
        rows = _parse_tuner_stdout(line, "")
        assert len(rows) == 1
        assert rows[0]["speedup"] == 2.0 and rows[0]["improved"] is True

    def test_parse_table_skip_action_not_improved(self):
        rows = _parse_tuner_stdout("(1, 5120, 5120) | 100.0 | 99.0 | 1.0% | SKIP", "")
        assert len(rows) == 1 and rows[0]["improved"] is False

    def test_parse_empty_returns_nothing(self):
        assert _parse_tuner_stdout("no shapes here\njust noise", "") == []


# aiter prints "N/A" for Pre/Improve% and marks the row NEW when a shape had no
# prior tuned entry (nothing to compare the freshly tuned config against).
_NEW_SHAPES_TABLE = """
--- Would update (2 shapes) ---
Shape                                    |    Pre(us) |   Post(us) |   Improve |             Action
(8192, 5120, 5120)                       |        N/A |     269.71 |       N/A |                NEW
(8192, 7168, 5120)                       |        N/A |     336.56 |       N/A |                NEW
Re-run with --update_improved to apply.
"""


class TestParseNewShapes:
    """Regression: an all-new-shape run must parse (not silently vanish and be
    misreported as no_improvement, which skips E2E validation of the new configs)."""

    def test_new_rows_are_parsed_as_is_new(self):
        rows = _parse_tuner_stdout(_NEW_SHAPES_TABLE, "")
        assert len(rows) == 2
        for r in rows:
            assert r["is_new"] is True
            # No baseline -> cannot claim a micro speedup.
            assert r["improved"] is False
            assert r["default_us"] is None
            assert r["speedup"] is None
        assert rows[0]["tuned_us"] == 269.71
        assert (rows[0]["M"], rows[0]["N"], rows[0]["K"]) == (8192, 5120, 5120)

    def test_mixed_new_and_update_rows(self):
        table = (
            "--- Would update (2 shapes) ---\n"
            "(8192, 5120, 5120) | 1037.74 | 269.71 | 74.01% | UPDATE\n"
            "(8192, 7168, 5120) |     N/A | 336.56 |    N/A | NEW\n"
        )
        rows = _parse_tuner_stdout(table, "")
        assert len(rows) == 2
        upd = next(r for r in rows if (r["M"], r["N"], r["K"]) == (8192, 5120, 5120))
        new = next(r for r in rows if (r["M"], r["N"], r["K"]) == (8192, 7168, 5120))
        assert upd["improved"] is True and not upd.get("is_new")
        assert new.get("is_new") is True and new["improved"] is False

    def test_all_new_summary_is_ok_with_unverified_not_improved(self):
        # Align with bf16: shapes without a baseline are unverified, not losers.
        # status=ok + n_improved=0 (do not claim improved).
        s = _summarize_shape_results(_parse_tuner_stdout(_NEW_SHAPES_TABLE, ""))
        assert s["status"] == "ok"
        assert s["total"] == 2 and s["n_improved"] == 0 and s["n_unverified"] == 2
        assert s["best"] == 1.0 and s["avg"] == 1.0  # no fabricated speedup


class TestSummarize:
    def test_empty_is_empty_output(self):
        s = _summarize_shape_results([])
        assert s["status"] == "empty_output" and s["total"] == 0

    def test_improved_is_ok(self):
        s = _summarize_shape_results(_parse_tuner_stdout(_REAL_AITER_TABLE, ""))
        assert s["status"] == "ok" and s["n_improved"] == 4 and s["best"] > 1.0

    def test_no_improvement(self):
        rows = [{"M": 1, "N": 2, "K": 3, "default_us": 10.0, "tuned_us": 10.0, "speedup": 1.0, "improved": False}]
        s = _summarize_shape_results(rows)
        assert s["status"] == "no_improvement" and s["total"] == 1


class TestCandidateCsvFallback:
    """Fallback for the aiter output mode that prints only a
    "Successfully tuned shapes" summary (no per-shape table) but still writes a
    valid candidate CSV. With no untuned baseline the rows are tuned-but-
    unverified: the summary reports ok + n_unverified>0 (bf16-aligned), not
    no_improvement. Promotion happens through candidate=True -- see
    test_force_candidate_wiring.test_candidate_csv_fallback_forces_candidate."""

    def _write_candidate(self, tmp_path):
        p = tmp_path / "candidate_a8w8_blockscale.csv"
        p.write_text(
            _CANDIDATE_HEADER + "\n"
            "gfx942,304,8192,5120,5120,a8w8,123,1,269.71,kern_a,10.0,20.0,0.0\n"
            "gfx942,304,8192,7168,5120,a8w8,456,1,336.56,kern_b,11.0,21.0,0.0\n",
            encoding="utf-8",
        )
        return p

    def test_parse_candidate_csv_real_format(self, tmp_path):
        rows = _parse_candidate_csv(self._write_candidate(tmp_path))
        assert len(rows) == 2
        assert (rows[0]["M"], rows[0]["N"], rows[0]["K"]) == (8192, 5120, 5120)
        assert rows[0]["tuned_us"] == 269.71
        assert rows[0]["default_us"] is None
        assert rows[0]["speedup"] is None
        # No baseline in this mode: rows are tuned-but-unverified, never claimed
        # as improved (the e2e run, not the micro summary, decides KEEP).
        assert not any(r["improved"] for r in rows)
        assert all(r["tuned_unverified"] for r in rows)

    def test_parse_candidate_missing_file_is_empty(self, tmp_path):
        assert _parse_candidate_csv(tmp_path / "does_not_exist.csv") == []
        assert _parse_candidate_csv(None) == []

    def test_parse_candidate_skips_bad_rows(self, tmp_path):
        p = tmp_path / "candidate_bad.csv"
        p.write_text(
            _CANDIDATE_HEADER + "\n"
            "gfx942,304,not_int,5120,5120,a8w8,1,1,10.0,k,1,1,0\n"  # bad M
            "short,row\n"  # too short
            "gfx942,304,64,5120,5120,a8w8,1,1,12.34,k,1,1,0\n",  # valid
            encoding="utf-8",
        )
        rows = _parse_candidate_csv(p)
        assert len(rows) == 1 and rows[0]["M"] == 64 and rows[0]["tuned_us"] == 12.34

    def test_fallback_empty_stdout_with_candidate_is_unverified(self, tmp_path):
        # Mirror run_aiter_dense_tuner's decision: empty stdout parse but a
        # candidate CSV with rows -> shape_results recovered from the candidate.
        # The tuned artifact exists (total>0, so NOT empty_output) but has no
        # measured baseline, so the summary reports ok with unverified_shapes>0.
        stdout_rows = _parse_tuner_stdout("Successfully tuned 2 shapes\n", "")
        assert stdout_rows == []
        shape_results = stdout_rows or _parse_candidate_csv(self._write_candidate(tmp_path))
        s = _summarize_shape_results(shape_results)
        assert s["status"] == "ok" and s["total"] == 2
        assert s["n_improved"] == 0 and s["n_unverified"] == 2
        # speedups unknown in this path -> best/avg stay 1.0 (no fabrication)
        assert s["best"] == 1.0 and s["avg"] == 1.0

    def test_fallback_empty_stdout_no_candidate_is_empty_output(self, tmp_path):
        stdout_rows = _parse_tuner_stdout("Successfully tuned 2 shapes\n", "")
        shape_results = stdout_rows or _parse_candidate_csv(tmp_path / "none.csv")
        s = _summarize_shape_results(shape_results)
        assert s["status"] == "empty_output" and s["total"] == 0


def _report(results):
    return build_report(
        results,
        [],
        profile=SimpleNamespace(model_path="/m/x"),
        framework="vllm-aiter",
        precision="fp8",
        quant_type="blockscale",
        gpu_type="mi355x",
        tp=1,
        conc=64,
        tokens=[1, 8],
        started_at="t",
        total_elapsed_s=1.0,
    )


def _res(status, **kw):
    return TuneResult(tuner_name=kw.pop("name", "a8w8_blockscale"), status=status, **kw)


class TestBuildReportStrictStatus:
    def test_empty_output_not_reported_as_no_improvement(self):
        rep = _report([_res("empty_output", total_shapes=0)])
        assert rep.micro_decision == "empty_output"
        assert rep.requires_e2e_validation is False

    def test_empty_plus_no_improvement_is_no_improvement(self):
        rep = _report([_res("empty_output"), _res("no_improvement", total_shapes=2, name="fmoe_ck")])
        assert rep.micro_decision == "no_improvement"

    def test_empty_plus_candidate_is_candidate(self):
        cand = _res(
            "ok",
            total_shapes=4,
            improved_shapes=4,
            best_micro_speedup=3.85,
            env_var="AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
            env_value="/tmp/c.csv",
            artifact_path="/tmp/c.csv",
        )
        rep = _report([_res("empty_output"), cand])
        assert rep.micro_decision == "candidate"

    def test_empty_plus_failed_is_partial_failure(self):
        # A crash alongside an empty run is reported as partial_failure: the
        # failure outranks the empty result so it cannot disappear behind a
        # sibling tuner's outcome. (It is still never "no_improvement".)
        rep = _report([_res("empty_output"), _res("failed", error="boom", name="fmoe_ck")])
        assert rep.micro_decision == "partial_failure"
        assert [f["tuner"] for f in rep.failed_tuners] == ["fmoe_ck"]

    def test_failed_plus_no_improvement_is_partial_failure(self):
        # The exact shape of the week-long blind spot: one tuner crashes, another
        # legitimately finds nothing, and the batch used to read as "no headroom".
        rep = _report(
            [
                _res("failed", error="boom", error_class="subprocess_error"),
                _res("no_improvement", total_shapes=2, name="fmoe_ck"),
            ]
        )
        assert rep.micro_decision == "partial_failure"
        assert rep.failed_tuners[0]["error_class"] == "subprocess_error"
        assert rep.failed_tuners[0]["error"] == "boom"

    def test_all_failed_stays_failed(self):
        rep = _report([_res("failed", error="a"), _res("failed", error="b", name="fmoe_ck")])
        assert rep.micro_decision == "failed" and rep.status == "failed"
        assert len(rep.failed_tuners) == 2

    def test_candidate_outranks_partial_failure_but_failure_stays_visible(self):
        # A deployable artifact must not be thrown away because a sibling tuner
        # crashed -- but the crash still has to be reported.
        cand = _res(
            "ok",
            total_shapes=4,
            improved_shapes=4,
            best_micro_speedup=3.85,
            env_var="AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
            env_value="/tmp/c.csv",
            artifact_path="/tmp/c.csv",
        )
        rep = _report([cand, _res("failed", error="boom", name="fmoe_ck")])
        assert rep.micro_decision == "candidate"
        assert rep.requires_e2e_validation is True
        assert [f["tuner"] for f in rep.failed_tuners] == ["fmoe_ck"]
        assert rep.to_dict()["failed_tuners"][0]["error"] == "boom"

    def test_no_failure_emits_no_failed_tuners_key(self):
        rep = _report([_res("no_improvement", total_shapes=2)])
        assert rep.failed_tuners == []
        assert "failed_tuners" not in rep.to_dict()

    def test_all_new_shapes_forced_candidate_requires_e2e(self):
        # What run_aiter_dense_tuner emits for an all-new-shape run: micro shows no
        # improvement (best==1.0) but candidate is forced so the freshly tuned
        # configs are validated end-to-end instead of dropped.
        new_shapes = _res(
            "no_improvement",
            total_shapes=2,
            improved_shapes=0,
            best_micro_speedup=1.0,
            candidate=True,
            env_var="AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
            env_value="/tmp/c.csv",
            artifact_path="/tmp/c.csv",
        )
        rep = _report([new_shapes])
        assert rep.micro_decision == "candidate"
        assert rep.requires_e2e_validation is True
        assert rep.artifacts.get("a8w8_blockscale") == "/tmp/c.csv"
