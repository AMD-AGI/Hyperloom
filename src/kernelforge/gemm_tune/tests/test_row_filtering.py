# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for dropping lost-comparison rows from the deployed artifact.

A tuned row that measured *slower* than stock is actively harmful once merged:
it overrides a better stock choice. But the filter must distinguish "measured
to be not better" from "never had anything to measure against" -- the second
covers newly-tuned shapes, the candidate-CSV fallback and hipblaslt-only runs,
which are exactly the configs the forced-e2e path exists to protect.
"""

from __future__ import annotations

from kernelforge.gemm_tune.tuners._aiter_dense_common import _filter_unimproved_rows

_HDR = "gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio"


def _row(m, n, k):
    return f"gfx950,256,{m},{n},{k},ck,1,0,16.0,knl,100,1000,0.0"


def _csv(tmp_path, rows):
    p = tmp_path / "candidate.csv"
    p.write_text("\n".join([_HDR, *rows]) + "\n", encoding="utf-8")
    return p


def _lost(m, n, k):
    return {"M": m, "N": n, "K": k, "default_us": 10.0, "tuned_us": 12.0, "speedup": 0.83, "improved": False}


def _won(m, n, k):
    return {"M": m, "N": n, "K": k, "default_us": 12.0, "tuned_us": 10.0, "speedup": 1.2, "improved": True}


def _new(m, n, k):
    return {
        "M": m,
        "N": n,
        "K": k,
        "default_us": None,
        "tuned_us": 10.0,
        "speedup": None,
        "improved": False,
        "is_new": True,
    }


def _unverified(m, n, k):
    return {
        "M": m,
        "N": n,
        "K": k,
        "default_us": None,
        "tuned_us": 10.0,
        "speedup": None,
        "improved": False,
        "tuned_unverified": True,
    }


def _rows_of(path):
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [tuple(line.split(",")[2:5]) for line in lines[1:]]


class TestDropsLosers:
    def test_row_that_lost_is_removed(self, tmp_path):
        csv = _csv(tmp_path, [_row(64, 5120, 5120), _row(128, 5120, 5120)])
        dropped, kept = _filter_unimproved_rows(csv, [_lost(64, 5120, 5120), _won(128, 5120, 5120)])
        assert (dropped, kept) == (1, 1)
        assert _rows_of(csv) == [("128", "5120", "5120")]

    def test_winner_is_untouched(self, tmp_path):
        csv = _csv(tmp_path, [_row(64, 5120, 5120)])
        dropped, _ = _filter_unimproved_rows(csv, [_won(64, 5120, 5120)])
        assert dropped == 0
        assert _rows_of(csv) == [("64", "5120", "5120")]


class TestKeepsUnmeasured:
    def test_new_shape_survives(self, tmp_path):
        # improved=False here means "no prior baseline", not "lost".
        csv = _csv(tmp_path, [_row(64, 5120, 5120)])
        dropped, _ = _filter_unimproved_rows(csv, [_new(64, 5120, 5120)])
        assert dropped == 0
        assert _rows_of(csv) == [("64", "5120", "5120")]

    def test_tuned_unverified_survives(self, tmp_path):
        csv = _csv(tmp_path, [_row(64, 5120, 5120)])
        dropped, _ = _filter_unimproved_rows(csv, [_unverified(64, 5120, 5120)])
        assert dropped == 0

    def test_mixed_batch_keeps_unmeasured_drops_losers(self, tmp_path):
        csv = _csv(tmp_path, [_row(1, 2, 3), _row(4, 5, 6), _row(7, 8, 9)])
        dropped, kept = _filter_unimproved_rows(csv, [_lost(1, 2, 3), _new(4, 5, 6), _won(7, 8, 9)])
        assert (dropped, kept) == (1, 2)
        assert _rows_of(csv) == [("4", "5", "6"), ("7", "8", "9")]


class TestRobustness:
    def test_no_losers_leaves_file_byte_identical(self, tmp_path):
        csv = _csv(tmp_path, [_row(64, 5120, 5120)])
        before = csv.read_bytes()
        assert _filter_unimproved_rows(csv, [_won(64, 5120, 5120)]) == (0, 0)
        assert csv.read_bytes() == before

    def test_missing_file_is_a_noop(self, tmp_path):
        assert _filter_unimproved_rows(tmp_path / "gone.csv", [_lost(1, 2, 3)]) == (0, 0)

    def test_header_without_mnk_is_left_alone(self, tmp_path):
        p = tmp_path / "weird.csv"
        p.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
        assert _filter_unimproved_rows(p, [_lost(1, 2, 3)]) == (0, 0)
        assert p.read_text(encoding="utf-8") == "a,b,c\n1,2,3\n"

    def test_unparseable_row_is_kept_not_guessed(self, tmp_path):
        csv = _csv(tmp_path, ["garbage,row", _row(1, 2, 3)])
        dropped, _ = _filter_unimproved_rows(csv, [_lost(1, 2, 3)])
        assert dropped == 1
        assert "garbage,row" in csv.read_text(encoding="utf-8")

    def test_empty_shape_results_is_a_noop(self, tmp_path):
        csv = _csv(tmp_path, [_row(1, 2, 3)])
        assert _filter_unimproved_rows(csv, []) == (0, 0)
