# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for refusing to deploy a kernel the tuner measured as wrong.

aiter's bf16 tuner records the fraction of output elements its accuracy check
found wrong, and then names the kernel that libtype's winner regardless. On
MI355X every split-K row it selected across four shapes carried a nonzero
figure -- flydsl split_k=7 at 0.0202, asm split_k=7 at 0.0203, asm split_k=4 at
0.0137 -- while every splitK=0 row was 0.0. Re-running those kernels confirms
the recorded number: 1.25-3.98% of elements are wrong, and which ones changes
between identical calls, so the split-K reduction races rather than merely
rounding differently.

The tuned CSV is deployed verbatim (``env_value`` is that file), so without this
filter the fastest wrong answer wins. It also inverts the backend comparison:
flydsl's 37% lead over hipblaslt at M=16 is the time saved by not computing 2%
of the output.
"""

from __future__ import annotations

from pathlib import Path

from kernelforge.gemm_tune.tuners import sglang_dense_bf16 as sd

_HDR = ",".join(
    [
        "gfx",
        "cu_num",
        "M",
        "N",
        "K",
        "bias",
        "dtype",
        "outdtype",
        "scaleAB",
        "bpreshuffle",
        "libtype",
        "solidx",
        "splitK",
        "us",
        "kernelName",
        "err_ratio",
        "tflops",
        "bw",
    ]
)


def _row(m, n, k, libtype, splitk, us, err_ratio):
    return (
        f"gfx950,256,{m},{n},{k},False,torch.bfloat16,torch.bfloat16,False,False,"
        f"{libtype},4492,{splitk},{us},knl,{err_ratio},800.0,3000.0"
    )


def _csv(tmp_path, rows, header=_HDR) -> Path:
    p = tmp_path / "tuned_dense_bf16.csv"
    p.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return p


def _shapes(path: Path) -> set[tuple[str, str, str]]:
    out = set()
    for line in path.read_text(encoding="utf-8").strip().splitlines()[1:]:
        f = line.split(",")
        out.add((f[2], f[3], f[4]))
    return out


class TestDropInaccurateRows:
    def test_drops_the_row_aiter_measured_as_wrong(self, tmp_path):
        # The real pair from the MI355X run: the split-K kernel is faster and
        # wrong, the hipblaslt one is slower and right.
        csv_path = _csv(
            tmp_path,
            [
                _row(16, 1536, 7168, "flydsl", 7, 8.116, 0.0202),
                _row(1024, 1536, 7168, "hipblaslt", 0, 35.739, 0.0),
            ],
        )

        dropped = sd.drop_inaccurate_rows(csv_path)

        assert len(dropped) == 1
        assert dropped[0]["libtype"] == "flydsl"
        assert _shapes(csv_path) == {("1024", "1536", "7168")}

    def test_keeps_everything_when_all_rows_are_accurate(self, tmp_path):
        csv_path = _csv(
            tmp_path,
            [
                _row(16, 1536, 7168, "hipblaslt", 0, 11.126, 0.0),
                _row(16, 4096, 7168, "hipblaslt", 0, 13.709, 0.0),
            ],
        )
        before = csv_path.read_text(encoding="utf-8")

        assert sd.drop_inaccurate_rows(csv_path) == []
        assert csv_path.read_text(encoding="utf-8") == before

    def test_boundary_is_kept(self, tmp_path):
        # Exactly at the limit is not above it; the fp8 split-K cap draws the
        # same line, and disagreeing would make one path deploy what the other
        # rejects.
        csv_path = _csv(tmp_path, [_row(16, 1536, 7168, "asm", 4, 12.0, 0.01)])

        assert sd.drop_inaccurate_rows(csv_path) == []
        assert len(_shapes(csv_path)) == 1

    def test_camel_case_column_is_honoured(self, tmp_path):
        hdr = _HDR.replace("err_ratio", "errRatio")
        csv_path = _csv(tmp_path, [_row(16, 1536, 7168, "flydsl", 7, 8.1, 0.02)], hdr)

        assert len(sd.drop_inaccurate_rows(csv_path)) == 1
        assert _shapes(csv_path) == set()

    def test_missing_accuracy_column_does_not_silently_pass_or_crash(self, tmp_path):
        # No column means no evidence of a problem, so nothing is dropped -- but
        # the operator has to be told the filter did not run, or a schema rename
        # upstream would disable it invisibly.
        hdr = ",".join(c for c in _HDR.split(",") if c != "err_ratio")
        row = ",".join(
            v
            for i, v in enumerate(_row(16, 1536, 7168, "flydsl", 7, 8.1, 0.02).split(","))
            if i != _HDR.split(",").index("err_ratio")
        )
        csv_path = _csv(tmp_path, [row], hdr)

        assert sd.drop_inaccurate_rows(csv_path) == []
        assert len(_shapes(csv_path)) == 1

    def test_unparseable_value_is_treated_as_no_evidence(self, tmp_path):
        csv_path = _csv(tmp_path, [_row(16, 1536, 7168, "flydsl", 7, 8.1, "n/a")])

        assert sd.drop_inaccurate_rows(csv_path) == []
        assert len(_shapes(csv_path)) == 1

    def test_empty_and_missing_files_are_safe(self, tmp_path):
        assert sd.drop_inaccurate_rows(tmp_path / "nope.csv") == []
        empty = tmp_path / "empty.csv"
        empty.write_text(_HDR + "\n", encoding="utf-8")
        assert sd.drop_inaccurate_rows(empty) == []

    def test_a_shape_losing_its_only_row_falls_back_to_aiter_default(self, tmp_path):
        # Dropping the row leaves the shape untuned, which is the intended
        # outcome: at serve time aiter picks its own kernel, and no tuned entry
        # beats a tuned entry that computes the wrong answer.
        csv_path = _csv(tmp_path, [_row(16, 4096, 7168, "flydsl", 4, 13.472, 0.0139)])

        dropped = sd.drop_inaccurate_rows(csv_path)

        assert len(dropped) == 1
        assert _shapes(csv_path) == set()
        assert csv_path.read_text(encoding="utf-8").strip() == _HDR

    def test_dropped_rows_reach_the_result(self, tmp_path, monkeypatch):
        from kernelforge.gemm_tune.tests.test_sglang_dense_bf16 import _prep, _run

        _prefix = "gfx950,256,{m},4096,4096,False,torch.bfloat16,torch.bfloat16,False,False"
        rows = [
            _prefix.format(m=1) + ",flydsl,4492,7,8.116,knl,0.0202,800.0,3000.0",
            _prefix.format(m=512) + ",hipblaslt,438549,0,35.7,knl,0.0,800.0,3000.0",
        ]
        _prep(tmp_path, monkeypatch, tuned_rows=rows)

        result = _run(tmp_path)

        assert len(result.dropped_inaccurate) == 1
        bad = result.dropped_inaccurate[0]
        assert bad["libtype"] == "flydsl" and bad["err_ratio"] == 0.0202
        assert "dropped_inaccurate" in result.to_dict()
        # The surviving shape is still deployable.
        assert result.total_shapes == 1

    def test_a_failed_write_leaves_the_artifact_alone(self, tmp_path, monkeypatch):
        # Truncating the real file first would leave a half-written table that
        # the caller is told nothing was filtered from -- worse than not
        # filtering, because the artifact is then neither original nor clean.
        rows = [
            _row(16, 1536, 7168, "flydsl", 7, 8.116, 0.0202),
            _row(1024, 1536, 7168, "hipblaslt", 0, 35.739, 0.0),
        ]
        csv_path = _csv(tmp_path, rows)
        before = csv_path.read_text(encoding="utf-8")
        monkeypatch.setattr(
            sd.os,
            "replace",
            lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
        )

        assert sd.drop_inaccurate_rows(csv_path) == []
        assert csv_path.read_text(encoding="utf-8") == before
        assert not list(tmp_path.glob("*.tmp"))


class TestReportingFollowsTheArtifact:
    def test_a_shape_whose_row_was_dropped_is_not_reported_as_a_win(self):
        # Reporting "1.24x on M=16" while the artifact holds nothing for M=16
        # is the exact failure this path exists to prevent.
        from kernelforge.gemm_tune.tuners import _aiter_dense_common as ac

        shape_results = [
            {"M": 16, "N": 1536, "K": 7168, "speedup": 1.24, "improved": True},
            {"M": 1024, "N": 1536, "K": 7168, "speedup": 1.05, "improved": True},
        ]
        dropped = [{"M": "16", "N": "1536", "K": "7168", "libtype": "flydsl"}]

        kept = ac._forget_shapes_that_lost_their_row(shape_results, dropped)

        assert [r["M"] for r in kept] == [1024]

    def test_shapes_that_kept_their_row_are_untouched(self):
        from kernelforge.gemm_tune.tuners import _aiter_dense_common as ac

        shape_results = [{"M": 1024, "N": 1536, "K": 7168, "improved": True}]

        assert ac._forget_shapes_that_lost_their_row(shape_results, []) == shape_results
