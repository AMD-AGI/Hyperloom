# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Serve-safe split-K capping for dense a8w8_blockscale tuning.

The aiter tuner can select a splitK the production dispatch cannot run
("This GEMM is not supported!" -> engine-init crash). ``_cap_splitk_to_serve_safe``
re-selects the fastest serve-safe (splitK <= max) candidate from the profile and
reports whether the deployed CSV still carries any split-K>0 (drives force_candidate).
"""

from __future__ import annotations

import csv
from pathlib import Path

from kernelforge.gemm_tune.tuners._aiter_dense_common import _cap_splitk_to_serve_safe

_HDR = ["gfx", "cu_num", "M", "N", "K", "libtype", "kernelId", "splitK", "us", "kernelName", "tflops", "bw", "errRatio"]


def _row(m, n, k, kid, sk, us, name="knl", er="0.0"):
    return ["gfx950", "256", str(m), str(n), str(k), "ck", str(kid), str(sk), str(us), name, "100", "1000", er]


def _write(path: Path, rows):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_HDR)
        w.writerows(rows)


def _read(path: Path):
    with path.open() as f:
        return list(csv.reader(f))


def test_unsafe_splitk_replaced_by_best_safe_candidate(tmp_path):
    art = tmp_path / "artifact.csv"
    prof = tmp_path / "profile.csv"
    # Winner picked splitK=3 (unsafe). Profile has safe alternatives.
    _write(art, [_row(64, 5120, 17408, 9, 3, 39.0, "sk3")])
    _write(
        prof,
        [
            _row(64, 5120, 17408, 9, 3, 39.0, "sk3"),  # fastest but unsafe
            _row(64, 5120, 17408, 8, 2, 39.6, "sk2"),  # best safe
            _row(64, 5120, 17408, 7, 1, 41.0, "sk1"),
            _row(64, 5120, 17408, 0, 0, 45.0, "sk0"),
        ],
    )
    n, has = _cap_splitk_to_serve_safe(art, prof, max_splitk=2)
    assert n == 1
    assert has is True  # replacement is splitK=2 (>0)
    rows = _read(art)
    assert len(rows) == 2  # header + one row
    assert rows[1][_HDR.index("splitK")] == "2"
    assert rows[1][_HDR.index("kernelName")] == "sk2"


def test_safe_winner_left_unchanged(tmp_path):
    art = tmp_path / "artifact.csv"
    prof = tmp_path / "profile.csv"
    _write(art, [_row(16, 5120, 5120, 8, 2, 16.0, "sk2ok")])
    _write(prof, [_row(16, 5120, 5120, 8, 2, 16.0, "sk2ok")])
    n, has = _cap_splitk_to_serve_safe(art, prof, max_splitk=2)
    assert n == 0
    assert has is True  # kept winner has splitK=2
    assert _read(art)[1][_HDR.index("kernelName")] == "sk2ok"


def test_all_splitk_zero_reports_no_splitk(tmp_path):
    art = tmp_path / "artifact.csv"
    prof = tmp_path / "profile.csv"
    _write(art, [_row(64, 5120, 5120, 8, 0, 16.0), _row(256, 5120, 5120, 0, 0, 30.0)])
    _write(prof, [_row(64, 5120, 5120, 8, 0, 16.0)])
    n, has = _cap_splitk_to_serve_safe(art, prof, max_splitk=2)
    assert n == 0
    assert has is False  # no row carries splitK>0 -> must NOT force_candidate


def test_high_errratio_safe_candidate_rejected(tmp_path):
    art = tmp_path / "artifact.csv"
    prof = tmp_path / "profile.csv"
    _write(art, [_row(32, 7168, 5120, 9, 3, 20.0)])
    # Only safe candidate has a bad errRatio -> must be rejected -> row dropped.
    _write(
        prof,
        [
            _row(32, 7168, 5120, 9, 3, 20.0),
            _row(32, 7168, 5120, 8, 2, 21.0, er="0.5"),
        ],
    )
    n, has = _cap_splitk_to_serve_safe(art, prof, max_splitk=2)
    assert n == 1
    assert has is False
    assert len(_read(art)) == 1  # header only; unsafe row dropped


def test_profile_missing_column_skips_candidate(tmp_path):
    # F2/F4: a profile row lacking a column the deployed CSV has must be skipped
    # (never deploy a malformed row / never silently bypass the errRatio filter).
    art = tmp_path / "artifact.csv"
    prof = tmp_path / "profile.csv"
    _write(art, [_row(64, 5120, 17408, 9, 3, 39.0, "sk3")])  # unsafe
    hdr_no_err = [c for c in _HDR if c != "errRatio"]
    with prof.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr_no_err)
        # a splitK=2 candidate that WOULD be selected, but its row lacks errRatio
        w.writerow(["gfx950", "256", "64", "5120", "17408", "ck", "8", "2", "39.6", "sk2", "100", "1000"])
    n, has = _cap_splitk_to_serve_safe(art, prof, max_splitk=2)
    assert n == 1  # candidate skipped -> unsafe row dropped
    assert has is False
    assert len(_read(art)) == 1


def test_missing_profile_drops_unsafe_row(tmp_path):
    art = tmp_path / "artifact.csv"
    _write(art, [_row(16, 5120, 5120, 9, 3, 15.0), _row(256, 5120, 5120, 0, 0, 30.0)])
    n, has = _cap_splitk_to_serve_safe(art, tmp_path / "nope.csv", max_splitk=2)
    assert n == 1
    assert has is False  # surviving row is splitK=0
    rows = _read(art)
    assert len(rows) == 2  # header + the safe splitK=0 row survives
    assert rows[1][_HDR.index("M")] == "256"


def test_missing_artifact_is_noop(tmp_path):
    assert _cap_splitk_to_serve_safe(tmp_path / "nope.csv", tmp_path / "p.csv", 2) == (0, False)


def test_support_fn_keeps_splitk_within_per_shape_max(tmp_path):
    # Per-shape production limit: shape A supports splitK=3 (keep it), shape B only
    # supports 2 (downgrade its splitK=3 pick). Captures gain cap=2 would drop.
    art = tmp_path / "a.csv"
    prof = tmp_path / "p.csv"
    _write(art, [_row(16, 5120, 5120, 9, 3, 10.0, "A3"), _row(64, 5120, 5120, 9, 3, 20.0, "B3")])
    _write(
        prof,
        [
            _row(16, 5120, 5120, 9, 3, 10.0, "A3"),
            _row(16, 5120, 5120, 8, 2, 10.5, "A2"),
            _row(64, 5120, 5120, 9, 3, 20.0, "B3"),
            _row(64, 5120, 5120, 8, 2, 20.6, "B2"),
        ],
    )
    support = lambda m, n, k: 3 if (m, n, k) == (16, 5120, 5120) else 2  # noqa: E731
    n, has = _cap_splitk_to_serve_safe(art, prof, 2, support_fn=support)
    assert n == 1 and has is True  # only shape B changed
    rows = {r[_HDR.index("M")]: r for r in _read(art)[1:]}
    assert rows["16"][_HDR.index("splitK")] == "3"  # kept (per-shape max=3)
    assert rows["16"][_HDR.index("kernelName")] == "A3"
    assert rows["64"][_HDR.index("splitK")] == "2"  # downgraded (per-shape max=2)
    assert rows["64"][_HDR.index("kernelName")] == "B2"


def test_support_fn_tightens_below_static_cap(tmp_path):
    # Per-shape max BELOW the static cap: a shape whose production kernel only
    # supports splitK<=1 must DOWNGRADE a splitK=2 pick the static cap=2 would
    # otherwise keep -- keeping it would crash serve ("not supported").
    art = tmp_path / "a.csv"
    prof = tmp_path / "p.csv"
    _write(art, [_row(64, 5120, 5120, 8, 2, 20.0, "B2")])
    _write(
        prof,
        [
            _row(64, 5120, 5120, 8, 2, 20.0, "B2"),  # static-cap-safe but per-shape UNSAFE
            _row(64, 5120, 5120, 7, 1, 20.4, "B1"),  # best within per-shape max=1
            _row(64, 5120, 5120, 0, 0, 22.0, "B0"),
        ],
    )
    n, has = _cap_splitk_to_serve_safe(art, prof, 2, support_fn=lambda m, n, k: 1)
    assert n == 1  # the splitK=2 row was rewritten despite sk <= static cap
    row = _read(art)[1]
    assert row[_HDR.index("splitK")] == "1"  # tightened to per-shape max
    assert row[_HDR.index("kernelName")] == "B1"
    assert has is True


def test_support_fn_none_falls_back_to_static_cap(tmp_path):
    # support_fn returning None (trial unavailable) -> static max_splitk per shape.
    art = tmp_path / "a.csv"
    prof = tmp_path / "p.csv"
    _write(art, [_row(64, 5120, 5120, 9, 3, 20.0, "B3")])
    _write(prof, [_row(64, 5120, 5120, 9, 3, 20.0, "B3"), _row(64, 5120, 5120, 8, 2, 20.6, "B2")])
    n, has = _cap_splitk_to_serve_safe(art, prof, 2, support_fn=lambda m, n, k: None)
    assert n == 1
    assert _read(art)[1][_HDR.index("splitK")] == "2"  # fell back to static cap=2


def test_schema_without_errratio_does_not_crash(tmp_path):
    # A CSV schema lacking the errRatio column must not KeyError-crash the tuner
    # (relevant when --splitK is extended to other dense tuners); absent -> 0.
    hdr = [
        "gfx",
        "cu_num",
        "M",
        "N",
        "K",
        "libtype",
        "kernelId",
        "splitK",
        "us",
        "kernelName",
        "tflops",
        "bw",
    ]  # no errRatio

    def _r(m, n, k, kid, sk, us, name):
        return ["gfx950", "256", str(m), str(n), str(k), "ck", str(kid), str(sk), str(us), name, "100", "1000"]

    def _w(p, rows):
        with p.open("w", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(hdr)
            cw.writerows(rows)

    art = tmp_path / "a.csv"
    prof = tmp_path / "p.csv"
    _w(art, [_r(64, 5120, 17408, 9, 3, 39.0, "sk3")])  # unsafe splitK=3
    _w(prof, [_r(64, 5120, 17408, 9, 3, 39.0, "sk3"), _r(64, 5120, 17408, 8, 2, 39.6, "sk2")])
    n, has = _cap_splitk_to_serve_safe(art, prof, max_splitk=2)  # must not raise
    assert n == 1
    assert has is True
    assert _read(art)[1][hdr.index("splitK")] == "2"


def test_cap_header_case_insensitive_no_unsafe_passthrough(tmp_path):
    # A differently-cased deployed header must NOT make the cap bail early and
    # pass an unsafe splitK row through unchanged (-> serve crash). With the
    # case-insensitive column lookup the cap still engages: the splitK=3 row is
    # capped/dropped so no splitK>max survives.
    art = tmp_path / "a.csv"
    prof = tmp_path / "p.csv"
    hdr = [
        "gfx",
        "cu_num",
        "m",
        "n",
        "k",
        "libtype",
        "kernelId",
        "SplitK",
        "us",
        "kernelName",
        "tflops",
        "bw",
        "errRatio",
    ]

    def _r(sk, name):
        return ["gfx950", "256", "64", "5120", "17408", "ck", "9", str(sk), "39.0", name, "100", "1000", "0.0"]

    with art.open("w", newline="") as f:
        csv.writer(f).writerows([hdr, _r(3, "sk3")])
    with prof.open("w", newline="") as f:
        csv.writer(f).writerows([hdr, _r(3, "sk3"), _r(2, "sk2")])

    n, has = _cap_splitk_to_serve_safe(art, prof, 2)
    assert n == 1  # cap engaged (would be 0 = no-op if the case mismatch bailed)
    out = _read(art)
    si = [h.lower() for h in out[0]].index("splitk")
    assert all(int(r[si]) <= 2 for r in out[1:])  # no unsafe splitK>2 remains
