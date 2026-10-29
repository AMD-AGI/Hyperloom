# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""asm split-K rows are judged on their relative-norm error, every other row on aiter's errRatio."""

from __future__ import annotations

import csv
import types
from pathlib import Path

import pytest

import kernelforge.workflows.gemm_tune.aiter_graph_retime as retime
import kernelforge.workflows.gemm_tune.aiter_splitk_validate as validate
import kernelforge.workflows.gemm_tune.tuners._aiter_dense_common as ac
import kernelforge.workflows.gemm_tune.tuners._row_accuracy as accuracy
from kernelforge.workflows.gemm_tune.tuners.base import TuneContext

_HDR = ["gfx", "cu_num", "M", "N", "K", "libtype", "kernelId", "splitK", "us", "kernelName", "tflops", "bw", "errRatio"]
_OP = "a8w8_blockscale_bpreshuffle"


def _row(libtype: str, splitk: int, err_ratio: str, *, m: int = 64, us: float = 24.4, name: str = "knl") -> list[str]:
    return [
        "gfx950",
        "256",
        str(m),
        "5120",
        "17408",
        libtype,
        "3",
        str(splitk),
        str(us),
        name,
        "100",
        "1000",
        err_ratio,
    ]


def _write(path: Path, rows: list[list[str]]) -> Path:
    with path.open("w", newline="") as fh:
        csv.writer(fh).writerows([_HDR, *rows])
    return path


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


class _Measured:
    """A relative-norm check that answers ``value`` and records the rows it was asked about."""

    def __init__(self, value: float | None):
        self.value = value
        self.asked: list[tuple] = []

    def __call__(self, m, n, k, split_k, kernel_name):
        self.asked.append((m, n, k, split_k, kernel_name))
        return self.value


class TestDropInaccurateRows:
    def test_an_asm_splitk_row_within_the_norm_bound_ships_with_its_figure(self, tmp_path):
        # A hand-tuned down_proj row: errRatio 0.0125 from bf16 atomic accumulation, relative-norm error 0.0047.
        art = _write(tmp_path / "a.csv", [_row("asm", 6, "0.0125", name="asm_sk6")])
        measured = _Measured(0.0047)

        applied = accuracy.drop_inaccurate_rows(art, measured)

        assert applied.completed and applied.dropped == []
        assert measured.asked == [(64, 5120, 17408, 6, "asm_sk6")]
        (row,) = _read(art)
        assert row["kernelName"] == "asm_sk6" and float(row["relNormErr"]) == 0.0047

    def test_an_asm_splitk_row_beyond_the_norm_bound_is_dropped_with_its_figure(self, tmp_path):
        art = _write(tmp_path / "a.csv", [_row("asm", 6, "0.0125")])

        applied = accuracy.drop_inaccurate_rows(art, _Measured(0.012))

        assert applied.completed
        assert [float(r["relNormErr"]) for r in applied.dropped] == [0.012]
        assert _read(art) == []

    def test_an_unmeasurable_row_keeps_the_err_ratio_verdict(self, tmp_path):
        art = _write(tmp_path / "a.csv", [_row("asm", 6, "0.0125")])

        applied = accuracy.drop_inaccurate_rows(art, _Measured(None))

        assert applied.completed and len(applied.dropped) == 1
        assert _read(art) == []

    @pytest.mark.parametrize(("libtype", "splitk"), [("asm", 1), ("asm", 0), ("ck", 6), ("cktile", 2)])
    def test_other_rows_keep_the_err_ratio_gate(self, tmp_path, libtype, splitk):
        art = _write(tmp_path / "a.csv", [_row(libtype, splitk, "0.0125")])
        measured = _Measured(0.0)

        applied = accuracy.drop_inaccurate_rows(art, measured)

        assert len(applied.dropped) == 1 and measured.asked == []

    def test_an_accurate_row_is_not_measured_and_the_table_is_untouched(self, tmp_path):
        art = _write(tmp_path / "a.csv", [_row("asm", 6, "0.0049")])
        before = art.read_text()
        measured = _Measured(0.0)

        applied = accuracy.drop_inaccurate_rows(art, measured)

        assert applied.completed and applied.dropped == [] and measured.asked == []
        assert art.read_text() == before

    def test_without_a_norm_check_errratio_decides(self, tmp_path):
        art = _write(tmp_path / "a.csv", [_row("asm", 6, "0.0125")])

        applied = accuracy.drop_inaccurate_rows(art)

        assert len(applied.dropped) == 1 and _read(art) == []


class TestServeSafeReplacement:
    def test_an_asm_splitk_candidate_above_err_ratio_can_replace_an_unservable_row(self, tmp_path):
        art = _write(tmp_path / "a.csv", [_row("asm", 8, "0.0", us=19.0, name="asm_sk8")])
        prof = _write(
            tmp_path / "p.csv",
            [
                _row("asm", 8, "0.0", us=19.0, name="asm_sk8"),
                _row("asm", 6, "0.0125", us=19.9, name="asm_sk6"),
                _row("asm", 2, "0.0", us=33.0, name="asm_sk2"),
            ],
        )

        limit = ac._SplitKLimit(2, lambda m, n, k: 6, _Measured(0.0047))

        ac._cap_splitk_to_serve_safe(art, prof, limit, forwarding_libtypes=frozenset({"asm"}))

        assert [r["kernelName"] for r in _read(art)] == ["asm_sk6"]

    @pytest.mark.parametrize("measured", [None, _Measured(0.3), _Measured(None)])
    def test_a_replacement_failing_both_gates_yields_to_the_next_accurate_candidate(self, tmp_path, measured):
        art = _write(tmp_path / "a.csv", [_row("asm", 8, "0.0", us=10.0, name="asm_sk8")])
        prof = _write(
            tmp_path / "p.csv",
            [
                _row("asm", 8, "0.0", us=10.0, name="asm_sk8"),
                _row("asm", 6, "0.5", us=11.0, name="asm_sk6"),
                _row("asm", 4, "0.0", us=12.0, name="asm_sk4"),
            ],
        )
        limit = ac._SplitKLimit(2, lambda m, n, k: 6, measured)

        ac._cap_splitk_to_serve_safe(art, prof, limit, forwarding_libtypes=frozenset({"asm"}))

        assert [r["kernelName"] for r in _read(art)] == ["asm_sk4"]


def _ctx(tmp_path: Path) -> TuneContext:
    return TuneContext(
        profile=types.SimpleNamespace(),
        framework="sglang",
        precision="fp8",
        quant_type="blockscale",
        gpu_type="mi355x",
        tp=1,
        conc=64,
        tokens=[64],
        mp=1,
        output_dir=tmp_path,
        iters=1,
        warmup=0,
        min_improvement_pct=3.0,
        timeout_s=60,
        untuned_csv=tmp_path / "in.csv",
    )


def _tune(tmp_path: Path, monkeypatch, rel_norm_err: float, *, trial_max: int = 8, profile=None):
    """Run the bpreshuffle driver on a tuner run whose only pick is a hand-tuned asm splitK=6 down_proj row by default."""
    monkeypatch.delenv("FORGE_SPLITK_TRIAL", raising=False)
    monkeypatch.setitem(validate._TRIAL_OPS, _OP, lambda m, n, k, sk, device="cuda": sk <= trial_max)
    monkeypatch.setitem(validate._REL_NORM_OPS, _OP, lambda mnk, sk, name, device="cuda": rel_norm_err)
    monkeypatch.delitem(retime._RETIME_OPS, _OP)
    (tmp_path / "in.csv").write_text("M,N,K\n64,5120,17408\n")
    profile = profile or [_row("asm", 6, "0.0125")]
    _write(tmp_path / "tuned_bpre.csv", [min(profile, key=lambda r: float(r[_HDR.index("us")]))])
    _write(tmp_path / "profile_bpre.csv", profile)
    monkeypatch.setattr(ac, "find_tuner_script", lambda key: tmp_path / "script.py")
    monkeypatch.setattr(ac, "_resolve_input_csv", lambda ctx, wd, needs_q_dtype_w=False: tmp_path / "in.csv")
    monkeypatch.setattr(ac, "resolve_aiter_root", lambda: str(tmp_path))
    monkeypatch.setattr(ac._tr, "is_isolation_enabled", lambda: False)
    monkeypatch.setattr(ac, "run_subprocess", lambda cmd, **kw: (0, "", ""))
    monkeypatch.setattr(ac, "_find_latest_candidate", lambda name, t: None)
    spec = ac.DenseTunerSpec("bpre", _OP, "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE", ("--splitK",))
    return ac.run_aiter_dense_tuner(spec, _ctx(tmp_path), tmp_path)


def test_the_bpreshuffle_tuner_ships_an_accurate_asm_splitk_row(tmp_path, monkeypatch):
    result = _tune(tmp_path, monkeypatch, 0.0047)

    assert result.dropped_inaccurate == []
    (row,) = _read(Path(result.env_value))
    assert (row["splitK"], float(row["relNormErr"])) == ("6", 0.0047)


def test_the_bpreshuffle_tuner_reports_the_norm_of_a_dropped_row(tmp_path, monkeypatch):
    result = _tune(tmp_path, monkeypatch, 0.02)

    (dropped,) = result.dropped_inaccurate
    assert (dropped["err_ratio"], dropped["rel_norm_err"]) == (0.0125, 0.02)
    assert _read(Path(result.env_value)) == []


def test_an_inaccurate_splitk_replacement_yields_to_the_next_accurate_candidate(tmp_path, monkeypatch):
    # The deployed asm splitK=8 exceeds the trialled limit of 6; the faster sk6 fails both gates, so sk4 must ship.
    profile = [
        _row("asm", 8, "0.0", us=10.0, name="asm_sk8"),
        _row("asm", 6, "0.5", us=11.0, name="asm_sk6"),
        _row("asm", 4, "0.0", us=12.0, name="asm_sk4"),
    ]

    result = _tune(tmp_path, monkeypatch, 0.3, trial_max=6, profile=profile)

    assert result.dropped_inaccurate == []
    assert [r["kernelName"] for r in _read(Path(result.env_value))] == ["asm_sk4"]
