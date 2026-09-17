"""MAIDAS-sourced L2 per-op PerfModel breakdown (``Verbose_Data`` sheet).

Covers: block-per-batch selection, nlayers scaling, reconciliation with the L1
``avg_lat`` ceiling (the consistency gate), and fail-soft fallback.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

# actions.executors first: pre-existing kernel<->executors circular import.
import hyperloom.orchestrator.actions.executors  # noqa: F401,E402
from hyperloom.orchestrator.kernel.maidas_excel_ceiling import (  # noqa: E402
    maidas_perfmodel_from_excel,
)

_NL = 2  # tiny model: 2 layers


def _write(path, *, avg_lat_scale=1.0):
    """A minimal workbook: two per-batch blocks (gbs 4/8), one op + its GEMM child.

    ``avg_lat`` is set so the L1 ceiling equals the Verbose-derived peak
    (``gbs*1e6/(time_us*nlayers)``); ``avg_lat_scale`` perturbs it to trip the gate.
    """
    vd = []
    for gbs, t in [(4, 10.0), (8, 20.0)]:
        common = dict(arch="mi355x", phase="decode_100", time_us=t, ideal_us=1.0,
                      read_us=8.0, write_us=2.0, tflops=100.0, perc_time_profile=100.0)
        vd.append({**common, "layer": "L", "M": 0})            # top-level parent
        vd.append({**common, "layer": "L/GEMM", "M": gbs})     # GEMM child carries batch
    asc = [dict(workload="m", soc="mi355x", bfp="FP8", hp=1, prefill=100, decode=100,
                nbs=gbs, gbs=gbs, avg_lat=t * _NL / 1000.0 * avg_lat_scale,
                scenario="uct_decode")
           for gbs, t in [(4, 10.0), (8, 20.0)]]
    uct = [{"soc name": "mi355x", "hbm_mem_bw": 8000, "mfma_flops": 3.5e15}]
    with pd.ExcelWriter(path) as w:
        pd.DataFrame(asc).to_excel(w, sheet_name="AllScenarios", index=False)
        pd.DataFrame(vd).to_excel(w, sheet_name="Verbose_Data", index=False)
        pd.DataFrame(uct).to_excel(w, sheet_name="uct_decode", index=False)
    return str(path)


def _rt(gbs):
    return SimpleNamespace(gpu_type="mi355x", precision="fp8", tp=1, concurrency=gbs,
                           isl=100, osl=100, model_path="m")


def test_l2_builds_reconciles_and_scales(tmp_path):
    f = _write(tmp_path / "m.xlsx")
    bd = maidas_perfmodel_from_excel(f, _rt(4), num_layers=_NL)
    assert bd is not None
    # gbs*1e6 / (time_us=10 * nlayers=2) == 4e6/20 == 200000; equals L1 gbs*1000/avg_lat.
    assert bd.decode_tok_per_s == pytest.approx(200000.0, rel=1e-6)
    assert bd.bound_kind == "memory"          # read+write(10) > ideal(1)
    assert bd.decode_mem_tok_per_s < bd.decode_cmp_tok_per_s
    assert bd.hbm_bw_gbps == 8000 and bd.peak_achievable_tflops == pytest.approx(3500.0)
    assert len(bd.ops) == 1 and bd.ops[0].name == "L"      # top-level only, no double-count
    assert bd.ops[0].time_s == pytest.approx(10.0 * _NL / 1e6)  # scaled by nlayers


def test_l2_batch_selection(tmp_path):
    f = _write(tmp_path / "m.xlsx")
    # gbs=8 must pick the 20us block, not the 10us one.
    bd = maidas_perfmodel_from_excel(f, _rt(8), num_layers=_NL)
    assert bd.decode_tok_per_s == pytest.approx(8e6 / (20.0 * _NL), rel=1e-6)


def test_l2_consistency_gate_rejects_divergence(tmp_path):
    # avg_lat off by 2x => L2 peak diverges >10% from L1 => None (native fallback).
    f = _write(tmp_path / "m.xlsx", avg_lat_scale=2.0)
    assert maidas_perfmodel_from_excel(f, _rt(4), num_layers=_NL) is None


def test_l2_fallback_when_no_verbose_sheet(tmp_path):
    with pd.ExcelWriter(tmp_path / "n.xlsx") as w:
        pd.DataFrame([{"soc": "mi355x"}]).to_excel(w, sheet_name="AllScenarios", index=False)
    assert maidas_perfmodel_from_excel(str(tmp_path / "n.xlsx"), _rt(4), num_layers=_NL) is None
