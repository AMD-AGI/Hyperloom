"""Tests for the MAIDAS Excel projection -> roofline ceiling override.

Self-contained: builds a tiny AllScenarios workbook in a tmp dir, so no
external MAIDAS artifact is required.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("openpyxl")

from hyperloom.orchestrator.kernel.roofline_ceiling import (
    compute_roofline_breakdown_from_state,
)


def _write_xlsx(path, *, soc="mi355x", bfp="FP8", hp=4, nbs=16,
                prefill=1024, decode=1024, avg_lat=23.32, spill=False):
    """Minimal AllScenarios sheet with a single uct_decode row."""
    df = pd.DataFrame(
        [{
            "workload": "llama405b", "soc": soc, "prefill": prefill,
            "decode": decode, "bfp": bfp, "nbs": nbs, "hp": hp,
            "pp": 1, "ep": 1, "cp": 1, "cpp": 1,
            "avg_lat": avg_lat, "QPS/TPS": 0.167, "nGPUs": hp,
            "spill": spill, "scenario": "uct_decode",
        }]
    )
    df.to_excel(path, sheet_name="AllScenarios", index=False)
    return str(path)


def _state(*, maidas_path, soc="mi355x", bfp="FP8", hp=4, nbs=16, isl=1024, osl=1024):
    return SimpleNamespace(
        gpu_type=soc, precision=bfp, framework="sglang",
        tp=hp, conc=nbs, isl=isl, osl=osl,
        model_path="/nonexistent/model",
        maidas_projection_path=maidas_path,
    )


def test_maidas_ceiling_used_when_path_set(tmp_path):
    xlsx = _write_xlsx(tmp_path / "l.xlsx", avg_lat=23.32, nbs=16)
    bd = compute_roofline_breakdown_from_state(_state(maidas_path=xlsx))
    assert bd.bound_kind == "maidas"
    # decode tok/s = nbs * (1000 / avg_lat_ms)
    assert bd.peak_tok_per_sec == pytest.approx(16 * 1000.0 / 23.32)
    assert bd.mem_tok_per_sec == bd.peak_tok_per_sec
    assert bd.cmp_tok_per_sec == bd.peak_tok_per_sec


def test_native_path_when_flag_unset(tmp_path):
    # No maidas path -> must NOT use MAIDAS (opt-in). Native path no-ops on a
    # bogus model dir, returning the empty/unknown breakdown.
    bd = compute_roofline_breakdown_from_state(_state(maidas_path=""))
    assert bd.bound_kind != "maidas"


def test_no_match_falls_back(tmp_path):
    # Workbook exists but has no row for this config -> fall back (not maidas).
    xlsx = _write_xlsx(tmp_path / "l.xlsx", soc="mi300x")  # different soc
    bd = compute_roofline_breakdown_from_state(_state(maidas_path=xlsx, soc="mi355x"))
    assert bd.bound_kind != "maidas"


def test_spilled_row_falls_back(tmp_path):
    # A spilled (infeasible) row must not be used as a ceiling.
    xlsx = _write_xlsx(tmp_path / "l.xlsx", spill=True)
    bd = compute_roofline_breakdown_from_state(_state(maidas_path=xlsx))
    assert bd.bound_kind != "maidas"
