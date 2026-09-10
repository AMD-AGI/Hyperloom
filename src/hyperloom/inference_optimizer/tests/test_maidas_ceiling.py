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


def _write_multi(path, rows):
    """Write an AllScenarios sheet with several uct_decode rows (per nbs)."""
    base = dict(workload="llama405b", soc="mi355x", prefill=1024, decode=1024,
                bfp="FP8", hp=4, pp=1, ep=1, cp=1, cpp=1, spill=False,
                scenario="uct_decode")
    pd.DataFrame([{**base, **r} for r in rows]).to_excel(
        path, sheet_name="AllScenarios", index=False)
    return str(path)


def test_conc_sweep_ceiling_uses_maidas_per_rung(tmp_path, monkeypatch):
    # The concurrency sweep must price each rung from MAIDAS (via the seam).
    # Import actions.executors first to establish the kernel<->executors import
    # order (a pre-existing circular import between the two conc_sweep modules).
    import hyperloom.orchestrator.actions.executors  # noqa: F401
    from hyperloom.orchestrator.kernel import conc_sweep as cs
    xlsx = _write_multi(tmp_path / "m.xlsx",
                        [{"nbs": 8, "avg_lat": 30.0}, {"nbs": 16, "avg_lat": 23.32}])
    # _build_roofline_ceiling early-guards on load_model_meta; supply a stub so
    # the model_meta report block has something (ceiling itself comes from MAIDAS).
    dummy = SimpleNamespace(weight_bytes=1, active_weight_bytes=1, num_experts=0,
                            experts_per_tok=0, expert_weight_bytes=0, num_layers=1,
                            num_kv_heads=1, head_dim=1, weight_dtype_bytes=2)
    monkeypatch.setattr(cs, "load_model_meta", lambda *a, **k: dummy)
    out = cs._build_roofline_ceiling(
        _state(maidas_path=xlsx), concs=[8, 16], isl=1024, osl=1024,
        baseline_points=[], optimized_points=[],
    )
    assert out is not None
    by_conc = {r["conc"]: r for r in out["rows"]}
    assert by_conc[8]["bound_kind"] == "maidas"
    assert by_conc[8]["t_peak_tok_s"] == pytest.approx(round(8 * 1000.0 / 30.0, 2))
    assert by_conc[16]["bound_kind"] == "maidas"
    assert by_conc[16]["t_peak_tok_s"] == pytest.approx(round(16 * 1000.0 / 23.32, 2))
