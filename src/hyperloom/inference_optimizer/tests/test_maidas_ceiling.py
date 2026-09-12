"""Tests for the MAIDAS Excel projection -> roofline ceiling override.

Self-contained: builds a tiny AllScenarios workbook in a tmp dir, so no
external MAIDAS artifact is required.

Matching is on the MAIDAS ``gbs`` (global batch) column, which equals
Hyperloom's whole-server ``concurrency``. MAIDAS zeroes ``gbs`` for spilled
rows, so a ``concurrency > 0`` never matches an infeasible config. The bogus
``model_path`` in ``_state`` makes the *native* fallback a no-op (peak == 0),
so a successful MAIDAS hit is distinguished by ``peak > 0``.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("openpyxl")

from hyperloom.orchestrator.kernel.roofline_ceiling import (
    compute_roofline_breakdown_from_state,
)


def _write_xlsx(path, *, soc="mi355x", bfp="FP8", nbs=16, gbs=None, hp=4,
                prefill=1024, decode=1024, avg_lat=23.32, spill=False):
    """Minimal AllScenarios sheet with a single uct_decode row.

    ``gbs`` defaults to ``nbs`` (pp=dp=1). Spilled rows carry ``gbs=0`` (and
    ``avg_lat`` would be inf), mirroring MAIDAS's infeasibility marker.
    """
    g = 0 if spill else (nbs if gbs is None else gbs)
    df = pd.DataFrame([{
        "workload": "llama405b", "soc": soc, "prefill": prefill,
        "decode": decode, "bfp": bfp, "nbs": nbs, "hp": hp,
        "pp": 1, "ep": 1, "cp": 1, "cpp": 1,
        "avg_lat": avg_lat, "QPS/TPS": 0.167, "gbs": g, "nGPUs": hp,
        "spill": spill, "scenario": "uct_decode",
    }])
    df.to_excel(path, sheet_name="AllScenarios", index=False)
    return str(path)


def _state(*, maidas_path, soc="mi355x", bfp="FP8", hp=4, conc=16, isl=1024, osl=1024):
    return SimpleNamespace(
        gpu_type=soc, precision=bfp, framework="sglang",
        tp=hp, conc=conc, isl=isl, osl=osl,
        model_path="/nonexistent/model",
        maidas_projection_path=maidas_path,
    )


def test_maidas_ceiling_used_when_path_set(tmp_path):
    xlsx = _write_xlsx(tmp_path / "l.xlsx", avg_lat=23.32, nbs=16)  # gbs==16
    bd = compute_roofline_breakdown_from_state(_state(maidas_path=xlsx, conc=16))
    assert bd.bound_kind == "memory"
    # whole-server decode tok/s = gbs * (1000 / avg_lat_ms), gbs == conc == 16
    assert bd.peak_tok_per_sec == pytest.approx(16 * 1000.0 / 23.32)
    assert bd.mem_tok_per_sec == bd.peak_tok_per_sec
    assert bd.cmp_tok_per_sec == 0.0  # no compute-bound side in AllScenarios


def test_native_path_when_flag_unset(tmp_path):
    # No maidas path -> native path (no-op on the bogus model dir -> peak 0).
    bd = compute_roofline_breakdown_from_state(_state(maidas_path=""))
    assert bd.peak_tok_per_sec == 0.0


def test_no_match_falls_back(tmp_path):
    # Workbook exists but no row for this config (different soc) -> native (peak 0).
    xlsx = _write_xlsx(tmp_path / "l.xlsx", soc="mi300x")
    bd = compute_roofline_breakdown_from_state(_state(maidas_path=xlsx, soc="mi355x"))
    assert bd.peak_tok_per_sec == 0.0


def test_spilled_row_is_skipped(tmp_path):
    # Spilled row -> gbs == 0 -> concurrency 16 never matches -> native (peak 0).
    xlsx = _write_xlsx(tmp_path / "l.xlsx", spill=True)  # gbs forced to 0
    bd = compute_roofline_breakdown_from_state(_state(maidas_path=xlsx, conc=16))
    assert bd.peak_tok_per_sec == 0.0


def test_match_is_on_gbs_not_nbs(tmp_path):
    # Row with nbs=8 but gbs=16 (as a pp=2 row would have). concurrency=16 must
    # match on gbs (16), not nbs (8). And concurrency=8 must NOT match this row.
    xlsx = _write_xlsx(tmp_path / "l.xlsx", nbs=8, gbs=16, avg_lat=23.32)
    hit = compute_roofline_breakdown_from_state(_state(maidas_path=xlsx, conc=16))
    assert hit.bound_kind == "memory"
    assert hit.peak_tok_per_sec == pytest.approx(16 * 1000.0 / 23.32)
    miss = compute_roofline_breakdown_from_state(_state(maidas_path=xlsx, conc=8))
    assert miss.peak_tok_per_sec == 0.0  # nbs=8 must not be matched


def test_cross_model_projection_warns(tmp_path, caplog):
    # Row is matched on hardware shape only; if the served model differs from the
    # xlsx workload the ceiling is for a DIFFERENT model -> must warn loudly.
    import logging
    xlsx = _write_xlsx(tmp_path / "l.xlsx", avg_lat=23.32, nbs=16)  # workload=llama405b
    st = _state(maidas_path=xlsx, conc=16)
    st.model_path = "/models/gpt-oss-120b"  # clearly not llama405b
    with caplog.at_level(logging.WARNING):
        bd = compute_roofline_breakdown_from_state(st)
    assert bd.peak_tok_per_sec == pytest.approx(16 * 1000.0 / 23.32)  # still returned
    assert any("CROSS-MODEL" in r.message for r in caplog.records)


def test_matching_model_does_not_warn(tmp_path, caplog):
    # When the served model reconciles with the xlsx workload, no cross-model warn.
    import logging
    xlsx = _write_xlsx(tmp_path / "l.xlsx", avg_lat=23.32, nbs=16)  # workload=llama405b
    st = _state(maidas_path=xlsx, conc=16)
    st.model_path = "/models/llama405b"
    with caplog.at_level(logging.WARNING):
        compute_roofline_breakdown_from_state(st)
    assert not any("CROSS-MODEL" in r.message for r in caplog.records)


def test_precision_alias_mxfp4_matches_MX4(tmp_path):
    # MAIDAS emits bfp='MX4'; a Hyperloom run reporting precision='mxfp4' (as in
    # the Llama-405B-MXFP4 benchmark) must still match, not silently fall back.
    xlsx = _write_xlsx(tmp_path / "l.xlsx", bfp="MX4", nbs=16, avg_lat=23.32)
    bd = compute_roofline_breakdown_from_state(
        _state(maidas_path=xlsx, bfp="mxfp4", conc=16))
    assert bd.bound_kind == "memory"
    assert bd.peak_tok_per_sec == pytest.approx(16 * 1000.0 / 23.32)


def test_precision_alias_fp8_variants_match(tmp_path):
    # fp8_e4m3 (a common serving tag) must match MAIDAS 'FP8'.
    xlsx = _write_xlsx(tmp_path / "l.xlsx", bfp="FP8", nbs=16, avg_lat=23.32)
    bd = compute_roofline_breakdown_from_state(
        _state(maidas_path=xlsx, bfp="fp8_e4m3", conc=16))
    assert bd.bound_kind == "memory"


def _write_multi(path, rows):
    """AllScenarios sheet with several uct_decode rows; gbs defaults to nbs."""
    base = dict(workload="llama405b", soc="mi355x", prefill=1024, decode=1024,
                bfp="FP8", hp=4, pp=1, ep=1, cp=1, cpp=1, spill=False,
                scenario="uct_decode", QPS_TPS=0.167)
    recs = []
    for r in rows:
        rec = {**base, **r}
        rec.setdefault("gbs", rec["nbs"])  # pp=1 -> gbs == nbs
        recs.append(rec)
    pd.DataFrame(recs).to_excel(path, sheet_name="AllScenarios", index=False)
    return str(path)


def test_conc_sweep_ceiling_uses_maidas_per_rung(tmp_path, monkeypatch):
    # The concurrency sweep must price each rung from MAIDAS (via the adapter).
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
    assert by_conc[8]["bound_kind"] == "memory"
    assert by_conc[8]["t_peak_tok_s"] == pytest.approx(round(8 * 1000.0 / 30.0, 2))
    assert by_conc[16]["bound_kind"] == "memory"
    assert by_conc[16]["t_peak_tok_s"] == pytest.approx(round(16 * 1000.0 / 23.32, 2))
    # Provenance: both rungs priced from MAIDAS, surfaced per-row and top-level.
    assert by_conc[8]["ceiling_source"] == "maidas"
    assert by_conc[16]["ceiling_source"] == "maidas"
    assert out["maidas_rungs"] == 2
    assert out["rungs_total"] == 2


def test_conc_sweep_records_source_per_rung_mixed(tmp_path, monkeypatch):
    # A spilled rung (gbs=0) must fall back to native and be labelled as such,
    # while a feasible rung is labelled maidas — provenance is per-rung.
    import hyperloom.orchestrator.actions.executors  # noqa: F401
    from hyperloom.orchestrator.kernel import conc_sweep as cs
    # nbs=4 spills (gbs=0); nbs=8 is feasible.
    xlsx = _write_multi(tmp_path / "m.xlsx",
                        [{"nbs": 4, "gbs": 0, "avg_lat": 0.0},
                         {"nbs": 8, "avg_lat": 30.0}])
    dummy = SimpleNamespace(weight_bytes=1, active_weight_bytes=1, num_experts=0,
                            experts_per_tok=0, expert_weight_bytes=0, num_layers=1,
                            num_kv_heads=1, head_dim=1, weight_dtype_bytes=2)
    monkeypatch.setattr(cs, "load_model_meta", lambda *a, **k: dummy)
    out = cs._build_roofline_ceiling(
        _state(maidas_path=xlsx), concs=[4, 8], isl=1024, osl=1024,
        baseline_points=[], optimized_points=[],
    )
    by_conc = {r["conc"]: r for r in out["rows"]}
    assert by_conc[4]["ceiling_source"] == "native"   # spill -> native
    assert by_conc[8]["ceiling_source"] == "maidas"    # feasible -> maidas
    assert out["maidas_rungs"] == 1
    assert out["rungs_total"] == 2
