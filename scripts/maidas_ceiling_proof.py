#!/usr/bin/env python3
"""Proof that Hyperloom's roofline ceiling is sourced from a MAIDAS projection
xlsx when --maidas-projection-path is set.

Drives the REAL production function
``compute_roofline_breakdown_from_state`` with a fake SharedState, once with the
MAIDAS path set and once without, and shows:

  * WITH path  -> RooflineBreakdown.bound_kind == "maidas" and peak matches the
                  value derived from the matching AllScenarios row.
  * WITHOUT    -> bound_kind != "maidas" (native path), proving it is opt-in.

Run:
    PYTHONPATH=src <venv>/bin/python scripts/maidas_ceiling_proof.py \
        /path/to/l-c_report.xlsx
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd

from hyperloom.orchestrator.kernel.roofline_ceiling import (
    compute_roofline_breakdown_from_state,
)


def _pick_matching_row(xlsx: str):
    """Find a feasible uct_decode row to target; key on gbs (global batch)."""
    df = pd.read_excel(xlsx, sheet_name="AllScenarios")
    # Feasible == gbs > 0 (MAIDAS zeroes gbs for spilled rows).
    dec = df[(df["scenario"] == "uct_decode") & (df["gbs"] > 0)]
    if dec.empty:
        raise SystemExit("No feasible uct_decode row found in the workbook.")
    # prefer a mid-size global batch so the number is comfortably feasible
    for gbs in (16, 8, 32, 4, 64):
        cand = dec[dec["gbs"] == gbs]
        if not cand.empty:
            r = cand.iloc[0]
            break
    else:
        r = dec.iloc[0]
    return {
        "soc": str(r["soc"]),
        "bfp": str(r["bfp"]),
        "hp": int(r["hp"]),
        "gbs": int(r["gbs"]),
        "isl": int(r["prefill"]),
        "osl": int(r["decode"]),
        "avg_lat": float(r["avg_lat"]),
    }


def _fake_state(key, *, maidas_path):
    """A minimal stand-in for SharedState carrying just what the ceiling reads."""
    return SimpleNamespace(
        gpu_type=key["soc"],
        precision=key["bfp"],
        framework="sglang",
        tp=key["hp"],
        conc=key["gbs"],   # Hyperloom concurrency == MAIDAS gbs (global batch)
        isl=key["isl"],
        osl=key["osl"],
        model_path="/nonexistent/model",   # native path no-ops (peak 0); MAIDAS path returns first
        maidas_projection_path=maidas_path,
        # no last_baseline / current_best -> resolve_runtime_workload uses the attrs above
    )


def main():
    xlsx = sys.argv[1] if len(sys.argv) > 1 else (
        "/home/abdubey/Abhi/to_delete/maidas-er-main/llama_405B_nonB/l-c_report.xlsx"
    )
    key = _pick_matching_row(xlsx)
    expected_peak = key["gbs"] * (1000.0 / key["avg_lat"])

    print("=" * 74)
    print("MAIDAS CEILING PROOF")
    print("=" * 74)
    print(f"xlsx: {xlsx}")
    print(f"target row: soc={key['soc']} bfp={key['bfp']} hp={key['hp']} "
          f"gbs(conc)={key['gbs']} isl={key['isl']} osl={key['osl']} avg_lat={key['avg_lat']}ms")
    print(f"expected MAIDAS peak = gbs*(1000/avg_lat) = {expected_peak:.3f} tok/s")
    print("-" * 74)

    # (1) WITH the MAIDAS path -> ceiling must come from MAIDAS
    bd_maidas = compute_roofline_breakdown_from_state(_fake_state(key, maidas_path=xlsx))
    print(f"[WITH  path] bound_kind={bd_maidas.bound_kind!r}  "
          f"peak_tok_per_sec={bd_maidas.peak_tok_per_sec:.3f}  cmp={bd_maidas.cmp_tok_per_sec:.3f}")

    # (2) WITHOUT the path -> native path (no-ops on bogus model -> peak 0)
    bd_native = compute_roofline_breakdown_from_state(_fake_state(key, maidas_path=""))
    print(f"[NO    path] bound_kind={bd_native.bound_kind!r}  "
          f"peak_tok_per_sec={bd_native.peak_tok_per_sec:.3f}")
    print("-" * 74)

    # MAIDAS now returns bound_kind='memory' (a valid Hyperloom enum); it's
    # distinguished from native by producing a positive peak (native no-ops
    # on the bogus model path -> peak 0). cmp is left unset (0.0).
    ok_used = bd_maidas.peak_tok_per_sec > 0 and bd_maidas.bound_kind == "memory"
    ok_value = abs(bd_maidas.peak_tok_per_sec - expected_peak) < 1e-6
    ok_cmp = bd_maidas.cmp_tok_per_sec == 0.0
    ok_optin = bd_native.peak_tok_per_sec == 0.0

    print(f"PROOF 1 - MAIDAS was used (peak>0, bound_kind=='memory'): {'PASS' if ok_used else 'FAIL'}")
    print(f"PROOF 2 - ceiling equals gbs*(1000/avg_lat):             {'PASS' if ok_value else 'FAIL'}")
    print(f"PROOF 3 - cmp side left unset (0.0, no fake split):       {'PASS' if ok_cmp else 'FAIL'}")
    print(f"PROOF 4 - native (peak 0) when flag unset (opt-in):       {'PASS' if ok_optin else 'FAIL'}")
    print("=" * 74)
    ok_used = ok_used and ok_cmp

    if ok_used and ok_value and ok_optin:
        print("ALL PROOFS PASSED — Hyperloom used the MAIDAS projection for its roofline.")
        return 0
    print("PROOF FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
