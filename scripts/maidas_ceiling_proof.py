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
    """Find a uct_decode row we can target, and return its lookup key + avg_lat."""
    df = pd.read_excel(xlsx, sheet_name="AllScenarios")
    dec = df[(df["scenario"] == "uct_decode") & (~df["spill"].astype(bool))]
    # prefer a mid-size batch so the number is comfortably feasible
    for nbs in (16, 8, 32, 4, 64):
        cand = dec[dec["nbs"] == nbs]
        if not cand.empty:
            r = cand.iloc[0]
            return {
                "soc": str(r["soc"]),
                "bfp": str(r["bfp"]),
                "hp": int(r["hp"]),
                "nbs": int(r["nbs"]),
                "isl": int(r["prefill"]),
                "osl": int(r["decode"]),
                "avg_lat": float(r["avg_lat"]),
            }
    raise SystemExit("No feasible uct_decode row found in the workbook.")


def _fake_state(key, *, maidas_path):
    """A minimal stand-in for SharedState carrying just what the ceiling reads."""
    return SimpleNamespace(
        gpu_type=key["soc"],
        precision=key["bfp"],
        framework="sglang",
        tp=key["hp"],
        conc=key["nbs"],
        isl=key["isl"],
        osl=key["osl"],
        model_path="/nonexistent/model",   # native path would no-op; MAIDAS path returns first
        maidas_projection_path=maidas_path,
        # no last_baseline / current_best -> resolve_runtime_workload uses the attrs above
    )


def main():
    xlsx = sys.argv[1] if len(sys.argv) > 1 else (
        "/home/abdubey/Abhi/to_delete/maidas-er-main/llama_405B_nonB/l-c_report.xlsx"
    )
    key = _pick_matching_row(xlsx)
    expected_peak = key["nbs"] * (1000.0 / key["avg_lat"])

    print("=" * 74)
    print("MAIDAS CEILING PROOF")
    print("=" * 74)
    print(f"xlsx: {xlsx}")
    print(f"target row: soc={key['soc']} bfp={key['bfp']} hp={key['hp']} "
          f"nbs={key['nbs']} isl={key['isl']} osl={key['osl']} avg_lat={key['avg_lat']}ms")
    print(f"expected MAIDAS peak = nbs*(1000/avg_lat) = {expected_peak:.3f} tok/s")
    print("-" * 74)

    # (1) WITH the MAIDAS path -> ceiling must come from MAIDAS
    bd_maidas = compute_roofline_breakdown_from_state(_fake_state(key, maidas_path=xlsx))
    print(f"[WITH  path] bound_kind={bd_maidas.bound_kind!r}  "
          f"peak_tok_per_sec={bd_maidas.peak_tok_per_sec:.3f}")

    # (2) WITHOUT the path -> native path (opt-in proof)
    bd_native = compute_roofline_breakdown_from_state(_fake_state(key, maidas_path=""))
    print(f"[NO    path] bound_kind={bd_native.bound_kind!r}  "
          f"peak_tok_per_sec={bd_native.peak_tok_per_sec:.3f}")
    print("-" * 74)

    ok_used = bd_maidas.bound_kind == "maidas"
    ok_value = abs(bd_maidas.peak_tok_per_sec - expected_peak) < 1e-6
    ok_optin = bd_native.bound_kind != "maidas"

    print(f"PROOF 1 - MAIDAS was used (bound_kind=='maidas'):        {'PASS' if ok_used else 'FAIL'}")
    print(f"PROOF 2 - ceiling equals MAIDAS-derived value:           {'PASS' if ok_value else 'FAIL'}")
    print(f"PROOF 3 - native path when flag unset (opt-in):          {'PASS' if ok_optin else 'FAIL'}")
    print("=" * 74)

    if ok_used and ok_value and ok_optin:
        print("ALL PROOFS PASSED — Hyperloom used the MAIDAS projection for its roofline.")
        return 0
    print("PROOF FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
