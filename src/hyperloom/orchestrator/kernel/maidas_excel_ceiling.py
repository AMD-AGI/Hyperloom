"""MAIDAS Excel projection -> roofline ceiling.

When the user passes ``--maidas-projection-path`` pointing at a MAIDAS
projection workbook (``*_report.xlsx``, the ``AllScenarios`` sheet) or a
directory of them, this module looks up the row matching the current run's
``(soc, precision, TP, concurrency, ISL, OSL)`` and returns a
``RooflineBreakdown`` built from MAIDAS's predicted decode latency.

The whole thing is best-effort: any parse error, missing file, or missing row
returns ``None`` so the caller silently falls back to the native ceiling.
"""

from __future__ import annotations

import glob
import logging
import os
from functools import lru_cache
from typing import Any

from .roofline_ceiling import RooflineBreakdown

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _load_allscenarios(path: str):
    """Load and concatenate the ``AllScenarios`` sheet from one xlsx or a dir."""
    try:
        import pandas as pd
    except Exception:  # noqa: BLE001 - pandas is optional at runtime
        logger.debug("pandas unavailable; MAIDAS ceiling disabled")
        return None

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.xlsx")))
    elif path.endswith(".xlsx") and os.path.exists(path):
        files = [path]
    else:
        return None

    frames = []
    for f in files:
        try:
            frames.append(pd.read_excel(f, sheet_name="AllScenarios"))
        except Exception:  # noqa: BLE001 - skip unreadable/foreign workbooks
            continue
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _soc_token(gpu_type: str) -> str:
    """Normalize the run's GPU name to a MAIDAS ``soc`` token.

    ``runtime.gpu_type`` is already a resolved AMD mi-name (mi300x/mi325x/
    mi355x/...) — the same tokens MAIDAS uses in the ``soc`` column — so we only
    lowercase/strip and pass it through. We deliberately do NOT reuse
    ``gpu_types._gpu_runner_type`` (it collapses mi325x/mi308x -> mi300x for the
    Magpie script, which would match the wrong MAIDAS SoC), and we do NOT extend
    the ``AMD_GPU_DISPATCH_IDENTITIES`` allowlist (that gate flags unsupported
    GPUs). New GPUs (e.g. mi4xx) match automatically once Hyperloom resolves
    them, with no change here.
    """
    return (gpu_type or "").strip().lower()


def maidas_breakdown_from_excel(path: str, runtime: Any) -> RooflineBreakdown | None:
    """Return a MAIDAS-derived ``RooflineBreakdown`` for *runtime*, or ``None``.

    ``runtime`` is a ``RuntimeWorkload`` (gpu_type, precision, tp, concurrency,
    isl, osl). Uses the ``uct_decode`` scenario row (output-token throughput)
    and converts MAIDAS's per-token decode latency into a tok/s ceiling.
    """
    df = _load_allscenarios(path)
    if df is None or getattr(df, "empty", True):
        return None

    needed = {"soc", "bfp", "hp", "nbs", "prefill", "decode", "avg_lat", "scenario"}
    if not needed.issubset(set(df.columns)):
        logger.debug("MAIDAS xlsx missing required columns; skipping")
        return None

    soc = _soc_token(runtime.gpu_type)
    prec = (runtime.precision or "").strip().lower()

    try:
        q = df[
            (df["soc"].astype(str).str.lower() == soc)
            & (df["bfp"].astype(str).str.lower() == prec)
            & (df["hp"] == int(runtime.tp or 0))
            & (df["nbs"] == int(runtime.concurrency or 0))
            & (df["prefill"] == int(runtime.isl or 0))
            & (df["decode"] == int(runtime.osl or 0))
            & (df["scenario"].astype(str) == "uct_decode")
        ]
    except Exception:  # noqa: BLE001 - defensive against odd dtypes
        return None

    if q.empty:
        logger.debug(
            "no MAIDAS row for soc=%s bfp=%s hp=%s nbs=%s isl=%s osl=%s",
            soc, prec, runtime.tp, runtime.concurrency, runtime.isl, runtime.osl,
        )
        return None

    try:
        lat_ms = float(q.iloc[0]["avg_lat"])
        spilled = bool(q.iloc[0].get("spill", False)) if "spill" in q.columns else False
    except Exception:  # noqa: BLE001
        return None

    if lat_ms <= 0 or spilled:
        return None

    # Unit reconciliation: decode tok/s = concurrency * (1000 / TPOT_ms).
    peak = float(runtime.concurrency or 1) * (1000.0 / lat_ms)
    if peak <= 0:
        return None

    logger.info(
        "MAIDAS PROJECTION USED for roofline ceiling | source=%s | "
        "matched row: workload=%s soc=%s bfp=%s hp(TP)=%s nbs(conc)=%s "
        "prefill(ISL)=%s decode(OSL)=%s scenario=uct_decode | "
        "data used: avg_lat(decode TPOT)=%.3f ms spill=%s | "
        "computed ceiling: peak=%.2f tok/s (= conc %s x 1000 / avg_lat)",
        path,
        str(q.iloc[0].get("workload", "?")) if "workload" in q.columns else "?",
        soc, prec, runtime.tp, runtime.concurrency,
        runtime.isl, runtime.osl,
        lat_ms, spilled, peak, runtime.concurrency,
    )
    # MAIDAS AllScenarios does not split mem/compute bound; expose peak on both.
    return RooflineBreakdown(
        mem_tok_per_sec=peak,
        cmp_tok_per_sec=peak,
        peak_tok_per_sec=peak,
        bound_kind="maidas",
    )
