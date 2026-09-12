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
        logger.warning("pandas unavailable; MAIDAS projection %s cannot be read "
                       "-> using native ceiling", path)
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


#: Map Hyperloom precision/dtype tags (its ``_DTYPE_BYTES`` / ``_QUANT_WEIGHT_BYTES``
#: vocabulary) to MAIDAS ``bfp`` tokens. ALIASES ONLY — tokens that already match
#: after lowercasing (``fp8``, ``bf16``, ``mx4`` from MAIDAS, ...) are handled by
#: the passthrough in ``_bfp_token``, so they are not listed.
#:
#: We deliberately do NOT reuse Hyperloom's own precision canonicalizer
#: (``cli/__init__``), which collapses ``mxfp4 -> fp4`` (same byte width): MAIDAS
#: models ``MX4`` (microscaling) and ``FP4`` (plain) as DISTINCT SoC formats with
#: different peak FLOPS, so that distinction must be preserved for the match.
_PRECISION_ALIASES = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
    "float8_e4m3fn": "fp8", "float8_e5m2": "fp8",
    "fp8_e4m3": "fp8", "fp8_e5m2": "fp8",
    "mxfp8": "mx8",
    "float4": "fp4",
    "mxfp4": "mx4",
}


def _bfp_token(precision: str) -> str:
    p = (precision or "").strip().lower()
    return _PRECISION_ALIASES.get(p, p)  # passthrough: fp8/bf16/mx4/... already match


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
    isl, osl). Uses the ``uct_decode`` scenario row (output-token throughput).

    Matching is on ``gbs`` (MAIDAS **global** batch = ``nbs*pp*dp``), because
    Hyperloom's ``concurrency`` is the whole-server in-flight count (the vLLM
    ``--max-concurrency``), i.e. the global batch — not the per-replica ``nbs``.
    ``gbs`` is 0 for spilled/infeasible rows, so a ``concurrency > 0`` never
    matches one; feasibility falls out of the match for free.

    Matching is EXACT on ``gbs`` (and precision is alias-normalized so e.g.
    ``mxfp4`` matches ``MX4``). If ``concurrency`` is not a swept ``gbs`` value,
    no row matches and we return ``None`` -> native fallback. We deliberately do
    NOT interpolate/nearest-match: the ceiling is not linear in batch, so a
    fabricated value would be worse than the native estimate.
    """
    df = _load_allscenarios(path)
    if df is None or getattr(df, "empty", True):
        return None

    needed = {"soc", "bfp", "hp", "gbs", "prefill", "decode", "avg_lat", "scenario"}
    if not needed.issubset(set(df.columns)):
        logger.warning("MAIDAS xlsx %s missing required columns %s -> using "
                       "native ceiling", path, sorted(needed - set(df.columns)))
        return None

    soc = _soc_token(runtime.gpu_type)
    prec = _bfp_token(runtime.precision)
    conc = int(runtime.concurrency or 0)
    if conc <= 0:
        return None  # degenerate concurrency cannot map to a global batch

    # Config filter WITHOUT the batch, so we can tell "no data" from
    # "config spills at this batch" (gbs == 0 rows).
    try:
        cfg = df[
            (df["soc"].astype(str).str.lower() == soc)
            & (df["bfp"].astype(str).str.lower().map(_bfp_token) == prec)
            & (df["hp"] == int(runtime.tp or 0))
            & (df["prefill"] == int(runtime.isl or 0))
            & (df["decode"] == int(runtime.osl or 0))
            & (df["scenario"].astype(str) == "uct_decode")
        ]
        q = cfg[cfg["gbs"] == conc]
    except Exception:  # noqa: BLE001 - defensive against odd dtypes
        return None

    if q.empty:
        spills = (not cfg.empty) and bool((cfg["gbs"] == 0).any())
        # INFO (not debug): the adapter only runs when a projection path was
        # provided, so the operator wants to know it didn't apply here.
        logger.info(
            "MAIDAS projection provided but no feasible row for soc=%s bfp=%s "
            "hp=%s gbs(conc)=%s isl=%s osl=%s%s -> using native ceiling",
            soc, prec, runtime.tp, conc, runtime.isl, runtime.osl,
            " (config spills/infeasible at this batch)" if spills else "",
        )
        return None

    try:
        lat_ms = float(q.iloc[0]["avg_lat"])
    except Exception:  # noqa: BLE001
        return None
    if lat_ms <= 0:  # inf/0 latency == infeasible (gbs>0 already implies feasible)
        return None

    # Whole-server decode-output throughput ceiling (tok/s):
    #   gbs * (1000 / TPOT_ms)   -- and gbs == conc for the matched row.
    peak = float(conc) * (1000.0 / lat_ms)
    if peak <= 0:
        return None

    logger.info(
        "MAIDAS PROJECTION USED for roofline ceiling | source=%s | "
        "matched row: workload=%s soc=%s bfp=%s hp(TP)=%s gbs(conc)=%s "
        "prefill(ISL)=%s decode(OSL)=%s scenario=uct_decode | "
        "data used: avg_lat(decode TPOT)=%.3f ms | "
        "computed ceiling: peak=%.2f tok/s (= gbs %s x 1000 / avg_lat, memory-bound)",
        path,
        str(q.iloc[0].get("workload", "?")) if "workload" in q.columns else "?",
        soc, prec, runtime.tp, conc, runtime.isl, runtime.osl, lat_ms, peak, conc,
    )
    # Decode is memory-bound, so the memory ceiling *is* the peak. MAIDAS
    # AllScenarios carries no compute-bound side projection, so cmp is left
    # unset (0.0 -> nulled by build_roofline_snapshot) rather than faked.
    return RooflineBreakdown(
        mem_tok_per_sec=peak,
        cmp_tok_per_sec=0.0,
        peak_tok_per_sec=peak,
        bound_kind="memory",
    )
