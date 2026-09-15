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
import re
from functools import lru_cache
from typing import Any

from .roofline_ceiling import RooflineBreakdown

logger = logging.getLogger(__name__)


@lru_cache(maxsize=16)
def _load_sheet(path: str, sheet: str):
    """Load and concatenate one named sheet from a single xlsx or a directory.

    Returns a concatenated DataFrame, or ``None`` if pandas is unavailable, the
    path is not a readable xlsx/dir, or no file carries the sheet (workbooks
    missing it are skipped). Shared by every MAIDAS reader so the file/dir and
    fail-soft semantics live in exactly one place.
    """
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
            frames.append(pd.read_excel(f, sheet_name=sheet))
        except Exception:  # noqa: BLE001 - skip unreadable/foreign workbooks
            continue
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _load_allscenarios(path: str):
    """The per-config decode/prefill summary sheet (L1 ceiling source)."""
    return _load_sheet(path, "AllScenarios")


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


def _norm_model(s: str) -> str:
    """Normalize a model / workload label to bare alphanumerics for comparison."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


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

    row_workload = str(q.iloc[0].get("workload", "")) if "workload" in q.columns else ""
    served = os.path.basename(str(getattr(runtime, "model_path", "") or "").rstrip("/"))
    # Cross-model guard: the row is matched on hardware shape only (soc/bfp/hp/gbs/
    # isl/osl), NOT on model. Applying a projection built for a DIFFERENT model
    # yields an invalid ceiling (empirically the measured throughput can exceed
    # it). Warn loudly when the xlsx workload cannot be reconciled with the served
    # model so the mis-projection is visible rather than silent.
    nm_served, nm_wl = _norm_model(served), _norm_model(row_workload)
    if nm_served and nm_wl and nm_served not in nm_wl and nm_wl not in nm_served:
        logger.warning(
            "MAIDAS CROSS-MODEL projection: xlsx workload=%r does not match served "
            "model=%r. The ceiling is for a DIFFERENT model (matched only on "
            "hardware shape) and may be INVALID. Provide the MAIDAS xlsx for the "
            "served model.", row_workload, served,
        )
    logger.info(
        "MAIDAS PROJECTION USED for roofline ceiling | source=%s | "
        "matched row: workload=%s soc=%s bfp=%s hp(TP)=%s gbs(conc)=%s "
        "prefill(ISL)=%s decode(OSL)=%s scenario=uct_decode | served_model=%s | "
        "data used: avg_lat(decode TPOT)=%.3f ms | "
        "computed ceiling: peak=%.2f tok/s (= gbs %s x 1000 / avg_lat, memory-bound)",
        path, row_workload or "?",
        soc, prec, runtime.tp, conc, runtime.isl, runtime.osl, served or "?",
        lat_ms, peak, conc,
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


# --------------------------------------------------------------------------- #
# L2: per-op PerfModel breakdown sourced from the ``Verbose_Data`` sheet.
# ``arch`` / ``soc name`` are matched verbatim against the run soc, exactly like
# the L1 adapter matches ``AllScenarios.soc`` — any GPU-name normalization must
# live in the shared ``_soc_token`` so L1 and L2 stay consistent.
# --------------------------------------------------------------------------- #
def _maidas_hw_consts(path: str, soc: str) -> tuple[float, float]:
    """``(hbm_bw_gbps, peak_tflops)`` for *soc* from the per-scenario HW columns.

    The ``uct_decode`` sheet carries one HW row per soc; picking the wrong row
    would use another GPU's bandwidth/FLOPS, so we filter on ``soc name``.
    """
    df = _load_sheet(path, "uct_decode")
    if df is None or "soc name" not in getattr(df, "columns", []):
        return 0.0, 0.0
    row = df[df["soc name"].astype(str).str.lower() == soc]
    if row.empty:
        return 0.0, 0.0
    try:
        return float(row.iloc[0].get("hbm_mem_bw") or 0.0), \
            float(row.iloc[0].get("mfma_flops") or 0.0) / 1e12
    except Exception:  # noqa: BLE001
        return 0.0, 0.0


def _select_layer_block(vd, soc: str, phase: str, seq_len: int, batch: int):
    """Top-level rows of the single transformer block matching the run, or None.

    ``Verbose_Data`` repeats ONE block per swept batch (parents carry ``M=0``;
    only ``*/GEMM`` children carry the batch), tagged by ``arch`` (soc) and
    ``phase`` (``<phase>_<seq_len>``). We slice to (soc, phase), split into
    per-block segments (a block restarts at the first op name), tag each block by
    its max GEMM ``M`` (= batch), keep the one where ``M == batch``, and return
    its top-level ops (``layer`` without ``/``) — the finest level that sums
    without double-counting parent+child.
    """
    if not {"arch", "phase", "layer", "M", "time_us"}.issubset(set(vd.columns)):
        return None
    seg = vd[(vd["arch"].astype(str).str.lower() == soc)
             & (vd["phase"].astype(str) == f"{phase}_{int(seq_len)}")]
    if seg.empty:
        return None
    seg = seg.reset_index(drop=True)
    first = str(seg["layer"].iloc[0])
    blk = (seg["layer"].astype(str) == first).cumsum()
    for _, block in seg.groupby(blk):
        gemm = block[block["layer"].astype(str).str.contains("/GEMM")]
        if not gemm.empty and int(gemm["M"].max()) == batch:
            top = block[~block["layer"].astype(str).str.contains("/")]
            return top[top["time_us"] > 0]
    return None


def maidas_perfmodel_from_excel(path: str, runtime: Any, *, num_layers: int):
    """MAIDAS-sourced L2 ``PerfModelBreakdown`` for *runtime*, or ``None``.

    Builds Hyperloom's bottom-up per-op breakdown from ``Verbose_Data`` so L2 is
    consistent with the L1 ``avg_lat`` ceiling instead of Hyperloom's independent
    PerfModel. Per-op fields come straight from MAIDAS: ``ideal_us`` = compute
    time, ``read_us+write_us`` = memory time, ``time_us`` = op time; each is a
    single transformer block, scaled by ``num_layers`` for the whole model.
    Fail-soft: any miss (no sheet, no matching block, or a >10% divergence from
    the L1 ceiling) returns ``None`` so the caller keeps the native PerfModel.
    """
    from .roofline_ceiling import OpBreakdown, PerfModelBreakdown  # reuse L2 types

    vd = _load_sheet(path, "Verbose_Data")
    if vd is None or getattr(vd, "empty", True) or num_layers <= 0:
        return None
    soc = _soc_token(runtime.gpu_type)
    gbs = int(runtime.concurrency or 0)
    if gbs <= 0:
        return None

    dec = _select_layer_block(vd, soc, "decode", runtime.osl or 0, gbs)
    if dec is None or dec.empty:
        return None
    bw_gbps, peak_tflops = _maidas_hw_consts(path, soc)
    bw_bps = bw_gbps * 1e9

    # Whole-model us = per-block us x num_layers.
    t_us = float(dec["time_us"].sum()) * num_layers
    cmp_us = float(dec["ideal_us"].sum()) * num_layers
    mem_us = float((dec["read_us"] + dec["write_us"]).sum()) * num_layers
    if t_us <= 0:
        return None

    def _tps(us: float) -> float:
        return gbs * 1e6 / us if us > 0 else 0.0

    decode_peak = _tps(t_us)

    # Consistency gate: the Verbose-derived peak must agree with the L1 avg_lat
    # ceiling; a divergence means wrong block/scaling -> fall back to native L2.
    l1 = maidas_breakdown_from_excel(path, runtime)
    if l1 is not None and l1.peak_tok_per_sec > 0:
        rel = abs(decode_peak - l1.peak_tok_per_sec) / l1.peak_tok_per_sec
        if rel > 0.10:
            logger.warning(
                "MAIDAS L2 decode=%.1f tok/s diverges %.0f%% from L1 ceiling=%.1f "
                "(soc=%s gbs=%s osl=%s) -> using native PerfModel",
                decode_peak, rel * 100, l1.peak_tok_per_sec, soc, gbs, runtime.osl,
            )
            return None

    block_t = float(dec["time_us"].sum()) or 1.0
    ops: list[OpBreakdown] = []
    for _, r in dec.iterrows():
        rmem = float(r["read_us"] + r["write_us"]) * num_layers
        flops = float(r["tflops"]) * 1e12 * (float(r["time_us"]) * 1e-6) * num_layers
        bytes_moved = rmem * 1e-6 * bw_bps
        ops.append(OpBreakdown(
            name=str(r["layer"]),
            flops=flops,
            bytes_moved=bytes_moved,
            ai=(flops / bytes_moved if bytes_moved else 0.0),
            time_s=float(r["time_us"]) * num_layers / 1e6,
            bound=("compute" if float(r["ideal_us"]) >= float(r["read_us"] + r["write_us"])
                   else "memory"),
            pct_time=float(r["time_us"]) / block_t,
        ))

    pre = _select_layer_block(vd, soc, "prefill", runtime.isl or 0, gbs)
    prefill_peak = 0.0
    if pre is not None and not pre.empty:
        pt = float(pre["time_us"].sum()) * num_layers
        prefill_peak = int(runtime.isl or 0) * gbs * 1e6 / pt if pt > 0 else 0.0

    logger.info(
        "MAIDAS L2 PerfModel USED | soc=%s gbs=%s osl=%s | decode peak=%.2f "
        "mem=%.2f cmp=%.2f tok/s | %d ops",
        soc, gbs, runtime.osl, decode_peak, _tps(mem_us), _tps(cmp_us), len(ops),
    )
    return PerfModelBreakdown(
        decode_tok_per_s=decode_peak,
        prefill_tok_per_s=prefill_peak,
        decode_mem_tok_per_s=_tps(mem_us),
        decode_cmp_tok_per_s=_tps(cmp_us),
        ops=ops,
        bound_kind="memory" if mem_us > cmp_us else "compute",
        hbm_bw_gbps=bw_gbps,
        peak_achievable_tflops=peak_tflops,
    )
