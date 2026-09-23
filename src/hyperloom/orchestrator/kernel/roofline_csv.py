"""Shared CSV interface for analytical roofline data (leaf module, no cycles).

This module is the single source of truth for the three roofline CSV schemas and
the only place that reads/writes them. Producers write analytical roofline here;
consumers read it here. When an external MAIDAS program authors the CSVs, Hyperloom
consumes them through this same reader with zero knowledge that MAIDAS produced them.

Design invariants (see CSV_INTERFACE_REFACTOR_PLAN.md §4):
  * Leaf module: imports only the stdlib, so producers/consumers can depend on it
    without creating an import cycle.
  * Fail-soft: a missing/malformed file yields an empty mapping, never an exception
    into the optimizer loop.
  * Numeric coercion on read: NUMERIC_COLUMNS are parsed str->float; a blank cell is
    ``None`` (not ``0.0``) so "ideal_us blank" != "ideal_us is 0". PEAK columns get a
    path-specific policy at the arithmetic sites (0.0 for the per-kernel recompute;
    absent-key for the arch peak-swap) — the reader itself keeps blanks as ``None``.
  * Normalization at the boundary: ``bound_type`` is canonicalized to {compute, memory}
    so no downstream code sees producer-specific spellings.
  * Atomic writes: temp file + ``os.replace`` (crash-safe).
  * The ceiling writer UPSERTS by key (arm / (op, op_name)) so multiple producers can
    each contribute a subset of rows without clobbering the file.
"""

from __future__ import annotations

import csv
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Schemas — the ONLY place column lists live.
# --------------------------------------------------------------------------- #

KERNEL_ROOFLINE_COLUMNS: list[str] = [
    "name",
    "kernel_id",
    "source_file",
    "reusable_native_kernel",
    "kernel_category",
    "precision",
    "flops",
    "bytes_moved",
    "arithmetic_intensity",
    "flops_per_byte",
    "bound_type",
    "ideal_us",
    "compute_us",
    "read_us",
    "write_us",
    "peak_tflops",
    "hbm_bw_gbps",
    "roofline_source",
    "tier",
]

CEILING_COLUMNS: list[str] = [
    "row_type",  # "arm" | "op"
    "arm",
    "op_name",
    "mem_tok_per_sec",
    "cmp_tok_per_sec",
    "peak_tok_per_sec",
    "prefill_tok_per_sec",
    "bound_kind",
    "theoretical_peak_tok_per_sec",
    "hbm_bw_gbps",
    "peak_achievable_tflops",
    "roofline_ideal_ms",
    "compute_peak_convention",
    "compute_peak_source",
    "flops",
    "bytes_moved",
    "ai",
    "time_s",
    "pct_time",
]

ARCH_PEAKS_COLUMNS: list[str] = [
    "name",
    "mem_bw_gbps",
    "matrix_bf16_tflops",
    "matrix_fp16_tflops",
    "matrix_fp32_tflops",
    "matrix_fp8_tflops",
    "matrix_mx4_tflops",
    "matrix_int8_tflops",
    "vector_fp32_tflops",
]

# Columns coerced str->float on read (blank -> None).
NUMERIC_COLUMNS: frozenset[str] = frozenset(
    {
        # kernel_roofline
        "flops",
        "bytes_moved",
        "arithmetic_intensity",
        "flops_per_byte",
        "ideal_us",
        "compute_us",
        "read_us",
        "write_us",
        "peak_tflops",
        "hbm_bw_gbps",
        # roofline_ceiling
        "mem_tok_per_sec",
        "cmp_tok_per_sec",
        "peak_tok_per_sec",
        "prefill_tok_per_sec",
        "theoretical_peak_tok_per_sec",
        "peak_achievable_tflops",
        "roofline_ideal_ms",
        "ai",
        "time_s",
        "pct_time",
        # gpu_arch_peaks
        "mem_bw_gbps",
        "matrix_bf16_tflops",
        "matrix_fp16_tflops",
        "matrix_fp32_tflops",
        "matrix_fp8_tflops",
        "matrix_mx4_tflops",
        "matrix_int8_tflops",
        "vector_fp32_tflops",
    }
)

# bound_type / bound_kind normalization -> canonical {compute, memory}.
BOUND_CANON: dict[str, str] = {
    "compute_bound": "compute",
    "memory_bound": "memory",
    "compute": "compute",
    "memory": "memory",
}


def canon_bound(value: Any) -> str | None:
    """Canonicalize a bound spelling to ``compute``/``memory`` (or ``None``)."""
    if value is None:
        return None
    return BOUND_CANON.get(str(value).strip().lower(), str(value).strip().lower() or None)


# --------------------------------------------------------------------------- #
# Canonical join key.
# --------------------------------------------------------------------------- #
#
# The default key mirrors tracelens_analysis.roofline_match_key so a CSV written by
# Hyperloom and one written by an external MAIDAS program join identically. The rules
# are inlined here (rather than imported) to keep this a dependency-free leaf module;
# tests/test_roofline_csv.py cross-checks it against the authoritative function.


def _match_key(name: str) -> str:
    """Default shape-agnostic key — mirror of ``roofline_match_key``."""
    lower = (name or "").lower()
    if "cijk_" in lower:
        return "hipblaslt_gemm"
    if "gemm_a16w16_asm" in lower or "a16w16" in lower:
        return "aiter_asm_gemm"
    if "attn_fwd" in lower or "flash_attn" in lower:
        return "attention"
    if "moe_ck2stages" in lower or "moe_ck_tile" in lower:
        return "moe_gemm"
    if "vectorized_layer_norm" in lower or "rms_norm" in lower:
        return "rms_norm"
    if "topk" in lower:
        return "topk"
    if "rope" in lower or "rotary" in lower:
        return "rope"
    if "nccl" in lower or "allreduce" in lower:
        return "allreduce"
    if "copy" in lower or "memcpy" in lower:
        return "memcpy"
    if "softmax" in lower:
        return "softmax"
    if "skinny" in lower:
        return "skinny_gemm"
    return lower[:80]


def canonical_key(
    name: str,
    *,
    category: str | None = None,
    mnk: str | None = None,
    precision: str | None = None,
) -> str:
    """Join key for a kernel/op row.

    Default (native parity): the shape-agnostic ``_match_key(name)``. When a
    shape-aware triple is supplied (``category`` + ``mnk`` + ``precision``), a
    distinct key ``category|MxNxK|precision`` is returned so N distinct-shape GEMMs
    do not collapse to one bucket (opt-in, behind the shape-aware key decision §3.4).
    """
    if category and mnk and precision:
        return f"{str(category).strip().lower()}|{str(mnk).strip()}|{str(precision).strip().lower()}"
    return _match_key(name)


# --------------------------------------------------------------------------- #
# Coercion helpers.
# --------------------------------------------------------------------------- #


def _coerce_cell(column: str, raw: str) -> Any:
    """Coerce one CSV cell: NUMERIC_COLUMNS str->float (blank -> None); else str."""
    if column in NUMERIC_COLUMNS:
        text = (raw or "").strip()
        if text == "":
            return None
        try:
            return float(text)
        except (TypeError, ValueError):
            return None
    return raw


def _row_out(column: str, value: Any) -> str:
    """Render one value for CSV write: None/'' -> '' ; bool -> lower ; else str."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


# --------------------------------------------------------------------------- #
# Writers.
# --------------------------------------------------------------------------- #


def _atomic_write_rows(path: Path, columns: list[str], rows: Iterable[dict]) -> None:
    """Write ``rows`` (dicts) as CSV with ``columns`` header, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            for row in rows:
                writer.writerow([_row_out(col, row.get(col)) for col in columns])
        os.replace(tmp_name, path)
    except (OSError, csv.Error):
        # Fail-soft: a write failure must not raise into the optimizer loop.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        log.warning("roofline_csv: failed to write %s", path, exc_info=True)


def write_kernel_roofline(rows: Iterable[dict], path: Path | str) -> None:
    """Write ``kernel_roofline.csv`` (analytical columns only), overwriting."""
    _atomic_write_rows(Path(path), KERNEL_ROOFLINE_COLUMNS, rows)


def write_arch_peaks(rows: Iterable[dict], path: Path | str) -> None:
    """Write ``gpu_arch_peaks.csv`` (one row per device), overwriting."""
    _atomic_write_rows(Path(path), ARCH_PEAKS_COLUMNS, rows)


def _ceiling_row_key(row: dict) -> tuple:
    """Upsert key for a ceiling row: ('arm', arm) or ('op', op_name)."""
    rtype = str(row.get("row_type") or "arm").strip().lower()
    if rtype == "op":
        return ("op", str(row.get("op_name") or ""))
    return ("arm", str(row.get("arm") or ""))


def write_ceiling(rows: Iterable[dict], path: Path | str) -> None:
    """UPSERT ceiling rows into ``roofline_ceiling.csv`` by key.

    ~7 producers each contribute a subset of arms/op rows at different times; a plain
    overwrite would clobber prior rows. This reads the existing file, merges the new
    rows by ``_ceiling_row_key`` (last write wins per key), and atomically rewrites.
    """
    path = Path(path)
    merged: dict[tuple, dict] = {}
    for existing in _read_rows(path):
        merged[_ceiling_row_key(existing)] = existing
    for row in rows:
        merged[_ceiling_row_key(row)] = dict(row)
    _atomic_write_rows(path, CEILING_COLUMNS, merged.values())


# --------------------------------------------------------------------------- #
# Readers.
# --------------------------------------------------------------------------- #


def _read_rows(path: Path | str) -> list[dict]:
    """Read a CSV into a list of coerced dicts; missing/malformed -> []."""
    path = Path(path)
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        with path.open("r", newline="", encoding="utf-8") as fh:
            # _coerce_cell absorbs a bad numeric cell (-> None), so a row never raises;
            # only a file/parse error can, and it's caught below.
            for raw in csv.DictReader(fh):
                out.append({col: _coerce_cell(col, val) for col, val in raw.items()})
    except (OSError, csv.Error):
        log.warning("roofline_csv: failed to read %s", path, exc_info=True)
        return []
    return out


def read_kernel_roofline(path: Path | str) -> dict[str, dict]:
    """Read ``kernel_roofline.csv`` -> ``{canonical_key(name): row}``.

    Numeric columns are coerced (blank -> None); ``bound_type`` is normalized to
    ``{compute, memory}``. Missing/malformed file -> ``{}``.
    """
    out: dict[str, dict] = {}
    for row in _read_rows(path):
        name = row.get("name")
        if not name:
            continue
        row["bound_type"] = canon_bound(row.get("bound_type"))
        out[canonical_key(str(name))] = row
    return out


def read_ceiling(path: Path | str) -> dict:
    """Read ``roofline_ceiling.csv`` -> ``{arm: row}`` plus ``{('op', op_name): row}``.

    ``bound_kind`` is normalized to ``{compute, memory}``. Missing file -> ``{}``.
    """
    out: dict = {}
    for row in _read_rows(path):
        row["bound_kind"] = canon_bound(row.get("bound_kind"))
        rtype = str(row.get("row_type") or "arm").strip().lower()
        if rtype == "op":
            out[("op", str(row.get("op_name") or ""))] = row
        else:
            arm = row.get("arm")
            if arm:
                out[str(arm)] = row
    return out


def read_arch_peaks(path: Path | str) -> dict[str, dict]:
    """Read ``gpu_arch_peaks.csv`` -> ``{device_name: row}``. Missing -> ``{}``."""
    out: dict[str, dict] = {}
    for row in _read_rows(path):
        name = row.get("name")
        if name:
            out[str(name)] = row
    return out


# --------------------------------------------------------------------------- #
# Resolver facade — consumers ask this, never touching CSV mechanics.
# --------------------------------------------------------------------------- #

# dtype-synonym -> canonical arch-peak column suffix (§6.3b).
_DTYPE_SYNONYMS: dict[str, str] = {
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp16": "fp16",
    "f16": "fp16",
    "float16": "fp16",
    "fp32": "fp32",
    "f32": "fp32",
    "float32": "fp32",
    "fp8": "fp8",
    "f8": "fp8",
    "float8_e4m3fn": "fp8",
    "float8_e5m2": "fp8",
    "mx4": "mx4",
    "mxfp4": "mx4",
    "fp4": "mx4",
    "f4": "mx4",
    "float4": "mx4",
    "int8": "int8",
    "i8": "int8",
}


class RooflineResolver:
    """Read-through facade over the three CSVs for consumers.

    ``kernel(name)``, ``arch_peak(device, dtype)``, ``ceiling(arm)``. All lookups are
    fail-soft (missing -> None). ``arch_peak`` returns the matrix peak for the dtype
    or ``None`` when the cell is blank/missing — the caller decides whether ``None``
    means "skip metric" (per-kernel recompute) or "fall through to bf16" (peak-swap).
    """

    def __init__(self, csv_dir: Path | str | None):
        self._dir = Path(csv_dir) if csv_dir else None
        self._kernels: dict[str, dict] | None = None
        self._ceiling: dict | None = None
        self._arch: dict[str, dict] | None = None

    def _p(self, name: str) -> Path | None:
        return (self._dir / name) if self._dir else None

    def kernel(self, name: str) -> dict | None:
        if self._dir is None:
            return None
        if self._kernels is None:
            self._kernels = read_kernel_roofline(self._p("kernel_roofline.csv"))
        return self._kernels.get(canonical_key(str(name)))

    def ceiling(self, arm: str) -> dict | None:
        if self._dir is None:
            return None
        if self._ceiling is None:
            self._ceiling = read_ceiling(self._p("roofline_ceiling.csv"))
        return self._ceiling.get(str(arm))

    def arch_peak(self, device: str, dtype: str) -> float | None:
        """Matrix peak (TFLOP/s) for ``device``/``dtype``, or ``None`` if blank/missing."""
        if self._dir is None:
            return None
        if self._arch is None:
            self._arch = read_arch_peaks(self._p("gpu_arch_peaks.csv"))
        row = self._arch.get(str(device))
        if not row:
            return None
        suffix = _DTYPE_SYNONYMS.get((dtype or "").strip().lower())
        if suffix is None:
            return None
        return row.get(f"matrix_{suffix}_tflops")

    def mem_bw(self, device: str) -> float | None:
        if self._dir is None:
            return None
        if self._arch is None:
            self._arch = read_arch_peaks(self._p("gpu_arch_peaks.csv"))
        row = self._arch.get(str(device))
        return row.get("mem_bw_gbps") if row else None
