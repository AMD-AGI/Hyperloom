"""Shared CSV interface for analytical roofline data (leaf module, no cycles).

This module is the single source of truth for the three roofline CSV schemas and
the only place that reads/writes them. Producers write analytical roofline here;
consumers read it here. When an external program authors the CSVs, Hyperloom
consumes them through this same reader with zero knowledge that an external program produced them.

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

#: model_roofline_meta.csv — one row of model-level analytical memory sizes + geometry
#: (mirrors ModelMeta). MAIDAS supplies weight/kv/activation bytes directly; HL otherwise
#: recomputes them from the HF model dir on every ceiling build.
MODEL_META_COLUMNS: list[str] = [
    "model_id",  # store identity (string) — not a ModelMeta field; used to validate external-store reuse
    "weight_bytes",
    "num_layers",
    "num_kv_heads",
    "head_dim",
    "weight_dtype_bytes",
    "active_weight_bytes",
    "num_experts",
    "experts_per_tok",
    "expert_weight_bytes",
    "expert_weight_dtype_bytes",
    "hidden_size",
    "intermediate_size",
    "moe_intermediate_size",
    "moe_hidden_size",
    "vocab_size",
    "num_attention_heads",
    "moe_layers",
    "dense_ffn_layers",
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
        # model_roofline_meta (all numeric)
        "weight_bytes",
        "num_layers",
        "num_kv_heads",
        "head_dim",
        "weight_dtype_bytes",
        "active_weight_bytes",
        "num_experts",
        "experts_per_tok",
        "expert_weight_bytes",
        "expert_weight_dtype_bytes",
        "hidden_size",
        "intermediate_size",
        "moe_intermediate_size",
        "moe_hidden_size",
        "vocab_size",
        "num_attention_heads",
        "moe_layers",
        "dense_ffn_layers",
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
# Hyperloom and one written by an external program join identically. The rules
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


def kernel_row_from_view(view: dict) -> dict | None:
    """Project one ``kernel_roofline.json`` per-kernel view into a CSV row, or ``None``.

    Keeps only ANALYTICAL + identity columns; MEASURED (``duration_us``/``gpu_pct``/
    ``call_count``/``rocprof_roofline``) and DERIVED (``efficiency_percent``/
    ``*_utilization_pct``) fields stay out of the CSV per the interface contract. The
    analytical magnitude columns ``flops``/``bytes_moved``/``ideal_us``/``compute_us``/
    ``read_us``/``write_us``/``peak_tflops``/``hbm_bw_gbps``/``precision`` are projected when
    the view supplies them (the native bypass estimator fills flops/bytes/compute_us/ideal_us/
    peaks; read_us/write_us come from an external author / MAIDAS). Blank when absent → the
    reader coerces to ``None`` and consumers fall back. Returns ``None`` for a nameless row.
    """
    name = view.get("name")
    if not name:
        return None
    row: dict = {
        "name": name,
        "kernel_id": view.get("kernel_id"),
        "source_file": view.get("source_file"),
        "reusable_native_kernel": view.get("reusable_native_kernel"),
        "kernel_category": view.get("kernel_category"),
        "precision": view.get("precision"),
        "flops": view.get("flops"),
        "bytes_moved": view.get("bytes_moved"),
        "arithmetic_intensity": view.get("arithmetic_intensity"),
        "flops_per_byte": view.get("flops_per_byte"),
        "bound_type": canon_bound(view.get("bound_type") or view.get("bottleneck")),
        "ideal_us": view.get("ideal_us"),
        "compute_us": view.get("compute_us"),
        "read_us": view.get("read_us"),
        "write_us": view.get("write_us"),
        "peak_tflops": view.get("peak_tflops"),
        "hbm_bw_gbps": view.get("hbm_bw_gbps"),
        "roofline_source": view.get("roofline_source"),
    }
    return row


def write_arch_peaks(rows: Iterable[dict], path: Path | str) -> None:
    """Write ``gpu_arch_peaks.csv`` (one row per device), overwriting."""
    _atomic_write_rows(Path(path), ARCH_PEAKS_COLUMNS, rows)


#: TraceLens arch-spec ``max_achievable_tflops`` key -> ``gpu_arch_peaks.csv`` column.
#: ``matrix_fp4`` (TraceLens) is the same peak this schema names ``matrix_mx4_tflops``.
_ARCH_SPEC_MATRIX_TO_COLUMN: dict[str, str] = {
    "matrix_bf16": "matrix_bf16_tflops",
    "matrix_fp16": "matrix_fp16_tflops",
    "matrix_fp32": "matrix_fp32_tflops",
    "matrix_fp8": "matrix_fp8_tflops",
    "matrix_fp4": "matrix_mx4_tflops",
    "matrix_mx4": "matrix_mx4_tflops",
    "matrix_int8": "matrix_int8_tflops",
    "vector_fp32": "vector_fp32_tflops",
}


def arch_row_from_spec(spec: dict) -> dict | None:
    """Project a TraceLens arch spec into one ``gpu_arch_peaks.csv`` row, or ``None``.

    The spec is the JSON ``populate_gpu_arch_json`` resolves (measured microbench or the
    native achievable table): ``{name, mem_bw_gbps, max_achievable_tflops{matrix_*}}``.
    Returns ``None`` when it carries no device name (nothing to key a row on).
    """
    name = spec.get("name")
    if not name:
        return None
    row: dict = {"name": str(name), "mem_bw_gbps": spec.get("mem_bw_gbps")}
    maf = spec.get("max_achievable_tflops")
    if isinstance(maf, dict):
        for matrix_key, column in _ARCH_SPEC_MATRIX_TO_COLUMN.items():
            val = maf.get(matrix_key)
            if isinstance(val, (int, float)) and val > 0:
                row[column] = float(val)
    return row


def _ceiling_row_key(row: dict) -> tuple:
    """Upsert key for a ceiling row: ``('arm', arm)`` or ``('op', arm, op_name)``.

    ``arm`` holds the row's config key, so op rows are scoped by config — per-config op
    breakdowns of different arms never clobber one another.
    """
    arm = str(row.get("arm") or "")
    rtype = str(row.get("row_type") or "arm").strip().lower()
    if rtype == "op":
        return ("op", arm, str(row.get("op_name") or ""))
    return ("arm", arm)


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
    """Read ``roofline_ceiling.csv`` -> ``{arm: row}`` plus ``{('op', arm, op_name): row}``.

    ``arm`` is the row's config key; op rows are scoped by it. ``bound_kind`` is normalized
    to ``{compute, memory}``. Missing file -> ``{}``.
    """
    out: dict = {}
    for row in _read_rows(path):
        row["bound_kind"] = canon_bound(row.get("bound_kind"))
        arm = row.get("arm")
        rtype = str(row.get("row_type") or "arm").strip().lower()
        if rtype == "op":
            out[("op", str(arm or ""), str(row.get("op_name") or ""))] = row
        elif arm:
            out[str(arm)] = row
    return out


def read_arch_peaks(path: Path | str) -> dict[str, dict]:
    """Read ``gpu_arch_peaks.csv`` -> ``{device_name: row}``. Missing -> ``{}``.

    Keyed by the verbatim ``name`` cell (e.g. ``MI355X``). Consumers that key off a
    lower-cased ``gpu_type`` should match case-insensitively — :class:`RooflineResolver`
    does; the arch producer writes ``normalize_platform`` (upper) names.
    """
    out: dict[str, dict] = {}
    for row in _read_rows(path):
        name = row.get("name")
        if name:
            out[str(name)] = row
    return out


def write_model_meta(row: dict, path: Path | str) -> None:
    """Write ``model_roofline_meta.csv`` (one model-level row), overwriting."""
    _atomic_write_rows(Path(path), MODEL_META_COLUMNS, [row])


def read_model_meta(path: Path | str) -> dict | None:
    """Read the single ``model_roofline_meta.csv`` row (numerics coerced) -> dict, or ``None``.

    ``None`` when the file is missing/empty so the caller falls back to the HF-dir computation.
    """
    rows = _read_rows(path)
    return rows[0] if rows else None


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


# --------------------------------------------------------------------------- #
# Ceiling content key — the identity of a roofline_ceiling.csv row.
# --------------------------------------------------------------------------- #
#
# Each ceiling row is keyed by the CONFIG that determines its analytical values, so
# distinct configs never overwrite each other across optimizer rounds and a consumer
# reads exactly its own config's row. Hyperloom (native) and an external MAIDAS bridge
# both build the key here, so one config yields one key everywhere. Design + the full
# determinant disposition: maidasToCsv/ceiling_key_mapping.md §6-§7.

CEILING_KEY_VERSION = "v1"

#: Ordered determinant fields per framework. model/gpu are per-run constants validated
#: separately (not keyed). Native ignores kv/ep/tep/pp/dcp/pcp — harmless over-keys.
LLM_CEILING_KEY_FIELDS: tuple[str, ...] = (
    "prec", "act", "kv", "tp", "ep", "tep", "pp", "dcp", "pcp", "conc", "isl", "osl",
)
XDIT_CEILING_KEY_FIELDS: tuple[str, ...] = (
    "prec", "act", "tp", "pp", "num_steps", "height", "width",
)

#: Fields normalized as dtype/precision tokens; the rest render as ints.
_CEILING_DTYPE_FIELDS: frozenset[str] = frozenset({"prec", "act", "kv"})


def _norm_ceiling_field(field: str, value: Any) -> str:
    """Normalize one ceiling-key field: unset -> ``na``; dtype fields via ``_DTYPE_SYNONYMS``
    (quantization-scheme tokens like ``mxfp4_fp8`` pass through unchanged); else a plain int."""
    if value is None or value == "":
        return "na"
    if field in _CEILING_DTYPE_FIELDS:
        tok = str(value).strip().lower()
        return _DTYPE_SYNONYMS.get(tok, tok)
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value).strip().lower()


def build_ceiling_key(config: dict, op_name: str | None = None) -> str:
    """Content key for a ceiling row from its ``config`` (framework-aware).

    ``config`` carries ``fw`` plus the determinant fields (:data:`LLM_CEILING_KEY_FIELDS`
    for LLM frameworks, :data:`XDIT_CEILING_KEY_FIELDS` for xDiT). Returns
    ``v1|fw=<f>|<field>=<norm>|...``; an op row appends ``|op=<op_name>``. Every field is
    always emitted (explicit ``na`` default) so both producers agree byte-for-byte.
    """
    # Key on the WORKLOAD CLASS, not the specific engine: the analytical ceiling is engine-agnostic
    # within LLM (vllm/sglang compute the same roofline), while diffusion (xdit) uses a wholly different
    # formula. So an external author need not know the exact engine to produce a matching key.
    fw = "xdit" if str(config.get("fw") or "").strip().lower() == "xdit" else "llm"
    fields = XDIT_CEILING_KEY_FIELDS if fw == "xdit" else LLM_CEILING_KEY_FIELDS
    parts = [CEILING_KEY_VERSION, f"fw={fw}"]
    parts.extend(f"{name}={_norm_ceiling_field(name, config.get(name))}" for name in fields)
    key = "|".join(parts)
    if op_name is not None:
        key = f"{key}|op={str(op_name).strip()}"
    return key


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
        self._model_meta: dict | None = None
        self._model_meta_loaded = False

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

    def model_meta(self) -> dict | None:
        """The single model-level memory-size/geometry row, or ``None`` when absent."""
        if self._dir is None:
            return None
        if not self._model_meta_loaded:
            self._model_meta = read_model_meta(self._p("model_roofline_meta.csv"))
            self._model_meta_loaded = True
        return self._model_meta

    def _arch_row(self, device: str) -> dict | None:
        """Case-insensitive device lookup: the CSV holds upper-case ``normalize_platform``
        names while consumers key off a lower-cased ``gpu_type``."""
        if self._dir is None:
            return None
        if self._arch is None:
            raw = read_arch_peaks(self._p("gpu_arch_peaks.csv"))
            self._arch = {str(k).strip().lower(): v for k, v in raw.items()}
        return self._arch.get((device or "").strip().lower())

    def arch_peak(self, device: str, dtype: str) -> float | None:
        """Matrix peak (TFLOP/s) for ``device``/``dtype``, or ``None`` if blank/missing."""
        row = self._arch_row(device)
        if not row:
            return None
        suffix = _DTYPE_SYNONYMS.get((dtype or "").strip().lower())
        if suffix is None:
            return None
        return row.get(f"matrix_{suffix}_tflops")

    def mem_bw(self, device: str) -> float | None:
        row = self._arch_row(device)
        return row.get("mem_bw_gbps") if row else None
