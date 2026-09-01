# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Common logic for aiter dense GEMM tuners (a8w8, blockscale, bpreshuffle, a4w4)."""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from .base import TuneContext, TuneResult
from ..script_probe import filter_args, probe_script
from ..utils import find_tuner_script, resolve_aiter_root, run_subprocess
from .. import tune_robustness as _tr

log = logging.getLogger(__name__)

# The only op whose production dispatch the per-shape split-K trial validates:
# aiter_splitk_validate hardcodes gemm_a8w8_blockscale_ck, so the trial is correct
# only for this script_key. Shared here as the single source of truth so the gate
# in run_aiter_dense_tuner and the A8W8BlockscaleTuner caller cannot drift apart.
SPLITK_TRIAL_SCRIPT_KEY = "a8w8_blockscale"


class AiterDtypeUnavailable(RuntimeError):
    """The installed aiter cannot supply a dtype the tuner CSV needs."""


def _aiter_dtype_str(attr: str) -> str:
    """Return the repr string aiter's tuner scripts accept for ``dtypes.<attr>``.

    The tuner scripts translate the CSV value through aiter's own
    ``dtype2str_dict``, and the torch dtype backing each alias is
    architecture-specific (``dtypes.fp8`` is ``torch.float8_e4m3fn`` on CDNA4 /
    gfx950 but ``torch.float8_e4m3fnuz`` on CDNA3 / gfx942). Resolving from the
    installed aiter -- and checking the result really is a key of that table --
    is the only way to emit a value this build accepts.

    Raises:
        AiterDtypeUnavailable: aiter is not importable, lacks the alias, or maps
            it to a dtype outside its own translation table. Failing here is
            deliberate: a guessed constant would be written into the CSV and only
            surface far later as a ``KeyError`` inside aiter, after the tuner has
            already produced zero tuned shapes.
    """
    try:
        from aiter import dtype2str_dict, dtypes  # type: ignore[import-untyped]
    except Exception as exc:  # noqa: BLE001 - any import failure is fatal here
        raise AiterDtypeUnavailable(
            f"cannot resolve the aiter dtype for {attr!r}: aiter is not importable ({exc})"
        ) from exc
    dtype = getattr(dtypes, attr, None)
    if dtype is None:
        raise AiterDtypeUnavailable(f"the installed aiter has no dtypes.{attr}")
    if dtype not in dtype2str_dict:
        raise AiterDtypeUnavailable(
            f"aiter maps dtypes.{attr} to {dtype!r}, which is absent from its own "
            "dtype2str_dict; the tuner would fail on this value"
        )
    return repr(dtype)


def _aiter_fp8_dtype_str() -> str:
    """Resolve the FP8 dtype string for this aiter build."""
    return _aiter_dtype_str("fp8")


def _safe_is_file(path: Path | None) -> bool:
    """``Path.is_file()`` guarded against ``OSError(ENAMETOOLONG)``.

    ``ctx.shapes_json`` / ``ctx.untuned_csv`` may be a ``Path`` built from inline
    JSON content rather than a real path; ``is_file()`` then raises
    ``OSError(36)`` and aborts the tuner. Treat any OSError as "not a file".
    """
    if path is None:
        return False
    try:
        return path.is_file()
    except OSError:
        return False


def _profile_has_derivable_shapes(ctx: TuneContext) -> bool:
    """True when the model config carries enough dims to derive dense shapes."""
    profile = getattr(ctx, "profile", None)
    if profile is None:
        return False
    return int(getattr(profile, "hidden_size", 0) or 0) >= 1 and int(getattr(profile, "intermediate_size", 0) or 0) >= 1


def validate_dense_tuner_inputs(ctx: TuneContext, script_key: str, *, script_label: str) -> str | None:
    """Shared validate() for the aiter dense fp8/fp4 tuners.

    A tuner can run when it has a real CSV, a shapes JSON, OR a model config it
    can derive shapes from. Returns an error string when none of these hold (or
    the aiter script is missing), else None.
    """
    if find_tuner_script(script_key) is None:
        return f"aiter {script_label} tuner script not found"
    if (
        getattr(ctx, "shapes_manifest", None)
        or ctx.untuned_csv
        or ctx.shapes_json
        or getattr(ctx, "demand_json", None)
        or _profile_has_derivable_shapes(ctx)
    ):
        return None
    return (
        "Requires --shapes-manifest, --untuned-csv, --shapes-json, --demand, "
        "or a model config to derive dense GEMM shapes"
    )


# Mean measured cost of tuning one shape; used to size the shape list against
# the time budget rather than tuning a list we cannot finish. Thorough mode
# searches every backend and measured ~407s/shape on MI355X against ~32s for the
# hipblaslt-only default, so the two cannot share one figure: sized with the
# fast cost, a thorough run claims 5.5x the shapes it can finish and the
# remainder are written as nothing.
_DEMAND_PER_SHAPE_COST_S = 74
_DEMAND_PER_SHAPE_COST_THOROUGH_S = 420
_DEMAND_RESERVE_S = 120
_DEMAND_MAX_SHAPES_ENV = "FORGE_DEMAND_MAX_SHAPES"


# Fraction of output elements aiter's own accuracy check found wrong. Same
# threshold ``cap_unsupported_splitk`` already applies when it picks a
# replacement candidate; a row that never needed replacing used to keep whatever
# figure it had.
_MAX_ERR_RATIO = 0.01
_ERR_RATIO_COLUMNS = ("err_ratio", "errRatio")


def _row_err_ratio(row: dict[str, str]) -> float | None:
    """The accuracy figure aiter recorded for a row, or None if it recorded none."""
    for col in _ERR_RATIO_COLUMNS:
        if col in row:
            try:
                return float(row[col] or 0.0)
            except (TypeError, ValueError):
                return None
    return None


def drop_inaccurate_rows(tuned_csv: Path) -> list[dict[str, str]]:
    """Remove rows aiter measured as numerically wrong, in place.

    The tuner records the error it measured and then names the kernel that
    libtype's winner regardless. On MI355X across four bf16 shapes, every
    split-K row it selected carried a nonzero figure -- flydsl split_k=7 at
    0.0202, asm split_k=7 at 0.0203, asm split_k=4 at 0.0137 -- while every
    splitK=0 row was 0.0. Re-running those kernels confirms the recorded number:
    1.25-3.98% of elements are wrong, and *which* ones changes between identical
    calls, so the split-K reduction races rather than merely rounding
    differently.

    This has to happen before the artifact is handed on, because ``env_value``
    is that file: without a filter the fastest wrong answer wins. It also
    inverts the backend comparison it came from -- flydsl's 37% lead over
    hipblaslt at M=16 is the time saved by not computing 2% of the output.

    Returns the dropped rows. A shape left with no row falls back to aiter's
    default kernel at serve time, which is the right outcome: no tuned entry
    beats a tuned entry that computes the wrong answer.
    """
    try:
        if not tuned_csv.is_file():
            return []
        with tuned_csv.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if r]
    except (OSError, csv.Error) as exc:
        log.warning("accuracy filter could not read %s: %s", tuned_csv, exc)
        return []
    if not rows:
        return []
    if not any(c in rows[0] for c in _ERR_RATIO_COLUMNS):
        log.warning(
            "%s has no accuracy column; deploying without the numerical filter (aiter schema drift?)",
            tuned_csv,
        )
        return []

    keep: list[dict[str, str]] = []
    dropped: list[dict[str, str]] = []
    for row in rows:
        er = _row_err_ratio(row)
        (dropped if er is not None and er > _MAX_ERR_RATIO else keep).append(row)
    if not dropped:
        return []

    # Write beside the artifact and rename over it. Truncating the real file
    # first means a failure part-way (full disk, revoked permission) leaves a
    # half-written table that the caller would still be told is filtered.
    tmp = tuned_csv.with_name(tuned_csv.name + ".filtered.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(keep)
        os.replace(tmp, tuned_csv)
    except (OSError, csv.Error) as exc:
        log.error(
            "%d inaccurate row(s) could not be removed from %s (%s); the original "
            "artifact is untouched and is NOT filtered",
            len(dropped),
            tuned_csv,
            exc,
        )
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            log.warning("could not remove the partial file %s", tmp)
        return []

    for row in dropped:
        log.error(
            "dropping M=%s N=%s K=%s (%s, splitK=%s, %sus) from %s -- aiter "
            "measured err_ratio=%s, above the %.2f limit",
            row.get("M"),
            row.get("N"),
            row.get("K"),
            row.get("libtype"),
            row.get("splitK"),
            row.get("us"),
            tuned_csv.name,
            _row_err_ratio(row),
            _MAX_ERR_RATIO,
        )
    return dropped


def _demand_budget(ctx: TuneContext) -> int:
    raw = os.environ.get(_DEMAND_MAX_SHAPES_ENV, "").strip()
    try:
        override = int(raw)
    except ValueError:
        override = 0
    if override > 0:
        return override
    cost = _DEMAND_PER_SHAPE_COST_THOROUGH_S if getattr(ctx, "thorough", False) else _DEMAND_PER_SHAPE_COST_S
    usable = max(int(getattr(ctx, "timeout_s", 0)) - _DEMAND_RESERVE_S, cost)
    return max(1, usable // cost)


def _demand_input_csv(
    ctx: TuneContext,
    work_dir: Path,
    tuner_name: str,
    *,
    needs_q_dtype_w: bool = False,
) -> Path | None:
    """Untuned CSV built from the keys the runtime actually missed.

    Returns None when no demand file was supplied, or when it carries nothing
    for this tuner -- both mean "fall back to the existing shape sources", not
    "tune nothing".
    """
    path = getattr(ctx, "demand_json", None)
    if not path:
        return None
    from ..evidence import demand_for_tuner, demand_shapes, load_demand

    report = load_demand(path)
    if report is None:
        return None
    entry = demand_for_tuner(report, tuner_name)
    if entry is None:
        return None
    budget = _demand_budget(ctx)
    # The a8w8 blockscale, a8w8 quant-type, and a4w4 lookup paths all retry the
    # exact M followed by get_padded_m(..., gl=0) and gl=1, using the same
    # gemm_op_common implementation as a16w16. Spend the budget on those lookup
    # buckets so one row covers every observed M that resolves to it.
    shapes = demand_shapes(entry, limit=budget)
    if not shapes:
        return None

    out = work_dir / f"untuned_{tuner_name}_demand.csv"
    header = "M,N,K,q_dtype_w" if needs_q_dtype_w else "M,N,K"
    q_dtype_w = _aiter_dtype_str("fp8") if needs_q_dtype_w else ""
    with out.open("w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for s in shapes:
            row = f"{s['M']},{s['N']},{s['K']}"
            if needs_q_dtype_w:
                row += f",{q_dtype_w}"
            fh.write(row + "\n")
    log.info(
        "%s: %d padded-M demand shapes (of %d distinct keys, budget %d) -> %s",
        tuner_name,
        len(shapes),
        entry.get("distinct_keys", 0),
        budget,
        out,
    )
    return out


def _resolve_input_csv(ctx: TuneContext, work_dir: Path, needs_q_dtype_w: bool = False) -> Path | None:
    """Resolve the input untuned CSV for a dense tuner.

    Priority:
    0. Caller-supplied ``shapes_manifest`` (weighted, variant-discriminating
       TraceShapeManifest; the P0-A Trace->CSV path -- real replay-weighted
       shapes, highest-impact first). Preferred when explicitly supplied.
    1. Caller-supplied ``untuned_csv`` (real recorded GEMM shapes; most accurate).
    2. Caller-supplied ``shapes_json`` (converted to CSV).
    3. Shapes derived from the model config (so the tuner runs even when nothing
       was recorded upstream -- same approach the bf16 dense tuner already uses).

    Every recorded source gets a decode-band guarantee: a capture that only
    recorded a large prefill M (CUDA Graph hides the small decode GEMMs from the
    profiler) would otherwise tune the wrong operating point -- a micro win that
    regresses E2E because the throughput-dominant small-M decode GEMMs get the
    prefill-tuned tile. ``_ensure_decode_m_coverage`` appends the missing decode
    rows per dispatch group and leaves already-representative groups untouched,
    so tuning time stays bounded.

    Thorough mode additionally crosses the recorded NK pairs with the full
    config-derived M grid -- except for a manifest, whose whole point is a
    curated, weight-ordered shape set. Expanding that into a full grid would
    discard the curation, so a manifest only ever gets the decode guard.
    """
    csv: Path | None = None
    from_manifest = False
    if _safe_is_file(getattr(ctx, "shapes_manifest", None)):
        from ..shape_manifest import write_manifest_untuned_csv

        csv = write_manifest_untuned_csv(ctx.shapes_manifest, work_dir, needs_q_dtype_w=needs_q_dtype_w)
        from_manifest = csv is not None
        # Manifest yielded no tunable target shapes: fall through to the other
        # sources rather than failing outright.
    if csv is None:
        if _safe_is_file(ctx.untuned_csv):
            csv = _conform_csv_columns(ctx.untuned_csv, work_dir, needs_q_dtype_w=needs_q_dtype_w)
        elif _safe_is_file(ctx.shapes_json):
            csv = _shapes_json_to_csv(ctx.shapes_json, work_dir, needs_q_dtype_w=needs_q_dtype_w)
        else:
            return _derive_input_csv_from_config(ctx, work_dir, needs_q_dtype_w=needs_q_dtype_w)

    if csv is not None:
        if ctx.thorough and not from_manifest:
            csv = _augment_with_config_m_values(csv, ctx, work_dir, needs_q_dtype_w=needs_q_dtype_w)
        else:
            csv = _ensure_decode_m_coverage(csv, ctx, work_dir, needs_q_dtype_w=needs_q_dtype_w)
    return csv


def _padded_m_gl0(m: int) -> int:
    """aiter's ``get_padded_m(..., gl=0)``: round up to a tile multiple.

    The granularity widens three times, mirroring ``getPaddedM`` in
    ``csrc/py_itfs_cu/gemm_common.cu``: 16 up to and including 256, then 32
    through 1024, then 64 through 4096, then 128. So 1 -> 16, 17 -> 32,
    257 -> 288, 1025 -> 1088 and 4097 -> 4224.
    """
    m = max(1, int(m))
    if m <= 256:
        step = 16
    elif m <= 1024:
        step = 32
    elif m <= 4096:
        step = 64
    else:
        step = 128
    return -(-m // step) * step


def _next_pow2(m: int) -> int:
    """Round up to a power of two (aiter's ``nextPow2``)."""
    m = max(1, int(m))
    return 1 << (m - 1).bit_length()


def _padded_m_gl1(m: int, n: int) -> int:
    """aiter's ``get_padded_m(..., gl=1)``, which is not a plain power of two.

    Past M=8192 a wide N collapses the bucket to 8192 instead of growing it, so
    the coarse key cannot be derived from M alone -- reading it as ``nextPow2``
    puts a large-M row in a bucket the runtime never looks in.
    """
    if int(m) > 8192 and int(n) > 4096:
        return 8192
    return _next_pow2(m)


def _dispatch_lookup_ms(m: int, n: int) -> set[int]:
    """The tuned-M values aiter will accept when serving runtime batch ``m``.

    ``get_CKGEMM_config`` probes the tuned table three times -- the exact ``M``,
    then ``get_padded_m(M, N, K, gl)`` for ``gl`` 0 and 1 -- and takes the first
    hit. A tuned row therefore serves a runtime shape only when its ``M`` is one
    of these; a row at M=64 does not serve M=16, while a row at M=16 does serve
    M=1/2/4/8 because they all pad into the same bucket.

    ``N`` is required because the ``gl=1`` bucket depends on it (see
    :func:`_padded_m_gl1`).

    Mirrored locally rather than imported so shape resolution stays usable on a
    host without aiter; ``test_padded_m_mirror_matches_installed_aiter`` pins the
    mirror against the real implementation wherever aiter is present.
    """
    return {int(m), _padded_m_gl0(m), _padded_m_gl1(m, n)}


def _ensure_decode_m_coverage(
    csv: Path,
    ctx: TuneContext,
    work_dir: Path,
    needs_q_dtype_w: bool = False,
) -> Path:
    """Guarantee every tuned dispatch group covers the decode-band M (fast mode).

    Shape capture can record only a large prefill M (e.g. M=2095) because CUDA
    Graph wraps the small-M decode GEMMs and hides them from the profiler.
    Tuning only that M optimizes the wrong operating point: the micro benchmark
    wins on the prefill shape while the throughput-dominant small-M decode GEMMs
    regress (observed as a -18.45% E2E drop that then reverted).

    aiter looks a config up per ``(M, N, K)``, so coverage is decided **per
    dispatch group** -- ``(N, K)`` plus ``q_dtype_w`` when present -- and, within
    a group, **per lookup bucket**. Holding any one decode-grid M is not enough:
    a tuned M=64 row is never consulted for runtime M=16 or M=32, which probe
    their own exact/padded keys. For each decode M still unserved the missing
    ``get_padded_m(gl=0)`` bucket is added, which is the row aiter would dispatch
    to and covers every grid M that pads into it (16 serves 1/2/4/8).

    Every original row is preserved verbatim and in order -- manifest CSVs arrive
    sorted by GPU-time weight, and rows can carry a per-row ``q_dtype_w`` -- and
    only the missing bucket rows are appended, inheriting their group's dtype.
    """
    from ..dense_shapes import compute_decode_m_values

    try:
        lines = csv.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return csv
    if len(lines) < 2:
        return csv
    header = [h.strip() for h in lines[0].split(",")]
    idx = {h.upper(): i for i, h in enumerate(header)}
    if not {"M", "N", "K"}.issubset(idx):
        return csv
    q_idx = idx.get("Q_DTYPE_W")

    # Group key = the aiter dispatch key. Keep first-appearance order so the
    # appended rows follow the same priority as the input.
    group_order: list[tuple[int, int, str]] = []
    group_m: dict[tuple[int, int, str], set[int]] = {}
    body: list[str] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        body.append(line)
        parts = [p.strip() for p in line.split(",")]
        try:
            m = int(parts[idx["M"]])
            n = int(parts[idx["N"]])
            k = int(parts[idx["K"]])
        except (ValueError, IndexError):
            continue
        q = parts[q_idx] if q_idx is not None and q_idx < len(parts) else ""
        key = (n, k, q)
        if key not in group_m:
            group_m[key] = set()
            group_order.append(key)
        group_m[key].add(m)

    if not group_order:
        return csv

    decode_m = compute_decode_m_values(ctx.conc)
    additions: list[str] = []
    uncovered: list[tuple[int, int, str]] = []
    for key in group_order:
        n, k, q = key
        tuned_m = set(group_m[key])
        added_here = False
        for m in decode_m:
            if tuned_m & _dispatch_lookup_ms(m, n):
                continue  # some tuned row is already reachable from this M
            bucket = _padded_m_gl0(m)
            tuned_m.add(bucket)  # also serves the other grid M padding into it
            added_here = True
            row = [""] * len(header)
            row[idx["M"]], row[idx["N"]], row[idx["K"]] = str(bucket), str(n), str(k)
            if q_idx is not None:
                row[q_idx] = q
            additions.append(",".join(row))
        if added_here:
            uncovered.append(key)

    if not additions:
        return csv

    out = work_dir / "decode_covered_dense.csv"
    work_dir.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([lines[0], *body, *additions]) + "\n", encoding="utf-8")
    log.info(
        "Fast-mode decode coverage: %d of %d dispatch group(s) lacked a decode-band "
        "M (grid %s for conc=%s); appended %d row(s), original %d row(s) untouched",
        len(uncovered),
        len(group_order),
        decode_m,
        ctx.conc,
        len(additions),
        len(body),
    )
    return out


def _augment_with_config_m_values(
    csv: Path,
    ctx: TuneContext,
    work_dir: Path,
    needs_q_dtype_w: bool = False,
) -> Path:
    """Augment profile-derived shapes with config-derived M values.

    Profile/trace shapes often only capture a narrow M range (e.g. M≈ISL from
    single-request profiling) because CUDA Graph wraps high-concurrency GEMM
    calls, making them invisible to TraceLens. This function extracts the NK
    pairs from the profile CSV, then generates a complete shape set using
    config-derived M values (which include high-concurrency batch sizes like
    M=4096, 8192) crossed with those NK pairs.

    The result replaces the original CSV so tuning covers the full workload.
    """
    try:
        lines = csv.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return csv
    if len(lines) < 2:
        return csv

    header = [h.strip().upper() for h in lines[0].split(",")]
    idx = {h: i for i, h in enumerate(header)}
    if "N" not in idx or "K" not in idx:
        return csv

    profile_nk: set[tuple[int, int]] = set()
    profile_m: set[int] = set()
    m_idx = idx.get("M")
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        try:
            n, k = int(parts[idx["N"]]), int(parts[idx["K"]])
            if n > 0 and k > 0:
                profile_nk.add((n, k))
            if m_idx is not None and m_idx < len(parts):
                profile_m.add(int(parts[m_idx]))
        except (ValueError, IndexError):
            pass

    if not profile_nk:
        return csv

    isl = max(ctx.tokens) if ctx.tokens else 0
    from ..dense_shapes import compute_dense_m_values

    config_m = compute_dense_m_values(ctx.conc, thorough=ctx.thorough, isl=isl)
    all_m = sorted(set(config_m) | profile_m)

    if set(all_m) == profile_m:
        return csv

    q_dtype = ""
    if needs_q_dtype_w:
        q_dtype = _aiter_fp8_dtype_str()

    out = work_dir / "augmented_dense.csv"
    seen: set[tuple[int, int, int]] = set()
    with out.open("w", encoding="utf-8") as f:
        f.write("M,N,K,q_dtype_w\n" if needs_q_dtype_w else "M,N,K\n")
        for m in all_m:
            for n, k in sorted(profile_nk):
                if (m, n, k) not in seen:
                    seen.add((m, n, k))
                    if needs_q_dtype_w:
                        f.write(f"{m},{n},{k},{q_dtype}\n")
                    else:
                        f.write(f"{m},{n},{k}\n")

    log.info(
        "Augmented shapes: %d M values × %d NK pairs = %d shapes (profile had %d M values)",
        len(all_m),
        len(profile_nk),
        len(seen),
        len(profile_m),
    )
    return out


def _conform_csv_columns(
    src: Path,
    work_dir: Path,
    needs_q_dtype_w: bool,
    default_q_dtype: str = "",
) -> Path:
    """Return a CSV whose columns match what this tuner expects.

    blockscale / a4w4 expect ``M,N,K``; a8w8 / bpreshuffle expect an extra
    ``q_dtype_w`` column. If ``src`` already matches it is returned unchanged;
    otherwise a conformed copy is written to ``work_dir`` (adding ``q_dtype_w``
    with a default, or dropping extra columns). On any read error the original
    is returned so behavior never regresses below "pass the file through".
    """
    try:
        lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return src
    if not lines:
        return src
    header = [h.strip() for h in lines[0].split(",")]
    idx = {h.upper(): i for i, h in enumerate(header)}
    if not {"M", "N", "K"}.issubset(idx):
        return src  # unknown layout; pass through unchanged
    has_q = "Q_DTYPE_W" in idx
    if has_q == needs_q_dtype_w:
        return src  # already in the expected shape

    out = work_dir / f"conformed_{src.name}"
    with out.open("w", encoding="utf-8") as f:
        f.write("M,N,K,q_dtype_w\n" if needs_q_dtype_w else "M,N,K\n")
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if max(idx["M"], idx["N"], idx["K"]) >= len(parts):
                continue
            m, n, k = parts[idx["M"]], parts[idx["N"]], parts[idx["K"]]
            if needs_q_dtype_w:
                q = (
                    parts[idx["Q_DTYPE_W"]]
                    if has_q and idx["Q_DTYPE_W"] < len(parts)
                    else (default_q_dtype or _aiter_fp8_dtype_str())
                )
                f.write(f"{m},{n},{k},{q}\n")
            else:
                f.write(f"{m},{n},{k}\n")
    log.info("Conformed %s columns (needs_q_dtype_w=%s) -> %s", src.name, needs_q_dtype_w, out)
    return out


def _derive_input_csv_from_config(ctx: TuneContext, work_dir: Path, needs_q_dtype_w: bool = False) -> Path | None:
    """Synthesize an untuned CSV from the model config when none was supplied.

    Returns None when the profile lacks the dimensions needed to derive shapes.
    """
    from ..dense_shapes import (
        compute_dense_m_values,
        compute_dense_nk_shapes,
        write_mnk_untuned_csv,
    )

    profile = getattr(ctx, "profile", None)
    if profile is None:
        return None
    hidden_size = int(getattr(profile, "hidden_size", 0) or 0)
    intermediate_size = int(getattr(profile, "intermediate_size", 0) or 0)
    q_lora_rank = int(getattr(profile, "q_lora_rank", 0) or 0)
    kv_lora_rank = int(getattr(profile, "kv_lora_rank", 0) or 0)
    if hidden_size < 1:
        return None
    if intermediate_size < 1 and not (q_lora_rank and not kv_lora_rank):
        return None
    num_heads = int(getattr(profile, "num_attention_heads", 0) or 0)
    num_kv_heads = int(getattr(profile, "num_key_value_heads", 0) or num_heads or 0)
    nk_shapes = compute_dense_nk_shapes(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        tp=ctx.tp,
        head_dim=int(getattr(profile, "head_dim", 0) or 0),
        v_head_dim=int(getattr(profile, "v_head_dim", 0) or 0),
        q_lora_rank=int(getattr(profile, "q_lora_rank", 0) or 0),
        kv_lora_rank=int(getattr(profile, "kv_lora_rank", 0) or 0),
        qk_nope_head_dim=int(getattr(profile, "qk_nope_head_dim", 0) or 0),
        qk_rope_head_dim=int(getattr(profile, "qk_rope_head_dim", 0) or 0),
        o_lora_rank=int(getattr(profile, "o_lora_rank", 0) or 0),
        o_groups=int(getattr(profile, "o_groups", 0) or 0),
    )
    if not nk_shapes:
        return None
    isl = max(ctx.tokens) if ctx.tokens else 0
    m_values = compute_dense_m_values(ctx.conc, thorough=ctx.thorough, isl=isl)
    return write_mnk_untuned_csv(
        nk_shapes,
        m_values,
        work_dir,
        needs_q_dtype_w=needs_q_dtype_w,
    )


def _shapes_json_to_csv(shapes_json: Path, work_dir: Path, needs_q_dtype_w: bool = False) -> Path:
    """Convert a shapes JSON file to aiter's untuned CSV format.

    Expected JSON format: [{"M": int, "N": int, "K": int}, ...]
    or {"shapes": [{"M": int, "N": int, "K": int}, ...]}

    Output CSV format depends on tuner:
    - blockscale/a4w4: M,N,K
    - a8w8/bpreshuffle: M,N,K,q_dtype_w
    """
    data = json.loads(shapes_json.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        shapes = data.get("shapes", [])
    else:
        shapes = data

    csv_path = work_dir / "untuned_dense.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        if needs_q_dtype_w:
            f.write("M,N,K,q_dtype_w\n")
            for shape in shapes:
                m = shape.get("M", shape.get("m", 0))
                n = shape.get("N", shape.get("n", 0))
                k = shape.get("K", shape.get("k", 0))
                q_dtype = shape.get("q_dtype_w") or _aiter_fp8_dtype_str()
                f.write(f"{m},{n},{k},{q_dtype}\n")
        else:
            f.write("M,N,K\n")
            for shape in shapes:
                m = shape.get("M", shape.get("m", 0))
                n = shape.get("N", shape.get("n", 0))
                k = shape.get("K", shape.get("k", 0))
                f.write(f"{m},{n},{k}\n")

    log.info("Converted %d shapes from JSON to CSV at %s", len(shapes), csv_path)
    return csv_path


# Format A (older aiter): "... M=8192 ... N=5120 ... K=5120 ... default: X us tuned: Y us speedup: Zx"
_STDOUT_KV_RE = re.compile(
    r"M=(\d+).*?N=(\d+).*?K=(\d+).*?"
    r"default:\s*([\d.]+)\s*us.*?"
    r"tuned:\s*([\d.]+)\s*us.*?"
    r"speedup:\s*([\d.]+)x",
    re.IGNORECASE,
)

# Format B (current aiter --compare table): the "Would update" comparison block
#   "(8192, 5120, 5120)   |   1037.74 |   269.71 |   74.01% |   UPDATE"
#   "(8192, 5120, 5120)   |   N/A     |   269.71 |   N/A    |   NEW"   (new shape)
# columns: (M, N, K) | Pre(us) | Post(us) | Improve% | Action
# A shape with no prior tuned entry has no baseline to compare against, so aiter
# prints "N/A" for Pre and Improve% and marks the row NEW. Accept "N/A" in those
# two columns (else an all-new run parses to nothing and is misreported as
# no_improvement, skipping E2E validation of the freshly tuned configs).
_COMPARE_TABLE_RE = re.compile(
    r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*"
    r"\|\s*(N/A|[\d.]+)\s*"  # Pre (default) us -- "N/A" for a NEW shape
    r"\|\s*([\d.]+)\s*"  # Post (tuned) us
    r"\|\s*(N/A|-?[\d.]+)\s*%?\s*"  # Improve % ("N/A"/no % for NEW; may be <0)
    r"\|\s*(\S+)",  # Action/Reason token (UPDATE, NEW, SKIP, ...)
    re.IGNORECASE,
)


def _parse_tuner_stdout(stdout: str, stderr: str) -> list[dict[str, Any]]:
    """Parse per-shape results from aiter dense tuner output.

    Handles two aiter output formats: the older ``M=.. default:.. tuned:..
    speedup:..x`` key-value lines (format A) and the current ``--compare``
    comparison table ``(M, N, K) | Pre(us) | Post(us) | Improve% | Action``
    (format B). A given aiter version emits one format; both are tried so the
    parser tracks aiter across versions instead of silently returning nothing
    (which the caller would otherwise misreport as ``no_improvement``).
    """
    results: list[dict[str, Any]] = []
    # The per-row Action column is authoritative, but track the optional
    # "--- Would update ---"/"--- Skipped ---" section headers as a fallback so
    # a table lacking a clear row action is not silently misreported.
    in_would_update = False
    for line in (stdout + "\n" + stderr).splitlines():
        if re.search(r"---\s*(?:Would update|Updated)\b", line, re.IGNORECASE):
            in_would_update = True
        elif re.search(r"---\s*Skipped\b", line, re.IGNORECASE):
            in_would_update = False
        m = _STDOUT_KV_RE.search(line)
        if m:
            results.append(
                {
                    "M": int(m.group(1)),
                    "N": int(m.group(2)),
                    "K": int(m.group(3)),
                    "default_us": float(m.group(4)),
                    "tuned_us": float(m.group(5)),
                    "speedup": float(m.group(6)),
                    # The KV-format line reports a speedup but never carries the
                    # "Would update"/"Updated" tokens, so treat speedup>1.0 as the
                    # improvement signal (keeping the tokens as an explicit override).
                    "improved": float(m.group(6)) > 1.0 or "Would update" in line or "Updated" in line,
                }
            )
            continue
        t = _COMPARE_TABLE_RE.search(line)
        if t:
            pre_tok, post = t.group(4), float(t.group(5))
            action = t.group(7).strip().upper()
            if pre_tok.upper() == "N/A" or action == "NEW":
                # Newly-tuned shape: no baseline to microcompare, so we cannot
                # claim a micro speedup (improved=False, like the CSV fallback).
                # It IS a real tuned config though, so flag it is_new; the caller
                # forces an E2E candidate so the new config is validated end-to-end
                # rather than silently dropped as no_improvement.
                results.append(
                    {
                        "M": int(t.group(1)),
                        "N": int(t.group(2)),
                        "K": int(t.group(3)),
                        "default_us": None,
                        "tuned_us": post,
                        "speedup": None,
                        "improved": False,
                        "is_new": True,
                    }
                )
                continue
            pre = float(pre_tok)
            results.append(
                {
                    "M": int(t.group(1)),
                    "N": int(t.group(2)),
                    "K": int(t.group(3)),
                    "default_us": pre,
                    "tuned_us": post,
                    "speedup": round(pre / post, 4) if post > 0 else 1.0,
                    "improved": action == "UPDATE" or in_would_update,
                }
            )
    return results


def _parse_candidate_csv(candidate_path: Path | str | None) -> list[dict[str, Any]]:
    """Parse a written candidate CSV into per-shape tuned results.

    The candidate CSV holds the best config the tuner selected per (M, N, K) --
    one data row per shape. A written row means the tuner tuned that shape, but
    this aiter mode gives no untuned baseline, so each row is marked
    ``tuned_unverified`` with ``improved=False``: we cannot assert the tuned
    config beats the stock kernel from the row alone, so it must not be claimed
    as a micro win. It is still a real tuned config, so the caller forces an E2E
    candidate for it (as it does for split-K and new shapes) and lets the
    end-to-end measurement decide KEEP.

    This is the fallback for the aiter output mode that prints only a
    "Successfully tuned shapes" summary (no per-shape Pre/Post table): a real
    tuned artifact exists even though stdout has nothing to parse.

    Expected header (columns resolved by name so index shifts across aiter
    versions do not break parsing):
        gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio

    ``default_us``/``speedup`` are left ``None``: this aiter mode gives no
    comparable untuned baseline, so we do not fabricate one.

    Robust to a missing file, header variants, and short/garbage rows: bad rows
    are skipped and the function never raises.
    """
    results: list[dict[str, Any]] = []
    if candidate_path is None:
        return results
    path = Path(candidate_path)
    try:
        if not path.is_file():
            return results
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return results
    if not lines:
        return results
    header = [h.strip() for h in lines[0].split(",")]
    idx = {h.upper(): i for i, h in enumerate(header)}
    if not {"M", "N", "K", "US"}.issubset(idx):
        return results
    mi, ni, ki, ui = idx["M"], idx["N"], idx["K"], idx["US"]
    need = max(mi, ni, ki, ui)
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if need >= len(parts):
            continue  # short row
        try:
            m, n, k = int(parts[mi]), int(parts[ni]), int(parts[ki])
            tuned_us = float(parts[ui])
        except (ValueError, IndexError):
            continue  # unparseable row
        results.append(
            {
                "M": m,
                "N": n,
                "K": k,
                "tuned_us": tuned_us,
                "default_us": None,
                "speedup": None,
                # No comparable default was measured in this aiter output mode, so we
                # cannot claim the tuned config beats the stock kernel. Mark the shape
                # tuned-but-unverified (improved=False) so no micro win is reported;
                # the caller sends it to E2E on the strength of tuned_unverified.
                "improved": False,
                "tuned_unverified": True,
            }
        )
    return results


def _summarize_shape_results(shape_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive a TuneResult status + metrics from parsed per-shape results.

    Strict status separation (A2b): an empty parse (rc==0 but nothing usable --
    a tuner quick-exit, an unrecognized output format, or an empty candidate)
    is ``empty_output``, NOT ``no_improvement``. ``no_improvement`` requires at
    least one parsed shape and neither an ``improved`` row nor an ``unverified``
    row (new shape or candidate-CSV fallback with no baseline).

    When aiter tunes more than 30 shapes it writes the Pre/Post table to
    ``/tmp/aiter_compare/*.compare.txt``; those rows arrive here with real
    ``default_us``/``tuned_us`` and can yield ``ok`` via ``improved`` or
    ``unverified`` the same as stdout-parsed rows.
    """
    total = len(shape_results)
    if total == 0:
        return {
            "status": "empty_output",
            "total": 0,
            "n_improved": 0,
            "n_unverified": 0,
            "best": 1.0,
            "avg": 1.0,
        }
    improved = [r for r in shape_results if r.get("improved")]
    # Tuned, but with nothing to compare against (new shape, or the candidate-CSV
    # fallback). Counted separately so a report cannot read "improved 0/N" as
    # "measured N shapes and none got faster".
    unverified = [r for r in shape_results if r.get("is_new") or r.get("tuned_unverified")]
    # A speedup may be None in the candidate-CSV fallback path (no comparable
    # default is available), so guard the numeric comparison. Improved rows
    # still make the status "ok" even when speedups are unknown.
    speedups = [
        r["speedup"] for r in shape_results if isinstance(r.get("speedup"), (int, float)) and r["speedup"] > 1.0
    ]
    return {
        "status": "ok" if (improved or unverified) else "no_improvement",
        "total": total,
        "n_improved": len(improved),
        "n_unverified": len(unverified),
        "best": max(speedups) if speedups else 1.0,
        "avg": sum(speedups) / len(speedups) if speedups else 1.0,
    }


def run_aiter_dense_tuner(
    *,
    tuner_name: str,
    script_key: str,
    env_var: str,
    ctx: TuneContext,
    work_dir: Path,
    extra_args: list[str] | None = None,
) -> TuneResult:
    """Run an aiter dense GEMM tuner subprocess.

    Args:
        tuner_name: Name for the TuneResult.
        script_key: Key in AITER_TUNER_SCRIPTS to find the script.
        env_var: Environment variable for the output.
        ctx: Tuning context.
        work_dir: Working directory for this tuner.
        extra_args: Additional CLI args (e.g. --libtype all).
    """
    script = find_tuner_script(script_key)
    if script is None:
        return TuneResult(
            tuner_name=tuner_name,
            status="failed",
            error=f"Tuner script not found for {script_key}",
            error_class="script_missing",
        )

    # a8w8 and bpreshuffle need q_dtype_w column in CSV
    needs_q_dtype_w = tuner_name in ("a8w8", "a8w8_bpreshuffle")
    # Demand outranks every other shape source: it is the set of keys the runtime
    # asked for and did not have. Everything else is inference about that set.
    input_csv = _demand_input_csv(ctx, work_dir, tuner_name, needs_q_dtype_w=needs_q_dtype_w) or _resolve_input_csv(
        ctx, work_dir, needs_q_dtype_w=needs_q_dtype_w
    )
    if input_csv is None:
        return TuneResult(
            tuner_name=tuner_name,
            status="failed",
            error="No input CSV or shapes JSON available",
            error_class="input_missing",
        )

    tuned_csv = work_dir / f"tuned_{tuner_name}.csv"
    profile_csv = work_dir / f"profile_{tuner_name}.csv"

    # Flags shared by every shape (everything except -i/-o). aiter --timeout is
    # injected below to activate mp_tuner's per-candidate GPU-fault isolation.
    base_args = [
        "-o2",
        str(profile_csv),
        "--mp",
        str(ctx.mp),
        "--compare",
        "--iters",
        str(ctx.iters),
        "--warmup",
        str(ctx.warmup),
        "--min_improvement_pct",
        str(ctx.min_improvement_pct),
        "-v",
    ]
    if extra_args:
        base_args.extend(extra_args)

    # Check the script's argparse surface before spending minutes on it. Losing
    # --splitK or --mxfp4-flydsl does not degrade the search, it empties it, so
    # those are refused up front instead of producing a completed run that
    # reports nothing gained.
    filtered = filter_args(base_args, probe_script(script))
    if not filtered.ok:
        return TuneResult(
            tuner_name=tuner_name,
            status="failed",
            error=(
                f"{script} does not accept {', '.join(filtered.rejected_required)}; "
                "without it the tuner has no candidates to search"
            ),
            error_class="unsupported_argument",
        )
    base_args = filtered.args

    aiter_root = resolve_aiter_root()

    import time

    run_start_time = time.time()

    iso_candidate: Path | None = None
    if _tr.is_isolation_enabled():
        # Per-shape process isolation + provenance-keyed fault blocklist.
        blocklist = _tr.FaultBlocklist(
            getattr(ctx, "faulted_blocklist_path", None),
            {
                "gpu_type": ctx.gpu_type,
                "quant_type": getattr(ctx, "quant_type", ""),
                "tp": getattr(ctx, "tp", 1),
                "tuner": tuner_name,
            },
        )
        rc, stdout, stderr, iso_candidate = _tr.run_isolated(
            script=str(script),
            base_args=base_args,
            input_csv=input_csv,
            tuned_stem=tuned_csv.stem,
            work_dir=work_dir,
            aiter_root=aiter_root,
            outer_timeout_s=ctx.timeout_s,
            task_timeout_s=_tr.DEFAULT_TASK_TIMEOUT_S,
            gpu_ids=getattr(ctx, "gpu_ids", "") or "",
            blocklist=blocklist,
        )
    else:
        # Default single invocation, now with --timeout so a faulting candidate
        # is isolated by aiter instead of hanging the whole run.
        cmd = _tr.with_task_timeout(["python3", str(script), "-i", str(input_csv), "-o", str(tuned_csv), *base_args])
        rc, stdout, stderr = run_subprocess(
            cmd,
            cwd=aiter_root,
            timeout_s=ctx.timeout_s,
            log_file=work_dir / "tune.log",
        )

    if rc == 124:
        return TuneResult(
            tuner_name=tuner_name,
            status="failed",
            error=f"Tuning timed out after {ctx.timeout_s}s",
            error_class="timeout",
        )

    if rc != 0:
        return TuneResult(
            tuner_name=tuner_name,
            status="failed",
            error=f"Tuner exited with code {rc}: {stderr[-500:]}",
            error_class="subprocess_error",
        )

    # Find candidate CSV. Isolation merges per-shape candidates into one file it
    # returns directly; otherwise glob the aiter compare dir (files newer than
    # our run start).
    candidate = iso_candidate if iso_candidate is not None else _find_latest_candidate(tuner_name, run_start_time)
    artifact = str(candidate) if candidate else str(tuned_csv)

    if candidate and candidate.is_file():
        dest = work_dir / f"candidate_{tuner_name}.csv"
        shutil.copy2(candidate, dest)
        artifact = str(dest)

    # When split-K search is enabled the aiter *tuner* can pick a splitK the
    # production dispatch cannot run (serving it raises "This GEMM is not
    # supported!" and crashes engine init). Re-select serve-safe splitK on the
    # deployed artifact using the full-candidate profile.
    # split-K's benefit is e2e-only: it is invisible to (or within the noise of)
    # the tuner microbench, so the micro-based candidate gate
    # (improved_shapes>0 and best_micro>1.0) would veto a real e2e gain (the tuned
    # CSV can report best_micro==1.0 yet deliver several % e2e). Force e2e
    # validation whenever the (serve-safe-capped) deployed CSV carries split-K>0.
    force_candidate = False
    if "--splitK" in (extra_args or []):
        max_splitk = int(os.environ.get("FORGE_MAX_SPLITK", "2"))
        # Prefer the REAL per-shape production split-K limit (trial-dispatch) over
        # the static FORGE_MAX_SPLITK: it keeps splitK>cap where the kernel
        # actually supports it and tightens below cap where it does not. Disable
        # with FORGE_SPLITK_TRIAL=0; falls back to the static cap per shape when
        # the trial can't run (no GPU / aiter not importable in this process).
        # The trial dispatches gemm_a8w8_blockscale_ck specifically, so it is only
        # correct for the a8w8_blockscale op; other dense ops (a8w8/bpreshuffle/
        # a4w4) would be validated against the WRONG kernel -> keep them on the
        # static cap until aiter_splitk_validate is made op-aware (see #27).
        support_fn = None
        if script_key == SPLITK_TRIAL_SCRIPT_KEY and os.environ.get("FORGE_SPLITK_TRIAL", "1") != "0":
            try:
                from ..aiter_splitk_validate import make_support_fn

                # Pin the in-process trial dispatch to the tuner's assigned card;
                # on a shared node the assigned GPU may not be device 0.
                support_fn = make_support_fn(gpu_ids=getattr(ctx, "gpu_ids", "") or "")
            except Exception:  # noqa: BLE001 — fall back to the static cap
                support_fn = None
        n_capped, force_candidate = _cap_splitk_to_serve_safe(
            Path(artifact), profile_csv, max_splitk, support_fn=support_fn
        )
        if n_capped:
            log.info(
                "serve-safe splitK cap: rewrote/dropped %d row(s) beyond production support",
                n_capped,
            )

    shape_results = _parse_tuner_stdout(stdout, stderr)
    if not shape_results:
        # aiter writes the --compare table to /tmp/aiter_compare/ when >30 shapes
        # (stdout carries only a "Successfully tuned N shapes" summary). Recover
        # per-shape Pre/Post timing from that report before falling back to the
        # candidate CSV (which has no baseline -> tuned_unverified).
        compare_report = _find_latest_compare_report(tuner_name, run_start_time)
        if compare_report is not None and compare_report.is_file():
            log.info(
                "compare report found for %s: %s",
                tuner_name,
                compare_report,
            )
            dest = work_dir / f"compare_{tuner_name}.txt"
            try:
                shutil.copy2(compare_report, dest)
            except OSError as exc:
                log.warning(
                    "failed to archive compare report for %s (%s -> %s): %s",
                    tuner_name,
                    compare_report,
                    dest,
                    exc,
                )
            else:
                log.info("archived compare report for %s to %s", tuner_name, dest)
            try:
                shape_results = _parse_tuner_stdout(compare_report.read_text(encoding="utf-8", errors="replace"), "")
            except OSError:
                shape_results = []
        else:
            log.info(
                "no compare report for %s under /tmp/aiter_compare after run start",
                tuner_name,
            )
    if not shape_results:
        # Some aiter versions print only a "Successfully tuned shapes" summary
        # (no per-shape Pre/Post table) while still writing a valid tuned
        # candidate CSV. Recover the tuned shapes from that CSV so a real tuned
        # artifact reports ok/candidate instead of empty_output. A genuinely
        # empty run (no stdout parse AND no candidate rows) still falls through
        # to empty_output.
        candidate_csv_path = work_dir / f"candidate_{tuner_name}.csv"
        fallback_rows = _parse_candidate_csv(candidate_csv_path)
        if fallback_rows:
            shape_results = fallback_rows

    # improved=False carries two different meanings: "compared against a baseline
    # and did not win", and "never had a baseline to compare against". Only the
    # first is a performance result. The second covers newly-tuned shapes
    # (is_new) and every row recovered from the candidate CSV
    # (tuned_unverified, the aiter output mode with no per-shape Pre/Post
    # table) -- neither can show a micro speedup, so the micro gate
    # (improved_shapes>0 and best_micro>1.0) would drop them. Force an E2E
    # candidate -- exactly as split-K does -- so those configs are proven
    # end-to-end instead of silently discarded as no_improvement.
    # Being wrong disqualifies a row before being slow does, so the accuracy
    # filter runs first: a kernel that computes the wrong answer must not reach
    # serving even when it won its comparison.
    dropped_inaccurate = drop_inaccurate_rows(Path(artifact))
    if dropped_inaccurate:
        shape_results = _forget_shapes_that_lost_their_row(shape_results, dropped_inaccurate)

    if any(r.get("is_new") or r.get("tuned_unverified") for r in shape_results):
        force_candidate = True

    # A row that lost its comparison would override a better stock choice once
    # merged, so it is removed from the deployed artifact. Rows with no baseline
    # survive -- see _filter_unimproved_rows.
    n_dropped, n_kept = _filter_unimproved_rows(Path(artifact), shape_results)
    if n_dropped:
        log.info(
            "deployed artifact: dropped %d row(s) that were compared and did not win, %d kept",
            n_dropped,
            n_kept,
        )

    summary = _summarize_shape_results(shape_results)

    return TuneResult(
        tuner_name=tuner_name,
        status=summary["status"],
        artifact_path=artifact,
        env_var=env_var,
        env_value=artifact,
        total_shapes=summary["total"],
        improved_shapes=summary["n_improved"],
        unverified_shapes=summary["n_unverified"],
        best_micro_speedup=summary["best"],
        avg_micro_speedup=summary["avg"],
        candidate=force_candidate,
        shape_results=shape_results,
        dropped_inaccurate=[
            {
                "M": r.get("M"),
                "N": r.get("N"),
                "K": r.get("K"),
                "libtype": r.get("libtype"),
                "splitK": r.get("splitK"),
                "us": r.get("us"),
                "err_ratio": _row_err_ratio(r),
            }
            for r in dropped_inaccurate
        ],
    )


def _forget_shapes_that_lost_their_row(
    shape_results: list[dict[str, Any]],
    dropped_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Stop reporting a speedup for a shape whose winner was just removed.

    The accuracy filter deletes rows from the artifact, but the per-shape
    numbers were parsed before that. Left alone, a run reports "1.24x on
    M=16" while the artifact holds nothing for M=16 -- a gain claimed for a
    kernel that will never be served, which is the exact failure this whole
    path exists to prevent.

    A shape is dropped from the report rather than rewritten: the tuner
    compared against the disqualified kernel, so the surviving rows have no
    trustworthy comparison behind them. Under-claiming here is the safe
    direction.
    """
    poisoned = {(str(r.get("M")), str(r.get("N")), str(r.get("K"))) for r in dropped_rows}
    kept = [r for r in shape_results if (str(r.get("M")), str(r.get("N")), str(r.get("K"))) not in poisoned]
    if len(kept) != len(shape_results):
        log.warning(
            "not reporting %d shape(s) whose best row was dropped as numerically "
            "wrong; %d shape(s) still have deployable results",
            len(shape_results) - len(kept),
            len(kept),
        )
    return kept


def _filter_unimproved_rows(
    artifact_csv: Path,
    shape_results: list[dict[str, Any]],
) -> tuple[int, int]:
    """Drop deployed rows for shapes that were compared and lost.

    A tuned row that lost its comparison is worse than useless: merged into the
    served table it *overrides* a stock choice that was already better.

    Rows whose shape had no comparable baseline are kept. "Not measured to be
    better" and "measured to be not better" are different claims, and only the
    second justifies deleting a row -- the first covers newly-tuned shapes, the
    candidate-CSV fallback and hipblaslt-only runs, i.e. exactly the configs
    the forced-e2e path exists to protect. Dropping them here would undo that.

    Returns ``(rows_dropped, rows_kept)``. Never raises: on any parse trouble
    the artifact is left exactly as it was.
    """
    losers: set[tuple[int, int, int]] = set()
    for r in shape_results:
        if r.get("improved"):
            continue
        if r.get("is_new") or r.get("tuned_unverified"):
            continue
        if r.get("speedup") is None and r.get("default_us") is None:
            # No baseline recorded at all -> not a loss, just unmeasured.
            continue
        try:
            losers.add((int(r["M"]), int(r["N"]), int(r["K"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not losers:
        return 0, 0

    try:
        lines = artifact_csv.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return 0, 0
    if len(lines) < 2:
        return 0, 0

    header = [h.strip().lower() for h in lines[0].split(",")]
    try:
        mi, ni, ki = header.index("m"), header.index("n"), header.index("k")
    except ValueError:
        log.warning("cannot filter unimproved rows: %s has no M/N/K header", artifact_csv)
        return 0, 0

    kept_lines = [lines[0]]
    dropped = 0
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        try:
            key = (int(parts[mi]), int(parts[ni]), int(parts[ki]))
        except (IndexError, ValueError):
            kept_lines.append(line)  # unparseable: keep rather than guess
            continue
        if key in losers:
            dropped += 1
            continue
        kept_lines.append(line)

    if dropped:
        try:
            artifact_csv.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
        except OSError as exc:
            # The file on disk still holds every row, so report what it holds.
            # Returning the filtered count here described a file that was never
            # written, and the caller logs those numbers as what it deployed.
            log.warning("could not rewrite %s after filtering: %s", artifact_csv, exc)
            return 0, len(lines) - 1
    return dropped, len(kept_lines) - 1


def _cap_splitk_to_serve_safe(
    artifact_csv: Path, profile_csv: Path, max_splitk: int, support_fn=None
) -> tuple[int, bool]:
    """Rewrite deployed rows whose splitK exceeds production-dispatch support.

    aiter's tuner (`gemm_a8w8_blockscale_*_tune`) benchmarks split-K values that
    the production kernel (`gemm_a8w8_blockscale_ck`) cannot dispatch; serving
    such a row raises "This GEMM is not supported!" and crashes engine init. Each
    row whose splitK exceeds what the kernel supports for its (M,N,K) is replaced
    by the fastest full-candidate-profile config within support (valid errRatio);
    a shape with no safe candidate is dropped (aiter default at serve, no crash).

    The per-shape limit is ``support_fn(M,N,K)`` when given -- the REAL production
    limit found by trial-dispatch, which varies per shape (some support splitK=3);
    ``None`` from it => fall back to the static ``max_splitk`` for that shape. When
    ``support_fn`` is None the static ``max_splitk`` is used for every shape.

    Returns ``(rows_rewritten_or_dropped, deployed_csv_has_any_splitk_gt0)``.
    Best-effort: on any read error the artifact is left unchanged.
    """
    try:
        with artifact_csv.open() as f:
            rows = list(csv.reader(f))
    except OSError:
        return 0, False
    if len(rows) < 2:
        return 0, False
    hdr = rows[0]
    # Case-insensitive column lookup: if the deployed-header case ever fails an
    # exact match the cap would return early (0, False) and pass unsafe splitK
    # rows through unchanged -> serve crash. Resolve columns case-insensitively
    # so a future aiter header-case change cannot silently disable the cap.
    _col = {str(h).strip().lower(): i for i, h in enumerate(hdr)}
    try:
        mi, ni, ki, ski = (_col[c] for c in ("m", "n", "k", "splitk"))
    except KeyError:
        return 0, False

    # Index every valid candidate per shape; the cap is applied per-shape at
    # selection so support_fn can keep splitK>max_splitk where the kernel supports.
    # ``us`` is read from the profile rows (r["us"]) below, not the deployed
    # header, so a tuned CSV without a "us" column can still be capped.
    by_shape: dict[tuple[str, str, str], list[tuple[float, int, list[str]]]] = defaultdict(list)
    schema_ok = True
    try:
        with profile_csv.open() as f:
            for r in csv.DictReader(f):
                # A candidate must carry every column the deployed CSV has, or the
                # rewritten row would get empty cells and a renamed/absent errRatio
                # would silently disable the correctness filter. Skip such rows.
                if any(c not in r for c in hdr):
                    schema_ok = False
                    continue
                try:
                    us, sk = float(r["us"]), int(r["splitK"])
                    er = float(r.get("errRatio") or 0)  # absent -> 0 (no KeyError)
                except (KeyError, ValueError, TypeError):
                    continue
                if us <= 0 or er > 0.01:
                    continue
                by_shape[(r["M"], r["N"], r["K"])].append((us, sk, [r[c] for c in hdr]))
    except OSError:
        by_shape = defaultdict(list)
    if not schema_ok:
        log.warning(
            "splitK cap: profile %s lacks columns present in the tuned CSV; some "
            "serve-safe candidates were skipped (possible aiter schema drift)",
            profile_csv,
        )

    def _shape_max(m: int, n: int, k: int) -> int:
        if support_fn is None:
            return max_splitk
        try:
            v = support_fn(m, n, k)
        except Exception:  # noqa: BLE001 — trial failure must not abort the cap
            return max_splitk  # degrade to the static cap, never crash the tuner
        return max_splitk if v is None else int(v)

    out, changed, has_splitk = [hdr], 0, False
    for row in rows[1:]:
        try:
            sk = int(row[ski])
        except (ValueError, IndexError):
            out.append(row)
            continue
        if sk == 0:
            # splitK=0 is the default dispatch: always serve-safe, and its
            # keep decision never depends on the per-shape max, so skip the
            # (GPU-dispatching) trial entirely for these rows.
            out.append(row)
            continue
        try:
            key = (row[mi], row[ni], row[ki])
            maxsk = _shape_max(int(row[mi]), int(row[ni]), int(row[ki]))
        except (ValueError, IndexError):
            out.append(row)
            continue
        if sk <= maxsk:
            out.append(row)
            has_splitk = has_splitk or sk > 0
            continue
        safe = min(
            (c for c in by_shape.get(key, ()) if c[1] <= maxsk),
            key=lambda c: c[0],
            default=None,
        )
        if safe is not None:
            out.append(safe[2])
            has_splitk = has_splitk or safe[1] > 0
        # else: drop the row -> serve falls back to the aiter default
        changed += 1
    if changed:
        with artifact_csv.open("w", newline="") as f:
            csv.writer(f).writerows(out)
    return changed, has_splitk


def _stem_matches(tuner_name: str, filename: str) -> bool:
    """Whether ``filename`` is a candidate CSV produced for ``tuner_name``.

    aiter names dense candidates ``tuned_<tuner>.candidate.csv`` and the isolated
    per-shape runner names them ``_iso_tuned_<tuner>_<idx>_tuned...``. A plain
    substring test on ``tuned_<tuner>`` is wrong: the tuner names nest by prefix
    (``a8w8`` < ``a8w8_blockscale`` < ``a8w8_blockscale_bpreshuffle``), so a
    shorter name would falsely claim a longer sibling's CSV. Require the stem to
    be followed by ``.`` (extension) or ``_<digit>`` (the shape index) so a
    sibling's trailing ``_<name>`` token can never match.
    """
    stem = f"tuned_{tuner_name}"
    return re.search(re.escape(stem) + r"(?:\.|_\d)", filename) is not None


def _find_latest_compare_report_impl(
    tuner_name: str,
    start_time: float,
    compare_dir: Path,
) -> Path | None:
    """Find the most recent compare report from ``compare_dir`` for THIS run.

    aiter names dense compare reports ``tuned_<tuner>.<pid>.compare.txt``.
    Uses the same stem whole-token matching and mtime gate as candidate CSV
    lookup so concurrent runs, stale files, and sibling tuner names cannot
    pollute results.
    """
    if not compare_dir.is_dir():
        return None
    reports = [
        p
        for p in compare_dir.glob("*.compare.txt")
        if p.stat().st_mtime > start_time and _stem_matches(tuner_name, p.name)
    ]
    if not reports:
        return None
    reports.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return reports[0]


def _find_latest_compare_report(tuner_name: str, start_time: float) -> Path | None:
    """Find the most recent compare report from /tmp/aiter_compare/ for THIS run."""
    return _find_latest_compare_report_impl(tuner_name, start_time, Path("/tmp/aiter_compare"))


def _find_latest_candidate(tuner_name: str, start_time: float) -> Path | None:
    """Find the most recent candidate CSV from /tmp/aiter_compare/ for THIS run.

    Matches by:
    1. mtime > start_time (rejects stale)
    2. filename carries the tuner_name stem as a whole token (rejects concurrent
       runs' candidates AND sibling tuners whose name merely EXTENDS this one --
       e.g. a8w8_blockscale must not pick up a8w8_blockscale_bpreshuffle)

    Returns None if no matching candidate (no fallback to avoid pollution).
    """
    compare_dir = Path("/tmp/aiter_compare")
    if not compare_dir.is_dir():
        return None
    candidates = [
        p
        for p in compare_dir.glob("*.candidate.csv")
        if p.stat().st_mtime > start_time and _stem_matches(tuner_name, p.name)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]
