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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .base import TuneContext, TuneResult
from ..script_probe import filter_args, probe_script
from ..utils import find_tuner_script, resolve_aiter_root, run_subprocess
from .. import tune_robustness as _tr

log = logging.getLogger(__name__)

# The only op whose production dispatch the per-shape split-K trial validates:
# aiter_splitk_validate can only build a trial for ops it has a registered
# production callable for, so the trial is correct only for this script_key.
SPLITK_TRIAL_SCRIPT_KEY = "a8w8_blockscale"

# Which (op, libtype) pairs reach a production wrapper that actually forwards
# the tuned row's ``splitK``. A tuned row is a measurement plus a set of kernel
# parameters, and the measurement is only honoured if every parameter survives
# the trip to the serving call -- a dropped ``splitK`` deploys a config that was
# benchmarked with split-K and runs without it, i.e. slower than measured while
# every engagement gate still reports the artifact as served. Read off
# ``aiter/ops/gemm_op_a8w8.py``:
#
#   gemm_a8w8_blockscale               ck -> splitK=splitK   cktile -> splitK=splitK
#   gemm_a8w8_blockscale_bpreshuffle   ck -> kernelName only  cktile -> kernelName only
#                                      asm -> splitK=splitK   opus/flydsl -> no splitK
#
# Fail closed: an op absent from this table is treated as forwarding nothing, so
# adding a tuner (or a new libtype) can only under-claim, never silently ship a
# row whose splitK the runtime discards. Verified against aiter
# d9e5ef7ce08ee7045d583aed768cff41aa9210fe; re-check on an aiter bump.
_SPLITK_FORWARDING_LIBTYPES: dict[str, frozenset[str]] = {
    "a8w8_blockscale": frozenset({"ck", "cktile"}),
    "a8w8_blockscale_bpreshuffle": frozenset({"asm"}),
}


class AiterDtypeUnavailable(RuntimeError):
    """The installed aiter cannot supply a dtype the tuner CSV needs."""


def _aiter_dtype_str(attr: str) -> str:
    """Return the repr string aiter's tuner scripts accept for ``dtypes.<attr>``."""
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
    """``Path.is_file()`` guarded against ``OSError(ENAMETOOLONG)``."""
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
    """Shared validate() for the aiter dense fp8/fp4 tuners."""
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


# Mean measured cost of tuning one shape; used to size the shape list against the time budget rather than tuning a
# list we cannot finish.
_DEMAND_PER_SHAPE_COST_S = 74
_DEMAND_PER_SHAPE_COST_THOROUGH_S = 420
_DEMAND_RESERVE_S = 120
_DEMAND_MAX_SHAPES_ENV = "FORGE_DEMAND_MAX_SHAPES"


# Fraction of output elements aiter's own accuracy check found wrong.
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
    """Remove rows aiter measured as numerically wrong, in place."""
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

    # Write beside the artifact and rename over it.
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
    """Total shapes the time budget affords, decode band included.

    The caller splits this: the decode band is mandatory and is reserved first,
    the request-ranked prefill tail claims what is left. Overrunning is not a
    soft failure -- ``ctx.timeout_s`` is the subprocess deadline, and a dense
    tuner killed at that deadline returns no candidate at all, so an
    unaffordable shape list costs the prefill rows too.
    """
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

    The ranking is sound for the prefill tail but structurally cannot see the
    decode band (see :func:`..evidence.demand_shapes`), so the band comes from
    the concurrency contract via :func:`_ensure_decode_m_coverage` and is paid
    for first: the prefill tail claims only the shapes still affordable once
    the band's rows are reserved.
    """
    path = getattr(ctx, "demand_json", None)
    if not path:
        return None
    from ..dense_shapes import compute_decode_m_values
    from ..evidence import demand_for_tuner, demand_shapes, load_demand

    report = load_demand(path)
    if report is None:
        return None
    entry = demand_for_tuner(report, tuner_name)
    if entry is None:
        return None
    # The a8w8 blockscale, a8w8 quant-type, and a4w4 lookup paths all retry the exact M followed by get_padded_m(...,
    # gl=0) and gl=1, using the same gemm_op_common implementation as a16w16.
    ranked = demand_shapes(entry)
    if not ranked:
        return None
    budget = _demand_budget(ctx)
    shapes = _demand_shapes_within_budget(ranked, compute_decode_m_values(ctx.conc), budget)

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
        "%s: %d of %d ranked padded-M demand shapes (of %d distinct keys), leaving the decode band its share of "
        "the %d-shape budget -> %s",
        tuner_name,
        len(shapes),
        len(ranked),
        entry.get("distinct_keys", 0),
        budget,
        out,
    )
    return _ensure_decode_m_coverage(out, ctx, work_dir, needs_q_dtype_w=needs_q_dtype_w)


def _demand_shapes_within_budget(
    ranked: list[dict[str, Any]],
    decode_m: Sequence[int],
    budget: int,
) -> list[dict[str, Any]]:
    """The highest-ranked demand shapes whose decode band ``budget`` can also pay for.

    Walks the ranking and takes a shape while the total -- shapes taken plus
    the band rows their dispatch groups still lack -- stays inside ``budget``.
    A shape opening a new group carries that group's whole band with it, so it
    can be passed over in favour of a lower-ranked shape in a group already
    paid for.

    Trimming groups rather than the band inside a group is deliberate: a group
    holding part of its band serves the uncovered decode M with a prefill tile,
    which is the regression the band exists to prevent, while a group left out
    entirely keeps whatever the shipped tables already give it.
    """
    group_m: dict[tuple[int, int], set[int]] = {}
    taken: list[dict[str, Any]] = []
    for shape in ranked:
        key = (int(shape["N"]), int(shape["K"]))
        trial = dict(group_m)
        trial[key] = group_m.get(key, set()) | {int(shape["M"])}
        band = sum(len(b) for b in _decode_band_buckets([(n, ms) for (n, _k), ms in trial.items()], decode_m))
        if len(taken) + 1 + band <= budget:
            group_m = trial
            taken.append(shape)
    if taken:
        return taken
    log.warning(
        "the top-ranked dispatch group's decode band alone exceeds the %d-shape budget; tuning it anyway and "
        "accepting the overrun, because a band with holes is the regression this guarantee exists to prevent",
        budget,
    )
    return ranked[:1]


def _resolve_input_csv(ctx: TuneContext, work_dir: Path, needs_q_dtype_w: bool = False) -> Path | None:
    """Resolve the input untuned CSV for a dense tuner."""
    csv: Path | None = None
    from_manifest = False
    if _safe_is_file(getattr(ctx, "shapes_manifest", None)):
        from ..shape_manifest import write_manifest_untuned_csv

        csv = write_manifest_untuned_csv(ctx.shapes_manifest, work_dir, needs_q_dtype_w=needs_q_dtype_w)
        from_manifest = csv is not None
        # Manifest yielded no tunable target shapes: fall through to the other sources rather than failing outright.
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
    """aiter's ``get_padded_m(..., gl=0)``: round up to a tile multiple."""
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
    """aiter's ``get_padded_m(..., gl=1)``, which is not a plain power of two."""
    if int(m) > 8192 and int(n) > 4096:
        return 8192
    return _next_pow2(m)


def _dispatch_lookup_ms(m: int, n: int) -> set[int]:
    """The tuned-M values aiter will accept when serving runtime batch ``m``."""
    return {int(m), _padded_m_gl0(m), _padded_m_gl1(m, n)}


def _decode_band_buckets(
    groups: Sequence[tuple[int, set[int]]],
    decode_m: Sequence[int],
) -> list[list[int]]:
    """Per group, the padded-M buckets it still needs to cover the decode band.

    ``groups`` is ``(N, tuned M)`` per dispatch group; N alone decides which
    tuned M a runtime batch can reach, through ``get_padded_m(..., gl=1)``.
    Sizing the reservation and appending the rows both read this, so the two
    cannot drift.
    """
    plan: list[list[int]] = []
    for n, tuned in groups:
        reachable = set(tuned)
        buckets: list[int] = []
        for m in decode_m:
            if reachable & _dispatch_lookup_ms(m, n):
                continue
            bucket = _padded_m_gl0(m)
            reachable.add(bucket)  # also serves the other grid M padding into it
            buckets.append(bucket)
        plan.append(buckets)
    return plan


def _ensure_decode_m_coverage(
    csv: Path,
    ctx: TuneContext,
    work_dir: Path,
    needs_q_dtype_w: bool = False,
) -> Path:
    """Guarantee every tuned dispatch group covers the decode-band M."""
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

    # Group key = the aiter dispatch key.
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
    plan = _decode_band_buckets([(key[0], group_m[key]) for key in group_order], decode_m)
    additions: list[str] = []
    uncovered = 0
    for (n, k, q), buckets in zip(group_order, plan):
        if not buckets:
            continue
        uncovered += 1
        for bucket in buckets:
            row = [""] * len(header)
            row[idx["M"]], row[idx["N"]], row[idx["K"]] = str(bucket), str(n), str(k)
            if q_idx is not None:
                row[q_idx] = q
            additions.append(",".join(row))

    if not additions:
        return csv

    out = work_dir / "decode_covered_dense.csv"
    work_dir.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([lines[0], *body, *additions]) + "\n", encoding="utf-8")
    log.info(
        "Decode coverage: %d of %d dispatch group(s) lacked a decode-band "
        "M (grid %s for conc=%s); appended %d row(s), original %d row(s) untouched",
        uncovered,
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
    """Augment profile-derived shapes with config-derived M values."""
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
    """Return a CSV whose columns match what this tuner expects."""
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
    """Synthesize an untuned CSV from the model config when none was supplied."""
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
    """Convert a shapes JSON file to aiter's untuned CSV format."""
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

# Format B (current aiter --compare table): the "Would update" comparison block "(8192, 5120, 5120) | 1037.74 | 269.71
# | 74.01% | UPDATE" "(8192, 5120, 5120) | N/A | 269.71 | N/A | NEW" (new shape) columns: (M, N, K) | Pre(us) |
# Post(us) | Improve% | Action A shape with no prior tuned entry has no baseline to compare against, so aiter prints
# "N/A" for Pre and Improve% and marks the row NEW.
_COMPARE_TABLE_RE = re.compile(
    r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*"
    r"\|\s*(N/A|[\d.]+)\s*"  # Pre (default) us -- "N/A" for a NEW shape
    r"\|\s*([\d.]+)\s*"  # Post (tuned) us
    r"\|\s*(N/A|-?[\d.]+)\s*%?\s*"  # Improve % ("N/A"/no % for NEW; may be <0)
    r"\|\s*(\S+)",  # Action/Reason token (UPDATE, NEW, SKIP, ...)
    re.IGNORECASE,
)


def _parse_tuner_stdout(stdout: str, stderr: str) -> list[dict[str, Any]]:
    """Parse per-shape results from aiter dense tuner output."""
    results: list[dict[str, Any]] = []
    # The per-row Action column is authoritative, but track the optional "--- Would update ---"/"--- Skipped ---"
    # section headers as a fallback so a table lacking a clear row action is not silently misreported.
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
                    # The KV-format line reports a speedup but never carries the "Would update"/"Updated" tokens, so
                    # treat speedup>1.0 as the improvement signal (keeping the tokens as an explicit override).
                    "improved": float(m.group(6)) > 1.0 or "Would update" in line or "Updated" in line,
                }
            )
            continue
        t = _COMPARE_TABLE_RE.search(line)
        if t:
            pre_tok, post = t.group(4), float(t.group(5))
            action = t.group(7).strip().upper()
            if pre_tok.upper() == "N/A" or action == "NEW":
                # Newly-tuned shape: no baseline to microcompare, so we cannot claim a micro speedup (improved=False,
                # like the CSV fallback).
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
    """Parse a written candidate CSV into per-shape tuned results."""
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
                # No comparable default was measured in this aiter output mode, so we cannot claim the tuned config
                # beats the stock kernel.
                "improved": False,
                "tuned_unverified": True,
            }
        )
    return results


def _summarize_shape_results(shape_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive a TuneResult status + metrics from parsed per-shape results."""
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
    # Tuned, but with nothing to compare against (new shape, or the candidate-CSV fallback).
    unverified = [r for r in shape_results if r.get("is_new") or r.get("tuned_unverified")]
    # A speedup may be None in the candidate-CSV fallback path (no comparable default is available), so guard the
    # numeric comparison.
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
    """Run an aiter dense GEMM tuner subprocess."""
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
    # Demand outranks every other shape source: it is the set of keys the runtime asked for and did not have.
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

    # Flags shared by every shape (everything except -i/-o). aiter --timeout is injected below to activate mp_tuner's
    # per-candidate GPU-fault isolation.
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

    # Check the script's argparse surface before spending minutes on it.
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
        # Default single invocation, now with --timeout so a faulting candidate is isolated by aiter instead of
        # hanging the whole run.
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

    # Find candidate CSV.
    candidate = iso_candidate if iso_candidate is not None else _find_latest_candidate(tuner_name, run_start_time)
    artifact = str(candidate) if candidate else str(tuned_csv)

    if candidate and candidate.is_file():
        dest = work_dir / f"candidate_{tuner_name}.csv"
        shutil.copy2(candidate, dest)
        artifact = str(dest)

    # The aiter *tuner* can pick a splitK the production dispatch cannot run (serving it raises "This GEMM is not
    # supported!" and crashes engine init), and some (op, libtype) pairs reach a wrapper that takes no splitK at all,
    # which deploys a config benchmarked with split-K and serves it without. Run unconditionally rather than only when
    # this tuner asked for --splitK: the cap also enforces that forwarding contract, and a caller-supplied CSV can
    # carry splitK>0 on its own. Rows at splitK=0 take a fast path inside the cap, so a table without split-K pays
    # nothing.
    max_splitk = int(os.environ.get("FORGE_MAX_SPLITK", "2"))
    # Prefer the REAL per-shape production split-K limit (trial-dispatch) over the static FORGE_MAX_SPLITK: it
    # keeps splitK>cap where the kernel actually supports it and tightens below cap where it does not. Falls back to
    # the static cap for any op the trial has no registered production callable for -- validating against the wrong
    # kernel is worse than not validating.
    support_fn = None
    if os.environ.get("FORGE_SPLITK_TRIAL", "1") != "0":
        try:
            from ..aiter_splitk_validate import make_support_fn

            # Pin the in-process trial dispatch to the tuner's assigned card; on a shared node the assigned GPU
            # may not be device 0.
            support_fn = make_support_fn(op=script_key, gpu_ids=getattr(ctx, "gpu_ids", "") or "")
        except Exception:  # noqa: BLE001 — fall back to the static cap
            support_fn = None
    n_capped, force_candidate = _cap_splitk_to_serve_safe(
        Path(artifact),
        profile_csv,
        max_splitk,
        support_fn=support_fn,
        forwarding_libtypes=_SPLITK_FORWARDING_LIBTYPES.get(script_key, frozenset()),
    )
    if n_capped:
        log.info(
            "serve-safe splitK cap: rewrote/dropped %d row(s) beyond production support",
            n_capped,
        )

    shape_results = _parse_tuner_stdout(stdout, stderr)
    if not shape_results:
        # aiter writes the --compare table to /tmp/aiter_compare/ when >30 shapes (stdout carries only a "Successfully
        # tuned N shapes" summary).
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
        # Some aiter versions print only a "Successfully tuned shapes" summary (no per-shape Pre/Post table) while
        # still writing a valid tuned candidate CSV.
        candidate_csv_path = work_dir / f"candidate_{tuner_name}.csv"
        fallback_rows = _parse_candidate_csv(candidate_csv_path)
        if fallback_rows:
            shape_results = fallback_rows

    # improved=False carries two different meanings: "compared against a baseline and did not win", and "never had a
    # baseline to compare against".
    dropped_inaccurate = drop_inaccurate_rows(Path(artifact))
    if dropped_inaccurate:
        shape_results = _forget_shapes_that_lost_their_row(shape_results, dropped_inaccurate)

    if any(r.get("is_new") or r.get("tuned_unverified") for r in shape_results):
        force_candidate = True

    # A row that lost its comparison would override a better stock choice once merged, so it is removed from the
    # deployed artifact.
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
    """Stop reporting a speedup for a shape whose winner was just removed."""
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
    """Drop deployed rows for shapes that were compared and lost."""
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
            log.warning("could not rewrite %s after filtering: %s", artifact_csv, exc)
            return 0, len(lines) - 1
    return dropped, len(kept_lines) - 1


def _cap_splitk_to_serve_safe(
    artifact_csv: Path,
    profile_csv: Path,
    max_splitk: int,
    support_fn=None,
    forwarding_libtypes: frozenset[str] | None = None,
) -> tuple[int, bool]:
    """Rewrite deployed rows whose splitK production cannot dispatch or forward.

    Two hazards, same remedy: a splitK the production kernel cannot dispatch
    (crashes engine init), and a splitK on a libtype whose wrapper takes no such
    argument (no crash, just a kernel slower than the one that won the
    benchmark, with every engagement gate still green). Either way the row is
    replaced by the fastest serve-safe candidate from the profile, and a shape
    with no safe candidate is dropped.

    ``forwarding_libtypes`` is the set of libtypes whose wrapper forwards splitK
    for this op (see ``_SPLITK_FORWARDING_LIBTYPES``); a row on any other
    libtype is capped at 0 regardless of dispatch support, and ``None`` disables
    the check.
    """
    try:
        with artifact_csv.open() as f:
            rows = list(csv.reader(f))
    except OSError:
        return 0, False
    if len(rows) < 2:
        return 0, False
    hdr = rows[0]
    # Case-insensitive column lookup: if the deployed-header case ever fails an exact match the cap would return early
    # (0, False) and pass unsafe splitK rows through unchanged -> serve crash.
    _col = {str(h).strip().lower(): i for i, h in enumerate(hdr)}
    try:
        mi, ni, ki, ski = (_col[c] for c in ("m", "n", "k", "splitk"))
    except KeyError:
        return 0, False

    # Index every valid candidate per shape; the cap is applied per-shape at selection so support_fn can keep
    # splitK>max_splitk where the kernel supports.
    by_shape: dict[tuple[str, str, str], list[tuple[float, int, list[str]]]] = defaultdict(list)
    schema_ok = True
    try:
        with profile_csv.open() as f:
            for r in csv.DictReader(f):
                # A candidate must carry every column the deployed CSV has, or the rewritten row would get empty cells
                # and a renamed/absent errRatio would silently disable the correctness filter.
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

    lti = _col.get("libtype")

    def _forwards(row: list[str]) -> bool:
        if forwarding_libtypes is None:
            return True
        if lti is None or lti >= len(row):
            # No libtype column to check against a contract that is keyed on it;
            # treat as non-forwarding, matching the fail-closed default.
            return False
        return str(row[lti]).strip().lower() in forwarding_libtypes

    def _shape_max(m: int, n: int, k: int) -> int:
        if support_fn is None:
            return max_splitk
        try:
            v = support_fn(m, n, k)
        except Exception:  # noqa: BLE001 — trial failure must not abort the cap
            return max_splitk  # degrade to the static cap, never crash the tuner
        return max_splitk if v is None else int(v)

    out, changed, has_splitk = [hdr], 0, False
    dropped_unforwarded = 0
    for row in rows[1:]:
        try:
            sk = int(row[ski])
        except (ValueError, IndexError):
            out.append(row)
            continue
        if sk == 0:
            # splitK=0 is the default dispatch: always serve-safe, and its keep decision never depends on the
            # per-shape max, so skip the (GPU-dispatching) trial entirely for these rows.
            out.append(row)
            continue
        if not _forwards(row):
            # The wrapper for this libtype takes no splitK, so the only
            # honourable value is 0 -- fall through to candidate replacement
            # with maxsk=0 rather than shipping a measurement production cannot
            # reproduce.
            maxsk = 0
            dropped_unforwarded += 1
            try:
                key = (row[mi], row[ni], row[ki])
            except IndexError:
                out.append(row)
                continue
            safe = min(
                (c for c in by_shape.get(key, ()) if c[1] <= maxsk),
                key=lambda c: c[0],
                default=None,
            )
            if safe is not None:
                out.append(safe[2])
            changed += 1
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
    if dropped_unforwarded:
        log.warning(
            "splitK forwarding: %d row(s) carried splitK>0 on a libtype whose "
            "production wrapper takes no splitK (forwarding set: %s); replaced "
            "with a splitK=0 candidate or dropped, because serving them would "
            "have run a config slower than the one benchmarked",
            dropped_unforwarded,
            sorted(forwarding_libtypes or ()),
        )
    return changed, has_splitk


def _stem_matches(tuner_name: str, filename: str) -> bool:
    """Whether ``filename`` is a candidate CSV produced for ``tuner_name``."""
    stem = f"tuned_{tuner_name}"
    return re.search(re.escape(stem) + r"(?:\.|_\d)", filename) is not None


def _find_latest_compare_report_impl(
    tuner_name: str,
    start_time: float,
    compare_dir: Path,
) -> Path | None:
    """Find the most recent compare report from ``compare_dir`` for THIS run."""
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
    """Find the most recent candidate CSV from /tmp/aiter_compare/ for THIS run."""
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
