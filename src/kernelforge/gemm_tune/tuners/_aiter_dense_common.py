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
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import TuneContext, TuneResult, micro_metrics
from ..aiter_graph_retime import RetimeTarget
from ._graph_retime import retime_table
from ._row_accuracy import RelNormFn, drop_inaccurate_rows, dropped_row_record
from ._serve_safe import _SPLITK_FORWARDING_LIBTYPES, _fastest_serve_safe, _serve_safe_candidates, _SplitKLimit
from ..script_probe import filter_args, probe_script
from ..utils import find_tuner_script, resolve_aiter_root, run_subprocess
from .. import tune_robustness as _tr

log = logging.getLogger(__name__)

# The a8w8_blockscale tuner's script_key, which must match the op key
# aiter_splitk_validate registers its trial under; a mismatch silently falls
# back to the static cap.
SPLITK_TRIAL_SCRIPT_KEY = "a8w8_blockscale"


class AiterDtypeUnavailable(RuntimeError):
    """The installed aiter cannot supply a dtype the tuner CSV needs."""


def _aiter_dtype_str(attr: str) -> str:
    """Return the repr string aiter's tuner scripts accept for ``dtypes.<attr>``."""
    try:
        from aiter import dtype2str_dict, dtypes  # type: ignore[import-untyped]
    except Exception as exc:
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
    from ..dense_shapes import MGrid, compute_decode_m_values
    from ..evidence import demand_for_tuner, demand_shapes, load_demand

    report = load_demand(path)
    if report is None:
        return None
    entry = demand_for_tuner(report, tuner_name)
    if entry is None:
        return None
    ranked = demand_shapes(entry, bucket=ctx.m_grid is MGrid.BUCKET)
    if not ranked:
        return None
    budget = _demand_budget(ctx)
    shapes = _demand_shapes_within_budget(ranked, compute_decode_m_values(ctx.conc, ctx.m_grid), budget)

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
        band = sum(len(b) for b in _decode_band_buckets(list(trial.values()), decode_m))
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


def _decode_band_buckets(groups: Sequence[set[int]], decode_m: Sequence[int]) -> list[list[int]]:
    """Per group, the decode-band M it has no row for.

    ``groups`` is the tuned M per dispatch group. ``decode_m`` already
    holds the M a row is tuned at, so a group needs exactly those rows: a row
    reached only through a later aiter lookup serves the batch with another
    batch size's kernel. Sizing the reservation and appending the rows both
    read this, so the two cannot drift.
    """
    return [[m for m in decode_m if m not in tuned] for tuned in groups]


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

    decode_m = compute_decode_m_values(ctx.conc, ctx.m_grid)
    plan = _decode_band_buckets([group_m[key] for key in group_order], decode_m)
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


def _augment_with_config_m_values(csv: Path, ctx: TuneContext, work_dir: Path, needs_q_dtype_w: bool) -> Path:
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
            if m_idx is not None:
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

    q_dtype = f",{_aiter_fp8_dtype_str()}" if needs_q_dtype_w else ""
    out = work_dir / "augmented_dense.csv"
    with out.open("w", encoding="utf-8") as f:
        f.write("M,N,K,q_dtype_w\n" if needs_q_dtype_w else "M,N,K\n")
        for m in all_m:
            for n, k in sorted(profile_nk):
                f.write(f"{m},{n},{k}{q_dtype}\n")

    log.info(
        "Augmented shapes: %d M values × %d NK pairs = %d shapes (profile had %d M values)",
        len(all_m),
        len(profile_nk),
        len(all_m) * len(profile_nk),
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
        DenseGeometry,
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
        DenseGeometry(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=int(getattr(profile, "head_dim", 0) or 0),
            v_head_dim=int(getattr(profile, "v_head_dim", 0) or 0),
            q_lora_rank=int(getattr(profile, "q_lora_rank", 0) or 0),
            kv_lora_rank=int(getattr(profile, "kv_lora_rank", 0) or 0),
            qk_nope_head_dim=int(getattr(profile, "qk_nope_head_dim", 0) or 0),
            qk_rope_head_dim=int(getattr(profile, "qk_rope_head_dim", 0) or 0),
        ),
        ctx.tp,
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
            results.append(
                {
                    "M": int(t.group(1)),
                    "N": int(t.group(2)),
                    "K": int(t.group(3)),
                    "default_us": float(pre_tok),
                    "tuned_us": post,
                    "speedup": round(float(pre_tok) / post, 4) if post > 0 else 1.0,
                    "improved": action == "UPDATE" or in_would_update,
                    "below_update_bar": action == "<",  # "< 3.0% improve"; not post-mismatch or post-error
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
            "n_improved": None,
            "n_unverified": 0,
            "best": None,
            "avg": None,
        }
    improved = [r for r in shape_results if r.get("improved")]
    # Tuned, but with nothing to compare against (new shape, or the candidate-CSV fallback).
    unverified = [r for r in shape_results if r.get("is_new") or r.get("tuned_unverified")]
    metrics = micro_metrics(shape_results)
    return {
        "status": "ok" if (improved or unverified) else "no_improvement",
        "total": total,
        "n_improved": metrics.improved,
        "n_unverified": len(unverified),
        "best": metrics.best,
        "avg": metrics.avg,
    }


def _invoke_tuner(
    invocation: _tr.TunerInvocation,
    ctx: TuneContext,
    tuner_name: str,
) -> tuple[int, str, str, Path | None]:
    """Run the aiter tuner, per shape under isolation or once for the whole file.

    Either way ``--timeout`` is injected so a faulting candidate is isolated by
    aiter instead of hanging the run.
    """
    if not _tr.is_isolation_enabled():
        rc, stdout, stderr = run_subprocess(
            invocation.command(),
            cwd=invocation.aiter_root,
            timeout_s=ctx.timeout_s,
            log_file=invocation.work_dir / "tune.log",
        )
        return rc, stdout, stderr, None
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
    return _tr.run_isolated(
        invocation,
        _tr.IsolationGuard(
            outer_timeout_s=ctx.timeout_s,
            gpu_ids=_gpu_ids(ctx),
            blocklist=blocklist,
        ),
    )


def _gpu_ids(ctx: TuneContext) -> str:
    """The cards a trial may run on; a context built without them leaves the choice to the trial."""
    return getattr(ctx, "gpu_ids", "") or ""


def _splitk_support_fn(script_key: str, ctx: TuneContext):
    """The per-shape production split-K limit for this op, or None to use the static cap.

    The trial dispatch keeps splitK above ``FORGE_MAX_SPLITK`` where the kernel
    actually supports it and tightens below it where it does not. An op with no
    registered production callable falls back to the static cap -- validating
    against the wrong kernel is worse than not validating.
    """
    if os.environ.get("FORGE_SPLITK_TRIAL", "1") == "0":
        return None
    from ..aiter_splitk_validate import make_support_fn

    # Pin the in-process trial dispatch to the tuner's assigned card; on a shared node the assigned GPU
    # may not be device 0.
    return make_support_fn(op=script_key, gpu_ids=_gpu_ids(ctx))


def _splitk_rel_norm_fn(script_key: str, ctx: TuneContext) -> RelNormFn | None:
    """The relative-norm check for this op's asm split-K rows, or None to let errRatio decide them.

    It dispatches in process like the split-K trial, so the same switch turns both off.
    """
    if os.environ.get("FORGE_SPLITK_TRIAL", "1") == "0":
        return None
    from ..aiter_splitk_validate import make_rel_norm_fn

    return make_rel_norm_fn(op=script_key, gpu_ids=_gpu_ids(ctx))


def _serve_safe_limit(script_key: str, ctx: TuneContext) -> _SplitKLimit:
    """The split-K each shape may ship and the accuracy check a replacement passes, for cap, filter and re-time."""
    support_fn, rel_norm = _splitk_support_fn(script_key, ctx), _splitk_rel_norm_fn(script_key, ctx)
    return _SplitKLimit(int(os.environ.get("FORGE_MAX_SPLITK", "2")), support_fn, rel_norm)


def _apply_splitk_cap(artifact: str, profile_csv: Path, limit: _SplitKLimit, *, script_key: str) -> bool:
    """Hold the deployed table to split-K values production can serve; say whether it changed.

    The aiter *tuner* can pick a splitK the production dispatch cannot run
    (serving it raises "This GEMM is not supported!" and crashes engine init),
    and some (op, libtype) pairs reach a wrapper that takes no splitK at all,
    which deploys a config benchmarked with split-K and serves it without. Run
    unconditionally rather than only when this tuner asked for ``--splitK``: the
    cap also enforces that forwarding contract, and a caller-supplied CSV can
    carry splitK>0 on its own. Rows at splitK=0 take a fast path inside the cap,
    so a table without split-K pays nothing. ``limit.rel_norm`` is the check
    :func:`drop_inaccurate_rows` applies afterwards, so a replacement is one
    that survives it.
    """
    n_capped, force_candidate = _cap_splitk_to_serve_safe(
        Path(artifact),
        profile_csv,
        limit,
        forwarding_libtypes=_SPLITK_FORWARDING_LIBTYPES.get(script_key, frozenset()),
    )
    if n_capped:
        log.info("serve-safe splitK cap: rewrote/dropped %d row(s) beyond production support", n_capped)
    return force_candidate


def _shape_results_from_compare_report(tuner_name: str, run_start_time: float, work_dir: Path) -> list[dict]:
    """Per-shape rows from the compare table aiter writes to disk instead of stdout.

    aiter writes the ``--compare`` table to ``/tmp/aiter_compare/`` when there
    are more than 30 shapes; stdout then carries only a "Successfully tuned N
    shapes" summary.
    """
    compare_report = _find_latest_compare_report(tuner_name, run_start_time)
    if compare_report is None or not compare_report.is_file():
        log.info("no compare report for %s under /tmp/aiter_compare after run start", tuner_name)
        return []
    log.info("compare report found for %s: %s", tuner_name, compare_report)
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
        return _parse_tuner_stdout(compare_report.read_text(encoding="utf-8", errors="replace"), "")
    except OSError:
        return []


def _shape_results_for_run(
    stdout: str,
    stderr: str,
    *,
    tuner_name: str,
    run_start_time: float,
    work_dir: Path,
) -> list[dict]:
    """What this run tuned per shape, from stdout or from whichever artifact still says.

    Some aiter versions print only a "Successfully tuned shapes" summary (no
    per-shape Pre/Post table) while still writing a valid tuned candidate CSV.
    """
    shape_results = _parse_tuner_stdout(stdout, stderr)
    if not shape_results:
        shape_results = _shape_results_from_compare_report(tuner_name, run_start_time, work_dir)
    if not shape_results:
        shape_results = _parse_candidate_csv(work_dir / f"candidate_{tuner_name}.csv")
    return shape_results


@dataclass(frozen=True)
class DenseTunerSpec:
    """One aiter dense tuner: its name, tuning script, deployment env var and extra script flags."""

    tuner_name: str
    script_key: str
    env_var: str
    extra_args: tuple[str, ...] = ()


def run_aiter_dense_tuner(spec: DenseTunerSpec, ctx: TuneContext, work_dir: Path) -> TuneResult:
    """Run an aiter dense GEMM tuner subprocess."""
    tuner_name, script_key = spec.tuner_name, spec.script_key
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
    if spec.extra_args:
        base_args.extend(spec.extra_args)

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
    run_start_time = time.time()
    deadline = run_start_time + ctx.timeout_s

    rc, stdout, stderr, iso_candidate = _invoke_tuner(
        _tr.TunerInvocation(script, base_args, input_csv, tuned_csv, work_dir, aiter_root),
        ctx,
        tuner_name,
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

    candidate = iso_candidate if iso_candidate is not None else _find_latest_candidate(tuner_name, run_start_time)
    artifact = str(candidate) if candidate else str(tuned_csv)

    if candidate and candidate.is_file():
        dest = work_dir / f"candidate_{tuner_name}.csv"
        shutil.copy2(candidate, dest)
        artifact = str(dest)

    limit = _serve_safe_limit(script_key, ctx)
    force_candidate = _apply_splitk_cap(artifact, profile_csv, limit, script_key=script_key)

    shape_results = _shape_results_for_run(
        stdout,
        stderr,
        tuner_name=tuner_name,
        run_start_time=run_start_time,
        work_dir=work_dir,
    )

    # improved=False carries two different meanings: "compared against a baseline and did not win", and "never had a
    # baseline to compare against".
    accuracy = drop_inaccurate_rows(Path(artifact), limit.rel_norm)
    if not accuracy.completed:
        # An unfiltered table is a table whose wrong rows are still in it, and this artifact is what gets deployed.
        return TuneResult(
            tuner_name=tuner_name,
            status="failed",
            error=f"aiter's accuracy figures were not applied to {artifact}: {accuracy.reason}",
            error_class="accuracy_filter_incomplete",
        )
    if accuracy.dropped:
        shape_results = _forget_shapes_that_lost_their_row(shape_results, accuracy.dropped)

    if any(r.get("is_new") or r.get("tuned_unverified") for r in shape_results):
        force_candidate = True

    # A row that lost its comparison would override a better stock choice once merged, so it is removed from the
    # deployed artifact.
    n_dropped, n_kept = _filter_unimproved_rows(Path(artifact), shape_results)
    if n_dropped:
        log.info("deployed artifact: dropped %d row(s) that lost their comparison, %d kept", n_dropped, n_kept)

    summary = _summarize_shape_results(shape_results)

    return TuneResult(
        tuner_name=tuner_name,
        status=summary["status"],
        artifact_path=artifact,
        env_var=spec.env_var,
        env_value=artifact,
        total_shapes=summary["total"],
        improved_shapes=summary["n_improved"],
        unverified_shapes=summary["n_unverified"],
        best_micro_speedup=summary["best"],
        avg_micro_speedup=summary["avg"],
        candidate=force_candidate,
        shape_results=shape_results,
        dropped_inaccurate=[dropped_row_record(r) for r in accuracy.dropped],
        graph_retime=retime_table(artifact, profile_csv, limit, RetimeTarget(script_key, _gpu_ids(ctx), deadline)),
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
    """Drop rows that lost to the pre-run config; unmeasured rows, and faster ones below aiter's update bar, stay."""
    losers: set[tuple[int, int, int]] = set()
    for r in shape_results:
        if r.get("improved"):
            continue
        if r.get("is_new") or r.get("tuned_unverified") or r.get("below_update_bar") and r["speedup"] > 1.0:
            continue
        if r.get("speedup") is None and r.get("default_us") is None:
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


@dataclass(frozen=True)
class _DeployedColumns:
    """Where the deployed CSV keeps the columns the splitK cap reads."""

    m: int
    n: int
    k: int
    splitk: int
    libtype: int | None

    def shape_key(self, row: list[str]) -> tuple[str, str, str]:
        """The ``(M, N, K)`` this row describes, as the profile index is keyed."""
        return row[self.m], row[self.n], row[self.k]


def _deployed_columns(header: list[str]) -> _DeployedColumns | None:
    """Locate the columns the cap reads, or None when the header lacks one.

    Case-insensitive: were the deployed header's case ever to stop matching
    exactly, the cap would return early and pass unsafe splitK rows through
    unchanged -> serve crash.
    """
    index = {str(name).strip().lower(): position for position, name in enumerate(header)}
    try:
        return _DeployedColumns(index["m"], index["n"], index["k"], index["splitk"], index.get("libtype"))
    except KeyError:
        return None


def _forwards_splitk(row: list[str], libtype: int | None, forwarding_libtypes: frozenset[str]) -> bool:
    """Whether this row's libtype has a production wrapper that forwards splitK."""
    if libtype is None or libtype >= len(row):
        # No libtype column to check against a contract that is keyed on it;
        # treat as non-forwarding, matching the fail-closed default.
        return False
    return str(row[libtype]).strip().lower() in forwarding_libtypes


def _cap_splitk_to_serve_safe(
    artifact_csv: Path,
    profile_csv: Path,
    limit: _SplitKLimit,
    *,
    forwarding_libtypes: frozenset[str],
) -> tuple[int, bool]:
    """Rewrite deployed rows whose splitK production cannot dispatch or forward.

    Two hazards, same remedy: a splitK the production kernel cannot dispatch
    (crashes engine init), and a splitK on a libtype whose wrapper takes no such
    argument (no crash, just a kernel slower than the one that won the
    benchmark, with every engagement gate still green). Either way the row is
    replaced by the fastest serve-safe candidate from the profile that would
    pass the accuracy gate, and a shape with no such candidate is dropped.

    ``forwarding_libtypes`` is the set of libtypes whose wrapper forwards splitK
    for this op (see ``_SPLITK_FORWARDING_LIBTYPES``); a row on any other
    libtype is capped at 0 regardless of dispatch support.
    """
    try:
        with artifact_csv.open() as handle:
            rows = list(csv.reader(handle))
    except OSError:
        return 0, False
    if len(rows) < 2:
        return 0, False
    hdr = rows[0]
    columns = _deployed_columns(hdr)
    if columns is None:
        return 0, False
    by_shape = _serve_safe_candidates(profile_csv, hdr)

    out, changed, has_splitk = [hdr], 0, False
    dropped_unforwarded = 0
    for row in rows[1:]:
        try:
            sk = int(row[columns.splitk])
        except (ValueError, IndexError):
            out.append(row)
            continue
        if sk == 0:
            # splitK=0 is the default dispatch: always serve-safe, and its keep decision never depends on the
            # per-shape max, so skip the (GPU-dispatching) trial entirely for these rows.
            out.append(row)
            continue
        if not _forwards_splitk(row, columns.libtype, forwarding_libtypes):
            # The wrapper for this libtype takes no splitK, so the only
            # honourable value is 0 -- fall through to candidate replacement
            # with maxsk=0 rather than shipping a measurement production cannot
            # reproduce.
            dropped_unforwarded += 1
            try:
                key = columns.shape_key(row)
            except IndexError:
                out.append(row)
                continue
            safe = _fastest_serve_safe(by_shape.get(key, ()), 0, limit)
            if safe is not None:
                out.append(safe.row)
            changed += 1
            continue
        try:
            key = columns.shape_key(row)
            shape = (int(row[columns.m]), int(row[columns.n]), int(row[columns.k]))
        except (ValueError, IndexError):
            out.append(row)
            continue
        maxsk = limit.ceiling(shape)
        if sk <= maxsk:
            out.append(row)
            has_splitk = has_splitk or sk > 0
            continue
        safe = _fastest_serve_safe(by_shape.get(key, ()), maxsk, limit)
        if safe is not None:
            out.append(safe.row)
            has_splitk = has_splitk or safe.splitk > 0
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
            sorted(forwarding_libtypes),
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
