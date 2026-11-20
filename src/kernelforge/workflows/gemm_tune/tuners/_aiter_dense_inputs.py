# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Input CSV preparation for the aiter dense GEMM tuners: demand, manifest, CSV, JSON or config-derived shapes."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ._decode_groups import decode_served_groups, dispatch_groups
from .base import TuneContext

log = logging.getLogger(__name__)


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


_DEMAND_MAX_SHAPES_ENV = "FORGE_DEMAND_MAX_SHAPES"


def _demand_budget(ctx: TuneContext) -> int | None:
    """The operator's cap on the shapes to tune, decode band included; without one the deadline decides, chunk by chunk,
    and the decode band is kept whole by the chunking instead (``_chunks.decode_band_units``)."""
    raw = os.environ.get(_DEMAND_MAX_SHAPES_ENV, "").strip()
    try:
        override = int(raw)
    except ValueError:
        override = 0
    return override if override > 0 else None


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
    decode_m = compute_decode_m_values(ctx.conc, ctx.m_grid)
    served = decode_served_groups(entry, decode_m)
    shapes = ranked if budget is None else _demand_shapes_within_budget(ranked, decode_m, budget, served=served)

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
        "the shape cap (%s) -> %s",
        tuner_name,
        len(shapes),
        len(ranked),
        entry.get("distinct_keys", 0),
        budget or "none",
        out,
    )
    return _ensure_decode_m_coverage(out, ctx, work_dir, needs_q_dtype_w=needs_q_dtype_w, served=served)


def _demand_shapes_within_budget(
    ranked: list[dict[str, Any]],
    decode_m: Sequence[int],
    budget: int,
    *,
    served: frozenset[tuple[int, int]],
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
        unserved = [ms for group, ms in trial.items() if group not in served]
        band = sum(len(b) for b in _decode_band_buckets(unserved, decode_m))
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


def _decode_band_rows(
    header: list[str],
    idx: dict[str, int],
    q_idx: int | None,
    group_order: list[tuple[int, int, str]],
    plan: list[list[int]],
) -> tuple[list[str], int]:
    """The CSV rows that fill each group's missing decode-band M, and how many groups needed any."""
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
    return additions, uncovered


def _ensure_decode_m_coverage(
    csv: Path,
    ctx: TuneContext,
    work_dir: Path,
    needs_q_dtype_w: bool = False,
    served: frozenset[tuple[int, int]] = frozenset(),
) -> Path:
    """Guarantee every tuned dispatch group covers the decode-band M, except the ``served`` (N, K) groups."""
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
    group_order, group_m, body = dispatch_groups(lines[1:], idx, q_idx)
    if not group_order:
        return csv

    decode_m = compute_decode_m_values(ctx.conc, ctx.m_grid)
    plan = [
        [] if key[:2] in served else buckets
        for key, buckets in zip(group_order, _decode_band_buckets([group_m[key] for key in group_order], decode_m))
    ]
    additions, uncovered = _decode_band_rows(header, idx, q_idx, group_order, plan)

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
    num_kv_heads = int(getattr(profile, "num_key_value_heads", 0) or num_heads)
    nk_shapes = compute_dense_nk_shapes(
        DenseGeometry(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=int(getattr(profile, "head_dim", 0) or 0),
            v_head_dim=int(getattr(profile, "v_head_dim", 0) or 0),
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
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
