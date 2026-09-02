# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""No-run uplift estimates mined from Recipe KB session documents.

Deciding whether a target is worth a session meant running one. This reads
per-session envelopes instead (``GET /v1/kb/{id}/sessions/{sid}``, or an
offline JSON list) and reports what prior sessions already settled: the
distribution of ``validated_e2e_gain``, and which parallelism layout won
inside a fixed GPU count.

It reads only what the Recipe KB already owns as replay material --
``validated_e2e_gain``, ``workload_shape``, and the accepted
``value.config.extra_server_args`` a layout is read out of. Execution
evidence such as roofline arms, host platform, or token spend belongs to the
session-evidence pipeline rather than to a replay record, so a
remaining-headroom forecast is deliberately not attempted here.

Scoping is the whole difficulty. Pooling every session a search returns makes
the median meaningless, so rows are grouped by identity and by the full
``tp/conc/isl/osl`` replay scope, and ``pool_warnings`` names any pool that
spans models, boards, frameworks, versions, or precisions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ._vendor.kb_store_client import KBStoreClient, KBStoreError

_SEARCH_PAGE = 50
_SEARCH_PAGE_CAP = 20

#: Order of the seven identity dimensions inside an ``inference:`` canonical id.
_IDENTITY_DIMENSIONS = (
    "model",
    "hardware",
    "framework_name",
    "model_type",
    "architectures",
    "framework_version",
    "precision",
)
#: Replay-sensitive workload dimensions; RecipeScope demands an exact match on
#: all four, so pooling gains across them is a prior, never a replayable claim.
_SHAPE_KEYS = ("tp", "conc", "isl", "osl")

#: Server-arg flags that re-shard the model, normalized across frameworks so a
#: vLLM ``--tensor-parallel-size`` and an SGLang ``--tp-size`` compare as ``tp``.
#: ``workload_shape.tp`` is the GPU count; these decide the layout inside it.
_PARALLELISM_VALUE_FLAGS = {
    "tp-size": "tp",
    "tensor-parallel-size": "tp",
    "dp-size": "dp",
    "data-parallel-size": "dp",
    "ep-size": "ep",
    "expert-parallel-size": "ep",
    "pp-size": "pp",
    "pipeline-parallel-size": "pp",
    "moe-dense-tp-size": "moe_dense_tp",
    "decode-tp-size": "decode_tp",
    "prefill-tp-size": "prefill_tp",
}
#: Boolean parallelism switches; presence alone changes the layout.
_PARALLELISM_BOOL_FLAGS = {
    "enable-dp-attention": "dp_attention",
    "enable-dp-lm-head": "dp_lm_head",
    "enable-expert-parallel": "expert_parallel",
    "enable-ep-moe": "ep_moe",
    "enable-deepep-moe": "deepep_moe",
}
#: Label used when a session accepted no explicit parallelism flag.
DEFAULT_PARALLELISM_LABEL = "framework-default"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def identity_dimensions(canonical_id: str) -> dict[str, str]:
    """Split an ``inference:`` canonical id into its identity dimensions."""
    parts = str(canonical_id or "").split(":")
    if len(parts) != len(_IDENTITY_DIMENSIONS) + 1:
        return {}
    return dict(zip(_IDENTITY_DIMENSIONS, (part.strip() for part in parts[1:])))


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def percentile(values: list[float], p: float) -> float | None:
    """Linear interpolation percentile; ``None`` when ``values`` is empty."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (max(0.0, min(100.0, p)) / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def project_session(document: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one session envelope into miner fields."""
    knowledge = _mapping(document.get("knowledge")) or (
        dict(document) if document.get("record_kind") or document.get("knowledge_schema_version") else {}
    )
    value = _mapping(knowledge.get("value"))
    gain = _finite(knowledge.get("validated_e2e_gain"))
    canonical_id = str(document.get("canonical_id") or knowledge.get("canonical_id") or "")
    dims = identity_dimensions(canonical_id)
    shape_source = _mapping(knowledge.get("workload_shape"))
    optimized_throughput = _finite(knowledge.get("optimized_throughput"))
    tp = _positive_int(shape_source.get("tp"))
    row = {
        "canonical_id": canonical_id,
        "session_id": str(document.get("session_id") or ""),
        "model": dims.get("model", ""),
        "hardware": dims.get("hardware", ""),
        "framework_name": dims.get("framework_name", ""),
        "framework_version": dims.get("framework_version", ""),
        "precision": dims.get("precision", ""),
        "optimized_throughput": optimized_throughput,
        "validated_e2e_gain": gain,
    }
    for key in _SHAPE_KEYS:
        row[key] = _positive_int(shape_source.get(key))
    accepted_args = str(_mapping(value.get("config")).get("extra_server_args") or "")
    knobs = extract_parallelism(accepted_args)
    row["parallelism"] = knobs
    row["parallelism_label"] = parallelism_label(knobs)
    # Per-GPU throughput is the only cross-TP comparable figure; total tok/s
    # rises with TP even when parallel efficiency is falling.
    row["tput_per_gpu"] = (
        optimized_throughput / tp if optimized_throughput is not None and optimized_throughput > 0 and tp else None
    )
    return row


def shape_key(row: Mapping[str, Any]) -> str:
    """Render a row's replay scope as ``tp8/conc64/isl1024/osl256``."""
    parts = []
    for key in _SHAPE_KEYS:
        value = row.get(key)
        parts.append(f"{key}{value}" if value else f"{key}?")
    return "/".join(parts)


def matches_shape(row: Mapping[str, Any], shape: Mapping[str, Any] | None) -> bool:
    """True when a row satisfies every requested workload dimension."""
    if not shape:
        return True
    for key in _SHAPE_KEYS:
        wanted = _positive_int(shape.get(key))
        if wanted is None:
            continue
        if _positive_int(row.get(key)) != wanted:
            return False
    return True


def _bucket_stats(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    gains = [row["validated_e2e_gain"] for row in rows if row.get("validated_e2e_gain") is not None]
    tputs = [row["optimized_throughput"] for row in rows if row.get("optimized_throughput") is not None]
    per_gpu = [row["tput_per_gpu"] for row in rows if row.get("tput_per_gpu") is not None]
    return {
        "sessions": len(rows),
        "p50_validated_e2e_gain_pct": percentile(gains, 50),
        "p50_optimized_throughput": percentile(tputs, 50),
        "best_optimized_throughput": max(tputs) if tputs else None,
        "p50_tput_per_gpu": percentile(per_gpu, 50),
    }


def extract_parallelism(server_args: str) -> dict[str, str]:
    """Pull normalized parallelism knobs out of an accepted server-arg string.

    Handles both ``--flag value`` and ``--flag=value`` spellings and maps
    framework-specific names onto shared keys, so ``--tp-size 1 --dp-size 2``
    and ``--tensor-parallel-size 1 --data-parallel-size 2`` both read as
    ``{"tp": "1", "dp": "2"}``.
    """
    tokens = str(server_args or "").split()
    knobs: dict[str, str] = {}
    for index, token in enumerate(tokens):
        if not token.startswith("--"):
            continue
        name, _, inline = token[2:].partition("=")
        name = name.strip().lower()
        if name in _PARALLELISM_BOOL_FLAGS:
            knobs[_PARALLELISM_BOOL_FLAGS[name]] = "on"
            continue
        canonical = _PARALLELISM_VALUE_FLAGS.get(name)
        if canonical is None:
            continue
        value = inline.strip() if inline else ""
        if not value and index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            value = tokens[index + 1].strip()
        if value:
            knobs[canonical] = value
    return knobs


def parallelism_label(knobs: Mapping[str, str]) -> str:
    """Render parallelism knobs as a stable, sortable label."""
    if not knobs:
        return DEFAULT_PARALLELISM_LABEL
    return " ".join(f"{key}={knobs[key]}" for key in sorted(knobs))


def workload_family(row: Mapping[str, Any]) -> str:
    """Everything a TP comparison must hold fixed, e.g.
    ``qwen3-32b/bf16/conc64/isl8192/osl1024``.

    TP is only comparable inside one family. Two confounds otherwise dominate
    and both were observed in live data: ISL changes arithmetic intensity, and
    model size dictates the TP that fits at all, so a 0.6B at TP1 against a
    70B at TP8 reads as catastrophic scaling when nothing scaled.
    """
    parts = [row.get("model") or "model?", row.get("precision") or "precision?"]
    for key in _SHAPE_KEYS:
        if key == "tp":
            continue
        value = row.get(key)
        parts.append(f"{key}{value}" if value else f"{key}?")
    return "/".join(str(part) for part in parts)


def _family_whatif(rows: list[Mapping[str, Any]], target_tp: int | None) -> dict[str, Any]:
    by_tp: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        tp = _positive_int(row.get("tp"))
        if tp is not None:
            by_tp.setdefault(tp, []).append(row)
    buckets = {str(tp): _bucket_stats(items) for tp, items in sorted(by_tp.items())}
    report: dict[str, Any] = {"observed_tp": sorted(by_tp), "by_tp": buckets}
    comparable = {
        tp: buckets[str(tp)]["p50_tput_per_gpu"]
        for tp in sorted(by_tp)
        if buckets[str(tp)]["p50_tput_per_gpu"] is not None
    }
    efficiency: float | None = None
    if len(comparable) >= 2:
        lo_tp = min(comparable)
        hi_tp = max(comparable)
        if comparable[lo_tp] > 0:
            efficiency = comparable[hi_tp] / comparable[lo_tp]
            report["measured_scaling"] = {
                "from_tp": lo_tp,
                "to_tp": hi_tp,
                "per_gpu_retention": efficiency,
                "reading": "1.0 = flat per-GPU throughput; below 1.0 means each added GPU returns less",
            }
    if target_tp is None:
        return report
    if not comparable:
        report["projection"] = {"target_tp": target_tp, "reason": "no per-GPU baseline in this family"}
        return report
    if target_tp in comparable:
        report["projection"] = {
            "target_tp": target_tp,
            "source": "observed",
            "sessions": buckets[str(target_tp)]["sessions"],
            "p50_optimized_throughput": buckets[str(target_tp)]["p50_optimized_throughput"],
            "p50_validated_e2e_gain_pct": buckets[str(target_tp)]["p50_validated_e2e_gain_pct"],
        }
        return report
    source_tp = min(comparable, key=lambda tp: (abs(tp - target_tp), tp))
    per_gpu = comparable[source_tp]
    ideal = per_gpu * target_tp
    adjusted = ideal * efficiency if efficiency is not None and target_tp > source_tp else ideal
    report["projection"] = {
        "target_tp": target_tp,
        "source": "scaled",
        "source_tp": source_tp,
        "source_p50_tput_per_gpu": per_gpu,
        "ideal_flat_per_gpu": ideal,
        "efficiency_adjusted": adjusted,
        "range": sorted({round(min(ideal, adjusted), 4), round(max(ideal, adjusted), 4)}),
        "assumption": "flat per-GPU is the optimistic bound; adjusted applies the measured retention",
    }
    return report


def replay_scope_key(row: Mapping[str, Any]) -> str:
    """Everything a sharding comparison must hold fixed, GPU count included.

    Comparing layouts only makes sense at a fixed world size: ``tp=1 dp=2`` on
    two GPUs against ``tp=2`` on two GPUs is a real choice, while the same
    label on eight GPUs is a different experiment.
    """
    return "/".join(
        (
            str(row.get("model") or "model?"),
            str(row.get("precision") or "precision?"),
            str(row.get("framework_name") or "framework?"),
            shape_key(row),
        )
    )


def sharding_whatif(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Rank the parallelism layouts prior runs accepted, at fixed world size.

    This is the parallelism what-if the store can actually answer today:
    ``workload_shape.tp`` fixes the GPU count, and the accepted server args
    show how the model was sharded inside it. A record keeps only the layout
    its session settled on, so an absent layout was never tried or never
    published rather than tried and beaten -- which is what ``winners_only``
    says, and why these rank rather than score.
    """
    scopes: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    knobs_seen: dict[str, int] = {}
    configured = 0
    for row in rows:
        knobs = _mapping(row.get("parallelism"))
        if knobs:
            configured += 1
            for key in knobs:
                knobs_seen[key] = knobs_seen.get(key, 0) + 1
        label = str(row.get("parallelism_label") or DEFAULT_PARALLELISM_LABEL)
        scopes.setdefault(replay_scope_key(row), {}).setdefault(label, []).append(row)

    compared: dict[str, Any] = {}
    single = 0
    for scope, by_label in sorted(scopes.items()):
        if len(by_label) < 2:
            single += 1
            continue
        strategies = {}
        for label, items in by_label.items():
            stats = _bucket_stats(items)
            strategies[label] = {
                "sessions": stats["sessions"],
                "p50_validated_e2e_gain_pct": stats["p50_validated_e2e_gain_pct"],
                "best_validated_e2e_gain_pct": max(
                    (row["validated_e2e_gain"] for row in items if row.get("validated_e2e_gain") is not None),
                    default=None,
                ),
                "p50_optimized_throughput": stats["p50_optimized_throughput"],
            }
        ranked = sorted(
            (label for label, s in strategies.items() if s["p50_validated_e2e_gain_pct"] is not None),
            key=lambda label: -strategies[label]["p50_validated_e2e_gain_pct"],
        )
        entry: dict[str, Any] = {"strategies": strategies, "ranked_by_p50_gain": ranked}
        if ranked:
            best = ranked[0]
            entry["best"] = {"label": best, **strategies[best]}
            baseline = strategies.get(DEFAULT_PARALLELISM_LABEL)
            if (
                baseline is not None
                and baseline["p50_validated_e2e_gain_pct"] is not None
                and best != DEFAULT_PARALLELISM_LABEL
            ):
                entry["gain_pct_points_over_default"] = (
                    strategies[best]["p50_validated_e2e_gain_pct"] - baseline["p50_validated_e2e_gain_pct"]
                )
        compared[scope] = entry

    return {
        "sessions_with_parallelism_config": configured,
        "knobs_seen": dict(sorted(knobs_seen.items())),
        "scopes_with_alternatives": compared,
        "scopes_without_alternatives": single,
        "winners_only": True,
        "note": (
            "layouts are compared at a fixed GPU count (workload_shape.tp); "
            "only accepted layouts are published, so a missing layout means "
            "untried or unpublished, not worse"
        ),
    }


def parallelism_whatif(
    rows: list[Mapping[str, Any]],
    *,
    target_tp: int | None = None,
) -> dict[str, Any]:
    """Compare TP within each workload family and project a target TP.

    Cross-TP is never a replay: ``RecipeScope`` requires an exact
    ``tp/conc/isl/osl`` match, so a different TP is a fresh run whose only
    inherited asset is the recipe's direction. Every figure is grouped by
    :func:`workload_family` so TP scaling is not confounded by ISL/OSL.
    """
    families: dict[str, list[Mapping[str, Any]]] = {}
    observed: set[int] = set()
    for row in rows:
        tp = _positive_int(row.get("tp"))
        if tp is None:
            continue
        observed.add(tp)
        families.setdefault(workload_family(row), []).append(row)
    report: dict[str, Any] = {
        "observed_tp": sorted(observed),
        "families": {key: _family_whatif(items, target_tp) for key, items in sorted(families.items())},
        "replayable_across_tp": False,
        "note": (
            "TP is compared only within a fixed conc/isl/osl family; cross-TP "
            "figures are priors, since RecipeScope requires an exact scope match"
        ),
    }
    if target_tp is not None and not families:
        report["projection"] = {"target_tp": target_tp, "reason": "no session carried a workload shape"}
    return report


def estimate_from_sessions(
    documents: list[Mapping[str, Any]],
    *,
    shape: Mapping[str, Any] | None = None,
    target_tp: int | None = None,
) -> dict[str, Any]:
    """Aggregate prior sessions into a scoped gain prior and layout ranking.

    ``shape`` restricts the pool to one replay scope (any subset of
    ``tp``/``conc``/``isl``/``osl``). Without it the headline numbers pool
    every workload shape, which is a weaker prior — ``by_shape`` and
    ``pool_warnings`` make that visible.
    """
    all_rows = [project_session(doc) for doc in documents if isinstance(doc, Mapping)]
    rows = [row for row in all_rows if matches_shape(row, shape)]
    gains = [row["validated_e2e_gain"] for row in rows if row.get("validated_e2e_gain") is not None]

    limitations: list[str] = []
    if not rows:
        limitations.append("no session documents")
    if rows and not gains:
        limitations.append("no validated_e2e_gain on any pooled session; nothing to form a prior from")
    # Only accepted layouts are published, so an absent arm is untried or
    # unpublished rather than beaten. Said once here so a reader does not have
    # to infer it from the winners_only flag alone.
    limitations.append("only winning layouts are published, so an absent layout is untried or unpublished, not worse")

    model_mix = sorted({row["model"] for row in rows if row.get("model")})
    hardware_mix = sorted({row["hardware"] for row in rows if row.get("hardware")})
    framework_mix = sorted({row["framework_name"] for row in rows if row.get("framework_name")})
    version_mix = sorted({row["framework_version"] for row in rows if row.get("framework_version")})
    precision_mix = sorted({row["precision"] for row in rows if row.get("precision")})
    shape_groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        shape_groups.setdefault(shape_key(row), []).append(row)

    pool_warnings: list[str] = []
    if len(model_mix) > 1:
        pool_warnings.append(
            f"pool mixes {len(model_mix)} models {model_mix[:6]}; absolute throughput is not comparable across them"
        )
    if len(hardware_mix) > 1:
        pool_warnings.append(f"pool mixes hardware {hardware_mix}; gains are not comparable across boards")
    if len(framework_mix) > 1:
        pool_warnings.append(f"pool mixes frameworks {framework_mix}; vllm and sglang gains are not interchangeable")
    if len(version_mix) > 1:
        pool_warnings.append(f"pool mixes framework versions {version_mix}; a fixed upstream regression inflates gain")
    if len(precision_mix) > 1:
        pool_warnings.append(f"pool mixes precisions {precision_mix}; replay requires an exact precision match")
    if shape is None and len(shape_groups) > 1:
        pool_warnings.append(
            f"pool mixes {len(shape_groups)} workload shapes; pass tp/conc/isl/osl to scope, or read by_shape"
        )

    n = len(rows)
    return {
        "sessions_scored": n,
        "sessions_dropped_by_shape_filter": len(all_rows) - n,
        "identity_mix": {
            "model": model_mix,
            "hardware": hardware_mix,
            "framework_name": framework_mix,
            "framework_version": version_mix,
            "precision": precision_mix,
        },
        "pool_warnings": pool_warnings,
        "requested_shape": {key: _positive_int((shape or {}).get(key)) for key in _SHAPE_KEYS},
        "by_shape": {key: _bucket_stats(items) for key, items in sorted(shape_groups.items())},
        "parallelism_whatif": parallelism_whatif(rows, target_tp=target_tp),
        "sharding_whatif": sharding_whatif(rows),
        "coverage": {
            "with_gain": len(gains),
        },
        "historical": {
            "p50_validated_e2e_gain_pct": percentile(gains, 50),
            "p90_validated_e2e_gain_pct": percentile(gains, 90),
        },
        "limitations": limitations,
        "sessions": rows,
    }


def _session_ids_from_rollup(rollup: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(rollup, Mapping):
        return []
    ids: list[str] = []
    for row in rollup.get("sessions") or []:
        if isinstance(row, str) and row.strip():
            ids.append(row.strip())
        elif isinstance(row, Mapping):
            sid = str(row.get("session_id") or "").strip()
            if sid:
                ids.append(sid)
    champion = rollup.get("champion")
    if isinstance(champion, Mapping):
        sid = str(champion.get("session_id") or "").strip()
        if sid:
            ids.append(sid)
    seen: set[str] = set()
    unique: list[str] = []
    for sid in ids:
        if sid not in seen:
            seen.add(sid)
            unique.append(sid)
    return unique


def search_inference_identities(
    store: KBStoreClient,
    *,
    match: dict[str, str] | None = None,
    hardware_in: list[str] | None = None,
    max_identities: int = 200,
) -> list[dict[str, Any]]:
    """Page ``POST /v1/kb/search`` for inference identities."""
    items: list[dict[str, Any]] = []
    offset = 0
    for _ in range(_SEARCH_PAGE_CAP):
        if len(items) >= max_identities:
            break
        result = store.search_identities(
            scheme="inference",
            match=match,
            hardware_in=hardware_in,
            offset=offset,
            limit=min(_SEARCH_PAGE, max_identities - len(items)),
        )
        page = result.get("items") if isinstance(result, Mapping) else None
        if not isinstance(page, list) or not page:
            break
        for item in page:
            if isinstance(item, Mapping):
                items.append(dict(item))
                if len(items) >= max_identities:
                    break
        if len(page) < _SEARCH_PAGE:
            break
        offset += len(page)
    return items


def fetch_session_documents(
    store: KBStoreClient,
    *,
    match: dict[str, str] | None = None,
    hardware_in: list[str] | None = None,
    max_identities: int = 50,
    canonical_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Download per-session envelopes for matching identities."""
    errors: list[str] = []
    if canonical_ids:
        identities = [{"canonical_id": cid} for cid in canonical_ids]
    else:
        identities = search_inference_identities(
            store,
            match=match,
            hardware_in=hardware_in,
            max_identities=max_identities,
        )
    documents: list[dict[str, Any]] = []
    for item in identities:
        cid = str(item.get("canonical_id") or "").strip()
        if not cid:
            continue
        try:
            rollup = store.get_rollup(cid)
        except KBStoreError as exc:
            errors.append(f"{cid}: rollup {exc}")
            continue
        session_ids = _session_ids_from_rollup(rollup)
        if not session_ids:
            errors.append(f"{cid}: no sessions in rollup")
            continue
        for sid in session_ids:
            try:
                envelope = store.get_session(cid, sid)
            except KBStoreError as exc:
                errors.append(f"{cid}/{sid}: {exc}")
                continue
            if isinstance(envelope, Mapping):
                documents.append(dict(envelope))
    return documents, errors


__all__ = [
    "DEFAULT_PARALLELISM_LABEL",
    "estimate_from_sessions",
    "extract_parallelism",
    "fetch_session_documents",
    "identity_dimensions",
    "matches_shape",
    "parallelism_label",
    "parallelism_whatif",
    "percentile",
    "project_session",
    "replay_scope_key",
    "search_inference_identities",
    "shape_key",
    "sharding_whatif",
    "workload_family",
]
