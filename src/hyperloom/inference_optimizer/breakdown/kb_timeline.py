"""Project the three Recipe KB touchpoints into additive V6 timeline events.

``warm_start``     which identity was asked for, and what the KB returned.
``warm_replay``    whether replaying that record reproduced its gain, and --
                   only when it did -- the configuration that was running.
``kb_write_back``  whether this session's own Recipe reached the KB Store.

Every value is read back from evidence the run already persisted, so these
events add no runtime cost and survive an offline re-export.

Recipe material follows the three published columns (``config`` / ``patch`` /
``kernel``). The retired ``patch_timeline`` is never read: an overlay ref's
lexicographic order is the replay order.

A replay that did not reproduce records why, not what it was carrying. The
configuration is only meaningful once something ran with it end to end, and a
rejected replay has already rolled its material back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from hyperloom.common.jsonio import read_jsonl

from ..session.session_paths import (
    recipe_kb_dead_letter_ndjson,
    recipe_kb_flushed_ndjson,
    recipe_kb_pending_ndjson,
    recipe_snapshot_audit_jsonl,
)
from .collectors._common import (
    _dict_rows,
    _first,
    _mapping,
    _optional_bool,
    _string_list,
    _to_float as _optional_float,
    _to_int as _optional_int,
)
from .schema import (
    V6KBWriteBackExt,
    V6WarmReplayApplied,
    V6WarmReplayExt,
    V6WarmStartExt,
    V6WarmStartMatched,
    V6WarmStartReads,
)


# ---------------------------------------------------------------------------
# warm_start
# ---------------------------------------------------------------------------

# recipe_kb_t0 publishes one of these four; anything else is treated as a miss.
_WARM_START_STATUS = {
    "hit": "matched",
    "seed_only": "not_matched",
    "miss": "not_matched",
    "error": "failed",
}
_EXACT_TIERS = frozenset({"exact"})


def _scope(state: dict[str, Any]) -> dict[str, Any]:
    """The workload dimensions a Recipe is partitioned by (RecipeScope)."""
    return {
        "kernel_optimizer": str(state.get("kernel_optimizer") or ""),
        "tp": _optional_int(state.get("tp")),
        "conc": _optional_int(state.get("conc")),
        "isl": _optional_int(state.get("isl")),
        "osl": _optional_int(state.get("osl")),
    }


def _matched_scope(recipe: dict[str, Any]) -> dict[str, Any] | None:
    """The matched record's own workload shape, when it published one."""
    shape = _mapping(recipe.get("workload_shape"))
    if not shape:
        return None
    return {
        "tp": _optional_int(shape.get("tp")),
        "conc": _optional_int(shape.get("conc")),
        "isl": _optional_int(shape.get("isl")),
        "osl": _optional_int(shape.get("osl")),
    }


def _requested_canonical_id(state: dict[str, Any], matched_id: str) -> str:
    """Resolve the identity this session asked the KB for.

    Read rather than rebuilt: the hardware dimension is topology-aware and
    resolved from the runtime environment, so recomputing it at export time
    could disagree with what the run actually queried. CLOSE derives the
    session's own identity through the same helper T0 used, which makes
    ``recipe_finalize`` the authority; an exact match is the same string.
    """
    finalize = _mapping(state.get("recipe_finalize_outcome"))
    recorded = str(finalize.get("canonical_id") or "").strip()
    if recorded:
        return recorded
    tier = str(_mapping(state.get("warm_start_recipe")).get("tier") or "").strip().lower()
    return matched_id if tier in _EXACT_TIERS else ""


def _warm_start_origin(recipe: dict[str, Any]) -> dict[str, Any] | None:
    """Which session wrote the matched record, and what it gained."""
    provenance = _mapping(recipe.get("provenance"))
    session_id = str(_first(recipe.get("remote_session_id"), provenance.get("session_id")) or "")
    sessions = _dict_rows(recipe.get("sessions"))
    gain = _optional_float(sessions[0].get("gain_pct")) if sessions else None
    if not session_id and gain is None:
        return None
    return {"session_id": session_id or None, "gain_pct": gain}


def _recipe_snapshot_reads(session_dir: Path, warnings: list[str]) -> V6WarmStartReads | None:
    """Per-source attribution of this session's Recipe KB reads.

    Read back from the recipe-snapshot audit log: how each T0 lookup resolved,
    which backend served it, and which source supplied the champion config.
    ``None`` when the session recorded no readable read, so the block is only
    attached to ``warm_start`` when it carries something.
    """
    path = recipe_snapshot_audit_jsonl(session_dir)
    if not path.exists():
        return None
    try:
        rows = [row for row in read_jsonl(path) if isinstance(row, dict)]
    except (OSError, ValueError) as exc:  # noqa: BLE001 — an unreadable audit is "no reads"
        warnings.append(f"timeline.warm_start.reads: failed to read audit {path}: {exc!r}"[:240])
        return None
    if not rows:
        return None
    rows = rows[-50:]

    by_resolution: dict[str, int] = {}
    by_remote: dict[str, int] = {}
    by_source: dict[str, int] = {}
    best_config_by_source: dict[str, int] = {}
    hits = 0
    for row in rows:
        resolution = str(row.get("resolution") or "unknown")
        by_resolution[resolution] = by_resolution.get(resolution, 0) + 1
        remote = str(row.get("remote") or "unknown")
        by_remote[remote] = by_remote.get(remote, 0) + 1
        if row.get("hit"):
            hits += 1
        result = row.get("result") if isinstance(row.get("result"), dict) else {}
        for src in result.get("sources") or []:
            by_source[str(src)] = by_source.get(str(src), 0) + 1
        best = result.get("best_config_source")
        for src in best if isinstance(best, list) else [best] if best else []:
            best_config_by_source[str(src)] = best_config_by_source.get(str(src), 0) + 1

    return {
        "count": len(rows),
        "hits": hits,
        "by_resolution": by_resolution,
        "by_remote": by_remote,
        "by_source": by_source,
        "best_config_by_source": best_config_by_source,
        # A short tail of the raw rows: downstream champion-config and donor
        # resolution read the most recent hit's own result off these.
        "tail": rows[-10:],
    }


def collect_warm_start_event(
    state: dict[str, Any],
    reads: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Project the PRELUDE Recipe KB lookup, or ``None`` when it never ran."""
    warm = _mapping(state.get("warm_start_recipe"))
    context = _mapping(state.get("warm_start_context"))
    ts = str(state.get("warm_start_ts") or "")
    if not (warm or context or ts):
        return None

    raw_status = str(context.get("status") or "").strip().lower()
    tier = str(warm.get("tier") or "").strip()
    recipe = _mapping(warm.get("recipe"))
    if raw_status in _WARM_START_STATUS:
        status = _WARM_START_STATUS[raw_status]
    else:
        status = "matched" if recipe and tier and tier.lower() != "miss" else "not_matched"

    match = _mapping(context.get("match"))
    matched_id = str(_first(match.get("canonical_id"), recipe.get("canonical_id")) or "")
    ext: V6WarmStartExt = {
        "requested": {
            "canonical_id": _requested_canonical_id(state, matched_id),
            "scope": _scope(state),
        },
    }
    if raw_status and raw_status not in {"hit"}:
        # ``seed_only`` is a hit that could not be executed; keeping the raw
        # value stops it reading as a plain miss.
        ext["match_status"] = raw_status

    if status == "matched":
        lessons = state.get("warm_start_lessons")
        pitfalls = state.get("warm_start_pitfalls")
        match_type: Literal["exact", "degraded"] = "exact" if tier.lower() in _EXACT_TIERS else "degraded"
        matched: V6WarmStartMatched = {
            "match_type": match_type,
            "tier": tier,
            "confidence": _optional_float(_first(warm.get("confidence"), match.get("confidence"))),
            "source": str(match.get("source") or ""),
            "canonical_id": matched_id,
            "optimized_throughput": _optional_float(recipe.get("best_throughput")),
            "validated_gain_pct": _optional_float(recipe.get("validated_gain_pct")),
            "expected_gain_pct": _optional_float(_mapping(context.get("recommended_replay")).get("expected_gain_pct")),
            "replayable": _optional_bool(recipe.get("replayable")),
            "replay_disabled_reason": str(recipe.get("replay_disabled_reason") or "") or None,
            "replay_material_available": _optional_bool(recipe.get("replay_material_available")),
            "view_source": str(recipe.get("view_source") or "") or None,
            "experience": {
                "lessons_count": len(lessons) if isinstance(lessons, list) else 0,
                "pitfalls_count": len(pitfalls) if isinstance(pitfalls, list) else 0,
            },
        }
        scope = _matched_scope(recipe)
        if scope is not None:
            matched["scope"] = scope
        origin = _warm_start_origin(recipe)
        if origin is not None:
            matched["origin"] = origin
        ext["matched"] = matched

    # Read attribution belongs to the T0 read regardless of whether it matched:
    # a miss still records which backends were consulted and how each resolved.
    if reads:
        ext["reads"] = reads

    return {
        "type": "warm_start",
        "kind": "warm_start",
        "status": status,
        # A T0 anchor is a single read; it has no measurable duration.
        "start_time": ts,
        "end_time": ts,
        "ext": ext,
    }


# ---------------------------------------------------------------------------
# warm_replay
# ---------------------------------------------------------------------------

_REPRODUCED_STATUSES = frozenset({"reproduced", "reproduced_but_no_params"})
_PASS_THROUGH_STATUSES = frozenset({"drift", "skipped", "in_flight"})

# Ordered: the first substring that appears in the raw reason wins, so the
# specific codes must precede the generic ones.
_SKIP_REASONS: tuple[tuple[str, str], ...] = (
    ("disabled_by_flag", "disabled_by_flag"),
    ("no_warm_start_recipe", "no_warm_start_recipe"),
    ("best_config_empty", "best_config_empty"),
    ("confidence_below_threshold", "confidence_below_threshold"),
    ("current_recipe_sdk_read_failed", "recipe_read_failed"),
    ("not_replayable", "recipe_not_replayable"),
    ("legacy native records do not satisfy", "recipe_not_replayable"),
    ("workload_config_incompatible", "workload_config_incompatible"),
    ("active_framework_root_missing", "framework_root_missing"),
    ("framework_patch_root", "framework_root_missing"),
    ("kernel_patch_root", "kernel_root_missing"),
    ("active_kernel_patch_root_missing", "kernel_root_missing"),
    ("warm_replay_patch_content_missing", "patch_content_missing"),
    ("no_matching_root", "patch_root_unresolved"),
    ("ambiguous_root", "patch_root_unresolved"),
    ("explicit_root", "patch_root_unresolved"),
    ("patch_targets_invalid", "patch_root_unresolved"),
    ("enqueue failed", "enqueue_failed"),
)
_FAILURE_STATUS_CODES = {
    "rollback_failed": "rollback_failed",
    "enqueue_failed": "enqueue_failed",
    "kernel_preparation_failed": "kernel_preparation_failed",
    "quality_failed": "quality_failed",
    "accuracy_failed": "accuracy_failed",
    "promotion_failed": "promotion_failed",
}


def _bucket(reason: str, table: tuple[tuple[str, str], ...], default: str) -> str:
    lowered = reason.strip().lower()
    for needle, code in table:
        if needle in lowered:
            return code
    return default


def _replay_status(raw_status: str) -> str:
    """Collapse the runtime vocabulary onto the five published outcomes."""
    if raw_status in _REPRODUCED_STATUSES:
        return "reproduced"
    if raw_status in _PASS_THROUGH_STATUSES:
        return raw_status
    return "failed"


def _replay_result_type(raw_status: str, reason: str, outcome: dict[str, Any]) -> str | None:
    """A stable code for why the replay ended where it did."""
    if raw_status == "reproduced":
        return None
    if raw_status == "in_flight":
        return None
    if raw_status == "reproduced_but_no_params":
        return "reproduced_without_params"
    if raw_status == "drift":
        if _optional_bool(outcome.get("below_historical_reproduce_pct")) is True:
            return "below_historical_reproduce_bar"
        return "below_keep_threshold"
    if raw_status == "skipped":
        return _bucket(reason, _SKIP_REASONS, "skipped_other")
    if raw_status in _FAILURE_STATUS_CODES:
        return _FAILURE_STATUS_CODES[raw_status]
    lowered = reason.strip().lower()
    if "invalid_tput" in lowered:
        return "invalid_throughput"
    if "interrupted" in lowered:
        return "interrupted"
    if "non_dict_result" in lowered:
        return "malformed_result"
    return "replay_task_failed"


def _replay_before_tput(state: dict[str, Any], after: float | None, gain: float | None) -> float | None:
    """The baseline the replay was judged against.

    The outcome records the measurement and the gain but not the anchor it was
    taken against, so the session baseline stands in. It is reconstructed from
    the pair only when the baseline is unavailable, which keeps a re-baselined
    session from reporting an anchor it never used.
    """
    baseline = _optional_float(state.get("baseline_tput"))
    if baseline and baseline > 0:
        return baseline
    if after is not None and gain is not None and gain != -100.0:
        return round(after / (1.0 + gain / 100.0), 6)
    return None


def _replay_stack_entry(state: dict[str, Any]) -> dict[str, Any]:
    """The ``replay_warm_recipe`` layer, present only once a replay was kept."""
    rows = _dict_rows(state.get("optimization_stack"))
    return next(
        (row for row in reversed(rows) if str(row.get("action") or "") == "replay_warm_recipe"),
        {},
    )


def _replay_applied(state: dict[str, Any], outcome: dict[str, Any]) -> V6WarmReplayApplied | None:
    """What was running when the replay reproduced its gain.

    One merged configuration, not a per-column split: the columns are applied
    together and measured together, so attributing the gain to one of them
    would be a guess.
    """
    entry = _replay_stack_entry(state)
    args = str(entry.get("candidate_extra_server_args") or "")
    envs = dict(_mapping(entry.get("candidate_extra_envs")))
    patches = _string_list(_first(outcome.get("replayed_patch_refs"), entry.get("replayed_patch_refs"), []))
    kernel = _mapping(outcome.get("kernel"))
    kernel_replay = _mapping(entry.get("kernel_replay"))
    if not (args or envs or patches or kernel):
        return None
    applied: V6WarmReplayApplied = {
        "config": {"extra_server_args": args, "extra_envs": envs},
        # Ref order is replay order; the separate timeline column is retired.
        "patch": patches,
    }
    if kernel:
        applied["kernel"] = {
            "status": str(kernel.get("status") or ""),
            "total": _optional_int(kernel.get("total")),
            "kept": _optional_int(kernel.get("kept")),
            "reverted": _optional_int(kernel.get("reverted")),
            "columns": _string_list(kernel_replay.get("columns")),
        }
    return applied


def collect_warm_replay_event(state: dict[str, Any]) -> dict[str, Any] | None:
    """Project the PRELUDE warm replay, or ``None`` when it never ran."""
    outcome = _mapping(state.get("warm_replay_outcome"))
    if not outcome and not state.get("warm_replay_attempted"):
        return None

    raw_status = str(outcome.get("status") or "").strip().lower()
    reason = str(outcome.get("reason") or "")
    status = _replay_status(raw_status)
    after = _optional_float(outcome.get("throughput_after"))
    gain = _optional_float(outcome.get("actual_gain_pct"))

    ext: V6WarmReplayExt = {
        "raw_status": raw_status,
        "tier": str(outcome.get("warm_recipe_tier") or "") or None,
        "confidence": _optional_float(outcome.get("warm_recipe_conf")),
        "config_source": str(outcome.get("config_source") or "") or None,
        "config_donor_tier": str(outcome.get("config_donor_tier") or "") or None,
        "before_tput": _replay_before_tput(state, after, gain),
        "after_tput": after,
        "gain_pct": gain,
        "expected_gain_pct": _optional_float(outcome.get("expected_gain_pct")),
        "keep_threshold_pct": _optional_float(outcome.get("keep_threshold_pct")),
        "historical_reproduce_bar_pct": _optional_float(outcome.get("historical_reproduce_bar_pct")),
        "below_historical_reproduce": _optional_bool(outcome.get("below_historical_reproduce_pct")),
        "accuracy": {
            "eval_ran": _optional_bool(outcome.get("eval_ran")),
            "baseline": _optional_float(outcome.get("baseline_accuracy")),
            "replay": _optional_float(outcome.get("replay_accuracy")),
            # No explicit verdict is stored; the gate is the only thing that
            # can end a replay as ``accuracy_failed``.
            "passed": (
                False
                if raw_status == "accuracy_failed"
                else True
                if _optional_bool(outcome.get("eval_ran")) is True
                else None
            ),
        },
    }
    result_type = _replay_result_type(raw_status, reason, outcome)
    if result_type is not None:
        ext["result_type"] = result_type
        ext["raw_reason"] = reason or None

    donor_id = str(outcome.get("donor_canonical_id") or "")
    if donor_id:
        ext["donor"] = {
            "canonical_id": donor_id,
            "model": str(outcome.get("donor_model") or "") or None,
            "session_id": str(outcome.get("donor_session_id") or "") or None,
            "gain_pct": _optional_float(outcome.get("donor_gain_pct")),
            "breakdown_link": str(outcome.get("donor_breakdown_link") or "") or None,
        }

    if status == "reproduced":
        applied = _replay_applied(state, outcome)
        if applied is not None:
            ext["applied"] = applied

    root = str(outcome.get("active_framework_root") or "")
    if root:
        ext["active_framework_root"] = root
    rollback = _mapping(outcome.get("rollback"))
    if rollback:
        ext["rollback"] = {
            "ok": _optional_bool(rollback.get("ok")),
            "errors": _string_list(rollback.get("errors")),
        }
    error_class = str(outcome.get("error_class") or "")
    if status == "failed" or error_class:
        ext["failure"] = {"error_class": error_class or None, "error": reason or None}

    start = str(outcome.get("enqueued_at") or "")
    end = str(outcome.get("settled_at") or "")
    if not start:
        # Skips are decided inline during PRELUDE, before a task exists; the
        # warm-start read is the closest instant the run recorded.
        start = str(state.get("warm_start_ts") or "")
    return {
        "type": "warm_replay",
        "kind": "warm_replay",
        "status": status,
        "start_time": start,
        "end_time": end or start,
        "ext": ext,
    }


# ---------------------------------------------------------------------------
# kb_write_back
# ---------------------------------------------------------------------------

_WRITE_BACK_REASONS: tuple[tuple[str, str], ...] = (
    ("no_new_keep", "no_new_keep_or_pure_warm_replay"),
    ("nonfinite_optimized_throughput", "invalid_throughput"),
    ("missing_optimized_throughput", "missing_throughput"),
    ("invalid_recipe_scope", "invalid_scope"),
    ("missing_model_or_hardware", "invalid_scope"),
    ("empty_replay_material", "empty_replay_material"),
    ("not_better_than_champion", "not_better_than_champion"),
    ("champion_not_promoted", "champion_not_promoted"),
    ("degraded_kb", "kb_disabled"),
    ("not configured", "kb_disabled"),
    ("no_recipe_backend", "kb_disabled"),
    ("configuration:", "configuration_failed"),
    ("remoterecipevalidationerror", "bundle_build_failed"),
    # ``agentx`` is a short, collision-prone needle -- it appears inside
    # exception class names such as ``configuration:AgentXConfigError`` -- so it
    # stays last: the more specific rules above claim their reasons first, and
    # ``_bucket`` is first-match-wins.
    ("agentx", "agentx_blocked"),
)
_WRITE_BACK_STATUS = {"written": "written", "skipped": "skipped", "disabled": "skipped"}


def _queue_depth(session_dir: Path, warnings: list[str]) -> dict[str, int]:
    """Line counts of the local KB write queues."""

    def _count(path: Path) -> int:
        try:
            if not path.exists():
                return 0
            with path.open("r", encoding="utf-8") as handle:
                return sum(1 for line in handle if line.strip())
        except OSError as exc:
            warnings.append(f"timeline.kb_write_back: failed to count {path}: {exc!r}")
            return 0

    return {
        "pending_lines": _count(recipe_kb_pending_ndjson(session_dir)),
        "flushed_bookmarks": _count(recipe_kb_flushed_ndjson(session_dir)),
        "dead_letter_lines": _count(recipe_kb_dead_letter_ndjson(session_dir)),
    }


def collect_kb_write_back_event(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any] | None:
    """Project the CLOSE Recipe publication, or ``None`` when it never ran."""
    outcome = _mapping(state.get("recipe_finalize_outcome"))
    raw_status = str(_first(state.get("recipe_finalize_status"), outcome.get("status")) or "").strip().lower()
    if not outcome and not raw_status:
        return None
    if raw_status == "pending":
        # A run that died mid-publish leaves the marker behind; reporting it as
        # a write would claim an outcome the KB never confirmed.
        raw_status = "failed"

    reason = str(outcome.get("reason") or "")
    status = _WRITE_BACK_STATUS.get(raw_status, "failed")
    result_type = _bucket(
        reason,
        _WRITE_BACK_REASONS,
        "" if status == "written" else ("transport_failed" if status == "failed" else "skipped_other"),
    )

    current_best = _mapping(state.get("current_best"))
    ext: V6KBWriteBackExt = {
        "backend": str(outcome.get("backend") or "") or None,
        "canonical_id": str(outcome.get("canonical_id") or "") or None,
        "session_id": str(outcome.get("session_id") or "") or None,
        "scope": _scope(state),
        "optimized_throughput": _optional_float(current_best.get("tput")),
        "validated_gain_pct": _optional_float(state.get("cumulative_gain_validated")),
        "attempts": _optional_int(state.get("recipe_finalize_attempts")),
        "source": str(outcome.get("source") or "") or None,
        "queue": _queue_depth(session_dir, warnings),
    }
    if result_type:
        ext["result_type"] = result_type
        ext["raw_reason"] = reason or None
    if status == "failed":
        # The publisher surfaces transport and build failures as the exception
        # class name, so the reason doubles as the class.
        ext["failure"] = {"error_class": reason or None, "error": reason or None}

    ts = str(outcome.get("updated_at") or "")
    return {
        "type": "kb_write_back",
        "kind": "kb_write_back",
        "status": status,
        "start_time": ts,
        "end_time": ts,
        "ext": ext,
    }


def collect_kb_events(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Every KB timeline event this session produced, in execution order."""
    events = [
        collect_warm_start_event(state, _recipe_snapshot_reads(session_dir, warnings)),
        collect_warm_replay_event(state),
        collect_kb_write_back_event(session_dir, state, warnings),
    ]
    return [event for event in events if event is not None]


__all__ = [
    "collect_kb_events",
    "collect_kb_write_back_event",
    "collect_warm_replay_event",
    "collect_warm_start_event",
]
