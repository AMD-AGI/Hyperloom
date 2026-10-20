# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared T0 (PRELUDE) Recipe KB anchor — KB warm-start only."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple

from hyperloom.common.perf_metric import agentx_active
from hyperloom.orchestrator.knowledge._warm_start_match import (
    _ConfigDonor,
    _DonorTarget,
    _SearchIdentity,
    _config_replay_args_envs,
    _donor_is_trustworthy,
    _find_config_donor,
    _has_replayable_config,
    _max_session_gain,
    _rank_warm_candidates,
    _recipe_is_actionable,
    _remote_candidate_matches,
    _search_warm_candidates,
    _select_remote_candidate,
    _take_unseen,
    _text,
    _trustworthy_candidates,
    _warm_search_tiers,
    _with_exact_history,
)
from hyperloom.orchestrator.knowledge.recipe_kb import (
    RecipeKB,
    cid_to_path_components,
    recipe_canonical_id,
)
from hyperloom.orchestrator.knowledge.workload_tags import WORKLOAD_SHAPE_FIELDS, workload_shape_tags
from hyperloom.inference_optimizer.breakdown.recorder import warm_start_event as _warm_start_event
from hyperloom.inference_optimizer.recipe_snapshot_constants import detect_framework_version, kb_hardware_slug
from hyperloom.inference_optimizer.session.session_paths import (
    recipe_kb_lessons_json,
    recipe_kb_pitfalls_json,
)


log = logging.getLogger(__name__)


class _T0Stack(NamedTuple):
    """The stack metadata the caller hands T0 for the anchor row."""

    extra: Mapping[str, Any]
    fingerprint: Mapping[str, Any]
    image_digest: str


class _T0Identity(NamedTuple):
    """The canonical id a T0 anchor is written under, with the dimensions it was built from."""

    canonical_id: str
    workload: str
    search: _SearchIdentity
    extras: dict[str, Any]


class _WarmMatch(NamedTuple):
    """What the KB offered for an identity: the row, the tier that reached it, and the tier's confidence."""

    recipe: dict[str, Any]
    tier: str
    confidence: float


class _WarmStartExperience(NamedTuple):
    """The experience rows a matched recipe carries into the session."""

    lessons: list[dict[str, Any]]
    pitfalls: list[dict[str, Any]]


def _default_status_emitter(line: str) -> None:
    """Default ``on_status`` callback — log the banner line at INFO."""
    log.info("%s", line)


def _mapping(value: Any) -> Mapping[str, Any]:
    """*value* when the caller supplied a mapping, else an empty one."""
    return value if isinstance(value, Mapping) else {}


def _build_warm_prefer(shared_state: Any, framework_version: str) -> dict[str, Any]:
    """Assemble the ``prefer`` similarity hints from SharedState so a closer-workload recipe is reranked first."""
    prefer: dict[str, Any] = {}
    for attr in WORKLOAD_SHAPE_FIELDS:
        val = getattr(shared_state, attr, None)
        if val not in (None, "", 0):
            prefer[attr] = val
    fv = _text(framework_version)
    if fv:
        prefer["framework_version"] = fv
    wl_extra = getattr(shared_state, "baseline_workload_extra", None) or {}
    if isinstance(wl_extra, Mapping):
        for key in ("quant_scheme", "workload_mode"):
            v = _text(wl_extra.get(key))
            if v:
                prefer[key] = v
    return prefer


def _warm_recipe_source(row: Mapping[str, Any] | None, kb: Any) -> str:
    """Return the source tag for a Recipe warm-start row."""
    del row
    return str(getattr(kb, "backend_name", "") or "recipe-kb")


def _experience_rows(raw: Any, required_key: str, label: str) -> list[dict[str, Any]]:
    """Normalise a recipe row's ``lessons`` / ``pitfalls`` into flat rows.

    The stored shape is flat — ``Recipe.to_dict`` and the ``_normalise_*``
    helpers write ``{statement, ...}`` / ``{description, ...}``. The remote
    projection (``knowledge_to_warm_recipe``) copies rows out of the record
    without going through ``Recipe.from_dict``, so a wrapped row can still arrive
    that way; this is the single place it is unwrapped, and no reader downstream
    has to know about two shapes.

    Rows missing *required_key* are dropped here rather than silently vanishing
    in the prompt renderer, which is what let a wrong shape pass unnoticed: the
    section rendered "(none)" off a non-empty list with no log and no error.

    Args:
        raw: The row list as read off the recipe.
        required_key: The field a usable row must carry.
        label: Field name for the warning.

    Returns:
        The flat rows that carry *required_key*.
    """
    out: list[dict[str, Any]] = []
    dropped = 0
    for item in raw or []:
        if not isinstance(item, Mapping):
            dropped += 1
            continue
        wrapped = item.get("attrs")
        row = dict(wrapped) if isinstance(wrapped, Mapping) and wrapped else dict(item)
        if not _text(row.get(required_key)):
            dropped += 1
            continue
        out.append(row)
    if dropped:
        log.warning(
            "warm_start_%s: dropped %d row(s) missing %r; kept %d",
            label,
            dropped,
            required_key,
            len(out),
        )
    return out


def _warm_start_scope(shared_state: Any) -> dict[str, Any]:
    """The workload dimensions the lookup was issued under (RecipeScope)."""
    return {
        "kernel_optimizer": str(getattr(shared_state, "kernel_optimizer", "") or ""),
        "tp": getattr(shared_state, "tp", None),
        "conc": getattr(shared_state, "conc", None),
        "isl": getattr(shared_state, "isl", None),
        "osl": getattr(shared_state, "osl", None),
    }


def _settle_warm_start_event(
    recorder: Any,
    *,
    match: _WarmMatch,
    status: str,
    source: str,
    canonical_id: str,
    context: Any,
    experience: _WarmStartExperience,
) -> None:
    """Close the warm_start event.

    Args:
        recorder (Any): The open warm-start recorder.
        match (_WarmMatch): The row the KB served, with its tier and confidence.
        status (str): T0's own ``hit`` / ``seed_only`` / ``miss`` verdict.
        source (str): Which store served the match.
        canonical_id (str): The identity that was queried.
        context (Any): The built ``warm_start_context``, read for the gain the
            matched record claims.
        experience (_WarmStartExperience): The record's lessons and pitfalls.
    """
    matched = None
    if status in _warm_start_event.MATCHED_STATUSES and match.recipe:
        matched = _warm_start_event.matched_block(
            tier=match.tier,
            confidence=match.confidence,
            source=source,
            canonical_id=canonical_id,
            recipe=match.recipe,
            expected_gain_pct=_claimed_gain_pct(context),
            lessons=experience.lessons,
            pitfalls=experience.pitfalls,
        )
    recorder.finish(match_status=status, matched=matched)


def _claimed_gain_pct(context: Any) -> float | None:
    """The gain the built warm-start context says its recommended replay should repeat."""
    replay = context.get("recommended_replay") if isinstance(context, Mapping) else None
    return replay.get("expected_gain_pct") if isinstance(replay, Mapping) else None


def _exact_recipe(kb: Any, *, cid: str, prefer: Any) -> Mapping[str, Any] | None:
    """The stored recipe filed under exactly *cid*, or None when the store has no such row."""
    try:
        row = kb.get_recipe(
            canonical_id=cid,
            prefer=prefer or None,
        )
    except Exception as exc:  # noqa: BLE001
        log.info("warm-start exact get non-fatal failure: %s", exc)
        return None
    if isinstance(row, Mapping) and row and str(row.get("canonical_id") or "") == cid:
        return row
    return None


def _identity_from_cid(cid: str, fallback: _SearchIdentity) -> _SearchIdentity:
    """The seven-tuple the canonical id encodes, or *fallback* when it does not parse."""
    try:
        (
            _model,
            hardware,
            framework,
            model_type,
            architectures,
            framework_version,
            precision,
        ) = cid_to_path_components(cid)
    except ValueError:
        return fallback
    return _SearchIdentity(
        hardware=hardware,
        framework=framework,
        model_type=model_type,
        arch_slug=architectures,
        framework_version=framework_version,
        precision=precision,
    )


def _cascade_warm_start_search(
    kb: "RecipeKB",
    *,
    cid: str,
    identity: _SearchIdentity,
    warm_prefer: Any,
    target: _DonorTarget,
) -> _WarmMatch:
    """Resolve exact plus cumulative four-tier seven-tuple fallbacks."""
    exact = _exact_recipe(kb, cid=cid, prefer=warm_prefer)
    if exact is not None and _recipe_is_actionable(exact):
        return _WarmMatch(dict(exact), "exact", 1.0)
    stored = _identity_from_cid(cid, identity)
    seen = {cid}
    for query in _warm_search_tiers(stored):
        rows = _search_warm_candidates(
            kb,
            labels=query.labels,
            hardware_in=query.hardware_in,
        )
        offered = [row for row in _take_unseen(rows, seen) if _remote_candidate_matches(kb, row, labels=query.labels)]
        usable = _trustworthy_candidates(offered, identity=stored, query=query, target=target)
        if rows and not usable:
            log.info(
                "warm-start tier %s rejected all %d candidates",
                query.tier,
                len(rows),
            )
        for candidate in _rank_warm_candidates(usable, target_framework_version=stored.framework_version):
            if _select_remote_candidate(kb, candidate):
                return _WarmMatch(_with_exact_history(candidate, exact), query.tier, query.confidence)
    if exact is not None:
        return _WarmMatch(dict(exact), "exact", 1.0)
    return _WarmMatch({}, "miss", 0.0)


def _advisory_source(recipe: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """The row a match's advisory content is read off: its exact-identity history when it carries one, else itself."""
    row = _mapping(recipe)
    history = row.get("exact_history")
    return history if isinstance(history, Mapping) else row


def _warm_start_priors(recipe: Mapping[str, Any] | None) -> dict[str, list[Any]]:
    """The advisory lists a matched identity carries, read off its exact-identity history when it has one."""
    source = _advisory_source(recipe)
    return {
        "proven_prior": list(source.get("what_worked") or []),
        "do_not_repeat": list(source.get("what_failed") or []),
        "lessons": list(source.get("lessons") or []),
        "pitfalls": list(source.get("pitfalls") or []),
    }


def _replay_donor(
    recipe: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """The row a recommended replay is built from: the borrowed candidate, else the identity row as self-donor."""
    from .remote_recipe import RECORD_KIND_HYPERLOOM_RECIPE

    if isinstance(recipe, Mapping) and recipe.get("record_kind") == RECORD_KIND_HYPERLOOM_RECIPE:
        return None
    if isinstance(candidate, Mapping):
        return candidate
    if isinstance(recipe, Mapping) and _has_replayable_config(recipe):
        return recipe
    return None


def _best_donor_session(donor: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The donor session that claims the largest gain, or None when the row records none."""
    best: Mapping[str, Any] | None = None
    best_gain = float("-inf")
    for session in donor.get("sessions") or []:
        if not isinstance(session, Mapping):
            continue
        try:
            gain = float(session.get("gain_pct") or 0.0)
        except (TypeError, ValueError):
            continue
        if gain > best_gain:
            best = session
            best_gain = gain
    return best


def _donor_expected_gain(donor: Mapping[str, Any]) -> float:
    """The gain a donor's config is expected to repeat: its validated gain, else its best session's."""
    try:
        validated = float(donor.get("validated_gain_pct") or 0.0)
    except (TypeError, ValueError):
        validated = 0.0
    return validated if validated > 0 else _max_session_gain(donor)


def _donor_best_throughput(donor: Mapping[str, Any]) -> float:
    """The throughput the donor's champion config reached, or 0.0 when the row does not say."""
    try:
        return float(donor.get("best_throughput") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _donor_provenance(
    donor: Mapping[str, Any],
    session: Mapping[str, Any] | None,
    gain_pct: float,
) -> dict[str, Any]:
    """Where a recommended replay config came from: the donor row, its best session, and the gain it claims."""
    family_tags = donor.get("family_tags") or donor.get("model_architectures")
    breakdown_link = str(donor.get("breakdown_link") or "")
    if session is not None:
        breakdown_link = str(session.get("breakdown_link") or session.get("session_breakdown_url") or breakdown_link)
    return {
        "donor_canonical_id": str(donor.get("canonical_id") or ""),
        "donor_model": str(donor.get("model") or ""),
        "donor_session_id": str(session.get("session_id") or "") if session is not None else "",
        "donor_family_tags": ([str(tag) for tag in family_tags] if isinstance(family_tags, (list, tuple, set)) else []),
        "donor_gain_pct": gain_pct,
        "donor_breakdown_link": breakdown_link,
    }


def _recommended_replay(donor: Mapping[str, Any], *, tier: str, confidence: float) -> dict[str, Any] | None:
    """The replay block a donor row offers, or None when it carries no champion config."""
    args, envs = _config_replay_args_envs(donor)
    if not (args or envs):
        return None
    expected_gain = _donor_expected_gain(donor)
    replay: dict[str, Any] = {
        "extra_server_args": args,
        "extra_envs": envs,
        "expected_gain_pct": expected_gain,
        "best_throughput": _donor_best_throughput(donor),
        "config_source": str(donor.get("canonical_id") or ""),
        "config_tier": tier,
        "config_confidence": confidence,
    }
    replay.update(_donor_provenance(donor, _best_donor_session(donor), expected_gain))
    return replay


def _build_warm_start_context(
    *,
    status: str,
    tier: str,
    confidence: float,
    canonical_id: str,
    source: str,
    recipe: Mapping[str, Any] | None,
    config_donor: Mapping[str, Any] | None = None,
    config_donor_tier: str = "",
    config_donor_confidence: float | None = None,
) -> dict[str, Any]:
    """Build the model-facing WarmStartContext from a KB recipe row."""
    ctx: dict[str, Any] = {
        "status": status,
        "match": {
            "tier": tier,
            "confidence": float(confidence),
            "source": source,
            "canonical_id": canonical_id,
        },
        **_warm_start_priors(recipe),
    }
    donor = _replay_donor(recipe, config_donor)
    if donor is None:
        return ctx
    replay = _recommended_replay(
        donor,
        tier=config_donor_tier or "self",
        confidence=float(confidence if config_donor_confidence is None else config_donor_confidence),
    )
    if replay is not None:
        ctx["recommended_replay"] = replay
    return ctx


def _stack_version_tags(stack: _T0Stack) -> dict[str, str]:
    """The stack identifiers worth recording, skipping every dimension the probe could not read."""
    tags: dict[str, str] = {}
    for tag, probed in (("rocm_version", "rocm"), ("aiter_version", "aiter")):
        value = _text(stack.fingerprint.get(probed))
        if value and value != "unknown":
            tags[tag] = value
    digest = _text(stack.image_digest)
    if digest and digest != "unknown":
        tags["image_digest"] = digest
    return tags


def _build_t0_trace_extras(shared_state: Any, *, stack: _T0Stack, model_class: str) -> dict[str, Any]:
    """Assemble operator-traceability + workload-shape tags for the recipe extras (skip empty/zero)."""
    extras: dict[str, Any] = {}
    if model_class:
        extras["model_class"] = model_class
    architectures = getattr(shared_state, "model_architectures", None) or []
    if isinstance(architectures, list):
        arch_list = [_text(a) for a in architectures if _text(a)]
        if arch_list:
            extras["architectures"] = arch_list
    model_type = _text(getattr(shared_state, "model_type", ""))
    if model_type:
        extras["model_type"] = model_type
    extras.update(_stack_version_tags(stack))
    for src_key in ("claw_session_id", "sandbox_user_id"):
        value = _text(stack.extra.get(src_key))
        if value:
            extras[src_key] = value
    extras.update(workload_shape_tags(shared_state))
    return extras


def _resolve_framework(shared_state: Any, extra: Mapping[str, Any]) -> str:
    """The serving framework: what the caller passed, else SharedState, else the FRAMEWORK environment."""
    return _text(
        extra.get("framework_name")
        or extra.get("framework")
        or getattr(shared_state, "framework", "")
        or os.environ.get("FRAMEWORK", "")
    )


def _resolve_framework_version(shared_state: Any, framework: str, fingerprint: Mapping[str, Any]) -> str:
    """The framework version: SharedState first, then the stack fingerprint, then importlib auto-detect."""
    version = _text(getattr(shared_state, "framework_version", ""))
    if not version and framework in ("sglang", "vllm"):
        version = _text(fingerprint.get(framework))
        if version == "unknown":
            version = ""
    if not version and framework:
        version = detect_framework_version(framework)
    return version


def _resolve_t0_identity(shared_state: Any, *, workload: str, hw: str, stack: _T0Stack) -> _T0Identity:
    """Resolve the seven-tuple a T0 anchor is written and searched under."""
    from hyperloom.inference_optimizer.recipe_snapshot_constants import _architectures_slug

    framework = _resolve_framework(shared_state, stack.extra)
    architectures = getattr(shared_state, "model_architectures", None) or []
    search = _SearchIdentity(
        hardware=hw,
        framework=framework,
        model_type=_text(getattr(shared_state, "model_type", "")),
        arch_slug=_architectures_slug(architectures),
        framework_version=_resolve_framework_version(shared_state, framework, stack.fingerprint),
        precision=_text(getattr(shared_state, "precision", "")),
    )
    return _T0Identity(
        canonical_id=recipe_canonical_id(
            model=workload,
            hardware=search.hardware,
            framework_name=search.framework,
            framework_version=search.framework_version,
            precision=search.precision,
            model_type=search.model_type,
            architectures=architectures,
            scheme=(
                "agentx" if agentx_active(benchmark_mode=getattr(shared_state, "benchmark_mode", "")) else "inference"
            ),
        ),
        workload=workload,
        search=search,
        extras=_build_t0_trace_extras(shared_state, stack=stack, model_class=_text(stack.extra.get("model_class"))),
    )


def _persist_identity_dimensions(shared_state: Any, search: _SearchIdentity) -> None:
    """Store framework and framework_version on SharedState so CLOSE/KEEP derives the same canonical id."""
    if search.framework:
        shared_state.framework = search.framework
    if search.framework_version:
        shared_state.framework_version = search.framework_version


#: Written by the anchor itself; anything else on the row is a caller's and survives a stamp.
_ANCHOR_OWNED_FIELDS = frozenset(
    {
        "canonical_id",
        "version",
        "created_at",
        "updated_at",
        "model",
        "hardware",
        "framework",
        "framework_version",
        "precision",
        "best_config",
        "best_throughput",
        "what_worked",
        "what_failed",
        "remaining_gaps",
        "pitfalls",
        "lessons",
        "last_profiled",
        "stack_fingerprint",
        "sessions",
        "authority",
        "confidence",
        "evidence_refs",
        "provenance",
        "_field_sources",
        "_sources",
    }
)


def _preserved_extras(live: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep the fields on an authority row that the T0 anchor does not own."""
    return {k: v for k, v in (live or {}).items() if k not in _ANCHOR_OWNED_FIELDS}


def _live_authority_row(kb: RecipeKB, canonical_id: str) -> Mapping[str, Any]:
    """The store's current authority row for *canonical_id*, empty when it holds none."""
    try:
        return kb.get_authoritative_recipe(canonical_id=canonical_id) or {}
    except Exception as exc:  # noqa: BLE001 — defensive
        log.info("T0 anchor authority get_recipe non-fatal failure: %s", exc)
        return {}


def _stamped_fingerprint(live: Mapping[str, Any], fingerprint: Mapping[str, Any]) -> dict[str, str]:
    """The stack fingerprint to store: this round's readings over whatever the row already carried."""
    payload: dict[str, str] = dict(live.get("stack_fingerprint") or {})
    for stored, probed in (("vllm_version", "vllm"), ("aiter_commit", "aiter"), ("rocm_version", "rocm")):
        value = _text(fingerprint.get(probed))
        if value and value != "unknown":
            payload[stored] = value
    return payload


def _measured_fields(live: Mapping[str, Any]) -> dict[str, Any]:
    """The measured fields a T0 stamp carries over untouched; CLOSE is what rewrites them."""
    return {
        "best_config": dict(live.get("best_config") or {}),
        "best_throughput": float(live.get("best_throughput") or 0.0),
        "what_worked": list(live.get("what_worked") or []),
        "what_failed": list(live.get("what_failed") or []),
        "remaining_gaps": list(live.get("remaining_gaps") or []),
        "pitfalls": list(live.get("pitfalls") or []),
        "lessons": list(live.get("lessons") or []),
        "last_profiled": str(live.get("last_profiled") or ""),
        "sessions": list(live.get("sessions") or []),
    }


def _t0_provenance(session_id: str) -> dict[str, Any]:
    """Who wrote an authority row and when."""
    return {
        "source": "hyperloom-inference-optimizer",
        "generator": "t0_anchor",
        "generated_at": datetime.now(timezone.utc).isoformat(
            timespec="microseconds",
        ),
        "details": {"sid": session_id},
    }


def _stamp_local_authority_row(
    kb: RecipeKB,
    *,
    session_id: str,
    fingerprint: Mapping[str, Any],
    identity: _T0Identity,
) -> None:
    """Stamp the T0 anchor onto the local store's own authority row.

    A remote store publishes from CLOSE instead, so this is a no-op there.
    """
    if getattr(kb, "mode", "") == "remote":
        log.debug("T0 anchor: remote publication is owned by CLOSE")
        return
    live = _live_authority_row(kb, identity.canonical_id)
    try:
        kb.put_recipe(
            canonical_id=identity.canonical_id,
            model=identity.workload,
            hardware=identity.search.hardware,
            framework_name=identity.search.framework,
            framework_version=identity.search.framework_version,
            precision=identity.search.precision,
            **_measured_fields(live),
            stack_fingerprint=_stamped_fingerprint(live, fingerprint),
            extras={**_preserved_extras(live), **identity.extras},
            provenance=_t0_provenance(session_id),
        )
    except Exception:
        log.exception("T0 anchor put_recipe raised unexpectedly")


def _kb_serves_agentx(kb: Any, shared_state: Any) -> bool:
    """True unless an agentx workload is asking the local JSON store, which stays inference-only."""
    if not agentx_active(benchmark_mode=getattr(shared_state, "benchmark_mode", "")):
        return True
    return str(getattr(kb, "mode", "") or "") == "remote"


def _anchor_session_id(shared_state: Any, session_dir: Path) -> str:
    """The session id the anchor stamps: SharedState's own, else the session directory's name."""
    return _text(getattr(shared_state, "recipe_kb_session_id", "")) or session_dir.name


def _anchor_is_settled(shared_state: Any, *, session_id: str, resume: bool) -> bool:
    """True when an earlier pass already anchored this session and T0 has nothing left to redo."""
    return bool(session_id and not resume and _text(getattr(shared_state, "warm_start_ts", "")))


def _record_anchor_session(
    shared_state: Any,
    *,
    session_id: str,
    resume: bool,
    emit: Callable[[str], None],
) -> bool:
    """Claim the session on SharedState and report whether this call opened the warm-start interval."""
    if session_id:
        shared_state.recipe_kb_session_id = session_id
        if resume:
            emit(f"Recipe KB        : resumed session_id={session_id}")
    began_now = not getattr(shared_state, "warm_start_ts", "")
    if began_now:
        shared_state.warm_start_ts = datetime.now(timezone.utc).isoformat(
            timespec="seconds",
        )
    return began_now


def _open_warm_start_event(shared_state: Any, canonical_id: str) -> Any:
    """Open the warm_start event before the KB is consulted.

    The identity is resolved by now, and the event's open interval is what tells
    the audit hook which reads were T0's.
    """
    return _warm_start_event.make_warm_start_recorder(
        macro_cycle=getattr(shared_state, "macro_cycle", 0) or 0,
        requested_canonical_id=canonical_id,
        scope=_warm_start_scope(shared_state),
        start_time=str(getattr(shared_state, "warm_start_ts", "") or ""),
    )


def _warm_replay_target(shared_state: Any, identity: _T0Identity) -> _DonorTarget:
    """The workload a borrowed replay config has to be safe for; an agentx identity pins no sequence lengths."""
    agentx = identity.canonical_id.startswith("agentx:")
    return _DonorTarget(
        conc=getattr(shared_state, "conc", None),
        isl=None if agentx else getattr(shared_state, "isl", None),
        osl=None if agentx else getattr(shared_state, "osl", None),
    )


def _demoted_for_bare_anchor(match: _WarmMatch) -> _WarmMatch:
    """A matched row with neither replay material nor priors serves as a seed only."""
    if match.recipe and not _recipe_is_actionable(match.recipe):
        return _WarmMatch(match.recipe, "seed_only", 0.0)
    return match


def _warm_start_status(match: _WarmMatch) -> str:
    """T0's own hit / seed_only / miss verdict on what the KB offered."""
    if not match.recipe:
        return "miss"
    return "seed_only" if match.tier == "seed_only" else "hit"


def _warm_replay_donor(
    kb: RecipeKB,
    *,
    match: _WarmMatch,
    identity: _T0Identity,
    target: _DonorTarget,
) -> _ConfigDonor:
    """Pick the row whose champion config T0 recommends replaying.

    A true-self (identity ``exact``) champion always replays; a cross-model borrow must clear the trustworthiness
    gate before it becomes the donor.
    """
    from .remote_recipe import RECORD_KIND_HYPERLOOM_RECIPE

    recipe = match.recipe
    if not recipe or recipe.get("record_kind") == RECORD_KIND_HYPERLOOM_RECIPE:
        return _ConfigDonor(None, "", 0.0)
    if _has_replayable_config(recipe) and (
        match.tier == "exact" or _donor_is_trustworthy(recipe, identity=identity.search, target=target)
    ):
        return _ConfigDonor(recipe, "self", match.confidence)
    return _find_config_donor(kb, cid=identity.canonical_id, identity=identity.search, target=target)


def _warm_start_experience(recipe: Mapping[str, Any]) -> _WarmStartExperience:
    """The lessons and pitfalls the matched identity carries, read off its exact-identity history when it has one."""
    source = _advisory_source(recipe)
    return _WarmStartExperience(
        lessons=_experience_rows(source.get("lessons"), "statement", "lessons"),
        pitfalls=_experience_rows(source.get("pitfalls"), "description", "pitfalls"),
    )


def _save_experience_snapshot(path: Path, payload: Mapping[str, Any], label: str) -> bool:
    """Write one experience snapshot into the session directory; False when the filesystem refused it."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        log.warning("%s snapshot write failed: %s", label, exc)
        return False
    return True


def _publish_warm_start_experience(
    shared_state: Any,
    session_dir: Path,
    *,
    identity: _T0Identity,
    experience: _WarmStartExperience,
) -> None:
    """Snapshot the matched pitfalls and lessons, and hand the non-empty ones to SharedState."""
    header = {
        "workload": identity.workload,
        "hw": identity.search.hardware,
        "framework": identity.search.framework,
    }
    pitfalls_saved = _save_experience_snapshot(
        recipe_kb_pitfalls_json(session_dir),
        {**header, "pitfalls": experience.pitfalls},
        "warm_start_pitfalls",
    )
    if pitfalls_saved and experience.pitfalls:
        shared_state.warm_start_pitfalls = experience.pitfalls
    lessons_saved = _save_experience_snapshot(
        recipe_kb_lessons_json(session_dir),
        {**header, "lessons": experience.lessons},
        "warm_start_lessons",
    )
    if lessons_saved and experience.lessons:
        shared_state.warm_start_lessons = experience.lessons


def _save_anchored_state(shared_state: Any, session_dir: Path, *, session_id: str, workload: str) -> None:
    """Persist SharedState once the anchor and its warm-start lookup have both been recorded."""
    try:
        shared_state.save(session_dir)
    except Exception:
        log.exception(
            "Recipe KB T0: SharedState.save failed (sid=%s, workload=%s)",
            session_id,
            workload,
        )


def _anchor_banner(
    *,
    session_id: str,
    identity: _T0Identity,
    match: _WarmMatch,
    experience: _WarmStartExperience,
) -> str:
    """The one-line T0 summary for the status channel."""
    if match.recipe and match.confidence > 0.0:
        warm_label = f"hit:{match.tier}@{match.confidence:.2f}"
    elif match.recipe:
        warm_label = "seed_only"
    else:
        warm_label = "empty"
    return (
        f"Recipe KB        : session_id={session_id} "
        f"workload={identity.canonical_id} "
        f"(warm={warm_label}, "
        f"pitfalls={len(experience.pitfalls)}, "
        f"lessons={len(experience.lessons)})"
    )


def run_t0_anchor(
    kb: RecipeKB,
    shared_state: Any,
    *,
    workload: str,
    hw: str,
    image_digest: str = "",
    stack_fingerprint: Mapping[str, str] | None = None,
    extra_attrs: Mapping[str, Any] | None = None,
    resume: bool = False,
    on_status: Callable[[str], None] | None = None,
    session_dir: Path | None = None,
    save_state: bool = True,
) -> None:
    """Run the T0 recipe-snapshot anchor and seven-tuple warm-start search."""
    emit = on_status or _default_status_emitter
    if session_dir is None:
        raise ValueError("run_t0_anchor requires an explicit session_dir")
    if not _kb_serves_agentx(kb, shared_state):
        log.info("run_t0_anchor: local Recipe KB does not serve agentx identities")
        return

    sd = Path(session_dir)
    sid = _anchor_session_id(shared_state, sd)
    if _anchor_is_settled(shared_state, session_id=sid, resume=resume):
        shared_state.recipe_kb_session_id = sid
        emit(f"Recipe KB        : already anchored session_id={sid}")
        return
    began_now = _record_anchor_session(shared_state, session_id=sid, resume=resume, emit=emit)

    # Topology-aware hardware slug (multi-node appends ``_ws{world_size}``), resolved once so the cid, the stored
    # ``hardware`` field, and every warm-start tier below share an identical, isolated key.
    from hyperloom.orchestrator.actions.executors._multi_node_env import resolve_kb_topology

    stack = _T0Stack(
        extra=_mapping(extra_attrs),
        fingerprint=_mapping(stack_fingerprint),
        image_digest=image_digest,
    )
    identity = _resolve_t0_identity(
        shared_state,
        workload=_text(workload) or "unknown_model",
        hw=kb_hardware_slug(_text(hw) or "unknown_gpu", **resolve_kb_topology()),
        stack=stack,
    )
    _persist_identity_dimensions(shared_state, identity.search)

    # Only on the anchoring pass -- a resume that re-runs T0 must not reopen an event the first pass already settled.
    ws_recorder = _open_warm_start_event(shared_state, identity.canonical_id) if began_now else None
    _stamp_local_authority_row(kb, session_id=sid, fingerprint=stack.fingerprint, identity=identity)

    target = _warm_replay_target(shared_state, identity)
    match = _demoted_for_bare_anchor(
        _cascade_warm_start_search(
            kb,
            cid=identity.canonical_id,
            identity=identity.search,
            warm_prefer=_build_warm_prefer(shared_state, identity.search.framework_version),
            target=target,
        )
    )
    donor = _warm_replay_donor(kb, match=match, identity=identity, target=target)

    shared_state.warm_start_recipe = {
        "workload": identity.workload,
        "hw": identity.search.hardware,
        "tier": match.tier,
        "confidence": match.confidence,
        "recipe": match.recipe,
    }
    status = _warm_start_status(match)
    source = _warm_recipe_source(match.recipe, kb)
    shared_state.warm_start_context = _build_warm_start_context(
        config_donor=donor.row,
        config_donor_tier=donor.tier,
        config_donor_confidence=donor.confidence,
        status=status,
        tier=match.tier,
        confidence=match.confidence,
        canonical_id=identity.canonical_id,
        source=source,
        recipe=match.recipe or None,
    )

    experience = _warm_start_experience(match.recipe)
    _publish_warm_start_experience(shared_state, sd, identity=identity, experience=experience)

    # Settled before the state save so the event stands even if the save fails: the lookup happened either way.
    if ws_recorder is not None:
        _settle_warm_start_event(
            ws_recorder,
            match=match,
            status=status,
            source=source,
            canonical_id=identity.canonical_id,
            context=getattr(shared_state, "warm_start_context", None),
            experience=experience,
        )
    if save_state:
        _save_anchored_state(shared_state, sd, session_id=sid, workload=identity.workload)
    if began_now:
        emit(_anchor_banner(session_id=sid, identity=identity, match=match, experience=experience))


__all__ = [
    "run_t0_anchor",
]
