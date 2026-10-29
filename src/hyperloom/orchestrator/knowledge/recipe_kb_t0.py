# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared T0 (PRELUDE) Recipe KB anchor — KB warm-start only."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, NamedTuple

from packaging.version import InvalidVersion, Version

from hyperloom_common.coerce import to_float
from hyperloom_common.gpu_identity import AMD_GPU_BOARDS
from hyperloom.common.perf_metric import agentx_active
from hyperloom.orchestrator.knowledge.recipe_kb import (
    LocalRecipeStoreError,
    RecipeKB,
    RecipeUpsert,
    cid_to_path_components,
    recipe_canonical_id,
    recipe_framework_version,
)
from hyperloom.orchestrator.knowledge.workload_tags import workload_shape_tags
from hyperloom.inference_optimizer.breakdown.recorder import warm_start_event as _warm_start_event
from hyperloom.inference_optimizer.recipe_snapshot_constants import (
    RecipeIdentity,
    detect_framework_version,
    kb_hardware_slug,
)
from hyperloom.inference_optimizer.session.session_paths import (
    recipe_kb_lessons_json,
    recipe_kb_pitfalls_json,
)


log = logging.getLogger(__name__)


def _text(value: Any) -> str:
    """A stripped string for a value that may be ``None``."""
    return str(value or "").strip()


def _default_status_emitter(line: str) -> None:
    """Default ``on_status`` callback — log the banner line at INFO."""
    log.info("%s", line)


def _warm_recipe_source(kb: Any) -> str:
    """Return the source tag for a Recipe warm-start row."""
    return str(getattr(kb, "backend_name", "") or "recipe-kb")


def _experience_rows(raw: Any, required_key: str, label: str) -> list[dict[str, Any]]:
    """Normalise a recipe row's ``lessons`` / ``pitfalls`` into flat rows.

    The stored shape is flat — ``Recipe.to_dict`` writes ``{statement, ...}`` /
    ``{description, ...}``. The remote
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
        if not str(row.get(required_key) or "").strip():
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


def _recipe_is_actionable(row: Mapping[str, Any]) -> bool:
    """True when a warm recipe carries something worth replaying or priors."""
    if not isinstance(row, Mapping):
        return False
    if isinstance(row.get("replay_material_available"), bool):
        return bool(row.get("replayable") and row.get("replay_material_available"))
    if isinstance(row.get("replayable"), bool):
        return bool(row.get("replayable"))
    best_config = row.get("best_config")
    if isinstance(best_config, Mapping) and best_config:
        # An env-only or args-only config is still actionable.
        args = str(best_config.get("extra_server_args") or "").strip()
        envs = best_config.get("extra_envs") or best_config.get("envs") or {}
        if args or (isinstance(envs, Mapping) and envs):
            return True
    try:
        if float(row.get("best_throughput") or 0.0) > 0.0:
            return True
    except (TypeError, ValueError):
        pass
    for key in ("what_worked", "what_failed", "pitfalls", "lessons"):
        if row.get(key):
            return True
    return False


def _with_exact_history(
    candidate: Mapping[str, Any],
    exact: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Attach exact-identity priors without changing donor replay metrics."""
    merged = dict(candidate)
    if not isinstance(exact, Mapping):
        return merged
    merged["exact_history"] = {
        "canonical_id": str(exact.get("canonical_id") or ""),
        "view": dict(exact.get("view") or {}),
        **{
            key: list(exact.get(key) or [])
            for key in (
                "what_worked",
                "what_failed",
                "remaining_gaps",
                "lessons",
                "pitfalls",
                "sessions",
            )
        },
        "validated_gain_pct": exact.get("validated_gain_pct"),
        "gain_pct": exact.get("gain_pct"),
    }
    return merged


def _config_replay_args_envs(row: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    """Extract a replayable ``(args, envs)`` pair from a row's best_config."""
    best_config = row.get("best_config") if isinstance(row.get("best_config"), Mapping) else {}
    args = str(best_config.get("extra_server_args") or "").strip()
    envs = best_config.get("extra_envs") or {}
    if not isinstance(envs, Mapping):
        envs = {}
    return args, {str(k): str(v) for k, v in envs.items()}


def _has_replayable_config(row: Mapping[str, Any]) -> bool:
    """True when ``row`` carries a non-empty champion config (args OR envs)."""
    if not isinstance(row, Mapping):
        return False
    if isinstance(row.get("replay_material_available"), bool):
        # Remote candidate material is inspected by its owning AgentKB SDKs while isolated; T0 receives only this
        # capability bit.
        return bool(row.get("replay_material_available"))
    if isinstance(row.get("replay_config_available"), bool):
        return bool(row.get("replay_config_available"))
    args, envs = _config_replay_args_envs(row)
    return bool(args or envs)


def _best_session(row: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, float]:
    """The row's highest-gain session and that gain; ``-inf`` when the row records none."""
    best: Mapping[str, Any] | None = None
    best_gain = float("-inf")
    sessions = row.get("sessions")
    # A truthy non-iterable ``sessions`` would otherwise raise here and abort T0.
    if isinstance(sessions, list):
        for session in sessions:
            if not isinstance(session, Mapping):
                continue
            gain = to_float(session.get("gain_pct") or 0.0)
            if gain is not None and gain > best_gain:
                best, best_gain = session, gain
    return best, best_gain


def _max_session_gain(row: Mapping[str, Any]) -> float:
    """Return the MAX ``gain_pct`` across a row's sessions (fallback flat gain_pct)."""
    best = max(_best_session(row)[1], 0.0)
    if best <= 0:
        best = max(best, to_float(row.get("validated_gain_pct") or row.get("gain_pct"), 0.0))
    return best


def _donor_is_trustworthy(donor: Mapping[str, Any], identity: Mapping[str, Any], shape: Mapping[str, Any]) -> bool:
    """Gate a BORROWED (cross-model) warm-replay config donor."""
    if not isinstance(donor, Mapping):
        return False
    if isinstance(donor.get("replayable"), bool) and not donor.get("replayable"):
        return False
    if not _has_replayable_config(donor):
        return False
    # Require evidence of a real positive gain.
    if _max_session_gain(donor) <= 0:
        return False
    # Require a concrete architecture matching the target.
    from hyperloom.inference_optimizer.recipe_snapshot_constants import _architectures_slug

    donor_arch_value = donor.get("architectures")
    if donor_arch_value in (None, "", []):
        donor_arch_value = _candidate_dimension(donor, "architectures")
    donor_arch = _architectures_slug(donor_arch_value)
    donor_mt = _candidate_dimension(donor, "model_type")
    _unknown = {"", "unknown", "unknown_arch", "unknown_model_type"}
    if donor_arch in _unknown or donor_mt in _unknown:
        return False
    target_arch = _text(identity.get("architectures"))
    if target_arch and donor_arch and donor_arch != target_arch:
        return False
    tgt_mt = _text(identity.get("model_type")).lower()
    if tgt_mt and donor_mt and donor_mt != tgt_mt:
        return False

    def _shape_matches(target_val: Any, donor_key: str) -> bool:
        try:
            tv = int(target_val)
        except (TypeError, ValueError):
            return True
        if tv <= 0:
            return True
        try:
            dv = int(donor.get(donor_key))
        except (TypeError, ValueError):
            return False
        return dv > 0 and tv == dv

    return all(_shape_matches(shape.get(key), key) for key in ("conc", "isl", "osl"))


def _select_remote_candidate(kb: Any, row: Mapping[str, Any]) -> bool:
    """Ask remote adapters to materialize T0's accepted candidate."""
    select = getattr(kb, "select_candidate", None)
    if not callable(select):
        return True
    from .remote_recipe import KBStoreError

    try:
        return bool(select(row))
    except (KBStoreError, OSError, TypeError, ValueError) as exc:
        log.warning(
            "warm-start candidate materialization failed for %s: %s",
            row.get("canonical_id"),
            exc,
        )
        return False


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
    context: Mapping[str, Any],
    recipe: Mapping[str, Any] | None,
    experience: Mapping[str, list[dict[str, Any]]],
) -> None:
    """Close the warm_start event on the context the lookup produced."""
    match = context.get("match") or {}
    status = str(context.get("status") or "")
    matched = None
    if status in _warm_start_event.MATCHED_STATUSES and recipe:
        replay = context.get("recommended_replay")
        matched = _warm_start_event.matched_block(
            match=match,
            recipe=recipe,
            experience=experience,
            expected_gain_pct=replay.get("expected_gain_pct") if isinstance(replay, Mapping) else None,
        )
    recorder.finish(match_status=status, matched=matched)


class WarmMatch(NamedTuple):
    """A Recipe KB row a warm-start tier matched, with that tier and its confidence."""

    recipe: Mapping[str, Any] | None
    tier: str
    confidence: float


def _build_warm_start_context(
    *,
    match: WarmMatch,
    canonical_id: str,
    source: str,
    config_donor: WarmMatch | None = None,
) -> dict[str, Any]:
    """Build the model-facing WarmStartContext from a KB recipe row."""
    from .remote_recipe import RECORD_KIND_HYPERLOOM_RECIPE

    recipe, tier, confidence = match
    # Priors ride the identity match independent of replay config, and come from the exact identity when the
    # cascade carried one across from a borrowed row.
    priors = recipe.get("exact_history") or recipe
    if config_donor is None:
        config_donor = WarmMatch(None, "", confidence)
    donor, config_donor_tier, config_donor_confidence = config_donor
    ctx: dict[str, Any] = {
        # A cascade that matched nothing reports tier ``miss``, so the tier is the status.
        "status": tier if tier in ("miss", "seed_only") else "hit",
        "match": {
            "tier": tier,
            "confidence": float(confidence),
            "source": source,
            "canonical_id": canonical_id,
        },
        **{
            field: list(priors.get(key) or [])
            for field, key in (
                ("proven_prior", "what_worked"),
                ("do_not_repeat", "what_failed"),
                ("lessons", "lessons"),
                ("pitfalls", "pitfalls"),
            )
        },
    }
    if recipe.get("record_kind") == RECORD_KIND_HYPERLOOM_RECIPE:
        return ctx
    # Replay config comes from the donor, or the identity recipe as self-donor.
    if donor is None:
        if not _has_replayable_config(recipe):
            return ctx
        donor, config_donor_tier = recipe, "self"
    args, envs = _config_replay_args_envs(donor)
    if not (args or envs):
        return ctx
    expected_gain = to_float(donor.get("validated_gain_pct"), 0.0)
    if expected_gain <= 0:
        expected_gain = _max_session_gain(donor)
    session = _best_session(donor)[0] or {}
    family_tags = donor.get("family_tags") or donor.get("model_architectures")
    donor_cid = _text(donor.get("canonical_id"))
    ctx["recommended_replay"] = {
        "extra_server_args": args,
        "extra_envs": envs,
        "expected_gain_pct": expected_gain,
        "best_throughput": to_float(donor.get("best_throughput"), 0.0),
        "config_source": donor_cid,
        "config_tier": config_donor_tier,
        "config_confidence": float(config_donor_confidence),
        "donor_canonical_id": donor_cid,
        "donor_model": _text(donor.get("model")),
        "donor_session_id": _text(session.get("session_id")),
        "donor_family_tags": [str(tag) for tag in family_tags] if isinstance(family_tags, (list, tuple, set)) else [],
        "donor_gain_pct": expected_gain,
        "donor_breakdown_link": _text(
            session.get("breakdown_link") or session.get("session_breakdown_url") or donor.get("breakdown_link")
        ),
    }
    return ctx


def _build_t0_trace_extras(
    shared_state: Any,
    *,
    extra: "Mapping[str, Any]",
    fp: "Mapping[str, Any]",
    image_digest: str,
) -> dict[str, Any]:
    """Assemble operator-traceability + workload-shape tags for the recipe extras (skip empty/zero)."""
    _extras: dict[str, Any] = {}
    _model_class = str(extra.get("model_class") or "").strip()
    if _model_class:
        _extras["model_class"] = _model_class
    _architectures = getattr(shared_state, "model_architectures", None) or []
    _arch_list = [str(a).strip() for a in _architectures if str(a or "").strip()]
    if _arch_list:
        _extras["architectures"] = _arch_list
    _model_type = str(getattr(shared_state, "model_type", "") or "").strip()
    if _model_type:
        _extras["model_type"] = _model_type
    for tag, probed in (("rocm_version", "rocm"), ("aiter_version", "aiter")):
        probed_value = str(fp.get(probed) or "").strip()
        if probed_value and probed_value != "unknown":
            _extras[tag] = probed_value
    if image_digest and image_digest != "unknown":
        _extras["image_digest"] = str(image_digest).strip()
    for src_key in ("claw_session_id", "sandbox_user_id"):
        v = str(extra.get(src_key) or "").strip()
        if v:
            _extras[src_key] = v
    _extras.update(workload_shape_tags(shared_state))
    return _extras


#: SKU -> ISA family. Derived from the board table, so a new board becomes a
#: fallback donor for its own ISA in the commit that adds it.
_GPU_ISA_BY_SKU = {sku: board.gfx for sku, board in AMD_GPU_BOARDS.items()}
#: ``ep`` and the partition mode suffix at any node count, so they appear both inside a cluster suffix and on their
#: own. Kept as a named fragment rather than repeated, so the two forms cannot drift.
_SHAPE_SUFFIX = r"(?:_ep[1-9]\d*)?(?:_(?:dpx|qpx|cpx))?"
#: A single-node shape suffix has to parse here too, or ``_hardware_fallback_values`` stops offering the same-ISA
#: SKUs for exactly the rows these suffixes were added for. The alternation keeps ``_ws`` mandatory for the cluster
#: form, so an unrecognised suffix still fails to parse instead of being mistaken for a backend name.
_TOPOLOGY_SUFFIX_RE = re.compile(
    r"(?:_ws[1-9]\d*(?:_pd[1-9]\d*p[1-9]\d*d)?"
    rf"(?:_tp[1-9]\d*)?{_SHAPE_SUFFIX}(?:_[a-z0-9-]+)?"
    rf"|{_SHAPE_SUFFIX})$"
)
#: The seven-tuple dimensions a canonical id is built from, in canonical-id order.
_CID_DIMENSIONS = (
    "model",
    "hardware",
    "framework_name",
    "model_type",
    "architectures",
    "framework_version",
    "precision",
)


def _cid_dimensions(cid: str) -> dict[str, str]:
    """The seven-tuple a canonical id spells out, empty when it does not parse."""
    try:
        return dict(zip(_CID_DIMENSIONS, cid_to_path_components(cid)))
    except ValueError:
        return {}


def _parse_hardware_topology(hardware: str) -> tuple[str, str, str] | None:
    """Return ``(SKU, ISA family, topology suffix)`` for a known GPU."""
    value = str(hardware or "").strip().lower()
    for sku, family in _GPU_ISA_BY_SKU.items():
        if value == sku:
            return sku, family, ""
        if value.startswith(f"{sku}_"):
            suffix = value[len(sku) :]
            if _TOPOLOGY_SUFFIX_RE.fullmatch(suffix):
                return sku, family, suffix
    return None


def _hardware_fallback_values(hardware: str) -> list[str]:
    """List same-ISA SKUs carrying the target's exact topology suffix."""
    parsed = _parse_hardware_topology(hardware)
    if parsed is None:
        value = str(hardware or "").strip().lower()
        return [value] if value else []
    _sku, family, suffix = parsed
    return [f"{sku}{suffix}" for sku, sku_family in _GPU_ISA_BY_SKU.items() if sku_family == family]


def _hardware_is_compatible(target: str, candidate: str) -> bool:
    """Accept exact hardware or a known same-ISA SKU with exact topology."""
    target_value = str(target or "").strip().lower()
    candidate_value = str(candidate or "").strip().lower()
    if candidate_value == target_value:
        return bool(target_value)
    target_parsed = _parse_hardware_topology(target_value)
    candidate_parsed = _parse_hardware_topology(candidate_value)
    return bool(
        target_parsed
        and candidate_parsed
        and target_parsed[1] == candidate_parsed[1]
        and target_parsed[2] == candidate_parsed[2]
    )


def _framework_semver(version: str) -> Version | None:
    """Parse a framework's PEP 440 version."""
    value = str(version or "").strip()
    if not value:
        return None
    try:
        return Version(value)
    except InvalidVersion:
        return None


def _framework_version_is_compatible(target: str, candidate: str) -> bool:
    """Accept the same release, or the nearest non-newer PEP 440 release in its line, whichever build each names."""
    target_value = recipe_framework_version(target)
    candidate_value = recipe_framework_version(candidate)
    if candidate_value == target_value:
        return bool(target_value)
    target_key = _framework_semver(target_value)
    candidate_key = _framework_semver(candidate_value)
    return bool(
        target_key
        and candidate_key
        and candidate_key.release[:2] == target_key.release[:2]
        and candidate_key <= target_key
    )


def _warm_search_tiers(
    identity: Mapping[str, Any],
) -> tuple[
    tuple[str, float, dict[str, str], list[str] | None, bool],
    ...,
]:
    """Build the shared ordered seven-tuple degradation tiers."""
    labels = {key: _text(identity.get(key)) for key in _CID_DIMENSIONS if key != "model"}
    hardware_values = _hardware_fallback_values(labels["hardware"])
    return (
        ("same_arch_class", 0.95, labels, None, False),
        (
            "same_gpu_isa",
            0.85,
            {key: value for key, value in labels.items() if key != "hardware"},
            hardware_values,
            False,
        ),
        (
            "compatible_framework_version",
            0.72,
            {key: value for key, value in labels.items() if key not in ("hardware", "framework_version")},
            hardware_values,
            True,
        ),
    )


def _candidate_dimension(row: Mapping[str, Any], key: str) -> str:
    """Read one seven-tuple dimension from metadata or canonical id."""
    value = row.get(key)
    if value not in (None, ""):
        return str(value).strip().lower()
    return str(_cid_dimensions(str(row.get("canonical_id") or "")).get(key) or "").strip().lower()


def _rank_warm_candidates(
    rows: list[Mapping[str, Any]],
    *,
    target_framework_version: str,
) -> list[Mapping[str, Any]]:
    """Rank framework proximity, validated gain, then recency."""
    target_version = str(target_framework_version or "")
    ranked = list(rows)
    ranked.sort(
        key=lambda row: str(row.get("updated_at") or ""),
        reverse=True,
    )
    ranked.sort(key=_max_session_gain, reverse=True)
    ranked.sort(
        key=lambda row: (
            (
                _framework_semver(_candidate_dimension(row, "framework_version"))
                if _framework_version_is_compatible(
                    target_version,
                    _candidate_dimension(row, "framework_version"),
                )
                else None
            )
            or Version("0.dev0")
        ),
        reverse=True,
    )
    return ranked


def _search_warm_candidates(
    kb: Any,
    *,
    labels: dict[str, str],
    hardware_in: list[str] | None = None,
) -> list[Mapping[str, Any]]:
    """Run exact identity search, keeping only mapping rows."""
    kwargs: dict[str, Any] = {"label_match": labels, "limit": 100}
    if hardware_in is not None and getattr(kb, "mode", "") == "remote":
        kwargs["hardware_in"] = hardware_in
    rows = kb.search(**kwargs)
    return [row for row in (rows or []) if isinstance(row, Mapping)]


def _remote_candidate_matches(
    kb: Any,
    row: Mapping[str, Any],
    *,
    labels: Mapping[str, str],
) -> bool:
    """Validate exact server dimensions and current replay metadata."""
    if getattr(kb, "mode", "") != "remote":
        return True
    if (
        row.get("replayable") is not True
        or str(row.get("view_source") or "") != "current"
        or row.get("replay_material_available") is not True
    ):
        return False
    return all(_candidate_dimension(row, key) == str(value).strip().lower() for key, value in labels.items())


def _tier_matches(
    kb: Any,
    tiers: tuple[tuple[str, float, dict[str, str], list[str] | None, bool], ...],
    identity: Mapping[str, Any],
    shape: Mapping[str, Any],
) -> Iterator[WarmMatch]:
    """Walk the degradation tiers, yielding each tier's trustworthy candidates best first."""
    seen = {_text(identity.get("canonical_id"))}
    hardware = _text(identity.get("hardware"))
    version = _text(identity.get("framework_version"))
    precision = _text(identity.get("precision")).lower()
    for tier, confidence, labels, hardware_in, relax_framework_version in tiers:
        rows = _search_warm_candidates(kb, labels=labels, hardware_in=hardware_in)
        usable: list[Mapping[str, Any]] = []
        for candidate in rows:
            candidate_id = str(candidate.get("canonical_id") or "")
            if not candidate_id or candidate_id in seen:
                continue
            seen.add(candidate_id)
            if not _remote_candidate_matches(kb, candidate, labels=labels):
                continue
            if hardware_in is not None and not _hardware_is_compatible(
                hardware,
                _candidate_dimension(candidate, "hardware"),
            ):
                continue
            if _candidate_dimension(candidate, "precision") != precision:
                continue
            if relax_framework_version and not _framework_version_is_compatible(
                version,
                _candidate_dimension(candidate, "framework_version"),
            ):
                continue
            if _donor_is_trustworthy(candidate, identity, shape):
                usable.append(candidate)
        if rows and not usable:
            log.info("warm-start tier %s rejected all %d candidates", tier, len(rows))
        for candidate in _rank_warm_candidates(usable, target_framework_version=version):
            yield WarmMatch(candidate, tier, confidence)


def _cascade_warm_start_search(
    kb: "RecipeKB",
    *,
    cid: str,
    identity: Mapping[str, Any],
    shape: Mapping[str, Any],
) -> WarmMatch:
    """Resolve exact plus cumulative four-tier seven-tuple fallbacks."""
    exact_history: Mapping[str, Any] | None = None
    exact = kb.get_recipe(canonical_id=cid)
    if isinstance(exact, Mapping) and exact and str(exact.get("canonical_id") or "") == cid:
        if _recipe_is_actionable(exact):
            return WarmMatch(dict(exact), "exact", 1.0)
        exact_history = exact

    # An unparseable cid — every agentx id is one — keeps the dimensions resolved from SharedState.
    stored = {**identity, **_cid_dimensions(cid)}
    for candidate, tier, confidence in _tier_matches(kb, _warm_search_tiers(stored), stored, shape):
        if _select_remote_candidate(kb, candidate):
            return WarmMatch(_with_exact_history(candidate, exact_history), tier, confidence)
    if exact_history is not None:
        return WarmMatch(dict(exact_history), "exact", 1.0)
    return WarmMatch({}, "miss", 0.0)


def _find_config_donor(
    kb: Any,
    identity: Mapping[str, Any],
    shape: Mapping[str, Any],
) -> WarmMatch:
    """Borrow a replayable config through the standard degradation tiers."""
    tiers = tuple(
        (
            tier,
            confidence,
            {
                key: value
                for key, value in labels.items()
                if value and value not in ("unknown_model_type", "unknown_arch")
            },
            hardware_in,
            relax_framework_version,
        )
        for tier, confidence, labels, hardware_in, relax_framework_version in _warm_search_tiers(identity)
    )
    return next(_tier_matches(kb, tiers, identity, shape), WarmMatch(None, "", 0.0))


def _warm_replay_donor(
    kb: "RecipeKB",
    match: WarmMatch,
    identity: Mapping[str, Any],
    shape: Mapping[str, Any],
) -> WarmMatch:
    """Pick the row whose champion config T0 recommends replaying."""
    from .remote_recipe import RECORD_KIND_HYPERLOOM_RECIPE

    recipe, tier, confidence = match
    if not recipe or recipe.get("record_kind") == RECORD_KIND_HYPERLOOM_RECIPE:
        return WarmMatch(None, "", 0.0)
    # A true-self (identity ``exact``) champion always replays; a cross-model borrow must clear the trustworthiness
    # gate before it becomes the donor.
    if _has_replayable_config(recipe) and (tier == "exact" or _donor_is_trustworthy(recipe, identity, shape)):
        return WarmMatch(recipe, "self", confidence)
    return _find_config_donor(kb, identity, shape)


def _resolve_t0_identity(
    shared_state: Any,
    *,
    workload: str,
    hw: str,
    extra: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the seven-tuple a T0 anchor is written and searched under."""
    from hyperloom.inference_optimizer.recipe_snapshot_constants import _architectures_slug
    from hyperloom.orchestrator.actions.executors._multi_node_env import resolve_kb_topology

    model = _text(workload) or "unknown_model"
    # Topology-aware hardware slug (multi-node appends ``_ws{world_size}``), resolved once so the cid, the stored
    # ``hardware`` field, and every warm-start tier share an identical, isolated key.
    hardware = kb_hardware_slug(_text(hw) or "unknown_gpu", resolve_kb_topology())
    framework = _text(
        extra.get("framework_name")
        or extra.get("framework")
        or getattr(shared_state, "framework", "")
        or os.environ.get("FRAMEWORK", "")
    )
    # framework_version: SharedState > stack_fingerprint > importlib auto-detect.
    version = _text(getattr(shared_state, "framework_version", ""))
    if not version and framework in ("sglang", "vllm"):
        version = _text(fingerprint.get(framework))
        if version == "unknown":
            version = ""
    if not version and framework:
        version = detect_framework_version(framework)
    architectures = getattr(shared_state, "model_architectures", None) or []
    identity: dict[str, Any] = {
        "model": model,
        "hardware": hardware,
        "framework_name": framework,
        "model_type": _text(getattr(shared_state, "model_type", "")),
        "architectures": _architectures_slug(architectures),
        "framework_version": recipe_framework_version(version),
        "precision": _text(getattr(shared_state, "precision", "")),
        # The resolved version with its build suffix, which the session state keeps.
        "session_framework_version": version,
    }
    identity["canonical_id"] = recipe_canonical_id(
        RecipeIdentity(
            model=model,
            hardware=hardware,
            framework_name=framework,
            framework_version=identity["framework_version"],
            precision=identity["precision"],
            model_type=identity["model_type"],
            architectures=architectures,
        ),
        scheme=("agentx" if agentx_active(benchmark_mode=getattr(shared_state, "benchmark_mode", "")) else "inference"),
    )
    return identity


def _stamp_local_authority_row(
    kb: RecipeKB,
    *,
    session_id: str,
    fingerprint: Mapping[str, Any],
    identity: Mapping[str, Any],
    extras: Mapping[str, Any],
) -> None:
    """Stamp the T0 anchor onto the local store's own authority row."""
    if getattr(kb, "mode", "") == "remote":
        log.debug("T0 anchor: remote publication is owned by CLOSE")
        return
    # Read-modify-write the selected store's exact authority row so the stamp does not clobber fields or trigger a
    # broad remote warm-start scan.
    live = kb.get_authoritative_recipe(canonical_id=identity["canonical_id"]) or {}
    # Stack fingerprint — preserve prior values not stamped this round.
    sfp_payload: dict[str, str] = dict(live.get("stack_fingerprint") or {})
    for stored, probed in (("vllm_version", "vllm"), ("aiter_commit", "aiter"), ("rocm_version", "rocm")):
        value = _text(fingerprint.get(probed))
        if value and value != "unknown":
            sfp_payload[stored] = value
    try:
        upsert = RecipeUpsert(
            canonical_id=identity["canonical_id"],
            model=identity["model"],
            hardware=identity["hardware"],
            framework_name=identity["framework_name"],
            framework_version=identity["framework_version"],
            precision=identity["precision"],
            best_config=dict(live.get("best_config") or {}),
            best_throughput=float(live.get("best_throughput") or 0.0),
            what_worked=list(live.get("what_worked") or []),
            what_failed=list(live.get("what_failed") or []),
            remaining_gaps=list(live.get("remaining_gaps") or []),
            pitfalls=list(live.get("pitfalls") or []),
            lessons=list(live.get("lessons") or []),
            last_profiled=str(live.get("last_profiled") or ""),
            stack_fingerprint=sfp_payload,
            sessions=list(live.get("sessions") or []),
            # put_recipe sets every field it owns before splatting extras with setdefault, so the row's own values
            # win and only what the anchor does not own survives from *live*.
            extras={**live, **extras},
            provenance={
                "source": "hyperloom-inference-optimizer",
                "generator": "t0_anchor",
                "generated_at": datetime.now(timezone.utc).isoformat(
                    timespec="microseconds",
                ),
                "details": {"sid": session_id},
            },
        )
        kb.put_recipe(upsert)
    except (LocalRecipeStoreError, OSError, TypeError, ValueError):
        log.exception("T0 anchor put_recipe raised unexpectedly")


def _publish_warm_start_experience(
    shared_state: Any,
    session_dir: Path,
    *,
    identity: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Snapshot the matched pitfalls and lessons, and hand the non-empty ones to SharedState."""
    header = {
        "workload": identity["model"],
        "hw": identity["hardware"],
        "framework": identity["framework_name"],
    }
    experience: dict[str, list[dict[str, Any]]] = {}
    for key, required_key, json_path in (
        ("pitfalls", "description", recipe_kb_pitfalls_json),
        ("lessons", "statement", recipe_kb_lessons_json),
    ):
        rows = _experience_rows(context.get(key), required_key, key)
        experience[key] = rows
        path = json_path(session_dir)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({**header, key: rows}, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning("warm_start_%s snapshot write failed: %s", key, exc)
            continue
        if rows:
            setattr(shared_state, f"warm_start_{key}", rows)
    return experience


def _warm_start_lookup(
    kb: RecipeKB,
    shared_state: Any,
    identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the warm-start search and project what it found into the model-facing context."""
    cid = str(identity["canonical_id"])
    shape: dict[str, Any] = {"conc": getattr(shared_state, "conc", None)}
    # An agentx identity pins no sequence lengths, so a donor's are not compared against it.
    if not cid.startswith("agentx:"):
        shape.update(isl=getattr(shared_state, "isl", None), osl=getattr(shared_state, "osl", None))
    recipe, tier, confidence = _cascade_warm_start_search(kb, cid=cid, identity=identity, shape=shape)
    # A bare T0 anchor (no best_config) demotes to seed_only.
    if recipe and not _recipe_is_actionable(recipe):
        tier, confidence = "seed_only", 0.0
    # Config-donor decoupling: the identity match supplies the priors; a champion config is borrowed from the
    # nearest same-arch sibling when the match carries none.
    identity_match = WarmMatch(recipe, tier, confidence)
    donor = _warm_replay_donor(kb, identity_match, identity, shape)
    match = {
        "workload": identity["model"],
        "hw": identity["hardware"],
        "tier": tier,
        "confidence": confidence,
        "recipe": recipe,
    }
    return match, _build_warm_start_context(
        match=identity_match,
        canonical_id=cid,
        source=_warm_recipe_source(kb),
        config_donor=donor,
    )


@dataclass(frozen=True)
class T0AnchorRequest:
    """The workload a T0 anchor stamps and warm-starts, and whether the session resumes."""

    workload: str
    hw: str
    image_digest: str = ""
    stack_fingerprint: Mapping[str, str] | None = None
    extra_attrs: Mapping[str, Any] | None = None
    resume: bool = False


def run_t0_anchor(
    kb: RecipeKB,
    shared_state: Any,
    request: T0AnchorRequest,
    *,
    on_status: Callable[[str], None] = _default_status_emitter,
    session_dir: Path,
) -> None:
    """Run the T0 recipe-snapshot anchor and seven-tuple warm-start search."""
    # The local JSON store remains inference-only.
    if getattr(kb, "mode", "") != "remote" and agentx_active(
        benchmark_mode=getattr(shared_state, "benchmark_mode", "")
    ):
        log.info("run_t0_anchor: local Recipe KB does not serve agentx identities")
        return

    sd = Path(session_dir)
    sid = _text(getattr(shared_state, "recipe_kb_session_id", "")) or sd.name
    shared_state.recipe_kb_session_id = sid
    # Short-circuit when already anchored; resume=True bypasses.
    if not request.resume and _text(getattr(shared_state, "warm_start_ts", "")):
        on_status(f"Recipe KB        : already anchored session_id={sid}")
        return
    if request.resume:
        on_status(f"Recipe KB        : resumed session_id={sid}")
    began_now = not getattr(shared_state, "warm_start_ts", "")
    if began_now:
        shared_state.warm_start_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    extra: Mapping[str, Any] = request.extra_attrs or {}
    fingerprint: Mapping[str, Any] = request.stack_fingerprint or {}
    identity = _resolve_t0_identity(
        shared_state, workload=request.workload, hw=request.hw, extra=extra, fingerprint=fingerprint
    )
    cid = str(identity["canonical_id"])
    # Each is empty only where SharedState already held nothing, and CLOSE/KEEP derives the same cid from them.
    shared_state.framework = identity["framework_name"]
    shared_state.framework_version = identity["session_framework_version"]

    # Opened before the KB is consulted: the event's open interval is what tells the audit hook which reads were
    # T0's. Only on the anchoring pass -- a resume that re-runs T0 must not reopen a settled event.
    ws_recorder = None
    if began_now:
        ws_recorder = _warm_start_event.make_warm_start_recorder(
            macro_cycle=getattr(shared_state, "macro_cycle", 0) or 0,
            requested_canonical_id=cid,
            scope=_warm_start_scope(shared_state),
            start_time=shared_state.warm_start_ts,
        )
    # T0 only stamps metadata (best_config preserved, rewritten at CLOSE).
    extras = _build_t0_trace_extras(shared_state, extra=extra, fp=fingerprint, image_digest=request.image_digest)
    _stamp_local_authority_row(kb, session_id=sid, fingerprint=fingerprint, identity=identity, extras=extras)
    match, context = _warm_start_lookup(kb, shared_state, identity)
    shared_state.warm_start_recipe = match
    shared_state.warm_start_context = context
    experience = _publish_warm_start_experience(shared_state, sd, identity=identity, context=context)

    # Settled before the state save so the event stands even if the save fails: the lookup happened either way.
    if ws_recorder is not None:
        _settle_warm_start_event(ws_recorder, context=context, recipe=match["recipe"], experience=experience)
    if began_now:
        hit_label = f"hit:{match['tier']}@{match['confidence']:.2f}"
        warm_label = {"hit": hit_label, "seed_only": "seed_only"}.get(context["status"], "empty")
        on_status(
            f"Recipe KB        : session_id={sid} workload={cid} (warm={warm_label}, "
            f"pitfalls={len(experience['pitfalls'])}, lessons={len(experience['lessons'])})"
        )
    try:
        shared_state.save(sd)
    except OSError:
        log.exception("Recipe KB T0: SharedState.save failed (sid=%s, workload=%s)", sid, identity["model"])


__all__ = [
    "T0AnchorRequest",
    "run_t0_anchor",
]
