# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared T0 (PRELUDE) Recipe KB anchor — KB warm-start only."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from packaging.version import InvalidVersion, Version

from hyperloom.orchestrator.knowledge.recipe_kb import (
    RecipeKB,
    cid_to_path_components,
    recipe_canonical_id,
)
from hyperloom.inference_optimizer.recipe_snapshot_constants import detect_framework_version, kb_hardware_slug
from hyperloom.inference_optimizer.session.session_paths import (
    recipe_kb_lessons_json,
    recipe_kb_pitfalls_json,
    recipe_kb_warm_json,
)


log = logging.getLogger(__name__)


def _default_status_emitter(line: str) -> None:
    """Default ``on_status`` callback — log the banner line at INFO.

    Args:
        line (str): The status banner line to emit.
    """
    log.info("%s", line)


# Numeric workload knobs threaded into the KB ``prefer`` block so a
# closer-workload recipe is reranked first.
_PREFER_NUMERIC_ATTRS: tuple[str, ...] = (
    "tp",
    "ep",
    "conc",
    "isl",
    "osl",
    "max_model_len",
)


def _build_warm_prefer(shared_state: Any, framework_version: str) -> dict[str, Any]:
    """Assemble the ``prefer`` similarity hints from SharedState.

    Only non-empty values are included; the dispatcher skips absent
    fields. ``quant_scheme`` / ``workload_mode`` ride on the per-baseline
    ``baseline_workload_extra`` map when present.

    Args:
        shared_state: The live SharedState carrying workload knobs.
        framework_version: The resolved framework version, included when set.

    Returns:
        The ``prefer`` similarity-hint dict (non-empty fields only).
    """
    prefer: dict[str, Any] = {}
    for attr in _PREFER_NUMERIC_ATTRS:
        val = getattr(shared_state, attr, None)
        if val not in (None, "", 0):
            prefer[attr] = val
    fv = str(framework_version or "").strip()
    if fv:
        prefer["framework_version"] = fv
    wl_extra = getattr(shared_state, "baseline_workload_extra", None) or {}
    if isinstance(wl_extra, Mapping):
        for key in ("quant_scheme", "workload_mode"):
            v = str(wl_extra.get(key) or "").strip()
            if v:
                prefer[key] = v
    return prefer


def _warm_recipe_source(row: Mapping[str, Any] | None, kb: Any) -> str:
    """Return the source tag for a Recipe warm-start row.

    Args:
        row: Unused; retained for call-site compatibility.
        kb: Recipe backend or read-only compatibility adapter.

    Returns:
        A stable backend source tag.
    """
    del row
    return str(getattr(kb, "backend_name", "") or "recipe-kb")


def _recipe_is_actionable(row: Mapping[str, Any]) -> bool:
    """True when a warm recipe carries something worth replaying or priors.

    A View that reports ``replayable`` / ``replay_material_available`` is
    trusted verbatim. Otherwise a bare draft anchor (identity + tracing tags
    but no champion / experiential lists) is NOT actionable, so warm-replay
    never applies an empty config or starves the specialist prompt.

    Args:
        row: A warm recipe row to inspect.

    Returns:
        ``True`` when the row is replayable, or carries a usable config,
        positive throughput, or any experiential list worth replaying.
    """
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
    """Extract a replayable ``(args, envs)`` pair from a row's best_config.

    Reads the canonical ``extra_server_args`` field and the nested env map
    under ``extra_envs`` / ``envs``. Returns empty values when nothing
    replayable is present.
    """
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
        # Remote candidate material is inspected by its owning AgentKB SDKs
        # while isolated; T0 receives only this capability bit.
        return bool(row.get("replay_material_available"))
    if isinstance(row.get("replay_config_available"), bool):
        return bool(row.get("replay_config_available"))
    args, envs = _config_replay_args_envs(row)
    return bool(args or envs)


def _max_session_gain(row: Mapping[str, Any]) -> float:
    """Return the MAX ``gain_pct`` across a row's sessions (fallback flat gain_pct)."""
    best = 0.0
    sessions = row.get("sessions")
    if isinstance(sessions, list):
        for s in sessions:
            if not isinstance(s, Mapping):
                continue
            try:
                g = float(s.get("gain_pct") or 0.0)
            except (TypeError, ValueError):
                continue
            best = max(best, g)
    if best <= 0:
        try:
            best = max(best, float(row.get("validated_gain_pct") or row.get("gain_pct") or 0.0))
        except (TypeError, ValueError):
            pass
    return best


def _donor_is_trustworthy(
    donor: Mapping[str, Any],
    *,
    target_arch_slug: str,
    target_model_type: str,
    target_conc: Any = None,
    target_isl: Any = None,
    target_osl: Any = None,
) -> bool:
    """Gate a BORROWED (cross-model) warm-replay config donor.

    Borrowing a champion config on a loose same-arch match empirically produced
    near-zero or negative replay gains: cross-architecture configs, donors whose
    architecture is ``unknown``, and donors whose own validated gain was zero
    ("reproduce baseline" no-ops). A borrowed donor must therefore satisfy ALL:

    * a replayable champion config (args or envs);
    * a positive validated/session gain (rejects zero-gain donors);
    * a concrete architecture (not ``unknown``) equal to the target's;
    * complete, matching workload-shape fields when the target declares them.

    This is only applied to BORROWED donors — a true-self (identity ``exact``)
    replay is never gated, preserving the "reproduce my own champion" contract.

    Args:
        donor: Candidate donor recipe row.
        target_arch_slug: Architectures slug of the workload being optimized.
        target_model_type: Model type of the workload being optimized.
        target_conc: Target concurrency (optional shape hint).
        target_isl: Target input sequence length (optional shape hint).
        target_osl: Target output sequence length (optional shape hint).

    Returns:
        ``True`` when the donor is safe to borrow for warm-replay.
    """
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
    if target_arch_slug and donor_arch and donor_arch != target_arch_slug:
        return False
    tgt_mt = str(target_model_type or "").strip().lower()
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

    if not all(
        (
            _shape_matches(target_conc, "conc"),
            _shape_matches(target_isl, "isl"),
            _shape_matches(target_osl, "osl"),
        )
    ):
        return False
    return True


def _select_remote_candidate(kb: Any, row: Mapping[str, Any]) -> bool:
    """Ask remote adapters to materialize T0's accepted candidate."""
    select = getattr(kb, "select_candidate", None)
    if not callable(select):
        return True
    try:
        return bool(select(row))
    except Exception as exc:  # noqa: BLE001 — reject and continue cascade
        log.warning(
            "warm-start candidate materialization failed for %s: %s",
            row.get("canonical_id"),
            exc,
        )
        return False


def _find_config_donor(
    kb: Any,
    *,
    cid: str,
    hardware: str,
    framework: str,
    model_type: str,
    arch_slug: str,
    framework_version: str,
    precision: str,
    target_conc: Any = None,
    target_isl: Any = None,
    target_osl: Any = None,
) -> tuple[Mapping[str, Any] | None, str, float]:
    """Borrow a replayable config through the standard degradation tiers.

    Used when no donor config has been established yet — the identity match
    may have no replayable best_config, or it may have one at a non-exact
    tier that failed :func:`_donor_is_trustworthy`. The identity row still
    supplies priors, but the active warm-replay needs a champion config to
    apply. Search and compatibility checks match the main T0 cascade:
    same-architecture class, same GPU ISA, then nearest non-newer framework
    version at the target precision. Each candidate must additionally clear
    :func:`_donor_is_trustworthy` (positive validated gain, concrete matching
    architecture, matching workload shape) so a borrowed config is both
    stack-compatible and evidence-backed. Returns
    ``(donor_row, donor_tier, donor_confidence)`` or ``(None, "", 0.0)``.
    """
    if not (model_type or arch_slug):
        return None, "", 0.0
    common = {
        "framework_name": framework or "",
        "model_type": model_type,
        "architectures": arch_slug,
    }
    tiers = _warm_search_tiers(
        common=common,
        hardware=hardware,
        framework_version=framework_version,
        precision=precision,
    )
    seen = {cid}
    target_precision = str(precision or "").strip().lower()
    for (
        tier,
        confidence,
        labels,
        hardware_in,
        relax_framework_version,
    ) in tiers:
        labels = {
            key: value for key, value in labels.items() if value and value not in ("unknown_model_type", "unknown_arch")
        }
        usable: list[Mapping[str, Any]] = []
        candidates = _search_warm_candidates(
            kb,
            labels=labels,
            hardware_in=hardware_in,
        )
        for candidate in candidates:
            candidate_id = str(candidate.get("canonical_id") or "")
            if not candidate_id or candidate_id in seen:
                continue
            seen.add(candidate_id)
            if hardware_in is not None and not _hardware_is_compatible(
                hardware,
                _candidate_dimension(candidate, "hardware"),
            ):
                continue
            if _candidate_dimension(candidate, "precision") != target_precision:
                continue
            if relax_framework_version and not _framework_version_is_compatible(
                framework_version,
                _candidate_dimension(candidate, "framework_version"),
            ):
                continue
            if _donor_is_trustworthy(
                candidate,
                target_arch_slug=arch_slug,
                target_model_type=model_type,
                target_conc=target_conc,
                target_isl=target_isl,
                target_osl=target_osl,
            ):
                usable.append(candidate)
        if candidates and not usable:
            log.info(
                "warm-start config donor tier %s rejected all %d candidates",
                tier,
                len(candidates),
            )
        ranked = _rank_warm_candidates(
            usable,
            target_framework_version=framework_version,
        )
        if ranked:
            return ranked[0], tier, confidence
    return None, "", 0.0


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
    """Build the model-facing WarmStartContext from a KB recipe row.

    ``status`` is one of ``hit`` / ``seed_only`` / ``miss`` / ``error``.
    Current remote records expose only match/history/advisory metadata here;
    PRELUDE reads replay data through the section SDKs. Local legacy records
    retain their ready-to-replay projection.

    Config-donor decoupling: the experiential lists (priors) always come from the
    identity match ``recipe``, while ``recommended_replay`` is sourced from
    ``config_donor`` (the identity row itself when ``config_tier="self"``, or a
    borrowed same-architecture sibling). The donor's transfer confidence governs
    the downstream replay gate, not the identity-match confidence.
    """
    from .remote_recipe import RECORD_KIND_HYPERLOOM_RECIPE

    current_remote = bool(isinstance(recipe, Mapping) and recipe.get("record_kind") == RECORD_KIND_HYPERLOOM_RECIPE)
    ctx: dict[str, Any] = {
        "status": status,
        "match": {
            "tier": tier,
            "confidence": float(confidence),
            "source": source,
            "canonical_id": canonical_id,
        },
        "proven_prior": [],
        "do_not_repeat": [],
        "lessons": [],
        "pitfalls": [],
    }
    # Priors ride the identity match independent of replay config.
    if isinstance(recipe, Mapping):
        history = recipe.get("exact_history")
        prior_source = history if isinstance(history, Mapping) else recipe
        ctx["proven_prior"] = list(prior_source.get("what_worked") or [])
        ctx["do_not_repeat"] = list(prior_source.get("what_failed") or [])
        ctx["lessons"] = list(prior_source.get("lessons") or [])
        ctx["pitfalls"] = list(prior_source.get("pitfalls") or [])
    # Replay config comes from the donor, or the identity recipe as self-donor.
    donor = config_donor if not current_remote and isinstance(config_donor, Mapping) else None
    if not current_remote and donor is None and isinstance(recipe, Mapping) and _has_replayable_config(recipe):
        donor = recipe
        config_donor_tier = config_donor_tier or "self"
        if config_donor_confidence is None:
            config_donor_confidence = confidence
    if donor is not None:
        args, envs = _config_replay_args_envs(donor)
        if args or envs:
            try:
                best_tput = float(donor.get("best_throughput") or 0.0)
            except (TypeError, ValueError):
                best_tput = 0.0
            try:
                expected_gain = float(donor.get("validated_gain_pct") or 0.0)
            except (TypeError, ValueError):
                expected_gain = 0.0
            if expected_gain <= 0:
                expected_gain = _max_session_gain(donor)
            donor_session: Mapping[str, Any] | None = None
            donor_session_gain = float("-inf")
            for session in donor.get("sessions") or []:
                if not isinstance(session, Mapping):
                    continue
                try:
                    session_gain = float(session.get("gain_pct") or 0.0)
                except (TypeError, ValueError):
                    continue
                if session_gain > donor_session_gain:
                    donor_session = session
                    donor_session_gain = session_gain
            recommended_replay: dict[str, Any] = {
                "extra_server_args": args,
                "extra_envs": envs,
                "expected_gain_pct": expected_gain,
                "best_throughput": best_tput,
                "config_source": str(donor.get("canonical_id") or ""),
                "config_tier": config_donor_tier or "self",
                "config_confidence": float(confidence if config_donor_confidence is None else config_donor_confidence),
            }
            donor_canonical_id = str(donor.get("canonical_id") or "")
            donor_model = str(donor.get("model") or "")
            family_tags = donor.get("family_tags") or donor.get("model_architectures")
            breakdown_link = str(donor.get("breakdown_link") or "")
            if donor_session is not None:
                breakdown_link = str(
                    donor_session.get("breakdown_link") or donor_session.get("session_breakdown_url") or breakdown_link
                )
            recommended_replay.update(
                {
                    "donor_canonical_id": donor_canonical_id,
                    "donor_model": donor_model,
                    "donor_session_id": (
                        str(donor_session.get("session_id") or "") if donor_session is not None else ""
                    ),
                    "donor_family_tags": (
                        [str(tag) for tag in family_tags] if isinstance(family_tags, (list, tuple, set)) else []
                    ),
                    "donor_gain_pct": expected_gain,
                    "donor_breakdown_link": breakdown_link,
                }
            )
            ctx["recommended_replay"] = recommended_replay
    return ctx


def _build_t0_trace_extras(
    shared_state: Any,
    *,
    extra: "Mapping[str, Any]",
    fp: "Mapping[str, Any]",
    image_digest: str,
    model_class: str,
) -> dict[str, Any]:
    """Assemble operator-traceability + workload-shape tags for the recipe extras (skip empty/zero)."""
    _extras: dict[str, Any] = {}
    if model_class:
        _extras["model_class"] = model_class
    _architectures = getattr(shared_state, "model_architectures", None) or []
    if isinstance(_architectures, list):
        _arch_list = [str(a).strip() for a in _architectures if str(a or "").strip()]
        if _arch_list:
            _extras["architectures"] = _arch_list
    _model_type = str(getattr(shared_state, "model_type", "") or "").strip()
    if _model_type:
        _extras["model_type"] = _model_type
    rocm_v = str(fp.get("rocm") or "").strip()
    if rocm_v and rocm_v != "unknown":
        _extras["rocm_version"] = rocm_v
    aiter_v = str(fp.get("aiter") or "").strip()
    if aiter_v and aiter_v != "unknown":
        _extras["aiter_version"] = aiter_v
    if image_digest and image_digest != "unknown":
        _extras["image_digest"] = str(image_digest).strip()
    for src_key in ("claw_session_id", "sandbox_user_id"):
        v = str(extra.get(src_key) or "").strip()
        if v:
            _extras[src_key] = v
    for src_attr, dst_key in (
        ("tp", "tp"),
        ("ep", "ep"),
        ("conc", "conc"),
        ("isl", "isl"),
        ("osl", "osl"),
        ("max_model_len", "max_model_len"),
    ):
        v = getattr(shared_state, src_attr, None)
        if v not in (None, "", 0):
            _extras[dst_key] = v
    if "ep" not in _extras:
        raw_ep = (os.environ.get("EP") or "").strip()
        try:
            n = int(raw_ep) if raw_ep else 0
        except ValueError:
            n = 0
        if n > 0:
            _extras["ep"] = n
    raw_pp = (os.environ.get("PP") or "").strip()
    try:
        pp_n = int(raw_pp) if raw_pp else 0
    except ValueError:
        pp_n = 0
    if pp_n > 0:
        _extras["pp"] = pp_n
    return _extras


_GPU_ISA_BY_SKU = {
    "mi300x": "gfx942",
    "mi308x": "gfx942",
    "mi325x": "gfx942",
    "mi355x": "gfx950",
}
_TOPOLOGY_SUFFIX_RE = re.compile(
    r"_ws[1-9]\d*(?:_pd[1-9]\d*p[1-9]\d*d)?"
    r"(?:_tp[1-9]\d*)?(?:_ep[1-9]\d*)?(?:_[a-z0-9-]+)?$"
)


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
    """Accept exact or nearest non-newer PEP 440 version in one release line."""
    target_value = str(target or "").strip().lower()
    candidate_value = str(candidate or "").strip().lower()
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
    *,
    common: dict[str, str],
    hardware: str,
    framework_version: str,
    precision: str,
) -> tuple[
    tuple[str, float, dict[str, str], list[str] | None, bool],
    ...,
]:
    """Build the shared ordered seven-tuple degradation tiers."""
    hardware_values = _hardware_fallback_values(hardware)
    return (
        (
            "same_arch_class",
            0.95,
            {
                **common,
                "hardware": hardware,
                "framework_version": framework_version,
                "precision": precision,
            },
            None,
            False,
        ),
        (
            "same_gpu_isa",
            0.85,
            {
                **common,
                "framework_version": framework_version,
                "precision": precision,
            },
            hardware_values,
            False,
        ),
        (
            "compatible_framework_version",
            0.72,
            {**common, "precision": precision},
            hardware_values,
            True,
        ),
    )


def _candidate_dimension(row: Mapping[str, Any], key: str) -> str:
    """Read one seven-tuple dimension from metadata or canonical id."""
    alias = "framework_name" if key == "framework" else key
    value = row.get(key)
    if value in (None, ""):
        value = row.get(alias)
    if value not in (None, ""):
        return str(value).strip().lower()
    try:
        dimensions = cid_to_path_components(str(row.get("canonical_id") or ""))
    except ValueError:
        return ""
    names = (
        "model",
        "hardware",
        "framework_name",
        "model_type",
        "architectures",
        "framework_version",
        "precision",
    )
    return str(dict(zip(names, dimensions)).get(alias) or "").strip().lower()


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
    """Run exact identity search and fail soft on backend errors."""
    kwargs: dict[str, Any] = {"label_match": labels, "limit": 100}
    if hardware_in is not None and getattr(kb, "mode", "") == "remote":
        kwargs["hardware_in"] = hardware_in
    try:
        rows = kb.search(**kwargs)
    except Exception as exc:  # noqa: BLE001
        log.info("warm-start exact candidate search failed: %s", exc)
        return []
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


def _cascade_warm_start_search(
    kb: "RecipeKB",
    *,
    cid: str,
    hw: str,
    framework: str,
    model_type_val: str,
    architectures_val: Any,
    arch_slug: str,
    fw_version: str,
    precision: str,
    warm_prefer: Any,
    target_conc: Any = None,
    target_isl: Any = None,
    target_osl: Any = None,
) -> "tuple[dict[str, Any], str, float]":
    """Resolve exact plus cumulative four-tier seven-tuple fallbacks."""
    del architectures_val
    exact_history: Mapping[str, Any] | None = None
    try:
        exact = kb.get_recipe(
            canonical_id=cid,
            prefer=warm_prefer or None,
        )
    except Exception as exc:  # noqa: BLE001
        log.info("warm-start exact get non-fatal failure: %s", exc)
        exact = None
    if isinstance(exact, Mapping) and exact and str(exact.get("canonical_id") or "") == cid:
        if _recipe_is_actionable(exact):
            return dict(exact), "exact", 1.0
        exact_history = exact

    try:
        (
            _,
            target_hardware,
            target_framework,
            target_model_type,
            target_architecture,
            target_framework_version,
            target_precision,
        ) = cid_to_path_components(cid)
    except ValueError:
        target_hardware = str(hw or "")
        target_framework = str(framework or "")
        target_model_type = str(model_type_val or "")
        target_architecture = str(arch_slug or "")
        target_framework_version = str(fw_version or "")
        target_precision = str(precision or "")
    common = {
        "framework_name": target_framework,
        "model_type": target_model_type,
        "architectures": target_architecture,
    }
    tiers = _warm_search_tiers(
        common=common,
        hardware=target_hardware,
        framework_version=target_framework_version,
        precision=target_precision,
    )
    seen = {cid}
    target_precision_value = str(target_precision or "").strip().lower()
    for (
        tier,
        confidence,
        labels,
        hardware_in,
        relax_framework_version,
    ) in tiers:
        rows = _search_warm_candidates(
            kb,
            labels=labels,
            hardware_in=hardware_in,
        )
        usable: list[Mapping[str, Any]] = []
        for candidate in rows:
            candidate_id = str(candidate.get("canonical_id") or "")
            if not candidate_id or candidate_id in seen:
                continue
            seen.add(candidate_id)
            if not _remote_candidate_matches(kb, candidate, labels=labels):
                continue
            if hardware_in is not None and not _hardware_is_compatible(
                target_hardware,
                _candidate_dimension(candidate, "hardware"),
            ):
                continue
            if _candidate_dimension(candidate, "precision") != target_precision_value:
                continue
            if relax_framework_version and not _framework_version_is_compatible(
                target_framework_version,
                _candidate_dimension(candidate, "framework_version"),
            ):
                continue
            if not _donor_is_trustworthy(
                candidate,
                target_arch_slug=target_architecture,
                target_model_type=target_model_type,
                target_conc=target_conc,
                target_isl=target_isl,
                target_osl=target_osl,
            ):
                continue
            usable.append(candidate)
        if rows and not usable:
            log.info(
                "warm-start tier %s rejected all %d candidates",
                tier,
                len(rows),
            )
        for candidate in _rank_warm_candidates(
            usable,
            target_framework_version=target_framework_version,
        ):
            if _select_remote_candidate(kb, candidate):
                return (
                    _with_exact_history(candidate, exact_history),
                    tier,
                    confidence,
                )
    if exact_history is not None:
        return dict(exact_history), "exact", 1.0
    return {}, "miss", 0.0


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
    """Run the T0 recipe-snapshot anchor and seven-tuple warm-start search.

    Anchors identity, then cascades exact/model/hardware/framework tiers; in
    remote mode a selected donor is materialized via ``select_candidate``.
    Mutates ``shared_state`` in place (warm_start_* fields) and persists when
    ``save_state=True``. ``session_dir`` is required.

    Args:
        kb: The recipe-KB dispatcher used for the read-modify-write anchor.
        shared_state: The live SharedState, mutated in place with warm-start
            results.
        workload: The model/workload identifier.
        hw: The hardware/GPU identifier.
        image_digest: Optional container image digest stamped as a trace tag.
        stack_fingerprint: Optional stack-version fingerprint mapping.
        extra_attrs: Optional extra identity/trace attributes (model_class,
            framework, session ids).
        resume: When ``True``, re-anchor even if already anchored.
        on_status: Optional status-line callback; defaults to INFO logging.
        session_dir: The session directory (required).
        save_state: When ``True``, persist the mutated SharedState.

    Raises:
        ValueError: If ``session_dir`` is ``None``.
    """
    emit = on_status or _default_status_emitter
    if session_dir is None:
        raise ValueError("run_t0_anchor requires an explicit session_dir")
    sd = Path(session_dir)

    sid = (getattr(shared_state, "recipe_kb_session_id", "") or "").strip()
    if not sid and sd is not None:
        sid = Path(sd).name

    workload = (workload or "").strip() or "unknown_model"
    hw = (hw or "").strip() or "unknown_gpu"
    # Topology-aware hardware slug (multi-node appends ``_ws{world_size}``),
    # resolved once so the cid, the stored ``hardware`` field, and every
    # warm-start tier below share an identical, isolated key.
    from hyperloom.orchestrator.actions.executors._multi_node_env import resolve_kb_topology

    hw = kb_hardware_slug(hw, **resolve_kb_topology())

    # Short-circuit when already anchored; resume=True bypasses.
    if sid and not resume and (getattr(shared_state, "warm_start_ts", "") or "").strip():
        shared_state.recipe_kb_session_id = sid
        emit(f"Recipe KB        : already anchored session_id={sid}")
        return

    if sid:
        shared_state.recipe_kb_session_id = sid
        if resume:
            emit(f"Recipe KB        : resumed session_id={sid}")
    began_now = not getattr(shared_state, "warm_start_ts", "")
    if began_now:
        shared_state.warm_start_ts = datetime.now(timezone.utc).isoformat(
            timespec="seconds",
        )

    # Backfill operator-tracing metadata; T0 only stamps metadata (best_config
    # preserved, rewritten at CLOSE).
    _extra: Mapping[str, Any] = extra_attrs if isinstance(extra_attrs, Mapping) else {}
    _model_class = str(_extra.get("model_class") or "").strip()
    _framework = str(
        _extra.get("framework_name")
        or _extra.get("framework")
        or getattr(shared_state, "framework", "")
        or os.environ.get("FRAMEWORK", "")
        or ""
    ).strip()
    _precision = str(getattr(shared_state, "precision", "") or "").strip()
    fp: Mapping[str, Any] = stack_fingerprint if isinstance(stack_fingerprint, Mapping) else {}
    # framework_version: SharedState > stack_fingerprint > importlib auto-detect.
    _fw_version = str(getattr(shared_state, "framework_version", "") or "").strip()
    if not _fw_version and _framework in ("sglang", "vllm"):
        _fw_version = str(fp.get(_framework) or "").strip()
        if _fw_version == "unknown":
            _fw_version = ""
    if not _fw_version and _framework:
        _fw_version = detect_framework_version(_framework)

    _extras = _build_t0_trace_extras(
        shared_state,
        extra=_extra,
        fp=fp,
        image_digest=image_digest,
        model_class=_model_class,
    )

    # Build canonical_id from the resolved 7-tuple.
    _model_type_val = str(getattr(shared_state, "model_type", "") or "").strip()
    _architectures_val = getattr(shared_state, "model_architectures", None) or []
    cid = recipe_canonical_id(
        model=workload,
        hardware=hw,
        framework_name=_framework or "",
        framework_version=_fw_version or "",
        precision=_precision or "",
        model_type=_model_type_val,
        architectures=_architectures_val,
    )

    # Persist framework + framework_version so CLOSE/KEEP derives the same cid.
    if _framework:
        shared_state.framework = _framework
    if _fw_version:
        shared_state.framework_version = _fw_version

    if getattr(kb, "mode", "") != "remote":
        # Read-modify-write the selected store's exact authority row so the stamp
        # does not clobber fields or trigger a broad remote warm-start scan.
        try:
            live = kb.get_authoritative_recipe(canonical_id=cid) or {}
        except Exception as exc:  # noqa: BLE001 — defensive
            log.info("T0 anchor authority get_recipe non-fatal failure: %s", exc)
            live = {}

        # Merge prior extras; new values win.
        merged_extras: dict[str, Any] = {}
        prior_extras = {
            k: v
            for k, v in (live or {}).items()
            if k
            not in {
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
        }
        merged_extras.update(prior_extras)
        merged_extras.update(_extras)

        # Stack fingerprint — preserve prior values not stamped this round.
        sfp_payload: dict[str, str] = dict(live.get("stack_fingerprint") or {})
        if isinstance(fp, Mapping):
            for fp_key in ("vllm_version", "aiter_commit", "rocm_version"):
                new = str(fp.get(fp_key.replace("_version", "").replace("_commit", "")) or "").strip()
                if new and new != "unknown":
                    sfp_payload[fp_key] = new

        # Third Recipe sink; see agentx_kb_write_blocked. _build_t0_trace_extras
        # copies SharedState.isl/osl into the row, which under AgentX are the
        # inert 1024/1024 placeholders -- so anchoring here mis-tags the
        # cross-session row exactly as the CLOSE-time write would.
        from hyperloom.orchestrator.actions.executors._workload_envs import (
            agentx_kb_write_blocked,
        )

        if agentx_kb_write_blocked(shared_state):
            log.info(
                "T0 anchor: skipping put_recipe (AgentX); the recipe row has no "
                "mode or workload dimension and isl/osl are placeholders here."
            )
            return

        try:
            kb.put_recipe(
                canonical_id=cid,
                model=workload,
                hardware=hw,
                framework_name=_framework or "",
                framework_version=_fw_version or "",
                precision=_precision or "",
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
                extras=merged_extras,
                provenance={
                    "source": "hyperloom-inference-optimizer",
                    "generator": "t0_anchor",
                    "generated_at": datetime.now(timezone.utc).isoformat(
                        timespec="microseconds",
                    ),
                    "details": {"sid": sid},
                },
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception("T0 anchor put_recipe raised unexpectedly")

    else:
        log.debug("T0 anchor: remote publication is owned by CLOSE")

    # Exact seven-tuple, then cumulative model/hardware/precision/version relaxations.
    warm_prefer = _build_warm_prefer(shared_state, _fw_version)

    # Architecture remains exact at every fallback tier.
    from hyperloom.inference_optimizer.recipe_snapshot_constants import _architectures_slug

    _arch_slug = _architectures_slug(_architectures_val)

    warm_point, warm_tier, warm_conf = _cascade_warm_start_search(
        kb,
        cid=cid,
        hw=hw,
        framework=_framework,
        model_type_val=_model_type_val,
        architectures_val=_architectures_val,
        arch_slug=_arch_slug,
        fw_version=_fw_version,
        precision=_precision,
        warm_prefer=warm_prefer,
        target_conc=getattr(shared_state, "conc", None),
        target_isl=getattr(shared_state, "isl", None),
        target_osl=getattr(shared_state, "osl", None),
    )

    # A bare T0 anchor (no best_config) demotes to seed_only.
    if warm_point and not _recipe_is_actionable(warm_point):
        warm_tier = "seed_only"
        warm_conf = 0.0

    # Config-donor decoupling: the identity match supplies priors; borrow a
    # champion config from the nearest same-arch sibling when it has none.
    config_donor: Mapping[str, Any] | None = None
    config_donor_tier = ""
    config_donor_conf = 0.0
    from .remote_recipe import RECORD_KIND_HYPERLOOM_RECIPE

    current_remote_point = bool(
        isinstance(warm_point, Mapping) and warm_point.get("record_kind") == RECORD_KIND_HYPERLOOM_RECIPE
    )
    _tgt_conc = getattr(shared_state, "conc", None)
    _tgt_isl = getattr(shared_state, "isl", None)
    _tgt_osl = getattr(shared_state, "osl", None)
    # A true-self (identity ``exact``) champion always replays; a cross-model
    # borrow must clear the trustworthiness gate before it becomes the donor.
    if (
        not current_remote_point
        and warm_point
        and _has_replayable_config(warm_point)
        and (
            warm_tier == "exact"
            or _donor_is_trustworthy(
                warm_point,
                target_arch_slug=_arch_slug,
                target_model_type=_model_type_val,
                target_conc=_tgt_conc,
                target_isl=_tgt_isl,
                target_osl=_tgt_osl,
            )
        )
    ):
        config_donor = warm_point
        config_donor_tier = "self"
        config_donor_conf = warm_conf
    if config_donor is None and warm_point and not current_remote_point:
        donor, dtier, dconf = _find_config_donor(
            kb,
            cid=cid,
            hardware=hw,
            framework=_framework or "",
            model_type=_model_type_val,
            arch_slug=_arch_slug,
            framework_version=_fw_version or "",
            precision=_precision or "",
            target_conc=_tgt_conc,
            target_isl=_tgt_isl,
            target_osl=_tgt_osl,
        )
        if donor is not None:
            config_donor = donor
            config_donor_tier = dtier
            config_donor_conf = dconf

    # Keep warm.json envelope shape stable; new readers prefer
    # shared_state.warm_start_recipe.
    warm_text = json.dumps(
        {"points": [warm_point] if warm_point else []},
        sort_keys=True,
    )
    try:
        warm_path = recipe_kb_warm_json(sd)
        warm_path.parent.mkdir(parents=True, exist_ok=True)
        warm_path.write_text(
            json.dumps(
                {
                    "workload": workload,
                    "hw": hw,
                    "tier": warm_tier,
                    "confidence": warm_conf,
                    "recipe": warm_point,
                    "raw": warm_text,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        shared_state.warm_start_recipe = {
            "workload": workload,
            "hw": hw,
            "tier": warm_tier,
            "confidence": warm_conf,
            "recipe": warm_point,
        }
    except OSError as exc:
        log.warning("warm_start snapshot write failed: %s", exc)

    # WarmStartContext: model-facing projection of the KB result, with an
    # explicit hit/seed_only/miss status.
    if not warm_point:
        wsc_status = "miss"
    elif warm_tier == "seed_only":
        wsc_status = "seed_only"
    else:
        wsc_status = "hit"
    warm_source = _warm_recipe_source(warm_point, kb)
    try:
        shared_state.warm_start_context = _build_warm_start_context(
            config_donor=config_donor,
            config_donor_tier=config_donor_tier,
            config_donor_confidence=config_donor_conf,
            status=wsc_status,
            tier=warm_tier,
            confidence=warm_conf,
            canonical_id=cid,
            source=warm_source,
            recipe=warm_point or None,
        )
    except Exception:  # noqa: BLE001 — defensive; context is advisory
        log.exception("warm_start_context build failed")

    # warm_start_pitfalls / warm_start_lessons are embedded recipe-row fields.
    exact_history = warm_point.get("exact_history")
    history_source = exact_history if isinstance(exact_history, Mapping) else warm_point
    pitfalls_list: list[dict[str, Any]] = list(history_source.get("pitfalls") or [])
    lessons_list: list[dict[str, Any]] = list(history_source.get("lessons") or [])
    try:
        pit_path = recipe_kb_pitfalls_json(sd)
        pit_path.parent.mkdir(parents=True, exist_ok=True)
        pit_path.write_text(
            json.dumps(
                {
                    "workload": workload,
                    "hw": hw,
                    "framework": _framework or "",
                    "pitfalls": pitfalls_list,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if pitfalls_list:
            shared_state.warm_start_pitfalls = pitfalls_list
    except OSError as exc:
        log.warning("warm_start_pitfalls snapshot write failed: %s", exc)
    try:
        les_path = recipe_kb_lessons_json(sd)
        les_path.parent.mkdir(parents=True, exist_ok=True)
        les_path.write_text(
            json.dumps(
                {
                    "workload": workload,
                    "hw": hw,
                    "framework": _framework or "",
                    "lessons": lessons_list,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if lessons_list:
            shared_state.warm_start_lessons = lessons_list
    except OSError as exc:
        log.warning("warm_start_lessons snapshot write failed: %s", exc)

    if save_state:
        try:
            shared_state.save(sd)
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "Recipe KB T0: SharedState.save failed (sid=%s, workload=%s)",
                sid,
                workload,
            )

    # warm_present = usable record (confidence > 0).
    warm_present = bool(warm_point) and warm_conf > 0.0
    if began_now:
        warm_label = f"hit:{warm_tier}@{warm_conf:.2f}" if warm_present else "seed_only" if warm_point else "empty"
        emit(
            f"Recipe KB        : session_id={sid} "
            f"workload={cid} "
            f"(warm={warm_label}, "
            f"pitfalls={len(pitfalls_list)}, "
            f"lessons={len(lessons_list)})"
        )
    return


__all__ = [
    "run_t0_anchor",
]
