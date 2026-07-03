# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared T0 (PRELUDE) Cortex anchor — KB warm-start only."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from ..recipe_kb import RecipeKB, recipe_canonical_id
from ..recipe_snapshot_constants import detect_framework_version
from ..session_paths import cortex_lessons_json, cortex_pitfalls_json, cortex_warm_json


log = logging.getLogger(__name__)


@dataclass
class T0Result:
    """Outcome of one :func:`run_t0_anchor` invocation. ``status`` ∈ {ok, resumed, skipped_already}."""

    status: str
    session_id: str = ""
    workload: str = ""
    hw: str = ""
    warm_present: bool = False
    pitfalls_present: bool = False
    lessons_present: bool = False


def _default_status_emitter(line: str) -> None:
    """Default ``on_status`` callback — log the banner line at INFO.

    Args:
        line (str): The status banner line to emit.
    """
    log.info("%s", line)


# ---------------------------------------------------------------------------
# Warm-start helpers (prefer rerank / seed-only detection / WarmStartContext)
# ---------------------------------------------------------------------------
# Numeric workload knobs threaded into the KB ``prefer`` block so a
# closer-workload recipe is reranked first (the dispatcher does the actual
# rerank; the 5-tuple ``required`` filter is unchanged).
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


def _row_best_config_source(row: Mapping[str, Any] | None) -> str:
    """Per-path tag for the backend that supplied ``best_config`` on a row.

    Reads the composite remote's ``_field_sources`` / ``_sources`` provenance
    markers. Returns ``""`` when the row carries no provenance (single-source
    backend / local-only row), letting the caller fall back to the remote type.

    Args:
        row (Mapping[str, Any] | None): A merged recipe row (may be empty).

    Returns:
        str: A short source tag (e.g. ``gbrain`` / ``cortex``), or ``""``.
    """
    if not isinstance(row, Mapping):
        return ""
    field_sources = row.get("_field_sources")
    if isinstance(field_sources, Mapping):
        best_config_src = field_sources.get("best_config")
        if isinstance(best_config_src, str) and best_config_src:
            return best_config_src
        if isinstance(best_config_src, list) and best_config_src:
            return str(best_config_src[0])
    sources = row.get("_sources")
    if isinstance(sources, list) and sources:
        return str(sources[0])
    return ""


def _warm_recipe_source(
    row: Mapping[str, Any] | None,
    kb: Any,
    *,
    config_donor: Mapping[str, Any] | None = None,
    config_donor_tier: str = "",
) -> str:
    """Resolve which KB path actually supplied the applied warm recipe.

    Under the composite remote, a merged row carries ``_field_sources`` /
    ``_sources`` recording which backend supplied each field. When the
    replayable champion is borrowed from a same-architecture sibling, the
    FINAL applied config comes from the donor's path — so prefer
    the donor's ``best_config`` source whenever a separate donor (tier other
    than ``self``) supplied it. Otherwise attribute to the identity match's
    ``best_config`` backend, then its first contributing source, and finally
    fall back to the dispatcher's remote type. This makes the WarmStartContext
    attribute the FINAL applied recipe to the correct path (gbrain vs cortex)
    instead of guessing.

    Args:
        row (Mapping[str, Any] | None): The identity-match warm recipe row.
        kb (Any): The RecipeKB dispatcher (for the remote-type fallback).
        config_donor (Mapping[str, Any] | None): The recipe that supplied the
            replayable ``best_config``, when borrowed from a sibling.
        config_donor_tier (str): The donor tier; ``self`` means the identity
            match supplied its own config (no borrow).

    Returns:
        str: A short source tag, e.g. ``gbrain`` / ``cortex`` / ``cortex-kb``.
    """
    # A borrowed config (donor != identity match) is the FINAL replayed config;
    # attribute it to the donor's path when provenance is available.
    if config_donor is not None and config_donor_tier and config_donor_tier != "self":
        donor_src = _row_best_config_source(config_donor)
        if donor_src:
            return donor_src
    own_src = _row_best_config_source(row)
    if own_src:
        return own_src
    return "gbrain" if _remote_is_gbrain(kb) else "cortex-kb"


def _remote_is_gbrain(kb: Any) -> bool:
    """Best-effort source tag for the WarmStartContext.

    Reports ``True`` when the dispatcher's active remote is the gbrain
    adapter; otherwise the source is the cortex kb-service (or local-only
    fallback, which we still label ``cortex-kb`` since that is the
    configured remote contract).

    Args:
        kb: The recipe-KB dispatcher whose active remote is inspected.

    Returns:
        ``True`` when the active remote is the gbrain adapter.
    """
    remote = getattr(kb, "remote", None)
    return type(remote).__name__ == "GbrainRemoteRecipeClient"


def _recipe_is_actionable(row: Mapping[str, Any]) -> bool:
    """True when a warm recipe carries something worth replaying / priors.

    A bare local draft anchor (T0 ``put_recipe`` stamps identity + tracing
    tags but no champion / experiential lists) is NOT actionable: treating
    it as a confident hit would let warm-replay apply an empty config and
    starve the specialist prompt of real priors.

    Args:
        row: A warm recipe row to inspect.

    Returns:
        ``True`` when the row carries a usable config, positive throughput, or
        any experiential list worth replaying.
    """
    if not isinstance(row, Mapping):
        return False
    best_config = row.get("best_config")
    if isinstance(best_config, Mapping) and best_config:
        # An env-only or args-only config is still actionable.
        args = str(best_config.get("extra_server_args") or best_config.get("args") or "").strip()
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


def _config_replay_args_envs(row: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
    """Extract a replayable ``(args, envs)`` pair from a row's best_config.

    Reads the canonical ``extra_server_args`` first, then the legacy
    ``extra_sglang_args`` / ``args`` aliases, and the nested env map under
    ``extra_envs`` / ``envs``. Returns empty values when nothing replayable
    is present.
    """
    best_config = row.get("best_config") if isinstance(row.get("best_config"), Mapping) else {}
    args = str(
        best_config.get("extra_server_args") or best_config.get("extra_sglang_args") or best_config.get("args") or ""
    ).strip()
    envs = best_config.get("extra_envs") or best_config.get("envs") or {}
    if not isinstance(envs, Mapping):
        envs = {}
    return args, {str(k): str(v) for k, v in envs.items()}


def _has_replayable_config(row: Mapping[str, Any]) -> bool:
    """True when ``row`` carries a non-empty champion config (args OR envs)."""
    if not isinstance(row, Mapping):
        return False
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
            pass  # non-numeric gain field; fall through with best=0.0
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
    * no explicit workload-shape (conc/isl/osl) mismatch with the target.

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
    if not _has_replayable_config(donor):
        return False
    # RC1: require evidence of a real positive gain (no zero-gain borrows).
    if _max_session_gain(donor) <= 0:
        return False
    # RC2: require a concrete architecture matching the target (no cross-arch / unknown).
    from ..recipe_snapshot_constants import _architectures_slug

    donor_arch = _architectures_slug(donor.get("architectures") or [])
    donor_mt = str(donor.get("model_type") or "").strip().lower()
    _unknown = {"", "unknown", "unknown_arch", "unknown_model_type"}
    if donor_arch in _unknown or donor_mt in _unknown:
        return False
    if target_arch_slug and donor_arch and donor_arch != target_arch_slug:
        return False
    tgt_mt = str(target_model_type or "").strip().lower()
    if tgt_mt and donor_mt and donor_mt != tgt_mt:
        return False

    # RC3: reject an explicit workload-shape mismatch (only when both are known).
    def _shape_conflict(target_val: Any, donor_key: str) -> bool:
        try:
            tv = int(target_val)
            dv = int(donor.get(donor_key))
        except (TypeError, ValueError):
            return False  # unknown on either side → not a conflict
        if tv <= 0 or dv <= 0:
            return False
        return tv != dv

    if (
        _shape_conflict(target_conc, "conc")
        or _shape_conflict(target_isl, "isl")
        or _shape_conflict(target_osl, "osl")
    ):
        return False
    return True


def _kg_native_config_donor(
    *,
    architectures: "list[str] | None",
    precision: str,
    hardware: str,
    framework: str,
    model_type: str,
) -> Mapping[str, Any] | None:
    """Borrow a cross-model warm-replay donor from the native KG link-graph.

    Active only when a NATIVE KG client (``GBRAIN_KG_NATIVE``) is reachable.
    Returns a recipe-shaped donor synthesized from the strongest cross-model
    ``KNOB_IMPROVES`` edge for the target ``arch+precision`` (single_top), or
    ``None`` to fall back to the recipe-KB sibling search. Fully
    degradation-safe — any failure yields ``None`` and never blocks warm-start.

    Args:
        architectures: Current model architecture list.
        precision: Baseline precision (folded into the KG object node).
        hardware: Hardware condition filter.
        framework: Framework condition filter.
        model_type: Target model type (stamped onto the synthesized donor).

    Returns:
        A recipe-shaped donor row, or ``None`` when KG is unavailable, not in
        native mode, or carries no usable cross-model knob.
    """
    archs = [a for a in (architectures or []) if str(a or "").strip()]
    if not archs:
        return None
    try:
        from ..recipe_kb.kg_client import (
            generate_warmstart_donor_graph_guided,
            get_kg_client,
        )

        kg = get_kg_client()
        if kg is None or not getattr(kg, "_native", False) or not kg.is_available():
            return None
        donor = generate_warmstart_donor_graph_guided(
            kg,
            architectures=archs,
            precision=precision or "",
            hardware=hardware or "",
            framework=framework or "",
            model_type=model_type or "",
        )
        return donor or None
    except Exception as exc:  # noqa: BLE001 — KG donor is best-effort/advisory
        log.info("KG-native config donor skipped: %s", exc)
        return None


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
    """Borrow a replayable champion config from the nearest same-arch sibling.

    Used when the exact (L1) identity match has no replayable best_config:
    the identity row still supplies priors, but the active warm-replay needs
    a champion config to apply. Searches L2 (same arch class, cross-model,
    conf 0.95) then L3 (same arch, any framework version, conf 0.5).

    Only same ``hw+framework+model_type+arch(+precision)(+fwv)`` siblings are
    searched, and each candidate must additionally clear
    :func:`_donor_is_trustworthy` (positive validated gain, concrete matching
    architecture, no workload-shape conflict) so a borrowed config is both
    stack-compatible and evidence-backed. Returns
    ``(donor_row, donor_tier, donor_confidence)`` or ``(None, "", 0.0)``.
    """
    if not (model_type or arch_slug):
        return None, "", 0.0
    cascade = (
        (
            "same_arch_class",
            0.95,
            {
                "hardware": hardware,
                "framework": framework or "",
                "model_type": model_type,
                "architectures": arch_slug,
                "framework_version": framework_version or "",
                "precision": precision or "",
            },
        ),
        (
            "same_arch_any_version",
            0.5,
            {
                "hardware": hardware,
                "framework": framework or "",
                "model_type": model_type,
                "architectures": arch_slug,
                "precision": precision or "",
            },
        ),
    )
    for tier, conf, labels in cascade:
        labels = {k: v for k, v in labels.items() if v and v not in ("unknown_model_type", "unknown_arch")}
        try:
            rows = kb.search(label_match=labels, limit=10)
        except Exception:  # noqa: BLE001 — donor search is best-effort/advisory
            rows = []
        for r in rows or []:
            if str(r.get("canonical_id") or "") == cid:
                continue
            # Borrowed donor must clear the trustworthiness gate (validated gain,
            # concrete matching arch, no shape conflict) — not just carry a config.
            if _donor_is_trustworthy(
                r,
                target_arch_slug=arch_slug,
                target_model_type=model_type,
                target_conc=target_conc,
                target_isl=target_isl,
                target_osl=target_osl,
            ):
                return r, tier, conf
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
    config_donor_confidence: float = 0.0,
    model_architectures: "list[str] | None" = None,
    hardware: str = "",
    framework: str = "",
    precision: str = "",
    kg_client: Any = None,
) -> dict[str, Any]:
    """Build the model-facing WarmStartContext from a KB recipe row.

    ``status`` is one of ``hit`` / ``seed_only`` / ``miss`` / ``error``.
    Decouples model consumption from the KB row layout: it surfaces a
    ready-to-replay champion plus the experiential lists, so consumers
    (warm-replay / specialist prompt / ledger) never parse the raw
    recipe shape.

    Config-donor decoupling: the experiential lists (priors) always come
    from the identity match ``recipe`` (even when its own best_config is
    empty), while ``recommended_replay`` is sourced from ``config_donor`` —
    which may be the identity row itself (``config_tier="self"``) or a
    borrowed same-architecture sibling. The donor's transfer confidence
    (``config_confidence``) governs the downstream replay gate, NOT the
    identity-match confidence.
    """
    ctx: dict[str, Any] = {
        "status": status,
        "match": {
            "tier": tier,
            "confidence": float(confidence),
            "source": source,
            "canonical_id": canonical_id,
        },
        "recommended_replay": {},
        "proven_prior": [],
        "do_not_repeat": [],
        "lessons": [],
        "pitfalls": [],
    }
    # Priors ride the identity match regardless of whether it carries a
    # replayable config (seed-only / empty-config anchors still anchor priors).
    if isinstance(recipe, Mapping):
        ctx["proven_prior"] = list(recipe.get("what_worked") or [])
        ctx["do_not_repeat"] = list(recipe.get("what_failed") or [])
        ctx["lessons"] = list(recipe.get("lessons") or [])
        ctx["pitfalls"] = list(recipe.get("pitfalls") or [])
    # Replay config comes from the donor (self or borrowed sibling). When no
    # explicit donor is supplied (legacy callers / direct unit use), fall back
    # to the identity recipe itself as a self-donor if it owns a replayable
    # config — preserving the original "hit recipe replays its own config"
    # contract.
    donor = config_donor if isinstance(config_donor, Mapping) else None
    if donor is None and isinstance(recipe, Mapping) and _has_replayable_config(recipe):
        donor = recipe
        config_donor_tier = config_donor_tier or "self"
        config_donor_confidence = config_donor_confidence or confidence
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
            ctx["recommended_replay"] = {
                "extra_server_args": args,
                "extra_envs": envs,
                "expected_gain_pct": expected_gain,
                "best_throughput": best_tput,
                "config_source": str(donor.get("canonical_id") or ""),
                "config_tier": config_donor_tier or "self",
                "config_confidence": float(config_donor_confidence or confidence),
            }
    # Extract replayable code patches from prs_tested (positive + negative).
    _extract_patches_from_prs_tested(ctx, recipe, model_architectures)
    # KG enhancement: cross-recipe revert aggregation + arch-graph advisory
    # blocks + graph-recommended knobs + validity. Best-effort and fully
    # degradable — never lose the local prs_tested patches on KG failure.
    _enhance_warm_start_with_kg(
        ctx,
        model_architectures=model_architectures,
        hardware=hardware,
        framework=framework,
        precision=precision,
        kg_client=kg_client,
    )
    return ctx


def _kg_guided_enabled() -> bool:
    """Return ``True`` when journal-knob graph guidance is enabled.

    Gated by ``GBRAIN_KG_GUIDED`` (default off) so emitting ``KNOB_*`` edges
    and surfacing ``graph_guided_knobs`` is a deliberate opt-in that never
    changes live behavior until explicitly turned on.
    """
    return os.environ.get("GBRAIN_KG_GUIDED", "").strip().lower() in ("1", "true", "yes")


def _arch_norm(value: Any) -> str:
    """Normalize an architecture/entity token to match KG fact slugs.

    Args:
        value: Raw architecture or entity name.

    Returns:
        The lowercased slug (spaces/slashes to underscores).
    """
    return str(value or "").strip().replace(" ", "_").replace("/", "_").lower()


def _enhance_warm_start_with_kg(
    ctx: dict[str, Any],
    *,
    model_architectures: "list[str] | None",
    hardware: str,
    framework: str,
    precision: str = "",
    kg_client: Any = None,
) -> None:
    """Augment the warm-start context with knowledge-graph signals.

    Adds, when a KG backend is reachable:

    * ``blocked_patches`` — cross-recipe hard blocks on the same arch
      (``REVERTED_ON`` / ``DEGRADES`` / ``CRASHES``).
    * ``advisory_blocked_patches`` — soft blocks inferred from related
      architectures (decayed confidence), reached via ``USES_ARCH`` /
      ``VARIANT_OF`` graph traversal.
    * ``recommended_knobs`` — ``IMPROVES`` candidates for the current
      arch+hw+fw (minus anything blocked).
    * ``expired`` markers on replay patches whose ``VALID_FOR`` window
      has lapsed.

    The whole step is wrapped in a degradation guard: any failure leaves
    ``ctx`` exactly as the local prs_tested logic produced it.

    Args:
        ctx: The warm-start context to mutate in place.
        model_architectures: The current model's architecture list.
        hardware: The current hardware/GPU identifier.
        framework: The current inference framework.
        kg_client: Optional injected KG client (defaults to the env-built
            singleton; ``None`` disables enhancement).
    """
    archs = [a for a in (model_architectures or []) if str(a or "").strip()]
    if not archs:
        return
    try:
        kg = kg_client
        if kg is None:
            from ..recipe_kb.kg_client import get_kg_client

            kg = get_kg_client()
        if kg is None or not kg.is_available():
            return
    except Exception as exc:  # noqa: BLE001 - availability probe must never raise out
        log.warning("KG warm-start enhancement skipped (unavailable): %s", exc)
        return

    try:
        own_arch = {_arch_norm(a) for a in archs}
        # 1. Architecture-family graph: reach related architectures.
        related: set[str] = set(own_arch)
        for arch in archs:
            for node in kg.graph_traverse_safe(
                start_entity=arch,
                predicate_filter=["USES_ARCH", "VARIANT_OF"],
                max_hops=2,
                direction="both",
            ):
                if node.entity:
                    related.add(node.entity)

        # 2. Cross-recipe negatives → hard (own arch) vs advisory (related).
        already_blocked = {
            _arch_norm(b.get("patch_file"))
            for b in (ctx.get("blocked_patches") or [])
            if isinstance(b, dict)
        }
        hard: list[dict[str, Any]] = []
        advisory: list[dict[str, Any]] = []
        seen_adv: set[str] = set()
        for fact in kg.query_facts_safe(
            object=sorted(related),
            predicate=["REVERTED_ON", "DEGRADES", "CRASHES"],
            limit=100,
        ):
            patch = fact.subject
            if not patch:
                continue
            if fact.object in own_arch:
                if patch in already_blocked:
                    continue
                already_blocked.add(patch)
                hard.append({
                    "patch_file": patch,
                    "reason": f"{fact.predicate}: {fact.properties.get('error') or fact.properties.get('reason', '')}",
                    "confidence": fact.confidence,
                    "block_type": "hard",
                    "source": "kg",
                })
            else:
                if patch in seen_adv or patch in own_arch:
                    continue
                seen_adv.add(patch)
                advisory.append({
                    "patch_file": patch,
                    "reason": f"{fact.predicate} on related arch {fact.object}",
                    "confidence": round(fact.confidence * 0.6, 3),
                    "block_type": "advisory",
                    "source": "kg",
                })
        if hard:
            ctx.setdefault("blocked_patches", []).extend(hard)
        if advisory:
            ctx["advisory_blocked_patches"] = advisory

        # 3. Positive candidates for current arch+hw+fw, minus blocked.
        conditions: dict[str, Any] = {}
        if hardware:
            conditions["hw"] = hardware
        if framework:
            conditions["fw"] = framework
        blocked_now = {_arch_norm(b.get("patch_file")) for b in (ctx.get("blocked_patches") or [])}
        recommended: list[dict[str, Any]] = []
        seen_rec: set[str] = set()
        for fact in kg.query_facts_safe(
            object=sorted(own_arch),
            predicate=["IMPROVES"],
            conditions=conditions or None,
            limit=50,
        ):
            knob = fact.subject
            if not knob or knob in blocked_now or knob in seen_rec:
                continue
            seen_rec.add(knob)
            recommended.append({
                "knob": knob,
                "expected_gain": fact.gain,
                "confidence": fact.confidence,
                "source": "kg_graph",
            })
        if recommended:
            recommended.sort(key=lambda r: -(r.get("expected_gain") or 0.0))
            ctx["recommended_knobs"] = recommended

        # 3b. Journal-derived config knobs (KNOB_IMPROVES), behind a flag.
        # These carry runnable args/envs and use the fingerprint vocabulary,
        # so they ride a separate ctx key and never mix with the patch-keyed
        # ``recommended_knobs`` above.
        if _kg_guided_enabled():
            from ..recipe_kb.kg_client import generate_knob_candidates_graph_guided

            knobs = generate_knob_candidates_graph_guided(
                kg,
                architectures=archs,
                precision=precision,
                hardware=hardware,
                framework=framework,
                max_variants=8,
            )
            if knobs:
                ctx["graph_guided_knobs"] = knobs

        # 4. Validity check: flag replay patches whose VALID_FOR has lapsed.
        replay = ctx.get("recommended_replay")
        if isinstance(replay, Mapping):
            for patch in replay.get("patches") or []:
                if not isinstance(patch, dict):
                    continue
                pf = patch.get("patch_file") or ""
                if not pf:
                    continue
                for v in kg.query_facts_safe(subject=pf, predicate=["VALID_FOR"], limit=10):
                    if _validity_expired(v.properties):
                        patch["expired"] = True
                        patch["expire_reason"] = f"valid_for {v.properties.get('version', '')}".strip()
                        break
    except Exception as exc:  # noqa: BLE001 - enhancement is advisory only
        log.warning("KG warm-start enhancement degraded: %s", exc)


def _validity_expired(props: Mapping[str, Any]) -> bool:
    """Return ``True`` when a ``VALID_FOR`` fact's ``expires`` date is past.

    Only an explicit ``expires`` ISO date triggers expiry; free-form
    version ranges are intentionally not parsed here (avoids false
    positives) and are deferred to the native KG validity engine.

    Args:
        props: The ``VALID_FOR`` fact's properties.

    Returns:
        ``True`` when ``expires`` is a past date, else ``False``.
    """
    expires = str(props.get("expires") or "").strip()
    if not expires:
        return False
    try:
        exp = datetime.fromisoformat(expires.replace("Z", "+00:00"))
    except ValueError:
        return False
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    return exp < datetime.now(timezone.utc)


def _extract_patches_from_prs_tested(
    ctx: dict,
    recipe: "Mapping[str, Any] | None",
    model_architectures: "list[str] | None" = None,
) -> None:
    """Populate ctx with replayable patches and blocked patches from prs_tested."""
    if not isinstance(recipe, Mapping):
        return
    prs = recipe.get("prs_tested")
    if not isinstance(prs, list) or not prs:
        return

    patches: list[dict] = []
    blocked: list[dict] = []
    arch_set = set(model_architectures or [])

    for pr in prs:
        if not isinstance(pr, dict):
            continue
        patch_content = pr.get("patch_content", "")
        if not patch_content:
            continue
        outcome = str(pr.get("outcome", "")).upper()
        applicable_arch = pr.get("applicable_arch") or []

        # Architecture match: if applicable_arch specified, at least one must match
        if applicable_arch and arch_set:
            if not any(a in arch_set for a in applicable_arch):
                continue

        if outcome == "KEEP":
            try:
                gain = float(pr.get("measured_gain_pct") or 0.0)
            except (TypeError, ValueError):
                gain = 0.0
            if gain > 0:
                # Limit patch_content to 50KB to avoid state.json bloat.
                pc = patch_content if len(patch_content) <= 50_000 else ""
                patches.append({
                    "patch_file": str(pr.get("patch_file") or ""),
                    "patch_content": pc,
                    "patch_ref": str(pr.get("patch_ref") or ""),
                    "measured_gain_pct": gain,
                    "repo": str(pr.get("repo") or ""),
                })
        elif outcome in ("REVERT", "FAILED"):
            # Only block when applicable_arch is specified; a REVERT with
            # no arch constraint is too broad to block all models.
            if not applicable_arch:
                continue
            blocked.append({
                "patch_file": str(pr.get("patch_file") or ""),
                "reason": f"{outcome} on {', '.join(applicable_arch)} ({pr.get('measured_gain_pct', '?')}%)",
                "blocked_arch": list(applicable_arch),
                "error_class": str(pr.get("error_class") or ""),
            })

    if patches:
        patches.sort(key=lambda p: -(p.get("measured_gain_pct") or 0))
        replay = ctx.setdefault("recommended_replay", {})
        replay.setdefault("extra_server_args", "")
        replay.setdefault("extra_envs", {})
        replay["patches"] = patches
    if blocked:
        ctx["blocked_patches"] = blocked


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
) -> T0Result:
    """Run the T0 recipe-snapshot anchor.

    Mutates ``shared_state`` in place (warm_start_* fields) and persists when
    ``save_state=True``. ``session_dir`` is required. Returns a :class:`T0Result`.

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

    Returns:
        A :class:`T0Result` describing the anchor outcome.

    Raises:
        ValueError: If ``session_dir`` is ``None``.
    """
    emit = on_status or _default_status_emitter
    if session_dir is None:
        raise ValueError("run_t0_anchor requires an explicit session_dir")
    sd = Path(session_dir)

    # Hyperloom-local session id (Cortex KB session protocol retired).
    sid = (getattr(shared_state, "cortex_session_id", "") or "").strip()
    if not sid and sd is not None:
        sid = Path(sd).name

    workload = (workload or "").strip() or "unknown_model"
    hw = (hw or "").strip() or "unknown_gpu"

    # Short-circuit when already anchored (via ``warm_start_ts``); resume=True bypasses.
    if sid and not resume and (getattr(shared_state, "warm_start_ts", "") or "").strip():
        shared_state.cortex_session_id = sid
        emit(f"Cortex KB        : already anchored session_id={sid}")
        return T0Result(
            status="skipped_already",
            session_id=sid,
            workload=workload,
            hw=hw,
            warm_present=bool(getattr(shared_state, "warm_start_recipe", {})),
            pitfalls_present=bool(getattr(shared_state, "warm_start_pitfalls", [])),
            lessons_present=bool(getattr(shared_state, "warm_start_lessons", [])),
        )

    if sid:
        shared_state.cortex_session_id = sid
        if resume:
            emit(f"Cortex KB        : resumed session_id={sid}")
    began_now = not getattr(shared_state, "warm_start_ts", "")
    if began_now:
        shared_state.warm_start_ts = datetime.now(timezone.utc).isoformat(
            timespec="seconds",
        )

    # Backfill operator-tracing metadata; T0 only stamps metadata (best_config preserved, rewritten at CLOSE).
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

    # Operator-traceability + workload-shape tags into ``extras`` (skip empty/zero).
    _extras: dict[str, Any] = {}
    if _model_class:
        _extras["model_class"] = _model_class
    # Architecture-identity tags from config.json (records base architecture for fine-tunes).
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
        v = str(_extra.get(src_key) or "").strip()
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

    # Persist framework + framework_version onto SharedState so CLOSE/KEEP derives an identical cid.
    if _framework:
        shared_state.framework = _framework
    if _fw_version:
        shared_state.framework_version = _fw_version

    # Read-modify-write the LOCAL store so the stamp doesn't clobber best_config / sessions / what_worked.
    try:
        live = kb.local.get_recipe(canonical_id=cid) or {}
    except Exception as exc:  # noqa: BLE001 — defensive
        log.info("T0 anchor local get_recipe non-fatal failure: %s", exc)
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
            "prs_tested",
            "pitfalls",
            "lessons",
            "last_profiled",
            "stack_fingerprint",
            "sessions",
            "authority",
            "confidence",
            "evidence_refs",
            "provenance",
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
            prs_tested=list(live.get("prs_tested") or []),
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

    # warm_start_recipe — 4-level cascading fallback:
    #   L1: 7-tuple exact (model+hw+fw+model_type+arch+fwv+prec) → conf=1.0, replay
    #   L2: drop model → (hw+fw+model_type+arch+fwv+prec) → conf=0.95, replay
    #       "same architecture class, cross-model" — empirically +21~37%
    #   L3: drop model+fwv → (hw+fw+model_type+arch+prec) → conf=0.5, advisory_only
    #       "same arch class, any fw version" — cross-version risk, lessons only
    #   L4: existing tier fallback (relative) → conf=0.3, advisory_only
    warm_point: dict[str, Any] = {}
    warm_tier: str = "miss"
    warm_conf: float = 0.0
    warm_prefer = _build_warm_prefer(shared_state, _fw_version)

    # Pre-compute architectures slug once for L2/L3 reuse.
    from ..recipe_snapshot_constants import _architectures_slug

    _arch_slug = _architectures_slug(_architectures_val)

    # L1: full 7-tuple exact
    try:
        row = kb.get_recipe(canonical_id=cid, prefer=warm_prefer or None)
    except Exception as exc:  # noqa: BLE001
        log.info("warm-start L1 get_recipe non-fatal failure: %s", exc)
        row = None
    if isinstance(row, dict) and row and str(row.get("canonical_id") or "") == cid:
        warm_point = row
        warm_tier = "exact"
        warm_conf = 1.0
    else:
        # L2: drop model — same (hw+fw+model_type+arch+fwv+prec)
        if _model_type_val or _architectures_val:
            l2_labels = {
                "hardware": hw,
                "framework": _framework or "",
                "model_type": _model_type_val,
                "architectures": _arch_slug,
                "framework_version": _fw_version or "",
                "precision": _precision or "",
            }
            l2_labels = {k: v for k, v in l2_labels.items() if v and v not in ("unknown_model_type", "unknown_arch")}
            try:
                l2_rows = kb.search(label_match=l2_labels, limit=5)
            except Exception:  # noqa: BLE001
                l2_rows = []
            for r in l2_rows or []:
                if str(r.get("canonical_id") or "") == cid:
                    continue
                if _recipe_is_actionable(r):
                    warm_point = r
                    warm_tier = "same_arch_class"
                    warm_conf = 0.95
                    break

        # L3: drop model + framework_version → (hw+fw+model_type+arch+prec)
        if not warm_point and (_model_type_val or _architectures_val):
            l3_labels = {
                "hardware": hw,
                "framework": _framework or "",
                "model_type": _model_type_val,
                "architectures": _arch_slug,
                "precision": _precision or "",
            }
            l3_labels = {k: v for k, v in l3_labels.items() if v and v not in ("unknown_model_type", "unknown_arch")}
            try:
                l3_rows = kb.search(label_match=l3_labels, limit=5)
            except Exception:  # noqa: BLE001
                l3_rows = []
            for r in l3_rows or []:
                if str(r.get("canonical_id") or "") == cid:
                    continue
                if _recipe_is_actionable(r):
                    warm_point = r
                    warm_tier = "same_arch_any_version"
                    warm_conf = 0.5
                    break

        # L4: if L1 returned a non-exact row (dispatcher relative match)
        if not warm_point and isinstance(row, dict) and row:
            warm_point = row
            warm_tier = "relative"
            warm_conf = 0.3

    # Bare T0 anchor (identity + tracing tags, no best_config) demotes to seed_only.
    if warm_point and not _recipe_is_actionable(warm_point):
        warm_tier = "seed_only"
        warm_conf = 0.0

    # Config-donor decoupling: the identity match (warm_point) supplies priors
    # and the 7-tuple anchor; if it carries no replayable champion config,
    # borrow one from the
    # nearest same-architecture sibling so the active warm-replay can still
    # fire. The donor's cross-model transfer confidence — not the identity
    # match's confidence — governs the downstream replay gate.
    config_donor: Mapping[str, Any] | None = None
    config_donor_tier = ""
    config_donor_conf = 0.0
    _tgt_conc = getattr(shared_state, "conc", None)
    _tgt_isl = getattr(shared_state, "isl", None)
    _tgt_osl = getattr(shared_state, "osl", None)
    # A true-self (identity ``exact``) champion always replays; a cross-model
    # warm_point that happens to carry a config is a BORROW and must clear the
    # trustworthiness gate before it is adopted as the replay donor.
    if warm_point and _has_replayable_config(warm_point) and (
        warm_tier == "exact"
        or _donor_is_trustworthy(
            warm_point,
            target_arch_slug=_arch_slug,
            target_model_type=_model_type_val,
            target_conc=_tgt_conc,
            target_isl=_tgt_isl,
            target_osl=_tgt_osl,
        )
    ):
        config_donor = warm_point
        config_donor_tier = "self"
        config_donor_conf = warm_conf
    # KG-native cross-model donor (single_top, gated by GBRAIN_KG_NATIVE): when
    # a borrow is needed, prefer the strongest cross-model KNOB_IMPROVES edge
    # over the recipe-KB sibling search. Degradation-safe → recipe-KB fallback.
    if config_donor is None and warm_point:
        kg_donor = _kg_native_config_donor(
            architectures=_architectures_val if isinstance(_architectures_val, list) else None,
            precision=_precision or "",
            hardware=hw,
            framework=_framework or "",
            model_type=_model_type_val,
        )
        if kg_donor is not None:
            config_donor = kg_donor
            config_donor_tier = "kg_cross_model"
            config_donor_conf = float(kg_donor.get("confidence") or 0.0)
    if config_donor is None and warm_point:
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

    # Keep warm.json envelope shape stable for existing readers (kb_explorer, breakdown); new readers prefer shared_state.warm_start_recipe.
    warm_text = json.dumps(
        {"points": [warm_point] if warm_point else []},
        sort_keys=True,
    )
    try:
        warm_path = cortex_warm_json(sd)
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
            "raw": warm_text,
        }
    except OSError as exc:
        log.warning("warm_start snapshot write failed: %s", exc)

    # WarmStartContext: model-facing projection of the KB result (parallel to warm_start_recipe/.kb_warm.json, not a replacement); explicit status so consumers branch on hit/seed_only/miss.
    if not warm_point:
        wsc_status = "miss"
    elif warm_tier == "seed_only":
        wsc_status = "seed_only"
    else:
        wsc_status = "hit"
    warm_source = _warm_recipe_source(
        warm_point,
        kb,
        config_donor=config_donor,
        config_donor_tier=config_donor_tier,
    )
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
            model_architectures=_architectures_val if isinstance(_architectures_val, list) else None,
            hardware=hw,
            framework=_framework or "",
            precision=_precision or "",
        )
    except Exception:  # noqa: BLE001 — defensive; context is advisory
        log.exception("warm_start_context build failed")

    # warm_start_pitfalls / warm_start_lessons are embedded recipe-row fields (exact-5-tuple-only); read from warm_point.
    pitfalls_list: list[dict[str, Any]] = list(warm_point.get("pitfalls") or [])
    lessons_list: list[dict[str, Any]] = list(warm_point.get("lessons") or [])
    try:
        pit_path = cortex_pitfalls_json(sd)
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
        les_path = cortex_lessons_json(sd)
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
                "Cortex T0: SharedState.save failed (sid=%s, workload=%s)",
                sid,
                workload,
            )

    # warm_present = usable record (tier != "miss" and confidence > 0).
    warm_present = bool(warm_point) and warm_conf > 0.0
    pitfalls_present = bool(pitfalls_list)
    lessons_present = bool(lessons_list)
    if began_now:
        warm_label = f"hit:{warm_tier}@{warm_conf:.2f}" if warm_present else "seed_only" if warm_point else "empty"
        emit(
            f"Recipe KB        : session_id={sid} "
            f"workload={cid} "
            f"(warm={warm_label}, "
            f"pitfalls={len(pitfalls_list)}, "
            f"lessons={len(lessons_list)})"
        )
    return T0Result(
        status="ok" if began_now else "skipped_already" if not resume else "resumed",
        session_id=sid,
        workload=workload,
        hw=hw,
        warm_present=warm_present,
        pitfalls_present=pitfalls_present,
        lessons_present=lessons_present,
    )


__all__ = [
    "T0Result",
    "run_t0_anchor",
]
