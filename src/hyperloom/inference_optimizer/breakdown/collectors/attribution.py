# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic collectors for ``session_breakdown.json``.

Each ``collect_<section>`` is a pure function over ``session_dir`` /
``state`` / ``manifest`` returning its schema section (see :mod:`.schema`).
Collectors never mutate state, fabricate values, or raise — failures are
recorded in ``warnings`` and the section returns a best-effort partial.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from ..agent_ownership import (
    LEVER_CONFIG,
    LEVER_KERNEL,
    LEVER_UPSTREAM_PR,
    UNATTRIBUTED,
    patch_author,
    patch_lever_kind,
)
from ._common import (
    _to_float,
    phase_at,
)


#: Phase bucket -> the lever that bucket could only have moved, for rows
#: written before ``lever_kind`` was stamped. ``explore`` and ``sweep`` only
#: ever moved configuration, ``framework`` only ever landed upstream PRs, and
#: the three kernel lanes only ever shipped kernels. ``prelude`` and ``close``
#: are absent because neither carries gain.
_LEVER_BY_PHASE_BUCKET = {
    "explore": LEVER_CONFIG,
    "sweep": LEVER_CONFIG,
    "framework": LEVER_UPSTREAM_PR,
    "kernel_agent": LEVER_KERNEL,
    "gemm_tuning": LEVER_KERNEL,
    "geak": LEVER_KERNEL,
}


def _normalize_specialist_key(provenance: str) -> str:
    """Map a raw ``provenance`` to a stable specialist key.

    ``specialist:<domain>`` → bare ``<domain>``; ``legacy:<action>`` →
    ``legacy_<action>``; ``default_grid`` / ``llm_direct`` pass through;
    empty/unknown → ``"unknown"`` (so by_domain is never keyed by ``""``).

    Args:
        provenance (str): The raw provenance label.

    Returns:
        str: The normalized specialist key.
    """
    s = (provenance or "").strip()
    if not s:
        return "unknown"
    if s.startswith("specialist:"):
        return s[len("specialist:") :] or "unknown"
    if s.startswith("legacy:"):
        return f"legacy_{s[len('legacy:') :]}"
    return s


# Ordered ``(predicate, family)`` table for :func:`_action_family`. Matched
# top-to-bottom on the lowercased action label; the FIRST hit wins, so the
# order is load-bearing (e.g. the ``kernel_opt`` prefix must precede the exact
# ``==`` checks below it). Falls through to ``"other"`` when nothing matches.
_ACTION_FAMILY_TABLE: tuple[tuple[Callable[[str], bool], str], ...] = (
    (
        lambda s: s.startswith("kernel_opt") or s in {"integrate", "fusion"},
        "kernel_agent",
    ),
    # Legacy stack-entry action labels from archived sessions.
    (lambda s: s == "backends", "backends"),
    (lambda s: s == "params", "params"),
    (lambda s: s == "validate_stack", "validate"),
    (lambda s: s == "sweep", "sweep"),
    # merged explore family subsuming the legacy backends + params buckets.
    (lambda s: s == "explore", "explore"),
    # REPLAY_WARM_RECIPE: warm-recipe / recipe KB best_config replay (a prep action).
    # Its own headline row so its gain reconciles against validated_total_pct
    # instead of vanishing into the non-emitted ``other`` family. The label may
    # carry a tier suffix (``replay_warm_recipe:exact``), so match the base token.
    (lambda s: s.split(":", 1)[0] == "replay_warm_recipe", "replay_warm_recipe"),
    # FRAMEWORK: exact legacy action label. ``integrate_patch`` serves both
    # levers and pre-baseline enablement, so ``_entry_family`` resolves it from
    # entry metadata rather than blanket-crediting it here.
    (lambda s: s == "framework", "framework"),
    # GEMM_TUNING: deterministic FP8 tuner KEEPs, bucketed apart from generic
    # ``kernel`` so the dashboard can split tuner vs source-level rewrite gain.
    (lambda s: s == "gemm_tuning", "gemm_tuning"),
    # COLLECTIVE: Coordinator-gated forge collective campaigns, bucketed apart
    # from generic ``kernel`` so multi-rank communication gain gets a dedicated
    # row instead of falling through to ``other``.
    (lambda s: s == "collective", "collective"),
    # GEAK e2e: whole-pipeline KERNEL-phase optimizer, bucketed apart
    # from generic ``kernel`` so its gain gets a dedicated row instead of vanishing into
    # ``other`` or being mis-credited to a backend.
    (lambda s: s == "geak_e2e", "geak"),
)


def _action_family(action: str) -> str:
    """Map an action label to a family for source_breakdown bucketing.

    Args:
        action (str): A stack-entry / gain-ledger action label.

    Returns:
        str: One of ``kernel_agent`` / ``backends`` / ``params`` /
        ``validate`` / ``sweep`` / ``explore`` / ``replay_warm_recipe`` /
        ``framework`` / ``gemm_tuning`` / ``collective`` / ``geak``, or
        ``"other"`` when unrecognized.
    """
    s = (action or "").lower()
    for predicate, family in _ACTION_FAMILY_TABLE:
        if predicate(s):
            return family
    return "other"


def _geak_provenance_names() -> frozenset[str]:
    """Return the provenance labels that mark an entry as GEAK-owned.

    Read from the executor that stamps them
    (:data:`~hyperloom.orchestrator.actions.executors.explore._CONFIG_REPLAY_PROVENANCE`)
    so there is exactly one list of these names in the tree. Collectors also
    run offline against a tarball, where the orchestrator package may not be
    importable at all; that case falls back to the same literal rather than
    dropping GEAK entries on the floor.

    Returns:
        frozenset[str]: The lowercased GEAK provenance labels.
    """
    try:
        from hyperloom.orchestrator.actions.executors.explore import (
            _CONFIG_REPLAY_PROVENANCE,
        )
    except Exception:  # pragma: no cover - offline replay without orchestrator
        return frozenset({"geak_revalidate"})
    return frozenset(str(p).strip().lower() for p in _CONFIG_REPLAY_PROVENANCE)


def _geak_name_resolver() -> Any:
    """Return the one resolver that turns an acceptance entry into a name.

    The canonical implementation lives beside the ledger that writes these
    entries
    (:func:`~hyperloom.orchestrator.loop.coordinator_helpers._geak_spec_name`).
    Importing it here keeps the collector and the ledger from drifting into two
    spellings of the same kernel. Collectors also run offline against a
    tarball, where the orchestrator package may not be importable at all; that
    case falls back to the same field order rather than dropping names.

    Returns:
        Any: A callable taking one acceptance entry and returning its name.
    """
    try:
        from hyperloom.orchestrator.loop.coordinator_helpers import _geak_spec_name
    except Exception:  # pragma: no cover - offline replay without orchestrator

        def _geak_spec_name(spec: Any) -> str:
            if isinstance(spec, str):
                return spec.strip()
            if not isinstance(spec, dict):
                return ""
            return str(spec.get("short_name") or spec.get("kernel_id") or spec.get("cand_tag") or "").strip()

    return _geak_spec_name


def _geak_env_test() -> Any:
    """Return the one test for "this acceptance is an env selection".

    Same sourcing rule as :func:`_geak_name_resolver`: the ledger owns the
    definition, the collector borrows it, and the offline fallback repeats the
    rule rather than inventing a looser one. The rule is deliberately
    one-sided — an acceptance is env only when it *says* ``kind: env``. A
    missing ``kind`` is unknown, and unknown is admitted.

    Returns:
        Any: A callable taking one acceptance entry and returning ``True``
        only when that entry is known to be an env selection.
    """
    try:
        from hyperloom.orchestrator.loop.coordinator_helpers import geak_spec_is_env
    except Exception:  # pragma: no cover - offline replay without orchestrator

        def geak_spec_is_env(spec: Any) -> bool:
            if not isinstance(spec, dict):
                return False
            return str(spec.get("kind") or "").strip().lower() == "env"

    return geak_spec_is_env


def _geak_kernel_names(entry: dict[str, Any]) -> list[str]:
    """Return the authored-kernel names an entry carries, in row order.

    Two things were wrong with reading ``accepted_kernels`` alone.

    An acceptance lands in one of two lanes, ``accepted_kernels`` or
    ``accepted_heads``, and which one it lands in is not a property of the
    kernel. Measured over ``/shared_nfs/hyperloom-claw``, all 7 stack entries
    with ``action=geak_e2e`` have ``accepted_kernels`` empty and 4 of them
    carry their kernel in ``accepted_heads`` alone. Reading one lane did not
    under-count the gain — the gain is on the entry either way — it mislabelled
    it: :func:`_geak_contribution` returned ``"config"`` for a row that had a
    kernel running. Both lanes are read here, in the same order the ledger
    reads them.

    ``kind == "env"`` entries are excluded. Those select an existing library or
    server flag; no kernel was authored, so they are config gain and counting
    them as kernels would double-book the same win.

    Both written shapes are accepted: the ``geak_e2e`` promotion copies GEAK's
    list of dicts, the revalidation path carries a flat list of names. Anything
    unnamed is dropped rather than keyed as ``"?"``.

    Args:
        entry (dict[str, Any]): A stack / gain-ledger entry.

    Returns:
        list[str]: The kernel names, de-duplicated, order preserved.
    """
    resolve = _geak_name_resolver()
    is_env = _geak_env_test()
    lanes = list(entry.get("accepted_kernels") or []) + list(entry.get("accepted_heads") or [])
    out: list[str] = []
    for item in lanes:
        if is_env(item):
            continue
        name = resolve(item)
        if name and name not in out:
            out.append(name)
    return out


def _geak_contribution(entry: dict[str, Any]) -> str:
    """Classify what a GEAK-family entry actually had running.

    The stack rebench measures flags, env and overlay together against
    ``baseline_tput``, so a row that carried both cannot be decomposed. Saying
    which of the three cases a row is beats inventing a share for each.

    Args:
        entry (dict[str, Any]): A stack / gain-ledger entry.

    Returns:
        str: ``"kernel"`` when only an authored kernel was in play,
        ``"config"`` when only server arguments or env were, and ``"joint"``
        when both were and the measurement cannot separate them.
    """
    # A stack entry can name kernels that never ran: the promote path copies GEAK's
    # self-reported lanes, and a rebench that stripped a dead overlay still promotes on
    # its config gain. ``overlay_loaded is False`` is proof of absence, so the row is
    # config gain whatever the lanes say -- the same call the per-kernel ledger makes.
    # A missing key means the writer predates the stamp; those are left to the lanes.
    kernels = [] if entry.get("overlay_loaded") is False else _geak_kernel_names(entry)
    has_config = bool(
        str(entry.get("candidate_extra_server_args") or "").strip()
        or str(entry.get("extra_server_args") or "").strip()
        or (entry.get("extra_envs") or {})
    )
    if kernels and has_config:
        return "joint"
    if kernels:
        return "kernel"
    return "config"


def _phase_timeline(state: dict[str, Any]) -> list[tuple[float, str]]:
    """Return normalized phase boundaries ordered by timestamp."""

    history = state.get("phase_history") or []
    if not isinstance(history, list):
        return []
    timeline: list[tuple[float, str]] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        try:
            ts = float(row.get("ts_unix") or 0.0)
        except (TypeError, ValueError):
            ts = 0.0
        phase = str(row.get("to_phase") or "").strip().upper()
        if phase:
            timeline.append((ts, phase))
    timeline.sort(key=lambda item: item[0])
    return timeline


def _entry_ts(entry: dict[str, Any]) -> float | None:
    """Return an entry timestamp suitable for phase-history lookup."""

    value = entry.get("ts_unix")
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    text = str(entry.get("ts") or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _phase_at(ts_unix: float | None, timeline: list[tuple[float, str]]) -> str:
    """Return the phase active at ``ts_unix``, or ``""`` when it is unknown."""
    if ts_unix is None:
        return ""
    return phase_at(ts_unix, timeline)


def _entry_family(entry: dict[str, Any]) -> str:
    """Resolve attribution family using phase/ownership metadata when needed.

    ``integrate_patch`` lands every patch source, so the action name says
    nothing about which lever moved. The source arm owns explicitly marked
    entries; the config arm owns its own patch applications; PRELUDE
    baseline-enablement entries are prerequisites and remain non-attributable.
    Missing ownership stays unattributed rather than inferred from the phase.
    """

    action = str(entry.get("action") or "").strip().lower()
    # The GEAK revalidation dispatches as a plain ``explore`` task, so its
    # action label says ``explore`` and only its provenance says GEAK. Reading
    # the label alone files every GEAK credit under the explore family — the
    # exact mis-crediting the ``geak`` bucket was added to prevent.
    if action == "explore":
        provenance = str(entry.get("provenance") or "").strip().lower()
        if provenance in _geak_provenance_names():
            return "geak"
    if not action.startswith("integrate_patch"):
        return _action_family(action)
    if entry.get("attribution_eligible") is False or entry.get("baseline_enablement"):
        return "unattributed"
    return {
        "framework_agent": "framework",
        "explore": "explore",
    }.get(patch_author(entry), "unattributed")


def _promote_legacy_gain_entries(
    state_entries: list[Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Lift a pre-v0.7 ``list[float | None]`` gain ledger into the V1 schema.

    Recovers action/variant/ts/args from the parallel
    ``optimization_stack`` and computes ``delta_pct`` against the prior
    entry; ``None`` ledger entries stay ``None`` to keep index alignment.

    Args:
        state_entries (list[Any]): The legacy numeric gain ledger.
        state (dict[str, Any]): Parsed ``state.json`` (supplies the parallel
            ``optimization_stack``).

    Returns:
        list[dict[str, Any]]: The promoted V1 ``StackGainEntry`` rows.
    """
    stack = state.get("optimization_stack") or []
    out: list[dict[str, Any]] = []
    prev_cum = 0.0
    for i, val in enumerate(state_entries):
        cum_after: float | None
        if isinstance(val, (int, float)):
            cum_after = float(val)
        else:
            cum_after = None
        delta = (cum_after - prev_cum) if cum_after is not None else None
        se = stack[i] if i < len(stack) and isinstance(stack[i], dict) else {}
        promoted: dict[str, Any] = {
            "ts": str(se.get("ts") or ""),
            "action": str(se.get("action") or ""),
            "variant_name": se.get("variant_name") or se.get("kernel_id"),
            "stack_len_before": i,
            "stack_len_after": i + 1,
            "cum_gain_before": prev_cum,
            "cum_gain_after": cum_after,
            "delta_pct": delta,
            "extra_server_args": str(se.get("extra_server_args") or se.get("candidate_extra_server_args") or ""),
        }
        # Carry the explore join key / source forward for provenance attribution.
        fp = str(se.get("fingerprint") or se.get("variant_fingerprint") or "")
        if fp:
            promoted["fingerprint"] = fp
        prov = str(se.get("provenance") or "").strip()
        if prov:
            promoted["provenance"] = prov
        for key in (
            "source_phase",
            "phase",
            "domain",
            "gap_layer",
            "kernel_id",
            "framework_agent_authoring",
            "baseline_enablement",
            "attribution_eligible",
        ):
            if key in se:
                promoted[key] = se.get(key)
        out.append(promoted)
        if cum_after is not None:
            prev_cum = cum_after
    return out


def collect_attribution(
    state: dict[str, Any],
    geak_invocations: list[dict[str, Any]],
    adopted_kernels: list[dict[str, Any]],
    warnings: list[str],
    forge_invocations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Attribute end-to-end gains to individual optimization-stack entries.

    Prefers the authoritative ``gain_per_stack_entry`` ledger and falls back to
    reconstructing attribution from the optimization stack.

    Args:
        state: Session state mapping.
        geak_invocations: GEAK backend invocation records.
        adopted_kernels: Kernels adopted into the optimized stack.
        warnings: Mutable list that collected warnings are appended to.
        forge_invocations: Forge backend invocation records (own lane).

    Returns:
        An attribution dict mapping stack entries to their measured gains.
    """
    forge_invocations = forge_invocations or []
    # Prefer the authoritative ledger; else reconstruct from optimization_stack.
    state_entries = state.get("gain_per_stack_entry")
    state_provided = isinstance(state_entries, list) and len(state_entries) > 0
    promoted_from_legacy = False
    if state_provided and any(not isinstance(e, dict) for e in state_entries):
        # Bare numeric ledger; promote into the V1 schema.
        entries = _promote_legacy_gain_entries(state_entries, state)
        promoted_from_legacy = True
    elif state_provided:
        entries = list(state_entries)
    else:
        entries = _reconstruct_gain_ledger(state, warnings)

    # Classify the attribution lineage for an honest provenance label.
    stack = state.get("optimization_stack") or []
    stack_len = len(stack) if isinstance(stack, list) else 0
    method: str
    if state_provided:
        if promoted_from_legacy:
            method = "reconstructed"
        else:
            all_deltas_set = all(isinstance(e, dict) and e.get("delta_pct") is not None for e in state_entries)
            method = "validated" if all_deltas_set else "reconstructed"
    elif stack_len == 1:
        method = "single_source"
    elif stack_len > 1:
        method = "reconstructed"
    else:
        method = "missing"

    # Bucket entries by family for source_breakdown; validated total is the denominator.
    validated_total = _to_float(state.get("cumulative_gain_validated")) or 0.0
    family_totals: dict[str, float] = {
        "kernel_agent": 0.0,
        "sweep": 0.0,
        "other": 0.0,
        "unattributed": 0.0,
        # Legacy buckets for archived (pre-merge) sessions.
        "backends": 0.0,
        "params": 0.0,
        "validate": 0.0,
        "explore": 0.0,
        "framework": 0.0,
        "replay_warm_recipe": 0.0,
        "gemm_tuning": 0.0,
        "collective": 0.0,
        "geak": 0.0,
    }
    unattributed_actions: set[str] = set()
    for e in entries:
        if not isinstance(e, dict):
            continue
        if e.get("attribution_eligible") is False or e.get("baseline_enablement"):
            continue
        delta = _to_float(e.get("delta_pct"))
        if delta is None:
            continue
        fam = _entry_family(e)
        family_totals[fam] = family_totals.get(fam, 0.0) + max(delta, 0.0)
        if fam in {"other", "unattributed"} and delta > 0:
            unattributed_actions.add(str(e.get("action") or "<missing>"))

    # Split "kernel_agent" between backends based on adopted KEEP entries.
    forge_kept_kids = {inv.get("kernel_id") for inv in forge_invocations if inv.get("decision") == "KEEP"}
    kernel_total = family_totals.get("kernel_agent", 0.0)
    forge_total = 0.0
    for k in adopted_kernels:
        kid = k.get("kernel_id")
        gain = _to_float(k.get("e2e_gain_pct")) or 0.0
        if kid in forge_kept_kids:
            forge_total += gain
    # No Forge KEEP evidence => do NOT credit Forge. Kernel-lane gain that is
    # not tied to a Forge KEEP stays unattributed instead of being reverse-
    # inferred onto Forge (which may not have run at all this session).
    kernel_unattributed = max(kernel_total - forge_total, 0.0)
    unattributed_total = family_totals.get("unattributed", 0.0) + family_totals.get("other", 0.0)

    notes: list[str] = []
    if not state_provided:
        notes.append(
            "gain_per_stack_entry not written by Coordinator; "
            "attribution reconstructed best-effort from optimization_stack."
        )
    elif promoted_from_legacy:
        notes.append(
            "gain_per_stack_entry was a pre-v0.7 numeric ledger; "
            "promoted to V1 StackGainEntry shape using parallel data from "
            "optimization_stack (delta_pct computed as diff vs prior entry's "
            "cum_gain_after)."
        )
    if unattributed_total > 0:
        actions = ", ".join(sorted(unattributed_actions))
        note = (
            f"{unattributed_total:.2f}% validated gain could not be assigned "
            f"to a known source family (actions: {actions}); reported as "
            "unattributed."
        )
        notes.append(note)
        warnings.append(f"attribution: {note}")

    # Per-phase gain breakdown (buckets each KEEP by its active phase), and the
    # same total split by lever instead. They are two views of one number: the
    # phase view answers "when", the lever view answers "what was changed", and
    # only the second survives a phase being merged away.
    phase_breakdown, lever_breakdown = _collect_phase_breakdown(state, entries, warnings)

    return {
        "gain_per_stack_entry": entries,
        "method": method,
        "source_breakdown": {
            "forge_pct_of_total": round(forge_total, 2),
            # Kernel-lane gain with no Forge KEEP evidence; surfaced honestly
            # instead of being credited to a backend that produced no KEEP.
            "kernel_unattributed_pct_of_total": round(kernel_unattributed, 2),
            "unattributed_pct_of_total": round(unattributed_total, 2),
            "explore_pct_of_total": round(family_totals.get("explore", 0.0), 2),
            "replay_warm_recipe_pct_of_total": round(family_totals.get("replay_warm_recipe", 0.0), 2),
            "framework_pct_of_total": round(family_totals.get("framework", 0.0), 2),
            "gemm_tuning_pct_of_total": round(family_totals.get("gemm_tuning", 0.0), 2),
            "collective_pct_of_total": round(family_totals.get("collective", 0.0), 2),
            "geak_pct_of_total": round(family_totals.get("geak", 0.0), 2),
            # Legacy rows, kept so archived-session reports reconcile.
            "backends_pct_of_total": round(family_totals.get("backends", 0.0), 2),
            "params_pct_of_total": round(family_totals.get("params", 0.0), 2),
            "sweep_pct_of_total": round(family_totals.get("sweep", 0.0), 2),
            "validated_total_pct": round(validated_total, 2),
        },
        "phase_breakdown": phase_breakdown,
        "lever_breakdown": lever_breakdown,
        "notes": notes,
    }


def _collect_phase_breakdown(
    state: dict[str, Any],
    entries: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    """Per-phase gain attribution.

    Assigns each KEEP entry to the phase active at its acceptance
    timestamp (explore further splits by domain, kernel by kernel_id), except
    ``integrate_patch`` which follows explicit proposal ownership.
    Missing phase_history → everything lands under ``unattributed``.

    Args:
        state (dict[str, Any]): Parsed ``state.json``.
        entries (list[dict[str, Any]]): The per-stack gain-ledger entries.
        warnings (list[str]): Shared warnings list (mutated in place when
            ``phase_history`` is empty).

    Returns:
        tuple[dict[str, Any], dict[str, float]]: Per-phase gain buckets
        (prelude / framework / explore / kernel / gemm_tuning / sweep / close,
        plus a conditional ``unattributed``), each with a ``total_gain_pct``
        and phase-specific sub-breakdowns; and the same total split by lever
        kind instead of by phase.
    """
    # Phase timeline: for an entry ts, pick the latest row with ts_unix <= ts.
    timeline = _phase_timeline(state)

    # Explore provenance: map fingerprint -> provenance from winners_history.
    # ``scope_by_fp`` carries the specialist dial as an additive analytics tag.
    explore_search = state.get("explore_search") or {}
    provenance_by_fp: dict[str, str] = {}
    scope_by_fp: dict[str, str] = {}
    if isinstance(explore_search, dict):
        for w in explore_search.get("winners_history") or []:
            if not isinstance(w, dict):
                continue
            fp = str(w.get("fingerprint") or "")
            prov = str(w.get("provenance") or "").strip()
            if fp and prov:
                provenance_by_fp[fp] = prov
            sc = str(w.get("scope") or "").strip()
            if fp and sc:
                scope_by_fp[fp] = sc

    #: Lever totals, keyed by :data:`LEVER_KINDS` plus ``unattributed``.
    lever_buckets: dict[str, float] = {}
    phase_buckets: dict[str, dict[str, Any]] = {
        "prelude": {"total_gain_pct": 0.0},
        # by_pr keyed per adopted PR.
        "framework": {"total_gain_pct": 0.0, "by_pr": {}},
        "explore": {"total_gain_pct": 0.0, "by_domain": {}},
        "kernel_agent": {"total_gain_pct": 0.0, "by_kernel_id": {}},
        # by_tuned_file keyed on the produced CSV.
        "gemm_tuning": {"total_gain_pct": 0.0, "by_tuned_file": {}},
        # GEAK runs inside KERNEL but is bucketed apart, like gemm_tuning.
        "geak": {"total_gain_pct": 0.0, "by_contribution": {}, "by_kernel_id": {}},
        "sweep": {"total_gain_pct": 0.0},
        "close": {"total_gain_pct": 0.0},
        "unattributed": {"total_gain_pct": 0.0},
    }

    for e in entries:
        if not isinstance(e, dict):
            continue
        if e.get("attribution_eligible") is False or e.get("baseline_enablement"):
            continue
        delta = _to_float(e.get("delta_pct"))
        if delta is None or delta <= 0:
            continue
        phase = _phase_at(_entry_ts(e), timeline).lower()
        action = str(e.get("action") or "").lower()
        fam = _entry_family(e)
        # Both levers run inside FRAMEWORK_AGENT, so the live phase no longer
        # says which one moved a KEEP. The entry's own family does.
        if phase == "framework_agent":
            phase = "framework" if fam == "framework" else "explore"
        if action.startswith("integrate_patch"):
            # For this delayed application mechanism, proposal ownership is the
            # attribution phase; the acceptance timestamp is only execution
            # context and must not manufacture a kernel_agent/"?" row.
            phase = fam if fam in {"framework", "explore"} else "unattributed"
        # gemm_tuning runs inside KERNEL but is bucketed separately.
        if fam == "gemm_tuning":
            phase = "gemm_tuning"
        # So does GEAK. Its own bucket keeps the gain out of ``unattributed``,
        # where the KERNEL phase name (absent from these buckets) sent it.
        elif fam == "geak":
            phase = "geak"
        # Fall back to action family when phase_history isn't usable.
        elif phase not in phase_buckets:
            if fam in ("explore", "backends", "params"):
                phase = "explore"
            elif fam in ("kernel_agent", "collective"):
                phase = "kernel_agent"
            elif fam == "sweep":
                phase = "sweep"
            elif fam == "framework":
                phase = "framework"
            else:
                phase = "unattributed"
        bucket = phase_buckets[phase]
        bucket["total_gain_pct"] = round(
            float(bucket["total_gain_pct"]) + float(delta),
            2,
        )
        # Lever split, accumulated from the same rows and the same deltas as the
        # phase split above. It is the phase-free view of the identical total:
        # a phase says when a KEEP landed, which a delayed application makes a
        # lie; the lever says what was changed, which stays true.
        lever = patch_lever_kind(e) or _LEVER_BY_PHASE_BUCKET.get(phase, "")
        if lever:
            lever_buckets[lever] = round(
                float(lever_buckets.get(lever, 0.0)) + float(delta),
                2,
            )
        else:
            lever_buckets[UNATTRIBUTED] = round(
                float(lever_buckets.get(UNATTRIBUTED, 0.0)) + float(delta),
                2,
            )
        if phase == "explore":
            by_domain = bucket.setdefault("by_domain", {})
            fp = str(e.get("fingerprint") or e.get("variant_fingerprint") or "")
            entry_domain = str(e.get("domain") or "").strip()
            raw_prov = (
                provenance_by_fp.get(fp)
                or str(e.get("provenance") or "")
                or (f"specialist:{entry_domain}" if entry_domain else "")
                or "default_grid"
            )
            domain = _normalize_specialist_key(raw_prov)
            by_domain[domain] = round(
                float(by_domain.get(domain, 0.0)) + float(delta),
                2,
            )
            # Additive scope split; missing ``scope`` collapses into ``unspecified``.
            by_scope = bucket.setdefault("by_scope", {})
            scope_key = scope_by_fp.get(fp) or str(e.get("scope") or "") or "unspecified"
            by_scope[scope_key] = round(
                float(by_scope.get(scope_key, 0.0)) + float(delta),
                2,
            )
        elif phase == "kernel_agent":
            by_kid = bucket.setdefault("by_kernel_id", {})
            kid = str(e.get("kernel_id") or e.get("action_kernel_id") or "?")
            by_kid[kid] = round(
                float(by_kid.get(kid, 0.0)) + float(delta),
                2,
            )
        elif phase == "framework":
            # Key on the PR ref (variant_name), falling back to ``ref`` then ``?``.
            by_pr = bucket.setdefault("by_pr", {})
            pr_key = str(e.get("variant_name") or "").strip() or str(e.get("ref") or "").strip() or "?"
            by_pr[pr_key] = round(
                float(by_pr.get(pr_key, 0.0)) + float(delta),
                2,
            )
        elif phase == "geak":
            # What was running, not a guessed share of it.
            by_contrib = bucket.setdefault("by_contribution", {})
            contribution = _geak_contribution(e)
            by_contrib[contribution] = round(
                float(by_contrib.get(contribution, 0.0)) + float(delta),
                2,
            )
            by_kid = bucket.setdefault("by_kernel_id", {})
            for key in _geak_kernel_names(e):
                # One kernel per row is the normal case. When a row names
                # several, each is named at the row's whole gain — the rebench
                # measured them together and cannot say who earned what.
                by_kid[key] = round(
                    float(by_kid.get(key, 0.0)) + float(delta),
                    2,
                )
        elif phase == "gemm_tuning":
            # Key on the tuned CSV path, falling back to ``variant_name`` then ``?``.
            by_tuned = bucket.setdefault("by_tuned_file", {})
            tuned_key = str(e.get("tuned_file") or "").strip() or str(e.get("variant_name") or "").strip() or "?"
            by_tuned[tuned_key] = round(
                float(by_tuned.get(tuned_key, 0.0)) + float(delta),
                2,
            )

    # Drop the unattributed bucket when nothing landed there.
    if phase_buckets["unattributed"]["total_gain_pct"] == 0.0:
        phase_buckets.pop("unattributed", None)
    if lever_buckets.get(UNATTRIBUTED) == 0.0:
        lever_buckets.pop(UNATTRIBUTED, None)

    if not timeline:
        warnings.append("attribution.phase_breakdown: phase_history empty; gains bucketed via action family fallback")

    return phase_buckets, lever_buckets


def _reconstruct_gain_ledger(
    state: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Approximate per-stack contribution (each entry's ``gain_pct`` as its delta) when Coordinator didn't record it.

    Args:
        state (dict[str, Any]): Parsed ``state.json``.
        warnings (list[str]): Shared warnings list (kept for signature
            symmetry; not mutated here).

    Returns:
        list[dict[str, Any]]: One reconstructed gain-ledger row per stack
        entry, with cumulative gain and ``delta_pct``. Empty when there is no
        ``optimization_stack``.
    """
    stack = state.get("optimization_stack") or []
    if not isinstance(stack, list):
        return []
    cum_before = 0.0
    out: list[dict[str, Any]] = []
    for i, entry in enumerate(stack):
        if not isinstance(entry, dict):
            continue
        delta = _to_float(entry.get("gain_pct"))
        cum_after = cum_before + (delta or 0.0)
        row: dict[str, Any] = {
            "ts": str(entry.get("ts") or ""),
            "stack_len_before": i,
            "stack_len_after": i + 1,
            "action": str(entry.get("action") or ""),
            "variant_name": str(entry.get("variant_name") or ""),
            "cum_gain_before": round(cum_before, 4),
            "cum_gain_after": round(cum_after, 4),
            "delta_pct": delta,
            "extra_server_args": str(entry.get("extra_server_args") or ""),
        }
        # Preserve the explore join key / source for provenance resolution.
        fp = str(entry.get("fingerprint") or entry.get("variant_fingerprint") or "")
        if fp:
            row["fingerprint"] = fp
        prov = str(entry.get("provenance") or "").strip()
        if prov:
            row["provenance"] = prov
        out.append(row)
        cum_before = cum_after
    return out
