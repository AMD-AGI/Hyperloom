# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parse and reason about a framework-rewrite switch manifest.

A ``framework_rewrite_specialist`` delivers a source patch in which every
rewrite sits behind its own environment switch that defaults OFF, plus a manifest
declaring those switches. This module turns that manifest into the three things
the orchestrator needs from it.

**The environment to bench with.** With every switch OFF the patch is inert, so
benching the patch as applied would measure nothing. :func:`switch_env` turns the
manifest into the env that turns the rewrites on.

**Per-lever attribution.** Once a manifest is accepted, its switches become
search levers, so each rewrite gets its own measured number instead of a position
in whatever order the author happened to write them.

**Dependency-aware bundling.** Some rewrites only pay once another one is in
place: memoizing a computation whose arguments are rebuilt every iteration has a
0% cache hit rate until the allocation is hoisted out of the loop. Measured
alone, the hoist looks like nothing and a greedy accept/reject loop discards it —
taking the ceiling of everything downstream with it. The manifest's
``depends_on`` / ``enables`` edges are what let the orchestrator bench an enabler
together with what it unlocks, and they drive attribution in both directions:

* when the whole stack cleared the throughput gate, the levers are already on, so
  attribution is **leave-one-out** — remove one lever (and anything that depends
  on it) and see what the stack loses;
* when the stack did not clear the gate, the code is kept inert and attribution
  is **additive** — turn on one lever plus its dependency closure and see what it
  adds.

Pure functions over already-parsed JSON; the executor owns all I/O.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import BLOCKED_EXTERNAL_ENV_NAMES, valid_env_key


log = logging.getLogger(__name__)


# Manifest key on the specialist's done payload / the integrate_patch params.
MANIFEST_KEY = "framework_switches"

# Environment reads a patch may legitimately add without declaring a switch: rank
# topology and the framework's own already-documented configuration. Anything else
# a patch newly reads is a gate, and a gate has to be declared.
_NON_SWITCH_ENV: frozenset[str] = frozenset(
    {
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
    }
)

# ``os.environ.get("NAME"`` / ``os.getenv("NAME"`` / ``os.environ["NAME"]`` on an
# added line. Only added lines matter: an untouched read was already there.
_ENV_READ_RE = re.compile(
    r"""os\.(?:environ\.get|getenv)\(\s*["']([A-Z][A-Z0-9_]*)["']|os\.environ\[\s*["']([A-Z][A-Z0-9_]*)["']\s*\]"""
)


def undeclared_switch_gates(
    patch_paths: "list[Path] | tuple[Path, ...]",
    switches: list[dict[str, Any]],
) -> list[str]:
    """Return environment switches a patch gates on but the manifest never declares.

    The whole lever scheme rests on the manifest describing every gate the patch
    introduces. When a gate is missing from it the scheme does not fail loudly, it
    stands down: nothing is turned on for the measurement, no switch-off parity leg
    runs, and no lever is registered — so the patch is benched as an ordinary diff
    and whatever it does when "off" is never checked. A live session delivered four
    env-gated patches with no manifest at all, measured +1.4%, moved the output past
    the quality band, and none of the guarantees that exist for exactly this case
    were in play.

    Only *added* lines are scanned, so a rewrite that merely moves an existing
    ``os.environ`` read is not flagged, and rank/topology variables are exempt
    because reading them is not gating behaviour.

    Args:
        patch_paths: Unified-diff files the deliverable applies.
        switches: The parsed manifest.

    Returns:
        Sorted env names the patch gates on and the manifest omits; empty when the
        deliverable is self-consistent.
    """
    declared = {str(s.get("switch") or "").strip() for s in switches}
    found: set[str] = set()
    for path in patch_paths or ():
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            log.warning("undeclared_switch_gates: cannot read %s: %s", path, exc)
            continue
        for line in text.splitlines():
            if not line.startswith("+") or line.startswith("+++"):
                continue
            for match in _ENV_READ_RE.finditer(line):
                name = match.group(1) or match.group(2)
                if name and name not in _NON_SWITCH_ENV and name not in declared:
                    found.add(name)
    return sorted(found)

# Default value assigned to a switch whose manifest entry omits one. The
# rewrites are boolean fast paths, so "on" is the only value that matters.
DEFAULT_SWITCH_VALUE = "1"

# Recognised rewrite categories, mirroring ``_framework_rewrite_evidence`` plus
# the two that are not host-observable and therefore never appear in evidence.
KNOWN_CATEGORIES: frozenset[str] = frozenset(
    {
        "memoize_invariant",
        "hoist_loop_invariant",
        "eliminate_host_round_trip",
        "eliminate_host_sync",
        "fuse_collectives",
        "keep_device_resident",
        "swap_vendor_kernel",
        "drop_noop_glue",
    }
)

# Cap on manifest entries. A patch that claims dozens of switches is either
# unreviewable or mis-parsed; either way, benching a lever per switch would eat
# the budget.
MAX_SWITCHES = 24

# Env names a manifest may never claim: setting one of these from a "rewrite
# switch" would silently retarget the benchmark rather than toggling a code path.
FORBIDDEN_SWITCHES: frozenset[str] = BLOCKED_EXTERNAL_ENV_NAMES


class SwitchManifestError(ValueError):
    """Raised when a manifest is structurally unusable."""


def _clean_list(raw: Any) -> list[str]:
    """Coerce ``raw`` to a list of non-empty stripped strings.

    Args:
        raw: A list, a bare string, or anything else.

    Returns:
        The cleaned list; ``[]`` for unusable input.
    """
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        return []
    return [s for s in (str(i or "").strip() for i in items) if s]


def parse_manifest(
    raw: Any,
    *,
    reserved_env: "frozenset[str] | set[str] | None" = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse and validate a switch manifest.

    Rejects rather than repairs anything that would make a lever unsound: an
    invalid or forbidden env name, a dependency edge pointing outside the
    manifest, or a dependency cycle. A dropped entry is reported, not silently
    ignored, because a lever the orchestrator never registers is a rewrite that
    can never be turned on and would look like dead code to the next reader.

    Args:
        raw: The manifest as delivered — a list of entry dicts, or a dict keyed
            by switch name.
        reserved_env: Env names already meaningful to the benchmark. A switch
            colliding with one of these is dropped, since the rewrite would be
            toggled by unrelated configuration.

    Returns:
        ``(switches, problems)``. ``switches`` holds the accepted entries with
        normalised fields; ``problems`` holds one human-readable line per
        rejection.
    """
    problems: list[str] = []
    entries: list[dict[str, Any]] = []
    if isinstance(raw, dict):
        for name, body in raw.items():
            entry = dict(body) if isinstance(body, dict) else {}
            entry.setdefault("switch", name)
            entries.append(entry)
    elif isinstance(raw, (list, tuple)):
        entries = [dict(e) for e in raw if isinstance(e, dict)]
    elif raw:
        return [], [f"manifest must be a list or dict, got {type(raw).__name__}"]
    if not entries:
        return [], problems

    if len(entries) > MAX_SWITCHES:
        problems.append(f"manifest declares {len(entries)} switches; keeping the first {MAX_SWITCHES}")
        entries = entries[:MAX_SWITCHES]

    reserved = {str(k).strip().upper() for k in (reserved_env or set())}
    accepted: dict[str, dict[str, Any]] = {}
    for entry in entries:
        name = str(entry.get("switch") or entry.get("name") or "").strip()
        if not name:
            problems.append("dropped an entry with no switch name")
            continue
        if not valid_env_key(name):
            problems.append(f"dropped {name!r}: not a valid environment variable name")
            continue
        upper = name.upper()
        if upper in FORBIDDEN_SWITCHES:
            problems.append(f"dropped {name!r}: reserved benchmark variable, not a rewrite switch")
            continue
        if upper in reserved:
            problems.append(
                f"dropped {name!r}: already set by the benchmark configuration, so the rewrite "
                f"would be toggled by unrelated config"
            )
            continue
        if upper in accepted:
            problems.append(f"dropped a duplicate entry for {name!r}")
            continue
        category = str(entry.get("category") or "").strip().lower()
        if category and category not in KNOWN_CATEGORIES:
            problems.append(f"{name}: unrecognised category {category!r}; recorded as-is")
        accepted[upper] = {
            "switch": upper,
            "value": str(entry.get("value") or DEFAULT_SWITCH_VALUE),
            "category": category,
            "target": str(entry.get("target") or "").strip(),
            "evidence": str(entry.get("evidence") or "").strip(),
            "depends_on": _clean_list(entry.get("depends_on")),
            "enables": _clean_list(entry.get("enables")),
        }

    # Resolve edges against the accepted set. An edge to a switch that was
    # dropped (or never declared) cannot be honoured, and keeping it would make
    # the closure silently incomplete.
    for name, entry in accepted.items():
        for field in ("depends_on", "enables"):
            resolved: list[str] = []
            for ref in entry[field]:
                ref_upper = ref.strip().upper()
                if ref_upper == name:
                    problems.append(f"{name}: dropped a self-reference in {field}")
                    continue
                if ref_upper not in accepted:
                    problems.append(f"{name}: dropped {field} reference {ref!r} (not in this manifest)")
                    continue
                if ref_upper not in resolved:
                    resolved.append(ref_upper)
            entry[field] = resolved

    # Make the two edge directions agree. A specialist that declares only one
    # side is stating a real relationship; inferring the mirror is safer than
    # honouring half of it, because a missing ``depends_on`` is what causes an
    # enabler to be benched alone.
    for name, entry in accepted.items():
        for dep in entry["depends_on"]:
            if name not in accepted[dep]["enables"]:
                accepted[dep]["enables"].append(name)
        for enabled in entry["enables"]:
            if name not in accepted[enabled]["depends_on"]:
                accepted[enabled]["depends_on"].append(name)

    cycles = _break_cycles(accepted)
    problems.extend(cycles)

    for entry in accepted.values():
        entry["enabler"] = bool(entry["enables"])

    return list(accepted.values()), problems


def _break_cycles(accepted: dict[str, dict[str, Any]]) -> list[str]:
    """Remove ``depends_on`` edges that form a cycle.

    A cycle makes the dependency closure unbounded and means no bundle can be
    constructed, so the offending edge is dropped and reported rather than left
    to hang the closure walk.

    Args:
        accepted: Manifest entries keyed by switch name, mutated in place.

    Returns:
        One problem line per removed edge.
    """
    problems: list[str] = []
    state: dict[str, int] = {}  # 0 = unvisited, 1 = on stack, 2 = done

    def visit(name: str) -> None:
        """Depth-first walk removing back edges out of ``name``."""
        state[name] = 1
        for dep in list(accepted[name]["depends_on"]):
            marker = state.get(dep, 0)
            if marker == 1:
                accepted[name]["depends_on"].remove(dep)
                if name in accepted[dep]["enables"]:
                    accepted[dep]["enables"].remove(name)
                problems.append(f"dropped the cyclic dependency {name} -> {dep}")
                continue
            if marker == 0:
                visit(dep)
        state[name] = 2

    for name in list(accepted):
        if state.get(name, 0) == 0:
            visit(name)
    return problems


def switch_env(switches: list[dict[str, Any]], *, only: "set[str] | None" = None) -> dict[str, str]:
    """Build the environment that turns the given switches on.

    Args:
        switches: Parsed manifest entries.
        only: Restrict to these switch names; ``None`` means all of them.

    Returns:
        Mapping of switch name to its value.
    """
    wanted = {s.strip().upper() for s in only} if only is not None else None
    return {
        entry["switch"]: entry["value"]
        for entry in switches
        if wanted is None or entry["switch"] in wanted
    }


def dependency_closure(name: str, switches: list[dict[str, Any]]) -> set[str]:
    """Return every switch ``name`` transitively depends on, plus ``name``.

    Args:
        name: Switch to close over.
        switches: Parsed manifest entries.

    Returns:
        The closure, including ``name`` itself. An unknown name closes to just
        itself so a caller never has to special-case it.
    """
    by_name = {entry["switch"]: entry for entry in switches}
    target = name.strip().upper()
    closure = {target}
    frontier = [target]
    while frontier:
        current = frontier.pop()
        for dep in (by_name.get(current) or {}).get("depends_on") or []:
            if dep not in closure:
                closure.add(dep)
                frontier.append(dep)
    return closure


def dependents_closure(name: str, switches: list[dict[str, Any]]) -> set[str]:
    """Return every switch that transitively depends on ``name``, plus ``name``.

    The inverse of :func:`dependency_closure`, used for leave-one-out
    attribution: turning a lever off has to turn off everything that needed it,
    or the measurement reports the cost of a broken configuration instead of the
    lever's contribution.

    Args:
        name: Switch to close over.
        switches: Parsed manifest entries.

    Returns:
        The reverse closure, including ``name`` itself.
    """
    by_name = {entry["switch"]: entry for entry in switches}
    target = name.strip().upper()
    closure = {target}
    frontier = [target]
    while frontier:
        current = frontier.pop()
        for enabled in (by_name.get(current) or {}).get("enables") or []:
            if enabled not in closure:
                closure.add(enabled)
                frontier.append(enabled)
    return closure


def additive_variants(
    switches: list[dict[str, Any]],
    *,
    name_prefix: str = "fwlever",
) -> list[dict[str, Any]]:
    """Build explore variants that switch levers ON one bundle at a time.

    Used when the authored stack did not clear the throughput gate and the code
    was kept inert: each variant turns on one lever plus its dependency closure,
    so an enabler is never measured without the rewrite it unlocks.

    Args:
        switches: Parsed manifest entries.
        name_prefix: Prefix for generated variant names.

    Returns:
        Variant dicts (``name`` / ``extra_envs`` / ``note``), smallest bundle
        first so single-lever attribution lands before the combinations, and
        deduplicated by the set of switches each one enables.
    """
    seen: set[frozenset[str]] = set()
    variants: list[dict[str, Any]] = []
    for entry in switches:
        bundle = dependency_closure(entry["switch"], switches)
        key = frozenset(bundle)
        if key in seen:
            continue
        seen.add(key)
        envs = switch_env(switches, only=bundle)
        extras = sorted(bundle - {entry["switch"]})
        note = f"framework rewrite lever {entry['switch']}"
        if extras:
            note += f" with its dependencies ({', '.join(extras)})"
        if entry.get("category"):
            note += f" [{entry['category']}]"
        variants.append(
            {
                "name": f"{name_prefix}_{entry['switch'].lower()}",
                "extra_envs": envs,
                "note": note,
                "provenance": "framework_rewrite_lever",
                "framework_lever": entry["switch"],
                "framework_lever_bundle": sorted(bundle),
            }
        )
    variants.sort(key=lambda v: (len(v["framework_lever_bundle"]), v["name"]))

    # The full stack, when it is not already one of the bundles above. The
    # authored combination is a real hypothesis and the cheapest way to find out
    # whether the whole is worth more than its measurable parts.
    if len(switches) > 1:
        full = frozenset(entry["switch"] for entry in switches)
        if full not in seen:
            variants.append(
                {
                    "name": f"{name_prefix}_all",
                    "extra_envs": switch_env(switches),
                    "note": f"all {len(switches)} framework rewrite levers together",
                    "provenance": "framework_rewrite_lever",
                    "framework_lever": "",
                    "framework_lever_bundle": sorted(full),
                }
            )
    return variants


def leave_one_out_variants(
    switches: list[dict[str, Any]],
    *,
    name_prefix: str = "fwlever_drop",
) -> list[dict[str, Any]]:
    """Build explore variants that switch one lever bundle OFF at a time.

    Used when the authored stack cleared the throughput gate and its switches are
    already part of the running configuration. Removing a lever measures what the
    stack loses without it, which is the attribution number a fixed-order
    marginal-gain report cannot give. A lever whose removal costs nothing is dead
    weight worth dropping; one whose removal costs a lot is the real win.

    Anything that depends on the removed lever is removed with it, since leaving
    a dependent enabled without its enabler measures a broken configuration.

    Args:
        switches: Parsed manifest entries, all currently on.
        name_prefix: Prefix for generated variant names.

    Returns:
        Variant dicts carrying ``unset_envs``, deduplicated by the removed set.
        Empty when there is only one lever (removing it just reproduces the
        pre-patch baseline, which is already measured).
    """
    if len(switches) < 2:
        return []
    seen: set[frozenset[str]] = set()
    variants: list[dict[str, Any]] = []
    for entry in switches:
        removed = dependents_closure(entry["switch"], switches)
        if len(removed) >= len(switches):
            # Removing this lever removes everything, which is the pre-patch
            # baseline rather than an attribution of this lever.
            continue
        key = frozenset(removed)
        if key in seen:
            continue
        seen.add(key)
        extras = sorted(removed - {entry["switch"]})
        note = f"drop framework rewrite lever {entry['switch']}"
        if extras:
            note += f" and its dependents ({', '.join(extras)})"
        variants.append(
            {
                "name": f"{name_prefix}_{entry['switch'].lower()}",
                "unset_envs": sorted(removed),
                "note": note,
                "provenance": "framework_rewrite_lever",
                "framework_lever": entry["switch"],
                "framework_lever_removed": sorted(removed),
            }
        )
    variants.sort(key=lambda v: (len(v["framework_lever_removed"]), v["name"]))
    return variants


def summarize(switches: list[dict[str, Any]], problems: list[str]) -> str:
    """Render a one-block summary of a parsed manifest for a log or a result.

    Args:
        switches: Parsed manifest entries.
        problems: Problem lines from :func:`parse_manifest`.

    Returns:
        A plain-text summary; ``""`` when there is no manifest and no problem.
    """
    if not switches and not problems:
        return ""
    lines: list[str] = []
    if switches:
        lines.append(f"{len(switches)} framework rewrite switch(es):")
        for entry in switches:
            bits = [entry["switch"]]
            if entry.get("category"):
                bits.append(f"category={entry['category']}")
            if entry.get("target"):
                bits.append(f"target={entry['target']}")
            if entry.get("depends_on"):
                bits.append(f"depends_on={','.join(entry['depends_on'])}")
            if entry.get("enables"):
                bits.append(f"enables={','.join(entry['enables'])}")
            lines.append("  - " + "  ".join(bits))
    for problem in problems:
        lines.append(f"  ! {problem}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_SWITCH_VALUE",
    "FORBIDDEN_SWITCHES",
    "KNOWN_CATEGORIES",
    "MANIFEST_KEY",
    "MAX_SWITCHES",
    "SwitchManifestError",
    "additive_variants",
    "dependency_closure",
    "dependents_closure",
    "leave_one_out_variants",
    "parse_manifest",
    "summarize",
    "switch_env",
]
