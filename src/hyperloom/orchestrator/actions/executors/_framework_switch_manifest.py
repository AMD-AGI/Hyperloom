# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parse and reason about a framework-rewrite switch manifest."""

from __future__ import annotations

import logging
from typing import Any

from hyperloom.common.env_safety import (
    BLOCKED_EXTERNAL_ENV_NAMES,
    is_secret_shaped_env_name,
    valid_env_key,
)


log = logging.getLogger(__name__)


# Manifest key on the specialist's done payload / the integrate_patch params.
MANIFEST_KEY = "framework_switches"

# Default value assigned to a switch whose manifest entry omits one. The
# rewrites are boolean fast paths, so "on" is the only value that matters.
DEFAULT_SWITCH_VALUE = "1"

# Recognised rewrite categories, mirroring ``_framework_rewrite_evidence`` plus the two that are not host-observable
# and therefore never appear in evidence.
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

# Cap on manifest entries.
MAX_SWITCHES = 24

# Env names a manifest may never claim: setting one of these from a "rewrite switch" would silently retarget the
# benchmark rather than toggling a code path.
FORBIDDEN_SWITCHES: frozenset[str] = BLOCKED_EXTERNAL_ENV_NAMES


def _clean_list(raw: Any) -> list[str]:
    """Coerce ``raw`` to a list of non-empty stripped strings."""
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
    """Parse and validate a switch manifest."""
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
        if is_secret_shaped_env_name(upper):
            problems.append(f"dropped {name!r}: credential-shaped name, not a rewrite switch")
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

    # Resolve edges against the accepted set.
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

    # Make the two edge directions agree.
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
    """Remove ``depends_on`` edges that form a cycle."""
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
    """Build the environment that turns the given switches on."""
    wanted = {s.strip().upper() for s in only} if only is not None else None
    return {entry["switch"]: entry["value"] for entry in switches if wanted is None or entry["switch"] in wanted}


def dependency_closure(name: str, switches: list[dict[str, Any]]) -> set[str]:
    """Return every switch ``name`` transitively depends on, plus ``name``."""
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
    """Return every switch that transitively depends on ``name``, plus ``name``."""
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
    """Build explore variants that switch levers ON one bundle at a time."""
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

    # The full stack, when it is not already one of the bundles above.
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
    """Build explore variants that switch one lever bundle OFF at a time."""
    if len(switches) < 2:
        return []
    seen: set[frozenset[str]] = set()
    variants: list[dict[str, Any]] = []
    for entry in switches:
        removed = dependents_closure(entry["switch"], switches)
        if len(removed) >= len(switches):
            # Removing this lever removes everything, which is the pre-patch baseline rather than an attribution of
            # this lever.
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
    """Render a one-block summary of a parsed manifest for a log or a result."""
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
    "additive_variants",
    "dependency_closure",
    "dependents_closure",
    "leave_one_out_variants",
    "parse_manifest",
    "summarize",
    "switch_env",
]
