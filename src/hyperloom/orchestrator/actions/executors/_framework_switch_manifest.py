# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parse and reason about a framework-rewrite switch manifest."""

from __future__ import annotations

import logging
import re
from typing import Any, NamedTuple

from hyperloom.common.env_safety import (
    BLOCKED_EXTERNAL_ENV_NAMES,
    is_launch_provided_env_name,
    is_secret_shaped_env_name,
    valid_env_key,
)

from ._framework_rewrite_evidence import _CATEGORY_TAXONOMY
from ._grid_base import TS_FAILED
from ._integrate_vocabulary import (
    ATTRIBUTION_CYCLE_BENCH,
    ATTRIBUTION_CYCLE_BROKEN,
    ATTRIBUTION_CYCLE_PENDING,
    ATTRIBUTION_REPLAY_BROKEN,
)

#: Attribution sources of a lever whose server failed with it on; it is neither pending nor bundled again.
_BROKEN_SOURCES = frozenset({ATTRIBUTION_CYCLE_BROKEN, ATTRIBUTION_REPLAY_BROKEN})


log = logging.getLogger(__name__)


# Manifest key on the specialist's done payload / the integrate_patch params.
MANIFEST_KEY = "framework_switches"

# Value of a switch whose manifest entry omits one: "on" for a flag; a size, count or threshold declares its own.
DEFAULT_SWITCH_VALUE = "1"

# Recognised rewrite categories, mirroring ``_framework_rewrite_evidence`` plus the two that are not host-observable
# and therefore never appear in evidence.
KNOWN_CATEGORIES: frozenset[str] = frozenset(_CATEGORY_TAXONOMY) | {"swap_vendor_kernel", "drop_noop_glue"}

# Cap on manifest entries.
MAX_SWITCHES = 24

# ``framework_lever_source`` of an arm that switches one active lever off.
LEAVE_ONE_OUT = "leave_one_out"


def _clean_list(raw: Any) -> list[str]:
    """Coerce ``raw`` to a list of non-empty stripped strings."""
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        return []
    return [s for s in (str(i or "").strip() for i in items) if s]


def _manifest_entries(raw: Any) -> list[dict[str, Any]] | None:
    """Switch entries from either manifest spelling, or ``None`` if ``raw`` is neither."""
    if isinstance(raw, dict):
        entries: list[dict[str, Any]] = []
        for name, body in raw.items():
            entry = dict(body) if isinstance(body, dict) else {}
            entry.setdefault("switch", name)
            entries.append(entry)
        return entries
    if isinstance(raw, (list, tuple)):
        return [dict(e) for e in raw if isinstance(e, dict)]
    return None if raw else []


class _AcceptedSwitches(NamedTuple):
    """Switch records keyed by upper-case name, with the reasons the other entries were dropped."""

    by_name: dict[str, dict[str, Any]]
    problems: list[str]


# A value no credential can be: a flag or a count. A token-cache switch reads credential-shaped by name alone.
_LITERAL_VALUE = re.compile(r"(?i)true|false|on|off|yes|no|-?\d{1,9}")


def _declared_value(entry: dict[str, Any]) -> str:
    """The entry's value as an env string, defaulting only when it gives none."""
    value = entry.get("value")
    return str(value).strip() if value not in (None, "") else DEFAULT_SWITCH_VALUE


def _accept_switches(entries: list[dict[str, Any]], reserved: set[str]) -> _AcceptedSwitches:
    """Keep the entries that name a usable rewrite switch, reporting why the rest were dropped."""
    problems: list[str] = []
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
        # A manifest may never claim these: setting one from a "rewrite switch" would silently retarget the benchmark
        # rather than toggling a code path.
        if upper in BLOCKED_EXTERNAL_ENV_NAMES:
            problems.append(f"dropped {name!r}: reserved benchmark variable, not a rewrite switch")
            continue
        value = _declared_value(entry)
        if is_secret_shaped_env_name(upper) and not _LITERAL_VALUE.fullmatch(value):
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
            "value": value,
            "category": category,
            "target": str(entry.get("target") or "").strip(),
            "evidence": str(entry.get("evidence") or "").strip(),
            "depends_on": _clean_list(entry.get("depends_on")),
            "enables": _clean_list(entry.get("enables")),
        }
    return _AcceptedSwitches(accepted, problems)


def _resolve_edge_references(accepted: dict[str, dict[str, Any]]) -> list[str]:
    """Rewrite every edge list to the upper-case names it actually reaches in ``accepted``."""
    problems: list[str] = []
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
    return problems


def _mirror_edge_directions(accepted: dict[str, dict[str, Any]]) -> None:
    """Make ``depends_on`` and ``enables`` agree, so either direction sees the whole graph."""
    for name, entry in accepted.items():
        for dep in entry["depends_on"]:
            if name not in accepted[dep]["enables"]:
                accepted[dep]["enables"].append(name)
        for enabled in entry["enables"]:
            if name not in accepted[enabled]["depends_on"]:
                accepted[enabled]["depends_on"].append(name)


def parse_manifest(
    raw: Any,
    *,
    reserved_env: "frozenset[str] | set[str] | None" = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse and validate a switch manifest."""
    entries = _manifest_entries(raw)
    if entries is None:
        return [], [f"manifest must be a list or dict, got {type(raw).__name__}"]
    if not entries:
        return [], []

    problems: list[str] = []
    if len(entries) > MAX_SWITCHES:
        problems.append(f"manifest declares {len(entries)} switches; keeping the first {MAX_SWITCHES}")
        entries = entries[:MAX_SWITCHES]

    reserved = {str(k).strip().upper() for k in (reserved_env or set())}
    accepted = _accept_switches(entries, reserved)
    switches = accepted.by_name
    problems.extend(accepted.problems)
    problems.extend(_resolve_edge_references(switches))
    _mirror_edge_directions(switches)
    problems.extend(_break_cycles(switches))

    for entry in switches.values():
        entry["enabler"] = bool(entry["enables"])

    return list(switches.values()), problems


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


# How a patch reads an environment variable: ``os.environ.get`` and
# ``os.environ[...]``, plus any ``getenv`` call, which covers ``os.getenv``
# and the C/C++ ``getenv``/``std::getenv`` a kernel-side rewrite uses.
_ENV_READ_RE = re.compile(r"""(?:environ\s*(?:\.\s*get\s*\(|\[)|\bgetenv\s*\()\s*["']([A-Za-z_][A-Za-z0-9_]*)["']""")

# Falsy defaults, which keep a name in the list. A read with any other literal
# default -- a tuning value (``os.getenv("X", "/opt/rocm")``) or an on default
# (``os.environ.get("X", "1") == "1"``, a rewrite shipped enabled) -- runs the
# same whether or not the name is set, so it is not worth reporting at all.
_OFF_DEFAULTS: frozenset[str] = frozenset({"", "0", "0.0", "false", "no", "off", "none", "null", "nullptr"})


def _read_could_default_off(tail: str) -> bool:
    """Whether a read whose argument list continues with ``tail`` has no default or a falsy one.

    ``tail`` is the rest of the line after the name. This is a filter on the
    literal text, not a finding about the patched path: an opt-out switch
    (``if not os.environ.get("HL_DISABLE_NEW")``), a write
    (``os.environ["X"] = "1"``) and a plain value read all look like this, and
    only reading the surrounding code tells them apart.
    """
    rest = tail.lstrip()
    if not rest.startswith(","):
        return True
    default = rest[1:].split(",")[0].split(")")[0].strip()
    if not default:
        return False
    if len(default) >= 2 and default[0] in "\"'" and default[-1] == default[0]:
        default = default[1:-1].strip()
    return default.lower() in _OFF_DEFAULTS


def _env_reads(text: str) -> set[str]:
    """The env names ``text`` reads with no default or a falsy one."""
    return {m.group(1).upper() for m in _ENV_READ_RE.finditer(text) if _read_could_default_off(text[m.end() :])}


def _patch_env_reads(patch_text: str) -> tuple[set[str], set[str]]:
    """Split the env reads a diff carries into the ones it adds and the ones already in the file.

    A read on a context or a removed line belongs to the file as it stands, so
    the patch did not introduce it and it says nothing about this round.
    """
    added: set[str] = set()
    preexisting: set[str] = set()
    for line in patch_text.splitlines():
        if line.startswith(("+++", "---", "@@")):
            continue
        names = _env_reads(line[1:])
        if not names:
            continue
        if line.startswith("+"):
            added |= names
        else:
            preexisting |= names
    return added, preexisting


def _could_be_a_round_switch(name: str) -> bool:
    """Whether ``name`` is one a round could set at all, by manifest or ``extra_envs``.

    A name neither route may set — a credential, a key the benchmark owns, or
    one the launch assigns per process — is read for what it holds, never as a
    switch of the patch's own, so reporting it would only be noise.
    """
    return (
        name not in BLOCKED_EXTERNAL_ENV_NAMES
        and not is_secret_shaped_env_name(name)
        and not is_launch_provided_env_name(name)
    )


def unarmed_env_reads(patch_texts: list[str], *, armed: "set[str] | frozenset[str] | None" = None) -> list[str]:
    """Return the env names the patches ADD a read of that nothing in ``armed`` sets.

    This is a lead, never a verdict. Scanning diff lines cannot tell whether a
    read keeps the patched path off: an opt-out switch
    (``if not os.getenv("HL_DISABLE_NEW")``) runs the change precisely while
    the name is unset, and a write or a value read is no switch at all. The
    names are worth reporting only beside a round that already measured as
    though its change were absent, where one of them may be the reason.

    Args:
        patch_texts: The diff text of every patch the round submitted.
        armed: Every env name the launch will set, manifest switches included.

    Returns:
        The unset names, sorted.
    """
    armed_names = {str(name).strip().upper() for name in (armed or ())}
    added: set[str] = set()
    preexisting: set[str] = set()
    for text in patch_texts:
        text_added, text_preexisting = _patch_env_reads(text)
        added |= text_added
        preexisting |= text_preexisting
    unarmed = added - preexisting - armed_names
    return sorted(name for name in unarmed if _could_be_a_round_switch(name))


def transitive_closure(name: str, switches: list[dict[str, Any]], edge: str) -> set[str]:
    """Return every switch reachable from ``name`` along the ``edge`` lists, plus ``name``.

    ``depends_on`` yields what ``name`` needs; ``enables`` yields what needs ``name``.
    """
    by_name = {entry["switch"]: entry for entry in switches}
    target = name.strip().upper()
    closure = {target}
    frontier = [target]
    while frontier:
        current = frontier.pop()
        for linked in (by_name.get(current) or {}).get(edge) or []:
            if linked not in closure:
                closure.add(linked)
                frontier.append(linked)
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
        bundle = transitive_closure(entry["switch"], switches, "depends_on")
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
    if len(switches) > 1 and frozenset(entry["switch"] for entry in switches) not in seen:
        variants.append(bundle_variant(switches, name=f"{name_prefix}_all"))
    return variants


def bundle_variant(switches: list[dict[str, Any]], *, name: str) -> dict[str, Any]:
    """Build the explore variant that switches every given lever ON together."""
    return {
        "name": name,
        "extra_envs": switch_env(switches),
        "note": f"all {len(switches)} framework rewrite levers together",
        "provenance": "framework_rewrite_lever",
        "framework_lever": "",
        "framework_lever_bundle": sorted(entry["switch"] for entry in switches),
    }


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
        removed = transitive_closure(entry["switch"], switches, "enables")
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


def unattributed_arms(variants: list[dict[str, Any]], pending: set[str]) -> list[dict[str, Any]]:
    """Keep the lever arms whose lever is in ``pending``, and the combination arm while any of its levers is."""
    return [
        v
        for v in variants
        if pending.intersection([v["framework_lever"]] if v["framework_lever"] else v["framework_lever_bundle"])
    ]


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
    "KNOWN_CATEGORIES",
    "MANIFEST_KEY",
    "MAX_SWITCHES",
    "additive_variants",
    "bundle_variant",
    "leave_one_out_variants",
    "parse_manifest",
    "summarize",
    "switch_env",
    "transitive_closure",
    "unarmed_env_reads",
    "unattributed_arms",
]


def framework_lever_grid(shared_state: Any) -> list[dict[str, Any]]:
    """Build explore variants that attribute each registered rewrite lever.

    A dormant lever is split out only of a patch that carried several, since its patch was benched with all of them
    on; dormant levers across patches get their own round (:func:`dormant_lever_bundle`).
    """
    rows = _lever_rows(shared_state)
    # Arms are built over every registered lever, so attribution only picks which run, never which envs they set.
    pending = {
        row["switch"]
        for row in rows
        if row.get("attributed_gain_pct") is None and row.get("attribution_source") not in _BROKEN_SOURCES
    }
    dormant = [row for row in rows if not row.get("default_on")]
    decomposable = [row for patch in _by_patch(dormant) if len(patch) > 1 for row in patch]
    payload: list[dict[str, Any]] = []
    for variant in additive_variants(decomposable):
        if variant["framework_lever"]:
            payload.append({**variant, "framework_lever_source": "additive"})
    for variant in leave_one_out_variants([row for row in rows if row.get("default_on")]):
        payload.append({**variant, "framework_lever_source": LEAVE_ONE_OUT})
    return unattributed_arms(payload, pending)


def dormant_lever_bundle(shared_state: Any) -> dict[str, Any] | None:
    """The variant that switches every dormant lever on together, once they come from more than one patch."""
    dormant = [row for row in _lever_rows(shared_state) if not row.get("default_on") and not _screened_out(row)]
    if len(_by_patch(dormant)) < 2:
        return None
    return bundle_variant(dormant, name="fwlever_bundle")


def _screened_out(row: dict[str, Any]) -> bool:
    """Whether the cycle bench screened the lever and found no gain, or no server, in it."""
    source = row.get("attribution_source")
    if source in _BROKEN_SOURCES:
        return True
    gain = row.get("attributed_gain_pct")
    return source == ATTRIBUTION_CYCLE_BENCH and not (isinstance(gain, (int, float)) and gain > 0)


def lever_verdict(outcome: dict[str, Any], lever_payload: list[Any], shared_state: Any, variant: Any) -> list[Any]:
    """The lever attributions one landed non-KEEP verdict carries: measured movements, or a solo lever broken."""
    attributions = framework_lever_attributions([outcome], lever_payload)
    if outcome.get("outcome") == TS_FAILED:
        attributions += solo_lever_failure(getattr(shared_state, "authored_framework_levers", None) or [], variant)
    return attributions


def solo_lever_failure(levers: list[dict[str, Any]], variant: Any) -> list[dict[str, Any]]:
    """Mark the registered lever broken when ``variant``, which failed to measure, set only that switch on."""
    envs = dict(getattr(variant, "extra_envs", None) or {})
    other_changes = (
        str(getattr(variant, "extra_server_args", "") or "").strip()
        or getattr(variant, "remove_args", None)
        or getattr(variant, "unset_envs", None)
    )
    if len(envs) != 1 or other_changes:
        return []
    ((name, value),) = envs.items()
    for row in levers or []:
        if not isinstance(row, dict) or str(row.get("switch") or "") != name.strip().upper():
            continue
        if str(value) != str(row.get("value") or DEFAULT_SWITCH_VALUE):
            return []
        return [{"switch": row["switch"], "gain_pct": None, "source": ATTRIBUTION_REPLAY_BROKEN}]
    return []


def _lever_rows(shared_state: Any) -> list[dict[str, Any]]:
    """The session's registered rewrite levers, less those the cycle bench has yet to screen."""
    rows = getattr(shared_state, "authored_framework_levers", None) or []
    return [r for r in rows if isinstance(r, dict) and r.get("attribution_source") != ATTRIBUTION_CYCLE_PENDING]


def _by_patch(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Lever rows grouped by the registration that recorded them together: one integrated patch's switches."""
    patches: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        patches.setdefault((str(row.get("specialist_task_id") or ""), str(row.get("ts") or "")), []).append(row)
    return list(patches.values())


def framework_lever_attributions(
    per_variant_outcomes: list[dict[str, Any]],
    lever_payload: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Derive each rewrite lever's own contribution from this round's outcomes."""
    if not lever_payload:
        return []
    by_name = {
        str(v.get("name") or ""): v
        for v in lever_payload
        if isinstance(v, dict) and str(v.get("framework_lever") or "")
    }
    if not by_name:
        return []
    out: list[dict[str, Any]] = []
    for row in per_variant_outcomes:
        seed = by_name.get(str(row.get("variant_name") or ""))
        if seed is None:
            continue
        # A REVERT keeps its measured movement in ``graded_gain_pct``; a regressed leave-one-out arm is that case.
        metrics = row.get("metrics") or {}
        measured = metrics.get("gain_pct", metrics.get("graded_gain_pct"))
        if not isinstance(measured, (int, float)):
            continue
        source = str(seed.get("framework_lever_source") or "")
        gain = -float(measured) if source == LEAVE_ONE_OUT else float(measured)
        out.append(
            {
                "switch": str(seed["framework_lever"]),
                "gain_pct": round(gain, 4),
                "source": source,
                "variant_name": str(row.get("variant_name") or ""),
                "outcome": str(row.get("outcome") or ""),
            }
        )
    return out
