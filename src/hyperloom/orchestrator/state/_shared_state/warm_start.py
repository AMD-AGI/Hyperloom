# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The blocks of the ``=== Warm start ===`` prompt section: the matched recipe, its replayable config, its priors and its pitfalls."""

from __future__ import annotations

from typing import Any

# Pitfall rows rendered in full; the rest are counted.
_PITFALLS_RENDERED = 5

# Free-text fields are bounded before they reach the prompt.
_FIELD_CHARS = 240

# Match statuses that mean nothing has been measured for this workload/hw yet.
FIRST_SESSION_LINES: dict[str, str] = {
    "miss": "recipe: none — first session for this workload/hw",
    # T0 seeds a row for every session, so ``seed_only`` — not ``miss`` — is what a
    # genuine first session looks like: a row exists, nothing measured it.
    "seed_only": "recipe: seed only — first session for this workload/hw",
}


def as_float(value: Any) -> float:
    """Coerce a warm-start context number, 0.0 when absent or unparseable."""
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def match_facts(match: Any) -> list[str]:
    """How good the match is: its tier and its confidence, each omitted when unset."""
    fields = match if isinstance(match, dict) else {}
    facts: list[str] = []
    tier = str(fields.get("tier") or "").strip()
    if tier:
        facts.append(f"tier={tier}")
    confidence = as_float(fields.get("confidence"))
    if confidence > 0:
        facts.append(f"confidence={confidence:.2f}")
    return facts


def donor_attribution(replay: dict[str, Any]) -> str:
    """Name the donor a borrowed config came from; empty for this session's own measurements.

    A borrowed config is another workload's measurement; say so, or the numbers
    above read as this session's own history.
    """
    config_tier = str(replay.get("config_tier") or "").strip()
    if not config_tier or config_tier == "self":
        return ""
    donor = str(replay.get("donor_model") or replay.get("donor_canonical_id") or "?")
    donor_conf = as_float(replay.get("config_confidence"))
    suffix = f", confidence={donor_conf:.2f}" if donor_conf > 0 else ""
    return f"  · borrowed from {donor} (config_tier={config_tier}{suffix})"


def replay_lines(ctx: dict[str, Any]) -> list[str]:
    """The config this match recommends replaying: what it reached, what it sets, and whose measurement it is."""
    recommended = ctx.get("recommended_replay")
    replay = recommended if isinstance(recommended, dict) else {}
    if not replay:
        # Remote Recipe hits replay through the section SDKs, so the context
        # carries no config; the priors below are still real.
        return ["  · (no replayable config on this match)"]
    out: list[str] = []
    best_tput = as_float(replay.get("best_throughput"))
    if best_tput > 0:
        out.append(f"  · best_throughput={best_tput:.1f}")
    args = str(replay.get("extra_server_args") or "").strip()
    if args:
        out.append(f"  · extra_server_args={args[:_FIELD_CHARS]}")
    envs = replay.get("extra_envs")
    if isinstance(envs, dict) and envs:
        rendered = " ".join(f"{k}={v}" for k, v in envs.items())
        out.append(f"  · extra_envs={rendered[:_FIELD_CHARS]}")
    gain = as_float(replay.get("expected_gain_pct"))
    if gain > 0:
        out.append(f"  · expected_gain={gain:.2f}%")
    donor = donor_attribution(replay)
    if donor:
        out.append(donor)
    return out


def priors_line(ctx: dict[str, Any]) -> str:
    """How many priors of each kind the match carries, omitting the kinds it has none of."""
    counts = [
        (label, len(ctx.get(key) or []))
        for label, key in (("proven", "proven_prior"), ("avoid", "do_not_repeat"), ("lessons", "lessons"))
    ]
    live = [f"{label}={n}" for label, n in counts if n]
    if not live:
        return ""
    return f"  · priors: {' '.join(live)}"


def pitfall_lines(ctx: dict[str, Any]) -> list[str]:
    """The pitfalls recorded against this match, capped, under a header counting what rendered.

    The header counts the rows that render, not the rows that exist: a header
    claiming "pitfalls (3)" above nothing is the same class of lie this block had.
    """
    rendered: list[str] = []
    elided = 0
    for entry in ctx.get("pitfalls") or []:
        if not isinstance(entry, dict):
            continue
        description = str(entry.get("description") or "").strip()
        if not description:
            continue
        if len(rendered) >= _PITFALLS_RENDERED:
            elided += 1
            continue
        severity = str(entry.get("severity") or "").strip()
        suffix = f" (severity={severity})" if severity else ""
        rendered.append(f"  · {description.splitlines()[0].strip()[:_FIELD_CHARS]}{suffix}")
    if not rendered:
        return []
    out = [f"pitfalls ({len(rendered)}):", *rendered]
    if elided:
        out.append(f"  · (+{elided} more elided; see state.json `warm_start_context`)")
    return out
