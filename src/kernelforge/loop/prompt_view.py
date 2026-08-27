# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compact, state-driven prompt view for the long-horizon forge-loop.

Renders a small "Long-Horizon Memory" header from the durable run state and
recent events, plus explicit pointers to the on-disk detail files. The goal is
the opposite of stuffing history into context: the prompt carries only the
overview an implementer needs to pick the next move, and tells it exactly which
files to Read when it needs the full error, diff, or profile of a past attempt.

This complements (does not replace) the candidate-archive digest: the digest
still carries curated full diffs, while this header carries the resumable
control state (best / stall / phase) and the retrieval map.
"""

from __future__ import annotations

from kernelforge.loop.run_state import RunState

# Canonical on-disk locations, shown to the agent so it can Read detail on
# demand. Relative to the loop workspace root.
_ARCHIVE_REL = "forge_experiments/candidates"
_HANDOFFS_REL = "forge_experiments/handoffs"
_STATE_REL = "forge_experiments/run_state.json"
_EVENTS_REL = "forge_experiments/events.jsonl"

# How many pins the retrieval map names. The state caps the pin list higher, so
# this is a prompt-budget slice of it rather than the cap itself.
_MAX_RENDERED_PINS = 6


def _fmt_ms(value: float | None) -> str:
    try:
        return f"{float(value):.4f} ms"
    except (TypeError, ValueError):
        return "?"


def _render_pins(state: RunState, result_events: list[dict]) -> list[str]:
    """Render the pinned iterations the map points at, best lineage first.

    ``run_state.pin_iteration`` holds the iteration behind the current best
    against eviction by later near-misses, and identifies it as
    ``state.best.iteration``. That pin is rendered first and marked, so a slice
    taken for prompt budget cannot drop the one pin held for this map.

    ``RunState.pinned_iterations`` carries iteration numbers only, so a pin's
    measured mean case speedup comes from the best record or from the supplied
    outcome events. A pin older than that event window renders as its iteration
    number alone.
    """
    pinned = list(state.pinned_iterations)
    if not pinned:
        return []

    best = state.best.iteration
    holds_best = best in pinned
    head = [best] if holds_best else []
    others = [iteration for iteration in pinned if iteration != best]
    # Guarded rather than sliced directly: ``others[-0:]`` is the whole list, so
    # a budget of zero would render every pin instead of none.
    recent_budget = max(0, _MAX_RENDERED_PINS - len(head))
    selected = head + (others[-recent_budget:] if recent_budget else [])

    speedups = {
        int(event["iter"]): event["mean_case_speedup"]
        for event in result_events
        if event.get("mean_case_speedup") is not None
    }
    if holds_best and state.best.mean_case_speedup is not None:
        # The best record carries the authoritative post-decision score, which
        # outlives the bounded event window the other pins are scored from.
        speedups[best] = state.best.mean_case_speedup

    rendered: list[str] = []
    for iteration in selected:
        parts = [str(iteration)]
        if holds_best and iteration == best:
            parts.append("best")
        speedup = speedups.get(iteration)
        if speedup is not None:
            parts.append(f"{float(speedup):.6f}x")
        rendered.append(" ".join(parts))
    return rendered


def _recent_line(event: dict) -> str:
    """One compact fact line for a recent iteration-result event."""
    it = event.get("iter", "?")
    decision = str(event.get("decision") or "").strip()
    plan = str(event.get("plan") or "").replace("\n", " ").strip()[:60]
    parts = [f"iter {it} {decision}".rstrip()]
    if plan:
        parts.append(plan)
    mean_case_speedup = event.get("mean_case_speedup")
    if mean_case_speedup is not None:
        parts.append(f"mean case speedup={float(mean_case_speedup):.6f}x")
    wall = event.get("wall_ms")
    if wall is not None:
        parts.append(f"wall={_fmt_ms(wall)}")
    err = str(event.get("error_sig") or "").replace("\n", " ").strip()[:80]
    if err:
        parts.append(f"error: {err}")
    return " | ".join(parts)


# How many recent attempt lines the header renders. Named so the loop can size
# its outcome window against it without reading this signature back.
MAX_RECENT_ATTEMPT_LINES = 6


def render_long_horizon_header(
    state: RunState,
    recent_events: list[dict],
    *,
    max_recent: int = MAX_RECENT_ATTEMPT_LINES,
    max_chars: int = 4000,
    include_handoffs: bool = False,
) -> str:
    """Render the compact long-horizon memory header, or "" when state is empty.

    Args:
        state: The durable run state (control checkpoint).
        recent_events: Recent factual events (oldest first); only
            iteration-result rows are shown.
        max_recent: Max recent attempt lines to include.
        max_chars: Target ceiling on the rendered header size. Recent attempts
            are removed first. The essential control state and retrieval map may
            exceed an unrealistically small budget rather than being truncated.
    """
    # Render nothing until there is substantive history, so the loop's
    # cold-start prompt (iteration 1, before any result) is unchanged. A bare
    # baseline/iteration_started marker is not enough to warrant the header.
    has_history = (
        state.best.iteration > 0
        or bool(state.best.commit_hash)
        or state.stall.unresolved_stall_iters > 0
        or any(e.get("type") == "iteration_result" for e in recent_events)
    )
    if not has_history:
        return ""

    # Control-state lead: the compact overview the implementer acts on.
    lead: list[str] = [
        "## Long-Horizon Memory (state-driven; full detail on disk)",
        f"Phase: {state.phase}",
    ]

    # Only claim a "best" once a real KEEP exists; before that, surface the
    # baseline so the agent still knows the bar to beat.
    if state.best.iteration > 0 or state.best.commit_hash:
        label = "validated KB warm-start" if state.best.source == "warm_start" else f"iter {state.best.iteration}"
        speedup_text = f"{state.best.mean_case_speedup:.6f}x" if state.best.mean_case_speedup is not None else "?"
        best_line = f"Current best: {label}, mean case speedup {speedup_text}, raw mean {_fmt_ms(state.best.wall_ms)}"
        if state.baseline_wall_ms is not None:
            best_line += f" (baseline {_fmt_ms(state.baseline_wall_ms)})"
        if state.best.plan:
            best_line += f' — plan: "{state.best.plan}"'
        lead.append(best_line)
    elif state.baseline_wall_ms is not None:
        lead.append(
            "Baseline: mean case speedup 1.000000x, raw mean "
            f"{_fmt_ms(state.baseline_wall_ms)} (no kept improvement yet)"
        )

    if state.stall.unresolved_stall_iters > 0:
        lead.append(f"Stall: {state.stall.unresolved_stall_iters} iteration(s) without improvement")

    # Iteration outcomes feed both the pin hint below and the recent attempts.
    result_events = [e for e in recent_events if e.get("type") == "iteration_result"]

    # Retrieval map — always kept, so the agent always knows where the full
    # detail lives even if the recent list is trimmed for budget.
    pins = _render_pins(state, result_events)
    pin_hint = f" (pinned: {', '.join(pins)})" if pins else ""
    retrieval: list[str] = [
        "Full detail lives on disk. Read on demand instead of guessing:",
        f"- {_STATE_REL} — current control state",
        f"- {_EVENTS_REL} — append-only event history",
        f"- {_ARCHIVE_REL}/index.jsonl — one summary row per attempt",
        f"- {_ARCHIVE_REL}/iter_NNN/"
        + "{kernel.py,change.diff,validation.txt,meta.json}"
        + f" — full attempt detail{pin_hint}",
        "- analysis/<commit>/ — evidence",
    ]
    if include_handoffs:
        retrieval.append(f"- {_HANDOFFS_REL}/iter_NNN.json — structured iteration handoffs")

    # Recent factual attempts — summaries only, never full logs.
    recent_lines = [f"- {_recent_line(e)}" for e in result_events[-max_recent:]]

    def _assemble(recent: list[str]) -> str:
        parts = list(lead)
        if recent:
            parts.append("")
            parts.append("Recent attempts (facts; read files for detail):")
            parts.extend(recent)
        parts.append("")
        parts.extend(retrieval)
        return "\n".join(parts).strip()

    # Enforce the ceiling by dropping the OLDEST recent line first; the lead +
    # retrieval map are always kept (they are what make the loop resumable).
    header = _assemble(recent_lines)
    while recent_lines and len(header) > max_chars:
        recent_lines.pop(0)
        header = _assemble(recent_lines)

    # The fixed control state + retrieval map is the minimum useful view. A hard
    # tail slice would remove the paths precisely when the caller supplied a
    # budget smaller than that minimum, leaving the agent unable to retrieve any
    # detail. Prefer a small, explicit budget overrun to returning a broken map.
    return header
