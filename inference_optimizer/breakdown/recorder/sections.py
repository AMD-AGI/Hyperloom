# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Per-section wire-shape registry for the breakdown recorder.

Each ``session_breakdown.json`` section has exactly one owning producer, so
there is never cross-producer write contention. A section is one of:

* ``singleton`` — one final dict; the owner rewrites its own file on update
  (last write by ``ts`` wins at assembly time).
* ``item`` — an append-only event stream concatenated into a list (ordered by
  ``seq`` then ``ts``) at assembly time.

Derived sections (see :data:`DERIVED_SECTIONS`) are NOT written by producers
during the run; they are computed at finalize from in-memory ``SharedState``
(the Coordinator owns every input), so they never appear as fragments.
"""

from __future__ import annotations

from typing import Literal

SectionShape = Literal["item", "singleton"]

# Producer-written sections and their fragment shape. Payloads match the
# corresponding ``schema.py`` TypedDict so assembly is structure-preserving.
SECTION_SHAPES: dict[str, SectionShape] = {
    "session":                     "singleton",
    "workload":                    "singleton",
    "baseline":                    "singleton",
    "final":                       "singleton",
    "phase_timeline":              "item",
    "geak_invocations":            "item",
    "oob_invocations":             "item",
    "forge_invocations":           "item",
    "kernel_lifecycle":            "singleton",
    "explore_search":              "singleton",
    "sweep":                       "singleton",
    "critic_robustness":           "singleton",
    # Author-time item substreams composed into the ``critic_robustness``
    # singleton at assembly (recorded per-iteration so the backend's workdir
    # pruning never erases history).
    "critic_iterations":           "item",
    "robustness_signals":          "item",
    "telemetry":                   "singleton",
    "kb_provenance":               "singleton",
    "specialist_runs":             "item",
    "optimization_stack":          "item",
    "kernel_roofline":             "singleton",
    "kernel_optimization_summary": "singleton",
    "conc_sweep_summary":          "singleton",
    "roofline":                    "item",
    "roofline_progress":           "singleton",
    # Kernel-major lifecycle substreams. Recorded by their respective owners at
    # author time and folded into the ``kernel_journey`` view at assembly (same
    # compose-on-read pattern as ``critic_robustness``); none of these leak into
    # the breakdown envelope on their own.
    "kernel_discovery":            "item",  # one per hot-kernel discovery run (tracelens/roofline)
    "kernel_dispatch":             "item",  # one per kernel: dispatched? which backends?
    "kernel_backend_result":       "item",  # one per backend attempt (geak/oob)
    "kernel_e2e":                  "item",  # one per kernel: e2e integrate gain
    # Authoritative external-tool versions (geak/tracelens/claude/codex/...),
    # one item per tool (idempotent by tool name); folded into the top-level
    # ``versions`` map at assembly.
    "versions":                    "item",
}

# Sections computed at finalize from in-memory state, never written as
# fragments. Listed so the assembler can distinguish "expected absent" from
# "missing producer".
DERIVED_SECTIONS: frozenset[str] = frozenset({
    "capability_summary",
    "attribution",
    "phase_segments",
    "source_files",
})


def section_shape(section: str) -> SectionShape | None:
    """Return the declared shape for ``section`` (``None`` if unregistered)."""
    return SECTION_SHAPES.get(section)


__all__ = [
    "DERIVED_SECTIONS",
    "SECTION_SHAPES",
    "SectionShape",
    "section_shape",
]
