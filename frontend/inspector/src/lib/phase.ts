// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// One colour per session phase, shared by the replay scrubber and the timeline's phase bands.

/**
 * The orchestrator's phases (`orchestrator/phases/phase_chain.py` `PHASE_NAMES`); each has a `--phase-*` colour.
 * SWEEP is the name CYCLE_END had before Hyperloom 1.2.0, kept for sessions recorded by an earlier version.
 */
const PHASES = new Set(["PRELUDE", "ENABLEMENT", "FRAMEWORK_AGENT", "KERNEL_AGENT", "CYCLE_END", "SWEEP", "CLOSE"]);

/** The CSS colour of *phase*; neutral for an unknown phase and for the unnamed time before the first one. */
export function phaseColor(phase: string | null | undefined): string {
  if (!phase || !PHASES.has(phase)) return "var(--phase-other)";
  return `var(--phase-${phase.toLowerCase().replaceAll("_", "-")})`;
}
