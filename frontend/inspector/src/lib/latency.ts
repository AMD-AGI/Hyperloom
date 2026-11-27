// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The latency measures a benchmark run carries (`latency[kind][stat]`, in ms) and the one the TTFT scatter plots.

import type { StreamItem } from "./types";

export const LATENCY_KINDS = [
  ["ttft", "TTFT"],
  ["tpot", "TPOT"],
  ["e2el", "E2E latency"],
] as const;
export const LATENCY_STATS = ["mean", "median", "p99"] as const;

export type LatencyKind = (typeof LATENCY_KINDS)[number][0];
export type LatencyStat = (typeof LATENCY_STATS)[number];
export interface LatencyPick {
  kind: LatencyKind;
  stat: LatencyStat;
}

/** The picked latency of a benchmark run, or null when the run lacks it (a baseline from a timeline has only means). */
export function latencyOf(run: StreamItem, { kind, stat }: LatencyPick): number | null {
  const table = run.latency as Partial<Record<string, Partial<Record<string, number | null>>>> | undefined;
  return table?.[kind]?.[stat] ?? null;
}

export function latencyLabel({ kind, stat }: LatencyPick): string {
  return `${stat} ${LATENCY_KINDS.find(([k]) => k === kind)?.[1]}`;
}

/** Digits for axis ticks over *span* ms: whole numbers for a wide range, decimals when a few ms is all there is. */
export const tickDigits = (span: number): number => (span >= 20 ? 0 : span >= 2 ? 1 : 2);
