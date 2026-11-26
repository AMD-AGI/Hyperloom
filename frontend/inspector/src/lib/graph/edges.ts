// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Graph edges: one drawn edge per node pair, and the recent bus traffic that drives its pulses.

import type { GraphEdge, StreamItem } from "../types";

/** Bus traffic counts as recent for this many seconds of session time. */
export const RECENT_SEC = 60;
const LLM_ROLES = new Set(["coordinator", "orchestration", "critic"]);

/** Bus agent name -> graph node id; the port of `agent_node` in inspector/graph.py. */
export function agentNode(name: string): string {
  if (name === "*" || name === "") return "bus";
  if (LLM_ROLES.has(name)) return name;
  if (name.startsWith("framework")) return "agent:framework";
  if (name.startsWith("kernel")) return "agent:kernel";
  return name;
}

export const pairKey = (source: string, target: string): string => `${source}>${target}`;

export interface PairEdge {
  id: string;
  source: string;
  target: string;
  /** "topic ×n" per topic, joined. */
  label: string;
  total: number;
  /** Whether the reverse pair has an edge too; the pair is bent further apart then. */
  back: boolean;
}

/** Groups the server's per-topic edges into one per ordered pair. */
export function pairEdges(edges: GraphEdge[]): PairEdge[] {
  const pairs = new Map<string, GraphEdge[]>();
  for (const e of edges) {
    const k = pairKey(e.source, e.target);
    pairs.set(k, [...(pairs.get(k) ?? []), e]);
  }
  return [...pairs.entries()].map(([id, list]) => ({
    id,
    source: list[0].source,
    target: list[0].target,
    label: list.map((e) => `${e.label} ×${e.count}`).join(", "),
    total: list.reduce((x, e) => x + e.count, 0),
    back: pairs.has(pairKey(list[0].target, list[0].source)),
  }));
}

/** Bus events per ordered pair in the `RECENT_SEC` before `now`; events are in release order. */
export function recentRates(events: StreamItem[], now: number): Map<string, number> {
  const rates = new Map<string, number>();
  for (let i = events.length - 1; i >= 0; i--) {
    const ev = events[i];
    if (ev._t > now) continue;
    if (ev._t < now - RECENT_SEC) break;
    const k = pairKey(agentNode(String(ev.from_agent ?? "")), agentNode(String(ev.to_agent ?? "")));
    rates.set(k, (rates.get(k) ?? 0) + 1);
  }
  return rates;
}

/** Pulses for `n` recent events: how many dots travel the edge, and how long one pass takes (seconds). */
export function pulses(n: number): { count: number; dur: number } {
  if (n <= 0) return { count: 0, dur: 0 };
  const level = Math.log2(1 + n);
  return { count: Math.min(3, 1 + Math.floor(level / 2)), dur: Math.max(0.8, 3 - level * 0.5) };
}

/** Stroke width by total count, on a log scale. */
export const strokeWidth = (total: number): number => Math.min(5, 1 + Math.log2(Math.max(1, total)));
