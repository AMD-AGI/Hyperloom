// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The token chart's data: `view.tokens.series` (tokens.py) decoded into per-bucket series of tokens, by token kind
// or by agent, then stacked, cumulatively or as a rate. Pure functions.

import type { TokenSeries, TokenTerm, TokenTotals } from "./types";

export const KINDS = ["input", "cache_write", "cache_read", "output", "reasoning"] as const;
export type Kind = (typeof KINDS)[number];
export type Mode = "cumulative" | "rate";
export type Group = "kind" | "node";
export type Terms = Record<string, TokenTerm> | undefined;

/** What the Tokens tab calls *key* (tokens.py `TERMS`), or the key itself from a server that sends no terms. */
export const termLabel = (terms: Terms, key: string): string => terms?.[key]?.label ?? key;
/** The definition of *key*, if the server sent one. */
export const termText = (terms: Terms, key: string): string | undefined => terms?.[key]?.definition;

// Each color clearly apart from every other as a chart band, on the light and the dark panel alike.
export const KIND_COLOR: Record<Kind, string> = {
  input: "#3b82f6",
  cache_write: "#7e22ce",
  cache_read: "#65a30d",
  output: "#ea580c",
  reasoning: "#fbbf24",
};
// Bottom to top. In an agentic run cache reads dwarf the rest; at the base they leave the thin bands readable on top.
const KIND_STACK: Kind[] = ["cache_read", "cache_write", "input", "output", "reasoning"];
// Shown only when there is some.
const OPTIONAL: ReadonlySet<string> = new Set(["reasoning"]);
const NODE_COLORS = ["#1d4ed8", "#f97316", "#22c55e", "#a21caf", "#fcd34d", "#b91c1c", "#f472b6", "#22d3ee"];
export const OTHER = "other";
const OTHER_COLOR = "#8994a3";
/** tokens.py's key for the calls that name no model. */
export const NO_MODEL = "unknown";
export interface Series {
  key: string;
  label: string;
  color: string;
  /** Tokens per bucket. */
  values: number[];
  total: number;
}

export interface Decoded {
  /** Bucket count; bucket i spans `start + i * step` to the next. */
  n: number;
  series: Series[];
  calls: number[];
}

export function bucketCount(s: TokenSeries): number {
  const last = s.rows.reduce((m, r) => Math.max(m, r[0]), -1);
  return Math.max(1, Math.ceil((s.end - s.start) / s.step), last + 1);
}

const zeros = (n: number) => new Array<number>(n).fill(0);
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

/**
 * Per-bucket series of *s* in tokens, restricted to the graph node *filter* when set: one per token kind
 * (reasoning only when there is some) labelled by *terms*, or one per agent node, the largest
 * first, past `maxNodes` folded into "other".
 */
export function decode(
  s: TokenSeries,
  group: Group,
  filter: string | null,
  { maxNodes = NODE_COLORS.length, terms }: { maxNodes?: number; terms?: Terms } = {},
): Decoded {
  const n = bucketCount(s);
  const calls = zeros(n);
  const kinds = s.kinds;
  const byKind = new Map<string, number[]>(kinds.map((k) => [k, zeros(n)]));
  const byNode = new Map<string, number[]>();
  const weight = new Map<string, number>();
  for (const row of s.rows) {
    const [i, ni] = row;
    const node = s.nodes[ni];
    if (filter && node !== filter) continue;
    let all = 0;
    kinds.forEach((k, j) => {
      const v = row[2 + j] ?? 0;
      byKind.get(k)![i] += v;
      all += v;
    });
    if (!byNode.has(node)) byNode.set(node, zeros(n));
    byNode.get(node)![i] += all;
    weight.set(node, (weight.get(node) ?? 0) + sum(row.slice(2, 2 + s.kinds.length)));
    calls[i] += row[2 + s.kinds.length];
  }
  if (group === "kind") {
    const series = KIND_STACK.filter((k) => byKind.has(k))
      .map((k) => ({ key: k, label: termLabel(terms, k), color: KIND_COLOR[k], values: byKind.get(k)!, total: 0 }))
      .map((x) => ({ ...x, total: sum(x.values) }))
      .filter((x) => !OPTIONAL.has(x.key) || x.total > 0);
    return { n, series, calls };
  }
  // Colours follow the node's name, so they hold while the totals reorder the stack.
  const names = [...s.nodes].sort();
  const color = (node: string) => NODE_COLORS[names.indexOf(node) % NODE_COLORS.length];
  const ranked = [...byNode.entries()]
    .map(([key, values]) => ({ key, label: key, color: color(key), values, total: sum(values) }))
    .sort((a, b) => weight.get(b.key)! - weight.get(a.key)! || a.key.localeCompare(b.key));
  if (ranked.length <= maxNodes) return { n, series: ranked, calls };
  const rest = ranked.slice(maxNodes - 1);
  const other = zeros(n);
  for (const r of rest) r.values.forEach((v, i) => (other[i] += v));
  const folded = { key: OTHER, label: `${rest.length} others`, color: OTHER_COLOR, values: other, total: sum(other) };
  return { n, series: [...ranked.slice(0, maxNodes - 1), folded], calls };
}

export interface Layer extends Series {
  lo: number[];
  hi: number[];
}

/**
 * Stacks the visible series. Cumulative layers have `n + 1` points, at the start and at each bucket's end; rate
 * layers one value per bucket, in tokens per minute.
 */
export function stack(series: Series[], hidden: ReadonlySet<string>, mode: Mode, step: number) {
  const n = series[0]?.values.length ?? 0;
  const base = zeros(mode === "cumulative" ? n + 1 : n);
  const layers: Layer[] = [];
  for (const s of series) {
    if (hidden.has(s.key)) continue;
    const v = mode === "cumulative" ? cumulative(s.values) : s.values.map((x) => (x * 60) / step);
    const lo = base.slice();
    v.forEach((x, i) => (base[i] += x));
    layers.push({ ...s, lo, hi: base.slice() });
  }
  return { layers, max: Math.max(0, ...base) };
}

/** Running sums, from 0 before the first bucket. */
export function cumulative(values: number[]): number[] {
  const out = [0];
  for (const v of values) out.push(out[out.length - 1] + v);
  return out;
}

/** 0 to at least *max* in about *n* round steps (1, 2, 2.5 or 5 times a power of ten). */
export function valueTicks(max: number, n = 4): number[] {
  if (!(max > 0)) return [0, 1];
  const raw = max / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((x) => x >= raw * (1 - 1e-9)) ?? 10 * mag;
  const out = [0];
  while (out[out.length - 1] < max * (1 - 1e-9)) out.push(out.length * step);
  return out;
}

/** Cache reads over everything sent as input (uncached, cache writes, cache reads); null before any input. */
export function cacheHit(t: TokenTotals): number | null {
  const sent = t.input + t.cache_write + t.cache_read;
  return sent > 0 ? t.cache_read / sent : null;
}
