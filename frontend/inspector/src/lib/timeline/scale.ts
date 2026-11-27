// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Timeline geometry: the default range, idle-gap compression, the piecewise time <-> x scale, axis ticks, pod start
// and resume markers, and the swimlane order. Pure functions, so the view can recompute them every animation frame
// while a zoom springs.

import { agentNode } from "../graph/edges";
import { GROUP_ROW } from "../graph/layout";
import type { Clock, GraphNode, RunInfo, StreamItem, Task, Timeline } from "../types";

export const LANE = 22;
export const AXIS = 22;
export const LABEL = 130;
export const GAP_PX = 28;
export const GAP_MIN = 300;
/** Narrowest zoom, in seconds. */
export const MIN_SPAN = 20;

export interface Range {
  t0: number;
  t1: number;
}
export interface Gap {
  a: number;
  b: number;
}
export interface Scale {
  X: (t: number) => number;
  T: (x: number) => number;
  gaps: Gap[];
  /** Seconds of the range outside the gaps. */
  active: number;
}

/** A stream record's time: its release time, else its `ts`. */
export const tsOf = (r: StreamItem): number => (r._t != null ? r._t : Date.parse(String(r.ts)) / 1000);

/**
 * The whole run, at least a minute: the recording in replay; live, first bar or phase to now, or to the run's last
 * activity once it has gone quiet (`tl.end`). While it runs, the end leaves 1% headroom (at least 2 s), so the now
 * line and running bars stay inside while *now* is a second behind; at most 5 minutes, for a long run.
 */
export function fitRange(clock: Clock, tl: Timeline, now: number): Range {
  const ts = [...tl.bars.map((b) => b.start), ...tl.phases.map((p) => p.start)].filter((t) => t != null);
  const replay = clock.mode === "replay";
  const start = replay && clock.start != null ? clock.start : ts.length ? Math.min(...ts) : now - 3600;
  const live = tl.end ?? now + Math.min(300, Math.max(2, (now - start) * 0.01));
  const end = replay && clock.end != null ? clock.end : live;
  return { t0: start, t1: Math.max(end, start + 60) };
}

export interface PodStart {
  i: number;
  t: number;
  acquired_at: string;
  hostname?: string;
  pid?: number;
}

/** Pod starts (`i` 0) and resumes from `pod_history`, up to *now*. */
export function podStarts(run: RunInfo | undefined, now: number): PodStart[] {
  return (run?.pod_history ?? [])
    .map((p, i) => ({ ...p, i, t: Date.parse(p.acquired_at) / 1000 }))
    .filter((p) => !isNaN(p.t) && p.t <= now);
}

/** Every time something happened, sorted; the idle gaps lie between them. */
export function activityPoints(tl: Timeline, pods: PodStart[], streams: StreamItem[][]): number[] {
  const pts: (number | null | undefined)[] = [
    ...tl.bars.flatMap((b) => [b.start, b.end]),
    ...tl.phases.map((p) => p.start),
    ...(tl.steps ?? []).map((s) => s.start),
    ...pods.map((p) => p.t),
    ...streams.flatMap((s) => s.map(tsOf)),
  ];
  return (pts.filter((t) => t != null && !isNaN(t)) as number[]).sort((a, b) => a - b);
}

/**
 * Idle spans in [t0, t1]: stretches longer than max(GAP_MIN, 3% of the range) with no activity, such as a paused pod.
 * *points* must be sorted; the range ends and *now* count as activity.
 */
export function idleGaps(points: number[], t0: number, t1: number, now: number): Gap[] {
  const inside = points.filter((t) => t >= t0 && t <= t1);
  const pts = [t0, t1, Math.min(now, t1), ...inside].sort((a, b) => a - b);
  const min = Math.max(GAP_MIN, (t1 - t0) * 0.03);
  const out: Gap[] = [];
  for (let i = 1; i < pts.length; i++) {
    const d = pts[i] - pts[i - 1];
    const pad = Math.min(120, d * 0.05);
    if (d > min) out.push({ a: pts[i - 1] + pad, b: pts[i] - pad });
  }
  return out;
}

/** Piecewise time <-> x map: each gap gets GAP_PX, active time shares the rest linearly. Too many gaps: none. */
export function tlScale(t0: number, t1: number, W: number, gaps: Gap[] = []): Scale {
  const px = W - LABEL - 8;
  if (gaps.length * GAP_PX > px / 2) gaps = [];
  const active = Math.max(1, t1 - t0 - gaps.reduce((a, g) => a + g.b - g.a, 0));
  const k = (px - gaps.length * GAP_PX) / active;
  const X = (t: number): number => {
    let x = LABEL;
    let prev = t0;
    for (const g of gaps) {
      if (t <= g.a) break;
      if (t < g.b) return x + (g.a - prev) * k + ((t - g.a) / (g.b - g.a)) * GAP_PX;
      x += (g.a - prev) * k + GAP_PX;
      prev = g.b;
    }
    return x + (t - prev) * k;
  };
  const T = (x: number): number => {
    let at = LABEL;
    let prev = t0;
    for (const g of gaps) {
      const xa = at + (g.a - prev) * k;
      if (x <= xa) break;
      if (x < xa + GAP_PX) return g.a + ((x - xa) / GAP_PX) * (g.b - g.a);
      at = xa + GAP_PX;
      prev = g.b;
    }
    return prev + (x - at) / k;
  };
  return { X, T, gaps, active };
}

const STEPS = [1, 5, 10, 30, 60, 300, 600, 900, 1800, 3600, 7200, 14400, 43200, 86400];

/** Round times in [t0, t1], about *n* of them over *active* seconds. */
export function niceTicks(t0: number, t1: number, n: number, active = t1 - t0): number[] {
  const span = active / n;
  const step = STEPS.find((x) => x >= span) ?? 86400;
  const out: number[] = [];
  for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) out.push(t);
  return out;
}

/** Axis ticks: round times plus the start of each active stretch, none inside a gap, at least 60 px apart. */
export function axisTicks(r: Range, W: number, sc: Scale): number[] {
  const n = Math.max(2, Math.floor((W - LABEL) / 90));
  const all = [...new Set([r.t0, ...sc.gaps.map((g) => g.b), ...niceTicks(r.t0, r.t1, n, sc.active)])];
  let lastX = -1e9;
  return all
    .sort((a, b) => a - b)
    .filter((t) => {
      if (sc.gaps.some((g) => t > g.a && t < g.b) || sc.X(t) - lastX < 60) return false;
      lastX = sc.X(t);
      return true;
    });
}

export interface PodMarker extends PodStart {
  x: number;
  label: string;
  /** Too close to the previous labelled marker to carry a label. */
  crowded: boolean;
}

/** Pod start and resume markers inside the range, with labels thinned where they crowd. */
export function podMarkers(pods: PodStart[], r: Range, X: (t: number) => number): PodMarker[] {
  let last = -1e9;
  const out: PodMarker[] = [];
  for (const p of pods) {
    // acquired_at has whole seconds, so the first start can precede the replay start slightly.
    if (p.t < r.t0 - 5 || p.t > r.t1) continue;
    const x = X(Math.max(p.t, r.t0));
    const crowded = x - last < 50;
    if (!crowded) last = x;
    out.push({ ...p, x, label: p.i ? `resume ${p.i}` : "start", crowded });
  }
  return out;
}

/** Swimlanes: the graph nodes except the bus, by graph row, then id. */
export function lanes(nodes: GraphNode[]): string[] {
  const row = (n: GraphNode) => GROUP_ROW[n.group] ?? 2;
  return nodes
    .filter((n) => n.id !== "bus")
    .sort((a, b) => row(a) - row(b) || a.id.localeCompare(b.id))
    .map((n) => n.id);
}

/** The lane of a bus event's sender: a node matching by id, label or `spec:` id, else the coordinator. */
export function eventLane(nodes: GraphNode[], agent: unknown): string {
  if (!agent) return "coordinator";
  const hit = nodes.find((n) => n.id === agent || n.label === agent || n.id === `spec:${String(agent)}`);
  return hit ? hit.id : "coordinator";
}

/** The lane of an LLM call, as graph.py counts it: its task's node (specialist calls), else its component's. */
export function callLane(tasks: Task[], call: StreamItem): string {
  const task = call.task_id ? tasks.find((t) => t.task_id === call.task_id) : undefined;
  return task?.node ?? agentNode(String(call.component || call.role || "llm"));
}

/** The range zoomed by *k* (above 1 zooms out) about time *at*, which keeps its place. */
export function zoomAbout(r: Range, at: number, k: number): Range {
  const frac = (at - r.t0) / (r.t1 - r.t0);
  const span = Math.max(MIN_SPAN, (r.t1 - r.t0) * k);
  return { t0: at - frac * span, t1: at + (1 - frac) * span };
}
