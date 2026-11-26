// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Graph layouts. Positions are node centres.

import { forceCollide, forceLink, forceManyBody, forceSimulation, forceX, forceY } from "d3-force";
import type { SimulationNodeDatum } from "d3-force";

export const GROUP_ROW: Record<string, number> = { llm: 0, core: 1, exec: 2, spec: 3, agent: 3 };
export const NW = 132;
export const NH = 42;
export const FORCE_H = 440;

export type Point = { x: number; y: number };
export type Positions = Record<string, Point>;
type LayoutNode = { id: string; group: string; layer?: number };
type LayoutEdge = { source: string; target: string };

const rowOf = (n: LayoutNode): number => GROUP_ROW[n.group] ?? n.layer ?? 2;

/** Rows by group, each sorted by id, centred and wrapped to the width `W`. */
export function layered(nodes: LayoutNode[], W: number): Positions {
  const rows = new Map<number, LayoutNode[]>();
  for (const n of nodes) rows.set(rowOf(n), [...(rows.get(rowOf(n)) ?? []), n]);
  const keys = [...rows.keys()].sort((a, b) => a - b);
  const pos: Positions = {};
  const perLine = Math.max(1, Math.floor((W - 20) / (NW + 12)));
  keys.forEach((k, ri) => {
    const row = [...rows.get(k)!].sort((a, b) => a.id.localeCompare(b.id));
    row.forEach((n, i) => {
      const line = Math.floor(i / perLine);
      const inLine = Math.min(perLine, row.length - line * perLine);
      const gap = Math.max(12, Math.min(90, (W - 20 - inLine * NW) / Math.max(1, inLine)));
      const span = inLine * (NW + gap);
      const x = (W - span) / 2 + (i % perLine) * (NW + gap) + NW / 2 + gap / 2;
      pos[n.id] = { x, y: 40 + ri * 100 + line * (NH + 10) };
    });
  });
  return pos;
}

type SimNode = SimulationNodeDatum & { id: string };

/**
 * Force-directed positions. Nodes already in `prev` stay pinned; only new nodes move, seeded from the layered
 * layout, so the graph doesn't jump when a node appears. Deterministic: d3-force seeds its own random source.
 */
export function forceLayout(nodes: LayoutNode[], edges: LayoutEdge[], prev: Positions, W: number): Positions {
  const seed = layered(nodes, W);
  const sim: SimNode[] = nodes.map((n, i) => {
    const p = prev[n.id];
    if (p) return { id: n.id, x: p.x, y: p.y, fx: p.x, fy: p.y };
    return { id: n.id, x: seed[n.id].x + ((i % 5) - 2) * 4, y: seed[n.id].y };
  });
  const out: Positions = {};
  if (sim.every((n) => n.fx !== undefined)) {
    for (const n of sim) out[n.id] = { x: n.x!, y: n.y! };
    return out;
  }
  const ids = new Set(nodes.map((n) => n.id));
  const links = edges.filter((e) => ids.has(e.source) && ids.has(e.target)).map((e) => ({ ...e }));
  forceSimulation(sim)
    .force("charge", forceManyBody().strength(-300))
    .force(
      "link",
      forceLink<SimNode, LayoutEdge>(links)
        .id((d) => d.id)
        .distance(170)
        .strength(0.3),
    )
    .force("collide", forceCollide(NW / 2 + 6))
    .force("x", forceX(W / 2).strength(0.03))
    .force("y", forceY(FORCE_H / 2).strength(0.06))
    .force("box", () => {
      // Clamped every tick rather than once at the end, so collisions resolve inside the box.
      for (const n of sim) {
        if (n.fx !== undefined) continue;
        n.x = Math.min(W - NW / 2, Math.max(NW / 2, n.x!));
        n.y = Math.min(FORCE_H - NH, Math.max(NH, n.y!));
      }
    })
    .stop()
    .tick(300);
  for (const n of sim) out[n.id] = { x: n.x!, y: n.y! };
  return out;
}

const ease = (t: number): number => 1 - Math.pow(1 - t, 3);

/** Positions part-way (`t` in 0..1, eased) from `a` to `b`; nodes missing from `a` sit at their target. */
export function lerpPositions(a: Positions, b: Positions, t: number): Positions {
  const k = ease(Math.min(1, Math.max(0, t)));
  const out: Positions = {};
  for (const [id, to] of Object.entries(b)) {
    const from = a[id] ?? to;
    out[id] = { x: from.x + (to.x - from.x) * k, y: from.y + (to.y - from.y) * k };
  }
  return out;
}

/** Where the line from a node's centre `p` toward `toward` leaves its box. */
export function clip(p: Point, toward: Point, w = NW, h = NH): Point {
  const dx = toward.x - p.x;
  const dy = toward.y - p.y;
  const sc = Math.min(Math.abs((w / 2 + 2) / (dx || 1e-9)), Math.abs((h / 2 + 2) / (dy || 1e-9)));
  return { x: p.x + dx * Math.min(1, sc), y: p.y + dy * Math.min(1, sc) };
}

export interface Curve {
  start: Point;
  control: Point;
  end: Point;
}

/** The quadratic curve between two node centres, clipped to both boxes; bent further for two-way pairs. */
export function curve(a: Point, b: Point, back: boolean): Curve {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len = Math.hypot(dx, dy) || 1;
  const bend = back ? 28 : 12;
  const control = { x: (a.x + b.x) / 2 - (dy / len) * bend, y: (a.y + b.y) / 2 + (dx / len) * bend };
  return { start: clip(a, control), control, end: clip(b, control) };
}

export const curvePath = ({ start: s, control: c, end: e }: Curve): string =>
  `M${s.x},${s.y} Q${c.x},${c.y} ${e.x},${e.y}`;

type Box = { x0: number; x1: number; y0: number; y1: number };
const box = (x: number, y: number, w: number, h: number): Box => ({
  x0: x - w / 2,
  x1: x + w / 2,
  y0: y - h / 2,
  y1: y + h / 2,
});
const overlaps = (r: Box, o: Box): boolean => r.x0 < o.x1 && o.x0 < r.x1 && r.y0 < o.y1 && o.y0 < r.y1;

export type LabelSpot = { x: number; y: number; text: string };

/**
 * Labels for curves, placed in order: the first text (longest first) at the first spot along its curve that
 * overlaps no node and no earlier label; null when nothing fits (the edge's tooltip still has the full label).
 */
export function placeLabels(nodes: Point[], items: { curve: Curve; texts: string[] }[]): (LabelSpot | null)[] {
  const taken = nodes.map((p) => box(p.x, p.y, NW + 4, NH + 4));
  return items.map(({ curve: { start: a, control: c, end: b }, texts }) => {
    for (const text of texts) {
      for (const t of [0.5, 0.38, 0.62, 0.28, 0.72]) {
        const u = 1 - t;
        const x = u * u * a.x + 2 * u * t * c.x + t * t * b.x;
        const y = u * u * a.y + 2 * u * t * c.y + t * t * b.y;
        const r = box(x, y, text.length * 5.6 + 6, 13);
        if (taken.some((o) => overlaps(r, o))) continue;
        taken.push(r);
        return { x, y, text };
      }
    }
    return null;
  });
}
