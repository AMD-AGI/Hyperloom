// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { clip, curve, FORCE_H, forceLayout, layered, lerpPositions, NH, NW, placeLabels } from "./layout";

const nodes = [
  { id: "orchestration", group: "llm" },
  { id: "critic", group: "llm" },
  { id: "coordinator", group: "core" },
  { id: "bus", group: "core" },
  { id: "exec:baseline", group: "exec" },
  { id: "spec:static_recon", group: "spec" },
];
const edges = [
  { source: "orchestration", target: "critic" },
  { source: "coordinator", target: "exec:baseline" },
  { source: "coordinator", target: "spec:static_recon" },
];

describe("layered", () => {
  it("puts each group on its own row, sorted by id", () => {
    const p = layered(nodes, 1000);
    expect(p.critic.y).toBe(40);
    expect(p.critic.y).toBe(p.orchestration.y);
    expect(p.critic.x).toBeLessThan(p.orchestration.x);
    expect(p.bus.y).toBe(140);
    expect(p["exec:baseline"].y).toBe(240);
    expect(p["spec:static_recon"].y).toBe(340);
  });

  it("wraps a row that doesn't fit the width", () => {
    const many = Array.from({ length: 5 }, (_, i) => ({ id: `exec:${i}`, group: "exec" }));
    const p = layered(many, 450);
    expect(new Set(Object.values(p).map((q) => q.y)).size).toBe(3);
    expect(p["exec:2"].y).toBe(40 + NH + 10);
    expect(p["exec:4"].y).toBe(40 + 2 * (NH + 10));
  });
});

describe("forceLayout", () => {
  it("is deterministic and stays inside the box", () => {
    const a = forceLayout(nodes, edges, {}, 900);
    expect(forceLayout(nodes, edges, {}, 900)).toEqual(a);
    for (const q of Object.values(a)) {
      expect(q.x).toBeGreaterThanOrEqual(NW / 2);
      expect(q.x).toBeLessThanOrEqual(900 - NW / 2);
      expect(q.y).toBeGreaterThanOrEqual(NH);
      expect(q.y).toBeLessThanOrEqual(FORCE_H - NH);
    }
  });

  it("keeps nodes from overlapping", () => {
    const crowd = Array.from({ length: 14 }, (_, i) => ({ id: `n${i}`, group: i < 3 ? "llm" : "exec" }));
    const p = Object.values(forceLayout(crowd, [], {}, 1100));
    for (let i = 0; i < p.length; i++)
      for (let j = i + 1; j < p.length; j++) {
        const apart = Math.abs(p[i].x - p[j].x) >= NW || Math.abs(p[i].y - p[j].y) >= NH;
        expect(apart).toBe(true);
      }
  });

  it("keeps placed nodes pinned and only places new ones", () => {
    const first = forceLayout(nodes.slice(0, 5), edges, {}, 900);
    const next = forceLayout(nodes, edges, first, 900);
    for (const [id, q] of Object.entries(first)) expect(next[id]).toEqual(q);
    expect(next["spec:static_recon"]).toBeDefined();
  });
});

describe("lerpPositions", () => {
  it("runs from a to b, placing new nodes at their target", () => {
    const a = { n: { x: 0, y: 0 } };
    const b = { n: { x: 100, y: 50 }, m: { x: 7, y: 8 } };
    expect(lerpPositions(a, b, 0).n).toEqual({ x: 0, y: 0 });
    expect(lerpPositions(a, b, 1)).toEqual(b);
    expect(lerpPositions(a, b, 0.5).n.x).toBeGreaterThan(50);
    expect(lerpPositions(a, b, 0.3).m).toEqual({ x: 7, y: 8 });
  });
});

describe("clip", () => {
  it("stops on the node's border", () => {
    expect(clip({ x: 0, y: 0 }, { x: 500, y: 0 })).toEqual({ x: NW / 2 + 2, y: 0 });
    expect(clip({ x: 0, y: 0 }, { x: 0, y: -500 })).toEqual({ x: 0, y: -(NH / 2 + 2) });
  });
});

describe("curve and placeLabels", () => {
  it("bends two-way pairs further, to the left of travel", () => {
    const one = curve({ x: 0, y: 0 }, { x: 400, y: 0 }, false);
    const two = curve({ x: 0, y: 0 }, { x: 400, y: 0 }, true);
    expect(one.control).toEqual({ x: 200, y: 12 });
    expect(two.control.y).toBe(28);
    expect(one.start.x).toBeCloseTo(NW / 2 + 2, 0);
  });

  it("falls back along the curve, then to shorter text, then to nothing", () => {
    const c = curve({ x: 0, y: 0 }, { x: 600, y: 0 }, false);
    const [first, second, third] = placeLabels(
      [
        { x: 0, y: 0 },
        { x: 600, y: 0 },
      ],
      [
        { curve: c, texts: ["a long label here", "×3"] },
        { curve: c, texts: ["a long label here", "×3"] },
        { curve: c, texts: ["x".repeat(200)] },
      ],
    );
    expect(first).toMatchObject({ x: 300, text: "a long label here" });
    expect(second!.x).not.toBe(300);
    expect(third).toBeNull();
  });
});
