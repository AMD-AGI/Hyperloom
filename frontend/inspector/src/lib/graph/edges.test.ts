// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { agentNode, pairEdges, pulses, recentRates, strokeWidth } from "./edges";

describe("agentNode", () => {
  it("maps bus agent names like graph.py", () => {
    expect(agentNode("*")).toBe("bus");
    expect(agentNode("")).toBe("bus");
    expect(agentNode("critic")).toBe("critic");
    expect(agentNode("framework_agent")).toBe("agent:framework");
    expect(agentNode("kernel-agent")).toBe("agent:kernel");
    expect(agentNode("spec:static_recon")).toBe("spec:static_recon");
  });
});

describe("pairEdges", () => {
  it("joins topics per pair and marks two-way pairs", () => {
    const out = pairEdges([
      { source: "orchestration", target: "critic", label: "proposal", count: 3 },
      { source: "orchestration", target: "critic", label: "note", count: 1 },
      { source: "critic", target: "orchestration", label: "review_verdict", count: 3 },
      { source: "coordinator", target: "bus", label: "decision", count: 2 },
    ]);
    expect(out).toHaveLength(3);
    expect(out[0]).toEqual({
      id: "orchestration>critic",
      source: "orchestration",
      target: "critic",
      label: "proposal ×3, note ×1",
      total: 4,
      back: true,
    });
    expect(out[2].back).toBe(false);
  });
});

describe("recentRates", () => {
  const ev = (t: number, from: string, to: string) => ({ _t: t, from_agent: from, to_agent: to });
  it("counts the last minute per pair, ignoring the future", () => {
    const events = [ev(10, "a", "b"), ev(100, "a", "b"), ev(130, "a", "*"), ev(150, "a", "b"), ev(200, "a", "b")];
    const r = recentRates(events, 160);
    expect(r.get("a>b")).toBe(2);
    expect(r.get("a>bus")).toBe(1);
    expect(recentRates(events, 1000).size).toBe(0);
  });
});

describe("pulses and stroke", () => {
  it("scale with traffic, within bounds", () => {
    expect(pulses(0)).toEqual({ count: 0, dur: 0 });
    expect(pulses(1).count).toBe(1);
    expect(pulses(1000).count).toBe(3);
    expect(pulses(1000).dur).toBe(0.8);
    expect(pulses(1).dur).toBeGreaterThan(pulses(20).dur);
    expect(strokeWidth(1)).toBe(1);
    expect(strokeWidth(1e6)).toBe(5);
  });
});
