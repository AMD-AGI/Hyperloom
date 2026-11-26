// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { bucketCount, cacheHit, cumulative, decode, OTHER, stack, valueTicks } from "./tokens";
import type { TokenSeries } from "./types";

const KINDS = ["input", "cache_write", "cache_read", "output", "reasoning"];
// Three 60 s buckets: orchestration in 0 and 2, a specialist in 1.
const series: TokenSeries = {
  start: 1000,
  end: 1170,
  step: 60,
  kinds: KINDS,
  nodes: ["orchestration", "spec:serving"],
  rows: [
    [0, 0, 100, 0, 0, 20, 0, 1],
    [1, 1, 2, 500, 9000, 0, 0, 3],
    [2, 0, 50, 0, 1000, 10, 0, 1],
  ],
};

describe("token series", () => {
  it("counts buckets over the span and past the last row", () => {
    expect(bucketCount(series)).toBe(3);
    expect(bucketCount({ ...series, end: 1000 })).toBe(3);
    expect(bucketCount({ ...series, rows: [] })).toBe(3);
    expect(bucketCount({ ...series, end: 1000, rows: [] })).toBe(1);
  });

  it("decodes by kind, cache reads at the base and no empty reasoning", () => {
    const d = decode(series, "kind", null);
    expect(d.series.map((s) => s.key)).toEqual(["cache_read", "cache_write", "input", "output"]);
    expect(d.series.find((s) => s.key === "input")!.values).toEqual([100, 2, 50]);
    expect(d.series.find((s) => s.key === "cache_read")!.total).toBe(10000);
    expect(d.calls).toEqual([1, 3, 1]);
  });

  it("labels kinds with the server's terms, else with their keys", () => {
    const term = { group: "Token types", label: "Input read from cache", definition: "..." };
    const labels = (terms?: Record<string, typeof term>) =>
      decode(series, "kind", null, { terms }).series.map((s) => s.label);
    expect(labels({ cache_read: term })).toEqual(["Input read from cache", "cache_write", "input", "output"]);
    expect(labels()).toEqual(["cache_read", "cache_write", "input", "output"]);
  });

  it("decodes by agent, the largest first, and filters to a node", () => {
    const d = decode(series, "node", null);
    expect(d.series.map((s) => [s.key, s.total])).toEqual([
      ["spec:serving", 9502],
      ["orchestration", 1180],
    ]);
    expect(d.series[1].values).toEqual([120, 0, 1060]);
    const f = decode(series, "kind", "orchestration");
    expect(f.series.find((s) => s.key === "cache_read")!.values).toEqual([0, 0, 1000]);
    expect(f.calls).toEqual([1, 0, 1]);
  });

  it("folds agents past the limit into one series", () => {
    const d = decode(series, "node", null, { maxNodes: 1 });
    expect(d.series.map((s) => [s.key, s.label, s.total])).toEqual([[OTHER, "2 others", 10682]]);
  });

  it("keeps an agent's colour when the stack reorders", () => {
    const before = decode(series, "node", null).series.find((s) => s.key === "orchestration")!.color;
    const grown = { ...series, rows: [...series.rows, [2, 0, 99999, 0, 0, 0, 0, 1]] };
    const after = decode(grown, "node", null).series;
    expect(after[0].key).toBe("orchestration");
    expect(after[0].color).toBe(before);
  });

  it("stacks running totals from zero and rates per minute", () => {
    const d = decode(series, "kind", null);
    const cum = stack(d.series, new Set(), "cumulative", 60);
    expect(cum.layers[0].lo).toEqual([0, 0, 0, 0]);
    expect(cum.layers[0].hi).toEqual([0, 0, 9000, 10000]);
    expect(cum.max).toBe(10682);
    const rate = stack(d.series, new Set(["cache_read"]), "rate", 120);
    expect(rate.layers.map((l) => l.key)).toEqual(["cache_write", "input", "output"]);
    expect(rate.layers[1].lo).toEqual([0, 250, 0]);
    expect(rate.layers[1].hi).toEqual([50, 251, 25]);
  });

  it("rounds the value axis", () => {
    expect(valueTicks(0)).toEqual([0, 1]);
    expect(valueTicks(6888915)).toEqual([0, 2e6, 4e6, 6e6, 8e6]);
    expect(valueTicks(100)).toEqual([0, 25, 50, 75, 100]);
    expect(cumulative([1, 2, 3])).toEqual([0, 1, 3, 6]);
  });

  it("measures cache hits against all input", () => {
    const t = { input: 10, cache_write: 30, cache_read: 60, output: 5, reasoning: 0, total: 105, calls: 1, errors: 0 };
    expect(cacheHit(t)).toBeCloseTo(0.6);
    expect(cacheHit({ ...t, input: 0, cache_write: 0, cache_read: 0 })).toBeNull();
  });
});
