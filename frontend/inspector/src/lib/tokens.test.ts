// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import {
  bucketCount,
  cacheHit,
  cumulative,
  decode,
  OTHER,
  stack,
  unpricedReasons,
  unpricedWhy,
  valueTicks,
} from "./tokens";
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
      decode(series, "kind", null, "tokens", { terms }).series.map((s) => s.label);
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
    const d = decode(series, "node", null, "tokens", { maxNodes: 1 });
    expect(d.series.map((s) => [s.key, s.label, s.total])).toEqual([[OTHER, "2 others", 10682]]);
  });

  it("groups agents alike for tokens and USD: by their tokens", () => {
    // a uses the most tokens and costs the least, c the reverse.
    const three: TokenSeries = {
      ...series,
      nodes: ["a", "b", "c"],
      rows: [
        [0, 0, 1000, 0, 0, 0, 0, 1, 0.001, 0, 0, 0, 0],
        [0, 1, 100, 0, 0, 0, 0, 1, 0.01, 0, 0, 0, 0],
        [0, 2, 10, 0, 0, 0, 0, 1, 0.1, 0, 0, 0, 0],
      ],
    };
    const tokens = decode(three, "node", null, "tokens", { maxNodes: 2 }).series;
    const usd = decode(three, "node", null, "usd", { maxNodes: 2 }).series;
    expect(tokens.map((x) => [x.key, x.label])).toEqual([
      ["a", "a"],
      [OTHER, "2 others"],
    ]);
    expect(usd.map((x) => [x.key, x.label])).toEqual([
      ["a", "a"],
      [OTHER, "2 others"],
    ]);
    expect(usd[1].total).toBeCloseTo(0.11, 9);
    const all = decode(three, "node", null, "usd").series.map((x) => x.key);
    expect(all).toEqual(["a", "b", "c"]);
  });

  it("decodes USD from the cost columns, by token kind", () => {
    const priced: TokenSeries = {
      ...series,
      rows: [
        [0, 0, 100, 0, 0, 20, 0, 1, 0.0005, 0, 0, 0.0005, 0],
        [1, 1, 2, 500, 9000, 0, 0, 3, 0.00001, 0.003125, 0.0045, 0, 0],
        [2, 0, 50, 0, 1000, 10, 0, 1, 0.00025, 0, 0.0005, 0.00025, 0],
      ],
    };
    const d = decode(priced, "kind", null, "usd");
    expect(d.series.map((s) => s.key)).toEqual(["cache_read", "cache_write", "input", "output"]);
    expect(d.series.find((s) => s.key === "input")!.values).toEqual([0.0005, 0.00001, 0.00025]);
    expect(d.calls).toEqual([1, 3, 1]);
    const byNode = decode(priced, "node", null, "usd").series.map((s) => [s.key, s.total]);
    expect(byNode[0][0]).toBe("spec:serving");
    expect(byNode[0][1]).toBeCloseTo(0.007635, 9);
    expect(byNode[1][1]).toBeCloseTo(0.002, 9);
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

  it("says why calls are unpriced, a reason each", () => {
    expect(unpricedReasons({ no_model: 306, no_price: 1 })).toEqual([
      ["no_model", "306 calls with no model recorded"],
      ["no_price", "1 call with no price for the model"],
    ]);
    expect(unpricedReasons({ no_price: 12, no_search_price: 1 })).toEqual([
      ["no_price", "12 calls with no price for the model"],
      ["no_search_price", "1 call with web searches and no price per search"],
    ]);
    expect(unpricedReasons({ no_model: 0 })).toEqual([]);
  });

  it("explains under the price table why calls are unpriced: a line per reason, with the price each model lacks", () => {
    const t = { input: 0, cache_write: 0, cache_read: 0, output: 0, reasoning: 0, total: 0, calls: 0, errors: 0 };
    const opus = { ...t, calls: 179, web_searches: 352, unpriced: 176, no_search_price: 176 };
    const pricing = {
      source: "hyperloom_common/llm/llm_prices.json",
      models: {
        "claude-opus-5": {
          id: "Claude-Opus-5",
          input: 5,
          cache_write: 6.25,
          cache_read: 0.5,
          output: 25,
          cache_basis: "listed",
          spent_usd: { input: 0.001, cache_write: 0, cache_read: 0, output: 0.002, reasoning: 0 },
        },
      },
      unpriced: {},
    };
    expect(unpricedWhy({ totals: opus, by_model: { "claude-opus-5": opus }, pricing })).toEqual({
      head: "176 of 179 LLM calls, left out of the estimated cost",
      lines: [
        "176 LLM calls with web searches and no price per search: claude-opus-5 (176 calls), as the price sheet " +
          "gives no price per search",
      ],
    });
    const unknown = { calls: 1, total: 30 };
    const mixed = { ...t, calls: 5, unpriced: 2, no_model: 1, no_price: 1 };
    const why = unpricedWhy({
      totals: mixed,
      by_model: {},
      pricing: { ...pricing, models: {}, unpriced: { unknown, "Qwen3.8-27B": { calls: 1, total: 70 } } },
    });
    expect(why.lines).toEqual([
      "1 LLM call with no model recorded, as neither the call log nor the specialist's log names its model",
      "1 LLM call with no price for the model: Qwen3.8-27B (1 call), as the price sheet lists no price for it",
    ]);
    expect(unpricedWhy({ totals: t, by_model: {}, pricing })).toEqual({ head: "", lines: [] });
  });
});
