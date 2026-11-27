// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { latencyLabel, latencyOf, tickDigits } from "./latency";
import type { StreamItem } from "./types";

const run = (latency: unknown): StreamItem => ({ _t: 0, latency });

describe("latency", () => {
  const full = run({
    ttft: { mean: 76.2, median: 64.4, p99: 217.4 },
    e2el: { mean: 3500, median: null, p99: 3614 },
  });

  it("reads the picked kind and statistic of a run", () => {
    expect(latencyOf(full, { kind: "ttft", stat: "p99" })).toBe(217.4);
    expect(latencyOf(full, { kind: "e2el", stat: "mean" })).toBe(3500);
  });

  it("gives null for a statistic, a kind or a whole table the run lacks", () => {
    expect(latencyOf(full, { kind: "e2el", stat: "median" })).toBeNull();
    expect(latencyOf(full, { kind: "tpot", stat: "mean" })).toBeNull();
    expect(latencyOf(run(undefined), { kind: "ttft", stat: "mean" })).toBeNull();
    expect(latencyOf({ _t: 0 }, { kind: "ttft", stat: "mean" })).toBeNull();
  });

  it("names the pick for axis titles and tooltips", () => {
    expect(latencyLabel({ kind: "ttft", stat: "p99" })).toBe("p99 TTFT");
    expect(latencyLabel({ kind: "e2el", stat: "mean" })).toBe("mean E2E latency");
  });

  it("keeps decimals on ticks only when the range is narrow", () => {
    expect([tickDigits(500), tickDigits(20), tickDigits(8), tickDigits(0.6)]).toEqual([0, 0, 1, 2]);
  });
});
