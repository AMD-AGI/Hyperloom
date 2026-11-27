// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { Artifact, StreamItem, View } from "../types";
import {
  accuracyResults,
  fmtPct,
  METRIC_READERS,
  metricSeries,
  resultSources,
  stairPath,
  upliftSeries,
  type MetricResult,
  type UpliftStep,
} from "./uplift";

const unit = (label: string, ts: number, output_throughput: number, earlier?: number[]) => ({
  unit: "variant",
  label,
  ts,
  output_throughput,
  earlier,
});

const makeView = (opts: { bars?: View["timeline"]["bars"]; summary?: Record<string, unknown> } = {}) =>
  ({
    summary: { cumulative_gain_validated: 18, ...opts.summary },
    timeline: { bars: opts.bars ?? [], phases: [], steps: [] },
    tasks: [
      {
        task_id: "e1",
        progress: {
          units: [
            { unit: "baseline", ts: 12, output_throughput: 999 },
            unit("a", 20, 110, [105]),
            unit("b", 30, 105),
            unit("c", 40, 120),
          ],
        },
      },
    ],
  }) as unknown as View;

const baselineEntry: StreamItem = { _t: 10, kind: "baseline", throughput: 100 };
const verdict = (_t: number, outcome: string, gain_pct?: number): StreamItem => ({
  _t,
  task_id: "e1",
  kind: "param",
  outcome,
  gain_pct,
});
const twoDecided = [baselineEntry, verdict(50, "KEEP", 10), verdict(50, "REVERT", -4)];
const allDecided = [...twoDecided, verdict(55, "keep", 9)];

const round = (steps: UpliftStep[]) => steps.map((s) => ({ t: s.t, pct: Math.round(s.pct), label: s.label }));

describe("upliftSeries", () => {
  it("places each variant at its measured time and judges it by the journal", () => {
    const up = upliftSeries(makeView(), twoDecided, 60);
    expect(up.baseline).toBe(100);
    expect(up.baseT).toBe(10);
    expect(up.points.map((p) => [p.t, p.label, Math.round(p.pct), p.verdict, p.stepGainPct, p.decidedAt])).toEqual([
      [20, "a", 10, "keep", 10, 50],
      [30, "b", 5, "revert", -4, 50],
      [40, "c", 20, "pending", null, null],
    ]);
    expect(up.points[0].earlierPct.map(Math.round)).toEqual([5]);
  });

  it("matches verdicts to variants in their order within the task", () => {
    const up = upliftSeries(makeView(), [baselineEntry, verdict(50, "REVERT")], 60);
    expect(up.points.map((p) => [p.label, p.verdict])).toEqual([
      ["a", "revert"],
      ["b", "pending"],
      ["c", "pending"],
    ]);
  });

  it("matches a verdict to its variant by name, whatever the verdict's kind", () => {
    const named = (kind: string, variant_name: string, outcome: string, gain_pct?: number): StreamItem => ({
      _t: 50,
      task_id: "e1",
      kind,
      variant_name,
      outcome,
      gain_pct,
    });
    const journal = [
      baselineEntry,
      named("backend", "c", "KEEP", 9),
      named("env", "a", "REVERT"),
      named("param", "b", "KEEP", 5),
    ];
    const up = upliftSeries(makeView(), journal, 60);
    expect(up.points.map((p) => [p.label, p.verdict, p.stepGainPct])).toEqual([
      ["a", "revert", null],
      ["b", "keep", 5],
      ["c", "keep", 9],
    ]);
    expect(up.validatedPct).toBe(18);
  });

  it("matches a verdict without a name in order, whatever its kind", () => {
    const kept: StreamItem = { _t: 50, task_id: "e1", kind: "other", outcome: "KEEP", gain_pct: 10 };
    const up = upliftSeries(makeView(), [baselineEntry, kept], 60);
    expect(up.points.map((p) => [p.label, p.verdict])).toEqual([
      ["a", "keep"],
      ["b", "pending"],
      ["c", "pending"],
    ]);
  });

  it("leaves out a task the journal records without a verdict, such as a concurrency sweep", () => {
    const recorded: StreamItem = { _t: 50, task_id: "e1", kind: "other", outcome: "recorded" };
    const up = upliftSeries(makeView(), [baselineEntry, recorded], 60);
    expect(up.points).toEqual([]);
    expect(round(up.provisional).map((s) => s.label)).toEqual(["baseline"]);
  });

  it("steps up only on kept improvements; provisional steps also count pending ones", () => {
    const up = upliftSeries(makeView(), twoDecided, 60);
    expect(round(up.steps)).toEqual([
      { t: 10, pct: 0, label: "baseline" },
      { t: 20, pct: 10, label: "a" },
    ]);
    expect(round(up.provisional)).toEqual([
      { t: 10, pct: 0, label: "baseline" },
      { t: 20, pct: 10, label: "a" },
      { t: 40, pct: 20, label: "c" },
    ]);
    expect(Math.round(up.bestPct)).toBe(10);
    expect(up.validatedPct).toBeNull();
  });

  it("steps up to the best kept variant and reports the validated gain once every variant is decided", () => {
    const up = upliftSeries(makeView(), allDecided, 60);
    expect(round(up.steps).map((s) => s.label)).toEqual(["baseline", "a", "c"]);
    expect(Math.round(up.bestPct)).toBe(20);
    expect(up.validatedPct).toBe(18);
  });

  it("shows only what was measured and decided by now", () => {
    const beforeVerdicts = upliftSeries(makeView(), allDecided, 45);
    expect(beforeVerdicts.points.every((p) => p.verdict === "pending")).toBe(true);
    expect(round(beforeVerdicts.steps).map((s) => s.label)).toEqual(["baseline"]);
    expect(upliftSeries(makeView(), allDecided, 25).points.map((p) => p.label)).toEqual(["a"]);
  });

  it("is empty until the baseline is measured", () => {
    const up = upliftSeries(makeView(), allDecided, 5);
    expect(up.baseline).toBeNull();
    expect(up.points).toEqual([]);
    expect(up.steps).toEqual([]);
    expect(up.bestPct).toBe(0);
  });

  it("falls back to the finished baseline bar and the summary's throughput", () => {
    const bar = { task_id: "b0", node: "exec:baseline", label: "baseline", start: 2, end: 12, state: "done" };
    const view = makeView({ bars: [bar as View["timeline"]["bars"][number]], summary: { baseline_tput: 100 } });
    expect(upliftSeries(view, [], 11).baseline).toBeNull();
    const up = upliftSeries(view, [], 60);
    expect([up.baseline, up.baseT]).toEqual([100, 12]);
  });
});

describe("resultSources", () => {
  it("finds each round's result file and the variant it belongs to", () => {
    const art = (path: string) => ({ path }) as Artifact;
    const artifacts = [
      "runs/baseline/b0/round_1/inferencex_result.json",
      "runs/explore/e1/v1_a/warmup_round_1/inferencex_result.json",
      "runs/explore/e1/v2_b/inferencex_result.json",
      "runs/explore/e1/notes/inferencex_result.json",
      "other/inferencex_result.json",
    ].map(art);
    expect(resultSources(artifacts, "inferencex_result.json")).toEqual([
      { path: artifacts[0].path, task_id: "b0", label: "baseline", warmup: false },
      { path: artifacts[1].path, task_id: "e1", label: "a", warmup: true },
      { path: artifacts[2].path, task_id: "e1", label: "b", warmup: false },
    ]);
  });
});

describe("metricSeries", () => {
  const up = upliftSeries(makeView(), allDecided, 60);
  const result = (task_id: string, label: string, value: number, t: number, warmup = false, se?: number) =>
    ({ path: `${task_id}/${label}/${t}`, task_id, label, value, t, warmup, se }) as MetricResult;
  const results = [
    result("b0", "baseline", 50, 9, false, 1),
    result("e1", "a", 48, 18, true),
    result("e1", "a", 45, 19),
    result("e1", "b", 60, 29),
    result("e1", "c", 40, 39),
    result("e1", "c", 1, 70),
    result("e2", "a", 99, 19),
  ];

  it("takes each variant's measured round, keeps the warm-up rounds as noise, and follows the kept steps", () => {
    const s = metricSeries(up, results, 60);
    expect([s.baseline, s.baselineSe]).toEqual([50, 1]);
    expect(s.points.map((p) => [p.t, p.label, p.value, p.earlier, p.verdict])).toEqual([
      [10, "baseline", 50, [], "keep"],
      [20, "a", 45, [48], "keep"],
      [30, "b", 60, [], "revert"],
      [40, "c", 40, [], "keep"],
    ]);
    expect(s.steps.map((st) => [st.t, st.pct, st.label])).toEqual([
      [10, 50, "baseline"],
      [20, 45, "a"],
      [40, 40, "c"],
    ]);
    expect(s.current).toBe(40);
  });

  it("leaves out a kept step whose metric was never read", () => {
    const s = metricSeries(
      up,
      results.filter((r) => r.label !== "c"),
      60,
    );
    expect(s.steps.map((st) => st.label)).toEqual(["baseline", "a"]);
    expect(s.current).toBe(45);
  });
});

describe("METRIC_READERS", () => {
  const ttft = METRIC_READERS["inferencex_result.json"];

  it("reads mean TTFT and the benchmark's end time", () => {
    expect(ttft({ mean_ttft_ms: 42.5, benchmark_end_time_unix: 1000 })).toEqual({ value: 42.5, t: 1000, se: null });
    expect(ttft({ mean_ttft_ms: 42.5 }).t).toBeNull();
  });
});

describe("accuracyResults", () => {
  const up = upliftSeries(makeView(), allDecided, 60);
  const acc = {
    baseline: 0.968,
    variants: [
      { variant: "a", value: 0.962, t: 21 },
      { variant: "c", value: 0.97, t: 41 },
    ],
  };
  const rows = (now: number) =>
    metricSeries(up, accuracyResults(acc, up.baseT), now).points.map((p) => [p.t, p.label, p.value.toFixed(1)]);

  it("places state.json's baseline and kept-variant accuracy on the uplift points, in percent", () => {
    expect(rows(60)).toEqual([
      [10, "baseline", "96.8"],
      [20, "a", "96.2"],
      [40, "c", "97.0"],
    ]);
    const s = metricSeries(up, accuracyResults(acc, up.baseT), 60);
    expect(s.steps.map((st) => st.label)).toEqual(["baseline", "a", "c"]);
    expect([s.baseline?.toFixed(1), s.current?.toFixed(1)]).toEqual(["96.8", "97.0"]);
  });

  it("shows a variant's accuracy once the clock passes the time it was measured", () => {
    expect(rows(30).map((r) => r[1])).toEqual(["baseline", "a"]);
  });

  it("has nothing to place when state.json records no accuracy", () => {
    expect(accuracyResults(undefined, 10)).toEqual([]);
    expect(accuracyResults({ baseline: null, variants: [] }, 10)).toEqual([]);
  });
});

describe("fmtPct", () => {
  it("signs the percentage and has a placeholder for missing values", () => {
    expect([fmtPct(17.5), fmtPct(-3), fmtPct(0), fmtPct(15, 0), fmtPct(null), fmtPct(undefined)]).toEqual([
      "+17.5%",
      "-3.0%",
      "+0.0%",
      "+15%",
      "—",
      "—",
    ]);
  });
});

describe("stairPath", () => {
  const steps = [
    { t: 0, pct: 0, tput: 100, label: "baseline" },
    { t: 10, pct: 5, tput: 105, label: "a" },
  ];
  const X = (t: number) => t;
  const Y = (p: number) => 100 - p;

  it("draws a step line held flat until the end time", () => {
    expect(stairPath(steps, X, Y, 20)).toBe("M0,100H10V95H20");
  });
  it("never draws back before the last step", () => {
    expect(stairPath(steps, X, Y, 5)).toBe("M0,100H10V95H10");
  });
  it("draws nothing without steps", () => {
    expect(stairPath([], X, Y, 20)).toBe("");
  });
});
