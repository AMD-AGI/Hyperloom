// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Throughput uplift against the baseline, and the per-variant metrics placed and judged the same way. UpliftStrip
// draws the uplift; MetricStrip draws the metrics.
//
// Each explored variant is placed at the time it was measured (task progress units), not at the journal entry's
// time: the journal writes every verdict of an explore task when the task ends, so the whole uplift would land in
// one instant. The verdict (KEEP / REVERT) comes from the journal; until it is released a point is "pending".

import type { AccuracyView, Artifact, StreamItem, View } from "../types";

/** Height of the uplift strip. */
export const STRIP_H = 104;

export type Verdict = "keep" | "revert" | "pending";

export interface UpliftPoint {
  t: number;
  label: string;
  task_id: string;
  tput: number;
  pct: number;
  /** Warm-up rounds of the same variant: the spread between them and `pct` is the run-to-run noise. */
  earlierPct: number[];
  verdict: Verdict;
  /** The journal's gain for this step, relative to the stack it was tried on. */
  stepGainPct: number | null;
  decidedAt: number | null;
}

export interface UpliftStep {
  t: number;
  pct: number;
  tput: number;
  label: string;
}

export interface Uplift {
  baseline: number | null;
  baseT: number | null;
  validatedPct: number | null;
  points: UpliftPoint[];
  /** Best kept configuration so far, from the baseline (0%) on. */
  steps: UpliftStep[];
  /** Like `steps`, but pending points count as kept: what the line becomes if they are accepted. */
  provisional: UpliftStep[];
  bestPct: number;
}

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

/** The journal's verdict outcomes; its other outcomes ("recorded", "skipped") judge no variant. */
const VERDICTS = new Set(["KEEP", "REVERT"]);

export function upliftSeries(view: View, journal: StreamItem[], now: number): Uplift {
  const released = journal.filter((j) => j._t <= now);
  const baseEntry = released.find((j) => j.kind === "baseline" && num(j.throughput) != null);
  const baseBar = view.timeline.bars.find((b) => b.label === "baseline" && b.end != null && b.end <= now);
  const baseline = num(baseEntry?.throughput) ?? (baseBar ? num(view.summary.baseline_tput) : null);
  const baseT = baseEntry?._t ?? baseBar?.end ?? null;
  const empty: Uplift = {
    baseline,
    baseT,
    validatedPct: null,
    points: [],
    steps: [],
    provisional: [],
    bestPct: 0,
  };
  if (baseline == null || baseT == null) return empty;
  const pct = (v: number) => (v / baseline - 1) * 100;

  const points: UpliftPoint[] = [];
  for (const task of view.tasks) {
    const units = ((task.progress?.units as Record<string, unknown>[] | undefined) ?? []).filter(
      (u) => u.unit === "variant" && num(u.output_throughput) != null && num(u.ts) != null && num(u.ts)! <= now,
    );
    if (!units.length) continue;
    const entries = released.filter((j) => j.task_id === task.task_id);
    const verdicts = entries.filter((j) => VERDICTS.has(String(j.outcome).toUpperCase()));
    // A task the journal records without a verdict (a concurrency sweep) measured no variant to keep or revert.
    if (entries.length && !verdicts.length) continue;
    // A verdict names its variant, of whatever kind (env, param, backend); one without a name is matched in order.
    const unnamed = verdicts.filter((j) => !j.variant_name);
    let next = 0;
    units.forEach((u) => {
      const v = verdicts.find((j) => j.variant_name && j.variant_name === u.label) ?? unnamed[next++];
      const tput = num(u.output_throughput)!;
      points.push({
        t: num(u.ts)!,
        label: String(u.label ?? "variant"),
        task_id: task.task_id,
        tput,
        pct: pct(tput),
        earlierPct: ((u.earlier as number[] | undefined) ?? []).filter((x) => num(x) != null).map(pct),
        verdict: !v ? "pending" : String(v.outcome).toUpperCase() === "KEEP" ? "keep" : "revert",
        stepGainPct: num(v?.gain_pct),
        decidedAt: v?._t ?? null,
      });
    });
  }
  points.sort((a, b) => a.t - b.t);

  const stairs = (accept: (p: UpliftPoint) => boolean): UpliftStep[] => {
    const out: UpliftStep[] = [{ t: baseT, pct: 0, tput: baseline, label: "baseline" }];
    for (const p of points) {
      if (accept(p) && p.tput > out.at(-1)!.tput) out.push({ t: p.t, pct: p.pct, tput: p.tput, label: p.label });
    }
    return out;
  };
  const steps = stairs((p) => p.verdict === "keep");
  const provisional = stairs((p) => p.verdict !== "revert");
  const bestPct = steps.at(-1)!.pct;
  const allDecided = points.length > 0 && points.every((p) => p.verdict !== "pending");
  return {
    ...empty,
    validatedPct: allDecided ? num(view.summary.cumulative_gain_validated) : null,
    points,
    steps,
    provisional,
    bestPct,
  };
}

/** Height of each metric strip (mean TTFT, accuracy) under the uplift strip. */
export const METRIC_H = 78;

// Mean TTFT is in no progress note, only in each benchmark's own result file, so the page reads those artifacts:
// inferencex_result.json per benchmark round. Accuracy comes with the snapshot, from state.json (accuracyResults).
export type ResultFile = "inferencex_result.json";

export interface ResultSource {
  path: string;
  task_id: string;
  label: string;
  warmup: boolean;
}

export function resultSources(artifacts: Artifact[], file: ResultFile): ResultSource[] {
  const re = new RegExp(`^runs/(baseline|explore)/([^/]+)/(.+)/${file.replace(".", "\\.")}$`);
  const out: ResultSource[] = [];
  for (const a of artifacts) {
    const m = re.exec(a.path);
    if (!m) continue;
    const [, kind, task_id, rest] = m;
    const label = kind === "baseline" ? "baseline" : /(?:^|\/)v\d+_([^/]+)\//.exec(rest + "/")?.[1];
    if (label) out.push({ path: a.path, task_id, label, warmup: rest.includes("warmup_round") });
  }
  return out;
}

/** One file's value; `t` is when it was produced, `se` its standard error when the file has one. */
export interface MetricResult extends ResultSource {
  value: number;
  t: number;
  se?: number | null;
}

export interface MetricPoint {
  t: number;
  label: string;
  value: number;
  /** Other rounds of the same variant (warm-up), for the noise whisker. */
  earlier: number[];
  se: number | null;
  verdict: Verdict;
}

export interface MetricSeries {
  baseline: number | null;
  baselineSe: number | null;
  points: MetricPoint[];
  /** The kept configuration's value over time, on the uplift steps' times (`pct` holds the value). */
  steps: UpliftStep[];
  /** Like `steps`, on the uplift's provisional steps: where the line goes if the pending variants are kept. */
  provisional: UpliftStep[];
  current: number | null;
}

/**
 * A metric per measured variant, placed and judged like the uplift points so every strip lines up. The measured
 * round's value wins; warm-up rounds become `earlier`. A metric only taken in one round (accuracy) uses that one,
 * and a result without a task (accuracy) matches by label alone.
 */
export function metricSeries(up: Uplift, results: MetricResult[], now: number): MetricSeries {
  const of = (task: string | null, label: string) =>
    results.filter((r) => r.label === label && (task == null || !r.task_id || r.task_id === task) && r.t <= now);
  const pick = (rs: MetricResult[]) => {
    const main = rs.find((r) => !r.warmup) ?? rs[0];
    return main && { main, earlier: rs.filter((r) => r !== main).map((r) => r.value) };
  };
  const base = pick(of(null, "baseline"));
  const points: MetricPoint[] = [];
  if (base) {
    const { main } = base;
    points.push({
      t: up.baseT ?? main.t,
      label: "baseline",
      value: main.value,
      earlier: [],
      se: main.se ?? null,
      verdict: "keep",
    });
  }
  for (const p of up.points) {
    const hit = pick(of(p.task_id, p.label));
    if (hit) {
      const { main, earlier } = hit;
      points.push({ t: p.t, label: p.label, value: main.value, earlier, se: main.se ?? null, verdict: p.verdict });
    }
  }
  const follow = (stairs: UpliftStep[]) =>
    stairs.flatMap((s) => {
      const hit = points.find((p) => p.label === s.label && (s.label === "baseline" || p.t === s.t));
      return hit ? [{ t: s.t, pct: hit.value, tput: s.tput, label: s.label }] : [];
    });
  const steps = follow(up.steps);
  return {
    baseline: base?.main.value ?? null,
    baselineSe: base?.main.se ?? null,
    points,
    steps,
    provisional: follow(up.provisional),
    current: steps.at(-1)?.pct ?? null,
  };
}

/** How to read a value out of each result file. */
export const METRIC_READERS: Record<
  ResultFile,
  (d: Record<string, unknown>) => { value: number; t: number | null; se: number | null }
> = {
  "inferencex_result.json": (d) => ({
    value: Number(d.mean_ttft_ms),
    t: Number.isFinite(Number(d.benchmark_end_time_unix)) ? Number(d.benchmark_end_time_unix) : null,
    se: null,
  }),
};

/**
 * The accuracy state.json records (the snapshot's `summary.accuracy`, fractions) as metric results in percent: the
 * baseline's at the baseline's time *baseT*, each kept variant's at the time it was measured. They name no task, so
 * a variant's matches the uplift point with its label.
 */
export function accuracyResults(acc: AccuracyView | undefined, baseT: number | null): MetricResult[] {
  const result = (label: string, value: number, t: number): MetricResult => ({
    path: "state.json",
    task_id: "",
    label,
    warmup: false,
    value: value * 100,
    t,
    se: null,
  });
  if (!acc) return [];
  const out = acc.variants.map((v) => result(v.variant, v.value, v.t ?? 0));
  return acc.baseline == null ? out : [result("baseline", acc.baseline, baseT ?? 0), ...out];
}

/** `+17.5%` / `-3.0%`. */
export const fmtPct = (v: number | null | undefined, digits = 1): string =>
  v == null ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(digits)}%`;

/** SVG path of a step line through *steps*, held flat until *tEnd*. */
export function stairPath(steps: UpliftStep[], X: (t: number) => number, Y: (p: number) => number, tEnd: number) {
  if (!steps.length) return "";
  let d = `M${X(steps[0].t)},${Y(steps[0].pct)}`;
  for (let i = 1; i < steps.length; i++) d += `H${X(steps[i].t)}V${Y(steps[i].pct)}`;
  return d + `H${X(Math.max(tEnd, steps.at(-1)!.t))}`;
}
