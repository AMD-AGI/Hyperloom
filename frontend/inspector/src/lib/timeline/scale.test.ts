// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { GraphNode, Timeline } from "../types";
import {
  activityPoints,
  callLane,
  axisTicks,
  eventLane,
  fitRange,
  GAP_PX,
  idleGaps,
  LABEL,
  lanes,
  MIN_SPAN,
  niceTicks,
  podMarkers,
  podStarts,
  tlScale,
  tsOf,
  zoomAbout,
} from "./scale";

const tl = (over: Partial<Timeline> = {}): Timeline => ({ bars: [], phases: [], steps: [], ...over });
const node = (id: string, group: string, label = id) => ({ id, group, label }) as GraphNode;

describe("fitRange", () => {
  it("uses the recording in replay and first activity to now live, with headroom", () => {
    const t = tl({ bars: [{ start: 500 } as never], phases: [{ phase: "P", start: 400, end: null }] });
    expect(fitRange({ mode: "replay", now: 0, start: 100, end: 900 }, t, 5000)).toEqual({ t0: 100, t1: 900 });
    expect(fitRange({ mode: "live", now: 0 }, t, 5000)).toEqual({ t0: 400, t1: 5046 });
    expect(fitRange({ mode: "live", now: 0 }, tl(), 5000)).toEqual({ t0: 1400, t1: 5036 });
    expect(fitRange({ mode: "live", now: 0 }, t, 410)).toEqual({ t0: 400, t1: 460 });
    // A long run: 5 minutes of headroom, not 1% of the weeks.
    expect(fitRange({ mode: "live", now: 0 }, t, 400 + 22 * 86400)).toEqual({ t0: 400, t1: 400 + 22 * 86400 + 300 });
  });

  it("ends a run that has gone quiet at its last activity, not now", () => {
    const quiet = tl({ bars: [{ start: 500 } as never], phases: [{ phase: "P", start: 400, end: null }], end: 2000 });
    expect(fitRange({ mode: "live", now: 0 }, quiet, 400 + 22 * 86400)).toEqual({ t0: 400, t1: 2000 });
    expect(fitRange({ mode: "live", now: 0 }, { ...quiet, end: 420 }, 5000)).toEqual({ t0: 400, t1: 460 });
  });
});

describe("idleGaps", () => {
  it("finds stretches above max(300 s, 3% of the range), padded", () => {
    const gaps = idleGaps([100, 200, 5000, 5100], 0, 6000, 6000);
    expect(gaps).toEqual([
      { a: 320, b: 4880 },
      { a: 5145, b: 5955 },
    ]);
  });
  it("ignores points outside the range and short pauses", () => {
    expect(idleGaps([-5000, 50, 250, 99999], 0, 300, 300)).toEqual([]);
  });
  it("counts now as activity", () => {
    expect(idleGaps([], 0, 10000, 2000).map((g) => [Math.round(g.a), Math.round(g.b)])).toEqual([
      [100, 1900],
      [2120, 9880],
    ]);
  });
});

describe("tlScale", () => {
  const gaps = [{ a: 1000, b: 9000 }];
  it("maps linearly without gaps", () => {
    const sc = tlScale(0, 1000, LABEL + 8 + 1000);
    expect(sc.X(0)).toBe(LABEL);
    expect(sc.X(500)).toBe(LABEL + 500);
    expect(sc.T(LABEL + 250)).toBe(250);
  });
  it("gives each gap GAP_PX and round-trips", () => {
    const sc = tlScale(0, 10000, LABEL + 8 + 2000 + GAP_PX, gaps);
    expect(sc.active).toBe(2000);
    expect(sc.X(1000)).toBe(LABEL + 1000);
    expect(sc.X(9000)).toBe(LABEL + 1000 + GAP_PX);
    expect(sc.X(5000)).toBe(LABEL + 1000 + GAP_PX / 2);
    for (const t of [0, 400, 1000, 3000, 9000, 9500, 10000]) expect(sc.T(sc.X(t))).toBeCloseTo(t, 6);
  });
  it("drops gaps that would take more than half the width", () => {
    const many = Array.from({ length: 20 }, (_, i) => ({ a: i * 100 + 10, b: i * 100 + 90 }));
    expect(tlScale(0, 2000, LABEL + 8 + 400, many).gaps).toEqual([]);
  });
  it("extrapolates past the range end", () => {
    const sc = tlScale(0, 1000, LABEL + 8 + 1000);
    expect(sc.X(1100)).toBe(LABEL + 1100);
  });
});

describe("ticks", () => {
  it("picks a round step", () => {
    expect(niceTicks(0, 3600, 4)).toEqual([0, 900, 1800, 2700, 3600]);
    expect(niceTicks(95, 160, 2)).toEqual([120]);
  });
  it("labels the start of each stretch, skips gaps and crowding", () => {
    const r = { t0: 10, t1: 10000 };
    const sc = tlScale(r.t0, r.t1, 1000, [{ a: 1000, b: 9000 }]);
    const ticks = axisTicks(r, 1000, sc);
    expect(ticks[0]).toBe(10);
    expect(ticks).toContain(9000);
    expect(ticks.some((t) => t > 1000 && t < 9000)).toBe(false);
    ticks.slice(1).forEach((t, i) => expect(sc.X(t) - sc.X(ticks[i])).toBeGreaterThanOrEqual(60));
  });
});

describe("pod markers", () => {
  const run = {
    pod_history: [
      { acquired_at: "2026-09-25T10:57:18Z", hostname: "a", pid: 1 },
      { acquired_at: "2026-09-25T10:57:40Z", hostname: "a", pid: 2 },
      { acquired_at: "2026-09-25T14:43:00Z", hostname: "b", pid: 3 },
      { acquired_at: "bogus", hostname: "c", pid: 4 },
    ],
  };
  const t0 = Date.parse("2026-09-25T10:57:19Z") / 1000;
  it("keeps parsed starts up to now", () => {
    expect(podStarts(run, t0 + 3600).map((p) => p.i)).toEqual([0, 1]);
    expect(podStarts(run, t0 + 86400).map((p) => p.pid)).toEqual([1, 2, 3]);
  });
  it("labels start and resumes, clamps to the range and thins crowded labels", () => {
    const r = { t0, t1: t0 + 20000 };
    const m = podMarkers(podStarts(run, r.t1), r, (t) => t - t0);
    expect(m.map((p) => [p.label, p.x, p.crowded])).toEqual([
      ["start", 0, false],
      ["resume 1", 21, true],
      ["resume 2", 13541, false],
    ]);
  });
});

describe("lanes", () => {
  const nodes = [node("spec:b", "spec"), node("bus", "core"), node("critic", "llm"), node("coordinator", "core")];
  it("orders by graph row then id, without the bus", () => {
    expect(lanes([...nodes, node("x", "other")])).toEqual(["critic", "coordinator", "x", "spec:b"]);
  });
  it("maps a sender to its lane", () => {
    expect(eventLane(nodes, "b")).toBe("spec:b");
    expect(eventLane(nodes, "critic")).toBe("critic");
    expect(eventLane(nodes, "unknown")).toBe("coordinator");
    expect(eventLane(nodes, undefined)).toBe("coordinator");
  });
});

describe("callLane", () => {
  const tasks = [{ task_id: "t1", node: "spec:research_scout" }] as never;
  it("puts a call on its task's node, else on its component's", () => {
    expect(callLane(tasks, { _t: 0, component: "specialist", task_id: "t1" })).toBe("spec:research_scout");
    expect(callLane(tasks, { _t: 0, component: "critic", task_id: "gone" })).toBe("critic");
    expect(callLane(tasks, { _t: 0, component: "framework_agent" })).toBe("agent:framework");
    expect(callLane(tasks, { _t: 0, role: "orchestration" })).toBe("orchestration");
  });
});

describe("helpers", () => {
  it("reads a record's time", () => {
    expect(tsOf({ _t: 5 })).toBe(5);
    expect(tsOf({ _t: undefined as never, ts: "1970-01-01T00:00:10Z" })).toBe(10);
  });
  it("collects activity, sorted", () => {
    const t = tl({ bars: [{ start: 5, end: null } as never], steps: [{ start: 3 } as never] });
    expect(activityPoints(t, [], [[{ _t: 4 }], [{ _t: 1 }]])).toEqual([1, 3, 4, 5]);
  });
  it("zooms about a point and keeps a minimum span", () => {
    expect(zoomAbout({ t0: 0, t1: 100 }, 25, 0.5)).toEqual({ t0: 12.5, t1: 62.5 });
    const z = zoomAbout({ t0: 0, t1: 30 }, 15, 0.1);
    expect(z.t1 - z.t0).toBe(MIN_SPAN);
  });
});
