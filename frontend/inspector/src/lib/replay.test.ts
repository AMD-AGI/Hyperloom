// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import {
  atEnd,
  chapterAt,
  chapterAtPx,
  chapters,
  GAP,
  MIN_WIDTH,
  playAction,
  progress,
  seekTarget,
  segments,
  SPEEDS,
  speedOptions,
} from "./replay";
import type { Clock } from "./types";

const clock = (c: Partial<Clock>): Clock => ({ mode: "replay", now: 150, start: 100, end: 200, speed: 60, ...c });

describe("replay helpers", () => {
  it("adds a speed the CLI set that the list lacks", () => {
    expect(speedOptions(60)).toBe(SPEEDS);
    expect(speedOptions(2)).toEqual([1, 2, 5, 10, 30, 60, 120, 300, 600]);
  });
  it("pauses, resumes, and starts over from the end", () => {
    expect(playAction(clock({}), 150)).toEqual({ paused: true });
    expect(playAction(clock({ paused: true }), 150)).toEqual({ paused: false });
    expect(atEnd(clock({}), 200)).toBe(true);
    expect(playAction(clock({}), 200)).toEqual({ t: 100, paused: false });
    expect(playAction(clock({ paused: true }), 200)).toEqual({ t: 100, paused: false });
  });
  it("places now within the recording", () => {
    expect(progress(clock({}), 150)).toBe(0.5);
    expect(progress(clock({}), 250)).toBe(1);
    expect(progress(clock({ end: null }), 150)).toBe(0);
  });
  it("splits the recording into chapters, with an unnamed lead-in", () => {
    const chs = chapters(clock({}), [
      { phase: "PRELUDE", start: 125, end: 175, reason: "start" },
      { phase: "SYSTEM", start: 175, end: null },
    ]);
    expect(chs.map((c) => [c.phase, c.start, c.end, c.from, c.to])).toEqual([
      ["", 100, 125, 0, 0.25],
      ["PRELUDE", 125, 175, 0.25, 0.75],
      ["SYSTEM", 175, 200, 0.75, 1],
    ]);
    expect(chs[1].reason).toBe("start");
  });
  it("clamps chapters to the recording and drops empty ones", () => {
    const chs = chapters(clock({}), [
      { phase: "PRELUDE", start: 50, end: 150 },
      { phase: "SYSTEM", start: 150, end: 250 },
      { phase: "CLOSE", start: 250, end: null },
    ]);
    expect(chs.map((c) => [c.phase, c.start, c.end])).toEqual([
      ["PRELUDE", 100, 150],
      ["SYSTEM", 150, 200],
    ]);
  });
  it("is one unnamed chapter without phases, and none without a range", () => {
    expect(chapters(clock({}), []).map((c) => [c.phase, c.from, c.to])).toEqual([["", 0, 1]]);
    expect(chapters(clock({ end: null }), [])).toEqual([]);
  });
  it("finds the chapter under a position", () => {
    const chs = chapters(clock({}), [{ phase: "PRELUDE", start: 150, end: null }]);
    expect(chapterAt(chs, 0.2)?.phase).toBe("");
    expect(chapterAt(chs, 0.5)?.phase).toBe("PRELUDE");
    expect(chapterAt(chs, 1)?.phase).toBe("PRELUDE");
    expect(chapterAt([], 0.5)).toBeUndefined();
  });
});

describe("chapter segments", () => {
  // 1000 px over 100 s: PRELUDE 0-50 s, a 0.2 s SWEEP, then CLOSE; no lead-in.
  const chs = chapters(clock({ start: 0, end: 100 }), [
    { phase: "PRELUDE", start: 0, end: 50 },
    { phase: "SWEEP", start: 50, end: 50.2 },
    { phase: "CLOSE", start: 50.2, end: null },
  ]);
  const px = 1000;

  it("draws chapters at their place, less the gap after each but the last", () => {
    const segs = segments(chs, px);
    expect(segs[0]).toEqual({ left: 0, width: 500 - GAP, widened: false });
    expect(segs[2].left).toBeCloseTo(502);
    expect(segs[2].width).toBeCloseTo(498);
  });
  it("widens a chapter too short to see, centred on its midpoint, leaving the others", () => {
    const s = segments(chs, px)[1];
    expect(s.widened).toBe(true);
    expect(s.width).toBe(MIN_WIDTH);
    expect(s.left + s.width / 2).toBeCloseTo(501);
  });
  it("keeps widened chapters on the scrubber and off each other", () => {
    // A 15 s SWEEP then a 1 s CLOSE at the very end, as when a run stops right after CLOSE starts.
    const edge = chapters(clock({ start: 0, end: 1000 }), [
      { phase: "SWEEP", start: 998.4, end: 999.9 },
      { phase: "CLOSE", start: 999.9, end: null },
    ]);
    const [, sweep, close] = segments(edge, px);
    expect(close.left).toBe(px - MIN_WIDTH);
    expect(sweep.left).toBe(close.left - GAP - MIN_WIDTH);
    expect(chapterAtPx(edge, close.left + 1, px)?.phase).toBe("CLOSE");
    expect(chapterAtPx(edge, sweep.left + 1, px)?.phase).toBe("SWEEP");
  });
  it("finds the chapter under a pixel, the widened one over its neighbours", () => {
    expect(chapterAtPx(chs, 200, px)?.phase).toBe("PRELUDE");
    expect(chapterAtPx(chs, 499, px)?.phase).toBe("SWEEP");
    expect(chapterAtPx(chs, 503, px)?.phase).toBe("SWEEP");
    expect(chapterAtPx(chs, 700, px)?.phase).toBe("CLOSE");
  });
  it("seeks a click to the chapter start, a drag or a key to the exact time", () => {
    expect(seekTarget(chs, 73.4, { x: 700, px })).toBe(50.2);
    expect(seekTarget(chs, 50.1, { x: 501, px })).toBe(50);
    expect(seekTarget(chs, 73.4, null)).toBe(73.4);
    expect(seekTarget([], 73.4, { x: 700, px })).toBe(73.4);
  });
});
