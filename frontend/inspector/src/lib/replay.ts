// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Pure helpers for the replay bar.

import type { Clock, PhaseBand } from "./types";

/** The replay speeds on offer, plus the current one when the CLI started with another. */
export const SPEEDS = [1, 5, 10, 30, 60, 120, 300, 600];

export function speedOptions(speed: number | undefined): number[] {
  return speed == null || SPEEDS.includes(speed) ? SPEEDS : [...SPEEDS, speed].sort((a, b) => a - b);
}

/** A running replay whose clock reached the end of the recording has nothing left to play. */
export function atEnd(clock: Clock, now: number): boolean {
  return clock.end != null && now >= clock.end - 0.5;
}

/**
 * What the play button does: resume or pause, and from the end start over, since resuming there would do nothing.
 */
export function playAction(clock: Clock, now: number): { t?: number; paused: boolean } {
  if (!clock.paused && !atEnd(clock, now)) return { paused: true };
  return atEnd(clock, now) && clock.start != null ? { t: clock.start, paused: false } : { paused: false };
}

/** Position of *now* within the recording, 0 to 1. */
export function progress(clock: Clock, now: number): number {
  const { start, end } = clock;
  if (start == null || end == null || end <= start) return 0;
  return Math.min(1, Math.max(0, (now - start) / (end - start)));
}

/** A stretch of the scrubber; `from` and `to` are its edges as fractions of the recording. */
export interface Chapter {
  phase: string;
  reason?: string;
  start: number;
  end: number;
  from: number;
  to: number;
}

/**
 * The phases as scrubber segments, clamped to the recording. Time before the first phase is a segment with no
 * name, and so is the whole recording when there are no phases.
 */
export function chapters(clock: Clock, phases: PhaseBand[]): Chapter[] {
  const { start, end } = clock;
  if (start == null || end == null || end <= start) return [];
  const lead: PhaseBand = { phase: "", start, end: phases[0]?.start ?? end };
  const at = (t: number) => (t - start) / (end - start);
  const out: Chapter[] = [];
  for (const p of [lead, ...phases]) {
    const a = Math.max(start, p.start);
    const b = Math.min(end, p.end ?? end);
    if (b <= a) continue;
    out.push({ phase: p.phase, reason: p.reason, start: a, end: b, from: at(a), to: at(b) });
  }
  return out;
}

/** The chapter at fraction *f* of the recording. */
export function chapterAt(chs: Chapter[], f: number): Chapter | undefined {
  return chs.find((c) => f < c.to) ?? chs[chs.length - 1];
}

/** Pixels between two chapters on the scrubber. */
export const GAP = 4;
/** The narrowest a chapter is drawn, so a short phase can still be seen and hovered. */
export const MIN_WIDTH = 6;
/** How far the pointer may move between press and release for the release to count as a click, not a drag. */
export const CLICK_SLOP = 4;

/** A chapter's box on a scrubber *px* wide. */
export interface Segment {
  left: number;
  width: number;
  /** Drawn at MIN_WIDTH, centred on its true midpoint and over its neighbours. */
  widened: boolean;
}

/**
 * Where each chapter is drawn: at its true place, less the gap after it, with chapters too short to see widened
 * in place rather than stretching the scale, so the segments stay lined up with the range input's thumb. Widened
 * chapters cover their neighbours but not each other: neighbouring ones are pushed apart, and all of them stay
 * on the scrubber.
 */
export function segments(chs: Chapter[], px: number): Segment[] {
  const segs = chs.map((c, i) => {
    const width = (c.to - c.from) * px - (i < chs.length - 1 ? GAP : 0);
    if (width >= MIN_WIDTH) return { left: c.from * px, width, widened: false };
    return { left: ((c.from + c.to) / 2) * px - MIN_WIDTH / 2, width: MIN_WIDTH, widened: true };
  });
  const widened = segs.filter((s) => s.widened);
  let floor = 0;
  for (const s of widened) {
    s.left = Math.max(s.left, floor);
    floor = s.left + MIN_WIDTH + GAP;
  }
  let ceiling = px;
  for (const s of widened.reverse()) {
    s.left = Math.min(s.left, ceiling - MIN_WIDTH);
    ceiling = s.left - GAP;
  }
  return segs;
}

/** The chapter under *x* pixels on a scrubber *px* wide; a widened chapter wins over the neighbour it covers. */
export function chapterAtPx(chs: Chapter[], x: number, px: number): Chapter | undefined {
  const segs = segments(chs, px);
  const i = segs.findIndex((s) => s.widened && x >= s.left && x <= s.left + s.width);
  return i >= 0 ? chs[i] : chapterAt(chs, px > 0 ? x / px : 0);
}

/**
 * Where a release on the scrubber seeks: a click goes to the start of the chapter under it, a drag or a key to
 * the exact time *t* the input reports.
 */
export function seekTarget(chs: Chapter[], t: number, click: { x: number; px: number } | null): number {
  return (click && chapterAtPx(chs, click.x, click.px)?.start) ?? t;
}
