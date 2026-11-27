// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { claim, fromSnapshot, merge, outOfSync, type StreamState } from "./streams";
import type { Cursors, StreamItem } from "./types";

const items = (...ts: number[]): StreamItem[] => ts.map((t) => ({ _t: t }));
const times = (s: StreamState, name: "event" = "event") => s.items[name].map((i) => i._t);

function merged(state: StreamState, starts: Cursors, batch: Record<string, StreamItem[]>): StreamState {
  const r = merge(state, starts, batch);
  if ("gap" in r) throw new Error("unexpected gap in " + r.gap);
  return r.state;
}

describe("stream cursors", () => {
  it("starts from the snapshot's released counts", () => {
    const s = fromSnapshot({ event: items(1, 2) }, { event: 5 });
    expect(s.cursors.event).toBe(5); // absolute: three older items were trimmed server-side
    expect(s.cursors.llm_call).toBe(0);
  });
  it("claims consecutive ranges on the connection", () => {
    const conn: Cursors = { event: 5 };
    expect(claim(conn, { event: items(6, 7) })).toEqual({ event: 5 });
    expect(claim(conn, { event: items(8), journal: items(9) })).toEqual({ event: 7, journal: 0 });
    expect(conn).toEqual({ event: 8, journal: 1 });
  });
  it("appends a batch that continues the cursor", () => {
    const s = merged(fromSnapshot({ event: items(1, 2) }, { event: 2 }), { event: 2 }, { event: items(3) });
    expect(times(s)).toEqual([1, 2, 3]);
    expect(s.cursors.event).toBe(3);
  });
  it("skips the overlap with what the snapshot already held", () => {
    const s = merged(fromSnapshot({ event: items(1, 2, 3) }, { event: 3 }), { event: 1 }, { event: items(2, 3, 4) });
    expect(times(s)).toEqual([1, 2, 3, 4]);
    expect(s.cursors.event).toBe(4);
  });
  it("ignores a batch that is entirely old", () => {
    const s = fromSnapshot({ event: items(1, 2) }, { event: 2 });
    expect(merge(s, { event: 0 }, { event: items(1) })).toEqual({ state: s });
  });
  it("reports a gap when items were missed", () => {
    const r = merge(fromSnapshot({ event: items(1) }, { event: 1 }), { event: 3 }, { event: items(4) });
    expect(r).toEqual({ gap: "event" });
  });
  it("does not modify the input state", () => {
    const s = fromSnapshot({ event: items(1) }, { event: 1 });
    merged(s, { event: 1 }, { event: items(2) });
    expect(times(s)).toEqual([1]);
    expect(s.cursors.event).toBe(1);
  });
  it("ignores unknown streams", () => {
    const batch = { future: items(1) } as Record<string, StreamItem[]>;
    const s = fromSnapshot({}, {});
    expect(merge(s, {}, batch)).toEqual({ state: s });
  });
  it("detects a hello that disagrees with the client's cursors", () => {
    const s = fromSnapshot({}, { event: 3 });
    expect(outOfSync({ event: 3 }, s.cursors)).toBe(false);
    expect(outOfSync({ event: 4 }, s.cursors)).toBe(true);
    expect(outOfSync({ event: 2 }, s.cursors)).toBe(true);
  });
});
