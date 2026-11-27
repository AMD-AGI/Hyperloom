// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { reuse } from "./reuse";

describe("reuse", () => {
  const a = { id: "a", data: { n: 1 } };
  const b = { id: "b", data: { n: 2 } };

  it("keeps the old array when nothing changed", () => {
    const prev = [a, b];
    expect(reuse(prev, [structuredClone(a), structuredClone(b)])).toBe(prev);
  });
  it("keeps unchanged items and takes changed ones", () => {
    const b2 = { id: "b", data: { n: 3 } };
    const out = reuse([a, b], [structuredClone(a), b2]);
    expect(out[0]).toBe(a);
    expect(out[1]).toBe(b2);
  });
  it("ignores fields only the old item has", () => {
    const measured = { ...a, measured: { width: 132 } };
    const prev = [measured];
    expect(reuse(prev, [structuredClone(a)])).toBe(prev);
  });
  it("returns a new array when items are added, removed or reordered", () => {
    const prev = [a, b];
    expect(reuse(prev, [structuredClone(a)])).toEqual([a]);
    expect(reuse(prev, [structuredClone(b), structuredClone(a)])).not.toBe(prev);
    expect(reuse(prev, [structuredClone(b), structuredClone(a)])[0]).toBe(b);
    expect(reuse([], [a])).toEqual([a]);
  });
});
