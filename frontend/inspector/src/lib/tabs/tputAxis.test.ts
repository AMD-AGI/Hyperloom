// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { CHART, tputAxis } from "./tputAxis";

describe("tputAxis", () => {
  it("pads the range by 5% and spans it with evenly spaced ticks", () => {
    const axis = tputAxis([100, 200]);
    expect(axis.ticks).toHaveLength(5);
    expect(axis.ticks[0]).toBeCloseTo(95);
    expect(axis.ticks[4]).toBeCloseTo(210);
    expect(axis.ticks[2]).toBeCloseTo(152.5);
  });

  it("puts the lowest tick at the bottom of the plot area and the highest at the top", () => {
    const axis = tputAxis([100, 200]);
    expect(axis.y(axis.ticks[0])).toBeCloseTo(CHART.H - CHART.BOTTOM);
    expect(axis.y(axis.ticks[4])).toBeCloseTo(CHART.TOP);
    expect(axis.y(200)).toBeLessThan(axis.y(100));
  });

  it("gives the same height for the same value, however it was reached", () => {
    expect(tputAxis([13736, 19276, 15000]).y(15000)).toBe(tputAxis([19276, 15000, 13736, 15000]).y(15000));
  });

  it("copes with one value and with none", () => {
    expect(Number.isFinite(tputAxis([150]).y(150))).toBe(true);
    const none = tputAxis([]);
    expect(none.ticks).toEqual([]);
    expect(Number.isFinite(none.y(5))).toBe(true);
  });
});
