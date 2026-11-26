// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The vertical tok/s axis the throughput chart and the TTFT scatter share: one range and one set of margins, so
// a throughput is at the same height in both.

export const CHART = { H: 190, LEFT: 60, RIGHT: 20, TOP: 12, BOTTOM: 34 };

const TICKS = [0, 0.25, 0.5, 0.75, 1];

export interface TputAxis {
  ticks: number[];
  y: (tput: number) => number;
}

/** An axis spanning *values*, padded by 5%. */
export function tputAxis(values: number[]): TputAxis {
  const lo = values.length ? Math.min(...values) * 0.95 : 0;
  const hi = values.length ? Math.max(...values) * 1.05 : 1;
  const plot = CHART.H - CHART.BOTTOM - CHART.TOP;
  return {
    ticks: values.length ? TICKS.map((t) => lo + (hi - lo) * t) : [],
    y: (v) => CHART.H - CHART.BOTTOM - ((v - lo) / (hi - lo || 1)) * plot,
  };
}
