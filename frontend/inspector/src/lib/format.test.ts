// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { fmtArtifactCount, fmtBytes, fmtCompact, fmtDur, fmtNum, fmtTime, fmtUsd, preview } from "./format";

describe("format", () => {
  it("formats numbers with a placeholder for missing values", () => {
    expect(fmtNum(null)).toBe("–");
    expect(fmtNum(1685.24)).toBe((1685.2).toLocaleString());
  });
  it("formats counts compactly", () => {
    const compact = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 });
    expect([fmtCompact(null), fmtCompact(NaN)]).toEqual(["–", "–"]);
    expect(fmtCompact(6888915)).toBe(compact.format(6888915));
    expect(fmtCompact(950)).toBe(compact.format(950));
  });
  it("formats dollars with the digits small amounts need", () => {
    const usd = (opts: Intl.NumberFormatOptions) =>
      new Intl.NumberFormat(undefined, {
        style: "currency",
        currency: "USD",
        currencyDisplay: "narrowSymbol",
        ...opts,
      });
    expect([fmtUsd(null), fmtUsd(Infinity)]).toEqual(["–", "–"]);
    expect(fmtUsd(10.717503)).toBe(usd({ minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(10.72));
    expect(fmtUsd(0)).toBe(usd({ minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(0));
    expect(fmtUsd(0.289475)).toBe(usd({ minimumFractionDigits: 2, maximumFractionDigits: 3 }).format(0.289));
    expect(fmtUsd(0.0042)).toBe(usd({ minimumFractionDigits: 2, maximumFractionDigits: 4 }).format(0.0042));
    expect(fmtUsd(5, true)).toBe(usd({ minimumFractionDigits: 0, maximumFractionDigits: 2 }).format(5));
    expect(fmtUsd(123456)).toBe(usd({ notation: "compact", maximumFractionDigits: 1 }).format(123456));
  });
  it("says when the artifact list is cut", () => {
    const n = (v: number) => v.toLocaleString();
    expect(fmtArtifactCount(12, null)).toBe("12 artifacts");
    expect(fmtArtifactCount(20000, 20000)).toBe(`${n(20000)}+ artifacts (list cut at ${n(20000)})`);
    // In a replay the clock can stand before the cut: the count has no "+", the cut is still named.
    expect(fmtArtifactCount(500, 20000)).toBe(`${n(500)} artifacts (list cut at ${n(20000)})`);
  });
  it("formats sizes in binary units", () => {
    expect([
      fmtBytes(undefined),
      fmtBytes(512),
      fmtBytes(2048),
      fmtBytes(3 * 2 ** 20),
      fmtBytes(0.8 * 2 ** 31),
    ]).toEqual(["–", "512 B", "2.0 KiB", "3.0 MiB", "1.6 GiB"]);
  });
  it("formats durations at two units of precision", () => {
    expect([fmtDur(null), fmtDur(Infinity), fmtDur(-3), fmtDur(42), fmtDur(125), fmtDur(3 * 3600 + 5 * 60)]).toEqual([
      "–",
      "–",
      "0s",
      "42s",
      "2m05s",
      "3h05m",
    ]);
  });
  it("adds the date only far from now", () => {
    expect(fmtTime(null)).toBe("–");
    expect(fmtTime(1000, 1000)).toMatch(/^\d{2}:\d{2}:\d{2}$/);
    expect(fmtTime(1000, 1000 + 2 * 86400)).toMatch(/^\d{2}:\d{2}:\d{2} \S+$/);
    const t = 1790347662;
    const d = new Date(t * 1000);
    const time = d.toLocaleTimeString([], { hour12: false });
    expect(fmtTime(t, t)).toBe(time);
    expect(fmtTime(t, t + 2 * 86400)).toBe(`${time} ${d.toLocaleDateString([], { month: "2-digit", day: "2-digit" })}`);
    expect(fmtNum(1234.56, 0)).toBe((1234.56).toLocaleString(undefined, { maximumFractionDigits: 0 }));
  });
  it("previews values", () => {
    expect(preview("abcdef", 3)).toBe("abc…");
    expect(preview({ a: 1 })).toBe('{"a":1}');
    expect(preview(undefined)).toBe("");
  });
});
