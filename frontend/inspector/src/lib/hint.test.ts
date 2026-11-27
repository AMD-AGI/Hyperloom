// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { GAP, placeHint } from "./hint";

describe("placeHint", () => {
  const el = { left: 100, top: 200, bottom: 220 };

  it("puts the hint below its element, left-aligned with it", () => {
    expect(placeHint(el, 300, 60, 1200, 800)).toEqual({ left: 100, top: 220 + GAP });
  });

  it("goes above the element near the bottom edge", () => {
    const low = { left: 100, top: 760, bottom: 780 };
    expect(placeHint(low, 300, 60, 1200, 800)).toEqual({ left: 100, top: 760 - GAP - 60 });
  });

  it("stays inside the viewport at the right edge", () => {
    expect(placeHint({ ...el, left: 1100 }, 300, 60, 1200, 800).left).toBe(1200 - GAP - 300);
  });

  it("keeps on screen when it fits neither below nor above", () => {
    expect(placeHint({ left: -20, top: 30, bottom: 70 }, 300, 60, 1200, 100)).toEqual({
      left: GAP,
      top: 100 - GAP - 60,
    });
  });
});
