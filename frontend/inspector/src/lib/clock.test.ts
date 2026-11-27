// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { sessionNow } from "./clock";

describe("sessionNow", () => {
  it("is wall time live or before the first view", () => {
    expect(sessionNow(null, 0, 500)).toBe(500);
    expect(sessionNow({ mode: "live", now: 100 }, 0, 500)).toBe(500);
  });
  it("advances a running replay by its speed", () => {
    expect(sessionNow({ mode: "replay", now: 100, speed: 60 }, 1000, 1002)).toBe(220);
  });
  it("holds a paused replay and stops at the end", () => {
    expect(sessionNow({ mode: "replay", now: 100, speed: 60, paused: true }, 1000, 1002)).toBe(100);
    expect(sessionNow({ mode: "replay", now: 100, speed: 60, end: 150 }, 1000, 1002)).toBe(150);
  });
});
