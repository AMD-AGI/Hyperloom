// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { MARGIN, menuEntries, placeMenu } from "./contextmenu";

describe("placeMenu", () => {
  it("puts the menu at the pointer when it fits", () => {
    expect(placeMenu(100, 50, 160, 80, 1000, 800)).toEqual({ left: 100, top: 50 });
  });

  it("flips to the pointer's left and top near the right and bottom edges", () => {
    expect(placeMenu(950, 780, 160, 80, 1000, 800)).toEqual({ left: 790, top: 700 });
  });

  it("stays inside a viewport too small to flip into", () => {
    expect(placeMenu(100, 30, 160, 80, 200, 100)).toEqual({ left: 200 - MARGIN - 160, top: 100 - MARGIN - 80 });
    expect(placeMenu(-10, -10, 160, 80, 1000, 800)).toEqual({ left: MARGIN, top: MARGIN });
  });

  it("keeps the top-left corner on screen when the menu is larger than the viewport", () => {
    expect(placeMenu(50, 50, 160, 80, 120, 60)).toEqual({ left: MARGIN, top: MARGIN });
  });
});

describe("menuEntries", () => {
  const task = { kind: "task", id: "t1" } as const;

  it("offers Explain then Open for an item the narrator can explain", () => {
    expect(menuEntries(task, true, null)).toEqual([
      { id: "explain", label: "Explain", disabled: false },
      { id: "open", label: "Open", disabled: false },
    ]);
  });

  it("disables Explain while the same item is pending, not another one", () => {
    expect(menuEntries(task, true, { kind: "task", id: "t1" })[0]).toMatchObject({ id: "explain", disabled: true });
    expect(menuEntries(task, true, { kind: "task", id: "t2" })[0]).toMatchObject({ id: "explain", disabled: false });
  });

  it("shows only Open without a narrator or without a reference", () => {
    expect(menuEntries(task, false, null).map((e) => e.id)).toEqual(["open"]);
    expect(menuEntries(null, true, null).map((e) => e.id)).toEqual(["open"]);
  });
});
