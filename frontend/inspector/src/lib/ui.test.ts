// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { UiState } from "./ui.svelte";

describe("UiState selection", () => {
  it("filters to a picked node and shows it, and clears both on a second pick", () => {
    const ui = new UiState();
    ui.pickNode("critic");
    expect(ui.filter).toBe("critic");
    expect(ui.selection).toEqual({ kind: "node", id: "critic" });
    ui.pickNode("critic");
    expect(ui.filter).toBeNull();
    expect(ui.selection).toBeNull();
  });

  it("keeps another record open when a node filter is cleared", () => {
    const ui = new UiState();
    ui.pickNode("critic");
    ui.select({ kind: "task", id: "t1" });
    ui.pickNode("critic");
    expect(ui.selection).toEqual({ kind: "task", id: "t1" });
  });

  it("walks back along links followed in the drawer, and a fresh pick drops the trail", () => {
    const ui = new UiState();
    ui.select({ kind: "task", id: "t1" });
    ui.follow({ kind: "node", id: "critic" });
    ui.follow({ kind: "task", id: "t2" });
    expect(ui.trail).toHaveLength(2);
    ui.back();
    expect(ui.selection).toEqual({ kind: "node", id: "critic" });
    ui.select({ kind: "task", id: "t3" });
    expect(ui.trail).toEqual([]);
    ui.back();
    expect(ui.selection).toEqual({ kind: "task", id: "t3" });
  });
});
