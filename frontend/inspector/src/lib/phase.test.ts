// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { phaseColor } from "./phase";

describe("phase colours", () => {
  it("has one colour per orchestrator phase and a neutral one for the rest", () => {
    expect(phaseColor("FRAMEWORK_AGENT")).toBe("var(--phase-framework-agent)");
    expect(phaseColor("CLOSE")).toBe("var(--phase-close)");
    expect(phaseColor("CYCLE_END")).toBe("var(--phase-cycle-end)");
    expect(phaseColor("SYSTEM")).toBe("var(--phase-other)");
    expect(phaseColor("")).toBe("var(--phase-other)");
    expect(phaseColor(undefined)).toBe("var(--phase-other)");
  });
  it("names a colour that app.css defines for every phase", () => {
    const css = readFileSync(new URL("../app.css", import.meta.url), "utf8");
    for (const ph of ["PRELUDE", "ENABLEMENT", "FRAMEWORK_AGENT", "KERNEL_AGENT", "CYCLE_END", "SWEEP", "CLOSE", ""]) {
      expect(css).toContain(`${phaseColor(ph).slice(4, -1)}:`);
    }
  });
});
