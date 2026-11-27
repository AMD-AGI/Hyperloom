// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { applyTheme, nextTheme, readTheme, THEME_KEY, writeTheme } from "./theme";

const memory = () => {
  const m = new Map<string, string>();
  return {
    m,
    getItem: (k: string) => m.get(k) ?? null,
    setItem: (k: string, v: string) => void m.set(k, v),
    removeItem: (k: string) => void m.delete(k),
  };
};
const broken = {
  getItem: () => {
    throw new Error("denied");
  },
  setItem: () => {
    throw new Error("denied");
  },
  removeItem: () => {
    throw new Error("denied");
  },
};

describe("theme", () => {
  it("cycles system, light, dark", () => {
    expect([nextTheme("system"), nextTheme("light"), nextTheme("dark")]).toEqual(["light", "dark", "system"]);
  });
  it("remembers a forced theme and forgets system", () => {
    const s = memory();
    writeTheme("dark", s);
    expect(s.m.get(THEME_KEY)).toBe("dark");
    expect(readTheme(s)).toBe("dark");
    writeTheme("system", s);
    expect(s.m.has(THEME_KEY)).toBe(false);
    s.m.set(THEME_KEY, "sepia");
    expect(readTheme(s)).toBe("system");
  });
  it("follows the system when storage is missing or throws", () => {
    expect(readTheme(null)).toBe("system");
    expect(readTheme(broken)).toBe("system");
    expect(() => writeTheme("light", broken)).not.toThrow();
  });
  it("sets data-theme only when forced", () => {
    const root = { dataset: {} as Record<string, string> } as unknown as HTMLElement;
    applyTheme("light", root);
    expect(root.dataset.theme).toBe("light");
    applyTheme("system", root);
    expect(root.dataset.theme).toBeUndefined();
  });
});
