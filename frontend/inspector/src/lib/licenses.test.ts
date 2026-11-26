// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { packageDir } from "../../licenses";

describe("packageDir", () => {
  it("finds plain and scoped packages", () => {
    expect(packageDir("/r/node_modules/svelte/src/internal/client/index.js")).toBe("/r/node_modules/svelte");
    expect(packageDir("/r/node_modules/@xyflow/svelte/dist/lib/index.js?v=1")).toBe("/r/node_modules/@xyflow/svelte");
  });
  it("uses the innermost node_modules", () => {
    expect(packageDir("/r/node_modules/a/node_modules/b/index.js")).toBe("/r/node_modules/a/node_modules/b");
  });
  it("ignores our sources and virtual modules", () => {
    expect(packageDir("/r/src/App.svelte")).toBeNull();
    expect(packageDir("\0vite/preload-helper")).toBeNull();
  });
});
