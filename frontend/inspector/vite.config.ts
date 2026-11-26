// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vitest/config";
import { thirdPartyLicenses } from "./licenses";

// The Inspector the dev server forwards /api to (SSE included), e.g. the smoke replay.
const target = process.env.INSPECTOR_URL ?? "http://127.0.0.1:8798";

export default defineConfig({
  plugins: [svelte(), thirdPartyLicenses()],
  // Relative asset URLs, so the page works wherever the server mounts it.
  base: "./",
  build: {
    // Served by the Python inspector from its package directory; gitignored.
    outDir: "../../src/hyperloom/inspector/static/modern",
    emptyOutDir: true,
    target: "es2022",
    sourcemap: false,
    // Keep everything in files: the server's CSP has no 'unsafe-inline' for scripts.
    assetsInlineLimit: 0,
  },
  server: {
    host: "127.0.0.1",
    proxy: { "/api": { target, changeOrigin: true } },
  },
  test: { include: ["src/**/*.test.ts"], environment: "node" },
});
