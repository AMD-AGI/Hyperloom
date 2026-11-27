// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { noSamplesNote } from "./resources";

describe("noSamplesNote", () => {
  it("waits only while a live run on this host is running", () => {
    expect(noSamplesNote({ replay: false, ended: false })).toBe("waiting for samples");
  });
  it("says a finished run ended before this Inspector sampled anything", () => {
    expect(noSamplesNote({ replay: false, ended: true })).toMatch(/^the run ended before this Inspector sampled/);
  });
  it("names the optimizer's host when it runs elsewhere, finished or not", () => {
    for (const ended of [false, true]) {
      expect(noSamplesNote({ replay: false, optimizerHost: "pod-a", ended })).toBe(
        "resources are sampled only on the optimizer's host (pod-a), not here",
      );
    }
  });
  it("keeps the replay message in a replay", () => {
    expect(noSamplesNote({ replay: true, optimizerHost: null, ended: false })).toMatch(/^no resource history/);
  });
});
