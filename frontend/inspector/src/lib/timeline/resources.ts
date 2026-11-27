// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// What the memory and GPU strip says when it has no samples in its range.

export interface SampleContext {
  replay: boolean;
  /** The optimizer's host when the session's lock names a host other than the Inspector's. */
  optimizerHost?: string | null;
  /** Live, whether the run has ended (the timeline stops at its last activity). */
  ended: boolean;
}

/** Why the strip is empty: samples are taken only by a live Inspector on the optimizer's host, while the run runs. */
export function noSamplesNote({ replay, optimizerHost, ended }: SampleContext): string {
  if (replay) return "no resource history for this session (kept only while an Inspector runs live)";
  if (optimizerHost) return `resources are sampled only on the optimizer's host (${optimizerHost}), not here`;
  if (ended) return "the run ended before this Inspector sampled anything (samples are kept only while one runs live)";
  return "waiting for samples";
}
