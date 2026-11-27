// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import type { Clock } from "./types";

/**
 * The session time shown at wall time *wall* (epoch seconds). Live mode is wall time; a running replay advances from
 * the last clock the server sent (received at *receivedAt*) by its speed, and stops at the end of the recording.
 */
export function sessionNow(clock: Clock | null | undefined, receivedAt: number, wall: number): number {
  if (!clock || clock.mode !== "replay") return wall;
  if (clock.paused) return clock.now;
  const t = clock.now + (wall - receivedAt) * (clock.speed || 1);
  return clock.end != null ? Math.min(t, clock.end) : t;
}
