// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { untrack } from "svelte";
import { prefersReducedMotion } from "svelte/motion";
import type { InspectorStore } from "./store.svelte";

/**
 * The session time for a component to display. A running replay updates it every animation frame, since at x600 a
 * half-second tick would jump five minutes; live, paused or under reduced motion it updates twice a second. Call it
 * during component setup (it registers effects).
 */
export function sessionClock(getStore: () => InspectorStore): { readonly current: number } {
  let now = $state(untrack(() => getStore().now()));
  const smooth = $derived.by(() => {
    const c = getStore().view?.clock;
    return c?.mode === "replay" && !c.paused && !prefersReducedMotion.current;
  });
  // Every view from the server resyncs; the timer below depends only on `smooth`, so a stream of views can't keep
  // restarting it before it fires.
  $effect(() => {
    void getStore().view;
    now = untrack(() => getStore().now());
  });
  $effect(() => {
    const store = untrack(getStore);
    if (smooth) {
      let id = requestAnimationFrame(function tick() {
        now = store.now();
        id = requestAnimationFrame(tick);
      });
      return () => cancelAnimationFrame(id);
    }
    const id = setInterval(() => (now = store.now()), 500);
    return () => clearInterval(id);
  });
  return {
    get current() {
      return now;
    },
  };
}
