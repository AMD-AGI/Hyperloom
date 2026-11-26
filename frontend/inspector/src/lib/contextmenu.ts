// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The right-click menu on the views' items: what a view hands over when it opens one, where the menu goes, and
// which entries it shows. Pure, so the placement and the entries are tested without a page.

import { sameItem } from "./narrator-store.svelte";
import type { ItemRef } from "./types";

/** A menu a view asked for: at the pointer, for an item (null when the item has no reference), and its left click. */
export interface MenuRequest {
  x: number;
  y: number;
  ref: ItemRef | null;
  open: () => void;
}

export type MenuEntry = { id: "explain" | "open"; label: string; disabled: boolean };

/** Space kept between the menu and the viewport's edges. */
export const MARGIN = 4;

/**
 * The menu's top-left corner: at the pointer, flipped to the pointer's other side when it would run past the right
 * or bottom edge, and kept inside the viewport when even that doesn't fit.
 */
export function placeMenu(
  x: number,
  y: number,
  w: number,
  h: number,
  vw: number,
  vh: number,
): { left: number; top: number } {
  const fit = (p: number, size: number, view: number) => {
    const at = p + size > view - MARGIN && p - size >= MARGIN ? p - size : p;
    return Math.max(MARGIN, Math.min(at, view - MARGIN - size));
  };
  return { left: fit(x, w, vw), top: fit(y, h, vh) };
}

/**
 * The menu's entries: **Explain** when the narrator can answer and the item has a reference (disabled while that
 * item's explanation is pending), then **Open**, which does what a left click does.
 */
export function menuEntries(ref: ItemRef | null, available: boolean, explaining: ItemRef | null): MenuEntry[] {
  const open: MenuEntry = { id: "open", label: "Open", disabled: false };
  if (!ref || !available) return [open];
  const busy = !!explaining && sameItem(explaining, ref);
  return [{ id: "explain", label: busy ? "Explaining…" : "Explain", disabled: busy }, open];
}
