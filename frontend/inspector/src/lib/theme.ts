// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The colour theme: follow the system, or force light or dark. The choice is a per-viewer convenience kept in
// localStorage; the key predates this UI, so renaming it would forget every saved choice. Storage can be missing
// or throw (private windows, blocked site data), and then the page follows the system.

export type Theme = "system" | "light" | "dark";

export const THEME_KEY = "hyperloom-inspector-theme";
const ORDER: Theme[] = ["system", "light", "dark"];

type Store = Pick<Storage, "getItem" | "setItem" | "removeItem">;

function storage(): Store | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

export function readTheme(s: Store | null = storage()): Theme {
  try {
    const v = s?.getItem(THEME_KEY);
    return v === "light" || v === "dark" ? v : "system";
  } catch {
    return "system";
  }
}

export function writeTheme(t: Theme, s: Store | null = storage()): void {
  try {
    if (t === "system") s?.removeItem(THEME_KEY);
    else s?.setItem(THEME_KEY, t);
  } catch {
    // Not remembered; the page still switches.
  }
}

export const nextTheme = (t: Theme): Theme => ORDER[(ORDER.indexOf(t) + 1) % ORDER.length];

/** Sets `data-theme` on the root element (absent for "system"), which the palettes in app.css key on. */
export function applyTheme(t: Theme, root: HTMLElement = document.documentElement): void {
  if (t === "system") delete root.dataset.theme;
  else root.dataset.theme = t;
}
