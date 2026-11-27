// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { applyTheme, nextTheme, readTheme, writeTheme, type Theme } from "./theme";

/** The page's theme; `main.ts` applies the remembered one before the first render. */
class ThemeState {
  current = $state<Theme>(readTheme());

  cycle(): void {
    this.current = nextTheme(this.current);
    applyTheme(this.current);
    writeTheme(this.current);
  }
}

export const theme = new ThemeState();
