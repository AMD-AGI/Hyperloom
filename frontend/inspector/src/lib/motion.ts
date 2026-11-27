// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { cubicOut } from "svelte/easing";
import { prefersReducedMotion } from "svelte/motion";

/** A duration in ms, or 0 when the user asked for reduced motion. */
export const ms = (duration: number): number => (prefersReducedMotion.current ? 0 : duration);

/** Options for `Tween`s that follow live values. */
export const follow = () => ({ duration: ms(600), easing: cubicOut });
