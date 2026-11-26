// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Svelte Flow re-adopts and re-measures every node or edge whose object changed. Views arrive several times a second
// while a replay plays, and most carry no change for a given node or edge, so equal items keep their old objects.

// Only the fields of the new item are compared: Svelte Flow writes its own (`measured`, …) into the bound objects.
const sameFields = (p: object, n: object): boolean =>
  JSON.stringify(n) === JSON.stringify(Object.fromEntries(Object.keys(n).map((k) => [k, (p as never)[k]])));

/** *next* with each item that equals its predecessor of the same id replaced by that predecessor; *prev* itself when
 *  nothing changed. */
export function reuse<T extends { id: string }>(prev: readonly T[], next: T[]): T[] {
  const old = new Map(prev.map((p) => [p.id, p]));
  let same = prev.length === next.length;
  const out = next.map((n, i) => {
    const p = old.get(n.id);
    const keep = p !== undefined && sameFields(p, n);
    if (!keep || prev[i] !== p) same = false;
    return keep ? p : n;
  });
  return same ? (prev as T[]) : out;
}
