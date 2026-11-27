// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Client side of the stream cursors in watcher.py. Cursors are absolute item counts per stream, so they stay valid
// after the server or the client drops old items. The SSE connection keeps its own cursors, starting at the `hello`
// event's `released`; each `items` batch continues from there, which gives every batch an absolute start. A batch
// that starts past the client's cursor means items were missed (the page must refetch the snapshot); one that starts
// before it overlaps what the snapshot already held, and the overlap is skipped.

import { STREAM_NAMES, type Cursors, type StreamItem, type StreamName } from "./types";

export type StreamItems = Record<StreamName, StreamItem[]>;

export interface StreamState {
  items: StreamItems;
  cursors: Record<StreamName, number>;
}

/** Most items kept per stream; older ones are dropped from the head (cursors stay absolute). */
export const CAPS: Partial<Record<StreamName, number>> = { metrics: 4000 };

export function emptyItems(): StreamItems {
  return Object.fromEntries(STREAM_NAMES.map((n) => [n, []])) as unknown as StreamItems;
}

export function fromSnapshot(streams: Partial<StreamItems>, released: Cursors): StreamState {
  const items = emptyItems();
  const cursors = {} as Record<StreamName, number>;
  for (const n of STREAM_NAMES) {
    items[n] = cap(n, streams[n] ?? []);
    cursors[n] = released[n] ?? items[n].length;
  }
  return { items, cursors };
}

/** Where each stream in *batch* starts, given the connection cursors; advances *conn* past the batch. */
export function claim(conn: Cursors, batch: Partial<StreamItems>): Cursors {
  const starts: Cursors = {};
  for (const [name, items] of Object.entries(batch) as [StreamName, StreamItem[]][]) {
    const at = conn[name] ?? 0;
    starts[name] = at;
    conn[name] = at + items.length;
  }
  return starts;
}

export type MergeResult = { state: StreamState } | { gap: StreamName };

/** Append *batch* (starting at *starts*) to *state*, returning a new state; the input is not modified. */
export function merge(state: StreamState, starts: Cursors, batch: Partial<StreamItems>): MergeResult {
  const items = { ...state.items };
  const cursors = { ...state.cursors };
  for (const [name, got] of Object.entries(batch) as [StreamName, StreamItem[]][]) {
    if (!(name in cursors)) continue; // a stream this client doesn't know
    const start = starts[name] ?? cursors[name];
    if (start > cursors[name]) return { gap: name };
    const fresh = got.slice(cursors[name] - start);
    if (!fresh.length) continue;
    items[name] = cap(name, items[name].concat(fresh));
    cursors[name] += fresh.length;
  }
  return { state: { items, cursors } };
}

/** True when the server's cursors (from `hello`) differ from the client's, so the snapshot must be refetched. */
export function outOfSync(server: Cursors, client: Record<StreamName, number>): boolean {
  return STREAM_NAMES.some((n) => (server[n] ?? 0) !== (client[n] ?? 0));
}

function cap(name: StreamName, items: StreamItem[]): StreamItem[] {
  const keep = CAPS[name];
  return keep != null && items.length > keep ? items.slice(items.length - keep) : items;
}
