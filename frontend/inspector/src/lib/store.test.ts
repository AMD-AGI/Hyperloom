// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { InspectorStore, type EventSourceLike } from "./store.svelte";
import type { StreamItem } from "./types";

class FakeSource implements EventSourceLike {
  listeners = new Map<string, (e: { data: string }) => void>();
  closed = false;
  onerror: ((e: Event) => void) | null = null;
  addEventListener(type: string, fn: (e: { data: string }) => void) {
    this.listeners.set(type, fn);
  }
  close() {
    this.closed = true;
  }
  emit(type: string, data: unknown) {
    this.listeners.get(type)!({ data: JSON.stringify(data) });
  }
}

const items = (...ts: number[]): StreamItem[] => ts.map((t) => ({ _t: t }));
const tick = () => new Promise((r) => setTimeout(r, 0));

function view(extra: Record<string, unknown> = {}) {
  return {
    version: 1,
    clock: { mode: "live", now: 0 },
    run: { session_id: "s1" },
    summary: { phase: "PRELUDE" },
    tasks: [],
    artifact_count: 1,
    released: { event: 0 },
    ...extra,
  };
}

/** A server with a mutable snapshot; counts requests per path. */
function harness() {
  const server = {
    snapshot: { ...view(), streams: {} } as Record<string, unknown>,
    hits: {} as Record<string, number>,
  };
  const fetcher = (async (url: string) => {
    const path = url.split("?")[0];
    server.hits[path] = (server.hits[path] ?? 0) + 1;
    const body = path === "api/artifacts" ? { artifacts: [{ path: "manifest.json" }] } : server.snapshot;
    return new Response(JSON.stringify(body), { status: 200 });
  }) as typeof fetch;
  const sources: FakeSource[] = [];
  const frames: (() => void)[] = [];
  const store = new InspectorStore({
    fetcher,
    openStream: () => {
      sources.push(new FakeSource());
      return sources.at(-1)!;
    },
    frame: (cb) => frames.push(cb),
    wall: () => 1000,
  });
  const flush = () => frames.splice(0).forEach((cb) => cb());
  return { server, store, sources, frames, flush };
}

describe("InspectorStore", () => {
  it("loads the snapshot and artifacts, then connects", async () => {
    const h = harness();
    h.server.snapshot = { ...view({ released: { event: 2 } }), streams: { event: items(1, 2) } };
    await h.store.start();
    expect(h.store.view?.summary.phase).toBe("PRELUDE");
    expect(h.store.streams.event.length).toBe(2);
    expect(h.store.artifacts.length).toBe(1);
    expect(h.store.connection).toBe("connecting");
    h.sources[0].emit("hello", { released: { event: 2 } });
    expect(h.store.connection).toBe("connected");
  });

  it("applies a burst of messages in one frame, last view wins", async () => {
    const h = harness();
    await h.store.start();
    const es = h.sources[0];
    es.emit("hello", { released: { event: 0 } });
    es.emit("items", { items: { event: items(1, 2) } });
    es.emit("view", view({ version: 2, summary: { phase: "SEARCH" } }));
    es.emit("items", { items: { event: items(3) } });
    es.emit("view", view({ version: 3, summary: { phase: "CLOSE" } }));
    expect(h.frames.length).toBe(1);
    expect(h.store.streams.event.length).toBe(0);
    h.flush();
    expect(h.store.streams.event.map((i) => i._t)).toEqual([1, 2, 3]);
    expect(h.store.view?.summary.phase).toBe("CLOSE");
    expect(h.server.hits["api/snapshot"]).toBe(1);
  });

  it("refetches the snapshot when hello disagrees, merging items that arrive meanwhile", async () => {
    const h = harness();
    await h.store.start();
    const es = h.sources[0];
    // Two events were released between the snapshot and the connection.
    h.server.snapshot = { ...view({ released: { event: 2 } }), streams: { event: items(1, 2) } };
    es.emit("hello", { released: { event: 2 } });
    h.flush();
    es.emit("items", { items: { event: items(3) } });
    expect(h.frames.length).toBe(0); // held until the snapshot is back
    await tick();
    h.flush();
    expect(h.store.streams.event.map((i) => i._t)).toEqual([1, 2, 3]);
    expect(h.server.hits["api/snapshot"]).toBe(2);
  });

  it("refetches again when a batch starts past a stale snapshot", async () => {
    const h = harness();
    await h.store.start();
    const es = h.sources[0];
    // The resync's snapshot lags the connection: it holds 1 event while the stream is already at 3.
    h.server.snapshot = { ...view({ released: { event: 1 } }), streams: { event: items(1) } };
    es.emit("hello", { released: { event: 3 } });
    h.flush();
    es.emit("items", { items: { event: items(4) } });
    await tick();
    h.server.snapshot = { ...view({ released: { event: 4 } }), streams: { event: items(1, 2, 3, 4) } };
    h.flush(); // the batch starts at 3, past the cursor 1: a gap
    await tick();
    expect(h.server.hits["api/snapshot"]).toBe(3);
    expect(h.store.streams.event.map((i) => i._t)).toEqual([1, 2, 3, 4]);
  });

  it("starts over on reset", async () => {
    const h = harness();
    await h.store.start();
    h.sources[0].emit("reset", { reset: true });
    await tick();
    await tick();
    expect(h.sources[0].closed).toBe(true);
    expect(h.sources.length).toBe(2);
    expect(h.server.hits["api/snapshot"]).toBe(2);
  });

  it("marks the connection reconnecting on an error", async () => {
    const h = harness();
    await h.store.start();
    h.sources[0].onerror!(new Event("error"));
    expect(h.store.connection).toBe("reconnecting");
  });

  it("reloads artifacts when the view's count changes", async () => {
    const h = harness();
    await h.store.start();
    h.sources[0].emit("view", view({ artifact_count: 2 }));
    h.flush();
    await tick();
    expect(h.server.hits["api/artifacts"]).toBe(2);
  });

  it("reports a failed start", async () => {
    const store = new InspectorStore({
      fetcher: (async () => new Response("{}", { status: 500 })) as typeof fetch,
      openStream: () => new FakeSource(),
      frame: (cb) => cb(),
    });
    await store.start();
    expect(store.error).toBe("HTTP 500");
    expect(store.connection).toBe("closed");
  });
});
