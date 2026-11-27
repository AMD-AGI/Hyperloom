// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Page state fed by /api/snapshot and the /api/stream SSE feed. Messages are queued as they arrive and applied once
// per animation frame, so a burst (a replay at x600, a reconnect) costs one render. Views are applied last-wins;
// stream items go through the cursor logic in streams.ts.

import { fetchArtifacts, fetchSnapshot, postReplay, type Fetcher, type ReplayControl } from "./api";
import { sessionNow } from "./clock";
import { claim, emptyItems, fromSnapshot, merge, outOfSync, type StreamItems, type StreamState } from "./streams";
import type { Artifact, Cursors, Snapshot, View } from "./types";

export type Connection = "connecting" | "connected" | "reconnecting" | "closed";

/** The part of EventSource the store uses, so tests can drive it. */
export interface EventSourceLike {
  addEventListener(type: string, listener: (e: { data: string }) => void): void;
  close(): void;
  onerror: ((e: Event) => void) | null;
}

export interface StoreDeps {
  fetcher?: Fetcher;
  openStream?: (url: string) => EventSourceLike;
  frame?: (cb: () => void) => void;
  wall?: () => number;
}

type Message =
  | { kind: "hello"; released: Cursors }
  | { kind: "items"; starts: Cursors; items: Partial<StreamItems> }
  | { kind: "view"; view: View };

export class InspectorStore {
  view = $state.raw<View | null>(null);
  streams = $state.raw<StreamItems>(emptyItems());
  artifacts = $state.raw<Artifact[]>([]);
  connection = $state<Connection>("connecting");
  error = $state("");
  /** Wall time (epoch seconds) when `view.clock` arrived; see `now()`. */
  clockAt = $state(0);

  #fetch: Fetcher;
  #open: (url: string) => EventSourceLike;
  #frame: (cb: () => void) => void;
  #wall: () => number;
  #state: StreamState = fromSnapshot({}, {});
  #conn: Cursors = {};
  #queue: Message[] = [];
  #scheduled = false;
  #syncing: Promise<void> | null = null;
  #es: EventSourceLike | null = null;

  constructor(deps: StoreDeps = {}) {
    this.#fetch = deps.fetcher ?? ((...a) => fetch(...a));
    this.#open = deps.openStream ?? ((url) => new EventSource(url));
    this.#frame = deps.frame ?? ((cb) => requestAnimationFrame(() => cb()));
    this.#wall = deps.wall ?? (() => Date.now() / 1000);
  }

  /** Session time now: wall time live, the extrapolated virtual clock in replay. */
  now(): number {
    return sessionNow(this.view?.clock, this.clockAt, this.#wall());
  }

  async start(): Promise<void> {
    try {
      await this.#load();
      await this.refreshArtifacts();
      this.#connect();
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
      this.connection = "closed";
    }
  }

  stop(): void {
    this.#es?.close();
    this.#es = null;
    this.#queue = [];
    this.connection = "closed";
  }

  async refreshArtifacts(): Promise<void> {
    this.artifacts = await fetchArtifacts(this.#fetch);
  }

  async replay(ctl: ReplayControl): Promise<void> {
    const clock = await postReplay(ctl, this.#fetch);
    if (this.view) this.#setView({ ...this.view, clock });
  }

  async #load(): Promise<void> {
    const snap: Snapshot = await fetchSnapshot(this.#fetch);
    this.#state = fromSnapshot(snap.streams ?? {}, snap.released ?? {});
    this.streams = this.#state.items;
    const { streams: _, ...view } = snap;
    this.#setView(view);
    this.error = "";
  }

  #setView(view: View): void {
    this.view = view;
    this.clockAt = this.#wall();
  }

  #connect(): void {
    this.#es?.close();
    const es = this.#open("api/stream");
    this.#es = es;
    this.connection = "connecting";
    es.addEventListener("hello", (e) => {
      this.connection = "connected";
      this.#conn = { ...(JSON.parse(e.data).released ?? {}) };
      this.#push({ kind: "hello", released: { ...this.#conn } });
    });
    es.addEventListener("items", (e) => {
      const items = (JSON.parse(e.data).items ?? {}) as Partial<StreamItems>;
      this.#push({ kind: "items", starts: claim(this.#conn, items), items });
    });
    es.addEventListener("view", (e) => this.#push({ kind: "view", view: JSON.parse(e.data) as View }));
    // The replay moved back past the released items: start over from a fresh snapshot and connection.
    es.addEventListener("reset", () => {
      es.close();
      this.#queue = [];
      void this.start();
    });
    es.onerror = () => {
      if (this.#es === es) this.connection = "reconnecting";
    };
  }

  #push(m: Message): void {
    this.#queue.push(m);
    this.#schedule();
  }

  #schedule(): void {
    if (this.#scheduled || this.#syncing) return;
    this.#scheduled = true;
    this.#frame(() => this.#flush());
  }

  #flush(): void {
    this.#scheduled = false;
    const queue = this.#queue;
    this.#queue = [];
    let state = this.#state;
    let view: View | null = null;
    let stale = false;
    for (const m of queue) {
      if (m.kind === "view") view = m.view;
      else if (m.kind === "hello") stale ||= outOfSync(m.released, state.cursors);
      else {
        const r = merge(state, m.starts, m.items);
        if ("gap" in r) stale = true;
        else state = r.state;
      }
    }
    if (state !== this.#state) {
      this.#state = state;
      this.streams = state.items;
    }
    if (view) {
      const artifactsChanged = view.artifact_count !== this.view?.artifact_count;
      this.#setView(view);
      if (artifactsChanged) void this.refreshArtifacts().catch(() => {});
    }
    if (stale) this.#resync();
  }

  /** Refetch the snapshot; messages that arrive meanwhile wait in the queue and are merged against it. */
  #resync(): void {
    if (this.#syncing) return;
    this.#syncing = this.#load()
      .catch((e: unknown) => {
        this.error = e instanceof Error ? e.message : String(e);
      })
      .finally(() => {
        this.#syncing = null;
        if (this.#queue.length) this.#schedule();
      });
  }
}
