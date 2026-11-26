// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// State for the narrator panel: the server's status and entries, the call in flight, and the viewer's choices (panel
// open, auto narration, pinned focus), which persist in localStorage when it is there.

import { ApiError, fetchNarrator, postAsk, postNarrate, type Fetcher } from "./api";
import { current } from "./narrator";
import type { NarratorEntry, NarratorStatus } from "./types";

const KEY = "hyperloom-inspector.narrator";

interface Saved {
  open: boolean;
  auto: boolean;
  focus: string;
}

function load(): Saved {
  const saved: Saved = { open: true, auto: true, focus: "" };
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) ?? "{}") as Partial<Saved>;
    if (typeof raw.open === "boolean") saved.open = raw.open;
    if (typeof raw.auto === "boolean") saved.auto = raw.auto;
    if (typeof raw.focus === "string") saved.focus = raw.focus.slice(0, 300);
  } catch {
    // Storage is a convenience; the defaults hold.
  }
  return saved;
}

export type Pending = null | "narrate" | "ask";

export class NarratorStore {
  status = $state.raw<NarratorStatus | null>(null);
  pending = $state<Pending>(null);
  error = $state("");
  /** A question sent and not answered yet, shown in the thread meanwhile. */
  asking = $state("");
  open = $state(true);
  auto = $state(true);
  focus = $state("");

  #fetch: Fetcher;
  #inflight: Promise<boolean> | null = null;

  constructor(fetcher?: Fetcher) {
    this.#fetch = fetcher ?? ((...a) => fetch(...a));
    Object.assign(this, load());
  }

  get available(): boolean {
    return this.status?.available ?? false;
  }

  get latest(): { narration: NarratorEntry | null; answers: NarratorEntry[] } {
    return current(this.status?.entries ?? []);
  }

  async refresh(): Promise<void> {
    try {
      this.status = await fetchNarrator(this.#fetch);
    } catch (e) {
      this.error = message(e);
    }
  }

  /**
   * Ask for a narration. The server reuses the latest one when nothing changed, unless forced; a refusal because of
   * the minimum gap or a call in flight only shows as an error when the viewer asked for it. A forced narration (the
   * viewer's refresh, or a new focus) waits for a call in flight; an automatic one is skipped.
   */
  async narrate(force = false): Promise<void> {
    if (!this.available || (this.pending && !force)) return;
    while (this.#inflight) await this.#inflight;
    await this.#run("narrate", () => postNarrate(force, this.focus, this.#fetch), force);
  }

  /** Ask a question; one sent while a narration is being written waits for it rather than being dropped. */
  async ask(question: string): Promise<boolean> {
    const q = question.trim().slice(0, 500);
    if (!q || this.asking || !this.available) return false;
    this.asking = q;
    while (this.#inflight) await this.#inflight;
    const ok = await this.#run("ask", () => postAsk(q, this.focus, this.#fetch), true);
    this.asking = "";
    return ok;
  }

  setOpen(open: boolean): void {
    this.open = open;
    this.#save();
  }

  setAuto(auto: boolean): void {
    this.auto = auto;
    this.#save();
  }

  setFocus(focus: string): void {
    this.focus = focus.trim().slice(0, 300);
    this.#save();
  }

  #run(kind: Pending, call: () => Promise<unknown>, loud: boolean): Promise<boolean> {
    const run = this.#send(kind, call, loud).finally(() => (this.#inflight = null));
    this.#inflight = run;
    return run;
  }

  async #send(kind: Pending, call: () => Promise<unknown>, loud: boolean): Promise<boolean> {
    this.pending = kind;
    this.error = "";
    let ok = true;
    try {
      await call();
    } catch (e) {
      ok = false;
      if (loud || !(e instanceof ApiError && e.status === 429)) this.error = message(e);
    } finally {
      this.pending = null;
    }
    await this.refresh();
    return ok;
  }

  #save(): void {
    try {
      localStorage.setItem(KEY, JSON.stringify({ open: this.open, auto: this.auto, focus: this.focus }));
    } catch {
      // Storage is a convenience; the choice still holds for this page.
    }
  }
}

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
