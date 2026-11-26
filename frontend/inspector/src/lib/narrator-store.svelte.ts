// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// State for the narrator panel: the server's status and entries, the call in flight, and the viewer's choices (panel
// open, auto narration, pinned focus), which persist in localStorage when it is there. One store serves the page:
// the drawer and the views explain items through it, and the answers land in the narrator's thread.

import { ApiError, fetchNarrator, postAsk, postNarrate, type Fetcher } from "./api";
import { current } from "./narrator";
import type { ItemRef, NarratorEntry, NarratorStatus } from "./types";

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
/** An item being explained or followed up on, with its display name. */
export interface Explained {
  ref: ItemRef;
  label: string;
}

export class NarratorStore {
  status = $state.raw<NarratorStatus | null>(null);
  pending = $state<Pending>(null);
  error = $state("");
  /** A question sent and not answered yet, shown in the thread meanwhile. */
  asking = $state("");
  open = $state(true);
  auto = $state(true);
  focus = $state("");
  /** The item whose explanation is pending. */
  explaining = $state.raw<Explained | null>(null);
  /** The item last explained: typed questions carry it until the viewer removes it. */
  followUp = $state.raw<Explained | null>(null);

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

  /**
   * Ask a question; one sent while a narration is being written waits for it rather than being dropped. It is about
   * `item` when given, else about the follow-up item when there is one.
   */
  async ask(question: string, item?: ItemRef): Promise<boolean> {
    const q = question.trim().slice(0, 500);
    if (!q || this.asking || !this.available) return false;
    const ref = item ?? this.followUp?.ref;
    this.asking = q;
    const ok = await this.#question(q, ref);
    this.asking = "";
    return ok;
  }

  /**
   * Explain one item: the action behind the drawer's Explain button and the context menus. Opens the panel, asks the
   * fixed question about the item (after a call in flight, as `ask` does) and, once answered, keeps the item for
   * follow-up questions. A press on the item already pending is ignored; one on another item waits its turn and
   * shows as the pending one. The pinned focus is left alone (D5).
   */
  async explain(ref: ItemRef, question: string, label: string): Promise<boolean> {
    const q = question.trim().slice(0, 500);
    if (!q || !this.available || (this.explaining && sameItem(this.explaining.ref, ref))) return false;
    this.setOpen(true);
    const mine: Explained = { ref, label };
    this.explaining = mine;
    const ok = await this.#question(q, ref);
    if (ok) this.followUp = mine;
    if (this.explaining === mine) this.explaining = null;
    return ok;
  }

  clearFollowUp(): void {
    this.followUp = null;
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

  async #question(q: string, item: ItemRef | undefined): Promise<boolean> {
    while (this.#inflight) await this.#inflight;
    return this.#run("ask", () => postAsk(q, this.focus, item, this.#fetch), true);
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

/** Whether two references name the same item. */
export function sameItem(a: ItemRef, b: ItemRef): boolean {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  return [...keys].every((k) => (a as Record<string, unknown>)[k] === (b as Record<string, unknown>)[k]);
}

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
