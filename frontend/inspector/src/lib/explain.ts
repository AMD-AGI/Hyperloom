// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// "Explain this": the reference the server resolves for a picked item, a short display name for it, and the fixed
// question per kind. Pure, so the drawer and the context menus share it.

import type { ItemRef, StreamItem, StreamName, View } from "./types";
import type { Selection } from "./ui.svelte";

export type { ItemRef };

/** What the labels and questions are looked up in; the InspectorStore fits. */
export interface Records {
  view: View | null;
  streams: Partial<Record<StreamName, StreamItem[]>>;
}

/** The server refuses longer strings in a reference, so such an item gets no reference. */
const MAX_FIELD = 200;
/** Journal changes are clipped to this in labels; the question keeps the whole change. */
const LABEL_CHARS = 60;

const field = (v: unknown): string | null =>
  typeof v === "string" && v.length > 0 && v.length <= MAX_FIELD ? v : null;
const clip = (s: string, n = LABEL_CHARS) => (s.length > n ? s.slice(0, n - 1) + "…" : s);

function num(v: unknown): number | null {
  const n = typeof v === "string" && v.trim() ? Number(v) : v;
  return typeof n === "number" && Number.isFinite(n) ? n : null;
}

/** The reference for a drawer selection; `raw` records only when they are journal entries. */
export function refFor(sel: Selection): ItemRef | null {
  switch (sel.kind) {
    case "node":
    case "task":
      return field(sel.id) ? { kind: sel.kind, id: sel.id } : null;
    case "event": {
      const seq = num(sel.item.seq);
      return seq == null ? null : { kind: "event", seq };
    }
    case "call":
      return callRef(sel.item);
    case "raw":
      return refForJournal(sel.data);
  }
}

/** An LLM call by `call_id` (orchestration, critic), else by specialist task and turn, as /api/conversation. */
function callRef(c: StreamItem): ItemRef | null {
  const callId = field(c.call_id);
  if (callId) return { kind: "call", call_id: callId };
  const taskId = field(c.task_id);
  if (!taskId) return null;
  const turn = num(c.turn);
  return turn == null ? { kind: "call", task_id: taskId } : { kind: "call", task_id: taskId, turn };
}

/** A journal entry is named by its task, change and time; anything else gets null. */
export function refForJournal(entry: unknown): ItemRef | null {
  if (!entry || typeof entry !== "object") return null;
  const j = entry as Record<string, unknown>;
  const [taskId, change, ts] = [field(j.task_id), field(j.change), field(j.ts)];
  return taskId && change && ts ? { kind: "journal", task_id: taskId, change, ts } : null;
}

/** A measured variant (a `variantUnits` row): its task's grid entry by label. */
export function refForVariant(unit: { task_id: string; label: string }): ItemRef | null {
  const [taskId, label] = [field(unit.task_id), field(unit.label)];
  return taskId && label ? { kind: "variant", task_id: taskId, label } : null;
}

// ---- lookups -----------------------------------------------------------------------------------------------

function findEvent(rec: Records, seq: number): StreamItem | undefined {
  return rec.streams.event?.find((e) => num(e.seq) === seq);
}

function findCall(rec: Records, ref: Extract<ItemRef, { kind: "call" }>): StreamItem | undefined {
  return rec.streams.llm_call?.find((c) =>
    ref.call_id
      ? c.call_id === ref.call_id
      : !c.call_id && c.task_id === ref.task_id && (ref.turn == null || num(c.turn) === ref.turn),
  );
}

function findJournal(rec: Records, ref: Extract<ItemRef, { kind: "journal" }>): StreamItem | undefined {
  return rec.streams.journal?.find((j) => j.task_id === ref.task_id && j.change === ref.change && j.ts === ref.ts);
}

const text = (v: unknown, fallback: string) => (v == null || v === "" ? fallback : String(v));
const component = (c: StreamItem | undefined) => text(c?.component ?? c?.role, "unknown");

// ---- labels and questions ----------------------------------------------------------------------------------

/** A short display name for the item: the thread's header and the follow-up chip. */
export function labelFor(ref: ItemRef, rec: Records): string {
  switch (ref.kind) {
    case "node":
      return rec.view?.graph.nodes.find((n) => n.id === ref.id)?.label || ref.id;
    case "task": {
      const t = rec.view?.tasks.find((x) => x.task_id === ref.id);
      return t?.label || t?.kind || ref.id;
    }
    case "event": {
      const e = findEvent(rec, ref.seq);
      return e
        ? `${text(e.topic, "message")} ${text(e.from_agent, "?")}→${text(e.to_agent, "?")}`
        : `event #${ref.seq}`;
    }
    case "call": {
      const c = findCall(rec, ref);
      const turn = ref.turn ?? num(c?.turn);
      return `${c ? component(c) : "LLM"} call` + (turn != null ? ` turn ${turn}` : "");
    }
    case "journal":
      return clip(ref.change);
    case "variant":
      return ref.label;
  }
}

/** "reverted" for REVERT, "dropped" for DROP, else the outcome in lower case ("recorded" when it is unknown). */
function verb(outcome: unknown): string {
  const o = text(outcome, "recorded").toLowerCase();
  return o === "revert" ? "reverted" : o === "drop" ? "dropped" : o;
}

/** The fixed question per kind (D6), worded as in the plan's contract. */
export function questionFor(ref: ItemRef, rec: Records): string {
  switch (ref.kind) {
    case "node":
      return (
        `Explain ${labelFor(ref, rec)}: ` +
        "what does this agent do, what has it done so far, and is anything wrong with it?"
      );
    case "task":
      return `Explain task ${labelFor(ref, rec)}: what it did, how it went and why.`;
    case "event": {
      const e = findEvent(rec, ref.seq);
      const [topic, from, to] = [text(e?.topic, "bus"), text(e?.from_agent, "unknown"), text(e?.to_agent, "unknown")];
      return `Explain this ${topic} message from ${from} to ${to}: what does it say and what came of it?`;
    }
    case "call":
      return (
        `Explain this ${component(findCall(rec, ref))} LLM call: ` +
        "what was it asked, what did it decide, and what came of it?"
      );
    case "journal": {
      const outcome = findJournal(rec, ref)?.outcome;
      const head = `Explain the change ${ref.change}: why was it`;
      if (/^keep$/i.test(String(outcome))) return `${head} kept, and how much did it improve on the best so far?`;
      return `${head} ${verb(outcome)}, and how did it compare with the best so far?`;
    }
    case "variant":
      return (
        `Explain the candidate ${ref.label}: ` +
        "what does this recipe change, where did it come from, and how did it do against the best so far?"
      );
  }
}
