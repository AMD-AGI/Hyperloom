// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { labelFor, questionFor, refFor, refForJournal, refForVariant, type Records } from "./explain";
import type { GraphNode, StreamItem, Task, View } from "./types";

const item = (fields: Record<string, unknown>) => ({ _t: 1, ...fields }) as StreamItem;
const journal = (outcome: string, change = "--enable-expert-parallel") =>
  item({ task_id: "t1", change, ts: "2026-10-20T10:00:00Z", outcome, phase: "EXPLORE" });

const rec: Records = {
  view: {
    graph: { nodes: [{ id: "spec:roofline", label: "roofline" } as GraphNode], edges: [] },
    tasks: [
      { task_id: "t1", kind: "explore", label: "explore" } as Task,
      { task_id: "t2", kind: "specialist", label: "" } as Task,
    ],
  } as unknown as View,
  streams: {
    event: [item({ seq: 7, topic: "delegated_result", from_agent: "explore", to_agent: "coordinator" })],
    llm_call: [item({ call_id: "c-1", component: "critic" }), item({ task_id: "t2", turn: 3, role: "specialist" })],
    journal: [journal("KEEP", "MTP n=1"), journal("REVERT"), journal("DROP", "--dflash"), journal("INFO", "note")],
  },
};

describe("refFor", () => {
  it("maps node, task, event and call selections", () => {
    expect(refFor({ kind: "node", id: "spec:roofline" })).toEqual({ kind: "node", id: "spec:roofline" });
    expect(refFor({ kind: "task", id: "t1" })).toEqual({ kind: "task", id: "t1" });
    expect(refFor({ kind: "event", item: item({ seq: 7 }) })).toEqual({ kind: "event", seq: 7 });
    expect(refFor({ kind: "call", item: item({ call_id: "c-1", task_id: "t2" }) })).toEqual({
      kind: "call",
      call_id: "c-1",
    });
    expect(refFor({ kind: "call", item: item({ task_id: "t2", turn: 3 }) })).toEqual({
      kind: "call",
      task_id: "t2",
      turn: 3,
    });
    expect(refFor({ kind: "call", item: item({ task_id: "t2" }) })).toEqual({ kind: "call", task_id: "t2" });
  });
  it("maps a raw journal entry and nothing else that is raw", () => {
    expect(refFor({ kind: "raw", title: "Journal entry", data: journal("KEEP") })).toEqual({
      kind: "journal",
      task_id: "t1",
      change: "--enable-expert-parallel",
      ts: "2026-10-20T10:00:00Z",
    });
    expect(refFor({ kind: "raw", title: "Unit v1", data: { unit: "variant", label: "v1" } })).toBeNull();
    expect(refFor({ kind: "raw", title: "x", data: "text" })).toBeNull();
    expect(refFor({ kind: "raw", title: "x", data: null })).toBeNull();
  });
  it("gives no reference when the identity is missing or too long for the server", () => {
    expect(refFor({ kind: "event", item: item({ topic: "x" }) })).toBeNull();
    expect(refFor({ kind: "call", item: item({ component: "critic" }) })).toBeNull();
    expect(refFor({ kind: "task", id: "x".repeat(201) })).toBeNull();
    expect(refForJournal({ task_id: "t1", change: "", ts: "2026" })).toBeNull();
    expect(refForJournal({ task_id: 5, change: "c", ts: "2026" })).toBeNull();
  });
  it("maps a variant row", () => {
    expect(refForVariant({ task_id: "t1", label: "aiter_moe_on" })).toEqual({
      kind: "variant",
      task_id: "t1",
      label: "aiter_moe_on",
    });
    expect(refForVariant({ task_id: "t1", label: "" })).toBeNull();
  });
});

describe("labelFor", () => {
  it("names each kind briefly", () => {
    expect(labelFor({ kind: "node", id: "spec:roofline" }, rec)).toBe("roofline");
    expect(labelFor({ kind: "node", id: "gone" }, rec)).toBe("gone");
    expect(labelFor({ kind: "task", id: "t1" }, rec)).toBe("explore");
    expect(labelFor({ kind: "task", id: "t2" }, rec)).toBe("specialist");
    expect(labelFor({ kind: "event", seq: 7 }, rec)).toBe("delegated_result explore→coordinator");
    expect(labelFor({ kind: "event", seq: 99 }, rec)).toBe("event #99");
    expect(labelFor({ kind: "call", call_id: "c-1" }, rec)).toBe("critic call");
    expect(labelFor({ kind: "call", task_id: "t2", turn: 3 }, rec)).toBe("specialist call turn 3");
    expect(labelFor({ kind: "journal", task_id: "t1", change: "x".repeat(80), ts: "t" }, rec)).toBe(
      "x".repeat(59) + "…",
    );
    expect(labelFor({ kind: "variant", task_id: "t1", label: "aiter_moe_on" }, rec)).toBe("aiter_moe_on");
  });
});

describe("questionFor", () => {
  const j = (e: StreamItem) => refForJournal(e)!;
  it("asks the fixed question per kind", () => {
    expect(questionFor({ kind: "node", id: "spec:roofline" }, rec)).toBe(
      "Explain roofline: what does this agent do, what has it done so far, and is anything wrong with it?",
    );
    expect(questionFor({ kind: "task", id: "t1" }, rec)).toBe(
      "Explain task explore: what it did, how it went and why.",
    );
    expect(questionFor({ kind: "event", seq: 7 }, rec)).toBe(
      "Explain this delegated_result message from explore to coordinator: what does it say and what came of it?",
    );
    expect(questionFor({ kind: "call", call_id: "c-1" }, rec)).toBe(
      "Explain this critic LLM call: what was it asked, what did it decide, and what came of it?",
    );
    expect(questionFor({ kind: "call", task_id: "t2", turn: 3 }, rec)).toBe(
      "Explain this specialist LLM call: what was it asked, what did it decide, and what came of it?",
    );
    expect(questionFor({ kind: "variant", task_id: "t1", label: "dflash" }, rec)).toBe(
      "Explain the candidate dflash: what does this recipe change, where did it come from, " +
        "and how did it do against the best so far?",
    );
  });
  it("words a journal question by its outcome", () => {
    const [keep, revert, drop, info] = rec.streams.journal!;
    expect(questionFor(j(keep), rec)).toBe(
      "Explain the change MTP n=1: why was it kept, and how much did it improve on the best so far?",
    );
    expect(questionFor(j(revert), rec)).toBe(
      "Explain the change --enable-expert-parallel: why was it reverted, and how did it compare with the best so far?",
    );
    expect(questionFor(j(drop), rec)).toBe(
      "Explain the change --dflash: why was it dropped, and how did it compare with the best so far?",
    );
    expect(questionFor(j(info), rec)).toBe(
      "Explain the change note: why was it info, and how did it compare with the best so far?",
    );
  });
});
