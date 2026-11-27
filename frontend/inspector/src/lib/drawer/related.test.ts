// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { emptyItems } from "../streams";
import type { Artifact, StreamItem, Task, View } from "../types";
import { commonDir, eventRelated, nodeRelated, producerArtifacts, taskRelated } from "./related";

const task = (task_id: string, node: string) => ({ task_id, node, kind: "k", state: "s", health: "ok", label: node });
const view = {
  summary: { specialist_rounds: [{ task_id: "t1", domain: "static_recon_specialist" }, { task_id: "t2" }] },
  tasks: [task("t1", "spec:static_recon"), task("t2", "exec:baseline")] as Task[],
  graph: {
    nodes: [],
    edges: [
      { source: "coordinator", target: "exec:baseline", label: "dispatch", count: 1 },
      { source: "critic", target: "orchestration", label: "verdict", count: 2 },
    ],
  },
} as unknown as View;
const ev = (seq: number, extra: Record<string, unknown>): StreamItem => ({ _t: seq, seq, ...extra });

describe("drawer relations", () => {
  const streams = emptyItems();
  const proposal = ev(1, { msg_id: "m1", topic: "proposal" });
  const verdict = ev(2, { msg_id: "m2", in_reply_to: "m1", topic: "review_verdict" });
  const result = ev(3, { msg_id: "m3", topic: "delegated_result", payload: { task_id: "t1" } });
  streams.event = [proposal, verdict, result];
  streams.review = [{ _t: 2, target_proposal_msg_id: "m1", verdict: "approve" }];
  streams.journal = [
    { _t: 4, task_id: "t1" },
    { _t: 5, task_id: "t2" },
  ];
  streams.llm_call = [
    { _t: 6, task_id: "t1", turn: 1 },
    { _t: 7, call_id: "c1" },
  ];

  it("collects a task's events, round, journal entries and calls", () => {
    const r = taskRelated("t1", view, streams);
    expect(r.events).toEqual([result]);
    expect(r.rounds).toHaveLength(1);
    expect(r.journal).toEqual([streams.journal[0]]);
    expect(r.calls).toEqual([streams.llm_call[0]]);
  });

  it("links an event to its parent, replies, verdicts and task", () => {
    expect(eventRelated(proposal, view, streams)).toMatchObject({ parent: null, replies: [verdict], task: null });
    expect(eventRelated(proposal, view, streams).reviews).toHaveLength(1);
    expect(eventRelated(verdict, view, streams).parent).toBe(proposal);
    expect(eventRelated(result, view, streams).task?.task_id).toBe("t1");
    expect(eventRelated(ev(9, {}), view, streams)).toEqual({ parent: null, replies: [], reviews: [], task: null });
  });

  it("gives a node its traffic and tasks", () => {
    const r = nodeRelated("exec:baseline", view);
    expect(r.edges.map((e) => e.label)).toEqual(["dispatch"]);
    expect(r.tasks.map((t) => t.task_id)).toEqual(["t2"]);
  });

  it("matches a producer's artifacts by node, or a task's by id", () => {
    const art = (i: number, task_id: string | null): Artifact =>
      ({ path: `f${i}`, size: 1, mtime: 0, type: "log", producer: "spec:static_recon", task_id }) as Artifact;
    const list = Array.from({ length: 6 }, (_, i) => art(i, i % 2 ? "t1" : null));
    expect(producerArtifacts(list, { node: "spec:static_recon" })).toHaveLength(6);
    expect(producerArtifacts(list, { node: "critic" })).toEqual([]);
    expect(producerArtifacts(list, { task: "t1" }).every((a) => a.task_id === "t1")).toBe(true);
  });
});

describe("commonDir", () => {
  it("returns the shared directory, never a file name", () => {
    expect(commonDir(["runs/a/t1/x.log", "runs/a/t1/v0/y.json"])).toBe("runs/a/t1/");
    expect(commonDir(["runs/a/t1/x.log"])).toBe("runs/a/t1/");
    expect(commonDir(["state.json", "runs/a/x"])).toBe("");
    expect(commonDir([])).toBe("");
  });
});
