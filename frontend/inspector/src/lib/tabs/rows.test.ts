// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { Artifact, GraphNode, StreamItem, Task } from "../types";
import {
  busRows,
  callRows,
  fileName,
  filterArtifacts,
  groupArtifacts,
  journalRows,
  jsonlKeys,
  MAX_ROWS,
  outcomeClass,
  roundNode,
  roundRows,
  stepText,
  taskRows,
  variantKey,
  variantThroughputs,
  variantUnits,
} from "./rows";

const task = (task_id: string, node: string, created_at: string, extra: Partial<Task> = {}): Task => ({
  task_id,
  node,
  created_at,
  kind: "k",
  state: "done",
  health: "ok",
  label: node,
  ...extra,
});
const node = (id: string, label = id): GraphNode => ({
  id,
  label,
  group: "core",
  layer: 0,
  tasks: {},
  llm_calls: 0,
  llm_tokens: 0,
  artifacts: 0,
  trees: {},
  health: "ok",
});
const item = (x: Record<string, unknown>): StreamItem => ({ _t: 0, ...x });
const art = (
  path: string,
  producer: string,
  task_id: string | null = null,
  extra: Partial<Artifact> = {},
): Artifact => ({
  path,
  producer,
  task_id,
  size: 10,
  mtime: 0,
  type: "log",
  ...extra,
});

describe("rows", () => {
  const tasks = [task("a", "exec:baseline", "2026-01-01"), task("b", "spec:static_recon", "2026-01-02")];

  it("orders tasks newest first and filters them to a node", () => {
    expect(taskRows(tasks, null).map((t) => t.task_id)).toEqual(["b", "a"]);
    expect(taskRows(tasks, "exec:baseline").map((t) => t.task_id)).toEqual(["a"]);
  });

  it("maps specialist rounds to their task's node, else to the domain node", () => {
    expect(roundNode({ task_id: "b", domain: "x" }, tasks)).toBe("spec:static_recon");
    expect(roundNode({ task_id: "zz", domain: "research_scout_specialist" }, tasks)).toBe("spec:research_scout");
    expect(roundRows([{ task_id: "b" }, { domain: "freeform" }], tasks, "spec:freeform")).toEqual([
      { domain: "freeform" },
    ]);
  });

  it("filters bus events on either end and caps them newest first", () => {
    const nodes = [node("coordinator"), node("critic"), node("orchestration")];
    const ev = [
      item({ seq: 1, from_agent: "orchestration", to_agent: "critic" }),
      item({ seq: 2, from_agent: "critic", to_agent: "orchestration" }),
      item({ seq: 3, from_agent: "coordinator", to_agent: "orchestration" }),
    ];
    expect(busRows(ev, nodes, "critic").map((e) => e.seq)).toEqual([2, 1]);
    const many = Array.from({ length: MAX_ROWS + 5 }, (_, i) => item({ seq: i }));
    const rows = busRows(many, nodes, null);
    expect(rows.length).toBe(MAX_ROWS);
    expect(rows[0].seq).toBe(MAX_ROWS + 4);
  });

  it("puts specialist calls on their task's node", () => {
    const calls = [item({ component: "specialist", task_id: "b" }), item({ component: "orchestration" })];
    expect(callRows(calls, tasks, "spec:static_recon")).toEqual([calls[0]]);
    expect(callRows(calls, tasks, "orchestration")).toEqual([calls[1]]);
  });

  it("filters journal entries through their task", () => {
    const j = [item({ task_id: "a" }), item({ task_id: null }), item({ task_id: "b" })];
    expect(journalRows(j, tasks, null)).toHaveLength(3);
    expect(journalRows(j, tasks, "exec:baseline")).toEqual([j[0]]);
  });

  it("takes each succeeded variant's measured tok/s from the task's progress units", () => {
    const explore = task("e", "exec:explore", "x", {
      progress: {
        units: [
          { unit: "variant", label: "v1", status: "succeeded", output_throughput: 100, ts: 1 },
          { unit: "variant", label: "v2", status: "failed", output_throughput: null, ts: 2 },
          { unit: "variant", label: "v3", status: "succeeded", output_throughput: null, ts: 3 },
          { unit: "baseline", label: "v4", status: "succeeded", output_throughput: 400, ts: 4 },
          { unit: "variant", label: "v1", status: "succeeded", output_throughput: 150, ts: 5 },
        ],
      },
    });
    const tputs = variantThroughputs([explore]);
    expect([...tputs.keys()]).toEqual(["e/v1"]);
    expect(tputs.get(variantKey(item({ task_id: "e", variant_name: "v1" })))).toBe(150);
    expect(variantThroughputs([])).toEqual(new Map());
  });

  it("lists measured variants newest first", () => {
    const t = task("c", "exec:explore", "x", {
      progress: {
        units: [
          { unit: "variant", label: "v1", status: "succeeded", ts: 1 },
          { unit: "baseline", label: "b", status: "succeeded", ts: 3 },
          { unit: "variant", label: "v2", status: "failed", ts: 2 },
        ],
      },
    });
    expect(variantUnits([t], null).map((u) => [u.label, u.key])).toEqual([
      ["v2", "c/v2#1"],
      ["v1", "c/v1#1"],
    ]);
    expect(variantUnits([t], "exec:baseline")).toEqual([]);
  });

  it("formats outcomes, steps and JSONL keys", () => {
    expect([outcomeClass("KEEP"), outcomeClass("revert"), outcomeClass(null)]).toEqual(["h-ok", "h-failed", ""]);
    expect(stepText({ label: "v3", status: "benchmark", index: 2, total: 5 })).toBe("v3 benchmark (2/5)");
    expect(stepText(null)).toBe("");
    expect(
      jsonlKeys(
        [
          { a: 1, b: 2 },
          { c: 3, a: 4 },
        ],
        2,
      ),
    ).toEqual(["a", "b"]);
  });
});

describe("artifact tree", () => {
  const list = [
    art("state.json", "session", null, { type: "json" }),
    art("runs/explore/t1/process.log", "exec:explore", "t1"),
    art("runs/explore/t1/worktree/a.py", "exec:explore", "t1", { bulk: "runs/explore/t1/worktree", size: 5 }),
    art("runs/explore/t1/worktree/b.py", "exec:explore", "t1", { bulk: "runs/explore/t1/worktree", size: 7 }),
    art("agents/critic/system_prompt.md", "critic", null, { type: "markdown" }),
  ];

  it("groups by producer and task, collapsing large subtrees", () => {
    const g = groupArtifacts(list);
    expect(g.map((p) => [p.producer, p.count])).toEqual([
      ["critic", 1],
      ["exec:explore", 3],
      ["session", 1],
    ]);
    const t1 = g[1].tasks[0];
    expect(t1.task).toBe("t1");
    expect(t1.files.map(fileName)).toEqual(["process.log"]);
    expect(t1.trees).toEqual([
      { root: "runs/explore/t1/worktree", name: "worktree", files: [list[2], list[3]], bytes: 12 },
    ]);
    expect(g[0].loose.files).toEqual([list[4]]);
  });

  it("narrows by search, type and node", () => {
    expect(filterArtifacts(list, { q: "PROCESS", type: "", filter: null })).toEqual([list[1]]);
    expect(filterArtifacts(list, { q: "", type: "json", filter: null })).toEqual([list[0]]);
    expect(filterArtifacts(list, { q: "", type: "", filter: "critic" })).toEqual([list[4]]);
  });
});
