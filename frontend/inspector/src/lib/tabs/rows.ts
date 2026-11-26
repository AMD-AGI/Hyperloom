// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Rows for the tabs: filtering to the selected node, ordering, and the artifact tree's grouping.

import { callLane, eventLane } from "../timeline/scale";
import type { Artifact, GraphNode, ProgressUnit, StreamItem, Task } from "../types";

/** Stream tabs show at most this many rows, newest first. */
export const MAX_ROWS = 1000;
/** Lists longer than this skip the FLIP reordering and the new-row highlight. */
export const ANIMATE_MAX = 200;
/** A collapsed large subtree builds at most this many rows when opened; the search box narrows the rest. */
export const BULK_ROWS = 300;

const newestFirst = <T>(rows: T[]): T[] => rows.slice(-MAX_ROWS).reverse();

export function taskRows(tasks: Task[], filter: string | null): Task[] {
  return tasks
    .filter((t) => !filter || t.node === filter)
    .sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));
}

export type Round = Record<string, unknown> & { task_id?: string; domain?: string };

/** The graph node of a specialist round: its task's node, else `spec:<domain>` without the `_specialist` suffix. */
export function roundNode(r: Round, tasks: Task[]): string {
  const t = r.task_id ? tasks.find((x) => x.task_id === r.task_id) : undefined;
  return t ? t.node : "spec:" + String(r.domain ?? "").replace(/_specialist$/, "");
}

/** A round's key, stable across view updates. */
export const roundKey = (r: Round): string => `${r.task_id ?? ""}|${r.domain ?? ""}|${r.completed_at ?? ""}`;

export function roundRows(rounds: Round[] | undefined, tasks: Task[], filter: string | null): Round[] {
  return (rounds ?? []).filter((r) => !filter || roundNode(r, tasks) === filter);
}

export function busRows(events: StreamItem[], nodes: GraphNode[], filter: string | null): StreamItem[] {
  const rows = filter
    ? events.filter((e) => eventLane(nodes, e.from_agent) === filter || eventLane(nodes, e.to_agent) === filter)
    : events;
  return newestFirst(rows);
}

export function callRows(calls: StreamItem[], tasks: Task[], filter: string | null): StreamItem[] {
  return newestFirst(filter ? calls.filter((c) => callLane(tasks, c) === filter) : calls);
}

export function journalRows(journal: StreamItem[], tasks: Task[], filter: string | null): StreamItem[] {
  if (!filter) return journal;
  const node = new Map(tasks.map((t) => [t.task_id, t.node]));
  return journal.filter((j) => j.task_id != null && node.get(String(j.task_id)) === filter);
}

/** Joins a journal entry (`variant_name`) and a measured variant (`variant`) to the same variant of a task. */
export const variantKey = (item: StreamItem): string => `${item.task_id}/${item.variant_name ?? item.variant ?? ""}`;

export type Unit = ProgressUnit & { task_id: string; key: string };

/** Measured variants across the (filtered) tasks, newest first. Each gets a key that is stable across view updates
 * (which rebuild the objects) and unique even when a label repeats. */
export function variantUnits(tasks: Task[], filter: string | null): Unit[] {
  const seen = new Map<string, number>();
  return tasks
    .filter((t) => !filter || t.node === filter)
    .flatMap((t) => {
      const units = (t.progress?.units ?? []) as ProgressUnit[];
      return units
        .filter((u) => u.unit === "variant")
        .map((u) => {
          const base = `${t.task_id}/${u.label}`;
          const n = (seen.get(base) ?? 0) + 1;
          seen.set(base, n);
          return { ...u, task_id: t.task_id, key: `${base}#${n}` };
        });
    })
    .sort((a, b) => (b.ts ?? 0) - (a.ts ?? 0));
}

/** Measured tok/s of each succeeded variant, by `variantKey`, newest measurement first. The journal leaves
 * `throughput_after` empty for explore variants, but the task's progress units have what each one measured. */
export function variantThroughputs(tasks: Task[]): Map<string, number> {
  const tputs = new Map<string, number>();
  for (const u of variantUnits(tasks, null)) {
    const key = variantKey({ _t: 0, task_id: u.task_id, variant: u.label });
    if (u.status === "succeeded" && u.output_throughput != null && !tputs.has(key)) tputs.set(key, u.output_throughput);
  }
  return tputs;
}

export const outcomeClass = (outcome: unknown): string =>
  /^keep$/i.test(String(outcome)) ? "h-ok" : /^(drop|revert)$/i.test(String(outcome)) ? "h-failed" : "";

export function stepText(c: Record<string, unknown> | null | undefined): string {
  if (!c) return "";
  const of = c.total ? ` (${c.index}/${c.total})` : "";
  return `${c.label || c.unit} ${c.status || ""}${of}`;
}

/** Up to `n` keys from the first rows of a JSONL page, in first-seen order. */
export function jsonlKeys(rows: Record<string, unknown>[], n = 8): string[] {
  const keys: string[] = [];
  for (const row of rows.slice(0, 50))
    for (const k of Object.keys(row)) if (!keys.includes(k) && keys.length < n) keys.push(k);
  return keys;
}

// ---- artifact tree -----------------------------------------------------------------------------------------

export interface BulkTree {
  root: string;
  name: string;
  files: Artifact[];
  bytes: number;
}
export interface TaskGroup {
  task: string;
  files: Artifact[];
  trees: BulkTree[];
}
export interface ProducerGroup {
  producer: string;
  count: number;
  loose: TaskGroup;
  tasks: TaskGroup[];
}
export interface TreeQuery {
  q: string;
  type: string;
  filter: string | null;
}

export function filterArtifacts(list: Artifact[], { q, type, filter }: TreeQuery): Artifact[] {
  const needle = q.toLowerCase();
  return list.filter(
    (a) =>
      (!needle || a.path.toLowerCase().includes(needle)) &&
      (!type || a.type === type) &&
      (!filter || a.producer === filter),
  );
}

function taskGroup(task: string, files: Artifact[]): TaskGroup {
  const plain: Artifact[] = [];
  const bulk = new Map<string, Artifact[]>();
  for (const a of files) {
    if (a.bulk) bulk.set(a.bulk, [...(bulk.get(a.bulk) ?? []), a]);
    else plain.push(a);
  }
  const trees = [...bulk.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([root, list]) => ({
      root,
      name: root.split("/").pop() ?? root,
      files: list,
      bytes: list.reduce((x, a) => x + (a.size || 0), 0),
    }));
  return { task, files: plain, trees };
}

/** Producer → task → files, with files under a large subtree collapsed into one tree per subtree. */
export function groupArtifacts(list: Artifact[]): ProducerGroup[] {
  const by = new Map<string, Map<string, Artifact[]>>();
  for (const a of list) {
    const prod = a.producer || "session";
    const tasks = by.get(prod) ?? new Map<string, Artifact[]>();
    by.set(prod, tasks);
    const key = a.task_id || "";
    tasks.set(key, [...(tasks.get(key) ?? []), a]);
  }
  return [...by.keys()].sort().map((producer) => {
    const tasks = by.get(producer)!;
    const keys = [...tasks.keys()].filter(Boolean).sort();
    return {
      producer,
      count: [...tasks.values()].reduce((n, l) => n + l.length, 0),
      loose: taskGroup("", tasks.get("") ?? []),
      tasks: keys.map((k) => taskGroup(k, tasks.get(k)!)),
    };
  });
}

/** A file's name inside its task directory (`runs/<kind>/<task>/…`), or its whole path. */
export const fileName = (a: Artifact): string => (a.task_id ? a.path.split("/").slice(3).join("/") || a.path : a.path);

export const artifactTypes = (list: Artifact[]): string[] => [...new Set(list.map((a) => a.type))].sort();

/** A table column: its heading, the cell text, and a class (fixed or per row, e.g. a health colour). */
export interface Col<T> {
  head: string;
  text: (row: T) => unknown;
  cls?: string | ((row: T) => string);
}
