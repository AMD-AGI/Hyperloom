// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The records related to a drawer selection: a task's bus events, specialist round and journal entries; an event's
// parent, replies and critic verdicts; a node's traffic and tasks.

import type { StreamItems } from "../streams";
import type { Round } from "../tabs/rows";
import type { Artifact, GraphEdge, StreamItem, Task, View } from "../types";

/** A drawer lists at most this many artifacts; the Artifacts tab has the rest. */
export const DRAWER_ARTIFACTS = 200;

const payloadTask = (e: StreamItem): unknown => (e.payload as Record<string, unknown> | undefined)?.task_id;

export interface TaskRelated {
  events: StreamItem[];
  rounds: Round[];
  journal: StreamItem[];
  calls: StreamItem[];
}

export function taskRelated(id: string, view: View, streams: StreamItems): TaskRelated {
  const rounds = (view.summary.specialist_rounds ?? []) as Round[];
  return {
    events: streams.event.filter((e) => payloadTask(e) === id),
    rounds: rounds.filter((r) => r.task_id === id),
    journal: streams.journal.filter((j) => j.task_id === id),
    calls: streams.llm_call.filter((c) => c.task_id === id),
  };
}

export interface EventRelated {
  parent: StreamItem | null;
  replies: StreamItem[];
  reviews: StreamItem[];
  task: Task | null;
}

export function eventRelated(e: StreamItem, view: View, streams: StreamItems): EventRelated {
  const id = e.msg_id;
  const tid = payloadTask(e);
  return {
    parent: e.in_reply_to ? (streams.event.find((x) => x.msg_id === e.in_reply_to) ?? null) : null,
    replies: id ? streams.event.filter((x) => x.in_reply_to === id) : [],
    reviews: id ? streams.review.filter((r) => r.target_proposal_msg_id === id) : [],
    task: tid ? (view.tasks.find((t) => t.task_id === tid) ?? null) : null,
  };
}

export interface NodeRelated {
  edges: GraphEdge[];
  tasks: Task[];
}

export function nodeRelated(id: string, view: View): NodeRelated {
  return {
    edges: view.graph.edges.filter((e) => e.source === id || e.target === id),
    tasks: view.tasks.filter((t) => t.node === id),
  };
}

/** The artifacts a producer node or a task wrote, bulk subtrees included; the drawer shows the first few. */
export function producerArtifacts(list: Artifact[], by: { node?: string; task?: string }): Artifact[] {
  return list.filter((a) => (by.task ? a.task_id === by.task : a.producer === by.node));
}

/** The directory every path shares ("" when none), so a list can show each file relative to it. */
export function commonDir(paths: string[]): string {
  if (!paths.length) return "";
  let parts = paths[0].split("/").slice(0, -1);
  for (const p of paths.slice(1)) {
    const q = p.split("/");
    let i = 0;
    while (i < parts.length && i < q.length - 1 && parts[i] === q[i]) i++;
    parts = parts.slice(0, i);
  }
  return parts.length ? parts.join("/") + "/" : "";
}
