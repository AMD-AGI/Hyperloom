// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shapes of the Inspector API (src/hyperloom/inspector/watcher.py). Only the fields the page reads are typed; the
// server sends more, and every record is redacted server-side.

export const STREAM_NAMES = ["event", "llm_call", "journal", "review", "metrics", "benchmark"] as const;
export type StreamName = (typeof STREAM_NAMES)[number];

/** A stream record; `_t` is its release time in epoch seconds. */
export type StreamItem = { _t: number } & Record<string, unknown>;
export type Cursors = Partial<Record<StreamName, number>>;

export interface Clock {
  mode: "live" | "replay";
  now: number;
  start?: number | null;
  end?: number | null;
  speed?: number;
  paused?: boolean;
}

export type Health = "crashed" | "failed" | "stalled" | "running" | "queued" | "ok" | "cancelled" | "idle";

export interface Task {
  task_id: string;
  kind: string;
  state: string;
  health: Health;
  reason?: string;
  node: string;
  label: string;
  created_at?: string;
  updated_at?: string;
  history?: { from: string; to: string; ts: string; evidence?: unknown }[];
  progress?: Record<string, unknown>;
  params?: Record<string, unknown>;
}

export interface GraphNode {
  id: string;
  label: string;
  group: string;
  layer: number;
  tasks: Record<string, number>;
  llm_calls: number;
  llm_tokens: number;
  artifacts: number;
  trees: Record<string, unknown>;
  health: Health;
}

export interface GraphEdge {
  source: string;
  target: string;
  label: string;
  count: number;
}

export interface TimelineBar {
  task_id: string;
  node: string;
  label: string;
  start: number;
  end: number | null;
  state: string;
  health: Health;
  step?: string | null;
}

/** A phase from one transition to the next; `end` is null for the phase still running. */
export interface PhaseBand {
  phase: string;
  start: number;
  end: number | null;
  reason?: string;
}

export interface Timeline {
  bars: TimelineBar[];
  /** Phases up to the clock. */
  phases: PhaseBand[];
  /** Replay only: the phases of the whole recording, for the scrubber. */
  chapters?: PhaseBand[];
  steps: { id: string; kind: string; status: string; start: number; end: number | null }[];
  /** Live, the run's last activity once it has gone quiet; null while it is active, and in replay. */
  end?: number | null;
}

export interface RunInfo {
  session_dir?: string;
  session_id?: string;
  model?: string;
  manifest?: Record<string, unknown>;
  pod_history?: { acquired_at: string; hostname: string; pid: number }[];
  /** `host` is set when the session's lock names a host other than the Inspector's. */
  optimizer?: { alive: boolean | null; reason?: string; host?: string };
}

/** `/api/snapshot` without streams, as the SSE `view` event sends it. */
export interface View {
  version: number;
  clock: Clock;
  run: RunInfo;
  summary: Record<string, unknown> & { phase?: string; session_id?: string };
  budget: { used_sec: number; total_sec: number; pct: number } | null;
  tasks: Task[];
  graph: { nodes: GraphNode[]; edges: GraphEdge[] };
  timeline: Timeline;
  llm: Record<string, Record<string, unknown>>;
  badges: { level: string; text: string }[];
  agents: { agent: string; age_sec: number; stale: boolean }[];
  artifact_count: number;
  /** Where the artifact list stops when the session holds more files than the index lists; null when it is whole. */
  artifacts_cut_at: number | null;
  released: Record<StreamName, number>;
}

export interface Snapshot extends View {
  streams: Record<StreamName, StreamItem[]>;
}

export interface Artifact {
  path: string;
  size: number;
  mtime: number;
  type: string;
  producer: string;
  task_id: string | null;
  /** The large subtree (more than 200 files) the file sits in; such files collapse into one tree entry. */
  bulk?: string;
}

/** `/api/diff`: rows are [tag, line a, text a, line b, text b]; a "hunk" row separates hunks. */
export type DiffRow = [string, number | null, string, number | null, string];
export interface DiffPayload {
  a: string;
  b: string;
  rows: DiffRow[];
  identical: boolean;
}

/** A progress unit of a task (derive.py `task_progress`): a finished variant or baseline round. */
export interface ProgressUnit {
  unit: string;
  label: string;
  status: string;
  output_throughput?: number | null;
  ts?: number | null;
  earlier?: number[];
  stage?: string;
  reason?: string;
  error?: string;
}

export interface ArtifactPayload {
  path: string;
  type: string;
  mode: "json" | "jsonl" | "text" | "meta";
  size: number;
  mtime: number;
  note?: string;
  data?: unknown;
  rows?: Record<string, unknown>[];
  offset?: number;
  total?: number;
  text?: string;
  start?: number;
  end?: number;
  truncated?: boolean;
  [key: string]: unknown;
}
