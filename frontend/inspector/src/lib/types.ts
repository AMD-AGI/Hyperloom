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

/** The accuracy state.json records, as fractions: the baseline's, and each kept variant's with when it was measured. */
export interface AccuracyView {
  baseline: number | null;
  variants: { variant: string; value: number; t: number | null }[];
}

/** Token sums of a rollup (tokens.py); `input` is uncached, so the five kinds add up to `total`. */
export interface TokenTotals {
  input: number;
  cache_write: number;
  cache_read: number;
  output: number;
  reasoning: number;
  total: number;
  calls: number;
  errors: number;
  web_searches?: number;
  /** USD at `pricing`'s prices; null where none of the calls is priced. */
  cost_usd?: number | null;
  /** Calls left out of `cost_usd`, as a price they need is missing; by why: `no_model`, with no model recorded;
   * `no_price`, of a model the price sheet has no price for; `no_search_price`, with web searches and no price per
   * search. */
  unpriced?: number;
  no_model?: number;
  no_price?: number;
  no_search_price?: number;
  /** Turns read from a specialist's log that the ledger has not got. */
  live?: number;
  /** Of those, a running specialist's, whose output is only known when it ends. */
  output_pending?: number;
  /** Of those, a specialist's that ended before reporting its output, which is unknown. */
  output_unreported?: number;
}

/**
 * Sparse buckets, `step` s from `start`: one `[bucket, node index, ...kinds, calls, ...kinds]` row per non-empty pair,
 * tokens then USD.
 */
export interface TokenSeries {
  start: number;
  end: number;
  step: number;
  kinds: string[];
  nodes: string[];
  rows: number[][];
}

/** One model's entry on the price sheet, USD per 1M tokens. */
export interface TokenPrice {
  id: string;
  input: number;
  cache_write: number;
  cache_read: number;
  output: number;
  /** "listed" on the sheet, or the input rate the sheet bills a cache write it lists none for at. */
  cache_basis: string;
  /** What this run's calls priced in full cost, by kind: the parts of the model's cost_usd. */
  spent_usd: Record<"input" | "cache_write" | "cache_read" | "output" | "reasoning", number>;
}

export interface TokenPricing {
  /** The built-in price sheet's file. */
  source: string;
  /** What the sheet says its rates come from. The sheet gives no price per search: a call that searches is unpriced. */
  built_from?: string | null;
  models: Record<string, TokenPrice>;
  unpriced: Record<string, { calls: number; total: number }>;
}

/** A figure's name on the Tokens tab and what it means (tokens.py `TERMS`). */
export interface TokenTerm {
  group: string;
  label: string;
  definition: string;
}

/** `view.tokens`: token consumption up to the clock. */
export interface TokenView {
  totals: TokenTotals;
  by_component: Record<string, TokenTotals>;
  by_node: Record<string, TokenTotals>;
  by_model: Record<string, TokenTotals>;
  pending_tasks: string[];
  /** Specialists that ended before reporting their output. */
  unreported_tasks?: string[];
  recent: { window_sec: number; total: number; calls: number; cost_usd?: number | null };
  series: TokenSeries;
  /** What the cost is priced by: the built-in price sheet. */
  pricing: TokenPricing;
  /** The Claude CLI's own cost of its finished specialists next to ours for them. */
  check?: { tasks: number; claude_cli_usd: number; priced_usd: number | null } | null;
  /** Names and definitions, keyed by field (`input`, `cache_read`, `total`, ...) or figure (`cache_hit`, ...). */
  terms?: Record<string, TokenTerm>;
  /** The token timeline file the Inspector keeps, or null when it keeps none. */
  file: string | null;
  saved_at: number | null;
}

/** `/api/snapshot` without streams, as the SSE `view` event sends it. */
export interface View {
  version: number;
  clock: Clock;
  run: RunInfo;
  summary: Record<string, unknown> & { phase?: string; session_id?: string; accuracy?: AccuracyView };
  budget: { used_sec: number; total_sec: number; pct: number } | null;
  tasks: Task[];
  graph: { nodes: GraphNode[]; edges: GraphEdge[] };
  timeline: Timeline;
  llm: Record<string, Record<string, unknown>>;
  tokens: TokenView;
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

/** A reference to one item for "Explain this" (POST /api/ask `item`); the server resolves it against its own records.
 * Journal entries have no id, so their task, change and time name one; a variant is a task's grid entry by label. */
export type ItemRef =
  | { kind: "node"; id: string }
  | { kind: "task"; id: string }
  | { kind: "event"; seq: number }
  | { kind: "call"; call_id?: string; task_id?: string; turn?: number }
  | { kind: "journal"; task_id: string; change: string; ts: string }
  | { kind: "variant"; task_id: string; label: string };

/** A narration or an answer from the narrator; times are epoch seconds, `clock` the session time it describes. */
export interface NarratorEntry {
  id: string;
  kind: "narration" | "answer";
  clock: number;
  written_at: number;
  duration_sec: number;
  model: string | null;
  text: string;
  focus?: string | null;
  question?: string;
  /** The item an answer explains, and the display name the server resolved for it. */
  item?: ItemRef;
  label?: string;
}

/** GET /api/narrator: whether an LLM is configured, and the entries at or before the clock. */
export interface NarratorStatus {
  available: boolean;
  reason: string | null;
  backend?: string;
  model?: string | null;
  busy: boolean;
  clock?: number;
  entries: NarratorEntry[];
}
