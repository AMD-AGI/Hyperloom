// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Typed client for the Inspector's HTTP API. Paths are relative, so the page also works behind a path prefix.

import type { Artifact, ArtifactPayload, Clock, DiffPayload, Snapshot } from "./types";

export type Fetcher = typeof fetch;

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function call<T>(fetcher: Fetcher, path: string, params?: Record<string, string>, post = false): Promise<T> {
  const q = params ? "?" + new URLSearchParams(params).toString() : "";
  const r = await fetcher(path + q, { cache: "no-store", method: post ? "POST" : "GET" });
  const body = (await r.json().catch(() => ({}))) as { error?: string };
  if (!r.ok) throw new ApiError(body.error || `HTTP ${r.status}`, r.status);
  return body as T;
}

export const fetchSnapshot = (fetcher: Fetcher = fetch) => call<Snapshot>(fetcher, "api/snapshot");

export const fetchArtifacts = async (fetcher: Fetcher = fetch) =>
  (await call<{ artifacts?: Artifact[] }>(fetcher, "api/artifacts")).artifacts ?? [];

export interface ArtifactQuery {
  mode?: "json" | "jsonl" | "text";
  offset?: number;
}

export function fetchArtifact(path: string, opts: ArtifactQuery = {}, fetcher: Fetcher = fetch) {
  const params: Record<string, string> = { path };
  for (const [k, v] of Object.entries(opts)) if (v != null) params[k] = String(v);
  return call<ArtifactPayload>(fetcher, "api/artifact", params);
}

export const fetchDiff = (a: string, b: string, fetcher: Fetcher = fetch) =>
  call<DiffPayload>(fetcher, "api/diff", { a, b });

/** A recorded LLM conversation, by call id or by specialist task and turn. */
export type ConversationKey = { call_id: string } | { task_id: string; turn?: number | string };

export function fetchConversation(key: ConversationKey, fetcher: Fetcher = fetch) {
  const params: Record<string, string> = {};
  for (const [k, v] of Object.entries(key)) params[k] = v == null ? "" : String(v);
  return call<Record<string, unknown>>(fetcher, "api/conversation", params);
}

export interface ReplayControl {
  t?: number;
  speed?: number;
  paused?: boolean;
}

export function postReplay(ctl: ReplayControl, fetcher: Fetcher = fetch) {
  const params: Record<string, string> = {};
  if (ctl.t != null) params.t = String(ctl.t);
  if (ctl.speed != null) params.speed = String(ctl.speed);
  if (ctl.paused != null) params.paused = ctl.paused ? "1" : "0";
  return call<Clock>(fetcher, "api/replay", params, true);
}
