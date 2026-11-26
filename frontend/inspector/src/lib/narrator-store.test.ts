// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { NarratorStore } from "./narrator-store.svelte";

type Call = { url: string; method: string; body?: string; type?: string };
type Reply = [number, unknown];

/** A fetch that records calls and answers GETs with `status`, POSTs from `posts` in order. */
function server(status: unknown, posts: Reply[] = []) {
  const calls: Call[] = [];
  const fetcher = (async (url: string, init?: RequestInit) => {
    const headers = (init?.headers ?? {}) as Record<string, string>;
    calls.push({ url, method: init?.method ?? "GET", body: init?.body as string, type: headers["Content-Type"] });
    const [code, body] = init?.method === "POST" ? (posts.shift() ?? [200, {}]) : [200, status];
    return new Response(JSON.stringify(body), { status: code });
  }) as typeof fetch;
  return { calls, fetcher };
}

const entry = (id: string, kind: "narration" | "answer", text = "x") => ({ id, kind, text, clock: 1 });
const ready = (entries: unknown[] = []) => ({ available: true, reason: null, busy: false, entries });

describe("NarratorStore", () => {
  it("narrates with the pinned focus as JSON and reloads the entries", async () => {
    const { calls, fetcher } = server(ready([entry("1", "narration", "All quiet")]), [[200, {}]]);
    const n = new NarratorStore(fetcher);
    await n.refresh();
    n.setFocus("  memory ");
    await n.narrate(true);
    const post = calls.find((c) => c.method === "POST")!;
    expect(post.url).toBe("api/narrate?force=1");
    expect(post.type).toBe("application/json");
    expect(JSON.parse(post.body!)).toEqual({ focus: "memory" });
    expect(n.latest.narration?.text).toBe("All quiet");
    expect(n.pending).toBeNull();
  });
  it("does nothing when no LLM is configured", async () => {
    const { calls, fetcher } = server({ available: false, reason: "no LLM credentials", busy: false, entries: [] });
    const n = new NarratorStore(fetcher);
    await n.refresh();
    await n.narrate(true);
    expect(await n.ask("why?")).toBe(false);
    expect(calls.every((c) => c.method === "GET")).toBe(true);
    expect(n.status?.reason).toBe("no LLM credentials");
  });
  it("keeps an automatic refusal quiet and shows one the viewer asked for", async () => {
    const busy: Reply = [429, { error: "the narrator is already writing" }];
    const { fetcher } = server(ready(), [busy, busy]);
    const n = new NarratorStore(fetcher);
    await n.refresh();
    await n.narrate(false);
    expect(n.error).toBe("");
    await n.narrate(true);
    expect(n.error).toBe("the narrator is already writing");
  });
  it("sends a trimmed question and reports failures", async () => {
    const { calls, fetcher } = server(ready(), [
      [200, {}],
      [502, { error: "RuntimeError: boom" }],
    ]);
    const n = new NarratorStore(fetcher);
    await n.refresh();
    expect(await n.ask("  ")).toBe(false);
    expect(await n.ask(" what is running? ")).toBe(true);
    expect(JSON.parse(calls.find((c) => c.method === "POST")!.body!)).toEqual({ question: "what is running?" });
    expect(await n.ask("again")).toBe(false);
    expect(n.error).toBe("RuntimeError: boom");
    expect(n.asking).toBe("");
  });
  it("holds a question asked during a narration until the narration is written", async () => {
    let release = () => {};
    const gate = new Promise<void>((r) => (release = r));
    const calls: string[] = [];
    const fetcher = (async (url: string, init?: RequestInit) => {
      if (init?.method === "POST") {
        calls.push(url);
        if (url.startsWith("api/narrate")) await gate;
      }
      return new Response(JSON.stringify(ready()), { status: 200 });
    }) as typeof fetch;
    const n = new NarratorStore(fetcher);
    await n.refresh();
    const narrating = n.narrate(false);
    const asking = n.ask("what now?");
    await Promise.resolve();
    expect(calls).toEqual(["api/narrate?force=0"]);
    release();
    await narrating;
    expect(await asking).toBe(true);
    expect(calls).toEqual(["api/narrate?force=0", "api/ask"]);
  });
  it("runs a forced narration after the call in flight and skips an automatic one", async () => {
    let release = () => {};
    const gate = new Promise<void>((r) => (release = r));
    const calls: string[] = [];
    const fetcher = (async (url: string, init?: RequestInit) => {
      if (init?.method === "POST") {
        calls.push(url);
        if (url === "api/ask") await gate;
      }
      return new Response(JSON.stringify(ready()), { status: 200 });
    }) as typeof fetch;
    const n = new NarratorStore(fetcher);
    await n.refresh();
    const asking = n.ask("what now?");
    await Promise.resolve();
    await n.narrate(false);
    const forced = n.narrate(true);
    expect(calls).toEqual(["api/ask"]);
    release();
    await Promise.all([asking, forced]);
    expect(calls).toEqual(["api/ask", "api/narrate?force=1"]);
  });
});
