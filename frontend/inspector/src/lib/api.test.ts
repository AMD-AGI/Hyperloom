// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { ApiError, fetchArtifact, fetchArtifacts, fetchConversation, fetchSnapshot, postAsk, postReplay } from "./api";

type Call = { url: string; method: string; body?: unknown };

function fake(status: number, body: unknown, calls: Call[] = []) {
  return (async (url: string, init?: RequestInit) => {
    const call: Call = { url, method: init?.method ?? "GET" };
    if (init?.body) call.body = JSON.parse(init.body as string);
    calls.push(call);
    return new Response(JSON.stringify(body), { status });
  }) as typeof fetch;
}

describe("api", () => {
  it("returns the parsed snapshot", async () => {
    const s = await fetchSnapshot(fake(200, { summary: { phase: "PRELUDE" } }));
    expect(s.summary.phase).toBe("PRELUDE");
  });
  it("throws the server's error message with the status", async () => {
    const err = await fetchArtifact("x", {}, fake(404, { error: "no such artifact" })).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.message).toBe("no such artifact");
    expect(err.status).toBe(404);
  });
  it("falls back to the HTTP status without a JSON body", async () => {
    const f = (async () => new Response("oops", { status: 503 })) as typeof fetch;
    await expect(fetchSnapshot(f)).rejects.toThrow("HTTP 503");
  });
  it("uses relative paths and encodes parameters", async () => {
    const calls: Call[] = [];
    await fetchArtifact("runs/a b.log", { mode: "text", offset: 10 }, fake(200, {}, calls));
    expect(calls[0].url).toBe("api/artifact?path=runs%2Fa+b.log&mode=text&offset=10");
  });
  it("unwraps the artifact list", async () => {
    expect(await fetchArtifacts(fake(200, {}))).toEqual([]);
    expect(await fetchArtifacts(fake(200, { artifacts: [{ path: "a" }] }))).toEqual([{ path: "a" }]);
  });
  it("keys conversations by call id or by task and turn", async () => {
    const calls: Call[] = [];
    await fetchConversation({ call_id: "c1" }, fake(200, {}, calls));
    await fetchConversation({ task_id: "t1", turn: 2 }, fake(200, {}, calls));
    expect(calls.map((c) => c.url)).toEqual(["api/conversation?call_id=c1", "api/conversation?task_id=t1&turn=2"]);
  });
  it("posts replay controls", async () => {
    const calls: Call[] = [];
    await postReplay({ t: 12.5, paused: true }, fake(200, { mode: "replay", now: 12.5 }, calls));
    expect(calls[0]).toEqual({ url: "api/replay?t=12.5&paused=1", method: "POST" });
  });
  it("sends the focus and the item of a question only when given", async () => {
    const calls: Call[] = [];
    await postAsk("why?", "", undefined, fake(200, {}, calls));
    await postAsk("why?", "memory", { kind: "task", id: "t1" }, fake(200, {}, calls));
    expect(calls.map((c) => c.body)).toEqual([
      { question: "why?" },
      { question: "why?", focus: "memory", item: { kind: "task", id: "t1" } },
    ]);
  });
});
