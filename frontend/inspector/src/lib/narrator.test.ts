// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import type { Records } from "./explain";
import { asked, blocks, chipper, current, explainHead } from "./narrator";
import type { GraphNode, NarratorEntry, View } from "./types";

const node = (id: string, label: string) => ({ id, label }) as GraphNode;

describe("blocks", () => {
  it("splits paragraphs and bullet lists and drops Markdown marks", () => {
    const text =
      "## **Explore** is benchmarking\n\n- variant `aiter_moe_on`\n* step 3\n  of 5\n1. next: roofline\nTail.";
    expect(blocks(text)).toEqual([
      { kind: "p", text: "Explore is benchmarking" },
      { kind: "list", items: ["variant aiter_moe_on", "step 3 of 5", "next: roofline"] },
      { kind: "p", text: "Tail." },
    ]);
  });
  it("keeps HTML as text", () => {
    expect(blocks("<img src=x onerror=alert(1)>")).toEqual([{ kind: "p", text: "<img src=x onerror=alert(1)>" }]);
  });
});

describe("chipper", () => {
  const nodes = [
    node("spec:research_scout", "research_scout"),
    node("exec:explore", "explore"),
    node("agent:kernel", "kernel agent"),
    node("bus", "broadcast"),
    node("x:tp", "tp"),
  ];
  const split = chipper(nodes);
  it("marks whole-word node names, longest first, case-insensitively", () => {
    expect(split("The kernel agent and Explore ran; research_scout crashed.")).toEqual([
      { text: "The " },
      { text: "kernel agent", node: "agent:kernel" },
      { text: " and " },
      { text: "Explore", node: "exec:explore" },
      { text: " ran; " },
      { text: "research_scout", node: "spec:research_scout" },
      { text: " crashed." },
    ]);
  });
  it("ignores names inside other words and names shorter than four characters", () => {
    expect(split("explored tp research_scout_2")).toEqual([{ text: "explored tp research_scout_2" }]);
  });
  it("passes text through without nodes", () => {
    expect(chipper([])("anything")).toEqual([{ text: "anything" }]);
  });
});

describe("current", () => {
  const e = (id: string, kind: NarratorEntry["kind"]) => ({ id, kind }) as NarratorEntry;
  it("takes the newest narration and the latest answers", () => {
    const got = current([e("1", "narration"), e("2", "answer"), e("3", "narration"), e("4", "answer")]);
    expect(got.narration?.id).toBe("3");
    expect(got.answers.map((a) => a.id)).toEqual(["2", "4"]);
    expect(current([]).narration).toBeNull();
  });
});

describe("asked", () => {
  const rec: Records = {
    view: { graph: { nodes: [node("spec:roofline", "roofline")] }, tasks: [] } as unknown as View,
    streams: {},
  };
  const answer = (extra: Partial<NarratorEntry>) => ({ id: "1", kind: "answer", ...extra }) as NarratorEntry;
  const ref = { kind: "node", id: "spec:roofline" } as const;
  const fixed = "Explain roofline: what does this agent do, what has it done so far, and is anything wrong with it?";

  it("shows a typed question without a header", () => {
    expect(asked(answer({ question: "why?" }), rec)).toEqual({ head: null, question: "why?" });
    expect(asked(answer({}), rec)).toEqual({ head: null, question: null });
  });
  it("heads an explanation with the item and drops its fixed question", () => {
    expect(asked(answer({ question: fixed, item: ref, label: "roofline agent" }), rec)).toEqual({
      head: "Explain: roofline agent",
      question: null,
    });
  });
  it("keeps a follow-up's question and falls back to the local label", () => {
    expect(asked(answer({ question: "and now?", item: ref }), rec)).toEqual({
      head: explainHead("roofline"),
      question: "and now?",
    });
  });
});
