// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Pure helpers for the narrator panel: split the model's text into paragraphs and bullets, turn graph node names in
// it into chips, and head the answers that explain an item. The text is always rendered as text, never as HTML.

import { labelFor, questionFor, type Records } from "./explain";
import type { GraphNode, NarratorEntry } from "./types";

export type Block = { kind: "p"; text: string } | { kind: "list"; items: string[] };

const BULLET = /^\s*(?:[-*•]|\d+[.)])\s+/;

/** Paragraphs and bullet lists; Markdown emphasis and heading marks are dropped, since the text isn't rendered. */
export function blocks(text: string): Block[] {
  const out: Block[] = [];
  let para: string[] = [];
  const flush = () => {
    if (para.length) out.push({ kind: "p", text: para.join(" ") });
    para = [];
  };
  for (const raw of text.split("\n")) {
    const line = raw.replace(/\*\*|__|`/g, "").replace(/^\s*#+\s*/, "");
    if (!line.trim()) {
      flush();
    } else if (BULLET.test(line)) {
      flush();
      const item = line.replace(BULLET, "").trim();
      const last = out[out.length - 1];
      if (last?.kind === "list") last.items.push(item);
      else out.push({ kind: "list", items: [item] });
    } else {
      const last = out[out.length - 1];
      // A wrapped bullet continues the item before it.
      if (!para.length && last?.kind === "list" && /^\s{2,}/.test(raw))
        last.items[last.items.length - 1] += " " + line.trim();
      else para.push(line.trim());
    }
  }
  flush();
  return out;
}

export type Segment = { text: string; node?: string };

/** Names a node is mentioned by: its label and its id without the kind prefix; short names are skipped. */
function names(nodes: GraphNode[]): Map<string, string> {
  const byName = new Map<string, string>();
  for (const n of nodes) {
    for (const name of [n.label, n.id.replace(/^[a-z]+:/, "")]) {
      if (name.length >= 4 && !byName.has(name.toLowerCase())) byName.set(name.toLowerCase(), n.id);
    }
  }
  return byName;
}

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** Build a splitter that cuts text into plain segments and node mentions (whole words, longest name first). */
export function chipper(nodes: GraphNode[]): (text: string) => Segment[] {
  const byName = names(nodes);
  if (!byName.size) return (text) => [{ text }];
  const alts = [...byName.keys()].sort((a, b) => b.length - a.length).map(escape);
  const re = new RegExp(`(?<![\\w-])(${alts.join("|")})(?![\\w-])`, "gi");
  return (text) => {
    const out: Segment[] = [];
    let at = 0;
    for (const m of text.matchAll(re)) {
      if (m.index > at) out.push({ text: text.slice(at, m.index) });
      out.push({ text: m[0], node: byName.get(m[0].toLowerCase()) });
      at = m.index + m[0].length;
    }
    if (at < text.length) out.push({ text: text.slice(at) });
    return out;
  };
}

/** The newest narration, and the answers written since the one before it, oldest first. */
export function current(entries: NarratorEntry[]): { narration: NarratorEntry | null; answers: NarratorEntry[] } {
  const narration = [...entries].reverse().find((e) => e.kind === "narration") ?? null;
  return { narration, answers: entries.filter((e) => e.kind === "answer").slice(-6) };
}

/** The thread's header for an explained item, e.g. "Explain: roofline". */
export const explainHead = (label: string): string => `Explain: ${label}`;

/**
 * The lines above an answer in the thread: a header when it explains an item, and the question unless it is just the
 * item's fixed question (the header says it); a typed follow-up about the item keeps its question.
 */
export function asked(entry: NarratorEntry, rec: Records): { head: string | null; question: string | null } {
  const question = entry.question || null;
  if (!entry.item) return { head: null, question };
  const head = explainHead(entry.label || labelFor(entry.item, rec));
  return { head, question: question === questionFor(entry.item, rec) ? null : question };
}
