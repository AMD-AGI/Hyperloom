// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Page state shared between views: the node filter, the graph layout, the open tab and artifact, and the selected
// record.

import { SvelteSet } from "svelte/reactivity";
import type { LatencyPick } from "./latency";
import type { Gap, Range } from "./timeline/scale";
import type { StreamItem } from "./types";

export type Layout = "layered" | "force";
/** A record picked in a view, for the drawer to show. */
export type Selection =
  | { kind: "node"; id: string }
  | { kind: "task"; id: string }
  | { kind: "event"; item: StreamItem }
  | { kind: "call"; item: StreamItem }
  | { kind: "raw"; title: string; data: unknown };

export const TABS = ["tasks", "artifacts", "bus", "llm", "journal"] as const;
export type Tab = (typeof TABS)[number];
/** The artifact in the viewer: a file (with a JSONL page offset) or a diff of two. */
export type ArtifactView = { kind: "file"; path: string; offset?: number } | { kind: "diff"; a: string; b: string };

const LAYOUT_KEY = "hyperloom-inspector.layout";
const TIMELINE_KEY = "hyperloom-inspector.timeline-open";
const UPLIFT_KEY = "hyperloom-inspector.uplift-open";

function savedLayout(): Layout {
  try {
    return localStorage.getItem(LAYOUT_KEY) === "force" ? "force" : "layered";
  } catch {
    return "layered";
  }
}

function savedOpen(key: string): boolean {
  try {
    return localStorage.getItem(key) !== "0";
  } catch {
    return true;
  }
}

function saveOpen(key: string, open: boolean): void {
  try {
    localStorage.setItem(key, open ? "1" : "0");
  } catch {
    // As for the layout: the choice still holds for this page.
  }
}

export class UiState {
  /** The graph node the views are filtered to, or null. */
  filter = $state<string | null>(null);
  layout = $state<Layout>(savedLayout());
  timelineOpen = $state(savedOpen(TIMELINE_KEY));
  upliftOpen = $state(savedOpen(UPLIFT_KEY));
  /** The timeline's visible range and compressed gaps, so panels below it share its time axis. */
  timeAxis = $state.raw<(Range & { gaps: Gap[] }) | null>(null);
  /** The zoom and pan of the timeline and the panels on its axis, null for the whole run; `instant` skips the spring. */
  zoom = $state.raw<{ range: Range; instant: boolean } | null>(null);
  selection = $state.raw<Selection | null>(null);
  /** Selections the drawer followed links from, for its back button; a pick in a view starts a new trail. */
  trail = $state.raw<Selection[]>([]);
  tab = $state<Tab>("tasks");
  artifact = $state.raw<ArtifactView | null>(null);
  /** Artifacts ticked for a diff. */
  picked = new SvelteSet<string>();
  /** The artifact tree's search box and type filter, and the entries opened or closed against their default. */
  artQuery = $state({ q: "", type: "" });
  artToggled = new SvelteSet<string>();
  /** The latency the throughput tab's scatter plots against tok/s. */
  latency = $state<LatencyPick>({ kind: "ttft", stat: "mean" });

  toggleFilter(node: string): void {
    this.filter = this.filter === node ? null : node;
  }

  /** A graph node click: filter to the node and show it in the drawer, or clear both on a second click. */
  pickNode(node: string): void {
    this.toggleFilter(node);
    if (this.filter) this.select({ kind: "node", id: node });
    else if (this.selection?.kind === "node" && this.selection.id === node) this.select(null);
  }

  select(sel: Selection | null): void {
    this.trail = [];
    this.selection = sel;
  }

  /** Follow a link inside the drawer, keeping the current record for `back()`. */
  follow(sel: Selection): void {
    if (this.selection) this.trail = [...this.trail, this.selection];
    this.selection = sel;
  }

  back(): void {
    if (!this.trail.length) return;
    this.selection = this.trail[this.trail.length - 1];
    this.trail = this.trail.slice(0, -1);
  }

  openArtifact(path: string, offset?: number): void {
    this.artifact = { kind: "file", path, offset };
    this.tab = "artifacts";
  }

  openDiff(): void {
    const [a, b] = [...this.picked];
    if (this.picked.size === 2) this.artifact = { kind: "diff", a, b };
  }

  setLayout(layout: Layout): void {
    this.layout = layout;
    try {
      localStorage.setItem(LAYOUT_KEY, layout);
    } catch {
      // Storage is a convenience; the choice still holds for this page.
    }
  }

  setTimelineOpen(open: boolean): void {
    this.timelineOpen = open;
    saveOpen(TIMELINE_KEY, open);
  }

  setUpliftOpen(open: boolean): void {
    this.upliftOpen = open;
    saveOpen(UPLIFT_KEY, open);
  }

  setZoom(range: Range | null, instant = false): void {
    this.zoom = range && { range, instant };
  }
}
