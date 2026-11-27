<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<script lang="ts">
  import "@xyflow/svelte/dist/style.css";
  import { Background, Controls, MarkerType, SvelteFlow } from "@xyflow/svelte";
  import type { EdgeTypes, NodeTypes } from "@xyflow/svelte";
  import { untrack } from "svelte";
  import { refFor } from "../explain";
  import { ms } from "../motion";
  import type { InspectorStore } from "../store.svelte";
  import type { Layout, UiState } from "../ui.svelte";
  import AgentNode from "./AgentNode.svelte";
  import BusEdge from "./BusEdge.svelte";
  import FitView from "./FitView.svelte";
  import { pairEdges, recentRates } from "./edges";
  import { reuse } from "./reuse";
  import {
    curve,
    curvePath,
    FORCE_H,
    forceLayout,
    GROUP_ROW,
    layered,
    lerpPositions,
    NH,
    NW,
    placeLabels,
    type Positions,
  } from "./layout";
  import { theme } from "../theme.svelte";
  import type { AgentNodeType, BusEdgeType } from "./types";

  let { store, ui }: { store: InspectorStore; ui: UiState } = $props();

  const nodeTypes: NodeTypes = { agent: AgentNode };
  const edgeTypes: EdgeTypes = { bus: BusEdge };
  const MORPH_MS = 550;

  let width = $state(0);
  const W = $derived(Math.max(600, width || 1100));
  const graph = $derived(store.view?.graph ?? { nodes: [], edges: [] });
  const pairs = $derived(pairEdges(graph.edges));
  const height = $derived(
    ui.layout === "force"
      ? FORCE_H + 20
      : Math.max(220, 40 + new Set(graph.nodes.map((n) => GROUP_ROW[n.group] ?? 2)).size * 100 + 20),
  );

  // Positions change only when the node set, the layout or the width does; data updates keep them.
  const shape = $derived(
    graph.nodes
      .map((n) => `${n.id}|${n.group}`)
      .sort()
      .join(","),
  );
  let forcePos: Positions = {};
  let shown = $state.raw<Positions>({});
  let fitKey = $state(0);
  let frame = 0;

  function morph(from: Positions, to: Positions, refit: boolean): void {
    cancelAnimationFrame(frame);
    const duration = ms(MORPH_MS);
    const done = () => {
      shown = to;
      if (refit) fitKey++;
    };
    if (!duration || !Object.keys(from).length) return done();
    const t0 = performance.now();
    const step = (now: number) => {
      const t = (now - t0) / duration;
      if (t >= 1) return done();
      shown = lerpPositions(from, to, t);
      frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
  }

  let lastLayout: Layout | null = null;
  $effect(() => {
    void shape;
    const layout = ui.layout;
    const w = W;
    untrack(() => {
      const target =
        layout === "force" ? (forcePos = forceLayout(graph.nodes, pairs, forcePos, w)) : layered(graph.nodes, w);
      const first = !Object.keys(shown).length;
      morph(shown, target, first || layout !== lastLayout);
      lastLayout = layout;
    });
  });
  $effect(() => () => cancelAnimationFrame(frame));

  // Recent traffic is re-read every second, so pulses stop a minute after the last event.
  let tick = $state(0);
  $effect(() => {
    const id = setInterval(() => tick++, 1000);
    return () => clearInterval(id);
  });
  const recent = $derived.by(() => {
    void tick;
    return recentRates(store.streams.event, store.now());
  });

  let nodes = $state.raw<AgentNodeType[]>([]);
  let edges = $state.raw<BusEdgeType[]>([]);
  $effect(() => {
    const filter = ui.filter;
    const next = graph.nodes
      .filter((n) => shown[n.id])
      .map((n) => ({
        id: n.id,
        type: "agent" as const,
        position: { x: shown[n.id].x - NW / 2, y: shown[n.id].y - NH / 2 },
        width: NW,
        height: NH,
        draggable: false,
        connectable: false,
        data: { node: n, selected: filter === n.id, faded: !!filter && filter !== n.id },
      }));
    nodes = untrack(() => reuse(nodes, next));
  });
  // Geometry and label placement follow positions and pairs only, not the once-a-second traffic rates.
  const geometry = $derived.by(() => {
    const pos = shown;
    const drawn = pairs.filter((p) => pos[p.source] && pos[p.target]);
    const curves = drawn.map((p) => curve(pos[p.source], pos[p.target], p.back));
    const labels = placeLabels(
      Object.values(pos),
      drawn.map((p, i) => ({
        curve: curves[i],
        texts: [p.label.length > 40 ? `${p.label.slice(0, 39)}…` : p.label, `×${p.total}`],
      })),
    );
    return { drawn, curves, labels };
  });
  $effect(() => {
    const filter = ui.filter;
    const { drawn, curves, labels } = geometry;
    const next = drawn.map((p, i) => ({
      id: p.id,
      source: p.source,
      target: p.target,
      type: "bus" as const,
      markerEnd: {
        type: MarkerType.ArrowClosed,
        width: 14,
        height: 14,
        markerUnits: "userSpaceOnUse",
        color: "var(--muted)",
      },
      data: {
        pair: p,
        recent: recent.get(p.id) ?? 0,
        faded: !!filter && filter !== p.source && filter !== p.target,
        path: curvePath(curves[i]),
        label: labels[i],
      },
    }));
    edges = untrack(() => reuse(edges, next));
  });
</script>

<section class="graph" data-testid="graph">
  <header>
    <h2>Harness graph</h2>
    <div class="layouts" role="radiogroup" aria-label="Graph layout">
      {#each ["layered", "force"] as const as l (l)}
        <button
          role="radio"
          aria-checked={ui.layout === l}
          class:on={ui.layout === l}
          data-testid="layout-{l}"
          onclick={() => ui.setLayout(l)}>{l}</button
        >
      {/each}
    </div>
    {#if ui.filter}
      <button class="filter" data-testid="graph-filter" onclick={() => (ui.filter = null)}>
        filter: {ui.filter} ✕
      </button>
    {/if}
  </header>
  <div class="canvas" bind:clientWidth={width} style:height="{height}px">
    <SvelteFlow
      bind:nodes
      bind:edges
      {nodeTypes}
      {edgeTypes}
      fitView
      fitViewOptions={{ padding: 0.08 }}
      colorMode={theme.current}
      nodesDraggable={false}
      nodesConnectable={false}
      elementsSelectable={false}
      zoomOnDoubleClick={false}
      minZoom={0.3}
      maxZoom={2.5}
      onnodeclick={({ node }) => ui.pickNode(node.id)}
      onnodecontextmenu={({ node, event }) =>
        ui.openMenu(event, refFor({ kind: "node", id: node.id }), () => ui.pickNode(node.id))}
      onpaneclick={() => (ui.filter = null)}
    >
      <FitView key={fitKey} />
      <Controls showLock={false} />
      <Background gap={24} />
    </SvelteFlow>
  </div>
</section>

<style>
  .graph {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    overflow: hidden;
  }
  header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.4rem 0.75rem;
    border-bottom: 1px solid var(--line);
  }
  h2 {
    margin: 0;
    font-size: 13px;
  }
  .layouts {
    display: flex;
    border: 1px solid var(--line);
    border-radius: 5px;
    overflow: hidden;
  }
  button {
    font: inherit;
    font-size: 12px;
    color: var(--muted);
    background: transparent;
    border: 0;
    padding: 2px 10px;
    cursor: pointer;
    transition:
      background 0.2s ease,
      color 0.2s ease;
  }
  button.on {
    background: var(--panel2);
    color: var(--fg);
  }
  .filter {
    margin-left: auto;
    color: var(--accent);
    border: 1px solid var(--line);
    border-radius: 5px;
  }
  /* No height transition: every frame of one would fire Svelte Flow's resize observer and re-render the graph. */
  .canvas {
    --xy-background-color: var(--panel);
    --xy-background-pattern-dots-color: var(--line);
    --xy-edge-label-background-color: transparent;
    --xy-controls-button-background-color: var(--panel2);
    --xy-controls-button-background-color-hover: var(--line);
    --xy-controls-button-color: var(--fg);
    --xy-controls-button-border-color: var(--line);
    --xy-attribution-background-color: transparent;
  }
  .canvas :global(.svelte-flow__node) {
    border: 0;
    padding: 0;
    background: none;
    box-shadow: none;
  }
</style>
