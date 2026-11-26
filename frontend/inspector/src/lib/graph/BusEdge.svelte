<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<script lang="ts">
  import { BaseEdge, EdgeLabel, type EdgeProps } from "@xyflow/svelte";
  import { prefersReducedMotion } from "svelte/motion";
  import { pulses, strokeWidth } from "./edges";
  import type { BusEdgeType } from "./types";

  // The graph computes the curve and label spot from its own node positions (see Graph.svelte), so every label
  // is placed knowing the others.
  let { id, data, markerEnd }: EdgeProps<BusEdgeType> = $props();

  const pulse = $derived(prefersReducedMotion.current || !data ? pulses(0) : pulses(data.recent));
</script>

{#if data}
  <g class="bus" class:faded={data.faded} class:live={data.recent > 0} data-testid="graph-edge">
    <BaseEdge
      {id}
      path={data.path}
      {markerEnd}
      style="stroke-width: {strokeWidth(data.pair.total)}px"
      interactionWidth={12}
    />
    {#each { length: pulse.count }, i (i)}
      <circle r="3" class="pulse" data-testid="edge-pulse">
        <animateMotion
          dur="{pulse.dur}s"
          begin="{(-i * pulse.dur) / pulse.count}s"
          repeatCount="indefinite"
          path={data.path}
        />
      </circle>
    {/each}
    <title>{data.pair.label}</title>
  </g>
  {#if data.label}
    <EdgeLabel x={data.label.x} y={data.label.y} class="bus-label{data.faded ? ' faded' : ''}" title={data.pair.label}>
      {data.label.text}
    </EdgeLabel>
  {/if}
{/if}

<style>
  .bus :global(.svelte-flow__edge-path) {
    stroke: var(--muted);
    transition: stroke 0.6s ease;
  }
  .bus.live :global(.svelte-flow__edge-path) {
    stroke: var(--accent);
  }
  .bus.faded {
    opacity: 0.25;
  }
  .pulse {
    fill: var(--accent);
  }
  :global(.bus-label) {
    font-size: 10px;
    color: var(--muted);
    background: color-mix(in srgb, var(--bg) 80%, transparent);
    padding: 0 3px;
    border-radius: 3px;
    white-space: nowrap;
    pointer-events: all;
  }
  :global(.bus-label.faded) {
    opacity: 0.25;
  }
</style>
