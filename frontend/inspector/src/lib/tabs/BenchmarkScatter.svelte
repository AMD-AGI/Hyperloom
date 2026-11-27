<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<!-- Output throughput against the picked latency for every measured benchmark run that has it, joined in time order
  like the throughput chart, on the tok/s axis it shares with it and colored like it by the journal's outcome for the
  variant; upper left is better. -->
<script lang="ts">
  import { fmtNum } from "../format";
  import { latencyLabel, latencyOf, tickDigits, type LatencyPick } from "../latency";
  import type { StreamItem } from "../types";
  import { outcomeClass, variantKey } from "./rows";
  import { CHART, type TputAxis } from "./tputAxis";
  import TputGrid from "./TputGrid.svelte";

  let {
    rows,
    outcomes,
    axis,
    latency,
    onpick,
  }: {
    rows: StreamItem[];
    outcomes: Map<string, string>;
    axis: TputAxis;
    latency: LatencyPick;
    onpick: (b: StreamItem) => void;
  } = $props();

  const { H, LEFT, RIGHT, BOTTOM } = CHART;
  const TICKS = [0, 0.25, 0.5, 0.75, 1];

  let width = $state(0);
  const W = $derived(Math.max(300, width));
  const tput = (b: StreamItem) => Number(b.output_throughput);
  const lat = (b: StreamItem) => Number(latencyOf(b, latency));
  const pts = $derived(rows.filter((b) => latencyOf(b, latency) != null));
  const name = $derived(latencyLabel(latency));
  const xlo = $derived(Math.min(...pts.map(lat)) * 0.95);
  const xhi = $derived(Math.max(...pts.map(lat)) * 1.05);
  const digits = $derived(tickDigits(xhi - xlo));
  const X = (v: number) => LEFT + ((v - xlo) / (xhi - xlo || 1)) * (W - LEFT - RIGHT);
  const line = $derived(pts.map((b, i) => `${i ? "L" : "M"}${X(lat(b))},${axis.y(tput(b))}`).join(""));

  function label(b: StreamItem): string {
    const conc = b.concurrency == null ? "" : ` at concurrency ${b.concurrency}`;
    const outcome = outcomes.get(variantKey(b));
    return `${b.variant ?? b.kind}${outcome ? ` (${outcome})` : ""}: ${fmtNum(tput(b))} tok/s, ${name} ${fmtNum(lat(b), 2)} ms${conc}`;
  }
</script>

<div class="chart" bind:clientWidth={width}>
  <svg viewBox="0 0 {W} {H}" width={W} height={H} data-testid="ttft-chart">
    {#if !pts.length}
      <text x="10" y="20" class="dim">no benchmark runs with a {name} yet</text>
    {:else}
      <TputGrid {axis} width={W} />
      {#each TICKS as t (t)}
        <text class="dim" x={X(xlo + (xhi - xlo) * t)} y={H - BOTTOM + 14} text-anchor="middle">
          {fmtNum(xlo + (xhi - xlo) * t, digits)}
        </text>
      {/each}
      <text class="dim" x={(LEFT + W - RIGHT) / 2} y={H - 4} text-anchor="middle">{name} (ms)</text>
      <path class="line" d={line} />
      {#each pts as b (b)}
        <circle
          class="pt {outcomeClass(outcomes.get(variantKey(b)))}"
          data-testid="ttft-point"
          cx={X(lat(b))}
          cy={axis.y(tput(b))}
          r="4.5"
          role="button"
          tabindex="-1"
          onclick={() => onpick(b)}
          onkeydown={() => {}}
        >
          <title>{label(b)}</title>
        </circle>
      {/each}
    {/if}
  </svg>
</div>

<style>
  .chart {
    width: 100%;
  }
  svg {
    display: block;
  }
  .dim {
    fill: var(--muted);
    font-size: 10px;
  }
  .line {
    fill: none;
    stroke: var(--line);
    stroke-width: 1.5;
    transition: d 0.5s ease;
  }
  .pt {
    fill: var(--accent);
    cursor: pointer;
    transform-box: fill-box;
    transform-origin: center;
    transition:
      cx 0.5s ease,
      cy 0.5s ease;
    animation: pop 0.45s cubic-bezier(0.3, 1.6, 0.5, 1);
  }
  .pt:global(.h-ok) {
    fill: var(--ok);
  }
  .pt:global(.h-failed) {
    fill: var(--failed);
  }
  @keyframes pop {
    from {
      transform: scale(0);
    }
  }
</style>
