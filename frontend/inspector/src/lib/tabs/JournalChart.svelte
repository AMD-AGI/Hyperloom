<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<!-- Throughput per journal entry against the baseline, on the tok/s axis it shares with the TTFT scatter; points move
  to their new place when the scale changes. -->
<script lang="ts">
  import { fmtNum } from "../format";
  import type { StreamItem } from "../types";
  import { outcomeClass } from "./rows";
  import { CHART, type TputAxis } from "./tputAxis";
  import TputGrid from "./TputGrid.svelte";

  let {
    rows,
    tput,
    axis,
    baseline,
    onpick,
    oncontext,
  }: {
    rows: StreamItem[];
    tput: (j: StreamItem) => number | null;
    axis: TputAxis;
    baseline: number | null | undefined;
    onpick: (j: StreamItem) => void;
    /** A right-click on a point, for the page's context menu. */
    oncontext?: (j: StreamItem, e: MouseEvent) => void;
  } = $props();

  const { H, LEFT, RIGHT } = CHART;
  let width = $state(0);
  const W = $derived(Math.max(300, width));
  const pts = $derived(rows.filter((j) => tput(j) != null));
  const X = (i: number) => LEFT + (i * (W - LEFT - RIGHT)) / Math.max(1, pts.length - 1);
  const line = $derived(pts.map((j, i) => `${i ? "L" : "M"}${X(i)},${axis.y(Number(tput(j)))}`).join(""));
</script>

<div class="chart" bind:clientWidth={width}>
  <svg viewBox="0 0 {W} {H}" width={W} height={H} data-testid="journal-chart">
    {#if !pts.length}
      <text x="10" y="20" class="dim">no throughput measurements yet</text>
    {:else}
      <TputGrid {axis} width={W} />
      <text class="dim" x={(LEFT + W - RIGHT) / 2} y={H - 4} text-anchor="middle">journal entries, oldest first</text>
      {#if baseline != null}
        <line class="base" x1={LEFT} x2={W - RIGHT} y1={axis.y(baseline)} y2={axis.y(baseline)} />
        <text class="dim" x={W - RIGHT} y={axis.y(baseline) - 4} text-anchor="end">baseline {fmtNum(baseline, 0)}</text>
      {/if}
      <path class="line" d={line} />
      {#each pts as j, i (j)}
        <circle
          class="pt {outcomeClass(j.outcome)}"
          data-testid="journal-point"
          cx={X(i)}
          cy={axis.y(Number(tput(j)))}
          r="4.5"
          role="button"
          tabindex="-1"
          onclick={() => onpick(j)}
          oncontextmenu={oncontext && ((e) => oncontext(j, e))}
          onkeydown={() => {}}
        >
          <title>{j.change}: {fmtNum(Number(tput(j)))} tok/s ({j.outcome})</title>
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
  .base {
    stroke: var(--muted);
    stroke-dasharray: 4 3;
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
