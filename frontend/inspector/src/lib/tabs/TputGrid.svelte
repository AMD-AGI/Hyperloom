<!-- The gridlines, tick labels and title of the shared tok/s axis. -->
<svelte:options namespace="svg" />

<script lang="ts">
  import { fmtNum } from "../format";
  import { CHART, type TputAxis } from "./tputAxis";

  let { axis, width }: { axis: TputAxis; width: number } = $props();
</script>

<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->

{#each axis.ticks as v (v)}
  <line class="grid" x1={CHART.LEFT} x2={width - CHART.RIGHT} y1={axis.y(v)} y2={axis.y(v)} />
  <text class="dim" x={CHART.LEFT - 6} y={axis.y(v) + 3} text-anchor="end">{fmtNum(v, 0)}</text>
{/each}
<text class="dim" transform="translate(11 {(CHART.TOP + CHART.H - CHART.BOTTOM) / 2}) rotate(-90)" text-anchor="middle"
  >output tok/s</text
>

<style>
  .dim {
    fill: var(--muted);
    font-size: 10px;
  }
  .grid {
    stroke: var(--line);
    stroke-width: 0.5;
  }
</style>
