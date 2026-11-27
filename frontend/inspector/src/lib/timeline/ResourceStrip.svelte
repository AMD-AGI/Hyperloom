<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<script lang="ts">
  import { fmtBytes, fmtNum } from "../format";
  import type { StreamItem } from "../types";
  import { noSamplesNote, type SampleContext } from "./resources";
  import { LABEL, RES, type Range } from "./scale";

  type Metric = StreamItem & {
    memory?: { used_bytes?: number | null; limit_bytes?: number | null };
    gpus?: { gfx_pct?: number | null; vram_used_mb?: number; vram_total_mb?: number }[];
  };

  let {
    metrics,
    range,
    X,
    y0,
    W,
    context,
  }: { metrics: StreamItem[]; range: Range; X: (t: number) => number; y0: number; W: number; context: SampleContext } =
    $props();

  const Y = (f: number) => y0 + RES - Math.max(0, Math.min(1, f)) * (RES - 4) - 2;
  const pts = $derived((metrics as Metric[]).filter((m) => m._t >= range.t0 && m._t <= range.t1));
  const mem = $derived(
    pts
      .filter((m) => m.memory?.used_bytes != null && m.memory.limit_bytes)
      .map((m) => `${X(m._t)},${Y(m.memory!.used_bytes! / m.memory!.limit_bytes!)}`)
      .join(" "),
  );
  const gpu = $derived(
    pts
      .filter((m) => m.gpus?.length)
      .map((m) => {
        const g = m.gpus!.map((x) => x.gfx_pct || 0);
        return `${X(m._t)},${Y(g.reduce((a, b) => a + b, 0) / g.length / 100)}`;
      })
      .join(" "),
  );
  const last = $derived(pts.at(-1));
  const vram = $derived(
    (last?.gpus ?? [])
      .map((g) => `${fmtNum((g.vram_used_mb ?? 0) / 1024, 0)}/${fmtNum((g.vram_total_mb ?? 0) / 1024, 0)}G`)
      .join(" "),
  );
</script>

<g class="strip" data-testid="resource-strip">
  <text x="4" y={y0 + 12} class="dim">memory</text>
  <text x="4" y={y0 + 26} class="dim">GPU util</text>
  <rect x={LABEL} y={y0} width={W - LABEL - 8} height={RES} fill="none" stroke="var(--line)" />
  {#if !last}
    <text x={LABEL + 6} y={y0 + RES / 2 + 4} class="dim" data-testid="resource-note">{noSamplesNote(context)}</text>
  {:else}
    {#if mem}<polyline points={mem} fill="none" stroke="var(--failed)" stroke-width="1.5" />{/if}
    {#if gpu}<polyline points={gpu} fill="none" stroke="var(--ok)" stroke-width="1.5" />{/if}
    <text x={W - 12} y={y0 + 12} text-anchor="end" fill="var(--failed)">
      mem {fmtBytes(last.memory?.used_bytes)} / {last.memory?.limit_bytes
        ? fmtBytes(last.memory.limit_bytes)
        : "no limit"}
    </text>
    <text x={W - 12} y={y0 + 26} text-anchor="end" fill="var(--ok)">
      {last.gpus?.length
        ? `GPU ${last.gpus.map((g) => fmtNum(g.gfx_pct, 0) + "%").join(" ")}  VRAM ${vram}`
        : "no GPU data"}
    </text>
  {/if}
</g>

<style>
  text {
    font-size: 10px;
  }
  .dim {
    fill: var(--muted);
  }
</style>
