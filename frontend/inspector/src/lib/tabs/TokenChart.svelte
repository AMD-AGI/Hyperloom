<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<!-- Tokens over the run: stacked areas of running totals, or stacked bars per minute; phase bands behind, a
     readout of the bucket under the pointer. On the timeline's time axis, with its now line: the running total runs
     on to the line, so the chart moves with the session clock as the timeline does. -->
<script lang="ts">
  import { fmtCompact, fmtNum, fmtTime } from "../format";
  import { axisTicks, LABEL, tlScale, type Gap, type Range } from "../timeline/scale";
  import { stack, valueTicks, type Decoded, type Mode } from "../tokens";
  import type { Timeline, TokenSeries } from "../types";

  let {
    series,
    decoded,
    hidden,
    mode,
    phases,
    axis,
    height = 240,
    now = null,
  }: {
    series: TokenSeries;
    decoded: Decoded;
    hidden: ReadonlySet<string>;
    mode: Mode;
    phases: Timeline["phases"];
    /** The timeline's time axis: its range (zoom and pan included) and compressed idle gaps. */
    axis: Range & { gaps: Gap[] };
    height?: number;
    /** The session clock: the now line, drawn when it falls inside the axis, and where running totals end. */
    now?: number | null;
  } = $props();

  const H = $derived(height);
  // tlScale's margins, so times sit where the timeline puts them.
  const R = 8;
  const TOP = 18;
  const BOT = 22;
  let width = $state(0);
  let svg = $state<SVGSVGElement>();
  let hover = $state<number | null>(null);

  const W = $derived(Math.max(320, width));
  const s0 = $derived(series.start);
  const s1 = $derived(Math.max(series.end, s0 + 1));
  const step = $derived(series.step);
  const n = $derived(decoded.n);
  const st = $derived(stack(decoded.series, hidden, mode, step));
  const ticks = $derived(valueTicks(st.max));
  const ymax = $derived(ticks[ticks.length - 1] || 1);
  const sc = $derived(tlScale(axis.t0, axis.t1, W, axis.gaps));
  const X = (t: number) => sc.X(t);
  const clampX = (t: number) => Math.max(LABEL, Math.min(W - R, sc.X(t)));
  const Y = (v: number) => TOP + (1 - v / ymax) * (H - TOP - BOT);
  /** Time of cumulative point k: the start, then each bucket's end (the last one cut at the series' end). */
  const tk = (k: number) => Math.min(s0 + k * step, s1);
  /** Past the last bucket nothing was spent, so a running total holds its value up to now. */
  const tEnd = $derived(now != null && now > s1 ? now : s1);
  const timeTicks = $derived(axisTicks(axis, W, sc));
  const nowIn = $derived(now != null && now >= axis.t0 && now <= axis.t1);
  const empty = $derived(!series.rows.length);
  const fmtAxis = (v: number) => fmtCompact(v);
  const exact = (v: number) => fmtNum(v, 0);
  const unit = $derived("tokens" + (mode === "rate" ? "/min" : ""));
  const clip = "tok-plot-tokens";

  const pts = (vals: number[]) => {
    const out = vals.map((v, k) => `${X(tk(k)).toFixed(1)},${Y(v).toFixed(1)}`);
    if (tEnd > s1 && vals.length) out.push(`${X(tEnd).toFixed(1)},${Y(vals[vals.length - 1]).toFixed(1)}`);
    return out;
  };
  const area = (lo: number[], hi: number[]) => `M${pts(hi).join("L")}L${pts(lo).reverse().join("L")}Z`;
  const edge = (hi: number[]) => "M" + pts(hi).join("L");
  const bars = $derived.by(() => {
    if (mode !== "rate") return [];
    const out: { x: number; y: number; w: number; h: number; color: string; key: string }[] = [];
    for (const layer of st.layers) {
      layer.hi.forEach((hi, i) => {
        if (hi <= layer.lo[i]) return;
        const x = X(s0 + i * step);
        const px = X(s0 + (i + 1) * step) - x;
        const w = Math.max(1, px - (px > 4 ? 1 : 0));
        out.push({ x, y: Y(hi), w, h: Y(layer.lo[i]) - Y(hi), color: layer.color, key: `${layer.key}:${i}` });
      });
    }
    return out;
  });
  const bands = $derived(
    phases.map((p, i) => ({ ...p, x: clampX(p.start), x2: clampX(p.end ?? tEnd), i })).filter((p) => p.x2 - p.x > 0.5),
  );

  function move(e: PointerEvent) {
    const box = svg?.getBoundingClientRect();
    if (!box || empty) return;
    const x = e.clientX - box.left;
    const t = sc.T(x);
    if (x < LABEL || x > W - R || t < s0 || t >= s1) return void (hover = null);
    hover = Math.max(0, Math.min(n - 1, Math.floor((t - s0) / step)));
  }

  const readout = $derived.by(() => {
    if (hover == null) return null;
    const i = hover;
    const k = mode === "cumulative" ? i + 1 : i;
    const all = st.layers
      .map((l) => ({ key: l.key, label: l.label, color: l.color, value: l.hi[k] - l.lo[k], tokens: l.values[i] }))
      .reverse();
    const rows = all.some((r) => r.value) ? all.filter((r) => r.value) : all;
    const x = mode === "cumulative" ? X(tk(i + 1)) : X(s0 + (i + 0.5) * step);
    const inBucket = rows.reduce((a, r) => a + r.tokens, 0);
    const calls = decoded.calls[i] ?? 0;
    const amount = `${fmtNum(inBucket, 0)} tokens`;
    return {
      x,
      left: x + 230 > W ? x - 232 : x + 12,
      from: s0 + i * step,
      to: Math.min(s0 + (i + 1) * step, s1),
      rows,
      total: rows.reduce((a, r) => a + r.value, 0),
      tally: `${amount} in ${calls} LLM call${calls === 1 ? "" : "s"} this ${fmtNum(step / 60, 1)} min`,
    };
  });
</script>

<div class="chart" bind:clientWidth={width}>
  <svg
    bind:this={svg}
    viewBox="0 0 {W} {H}"
    width={W}
    height={H}
    role="img"
    aria-label="Token consumption, {mode === 'cumulative' ? 'running total' : unit}"
    data-testid="token-chart"
    onpointermove={move}
    onpointerleave={() => (hover = null)}
  >
    <defs>
      <clipPath id={clip}><rect x={LABEL} y="0" width={W - LABEL - R} height={H} /></clipPath>
    </defs>
    {#each bands as b (b.i)}
      <rect class="band" class:alt={b.i % 2 === 1} x={b.x} y={TOP} width={b.x2 - b.x} height={H - TOP - BOT} />
      {#if b.phase && b.x2 - b.x > b.phase.length * 6 + 10}
        <text class="phase" x={b.x + 4} y={TOP - 5}>{b.phase}</text>
      {/if}
    {/each}
    {#each timeTicks as t (t)}
      {@const x = X(t)}
      <line class="tick" x1={x} x2={x} y1={TOP} y2={H - BOT} />
      <text class="dim" x={x + 2} y={H - 6}>{x < W - 90 ? fmtTime(t) : ""}</text>
    {/each}
    {#each ticks as v (v)}
      <line class="grid" x1={LABEL} x2={W - R} y1={Y(v)} y2={Y(v)} />
      <text class="dim" x={LABEL - 6} y={Y(v) + 3} text-anchor="end">{fmtAxis(v)}</text>
    {/each}
    <text class="dim unit" transform="translate(11 {(H - BOT + TOP) / 2}) rotate(-90)" text-anchor="middle">
      {unit}
    </text>
    {#if empty}
      <text class="dim" x={LABEL + 8} y={TOP + 24}>no LLM calls yet</text>
    {:else}
      <g clip-path="url(#{clip})">
        {#if mode === "cumulative"}
          {#each st.layers as layer (layer.key)}
            <path class="area" d={area(layer.lo, layer.hi)} fill={layer.color} />
            <path class="edge" d={edge(layer.hi)} stroke={layer.color} />
          {/each}
        {:else}
          {#each bars as b (b.key)}
            <rect class="bar" x={b.x} y={b.y} width={b.w} height={b.h} fill={b.color} />
          {/each}
        {/if}
      </g>
    {/if}
    <line class="axis" x1={LABEL} x2={W - R} y1={H - BOT} y2={H - BOT} />
    {#if nowIn && now != null}
      <line class="now" x1={X(now)} x2={X(now)} y1="0" y2={H - BOT} data-testid="token-now">
        <title>now {fmtTime(now)}</title>
      </line>
    {/if}
    {#if readout}
      <line class="guide" x1={readout.x} x2={readout.x} y1={TOP} y2={H - BOT} />
    {/if}
  </svg>
  {#if readout}
    <div class="tip" style="left: {readout.left}px" data-testid="token-readout">
      <div class="when">{fmtTime(readout.from, s1)} – {fmtTime(readout.to, s1)}</div>
      {#each readout.rows as r (r.key)}
        <div class="row">
          <span class="sw" style="background: {r.color}"></span>
          <span class="name">{r.label}</span>
          <span class="v">{exact(r.value)}{mode === "rate" ? "/min" : ""}</span>
        </div>
      {/each}
      <div class="row sum">
        <span class="name">{mode === "cumulative" ? "running total" : "rate"}</span>
        <span class="v">{exact(readout.total)}{mode === "rate" ? "/min" : ""}</span>
      </div>
      <div class="muted">{readout.tally}</div>
    </div>
  {/if}
</div>

<style>
  .chart {
    position: relative;
    width: 100%;
  }
  svg {
    display: block;
    touch-action: none;
  }
  text {
    font-size: 10px;
  }
  .dim {
    fill: var(--muted);
  }
  .unit {
    font-style: italic;
  }
  .band {
    fill: var(--panel2);
    opacity: 0.55;
  }
  .band.alt {
    opacity: 0.25;
  }
  .phase {
    fill: var(--muted);
    font-size: 9px;
    letter-spacing: 0.04em;
  }
  .grid {
    stroke: var(--line);
    stroke-dasharray: 2 3;
  }
  .axis,
  .tick {
    stroke: var(--line);
  }
  .area {
    fill-opacity: 0.7;
  }
  .edge {
    fill: none;
    stroke-width: 1.4;
  }
  .bar {
    fill-opacity: 0.85;
  }
  .now {
    stroke: var(--crashed);
    stroke-width: 1.5;
  }
  .guide {
    stroke: var(--fg);
    stroke-opacity: 0.45;
    pointer-events: none;
  }
  .tip {
    position: absolute;
    top: 14px;
    width: 220px;
    padding: 0.4rem 0.5rem;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 6px;
    box-shadow: 0 4px 14px rgb(0 0 0 / 0.25);
    font-size: 11px;
    pointer-events: none;
  }
  .when {
    color: var(--muted);
    margin-bottom: 0.2rem;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 0.35rem;
  }
  .row.sum {
    border-top: 1px solid var(--line);
    margin-top: 0.2rem;
    padding-top: 0.2rem;
    font-weight: 600;
  }
  .sw {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    flex: none;
  }
  .name {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .v {
    font-variant-numeric: tabular-nums;
  }
</style>
