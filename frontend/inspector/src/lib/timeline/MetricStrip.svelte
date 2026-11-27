<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<!-- One per-variant metric (mean TTFT, accuracy) under the uplift strip, on the same time axis.
     The step line is the kept configuration's value; the optional band is the baseline's noise. -->
<script lang="ts">
  import { fmtTime } from "../format";
  import { LABEL, type Range } from "./scale";
  import { fmtPct, METRIC_H, stairPath, type MetricPoint, type MetricSeries } from "./uplift";

  let {
    s,
    title,
    fmt,
    higherIsBetter,
    delta,
    color,
    band = false,
    range,
    X,
    y0,
    W,
    tEnd,
    loading,
    onpick,
  }: {
    s: MetricSeries;
    title: string;
    fmt: (v: number) => string;
    higherIsBetter: boolean;
    /** How the change against the baseline reads: relative (%) or in points of the value itself. */
    delta: "relative" | "points";
    color: string;
    /** Shade baseline ± 2σ of a difference of two scores, from the baseline's standard error. */
    band?: boolean;
    range: Range;
    X: (t: number) => number;
    y0: number;
    W: number;
    tEnd: number;
    loading: boolean;
    onpick: (p: MetricPoint) => void;
  } = $props();

  const noise = $derived(band && s.baselineSe != null ? 2 * Math.SQRT2 * s.baselineSe : null);
  const all = $derived([
    ...s.points.flatMap((p) => [p.value, ...p.earlier]),
    ...(noise != null && s.baseline != null ? [s.baseline - noise, s.baseline + noise] : []),
  ]);
  const pad = $derived((Math.max(...all) - Math.min(...all)) * 0.15 || Math.abs(Math.max(...all)) * 0.05 || 1);
  const lo = $derived(Math.min(...all) - pad);
  const hi = $derived(Math.max(...all) + pad);
  const Y = (v: number) => y0 + METRIC_H - 6 - ((v - lo) / (hi - lo || 1)) * (METRIC_H - 16);
  const inRange = (t: number) => t >= range.t0 && t <= range.t1;
  const diff = (v: number) =>
    s.baseline == null
      ? null
      : delta === "relative"
        ? fmtPct((v / s.baseline - 1) * 100)
        : `${v >= s.baseline ? "+" : ""}${(v - s.baseline).toFixed(2)} pt`;
  const change = $derived(s.current != null ? diff(s.current) : null);
  const better = $derived(
    s.current == null || s.baseline == null ? null : higherIsBetter ? s.current >= s.baseline : s.current <= s.baseline,
  );
  const ahead = $derived(s.provisional.length > s.steps.length);
  const stroke = (p: MetricPoint) => (p.verdict === "revert" ? "var(--muted)" : color);
</script>

<g class="metric" data-testid="metric-strip">
  <text x="4" y={y0 + 12} class="dim">{title} ({higherIsBetter ? "higher" : "lower"} is better)</text>
  <rect x={LABEL} y={y0} width={W - LABEL - 8} height={METRIC_H} fill="none" stroke="var(--line)" />
  {#if !s.points.length}
    <text x={LABEL + 6} y={y0 + METRIC_H / 2 + 4} class="dim"
      >{loading ? "reading benchmark results…" : "no result for this metric yet"}</text
    >
  {:else}
    {#if s.current != null}
      <text x="4" y={y0 + 32} class="big">{fmt(s.current)}</text>
    {/if}
    {#if change != null}
      <text x="4" y={y0 + 48} fill={better ? "var(--ok)" : "var(--failed)"}>{change} vs baseline</text>
    {/if}
    {#if s.baseline != null}
      <text x="4" y={y0 + 62} class="dim"
        >baseline {fmt(s.baseline)}{noise != null ? ` ± ${noise.toFixed(2)}` : ""}</text
      >
      {#if noise != null}
        <rect
          x={LABEL}
          y={Y(s.baseline + noise)}
          width={W - LABEL - 8}
          height={Y(s.baseline - noise) - Y(s.baseline + noise)}
          class="band"
        >
          <title>within noise: baseline ± 2σ for a difference of two scores</title>
        </rect>
      {/if}
      <line x1={LABEL} x2={W - 8} y1={Y(s.baseline)} y2={Y(s.baseline)} class="base" />
    {/if}
    {#if ahead}
      <path d={stairPath(s.provisional, X, Y, tEnd)} class="prov" stroke={color} />
    {/if}
    <path d={stairPath(s.steps, X, Y, tEnd)} fill="none" stroke={color} stroke-width="2" />
    {#each s.points as p (p.label + p.t)}
      {#if inRange(p.t)}
        {@const x = X(p.t)}
        {@const ys = [p.value, ...p.earlier].map(Y)}
        <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
        <g class="pt" data-testid="metric-point" data-verdict={p.verdict} onclick={() => onpick(p)}>
          {#if p.earlier.length}
            <line x1={x} x2={x} y1={Math.min(...ys)} y2={Math.max(...ys)} stroke={stroke(p)} opacity="0.6" />
            {#each p.earlier as e (e)}
              <line x1={x - 3} x2={x + 3} y1={Y(e)} y2={Y(e)} stroke={stroke(p)} opacity="0.6" />
            {/each}
          {/if}
          <rect
            x={x - 4}
            y={Y(p.value) - 4}
            width="8"
            height="8"
            fill={p.verdict === "keep" ? stroke(p) : "var(--panel)"}
            stroke={stroke(p)}
            stroke-width="1.5"
          />
          <text x={x + 7} y={Y(p.value) + 3} fill={stroke(p)}>{fmt(p.value)}</text>
          <title
            >{p.label} ({p.verdict}) at {fmtTime(p.t)}&#10;{title}
            {fmt(p.value)}{p.se != null ? ` ± ${p.se.toFixed(2)}` : ""}{p.label !== "baseline" && diff(p.value)
              ? ` (${diff(p.value)} vs baseline)`
              : ""}{p.earlier.length ? `\nother rounds ${p.earlier.map(fmt).join(", ")}` : ""}</title
          >
        </g>
      {/if}
    {/each}
  {/if}
</g>

<style>
  text {
    font-size: 10px;
  }
  .dim {
    fill: var(--muted);
  }
  .big {
    font-size: 15px;
    font-weight: 600;
    fill: var(--fg);
  }
  .base {
    stroke: var(--muted);
    stroke-dasharray: 2 3;
  }
  .band {
    fill: var(--muted);
    opacity: 0.12;
  }
  .prov {
    fill: none;
    stroke-width: 1.5;
    stroke-dasharray: 4 3;
  }
  .pt {
    cursor: pointer;
  }
</style>
