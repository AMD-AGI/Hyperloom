<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<!-- The quantitative uplift strip of the uplift panel, on the timeline's time axis. -->
<script lang="ts">
  import { fmtNum, fmtTime } from "../format";
  import { LABEL, type Range } from "./scale";
  import { fmtPct, stairPath, STRIP_H, type Uplift, type UpliftPoint } from "./uplift";

  let {
    up,
    range,
    X,
    y0,
    W,
    tEnd,
    onpick,
  }: {
    up: Uplift;
    range: Range;
    X: (t: number) => number;
    y0: number;
    W: number;
    tEnd: number;
    onpick: (p: UpliftPoint) => void;
  } = $props();

  const all = $derived(up.points.flatMap((p) => [p.pct, ...p.earlierPct]));
  const lo = $derived(Math.min(0, ...all) - 4);
  const hi = $derived(Math.max(0, ...all) + 6);
  const Y = (p: number) => y0 + STRIP_H - 6 - ((p - lo) / (hi - lo || 1)) * (STRIP_H - 14);
  const inRange = (t: number) => t >= range.t0 && t <= range.t1;
  const line = $derived(stairPath(up.steps, X, Y, tEnd));
  const area = $derived(line && `${line}V${Y(0)}H${X(up.steps[0].t)}Z`);
  const ahead = $derived((up.provisional.at(-1)?.pct ?? 0) > up.bestPct);
  const fill = (p: UpliftPoint) => (p.verdict === "keep" ? "var(--ok)" : "var(--panel)");
  const stroke = (p: UpliftPoint) => (p.verdict === "revert" ? "var(--muted)" : "var(--ok)");
</script>

<g class="uplift" data-testid="uplift-a">
  <text x="4" y={y0 + 12} class="dim">throughput (higher is better)</text>
  {#if up.baseline == null}
    <text x="4" y={y0 + 32} class="big">—</text>
    <rect x={LABEL} y={y0} width={W - LABEL - 8} height={STRIP_H} fill="none" stroke="var(--line)" />
    <text x={LABEL + 6} y={y0 + STRIP_H / 2 + 4} class="dim">waiting for the baseline measurement</text>
  {:else}
    <text x="4" y={y0 + 34} class="big" fill="var(--fg)">{fmtPct(up.bestPct)}</text>
    <text x="4" y={y0 + 50} class="dim">{fmtNum(up.steps.at(-1)!.tput, 0)} / {fmtNum(up.baseline, 0)} tok/s</text>
    {#if up.validatedPct != null}
      <text x="4" y={y0 + 78} fill="var(--ok)">validated {fmtPct(up.validatedPct)}</text>
    {/if}
    {#if ahead}
      <text x="4" y={y0 + 92} fill="var(--ok)">pending → {fmtPct(up.provisional.at(-1)!.pct)}</text>
    {/if}

    <rect x={LABEL} y={y0} width={W - LABEL - 8} height={STRIP_H} fill="none" stroke="var(--line)" />
    <line x1={LABEL} x2={W - 8} y1={Y(0)} y2={Y(0)} class="base" />
    <text x={W - 12} y={Y(0) - 3} text-anchor="end" class="dim">baseline {fmtNum(up.baseline, 0)} tok/s</text>

    <path d={area} fill="var(--ok)" opacity="0.12" />
    {#if ahead}
      <path d={stairPath(up.provisional, X, Y, tEnd)} class="prov" />
    {/if}
    <path d={line} class="stairs" />

    {#each up.points as p (p.task_id + p.label)}
      {#if inRange(p.t)}
        {@const x = X(p.t)}
        {@const ys = [p.pct, ...p.earlierPct].map(Y)}
        <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
        <g class="pt" data-testid="uplift-point" data-verdict={p.verdict} onclick={() => onpick(p)}>
          <line x1={x} x2={x} y1={Math.min(...ys)} y2={Math.max(...ys)} stroke={stroke(p)} opacity="0.6" />
          {#each p.earlierPct as e (e)}
            <line x1={x - 3} x2={x + 3} y1={Y(e)} y2={Y(e)} stroke={stroke(p)} opacity="0.6" />
          {/each}
          <circle cx={x} cy={Y(p.pct)} r="4.5" fill={fill(p)} stroke={stroke(p)} stroke-width="1.5" />
          {#if p.verdict === "keep"}
            <text x={x + 6} y={Y(p.pct) - 5} fill="var(--ok)">{fmtPct(p.pct)}</text>
          {/if}
          <title
            >{p.label} ({p.verdict}) at {fmtTime(p.t)}&#10;{fmtNum(p.tput, 0)} tok/s, {fmtPct(p.pct)} vs baseline{p
              .earlierPct.length
              ? `\nwarm-up ${p.earlierPct.map((e) => fmtPct(e)).join(", ")}`
              : ""}{p.stepGainPct != null
              ? `\nstep gain ${fmtPct(p.stepGainPct)} on the stack it was tried on`
              : ""}</title
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
    font-size: 17px;
    font-weight: 600;
  }
  .base {
    stroke: var(--muted);
    stroke-dasharray: 2 3;
  }
  .stairs {
    fill: none;
    stroke: var(--ok);
    stroke-width: 2;
  }
  .prov {
    fill: none;
    stroke: var(--ok);
    stroke-width: 1.5;
    stroke-dasharray: 4 3;
  }
  .pt {
    cursor: pointer;
  }
</style>
