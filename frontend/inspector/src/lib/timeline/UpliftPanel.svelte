<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<!-- Uplift vs baseline in its own foldable panel, on the timeline's time axis (range, zoom and compressed gaps) so
     its points sit under the bars that produced them. -->
<script lang="ts">
  import { onDestroy, untrack } from "svelte";
  import { SvelteMap } from "svelte/reactivity";
  import { slide } from "svelte/transition";
  import { fetchArtifact } from "../api";
  import { fmtNum, fmtTime } from "../format";
  import { ms } from "../motion";
  import { phaseColor } from "../phase";
  import type { InspectorStore } from "../store.svelte";
  import type { Selection, UiState } from "../ui.svelte";
  import { AXIS, axisTicks, fitRange, LABEL, tlScale, zoomAbout, type Range } from "./scale";
  import MetricStrip from "./MetricStrip.svelte";
  import UpliftStrip from "./UpliftStrip.svelte";
  import {
    accuracyResults,
    fmtPct,
    METRIC_H,
    METRIC_READERS,
    metricSeries,
    resultSources,
    STRIP_H,
    upliftSeries,
    type MetricResult,
    type ResultFile,
    type ResultSource,
  } from "./uplift";

  let { store, ui }: { store: InspectorStore; ui: UiState } = $props();

  let width = $state(0);
  const W = $derived(Math.max(600, width || 1100));
  const view = $derived(store.view!);

  let now = $state(untrack(() => store.now()));
  const timer = setInterval(() => (now = store.now()), 500);
  onDestroy(() => clearInterval(timer));
  $effect(() => {
    void view.clock;
    now = untrack(() => store.now());
  });

  const up = $derived(upliftSeries(view, store.streams.journal, now));
  const axis = $derived(ui.timeAxis ?? { ...fitRange(view.clock, view.timeline, now), gaps: [] });
  const r = $derived({ t0: axis.t0, t1: axis.t1 });
  const sc = $derived(tlScale(r.t0, r.t1, W, axis.gaps));
  const ticks = $derived(axisTicks(r, W, sc));
  const clampX = (t: number) => Math.max(LABEL, Math.min(W - 8, sc.X(t)));

  // Wheel zoom about the pointer, drag to pan, double-click to fit: the same zoom as the timeline's.
  let svg = $state<SVGSVGElement>();
  const svgX = (clientX: number) => {
    const b = svg!.getBoundingClientRect();
    return ((clientX - b.left) * W) / b.width;
  };
  $effect(() => {
    if (!svg) return;
    const el = svg;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const at = Math.max(r.t0, Math.min(r.t1, sc.T(svgX(e.clientX))));
      ui.setZoom(zoomAbout(ui.zoom?.range ?? r, at, e.deltaY > 0 ? 1.25 : 0.8));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  });
  let dragging = $state(false);
  // The click that ends a drag does not select.
  let dragged = false;
  let drag: { x: number; r: Range; T: (x: number) => number; moved: boolean } | null = null;
  function down(e: PointerEvent) {
    if (e.button !== 0) return;
    dragged = false;
    drag = { x: e.clientX, r: ui.zoom?.range ?? r, T: sc.T, moved: false };
  }
  function move(e: PointerEvent) {
    if (!drag || (!drag.moved && Math.abs(e.clientX - drag.x) < 3)) return;
    drag.moved = dragging = true;
    const dt = drag.T(svgX(e.clientX)) - drag.T(svgX(drag.x));
    ui.setZoom({ t0: drag.r.t0 - dt, t1: drag.r.t1 - dt }, true);
  }
  function release() {
    dragged = !!drag?.moved;
    drag = null;
    dragging = false;
  }
  const pick = (sel: Selection) => dragged || ui.select(sel);
  const H = AXIS + STRIP_H + 8 + (METRIC_H + 8) * 2;

  // Each result file is read once. Live, a file can be indexed while it is still being written, so a failed or
  // value-less read is dropped after a while and read again.
  const RETRY_MS = 15000;
  const results = new SvelteMap<string, (MetricResult & { file: ResultFile }) | null>();
  const sources = $derived(
    (Object.keys(METRIC_READERS) as ResultFile[]).flatMap((file) =>
      resultSources(store.artifacts, file).map((src): ResultSource & { file: ResultFile } => ({ ...src, file })),
    ),
  );
  const loading = $derived(sources.some((src) => !results.has(src.path)));
  let retry = $state(0);
  const retryLater = (path: string) =>
    setTimeout(() => {
      results.delete(path);
      retry++;
    }, RETRY_MS);
  $effect(() => {
    void retry;
    for (const src of sources) {
      if (untrack(() => results.has(src.path))) continue;
      results.set(src.path, null);
      fetchArtifact(src.path, { mode: "json" })
        .then((a) => {
          const { value, t, se } = METRIC_READERS[src.file]((a.data ?? {}) as Record<string, unknown>);
          if (Number.isFinite(value)) results.set(src.path, { ...src, value, t: t ?? a.mtime, se });
          else retryLater(src.path);
        })
        .catch(() => retryLater(src.path));
    }
  });
  const seriesOf = (file: ResultFile) =>
    metricSeries(
      up,
      [...results.values()].filter((x): x is MetricResult & { file: ResultFile } => x?.file === file),
      now,
    );
  const ttft = $derived(seriesOf("inferencex_result.json"));
  const acc = $derived(metricSeries(up, accuracyResults(view.summary.accuracy, up.baseT), now));
  const fmtMs = (v: number) => `${fmtNum(v, 0)} ms`;
  const fmtAcc = (v: number) => `${v.toFixed(2)}%`;

  const peek = $derived(
    up.baseline == null
      ? "waiting for the baseline"
      : [
          `${fmtPct(up.bestPct)} vs baseline`,
          `${fmtNum(up.steps.at(-1)!.tput, 0)} / ${fmtNum(up.baseline, 0)} tok/s`,
          up.validatedPct != null ? `validated ${fmtPct(up.validatedPct)}` : "",
          ttft.current != null && ttft.baseline
            ? `TTFT ${fmtNum(ttft.current, 0)} ms (${fmtPct((ttft.current / ttft.baseline - 1) * 100)})`
            : "",
          acc.current != null && acc.baseline != null
            ? `accuracy ${fmtAcc(acc.current)} (${acc.current >= acc.baseline ? "+" : ""}${(acc.current - acc.baseline).toFixed(2)} pt)`
            : "",
          `${up.points.length} variants measured`,
        ]
          .filter(Boolean)
          .join(" · "),
  );
</script>

<svelte:window onpointermove={move} onpointerup={release} />

<section class="uplift-panel" data-testid="uplift-panel">
  <header class:closed={!ui.upliftOpen}>
    <button
      class="fold"
      aria-expanded={ui.upliftOpen}
      data-testid="uplift-toggle"
      onclick={() => ui.setUpliftOpen(!ui.upliftOpen)}
    >
      <span class="caret" class:open={ui.upliftOpen}>▸</span>
      <h2>Uplift vs baseline</h2>
    </button>
    {#if ui.upliftOpen}
      <span class="muted hint">wheel to zoom · drag to pan · double-click to fit, shared with the timeline</span>
      {#if ui.zoom}
        <button class="fit" data-testid="uplift-fit" onclick={() => ui.setZoom(null)}>fit</button>
      {/if}
    {:else}
      <span class="muted peek" data-testid="uplift-peek">{peek}</span>
    {/if}
  </header>
  {#if ui.upliftOpen}
    <div bind:clientWidth={width} transition:slide={{ duration: ms(180) }}>
      <svg
        bind:this={svg}
        viewBox="0 0 {W} {H}"
        style:height="{H}px"
        class:dragging
        data-testid="uplift-svg"
        role="img"
        aria-label="throughput, mean TTFT and accuracy over time"
        onpointerdown={down}
        ondblclick={() => ui.setZoom(null)}
      >
        {#each view.timeline.phases as p (p.phase + p.start)}
          {@const x1 = clampX(p.start)}
          {@const x2 = clampX(p.end ?? now)}
          {#if x2 > x1}
            <rect x={x1} y={AXIS} width={x2 - x1} height={H - AXIS - 8} fill={phaseColor(p.phase)} opacity="0.1">
              <title>{p.phase} ({p.reason ?? ""})</title>
            </rect>
          {/if}
        {/each}
        {#each ticks as t (t)}
          {@const x = sc.X(t)}
          <line x1={x} x2={x} y1={AXIS - 4} y2={H - 8} stroke="var(--line)" />
          <text x={x + 2} y="12" class="dim">{x < W - 90 ? fmtTime(t) : ""}</text>
        {/each}
        <UpliftStrip
          {up}
          range={r}
          X={clampX}
          y0={AXIS}
          {W}
          tEnd={now}
          onpick={(p) => pick({ kind: "raw", title: `Uplift ${p.label}`, data: p })}
        />
        <MetricStrip
          s={ttft}
          title="mean TTFT"
          fmt={fmtMs}
          higherIsBetter={false}
          delta="relative"
          color="var(--queued)"
          range={r}
          X={clampX}
          y0={AXIS + STRIP_H + 8}
          {W}
          tEnd={now}
          {loading}
          onpick={(p) => pick({ kind: "raw", title: `Mean TTFT ${p.label}`, data: p })}
        />
        <MetricStrip
          s={acc}
          title="accuracy"
          fmt={fmtAcc}
          higherIsBetter
          delta="points"
          color="var(--running)"
          band
          range={r}
          X={clampX}
          y0={AXIS + STRIP_H + 8 + METRIC_H + 8}
          {W}
          tEnd={now}
          loading={false}
          onpick={(p) => pick({ kind: "raw", title: `Accuracy ${p.label}`, data: p })}
        />
        {#if now >= r.t0 && now <= r.t1}
          <line class="now" x1={sc.X(now)} x2={sc.X(now)} y1="0" y2={H}>
            <title>now {fmtTime(now)}</title>
          </line>
        {/if}
      </svg>
    </div>
  {/if}
</section>

<style>
  .uplift-panel {
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
    font-size: 12px;
  }
  header.closed {
    border-bottom: none;
  }
  h2 {
    margin: 0;
    font-size: 13px;
  }
  .hint {
    margin-left: auto;
  }
  button.fold {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font: inherit;
    color: inherit;
    background: transparent;
    border: none;
    padding: 0;
    cursor: pointer;
  }
  .caret {
    display: inline-block;
    color: var(--muted);
    transition: transform 0.15s;
  }
  .caret.open {
    transform: rotate(90deg);
  }
  .peek {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  svg {
    display: block;
    width: 100%;
    user-select: none;
    cursor: grab;
    touch-action: none;
  }
  svg.dragging {
    cursor: grabbing;
  }
  button.fit {
    font: inherit;
    color: var(--accent);
    background: transparent;
    border: 1px solid var(--line);
    border-radius: 5px;
    padding: 1px 10px;
    cursor: pointer;
  }
  text {
    font-size: 10px;
  }
  .dim {
    fill: var(--muted);
  }
  .now {
    stroke: var(--crashed);
    stroke-width: 1.5;
  }
</style>
