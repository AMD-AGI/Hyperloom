<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<script lang="ts">
  import { onDestroy, untrack } from "svelte";
  import { prefersReducedMotion, Spring } from "svelte/motion";
  import { slide } from "svelte/transition";
  import { fmtDur, fmtTime } from "../format";
  import { ms } from "../motion";
  import { phaseColor } from "../phase";
  import type { InspectorStore } from "../store.svelte";
  import type { StreamItem, TimelineBar } from "../types";
  import type { Selection, UiState } from "../ui.svelte";
  import ResourceStrip from "./ResourceStrip.svelte";
  import {
    activityPoints,
    AXIS,
    axisTicks,
    callLane,
    eventLane,
    fitRange,
    GAP_PX,
    idleGaps,
    LABEL,
    LANE,
    lanes,
    podMarkers,
    podStarts,
    RES,
    tlScale,
    tsOf,
    zoomAbout,
    type Range,
  } from "./scale";

  let { store, ui }: { store: InspectorStore; ui: UiState } = $props();

  let width = $state(0);
  const W = $derived(Math.max(600, width || 1100));
  const view = $derived(store.view!);
  const replay = $derived(view.clock.mode === "replay");

  // Two clocks: `tick` (1 s) moves the fitted range and the gaps; `now` (every frame) moves the now line and the
  // running bars, so they glide instead of jumping.
  let tick = $state(untrack(() => store.now()));
  let now = $state(untrack(() => store.now()));
  const timer = setInterval(() => (tick = store.now()), 1000);
  let frame = requestAnimationFrame(function loop() {
    now = prefersReducedMotion.current ? tick : store.now();
    frame = requestAnimationFrame(loop);
  });
  onDestroy(() => {
    clearInterval(timer);
    cancelAnimationFrame(frame);
  });
  $effect(() => {
    void view.clock;
    tick = untrack(() => store.now());
  });

  // The range springs to its target: the user's zoom and pan, or the whole run.
  const user = $derived(ui.zoom?.range ?? null);
  let gapsOn = $state(true);
  const fit = $derived(fitRange(view.clock, view.timeline, tick));
  const target = $derived(user ?? fit);
  const range = new Spring<Range>(
    untrack(() => target),
    { stiffness: 0.12, damping: 0.8 },
  );
  $effect(() => {
    const t = target;
    untrack(() => void range.set(t, { instant: prefersReducedMotion.current || !!ui.zoom?.instant }));
  });

  const pods = $derived(podStarts(view.run, tick));
  const points = $derived(
    activityPoints(view.timeline, pods, [store.streams.event, store.streams.llm_call, store.streams.metrics]),
  );
  const r = $derived(range.current);
  const sc = $derived(tlScale(r.t0, r.t1, W, gapsOn ? idleGaps(points, r.t0, r.t1, tick) : []));
  const ticks = $derived(axisTicks(r, W, sc));
  const markers = $derived(podMarkers(pods, r, sc.X));

  const nodes = $derived(view.graph.nodes);
  const ids = $derived(lanes(nodes));
  const laneY = $derived(new Map(ids.map((id, i) => [id, AXIS + i * LANE])));
  const bottom = $derived(AXIS + ids.length * LANE);
  const H = $derived(bottom + 8 + RES + 16);
  const clampX = (t: number) => Math.max(LABEL, Math.min(W - 8, sc.X(t)));
  const inRange = (t: number) => t >= r.t0 && t <= r.t1;
  const faded = (id: string) => !!ui.filter && ui.filter !== id;

  const bar = (b: TimelineBar) => {
    const x1 = clampX(b.start);
    return { x1, x2: Math.max(x1 + 2, clampX(b.end ?? now)) };
  };
  const barTitle = (b: TimelineBar) =>
    `${b.label} ${b.task_id}\n${b.health} (${b.state})\n${fmtTime(b.start)} → ${b.end ? fmtTime(b.end) : "…"}` +
    ` (${fmtDur((b.end ?? now) - b.start)})${b.step ? "\nnow: " + b.step : ""}`;
  const tri = (x: number, y: number) => `M${x - 4},${y + LANE - 2}L${x + 4},${y + LANE - 2}L${x},${y + LANE - 9}z`;
  const selected = (item: unknown) =>
    !!ui.selection && "item" in ui.selection && ui.selection.item === (item as StreamItem);

  // Unset while the panel is folded.
  let svg = $state<SVGSVGElement>();
  const svgX = (clientX: number) => {
    const b = svg!.getBoundingClientRect();
    return ((clientX - b.left) * W) / b.width;
  };

  // Wheel zoom about the pointer; drag to pan; double-click to fit the whole run again.
  $effect(() => {
    if (!svg) return;
    const el = svg;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const base = user ?? range.target;
      const at = Math.max(r.t0, Math.min(r.t1, sc.T(svgX(e.clientX))));
      ui.setZoom(zoomAbout(base, at, e.deltaY > 0 ? 1.25 : 0.8));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  });
  let dragging = $state(false);
  // The click that ends a drag neither seeks nor selects.
  let dragged = false;
  let drag: { x: number; r: Range; T: (x: number) => number; moved: boolean } | null = null;
  function down(e: PointerEvent) {
    if (e.button !== 0) return;
    dragged = false;
    drag = { x: e.clientX, r: range.target, T: sc.T, moved: false };
  }
  function move(e: PointerEvent) {
    if (!drag || (!drag.moved && Math.abs(e.clientX - drag.x) < 3)) return;
    drag.moved = dragging = true;
    const dt = drag.T(svgX(e.clientX)) - drag.T(svgX(drag.x));
    ui.setZoom({ t0: drag.r.t0 - dt, t1: drag.r.t1 - dt }, true);
  }
  function up() {
    dragged = !!drag?.moved;
    drag = null;
    dragging = false;
  }
  const pick = (sel: Selection) => dragged || ui.select(sel);
  // A click on the axis seeks the replay.
  function seek(e: MouseEvent) {
    if (!replay || dragged) return;
    void store.replay({ t: sc.T(svgX(e.clientX)) });
  }
</script>

<svelte:window onpointermove={move} onpointerup={up} />

<section class="timeline" data-testid="timeline">
  <header class:closed={!ui.timelineOpen}>
    <button
      class="fold"
      aria-expanded={ui.timelineOpen}
      data-testid="tl-toggle"
      onclick={() => ui.setTimelineOpen(!ui.timelineOpen)}
    >
      <span class="caret" class:open={ui.timelineOpen}>▸</span>
      <h2>Timeline</h2>
    </button>
    {#if ui.timelineOpen}
      <label class="muted"
        ><input type="checkbox" bind:checked={gapsOn} data-testid="tl-gaps" /> compress idle gaps</label
      >
      <span class="muted hint">wheel to zoom · drag to pan · double-click to fit</span>
      {#if user}
        <button data-testid="tl-fit" onclick={() => ui.setZoom(null)}>fit</button>
      {/if}
    {:else}
      <span class="muted peek" data-testid="tl-peek">
        {view.summary.phase ?? ""} · {view.timeline.bars.length} tasks
      </span>
    {/if}
  </header>
  {#if ui.timelineOpen}
    <div bind:clientWidth={width} transition:slide={{ duration: ms(180) }}>
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <svg
        bind:this={svg}
        viewBox="0 0 {W} {H}"
        style:height="{H}px"
        class:dragging
        onpointerdown={down}
        ondblclick={() => ui.setZoom(null)}
        data-testid="timeline-svg"
        data-t0={r.t0}
        data-t1={r.t1}
      >
        {#each view.timeline.phases as p (p.phase + p.start)}
          {@const x1 = clampX(p.start)}
          {@const x2 = clampX(p.end ?? now)}
          {#if x2 > x1}
            <rect x={x1} y={AXIS} width={x2 - x1} height={bottom - AXIS} fill={phaseColor(p.phase)} opacity="0.1">
              <title>{p.phase} ({p.reason ?? ""})</title>
            </rect>
            <text x={x1 + 3} y={H - 4} class="dim">{x2 - x1 > 40 ? p.phase : ""}</text>
          {/if}
        {/each}
        {#each ticks as t (t)}
          {@const x = sc.X(t)}
          <line x1={x} x2={x} y1={AXIS - 4} y2={H - RES - 16} stroke="var(--line)" />
          <text x={x + 2} y="12" class="dim">{x < W - 90 ? fmtTime(t) : ""}</text>
        {/each}
        <!-- svelte-ignore a11y_click_events_have_key_events -->
        <rect
          class="axis"
          class:seek={replay}
          x={LABEL}
          y="0"
          width={W - LABEL}
          height={AXIS - 2}
          fill="transparent"
          data-testid="tl-axis"
          onclick={seek}
        />
        {#each ids as id, i (id)}
          {@const y = AXIS + i * LANE}
          {@const n = nodes.find((x) => x.id === id)}
          <g class="lane" class:faded={faded(id)}>
            <line x1="0" x2={W} y1={y + LANE} y2={y + LANE} stroke="var(--line)" stroke-dasharray="2 3" />
            <!-- svelte-ignore a11y_click_events_have_key_events -->
            <text
              x="4"
              y={y + 15}
              class="label"
              fill="var(--{n?.health ?? 'idle'})"
              data-testid="tl-lane"
              onclick={() => ui.toggleFilter(id)}>{n ? n.label.slice(0, 20) : id}</text
            >
          </g>
        {/each}
        {#each view.timeline.bars as b (b.task_id)}
          {@const y = laneY.get(b.node)}
          {#if y != null && b.start != null}
            {@const { x1, x2 } = bar(b)}
            <!-- svelte-ignore a11y_click_events_have_key_events -->
            <rect
              class="bar"
              class:sel={ui.selection?.kind === "task" && ui.selection.id === b.task_id}
              x={x1}
              y={y + 4}
              width={x2 - x1}
              height={LANE - 8}
              rx="3"
              fill="var(--{b.health})"
              opacity={faded(b.node) ? 0.3 : 0.85}
              data-testid="tl-bar"
              data-health={b.health}
              onclick={() => pick({ kind: "task", id: b.task_id })}
            >
              <title>{barTitle(b)}</title>
            </rect>
            {#if b.step && !b.end && x2 - x1 > 60}
              <text x={x1 + 4} y={y + LANE - 8} class="step">{b.step.slice(0, Math.floor((x2 - x1 - 8) / 6))}</text>
            {/if}
          {/if}
        {/each}
        {#each sc.gaps as g (g.a)}
          {@const x = sc.X(g.a)}
          <g class="gap" data-testid="tl-gap">
            <rect
              x={x + 2}
              y={AXIS - 4}
              width={GAP_PX - 4}
              height={bottom - AXIS + 4}
              fill="var(--panel)"
              opacity="0.85"
            />
            <path
              d="M{x + 6},{AXIS - 4}l-4,{bottom - AXIS + 4}M{x + GAP_PX - 2},{AXIS - 4}l-4,{bottom - AXIS + 4}"
              stroke="var(--muted)"
              fill="none"
            />
            <text x={x + GAP_PX / 2} y="12" text-anchor="middle" class="dim">≈</text>
            <title>idle {fmtDur(g.b - g.a)} compressed ({fmtTime(g.a)} → {fmtTime(g.b)})</title>
          </g>
        {/each}
        {#each markers as p (p.i)}
          <!-- svelte-ignore a11y_click_events_have_key_events -->
          <g class="pod" data-testid="tl-pod" onclick={() => pick({ kind: "raw", title: `Pod ${p.label}`, data: p })}>
            <line
              x1={p.x}
              x2={p.x}
              y1={AXIS - 4}
              y2={bottom}
              stroke="var(--stalled)"
              stroke-width="1.5"
              stroke-dasharray="4 3"
            />
            <text x={p.x + 3} y={AXIS + 9} fill="var(--stalled)">{p.crowded ? "" : p.label}</text>
            <title>{p.label}: {p.acquired_at} on {p.hostname || "?"} (pid {p.pid ?? "?"})</title>
          </g>
        {/each}
        {#each store.streams.event as e, i (i)}
          {@const t = tsOf(e)}
          {@const y = laneY.get(eventLane(nodes, e.from_agent))}
          {#if y != null && inRange(t)}
            <!-- svelte-ignore a11y_click_events_have_key_events -->
            <circle
              class="mark"
              class:sel={selected(e)}
              cx={sc.X(t)}
              cy={y + 5}
              r="3"
              fill="var(--muted)"
              data-testid="tl-event"
              onclick={() => pick({ kind: "event", item: e })}
            >
              <title>{e.topic}: {e.from_agent} → {e.to_agent}&#10;{fmtTime(t)}</title>
            </circle>
          {/if}
        {/each}
        {#each store.streams.llm_call as c, i (i)}
          {@const t = tsOf(c)}
          {@const y = laneY.get(callLane(view.tasks, c))}
          {#if y != null && inRange(t)}
            <!-- svelte-ignore a11y_click_events_have_key_events -->
            <path
              class="mark"
              class:sel={selected(c)}
              d={tri(sc.X(t), y)}
              fill={c.status === "ok" ? "var(--accent)" : "var(--crashed)"}
              data-testid="tl-call"
              onclick={() => pick({ kind: "call", item: c })}
            >
              <title
                >LLM {c.component}
                {c.model}&#10;{c.input_tokens}/{c.output_tokens} tok, {c.latency_ms} ms, {c.status}</title
              >
            </path>
          {/if}
        {/each}
        {#each view.timeline.steps ?? [] as st (st.id)}
          {@const y = laneY.get("coordinator")}
          {#if y != null && st.start != null && inRange(st.start)}
            <!-- svelte-ignore a11y_click_events_have_key_events -->
            <path
              class="mark"
              d={tri(sc.X(st.start), y)}
              fill="var(--stalled)"
              data-testid="tl-step"
              onclick={() => pick({ kind: "raw", title: `Lifecycle step ${st.id}`, data: st })}
            >
              <title>step {st.id} {st.kind} ({st.status})</title>
            </path>
          {/if}
        {/each}
        <ResourceStrip
          metrics={store.streams.metrics}
          range={r}
          X={sc.X}
          y0={bottom + 8}
          {W}
          context={{ replay, optimizerHost: view.run.optimizer?.host, ended: view.timeline.end != null }}
        />
        {#if inRange(now)}
          <line class="now" x1={sc.X(now)} x2={sc.X(now)} y1="0" y2={H} data-testid="tl-now">
            <title>now {fmtTime(now)}</title>
          </line>
        {/if}
      </svg>
    </div>
  {/if}
</section>

<style>
  .timeline {
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
    border: none;
    padding: 0;
    color: inherit;
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
  button {
    font: inherit;
    color: var(--accent);
    background: transparent;
    border: 1px solid var(--line);
    border-radius: 5px;
    padding: 1px 10px;
    cursor: pointer;
  }
  svg {
    display: block;
    width: 100%;
    cursor: grab;
    user-select: none;
    touch-action: none;
  }
  svg.dragging {
    cursor: grabbing;
  }
  text {
    font-size: 10px;
  }
  .dim {
    fill: var(--muted);
  }
  .axis.seek,
  .label,
  .bar,
  .mark,
  .pod {
    cursor: pointer;
  }
  .label {
    font-size: 11px;
    transition: fill 0.4s ease;
  }
  .lane {
    transition: opacity 0.3s ease;
  }
  .lane.faded {
    opacity: 0.4;
  }
  .bar {
    transition:
      fill 0.4s ease,
      opacity 0.3s ease;
  }
  .bar.sel,
  .mark.sel {
    stroke: var(--fg);
    stroke-width: 1.5;
  }
  .step {
    fill: var(--bg);
    pointer-events: none;
  }
  .mark {
    transform-box: fill-box;
    transform-origin: center;
    animation: pop 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
  }
  .now {
    stroke: var(--crashed);
    stroke-width: 1.5;
  }
  @keyframes pop {
    from {
      transform: scale(0);
      opacity: 0;
    }
  }
</style>
