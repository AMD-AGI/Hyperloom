<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<!--
  Replay controls: play/pause, speed and a scrubber. Between server updates the scrubber and time follow the
  interpolated clock (`sessionClock`); dragging previews a time and seeks on release. From the end, play starts over.
  The scrubber is drawn as one segment per phase of the recording, in the phase's colour, and hovering it names the
  phase under the cursor. A click seeks to the start of the phase under it; a drag seeks to the exact time.
-->
<script lang="ts">
  import { fmtDur, fmtTime } from "./format";
  import { sessionClock } from "./now.svelte";
  import { phaseColor } from "./phase";
  import {
    atEnd,
    chapterAtPx,
    chapters,
    CLICK_SLOP,
    playAction,
    progress,
    seekTarget,
    segments,
    speedOptions,
  } from "./replay";
  import type { InspectorStore } from "./store.svelte";

  let { store }: { store: InspectorStore } = $props();

  /** Half the thumb's width: the thumb's centre, and so the segments, stop this far inside the track's ends. */
  const INSET = 6;

  const clock = $derived(store.view?.clock);
  const clockNow = sessionClock(() => store);
  const now = $derived(clockNow.current);
  let drag = $state<number | null>(null);
  let busy = $state(false);
  /** The pointer's x on the segments, in pixels, while it is over the scrubber. */
  let hover = $state<number | null>(null);
  /** Where the current press started and how far it has moved; null between presses and for the keyboard. */
  let press: { x: number; moved: number } | null = null;
  let px = $state(0);

  const shown = $derived(drag ?? now);
  const ended = $derived(!!clock && atEnd(clock, now));
  const playing = $derived(!!clock && !clock.paused && !ended);
  const p = $derived(clock ? progress(clock, shown) : 0);
  const chs = $derived(clock ? chapters(clock, store.view?.timeline.chapters ?? []) : []);
  const segs = $derived(segments(chs, px));
  const hovered = $derived(hover == null ? undefined : chapterAtPx(chs, hover, px));

  async function send(ctl: Parameters<InspectorStore["replay"]>[0]) {
    busy = true;
    try {
      await store.replay(ctl);
    } finally {
      busy = false;
    }
  }
  const toggle = () => clock && send(playAction(clock, now));
  function seek(e: Event) {
    const t = Number((e.currentTarget as HTMLInputElement).value);
    const click = press && press.moved < CLICK_SLOP ? { x: press.x, px } : null;
    press = null;
    drag = null;
    void send({ t: seekTarget(chs, t, click) });
  }
  /** The pointer's x on the segments, which stop INSET inside the input at both ends. */
  function at(e: PointerEvent): number {
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    return Math.min(px, Math.max(0, e.clientX - r.left - INSET));
  }
  function point(e: PointerEvent) {
    hover = at(e);
    if (press) press.moved = Math.max(press.moved, Math.abs(hover - press.x));
  }
</script>

{#if clock?.mode === "replay" && clock.start != null && clock.end != null}
  <div class="bar" data-testid="replaybar">
    <button
      class="play"
      data-testid="rp-play"
      disabled={busy}
      aria-label={playing ? "Pause" : ended ? "Replay from the start" : "Play"}
      title={playing ? "Pause" : ended ? "Replay from the start" : "Play"}
      onclick={toggle}
    >
      {#if playing}
        <svg viewBox="0 0 16 16" aria-hidden="true"
          ><rect x="3" y="2" width="3.5" height="12" /><rect x="9.5" y="2" width="3.5" height="12" /></svg
        >
      {:else if ended}
        <svg viewBox="0 0 16 16" aria-hidden="true"
          ><path d="M8 2a6 6 0 1 1-5.2 3H1l3-3.5L7 5H4.6A4.5 4.5 0 1 0 8 3.5z" /></svg
        >
      {:else}
        <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2l10 6-10 6z" /></svg>
      {/if}
    </button>
    <select
      data-testid="rp-speed"
      aria-label="Replay speed"
      value={clock.speed}
      disabled={busy}
      onchange={(e) => send({ speed: Number(e.currentTarget.value) })}
    >
      {#each speedOptions(clock.speed) as sp (sp)}<option value={sp}>×{sp}</option>{/each}
    </select>
    <span class="mono muted">{fmtTime(clock.start)}</span>
    <div class="track" style:--inset="{INSET}px">
      <div class="chapters" aria-hidden="true" bind:clientWidth={px}>
        {#each chs as c, i (c.start)}
          <i
            data-testid="rp-segment"
            data-phase={c.phase}
            class:hot={c === hovered}
            class:widened={segs[i].widened}
            style:left="{segs[i].left}px"
            style:width="{segs[i].width}px"
            style:--c={phaseColor(c.phase)}
            style:--f="{Math.min(1, Math.max(0, (p - c.from) / (c.to - c.from))) * 100}%"
          ></i>
        {/each}
      </div>
      <input
        type="range"
        data-testid="rp-scrub"
        aria-label="Replay position"
        min={clock.start}
        max={clock.end}
        step="any"
        value={shown}
        oninput={(e) => (drag = Number(e.currentTarget.value))}
        onchange={seek}
        onkeydown={() => (press = null)}
        onpointerdown={(e) => (press = { x: at(e), moved: 0 })}
        onpointermove={point}
        onpointerleave={() => (hover = null)}
      />
      {#if hover != null && hovered?.phase}
        <div class="tip" data-testid="rp-chapter" style:left="{hover + INSET}px">
          <b>{hovered.phase}</b>
          <span class="mono muted">{fmtTime(hovered.start)} · {fmtDur(hovered.end - hovered.start)}</span>
          {#if hovered.reason}<div class="muted">{hovered.reason}</div>{/if}
        </div>
      {/if}
    </div>
    <span class="mono muted">{fmtTime(clock.end)}</span>
    <span class="mono now" data-testid="rp-now" class:drag={drag != null}
      >{fmtTime(shown)} <span class="muted">+{fmtDur(shown - clock.start)}</span></span
    >
  </div>
{/if}

<style>
  .bar {
    position: sticky;
    top: 0;
    z-index: 20;
    display: flex;
    gap: 10px;
    align-items: center;
    padding: 6px 12px;
    background: var(--panel2);
    border-bottom: 1px solid var(--line);
  }
  .play {
    display: grid;
    place-items: center;
    width: 30px;
    height: 26px;
    padding: 0;
    border: 1px solid var(--line);
    border-radius: 5px;
    background: var(--panel);
    color: var(--accent);
    cursor: pointer;
  }
  .play svg {
    width: 14px;
    height: 14px;
    fill: currentColor;
  }
  select {
    background: var(--panel);
    color: var(--fg);
    border: 1px solid var(--line);
    border-radius: 4px;
  }
  .track {
    position: relative;
    flex: 1;
    min-width: 8rem;
  }
  .chapters {
    position: absolute;
    inset: 0 var(--inset);
    z-index: 0;
    pointer-events: none;
  }
  /* Each segment is its phase colour --c, at full strength up to --f (played) and dimmed after; segments() leaves
     the gaps between them. */
  .chapters i {
    position: absolute;
    top: 50%;
    height: 6px;
    border-radius: 3px;
    transform: translateY(-50%);
    background: linear-gradient(to right, var(--c) var(--f), color-mix(in srgb, var(--c) 30%, transparent) var(--f));
    transition: height 0.12s;
  }
  .chapters i.widened {
    z-index: 1;
    box-shadow: 0 0 0 1px var(--panel2);
  }
  .chapters i.hot {
    height: 10px;
  }
  input[type="range"] {
    position: relative;
    display: block;
    width: 100%;
    height: 16px;
    margin: 0;
    appearance: none;
    background: transparent;
    cursor: pointer;
  }
  input[type="range"]::-webkit-slider-runnable-track {
    height: 16px;
  }
  input[type="range"]::-webkit-slider-thumb {
    appearance: none;
    width: calc(2 * var(--inset));
    height: calc(2 * var(--inset));
    margin-top: calc(8px - var(--inset));
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 0 2px var(--panel2);
  }
  input[type="range"]::-moz-range-track {
    background: transparent;
  }
  input[type="range"]::-moz-range-thumb {
    width: calc(2 * var(--inset));
    height: calc(2 * var(--inset));
    border: none;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 0 2px var(--panel2);
  }
  .tip {
    position: absolute;
    top: calc(100% + 6px);
    transform: translateX(-50%);
    padding: 4px 8px;
    white-space: nowrap;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 5px;
    box-shadow: 0 2px 8px rgb(0 0 0 / 0.25);
    pointer-events: none;
  }
  .now {
    min-width: 11rem;
  }
  .now.drag {
    color: var(--accent);
  }
</style>
