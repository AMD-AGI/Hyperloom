<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<script lang="ts">
  import { flip } from "svelte/animate";
  import { fly } from "svelte/transition";
  import { fmtDur, fmtNum, fmtTime } from "./format";
  import { ms } from "./motion";
  import { sessionClock } from "./now.svelte";
  import { theme } from "./theme.svelte";
  import RunInfo from "./RunInfo.svelte";
  import type { InspectorStore } from "./store.svelte";
  import Tweened from "./Tweened.svelte";

  let { store }: { store: InspectorStore } = $props();

  type Best = { action?: string; hot_tput?: number | null; source?: string };

  const view = $derived(store.view);
  const sm = $derived(view?.summary ?? {});
  const best = $derived((sm.current_best ?? {}) as Best);
  const clock = $derived(view?.clock);
  const mode = $derived(
    clock?.mode === "replay" ? `replay ×${clock.speed}${clock.paused ? " (paused)" : ""}` : clock ? "live" : "",
  );
  const bestTitle = $derived(
    best.source === "result"
      ? "state.json has no current best; highest succeeded task result"
      : best.source === "journal"
        ? "state.json has no current best; highest KEEP journal entry"
        : undefined,
  );
  const optimizer = $derived.by(() => {
    const o = view?.run.optimizer as { alive: boolean | null; reason?: string; pid?: number } | undefined;
    if (!o) return { text: "n/a in replay", cls: "muted" };
    if (o.alive === true) return { text: `alive (pid ${o.pid})`, cls: "h-ok" };
    if (o.alive === false) return { text: `not running (pid ${o.pid})`, cls: "h-crashed" };
    return { text: `unknown: ${o.reason ?? ""}`, cls: "muted" };
  });
  const badges = $derived([
    ...(view?.badges ?? []).map((b) => ({ level: b.level || "info", text: b.text })),
    ...(view?.agents ?? [])
      .filter((a) => a.stale)
      .map((a) => ({
        level: "warn",
        text: `${a.agent} quiet for ${fmtDur(a.age_sec)}`,
      })),
  ]);

  // The clock line runs on its own between server updates.
  const clockNow = sessionClock(() => store);
  const num = (n: number) => fmtNum(n);
</script>

{#snippet tput(v: unknown)}
  {#if typeof v === "number"}<Tweened value={v} format={num} /> tok/s{:else}–{/if}
{/snippet}

<header>
  <div class="row">
    <span class="title">Hyperloom Inspector</span>
    {#if mode}<span class="pill">{mode}</span>{/if}
    <span class="muted" data-testid="connection">{store.connection}</span>
    <span class="kv">session <b>{view?.run.session_id ?? "–"}</b></span>
    <span class="kv">model <b>{view?.run.model ?? sm.model_name ?? "–"}</b></span>
    <span class="kv">phase <b data-testid="phase">{sm.phase ?? "–"}</b></span>
    <span class="kv">tick <b>{sm.tick ?? "–"}</b></span>
    <span class="kv">baseline <b>{@render tput(sm.baseline_tput)}</b></span>
    <span class="kv" title={bestTitle}
      >best <b
        >{@render tput(best.hot_tput)}{best.action && typeof best.hot_tput === "number" ? ` (${best.action})` : ""}</b
      ></span
    >
    <span class="kv"
      >budget
      {#if view?.budget?.total_sec}
        <span class="bar"><i style:width="{Math.min(100, view.budget.pct || 0)}%"></i></span>
        <b><Tweened value={view.budget.used_sec} format={fmtDur} /> / {fmtDur(view.budget.total_sec)}</b>
      {:else}<b>–</b>{/if}
    </span>
    <span class="kv">optimizer <b class={optimizer.cls}>{optimizer.text}</b></span>
    <span class="kv">clock <b class="mono">{fmtTime(clockNow.current)}</b></span>
    <button
      class="theme"
      data-testid="theme"
      title="Colour theme: follow the system, light or dark (remembered in this browser)"
      onclick={() => theme.cycle()}>theme: {theme.current === "system" ? "auto" : theme.current}</button
    >
  </div>
  {#if badges.length}
    <div class="row badges" data-testid="badges">
      {#each badges as b (b.text)}
        <span class="badge {b.level}" in:fly={{ y: -6, duration: ms(250) }} animate:flip={{ duration: ms(250) }}
          >{b.text}</span
        >
      {/each}
    </div>
  {/if}
  {#if view}<RunInfo run={view.run} summary={sm} />{/if}
</header>

<style>
  header {
    padding: 8px 12px;
    background: var(--panel);
    border-bottom: 1px solid var(--line);
  }
  .row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 14px;
    align-items: center;
  }
  .badges {
    margin-top: 6px;
  }
  .title {
    font-weight: 600;
    font-size: 15px;
  }
  .kv {
    color: var(--muted);
  }
  .kv b {
    color: var(--fg);
    font-weight: 600;
  }
  .pill {
    padding: 1px 8px;
    border-radius: 10px;
    border: 1px solid var(--line);
    font-size: 12px;
  }
  .badge {
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 12px;
  }
  .badge.error {
    background: color-mix(in srgb, var(--crashed) 22%, transparent);
  }
  .badge.warn {
    background: color-mix(in srgb, var(--stalled) 22%, transparent);
  }
  .badge.info {
    background: color-mix(in srgb, var(--accent) 18%, transparent);
  }
  .bar {
    width: 120px;
    height: 8px;
    background: var(--panel2);
    border-radius: 4px;
    overflow: hidden;
    display: inline-block;
    vertical-align: middle;
  }
  .bar > i {
    display: block;
    height: 100%;
    background: var(--accent);
    transition: width 0.6s ease-out;
  }
  .theme {
    margin-left: auto;
    font: inherit;
    font-size: 12px;
    color: var(--muted);
    background: var(--panel2);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 1px 8px;
    cursor: pointer;
  }
</style>
