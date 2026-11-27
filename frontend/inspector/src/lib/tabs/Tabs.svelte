<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<!-- The tab bar, with an indicator that slides to the open tab, and the tabs, which fade in. -->
<script lang="ts">
  import { fade } from "svelte/transition";
  import { ms } from "../motion";
  import type { InspectorStore } from "../store.svelte";
  import { TABS, type Tab, type UiState } from "../ui.svelte";
  import ArtifactsTab from "./ArtifactsTab.svelte";
  import BusTab from "./BusTab.svelte";
  import JournalTab from "./JournalTab.svelte";
  import LlmTab from "./LlmTab.svelte";
  import TasksTab from "./TasksTab.svelte";
  import TokensTab from "./TokensTab.svelte";

  let { store, ui }: { store: InspectorStore; ui: UiState } = $props();

  const NAMES: Record<Tab, string> = {
    tasks: "Tasks",
    artifacts: "Artifacts",
    bus: "Bus",
    llm: "LLM",
    tokens: "Tokens",
    journal: "Throughput",
  };
  const counts = $derived<Record<Tab, number>>({
    tasks: store.view?.tasks.length ?? 0,
    artifacts: store.artifacts.length,
    bus: store.streams.event.length,
    llm: store.streams.llm_call.length,
    tokens: store.view?.tokens.totals.calls ?? 0,
    journal: store.streams.journal.length,
  });

  // A "+" once the artifact list stops at the index's cap.
  const cut = $derived(store.view?.artifacts_cut_at ?? null);
  const shown = (t: Tab) =>
    counts[t].toLocaleString() + (t === "artifacts" && cut != null && counts[t] >= cut ? "+" : "");

  const buttons: Partial<Record<Tab, HTMLButtonElement>> = $state({});
  const ind = $derived.by(() => {
    void counts; // the counts change the button widths
    const b = buttons[ui.tab];
    return b ? { x: b.offsetLeft, w: b.offsetWidth } : { x: 0, w: 0 };
  });
</script>

<section class="tabs" data-testid="tabs">
  <nav>
    {#each TABS as t (t)}
      <button
        bind:this={buttons[t]}
        class:on={ui.tab === t}
        onclick={() => (ui.tab = t)}
        data-testid="tab-{t}"
        aria-pressed={ui.tab === t}
      >
        {NAMES[t]} <span class="n">{shown(t)}</span>
      </button>
    {/each}
    <span class="ind" style="transform: translateX({ind.x}px); width: {ind.w}px"></span>
    {#if ui.filter}<span class="muted filter">filtered to {ui.filter}</span>{/if}
  </nav>
  {#key ui.tab}
    <div class="pane" data-testid="pane-{ui.tab}" in:fade={{ duration: ms(180) }}>
      {#if ui.tab === "tasks"}
        <TasksTab {store} {ui} />
      {:else if ui.tab === "artifacts"}
        <ArtifactsTab {store} {ui} />
      {:else if ui.tab === "bus"}
        <BusTab {store} {ui} />
      {:else if ui.tab === "llm"}
        <LlmTab {store} {ui} />
      {:else if ui.tab === "tokens"}
        <TokensTab {store} {ui} />
      {:else}
        <JournalTab {store} {ui} />
      {/if}
    </div>
  {/key}
</section>

<style>
  .tabs {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    min-width: 0;
  }
  nav {
    position: relative;
    display: flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0 0.5rem;
    border-bottom: 1px solid var(--line);
  }
  button {
    font: inherit;
    color: var(--muted);
    background: none;
    border: 0;
    padding: 0.5rem 0.7rem;
    cursor: pointer;
    transition: color 0.2s ease;
  }
  button.on {
    color: var(--fg);
  }
  .n {
    font-size: 11px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }
  .ind {
    position: absolute;
    left: 0;
    bottom: -1px;
    height: 2px;
    background: var(--accent);
    border-radius: 2px;
    transition:
      transform 0.3s cubic-bezier(0.3, 1.2, 0.5, 1),
      width 0.3s ease;
  }
  .filter {
    margin-left: auto;
    font-size: 12px;
  }
  .pane {
    padding: 0.5rem 0.75rem 0.75rem;
    max-height: 40rem;
    overflow: auto;
  }
</style>
