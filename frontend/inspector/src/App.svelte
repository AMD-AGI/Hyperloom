<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<script lang="ts">
  import { onDestroy } from "svelte";
  import Drawer from "./lib/drawer/Drawer.svelte";
  import { fmtArtifactCount } from "./lib/format";
  import Header from "./lib/Header.svelte";
  import ReplayBar from "./lib/ReplayBar.svelte";
  import Graph from "./lib/graph/Graph.svelte";
  import Tabs from "./lib/tabs/Tabs.svelte";
  import Timeline from "./lib/timeline/Timeline.svelte";
  import { InspectorStore } from "./lib/store.svelte";
  import { UiState } from "./lib/ui.svelte";

  const store = new InspectorStore();
  const ui = new UiState();
  void store.start();
  onDestroy(() => store.stop());

  const view = $derived(store.view);
</script>

<Header {store} />
<ReplayBar {store} />
<main>
  {#if store.error}
    <p class="muted">Could not load the snapshot: {store.error}</p>
  {:else if view}
    <p class="muted" data-testid="counts">
      {view.tasks.length} tasks · {store.streams.event.length} bus events · {store.streams.llm_call.length} LLM calls ·
      {fmtArtifactCount(store.artifacts.length, view.artifacts_cut_at)}
    </p>
    <Graph {store} {ui} />
    <Timeline {store} {ui} />
    <Tabs {store} {ui} />
  {/if}
</main>
<Drawer {store} {ui} />

<style>
  main {
    display: grid;
    gap: 0.75rem;
    padding: 1rem;
  }
  p {
    margin: 0;
  }
</style>
