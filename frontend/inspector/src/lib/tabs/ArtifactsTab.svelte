<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<script lang="ts">
  import { fmtArtifactCount } from "../format";
  import type { InspectorStore } from "../store.svelte";
  import type { UiState } from "../ui.svelte";
  import ArtifactTree from "./ArtifactTree.svelte";
  import ArtifactViewer from "./ArtifactViewer.svelte";
  import DiffView from "./DiffView.svelte";

  let { store, ui }: { store: InspectorStore; ui: UiState } = $props();
  const a = $derived(ui.artifact);
</script>

{#if store.view?.artifacts_cut_at != null}
  <p class="muted" data-testid="art-cut">
    {fmtArtifactCount(store.artifacts.length, store.view.artifacts_cut_at)}: the session holds more files than the
    Inspector lists, and the ones past the cut are not shown.
  </p>
{/if}
<div class="split">
  <div class="left"><ArtifactTree {store} {ui} /></div>
  <div class="right">
    {#if !a}
      <p class="muted">Pick a file to view it, or tick two and press diff.</p>
    {:else if a.kind === "diff"}
      <DiffView a={a.a} b={a.b} />
    {:else}
      <ArtifactViewer path={a.path} offset={a.offset} {ui} />
    {/if}
  </div>
</div>

<style>
  .split {
    display: grid;
    grid-template-columns: minmax(280px, 2fr) minmax(0, 3fr);
    gap: 0.75rem;
    height: 32rem;
  }
  .left,
  .right {
    min-height: 0;
    overflow: auto;
  }
  .right {
    border-left: 1px solid var(--line);
    padding-left: 0.75rem;
  }
  p {
    margin: 0.4rem 0;
  }
</style>
