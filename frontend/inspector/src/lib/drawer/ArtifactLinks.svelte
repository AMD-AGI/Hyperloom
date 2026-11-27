<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<!-- A producer's or task's files; a click opens one in the Artifacts tab and scrolls it into view. -->
<script lang="ts">
  import { prefersReducedMotion } from "svelte/motion";
  import { fmtBytes } from "../format";
  import type { Artifact } from "../types";
  import type { UiState } from "../ui.svelte";
  import { commonDir, DRAWER_ARTIFACTS } from "./related";

  let { list, ui }: { list: Artifact[]; ui: UiState } = $props();
  const shown = $derived(list.slice(0, DRAWER_ARTIFACTS));
  const dir = $derived(commonDir(shown.map((a) => a.path)));

  function open(path: string): void {
    ui.openArtifact(path);
    const pane = document.querySelector("[data-testid=tabs]");
    pane?.scrollIntoView({ behavior: prefersReducedMotion.current ? "auto" : "smooth", block: "start" });
  }
</script>

{#if !list.length}
  <p class="muted">none</p>
{:else}
  {#if dir}<p class="muted mono" data-testid="dr-artifact-dir">in {dir}</p>{/if}
  {#each shown as a (a.path)}
    <div class="file" data-testid="dr-artifact">
      <span class="type">{a.type}</span>
      <button class="link mono" title={a.path} onclick={() => open(a.path)}>{a.path.slice(dir.length)}</button>
      <span class="muted">{fmtBytes(a.size)}</span>
    </div>
  {/each}
  {#if list.length > DRAWER_ARTIFACTS}
    <p class="muted">… {list.length - DRAWER_ARTIFACTS} more in the Artifacts tab</p>
  {/if}
{/if}

<style>
  .file {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) auto;
    gap: 0.5rem;
    align-items: baseline;
    font-size: 11.5px;
    padding: 1px 0;
  }
  .file button {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .type {
    font-size: 10px;
    color: var(--muted);
    border: 1px solid var(--line);
    border-radius: 3px;
    padding: 0 4px;
  }
  p {
    margin: 0.2rem 0;
    font-size: 12px;
  }
</style>
