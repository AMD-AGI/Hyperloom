<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<!-- Every session file by producer and task. A large subtree is one entry whose rows are built only when opened. -->
<script lang="ts">
  import { slide } from "svelte/transition";
  import { fmtBytes, fmtTime } from "../format";
  import { ms } from "../motion";
  import type { InspectorStore } from "../store.svelte";
  import type { Artifact } from "../types";
  import type { UiState } from "../ui.svelte";
  import { artifactTypes, BULK_ROWS, fileName, filterArtifacts, groupArtifacts } from "./rows";

  let { store, ui }: { store: InspectorStore; ui: UiState } = $props();

  const all = $derived(store.artifacts);
  const types = $derived(artifactTypes(all));
  const q = $derived(ui.artQuery);
  const list = $derived(filterArtifacts(all, { ...q, filter: ui.filter }));
  const groups = $derived(groupArtifacts(list));
  const narrowed = $derived(!!(ui.filter || q.q || q.type));
  const current = $derived(ui.artifact?.kind === "file" ? ui.artifact.path : null);

  // An entry is open when narrowed, or when the user flipped it from its default.
  const isOpen = (key: string, byDefault: boolean) => narrowed || ui.artToggled.has(key) !== byDefault;
  function toggled(e: Event, key: string, byDefault: boolean) {
    const open = (e.currentTarget as HTMLDetailsElement).open;
    if (open === isOpen(key, byDefault) || narrowed) return;
    if (ui.artToggled.has(key)) ui.artToggled.delete(key);
    else ui.artToggled.add(key);
  }
  const pick = (path: string, on: boolean) => (on ? ui.picked.add(path) : ui.picked.delete(path));
  const d = () => ({ duration: ms(160) });
</script>

{#snippet fileRow(a: Artifact)}
  <div class="file" class:sel={current === a.path} data-testid="art-file">
    <input
      type="checkbox"
      checked={ui.picked.has(a.path)}
      onchange={(e) => pick(a.path, e.currentTarget.checked)}
      aria-label="pick for diff"
    />
    <span class="type">{a.type}</span>
    <button
      class="link mono"
      title="{a.path}&#10;{fmtBytes(a.size)}, {fmtTime(a.mtime)}"
      onclick={() => ui.openArtifact(a.path)}>{fileName(a)}</button
    >
    <span class="muted size">{fmtBytes(a.size)}</span>
  </div>
{/snippet}

{#snippet files(list: Artifact[], trees: ReturnType<typeof groupArtifacts>[number]["loose"]["trees"])}
  {#each list as a (a.path)}{@render fileRow(a)}{/each}
  {#each trees as t (t.root)}
    {@const byDefault = !!q.q && t.files.length <= BULK_ROWS}
    {@const open = isOpen(t.root, byDefault)}
    <details class="bulk" {open} ontoggle={(e) => toggled(e, t.root, byDefault)} data-testid="art-bulk">
      <summary class="mono" title={t.root}>
        {t.name}/ <span class="muted">tree of {t.files.length} files, {fmtBytes(t.bytes)}</span>
      </summary>
      {#if open}
        <div transition:slide={d()}>
          {#each t.files.slice(0, BULK_ROWS) as a (a.path)}{@render fileRow(a)}{/each}
          {#if t.files.length > BULK_ROWS}
            <div class="muted file">
              … {t.files.length - BULK_ROWS} more; type part of a path in the search box to narrow
            </div>
          {/if}
        </div>
      {/if}
    </details>
  {/each}
{/snippet}

<div class="bar">
  <input type="search" placeholder="filter paths" bind:value={ui.artQuery.q} data-testid="art-q" />
  <select bind:value={ui.artQuery.type} data-testid="art-type">
    <option value="">all types</option>
    {#each types as t (t)}<option value={t}>{t}</option>{/each}
  </select>
  <button disabled={ui.picked.size !== 2} onclick={() => ui.openDiff()} data-testid="art-diff">diff</button>
  {#if ui.picked.size}
    <button class="ghost" onclick={() => ui.picked.clear()}>clear {ui.picked.size}</button>
  {/if}
</div>
<p class="muted" data-testid="art-count">{list.length} of {all.length} file(s)</p>
<div class="tree" data-testid="art-tree">
  {#each groups as g (g.producer)}
    {@const byDefault = list.length < 60}
    {@const open = isOpen(g.producer, byDefault)}
    <details {open} ontoggle={(e) => toggled(e, g.producer, byDefault)}>
      <summary>{g.producer} <span class="muted">({g.count})</span></summary>
      {#if open}
        <div class="kids" transition:slide={d()}>
          {@render files(g.loose.files, g.loose.trees)}
          {#each g.tasks as t (t.task)}
            {@const key = g.producer + "/" + t.task}
            {@const tOpen = isOpen(key, false)}
            <details open={tOpen} ontoggle={(e) => toggled(e, key, false)}>
              <summary class="mono">
                {t.task}
                <span class="muted"
                  >({t.files.length}{t.trees.length
                    ? ` + ${t.trees.length} tree${t.trees.length > 1 ? "s" : ""}`
                    : ""})</span
                >
              </summary>
              {#if tOpen}
                <div class="kids" transition:slide={d()}>{@render files(t.files, t.trees)}</div>
              {/if}
            </details>
          {/each}
        </div>
      {/if}
    </details>
  {/each}
</div>

<style>
  .bar {
    display: flex;
    gap: 0.4rem;
    align-items: center;
  }
  input[type="search"] {
    flex: 1;
    min-width: 0;
  }
  input[type="search"],
  select,
  button:not(.link) {
    font: inherit;
    font-size: 12px;
    color: var(--fg);
    background: var(--panel2);
    border: 1px solid var(--line);
    border-radius: 5px;
    padding: 3px 8px;
  }
  button:not(.link) {
    cursor: pointer;
  }
  button:disabled {
    opacity: 0.5;
    cursor: default;
  }
  button.ghost {
    background: none;
    color: var(--muted);
  }
  p {
    margin: 0.4rem 0;
    font-size: 12px;
  }
  .tree {
    font-size: 12px;
  }
  summary {
    cursor: pointer;
    padding: 2px 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .kids {
    margin-left: 14px;
  }
  .bulk .file {
    margin-left: 14px;
  }
  .file {
    display: flex;
    gap: 0.4rem;
    align-items: center;
    padding: 1px 4px;
    border-radius: 4px;
    white-space: nowrap;
    transition: background 0.2s ease;
  }
  .file.sel {
    background: color-mix(in srgb, var(--accent) 16%, transparent);
  }
  .type {
    font-size: 10px;
    color: var(--muted);
    border: 1px solid var(--line);
    border-radius: 3px;
    padding: 0 4px;
    min-width: 3.5em;
    text-align: center;
  }
  .link {
    font-size: 11.5px;
    color: var(--accent);
    background: none;
    border: 0;
    padding: 0;
    cursor: pointer;
    overflow: hidden;
    text-overflow: ellipsis;
    text-align: left;
  }
  .size {
    margin-left: auto;
    padding-left: 0.5rem;
  }
</style>
