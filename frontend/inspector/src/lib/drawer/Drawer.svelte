<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<!-- The click-through drawer: the selected record (redacted by the server), the records related to it and its
     producer's artifacts. Links inside it keep a trail for the back button; Escape closes it. -->
<script lang="ts">
  import { cubicOut } from "svelte/easing";
  import { fade, fly } from "svelte/transition";
  import Conversation from "../Conversation.svelte";
  import { fmtTime } from "../format";
  import JsonTree from "../JsonTree.svelte";
  import { ms } from "../motion";
  import type { InspectorStore } from "../store.svelte";
  import { tsOf } from "../timeline/scale";
  import type { UiState } from "../ui.svelte";
  import EventBody from "./EventBody.svelte";
  import NodeBody from "./NodeBody.svelte";
  import Section from "./Section.svelte";
  import TaskBody from "./TaskBody.svelte";

  let { store, ui }: { store: InspectorStore; ui: UiState } = $props();

  const sel = $derived(ui.selection);
  const view = $derived(store.view);
  const task = $derived(sel?.kind === "task" ? (view?.tasks.find((t) => t.task_id === sel.id) ?? null) : null);
  const node = $derived(sel?.kind === "node" ? (view?.graph.nodes.find((n) => n.id === sel.id) ?? null) : null);
  const callTask = $derived(
    sel?.kind === "call" && sel.item.task_id ? view?.tasks.find((t) => t.task_id === sel.item.task_id) : undefined,
  );

  const title = $derived.by(() => {
    if (!sel) return "";
    if (sel.kind === "node") return `Node ${node?.label ?? sel.id}`;
    if (sel.kind === "task") return `Task ${sel.id}`;
    if (sel.kind === "event") return `Bus event #${sel.item.seq} ${sel.item.topic}`;
    if (sel.kind === "call") {
      const turn = sel.item.turn != null ? ` turn ${sel.item.turn}` : "";
      return `LLM call: ${sel.item.component ?? sel.item.role ?? ""}${turn}`;
    }
    return sel.title;
  });

  function onkeydown(e: KeyboardEvent): void {
    if (e.key === "Escape" && ui.selection) ui.select(null);
  }
</script>

<svelte:window {onkeydown} />

{#if sel && view}
  <aside
    class="drawer"
    data-testid="drawer"
    aria-label="details"
    transition:fly={{ x: 480, opacity: 1, duration: ms(280), easing: cubicOut }}
  >
    <header>
      {#if ui.trail.length}
        <button onclick={() => ui.back()} aria-label="back" data-testid="dr-back">←</button>
      {/if}
      <b class="title" data-testid="dr-title" {title}>{title}</b>
      <button onclick={() => ui.select(null)} aria-label="close" data-testid="dr-close">✕</button>
    </header>
    {#key sel}
      <div class="body" in:fade={{ duration: ms(160) }}>
        {#if sel.kind === "task"}
          {#if task}<TaskBody {task} {store} {ui} />{:else}<p class="muted">
              This task is not in the view at the current clock.
            </p>{/if}
        {:else if sel.kind === "node"}
          {#if node}<NodeBody {node} {store} {ui} />{:else}<p class="muted">
              This node is not in the view at the current clock.
            </p>{/if}
        {:else if sel.kind === "event"}
          <EventBody event={sel.item} {store} {ui} />
        {:else if sel.kind === "call"}
          {@const call = sel.item}
          <p>
            {fmtTime(tsOf(call))}
            {#if callTask}
              {@const id = callTask.task_id}
              · task <button class="link mono" onclick={() => ui.follow({ kind: "task", id })}>{id}</button>
            {/if}
          </p>
          <Section title="Record" open={false} testid="dr-record"><JsonTree value={call} /></Section>
          <Section title="Conversation"><Conversation {call} {ui} /></Section>
        {:else}
          <Section title="Record" testid="dr-record"><JsonTree value={sel.data} /></Section>
        {/if}
      </div>
    {/key}
  </aside>
{/if}

<style>
  .drawer {
    position: fixed;
    top: 0;
    right: 0;
    bottom: 0;
    width: min(42rem, 94vw);
    z-index: 50;
    display: flex;
    flex-direction: column;
    background: var(--panel);
    border-left: 1px solid var(--line);
    box-shadow: -12px 0 32px rgb(0 0 0 / 0.28);
  }
  header {
    display: flex;
    gap: 0.5rem;
    align-items: center;
    padding: 0.6rem 0.9rem;
    border-bottom: 1px solid var(--line);
  }
  .title {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  header button {
    font: inherit;
    color: var(--muted);
    background: none;
    border: 0;
    cursor: pointer;
    padding: 0 0.3rem;
  }
  header button:hover {
    color: var(--fg);
  }
  .body {
    flex: 1;
    overflow: auto;
    padding: 0.4rem 0.9rem 1rem;
  }
  p {
    margin: 0.3rem 0;
    font-size: 12.5px;
  }
</style>
