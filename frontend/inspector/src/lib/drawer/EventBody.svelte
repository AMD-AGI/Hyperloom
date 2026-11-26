<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<script lang="ts">
  import { fmtTime } from "../format";
  import JsonTree from "../JsonTree.svelte";
  import type { InspectorStore } from "../store.svelte";
  import DataTable from "../tabs/DataTable.svelte";
  import type { Col } from "../tabs/rows";
  import { tsOf } from "../timeline/scale";
  import type { StreamItem } from "../types";
  import type { UiState } from "../ui.svelte";
  import { eventRelated } from "./related";
  import Section from "./Section.svelte";

  let { event, store, ui }: { event: StreamItem; store: InspectorStore; ui: UiState } = $props();

  const rel = $derived(eventRelated(event, store.view!, store.streams));
  const cols: Col<StreamItem>[] = [
    { head: "seq", text: (x) => x.seq },
    { head: "topic", text: (x) => x.topic },
    { head: "from", text: (x) => x.from_agent },
    { head: "time", text: (x) => fmtTime(tsOf(x)) },
  ];
</script>

<p>{event.from_agent} → {event.to_agent} at {fmtTime(tsOf(event))}</p>
{#if rel.task}
  {@const id = rel.task.task_id}
  <p>task <button class="link mono" onclick={() => ui.follow({ kind: "task", id })}>{id}</button></p>
{/if}
{#if rel.parent}
  {@const parent = rel.parent}
  <p>
    in reply to
    <button class="link" onclick={() => ui.follow({ kind: "event", item: parent })}>
      #{parent.seq}
      {parent.topic}
    </button>
  </p>
{/if}
<Section title="Record" testid="dr-record"><JsonTree value={event} /></Section>
{#if rel.replies.length}
  <Section title="Replies ({rel.replies.length})">
    <DataTable rows={rel.replies} {cols} onpick={(x) => ui.follow({ kind: "event", item: x })} />
  </Section>
{/if}
{#if rel.reviews.length}
  <Section title="Critic verdicts ({rel.reviews.length})" testid="dr-verdicts">
    <JsonTree value={rel.reviews} />
  </Section>
{/if}

<style>
  p {
    margin: 0.3rem 0;
    font-size: 12.5px;
  }
</style>
