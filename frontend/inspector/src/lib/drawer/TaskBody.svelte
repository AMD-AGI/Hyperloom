<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<script lang="ts">
  import { fmtNum, fmtTime } from "../format";
  import JsonTree from "../JsonTree.svelte";
  import type { InspectorStore } from "../store.svelte";
  import DataTable from "../tabs/DataTable.svelte";
  import { stepText, type Col } from "../tabs/rows";
  import { tsOf } from "../timeline/scale";
  import type { ProgressUnit, StreamItem, Task } from "../types";
  import type { UiState } from "../ui.svelte";
  import ArtifactLinks from "./ArtifactLinks.svelte";
  import { producerArtifacts, taskRelated } from "./related";
  import Section from "./Section.svelte";

  let { task, store, ui }: { task: Task; store: InspectorStore; ui: UiState } = $props();

  const rel = $derived(taskRelated(task.task_id, store.view!, store.streams));
  const files = $derived(producerArtifacts(store.artifacts, { task: task.task_id }));
  const current = $derived(task.progress?.current as (Record<string, unknown> & { ts?: number }) | undefined);
  const units = $derived(((task.progress?.units ?? []) as ProgressUnit[]).slice().reverse());

  const unitCols: Col<ProgressUnit>[] = [
    { head: "unit", text: (u) => u.label, cls: "mono" },
    { head: "status", text: (u) => u.status, cls: (u) => (u.status === "succeeded" ? "h-ok" : "h-" + u.status) },
    { head: "tok/s", text: (u) => fmtNum(u.output_throughput) },
    { head: "time", text: (u) => fmtTime(u.ts) },
    { head: "why", text: (u) => [u.stage, u.reason, u.error].filter(Boolean).join(" · "), cls: "trunc" },
  ];
  const eventCols: Col<StreamItem>[] = [
    { head: "time", text: (e) => fmtTime(tsOf(e)) },
    { head: "topic", text: (e) => e.topic },
    { head: "from", text: (e) => e.from_agent },
  ];
  const callCols: Col<StreamItem>[] = [
    { head: "time", text: (c) => fmtTime(tsOf(c)) },
    { head: "turn", text: (c) => c.turn },
    { head: "out", text: (c) => c.output_tokens },
    { head: "status", text: (c) => c.status, cls: (c) => (c.status === "ok" ? "h-ok" : "h-crashed") },
  ];
</script>

<p class="head">
  <b class="h-{task.health}">{task.health}</b>
  <span>{task.kind} · {task.state}</span>
  <button class="link" onclick={() => ui.follow({ kind: "node", id: task.node })}>{task.label}</button>
</p>
{#if task.reason}<p class="h-crashed">{task.reason}</p>{/if}
{#if current}
  <Section title="Progress ({units.length} done, {task.progress?.failed ?? 0} failed)" testid="dr-progress">
    <p class="muted">latest: {stepText(current)} at {fmtTime(current.ts)}</p>
    {#if units.length}
      <DataTable
        rows={units}
        cols={unitCols}
        key={(u) => `${u.label}@${u.ts}`}
        onpick={(u) => ui.follow({ kind: "raw", title: `Unit ${u.label}`, data: u })}
      />
    {/if}
  </Section>
{/if}
<Section title="Record" open={false} testid="dr-record"><JsonTree value={task} /></Section>
<Section title="Bus events about it ({rel.events.length})">
  <DataTable rows={rel.events} cols={eventCols} onpick={(e) => ui.follow({ kind: "event", item: e })} />
</Section>
{#if rel.calls.length}
  <Section title="LLM calls ({rel.calls.length})" open={false}>
    <DataTable rows={rel.calls} cols={callCols} onpick={(c) => ui.follow({ kind: "call", item: c })} />
  </Section>
{/if}
{#if rel.rounds.length}
  <Section title="Specialist round (state.json)"><JsonTree value={rel.rounds} /></Section>
{/if}
{#if rel.journal.length}
  <Section title="Journal ({rel.journal.length})"><JsonTree value={rel.journal} /></Section>
{/if}
<Section title="Artifacts ({files.length})" testid="dr-artifacts"><ArtifactLinks list={files} {ui} /></Section>

<style>
  .head {
    display: flex;
    gap: 0.6rem;
    align-items: baseline;
  }
  p {
    margin: 0.3rem 0;
    font-size: 12.5px;
  }
</style>
