<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<script lang="ts">
  import { fmtTime } from "../format";
  import type { InspectorStore } from "../store.svelte";
  import type { Task } from "../types";
  import type { UiState } from "../ui.svelte";
  import DataTable from "./DataTable.svelte";
  import { roundKey, roundRows, stepText, taskRows, type Col, type Round } from "./rows";

  let { store, ui }: { store: InspectorStore; ui: UiState } = $props();

  const view = $derived(store.view!);
  const tasks = $derived(taskRows(view.tasks, ui.filter));
  const rounds = $derived(roundRows(view.summary.specialist_rounds as Round[] | undefined, view.tasks, ui.filter));
  const at = (iso: unknown) => (iso ? fmtTime(Date.parse(String(iso)) / 1000) : "");

  const cols: Col<Task>[] = [
    { head: "task", text: (t) => t.task_id, cls: "mono" },
    { head: "node", text: (t) => t.label },
    { head: "kind", text: (t) => t.kind },
    { head: "health", text: (t) => t.health, cls: (t) => "h-" + t.health },
    { head: "state", text: (t) => t.state },
    { head: "now", text: (t) => stepText(t.progress?.current as Record<string, unknown> | undefined), cls: "trunc" },
    { head: "created", text: (t) => at(t.created_at) },
    { head: "updated", text: (t) => at(t.updated_at) },
    { head: "why", text: (t) => t.reason, cls: "trunc" },
  ];
  const roundCols: Col<Round>[] = [
    { head: "domain", text: (r) => r.domain },
    { head: "task", text: (r) => r.task_id, cls: "mono" },
    {
      head: "reason",
      text: (r) => r.reason,
      cls: (r) => (/exit_code:[1-9]/.test(String(r.reason)) ? "h-crashed" : ""),
    },
    { head: "proposals", text: (r) => r.proposals_total },
    { head: "completed", text: (r) => at(r.completed_at) },
    { head: "summary", text: (r) => r.summary, cls: "trunc" },
  ];
  const hasTask = (id: unknown) => view.tasks.some((t) => t.task_id === id);
</script>

<p class="muted">{tasks.length} task(s), newest first</p>
<DataTable
  testid="tasks-table"
  rows={tasks}
  {cols}
  key={(t) => t.task_id}
  onpick={(t) => ui.select({ kind: "task", id: t.task_id })}
  selected={(t) => ui.selection?.kind === "task" && ui.selection.id === t.task_id}
/>
{#if rounds.length}
  <h4 class="muted">Specialist rounds (state.json)</h4>
  <DataTable
    testid="rounds-table"
    rows={rounds}
    cols={roundCols}
    key={roundKey}
    onpick={(r) =>
      hasTask(r.task_id)
        ? ui.select({ kind: "task", id: String(r.task_id) })
        : ui.select({ kind: "raw", title: "Specialist round", data: r })}
  />
{/if}

<style>
  p,
  h4 {
    margin: 0.4rem 0;
    font-size: 12px;
  }
</style>
