<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<script lang="ts">
  import { fmtNum, fmtTime } from "../format";
  import type { InspectorStore } from "../store.svelte";
  import { tsOf } from "../timeline/scale";
  import type { StreamItem } from "../types";
  import type { UiState } from "../ui.svelte";
  import DataTable from "./DataTable.svelte";
  import JournalChart from "./JournalChart.svelte";
  import { tputAxis } from "./tputAxis";
  import { journalRows, outcomeClass, variantKey, variantThroughputs, variantUnits, type Col, type Unit } from "./rows";

  let { store, ui }: { store: InspectorStore; ui: UiState } = $props();

  const view = $derived(store.view!);
  const rows = $derived(journalRows(store.streams.journal, view.tasks, ui.filter));
  const measured = $derived(variantThroughputs(view.tasks));
  const tputOf = (j: StreamItem) => (j.throughput as number | null) ?? measured.get(variantKey(j)) ?? null;
  const baseline = $derived(view.summary.baseline_tput as number | null | undefined);
  const axis = $derived(
    tputAxis([...rows.map(tputOf).filter((v): v is number => v != null), ...(baseline == null ? [] : [baseline])]),
  );
  const newest = $derived(rows.slice().reverse());
  const units = $derived(variantUnits(view.tasks, ui.filter));
  const hasTask = (id: unknown) => id != null && view.tasks.some((t) => t.task_id === id);
  const pick = (j: StreamItem) =>
    hasTask(j.task_id)
      ? ui.select({ kind: "task", id: String(j.task_id) })
      : ui.select({ kind: "raw", title: "Journal entry", data: j });
  const pickUnit = (u: Unit) => ui.select({ kind: "task", id: u.task_id });
  const num = (v: unknown) => fmtNum(v as number | null);

  const cols: Col<StreamItem>[] = [
    { head: "time", text: (j) => fmtTime(tsOf(j)) },
    { head: "phase", text: (j) => j.phase },
    { head: "kind", text: (j) => j.kind },
    { head: "change", text: (j) => j.change, cls: "trunc" },
    { head: "outcome", text: (j) => j.outcome, cls: (j) => outcomeClass(j.outcome) },
    { head: "tok/s", text: (j) => num(tputOf(j)) },
    { head: "gain", text: (j) => (j.gain_pct == null ? "" : num(j.gain_pct) + "%") },
    { head: "task", text: (j) => j.task_id, cls: "mono" },
  ];
  const unitCols: Col<Unit>[] = [
    { head: "unit", text: (u) => u.label, cls: "mono" },
    { head: "status", text: (u) => u.status, cls: (u) => (u.status === "succeeded" ? "h-ok" : "h-" + u.status) },
    { head: "tok/s", text: (u) => num(u.output_throughput) },
    { head: "time", text: (u) => fmtTime(u.ts) },
    { head: "earlier rounds", text: (u) => (u.earlier ?? []).map((x) => fmtNum(x)).join(", ") },
    { head: "why", text: (u) => [u.stage, u.reason, u.error].filter(Boolean).join(" · "), cls: "trunc" },
    { head: "task", text: (u) => u.task_id, cls: "mono" },
  ];
</script>

<JournalChart {rows} tput={tputOf} {axis} {baseline} onpick={pick} />
<p class="muted">{rows.length} journal entr{rows.length === 1 ? "y" : "ies"}, newest first</p>
<DataTable testid="journal-table" rows={newest} {cols} onpick={pick} />
{#if units.length}
  <h4 class="muted">Variants measured ({units.length}), newest first</h4>
  <DataTable testid="variants-table" rows={units} cols={unitCols} key={(u) => u.key} onpick={pickUnit} />
{/if}

<style>
  p,
  h4 {
    margin: 0.4rem 0;
    font-size: 12px;
  }
</style>
