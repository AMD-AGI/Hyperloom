<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<script lang="ts">
  import { refForJournal, refForVariant } from "../explain";
  import { fmtNum, fmtTime } from "../format";
  import { LATENCY_KINDS, LATENCY_STATS, latencyLabel, latencyOf } from "../latency";
  import type { InspectorStore } from "../store.svelte";
  import { tsOf } from "../timeline/scale";
  import type { StreamItem } from "../types";
  import type { UiState } from "../ui.svelte";
  import DataTable from "./DataTable.svelte";
  import BenchmarkScatter from "./BenchmarkScatter.svelte";
  import JournalChart from "./JournalChart.svelte";
  import { tputAxis } from "./tputAxis";
  import {
    journalRows,
    outcomeClass,
    variantKey,
    variantOutcomes,
    variantThroughputs,
    variantUnits,
    type Col,
    type Unit,
  } from "./rows";

  let { store, ui }: { store: InspectorStore; ui: UiState } = $props();

  const view = $derived(store.view!);
  const rows = $derived(journalRows(store.streams.journal, view.tasks, ui.filter));
  const runs = $derived(journalRows(store.streams.benchmark, view.tasks, ui.filter));
  const outcomes = $derived(variantOutcomes(store.streams.journal));
  const measured = $derived(variantThroughputs(view.tasks));
  const tputOf = (j: StreamItem) => (j.throughput as number | null) ?? measured.get(variantKey(j)) ?? null;
  const baseline = $derived(view.summary.baseline_tput as number | null | undefined);
  const axis = $derived(
    tputAxis([
      ...rows.map(tputOf).filter((v): v is number => v != null),
      ...runs.map((r) => Number(r.output_throughput)),
      ...(baseline == null ? [] : [baseline]),
    ]),
  );
  const plotted = $derived(runs.filter((r) => latencyOf(r, ui.latency) != null).length);
  const newest = $derived(rows.slice().reverse());
  const units = $derived(variantUnits(view.tasks, ui.filter));
  const hasTask = (id: unknown) => id != null && view.tasks.some((t) => t.task_id === id);
  const picker = (title: string) => (j: StreamItem) =>
    hasTask(j.task_id)
      ? ui.select({ kind: "task", id: String(j.task_id) })
      : ui.select({ kind: "raw", title, data: j });
  const pick = picker("Journal entry");
  // A right-click on a journal row or chart point explains the change; Open is the left click.
  const menu = (j: StreamItem, e: MouseEvent) => ui.openMenu(e, refForJournal(j), () => pick(j));
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

<div class="charts">
  <JournalChart {rows} tput={tputOf} {axis} {baseline} onpick={pick} oncontext={menu} />
  <div>
    <BenchmarkScatter rows={runs} {outcomes} {axis} latency={ui.latency} onpick={picker("Benchmark run")} />
    <p class="muted">
      <select bind:value={ui.latency.stat} aria-label="latency statistic" data-testid="latency-stat">
        {#each LATENCY_STATS as stat (stat)}<option value={stat}>{stat}</option>{/each}
      </select>
      <select bind:value={ui.latency.kind} aria-label="latency measure" data-testid="latency-kind">
        {#each LATENCY_KINDS as [kind, name] (kind)}<option value={kind}>{name}</option>{/each}
      </select>
      {plotted} of {runs.length} benchmark run{runs.length === 1 ? "" : "s"}: output tok/s against {latencyLabel(
        ui.latency,
      )}, upper left is better; joined in time order and colored by outcome like the chart on the left
    </p>
  </div>
</div>
<p class="muted">{rows.length} journal entr{rows.length === 1 ? "y" : "ies"}, newest first</p>
<DataTable testid="journal-table" rows={newest} {cols} onpick={pick} oncontext={menu} />
{#if units.length}
  <h4 class="muted">Variants measured ({units.length}), newest first</h4>
  <DataTable
    testid="variants-table"
    rows={units}
    cols={unitCols}
    key={(u) => u.key}
    onpick={pickUnit}
    oncontext={(u, e) => ui.openMenu(e, refForVariant(u), () => pickUnit(u))}
  />
{/if}

<style>
  .charts {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 0.75rem;
    align-items: start;
  }
  p,
  h4 {
    margin: 0.4rem 0;
    font-size: 12px;
  }
  select {
    font: inherit;
    color: var(--fg);
    background: var(--panel2);
    border: 1px solid var(--line);
    border-radius: 5px;
    padding: 1px 6px;
  }
</style>
