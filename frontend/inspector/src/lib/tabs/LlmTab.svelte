<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<script lang="ts">
  import { refFor } from "../explain";
  import { fmtNum, fmtTime } from "../format";
  import type { InspectorStore } from "../store.svelte";
  import { tsOf } from "../timeline/scale";
  import type { StreamItem } from "../types";
  import type { UiState } from "../ui.svelte";
  import DataTable from "./DataTable.svelte";
  import { callRows, type Col } from "./rows";

  let { store, ui }: { store: InspectorStore; ui: UiState } = $props();

  type Role = [string, Record<string, unknown>];
  const view = $derived(store.view!);
  const roles = $derived(Object.entries(view.llm ?? {}) as Role[]);
  const calls = $derived(callRows(store.streams.llm_call, view.tasks, ui.filter));
  const picked = $derived(ui.selection?.kind === "call" ? ui.selection.item : null);
  const num = (v: unknown) => fmtNum(v as number, 0);

  const roleCols: Col<Role>[] = [
    { head: "role", text: ([r]) => r },
    { head: "calls", text: ([, x]) => x.calls },
    { head: "errors", text: ([, x]) => x.errors, cls: ([, x]) => (Number(x.errors) ? "h-crashed" : "") },
    { head: "input tok", text: ([, x]) => num(x.input_tokens) },
    { head: "output tok", text: ([, x]) => num(x.output_tokens) },
    { head: "cache read", text: ([, x]) => num(x.cache_read_tokens) },
    { head: "avg latency", text: ([, x]) => num(x.latency_ms_avg) + " ms" },
    {
      head: "models",
      text: ([, x]) =>
        Object.entries((x.models ?? {}) as Record<string, number>)
          .map(([m, n]) => `${m} ×${n}`)
          .join(", "),
    },
  ];
  const callCols: Col<StreamItem>[] = [
    { head: "time", text: (c) => fmtTime(tsOf(c)) },
    { head: "role", text: (c) => c.component },
    { head: "model", text: (c) => c.model },
    { head: "turn", text: (c) => c.turn },
    { head: "in", text: (c) => c.input_tokens },
    { head: "out", text: (c) => c.output_tokens },
    { head: "latency", text: (c) => c.latency_ms },
    { head: "status", text: (c) => c.status, cls: (c) => (c.status === "ok" ? "h-ok" : "h-crashed") },
    { head: "call", text: (c) => c.call_id ?? c.task_id, cls: "mono" },
  ];
</script>

<DataTable testid="llm-roles" rows={roles} cols={roleCols} key={([r]) => r} onpick={([r]) => ui.toggleFilter(r)} />
<p class="muted">{calls.length} call(s), newest first; pick one to read its conversation</p>
<DataTable
  testid="llm-calls"
  rows={calls}
  cols={callCols}
  onpick={(c) => ui.select({ kind: "call", item: c })}
  oncontext={(c, e) => {
    const sel = { kind: "call", item: c } as const;
    ui.openMenu(e, refFor(sel), () => ui.select(sel));
  }}
  selected={(c) => c === picked}
/>

<style>
  p {
    margin: 0.4rem 0;
    font-size: 12px;
  }
</style>
