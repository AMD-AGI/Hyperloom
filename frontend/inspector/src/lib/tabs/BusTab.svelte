<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<script lang="ts">
  import { fmtTime, preview } from "../format";
  import type { InspectorStore } from "../store.svelte";
  import { tsOf } from "../timeline/scale";
  import type { StreamItem } from "../types";
  import type { UiState } from "../ui.svelte";
  import DataTable from "./DataTable.svelte";
  import { busRows, type Col } from "./rows";

  let { store, ui }: { store: InspectorStore; ui: UiState } = $props();

  const rows = $derived(busRows(store.streams.event, store.view!.graph.nodes, ui.filter));
  const cols: Col<StreamItem>[] = [
    { head: "seq", text: (e) => e.seq },
    { head: "time", text: (e) => fmtTime(tsOf(e)) },
    { head: "from → to", text: (e) => `${e.from_agent} → ${e.to_agent}` },
    { head: "topic", text: (e) => e.topic },
    { head: "reply to", text: (e) => e.in_reply_to, cls: "mono" },
    { head: "payload", text: (e) => preview(e.payload), cls: "trunc mono" },
  ];
</script>

<p class="muted">{rows.length} event(s), newest first</p>
<DataTable
  testid="bus-table"
  {rows}
  {cols}
  onpick={(e) => ui.select({ kind: "event", item: e })}
  selected={(e) => ui.selection?.kind === "event" && ui.selection.item === e}
/>

<style>
  p {
    margin: 0.4rem 0;
    font-size: 12px;
  }
</style>
