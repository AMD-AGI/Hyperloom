<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<script lang="ts">
  import JsonTree from "../JsonTree.svelte";
  import type { InspectorStore } from "../store.svelte";
  import DataTable from "../tabs/DataTable.svelte";
  import type { Col } from "../tabs/rows";
  import type { GraphEdge, GraphNode, Task } from "../types";
  import type { UiState } from "../ui.svelte";
  import ArtifactLinks from "./ArtifactLinks.svelte";
  import { nodeRelated, producerArtifacts } from "./related";
  import Section from "./Section.svelte";

  let { node, store, ui }: { node: GraphNode; store: InspectorStore; ui: UiState } = $props();

  const rel = $derived(nodeRelated(node.id, store.view!));
  const files = $derived(producerArtifacts(store.artifacts, { node: node.id }));

  const edgeCols: Col<GraphEdge>[] = [
    { head: "from", text: (e) => e.source },
    { head: "to", text: (e) => e.target },
    { head: "topic", text: (e) => e.label },
    { head: "count", text: (e) => e.count },
  ];
  const taskCols: Col<Task>[] = [
    { head: "task", text: (t) => t.task_id, cls: "mono" },
    { head: "kind", text: (t) => t.kind },
    { head: "health", text: (t) => t.health, cls: (t) => "h-" + t.health },
    { head: "why", text: (t) => t.reason, cls: "trunc" },
  ];
</script>

<p class="muted">
  {#if ui.filter === node.id}
    The timeline and tabs are filtered to this node; click it in the graph again to clear.
  {:else}
    <button class="link" onclick={() => ui.toggleFilter(node.id)}>Filter the timeline and tabs to this node</button>
  {/if}
</p>
<Section title="Record" open={false} testid="dr-record"><JsonTree value={node} /></Section>
<Section title="Traffic ({rel.edges.length})">
  <DataTable rows={rel.edges} cols={edgeCols} key={(e) => `${e.source}>${e.target}:${e.label}`} />
</Section>
<Section title="Tasks ({rel.tasks.length})">
  <DataTable
    rows={rel.tasks}
    cols={taskCols}
    key={(t) => t.task_id}
    onpick={(t) => ui.follow({ kind: "task", id: t.task_id })}
  />
</Section>
<Section title="Artifacts ({files.length})" testid="dr-artifacts"><ArtifactLinks list={files} {ui} /></Section>

<style>
  p {
    margin: 0.3rem 0;
    font-size: 12.5px;
  }
</style>
