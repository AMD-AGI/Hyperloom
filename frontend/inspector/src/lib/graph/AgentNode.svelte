<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<script lang="ts">
  import { Handle, Position, type NodeProps } from "@xyflow/svelte";
  import type { GraphNode } from "../types";
  import type { AgentNodeType } from "./types";

  let { data }: NodeProps<AgentNodeType> = $props();

  function subline(n: GraphNode): string {
    const counts = Object.entries(n.tasks ?? {}).map(([k, c]) => `${c} ${k}`);
    const trees = Object.keys(n.trees ?? {}).length;
    return [
      ...counts,
      n.llm_calls ? `${n.llm_calls} calls` : "",
      n.artifacts ? `${n.artifacts} files` : "",
      trees ? `${trees} tree${trees > 1 ? "s" : ""}` : "",
    ]
      .filter(Boolean)
      .join(" · ");
  }

  const n = $derived(data.node);
  const sub = $derived(subline(n));
  const tip = $derived(
    [n.id, `health: ${n.health}`, sub, n.llm_tokens ? `${n.llm_tokens} tokens` : ""].filter(Boolean).join("\n"),
  );
</script>

<div
  class="agent"
  class:selected={data.selected}
  class:faded={data.faded}
  style:--health="var(--{n.health})"
  title={tip}
  data-testid="graph-node"
  data-node={n.id}
>
  <Handle type="target" position={Position.Top} isConnectable={false} />
  <div class="label">{n.label}</div>
  <div class="sub">{sub}</div>
  <Handle type="source" position={Position.Bottom} isConnectable={false} />
</div>

<style>
  .agent {
    box-sizing: border-box;
    width: 132px;
    height: 42px;
    padding: 3px 8px;
    border: 2px solid var(--health);
    border-radius: 6px;
    background: var(--panel2);
    color: var(--fg);
    cursor: pointer;
    transition:
      border-color 0.6s ease,
      border-width 0.2s ease;
    animation: pop 0.4s cubic-bezier(0.2, 0.9, 0.3, 1.2);
  }
  .agent:hover {
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--health) 25%, transparent);
  }
  .selected {
    border-width: 4px;
    padding: 1px 6px;
  }
  /* Faded nodes keep an opaque background, so edges behind them stay hidden. */
  .faded {
    border-color: color-mix(in srgb, var(--health) 35%, var(--panel2));
  }
  .faded .label,
  .faded .sub {
    opacity: 0.4;
  }
  .label,
  .sub {
    transition: opacity 0.3s ease;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
  }
  .label {
    font-weight: 600;
    font-size: 12px;
  }
  .sub {
    color: var(--muted);
    font-size: 10px;
  }
  .agent :global(.svelte-flow__handle) {
    opacity: 0;
    pointer-events: none;
  }
  @keyframes pop {
    from {
      opacity: 0;
      transform: scale(0.8);
    }
  }
</style>
