<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<!-- A collapsible JSON tree; children are only built once their node is opened. -->
<script lang="ts">
  import JsonTree from "./JsonTree.svelte";

  let { value, name = null, depth = 0 }: { value: unknown; name?: string | number | null; depth?: number } = $props();

  const entries = $derived.by((): [string | number, unknown][] => {
    if (!value || typeof value !== "object") return [];
    return Array.isArray(value) ? value.map((x, i) => [i, x]) : Object.entries(value);
  });
  const isObj = $derived(!!value && typeof value === "object");
  // svelte-ignore state_referenced_locally
  let open = $state(depth === 0 || (depth < 2 && entries.length < 50));
  const kind = $derived(typeof value === "string" ? "s" : typeof value === "number" ? "n" : "");
</script>

{#if isObj}
  <details bind:open class:root={depth === 0}>
    <summary>
      {#if name != null}<span class="k">{JSON.stringify(name)}</span>:
      {/if}<span class="muted">{Array.isArray(value) ? `[${entries.length}]` : `{${entries.length}}`}</span>
    </summary>
    {#if open}
      {#each entries as [k, v] (k)}
        <JsonTree value={v} name={Array.isArray(value) ? null : k} depth={depth + 1} />
      {/each}
    {/if}
  </details>
{:else}
  <div class="leaf" class:root={depth === 0}>
    {#if name != null}<span class="k">{JSON.stringify(name)}</span>:
    {/if}<span class={kind}>{JSON.stringify(value)}</span>
  </div>
{/if}

<style>
  details,
  .leaf {
    margin-left: 14px;
    font-family: var(--mono);
    font-size: 12px;
    word-break: break-word;
  }
  .leaf > span {
    white-space: pre-wrap;
  }
  details.root,
  .leaf.root {
    margin-left: 0;
  }
  summary {
    cursor: pointer;
  }
  .k {
    color: var(--accent);
  }
  .s {
    color: var(--ok);
  }
  .n {
    color: var(--stalled);
  }
</style>
