<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<!-- "Explain this": asks the narrator the fixed question about one item; the answer lands in the narrator's thread.
     Hidden without a reference or an LLM, and disabled while that item's explanation is pending. -->
<script lang="ts">
  import { labelFor, questionFor, type ItemRef, type Records } from "./explain";
  import { sameItem, type NarratorStore } from "./narrator-store.svelte";

  let { narrator, store, ref }: { narrator: NarratorStore; store: Records; ref: ItemRef | null } = $props();

  const busy = $derived(!!ref && !!narrator.explaining && sameItem(narrator.explaining.ref, ref));

  function explain(): void {
    if (ref) void narrator.explain(ref, questionFor(ref, store), labelFor(ref, store));
  }
</script>

{#if ref && narrator.available}
  <button
    class="explain"
    data-testid="explain"
    title="Ask the narrator to explain this"
    disabled={busy}
    onclick={explain}>{busy ? "explaining…" : "Explain"}</button
  >
{/if}

<style>
  .explain {
    flex: none;
    font: inherit;
    font-size: 12px;
    color: var(--accent);
    background: transparent;
    border: 1px solid var(--line);
    border-radius: 5px;
    padding: 1px 10px;
    cursor: pointer;
    white-space: nowrap;
  }
  .explain:hover:not(:disabled) {
    border-color: var(--accent);
  }
  .explain:disabled {
    opacity: 0.5;
    cursor: default;
  }
</style>
