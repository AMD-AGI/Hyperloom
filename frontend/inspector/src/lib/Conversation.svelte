<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<!-- The prompt and response of one LLM call. Orchestration and critic calls are looked up by call id; specialist
     calls by task and turn, which the server resolves from conversations.jsonl or the specialist's process.log. -->
<script lang="ts">
  import { fade } from "svelte/transition";
  import { fetchConversation, type ConversationKey } from "./api";
  import JsonTree from "./JsonTree.svelte";
  import { ms } from "./motion";
  import type { StreamItem } from "./types";
  import type { UiState } from "./ui.svelte";

  let { call, ui }: { call: StreamItem; ui: UiState } = $props();

  const key = $derived.by((): ConversationKey | null => {
    if (call.call_id) return { call_id: String(call.call_id) };
    if (call.task_id) return { task_id: String(call.task_id), turn: call.turn == null ? "" : String(call.turn) };
    return null;
  });

  let rec = $state.raw<Record<string, unknown> | null>(null);
  let error = $state("");
  let loading = $state(false);
  let seq = 0;
  $effect(() => {
    const k = key;
    const mine = ++seq;
    rec = null;
    error = "";
    if (!k) return;
    loading = true;
    fetchConversation(k)
      .then((r) => mine === seq && (rec = r))
      .catch((e: unknown) => mine === seq && (error = e instanceof Error ? e.message : String(e)))
      .finally(() => mine === seq && (loading = false));
  });
  const text = (v: unknown) => (v == null ? "" : typeof v === "string" ? v : JSON.stringify(v, null, 2));
</script>

<div class="conv" data-testid="conversation">
  {#if !key}
    <p class="muted">No call id or task id to look this call up by.</p>
  {:else if loading}
    <p class="muted">loading the conversation…</p>
  {:else if error}
    <p class="muted">{error}</p>
  {:else if rec}
    <div in:fade={{ duration: ms(200) }}>
      {#if rec.source}
        <p class="muted">
          Turn {rec.turn} from
          <button class="link mono" onclick={() => ui.openArtifact(String(rec!.source))}>{rec.source}</button>; only the
          first specialist turn is recorded in conversations.jsonl.
        </p>
      {/if}
      <details>
        <summary>prompt <span class="muted">({text(rec.prompt).length.toLocaleString()} chars)</span></summary>
        <pre data-testid="conv-prompt">{text(rec.prompt)}</pre>
      </details>
      <details open>
        <summary>response</summary>
        <pre data-testid="conv-response">{text(rec.response)}</pre>
      </details>
      <details>
        <summary>record</summary>
        <JsonTree value={rec} />
      </details>
    </div>
  {/if}
</div>

<style>
  .conv p {
    margin: 0.3rem 0;
  }
  summary {
    cursor: pointer;
    margin: 0.3rem 0;
    font-weight: 600;
  }
  pre {
    margin: 0 0 0.5rem;
    padding: 0.5rem;
    max-height: 22rem;
    overflow: auto;
    background: var(--panel2);
    border-radius: 6px;
    font-family: var(--mono);
    font-size: 11.5px;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .link {
    font: inherit;
    font-family: var(--mono);
    color: var(--accent);
    background: none;
    border: 0;
    padding: 0;
    cursor: pointer;
  }
</style>
