<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<!-- One artifact by type: JSON as a tree, JSONL as a paged table, text as a tail that follows a growing log, and
     traces and binaries as metadata only. -->
<script lang="ts">
  import { onDestroy, tick } from "svelte";
  import { fetchArtifact } from "../api";
  import { fmtBytes, fmtTime, preview } from "../format";
  import JsonTree from "../JsonTree.svelte";
  import type { ArtifactPayload } from "../types";
  import type { UiState } from "../ui.svelte";
  import DataTable from "./DataTable.svelte";
  import { jsonlKeys, type Col } from "./rows";

  let { path, offset, ui }: { path: string; offset?: number; ui: UiState } = $props();

  const PAGE = 500;
  const FOLLOW_MS = 2000;
  let r = $state.raw<ArtifactPayload | null>(null);
  let error = $state("");
  let text = $state("");
  let follow = $state(false);
  let pre = $state<HTMLPreElement>();
  let seq = 0;
  let timer: ReturnType<typeof setInterval> | undefined;

  $effect(() => {
    const [p, o] = [path, offset];
    const mine = ++seq;
    stop();
    r = null;
    error = "";
    fetchArtifact(p, o != null ? { offset: o } : {})
      .then(async (res) => {
        if (mine !== seq) return;
        r = res;
        text = res.text ?? "";
        follow = res.mode === "text" && res.type === "log";
        await tick();
        if (follow && pre) pre.scrollTop = pre.scrollHeight;
      })
      .catch((e: unknown) => mine === seq && (error = e instanceof Error ? e.message : String(e)));
  });

  // Following polls from the last byte read and appends; the view sticks to the bottom only if it was there.
  $effect(() => {
    if (!follow || !r || r.mode !== "text") return stop();
    let end = r.end ?? 0;
    const mine = seq;
    timer = setInterval(async () => {
      try {
        const more = await fetchArtifact(path, { mode: "text", offset: end });
        if (mine !== seq) return;
        end = more.end ?? end;
        if (!more.text) return;
        const atEnd = !!pre && pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 20;
        text += more.text;
        await tick();
        if (atEnd && pre) pre.scrollTop = pre.scrollHeight;
      } catch {
        follow = false;
      }
    }, FOLLOW_MS);
    return stop;
  });
  function stop() {
    if (timer) clearInterval(timer);
    timer = undefined;
  }
  onDestroy(stop);

  const rows = $derived((r?.rows ?? []) as Record<string, unknown>[]);
  const cols = $derived(
    jsonlKeys(rows).map((k): Col<Record<string, unknown>> => ({
      head: k,
      text: (row) => preview(row[k], 120),
      cls: "trunc mono",
    })),
  );
  const from = $derived(r?.offset ?? 0);
  const total = $derived(r?.total ?? 0);
</script>

<div class="viewer" data-testid="art-view">
  {#if error}
    <p class="h-crashed">{path}: {error}</p>
  {:else if !r}
    <p class="muted">loading {path}</p>
  {:else}
    <div class="head">
      <b class="mono">{r.path}</b>
      <span class="type">{r.type}</span>
      <span class="muted">{fmtBytes(r.size)} · {fmtTime(r.mtime)}</span>
      {#if r.note}<span class="h-stalled">{r.note}</span>{/if}
      {#if r.mode === "text"}
        <label><input type="checkbox" bind:checked={follow} data-testid="art-follow" /> follow</label>
        {#if r.truncated}<span class="muted">showing the last {fmtBytes((r.end ?? 0) - (r.start ?? 0))}</span>{/if}
      {/if}
    </div>
    {#if r.mode === "json"}
      <div class="body"><JsonTree value={r.data} /></div>
    {:else if r.mode === "jsonl"}
      <div class="nav">
        <span class="muted">rows {from + 1}–{from + rows.length} of {total}</span>
        {#if from > 0}<button onclick={() => ui.openArtifact(path, Math.max(0, from - PAGE))}>prev</button>{/if}
        {#if from + rows.length < total}
          <button onclick={() => ui.openArtifact(path, from + PAGE)}>next</button>
          <button onclick={() => ui.openArtifact(path, Math.max(0, total - PAGE))}>last</button>
        {/if}
      </div>
      <div class="body">
        <DataTable
          testid="art-jsonl"
          {rows}
          {cols}
          onpick={(row) => ui.select({ kind: "raw", title: `${path} row`, data: row })}
        />
      </div>
    {:else if r.mode === "text"}
      <pre bind:this={pre} class="body" data-testid="art-text">{text}</pre>
    {:else}
      <p class="muted">
        {r.type === "trace"
          ? "Profiler trace: metadata only (these run to ~0.8 GB each). Open it with TraceLens or Perfetto."
          : "Binary or unknown file: metadata only."}
      </p>
    {/if}
  {/if}
</div>

<style>
  .viewer {
    display: flex;
    flex-direction: column;
    min-height: 0;
    height: 100%;
  }
  .head,
  .nav {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    align-items: center;
    margin-bottom: 0.4rem;
  }
  .type {
    font-size: 10px;
    color: var(--muted);
    border: 1px solid var(--line);
    border-radius: 3px;
    padding: 0 4px;
  }
  .body {
    flex: 1;
    min-height: 0;
    overflow: auto;
  }
  pre {
    margin: 0;
    padding: 0.5rem;
    background: var(--panel2);
    border-radius: 6px;
    font-family: var(--mono);
    font-size: 11.5px;
    white-space: pre-wrap;
    word-break: break-word;
  }
  button {
    font: inherit;
    font-size: 12px;
    color: var(--fg);
    background: var(--panel2);
    border: 1px solid var(--line);
    border-radius: 5px;
    padding: 1px 8px;
    cursor: pointer;
  }
</style>
