<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<script lang="ts">
  import { fetchDiff } from "../api";
  import type { DiffPayload } from "../types";

  let { a, b }: { a: string; b: string } = $props();

  let diff = $state.raw<DiffPayload | null>(null);
  let error = $state("");
  let seq = 0;
  $effect(() => {
    const mine = ++seq;
    diff = null;
    error = "";
    fetchDiff(a, b)
      .then((r) => mine === seq && (diff = r))
      .catch((e: unknown) => mine === seq && (error = e instanceof Error ? e.message : String(e)));
  });
</script>

<div class="head">
  <b>diff</b> <span class="mono">{a}</span> ↔ <span class="mono">{b}</span>
</div>
{#if error}
  <p class="h-crashed">{error}</p>
{:else if !diff}
  <p class="muted">loading the diff…</p>
{:else if diff.identical}
  <p class="h-ok">identical</p>
{:else}
  <table class="diff" data-testid="diff">
    <colgroup><col class="ln" /><col /><col class="ln" /><col /></colgroup>
    <tbody>
      {#each diff.rows as [tag, ai, at, bi, bt], i (i)}
        {#if tag === "hunk"}
          <tr class="hunk"><td colspan="4"></td></tr>
        {:else}
          <tr class={tag}>
            <td class="ln">{ai ?? ""}</td>
            <td class="a">{at}</td>
            <td class="ln">{bi ?? ""}</td>
            <td class="b">{bt}</td>
          </tr>
        {/if}
      {/each}
    </tbody>
  </table>
{/if}

<style>
  .head {
    margin-bottom: 0.4rem;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
    font-family: var(--mono);
    font-size: 11.5px;
  }
  td {
    padding: 0 6px;
    white-space: pre-wrap;
    word-break: break-all;
    vertical-align: top;
  }
  col.ln {
    width: 3.5em;
  }
  td.ln {
    color: var(--muted);
    text-align: right;
    user-select: none;
  }
  tr.hunk td {
    height: 6px;
    background: var(--panel2);
  }
  tr.replace td.a,
  tr.delete td.a {
    background: color-mix(in srgb, var(--crashed) 18%, transparent);
  }
  tr.replace td.b,
  tr.insert td.b {
    background: color-mix(in srgb, var(--ok) 18%, transparent);
  }
</style>
