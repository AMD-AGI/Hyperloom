<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<script lang="ts">
  import { slide } from "svelte/transition";
  import { preview } from "./format";
  import { ms } from "./motion";
  import type { RunInfo } from "./types";

  let { run, summary }: { run: RunInfo; summary: Record<string, unknown> } = $props();

  let open = $state(false);

  const flat = (v: unknown): string =>
    v && typeof v === "object"
      ? Object.entries(v)
          .map(([k, x]) => `${k}=${x && typeof x === "object" ? JSON.stringify(x) : x}`)
          .join("  ")
      : String(v);

  const rows = $derived.by(() => {
    const out: [string, string][] = [["session dir", run.session_dir ?? "–"]];
    for (const [k, v] of Object.entries(run.manifest ?? {})) out.push([k, flat(v)]);
    out.push(["started", String(summary.start_ts ?? "–")]);
    (run.pod_history ?? []).forEach((p, i) =>
      out.push([
        i ? `resume ${i}` : "first start",
        `${p.acquired_at ?? "?"} on ${p.hostname ?? "?"} (pid ${p.pid ?? "?"})`,
      ]),
    );
    out.push(["crash count", String(summary.crash_count ?? "–")]);
    out.push(["failures", String(summary.failures_count ?? "–")]);
    for (const w of (summary.model_warnings as unknown[] | undefined) ?? [])
      out.push(["model warning", preview(w, 300)]);
    return out;
  });
</script>

<button class="toggle" aria-expanded={open} onclick={() => (open = !open)}>
  <span class="chev" class:open>▸</span> Run information
</button>
{#if open}
  <div class="grid" transition:slide={{ duration: ms(200) }}>
    {#each rows as [k, v], i (i)}
      <span class="muted">{k}</span><span class="mono">{v}</span>
    {/each}
  </div>
{/if}

<style>
  .toggle {
    margin-top: 6px;
    padding: 0;
    border: 0;
    background: none;
    color: var(--muted);
    font: inherit;
    cursor: pointer;
  }
  .chev {
    display: inline-block;
    transition: transform 0.2s ease-out;
  }
  .chev.open {
    transform: rotate(90deg);
  }
  .grid {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 2px 12px;
    margin-top: 6px;
  }
  .mono {
    font-family: var(--mono);
    font-size: 12px;
    word-break: break-word;
  }
  .muted {
    color: var(--muted);
  }
</style>
