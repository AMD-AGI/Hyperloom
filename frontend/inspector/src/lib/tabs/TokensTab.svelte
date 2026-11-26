<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<!-- Token consumption: totals per token type, the token chart over the run on the timeline's time axis, and the
     split by agent and by model. Filtered to the picked graph node like the other tabs. Names and their
     definitions, shown as hover hints, come from the server (tokens.py TERMS). -->
<script lang="ts">
  import { fmtCompact, fmtNum } from "../format";
  import { sessionClock } from "../now.svelte";
  import type { InspectorStore } from "../store.svelte";
  import { fitRange } from "../timeline/scale";
  import {
    cacheHit,
    decode,
    KIND_COLOR,
    KINDS,
    NO_MODEL,
    termLabel,
    termText,
    type Group,
    type Kind,
    type Mode,
  } from "../tokens";
  import Tweened from "../Tweened.svelte";
  import type { TokenTotals } from "../types";
  import type { UiState } from "../ui.svelte";
  import DataTable from "./DataTable.svelte";
  import type { Col } from "./rows";
  import TokenChart from "./TokenChart.svelte";

  let { store, ui }: { store: InspectorStore; ui: UiState } = $props();

  type Row = [string, TokenTotals];
  const EMPTY: TokenTotals = {
    input: 0,
    cache_write: 0,
    cache_read: 0,
    output: 0,
    reasoning: 0,
    total: 0,
    calls: 0,
    errors: 0,
  };
  const view = $derived(store.view!);
  const tv = $derived(view.tokens);
  const terms = $derived(tv?.terms);
  const L = (key: string) => termLabel(terms, key);
  /** The hover hint of *key* (lib/hint.ts): its definition, under the exact figure if any. */
  const hint = (key: string, exact?: string) => {
    const text = termText(terms, key);
    return text ? { "data-hint": text, "data-hint-head": exact } : { "data-hint": exact };
  };
  const clock = sessionClock(() => store);
  /** No now line once a live run has gone quiet: its charts end at its last activity, where the series does. */
  const chartNow = $derived(view.timeline.end != null ? null : clock.current);
  // The timeline's axis (range, zoom, idle gaps); the whole run until the timeline reports one.
  const axis = $derived(ui.timeAxis ?? { ...fitRange(view.clock, view.timeline, clock.current), gaps: [] });
  const decoded = $derived(decode(tv.series, ui.tokenGroup, ui.filter, { terms }));
  const totals = $derived(ui.filter ? (tv.by_node[ui.filter] ?? EMPTY) : tv.totals);
  const kinds = $derived(KINDS.filter((k) => k !== "reasoning" || totals.reasoning > 0));
  const hit = $derived(cacheHit(totals));
  const pending = $derived(ui.filter ? 0 : (tv.totals.output_pending ?? 0));
  const unreported = $derived(ui.filter ? 0 : (tv.totals.output_unreported ?? 0));
  const turns = (n: number) => `${n} turn${n === 1 ? "" : "s"}`;
  const live = $derived(view.clock.mode === "live");
  const bucketMin = $derived(fmtNum(tv.series.step / 60, 1));
  const active = $derived(live && Math.abs(tv.series.end - view.clock.now) < 5);
  const nodes = $derived(Object.entries(tv.by_node) as Row[]);
  const models = $derived(Object.entries(tv.by_model) as Row[]);
  const share = (x: TokenTotals) => (tv.totals.total ? fmtNum((100 * x.total) / tv.totals.total, 1) + "%" : "–");
  const compact = (n: number) => fmtCompact(n);
  const legend = $derived(decoded.series.slice().reverse());

  const col = <T,>(key: string, text: Col<T>["text"], extra: Partial<Col<T>> = {}): Col<T> => ({
    head: L(key),
    hint: termText(terms, key) || undefined,
    text,
    ...extra,
  });
  const count = (k: Kind) => col<Row>(k, ([, x]) => fmtNum(x[k], 0));
  const cols = (key: "agent" | "model", withReasoning: boolean): Col<Row>[] => [
    col<Row>(key, ([k]) => (key === "model" && k === NO_MODEL ? "no model recorded" : k), { cls: "mono" }),
    col<Row>("calls", ([, x]) => x.calls),
    col<Row>("errors", ([, x]) => x.errors, { cls: ([, x]) => (x.errors ? "h-crashed" : "") }),
    count("input"),
    count("cache_write"),
    count("cache_read"),
    count("output"),
    ...(withReasoning ? [count("reasoning")] : []),
    col<Row>("total", ([, x]) => fmtNum(x.total, 0)),
    col<Row>("share", ([, x]) => share(x)),
  ];
  const reasoning = $derived(tv.totals.reasoning > 0);
  const nodeCols = $derived(cols("agent", reasoning));
  const modelCols = $derived(cols("model", reasoning));

  function setMode(m: Mode) {
    ui.tokenMode = m;
  }
  function setGroup(g: Group) {
    ui.tokenGroup = g;
  }
  function toggle(key: string) {
    if (ui.tokenHidden.has(key)) ui.tokenHidden.delete(key);
    else ui.tokenHidden.add(key);
  }
</script>

<div class="cards" data-testid="token-cards">
  <div class="card main" {...hint("total", `${fmtNum(totals.total, 0)} tokens`)} data-testid="token-total">
    <span class="label">{L("total")}</span>
    <span class="big"><Tweened value={totals.total} format={compact} /></span>
    <span class="sub" {...hint("calls")}>
      {fmtNum(totals.calls, 0)} LLM call{totals.calls === 1 ? "" : "s"}{#if totals.errors}
        · <span class="h-crashed" {...hint("errors")}>{totals.errors} failed</span>{/if}
    </span>
  </div>
  {#each kinds as k (k)}
    <div class="card" {...hint(k, `${fmtNum(totals[k as Kind], 0)} tokens`)} data-testid="token-kind-{k}">
      <span class="label"><span class="sw" style="background: {KIND_COLOR[k]}"></span>{L(k)}</span>
      <span class="big"><Tweened value={totals[k as Kind]} format={compact} /></span>
      <span class="sub">
        {#if k === "output" && (pending || unreported)}
          {#if pending}<span {...hint("output_pending")} data-testid="token-output-pending"
              >{L("output_pending")}: {turns(pending)}</span
            >{/if}{#if pending && unreported}&nbsp;·
          {/if}{#if unreported}<span {...hint("output_unreported")} data-testid="token-output-unreported"
              >{L("output_unreported")}: {turns(unreported)}</span
            >{/if}
        {:else}
          {totals.total ? fmtNum((100 * totals[k as Kind]) / totals.total, 1) : 0}% of total tokens
        {/if}
      </span>
    </div>
  {/each}
  <div class="card" {...hint("cache_hit")} data-testid="token-cache-hit">
    <span class="label">{L("cache_hit")}</span>
    <span class="big">{hit == null ? "–" : fmtNum(100 * hit, 1) + "%"}</span>
    <span class="sub">of all input was read from cache</span>
  </div>
  {#if active && !ui.filter}
    <div class="card" {...hint("recent")}>
      <span class="label">{L("recent")}</span>
      <span class="big"><Tweened value={tv.recent.total} format={compact} /></span>
      <span class="sub">
        {tv.recent.calls} LLM call{tv.recent.calls === 1 ? "" : "s"}
      </span>
    </div>
  {/if}
</div>

<div class="bar">
  <div class="seg" role="group" aria-label="chart kind">
    <button class:on={ui.tokenMode === "cumulative"} onclick={() => setMode("cumulative")} {...hint("running_total")}
      >{L("running_total")}</button
    >
    <button
      class:on={ui.tokenMode === "rate"}
      onclick={() => setMode("rate")}
      {...hint("per_minute")}
      data-testid="token-rate">{L("per_minute")}</button
    >
  </div>
  <div class="seg" role="group" aria-label="stack by">
    <button class:on={ui.tokenGroup === "kind"} onclick={() => setGroup("kind")} {...hint("by_type")}
      >{L("by_type")}</button
    >
    <button
      class:on={ui.tokenGroup === "node"}
      onclick={() => setGroup("node")}
      {...hint("by_agent")}
      data-testid="token-by-agent">{L("by_agent")}</button
    >
  </div>
</div>

<TokenChart
  series={tv.series}
  {decoded}
  hidden={ui.tokenHidden}
  mode={ui.tokenMode}
  phases={view.timeline.phases}
  {axis}
  height={240}
  now={chartNow}
/>

<div class="legend" data-testid="token-legend">
  {#each legend as s (s.key)}
    <button
      class="chip"
      class:off={ui.tokenHidden.has(s.key)}
      onclick={() => toggle(s.key)}
      data-hint={[termText(terms, s.key), "Click to show or hide."].filter(Boolean).join("\n")}
    >
      <span class="sw" style="background: {s.color}"></span>{s.label}
      <span class="muted">{fmtCompact(s.total)}</span>
    </button>
  {/each}
</div>
<p class="muted note">
  Each LLM call sits at its <span {...hint("time")}>call time</span>, when its request was sent; bucket width {bucketMin}
  min. A Claude CLI specialist reports its output once, when it finishes, and it is counted on its last turn; its side calls
  (WebSearch, WebFetch) are counted at its end.
  {#if pending}
    {tv.pending_tasks.length} specialist{tv.pending_tasks.length === 1 ? " is" : "s are"} running: their turns come from the
    live log until the ledger has them, with output pending.
  {/if}
  {#if unreported}
    {tv.unreported_tasks?.length ?? 0} specialist{tv.unreported_tasks?.length === 1 ? "" : "s"} ended without reporting
    {tv.unreported_tasks?.length === 1 ? "its" : "their"} output (cancelled, say): their turns come from the live log, and
    their output is unknown and left out of the totals.
  {/if}
</p>

<h4 class="muted">By agent; pick one to filter</h4>
<DataTable
  testid="token-agents"
  rows={nodes}
  cols={nodeCols}
  key={([k]) => k}
  onpick={([k]) => ui.toggleFilter(k)}
  selected={([k]) => k === ui.filter}
/>
<h4 class="muted">By model</h4>
<DataTable testid="token-models" rows={models} cols={modelCols} key={([k]) => k} />

<style>
  .cards {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.25rem 0 0.6rem;
  }
  .card {
    display: flex;
    flex-direction: column;
    min-width: 7.5rem;
    padding: 0.4rem 0.6rem;
    background: var(--panel2);
    border: 1px solid var(--line);
    border-radius: 6px;
  }
  .card.main {
    min-width: 9rem;
    border-color: var(--accent);
  }
  .label {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 11px;
    color: var(--muted);
  }
  .big {
    font-size: 20px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }
  .sub {
    font-size: 11px;
    color: var(--muted);
  }
  .sw {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 2px;
    flex: none;
  }
  .bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.35rem;
    font-size: 12px;
  }
  .seg {
    display: inline-flex;
    border: 1px solid var(--line);
    border-radius: 6px;
    overflow: hidden;
  }
  .seg button {
    font: inherit;
    color: var(--muted);
    background: none;
    border: 0;
    padding: 0.2rem 0.6rem;
    cursor: pointer;
  }
  .seg button + button {
    border-left: 1px solid var(--line);
  }
  .seg button.on {
    color: var(--fg);
    background: var(--panel2);
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin: 0.35rem 0;
  }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font: inherit;
    font-size: 12px;
    color: var(--fg);
    background: var(--panel2);
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 0.1rem 0.55rem;
    cursor: pointer;
  }
  .chip.off {
    opacity: 0.45;
    text-decoration: line-through;
  }
  .note,
  h4 {
    margin: 0.4rem 0;
    font-size: 12px;
  }
  .card[data-hint],
  .sub [data-hint],
  .note [data-hint],
  .note[data-hint] {
    cursor: help;
  }
</style>
