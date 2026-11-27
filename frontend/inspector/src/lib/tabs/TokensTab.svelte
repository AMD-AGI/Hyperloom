<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<!-- Token consumption and what it cost at the built-in price sheet's prices: totals per token type, the token and
     USD charts over the run on one time axis, the split by agent and by model, and the prices used. Filtered to the
     picked graph node like the other tabs. Names and their definitions, shown as hover hints, come from the server
     (tokens.py TERMS). -->
<script lang="ts">
  import { fmtCompact, fmtNum, fmtTime, fmtUsd } from "../format";
  import { sessionClock } from "../now.svelte";
  import type { InspectorStore } from "../store.svelte";
  import { fitRange } from "../timeline/scale";
  import {
    cacheHit,
    decode,
    KIND_COLOR,
    KINDS,
    NO_MODEL,
    unpricedReasons,
    unpricedWhy,
    termLabel,
    termText,
    type Group,
    type Kind,
    type Mode,
  } from "../tokens";
  import Tweened from "../Tweened.svelte";
  import type { TokenPrice, TokenTotals } from "../types";
  import type { UiState } from "../ui.svelte";
  import DataTable from "./DataTable.svelte";
  import type { Col } from "./rows";
  import TokenChart from "./TokenChart.svelte";

  let { store, ui }: { store: InspectorStore; ui: UiState } = $props();

  type Row = [string, TokenTotals];
  /** A priced model's lines in the price table: its prices, the tokens this run used, and what they cost. */
  type PriceRow = { model: string; line: "price" | "used" | "spent"; p: TokenPrice; x: TokenTotals };
  const EMPTY: TokenTotals = {
    input: 0,
    cache_write: 0,
    cache_read: 0,
    output: 0,
    reasoning: 0,
    total: 0,
    calls: 0,
    errors: 0,
    cost_usd: 0,
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
  const decoded = $derived(decode(tv.series, ui.tokenGroup, ui.filter, "tokens", { terms }));
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
  const usd = (n: number) => fmtUsd(n);

  // Cost: priced by the built-in price sheet; the USD columns come with every series.
  const pricing = $derived(tv.pricing);
  const priced = $derived(tv.totals.cost_usd != null);
  const cost = $derived(priced ? (totals.cost_usd ?? null) : null);
  const costs = $derived(priced ? decode(tv.series, ui.tokenGroup, ui.filter, "usd", { terms }) : null);
  const costByKind = $derived(
    priced
      ? new Map(decode(tv.series, "kind", ui.filter, "usd").series.map((s) => [s.key, s.total]))
      : new Map<string, number>(),
  );
  // The cost card's hint: what the figure is, else why there is none.
  const costHint = $derived(
    cost != null
      ? hint("cost_usd", `${fmtUsd(cost)} at the prices below`)
      : totals.unpriced
        ? hint("unpriced")
        : { "data-hint": "no call priced" },
  );
  const reasons = $derived(unpricedReasons(totals));
  const unpricedCell = (x: TokenTotals) =>
    x.unpriced
      ? ` · unpriced: ${unpricedReasons(x)
          .map(([, t]) => t)
          .join(", ")}`
      : "";
  // What the estimated cost leaves out and why, in red under the price source.
  const why = $derived(unpricedWhy(tv));
  const searches = $derived(
    totals.web_searches ? ` · ${totals.web_searches} web search${totals.web_searches === 1 ? "" : "es"}` : "",
  );
  // The price sheet in a word: its file's name.
  const where = $derived(pricing.source.split("/").pop());
  const prices = $derived(
    Object.entries(pricing.models).flatMap(([model, p]) =>
      (["price", "used", "spent"] as const).map((line): PriceRow => ({
        model,
        line,
        p,
        x: tv.by_model[model] ?? EMPTY,
      })),
    ),
  );
  const check = $derived(tv.check ?? null);
  const agrees = $derived(
    check?.priced_usd != null &&
      Math.abs(check.priced_usd - check.claude_cli_usd) <= Math.max(0.01, 0.005 * check.claude_cli_usd),
  );
  // One legend for both charts: a series' tokens and its cost.
  const legend = $derived.by(() => {
    const usdOf = new Map((costs?.series ?? []).map((s) => [s.key, s.total]));
    return decoded.series.map((s) => ({ ...s, usd: usdOf.get(s.key) ?? null })).reverse();
  });

  const col = <T,>(key: string, text: Col<T>["text"], extra: Partial<Col<T>> = {}): Col<T> => ({
    head: L(key),
    hint: termText(terms, key) || undefined,
    text,
    ...extra,
  });
  const count = (k: Kind) => col<Row>(k, ([, x]) => fmtNum(x[k], 0));
  const cols = (key: "agent" | "model", withCost: boolean, withReasoning: boolean): Col<Row>[] => [
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
    ...(withCost
      ? [
          col<Row>("cost_usd", ([, x]) => fmtUsd(x.cost_usd) + unpricedCell(x), {
            cls: ([, x]) => (x.unpriced ? "mono h-crashed wrap" : "mono"),
          }),
          col<Row>("cost_share", ([, x]) => costShare(x)),
        ]
      : []),
  ];
  const costShare = (x: TokenTotals) =>
    tv.totals.cost_usd && x.cost_usd != null ? fmtNum((100 * x.cost_usd) / tv.totals.cost_usd, 1) + "%" : "–";
  const reasoning = $derived(tv.totals.reasoning > 0);
  const nodeCols = $derived(cols("agent", priced, reasoning));
  const modelCols = $derived(cols("model", priced, reasoning));
  const PRICED = ["input", "cache_write", "cache_read", "output"] as const;
  const LINES = { price: "Price", used: "Used in this run", spent: "Cost in this run" };
  // Reasoning output is billed at the output price, so the output column counts it too.
  const billed = (x: TokenTotals, k: (typeof PRICED)[number]) => (k === "output" ? x.output + x.reasoning : x[k]);
  const spent = (p: TokenPrice, k: (typeof PRICED)[number]) =>
    k === "output" ? p.spent_usd.output + p.spent_usd.reasoning : p.spent_usd[k];
  const priceCell =
    (k: (typeof PRICED)[number]) =>
    ({ line, p, x }: PriceRow): string => {
      if (line === "price") return `${fmtUsd(p[k])} / million-tokens`;
      if (line === "spent") return fmtUsd(spent(p, k));
      const of = k === "output" && x.reasoning ? `, ${fmtNum(x.reasoning, 0)} of them reasoning` : "";
      return `${fmtNum(billed(x, k), 0)} tokens${of}`;
    };
  // The price sheet gives no price per search: a call that searches is unpriced.
  const lacksSearch = ({ line, x }: PriceRow) => (line !== "used" && x.web_searches ? "h-crashed" : "");
  const searchCell = ({ line, x }: PriceRow): string => {
    const n = x.web_searches ?? 0;
    if (line === "price") return "not listed";
    if (line === "used") return `${n} search${n === 1 ? "" : "es"}`;
    return n ? "unpriced" : "–";
  };
  const first = (r: PriceRow) => r.line === "price";
  const priceCols = $derived<Col<PriceRow>[]>([
    col<PriceRow>("model", (r) => (first(r) ? r.model : ""), { cls: "mono" }),
    col<PriceRow>("priced_as", (r) => (first(r) ? r.p.id : ""), { cls: "mono" }),
    { head: "", text: (r) => LINES[r.line] },
    ...PRICED.map((k) => col<PriceRow>(k, priceCell(k))),
    ...(prices.some((r) => r.x.web_searches) ? [col<PriceRow>("web_search", searchCell, { cls: lacksSearch })] : []),
    {
      head: "Total",
      hint: "The model's tokens in this run and their estimated cost.",
      text: ({ line, x }) =>
        line === "price" ? "" : line === "used" ? `${fmtNum(x.total, 0)} tokens` : fmtUsd(x.cost_usd),
    },
    col<PriceRow>("cache_basis", (r) => (first(r) ? r.p.cache_basis : "")),
  ]);

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
  <div class="card main" {...costHint} data-testid="token-cost">
    <span class="label">{L("cost_usd")}</span>
    <span class="big"
      >{#if cost == null}–{:else}<Tweened value={cost} format={usd} />{/if}</span
    >
    <span class="sub">
      {#if reasons.length}
        <span class="h-crashed"
          >Unpriced: {#each reasons as [k, text], i (k)}{i ? ", " : ""}<span {...hint(k)}>{text}</span>{/each}</span
        >
      {:else if cost == null}
        no call priced
      {:else}
        <span {...hint("source")}>US dollars at {where} prices</span>{searches}
      {/if}
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
        {#if cost != null}· {fmtUsd(costByKind.get(k) ?? 0)}{/if}
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
        {tv.recent.calls} LLM call{tv.recent.calls === 1 ? "" : "s"}{#if priced && tv.recent.cost_usd != null}
          · {fmtUsd(tv.recent.cost_usd)}{/if}
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
  <span class="grow"></span>
  {#if tv.file}
    <span class="muted file" title={tv.file}>
      saved to <span class="mono">{tv.file}</span>{#if tv.saved_at}&nbsp;at {fmtTime(tv.saved_at)}{/if}
    </span>
  {/if}
  <a class="dl" href="api/tokens?download=1" download data-testid="token-download">Download receipt in JSON</a>
</div>

<TokenChart
  series={tv.series}
  {decoded}
  hidden={ui.tokenHidden}
  mode={ui.tokenMode}
  phases={view.timeline.phases}
  {axis}
  height={costs ? 220 : 240}
  now={chartNow}
/>
{#if costs}
  <TokenChart
    series={tv.series}
    decoded={costs}
    hidden={ui.tokenHidden}
    mode={ui.tokenMode}
    phases={view.timeline.phases}
    {axis}
    metric="usd"
    height={180}
    now={chartNow}
  />
{/if}

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
      {#if s.usd != null}<span class="muted">{fmtUsd(s.usd)}</span>{/if}
    </button>
  {/each}
</div>
<p class="muted note">
  Each LLM call sits at its <span {...hint("time")}>call time</span>, when its request was sent; bucket width {bucketMin}
  min. A Claude CLI specialist reports its output once, when it finishes, and it is counted on its last turn; its side calls
  (WebSearch, WebFetch) are counted at its end.
  {#if pending}
    {tv.pending_tasks.length} specialist{tv.pending_tasks.length === 1 ? " is" : "s are"} running: their turns come from the
    live log until the ledger has them, with output pending{#if priced}, so their cost grows when they finish{/if}.
  {/if}
  {#if unreported}
    {tv.unreported_tasks?.length ?? 0} specialist{tv.unreported_tasks?.length === 1 ? "" : "s"} ended without reporting
    {tv.unreported_tasks?.length === 1 ? "its" : "their"} output (cancelled, say): their turns come from the live log, and
    their output is unknown and left out of the totals{priced ? " and the cost" : ""}.
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

<h4 class="muted" {...hint("price")}>{L("price")} (US dollars), and what this run used and spent at it</h4>
{#if prices.length}
  <DataTable testid="token-prices" rows={prices} cols={priceCols} key={(r) => `${r.model}/${r.line}`} />
{/if}
<dl class="pricing muted" data-testid="token-pricing">
  <dt {...hint("source")}>{L("source")}</dt>
  <dd><span class="mono">{pricing.source}</span>, the built-in price sheet</dd>
  {#if pricing.built_from}
    <dt>Built from</dt>
    <dd>
      <span class="mono" data-testid="token-built-from">{pricing.built_from}</span>
    </dd>
  {/if}
  {#if why.head}
    <dt class="h-crashed" {...hint("unpriced")}>{L("unpriced")}</dt>
    <dd class="h-crashed" data-testid="token-unpriced">
      <ul>
        <li>{why.head}.</li>
        {#each why.lines as line (line)}<li>{line}.</li>{/each}
      </ul>
    </dd>
  {/if}
  <dt>How it is priced</dt>
  <dd>
    <ul>
      <li>
        Cost in this run = tokens used × price per 1M tokens ÷ 1,000,000, over the calls priced in full: a call left out
        of the estimated cost is left out of every part of it.
      </li>
      <li>
        The built-in price sheet, the one multi-branch spends by, gives each model's uncached input, output and cache
        prices; a cache write it lists no price for is billed at the input price.
      </li>
      <li>Reasoning output is billed at the output price.</li>
      <li>Web searches have no price: the sheet gives none per search, so a call that searches is unpriced.</li>
      <li>A call that needs a missing price is unpriced, never costed at a guess.</li>
    </ul>
  </dd>
  {#if check}
    <dt {...hint("check")}>{L("check")}</dt>
    <dd data-testid="token-check">
      The Claude CLI's own cost of its {check.tasks} finished specialist{check.tasks === 1 ? "" : "s"} is
      {fmtUsd(check.claude_cli_usd)}; priced here, {fmtUsd(check.priced_usd)}{#if agrees}: they agree.{:else if check.priced_usd != null}.
        The CLI uses Anthropic's list prices; the sheet's may differ.{:else}.{/if}
    </dd>
  {/if}
</dl>

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
  .grow {
    flex: 1;
  }
  .file {
    max-width: 32rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .dl {
    color: var(--accent);
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
  .pricing {
    display: grid;
    grid-template-columns: max-content minmax(0, 1fr);
    gap: 0.35rem 1rem;
    margin: 0.5rem 0;
    font-size: 12px;
  }
  .pricing dt {
    font-weight: 600;
  }
  .pricing dd {
    margin: 0;
  }
  .pricing ul {
    margin: 0.1rem 0;
    padding-left: 1.1rem;
  }
  .card[data-hint],
  .sub [data-hint],
  h4[data-hint],
  .note [data-hint],
  .note[data-hint],
  .pricing [data-hint] {
    cursor: help;
  }
</style>
