<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<!--
  The narrator: a short summary of what the run is doing at the clock, written by the configured LLM, with a
  question box and a pinned focus. Node names in the text are chips that filter the views like a graph click. Auto
  mode asks for a narration when the panel opens, on a phase change and every two minutes; the server answers with
  the latest one when nothing material changed. Explained items head their answers in the thread, the pending one
  scrolls the card into view, and the item last explained rides along with typed questions until removed.
-->
<script lang="ts">
  import { tick, untrack } from "svelte";
  import { prefersReducedMotion } from "svelte/motion";
  import { slide } from "svelte/transition";
  import { fmtDur, fmtTime } from "./format";
  import { ms } from "./motion";
  import type { NarratorStore } from "./narrator-store.svelte";
  import { asked, blocks, chipper, explainHead } from "./narrator";
  import type { InspectorStore } from "./store.svelte";
  import type { NarratorEntry } from "./types";
  import type { UiState } from "./ui.svelte";

  let { store, ui, narrator }: { store: InspectorStore; ui: UiState; narrator: NarratorStore } = $props();

  const AUTO_SEC = 120;
  const POLL_MS = 5000;
  /** A narration further than this behind the clock is marked as describing an earlier time. */
  const STALE_SEC = 300;

  let question = $state("");
  let card: HTMLElement | undefined = $state();
  let wall = $state(Date.now() / 1000);
  let sessionNow = $state(untrack(() => store.now()));

  const nodes = $derived(store.view?.graph.nodes ?? []);
  const split = $derived(chipper(nodes));
  const phase = $derived(store.view?.summary.phase ?? "");
  const latest = $derived(narrator.latest);
  const replay = $derived(store.view?.clock.mode === "replay");
  // A boolean, so the auto effect reruns when availability changes, not on every status refresh.
  const available = $derived(narrator.available);

  // Poll the entries (in memory on the server, so cheap) while the panel is open: a replay seek changes which ones
  // are visible, and another viewer may have asked something.
  $effect(() => {
    if (!narrator.open) return;
    untrack(() => void narrator.refresh());
    const id = setInterval(() => {
      wall = Date.now() / 1000;
      sessionNow = store.now();
      void narrator.refresh();
    }, POLL_MS);
    return () => clearInterval(id);
  });

  // Auto narration: on open, on a phase change, then every two minutes.
  $effect(() => {
    void phase;
    if (!narrator.open || !narrator.auto || !available) return;
    // narrate() reads the store's state; untracked, so its own updates don't rerun this effect.
    untrack(() => void narrator.narrate(false));
    const id = setInterval(() => void narrator.narrate(false), AUTO_SEC * 1000);
    return () => clearInterval(id);
  });

  // Auto narration also fills a gap: a replay that starts over or is scrubbed back hides the later narrations.
  const missing = $derived(narrator.status != null && latest.narration == null);
  $effect(() => {
    if (narrator.open && narrator.auto && available && missing) untrack(() => void narrator.narrate(false));
  });

  // An explanation started (from the drawer or a context menu): bring the card, opened by explain(), into view. Keyed
  // on the item, so status refreshes don't scroll again.
  const explainingKey = $derived(narrator.explaining ? JSON.stringify(narrator.explaining.ref) : "");
  $effect(() => {
    if (!explainingKey) return;
    untrack(() => {
      void tick().then(() =>
        card?.scrollIntoView({ block: "nearest", behavior: prefersReducedMotion.current ? "auto" : "smooth" }),
      );
    });
  });

  async function ask(e: SubmitEvent) {
    e.preventDefault();
    // Clear the box on send, so typing the next question while this one is answered isn't wiped; a failed question
    // comes back if the box is still empty.
    const q = question;
    question = "";
    if (!(await narrator.ask(q)) && !question) question = q;
  }

  function pin() {
    narrator.setFocus(question);
    question = "";
    void narrator.narrate(true);
  }

  function unpin() {
    narrator.setFocus("");
    void narrator.narrate(true);
  }

  function headline(body: string): string {
    const first = blocks(body)[0];
    return !first ? "" : first.kind === "p" ? first.text : first.items[0];
  }

  function when(entry: NarratorEntry): string {
    const age = fmtDur(wall - entry.written_at);
    const at = replay ? `session time ${fmtTime(entry.clock, entry.clock)} · ` : "";
    return `${at}written ${age} ago in ${fmtDur(entry.duration_sec)}${entry.model ? ` by ${entry.model}` : ""}`;
  }
</script>

{#snippet text(body: string)}
  {#each blocks(body) as block, i (i)}
    {#if block.kind === "p"}
      <p>{@render inline(block.text)}</p>
    {:else}
      <ul>
        {#each block.items as item, j (j)}<li>{@render inline(item)}</li>{/each}
      </ul>
    {/if}
  {/each}
{/snippet}

{#snippet inline(line: string)}
  {#each split(line) as seg, i (i)}
    {#if seg.node}
      <button
        class="chip"
        class:on={ui.filter === seg.node}
        title="Filter the views to {seg.node}"
        onclick={() => ui.pickNode(seg.node!)}>{seg.text}</button
      >
    {:else}{seg.text}{/if}
  {/each}
{/snippet}

<section class="narrator" data-testid="narrator" bind:this={card}>
  <header>
    <button
      class="fold"
      aria-expanded={narrator.open}
      data-testid="nar-toggle"
      onclick={() => narrator.setOpen(!narrator.open)}
    >
      <span class="caret" class:open={narrator.open}>▸</span>
      <h2>Narrator</h2>
    </button>
    {#if narrator.pending}
      <span class="muted thinking" data-testid="nar-busy"
        >{narrator.pending === "ask" ? "answering…" : "thinking…"}</span
      >
    {:else if !narrator.open && latest.narration}
      <span class="muted peek">{headline(latest.narration.text)}</span>
    {/if}
    {#if narrator.available}
      <label class="auto" title="Narrate on open, on a phase change and every two minutes">
        <input type="checkbox" checked={narrator.auto} onchange={(e) => narrator.setAuto(e.currentTarget.checked)} />
        auto
      </label>
      <button data-testid="nar-refresh" onclick={() => narrator.narrate(true)}>refresh</button>
    {/if}
  </header>

  {#if narrator.open}
    <div class="body" transition:slide={{ duration: ms(180) }}>
      {#if narrator.status && !narrator.available}
        <p class="muted" data-testid="nar-off">
          The narrator is off: {narrator.status.reason ?? "no LLM configured"}. Configure an LLM for Hyperloom, then
          start the Inspector with <code>--narrate</code> (and <code>--llm-env-file</code> if its environment lacks the LLM
          settings) to get a running summary here.
        </p>
      {:else if latest.narration}
        <div
          class="narration"
          class:stale={Math.abs(sessionNow - latest.narration.clock) > STALE_SEC}
          data-testid="nar-text"
        >
          {@render text(latest.narration.text)}
        </div>
        <p class="meta muted">
          {when(latest.narration)}{#if latest.narration.focus}
            · focus: {latest.narration.focus}{/if}{#if Math.abs(sessionNow - latest.narration.clock) > STALE_SEC}
            · <span class="warn">describes session time {fmtTime(latest.narration.clock, latest.narration.clock)}</span
            >{/if}
        </p>
      {:else if narrator.status}
        <p class="muted">No narration at this point of the session yet{narrator.auto ? "" : "; press refresh"}.</p>
      {/if}

      {#if latest.answers.length || narrator.asking || narrator.explaining}
        <ol class="thread" data-testid="nar-thread">
          {#each latest.answers as a (a.id)}
            {@const lines = asked(a, store)}
            <li transition:slide={{ duration: ms(160) }}>
              {#if lines.head}<p class="q head" data-testid="nar-explain-head">{lines.head}</p>{/if}
              {#if lines.question}<p class="q">{lines.question}</p>{/if}
              <div class="a">{@render text(a.text)}</div>
              <p class="meta muted">{when(a)}</p>
            </li>
          {/each}
          {#if narrator.asking}
            <li>
              <p class="q">{narrator.asking}</p>
              <p class="muted">answering…</p>
            </li>
          {/if}
          {#if narrator.explaining}
            <li data-testid="nar-explaining">
              <p class="q head" data-testid="nar-explain-head">{explainHead(narrator.explaining.label)}</p>
              <p class="muted thinking">answering…</p>
            </li>
          {/if}
        </ol>
      {/if}

      {#if narrator.error}<p class="err" data-testid="nar-error">{narrator.error}</p>{/if}

      {#if narrator.available}
        {#if narrator.focus}
          <p class="focus" data-testid="nar-focus">
            Focus: <strong>{narrator.focus}</strong>
            <button onclick={unpin} disabled={!!narrator.pending}>clear</button>
          </p>
        {/if}
        <form onsubmit={ask}>
          {#if narrator.followUp}
            {@const label = narrator.followUp.label}
            <span class="followup" data-testid="nar-followup" title="Questions are about {label} until you remove it">
              <span class="label">about {label}</span>
              <button type="button" aria-label="Stop asking about {label}" onclick={() => narrator.clearFollowUp()}
                >✕</button
              >
            </span>
          {/if}
          <input
            data-testid="nar-question"
            bind:value={question}
            maxlength="500"
            placeholder={narrator.followUp
              ? `Ask a follow-up about ${narrator.followUp.label}`
              : "Ask about the run, e.g. why did a specialist crash?"}
          />
          <button type="submit" disabled={!question.trim() || !!narrator.asking}>ask</button>
          <button
            type="button"
            title="Emphasise this in every narration from now on"
            disabled={!question.trim() || !!narrator.pending}
            onclick={pin}>pin as focus</button
          >
        </form>
      {/if}
    </div>
  {/if}
</section>

<style>
  .narrator {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    overflow: hidden;
  }
  header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.4rem 0.75rem;
    font-size: 12px;
  }
  .narrator:has(.body) header {
    border-bottom: 1px solid var(--line);
  }
  h2 {
    margin: 0;
    font-size: 13px;
    color: var(--fg);
  }
  .fold {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    border: none;
    padding: 0;
  }
  .caret {
    display: inline-block;
    color: var(--muted);
    transition: transform 0.15s;
  }
  .caret.open {
    transform: rotate(90deg);
  }
  .peek {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .thinking {
    animation: pulse 1.2s ease-in-out infinite;
  }
  @keyframes pulse {
    50% {
      opacity: 0.4;
    }
  }
  .auto {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 0.25rem;
    color: var(--muted);
  }
  button {
    font: inherit;
    color: var(--accent);
    background: transparent;
    border: 1px solid var(--line);
    border-radius: 5px;
    padding: 1px 10px;
    cursor: pointer;
  }
  button:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .body {
    padding: 0.6rem 0.75rem;
    display: grid;
    gap: 0.5rem;
    font-size: 13px;
    line-height: 1.45;
  }
  .body p,
  .body ul {
    margin: 0;
  }
  .body ul {
    padding-left: 1.2rem;
  }
  .narration {
    display: grid;
    gap: 0.3rem;
    transition: opacity 0.2s;
  }
  .narration.stale {
    opacity: 0.6;
  }
  .narration > p:first-child {
    font-weight: 600;
  }
  .meta {
    font-size: 11px;
  }
  .warn {
    color: var(--stalled);
  }
  .err {
    color: var(--crashed);
  }
  .chip {
    padding: 0 5px;
    border-radius: 4px;
    border-color: color-mix(in srgb, var(--accent) 40%, transparent);
    background: color-mix(in srgb, var(--accent) 10%, transparent);
    font-size: 12px;
  }
  .chip.on {
    background: color-mix(in srgb, var(--accent) 30%, transparent);
  }
  .thread {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: 0.5rem;
    border-top: 1px solid var(--line);
    padding-top: 0.5rem;
  }
  .q {
    font-weight: 600;
  }
  .q::before {
    content: "Q: ";
    color: var(--muted);
  }
  .q.head::before {
    content: none;
  }
  .q.head {
    color: var(--accent);
  }
  .q.head + .q {
    font-weight: 400;
  }
  .a {
    display: grid;
    gap: 0.2rem;
  }
  .focus {
    font-size: 12px;
  }
  form {
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }
  .followup {
    flex: none;
    max-width: 40%;
    display: flex;
    align-items: center;
    gap: 0.2rem;
    padding: 0 0 0 6px;
    border: 1px solid color-mix(in srgb, var(--accent) 40%, transparent);
    border-radius: 4px;
    background: color-mix(in srgb, var(--accent) 10%, transparent);
    font-size: 12px;
    white-space: nowrap;
  }
  .followup .label {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .followup button {
    border: 0;
    padding: 0 5px;
    color: var(--muted);
  }
  .followup button:hover {
    color: var(--fg);
  }
  input:not([type]) {
    flex: 1;
    min-width: 0;
    font: inherit;
    color: var(--fg);
    background: var(--panel2);
    border: 1px solid var(--line);
    border-radius: 5px;
    padding: 3px 8px;
  }
</style>
