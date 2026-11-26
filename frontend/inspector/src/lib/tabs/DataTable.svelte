<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<script lang="ts" generics="T">
  import { cubicOut } from "svelte/easing";
  import { flip } from "svelte/animate";
  import { ms } from "../motion";
  import { ANIMATE_MAX, type Col } from "./rows";

  let {
    rows,
    cols,
    key = (r: T) => r,
    onpick,
    selected = () => false,
    testid,
  }: {
    rows: T[];
    cols: Col<T>[];
    key?: (row: T) => unknown;
    onpick?: (row: T) => void;
    selected?: (row: T) => boolean;
    testid?: string;
  } = $props();

  // Short lists reorder with FLIP and flash new rows; long ones (the live session has thousands) just update.
  const animated = $derived(rows.length <= ANIMATE_MAX);
  const dur = $derived(animated ? ms(350) : 0);

  function fresh(_: Element, { duration }: { duration: number }) {
    return {
      duration,
      css: (t: number) =>
        `background-color: color-mix(in srgb, var(--accent) ${Math.round((1 - t) * 22)}%, transparent)`,
    };
  }
  const text = (v: unknown): string => (v == null ? "" : String(v));
  const cls = (c: Col<T>, r: T): string => (typeof c.cls === "function" ? c.cls(r) : (c.cls ?? ""));
</script>

<table data-testid={testid}>
  <thead>
    <tr
      >{#each cols as c (c.head)}<th data-hint={c.hint} class:hint={!!c.hint}>{c.head}</th>{/each}</tr
    >
  </thead>
  <tbody>
    {#each rows as r (key(r))}
      <tr
        class:click={!!onpick}
        class:sel={selected(r)}
        onclick={() => onpick?.(r)}
        animate:flip={{ duration: dur, easing: cubicOut }}
        in:fresh={{ duration: animated ? ms(1600) : 0 }}
      >
        {#each cols as c (c.head)}
          {@const v = text(c.text(r))}
          <td class={cls(c, r)} title={v.length > 60 ? v : undefined}>{v}</td>
        {/each}
      </tr>
    {/each}
  </tbody>
</table>

<style>
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  th {
    position: sticky;
    top: 0;
    z-index: 1;
    background: var(--panel);
    text-align: left;
    font-weight: 600;
    color: var(--muted);
    padding: 4px 8px;
    border-bottom: 1px solid var(--line);
    white-space: nowrap;
  }
  th.hint {
    cursor: help;
    text-decoration: underline dotted color-mix(in srgb, var(--muted) 60%, transparent);
    text-underline-offset: 3px;
  }
  td {
    padding: 3px 8px;
    border-bottom: 1px solid color-mix(in srgb, var(--line) 60%, transparent);
    white-space: nowrap;
    max-width: 420px;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  td:global(.trunc) {
    max-width: 360px;
  }
  td:global(.mono) {
    font-family: var(--mono);
    font-size: 11px;
  }
  tr.click {
    cursor: pointer;
  }
  tr.click:hover {
    background: var(--panel2);
  }
  tr.sel {
    background: color-mix(in srgb, var(--accent) 16%, transparent);
    box-shadow: inset 3px 0 0 var(--accent);
  }
</style>
