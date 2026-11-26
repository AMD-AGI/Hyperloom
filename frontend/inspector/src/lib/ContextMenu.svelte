<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<!-- The page's one right-click menu, opened by the views through `ui.openMenu`: Explain the item through the
     narrator, or Open it as a left click does. Escape, a click elsewhere, a scroll or another menu closes it; the
     drawer's Explain button is the keyboard path to the same action. -->
<script lang="ts">
  import { tick } from "svelte";
  import { menuEntries, placeMenu, type MenuEntry } from "./contextmenu";
  import { labelFor, questionFor } from "./explain";
  import type { NarratorStore } from "./narrator-store.svelte";
  import type { InspectorStore } from "./store.svelte";
  import type { UiState } from "./ui.svelte";

  let { store, ui, narrator }: { store: InspectorStore; ui: UiState; narrator: NarratorStore } = $props();

  let el = $state<HTMLDivElement>();
  let w = $state(0);
  let h = $state(0);
  let vw = $state(0);
  let vh = $state(0);

  const menu = $derived(ui.menu);
  const entries = $derived(menu ? menuEntries(menu.ref, narrator.available, narrator.explaining?.ref ?? null) : []);
  // Until the menu is measured it sits at the pointer; then it moves inside the viewport.
  const at = $derived(menu ? placeMenu(menu.x, menu.y, w, h, vw, vh) : { left: 0, top: 0 });

  function run(entry: MenuEntry): void {
    const m = menu;
    if (!m || entry.disabled) return;
    ui.closeMenu();
    if (entry.id === "open") m.open();
    else if (m.ref) void narrator.explain(m.ref, questionFor(m.ref, store), labelFor(m.ref, store));
  }

  // While a menu is open: Escape closes it (before the drawer's own Escape sees the key), as do a press outside
  // it, a scroll anywhere (the item would move away from it), a resize and leaving the window. Scrolls count from
  // the next frame on: scroll events are sent a frame late, so one that brought the item into view would close the
  // menu it opened.
  $effect(() => {
    if (!menu) return;
    const close = () => ui.closeMenu();
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.stopPropagation();
      close();
    };
    const onDown = (e: PointerEvent) => {
      if (!el?.contains(e.target as Node)) close();
    };
    const opts = { capture: true };
    window.addEventListener("keydown", onKey, opts);
    window.addEventListener("pointerdown", onDown, opts);
    const frame = requestAnimationFrame(() => window.addEventListener("scroll", close, opts));
    window.addEventListener("resize", close);
    window.addEventListener("blur", close);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("keydown", onKey, opts);
      window.removeEventListener("pointerdown", onDown, opts);
      window.removeEventListener("scroll", close, opts);
      window.removeEventListener("resize", close);
      window.removeEventListener("blur", close);
    };
  });

  // Each new menu takes the focus on its first entry, so the arrow keys and Enter work at once.
  $effect(() => {
    if (!menu) return;
    void tick().then(() =>
      el?.querySelector<HTMLButtonElement>("button:not(:disabled)")?.focus({ preventScroll: true }),
    );
  });

  // Up and Down move between the entries, wrapping; Tab closes the menu like a click elsewhere.
  function onkeydown(e: KeyboardEvent): void {
    if (e.key === "Tab") return ui.closeMenu();
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    e.preventDefault();
    const items = [...(el?.querySelectorAll<HTMLButtonElement>("button:not(:disabled)") ?? [])];
    const i = items.indexOf(document.activeElement as HTMLButtonElement);
    const step = e.key === "ArrowDown" ? 1 : -1;
    items[(i + step + items.length) % items.length]?.focus();
  }
</script>

<svelte:window bind:innerWidth={vw} bind:innerHeight={vh} />

{#if menu}
  <div
    bind:this={el}
    bind:clientWidth={w}
    bind:clientHeight={h}
    class="menu"
    role="menu"
    tabindex="-1"
    aria-label="item actions"
    data-testid="context-menu"
    style:left="{at.left}px"
    style:top="{at.top}px"
    {onkeydown}
    oncontextmenu={(e) => e.preventDefault()}
  >
    {#each entries as entry (entry.id)}
      <button role="menuitem" disabled={entry.disabled} data-testid="cm-{entry.id}" onclick={() => run(entry)}>
        {entry.label}
      </button>
    {/each}
  </div>
{/if}

<style>
  .menu {
    position: fixed;
    z-index: 60;
    display: flex;
    flex-direction: column;
    min-width: 9rem;
    padding: 4px;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 6px;
    box-shadow: 0 6px 20px rgb(0 0 0 / 0.25);
    font-size: 12px;
  }
  .menu:focus {
    outline: none;
  }
  button {
    font: inherit;
    color: var(--fg);
    background: none;
    border: 0;
    border-radius: 4px;
    padding: 4px 10px;
    text-align: left;
    cursor: pointer;
  }
  button:hover:not(:disabled),
  button:focus-visible {
    background: var(--panel2);
    outline: none;
  }
  button:disabled {
    color: var(--muted);
    cursor: default;
  }
</style>
