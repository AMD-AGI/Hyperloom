<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<!-- A collapsible drawer section; `open` is the initial state only. -->
<script lang="ts">
  import type { Snippet } from "svelte";

  let {
    title,
    open = true,
    testid,
    children,
  }: { title: string; open?: boolean; testid?: string; children: Snippet } = $props();
</script>

<details {open} data-testid={testid}>
  <summary>{title}</summary>
  <div class="body">{@render children()}</div>
</details>

<style>
  details {
    border-top: 1px solid var(--line);
    padding: 0.4rem 0;
  }
  summary {
    cursor: pointer;
    font-weight: 600;
    font-size: 12px;
    color: var(--muted);
  }
  .body {
    padding-top: 0.35rem;
    overflow-x: auto;
  }
</style>
