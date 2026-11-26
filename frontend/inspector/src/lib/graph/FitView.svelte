<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<script lang="ts">
  // Refits the viewport whenever `key` changes; lives inside <SvelteFlow> for useSvelteFlow().
  import { useSvelteFlow } from "@xyflow/svelte";
  import { untrack } from "svelte";
  import { ms } from "../motion";

  let { key }: { key: number } = $props();
  const flow = useSvelteFlow();

  $effect(() => {
    // fitView reads and writes the viewport state; untracked, so only `key` reruns this.
    if (key > 0) untrack(() => void flow.fitView({ duration: ms(400), padding: 0.08 }));
  });
</script>
