<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->
<script lang="ts">
  import { Tween } from "svelte/motion";
  import { follow } from "./motion";

  // A number that eases to each new value; `format` renders the in-between values too.
  let { value, format }: { value: number; format: (n: number) => string } = $props();

  const tween = new Tween(0, follow());
  $effect(() => {
    void tween.set(value, follow());
  });
</script>

<span class="num">{format(tween.current)}</span>

<style>
  .num {
    font-variant-numeric: tabular-nums;
  }
</style>
