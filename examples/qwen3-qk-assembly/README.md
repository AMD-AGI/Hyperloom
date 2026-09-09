<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Qwen3 Q/K normalization and RoPE assembly example

This opt-in example executes hand-written gfx950 assembly through Forge's
assembler and standalone HIP loader. It combines BF16 Q/K RMSNorm and full-width
NeoX RoPE in one kernel. The value tensor, attention implementation, projections,
weights, scheduler, and token budget retain their existing vLLM behavior.

The measured model is Qwen3-4B, BF16, tensor parallel size 1, on one MI355X with
ROCm 7.2.3 and vLLM 0.25.1. The integration selects 32 query heads, eight KV heads,
and head dimension 128. Other model shapes retain the original forward function.
The assembly requires gfx950; this is an experimental example, not a universal
Qwen3 optimization or a model-quality certification.

## Install and select

Use an isolated environment with the existing ROCm PyTorch/vLLM installation and
this Hyperloom PR installed. From the Hyperloom repository root:

```bash
python -m pip install --no-deps -e examples/qwen3-qk-assembly
VLLM_PLUGINS=forge_qwen3_assembly vllm serve /path/to/Qwen3-4B \
  --host 127.0.0.1 --tensor-parallel-size 1 --max-model-len 2048 \
  --max-num-seqs 64 --max-num-batched-tokens 2048 \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64],"cache_dir":"/tmp/qwen3-forge-assembly-fresh"}'
```

Choose a fresh compilation cache for each changed candidate. vLLM's existing
model cache does not necessarily account for a plugin replacing a Python method.
Use the documented `VLLM_PLUGINS` selection so worker processes activate the same
plugin. A launch succeeding is insufficient: confirm `forge_qk_norm_rope_h128`
appears in the GPU trace. Two earlier experiments that executed the original
model were rejected as unevaluated assembly candidates; changing the private
cache alone did not establish correct worker dispatch.

Baseline runs omit this plugin and use their own fresh cache. Keep model,
requests, graph settings, scheduler configuration, and warmup identical. The
plugin changes methods only within its selected Python processes; it does not
edit installed vLLM, AITER, FlyDSL, or model checkpoint files.

## Numerical and launch contract

`kernel.py` makes the ABI explicit: seven pointers, FP32 epsilon, query/KV head
counts, and the QKV byte stride. It forwards the current PyTorch stream. Build and
load occur before graph replay. The kernel reads packed QKV, writes new Q and K
tensors, and leaves V and all inputs unchanged. Position values must index valid
rows of the BF16 cosine/sine cache. The tested token range is 1 through 2048.
The wrapper rejects tensor spans of 4 GiB or greater because relative offsets
are 32-bit; complete base-pointer additions propagate carry across a 4-GiB
absolute-address boundary. These are different constraints.

FP64 RMS reduction and normalization are converted through FP32 to BF16 before the
weight multiply and at each native BF16 RoPE product/addition. Expanded testing
found two out-of-tolerance Q elements in the earlier FP32 implementation at seed
910, 2048 tokens, and input scale 0.5. The regression test preserves that input
and the original `rtol=0.01, atol=0.01`; the precision change repairs the kernel
rather than relaxing its oracle. Historical FP32 timings do not describe this
repaired source.

DPP reductions and reciprocal-square-root instructions retain their required
dependency spacing. Removing that spacing produced wrong results in an earlier
candidate, despite successful assembly. Exact generated-token equivalence is
not promised. Keep production numerical and model-quality requirements in the
driver when expanding this experiment.

```bash
pytest src/kernelforge/tests/test_assembly_hip.py
pytest src/kernelforge/tests/test_assembly_hip_gpu.py
```

The GPU test covers an independent FP64 reference with BF16 rounding, multiple
scales and token counts, preserved inputs, new pointers, nondefault-stream graph
replay, and rejection of a deliberately wrong instruction edit. Separate case
validation also exercises valid tensors crossing a 4-GiB address boundary.

This is a fusion implemented in assembly. Compare a similarly fused high-level
kernel separately before attributing the entire end-to-end gain exclusively to
instruction scheduling. A larger microbenchmark ratio is not a model-throughput
ratio, and decode throughput gains do not imply improved time to first token.
The [measured case card](../../src/kernelforge/data/local_knowledge/languages/assembly/cases/qwen3_qk_rope_gfx950.md)
records the serving workload, same-fusion Triton comparison, and numerical limits.
