# Cache Topology & Cold-start Discipline

SGLang/vLLM on ROCm route hot fused kernels (RMSNorm / attention / MoE / GEMM /
RoPE) through `aiter`, which JIT-compiles per-shape variants on first sight and
caches `.so` on disk. First launch of a fresh (model, dtype, TP, `max_model_len`,
`max_num_seqs`, `gpu_memory_utilization`) signature can spend 30+ min in `hipcc`
for 671B FP8 MoE; later launches reuse the cache in seconds.

## Cache locations

| Cache | Path | Clear |
|---|---|---|
| aiter JIT (primary cold-start cost) | `<aiter pkg root>/jit/` (resolved via `import aiter`; wheel installs hold ~80 pre-built `.so` here, plus runtime-JIT staging under `jit/build/<module>/build/`) | `rm -rf <aiter pkg root>/jit/build/` (clears JIT staging only; do NOT delete `jit/*.so` — those are wheel-bundled) |
| Triton | `~/.triton/cache/` (resolves via `$HOME`) | `rm -rf ~/.triton/cache` |
| torch.compile / Inductor | `/tmp/torchinductor_<user>/` (override `$TORCHINDUCTOR_CACHE_DIR`) | `rm -rf /tmp/torchinductor_root` |

`sgl_kernel` (`site-packages/sgl_kernel/common_ops.*.so`) is build-time only;
only `kernel_opt` / `integrate` may rebuild it.

## Cold-start triggers

First launch on this pod; change to `--max-model-len` / `--max-num-seqs` /
`--gpu-memory-utilization` / `--cuda-graph-max-bs` / `--quantization` /
`--enable-torch-compile`; pod rebuild; manual cache `rm`; aiter source patch.

## Auto-detection + timeout

The baseline/profile executors count aiter `.so` files (**< 20 = COLD**) and
pick a subprocess timeout accordingly: COLD → 3600s, WARM → 2400s
(`task.params['timeout_sec']` always wins). Each launch logs a
`baseline_executor: ...` marker and the cache state lands in the
`Preflight diagnostics:` block. If COLD_START repeats across retries the JIT was
killed mid-`hipcc` — bump `INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC=5400`
(don't just relaunch). Override the probe dir via
`INFERENCE_OPTIMIZER_AITER_JIT_DIR`.
