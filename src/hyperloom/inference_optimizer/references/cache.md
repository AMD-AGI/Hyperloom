# Cache Topology & Cold-start Discipline

SGLang/vLLM on ROCm route hot fused kernels (RMSNorm / attention / MoE / GEMM /
RoPE) through `aiter`, which JIT-compiles per-shape variants on first sight and
caches `.so` on disk. First launch of a fresh (model, dtype, TP, `max_model_len`,
`max_num_seqs`, `gpu_memory_utilization`) signature can spend 30+ min in `hipcc`
for 671B FP8 MoE; later launches reuse the cache in seconds.

## Cache locations

| Cache | Path | Clear |
|---|---|---|
| aiter JIT (primary cold-start cost) | The runtime-selected JIT directory (`AITER_JIT_DIR`, otherwise the package `jit/` or initialized user cache), containing serving `.so` modules and `build/` staging | Manual staging cleanup is limited to `build/`. Serving modules are invalidated through the shared JIT transaction, with the scope chosen by the caller. |
| Triton | `~/.triton/cache/` (resolves via `$HOME`) | `rm -rf ~/.triton/cache` |
| torch.compile / Inductor | `/tmp/torchinductor_<user>/` (override `$TORCHINDUCTOR_CACHE_DIR`) | `rm -rf /tmp/torchinductor_root` |

An AITER runtime-JIT source patch invalidates the full serving-module set together
with `build/`; revert restores its baseline and removes candidate artifacts in that scope.
CSV/GEMM registry preparation invalidates only its selected modules, preserving
unrelated serving modules. Full invalidation can require substantial first-use
compilation; its cost depends on the installed cache and workload. Do not delete
serving modules manually or infer patch success from a warm, stale module.

`sgl_kernel` (`site-packages/sgl_kernel/common_ops.*.so`) is build-time only;
only `kernel_opt` / `integrate` may rebuild it.

## Cold-start triggers

First launch on this pod; change to `--max-model-len` / `--max-num-seqs` /
`--gpu-memory-utilization` / `--cuda-graph-max-bs` / `--quantization` /
`--enable-torch-compile`; pod rebuild; manual cache `rm`; aiter source patch.

## Cold-start diagnostics and benchmark limits

Cold-start detection remains diagnostic; it does not select a separate
benchmark deadline. Every actual benchmark spawn uses
`INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC` (default `7800` seconds), including
weight load, JIT startup, and accuracy evaluation. After real server readiness,
`INFERENCE_OPTIMIZER_BENCHMARK_SILENCE_TIMEOUT_SEC` (default `600` seconds) also
bounds output silence. Neither startup before readiness nor a server-less
scriptable workload arms that silence timer. Both values must be finite and
positive; output never extends the hard deadline.

Inspect server/compiler logs and cache artifacts before deciding a repeated cold
start was interrupted. Quiet logs alone do not prove a hang, and busy logs alone
do not prove useful progress. Profile, KernelForge, GEAK, and LLM budgets remain
separate. Session `--max-hours` and cancellation still apply; session shutdown is
cooperative and cannot guarantee termination of a frozen Coordinator.

See [Benchmark config](benchmark.md#benchmark-deadlines) for output-buffering
limitations and the full benchmark policy.
