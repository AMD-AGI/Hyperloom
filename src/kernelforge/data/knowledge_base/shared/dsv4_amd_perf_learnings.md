# DeepSeek-V4 on AMD MI355X — perf/correctness learnings (2026-04-24)

Findings from bring-up of an SGLang DSv4 enablement branch on 8× MI355X / gfx950, running DeepSeek-V4-Flash-Base (FP8, 295 GB, 43 layers, 256 routed experts, compressed attention).

## 1. Correctness fix (root cause of gibberish output)

The PR creates `wo_a` as a bf16 `ColumnParallelLinear` when `SGLANG_OPT_FP8_WO_A_GEMM=False` (default) but the HF release ships `wo_a` as `fp8_e4m3fn` weight + `float32` block-scale `[64, 32]`. The PR's dequant path `_dequant_fp8_wo_a` only fires on `SGLANG_DSV4_MODE=="2604" AND SGLANG_DSV4_FP4_EXPERTS=true` and asserts `scale.dtype == fp8_e8m0fnu`. Neither condition applies to Flash-Base. Net: fp8 bytes copied into bf16 params without scale multiplication → wo_a values wrong per-block by 1–10000× → compounding across 43 layers → deterministic gibberish.

Fix (applied in `sglang_v4_pr/python/sglang/srt/models/deepseek_v4.py`):
- Relax `_dequant_fp8` to accept `float32` scale dtype.
- Replace the `SGLANG_DSV4_MODE=="2604"` gated branch with a layout-detecting helper `_maybe_dequant_fp8_wo_a` that probes the first wo_a tensor dtype. FP8 → dequant inline. bf16 → drop stale scale.

**Signal it worked**: prompt "The capital of France is" changed from ", Eisenhower星期二..." to " Paris. The capital of Germany is Berlin...".

**Lesson**: checkpoint variant detection must be automatic for public distribution — env-var gated paths are fragile and the existing branch was wrong for the actual HF release.

## 2. Performance: torch.compile wrap on `ref_sparse_attn_decode`

Baseline single-request decode (8K prompt, 256 out): **324 ms/token** after fix.

Pre-optimization: single-request at `/generate` with 16 out tokens took ~49s → ~3 s/token (attention is 100% pytorch eager `ref_sparse_attn_decode` — `q @ K^T → softmax → @ V` with no fused kernel, `SGLANG_HACK_FLASHMLA_BACKEND=torch` on AMD).

Refactor: extracted the tensor-only inner compute into `_sparse_attn_decode_inner` (same math, no Python-level control flow), wrapped with `torch.compile(dynamic=True, fullgraph=False)`. Applied in `python/sglang/srt/flashmla_tests/ref.py`.

Measured single-request: **324 ms/token** — **~10× speedup** over eager. First call has compile overhead (~35s extra). Warm recompiles occur when shapes change significantly; `dynamic=True` mitigates but doesn't eliminate.

**Lesson**: for AMD ROCm Triton codegen, torch.compile on modest tensor-only subgraphs is reliably giving ~5–10× over pytorch eager, even without MFMA-native kernels. Low-risk first-pass optimization. Non-tensor-only wrappers (dataclass param accesses, `.item()` calls) cause graph breaks that silently kill the speedup — extract the tensor core first.

**Critical caveat — torch.compile is shape-sensitive.** Attempted the same treatment on `fp8_paged_mqa_logits_torch` (the indexer's q·K scoring). Vectorized the Python for-loop + wrapped the core with `torch.compile(dynamic=True)`. Result: **3× SLOWER** (324 → ~1000 ms/tok). Root cause: decode steps feed growing `padded_seq_len` (64, 128, 192, …), so `dynamic=True` still triggers a recompile per shape bucket. Additionally the vectorized path materializes larger intermediates than the per-batch loop at B=1.

**Rule of thumb**: torch.compile only helps when the compiled function sees **stable input shapes**. `ref_sparse_attn_decode` has stable shape (topk=512 is fixed by the indexer). Any function whose shape grows with seq_len during decode is NOT a torch.compile target unless you first pad inputs to a fixed bucket size. Reverted to the Python-loop version; this function is a candidate for a real Triton kernel, not torch.compile.

## 3. CUDA graph capture: DOES NOT WORK on this V4 path

Tried `--cuda-graph-max-bs 8` (only 4 buckets: 1/2/4/8). Capture stuck at 0/4 for 20+ minutes across all 4 DP ranks.

Root cause: the V4 attention path has:
- `seq_len = int(seq_lens[i].item())` in `fp8_paged_mqa_logits_torch` — forces GPU→CPU sync inside loop.
- `num_pages_per_batch.max().item()` etc. in other indexer paths.
- Dynamic tensor shapes in compressor writes.

CUDA graphs require fully static shapes and no CPU syncs. Can't capture until those paths are rewritten with host-side shape knowledge + static-shape kernels.

**Lesson**: don't try CUDA graphs on PR code that has `.item()` sprinkled in hot paths. Either (a) rewrite the paths first, (b) rely on torch.compile for dispatch-overhead savings, or (c) wait for HIP/Triton kernel replacements.

## 4. aiter JIT cold-start pathology

First run with `SGLANG_USE_AITER=1` triggers `module_rmsnorm` JIT compile — **1361 cpp variants** under default `MAX_JOBS=8` parallel. Takes 30+ minutes. SGLang's internal warmup uses a 600s ReadTimeout so cold start blocks server launch indefinitely.

Mitigations (all applied):
- `--skip-server-warmup` so server boots before first /generate drives JIT.
- `MAX_JOBS=128` (on a 128-core node) — cuts rmsnorm compile to ~5 min.
- Persistent bind-mount of `/sgl-workspace/aiter/aiter/jit` onto host — subsequent runs are <1 min.

**Lesson**: always persist the aiter JIT dir across container restarts. Consider `BUILD_AITER_ALL=1` at image build time. Also: the 1361-variant fan-out comes from aiter generating every (N, dtype, quant_flag) combination eagerly — most DSv4 shapes use <10 variants, so a narrowing patch to aiter gen_rmsnorm would cut 100× compile.

## 5. Hidden PR bugs that block boot (file upstream)

Beyond the accuracy bug in §1:

- `python/sglang/srt/flashmla_tests/kernelkit/__init__.py` imports `bench`, `from .bench import bench_by_cuda_events, bench_kineto`. `bench.py` was never committed (probably `.gitignore`d). Torch fallback path cannot import. Stub fix required.
- `_load_deepseek_temp_model` writes config.json content to a FILE path then calls `AutoConfig.from_pretrained` on it — HF expects a directory. Fallback doesn't actually work as written.
- `model_type=deepseek_ref` in the config dataclass but the HF checkpoint has `deepseek_v4`. Fallback substring match misses. Patch the checkpoint (model_type=deepseek_v3, archs=DeepseekV4ForCausalLM) via a symlinked sibling dir.
- `tokenizer_class=PreTrainedTokenizerFast` in patched tokenizer_config.json is required — otherwise HF routes to LlamaTokenizerFast which unconditionally constructs LlamaTokenizer slow class → wants `tokenizer.model` SPM that DSv4 doesn't ship.

## 5b. UPDATED PROFILE INTERPRETATION — the "AR is bottleneck" was a misread

Initial analysis of the kineto trace attributed 4.5 ms per AR and concluded
all-reduce was the bottleneck. That was wrong. Follow-up microbench and A/B
proved the kernel is fine — the profile's 4.5 ms dur was a **straggler wait**
polluting the kernel body time.

**Evidence**:

1. `benchmark/kernels/all_reduce/benchmark_aiter.py` at torchrun --nproc 4 on
   MI355X shows SGLang custom AR = **0.04 ms at 32 KB**, 0.05 ms at 2 MB,
   0.64 ms at 64 MB. Our decode-step messages are ~8 KB. Real kernel body is
   <50 μs.

2. A/B at identical shape, one flag difference:
   - `--tp 4` (no DP-attn): **21.6 s / 64 tok = 338 ms/tok**
   - `--tp 4 --dp 4 --enable-dp-attention`: **45.6 s / 64 tok = 712 ms/tok**

   The PR's recommended command is **2× slower** for single-request decode
   because `--enable-dp-attention` creates rank imbalance (3 of 4 DP-attn
   ranks idle at batch=1). Every layer's AR then waits for the single active
   rank. aiter::cross_device_reduce uses IPC + atomic sync — the kernel body
   blocks until all ranks arrive, so the "dur" in the trace = max(pre-AR
   compute across ranks) + actual AR exec. At B=1 the imbalance inflates dur
   ~100×.

**Recommendation for low-concurrency serving on AMD**: drop
`--enable-dp-attention`. The PR's `--tp 4 --dp 4 --enable-dp-attention` is
tuned for high-throughput regimes where each DP-attn rank has real batched
work. For B=1, pure `--tp 4` is significantly faster and doesn't change
correctness.

**Open question**: at what concurrency does DP-attn cross over and become
faster? Worth sweeping c=1, 4, 16, 64 to find the knee.

## 5a. PROFILE FINDING — all-reduce is the dominant bottleneck (NOT attention)

Kineto trace of 8 decode steps at `tp=4 dp=4 --enable-dp-attention`, isl=256/osl=16/c=1:

```
aiter::cross_device_reduce_1stage<bf16,4>    523 calls  2341 ms  TOP
aiter::cross_device_reduce_2stage<bf16,4>     86 calls  1672 ms  #2
sglang::inplace_fused_experts                301 calls   168 ms
_gemm_a16_w16_atomic_kernel                  301 calls    32 ms
fused_moe_kernel                             602 calls    20 ms
```

- **4012 ms of AR** vs ~220 ms of all compute combined.
- 609 AR calls / 8 decode steps = **~76 AR per decode step**. That's attention output AR + MoE TP-AR + DP scatter/gather AR × 43 layers.
- Per-call cost ~4.5 ms for bf16 messages that should land in <100 μs on MI355X's Infinity Fabric. 100× slower than bandwidth-bound — these are launch/fence latency-bound, not BW-bound.

**Implication**: fixing the attention kernel alone has diminishing returns. Per-step AR cost (~500 ms) dwarfs the entire torch.compile'd attention path (~10-30 ms). The ~10× perf ceiling is communication, not compute.

**New top priorities**:
1. **Reduce AR count** — fuse multiple ARs per layer into one (MSCCLPP ring? flashinfer-allreduce-fusion? aiter primitive?). Each AR saved = ~4.5 ms/step.
2. **Overlap comm with compute** — tokens are currently serialized on AR completion. Comm/compute overlap via multi-stream is the obvious fix but the V4 backend's `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP` path may not be wired for AMD.
3. **Check if custom-all-reduce is the right path** — try `--disable-custom-all-reduce` (fall back to RCCL) and measure. aiter's `cross_device_reduce_*` kernels at 4.5 ms/tiny-message look suspicious; pure RCCL might be faster.
4. **Larger-message AR** — if many small ARs can be batched into a single AR-of-cat-buffer, the latency amortizes. Profile the per-layer work to see if multiple ARs could be safely coalesced.

## 6. Kernel-level opportunities that remain (priority order)

On AMD MI355X with the PR's env-var set, 7 of 7 V4 attention ops run in pytorch eager / Triton. The ref-attention torch.compile wrap is the only real fusion so far. Real kernel work needed:

| Op | Current | Target | Expected gain |
|---|---|---|---|
| Sparse-attn compute | torch.compile'd pytorch ref | Triton sparse-decode (or port user's VSA CK kernel) | 3–5× over current, 30× over eager |
| FP8 paged MQA logits (indexer) | pytorch Python loop (`.item()` per batch) | Triton or aiter batched kernel | 5–10× for batched, unblocks CUDA graphs |
| Compressor (c4/c128 cache writes) | pytorch eager | HIP port of PR's `c4.cuh` / `c128.cuh` | 2–3× on memory-bound op |
| Indexer top-512 | torch.topk | Triton radix / bitonic top-K | 2–3× |
| KV quant + RoPE pack | Triton (works) | Add MI355X autotune block-sizes | 20–30% |
| RMSNorm + FP8 quant | aiter CK-tile | trim variant fan-out (1361 → ~10) | 100× compile-time only |
| SWA prefill index prep | pytorch | Tilelang gfx950 fix upstream | small |

After these, CUDA graphs become viable (with `.item()` calls removed) — another 1.5–2× dispatch-overhead elimination.

**Kernel-agents-specific recommendation**: the "Triton sparse-decode kernel to replace `ref_sparse_attn_decode`" is the single highest-leverage work item. Inputs: `q[b, s_q, h_q=64, d_qk=576]`, pre-gathered `K[b, s_q, topk≤512, d_qk=576]` (already sparse-selected by indexer), attn_sink `[h_q]`. Output: `o[b, s_q, h_q, d_v=512]`. This is the FlashAttention-decode pattern with a pre-gathered K — the tile layout and masking are standard; no new silicon features needed. AMD MFMA `v_mfma_f32_16x16x16_bf16` fits this shape cleanly. An autotuned Triton kernel should land 5–8× over the torch.compiled pytorch ref, and 30× over the original eager.

## 7. Env vars that must be set (for reproducibility)

```
SGLANG_OPT_USE_FUSED_COMPRESS=false
SGLANG_OPT_USE_OLD_COMPRESSOR=true
SGLANG_OPT_USE_TILELANG_SWA_PREPARE=false
SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=false
SGLANG_OPT_USE_FUSED_HASH_TOPK=false
SGLANG_HACK_FLASHMLA_BACKEND=torch
SGLANG_OPT_DEEPGEMM_HC_PRENORM=false
SGLANG_OPT_USE_TILELANG_MHC_PRE=false
SGLANG_OPT_USE_TILELANG_MHC_POST=false
SGLANG_TOPK_TRANSFORM_512_TORCH=1
SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
SGLANG_DSV4_FP4_EXPERTS=false     # Flash-Base is FP8; Flash (MXFP4) needs =true
SGLANG_OPT_DPSK_V4_RADIX=0
SGLANG_OPT_USE_OVERLAP_STORE_CACHE=false
SGLANG_OPT_USE_FUSED_STORE_CACHE=false
SGLANG_FORCE_TRITON_MOE_FP8=1
SGLANG_USE_AITER=1
SGLANG_USE_ROCM700A=1
MAX_JOBS=128                      # for aiter JIT
```

And launch with `--tp 4 --dp 4 --enable-dp-attention --disable-cuda-graph --skip-server-warmup` on 4 GPUs.

## 8. Numbers snapshot (MI355X, Flash-Base FP8)

A/B-validated single-request decode (64-token output, "The capital of France is" prompt, 30-iter median latency):

| Config | ms/tok | speedup vs initial | notes |
|---|---|---|---|
| Initial (gibberish, eager attn, PR-default `tp4 dp4 dp-attn`) | ~3000 | 1× | broken numerics |
| + wo_a dequant fix (coherent) | ~3000 | 1× | correctness only |
| + `torch.compile` on `_sparse_attn_decode_inner` | **338** | 8.9× | attention compile-fuse |
| + Triton kernel from kernel-agents | ~336 | ~9× | kernel saves ~25 μs × 43 = ~1ms (negligible E2E) |
| **+ DROP `--dp 4 --enable-dp-attention` → pure `--tp 4`** | **same 336** | ~9× | now consistent without straggler variance |
| (PR command exactly: `--tp 4 --dp 4 --enable-dp-attention`) | **712** | 4.2× only | DP-attn at B=1 is a 2× tax |
| + `--enable-piecewise-cuda-graph` | crash | n/a | torchdynamo incompatibility on V4 path |
| + `--cuda-graph-max-bs 8` (full CUDA graph) | hangs | n/a | `.item()` calls in V4 indexer block capture |

**Profile of current best (pure TP4 + Triton, 8 decode steps)**:
```
8232 elementwise_kernel       732 ms  (~91 ms/step of tiny pytorch ops)
 522 cross_device_reduce      544 ms  (~68 ms/step real AR)
4797 elementwise_unroll #1    275 ms
4116 elementwise_unroll #2    228 ms
  87 cross_device_reduce_2st  157 ms
  60 various GEMM kernels     ~120 ms
 301 _sparse_attn_decode_v3   35.7 ms (Triton kernel ✓)
```

~30,000 small kernel launches per 8 decode steps → ~3750 launches/step → ~190 ms/step
of pure pytorch eager dispatch overhead. **The remaining bottleneck is op-launch
overhead, not kernel execution.**

Expected landing if CUDA graph capture becomes possible (after `.item()` removal
in `fp8_paged_mqa_logits_torch`, indexer compressor, and the Compressor
`_get_state_pool` paths): **~50-80 ms/tok** = 6× more on top of current = ~50× over
initial.

## 9. Config change with biggest E2E win discovered

Drop `--dp 4 --enable-dp-attention` for low-concurrency decode. The PR's recommended
command is tuned for high-throughput batched serving. At batch=1, only one of four
DP-attn ranks has real work, the other three idle, every per-layer AR waits for
the active rank → 2× E2E penalty observed reproducibly. Pure `--tp 4` does not
have this rank-imbalance issue.

This is a **runtime config** change, no code changes needed. The PR's setup script
should branch on expected concurrency or document this.
