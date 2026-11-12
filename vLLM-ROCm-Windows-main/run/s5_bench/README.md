# S5 native paged-attention benches + the ablation method

Harnesses from the S5 (native decode attention) investigation and the decode-lever ablation.
Run with a clean-VRAM MoE model (see `docs/s5-attention-lever-and-aiter-rdna3.md`;
gemma-4-26B spills on a 20GB card -> use ERNIE-4.5-21B-A3B).

- `perf_ernie.py`  -- ERNIE-4.5-21B-A3B A/B: TRITON_ATTN baseline vs S5 (ROCM_ATTN + native decode).
  `S5_ROUTE=1` on, `OPT=1` for the MoE-decode-GEMV + wvSplitK opt path. Prints free-VRAM (proves no
  spill), native-call count, coherence. Clean result: 79.2 baseline vs 75.2 S5 (-5%).
- `perf_ablation.py` -- the RELIABLE decode map (torch.profiler is unreliable here): no-op a component
  (dense / moe / attn / attncompute) under cudagraph, the tok/s delta = its real cost. Finding:
  attention COMPUTE ~27% is the biggest lever; dense ~3.3%; MoE experts ~0%.
- `mb_native.py` / `mb_kpa2d.py` -- isolated microbench: native paged_attention_v1 vs Triton
  kernel_paged_attention_2d on the v0 cache (separate processes to dodge the `_C` namespace clash).
  native is ~3.2x faster.
- `mb_sliding.py` / `mb_native_correct.py` -- correctness of the native kernel + the sliding-window
  mask vs an fp32 reference (rel err ~5e-4).

NOTE: these load an 18GB-class model; always `taskkill /F /IM python.exe /T` and verify VRAM free
before/after each run.
