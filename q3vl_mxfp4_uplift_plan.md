# Q3VL MXFP4 — Uplift Plan to +30% (TP1, mc=32/isl1k/osl150)

Grounded in the rocprofv3 profile and the config/env/kernel-tune experiments already run
(see `q3vl_mxfp4_results.md`). Goal: **≥30% throughput uplift** over the current TP1 baseline
(**6.43 req/s** stock → **6.61 req/s** with fusion-shared-experts) → **target ≈ 8.4–8.6 req/s/GPU**.

Workload fixed: multimodal `random-mm`, 1×512×512 image, **mc=32, ISL=1024, OSL=150**, ignore-eos.
Stack: `/opt/vllm @ a65093c` (patch 0005), `/opt/aiter` (fuse_quant 0006), 8× gfx950, quark MXFP4 W4A4.

---

## 0. Where the time goes (decode-dominated profile, the budget we're cutting from)

| bucket | % decode GPU time | kernels | tractability |
|---|---|---|---|
| **MoE GEMM** stage1+stage2 | **~32%** | `mfma/flydsl_moe1_silu_mul_afp4_wfp4`, `moe2_..._cshuffle` | hard (in-graph tile already near-best) |
| **MoE small ops** | **~15%** | `_dynamic_mxfp4_quant`, `add_rmsnorm_quant`, `opus_moe_sorting`, `topkGatingSoftmax`, `fused_mx_quant_moe_sort` | **best ROI — fusion/launch-bound** |
| **Attention** | **~16%** | FMHA prefill, `paged_attention_ll4mi_QKV/reduce` | medium |
| **aux GEMM / elementwise / triton glue** | ~20% | `_gemm_afp4wfp4 M32N32K1024`, `wvSplitK`, `elementwise`, `triton_poi_fused_*` | medium |
| reshape/cache, rope, misc | ~17% | `reshape_and_cache_flash`, rope, etc. | low |

Two hard facts from experiments that constrain the plan:
- The MoE GEMM is **launched ~39k×/run** and each small op a similar count → the decode loop is
  **launch/overhead-bound**, not pure FLOP-bound. Cutting *kernel count* pays as much as cutting FLOPs.
- The aiter standalone MoE-GEMM autotuner **regresses in-graph** (-4%): any kernel work must be
  validated *end-to-end under CUDA-graph capture*, never on a standalone microbenchmark.

---

## 1. Banked + immediate (already proven; do first, ~0 risk)

| # | action | expected | status |
|---|---|---|---|
| 1.1 | `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1` | **+2.8%** (6.43→6.61) | ✅ confirmed, ship it |
| 1.2 | `--max-num-seqs 768 --max-num-batched-tokens 16384` | neutral tput, −mem, better P99 | ✅ confirmed |

Running total after §1: **~6.61 req/s (+2.8%)**. Remaining gap to +30%: need ~+27% more.

---

## 2. High-ROI kernel work — the MoE small-op overhead (~15%) ← primary GEAK target

This is the single best lever: ~15% of decode time is spread across **5 separate tiny kernels**,
each launched ~39k×, that surround every MoE call. They are fusion/launch-bound, exactly GEAK's wheelhouse.

- **2.1 Fuse `_dynamic_mxfp4_quant` into the MoE stage1 epilogue/prologue.**
  Today activation→MXFP4 quant is a standalone 4.1% kernel feeding the GEMM. Author a GEAK kernel
  that does the per_1x32 activation quant *inside* the stage1 producer (or as a fused prologue), removing
  a full launch + a global round-trip of the activation tensor per layer (×94 layers × decode steps).
  *Profitability:* eliminating the standalone quant kernel ≈ up to 4% wall-clock; fusion also cuts an HBM
  round-trip so realistic gain is **3–6%**.

- **2.2 Fuse `add_rmsnorm_quant` (3.9%).** Already a partly-fused aiter kernel; check whether the
  +rms_norm/+quant_fp8 custom-ops in `compilation_config` are actually firing for the MoE path or only
  attention. If not fused there, extend the inductor fusion pass (cheap) or GEAK a combined
  add+rmsnorm+mxfp4-quant. **2–4%**.

- **2.3 Collapse MoE sorting+routing (`opus_moe_sorting` 3.7% + `topkGatingSoftmax` 2.9% +
  `fused_mx_quant_moe_sort` 2.0% = ~8.6%).** Three kernels do topk → sort → quant-sort sequentially.
  GEAK a single fused "route+sort+quant-pack" kernel for E=128/topk=8. This is the highest-count
  launch cluster; fusing 3→1 saves 2 launches/layer/step. **3–6%**.

§2 realistic envelope: **+8–15%** end-to-end, the bulk of the path to 30%. Order by profile %: 2.3 → 2.1 → 2.2.

---

## 3. Graph-aware MoE GEMM autotune (~32%) — recover what the naive tuner lost

The standalone tuner picked `t32x64x256` and regressed. The kernels exist and are 8-13% faster in
isolation — the problem is the *objective*, not the kernels.

- **3.1 Build a graph-context tuner harness.** Drive the candidate FlyDSL/CK MoE kernels through the
  *captured decode graph* at the real `block_m=32` + cudagraph batch sizes (1..512 capture list), measure
  end-to-end TPOT, pick per-bucket winners. Reuse `/tmp/q3vl-run/untuned_decode.csv` shapes but score with
  the served loop, not `gemm_moe_tune.py`'s microbench.
- **3.2 Sweep `VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE`** (currently 163840) and MoE padding
  (`VLLM_ROCM_MOE_PADDING`) — these change the effective per-expert tile and may unlock a better
  in-graph kernel without custom code. Cheap, do before 3.1.
- **3.3 If a kernel is genuinely suboptimal at the real shape, GEAK a stage1 `silu_mul_afp4_wfp4`
  variant** tuned for (M≈256 sorted, N=1536, K=4096, per_1x32). Validate end-to-end only.

§3 realistic envelope: **+3–8%** (the in-graph tile is already decent; upside is bounded but real if 3.1 finds a better per-bucket map).

---

## 4. Attention (~16%) — secondary

- **4.1** Confirm decode uses `ROCM_AITER_UNIFIED_ATTN` end-to-end (it does; beats AITER_FA by ~6% — already tested).
- **4.2** Try `VLLM_ROCM_USE_AITER_MHA=1` interaction and `paged_attention` block-size sweep
  (`--block-size 16/32`) — `paged_attention_ll4mi_QKV` + `reduce` together are ~6.4%; a better block size
  or the unified-attn decode path may shave the reduce kernel. **1–3%**.
- **4.3** Vision encoder: only matters at prefill (1 img/req). Confirm `mm_encoder_tp_mode=data` +
  `compile_mm_encoder` are optimal; low priority at osl150 (decode-dominated). 

---

## 5. Method / guardrails (learned the hard way)

- **Always rebench end-to-end, 640 prompts, warmed.** The 320-prompt "baseline" cold-start artifact and
  the microbench-vs-graph divergence both produced false signals. One isolated server per config, sequential
  or contention-free.
- **Coherence gate every iteration:** `AITER_MXFP4_MXFP4` backend + `Q3VL fuse_quant` log line + a text
  and an image prompt must return non-repeating correct output. MXFP4 fails silently to garbage.
- **8-GPU parallelism:** run 7 candidate configs vs 1 baseline per round (one GPU each), as in the sweeps.
- **Attribute wins to kernel-count and HBM traffic, not just TFLOPs** — this workload is launch-bound.

---

## 6. Sequenced execution (next session, ~1 day)

1. Ship §1 (done) → **6.61 req/s**.
2. §3.2 env sweeps (cheap, hours) — bank any free MoE-GEMM/padding gain.
3. §2.3 fused route+sort+quant (GEAK) — biggest single cluster (~8.6%).
4. §2.1 fused activation-quant into stage1 (GEAK).
5. §2.2 add_rmsnorm_quant fusion check/extend.
6. §3.1 graph-aware MoE GEMM tuner; §3.3 only if it points at a specific bad kernel.
7. §4.2 attention block-size/decode-path sweep to top up.

**Projected:** §1 (+2.8%) + §2 (+8–15%) + §3 (+3–8%) + §4 (+1–3%) ≈ **+15–29%** realistic,
**+30% reachable** if §2.3 + §2.1 land near the top of their ranges. The 30% is a kernel-engineering
result, not a config result — the profile says it lives in MoE small-op fusion + a graph-aware GEMM map.

## 7. Artifacts to reuse
- Profile: `/tmp/q3vl-run/rocprof_dec/` (decode), `/tmp/q3vl-run/rocprof_out/` (mixed)
- Tuner harness + shapes: `/tmp/q3vl-run/untuned_decode.csv`, `gemm_moe_tune.py` (`--mp`, but rescore in-graph)
- Serve/bench scripts: `/tmp/q3vl-run/serve_env.sh`, `bench.sh`; merged-config example (regressing) `tuned_fmoe_merged.csv`
- GEAK backend: `/workspace/Hyperloom/kernel-agent/` (SKILL.md, tools/)
- Checkpoint: `/tmp/q3vl-mxfp4` (128GB, verified)
