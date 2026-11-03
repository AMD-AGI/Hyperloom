# Regime tables: what InferenceX changes with concurrency, per model, GPU and engine

Built 2026-10-20 from the runner's docs/research/inferencex-study (records.jsonl ids) and InferenceX origin/main.

**How to read this file**

- **Source.** InferenceX `origin/main` at `6d6d296c` (2026-08-30). Configs live in `configs/amd-master.yaml`,
  `configs/nvidia-master.yaml` and `configs/deprecated/` (they moved out of `.github/configs/` on 2026-07-02,
  commit `e9fc9880`). Launch logic is in `benchmarks/single_node/{fixed_seq_len,agentic}/*.sh` and
  `benchmarks/multi_node/amd_utils/`. `#NNN` is a `records.jsonl` id; it says *why* a band switches.
- **Scenarios.** Active fixed-seq-len is **8k1k only** (ISL 8192 / OSL 1024). 1k1k was deprecated on 2026-07-17,
  so its rows survive only in `configs/deprecated/amd-1k1k-master.yaml`. **Agentic** (AgentX trace replay) counts
  CONC as live session trees, not requests, which is why agentic recipes set `max-running = 2×CONC`.
- **Rows are concurrency bands.** `conc a–b` means a, 2a, 4a … b. Overlapping rows on different layouts are deliberate.
  InferenceX runs both and the Pareto frontier (tok/s/GPU against interactivity) picks per point. The overlap marks
  where the crossover was measured or rechecked.
- **Abbreviations.**
  - TP: tensor parallel.
  - EP: expert parallel.
  - DPA: data-parallel attention (`--enable-dp-attention`).
  - DEPn: DPA plus EP across n GPUs.
  - TEPn: TP attention with an EP MoE.
  - DCP: decode context parallel.
  - TBO: two-batch overlap.
  - MTP n: n draft tokens. SGLang EAGLE is written steps/topk/draft-tokens.
  - AL: acceptance length.
- **Markers.**
  - **⚠num**: changes numerics, so re-run GSM8K.
  - **[img]**: true only for that image.
  - **↔**: the recipes disagree.
  - **[dep]**: the config is deprecated and moved to `configs/deprecated/`. It is the last tuned state, not live.
- **Arbor caveat.** InferenceX scores **tok/s per GPU**. A TP4 row at the same CONC as a TP8 row is a different
  deployment, with two replicas per node. A fixed-TP Arbor pack measures absolute tok/s on one server and cannot see that
  win (REPORT §2.1). Every fact in this file depends on the operating point by construction. Before acting on a row,
  check that your CONC, ISL/OSL and TP/EP/DPA match it.

**Checking that a band setting took effect.** Silent no-ops are common: a renamed env var (#286, #367), a variable
set but never exported (#310), labels that do not match the launch (#322, #462), and an offload tier that declines to
build with one log line (#1214).
- Read the server's own argument dump at startup: SGLang's ServerArgs, vLLM's non-default args.
- Read the live process environment (`/proc/<pid>/environ`). Do not trust the script.
- The DSV4 agentic SGLang recipe prints every `SGLANG_*` variable into server.log for exactly this reason.
- For MTP, read the acceptance length in the server metrics. Under synthetic acceptance (agentic), the AL is an
  input you set, not a measurement (#1167, #1082).

---

## 1. AMD MI355X (gfx950) single-node, fixed-seq-len 8k1k

### 1.1 DeepSeek-V4-Pro FP4, SGLang

`dsv4-fp4-mi355x-sglang`: `lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260706`. The MTP twin,
`dsv4-fp4-mi355x-sglang-mtp`, runs on `…-20260708`. Scripts: `dsv4_fp4_mi355x_sglang{,_mtp}.sh`.

These settings hold in every band:
- `--attention-backend dsv4`, page 256, KV `fp8_e4m3` (⚠num).
- `--mem-fraction-static 0.90`, `--swa-full-tokens-ratio 0.15`, `--disable-shared-experts-fusion`, radix off.
- `--max-running-requests CONC` and `--cuda-graph-max-bs CONC` (#995).
- Env: `SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton`, `AITER_BF16_FP8_MOE_BOUND=0`, `SGLANG_USE_ROCM700A=0`.

| CONC | Layout | MTP twin | Band-switched flags / env | Chunked prefill |
| --- | --- | --- | --- | --- |
| 1–32 | TP4, EP1, no DPA | TP8 no DPA, c1–32; EAGLE 3/1/4 | none | `ISL` (#901) |
| 16–128 | TP4 + DPA (DP4), EP1 | no TP4 rows | `--dp TP --enable-dp-attention --enable-prefill-delayer --enable-two-batch-overlap`; `SGLANG_SHARED_EXPERT_TP1=1 SGLANG_DP_SHARED_EXPERT_LOCAL=1 SGLANG_DP_USE_GATHERV=1 SGLANG_DP_USE_REDUCE_SCATTER=1 GPU_MAX_HW_QUEUES=5` (#995) | `ISL×TP` (#859) |
| 64–2048 | TP8 + DPA (DP8), EP1 | TP8 DPA, c64–2048; EAGLE 3/1/4; the DP env set but **no TBO and no** `GPU_MAX_HW_QUEUES` | same as the row above | `ISL×TP` = 65536 |

Why the bands sit where they do:
- **Split points.** #755 moved the split: DPA from c64, TP-only up to c32. #864 moved the low and middle bands from
  TP8 to TP4.
- **SWA ratio.** At c1024–2048 the full-attention pool filled. Ratio 0.15 plus mem 0.90 fixed it (#771).
- **Chunk size under DPA.** SGLang divides `chunked_prefill_size` by the DP size, so under DPA the chunk must be
  ISL×TP (#859, #901).
- **TBO, local shared expert, gatherv and HW queues pay only together.** They pay on the DPA topology. On ROCm, TBO
  without `GPU_MAX_HW_QUEUES` serializes its streams (#995).
- **EP stays off on the DPA rows.** EP8 under DPA gave GSM8K 0.0000 against 0.9318 with EP silently 1
  (#595, ⚠num, [img] rocm720-mi35x-a8410de-20260502-DSv4).
- **MTP depth.** The MTP twin first used EAGLE 2/1/3 on the DPA band and 3/1/4 on the TP band (#798). Main now uses
  3/1/4 everywhere.
- **MTP prerequisites.** MTP needs sglang#26383, which fixes a false EOS under graph capture; GSM8K reads 0.950 with
  it. Benchmark through the `--dsv4` chat encoding: raw random tokens lower acceptance (#798).

### 1.2 DeepSeek-V4-Pro FP4, vLLM and ATOM

**vLLM**, `dsv4-fp4-mi355x-vllm{,-mtp}` on `vllm/vllm-openai-rocm:nightly-09663abd…`, has a single band: **TP8, c4–512**.
- Flags: `VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_USE_AITER_MOE=1` with `--moe-backend aiter` (replaces triton_unfused, #955),
  fp8 KV, `--gpu-memory-utilization 0.8`, `{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}` (#756), async scheduling.
- **MTP twin.** `num_speculative_tokens 2` at every CONC. Upstream data: +75% at c1, +38% at c8, −37% at c32 (#796).
  MTP is on the frontier only below about c32. A C32 campaign rejects it and a C8 campaign keeps it.
- **DP+EP.** The script has a DP+EP path (TP1×DP8 plus `--enable-expert-parallel`), but no config row uses it.

**ATOM**, `dsv4-fp4-mi355x-atom{,-mtp}` on `rocm/atom-dev:nightly_202607231538`, script `dsv4_fp4_mi355x_atom{,_mtp}.sh`.

These settings hold in every band:
- fp8 KV, `--gpu-memory-utilization 0.9`, prefix caching off.
- Capture sizes `[1,2,4,8,16,32,48,64,128,256,512]`.
- `ATOM_MOE_GU_ITLV=1`, `AITER_BF16_FP8_MOE_BOUND=0`, `ATOM_DISABLE_MMAP=true`.

| CONC | Layout (STP) | MTP twin (`--method mtp --num-speculative-tokens 3`) | Band-switched | Batch / memory |
| --- | --- | --- | --- | --- |
| 1–64 | TP4 no DPA; TP8 no DPA (kept "to recheck the crossover") | TP8 no DPA, c4–256 | none | ATOM default `max_num_seqs` below c64 (about 4% better at c2, #1068); `--max-num-seqs CONC` from c64 |
| 16–128 | TP4 + DPA, EP1 | TP4 removed: TP4+MTP+DPA leaves a **negative** KV budget (−10.98 GB at 0.90, #1067) | `--enable-tbo` + `GPU_MAX_HW_QUEUES=5` at CONC≥64 | `--max-num-seqs CONC`, mandatory under DPA: each rank keeps a full KV pool, so the default OOMs (#1068, #550) |
| 64–2048 | TP8 + DPA, EP1 | TP8 DPA, c32–1024; **prefill-only** TBO + `GPU_MAX_HW_QUEUES=5` at CONC≥256 | TBO at ≥64 | `--max-num-seqs CONC` |

- **The TBO threshold depends on the image and on spec decoding.**
  - nightly_202606161823 gated it at 1k1k c≥1024 and 8k1k c≥256 (#877).
  - nightly_202607231538 gates it at c≥64 without MTP (#1068).
  - With MTP the gate is c≥256. Same image, TBO on against off: −14% at c64, −10% at c128, +11%, +8% and +14% at c256
    and above (#1067).
  - Decode TBO (`--enable-tbo all`) silently drops MTP's spec metadata. Only prefill TBO is MTP-safe (#1067).
- **Forbidden override.** `--hf-overrides {"use_index_cache":true,"index_topk_freq":4}` was removed. It skips the
  indexer, a FLOP reduction the review checklist forbids (⚠num, #959).

### 1.3 DeepSeek-R1-0528, SGLang and ATOM

| Config (image) | Bands | Band-switched settings |
| --- | --- | --- |
| `dsr1-fp4-mi355x-sglang` (`lmsysorg/sglang:v0.5.12-rocm700-mi35x`, MXFP4-Preview) | TP4 c4–64; TP8 c4–64 (TP4 added at 8k1k, #825) | `--chunked-prefill-size` and `--max-prefill-tokens` are 196608, **32768 at 8k1k when CONC>32** to fit c64 in memory (#205). Fixed `--cuda-graph-max-bs 128`, `--num-continuous-decode-steps 4`, aiter attention, fp8 KV, `ROCM_QUICK_REDUCE_QUANTIZATION=INT4` (⚠num) |
| `…-sglang-mtp` (same image, MXFP4) | TP8 EP1 c4–64 | Same prefill switch at CONC>32. EAGLE 3/1/4, `SGLANG_ENABLE_SPEC_V2=1`, `SGLANG_AITER_MLA_PERSIST=1` (#744) |
| `dsr1-fp8-mi355x-sglang` (same image tag) | TP8 c4–64; **TP4 c32–64** (#385) | `--cuda-graph-max-bs CONC` (#301), `--num-continuous-decode-steps 8` (+4.7%, found by an agent campaign, #596), chunked prefill 196608, `RCCL_MSCCL_ENABLE=0`, QR INT4 (⚠num) |
| `…-sglang-mtp` | TP8 EP1 c4–64 | `--max-running-requests` and `--cuda-graph-max-bs` = CONC; EAGLE 3/1/4 (#746) |
| `dsr1-fp4-mi355x-atom` (`rocm/atom:…atom20260511`) | **TP4 c4–256; TP8 c4 only** | TP4 at higher CONC and TP8 at the lowest (#266, #361). Block 16, `--max-model-len 10240` at 8k |
| `…-atom-mtp` | TP8 c4–256 | `--method mtp` without `--num-speculative-tokens`, so the default of 1 draft (#445). `AMDGCN_USE_BUFFER_OPS=1` |
| `dsr1-fp8-mi355x-atom` / `-atom-mtp` (`…atom20260511` / `rocm7.2.4 …atom0.1.3`) | TP8 c4–128 / TP8 c4–256 | MTP 3 tokens (#445). The atom0.1.3 bump read +47–131% against the previous numbers, mostly from the image (#809) |

### 1.4 Qwen3.5-397B-A17B, SGLang and ATOM

| Config (image) | Bands | Notes |
| --- | --- | --- |
| `qwen3.5-fp8-mi355x-sglang{,-mtp}` (`lmsysorg/sglang-rocm:v0.5.16-rocm720-mi35x-20260726`) | **TP4 EP1 c4–256**, a single layout | The old band map (TP8 at low CONC; TP8/EP8 and TP2/EP2 at high CONC; TP4 at c32–256, #469) collapsed to TP4 once AITER unified attention arrived (`SGLANG_USE_AITER_UNIFIED_ATTN=1`, page 16, #819). Fixed flags: `--max-running-requests` and `--cuda-graph-max-bs` = CONC, chunked 32768, `--scheduler-recv-interval 30`, `--enable-aiter-allreduce-fusion`, 6 tokenizer workers. MTP: EAGLE 3/1/4 |
| `qwen3.5-fp4-mi355x-sglang` (`…v0.5.16-rocm720-mi35x-20260730`) | **TP2 c4–256**; TP4 c4–16 | TP4 only at the latency end. `AITER_FLYDSL_FORCE=1` (#863), `SGLANG_MAMBA_SSM_DTYPE=bfloat16` (#1025, ⚠num: SSM state precision), `--max-running-requests CONC` |
| `…-sglang-mtp` | TP2 c4–128; TP4 c4–16 | TP2 at c256 OOMs with the draft (#827). EAGLE 3/1/4 |
| `qwen3.5-fp8-mi355x-atom` (`rocm/atom:…atom20260511`) / `-atom-mtp` (`…atom0.1.2.post`) | TP2, TP4 and TP8, all c4–256 / TP4 and TP8 c4–256, MTP 3 | All widths are measured and the frontier picks. The image bump read +3.2–16.3% at TP4 (#745) |
| `qwen3.5-fp4-mi355x-atom` (`…atom0.1.2.post`) | TP2 c4–256; TP4 c4–16 | Same shape as SGLang FP4 |

### 1.5 Deprecated MI355X fixed-seq-len configs (last tuned state)

| Model / engine [dep date] | Bands | Band-switched settings (evidence) |
| --- | --- | --- |
| GLM-5 FP8, SGLang [07-18] (`sglang-rocm v0.5.12-…-20260517`) | TP4 c4–256. MTP twin: TP4 c4–128 plus TP8 c4–8 | TP8 to TP4, "turn to tp=4 for best perf". `--max-running-requests` and `--cuda-graph-max-bs` = CONC (#653, #582). NSA tilelang prefill, `SGLANG_ROCM_FUSED_DECODE_MLA=0` |
| GLM-5 FP8, ATOM | TP4 and TP8 c4–256 | none |
| GLM-5.1 MXFP4, SGLang [07-18] | **TP2 c4–256**; TP4 c4–16 | TP2 collapsed GSM8K to about 0.32 on `v0.5.12-…-20260517`, fixed in `.post1` (#779, ⚠num [img]). `--enable-aiter-allreduce-fusion` (#944) |
| MiniMax-M3 MXFP8, vLLM [8k1k 08-03] (`nightly-9e57de71`) | TP4 c1–512, a single layout (TP8 and TP4/EP4 dropped, #972) | **Fast path when ISL≥8192 and CONC≥64**: `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` (AITER page-16 sparse paged attention), `VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=0`, `…_QUANTIZATION_MIN_SIZE_KB=256`, `--linear-backend emulation`. Otherwise native Triton MXFP8 linears and an unshuffled `TRITON_ATTN`. Gain +2.5–4.7% at c128–512, neutral elsewhere (#1050). QR INT4 at every CONC (⚠num), `--max-num-batched-tokens 32768` |
| ↔ the emulation linear backend | | On `nightly-09663abd`, emulation won most at **c1** (+26%, #972). On `nightly-9e57de71`, native wins at low CONC and emulation only at high CONC (#1050). The answer depends on the image. Re-measure |
| MiniMax-M3 MXFP8, vLLM MTP | TP4 c1–512, EAGLE3 with 3 tokens | QR INT6. ↔ An AITER master-switch gate on EP: #935 turned AITER on only *with* EP (TP-only output degenerate), while the later MTP script turns it *off* with EP. Both say vllm#47158 fixed it (#972). ⚠num: check GSM8K |
| MiniMax-M3 MXFP4, vLLM | TP2 c1–512; TP4 c1–64 | INT4 quick-reduce needs `MIN_SIZE_KB=256`, or the 16 MB default gate never fires for about 1.5 MB decode all-reduces: −12–17% TPOT at c64–256 (#1089). MTP twin: TP8 c1–64, TEP8 c1–256, TP4 c1–64, **DEP8 c128–256** |
| MiniMax-M3 FP8/FP4, ATOM | TP4 c1–256, with and without MTP | Block 128, required by MSA |
| MiniMax-M2.5 FP8, vLLM [06-20] (`v0.22.0`) | TP2/EP2 c2–256; TP4/EP4 c4–512; TP8/EP8 c2 | Per-point table (#611). TP8/EP8: block 32, no shuffle, `--no-async-scheduling`, and at 8k `VLLM_ROCM_USE_AITER_MOE=0`. Other TP widths at 8k: async off below c64; block 16 plus shuffled KV from c64; async back on above c64 |
| MiniMax-M2.5 MXFP4, ATOM / vLLM | ATOM: TP1 and TP2 c4–1024, TP4 c4–128, TP8 c4–16. vLLM: TP1 c4–32, TP2 and TP4 c4–64 | TP1 carries the high-CONC end. `VLLM_USE_RUST_FRONTEND=1`: TTFT −8% at 1k1k and −22% at 8k1k, tok/s neutral (#823) |
| Kimi-K2.5 MXFP4 / INT4, vLLM [08-06] | TP8 and TP4 c4–128 | **`VLLM_ROCM_USE_AITER_RMSNORM=0` when TP<8, for accuracy** (#425, ⚠num). INT4: `--moe-backend flydsl`, `fuse_allreduce_rms` off (#867) |
| Kimi-K2.5 MXFP4, ATOM | TP4 c4–128 | `--scheduler-delay-factor 1`, INT4 quick-reduce (#1003) |
| gpt-oss-120b, vLLM [07-06] (`v0.22.0`) | **TP1 c4–128**; TP4 c4; TP8 c4–8 | ATOM: TP1 c4–256, TP8 c4–16. TP1 above about c16, TP8 only at low CONC (#266) |
| Llama-3.3-70B, vLLM [left the sweep 2025-10, #212; no scripts on main] | MI355X FP4: at 1k-input with **CONC≤16** `VLLM_ROCM_USE_AITER_TRITON_BF16_GEMM=0`, else `VLLM_TRITON_FP4_GEMM_USE_ASM=1`; at 8k1k with CONC>16, `VLLM_ROCM_USE_AITER_MHA=1` (#194, commit `d25f4f64`) | `--max-num-batched-tokens CONC×128` (#81) was reverted to 131072 because it starved prefill at low CONC (#125) |

---

## 2. AMD MI355X single-node, agentic (AgentX)

These rules hold for every agentic recipe:
- **Acceptance is synthetic.** Throughput runs pin the AL with `SGLANG_SIMULATE_ACC_LEN`, vLLM's
  `rejection_sample_method: synthetic`, or ATOM's `--spec-decode-acceptance-length`. Real acceptance on 100K-token
  traces can be far lower (#1082). Draft depth is therefore a benchmark input: deeper only adds verify cost unless the
  AL input moves with it (#1167).
- **Evals verify for real.** `EVAL_ONLY=true` runs real verification.
- **Batch sizing.** `max-running` or `max-num-seqs` is 2×CONC, for subagent fan-out.
- **DRAM tier.** A host-DRAM KV tier is switched on only past each width's HBM knee.

| Model / engine (image) | CONC band → layout, offload | Spec | Band-switched settings |
| --- | --- | --- | --- |
| DSV4-Pro FP4, vLLM (`nightly-311b3513`) | c1–64: TP8, GPU-resident. c64: TP1×DP8 attention, EP off, behind `vllm-router 0.1.14` consistent-hash (session id taken from the AIPerf correlation id) | MTP 3, AL 2.49 | DPA: `--max-num-seqs CONC` (2×CONC otherwise), `--prefill-schedule-interval 8`, `--long-prefill-token-threshold 16384`. `--max-num-batched-tokens 8192` raised KV capacity from 4.73M to 8.52M tokens (#1201). The LMCache arms were removed until vllm#45497 lands (MTP creates two KV groups) |
| DSV4-Pro FP4, SGLang (`v0.5.18-rocm720-mi35x-20260822`) | **c1–10: TP4, resident. c16–48: TP8 + HiCache** (ratio 1.5, write_through) | EAGLE 3/1/4, AL 2.49 (NEXTN crashes on V4) | Chunked prefill 8192, so a 100K prefill cannot monopolize the engine (8192×TP under DPA). Mem 0.89. Max-running 2×CONC, capped at 256. Graph bs = min(max-running, 128). `AGENTIC_WARMUP_GRACE_PERIOD=3600` at CONC≥32. `--enforce-shared-experts-fusion`, `SGLANG_OPT_USE_AITER_BATCHED_GEMM=1`. Resident arm moved from TP8 c1–16 to TP4 c1–10 (#1232). The DPA arm stays dormant (#1181) |
| DSV4-Pro FP4, ATOM (`nightly_202608201032`) | c1–48: TP8, resident | MTP 3, AL 2.49 | Max-num-seqs 2×CONC. `--index-cache-dtype fp4` (⚠num; #1223 calls it FP8, but the script says fp4). 16K batch tokens and prefill chunk, `--cudagraph-mode FULL` |
| Kimi-K3 MXFP4, vLLM (`nightly-cb810483`) | c1–8: TP8, resident. c10: TP8 + `vllm-simple` CPU offload | DSpark K=2, AL 2.51 | TP8 only, because the 1.56 TB checkpoint is about 195 GB per GPU. `max-num-seqs 20` and capture sizes 1–60 at every CONC. Container patches backport nine AITER/vLLM PRs (#1171) |
| Kimi-K3 MXFP4, ATOM (`…kimi_k3_agentic_0821`) | **c1–4: TP8, no DCP, resident. c8–12: DCP8 + LMCache paged KV. c16: DCP8 + LMCache + CPU state tier (32 GB/rank). c32–56: the same tiers, no draft** | **K=7 (AL 3.84) at c1–4; K=3 (AL 3.00) at c8–16; none at c≥32** | Per CONC (#1254): max-num-seqs 32 / 32 / 24 / 32 / 64 / 80 / 72; batch tokens 8192, or 4096 at c8–12 and c56; ReplaySSM with 96 checkpoint slots only at c8–12; `LMCACHE_CHUNK_SIZE=1024` (page 128 × DCP 8). The paged-KV tier was halved because pinning skew between ranks tripped the 600 s NCCL timeout. The config omits c16, but the script handles it |
| GLM-5.2 MXFP4, SGLang (`v0.5.16-rocm720-mi35x-20260728`) | **TP4/EP4 + HiCache c1–16; TP8/EP8 resident c1–4** | EAGLE **5/1/6**, AL 3.61 (#1167) | Chunked 32768 at every CONC, with mem 0.85. The first recipe used chunk 131072 at c≤16, which needed mem 0.80: 0.85 OOMed on a 6.86 GiB activation (#1158). The 32K chunk lets decode interleave and frees the headroom (#1167). `max-running = CONC` (not 2×), capped at 256. HiCache ratio 1.5, or 0.5 under DPA (dormant: DSA+DP hangs a collective on ROCm). `SGLANG_OPT_USE_TOPK_V2=false` [img gfx950]. The AITER all-reduce + RMSNorm, fused QK-norm-RoPE and FlyDSL MoE sorting together gave +23.6% P90 interactivity at c8 (#1255, fork merge `6d6d296c`) |
| GLM-5.2 MXFP4, ATOM (`…glm52_agentic_0813`) | TP4 + LMCache c2–10; TP8 resident c1–4 | MTP 3, acceptance (2.99−1)/3 | **Capture sizes chosen per CONC**: c1 `[1,2]` … c16 `[1..32]`. Max-num-seqs 2×CONC. `ptpc_fp8` online quantization of the non-expert layers (⚠num, #1208) |
| MiniMax-M3 MXFP4, vLLM (`v0.27.1`) | **TP4 resident c1–32; TP2 resident c1–5; TP4 + LMCache c32–40** | EAGLE3-GQA 3, AL 2.78 | `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` **only at TP4**: the AITER page-16 sparse paged attention needs exactly 1 KV head per rank, and M3 has 4 (#1165). QR INT4 with a 256 KB minimum. ↔ ⚠num: the recipe carries the `index_topk_freq 4` indexer skip (commit `73bd8e20`), which the fixed-seq-len recipes ban (#959, #1050, #1089) |
| Qwen3.5 MXFP4, SGLang (`v0.5.17-rocm720-mi35x-20260818`) | **TP4 resident c1–40; TP2 resident c1–20; TP4 + HiCache c40–64; TP2 + HiCache c20–32** (#1249) | EAGLE 3/1/4, AL 3.39 | Max-running 2×CONC. Graph bs min(CONC, 64). Recv interval 30, stream interval 50. 6 tokenizer workers at TP≥4. fp8 KV (#1215). All-reduce fusion off: unstable for TP2/EP2 EAGLE (#1168) |

---

## 3. AMD MI300X / MI325X (gfx942)

### 3.1 Fixed-seq-len 8k1k, active

| Config (image) | Bands | Settings that matter |
| --- | --- | --- |
| `dsv4-fp8-mi300x-vllm` and `mi325x` (`vllm-openai-rocm:nightly-09663abd`) | TP8 c4–**256** (MI325X runs on `cluster:mi325x-tw`) | TP8 only, because the dequantized FP8 (about 1.05 TB) fits 8×192 GB only at TP8. STP capped at c256: 9472-token requests fit about 20× concurrency in KV, and c512 timed out. `--quantization deepseek_v4_fp8`, MoE backend left on auto, which picks `AITER_MXFP4_BF16` (W4A16): forcing `aiter` selects W4A4, which gfx942 rejects (#1011, #1020). MI300X sets `EVAL_CONCURRENT_REQUESTS=8` |
| `…-vllm-mtp` (both SKUs) | TP8 c4–**128** | MTP 2. c256 had 19.2% request failures on MI300X, because the draft eats KV. Use `--dsv4` chat encoding (#1012, #1021) |
| `dsr1-fp8-mi300x-sglang` / `mi325x` (`lmsysorg/sglang:v0.5.12-rocm700-mi30x`) | TP8 c4–64 | Fixed `--cuda-graph-max-bs 128`, chunked 131072, `--num-continuous-decode-steps 4`, `SGLANG_AITER_MLA_PERSIST=1`. MI300X sets `HSA_NO_SCRATCH_RECLAIM=1` when MEC firmware is below 177. The MI325X MTP twin: TP8 EP1 c4–64, EAGLE 3/1/4 |
| `qwen3.5-fp8-mi300x-sglang` / `mi325x{,-mtp}` (`v0.5.12-rocm720-mi30x`) | TP8 c4–64 | `--cuda-graph-max-bs CONC`, max-prefill 32768, recv interval 30, AITER all-reduce fusion, mem 0.75 (#494, #718) |

### 3.2 Deprecated gfx942 fixed-seq-len configs

| Model [dep] | Bands | Notes |
| --- | --- | --- |
| MiniMax-M3 MXFP8, vLLM [8k1k 08-03] | MI300X: TP8 c1–256; MTP twin TP8 c1–64 and TEP8 c128–256. MI325X: TP4 c1–32, TP8 c1–128, TEP8 c256, **DEP8 c256–512** (H200-style map, #854) | **BF16 KV on gfx942.** FP8 KV fell back to scale 1.0 and corrupted accuracy (#886, ⚠num) |
| MiniMax-M2.5 FP8, MI325X [06-20] | TP2 c4–64; TP8/EP8 c4–256 | Shuffled KV only where a gfx942 ASM paged-attention kernel exists: TP2/EP1 at c≤16 and TP8/EP8 at c≤64. Elsewhere `pa_fwd_asm` crashes. Pin `ROCM_AITER_FA`: an upstream default flip had cost about 38% (#794) |
| Kimi-K2.5 INT4 / GLM-5 FP8 / gpt-oss | Kimi TP8 c4–64. GLM-5 TP8 c4–64 (+MTP). gpt-oss MI300X: TP1–4 c4–64, TP8 c1–16 | none |

### 3.3 Agentic on gfx942

| Config (image) | Bands | Notes |
| --- | --- | --- |
| Qwen3.5 FP8, MI325X, SGLang (`v0.5.16-rocm720-mi30x`) | TP2/EP2 c1–6; TP4 c1–48; TEP4 c4–48; TP8 c1–80; TEP8 c4–64 | Picked from a 62-point sweep. Knees: TP2 at c4, TP4/TEP4 at c40, TP8/TEP8 at c64 (#1163). Mem 0.80, or 0.95 at TP2 because EAGLE reserves 15% for the draft. Graph bs min(CONC, 64) |
| Qwen3.5 FP8, MI300X (`…-mi30x-20260730`) | TP8 and TEP8 c4–32 | Peak at c24, regression at c32 (#1169) |
| MiniMax-M3 FP8, vLLM (`v0.27.1`) | MI300X: TP8 c2–10 resident, LMCache at c16 (#1175). MI325X: TP8 c1–18 resident, knee at c16–18 (#1172) | MI300X: `GPU_MAX_HW_QUEUES=2`, `NCCL_MIN_NCHANNELS=112`, `VLLM_ROCM_USE_AITER_MHA=0` |
| GLM-5.2 FP8, MI325X, SGLang (`v0.5.16-rocm720-mi30x`) | TP8 c1–8 resident | BF16 KV, chunked 131072, torch DSA top-k (the v2 JIT is CUDA-only on gfx942), EAGLE 3/1/4 with AL 2.99 (#1141) |

---

## 4. AMD multi-node disaggregated layouts (MoRI / MoRI-IO / Mooncake)

All of these are 8k1k unless marked. P = prefill workers, D = decode workers.

| Config (image) | Low CONC | Mid CONC | High CONC | MTP depth by band |
| --- | --- | --- | --- | --- |
| DSR1 FP8, SGLang (`rocm/sgl-dev:sglang-0.5.9-rocm720-mi35x-mori-0227-2`) | c1–64: 1P TP4 + 1D TP8 | c4–256: 1P TP8 + **2D TP8** | c1024–2048: **2P DEP8 + 1D DEP8** | 2 at the bottom, 1 at the top |
| DSR1 FP4, SGLang (`sglang-rocm v0.5.12-rocm720-mi35x-20260519`) | c1–8: 1P1D TP8 | c2–256: 1P2D TP8 (a TP4-prefill variant at c64–256); c128–512: 1P1D DEP8 | c1024–4096: 2P DEP8 + 1D DEP8 | From the config: 3 at c≤32, 2 at 1P2D c64–256, 1 on DEP (history: #342, #795). The 8k1k-mtp variant uses 3 everywhere except 1 at 2P1D (#795) |
| DSV4-Pro FP4, SGLang (`v0.5.14-…-20260701`; MTP on `v0.5.15-…-20260713`) | c1–128: 1P1D TP8 | c256: 1P1D DEP8 (MoRI all-to-all) | c512: 1P1D DEP8 | 2 on TP8, 1 on DEP8. ↔ ⚠num: comments say 2-step EAGLE is lossy (GSM8K about 0.85) on 20260701, yet the TP8 arm sets 2. Decode-only spec flags silently dropped GSM8K from 0.96 to 0.88 (#1059) |
| DSV4-Pro FP4, ATOM disagg (`atom-dev nightly_202606101403`, Mooncake, atomesh) | c4–128: 1P1D TP8 | none | c256–2048: 2P DPA-TP8 + 1D DPA-TP8 | none |
| DSV4-Pro agentic, HiCache (`v0.5.17-…-20260817`) | c1–64: 1P1D TP8 | c64–192: 1P1D DEP8 | none | 3 (#1227) |
| Qwen3.5 FP8 / FP4, SGLang | FP8 c8–128: **1P TP4 + 1D TP8**, +24–31% tok/s per GPU against TP8P (#1210). FP4 c8–512: 1P1D TP8 | none | none | None. DPA off: with MoRI all-to-all, SGLang miscounts the shared-expert slots for 512+1 experts (#768) |
| Kimi-K2.5 MXFP4, vLLM / ATOM [dep] | vLLM c1–4: 1P TP4 + 1D TP4. ATOM c1–16: 1P TP4 + 2D TP8 | vLLM c8–512: 1P TP4 + **2D TP4**. ATOM c16–256: 1P1D TP4 | none | None. EP was 14–27% slower than dense TP (#1051) |
| MiniMax-M3 FP8, vLLM [dep] | c1–512: 1P1D TP4 | 8k1k c128–512: **2P TP4 + 1D TP4**. 1k1k: 1P2D | none | None. Retuning P:D per ISL gave 2.9× tok/s per GPU at 8k1k (#1001). The pool held at most 3 workers |
| MiniMax-M3, ATOM [dep] | c1–256: 1P1D TP4 | c256–1024: 2P DPA-TP4 + 1D DPA-TP4 | none | none |

How the launcher scales decode with CONC (`amd_utils/server_sglang.sh`, `models.yaml`):
- **DP+EP decode.** `max-running-requests` is the sweep's maximum CONC, and
  `MORI_MAX_DISPATCH_TOKENS_DECODE = maxCONC / dp_ranks × (MTP+1)`.
- **No-DP decode.** Graph bs 1–128, with max-running 128.
- **EP-only decode.** Graph bs 1–256.
- **DPA decode.** Graph bs 1–160 per rank.

---

## 5. NVIDIA B200 single-node (for comparison; brief)

All rows are 8k1k. Images come from `configs/nvidia-master.yaml`.

| Config (image) | Bands | Band-switched settings |
| --- | --- | --- |
| `dsv4-fp4-b200-sglang` (`sglang:nightly-dev-cu13-20260628`) | TP8 EP1 c1–32; **DEP8 c64–1024** | TP: `flashinfer_mxfp4` MoE, chunked 8192, max-running 512, graph bs 512, mem 0.90. DEP: `--moe-a2a-backend megamoe` plus about a dozen `SGLANG_OPT_*` MegaMoE switches (a bundle: flipping only the main switch fails, #539, #542, #549), graph bs 544, max-running 4352, chunked 16384, `--enable-mixed-chunk`, mem 0.835. SWA ratio 0.5 at 1k, 0.1 at 8k, 0.12 under DEP |
| `dsv4-fp4-b200-vllm{,-mtp}` (`vllm-openai:v0.25.0`) | TP8 c1–64; TEP8 c128–256; DEP8 c64–1024 | DEP: `--moe-backend deep_gemm_mega_moe`, EPLB, `--prefill-schedule-interval 4` (#559). `max-num-batched-tokens` = ISL at 8k when CONC≤128, otherwise 2048. MTP 2 everywhere, with batch tokens 2×ISL |
| `dsv4-fp4-b200-trt{,-mtp}` (`trtllm-deepseek-v4:…c185066`) | TP8 c1–32; DEP8 c32–512 (MTP: to c256) | `MEGAMOE_DEEPGEMM` MoE only at ISL≤1024 with CONC≥2048 (MTP: CONC≥512), TRTLLM otherwise. **MTP layers 3, or 2 at ISL≥4096 with CONC≥128**. KV free fraction 0.9 on TP, 0.7 on DPA (MTP 0.6). `max_batch = max(CONC, 16)`. `attention_dp_config.enable_balance` (#841) |
| `dsr1-fp4-b200-sglang{,-mtp}` (`v0.5.16-cu130`) | TP4 c1–32; **DEP4 c64–256** | `--scheduler-recv-interval` 10 below c16, 30 at c16 and above, 1 under DPA. Chunked 16384, or 32768 under DPA. DPA adds `--enable-dp-attention-local-control-broadcast`, `--enable-dp-lm-head`, `--schedule-conservativeness 3.33`, prefill delayer. MTP EAGLE 2/1/3 |
| `dsr1-fp8-b200-sglang` (`v0.5.12-cu130`) | TP8 c1–4; TP4 c1–32 | TP8: max-running and graph bs 128, mem 0.82, chunked 32768. TP4: max-running and graph bs 32, **mem 0.95**, chunked 8192, recv 10. TP4 fits only with these caps (#265). MTP twin: TP8 c1–512 |
| `dsr1-fp4-b200-trt{,-mtp}` (`trtllm 1.3.0rc14`) | TP4 c4–32, TEP4 c32, DEP4 c256, TP8 c4, DEP8 c128–256 | DPA: CUTLASS MoE and graph batch CONC/4 (FP8: CONC/8). **MTP 3 on TP, 1 under DPA** (#269, #312). `max_num_tokens = roundup64(CONC+ISL+64)` (#130). FP8 uses piecewise graphs at CONC≥64 |
| `qwen3.5-fp8-b200-sglang{,-mtp}` (`v0.5.14-cu130`) | TP8 c4; TP4 c4–256 | Earlier: TP8/EP1 at c4–16, TP4/EP4 at c16–128, all-reduce fusion only at TP8 (#414) |
| `qwen3.5-fp4-b200-sglang{,-mtp}` | TP4 c4; TP2 c4–128 (MTP: TP2 c4–64, TEP2 c16–64) | none |
| `qwen3.5-fp4-b200-trt{,-mtp}` (`1.3.0rc18`) | TP2 at c4–16; TP4 and TEP8 at c4; TEP2 c8–32; DEP8 c256–1024 (MTP: c128–1024); DEP4 c1024 | none |

B200 agentic configs:
- **DSV4 SGLang:** TP8 resident c1–5, TP8 + HiCache c8–32, DEP8 + HiCache with a router c32–96. Max-running and graph bs 2×CONC.
- **DSV4 vLLM:** TP8 with `vllm-simple` offload c1–14, DEP8 c32–196.
- **Qwen3.5 FP8:** TP8 resident to c64, then HiCache; TP4 resident to c14, then HiCache.
- **MiniMax-M3 FP4 vLLM:** TP8 c1; TP4 c1–20; TP4 + offload c15–40.
- **GLM-5.2 FP4:** TP8 + HiCache c1–16.

Deprecated B200 configs, one line each:
- **GLM-5 FP4:** TP8 at c4, TP4 at c4–256. **GLM-5 FP8:** TP8 c4–256.
- **gpt-oss vLLM:** TP1 and TP2 to c128, TP8 at c4 only.
- **Kimi-K2.5 FP4:** TP4, TP8, TEP4 and TEP8 at c1–512, plus DEP8 at c512–1024.
- **MiniMax-M3 FP8:** TP4 and TP8 at c1–64, TEP to c256, DEP at c64–256.
- **MiniMax-M2.5 FP4 TRT:** TP1 at low CONC, DEP at c128–1024.

---

## 6. How multi-node disaggregated layouts change with concurrency

The NVIDIA recipes show the shape most clearly (`nvidia-master.yaml`, `dsv4-fp4-*-dynamo-*`).

**At low CONC, many narrow decode workers sit behind one prefill worker.**
- GB200 SGLang: c1 runs 1P TP8 + 1D TP8. c64 runs 1P DEP8 + **4D TP8**.
- The MTP twin runs 1P DEP8 + **6D TP8** at c128.
- B200 SGLang MTP runs 1P6D at c32–128.
- Each decode worker keeps a small batch, so TPOT stays low.

**At mid CONC, one wide DEP decode takes over.**
- GB200: c256 runs 1P DEP8 + 2D DEP16, and c512 runs 1P1D DEP16.
- B300 vLLM runs 1P1D DEP8 with MegaMoE at c256–1024.

**At high CONC, the prefill share keeps growing.** An 8k prompt needs far more prefill capacity once each decode rank's
batch saturates.
- GB200 grows from 2P1D at c1536 to 4P1D at c4096 and 5–6P1D at c8192.
- GB300 SGLang runs **8–15 DEP4 prefill workers against 1 decode** of DEP12 to DEP40 at c2048–12000.
- Decode width grows with CONC (DEP8 → DEP16/24/32/40) to hold the KV and the batch (#569, #570, #615, #620).

**MTP depth falls along the curve.**
- DSR1 B300 TRT goes from many TP8 decode workers with MTP3 down to DEP with MTP1 (#316).
- DSV4 GB300 TRT reaches 12 DEP4 prefill + 1 DEP8 decode with MTP1 at c4301.

**AMD's MoRI layouts follow the same direction, inside 3–9-node pools, so P:D tops out near 2:1** (#1001).
- **DSR1 FP8:** 1P TP8 + 2D TP8 at the bottom, 2P DEP8 + 1D DEP8 at the top.
- **DSR1 FP4:** 1P1D TP8 → 1P2D TP8 → 1P1D DEP8 → 2P DEP8 + 1D DEP8.
- **Per-GPU wins come from narrower prefill, not wider decode.**
  - Qwen3.5 1P TP4 + 1D TP8: +24–31% per GPU (#1210).
  - Kimi-K2.5 all TP4 with 2D at high CONC (#1051).
  - MiniMax-M3 2P TP4 + 1D TP4 for the prefill-bound 8k1k tail (#1001).

---

## 7. Single-setting configs (one layout; only CONC-scaled caps vary)

- **MI355X:**
  - `dsv4-fp4-mi355x-vllm{,-mtp}`: TP8 c4–512.
  - `qwen3.5-fp8-mi355x-sglang{,-mtp}`: TP4 c4–256.
  - `dsr1-fp8-mi355x-atom{,-mtp}`: TP8.
  - `dsr1-fp4/fp8-mi355x-sglang-mtp`: TP8 c4–64.
  - `dsv4-fp4-mi355x-atom-agentic-mtp`: TP8 c1–48.
- **gfx942:**
  - `dsr1-fp8-mi300x/mi325x-sglang{,-mtp}`: TP8 c4–64.
  - `qwen3.5-fp8-mi300x/mi325x-sglang{,-mtp}`: TP8 c4–64.
  - `dsv4-fp8-mi300x/mi325x-vllm{,-mtp}`: TP8, capped at c256 (STP) and c128 (MTP).
  - `glm5.2-fp8-mi325x-sglang-agentic-mtp`: TP8 c1–8.
  - `minimaxm3-fp8-mi325x-vllm-agentic-mtp`: TP8 c1–18.

---

## 8. Cross-model rules (with the configs that support each)

1. **TP-only at low CONC; DPA (often with EP) at high CONC.** Where the crossover sits:
   - **MI355X DSV4 SGLang:** DPA from c16 at TP4 and from c64 at TP8 (#755, #864).
   - **DSV4 ATOM:** DPA from c16 at TP4 and from c64 at TP8; with MTP, from c32 (#1067, #1068).
   - **B200 DSV4 SGLang, vLLM and TRT:** DEP from c32–64. **DSR1 B200 SGLang:** DEP4 from c64.
   - **AMD DSR1 MoRI decode:** DEP8 from c128–512.
   - **The DPA crossover is lower on TP4 replicas** than on TP8.
   - **Models whose AMD recipes never use DPA:**
     - Qwen3.5, because of the MoRI shared-expert slot assertion (#768, #1210).
     - GLM-5.x and DSV4 agentic, because DSA+DP hangs a collective on ROCm (#1158, #1181).
     - MiniMax-M3 and Kimi-K2.5, where a TP4 replica beats DPA per GPU (#972, #1051).
2. **Fewer GPUs per replica win under per-GPU scoring, wherever weights plus KV fit.**
   - **MI355X:**
     - TP4 for DSR1 (#385, #825), GLM-5 FP8 (#653), Qwen3.5 FP8 (#819), MiniMax-M3 (#972) and Kimi-K2.5 (#1051).
     - TP4 for DSV4 SGLang, fixed-seq-len (#864) and agentic (#1232).
     - TP2 for Qwen3.5 FP4, GLM-5.1 FP4 and MiniMax-M3 FP4 (#1089). TP1 for gpt-oss and MiniMax-M2.5 FP4.
     - TP4 prefill in disaggregated serving (#1210).
   - **B200:** DSR1 FP8 TP4 (#265), Qwen3.5 TP4 and TP2.
   - **Limits:**
     - Kimi-K3 is TP8-only at about 195 GB per GPU.
     - DSV4 TP4 with MTP and DPA leaves a negative KV budget (#1067).
     - DSV4 FP8 on gfx942 is TP8-only.
     - TP4 needs its memory and batch caps changed with it, as one bundle (#265).
3. **The widest TP appears only at the lowest CONC, for latency.**
   - DSR1 FP4 ATOM: TP8 at c4 only.
   - GLM-5 MTP: TP8 at c4–8.
   - Qwen3.5 B200: TP8 at c4.
   - gpt-oss: TP8 at c4–16.
   - MiniMax-M3 at 8k1k: TP8 at c1–2 (#935).
   - GLM-5.2 agentic: TP8/EP8 resident at c1–4, beside TP4/EP4 + HiCache. DSV4 agentic SGLang went the other way and
     replaced resident TP8 with TP4 at c1–10 (#1232).
4. **MTP depth falls as the batch grows, and drafting stops at the top.**
   - DSR1 TRT: 3 on TP, 1 on DPA (#269). DSV4 TRT: 3, then 2 at c≥128 for 8k.
   - DSR1 FP4 MoRI: 3 → 2 → 1 (#342). DSV4 MoRI: 2 on TP8, 1 on DEP8 (#1059).
   - DSV4 SGLang MTP first used a shorter DPA chain (2/1/3) than the TP chain (3/1/4) (#798).
   - Kimi-K3 ATOM: 7 → 3 → none at c≥32 (#1254).
   - DSV4 vLLM MTP: loses from about c32 (#796).
   - MiniMax-M3 MTP rows are trimmed at extreme CONC.
   - The one deep-at-low-CONC case is GLM-5.2: EAGLE 5/6 at c1–16 (#1167). It is valid only with its AL input raised to
     match, since agentic AL is synthetic.
5. **TBO only above a token threshold, and on ROCm only with `GPU_MAX_HW_QUEUES=5`.**
   - DSV4 ATOM: c≥64 without MTP (#1068), c≥256 prefill-only with MTP (#1067: −14% at c64, +8–14% at c≥256).
   - DSV4 SGLang: on DPA cells only (#995). The threshold moves with the image (#877 → #1068).
6. **Capacity caps follow CONC.**
   - **SGLang:** `--max-running-requests` and `--cuda-graph-max-bs` = CONC in fixed-seq-len (DSV4, Qwen3.5,
     GLM-5, DSR1 FP8, #301, #653, #995). Agentic uses 2×CONC, capped at 256, with graph bs capped at 64–128.
   - **ATOM:** `--max-num-seqs CONC` on every DPA cell, because each rank keeps a full KV pool (#550, #1068).
   - **TRT:** `max_batch = CONC`, divided by 4 or 8 per rank under DPA (#312). `max_num_tokens = CONC+ISL+64` (#130).
   - **Exception:** DSR1 FP4 MI355X and DSR1 on MI30x keep a fixed graph bs of 128.
7. **The prefill chunk follows ISL and the DP degree.** SGLang divides the chunk across DP ranks, so the chunk is ISL
   on TP and ISL×TP under DPA (#859, #901). DSR1 FP4 shrinks the chunk at 8k when CONC>32 (#205). Agentic recipes use
   8–32K chunks to interleave 100K-token prefills (#1181, #1167).
8. **Kernel and backend choice is a band policy, not a value.**
   - MiniMax-M3: sparse paged attention plus the emulation linears only at ISL≥8192 with CONC≥64 (#1050).
   - MiniMax-M2.5: block, shuffle and async per point (#611, #794).
   - DSV4 on B200: MegaMoE only on DEP, and in TRT only at c≥2048 for 1k (#564).
   - Llama-70B FP4: GEMM path at CONC≤16 (#194).
   - Scheduler receive interval varies by CONC: 2 to 1920 on H100 (#814); 10, 30, or 1 under DPA on B200 DSR1.
9. **A DRAM KV tier is added only past each width's HBM knee; the low CONC stays resident.**
   - Qwen3.5 FP4: TP4 from c40, TP2 from c20 (#1249).
   - DSV4 SGLang: TP8 from c16 (#1181).
   - MiniMax-M3: TP4 from c32 (#1165).
   - Kimi-K3: from c8 (#1254).
   - B200 Qwen3.5: TP8 from c64.
   - The host tier adds per-request latency at low load (#1167).
10. **DCP for long-context, KV-bandwidth-bound decode.** Kimi-K3 ATOM uses DCP8 from c8, with the offload chunk
    aligned to page × DCP (#1254). It is the only single-node recipe on main that uses DCP.
