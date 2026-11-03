# Image and capability manifest, and workaround-flag inventory

**What this is:** which serving image (vLLM, SGLang, ATOM, TRT-LLM, Dynamo; with AITER, MoRI and ROCm
inside) first carried which capability, kernel or fix that an InferenceX recipe then used; what
failed before it; which defaults changed silently between versions; and every flag the recipes set
to disable a fast path or dodge a bug, with when it went away.

Built 2026-10-20 from the runner's docs/research/inferencex-study (records.jsonl ids) and InferenceX
origin/main (`6d6d296c6`, 2026-08-30).

**How to read it:**
- `#425` is a record id in `evidence/records.jsonl` (this folder). Its
  `commit` field opens the change in a clone of github.com/xiaobochen-amd/InferenceX: `git show <commit>`.
  `vllm#N`, `sglang#N`, `aiter#N` are upstream PRs or issues as the recipe named them.
- "First used" is the date an InferenceX recipe first used the image, not the upstream release date.
- MI355X = gfx950; MI300X and MI325X = gfx942. **[numerics]** = changes model output: run the
  accuracy gate (GSM8K at the golden score) before a KEEP. **[regime]** = the sign depends on the
  operating point (concurrency, ISL/OSL, TP/EP/DP); the entry says where it holds.
- Sections: 0 rules; A AMD (A1 vLLM ROCm, A2 SGLang ROCm, A3 ATOM, A4 AITER and FlyDSL, A5 MoRI,
  A6 ROCm and firmware, A7 blocked before, A8 default changes and silent no-ops, A9 known-bad
  images); N NVIDIA (N1 vLLM, N2 SGLang, N3 TRT-LLM and Dynamo); W workaround inventory (W1–W4
  AMD, W5 NVIDIA).

## 0. Rules for an agent

1. **Re-test every workaround on each image change. Removing a stale workaround is a candidate
   optimization.** Examples: two FP8 MoE disables outlived their bug and their removal was part of a
   B200 win (#448); 15 DSV4 env overrides became defaults in SGLang v0.5.14 (#995, #998); a whole
   DSV4 ATOM overlay, a CONC==1 guard and `--enforce-eager` went away with one nightly (#652). §W
   lists every one, with its status at origin/main.
2. **"Blocked by image", not a dead end.** When a feature fails or does nothing in the pinned image,
   look it up in A7 (and N). If a later image fixed it, record
   `BLOCKED_BY_IMAGE <pinned tag or digest>; works from <tag> (#id)`, not DEAD_END: a dead end written
   here steers later iterations away from a feature that works one image later (#425, #955; study
   REPORT §2.2).
3. **Image facts expire, both ways.** ASM paged attention won on one ATOM image and lost to Triton a
   week later (#926 → #941); Marlin won one week and lost the next on NVIDIA (#869 → #899, #902). A
   kill or a KEEP carries its image digest.
4. **Prove the flag took effect, per arm.** Env names get renamed and are then ignored silently
   (#286, #367); size gates make a flag a no-op (#1089, #1118); launchers drop flags (#595, #162).
   Check the backend the server log says it selected and the kernel names in the profile (#286,
   #367), not only the throughput delta.
5. **Pull by digest.** Nightly tags get retired or garbage-collected (#1142, #1051); the enroot cache
   keys on the tag string, so recipes pin digest-suffixed tags (#731).
6. **What is in the image?** Read it from the container before reasoning about a feature:
   `pip list | grep -i -E 'vllm|sglang|atom|aiter|mori|triton|torch'`, `cat /opt/rocm/.info/version`,
   and grep the installed tree for the env var (an unread `VLLM_*` var is silently ignored, #286).

## A. AMD

### A1. vLLM ROCm (`vllm/vllm-openai-rocm`, and AMD's `rocm/7.0` vLLM builds)

**Releases**

| Image | First used | What became possible, or what it fixed | Upstream | Recipes | Ids |
| --- | --- | --- | --- | --- | --- |
| `rocm/7.0` vLLM 0.10.1 `20250915` (ROCm 7.0) | 2025-09-20 | ROCm 7.0 base; INT4 quick-reduce (`VLLM_ROCM_QUICK_REDUCE_QUANTIZATION`) and the AITER CK MHA switch used per workload | — | Llama-70B FP8, gpt-oss | #152 |
| `rocm/7.0` vLLM 0.10.1 `20250927_rc1` ("0927 final") | 2025-09-28 | AITER paths on by default: `VLLM_ROCM_USE_AITER=1`, AITER RMSNorm, the `TRITON_HIP_*` block, `NCCL_MIN_NCHANNELS=112`, `ROCM_TRITON_MOE_PRESHUFFLE_SCALES=0` became redundant or slower and were removed. **Changed the default of `VLLM_ROCM_USE_AITER_MHA`** (set it explicitly) | — | 70B FP8/FP4, gpt-oss; MI300X, MI325X, MI355X | #182 #186 #198 #199 #204 |
| `v0.14.0` (first upstream ROCm image used) | 2026-01-21 | `--async-scheduling` is the default. AITER needs the master switch `VLLM_ROCM_USE_AITER=1` again, and `VLLM_USE_AITER_UNIFIED_ATTENTION` became `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION` (old name silently ignored) | — | gpt-oss MI300X/MI325X | #274 #286 |
| `v0.15.1` | 2026-02-24 | Skinny-GEMM dispatch fixed (small-M GEMMs back on skinny kernels); MoRI EP all2all backend; AITER shuffled KV layout with matching paged attention (`VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1`) | — | gpt-oss; MiniMax-M2.5 | #359 #561 |
| `v0.16.0` | 2026-02-26 | Fused add+RMSNorm+pad for GPT-OSS by the compile PassManager; AITER attention block-size fix; `cudagraph_mode FULL_AND_PIECEWISE` replaces hand-listed capture sizes | — | gpt-oss, all AMD | #367 |
| `v0.17.0` | 2026-03-09 | `--attention-backend ROCM_AITER_UNIFIED_ATTN` replaces the env pair; compile pass `fuse_rope_kvcache` with `use_inductor_graph_partition` (RoPE fused into the KV write) | — | gpt-oss | #396 |
| `v0.18.0` | 2026-03-23 | **AITER MLA decode accepts 8 heads per rank** (64-head MLA at TP8), so `VLLM_ROCM_USE_AITER=1` works for Kimi-K2.5; tuned Triton fused-MoE configs | vllm issue #35641; vllm PR #35093 | Kimi-K2.5 FP4/INT4 MI355X, MI325X; MiniMax-M2.5 TP8/EP8 | #374 #425 #426 #427 #412 #436 |
| `v0.19.0`, `v0.19.1` | 2026-04-05 | `ROCM_AITER_FA` attention with fp8 KV and INT4 QR for MiniMax-M2.5. `v0.19.1` is the KV-offload floor for the agentic scenarios; older images hit HMA and FlashInfer assertions there | — | MiniMax-M2.5; agentic entries | #449 #517 #586 |
| `v0.20.2` | 2026-05-31 | Last image before `v0.21.0`'s AITER-MoE graph-capture crash on MiniMax-M2.5; the upstream recipe's `min_vllm_version` | recipes#481 | MiniMax-M2.5 MI325X | #794 |
| `v0.21.0` | 2026-05-16 | Bot bumps with no stated gain. Carries the ROCm attention default flip (A8) and crashes in AITER MoE graph capture on MiniMax-M2.5 (A9) | vllm#36702 | 10 recipes | #657 #665 #685 #692 #721 #794 |
| `v0.22.0` | 2026-05-30 | Ships the `vllm-rs` binary: `VLLM_USE_RUST_FRONTEND=1` cuts TTFT 8% (1k1k) and 22% (8k1k), tok/s and TPOT flat; ROCm DSV4 MTP | vllm#43385 | MiniMax-M2.5 FP4; DSV4 MTP | #777–#793 #823 #796 |
| `v0.24.0` | 2026-07-05 | MoRI-IO fixes upstreamed (disagg runtime patches dropped); a2a backend `mori` → `mori_low_latency`; MoRI-IO read mode moves from an env var to `kv_connector_extra_config` | vllm#40344 | Kimi-K2.5, MiniMax-M2.5 disagg | #987 #966 |
| `v0.27.1` | 2026-08-12 | Current pin of the MiniMax-M3 agentic recipes (MI355X, MI325X, MI300X) | — | MiniMax-M3 | #1165 |

**Nightlies and special tags** (`vllm/vllm-openai-rocm:<tag>` unless named)

| Tag | First used | What became possible, or what it fixed | Upstream | Recipes | Ids |
| --- | --- | --- | --- | --- | --- |
| `nightly-b50646e5` | 2026-05-19 | DSV4 ROCm base support merged: multi-request DSV4 on a stock image. Before it, vLLM was rebuilt at start-up from two unmerged PRs on an ATOM base, conc 1 only. Still eager, `triton_unfused` MoE | vllm#40871 (merged 2026-05-05), #40889; recipes#433 | DSV4-Pro MI355X | #555 #731 |
| `nightly-4f940896` | 2026-05-21 | DSV4 graph capture on ROCm: `--enforce-eager` dropped for `FULL_AND_PIECEWISE`; sweep conc 128 → 512 | — | DSV4 | #756 |
| `minimax-m3` | 2026-06-13 | MiniMax-M3 day 0. Decode asserts in the breakable-cudagraph path; the AMD model class lacks `SupportsEagle3` | — | MiniMax-M3, all AMD | #844 #849 #848 |
| `nightly-b8336c3c` | 2026-06-16 | `--moe-backend flydsl` (FlyDSL w4a16 MoE) | — | Kimi-K2.5 INT4 MI355X | #867 |
| `nightly-b53b1c7f` | 2026-06-18 | `SupportsEagle3` upstream for the AMD MiniMax-M3 model | vllm#45546 | M3 MTP MI300X | #883 |
| `nightly-3f5a1e17` | 2026-06-25 | Quark MXFP4 loading | vllm#45794 | MiniMax-M3 MXFP4 | #917 #923 |
| `nightly-4559c43a` | 2026-06-30 | AITER MoE for M3; shared-expert fusion; gfx950 MXFP8 MoE/linear tuning; AITER FlyDSL MoE backend. Bugs: the AITER master switch gives degenerate M3 output in one layout (A7); EP with `moe_backend=auto` falls to EMULATION and hangs about 8 h | vllm#46419 #46545 #45725 #46184; recipes#581 | MiniMax-M3 FP8/FP4, M3 MI300X | #934 #935 #939 #982 |
| `nightly-09663abd` | 2026-07-04 | Split-KV (two-stage) decode attention; the AITER MXFP4 MoE path registers FP4 scales, so `--moe-backend aiter` loads DSV4; AITER on TP-only M3 fixed | vllm#47158 | DSV4 STP/MTP MI355X, DSV4 FP8 MI325X/MI300X, M3 | #955 #957 #972 #1011 |
| `nightly-9e57de71` | 2026-07-22 | AITER Gluon page-16 sparse paged attention. Needs exactly one KV head per rank (TP4 for M3) | vllm#47287 | MiniMax-M3 MXFP8 | #1050 #1165 |
| `nightly-dcfebf93` | 2026-07-28 | Native `OffloadingConnector` with the hybrid KV manager left on | — | M3 MXFP4 agentic | #1073 |
| `nightly-387189c4` | 2026-07-30 | AITER ≥ v0.1.16.post5 with the gfx950 MXFP4 MoE backend | vllm#48683, aiter#3832 | Kimi-K2.5 MXFP4 | #1096 |
| `nightly-cb8104839c` | 2026-08-13 | Kimi-K3 base; still needs nine backported PRs (A4) | see A4 | Kimi-K3 | #1171 |
| `nightly-311b3513` | 2026-08-17 | Native 384-expert MXFP4 at TP8 (no padding to 512: 30% less MoE compute); gfx950 Triton sparse-MLA decode; DSV4 regression fix; MLA `num_kv_split` retune; cached e8m0 scale upcast in AITER `scaled_mm`. Nightly default `max-num-batched-tokens` 16384 | vllm#51473 #52212 #52401 | DSV4 AgentX MTP | #1201 |
| `nightly-bf610c2f`, `nightly-a6682d1d` | 2026-05-28 | MoRI-EP and `MoRIIOConnector` disagg, with runtime patches | — | Kimi-K2.5, MiniMax-M2.5 disagg | #769 #770 |
| `rocm/vllm-dev:vllm-0.23.1-rocm723-mi35x-mori-0625` | 2026-06-26 | MoRI-IO KV disagg on ROCm 7.2.3 | — | M3 FP4 disagg | #929 #1118 |

### A2. SGLang ROCm (`lmsysorg/sglang`, `lmsysorg/sglang-rocm`, `rocm/sgl-dev`)

| Image | First used | What became possible, or what it fixed | Upstream | Recipes | Ids |
| --- | --- | --- | --- | --- | --- |
| `v0.5.5.post2/post3-rocm700-mi35x` (before: `rocm/7.0 sgl-dev-v0.5.2`, #152) | 2025-12-05 | First upstream ROCm images used: INT4 QR (`ROCM_QUICK_REDUCE_QUANTIZATION`), `--attention-backend aiter`, fp8 KV. Reverted the same day, re-landed in parts | — | DSR1 MI355X | #236 #238 #244 #255 |
| `v0.5.7-rocm700` | 2026-01-22 | `SGLANG_AITER_MLA_PERSIST=1`: persistent AITER MLA decode, valid only with FP8 KV | — | DSR1 FP8 MI300X/MI325X | #279 #282 |
| `v0.5.8-rocm700` | 2026-01-27 | Persistent MLA turned off automatically when KV is not FP8 | sglang#17327 | DSR1 | #287 |
| `v0.5.9-rocm700` | 2026-02-26 | AITER v0.1.10.post3: FP8 prefill/decode attention and FP8 KV, an FP8 prefill attention kernel; MoRI EP two-batch overlap; DeepSeek weight-load OOM fix | — | DSR1, all AMD | #368 |
| `rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260219` | 2026-03-06 | GLM-5 day 0: TileLang NSA prefill/decode, transformers from git | — | GLM-5 FP8 | #392 |
| `v0.5.10-rocm720` | 2026-04-08 | Qwen3.5 on MI30X: aiter attention and `--enable-aiter-allreduce-fusion`. The GLM-5 bump was reverted the same day, reason not stated | — | Qwen3.5, GLM-5 | #455 #457 #494 |
| `v0.5.10rc0-rocm720-mi35x-20260413/14/15` | 2026-04-16 | Qwen3.5 hybrid-model ROCm paths; fp8 KV with GLM-5's NSA | sglang#21188 #21421 #20736 | Qwen3.5, GLM-5 | #469 #470 #481 |
| `rocm/sgl-dev:deepseek-v4-mi35x` | 2026-04-24 | DSV4 day 0, with about 20 correctness fallbacks and no graphs (W2) | sglang PR #23608 | DSV4 FP8 | #524 #719 |
| `-DSv4` builds (amd/deepseek_v4 branch): `a8410de-20260502`, `bfd32b6-20260507`, `b19052c-20260518`, `f96ac98-20260526` | 2026-05-03 | Fuse-compress-decode (sglang#24249); AITER MHC pre/post and Triton SWA prepare; fused hash top-k, fused store-cache, fused paged compress, fused WQA/WKV, Triton FlashMLA, FlyDSL MoE. **None carry sglang#26383** (MTP crashes) | sglang#24249 | DSV4 FP4 | #595 #617 #741 #771 #798 |
| `v0.5.10.post1-rocm720-mi35x-20260501` | 2026-05-07 | MoRI FP4 dispatch and FP8 combine [numerics], MoRI SDMA with two-batch overlap, spec v2 | — | DSR1 FP4 disagg | #616 |
| `v0.5.12-rocm700/720`, `sglang-rocm:v0.5.12-…-mi35x-20260517` | 2026-05-17 | Bot bumps, no stated gain. The `-20260517` build is known bad for GLM-5.1 FP4 (A9) | sglang#25742 | many | #672 #694 #705 #725 #779 |
| `v0.5.12-…-20260519` | 2026-05-27 | MoRI per-phase combine dtypes (`MORI_COMBINE_DTYPE_PREFILL/DECODE`, blockwise-scaled FP8) | — | DSR1 FP4 disagg | #766 |
| `v0.5.12.post1-…-20260529` | 2026-05-30 | Fix for the GSM8K collapse | sglang#25742 | GLM-5.1 FP4 | #779 #795 |
| `v0.5.12.post1-…-20260601` (mainline nightly) | 2026-06-01 | DSV4 MTP graph fix and sparse Triton attention | sglang#26383 | DSV4 MTP | #798 |
| `v0.5.12.post1-…-20260528/0604` | 2026-06-06 | AITER unified attention for Qwen3.5 (`SGLANG_USE_AITER_UNIFIED_ATTN=1`, `--page-size 16`) | — | Qwen3.5 FP8/FP4 | #819 #827 |
| `v0.5.12.post1-…-20260610` (main) | 2026-06-11 | DSV4 on SGLang main: `--attention-backend dsv4`, `unified_kv_triton` sparse MLA, AITER indexer; env vars renamed (A8); `model_type` patch no longer needed | — | DSV4 | #828 #831 |
| `v0.5.13-…-20260612` | 2026-06-15 | MoE `intermediate_pad` fix (no GEMM work on padding); FlyDSL kernels worth forcing (`AITER_FLYDSL_FORCE=1`) | sglang#27858 | DSV4; Qwen3.5 FP4 | #859 #863 |
| `v0.5.13.post1-…-20260618/0622` | 2026-06-22 | `SGLANG_DP_USE_GATHERV=1` | — | DSV4; GLM-5.1 | #901 #944 |
| `v0.5.14-…-20260706/08` | 2026-07-10 | 15 DSV4 `SGLANG_OPT_*` overrides are now defaults; deep_gemm-absence fallbacks removable; DP-attention local shared expert, gatherv/reduce-scatter and TBO | — | DSV4 | #995 #998 |
| `v0.5.15-…-20260713` | 2026-07-16 | Rejects mismatched speculative state between prefill and decode (v0.5.14 accepted it and lost accuracy, A8); `SGLANG_MAMBA_SSM_DTYPE` | — | DSV4 disagg MTP; Qwen3.5 | #1059 #1025 |
| `v0.5.16-…-20260726/28/30` | 2026-07-28 | Carries AITER all-reduce+RMSNorm fusion, fused QK-norm+RoPE+KV-write and FlyDSL MoE sorting (GLM-5.2 turned them on: +23.6% P90 interactivity at conc 8); Qwen3.5 disagg +3–5% | — | GLM-5.2, Qwen3.5 | #0 #1255 #1210 |
| `v0.5.18-…-20260822` | 2026-08-24 | Tuned MoE tables cover DSV4's fused shape (385 experts, top-k 7), so `--enforce-shared-experts-fusion` pays; `SGLANG_OPT_USE_AITER_BATCHED_GEMM=1` | — | DSV4 AgentX | #1232 |

### A3. ATOM (`rocm/atom`, `rocm/atom-dev`)

| Image | First used | What became possible, or what it fixed | Upstream | Recipes | Ids |
| --- | --- | --- | --- | --- | --- |
| `rocm/atom:rocm7.1.1-…-atom0.1.1-MI350x` | 2026-01-14 | ATOM, AMD's AITER-based engine | — | DSR1 FP8/FP4, gpt-oss | #266 |
| `atom0.1.2` (rocm7.2.1) | 2026-04-02 | Multi-token MTP (`--num-speculative-tokens 3`; one before) | — | DSR1 FP8 MTP | #445 |
| `atom0.1.2.post` (rocm7.2.1/7.2.2) | 2026-04-09 | GLM-5, Qwen3.5, MiniMax-M2.5 configs. DSV4 only with the unmerged ATOM#650 overlay, conc 1 | ATOM#650 | | #459 #531 #551 #580 |
| `rocm/atom-dev:nightly_202605130853` | 2026-05-14 | DSV4 multi-request sparse attention and the MXFP4 MoE pieces ship in the image: overlay, conc-1 guard and eager removed; conc 1 → 512 | aiter#2822, aiter#2900, aiter#2642, aiter#2916 | DSV4 FP4 | #652 |
| `rocm/atom:rocm7.2.3_…_atom20260511` | 2026-05-19 | Component not named. Kimi-K2.5 up to +31% tok/s/GPU at TP4 conc 4–16; Qwen3.5 FP8 TP4 +3.2 to +16.3% | — | | #733 #736 #745 |
| `atom0.1.3` (rocm7.2.4) | 2026-06-02 | Component not named: DSR1 FP8 MTP +47% to +131% over prior numbers; DSV4 DP attention | — | DSR1, DSV4 | #805 #809 |
| `nightly_202606101403` | 2026-06-12 | DSV4 disagg: mooncake RDMA KV transfer, atomesh router | — | DSV4 disagg | #832 |
| `nightly_202606161823` (atom0.1.4_20260612) | 2026-06-18 | DSV4 two-batch overlap (`--enable-tbo`) | — | DSV4 | #877 |
| `rocm/atom-dev:M3` | 2026-06-17 | MiniMax-M3 day 0; FP8 KV blocked (A7) | — | M3 FP4 | #874 |
| `MiniMax-M3-20260623` | 2026-06-26 | FP8 KV works; `ATOM_M3_SPARSE_USE_ASM_PA=1` | — | M3 FP4/FP8 | #926 #927 |
| `MiniMax-M3-20260630`, `nightly_202607011530` | 2026-07-01 | Triton sparse attention beats ASM PA (`ATOM_FORCE_ATTN_TRITON=1`); `--online_quant_config` ptpc_fp8 [numerics]; two env names deprecated (A8) | — | M3 | #941 #952 |
| `atom0.1.4_202607091539` (rocm7.2.4) | 2026-07-13 | `AITER_MXFP4_INTERMEDIATE=1` [numerics] | — | Kimi-K2.5 FP4 | #1003 |
| `nightly_202607231538` | 2026-07-28 | DSV4: TBO from conc 64 without MTP; prefill-only TBO with MTP | — | DSV4 | #1067 #1068 |
| `…_glm52_agentic_0813`, `nightly_202608201032` | 2026-08-19 | GLM-5.2 and DSV4 agentic MTP | ROCm/ATOM#1877, ROCm/ATOM#1948 | | #1208 #1223 |
| `…_kimi_k3_agentic_0821` | 2026-08-28 | DCP, DSpark drafting, ReplaySSM, CPU state-offload tier | — | Kimi-K3 | #1214 #1254 |

### A4. AITER and FlyDSL

| AITER change | How it reached a recipe | What it enables | Ids |
| --- | --- | --- | --- |
| v0.1.10.post3 | SGLang `v0.5.9` | FP8 prefill/decode attention, FP8 KV, FP8 prefill attention kernel | #368 |
| base `dde1703e` (has aiter#2770, a16w4 MoE) + aiter#2822 (batched MXFP4 GEMM, gfx950) + aiter#2900 (MXFP4 scale padding when K is not a multiple of 256) + aiter#2642 (TP4/8 MXFP4 MoE dispatch) + `sunway513/aiter@e450e4d` (DSV4 FP4 tuned fMoE CSV, routed to FlyDSL through `AITER_CONFIG_FMOE`) | rebuilt at start-up in the ATOM recipe, then shipped in `rocm/atom-dev:nightly_202605130853` | native AITER MXFP4 MoE instead of Triton `matmul_ogs` | #560 #652 |
| aiter#2916 | in-place patch, then the same nightly | `mhc_pre` intermediates on the residual's device (fixes a HIP device-guard crash) | #551 #560 #652 |
| ≥ v0.1.16.post5 with aiter#3832 | vLLM `nightly-387189c4` (vllm#48683) | gfx950 MXFP4 MoE backend | #1096 |
| aiter#4474, aiter#4494, aiter#4521 | runtime backport (`apply_k3_container_patches.sh`) on `nightly-cb8104839c`, with vllm#51171, vllm#50578, vllm#51040, vllm#51011, vllm#51682 and part of vllm#50619 | int64 KV stride in `mla_gluon` (KV > 2 GB); fresh split-K semaphore under graph capture; fp8 ASM MLA verify kernels (rebuild needs network, hipcc and a GPU); FULL graphs for AITER MLA spec decode; ASM MLA for small non-divisor head counts (12 → 16 at TP8) | #1171 |
| FlyDSL kernels | SGLang `v0.5.16` (`AITER_USE_FLYDSL_MOE_SORTING=1`); `v0.5.13` (`AITER_FLYDSL_FORCE=1`); vLLM `nightly-b8336c3c` (`--moe-backend flydsl`); vllm#46184; DSV4 `-DSv4` image | MoE sorting, FP4 MoE/GEMM, w4a16 MoE | #0 #863 #867 #935 #741 #1254 |

**Known AITER kernel-table gaps.** Each fails or regresses in the image named; check the table, not
the flag, before writing a dead end:
- CK fused MoE lacks tiles for N = 384 at TP ≥ 4 (MiniMax-M2.5 FP4; vllm issue #35637) (#517).
- `fmoe_bf16_blockscaleFp8_g1u1_vs_silu_32x384` (gfx950) writes out of bounds at M = 8K tokens on
  MiniMax-M2.5; the disagg recipe caps `--max-num-batched-tokens 4K` instead of disabling AITER MoE
  (#770; comment in `models_vllm.yaml`).
- gfx942 ASM paged-attention heuristic table lacks MiniMax-M2.5's shape (gqa 6, block 32) beyond
  TP2/EP1 conc ≤ 16 and TP8/EP8 conc ≤ 64: `get_heuristic_kernel: cannot get heuristic kernel!` (#794).
- CK fused MoE raises "Unsupported kernel config for moe heuristic dispatch" at decode on DSV4 FP4
  (disagg); the recipe routes around it with `SGLANG_OPT_*`/AITER env (#999, #1041; `env.sh`).
- AITER MXFP4 W4A4 is rejected on gfx942 (no FP4 matrix instructions): leave `--moe-backend` on auto,
  which picks W4A16 (#1011).
- A fusion pays only if the tuned table covers the new shape: DSV4 shared-expert fusion waited for
  tables covering 385 experts and top-k 7 (#1232).

### A5. MoRI and disaggregation images

| Image | First used | What became possible | Ids |
| --- | --- | --- | --- |
| `rocm/sgl-dev:sglang-0.5.6.post1-rocm700-mi35x-mori-1224` | 2026-01-08 | First AMD prefill/decode disagg: MoRI RDMA KV transfer and EP | #261 |
| `…-mori-0113`, `sglang-0.5.7-…-mori-fp4-0122` | 2026-01-17 | FP4 disagg | #272 #305 |
| `sglang-0.5.8-…-mori-0210` | 2026-02-12 | TP with MTP2 below conc 128, DEP with MTP1 above [regime] | #341 |
| `sglang-0.5.9-rocm720-mi35x-mori-0227-2/-3` | 2026-03-04 | ROCm 7.2; sub-node workers (TP4, DEP4); `--deepep-mode normal`; MXFP4 dispatch caps | #383 #406 |
| `…-mori-0313-2` | 2026-04-19 | Tried and reverted the next day (reason not stated) | #507 #511 |
MoRI in mainline images: SGLang `v0.5.10.post1`, `v0.5.12-20260519` (A2); vLLM `v0.15.1`,
`nightly-bf610c2f`, `v0.24.0` (A1). MoRI knobs (dispatch dtype per checkpoint format, dispatch token caps, RDMA queue depth, traffic
class from the NIC's PFC no-drop priority) matter only across nodes (#406, #616, #929).

### A6. ROCm and firmware

- **ROCm 7.0** (`rocm700` tags) from 2025-09 (#152). DSR1 SGLang still runs `v0.5.12-rocm700` at
  origin/main.
- **ROCm 7.2** (`rocm720`) from `v0.5.8.post1-rocm720` (#392) and the MoRI 0227 images (#383); Qwen3.5
  disagg moved 7.0 → 7.2 (#1033).
- **ATOM's ROCm:** 7.1.1 (#266) → 7.2.0/7.2.1 (#445) → 7.2.2 (#580) → 7.2.3 (#733) → 7.2.4 (#809,
  #1003).
- **ROCm ≤ 7.2.0 HIP bug:** the runtime ignores `cudaStreamCaptureModeThreadLocal`, so the NCCL
  watchdog's `hipEventQuery` invalidates decode graph capture (TP8+EP8 with MoRI); "Rank 0 scheduler
  died during initialization". Workaround `TORCH_NCCL_BLOCKING_WAIT=1`; real fix ROCm ≥ 7.2.2
  (sglang#29235, #24011; ROCm/hip#3876; pytorch#176251) (#1154, diff `23f879df2`, `env.sh`).
- **MEC firmware < 177:** RCCL needs `HSA_NO_SCRATCH_RECLAIM=1` (ROCm 6.4.3 driver note). Check with
  `rocm-smi --showfw | grep MEC` (#182, #367, #425, #1051).

### A7. Blocked before (AMD): report "blocked by image", not a dead end

| Capability | What the older image did | First image where it works | Ids |
| --- | --- | --- | --- |
| AITER MLA at 8 heads per rank (64-head MLA at TP8: Kimi-K2.5) | rejects `num_heads=8`; AITER left off | vLLM `v0.18.0` (vllm#35641) | #374 #425–#427 |
| AITER MoE for DSV4 FP4 (vLLM) | the auto backend did not register `w13_weight_scale`/`w2_weight_scale`: KeyError at load, so `triton_unfused` was forced. The diff's comment blames auto; the study's record says `--moe-backend aiter` | `nightly-09663abd` | #731 #955 #957 |
| DSV4 multi-request on vLLM ROCm | PR build, conc 1 only | `nightly-b50646e5` (vllm#40871) | #555 #731 |
| DSV4 CUDA/HIP graphs on vLLM ROCm | `--enforce-eager` | `nightly-4f940896` | #756 |
| DSV4 multi-request on ATOM | CONC==1 guard; OOM in the torch `sparse_attn` fallback | `rocm/atom-dev:nightly_202605130853` | #551 #560 #652 |
| DSV4 MTP on SGLang ROCm | crash at MTP graph capture on the `-DSv4` images | `sglang-rocm v0.5.12.post1-20260601` (sglang#26383) | #798 |
| DSV4 MTP on vLLM ROCm | — | `v0.22.0` (vllm#43385) | #796 |
| Persistent AITER MLA in SGLang | wrong or slow without FP8 KV | `v0.5.7`, guarded in `v0.5.8` (sglang#17327) | #279 #287 |
| FP8 KV with GLM-5 NSA (SGLang) [numerics] | first bump reverted | `v0.5.10rc0-…-20260413` | #455 #457 #481 |
| FP8 KV on ATOM MiniMax-M3 [numerics] | `k_scale is None` assertion in MSA `fused_qknorm` | `rocm/atom-dev:MiniMax-M3-20260623` | #874 #926 |
| AITER on MiniMax-M3 in all layouts (vLLM) [numerics] | degenerate output. #935 says TP-only breaks; the deprecated `minimaxm3_fp8_mi355x_mtp.sh` comment says EP: the recipes disagree | `nightly-09663abd` (vllm#47158); GSM8K 0.961 with AITER on | #935 #972 |
| AITER MoE with EP on MiniMax-M3 | `moe_backend=auto` → EMULATION JIT build; workers deadlock on the build lock, about 8 h | force `--moe-backend aiter` in `nightly-4559c43a` | #939 |
| EAGLE3 on AMD MiniMax-M3 | `SupportsEagle3` missing; runtime source patch | `nightly-b53b1c7f`, `nightly-3f5a1e17` (vllm#45546) | #848 #850 #883 #923 |
| MiniMax-M3 decode graphs | CUDA-graph assertion | same image with `VLLM_USE_BREAKABLE_CUDAGRAPH=0` | #844 #849 |
| Quark MXFP4 MiniMax-M3 | — | `nightly-3f5a1e17` (vllm#45794) | #917 |
| AITER page-16 sparse paged attention | — | `nightly-9e57de71` (vllm#47287); one KV head per rank only | #1050 #1165 |
| Native 384-expert MXFP4 (DSV4) | experts padded to 512 | `nightly-311b3513` (vllm#51473) | #1201 |
| Multi-token MTP on ATOM | one draft token | `atom0.1.2` | #445 |
| MoRI PD with hybrid-attention models (GLM-5, Qwen3.5) | startup crash on `v0.5.12.post1` | bind-mounted `mori_conn.py` overlay | #767 |
| Accurate MoRI FP8 combine [numerics] | `SGLANG_MORI_FP8_COMB` fails GSM8K at high conc | per-phase combine dtypes, `v0.5.12-20260519` | #609 #616 #766 |
| SGLang DSV4 on gfx942 | DSV4 nvfp4 MoE and TileLang MLA kernels exist for gfx950 only | not unblocked; vLLM used instead | #1011 |
| DSA (GLM-5.2, DSV4) with DP attention on SGLang ROCm | collective hang in long-context prefill | open at origin/main; DP-attention arm dormant | #1158 #1181 |
| `NEXTN` speculative loader on DSV4 | crashes the decode server | use `EAGLE` with top-k 1 | #1059 #1181 |
| Kimi-K3 fp8 KV + full graphs + DSpark at TP8 | stock image cannot serve it | nine backports (A4) on `nightly-cb8104839c` | #1171 |

### A8. Default changes and silent no-ops (AMD)

- **Image family:** AMD's `rocm/7.0` 0927 image turned AITER on by default (#186); upstream
  `vllm-openai-rocm` needs `VLLM_ROCM_USE_AITER=1` (#286). A recipe carried across families silently
  loses AITER.
- **`VLLM_ROCM_USE_AITER_MHA`** default changed in the 0927 image; recipes set it in both branches
  (#199, #204).
- **Renames that are silently ignored:** `VLLM_USE_AITER_UNIFIED_ATTENTION` →
  `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION` (v0.14), so the feature was never on (#286, #367); v0.17
  moved it to `--attention-backend ROCM_AITER_UNIFIED_ATTN` (#396). vLLM ignores unknown `VLLM_*` vars.
- **ROCm attention default flip:** vllm#36702 (between v0.18.0 and v0.21.0) changed the dense
  full-attention default from `ROCM_AITER_FA` to `ROCM_ATTN`: about −38% on MiniMax-M2.5 MI325X;
  pin `--attention-backend ROCM_AITER_FA` (#794).
- **vLLM CLI and defaults:** async scheduling default from v0.14 (#274); v0.20+ rejects
  `--disable-log-requests` (#643); v0.24 moved MoRI-IO read mode into `kv_connector_extra_config` and
  renamed the a2a backend `mori_low_latency` (#987); the 2026-08 nightly's default
  `--max-num-batched-tokens` is 16384, and 8192 nearly doubled DSV4's KV capacity (#1201).
- **Quick-reduce size gates [numerics]:** vLLM's INT4 QR minimum is 16 MB for bf16 TP4, so it never
  fires on decode-sized all-reduces until `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB=256`
  (then −12 to −17% TPOT at conc 64–256 on M3, #1089, #1073); prefill all-reduces exceed the maximum
  until `VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=2048` (#1118).
- **Three names for quick-reduce:** vLLM `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` (#152), SGLang
  `ROCM_QUICK_REDUCE_QUANTIZATION` (#236), ATOM/AITER `AITER_QUICK_REDUCE_QUANTIZATION` (#926). Setting
  the wrong stack's name does nothing.
- **SGLang main renamed DSV4 env vars** (`SGLANG_HACK_FLASHMLA_BACKEND` triton → `unified_kv_triton`;
  `--attention-backend compressed` → `dsv4`) (#828); v0.5.14 made 15 overrides defaults (#995).
- **SGLang divides `--chunked-prefill-size` across DP-attention ranks:** 8192 at DP8 is 1024 per rank;
  recipes use 8192×TP, later ISL×TP (#859, #901).
- **SGLang v0.5.14 accepted decode-only speculative flags in PD** and verified against unseeded
  draft state: GSM8K 0.96 → 0.88, no error, no slowdown. v0.5.15 rejects it [numerics] (#1059).
- **ATOM `nightly_202607011530` deprecated** `ATOM_M3_SPARSE_USE_ASM_PA` and
  `AITER_QUICK_REDUCE_CAST_BF16_TO_FP16` (#952).
- **`AITER_BF16_FP8_MOE_BOUND`** went 0 → 1 → 0 on the DSV4 SGLang recipe within a week (#741, #771);
  its meaning is not stated in the records.
- **Launcher no-ops:** lowercase `$isl/$osl` never reached the container, so a per-workload setting
  was dead for weeks (#152, #162); the launcher dropped `--ep-size` and a DSV4 PR sweep passed with
  EP silently off (#595).

### A9. Known-bad images (AMD)

- vLLM ROCm `v0.21.0`: AITER MoE graph-capture crash on MiniMax-M2.5, silent worker death; use
  `v0.20.2`, then `v0.22.0` (#794, #792).
- SGLang `v0.5.12-rocm720-mi35x-20260517`: GSM8K about 0.32 at TP2 on GLM-5.1 FP4 [numerics]; use
  `v0.5.12.post1-20260529` (#779).
- The rest are in A7: the `-DSv4` builds (#798), `v0.5.12.post1` with MoRI PD (#767),
  `nightly-4559c43a` (#935, #939), and any image on ROCm ≤ 7.2.0 (A6, #1154).

## N. NVIDIA (briefer)

"Adopted with" means the flag arrived with that image, but the record does not say the image
introduced it. An unexported shell assignment never reaches the server (#310); a sed patch should
grep for its own marker and exit when it is absent (#526).

### N1. vLLM (`vllm/vllm-openai`)

| Image | First used | What became possible, or what it fixed | Upstream | Recipes | Ids |
| --- | --- | --- | --- | --- | --- |
| squash image + `nvidia-nccl-cu12==2.26.2.post1` + FlashInfer `97201824`, then 0.3.0 | 2025-09-01 | FI all-reduce+RMSNorm(+FP8 quant) fusion (`VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB`), attention+quant fusion (needs `custom_ops +quant_fp8`), FULL_DECODE_ONLY with `splitting_ops []`: they pay only as a set | — | Llama-70B FP8/FP4 B200 | #15 #24 #50 |
| `v0.11.0` | 2025-11-04 | Default `cudagraph_mode` became FULL_AND_PIECEWISE (Hopper recipes pinned PIECEWISE until `v0.11.2`, #237) | — | GPT-OSS H100/H200 | #217 |
| `v0.13.0` | 2026-01-02 | pass_config renamed: `enable_fi_allreduce_fusion` → `fuse_allreduce_rms`, `enable_noop` → `eliminate_noops`; image has no `git` | — | GPT-OSS | #254 |
| `v0.15.1` | 2026-02-25 | Changelog: CUTLASS MoE about +8%, FP4 kernels about +4% E2E on B200; GPT-OSS hybrid-attention prefix-cache fix | — | GPT-OSS | #365 |
| `v0.19.0-cu130` | 2026-04-03 | FlashInfer FP8 MoE and DeepGEMM MoE usable (disables removed, W5); `VLLM_FLASHINFER_ALLREDUCE_BACKEND=mnnvl` | — | MiniMax-M2.5 FP8 B200 | #448 |
| `v0.20.0-cu130` (DeepGEMM inside) | 2026-04-28 | `--moe-backend deep_gemm_mega_moe` (MegaMoE), DP-attention/EP path only, GMU 0.85 | — | DSV4 FP4 B200/B300 | #559 #568 |
| `v0.20.1`, `v0.20.2+` | 2026-05-04 | Profiler pre-reserves CUDA-graph memory (about 57 GB/GPU): startup fails at GMU 0.90 unless the estimate is off (W5) | — | Kimi-K2.5 B200; DSV4 H200 MTP | #600 #680 |
| `v0.21.0` | 2026-05-21 | More memory at load: GMU 0.95 OOMs; explicit `--quantization deepseek_v4_fp8` | — | DSV4 FP8 H200 | #753 |
| `v0.22.0` | 2026-05-29 | MegaMoE DP path without the GMU 0.85 cap | — | DSV4 FP4 B200 | #773 |
| `nightly-3f0a91bb` | 2026-06-05 | EPLB with MegaMoE (`--eplb-config '{"communicator":"torch_nccl","use_async":false}'`); the record infers the nightly was needed | — | DSV4 B200 DEP | #816 |
| `minimax-m3` | 2026-06-12 | MiniMax-M3 day 0: `--block-size 128` (MSA), `--language-model-only`; EAGLE3 drafter must run FLASH_ATTN (FI does page 128 only in its GQA-only trtllm-gen kernel) | vllm#45381 | M3 B200/B300/H100/H200 | #834 #840 |
| `minimax-m3-0618-x86_64-cu130` | 2026-06-22 | FI TRT-LLM attention, FP8 indexer KV; default FI TRT-LLM MoE now beats the Marlin override; needs the MSA top-k patch (W5) | — | M3 FP8 B200/B300 | #899 #902 |
| `nightly-aarch64`, `minimax-m3-perf-arm64-…-7a67223` | 2026-06-20 | `head_ratio` fix for the 0618 `gemm1_alpha` crash on GB300 | vllm#45879 | M3 GB300/GB200 | #891 #910 |
| `vllm-minimax-m3-perf-x86_64-…-8b00f41` | 2026-06-25 | modelopt NVFP4 for M3 | vllm#46380 | M3 NVFP4 | #919 |
| `dsv4-megamoe-mxfp4-arm64-cu130-4ba0a72` (custom) | 2026-07-05 | W4A4 MegaMoE [numerics] | — | DSV4 GB300 Dynamo | #969 |
| `nightly-dev-…-cu13.0.1-904e4ec` → `-426e59f` | 2026-07-16 | `FLASHINFER_MLA_SPARSE_DSV4`; `deep_gemm_amxf4_mega_moe` with `VLLM_DSV4_MEGA_FP8_COMBINE=1` [numerics]; V2 model runner; Rust frontend | — | DSV4 B200/B300 AgentX | #1026 #1159 #1194 |
| `nightly-4080263b` | 2026-07-26 | MSA contiguity, MNNVL one-shot and NIXL hetero-TP fixes upstreamed: setup patches go; EAGLE3-GQA drafter | — | M3, all NVIDIA | #1060 #1080 #1088 |
| `nightly-e2fa2859` | 2026-08-07 | `--linear-backend flashinfer_cutlass`; FI MLA prefill with query quantization | — | Kimi-K2.5 NVFP4 | #1129 #1132 |
| Kimi-K3: plugin → `nightly-b22afe45` → `agentx-k3@5894fdf` → `nightly-dev-…-728d3ad` | 2026-07-28 | upstream K3 + DSpark; DCP and DSpark under DCP (branch image only); TP8×PP2×DCP8 | — | Kimi-K3 | #1064 #1155 #1189 #1216 |

About 39 plain bumps (the `v0.20.2`–`v0.25.x` bot series, e.g. #628, #1036) name no capability.
`v0.16.0-cu130` was reverted the same day (#401, #402), and #757 rolled MiniMax-M2.5 H100 back from
`v0.21.0` to `v0.19.1`, reasons not stated.

- **Blocked before:** MegaMoE needs `v0.20.0-cu130` (#559); AMXF4 MegaMoE needs `904e4ec` or the
  custom `4ba0a72` (#1026, #969); M3 needs the `minimax-m3` tag (#840) and its NVFP4 the `8b00f41`
  build (#919); GB300 crashes on 0618 until vllm#45879 (#891); Kimi-K3 DCP exists only in the
  `agentx-k3` branch image (#1189); Kimi-K3 cross-node TP16 hangs in MNNVL init, so TP stays inside a
  node (#1156); DSV4 B300 1k1k graph capture raised "k out of range" in `persistent_topk` on
  `deepseekv4-cu130` (#526), gone in the vendor recipe the same day (#532).
- **Default changes and silent no-ops:**
  - Async scheduling is the default from `v0.14.0` (#365).
  - Under spec decode vLLM rounds capture sizes to multiples of (1+K): give sizes in tokens, or
    graphs cover only MAX_NUM_SEQS/(1+K) sequences (#1082).
  - Under DP attention `--max-num-seqs` is per DP engine (#1194).
  - The Rust frontend ignores `--stream-interval` (#1146).
  - `--moe-backend marlin` (TP-only) won on `minimax-m3` (#869) and lost on 0618 (#899).
  - `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm` is called deprecated in #1055, yet #1114 sets it and
    it is still in `agentic/minimaxm3_fp4_b{200,300}_mtp.sh`; whether it is still read is unverified.
  - The launcher test `[ "$EP_SIZE" -ge 1 ]` ran EP at every MiniMax-M2.5 point (#405) until #448.

### N2. SGLang (`lmsysorg/sglang`)

| Image | First used | What became possible, or what it fixed | Upstream | Recipes | Ids |
| --- | --- | --- | --- | --- | --- |
| `v0.5.0rc1-cu128-b200` | 2025-09-02 | `SGLANG_ENABLE_FLASHINFER_GEMM=true` (FI Blackwell FP8 block-scale GEMM, not in `--help`); FP8 KV [numerics]; shared-experts fusion on | — | DSR1 FP8 B200 | #17 #23 |
| `v0.5.3rc1-cu129-b200` | 2025-09-23 | `--enable-flashinfer-allreduce-fusion`; trtllm_mla with FI trtllm MoE | — | DSR1 FP8 B200 | #160 |
| `v0.5.5-cu129` | 2025-11-10 | Renames: `--enable-ep-moe` → `--ep-size`, `--enable-flashinfer-trtllm-moe` → `--moe-runner-backend flashinfer_trtllm` | — | DSR1 B200/H200 | #224 #229 |
| `v0.5.8-cu130` | 2026-02-05 | `--fp8-gemm-backend=flashinfer_trtllm`; EAGLE MTP | — | DSR1 FP8 B200 MTP | #310 |
| `v0.5.10.post1` | 2026-04-14 | `SGLANG_ENABLE_SPEC_V2=1` (spec decode under the overlap scheduler) works for Qwen3.5 | — | Qwen3.5 H200, DSR1 B300 MTP | #463 #497 |
| `deepseek-v4-blackwell/-b300/-hopper`, then digest-pinned | 2026-04-24 | DSV4 day 0. The digest with DeepGEMM `transform_weights_for_mega_moe` enables `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1`, which needs `FIX_HASH_MEGA_MOE`, `FIX_MEGA_MOE_MEMORY`, `FIX_NEXTN_MEGA_MOE` and `USE_FAST_MASK_EP` together; `-b300` bakes in `CUDA_VISIBLE_DEVICES=4,5,6,7` | — | DSV4 B200/B300/H200 | #523 #539 #542 #547 |
| `nightly-dev-20260422-de962f32` | 2026-04-30 | `--enable-symm-mem`, replacing AR fusion, NVLS, FI GEMM and JIT-DeepGEMM-off, which must all go together | — | Qwen3.5 B200 | #576 #585 #598 #604 |
| `nightly-dev-cu13-20260514`…`0610` | 2026-05-20 | `--moe-a2a-backend megamoe` replaces the env flags; W4A4 MegaMoE (`…_USE_FP4_ACTS=1` + `_USE_MXF4_KIND=1`) [numerics]; `--enable-deepseek-v4-fp4-indexer` [numerics] | — | DSV4 GB300, B300 | #742 #802 #829 #938 |
| `v0.5.12-cu130` | 2026-05-18 | Regressions (below) | sglang#25551 #25564 #25563 | GLM-5, Qwen3.5 B200/B300 | #704 #707 #716 #762 |
| `nightly-dev-cu13-20260605-7dc73766` | 2026-06-12 | JIT DeepGEMM works again, renamed `SGLANG_ENABLE_JIT_DEEPGEMM=1` | sglang#25551 | GLM-5 B200/B300 | #836 |
| `v0.5.12.post1` | 2026-06-23 | Adopted with `SGLANG_RADIX_FORCE_MISS=1`, `--enable-dp-attention-local-control-broadcast`, `--enable-dp-lm-head`, `--enable-prefill-delayer` | — | DSR1 FP4 B200 | #907 #948 |
| `nightly-dev-cu13-20260707-b4155233` | 2026-07-09 | `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1`, `SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS=1` (SWA prefix hits under chunked prefill) | — | DSV4 agentic | #985 |
| `v0.5.16-cu130` | 2026-08-07 | `SGLANG_SIMULATE_ACC_TOKEN_MODE` (synthetic acceptance) | — | Qwen3.5 H100 MTP | #1138 |
| nightly 20260719 | 2026-08-10 | Pinned as the last nightly before a scheduler regression | sglang#31687 | DSV4 GB300 Dynamo | #1150 |
| `nightly-dev-20260818-c0b6474b` | 2026-08-19 | `--mamba-radix-cache-strategy extra_buffer_lazy`; ReplaySSM spec verification (not in `v0.5.17`) | — | Qwen3.5 GB200 | #1209 |
| `dev-nightly-0820`; `sglang-staging:dev-cu13-pr-35880` | 2026-08-22 | `--enable-prefill-delayer --prefill-decode-interval N`: +28% output tok/s at c128 on B300 at 2.2× p50 TTFT [regime]; staging adds a HiCache write_back fix | — | DSV4 B200/B300 AgentX | #1229 #1235 |

About 45 plain bumps (`v0.5.6`–`v0.5.16` series, e.g. #251, #1151).

- **Blocked before:** on `deepseek-v4-b300`, DeepEP's FP8 weight post-processing broke the
  checkpoint, so DP attention stayed off (#533) until the digest-pinned images (#537, #547);
  `v0.5.12` breaks JIT DeepGEMM in graph capture on B300/B200 (#707, #716) until nightly 20260605
  (#836), trips a flash-attn cute assertion in vision attention on B300 sm_103 (#704), and crashes in
  trtllm batched GEMM at bs 128 with EAGLE on B300, where a FlashInfer downgrade was tried and
  reverted (#762, #763); nightly 20260519 crashed in `topk_v2`, fixed in 20260520 (sglang#25805,
  #754); HiCache write_back at DEP8 c384/512 killed all eight schedulers with zero client errors
  until pr-35880 (#1235).
- **Default changes and silent no-ops:**
  - `SGLANG_ENABLE_SPEC_V2=1` is assigned without `export` in `dsr1_fp8_b200_mtp.sh`, so it never
    reaches the server; still so on main (#310).
  - The `--ep-size` rename made DSR1 FP8 B200 run EP8 (#224) until EP1 was restored (#229).
  - `SGL_ENABLE_JIT_DEEPGEMM` became `SGLANG_ENABLE_JIT_DEEPGEMM` (#836); main still sets the old name
    in the Qwen3.5 GB200/GB300 agentic recipes. Check which name the image reads.
  - `--disable-radix-cache` takes a slower cache path than `SGLANG_RADIX_FORCE_MISS=1` (#907).
  - Under DP attention `--chunked-prefill-size` is divided by dp_size (#1229), as on AMD (A8).
  - MegaMoE's ~7 GB workspace sits outside `--mem-fraction-static`; an OOM on one rank hangs the
    engine (#1235).
  - From nightly 20260519 several `SGLANG_OPT_*` flags are deleted or defaulted (JIT_NORM, TOPK_V2,
    CUSTOM_ALL_REDUCE_V2, the MegaMoE FIX_* flags) (#742, #802, #806, #938).
  - `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` must be ≥ ceil(cuda_graph_max_bs / dp_size)
    (#984).

### N3. TRT-LLM (`nvcr.io/nvidia/tensorrt-llm/release`) and Dynamo

| Image | First used | What became possible, or what it fixed | Recipes | Ids |
| --- | --- | --- | --- | --- |
| TRT-LLM `1.2.0rc0.post1` | 2025-10-20 | TRTLLM MoE under DP-attention+EP (CUTLASS fallback before); FP8 KV [numerics] | GPT-OSS B200 | #210 |
| TRT-LLM `1.2.0rc2` | 2025-12-02 | MNNVL throughput all-to-all (`TRTLLM_MOE_ALLTOALL_BACKEND=mnnvlthroughput`, `TRTLLM_FORCE_ALLTOALL_METHOD=MNNVL`) makes DP attention pay | GPT-OSS B200 | #234 |
| TRT-LLM `1.2.0rc6.post2/.post3` | 2026-01-29 | Piecewise graphs via `torch_compile_config.capture_num_tokens`; TRTLLM MoE beats DEEPGEMM at small and mid batch; `ENABLE_CONFIGURABLE_MOE=1` | DSR1 B200 | #297 #303 #312 |
| TRT-LLM `1.3.0rc11` | 2026-04-13 | Fixes `executor/result.py` (the sed goes); `enable_block_reuse` defaults to true | GPT-OSS H200 | #461 |
| `trtllm-deepseek-v4:feat-deepseek_v4-4999884` → `9aa3715` → `c185066` | 2026-05-02 | DSV4 day 0; `9aa3715` fixes fused MHC at hidden 7168 (W5); `c185066`: `MEGAMOE_DEEPGEMM` at ISL ≤ 1k and conc ≥ 2048 (≥ 512 with MTP), `use_low_precision_moe_combine` [numerics], GC off, `MIMALLOC_PURGE_DELAY=0`, `batching_wait_iters 30` | DSV4 B200/B300 | #592 #607 #830 #841 |
| Dynamo `tensorrtllm-runtime 0.5.1` → `0.8.x` → `1.3.0-*` | 2025-09-20 | GB200 NVL72 disagg; srt-slurm recipes, DEP32 decode with EPLB; DSV4 MXFP4 disagg; NIXL replaces UCX (`1.3.0rc24`) | DSR1, Kimi-K2.5, DSV4, Qwen3.5 | #153 #290 #318 #855 #1252 |
| Dynamo over vLLM (`v0.18.0-cu130` … `v0.25.1`) | 2026-04-07 | NixlConnector; MegaMoE disagg needs `TORCH_SYMMMEM=NVSHMEM` (then no CPU expert offload); `enable_multiple_frontends` | Kimi-K2.5, DSV4 GB200 | #451 #566 #1094 |
| Dynamo `1.3.0.dev20260718` | 2026-08-10 | Router queueing off at high conc; `X-Dynamo-Session-ID` affinity | DSV4 GB300 AgentX | #1150 |

- **Blocked before:** fused MHC in `4999884` corrupts DSV4 output, fast but wrong, so only an
  accuracy check catches it (#592, #607); B300 graph warmup at decode batch ≥ 1088 is a
  use-after-free (TensorRT-LLM#15265, #830); Kimi-K3 day 0 under Dynamo 404'd every request, so it
  runs direct multi-node `vllm serve` (#1075); DeepGEMM NVLink barriers time out at c4096 (#1147).
- **Defaults:** TRT-LLM's delay-batching knobs (`batch_wait_timeout_iters`,
  `batch_wait_max_tokens_ratio`) default to 0 and exist only in the LLM-API YAML (#124).
- **Library pins named in records:** NCCL `2.26.2.post1` (install `--no-deps`) for FI AR fusion on
  B200 (#15, #115); FlashInfer `97201824` → 0.3.0/0.3.1 against a cubin race (#24, #180–#196);
  DeepGEMM inside vLLM `v0.20.0-cu130` (#559); Mooncake 0.3.11.post1 with `PYTHONHASHSEED` fixed so
  hashes match across ranks (#1026).

## W. Workaround-flag inventory

**Rule: re-test every workaround on each image change; removing a stale workaround is a candidate
optimization.** The tree does not record why a disable was added; this table does. For each row:
confirm the bug is gone in the current image (rerun the failure, or read the fix PR), then A/B the
removal at two operating points, with the accuracy gate when the row is [numerics]. "Still" means the setting is present at origin/main (`deprecated/`
scripts are kept but no longer swept).

### W1. AMD: vLLM ROCm

| Flag or setting | Disables or works around | Why (image, bug) | Added | Removed | Removing it bought; status |
| --- | --- | --- | --- | --- | --- |
| `VLLM_ROCM_USE_AITER` left unset | every AITER path | AITER MLA rejected 8 heads per rank (vllm#35641) | #374 (v0.16.0) | #425–#427 (v0.18.0) | AITER MLA, MoE, GEMM on; INT4 QR and TP4 layouts followed |
| `--moe-backend triton_unfused` | AITER MXFP4 MoE | FP4 scale params not registered: KeyError at load | #555 #731 | #955 #957 (`nightly-09663abd`) | fused AITER experts; split-KV decode came with the same image |
| `--enforce-eager` (DSV4) | graphs | DSV4 capture not ready on ROCm | #555 #731 | #756 (`nightly-4f940896`) | FULL_AND_PIECEWISE graphs; conc to 512 |
| `--enforce-eager` (MiniMax-M3) | graphs | decode CUDA-graph assertion, `minimax-m3` image | #844 #858 | #849 #851–#853 | graphs on, via the next row |
| `VLLM_USE_BREAKABLE_CUDAGRAPH=0` | breakable cudagraphs | M3 decode assertion in the breakable path | #849 #854 #917 | — | still, in 13 files; re-test per image |
| `VLLM_ROCM_USE_AITER=0` in one M3 layout [numerics] | AITER master switch | degenerate output (layout disputed, A7) | #935 | #972 #973 (vllm#47158) | AITER MoE for TP-only; still in deprecated `minimaxm3_fp8_mi355x_mtp.sh` (EP branch) |
| `--moe-backend aiter` forced with EP | auto selection | auto falls to EMULATION, JIT-lock deadlock (~8 h) | #939 | — | keep while auto can pick EMULATION |
| `--moe-backend` left auto (gfx942) | forcing aiter | forced aiter picks W4A4, which gfx942 rejects | #1011 | — | hardware, not image |
| `VLLM_ROCM_USE_AITER_MHA=0` [regime] | AITER CK MHA | loses on short prompts and low conc; image default flipped | #152 #162 #204 #209 | for gpt-oss, by the `ROCM_AITER_UNIFIED_ATTN` backend (#396) | still in `models_vllm.yaml` (gpt-oss) |
| `VLLM_ROCM_USE_AITER_MHA=0` (M3, MI300X) [numerics] | AITER attention | no calibrated FP8 q/prob scales on gfx942 | #982 | — | still |
| BF16 KV on gfx942 M3 [numerics] | FP8 KV | fallback scale 1.0 corrupts accuracy | #846 #854 | tried #875, reverted #886; the MTP twin kept FP8 KV (#883): recipes disagree | |
| `VLLM_ROCM_USE_AITER_RMSNORM=0` at TP < 8 [numerics] | AITER RMSNorm | accuracy failure, Kimi-K2.5 MXFP4 on v0.18.0 | #425 | — | still in deprecated `kimik2.5_fp4_mi355x.sh` |
| `--block-size 1` | larger pages | AITER MLA decode built for page 1 | #425 | #1096 (block 16 with fp8 KV) | |
| `VLLM_ROCM_USE_AITER_TRITON_BF16_GEMM=0` [regime] | AITER Triton BF16 GEMM | slower at low conc on the 2025-09 images | #152 #167 #199 | gpt-oss #164 #286 | still in `models_vllm.yaml` (gpt-oss) |
| `ROCM_TRITON_MOE_PRESHUFFLE_SCALES=0`, `NCCL_MIN_NCHANNELS=112` | preshuffled MoE scales; RCCL channel count | 0915 image | #152 | #164 #198 (0927 image) | NCCL setting re-added for M3 MI300X (#982) |
| `VLLM_USE_AITER_UNIFIED_ATTENTION=1` | nothing: misnamed, a silent no-op | ignored by v0.14 (#286); #367 says it was never on, even on the `rocm/7.0` image | #59 #63 #156 | #286 #367 | the feature turned on for the first time |
| `VLLM_ROCM_USE_AITER_MOE=0` at TP ≥ 4 (commented out) | AITER CK MoE | missing N = 384 tiles (vllm#35637) | #517 (dormant) | — | |
| `--max-num-batched-tokens 4K` | 8K-token steps | AITER MoE kernel out-of-bounds write at M = 8K | #770 | — | still in `models_vllm.yaml` |
| `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` only for some shapes [regime] | ASM paged attention elsewhere | `pa_fwd_asm` crashes on shapes missing from the gfx942 table | #794 | — | recipe note: collapse to always-on once AITER registers the kernel |
| `fuse_minimax_qk_norm` omitted | QK-norm fusion pass | NameError on ROCm (pass imported under `is_cuda()`) | #794 | — | |
| image pinned to `v0.20.2` | newer image | `v0.21.0` graph-capture crash | #794 | #792 (`v0.22.0`) | |
| `--attention-backend ROCM_AITER_FA` pinned | the new default `ROCM_ATTN` | vllm#36702 default flip | #794 | — | recovers about 38% on MI325X |
| `pass_config.fuse_allreduce_rms=false` | AR+RMSNorm fusion | with `--moe-backend flydsl` (reason not stated) | #867 | — | still in deprecated `kimik2.5_int4_mi355x.sh` |
| `VLLM_ROCM_USE_SKINNY_GEMM=0` | skinny GEMM | AITER BF16 GEMM preferred with the new MoE backend | #1096 | — | still (deprecated Kimi FP4) |
| `VLLM_ROCM_USE_AITER_PAGED_ATTN=0`, `VLLM_USE_AITER_TRITON_SILU_MUL=0`, `cudagraph_mode PIECEWISE` | AITER paged attention, AITER SiLU-mul, full graphs | not stated (Kimi disagg on `nightly-bf610c2f`) | #769 | #1051 (`nightly-2afa3f7e`) | synced with the single-node recipe, full graphs |
| `VLLM_ROCM_USE_AITER_MOE=0` (8k1k TP8/EP8), `--no-async-scheduling` (small batches) [regime] | AITER MoE; async scheduling | measured per point (MiniMax-M2.5 FP8) | #611 | — | still (deprecated) |
| `AMDGCN_USE_BUFFER_OPS=0` | Triton buffer-load codegen | slower for gpt-oss kernels on v0.17.0 | #396 | — | still (deprecated gpt-oss); other recipes set `=1` (#69) |
| drafter `"attention_backend": "TRITON_ATTN"` in `--speculative-config` | the drafter's slow default backend | global `--attention-backend` does not reach the drafter | #977 | — | |
| runtime source patches | — | `SupportsEagle3` (#848, #850); MoRI-IO (#769); Kimi-K3's nine PRs (#1171) | | #883 #923; #987; — | check each against the next nightly |

### W2. AMD: SGLang ROCm

| Flag or setting | Disables or works around | Why (image, bug) | Added | Removed | Removing it bought; status |
| --- | --- | --- | --- | --- | --- |
| DSV4 day-0 set: `SGLANG_HACK_FLASHMLA_BACKEND=torch`, `SGLANG_FORCE_TRITON_MOE_FP8=1`, FP4 experts off, fused compress/top-k/MHC off, torch top-k transform and paged MQA logits, `--disable-cuda-graph` | about 20 optimized paths | correctness on `deepseek-v4-mi35x` | #524 | in steps: #593 (FP4 experts via AITER), #605 (fused compress, TileLang MHC post, native top-k), #617 (AITER MHC), #741 (fused kernels), #719 (recipe dropped), #995 #998 (now defaults) | each step re-enabled fused kernels |
| deep_gemm-absence set: `SGLANG_ENABLE_JIT_DEEPGEMM=0`, `SGLANG_OPT_FP8_WO_A_GEMM=0` (wo_a dequantized to BF16), `SGLANG_TOPK_TRANSFORM_512_TORCH=1`, `SGLANG_OPT_USE_TOPK_V2=0` | FP8 `wo_a` GEMM, native top-k | deep_gemm missing on ROCm; TopK v2's JIT kernel is CUDA-only | #798 | #998 (`v0.5.14-20260708`) | `SGLANG_OPT_FP8_WO_A_GEMM=false` still in disagg `env.sh`; `TOPK_V2=false` still in GLM-5.2 agentic (#1158) |
| `config.json` `model_type` deepseek_v4 → deepseek_v3 | — | transformers could not load deepseek_v4 | #524 #593 #798 | #831 #998 | |
| transformers from git (`glm_moe_dsa`) | — | image lacked GLM-5 support | #392 | #582 #653 (`v0.5.11`) | |
| `--disable-shared-experts-fusion` (DSV4) | shared-expert fusion | fused shape not in the tuned tables | #524 #1181 | #1232 (`v0.5.18`, `--enforce-shared-experts-fusion`) | still in fixed-seq-len DSV4 (`v0.5.14`) and disagg `models.yaml`: a candidate |
| `--enable-torch-compile` dropped | torch.compile | no decode gain on graphs + AITER; 30 min compile per start | #255 | #301 | perf 1.00–1.03×, jobs 16–45 min shorter |
| `RCCL_MSCCL_ENABLE=0` | RCCL MSCCL algorithms | not stated | #236 #255 | — | still (`dsr1_fp8_mi355x*.sh`): re-test |
| `SGLANG_ROCM_FUSED_DECODE_MLA=0` | fused ROCm decode MLA | GLM-5 uses NSA/DSA, not MLA | #392 #503 #767 | — | still (`env.sh`, deprecated GLM-5) |
| EP8 removed from DSV4 DP-attention rows [numerics] | EP | GSM8K 0.93 → 0.00 once EP was actually applied | #595 | — | |
| `SGLANG_MORI_FP8_COMB=true` → `MORI_COMBINE_DTYPE_PREFILL=fp8_direct_cast`, `MORI_COMBINE_DTYPE_DECODE=fp8` [numerics] | BF16 combine | FP8 combine failed GSM8K at high conc | #616 | #766 | keeps FP8 bandwidth with accuracy |
| `SGLANG_MORI_FP8_DISP=False` (MXFP4) | FP8 dispatch | no re-quantizing FP4 activations | #406 | #616 (FP4 dispatch) | |
| `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=0` | overlapped plan stream | not stated | #766 | — | still, global in `env.sh` |
| `PYTHONPATH=/sgl-workspace/aiter` | — | "WA for latest upstream 0305 image" | #406 | — | still (`env.sh`) |
| `TORCH_NCCL_BLOCKING_WAIT=1` | async NCCL watchdog | ROCm ≤ 7.2.0 HIP capture bug (A6) | #1154 | — | remove once the base is ROCm ≥ 7.2.2 |
| `mori_conn.py` overlay | — | MoRI PD startup crash, hybrid models | #767 | — | still |
| DSV4 PD: `SGLANG_AITER_MLA_PERSIST=0`, `HSA_NO_SCRATCH_RECLAIM=0`, `SGLANG_OPT_*`/AITER routing | AITER CK fused-MoE path | "Unsupported kernel config for moe heuristic dispatch" | #999 #1041 | — | still (`env.sh`) |
| DP attention dormant for DSA models | DP attention | collective hang, long-context prefill | #1158 #1181 | — | open |
| AITER all-reduce fusion off at TP2/EP2 EAGLE | AR fusion | rank consistency unstable | #1168 | — | |
| `SGLANG_USE_ROCM700A=0` | not stated | not stated | #524 | — | still in every DSV4 recipe: re-test |

### W3. AMD: ATOM

| Flag or setting | Disables or works around | Why (image, bug) | Added | Removed | Removing it bought; status |
| --- | --- | --- | --- | --- | --- |
| DSV4 day-0 set: CONC==1 and EP==1 guards, `--enforce-eager`, `--block-size 16`, ATOM#650 overlay, `triton_kernels` pin (image deleted it), `ATOM_USE_TRITON_MOE=1`, `mhc_pre` sed-patched to torch | graphs, batching, AITER MoE, AITER MHC | AITER `fused_moe` broken on gfx950 for a16w4+SwiGLU; `mhc_pre` HIP device-guard crash; single-request sparse attention | #551 | #560 (AITER MXFP4 MoE, aiter#2916), #652 (the rest) | conc 1 → 512 with graphs |
| `--max-num-seqs max(CONC,4)` | default 512 | GDN-mamba KV overshoot; 1 hangs warmup | #550 | #652 | |
| `ATOM_DSV4_SPARSE_ATTN_CHUNK_TOKENS=256` + decode fast-path patch | torch `sparse_attn` fallback | OOM and large temporaries | #560 | #652 | |
| default KV dtype (MiniMax-M3) | FP8 KV | `k_scale` assertion | #874 | #926 | |
| `ATOM_M3_SPARSE_USE_ASM_PA=1`, `AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=0` | Triton sparse attention | ASM PA won on the 0623 image | #926 #927 | #941 #952 (`ATOM_FORCE_ATTN_TRITON=1`) | Triton faster on the 0630 image; names deprecated |
| prefill-only TBO with MTP | decode TBO | decode TBO drops `spec_decode_metadata` | #1067 | — | |
| `--max-num-seqs CONC` on DP-attention cells | default | a full KV pool per rank OOMs | #1068 | — | |
| `--hf-overrides use_index_cache` removed [numerics] | — | policy: overrides that cut model FLOPs are disallowed | #877 #941 | #959 #945 #952 | not a bug; a benchmark rule |
| `OFFLOAD_GPU_STAGING_CHUNKS=16`; `LMCACHE_CHUNK_SIZE=1024` under DCP 8; LMCache tier halved | — | state tier silently declines; offload hashes misalign; NCCL's 600 s timeout during pinning | #1214 #1254 | — | |
| `AITER_BF16_FP8_MOE_BOUND=0` | not stated | not stated | #652 #771 | — | gated to DSV4 (#918) |

### W4. AMD: ROCm runtime

| Flag or setting | Works around | Added | Removed | Status |
| --- | --- | --- | --- | --- |
| `HSA_NO_SCRATCH_RECLAIM=1` if MEC firmware < 177 | RCCL with the ROCm 6.4.3 driver change | #182 #367 #425 #1051 | — | still, in 8 files; a firmware fact about the host |
| `TORCH_NCCL_BLOCKING_WAIT=1` | ROCm ≤ 7.2.0 graph-capture bug | #1154 | — | see W2 |

`GPU_MAX_HW_QUEUES=5` is an enabler, not a workaround: without it ROCm runs the two TBO streams
serially and TBO shows no gain (#877, #995, #1067).

### W5. NVIDIA

| Flag / env | Disables or works around | Motivating image / bug | Added | Removed | What removing it bought |
| --- | --- | --- | --- | --- | --- |
| `VLLM_USE_FLASHINFER_MOE_FP8=0`, `VLLM_MOE_USE_DEEP_GEMM=0` | FI and DeepGEMM FP8 MoE | MiniMax-M2.5 B200 v0.17.0. Records disagree: #405 calls it a speed choice, #448 a bug workaround. | #405 | #448 (v0.19.0) | Fast MoE, bundled with other changes; no separate number |
| `VLLM_USE_DEEP_GEMM=0`, `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0` | DeepGEMM, FI block-FP8 linears | Speed choice, H200 v0.20.1 | #765 | Still on main (deprecated script) | — |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` | Graph-memory pre-reservation | v0.20.1+ startup failure | #600 #680 #1097 | Still on main (3 files) | — |
| `cudagraph_mode: PIECEWISE` | FULL_AND_PIECEWISE default | v0.11.0, H100/H200 | #217 | #237 (v0.11.2) | Full graphs |
| sed bypass of `persistent_topk` | Persistent top-k kernel | "k out of range" in capture, DSv4 B300, deepseekv4-cu130 | #526 | #532 | The patch's own comment called its cost "small decode-time" |
| MiniMax-M3 setup patches (MSA `.contiguous()`, FI `force_oneshot_support`, NIXL hetero-TP, srt-slurm#38) | 0618 bugs; one-shot MNNVL AR capture across GB300 trays | minimax-m3-0618 / perf-arm64 | #899 #903 #910 #933 | #1060 #1080 #1081 #1095 (nightly-4080263b) | — |
| DeepGEMM NVLink barrier patch (`vllm-container-deps.sh`) | Barrier timeouts | DSv4 GB200 Dynamo-vLLM | Not in records | #1094 (v0.25.1) | — |
| CPU expert offload + `--enforce-eager` on prefill | Prefill graphs; memory (KV is −16 GiB without offload) | DSv4 GB200, deepseekv4-cu130 | #525 #563 | Offload: #566 (MegaMoE, prefill GMU 0.95). Eager still on main. | Offload: MegaMoE pays only without it |
| `fuse_allreduce_rms: false` | AR+RMSNorm fusion | Kimi-K3 H200 ("upstream guidance"), DSv4 GB200 | #525 #1127 | Still on main | — |
| `--no-enable-flashinfer-autotune` / `--disable-flashinfer-autotune` | FI autotune | Decode accuracy stability (#569); DSv4 cookbook | #523 #569 #1229 | Still on main | — |
| `VLLM_USE_V2_MODEL_RUNNER=0` | V2 runner | nightly-e2fa2859; reason not stated | #1129 | Still on main (deprecated scripts) | — |
| `VLLM_USE_DIRECT_DCP_KV_GATHER: "0"` | Direct DCP gather ("keeps measured fallback") | Kimi-K3 GB200 DCP16 | #1212 | Still on main; the B200 PP2 recipes set "1" | — |
| `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=900` | Keep-alive races that aborted a replay | Kimi-K3 agentic | #1064 | Still on main | — |
| `--disable-shared-experts-fusion` | Shared-expert fusion | DSR1 B200 v0.4.10.post1; reason not stated | pre-#17 | #17/#23 (#22 reverted once) | One grouped GEMM instead of two. Still on main in GLM-5 recipes, and DSv4 v0.5.17 turns it off (#1173). |
| `SGL_ENABLE_JIT_DEEPGEMM=0` (B300) / `--fp8-gemm-backend cutlass` (B200) | JIT DeepGEMM | v0.5.12 TMA `CUDA_ERROR_ILLEGAL_ADDRESS` in capture (sglang#25551) | #707 #716 #749 | #836 | DeepGEMM back |
| `SGL(ANG)_ENABLE_JIT_DEEPGEMM=false` | DeepGEMM, in favour of FI GEMM (a choice) | DSR1/Qwen3.5 B200 | #160 #310 #452 | Qwen3.5: #576 #598 (with `--enable-symm-mem`). Still on main in `dsr1_fp8_b200_mtp.sh` and the Qwen3.5 agentic recipes. | Part of a coupled reset |
| `--mm-attention-backend triton_attn` | flash-attn cute vision attention | sm_103 assertion, v0.5.12 B300 (sglang#25564) | #704 #751 | Still on main (6 files) | — |
| `SGLANG_OPT_FP8_WO_A_GEMM=0` | FP8 wo_a GEMM | topk_v2 crash, nightly 20260519 | #742 | #754 (sglang#25805) | FP8 GEMM back |
| `--disable-radix-cache` | Prefix reuse | Slower cache path | — | → `SGLANG_RADIX_FORCE_MISS=1` (#907 #948) | Faster radix path |
| `--disable-piecewise-cuda-graph` | Piecewise graphs | DSR1 FP4 B200; reason not stated | #907 #948 | Still on main | — |
| `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0` | DeepGEMM precompile | DSv4 day 0 | #523 #533 | → `SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1` (#829); dropped (#938). Still on main in 3 DSv4 files. | — |
| DSv4 B300 low-latency only (TP8, no DeepEP) | DP-attn + DeepEP | DeepEP FP8 weight post-processing broken | #533 | #537 #547 | High-concurrency recipes |
| `SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH=1` | Draft-extend graph | DSv4 EAGLE capture >1 h at TP4/bs128 | #1150 | Still on main | — |
| `--disable-custom-all-reduce`, `NCCL_IB_DISABLE=1` | Custom AR; IB probing | PCIe RTX PRO 6000; NCCL 2.28.9 bnxt_re segfault | #1076 | Still on main (18 files; other reasons not in records) | — |
| `NCCL_GRAPH_REGISTER=0` | NCCL graph buffer registration ("can reduce perf") | CI stability; TRT 1.2.0rc0.post1 | #200 #210 | #234 (1.2.0rc2) | Registered collectives |
| `TRTLLM_MHC_ENABLE_FUSED_HC=0` [numerics] | Fused MHC kernel | Corrupts DSv4 hidden-7168 generations (4999884) | #592 #602 | #607 (9aa3715) | Fused MHC |
| `NCCL_NVLS_ENABLE` default 0 | NVLS | DSv4 TRT; reason not stated | #592 #602 | Still on main (4 files) | — |
| sed `417d` on `tensorrt_llm/executor/result.py` | TRT-LLM bug (evals) | gpt-oss-dev | #294 | #461 (1.3.0rc11) | — |
| `TLLM_OVERRIDE_LAYER_NUM=61` | Loading the MTP layer (OOM) | DSR1 FP8 B200 TRT | #297 | Still on main | — |
| CUDA-graph `max_batch_size` ≤1024 | Graphs above 1024 (run eager) | `TRTLLM_MLA_EXTRA_OVERLAP` use-after-free from batch 1088, B300 (TensorRT-LLM#15265) | #830 | Still on main | — |
| `TRTLLM_FORCE_ALLTOALL_METHOD=NotEnabled` | a2a at EP=1 | GPT-OSS B200 1.2.0rc2.post1 | #270 | #271; still in a deprecated script on main | — |
| sed of `flashinfer/jit/cubin_loader.py` 102–108 | Cubin race (~1% of runs; flashinfer PR 1779) | B200 | #173 | #284 | — |
| `DYN_ROUTER_TEMPERATURE=10000000` | KV-aware routing (near-random) | Dynamo AgentX | #1016 #1140 | #1150 (temperature 0 + affinity). Still on main in `glm5.2/agentic/disagg-h200-2p2d-*.yaml`. | Cache hits on later turns |
