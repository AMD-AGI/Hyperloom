# AMD serving knobs: vLLM-ROCm, SGLang-ROCm, ATOM, AITER, MoRI, RCCL, HSA, Triton-AMD

A catalog of the AMD-side knobs (environment variables and ROCm-relevant CLI flags) that the
InferenceX MI300X/MI325X/MI355X recipes use: what each does, when it paid, what it needs, what
breaks it, and whether it changes numerics.

Built 2026-10-20 from the runner's docs/research/inferencex-study (records.jsonl ids) and
InferenceX origin/main.

**How to read it:**
- Read §0 (Rules) first. Then go to the layer you are touching: §1 vLLM-ROCm, §2 SGLang-ROCm,
  §3 ATOM, §4 AITER, §5 quick-reduce (all frameworks), §6 MoRI, §7 RCCL/collectives, §8 HSA/HIP
  runtime, §9 Triton-AMD compiler, §10 load-time and plumbing.
- `#655` is a record id in `evidence/records.jsonl`. `main` means the setting is in InferenceX origin/main
  (6d6d296c6, 2026-08-30), including the `deprecated/` scripts still in the tree; the file is named
  when it matters. Short hex strings are InferenceX commits.
- Marks:
  - **NUM**: changes model numerics or outputs. Gate it on accuracy before a KEEP; see
    `numerics-knobs.md`.
  - **OP**: the best value depends on the operating point; the regime where it held is given.
  - **WA**: a workaround for a crash, hang or wrong result, not a speedup. Re-test removing it after
    every image change.
- GPUs: gfx942 = MI300X/MI325X (CDNA3, no FP4 matrix instructions: #114, #1011); gfx950 = MI355X
  (CDNA4). A fact measured on one is not a fact about the other.
- The study counted 101 distinct AMD-side names in the records; the ones that only name ports,
  paths, timeouts or logs are grouped in §10.

## 0. Rules

1. **Prove the knob reached the server process.** The recipes lost settings on the way several
   times:
   - a per-workload `VLLM_ROCM_USE_AITER_MHA` branch tested lowercase `$isl`/`$osl`, which the
     container never received, so it never fired (#152, fixed by #162);
   - an SGLang variable was assigned but not exported (#310, NVIDIA, same mechanism);
   - a launcher silently dropped `--ep-size`, so a DSv4 recipe labelled EP8 ran EP1 (#595).

   Check the live process, not the script:
   `tr '\0' '\n' < /proc/<pid>/environ | grep -E '^(VLLM|AITER|SGLANG|ATOM|HSA|GPU_MAX|MORI|ROCM|TRITON|AMDGCN|NCCL|RCCL)_'`.
   vLLM's engine core and workers are child processes; check one of those. The DSv4 SGLang agentic
   recipe prints `env | grep ^SGLANG_` at launch for this reason (main,
   `agentic/dsv4_fp4_mi355x_sglang_mtp.sh`).
2. **Prove it took effect.** A knob can reach the process and still select nothing:
   - the backend-selection line in the server log names the attention/MoE/linear backend;
   - a profile names the kernels (AITER `fmoe`/CK/ASM such as `pa_fwd_asm`, FlyDSL, Triton);
   - a size gate can keep a feature off for your tensors: INT4 quick-reduce never fired for
     decode all-reduces until its minimum size was lowered (#1089).
3. **Unknown or renamed names are silently ignored.**
   - `VLLM_USE_AITER_UNIFIED_ATTENTION` is ignored from vLLM v0.14; the right name is
     `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION` plus the master switch (#286). On gpt-oss the
     misnamed variable meant the feature had never been on (#367). The multi-node
     `models_vllm.yaml` gpt-oss entry still carries the old name (main).
   - The Kimi-K3 vLLM recipe's header names `AITER_SITUV2_A8W4=1`, while its body exports
     `VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1` (main, `agentic/kimik3_fp4_mi355x_mtp.sh`). Grep the
     image for the name before trusting either.
   - A no-op reads exactly like "no effect", and gets written down as a dead end (REPORT §2.6).
4. **Re-check after every image change.** Defaults and winners move with the image:
   - the 2025-09-27 ROCm image turned AITER paths on by default and made the old env block
     redundant or slower (#182, #186);
   - the same image changed the default of `VLLM_ROCM_USE_AITER_MHA`, so the recipes pinned it
     explicitly (#199, #204);
   - vLLM PR #36702 flipped the ROCm dense-attention default from `ROCM_AITER_FA` to `ROCM_ATTN`,
     a ~38% MI325X regression until `--attention-backend ROCM_AITER_FA` was pinned (#794);
   - AITER's ASM sparse paged attention won on one ATOM image and lost to Triton the next week
     (#926 -> #941/#952);
   - SGLang v0.5.14 made 15 DSv4 `SGLANG_OPT_*`/AITER overrides default, so they were deleted (#995, #998).

   A kill recorded on one image is stale on the next.
5. **Sub-switches need the master switch.** In vLLM, `VLLM_ROCM_USE_AITER=1` gates the
   `VLLM_ROCM_USE_AITER_*` family:
   - #286 restored the master with the renamed variable;
   - on #982 the master was unset, so GEMMs and the fused MoE ran on generic kernels.

   Exception: `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` reads the environment directly
   (c512990ce, #935).
6. **Bundles that only pay together.** Each part alone is a no-op, a crash or under 1%. Claim the
   bundle as one change.

| Bundle | Parts | Evidence |
| --- | --- | --- |
| INT4 quick-reduce for decode | `*_QUICK_REDUCE_QUANTIZATION=INT4` + `*_MIN_SIZE_KB=256` (+ `*_CAST_BF16_TO_FP16=0`) | #1089, #1073, #1165 |
| Two-batch overlap on ROCm | `--enable-tbo` (ATOM) or `--enable-two-batch-overlap` (SGLang) + `GPU_MAX_HW_QUEUES=5` | #877, #995, #1067 |
| SGLang DP-attention on ROCm | `--enable-dp-attention` + `SGLANG_DP_USE_GATHERV=1` + `SGLANG_DP_USE_REDUCE_SCATTER=1` + `SGLANG_SHARED_EXPERT_TP1=1` + `SGLANG_DP_SHARED_EXPERT_LOCAL=1` + `GPU_MAX_HW_QUEUES=5` | #901, #995; main |
| AITER sparse paged attention (MiniMax-M3) | `VLLM_ROCM_USE_AITER=1` + `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` + fp8 KV + exactly one KV head per rank (TP4) | #1050, #1165 |
| AITER paged attention (MiniMax-M2.5) | `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` + `--block-size 16` | #611 |
| AITER unified attention (vLLM) | unified attention on + `VLLM_ROCM_USE_AITER_MHA=0` + `--block-size 64` | #156, #79 |
| AITER unified attention (SGLang) | `SGLANG_USE_AITER_UNIFIED_ATTN=1` + `--attention-backend aiter` + `--page-size 16` | #819, #827 |
| AITER persistent MLA (SGLang) | `SGLANG_AITER_MLA_PERSIST=1` + `--attention-backend aiter` + `--kv-cache-dtype fp8_e4m3` | #279 |
| AITER MoE (vLLM, newer nightlies) | `VLLM_ROCM_USE_AITER=1` + `VLLM_ROCM_USE_AITER_MOE=1` + `--moe-backend aiter` | #934, #955 |
| GLM-5.2 fusions (SGLang) | `--enable-aiter-allreduce-fusion` + `--enable-fused-qk-norm-rope` + `AITER_USE_FLYDSL_MOE_SORTING=1` | #0, #1255 |

7. **Several knobs are policies, not values (OP).** The recipes switch these by concurrency or
   input length; one measured point finds one row of the table:
   - AITER MHA (#209);
   - GEMM path (#167, #199);
   - shuffled KV (#611, #794, #1050);
   - linear backend (#1050);
   - TBO threshold (#1067, #1068).

   Check a second operating point before killing one.

## 1. vLLM on ROCm

### 1.1 Master switch, MoE, linear

| Knob | What it does | When it paid (regime) | Needs, conflicts | Image notes | NUM | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| `VLLM_ROCM_USE_AITER=1` | Master switch: AITER kernels for MoE, GEMM, norms, attention, MLA. | Set in almost every vLLM MI3xx recipe at main. MiniMax-M3 MXFP8 MI300X, master previously unset: +4.9% to +17.5% tok/s/GPU at c4-c256 1k1k, GSM8K 0.959 held (#982). | Kimi-K2.5 TP8 (8 heads per GPU): AITER MLA did not support it until vLLM v0.18.0, so the switch stayed off (#374 -> #425-427). | 2025-09-27 image turned AITER on by default (#182, #186); recent upstream images need it explicitly. | **NUM, case by case**: TP-only MiniMax-M3 gave degenerate output on nightly-4559c43a, so the switch was gated on EP (#935). vllm#47158 (nightly-09663abd) fixed it: GSM8K 0.9613/0.9621 with AITER on (#972). The M3 MTP script's comment then names EP as the degenerate case (17046f446, #973); its rows are TP-only, so that claim is unverified. | #982, #935, #972 |
| `VLLM_ROCM_USE_AITER_MOE` | AITER fused MoE. | DSv4-Pro FP4 MI355X: the AITER MXFP4 path registers the FP4 scale params, so decode runs fused AITER experts instead of `triton_unfused` (#955, #957; main). | On newer nightlies, pair with `--moe-backend aiter`. Off for MiniMax-M2.5 8k1k TP8/EP8, where AITER MoE lost (#611, OP). | **WA** notes: a commented-out block in the MiniMax-M2.5 FP4 recipe says AITER CK fused MoE lacked tiles for N=384 at TP4 (vllm#35637), with Triton MoE as the fallback (#517). A gfx950 `fmoe_bf16_blockscaleFp8` kernel writes out of bounds at M=8K, so the batched-token budget is capped at 4K instead of turning AITER MoE off (main, `models_vllm.yaml`). | no | #955, #611, #517 |
| `--moe-backend aiter` | Forces the AITER MoE backend. | MiniMax-M3 MXFP4 (#934). MiniMax-M3 MXFP8 TP-only (#972). DSv4 FP4 (#955). | Required with EP on M3 MXFP4: with `auto`, EP skipped AITER and picked the EMULATION backend, which JIT-builds a Quark C++ extension on every worker; the workers deadlocked on the build lock for about 8 h (**WA**, #939). | gfx942: forcing `aiter` selects `AITER_MXFP4_MXFP4` (W4A4), which the gfx942 kernel rejects; leave `auto`, which picks `AITER_MXFP4_BF16` (W4A16) (#1011, #1020; main `dsv4_fp8_mi300x.sh`). | **NUM** on MXFP4 models: the backend sets the activation precision (W4A4 vs W4A16); see `numerics-knobs.md` §3 | #934, #939, #1011 |
| `--moe-backend flydsl` | FlyDSL w4a16 grouped-GEMM MoE (INT4 weights dequantized in-kernel). | Kimi-K2.5 INT4 MI355X, decode-dominated MoE (#867). | Shipped with `fuse_allreduce_rms=false` (#867). | nightly-b8336c3c+ (#867). | no | #867 |
| `--moe-backend triton_unfused` (`TRITON_UNFUSED`) | Unfused Triton MoE. | DSv4 FP4 day-0: required for the FP4 expert format then (#555, #731). | | Replaced by `aiter` on nightly-09663abd (#955). | no | #731, #955 |
| `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1` | Router-append fusion: the shared expert joins the AITER fused-MoE launch (vllm#46545). | MiniMax-M3 (#934, #935, #1118), Kimi-K2.5 (#1096), DSv4 MTP (#1201); main. | Independent of the master switch; self-disables under EP (c512990ce message; main `models_vllm.yaml` comment). | nightly-4559c43a+ for M3 (#935). Named 7 times in the records; leaving it unset kept a model-w campaign's iterations dormant (REPORT §2.8). | no | #935, #1096 |
| `--linear-backend emulation` | Dequantizes MXFP8 dense weights and runs the BF16 hipBLASLt GEMM. | MiniMax-M3 MXFP8 MI355X, nightly-09663abd: +26% throughput, -21% TPOT at 8k1k c1; +2-3% at high concurrency (#972). On nightly-9e57de71 the native Triton MXFP8 GEMM wins memory-bound (low concurrency) and emulation wins compute-bound (8k1k c>=64, +3-5%) (#1050, OP). | Counterintuitive: the "emulated" path beat the native one (REPORT §2.5). | The winner flipped between nightlies (#972 vs #1050). | possibly: different rounding from native MXFP8. GSM8K 0.9613 recorded with it on (#972). | #972, #1050 |
| `VLLM_ROCM_USE_AITER_TRITON_BF16_GEMM=0` | Turns off AITER's Triton BF16 GEMM for unquantized layers (falls back to the default GEMM). | 70B FP4 MI355X at CONC<16 and TP>1 (#167, #199, OP). | Above that, the ASM FP4 GEMM (next row) wins. | Dropped for gpt-oss on newer images (#164, #286). | no | #167, #199 |
| `VLLM_TRITON_FP4_GEMM_USE_ASM=1`, `VLLM_TRITON_FP4_GEMM_SPLITK_USE_BF16=1`, `VLLM_TRITON_FP4_GEMM_BPRESHUFFLE` | AITER hand-written assembly FP4 GEMM with BF16 split-K reduction. | 70B FP4 MI355X at large decode batch (CONC>=16, or TP1) (#167, #199, OP). | gfx950 only: the MI300X/MI325X FP4 scripts were deleted, since CDNA3 has no FP4 (#114). | `SPLITK_USE_BF16` dropped on the 09/27 image (#186). | no | #94, #167 |
| `VLLM_ROCM_USE_SKINNY_GEMM=0` | Skips ROCm skinny-GEMM dispatch for small-M GEMMs; uses AITER BF16 GEMM. | Kimi-K2.5 MXFP4 MI355X (#1096). | | v0.15.1 had fixed skinny-GEMM dispatch (#359): re-check per image. | no | #1096, #359 |
| `VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4=1` | AITER A16W4 MoE for gpt-oss on gfx950. | gpt-oss MI355X (#176, #198). | | Removed in the v0.16 cleanup (#367). | no | #176, #367 |
| `VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1` | AITER "SITUV2" A8W4 MoE: 8-bit activations, 4-bit weights. `0` = A16W4 path (script comment, where the recipe's own perf-search variable is called `AITER_A8W4`). | Kimi-K3 MXFP4 MI355X reference recipe (#1171; main). | The header names `AITER_SITUV2_A8W4` instead (§0.3). | Needs backports (`apply_k3_container_patches.sh`, #1171). | **NUM** (activations quantized) | #1171 |
| `VLLM_ROCM_USE_AITER_LINEAR=1` | AITER linear kernels. | DSv4 FP4 first nightly recipe (#731). | | Dropped one nightly later (#756). | no | #731, #756 |
| `TORCH_BLAS_PREFER_HIPBLASLT=1` | PyTorch BLAS prefers hipBLASLt. | MiniMax-M3 MXFP8 MI300X, together with AITER on (#982; main). | Not measured alone. | | no | #982 |

### 1.2 Attention and KV layout

| Knob | What it does | When it paid (regime) | Needs, conflicts | Image notes | NUM | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| `VLLM_ROCM_USE_AITER_MHA` | AITER CK MHA. | Llama-70B FP8: `=1` only for 8k1k at CONC>=16, `=0` for 1k1k and 1k8k: it pays for long batched prefills, not decode-dominated runs (#152, #182, #209, OP). | Must be `0` for unified attention to be selected (#156). MiniMax-M3 MI300X keeps `=0` because the checkpoint has no calibrated FP8 attention scales, so attention stays `TRITON_ATTN` (#982; main). | The default changed in the 09/27 image: set it explicitly (#199, #204). | no | #152, #209 |
| `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1` / `--attention-backend ROCM_AITER_UNIFIED_ATTN` | AITER unified attention: prefill and decode in one Triton kernel tuned for gfx950. | gpt-oss all GPUs (#59, #156, #396). | `VLLM_ROCM_USE_AITER_MHA=0` and `--block-size 64` (#156). | The old name `VLLM_USE_AITER_UNIFIED_ATTENTION` is ignored from v0.14 (#286). Since v0.17 use the CLI flag (#396). | no | #156, #286, #396 |
| `--attention-backend ROCM_AITER_FA` | AITER flash attention for dense full attention. | MiniMax-M2.5 MI355X (#449) and MI325X (#794). Pinning it recovered a ~38% MI325X regression (#794). | | Needed because vLLM PR #36702 made `ROCM_ATTN` the default (#794). | no | #449, #794 |
| `--attention-backend TRITON_ATTN` | Triton attention. | MiniMax-M3 everywhere (MSA, `--block-size 128` mandatory) (#844, #1165; main). | The drafter does not inherit the global flag: put `"attention_backend": "TRITON_ATTN"` in `--speculative-config`, or the drafter picks a slow default. That gave a win at every concurrency with acceptance unchanged (#977). | | no | #844, #977 |
| `TRITON_MLA` (draft attention backend) | Triton MLA for a DSpark drafter. | Kimi-K3 (#1171; main). | | | no | #1171 |
| `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` | Stores KV in the pre-shuffled layout that AITER paged attention (ASM / Gluon) reads natively. | MiniMax-M2.5 MI355X 1k1k non-EP and 8k1k c>=64 with block 16 (#561, #611, OP). MiniMax-M3: AITER page-16 sparse paged attention (vllm#47287). Gated to 8k1k c>=64 together with emulation linear and the quick-reduce thresholds; the whole change gave +3.3/+2.5/+4.7% tok/s/GPU at c128/256/512 and was neutral elsewhere. At short context or low batch the sparse path adds overhead (#1050, OP). | M2.5: block 16 (#611). M3: fp8 KV and exactly one KV head per rank, so TP4 for M3's 4 KV heads; at TP2 use the Triton sparse fallback (#1165; main). **WA** on gfx942: only shapes with an ASM PA kernel work (MI325X TP2/EP1 c<=16, TP8/EP8 c<=64); elsewhere `pa_fwd_asm` crashes with "cannot get heuristic kernel" (gqa=6, block 32) (#794). | M3 needs nightly-9e57de71+ (#1050). | no | #611, #794, #1050 |
| `VLLM_ROCM_AITER_MLA_ASM_PADDING=asm` | Pads small head counts for AITER ASM MLA (12 to 16 heads at TP8). | Kimi-K3 MI355X (main). | Came with backports vllm#50578/#51040 (#1171). | Patched container (#1171). | no | #1171 |
| `--block-size 1` (vLLM) | The page size AITER MLA decode is built for. | Kimi-K2.5 TP8 with AITER on (#425). | | Moved to block 16 with fp8 KV on a later nightly (#1096). | no | #425, #1096 |
| `VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1`, `VLLM_USE_V1=1` | Split prefill/decode Triton attention on the V1 engine. | Llama-70B FP8 MI300X ROCm 7.0-rc1 recipe (#69), which replaced `VLLM_USE_TRITON_FLASH_ATTN=0`. | | Removed on the 09/27 image (#167, #182). `VLLM_USE_V1` is obsolete (V1 is the default). | no | #69, #182 |
| `MM3_HIGH_CONC_FASTPATH`, `VLLM_BLOCK_SIZE` | **Not vLLM variables**: recipe-script switches. The first turns on the MiniMax-M3 shuffled-KV + emulation-linear + quick-reduce fast path for isl>=8192 and conc>=64 (#1050); the second feeds `--block-size` (#611). | | Setting them in a server environment does nothing. | | no | #611, #1050 |
| `VLLM_USE_AITER_TRITON_ROPE` / `VLLM_ROCM_USE_AITER_TRITON_ROPE` | AITER Triton RoPE. | Llama-70B (#69), gpt-oss MI355X (#396). | | Name changed between images. Removed for 70B on 09/27 (#182). | no | #69, #396 |
| `VLLM_USE_AITER_TRITON_FUSED_SPLIT_QKV_ROPE`, `VLLM_USE_AITER_TRITON_FUSED_ADD_RMSNORM_PAD`, `VLLM_USE_AITER_TRITON_GEMM` | Fused QKV split+RoPE; residual add+RMSNorm+pad for the MXFP4 GEMM; AITER Triton GEMMs. | gpt-oss ROCm 7.0-rc1 (#64, #79). | | v0.16: add+RMSNorm+pad is applied by the compile PassManager (#367). Likely stale names on new images. | no | #79, #367 |
| `VLLM_ROCM_USE_AITER_PAGED_ATTN=0`, `VLLM_USE_AITER_TRITON_SILU_MUL=0` | Overrides that turned two AITER paths off. | Kimi-K2.5 disagg (pre-#1051). | | Dropped in favor of "AITER defaults" on nightly-2afa3f7e (#1051, e97a67ffe). | no | #1051 |

### 1.3 Compilation, graphs, runtime

| Knob | What it does | When it paid (regime) | Needs, conflicts | Image notes | NUM | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| `VLLM_USE_BREAKABLE_CUDAGRAPH=0` | Disables the breakable-cudagraph path so full graphs capture. | **WA**: MiniMax-M3 decode tripped an assertion on that path, so the recipe ran `--enforce-eager`. `=0` dropped eager: graph replay instead of per-kernel launches (#849-#853). Kimi-K3: "REQUIRED on ROCm per the upstream recipe: the build auto-enables this to 1" (main). | Model-specific: check per model before copying. | | no | #849, #852 |
| `--compilation-config` `custom_ops: ["-rms_norm","-quant_fp8","-silu_and_mul"]` | Removes vLLM's hand-written custom ops so Inductor fuses add+RMSNorm+FP8-quant and SiLU-mul+quant. | Llama-70B FP8 MI300X/MI325X/MI355X: +6-7% on 6/8 configs (#208). | A removal that wins (REPORT §2.5). | | no | #208 |
| `pass_config.fuse_rope_kvcache=True` + `use_inductor_graph_partition=True` | RoPE fused with the KV-cache write. | gpt-oss v0.17 (#396). | | | no | #396 |
| `pass_config.fuse_allreduce_rms=false` | Turns off the all-reduce+RMSNorm fusion pass. | Kimi-K2.5 INT4 with `--moe-backend flydsl` (#867). | Reason not stated ("presumably incompatible or slower"). | | no | #867 |
| `cudagraph_mode FULL_AND_PIECEWISE`, `mode 3` | Full graphs for decode, piecewise for prefill. | DSv4 FP4: replaced `--enforce-eager` (#756); default on nightly-2afa3f7e (#1051). | Capture sizes must cover the MTP verify batch (#1171). | | no | #756, #1051 |
| `custom_ops: ["+fused_rms_norm_gated"]` | Enables the fused gated RMSNorm custom op. | Kimi-K3 MI355X (main). | | | no | main |
| `VLLM_DISABLE_COMPILE_CACHE=1` | Disables vLLM's torch.compile cache. | gpt-oss ROCm 7.0-rc1 recipe (#74, #79); no reason recorded. | | | no | #79 |
| `VLLM_USE_RUST_FRONTEND=1` | Rust API frontend (HTTP, tokenization, detokenization) in front of the same Python EngineCore. | MiniMax-M2.5 FP4 MI355X, v0.22.0 ROCm image: tok/s/GPU within 1.5%, TPOT flat, TTFT -8% (1k1k) and -22% (8k1k) at every point (#823). | Invisible to a tok/s-only gate (REPORT §2.4). | Needs an image that ships the `vllm-rs` binary (#823). | no | #823 |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=32768` | Keeps prefix-cache tails of sliding-window/Mamba layers (vllm#43447, #44774). | DSv4 agentic trace replay (#997, #1142). | The recipe contradicts itself: main `agentic/dsv4_fp4_mi355x_vllm_mtp.sh` exports it, then comments that it "only applies to sliding-window/Mamba models; this vLLM build raises ValueError if it is set for DSv4". Check the server log on your image. | | no | #997, #1142 |
| `VLLM_USE_SIMPLE_KV_OFFLOAD=1` | Switches the native offload backend to `SimpleCPUOffloadConnector`. | Left unset on purpose: the regular `OffloadingConnector` with the hybrid KV manager gave more prefix hits (main `dsv4_fp4_mi355x_vllm_mtp.sh`; #1073). | | | no | #1073 |
| `VLLM_MORIIO_CONNECTOR_READ_MODE=1` | MoRI-IO KV connector read mode. | Kimi-K2.5 disagg (#769). | | Moved to `kv_connector_extra_config` on v0.24.0 (#987): a rename. | no | #769, #987 |
| `--all2all-backend mori` / `mori_low_latency` | MoRI EP all-to-all. | Disagg EP decode (#769, #987). | Kimi-K2.5: EP with MoRI was 14-27% slower than dense TP in a TP8 sweep (#1051). | `mori` -> `mori_low_latency` on v0.24.0 (#987). | no | #987, #1051 |

## 2. SGLang on ROCm

| Knob | What it does | When it paid (regime) | Needs, conflicts | Image notes | NUM | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| `SGLANG_USE_AITER=1` | Routes MLA decode, fused MoE, block-scale FP8 GEMM and norms to AITER. | DSR1 FP8 MI300X (#67), DSR1 FP4 MI355X (#156), Qwen3.5 (#460); main. | | | no | #67, #156 |
| `--attention-backend aiter` | AITER attention/MLA backend. | DSR1 MI355X (#244, #279); Qwen3.5 triton -> aiter (#465/#494, #819). | Qwen3.5 needed `triton` on early images, for its hybrid layers (#349). | | no | #244, #819 |
| `SGLANG_USE_AITER_UNIFIED_ATTN=1` | AITER unified paged prefill+decode kernel. | Qwen3.5 FP8/FP4 MI355X: with it, one TP4 instance replaced the TP8/TP8-EP8/TP2-EP2 mix (#819, #827); MI325X (#1163). | `--page-size 16`, `--attention-backend aiter`. | v0.5.12.post1+ (#819). | no | #819, #827 |
| `SGLANG_AITER_MLA_PERSIST=1` | Persistent AITER MLA decode kernel: resident workgroups loop over work items. | DSR1 FP8 MI300X/MI325X (#279, #282), DSR1 MTP (#711, #744). | Built for an FP8 KV cache (#279). DSv4 disagg sets `0` (main `env.sh`). | v0.5.7+ (#279). | no | #279 |
| `--enable-aiter-allreduce-fusion` | TP all-reduce fused with residual add + RMSNorm in one AITER kernel. | Qwen3.5 MI30X (#465/#494), Qwen3.5 FP4 (#827), GLM-5.1 FP4 (#944). GLM-5.2 FP4 MI355X with the next two rows: +23.6% P90 interactivity at c8 TP4/EP4 (#0, #1255). | Left off on Qwen3.5 MXFP4 TP2/EP2 EAGLE: "unstable for rank consistency" (#1168). | | no; but see #1168 | #0, #944, #1168 |
| `--enable-fused-qk-norm-rope` | QK-norm + RoPE + KV write in one kernel per attention layer. | GLM-5.2 FP4 MI355X (#0, #1255). | Kernel already in v0.5.16-rocm720-mi35x-20260728 (#0). | | no | #1255 |
| `--num-continuous-decode-steps N` | N decode steps per scheduler pass, amortizing Python scheduling over more GPU work. | DSR1 recipes: 4 (#67); 4 -> 8 gave +4.7% average output throughput on DSR1 FP8 MI355X, found by an agent campaign (#596). | Its effect on TTFT was not recorded: check TTFT, not only tok/s. | | no | #67, #596 |
| `SGLANG_ROCM_FUSED_DECODE_MLA=0` | Turns off the fused ROCm decode-MLA path. | GLM-5/5.1 (NSA, not MLA), with TileLang NSA backends (#392, #503, #516, #767). | | | no | #392 |
| `--nsa-*-backend tilelang`, `--dsa-{prefill,decode}-backend tilelang`, `--dsa-topk-backend torch` | Sparse-attention kernels for GLM-5.x on ROCm. | GLM-5 (#392), GLM-5.2 (#1158; main). MI325X uses the torch top-k (main `glm5.2_fp8_mi325x_mtp.sh`). | | | no | #392, #1158 |
| `SGLANG_OPT_USE_TOPK_V2=false`, `SGLANG_DSA_FUSE_TOPK=false` | Use the precompiled HIP top-k (v1) instead of the v2 JIT kernel. | **WA**: v2 is JIT-compiled from CUDA-only source (`cooperative_groups.h`) and cannot build for gfx950/gfx942 (#1158; main). | | v2 is default since v0.5.14 (main comment). | no | #1158 |
| `SGLANG_MAMBA_SSM_DTYPE=bfloat16` / `--mamba-ssm-dtype bfloat16` | Stores the linear-attention recurrent state in BF16 instead of FP32: halves state memory and bandwidth. | Qwen3.5 FP4 MI355X (#1025; main). | | | **NUM** | #1025 |
| `SGLANG_ENABLE_SPEC_V2=1` | Overlapped speculative scheduling (spec v2). | DSR1/GLM/Qwen MTP recipes (#503, #616, #744; main). | NVIDIA #310: assigned but not exported, so it never reached the server. | | no | #616, #744 |
| `SGLANG_ENABLE_OVERLAP_PLAN_STREAM` | Overlaps MTP plan scheduling. | `1` in #616. | Back to `0` in #766 (no reason recorded); main `env.sh` keeps `0`. | | no | #616, #766 |
| `SGLANG_DP_USE_GATHERV=1`, `SGLANG_DP_USE_REDUCE_SCATTER=1` | DP-attention token exchange with all_gatherv/reduce_scatterv instead of padded all-gather/all-reduce. | DSv4 FP4 MI355X DP-attention (#901, #995). | GLM-5.2 sets them on its DP arm because the defaults hung a collective under long-context prefill. The arm stays dormant: DSA + DP-attention hung on ROCm v0.5.14 with these settings as well (main comments, `glm5.2_fp4_mi355x_sglang_mtp.sh`). | | no | #901, #995 |
| `SGLANG_SHARED_EXPERT_TP1=1`, `SGLANG_DP_SHARED_EXPERT_LOCAL=1` | Shared expert runs locally at TP1 on each DP rank instead of gathering tokens for it. | DSv4 DP-attention (#995; main). | Part of the §0.6 DP-attention bundle. | | no | #995 |
| `--enable-two-batch-overlap` | Two micro-batches: one's MoE dispatch/combine overlaps the other's compute. | DSv4 DP-attention (#995); prefill with MoRI SDMA (#616). | Needs `GPU_MAX_HW_QUEUES=5` (#995). | | no | #995 |
| `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1`, `SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS=1` | Per-component (full/SWA) radix cache; frees out-of-window SWA slots during chunked prefill. | DSv4 agentic: without them, in-flight requests pin SWA KV for their whole context and the prefix hit rate collapses (#1181; main). | Workload: multi-turn. | | no | #1181 |
| `SGLANG_HACK_FLASHMLA_BACKEND` | DSv4 sparse-MLA kernel choice. | `torch` (#524) -> `tilelang` (#593) -> `triton` (#741) -> `unified_kv_triton`: reads compressed and SWA KV in one kernel (#828; main). | | Image-bound: the value set grew per image. | no | #828 |
| DSv4 kernel routing: `SGLANG_OPT_USE_FUSED_COMPRESS`, `SGLANG_OPT_USE_FUSED_COMPRESS_TRITON`, `SGLANG_OPT_USE_AITER_INDEXER` (vs `SGLANG_OPT_USE_TILELANG_INDEXER`), `SGLANG_OPT_USE_AITER_MHC_PRE`, `SGLANG_OPT_USE_AITER_MHC_POST`, `SGLANG_OPT_USE_TILELANG_MHC_PRE`, `SGLANG_OPT_USE_TILELANG_MHC_POST`, `SGLANG_OPT_USE_TRITON_SWA_PREPARE`, `SGLANG_OPT_USE_TRITON_FUSED_MHC`, `SGLANG_OPT_FUSE_WQA_WKV`, `SGLANG_OPT_USE_OLD_COMPRESSOR`, `SGLANG_OPT_DEEPGEMM_HC_PRENORM=false`, `SGLANG_OPT_USE_JIT_INDEXER_METADATA=false`, `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false`, `SGLANG_OPT_USE_FUSED_HASH_TOPK`, `SGLANG_OPT_USE_FUSED_STORE_CACHE`, `SGLANG_OPT_USE_FUSED_PAGED_COMPRESS`, `SGLANG_OPT_DPSK_V4_RADIX=1`, `SGLANG_EAGER_INPUT_NO_COPY`, `SGLANG_USE_ROCM700A=0`, `SGLANG_ROCM_USE_MULTI_STREAM=false`. Record #741 drops the `SGLANG_OPT_` prefix on four of these (`USE_FUSED_HASH_TOPK`, ...); the commit (6ccc5c784) has the full names. | Select DSv4 fused kernels (compressor, indexer, hyper-connection pre/post, SWA prepare, fused projections) vs fallbacks. | Turned on as each image gained ROCm support (#593, #605, #617, #741, #798, #828). | Day-0 images needed about 20 of these off for correctness (#524). Many became defaults, and 15 were deleted (#995). | Grep `sglang/srt/environ.py` in the image; names and defaults change per image. | no (as kernel swaps) | #593, #741, #828, #995 |
| `SGLANG_OPT_C4_SPARSE_TOPK=512` | Sets a sparse top-k for DSv4's compressed attention; the meaning is not stated in the records. | DSv4 MTP (#798). | If it changes the model's own top-k it changes numerics: compare with the model config's default before touching it. | | unknown | #798 |
| `SGLANG_DEFAULT_THINKING=1`, `SGLANG_DSV4_REASONING_EFFORT=high/max` | Thinking mode and reasoning effort for DSv4. They change what the model generates (output length and content), so they define the workload. | `max` in the fixed-seq-len recipes, `high` in the agentic ones (main). | Keep identical across arms; a change is a workload change, not an optimization. | | changes outputs | main |
| `SGLANG_OPT_FP8_WO_A_GEMM=0`, `SGLANG_TOPK_TRANSFORM_512_TORCH=1`, `SGLANG_ENABLE_JIT_DEEPGEMM=0` | Torch/BF16 fallbacks when `deep_gemm` is missing (wo_a dequantized to BF16). | **WA** on images without deep_gemm (#798). | | Removed on v0.5.14 (#998). | possibly (BF16 wo_a) | #798, #998 |
| `SGLANG_OPT_USE_AITER_BATCHED_GEMM=1` | AITER batched GEMM for the absorbed MLA projections (default off in `environ.py`). | DSv4 MI355X v0.5.18 (#1232; main). | | v0.5.18 image (#1232). | no | #1232 |
| `--enforce-shared-experts-fusion` | Shared expert joins the routed list (385 experts, topk 7). | DSv4 MI355X (#1232). | Only because the image's tuned MoE tables cover that shape (#1232). | | no | #1232 |
| `SGLANG_DSV4_FP4_EXPERTS=True`, `SGLANG_FORCE_TRITON_MOE_FP8=0` | FP4 experts through AITER instead of the Triton FP8 MoE. | DSv4 FP4 (#593). | The day-0 FP8 image forced Triton MoE FP8 for correctness (#524). | | no | #524, #593 |
| `AITER_BF16_FP8_MOE_BOUND` | An AITER MoE dispatch bound; its meaning is not stated in the recipes or the records. | `0` on every DSv4 recipe (SGLang, ATOM) and the Kimi-K3 vLLM recipe at main (#652, #771, #1171, #1181, #1223). | Was `1` for one image (#741), then reverted to `0` (#771). | Read `aiter/` in the image before changing it. | unknown | #741, #771 |
| `AITER_FLYDSL_FORCE=1` | Forces AITER's FlyDSL kernels over its default heuristics (FP4 MoE/GEMM). | Qwen3.5 FP4 MI355X (#863, #1168; main). | | v0.5.13+ (#863). | no | #863 |
| `--disable-custom-all-reduce` | Plain RCCL instead of the custom all-reduce. | **WA**: "avoids a known aiter fault on MI355X" (DSv4 agentic disagg; main, 23f879df2, #1154). | | | no | #1154 |
| `--enable-torch-compile` | torch.compile. | Added for DSR1 FP8 (#255); removed: throughput/GPU 1.00-1.03x without it, ~30 min compile per start (#301). | | | no | #301 |
| `SGLANG_SIMULATE_ACC_LEN`, `SGLANG_SIMULATE_ACC_METHOD=match-expected`, `SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token` | Synthetic speculative acceptance for throughput replay. | AgentX throughput runs (#1167; main). | Never in evals: evals keep real verification (main comments). | | **NUM** (outputs are not the model's) | #1167 |

## 3. ATOM

| Knob | What it does | When it paid (regime) | Needs, conflicts | Image notes | NUM | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| `ATOM_MOE_GU_ITLV=1` | Interleaved gate/up MoE weight layout: the fused MoE kernel reads gate and up rows together. | DSv4-Pro FP4 (#652, #807; main). | Gated to DeepSeek-V4-Pro only (#918). | atom-dev nightly 2026-05-13+ (#652). | no | #652, #918 |
| `ATOM_DISABLE_MMAP=true` | Loads weights without mmap: faster load. | DSv4 (#652), Kimi-K2.5 (#1003). | | | no | #1003 |
| `ATOM_M3_SPARSE_USE_ASM_PA=1` vs `ATOM_FORCE_ATTN_TRITON=1` | MiniMax-M3 sparse (MSA) attention: AITER assembly paged attention vs Triton. | ASM won on the 0623 image (#926, #927); Triton won on the 0630 image (#941, #943). Reverted (#945), then relanded with Triton (#952; main `models_atom.yaml`). | | Image-bound; #952 calls the ASM var deprecated. | no | #926, #941, #952 |
| `ATOM_USE_TRITON_MOE=1` | Triton `matmul_ogs` MoE. | **WA**: DSv4 day-0, AITER fused MoE broken on gfx950 for a16w4+SwiGLU (#551). | | Replaced by native AITER MXFP4 MoE (#560). | no | #551, #560 |
| `ATOM_DSV4_SPARSE_ATTN_CHUNK_TOKENS=256` | Chunks prefill over M in a patched torch sparse-attention fallback (bounds score-tensor memory). | DSv4 day-0 runtime patch (#560). | | Obsolete once the nightly shipped AITER sparse attention (#652). | no | #560 |
| `--enable-tbo` / `--enable-tbo prefill` | Two-batch overlap. | DSv4 FP4 DP-attention: non-MTP from c>=64 on nightly 2026-07-23 (#1068). Earlier image: 1k1k c>=1024, 8k1k c>=256 (#877). With MTP: -14%/-10% at c64/c128, +11/+8/+14% at c>=256, so the gate is 256 (#1067, OP). | `GPU_MAX_HW_QUEUES=5` (#877). With MTP use prefill-only TBO: decode TBO drops MTP's `spec_decode_metadata` (#1067). | The threshold moved with the image (#1068). | no | #877, #1067, #1068 |
| `--max-num-seqs CONC` (DP-attention) | Caps sequences. | **WA**: DP-attention keeps a full KV pool per rank, so the default `max_num_seqs` OOMs even at c16 (#1067, #1068; main). | MTP reserves num_spec+1 slots per request (#1067). | | no | #1068 |
| `ATOM_CPU_AFFINITY=1` | CPU affinity for ATOM workers. | DSv4 disagg TP+DPA (main `models_atom.yaml`). | | | no | main |
| `--online_quant_config '{"global_quant_config":"ptpc_fp8","exclude_layer":[...]}'` | Online per-token-per-channel FP8 quantization of BF16 layers at load, run on FP8 GEMMs. | MiniMax-M3 (#941, #952), GLM-5.2 (#1208), Kimi-K3 (#1214). | The exclude list matters: lm_head, embeddings, router gate, experts, vision; under DPA also `*.gate.*` (#990). | | **NUM** | #952, #990, #1208 |
| `--hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}'` | Reuses the sparse-attention indexer's top-k across 4 layers. | DSv4 ATOM (#877), M3 ATOM (#941). | Removed: reduces model FLOPs, disallowed by InferenceX review (#945, #959). | Still in the M3 FP4 vLLM agentic recipe (73bd8e200, #1165; main). | **NUM** | #959 |
| `--kv_cache_dtype fp8` (ATOM) | FP8 KV. | Most ATOM recipes; main. | **WA** on early images: M3 FP4 tripped a `k_scale` assertion (no FP8 KV scales in the checkpoint) (#874); possible from the 0623 image (#926). | | **NUM** | #874, #926 |
| `--index-cache-dtype fp4` | FP4 sparse-indexer cache. | DSv4 agentic ATOM (main; commit c2b58030b). | Record #1223 says "FP8 ... index caches"; the committed script passes `fp4`. | | **NUM** | #1223 |
| `--scheduler-delay-factor 1` | Waits to batch arriving prefills. | Kimi-K2.5 (#1003, #1028). | | | no | #1003 |
| `ATOM_ENABLE_REPLAYSSM`, `ATOM_STATE_CHECKPOINT_DEMAND=0`, `OFFLOAD_STATE=1`, `OFFLOAD_STATE_CPU_SIZE`, `OFFLOAD_STATE_STAGING_GROUPS`, `OFFLOAD_STATE_MIN_LOAD_TOKENS`, `OFFLOAD_MIN_LOAD_TOKENS`, `OFFLOAD_GPU_STAGING_CHUNKS`, `OFFLOAD_KV_FOR_HYBRID=1` | Hybrid-state handling for Kimi-K3 (KDA recurrent state): in-GPU checkpoints vs a CPU state tier. | Chosen per concurrency band (#1254, OP). | `OFFLOAD_GPU_STAGING_CHUNKS` must hold one state entry (54.78 MiB). The default 8 MiB makes the tier silently decline after one log line (#1214). | | no | #1214, #1254 |
| `OMP_NUM_THREADS=1` | One OpenMP thread per process. | ATOM recipes (#266, #332, #442), vLLM Kimi-K2.5 (#1096). | | | no | #266 |

## 4. AITER (cross-framework)

| Knob | What it does | When it paid (regime) | Needs, conflicts | Image notes | NUM | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| `AITER_CONFIG_FMOE=<csv>` | Tuned fused-MoE table overlay. Rows route eligible token counts to FlyDSL FP4 MoE kernels. | DSv4 FP4 ATOM, from sunway513/aiter@e450e4d (#560). | Only the shapes in the table are routed. | Deleted when the nightly shipped equivalent tables (#652). REPORT §2.8 lists it among the knobs no `--help` or `envs.py` shows. | no | #560, #652 |
| `AITER_USE_FLYDSL_MOE_SORTING=1` | FlyDSL MoE token sorting in dispatch (ATOM's default for GLM-5.2). | GLM-5.2 SGLang (#0, #1255), ATOM (#1208). | | | no | #0, #1208 |
| `AITER_FLYDSL_FORCE=1` | See §2. | | | | no | #863 |
| `AITER_MXFP4_INTERMEDIATE=1` | Keeps the MoE gate-up output in MXFP4, so the down projection is an FP4 GEMM without a BF16 round trip. | Kimi-K2.5 ATOM (#1003, #1028) and vLLM (#1096). | | AITER >= v0.1.16.post5 for vLLM (#1096). | **NUM** | #1003, #1096 |
| `AITER_SITUV2_A4W4=1`, `AITER_FLYDSL_STAGE2_FP8=1` | A4W4 (FP4 activations) MoE; FlyDSL stage-2 MoE GEMM in FP8. | Kimi-K3 ATOM (#1214, #1254; main). | | | **NUM** | #1214 |
| `AITER_BYPASS_TUNE_CONFIG=0`, `AITER_MOE_SORT_BACKEND=auto` | Use AITER tuned configs; auto MoE sort backend. | Kimi-K2.5 vLLM (#1096). | | | no | #1096 |
| `AITER_BF16_FP8_MOE_BOUND` | See §2. | | | | unknown | #771 |
| `AITER_LOG_LEVEL=ERROR/WARNING` | Log volume. | **WA**: "Required by ATOM: without it the aiter kernel logs flood the server log" (main `kimik3_fp4_mi355x_atom_mtp.sh`); #795. | | | no | #795 |
| `AITER_MXFP4_MXFP4`, `AITER_MXFP4_BF16` | **Not env vars**: vLLM MXFP4 MoE backend names (W4A4 native vs W4A16 dequant). | See `--moe-backend` (§1.1). | | | | #1011, #939 |
| Tuned-table coverage | AITER's tuned tables cover specific (inter_dim, experts, topk) shapes. | DSv4 shared-expert fusion was chosen because the tables covered the new shape (#1232). | Out-of-table shapes fall back to heuristics or fail ("Unsupported kernel config for moe heuristic dispatch", main `env.sh`). | | | #1232 |

## 5. Quick-reduce: quantized TP all-reduce (all frameworks)

**What it is:** AMD's quick all-reduce quantizes the TP all-reduce payload.
- INT4 moves about 4x fewer bytes than BF16 (#152, #244); INT6 about 2.7x fewer (#935).
- A quick-reduce benchmark cited 2.2x speedup for INT4 against 1.17x for INT8 (#425 commit).
- Every value except full precision is **NUM**.

| Framework | Variables | Notes | Evidence |
| --- | --- | --- | --- |
| vLLM | `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` = `INT4` / `INT6` / `INT8` | INT8 -> INT4 (#425); INT6 -> INT4, "accuracy guarded by the 8k1k evals" (#1050, main comment); INT6 for M3 disagg (#1001, main). | #425, #935, #1050 |
| vLLM | `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB=256` | **Required for decode.** The default INT4 gate for (bf16, TP4) is 16 MB, so ~1.5 MB decode all-reduces never quantize. Lowered, it gave -12% to -17% TPOT at c64/128/256 (MiniMax-M3 MXFP4 MI355X). | #1089, #1073, #1165 |
| vLLM | `VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=0` | Keeps BF16 instead of casting to FP16 before the reduce ("keep the bf16 accumulation", main comment). Set with the MIN_SIZE override. | #1050, #1089 |
| vLLM | `VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=2048` | Prefill all-reduces at TP8 (hundreds of MB) exceed the default maximum. Raising it quantizes them. Set only on prefill workers in disagg; decode stays full precision. | #1118 |
| SGLang | `ROCM_QUICK_REDUCE_QUANTIZATION=INT4` | DSR1 MI355X (#244, #255; main), GLM-5/5.1 (#392, #516). | #244, #392 |
| ATOM | `AITER_QUICK_REDUCE_QUANTIZATION=INT4`, `AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=0` | MiniMax-M3 (#915, #926), Kimi-K2.5 (#1003), GLM-5.2 (#1208), Kimi-K3 (#1214). #952 calls the cast variable deprecated. | #926, #952 |

**When it helps:** TP decode where the all-reduce is a visible share of the step (MiniMax-M3 high
concurrency, #1089), and TP8 prefill in disagg (#1118). gpt-oss added INT4 and removed it on
MI325X/MI355X with no reason stated (#156, #158); do not assume it always pays.

**How to check:** confirm the decode all-reduce kernel in a trace is the quick-reduce one, at the
tensor sizes you run. A set variable with no kernel change means the size gate kept it off (#1089).

## 6. MoRI (EP all-to-all and KV transfer; multi-node)

Mostly outside a single-node campaign's reach. Listed so that a recipe copied from multi-node is
understood.

| Knob | What it does | Notes | NUM | Evidence |
| --- | --- | --- | --- | --- |
| `SGLANG_MORI_FP8_DISP` | FP8 EP dispatch. | `True` (#352); off for MXFP4 models, to avoid re-quantizing FP4 activations (#406). | **NUM** | #352, #406 |
| `SGLANG_MORI_DISPATCH_DTYPE=auto` | FP4 dispatch for MXFP4 (#616); `auto` for DSv4 (#996; main). | | **NUM** | #616, #996 |
| `SGLANG_MORI_FP8_COMB=true` | FP8 combine. | Failed GSM8K at high concurrency, far from the 0.95 golden score (#609). Replaced, see next row (#766). | **NUM** | #609, #766 |
| `MORI_COMBINE_DTYPE_PREFILL=fp8_direct_cast`, `MORI_COMBINE_DTYPE_DECODE=fp8` | Per-phase combine dtype; decode uses blockwise-scaled FP8 to restore accuracy. | main `env.sh` now sets both to `""` globally and `auto` for DSv4. | **NUM** | #766 |
| `MORI_MAX_DISPATCH_TOKENS_PREFILL`, `MORI_MAX_DISPATCH_TOKENS_DECODE`, `MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK_PREFILL`, `MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK_DECODE`, `MORI_MOE_MAX_INPUT_TOKENS_PREFILL`, `MORI_MOE_MAX_INPUT_TOKENS_DECODE`, `SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD`; the harness also derives `SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK`, `SGLANG_MORI_MOE_MAX_INPUT_TOKENS`, `SGLANG_MORI_COMBINE_DTYPE` from them (main `server_sglang.sh`) | Per-rank dispatch buffer sizing; the intra/inter-node kernel switch at 2x the decode cap. | Size to real per-rank load: decode cap = conc / dp_ranks; MoE input = 0.7 x that x ranks (#766); 256 -> 160 (#406). Oversized buffers waste memory and graphs. | no | #352, #406, #766, #996 |
| `MORI_ENABLE_SDMA=true` (+ `HSA_ENABLE_SDMA=1`) | SDMA engines move all-to-all data without CUs, so TBO can hide prefill all-to-all. | Prefill with `--enable-two-batch-overlap` (#616). | no | #616 |
| `MORI_SHMEM_MODE=ISOLATION`, `MORI_EP_LAUNCH_CONFIG_MODE=AUTO` | Allocation and launch config. | main `env.sh`. | no | #352 |
| `MORI_IO_QP_MAX_SEND_WR`, `MORI_IO_QP_MAX_CQE`, `MORI_IO_QP_MAX_SGE`, `MORI_IO_SQ_BACKOFF_TIMEOUT_US`, `SGLANG_MORI_QP_PER_TRANSFER`, `SGLANG_MORI_NUM_WORKERS` | RDMA queue depths for KV transfer. | Deep queues keep KV transfers from stalling under load (#406, #929). | no | #406, #929 |
| `MORI_RDMA_TC`, `MORI_RDMA_SL`, `MORI_IO_TC`, `MORI_IO_SL`, `MORI_IO_TC_DISABLE`, `MORI_NIC_TYPE` | RDMA traffic class / service level from the NIC's PFC no-drop priority (`nicctl`). | A misconfigured NIC silently degrades tail latency; `rdma_check.sh` fails fast on PFC/DCQCN (main). | no | #929 |
| `MORI_CONN_PATCH` | Bind-mounts a patched SGLang `disaggregation/mori/conn.py`. | **WA** for MoRI PD startup crashes on hybrid-attention models (GLM-5, Qwen3.5); skipped for DSv4 (#767; main). | no | #767 |
| SGLang + MoRI + DP-attention | MoRI is not counted as a DeepEP-class backend, so an expert-count assertion fires. | dp-attn stays off with MoRI (#768, #1210). | | #768 |

## 7. RCCL and collectives

| Knob | What it does | Notes | NUM | Evidence |
| --- | --- | --- | --- | --- |
| `NCCL_MIN_NCHANNELS=112` | More RCCL channels (more CUs on communication). | gpt-oss/70B (#152, #156); dropped on the 09/27 image (#198); MiniMax-M3 MI300X TP8 (#982; main). | no | #152, #198, #982 |
| `RCCL_MSCCL_ENABLE=0` | Keeps RCCL on its default algorithms instead of MSCCL. | DSR1 MI355X SGLang (#236, #255, #746; main). | no | #255 |
| `TORCH_NCCL_BLOCKING_WAIT=true` | Blocking wait instead of the watchdog's `hipEventQuery` poll. | **WA**: ROCm <= 7.2.0 does not honor thread-local capture mode, so the watchdog's query invalidated graph capture (TP8+EP8 MoRI). Real fix: ROCm 7.2.2+ (main `env.sh`). | no | main |
| `--disable-custom-all-reduce` | See §2. | **WA** for an AITER fault on MI355X. | no | #1154 |
| Quick-reduce, AITER AR+RMSNorm fusion | See §5 and §2. | | §5 | |

## 8. HSA and HIP runtime

| Knob | What it does | When, needs | NUM | Evidence |
| --- | --- | --- | --- | --- |
| `HSA_NO_SCRATCH_RECLAIM=1` | Keeps kernel scratch memory instead of reclaiming it. | **WA**: set when the MEC firmware is older than 177 (`rocm-smi --showfw \| grep MEC`), because RCCL "cannot reclaim some memory" and crashes (main `dsr1_fp8_mi300x.sh`, `kimik3_fp4_mi355x_mtp.sh`; #182, #367, #425). #79 also read it as a performance setting. DSv4 disagg sets `0` "to resolve the OOR issue" (main `env.sh`). | no | #182, #425 |
| `GPU_MAX_HW_QUEUES=5` | More HIP hardware queues, so two overlapped streams actually run concurrently. | Required with TBO and DP-attention overlap (#877, #995, #1067). Without it the streams serialize (REPORT §2.3). | no | #877, #995 |
| `GPU_MAX_HW_QUEUES=2` | Fewer hardware queues. | MiniMax-M3 MXFP8 MI300X (#982; main); effect not measured alone. | no | #982 |
| `HSA_ENABLE_SDMA=1` | Enables SDMA copy engines. | Multi-node `job.slurm` (main); pairs with `MORI_ENABLE_SDMA` (#616). | no | main |
| `HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES` | Device visibility for Ray in vLLM 0.14+. | Plumbing (#367). | no | #367 |

## 9. Triton-AMD compiler

| Knob | What it does | When, needs | NUM | Evidence |
| --- | --- | --- | --- | --- |
| `TRITON_HIP_USE_ASYNC_COPY=1`, `TRITON_HIP_ASYNC_FAST_SWIZZLE=1`, `TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1` | Async global->LDS copies with a fast swizzle in every Triton kernel. | Llama-70B FP8 MI300X/MI325X/MI355X ROCm 7.0-rc1 recipe (#69, #74, #81). Removed on the 09/27 image (#167, #182, #186). Each alone sits under a 1% gate (REPORT §2.3). | no | #69 |
| `TRITON_HIP_USE_BLOCK_PINGPONG=1` | Two-wavefront ping-pong scheduling: memory overlaps MFMA (CDNA3). | Same recipe (#69); 70B FP4 MI355X (#114). | no | #69, #114 |
| `AMDGCN_USE_BUFFER_OPS` | Buffer loads in Triton codegen. | `=1` in the ROCm 7.0-rc1 set (#69) and ATOM DSR1 FP4 MTP (#332; main). `=0` for gpt-oss on vLLM v0.17 (#396). The best value flips by kernel. | no | #69, #396 |
| `TRITON_HIP_PRESHUFFLE_SCALES`, `ROCM_TRITON_MOE_PRESHUFFLE_SCALES` | Preshuffled MXFP4 scales for the Triton MoE GEMM. | `=1` gpt-oss (#64, #79). `=0` with the ASM FP4 GEMM (#94). `ROCM_TRITON_MOE_PRESHUFFLE_SCALES=0` dropped on the 09/27 image (#164, #198). | no | #79, #94 |
| `TRITON_CACHE_DIR` | Per-job Triton cache in `/tmp`. | Plumbing (#854). | no | #854 |

## 10. Load time and plumbing (no throughput effect)

- **Load time:**
  - `SAFETENSORS_FAST_GPU=1` (#392; main);
  - `--load-format fastsafetensors` (Kimi-K3, main);
  - `ATOM_DISABLE_MMAP=true` (#1003).
- **Timeouts:**
  - `VLLM_ENGINE_READY_TIMEOUT_S` (7200 for the 2.8T Kimi-K3, 10800 for DSv4-Pro's 805 GiB
    checkpoint; main) and
    `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`;
  - `SGLANG_TIMEOUT_KEEP_ALIVE=900/1800`: uvicorn's 5 s default races AIPerf's 300 s keep-alive
    pool (#1158; main);
  - `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT`, `SGLANG_DISAGGREGATION_WAITING_TIMEOUT`,
    `SGLANG_HEALTH_CHECK_TIMEOUT`, `ATOM_PROFILER_TIMEOUT`, `VLLM_RPC_TIMEOUT`.
- **Disagg plumbing:** `SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS=32` (#616),
  `VLLM_SERVER_DEV_MODE=0`, `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` (main `env.sh`).
- **Logs and debug:**
  - `AITER_LOG_LEVEL`, `MORI_APP_LOG_LEVEL`, `ATOM_LOG_LEVEL`, `VLLM_LOG_LEVEL`, `NCCL_DEBUG`,
    `ATOM_DEBUG_PREFIX_HITS`, `ATOM_PROFILER_MORE`;
  - `OFFLOAD_PROFILE` (statistics only: kept on because the submitted numbers were measured with it
    on; main).
- **Ports, paths and process handles:** `*_PORT`, `*_WS_PATH`, `*_CMD`, `*_PID`,
  `LMCACHE_*` sizing (see the KV-offload notes in the agentic recipes), `ROCM_PATH`,
  `VLLM_CACHE_ROOT`, `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`.
- **Recipe bookkeeping:** `ATOM_PR_SHA` (the ATOM PR overlaid at runtime on day 0, #551),
  `ATOM_GPT_OSS_MODEL`.
