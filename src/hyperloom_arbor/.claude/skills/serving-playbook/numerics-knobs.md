# Numerics-changing knobs: what needs an accuracy check before a KEEP

Every knob or setting in the InferenceX recipes (AMD and NVIDIA) that changes model numerics or
outputs: what it changes, the accuracy evidence on record, the check to run, and whether packs
usually leave it open.

Built 2026-10-20 from the runner's docs/research/inferencex-study (records.jsonl ids) and
InferenceX origin/main.

**How to read it:**
- Read §0 (Rules) and §1 (golden scores) first, then find the knob by category:
  - §2 quantized collectives;
  - §3 activation and weight precision at run time;
  - §4 KV cache;
  - §5 compute dtype and math mode;
  - §6 architecture overrides;
  - §7 kernel paths with recorded correctness failures;
  - §8 speculative decoding.
- `#655` is a record id of the InferenceX study (its records are not carried in this copy). `main` means the setting is in InferenceX origin/main
  (6d6d296c6, 2026-08-30). `amd-knobs.md` (same folder) has each AMD knob's performance side.
- **Pack column:**
  - **fixed**: the pack usually fixes it. Arbor packs often fix the TP size, the quantization and the
    KV dtype (study `method/INSTRUCTIONS.md`); change it only if the pack allows.
  - **gate**: open to an agent, but keep it only after the accuracy gate in §0.
  - **no**: disallowed by InferenceX's review rules, or valid only for throughput runs.
- An empty "accuracy evidence" cell means the records hold no accuracy number for that knob. Say so
  in the ledger instead of assuming it is safe.

## 0. Rules

1. **Throughput cannot see accuracy** (REPORT §2.4):
   - an FP8 MoE combine was merged on its throughput, then failed GSM8K far below the 0.95 golden
     score at high concurrency (#609, #766);
   - the AITER master switch produced degenerate MiniMax-M3 output under TP-only (#935);
   - a decode-only MTP setting cut GSM8K from 0.96 to 0.88 with no error and no slowdown (#1059).
2. **The pack's correctness validator is not an accuracy gate.**
   - What it does: `validator/correctness.sh` in e.g. `packs/agentx-dsv41-6` sends one
     "What is 2+2?" request.
   - What it catches: garbage output such as #935, #982, #595.
   - What it misses: a drop like 0.96 -> 0.88 (#1059).
   - Every knob in this file needs GSM8K.
3. **The check.**
   - **Harness:** GSM8K through lm-eval, as InferenceX's `run_eval` does (#275; dataset
     `openai/gsm8k`, #314).
   - **Arms:** both arms, same image, same harness, same chat encoding.
   - **Metrics:** report strict and flexible match.
   - **Reference:** compare with the baseline arm and with the model's golden score (§1).
   - **Verification:** evals use real speculative verification, never synthetic acceptance (main
     agentic recipes; #1185).
4. **Check every layout and load the KEEP will run.** The recorded failures were specific to one:
   - TP-only (#935);
   - EP (#595, #982);
   - TP=2 (#779);
   - TP<8 (#425);
   - high concurrency (#609, "4k");
   - prefill vs decode roles (#1059).

   Two practical points:
   - Run the eval at concurrency >= 16 where possible. InferenceX evaluates at the median eligible
     concurrency >= 16 (#507).
   - Cap the eval's concurrency where KV is tight, or the server OOMs mid-GSM8K (#1020; main
     `dsv4_fp8_mi300x.sh`).
5. **Prove the knob was on during the eval.** #595 passed GSM8K (0.9318) because the launcher had
   silently dropped `--ep-size`; with EP actually applied, GSM8K was 0.0000. See `amd-knobs.md` §0.1.
6. **Re-run after an image change.** Fixes and regressions both arrive with images:
   - vllm#47158 fixed AITER under TP-only (#972);
   - SGLang v0.5.12-20260517 broke GLM-5.1 at TP=2 and v0.5.12.post1 fixed it (#779);
   - a later TRT-LLM build fixed a corrupting fused kernel (#607).
7. **Restoring the checkpoint's format is not adding quantization, and both need the check.**
   - Restoring: consuming a format the checkpoint already stores (the agentx-dsv41 packs' "datatype
     gap": FP8/FP4 weights run through a BF16 dequant fallback) moves numerics back towards the
     reference.
   - Adding: quantizing what the checkpoint keeps in BF16 (online FP8, INT4 all-reduce, FP4
     activations) moves away from it.
   - Either way a different kernel computes the result, so run the check.

## 1. Golden scores on record (GSM8K)

| Model, setting | Score | Evidence |
| --- | --- | --- |
| DeepSeek-R1-0528 (FP8/FP4) | golden 0.95 (InferenceX harness) | #609 |
| DeepSeek-V4-Pro FP4, SGLang MI355X, EP silently 1 | 0.9318 strict | #595 |
| DeepSeek-V4-Pro, SGLang with MTP (sglang#26383) | 0.950 | #798 |
| DeepSeek-V4-Pro, SGLang PD disagg, MTP depth 1 | ~0.96 (depth 2 ~0.85; decode-only MTP flags 0.88) | #1059 |
| GLM-5 FP8, MI355X SGLang disagg | 0.971 strict / 0.970 flexible | #767 |
| GLM-5.1 FP4, MI355X SGLang | 0.975 (TP2 c64), 0.974 (TP4 c16); upstream gate 0.92 | #779 |
| MiniMax-M3 MXFP8, MI355X vLLM TP4, AITER on | 0.9613 flexible / 0.9621 strict | #972 |
| MiniMax-M3 MXFP8, MI300X vLLM | 0.959 | #982 |
| DSR1 FP8 GB300 TRT-LLM (NVIDIA) | 95%; 88% with a numeric bug | #872 |

The failures on record were mostly large: 0.0000 (#595), ~0.32 (#779), ~0.85 (#1059), "nowhere
near" 0.95 (#609). One was not: 0.96 -> 0.88 (#1059).

## 2. Quantized collectives

| Setting | What it changes | Accuracy evidence | Check | Pack |
| --- | --- | --- | --- | --- |
| Quick-reduce INT4 / INT6 / INT8: `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` (vLLM), `ROCM_QUICK_REDUCE_QUANTIZATION` (SGLang), `AITER_QUICK_REDUCE_QUANTIZATION` (ATOM). AMD. | The TP all-reduce payload is quantized (INT4 ~4x fewer bytes than BF16, INT6 ~2.7x), so every layer's partial sums are rounded. | No recorded GSM8K failure. INT6 -> INT4 "at all concurrencies (accuracy is guarded by the 8k1k evals)" (main comment, `deprecated/minimaxm3_fp8_mi355x.sh`; #1050). The study reader called it "a small accuracy cost" (#392). gpt-oss added INT4 and removed it with no reason stated (#156, #158). | GSM8K at each TP size used. Re-run when the level changes (INT8 -> INT6 -> INT4: #425, #935, #1050). | gate |
| `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB`, `..._MAX_SIZE_BYTES_MB` | Widen which tensors get quantized: the decode all-reduces (~1.5 MB, below the default 16 MB INT4 gate at bf16 TP4), or prefill all-reduces above the default maximum. | None recorded. The coverage change is large: without MIN_SIZE the INT4 setting never touched decode (#1089). In disagg, MAX_SIZE was applied on prefill workers only; decode kept full precision (#1118). | Treat a threshold change as a new numerics change: re-run GSM8K. | gate |
| `VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16`, `AITER_QUICK_REDUCE_CAST_BF16_TO_FP16` | `1` casts BF16 to FP16 before the quick reduce; `0` keeps BF16 ("keep the bf16 accumulation", main comment). | None recorded. | GSM8K with the quick-reduce setting. | gate |
| MoRI FP8 combine: `SGLANG_MORI_FP8_COMB=true` (AMD, multi-node EP) | MoE combine all-to-all in FP8. | **Failed**: GSM8K "nowhere near the golden score of 0.95" at high concurrency (4k), on both the InferenceX and SGLang harnesses (#609). | Replaced by per-phase dtypes (next row). | gate |
| `MORI_COMBINE_DTYPE_PREFILL=fp8_direct_cast`, `MORI_COMBINE_DTYPE_DECODE=fp8` | Direct-cast FP8 combine in prefill; blockwise-scaled FP8 combine in decode. | Introduced to restore accuracy (#766); no score recorded. main `env.sh` now leaves both empty globally and `auto` for DSv4. | GSM8K at the highest concurrency served. | gate |
| MoRI FP8/FP4 dispatch: `SGLANG_MORI_FP8_DISP`, `SGLANG_MORI_DISPATCH_DTYPE=auto` | MoE dispatch all-to-all in FP8, or FP4 for MXFP4 models. | Turned off for MXFP4 models to avoid re-quantizing already-FP4 activations to FP8 "and its error" (#406). FP4 dispatch enabled in #616. No score. | GSM8K. | gate |
| NVIDIA: `VLLM_DSV4_MEGA_FP8_COMBINE=1`, TRT-LLM `use_low_precision_moe_combine: true`, `SGLANG_MOE_NVFP4_DISPATCH` | FP8 combine in DeepGEMM MegaMoE; low-precision MoE combine in TRT-LLM; NVFP4 MoE dispatch. | None recorded. | GSM8K. | gate |

## 3. Activation and weight precision at run time

| Setting | What it changes | Accuracy evidence | Check | Pack |
| --- | --- | --- | --- | --- |
| vLLM ROCm MXFP4 MoE backend: `--moe-backend aiter` vs `auto` | On gfx950, `aiter` selects `AITER_MXFP4_MXFP4` (W4A4: activations quantized to MXFP4). `auto` on the DSv4/gfx942 path picks `AITER_MXFP4_BF16` (W4A16, BF16 activations) (main `dsv4_fp8_mi300x.sh` comment; #1011, #1020). So a backend switch changes activation precision. | None recorded for the switch. | GSM8K after any MoE backend change on an MXFP4 model. | gate |
| AITER MoE activation formats: `VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4=1` (A8W4; `0` = A16W4), `AITER_SITUV2_A4W4=1` (A4W4), `AITER_FLYDSL_STAGE2_FP8=1` (stage-2 MoE GEMM in FP8) | MoE activations quantized to 8 or 4 bits. | None recorded (Kimi-K3: #1171, #1214, #1254). | GSM8K. | gate |
| `AITER_MXFP4_INTERMEDIATE=1` | Keeps the MoE gate-up output in MXFP4; the down projection runs as an FP4 GEMM. | None recorded (Kimi-K2.5: #1003, #1028, #1096). | GSM8K. | gate |
| NVIDIA FP4 activations in MegaMoE: `SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=1` + `..._USE_MXF4_KIND=1`; vLLM `--moe-backend deep_gemm_amxf4_mega_moe` | Expert GEMMs run W4A4 on FP4 tensor cores; dispatched activations are MXFP4. | None recorded (#752, #802, #938, #969, #1200). | GSM8K. | gate |
| ATOM online FP8: `--online_quant_config '{"global_quant_config":"ptpc_fp8","exclude_layer":[...]}'` | Quantizes at load time, per token and per channel, the linears the checkpoint left in BF16 (attention projections, dense layers), and runs them on FP8 GEMMs. | None recorded. The ATOM 0630 change carrying it was reverted once, together with the index-reuse override (#945), and relanded without that override (#952). | GSM8K. Check that the exclude list matches this model's module names: the patterns differ per model (`*block_sparse_moe` for M3, #952; `*.mlp.gate`/`*expert*` for GLM-5.2, #1208; `conv1d`/experts/vision for Kimi-K3, main). The DP-attention variant also excludes `*.gate.*` (#990). | gate |
| Checkpoint or quantization swaps: `--quantization deepseek_v4_fp8` on gfx942; FP8-activation checkpoint `amd/gpt-oss-120b-w-mxfp4-a-fp8` | Runs DSv4 on a SKU without FP4 (#1011, #1020). The main comments disagree on how: the header says the FP4 experts are dequantized to FP8; the MoE note says they stay MXFP4 and run W4A16. Or swaps the checkpoint for one with FP8 activations (#396). | None recorded. | GSM8K; read the server log for the MoE backend actually selected. | fixed |
| `--quantization fp8` on an NVFP4 checkpoint (NVIDIA GLM-5 B200/B300 MTP drafts) | The launch line was "copied verbatim" from the FP8 recipe (#501, #504). | None. | Prove what quantization the server loaded before trusting any number. | fixed |
| `--linear-backend emulation` (AMD vLLM, MXFP8) | Dequantizes MXFP8 weights and runs BF16 hipBLASLt GEMMs instead of native MXFP8 kernels: different rounding. | GSM8K 0.9613 flexible / 0.9621 strict measured with it on (#972). | GSM8K when switching on a new model. | gate |
| `SGLANG_OPT_FP8_WO_A_GEMM=0` (AMD SGLang DSv4) | Dequantizes `wo_a` to BF16 when `deep_gemm` is missing (fallback). | None recorded (#798); removed on v0.5.14 (#998). | GSM8K. | gate |
| NVIDIA prefill query quantization: `use_prefill_query_quantization` (FlashInfer MLA / sparse-MLA) | Quantizes the prefill attention queries. | None recorded (#1026, #1061, #1129, #1155). | GSM8K plus a long-context sample. | gate |
| Indexer-cache dtype: `use_fp4_indexer_cache` / `--enable-deepseek-v4-fp4-indexer` (NVIDIA), `indexer_kv_dtype fp8` (NVIDIA), ATOM `--index-cache-dtype fp4` (AMD) | Lower-precision sparse-attention indexer keys, which can change which tokens each query attends to. | None recorded (#532, #899, #1200; ATOM: commit c2b58030b, whose record #1223 says "FP8" while the script passes `fp4`). Omitted on H200: no FP4 (#530). | GSM8K plus a long-context check. Selection errors grow with context length. | gate |

## 4. KV cache

| Setting | What it changes | Accuracy evidence | Check | Pack |
| --- | --- | --- | --- | --- |
| `--kv-cache-dtype fp8` / `fp8_e4m3` (vLLM, SGLang), `--kv_cache_dtype fp8` (ATOM) | KV stored in FP8: half the bytes, twice the capacity. Nearly universal in the recipes. | **Failed** on gfx942 MiniMax-M3: the checkpoint has no calibrated q/prob scales for ROCm FP8 attention, and vLLM's fallback scale of 1.0 "corrupts model accuracy". Kept BF16 (#846, #886; main MI325X MTP comment). ATOM M3 FP4: FP8 KV tripped a `k_scale` assertion (no KV scales) until the 0623 image (#874, #926). Unresolved: the later gfx942 agentic M3 recipes use FP8 KV (#1172 MI325X, #1175 MI300X; main) with no accuracy record of the scale issue. GLM-5.2 FP8 MI325X keeps BF16 KV (#1141; reason not stated). | Check that the checkpoint ships KV scales (`k_scale`/`v_scale`; q/prob scales for ROCm FP8 attention); then GSM8K. | fixed |
| Kernels that require FP8 KV | `SGLANG_AITER_MLA_PERSIST=1` is built for an FP8 KV cache (#279); the AITER page-16 sparse paged attention needs FP8 KV (#1050). | Turning these on drags the KV dtype with them. | As above. | fixed |

## 5. Compute dtype and math mode

| Setting | What it changes | Accuracy evidence | Check | Pack |
| --- | --- | --- | --- | --- |
| `--dtype float16` instead of BF16 (AMD) | 16-bit activations, GEMM outputs, attention and norms in FP16: more mantissa, less range. Llama-70B FP8 MI300X/MI325X: "float16 performs better than ... bfloat16" (#194, #199). | None recorded. | GSM8K; watch for overflow (inf/NaN) on long contexts. | fixed |
| `VLLM_FLOAT32_MATMUL_PRECISION=high` (NVIDIA) | TF32 tensor cores for FP32 matmuls (MiniMax's FP32 router GEMM). | None recorded (#513-#518, #860, #870). Not used in AMD recipes; gfx950 has no TF32 (REPORT §4.3). | GSM8K; routing changes can flip expert choice. | gate |
| Mamba/SSM state dtype: `SGLANG_MAMBA_SSM_DTYPE=bfloat16`, `--mamba-ssm-dtype bfloat16` | Recurrent state of linear-attention layers in BF16 instead of FP32: half the memory and bandwidth (#1025 MI355X; #382, #384 NVIDIA). | None recorded. Kernel constraints override the choice: Hopper's flashinfer `gated_delta_rule_mtp` verify kernel asserts float32 and aborts graph capture on BF16 (#1251, #1237). SM100's flashinfer linear-attention decode requires BF16 (#1239). | GSM8K plus a long-context sample: state error accumulates over tokens. | gate |

## 6. Architecture overrides (FLOP reductions)

| Setting | What it changes | Accuracy evidence | Check | Pack |
| --- | --- | --- | --- | --- |
| `SGLANG_OPT_C4_SPARSE_TOPK=512` (AMD SGLang DSv4, #798) | Sets a sparse top-k for DSv4's compressed attention; the records do not say whether it overrides the model's own top-k. | None recorded. | Before changing it, compare with the model config's top-k; if it differs, treat it like the next row. | gate |
| `--hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}'` (or under `text_config`) | Reuses the sparse-attention indexer's top-k selection across 4 layers instead of recomputing it per layer. The model attends to different tokens. | No score recorded. Disallowed by InferenceX's `PR_REVIEW_CHECKLIST.md` because it reduces model-architecture FLOPs: removed from DSv4 ATOM (#959) and MiniMax-M3 ATOM (#945); deliberately not used in #1050 and #1089. Still present at main in the MiniMax-M3 FP4 vLLM agentic recipe (73bd8e200, #1165). | Treat as disallowed unless the pack says otherwise. If used: GSM8K plus a long-context retrieval check. | no |

## 7. Kernel paths with recorded correctness failures

These were meant as speedups or plain kernel choices, not numerics changes, and still changed
outputs. A kernel or layout switch that only "should" be exact gets the check too.

| Path | Failure | Resolution | Evidence |
| --- | --- | --- | --- |
| `VLLM_ROCM_USE_AITER=1`, MiniMax-M3 MXFP8, TP-only, nightly-4559c43a | Degenerate output. | Switch gated on EP (#935); fixed by vllm#47158 (nightly-09663abd), GSM8K 0.9613 (#972). The M3 MTP script's later comment names EP as the degenerate case (17046f446); unverified, since its rows are TP-only. | #935, #972 |
| `VLLM_ROCM_USE_AITER_RMSNORM=1` at TP<8, Kimi-K2.5 FP4 | "accuracy issues" | Set to 0 when TP<8. | #425 |
| EP8 on MiniMax-M3 MXFP8, MI300X vLLM | Garbage output (and 434 vs 905 tok/s/GPU at c256). | EP1. | #982 |
| EP8 with DP-attention on DSv4 SGLang MI355X | GSM8K 0.9318 -> 0.0000 once EP was actually applied. | EP removed from the DP-attention rows. | #595 |
| SGLang image v0.5.12-20260517, GLM-5.1 FP4 at TP=2 | GSM8K ~0.32 (sglang#25742). | v0.5.12.post1-20260529: 0.975. | #779 |
| MoRI FP8 combine at high concurrency | GSM8K far below 0.95. | Per-phase combine dtypes. | #609, #766 |
| MTP flags on decode only, SGLang PD disagg v0.5.14 | GSM8K 0.96 -> 0.88: decode's draft state was never seeded from prefill. No error, no slowdown. | Flags on both roles; v0.5.15 rejects the mismatch. | #1059 |
| DSv4 MTP on the ROCm HIP-radix backend under CUDA graph | Per-step draft `out_cache_loc` slicing bug: false EOS, truncated generations (sgl#20404). | sglang#26383 (GSM8K 0.950); images from the amd/deepseek_v4 branch lacked it. | #798 |
| `--enable-aiter-allreduce-fusion` with TP2/EP2 EAGLE (Qwen3.5 MXFP4) | "unstable ... rank consistency" | Left off for that recipe. | #1168 |
| FP8 KV without calibrated scales on gfx942 | Accuracy corrupted by the 1.0 fallback scale. | BF16 KV. | #846, #886 |
| NVIDIA TRT-LLM fused mHC kernel (`TRTLLM_MHC_ENABLE_FUSED_HC`) | Corrupted DSv4 generations at hidden size 7168. | Off (#592); back on with a fixed build (#607). | #592, #607 |
| NVIDIA TRT-LLM DSR1 FP8 GB300 | Numeric issue: GSM8K 88% vs 95%, plus MTP performance anomalies. | Newer TRT-LLM release. | #872 |
| NVIDIA FlashInfer autotune on decode workers | Accuracy-unstable kernel picks. | Autotune off "for accuracy stability". | #569 |

## 8. Speculative decoding

| Setting | What it changes | Accuracy evidence | Check | Pack |
| --- | --- | --- | --- | --- |
| Synthetic acceptance: `SGLANG_SIMULATE_ACC_LEN` / `_METHOD` / `_TOKEN_MODE`; vLLM `rejection_sample_method: synthetic` + `synthetic_acceptance_length`; TRT-LLM `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS` | Forces the acceptance length. Emitted tokens are not the model's verified output. | Throughput-only by design: every agentic recipe switches it off for evals (main comments; #1185 "stripped for evals"). It also hides draft quality: on Kimi-K3 B300 the real acceptance length on agentic traces was 1.16-2.01 against 4.18 on short context, and with real verification MTP ran 20-63% slower than no MTP at c4/c8, while the reported numbers used a synthetic 2.51 (#1082). | Never on during an accuracy check. A change to draft depth or drafter under synthetic acceptance measures cost only, not benefit (REPORT §2.4). | no |
| Draft depth (EAGLE/MTP steps, draft tokens) | Should be lossless with correct verification. | #1059's comments call 2-step EAGLE on DSv4 disagg "lossy (gsm8k ~0.85)" against ~0.96 at depth 1. A later disagg arm runs depth 3 on v0.5.15.post1 with no accuracy record (#1154). | GSM8K with real verification at the depth you set. A drop means an implementation fault (§7), not an accepted trade. | gate |
| Spec flags per role in PD disagg | Draft state must be produced on prefill and shipped to decode. | 0.96 -> 0.88 with decode-only flags (#1059). | Flags on both roles; GSM8K. | gate |
| Drafter attention backend pin (`"attention_backend": "TRITON_ATTN"` in `--speculative-config`) | Speed only. | Acceptance unchanged (#977). | none beyond the usual | gate |
| Benchmark prompt encoding (not a server knob) | Changes measured acceptance, not outputs. | Without the chat template, MTP acceptance was "artificially high" (#315). Raw random tokens "silently regress" EAGLE/MTP acceptance (main DSv4 MTP comments). | Benchmark MTP with the model's chat encoding (`--use-chat-template`, `--dsv4`). | n/a |

## 9. Not numerics (for contrast)

These keep the math, but they swap kernels, so one GSM8K run at the KEEP's layout is still cheap
insurance (§7):
- all-reduce + RMSNorm fusion (#944, #1255);
- fused QK-norm + RoPE + KV write (#1255);
- shared-expert fusion (#935);
- shuffled KV layout (#611);
- num-continuous-decode-steps (#596);
- graph modes (#756);
- TBO (#1067);
- GPU_MAX_HW_QUEUES (#877).

Settings that change outputs without being numerics: `SGLANG_DEFAULT_THINKING` and
`SGLANG_DSV4_REASONING_EFFORT` (`max` in the fixed-seq-len DSv4 recipes, `high` in the agentic
ones; main) change what the model generates and so the workload. Keep them identical across arms.
