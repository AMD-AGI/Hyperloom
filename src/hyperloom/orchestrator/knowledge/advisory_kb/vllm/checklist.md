# vLLM — Static-recon checklist (un-bridged capability patterns)

Patterns the static-recon specialist hunts: a fast path that *should* be on for this
(model, GPU, precision) but is silently disabled. gpu/precision gating via `applies_when`;
the folder already gates framework=vllm.

## rocm.fp8.cutlass_only_guard
- kind: checklist
- source: vllm#45854, session:Qwen3-32B/20260622T032133Z
- applies_when: gpu=rocm, precision=fp8
- domain_hint: freeform
- source_dirs: vllm/model_executor/layers/quantization/
- consequence: Per-tensor scales disqualify every AITER fp8 kernel (AiterHipbMM/AiterPerToken/AiterPreshuffled require per-token act + per-channel weight), so dense GEMMs fall back to bf16 rocm_unquantized_gemm / per-tensor torch._scaled_mm.
- bridge: On the ROCm+AITER fp8 Linear path select per-token activation (kFp8DynamicTokenSym) + per-channel weight (kFp8StaticChannelSym), and make the online weight quant per-channel, so dense Linears route to the AITER fp8 GEMM.
- detect: grep for `cutlass_fp8_supported` usage in vllm/model_executor/layers/quantization/. On ROCm it is CUDA-only (returns False), so `Fp8LinearMethod.__init__` falls to per-tensor activation + per-tensor weight scales. Confirm the dense Linear path lands on per-tensor scales rather than per-token/per-channel.

## rocm.fp8.aiter_linear_disabled
- kind: checklist
- source: vllm#45854
- applies_when: gpu=rocm, precision=fp8
- domain_hint: freeform
- source_dirs: vllm/model_executor/layers/quantization/, vllm/model_executor/kernels/linear/
- consequence: With AITER linear disabled the per-token/per-channel fp8 GEMM selection never triggers even when scales are correct, leaving dense Linears on the slower scaled_mm / bf16 path.
- bridge: Enable the AITER linear path (env/flag) and confirm AiterHipbMMPerTokenFp8ScaledMMLinearKernel is selected; pair with the per-channel scale bridge above.
- detect: grep for `is_linear_enabled` / `is_linear_fp8_enabled` (vllm/_aiter_ops.py) and the AITER linear env gates (VLLM_ROCM_USE_AITER_LINEAR / _LINEAR_HIPBMM). Confirm whether the AITER dense-linear fp8 path is gated off for the current run.

## rocm.mxfp8.smallm_dispatch_gap
- kind: checklist
- source: vllm#46063
- applies_when: gpu=rocm, precision=mxfp8
- domain_hint: kernel_switch_specialist
- source_dirs: vllm/model_executor/kernels/linear/mxfp8/, vllm/model_executor/layers/fused_moe/experts/
- consequence: Without small-M dispatch, low-concurrency decode MXFP8 GEMMs run the Triton dot_scaled kernel which is weight-bandwidth/occupancy bound at small M, leaving decode TPOT on the table.
- bridge: Add a try-import dispatch to the AITER small-M MXFP8 GEMM/grouped GEMM (guarded by the AITER master switch and a None-fallback to Triton) on the non-EP decode path.
- detect: grep for `dot_scaled` / MXFP8 native linear+grouped-GEMM dispatch (rocm_native.py, mxfp8_native_moe.py). Confirm whether a low-M (decode) path tries an AITER small-M HIP kernel before falling back to the Triton dot_scaled kernel.

## rocm.moe.aiter_backend_activation_gap
- kind: checklist
- source: vllm#46419
- applies_when: gpu=rocm, precision=*
- domain_hint: kernel_switch_specialist
- source_dirs: vllm/model_executor/layers/fused_moe/
- consequence: An unsupported activation/pad config makes the AITER MoE backend self-reject, so MoE runs the slower Triton/unfused path even when --moe-backend aiter is requested.
- bridge: Add the model's activation to `_supports_activation` and thread the required pad / GateMode config so the AITER MoE backend accepts it.
- detect: For MoE models, grep the MoE backend selection (fused_moe/oracle/*.py, rocm_aiter_moe.py) and `_supports_activation`. Confirm whether the model's activation (e.g. SWIGLUOAI_UNINTERLEAVE) and pad config are accepted by the AITER MoE backend, or silently rejected so it falls back to a slower backend.

## rocm.moe.shared_expert_fusion
- kind: checklist
- source: vllm#46545, MiniMax-M3-shared-expert-fusion-MI355X-mxfp8
- applies_when: gpu=rocm, precision=mxfp8
- domain_hint: freeform
- source_dirs: vllm/model_executor/layers/fused_moe/, vllm/model_executor/models/, python/sglang/srt/layers/moe/, python/sglang/srt/models/
- consequence: A separate shared-expert MLP adds one extra GEMM launch per MoE layer during decode. At low-to-medium concurrency this makes decode launch-bound, degrading throughput significantly (validated: up to +20-30% at concurrency 1, +6-11% at concurrency 64 on MiniMax-M3 MXFP8 MI355X).
- bridge: Fold the shared expert into the routed grouped-GEMM path as an always-selected extra expert slot: (1) append shared expert ids to the router top-k selection, (2) pass n_shared_experts to FusedMoE so it adjusts expert count, (3) load shared expert weights into the routed expert weight tensor at the end, (4) handle MXFP8 native MoE bin count to match actual weight rows. A/B gate with <FUSE_FLAG>=0 vs 1 (confirm actual env-flag name from the framework build or generated patch; do not treat env-only no-op as KEEP). Require accuracy gate; check for routed scale compensation to avoid double-counting shared expert output. Reference: vLLM PR #46545, upstream MiniMax-M3 shared-expert fusion.
- detect: For MoE models with always-on shared experts (n_shared_experts / num_shared_experts in config.json), grep whether the shared expert still runs as a separate dense MLP per layer. In vLLM: check vllm/model_executor/models/ for a `shared_experts` forward call outside FusedMoE, and vllm/model_executor/layers/fused_moe/ for whether n_shared_experts is passed to FusedMoE or handled separately. Anti-signatures (do NOT proceed): expert parallelism enabled, non-uniform precision between shared and routed experts, prefill-only or high-concurrency-only workload.

## rocm.fp4.moe_tuned_tile_config_gap
- kind: checklist
- source: cph-perf-tuning:KNOWLEDGE.md#step2.2-missing-tuned-moe-configs, session:gpt-oss-120b/20260729T193315Z
- applies_when: gpu=rocm, precision=fp4
- domain_hint: kernel_switch_specialist
- source_dirs: vllm/model_executor/layers/fused_moe/configs/, vllm/model_executor/layers/fused_moe/
- consequence: A missing tuned tile config for the (E, N, gfx950, fp4) combination makes the fused-MoE grouped GEMM run generic Triton tiles that are weight-bandwidth/occupancy bound at decode M=1, which is the dominant weighted op for this workload — leaving decode throughput and TPOT on the table.
- bridge: Generate + commit the tuned tile config via `benchmarks/kernels/benchmark_moe.py --num-experts <E> --intermediate-size <N> --dtype fp4 --device <gfx950>` so the MoE path selects the tuned config instead of the generic Triton fallback. Confirm the AITER master switch is on (see rocm.fp4.aiter_master_switch_gap) so the AITER MXFP4 MoE kernel is eligible. A/B gate on validated throughput; require accuracy gate.
- detect: For MXFP4/FP4 MoE models (e.g. GptOssForCausalLM), grep the tuned fused-MoE tile-config directory (vllm/model_executor/layers/fused_moe/configs/) for a file keyed to THIS model+GPU: `E=<num_experts>,N=<intermediate_size>,device_name=<gfx950/MI355X>,dtype=fp4`. Confirm whether a matching tuned config exists, or the MoE grouped-GEMM falls back to generic Triton tiles. Also confirm the AITER MXFP4 MoE path (rocm_aiter_fused_experts / MoeFlatmm) is actually selected at the run's decode M (concurrency=1 → M=1).

## rocm.fp4.aiter_master_switch_gap
- kind: checklist
- source: cph-perf-tuning:KNOWLEDGE.md#step0.2.1-aiter-master-switch, session:gpt-oss-120b/20260729T193315Z
- applies_when: gpu=rocm, precision=fp4
- domain_hint: kernel_switch_specialist
- source_dirs: vllm/model_executor/layers/quantization/, vllm/model_executor/layers/fused_moe/
- consequence: With the AITER master switch off, MXFP4 MoE and dense GEMMs run the slower Triton / rocm_unquantized fallback even though tuned AITER kernels exist for gfx950, capping throughput regardless of other tuning.
- bridge: Set `VLLM_ROCM_USE_AITER=1` (plus `VLLM_ROCM_USE_AITER_MOE=1` where applicable) and confirm the AITER MXFP4 MoE / GEMM kernels are selected. Pair with a compatible attention backend (NOT ROCM_AITER_FA for gpt-oss). A/B gate on validated throughput.
- detect: Confirm the AITER master switch `VLLM_ROCM_USE_AITER=1` is set for the run. It is OFF by default and gates ALL AITER GEMM / RMSNorm / MoE kernels — and is required even when `--attention-backend` is set explicitly (that flag only changes the attention kernel, not the GEMM/MoE path). Grep `_aiter_ops.py` / the AITER enable gates and confirm the fp4 MoE + dense-GEMM paths are not silently on the non-AITER fallback. Known gpt-oss caveat: `ROCM_AITER_FA` is incompatible with attention-sink models, so do NOT use it to enable AITER — use the master switch + a compatible attention backend.
