# MoE dispatch path on ROCm (sglang DSv2 / DSv4)

Static analysis of `sglang` @ b4fe0246c.
DSv4 reuses `DeepseekV2MoE` for its sparse layers
(`models/deepseek_v4.py:1796` → `models/deepseek_v2.py:362`), so the
dispatch below applies to both.

## Layer object hierarchy

`DeepseekV2MoE` (`models/deepseek_v2.py:362`) holds:
- `self.gate` (a `MoEGate`-like linear) — uses `aiter_dsv3_router_gemm`
  on HIP (see router GEMM below).
- `self.topk` — `TopKConfig` configured `BiasedGroupedTopK` (DSv4 uses
  `n_group`, `topk_group`, `correction_bias`).
- `self.experts` — `FusedMoE` (TP path) or `DeepEPMoE` / `MoriEPMoE` /
  `NpuFuseEPMoE` (a2a paths).

Router GEMM dispatch (`models/deepseek_v2.py:333-357`):
```python
# Inside MoEGate.forward
if (_is_cuda and hidden_states.shape[0] <= 16 and hidden_states.shape[1] == 7168
        and (self.weight.shape[0] in (256, 384)) and _device_sm >= 90):
    if _device_sm in [100, 103] and self.weight.shape[0] == 256:
        flashinfer_dsv3_router_gemm(logits, hidden_states, self.weight)
    else:
        logits = dsv3_router_gemm(hidden_states, self.weight, out_dtype=torch.float32)
elif _use_aiter:
    logits = aiter_dsv3_router_gemm(hidden_states, self.weight)   # :355
else:
    logits = F.linear(hidden_states, self.weight, None)
```
`aiter_dsv3_router_gemm` is one line
(`layers/rocm_linear_utils.py:9-14`):
```python
def aiter_dsv3_router_gemm(hidden_states, weight):
    return tgemm.mm(hidden_states, weight.detach(), otype=hidden_states.dtype)
```

## `forward_normal` (`deepseek_v2.py:703-793`)

```python
def forward_normal(self, hidden_states, ...):
    if hidden_states.shape[0] > 0:
        if not self._fuse_shared_experts_inside_sbo:
            shared_output = self._forward_shared_experts(hidden_states, ...)   # :726
        router_logits = self.gate(hidden_states, gemm_output_zero_allocator)    # :730
        topk_output = self.topk(hidden_states, router_logits, ...)              # :732
    else:
        shared_output = None
        topk_output = self.topk.empty_topk_output(hidden_states.device)

    # SBO hook path (alt stream)…  :742-770

    final_hidden_states = self.experts(hidden_states, topk_output)              # :772
    # `routed_scaling_factor` is folded into `biased_grouped_topk` on aiter,
    # so the explicit scale is SKIPPED whenever _use_aiter is True:
    if (not _is_cuda and not _is_musa and not _is_xpu
            and not _use_aiter                              # :780  ←  skip-on-aiter
            or isinstance(self.experts.quant_method, KTEPWrapperMethod)):
        final_hidden_states *= self.routed_scaling_factor   # :784
    if shared_output is not None:
        final_hidden_states += shared_output
    if self.tp_size > 1 and not should_skip_post_experts_all_reduce(...):
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)  # :792
    return final_hidden_states
```

## TopK dispatch (`layers/moe/topk.py`)

```python
# topk.py:83
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

# topk.py:136-141  (aiter imports)
if _use_aiter:
    from aiter import biased_grouped_topk as aiter_biased_grouped_topk
    from aiter.fused_moe import fused_topk as aiter_fused_topk

# topk.py:547-557  (BiasedGroupedTopK call site, scoring_func == "sigmoid")
if _use_aiter and correction_bias is not None:
    aiter_biased_grouped_topk(
        gating_output, correction_bias.to(dtype=gating_output.dtype),
        topk_weights, topk_ids,
        num_expert_group=1, topk_group=1, need_renorm=renormalize)
else:
    topk_sigmoid(topk_weights, topk_ids, gating_output, renormalize,
                 correction_bias, ...)
```

DSv4-specific topk reference lives at `layers/moe/deepseek_v4_topk.py:30`
(same `_use_aiter` line). The "topk transform 512" JIT kernel is at
`sglang.jit_kernel.deepseek_v4.topk_transform_512` (imported by
`layers/attention/compressed/indexer.py:10`).

## Expert GEMM0 / activation / GEMM1 on ROCm

Two main runner classes exist; both use `aiter.fused_moe.fused_moe` for
the actual GEMM0→silu→GEMM1→combine fusion.

### A. TP-only `FusedMoE` (`layers/moe/fused_moe_triton/layer.py:139`)

`forward → forward_impl` (`:1030, :1057`):
```python
def forward_impl(self, hidden_states, topk_output):
    dispatch_output = self.dispatcher.dispatch(...)        # :1061
    combine_input = self.run_moe_core(dispatch_output)     # :1065
    with use_symmetric_memory(get_tp_group(), ...):
        final_hidden_states = self.dispatcher.combine(combine_input=combine_input)
    if self.reduce_results and (self.moe_tp_size > 1 or self.moe_ep_size > 1):
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
    return final_hidden_states
```
`run_moe_core` (`:1084`) calls `self.quant_method.apply(...)`, which on
HIP+aiter is the registered fused_func:
```python
# layers/moe/moe_runner/aiter.py:49-93
@register_fused_func("none", "aiter")
def fused_experts_none_to_aiter(dispatch_output, quant_info, runner_config):
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    ...
    output = fused_moe(
        hidden_states=hidden_states,
        w1=quant_info.w13_weight, w2=quant_info.w2_weight,
        topk_weight=topk_weights, topk_ids=topk_ids.to(torch.int32),
        quant_type=getattr(QuantType, quant_info.quant_type.value),
        activation=getattr(ActivationType,
            _AITER_ACTIVATIONS.get(activation, "Gelu")),
        w1_scale=quant_info.w13_scale, w2_scale=quant_info.w2_scale,
        a1_scale=quant_info.a13_scale, a2_scale=quant_info.a2_scale,
        bias1=quant_info.b13, bias2=quant_info.b2,
        expert_mask=quant_info.expert_mask,
        doweight_stage1=quant_info.doweight_stage1,
        hidden_pad=quant_info.hidden_pad,
        intermediate_pad=quant_info.intermediate_pad)
    return StandardCombineInput(hidden_states=output)
```
`AiterMoeQuantInfo` (`moe_runner/aiter.py:30`) tags `quant_type`. Values
declared as a string enum (`:22-26`):
```python
class AiterQuantType(str, Enum):
    NONE        = "No"
    PER_TOKEN   = "per_Token"
    PER_128X128 = "per_128x128"   # DSv4 default (fp8 block scale)
    PER_1X32    = "per_1x32"      # mxfp4 path
```

### B. DeepEP path (`layers/moe/ep_moe/layer.py:70` `DeepEPMoE`)

```python
# ep_moe/layer.py:54-58 (top-of-file gates)
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
if _use_aiter:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
```

`forward → forward_impl → run_moe_core`
(`:163, :182, :215-260`):
```python
def run_moe_core(self, dispatch_output):
    from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker
    if _use_aiter:
        assert DispatchOutputChecker.format_is_deepep(dispatch_output)
        output = self.forward_aiter(dispatch_output)              # :230
    elif _is_npu:
        output = self.forward_npu(dispatch_output)
    elif DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
        if self.use_w4afp8:
            output = self.forward_cutlass_w4afp8(dispatch_output)
    elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
        if (get_moe_runner_backend().is_flashinfer_cutedsl() and
                self.quant_config.get_name() == "modelopt_fp4"):
            output = self.forward_flashinfer_cutedsl(dispatch_output)
        elif self.use_w4afp8:
            output = self.forward_cutlass_w4afp8_masked(dispatch_output)
```

`forward_aiter` (`ep_moe/layer.py:276-310`):
```python
def forward_aiter(self, dispatch_output):
    hidden_states, topk_ids, topk_weights = (dispatch_output.hidden_states,
        dispatch_output.topk_ids, dispatch_output.topk_weights)
    if hidden_states.shape[0] == 0:
        return hidden_states
    # in original deepep, idx == -1 meaning invalid; aiter does not accept -1.
    # We map invalid to num_local_experts and pass an expert_mask.
    topk_ids_copy = topk_ids.to(torch.int32)
    topk_ids_copy[topk_ids_copy == -1] = self.num_local_experts   # :293
    return fused_moe(
        hidden_states, self.w13_weight, self.w2_weight,
        topk_weights, topk_ids_copy,
        w1_scale=self.w13_weight_scale_inv,
        w2_scale=self.w2_weight_scale_inv,
        quant_type=QuantType.per_128x128,
        activation=(ActivationType.Silu
                    if self.moe_runner_config.activation == "silu"
                    else ActivationType.Gelu),
        expert_mask=self.expert_mask,                              # :309
    )
```

### Mori / NPU EP variants

```python
# ep_moe/layer.py:597-628
class MoriEPMoE(DeepEPMoE):
    def __init__(...):
        super().__init__(...)
        assert _use_aiter, "Mori need to be used together with aiter as of now"

# ep_moe/layer.py:436
class NpuFuseEPMoE(DeepEPMoE):  # Ascend
    ...
```

## Asm MoE path (rocm_moe_utils)

`layers/moe/rocm_moe_utils.py` adapts vLLM's `rocm_aiter_fused_moe`:
```python
# rocm_moe_utils.py:27-49
@register_custom_op(out_shape="hidden_states", eager=_use_aiter)
def rocm_aiter_asm_moe_tkw1(hidden_states, w1, w2, topk_weights, topk_ids,
                            fc1_scale=None, fc2_scale=None,
                            fc1_smooth_scale=None, fc2_smooth_scale=None,
                            a16=False, per_tensor_quant_scale=None,
                            expert_mask=None,
                            activation_method=ActivationMethod.SILU.value):
    from aiter.fused_moe_bf16_asm import asm_moe_tkw1
    return asm_moe_tkw1(hidden_states, w1, w2, topk_weights, topk_ids,
        fc1_scale=fc1_scale, fc2_scale=fc2_scale, ...)
```
Triggered by `tkw1` shapes (topk-weight=1 with router-on-input). Caller
gate is `per_channel_quant and apply_router_weight_on_input and use_fp8_w8a8`
(`rocm_moe_utils.py:91-118`). Specific model paths that hit this are not
identified here.

## How to identify which path a model takes at runtime

Decision tree at MoE call time:
1. `get_moe_a2a_backend()` (server_args). If `.is_none()` → TP path →
   `FusedMoE.forward_impl` → registered `fused_func("none", "aiter")` on
   HIP+aiter.
2. Else (`deepep` / `mori` / `flashinfer` / `flashinfer_cutedsl`) →
   `DeepEPMoE.run_moe_core` → `forward_aiter` on HIP+aiter
   (`ep_moe/layer.py:227-230`).

`SGLANG_USE_AITER` env (`environ.py:323`, default False) gates **all** of
the above on ROCm. Without it, sglang falls back to the Triton fused_moe
runner.

## Not investigated

- `forward_normal_dual_stream` (`deepseek_v2.py:658`) — alternate path.
- `deep_gemm.py` MoE runner (CUDA-only).
- Exact callers of `asm_moe_tkw1`.
- DeepEP cross-node a2a kernel paths.
