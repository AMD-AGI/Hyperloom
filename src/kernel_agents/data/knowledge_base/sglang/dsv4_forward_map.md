# DeepSeek-V4 forward path map (sglang `amd/deepseek_v4`)

Static analysis of `sglang` @ b4fe0246c
(`amd/deepseek_v4`, 2026-05-04). Model class = `DeepseekV4ForCausalLM`.
This is the model the team is actively shipping kernels for.

## Top-down call chain

```
DeepseekV4ForCausalLM.forward          deepseek_v4.py:2337
└── DeepseekV4Model.forward             deepseek_v4.py:2162  (5 alt CUDA streams for CUDA only, :2108-2110)
    └── DeepseekV4DecoderLayer.forward  deepseek_v4.py:1987
        ├─ hc_pre  (Hash-Coding pre-norm; replaces RMSNorm pre)  :881..1140 → :1875..1940
        ├─ input_layernorm (RMSNorm)                              :2002
        ├─ self_attn = MQALayer.forward                           :1667
        ├─ hc_post                                                :2010 → :1942..1985
        ├─ post_attention_layernorm                               :2015
        ├─ DP-gather / MoE dispatch (see moe_dispatch_path.md)    :2017..2076
        └─ self.mlp = DeepseekV2MoE                               :1796 → deepseek_v2.py:362
```

Alt-stream init is CUDA-only (`deepseek_v4.py:2108-2110`):
```python
self.alt_streams = (
    [torch.cuda.Stream() for _ in range(5)] if (_is_cuda) else None
)
```

## `hc_pre` branches (`deepseek_v4.py:1875..1940`)

```python
# Priority 1: TileLang (env default True)
if envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get():
    from sglang.srt.layers.mhc import mhc_pre
    post, comb, y = mhc_pre(residual=x, fn=hc_fn, hc_scale=hc_scale,
        hc_base=hc_base, rms_eps=self.rms_norm_eps, hc_pre_eps=self.hc_eps,
        hc_sinkhorn_eps=self.hc_eps, hc_post_mult_value=2.0,
        sinkhorn_repeat=self.hc_sinkhorn_iters)
    return y, post.squeeze(-1), comb               # :1875-1890

# Priority 2: HIP+aiter (env default True)
if _is_hip and envs.SGLANG_OPT_USE_AITER_MHC_PRE.get():
    from aiter.ops.mhc import mhc_pre
    ...same signature...                            # :1892-1907

# Priority 3: DeepGEMM (env default True)
if envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get():
    import deep_gemm
    deep_gemm.tf32_hc_prenorm_gemm(x_flat, hc_fn.float().contiguous(),
        d_out, s_out, num_splits=None)              # :1909-1924

# Fallback: Torch naive hc_pre_torch_impl                  :1927
```

`hc_post` (`:1942..1985`) mirrors the same priority order:
TileLang `mhc_post` → HIP+aiter `aiter.ops.mhc.mhc_post` → Torch impl.

Env defaults (`environ.py:561-593`):
```python
SGLANG_OPT_DEEPGEMM_HC_PRENORM   = EnvBool(True)
SGLANG_OPT_USE_TILELANG_MHC_PRE  = EnvBool(True)
SGLANG_OPT_USE_TILELANG_MHC_POST = EnvBool(True)
SGLANG_OPT_USE_AITER_MHC_PRE     = EnvBool(True)
SGLANG_OPT_USE_AITER_MHC_POST    = EnvBool(True)
SGLANG_OPT_FP8_WO_A_GEMM         = EnvBool(False)
SGLANG_HACK_FLASHMLA_BACKEND     = EnvStr("kernel")   # but adapter overrides to "tilelang" on HIP
```

Effective on HIP, TileLang wins (first branch evaluated true). To force
aiter mhc, set `SGLANG_OPT_USE_TILELANG_MHC_PRE=0`.

## MQALayer internals (attention hot block)

```python
# deepseek_v4.py:1591  _forward_prepare
q, _ = self.wq_a(x)                              # q_lora_rank
q = self.q_norm(q)                               # RMS
q_lora = q                                       # kept for indexer
q, _ = self.wq_b(q)
q = q.view(-1, self.n_local_heads, self.head_dim)
q = rms_normalize_triton(q, self.eps)            # layers/layernorm.py
kv, _ = self.wkv(x); kv = self.kv_norm(kv)
fused_rope(q[..., -self.qk_rope_head_dim:],
           kv[..., -self.qk_rope_head_dim:].unsqueeze(1),
           self.freqs_cis, positions=positions)  # → aiter on HIP
# indexer / compressor side-effects (writes K-cache, builds indices)
if self.indexer is not None:    self.indexer(x=x, q_lora=q_lora, ...)
if self.compressor is not None: attn_backend.forward_core_compressor(...)
```

`MQALayer.forward` (`:1667-1758`) wraps prepare + core attn + inverse
rope + wo_a/wo_b:
```python
o = attn_backend.forward(q=q_padded if q_padded is not None else q,
                         k=kv, v=kv, layer=self.attn_mqa, ...)   # :1712
fused_rope(o[..., -self.qk_rope_head_dim:], None, self.freqs_cis,
           positions=positions, inverse=True)                    # :1724
# Output proj — FP8 vs bf16 einsum:
if _FP8_WO_A_GEMM:                                               # :1734
    import deep_gemm
    o_fp8, o_s = sglang_per_token_group_quant_fp8(..., group_size=128)
    deep_gemm.fp8_einsum("bhr,hdr->bhd", (o_fp8, o_s),
        (self.wo_a.weight, self.wo_a.weight_scale_inv.data),
        output, recipe=(1,1,128))                                # :1744-1750
else:
    o = torch.einsum("tgd,grd->tgr", o, wo_a)                    # :1754
o, _ = self.wo_b(o.flatten(1))                                   # :1756
```
where `_FP8_WO_A_GEMM = envs.SGLANG_OPT_FP8_WO_A_GEMM.get()`
(`deepseek_v4.py:93`, default False).

`compressor` / `indexer` are set up by `compress_ratio`
(`deepseek_v4.py:1292-1399`):
```python
compress_ratio = config.compress_ratios[layer_id]   # 0, 4, or 128
if compress_ratio:                                  # :1386
    self.compressor = Compressor(..., compress_ratio=compress_ratio)
if compress_ratio == 4:                             # :1398
    self.indexer = C4Indexer(...)
```
`C4Indexer.forward` (`:1226-1246`) delegates to
`attn_backend.forward_c4_indexer(...)`.

## Where hot kernels actually land on ROCm

| step | resolves to (HIP path) | source path |
| ---- | ---------------------- | ----------- |
| `hc_pre` (default, both env True) | `sglang.srt.layers.mhc.mhc_pre` (TileLang) | `deepseek_v4.py:1875` |
| `hc_pre` (Tilelang off, aiter on) | `aiter.ops.mhc.mhc_pre` | `deepseek_v4.py:1893` |
| `hc_post` (default) | `sglang.srt.layers.mhc.mhc_post` (TileLang) | `deepseek_v4.py:1962` |
| `input_layernorm` (RMSNorm fwd) | `aiter.rmsnorm2d_fwd as rms_norm` | `layernorm.py:74` |
| add+RMSNorm fused | `aiter.rmsnorm2d_fwd_with_add as fused_add_rms_norm` | `layernorm.py:75` |
| `fused_rope` | `aiter.ops.triton.fused_qk_concat.fused_qk_rope_cat` | `rocm_linear_utils.py:3` |
| router GEMM | `aiter.tuned_gemm.tgemm.mm` via `aiter_dsv3_router_gemm` | `rocm_linear_utils.py:9` |
| MoE expert GEMMs | `aiter.fused_moe.fused_moe` (per_128x128 fp8) | `ep_moe/layer.py:295` |
| core MLA attn (compressed) | `dpsk_v4_fp8_attention_fwd` (TileLang) | `nsa/tilelang_kernel.py:2316` |
| K-cache act_quant (indexer) | `aiter.ops.cache.indexer_k_quant_and_cache` | `nsa/nsa_indexer.py:43` |
| fp8 MQA logits | `aiter.ops.triton.fp8_mqa_logits.fp8_mqa_logits` | `nsa/nsa_indexer.py:625` |

## Defaults set automatically when `model_arch == DeepseekV4ForCausalLM`

```python
# python/sglang/srt/server_args.py:1675-1689
if model_arch in ["DeepseekV4ForCausalLM"]:
    if self.is_attention_backend_not_set():
        self.attention_backend = "compressed"          # → DeepseekV4Backend(Radix)
    if self.page_size is None:
        self.page_size = 256
    if self.kv_cache_dtype == "auto":
        self.kv_cache_dtype = "fp8_e4m3"
```

The `"compressed"` backend resolves to `DeepseekV4BackendRadix` when
`SGLANG_OPT_DPSK_V4_RADIX=True` (default, `environ.py:578`) else
`DeepseekV4Backend` (see `attention_backend_dispatch.md` for the
factory snippet).

## Not investigated

- Per-tensor numeric ranges / scales — needs a live run.
- Per-step wall-time split — static.
- nextn / MTP `DeepseekV4ForCausalLM` path
  (`deepseek_v4_nextn.py`) — not opened.
