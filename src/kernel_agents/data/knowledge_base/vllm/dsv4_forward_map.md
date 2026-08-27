# DeepSeek V4 / V3.2 forward-path map (V1 engine, ROCm)

**Model file:** `vllm/model_executor/models/deepseek_v4.py` (1589 lines).
**Engine:** V1 (only engine on this checkout — see `v0_v1_engine_differences.md`).
DeepSeek V3.2 ships through the same code path. Lines are from
`vllm-amd` (commit `0072ce09b9`).

## Top-level call chain

```
ForCausalLM.forward                deepseek_v4.py:1564
  Model.forward                    deepseek_v4.py:1317
    for each layer:
      DecoderLayer.forward         deepseek_v4.py:1197
        hc_pre (mhc_pre)           deepseek_v4.py:1168  -> torch.ops.vllm.mhc_pre
        attn_norm (RMSNorm)
        Attention.forward          deepseek_v4.py:1082
          MLA wrapper.forward      deepseek_v4_attention.py:287
        hc_post (mhc_post)         deepseek_v4.py:1188
        ffn_norm (RMSNorm)
        MoE.forward                deepseek_v4.py:856
        hc_post (mhc_post)
    norm (RMSNorm)
  lm_head -> logits
```

`hc_pre` / `hc_post` are HC (head-coupling) residual ops, lazy-imported
from `vllm/model_executor/layers/mhc.py` at deepseek_v4.py:1103
(registers `torch.ops.vllm.mhc_pre` / `mhc_post`).

`Model.forward` body (deepseek_v4.py:1324-1338):

```python
hidden_states = self.embed_input_ids(input_ids)
hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)
if self.use_mega_moe:
    input_ids = input_ids.to(torch.int64)
for layer in islice(self.layers, self.start_layer, self.end_layer):
    hidden_states = layer(hidden_states, positions, input_ids)
# Stash pre-hc_head residual for the MTP draft (captured copy_).
num_tokens = hidden_states.shape[0]
self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))
```

## Attention block — `DeepseekV4Attention` (deepseek_v4.py:923-1088)

Linear-layer construction (deepseek_v4.py:961-1005):

```python
# Padded to min 64 heads for FlashMLA, initialized to -inf (no sink effect).
padded_heads = max(self.n_local_heads, 64)
self.attn_sink = nn.Parameter(
    torch.full((padded_heads,), -float("inf"), dtype=torch.float32),
    requires_grad=False,
)

self.fused_wqa_wkv = MergedColumnParallelLinear(
    self.hidden_size, [self.q_lora_rank, self.head_dim],
    bias=False, quant_config=quant_config,
    prefix=f"{prefix}.fused_wqa_wkv",
    disable_tp=True,                            # fused ReplicatedLinear
)
self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
self.wq_b = ColumnParallelLinear(self.q_lora_rank,
    self.n_heads * self.head_dim, bias=False, ...)
self.kv_norm = RMSNorm(self.head_dim, self.eps)
self.wo_a = ColumnParallelLinear(
    self.n_heads * self.head_dim // self.n_groups,
    self.n_groups * self.o_lora_rank, bias=False, ...)
self.wo_a.is_bmm = True                          # marker for FP8 BMM lowering
self.wo_a.bmm_batch_size = self.n_local_groups
self.wo_b = RowParallelLinear(...)
self.rotary_emb = get_rope(...)                  # L1026; deepseek_yarn / llama_scaling
```

Sparse indexer is only added for C4A layers (deepseek_v4.py:1033-1046):

```python
self.indexer = None
if self.compress_ratio == 4:
    # Only C4A uses sparse attention and hence has indexer.
    self.indexer = DeepseekV4Indexer(...)
```

### MLA wrapper — `DeepseekV4MultiHeadLatentAttentionWrapper.forward`

Defined at `vllm/model_executor/layers/deepseek_v4_attention.py:113`,
forward at L287. Per call:

```python
# deepseek_v4_attention.py:295-309
num_tokens = hidden_states.shape[0]
o_padded = torch.empty(
    (num_tokens, self.padded_heads, self.head_dim),
    dtype=hidden_states.dtype, device=hidden_states.device,
)
# Attention (inside custom op for torch.compile boundary)
torch.ops.vllm.deepseek_v4_attention(
    hidden_states, positions, o_padded, self.layer_name,
)
o = o_padded[:, : self.n_local_heads, :]
```

The custom op is registered as a torch.library op
(`deepseek_v4_attention.py:580-585`):

```python
direct_register_custom_op(
    op_name="deepseek_v4_attention",
    op_func=deepseek_v4_attention,         # calls self.attention_impl(...)
    mutates_args=["out"],
    fake_impl=deepseek_v4_attention_fake,
)
```

Its `attention_impl` (deepseek_v4_attention.py:416) runs:

1. **`attn_gemm_parallel_execute(hidden_states)`** (L356-414): runs
   `fused_wqa_wkv` on default stream and overlaps three lighter input
   GEMMs (compressor kv_score, indexer.weights_proj,
   indexer.compressor.kv_score) on aux streams 0/1/2 via
   `execute_in_parallel`. Gate (L410-411):
   ```python
   enable = hidden_states.shape[0] <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD
   ```
   **On ROCm, `aux_stream_list is None`** (deepseek_v4.py:1249-1253),
   so this runs serially — no overlap of those four GEMMs.

2. **`fused_q_kv_rmsnorm`** (L430-436): fused Q+KV RMSNorm. aiter on
   ROCm, custom kernel on CUDA.

3. **`wq_b + KV-insert + (optional) MLA compressor`**: two
   sibling functions selected by `if self.indexer is not None`:
   ```python
   # deepseek_v4_attention.py:451-470 (sparse layer with indexer)
   def wq_b_kv_insert_and_compress() -> torch.Tensor:
       q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
       self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
       compressor(kv_score, positions, self.rotary_emb)
       return q
   ```

4. **Fused QNorm-RoPE-KV-Quant-Insert** via the HIP/CUDA C++ kernel
   (deepseek_v4_attention.py:548-557):
   ```python
   # Horizontally fused:
   #   Q side:  q_head_norm (per-head RMSNorm, no weight) + GPT-J RoPE
   #   KV side: GPT-J RoPE + UE8M0 FP8 quant + paged cache insert
   torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
       q, kv, swa_kv_cache_2d, swa_metadata.slot_mapping,
       positions.to(torch.int64), self.rotary_emb.cos_sin_cache, ...
   )
   ```
   Backed by `csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`.

5. **Pad q to padded_heads** (L515-517) if `n_local_heads < padded_heads`.

6. **`self.mla_attn(q, kv, positions, output=out)`** (L521): dispatches
   to the selected MLA backend (`ROCM_AITER_MLA` /
   `ROCM_AITER_MLA_SPARSE` / `TRITON_MLA`; see
   `attention_backend_dispatch.md`).

### O-projection diverges by platform — deepseek_v4_attention.py:311-352

ROCm currently keeps a BF16 reference path:

```python
# deepseek_v4_attention.py:311-322 — ROCm branch
# "Keep ROCm on the BF16 reference wo_a path util kernel ready."
if current_platform.is_rocm():
    z = rocm_inv_rope_einsum(
        self.rotary_emb, o, positions,
        self.rope_head_dim, self.n_local_groups,
        self.o_lora_rank, self.wo_a,
    )
    return self.wo_b(z.flatten(1))
```

CUDA uses the fused FP8 inverse-RoPE + einsum kernel
(deepseek_v4_attention.py:325-352):

```python
o_fp8, o_scale = fused_inv_rope_fp8_quant(o, positions,
    self.rotary_emb.cos_sin_cache,
    n_groups=self.n_local_groups,
    heads_per_group=self.n_local_heads // self.n_local_groups,
    nope_dim=self.nope_head_dim, rope_dim=self.rope_head_dim,
    tma_aligned_scales=self._tma_aligned_scales,
)
torch.ops.vllm.deepseek_v4_fp8_einsum(
    o_fp8, o_scale, wo_a_fp8, wo_a_scale, z,
    "bhr,hdr->bhd",                          # einsum string baked in
    list(self._einsum_recipe),
)
return self.wo_b(z.flatten(1))
```

The fused FP8 einsum is registered at deepseek_v4_attention.py:612-617
(`direct_register_custom_op(op_name="deepseek_v4_fp8_einsum", ...)`).
**Known ROCm divergence** — the fused FP8 inverse-RoPE + einsum kernel
is CUDA-only on this checkout.

### Sparse indexer path (C4A only)

`DeepseekV4Indexer` forward at `deepseek_v4_attention.py:1187`. Calls
`vllm/model_executor/layers/sparse_attn_indexer.py:84` (custom op).
The top-k step branches by platform (sparse_attn_indexer.py:323-358):

```python
if current_platform.is_cuda() and topk_tokens in (512, 1024, 2048):
    workspace_manager = current_workspace_manager()
    (topk_workspace,) = workspace_manager.get_simultaneous(
        ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
    )
    torch.ops._C.persistent_topk(
        logits, seq_lens, topk_indices, topk_workspace,
        topk_tokens, attn_metadata_narrowed.max_seq_len,
    )
else:
    # ROCm/XPU fall through here; calls torch.ops._C.top_k_per_row_decode
    torch.ops._C.top_k_per_row_decode(
        logits, next_n, seq_lens, topk_indices, num_rows,
        logits.stride(0), logits.stride(1), topk_tokens,
    )
```

Output is `topk_indices_buffer[:, :topk_tokens]` consumed by the MLA
sparse backend.

## MoE block — `DeepseekV4MoE` (deepseek_v4.py:706-921)

Two execution modes selected by `use_mega_moe`:

```python
# deepseek_v4.py:1229-1237 — gate
self.use_mega_moe = (
    vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
)
if self.use_mega_moe and not vllm_config.parallel_config.enable_expert_parallel:
    raise NotImplementedError(
        "DeepSeek V4 MegaMoE currently requires expert parallel. ..."
    )
```

- **Default (`use_mega_moe = False`)** — `_init_fused_moe_experts`
  (L825-854) builds a single `FusedMoE` from
  `vllm/model_executor/layers/fused_moe/layer.py`. Internal dispatch
  picks Triton vs aiter via `rocm_aiter_ops.is_fused_moe_enabled()`
  (`fused_moe/layer.py:362-364`; see `moe_and_quant_kernels.md`).
- **`use_mega_moe = True`** — `_init_mega_moe_experts` (L799-823) uses
  `DeepseekV4MegaMoEExperts` (L391). Requires expert parallel and
  `expert_dtype == "fp4"` (L741-746). deep_gemm-backed; CUDA-side.

Routing for mega path (deepseek_v4.py:867-880):

```python
topk_weights, topk_ids = fused_topk_bias(
    hidden_states=hidden_states,
    gating_output=router_logits,
    scoring_func=self.scoring_func,
    e_score_correction_bias=self.gate.e_score_correction_bias.data
        if self.gate.e_score_correction_bias is not None else None,
    topk=self.n_activated_experts,
    renormalize=self.renormalize,
    indices_type=self.hash_indices_dtype,
    input_tokens=input_ids,
    hash_indices_table=self.gate.tid2eid,
    routed_scaling_factor=self.routed_scaling_factor,
)
```

For the default path, routing is internal to `FusedMoE`
(`is_internal_router`, L901). **Hash MoE layers** (first
`config.num_hash_layers` decoder blocks) replace topk with a precomputed
`tid2eid` table (L757-771).

**Shared experts**: a separate `DeepseekV4MLP` (L784-792). With
`VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1`, this is folded into the
aiter `fused_moe` call (`fused_moe/layer.py:362-382`).

## Per-layer aux-tensor (HC) parameters

Each `DecoderLayer` carries `hc_attn_fn`, `hc_ffn_fn`,
`hc_attn_base`, `hc_ffn_base`, `hc_attn_scale`, `hc_ffn_scale`
(deepseek_v4.py:1125-1166). Used only via `mhc_pre`/`mhc_post`:

```python
# deepseek_v4.py:1175-1186
post_mix, res_mix, layer_input = torch.ops.vllm.mhc_pre(
    residual=x, fn=hc_fn,
    hc_scale=hc_scale, hc_base=hc_base,
    rms_eps=self.rms_norm_eps,
    hc_pre_eps=self.hc_eps,
    hc_sinkhorn_eps=self.hc_eps,
    hc_post_mult_value=self.hc_post_alpha,
    sinkhorn_repeat=self.hc_sinkhorn_iters,
)
```

## Kernels you'll see in a ROCm decode trace (ordered)

1. `mhc_pre` (Triton/lazy)
2. `rmsnorm` (aiter) — attn_norm
3. `fused_wqa_wkv` GEMM (hipBLASLt or aiter linear)
4. `fused_q_kv_rmsnorm`
5. `wq_b` GEMM
6. `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` (HIP)
7. MLA attention kernel — varies by backend (aiter `mla_decode_fwd` / triton MLA)
8. `rocm_inv_rope_einsum` (ROCm BF16 fallback; **NOT** fused FP8 einsum)
9. `wo_b` GEMM
10. `mhc_post`
11. `ffn_norm`
12. Gate GEMM + topk
13. Fused MoE (aiter `rocm_aiter_fused_moe` or Triton `fused_experts`)
14. AllReduce (custom QR or aiter shim)

## Not investigated

- The exact CUDA-only path of `deepseek_v4_fp8_einsum` (kernel
  source) — known to be NVIDIA-only by the runtime gate.
- MTP (`deepseek_mtp.py`, `deepseek_v4_mtp.py`) forward; this file
  documents only the base model.
- Spec-decode interaction (`deepseek_eagle.py` / `deepseek_eagle3.py`).
