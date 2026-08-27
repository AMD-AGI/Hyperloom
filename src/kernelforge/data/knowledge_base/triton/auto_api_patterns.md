# TRITON API Patterns (auto-extracted)

*754 kernel functions found*

## triton (754)

- `_silu_exp2` (${KA_WORKSPACE}/kernel-agents-workspace/aiter/aiter/ops/triton/_triton_kernels/activation.py)
  Params: `x`
- `_silu` (${KA_WORKSPACE}/kernel-agents-workspace/aiter/aiter/ops/triton/_triton_kernels/activation.py)
  Params: `x`
- `_tanh` (${KA_WORKSPACE}/kernel-agents-workspace/aiter/aiter/ops/triton/_triton_kernels/activation.py)
  Params: `x`
- `_gelu` (${KA_WORKSPACE}/kernel-agents-workspace/aiter/aiter/ops/triton/_triton_kernels/activation.py)
  Params: `x`
- `_gelu_tanh` (${KA_WORKSPACE}/kernel-agents-workspace/aiter/aiter/ops/triton/_triton_kernels/activation.py)
  Params: `x`
- `_relu` (${KA_WORKSPACE}/kernel-agents-workspace/aiter/aiter/ops/triton/_triton_kernels/activation.py)
  Params: `x`
- `_apply_activation_from_str` (${KA_WORKSPACE}/kernel-agents-workspace/aiter/aiter/ops/triton/_triton_kernels/activation.py)
  Params: `x, activation: tl.constexpr`
- `_act_mul_and_dynamic_mxfp4_quant_kernel` (${KA_WORKSPACE}/kernel-agents-workspace/aiter/aiter/ops/triton/_triton_kernels/activation.py)
  Params: `x_ptr,
    x_fp4_ptr,
    bs_ptr,
    stride_x_m_in,
    stride_x_n_in,
    stride_x_fp4_m_in,
    s`
- `_act_mul_and_dynamic_fp8_group_quant_kernel` (${KA_WORKSPACE}/kernel-agents-workspace/aiter/aiter/ops/triton/_triton_kernels/activation.py)
  Params: `x_ptr,
    x_fp8_ptr,
    x_bs_ptr,
    stride_x_m_in,
    stride_x_n_in,
    stride_x_fp8_m_in,
   `
- `_causal_conv1d_fwd_kernel` (${KA_WORKSPACE}/kernel-agents-workspace/aiter/aiter/ops/triton/_triton_kernels/causal_conv1d.py)
  Params: `# continuous batching
    # Pointers to matrices
    x_ptr,  # (dim, cu_seqlen`
- `_causal_conv1d_update_kernel` (${KA_WORKSPACE}/kernel-agents-workspace/aiter/aiter/ops/triton/_triton_kernels/causal_conv1d.py)
  Params: `# Pointers to matrices
    x_ptr,  # (batch, dim, seqlen`
- `_load_unshuffle_segment` (${KA_WORKSPACE}/kernel-agents-workspace/aiter/aiter/ops/triton/_triton_kernels/gather_kv_b_proj.py)
  Params: `base_ptr,
    seg_idx,
    HeadDim: tl.constexpr,
    PaddedHeadDim: tl.constexpr,
    KV_CDim: tl.c`
- `_triton_gather_kv_b_proj` (${KA_WORKSPACE}/kernel-agents-workspace/aiter/aiter/ops/triton/_triton_kernels/gather_kv_b_proj.py)
  Params: `batch_size,
    k_buffer,  # [num_block, block_size, kv_c_dim + kv_pe_dim]
    k_scale,  # [1] or No`
- `_remap_xcd_tile_grid` (${KA_WORKSPACE}/kernel-agents-workspace/aiter/aiter/ops/triton/_triton_kernels/gmm.py)
  Params: `tile_in_mm,
    num_row_tiles,
    num_col_tiles,
    GROUP_SIZE: tl.constexpr = 1,
    NUM_XCDS: tl`
- `gmm_kernel` (${KA_WORKSPACE}/kernel-agents-workspace/aiter/aiter/ops/triton/_triton_kernels/gmm.py)
  Params: `# Tensor pointers:
    lhs_ptr,
    rhs_ptr,
    group_sizes_ptr,
    out_ptr,
    bias_ptr,
    # T`

## MFMA Instructions Used

- `mfma_scale_f32_16x16x128_f8f6f4`
- `mfma_i32_16x16x64_i8`
- `mfma_layout`
- `mfma_scaled`
- `mfma_f32_16x16x32_fp8_fp8`
- `mfma_i32_16x16x32_i8`
- `mfma_out`
- `mfma_layout_a`
- `mfma_layout_b`
- `mfma_q`
- `mfma_k`
