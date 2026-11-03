# TRITON Known Issues (auto-extracted)

*198 TODO/FIXME/HACK comments found*

- aiter/ops/triton/_triton_kernels/gmm.py:47: # TODO: Fine tune GMM kernels and use (M, K, N, G) shape to query the best
- aiter/ops/triton/attention/fav3_sage_attention_mxfp4_wrapper.py:88: # TODO: fused quant has perf downgrade
- aiter/ops/triton/attention/fp8_mqa_logits.py:32: # TODO: Currently assuming num_heads and head_size is power of 2.
- aiter/ops/triton/attention/hstu_attention.py:208: # TODO (linjianma): avoid hardcoding the value.
- aiter/ops/triton/attention/lean_atten.py:183: # TODO: add other scenarios
- aiter/ops/triton/attention/mla_decode_rope.py:39: # TODO rope offset
- aiter/ops/triton/attention/pa_decode.py:46: num_seq_partitions: int = 0,  # TODO use this below
- aiter/ops/triton/attention/pa_decode.py:177: #TODO: Add Doc
- aiter/ops/triton/attention/pa_decode.py:278: #TODO: Add Doc
- aiter/ops/triton/attention/pa_decode.py:458: #TODO: Add Doc
- aiter/ops/triton/attention/pa_decode.py:565: #TODO: Add Doc
- aiter/ops/triton/attention/pod_attention.py:152: # TODO: add other scenarios
- aiter/ops/triton/attention/pod_attention.py:195: # TODO: need to tune
- aiter/ops/triton/attention/pod_attention.py:326: # TODO: Support Grouped-Query Attention
- aiter/ops/triton/attention/unified_attention_sparse_mla.py:38: # TODO: This kernel is not optimized and simplified for initial development.
- aiter/ops/triton/gluon/gemm_a8w8_blockscale.py:527: # TODO: need a better way to pass scale block sizes around
- aiter/ops/triton/moe/moe_op.py:118: # TODO: Add support for per token group quantization
- aiter/ops/triton/moe/moe_op_e2e.py:114: #         #TODO: Add support for per token group quantization
- aiter/ops/triton/moe/moe_op_e2e.py:150: # TODO add N_split support to get more parallelism
- aiter/ops/triton/moe/moe_op_gelu.py:111: # TODO: Add support for per token group quantization
- aiter/ops/triton/moe/moe_op_mxfp4.py:137: SWIZZLE_MX_A=swizzle_mx_a,  # TODO add swizzle support
- aiter/ops/triton/moe/moe_op_mxfp4.py:138: SWIZZLE_MX_B=swizzle_mx_b,  # TODO add swizzle support
- aiter/ops/triton/moe/moe_op_mxfp4_silu_fused.py:136: SWIZZLE_MX_A=swizzle_mx_a,  # TODO add swizzle support
- aiter/ops/triton/moe/moe_op_mxfp4_silu_fused.py:137: SWIZZLE_MX_B=swizzle_mx_b,  # TODO add swizzle support
- aiter/ops/triton/moe/moe_op_silu_fused.py:118: # TODO: Add support for per token group quantization
- aiter/ops/triton/rope/rope.py:47: # TODO: For now BLOCK_D is assumed to be power of 2. Expand to handle other value of D.
- aiter/ops/triton/rope/rope.py:77: # TODO: performance optimization
- aiter/ops/triton/rope/rope.py:184: # TODO: performance optimization
- aiter/ops/triton/rope/rope.py:269: # TODO: performance optimization
- aiter/ops/triton/rope/rope.py:382: # TODO: performance optimization