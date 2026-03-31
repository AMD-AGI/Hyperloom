import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_gemm_a16w16_mean_mul_pow_rsqrt_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': '56705705E4E18AA14DD94A2CF51212D7F933A5891109FEF7FC153255E87F5862', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'is_hip': True, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'add_persistent_rblock': True, 'tiling_scores': {'x': 0, 'r0_': 256000}}
)
@triton.jit
def triton_red_fused__to_copy_add_gemm_a16w16_mean_mul_pow_rsqrt_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4
    r0_numel = 5120
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex

    # Since R0_BLOCK (8192) >= r0_numel (5120), the loop executes exactly once.
    # We eliminate the second loop entirely by doing everything in a single pass.

    # Step 1: Load in_ptr0 and in_ptr1 ONCE
    r0_index = r0_base
    r0_mask = r0_index < r0_numel

    tmp0 = tl.load(in_ptr0 + (r0_index + 5120 * x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_index + 5120 * x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)

    # Step 2: Compute residual in fp32
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3  # residual sum in fp32

    # Step 3: Store residual to out_ptr1 (as bf16)
    tmp9 = tmp4.to(tl.float32)
    tl.store(out_ptr1 + (r0_index + 5120 * x0), tmp9, r0_mask & xmask)

    # Step 4: Compute sum of squares for RMSNorm
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp6_masked = tl.where(r0_mask & xmask, tmp6, 0.0)
    tmp7 = tl.sum(tmp6_masked, 1)[:, None]

    # Step 5: Compute rsqrt(mean(x^2) + eps)
    tmp15 = 5120.0
    tmp16 = tmp7 / tmp15
    tmp17 = 1e-06
    tmp18 = tmp16 + tmp17
    tmp19 = tl.rsqrt(tmp18)

    # Step 6: Load weight (in_ptr2) and normalize
    tmp21 = tl.load(in_ptr2 + (r0_index), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp22 = tmp21.to(tl.float32)

    # Step 7: x * rsqrt_val * weight
    tmp20 = tmp4 * tmp19
    tmp23 = tmp20 * tmp22

    # Step 8: Store normalized output as bf16
    tmp24 = tmp23.to(tl.float32)
    tl.store(out_ptr2 + (r0_index + 5120 * x0), tmp24, r0_mask & xmask)
