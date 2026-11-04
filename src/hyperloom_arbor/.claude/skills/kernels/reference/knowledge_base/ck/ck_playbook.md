# CK Performance Playbook (Composable Kernel)

> Bit-level analysis of optimization techniques in
> [`aiter-amd/3rdparty/composable_kernel`](aiter-amd/3rdparty/composable_kernel) — both
> classic CK (device-instance / gridwise / blockwise / xdlops) and the modern CK_tile DSL.
>
> **Submodule pin**: `composable_kernel @ fdf4bb7fcc984811cef48ce817d89aac064b984a`
> (parent `aiter-amd @ 3cbdcb371b`).
>
> Companion to [`ASM_PERF_PLAYBOOK_V2.md`](ASM_PERF_PLAYBOOK_V2.md) — that doc analyzes
> the prebuilt `.co` ASM kernels; this doc analyzes the C++/HIP CK source they are
> compiled from. Cross-reference citation styles:
>
> - **`[ck:include/...]`** — file path inside `aiter-amd/3rdparty/composable_kernel/`
> - **`[pdf:pN]`** — AMD CDNA4 ISA spec PDF page
> - **`[asm-v2:§N]`** — section in the ASM playbook V2
>
> ## Status
>
> Synthesis is incremental — one slice per commit. Tracker:
>
> | # | Slice | Status |
> |---|---|---|
> | 1 | CK_tile core abstractions (tile_distribution / tile_window / static_distributed_tensor / load/store_tile / space_filling_curve) | ✅ §1 |
> | 2 | CK_tile arch wrappers (waitcnt layouts, MFMA gfx9/gfx950, async buffer, intrinsics) | ✅ §2 |
> | 3 | CK_tile GEMM pipelines (V1–V6, async, comp_async eight-waves, hot-loop scheduler) | ✅ §3 |
> | 4 | CK_tile FMHA (qr_ks_vs canonical, online softmax, masking, splitkv, appendkv, bwd, fp8/MX) | ✅ §4 |
> | 5 | CK_tile fused_moe + gemm_quant + gemm_mx + flatmm + batched_contraction | ✅ §5 |
> | 6 | CK_tile reduce / softmax / norm / rmsnorm2d / add_rmsnorm2d_rdquant / smoothquant / topk / elementwise | ✅ §6 |
> | 7 | Classic CK device-instance (DeviceGemm / Gridwise / Blockwise / XdlopsGemm hierarchy, V1→V2→V3) | ✅ §7 |
> | 8 | Classic CK FMHA + dispatcher + library | ✅ §8 |
> | 9 | CK_tile epilogue + cshuffle (LDS bank-padding, scale modes, quantization, permute, batched_transpose) | ✅ §9 |
> | 10 | CK scheduling + codegen + LDS patterns (sched_group_barrier patterns, iglp_opt, inline-asm) | ✅ §10 |
> | 11 | CK build + codegen + instance archive (10–20k instances, FMHA Python codegen, Dockerfile.aiter) | ✅ §11 |
> | 12 | Cross-cutting summary: technique catalog + reading order + use-case index | ✅ §12 |
> | 13 | Grouped convolution (fwd / bwd-data / bwd-weight) + image_to_column + pooling | ✅ §13 |
> | 14 | Backward pipelines beyond FMHA (RMSNorm / LayerNorm bwd, conv bwd variants, FMHA-bwd siblings) | ✅ §14 |
> | 15 | Host launcher + StreamK + sparse_attn + topk_softmax + standalone permute/transpose + tensor_adaptor | ✅ §15 |
> | 16 | Perf-engineering toolkit + MI300 / MI355 specifics (occupancy, cache hints, swizzle, magic-div, ILP, prefetch) | ✅ §16 |
>
> *Slices 13–15 added after the §1–§12 gap review identified missing op families and host machinery. Slice 16 added after a second review surfaced thin coverage of perf-engineering primitives and MI300 / MI355 per-arch features.*
>
> Each slice produces one section below + one commit. Slices may add appendix material
> (LDS layout diagrams, scheduler trace excerpts) under `kernel-analysis/ck-notes/`.

## Table of Contents

- [§1. CK_tile core abstractions](#1-ck_tile-core-abstractions) — tile_distribution, tile_window, static_distributed_tensor, load/store_tile, space_filling_curve
- [§2. CK_tile arch wrappers](#2-ck_tile-arch-wrappers-waitcnt-mfma-async-intrinsics) — waitcnt layouts, MFMA gfx9/gfx950, async buffer, intrinsics
- [§3. CK_tile GEMM pipelines](#3-ck_tile-gemm-pipelines) — V3–V6, comp_async, eight_waves, sched_group_barrier patterns
- [§4. CK_tile FMHA](#4-ck_tile-fmha-flashattention) — qr_ks_vs, online softmax, masking, splitkv, paged, bwd, fp8/MX
- [§5. fused_moe + gemm_quant + gemm_mx + flatmm + batched_contraction](#5-ck_tile-fused_moe--gemm_quant--gemm_mx--flatmm--batched_contraction)
- [§6. reduce / softmax / norm / topk / elementwise / non-cshuffle epilogue](#6-ck_tile-reduce--softmax--norm--topk--elementwise--non-cshuffle-epilogue)
- [§7. Classic CK device-instance hierarchy](#7-classic-ck-device-instance-hierarchy) — Device→Grid→Pipeline→Block→Warp→Thread, V1→V2→V3
- [§8. Classic CK FMHA / dispatcher / library](#8-classic-ck-fmha--dispatcher--library) — why classic FMHA is gone, registry + heuristic
- [§9. CShuffle epilogue + LDS bank-padding](#9-cshuffle-epilogue--lds-bank-padding--quantpermute)
- [§10. Scheduling + LDS-layout patterns](#10-scheduling--lds-layout-patterns-cross-pipeline) — iglp_opt, sched_barrier, mask catalog
- [§11. Build + codegen + instance archive](#11-build--codegen--instance-archive)
- [§12. Cross-cutting catalog + reading order](#12-cross-cutting-technique-catalog--reading-order)
- [§13. Grouped convolution + image_to_column + pooling](#13-grouped-convolution--image_to_column--pooling)
- [§14. Backward pipelines (beyond FMHA bwd)](#14-backward-pipelines-beyond-fmha-bwd)
- [§15. Host launcher + StreamK + sparse_attn + topk_softmax + tensor adaptors](#15-host-launcher--streamk--sparse_attn--topk_softmax--standalone-permutetranspose--tensor-adaptors)
- [§16. Perf-engineering toolkit + MI300 / MI355 specifics](#16-perf-engineering-toolkit--mi300--mi355-specifics)

---

<!-- SLICE-1-INSERT -->
## §1. CK_tile core abstractions

The CK_tile DSL is built on five interlocking abstractions. Together they let a writer
express *"thread T owns these elements of this tile, accessed in this order"* as a
zero-cost compile-time computation — every coordinate, stride, and bank-mapping resolves
before codegen.

### 1.1 `tile_distribution_encoding` — the lattice spec

The encoding is a 4-tuple `(Rs, Hss, Ps2RHss, Ys2RHs)` that describes how a tile is
laid out across **R** (replication), **H** (hierarchical span), **P** (partition / lane
or warp), and **Y** (per-thread element) dimensions. Every other CK_tile abstraction
indexes through tables it precomputes.

1. **Encoding fields**
   [tile_distribution_encoding.hpp:19-41](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_distribution_encoding.hpp#L19) —
   `RsLengths`, `HsLengthss`, `Ps2RHssMajor/Minor`, `Ys2RHsMajor/Minor`. Compile-time
   arrays replace runtime conditionals on thread ownership.
2. **R = replication, H = merged span, P = partition, Y = thread element**
   [tile_distribution_encoding.hpp:42-48](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_distribution_encoding.hpp#L42)
   — `NDimR/NDimX/NDimP/NDimY` give rank along each axis. P axes map to lanes/warps,
   Y axes to VGPRs.
3. **`does_p_own_r_` ownership lookup**
   [tile_distribution_encoding.hpp:201-227](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_distribution_encoding.hpp#L201)
   — precomputes which partition owns which R dim. Replaces dynamic `if (lane & mask)`
   with array lookup.
4. **`ps_over_rs_derivative_` flattening stride**
   [tile_distribution_encoding.hpp:229-261](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_distribution_encoding.hpp#L229)
   — change in flat partition index per R-dim step. Index math is `+stride` instead
   of integer divide.
5. **`rhs_major_minor_to_ys_` reverse map**
   [tile_distribution_encoding.hpp:100-112](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_distribution_encoding.hpp#L100)
   — R/H (major,minor) → Y index. Enables thread-local indexing without per-call search.
6. **`distributed_spans_lengthss_` for span granularity**
   [tile_distribution_encoding.hpp:168-186](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_distribution_encoding.hpp#L168)
   — per-partition span lengths (contiguous work units). Enables conflict-free LDS
   banking by aligning span boundaries to bank rows.
7. **`unmerge` transform from H tree**
   [tile_distribution_encoding.hpp:264-273](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_distribution_encoding.hpp#L264)
   — H is a tuple-of-sequences; `unmerge` unpacks a flat X index into nested H indices
   without modulo.
8. **Per-partition footprint via `get_uniformed_p_dim_lengths_over_h()`**
   [tile_distribution_encoding.hpp:276-305](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_distribution_encoding.hpp#L276)
   — accumulates stride across H only (skips R). Tile-boundary checks become a single
   compare against a constant.
9. **Y-loop ordering via `get_sorted_y_to_h_info()`**
   [tile_distribution_encoding.hpp:427-430](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_distribution_encoding.hpp#L427)
   — sorts Y dims by parent H. Row-major Y iteration → minimal bank conflicts on
   register-resident tiles.
10. **Embed / reduce ops preserve encoding structure**
    [tile_distribution_encoding.hpp:457-560](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_distribution_encoding.hpp#L457),
    [564-794](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_distribution_encoding.hpp#L564)
    — remap via `in2out_rh_major/minor` instead of recompiling the encoding.
11. **Prefix-sum cache for flat offsets**
    [tile_distribution_encoding.hpp:321-337](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_distribution_encoding.hpp#L321)
    — cumulative H lengths so `offset = prefix[h_major] + h_minor` is O(1).

### 1.2 `tile_window` — the load/store engine

A tile_window pairs a tensor view (DRAM/LDS) with a distribution and acts as the DMA
front-end. All optimizations here aim to **amortize address arithmetic** across an
inner loop.

1. **`pre_computed_coords_` amortization**
   [tile_window.hpp:81-149](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_window.hpp#L81)
   — caches `(WindowAdaptorCoord, BottomTensorCoord)` per `NumCoord` batch. One
   prepare call hides descriptor lookups across the warp.
2. **`move_window_adaptor_and_bottom_tensor_thread_coordinate()` = single chained DMA bump**
   [tile_window_base.hpp:128-144](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_window_base.hpp#L128)
   — updates [p,y]→[x] and [x]→offset in sequence; one call instead of two.
3. **Iteration order picked by `SFC_Ys`**
   [tile_window_base.hpp:196-202](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_window_base.hpp#L196)
   — space-filling-curve instance over Y lengths with scalar-per-access. Toggle
   row-major / snake at type level.
4. **Automatic `Traits::VectorDimY` selection**
   [tile_window_base.hpp:152-174](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_window_base.hpp#L152)
   — scans Y dims for stride-1 max length. Packs scalars per VGPR; avoids per-scalar
   load.
5. **`NumAccessPerCoord = NumAccess / NumCoord` for unroll balance**
   [tile_window.hpp:63](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_window.hpp#L63),
   [tile_window.hpp:238](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_window.hpp#L238)
   — `static_for` unroll factor that trades register reuse vs I-cache footprint.
6. **Fused offset: compile-time `idx_ys_offset` + runtime `linear_off`**
   [tile_window.hpp:324-413](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_window.hpp#L324)
   — load avoids recomputing two separate GCDs; only one runtime add into the
   immediate offset field.
7. **`oob_conditional_check` template flag enables DCE of bounds logic**
   [tile_window.hpp:154-162](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_window.hpp#L154)
   — `if constexpr (oob_check)` removed at compile time on bulk-aligned paths.
8. **`load_raw()` vector-reinterpret**
   [tile_window.hpp:435](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_window.hpp#L435)
   — `reinterpret_cast<vectorized_tbuf&>` lets one `buffer_load_dwordx4` fill several
   thread-buffer slots without unpack.
9. **`async_load_raw()` M0 per-warp, per-issue increment**
   [tile_window.hpp:521-565](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_window.hpp#L521)
   — uses M0 SGPR for LDS offset; bumped per issue. Pipelined LDS writes don't
   recompute descriptor offsets.
10. **`load_transpose_with_offset()` applies `Policy::group_func` mid-load**
    [tile_window.hpp:704-771](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_window.hpp#L704)
    — transpose/permute fuses into the load; no double buffer.
11. **`store_raw()` strips offset-recompute path**
    [tile_window.hpp:864-923](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_window.hpp#L864)
    — hardwires `oob_conditional_check=true` and skips `static_move_ys`. Smaller
    code in prefetch pipelines.
12. **Lane-0 warp coords for global→LDS async**
    [tile_window.hpp:83-91](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_window.hpp#L83)
    — `pre_computed_warp_coords_` holds lane-0 descriptor only (warp-uniform path);
    DRAM per-lane coords stay separate.

### 1.3 `static_distributed_tensor` — register-resident tile

The result of a `load_tile`. A `thread_buffer<DataType, kPackedSize>` aliased to the
distribution descriptor so reads/writes through it are constexpr VGPR indexing.

1. **VGPR storage at compile-time size**
   [static_distributed_tensor.hpp:142](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/static_distributed_tensor.hpp#L142)
   — `thread_buffer<DataType, PackedSize>` member; size = `kThreadElementSpaceSize /
   PackedSize` ([line 36](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/static_distributed_tensor.hpp#L36)).
   No stack spill, no LDS allocation.
2. **`get_thread_buffer_size()` for unroll planning**
   [static_distributed_tensor.hpp:65-68](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/static_distributed_tensor.hpp#L65)
   — compile-time scalar count; informs `static_for` bounds.
3. **`operator[](TileDistributedIndices)` is constexpr offset math**
   [static_distributed_tensor.hpp:118-127](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/static_distributed_tensor.hpp#L118)
   — span coords → Y indices → flat VGPR index. No branches.
4. **`get_y_sliced_thread_data()` = view, not copy**
   [static_distributed_tensor.hpp:70-94](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/static_distributed_tensor.hpp#L70)
   — `static_ford` over Y range; compiler lowers to VGPR moves, not memory ops.
5. **`set_y_sliced_thread_data()` for partial-tile updates**
   [static_distributed_tensor.hpp:96-115](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/static_distributed_tensor.hpp#L96)
   — gather-scatter in VGPRs; lets a prefetch residual land without RMW of the whole
   tile.
6. **`is_similiar_distributed_tensor` enables safe in-place reinterpret**
   [static_distributed_tensor.hpp:250-268](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/static_distributed_tensor.hpp#L250)
   — type+buffer-size check, so e.g. FP32→BF16 conversion can reuse the same VGPRs.
7. **`set_tile_if(predicate)` for masked writes**
   [static_distributed_tensor.hpp:188-230](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/static_distributed_tensor.hpp#L188)
   — sweeps spans and conditionally writes; avoids loop-carried predicates.
8. **`PackedSize` from `numeric_traits<DataType>`**
   [static_distributed_tensor.hpp:33-34](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/static_distributed_tensor.hpp#L33)
   — e.g. bf16 → 2 scalars/dword. Divides element-space to size the VGPR array.

### 1.4 `load_tile` / `store_tile` / `async_load_tile`

Thin facades over `tile_window`. The optimization payload is the **buffer-instruction
lowering** in `arch/amd_buffer_addressing*.hpp`.

1. **`load_tile()` dispatches to window**
   [load_tile.hpp:36-41](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/load_tile.hpp#L36)
   — selects linear vs static-distribution paths at template-resolution time.
2. **`buffer_load<16>` → `buffer_load_dwordx4` (128-bit)**
   [amd_buffer_addressing_builtins.hpp:151-176](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp#L151)
   — inline asm with V# in SGPR, offset as `:n` immediate.
3. **`buffer_load<8>` → `buffer_load_dwordx2`**
   [amd_buffer_addressing_builtins.hpp:179-204](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp#L179)
   — halves VGPR pressure for BF16/FP16 paths.
4. **`buffer_load<4>` → `buffer_load_dword`**
   [amd_buffer_addressing_builtins.hpp:207-232](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp#L207)
   — fallback when scalar-per-vector=1.
5. **`buffer_load_if<>` zero-fills OOB without branch**
   [amd_buffer_addressing_builtins.hpp:293](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp#L293)
   — in-instruction range check using V# extent; no `s_cmp/s_cbranch`.
6. **`buffer_load_trait<Bytes, T>::payload_t` packs scalar type → vec**
   [amd_buffer_addressing_builtins.hpp:127-139](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp#L127)
   — `bf16x8` reinterpreted to `fp32x4`; uniform instruction emission.
7. **`pre_nop` template flag emits `s_nop 4` prologue**
   [amd_buffer_addressing_builtins.hpp:164-169](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp#L164)
   — 4-cycle stall to hide previous wave's RAW.
8. **`buffer_store<>` mirror of load**
   [amd_buffer_addressing.hpp:148-149](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/amd_buffer_addressing.hpp#L148)
   — `buffer_store_dwordx4 %0, %1, %2, 0 offen offset:%3`.
9. **`async_load_tile_raw()` → `global_load_lds`**
   [load_tile.hpp:184-194](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/load_tile.hpp#L184)
   — fire-and-forget; vmcnt tracks completion.
10. **`async_load_fence(cnt)` = explicit `s_waitcnt vmcnt(cnt)`**
    [load_tile.hpp:196-199](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/load_tile.hpp#L196)
    — partial drain so FMAs can run while later loads are still in flight.
11. **`load_tile_with_elementwise()` fuses transform into the load**
    [load_tile.hpp:55-64](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/load_tile.hpp#L55)
    — saves the standalone transform pass over VGPRs.
12. **`store_tile()` reuses inverse distribution**
    [store_tile.hpp:23-40](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/store_tile.hpp#L23)
    — register→memory layout re-order happens implicitly via descriptor.

### 1.5 `space_filling_curve` — iteration order

A compile-time iteration order over Y. Choice of order controls L1 / LDS-bank /
register-port locality.

1. **`SnakeCurved` toggle**
   [space_filling_curve.hpp:18-19](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/algorithm/space_filling_curve.hpp#L18)
   — `false` = pure row-major (best for sequential LDS writes); `true` = reverse odd
   rows (best for adjacency reuse).
2. **`get_index(AccessIdx1d)` inverse transform**
   [space_filling_curve.hpp:87-163](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/algorithm/space_filling_curve.hpp#L87)
   — O(NDimY) constexpr math; no modulo at runtime.
3. **`forward_sweep[]` parity cache**
   [space_filling_curve.hpp:122-137](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/algorithm/space_filling_curve.hpp#L122)
   — precomputes which dims reverse per row; replaces `%2` in snake mode.
4. **`get_forward_step(i)` / `get_backward_step(i)`**
   [space_filling_curve.hpp:70-82](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/algorithm/space_filling_curve.hpp#L70)
   — adjacent-index deltas. Epilogues use the backward step for residual sweep.
5. **`ScalarPerVector = ∏ ScalarsPerAccess[i]`**
   [space_filling_curve.hpp:30-31](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/algorithm/space_filling_curve.hpp#L30)
   — total scalars touched per access; controls vector instruction width.
6. **`access_lengths = TensorLengths / ScalarsPerAccess`**
   [space_filling_curve.hpp:33](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/algorithm/space_filling_curve.hpp#L33)
   — divides iteration domain by vector width.
7. **`ordered_access_lengths` = permuted access lengths**
   [space_filling_curve.hpp:35-36](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/algorithm/space_filling_curve.hpp#L35)
   — row-major, column-major, or custom dim order via `dim_access_order`.
8. **`get_num_of_access()` divisibility check**
   [space_filling_curve.hpp:46-53](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/algorithm/space_filling_curve.hpp#L46)
   — fails compile if tile not divisible by access vector — no residual logic needed
   at run time.
9. **`compute_index` unrolled via `static_for`**
   [space_filling_curve.hpp:103-118](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/algorithm/space_filling_curve.hpp#L103)
   — div+mod chain inlined with constant strides.
10. **Final reorder = `dim_access_order × ScalarsPerAccess`**
    [space_filling_curve.hpp:150-151](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/algorithm/space_filling_curve.hpp#L150)
    — accounts for vector packing per dimension in the emitted index.

### 1.6 Cross-cutting techniques (this slice)

- **T1.A — Compile-time distribution descriptor resolves thread-ID → VGPR offset in O(1)**:
  the `does_p_own_r_`, `ps_over_rs_derivative_`, and `rhs_major_minor_to_ys_` tables in
  `tile_distribution_encoding` collapse to constants under template instantiation.
- **T1.B — VGPR-resident tile via `static_distributed_tensor`**: no stack, no LDS for
  the live working set; allocator is the template at
  [static_distributed_tensor.hpp:36](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/static_distributed_tensor.hpp#L36).
- **T1.C — Prefetch amortization via `pre_computed_coords_` + `NumCoord`**: groups
  `NumCoord` accesses behind one `prepare_coords` call
  ([tile_window.hpp:81-149](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_window.hpp#L81)).
- **T1.D — Buffer instructions with immediate offset avoid SGPR spill**: offset is
  emitted as `offset:%3` (immediate)
  ([amd_buffer_addressing_builtins.hpp:167](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp#L167))
  — uses the 12-bit `offset` slot of the buffer-instruction format.
- **T1.E — Space-filling curve + M0-per-warp = LDS bank parallelism**:
  row-major SFC ([line 145](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/algorithm/space_filling_curve.hpp#L145))
  paired with per-warp M0 in `async_load_raw`
  ([tile_window.hpp:524-565](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_window.hpp#L524))
  yields consecutive-bank writes without warp-sync.
- **T1.F — Auto-vectorization through `Traits::VectorDimY`**: stride-1 max-length axis
  is found at compile time
  ([tile_window_base.hpp:152-174](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_window_base.hpp#L152));
  the SFC's `ScalarsPerAccess` then packs scalars into the buffer instruction width.
- **T1.G — DCE-driven bounds-check elimination**: `oob_conditional_check` as a
  `bool` template ([tile_window.hpp:154-162](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_window.hpp#L154))
  lets the compiler drop the entire guard branch on bulk-aligned tiles.

Cross-references: [asm-v2:§3] (buffer descriptor), [asm-v2:§12] (waitcnt counters),
[pdf:p35] (V# layout), [pdf:p150] (buffer instructions).

<!-- SLICE-2-INSERT -->
## §2. CK_tile arch wrappers (waitcnt, MFMA, async, intrinsics)

This layer sits between the DSL (`§1`) and the codegen pipeline. It abstracts the
target-specific bits: which `s_waitcnt` encoding to use, which MFMA opcode width is
legal, and how to issue an `s_barrier` or `ds_bpermute`. Everything is templated on
`amdgcn_target_id`, so a single source file specializes across gfx908/90a/942/950.

### 2.1 `arch.hpp` / `arch_config` — target traits

1. **Target ID enum**
   [arch.hpp:82-109](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/arch.hpp#L82) —
   `amdgcn_target_id` with codes 0x0908/0x090A/0x0942/0x0950; `get_compiler_target()`
   macro expansion is compile-time.
2. **Family / arch tier split**
   [arch.hpp:111-118](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/arch.hpp#L111) —
   `amdgcn_target_family_id` (GFX9/10_3/11/12) + `amdgcn_target_arch_id` (CDNA/RDNA).
   Drives SFINAE traits like `enable_if_target_family_gfx9_t`.
3. **Wave size per target**
   [arch.hpp:127-132](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/arch.hpp#L127) —
   GFX9=64, GFX10.3/11/12=32. `get_warp_size()` reads it; no `__AMDGCN_WAVEFRONT_SIZE`
   probing at runtime.
4. **LDS banks doubled on CDNA4**
   [arch.hpp:1179-1189](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/arch.hpp#L1179)
   — gfx950 returns 64 banks (up from 32). All tile-distribution padding logic reads
   this constant.
5. **SMEM capacity 160 KB on gfx950 vs 64 KB**
   [arch.hpp:1112-1118](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/arch.hpp#L1112)
   — drives static LDS allocation budget; lets pipelines size larger double buffers
   on CDNA4 without runtime check.
6. **Tag-type dispatch (`gfx9_t`, `gfx950_t`, …)**
   [arch.hpp:1137-1173](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/arch.hpp#L1137)
   — `get_device_arch()` returns a tag; overloads pick implementations without
   enum-switching.
7. **SFINAE alias `enable_if_target_id_t`**
   [arch.hpp:386-408](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/arch.hpp#L386)
   — gates gfx950-only builtins at type-check time.
8. **LLVM sched-group masks**
   [arch.hpp:1199-1214](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/arch.hpp#L1199)
   — `LLVMSchedGroupMask` (MFMA/VMEM/DS/TRANS) for `__builtin_amdgcn_sched_group_barrier`
   metadata. Used in pipeline hot loops (§3).

### 2.2 Waitcnt layouts

GCN encodes the wait counter as a single 16-bit immediate; field widths change across
families. CK_tile picks the right layout at compile time and exposes a single
`s_waitcnt<vm, exp, lgkm>()` API.

1. **Legacy (gfx9/10.3/11pre) packing**
   [arch.hpp:936-950](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/arch.hpp#L936)
   — vmcnt split [3:0]+[15:12], lgkm [11:8], expcnt [6:4]; `pack_vm()` rejoins the
   split via `((c & 0xF) << 0) | ((c & 0x30) << 10)`.
2. **GFX11 unified layout**
   [arch.hpp:924-934](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/arch.hpp#L924)
   — vmcnt [15:10], lgkm [9:4], exp [2:0]; contiguous, no split.
3. **GFX12 split into loadcnt/dscnt** (no expcnt)
   [arch.hpp:913-922](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/arch.hpp#L913)
   — uses `s_wait_loadcnt_dscnt` instead of `s_waitcnt`; `HAS_EXP=false`.
4. **`Waitcnt` typedef selects layout**
   [arch.hpp:952-959](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/arch.hpp#L952)
   — single `using Waitcnt = …` so callers stay arch-agnostic.
5. **Constexpr packers**
   [arch.hpp:982-1014](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/arch.hpp#L982)
   — `waitcnt_arg::from_vmcnt<cnt>()` etc.; folds to immediate, static_assert on max.
6. **Max-count tables**
   [arch.hpp:966-979](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/arch.hpp#L966)
   — vm/lgkm/exp ceilings per family; gfx12 lgkm grows to 0x3F.
7. **Single `s_waitcnt<vm,exp,lgkm>()` API**
   [arch.hpp:1017-1034](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/arch.hpp#L1017)
   — emits `__builtin_amdgcn_s_waitcnt()` pre-gfx12 or `s_wait_loadcnt_dscnt %0` on
   gfx12; `"memory"` clobber blocks reorder.
8. **LDS-only tightening**
   [arch.hpp:1035-1039](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/arch.hpp#L1035)
   — `s_waitcnt_lgkm<lgkmcnt>()` clears vm/exp but tightens lgkm only; standard
   post-`ds_read` pattern.
9. **Fused waitcnt+barrier**
   [arch.hpp:1044-1063](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/arch.hpp#L1044)
   — gfx12: `s_wait_loadcnt_dscnt %0\ns_barrier_signal -1\ns_barrier_wait -1`;
   pre-gfx12: separate `s_waitcnt` + `__builtin_amdgcn_s_barrier()`.
10. **Block-sync flavors**
    [arch.hpp:1066-1075](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/arch.hpp#L1066)
    — `block_sync_lds<lgkmcnt>()` vs `block_sync_lds_direct_load<vmcnt>()` for
    DS-vs-async path.
11. **Strategy A — early drain**: emit `s_waitcnt vmcnt(0)` immediately after a
    burst of global loads so MFMAs can issue without per-instruction stall.
12. **Strategy B — lazy paired**: keep `K` async loads in flight, drain only
    `vmcnt(K-1)` before the dependent MFMA, then issue the next batch.
13. **Strategy C — fused barrier**: `s_waitcnt_barrier<vm,exp,lgkm>()` collapses
    drain + workgroup sync into one stall window — paired with LDS double-buffer
    flips.

### 2.3 MFMA wrappers (gfx9 vs gfx950)

CK_tile exposes MFMA through an `amdgcn_mma<…>` template specialized per
`(A, B, C, M, N, K, CompilerTarget)`. The gfx9/gfx950 split is by **K width**:
gfx950 doubles K, halving the loop trip count but doubling input VGPR pressure.

1. **gfx9 16x16x16 fp16→fp32**
   [mfma_gfx9.hpp:35-80](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/mfma_gfx9.hpp#L35)
   — `__builtin_amdgcn_mfma_f32_16x16x16f16(aVec4, bVec4, cVec4, Cbsz, Abid, Blgp)`.
2. **gfx950 16x16x32 fp16→fp32**
   [mfma_gfx9.hpp:94-138](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/mfma_gfx9.hpp#L94)
   — `__builtin_amdgcn_mfma_f32_16x16x32_f16(aVec8, bVec8, cVec4)`. Same C width,
   double K input.
3. **Layout constants gfx9** — `kAMLane=16, kBNLane=16, kABKLane=4, kABKPerLane=4,
   kCMLane=4, kCNLane=16`
   [mfma_gfx9.hpp:60-68](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/mfma_gfx9.hpp#L60).
4. **Layout constants gfx950** — `kABKLane=8, kABKPerLane=8` (doubled)
   [mfma_gfx9.hpp:118-126](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/mfma_gfx9.hpp#L118).
   Same C distribution.
5. **`CtrlFlags::{Cbsz, Abid, Blgp}` passed as immediates** — compile-time, cast
   to `int` at call site; zero runtime cost.
6. **Builtins, not asm strings** — `__builtin_amdgcn_mfma_f32_*` are LLVM intrinsics;
   the AMDGPU backend chooses the right `v_mfma_*` encoding. No inline-asm string
   for MFMA in this layer.
7. **`ext_vector_t<T, N>` for A/B** — compiler vector packing; 4×fp16 = 1 VGPR,
   8×fp16 = 2 VGPRs. No explicit alignment attributes.
8. **`MmaOpFamily::DENSE` tag**
   [mfma_gfx9.hpp:49](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/mfma_gfx9.hpp#L49)
   — used by pipeline dispatchers to route to dense vs sparse paths.
9. **Block metadata `kAMBlock=kBNBlock=1`** — implies one MFMA tile per call;
   outer loop multiplies by the tile-iteration count.
10. **Throughput trade-off** — gfx950 doubles per-instruction work but doubles
    operand-VGPR pressure; allowing fewer waves per CU. Choice between gfx9 and
    gfx950 paths is part of pipeline tuning (§3).
11. **scaled / MX / fp8 / bf16 variants live elsewhere** — current `mfma_gfx9.hpp`
    only exposes fp16→fp32; mfma_f8 / mfma_scale lives in `mfma_*.hpp` siblings
    (covered as needed in §3/§4).

### 2.4 Async buffer loads

`global_load_lds` is the gfx9 instruction that DMAs a dword straight from DRAM into
LDS without going through a VGPR. CK_tile wraps it with the M0-setup discipline.

1. **`async_global_load_lds_dwordxn<num_dwords, pre_nop>()`**
   [amd_buffer_addressing_builtins.hpp:1362-1401](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp#L1362)
   — inline asm `global_load_lds_dwordx4 %1, off offset:0`; LDS dest is implicit via
   M0.
2. **M0 set via `m0_set_with_memory()`**
   [amd_buffer_addressing_builtins.hpp:1333-1343](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp#L1333)
   — needs SGPR; uses `amd_wave_read_first_lane()` to lift VGPR to SGPR; `"memory"`
   clobber stops reorder.
3. **M0 increment between issues** — `m0_inc_with_memory(size_per_issue)`; required
   because M0[17:2]×4 holds the LDS byte offset.
4. **`pre_nop` flag emits `s_nop 4`** — hides one prior MFMA's latency before the
   global_load_lds issues.
5. **LDS write is implicit** — no VGPR operand for destination; lane's LDS offset
   advances automatically per-thread.
6. **Buffer resource (V#) packed as 128-bit SGPR quad**
   [amd_buffer_addressing_builtins.hpp:97-105](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp#L97)
   — `ptr / range / config`; passed as `"s"(res)`.
7. **Fallback path for non-async**
   [amd_buffer_addressing.hpp:188-205](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/amd_buffer_addressing.hpp#L188)
   — `__builtin_amdgcn_raw_buffer_load_b128()` if available; else inline
   `buffer_load_dwordx4 ... offen offset:%3`.
8. **Hybrid addressing — `offen` + 12-bit imm offset** — V# + dynamic v_offset +
   imm constant; covers tile-corner address without extra ALU.
9. **`ds_read_tr16_b64_v4f16` transpose-on-read**
   [amd_buffer_addressing_builtins.hpp:3025-3049](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp#L3025)
   — hardware transpose during LDS read; zero-cost transpose for 16×16 tiles.
10. **`amd_direct_load_global_to_lds`** alternate path
    [amd_buffer_addressing_builtins.hpp:2965-2983](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp#L2965)
    — `buffer_load_dword … lds` form (uses V# instead of `global_*`); useful where
    buffer descriptor already in SGPR.
11. **vmcnt grouping** — burst N loads, then `s_waitcnt vmcnt(N-1)` so first load's
    LDS write is visible while last N-1 still in flight.

### 2.5 DPP / permlane / ds_swizzle / s_barrier / v_pk

1. **`ds_bpermute32` for lane shuffle**
   [utility.hpp:40, 55, 100](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/utility.hpp#L40)
   — `__builtin_amdgcn_ds_bpermute(src_lane << 2, bit_cast<int32_t>(v))`; ~4 cycles.
2. **`warp_shuffle_down/up` wrappers**
   [utility.hpp:31-60](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/utility.hpp#L31)
   — recursive bit_cast for >32-bit values; falls back to chained `ds_bpermute`.
3. **`__builtin_amdgcn_permlane32_swap`**
   [utility.hpp:67](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/utility.hpp#L67)
   — single op returns `int32x2_t`; pair-swap in one instruction instead of two
   shuffles.
4. **Memory clobber on shuffle paths** — defensive against CSE inside loops with
   loop-carried lane dependencies.
5. **`__builtin_amdgcn_s_sleep(1)`**
   [workgroup_barrier.hpp:47](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/workgroup_barrier.hpp#L47)
   — power-friendly busy-wait inside `wait_eq_wave()`.
6. **`flag_to_exec()` / `cmp_lt_to_exec()`**
   [utility.hpp:126-142](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/utility.hpp#L126)
   — `v_cmp_ge_u32 %s_exec_flag, %v_flag, 1`; lifts a per-lane condition into EXEC
   so subsequent VALU ops mask without branching.
7. **Named barrier on gfx12**
   [arch.hpp:1054-1055](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/arch.hpp#L1054)
   — `s_barrier_signal -1 / s_barrier_wait -1`; workgroup-scope, no implicit stall.
8. **`wait_eq_wave()` — only lane-0 of each wave polls**
   [workgroup_barrier.hpp:30-57](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/workgroup_barrier.hpp#L30)
   — broadcasts result via `__shfl`; reduces LDS contention on multi-wave barriers.
9. **`wait_set()` via `atomicCAS`**
   [workgroup_barrier.hpp:68-75](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/workgroup_barrier.hpp#L68)
   — pre-gfx12 critical-section primitive; ~40-cycle `buffer_atomic_cmpswap`.
10. **`amd_wave_read_first_lane()`**
    [amd_buffer_addressing_builtins.hpp:31-86](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp#L31)
    — VGPR→SGPR per-chunk broadcast via `__builtin_amdgcn_readfirstlane`. Needed
    everywhere SGPR-only constraints exist (M0 setup, V# load).
11. **v_pk_* packed math intentionally absent here** — packed FP8/BF16 fused ops
    are emitted by tile-level templates (§3, §4) via MFMA + `v_pk_add_f32` builtins,
    not from this arch layer.

### 2.6 Cross-cutting techniques (this slice)

- **T2.A — Compile-time arch selection**: `WaitcntLayout*` + `enable_if_target_id_t`
  collapse every `#ifdef` into a constexpr template choice; one source builds for
  all gfx9/10/11/12 targets without runtime cost.
- **T2.B — Three waitcnt strategies (early-drain, lazy-paired, fused barrier)**:
  exposed through `s_waitcnt<>`, `s_waitcnt_lgkm<>`, `s_waitcnt_barrier<>`. Pipelines
  pick per-stage; §3 hot-loops mix all three.
- **T2.C — gfx9 vs gfx950 MFMA = K-width trade-off**: doubled K halves loop trip
  count but doubles operand VGPRs; pipeline tuning sees this through a single
  `amdgcn_mma<…, CT>` instantiation.
- **T2.D — `global_load_lds` + M0 discipline**: V→L bypasses VGPRs; the M0
  `set/inc` ritual is encoded once in `amd_buffer_addressing_builtins.hpp` so callers
  cannot drop a serialization step.
- **T2.E — `ds_read_tr*` for free transpose**: pairs with `tile_distribution_encoding`
  rotations so the K→N transpose for matrix-B never touches the ALU.
- **T2.F — Wave-0 polling barrier**: `wait_eq_wave` cuts LDS atomic traffic on
  multi-wave WGs by 1/waveCount before broadcasting via `__shfl`.
- **T2.G — `"memory"` clobbers as scheduling fences**: every M0-set, every
  ds_bpermute, every barrier carries a `: "memory"` clobber so the LLVM scheduler
  treats them as ordering points without explicit NOPs.

Cross-references: [asm-v2:§12] (waitcnt counters), [asm-v2:§9] (MFMA), [pdf:p150]
(buffer instructions), [pdf:p245] (mfma encodings), [pdf:p310] (DPP encoding).

<!-- SLICE-3-INSERT -->
## §3. CK_tile GEMM pipelines

`gemm_pipeline_ag_bg_cr_*` is the family of templates that wires the abstractions of
§1 and §2 into a working matmul. **Naming**: `ag` = A in global, `bg` = B in global,
`cr` = C in registers. Each version (V3–V6, plus `comp_async`, `comp_async_eight_waves`,
`comp_v5`) trades prefetch depth, LDS buffer count, and scheduler aggression.

### 3.1 Pipeline family

1. **`comp_async`** —
   [gemm_pipeline_ag_bg_cr_comp_async.hpp:102-103](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp#L102),
   [:170](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp#L170)
   — `PrefetchStages=2`, `PrefillStages=1`, `GlobalBufferNum=1`, `DoubleSmemBuffer=true`.
   Uses `global_load_lds` to bypass VGPRs entirely.
2. **`comp_async_eight_waves`** —
   [gemm_pipeline_ag_bg_cr_comp_async_eight_waves.hpp:19](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async_eight_waves.hpp#L19)
   — 8-warp WG variant for preshuffled B. Hot-loop count = `MIterPerWarp * NIterPerWarp *
   KIterPerWarp` ([:129](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async_eight_waves.hpp#L129)).
3. **`comp_v3`** —
   [gemm_pipeline_ag_bg_cr_comp_v3.hpp:94](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp#L94)
   — Compute-optimized, sync buffer load + LDS double-buffer.
   Special-cases 8-warp via `GetBlockLoopTailNum`
   ([:26-42](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp#L26)).
4. **`comp_v4`** — Enhanced scheduler; 5-barrier pattern per buffer load
   ([gemm_pipeline_ag_bg_cr_comp_v4.hpp:266-276](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4.hpp#L266)).
5. **`comp_v5`** —
   [gemm_pipeline_ag_bg_cr_comp_v5.hpp:38](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v5.hpp#L38)
   — `PrefetchStages=1`, single LDS buffer, `TailNumber::Empty`. Always-hot-loop;
   no tail-epilogue control flow.
6. **`comp_v6`** —
   [gemm_pipeline_ag_bg_cr_comp_v6.hpp:18-21](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v6.hpp#L18),
   [:93](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v6.hpp#L93)
   — `PrefetchStages=3`, `GlobalBufferNum=2`, `HotloopUnroll=2`. Triple-stage
   scheduler.
7. **Shared base** —
   [gemm_pipeline_ag_bg_cr_base.hpp:12](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp#L12)
   — `GetAWindows`/`GetBWindows` unpack 1, 2, or 3 LDS buffers via tuple
   ([:305-412](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp#L305)).

### 3.2 `comp_async` hot-loop walkthrough

Source: [gemm_pipeline_ag_bg_cr_comp_async.hpp:441-490](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp#L441)
(Intrawave scheduler, `HasHotLoop=true`).

**Prologue** ([:357-438](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp#L357)):
1. Async prefetch 0 → LDS[0]; async prefetch 1 → LDS[1]; advance DRAM `KPerBlock` each.
2. `block_sync_lds_direct_load()` so LDS[0] writes are visible.
3. LocalPrefetch A/B from LDS[0] → register tiles 0.
4. `block_sync_lds()` before LDS[0] overwrite.
5. Async prefetch 2 → LDS[0]; DRAM advance.

**Hot-loop body — ping** ([:449-466](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp#L449)):
6. LocalPrefetch A/B from LDS[1] → register tiles 1.
7. `block_sync_lds()` before LDS[1] overwrite.
8. Async prefetch `i` → LDS[1]; DRAM advance.
9. `Compute C += A(i-3) @ B(i-3)` using register tiles 0 (3-stage skew).
10. `HotLoopScheduler()` injects `C_MFMA_Inst_Num/num_issue - 2` MFMA-intent barriers
    ([:236](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp#L236)).

**Hot-loop body — pong** ([:468-487](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp#L468)):
11. `block_sync_lds_direct_load()` (LDS[0] ready).
12. LocalPrefetch → register tiles 0.
13. `block_sync_lds()`; Async prefetch `i+1` → LDS[0].
14. `Compute C += A(i-2) @ B(i-2)` using register tiles 1.
15. `HotLoopScheduler()`; increment global counter by 2.

**Tail** ([:492-537](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp#L492)):
16. `TailNumber::Three/Two/One` runs final 3/2/1 compute steps using pre-loaded
    registers; `__builtin_amdgcn_sched_barrier(0)` final drain.

### 3.3 `sched_group_barrier` patterns

Mask key (matches `LLVMSchedGroupMask`): `0x008` MFMA, `0x004` SALU/lgkmcnt,
`0x020` VMEM_READ, `0x100` DS_READ, `0x200` DS_WRITE.

1. **Pattern P1 — `comp_async` linear interleave**
   [comp_async.hpp:229-237](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp#L229)
   — `MFMA(1) → DS_READ(1) → MFMA(1) → VMEM_READ(1) → MFMA(6-2)` per buffer-load issue.
2. **Pattern P2 — `eight_waves` LDS-drain prologue**
   [eight_waves.hpp:189-197](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async_eight_waves.hpp#L189)
   — `MFMA×3 → s_waitcnt_lgkm<4> → SALU(1) → MFMA(MFMA_INST-3)`.
3. **Pattern P3 — `comp_v3` A/B dual-stage**
   [comp_v3.hpp:339-365](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp#L339)
   — writes (`DS_WRITE,MFMA`)×N, then `VMEM_READ,MFMA(bulk)`; A and B paths
   independent; finally interleaved DS_READ with `ds_read_a_mfma_rate` count.
4. **Pattern P4 — `comp_v4` unified anchor**
   [comp_v4.hpp:266-276](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4.hpp#L266)
   — `MFMA(1), DS_READ(N/num_issue), MFMA(1), DS_WRITE(M/num_issue), MFMA(1),
   VMEM_READ(1), MFMA(C_MFMA_Inst_Num/num_issue - 3)`.
5. **Pattern P5 — `comp_v6` stage-1 (read prefill)**
   [comp_v6.hpp:277-292](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v6.hpp#L277)
   — adaptive: per-iteration `DS_READ_A(min(rate, rem)), MFMA(1)` until A drained;
   then same for B. `ds_read_a_mfma_rate` computed at compile time
   ([:249-250](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v6.hpp#L249)).
6. **Pattern P6 — `comp_v6` stage-2 (write+VMEM)**
   [comp_v6.hpp:311-332](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v6.hpp#L311)
   — `[DS_WRITE,MFMA]×num_dswrite_per_issue, VMEM_READ(1), MFMA(num_mfma_per_issue
   - num_dswrite_per_issue)`, looped `A_Buffer_Load_Inst_Num` times.
7. **Pattern P7 — `comp_v6` stage-3 (read epilogue)**
   [comp_v6.hpp:335-366](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v6.hpp#L335)
   — symmetric mirror of P5 to drain LDS before tail.

### 3.4 LDS double / triple buffering

1. **`comp_async` double-buffer** — line
   [:170](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp#L170)
   asserts `DoubleSmemBuffer=true`. Two LDS blocks offset 0 and `+smem_size`
   ([:318-320](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp#L318));
   four copy windows + four load windows
   ([:338-344, 411-418](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp#L338)).
2. **`comp_v5` single-buffer** — only one LDS block; ping-pong sits in register
   space ([:120](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v5.hpp#L120)).
3. **`comp_v6` dual global buffer** — `GlobalBufferNum=2`
   ([:18-20](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v6.hpp#L18))
   — two DRAM→LDS async queues feed one LDS block; depth comes from 3-stage register
   prefetch.
4. **Aligned LDS sub-buffer layout** —
   [base.hpp:140-162](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp#L140)
   `a_lds_block_space_size_aligned = integer_least_multiple(size, 16)`; B starts
   after aligned A.
5. **Transpose-load LDS shape** —
   [base.hpp:32-36](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp#L32)
   — when `is_a_load_tr`, swap to `(KPerBlock, MPerBlock)` for `ds_read_tr*`.

### 3.5 Policy / prefetch / k-loop tail

1. **`UniversalGemmBasePolicy`** —
   [comp_async_default_policy.hpp:16](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async_default_policy.hpp#L16),
   [comp_v4_default_policy.hpp:18](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4_default_policy.hpp#L18)
   — common `Make{A,B}LdsBlockDescriptor` and `GetBlockGemm` hooks.
2. **BlockGemm dispatch** —
   [comp_async_default_policy.hpp:112-129](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async_default_policy.hpp#L112)
   — returns `BlockGemmARegBRegCRegV1<Problem, BlockGemmPolicy>` via WarpGemmDispatcher.
3. **`PrefetchStages` semantics**
   - V3/V4/async: 2 → prologue prefetches 0/1/2; loop bumps by 2.
   - V5: 1 → prologue prefetches 0/1; always-hot.
   - V6: 3 → prologue prefetches 0/1/2/3; `HotloopUnroll=2`.
4. **Vector size lookup** —
   [comp_async.hpp:148-156](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp#L148)
   — `GetVectorSize{A,B,C}` resolved at compile-time by policy.
5. **DRAM stride per prefetch** —
   [comp_async.hpp:350-351](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp#L350)
   — `is_a_col_major ? {KPerBlock, 0} : {0, KPerBlock}`; one full K-slice per
   advance.
6. **`TailNumber` enum** —
   [scheduler.hpp:26-46](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp#L26)
   — `Odd/Even` for prefetch=2; `One/Two/Three` for deeper prefetch;
   `Empty/Full` for unroll-driven.
7. **`block_sync_lds_direct_load()` vs `block_sync_lds()`** — direct variant drains
   only vmcnt (post-async-load RAW); generic drains lgkm (post-DS-load).

### 3.6 Cross-cutting techniques (this slice)

- **T3.A — Async load skips VGPRs**: `comp_async` paths use `global_load_lds`
  (§2.4) so the DRAM→LDS transfer never touches the register file.
- **T3.B — `sched_group_barrier` counts derived at compile time**: every count
  (e.g. `C_MFMA_Inst_Num / num_buffer_load_inst`) is a constexpr expression of
  problem geometry — zero runtime cost, the LLVM scheduler honors them.
- **T3.C — Adaptive `ds_read_*_mfma_rate`** (V6):
  `ceil((mfma_cycle - 4 + 2*issue_cycle - 1) / (2*issue_cycle))` — barrier count
  scales inversely with DS_READ width; fp8 (4-cycle) gets more interleaved
  DS_READs than fp16 (8-cycle).
- **T3.D — Register double-buffer with 3-stage skew**: even with single LDS
  buffer, register tiles 0/1 hold A/B from K-iterations `i-3` and `i-2`; MFMA on
  `i-3` while async-prefetch for `i` is in flight.
- **T3.E — Distinct sync flavors for distinct RAW hazards**:
  `block_sync_lds_direct_load()` for vmcnt drain (async path),
  `block_sync_lds()` for lgkm drain (DS path) — picking the right one keeps the
  waitcnt mask tight.
- **T3.F — `ds_read_tr*` paired with LDS-shape swap**: V6+ select transpose-load
  shape `(KPerBlock, MPerBlock)` so DS_READ_TR turns column-major A into
  row-major register tiles without any ALU.
- **T3.G — Preshuffled B for 8-warp WG**: `comp_async_eight_waves` uses
  B-window shape `(NWarps, flatKPerBlock)` to feed 8 warps without gather.
- **T3.H — Intrawave scheduler specialization**: `PipelineImpl<GemmPipelineScheduler::Intrawave>`
  injects `HotLoopScheduler()` only in the intra-wave path; Interwave/default
  paths can opt out.
- **T3.I — Static instruction-count forecast**: `A_Buffer_Load_Inst_Num`,
  `num_ds_read_inst`, `C_MFMA_Inst_Num` resolve at template-instantiation time.
  Scheduler hints get amortized across per-iteration loops automatically.

Cross-references: [asm-v2:§22] (scheduler patterns), [asm-v2:§18] (LDS double-buffer),
[asm-v2:§14] (multi-wave WG), [pdf:p195] (sched_group_barrier semantics).

<!-- SLICE-4-INSERT -->
## §4. CK_tile FMHA (FlashAttention)

FMHA pipelines in `ck_tile/ops/fmha/` are FlashAttention-2 variants, sharing the
**online-softmax** core but specializing for prefill / decode / split-KV / paged /
appendKV / fp8 / MX. The canonical entry is `block_fmha_pipeline_qr_ks_vs.hpp`
(Q in **r**egisters, K & V *streamed* into LDS / **s**hared).

### 4.1 Forward pipeline family

1. **`block_fmha_pipeline_qr_ks_vs`**
   [block_fmha_pipeline_qr_ks_vs.hpp:145](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L145)
   — Q load-once into registers, K/V iterated through LDS. Prefill workhorse.
2. **`qs_ks_vs`** —
   [block_fmha_pipeline_qs_ks_vs.hpp](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qs_ks_vs.hpp)
   — Q re-loaded per K tile (lower register pressure; helps V-prefetch scheduling).
3. **`qr_ks_vs_async`** —
   [block_fmha_pipeline_qr_ks_vs_async.hpp:25](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp#L25)
   — `global_load_lds` for K tiles, double-buffered.
4. **SplitKV** —
   [block_fmha_fwd_splitkv_pipeline_qr_ks_vs.hpp:16](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_qr_ks_vs.hpp#L16)
   — per-block reduces over disjoint K shards `i_split`; emits `Oacc/LSEacc` for
   the combine kernel.
5. **AppendKV** —
   [block_fmha_fwd_appendkv_pipeline.hpp:13](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_appendkv_pipeline.hpp#L13)
   — writes new K/V into existing cache with optional rotary; no softmax.
6. **Paged-KV** —
   [block_fmha_fwd_pagedkv_pipeline_qr_ks_vs.hpp:18](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_pagedkv_pipeline_qr_ks_vs.hpp#L18)
   — block-table indirection via `PageBlockNavigator`.
7. **FP8** —
   [block_fmha_pipeline_qr_ks_vs_fp8.hpp:15](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_fp8.hpp#L15)
   — global `descale_qk` folded into `scale_s`.
8. **Whole-K prefetch** —
   [block_fmha_pipeline_qr_ks_vs_whole_k_prefetch.hpp](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_whole_k_prefetch.hpp)
   — load all of K up-front (small-hdim regime).
9. **BLOCKSCALE quant** —
   [qr_ks_vs.hpp:464-469](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L464)
   — per-block k/v descale factors indexed by `kv_idx`.
10. **MX-fp4** —
    [qr_ks_vs.hpp:897-911](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L897)
    — `cast_tile_mx<kVScaleGranularity>` emits packed fp4 + e8m0 scales.
11. **LogitsSoftCap** —
    [qr_ks_vs.hpp:644-659](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L644)
    — `variant.LogitsTransform()` for tanh-soft-cap fused before softmax.
12. **`kBlockPerCu` occupancy tuning** —
    [qr_ks_vs.hpp:114-143](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L114)
    — picks 1/2/3 blocks/CU by hdim to avoid register spill.

### 4.2 `qr_ks_vs` hot-loop walkthrough

**Stage 1 — QK GEMM with K prefetch ping-pong**
[:461-590](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L461)
— prefetch K[i+2] HBM ← while computing `QK·K[i]` and writing K[i+1] to LDS. Inner
`k0_loops` over hdim chunks. `sched_group_barrier` clusters MFMA with DS_READ at
[:434-457](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L434).

**Stage 2 — Online softmax & accumulator rescale**
[:713-829](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L713)
— see §4.3.

**Stage 3 — PV GEMM + final normalize**
[:933-1006](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L933),
[:1054-1071](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L1054)
— `O += P @ V`; final `O /= l` per row.

**LSE output**
[:1010-1048](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L1010)
— `LSE = m + log(l)` (or `m/C_LOG2E + log(l)` for the exp2 fast path); masked rows
yield `-inf`.

### 4.3 Online softmax (rescale formulas)

The CK_tile FMHA softmax uses the FlashAttention-2 running-(m,l) update with three
exp variants chosen at compile time. Quotes are exact (paraphrased only to fit):

1. **m-update**
   [:721-723](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L721)
   `m_old = m; m = max(m_old, m_local)`.
2. **P compute — bias / alibi path (fast exp2)**
   [:770](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L770)
   `p_compute = exp2(s - validated_m)`.
3. **P compute — no bias + scale_s (fast exp2)**
   [:776](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L776)
   `p_compute = exp2(scale_s * s - row_max)`.
4. **P compute — standard `exp`**
   [:780](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L780)
   `p_compute = exp(s - validated_m)`.
5. **FP8 BLOCKSCALE shift**
   [:754-761](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L754)
   subtracts `OCP_FP8_SHIFT (8.0)` or `FNUZ_FP8_SHIFT (7.0)` from m to keep the
   exp2 argument in the fp8-safe range.
6. **Rescale factor**
   [:802 or :819](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L802)
   `tmp = exp(m_old - m)`.
7. **l-update**
   [:821](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L821)
   `l = tmp * l_old + rowsum(P)`.
8. **O-rescale**
   [:827](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L827)
   `o_acc *= tmp` before adding the new PV product.
9. **`get_validated_m()`**
   [:728-741](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L728)
   — converts `-inf` to 0 so `exp(-inf)` never appears.
10. **Final normalize**
    [:1057-1066](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L1057)
    — `if (l == 0) o_acc *= 0` zeros fully-masked rows.

### 4.4 Masking strategies

1. **Tile-range elision (causal / sliding window)** —
   [block_masking.hpp:112-143](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/block/block_masking.hpp#L112)
   — `GenericAttentionMask::GetTileRangeAlongX` returns `[x_start, x_end]`, so
   fully-masked tiles are *skipped entirely*; no per-element check inside.
2. **Per-element masking via `set_tile_if`** —
   [qr_ks_vs.hpp:680-710](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L680)
   — predicate lambda; matches set to `-inf` without branching.
3. **ALiBi fused with S** —
   [qr_ks_vs.hpp:622-640](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L622)
   — `position_encoding.update(s_acc, row, col)` adds bias during S accumulation;
   no extra pass.
4. **Variant-dispatched masks** — `variant.LogitsMask()` /
   `variant.LogitsSinkMask()` toggle generic-vs-sink masking at compile time.
5. **Sink-token init**
   [qr_ks_vs.hpp:295-324](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L295)
   — `m`/`l` initialized with a virtual prefix token when `kHasSink`.
6. **Masked rows safe under exp** — combined `get_validated_m` (§4.3 #9) + final
   `o_acc *= 0` (§4.3 #10) keep NaN out of the output.

### 4.5 SplitKV + AppendKV

1. **Per-split outputs** —
   [splitkv_pipeline.hpp:158-168](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_qr_ks_vs.hpp#L158)
   — `operator()(... i_split)` writes `Oacc[b,h,split,m,:]` and `LSEacc[b,h,split,m]`.
2. **Combine kernel — load LSEacc**
   [combine_pipeline.hpp:114-145](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_combine_pipeline.hpp#L114)
   — pulls per-split LSE into LDS.
3. **Per-row `lse_max`**
   [combine_pipeline.hpp:181-183](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_combine_pipeline.hpp#L181).
4. **Logsumexp**
   [combine_pipeline.hpp:185-222](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_combine_pipeline.hpp#L185)
   — `lse_sum = Σ exp(LSEacc[i] - lse_max)`; final `LSE = lse_max + log(lse_sum)`.
5. **O-combine** —
   [combine_pipeline.hpp:252-264](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_combine_pipeline.hpp#L252)
   — `O = Σ_i exp(LSEacc[i] - LSE_final) * Oacc[i]`.
6. **Uneven splits handled by mask**
   [combine_pipeline.hpp:270-289](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_combine_pipeline.hpp#L270)
   — last split smaller; mask ensures zero contribution from padding.
7. **AppendKV core** —
   [appendkv_pipeline.hpp:89-185](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_appendkv_pipeline.hpp#L89)
   — direct writes; optional rotary at [:122-149](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_appendkv_pipeline.hpp#L122).
8. **AppendKV paged path** —
   [appendkv_pipeline.hpp:154-161](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_appendkv_pipeline.hpp#L154)
   — `page_block_navigator.is_cross_block()` triggers physical-block re-binding.

### 4.6 Backward pipeline

1. **`kr_ktr_vr` layout** —
   [block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp:14](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp#L14)
   — K-row / K-transpose-row / V-row layout chosen so K and Kᵀ live in the same
   LDS allocation with `ds_read_tr*`.
2. **Five GEMMs per backward block** —
   [:141-144](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp#L141):
   `gemm_0: dV += Pᵀ dO`, `gemm_1: dP = dO Vᵀ`, `gemm_2:` intermediate,
   `gemm_3: dQ += dS Qᵀ`, `gemm_4: dK += dSᵀ K`.
3. **Recompute S from Q,K** —
   [:236-237](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp#L236)
   — saves bandwidth vs. materializing S in forward.
4. **Shuffled K storage** —
   [:213-232](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp#L213)
   — pre-shuffle K's LDS layout so `Kᵀ` access is free.
5. **dS via `D = Σ dO ⊙ O`** — gradient identity
   `dS = softmax'(S) ⊙ (dP - D)` keeps backward numerically aligned with the
   forward online softmax.

### 4.7 fp8 / MX-fp4 variants

1. **FP8 global descale** —
   [qr_ks_vs_fp8.hpp:243](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_fp8.hpp#L243)
   — `scale_s = scale_s * descale_qk`; no per-tile descale.
2. **MX-fp4 P cast** —
   [qr_ks_vs.hpp:908-911](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L908)
   — `cast_tile_mx<kVScaleGranularity, WarpGemmAttribute::kAMLane>` emits
   `(p_result, p_scale_result)` consumed jointly by the SV `mfma_scale`.
3. **BLOCKSCALE index** —
   [qr_ks_vs.hpp:464-469](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L464)
   — `kv_idx = (kv_load_start + i_total_loops * kN0) / block_scale_size_kv`;
   shared by K & V scale arrays.
4. **MX scale granularity** —
   [qr_ks_vs.hpp:72-73](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L72)
   — `kQKScaleGranularity` / `kVScaleGranularity` (8 / 32 / 128) trade precision
   vs scale storage.
5. **OCP vs FNUZ shift** —
   [qr_ks_vs.hpp:754-761](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L754)
   — different bias constants for OCP fp8 (8.0) vs FNUZ fp8 (7.0).

### 4.8 Page-attention

1. **`PageBlockNavigator`** —
   [page_block_navigator.hpp:93-200](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/block/page_block_navigator.hpp#L93)
   — holds `physical_blocks`, `physical_block_indices`, `page_block_size`.
2. **Logical→physical lookup** —
   [page_block_navigator.hpp:124-142](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/block/page_block_navigator.hpp#L124)
   — `get_block_index(window_origin) → block_ptr`; `set_bottom_tensor_view_data_ptr`
   swaps the V# pointer per page.
3. **Cross-block re-bind** —
   [page_block_navigator.hpp:156-175](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/block/page_block_navigator.hpp#L156)
   — `move_tile_window` detects page-boundary crossing and updates the descriptor.
4. **Paged forward** —
   [pagedkv_pipeline.hpp:147-149](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_pagedkv_pipeline_qr_ks_vs.hpp#L147)
   — wraps both K and V tile windows in the navigator.

### 4.9 Cross-cutting techniques (this slice)

- **T4.A — Predicate-free masking via tile-range elision + `set_tile_if`**:
  whole-tile skip in `GenericAttentionMask` + per-element no-branch in §4.4 #2.
- **T4.B — Three exp paths chosen at compile time**: exp2 fast-path for fp16 bias,
  exp2 fast-path for plain QK, and standard `exp` — picked from variant traits.
- **T4.C — FP8 shift trick keeps `exp2(s-m)` in fp8 range**: subtract 8.0 (OCP) or
  7.0 (FNUZ) from m before the rescale, then re-apply when computing LSE.
- **T4.D — Logsumexp split-reduction**: split kernel writes per-shard (Oacc, LSEacc);
  combine kernel does numerically stable `lse_max + log(Σ exp(LSEacc - lse_max))`.
- **T4.E — `kr_ktr_vr` LDS layout for free Kᵀ in bwd**: shuffled K layout lets
  `ds_read_tr*` produce both K and Kᵀ from one LDS allocation.
- **T4.F — Page-table indirection at tile-window level**: `PageBlockNavigator`
  swaps the buffer-descriptor pointer per page boundary, so the rest of the
  pipeline is unchanged from contiguous-KV.
- **T4.G — Per-block descale folded into `scale_s`**: avoids per-iteration descale
  multiply by combining with the existing softmax temperature.
- **T4.H — `kBlockPerCu` chosen by hdim**: small hdim → 3 blocks/CU (parallelism);
  large hdim → 1 block/CU (avoid spill).
- **T4.I — Reduce-then-scale O**: `o_acc *= tmp` *before* adding the new PV
  product, not after, keeps the running accumulator numerically aligned with the
  current m without extra storage.
- **T4.J — Sink-token via virtual prefix**: `m`/`l` initialized with sink logits
  so streaming attention requires no special inner-loop branch.

Cross-references: [asm-v2:§23] (FMHA online softmax), [asm-v2:§24] (PA paged-KV),
[asm-v2:§25] (MLA), [pdf:p266] (mfma_scale).

<!-- SLICE-5-INSERT -->
## §5. CK_tile fused_moe + gemm_quant + gemm_mx + flatmm + batched_contraction

These are the "specialized GEMM" families that reuse the warp-GEMM core from §3 but
extend it with quantization scale handling, MoE expert dispatch, MX-fp4 OpSel
selection, flattened-batch layouts, and N-D tensor contractions.

### 5.1 fused_moe / fused_moegemm

1. **Canonical pipeline** —
   [fused_moegemm_pipeline_flatmm_ex.hpp:22](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fused_moegemm/pipeline/fused_moegemm_pipeline_flatmm_ex.hpp#L22)
   — dual GEMMs (gate-up + down) with sorted-expert dispatch.
2. **A/G/D tile distributions** —
   `MakeGlobalTileDistribution_A` ([:85](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fused_moegemm/pipeline/fused_moegemm_pipeline_flatmm_ex.hpp#L85)),
   `..._G` ([:156](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fused_moegemm/pipeline/fused_moegemm_pipeline_flatmm_ex.hpp#L156)),
   `..._D` ([:161](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fused_moegemm/pipeline/fused_moegemm_pipeline_flatmm_ex.hpp#L161)).
3. **Dual accumulator** —
   [:315-317](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fused_moegemm/pipeline/fused_moegemm_pipeline_flatmm_ex.hpp#L315)
   — `acc_0` (gate-up) + `acc_1s[2]` (down, ping-pong); two warp-GEMMs per K-iter
   ([:320-425](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fused_moegemm/pipeline/fused_moegemm_pipeline_flatmm_ex.hpp#L320)).
4. **Sequencer bit-slots** —
   [:54](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fused_moegemm/pipeline/fused_moegemm_pipeline_flatmm_ex.hpp#L54)
   `FusedMoeGemmPipelineSequencerEnum` enum (SLD_A / GLD_A / GLD_B / GST_O);
   `static_for` unrolls at compile time ([:444-454](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fused_moegemm/pipeline/fused_moegemm_pipeline_flatmm_ex.hpp#L444)).
5. **A double-buffer LDS** —
   [:115-118](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fused_moegemm/pipeline/fused_moegemm_pipeline_flatmm_ex.hpp#L115)
   — `smem_0/smem_1`; intermediate bridge buffer recast for Y output
   ([:230](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fused_moegemm/pipeline/fused_moegemm_pipeline_flatmm_ex.hpp#L230)).
6. **Atomic-add epilogue** —
   [:311](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fused_moegemm/pipeline/fused_moegemm_pipeline_flatmm_ex.hpp#L311)
   — `atomic_add_o()` merges GEMM0+GEMM1 outputs; TopkWeight scale deferred.
7. **`kBlockPerCu = 2`** —
   [:65](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fused_moegemm/pipeline/fused_moegemm_pipeline_flatmm_ex.hpp#L65)
   — MoE load balance; sorted-expert ordering keeps coalesced loads.
8. **SmoothQuant flags** —
   [:45-47, :129-150](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fused_moegemm/pipeline/fused_moegemm_pipeline_flatmm_ex.hpp#L45)
   — `UseSmoothQuant`, `PadIntermediateSize` route to preshuffled weight layouts.
9. **Tail-vs-hot duality** — full pipeline branch
   [:435-480](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fused_moegemm/pipeline/fused_moegemm_pipeline_flatmm_ex.hpp#L435);
   tail branch consumes final GEMM0 buffer without reload
   ([:482-499](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fused_moegemm/pipeline/fused_moegemm_pipeline_flatmm_ex.hpp#L482)).
10. **`block_sync_load_raw()` fence** —
    [:457](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fused_moegemm/pipeline/fused_moegemm_pipeline_flatmm_ex.hpp#L457)
    — gates the prefetch handoff between gate-up and down.

### 5.2 `gemm_quant` (a8w4 / a8w8 / wint4)

1. **Three pipeline variants** —
   [gemm_abquant_pipeline_ag_bg_cr_v3.hpp:24](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm_quant/pipeline/gemm_abquant_pipeline_ag_bg_cr_v3.hpp#L24)
   (both quantized), `gemm_aquant_pipeline_*`, `gemm_bquant_pipeline_*`.
2. **Group sizes** —
   [:35-36](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm_quant/pipeline/gemm_abquant_pipeline_ag_bg_cr_v3.hpp#L35)
   `AQuantGroupSize`, `BQuantGroupSize` — controls K-stride per scale.
3. **Scale tile shapes** —
   [:75-79](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm_quant/pipeline/gemm_abquant_pipeline_ag_bg_cr_v3.hpp#L75)
   — `KPerBlockAQ = kK / AQuantGroupSize::kK`; B grid is N×K.
4. **Descale fold inside block-GEMM** —
   [block_universal_gemm_as_aquant_bs_bquant_cr.hpp:371-373](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_aquant_bs_bquant_cr.hpp#L371)
   — `c_block[c_row] += c_warp[c_row] * a_scale_reg_f * b_scale_reg_f`. Both
   scales folded into the same FMA.
5. **Cross-lane scale broadcast** —
   [block_gemm_quant_common.hpp:362-366](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm_quant/block/block_gemm_quant_common.hpp#L362)
   — `__builtin_amdgcn_ds_bpermute(pull_from_lane << 2, …)` gathers preshuffled
   B-scale to the right lane.
6. **`BPreshuffleQuant` flag** —
   [block_gemm_quant_common.hpp:102](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm_quant/block/block_gemm_quant_common.hpp#L102)
   — pre-scattered scales avoid LDS-scale-load stalls.
7. **fp8/bf8 scale unpack** —
   [block_gemm_quant_common.hpp:41-62](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm_quant/block/block_gemm_quant_common.hpp#L41)
   `cvt_scale_to_fp32()` uses `__builtin_amdgcn_cvt_f32_fp8/bf8` to dequant
   in-register, fused with the gather.
8. **Per-K-group iteration** —
   [gemm_abquant_pipeline_ag_bg_cr_v3.hpp:432](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm_quant/pipeline/gemm_abquant_pipeline_ag_bg_cr_v3.hpp#L432)
   — outer loop over K-quantization groups; each iter loads matching AQ/BQ scale,
   runs warp-GEMM, folds.
9. **Smem doubled by `PackedSize`** —
   [:132-135](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm_quant/pipeline/gemm_abquant_pipeline_ag_bg_cr_v3.hpp#L132)
   — `APackedSize`/`BPackedSize` ([:49-57](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm_quant/pipeline/gemm_abquant_pipeline_ag_bg_cr_v3.hpp#L49))
   inflate LDS allocation for int4/fp4 inputs.
10. **Reused warp GEMM** —
    [:65](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm_quant/pipeline/gemm_abquant_pipeline_ag_bg_cr_v3.hpp#L65)
    — `Policy::template GetBlockGemm<Problem>()`; the warp kernel is identical to §3.

### 5.3 `gemm_mx` (MX-fp4/fp6/fp8 via OpSel)

1. **Pipeline header** —
   [gemm_pipeline_ag_bg_cr_comp_async.hpp:102](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm_mx/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp#L102)
   (the `gemm_mx` namespace's own file). Comment at
   [:17](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm_mx/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp#L17):
   "MX scaling support with OpSel."
2. **`ScaleBlockSize = 32` K-elements/scale** —
   [:128](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm_mx/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp#L128).
3. **`MXScalePointer<e8m0_t, MN_gran, K_gran>`** —
   [scale_pointer.hpp:10-45](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm_mx/scale_pointer.hpp#L10)
   — packs 2M×2K e8m0 scales per int32; `operator+` strides by granularity,
   `operator[]` divides by `GranularityMN` when K-gran = 0.
4. **`GranularityMN = -1` = no scale** —
   [scale_pointer.hpp:109](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm_mx/scale_pointer.hpp#L109)
   returns 1 (broadcast); supported alongside per-pair / per-quad granularities.
5. **OpSel byte-select**: MFMA uses the OpSel field [0..3] to pick which byte of
   the packed int32 holds the e8m0 exponent for this tile; scaling is applied
   *inside* the MFMA, never unpacked to full precision.
6. **Sched_group_barrier orchestration** —
   [:222-234](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm_mx/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp#L222)
   — MFMA / DS_READ / VMEM_READ pattern with one VMEM per 6–8 MFMA cycles, same
   spirit as §3's V4 pattern.
7. **No post-GEMM descale**: because MFMA already applies the e8m0 exponent,
   the C accumulator is pre-scaled; epilogue receives final fp32.
8. **Cross-precision support**: A and B can independently be fp8/bf8/fp6/fp4
   with separate scale pointers; C stays fp32.
9. **Persistent-kernel variant** —
   [:25](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm_mx/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp#L25)
   `UsePersistentKernel` enables multi-pass kernels for huge N; scale pointers
   re-index per block offset.

### 5.4 `flatmm` (flattened batch×seq)

1. **Pipeline** —
   [flatmm_pipeline_agmem_bgmem_creg_v1.hpp:69](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1.hpp#L69)
   — batch×seq folded into effective M.
2. **`FlatmmShape`** —
   [:98-99](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1.hpp#L98)
   — M/N/K block size plus flat-K / flat-N per warp.
3. **`MIterPerWarp` / `KIterPerWarp`** —
   [:126-128](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1.hpp#L126).
4. **`SchedulerPerM()`** —
   [:220-264](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1.hpp#L220)
   — explicit per-M-iteration order of `ds_read` / `ds_write` / `mfma` /
   `vmem_load` to keep LDS banks open.
5. **`DsReadPreload = 2`** —
   [:195](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1.hpp#L195)
   — two A-load stages ahead of compute.
6. **Mixed-prec variant** —
   [mixed_prec_flatmm_pipeline_agmem_bgmem_creg_v1.hpp:25-49](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/flatmm/pipeline/mixed_prec_flatmm_pipeline_agmem_bgmem_creg_v1.hpp#L25)
   — fp16-A × MX-fp4-B; `ContinuousKPerThread=32`, scale-load counts
   ([:165-176](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/flatmm/pipeline/mixed_prec_flatmm_pipeline_agmem_bgmem_creg_v1.hpp#L165)).
7. **MX flatmm** —
   [mx_flatmm_pipeline_agmem_bgmem_creg_v1.hpp:52](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/flatmm/pipeline/mx_flatmm_pipeline_agmem_bgmem_creg_v1.hpp#L52)
   — applies OpSel scaling on top of flatmm structure.
8. **B-preshuffle dwordx4-aligned** —
   [mixed_prec_flatmm_pipeline_agmem_bgmem_creg_v1.hpp:191](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/flatmm/pipeline/mixed_prec_flatmm_pipeline_agmem_bgmem_creg_v1.hpp#L191)
   `BPreShufflePermute` flag.
9. **Register-only C** — no C-LDS; all accumulate in VGPRs (saves LDS for A/B
   double buffer; primary occupancy constraint).
10. **Persistent kernel** —
    [:100](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1.hpp#L100)
    `UsePersistentKernel` enables grid-loop over M/N slices.

### 5.5 `batched_contraction`

1. **Kernel** —
   [batched_contraction_kernel.hpp:95](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/batched_contraction/kernel/batched_contraction_kernel.hpp#L95)
   wraps `UniversalGemmKernel` with N-D flattening
   ([:47-77](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/batched_contraction/kernel/batched_contraction_kernel.hpp#L47)).
2. **Host args** —
   [:180-250](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/batched_contraction/kernel/batched_contraction_kernel.hpp#L180)
   `BatchedContractionKernelArgs` carries M/N/K/G dim arrays + strides.
3. **D-tensor fusion** — `NumDTensor` template param folds 0+ auxiliary inputs
   into the epilogue (bias / residual / multiple-D add).
4. **On-device descriptor builder** —
   `TensorDescriptorUtils::Make_A_GridDescriptor_M_K()` ([:223-230](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/batched_contraction/kernel/batched_contraction_kernel.hpp#L223))
   — strides + padding constructed inside the kernel; no host-codegen needed.
5. **k_batch split-K** —
   [:194](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/batched_contraction/kernel/batched_contraction_kernel.hpp#L194)
   — same merge logic as universal GEMM.
6. **Reuses GEMM core** — single `UniversalGemmKernel` instantiation; no
   quantization variant inside (compose via wrapping if needed).

### 5.6 Cross-cutting techniques (this slice)

- **T5.A — Preshuffle weight layout to align quant groups with K-tiles**: host
  pre-reorder so each warp-K-iter sees one scale per fetch (§5.2 #6, §5.4 #8).
- **T5.B — Descale folded into the accumulator FMA**: `c += c_warp * a_scale *
  b_scale` (§5.2 #4) — never materializes a dequantized A or B.
- **T5.C — `ds_bpermute` for cross-lane scale broadcast**: avoids a second LDS
  read on hot path (§5.2 #5).
- **T5.D — OpSel byte-select replaces explicit descale**: MX MFMA applies e8m0
  exponent in hardware; epilogue gets pre-scaled fp32 (§5.3 #5).
- **T5.E — Dual accumulator for gate-up + down fusion**: MoE keeps two C-tile
  banks alive so the second GEMM can start while the first commits (§5.1 #3).
- **T5.F — Atomic-add MoE epilogue + sorted-expert ordering**: handles top-K
  routing without per-token branches (§5.1 #6/#7).
- **T5.G — Flatten batch×seq into M for MoE/persistent kernels**: removes 3-D
  tensor stride overhead and tightens warp tiling (§5.4 #1).
- **T5.H — Per-K-iter scheduler (`SchedulerPerM`)**: hand-tuned `ds_read`/
  `ds_write`/`mfma` ordering inside flatmm; same spirit as §3 sched-group
  patterns but per-M-iter granularity.
- **T5.I — Tensor-descriptor builders on-device**: `batched_contraction` builds
  strides from multi-D arrays at kernel-entry, so the same binary supports
  arbitrary N-D contractions.
- **T5.J — Reused warp-GEMM core**: every variant calls the same warp operator
  from §3 — quantization/scale logic is layered around it, not threaded
  through it.

Cross-references: [asm-v2:§16] (FMOE), [asm-v2:§17] (flatmm), [asm-v2:§27]
(MX MFMA), [pdf:p266] (`mfma_scale_f32`), [pdf:p278] (OpSel encoding).

<!-- SLICE-6-INSERT -->
## §6. CK_tile reduce / softmax / norm / topk / elementwise / non-cshuffle epilogue

These are the "row-reduction-shape" op families. They share the same three-stage
reduction primitive (thread → warp → block-LDS) and differ only in the fusion of
inputs (residual add, smooth scale) and the choice of pass count (one-pass vs
two-pass vs three-pass).

### 6.1 `reduce`

1. **Three-stage hierarchy** —
   [block_reduce2d.hpp:11-41](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/reduce/block/block_reduce2d.hpp#L11)
   — thread sweep → warp tree (`warp_shuffle_xor`) → cross-warp LDS scratch.
2. **Distribution metadata drives reduce axis** —
   [:213-275](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/reduce/block/block_reduce2d.hpp#L213)
   `DstrEncode::does_p_own_r_` + `ps_over_rs_derivative_` from §1 select R-dims
   mapped to lanes.
3. **Cross-warp via LDS** —
   [:348-463](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/reduce/block/block_reduce2d.hpp#L348)
   — lane-0-of-each-warp writes partial; all threads reload+reduce.
4. **Power-of-two assert** —
   [:247, :434](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/reduce/block/block_reduce2d.hpp#L247)
   — enables log2-stage compile-time-unrolled tree.
5. **Argmax/argmin overload** —
   [:84-88, :265-267](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/reduce/block/block_reduce2d.hpp#L84)
   — index tensors carried through reduction (regs in warp phase, LDS in
   cross-warp).
6. **`ReducePacksPerXDim`** —
   [:105-119](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/reduce/block/block_reduce2d.hpp#L105)
   — `{M,N}` packs (e.g. 2×4) sweep multi-element batches for ILP.
7. **XOR-based shuffle**
   [:254-260](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/reduce/block/block_reduce2d.hpp#L254)
   — branchless cross-lane source-ID computation.
8. **Optional broadcast post-reduce**
   [block_reduce.hpp:85-100](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/reduce/block/block_reduce.hpp#L85)
   — flag-gated replicate of the reduced scalar back to all lanes.
9. **Linear cross-warp variant** —
   [:486-549](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/reduce/block/block_reduce2d.hpp#L486)
   `BlockReduce2dLinearCrossWarpSync` (sequential LDS layout) for alignment
   constraints.
10. **Register-only fallback** —
    [:398-399](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/reduce/block/block_reduce2d.hpp#L398)
    — single-warp problem skips LDS entirely.
11. **`MakeYBlockTile` factory** —
    [:144-157](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/reduce/block/block_reduce2d.hpp#L144)
    — output distribution derived by dropping reduce axes from input.

### 6.2 `softmax`

1. **Two-pass row-wise** —
   [block_softmax_2d.hpp:43-53](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/softmax/block/block_softmax_2d.hpp#L43)
   — pass 1 row-max, pass 2 `exp(x - max)`.
2. **Row-sum reciprocal stored once** —
   [:57-70](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/softmax/block/block_softmax_2d.hpp#L57)
   — `1/sum` reused across all elements in row.
3. **Built on `BlockReduce2D`** with `f_max` / `f_sum` lambdas
   ([:44, :57](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/softmax/block/block_softmax_2d.hpp#L44)).
4. **Constraint: row fits in one warp** —
   [:13-17](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/softmax/block/block_softmax_2d.hpp#L13)
   — no cross-warp sync needed.
5. **DWORD assumption** —
   [:17-18](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/softmax/block/block_softmax_2d.hpp#L17)
   — fp16/bf16 require packing.
6. **`sweep_tile` re-iterates without reload** — pass 2 replays tile from the
   same VGPR state.
7. **`v_max3_f32` macro path** —
   [:9, :33-39](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/softmax/block/block_softmax_2d.hpp#L9)
   `_BLOCK_SOFTMAX_USE_UNPACK2` flag enables 3-way max in one inst.

### 6.3 norm family (rmsnorm2d / layernorm2d / add+rmsnorm+rdquant)

1. **Welford online estimator** —
   [thread_welford.hpp:11-25](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/welford/thread/thread_welford.hpp#L11)
   `mean += delta/count; var += delta*delta2` — single pass, numerically stable.
2. **`kFastFDiv` switch** —
   [:15-18, :41-44](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/welford/thread/thread_welford.hpp#L15)
   `__builtin_amdgcn_rcpf()` replaces `fdiv` (~2× faster, lower accuracy).
3. **Welford merge for multi-pass** —
   [:28-55](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/welford/thread/thread_welford.hpp#L28)
   — combine partial stats with M2 correction term.
4. **Two-pass RMSNorm** —
   `rmsnorm2d_fwd_pipeline_two_pass.hpp` — pass 1 `Σ x²`, pass 2 `1/√(ε+var)·γ·x`.
5. **Three-pass add+rmsnorm+rdquant** —
   `add_rmsnorm2d_rdquant_fwd_pipeline_three_pass.hpp` ([:77-91](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/add_rmsnorm2d_rdquant/pipeline/add_rmsnorm2d_rdquant_fwd_pipeline_three_pass.hpp#L77))
   — pass 1 `add + Σ x²`, pass 2 var, pass 3 `y·scale+quantize`. Single kernel,
   no intermediate writes.
6. **`v_max3_f32` for absmax in quant pass** —
   [:82-85](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/add_rmsnorm2d_rdquant/pipeline/add_rmsnorm2d_rdquant_fwd_pipeline_three_pass.hpp#L82)
   — three-operand max in a single instruction.
7. **Gamma-fusion gate `kHasGamma`** — `null_type` branch omits gamma load entirely
   when absent ([rmsnorm2d_fwd_pipeline_two_pass.hpp:28](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/rmsnorm2d/pipeline/rmsnorm2d_fwd_pipeline_two_pass.hpp#L28)).
8. **`kNeedCrossWarpSync` knob** — warp-per-row vs block-per-row routing
   ([:31, :38](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/rmsnorm2d/pipeline/rmsnorm2d_fwd_pipeline_two_pass.hpp#L31)).
9. **`kPadN` boundary handling** — masked stores for ragged rows.
10. **`kSaveInvRms`** — saves `1/rms` for backward; trades 4B/row memory for
    skipping the recompute in bwd.
11. **`tile_elementwise_in` for fused residual add** — `kFusedAdd` chains
    residual into pass 1's sweep.

### 6.4 `topk` (streaming K-selection)

1. **Iterative argmax** —
   [block_topk_stream_2d.hpp:53-110](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/topk/block/block_topk_stream_2d.hpp#L53)
   — per-k row sweep: argmax → store → mark `-inf` → repeat.
2. **`ArgmaxPacket{value, index}`** —
   [:25-29](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/topk/block/block_topk_stream_2d.hpp#L25)
   — value+index packed for register-friendly compare.
3. **`block_tile_reduce_xor_sync`** —
   [:75](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/topk/block/block_topk_stream_2d.hpp#L75)
   — XOR warp-reduce.
4. **Output window slides per-k** —
   [:107-108](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/topk/block/block_topk_stream_2d.hpp#L107)
   — `move_tile_window` after each store; both value & index windows in lockstep.
5. **Column-gated store** —
   [:102-106](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/topk/block/block_topk_stream_2d.hpp#L102)
   — only `tid % ColLanes == 0` writes; avoids redundant stores.
6. **`-inf` sentinel for removal** —
   [:97-98](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/topk/block/block_topk_stream_2d.hpp#L97).
7. **Warp-bounded row constraint** —
   [:13](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/topk/block/block_topk_stream_2d.hpp#L13)
   — same single-warp assumption as softmax.

### 6.5 elementwise / smoothquant

1. **Per-channel smooth scale** —
   [smoothquant_pipeline_one_pass.hpp:73-80](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/smoothquant/pipeline/smoothquant_pipeline_one_pass.hpp#L73)
   — shape `(1, N)` broadcast across rows.
2. **Fused `(x·smooth)/(1/max(|y|))`** in single sweep
   ([:73-80](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/smoothquant/pipeline/smoothquant_pipeline_one_pass.hpp#L73))
   — scale + normalize + quant all collapsed.
3. **`v_max3_f32` absmax** —
   [:59-65](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/smoothquant/pipeline/smoothquant_pipeline_one_pass.hpp#L59)
   — three abs values per inst.
4. **`BlockElementWiseKernel`** —
   [elementwise_kernel.hpp:49-71](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/elementwise/kernel/elementwise_kernel.hpp#L49)
   — variadic input tuple via `generate_tuple`.
5. **`merge_transform` coalesces dims** so a 4-D activation can be swept as 1-D.
6. **Two-pass smoothquant** mirrors the RMSNorm split (scale-pass + quant-pass)
   for higher precision.

### 6.6 Non-cshuffle epilogue

1. **`Default2DEpilogue`** —
   [default_2d_epilogue.hpp:14-76](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/default_2d_epilogue.hpp#L14)
   — simple cast Acc→O with optional padding.
2. **`DynamicQuantEpilogue`** —
   [dynamic_quant_epilogue.hpp:44-73](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/dynamic_quant_epilogue.hpp#L44)
   — per-row absmax inline; uses BlockReduce2d chain.
3. **`UseSmoothInputScale`** —
   [:75-99](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/dynamic_quant_epilogue.hpp#L75)
   — applies channel smooth before computing per-token scale.
4. **Two scale tensors carried separately** — `SmoothScaleDataType`
   (broadcasted) and `YScaleDataType` (per-token, written back).
5. **`PermuteN` epilogue** —
   [permuten_epilogue.hpp:17-84](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/permuten_epilogue.hpp#L17)
   — N-dim permutation via warp-GEMM dispatcher (no extra LDS).
6. **`UseRawStore`** flag bypasses DLF on aligned paths.
7. **`NumDTensor` D-list** — fused element-wise op chaining (bias / activation)
   stays inside this epilogue.

### 6.7 Cross-cutting techniques (this slice)

- **T6.A — Single reduction primitive feeds all op families**: `BlockReduce2D`
  is parameterized by `(reduce_func, distribution)`; softmax, norm, topk, smoothquant
  all just supply lambdas.
- **T6.B — Welford one-pass with `kFastFDiv`**: turns var into one sweep + a
  builtin reciprocal — replaces the textbook two-pass mean+var.
- **T6.C — `v_max3_f32` everywhere there's an absmax**: smoothquant, rdquant,
  softmax (optionally) — 3 ops in 1 inst.
- **T6.D — `kNeedCrossWarpSync` is a routing knob**: warp-bounded path skips
  all LDS sync; block-bounded path uses scratch.
- **T6.E — Three-pass `add+rmsnorm+rdquant` collapses three kernels into one
  pipeline** — saves two DRAM round-trips on residual paths.
- **T6.F — Streaming topk avoids sort**: `argmax → store → mark -inf` runs in
  K passes over registers, never sorting.
- **T6.G — Pack `(value, index)` for argmax**: lets the warp tree-reduce stay
  scalar; no parallel index pipeline.
- **T6.H — `kHasGamma`/`kPadN`/`kSaveInvRms`/`kHasSink` traits** — bit-flag
  config gates compile-time dead-code elimination, so a single template covers
  many shipping variants.

Cross-references: [asm-v2:§19] (LDS scratch reduction), [asm-v2:§26] (smoothquant),
[pdf:p266] (v_max3_f32), [pdf:p159] (ds_bpermute).

<!-- SLICE-7-INSERT -->
## §7. Classic CK device-instance hierarchy

"Classic CK" predates CK_tile. It's an explicit nested template hierarchy —
**Device → Grid → Pipeline → Block → Warp → Thread** — where each level is a
struct with `Run()` that calls the level below. The CK_tile DSL replaces the
manual descriptor plumbing with the compile-time abstractions of §1, but the
classic stack is still where most shipping device-instance kernels live.

### 7.1 `DeviceGemm` V1 → V2 → V2BScale

1. **`DeviceGemm` (V1)** —
   [device_gemm.hpp:21](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/device/device_gemm.hpp#L21)
   — virtual base; `MakeArgumentPointer` + `MakeInvokerPointer`; plain `C = A @ B`.
2. **`DeviceGemmV2`** —
   [device_gemm_v2.hpp:21](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/device/device_gemm_v2.hpp#L21)
   — adds `KSplit` for split-K; `GetPermuteA/B()`, `GetKPerBlock()` for metadata;
   `V2R1` flavor supports D-tensors.
3. **`DeviceGemmV2BScale`** —
   [device_gemm_v2.hpp:92](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/device/device_gemm_v2.hpp#L92)
   — block-wise B-scale (`ScaleBlockN`, `ScaleBlockK`); mixed-precision path.
4. **XDL-CShuffle dispatch** —
   [device_gemm_xdl_cshuffle.hpp:71](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/device/device_gemm_xdl_cshuffle.hpp#L71)
   `DeviceGemm_Xdl_CShuffle` plugs a `GridwiseGemm_xdl_cshuffle_v1` with the
   `PipelineVersion` enum.
5. **Split-K / Stream-K wrappers** — `device_gemm_splitk.hpp` /
   `device_gemm_streamk.hpp` use `atomicAdd` to merge partial C tiles.
6. **Multi-D variants** — `device_gemm_multiple_d*.hpp` accept
   `std::array<const void*, NumDTensor>` for bias/residual fusion.
7. **Quantization variants** — `device_gemm_dequantB.hpp` and `device_gemm_mx.hpp`
   inject dequant + scale handling at the device layer.

### 7.2 Gridwise GEMM pipelines

1. **`GridwiseGemmPipeline_v1`** —
   [gridwise_gemm_pipeline_v1.hpp:13](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v1.hpp#L13)
   — 1- or 2-stage prefetch; single LDS buffer; `read → sync → gemm → sync → write`.
2. **`GridwiseGemmPipeline_v2`** —
   [gridwise_gemm_pipeline_v2.hpp:10](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v2.hpp#L10)
   — 2-stage prefetch *overlapped*: `read(i+2)` while computing `gemm(i)`; even
   `num_loop` required. IGLP pragma at
   [:82](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v2.hpp#L82).
3. **`GridwiseGemmPipeline_v3`** —
   [gridwise_gemm_pipeline_v3.hpp:10](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v3.hpp#L10)
   — unconditional applicability; single `while (num_loop--)`; relies on compiler
   scheduling for the overlap V2 did by hand.
4. **`gridwise_gemm_pipeline_selector.hpp`** — maps the `PipelineVersion` enum to
   the template instance. V1=gfx908, V2=gfx90a, V3=gfx94+ default.
5. **`GridwiseGemm_xdl_cshuffle_v{1,2,3}`** —
   [gridwise_gemm_xdl_cshuffle_vN.hpp](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/)
   — kernel entry that owns `a_block_buf` / `b_block_buf` / `c_thread_buf` and
   passes them to the pipeline.
6. **XDLOps `v2r3` / `v3r1` low-level**
   ([gridwise_gemm_xdlops_v2r3.hpp](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r3.hpp),
   `v3r1`) — without CShuffle, threads write directly to MFMA outputs; `v3r1`
   introduces `Block2CTileMap` for flexible block-to-tile assignment.
7. **Double-LDS via two `__shared__` arrays** — V2's kernel declares `p_shared_0`
   and `p_shared_1` separately so the compiler treats them as non-aliasing; needs
   `TailNum=3` to drain.

### 7.3 Blockwise GEMM

1. **`BlockwiseGemmXdlops_v1`** —
   [blockwise_gemm_xdlops.hpp:44](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp#L44)
   — `MRepeat × NRepeat` XDLOP invocations over `MWaves × NWaves` thread waves.
2. **`BlockwiseGemmXdlops_pipeline_v4`** —
   [blockwise_gemm_pipeline_xdlops.hpp:104](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops.hpp#L104)
   — adds prefetch stages; tracks `KPerThread = KPerBlock / xdlops_gemm.K0PerXdlops`.
3. **Wave layout** — `MWaves = MPerBlock/(MRepeat*MPerXDL)`,
   `NWaves = NPerBlock/(NRepeat*NPerXDL)`; `WaveSize = BlockSize/(MWaves*NWaves)`.
4. **`c_thread_buf` register tile** —
   `StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, FloatAcc, MRepeat*NRepeat,
   GetRegSizePerXdlops()>` — C accumulated entirely in VGPRs.
5. **A/B LDS descriptors** — `AK0_M_AK1` and `BK0_N_BK1`; K1 is the vector-load
   packing dim, K0 is the MFMA K-reduction dim.
6. **A-thread origin index** — `CalculateAThreadOriginDataIndex()` aligns thread
   reads to the XDLOPS input format.
7. **`A/B/C ElementwiseOperation`** — invoked per element before MFMA input
   (bias / activation / scale) — function-object composition.

### 7.4 XDLOPS dispatcher

1. **`XdlopsGemm`** —
   [xdlops_gemm.hpp:1871](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/warp/xdlops_gemm.hpp#L1871)
   — `<FloatAB, MPerXdlops, NPerXdlops, KPack, FloatAcc, TransposeC>`. Encodes
   register layout + instruction shape.
2. **`GetNumXdlops()`** — `MPerXdlops*NPerXdlops /
   (m_per_blk * n_per_blk * num_output_blks)` for the chosen MFMA.
3. **MFMA-type table** — `mfma_type<MfmaInstr::mfma_f32_32x32x2f32>` etc., each
   exposing `m_per_blk / n_per_blk / k_per_blk / num_output_blks`.
4. **C-descriptor unmerge** — `MakeCDescriptor_M0_N0_M1_N1_M2_M3_M4_N2()`
   transforms `M0_N0_M1_N1_M2_N2` into the 8-D register layout by unmerging M2
   into `(num_groups_per_blk, num_blocks, group_size)`.
5. **`KPack` constraint** — must be divisible by `mfma_instr.k_per_blk`; larger
   KPack ⇒ fewer iterations + more VGPR pressure.
6. **Mixed-precision via specialization** — same struct supports fp16/bf16/fp8/
   bf8/i8 input + fp32 acc; gfx94+ scale-MFMA selected when scale pointers given.

### 7.5 Warp / thread-level

1. **`ThreadwiseGemmDlops_km_kn_mn_v3`** —
   [threadwise_gemm_dlops_v3.hpp:28](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/thread/threadwise_gemm_dlops_v3.hpp#L28)
   — single-threaded GEMM via `amd_assembly_outer_product_1x4`.
2. **`amd_assembly_outer_product_1x4(a, b0..b3, c0..c3)`** — broadcasts `a` and
   does four FMAs in one inline-asm block; used by 1x1 conv per-thread workloads.
3. **`ThreadwiseGenericTensorSliceTransfer_v1`** — used by `BlockwiseGemmXdlops`
   to stream A/B globally into LDS per-thread.
4. **Static unrolls** — `static_for` and `static_ford` produce flat instruction
   streams; no runtime loop overhead for small (KxHoxWo) tiles.

### 7.6 V1 → V2 → V3 evolution

1. **V1 trade-off** — smallest LDS (single buffer); strict `num_loop` parity;
   `block_sync_lds` per iteration. Best on gfx908 / when occupancy is the bottleneck.
2. **V2 optimization** — explicit dual `__shared__` arrays + `IGLP` pragma so
   the compiler overlaps `read(i+2)` with `gemm(i)`; doubled LDS budget.
3. **V3 design** — same throughput as V2 but no parity constraint; control flow
   is plain `while (num_loop--)` and the compiler does the overlap. Default on
   gfx94+ where compiler scheduling is mature.
4. **Heuristic ladder** — `num_loop == 1` → V1 (prefetch-only); even K & gfx90a →
   V2; everything else on gfx94+ → V3.
5. **`NumGemmKPrefetchStage`** — `1` = immediate write after read (no buffer);
   `2` = preload two tiles before main loop. Higher stages reduce occupancy but
   hide DRAM latency.

### 7.7 Cross-cutting techniques (this slice)

- **T7.A — Six-level template stack**: Device → Grid → Pipeline → Block → Warp
  → Thread, each with `Run()` calling the level below; clean separation of
  concerns at the cost of verbose descriptor plumbing.
- **T7.B — `PipelineVersion` selector**: arch-aware default (V1 gfx908 / V2
  gfx90a / V3 gfx94+) but overridable via template param; lets one codebase
  ship to all CDNA targets.
- **T7.C — Double-LDS via independent `__shared__` arrays**: the compiler
  can't disambiguate offsets within one array, so V2 declares two; achieves the
  ping-pong without rewriting the pipeline.
- **T7.D — `Block2CTileMap` for cache-friendly grid order**: classic CK's
  answer to "thread-block swizzle" — Z-order / space-filling-curve over the
  output tile grid.
- **T7.E — C in VGPRs via `StaticBufferTupleOfVector`**: same idea as §1.3's
  `static_distributed_tensor`, just explicit (no descriptor abstraction).
- **T7.F — Element-wise ops as function objects**: `AElementwiseOperation`
  etc. are stateless functors composed at template-instantiation time; fused
  bias/activation/scale "for free."
- **T7.G — Inline-asm outer product for thread-level GEMM**: classic CK uses
  `amd_assembly_outer_product_1x4` for tiny per-thread tiles where MFMA
  overhead dominates.
- **T7.H — Split-K via atomic merge**: `device_gemm_splitk.hpp` launches
  `num_kblocks = ceil(K/KPerBlock)` and uses `atomicAdd` for the C-tile reduce
  — enables high occupancy on tall-thin GEMMs.

Cross-references: [asm-v2:§1] (kernel structure), [asm-v2:§8] (XDLOPS),
[asm-v2:§18] (LDS double-buffer), [pdf:p245] (MFMA encodings), §1 (CK_tile
abstractions that replace this stack).

<!-- SLICE-8-INSERT -->
## §8. Classic CK FMHA / dispatcher / library

### 8.1 Classic FMHA is gone — everything is CK_tile

1. **No standalone classic-CK FMHA pipelines exist in this submodule snapshot.**
   `library/src/tensor_operation_instance/gpu/mha/CMakeLists.txt:4-6` points its
   `FMHA_SRC_FOLDER` at `example/ck_tile/01_fmha/` and runs
   `codegen/generate.py`. Every FMHA instance shipped through `library/` is
   ultimately a CK_tile kernel (§4).
2. **WMMA-based attention exists for RDNA/WMMA paths** —
   `include/ck/tensor_operation/gpu/device/impl/device_grouped_query_attention_forward_wmma.hpp`
   and `device_multi_query_attention_forward_wmma.hpp` — these target the WMMA
   instructions, not XDLOPS, and live as classic device-instance kernels.
3. **`example/ck_tile/01_fmha/fmha_fwd.hpp`** defines `FmhaFwdFp16` /
   `FmhaFwdBf16` / `FmhaFwdFp8` *type-config* structs that are parameter packs
   passed to CK_tile templates — no separate "device class" wrapping FMHA.
4. **MHA codegen filter** — `mha/CMakeLists.txt:38-62` runs
   `python codegen/generate.py --api fwd,fwd_splitkv,fwd_appendkv,bwd …`; the
   "library" for FMHA is a *generated* set of `.cpp` files compiled into the
   shared object.
5. **Dispatcher has no FMHA Problem type** — `dispatcher/include/ck_tile/dispatcher/`
   exposes only `Problem`, `KernelKey`, `Signature` for GEMM; attention is out
   of scope for the runtime selector.

With classic FMHA absent, the rest of this slice analyzes the **classic GEMM
dispatcher + library + profiler** — the same machinery that ships every other
device-instance kernel.

### 8.2 Dispatcher (registry + heuristic + selection)

1. **Registry** —
   [registry.hpp:33-103](aiter-amd/3rdparty/composable_kernel/dispatcher/include/ck_tile/dispatcher/registry.hpp#L33)
   — `Registry : BaseRegistry<Registry, std::string, KernelInstance>`;
   `register_kernel()` ([:58](aiter-amd/3rdparty/composable_kernel/dispatcher/include/ck_tile/dispatcher/registry.hpp#L58))
   stores a `KernelInstancePtr` with `Priority` enum
   (`Normal | High | Critical`).
2. **`HeuristicFunction`** —
   [dispatcher.hpp:40](aiter-amd/3rdparty/composable_kernel/dispatcher/include/ck_tile/dispatcher/dispatcher.hpp#L40)
   `std::function<std::vector<std::string>(const Problem&)>` returns ranked
   kernel IDs; example `size_based_heuristic` in
   [examples/04_heuristics.cpp:57-71](aiter-amd/3rdparty/composable_kernel/dispatcher/examples/gemm/cpp/04_heuristics.cpp#L57)
   returns `{"gemm_128x128", "gemm_64x64"}` based on `M*N` threshold.
3. **`SelectionStrategy` enum** —
   [dispatcher.hpp:48-52](aiter-amd/3rdparty/composable_kernel/dispatcher/include/ck_tile/dispatcher/dispatcher.hpp#L48)
   — `FirstFit` (first that supports the problem) or `Heuristic` (custom rank).
4. **Two run paths** —
   [dispatcher.hpp:83-87](aiter-amd/3rdparty/composable_kernel/dispatcher/include/ck_tile/dispatcher/dispatcher.hpp#L83)
   plain GEMM,
   [:98-103](aiter-amd/3rdparty/composable_kernel/dispatcher/include/ck_tile/dispatcher/dispatcher.hpp#L98)
   `run_fused` with multiple D-tensors.
5. **Arch filtering** —
   [registry.hpp:90](aiter-amd/3rdparty/composable_kernel/dispatcher/include/ck_tile/dispatcher/registry.hpp#L90)
   `filter_by_arch(gpu_arch)` trims the registry in-place — essential before
   benchmarking on a specific GPU.
6. **`run_explicit(kernel_id, …)`** —
   [dispatcher.hpp:116-122](aiter-amd/3rdparty/composable_kernel/dispatcher/include/ck_tile/dispatcher/dispatcher.hpp#L116)
   bypasses the heuristic for debug/offline benchmark.
7. **Mutex-protected registry** —
   [registry.hpp:12](aiter-amd/3rdparty/composable_kernel/dispatcher/include/ck_tile/dispatcher/registry.hpp#L12)
   — concurrent registration safe.

### 8.3 Library / instance manifest

1. **Hundreds of instances per dtype/layout combo** —
   [gemm/CMakeLists.txt:5-115](aiter-amd/3rdparty/composable_kernel/library/src/tensor_operation_instance/gpu/gemm/CMakeLists.txt#L5)
   enumerates 200+ `device_gemm_*_instance.cpp` files. Naming
   `device_gemm_[pipeline]_[A][B][C]_[layout].cpp`.
2. **`std::tuple<DeviceGemmDl<…>, DeviceGemmDl<…>, …>`** —
   [device_gemm_dl_f16_f16_f16_km_kn_mn_instance.cpp:31-71](aiter-amd/3rdparty/composable_kernel/library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_f16_f16_f16_km_kn_mn_instance.cpp#L31)
   — 8+ instantiations varying `BlockSize`, tile M/N, thread cluster.
3. **`MPerBlock` / `NPerBlock` / `KPerBlock` are the primary tuning knobs**
   ([:37-67](aiter-amd/3rdparty/composable_kernel/library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_f16_f16_f16_km_kn_mn_instance.cpp#L37))
   — typical sweep is 128×128, 128×64, 64×128, …, 8×8.
4. **A/B BlockTransfer params** — `ThreadSliceLengths`, `ThreadClusterLengths`,
   `SrcAccess`, `VectorDim` control coalesced global→LDS pattern.
5. **C epilogue** — `SrcDstAccess` (permute) + `DstScalarPerVector` (1, 2, 4, 8)
   controls the output write width.
6. **Irregular variants** — `device_gemm_*_irregular_instance.cpp` covers small
   M or N edge cases.
7. **Factory registration** —
   [:73-79](aiter-amd/3rdparty/composable_kernel/library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_f16_f16_f16_km_kn_mn_instance.cpp#L73)
   `add_device_gemm_dl_…_instances()` appends to a global vector via
   `add_device_operation_instances()`; linked into the shared object and called
   at init.

### 8.4 Profiler tuning knobs

1. **`profile_gemm_impl()`** —
   [profile_gemm_impl.hpp:38-49](aiter-amd/3rdparty/composable_kernel/profiler/include/profiler/profile_gemm_impl.hpp#L38)
   accepts `n_warmup`, `n_iter`, `do_verification`.
2. **`StreamConfig{nullptr, time_kernel, 0, n_warmup, n_iter}`** —
   [:173](aiter-amd/3rdparty/composable_kernel/profiler/include/profiler/profile_gemm_impl.hpp#L173)
   wraps the GPU-timer + iteration counters.
3. **Instance discovery** —
   [:117-118](aiter-amd/3rdparty/composable_kernel/profiler/include/profiler/profile_gemm_impl.hpp#L117)
   `DeviceOperationInstanceFactory<DeviceOp>::GetInstances()`; no filter,
   returns every kernel matching the signature.
4. **Reference verification** —
   [:123-140](aiter-amd/3rdparty/composable_kernel/profiler/include/profiler/profile_gemm_impl.hpp#L123)
   runs the host `ReferenceGemm` for correctness.
5. **TFLOPS / GB/s metrics** —
   [:180-182](aiter-amd/3rdparty/composable_kernel/profiler/include/profiler/profile_gemm_impl.hpp#L180)
   `tflops = 2*M*N*K / 1E9 / avg_time_ms`; `gb_per_sec = num_bytes / 1E6 / avg_time_ms`.
6. **`IsSupportedArgument()` gate** —
   [:165](aiter-amd/3rdparty/composable_kernel/profiler/include/profiler/profile_gemm_impl.hpp#L165)
   — each instance checks M/N/K divisibility against its block size; non-conformant
   instances are skipped silently.
7. **Best-instance re-profile** —
   [:223-246](aiter-amd/3rdparty/composable_kernel/profiler/include/profiler/profile_gemm_impl.hpp#L223)
   — winner is re-run with 50 warmups + 200 iters for final TFLOPS report.

### 8.5 Client example flow (dispatcher)

1. **`DECL_KERNEL_SET` macro** —
   [01_basic_gemm.cpp:44-85](aiter-amd/3rdparty/composable_kernel/dispatcher/examples/gemm/cpp/01_basic_gemm.cpp#L44)
   declares a set inline with `.add(Signature, Algorithm, arch)`.
2. **Autofill / Autocorrect / FULL patterns** —
   [01_basic_gemm.cpp:47-85](aiter-amd/3rdparty/composable_kernel/dispatcher/examples/gemm/cpp/01_basic_gemm.cpp#L47)
   — invalid `wave(1,1,1)` corrected to `(2,2,1)`; missing fields filled with
   defaults like `epilogue("cshuffle")`.
3. **`REGISTER_GENERATED_KERNELS(registry, gfx_arch)`** —
   [01_basic_gemm.cpp:93-94](aiter-amd/3rdparty/composable_kernel/dispatcher/examples/gemm/cpp/01_basic_gemm.cpp#L93)
   expands to loop over instance tuples + `registry.register_kernel(…)`.
4. **Heuristic installation** —
   [04_heuristics.cpp:97-98](aiter-amd/3rdparty/composable_kernel/dispatcher/examples/gemm/cpp/04_heuristics.cpp#L97)
   `set_strategy(Heuristic)` + `set_heuristic(size_based_heuristic)`.
5. **`dispatcher.run(a, b, c, problem, stream)`** —
   [04_heuristics.cpp:141](aiter-amd/3rdparty/composable_kernel/dispatcher/examples/gemm/cpp/04_heuristics.cpp#L141)
   returns elapsed ms; selection is internal.
6. **`registry.export_json_to_file(name)`** (in `05_json_export.cpp`) emits
   kernel metadata for offline analysis or training a learned selector.

### 8.6 Cross-cutting techniques (this slice)

- **T8.A — Two-tier kernel architecture**: library/ (pre-compiled instance
  tuples) + dispatcher/ (runtime selection + heuristic). The library tier
  decouples compile-time tile-size enumeration from runtime selection.
- **T8.B — FMHA is codegen-driven, not instance-driven**: every shipping FMHA
  kernel comes from `codegen/generate.py` over `example/ck_tile/01_fmha/`.
  There's no FMHA-specific dispatcher because the type-config struct fully
  determines the binary.
- **T8.C — `Priority` enum lets curated kernels float to top of selection**:
  `Critical > High > Normal` so a hand-tuned kernel always wins the heuristic
  tie.
- **T8.D — Heuristic = user function**: just `std::function<vector<string>
  (const Problem&)>`. Permits ML-driven selectors, table lookups, or rule-based.
- **T8.E — `IsSupportedArgument()` per-instance**: divisibility checks happen
  at the device class, not the dispatcher; selector doesn't need to know the
  detailed constraint set.
- **T8.F — Library is hand-enumerated for GEMM, generated for FMHA**: GEMM has
  ~200 hand-written instance `.cpp` files; FMHA goes through `generate.py`.
  Reflects the relative regularity of the kernel families.
- **T8.G — Profiler does the tuning** — no JIT, no autotune at the kernel
  level. The "tuning" is `IsSupported` filter + `TFLOPS sort` over the
  pre-compiled library. Production callers typically cache the winner.

Cross-references: [asm-v2:§29] (kernel selection at deploy time), §11 (codegen
that produces these instances).

<!-- SLICE-9-INSERT -->
## §9. CShuffle epilogue + LDS bank-padding + quant/permute

### 9.1 Why CShuffle exists

1. **MFMA leaves C scattered across VGPRs** — each warp holds non-contiguous
   (M×N / warp_size) elements; direct DRAM store would replay wavefronts.
2. **Three-stage shuffle (VGPR→LDS→DRAM)** —
   [gridwise_gemm_xdl_cshuffle_common.hpp:1326-1355](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_common.hpp#L1326)
   — write scattered, sync, read coalesced; guarantees aligned 128-bit DRAM stores.
3. **Bandwidth amortization** — the LDS-resident C tile becomes the natural place
   to fuse scaling, quantization, permutation, D-tensor adds — no second DRAM
   round-trip.
4. **Layout flexibility** — RowMajor / ColumnMajor / transposed / N-permuted
   outputs all expressible as a different LDS read pattern.
5. **`NumMXdlPerWavePerShuffle` / `NumNXdlPerWavePerShuffle`** —
   [cshuffle_epilogue.hpp:251-287](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp#L251)
   decouples per-warp VGPR capacity from per-block LDS bandwidth; 4-8 waves can
   overlap LDS stores while others compute the next K-block.

### 9.2 LDS bank-padding

1. **32 banks on gfx942 / 64 banks on gfx950** —
   [cshuffle_epilogue.hpp:320-343](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp#L320)
   — `#if defined(__gfx950__)` toggles padding strategy.
2. **gfx950 word-padding formula** —
   [:335-352](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp#L335):
   `MLdsLayerRequired = 32*4 / NPerIterationShuffle / sizeof(ODataType)`;
   then check `BaseWords = NPerIterationShuffle * MLdsLayer * sizeof(ODataType) /
   4`; if odd, `PadWords = 1`. Final `PaddingAmount = PadWords * elems_per_word`.
3. **Why odd vs even matters** —
   [:336-342](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp#L336)
   — gfx950's 64-bank layout groups every 2 consecutive words; padding by one
   word offsets the next row to a different bank when `BaseWords` is even.
4. **Non-gfx950 fallback `PaddingAmount = 0`** —
   [:374-384](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp#L374)
   — small per-warp tile rarely hits multiple banks per access on 32-bank archs.
5. **Stride applied through `make_naive_tensor_descriptor`** —
   [:348-350](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp#L348)
   — final row stride = `NPerIterationShuffle * MLdsLayer + PaddingAmount`
   (e.g. 130 instead of 128 for fp16, 128-wide rows).
6. **Merge-transform folds padding back to (M,N)** —
   [:363-370](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp#L363)
   — `make_merge_transform_v3_division_mod` keeps the padded layout invisible to
   the consumer; constant-folded by the compiler.
7. **Column-major mirror** —
   [:387-439](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp#L387)
   — same construction with M↔N roles swapped.
8. **Static assert for 4-byte alignment** —
   [:333, :396](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp#L333)
   `(BaseStrideElems * DataTypeSize) % 4 == 0` — guarantees compiler emits
   aligned `ds_write`.
9. **Classic CK XOR-bank transform** —
   [gridwise_gemm_xdl_cshuffle_common.hpp:177-211](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_common.hpp#L177)
   — applies `make_xor_with_modulo_transform(M/MLdsLayer, AK0*MLdsLayer)` to the
   *A-LDS* descriptor (different mechanism, same goal).

### 9.3 CK_tile `cshuffle_epilogue`

1. **`MakeLdsDistributionEncode()`** —
   [cshuffle_epilogue.hpp:446-512](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp#L446)
   — explicit (wave, thread-in-wave, xdl-output-id) → LDS-coord mapping.
2. **`BlockedXDLN_PerWarp != 1` blocked layout** —
   [:469-493](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp#L469)
   — keeps per-thread xdl outputs contiguous; required for gfx950 a16w4.
3. **`slice_acc_tile`** —
   [:563-579](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp#L563)
   — `get_y_sliced_thread_data()` extracts only the row/col range for this
   shuffle iter.
4. **Three sync points** —
   [:741-758](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp#L741)
   `block_sync_lds()` before VGPR→LDS, after LDS write, implicit before
   LDS→DRAM.
5. **Cast inside LDS** —
   [:582-587](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp#L582)
   `cast_lds_tile()` converts AccDataType→ODataType while still in LDS — no
   DRAM round-trip for fp32→fp8/fp16.
6. **Per-row/column scale apply** —
   [:520-560](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp#L520)
   — `scale_m` / `scale_n` tile windows multiplied via
   `tile_elementwise_inout`; scalar scales skip window creation.
7. **D-tensor element-wise fusion** —
   [:589-601](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp#L589)
   — bias / residual / activation tensors loaded from DRAM and folded into the
   shuffled tile before store.
8. **`memory_operation_enum::{set, update}`** —
   [:603-616](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp#L603)
   `set` = overwrite, `update` = atomic add (split-K reduce).
9. **SMEM safeguard** —
   [:232-243](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp#L232)
   `AlignShuffleTileWithSmem()` caps shuffle tile to (1,1) if it overflows
   capacity or `DoubleSmemBuffer` is set.

### 9.4 Quantization in epilogue (`dynamic_quant_epilogue`)

1. **Absmax row-reduce inline** —
   [dynamic_quant_epilogue.hpp:120-144](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/dynamic_quant_epilogue.hpp#L120)
   — warp reduce + cross-warp LDS sync inside the epilogue; no separate scale
   kernel.
2. **`v_max3_f32` absmax** —
   [:126-135](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/dynamic_quant_epilogue.hpp#L126)
   — inline asm `v_max3_f32 %0, %1, abs(%2), abs(%3)`.
3. **Smooth pre-scale** —
   [:186-199](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/dynamic_quant_epilogue.hpp#L186)
   — `o_acc *= sm_scale[j]` before absmax; reduces quant noise.
4. **`y_scale = row_absmax / numeric<ODataType>::max()`** —
   [:147-151](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/dynamic_quant_epilogue.hpp#L147).
5. **Per-row divide-then-cast** —
   [:155-168](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/dynamic_quant_epilogue.hpp#L155)
   `o_acc /= y_scale[row]; cast_tile<ODataType>(o_acc)`.
6. **`y_scale_window` writeback** —
   [:153](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/dynamic_quant_epilogue.hpp#L153)
   — scales emitted as a separate output tensor for downstream dequant.
7. **Per-token vs per-channel** —
   [:75-100](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/dynamic_quant_epilogue.hpp#L75)
   — `MakeSmoothInputScaleTileDistribution` chooses axis.

### 9.5 PermuteN epilogue + batched-transpose

1. **VGPR→VGPR shuffle, no LDS** —
   [permuten_epilogue.hpp:204](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/permuten_epilogue.hpp#L204)
   `GetSmemSize() = 0`.
2. **`IntrThreadShuffleEncode`** —
   [:259-265](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/permuten_epilogue.hpp#L259)
   factorizes accumulator layout `(MWave, MPerXdl/RowsPerLane, RowsPerLane,
   NWave, NPerXdl, NRepeat)`.
3. **N-permute index swap** —
   [:331-354](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/permuten_epilogue.hpp#L331)
   `src = n_idx*plane + m_lane`; `dst = n_idx + m_lane*NRepeat`.
4. **Scale-then-cast in shuffle** —
   [:336-353](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/permuten_epilogue.hpp#L336)
   — scale multiply before `cast<ODataType>` minimizes rounding error.
5. **Static_assert `NumDTensor == 0`** —
   [:119-125](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/epilogue/permuten_epilogue.hpp#L119)
   — D-tensors require the full `cshuffle_epilogue` path.

### 9.6 Classic-CK CShuffle write-back

1. **A/B LDS XOR-bank transform** —
   [gridwise_gemm_xdl_cshuffle_common.hpp:177-211](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_common.hpp#L177).
2. **`MLdsLayer = max(1, 32*4 / KPerBlockInByte)`** —
   [:178-184](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_common.hpp#L178)
   folds 4-byte LDS rows to align with 32-bank banks.
3. **`GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock()`** —
   [:461-473](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_common.hpp#L461)
   — naive packed LDS tile `(1, MShuf*MWave*MPerXdl, 1, NShuf*NWave*NPerXdl)`.
4. **`GetCThreadCopyVgprToLds`** —
   [:1272-1273](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_common.hpp#L1272)
   — ThreadwiseTensorSliceTransfer with optional `vgpr_to_lds_element_op`.
5. **Per-warp SFC over xdl outputs** —
   [:1310-1320](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_common.hpp#L1310)
   `GetCThreadWiseSpaceFillingCurve<TransposeC>()`.
6. **`ThreadGroupTensorSliceTransfer_v6r1` for LDS→DRAM** —
   [:1282-1307](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_common.hpp#L1282)
   coordinates the whole block; `VectorDim=3` (N-contiguous).
7. **`DoElementwiseBeforeCShuffle`** —
   [:1239-1257](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_common.hpp#L1239)
   — trade register pressure for memory bandwidth: fuse element op into VGPR→LDS
   (fast) vs LDS→DRAM (deferred).
8. **`gridwise_gemm_xdl_cshuffle_v3_2lds`** —
   [v3.hpp:75-86](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3.hpp#L75)
   — two shared pointers for overlapped A/B-LDS vs C-LDS access.
9. **Split-K reduction via `CGlobalMemoryDataOperation`** —
   [v3.hpp:29-51](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3.hpp#L29)
   `set` vs `update` (atomic_add) — single epilogue path covers both reduction
   modes.

### 9.7 Cross-cutting techniques (this slice)

- **T9.A — Word-aware bank padding on gfx950**: only-when-odd padding logic
  keeps the layout dense while breaking the 2-word bank group.
- **T9.B — `make_xor_with_modulo_transform` for A/B-LDS**: classic CK's analog
  — different mechanism (XOR address) than CK_tile's stride padding, same
  end (no bank conflicts).
- **T9.C — Cast / scale / quant *inside* the LDS hop**: avoids a second DRAM
  round-trip; absmax + divide + cast all happen between the LDS write and the
  DRAM store.
- **T9.D — `memory_operation_enum` unifies set/update**: same epilogue path
  serves overwrite and split-K atomic-add; no separate reduce kernel.
- **T9.E — `BlockedXDLN_PerWarp != 1` for gfx950 a16w4**: special LDS
  distribution that keeps per-thread xdl outputs contiguous so the cast to fp4
  doesn't shuffle inside a thread.
- **T9.F — `PermuteN` skips LDS entirely**: simple N-permutation done via
  VGPR-only intrathread shuffle; useful for transpose-only epilogues.
- **T9.G — `DoElementwiseBeforeCShuffle` knob**: trade VGPR pressure for
  DRAM-bandwidth savings; element-op runs on hotter side of the LDS depending
  on the choice.
- **T9.H — SMEM safeguard auto-downsizes shuffle tile**: prevents register
  spill on borderline configs without user intervention.

Cross-references: [asm-v2:§13] (LDS bank conflict avoidance),
[asm-v2:§19] (CShuffle in classic CK), [pdf:p162-165] (LDS organization).

<!-- SLICE-10-INSERT -->
## §10. Scheduling + LDS-layout patterns (cross-pipeline)

This slice catalogs the scheduling hints, hazard-padding, and LDS-layout tricks
*across all* CK / CK_tile pipelines — the patterns that aren't tied to one kernel
family.

### 10.1 `__builtin_amdgcn_iglp_opt` modes

1. **Single visible call site** —
   [gridwise_gemm_pipeline_v2.hpp:82-84](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v2.hpp#L82)
   gated by `#if CK_EXPERIMENTAL_PIPELINE_V2_IGLP_OPT`.
2. **Mode integer** — argument is `CK_EXPERIMENTAL_PIPELINE_V2_IGLP_OPT` (default
   0); non-zero → tells the LLVM scheduler to apply *Inter-Group Latency
   Propagation* re-grouping at this point.
3. **Placement** — emitted right before `block_sync_lds()` in the V2 hot loop
   so the scheduler can re-rank the pending ops before the sync barrier.
4. **No GFX guard** — applies to whatever target is currently being built;
   experimental, off by default.

### 10.2 `sched_barrier(0)` fences vs `sched_group_barrier`

1. **`__builtin_amdgcn_sched_barrier(0)`** — unconditional fence; no mask, no
   count. Forbids the scheduler from moving any instruction across this point.
2. **WMMA pipelines (9 sites)** —
   [blockwise_gemm_pipeline_wmmaops_v3.hpp:375, 416, 476, 530, 609, 679, 758, 833](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v3.hpp#L375)
   — wrap `RunWrite`/`RunRead` so the LDS write fully commits before the
   subsequent GEMM compute.
3. **FMHA bias path** —
   [block_fmha_pipeline_qr_ks_vs.hpp:515-522](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L515)
   — surrounds the `ELEMENTWISE_BIAS` load so the bias DRAM read doesn't migrate
   into the QK GEMM.
4. **vs `sched_group_barrier(mask, count, wait_state)`** — `sched_barrier(0)`
   is the "kill switch"; `sched_group_barrier` is the "shape the schedule"
   directive (§3).

### 10.3 LDS layout pattern catalog

1. **XOR-bank transform** —
   [gemm_universal_pipeline_ag_bg_cr_policy.hpp:289-290, 489-490](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp#L289)
   `make_xor_transform(make_tuple(...))` — XOR thread indices with high address
   bits to break strided bank conflicts on 32-bank LDS.
2. **`MLdsLayer` multi-layer fold** —
   [:270-280](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp#L270)
   — LDS subdivided into `MLdsLayer` rows-stacks in K dim; reduces full-occupancy
   thrashing.
3. **`ds_read_tr*` transpose-on-read** —
   [amd_buffer_addressing_builtins.hpp:3029-3049](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp#L3029):
   `ds_read_tr16_b64_v4f16` / `ds_read_tr16_b64_v4bf16` /
   `ds_read_tr8_b64_v2i32` / `ds_read_tr4_b64_v2i32` — free transpose for fp16
   / bf16 / 8-bit / 4-bit packed types.
4. **`unmerge_transform` for layered LDS** —
   [fmha_fwd_kernel.hpp:2619](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp#L2619)
   `make_unmerge_transform(make_tuple(number<LDSLayerSize/XorGroupSize>{}, …))`
   splits a single LDS extent into `(layer, group)` logical dims.
5. **CShuffle word-padding** (covered in §9.2).
6. **A/B-LDS XOR-modulo (classic CK)** (covered in §7.7 / §9.6.1).
7. **Combined XOR + unmerge**: outer XOR on thread coord, inner unmerge for
   factorization — lets `ds_read_tr*` hit 16-lane coalescence on reshaped tiles.

### 10.4 Inline-asm clobber idioms

1. **`"memory"` for M0 async-copy fence** —
   [utility.hpp:21, 27](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/utility.hpp#L21):
   ```
   asm volatile("s_mov_b32 m0, %0" : : "s"(v) : "memory");
   asm volatile("s_add_u32 m0, %0, m0" : : "n"(v) : "memory");
   ```
   The clobber tells the compiler "do not reorder loads/stores across this op"
   — required because the subsequent `global_load_lds` reads M0 implicitly.
2. **`+v` accumulator output** — buffer-load templates use `"+v"(payload)` so
   the compiler treats the destination VGPR as both read and write; prevents
   it from being killed across the asm.
3. **`"s"` constraint for SGPR-only operands** — V# buffer resource always
   binds to `"s"` so the compiler emits it via an SGPR; mismatches are
   compile-time errors.
4. **`offset:%N` immediate constraint** — the 12-bit imm offset of `buffer_*`
   instructions is fed via `"n"` constraint; lets the compiler fold compile-time
   constants into the instruction encoding.

### 10.5 `s_nop` insertion points

1. **`s_nop 2` after `v_dot2_f32_f16` / `v_dot4_i32_i8`** —
   [inner_product.hpp:95-100, 186-191](aiter-amd/3rdparty/composable_kernel/include/ck/utility/inner_product.hpp#L95)
   ```
   asm volatile("\n v_dot2_f32_f16 %0, %1, %2, %0\n s_nop 2 \n"
                : "=v"(c) : "v"(a), "v"(b), "0"(c));
   ```
   Hides the 3-cycle DOT-instruction latency before the next dependent op.
2. **`s_nop 4` before `buffer_load_dwordx4`** —
   [amd_buffer_addressing_builtins.hpp:165-174](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp#L165)
   (and 9 other sites for the smaller width templates) — `pre_nop` template
   flag emits `s_nop 4` so the prior MFMA's VGPR write is visible before the
   load.
3. **`s_nop 0 × 16` for I-cache flush** —
   [flush_icache.hpp:12-27](aiter-amd/3rdparty/composable_kernel/include/ck/utility/flush_icache.hpp#L12)
   — used only after self-modifying code; rare in production.
4. **Generic `s_nop()` wrapper** —
   [synchronization.hpp:61-70](aiter-amd/3rdparty/composable_kernel/include/ck/utility/synchronization.hpp#L61)
   wraps `"s_nop 0\n"`; alternative to `sched_barrier(0)` where a hard stall is
   wanted.

### 10.6 `sched_group_barrier` mask catalog (cross-pipeline)

1. **MFMA/WMMA (`0x008`)** — most common; appears in every pipeline
   ([blockwise_gemm_pipeline_xdlops.hpp:392-429](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops.hpp#L392),
   [block_fmha_pipeline_qr_ks_vs.hpp:450-454](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L450)).
2. **DS_READ (`0x100`)** —
   [blockwise_gemm_pipeline_xdlops.hpp:428, 449](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops.hpp#L428),
   plus FMHA uses with variable `wait_cnt` for rate-limiting concurrent reads.
3. **DS_WRITE (`0x200`)** —
   [block_fmha_bwd_pipeline_trload_default_policy.hpp:1196](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_trload_default_policy.hpp#L1196)
   `__builtin_amdgcn_sched_group_barrier(0x200, 1, 0)`. Used in fwd-bwd
   chained kernels.
4. **VMEM_READ (`0x020`)** —
   [blockwise_gemm_pipeline_xdlops.hpp:399](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops.hpp#L399),
   [trload_default_policy.hpp:1155](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_trload_default_policy.hpp#L1155).
5. **FMHA mask constants** —
   [block_fmha_pipeline_qr_ks_vs.hpp:79-80](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp#L79)
   `DS_READ = 0x100; MFMA = 0x008;` — exported as `static constexpr` so the
   sched-group args are readable.
6. **No `VMEM_WRITE (0x040)`** in the visible code — CK relies on `s_waitcnt`
   for global-store ordering instead of grouping VMEM writes.
7. **`(mask, count, wait_state)` rate-limit** — the `count` is the *number of
   instructions in this group*, not "wait for N to retire"; the `wait_state`
   (third arg) is almost always 0 = "current pipeline only".

### 10.7 Cross-cutting techniques (this slice)

- **T10.A — `iglp_opt` as a single-knob scheduler nudge**: experimental but
  pragmatically used as a "rearrange this kernel" lever in V2.
- **T10.B — `sched_barrier(0)` is the kill-switch**; reserve for when you
  *know* the schedule must not migrate, e.g. around a register-pressure-cliff.
- **T10.C — XOR + unmerge composes to "free transpose on read"**: `ds_read_tr`
  + XOR-bank + unmerge-layer lets a column-major matrix be read into row-major
  VGPR registers with no extra ALU.
- **T10.D — `"memory"` clobber on M0**: discipline that keeps the compiler
  from sneaking other LDS ops across the M0 update — the cost of forgetting
  this is silent data corruption.
- **T10.E — `pre_nop = true` on buffer_load**: opt-in, used by the pipelines
  that know the prior MFMA's VGPR write is still in flight.
- **T10.F — `wait_state = 0` is always used** — CK never asks the scheduler
  to wait across pipelines (it manages cross-pipeline timing via `s_waitcnt`).
- **T10.G — Mask + count pair is *grouping*, not wait-count**: a common
  reading mistake; the count is how many in-group instructions before the
  next group, not "wait for N to finish."

Cross-references: [asm-v2:§22] (sched_group_barrier), [asm-v2:§21] (s_nop),
[asm-v2:§13] (LDS bank conflicts), [pdf:p195] (sched_barrier intrinsics),
[pdf:p159] (ds_read_tr encodings).

<!-- SLICE-11-INSERT -->
## §11. Build + codegen + instance archive

The story of how 1,981 instance `.cpp` files become a single `libck.so` — and
why that pipeline is itself an optimization knob.

### 11.1 CMake feature gates

1. **`CK_USE_XDL`** —
   [CMakeLists.txt:265-279](aiter-amd/3rdparty/composable_kernel/CMakeLists.txt#L265)
   — auto-on for gfx9/11/12; force-off via `FORCE_DISABLE_XDL`.
2. **`CK_USE_WMMA`** —
   [:288-293](aiter-amd/3rdparty/composable_kernel/CMakeLists.txt#L288)
   — gfx11/12 only; `CK_TILE_USE_WMMA=1` toggles MFMA↔WMMA in tile examples.
3. **Per-fp8 family gates** —
   [:318-330](aiter-amd/3rdparty/composable_kernel/CMakeLists.txt#L318)
   — TF32 = gfx942+, FNUZ-fp8 = gfx90a/94, OCP-fp8 = gfx12/950, MX = gfx950
   only. Prevents codec-mismatch silently miscompiling.
4. **`DTYPES` filter** —
   [:84-129](aiter-amd/3rdparty/composable_kernel/CMakeLists.txt#L84)
   — `cmake -DDTYPES="fp16;bf16"` skips entire families of instances; default
   compiles all 7.
5. **LLVM tuning flags** —
   [:336-367](aiter-amd/3rdparty/composable_kernel/CMakeLists.txt#L336)
   — `-mllvm -amdgpu-early-inline-all=true` etc; version-conditional.
6. **`CK_PARALLEL_LINK_JOBS` / `CK_PARALLEL_COMPILE_JOBS`** —
   [:372-392](aiter-amd/3rdparty/composable_kernel/CMakeLists.txt#L372)
   — required because the link farm and per-instance template instantiation
   each peak at 50–100 GB RAM per job.
7. **`CK_EXPERIMENTAL_BUILDER`** —
   [:51, :59-62](aiter-amd/3rdparty/composable_kernel/CMakeLists.txt#L51)
   — gates the new `tile_engine` builder.
8. **Library-level instance filter** —
   [library/.../gpu/CMakeLists.txt:66-120](aiter-amd/3rdparty/composable_kernel/library/src/tensor_operation_instance/gpu/CMakeLists.txt#L66)
   — `add_instance_library()` matches filenames against `_xdl`, `_wmma`, `_mx`,
   `_mha` and drops them when the GPU target doesn't support the kernel.
9. **`MIOPEN_REQ_LIBS_ONLY` / `HIPTENSOR_REQ_LIBS_ONLY`** —
   [:674-678](aiter-amd/3rdparty/composable_kernel/CMakeLists.txt#L674)
   — strip examples for downstream consumers.

### 11.2 codegen Python pipeline

1. **`unified_gemm_codegen.py`** — emits both kernel `.cpp` and dispatcher
   wrapper in a single pass; keeps the registry in lock-step with the binary.
2. **TileConfig × TraitConfig sweep** — `(tile_m, tile_n, tile_k, warp_m,
   warp_n, warp_k, warp_tile_*)` × `(pipeline ∈ {mem, compv3..compv6,
   comp_async, basic_async_v1}, epilogue ∈ {cshuffle, default}, scheduler ∈
   {intrawave, interwave})` — hundreds of variants per dtype.
3. **Validation prunes the cross-product** —
   `codegen_common.py:85-103` — disallowed combos (e.g. `compv3 + interwave`)
   short-circuit before code is emitted.
4. **Preshuffle alignment check** —
   `unified_gemm_codegen.py:61-100` — vector-load + M0/M1/M2 must be 16B-aligned.
5. **FMHA codegen** —
   [example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py:1-120](aiter-amd/3rdparty/composable_kernel/example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py#L1)
   — sweeps `(bm0, bn0, bk0, bn1, bk1)` × register tile × warp tile; emits
   arch-specific code (`#if defined(__gfx1100__)`).
6. **`ArchTrait`** — `arch.py:1-43` — central preprocessor-check + namespace
   tag table; `get_factories_for_targets()` orders gfx90a before gfx9 so a
   specific match wins.
7. **`ThreadPoolExecutor` parallel emit** —
   `unified_gemm_codegen.py:26` — generates the GEMM variant zoo in seconds.
8. **Idempotent `update_file()`** —
   `codegen/utils.py:9-23` — writes only if content differs, so incremental
   rebuilds don't churn the whole archive.
9. **Centralized type maps** — `codegen_common.py:114-139` — single source of
   truth for `fp16 → fp16_t` etc., used by both GEMM and conv codegen.

### 11.3 Instance archive

1. **1,981 instance `.cpp` files** across ~100 op directories;
   ~85 K LOC compiled total.
2. **Dtype distribution** — fp16 64%, bf16 46%, fp32 19%, int8 2%, fp8 2%,
   bf8 1% (overlap: many instances cover multiple dtypes via template).
3. **Naming convention** —
   `device_<op>_<algo>_<a>_<b>_<c>_<layout>_instance.cpp` (e.g.
   `device_gemm_xdl_f16_f16_f16_km_kn_mn_instance.cpp`).
4. **One file = many kernels** — each `.cpp` defines `std::tuple<DeviceGemm<…>,
   DeviceGemm<…>, …>` with 8–12 specializations differing only in block tile
   and warp config; the linker dedups symbols across files.
5. **Manual CMakeLists per-op** — `gemm/CMakeLists.txt:1-150` hand-lists 100+
   instance files; adding a tile requires re-running codegen *and* editing
   CMakeLists.
6. **Filename-regex filter at configure time** — operations matching `_xdl`
   on a gfx11 build are dropped; one binary stays target-locked.
7. **Cross-target fat binary is *not* supported** — each library build pins to
   one arch (or close family).
8. **Memory ceiling = real constraint** — full build needs `-jN` carefully
   chosen; each parallel compile peaks ~50 GB.

### 11.4 `Dockerfile.aiter`

1. **`rocm/pytorch:latest` base** —
   [Dockerfile.aiter:1](aiter-amd/3rdparty/composable_kernel/Dockerfile.aiter#L1)
   — full ROCm + PyTorch stack pre-installed.
2. **Pinned deps**: `numpy==1.26.2`, plus `pandas / zmq / einops / ninja /
   tabulate / vcs_versioning` ([:7](aiter-amd/3rdparty/composable_kernel/Dockerfile.aiter#L7)).
3. **Sparse CK checkout** —
   [:12-20](aiter-amd/3rdparty/composable_kernel/Dockerfile.aiter#L12)
   — `git sparse-checkout` of `projects/composablekernel/` from `rocm-libraries`
   when `CK_FROM_ROCM_LIBRARIES=1`.
4. **Decoupled branches** —
   [:4, :18](aiter-amd/3rdparty/composable_kernel/Dockerfile.aiter#L4)
   `AITER_BRANCH` and `CK_AITER_BRANCH` so CK hotfixes don't force an aiter
   change.
5. **CK replaces `3rdparty/composable_kernel/`** —
   [:32](aiter-amd/3rdparty/composable_kernel/Dockerfile.aiter#L32).
6. **`python3 setup.py develop`** —
   [:34](aiter-amd/3rdparty/composable_kernel/Dockerfile.aiter#L34)
   — aiter installed editable.
7. **No baked CK build** — the image is a *codegen + launcher* environment;
   CK is compiled on demand inside the container.

### 11.5 `example/ck_tile/` tour

1. **`01_fmha`** — codegen-driven; `codegen/{arch, utils, ops/*}.py` emits all
   FMHA variants per arch + pipeline.
2. **`03_gemm`** — uses `unified_gemm_codegen.py`; toggles
   `CK_TILE_USE_MFMA` / `CK_TILE_USE_WMMA` at CMake-time.
3. **`#if defined(__gfx1100__)` guards** in emitted code — example of
   per-target buffer-load tricks (fast-exp2, OOB hacks).
4. **Tile parameters surface optimization knobs** — examples instantiate
   specific `(BM, BN, BK)` so tuning is "edit codegen → rebuild → measure."
5. **`hipcc` + add_executable** — each example links against CK headers
   directly; no precompiled-lib dependency.
6. **`19_gemm_multi_d`** — stress-tests epilogue fusion across dtype+pipeline
   combos; codegen must validate all fusions.

### 11.6 Cross-cutting techniques (this slice)

- **T11.A — Compile-time gating dominates runtime dispatch**: most CK
  decisions are baked at `cmake -D…` configure; the runtime selector (§8)
  picks among the survivors.
- **T11.B — Multi-specialization `std::tuple` per `.cpp` file**: shifts
  template-instantiation work from the linker (sequential) to the compiler
  (parallel). Keeps build wall-time tractable.
- **T11.C — Idempotent codegen**: `update_file()` writes only on diff. Without
  this, incremental rebuilds re-compile all 1,981 instances on every edit.
- **T11.D — Arch-parameterized emit, not arch-parameterized dispatch**:
  `ArchTrait` decides at *emit* time which preprocessor checks live in the
  source; runtime never sees a chain of `if (device_name == …)`.
- **T11.E — Sparse-checkout for downstream consumers**: aiter pulls only
  `projects/composablekernel/`, not the full `rocm-libraries` tree — order-of-
  magnitude smaller clones.
- **T11.F — Memory ceiling is a tuning param**: `CK_PARALLEL_COMPILE_JOBS`
  prevents OOM. Effectively, the build system is itself rate-limited and the
  knob is operator-known.
- **T11.G — Codegen iteration is the tuning workflow**: developers edit
  `codegen/ops/*.py`, regenerate, rebuild, benchmark — the fast iteration
  loop lives in Python, not C++.

Cross-references: [asm-v2:§28] (technique catalog), §3 (which pipelines codegen
emits), §8 (how instances are dispatched).

<!-- SLICE-12-INSERT -->
## §12. Cross-cutting technique catalog + reading order

This section is the index. Every technique cited across §1–§11 is tagged
`T<§>.<letter>` so search and cross-reference is trivial.

### 12.1 One-line technique index

| Tag | Technique | Section |
|---|---|---|
| T1.A | Compile-time distribution descriptor (O(1) thread-ID→VGPR) | §1.1 |
| T1.B | VGPR-resident tile via `static_distributed_tensor` | §1.3 |
| T1.C | `pre_computed_coords_` + `NumCoord` amortization | §1.2 |
| T1.D | Buffer instruction immediate-offset (no SGPR spill) | §1.4 |
| T1.E | SFC + M0-per-warp = LDS bank parallelism | §1.5 |
| T1.F | `Traits::VectorDimY` auto-vectorization | §1.2 |
| T1.G | `oob_conditional_check` template ⇒ bounds-check DCE | §1.2 |
| T2.A | `WaitcntLayout*` + `enable_if_target_id_t` compile-time arch select | §2.1 |
| T2.B | Three waitcnt strategies (early / lazy / fused) | §2.2 |
| T2.C | gfx9 vs gfx950 MFMA K-width trade-off | §2.3 |
| T2.D | `global_load_lds` + M0 discipline | §2.4 |
| T2.E | `ds_read_tr*` free transpose | §2.4 |
| T2.F | Wave-0 polling barrier | §2.5 |
| T2.G | `"memory"` clobber as scheduling fence | §2.5 |
| T3.A | Async load skips VGPRs | §3.6 |
| T3.B | `sched_group_barrier` counts derived at compile time | §3.3 |
| T3.C | Adaptive `ds_read_*_mfma_rate` (V6) | §3.3 |
| T3.D | Register double-buffer with 3-stage skew | §3.2 |
| T3.E | Distinct sync flavors for distinct RAW hazards | §3.4 |
| T3.F | `ds_read_tr*` paired with LDS-shape swap | §3.5 |
| T3.G | Preshuffled B for 8-warp WG | §3.1 |
| T3.H | Intrawave scheduler specialization | §3.6 |
| T3.I | Static instruction-count forecast | §3.6 |
| T4.A | Tile-range elision + `set_tile_if` for masks | §4.4 |
| T4.B | Three exp paths chosen at compile time | §4.3 |
| T4.C | FP8 shift trick keeps `exp2(s-m)` in range | §4.3 |
| T4.D | Logsumexp split-reduction | §4.5 |
| T4.E | `kr_ktr_vr` LDS layout for free Kᵀ in bwd | §4.6 |
| T4.F | Page-block navigator at tile-window level | §4.8 |
| T4.G | Per-block descale folded into `scale_s` | §4.7 |
| T4.H | `kBlockPerCu` chosen by hdim | §4.1 |
| T4.I | Reduce-then-scale O (pre-add rescale) | §4.3 |
| T4.J | Sink-token via virtual prefix | §4.4 |
| T5.A | Preshuffle weight layout aligned to quant groups | §5.2 |
| T5.B | Descale folded into accumulator FMA | §5.2 |
| T5.C | `ds_bpermute` cross-lane scale broadcast | §5.2 |
| T5.D | OpSel byte-select replaces explicit descale | §5.3 |
| T5.E | Dual accumulator (gate-up + down fusion) | §5.1 |
| T5.F | Atomic-add MoE epilogue + sorted-expert | §5.1 |
| T5.G | Flatten batch×seq into M | §5.4 |
| T5.H | Per-K-iter `SchedulerPerM` | §5.4 |
| T5.I | Tensor-descriptor builders on-device | §5.5 |
| T5.J | Reused warp-GEMM core | §5.6 |
| T6.A | Single reduction primitive feeds all op families | §6.1 |
| T6.B | Welford one-pass with `kFastFDiv` | §6.3 |
| T6.C | `v_max3_f32` for absmax everywhere | §6.3 |
| T6.D | `kNeedCrossWarpSync` routing knob | §6.3 |
| T6.E | Three-pass `add+rmsnorm+rdquant` collapse | §6.3 |
| T6.F | Streaming topk avoids sort | §6.4 |
| T6.G | Pack `(value, index)` for argmax | §6.4 |
| T6.H | Bit-flag traits ⇒ DCE | §6.3 |
| T7.A | Six-level template stack | §7.7 |
| T7.B | `PipelineVersion` arch-aware default | §7.6 |
| T7.C | Double-LDS via independent `__shared__` arrays | §7.7 |
| T7.D | `Block2CTileMap` cache-friendly grid | §7.7 |
| T7.E | C in VGPRs via `StaticBufferTupleOfVector` | §7.3 |
| T7.F | Element-wise ops as function objects | §7.3 |
| T7.G | Inline-asm outer product (thread-level) | §7.5 |
| T7.H | Split-K via atomic merge | §7.1 |
| T8.A | Two-tier kernel architecture (library + dispatcher) | §8.6 |
| T8.B | FMHA codegen-driven, GEMM instance-driven | §8.6 |
| T8.C | `Priority` enum for curated kernels | §8.2 |
| T8.D | Heuristic = user function (allows ML selectors) | §8.2 |
| T8.E | `IsSupportedArgument()` per-instance | §8.4 |
| T8.F | Library hand-enumerated for GEMM, generated for FMHA | §8.6 |
| T8.G | Profiler does the tuning (no JIT) | §8.4 |
| T9.A | Word-aware bank padding on gfx950 | §9.2 |
| T9.B | `make_xor_with_modulo_transform` (classic CK) | §9.6 |
| T9.C | Cast / scale / quant *inside* the LDS hop | §9.3 |
| T9.D | `memory_operation_enum` unifies set/update | §9.3 |
| T9.E | `BlockedXDLN_PerWarp != 1` for gfx950 a16w4 | §9.3 |
| T9.F | `PermuteN` skips LDS entirely | §9.5 |
| T9.G | `DoElementwiseBeforeCShuffle` knob | §9.6 |
| T9.H | SMEM safeguard auto-downsizes shuffle tile | §9.3 |
| T10.A | `iglp_opt` as scheduler nudge | §10.1 |
| T10.B | `sched_barrier(0)` is the kill-switch | §10.2 |
| T10.C | XOR + unmerge ⇒ free transpose on read | §10.3 |
| T10.D | `"memory"` clobber on M0 (discipline) | §10.4 |
| T10.E | `pre_nop=true` on buffer_load | §10.5 |
| T10.F | `wait_state=0` for sched-group | §10.6 |
| T10.G | Mask+count is *grouping*, not wait-count | §10.6 |
| T11.A | Compile-time gating > runtime dispatch | §11.6 |
| T11.B | Multi-specialization `std::tuple` per `.cpp` | §11.3 |
| T11.C | Idempotent codegen (`update_file()`) | §11.2 |
| T11.D | Arch-parameterized *emit*, not dispatch | §11.2 |
| T11.E | Sparse-checkout for downstream consumers | §11.4 |
| T11.F | Memory ceiling = real build-system tuning | §11.1 |
| T11.G | Codegen iteration is the tuning workflow | §11.5 |

### 12.2 Reading order by use case

**You want to read a CK_tile kernel from scratch**
→ §1 (core abstractions) → §2 (arch wrappers) → §3 (GEMM pipelines) — these
three are *foundational*. Then jump to the op family you care about (§4 FMHA /
§5 MoE-quant / §6 reduce-norm).

**You want to *modify* a CK_tile kernel for a new shape**
→ §1.5 (SFC = iteration order knob) → §3.5 (pipeline policy / prefetch /
k-loop tail) → §10 (scheduling masks). The first thing you change is usually
the pipeline class or tile sizes, not the kernel body.

**You want to find an optimization opportunity**
→ §3.3 (sched_group_barrier patterns) → §9.2 (LDS bank-padding) →
§10 (scheduling catalog) → §4.3 (FMHA softmax variants). Most wins come from
re-ordering loads vs MFMA, or from breaking bank conflicts.

**You want to consume CK from a Triton/HIP backend**
→ §8 (dispatcher + library) → §11 (build + codegen) — these explain the
deploy/select pipeline. Read §11.5 to see how example kernels are wired
end-to-end.

**You're chasing a numerical bug**
→ §4.3 (online softmax with rescale formulas) → §6.3 (Welford one-pass) →
§9.4 (quantization in epilogue). FMHA exp2 fast-path and fp8 shift are
common suspects.

**You want classic CK history / context**
→ §7 (device-instance hierarchy + V1→V2→V3 evolution) → §8.6 (why classic
FMHA was retired). CK_tile DSL replaces the explicit descriptor plumbing of
the classic stack.

### 12.3 The seven "always-true" rules

These appear in *every* CK family — print them on a sticky note:

1. **C lives in VGPRs** (`static_distributed_tensor` or
   `StaticBufferTupleOfVector`). Never spill to LDS.
2. **LDS is double-buffered or has XOR/word-pad bank avoidance.** Plain LDS
   layouts cost 20–40 % throughput on gfx942/950.
3. **Async load (`global_load_lds`) bypasses VGPRs** wherever the pipeline
   variant supports it. Pair with `vmcnt` drain, not `lgkmcnt`.
4. **`sched_group_barrier` counts are derived from problem geometry at
   compile time.** Hand-tweaking a runtime count is wrong by construction.
5. **`ds_read_tr*` is a free transpose** — design the LDS layout so K↔M (or
   K↔N) accesses match a tr variant; avoid ALU shuffles.
6. **Quantization / scaling fuses into the accumulator FMA or OpSel** — the
   epilogue is the right place to dequantize, not a separate pass.
7. **Compile-time arch selection > runtime dispatch.** `enable_if_target_id_t`
   and `WaitcntLayout*` patterns keep one source compiling for the whole
   gfx9/10/11/12 family without a single runtime branch.

### 12.4 Status

This playbook covers the CK submodule at
`composable_kernel @ fdf4bb7fcc984811cef48ce817d89aac064b984a`
(parent `aiter-amd @ 3cbdcb371b`). All 12 slices were synthesized from focused
deep-read passes over `include/ck_tile/` and `include/ck/`, with cross-checks
against the upstream ROCm CK and the AMD CDNA4 ISA spec PDF (citations
`[pdf:pN]`). Future CK updates (new MFMA encodings, new pipelines) should be
analyzed in fresh sections appended after §12, with the submodule pin bumped
in `.gitmodules`.

---

## §13. Grouped convolution + image_to_column + pooling

Convolution in CK_tile is **lowered to GEMM** via descriptor algebra: pad +
embed + merge transforms turn an (N,H,W,C) input into an `(M=N·Hout·Wout, K=C·Y·X)`
matrix view, and the rest is §3. The same machinery powers both forward and the
two backward paths; pooling reuses the descriptor pad/embed but feeds
`block_tile_reduce` instead of warp-GEMM.

### 13.1 Grouped convolution forward

1. **Kernel entry** —
   [grouped_convolution_forward_kernel.hpp:28-100](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_forward_kernel.hpp#L28)
   `GroupedConvFwdKernelArgs` carries descriptor makers; the kernel never sees
   raw strides.
2. **Conv → GEMM mapping**: `M = N · Hout · Wout · merged_groups`,
   `K = C · Y · X`, `N = Kout`.
3. **Data layouts** —
   [tensor_layout.hpp:104-111](aiter-amd/3rdparty/composable_kernel/include/ck_tile/utility/tensor_layout.hpp#L104)
   — NWGC / NHWGC / NDHWGC (channels-last). Forward GEMM:
   `A = RowMajor` (or ColMajor when groups merged), `B = ColMajor`, `C = RowMajor`
   ([grouped_convolution_utils.hpp:111-115](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp#L111)).
4. **Descriptor chain for Filter3x3** —
   [transform_conv_fwd_to_gemm.hpp:510-574](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/grouped_convolution/utils/transform_conv_fwd_to_gemm.hpp#L510):
   - `make_pad_transform(Wi, left_pad, right_pad)` → padded spatial,
   - `make_embed_transform(tuple<3, Wo>, tuple<dilation, stride>)` → unfolded
     filter,
   - `make_merge_transform(tuple<N, Wo>)` → `M` for GEMM.
5. **`ConvolutionSpecialization` enum** —
   [convolution_specialization.hpp:10-16](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/grouped_convolution/utils/convolution_specialization.hpp#L10):
   `Default / Filter1x1Pad0 / Filter1x1Stride1Pad0 / Filter3x3`. `1×1` stride-1
   pad-0 skips both pad and embed (line 482-492).
6. **`make_xor_transform` on group index** —
   [transform_conv_fwd_to_gemm.hpp:1368-1372](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/grouped_convolution/utils/transform_conv_fwd_to_gemm.hpp#L1368)
   — when `NumGroupsToMerge > 1`, remaps the merged group dimension to block
   diagonals so an `M×K_merged` GEMM doesn't write across groups.
7. **2 GB split-N threshold** —
   [transform_conv_fwd_to_gemm.hpp:37, :122-259](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/grouped_convolution/utils/transform_conv_fwd_to_gemm.hpp#L37)
   — hierarchical split (D→H→W) into ≤64 pieces; each sub-kernel launches with
   reduced output spatial dims.
8. **Group-merge for tiny groups** —
   [:1353-1379](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/grouped_convolution/utils/transform_conv_fwd_to_gemm.hpp#L1353)
   — fuses `NumGroupsToMerge` convolutions into one wider GEMM via
   `make_pad_transform(1, 0, NumGroupsToMerge-1)` then xor-diagonalize.
9. **Pipeline glue** —
   [grouped_conv_universal_pipeline_ag_bg_cr_policy.hpp:14](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/grouped_convolution/pipeline/grouped_conv_universal_pipeline_ag_bg_cr_policy.hpp#L14)
   — same `UniversalGemmBasePolicy` family as §3, just with conv-derived
   descriptors.
10. **Asymmetric padding kept separate** — left and right pads stored as
    independent fields so 3×3 stride-2 conv with `left=1, right=0` is exact.
11. **Stride + dilation in one embed**: `make_embed_transform(tuple<X>,
    tuple<dilation, stride>)` does both at once; no separate stride compaction
    pass.

### 13.2 Backward-data

1. **Kernel** —
   [grouped_convolution_backward_data_kernel.hpp:24-80](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_data_kernel.hpp#L24)
   — `dy` plays A, weight is transposed B, `dx` is C.
2. **Reduce mapping**: `M = C_in`, `K = N · spatial_out · merged_groups`,
   `N = filter_volume`.
3. **Split-N still enabled** — `TransformConvBwdDataToGemm<…, true>` mirrors
   forward to avoid OOM on the gradient tensor.
4. **Descriptor builder** —
   `MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N()` ([transform_conv_bwd_data_to_gemm.hpp:824+](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/grouped_convolution/utils/transform_conv_bwd_data_to_gemm.hpp#L824))
   — merges dy spatial + groups into the K dimension.
5. **Padded regions back-propagate zero**: padding is preserved on the descriptor
   so gradient never leaks outside the original spatial region.
6. **Auto split-K** —
   [split_k_utils.hpp:29-79](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/grouped_convolution/utils/split_k_utils.hpp#L29)
   — `GetKBatch()` picks `k_batch` from occupancy + grid size; multiple blocks
   accumulate into dx via atomic merge.
7. **Per-group accumulation** — when groups merged, xor-remap keeps each
   group's gradient on its own block diagonal before the merge-back.

### 13.3 Backward-weight

1. **Kernel** —
   [grouped_convolution_backward_weight_kernel.hpp:48-100](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_weight_kernel.hpp#L48)
   — A = dy (ColMajor), B = input (RowMajor), C = dW (RowMajor).
2. **No split-N** — `TransformConvBwdWeightToGemm<…, false>` (the second
   template argument flips off split-N because dW is `Kout × filter_vol` and
   rarely > 2 GB).
3. **Descriptor** —
   `MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N()` ([transform_conv_bwd_weight_to_gemm.hpp:940+](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/grouped_convolution/utils/transform_conv_bwd_weight_to_gemm.hpp#L940))
   — merges `(N, spatial, C_in)` into K_actual.
4. **Stream-K partitioning** —
   [backward_weight_kernel.hpp:1113+](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_weight_kernel.hpp#L1113)
   — paired with `streamk_gemm_coherency.hpp` so K-split blocks don't deadlock
   on the dW atomic.
5. **Atomic dW accumulation** for split-K > 1.
6. **Padded inputs contribute zero** — same descriptor trick as bwd-data;
   masks live in stride encoding, not in the kernel body.

### 13.4 `transform_conv_*_to_gemm` utilities

1. **Three transformer templates** share a common base
   ([transform_conv_fwd_to_gemm.hpp:266-299](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/grouped_convolution/utils/transform_conv_fwd_to_gemm.hpp#L266))
   parameterized by `(NDimSpatial, ConvSpecialization, VectorSizes,
   NumGroupsToMerge, IndexType)`.
2. **Member state** — G, N, spatial in/out, filter, K, C, strides, dilations,
   left/right pads. Conversion ctor lets a derived transformer mutate one
   field (e.g. for split-image).
3. **Descriptor composition recipe**: `input_desc → pad → embed → merge`. Each
   step is a `transform_tensor_descriptor` call that the compiler folds.
4. **Vector alignment via `number<VectorSize>{}`** passed to
   `make_naive_tensor_descriptor` — guides coalesced 128/256B loads.
5. **Specialization branches inside descriptor methods** — `if constexpr` over
   `ConvSpecialization` so the binary size grows linearly, not multiplicatively.

### 13.5 image_to_column

1. **Standalone im2col** —
   [image_to_column_kernel.hpp:11-200](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/image_to_column/kernel/image_to_column_kernel.hpp#L11)
   — same pad+embed+merge pipeline as conv, but writes the lowered matrix to
   DRAM for an external GEMM.
2. **Currently 2-D only** —
   [:30, :98](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/image_to_column/kernel/image_to_column_kernel.hpp#L30)
   — 1-D / 3-D would need new specializations.
3. **`MakeImageMKDesc`** — chain: `n_hi_wi_c → pad → embed(Y,Ho,X,Wo) →
   merge(n,Ho,Wo) × merge(Y,X,C)`.
4. **`ConvTensorRearrange` per-block kernel** —
   [:174-195](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/image_to_column/kernel/image_to_column_kernel.hpp#L174)
   — loads image tile via descriptor, stores K-major.
5. **`GridSize(GemmM, GemmK, Batch)`** —
   [:88-92](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/image_to_column/kernel/image_to_column_kernel.hpp#L88).

### 13.6 Pooling

1. **`PoolKernel<Problem, Policy>`** —
   [pool_kernel.hpp:78-100](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/pooling/kernel/pool_kernel.hpp#L78).
2. **`PoolProblem` traits** —
   [pool_problem.hpp:18-33](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/pooling/pool_problem.hpp#L18)
   — `kOutputIndex` (argmax), `kPropagateNan`, `kNeedCrossLaneSync`,
   `kNeedCrossWarpSync`.
3. **`ReduceOp` template param** — `MaxOp` / `AvgOp` etc.; provides
   `GetIdentityValue<T>()` (max ⇒ -inf, avg ⇒ 0).
4. **Window parameters** — `window_lengths`, `window_strides`,
   `window_dilations`, `input_left_pads`, `input_right_pads`. Dilations
   space kernel taps; strides set output sampling pitch.
5. **Reuses §6 reduction infra** —
   [pool_default_policy.hpp:31-47](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/pooling/pool_default_policy.hpp#L31)
   `BlockReduce2d` / `…Sync` / `…CrossWarpSync`.
6. **`MakeXBlockTileDistribution`** —
   [:15-27](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/pooling/pool_default_policy.hpp#L15)
   — Repeat / WarpPerBlock / ThreadPerWarp / ThreadTile factoring.
7. **Argmax index buffer** —
   [pool_kernel.hpp:85-99](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/pooling/kernel/pool_kernel.hpp#L85)
   `GetIndicesSmemSize()`; index tile flows through `BlockReduce2d::
   MakeYIndexBlockTile<X_tile, IndexDataType>()`.
8. **Padded reads return zero (avg) or `-inf` (max)**; no explicit mask kernel.
9. **Forward loop** —
   [pool_kernel.hpp:404-476](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/pooling/kernel/pool_kernel.hpp#L404)
   — load window, initialize y_tile to identity, reduce, sync, store
   (with index when argmax).

### 13.7 Cross-cutting techniques (this slice)

- **T13.A — Conv = descriptor algebra over GEMM**: `pad + embed + merge`
  transforms compose the conv view into an `M×K` matrix; the rest of the
  pipeline is §3 unmodified.
- **T13.B — Specialization enum compresses code paths**: `Filter1x1Stride1Pad0`
  drops the pad/embed stages entirely; `if constexpr` ensures one binary
  per problem class.
- **T13.C — Group-merge via xor-diagonalize + merge**: tiny-group conv (G=8,
  16) gets folded into one wide GEMM at no extra ALU cost.
- **T13.D — 2 GB split-N heuristic**: hierarchical D→H→W split keeps any
  single sub-kernel's tensors under the V# range; uniform across fwd and
  bwd-data.
- **T13.E — Stream-K for bwd-weight**: K-axis split with coherency-aware atomic
  merge; required because dW is small but K (= batch × spatial) is huge.
- **T13.F — Padding is in the descriptor, not the loop**: pad/embed encode
  out-of-bounds zeros structurally, so kernels are mask-free.
- **T13.G — Pooling reuses the reduction stack**: no separate kernel family;
  the pad/embed chain feeds the same `BlockReduce2d` used by softmax/norm/topk.
- **T13.H — `make_embed_transform` fuses stride + dilation**: a single
  primitive linearizes the 2-D receptive field into a 1-D K-dim with both
  strides and dilations baked in.

Cross-references: [asm-v2:§16 / §17] (gemm + conv mapping); §3 (warp-GEMM
core that conv reuses); §6.1 (`BlockReduce2d` that pooling reuses).

---

## §14. Backward pipelines (beyond FMHA bwd)

§4.6 walked the canonical `kr_ktr_vr` FMHA backward. This section covers the
other backward families and, importantly, **calls out what does not exist** so
downstream consumers don't assume training-grade support where there is none.

### 14.1 FMHA backward — full family

Selector picks one of **four** pipeline variants at compile time
([block_fmha_bwd_dq_dk_dv_pipeline_selector.hpp:14-43](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_selector.hpp#L14)):

1. **`KRKTRVR`** (the §4.6 variant) —
   [block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp:95-500](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp#L95)
   — `kUseTrLoad=false` + training mode + no padding.
2. **`KRKTRVR_IGLP`** —
   [block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr_iglp.hpp:1-80](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr_iglp.hpp#L1)
   — same algorithm, IGLP-tuned scheduler; selected when **at least one head-dim
   is padded** (`kPadHeadDimQ == 1 || kPadHeadDimV == 1`).
3. **`TRLOAD_KRKTRVR`** —
   [block_fmha_bwd_dq_dk_dv_pipeline_trload_kr_ktr_vr.hpp:1-80](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_trload_kr_ktr_vr.hpp#L1)
   — `kUseTrLoad=true` + training mode; uses `ds_read_tr*` to fold the
   transpose into the LDS read.
4. **`TRLOAD_QRQTRDOR`** —
   [block_fmha_bwd_dq_dk_dv_pipeline_trload_qr_qtr_dor.hpp:211-340](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_trload_qr_qtr_dor.hpp#L211)
   — **decode/persistent mode**: flips the outer loop to N0 (K-sequence) so Q
   is loaded once per K-block; Q/∂O/LSE/D parked in LDS for replay.

**The five GEMMs per Q-tile** (default policy
[block_fmha_bwd_pipeline_default_policy.hpp:33-100](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_default_policy.hpp#L33)):

- `GEMM-0`: `Q · Kᵀ → S` (recompute attention logits)
- `GEMM-1`: `Sᵀ · ∂O → ∂V` (V-grad accumulates over Q-loop)
- `GEMM-2`: `∂O · Vᵀ → ∂S_raw`
- `GEMM-3`: `(∂S_raw - D) ⊙ P · Qᵀ → ∂K` (with `P = softmax(S)`, `D = Σ dO ⊙ O`)
- `GEMM-4`: `∂S_scaled · Kᵀ → ∂Q`

**Auxiliary kernels**:

- **`block_fmha_bwd_dot_do_o`** —
  [:11-162](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dot_do_o.hpp#L11)
  — computes `D[q] = Σ_j O[q,j] * dO[q,j]` *before* the main bwd kernel, plus
  an optional sink-token gradient path that uses `atomicAdd` when
  `atomic_sink_grad_ptr != nullptr` ([:106-159](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dot_do_o.hpp#L106)).
- **`block_fmha_bwd_convert_dq`** —
  [:11-139](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_convert_dq.hpp#L11)
  — two paths: (a) plain cast `AccDataType → QGradDataType` store
  ([:36-61](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_convert_dq.hpp#L36)),
  (b) deterministic split-K *reduce-then-cast* ([:64-138](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_convert_dq.hpp#L64))
  that merges `nsplits - 1` partial dQ accumulators in register before the
  store. Path (b) is selected when `kIsDeterministic && !kUseQrQtrDorPipeline`.
- **`fmha_bwd_kernel.hpp:30-786`** — top-level dispatcher; persistent-mode
  branch ([:744-782](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp#L744))
  is **mutually exclusive** with the `QrQtrDor` pipeline.

### 14.2 RMSNorm2d / LayerNorm2d backward — **does not exist in this submodule**

Confirmed by directory listing:

- `rmsnorm2d/` contains only `rmsnorm2d_fwd_kernel.hpp`, `rmsnorm2d_fwd_pipeline_*.hpp`, `rmsnorm2d_fwd_traits.hpp`. **No `*bwd*` files.**
- `layernorm2d/` contains only the corresponding forward set. **No `*bwd*` files.**

The `kSaveInvRms` flag mentioned in §6.3 saves `1/rms` *so a downstream backward
kernel doesn't have to recompute it* — but the backward kernel itself lives
outside CK_tile (typically materialized by the framework's autograd engine over
the forward primitives, or hand-written upstream in aiter / PyTorch).

### 14.3 `add_rmsnorm2d_rdquant` backward — **does not exist**

`add_rmsnorm2d_rdquant/` contains only the forward three-pass pipeline. The
fused add + RMSNorm + per-token quant is forward-only because:

- The `add` backward is pass-through (`∂a = ∂b = ∂(a+b)`).
- RMSNorm backward would need the absent §14.2 pipeline.
- Per-token quant backward is typically STE / no-op in production training paths.

### 14.4 Convolution backward — inner pipeline

§13 covered the *forward* descriptor reduction; here are the inner-pipeline
specifics that differ in bwd:

1. **Bwd-data role swap** —
   [transform_conv_bwd_data_to_gemm.hpp:20-120](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/grouped_convolution/utils/transform_conv_bwd_data_to_gemm.hpp#L20)
   — `A = ∂Y` (M×K), `B = W` (N×K), `C = ∂X` (M×N); `K = C_out`,
   `N = ∂C_in`, `M = ∂spatial`.
2. **Bwd-data split-N** —
   [:46-114](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/grouped_convolution/utils/transform_conv_bwd_data_to_gemm.hpp#L46)
   — same 2 GB heuristic as forward.
3. **Bwd-data kernel** —
   [grouped_convolution_backward_data_kernel.hpp:24-275](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_data_kernel.hpp#L24)
   — multiple grouped GEMMs (one per input-feature-group × spatial-slice
   combo); per-GEMM grid descriptors (A_m_k, B_n_k, C_m_n).
4. **Bwd-weight role swap** —
   [transform_conv_bwd_weight_to_gemm.hpp:20-100](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/grouped_convolution/utils/transform_conv_bwd_weight_to_gemm.hpp#L20)
   — `A = X` (M×K), `B = ∂Yᵀ` (K×N), `C = ∂W` (M×N); `M = ∂C_in`,
   `K = spatial`, `N = ∂C_out`.
5. **StreamK enabled for bwd-weight** —
   [grouped_convolution_backward_weight_kernel.hpp:17-45](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_weight_kernel.hpp#L17)
   — `is_streamk_partitioner` triggers finer K-axis split than grid
   partitioning gives.
6. **GEMM-A and GEMM-B are *transposed* relative to forward**: forward is
   `GEMM(im2col(X), Wᵀ)`; bwd-data is `GEMM(∂Y, W)`; bwd-weight is
   `GEMM(X, ∂Yᵀ)`. The transpose lives in the descriptor, not in the kernel.

### 14.5 Welford / norm-stat backward — **does not exist**

`thread_welford.hpp` is forward-only. `block_norm_reduce.hpp` and
`block_norm_reduce_problem.hpp` are likewise forward-only. There is no Welford
backward primitive in CK_tile — gradient flow through variance is handled by
whichever norm-bwd kernel a downstream framework writes (none in this
submodule).

### 14.6 Cross-cutting techniques (this slice)

- **T14.A — FMHA bwd is the *only* hand-optimized backward in CK_tile**: every
  other "backward" in the source tree is forward-only with autograd / framework
  glue expected on top.
- **T14.B — Four-way FMHA bwd selector** = `{kUseTrLoad} × {decode | training |
  has_pad}` resolves at compile time so the binary contains exactly one
  pipeline per problem class.
- **T14.C — `dot_do_o` precompute kernel**: `D = Σ dO ⊙ O` is a *separate*
  kernel that runs before the main bwd; lets the main bwd treat `D` as an
  input-tile, not a reduction.
- **T14.D — `convert_dq` deterministic mode**: reduce-then-cast path merges
  `nsplits - 1` partial dQ accumulators *in register* before the DRAM store —
  no atomic, no race-induced nondeterminism.
- **T14.E — IGLP variant gated on padding**: the non-IGLP variant assumes
  fully aligned head dims; the IGLP variant takes over when at least one head
  dim is padded.
- **T14.F — TRLOAD variants fold transpose into the LDS read**: `ds_read_tr*`
  + `kUseTrLoad=true` means `Kᵀ`, `Qᵀ`, `∂Oᵀ` shuffles never go through ALU.
- **T14.G — Persistent FMHA bwd is mutually exclusive with `QrQtrDor`**:
  selector enforces this — both variants would compete for the same
  per-CU worker discipline.
- **T14.H — Convolution backward is a *role swap*, not a new pipeline**: the
  same descriptor algebra (§13) feeds the same warp-GEMM core; only which
  argument is which (A/B/C) changes.

Cross-references: [asm-v2:§23] (forward FMHA), §4.6 (canonical FMHA bwd
walkthrough), §6.3 (Welford forward + the missing bwd story), §13 (conv
forward / shared transform machinery).

---

## §15. Host launcher + StreamK + sparse_attn + topk_softmax + standalone permute/transpose + tensor adaptors

The catch-all slice. These pieces are individually small but each fills a real
gap: the *host* layer the playbook had ignored, the StreamK kernel family, the
sparse-attention and topk-softmax ops, the *standalone* permute and transpose
kernels (separate from the §9 epilogue variants), and the core tensor
abstractions that underlie §1.

### 15.1 Host launcher (`include/ck_tile/host/`)

1. **`make_kernel<MinBlockPerCu, Attr>`** —
   [kernel_launch.hpp:114-133](aiter-amd/3rdparty/composable_kernel/include/ck_tile/host/kernel_launch.hpp#L114)
   — factory wrapping a kernel entry; `MinBlockPerCu` is the occupancy hint
   the compiler passes to the back-end.
2. **Attribute symbol mangling** —
   [kernel_attr.hpp:32-64](aiter-amd/3rdparty/composable_kernel/include/ck_tile/host/kernel_attr.hpp#L32)
   — `kernel_attr_for<ArchTag, Attrs...>` packs e.g. "no packed-fp32-ops"
   into the symbol name; lets one object file ship multiple gfx variants.
3. **`launch_kernel(stream_config, Callables...)`** —
   [kernel_launch.hpp:266-286](aiter-amd/3rdparty/composable_kernel/include/ck_tile/host/kernel_launch.hpp#L266)
   — variadic dispatcher; routes to `gpu_timer{}` or `cpu_timer{}` based on
   `is_gpu_timer_`.
4. **Optional cache-flush timing path** —
   [:314](aiter-amd/3rdparty/composable_kernel/include/ck_tile/host/kernel_launch.hpp#L314)
   `launch_kernel_time_mask_flush_cache` — isolates measurements from the
   previous run's L2 residency.
5. **`stream_config`** —
   [stream_config.hpp:29-39](aiter-amd/3rdparty/composable_kernel/include/ck_tile/host/stream_config.hpp#L29)
   — fields: `stream_id_`, `time_kernel_`, `cold_niters_=3`, `nrepeat_=10`,
   `is_gpu_timer_=true`, `rotating_count_` for rotating-buffer perf tests.
6. **`device_prop` utilities** —
   [device_prop.hpp:19-68](aiter-amd/3rdparty/composable_kernel/include/ck_tile/host/device_prop.hpp#L19)
   — FNV1a compile-time hash for device detection; `get_num_cus()` returns
   `multiProcessorCount` for CU-aware launch sizing.
7. **`DeviceMem` RAII** —
   [device_memory.hpp:50-193](aiter-amd/3rdparty/composable_kernel/include/ck_tile/host/device_memory.hpp#L50)
   — `hipMalloc / hipFree` lifecycle; `ToDevice / FromDevice` bulk transfers;
   `SetValue<T>(x)` kernel-driven init.

### 15.2 StreamK GEMM (`ops/gemm/kernel/streamk_gemm/`)

1. **`StreamKKernel<TilePartitioner, GemmPipeline, EpiloguePipeline>`** —
   [streamk_gemm_kernel.hpp:62-70](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/kernel/streamk_gemm/streamk_gemm_kernel.hpp#L62)
   — inherits `UniversalGemmKernel`; two execution paths share the same warp
   GEMM (§3).
2. **Two-tier work split**:
   - **Data-parallel tiles** (early CTAs) own a full M×N output tile; no
     sync.
   - **Stream-K tiles** (later CTAs) split K-iterations and accumulate into
     a shared `workspace_ptr` ([streamk_gemm_tile_partitioner.hpp:147](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/kernel/streamk_gemm/streamk_gemm_tile_partitioner.hpp#L147))
     before a single CTA runs the epilogue.
3. **Reduction strategy enum** —
   [streamk_gemm_tile_partitioner.hpp:29](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/kernel/streamk_gemm/streamk_gemm_tile_partitioner.hpp#L29)
   — `atomic_add` (high contention but simple) vs `set` (per-CTA slots + tile-
   completion flags).
4. **K-iter assignment** —
   `get_start_iter()` / `get_iter_boundaries()` ([:61, :76](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/kernel/streamk_gemm/streamk_gemm_tile_partitioner.hpp#L61))
   — CTAs balanced by cumulative K-iters, not static tile count.
5. **`StreamKCoherency<CompilerTarget>`** —
   [streamk_gemm_coherency.hpp:8-35](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/gemm/kernel/streamk_gemm/streamk_gemm_coherency.hpp#L8)
   — per-arch memory flags:
   - gfx942 / gfx950 → `SYSTEM_NT0`
   - gfx908 / gfx90a → `glc_slc`
   - others → `coherence_default`.

### 15.3 `sparse_attn` (block-sparse FMHA)

1. **`FmhaFwdVSAKernel`** —
   [fmha_fwd_vsa_kernel.hpp:24-70](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/sparse_attn/kernel/fmha_fwd_vsa_kernel.hpp#L24)
   — Vector Sparse Attention forward; sibling kernel `fmha_fwd_jenga_kernel.hpp`
   has the same interface, different scheduling style.
2. **Common kargs** —
   [:81-110](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/sparse_attn/kernel/fmha_fwd_vsa_kernel.hpp#L81)
   — Q/K/V pointers, `nhead_ratio_qk` (MQA/GQA), `scale_s`, strides.
3. **Mask kargs** —
   [:112-116](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/sparse_attn/kernel/fmha_fwd_vsa_kernel.hpp#L112)
   `window_size_left`, `window_size_right`, `GenericAttentionMaskEnum mask_type`
   (causal / local / etc.).
4. **Iterator inversion under causal masks** —
   [:218-242](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/sparse_attn/kernel/fmha_fwd_vsa_kernel.hpp#L218)
   — when `kHasMask`, iterate `gridDim.x - 1 - i_tile_m` so the loop visits
   only valid blocks.
5. **No bias / no LSE / no dropout / no logits softcap** —
   [:57-64](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/sparse_attn/kernel/fmha_fwd_vsa_kernel.hpp#L57)
   — deliberate simplifications vs §4's canonical FMHA, in exchange for the
   sparse-block iteration.
6. **`valid_block_num_ptr` LUT** —
   [:86](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/sparse_attn/kernel/fmha_fwd_vsa_kernel.hpp#L86)
   — precomputed per-tile non-zero block map; the kernel loops over the LUT
   rather than the dense `seqlen²` grid.
7. **Pipeline** —
   `block_fmha_pipeline_qr_ks_vs_async_vsa.hpp` — reuses §4.1's `qr_ks_vs`
   structure but accepts a *sparse iterator* over K-blocks.

### 15.4 `topk_softmax` (combined op, e.g. MoE routing)

1. **`TopkSoftmaxKernel<Pipeline>`** —
   [topk_softmax_kernel.hpp:28-167](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/topk_softmax/kernel/topk_softmax_kernel.hpp#L28)
   — fused topk + softmax-normalize; output = topk weights + indices per
   row.
2. **Host args** —
   [:15-25](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/topk_softmax/kernel/topk_softmax_kernel.hpp#L15)
   `num_rows`, `num_experts`, `topk`, `stride_input ≥ num_experts`,
   `stride_output ≥ topk`.
3. **Persistent-grid mode** —
   [:54-74](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/topk_softmax/kernel/topk_softmax_kernel.hpp#L54)
   — if `LaunchType > 0`, grid = `num_cu * LaunchType`; else
   `ceil(num_rows / RowsPerBlock)` blocks. Persistent variant amortizes the
   MoE tail.
4. **Padded windows** —
   [:103-157](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/topk_softmax/kernel/topk_softmax_kernel.hpp#L103)
   — input `(RowsPerBlock, num_experts)` and output `(RowsPerBlock, 1)`
   wrapped with `pad_tensor_view()` so the warp-per-row pipeline never
   touches OOB.
5. **Warp-per-row pipeline** —
   `topk_softmax_warp_per_row_pipeline.hpp` — each warp handles one row,
   fuses argmax-extraction (§6.4 style) with softmax normalization
   (§6.2 style) without writing intermediates to LDS.

### 15.5 Standalone `permute/` and `batched_transpose/`

1. **`GenericPermute<Problem>`** —
   [generic_permute_kernel.hpp:38-170](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/permute/kernel/generic_permute_kernel.hpp#L38)
   — up to 8 ranks; `perm_length[8]` + `perm_stride[8]` describe the
   permutation as strides over the original layout.
2. **Coordinate-transform implementation** —
   [:120-166](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/permute/kernel/generic_permute_kernel.hpp#L120)
   — linearize → `transform_tensor_view` chain of merge transforms → permuted
   coordinate; one element per thread (no tiling).
3. **`BatchedTransposeKernel<Pipeline>`** —
   [batched_transpose_kernel.hpp:28-129](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/batched_transpose/kernel/batched_transpose_kernel.hpp#L28)
   — 2-D transpose over a batch dim; grid `(ceil(H/MPerBlock),
   ceil(W/NPerBlock), batch)`.
4. **Padded views both sides** —
   [:91-126](aiter-amd/3rdparty/composable_kernel/include/ck_tile/ops/batched_transpose/kernel/batched_transpose_kernel.hpp#L91)
   — input view `(H, W)` stride `(W, 1)`, output view `(W, H)` stride
   `(H, 1)`; tile windows at offsets `(iM, iN)` and `(iN, iM)`.
5. **Two pipeline flavors** under `ops/batched_transpose/pipeline/` — LDS-based
   (lower register pressure) vs register-only (lower latency).
6. **Difference from §9.5** — §9.5's `PermuteN` is an epilogue *baked into a
   GEMM*; these are *standalone* kernels you'd dispatch separately.

### 15.6 Core tensor abstractions (the missing §1 foundation)

These power every `tile_window`, every load, every coordinate computation.

1. **`tensor_adaptor`** —
   [tensor_adaptor.hpp:30-250](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tensor_adaptor.hpp#L30)
   — chain of `Transforms` (merge / unmerge / pad / xor / pass-through);
   maps top-level (user) indices to bottom-level (memory) indices. The thing
   that makes "logical shape" decoupled from "memory layout."
2. **`tensor_view<BufferView, TensorDesc, DstInMemOp>`** —
   [tensor_view.hpp:40-612](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tensor_view.hpp#L40)
   — stateless accessor: `get_vectorized_elements<X>(coord, linear_off)`,
   `set_vectorized_elements()`, `async_get()`. `DstInMemOp` enum is the
   `{set, add}` toggle that §9.3 plumbs through.
3. **`buffer_view<AddressSpace, T, SizeType, InvalidUseZero, Coherence>`** —
   [buffer_view.hpp:30-55](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/buffer_view.hpp#L30)
   — owns the V# / address-space tag and an `invalid_element_value_` so OOB
   loads can return a numeric zero without branching. `Coherence` is what
   `StreamKCoherency` (§15.2) ultimately writes into.
4. **`tensor_coordinate`** —
   [tensor_coordinate.hpp:20-57](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tensor_coordinate.hpp#L20)
   — compact coordinate cache; `adaptor_coordinate_is_valid()` for boundary
   checks.
5. **`tensor_adaptor_coordinate`** —
   [tensor_adaptor_coordinate.hpp:22-199](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tensor_adaptor_coordinate.hpp#L22)
   — keeps the *hidden* index; `move_tensor_adaptor_coordinate()` does the
   incremental update used by `move_tile_window()` in §1.2.
6. **`pad_tensor_view()`** —
   [tensor_view.hpp:567-608](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/tensor/tensor_view.hpp#L567)
   — wraps `make_right_pad_transform(old_length, pad_length)` per dim;
   skips dims with `make_pass_through_transform()`. Every tile-window-with-
   ragged-shape relies on this.
7. **`merge_v2_magic_division`** —
   [coordinate_transform.hpp:562-688](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/algorithm/coordinate_transform.hpp#L562)
   — ND→1D linearization via magic-division (precomputed reciprocal); avoids
   integer-divide in the hot path.
8. **`merge_v3_division_mod`** —
   [coordinate_transform.hpp:705-828](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/algorithm/coordinate_transform.hpp#L705)
   — fallback merge using div/mod (compile-time dims).
9. **`unmerge<UpLengths, Use24BitIntegerCalculation>`** —
   [coordinate_transform.hpp:830-935](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/algorithm/coordinate_transform.hpp#L830)
   — 1D→ND via reverse-scan cumulative strides; the 24-bit option packs
   indices into a register for sub-dword address math.

### 15.7 Cross-cutting techniques (this slice)

- **T15.A — Symbol-mangled per-arch kernels in one binary**: `kernel_attr_for`
  encodes the gfx target into the symbol so a fat library can ship gfx9 +
  gfx11 + gfx12 implementations and the loader picks at runtime — no
  preprocessor split across .so files.
- **T15.B — StreamK two-tier work split**: data-parallel CTAs and stream-K
  CTAs use the *same warp GEMM core* — load balancing is purely a partitioner
  trick.
- **T15.C — `StreamKCoherency` is a per-arch memory-flag table**: same C++
  code, different cache-hint bits per gfx; `SYSTEM_NT0` is what makes the
  workspace reduction race-free on gfx94+.
- **T15.D — Block-sparse FMHA via LUT iteration**: `valid_block_num_ptr`
  encodes the sparsity pattern; the kernel does not check masks per element,
  just iterates only valid blocks.
- **T15.E — Persistent-grid for variable-cost ops**: topk_softmax + StreamK
  pick `LaunchType > 0` for tail amortization; the grid size becomes
  `num_cu × launchType`, not `num_rows / RowsPerBlock`.
- **T15.F — `pad_tensor_view` is the universal OOB story**: every CK_tile
  kernel that handles ragged shapes uses it; combined with `buffer_view`'s
  `invalid_element_value_`, OOB reads return zero without a single branch.
- **T15.G — `tensor_adaptor` chains the whole "logical → memory" translation**:
  reshape / transpose / pad / xor-bank-permute compose by stacking transforms;
  the same adaptor is fed to load_tile, store_tile, and the FMHA softmax —
  one source of truth for layout.
- **T15.H — Magic-division merge avoids `idiv`**: hot-path coordinate math
  uses `merge_v2_magic_division` so address arithmetic compiles to a
  `mul + shift`, not a full integer divide.
- **T15.I — `kernel_launch` integrates the perf-test loop**: warmup,
  rotating-buffer, cache-flush, gpu-timer all controlled by `stream_config`;
  benchmarking is part of the launcher, not a separate harness.

Cross-references: §1 (the abstractions above sit beneath every tile_window /
load_tile call); §3 (warp-GEMM core that StreamK reuses); §4 (canonical FMHA
that sparse_attn specializes); §9 (the *epilogue* permute / transpose, which
these standalone kernels complement, not duplicate).

---

## §16. Perf-engineering toolkit + MI300 / MI355 specifics

Slices §1–§15 covered *what* CK does. This section covers the perf-engineering
*levers* — `__launch_bounds__`, the full cache-coherence enum, L2 swizzle math,
magic-division precompute, ILP, prefetch-stages occupancy math — plus everything
specific to **MI300 (gfx940/941/942)** and **MI355 (gfx950 / CDNA4)** beyond
what §2 already documents.

Several findings here are honest **absences**. Reading these matters because
the playbook is *also* a guide to what CK does *not* yet exploit on the latest
hardware.

### 16.1 `__launch_bounds__` discipline

1. **Global defaults** —
   [ck.hpp:30-31](aiter-amd/3rdparty/composable_kernel/include/ck/ck.hpp#L30):
   `CK_MAX_THREAD_PER_BLOCK = 256`, `CK_MIN_BLOCK_PER_CU = 2`. Floor of two
   blocks/CU on every classic kernel — registers + LDS must fit twice.
2. **Per-kernel override pattern** —
   [device_batched_gemm_gemm_xdl_cshuffle.hpp:42](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_gemm_xdl_cshuffle.hpp#L42)
   — `__launch_bounds__(GridwiseGemm::MaxBlockSize, MinBlockPerCu)`; the
   per-spec block size flows through to the back-end VGPR allocator.
3. **WMMA dual-dispatch** —
   [device_batched_gemm_multiple_d_wmma_cshuffle_v3.hpp:32](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_wmma_cshuffle_v3.hpp#L32)
   — WMMA-side uses `__launch_bounds__(CK_MAX_THREAD_PER_BLOCK,
   MinimumOccupancy)` (custom `MinimumOccupancy`) when the tile is wide and
   per-wave VGPR pressure is high.
4. **Codegen-emitted bounds** —
   [codegen_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp:212](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/device/impl/codegen_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp#L212)
   — codegen always hard-wires `__launch_bounds__(CK_MAX_THREAD_PER_BLOCK,
   CK_MIN_BLOCK_PER_CU)` so the instance archive is occupancy-bounded.
5. **CK_tile uses a different mechanism** — `MinBlockPerCu` template param to
   `make_kernel` (§15.1) rather than the C++ attribute; same effect, different
   knob. The attribute discipline is a classic-CK pattern.

### 16.2 `amd_buffer_coherence_enum` — the full cache-hint table

Cache coherence on CDNA encodes both a **temporal hint** and a **scope**:

1. **Temporal hints** —
   [amd_buffer_coherence.hpp:18-26](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/amd_buffer_coherence.hpp#L18):
   `RT` (regular temporal, 0), `NT` (non-temporal, 1), `HT` (high-priority
   temporal, 2), `WB` (write-back/last-use, 3); plus four crossover hints
   `NT_RT/RT_NT/NT_HT/NT_WB` for near-vs-far cache asymmetry.
2. **Scope prefixes** —
   [amd_buffer_coherence.hpp:29-67](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/amd_buffer_coherence.hpp#L29):
   `CU_` (compute-unit local), `SE_` (shader engine), `DEVICE`, `SYSTEM` —
   giving the full 28-value matrix `temporal × scope`.
3. **gfx942 / gfx950 scope collapse** —
   [amd_buffer_coherence.hpp:84-105](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/amd_buffer_coherence.hpp#L84)
   — MI300 family **drops `SE` scope** entirely; only `WAVE, GROUP, DEVICE,
   SYSTEM` exist. The flat table compresses to 16 values
   (`WAVE_NT0`, …, `SYSTEM_NT0`, etc.).
4. **gfx908 / gfx90a compatibility aliases** —
   `GLC = DEVICE_NT1`, `SLC = SYSTEM_NT1` — keep older atomics compiling on
   newer arches.
5. **Where the values are *used*** — StreamK workspace reduction (§15.2) is the
   only consumer that picks `SYSTEM_NT0` on gfx942/950 vs `glc_slc` on
   gfx908/90a; most other kernels pass `coherence_default = 0`. **The
   non-default values are largely *available but unused* in CK** — a real
   optimization opportunity for downstream consumers.
6. **gfx942+ BF16 global atomic add** —
   [amd_buffer_addressing.hpp:563-572](aiter-amd/3rdparty/composable_kernel/include/ck/utility/amd_buffer_addressing.hpp#L563)
   — `__builtin_amdgcn_global_atomic_fadd_v2bf16` is the one place the
   coherence enum drives a per-arch instruction selection.
7. **Direct-load ignores coherence** —
   [amd_buffer_addressing.hpp:1001-1040](aiter-amd/3rdparty/composable_kernel/include/ck/utility/amd_buffer_addressing.hpp#L1001)
   — `amd_direct_load_global_to_lds` hardcodes the M0-SGPR path; the
   buffer-coherence override is ignored. Consequence: async LDS loads cannot
   pick `NT` to bypass L1 (they always populate L1 *and* LDS).

### 16.3 L2 swizzle / `Block2CTileMap` math

1. **`BlockToCTileMap_M00_N0_M01`** —
   [block_to_ctile_map.hpp:18-113](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp#L18)
   — encodes the 2-D C-tile grid as `(M00, N0, M01)`. A flat `blockIdx.x` is
   expanded then contracted through a tensor-adaptor chain.
2. **`M01` controls L2-partition residency** —
   [:32-48](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp#L32)
   — typical `M01 ≈ 8`: consecutive blocks visit `(M00=0, N0=0, M01=0..7)`
   before advancing `N0`, so 8 blocks pack into one L2 column before
   striding.
3. **Index math is divide-free** —
   [:91-107](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp#L91)
   — two-stage adaptor (Merge then Unmerge); blockId expansion uses only
   add/compare/precomputed-stride. No `idiv`.
4. **Validity check** —
   [:66-80](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp#L66)
   — `M0 % M01 == 0` enforced at host; failure routes to
   `BlockToCTileMap_M00_N0_M01Adapt` which shrinks `M01` on edge blocks.
5. **No explicit L2-partition enumeration** — CK trusts that consecutive
   blocks naturally land in different L2 slices because the output address
   strides through partition-sized regions; there is **no L2-hash math** in
   the source. If a workload mis-aligns, the only knob is `M01`.

### 16.4 Multi-level atomic reduction — **honest absence**

CK does **not** implement warp-aggregated reductions before `atomic_add`.
Patterns observed:

1. **Block-local reduce → single-thread atomic** — entire block reduces to
   one scalar (via `BlockReduce2d`), then thread 0 of warp 0 issues the
   atomic. No `__shfl_xor` + per-warp atomic intermediate.
2. **Split-K workspace + final reduction kernel** — StreamK (§15.2) writes
   partial tiles to a workspace, then a *separate* reduction kernel finishes
   the sum. Effectively two-level, but pipelined via workspace, not
   warp-aggregation.
3. **Vector atomics are unrolled, not bundled** —
   [amd_buffer_addressing.hpp:576-731](aiter-amd/3rdparty/composable_kernel/include/ck/utility/amd_buffer_addressing.hpp#L576)
   `amd_buffer_atomic_add_impl<T, N>` emits *N separate* atomic instructions,
   one per element offset. There is no warp-level aggregation primitive.
4. **Opportunity**: a real multi-level (`__shfl_xor_sync` → block-LDS reduce →
   single atomic) primitive could shrink atomic traffic by `warp_size×`
   factor for split-K bias / residual accumulation. Not present today.

### 16.5 Wave32 vs Wave64

1. **MFMA intrinsics implicitly Wave64** —
   [amd_xdlops.hpp:33-47](aiter-amd/3rdparty/composable_kernel/include/ck/utility/amd_xdlops.hpp#L33)
   — `__builtin_amdgcn_mfma_f32_32x32x1f32` and siblings only exist in
   Wave64 form on gfx9 / CDNA. No Wave32 specialization in CK.
2. **DPP kernels are explicitly Wave32** —
   `warp/dpp_gemm.hpp` — `static constexpr index_t wave_size = 32` everywhere.
   DPP cross-lane is a Wave32-only instruction on RDNA / gfx10+.
3. **No runtime wave-size selection** — `amdgcn_target_wave_size_id`
   ([arch.hpp:127-132](aiter-amd/3rdparty/composable_kernel/include/ck_tile/core/arch/arch.hpp#L127),
   §2.1.3) is read at compile time only. The link-time
   `--amdgpu-waves-per-execution-unit` flag is the only practical knob.
4. **Implication**: on RDNA / MI3xx-Wave32-capable parts, CK MFMA kernels are
   stuck in Wave64 mode by virtue of intrinsic availability. Wave32 paths
   exist only for DPP.

### 16.6 Magic-division host precompute

`merge_v2_magic_division` (§15.6.7) replaces `idiv` with `mul-shift`. The
host-side precompute is the part the playbook hadn't shown:

1. **Struct fields** —
   [multi_index_transform.hpp:1047-1187](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_description/multi_index_transform.hpp#L1047)
   — `LowLengths`, `LowLengthsMagicDivisorMultiplier`, `LowLengthsMagicDivisorShift`.
2. **`CalculateMagicMultiplier(len)`** —
   [:1074-1076](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_description/multi_index_transform.hpp#L1074)
   — Hacker's-Delight reciprocal: computes 32-bit `m` such that
   `(x*m) >> shift ≈ x/len` for `x` in the valid range.
3. **`CalculateMagicShift(len)`** —
   [:1077-1078](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_description/multi_index_transform.hpp#L1077)
   — typically `32 + log2(len)` so the 31-bit dividend has headroom.
4. **Kernel-side unroll** —
   [:1103-1112](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_description/multi_index_transform.hpp#L1103)
   — `static_for<NDim-1, …, -1>` reverses through dims; each iteration is
   `quotient = DoMagicDivision(tmp, mul[i], shift[i]); residue = tmp - q*len[i]`.
   All constants fold; output is straight-line `mulhi + shift + sub`.
5. **Scope** — used for *multi-index flattening* (e.g. `(H, W, C) → linear`).
   Block-to-tile coordinate math uses adaptor chains instead.

### 16.7 Persistent grid-stride loops — **only StreamK uses them**

1. **StreamK grid-stride** —
   [gridwise_gemm_xdl_cshuffle_streamk_v3.hpp:899-924](aiter-amd/3rdparty/composable_kernel/include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_streamk_v3.hpp#L899)
   `for (auto block_idx = get_block_1d_id(); block_idx <
   block_2_ctile_map_streamk.get_grid_dims(); block_idx += gridDim.x)`.
2. **Classic kernels: one block, one tile.** No looping; grid size = tile
   count.
3. **MoE expert iteration** — `gridwise_moe_gemm*` outer-loops over
   `expert_id` along `gridDim.z`, but that's a per-expert sub-grid, not a
   grid-stride over tiles.
4. **Reduce / norm / conv** — no grid-stride loop variants; tail is handled
   by a separate epilogue kernel when needed.
5. **Consequence**: persistent-kernel patterns (work-stealing, software
   queues) live entirely in StreamK; reductions and norms cannot exploit
   them today.

### 16.8 EXEC-mask tricks — **OOB-trick dominates over hand-written predicates**

1. **OOB-trick = exec-mask via negative offset** —
   [amd_buffer_addressing.hpp:821-831](aiter-amd/3rdparty/composable_kernel/include/ck/utility/amd_buffer_addressing.hpp#L821)
   — if `!src_thread_element_valid`, set the top bit of `voffset`
   (`0x80000000`); the V# range check fires and the lane reads zero. No
   `s_cbranch`, no EXEC manipulation.
2. **`flag_to_exec` / `cmp_lt_to_exec`** (§2.5.6) — used for masked
   writes; same idea but lifted from per-lane condition into EXEC.
3. **No `s_cbranch_execz` short-circuit** found in CK source. Branches
   come from C++ `if(...)`; the compiler decides whether to compile them
   as predicated VALU, `s_cbranch_execz`, or normal branches.
4. **Inline-asm `m0` setup** —
   [amd_buffer_addressing.hpp:1022-1026](aiter-amd/3rdparty/composable_kernel/include/ck/utility/amd_buffer_addressing.hpp#L1022)
   — all lanes participate in `global_load_lds`; M0 is wave-uniform, no
   per-lane mask.

### 16.9 Wave-level MFMA ILP

1. **Back-to-back MFMA emission** —
   [amd_xdlops.hpp:33-100](aiter-amd/3rdparty/composable_kernel/include/ck/utility/amd_xdlops.hpp#L33)
   `intrin_mfma_f32_32x32x1f32::Run()` issues two MFMA builtins in a row
   with **no intermediate `s_waitcnt`** — the second reads accumulator
   while the first is still in flight.
2. **MFMA latency ≈ 8 cycles on CDNA2** — back-to-back hides one full
   issue under the next; pipeline depth determines achievable ILP.
3. **No explicit ILP counter in CK** — `PrefetchStages` + `GlobalBufferNum`
   indirectly set unroll factor, but the compiler picks the actual MFMA
   schedule. CK never sets a "number of MFMA in flight" knob.
4. **Register file caps ILP**: with one C-accumulator live across the K-loop,
   only one MFMA per accumulator at a time. The §3 V3+ pipelines that
   unroll K to two register banks effectively double ILP.

### 16.10 Prefetch-stages occupancy implication

The §3 `PrefetchStages` enum has a *concrete* LDS-byte cost, hence an
occupancy cliff:

1. **128×128 fp32 tile, gfx940** — A tile = 128×32 = 16 KB, B tile = 32×128 =
   16 KB; both doubled at `PrefetchStages=2` → **64 KB LDS per block**.
2. **96 KB LDS per CU** (gfx940) → 64 KB = 67 % utilization → 1 block/CU.
   `PrefetchStages=3` (96 KB) → 100 % → **strictly 1 block/CU**, no
   occupancy headroom for hiding DRAM.
3. **gfx950 has 160 KB LDS** — same `PrefetchStages=3` (96 KB) now leaves
   64 KB headroom for a second block (assuming VGPR also fits). CDNA4 widens
   the occupancy budget without changing the pipeline math.
4. **VGPR cost from `GlobalBufferNum`** — `GlobalBufferNum=2` doubles the
   in-flight global-load VGPR set; on 256-thread block, that's hundreds of
   bytes per thread. Pair it with high `PrefetchStages` and the kernel will
   spill.
5. **Heuristic**: pick `PrefetchStages=1` for 3-block occupancy, or
   `PrefetchStages=2` for 1-2 blocks with deep K-unroll providing ILP.
   `PrefetchStages=3` is **only safe on gfx950** (or very small tiles).
6. **CK_tile auto-budgets**: tile_distribution policies compute LDS
   allocation at template-instantiation time and *won't* compile if you
   exceed `get_smem_capacity()` (§2.1.5). Classic CK has no such guard.

### 16.11 MI300 (gfx940 / gfx941 / gfx942) — specifics CK exploits

1. **`__gfx94__` macro** —
   [ck.hpp:9-10](aiter-amd/3rdparty/composable_kernel/include/ck/ck.hpp#L9)
   — gates common MI300 codepaths; *no per-die specialization for the 8 XCDs.*
2. **f8 / bf8 MFMA `mfma_f32_32x32x64_f8f6f4`** —
   [amd_xdlops.hpp:500-682](aiter-amd/3rdparty/composable_kernel/include/ck/utility/amd_xdlops.hpp#L500)
   — fused 32×32 tile, K=64 packed-fp8 / fp6 / fp4 lane. Single 32x32x64
   variant; CK does *not* expose a 16x16x32 fp8 variant.
3. **Scaled MFMA `mfma_scale_f32_*`** —
   [amd_xdlops.hpp:685-770](aiter-amd/3rdparty/composable_kernel/include/ck/utility/amd_xdlops.hpp#L685)
   — `Run(reg_a, scale_a, reg_b, scale_b)`; e8m0 scale in the top 4 bits of
   `int32_t` (OpSel field). Same builtin handles fp8/bf8/fp6/fp4 via the
   `_f8f6f4` suffix.
4. **BF16 global atomic_fadd (gfx942+)** —
   [amd_buffer_addressing.hpp:563-572](aiter-amd/3rdparty/composable_kernel/include/ck/utility/amd_buffer_addressing.hpp#L563)
   — `__builtin_amdgcn_global_atomic_fadd_v2bf16` packs 2 bf16 per atomic.
   Pre-gfx942: fall back to float atomics + truncate.
5. **Coherence scope collapse** (§16.2.3) — gfx942 drops `SE` scope, leaving
   `WAVE / GROUP / DEVICE / SYSTEM`. Affects how fine-grained you can ask
   for cache scope in atomics.
6. **8-XCD topology — uncovered**. CK has **no `cluster_idx` / `XCC_ID` /
   cross-die awareness**. Every kernel sees a flat 304-CU grid; the runtime
   round-robins blocks across XCDs and CK does nothing to keep working sets
   die-local. **This is the single biggest perf opportunity untouched on
   MI300X.**
7. **96 KB LDS retained** — same as gfx90a. Direct-load (`global_load_lds`)
   stays 1 dword/thread on gfx942.

### 16.12 MI355 (gfx950 / CDNA4) — specifics beyond §2

§2 covered: 64 LDS banks, 160 KB SMEM, doubled-K MFMA, `s_wait_loadcnt_dscnt`.
Things §2 missed:

1. **gfx950 inherits gfx942's scaled MFMA** —
   [amd_xdlops.hpp:500-770](aiter-amd/3rdparty/composable_kernel/include/ck/utility/amd_xdlops.hpp#L500)
   — same `_f8f6f4` family, no new MFMA shapes or larger-K variants on top
   of what gfx942 already has.
2. **MX-fp4 / MX-fp6 / MX-fp8 tile load/store — not in CK**. CK supports
   scaled MFMA with e8m0 (§5.3, OpSel byte-select) but **does not have
   first-class MX-format tile primitives**; quantization / packing must
   happen in the epilogue or upstream. This is a *real* gap on CDNA4.
3. **64-bank LDS swizzle — not specialized**. All LDS layouts in CK assume
   32-bank aliasing (§9.2). On gfx950 the existing swizzle is *over-conservative*;
   doubling the bank count means the gfx9-era padding is sometimes
   unnecessary. **No gfx950-specific swizzle in the source.**
4. **160 KB LDS budget — not exploited**. Kernels still budget against the
   conservative 96 KB ceiling (`get_smem_capacity()` returns 160 KB
   correctly — §2.1.5 — but no kernel cranks `PrefetchStages` or tile size
   to fill it). Significant occupancy headroom is left on the table.
5. **`ds_read_tr*` (transpose-on-read) — declared in builtins but not used
   in CK pipelines.** §2.4.9 cites the builtin's existence; grepping the
   pipelines for `ds_read_tr` finds **zero call sites**. Free transpose
   that the playbook lists as a technique is not yet wired in.
6. **Direct-load extended widths (gfx950)** —
   [amd_buffer_addressing.hpp:1001-1007](aiter-amd/3rdparty/composable_kernel/include/ck/utility/amd_buffer_addressing.hpp#L1001)
   — gfx950 accepts 1, 3, or 4 dwords/thread for `global_load_lds`
   (vs gfx942's 1-dword-only). Higher-throughput direct DRAM→LDS.
7. **No new MFMA K-width beyond 32 for f16** — §2.3.2 covered 16x16x32_f16;
   gfx950's instruction set does not add a 16x16x64_f16 in CK's table.

### 16.13 Cross-cutting techniques (this slice)

- **T16.A — `__launch_bounds__` is the VGPR-allocator's contract**: every
  classic-CK kernel passes `(blockSize, MinBlockPerCu)` so the back-end
  knows how aggressively to spill. CK_tile uses `make_kernel<MinBlockPerCu>`
  for the same purpose.
- **T16.B — 28-value coherence enum is largely *available, unused***:
  StreamK is the lone non-default consumer; most kernels could opt into
  `NT` / `DEVICE` scope for inputs they read once.
- **T16.C — L2 swizzle is a parameter (`M01`), not a math primitive**:
  CK trusts implicit address striding rather than computing an explicit hash.
- **T16.D — Multi-level atomics are absent**: split-K still reduces via
  per-element global atomics; warp-aggregation would cut traffic by
  `warp_size×`.
- **T16.E — Wave32 is unavailable for MFMA paths**: only DPP kernels are
  Wave32; Wave64 is the implicit assumption everywhere else.
- **T16.F — Magic-division precompute lives in `multi_index_transform`**:
  the host fills `multiplier[]` and `shift[]`; the kernel sees only
  `mulhi-shift-sub`. The shape must be known at template time.
- **T16.G — Persistent grid-stride is StreamK-only**: every other op family
  uses 1-block-per-tile; opportunity for persistent reductions / norms is
  untouched.
- **T16.H — OOB-trick replaces EXEC manipulation**: setting a negative
  V-offset to make the hardware ignore a lane is the dominant pattern; raw
  EXEC-bit tricks are rare.
- **T16.I — `PrefetchStages` cliff is arch-dependent**: gfx940 saturates at
  `PrefetchStages=2`; gfx950's 160 KB LDS makes `=3` feasible but no
  shipping kernel exploits it.
- **T16.J — MI300X 8-XCD topology is invisible to CK**: every kernel treats
  the chip as a flat 304-CU grid. *Largest single perf opportunity on
  MI300X.*
- **T16.K — gfx950's `ds_read_tr*` is declared but not used**: the free
  transpose listed as a technique in §2 is currently aspirational on
  CDNA4.
- **T16.L — gfx950 LDS-bank padding is over-conservative**: existing
  gfx9-era swizzle assumes 32 banks; CDNA4's 64-bank layout makes some
  of that padding redundant.
- **T16.M — MX-format tile primitives are missing**: CK has scaled MFMA
  but not first-class MX-fp4/fp6/fp8 tile load/store — quantization must
  bracket the kernel.

### 16.14 Top perf opportunities on MI300X / MI355X (not yet in CK)

Concrete things that would help if added to the source (and then to the
playbook):

1. **XCC-aware grid scheduling on MI300X** — pin tile clusters to one XCD so
   working sets stay in that die's L2; reroute split-K reductions to
   intra-XCD atomics. Single biggest MI300X win.
2. **gfx950 64-bank-aware swizzle** — relax the gfx9-era padding in §9.2 so
   shuffle tiles don't waste the extra LDS budget.
3. **Wire `ds_read_tr*` into the gfx950 pipelines** — replace ALU transposes
   with the hardware-free read.
4. **First-class MX-fp4 / MX-fp6 tile primitives** — load/store with e8m0
   scale packing handled by the tile_window, not the kernel body.
5. **Warp-aggregated atomic reduction** — `__shfl_xor_sync` + per-warp
   atomic before global merge; cut split-K traffic by 64×.
6. **`PrefetchStages=3` on gfx950** — exploit the 160 KB LDS for a deeper
   pipeline; current pipelines stop at 2 to fit 96 KB.
7. **Coherence-aware loads for read-once tensors** — pass `DEVICE_NT0` for
   weight loads that won't be reused; reserves L1 for higher-reuse traffic.

Cross-references: §2 (waitcnt / MFMA, the foundation §16 builds on),
§3 (PrefetchStages enum that §16.10 quantifies), §9.2 (LDS bank padding
that §16.12.3-4 critiques), §15.2 (StreamK, the lone persistent-grid
consumer).





