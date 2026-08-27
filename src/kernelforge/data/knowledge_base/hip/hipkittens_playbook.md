# HipKittens Performance Playbook

> Bit-level analysis of optimization techniques in
> [HazyResearch's HipKittens](https://github.com/HazyResearch/HipKittens) —
> the HIP/ROCm port of ThunderKittens. Slice-based incremental synthesis,
> one section per commit.
>
> Upstream pin: `HipKittens @ 5294d1c5` (main branch).
> Reachable via the `HipKittens/` symlink at the repo root.
>
> Companion to [`../aiter-kernel-analysis/CK_PLAYBOOK.md`](../aiter-kernel-analysis/CK_PLAYBOOK.md).
> Citation styles:
> - **`HipKittens/include/...`** / **`HipKittens/kernels/...`** — relative file
>   path under the upstream tree (resolved via this wrapper's symlink).
> - **`[pdf:pN]`** — AMD CDNA4 ISA spec PDF page.
> - **`[ck:§N]`** — cross-reference to a section of the CK playbook.
>
> ## Status
>
> Each slice = one focused deep-read + one playbook section + one commit.
>
> | # | Slice | Status |
> |---|---|---|
> | 1 | Core types (register / shared / global tiles, layouts, allocator) | ✅ §1 |
> | 2 | Warp-level ops (memory / register / shared sub-trees) | ✅ §2 |
> | 3 | Group-level ops (multi-warp coordination) | ✅ §3 |
> | 4 | GEMM kernels (bf16fp32 + fp8fp32 + FP6 experiments) | ✅ §4 |
> | 5 | Attention kernels (gqa fwd / bwd / causal variants) | ✅ §5 |
> | 6 | LayerNorm + Rotary + torch_scaled | ✅ §6 |
> | 7 | Distributed kernels (iris + bf16_gemm) | ✅ §7 |
> | 8 | Training kernels (bert + llama) | ✅ §8 |
> | 9 | MI300 (gfx942) + MI355 (gfx950 / CDNA4) specifics | ✅ §9 |
> | 10 | Perf-engineering catalog + cross-cutting + reading order | ✅ §10 |
> | 11 | Assembly subtree + FP8_8wave + micros + analysis infra + sched_barrier numerics + FP4 stub | ✅ §11 |
>
> *Slice 11 added after a second review surfaced thin coverage of the inline-asm fallback subtree, micro-benchmark infra, and the actual `sched_barrier_pairs` numerics.*

## Table of Contents

- [§1. Core type system](#1-core-type-system-register--shared--global-tiles) — `rt / rv / art / st / sv / gl`
- [§2. Warp-level ops](#2-warp-level-ops) — memory / register / shared sub-trees
- [§3. Group-level ops](#3-group-level-ops) — `group<N_WARPS>` multi-warp coordination
- [§4. GEMM kernels](#4-gemm-kernels--bf16fp32-fp8fp32-fp6-experiments) — bf16fp32, fp8fp32, FP6 dwordx3/dwordx4, MXFP8 agent walk
- [§5. Attention kernels](#5-attention-kernels-gqa-fwd--bwd--causal-variants) — gqa fwd / bwd / causal variants
- [§6. LayerNorm + Rotary + torch_scaled](#6-layernorm--rotary--torch_scaled)
- [§7. Distributed kernels](#7-distributed-kernels-iris--bf16_gemm) — Iris + bf16_gemm
- [§8. Training kernels](#8-training-kernels-bert--llama) — bert + llama
- [§9. MI300 + MI355 specifics](#9-mi300-gfx942--mi355-gfx950--cdna4-specifics)
- [§10. Perf-engineering catalog + HK ↔ CK comparison](#10-perf-engineering-catalog--reading-order--hk--ck-comparison)
- [§11. Assembly subtree + infrastructure + sched_barrier numerics + FP4 stub](#11-assembly-subtree--infrastructure--sched_barrier-numerics--fp4-stub--errata)

---

<!-- SLICE-1-INSERT -->
## §1. Core type system (register / shared / global tiles)

HipKittens is **header-only tile primitives** plus a small set of hand-written
kernels. The whole library hangs off five type families: `rt` (register tile),
`rv` (register vector), `art` (assembly-mode accumulator register tile), `st`
(shared tile), `sv` (shared vector), and `gl` (global layout descriptor). All
shapes are compile-time; all packing factors / layouts / swizzles are template
parameters; nothing is dynamic.

### 1.1 `kittens.cuh` + `common/`

1. **Top-level aggregator** —
   [kittens.cuh:8-11](HipKittens/include/kittens.cuh#L8-L11)
   — pulls `common.cuh`, `types.cuh`, `ops.cuh`. The header file *is* the API.
2. **`WARP_THREADS = 64`** —
   [common/util.cuh:32](HipKittens/include/common/util.cuh#L32)
   — Wave64-only; CDNA-targeted. No Wave32 alternate path.
3. **Hard-coded dtype list** —
   [common/base_types.cuh:27-79](HipKittens/include/common/base_types.cuh#L27-L79)
   — `bf16, half, float, fp8e4m3, fp4e2m1`. No dynamic dtype.
4. **`packing<T>::num()` (1 / 2 / 4)** —
   [common/base_types.cuh:181-269](HipKittens/include/common/base_types.cuh#L181-L269)
   — determines how many scalars share one register; fp8x4 packs 4-wide, bf16_2
   packs 2-wide, float stays 1-wide.
5. **`convertor<T,U>` specializations** —
   [common/base_types.cuh:277-416](HipKittens/include/common/base_types.cuh#L277-L416)
   — inline-asm `v_cvt_pk_bf16_f32` etc.; type-dispatched at instantiation.
6. **`std::bit_cast` for constants** —
   [common/base_types.cuh:123-166](HipKittens/include/common/base_types.cuh#L123-L166)
   — `bf16::ones = 0x3F80`; constexpr without non-constexpr intrinsics.
7. **`shared_allocator<default_alignment=16>`** —
   [common/util.cuh:268-334](HipKittens/include/common/util.cuh#L268-L334)
   — bump-pointer; aligns to 16 B, returns reference. No free; lives for
   one kernel.
8. **`KITTENS_DEFAULT_ALIGN = alignas(16)`** —
   [common/util.cuh:258](HipKittens/include/common/util.cuh#L258)
   — applied to every tile struct.
9. **Inline-asm-offset clamps** —
   [common/macros.cuh:161-181](HipKittens/include/common/macros.cuh#L161-L181)
   — `max_ds_inst_offset()`, `max_mubuf_inst_offset()` keep the 12/13-bit
   immediate-offset fields from overflowing.
10. **AGPR / VGPR clobber range macros** —
    [common/macros.cuh:15-156](HipKittens/include/common/macros.cuh#L15-L156)
    — enumerate every register index 0..255 so inline-asm constraints can be
    written by logical range, not by name.
11. **`ducks::base_types::{T1, T2}` concepts** —
    [common/base_types.cuh:65-81](HipKittens/include/common/base_types.cuh#L65-L81)
    — `T1` = scalar (float / bf16 / half / fp8 / fp4); `T2` = packed
    (`float2 / bf16_2 / half_2 / fp8x4`). Used to constrain template params.
12. **No `KITTENS_CDNA3` / `KITTENS_CDNA4` in core/** — arch detection is
    deferred to ops layer; tile types are arch-agnostic.

### 1.2 Register tiles (`rt`) and vectors (`rv`)

1. **`rt_base<T, layout, shape>`** —
   [types/register/rt_base.cuh:45-81](HipKittens/include/types/register/rt_base.cuh#L45-L81)
   — `dtype data[packed_per_thread]` inline array; no pointer.
2. **`rt_shape`** —
   [types/register/rt_shape.cuh:20-47](HipKittens/include/types/register/rt_shape.cuh#L20-L47)
   — fields `rows`, `cols`, `stride`; derives
   `elements_per_thread = (rows*cols)/WARP_THREADS` and
   `num_strides = elements_per_thread / stride`.
3. **Layout tags `row` / `col`** —
   [types/register/rt_layout.cuh:19-40](HipKittens/include/types/register/rt_layout.cuh#L19-L40)
   — picked at instantiation; transpose swaps row↔col.
4. **Packed footprint** —
   [types/register/rt_base.cuh:72-75](HipKittens/include/types/register/rt_base.cuh#L72-L75)
   — `packed_per_thread = elements_per_thread / packing::num()`;
   `registers_per_thread = packed_per_thread * sizeof(dtype) / 4`. Densest
   packing = fp8x4 (4 scalars per dword).
5. **Reduction-axis selector** —
   [types/register/rt_base.cuh:66-68](HipKittens/include/types/register/rt_base.cuh#L66-L68)
   — `reductions = (layout == row) ? cols : rows`; tells the matmul which axis
   is the contraction.
6. **`rv<T, length, tile_length, shape, layout>`** —
   [types/register/rv.cuh:48-83](HipKittens/include/types/register/rv.cuh#L48-L83)
   — `outer_dim × inner_dim` elements with layout-dependent replication.
7. **Three `rv` layouts: `naive` / `ortho` / `align`** —
   [types/register/rv_layout.cuh:17-36](HipKittens/include/types/register/rv_layout.cuh#L17-L36)
   — `naive` (no replication; layernorm-style), `ortho` (2× replication for
   minor-axis sums), `align` (32× replication for register-friendly
   broadcast). Trade register count for coalesced ops.
8. **`rv::inner_dim` formula** —
   [types/register/rv.cuh:64-65](HipKittens/include/types/register/rv.cuh#L64-L65)
   — `naive: ceil(length/64); ortho: 1; align: elements_per_thread/packing`.
9. **Transpose specializations** —
   [types/register/rt_shape.cuh:53-59](HipKittens/include/types/register/rt_shape.cuh#L53-L59)
   — `rt_16x32 ↔ rt_32x16`, `rt_32x32` self-transposes, `rt_16x128` not
   transposed.
10. **`rv` indexing** —
    [types/register/rv.cuh:79-82](HipKittens/include/types/register/rv.cuh#L79-L82)
    — `operator[int]` for outer, `operator[int2]` for `(outer, inner)`. No
    bounds checks; compile-time-validated shape.
11. **`rt<T, rows, cols, layout, shape>` composite** —
    [types/register/rt.cuh:55-89](HipKittens/include/types/register/rt.cuh#L55-L89)
    — `height × width` grid of `rt_base` cells (each 16×16 or 32×32).
    Hierarchical matmul applies per cell.
12. **Layout *inverts* `row_vec` / `col_vec`** —
    [types/register/rt_base.cuh:77-78](HipKittens/include/types/register/rt_base.cuh#L77-L78)
    — row-layout ⇒ `row_vec = align` (column reductions),
    `col_vec = ortho` (row reductions). Col-layout flips both.

### 1.3 `art` — assembly-mode accumulator register tile

`art` mirrors `rt` but replaces the inline data array with a **register range**
so MFMA output registers can be scheduled explicitly without an intermediate
VGPR copy.

1. **`art_base<T, layout, shape, register_range>`** —
   [types/register/art_base.cuh:47-90](HipKittens/include/types/register/art_base.cuh#L47-L90).
2. **`register_range::size == registers_per_thread`** static assert —
   [art_base.cuh:83-84](HipKittens/include/types/register/art_base.cuh#L83-L84)
   — guarantees no silent spill.
3. **Per-stride register grouping** —
   [art_base.cuh:78-80](HipKittens/include/types/register/art_base.cuh#L78-L80)
   `registers_per_stride = registers_per_thread / num_strides`. Matches the
   MFMA's stride-group output pattern.
4. **fp8 family supported** (e4m3, e4m3fnuz, e5m2) —
   [art_base.cuh:58-63](HipKittens/include/types/register/art_base.cuh#L58-L63)
   — wider dtype coverage than `rt_base`, reflecting MFMA-f8 paths.
5. **`row_vec` / `col_vec` mirror rt rules** —
   [art_base.cuh:86-87](HipKittens/include/types/register/art_base.cuh#L86-L87).
6. **`split_one<L, R, N>` range partitioner** —
   [art.cuh:46-88](HipKittens/include/types/register/art.cuh#L46-L88)
   — recursively splits `[L, R]` into N-aligned blocks (e.g. N=4 for the
   MFMA's contiguous-4-VGPR-source constraint).
7. **Type-list composition** —
   [art.cuh:29-44](HipKittens/include/types/register/art.cuh#L29-L44)
   `type_list<range<L,R>, …>` with `concat` for heterogeneous accumulator
   sets; compile-time verifies non-overlapping ranges.
8. **Convenience aliases `art_base_fl/bf/hf`** —
   [art_base.cuh:114-116](HipKittens/include/types/register/art_base.cuh#L114-L116)
   — default `range<0, 1>` for small single-VGPR accumulators.

### 1.4 Shared tiles (`st`) and vectors (`sv`) + LDS allocator

1. **`st<T, rows, cols, shape>`** —
   [types/shared/st.cuh:49-92](HipKittens/include/types/shared/st.cuh#L49-L92)
   — flat `dtype data[rows*cols]`; element address goes through
   `shape::swizzle(coord)`.
2. **Swizzle lives in the shape, not the tile** —
   [st.cuh:83-85](HipKittens/include/types/shared/st.cuh#L83-L85)
   — `swizzle(int2 coord)` returns the byte offset; the tile is a thin wrapper.
3. **`st_16x16` (4-byte dtypes, no swizzle)** —
   [st_shape.cuh:19-46](HipKittens/include/types/shared/st_shape.cuh#L19-L46)
   — linear `T_size * (r * cols + c)`.
4. **`st_16x16_swizzled` (2-byte dtypes)** —
   [st_shape.cuh:48-81](HipKittens/include/types/shared/st_shape.cuh#L48-L81)
   — XOR the byte offset with `((offset % 512) >> 7) << 3` to break 4-way
   bank conflicts.
5. **`st_32x32` two-stage XOR** —
   [st_shape.cuh:83-114](HipKittens/include/types/shared/st_shape.cuh#L83-L114)
   — combines `((offset % 1024) >> 9) << 5` and `((offset % 2048) >> 10) << 4`.
6. **`st_16x32` / `st_32x16` asymmetric** —
   [st_shape.cuh:116-178](HipKittens/include/types/shared/st_shape.cuh#L116-L178)
   — single XOR window sized by the longer axis.
7. **`st_16x128` for fp8 (1-byte)** —
   [st_shape.cuh:180-236](HipKittens/include/types/shared/st_shape.cuh#L180-L236)
   — `((offset % (16*128)) >> 8) << 4`; tall-narrow fp8 tiles need extra
   shuffle.
8. **`bytes_per_thread<T>()` constexpr** —
   [st_shape.cuh:23-30, 88-95](HipKittens/include/types/shared/st_shape.cuh#L23-L30)
   — 16 B per thread for 2/4-byte types; sets LDS-load bandwidth quota.
9. **`st_subtile<ST, sub_rows, sub_cols>`** —
   [st.cuh:105-159](HipKittens/include/types/shared/st.cuh#L105-L159)
   — pointer + offsets into a parent tile; zero-copy hierarchical view.
10. **Subtile constructor** —
    [st.cuh:142-150](HipKittens/include/types/shared/st.cuh#L142-L150)
    — base ptr = `subtile_id * underlying_subtile_elements`; inherits the
    parent's swizzle.
11. **`sv<T, length>` flat shared vector** —
    [sv.cuh:44-65](HipKittens/include/types/shared/sv.cuh#L44-L65)
    — no swizzle / replication; used where coalesced row-major beats
    bank-conflict avoidance (reductions, accumulators).
12. **`shared_allocator` bump pointer** —
    [common/util.cuh:268-334](HipKittens/include/common/util.cuh#L268-L334)
    — `allocate<[align], T, dims…>()` advances and returns a reference. No
    free; alignment passable per call.

### 1.5 Global layout (`gl`)

1. **`gl<T, b, d, r, c, TMA_Types...>`** —
   [types/global/gl.cuh:33-91](HipKittens/include/types/global/gl.cuh#L33-L91)
   — wraps `T*` + 4-D shape; template `-1` ⇒ runtime, positive ⇒ compile-time.
2. **Static-or-dynamic dim dispatch** —
   [gl.cuh:50-57](HipKittens/include/types/global/gl.cuh#L50-L57)
   — `if (B > 0)` returns `constexpr`; else falls back to a runtime member.
   Zero-cost when known.
3. **`stride<axis>()`** —
   [gl.cuh:84-90](HipKittens/include/types/global/gl.cuh#L84-L90)
   — row-major product of later dims; `stride[3] = 1`.
4. **`operator[](coord)`** —
   [gl.cuh:71-76](HipKittens/include/types/global/gl.cuh#L71-L76)
   — inline `((b*d + d)*r + r)*c + c` arithmetic; no stride-table lookup.
5. **TMA descriptor slot** —
   [gl.cuh:45-48, 59](HipKittens/include/types/global/gl.cuh#L45-L48)
   — variadic `TMA_Types` + `tma_descs` member preserve compiled TMA
   descriptors per layout (forward-looking; CDNA TMA-equivalent paths).
6. **`make_gl<GL, safe>(...)` factory** —
   [gl.cuh:118-140](HipKittens/include/types/global/gl.cuh#L118-L140)
   — optional bounds check; returns a typed `GL`.
7. **`make_unsafe_gl_arg<N>(param)`** —
   [gl.cuh:114-117](HipKittens/include/types/global/gl.cuh#L114-L117)
   — returns `nullptr` for compile-time dims, runtime value otherwise;
   template-driven arg pack.
8. **No in-place swizzle on global** — global memory is treated as raw
   row-major; tiling decisions live in `st` and the ops layer.
9. **`ducks::gl::all` concept** —
   [gl.cuh:102-104](HipKittens/include/types/global/gl.cuh#L102-L104)
   — `T::identifier == ducks::gl::identifier`; lets ops template-dispatch on
   "is global layout".
10. **4-D shape covers batched matmul / multi-head attention** without
    recursion or variadic strides.

### 1.6 Cross-cutting techniques (this slice)

- **T1.A — Compile-time shape algebra**: every tile's `rows / cols / stride /
  elements_per_thread / num_strides / packed_per_thread` is a constexpr
  computed once; the compiler folds all division and modulo before codegen.
- **T1.B — Packing factor controls register density**: `fp8x4` (packing=4)
  is 4× denser than `bf16_2` (packing=2), which is 2× denser than `float`
  (packing=1) — the same `rt_base<…>` instantiation scales its
  `registers_per_thread` accordingly.
- **T1.C — Layout *inverts* row/col reduction shape**: `row` layout makes
  `row_vec = align` and `col_vec = ortho`; `col` layout flips them — so the
  same algorithm code reads the right replication pattern for either matmul
  operand.
- **T1.D — Swizzle as XOR mask, not pre-rotation**: `st_16x16_swizzled` and
  friends XOR the byte offset with windowed bit shifts so accesses scatter
  across LDS banks without rearranging the data.
- **T1.E — `art` = explicit register-range allocator**: assembly-mode
  accumulator type lets MFMA results land in named register ranges,
  enforced by compile-time non-overlap checks via `type_list`.
- **T1.F — `st_subtile` is a zero-copy view**: hierarchical tile decomposition
  for staged matmul / softmax falls out of pointer arithmetic + inherited
  static swizzle.
- **T1.G — Bump-pointer LDS allocator with default 16 B align**: no free, one
  kernel lifetime, alignment overridable per call — fits the
  warp-decides-its-own-LDS-layout posture of HipKittens.
- **T1.H — Compile-or-runtime dims in `gl`**: `-1` template param ⇒ runtime,
  positive ⇒ compile-time. Same source supports both batched and
  fixed-batch kernels.

Cross-references: [ck:§1] (CK_tile's `tile_distribution / tile_window`
abstractions cover the same conceptual ground in a different shape);
[ck:§9.2] (LDS bank padding — same problem, different solution).

<!-- SLICE-2-INSERT -->
## §2. Warp-level ops

Warp ops are split into **memory** (global ↔ shared ↔ register), **register**
(MFMA / transpose / cast / elementwise / reduce), and **shared** (st ↔ rt
swizzle-aware loads / stores). The aggregator
[warp.cuh:1-13](HipKittens/include/ops/warp/warp.cuh#L1-L13) just pulls the
three subtrees.

### 2.1 Memory ops — global ↔ shared ↔ register

1. **Buffer-resource (V#) builder** —
   [memory/util/util.cuh:69-88](HipKittens/include/ops/warp/memory/util/util.cuh#L69-L88)
   — `buffer_resource` packs ptr + range + swizzle config (bits[13:0] stride,
   bit 14 cache-swizzle, bit 15 enable). Cache-swizzle stride capped at
   `0x3FFF` (≤ 8 KiB).
2. **`load_global_vec2/vec4`** —
   [memory/util/util.cuh:43-67](HipKittens/include/ops/warp/memory/util/util.cuh#L43-L67)
   — inline asm `"global_load_dwordx2 %0, %1, off\n"` /
   `"global_load_dwordx4 …"`; avoids the generic flat-load instruction
   selector.
3. **Global → register row-major vectorized** —
   [memory/tile/global_to_register.cuh:58-122](HipKittens/include/ops/warp/memory/tile/global_to_register.cuh#L58-L122)
   — picks `buffer_load_b128` (stride==8, 16 B) or `buffer_load_b64`
   (stride==4, 8 B) at compile time.
4. **Global → register col-major serial** —
   [memory/tile/global_to_register.cuh:133-168](HipKittens/include/ops/warp/memory/tile/global_to_register.cuh#L133-L168)
   — each lane scalars `(row + l*2)*row_stride + col`; no vectorization
   possible without transpose.
5. **Coherency config `0x00020000`** —
   [memory/tile/global_to_register.cuh:39-42](HipKittens/include/ops/warp/memory/tile/global_to_register.cuh#L39-L42)
   — the standard `cache_all` bit pattern; reused across unroll loops to
   amortize V# setup.
6. **Global → shared async DMA** —
   [memory/tile/global_to_shared.cuh:61-68](HipKittens/include/ops/warp/memory/tile/global_to_shared.cuh#L61-L68)
   — `llvm_amdgcn_raw_buffer_load_lds(srsrc, lds_ptr, bytes_per_thread,
   swizzled_global_byte_offset, 0, 0, coherency::cache_all)`. LDS dest is
   *implicit* via M0.
7. **`prefill_swizzled_offsets()` separates address math from DMA** —
   [memory/tile/global_to_shared.cuh:117-149, 183-243](HipKittens/include/ops/warp/memory/tile/global_to_shared.cuh#L117-L243)
   — kernel computes all offsets once, replays them per launch.
8. **SGPR-pinned LDS base bump** —
   [memory/tile/global_to_shared.cuh:258-314](HipKittens/include/ops/warp/memory/tile/global_to_shared.cuh#L258-L314)
   — `asm("s_mov_b32 m0, %0" :: "s"(lds_byte))` locks M0; per-iteration
   `s_add_u32`; no `readfirstlane()` in the hot loop.
9. **`bytes_per_thread = 16` hardcoded** for the fast path —
   [memory/tile/global_to_shared.cuh:268-271](HipKittens/include/ops/warp/memory/tile/global_to_shared.cuh#L268-L271)
   — forces 16-aligned bumps and lets the compiler drop modulo arithmetic.
10. **Leftover-lane tail loop** —
    [memory/tile/global_to_shared.cuh:72-106](HipKittens/include/ops/warp/memory/tile/global_to_shared.cuh#L72-L106)
    — gates `warpid < leftover_warps` so partial tiles don't overshoot LDS.
11. **No explicit `s_waitcnt lgkmcnt` between sequential DMAs** —
    [memory/tile/global_to_shared.cuh:61](HipKittens/include/ops/warp/memory/tile/global_to_shared.cuh#L61)
    — LDS ordering is implied by the consuming `ds_read`; the launcher does
    not insert software fences.
12. **`coherency` enum** —
    [memory/util/util.cuh:14-19](HipKittens/include/ops/warp/memory/util/util.cuh#L14-L19)
    — `cache_all / cache_global / cache_stream / non_temporal`; only
    `cache_all` is currently used in the DMA path.
13. **`as3_uint32_ptr` typed-LDS attribute** —
    [memory/util/util.cuh:114](HipKittens/include/ops/warp/memory/util/util.cuh#L114)
    — `address_space(3)` annotation; stops LLVM from inserting a generic→LDS
    conversion.
14. **Conditional vectorization width for register vectors** —
    [memory/vec/global_to_register.cuh:68-124](HipKittens/include/ops/warp/memory/vec/global_to_register.cuh#L68-L124)
    — `buffer_load_b128 / b64 / b32` dispatch on `inner_dim_bytes`; scalar
    fallback otherwise.

### 2.2 Async loads + fences

1. **`llvm_amdgcn_raw_buffer_load_lds` extern signature** —
   [memory/util/util.cuh:117-124](HipKittens/include/ops/warp/memory/util/util.cuh#L117-L124)
   `(rsrc, lds_ptr, size, voffset, soffset, offset=0, aux)` — maps to
   `BUFFER_LOAD_LDS`; bypasses VGPR entirely.
2. **LDS destination via M0 auto-increment** — caller writes M0 once;
   hardware increments per DMA beat. The intrinsic's `lds_ptr` argument is
   compiled to a `v_` register holding the *base* LDS offset.
3. **M0 SGPR contract** —
   [memory/tile/global_to_shared.cuh:300](HipKittens/include/ops/warp/memory/tile/global_to_shared.cuh#L300)
   — inline asm `s_mov_b32 m0, %0` with `"s"` constraint and `"memory"`
   clobber; same discipline as CK [ck:§2.4].
4. **Global swizzle mirrors LDS swizzle for bank-free landing** —
   [memory/tile/global_to_shared.cuh:52-56](HipKittens/include/ops/warp/memory/tile/global_to_shared.cuh#L52-L56)
   `swizzled_global_byte_offset = (swizzled_global_row * row_stride +
   swizzled_global_col) * sizeof(T)` so the DMA writes consecutive LDS banks.
5. **Pre-fill / launch split** lets the address calculation be hoisted out
   of the hot loop: one global offset is reused for every double-buffer
   flip.
6. **No DMA-completion polling** — unlike CUDA's `__wait_lds_remote_writes()`.
   Implicit ordering: the next `ds_read` stalls until the prior
   `BUFFER_LOAD_LDS` retires.
7. **Async fence inserted at the *callsite* of the consumer**, not by the
   memory op — caller's `__syncthreads()` or block-level barrier provides the
   coarse fence.

### 2.3 Register-tile ops — MFMA / transpose / cast / elementwise / reduce

1. **`mfma161632` (bf16 → fp32, 16×16×32)** —
   [register/tile/mma.cuh:13-41](HipKittens/include/ops/warp/register/tile/mma.cuh#L13-L41)
   — `__builtin_amdgcn_mfma_f32_16x16x32_bf16(*(bf16x8_t*)A, *(bf16x8_t*)B,
   *(float4_t*)C, 0, 0, 0)`; vector-type casts pass operands as registers.
2. **`mfma323216` (bf16 → fp32, 32×32×16)** —
   [register/tile/mma.cuh:42-56](HipKittens/include/ops/warp/register/tile/mma.cuh#L42-L56)
   — 8-wide float-pair accumulator.
3. **`mfma323232` chains two `mfma_f32_32x32x16_bf16`** —
   [register/tile/mma.cuh:74-95](HipKittens/include/ops/warp/register/tile/mma.cuh#L74-L95)
   — back-to-back over `A[0..3]` then `A[4..7]`; latency-hiding pattern.
4. **`mfma_scale_f32_32x32x64_f8f6f4` (gfx950)** —
   [register/tile/mma.cuh:97-110](HipKittens/include/ops/warp/register/tile/mma.cuh#L97-L110)
   — six trailing zeros = scale / bias indices + flags; OpSel-style scale
   wiring matches [ck:§5.3] / [ck:§16.11].
5. **`vector_size` typedefs for register packing** —
   [register/tile/mma.cuh:18-25](HipKittens/include/ops/warp/register/tile/mma.cuh#L18-L25)
   — `__attribute__((vector_size(8*sizeof(__bf16)))) __bf16 bf16x8_t` so
   operands lower to VGPR pairs directly.
6. **`permlane16_swap` for 16×32 ↔ 16×16 col-layout merge** —
   [register/tile/conversions.cuh:37-54](HipKittens/include/ops/warp/register/tile/conversions.cuh#L37-L54)
   — packs two bf16_2[2] tiles into one bf16_2[4]; odd src[0] ↔ even
   src[1].
7. **`permlane32_swap` row reduction** —
   [register/tile/reductions.cuh:59-68](HipKittens/include/ops/warp/register/tile/reductions.cuh#L59-L68)
   — one-instruction 32-lane horizontal reduction for bf16 / half / float.
8. **`__shfl_down` butterfly fallback** —
   [register/tile/reductions.cuh:70-74](HipKittens/include/ops/warp/register/tile/reductions.cuh#L70-L74)
   — used when `base_tile_rows != 32`.
9. **`unary_map<op, T>` element-wise** —
   [register/tile/maps.cuh:24-35](HipKittens/include/ops/warp/register/tile/maps.cuh#L24-L35)
   — iterates `height * width * packed_per_base_tile`; `#pragma unroll`
   inlines the op.
10. **`bin_map<op, T>` binary element-wise** —
    [register/tile/maps.cuh:82-94](HipKittens/include/ops/warp/register/tile/maps.cuh#L82-L94)
    — no loop-carried dependency.
11. **`convertor<T2, U2>::convert` cast hook** —
    [memory/tile/global_to_register.cuh:117](HipKittens/include/ops/warp/memory/tile/global_to_register.cuh#L117)
    — bf16 ↔ float, fp8 ↔ float applied per-element during load.
12. **Packed-iteration width = `packing<T>::num()`** —
    [register/tile/maps.cuh:30-31](HipKittens/include/ops/warp/register/tile/maps.cuh#L30-L31)
    — 2 for bf16/half, 1 for float, 4 for fp8.
13. **Chained MFMA with no `s_waitcnt` between calls** —
    [register/tile/mma.cuh:82-95](HipKittens/include/ops/warp/register/tile/mma.cuh#L82-L95)
    — exactly the wave-level MFMA ILP pattern of [ck:§16.9]: second MFMA
    issues while first is in flight.
14. **In-place layout swap for bf16** —
    [register/tile/conversions.cuh:94-96](HipKittens/include/ops/warp/register/tile/conversions.cuh#L94-L96)
    — pointer-pun `swap_layout_inplace()`; zero data movement.

### 2.4 Shared-tile ops — st ↔ rt, swizzle-aware

1. **`subtile_inplace<rows, cols>(src, rowcol)`** —
   [shared/tile/conversions.cuh:29-35](HipKittens/include/ops/warp/shared/tile/conversions.cuh#L29-L35)
   — returns an `st_subtile<…>` proxy; compile-time divisibility check.
2. **`dst.swizzle({row, col})` everywhere** — single API hides the actual
   XOR-mask math from callers (§1.4); same call site for load, store, sub-
   tile addressing.
3. **`ds_read_b64` / `ds_read_b128` row-major** —
   [memory/tile/shared_to_register.cuh:78-94](HipKittens/include/ops/warp/memory/tile/shared_to_register.cuh#L78-L94)
   — inline asm `"ds_read_b64 %0, %1 offset:%2\n"` /
   `"ds_read_b128 %0, %1 offset:%2\n"` with the swizzled offset.
4. **Free transpose via `ds_read_b64_tr_b16`** —
   [memory/tile/shared_to_register.cuh:247-275](HipKittens/include/ops/warp/memory/tile/shared_to_register.cuh#L247-L275)
   — dual `"ds_read_b64_tr_b16 %0, %2 offset:%3\n"`; bf16 / fp16
   16-element transpose during the read itself. Same instruction CK lists
   but doesn't wire up — HipKittens uses it.
5. **Dual `ds_write_b64` row-major store** —
   [memory/tile/shared_to_register.cuh:478-497](HipKittens/include/ops/warp/memory/tile/shared_to_register.cuh#L478-L497)
   — adjacent-bank pair for stride==8.
6. **Col-major store strides by rows** —
   [memory/tile/shared_to_register.cuh:575-689](HipKittens/include/ops/warp/memory/tile/shared_to_register.cuh#L575-L689)
   — `row_offset = base_tile_stride * (laneid / base_tile_cols)` plus
   stride-group index.
7. **Row-major load size-dispatch** —
   [memory/tile/shared_to_register.cuh:50-115](HipKittens/include/ops/warp/memory/tile/shared_to_register.cuh#L50-L115)
   — splits on `ST::underlying_subtile_rows >= RT::base_tile_rows` vs
   `<=`; one or many register subtiles per shared subtile.
8. **`sv` (shared vector) load — `align` layout** —
   [memory/vec/shared_to_register.cuh:36-50](HipKittens/include/ops/warp/memory/vec/shared_to_register.cuh#L36-L50)
   — `idx = w*RV::reductions + RV::stride * (laneid / RV::aligned_threads)`;
   coalesced stride across lanes.
9. **`sv` store — `ortho` layout** —
   [memory/vec/shared_to_register.cuh:104-110](HipKittens/include/ops/warp/memory/vec/shared_to_register.cuh#L104-L110)
   — `idx = w*RV::reductions + (laneid % RV::reductions)`; one scalar per
   lane spread across the warp.
10. **No LDS-to-LDS instruction** — smem-to-smem moves always route through
    a register intermediary; the register file is the implicit pipeline
    stage.
11. **Subtile divisibility asserts** —
    [shared/tile/conversions.cuh:31-33](HipKittens/include/ops/warp/shared/tile/conversions.cuh#L31-L33)
    `ST::rows % subtile_rows == 0`; warp-local subtiles cannot alias.

### 2.5 Cross-cutting techniques (this slice)

- **T2.A — `BUFFER_LOAD_LDS` + M0 + matching global / shared swizzle**: the
  three together give bank-conflict-free DRAM→LDS with no VGPR detour.
- **T2.B — Pre-fill swizzled offsets**: address arithmetic hoisted *out* of
  the inner loop; the DMA loop just steps through a precomputed array.
- **T2.C — `ds_read_b64_tr_b16` *is* wired up here**: where CK has the
  builtin but no pipeline calls it [ck:§16.12.5], HipKittens uses it on
  every col-major shared→register load.
- **T2.D — Back-to-back MFMA without `s_waitcnt`**: same wave-level ILP
  pattern as [ck:§16.9]; compiler picks up the implicit dependency edge.
- **T2.E — `permlane16/32_swap` for register-only transpose**: zero-cycle
  shuffle relative to MFMA latency; overlapped with compute.
- **T2.F — Compile-time vector-width dispatch**: `buffer_load_b128 / 64 / 32`
  selected from `inner_dim_bytes`; no runtime branch.
- **T2.G — `as3_uint32_ptr` (address-space-3 attribute)** — prevents LLVM
  from inserting a generic-to-LDS conversion before the inline asm.
- **T2.H — No explicit `s_waitcnt lgkmcnt` between DMA launches**: relies on
  the consumer's `ds_read` to stall implicitly; software fences live at
  block-sync points only.
- **T2.I — gfx950 `mfma_scale_f32_32x32x64_f8f6f4` exposed at warp level**:
  scale arguments threaded through directly; no codegen layer.

Cross-references: [ck:§2] (CK_tile arch wrappers); [ck:§16.9] (wave-level
MFMA ILP); [ck:§16.12.5] ("`ds_read_tr*` declared but unused" — HipKittens
*does* use it).

<!-- SLICE-3-INSERT -->
## §3. Group-level ops

The `group<N_WARPS>` template extends the warp-scope ops of §2 to *N* warps
that share an LDS region. Notably, the group layer has **no explicit
synchronization primitive** — coordination relies entirely on disjoint LDS
ranges per warp and on the implicit ordering already present in the warp-level
memory operations.

### 3.1 The `group<N_WARPS>` abstraction

1. **`group<N_WARPS>`** —
   [ops/group/group.cuh:20](HipKittens/include/ops/group/group.cuh#L20)
   `GROUP_WARPS = N_WARPS`, `GROUP_THREADS = N_WARPS * 64`.
2. **Identity helpers** —
   `laneid() = threadIdx.x % GROUP_THREADS`,
   `warpid() = laneid() / 64`,
   `groupid() = threadIdx.x / GROUP_THREADS`.
3. **`warpgroup = group<4>` alias** —
   [ops/group/group.cuh:30](HipKittens/include/ops/group/group.cuh#L30)
   — convention name for the canonical 4-warp coordination unit (mirrors
   Hopper's "warpgroup" terminology even though CDNA has no real WGMMA).
4. **No barrier / broadcast / reduce primitives** at the group level — all
   collective behavior comes from how disjoint LDS regions are addressed.
5. **`WARP_THREADS = 64` hard floor** —
   [common/util.cuh:32](HipKittens/include/common/util.cuh#L32)
   — group ops are Wave64-only, consistent with §1.

### 3.2 Group memory ops

Each group memory op is a thin wrapper that re-instantiates the warp-level op
with `GROUP_THREADS` instead of `WARP_THREADS`:

1. **`group::load(ST&, GL&, coord)`** —
   [ops/group/group.cuh:7](HipKittens/include/ops/group/group.cuh#L7)
   — forwards to `kittens::load<…, GROUP_THREADS>(…)` so the underlying
   warp op derives `num_warps = GROUP_THREADS / 64`.
2. **Warp ID masking inside the warp op** —
   [ops/warp/memory/tile/global_to_shared.cuh:26](HipKittens/include/ops/warp/memory/tile/global_to_shared.cuh#L26)
   — `num_warps = GROUP_THREADS / 64`; each warp uses `warpid % num_warps`
   so unused warps are silently skipped.
3. **Per-warp disjoint LDS region** —
   [ops/warp/memory/tile/global_to_shared.cuh:36](HipKittens/include/ops/warp/memory/tile/global_to_shared.cuh#L36)
   — `lds_base + warpid * bytes_per_warp`; no two warps target the same
   bank set.
4. **Strided iteration over the tile** —
   [ops/warp/memory/tile/global_to_shared.cuh:43](HipKittens/include/ops/warp/memory/tile/global_to_shared.cuh#L43)
   — `for (i = 0; i < memcpy_per_tile; i++)` advances by `i * num_warps *
   bytes_per_warp`.
5. **`BUFFER_LOAD_LDS` re-used unchanged** —
   [ops/warp/memory/tile/global_to_shared.cuh:61](HipKittens/include/ops/warp/memory/tile/global_to_shared.cuh#L61)
   — the same intrinsic as §2.2; multi-warp scaling is purely an offset trick.
6. **`group::prefill_swizzled_offsets(ST, GL, uint32_t*)`** —
   [ops/group/memory/tile/global_to_shared.cuh:22](HipKittens/include/ops/group/memory/tile/global_to_shared.cuh#L22)
   — multi-warp offset pre-fill; offsets stored in an array so the inner
   loop has no swizzle math.
7. **Vec loads** —
   [ops/group/memory/vec/global_to_shared.cuh:7](HipKittens/include/ops/group/memory/vec/global_to_shared.cuh#L7)
   — `num_memcpys = (SV::length * sizeof(T)) / (GROUP_THREADS * 4)`;
   per-thread bytes fixed at 4.
8. **`group::store<axis>(…)`** —
   [ops/group/group.cuh:14-15](HipKittens/include/ops/group/group.cuh#L14-L15)
   — mirrors `load`; same `GROUP_THREADS` re-instantiation.
9. **No special-cased intra-LDS group move** — smem-to-smem still routes
   through registers (same as §2.4 / [ck:§9.6]).

### 3.3 Cross-warp barriers / broadcast / reduction — *deliberately absent*

1. **No `__syncthreads()` or named barrier in group ops** — the only
   serialization point is whatever the kernel author drops between group
   calls (typically a single `__syncthreads()` between LDS-fill and
   LDS-consume phases).
2. **Coherency is enforced by SGPR-uniform LDS-bump** (§2.1.8) —
   [ops/warp/memory/tile/global_to_shared.cuh:300-312](HipKittens/include/ops/warp/memory/tile/global_to_shared.cuh#L300-L312)
   — all warps share the same M0 SGPR; the hardware retires per-DMA in order.
3. **No group-scope reduction** — if a kernel needs cross-warp reduction,
   it has to write through a shared vector (`sv`) and use warp shuffles
   (`__shfl_down`, `packed_shfl_down` in
   [common/util.cuh:154-178](HipKittens/include/common/util.cuh#L154-L178))
   or build it manually on top of the warp ops.
4. **No group-scope broadcast** — same story; broadcast is just "store to
   `sv`, every warp reads `sv`."
5. **Leftover-warp handling at compile time** —
   `if constexpr` branches generate a tail-loop only when the tile size
   doesn't divide evenly by `GROUP_THREADS * bytes_per_thread`; no runtime
   branch in the common case.

### 3.4 Cross-cutting techniques (this slice)

- **T3.A — Group ops are warp ops with `GROUP_THREADS` instead of
  `WARP_THREADS`**: a single re-instantiation gives N-warp coordination
  with no new memory primitive.
- **T3.B — Disjoint LDS regions per warp**: synchronization is structural,
  not instructional — by giving each warp its own LDS slice, the group
  never needs an explicit barrier inside the op.
- **T3.C — `prefill_swizzled_offsets()` at group scope hoists *all* address
  math out of the loop**: the inner DMA loop reads offsets from an array.
- **T3.D — `warpid % num_warps` mask silently drops unused warps**: a block
  launched with more warps than the op uses still produces correct results.
- **T3.E — Group does not own a barrier**: the kernel author is responsible
  for `__syncthreads()` between phases — a deliberate design choice that
  keeps the primitive small but pushes correctness up the stack.
- **T3.F — `warpgroup = group<4>` is a naming convention, not a hardware
  unit**: CDNA has no WGMMA; the alias is purely ergonomic.

Cross-references: [ck:§1.2] (`tile_window` does similar warp coordination
through its `pre_computed_warp_coords_`); [ck:§3.6] (CK's `kBlockSize` /
`NumWaves` knobs which HipKittens replaces with template-time `N_WARPS`).

<!-- SLICE-4-INSERT -->
## §4. GEMM kernels — bf16fp32, fp8fp32, FP6 experiments

The `kernels/gemm/` tree holds hand-written matmuls; `fp6_dwordx{3,4}` are two
parallel forks of the whole library with FP6 modifications. The
`agent-mxfp8-gemm-optimization.md` design note documents an 11.9 % speedup
walk-through.

### 4.1 bf16fp32 GEMM

1. **File naming `256_256_64_32_with{16x32,32x16}.cpp`** —
   [kernels/gemm/bf16fp32/](HipKittens/kernels/gemm/bf16fp32/)
   — `256_256_64_32` = block M / N / K_STEP / hidden-K factor;
   `with16x32` / `with32x16` = warp-tile arrangement in register space.
2. **Block tile 256 × 256, K_STEP=64, 8 warps (2 × 4)** —
   `WARPS_M=2`, `WARPS_N=4` (256 threads).
3. **Per-warp register tile 128 × 64** — derived `BLOCK / WARPS_*`.
4. **`st_bf<128, 64, st_16x32_s>` shared tiles** — uses the 16×32 swizzle
   from §1.4.
5. **Accumulator = 4× `rt_fl<64, 64, col_l, rt_16x16_s>` per warp**.
6. **Tic/toc double-buffer** —
   [256_256_64_32_with16x32.cpp:104-105](HipKittens/kernels/gemm/bf16fp32/256_256_64_32_with16x32.cpp#L104-L105)
   — `As[2][2][2]`, `Bs[2][2][2]` indexed by `tile & 1` flip.
7. **Prefill** —
   [:116-131](HipKittens/kernels/gemm/bf16fp32/256_256_64_32_with16x32.cpp#L116-L131)
   — load K=0 into tic; prefetch K=1 into toc.
8. **Hot loop processes 2 K-tiles per iter (`tile += 2`)** —
   [:136-232](HipKittens/kernels/gemm/bf16fp32/256_256_64_32_with16x32.cpp#L136-L232)
   — alternates `load(shared→register)`, prefetch
   `(global→shared)`, four `mma_ABt()` per phase.
9. **Wait-count discipline** — `s_waitcnt vmcnt(6)` paces memory
   ([:132, :179, :225](HipKittens/kernels/gemm/bf16fp32/256_256_64_32_with16x32.cpp#L132));
   `lgkmcnt(8)` after global load
   ([:143](HipKittens/kernels/gemm/bf16fp32/256_256_64_32_with16x32.cpp#L143));
   `lgkmcnt(0)` before each MFMA cluster
   ([:146, :158, :169, :194](HipKittens/kernels/gemm/bf16fp32/256_256_64_32_with16x32.cpp#L146)).
10. **`s_setprio` alternation** — `s_setprio(1)` during MFMA
    ([:147, :149](HipKittens/kernels/gemm/bf16fp32/256_256_64_32_with16x32.cpp#L147)),
    `s_setprio(0)` after; biases the wave scheduler to favor MFMA ALU issue.
11. **MFMA = 16×16×32 bf16** (implicit from "with16x32" naming).
12. **Epilogue** —
    [:234-326](HipKittens/kernels/gemm/bf16fp32/256_256_64_32_with16x32.cpp#L234-L326)
    — last 2 K-tiles unrolled without prefetch; direct
    `store(g.c, C_accum[m][n], …)` with fp32→bf16 cast on store.

### 4.2 fp8fp32 GEMM

1. **`FP8_4wave/4_wave.cu`** — three variants: `matmul_device`,
   `matmul_device_1024`, `matmul_device_2048`. Each picks a different
   `BLOCK_SIZE` (64, 128, 256) but holds 4 waves and `BLOCK_K=128`.
2. **256 × 256 × 128 baseline**, `WARPS_ROW=WARPS_COL=2`, 4 warps total.
3. **`rt_fp8e4m3<128, 128>`** A/B register tiles; **`rt_fl<64, 64,
   col_l, rt_16x16_s>`** accumulator.
4. **`st_fp8e4m3<128, 128, st_16x128_s>` shared tile** — the
   `st_16x128` 1-byte swizzle from §1.4.
5. **Prefill K=0 + K=1 into curr/next** —
   [4_wave.cu:151-183](HipKittens/kernels/gemm/fp8fp32/FP8_4wave/4_wave.cu#L151-L183)
   — accumulators initialized at [:160-163](HipKittens/kernels/gemm/fp8fp32/FP8_4wave/4_wave.cu#L160-L163).
6. **`do_interleaved_cluster()` 4-phase hot loop** —
   [:187-214](HipKittens/kernels/gemm/fp8fp32/FP8_4wave/4_wave.cu#L187-L214)
   — each phase calls `mma_ABt_one(k_phase, warp_offset)` + global
   prefetch + sched barrier.
7. **`s_waitcnt vmcnt(16)` and `lgkmcnt(0)` punctuate phases** —
   [:189, :198, :205, :207, :448](HipKittens/kernels/gemm/fp8fp32/FP8_4wave/4_wave.cu#L189).
8. **MFMA = 16×16×128 fp8** (implicit from K_STEP=128 + e4m3 tile).
9. **Two final K-tiles unrolled in epilogue** —
   [:217-310](HipKittens/kernels/gemm/fp8fp32/FP8_4wave/4_wave.cu#L217-L310)
   — tighter barrier sequencing now that the K bound is known.
10. **Epilogue store** —
    [:315-318](HipKittens/kernels/gemm/fp8fp32/FP8_4wave/4_wave.cu#L315-L318)
    — `store(C, c[warp_m][warp_n], …)`; fp32 accumulator → bf16 output on
    store.

### 4.3 FP6 experiments — `dwordx3` vs `dwordx4`

The two `fp6_dwordx{3,4}/` directories are *complete forks* of the library
plus a kernel; the README documents the comparison.

1. **`fp6_dwordx3` GEMM** —
   [fp6_dwordx3/kernels/gemm/fp6fp32/16x16x128_4wave.cpp](HipKittens/fp6_dwordx3/kernels/gemm/fp6fp32/16x16x128_4wave.cpp)
   — 256×256 output, K_STEP=128, 4 warps. Loads via
   `buffer_load_dwordx3` (12 B; 16 FP6 elements per load).
2. **`load_global_to_shared_direct_unit_fp6`** —
   [:106-114](HipKittens/fp6_dwordx3/kernels/gemm/fp6fp32/16x16x128_4wave.cpp#L106-L114)
   — direct global → LDS via inline asm + M0 bump.
3. **Paired `ds_read_b96` (12 B)** —
   [:179-212](HipKittens/fp6_dwordx3/kernels/gemm/fp6fp32/16x16x128_4wave.cpp#L179-L212)
   — two reads per 16-element tile row.
4. **No explicit `vmcnt` / `lgkmcnt` between clusters** —
   [:174, :194, :222, etc.](HipKittens/fp6_dwordx3/kernels/gemm/fp6fp32/16x16x128_4wave.cpp#L174)
   — comments removed; relies on instruction-latency hiding alone.
5. **`fp6_dwordx4` GEMM** —
   [fp6_dwordx4/analysis/gemm/mi325x/kernel_8192.cpp](HipKittens/fp6_dwordx4/analysis/gemm/mi325x/kernel_8192.cpp)
   — 256×256 output, K_STEP=64 (half), **8 warps** (vs 4); register tile
   `rt_bf<64, 16>` (smaller K).
6. **Register-buffered prefetch** —
   [:91-100](HipKittens/fp6_dwordx4/analysis/gemm/mi325x/kernel_8192.cpp#L91-L100)
   — `load_global_to_registers` lands K+1 in `float4` VGPRs, *not* LDS.
7. **Simpler hot loop** —
   [:87-99](HipKittens/fp6_dwordx4/analysis/gemm/mi325x/kernel_8192.cpp#L87-L99)
   — `load_async(shared → register)` overlapped with prior MFMA; no
   `do_interleaved_cluster`.
8. **Why dwordx4 wins** —
   [README.md](HipKittens/README.md)
   — quote: *"GEMM kernels are global-to-LDS bottlenecked"*; dwordx4 evicts
   prefetch into VGPRs and frees LDS bandwidth; dwordx3 stays LDS-centric and
   hits the same port.

### 4.4 `agent-mxfp8-gemm-optimization.md` — design highlights

A first-person record of an 11.9 % speedup walk from the V3 baseline.

1. **Baseline** —
   [agent-mxfp8-gemm-optimization.md:54-83](HipKittens/agent-mxfp8-gemm-optimization.md#L54-L83)
   — V3 pinned-register pipeline; 2205 → 2291 TFLOPS @ 4K, 2526 → 2643 @ 8K.
2. **Step 1 — sync-instruction swap (QW)** +0.3 % (:74).
3. **Step 2 — SGPR-precomputed `soff`** +3.3 % (:75).
4. **Step 3 — SGPR-precomputed M0** +5.1 % (:76) — 36 VGPR→SGPR address
   conversions removed (:95-96).
5. **Step 4 — fewer `sched_barrier` calls** +5.6 % (:77).
6. **Step 5 — SpreadLDG** +8.9 % (:78) — spread 6 global loads across 8–12
   compute instructions; L2 hit 74 % → 78.5 %, MFMA 52.3 % → 54.2 %
   (:99-103).
7. **Step 6 — earlier data prefetch** +9.7 % (:79) — load N+2 while
   computing N; hides 67 % of memory delay (:107-108).
8. **Step 7 — adaptive WGM (workgroup-mapping for L2 reuse)** +10.2 %
   (:80) — 16K: 2673 → 2950 TFLOPS (:111).
9. **Step 8 — epilogue SpreadLDG** +10.7 % (:81).
10. **Step 9 — hoist loop constants** +11.7 % (:82) — let the compiler
    schedule freely (:115).
11. **Step 10 — unified SRD (single buffer-resource descriptor)** +11.9 %
    (:83).
12. **Failed attempts** —
    [:119-128](HipKittens/agent-mxfp8-gemm-optimization.md#L119-L128)
    — moving loads into compute (-2 to -4 % because `buffer_load_lds` and
    `ds_read` share the LDS port); NT store -18 %; LLC-aware swizzle breaks
    odd shapes; `vmcnt(6)` is optimal (more in-flight doesn't help);
    `s_setprio` / `launch_bounds` no-op at occupancy=1.
13. **End-state** —
    [:30-35](HipKittens/agent-mxfp8-gemm-optimization.md#L30-L35)
    — 8192³ MXFP8 = 2991 TFLOPS (Hyperloom) vs 1512 hipBLASLt (197.8 %),
    cuBLASLt B200 = 3105 TFLOPS (96.3 %).

### 4.5 Cross-cutting techniques (this slice)

- **T4.A — Tic/toc LDS double-buffer + register prefetch one K ahead**:
  every GEMM here pre-loads K+1 into VGPRs / LDS while MFMA chews K.
- **T4.B — `s_waitcnt vmcnt(6)` is the magic number**: the agent doc
  confirms it experimentally — 6 in-flight global loads = sweet spot on
  MI300X / MI355X.
- **T4.C — `s_setprio` alternation favors MFMA over memory**: cheap signal
  to the wave scheduler; sprinkled around every MFMA cluster.
- **T4.D — SGPR-precomputed addresses (soff / M0)**: hoist
  `readfirstlane()` results into SGPRs *once* — turned the +3.3 % and
  +5.1 % wins in the agent walk.
- **T4.E — SpreadLDG spreads global loads across compute**: shares the L2
  port; +4 % wall-clock by reducing port contention.
- **T4.F — Adaptive WGM (workgroup mapping)** tiles workgroups by the
  *N*-dimension of B so that consecutive workgroups reuse B in L2; biggest
  single optimization (+10 % at 16K).
- **T4.G — dwordx4 over dwordx3 for FP6**: register-buffered prefetch
  avoids the LDS port contention that limits dwordx3. Confirms the
  general lesson that *the LDS port, not bandwidth, is often the
  bottleneck.*
- **T4.H — `buffer_load_lds` and `ds_read` share LDS port**: documented
  failed-attempt in the agent walk — moving loads into compute *slows*
  things down because they fight for the same port. This is a
  HipKittens-specific finding the CK playbook doesn't have.
- **T4.I — At occupancy=1 (one block / CU), `s_setprio` and
  `launch_bounds` are no-ops**: documented in the failed-attempts list.
  Confirms [ck:§16.10]'s observation that prefetch depth pushes
  occupancy to 1, after which scheduling tricks don't help.
- **T4.J — Let the compiler schedule**: removing manual scheduling
  constraints (hoisting loop constants) was a +1.0 % win — the human-
  guided assembly-tuning ceiling is real but the compiler often finds
  better schedules than hand-tuning at this granularity.

Cross-references: [ck:§3] (CK GEMM pipelines — different tiling, same
trade-offs); [ck:§16.10] (PrefetchStages occupancy cliff — same effect,
different vocabulary); [ck:§16.11] (MI300 gfx942 features); [asm-v2:§17]
(flatmm-style block layout).

<!-- SLICE-5-INSERT -->
## §5. Attention kernels (gqa fwd / bwd / causal variants)

Four hand-written kernel families: `gqa`, `gqa_causal`, `gqa_backwards`,
`gqa_causal_backwards`. Each has a `kernel.cpp` (head-dim 128) plus a
`kernel_d64.cpp` (head-dim 64 specialization). They are FlashAttention-2 in
shape but with HipKittens-style explicit register pinning.

### 5.1 `gqa` — forward, non-causal

1. **Tile geometry** —
   [kernel.cpp:26-29](HipKittens/kernels/attn/gqa/kernel.cpp#L26-L29)
   — `block_M = 32` (Q), `block_N = 64` (KV), `head_dim = 128 / 64`,
   `num_warps = 8`, 256 threads.
2. **Q loaded once and scaled** —
   [kernel.cpp:180-182](HipKittens/kernels/attn/gqa/kernel.cpp#L180-L182)
   — temperature applied once; Q transposed to enable `K · Qᵀ`.
3. **K/V double-buffer in LDS** —
   [kernel.cpp:107-108, 174-189](HipKittens/kernels/attn/gqa/kernel.cpp#L107-L189)
   — `st_bf<64, 128, st_32x32_s>` for K, `st_bf<64, 128, st_8x32_s>` for V;
   ping-pong on tile index.
4. **Online softmax with rescale-if-shifted predicate** —
   [kernel.cpp:71, 270](HipKittens/kernels/attn/gqa/kernel.cpp#L71)
   — `rv_all_below(max_prev, max_cur, 8.0f)`: only emit rescale when the
   new row-max exceeds the old by more than 8 in log₂ — saves a multiply
   when the running max is stable.
5. **`rescale = exp2(max_prev - max_cur)`** —
   [kernel.cpp:276-278, 348-350](HipKittens/kernels/attn/gqa/kernel.cpp#L276-L350)
   — applied via `mul_col` to both `O` accumulator and `l` sum.
6. **Hot-loop unroll 2** —
   [kernel.cpp:230-376](HipKittens/kernels/attn/gqa/kernel.cpp#L230-L376)
   — `j += 2`; clusters 0–7 interleave two QK + two AV matmuls with the
   softmax body.
7. **`sched_barrier_pairs<10, 5, group>` and `sched_barrier_exp_pairs<6, 3, group>`** —
   [kernel.cpp:246-247, 318-319](HipKittens/kernels/attn/gqa/kernel.cpp#L246-L319)
   — explicit wave-scheduler hints around the softmax.
8. **`s_setprio(1)` during AV MFMA** —
   [kernel.cpp:265, 337](HipKittens/kernels/attn/gqa/kernel.cpp#L265-L337)
   — same pattern as §4.
9. **`readfirstlane` pointer hoist** —
   [kernel.cpp:128-139](HipKittens/kernels/attn/gqa/kernel.cpp#L128-L139)
   — `k_base`, `v_base`, LDS addresses lifted into SGPRs once per warp.
10. **`vmcnt(2)` / `vmcnt(4)` between phases** —
    [kernel.cpp:192, 221](HipKittens/kernels/attn/gqa/kernel.cpp#L192-L221)
    — keeps ~2 async loads outstanding without backing the MFMA up.
11. **`pending_scale` flag** — defers `o_reg *= rescale` until the next
    accumulate completes; avoids redundant scaling when consecutive blocks
    don't shift the row max.
12. **Final normalize + LSE** —
    [kernel.cpp:555, 570-572](HipKittens/kernels/attn/gqa/kernel.cpp#L555-L572)
    — `div_col(o_reg, norm_vec)`; `L_vec = log(norm_vec) + max_vec * ln(2)`.

### 5.2 `gqa_causal` — forward, causal

1. **Branch-free per-lane mask** —
   [kernel.cpp:287-292, 418-422, 560-565](HipKittens/kernels/attn/gqa_causal/kernel.cpp#L287-L565)
   — `mask_kv_tile<RT>(att_block, q_abs, k_abs, neg_inf_v, lane)` applies
   `v_cmp_lt_i32` + `v_cndmask_b32` to set `-inf` where `q_pos < k_pos`.
2. **Whole-tile elision** —
   [kernel.cpp:223-227](HipKittens/kernels/attn/gqa_causal/kernel.cpp#L223-L227)
   — `max_num_tiles = min(ceil((max_q_end_pos) / block_N), num_tiles)`;
   tiles past the causal frontier are *not iterated*.
3. **Mask placed *post-MFMA, pre-softmax*** —
   [kernel.cpp:285, 398, 543](HipKittens/kernels/attn/gqa_causal/kernel.cpp#L285-L543)
   — `-inf` rows contribute zero to `exp2`.
4. **Boundary guard on async load** —
   [kernel.cpp:289, 419-420](HipKittens/kernels/attn/gqa_causal/kernel.cpp#L289-L420)
   — `qo_start_pos < kv_end_pos` predicate prevents OOB K/V fetches.
5. **`__builtin_expect` predicts the common-case "fully causal"** —
   `if (__builtin_expect(q_start_pos < kv_end_pos, 0))` so the branch
   predictor treats the masking branch as cold.
6. **Constexpr `neg_inf_v` register** —
   `constexpr int neg_inf_v = 29;` + `v_mov_b32_up2p<neg_inf_v>(0xff800000)`
   pre-loads `-inf` into VGPR 29; reused by every mask op.
7. **Otherwise identical to §5.1** — same online softmax, same K/V
   double-buffer, same `s_setprio` discipline.

### 5.3 `gqa_backwards` — backward, non-causal

1. **Files** — `attn_bkwd_non_causal.cpp` (main), `utils.cpp` (helpers).
2. **Geometry** —
   [attn_bkwd_non_causal.cpp:24-30](HipKittens/kernels/attn/gqa_backwards/attn_bkwd_non_causal.cpp#L24-L30)
   `STEP_QO = 64`, `BLOCK_SIZE_KV = 256`, `DOT_SLICE_QO = 16`,
   `WARP_SIZE_KV = 64`, `num_warps = 4`.
3. **Forward recomputed inside the kernel** — `P_ij = exp2(QKᵀ - L_i)` with
   `L_i` loaded from the forward pass's LSE output ([:220, :240](HipKittens/kernels/attn/gqa_backwards/attn_bkwd_non_causal.cpp#L220-L240)).
4. **Four GEMMs per (Q, KV) tile** —
   `dVᵀ += dOᵢ_col · P_ij_bf16_col` ([:329-361](HipKittens/kernels/attn/gqa_backwards/attn_bkwd_non_causal.cpp#L329-L361)),
   `dKᵀ += Qᵢ_col · dP_ij_bf16_colᵀ` ([:373-400](HipKittens/kernels/attn/gqa_backwards/attn_bkwd_non_causal.cpp#L373-L400)),
   `dQᵢᵀ += Kⱼ_col · dP_ij_bf16_colᵀ` ([:405-427](HipKittens/kernels/attn/gqa_backwards/attn_bkwd_non_causal.cpp#L405-L427)).
5. **`D = dO ⊙ O` precomputed externally** —
   [:154](HipKittens/kernels/attn/gqa_backwards/attn_bkwd_non_causal.cpp#L154)
   — `delta_smem` loaded; used as `dS = P · (dP - D)` ([:335, :345](HipKittens/kernels/attn/gqa_backwards/attn_bkwd_non_causal.cpp#L335-L345)).
6. **Float accumulators for dK / dV** —
   [:129-130](HipKittens/kernels/attn/gqa_backwards/attn_bkwd_non_causal.cpp#L129-L130)
   `art<float, D, WARP_SIZE_KV, col_l, rt_32x32_s>` — preserve precision
   versus bf16 spill.
7. **Scaling constants** —
   [:63-64](HipKittens/kernels/attn/gqa_backwards/attn_bkwd_non_causal.cpp#L63-L64)
   `P_SCALE_FACTOR = 0.08838834 * 1.44269504` (= `1/√d · log₂(e)`);
   `dP_SCALE_FACTOR = 0.08838834` (= `1/√d`).
8. **Register-range pinning via `art::clobber`** —
   [:78-106](HipKittens/kernels/attn/gqa_backwards/attn_bkwd_non_causal.cpp#L78-L106)
   — explicit `a[112..127]`, `a[0..47] & v[56..71]` AGPR/VGPR ranges per
   tile. Same `art` machinery from §1.3.
9. **`swap_layout_inplace` for P → Pᵀ** —
   [:325, :370](HipKittens/kernels/attn/gqa_backwards/attn_bkwd_non_causal.cpp#L325-L370)
   — bf16 layout flip via pointer-pun (§2.3.11); zero data movement.
10. **dQ scale applied at write** —
    [:438-445](HipKittens/kernels/attn/gqa_backwards/attn_bkwd_non_causal.cpp#L438-L445)
    — `mul<0,0,0>(dQᵢᵀ, dQᵢᵀ, dP_SCALE_FACTOR)` deferred to the end.

### 5.4 `gqa_causal_backwards` — backward, causal

1. **Causal-skip loop bounds** —
   [attn_bkwd_causal.cpp:65-66](HipKittens/kernels/attn/gqa_causal_backwards/attn_bkwd_causal.cpp#L65-L66)
   — `first_step = max(0, (j_min * WARP_SIZE_KV) / STEP_QO)`; iterate
   only over Q steps overlapping K block.
2. **Diagonal-tile mask** —
   [:316-324](HipKittens/kernels/attn/gqa_causal_backwards/attn_bkwd_causal.cpp#L316-L324)
   — `if (q_pos == k_pos) make_causal<0,0,neg_inf_v>(P_ij)` then
   `mov<0, 1:4, neg_inf_v>` for the upper-triangle.
3. **Whole-block mask short-circuit** — `if (q_pos < k_pos) mov<neg_inf_v>(P_ij)`
   ([:316](HipKittens/kernels/attn/gqa_causal_backwards/attn_bkwd_causal.cpp#L316))
   — eliminate this row's contribution entirely.
4. **`__builtin_expect(q_start_pos < kv_end_pos, 0)`** —
   [:289, :499, :562, :624](HipKittens/kernels/attn/gqa_causal_backwards/attn_bkwd_causal.cpp#L289-L624)
   — same cold-branch hint as forward.
5. **Constexpr `neg_inf_v = 29`** plus VGPR init —
   [:125-128](HipKittens/kernels/attn/gqa_causal_backwards/attn_bkwd_causal.cpp#L125-L128)
   — single pre-load of `-inf` reused for every mask op.
6. **Otherwise identical to §5.3** — same dQ/dK/dV order, scales,
   register pinning, float accumulators.

### 5.5 Cross-cutting techniques (this slice)

- **T5.A — Online softmax with `rv_all_below(prev, cur, 8.0f)`**: only emit
  the rescale multiply when the running max has actually shifted by more
  than `8.0` in log₂. Skips the multiply in the common case where the row
  max is stable.
- **T5.B — Branch-free per-lane causal mask**: `v_cmp_lt_i32 + v_cndmask_b32`
  applied to packed values + `__builtin_expect` to make the mask path cold.
- **T5.C — Whole-tile elision before per-element mask**: causal forward
  computes `max_num_tiles = min(...)` so the inner loop *literally
  doesn't visit* fully-masked tiles. Same idea as [ck:§4.4.1]
  `GetTileRangeAlongX`.
- **T5.D — `neg_inf_v` parked in a VGPR**: one constexpr `v_mov_b32`
  primes the register; every mask op references it without re-emitting
  the constant.
- **T5.E — `swap_layout_inplace` is free**: pointer-pun + type-safe cast
  for bf16 layout flips in the backward (§2.3.11). Replaces what a
  CK_tile backward would do with `ds_read_tr*` shuffled LDS.
- **T5.F — `art::clobber` ranges pin AGPR/VGPR allocation**: the
  backward kernel reserves specific register ranges per tile so the
  three independent `dQ / dK / dV` accumulators don't fight for the same
  registers.
- **T5.G — D-precomputed-externally**: `D = Σ dO ⊙ O` runs in a separate
  preceding kernel (mirrors [ck:§14.1] `block_fmha_bwd_dot_do_o`).
- **T5.H — Float accumulators for `dK / dV`, scaled-at-write for `dQ`**:
  precision-vs-bandwidth trade-off — `dK/dV` accumulate many partial
  contributions and need fp32 headroom; `dQ` aggregates fewer and can
  scale on store.
- **T5.I — Same `s_setprio` / `sched_barrier` discipline as §4**:
  attention kernels reuse the GEMM-tuned wave-scheduler hints exactly.
- **T5.J — `pending_scale` deferred rescale**: avoids redundant scaling
  when the running max is stable for several blocks in a row; the
  per-block scaling commits only when the *next* block actually causes a
  rescale.

Cross-references: [ck:§4] (CK_tile FMHA — same online-softmax recipe,
different abstraction layer); [ck:§14.1] (CK FMHA backward with
`block_fmha_bwd_dot_do_o` precompute); [ck:§4.4] (CK causal masking).

<!-- SLICE-6-INSERT -->
## §6. LayerNorm + Rotary + torch_scaled

Three "small" kernel families that show how HipKittens composes its
primitives outside of the heavyweight GEMM / attention paths.

### 6.1 LayerNorm

1. **Single file** —
   [kernels/layernorm/kernel.cpp](HipKittens/kernels/layernorm/kernel.cpp)
   (~150 lines).
2. **Warp-per-token (1 warp, 64 threads)** —
   [kernel.cpp:80](HipKittens/kernels/layernorm/kernel.cpp#L80)
   — `dim3(NUM_THREADS) = 64`. Grid is `dim3(n_tile_size, B)`,
   `n_tile_size = N/4` so 4 tokens per launch tile
   ([:77](HipKittens/kernels/layernorm/kernel.cpp#L77)).
3. **`rv_naive<bf16, d_model>` for whole-row residency** —
   [:92](HipKittens/kernels/layernorm/kernel.cpp#L92)
   — full 2048-dim row lives in the register file; no LDS.
4. **Zero shared memory** —
   [:81](HipKittens/kernels/layernorm/kernel.cpp#L81)
   `dynamic_shared_memory() { return 0; }`.
5. **Inline reduction graph** — mean via `sum()`
   ([:112](HipKittens/kernels/layernorm/kernel.cpp#L112)),
   variance via subtract → square → sum
   ([:115-117](HipKittens/kernels/layernorm/kernel.cpp#L115-L117)),
   `1/√(var + 1e-5)` ([:119](HipKittens/kernels/layernorm/kernel.cpp#L119)).
   This is *not* Welford one-pass — it's a single-warp two-stage compute
   over registers.
6. **Fused residual + dropout** —
   [:98-109](HipKittens/kernels/layernorm/kernel.cpp#L98-L109)
   — dropout mask + rescale + residual add all in-register before the norm.
7. **bf16 throughout** — no upcast to fp32; arithmetic uses native bf16
   intrinsics.
8. **Two outputs** — `o_resid` (post-residual, for downstream) and `o`
   (post-norm) — [:109, :126](HipKittens/kernels/layernorm/kernel.cpp#L109-L126).

### 6.2 Rotary (RoPE)

1. **Single file** —
   [kernels/rotary/kernel.cpp](HipKittens/kernels/rotary/kernel.cpp)
   (~91 lines).
2. **Purely element-wise — no reduction**.
3. **`rt<bf16, 32, 128, row_l, rt_32x32_8_s>`** input/output —
   [:55](HipKittens/kernels/rotary/kernel.cpp#L55).
4. **`rt<bf16, 32, 64, ...>` for cos / sin (half-dim)** —
   [:56](HipKittens/kernels/rotary/kernel.cpp#L56).
5. **Cos / sin pre-loaded once before the main loop** —
   [:58-59](HipKittens/kernels/rotary/kernel.cpp#L58-L59).
6. **Half-dimension pair rotation** —
   [:64-72](HipKittens/kernels/rotary/kernel.cpp#L64-L72)
   — each iter processes `x[i]` and `x[i + half_dim_tiles]`; complex
   rotation `(x1·cos - x2·sin, x2·cos + x1·sin)` lowered to
   `__hsub2(__hmul2, __hmul2)` / `__hadd2(__hmul2, __hmul2)`.
7. **`#pragma unroll` on the inner pair loop** —
   [:63-66](HipKittens/kernels/rotary/kernel.cpp#L63-L66).
8. **No scaling on output** — RoPE is norm-preserving.

### 6.3 `torch_scaled` (scaled fp8 matmul)

1. **`scaled_matmul.cu`** —
   [kernels/torch_scaled/scaled_matmul.cu](HipKittens/kernels/torch_scaled/scaled_matmul.cu)
   (~575 lines); plus `profile_utils.cpp` (timing) and `utils.cpp` (fp8
   shared-tile loaders).
2. **Tile geometry** —
   [:22-24](HipKittens/kernels/torch_scaled/scaled_matmul.cu#L22-L24)
   — 256×256 output, 4×2 warp tiles, `BLOCK_K=128`, `k_iters = K/128`.
3. **Per-warp register tiles** —
   [:40-42](HipKittens/kernels/torch_scaled/scaled_matmul.cu#L40-L42)
   — `RT_A = 64×128`, `RT_B = 32×128`, `RT_C = 64×32` (`rt_fl<… col_l,
   rt_16x16_s>` accumulator).
4. **fp8e4m3 input + fp32 accumulator + fp32 output** — 4× memory-traffic
   savings vs fp16 inputs with no accumulation precision loss.
5. **Double-buffered LDS** —
   [:37-38, :65](HipKittens/kernels/torch_scaled/scaled_matmul.cu#L37-L65)
   — `As[2][2]` / `Bs[2][2]` ping-pong via tic/toc toggle.
6. **`subtile_inplace<…>(shared)`** for in-flight subtile views —
   [:109-112, :124-125](HipKittens/kernels/torch_scaled/scaled_matmul.cu#L109-L125)
   — same primitive as §1.4.
7. **`__builtin_amdgcn_s_setprio()` around MMA** —
   [:118, :130, :140, :151](HipKittens/kernels/torch_scaled/scaled_matmul.cu#L118-L151).
8. **`#pragma unroll 2` outer K loop** —
   [:106](HipKittens/kernels/torch_scaled/scaled_matmul.cu#L106)
   — software pipeline: load K+1 while computing K.
9. **Per-row × per-col scale fused into accumulator** —
   [:254-265](HipKittens/kernels/torch_scaled/scaled_matmul.cu#L254-L265)
   — `mul_col(cA, cA, scale_b0_rv); mul_row(cA, cA, scale_a0_rv);`
   replicated for `cA / cB / cC / cD`.
10. **Direct fp32 store** —
    [:271-274](HipKittens/kernels/torch_scaled/scaled_matmul.cu#L271-L274)
    — output already scaled in-register; no quantization round-trip.

### 6.4 Cross-cutting techniques (this slice)

- **T6.A — Single-warp register-resident reductions** (layernorm): when
  the row fits in VGPRs, no LDS / cross-warp sync is needed — the whole
  reduction chain is in-register.
- **T6.B — RoPE is two `__hmul2` + one `__hadd2`/`__hsub2` per pair**:
  native bf16 packed math; no upcast. The pre-loaded cos/sin tile is
  the only "memory" the kernel touches per iter.
- **T6.C — Two-output kernels (LayerNorm)**: emitting both the residual
  and the normalized output in one pass saves a follow-up kernel.
- **T6.D — `subtile_inplace` view zero-copy**: the scaled matmul reuses
  the §1.4 primitive to slice shared tiles without LDS round-trips.
- **T6.E — `mul_col` + `mul_row` apply per-axis scales without an extra
  pass**: in-register element-wise after the reduction, before the store.
- **T6.F — `s_setprio` + sched_barrier discipline is universal**: every
  kernel (GEMM, attention, scaled-matmul) uses the same wave-scheduler
  hint pattern.
- **T6.G — Single-pass mean+var (not Welford)**: LayerNorm's two
  reductions (sum, then subtract+square+sum) are *cheaper than Welford*
  on a 64-thread warp because the data fits in registers and there's no
  numerical-stability issue at d=2048.

Cross-references: [ck:§6] (CK reduce / softmax / norm — the
multi-warp variant of what HipKittens does single-warp);
[ck:§5.3] / [ck:§16.11] (MX scale handling — torch_scaled uses
explicit pre-/post-multiply rather than OpSel).

<!-- SLICE-7-INSERT -->
## §7. Distributed kernels (Iris + bf16_gemm)

The distributed-kernels directory wires HipKittens to **Iris** (a ROCm
distributed-memory abstraction over MPI) and provides one example —
row-partitioned bf16 GEMM.

### 7.1 Iris integration

1. **MPI init wrapper** —
   [iris_py.cpp:54-57](HipKittens/distributed-kernels/iris_py.cpp#L54-L57)
   — `iris::mpi::initialize()` returns `(rank, world_size)`; one rank per GPU.
2. **Distributed heap allocator** —
   [iris_py.cpp:72-95](HipKittens/distributed-kernels/iris_py.cpp#L72-L95)
   `iris.empty([dims…], dtype)` returns per-rank local allocations; each
   rank has its own slice of the cluster heap.
3. **`iris_device_view` passed to kernels** —
   [iris_py.cpp:104-106](HipKittens/distributed-kernels/iris_py.cpp#L104-L106),
   [bf16_gemm/kernel.cpp:75](HipKittens/distributed-kernels/bf16_gemm/kernel.cpp#L75)
   — exposes `cur_rank()` / `world_size()` to device code.
4. **`iris.barrier()` collective** —
   [iris_py.cpp:99-101](HipKittens/distributed-kernels/iris_py.cpp#L99-L101)
   — MPI barrier; used to keep ranks aligned between phases.
5. **`IrisTensor` wrapper** —
   [iris_py.cpp:11-43](HipKittens/distributed-kernels/iris_py.cpp#L11-L43)
   — device-ptr + shape + dtype + lifetime back-ref to `iris_ctx`;
   destructor deallocates.
6. **`__cuda_array_interface__` zero-copy to PyTorch** —
   [example.py:28-56](HipKittens/distributed-kernels/bf16_gemm/example.py#L28-L56)
   — interface dict wraps the Iris GPU pointer; PyTorch consumes it without
   host round-trip.

### 7.2 Distributed bf16_gemm

1. **Row partitioning** —
   [kernel.cpp:77-80](HipKittens/distributed-kernels/bf16_gemm/kernel.cpp#L77-L80),
   [example.py:70-72](HipKittens/distributed-kernels/bf16_gemm/example.py#L70-L72)
   — `M_local = M / world_size`; rank owns
   `A[rank*M_local : (rank+1)*M_local, :]` and the corresponding C slab.
   B is *replicated* on every rank.
2. **No all-reduce** — each rank produces a disjoint slab of C, so the
   distributed layer is *partitioning*, not *reduction*. Communication
   cost is the B-broadcast (done outside the kernel).
3. **Producer / consumer warps** —
   [kernel.cpp:15-17, :121-126](HipKittens/distributed-kernels/bf16_gemm/kernel.cpp#L15-L126)
   — 4 producer warps + 8 consumer warps per block (64 threads).
   Producers do global → LDS; consumers do LDS → register + MFMA + store.
4. **Double-buffered tic/toc LDS** —
   [kernel.cpp:138-210](HipKittens/distributed-kernels/bf16_gemm/kernel.cpp#L138-L210)
   — `As[2][M_BLOCK][2]` / `Bs[2][N_BLOCK][2]`; tic/toc flipped per K-tile.
5. **`__builtin_amdgcn_sched_barrier()` + `s_barrier()` between producer
   and consumer** —
   [kernel.cpp:206-207](HipKittens/distributed-kernels/bf16_gemm/kernel.cpp#L206-L207)
   — wave-level sync without a full `__syncthreads()`.
6. **`mma_ABt()` unrolled 4×** —
   [kernel.cpp:188-204](HipKittens/distributed-kernels/bf16_gemm/kernel.cpp#L188-L204)
   — 2 A-subtiles × 2 B-subtiles per K-iter; interleaved with `s_setprio`.
7. **Iris-aware store** —
   [kernel.cpp:24-70, :239-244](HipKittens/distributed-kernels/bf16_gemm/kernel.cpp#L24-L244)
   — custom `kittens_store()` calls `iris_ctx.store()` per pair; writes land
   on the rank's local GPU memory (no cross-die traffic).
8. **Chiplet-aware swizzle** —
   [kernel.cpp:107-119](HipKittens/distributed-kernels/bf16_gemm/kernel.cpp#L107-L119)
   — workgroup → tile remapping to keep an XCD's hot working set in its own
   L2 (the MI300X 8-XCD topology that CK doesn't address — §9 below).

### 7.3 Python entry + build

1. **`iris_py.cpp` as standalone pybind11 module** —
   [iris_py.cpp:109-155](HipKittens/distributed-kernels/iris_py.cpp#L109-L155),
   [CMakeLists.txt:184-194](HipKittens/distributed-kernels/CMakeLists.txt#L184-L194)
   — produces `iris_py.*.so`; exports `Iris.{empty, barrier,
   get_device_view, cur_rank, world_size}` and free
   `mpi_init / mpi_finalize`.
2. **Per-kernel `tk_kernel` pybind module** —
   [kernel.cpp:253-265](HipKittens/distributed-kernels/bf16_gemm/kernel.cpp#L253-L265),
   [CMakeLists.txt:159](HipKittens/distributed-kernels/CMakeLists.txt#L159)
   — `dispatch_micro()` Python entry takes PyTorch tensors + `iris_device_view`
   + `M_local / N / K / row_offset`.
3. **CPM-fetched Iris** —
   [CMakeLists.txt:75-90](HipKittens/distributed-kernels/CMakeLists.txt#L75-L90)
   — downloaded from a ROCm GitHub fork; benchmarks/tests disabled in
   build.
4. **`GPU_TARGET=CDNA4`** —
   [CMakeLists.txt:39-51](HipKittens/distributed-kernels/CMakeLists.txt#L39-L51)
   — maps to `gfx950 + KITTENS_CDNA4 + warp-sync-builtins + fast-math`.
5. **`-DDK_BUILD=bf16_gemm` selects kernel target** —
   [README.md:37-39](HipKittens/distributed-kernels/README.md#L37-L39).
6. **Auto-discovery of `*/kernel.cpp`** —
   [CMakeLists.txt:132-179](HipKittens/distributed-kernels/CMakeLists.txt#L132-L179)
   — each kernel becomes its own `tk_kernel.*.so`.

### 7.4 Cross-cutting techniques (this slice)

- **T7.A — Distributed by partition, not reduction**: bf16_gemm parallelizes
  by *splitting the output across ranks*, so the kernel itself is
  rank-oblivious except for `cur_rank()` reads. The communication cost
  is fully outside the GPU kernel.
- **T7.B — Producer / consumer warp split**: 4 producers + 8 consumers
  per block is a clean way to overlap global → LDS with LDS → MFMA without
  needing async DMA instructions.
- **T7.C — `iris_ctx.store()` colors writes by rank**: lets the runtime
  route the store to the rank's GPU rather than going through an MPI
  collective — same effect as a static sharding but without explicit
  send/recv.
- **T7.D — Chiplet-aware workgroup swizzle**: this is the MI300X
  "8-XCD-aware" technique that CK doesn't have ([ck:§16.14] lists it as
  the top opportunity); HipKittens *does* have it, here.
- **T7.E — Single `.so` per kernel via pybind11 + hipcc**: the build
  system compiles HIP device code and pybind binding together so Python
  just does `import tk_kernel`.
- **T7.F — Zero-copy GPU pointer to PyTorch via
  `__cuda_array_interface__`**: avoids a host round-trip when handing the
  Iris tensor back to the framework.

Cross-references: [ck:§15.2] (StreamK — the closest CK analog of producer/
consumer overlap); [ck:§16.11.6] / [ck:§16.14.1] (MI300X XCD scheduling —
HipKittens has it via the workgroup swizzle in §7.2.8).

<!-- SLICE-8-INSERT -->
## §8. Training kernels (bert + llama)

The `training/` directory wraps the §5 attention kernels into PyTorch
training loops. Two model families: BERT (encoder-only) and LLaMA
(decoder-only with causal attention).

### 8.1 `training/bert/`

1. **Three attention swap targets** —
   [bert/models/](HipKittens/training/bert/models/)
   has `base.py` (PyTorch reference), `aiter.py` (AITER kernel), and
   `hipkittens.py` (HipKittens kernels). User picks via task config.
2. **`HipKittensFlashAttnFn` autograd wrapper** —
   [bert/models/hipkittens.py](HipKittens/training/bert/models/hipkittens.py)
   — forward saves `Q, K, V, O, L`; backward dispatches three kernels.
3. **Three-stage backward** —
   `dispatch_prep` (`D = dO ⊙ O` reduction; same as [ck:§14.1.5]),
   `dispatch_bkwd_combined` (fused dQ/dK/dV computation reusing L from
   forward), `dispatch_dq_shuffle` (BHND → BNHD layout swap).
4. **dQ pre-shuffle layout** —
   [hipkittens.py:281](HipKittens/training/bert/models/hipkittens.py#L281)
   — backward keeps dQ in BHND for register efficiency, swaps to BNHD on
   the final store. Saves cross-stage data movement.
5. **GQA expand in forward only** —
   [hipkittens.py:380-382](HipKittens/training/bert/models/hipkittens.py#L380-L382)
   — Q heads `H`, KV heads `H_KV`; HipKittens projects K/V at `H_KV*D`
   and keeps them un-expanded through the kernel.
6. **bf16 compute + fp32 LSE** —
   [hipkittens.py:58-66](HipKittens/training/bert/models/hipkittens.py#L58-L66)
   — same numerical pattern as the standalone kernels in §5.
7. **Training task** —
   [bert/tasks.py:194](HipKittens/training/bert/tasks.py#L194)
   — swaps HuggingFace `BertSelfAttention` with `HipKittensBertSelfAttention`;
   AdamW + gradient clip + linear warmup.

### 8.2 `training/llama/`

1. **PyTorch Lightning stack** —
   [llama/setup.py](HipKittens/training/llama/setup.py)
   pulls `pytorch-lightning` + `hydra`.
2. **C++ source kernels in `csrc/`** —
   [llama/csrc/](HipKittens/training/llama/csrc/)
   — `attn_fwd_causal.cpp` (forward), `attn_bkwd_causal_HBN.cpp` and
   `attn_bkwd_causal_HNB.cpp` (two backward layouts), `attn_bkwd_prep.cpp`
   (the `D = dO ⊙ O` precompute).
3. **`HipAttnFunction` mirrors BERT pattern** —
   [llama/models/attentions/hipkittens.py:110-112](HipKittens/training/llama/llama/models/attentions/hipkittens.py#L110-L112)
   — same three-stage backward dispatch.
4. **Triton ops alongside HIP kernels** —
   [llama/llama/ops/triton/layer_norm.py:19-72](HipKittens/training/llama/llama/ops/triton/layer_norm.py#L19-L72),
   `rotary.py:21-79`, `cross_entropy.py`. LN+dropout+residual fused in
   the *Triton* reference, not yet in a HipKittens kernel.
5. **Pre-norm block** —
   [llama/llama/models/](HipKittens/training/llama/llama/models/)
   `block.py` line 99 — dropout → residual-add → layer-norm chain.
6. **Cross-entropy with label smoothing + z-loss + in-place backward** —
   `cross_entropy.py:6-64` — vocab-parallel-TP friendly via
   `process_group`; in-place backward overwrites logits to save memory.
7. **Loss-scale + grad-norm monitor callbacks** —
   `train/callbacks/loss_scale_monitor.py:13-14`,
   `train/callbacks/norm_monitor.py:19` — Lightning hooks at
   `on_before_optimizer_step` for AMP scale tracking.
8. **`dispatch_<…>` PYBIND11 binding** — kernels exported as
   compile-time-fixed-shape entry points (`ATTN_D`, `ATTN_H`,
   `ATTN_H_KV` via `#ifndef` guards in `.cpp`). No runtime shape
   polymorphism.

### 8.3 Cross-cutting techniques (this slice)

- **T8.A — `torch.autograd.Function` is the integration surface**: both
  BERT and LLaMA use the same wrap-forward-save-tensors-for-bwd pattern.
  No JIT compilation at run time — kernels are pre-compiled once via
  the C++/pybind11 extension build.
- **T8.B — Three-stage backward (prep + combined + shuffle) per model**:
  the `D` precompute (matches [ck:§14.1.5]), the fused dQ/dK/dV, and the
  final layout shuffle each get their own kernel — the boundaries
  exist because each stage has a different optimal register / LDS
  layout.
- **T8.C — GQA "don't expand K/V"**: training keeps Q at `H*D` and K/V at
  `H_KV*D`; the kernel internally broadcasts. Avoids materializing
  `H*D` K/V tensors in DRAM.
- **T8.D — bf16 compute / fp32 LSE / fp32 dK,dV / scaled-at-write dQ**:
  the precision pattern from §5.3 carries over unchanged into training.
- **T8.E — Compile-time-fixed shapes via `#ifndef`**: every kernel
  template is monomorphized at C++ compile time on `(ATTN_D, ATTN_H,
  ATTN_H_KV)`. There is no run-time shape dispatch — adding a new
  config requires a recompile.
- **T8.F — In-place cross-entropy backward**: standard Triton CE kernel
  reused; logits are overwritten with gradients, avoiding a separate
  buffer.
- **T8.G — Loss-scale & grad-norm callbacks**: monitoring lives in
  Lightning callbacks, not in the kernel — keeps the HIP layer pure.

Cross-references: [ck:§14] (CK FMHA backward — same `D` precompute +
fused dQ/dK/dV story); [ck:§11.4] (`Dockerfile.aiter` — HipKittens has
no equivalent build container, relies on the user's ROCm + Lightning
install).

<!-- SLICE-9-INSERT -->
## §9. MI300 (gfx942) + MI355 (gfx950 / CDNA4) specifics

HipKittens uses `KITTENS_CDNA3 / KITTENS_CDNA4` (not raw `__gfx94*__`
macros) to gate per-arch features. The headline difference: gfx950 adds
FP8 MFMA, 32×32 MFMA shapes, warp-sync builtins, and 8-die XCD topology
*that HipKittens actually uses* (unlike CK [ck:§16.11.6]).

### 9.1 MI300 / MI325X (gfx942, CDNA3) features HK uses

1. **`KITTENS_CDNA3 --offload-arch=gfx942`** —
   [analysis/bf16_gemm/mi325x/Makefile:17-18](HipKittens/analysis/bf16_gemm/mi325x/Makefile#L17-L18).
2. **`v_mfma_f32_16x16x32_bf16` / `v_mfma_f32_32x32x16_bf16`** —
   [common/macros.cuh:575-690](HipKittens/include/common/macros.cuh#L575-L690)
   — bf16 MFMA across multiple register-allocation overloads.
3. **`mfma_f32_16x16x32_bf16_zero_accum`** —
   [common/macros.cuh:762-800](HipKittens/include/common/macros.cuh#L762-L800)
   — explicit zero-init accumulator variant; avoids the carry-in
   dependency in epilogues.
4. **No `mfma_f32_*_fp8_*` on gfx942** —
   [register/tile/assembly/mma.cuh:27-34](HipKittens/include/ops/warp/register/tile/assembly/mma.cuh#L27-L34)
   — the fp8 MFMA is conditionally compiled and only fires on gfx950.
   gfx942 ships without it in HK.
5. **160 KB LDS / 256 CU assumed** —
   [kernels/gemm/bf16fp32/README.md:30](HipKittens/kernels/gemm/bf16fp32/README.md#L30)
   — matches gfx942 spec.

### 9.2 MI355 / MI350X (gfx950, CDNA4) features HK uses

1. **`KITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS
   -ffast-math`** —
   [analysis/bf16_gemm/mi350x/Makefile:18-19](HipKittens/analysis/bf16_gemm/mi350x/Makefile#L18-L19).
2. **`mfma_f32_16x16x32_fp8_fp8` with 16 register-allocation overloads** —
   [common/macros.cuh:693-759](HipKittens/include/common/macros.cuh#L693-L759)
   — every (A, B, C, D) ∈ {AGPR, VGPR} combination is wired up.
3. **fp8 path gated by input type** —
   [register/tile/assembly/mma.cuh:29](HipKittens/include/ops/warp/register/tile/assembly/mma.cuh#L29)
   — `if (InputType == fp8e4m3) call mfma_f32_16x16x32_fp8_fp8` —
   executes only on gfx950.
4. **`mfma_f32_32x32x16_bf16` for `rt_32x32`** —
   [register/tile/assembly/mma.cuh:75-80](HipKittens/include/ops/warp/register/tile/assembly/mma.cuh#L75-L80)
   — gfx950 only; gfx942 lacks the 32×32 form in HK.
5. **`HIP_ENABLE_WARP_SYNC_BUILTINS` flag** — gfx950-specific
   warp-level synchronization intrinsics used by attention kernels (§5).
6. **MXFP8 agent-optimized GEMM is MI355X-only** —
   [agent-mxfp8-gemm-optimization.md:16, :18-20](HipKittens/agent-mxfp8-gemm-optimization.md#L16-L20).
7. **HBM3E + 5.2 TB/s bandwidth assumption** —
   [kernels/gemm/bf16fp32/README.md:15-42](HipKittens/kernels/gemm/bf16fp32/README.md#L15-L42).
8. **LDS-port contention discovery (gfx950)** —
   [agent-mxfp8-gemm-optimization.md:123](HipKittens/agent-mxfp8-gemm-optimization.md#L123)
   — *"`buffer_load_lds` and `ds_read` share LDS port, can't run
   together"*: a CDNA4 hardware-port finding documented by experiment.

### 9.3 XCD / chiplet-aware grid scheduling

1. **`NUM_XCDS = 8`** —
   [analysis/paper_experiments/grid_micro/kernel_8192_w*.cpp:52](HipKittens/analysis/paper_experiments/grid_micro/)
   — 8-die topology shared between MI300X and MI350X.
2. **`CUS_PER_XCD = 32`** —
   [grid_micro/kernel_8192_w*.cpp:53](HipKittens/analysis/paper_experiments/grid_micro/)
   — yields 256 CUs total; matches both gfx942 and gfx950 (internal XCD
   structure differs but external count matches).
3. **`chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM*WGM)`** —
   [analysis/bf16_gemm/mi325x/kernel_2048.cpp:50 etc.](HipKittens/analysis/bf16_gemm/mi325x/)
   — remap workgroup IDs so a `WGM*WGM` tile of workgroups (typically 4×4
   or 8×8) lands in the *same* XCD. The L2 stays die-local.
4. **Adaptive Tile Grouping (WGM tuning)** —
   [agent-mxfp8-gemm-optimization.md:109-111](HipKittens/agent-mxfp8-gemm-optimization.md#L109-L111)
   — `WGM=8` (16K matrices), `WGM=4` (small matrices); +10.4 % @ 16K via
   B-matrix L2 reuse.
5. **Distributed bf16_gemm also swizzles by XCD** —
   [distributed-kernels/bf16_gemm/kernel.cpp:107-119](HipKittens/distributed-kernels/bf16_gemm/kernel.cpp#L107-L119)
   — same `chiplet_transform_*` family in the multi-GPU kernel (§7.2.8).
6. **This is exactly what CK is missing** — [ck:§16.11.6] /
   [ck:§16.14.1] flag XCC-aware scheduling as the *single biggest*
   untapped MI300X perf opportunity. HipKittens has it as a default
   pattern.

### 9.4 Build / Makefile arch flags

1. **No `__gfx940__` / `__gfx941__` usage** — HipKittens does not target
   the APU-only variants; MI300X is gfx942 only.
2. **CDNA3 flag set** —
   [analysis/bf16_gemm/mi325x/Makefile:17-18](HipKittens/analysis/bf16_gemm/mi325x/Makefile#L17-L18)
   — `KITTENS_CDNA3 --offload-arch=gfx942`. Used by MI325X kernels +
   `fp6_dwordx{3,4}/analysis/gemm/mi325x/`.
3. **CDNA4 flag set** —
   [analysis/bf16_gemm/mi350x/Makefile:18-19](HipKittens/analysis/bf16_gemm/mi350x/Makefile#L18-L19)
   — `KITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS
   -ffast-math`. Used by `analysis/attn/`, `analysis/bf16_gemm/mi350x/`,
   distributed kernels.
4. **`-ffast-math` on gfx950 only** — fast-math is opted into for
   CDNA4 + the warp-sync builtins.
5. **`HIP_ENABLE_WARP_SYNC_BUILTINS` is a `-D` macro, not arch-implicit**
   — even on gfx950 the user must define it; HK opts in via the
   Makefile.

### 9.5 Benchmarks from `agent-mxfp8-gemm-optimization.md` + READMEs

1. **MXFP8 GEMM on MI355X** —
   [agent-mxfp8-gemm-optimization.md:30-34](HipKittens/agent-mxfp8-gemm-optimization.md#L30-L34):
   - 4 K: **2531 TFLOPS** (100.2 % vs hipBLASLt FP8 tensorwise; 169.7 %
     vs hipBLASLt MXFP8).
   - 8 K: **2991 TFLOPS** (93.6 % vs FP8; 197.8 % vs MXFP8); ~96.3 % of
     cuBLASLt B200 MXFP8 = 3105 TFLOPS.
   - 16 K: **3029 TFLOPS** (92.9 % vs FP8; 246.1 % vs MXFP8).
2. **Primus-Turbo end-to-end (quant + GEMM on MI355X)** —
   [agent-mxfp8-gemm-optimization.md:44-50](HipKittens/agent-mxfp8-gemm-optimization.md#L44-L50):
   forward 1795 TFLOPS (59 % faster than hipBLASLt, ~90 % B200 TE);
   backward 1977 TFLOPS (68 % faster, ~96 % B200 TE).
3. **Attention forward / backward graphs** —
   [analysis/attn/fwd/README.md](HipKittens/analysis/attn/fwd/README.md),
   [analysis/attn/bkwd/README.md](HipKittens/analysis/attn/bkwd/README.md)
   — both MI355X graphs cover MHA (16 heads, D=128) and GQA (64 Q, 8 KV,
   D=128), causal + non-causal.
4. **MI325X has separate benchmark dirs** — both `analysis/bf16_gemm/`
   and `fp6_dwordx{3,4}/analysis/gemm/` carry parallel mi325x/ and
   mi350x/ subdirs, so the comparison is apples-to-apples.

### 9.6 Cross-cutting techniques (this slice)

- **T9.A — `KITTENS_CDNA3 / CDNA4` macros instead of `__gfx94*__`**:
  a higher-level toggle than CK's raw arch checks; lets the same source
  compile against either CDNA generation by changing one `-D` flag.
- **T9.B — fp8 MFMA gated by input type, not by arch macro**: HK's
  template specializes on the dtype; the gfx942-vs-gfx950 split happens
  implicitly via "does the instruction exist on this target."
- **T9.C — `mfma_*_zero_accum` epilogue variant**: gfx942 introduces an
  explicit zero-init MFMA that drops the carry-in dependency at the start
  of the epilogue. HK exposes it; CK does not specifically wire one up.
- **T9.D — XCD-aware grid swizzle via `chiplet_transform_chunked`**: the
  single most important MI300X / MI355X perf technique HK has and CK
  doesn't. Every shipping HK kernel that targets MI300/MI355 uses it.
- **T9.E — `WGM` (workgroup-mapping) is the tunable per shape**: small
  matrices want small `WGM`; 16K matrices want `WGM=8` for B-matrix L2
  reuse — agent walk +10.4 % at 16K.
- **T9.F — LDS-port contention is *experimentally* known on gfx950**:
  documented in the agent doc as a failed-attempt entry; constrains
  future kernel design (can't co-issue `buffer_load_lds` and `ds_read`).
- **T9.G — gfx950 wants `-ffast-math` + warp-sync builtins**: both are
  opt-in via Makefile macros. The fast-math is required for the
  flash-attention-style log-sum-exp paths to vectorize cleanly.
- **T9.H — No `__gfx940__` / `__gfx941__` paths**: HipKittens targets
  MI300X (gfx942) explicitly; the APU variants are out of scope.

Cross-references: [ck:§16.11] (CK's MI300 features — HK's are similar
plus the XCD swizzle); [ck:§16.12] (CK's MI355 features — HK uses
`mfma_f32_*_fp8_fp8` and `mfma_f32_32x32x*` that CK has but doesn't
necessarily wire into ops); [ck:§16.14] (top forward-looking MI300X
opportunities — HK already has the XCD swizzle on the list).

<!-- SLICE-10-INSERT -->
## §10. Perf-engineering catalog + reading order + HK ↔ CK comparison

Tags every technique mentioned in §1–§9 so search and cross-reference is
trivial.

### 10.1 Technique index (one-line catalog)

| Tag | Technique | §  |
|---|---|---|
| T1.A | Compile-time shape algebra | §1.1 |
| T1.B | Packing factor → register density | §1.2 |
| T1.C | Layout inverts `row_vec` / `col_vec` | §1.2 |
| T1.D | LDS swizzle as XOR mask | §1.4 |
| T1.E | `art` explicit register-range allocator | §1.3 |
| T1.F | `st_subtile` zero-copy view | §1.4 |
| T1.G | Bump-pointer LDS allocator | §1.4 |
| T1.H | Compile-or-runtime dims in `gl` | §1.5 |
| T2.A | `BUFFER_LOAD_LDS` + M0 + matching swizzle | §2.1 |
| T2.B | Pre-fill swizzled offsets | §2.1 |
| T2.C | `ds_read_b64_tr_b16` free transpose | §2.4 |
| T2.D | Back-to-back MFMA without `s_waitcnt` | §2.3 |
| T2.E | `permlane16/32_swap` zero-cycle transpose | §2.3 |
| T2.F | Compile-time vector-width dispatch | §2.1 |
| T2.G | `address_space(3)` typed-LDS pointer | §2.1 |
| T2.H | No `s_waitcnt lgkmcnt` between DMAs | §2.2 |
| T2.I | gfx950 scaled MFMA exposed at warp scope | §2.3 |
| T3.A | Group ops = warp ops with `GROUP_THREADS` | §3.1 |
| T3.B | Disjoint per-warp LDS regions = no barrier | §3.3 |
| T3.C | `prefill_swizzled_offsets` at group scope | §3.2 |
| T3.D | `warpid % num_warps` silently drops idle warps | §3.2 |
| T3.E | Group owns no barrier (kernel does) | §3.3 |
| T3.F | `warpgroup = group<4>` is naming, not HW | §3.1 |
| T4.A | Tic/toc LDS double-buffer + reg prefetch | §4.5 |
| T4.B | `s_waitcnt vmcnt(6)` magic number | §4.4 |
| T4.C | `s_setprio` alternation favors MFMA | §4.1 |
| T4.D | SGPR-precomputed addresses (soff / M0) | §4.4 |
| T4.E | SpreadLDG (spread loads across compute) | §4.4 |
| T4.F | Adaptive WGM for B-matrix L2 reuse | §4.4 |
| T4.G | dwordx4 > dwordx3 (LDS-port-bound) | §4.3 |
| T4.H | `buffer_load_lds` + `ds_read` share LDS port | §4.4 |
| T4.I | At occupancy=1, `s_setprio` is a no-op | §4.4 |
| T4.J | Let the compiler schedule (hoist constants) | §4.4 |
| T5.A | `rv_all_below(prev, cur, 8.0f)` rescale gate | §5.1 |
| T5.B | Branch-free per-lane causal mask | §5.2 |
| T5.C | Whole-tile elision via `max_num_tiles` | §5.2 |
| T5.D | `neg_inf_v` parked in a VGPR constexpr | §5.2 |
| T5.E | `swap_layout_inplace` zero data movement | §5.3 |
| T5.F | `art::clobber` register-range pinning | §5.3 |
| T5.G | `D = dO ⊙ O` precomputed externally | §5.3 |
| T5.H | Float `dK/dV` + scaled-at-write `dQ` | §5.3 |
| T5.I | Same `s_setprio` discipline as GEMM | §5.1 |
| T5.J | `pending_scale` deferred rescale | §5.1 |
| T6.A | Single-warp register-resident reductions | §6.1 |
| T6.B | RoPE = `__hmul2` + `__hadd2` / `__hsub2` | §6.2 |
| T6.C | Two-output kernels (residual + norm) | §6.1 |
| T6.D | `subtile_inplace` view zero-copy | §6.3 |
| T6.E | `mul_col` + `mul_row` per-axis scaling | §6.3 |
| T6.F | Universal `s_setprio` + sched_barrier | §6.4 |
| T6.G | Two-pass mean/var (not Welford) | §6.4 |
| T7.A | Distributed by partition, not reduction | §7.4 |
| T7.B | Producer / consumer warp split | §7.2 |
| T7.C | `iris_ctx.store()` colors writes by rank | §7.4 |
| T7.D | Chiplet-aware workgroup swizzle | §7.2 |
| T7.E | Single `.so` per kernel via pybind11 | §7.3 |
| T7.F | Zero-copy GPU ptr via `__cuda_array_interface__` | §7.4 |
| T8.A | `torch.autograd.Function` integration | §8.3 |
| T8.B | Three-stage backward (prep + combined + shuffle) | §8.3 |
| T8.C | GQA "don't expand K/V" | §8.3 |
| T8.D | bf16 / fp32 LSE / fp32 dK,dV / scaled-at-write dQ | §8.3 |
| T8.E | Compile-time-fixed shapes via `#ifndef` | §8.3 |
| T8.F | In-place cross-entropy backward | §8.2 |
| T8.G | Loss-scale / grad-norm Lightning callbacks | §8.2 |
| T9.A | `KITTENS_CDNA3 / CDNA4` macros | §9.1 |
| T9.B | fp8 MFMA gated by input type, not arch | §9.2 |
| T9.C | `mfma_*_zero_accum` epilogue variant | §9.1 |
| T9.D | `chiplet_transform_chunked` XCD swizzle | §9.3 |
| T9.E | `WGM` tunable per shape | §9.3 |
| T9.F | LDS-port contention experimentally known | §9.2 |
| T9.G | gfx950 wants `-ffast-math` + warp-sync builtins | §9.4 |
| T9.H | gfx940 / gfx941 (APUs) not targeted | §9.4 |

### 10.2 Reading order by use case

**You want to read a HipKittens kernel from scratch**:
§1 (core types) → §2 (warp ops) → §3 (group ops) — then jump to §4 (GEMM)
or §5 (attention) depending on which you need.

**You want to *modify* a HipKittens kernel**:
§1.4 (LDS swizzle) → §4.5 (cross-cutting GEMM techniques) → §9 (per-arch
flags). The tile shapes and warp counts are usually the first knobs you
turn.

**You want to find an optimization opportunity**:
§4.4 (agent walk: which techniques *worked* and which *failed*) → §9.3
(XCD swizzle) → §10.3 (HK vs CK gap analysis below).

**You want to integrate HipKittens into PyTorch**:
§8 (training kernel wrapping pattern) → §7 (distributed) for multi-GPU.

**You're chasing a numerical bug**:
§5.1 (online softmax recipe) → §5.3 (backward gradient algebra) →
§6.1 (layernorm reduction).

**You're porting from CUDA ThunderKittens to HipKittens**:
§1 (types are isomorphic, but Wave64 vs Wave32) → §2.3 (MFMA replaces
WGMMA) → §9 (per-arch macros differ).

### 10.3 HipKittens vs Composable Kernel — concrete differences

Reading both playbooks side-by-side makes the design philosophy stark.

| Dimension | HipKittens | Composable Kernel |
|---|---|---|
| Posture | Research-grade primitives + hand-tuned kernels | Production library + 1981-instance codegen archive |
| Abstraction | Tile types in C++ headers; one `.cpp` per kernel | Tile types + pipeline policy + grid map + epilogue + dispatcher |
| Codegen | None — `.cpp` files are hand-written | Python codegen produces hundreds of instances per dtype |
| Build artifact | One `.so` per kernel via pybind11 | One large `libck.so` with 1981 instance entry points |
| Synchronization | Implicit via SGPR-uniform M0 + disjoint LDS regions | Explicit `block_sync_lds` / `s_waitcnt` discipline |
| MFMA driver | Direct inline-asm + builtin call from kernel | Wrapped in `XdlopsGemm` / pipeline policy |
| Backward attention | Hand-written `_combined` kernel | Four-way selector + `dot_do_o` precompute + `convert_dq` |
| MI300X 8-XCD topology | **Always swizzled** via `chiplet_transform_chunked` | **Absent**; documented as top opportunity ([ck:§16.14.1]) |
| `ds_read_tr*` free transpose | **Used** on every col-major LDS → register load | **Declared but unused** ([ck:§16.12.5]) |
| FP6 experiments | `fp6_dwordx{3,4}` forks of the whole library | No FP6 path |
| `s_waitcnt vmcnt(6)` | Documented as magic number from experiment | Pipeline-derived; opaque |
| `__launch_bounds__` | Implicit via dim3 launch — kernels are 64×N-thread | Explicit `(blockSize, MinBlockPerCu)` per kernel |
| Iris multi-GPU | First-class; rank-aware kernels | Out of scope; CK is single-device |
| Coherence enum usage | Only `coherency::cache_all` used in DMA path | 28-value `amd_buffer_coherence_enum`, mostly unused |

### 10.4 Six always-true rules (HipKittens edition)

1. **Tile types in C++ headers; everything is compile-time**. Shape /
   layout / packing / register-range are template parameters; the
   compiler folds all index math.
2. **Wave64-only**. `WARP_THREADS = 64`; no Wave32 alternate path.
   Group ops scale this to `N * 64`.
3. **LDS swizzle is an XOR mask in the byte offset**, not a re-layout.
   Loads and stores both XOR; the data is never physically rearranged.
4. **MFMA, `ds_read_tr*`, `permlane*` are the per-instruction perf
   levers**. HipKittens wires up *all three* on gfx950 — CK is still
   catching up on `ds_read_tr`.
5. **XCD-aware workgroup swizzle is on by default**. Every MI300/MI355
   kernel calls `chiplet_transform_chunked` so the workgroup grid maps
   cleanly to the 8-die topology.
6. **The kernel author owns scheduling**. `s_setprio`, `sched_barrier`,
   `s_waitcnt vmcnt(N)`, M0 SGPR pinning, address pre-fill — all are
   explicit in user code. No pipeline-policy abstraction hides them.

### 10.5 Status

This playbook covers HipKittens at upstream pin `5294d1c5` (main).
10 slices were synthesized from focused deep-read passes over
`include/`, `kernels/`, `distributed-kernels/`, `training/`, and
`fp6_dwordx{3,4}/`, with cross-checks against the
`agent-mxfp8-gemm-optimization.md` design note and the matched
[CK playbook](../aiter-kernel-analysis/CK_PLAYBOOK.md).

Future HipKittens updates (new kernels, new arch paths) should be
analyzed in fresh sections appended after §10, with the symlink target
bumped (or replaced by a real submodule pin) at the wrapper repo root.

---

## §11. Assembly subtree + infrastructure + sched_barrier numerics + FP4 stub + errata

Fill-in slice added after a second review surfaced gaps in §1–§10. Six
topics plus an errata note correcting §5.1.7's filename.

### 11.1 The `assembly/` subtrees — inline-asm fast path

Four parallel directories under `include/ops/warp/` mirror the
high-level `.cuh` headers with hand-written inline-asm implementations:

1. **`memory/tile/assembly/`** —
   [include/ops/warp/memory/tile/assembly/](HipKittens/include/ops/warp/memory/tile/assembly/)
   contains `global_to_register.cuh` (15.5 KB), `shared_to_register.cuh`
   (25.8 KB), `tile.cuh` (aggregator).
2. **`memory/vec/assembly/`** —
   [include/ops/warp/memory/vec/assembly/](HipKittens/include/ops/warp/memory/vec/assembly/)
   — `shared_to_register.cuh` (1.4 KB) + `vec.cuh`.
3. **`register/tile/assembly/`** —
   [include/ops/warp/register/tile/assembly/](HipKittens/include/ops/warp/register/tile/assembly/)
   — `conversions.cuh` (10.1 KB), `maps.cuh` (21.9 KB), `mma.cuh` (24.6 KB),
   `tile.cuh`.
4. **`register/vec/assembly/`** —
   [include/ops/warp/register/vec/assembly/](HipKittens/include/ops/warp/register/vec/assembly/)
   — `maps.cuh` (854 B) + `vec.cuh`.
5. **Register-direct addressing via `<start_gpr>` template args** —
   [memory/tile/assembly/global_to_register.cuh:62-76](HipKittens/include/ops/warp/memory/tile/assembly/global_to_register.cuh#L62-L76)
   — assembly wrappers (`macros::buffer_load_dwordx4<start_gpr>()`) compute
   the GPR index at compile time so the inline-asm uses *named* registers
   rather than letting LLVM allocate. Avoids extra VGPR→VGPR moves.
6. **Dispatch by parent `tile.cuh`** — each `tile.cuh` aggregator pulls in
   *both* the builtin (`global_to_register.cuh`) *and* the assembly
   (`assembly/global_to_register.cuh`) variant. Selection happens at
   template-resolution time based on the tile shape / `art` register-range
   parameter.
7. **No `KITTENS_CDNA3 / CDNA4` gate inside the assembly files** — they
   compile on both targets; arch-specific differences come from the
   target-aware `macros.cuh` register-range tables (§9.1.3).
8. **When to use the assembly variant** — the §4 GEMM and §5 attention
   kernels use the assembly path because they need explicit AGPR/VGPR
   placement to match `art::clobber` ranges. The builtin path is used
   in §6 (LayerNorm, RoPE, torch_scaled) where register placement doesn't
   matter.

### 11.2 `FP8_8wave/` GEMM variant

The §4.2 walk-through covered `FP8_4wave/`; the sibling
`FP8_8wave/` exists alongside it.

1. **`FP8_8wave/8_wave.cu`** —
   [kernels/gemm/fp8fp32/FP8_8wave/8_wave.cu](HipKittens/kernels/gemm/fp8fp32/FP8_8wave/8_wave.cu)
   — 8-wave variant of the same 256×256 fp8 GEMM.
2. **Launch bounds `(512, 2)` vs `(256, 1)`** — twice the threads per
   block (512), 2 blocks/CU instead of 1.
3. **Warp layout `2 × 4` (vs `2 × 2`)** — `WARPS_ROW = 2`,
   `WARPS_COL = 4`; doubles the N-axis warp count.
4. **Register tile shrinks `64 × 32`** — derived
   `REG_BLOCK_M = 256 / 2 / 2 = 64`,
   `REG_BLOCK_N = 256 / 4 / 2 = 32`; half the per-warp register footprint
   of the 4-wave path.
5. **Shared-mem 2D double-buffer `As[2][2] / Bs[2][2]`** vs the 4-wave
   path's 1-D ping-pong — more LDS pressure but better interleaving with
   8 producers.
6. **Use case**: wider batch / N-axis problems. The 4-wave version is
   tuned for tall-skinny M ≫ N; the 8-wave is tuned for the opposite.
7. **`s_waitcnt vmcnt(4)`** —
   [FP8_8wave/8_wave.cu:94](HipKittens/kernels/gemm/fp8fp32/FP8_8wave/8_wave.cu#L94)
   — same wait-count tuning pattern as 4-wave but with `vmcnt(4)` rather
   than `vmcnt(6)`; the extra producers need a tighter drain.

### 11.3 `kernels/gemm/bf16fp32/micros/` — incremental tuning bench

Where the §4.4 agent walk's hypotheses were tested:

1. **`micros/192x256/`** — fixed output tile (128 or 192 rows × 256 cols);
   `kernel.cpp` + `kernelv2.cpp` are iterative refinements on
   shared-memory layout vs producer/consumer balance.
2. **`micros/hint_based/`** — `schedule_utils.cpp` wraps
   `__builtin_amdgcn_sched_*` with different hint values; tests impact of
   explicit pipeline scheduling on a fixed tile.
3. **`micros/producer_consumer/{16x32, 32x16}/`** — sweeps over
   producer/consumer ratios (`8c4p`, `12c4p`, `16c2p`) and stage depth
   (2 or 3). Files like `micro_N1_2stage_8c4p.cpp` evaluate one
   hypothesis in isolation.
4. **`micros/archive/`** — superseded variants. Per the README,
   `micro_05_2stage_16c2p.cpp` is marked "Above SW limit" — kept as a
   data point even though it lost.
5. **These are the artifacts of the agent walk** — §4.4 documents the
   11.9 % speedup as an evolution; `micros/` is where each step was
   prototyped before being folded into the production
   `256_256_64_32_with{16x32, 32x16}.cpp` kernels.
6. **No README in micros/** — the harness is internal; consumers are
   expected to read the agent doc and follow back.

### 11.4 `analysis/` — evaluation methodology

The eight subdirs under `analysis/` carry the paper-replication and
per-arch benchmark infra:

1. **`analysis/baselines/`** —
   [analysis/baselines/README.md](HipKittens/analysis/baselines/README.md)
   reference impls + exact build pins: CK commit `d88ea05c`, Triton
   branch `76076e1d`, hipBLASLt flags. Documents 500 warmup + 100
   measurement iters as the standard protocol.
2. **`analysis/paper_experiments/`** — `compile_time/`,
   `producer_consumer_micro/`, `grid_micro/`, `pingpong_micro/`,
   `phases/` — each replicates one paper figure.
3. **`analysis/bf16_gemm/{mi325x, mi350x}/`** — per-arch GEMM benchmark
   curves; `plot.py` reads JSON logs and emits PNGs.
4. **`analysis/attn/{fwd, bkwd}/benchmark/`** — full attention with
   causal / non-causal variants;
   [analysis/attn/fwd/benchmark/mi355x_benchmark.sh](HipKittens/analysis/attn/fwd/benchmark/mi355x_benchmark.sh)
   is the canonical shape-sweep driver.
5. **`analysis/{fp6_gemm, fp8_gemm, layernorm, rotary}/`** — operator-
   specific benchmarks with MI350X / MI355X JSON logs and publication-
   ready plots.
6. **Standard methodology**: shape sweep → JSON log per shape →
   plot.py against CK / Triton / hipBLASLt baseline → compile-time
   measurement in isolation to call out template-compile cost.

### 11.5 `sched_barrier_pairs<…>` / `sched_barrier_exp_pairs<…>` — exact numerics

The §5 callouts cited these wrappers without explaining what their
template integers mean. Verified from source:

1. **Defined in the head-dim-64 attention kernel** —
   [kernels/attn/gqa/kernel_d64.cpp:41-55](HipKittens/kernels/attn/gqa/kernel_d64.cpp#L41-L55)
   — `SCHED_BARRIER` macro wraps `__builtin_amdgcn_sched_group_barrier(mask,
   cnt, group)`.
2. **Mask values** —
   [kernel_d64.cpp:32-34](HipKittens/kernels/attn/gqa/kernel_d64.cpp#L32-L34):
   ```
   #define MFMA_MASK 0x08
   #define VALU_MASK 0x02
   #define EXP_MASK  0x400
   ```
   `MFMA_MASK = 0x08` matches CK's `0x008` ([ck:§10.6]); `VALU_MASK = 0x02`
   is V-ALU (compute); `EXP_MASK = 0x400` is the export channel.
3. **`sched_barrier_pairs<Pairs, VALU_CNT, Group>`** —
   [kernel_d64.cpp:44-47](HipKittens/kernels/attn/gqa/kernel_d64.cpp#L44-L47)
   ```
   template<int Pairs, int VALU_CNT, int Group>
   sched_barrier_pairs() {
       SCHED_BARRIER(MFMA_MASK, 1, Group);     // 1 MFMA in this group
       SCHED_BARRIER(VALU_MASK, VALU_CNT, Group);  // VALU_CNT VALU ops
       if constexpr (Pairs > 1) sched_barrier_pairs<Pairs-1, VALU_CNT, Group>();
   }
   ```
   So `sched_barrier_pairs<10, 5, 1>()` emits **10 pairs of `(1 MFMA, 5
   VALU)` in scheduler group 1** — locks in a tight `MFMA → 5 VALU →
   MFMA → 5 VALU → …` schedule.
4. **`sched_barrier_exp_pairs<Pairs, EXP_CNT, Group>`** —
   [kernel_d64.cpp:51-54](HipKittens/kernels/attn/gqa/kernel_d64.cpp#L51-L54)
   — same pattern with `EXP_MASK / EXP_CNT` instead. `<6, 3, 1>` =
   "6 × (1 MFMA + 3 exports) in group 1".
5. **`Group` arg = scheduler group ID** — independent groups can be
   interleaved by the wave scheduler; HK uses groups 1–7 in the
   attention hot loop ([kernel_d64.cpp:246-449](HipKittens/kernels/attn/gqa/kernel_d64.cpp#L246-L449)).
6. **Comparison to CK**: CK passes raw `__builtin_amdgcn_sched_group_barrier`
   calls with explicit masks per-site ([ck:§3.3, §10.6]); HK wraps the
   pattern recursively. Same primitive, terser API.
7. **Why this lives in `kernel_d64.cpp` not `kernel.cpp`** — the head-dim-64
   path has tighter register pressure and benefits more from explicit
   scheduling; the head-dim-128 `kernel.cpp` (also under
   `kernels/attn/gqa/`) does not use the recursive wrappers.

### 11.6 FP4 (`fp4e2m1`) — forward-compat stub

1. **Type declared** —
   [include/common/base_types.cuh:16, :55, :59, :63](HipKittens/include/common/base_types.cuh#L16)
   — `#include <hip/hip_fp4.h>` plus `fp4e2m1`, `fp4e2m1_2`, `fp4e2m1_4`
   typedefs and `packing<…>` specializations.
2. **Constants + convertors registered** —
   [base_types.cuh:159-165, :397-412](HipKittens/include/common/base_types.cuh#L159-L412)
   — `ones / zeros` constants and `convertor<fp4_2, …>` cast hooks.
3. **No kernel uses it** — exhaustive grep over `kernels/`, `training/`,
   `fp6_dwordx{3,4}/` finds zero instantiations of `rt_fp4*`, `st_fp4*`,
   or `mma_*_fp4`.
4. **Status**: forward-compatibility stub for future CDNA fp4 MFMA. If
   you're building on HipKittens and need fp4, you'll need to add the
   register-tile / shared-tile specializations and an MFMA wrapper —
   the type-system foundation is already there.

### 11.7 Deliberately absent — verified

Exhaustive grep over the whole tree (`include/`, `kernels/`, `training/`,
`fp6_dwordx{3,4}/`, `analysis/`, `distributed-kernels/`):

| Symbol | Hits | Notes |
|---|---|---|
| `__builtin_amdgcn_iglp_opt` | 0 | HK uses `s_setprio` + `sched_barrier` instead |
| `s_clause` | 0 | gfx10+; irrelevant on CDNA |
| `dpp_row_shr` | 0 | DPP not used at all |
| `dpp_quad_perm` | 0 | DPP not used at all |
| `permlanex_b32` | 0 | HK uses `permlane16/32_swap` only |
| `MOV_64x4` | 0 | No 256-bit stores |
| `__syncwarp` | 64 (all in comments / fp6_dwordx3 dead code) | Not actually called |

HipKittens deliberately avoids the wave-level DPP / clause / syncwarp
families; synchronization is **threadblock-level via `__builtin_amdgcn_s_barrier()`**
or **implicit via SGPR-uniform M0 + disjoint LDS regions** (§3.3).

### 11.8 Errata to earlier slices

1. **§5.1 item 7 cited `kernel.cpp:246-247, 318-319`** for
   `sched_barrier_pairs<10, 5, group>` / `sched_barrier_exp_pairs<6, 3, group>`.
   The wrappers are defined in **`kernel_d64.cpp`** (head-dim-64 path),
   not `kernel.cpp` (head-dim-128). The line numbers match; only the
   filename was wrong. The §5.1.7 cross-references should read:
   - `sched_barrier_pairs<10, 5, group>` →
     [kernel_d64.cpp:247, 319](HipKittens/kernels/attn/gqa/kernel_d64.cpp#L247)
   - `sched_barrier_exp_pairs<6, 3, group>` →
     [kernel_d64.cpp:246, 318](HipKittens/kernels/attn/gqa/kernel_d64.cpp#L246)

### 11.9 Cross-cutting techniques (this slice)

- **T11.A — Dual implementation per warp op (builtin + assembly)**: the
  `assembly/` subtree exists *in parallel* with the `.cuh` headers; the
  parent `tile.cuh` includes both and template resolution picks. This
  is how HK keeps a portable LLVM-builtin path *and* a register-named
  inline-asm path in the same binary without runtime cost.
- **T11.B — `<start_gpr>` template argument is the AGPR/VGPR placement
  contract**: assembly variants take the register index as a template
  param so the compiler can't reorder; pair with `art::clobber` for
  end-to-end register pinning.
- **T11.C — 4-wave vs 8-wave is a tile-aspect tuning knob**: same MFMA
  shape, different warp layout — 4-wave tunes for tall-skinny, 8-wave
  for wide; `vmcnt` is tuned to match (6 vs 4).
- **T11.D — `sched_barrier_pairs<Pairs, VALU_CNT, Group>` is a recursive
  schedule lock**: each "pair" emits `(MFMA, VALU_CNT VALU)` in
  `Group`; the recursive template unfolds to a flat instruction stream
  the scheduler must respect.
- **T11.E — `Group` parameter gives the scheduler explicit cross-group
  freedom**: independent groups can be interleaved freely while
  intra-group order is locked. Attention uses groups 1–7 to mark
  independent phases of the softmax + AV pipeline.
- **T11.F — `analysis/baselines/` pins CK / Triton / hipBLASLt versions**:
  reproducible head-to-head numbers depend on the exact commit pins
  documented in the baseline README.
- **T11.G — `micros/` is the experiment journal**: each `.cpp` is one
  hypothesis from the agent walk; the dirs are the design history. To
  understand why a production kernel made specific choices, read its
  `micros/` cousins.
- **T11.H — FP4 is staged for later**: the type system is in place,
  the kernels are not. Documents HK's design posture — "lay the
  scaffolding for the next dtype before the hardware ships."

Cross-references: [ck:§3.3, §10.6] (CK's `sched_group_barrier` with raw
masks — HK wraps them); [ck:§16.10] (`PrefetchStages` occupancy cliff —
HK's `micros/` is where this gets tuned empirically); [ck:§11] (CK's
1981-instance codegen archive — HK's tuning happens through `micros/`
and the agent walk instead).


