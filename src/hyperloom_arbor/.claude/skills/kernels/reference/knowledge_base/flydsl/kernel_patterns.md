# FlyDSL Kernel Patterns

> Five canonical patterns, each shown with the smallest working skeleton.
> Use these as templates — when in doubt, copy and adapt.

## Mental Model: Layout Is the Glue

```
make_layout(shape, stride)            ─┐
                                       │  Layout = coord → index
                                       │
zipped_divide(tensor, tile)           ─┤
slice(divided, (None, bid))           ─┤  Partition → this block's tile
                                       │
CopyAtom(BufferCopy128b, f32)         ─┤  Atom = one hardware copy/MMA inst
MmaAtom(MFMA(16,16,32,f8))            ─┤
                                       │
make_tiled_copy(atom, layout_tv, tile)─┤  TiledCopy = atom × thread cooperation
make_tiled_mma(atom, atom_layout)     ─┤
                                       │
tc.get_slice(tid).partition_S(t)      ─┤  ThrCopy.partition_S/D — this thread
tm.thr_slice(tid).make_fragment_C(t)  ─┤
                                       │
fx.copy(atom, src, dst)               ─┤  Execute
fx.gemm(atom, D, A, B, C)             ─┘
```

## Pattern A: Elementwise (vec_add, scale, relu, …)

```python
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu
from flydsl.expr.typing import Vector as Vec

BLOCK_DIM = 256
VEC_WIDTH = 4

@flyc.kernel
def elemwise(A: fx.Tensor, Out: fx.Tensor,
             BLOCK_DIM: fx.Constexpr[int], VEC_WIDTH: fx.Constexpr[int]):
    bid = gpu.block_id("x")
    tid = gpu.thread_id("x")
    tile = BLOCK_DIM * VEC_WIDTH

    # Divide into block tiles, select this block's tile
    tA = fx.slice(fx.logical_divide(A, fx.make_layout(tile, 1)), (None, bid))
    tOut = fx.slice(fx.logical_divide(Out, fx.make_layout(tile, 1)), (None, bid))

    # Further divide for per-thread VEC_WIDTH access
    tA = fx.logical_divide(tA, fx.make_layout(VEC_WIDTH, 1))
    tOut = fx.logical_divide(tOut, fx.make_layout(VEC_WIDTH, 1))

    # Register fragments
    copy_atom = fx.make_copy_atom(fx.UniversalCopy(VEC_WIDTH * 32), fx.Float32)
    rA = fx.make_rmem_tensor(VEC_WIDTH, fx.Float32)
    rOut = fx.make_rmem_tensor(VEC_WIDTH, fx.Float32)

    # Load → compute → store
    fx.copy_atom_call(copy_atom, fx.slice(tA, (None, tid)), rA)
    vA = Vec(fx.memref_load_vec(rA))
    vOut = vA * vA                                   # ← compute
    fx.memref_store_vec(vOut, rOut)
    fx.copy_atom_call(copy_atom, rOut, fx.slice(tOut, (None, tid)))

@flyc.jit
def launch(A: fx.Tensor, Out: fx.Tensor, N: fx.Int32,
           stream: fx.Stream = fx.Stream(None)):
    grid_x = (N + BLOCK_DIM * VEC_WIDTH - 1) // (BLOCK_DIM * VEC_WIDTH)
    elemwise(A, Out, BLOCK_DIM, VEC_WIDTH).launch(
        grid=(grid_x, 1, 1), block=(BLOCK_DIM, 1, 1), stream=stream)
```

Recipes for the compute block (all on `Vec`):

```python
# Add: vC = vA + vB
vC = vA + vB

# FMA: vD = vA * vB + vC
vD = vA * vB + vC

# ReLU: vC = max(vA, 0)
zero = Vec.filled(VEC_WIDTH, 0.0, fx.Float32)
vC = vA.maximumf(zero)

# Abs (arith.absf does NOT exist):
neg = -vA
is_neg = vA < zero
vC = is_neg.select(neg, vA)

# Type conversion:
vC = vA.to(fx.BFloat16)        # f32 → bf16
vC = vA.bitcast(fx.Int32)      # reinterpret bits
```

## Pattern B: 2D Tiled Copy (transpose / gather)

```python
@flyc.kernel
def tiled_copy(A: fx.Tensor, B: fx.Tensor):
    tid = gpu.thread_id("x")
    bid = gpu.block_id("x")

    block_m, block_n = 8, 24
    tile = fx.make_tile([fx.make_layout(block_m, 1), fx.make_layout(block_n, 1)])
    A_buf = fx.rocdl.make_buffer_tensor(A)
    B_buf = fx.rocdl.make_buffer_tensor(B)

    bA = fx.slice(fx.zipped_divide(A_buf, tile), (None, bid))
    bB = fx.slice(fx.zipped_divide(B_buf, tile), (None, bid))

    # Thread-value layout
    thr_layout = fx.make_layout((4, 1), (1, 1))   # 4 threads along M
    val_layout = fx.make_layout((1, 8), (1, 1))   # 8 values per thread along N

    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
    layout_tv = fx.raked_product(thr_layout, val_layout)
    tile_mn = fx.make_tile(4, 8)
    tiled_copy = fx.make_tiled_copy(copy_atom, layout_tv, tile_mn)

    thr_copy = tiled_copy.get_slice(tid)
    src = thr_copy.partition_S(bA)
    dst = thr_copy.partition_D(bB)
    frag = fx.make_fragment_like(src)

    fx.copy(copy_atom, src, frag)
    fx.copy(copy_atom, frag, dst)
```

## Pattern C: Tiled MMA (canonical GEMM kernel; from examples/03-tiledMma.py)

```python
block_m, block_n, block_k = 64, 64, 8

@flyc.kernel
def gemm(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    A_buf = fx.rocdl.make_buffer_tensor(A)
    B_buf = fx.rocdl.make_buffer_tensor(B)
    C_buf = fx.rocdl.make_buffer_tensor(C)

    bA = fx.slice(fx.zipped_divide(A_buf, (block_m, block_k)), (None, bid))
    bB = fx.slice(fx.zipped_divide(B_buf, (block_n, block_k)), (None, bid))
    bC = fx.slice(fx.zipped_divide(C_buf, (block_m, block_n)), (None, bid))

    # MFMA atom + tile across threads
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
    thr_mma = tiled_mma.thr_slice(tid)

    # Matched TiledCopy for A, B, C operands
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    tcA = fx.make_tiled_copy_A(copy_atom, tiled_mma)
    tcB = fx.make_tiled_copy_B(copy_atom, tiled_mma)
    tcC = fx.make_tiled_copy_C(copy_atom, tiled_mma)

    tcAs, tcBs, tcCs = tcA.get_slice(tid), tcB.get_slice(tid), tcC.get_slice(tid)

    # Copy partitions (data movement view)
    src_A, src_B, dst_C = tcAs.partition_S(bA), tcBs.partition_S(bB), tcCs.partition_S(bC)

    # MMA fragments (register view)
    frag_A = thr_mma.make_fragment_A(bA)
    frag_B = thr_mma.make_fragment_B(bB)
    frag_C = thr_mma.make_fragment_C(bC)

    # Retile fragments for copy compatibility
    cf_A, cf_B, cf_C = tcAs.retile(frag_A), tcBs.retile(frag_B), tcCs.retile(frag_C)

    fx.copy(copy_atom, src_A, cf_A, pred=None)
    fx.copy(copy_atom, src_B, cf_B, pred=None)
    frag_C.fill(0)
    fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)
    fx.copy(copy_atom, cf_C, dst_C, pred=None)
```

Atom layout `(2, 2, 1)` with stride `(1, 2, 0)` means: 2 M-replicas, 2 N-replicas,
1 K-replica per warp; the stride says M varies fastest, then N, K is
broadcast. With block (256, 1, 1) this is 4 warps × 4 atoms = 16 MFMA instances.

## Pattern D: Buffer Load/Store (low-level, no layout algebra)

Best when you need explicit address arithmetic (paged-KV indices, irregular
access). The offset is in **elements of `dtype`**, NOT bytes.

```python
from flydsl.expr import buffer_ops

@flyc.kernel
def buf_kernel(A: fx.Tensor, B: fx.Tensor, N: fx.Constexpr[int]):
    tid = gpu.thread_id("x")
    bid = gpu.block_id("x")
    gid = bid * 256 + tid

    rsrc_a = buffer_ops.create_buffer_resource(A)
    rsrc_b = buffer_ops.create_buffer_resource(B)

    # Vectorized 128-bit load = 4×i32 = 16 bytes
    data = buffer_ops.buffer_load(rsrc_a, gid * 4, vec_width=4, dtype=fx.Int32)
    buffer_ops.buffer_store(data, rsrc_b, gid * 4)
```

For FP8 (1 byte/elem) addressed in bytes, divide by 4 when loading as i32:
```python
k_4xi32 = buffer_ops.buffer_load(k_rsrc, k_addr_bytes // 4,
                                  vec_width=4, dtype=fx.Int32)
```

## Pattern E: Reduction (sum, max, softmax — block-wide via warp shuffle + LDS)

```python
WARP_SIZE = 64
BLOCK_THREADS = 256
NUM_WAVES = BLOCK_THREADS // WARP_SIZE  # 4

# 1. Vector reduction (in registers)
def reduce_vec_max(vec, vec_width):
    acc = vec[0]
    for i in range_constexpr(1, vec_width):
        acc = acc.maximumf(vec[i])
    return acc

# 2. Intra-wave XOR-shuffle reduction (wave64)
def warp_reduce_max(val):
    width_i32 = fx.Int32(WARP_SIZE)
    for sh in [32, 16, 8, 4, 2, 1]:
        peer = gpu.ShuffleOp(val, fx.Int32(sh), width_i32, mode="xor").shuffleResult
        val = val.maximumf(peer)
    return val

# 3. Block-wide reduction = warp reduce → LDS → wave0 finalize
@flyc.kernel
def reduce_kernel(...):
    ...
    local_max = reduce_vec_max(vec, VEC_WIDTH)
    wave_max = warp_reduce_max(local_max)

    # Lane 0 of each wave writes its partial to LDS
    lane = tid % fx.Int32(WARP_SIZE)
    wave_id = tid // fx.Int32(WARP_SIZE)
    if lane == fx.Int32(0):
        lds_partial_ptr.store(wave_max, [wave_id])
    gpu.barrier()

    # Wave 0 reads all NUM_WAVES partials and reduces
    if wave_id == fx.Int32(0):
        if lane < fx.Int32(NUM_WAVES):
            partial = lds_partial_ptr.load([lane])
        else:
            partial = NEG_INF
        global_max = warp_reduce_max(partial)
        # Lane 0 broadcasts via LDS for the rest of the block to read
        if lane == fx.Int32(0):
            lds_final_ptr.store(global_max, [0])
    gpu.barrier()
    global_max = lds_final_ptr.load([0])
```

Reusable versions in `kernels/kernels_common.py`:
`reduce_vec_max`, `reduce_vec_sum`, `make_block_reduce`, `make_block_reduce_add`,
`make_block_reduce_add2`.

## Pattern F: Loop-Carried State (Software Pipelining)

For prefetch patterns where data flows across loop iterations:

```python
# Prologue — load iteration 0
tile_0 = prefetch(0)
init_state = [acc_init, *tile_0_flat]

# Runtime loop with explicit SSA phi nodes
# BOUNDS MUST BE DSL VALUES (fx.Index), else AST rewriter unrolls and ignores init=
start, stop, step = fx.Index(0), fx.Index(N - 1), fx.Index(1)
for iv, state in range(start, stop, step, init=init_state):
    acc_in = state[0]
    tile_in = state[1:]

    next_tile = prefetch(iv + 1)              # load NEXT
    acc_in = compute(acc_in, tile_in)         # compute CURRENT

    results = yield [acc_in, *next_tile]      # carry to next iter

# Epilogue — clear SmemPtr view cache before reading shared mem
# (else SSA dominance error from loop-scoped view)
my_smem_ptr._view_cache = None
acc_final = results[0]
tile_final = results[1:]
compute(acc_final, tile_final)
```

## Cross-Cutting Practical Rules

1. **`range_constexpr(N)` for compile-time loops; `range(...)` for runtime.**
   Plain `range(int, int, int)` is treated as Python `range` and unrolled
   silently — must use `fx.Index(...)` bounds to get an actual `scf.for`.
2. **Don't `const_expr(...)` runtime GPU values.** Even with `known_block_size`,
   `gpu.thread_id("x")`, `lane`, `warp_id` are runtime SSA values.
3. **Single definition path through if/else.** Don't define a var in only one
   branch and use it after — hoist it. `x = a if c else b` is fine for single
   SSA values.
4. **No mutation of captured outer vars in nested helpers.** Read-only capture
   OK; passes-through-args required for writes.
5. **Avoid early `return` / branch-local `return`/`yield` in traced functions.**
   Single explicit exit.
6. **Runtime branches with side effects** must wrap in a local `@flyc.jit`
   helper to keep MLIR result types well-defined.
7. **SmemPtr._view_cache** must be cleared before epilogue if the SmemPtr was
   used inside the loop body.
8. **`buffer_ops.buffer_load` offsets are in elements**, not bytes.
9. **MFMA operand order**: `mfma(LHS, RHS, acc)` — LHS → M dimension, RHS → N
   dimension. Swapping silently produces transposed output.
10. **Vector stores need vector values**, not scalars — splat with
    `Vec.from_elements([x], fx.Float32)`.

## Where to Look in Production Kernels

- Simplest: [examples/01-vectorAdd.py](`/tmp_flydsl_src/FlyDSL/examples/01-vectorAdd.py`)
- 2D copy: examples/02-tiledCopy.py
- GEMM atom: examples/03-tiledMma.py
- Full preshuffle GEMM: examples/04-preshuffle_gemm.py, kernels/preshuffle_gemm.py
- Reduction kernels: kernels/{layernorm,rmsnorm,softmax}_kernel.py
- Attention: kernels/{pa_decode_fp8,flash_attn_func,mla_fwd_decode}.py
- MoE: kernels/moe_{gemm,blockscale}_2stage.py
- RDNA/gfx1250: kernels/{rdna_*,wmma_*,gemm_*_gfx1250}.py
