---
name: flydsl-tile-programming
description: >
  Procedure for producing a new FlyDSL kernel: classify the pattern, take the matching skeleton,
  fill in compute, add control flow / sync / LDS, then verify on GPU. Use when writing a new
  kernel, porting a Triton kernel to FlyDSL, or learning tile programming by following steps.
  For API lookups, per-op tables, and the exhaustive troubleshooting list, use
  flydsl-kernel-authoring instead; for diagnosing a kernel that already runs and is wrong,
  use debug-flydsl-kernel.
allowed-tools: Read Edit Bash Grep Glob Agent
---

# Writing a FlyDSL kernel

## Route here when
You are **producing** a kernel — from a requirement, or by porting one from Triton — and want a
procedure to follow in order.

**Go elsewhere when:**

| You want | Go to |
|---|---|
| To look up an op, a layout-algebra rule, or an env var | `flydsl-kernel-authoring` (the reference) |
| To fix a kernel that runs and is wrong | `../skills/bottleneck/debug-flydsl-kernel.md` |
| To use a kernel FlyDSL already ships | `../skills/optimize/flydsl_levers/flydsl_kernel_library.md` |
| To make a working kernel faster | `../skills/optimize/flydsl_levers/flydsl_authoring_method.md` |

**Prerequisites:** FlyDSL installed editable (`pip install -e .`), and a GPU — every step below ends
in a real launch, because tile-programming bugs do not show up statically.

## The mental model, first
Everything in FlyDSL is layout algebra. Read this before the skeletons; the skeletons are just this
diagram instantiated.

```
Layout            make_layout(shape, stride)   →  a map: logical coord → physical index

Divide            zipped_divide(Tensor, Tile)  →  (tile_interior, tile_id)
                  slice(divided, (None, bid))  →  this block's tile

Atom              CopyAtom  = ONE hardware copy instruction (32b / 64b / 128b)
                  MmaAtom   = ONE MFMA instruction

Tiled operation   TiledCopy = CopyAtom × thread_layout   →  threads cooperate on a copy
                  TiledMma  = MmaAtom  × atom_layout     →  threads cooperate on an MMA

Per-thread view   ThrCopy.partition_S/D(tensor)   →  this thread's source / destination
                  ThrMma.partition_A/B/C(tensor)  →  this thread's operands

Fragment          make_fragment_like(partition)   →  a register tile
                  retile(fragment)                →  reshape so a copy can consume it

Execute           fx.copy(atom, src, dst)         →  data movement
                  fx.gemm(atom, D, A, B, C)       →  D = A @ B + C
```

**Layout is the glue.** Divide, partition, copy, and gemm are all defined in terms of layouts. Getting
the layouts right is most of the work; the compute is usually three lines.

## gfx950 constants you will need
| Fact | Value | Where it bites |
|---|---|---|
| LDS per workgroup | **160 KiB** | `SmemAllocator` sizing, tile-size ceilings |
| LDS banks | **64** | any padding or swizzle you inherited from a 32-bank design is wrong here |
| Wavefront | 64 lanes | thread-layout arithmetic is mod 64, not mod 32 |
| CU count | 256 | grid sizing — query it, do not hardcode |
| fp8 encoding | **OCP** | FNUZ is CDNA3; a checkpoint in the wrong dialect is a silent ~2× error |

Full numbers: `local_knowledge/hardware/mi350_lds.md`, `mi350_matrix_core.md`, `mi350_overview.md`.

---

## Step 1 — Classify the pattern
Every FlyDSL kernel falls into one of five shapes. Pick one; it decides which primitives you need.

| Pattern | Examples | Key primitives | Skeleton |
|---|---|---|---|
| **Elementwise** | vecadd, scale, relu | `logical_divide` + `copy_atom_call` | [A](#pattern-a--elementwise) |
| **Reduction** | sum, max, softmax, layernorm | `buffer_load` + cross-lane shuffle + LDS | build on A, add §5–§6 |
| **Tiled copy** | transpose, permute, gather | `zipped_divide` + `TiledCopy` | [B](#pattern-b--tiled-2-d-copy) |
| **GEMM** | matmul, batched GEMM | `TiledMma` + `TiledCopy` + LDS | [C](#pattern-c--tiled-mma-gemm) |
| **Fused** | attention, GEMM + epilogue | GEMM skeleton + elementwise epilogue | C, then Step 2 |

If you cannot decide between two, start with the simpler one and get it correct on GPU before adding
the second half. A wrong fused kernel is extremely hard to bisect.

## Step 2 — Take the skeleton
Every FlyDSL kernel is two functions: a `@flyc.kernel` device body and a `@flyc.jit` launcher.

```python
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx

@flyc.kernel
def my_kernel(A: fx.Tensor, B: fx.Tensor, ...):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    ...

@flyc.jit
def my_launch(A: fx.Tensor, B: fx.Tensor, ...,
              stream: fx.Stream = fx.Stream(None)):
    my_kernel(A, B, ...).launch(
        grid=(grid_x, grid_y, grid_z),
        block=(block_x, 1, 1),
        stream=stream,
    )
```

### Pattern A — elementwise
Each thread owns `VEC_WIDTH` elements. Data flow: global → register → compute → register → global.

```python
from flydsl.expr.typing import Vector as Vec

BLOCK_DIM, VEC_WIDTH = 256, 4

@flyc.kernel
def elementwise_kernel(A: fx.Tensor, Out: fx.Tensor,
                       BLOCK_DIM: fx.Constexpr[int], VEC_WIDTH: fx.Constexpr[int]):
    bid, tid = fx.block_idx.x, fx.thread_idx.x

    # 1. cut the global tensor into block-sized tiles
    tile_size = BLOCK_DIM * VEC_WIDTH
    tA   = fx.logical_divide(A,   fx.make_layout(tile_size, 1))
    tOut = fx.logical_divide(Out, fx.make_layout(tile_size, 1))

    # 2. take this block's tile
    tA   = fx.slice(tA,   (None, bid))
    tOut = fx.slice(tOut, (None, bid))

    # 3. cut again for per-thread vectorized access
    tA   = fx.logical_divide(tA,   fx.make_layout(VEC_WIDTH, 1))
    tOut = fx.logical_divide(tOut, fx.make_layout(VEC_WIDTH, 1))

    # 4. registers + the copy instruction to use
    copy_atom = fx.make_copy_atom(fx.UniversalCopy(VEC_WIDTH * 32), fx.Float32)
    rA   = fx.make_rmem_tensor(VEC_WIDTH, fx.Float32)
    rOut = fx.make_rmem_tensor(VEC_WIDTH, fx.Float32)

    # 5. load → compute → store
    fx.copy_atom_call(copy_atom, fx.slice(tA, (None, tid)), rA)
    vA = Vec(fx.memref_load_vec(rA))
    vOut = vA * vA                      # <<< YOUR COMPUTE
    fx.memref_store_vec(vOut, rOut)
    fx.copy_atom_call(copy_atom, rOut, fx.slice(tOut, (None, tid)))

@flyc.jit
def elementwise_launch(A: fx.Tensor, Out: fx.Tensor, N: fx.Int32,
                       stream: fx.Stream = fx.Stream(None)):
    tile_size = BLOCK_DIM * VEC_WIDTH
    elementwise_kernel(A, Out, BLOCK_DIM, VEC_WIDTH).launch(
        grid=((N + tile_size - 1) // tile_size, 1, 1),
        block=(BLOCK_DIM, 1, 1), stream=stream)
```

Note the **two-level divide** in steps 1 and 3 — first to blocks, then to per-thread vectors. That
nesting is the whole idiom; the rest is bookkeeping.

### Pattern B — tiled 2-D copy
Uses `zipped_divide` plus `TiledCopy` for an explicit thread-value mapping. Data flow: global[M,N] →
fragment → global[M,N] with a layout change.

```python
@flyc.kernel
def tiled_copy_kernel(A: fx.Tensor, B: fx.Tensor):
    tid, bid = fx.thread_idx.x, fx.block_idx.x

    block_m, block_n = 8, 24
    tile = fx.make_tile([fx.make_layout(block_m, 1), fx.make_layout(block_n, 1)])

    A = fx.rocdl.make_buffer_tensor(A)        # AMD buffer descriptors
    B = fx.rocdl.make_buffer_tensor(B)

    bA = fx.slice(fx.zipped_divide(A, tile), (None, bid))
    bB = fx.slice(fx.zipped_divide(B, tile), (None, bid))

    # thread-value layout: how threads split the tile
    thr_layout = fx.make_layout((4, 1), (1, 1))     # 4 threads along M
    val_layout = fx.make_layout((1, 8), (1, 1))     # 8 values each along N
    copy_atom  = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
    layout_tv  = fx.raked_product(thr_layout, val_layout)

    tiled_copy = fx.make_tiled_copy(copy_atom, layout_tv, fx.make_tile(4, 8))
    thr_copy   = tiled_copy.get_slice(tid)
    src, dst   = thr_copy.partition_S(bA), thr_copy.partition_D(bB)
    frag       = fx.make_fragment_like(src)

    fx.copy(copy_atom, src, frag)
    fx.copy(copy_atom, frag, dst)
```

The `thr_layout × val_layout` product is where a transpose actually happens — you change *which*
thread reads *which* element, not the addresses.

### Pattern C — tiled MMA (GEMM)
Data flow: global → TiledCopy → fragments A,B → MFMA → fragment C → global.

```python
block_m, block_n, block_k = 64, 64, 8

@flyc.kernel
def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    tid, bid = fx.thread_idx.x, fx.block_idx.x

    tileA, tileB, tileC = (fx.make_tile(block_m, block_k),
                           fx.make_tile(block_n, block_k),
                           fx.make_tile(block_m, block_n))

    A, B, C = (fx.rocdl.make_buffer_tensor(A),
               fx.rocdl.make_buffer_tensor(B),
               fx.rocdl.make_buffer_tensor(C))

    bA = fx.slice(fx.zipped_divide(A, tileA), (None, bid))
    bB = fx.slice(fx.zipped_divide(B, tileB), (None, bid))
    bC = fx.slice(fx.zipped_divide(C, tileC), (None, bid))

    # MMA: pick the instruction, then tile it across threads
    mma_atom  = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))   # see the note below
    tiled_mma = fx.make_tiled_mma(mma_atom,
                                  fx.make_layout((2, 2, 1), (1, 2, 0)))  # (M_rep, N_rep, K_rep)
    thr_mma   = tiled_mma.thr_slice(tid)

    # copies must be built FROM the mma so the layouts agree
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    thr_copy_A = fx.make_tiled_copy_A(copy_atom, tiled_mma).get_slice(tid)
    thr_copy_B = fx.make_tiled_copy_B(copy_atom, tiled_mma).get_slice(tid)
    thr_copy_C = fx.make_tiled_copy_C(copy_atom, tiled_mma).get_slice(tid)

    frag_A = thr_mma.make_fragment_A(thr_mma.partition_A(bA))
    frag_B = thr_mma.make_fragment_B(thr_mma.partition_B(bB))
    frag_C = thr_mma.make_fragment_C(thr_mma.partition_C(bC))

    fx.copy(copy_atom, thr_copy_A.partition_S(bA), thr_copy_A.retile(frag_A), pred=None)
    fx.copy(copy_atom, thr_copy_B.partition_S(bB), thr_copy_B.retile(frag_B), pred=None)
    fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)
    fx.copy(copy_atom, thr_copy_C.retile(frag_C), thr_copy_C.partition_S(bC), pred=None)
```

Two things to get right, and they are the usual failures:

- **Build the copies from the MMA** (`make_tiled_copy_A(copy_atom, tiled_mma)`), never independently.
  An independently-built copy will compile and produce a wrong operand layout.
- **`retile` before every copy that touches a fragment.** The MMA's fragment layout and the copy's
  expected layout are not the same shape.

> **On the MMA shape.** `MFMA(16, 16, 4, Float32)` is an fp32 instruction and is fine for a first
> correct kernel, but it is nowhere near peak on gfx950. The native shapes there are
> `16x16x32` / `32x32x16` for f16/bf16 and `16x16x128` / `32x32x64` for the f8f6f4 family. Prefer
> **16×16 over 32×32**: it draws less power (so clocks stay higher) *and* holds only 4 C registers per
> lane versus 16, which is what actually frees the register budget. Shape table:
> `local_knowledge/hardware/mi350_matrix_core.md`; the `make_tiled_mma` atom-layout rules are in
> **flydsl-kernel-authoring** §6.

### Pattern D — raw buffer ops
Direct AMD buffer intrinsics, bypassing the layout algebra. Use when you need address control the
algebra will not give you.

```python
from flydsl.expr import buffer_ops

@flyc.kernel
def buffer_kernel(A: fx.Tensor, B: fx.Tensor, N: fx.Constexpr[int]):
    gid = fx.block_idx.x * 256 + fx.thread_idx.x
    rsrc_a = buffer_ops.create_buffer_resource(A)
    rsrc_b = buffer_ops.create_buffer_resource(B)

    # offset is in ELEMENTS of dtype, not bytes
    data = buffer_ops.buffer_load(rsrc_a, gid * 4, vec_width=4, dtype=fx.T.f32())
    buffer_ops.buffer_store(data, rsrc_b, gid * 4)
```

The element-vs-byte offset is the classic bug here: a 4× address error usually lands *inside* the
buffer, so it fails as garbage rather than as a fault.

## Step 3 — Fill in the compute
All of these operate on vectors:

```python
from flydsl.expr.typing import Vector as Vec

vC = Vec(vA) * Vec.filled(VEC_WIDTH, 2.0, fx.Float32)   # scale
vC = Vec(vA) + Vec(vB)                                  # add
vC = Vec(vA) * Vec(vB) + Vec(vC)                        # fma
vC = Vec(vA).maximumf(Vec.filled(VEC_WIDTH, 0.0, fx.Float32))   # relu

v, zero = Vec(vA), Vec.filled(VEC_WIDTH, 0.0, fx.Float32)       # abs
vC = (v < zero).select(-v, v)

vC = Vec(vI32).to(fx.Float32)                           # int → float
vC = Vec(vF32).to(fx.Float16)                           # f32 → f16
```

Note `abs` is built from `select`, not from an `arith.absf` — that op does not exist. The same
`select`-based idiom covers most missing "obvious" ops.

## Step 4 — Control flow
```python
from flydsl.expr import range_constexpr, const_expr

for i in range_constexpr(K):        # compile-time unrolled; i is a Python int
    ...

for i in range(runtime_N):          # runtime loop; i is an ArithValue
    ...

# loop-carried state (software pipelining)
start, stop, step = fx.Index(0), fx.Index(N - 1), fx.Index(1)
for iv, state in range(start, stop, step, init=[acc_init, ...]):
    acc = state[0]
    results = yield [new_acc, ...]
final_acc = results[0]

if const_expr(USE_FAST_PATH):       # compile-time; emits no MLIR
    ...

if bid == 0:                        # runtime; rewritten to scf.IfOp
    ...
```

The distinction that costs the most time: a `range()` induction variable **cannot index a Python
list**, because it is an SSA value rather than an int. If you are indexing a Python-side structure,
you need `range_constexpr`.

## Step 5 — Synchronization
```python
fx.gpu.barrier()             # workgroup barrier (__syncthreads equivalent)

# gfx950 (CDNA4): split wait counters — prefer these
fx.rocdl.s_wait_loadcnt(0)
fx.rocdl.s_wait_storecnt(0)
fx.rocdl.s_wait_dscnt(0)

fx.rocdl.s_waitcnt(0)        # CDNA3-era combined counter; coarser on gfx950

# scheduling hints
fx.rocdl.sched_mfma(N)       # N MFMA before the next barrier
fx.rocdl.sched_vmem(N)       # N VMEM reads
fx.rocdl.sched_dsrd(N)       # N DS reads
fx.rocdl.sched_dswr(N)       # N DS writes
```

Use the **split counters on gfx950**. `s_waitcnt(0)` waits on everything, which serializes loads
against LDS traffic you did not need to wait for — the single most common reason a hand-written
pipeline shows no overlap.

Barriers must be reached by every thread in the workgroup. A barrier inside a runtime `if` deadlocks;
hoist it out.

## Step 6 — Shared memory
```python
from flydsl.utils.smem_allocator import SmemAllocator
from flydsl.compiler.kernel_function import CompilationContext
from flydsl._mlir import ir

allocator = SmemAllocator(None, arch="gfx950", global_sym_name="smem0")
lds_buf   = allocator.allocate_array(fx.T.f16, num_elements)

@flyc.kernel
def kernel_with_lds(A: fx.Tensor, ...):
    lds_ptr = lds_buf(allocator.get_base())

    lds_ptr.store(value, [idx])
    fx.gpu.barrier()
    val = lds_ptr.load([idx])

    # finalize inside the GPU module body, before launch
    comp_ctx = CompilationContext.get_current()
    with ir.InsertionPoint(comp_ctx.gpu_module_body):
        allocator.finalize()
```

**gfx950 LDS is 160 KiB per workgroup across 64 banks.** Both numbers matter:
- 160 KiB means tile sizes that overflowed on CDNA3 now fit — but it also means LDS is rarely the
  binding occupancy limit on gfx950; registers usually are.
- 64 banks means **any padding or XOR swizzle you carried over from a 32-bank design is wrong**.
  Re-derive it. A `+1` pad that removed conflicts at 32 banks does not at 64.

Before adding LDS at all: it only pays when there is **cross-thread reuse**. Staging data that each
thread reads once adds a round trip and a barrier for nothing. See
`../skills/optimize/flydsl_levers/flydsl_authoring_method.md`.

## Step 7 — Run it
```bash
PYTHONPATH=./ python my_kernel.py                      # run
FLYDSL_DUMP_IR=1 PYTHONPATH=./ python my_kernel.py     # dump IR when it misbehaves
```

## Step 8 — Verify
Correctness is not optional at this stage, because tile-programming bugs are layout bugs and layout
bugs do not announce themselves.

```python
torch.cuda.synchronize()                  # required before reading results
assert torch.allclose(Out, reference, atol=1e-5)
```

| Check | Why |
|---|---|
| `torch.cuda.synchronize()` before every result read | otherwise you are asserting on unwritten memory |
| Compare against a torch reference of the same math | not against a previous run of your own kernel |
| Test at a shape that is **not** a multiple of the tile | masking and predication bugs only appear there |
| For GEMM: check with non-symmetric A and B | a symmetric input hides operand-order bugs |

If it is wrong, the classification table in
`../skills/bottleneck/debug-flydsl-kernel.md` maps the symptom to the cause. If it does not compile,
the error → cause → fix table is in **flydsl-kernel-authoring** §10.

## Checklist
- [ ] Pattern identified before writing any code
- [ ] Copy atom width matches the data: `VEC_WIDTH * sizeof(elem) ≤ atom bits`
- [ ] For GEMM: tiles sized to the MFMA instruction shape, and copies built **from** the `tiled_mma`
- [ ] `retile()` applied to every fragment before a copy touches it
- [ ] `Constexpr[int]` for compile-time constants, `Int32` for runtime values
- [ ] `range_constexpr()` wherever the induction variable indexes a Python structure
- [ ] gfx950 split wait counters, not a blanket `s_waitcnt(0)`
- [ ] LDS added only where there is real cross-thread reuse; swizzle re-derived for 64 banks
- [ ] `torch.cuda.synchronize()` before checking results
- [ ] Tested at a non-tile-multiple shape
