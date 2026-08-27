# Fly + FlyROCDL MLIR Dialect Reference

> C++ MLIR layer that backs the Python DSL. Source under
> `include/flydsl/Dialect/` (TableGen) and `lib/Dialect/` (C++ impls).

## 1. Types (`FlyTypeDefs.td`)

| MLIR Type | Description | Lines |
|---|---|---|
| `!fly.int_tuple` | Hierarchical integer tuple with nested BasisAttr | 27–44 |
| `!fly.basis` | Leaf integer + mode hierarchy depth | 10–25 |
| `!fly.layout` | (Shape IntTuple, Stride IntTuple); `isStaticShape/Stride` predicates | 46–69 |
| `!fly.swizzle` | Bank swizzle (mask, base, shift) — `isTrivialSwizzle()` | 71–82 |
| `!fly.coord_swizzle` | 2D coord-swizzle for LDS bank patterns | 84–95 |
| `!fly.composed_layout` | inner + offset + outer composition | 97–121 |
| `!fly.tile` | Tiling-mode descriptor | 123–141 |
| `!fly.ptr` | Typed pointer + AddressSpace + AlignAttr + SwizzleAttr | 143–169 |
| `!fly.coord_tensor` | Tensor view = base IntTuple + Layout | 171–188 |
| `!fly.memref` | elemTy + AddressSpace + Layout + AlignAttr + SwizzleAttr | 190–228 |
| `!fly.tiled_copy` | copy_atom + Layout (thr/val) + Tile | 230–239 |
| `!fly.tiled_mma` | mma_atom + thr Layout + optional Tile | 241–254 |
| `!fly.copy_atom` | copyOp Type + valBits; `emitAtomCall` builders | 256–295 |
| `!fly.mma_atom` | mmaOp Type; `getThrLayout/ShapeMNK/ValType{A,B,C}` | 297–333 |
| `!fly.universal_copy<N>` | bitSize-parameterized CopyOp type | 335–343 |
| `!fly.universal_atomic` | atomicOp (Add/Max/Min/And/Or/Inc/Dec) + valType | 345–360 |
| `!fly.universal_fma` | elemTy-parameterized FMA fallback | 362–374 |

AddressSpace enum (FlyAttrDefs.td:17–64): `Generic` (0), `Global` (1),
`Shared` (3), `Register` (5), `BufferDesc` (8 — internal).

## 2. Attributes (`FlyAttrDefs.td`)

| Attribute | Purpose |
|---|---|
| `Fly_AlignAttr` | byte alignment; `getTrivialAlignment` builder |
| `Fly_IntAttr` | hybrid static/dynamic; `isStaticValue`, `getDynamic(width, divisibility)` |
| `Fly_BasisAttr` | value + modes (int32 array); `getStatic` |
| `Fly_SwizzleAttr` | (mask, base, shift); `period() = 1 << (mask+base+shift)` |
| `Fly_CoordSwizzleAttr` | separate row/col swizzles |
| `Fly_IntTupleAttr` | nested IntTuple; `getLeafAs{Int,Basis}`, `at()` |
| `Fly_LayoutAttr` | (shape, stride); `isStaticShape/Stride` |
| `Fly_ComposedLayoutAttr` | (inner, offset, outer) |
| `Fly_TileAttr` | nested mode value |
| `Fly_ExplicitModuleAttr` | OffloadingTranslationAttr — embeds compiled GPU binary into host IR |
| `Fly_CachePolicy` enum | `CacheGlobal` / `CacheAlways` |
| `Fly_MmaOperand` enum | A / B / C / D |
| `Fly_AtomicOp` enum | Add/Max/Min/And/Or/Inc/Dec |
| `Fly_GemmTraversalOrder` | 12 modes (KMN, MNK, …, + Serpentine variants) |

## 3. Fly Op Catalog (`FlyOps.td`)

**Construction** (42–108): `fly.static`, `fly.make_int_tuple`, `fly.make_shape`,
`fly.make_stride`, `fly.make_coord`, `fly.make_layout`, `fly.make_layout_like`,
`fly.make_ordered_layout`, `fly.make_identity_layout`,
`fly.make_composed_layout`, `fly.make_view`, `fly.make_fragment_like`,
`fly.make_fragment_layout_like`.

**Extractors** (114–158): `fly.get_scalar`, `fly.get_leaves`, `fly.get_shape`,
`fly.get_stride`, `fly.get_layout`, `fly.get_iter`,
`fly.composed_get_{inner,offset,outer}`.

**IntTuple arithmetic** (164–196): `fly.int_tuple_{add,sub,mul,div,mod}`,
`fly.int_tuple_product`, `fly.int_tuple_product_each`,
`fly.int_tuple_product_like`, `fly.shape_div`, `fly.ceil_div`,
`fly.elem_less`, `fly.equal`.

**Layout algebra** (202–336):
- Structural: `fly.get`, `fly.take`, `fly.select`, `fly.group`, `fly.append`, `fly.prepend`.
- Coordinate: `fly.slice`, `fly.dice`, `fly.crd2idx`, `fly.idx2crd`,
  `fly.get_flat_coord`, `fly.get_1d_coord`.
- Queries: `fly.size`, `fly.coprofile`, `fly.coshape`, `fly.cosize`.
- Transforms: `fly.coalesce`, `fly.composition`, `fly.complement`,
  `fly.right_inverse`, `fly.left_inverse`, `fly.recast_layout`, `fly.tile_to_shape`.
- Partitioning: `fly.logical_divide`, `fly.zipped_divide`, `fly.tiled_divide`, `fly.flat_divide`.
- Composition: `fly.logical_product`, `fly.zipped_product`, `fly.tiled_product`,
  `fly.flat_product`, `fly.blocked_product`, `fly.raked_product`.

**Atoms + Copy/MMA** (342–427):
- `fly.make_copy_atom`, `fly.make_mma_atom`, `fly.atom.set_value`.
- `fly.copy_atom_call`, `fly.mma_atom_call` (memref-level, blocking).
- `fly.copy_atom_call_ssa`, `fly.mma_atom_call_ssa` (SSA values).
- `fly.make_tiled_copy`, `fly.make_tiled_mma`.
- `fly.tiled_copy.partition_src`, `fly.tiled_copy.partition_dst`, `fly.tiled_copy.retile`.
- `fly.tiled_mma.partition`, `fly.tiled_mma.partition_shape`,
  `fly.mma.make_fragment`.
- `fly.copy` (high-level tiled copy, optional `traversalLayout`).
- `fly.gemm` (high-level tiled MMA).

**Pointers / memrefs** (433–495):
- `fly.make_ptr` (register / shared / buffer_rsrc), `fly.get_dyn_shared`.
- `fly.inttoptr`, `fly.ptrtoint`, `fly.add_offset`, `fly.apply_swizzle`.
- `fly.decomposition`, `fly.ptr.{load,store}`, `fly.recast_iter`.
- `fly.memref.{alloca,load_vec,store_vec,load,store}`.

**Utilities** (501–519): `fly.print`, `fly.assume`,
`fly.extract_aligned_pointer_as_index` (deprecated).

## 4. FlyROCDL Dialect — Target Atoms

**MMA atoms** (`FlyROCDL/IR/MmaAtom.td:13-91`):

| Op | Variant | What |
|---|---|---|
| `fly_rocdl.cdna3.mfma` | CDNA3 baseline | `(m, n, k, elemTyA, elemTyB, elemTyAcc)`. Supported shapes catalogued in `lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp:98-222`: 4×4×1, 16×16×{1,2,4,8,16,32,64}, 32×32×{1,2,4,8,16,32} |
| `fly_rocdl.cdna4.mfma_scale` | CDNA4 scaled | adds `opselA`, `opselB` for low-precision scaling (MXFP4/FP6/FP8) |
| `fly_rocdl.gfx1250.wmma` | RDNA3 wave32 | WMMA instruction set; wave32-compatible |

**Copy atoms** (`FlyROCDL/IR/CopyAtom.td:9-44`):

| Op | What |
|---|---|
| `fly_rocdl.cdna3.buffer_copy` | CDNA3 buffer load/store via buffer descriptor |
| `fly_rocdl.cdna3.buffer_copy_lds` | CDNA3 buffer → LDS (direct DMA via `buffer_load_lds`) |
| `fly_rocdl.cdna3.buffer_atomic` | atomic via buffer descriptor (Add/Max/Min/...) |
| `fly_rocdl.cdna4.lds_read_trans` | CDNA4 `DS_READ_B{64,96}_TR_B{4,8,16,6}` — transpose-load from LDS to VGPR; eliminates explicit transpose for MFMA operand prep |

## 5. Pass Catalog

`include/flydsl/Dialect/Fly/Transforms/Passes.td`:

| Pass | Lines | What it does |
|---|---|---|
| `fly-rewrite-func-signature` | 9–30 | Lowers DSL types to packed LLVM structs; reconstructs static sub-components via `fly.static`; removes fully-static args |
| `fly-canonicalize` | 32–37 | Algebraic simplification of fly ops (constant folding, identity layouts, etc.) |
| `fly-layout-lowering` | 39–50 | Lowers `fly.crd2idx`, `fly.logical_divide`, etc. to `arith`/`vector`/`gpu` |
| `fly-convert-atom-call-to-ssa-form` | 52–63 | memref-level `copy_atom_call`/`mma_atom_call` → SSA form; promotes register tensors to vector |
| `fly-int-swizzle-simplify` | 65–88 | Recognizes canonical swizzle bit patterns; extracts peelable addends for CSE |
| `fly-promote-regmem-to-vectorssa` | 90–104 | Promotes `fly.make_ptr(register)` semantics to vector SSA (requires the SSA conversion pass first) |

`include/flydsl/Conversion/FlyToROCDL/Passes.td`:

| Pass | Lines | What it does |
|---|---|---|
| `convert-fly-to-rocdl` | 6–15 | Lowers Fly to ROCDL/LLVM intrinsics. Depends on `arith`, `scf`, `vector`, `llvm`, `rocdl`. |
| `fly-rocdl-cluster-attr` | 17–30 | Injects `rocdl.cluster_dims` into `llvm.func` passthrough for upstream ROCDL compatibility (gfx950 cluster mode) |

### Key lowering patterns in `convert-fly-to-rocdl` (`lib/Conversion/FlyToROCDL/FlyToROCDL.cpp:88-230`)

- **MakePtrOpLowering**:
  - `Register` → `llvm::AllocaOp(elemTy, nElems)`
  - `Shared` → `createLDSGlobal()` + `llvm::AddressOfOp`
  - `BufferDesc` → `rocdl::MakeBufferRsrcOp` + `BufferFatPtr::pack`
- **GetDynSharedOpLowering**: re-uses or creates `[0 x i8]` `__dynamic_shared`
  global at module scope
- AddressSpace → LLVM AS mapping: Generic→0, Global→1, Shared→3, Register→5,
  BufferDesc→8

### MFMA dispatch
`lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp:160-194` uses `DISPATCH_MFMA_SSA` macros
to lower `fly_rocdl.cdna3.mfma` to the right `rocdl.mfma_*` intrinsic:
`mfma_f32_{4,16,32}×{4,16,32}×{1,2,4,8,16,32}_{f32,f16,bf16}`,
`mfma_f32_{16,32}×{16,32}×{16,32}_{fp8,bf8}_{fp8,bf8}`.

## 6. CAPI / Python Bindings

| File | Provides |
|---|---|
| `lib/CAPI/Dialect/Fly/FlyDialect.cpp` | Registers Fly dialect, all passes (`FlyRewriteFuncSignature`, `FlyCanonicalize`, …), LLVM offloading translation |
| `lib/CAPI/Dialect/FlyROCDL/FlyROCDLDialect.cpp` | Registers FlyROCDL dialect |
| `lib/Bindings/Python/FlyExtension.cpp` | `IntTupleAttrBuilder` (hybrid static/dynamic), `rank()`, `depth()`, `isProfileCongruent`, `isProfileWeaklyCongruent`, `getAddressSpaceFromObj()`, nanobind exposure of type constructors and queries |
| `lib/Bindings/Python/TiledOpTraits.h` | Trait dispatch for tiled copy/MMA partition operations |
| `lib/Bindings/Python/DLTensorAdaptor.h` | Buffer-protocol bridge: numpy/torch → HIP device memory |

## 7. Runtime Wrappers (`lib/Runtime/ROCm/FlyRocmRuntimeWrappers.cpp`)

| Symbol | Purpose |
|---|---|
| `mgpuModuleLoad(data)` | `hipModuleLoadData()` |
| `mgpuModuleUnload(module)` | `hipModuleUnload()` |
| `mgpuModuleGetFunction(module, name)` | `hipModuleGetFunction()` |
| `mgpuLaunchKernel(fn, gx,gy,gz, bx,by,bz, smem, stream, params, extra)` | `hipModuleLaunchKernel()` |
| `mgpuLaunchClusterKernel(...)` | `hipDrvLaunchKernelEx` with `hipLaunchAttributeClusterDimension` (gfx950 cluster mode) |

All wrappers go through `HIP_REPORT_IF_ERROR`, which prints
`hipGetErrorName(err)` on failure.

## 8. fly-opt Tool

`tools/fly-opt/` is the dedicated MLIR opt tool with all Fly passes registered.
Useful for FileCheck tests under `tests/mlir/` and for hand-running individual
passes during pass development:

```bash
fly-opt input.mlir --fly-canonicalize --fly-layout-lowering \
                   --convert-fly-to-rocdl --canonicalize
```
