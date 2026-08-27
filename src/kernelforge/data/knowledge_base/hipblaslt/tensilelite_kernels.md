# TensileLite Kernel Generation System

## What TensileLite Is

TensileLite is the GEMM kernel generation and selection engine inside hipBLASLt. It produces hand-tuned AMD GCN/CDNA assembly kernels for matrix multiplication (contractions) and ships a pre-built library of solutions alongside a runtime selection mechanism that picks the best kernel for a given problem at dispatch time.

Key directories:
- `tensilelite/Tensile/` -- Python package: kernel writer, solution structs, benchmarking, library logic
- `tensilelite/Tensile/Components/` -- Assembly code-generation components (StreamK, GSU, LocalRead, GlobalWriteBatch, etc.)
- `tensilelite/Tensile/SolutionStructs/` -- Solution and ProblemType data model
- `tensilelite/Tensile/KernelWriter.py` (437 KB) + `KernelWriterAssembly.py` (893 KB) -- the actual assembly codegen
- `library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/` -- pre-built YAML solution libraries per architecture
- `library/src/amd_detail/rocblaslt/src/tensile_host.cpp` -- C++ host-side interface that loads the library and dispatches kernels

Flow: YAML problem config --> benchmarking (optional) --> solution selection --> assembly codegen --> code object (.co) --> runtime library with matching tables.

---

## YAML File Naming Convention

File names encode the problem type using the **Cijk** notation:

```
gfx950_Cijk_Alik_Bljk_BBS_BH_BiasSB_HAS_SAV_UserArgs.yaml
       ^^^^  ^^^^  ^^^^  ^^^  ^^  ^^^^^^  ^^^  ^^^
       |     |     |     |    |   |       |    |
       C_ijk A_lik B_ljk |    |   Bias    HPA  ScaleAlphaVec
                         |    BetaHigh
                     DataTypes: B=bf16, S=fp32 (compute)
```

- `Cijk` = output tensor C indexed by (i, j, k=batch)
- `Ailk` = A transposed (A[i,l,k]), `Alik` = A not transposed
- `Bjlk` = B transposed, `Bljk` = B not transposed
- Data type codes: `H`=fp16, `B`=bf16, `S`=fp32, `D`=fp64, `I8`=int8, `F8`=fp8, `B8`=bf8
- Compound type string `BBS` = input bf16, output bf16, compute fp32
- `BH` = UseBeta + HighPrecisionAccumulate
- `HAS` = ActivationType hipblaslt_all + compute fp32
- `SAV` = ScaleAlphaVec, `SAB` = ScaleAB
- `UserArgs` = supports runtime user arguments (GSU, WGM overrides)

---

## YAML Solution File Structure

Each YAML file is a list with this structure:

```yaml
- {MinimumRequiredVersion: 5.0.0}          # [0] version
- gfx950                                    # [1] architecture name
- gfx950                                    # [2] device name
- [Device 75a0]                             # [3] device ID list
- <ProblemType dict>                         # [4] problem type definition
- [<Solution0>, <Solution1>, ...]            # [5] list of kernel solutions
- [2, 3, 0, 1]                              # [6] index order for matching
- [<matching table entries>]                 # [7] problem-size-to-solution mapping
- null                                       # [8] (reserved)
- null                                       # [9] (reserved)
- DeviceEfficiency                           # [10] performance metric
- Equality|Range|GridBased                   # [11] distance function
```

---

## Solution Organization

### By Architecture

```
Logic/asm_full/
  aldebaran/          # gfx90a (MI210/MI250)
    104CU/            # CU-count variants
    110CU/
  aquavanjaram/       # gfx942 (MI300X/MI300A)
    gfx942/           # full chip
    gfx942_152cu/     # reduced CU configs
    gfx942_228cu/
    ...
  gfx950/             # MI350/MI355X
    gfx950/
    gfx950_id75a3/    # specific device ID variant
  arcturus/           # gfx908 (MI100)
  navi31/navi32/navi33/ gfx1103/ gfx1150-53/ gfx12xx/
```

### By Selection Strategy (subdirectories)

- **Equality/** -- exact problem-size matching. The matching table at the end maps `[M, N, batch, K]` tuples directly to solution indices. Used for known hot sizes (e.g., LLM inference shapes like 12288x12288x1x6144).

- **Range/** -- range-based heuristic matching. Uses min/max bounds on FreeSizeA, FreeSizeB, BoundSize to partition the problem space. The logic tree recursively splits dimensions.

- **GridBased/** -- grid-based matching. Maps (batch, M, N, K) to solutions using a multi-dimensional grid with nearest-neighbor lookup. Default strategy for most configs.

- **Origami/** -- specialized solutions with specific NonTemporal cache hint patterns (e.g., `Origami_nta4/` = NonTemporalA=4). Targets specific memory access patterns.

### Selection Priority

At runtime, the library searches in order:
1. **Equality** (exact match on M, N, K, batch) -- fastest, best accuracy
2. **GridBased** or **Range** (approximate match) -- fallback for unseen sizes
3. Across all matching solutions, `findTopSolutions()` returns the top N candidates

---

## Solution Selection Mechanism

### Host-Side (C++, `tensile_host.cpp`)

```cpp
// Main entry point for getting solutions:
auto solutions = library->findTopSolutions(problem, hardware, requestedAlgoCount);
// Uses MasterSolutionLibrary -> ProblemMatchingLibrary -> MatchingTable

// Users can also request a specific solution by index:
auto solution = library->getSolutionByIndex(problem, hardware, solutionIndex);
```

The `getBestSolutions()` function:
1. Constructs a `ContractionProblemGemm` from hipBLASLt arguments (M, N, K, types, strides)
2. Calls `findTopSolutions()` which traverses the library hierarchy
3. `ProblemMatchingLibrary` uses `MatchingTable::findBestMatch()` or `findTopMatch()`
4. Converts solutions to `rocblaslt_matmul_heuristic_result` with workspace sizes
5. Supports fallback: if no xfloat32 solutions found, retries with fp32

### Python-Side Library Logic (Tuning)

`LibraryLogic.py` analyzes benchmark data to produce the solution tables:
1. Merges solutions from benchmark groups, deduplicating by identity
2. `removeLeastImportantSolutions()` -- drops solutions that contribute < 1% of total time saved
3. `keepWinnerSolutions()` -- alternative: keep only per-size winners
4. Generates **Range Logic** (recursive dimension partitioning) and **Exact Logic** (per-size winners)
5. Outputs YAML files consumed by `TensileCreateLibrary`

### Lazy Loading

The C++ `MasterSolutionLibrary` supports **lazy loading**: solution code objects are split into multiple files mapped by `TensileLiteLibrary_lazy_Mapping.dat`. When a solution index is requested, only its containing file is loaded. This reduces startup time for large libraries (gfx950 BBS Equality alone is 16.8 MB YAML).

---

## Key Kernel Parameters

### MacroTile (MacroTile0 x MacroTile1)

The tile of output matrix C computed by one workgroup.

- `MacroTile0` = tile dimension along M (Free0)
- `MacroTile1` = tile dimension along N (Free1)
- Derived from: `MIWaveTile * MIWaveGroup * MatrixInst{M,N}`
- Typical values: 16x16 (tiny), 64x64, 128x128, 256x256 (large), 224x288 (non-power-of-2)
- Valid sides: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 6, 12, 24, 48, 96, 192, 384, 768

**Tuning impact:**
- Larger tiles = fewer workgroups = better compute efficiency but worse tail effects for small/non-aligned problems
- Smaller tiles = more workgroups = better GPU occupancy for small problems but higher overhead per element
- For square problems (M~N~4096+): 128x128 or 256x256
- For tall-skinny (M>>N or N<<128): small MacroTile1 (16 or 32), large MacroTile0
- For batch GEMM with small M,N: 16x16 or 16x32 with high DepthU

### MIBlock (MatrixInstruction block)

`MIBlock: [MatrixInstM, MatrixInstN, MatrixInstK, MatrixInstB, MatrixInstBM, MatrixInstBN]`

Specifies the hardware MFMA instruction and its blocking:
- `[16, 16, 32, 1, 1, 1]` -- bf16/fp16: MFMA_16x16x32 (most common for gfx950)
- `[32, 32, 16, 1, 1, 1]` -- bf16/fp16: MFMA_32x32x16
- `[16, 16, 128, 1, 1, 1]` -- fp8: MFMA_16x16x128
- `MatrixInstB` = batch dimension of the instruction
- `MatrixInstBM/BN` = blocking factor within the tile

**Tuning impact:**
- 16x16 instructions have higher instruction throughput per CU but need more instructions per tile
- 32x32 instructions cover more output elements per instruction but use more registers
- For gfx950 bf16: `[16, 16, 32, 1, 1, 1]` is the standard choice
- K dimension (32 for bf16, 128 for fp8) determines elements consumed per MFMA

### MIWaveTile and MIWaveGroup

- `MIWaveTile: [waveTileM, waveTileN]` -- how many MFMA tiles each wave computes (e.g., [8, 8])
- `MIWaveGroup: [waveGroupM, waveGroupN]` -- how many waves are grouped along M and N (e.g., [2, 2])

Relationship: `MacroTile0 = MatrixInstM * MatrixInstBM * MIWaveTile[0] * WavefrontSize / MatrixInstN * MIWaveGroup[0]`

- Larger MIWaveTile = more work per wave = fewer waves needed = potentially better register reuse but higher register pressure
- Larger MIWaveGroup = more waves per workgroup = more parallelism within a workgroup

### WorkGroup

`WorkGroup: [WG0, WG1, WG2]`

- WG0 x WG1 = number of threads in the workgroup
- WG2 = LocalSplitU factor (when > 1, splits K across WG2 groups within one workgroup)
- Typical: `[32, 8, 1]` = 256 threads, `[16, 8, 1]` = 128 threads, `[16, 4, 2]` = 128 threads with LSU=2
- NumThreads must be a multiple of WavefrontSize (64)

### WorkGroupMapping (WGM)

Controls how workgroups are mapped to output tiles.

- WGM > 0: row-major mapping with swizzle factor WGM (e.g., WGM=4 maps 4 WGs along M before advancing N)
- WGM = 1: simple row-major
- WGM = 0: column-major
- WGM < 0: special mapping modes

**Tuning impact:**
- Higher WGM improves L2 cache reuse for tall matrices by clustering WGs that share B-tile data
- WGM=1 is safe default; WGM=4-32 can give 5-15% improvement on large square GEMMs
- This parameter can be overridden at runtime via UserArgs (SupportCustomWGM)

### WorkGroupMappingXCC

Maps workgroups across XCCs (chiplets) on multi-die GPUs (MI300X has 8 XCCs, MI355X has 4+).

- WGM_XCC > 0: distributes every WGM_XCC consecutive workgroups across XCCs
- Typical values: 8, 16
- `WorkGroupMappingXCCGroup`: groups of XCCs to treat as one (-1 = auto)

### DepthU (Unroll Depth)

Number of elements along the K (summation) dimension processed per main loop iteration.

- Typical values: 32, 64, 128, 256
- `LoopIters = DepthU / MatrixInstK` (e.g., DepthU=64 with MI_K=32 gives 2 iterations)

**Tuning impact:**
- Larger DepthU = more K elements per loop = fewer loop iterations = less loop overhead
- But: more LDS required (LDS ~ MacroTile0 * DepthU + MacroTile1 * DepthU), may reduce occupancy
- For large K: DepthU=64 is standard; DepthU=128 can help if LDS allows
- For small K (K < 256): smaller DepthU avoids waste in tail loop

### GlobalSplitU (GSU) -- Split-K

Splits the K dimension across multiple workgroups, each computing a partial result that is then reduced.

- GSU=0: disabled, no split-K support compiled into kernel
- GSU=1: no split (default)
- GSU=-1: auto-select at runtime (MIN_K_FOR_GSU = 32)
- GSU=2..1024: explicit split factor

`GlobalSplitUAlgorithm`:
- `SingleBuffer`: atomic accumulation into one buffer
- `MultipleBuffer`: each GSU group writes to separate buffer, reduced by another kernel
- `MultipleBufferSingleKernel`: same kernel does both compute and reduction

**Tuning impact:**
- Essential for tall-skinny problems (large K, small M or N) where there aren't enough output tiles to fill the GPU
- GSU=4-16 typical for K>4096 with M,N<1024
- Requires workspace memory (WorkspaceCheck parameter)
- MultipleBuffer algorithm avoids atomic contention but needs 2 kernel launches

### StreamK

Advanced work distribution that splits work at the granularity of K-iterations rather than whole tiles.

- StreamK=0: disabled
- StreamK=3: enabled with full StreamK algorithm (most common in gfx950 solutions)
- `StreamKAtomic`: 0=non-atomic reduction
- `StreamKXCCMapping`: distributes StreamK work across XCCs (values 0, 6, 8)

**Tuning impact:**
- Better load balancing than data-parallel for non-uniform tile counts
- Especially beneficial when numTiles is not evenly divisible by numCUs
- Adds overhead for synchronization and fixup; not always faster for large, evenly-tiling problems

### DirectToLds / DirectToVgpr

- `DirectToLds=1`: global memory loads write directly to LDS (skipping VGPRs). Saves VGPR pressure, requires aligned access.
- `DirectToVgprA/B=True`: keeps data in VGPRs without going through LDS at all. Maximum register pressure but eliminates LDS bank conflicts. Best for small tiles.

**Tuning impact:**
- DTL is the standard for large tiles on CDNA (gfx90a, gfx942, gfx950)
- DTV is used for specialized small-tile kernels where LDS overhead dominates
- DTL requires `LdsBlockSizePerPad` alignment and specific `NonTemporal` hints

### Vector Widths

- `GlobalReadVectorWidthA/B`: elements loaded per global memory instruction (1, 2, 4, 8). Higher = better bandwidth utilization but requires alignment.
- `LocalReadVectorWidth`: elements read per LDS instruction (typically 8 for bf16)
- `GlobalWriteVectorWidth`: elements stored per global write (1-8)
- `StoreVectorWidth`: output vector width for global store

**Tuning impact:**
- GRVW=8 is optimal for bf16 (8 * 2 bytes = 16 bytes = one cache line fraction)
- For non-aligned problems (M or N not divisible by GRVW), use `EdgeType: ShiftPtr` to handle boundaries
- Smaller VW for small dimensions (M=16: GRVW=4 or less)

### LDS Parameters

- `LdsNumBytes`: total LDS allocation per workgroup (e.g., 131072 = 128 KB for 256x256 tiles)
- `LdsPadA/B`: padding to avoid bank conflicts (0 or 16)
- `LdsBlockSizePerPadA/B`: padding granularity (256, 512, 1024, 4096)
- `1LDSBuffer`: use single LDS buffer instead of double-buffer (saves LDS at cost of pipeline stalls)
- `MaxLDS`: device limit (163840 = 160 KB on gfx950)

**Occupancy relationship:**
- `DeviceLDS / MaxOccupancy` = maximum LDS per workgroup to achieve target occupancy
- On gfx950: 160 KB total LDS, MaxOccupancy=40 waves, so 4 KB per wave for full occupancy
- Large tiles (256x256) with double buffering use 128 KB+ LDS, limiting occupancy to ~1 workgroup per CU

### Prefetch Settings

- `PrefetchGlobalRead`: 0=none, 1=double buffer, 2=triple buffer (prefetch during LDS write), 3+=DTL multi-buffer
- `PrefetchLocalRead`: iterations of LDS-to-VGPR prefetch (0-128, typical 1)
- `ClusterLocalRead`: dedicate VGPR buffers per iteration for scheduling flexibility

**Tuning impact:**
- PGR=2 is the standard for hiding global memory latency
- PLR=1 hides LDS latency with one iteration lookahead
- Higher PGR values need proportionally more LDS (PGR+1 buffers)

### StaggerU

Offsets the starting K position of each workgroup to reduce memory bank conflicts.

- `StaggerU`: number of DepthU steps to stagger (0, 8, 16, 32)
- `StaggerUStride`: stride of the stagger (128, 256)
- `StaggerUMapping`: 0=simple, 1=wg-mapped

**Tuning impact:**
- StaggerU=8-32 with Stride=128-256 reduces TLB thrashing on large GEMMs
- No effect on small problems
- Can be overridden at runtime via UserArgs

### NonTemporal (Cache Hints)

Per-tensor cache behavior hints (gfx950 specific values 0-7):
- `NonTemporalA`: cache hint for A reads
- `NonTemporalB`: cache hint for B reads
- `NonTemporalC`: cache hint for C reads (beta != 0)
- `NonTemporalD`: cache hint for D writes

Values encode L1/L2 cache bypass behavior. The Origami variants specifically tune these for different access patterns.

### Scheduling

- `ScheduleIterAlg`: iteration scheduling algorithm (3 is most common, enables aggressive instruction interleaving)
- `ScheduleGlobalRead`: 1=schedule GR into local read iterations
- `ScheduleLocalWrite`: 1=schedule LW into local read iterations
- `UseCustomMainLoopSchedule`: 0=auto, 1=use custom schedule from Components/CustomSchedule.py

---

## Occupancy and wavesCount

Occupancy = number of waves simultaneously resident on a CU. Determined by:

1. **VGPR usage**: each MFMA output uses VGPRs. More MIWaveTile = more VGPRs per wave.
2. **LDS usage**: `DeviceLDS / LdsNumBytes` limits workgroups per CU. For gfx950: 160 KB / LdsNumBytes.
3. **SGPR usage**: rarely the bottleneck.
4. **MaxOccupancy**: hard cap (default 40 waves per CU).

For a 256x256 tile with 128 KB LDS: only 1 workgroup (4 waves) per CU.
For a 16x32 tile with 30 KB LDS: multiple workgroups per CU, high occupancy.

The `CUOccupancy` parameter in solutions indicates the target occupancy (-1 = not constrained).

---

## Adding Custom Solutions or Overriding Selection

### Runtime Overrides (UserArgs)

Solutions with `SupportUserArgs: true` and `UseUniversalArgs: true` accept runtime parameter overrides:
- `SupportCustomWGM`: override WorkGroupMapping
- `SupportCustomStaggerU`: override StaggerU/StaggerUStride
- `SupportUserGSU`: override GlobalSplitU

These are passed through the hipBLASLt `algo.data` field.

### Adding Solutions to the Library

1. Create a benchmark YAML config with your desired parameter space
2. Run `Tensile/bin/Tensile <config.yaml> <output-dir>` to benchmark
3. Use `TensileCreateLibrary` to compile solutions into code objects
4. Merge with existing library using `TensileMergeLibrary.py`

For iterative tuning:
```bash
# Build and benchmark
cmake --preset tensilelite -S .. -B build
cmake --build build --parallel
./build/Tensile.sh <config>.yaml tensile-out

# Modify assembly directly
# edit tensile-out/.../assembly/<kernel>.s
make co TENSILE_OUT=tensile-out ARCH="gfx950" WAVE=64
```

### Selecting a Specific Solution

```cpp
// In application code via hipBLASLt API:
hipblasLtMatmulAlgoGetHeuristic(handle, matmul, ..., heuristic_results, &algo_count);
// heuristic_results[i].algo.data contains the solution index
// Can override by setting the index directly
```

### Custom Kernels

Place custom assembly in `tensilelite/Tensile/CustomKernels/` with matching `CustomKernelName` in solution config.

---

## Architecture-Specific Variants

### gfx90a (MI210/MI250 -- Aldebaran)

- 104 or 110 CUs
- MFMA: 16x16x16 (bf16), 32x32x8 (bf16)
- 64 KB LDS per CU
- Typical tiles: 128x128, 64x256
- No StreamK support in older variants

### gfx942 (MI300X/MI300A -- Aqua Vanjaram)

- Multiple CU configs: 20, 38, 64, 80, 152, 228 CUs (different SKUs and partitions)
- MFMA: 16x16x32 (bf16), 32x32x16 (bf16)
- 8 XCCs, WorkGroupMappingXCC is critical
- 64 KB LDS per CU
- StreamK support, GSU widely used
- FP8 support (MFMA 16x16x128)

### gfx950 (MI355X)

- MFMA: 16x16x32 (bf16), 32x32x16 (bf16), 16x16x128 (fp8)
- 160 KB LDS per CU (MaxLDS=163840)
- 4+ XCCs, StreamKXCCMapping typically 6 or 8
- Full StreamK=3 as default
- Aggressive NonTemporal cache hints (Origami strategy)
- MX (microscaling) FP4/FP8 support (MXBlockA, MXBlockB, DataTypeMXSA/B)
- `LDSTrInst=true` for some solutions (LDS transpose instructions)
- Largest solution libraries (BBS Equality: 16.8 MB YAML, 29+ solutions)
- PreloadKernArgs=true for all solutions

---

## Tuning Recipes by Problem Shape

### Large Square (M=N=4096+, K=4096+)

- MacroTile: 256x256 or 128x128
- MIWaveTile: [8,8] or [4,4]
- DepthU: 64
- GSU: 0 or 1 (enough tiles to fill GPU)
- StreamK: 3
- StaggerU: 8-32
- WorkGroupMapping: 4-16
- DirectToLds: 1
- PrefetchGlobalRead: 2

### Tall-Skinny (M>>N, e.g., M=32768, N=128, K=4096)

- MacroTile: 256x32 or 128x64
- GSU: 4-16 (critical for K utilization)
- StreamK: 3 with XCCMapping
- Smaller WorkGroup (128 threads)
- Higher DepthU if LDS allows
- StaggerU: 0 (fewer WGs means less conflict)

### Small Batch (M=N=128, batch=64)

- MacroTile: 16x16 or 16x32
- DepthU: 128
- LocalSplitU: 2 (WorkGroup[2]=2)
- DirectToVgpr: consider for very small tiles
- GSU: 0 (batch provides parallelism)
- Low LDS usage for high occupancy

### FP8 / MX Formats

- MatrixInstruction: [16, 16, 128, 1] or [32, 32, 64, 1]
- Higher K throughput per MFMA, so smaller DepthU sufficient
- ScaleAB: Vector or Scalar (adds overhead)
- OutputAmaxD: true if amax tracking needed

---

## Key Files Reference

| File | Description |
|------|-------------|
| `Tensile/KernelWriter.py` | Main kernel code generation (436 KB) |
| `Tensile/KernelWriterAssembly.py` | Assembly-level codegen (893 KB) |
| `Tensile/SolutionStructs/Solution.py` | Solution parameter assignment and validation (226 KB) |
| `Tensile/SolutionStructs/Problem.py` | ProblemType definition and Cijk naming |
| `Tensile/SolutionLibrary.py` | Python-side library model (Matching, FreeSize, Prediction, MLP) |
| `Tensile/LibraryLogic.py` | Benchmark analysis and Range/Exact logic generation |
| `Tensile/Common/ValidParameters.py` | All valid parameter values and MFMA configs |
| `Tensile/Common/GlobalParameters.py` | Default solution parameters |
| `Tensile/Components/StreamK.py` | StreamK + XCC mapping codegen |
| `Tensile/Components/GSU.py` | GlobalSplitU codegen |
| `Tensile/Components/CustomSchedule.py` | Custom main loop scheduling |
| `include/Tensile/MasterSolutionLibrary.hpp` | C++ library root with lazy loading |
| `include/Tensile/MatchingLibrary.hpp` | C++ distance-based solution matching |
| `library/.../tensile_host.cpp` | hipBLASLt <-> TensileLite C++ bridge |
