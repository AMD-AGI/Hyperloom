# High-Performance AMD GPU ASM Kernels in aiter-amd

A synthesized knowledge base mined by 8 parallel agents across the toolchain, kernel families, dispatch pipeline, and CK_tile template reference. Every claim is cross-referenced to source — paths are relative to `aiter-amd/`.

---

## 1. What ships in this repo

This repo does **not** check in `.s` assembly source. It ships:

1. **Pre-compiled HSACO code-objects** (`.co` ELF files) in [hsa/gfx942/](hsa/gfx942/) (CDNA3 / MI300X) and [hsa/gfx950/](hsa/gfx950/) (CDNA4 / MI350X).
2. **CSV metadata** per kernel family ([hsa/gfx942/pa/pa_asm.csv](hsa/gfx942/pa/pa_asm.csv), [hsa/gfx942/mla/mla_asm.csv](hsa/gfx942/mla/mla_asm.csv), [hsa/gfx942/topksoftmax/topksoftmax.csv](hsa/gfx942/topksoftmax/topksoftmax.csv), and per-dtype gemm CSVs).
3. **A CSV→C++ codegen** ([hsa/codegen.py](hsa/codegen.py)) that emits dispatch tables consumed by C++ host code.
4. **A round-trip ISA toolchain** ([docs/isa_kernel_optimization.md](docs/isa_kernel_optimization.md), 475 lines) plus runnable scripts in [docs/examples/isa_optimization/](docs/examples/isa_optimization/) to disassemble, edit, and recompile any `.co`.
5. **CK_tile** (3rdparty/composable_kernel) — the readable C++ template DSL that encodes the same patterns the hand-tuned `.co` files implement.

Kernel families on disk:

| Family | Location | Variants | Purpose |
|---|---|---|---|
| **PA (Paged Attention)** | `hsa/gfx{942,950}/pa/` + `hsa/gfx{942,950}/pa_*.co` | ~55 (942) + ~52 (950) | decode + prefill attention with paged KV cache |
| **MLA** | `hsa/gfx{942,950}/mla/` | 23 (942) / 34 (950) | DeepSeek-V2/V3 multi-head latent attention |
| **FMOE** | `hsa/gfx{942,950}/fmoe_*.co` | ~30 binaries × ~1000 config rows | Fused MoE GEMM-A + act + GEMM-B |
| **GEMM** | `hsa/gfx942/{bf16,f4,i8,fp8gemm_blockscale}/`, `flatmm_uk_*.co`, `gemm_a8w8_*.co` | tens per dtype | Tuned per-shape GEMMs |
| **f8 block-scale (MI350)** | `hsa/gfx950/f8_block_scale_mi350_x{32,64,96,128}.co` | 4 | DeepSeek-V3 style block-scaled fp8 GEMM |
| **gemv_router** | `hsa/gfx950/gemv_router.co` | 1 | MoE gate-projection GEMV (M≤8, N=256, K=3072) — **35.6× hipBLASLt** per [aiter/ops/gemv_router.py:1-2](aiter/ops/gemv_router.py) |
| **TopK-softmax** | `hsa/gfx942/topksoftmax/` + `topk_per_row_{decode,prefill}/` | 23 + 2 | Pre-FMOE expert selection |
| **AllReduce** | `hsa/gfx942/all_reduce.co`, `allreduce_{layernorm,rmsnorm,rmsnorm_qnt}_N8192.co` | 4 | Hand-tuned XGMI ring + fused post-attn norm at hidden=8192 |

---

## 2. The round-trip ISA toolchain

There is no `.s` source — but every `.co` can be **disassembled, edited, recompiled, and re-injected** without losing HIP-runtime-required metadata. This is the workflow documented in [docs/isa_kernel_optimization.md](docs/isa_kernel_optimization.md) and automated by [docs/examples/isa_optimization/roundtrip.sh](docs/examples/isa_optimization/roundtrip.sh).

### 2.1 Pipeline

```
kernel.co
  ├── llvm-objdump -d --mcpu=gfx942 ──► kernel.isa
  │                                       │
  │                              extract_asm.py ──► kernel.s  (reassemblable)
  │                                                 │
  │                                          (edit here)
  │                                                 │
  │                                clang++ -x assembler ──► kernel_recompiled.co
  │                                                          │
  │                              llvm-objcopy -O binary -j .text ──► recompiled_text.bin
  │                                                                    │
  └── cp ──► kernel_modified.co ◄── llvm-objcopy --update-section .text=recompiled_text.bin
                  │
           (loadable, retains original .note metadata)
```

Verified round-trip on a PA decode kernel showed ±3 % noise vs original ([docs/isa_kernel_optimization.md:231-239](docs/isa_kernel_optimization.md)) — the assembler produces bit-identical `.text` from the extracted `.s`.

### 2.2 Why raw `llvm-objdump` is NOT reassemblable, and what extract_asm.py fixes

Raw disassembly has four flaws the script repairs:

| Flaw | Fix in [extract_asm.py](docs/examples/isa_optimization/extract_asm.py) |
|---|---|
| Branch targets shown as `label_XXXXX` are **word-offsets**: `target_addr = base + label * 4` | Resolve to real addresses, emit `label_XXXXX:` directives at the target lines |
| Trailing `// hex` annotations are not valid asm syntax | `re.sub(r"\s*//\s*[0-9A-Fa-f]+$", "", instr)` |
| Missing `.amdgcn_target "amdgcn-amd-amdhsa--gfx942"` directive | Inserted into the preamble |
| Missing `.globl / .type / .size symbol, .-symbol` directives | Emitted around the kernel symbol |

### 2.3 Why metadata-section preservation matters

`clang++ -x assembler` regenerates a minimal `.note` section. The HIP runtime needs the **original `.note` AMDHSA metadata** (SGPR / VGPR / AGPR counts, LDS size, wavefront size, argument descriptors) — otherwise `hipModuleLoad` fails with "no kernel image available." Thus the canonical workflow is to copy the original `.co` and **only swap the `.text` section in** via `llvm-objcopy --update-section`.

### 2.4 Instruction-mix analysis with [analyze_kernel.py](docs/examples/isa_optimization/analyze_kernel.py)

The script counts 10 instruction families and computes a **compute-to-memory ratio**:

| Family | What it tells you |
|---|---|
| `v_mfma_*` | Matrix compute; high count → compute-bound. Each MFMA fuses ~2000 FLOPs with 64–128-cycle latency. |
| `buffer_load_*` | Global memory reads via buffer descriptors (hw bounds-checked on gfx9 — OOB silently returns 0) |
| `global_load_*` | Flat-address loads (less common; no hw bounds check) |
| `ds_read_*`, `ds_write_*` | LDS (shared memory); high count → cooperative tiling. 32 banks; conflicts serialize. |
| `*_dpp` | Cross-lane permute within a wave (reductions, transposes, broadcasts) |
| `s_*` | Scalar/control flow — high count = high control overhead |
| `v_*` (non-MFMA) | Vector ALU |
| `s_waitcnt`, `s_nop` | Wait states — high % means missed ILP opportunity |

A "well-optimized attention kernel" on gfx942 typically shows ~192 MFMA, ~100 buffer_load, ~300 ds_, ~300 DPP, ~200 scalar — compute-to-memory ratio ≈ 1.9 ([docs/isa_kernel_optimization.md:446-457](docs/isa_kernel_optimization.md)).

### 2.5 rocprofv3 — closing the loop

```bash
rocprofv3 --kernel-trace -d ./profile_out -- python benchmark.py
```

Produces a SQLite DB with `kernel_dispatch` (per-launch timings) joined to `kernel_symbol` (which exposes `arch_vgpr_count`, `accum_vgpr_count`, `sgpr_count`, `group_segment_size`). Standard query in [docs/isa_kernel_optimization.md:322-337](docs/isa_kernel_optimization.md) ranks kernels by avg duration. For instruction-level traces, `rocprofv3 --att --att-target-cu 1` produces Advanced Thread Trace; the [Dockerfile](docs/examples/isa_optimization/Dockerfile) builds `rocprof-trace-decoder` from source because it is not in ROCm 7.2.x.

---

## 3. The codegen + dispatch pipeline

This is the connective tissue. CSV → generated C++ header → host dispatch → `hipModuleLaunchKernel`.

### 3.1 CSV → header ([hsa/codegen.py](hsa/codegen.py))

`codegen.py` reads per-family CSVs (one or more per arch), merges rows, infers column types (numeric → `int`, else `std::string`), and emits a header containing:

1. A struct (e.g. `i8gemmConfig { knl_name; co_name; arch; tile_m; tile_n; splitK; bpreshuffle; }`)
2. A `using CFG = std::unordered_map<std::string, ThatStruct>;`
3. An `ADD_CFG(...)` macro
4. One `static CFG cfg_<family> = { ADD_CFG(...), ... };` per CSV

Output for the `gemm_a8w8` family lives at [aiter/jit/build/module_gemm_a8w8_asm/blob/asm_i8gemm_configs.hpp](aiter/jit/build/module_gemm_a8w8_asm/blob/asm_i8gemm_configs.hpp). The map key is **`arch + knl_name`** (e.g. `"gfx950_ZN5aiter41I8gemm_..."`), allowing the same mangled symbol on multiple arches without collision.

### 3.2 Runtime loader

The kernel-loading class is `AiterAsmKernel` in [csrc/include/aiter_hip_common.h](csrc/include/aiter_hip_common.h) (or `aiter_hip_common_hip.h`). Two modes:

- **AITER_ASM_DIR env var set** → `mmap` `${AITER_ASM_DIR}/${arch}/${hsaco_path}` and register via `__hipRegisterFatBinary` / `__hipRegisterFunction` with the mangled `knl_name`.
- **Otherwise** → lookup in compile-time `AITER_EMBEDDED_HSA_MAP[std::string("hsa/") + arch + "/" + co_name]`. (Embedded mode lets you ship a single `.so` without external files.)

Architecture is detected via `hipGetDeviceProperties().gcnArchName`, stripping any `:sramecc+xnack-` suffix.

Launch path:

```cpp
hipFunction_t f = nullptr;
hipGetFuncBySymbol(&f, this);                  // 'this' was registered as the func handle
hipModuleLaunchKernel(f, gdx,gdy,gdz, bdx,bdy,bdz, 0, stream, nullptr, config);
```

### 3.3 Heuristic kernel selection

Each family has a host-side selector that iterates `CFG*`, filters by `arch`, then picks the kernel that **minimizes CU rounds** (load balance), tiebreaks on **empty-CU count** (waste), then on the kernel's `ps` (persistent-scheduling) flag. Examples:

- FMOE: [csrc/py_itfs_cu/asm_fmoe.cu:229-306](csrc/py_itfs_cu/asm_fmoe.cu) — for `(inter_dim, sub_X_cnt)` finds best `subGU_n ∈ {128,192,256,320,384,448,512}` with matching `vskip`, `smf`, `subGU_m`.
- GEMM a8w8: [csrc/py_itfs_cu/asm_gemm_a8w8.cu:57-122](csrc/py_itfs_cu/asm_gemm_a8w8.cu) — filters by `bpreshuffle`, `splitK`, `N % tile_n == 0`.
- MLA: [csrc/py_itfs_cu/asm_mla.cu:240-355](csrc/py_itfs_cu/asm_mla.cu) — special-cases gqa=128 on gfx942 (force `ps=0`), pads `max_seqlen_q` to nearest supported, and on gfx950 with `(gqa*max_seqlen_q) % 128 == 0` folds the gqa ratio into `s_Q_Bs` and uses a `gqa=32` kernel.

### 3.4 Python → C++ binding

The `@compile_ops` decorator in [aiter/](aiter/) declares the C++ function, FFI type (usually `ctypes`), and the JIT module name. Example:

```python
@compile_ops("module_gemm_a8w8_asm", fc_name="gemm_a8w8_asm", ffi_type="ctypes")
def _gemm_a8w8_asm(...): ...
```

The "JIT module" is **not** runtime-JIT'd — it's a pre-built `.so` containing the dispatch C++ compiled against the generated config header. Kernel binaries (`.co`) are kept separate and loaded at launch time. This decoupling lets engineers swap `.co` files (e.g. an ISA-edited variant) without rebuilding the `.so`.

### 3.5 Adding a new ASM kernel — step-by-step

1. Write `.s` (or disassemble + edit an existing kernel), compile to `.co` with `clang++ -x assembler -target amdgcn-amd-amdhsa -mcpu=gfxXXX`.
2. Drop into `hsa/<arch>/<family>/your_kernel.co`.
3. Append a row to the family's CSV with **mangled `knl_name`** and `co_name`.
4. Rerun `hsa/codegen.py` to regenerate the dispatch header.
5. No C++ changes needed if the family already exists — the selector auto-discovers the new kernel from the CFG map.
6. Rebuild the C++ extension; ship.

---

## 4. The kernel families decoded

### 4.1 Paged Attention (PA)

CSV columns ([hsa/gfx942/pa/pa_asm.csv](hsa/gfx942/pa/pa_asm.csv)): `qType, kvType, Gqa, Mtp, Msk, Hp, blkSz, knl_name, co_name, ps, qTile, quant_type`.

Name tokens (e.g. `pa_a16w8_bf16_2tg_g8_f8_gemm1_bf16`):

- `a16` / `w8` — Q is 16-bit (bf16/fp16), KV cache is 8-bit (fp8/int8)
- `2tg` / `1tg` — two cooperating thread-groups vs one. 2tg overlaps K-prefetch in tg-0 with V-prefetch in tg-1.
- `g8` / `g16` — GQA ratio
- `f8` / `i8` — KV quant scheme (per-token fp8 vs int8)
- `gemm1_bf16` — GEMM1 (Q·K) variant with bf16 intermediate (better fp8 precision)
- `tail_bf16` — tail-refinement path for masked tokens
- `qlen{16,32,40,48,64}` — query-tile size (prefill blocking)
- `blk{256,1024}` — KV page size
- `msk{0,1}` — causal-mask off / on
- `_ps` — persistent-scheduling flag (`ps=1` in CSV)
- `Hp ∈ {0,1,2}` — standard / hp / uhp fp8 precision modes

There are **~55 PA variants on gfx942 and ~52 on gfx950**. Each is a separate `.co`, picked at runtime. The reason for so many specialized variants rather than one generic kernel:

- Register allocation is hand-tuned per shape — no compiler-induced spill
- LDS layout differs per quant (per-token scales broadcast differently than per-block scales)
- 1tg vs 2tg uses different barrier patterns
- MFMA instruction choice (`mfma_f32_16x16x16` vs `16x16x32`) depends on K-pack width

Python entry: [aiter/ops/attention.py:129-190](aiter/ops/attention.py) — heuristic `_should_use_asm_kernel` gates on `head_size == 128`, `kv_cache_dtype`, `high_precision`, and `total_heads > 2*cu_count`. C++ dispatch parallels MHA's path at [csrc/cpp_itfs/mha_fwd.cu:208-268](csrc/cpp_itfs/mha_fwd.cu).

Device-variant subpath: MI300 vs MI308 distinguished by `get_pci_chip_id()`, kernel loaded from `pa/MI300/...co` or `pa/MI308/...co` accordingly ([csrc/cpp_itfs/mha_fwd.cu:63-79](csrc/cpp_itfs/mha_fwd.cu)).

### 4.2 MLA (DeepSeek V2/V3)

CSV: [hsa/gfx942/mla/mla_asm.csv](hsa/gfx942/mla/mla_asm.csv) — columns include `qType, kvType, Gqa, ps, qSeqLen, prefill, causal, lse, knl_name, co_name`.

MLA needs its own kernel family because KV is stored as **LoRA-decompressed packed tensors** `[num_blocks, num_kv_heads=1, kv_lora_rank + scale_dim + qk_rope_head_dim, block_size]`, not the standard `[B, H_kv, head_size, page]` PA layout. Decompression happens inside the kernel.

Variant name `mla_a16w16_qh16_m16x4_n16x1_coex0_mask1_ps`:

- `qh16` — query head-dim sub-tile = 16
- `m16x4 / n16x1` — workgroup tile in M (seqs) and N (output) with iteration depths
- `coex0` — cooperative-load exponent (LDS pipelining depth)
- `mask1` — causal/dynamic masking on
- `_ps` — persistent kernel

Dispatch ([csrc/py_itfs_cu/asm_mla.cu:262-353](csrc/py_itfs_cu/asm_mla.cu)) does shape-aware key construction; on gfx950 with `(gqa_ratio * max_seqlen_q) % 128 == 0` it folds the gqa axis into the metadata and reuses a `gqa=32` kernel.

### 4.3 FMOE (Fused MoE)

~30 binaries × ~1000 config rows in [aiter/jit/build/module_moe_fmoe_asm/blob/asm_fmoe_configs.hpp](aiter/jit/build/module_moe_fmoe_asm/blob/asm_fmoe_configs.hpp).

Name `fmoe_fp8_g1u1_multix_subGU_256`:

- `g1u1` — fused gate+up (W1 is `[E, 2*inter_dim, dim]`); detected by `w2.shape[2] * 2 == w1.shape[1]` in [aiter/fused_moe_bf16_asm.py:681-683](aiter/fused_moe_bf16_asm.py).
- `g1u0` — gate only
- `multix` / `smf` / `novs` — three dispatch strategies, encoded as `smf={0|1|2}` and `vskip={0|1}` in the config row:
  - **multix** (`smf=2`): multi-expert cooperative LDS share, higher LDS BW for higher occupancy
  - **smf** (`smf=1`): uses SMFMAC sparse-matrix MFMAs when geometry allows
  - **novs** (`vskip=0`): disables late-store optimization; simpler addressing for blockscale variants
- `subGU_{128..512}` — the **N-tile** for the gate/up GEMM (inter_dim blocking)
- `pertokenFp8` / `blockscale` / `int4fp8` / `int8` / `b16` / `f16` — quant scheme; `blockscale` is hardcoded to `(128, 128)` block in [aiter/fused_moe_bf16_asm.py:141-144](aiter/fused_moe_bf16_asm.py).

A single `.co` performs steps (c)–(f) of the MoE pipeline:

```
[host CK kernel] topk_softmax → moe_align_block_size → scatter (produces sorted_token_ids, sorted_weights, sorted_expert_ids)
                                                            │
                                                            ▼
[one .co kernel]  load input[sorted_ids[m]]
                   → GEMM-A: input × W1 (fp8 mfma_f32_16x16x32_fp8 in inner loop)
                   → act + (gate*up if g1u1)
                   → GEMM-B: hidden × W2
                   → gather/atomicAdd via sorted_weights[expert]
```

Selector at [csrc/py_itfs_cu/asm_fmoe.cu:229-306](csrc/py_itfs_cu/asm_fmoe.cu) sweeps `subGU` choices for minimum `(inter_dim/subGU * sub_X_cnt + num_cu - 1)/num_cu` (CU rounds), tiebreaks on empty-CU count, prefers `ps=1`.

### 4.4 GEMM family

- **`gemm_a8w8_m128_{noSplitK, splitK}.co`** — int8 × int8 → bf16 GEMM. Split-K activated when `cusPerTile >= 2^(splitK+1)` and `2^(splitK+1) * tile_k < 2*K` ([aiter/ops/gemm_op_a8w8.py:337-345](aiter/ops/gemm_op_a8w8.py)), recovering CU utilization on skinny-M shapes.
- **`flatmm_uk_gfx9_f16f8_128x256x128_1x4x1_16x16x32.co`** — persistent-grid f16/bf16 × fp8 GEMM. Tokens decode: M small, N large. Tile 128×256×128, warps 1×4×1, MFMA shape 16×16×32 → uses `v_mfma_f32_16x16x32_f8` (inferred).
- **`f8_block_scale_mi350_x{32,64,96,128}.co`** — gfx950 / MI350 block-scaled fp8 GEMM; x{N} is the BlockM. Weight scale stored as `[N/bs, K/bs]` and dequantized per-block during the inner K loop.
- **`bf16gemm_fp32bf16_*.co`** — CSV columns `tn, tileM, tileN, pf (prefetch stages), bPreshuffle, splitK, subK, bias` ([hsa/gfx942/bf16gemm/bf16gemm_fp32bf16.csv](hsa/gfx942/bf16gemm/bf16gemm_fp32bf16.csv)).
- **`gemv_router.co`** — gfx950 only. M ≤ 8, N = 256, K = 3072 bf16 MoE-router GEMV. 256 workgroups × 256 threads → fully saturates 120-CU gfx950. **35.6× faster than `hipBLASLt.torch.mm` for this shape** ([aiter/ops/gemv_router.py:1-2](aiter/ops/gemv_router.py)).

### 4.5 Allreduce + fused norm

Three fused kernels at hidden=N=8192:

- [hsa/gfx942/allreduce_layernorm_N8192.co](hsa/gfx942/allreduce_layernorm_N8192.co)
- [hsa/gfx942/allreduce_rmsnorm_N8192.co](hsa/gfx942/allreduce_rmsnorm_N8192.co)
- [hsa/gfx942/allreduce_rmsnorm_qnt_N8192.co](hsa/gfx942/allreduce_rmsnorm_qnt_N8192.co) — with output fp8 quantization

The trick: while the XGMI ring is still rotating chunks between peers, the kernel **computes the (residual + norm) on already-arrived chunks** — hiding the post-attention norm cost in the all-reduce bubble. Activated only when `hidden_size == 8192` and dtype ∈ {bf16, fp16}; otherwise fall through to the generic `quick_all_reduce` ([csrc/kernels/quick_all_reduce.hip](csrc/kernels/quick_all_reduce.hip)) which supports world_size ∈ {2,4,8}, codecs `{FP, Q4, Q6, FP8}`, and bf16/fp16.

Plain `all_reduce.co` is a hand-tuned ring (no fused norm); `quick_all_reduce` is the source-form alternative covering arbitrary shapes/quant codecs.

### 4.6 TopK-softmax

Split per phase:

- [hsa/gfx942/topk_per_row_decode/asm_top_k_per_row_decode.co](hsa/gfx942/topk_per_row_decode/) (30 KB) — 1 token per block, radix-sort in LDS
- [hsa/gfx942/topk_per_row_prefill/asm_top_k_per_row_prefill.co](hsa/gfx942/topk_per_row_prefill/) (31 KB) — multi-token blocks, cooperative LDS load
- 23 `topksoftmax/*.co` variants for `(num_experts ∈ {128,256,384}, topk ∈ {4,6,8}, dtype ∈ {fp32,bf16}, subm ∈ {4,12})`

Why hand-tuned ASM for a "simple" sort: each token sorts 128–384 floats independently in LDS; bandwidth-bound at ~200 GB/s; in decode steady state, every cycle counts.

---

## 5. How to write a high-performance kernel — playbook

Distilled from [docs/isa_kernel_optimization.md](docs/isa_kernel_optimization.md), the CK_tile sources, and the [.claude/skills/opus-kernel-best-practice/SKILL.md](.claude/skills/opus-kernel-best-practice/SKILL.md).

### 5.1 The ISA primer for gfx942/gfx950 (CDNA3/CDNA4)

**Register file** (per-CU): 256 VGPRs per thread max, 104 SGPRs per wave, plus **AGPRs** (accumulator GPRs) used by MFMA C-outputs. Wave size = **64** on both gfx942 and gfx950. Occupancy rule of thumb on MI300X with 16384 scalar slots per wave:

| VGPR/thread | Waves/CU |
|---|---|
| 256 | 1 |
| 128 | 2 |
| 64 | 4 |

Memory-bound kernels want ≥ 4 waves/CU to hide load latency; compute-bound kernels can sit at 1–2 if registers buy more ILP.

**MFMA instruction zoo** (the ones you'll actually see):

| Inst | A/B in | C out | M×N×K tile | Throughput per wave |
|---|---|---|---|---|
| `v_mfma_f32_16x16x16_f16` | f16 | f32 | 16×16×16 | gfx9 baseline |
| `v_mfma_f32_16x16x32_{f16,bf16,fp8,bf8}` | 16-bit / 8-bit | f32 | 16×16×32 | gfx942+, double K-rate |
| `v_mfma_f32_32x32x{8,16}_{f16,bf16,fp8}` | as named | f32 | 32×32 | bigger tile, fewer issues, more A/B per inst |
| `v_mfma_i32_{16x16x32, 32x32x16}_i8` | i8 | i32 | as named | integer GEMM |

Each MFMA has ~64–128 cycle latency. Lane→element mapping is fixed by the instruction (e.g. `16x16x16_f16` consumes 4 fp16/lane and produces 4 fp32/lane).

**Memory**: `buffer_load_dwordx{1,2,4}` with hw bounds-check on gfx9 (OOB → 0); `global_load_*` is flat-address (no bounds); `ds_read_b{32,64,128}` and `ds_write_b{32,64,128}` for LDS (32 banks, conflicts serialize); `s_load_dwordx*` for scalar broadcasts.

**Waitcnt encoding**: 6 bits `vmcnt` (≤63 in-flight vector mem ops), 4 bits `lgkmcnt` (≤15 in-flight LDS/sgpr ops), 3 bits `expcnt`. Pattern:

```asm
buffer_load_dwordx4 v[8:11], v4, s[0:3], 0     ; +1 vmcnt
buffer_load_dwordx4 v[12:15], v5, s[0:3], 0    ; +1 vmcnt
; ... independent work hides latency ...
s_waitcnt vmcnt(1)                             ; allow 1 still in flight
ds_write_b128 ..., v[8:11]
s_waitcnt vmcnt(0) lgkmcnt(0)
ds_read_b128 v[20:23], ...
```

### 5.2 The hot-loop pattern — software-pipelined GEMM

Every high-perf GEMM kernel in this repo, ASM or CK_tile, implements the same shape:

```
prolog:  load K=0, K=1 into LDS slots [0],[1]
hot loop for K=2..N-1:
   issue MFMA on data from K-2 (already in registers)
   issue buffer_load for K+1 (target alt LDS slot)
   issue ds_read for K-1 (filling registers for next MFMA)
   s_waitcnt to fence stages
   swap LDS slot pointer
epilog:  flush last two K iterations
```

CK_tile encodes this with `__builtin_amdgcn_sched_group_barrier(MFMA, 1, 0)` / `DS_READ, 1, 0)` / `VMEM_READ, 1, 0)` triplets ([3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp](3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp)) — interleaving 1 MFMA, 1 DS_READ, 1 MFMA, 1 VMEM_READ, then bulk MFMAs. Hand-tuned ASM does the same by hand and frees itself from the compiler's conservative `lgkmcnt(0)` barriers.

Tile decoding example for `flatmm_uk_gfx9_f16f8_128x256x128_1x4x1_16x16x32`:

- Block tile 128 (M) × 256 (N) × 128 (K)
- 1 × 4 × 1 wave grid (4 waves per block)
- MFMA `16x16x32` → per-wave tile = 128 / 1 = 128 M × 256 / 4 = 64 N; each wave issues `(128/16) × (64/16) × (128/32) = 8×4×4 = 128` MFMAs per K-loop pass

### 5.3 The workflow — *don't write from scratch*

1. **Find a close-shape `.co`** in [hsa/gfx*/](hsa/gfx942/). Names encode shape.
2. **Disassemble**: `llvm-objdump -d --mcpu=gfx942 kernel.co > kernel.isa`
3. **Look up the mangled symbol** in the family's CSV.
4. **Extract**: `python3 docs/examples/isa_optimization/extract_asm.py kernel.isa <symbol> -o kernel.s`
5. **Profile baseline**: `analyze_kernel.py isa kernel.co` for instruction mix; `rocprofv3 --kernel-trace ...` for timings + register/LDS metadata.
6. **Identify the bottleneck**: high `s_nop`/`s_waitcnt` % → ILP problem; high VGPR + low occupancy → register pressure; high LDS reads + slow → bank conflicts.
7. **Edit `kernel.s`**: reorder MFMA/buffer_load/ds_read; collapse redundant waitcnt; reuse dead registers.
8. **Recompile**: `clang++ -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx942 -o k_recompiled.co kernel.s`
9. **Verify round-trip**: extract `.text` from both, `md5sum` must match if you intended no change. For an actual edit, compare instruction counts via `analyze_kernel.py`.
10. **Inject `.text` into original**: `cp orig.co modified.co && llvm-objcopy --update-section .text=k_recompiled_text.bin modified.co`
11. **Profile modified** with `rocprofv3` and diff timings.

Or: `./docs/examples/isa_optimization/roundtrip.sh kernel.co --mcpu gfx942` does steps 2–10 in one shot.

### 5.4 Bug classes and how to find them

| Bug | Symptom | Diagnostic |
|---|---|---|
| Missing waitcnt before `ds_read` | Intermittent wrong outputs, shape-dependent | Compare `s_waitcnt lgkmcnt` count to original; every `ds_write` must be followed (within ~10 insts) by `vmcnt(0)` before its consumer reads |
| VGPR over-allocation | Modified kernel is slower despite cleaner scheduling | `llvm-objdump -s -j .note` on both — if `arch_vgpr_count` jumped from 80 → 120, you crossed the 2→1 wave threshold |
| LDS bank conflict | `ds_read` latency ~20 cycles instead of ~4 | rocprofv3 metric `LDS_Bank_Conflict`; fix by padding LDS row stride or transposing |
| Buffer-load OOB returning 0 | Tail rows of an output silently zeroed | gfx9 hw bounds-check on `buffer_load` returns 0 OOB; verify buffer descriptor `num_records` covers your tile |

### 5.5 When NOT to write ASM

Hand-tuned ASM is profitable only when **all** of:

- Kernel is ≥ 5 % of E2E runtime
- Shape is stable (fixed M/N/K or small set)
- ≥ 20 % expected speedup vs CK_tile/Triton baseline
- You can amortize 2–4 engineer-weeks of work over many deployments

Otherwise: CK_tile templates (`3rdparty/composable_kernel/include/ck_tile/`) for parameterized GEMM/FMHA/MoE that hit 85–95 % of peak, Triton for fast iteration on novel ops, or HIP source for prototyping.

The repo follows exactly this stratification:

| Layer | Used for |
|---|---|
| **ASM `.co`** | PA decode, MLA decode, FMOE, GEMM at fixed-shape hot points, AllReduce@N=8192 |
| **CK_tile templates** | Variable-shape GEMM, FMHA training fwd/bwd, fallback FMOE |
| **Triton** | All-gather / reduce-scatter (Iris), experimental quant ops |
| **HIP source** | Normalization, activations, embedding, transpose, gather |

---

## 6. CK_tile as the readable specification

`3rdparty/composable_kernel/include/ck_tile/` is the readable C++ source-form of every pattern the `.co` kernels encode. Four primitives:

- **`tile_window`** ([3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_window.hpp](3rdparty/composable_kernel/include/ck_tile/core/tensor/tile_window.hpp)) — a rectangular sub-tensor view with thread-distribution baked in. `load_tile(win)` and `store_tile(win, data)` generate coalesced scatter/gather automatically.
- **`static_distributed_tensor`** ([3rdparty/composable_kernel/include/ck_tile/core/tensor/static_distributed_tensor.hpp](3rdparty/composable_kernel/include/ck_tile/core/tensor/static_distributed_tensor.hpp)) — per-thread register buffer + distribution descriptor.
- **MFMA dispatcher** ([3rdparty/composable_kernel/include/ck_tile/core/arch/mma/mfma/mfma_gfx9.hpp](3rdparty/composable_kernel/include/ck_tile/core/arch/mma/mfma/mfma_gfx9.hpp)) — wraps `__builtin_amdgcn_mfma_*` with the lane→element mapping for each tile shape.
- **`block_sync_lds<lgkmcnt>()`** — expands to `s_waitcnt + s_barrier` with a controllable wait count.

Pipelines to read:

- GEMM hot loop: [3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp](3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp) — async copy global→LDS, double-buffered, prefetch=2 stages.
- FMOE: [3rdparty/composable_kernel/include/ck_tile/ops/fused_moe/pipeline/fused_moegemm_pipeline_flatmm_ex.hpp](3rdparty/composable_kernel/include/ck_tile/ops/fused_moe/pipeline/fused_moegemm_pipeline_flatmm_ex.hpp) — two LDS buffers `smem_0`/`smem_1` for gate/up, `IsGateOnly` flag, smooth-quant support.
- FMHA online softmax: [3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp](3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp) — `kQLoadOnce` keeps Q resident, streams K and V tiles, online softmax with `lse` carry.
- Copy-kernel tutorial: [3rdparty/composable_kernel/tutorial/ck_tile/00_copy_kernel/copy_basic.hpp](3rdparty/composable_kernel/tutorial/ck_tile/00_copy_kernel/copy_basic.hpp) — minimal end-to-end demo of `tile_window` + `load_tile` + `store_tile` + `move_tile_window`.

The CK_tile source contains explicit `FIXME` and `TODO` comments where the authors acknowledge perf gaps (`FIXME: using array will cause register spill`, `TODO: better LDS descriptor for performance`, `TODO: scheduler will likely need to be redesigned`) — these are exactly the gaps the hand-tuned ASM kernels close.

---

## 7. Institutional context

- **OPUS kernel best-practices skill** ([.claude/skills/opus-kernel-best-practice/SKILL.md](.claude/skills/opus-kernel-best-practice/SKILL.md)) — compile-time optimization (GQA flash-attn: 4.8s → 1.5s). Rule #0 is `#ifdef __HIP_DEVICE_COMPILE__` to skip `opus.hpp` on the host pass (~50 % compile saving). Other rules: replace `static_for<N>` with plain `for` when no compile-time index is needed (30–60 % frontend), `__builtin_convertvector`/`__builtin_shufflevector` (5–10 %), `-ftime-trace` to find slow instantiations.
- **CONTRIBUTE.md** ([CONTRIBUTE.md](CONTRIBUTE.md)) — PR conventions (`[Kernel]`, `[Perf]`, `[HIP]`, `[CK]`, `[Triton]`, `[JIT]` prefixes), perf PRs must include hardware, baseline/optimized timing, % gain, % of peak bandwidth, roofline analysis. Pre-commit hooks: `black`, `ruff`, `clang-format-18`.
- **Kernel-agents framework** ([.kernel-agents/](.kernel-agents/), [agent_logs/](agent_logs/)) — orchestrator + specialist agents (aiter-fellow, ck-fellow, triton-fellow, flydsl-fellow, hip-fellow, hipblaslt-fellow) for autonomous multi-kernel tuning campaigns. ENV_HINTS documents that GPU benches may run on a different host (inside `flydsl_gemm_bench` docker) than the dev box.
- **REPO_CLEANUP_PLAN.md** — Phase-2 `git-filter-repo` history rewrite pending (420 MB → 105 MB).
- **`4f1679f7dae9/4135_results.db`** — cached rocprofv3 autotuner results DB. Hash-named dir, owned by root. Safe to delete if space is needed.

---

## 8. Quick-reference command sheet

```bash
# Disassemble + analyze a kernel
KCO=hsa/gfx942/pa_a16w8_bf16_2tg_g8_f8_gemm1_bf16.co
/opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx942 $KCO > kernel.isa
python3 docs/examples/isa_optimization/analyze_kernel.py isa $KCO --mcpu gfx942

# Look up the mangled symbol
grep -oE '<[^>]+>:' kernel.isa | head -1

# Extract reassemblable .s
python3 docs/examples/isa_optimization/extract_asm.py kernel.isa <SYMBOL> \
    --target amdgcn-amd-amdhsa--gfx942 -o kernel.s

# Round-trip (verify or after edit)
./docs/examples/isa_optimization/roundtrip.sh $KCO --mcpu gfx942

# Inject modified .text into original
cp $KCO modified.co
/opt/rocm/llvm/bin/llvm-objcopy -O binary -j .text k_recompiled.co recompiled_text.bin
/opt/rocm/llvm/bin/llvm-objcopy --update-section .text=recompiled_text.bin modified.co

# Read register / LDS metadata
/opt/rocm/llvm/bin/llvm-objdump --mcpu=gfx942 -s -j .note $KCO

# Profile
rocprofv3 --kernel-trace -d ./profile_out -- python benchmark.py
python3 docs/examples/isa_optimization/analyze_kernel.py profile ./profile_out --filter pa_

# Compile a new .co from .s
/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn-amd-amdhsa \
    -mcpu=gfx942 -o new_kernel.co new_kernel.s
```

---

## 9. Reading order for a new engineer

1. [docs/isa_kernel_optimization.md](docs/isa_kernel_optimization.md) — the master guide (1 hour).
2. [docs/examples/isa_optimization/README.md](docs/examples/isa_optimization/README.md) + [roundtrip.sh](docs/examples/isa_optimization/roundtrip.sh) — try the round-trip on any PA kernel.
3. [.claude/skills/opus-kernel-best-practice/SKILL.md](.claude/skills/opus-kernel-best-practice/SKILL.md) — compile-time hygiene.
4. [3rdparty/composable_kernel/tutorial/ck_tile/00_copy_kernel/copy_basic.hpp](3rdparty/composable_kernel/tutorial/ck_tile/00_copy_kernel/copy_basic.hpp) — minimal CK_tile.
5. [3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp](3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp) — the canonical SW-pipelined GEMM loop.
6. [hsa/codegen.py](hsa/codegen.py) + one generated header (e.g. [aiter/jit/build/module_gemm_a8w8_asm/blob/asm_i8gemm_configs.hpp](aiter/jit/build/module_gemm_a8w8_asm/blob/asm_i8gemm_configs.hpp)) — the dispatch story.
7. [csrc/include/aiter_hip_common.h](csrc/include/aiter_hip_common.h) — `AiterAsmKernel`, the runtime loader.
8. [csrc/py_itfs_cu/asm_fmoe.cu](csrc/py_itfs_cu/asm_fmoe.cu) or [asm_mla.cu](csrc/py_itfs_cu/asm_mla.cu) — one full dispatch example end-to-end.

Then disassemble a PA decode kernel, run `analyze_kernel.py` on it, and try to identify three optimization opportunities by reading the ISA.
