# AMD CDNA3/CDNA4 (gfx942/gfx950) Kernel Performance Reference — v2

Long-form synthesis of 16 agent reports + the AMD CDNA4 ISA spec (608 pages — at `../amd-instinct-cdna4-instruction-set-architecture.pdf`, text extracts in [`kernel-analysis/cdna4-isa/`](kernel-analysis/cdna4-isa/)). Every entry cites either `[pdf:pN]` for ISA spec pages, `[file:LN]` for source, or `[kernel-analysis/disassembly/{kernel}/kernel.isa:LN]` for actual disassembled `.co` files.

> **Path conventions** — this doc lives in the `aiter-kernel-analysis` wrapper repo with [`aiter-amd/`](aiter-amd/) as a submodule:
> - `kernel-analysis/...` — paths in **this wrapper repo** (disassembly archive, ISA extracts)
> - `hsa/...`, `csrc/...`, `docs/...`, `3rdparty/composable_kernel/...`, `aiter/ops/...`, `aiter/jit/...`, `.claude/...`, `setup.py` — paths in the **aiter-amd submodule**; prefix with `aiter-amd/` to resolve from this repo's root
> - `[pdf:pN]` — page in the AMD CDNA4 ISA spec PDF (at `../` from this repo)

This replaces the high-level [ASM_PERF_PLAYBOOK.md](ASM_PERF_PLAYBOOK.md) with bit-level, instruction-exact detail.

> ### Errata (verified against spec + actual disassembly during a review pass)
>
> 1. **§9.1 DPP encoding bit positions corrected** — previous draft had positions copied from the wrong table.
> 2. **§16.2 flatmm scale-application path corrected** — `v_mul_f32_e32` / `v_fmac_f32_e32` interleaved with MFMA, NOT `v_pk_mul_f32` (that instruction does not appear in the `flatmm_uk_...` binary at all).
> 3. **§15.1 PA fp8 dequant line refs corrected** — `s[44:45] = 0xF0F0F0F0` is loaded at [kernel-analysis/disassembly/pa/kernel.isa:148-149] and consumed by `v_cndmask_b32_e64` at L383-386 (not L365-378 which is the surrounding MFMA chain).
> 4. **§6.8 MFMA dep-table SGEMM SrcC value ordering** — the four-pass values are `2/0/0/0`, not `0/0/0/2` (the `2` is the first opcode pass-count, not the last).
> 5. **§7 V# table citation** — the layout table is on `[pdf:p91]`, not `[pdf:p90]` (which is the section heading).
>
> Other cycle counts, opcode numbers, DPP_CTRL enum values, DS encoding bit positions, V# bit layout, manual-NOP table, and MFMA opcode table all verified accurate.

---

# Part I — Foundation

## 1. Wave / register architecture (CDNA4)

| Property | Value | Source |
|---|---|---|
| Wave size | 64 work-items (fixed, no wave32) | [pdf:p11-12] |
| Max workgroup | 16 waves = 1024 work-items | [pdf:p27] |
| VGPR + AGPR | Unified physical register file; MFMA's `ACC` and `ACC_CD` bits select VGPR (0) vs AGPR (1) per matrix operand | [pdf:p20, p49] |
| SGPR file | 104 SGPRs per wave; VCC aliases SGPR 106–107 | [pdf:p18-20] |
| EXEC | 64-bit mask, 1 bit per lane; EXECZ flag = (EXEC==0) | [pdf:p17] |
| SCC | 1-bit scalar condition code, workgroup-uniform | [pdf:p21] |
| M0 | 32-bit; LDS offset/size or GPR-indexing base | [pdf:p21] |
| HW_ID | Read-only via `s_getreg_b32 hwreg(HW_REG_HW_ID, ...)`: WAVE_ID[3:0], SIMD_ID[5:4], PIPE_ID[7:6], CU_ID[11:8], SH_ID[12], SE_ID[15:13], TG_ID[19:16], VM_ID[23:20], QUEUE_ID[26:24], ME_ID[31:30] | [pdf:p24] |
| XCC_ID | XCC[3:0] (eXtreme Cache Cluster on gfx942+) | [pdf:p24] |
| LDS bytes/CU | gfx942: standard; **gfx950: 160 KB total** | per CK_tile arch.hpp + [pdf:p105] |
| LDS banks | gfx942: 32 banks × 4 B; **gfx950: 64 banks** × 4 B | CK_tile arch.hpp:1179-1197 |

**Disassembled `.note` examples**:

| Kernel | SGPR | VGPR | AGPR | LDS | ISA lines |
|---|---|---|---|---|---|
| `pa_a16w8_bf16_2tg_g8_f8_gemm1_bf16.co` (per-token fp8) | ~70 | 256 | impl. | 5.4 KB | 2946 |
| `pa_bf16_perblockFp8_blk256_1tg_4w_qlen16_msk1_ps.co` | 94 | 132+ | impl. | 5.6 KB | 2898 |
| `pa_a16w16_b16.co` (no-quant bf16) | 73 | 412 | impl. | 256 B | 2307 |
| `flatmm_uk_gfx9_f16f8_128x256x128_1x4x1_16x16x32.co` | 93 | 717 | — | 64 KB | (large) |
| `fmoe_fp8_blockscale_g1u1_novs_subGU_256.co` | 95 | 255 | 127 | 53.4 KB | 4423 |
| `fmoe_fp8_g1u1_smf_subGU_320.co` | 95 | 231 | 159 | 17.3 KB | 5626 |
| `fmoe_fp8_g1u1_multix_subGU_256.co` | 95 | 223 | 127 | 42.1 KB | 4037 |
| `gemm_a8w8_m128_splitK.co` | 96 | 512 | 0 | 8 KB | 1449 |
| `gemv_router.co` (gfx950) | 23 | 25 | 0 | 16 KB | 531 |
| `allreduce_rmsnorm_N8192.co` | 96 | 512 | 0 | 4 KB | 1529 |
| `mla_dec_stage1_bf16_a16w16_subQ128_mqa128.co` | 96 | 512 | impl. | 256 B | 6867 |
| `topksoftmax_4x256x8_bf16.co` | 112 | 24 | 0 | 0 B | 1030 |

Three observations across this table:

1. **VGPR scales with unrolling depth, not just dtype.** fp8 PA uses 256 VGPR; bf16 PA uses 412 VGPR (loading 2× the data → more reg-resident tiles). MLA uses 512 VGPR (compressed KV + LoRA basis + RoPE rotation state).
2. **AGPR usage signals MFMA design.** Hand-tuned `flatmm` allocates 0 AGPR (uses VGPR-128-wide accumulator); SMF FMOE pushes AGPR to 159 to hold *two experts* worth of accumulator state per workgroup. Setting `ACC_CD=1` in the VOP3P-MAI encoding is what routes MFMA's C output to AGPR [pdf:p49].
3. **Workgroup-zero-LDS kernels exist.** TopK-softmax uses 0 B LDS and reduces purely via DPP + GPR-indexed scatter ([§18](#18-topk-softmax--dpp-only-reduction-with-gpr-indexed-scatter)).

## 2. The three completion counters

Software-visible memory completion is tracked by three counters, all readable in `IB_STS` and all reset to zero on kernel entry [pdf:p27-28]:

| Counter | Bit width | Inc-by | Dec-by | Order | s_waitcnt encoding |
|---|---|---|---|---|---|
| **vmcnt** | 6 (max 63) | +1 per MUBUF/MTBUF/FLAT op | -1 per VGPR write-back (read) or L2 commit (write) | Same-type in-order; mixed VMEM in-order | SIMM16[15:14] : SIMM16[3:0] |
| **lgkmcnt** | 4 (max 15) | +1 per LDS op; +dword-count per scalar load (`s_load_dwordx2` → +2); +1 per FLAT (also touches vmcnt); +1 per `s_sendmsg` | -1 per LDS read return, LDS write commit, scalar dword return, sendmsg complete | Different-type out-of-order; same-type in-order **except** scalar-mem-reads (which can return OoO — only `s_waitcnt 0` is legitimate) | SIMM16[11:8] |
| **expcnt** | 3 | UNUSED on CDNA4 | — | — | SIMM16[6:4] |

> **No vscnt on CDNA4.** Search of the spec turns up no separate VMEM-store counter; the single 6-bit vmcnt covers both loads and stores. RDNA-style `s_waitcnt_vscnt` does not apply here.

`s_waitcnt SIMM16` encoding [pdf:p153]:
```
SIMM16[3:0]   = vmcnt[3:0]            (low 4 bits)
SIMM16[6:4]   = expcnt[2:0]           (unused, normally 0)
SIMM16[11:8]  = lgkmcnt[3:0]
SIMM16[15:14] = vmcnt[5:4]            (high 2 bits)
```

So `s_waitcnt vmcnt(8) lgkmcnt(0)` packs as: vmcnt[5:4]=00, lgkmcnt=0, expcnt=0, vmcnt[3:0]=1000 → `0x0008`. And `s_waitcnt 0` (drain all) packs as `0x0000`.

**CK_tile's three arch tiers** ([3rdparty/composable_kernel/include/ck_tile/core/arch/arch.hpp:913-959](3rdparty/composable_kernel/include/ck_tile/core/arch/arch.hpp)):

```cpp
struct WaitcntLayoutGfx12  { /* dscnt[5:0], mem[13:8] — no expcnt   */ };
struct WaitcntLayoutGfx11  { /* lgkm[9:4], exp[2:0], vm[15:10]      */ };
struct WaitcntLayoutLegacy { /* CDNA: vm low4+hi2, lgkm[11:8], exp[6:4] */ };
```

CDNA4 uses the *Legacy* layout — gfx12 has restructured the encoding, but gfx942/gfx950 still use the AMD-traditional bit layout above.

## 3. Manual NOPs (HW does NOT check) — full table

Per [pdf:p28-29, Table 11], the following producer→consumer pairs **must** have NOPs (or independent instructions) inserted by software:

| First | Second | NOPs | Notes |
|---|---|---|---|
| `S_SETREG <r>` | `S_GETREG <same r>` | 2 | |
| `S_SETREG <r>` | `S_SETREG <same r>` | 2 | |
| `SET_VSKIP` | `S_GETREG MODE` | 2 | Reads VSKIP from MODE |
| `S_SETREG MODE.vskip` | any vector op | 2 | |
| VALU sets VCC or EXEC | VALU using EXECZ/VCCZ as **data** | **5** | |
| VALU writes SGPR/VCC | `V_{READ,WRITE}LANE` using that as lane-select | **4** | |
| VALU writes VCC | `V_DIV_FMAS` | 4 | VCC carry-in |
| `FLAT_STORE_X3/X4` / `FLAT_ATOMIC_CMPSWAP_X2` / `BUFFER_STORE_DWORD_X3/X4` / `BUFFER_STORE_FORMAT_XYZ/XYZW` / `BUFFER_ATOMIC_CMPSWAP_X2` | Write writedata VGPRs from those instructions | 1 | If SOFFSET=SGPR, no wait needed |
| Same as above | VALU writes writedata VGPRs | 2 | |
| **VALU writes SGPR** | **VMEM reads that SGPR** | **5** | HW does **not** check; user MUST add 5 nops |
| `SALU writes M0` | `S_SENDMSG` | 1 | |
| **VALU writes VGPR** | **VALU DPP reads that VGPR** | **2** | |
| **VALU writes EXEC** | **VALU DPP op** | **5** | ALU does NOT forward EXEC to DPP |
| VCC alias use vs name use | VALU reading VCC as constant (not carry-in) | 1 | The dependency-check hardware does not understand VCC ↔ SGPR# aliasing |
| `S_SETREG TRAPSTS` | RFE/RFE_restore | 1 | |
| SALU writes M0 | LDS "add-TID" instruction, `buffer_store_LDS_dword`, scratch/global with LDS=1 | 1 | |
| SALU writes M0 | `S_MOVEREL` | 1 | |
| VALU writes SGPR/VCC | VALU reads SGPR as constant | 2 | |
| (same producer) | VALU reads SGPR as carry-in | 0 | |
| `v_readlane` / `v_writelane` reads SGPR as lane-select | (after VALU SGPR write) | 4 | |
| `v_cmpx` | VALU reads EXEC as constant | 2 | |
| `v_readlane/v_readfirstlane/v_writelane` | Other VALU | 0 | |
| VALU writes VGPRn | `v_readlane vsrc0` reads VGPRn | 1 | |
| VALU op using OPSEL or SDWA (changes bit-position) | VALU op consumes its result | 1 | |
| **Trans op (`v_exp_f32`, `v_rcp_f32`, `v_rsq_f32`, …)** | Non-trans VALU consumes result | **1** | Trans pipeline is shared, single-issue per CU |
| `V_CMPX` (writes EXEC) | `V_PERMLANE*` | 4 | |
| VALU* writes vdst | `V_PERMLANE*` reads vdst | 2 | |

**Trans Ops list** [pdf:p29]: `V_EXP_F32`, `V_LOG_F32`, `V_RCP_F32`, `V_RCP_IFLAG_F32`, `V_RSQ_F32`, `V_RCP_F64`, `V_RSQ_F64`, `V_SQRT_F32`, `V_SQRT_F64`, `V_SIN_F32`, `V_COS_F32`, plus F16/legacy variants. All share one pipeline.

**Inline-asm pattern used by CK_tile for MFMA→MFMA spacing** ([warp_gemm_attribute_mfma_impl.hpp](3rdparty/composable_kernel/include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp)):

```cpp
asm volatile(mfma_ " %0, %1, %2, %3\n"
                   "s_nop 3"
             : "+a"(c_vec)      // AGPR read-modify-write
             : "v"(a_vec), "v"(b_vec), "a"(c_vec)
             :);
```

The `s_nop 3` satisfies the MFMA dependency rule (see §6). Compiler-emitted LLVM is often `s_nop 7` to be safe; hand-tuned ASM trims this down by scheduling independent ops in the gap.

## 4. SOPP (control) instruction reference

[pdf:p150-158]. Complete list:

| Opc | Name | Semantics |
|---|---|---|
| 0 | `S_NOP <SIMM16>` | Insert `SIMM16[3:0]` NOPs (0..15 cycles) |
| 1 | `S_ENDPGM` | Terminate wave; **implicit `s_waitcnt 0`** before exit |
| 2 | `S_BRANCH <off>` | PC += signext(SIMM16 * 4) + 4 |
| 3 | `S_WAKEUP` | Wake all sleeping waves in TG (NOP if not in TG) |
| 4-9 | `S_CBRANCH_{SCC0, SCC1, VCCZ, VCCNZ, EXECZ, EXECNZ}` | Conditional jump on flag |
| 10 | `S_BARRIER` | Synchronize all waves in TG. **Does NOT wait for memory counters** — explicit `s_waitcnt` first if needed |
| 11 | `S_SETKILL` | Kill wave if `SIMM16[0]==1` |
| 12 | `S_WAITCNT <SIMM16>` | See §2 for encoding |
| 13 | `S_SETHALT` | Set HALT bit |
| 14 | `S_SLEEP <SIMM16>` | Sleep ≈ 64·(SIMM16[6:0]−1) … 64·SIMM16[6:0] cycles (max ~8000) |
| 15 | `S_SETPRIO <SIMM16>` | User priority = SIMM16[1:0]; 0=lowest, 3=highest. Overall priority = {SPIPrio, UserPrio, WaveAge} |
| 16 | `S_SENDMSG` | M0[23:0] payload; SIMM16[9:0] msg type (interrupt, save_wave, stall_wave_gen, halt_waves, get_doorbell_id) |
| 17 | `S_SENDMSGHALT` | Send and halt |
| 18 | `S_TRAP` | Enter trap; PC saved in TTMP1:TTMP0; implicit wait-for-all |
| 19 | `S_ICACHE_INV` | Invalidate L1 I-cache. **MUST be followed by 16× `S_NOP` or jump/branch** |
| 20/21 | `S_INCPERFLEVEL / S_DECPERFLEVEL` | Perf counter |
| 22 | `S_TTRACEDATA` | Send M0 to thread-trace stream |
| 23-26 | `S_CBRANCH_CDBG{SYS,USER,SYS_OR_USER,SYS_AND_USER}` | Debug-flag branches |
| 27 | `S_ENDPGM_SAVED` | Context-save terminate |
| 28/29 | `S_SET_GPR_IDX_OFF/MODE` | Toggle/configure GPR indexing |

**No `s_clause` on CDNA4.** Search of the spec confirms — RDNA's clause-grouping doesn't apply; memory-op ordering is automatic within a counter class.

## 5. Scheduler hints (LLVM intrinsics — *not* ISA)

These are LLVM compiler directives, not real instructions, but they drive code generation that ends up in `.co`:

| Intrinsic | Effect |
|---|---|
| `__builtin_amdgcn_sched_group_barrier(mask, count, sync_id)` | Reserve next `count` instructions of class `mask` in scheduler |
| `__builtin_amdgcn_sched_barrier(0)` | Global fence — no reorder across this point |
| `__builtin_amdgcn_iglp_opt(N)` | Intra-loop-group pipeline optimization strategy id (0=default) |
| `__builtin_amdgcn_s_setprio(prio)` | Direct `s_setprio` emit |

Mask values:

| Hex | Class |
|---|---|
| 0x002 | VALU / ALU |
| 0x004 | SALU |
| 0x008 | **MFMA** |
| 0x040 | **VMEM_READ** |
| 0x080 | VMEM_WRITE |
| 0x100 | **DS_READ** |
| 0x200 | DS_WRITE |

Real use in CK_tile GEMM hot loop [[gemm_pipeline_ag_bg_cr_comp_async.hpp:205-238](3rdparty/composable_kernel/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp)]:

```cpp
static_for<0, num_buffer_load_inst, 1>{}([&](auto i) {
    __builtin_amdgcn_sched_group_barrier(MFMA, 1, 0);          // 1 MFMA
    __builtin_amdgcn_sched_group_barrier(DS_READ, 1, 0);       // 1 LDS read
    __builtin_amdgcn_sched_group_barrier(MFMA, 1, 0);          // 1 MFMA
    __builtin_amdgcn_sched_group_barrier(VMEM_READ, 1, 0);     // 1 global load
    __builtin_amdgcn_sched_group_barrier(MFMA,
                                         C_MFMA_Inst_Num / num_issue - 2, 0);
});
__builtin_amdgcn_sched_barrier(0);
```

Pattern: **MFMA → DS_READ → MFMA → VMEM_READ → MFMA-N** repeated per prefetch unit, then a global fence. This is exactly the interleave a hand-tuned ASM kernel would produce.

FMHA uses denser packs for head_dim=256 [[block_fmha_pipeline_qr_ks_vs.hpp:434-456](3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp)]: `DS_READ(2) → MFMA(2) → DS_READ(1) → MFMA(2) → DS_READ(1) → MFMA(4)`.

---

# Part II — Instruction-level reference

## 6. MFMA — the master compute instruction

### 6.1 Opcode table (CDNA4) [pdf:p51, Table 28]

| Instruction | M×N×K | Blocks | Cycles | A in / lane | B in / lane | C out / lane |
|---|---|---|---|---|---|---|
| `v_mfma_f32_*_f32` | 32×32×2 / 16×16×4 | 1 | 64 / 32 | 4 f32 | 4 f32 | 16 f32 |
| `v_mfma_f32_*_f32` | 32×32×1_2B / 16×16×1_4B / 4×4×1_16B | 2 / 4 / 16 | 64 / 32 / 8 | batched | batched | 16 f32 |
| `v_mfma_f32_*_f16` | 32×32×8 / 16×16×16 | 1 | 32 / 16 | 4 f16 | 4 f16 | 16 f32 |
| `v_mfma_f32_*_f16` (CDNA4-new) | 32×32×16 / 16×16×32 | 1 | 32 / 16 | 8 f16 | 8 f16 | 16 f32 |
| `v_mfma_f32_*_bf16` | 32×32×8 / 16×16×16 | 1 | 32 / 16 | 4 bf16 | 4 bf16 | 16 f32 |
| `v_mfma_f32_*_bf16` (CDNA4-new) | 32×32×16 / 16×16×32 | 1 | 32 / 16 | 8 bf16 | 8 bf16 | 16 f32 |
| `v_mfma_i32_*_i8` | 32×32×4_2B / 16×16×4_4B / 4×4×4_16B | 2 / 4 / 16 | 64 / 32 / 8 | batched | batched | 16 i32 |
| `v_mfma_i32_*_i8` | 32×32×16 / 16×16×32 | 1 | 32 / 16 | 2 i8 | 2 i8 | 16 i32 |
| **`v_mfma_i32_16x16x64_i8`** | 16×16×64 | 1 | 16 | 16 i8 | 16 i8 | 16 i32 |
| **`v_mfma_i32_32x32x32_i8`** | 32×32×32 | 1 | 32 | 8 i8 | 8 i8 | 16 i32 |
| `v_mfma_f32_*_{bf8,fp8}_{bf8,fp8}` | 16×16×32 / 32×32×16 | 1 | 16 / 32 | 4 fp8 | 4 fp8 | 16 f32 |
| `v_mfma_f64_*_f64` | 16×16×4 / 4×4×4_4B | 1 / 4 | 64 / 32 | f64 | f64 | 4 f64 |
| **`v_mfma_f32_16x16x128_f8f6f4`** | 16×16×128 | 1 | 16 (F4/F6) or 32 (F8) | mixed fp4/fp6/fp8 | mixed | 16 f32 |
| **`v_mfma_f32_32x32x64_f8f6f4`** | 32×32×64 | 1 | 32 / 64 | mixed | mixed | 16 f32 |
| **`v_mfma_scale_f32_16x16x128_f8f6f4`** | 16×16×128 | 1 | 16 / 32 | mixed + E8M0 scale | mixed + scale | 16 f32 |
| **`v_mfma_scale_f32_32x32x64_f8f6f4`** | 32×32×64 | 1 | 32 / 64 | mixed + scale | mixed + scale | 16 f32 |

CDNA4-new entries are **bold**. The big new one is the block-scaled mixed-format family.

### 6.2 Lane-to-element mapping [pdf:p55-62, §7.1.4]

The fundamental formula:

```
K_L = K / (64 / (M * B))    where K = K-dim, M = M-dim, B = blocks
```

For A: element A[b, i, k] is placed at **item** `k % K_L` of **lane** `i + M·(b + B·(k / K_L))`.
For B: element B[b, k, j] is placed at **item** `k % K_L` of **lane** `j + N·(b + B·(k / K_L))`.

Worked example: `v_mfma_f32_16x16x32_fp8_fp8` (CDNA3+, the workhorse of `flatmm` and `pa`).
- M=16, N=16, K=32, B=1
- K_L = 32 / (64 / 16) = 8
- Each lane holds 8 K-elements of A and 8 of B = 8 fp8 bytes = 1 register pair (`a[0:1]` is 2 dwords = 8 fp8 values)

So when you see [`kernel-analysis/disassembly/flatmm/kernel.isa:393`](_):
```asm
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], a[0:1], a[64:65], 0
```
- `v[56:59]` = 4 fp32 outputs/lane (the C-accumulator tile, 64 lanes × 4 = 256 fp32 = 1 16×16 output)
- `a[0:1]` = 2 dwords = 8 fp8 K-values per lane = A's first K=8 of the M=16 row this lane owns
- `a[64:65]` = 2 dwords = 8 fp8 K-values per lane = B's first K=8 of the N=16 col this lane owns
- The final `0` is the *initial* C accumulator (start fresh, no add)

Subsequent issues with `v[56:59]` as the third operand:
```asm
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], a[2:3], a[66:67], v[56:59]
```
mean "accumulate K=8..15 onto the existing C".

### 6.3 Output layout [pdf:p57-58, §7.1.4.2]

```
H = group height (=4 for everything except F64 where H=1)
B_I = ceil(64 / (N · M / H))
M_I = (64 / B_I) / N
G = M / (H · M_I)
```

D[b, i, j] is at **item** `(i % H) + H·(i/(H·M_I) + G·(b/B_I))` of **lane** `j + N·((i/H) % M_I + M_I·(b % B_I))`.

For `v_mfma_f32_16x16x16_f16`: H=4, B_I=1, M_I=4, G=1. **One register holds one row group** (4 rows of f32 outputs).

For `v_mfma_f32_32x32x1_2B_f32`: H=4, B_I=1, M_I=2, G=4. Four registers per group × 4 groups = 16 registers per block, two blocks → 32 registers total holding 32×32 = 1024 outputs over 64 lanes.

### 6.4 CBSZ / ABID / BLGP modifiers [pdf:p61-62, §7.1.6]

For *dense* MFMA:

- **CBSZ[2:0]** = broadcast block size. S = `64 / (1<<CBSZ)`. Lane transformation `l_a' = (l_a % S) + S·ABID`.
  - CBSZ=0: no broadcast
  - CBSZ=1: S=32 (first/second 32-lane block broadcasts)
  - CBSZ=2: S=16
  - CBSZ=3: S=8
  - CBSZ=4: S=4

- **ABID[3:0]** = which block of A broadcasts. Range depends on CBSZ.

- **BLGP[2:0]** = B-matrix lane-group permute [pdf:p62, Table 29]:
  - 0: no permute
  - 1: broadcast lanes 0–31
  - 2: broadcast lanes 32–63
  - 3: rotate left 16 lanes ((l+16) % 64)
  - 4–7: broadcast 16-lane block 0/1/2/3

For *F64* MFMA, BLGP repurposed: BLGP[0]=negate A, BLGP[1]=negate B, BLGP[2]=negate C [pdf:p62-63 §7.1.6.4].

### 6.5 F8F6F4 dtype encoding via CBSZ/BLGP [pdf:p59-60, §7.1.5]

For `v_mfma_f32_*_f8f6f4` and the scaled variants, CBSZ selects **A's format**, BLGP selects **B's format** [pdf:p59]:

| Code | Format | Bits | Sign-Exp-Mant | Bias | Max | Min(norm) |
|---|---|---|---|---|---|---|
| 000 | FP8 (E4M3) | 8 | 1-4-3 | 7 | 448 | 2⁻⁶ |
| 001 | BF8 (E5M2) | 8 | 1-5-2 | 15 | 57344 | 2⁻¹⁴ |
| 010 | FP6 (E2M3) | 6 | 1-2-3 | 1 | 7.5 | 1.0 |
| 011 | BF6 (E3M2) | 6 | 1-3-2 | 3 | 28.0 | 0.25 |
| 100 | FP4 (E2M1) | 4 | 1-2-1 | 1 | 6.0 | 1.0 |

Behavior: ignores MODE denorm/round, forced RNE, supports CLAMP via FP16_OVFL bit. SRC0/1/VDST must be **even-aligned VGPR/AGPR**; SRC2 can be inline constant [pdf:p59-60].

### 6.6 V_MFMA_SCALE_F32_*_F8F6F4 — block-exponent scaling (CDNA4-new) [pdf:p63-65, §7.2.1]

This is a **4-DWORD instruction** (double-VOP3P). The first two dwords are an implicit "Load-Scale" command (encoded 0xCC35 in bits [31:16]); the second two are the MFMA proper. From the dispatch perspective:

```cpp
__builtin_amdgcn_mfma_scale_f32_16x16x128f8f6f4(
    a, b, c,                  // matrices
    scale_a, scale_b,          // 8-bit E8M0 scales
    cbsz, blgp, abid           // A fmt, B fmt, scale-select
);
```

Scale fields:
- E8M0: 8-bit exponent only, bias 127, valid range −127..127 (0xFF is NaN). One scale per 32-K block.
- For `v_mfma_scale_f32_16x16x128_f8f6f4`: K=128, scales per row = 128/32 = 4; M=16 rows; total = 64 8-bit scales = ¼ of one VGPR per lane.
- `ABID[0] = 1`: scales active. `ABID[0] = 0`: scales forced to 1.0 (exponent = 0x7F bias).
- `{OP_SEL_HI[i], OP_SEL[i]}` selects which 8-bit byte of the source register holds the scale (00=Src[7:0], 01=Src[15:8], 10=Src[23:16], 11=Src[31:24]).

Hardware accumulator formula [pdf:p64]:
```
d_exp = Σ(a_i_exp + b_i_exp) + c_exp + scale_a + scale_b
```

**This is the obvious target for FMOE blockscale upgrades.** The current FMOE blockscale `.co` is gfx942-compiled and applies scales as a separate post-MFMA `v_mul_f32` chain ([kernel-analysis/disassembly/fmoe/kernel.isa:302-318](_) loads scales into v36..v43; lines 870..889 multiply). On gfx950 the `v_mfma_scale_f32_*` family would fold the dequant entirely into the MFMA unit.

### 6.7 V_SMFMAC — 4:2 structural sparse MFMA [pdf:p68-77, §7.5]

- A is sparse (2 of every 4 K-elements are zero); B/C/D dense.
- SRC2 holds the sparsity indices (compressed 2-bit-per-quad). Must be even-aligned for ≤8-bit data, no alignment for 16-bit.
- Same opcodes available as dense, with K dimensions effectively doubled (since A is compressed 2:1):

| Sparse instruction | K | Cycles |
|---|---|---|
| `v_smfmac_f32_16x16x32_{f16,bf16}` | 32 | 16 |
| `v_smfmac_f32_32x32x16_{f16,bf16}` | 16 | 32 |
| `v_smfmac_i32_16x16x64_i8` | 64 | 16 |
| `v_smfmac_i32_32x32x32_i8` | 32 | 32 |
| `v_smfmac_f32_16x16x64_{bf8,fp8}_{bf8,fp8}` | 64 | 16 |
| `v_smfmac_f32_32x32x32_{bf8,fp8}_{bf8,fp8}` | 32 | 32 |
| `v_smfmac_f32_16x16x64_{bf16,f16}` (CDNA4) | 64 | 16 |
| `v_smfmac_i32_16x16x128_i8` (CDNA4) | 128 | 16 |
| `v_smfmac_f32_16x16x128_{bf8,fp8}_{bf8,fp8}` (CDNA4) | 128 | 16 |
| `v_smfmac_f32_32x32x32_{bf16,f16}` | 32 | 32 |
| `v_smfmac_i32_32x32x64_i8` (CDNA4) | 64 | 32 |
| `v_smfmac_f32_32x32x64_{bf8,fp8}_{bf8,fp8}` (CDNA4) | 64 | 32 |

No SMFMAC kernels found in `hsa/gfx{942,950}` — the aiter-amd repo does not currently exploit 4:2 sparsity. Open opportunity.

### 6.8 MFMA dependency rules [pdf:p74-77, Table 38]

The critical few (HW does NOT enforce automatic stalls beyond what's listed; user/compiler must insert NOPs):

| First | Second | Required cycles |
|---|---|---|
| Non-DLop VALU write VGPR | DLop reads VGPR (any operand) | 2 |
| DLop writes VGPR | DLop reads same opcode, same SrcC | 0 |
| DLop writes VGPR | DLop reads same opcode, SrcA/B | 3 |
| DLop writes VGPR | Different opcode or non-DLop read | 3 |
| XDL write VGPR | XDL reads exact same SrcC VGPR | 0 (HW handles) |
| XDL write VGPR | XDL reads overlapped SrcC | 4 / 6 / 10 / 18 (depends on overlap pattern) |
| XDL write VGPR | SrcA/B read | 5 / 8 / 12 / 20 |
| SGEMM write VGPR | SGEMM reads exact same SrcC | 2 / 0 / 0 / 0 (per-opcode pass count) |
| SGEMM write VGPR | SGEMM reads overlapped SrcC | 2 / 4 / 8 / 16 |
| F64 MFMA 16x16x4 → same SrcC | F64 MFMA 16x16x4 | 0 (2 HW cycles internal) |
| F64 MFMA 4x4x4 → same SrcC | F64 MFMA 4x4x4 | 4 |
| `V_CMPX` writes EXEC | V_MFMA execute | 4 |
| XDL reads SrcC | VALU writes overlapping (WAR) | 1 / 3 / 7 / 15 |

The CK_tile `s_nop 3` matches "DLop SrcA/B" and "different opcode" — the safe minimum.

### 6.9 MFMA forces RNE rounding [pdf:p51, p68]

Every MFMA — F32, F16, BF16, I8, F64, F8/F6/F4 mixed — **ignores MODE.round_mode and MODE.denorm_mode**:
- Round: forced RNE
- Denorm: flush input + output (except F64 which allows denorms)
- Clamp: supported via FP16_OVFL bit (for FP16 output)

This is important because it means MFMA results may differ from a manual `v_fma_f32` chain that respects MODE — for bit-exact correctness checking against a reference implementation, ensure the reference uses RNE+FTZ.

## 7. Buffer descriptor (V#) — 128 bits [pdf:p91, Table 48 in §9.1.8]

V# is held in 4 consecutive SGPRs (aligned mod 4). Fields:

| Bits | Field | Semantics |
|---|---|---|
| 47:0 | `base_address` | 48-bit byte address. LSBs ignored (≥dword-align). |
| 61:48 | `stride` | Bytes per record (0–16383). Extended to 18 bits if ADD_TID_EN=1 & non-format MUBUF. |
| 62 | `cache_swizzle` | 0=linear, 1=swizzle TC L1 banks |
| 63 | `swizzle_enable` | 0=linear addressing, 1=apply swizzle formula |
| 95:64 | `num_records` | Bounds check denominator |
| 98:96 | `dst_sel_x` | Format channel select: 0=0, 1=1, 4=R, 5=G, 6=B, 7=A |
| 101:99 / 104:102 / 107:105 | `dst_sel_y/z/w` | Same |
| 110:108 | `nfmt` | Numeric format: 0=unorm, 1=snorm, 2=uscaled, 3=sscaled, 4=uint, 5=sint, 7=float |
| 114:111 | `dfmt` | Data format: 0=invalid, 1=8, 2=16, 3=8_8, 4=32, 5=16_16, 6=10_11_11, 7=11_11_10, 8=10_10_10_2, 9=2_10_10_10, 10=8_8_8_8, 11=32_32, 12=16_16_16_16, 13=32_32_32, 14=32_32_32_32 |
| 115 | `user_vm_enable` | Tiled pool/heap routing |
| 116 | `user_vm_mode` | 0=null (return 0/drop), 1=invalid (error) |
| 118:117 | `index_stride` | Swizzled addressing: 00=8, 01=16, 10=32, 11=64 indices per element |
| 119 | `add_tid_enable` | 0=no, 1=add lane ID to index automatically |
| 122:120 | reserved | Must be 0 |
| 123 | `nv` | Non-volatile (atomic ordering) |
| 125:124 | reserved | Must be 0 |
| 127:126 | `type` | Must be 0 for buffer |

### 7.1 Buffer addressing [pdf:p86-90, §9.1.5]

```
Final address = V#.base_address
              + SOFFSET  (SGPR or M0 byte offset)
              + (OFFEN ? vgpr_offset : 0)        // per-lane
              + inst_offset                        // 12-bit immediate
```

IDXEN/OFFEN combos [pdf:p85, Table 42]:

| IDXEN | OFFEN | VADDR | VADDR+1 | Use |
|---|---|---|---|---|
| 0 | 0 | — | — | No dynamic offset |
| 0 | 1 | offset | — | Raw buffer, per-lane offset |
| 1 | 0 | index | — | Structured buffer, index × stride |
| 1 | 1 | index | offset | Structured, both |

### 7.2 OOB behavior [pdf:p83, §9.1.5.1]

Hardware bounds-checks at **element granularity**:
- Read OOB → returns 0 (or 1 if V#.dst_sel=1) **no fault**
- Write OOB → dropped, no fault
- Atomic OOB → dropped

This is the basis for the AITER trick of **using OOB-zero to mask without `s_and_saveexec`** — cheaper by 2–4 instructions per load.

### 7.3 Memory scope SC[1:0] + NT [pdf:p92-94, Tables 49-50]

Three bits encode load/store/atomic cache behavior:

| SC1 | SC0 | Scope | Use |
|---|---|---|---|
| 0 | 0 | Wave | Single wave's data |
| 0 | 1 | Group | Workgroup-shared |
| 1 | 0 | Device | Single-GPU collective |
| 1 | 1 | System | **Cross-GPU peer atomic, host-coherent** |

| NT | Effect |
|---|---|
| 0 | Cache LRU (expect reuse) |
| 1 | Stream / cache-bypass-on-hit |

Loads (Table 49 [pdf:p93]): full 8-row cross product of (Wave/Group/Device/System) × (NT=0/1) showing CU/L2/last-level-cache behavior. Stores Table 50 similar.

Atomic-specific:
- SC0=1 → return pre-op value to VDATA
- SC0=0 → no return (fire-and-forget atomic)

### 7.4 Buffer load to LDS (`*_lds`) [pdf:p91-92, §9.1.9]

Direct DRAM → LDS routing, bypassing VGPRs:

```
LDS address per lane = M0[17:0] + TIDinWave * payload_bytes_per_thread
Mem address          = V#.base + SOFFSET + (IDXEN ? VADDR : 0)·stride + (OFFEN ? VADDR_next : 0)
```

For dwordx3/x4: per-thread LDS stride is 16 bytes (LDS_offset = M0 + TIDinWave*16). Single-dword: 4 bytes.

Supported widths: ubyte, sbyte, ushort, sshort, dword, dwordx3, dwordx4, format_x. Saves VGPR pressure — no intermediate register held.

Real use in [kernel-analysis/disassembly/fmoe/kernel.isa:302-318](_):
```asm
buffer_load_dword v36, s[20:23], 0 offen lds          ; per-block scale 0 → LDS
s_add_u32 m0, 0x100, s50                              ; advance LDS pointer
buffer_load_dword v37, s[20:23], 0 offen lds          ; scale 1 → LDS at +256B
...
```

CK_tile wraps this in `async_buffer_load_dwordxN_v` with a leading `s_nop 4` to satisfy the "SALU writes M0" → "LDS-typed op" 1-NOP rule from the manual-NOP table.

### 7.5 Global atomics (the FMOE scatter pattern)

`global_atomic_pk_add_bf16` and its f16 cousin write **two bf16 lanes per atomic** (32 bits total). Used in FMOE for the expert-weighted output scatter:

```asm
global_atomic_pk_add_bf16 v80, v64, s[8:9]           ; output[v80] += v64[15:0] || v64[31:16]
global_atomic_pk_add_bf16 v80, v65, s[8:9] offset:256 ; output[v80 + 256B] += v65 packed
```

Rounding RNE; denorms pass-through [pdf:p95-96, §9.2]. For float-atomic interactions with NaN see Table 53.

### 7.6 Wait-count and `s_waitcnt_vscnt`

No separate VMEM-store counter on CDNA4. The 6-bit unified `vmcnt` decrements when the store hits L2 (not when it's globally visible — but that's what fences on remote ranks see anyway in the peer-atomic pattern).

## 8. LDS — bank model and complete DS_* table

### 8.1 LDS architecture [pdf:p103-107 + CK_tile arch.hpp]

| Param | gfx942 | gfx950 |
|---|---|---|
| LDS / CU | typical 64 KB | 160 KB |
| Banks | 32 | **64** |
| Bank width | 4 B (dword) | 4 B |
| Bank addr | (byte_addr >> 2) mod {32,64} | |
| Conflict latency | 4–8 cycles for single conflict; up to 64 cycles worst case (all 64 lanes one bank) | |

Bank conflict mitigation tricks observed in disassembly:
1. **Non-power-of-2 stride** — rows padded to 132 dwords (528 B) instead of 128 → consecutive lanes hit distinct banks
2. **V# swizzle bit** (V#[63]) — hardware XOR re-map at descriptor level
3. **DS_READ2_B32 with non-conflicting offsets** — both `OFFSET0` and `OFFSET1` chosen so banks differ
4. **`DS_READ_TR*` family (CDNA4-new)** — see §8.3

### 8.2 M0 register

LDS access uses M0[31:16] = LDS_SIZE (clamp) and M0[15:0] = LDS_BASE (segment offset). For unindexed LDS in a kernel: `s_mov_b32 m0, 0xFFFFFFFF` (no clamp).

M0 also drives `buffer_load_*_lds` (the LDS destination offset) and `s_sendmsg` payload.

Manual-NOP table requires: `SALU writes M0` → `LDS add-TID op` or `buffer_store_LDS_dword` or `scratch/global LDS=1` requires **1 NOP**.

### 8.3 DS_* instruction set (full [pdf:p465-502, §12.12])

Reads:
- `DS_READ_B32`, `_B64`, `_B96`, `_B128`
- `DS_READ_U8 / I8 / U16 / I16` (sign- or zero-extend)
- `DS_READ_U8_D16 / U8_D16_HI / I8_D16 / I8_D16_HI` (write low or high 16-bit half of dst, preserve other)
- `DS_READ2_B32`, `_B64` — two offset gather (16-bit OFFSET0, OFFSET1, scaled by ADJ=4 for ≤32-bit, 8 for 64-bit)
- `DS_READ2ST64_B32`, `_B64` — 64-element stride: OFFSET*256 (32-bit) or 512 (64-bit)
- `DS_READ_ADDTID_B32` — `LDS_addr = LDS_BASE + {OFFSET1,OFFSET0} + laneID*4`
- **`DS_READ_B64_TR_B4`** — CDNA4 transpose load, 4-bit data
- **`DS_READ_B96_TR_B6`** — CDNA4 transpose load, 6-bit data, **no even-VGPR alignment required**
- **`DS_READ_B64_TR_B8`** — CDNA4 transpose load, 8-bit
- **`DS_READ_B64_TR_B16`** — CDNA4 transpose load, 16-bit

Writes:
- `DS_WRITE_B8 / B16 / B32 / B64 / B128`
- `DS_WRITE_B8_D16_HI / B16_D16_HI` (write from high 16-bit half of source)
- `DS_WRITE2_B32`, `_B64` — two-offset scatter
- `DS_WRITE2ST64_*` — stride writes
- `DS_WRITE_ADDTID_B32` — per-lane TID-offset write

Atomics (all have RTN return-value variants):
- `DS_ADD_*`, `SUB_*`, `RSUB_*`, `INC_U32`, `DEC_U32`
- `DS_AND_*`, `OR_*`, `XOR_*`, `MSKOR_*`
- `DS_MIN_*`, `MAX_*` (signed and unsigned, plus F32, F64)
- `DS_ADD_F32`, `MIN_F32`, `MAX_F32`, `ADD_F64`, `MIN_F64`, `MAX_F64`
- `DS_CMPST_B32/B64/F32/F64` (compare-store)
- `DS_WRXCHG_RTN_B32` (write-exchange-return)
- `DS_WRXCHG2_RTN_B32`, `WRXCHG2ST64_RTN_B32`
- `DS_PK_ADD_F16` / `DS_PK_ADD_BF16` — packed atomic adds with optional RTN

Cross-lane (no real LDS storage; uses LDS routing hardware):
- **`DS_PERMUTE_B32`** — forward: lane `i` writes to lane `src_lane[i]`
- **`DS_BPERMUTE_B32`** — backward: lane `i` reads from lane `src_lane[i]`
- **`DS_SWIZZLE_B32`** — fixed patterns (FFT butterfly, rotate, quad swizzle, 32-lane XOR/OR/AND-based)

Other:
- `DS_NOP`, `DS_GWS_*` (global-wave-sync), `DS_APPEND`, `DS_CONSUME`, `DS_ORDERED_COUNT`

### 8.4 MFMA Transpose Load (CDNA4-new) [pdf:p106, §11.4]

Eliminates the in-register transpose that classic MFMA layouts require. Pre-conditions:
- EXEC must be all-ones
- LDS address aligned to element size
- Even-VGPR alignment for 64-bit (except `DS_READ_B96_TR_B6` which doesn't require it)

| Inst | Element | K-pattern (each of 2 instructions covers half) |
|---|---|---|
| `DS_READ_B64_TR_B4` | 4-bit | K={0..15,32..47} then K={16..31,48..63} |
| `DS_READ_B96_TR_B6` | 6-bit | K={0..15,32..47} then K={16..31,48..63} (3 VGPRs each) |
| `DS_READ_B64_TR_B8` | 8-bit | K={0..7,16..23,32..39,48..55} then rest |
| `DS_READ_B64_TR_B16` | 16-bit | K={0..3,8..11} then K={4..7,12..15} (4 M or N per lane) |

Two instructions to complete one 64-element matrix load. Result: lane data lands in **exact MFMA input order**, no `v_perm_b32` shuffle needed afterward.

This is what gfx950 PA/MLA/FMOE kernels should be moving to for bf16/fp8 K/V loads — the existing kernels still use `ds_read_b128/b64` + manual interleave.

### 8.5 DS encoding [pdf:p600, §13.4.1]

```
[7:0]    OFFSET0      (8-bit unsigned byte offset)
[15:8]   OFFSET1      (8-bit; combines with OFFSET0 to 16-bit for single-addr)
[24:17]  OP           (opcode 0–255)
[25]     ACC          (VDST is AccVGPR)
[31:26]  ENCODING     (must be 110110 = 0x36)
[39:32]  ADDR         (VGPR holding byte address)
[47:40]  DATA0
[55:48]  DATA1
[63:56]  VDST
```

So `ds_read_b32 v36, v228 offset:1024` encodes OFFSET0=0x00, OFFSET1=0x04 (1024 = 0x400 = {0x04, 0x00}), op=54 (DS_READ_B32).

## 9. DPP — the cross-lane workhorse

### 9.1 DPP modifier encoding [pdf:p564-565, §12.16 + p599-600, §13.3.9]

DPP is a **64-bit suffix word** appended to VOP1/VOP2/VOPC instructions. Bit layout of the DPP word [pdf:p599-600, Table 93]:

| Bits | Field |
|---|---|
| [39:32] | SRC0 (real source-0 VGPR) |
| **[48:40]** | **DPP_CTRL** (9-bit, see enumeration below) |
| **[51]** | **BC** (Bounds Control: 0 = don't write on OOB, 1 = write with 0/wraparound) |
| **[52]** | SRC0_NEG |
| **[53]** | SRC0_ABS |
| **[54]** | SRC1_NEG |
| **[55]** | SRC1_ABS |
| [59:56] | BANK_MASK (bit i=0 disables lanes in bank i; 4 lanes × 8 banks) |
| [63:60] | ROW_MASK (bit i=0 disables row i; 16 lanes × 4 rows) |

DPP_CTRL[8:0] enumeration [pdf:p600, Table 94]:

| Value | Name | Meaning |
|---|---|---|
| 0x000–0x0FF | quad_perm:[a,b,c,d] | Per 4-lane quad, take source from lane `a,b,c,d`. Bits [1:0]=a, [3:2]=b, [5:4]=c, [7:6]=d |
| 0x101–0x10F | row_shl:N | Shift left by N within 16-lane row (N ∈ {1..15}). OOB → bound_ctrl |
| 0x111–0x11F | row_shr:N | Shift right by N within 16-lane row |
| 0x121–0x12F | row_ror:N | Rotate right by N (no OOB; wraps) |
| 0x130 | wave_shl1 | 64-lane shift left by 1 |
| 0x134 | wave_rl1 | 64-lane rotate left by 1 |
| 0x138 | wave_shr1 | 64-lane shift right by 1 |
| 0x13C | wave_rr1 | 64-lane rotate right by 1 |
| 0x140 | row_mirror | Mirror within 16-lane row: lane i ← lane (15−i) |
| 0x141 | row_half_mirror | Mirror within 8-lane half: lane i ← lane (7−(i&7)) |
| 0x142 | row_bcast15 | Broadcast lane 15 to rows 1–3 |
| 0x143 | row_bcast31 | Broadcast lane 31 to rows 2–3 |
| 0x150–0x15F | row_bcast:N | Broadcast lane N to entire row (N ∈ {0..15} in low 4 bits) |

(BANK_MASK / ROW_MASK / BC / SRC*_NEG / SRC*_ABS bit positions as in the table above.)

### 9.2 Decoded examples from real disassembly

From [kernel-analysis/disassembly/pa/kernel.isa:378-410](_) (PA softmax max-reduction):

```asm
v_mov_b32_dpp v38, v44 row_shr:4 row_mask:0xf bank_mask:0xf
;  encoding 7E4C02FA FF01142C → DPP_CTRL=0x114 (row_shr, N=4), all masks enabled
```

From [kernel-analysis/disassembly/pa/kernel.isa:157](_):
```asm
v_mov_b32_dpp v9, v9 row_shl:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
;  DPP_CTRL = 0x108 (row_shl, N=8), bound_ctrl=1 → OOB lanes masked
```

From [kernel-analysis/disassembly/ar/kernel.isa:1040-1058](_) (allreduce butterfly):
```asm
v_mov_b32_dpp v216, v36 row_ror:2          ; DPP_CTRL=0x122
v_add_f32_e32 v36, v216, v36
v_mov_b32_dpp v216, v36 row_ror:1          ; DPP_CTRL=0x121
v_add_f32_e32 v36, v216, v36
```

This is a butterfly reduction: rotate by 2, add; rotate by 1, add. After 6 such passes the sum across all 64 lanes lands in every lane.

### 9.3 DPP latency [pdf:p29]

Manual-NOP rule: **VALU writes VGPR → VALU DPP reads same VGPR: 2 NOPs**. CK_tile inserts `s_waitcnt vmcnt(0)` (overkill but safe) before chaining DPP operations on freshly-produced VGPRs.

### 9.4 SDWA — Sub-Dword Addressing companion

Lets VALU ops access 8- or 16-bit lanes of a 32-bit VGPR. Field encoding:
- `SRC0_SEL[1:0]`, `SRC1_SEL[1:0]`, `VDST_SEL[1:0]` — lane select (0=byte0, 1=byte1, 2=byte2, 3=word/dword)

Many VALU ops cannot use SDWA (see [pdf:p564] limitations) including `V_READFIRSTLANE_B32`, MAC/MADMK/MADAK families.

### 9.5 Cross-lane reduction recipes (with measured cycle costs)

| Recipe | Use | Cycles (approx) |
|---|---|---|
| **Butterfly XOR (DPP row_shr halving)** | 64-lane sum/max | 6 stages × (1 issue + 2 wait) ≈ 18 cycles |
| **`v_max3_f32` cascade** | 3-at-a-time reduction (PA softmax) | 64 → 22 → 8 → 3 → 1 in 4 steps ≈ 12 cycles |
| **`ds_swizzle_b32` (hardcoded FFT/rotate)** | 1-step swap-like permute | 1 issue + 1 wait ≈ 2 cycles (no LDS banks) |
| **`ds_bpermute_b32`** | Arbitrary backward permute via LDS routing | 1 issue + 1 wait ≈ 2 cycles |
| **LDS write+barrier+read** | Cross-wave reduction | 3+ stages + barrier ≈ 15–20 cycles |
| **`v_readfirstlane` + scalar broadcast** | Extract lane 0 to SGPR | 1 issue + 0 wait ≈ 1 cycle |

PA decode uses the v_max3 cascade for intra-wave max [kernel-analysis/disassembly/pa/kernel.isa:382-410](_); allreduce uses both DPP butterflies (intra-wave) AND LDS+barrier (cross-wave) [kernel-analysis/disassembly/ar/kernel.isa:879-1058](_).

## 10. VOP3P — packed math + dot products

### 10.1 Encoding [pdf:p594, §13.3.6]

64-bit instruction. Dword 0 fields:
- `[7:0]` VDST
- `[10:8]` NEG_HI[2:0] — negate (or abs in MAD_MIX) for high-lane sources
- `[13:11]` OPSEL[2:0] — low-lane source select (0=src[15:0], 1=src[31:16])
- `[14]` OPSEL_HI2
- `[15]` CLAMP
- `[22:16]` OP[6:0]
- `[31:24]` ENCODING = 0xD7

Dword 1 fields:
- `[40:32]` SRC0 (9-bit)
- `[49:41]` SRC1
- `[58:50]` SRC2
- `[60:59]` OPSEL_HI[1:0]
- `[63:61]` NEG[2:0] — low-lane negate

### 10.2 Packed F16

| Opc | Name |
|---|---|
| 14 | `V_PK_FMA_F16` |
| 15 | `V_PK_ADD_F16` |
| 16 | `V_PK_MUL_F16` |
| 17 | `V_PK_MIN_F16` |
| 18 | `V_PK_MAX_F16` |
| 27 | `V_PK_MINIMUM3_F16` |
| 28 | `V_PK_MAXIMUM3_F16` |

Plus `V_PK_FMAC_F16` (FMA-accumulate, VOP2 form, opcode 60).

### 10.3 Packed F32 (operate on two f32 lanes per VGPR pair) — even-aligned

| Opc | Name |
|---|---|
| 48 | `V_PK_FMA_F32` |
| 49 | `V_PK_MUL_F32` |
| 50 | `V_PK_ADD_F32` |
| 51 | `V_PK_MOV_B32` |

Real use: AllReduce+RMSNorm kernel sums via `v_pk_add_f32` [kernel-analysis/disassembly/ar/kernel.isa:452](_); RMSNorm sum-of-squares via `v_pk_fma_f32 v[38:39], v[4:5], v[4:5], v[38:39]` [kernel-analysis/disassembly/ar/kernel.isa:850-862](_) (square-and-accumulate, two lanes per issue).

### 10.4 Packed I16

`V_PK_MAD_{I,U}16`, `V_PK_ADD_{I,U}16`, `V_PK_SUB_{I,U}16`, `V_PK_MUL_LO_U16`, `V_PK_LSHLREV_B16`, `V_PK_LSHRREV_B16`, `V_PK_ASHRREV_I16`, `V_PK_MAX_{I,U}16`, `V_PK_MIN_{I,U}16`.

### 10.5 Dot products (2/4/8-lane)

| Opc | Name | Inputs | Output |
|---|---|---|---|
| 26 | `V_DOT2_F32_BF16` | 2 bf16 / 2 bf16 | f32 |
| 35 | `V_DOT2_F32_F16` | 2 f16 / 2 f16 | f32 |
| 38 | `V_DOT2_I32_I16` | 2 i16 / 2 i16 | i32 |
| 39 | `V_DOT2_U32_U16` | 2 u16 / 2 u16 | u32 |
| 40 | `V_DOT4_I32_I8` | 4 i8 / 4 i8 | i32 |
| 41 | `V_DOT4_U32_U8` | 4 u8 / 4 u8 | u32 |
| 42 | `V_DOT8_I32_I4` | 8 i4 / 8 i4 | i32 |
| 43 | `V_DOT8_U32_U4` | 8 u4 / 8 u4 | u32 |

All `V_DOT*` take SRC2 as accumulator and add the dot product to it. CDNA4 does NOT add new fp8 dot products at this VOP3P level — for fp8 you go straight to MFMA.

### 10.6 OPSEL semantics

For `v_pk_fma_f16 v0, v1, v2, v3 op_sel:[a,b,c] op_sel_hi:[d,e,f]`:
- OPSEL[0]=a → SRC0[15:0] if 0, SRC0[31:16] if 1 (for the low lane of result)
- OPSEL[1]=b → same for SRC1
- OPSEL[2]=c → same for SRC2
- OPSEL_HI[0]=d, [1]=e, [2]=f → same for the high lane

OPSEL_HI for `V_MAD_MIX_F32`: bits select 32-bit full or 16-bit half (with NEG_HI repurposed as ABS modifier in that family).

### 10.7 OMOD and CLAMP

- **VOP3A has OMOD** (output multiplier: ×1, ×2, ×4, ÷2). VOP3P does NOT.
- **CLAMP** (bit 15): saturate FP to [0,1], unsigned int to [0, 2^N−1], signed to [−2^(N−1), 2^(N−1)−1].
- MFMA: CLAMP supported but uses FP16_OVFL bit; OMOD ignored.

## 11. Packed conversions and fast-math

### 11.1 F32 ↔ F16 / BF16 / FP8 / BF8 / FP4 / FP6

| Instruction | Action |
|---|---|
| `V_CVT_PK_F32_F16` | 2 f16 → 2 f32 (low + high) |
| `V_CVT_PK_F16_F32` | 2 f32 → 2 f16 (RNE) |
| `V_CVT_PK_BF16_F32` | 2 f32 → 2 bf16 (RNE truncate) |
| `V_CVT_PK_F32_FP8` | 2 fp8 (E4M3) → 2 f32 |
| `V_CVT_PK_FP8_F32(a, b, prev, hi_lo)` | 2 f32 → 2 fp8 packed; `hi_lo` selects which 16-bit half of `prev` to overwrite |
| `V_CVT_PK_F32_BF8` | 2 bf8 (E5M2) → 2 f32 |
| `V_CVT_PK_BF8_F32` | 2 f32 → 2 bf8 |
| `V_CVT_PK_F16_FP8` | 2 fp8 → 2 f16 |
| `V_CVT_PK_FP8_F16` | 2 f16 → 2 fp8 |
| `V_CVT_PKRTZ_F16_F32` | 2 f32 → 2 f16 (round-to-zero) |
| `V_CVT_PKACCUM_U8_F32` | f32 + accumulator → packed u8 |
| `V_CVT_SCALEF32_PK_FP8_F32` | 2 f32 + scale → 2 fp8 with E8M0 bias |
| `V_CVT_SCALEF32_SR_FP8_F32` | 2 fp8 + scale → 2 f32 (scale-reverse) |

Quick-allreduce CodecFP8 uses these directly:
```cpp
qw[0] = __builtin_amdgcn_cvt_pk_fp8_f32(wf[0], wf[1], qw[0], 0);
qw[0] = __builtin_amdgcn_cvt_pk_fp8_f32(wf[2], wf[3], qw[0], 1);
```
4 f32 → 4 fp8 in two issues (one for each 16-bit half of `qw[0]`).

### 11.2 Fast-math intrinsics

All Trans Ops [pdf:p29]: `v_exp_f32`, `v_log_f32`, `v_rcp_f32`, `v_rsq_f32`, `v_sqrt_f32`, `v_sin_f32`, `v_cos_f32` + F16 and legacy variants.

Important: **`v_exp_f32` computes exp2(x), not exp(x).** CK_tile makes this explicit in the FMHA softmax:

```cpp
#if CK_TILE_FMHA_FWD_FAST_EXP2
    p_compute(i_j_idx) = exp2(scale_s * s[i_j_idx] - row_max);
#else
    p_compute(i_j_idx) = exp(scale_s * s[i_j_idx] - row_max);
#endif
```

For natural-base exp, multiply argument by `log2(e) ≈ 1.4427` before `v_exp_f32`. Pre-scaling `scale_s` by this constant collapses the entire softmax exp into a single Trans op.

Latency: Trans → non-Trans consumer needs **1 NOP** (Trans pipeline is shared).

## 12. SALU + integer arithmetic + address computation

### 12.1 Common SALU instructions (subset of [pdf:p32-38])

Integer:
- `S_MOV_B32 / B64`
- `S_ADD_{I,U}32`, `S_SUB_{I,U}32`, `S_ADDC_U32`, `S_SUBB_U32` (carry through SCC)
- `S_MUL_I32` (32×32→32 low only; no scalar MUL_HI)
- `S_MIN_{I,U}32`, `S_MAX_{I,U}32`, `S_ABSDIFF_I32`

Bitwise:
- `S_AND_B{32,64}`, `S_OR_*`, `S_XOR_*`, `S_NAND_*`, `S_NOR_*`, `S_XNOR_*`
- `S_ANDN1_*`, `S_ORN1_*`, `S_ANDN2_*`, `S_ORN2_*`
- `S_NOT_B{32,64}`, `S_BREV_B{32,64}`

Shifts:
- `S_LSHL_B{32,64}`, `S_LSHR_B{32,64}`, `S_ASHR_I{32,64}`

Bit-field:
- `S_BFM_B{32,64}` (mask: `((1<<S0)-1)<<S1`)
- `S_BFE_{U,I}{32,64}` (extract: width=S1[22:16], offset=S1[5:0])
- `S_BCNT0/1_I32_B{32,64}` (count zero/one bits)
- `S_FF0/1_I32_B{32,64}` (find first zero/one from LSB; -1 if not found)
- `S_FLBIT_I32_B{32,64}` (leading-bit count)

Compares (set SCC):
- `S_CMP_{EQ,NE,GT,GE,LT,LE}_{I,U}{32,64}`
- `S_BITCMP0/1_B{32,64}`

Conditional moves:
- `S_CSELECT_B{32,64}` (D = SCC ? S0 : S1)
- `S_CMOV_B{32,64}` (if SCC then D = S0, else NOP)

EXEC save:
- `S_AND_SAVEEXEC_B64`, `S_OR_*`, `S_XOR_*`, `S_ANDN2_*`, etc.
- `S_ANDN1_WREXEC_B64`, `S_ANDN2_WREXEC_B64` (write EXEC and D simultaneously)

HW register access:
- `S_GETREG_B32` / `S_SETREG_B32` / `S_SETREG_IMM32_B32`
- HW reg IDs: 1=MODE, 2=STATUS, 3=TRAPSTS, 4=HW_ID, 5=GPR_ALLOC, 6=LDS_ALLOC, 7=IB_STS, 16–19=TBA/TMA, 20=XCC_ID, 21–24=PERF_SNAPSHOT_*

Scalar memory:
- `S_LOAD_DWORD/X2/X4/X8/X16` (offset 21-bit signed immediate, or SGPR holding unsigned byte offset)
- `S_BUFFER_LOAD_*` (load via V# descriptor)
- `S_SCRATCH_LOAD_*` (per-wave scratch)
- `S_ATOMIC_*` (scalar atomics — used for split-K counters)
- `S_DCACHE_INV` / `S_DCACHE_WB`
- `S_MEMTIME` / `S_MEMREALTIME` (64-bit timestamps)

### 12.2 Common address-compute patterns

```asm
v_lshlrev_b32_e32 v0, 4, v1          ; v0 = v1 << 4 (per-lane offset = lane × 16)
v_mul_u32_u24_e32 v0, s2, v1         ; 24-bit unsigned multiply via FP32 pipe (fast)
v_mul_i32_i24_e32 v0, s2, v1         ; signed 24-bit
v_mul_hi_u32_u24_e32 v0, s2, v1      ; high 32 bits
v_mad_u32_u24_e32 v0, v1, v2, v3     ; single-instruction multiply-add
v_add_co_u32_e32 v0, vcc, v1, v2     ; add with carry-out to VCC
v_addc_co_u32_e32 v3, vcc, v4, v5, vcc ; add-with-carry-in (composes 64-bit add)
v_readfirstlane_b32 s0, v0           ; extract lane 0 → SGPR
v_mul_u32_u24_dpp v0, v1, v2 row_newbcast:0 row_mask:0xf bank_mask:0xf ; 24-bit mul + broadcast lane 0
v_max3_f32 v0, v1, v2, v3            ; 3-way max
v_med3_f32 v0, v1, v2, v3            ; median (used for clamping: med3(x, lo, hi))
```

### 12.3 V_FMA family

- `V_FMA_F32` — fully IEEE-compliant FMA
- `V_FMA_F64`, `V_FMA_F16`, `V_FMA_LEGACY_F32`
- `V_FMAAK_F32` — FMA with one operand as instruction-stream constant
- `V_FMAMK_F32` — FMA with middle operand as constant
- `V_FMAC_F32` — FMA-accumulate (D += S0·S1)

`V_FMAAK_F32` and `V_FMAMK_F32` cannot use VOP3 form (no 3-source register form) — they avoid a separate `v_mov` to a register for the literal, saving 1 instruction and 1 VGPR.

### 12.4 V_MAX3 / V_MIN3 / V_MED3

For PA softmax (the 3-at-a-time max cascade speeds reduction):

```asm
v_max3_f32 v92, |v8|, |v9|, v92       ; |·| = absolute-value modifier (VOP3A)
v_max3_f32 v92, |v10|, |v11|, v92
v_max3_f32 v92, |v12|, |v13|, v92
v_max3_f32 v92, |v14|, |v15|, v92
```

4 instructions reduce 8 values to 1 — better than 7 v_max_f32 ops, with same single-cycle throughput.

`V_MED3_*` is critical for clamping: `clamp(x, lo, hi) = med3(x, lo, hi)` in one instruction (otherwise 2 mins/maxes).

## 13. Inline-asm GCC constraints (CK_tile-style)

Constraint reference for hipcc inline asm:

| Char | Class | R/W |
|---|---|---|
| `"v"` | VGPR | read-only |
| `"a"` | AGPR (CDNA) | read-only |
| `"+v"` | VGPR | read-write |
| `"+a"` | AGPR | read-write (the C-accumulator pattern for MFMA) |
| `"=v"`, `"=a"` | write-only | |
| `"s"` | SGPR | read-only |
| `"=s"` | SGPR | write-only |
| `"n"` | immediate integer | (compiler picks best repr) |
| `"i"` | generic integer constant | |
| `"I"` | 8-bit signed immediate (−128..127) | |

The `"+a"` modifier on the destination of an inline `v_mfma_*` is what pins the C accumulator to AGPR — see [warp_gemm_attribute_mfma_impl.hpp](3rdparty/composable_kernel/include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp).

---

# Part III — Kernel pattern case studies

## 14. Multi-wave workgroup cooperation — the foundational pattern

The technique that makes everything else in Part III possible. A workgroup of more than one wave (typically 4 or 16 waves on CDNA) lets multiple waves on a single CU share LDS, coordinate via `s_barrier`, and divide work along orthogonal axes. **Every kernel in §15-§20 depends on this; v1 of the playbook left it implicit.** When the playbook's resource-envelope table at §1 shows "256 threads" or "1024 threads" for a kernel, this section explains *why those numbers are what they are.*

### 14.1 Why 4-wave WG is the inference sweet spot

CDNA has **4 SIMDs per CU**, each running one wave at a time. With:

| Waves/WG | Threads | SIMD coverage | Use case |
|---|---|---|---|
| 1 | 64 | 1 SIMD active in this WG | Memory-bound point-ops; GEMV when M is tiny; not common for ML inference |
| 2 | 128 | 2 SIMDs | Register-heavy kernels that can't fit 4 |
| **4** | **256** | **All 4 SIMDs of the CU** | **Sweet spot — PA decode, flatmm, FMOE, FMHA, TopK, MLA** |
| 8 | 512 | 4 SIMDs × 2 deep | Larger LDS regions, half the WG concurrency |
| 16 | 1024 | 4 SIMDs × 4 deep | Workgroup-wide reductions (AllReduce hidden=8192) |

A 1-wave WG occupies 1 SIMD; the other 3 SIMDs in the CU need 3 *different* WGs from the grid to be busy. If the grid is narrow (small batch × small num_heads), 1-wave WGs leave SIMDs idle → typically **<30% MFMA utilization on attention**. A 4-wave WG covers all 4 SIMDs of the CU within one WG, eliminating intra-CU SIMD-vacancy and allowing the SIMD scheduler to switch between the WG's waves on memory stalls.

This is why CK_tile's default `BlockSize` for inference GEMM/FMHA is 256 (= 4 waves), and why a quick way to tell whether a `.co` was tuned for inference vs prototype is to check its `group_segment_fixed_size` and workgroup-size hint in `.note`.

### 14.2 Five wave-specialization patterns

Real kernels in this repo use these orthogonal divisions of work across waves within a WG:

| Pattern | How waves divide work | Example | Disassembly |
|---|---|---|---|
| **Output-partition** (dominant) | Each wave produces a different M×N quadrant of the same output tile | CK_tile GEMM, FMHA `qr_ks_vs` — `NWarp` waves split N (or M) | [block_gemm_areg_bsmem_creg_v1.hpp:60-181] |
| **K-split** | Each wave handles a different K-slice; cross-wave LDS reduction at end | split-K i8 GEMM (cross-WG atomic add at output); MLA stage-1 inner K-loop | (⚠️ splitk disassembly not archived — see kernel-analysis/README.md to regenerate) |
| **Expert-partition (SMF)** | Pairs of waves each handle one expert; share weight loads via LDS interleave | FMOE `smf_subGU_320` — waves 0-1 → expert 0, waves 2-3 → expert 1 | [kernel-analysis/disassembly/smf/kernel.isa:2654-2661] (§17.1, formerly §16.1) |
| **TG-split** | Workgroup divided into two thread-groups handling disjoint Q-tile ranges | PA `2tg` decode (per-token fp8) | [kernel-analysis/disassembly/pa/kernel.isa] (§15, formerly §14) |
| **Cross-wave reduction** | All N waves accumulate into LDS, final reduce via LDS + DPP | AllReduce+RMSNorm (16-wave WG); TopK-softmax (4-wave WG butterfly) | [kernel-analysis/disassembly/ar/kernel.isa:879-901] (§20, formerly §19) |

A rarer **producer-consumer** pattern (one wave loads, another computes) exists in some FMHA-bwd variants and the FA3 fast-path, but is less common because the barrier choreography is harder. Most inference kernels use output-partition — every wave does the same compute, just on different output coordinates, with shared K/V tiles in LDS.

### 14.3 Lane / wave-id math

The bedrock encoding all five patterns depend on:

```c
// Decode the flattened threadIdx.x
int lane_id = threadIdx.x & 63;          // 0..63 within wave
int wave_id = threadIdx.x >> 6;          // 0..waves_per_block-1

// HW-intrinsic form (avoid VGPR roundtrip):
int lane_id = __builtin_amdgcn_mbcnt_hi(-1, __builtin_amdgcn_mbcnt_lo(-1, 0));
int wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x) >> 6;
```

The `v_readfirstlane_b32 sX, vY` instruction reads lane 0 of a VGPR into an SGPR, promoting wave-uniform values for SALU control flow. `wave_id` is wave-uniform by definition, so it lives in SGPR — letting `s_cmp` / `s_cbranch` make wave-specialized branches without per-lane masking.

Typical prologue in disassembly:
```asm
v_lshrrev_b32_e32 v3, 6, v0             ; v3 = tid >> 6 = wave_id (per lane, all same)
v_and_b32_e32 v0, 63, v0                ; v0 = tid & 63 = lane_id
v_readfirstlane_b32 s57, v3             ; s57 = wave_id (now in SGPR for SALU)
```

This pattern appears in every multi-wave kernel in the repo (PA, MLA, FMOE, AllReduce, TopK). The `v_readfirstlane_b32` is the linchpin: without it, branches on wave_id would be per-lane VALU compares (slow + lane-divergent), instead of single scalar branches.

### 14.4 Per-wave LDS layout

LDS is allocated **per workgroup**, but within that allocation you typically partition by wave:

```c
__shared__ char lds[LDS_BYTES_PER_WG];

// Option A: per-wave private region (TopK-style reductions, each wave its own scratch)
size_t lds_per_wave = LDS_BYTES_PER_WG / waves_per_block;
char* my_lds = &lds[wave_id * lds_per_wave];

// Option B: shared K/V tile (all waves read), per-wave Q-row partition (FMHA-style)
char* k_buf = &lds[0];                                 // shared
char* v_buf = &lds[K_BUF_SIZE];                        // shared
char* q_per_wave = &lds[K_BUF_SIZE + V_BUF_SIZE + wave_id * Q_PER_WAVE];

// Option C: SMF expert-partition interleave (FMOE smf_subGU_320)
// Expert 0 tiles at {2048, 4224, 6400, 8576}; expert 1 tiles at {10752, 12928, 15104, 17280}.
// Waves 0-1 → expert 0 slots; waves 2-3 → expert 1 slots. See §17.1.

// Option D: bank-conflict avoidance — pad each wave's stride to non-power-of-2
// (avoids cross-wave conflicts when multiple waves read the same column index)
size_t stride = 132 * sizeof(float);   // 132 dwords, not 128
```

CK_tile's `tile_distribution_encoding` (§22, formerly §23) encodes this layout at the type level: the `P` dimension splits into `(warp_id, lane_id)`, the `H` dimensions stack within each wave's region, and the adaptor generates the per-thread byte-offsets automatically. This is what makes `tile_window.load()` / `.store()` produce coalesced + non-conflicting LDS access without the user writing index math.

### 14.5 Barrier discipline

`s_barrier` does NOT wait for memory counters [pdf:p153]. Every wave must drain its own `s_waitcnt` FIRST before barrier-synchronizing:

```asm
; Wrong — barrier before LDS writes complete:
ds_write_b32 v0, v1
s_barrier                  ; other waves may read undefined data

; Right:
ds_write_b32 v0, v1
s_waitcnt lgkmcnt(0)       ; drain THIS wave's LDS writes
s_barrier                  ; safe — other waves see committed data
```

CK_tile's `block_sync_lds<lgkm=0>` and `block_sync_lds_direct_load<vm=0>` (§21, formerly §20) bundle the wait + barrier with **selective** counter classes — letting one class proceed while the other drains. This is foundational for the SW-pipelined hot loop: one wave can keep issuing async global→LDS copies (vmcnt) while another wave consumes data already in LDS (lgkmcnt).

### 14.6 Occupancy table

CDNA per-CU: 4 SIMDs, max 8 waves per SIMD = **32 waves resident per CU max**.

| Config | Waves/WG | WGs/CU | Total resident waves/CU | Notes |
|---|---|---|---|---|
| Heavy register kernel | 1 | up to 8 | 8 | Each WG on one SIMD; 3 SIMDs need other WGs to fill |
| **Standard inference** | **4** | **up to 8** | **32** | **Sweet spot — full SIMD coverage, max WG concurrency** |
| Larger LDS need | 8 | up to 4 | 32 | Same total wave count, larger contiguous LDS per WG |
| Workgroup-wide reduction | 16 | up to 2 | 32 | AllReduce N=8192; one WG covers a full chunk |
| Maximum WG | 16 | 1 (limit hit) | 16 | Only when register pressure forces it |

The "32 resident waves" cap is what hides memory latency on CDNA — the SIMD scheduler switches to another resident wave whenever one stalls on `s_waitcnt`. Below 32 resident waves you start seeing exposed `vmcnt` latency in profiling traces.

The configuration is decided by the kernel author at compile time via `__launch_bounds__(threads_per_block, blocks_per_cu)` or CK_tile's `kBlockPerCu` heuristic ([§14.3 of v1 / §15.3 of v2 kBlockPerCu heuristic in FMHA](3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp:114-143)). Picking too high a `blocks_per_cu` value forces register-file partitioning that may not match the kernel's actual VGPR need; too low and the CU is underutilized.

### 14.7 Cross-reference index

| Section | Multi-wave use |
|---|---|
| §15 (was §14) PA decode | `2tg` variant = TG-split (4 waves split into 2 TGs of 2 waves each); `1tg` variant = single-TG 4-wave cooperation |
| §16 (was §15) flatmm GEMM | Output-partition with 1×4×1 wave grid (4 waves along N) |
| §17 (was §16) FMOE SMF | Expert-partition (4 waves → 2 experts via LDS interleave §17.1) |
| §18 (was §17) TopK | Cross-wave reduction (4-wave DPP-butterfly via LDS, even though LDS allocation is 0 — uses DPP within waves and `ds_bpermute_b32` across) |
| §20 (was §19) AllReduce | Cross-wave reduction at maximum scale (16 waves, 1024 threads) |
| §23 (was §22) `tile_distribution` | The `(warp_id, lane_id)` P-decomposition that encodes all multi-wave patterns at the type level |
| §28 (was §27) catalog #97-101 | The five patterns named as catalog entries |

---

## 15. PA decode — per-token vs per-block fp8 vs bf16

### 15.1 The three variants disassembled

| | per-token fp8 (2tg) | per-block fp8 (1tg) | no-quant bf16 |
|---|---|---|---|
| MFMA | `v_mfma_f32_16x16x32_fp8_fp8` | same | `v_mfma_f32_16x16x16_bf16` (note: NOT the doubled-K `_16x16x32_bf16`) |
| Dequant pattern | nibble-mask: `s[44:45] = 0xF0F0F0F0` loaded at [kernel-analysis/disassembly/pa/kernel.isa:148-149](_), consumed by `v_cndmask_b32_e64` at L383-386 | per-block scale loaded once, broadcast | none — bf16 is native MFMA input |
| Scale-load bandwidth per K-tile | ~200-250 B (per-row scales) | **~4 B** (one f32 per 256-token block) | none |
| Thread-group count | 2 (queries 0-7, 8-15 split) | 1 (all 4 waves cooperate on 16 queries) | (verify) |
| `s_waitcnt` depth | vmcnt(8) | vmcnt(10-12) | implicit (~16) |
| Cross-TG sync | global atomic counter | LDS barriers (intra-TG) | LDS barriers |
| MFMA per K-tile | 16 (4 Q-rows × 4 K-chunks) | 16 | 16 (but with K=16 not K=32) |

### 15.2 Per-block fp8 prologue (key sequence) [kernel-analysis/disassembly/pa2/kernel.isa:278-284](_)

```asm
s_lshl_b32 s54, s72, 2                    ; s72 = block_idx, shift by 2 = ×4 bytes
s_load_dword s60, s[44:45], s54           ; ← single fp32 K-scale for this block
s_load_dword s61, s[40:41], s54           ; ← single fp32 V-scale for this block
s_lshl_b32 s68, s76, 2
s_cmp_lt_u32 s76, s77
s_cselect_b32 s68, s68, 0                 ; zero offset if past bounds (OOB guard)
s_addk_i32 s76, 0x1
s_load_dword s59, s[42:43], s68
```

That's the entire per-block scale fetch. Compare to per-token where the kernel would have ~16 `s_load_dword` instructions per K-tile.

### 15.3 Causal mask application [kernel-analysis/disassembly/pa2/kernel.isa:1638-1670](_)

Per-K-position check applied AFTER MFMA, BEFORE softmax:

```asm
v_add_u32_e32 v72, s64, v106             ; v72 = current Q index
v_cmp_lt_u32_e64 s[98:99], v73, v105     ; is q_idx < k_idx? (causal)
s_nop 0
v_cndmask_b32_e64 v16, v107, v16, s[98:99]  ; if future → set v16 = v107 (-inf)
```

`v107 = 0xff800000` (negative infinity for fp32), set once in the prologue. After softmax: `exp(-inf) = 0` → masked positions contribute zero. No branch needed in the hot loop.

### 15.4 Online softmax (dtype-independent)

PA's softmax structure is identical across fp8/bf16:

```asm
; Stage 1: per-lane max via v_max3_f32 cascade
v_mov_b32_e32 v92, 0xff800000             ; init max = -inf
v_max3_f32 v92, |v8|, |v9|, v92
v_max3_f32 v92, |v10|, |v11|, v92
v_max3_f32 v92, |v12|, |v13|, v92
v_max3_f32 v92, |v14|, |v15|, v92

; Stage 2: cross-wave max via LDS
ds_write_b32 v122, v92 offset:1280
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b32 v76, v123 offset:1280
...

; Stage 3: rescale + exp (exp2 → v_exp_f32)
v_mul_f32_e32 v50, s64, v15               ; scale × global_max
v_fma_f32 v88, v88, s64, -v50             ; (S-max)·scale, all 8 elements
...
v_exp_f32_e32 v88, v88
```

The bf16 PA kernel uses the same pattern with `v_max_f32` substituted for `v_max3_f32` in some places — instruction-mix counts match within 10%.

## 16. Flatmm GEMM 128×256×128 — the canonical fp8 GEMM ASM

### 16.1 Hot-loop instruction stream [kernel-analysis/disassembly/flatmm/kernel.isa:393-407](_)

```asm
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], a[0:1], a[64:65], 0          ; init C-tile from zero
buffer_load_dwordx4 a[16:19], v4, s[20:23], 0 offen                ; prefetch next K=8 of A
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], a[2:3], a[66:67], v[56:59]   ; accumulate K=8..15
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], a[4:5], a[68:69], v[56:59]   ; K=16..23
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], a[6:7], a[70:71], v[56:59]   ; K=24..31
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], a[8:9], a[64:65], 0          ; second output tile
buffer_load_dwordx4 a[20:23], v4, s[20:23], 0 offen offset:1024    ; prefetch
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], a[10:11], a[66:67], v[60:63]
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], a[12:13], a[68:69], v[60:63]
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], a[14:15], a[70:71], v[60:63]
```

Every 4 MFMAs there's a `buffer_load_dwordx4` prefetching the next K-stripe — software pipelining depth ≈ 4 MFMAs per outstanding load. `s_waitcnt vmcnt(16)` at the loop top lets up to 16 outstanding loads in flight.

### 16.2 No separate fp8 dequant; scales fused into compute path

CDNA's `v_mfma_f32_*_fp8_fp8` decodes fp8 inside the MFMA unit — there is **no** `v_cvt_pk_f32_fp8` step. The kernel applies scales in two places:

**Before the K-loop** (B-scale and A-scale broadcast applied to the loaded weight tiles, [kernel-analysis/disassembly/flatmm/kernel.isa:375-392](_)):
```asm
v_mov_b32_e32 v50, s31                    ; B-scale into v50
v_mul_f32_e32 v240, v224, v50             ; v240 = v224 (B-elt) × B-scale
v_mul_f32_e32 v241, v225, v50
... (8 such mults — v240..v247 holds scaled B)
v_mov_b32_e32 v50, s30                    ; A-scale
v_mul_f32_e32 v224, v224, v50             ; in-place rescale of A-elements
... (8 such mults)
```

**Inside the MFMA chain** (the `v_fmac_f32` ops at lines 400, 402, 404… interleave with the MFMAs and accumulate the scaled output into v[64:71]):
```asm
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], a[8:9], a[64:65], 0   ; line 398
buffer_load_dwordx4 a[20:23], v4, s[20:23], 0 offen offset:1024
v_fmac_f32_e32 v64, v56, v224                                ; line 400: v64 += v56·A-scale
v_mfma_f32_16x16x32_fp8_fp8 v[60:63], a[10:11], a[66:67], v[60:63]
v_fmac_f32_e32 v65, v57, v224                                ; line 402
```

The kernel uses **`v_mul_f32_e32`** (single, not packed) and **`v_fmac_f32_e32`** — not `v_pk_mul_f32`. `v_pk_mul_f32` does not appear in this binary (`grep -c v_pk_mul_f32 kernel-analysis/disassembly/flatmm/kernel.isa` = 0). The packed-math optimization opportunity *does* exist (would halve the ~16-instruction scale-multiply prologue), but the gfx9 compiler chose scalar-form here.

For per-block-scale fp8 GEMM the scales would be loaded once per K-block (the gfx950 `f8_block_scale_mi350_x{32,64,96,128}.co` family) and applied via the same `v_mul_f32` pattern at K-tile boundaries.

### 16.3 Output epilogue [kernel-analysis/disassembly/flatmm/kernel.isa:1108-1139](_)

```asm
v_cvt_f16_f32_e32 v3, v64                 ; f32 → f16
v_cvt_f16_f32_e32 v6, v65
v_pack_b32_f16 v6, v3, v6                 ; pack two f16 into one dword
...
global_store_dwordx2 v[4:5], v[6:7], off  ; 4 f16 outputs per lane = 64 lanes × 4 = 256 outputs
```

The output cast + pack happens entirely after the K-loop; no fp16 intermediate during accumulation.

## 17. FMOE blockscale / SMF / multix — three dispatch strategies decoded

The agent comparison revealed concrete ISA-level differences:

| Aspect | blockscale_novs_subGU_256 | smf_subGU_320 | multix_subGU_256 |
|---|---|---|---|
| ISA lines | 4423 | 5626 (+27%) | 4037 (−9%) |
| Max VGPR | 255 | 231 | 223 |
| Max AGPR | 127 | **159** | 127 |
| LDS bytes | 53,384 | **17,288** (−68%) | 42,120 |
| MFMA count | 768 | **960** (+25%) | 768 |
| `v_pk_mul_f32` count | 128 | **608** | 0 |
| `v_mul_f32_e32` count | 200 | 94 | 580 |
| `ds_read_b128` count | 40 | 0 | 40 |
| `ds_read_b64` count | 96 | **272** | 64 |
| `global_atomic_pk_add_bf16` count | 96 | 64 | 64 |
| `s_barrier` count | 23 | 31 | 21 |

### 17.1 SMF = Shared-Memory Fusion (confirmed at ISA level)

[kernel-analysis/disassembly/smf/kernel.isa:2654-2661](_) shows two expert tiles **interleaved** in LDS:

```asm
ds_write_b64 v3, v[168:169] offset:2048   ; Expert 0, tile A
ds_write_b64 v3, v[170:171] offset:10752  ; Expert 1, tile A
ds_write_b64 v3, v[172:173] offset:4224   ; Expert 0, tile B
ds_write_b64 v3, v[174:175] offset:12928  ; Expert 1, tile B
ds_write_b64 v3, v[176:177] offset:6400   ; Expert 0, tile C
ds_write_b64 v3, v[178:179] offset:15104  ; Expert 1, tile C
ds_write_b64 v3, v[180:181] offset:8576   ; Expert 0, tile D
ds_write_b64 v3, v[182:183] offset:17280  ; Expert 1, tile D
```

Two experts share a single LDS buffer with 8704-byte separation between their tile sets. Total LDS = 17.3 KB vs 53.4 KB for single-expert blockscale → **68% reduction**. SMF also uses `v_pk_mul_f32` (608 issues) extensively for the gate×sigmoid×up chain — processes two experts' gating in parallel per instruction.

The trade-off: +8 barriers (31 vs 23) to synchronize between the two experts within a TG, and +25% MFMA count due to larger `subGU=320` (vs 256).

### 17.2 SiLU activation (identical across all three FMOE variants)

```asm
v_mul_f32_e64 v56, -v128, s6                ; s6 = 0x3fb8aa3b (log2(e))
v_exp_f32_e32 v56, v56                      ; exp2(-gate · log2(e)) = e^(-gate)
v_add_f32_e64 v56, v56, 1.0                 ; 1 + e^(-gate)
v_rcp_f32_e32 v56, v56                      ; sigmoid(gate)
v_mul_f32_e32 v128, v128, v56               ; gate × sigmoid(gate) = SiLU(gate)
v_mul_f32_e32 v128, v128, v64               ; × up (g1u1 elementwise multiply)
```

5 VALU ops + 1 Trans (v_exp_f32) per gate element. The cost is **dominated by the `v_exp_f32`** (Trans pipeline, ~10 cycles). SMF saves issue count by replacing the two final `v_mul_f32_e32` with `v_pk_mul_f32 v[128:129], ...` to do two experts in parallel.

### 17.3 Output scatter

```asm
global_atomic_pk_add_bf16 v80, v64, s[8:9]               ; +bf16[0:1] to output[v80]
global_atomic_pk_add_bf16 v80, v65, s[8:9] offset:256    ; next pair
```

All three variants use the same global packed-bf16 atomic. SMF reduces total atomic count by 33% (64 vs 96) by processing two experts per launch.

## 18. TopK-softmax — DPP-only reduction with GPR-indexed scatter

[`topksoftmax_4x256x8_bf16.co`] — 1030 ISA lines, **0 MFMAs**, 0 LDS bytes, only 24 VGPR. This is a workload that wins by NOT using MFMA.

### 18.1 Per-token softmax

```asm
buffer_load_short_d16 v11, v8, s[12:15], 0 offen           ; load 1 bf16 gate logit
buffer_load_short_d16 v12, v8, s[12:15], 0 offen offset:128 ; lane-stride 128 bytes
buffer_load_short_d16 v13, v8, s[12:15], 0 offen offset:256
buffer_load_short_d16 v14, v8, s[12:15], 0 offen offset:384
... ; 4 bf16 logits per workitem, 256 workitems = 1024 bf16 covering 256 experts

; Unpack bf16 → fp32 by left-shift 16 bits
v_lshlrev_b32_e32 v11, 16, v11
v_mul_f32_e64 v11, v11, s57                              ; s57 ≈ 0x3FB8AA3B (1/sqrt(d) × log2e)
v_exp_f32_e32 v11, v11                                   ; exp2(scaled_logit)
v_add_f32_e32 v17, v17, v11                              ; accumulate sum
...

; DPP butterfly reduction (256 threads → 1)
v_add_f32_dpp v4, v17, v17 quad_perm:[1,0,3,2]           ; step 1: pairs swap
v_add_f32_dpp v4, v4, v4 quad_perm:[2,3,0,1]             ; step 2: quads
v_add_f32_dpp v4, v4, v4 row_shr:4                       ; step 3
v_add_f32_dpp v4, v4, v4 row_shr:8                       ; step 4
v_add_f32_dpp v4, v4, v4 row_bcast:15                    ; step 5
v_add_f32_dpp v4, v4, v4 row_bcast:31                    ; step 6

; Normalize: divide by sum
v_rcp_f32_e32 v17, v18
v_mul_f32_e32 v11, v11, v17
v_mul_f32_e32 v12, v12, v17
v_mul_f32_e32 v13, v13, v17
v_mul_f32_e32 v14, v14, v17
```

Zero LDS usage — everything happens in registers via DPP. Cycle estimate: ~30 cycles total reduction.

### 18.2 Top-K selection (K-pass selection sort)

```asm
v_max_f32_e32 v19, v11, v12              ; per-lane max of 4 weights
v_max3_f32 v19, v19, v13, v14            ; 3-way max

; Cross-lane max via DPP
v_max_f32_dpp v4, v19, v19 quad_perm:[1,0,3,2]
v_max_f32_dpp v4, v4, v4 quad_perm:[2,3,0,1]
...
v_readlane_b32 s20, v4, 63                ; pick the final max → scalar

; Find which lane held the max
v_cmp_eq_f32_e64 s[24:25], v19, v11
v_cmp_eq_f32_e64 s[26:27], v19, v12
v_cmp_eq_f32_e64 s[28:29], v19, v13
v_cmp_eq_f32_e64 s[30:31], v19, v14
s_ff1_i32_b64 s32, s[24:25]               ; find first 1 = lane index where v11 matched
s_ff1_i32_b64 s33, s[26:27]

; GPR-indexed scatter — write score/index pair to dynamic SGPR slot
s_set_gpr_idx_on s40, gpr_idx(DST)
v_writelane_b32 v11, 0, s22                ; scatter to lane s22 of v11
s_set_gpr_idx_off
```

Repeats K=8 times. Each pass: find max, mark it (set to −inf in the working register), find next max, repeat. **No LDS** at all; 8 (score, expert_index) pairs live in SGPR via GPR-indexed writes.

## 19. MLA — inline KV-LoRA decompression

DeepSeek MLA decompresses K in-kernel. The K-loop interleaves RoPE rotation with the standard PA structure.

### 19.1 K-RoPE rotation [kernel-analysis/disassembly/mla/kernel.isa (⚠️ not archived — regenerate; see kernel-analysis/README.md):1239-1250](_)

```asm
v_mul_f32_e32 v21, s5, v20             ; s5 = sin(theta_i), v20 = K_perp component
v_mul_f32_e32 v16, s5, v16             ; sin × K_alt
v_fma_f32 v32, v32, s5, -v21           ; K' = K · cos − K_perp · sin (rotation formula)
v_fma_f32 v33, v33, s5, -v21
v_fma_f32 v34, v34, s5, -v21
v_fma_f32 v35, v35, s5, -v21
v_mul_f32_e32 v14, v16, v14            ; complete rotation
```

`s5 = sin(theta_i)`, `s6 = cos(theta_i)` for position `i`, pre-computed and broadcast as scalars. Each Q row gets its sin/cos pair multiplied into the K component on the fly — no separate decompression kernel pass.

### 19.2 Two MFMA streams (QK and PV) with different K-dims

QK uses `qk_rope_head_dim` (typically 192 = 128 nope + 64 rope). PV uses `kv_lora_rank` (typically 512). The kernel issues **816 MFMAs total** (vs 416 for plain PA), reflecting the deeper accumulator chain on PV.

## 20. AllReduce + RMSNorm — fused-N=8192 hand-tuned ring

### 20.1 Peer pointer setup [csrc/include/custom_all_reduce_hip.cuh:36-42](csrc/include/custom_all_reduce_hip.cuh)

```cpp
struct Signal {
    alignas(128) uint32_t start[kMaxBlocks][8];   // 128-B aligned → no false sharing
    alignas(128) uint32_t end[kMaxBlocks][8];
    alignas(128) uint32_t _flag[kMaxBlocks];
};
```

Each rank allocates output buffer via `hipExtMallocWithFlags(...hipDeviceMallocUncached)` (custom_all_reduce.cu:335) — cache-bypass so peer writes are immediately visible after L2 write.

### 20.2 start_sync — peer atomic + spin

```cpp
uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
if (threadIdx.x < ngpus) {
    __scoped_atomic_store_n(&sg.signals[threadIdx.x]->start[blockIdx.x][rank],
                            flag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    while (__scoped_atomic_load_n(&self_sg->start[blockIdx.x][threadIdx.x],
                                  __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) < flag)
        ;
}
__syncthreads();
```

Disassembled as `s_atomic_inc s80, s[40:41], s93` × 8 (one per peer rank) [kernel-analysis/disassembly/ar/kernel.isa:25AC-25E4](_), followed by polled `s_cmp_eq_u32`.

The `__MEMORY_SCOPE_SYSTEM` is critical — peer GPUs must see the write across the XGMI/PCIe fabric, not just in local L2.

### 20.3 Sum-of-squares via v_pk_fma_f32 [kernel-analysis/disassembly/ar/kernel.isa:3B2C-3BA4](_)

```asm
v_pk_fma_f32 v[38:39], v[4:5], v[4:5], v[38:39]
v_pk_fma_f32 v[38:39], v[6:7], v[6:7], v[38:39]
...
```

16 such instructions per token chunk. Two squares per issue (packed f32), accumulating into `v[38:39]`.

### 20.4 Cross-wave reduction via LDS + intra-wave DPP

```asm
ds_write_b32 v229, v36 offset:1024        ; per-wave partial sum
ds_write_b32 v229, v38 offset:2048        ; per-wave sum-of-squares
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b32 v36, v228 offset:1024         ; pick up summed value
...
v_mov_b32_dpp v216, v36 row_ror:2
v_add_f32_e32 v36, v216, v36
v_mov_b32_dpp v217, v52 row_ror:2
v_add_f32_e32 v52, v217, v52
v_mov_b32_dpp v216, v36 row_ror:1
v_add_f32_e32 v36, v216, v36
```

Mixed strategy: LDS at workgroup level (32 banks → padding avoids conflicts), DPP within wave.

### 20.5 RSQRT + scale [kernel-analysis/disassembly/ar/kernel.isa:3DA8-3DC4](_)

```asm
v_mul_f32_e32 v36, v36, v230              ; v36 *= 1/hidden_dim
v_mul_f32_e32 v38, v52, v230
v_add_f32_e64 v38, v38, s63               ; + eps
v_rsq_f32_e32 v38, v38                    ; rsqrt — single Trans op
v_mov_b32_e32 v39, v38                    ; broadcast to pair
v_pk_mul_f32 v[4:5], v[4:5], v[38:39]     ; scale accumulator by rsqrt
```

One `v_rsq_f32` ~4 cycles (Trans). Followed by `v_pk_mul_f32` to apply to two values at once. Then multiply by gamma (norm weights), permute bf16 lanes, write out.

The fusion benefit: while peer ranks are still transferring chunks in the ring, this CU computes the sum-of-squares and rsqrt on locally-arrived data — total kernel latency ≈ max(ring_latency, norm_latency) instead of sum.

---

# Part IV — Toolchain and dispatch

## 21. CK_tile's three waitcnt arch tiers

[arch.hpp:913-959](3rdparty/composable_kernel/include/ck_tile/core/arch/arch.hpp):

```cpp
struct WaitcntLayoutGfx12 {
    static constexpr index_t kVmCntShift   = 8;
    static constexpr index_t kVmCntWidth   = 6;       // [13:8]
    static constexpr index_t kLgkmCntShift = 0;
    static constexpr index_t kLgkmCntWidth = 6;       // [5:0]  (renamed dscnt on gfx12)
    static constexpr index_t kMaxVmCnt   = 63;
    static constexpr index_t kMaxLgkmCnt = 63;
    static constexpr index_t kExpCnt     = 0;
};

struct WaitcntLayoutGfx11 {
    static constexpr index_t kVmCntShift  = 10;
    static constexpr index_t kVmCntWidth  = 6;        // [15:10]
    static constexpr index_t kLgkmCntShift = 4;
    static constexpr index_t kLgkmCntWidth = 6;       // [9:4]
    static constexpr index_t kExpCntShift = 0;
    static constexpr index_t kExpCntWidth = 3;        // [2:0]
};

struct WaitcntLayoutLegacy {                          // CDNA-family (gfx942, gfx950)
    static constexpr index_t kVmCntLowShift  = 0;
    static constexpr index_t kVmCntLowWidth  = 4;     // [3:0]
    static constexpr index_t kVmCntHighShift = 14;
    static constexpr index_t kVmCntHighWidth = 2;     // [15:14]
    static constexpr index_t kLgkmCntShift   = 8;
    static constexpr index_t kLgkmCntWidth   = 4;     // [11:8]
    static constexpr index_t kExpCntShift    = 4;
    static constexpr index_t kExpCntWidth    = 3;     // [6:4]
};
```

CDNA4 uses *Legacy*. gfx12 (future RDNA4?) splits the vmcnt into a single 6-bit range and renames lgkmcnt as dscnt — different encoding entirely.

The compile-time `s_waitcnt_barrier<vmcnt, expcnt, lgkmcnt>()` template selects the layout via target detection:

```cpp
template <index_t vmcnt = kMaxVmCnt, index_t expcnt = kMaxExpCnt, index_t lgkmcnt = kMaxLgkmCnt>
CK_TILE_DEVICE void s_waitcnt_barrier() {
    s_waitcnt<vmcnt, expcnt, lgkmcnt>();
    __builtin_amdgcn_s_barrier();
}

// block_sync_lds_direct_load: vmcnt=0, keep lgkm at max (wait for VMEM to LDS, not LDS reads yet)
template <index_t vmcnt = 0>
CK_TILE_DEVICE void block_sync_lds_direct_load() {
    s_waitcnt_barrier<vmcnt, kMaxExpCnt, kMaxLgkmCnt>();
}

// block_sync_lds: lgkm=0, keep vm at max (LDS reads done before writing fresh data)
template <index_t lgkmcnt = 0>
CK_TILE_DEVICE void block_sync_lds() {
    s_waitcnt_barrier<kMaxVmCnt, kMaxExpCnt, lgkmcnt>();
}
```

This separation — "wait for VMEM only" vs "wait for LDS only" — is the foundation of the SW-pipelined GEMM loop. Without it the kernel would have to drain both classes at every barrier.

## 22. CK_tile async copy → LDS

[amd_buffer_addressing_builtins.hpp](3rdparty/composable_kernel/include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp):

```cpp
template <unsigned num_dwords, bool pre_nop = false>
CK_TILE_DEVICE void async_buffer_load_dwordxn_v(...)
{
    if constexpr (num_dwords == 4) {
        asm volatile("s_nop 4\n"
                     "buffer_load_dwordx4 %1, %2, %3, 0 offen offset:%4 lds"
                     : "=r"(smem)
                     : "v"(voffset), "s"(rsrc), "n"(ioffset)
                     : "memory");
    } else if constexpr (num_dwords == 1) {
        asm volatile("buffer_load_dword %1, %2, 0 offen offset:%3 lds"
                     : "=r"(smem)
                     : "v"(voffset), "s"(rsrc), "n"(ioffset)
                     : "memory");
    }
    // ...
}
```

The leading `s_nop 4` (for 4-dword variant) covers the manual-NOP rule: "SALU writes M0 → LDS-typed op = 1 NOP" with margin. For the 1-dword variant, there's no leading nop — the compiler is trusted to interleave naturally.

The `"=r"` constraint marks `smem` as the *target* of the LDS write — it's not actually written to a VGPR (the `lds` suffix on the buffer_load routes data into LDS via M0). The constraint is there to keep the optimizer from reordering across the asm block.

CDNA3+ has `global_load_lds_dwordx4` (no V# needed, 64-bit address) — used for the rare case of >2GB offset (large KV cache in long-context FMHA).

## 23. CK_tile MFMA wrapper templates

[mfma_gfx9.hpp:35-138](3rdparty/composable_kernel/include/ck_tile/core/arch/mma/mfma/mfma_gfx9.hpp). Each (InputDtype × OutputDtype × M × N × K × Target) is a separate specialization. Example:

```cpp
// f16 × f16 → f32, 16×16×16, GFX9 family
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 16u, 16u, CtrlFlags, CompilerTarget,
                  MmaOpFamily::DENSE, enable_if_target_family_gfx9_t<CompilerTarget>>
{
    using AVecType = ext_vector_t<fp16_t, 4>;   // 4 fp16/lane
    using BVecType = ext_vector_t<fp16_t, 4>;
    using CVecType = ext_vector_t<fp32_t, 4>;   // 4 fp32/lane

    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 4;
    static constexpr index_t kCMLane     = 4;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 4;

    CK_TILE_DEVICE static auto exec(AVecType const& a, BVecType const& b, CVecType const& c) {
        return {__builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c,
                static_cast<int>(CtrlFlags::Cbsz),
                static_cast<int>(CtrlFlags::Abid),
                static_cast<int>(CtrlFlags::Blgp))};
    }
};

// f16 × f16 → f32, 16×16×32, GFX950 ONLY (CDNA4-new doubled-K)
struct amdgcn_mma<fp16_t, fp16_t, fp32_t, 16u, 16u, 32u, ..., 
                  enable_if_target_id_t<CompilerTarget, amdgcn_target_id::GFX950>>
{
    using AVecType = ext_vector_t<fp16_t, 8>;   // 8 fp16/lane
    using BVecType = ext_vector_t<fp16_t, 8>;
    using CVecType = ext_vector_t<fp32_t, 4>;
    static constexpr index_t kABKLane    = 8;
    static constexpr index_t kABKPerLane = 8;
    // ... rest same

    CK_TILE_DEVICE static auto exec(...) {
        return {__builtin_amdgcn_mfma_f32_16x16x32_f16(a, b, c, cbsz, abid, blgp)};
    }
};
```

The `enable_if_target_id_t<..., GFX950>` SFINAE guard means this overload is only compiled when targeting gfx950 — gfx942 builds get a different specialization (only the 16x16x16 variant) and a compile error if you ask for 16x16x32.

The MFMA constants (`kAMLane`, `kABKLane`, etc.) describe the lane-to-element mapping derived in §6.2 — they're the source of truth for the tile_window distribution math.

## 24. AiterAsmKernel loader

[csrc/include/aiter_hip_common.h:174-289](csrc/include/aiter_hip_common.h). Three layers:

**Layer 1 — load the .co binary:**

```cpp
const void* load_hsaco_file(const char* kernel_name, const char* hsaco_path) {
    const char* env = std::getenv("AITER_ASM_DIR");
    std::string arch_name = get_gpu_arch();   // "gfx942" stripped of suffix

    if (env) {
        std::string full_path = std::string(env) + "/" + arch_name + "/" + hsaco_path;
        std::ifstream file(full_path, std::ios::binary | std::ios::ate);
        AITER_CHECK(file.is_open(), "failed to open ", full_path);
        size_t file_size = file.tellg();
        hsaco_data.reset(new char[file_size]);
        file.seekg(0, std::ios::beg);
        file.read(hsaco_data.get(), file_size);
        return hsaco_data.get();
    } else {
#if defined(AITER_EMBEDDED_HSA_MAP)
        std::string fname = "hsa/" + arch_name + "/" + hsaco_path;
        auto it = AITER_EMBEDDED_HSA_MAP.find(fname);
        AITER_CHECK(it != AITER_EMBEDDED_HSA_MAP.end(), "hsaco not found");
        return it->second.data();
#else
        AITER_CHECK(false, "AITER_ASM_DIR not set and no embedded HSA");
#endif
    }
}
```

**Layer 2 — register with HIP:**

```cpp
void init(const char* kernel_name, const void* hsaco) {
    aiter_detail::FatBinaryWrapper fat_bin{};
    fat_bin.magic   = 0x48495046;  // "HIPF"
    fat_bin.version = 1;
    fat_bin.binary  = hsaco;

    module = aiter_detail::__hipRegisterFatBinary(&fat_bin);
    aiter_detail::__hipRegisterFunction(module, this,
                                        kernel_name, kernel_name,
                                        -1, nullptr, nullptr, nullptr, nullptr, nullptr);
}
```

The `FatBinaryWrapper` is a HIP-private structure. The magic `0x48495046` ("HIPF" LE) tells the HIP runtime that this is a HIP fat binary, not a CUDA one.

**Layer 3 — launch:**

```cpp
void launch_kernel(const AiterAsmKernelArgs& kargs) {
    void* config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, kargs.args_ptr,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,    kargs.arg_size_ptr,
        HIP_LAUNCH_PARAM_END
    };
    hipFunction_t fn = nullptr;
    hipGetFuncBySymbol(&fn, reinterpret_cast<void*>(this));
    hipModuleLaunchKernel(fn,
        kargs.gdx, kargs.gdy, kargs.gdz,
        kargs.bdx, kargs.bdy, kargs.bdz,
        0, kargs.stream, nullptr, (void**)&config);
}
```

The `HIP_LAUNCH_PARAM_BUFFER_POINTER` config-buffer pattern avoids the C++ ABI: instead of `hipLaunchKernelGGL(...args...)` (which needs an exact-match function signature), it passes raw arg bytes + size. The `.co` knows how to decode them.

## 25. JIT codegen — CSV to dispatch table

[hsa/codegen.py](hsa/codegen.py) (~250 lines). Walks `hsa/{arch}/{family}/*.csv`, infers struct types per column (numeric → `int`, else `std::string`), and emits a single header:

```cpp
// asm_i8gemm_configs.hpp (generated)
struct i8gemmConfig {
    std::string knl_name;
    std::string co_name;
    std::string arch;
    int tile_m;
    int tile_n;
    int splitK;
    int bpreshuffle;
};

using CFG = std::unordered_map<std::string, i8gemmConfig>;

#define ADD_CFG(tile_m, tile_n, splitK, bpreshuffle, arch, path, knl_name, co_name) \
    {                                                                                \
        arch knl_name, { knl_name, path co_name, arch, tile_m, tile_n, splitK, bpreshuffle } \
    }

static CFG cfg_i8gemm_bf16_perTokenI8 = {
    ADD_CFG(16, 128, 1, 1, "gfx950", "i8gemm/", 
            "_ZN5aiter41I8gemm_bf16_perTokenI8_BpreShuffle_16x128E",
            "I8gemm_bf16_perTokenI8_BpreShuffle_16x128.co"),
    ADD_CFG(32, 128, 1, 1, "gfx950", "i8gemm/",
            "_ZN5aiter41I8gemm_bf16_perTokenI8_BpreShuffle_32x128E",
            "I8gemm_bf16_perTokenI8_BpreShuffle_32x128.co"),
    ...
};
```

The map key is `arch + knl_name` (e.g., `"gfx950_ZN5aiter41I8gemm_..."`) — allows multi-arch deployment with O(1) lookup.

### 25.1 Heuristic kernel selector

[csrc/py_itfs_cu/asm_fmoe.cu:229-306](csrc/py_itfs_cu/asm_fmoe.cu) — pseudocode:

```python
def select_kernel(inter_dim, sub_X_cnt, smf, kernel_name, block_size_M):
    num_cu = get_num_cu()
    best_rounds = inf
    best_kernel = None

    for cfg in cfg_table.values():
        if not cfg.arch.startswith(arch_id): continue
        if cfg.vskip != requested_vskip: continue   # AITER_ENABLE_VSKIP env
        if cfg.smf   != smf:           continue
        if cfg.subGU_m != block_size_M: continue
        if inter_dim % cfg.subGU_n != 0: continue

        tg_num   = (inter_dim // cfg.subGU_n) * sub_X_cnt
        rounds   = (tg_num + num_cu - 1) // num_cu
        empty_cu = rounds * num_cu - tg_num

        is_better = (rounds < best_rounds) or \
                    (rounds == best_rounds and
                     (empty_cu > best_empty_cu or
                      (empty_cu == best_empty_cu and cfg.ps == 1)))

        if is_better:
            best_rounds   = rounds
            best_empty_cu = empty_cu
            best_kernel   = cfg
            if cfg.ps == 1:
                num_persistent_tgs = cfg.tg_num_perCU * num_cu

    return cache.get_or_create(best_kernel.knl_name,
                              lambda: FMoeKernel(best_kernel.knl_name, best_kernel.co_name,
                                                 best_kernel.subGU_n, num_persistent_tgs))
```

Tiebreak order: fewest CU rounds → fewest empty CUs → prefer persistent. The selector is invoked once per shape; result cached in `SynchronizedCache`.

GEMM a8w8 selector at [csrc/py_itfs_cu/asm_gemm_a8w8.cu:57-122](csrc/py_itfs_cu/asm_gemm_a8w8.cu) follows the same pattern but additionally iterates over candidate `splitK` values when not explicitly specified.

## 26. Python @compile_ops flow

[aiter/jit/core.py:1369-1648](aiter/jit/core.py):

```python
@compile_ops(
    "module_gemm_a8w8_asm",
    fc_name="gemm_a8w8_asm",
    ffi_type="ctypes",
)
def _gemm_a8w8_asm(
    XQ: Tensor, WQ: Tensor, x_scale: Tensor, w_scale: Tensor, Out: Tensor,
    kernelName: Optional[str] = None,
    bias: Optional[Tensor] = None,
    bpreshuffle: bool = True,
    splitK: int = -1,
) -> None: ...
```

ctypes type mapping:

| Python | ctypes | C type |
|---|---|---|
| `Tensor` | `POINTER(aiter_tensor_t)` | `aiter_tensor_t*` |
| `Optional[Tensor]` | same; NULL if None | `aiter_tensor_t*` |
| `int` | `c_int64` | `int64_t` |
| `Optional[int]` | `c_int64` (-1 if None) | `int64_t` |
| `str` | `c_char_p` | `char*` (UTF-8) |
| `bool` | `c_int` | `int` (0/1) |
| `float` | `c_float` | `float` |
| (auto-appended) | `c_void_p` | `hipStream_t` |

First call triggers `build_module()` → ninja invokes hipcc with `--offload-arch=gfx942 --offload-arch=gfx950` → produces `.so` cached in `~/.aiter/jit/`. Subsequent calls hit the cache. Module-level state: `aiter/install_mode` file determines develop vs wheel paths.

Error handling: C++ catches exceptions, writes to TLS `g_aiter_last_error`, returns nonzero status. Python `caller()` reads via `err_getter()` and raises `RuntimeError`.

---

# Part V — Optimization opportunities + technique catalog

## 27. Opportunities CDNA4 enables but the repo doesn't yet exploit

1. **V_MFMA_SCALE_F32_*_F8F6F4** [pdf:p51, p64] — block-exponent scaled MFMA folds the post-MFMA `v_mul_f32` scale chain into the MFMA unit itself. The FMOE blockscale kernel currently uses gfx942 patterns (loads scales separately to v36..v43, multiplies post-MFMA via `v_pk_mul_f32 v[608]` issues). Rewriting to gfx950 + `v_mfma_scale_*` would eliminate ~600 scale instructions per kernel.

2. **V_MFMA_I32_16x16x64_I8 / V_MFMA_I32_32x32x32_I8** [pdf:p51] — doubled-K integer MFMAs. The current `gemm_a8w8_m128_*` uses `v_mfma_i32_16x16x32_i8` (older K=32). The newer K=64 halves K-loop trip count at same per-instruction cycles.

3. **V_SMFMAC** sparse MFMAs [pdf:p68-77] — 4:2 structurally sparse A. No SMFMAC kernel found anywhere in `hsa/`. For sparse-weight inference this is 2× peak. Open territory.

4. **DS_READ_B64_TR_B16 + DS_READ_B96_TR_B6** [pdf:p106] — MFMA-transpose load directly into MFMA-input layout. Replaces `ds_read_b128/b64` + manual interleave. Saves ~10-15% in PA/MLA K/V loads.

5. **GFX950 64-bank LDS** — bank-conflict frequency halves; the existing kernels' padding tricks (e.g. 132-dword stride) are less necessary on gfx950.

6. **`v_bitop3_b32`** (gfx950) — three-operand bitwise (AND-NOT, etc.) in one instruction. The `gemv_router.co` already uses this at line 79: `v_bitop3_b32 v10, v1, 63, v1 bitop3:0xc` (= `v1 & ~63`). Other gfx950 kernels could collapse compound boolean expressions.

7. **`v_mfma_f32_*_bf16`** doubled-K [pdf:p51] — `v_mfma_f32_16x16x32_bf16` exists on CDNA4. The bf16 PA kernel (pa_a16w16_b16.co) still uses the older K=16 variant — verify whether the compiler picks the new one when targeting gfx950 explicitly.

## 28. Complete technique catalog with ISA references

| # | Technique | ISA citation | Real-code example |
|---|---|---|---|
| 1 | Selective `vmcnt(N)` for SW pipelining | [pdf:p27-28, p153] | PA: `s_waitcnt vmcnt(8)` at [pa_dis:348]; flatmm: `vmcnt(16)` at [flatmm_dis:372] |
| 2 | Selective `lgkmcnt(N)` for LDS pipeline | [pdf:p27-28] | CK_tile `block_sync_lds<0>()` [arch.hpp:1066] |
| 3 | `buffer_load_dwordx{1,4}_lds` direct VRAM→LDS | [pdf:p91-92, §9.1.9] | FMOE [fmoe_dis:302-318]; CK_tile [amd_buffer_addressing_builtins.hpp:144-288] |
| 4 | `global_load_lds_dwordxN` (CDNA3+) | [pdf:p101-102] | (Used for long-context KV cache when V# 32-bit offset overflows) |
| 5 | AGPR pinning of C accumulator via `+a` constraint | [pdf:p20, p49] | flatmm [flatmm_dis:393-407]; CK_tile [warp_gemm_attribute_mfma_impl.hpp] |
| 6 | `ACC_CD=1` route MFMA result to AGPR | [pdf:p49] | All MFMA-using kernels |
| 7 | Two-stage SW pipeline, PrefetchStages=2 | — | [gemm_pipeline_ag_bg_cr_comp_async.hpp:18, 357-374] |
| 8 | LDS double-buffer (ping-pong) | — | [gemm_pipeline_ag_bg_cr_comp_async.hpp:319-320] |
| 9 | `sched_group_barrier` interleave MFMA/DS/VMEM | — | [gemm_pipeline_ag_bg_cr_comp_async.hpp:205-238] |
| 10 | `iglp_opt(N)` strategy hint | — | (Used in some flatmm builds; check `__builtin_amdgcn_iglp_opt`) |
| 11 | `v_max3_f32` 3-at-a-time reduction | [pdf:p276+] | PA softmax [pa_dis:383-396] |
| 12 | `v_med3_f32` for clamping | [pdf:p276+] | (= `clamp(x, lo, hi) = med3(x, lo, hi)` — saves one instruction) |
| 13 | DPP `row_shr:N` butterfly reduce | [pdf:p564, p599] | AR [ar_dis:1050-1058]; PA [pa_dis:378-410] |
| 14 | DPP `row_ror:N` rotate reduce | same | AR [ar_dis:1050-1058] |
| 15 | DPP `row_bcast:N` broadcast lane | same | PA [pa_dis:202-205] |
| 16 | `ds_swizzle_b32` fixed-pattern shuffle | [pdf:p472] | (Used for FFT-like patterns; not seen in inference kernels) |
| 17 | `ds_bpermute_b32` arbitrary permute | [pdf:p472] | gemv_router (lane reshuffle), small reduction trees |
| 18 | LDS cross-wave reduce | [pdf:p103-105] | AR [ar_dis:879-901]; PA softmax cross-wave max |
| 19 | `v_pk_fma_f32` packed two-f32 FMA | [pdf:p276+] | AR sum-of-squares [ar_dis:3B2C-3BA4] |
| 20 | `v_pk_mul_f32` packed two-f32 multiply | same | AR final scale, SMF FMOE gate×up |
| 21 | `v_pk_add_f32` packed two-f32 add | same | AR allreduce accumulator |
| 22 | `v_pk_fma_f16` / `v_pk_add_f16` | same | F16 normalization paths |
| 23 | `v_cvt_pk_fp8_f32` packed F32→FP8 | [pdf:p47, §6.7.1] | Quick-allreduce CodecFP8 |
| 24 | `v_cvt_pk_f32_fp8` packed FP8→F32 | same | Same; symmetric decode |
| 25 | `v_cvt_scalef32_pk_fp8_f32` block-scaled | [pdf:p65-67] | (CDNA4-new; not yet used in repo) |
| 26 | `v_perm_b32` for bf16 packing | [pdf:p276+] | PA bf16 epilogue [kernel-analysis/disassembly/pa2/kernel.isa:2849] |
| 27 | Nibble-mask fp8 dequant via `v_cndmask` | (not a single op) | PA per-token [pa_dis:365-378] with `s[44:45]=0xF0F0F0F0` |
| 28 | Hardware fp8 decode inside MFMA | [pdf:p65] | flatmm, FMOE — no explicit cvt needed |
| 29 | E8M0 block scales in V_MFMA_SCALE | [pdf:p64-65] | (CDNA4-new; opportunity) |
| 30 | `v_rsq_f32` reciprocal-square-root (Trans) | [pdf:p29] | RMSNorm [ar_dis:3DB8] |
| 31 | `v_exp_f32` (actually exp2) + scale | [pdf:p29] | Softmax (PA, FMHA, FMOE SiLU) |
| 32 | Pre-scale by log2(e) → single `v_exp_f32` for natural exp | — | CK_tile `CK_TILE_FMHA_FWD_FAST_EXP2` |
| 33 | Per-block fp8 scale broadcast | (kernel design) | PA per-block [pa2_dis:278-284] |
| 34 | TopK via K-pass `v_max3_f32` + `v_writelane` | — | TopK kernel |
| 35 | `s_set_gpr_idx_on` indirect GPR write | [pdf:p149] | TopK selection-sort scatter |
| 36 | Buffer OOB returns 0 (bounds-check trick) | [pdf:p83] | All kernels for tile-edge masking |
| 37 | V# `swizzle_enable` for bank distribution | [pdf:p90] | Structured AoS layouts |
| 38 | V# `num_records` for free bounds check | [pdf:p90] | All kernels |
| 39 | Memory scope SC[1:0] system for peer atomics | [pdf:p92-93] | custom_all_reduce `__MEMORY_SCOPE_SYSTEM` |
| 40 | NT bit for cache-bypass stores | [pdf:p92-93] | Quick-allreduce `__builtin_nontemporal_store` |
| 41 | `hipExtMallocWithFlags(hipDeviceMallocUncached)` | (HIP API) | AR for peer buffer [custom_all_reduce.cu:335] |
| 42 | `Signal alignas(128)` no-false-share | (struct layout) | [custom_all_reduce_hip.cuh:36-42] |
| 43 | `__scoped_atomic_store_n` system-scope | (Clang builtin) | start_sync / end_sync |
| 44 | `__builtin_nontemporal_store/load` (NT=1) | [pdf:p93] | Codec data path |
| 45 | `global_atomic_pk_add_bf16` packed atomic | [pdf:p101-102] | FMOE scatter output |
| 46 | `global_atomic_add_f32` with relaxed ordering | [pdf:p101-102] | SplitK GEMM scale + atomic |
| 47 | Persistent grid (deterministic per-WG tile) | (kernel design) | flatmm [flatmm_dis:32-34, 1095] |
| 48 | Persistent grid with `s_atomic_inc` counter | (kernel design) | Some FMHA persistent variants |
| 49 | `s_setprio` priority lowering during memory waits | [pdf:p154] | Not commonly seen in this repo |
| 50 | `__builtin_amdgcn_readfirstlane` SGPR broadcast | [pdf:p277+] | All kernels for workgroup-id extract |
| 51 | `v_mul_u32_u24` fast 24-bit multiply via FP pipe | [pdf:p185+] | PA address compute |
| 52 | `v_mad_u32_u24` single-inst 24-bit multiply-add | [pdf:p185+] | Address computation |
| 53 | DPP `row_newbcast:N` quad-broadcast | [pdf:p564-565] | PA decode address pattern |
| 54 | Inline `s_nop 3` after MFMA to cover dep rule | [pdf:p74] | CK_tile MFMA wrappers |
| 55 | Inline `s_nop 4` before LDS-typed buffer_load | [pdf:p29] | CK_tile async_buffer_load |
| 56 | `__builtin_amdgcn_s_setprio` | [pdf:p154] | (Hand-tuned kernels can prioritize critical waves) |
| 57 | Hardware ID read via `s_getreg_b32 hwreg(HW_REG_HW_ID, ...)` | [pdf:p155] | POD attention CU-ticket scheme |
| 58 | XCC ID for cross-die affinity | [pdf:p24] | POD persistent dispatch |
| 59 | `v_bitop3_b32` 3-operand bitwise (gfx950) | (new) | gemv_router |
| 60 | `ds_read_b96_tr_b6` MFMA-transpose load (gfx950) | [pdf:p106] | (Open opportunity) |
| 61 | `v_smfmac_*` sparse MFMA | [pdf:p68-77] | (Open opportunity) |
| 62 | Compile-time `__HIP_DEVICE_COMPILE__` guard | (OPUS skill) | [.claude/skills/opus-kernel-best-practice/SKILL.md] — 50% compile time saving |
| 63 | `static_for` → `for` to reduce template instantiation | (OPUS skill) | 30-60% frontend savings |
| 64 | `__builtin_convertvector` for vector cast | (OPUS skill) | 5-10% frontend savings |
| 65 | Hipcc `-mllvm --amdgpu-kernarg-preload-count=16` | (build flag) | Preload first 16 kernarg slots into SGPR before kernel entry |
| 66 | `--offload-arch=gfx942 --offload-arch=gfx950` multi-arch fatbin | (build flag) | One `.so` runs on both |
| 67 | Embedded HSA mode via `AITER_EMBEDDED_HSA_MAP` | (build flag) | Single self-contained `.so` (no external `.co` files) |
| 68 | ctypes (vs pybind11) FFI | (binding choice) | ~10× lower call overhead |
| 69 | `SynchronizedCache` for kernel lookup memoization | (host) | One-time .co load per shape |
| 70 | Heuristic kernel selector minimizing CU rounds | (host) | All asm_*.cu dispatch files |
| 71 | Causal mask via post-MFMA `v_cndmask` to -inf | (kernel design) | PA per-block [pa2_dis:1638-1670] |
| 72 | Online softmax with `m`, `l` running stats | (algorithm) | FMHA, MLA |
| 73 | Q-in-registers (loaded once, sliced per K-tile) — **requires multi-wave WG (§14) to be useful** | (kernel design) | [block_fmha_pipeline_qr_ks_vs.hpp:272, 384, 526] |
| 74 | KV-LoRA inline decompression | (algorithm) | MLA [kernel-analysis/disassembly/mla/kernel.isa (⚠️ not archived — regenerate; see kernel-analysis/README.md):1239-1250] |
| 75 | RoPE inline rotation via `v_fma_f32` chain | (algorithm) | MLA same lines |
| 76 | SMF dual-expert LDS interleaving | (kernel design) | SMF FMOE [smf_dis:2654-2661] |
| 77 | Multix multi-iteration with wider LDS reads | (kernel design) | multix FMOE |
| 78 | "novs" disable vector-skip — explicit lane ops | (compile flag) | blockscale_novs FMOE |
| 79 | Fused allreduce + RMSNorm hardcoded N=8192 | (kernel design) | AR kernel |
| 80 | Cross-wave reduction overlapping ring-bubble | (kernel design) | AR kernel |
| 81 | Split-K with atomic accumulation | (kernel design) | gemm_a8w8_m128_splitK |
| 82 | Scale applied per split BEFORE atomic (correctness) | (kernel design) | splitK GEMM |
| 83 | GEMV (no MFMA) for M ≤ 8 | (kernel design) | gemv_router — 35.6× hipBLASLt |
| 84 | Triton `waves_per_eu` autotuning | (Triton) | bwd.py configs |
| 85 | Triton `matrix_instr_nonkdim` forcing MFMA size | (Triton) | All Triton GEMM/attention |
| 86 | Triton `kpack=2` for doubled-K MFMA | (Triton) | Triton MoE |
| 87 | Triton `num_stages=2` async copy pipeline | (Triton) | Triton fwd_decode |
| 88 | Triton `tl.atomic_add(..., sem="relaxed")` | (Triton) | gmm bias-grad reduction |
| 89 | Triton `cache_modifier=".cg"` L1-cache hint | (Triton) | lean_atten_paged |
| 90 | Gluon `AMDMFMALayout(instr_shape=[32,32,16])` | (Gluon) | moe_op_gemm_int8_smoothquant |
| 91 | Gluon `BlockedLayout` explicit thread distribution | (Gluon) | Same |
| 92 | Gluon `gl.amd.cdna3.buffer_load` direct intrinsic | (Gluon) | Same |
| 93 | FlyDSL `rocdl.exp2`, `rocdl.rcp` direct lowering | (FlyDSL) | silu_and_mul_fq |
| 94 | FlyDSL E8M0 quantization scale derivation | (FlyDSL) | silu_and_mul_fq:155-285 |
| 95 | FlyDSL block-validity gate (skip OOB blocks) | (FlyDSL) | moe_gemm_2stage:344-353 |
| 96 | POD attention HW_ID-based CU ticket | (Triton kernel) | pod_attention.py:49-62 |
| **97** | **Multi-wave WG (typ. 4 waves = 256 threads), full 4-SIMD CU coverage** | **(foundational, §14)** | **Every inference kernel: PA, flatmm, FMOE, FMHA, TopK, MLA — see §1 resource table** |
| **98** | **Wave specialization: output-partition** (each wave produces a different M×N quadrant of the same output tile, shares K/V tiles in LDS) | **(§14.2)** | **CK_tile `block_gemm_areg_bsmem_creg_v1`, FMHA `qr_ks_vs`, flatmm 1×4×1 wave grid** |
| **99** | **Wave specialization: expert-partition (SMF)** — pairs of waves handle different experts with interleaved LDS slots | **(§14.2, §17.1)** | **FMOE `smf_subGU_320` — [kernel-analysis/disassembly/smf/kernel.isa:2654-2661]; offsets {2048,4224,6400,8576} vs {10752,12928,15104,17280}** |
| **100** | **Wave specialization: cross-wave LDS reduction** — all N waves accumulate into LDS, final reduce via LDS+DPP butterfly | **(§14.2, §20.4)** | **AllReduce+RMSNorm 16-wave WG [kernel-analysis/disassembly/ar/kernel.isa:879-901]; TopK 4-wave butterfly** |
| **101** | **Wave specialization: TG-split** — workgroup divided into two thread-groups handling disjoint Q-tile ranges | **(§14.2, §15)** | **PA `2tg` per-token fp8 decode** |
| **102** | **`v_readfirstlane_b32 sX, vY` to promote wave-uniform values to SGPR for SALU branches** | **(pdf:p278+)** | **Every multi-wave kernel — wave_id extract in prologue** |

That's 102 distinct techniques drawn from the deep-read of 8 disassembled binaries + the full CK_tile source + the Triton/Gluon/FlyDSL DSLs + the CDNA4 ISA spec.

## 29. Closing — how to use this document

1. **For reading disassembly**: jump to §6 (MFMA), §8 (LDS/DS), §9 (DPP), §10 (VOP3P) with the ISA spec PDF open at the cited pages.
2. **For writing/modifying a `.co`**: §3 (manual NOPs), §6.8 (MFMA dep rules), §14 (multi-wave WG patterns), §21-23 (CK_tile primitives), §24 (loader), §25 (codegen).
3. **For finding optimization opportunities on gfx950**: §27 (V_MFMA_SCALE, doubled-K, V_SMFMAC, DS_READ_TR*).
4. **For Triton/Gluon kernels on AMD**: §28 entries 84-96 with the autotune config patterns.

Cross-references:
- High-level structure: [ASM_KERNEL_KNOWLEDGE.md](ASM_KERNEL_KNOWLEDGE.md)
- v1 playbook (superseded by this document): [ASM_PERF_PLAYBOOK.md](ASM_PERF_PLAYBOOK.md)
- ISA spec: [amd-instinct-cdna4-instruction-set-architecture.pdf](amd-instinct-cdna4-instruction-set-architecture.pdf)
- Round-trip toolchain: [docs/isa_kernel_optimization.md](docs/isa_kernel_optimization.md)
- OPUS compile-time skill: [.claude/skills/opus-kernel-best-practice/SKILL.md](.claude/skills/opus-kernel-best-practice/SKILL.md)
