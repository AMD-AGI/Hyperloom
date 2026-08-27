# CDNA4 ISA Reference (gfx950) — raw spec extracts

Pointer file. Chapter-level **text extracts** of the AMD Instinct CDNA4 ISA
spec are vendored into this knowledge base — `Read`/`Grep` them directly when
you need exact instruction semantics, encoding, or wait-state rules. Do not
guess ISA behavior.

## Where it lives (in-repo, self-contained)

- ISA text extracts: `local_knowledge/languages/asm/API_docs/cdna4-isa/` (30 files, ~588 KB)
- Disassembly dumps of real aiter `.co` kernels: `knowledge_base/aiter/asm-disassembly/`
  (ar, flatmm, fmoe, multix, pa, pa2, smf — `kernel.isa` + `note.txt` per family)

The full 608-page PDF is not vendored; if a detail is missing from the extracts
it lives at `amd-instinct-cdna4-instruction-set-architecture.pdf`
on the original host.

## Extract index (`local_knowledge/languages/asm/API_docs/cdna4-isa/`)

| File | Topic |
|---|---|
| `Ch1_intro.txt`, `Ch2_program_org.txt`, `Ch3_kernel_state.txt` | Wavefront/program model, kernel state regs |
| `Ch4_program_flow_wait_states.txt` | Branching + **wait-state hazards** (s_waitcnt, NOPs) |
| `Ch5_SALU.txt`, `Ch6_VALU_full.txt` | Scalar / vector ALU instruction set |
| `Ch7_MFMA_complete.txt`, `Ch7_MFMA_layout_p55-62.txt` | MFMA ops + per-lane operand/accumulator layout |
| `Ch7_sparse_MFMA_p68-77.txt` | Structured-sparse MFMA |
| `Ch7_block_scaled_p63-68.txt`, `Ch7_FP8_BF8_formats_p65-68.txt`, `Ch7_FP_handling_p68-70.txt` | Block-scaled + FP8/BF8 numeric formats |
| `Ch8_SMEM.txt`, `Ch11_LDS_full.txt`, `Sec12_12_LDS_full.txt`, `Sec13_4_LDS_format.txt` | Scalar memory + LDS |
| `Ch9_full_p82-97.txt`, `Ch10_flat_full.txt`, `Sec12_13_MUBUF.txt`, `Sec12_15_FLAT.txt` | Global/FLAT/MUBUF memory ops |
| `Sec12_10_VOP3P_full_p276.txt`, `Sec12_7_VOP2_sample.txt`, `Sec13_3_VOP_formats.txt` | VOP3P/VOP2 packed-math encodings |
| `Sec12_16_DPP_SDWA.txt`, `Sec12_16_DPP_limitations.txt`, `Sec13_3_9_DPP_encoding.txt` | DPP cross-lane ops + limitations |
| `Sec12_5_SOPP_full.txt`, `Sec12_6_SMEM_full.txt` | SOPP / SMEM encodings |

## How to use

The ASM, CK, and HipKittens playbooks cite this spec with `[pdf:pN]`. When a
playbook technique hinges on an exact ISA detail, open the matching extract
above rather than guessing.
