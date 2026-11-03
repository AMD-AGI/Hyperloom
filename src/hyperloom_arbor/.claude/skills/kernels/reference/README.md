# reference/: a curated copy of KernelForge

**Source:** KernelForge (this repository carries it as `src/kernelforge`), MIT licence
(`LICENSE-KernelForge`, copied unchanged). Copied on 2026-10-20 from the runner's read-only copy at
`src/kernelforge (this repository)` (the same tree as `src/kernelforge (this repository)` on
Cluster B).
- **Commit:** `cd9c585` (2026-08-17), from that copy's `RUNNER_README.txt`. The copy is not a git
  checkout, so `git log` cannot confirm it.
- KernelForge's AITER pages cite AITER `b467ce342`; its FlyDSL pages, FlyDSL `2812676` (2026-05-18).
  Check every claim against the source installed in the image (`../probe-image.sh` prints the paths).

The layout under this directory is KernelForge's own. The files are copied unchanged: never edit
them. Corrections go in `STALE.md` and in `../SKILL.md`.

## What was copied

| Path | Files | Bytes | Why |
| --- | --- | --- | --- |
| `knowledge_base/flydsl/` | all 13 | 108,973 | FlyDSL: overview, API, dialect, pipeline, patterns, GEMM and LDS optimization, kernel inventory, env vars, pitfalls |
| `knowledge_base/triton/` | all 7 | 18,663 | Triton on AMD: pitfalls, autotuning, `tl.dot_scaled` on gfx950 |
| `knowledge_base/ck/` | all 4 | 220,635 | CK: tile model, pitfalls, the CK playbook |
| `knowledge_base/hip/` | all 8 | 169,556 | HIP: patterns, MFMA builtins and registers, build, HipKittens, an MXFP8 grouped GEMM, pitfalls |
| `knowledge_base/aiter/` | 5 of its 11 pages: `operator_catalog.md`, `backend_dispatch.md`, `integration_patterns.md`, `jit_and_build.md`, `pitfalls.md` | 42,101 | AITER's ops, dispatch, build and traps |
| `knowledge_base/skills/` | 2 of 36: `mfma_16x16x128_f8f6f4_fp4_operand_layout.json`, `mfma_16x16x128_f8f6f4_operand_layout.json` | 9,963 | the scaled-MFMA operand layouts the SKILL.md cites for block-scaled FP4/FP8 on gfx950 |
| `knowledge_base/README.md` | 1 | 1,687 | the citation prefixes the CK and HIP playbooks use |
| `examples/triton2flydsl-mxfp8-grouped-gemm/` | all 7 | 27,798 | a Triton → FlyDSL port of SGLang's MXFP8 MoE grouped GEMM (MiniMax-M3, MI355X) |
| `examples/triton2flydsl-softmax-flydsl-rewrite/` | all 6 | 28,194 | the smallest Triton → FlyDSL port |
| `examples/flydsl_gemma_rmsnorm/` | all 5 | 24,995 | a FlyDSL kernel with driver, oracle and graph harness |
| `examples/flydsl-softmax-forge-loop/` | all 5 | 34,096 | the same for softmax, with the capture-validity guard |
| `local_knowledge/framework/aiter/overall/tuning_db.md`, `authoring_delegation.md` | 2 | 10,355 | tune AITER's table or write a kernel; the tuner recipe |
| `local_knowledge/framework/aiter/skills/optimize/aiter_levers/flydsl_path.md` | 1 | 5,350 | how AITER selects a FlyDSL kernel, and its fallback |
| `local_knowledge/framework/aiter/operators/fused_moe_grouped_gemm/tuning.md` | 1 | 8,650 | the fused-MoE tuner and `AITER_CONFIG_FMOE` |
| `local_knowledge/languages/flydsl/skills/optimize/flydsl_levers/kernel_families.md` | 1 | 4,385 | the FlyDSL families AITER ships |
| `LICENSE-KernelForge` | 1 | 1,476 | MIT licence |

Total: 69 files, 716,877 bytes (this README and `STALE.md` are not counted).

The plan listed `knowledge_base/{flydsl,triton,ck,hip}`, five AITER pages and four examples. The
`local_knowledge/` pages, the two operand-layout files and `knowledge_base/README.md` were added
because the SKILL.md cites them; they total 40,390 bytes. Their relative links to pages that were not
copied (`fmoe.md`, `dispatch_and_rebind.md`, `languages/flydsl/...`) do not resolve here.

## What was not copied, and why

- **`knowledge_base/shared/gpu_arch_gfx950.md`:** stale. It says gfx950 has 304 CUs, 32 LDS banks
  and HBM "similar to MI300X's 5.3 TB/s"; it has 256, 64 and 8.0 TB/s (the hardware skill).
- **Nothing was skipped for size.** No copied directory holds a binary, trace or log; the largest
  file is `knowledge_base/ck/ck_playbook.md` (211,879 bytes), a document, kept.
- **Outside the plan's list:** the rest of `knowledge_base/aiter/` (the asm playbooks and notes,
  about 160 KB, and two subfolders), `knowledge_base/{hipblaslt,sglang,vllm,shared,tuning_db}/`, the
  other 34 `knowledge_base/skills/` files, the other examples (`aiter-allreduce-forge-loop`, `hip_gemma_fused_add_rmsnorm`,
  `mori_ep_dispatch_combine`, `triton-softmax-forge-loop`, `triton_mixtral_dynamic_quant`), the
  rest of `local_knowledge/` (the per-language and per-operator folders), `tasks/`, `src/`. They stay
  in the runner's copy.

## Known-wrong numbers

`STALE.md` lists every line in this copy that disagrees with the hardware skill on gfx950 or gfx942.
