---
name: kernels
description: Before writing, porting or tuning a GPU kernel for LLM serving on AMD Instinct (MI350X/MI355X gfx950, MI300X gfx942) — AITER's tuned tables and tuner first; then which language for which kernel (Triton, FlyDSL, CK, HIP/assembly) and why; what the container image actually ships (probe-image.sh); and proving correctness with a negative control and speed graph-captured at the served shapes (kbench.py), with KernelForge's FlyDSL, Triton, CK, HIP and AITER references. Use when a kernel plan must name its language, when choosing Triton vs FlyDSL vs CK vs HIP, or before stages S1–S3 of a rewrite.
---

# Kernels: whether to write one, in which language, and how to prove it

A campaign that rewrites a kernel has to answer three questions before its first line of code: does
the image already have a faster kernel for this op, which language fits the kernel, and does the
image ship that language. Then it has to prove the kernel correct and fast at the shapes the server
runs. The dossier's kernel plan (`.claude/skills/dossier/kernel-plan.md`) says what a rewrite must
show at each stage; this skill says how.

| File | What it is |
| --- | --- |
| `probe-image.sh` | prints the container's ROCm, hipcc, torch, Triton, FlyDSL, AITER (with its asm, FlyDSL, Triton and CK kernels and tuned tables) and CK versions, and whether each imports (§3) |
| `kbench.py` | a reference op against a candidate at given shapes: correctness with negative controls, then graph-captured timing, as JSON (§5) |
| `reference/` | a curated copy of KernelForge (AMD-AGI, MIT licence: `reference/LICENSE-KernelForge`), taken 2026-10-20 from the runner's copy of commit `cd9c585`; what was copied and skipped is in `reference/README.md`, its known-wrong numbers in `reference/STALE.md` |

Hardware facts (LDS, MFMA, bandwidth) are the `hardware` skill's; this skill points to them. Knobs
and the images that carry each kernel family are the `serving-playbook` skill's.

**How sources are cited here:**
- Short paths such as `flydsl/overview.md` or `aiter/pitfalls.md` are under `reference/knowledge_base/`.
- `tuning_db.md`, `authoring_delegation.md`, `flydsl_path.md`, `kernel_families.md` and the fused-MoE
  `tuning.md` are under `reference/local_knowledge/`; §7 gives their full paths.
- `#NNN` is a record id of the serving playbook's InferenceX study (not carried in this copy);
  `amd-knobs`, `big-directions` (D1–D22) and `image-manifest` name that skill's reference pages, which are not
  carried in this copy.

## 1. Before writing anything: the order of resort

Write a kernel last. Each step before the last is cheaper, reversible and already in the served path.

1. **Find what runs now and what else could.** AITER carries several kernel families per op
   (FlyDSL, CK, asm, Triton, hipBLASLt) and dispatches by shape; for a new model's shapes the
   default heuristic often picks a slower one, and the switch is an env var or a table
   (`serving-playbook/big-directions.md` D19). The dossier's part 8 (`dossier/kernels.md` §2–4)
   finds the provider, the dispatcher's other branches and why each is unused.
   - `AITER_LOG_TUNED_CONFIG=1` logs which tuned row each call took (`aiter/backend_dispatch.md`).
   - A kernel name in a trace proves the dispatch; a flag being set does not.
2. **Select an existing family.** `AITER_FLYDSL_FORCE=1` (#863), `--moe-backend flydsl` (#867),
   `AITER_USE_FLYDSL_MOE_SORTING=1` (#0, #1255) and the attention and MoE backend flags
   (`serving-playbook/amd-knobs.md` §1, §4) are increments or bundles: one iteration each, with
   proof that the arm ran the family it claims.
3. **Tune AITER's table for your shapes.** Capture the live shapes, race the library's kernels per
   shape, deploy the winners as an `AITER_CONFIG_*` overlay, prove engagement, then A/B.
   - GEMM: `AITER_CONFIG_GEMM_BF16`, tuned by `csrc/gemm_a16w16/gemm_a16w16_tune.py`, which races
     asm, opus, FlyDSL, Triton, skinny and torch kernels (`tuning_db.md`).
   - Fused MoE: `AITER_CONFIG_FMOE`, tuned by `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`
     (the fused-MoE `tuning.md`).
   - **Evidence:** a tuned fMoE CSV routed DSV4 FP4's eligible token counts to FlyDSL FP4 MoE
     kernels on MI355X (the overlay was deleted once a nightly shipped equivalent tables).
     GEMM table tuning alone can move end-to-end throughput (`tuning_db.md`).
   - **Traps:** the lookup key includes `gfx`, `cu_num` (256 on MI355X), the padded M bucket, bias
     and the full quantization signature; one wrong field and the table silently does nothing.
     Prove engagement first (`grep -c 'is tuned on cu_num' server.log` > 0, `tuning_db.md`). The CSV
     is build-locked: regenerate it on every image change.
   - **Size the sweep to your turn.** A tuner sweep over many shapes can outlast an agent turn; a
     turn cut off by its cap is not measured, and its uncommitted work is discarded (Cluster C,
     2026-10-29: a fused-MoE sweep used a whole 30-minute turn, half the run). Tune the shapes the
     served load runs, a batch at a time, writing each batch's rows to the CSV in the workspace,
     so the next iteration continues. Inside a host the prompt says how long the turn is.
   - This is a small rewrite in kernel-plan terms: tuner run, correctness, A/B (D19).
4. **Only then write a kernel**: when no library kernel covers the shape or the fusion, or tuning
   has plateaued and the profile shows a ceiling worth the work (`authoring_delegation.md`). That is
   a rewrite, with a kernel plan.

**Where the result lands** (kernel-plan §7):

| What you made | Where it goes first | Where it ends | Source |
| --- | --- | --- | --- |
| Tuned rows for shapes AITER's tables miss | an overlay CSV named by `AITER_CONFIG_*` in the server's environment, kept with the campaign's tree | AITER's `aiter/configs/` upstream | `tuning_db.md`; amd-knobs §4 (#560, #652) |
| A new kernel for an op AITER owns (GEMM, MoE stage, attention, norm and quant) | the served tree, behind a `VLLM_*` switch that defaults off, written to AITER's interfaces: its op signature, weight preshuffle, scale layout and, for FlyDSL, its name-encoded configs | AITER (`aiter/ops/flydsl/`, `aiter/ops/triton/`, `csrc/`), where its tuner can race it | kernel-plan §7; `authoring_delegation.md`; `flydsl_path.md` |
| A fusion across ops the framework owns | the served tree (model or layer code), behind a switch | the framework, by PR | kernel-plan §7 |

## 2. Which tool for which kernel

| Tool | Reach for it when | Why | What AITER already ships in it | Evidence, limits |
| --- | --- | --- | --- | --- |
| **Triton** | Elementwise and reduction kernels (norms, activation with quant, RoPE, softmax, sorting helpers); fusions across ops Inductor cannot see through; S1 references and quick prototypes; MFMA work where its best config is good enough | Fast iteration, Python-level autotuning; a dependency of vLLM and SGLang, so in their images (`probe-image.sh` shows it). `tl.dot` maps to MFMA (`triton/pitfalls.md`). Most decode kernels are HBM-bound, where layout control matters less than bytes moved (the hardware skill's `reference/INDEX.md`, golden rules) | Many ops under `aiter/ops/triton/` (attention, MoE, RoPE, GEMM; `triton/auto_known_issues.md`); some Gluon kernels, Triton's lower-level dialect with explicit layouts (big-directions D18, #1050) | AITER's asm sparse paged attention won on one image and lost to Triton the next (amd-knobs §0 rule 4, #926 → #941). **Limit:** layouts and LDS are implicit. On gfx950 `tl.dot_scaled` emits native `v_mfma_scale_f32_32x32x64_f8f6f4` but runs slower than plain `tl.dot` at the same MFMA count (scale traffic through LDS); no config fixed it (`triton/tl_dot_scaled_gfx950.md`, one MXFP8 grouped GEMM) |
| **FlyDSL** | MFMA-heavy tiles where layout decides speed: GEMM (split-K, small-M decode), fused MoE stages, block-scaled FP4/FP8 on gfx950, attention where Triton leaves the matrix cores idle; extending a FlyDSL family AITER already ships | Explicit layout algebra (CuTe-style), explicit LDS allocation and swizzles, explicit MFMA atoms including `mfma_scale_f32_16x16x128_f8f6f4`, async global→LDS copies; Python with JIT, no C++ build. "The right tool when you need to hit close to peak MFMA utilization or beat the CUTLASS/CK pattern catalog on a specific shape"; not for fast iteration (`flydsl/overview.md`) | Split-K HGEMM, small-M HGEMM for M = 1–16 (bf16), preshuffle A8 GEMM, two-stage fused MoE (mixed W4A16, A4W4), gated-delta-rule decode, `silu_and_mul_fq` (`kernel_families.md`). FlyDSL's own repo adds blockscale GEMM, paged-attention decode and flash attention (both "WIP perf tuning"), MLA decode, norms, RoPE with cache write (`flydsl/kernels_inventory.md`) | Recipes chose AITER's FlyDSL kernels on MI355X: #560, #867, #863, #1214 (FP8 stage 2, NUM), and #1255, where FlyDSL MoE sorting was one of three changes that together raised P90 interactivity. **No record separates FlyDSL's own gain on gfx950.** The one sized FlyDSL rewrite is vendor-reported on MI300X: Kimi-K2.5's fused MoE, +162% throughput at concurrency 40 (`kernel_families.md`; [AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html)). The `triton2flydsl-*` examples show the porting method, not a result. **Limits:** verbose; a separate package, so if the image lacks it every AITER FlyDSL path is inert (`flydsl_path.md`); the repo's MX-scaled MoE targets gfx1250 (`kernels_inventory.md`) |
| **CK** (Composable Kernel) | Templated GEMM, attention (FMHA), fused MoE and norms, mostly by instantiating or tuning instances through AITER's codegen (`gen_instances.py`, `--gen-tune`) rather than writing new templates | AITER's default backend for GEMM, prefill attention, normalization and RoPE (`aiter/backend_dispatch.md`); thousands of instances, a mature pipeline catalog (`ck/ck_playbook.md` §12) | `gemm_a8w8_ck`, `ck_moe_stage1/2`, `mha_fwd`, `*_cktile` (`aiter/operator_catalog.md`) | **Limits:** C++ templates, long builds, stale-`.so` and header-tracking traps (`ck/pitfalls.md`). Its LDS swizzles assume 32 banks and no kernel uses gfx950's 160 KB LDS: headroom, but work (`ck_playbook.md` §16.12). Fused-MoE tile gaps on real models (image-manifest A4: N = 384 at TP ≥ 4, #517; "Unsupported kernel config", #999) |
| **HIP / assembly** | The last mile: register pinning, exact instruction scheduling, instructions the compilers do not emit; kernels outside any DSL (custom collectives) | Full control. AITER's asm kernels (`hsa/<gfx>/<op>/*.co`: paged attention, MLA, fused MoE, FP4 GEMM) are its lowest-latency paths for the shapes they cover (`aiter/backend_dispatch.md`) | The asm families above; HIP ops in `csrc/`; HipKittens as a C++ tile library (`hip/hipkittens_guide.md`) | **Limits:** per-arch hand maintenance; silent traps (an AGPR inline-asm constraint drops register 0: 21 dB SNR; the 256-VGPR occupancy cliff; buffer loads clamp out-of-bounds silently: `hip/pitfalls.md`). Hand asm does not stay ahead by itself: #926 → #941 above |

Gluon and Triton are both Triton; CK and FlyDSL overlap on GEMM and MoE. Where two fit, S1–S3
decide, at the served shapes.

### Signals that the kernel wants FlyDSL or CK rather than Triton

- **It is MFMA-bound at the served shapes** (the hardware skill's roofline says compute-bound), and
  Triton's best config sits well under the MFMA peak with LDS stalls or bank conflicts in the
  counters (`rocprofv3`: the `profiler` skill, section 5). Triton gives no direct control over LDS layout.
- **Block-scaled FP4/FP8 where the scales are the bottleneck.** `tl.dot_scaled`'s cost is scale
  redistribution through LDS (+24 `ds_read_u8` and +14 waits per MFMA cluster); its fix path is
  upstream Triton or a kernel that bypasses it (`tl_dot_scaled_gfx950.md`). FlyDSL exposes the
  scaled MFMA directly (`flydsl/overview.md`, atom catalog).
- **AITER already has a FlyDSL family for the op.** Extend it: its tuner already races FlyDSL
  (`--libtype flydsl`, `tuning_db.md`), and the result can land upstream.
- **The design needs explicit placement:** ping-pong LDS buffers, XOR swizzles, a preshuffled weight
  layout that matches AITER's, persistent grids (`flydsl/gemm_optimization.md`, `kernels_inventory.md`).

Signals for Triton: the kernel is bandwidth-bound; the work is a fusion of elementwise steps; the
op's best existing kernel is already Triton; or this is the S1 reference.

### gfx950 facts that change a kernel's design

The facts are the hardware skill's (`.claude/skills/hardware/SKILL.md`, quick facts); read the page
named before relying on one. In the last column, `hw:` paths are under
`.claude/skills/hardware/reference/`, the others under this skill's `reference/knowledge_base/`.

| Fact on gfx950 (gfx942 in brackets) | What it changes | Read |
| --- | --- | --- |
| LDS 160 KiB per CU, 64 banks (64 KiB, 32) | Re-derive swizzles and padding (the XOR mask doubles); bigger tiles or a third pipeline stage fit | hw: `cdna4_mi350/memory.md`, `shared/memory_model_lds_bank.md`; `flydsl/lds_optimization.md` |
| Direct global→LDS up to 128 bits per lane, `global_load_lds_dwordx4` (32 bits) | Async copies without staging through VGPRs; guard HIP code by arch | hw: `cdna4_mi350/memory.md`; `hip/pitfalls.md`; FlyDSL's `cluster_load_async_to_lds_b128` (`flydsl/overview.md`) |
| Scaled MFMA `v_mfma_scale_f32_*_f8f6f4`, an E8M0 scale per 32 elements (none) | MXFP4, MXFP6 and MXFP8 run natively; the instruction fixes the operand and scale layouts | hw: `cdna4_mi350/matrix_core_blockscale.md`, `cdna4_mi350/fp4_fp6_microscaling.md`; `skills/mfma_16x16x128_f8f6f4*_operand_layout.json` |
| FP8 is OCP e4m3fn/e5m2 (FNUZ) | Another maximum and NaN encoding: re-cast, never bit-copy; `torch.float8_e4m3fn`, not `_fnuz` | hw: `shared/dtype_numerics.md` |
| No TF32 (TF32 MFMA) | fp32 GEMMs need another path | hardware SKILL.md, quick facts |
| 256 CUs in 8 XCDs, 4 MB L2 per XCD (304 CUs) | Grid and wave arithmetic; tile counts in multiples of 8; AITER's tables key on `cu_num` = 256 | hw: `shared/l2_xcd_swizzle.md`; `../STALE.md` lists KernelForge pages that say 304 |
| wave64, VGPR and AGPR budgets as on gfx942 | Block sizes in multiples of 64; occupancy drops in steps (one at 256 VGPRs) | hw: `shared/wavefront_simd_vgpr_agpr.md`; `triton/pitfalls.md`; `hip/pitfalls.md` |

Price the kernel against measured numbers (the hardware skill's "Measured, not spec"), not the
spec peaks.

## 3. Check what the image ships: `probe-image.sh`

Run it once per image digest, before a plan depends on a DSL:

```bash
docker exec -i "$C" bash -s < .claude/skills/kernels/probe-image.sh
docker inspect --format '{{.Image}}' "$C"        # the digest, from the host
```

It prints one `OK` or `MISSING` line per piece, with details indented: ROCm, hipcc, `rocminfo`'s gfx
name, CU count and LDS size, CK headers, hipBLASLt, ninja, rocprofv3; Python, torch (HIP version,
device arch, fp8/fp4/e8m0 dtypes), Triton (target, `tl.dot_scaled`, Gluon), FlyDSL (version,
`flydsl.compiler` and `flydsl.expr`, cache), AITER (its asm `.co` files per op for each gfx,
`aiter/ops/flydsl/` and `is_flydsl_available()`, its Triton ops, CK sources for JIT, prebuilt
modules, tuned CSVs with their row counts and the rows that name FlyDSL); vLLM, SGLang, ATOM and
MoRI versions; the kernel-related environment. It never fails.

What to do with it:
- **FlyDSL MISSING:** AITER's FlyDSL rows and paths are inert in this image, and a FlyDSL kernel
  plan is blocked by the image, which is not a dead end (`serving-playbook` rules; image-manifest
  A4 lists the images that carry FlyDSL kernels). Installing a package into the container changes
  the environment under test: only if the pack allows it, recorded with its version.
- **Tuned CSVs with rows that name FlyDSL** mean the table already routes some shapes there: check
  whether yours are among them before writing one.
- **Record the output** in the plan's S1 record. KernelForge's pages cite AITER `b467ce342` and FlyDSL
  `2812676`; the image's versions win wherever they differ.

## 4. The workflow: kernel-plan stages with these tools

| Stage (kernel-plan §5) | Do | Tool |
| --- | --- | --- |
| Before S1 | Probe the image; find the dispatcher's options and the tuned tables for the op (§1); pick the language (§2) and say why in plan §4 | `probe-image.sh`; dossier `kernels.md` |
| **S1 Reference** | Wrap the path served now as a callable `ref`, called exactly as the server calls it, at the served shapes from dossier part 5 (rows per step and context as distributions, not the client's caps). Time it graph-captured and compare with the in-situ profile | `kbench.py --ref-only` |
| **S2 Correct** | The candidate against `ref` over the served shapes and the edge cases (one row, the largest capture size, empty experts, degenerate routing), at the plan's §4 tolerance; the negative controls must fire | `kbench.py` (correctness half) |
| **S3 Fast** | Graph-captured time at the served shapes, worst of N and median, with control drift; compare with the share of the gap the plan set | `kbench.py` (timing half) |
| **S4 In tree** | The candidate at its call site behind a `VLLM_*` switch, default off, declared `SWITCH: NAME=1`; the harness, at the KEEP, serves it on and checks: a log line naming the switch from every rank, 0 JIT compiles while serving, the accuracy gate | the campaign; the `profiler` skill for kernel names |
| **S5 A/B** | The campaign's own gate, at every operating point the plan names | the campaign |

**Zero JIT compiles while serving:** Triton, FlyDSL and AITER all compile on first call. Warm them
before graph capture. `FLYDSL_RUNTIME_RUN_ONLY=1` turns a FlyDSL cache miss into an error instead
of an inline compile (`flydsl/env_vars.md`); AITER's first-call JIT takes 30 s or more
(`aiter/pitfalls.md`). Per-step counts are runtime arguments, not compile-time constants
(kernel-plan §4).

**Run kbench only when nothing is being measured on the node.** It shares HBM bandwidth, power and
the host CPU with whatever else runs (runner `AGENTS.md` §4). Pin it to one GPU with
`HIP_VISIBLE_DEVICES`.

## 5. `kbench.py`

```text
python3 kbench.py MODULE [--ref NAME] [--cand NAME] (--spec JSON|@file | --case NAME=SHAPE:DTYPE,...)...
```

**The module** defines the callables (default names `ref` and `cand`; `FILE.py:NAME` picks one from
another file) and, optionally, `make_inputs(case, gen, device)` returning the input tensors when the
spec's random tensors are not enough (routing tables, preshuffled weights, scales).
- Each callable takes the inputs positionally (plus the case's `kwargs`) and returns its output
  tensor or a tuple of them. An in-place kernel returns the tensor it wrote; kbench gives each call
  its own copies of the inputs.
- **It must launch on torch's current stream**, or graph capture records nothing. A FlyDSL launcher
  takes `stream=fx.Stream(torch.cuda.current_stream().cuda_stream)`
  (`reference/examples/flydsl_gemma_rmsnorm/program.md`, rule 2).

**Inputs** come from a fixed seed (`--seed`, default 42). A spec input is
`{"shape": [...], "dtype": "bf16", "init": "randn", "scale": 1.0}`. Dtypes: `bf16 fp16 fp32 fp8
(e4m3fn) e5m2 e4m3fnuz e5m2fnuz fp4x2 e8m0 i8 u8 i16 i32 i64 bool`. Inits: `randn` (default),
`rand`, `zeros`, `ones`, `arange`; integers, packed fp4 and e8m0 take `low` and `high` (e8m0 defaults
to exponents near 2^0).

**Correctness**, per output:
- bf16, fp16, fp32: elementwise `|c − r| ≤ atol + rtol·|r|`. Defaults (rtol, atol): bf16 (1e-2,
  1e-2) and fp16 (1e-3, 1e-3), AITER's test table (`aiter/integration_patterns.md`); fp32 (1e-4,
  1e-5), kbench's own choice. Override with `--rtol`, `--atol`, or per case in the spec.
- fp8 outputs: SNR in dB, default 25 for e4m3 and 20 for e5m2 (`triton/tl_dot_scaled_gfx950.md`:
  e4m3's quantization floor is about 28 dB). `--metric snr --snr-db X` uses SNR on any float output,
  for a candidate that quantizes inside; X comes from the plan's §4.
- Integers, packed fp4, e8m0: exact bits. Non-finite values must match in every metric.
- **Negative controls:** the candidate's output shifted by one element, and the output pushed past
  the tolerance (ten times it; 6 dB under the SNR threshold; flipped low bits), must both fail. A
  shifted output that passes means near-constant outputs, degenerate inputs or a tolerance too wide,
  and the case fails.
- Timing runs only after correctness passes (`--time-anyway` overrides; `--no-time` skips it).

**Timing:**
- `--calls-per-graph` K calls (default 10) are captured in one HIP graph, so the in-graph cost of
  every launch is counted. Replays run back to back; µs per call = replay time / K.
- Blocks run ref, cand, cand, ref, N replays each (`--replays`, default 50). The report gives min,
  median, p90 and worst per side, drift between each side's two blocks, the ref/cand ratio of
  medians, and every raw sample.
- **Capture guard:** a kernel launched off the capture stream leaves an empty graph that replays in
  no time (`reference/examples/*/graph_harness.py`). kbench fills the graph's outputs with a
  sentinel, replays once and requires the right answer; otherwise the graph is invalid and the run
  fails.
- Eager per-call time is reported beside it, for comparison only: it is host-bound.
- Inputs smaller than the 256 MiB Infinity Cache are read warm by every replay. `--rotate R` cycles R
  copies inside the graph when the served path reads them cold.

**Exit status:** 0 when every case is correct and validly timed; 1 when a check fails, a negative
control does not fire, or a graph is invalid; 2 for a usage error. The JSON (`--out`) has the device,
torch and HIP versions, the kernel-related environment and, per case, every check and sample.

**Example**, a fused residual-add + RMSNorm candidate against AITER's op:

```python
# bench_addnorm.py
import torch
from aiter.ops.rmsnorm import fused_add_rms_norm_cu   # KernelForge operator_catalog.md; check the image's AITER
from addnorm_triton import addnorm                     # the candidate

EPS = 1e-6

def ref(x, res, w):                  # the served path, called as the server calls it; in place on x and res
    fused_add_rms_norm_cu(x, res, w, EPS)
    return x, res

def cand(x, res, w):                 # launches on torch's current stream
    return addnorm(x, res, w, EPS)   # returns (normed, new residual)
```

```json
{"cases": [
  {"name": "rows-1",   "inputs": [{"shape": [1, 7168],   "dtype": "bf16"}, {"shape": [1, 7168],   "dtype": "bf16"}, {"shape": [7168], "dtype": "bf16"}]},
  {"name": "rows-16",  "inputs": [{"shape": [16, 7168],  "dtype": "bf16"}, {"shape": [16, 7168],  "dtype": "bf16"}, {"shape": [7168], "dtype": "bf16"}]},
  {"name": "rows-512", "inputs": [{"shape": [512, 7168], "dtype": "bf16"}, {"shape": [512, 7168], "dtype": "bf16"}, {"shape": [7168], "dtype": "bf16"}]}
]}
```

```bash
docker cp .claude/skills/kernels/kbench.py "$C":/tmp/ && docker cp bench_addnorm.py "$C":/tmp/   # and the spec
docker exec -e HIP_VISIBLE_DEVICES=0 "$C" python3 /tmp/kbench.py /tmp/bench_addnorm.py \
    --spec @/tmp/addnorm.json --out /tmp/kbench-addnorm.json
```

It prints one block per case: `correct PASS|FAIL`, each output's metric, mismatches, SNR and
whether both negative controls fired; then each side's graph median, min, worst and drift, the eager
median, and the ref/cand ratio. Copy the JSON next to the plan's stage record. The row counts above
are placeholders: use dossier part 5's.

## 6. Pitfalls that cost time on gfx950

Each from the KernelForge page named (under `reference/knowledge_base/` unless the path says otherwise).

**Any language**
- **Stale caches are the first suspect when a fix "did nothing":** FlyDSL `~/.flydsl` (`rm -rf
  ~/.flydsl /tmp/flydsl*`; the disk cache does not see edits to imported helpers), Triton
  `~/.triton/cache`, AITER's JIT `~/.aiter/jit` (`AITER_REBUILD=1`), hipcc's `/tmp/comgr_*`, and CK
  builds whose ninja does not track headers or copy the new `.so` (`flydsl/pitfalls.md`,
  `triton/pitfalls.md`, `aiter/pitfalls.md`, `hip/pitfalls.md`, `ck/pitfalls.md`).
- **A launch off the capture stream** is not recorded: an empty graph, impossibly fast (examples'
  `graph_harness.py`).
- **`.clone()` keeps non-contiguous strides**; the kernel then pays a hidden `.contiguous()` copy,
  which can dominate the kernel's own time (`flydsl/pitfalls.md`, `hip/pitfalls.md`).
- **Monkey-patching an op misses bindings imported earlier** (`from x import f`); patch every
  binding, and prove the switch's path ran (`aiter/pitfalls.md`, `ck/pitfalls.md`).
- **Isolated timings mislead:** a CK kernel ran slower after a FlyDSL kernel than after CK kernels,
  from instruction-cache eviction at the backend boundary (`flydsl/pitfalls.md`). Only S5 books a gain.
- **FP8 tolerances are wide.** AITER's FP8 tests pass at atol 0.3, rtol 0.1 (`aiter/pitfalls.md`);
  gate fp8 on SNR and say which threshold.

**FlyDSL** (`flydsl/pitfalls.md`)
- `range_constexpr(n)` unrolls at trace time; `range(n)` is a runtime loop whose index cannot index a
  Python list; `range(..., init=[...])` with Python-int bounds silently drops `init`.
- No definition inside a branch used after it, no early return, no mutation of captured variables:
  the tracer needs one value per path.
- `mfma(LHS, RHS, acc)`: LHS is M, RHS is N; swapped operands give a transposed result. An LSE kept
  in the wrong domain reads as 20–25 dB SNR instead of > 30.
- `buffer_load` offsets are in elements, not bytes. `gpu.barrier()` in a divergent branch hangs.
- `waves_per_eu` is per kernel: never inherit it. `fx.Constexpr` arguments are part of the compile
  cache key (`flydsl/overview.md`).

**Triton** (`triton/pitfalls.md`, `triton/autotune_guide.md`)
- wave64: blocks under 64 lanes waste lanes; `BLOCK_M` or `BLOCK_N` of 32 underuses MFMA on gfx950.
- Autotune at the served shape; a config tuned on another shape does not carry over.
- The `TRITON_HIP_*` and `AMDGCN_USE_BUFFER_OPS` codegen knobs flip sign by kernel and image
  (amd-knobs §9, #69, #396).

**CK** (`ck/pitfalls.md`, `ck_playbook.md` §16)
- The warp tile is not an independent knob: change it with the block tile and LDS descriptors.
- Aliasing LDS slots can decide occupancy; a redesign proves its LDS budget first.
- CK's swizzles assume 32 banks; on gfx950 some padding is unnecessary.

**HIP and assembly** (`hip/pitfalls.md`)
- An inline-asm `"+a"` constraint drops AGPR index 0 (output at 21 dB SNR); use the builtin with an
  empty `"+v"` asm barrier.
- Occupancy steps down at 256 VGPRs; dropping under it can cut a kernel's latency substantially.
- Buffer loads clamp out-of-bounds reads instead of faulting: test non-power-of-two shapes.
- Direct-to-LDS loads are 4, 12 or 16 bytes per thread on gfx950 and 4 on gfx942: guard by arch.

**AITER** (`aiter/pitfalls.md`, `tuning_db.md`)
- `@compile_ops` hides a failed build: the op becomes `None` (`'NoneType' object is not callable`);
  `ENABLE_CK=0` silently removes CK ops.
- Preshuffled weights need N % 16 == 0 and K % 32 == 0, or the result is silently wrong; a scale of the
  wrong dtype is silently wrong too.
- `fused_add_rms_norm_cu` writes both its input and its residual in place.
- A tuned row whose key differs from the live call in any field (bias, `gfx`, `cu_num`) is never used.

## 7. Where to read, by question

All paths are under `reference/`. KernelForge's own entry points: `knowledge_base/flydsl/README.md`
and `knowledge_base/ck/ck_playbook.md` §12; `knowledge_base/README.md` explains the citation
prefixes the CK and HIP playbooks use (`aiter-amd/...`, `[pdf:pN]`, `[asm-v2:§N]`), whose targets
are not copied.

| Question | Read |
| --- | --- |
| What FlyDSL is; decorators, layouts, atoms, compile pipeline | `knowledge_base/flydsl/overview.md`, then `dsl_api.md`, `compilation_pipeline.md`, `dialect.md` |
| The five canonical FlyDSL kernel patterns | `knowledge_base/flydsl/kernel_patterns.md` |
| A FlyDSL GEMM: preshuffle, ping-pong LDS, XOR16 swizzle, scheduler | `knowledge_base/flydsl/gemm_optimization.md` |
| LDS bank conflicts and swizzle math, gfx942 and gfx950 | `knowledge_base/flydsl/lds_optimization.md`; the hardware skill's LDS pages |
| Which FlyDSL kernels exist, in FlyDSL's repo and in AITER | `knowledge_base/flydsl/kernels_inventory.md`; `local_knowledge/languages/flydsl/skills/optimize/flydsl_levers/kernel_families.md` |
| How AITER selects a FlyDSL kernel, and its fallback | `local_knowledge/framework/aiter/skills/optimize/aiter_levers/flydsl_path.md` |
| FlyDSL environment variables, caches, arch override | `knowledge_base/flydsl/env_vars.md` |
| Porting a Triton kernel to FlyDSL | `examples/triton2flydsl-softmax-flydsl-rewrite/` (start here), `examples/triton2flydsl-mxfp8-grouped-gemm/` (MXFP8 MoE grouped GEMM, decode and prefill shapes) |
| A FlyDSL kernel with a driver, oracle and graph harness | `examples/flydsl_gemma_rmsnorm/`, `examples/flydsl-softmax-forge-loop/` |
| Triton on AMD: pitfalls, autotuning, `tl.dot_scaled` on gfx950 | `knowledge_base/triton/pitfalls.md`, `autotune_guide.md`, `tl_dot_scaled_gfx950.md` |
| CK: the tile model; the full technique catalog | `knowledge_base/ck/tile_programming.md`; `ck_playbook.md` (§12 reading order, §16 MI300/MI355 specifics) |
| HIP: patterns, MFMA builtins and registers, build, HipKittens | `knowledge_base/hip/kernel_patterns.md`, `mfma_and_register_ops.md`, `compilation_and_build.md`, `hipkittens_guide.md` |
| An MXFP8 grouped GEMM in HIP | `knowledge_base/hip/mxfp8_grouped_gemm_hip.md` |
| The FP4 and FP8 scaled-MFMA operand layouts | `knowledge_base/skills/mfma_16x16x128_f8f6f4_fp4_operand_layout.json`, `mfma_16x16x128_f8f6f4_operand_layout.json` |
| AITER's op signatures | `knowledge_base/aiter/operator_catalog.md` |
| AITER's backends, dispatch rules, asm directories, tuned CSVs | `knowledge_base/aiter/backend_dispatch.md` |
| AITER's JIT, build and environment | `knowledge_base/aiter/jit_and_build.md` |
| AITER integration: quantization flows, MoE forward, test tolerances | `knowledge_base/aiter/integration_patterns.md` |
| AITER's traps | `knowledge_base/aiter/pitfalls.md` |
| Tune AITER's table or write a kernel; the tuner recipe | `local_knowledge/framework/aiter/overall/authoring_delegation.md`, `tuning_db.md` |
| Tune AITER's fused MoE | `local_knowledge/framework/aiter/operators/fused_moe_grouped_gemm/tuning.md` |
| A KernelForge number that disagrees with the hardware skill | `STALE.md` |

## Rules

- **No performance claim without a source.** A gain measured on another model, GPU, image or
  operating point is a lead; price it here. Where the evidence is thin (FlyDSL's own gain on gfx950
  is, today), say so in the plan.
- **Probe the image before a plan depends on a language,** and record what it printed.
- **Correctness before timing, with a negative control that fires.** Time graph-captured, at the
  served shapes. A kernel-level win is not a serving win: S5 decides.
- **Never edit KernelForge's files in `reference/`.** They are as copied; corrections go in
  `reference/STALE.md` and in this file.
