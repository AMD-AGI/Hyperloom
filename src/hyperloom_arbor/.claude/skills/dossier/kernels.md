# Part 8: kernel usage

Which kernels serve each op at the served shapes, why those were chosen, and
whether a better kernel exists. Inputs: the profile (part 7, from the
`profiler` skill), the code reading (part 3) and, when the user provides it,
the reference platform (part 9). Work op family by op family, largest first.

## 1. What ran

From the profile report's stratified tables (`by_kernel`, `by_module_op`; with
the CPU op tree, `by_aten_op`), per regime and step kind, take every kernel
with at least 1% of GPU time in any regime:

- symbol, calls per step, GPU time per step and its share;
- the module scope that launched it, and its shapes (the first input, or the
  aten op's arguments with the CPU op tree);
- the step kinds it runs in (decode, prefill, mixed; forward, backward).

**One symbol can serve many call sites**: every MXFP8 linear in DSV4.1 ran as
one kernel, `_mxfp8_linear_kernel`, whatever its projection. Split such a
symbol by scope before ranking.

## 2. Who provides it

Find the provider from the code, not the name: follow the call site (part 3)
down to the library function that launches the kernel. Names are only hints:
`Cijk_` is a hipBLASLt (Tensile) solution, `triton_` an Inductor-generated
kernel, `ck::` or `ck_tile::` (demangled) Composable Kernel. AITER ships
precompiled assembly per arch in `hsa/<arch>/<op>/` and JIT-built CK and
Triton kernels; vLLM has its own HIP and Triton ops; PyTorch has native ones;
collectives come from RCCL or a custom all-reduce.

## 3. Why this kernel

At each dispatch site, record what chose it:

- the conditions in the dispatcher: dtype, shape bucket, arch, environment
  flags;
- **tuned-table hit or fallback.** AITER's tuned tables live in its
  `aiter/configs/` (per-model tables under `model_configs/`), overlaid with
  `AITER_CONFIG_*` variables (`AITER_CONFIG_FMOE` for fused MoE). The lookup
  key includes `gfx`, `cu_num` (256 on MI355X), the padded-token bucket and
  the full quantization signature; a miss falls back to a default
  configuration silently.
- what the server logged at that site, per run. A flag being set is not the
  path running.

## 4. What else could run

Search the code already shipped for other implementations of the same op:

- **The dispatcher's other branches** (assembly, CK, Triton; other attention
  or MoE backends the framework selects by flag). For each: the dtypes,
  shapes and arch it covers, and why it does not run here.
- **The kernel inventories**: AITER's `hsa/gfx950/<op>/` code objects and
  their CSV indexes, its CK and Triton ops, vLLM's ROCm ops. Read them in the
  container's installed source (`python3 -c "import aiter, os;
  print(os.path.dirname(aiter.__file__))"`), not from a pin quoted in a doc.
- **Why unused**, in one of these classes: shape family not covered, arch not
  covered, behind a flag that is off, tuned table missing, numerics differ,
  or no reason found (a lead in itself).
- **How to try it**: the flag, the table overlay, or the code change. Each
  is a change behind a flag, through the campaign's gates.

## 5. What AMD has written down

Read KernelForge (AMD-AGI's kernel-optimization knowledge base) before
writing or tuning anything:

- `local_knowledge/framework/aiter/operators/<op>/`: overview, dispatch,
  knobs, tuning and numerics per AITER operator;
- its tuning database, `examples/` (worked kernel optimizations), `tasks/`
  (task templates), and ISA notes (operand layouts).

Its pages cite AITER `b467ce342`: verify every claim against the installed
source. The runner keeps a read-only copy at
`$HOME/runner-ws/tools/KernelForge`, and `docs/kernelforge.md` maps
it to DSV4.1's shapes.

## 6. What the reference platform runs

From part 9 (the B200 trace's kernel census, or the recipe's backend flags):
the kernel serving the same op, what it implies (a fused path, another tile
shape, another number format, another backend), and the gfx950 gap that may
block a port (formats, matrix instructions, wavefront 64, on-chip memory,
asynchronous copies). A B200 choice is a lead, never evidence for gfx950.

## 7. Check a candidate before proposing it

- Correctness against the exact path it replaces.
- GPU time at the served row counts and contexts (part 5), graph-captured
  where the served path is.
- Counters for the ones that move (`rocprofv3`: the `profiler` skill, section 5).
- Keep the raw samples and the script that made them.

A kernel-level win is still not a serving win: part 11 gives it a ceiling,
and an A/B decides.

## The table

| Op family | Kernels that ran (provider) | Share per regime | Shapes | Why this kernel | Unused alternatives (why; how to try) | KernelForge | Reference platform | gfx950 gap | Lead |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

A worked row, DSV4.1 MoE experts on MI355X (runner's D06 and
`docs/kernelforge.md`, 2026-09-29):

- **Ran:** AITER's sort and quant-sort, then
  `mfma_moe1_silu_mul_afp8_wfp4_bf16_*` (stage 1) and
  `mfma_moe2_afp8_wfp4_bf16_*` (stage 2); FP8 activations, MXFP4 weights,
  hidden 5120, intermediate 640 per rank, 384 experts, top-6.
- **Why:** `tuned_fmoe.csv` (2,462 rows) had no row for the target's or the
  drafter's key, so every token tier ran the untuned fallback (`block_m` 32).
- **Unused:** AITER's opus A8W4 stage-2 kernel covers intermediate 512 and
  768, not 640: shape family not covered.
- **Lead:** AITER's MoE tuner, deployed as an `AITER_CONFIG_FMOE` overlay
  (KernelForge's `fused_moe_grouped_gemm/tuning.md`); failing that, a new
  opus family or a FlyDSL A8W4 kernel for 640.
