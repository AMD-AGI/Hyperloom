# Part 8: kernel usage

Which kernels serve each op at the served shapes, why those were chosen, and
whether a better kernel exists. Inputs: the profile (part 7, from the
`profiler` skill) and the code reading (part 3). Work op family by op family,
largest first.

## 1. What ran

From the profile report's stratified tables (`by_kernel`, `by_module_op`; with
the CPU op tree, `by_aten_op`), per regime and step kind, take every kernel
with at least 1% of GPU time in any regime:

- symbol, calls per step, GPU time per step and its share;
- the module scope that launched it, and its shapes (the first input, or the
  aten op's arguments with the CPU op tree);
- the step kinds it runs in (decode, prefill, mixed; forward, backward).

**One symbol can serve many call sites**: every MXFP8 linear of a model can
run as one kernel, whatever its projection. Split such a symbol by scope
before ranking.

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
source. This repository keeps a copy at `src/kernelforge`.

## 6. Check a candidate before proposing it

- Correctness against the exact path it replaces.
- GPU time at the served row counts and contexts (part 5), graph-captured
  where the served path is.
- Counters for the ones that move (`rocprofv3`: the `profiler` skill, section 5).
- Keep the raw samples and the script that made them.

A kernel-level win is still not a serving win: part 11 gives it a ceiling,
and an A/B decides.

## The table

| Op family | Kernels that ran (provider) | Share per regime | Shapes | Why this kernel | Unused alternatives (why; how to try) | KernelForge | gfx950 gap | Lead |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

An illustrative row, made up to show the form (not a measurement), for an
FP8-activation, MXFP4-weight MoE's experts:

- **Ran:** AITER's sort and quant-sort, then its fused-MoE stage-1 and
  stage-2 kernels; FP8 activations, MXFP4 weights, intermediate `<I>` per
  rank.
- **Why:** the tuned table has no row for the model's key, so every token
  tier runs the untuned fallback.
- **Unused:** another stage-2 kernel family, which covers other intermediate
  sizes, not `<I>`: shape family not covered.
- **Lead:** AITER's MoE tuner, deployed as an `AITER_CONFIG_FMOE` overlay
  (KernelForge's `fused_moe_grouped_gemm/tuning.md`); failing that, a new
  kernel for `<I>`.
