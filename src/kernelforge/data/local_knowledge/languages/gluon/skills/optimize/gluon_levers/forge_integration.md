---
title: Authoring Gluon inside a forge-loop campaign — change shape, measurement hygiene, version traps
kind: skill
gens: [gfx942, gfx950]
dtypes: [both]
regimes: [both]
status: experimental
updated: 2026-08-23
sources:
  - https://github.com/triton-lang/triton/releases
  - https://github.com/triton-lang/triton/issues/10265
  - https://github.com/ROCm/gfx950-gluon-tutorials
  - https://github.com/ROCm/aiter
---

# Gluon inside a forge campaign

**Read this before your first edit.** Gluon is a fine language and a poor fit for a careless change
shape: the most natural way to add a Gluon kernel — a new file — is the one shape a forge KEEP cannot
carry. This card is about how to shape the change so the loop can actually keep it, how to keep the
measurement honest, and which version traps burn a whole session.

## TL;DR
1. **Put the Gluon kernel in the same tracked file as the code it replaces**, keep the public entry
   signature identical, and select the backend at dispatch. A brand-new file is not committed by a
   KEEP unless the campaign was launched with `--commit-new-path`.
2. **Probe the toolchain before you write anything.** Gluon is `triton.experimental`, it is not
   stabilized, and 3.7.0 shipped with a symbol missing that breaks its own tutorial.
3. **Environment variables are part of the measurement, not part of the kernel.** `TRITON_ENABLE_LLIR_SCHED`
   and `TRITON_ENABLE_AMDGCN_AS` change the generated code. A number measured with them set is not
   comparable to one measured without unless they travel with the candidate.
4. **SNR is a pre-filter, not the gate.** A layout or scale-packing error produces plausible garbage;
   the task's own `correctness_command` is what decides.

## 1. Change shape: same file, same entry, dispatch inside

### Why
Forge's KEEP commits **tracked modifications**. Untracked files the agent created are committed only
if they match the campaign's `--commit-new-path` allowlist; otherwise **a KEEP cannot carry them and a
REVERT cannot remove them**, and the loop will tell you so under a `## New files that cannot ship`
heading. At that point the measured tree is not the committed tree, and the iteration is wasted.

Meanwhile the measurement surface is protected and you cannot edit it: the driver, anything matching
`*harness*.py` / `test_*.py` / `*_ref.py` / `*_reference.py` / `config.yaml` / `task_runner.py`, and
anything under a `test/`, `tests/`, `benchmark(s)/`, `script(s)/` or `perf/` directory. An edit there is
blocked in-session, and a tampered candidate is force-reverted. **So the public entry point the driver
calls must keep working with exactly the same signature.**

Both constraints point at the same shape, and it happens to be what production already does.

### The shape

```python
# same tracked file that already holds the Triton kernel

@triton.jit
def _op_kernel_triton(...):        # the incumbent, kept as the fallback
    ...

@gluon.jit
def _op_kernel_gluon(...):         # the new path
    ...

def op(...):                       # UNCHANGED public entry — this is what the driver calls
    if _use_gluon():               # cheap, cached, arch- and toolchain-gated
        return _launch_gluon(...)
    return _launch_triton(...)
```

`_use_gluon()` should be decided once and cached, not probed per call — see § 2.

### Precedent
This is `aiter/ops/triton/attention/pa_mqa_logits.py`: one file, one public entry
(`deepgemm_fp8_paged_mqa_logits`), a Gluon path selected by `enable_gluon_pa_mqa_logits` and a plain
Triton JIT kernel as the fallback. Note two things about it. First, the *directory* is named `triton`
and holds both — **a path never tells you the language**. Second, the Gluon path is the more capable
one: it supports `Preshuffle` and `KVBlockSize > 1`, which the Triton path does not. Going lower-level
bought capability, not just speed. Read the file itself — this repo keeps no card for it.

### Keeping the fallback is not optional politeness
It is what makes the candidate safe to keep. The task's `compile_command` often builds a **smaller
shape** than the one the loop benchmarks, and a Gluon path with a shape or arch constraint that the
benchmark satisfies and the compile check does not will fail acceptance after passing everything else.
A live fallback turns that from a failed candidate into a taken branch.

### If it genuinely cannot be one file
Say so in your findings, name the exact path, and say why the change cannot live in a tracked file.
The operator has to add `--commit-new-path <path>` to the campaign — you cannot add it yourself, and it
is immutable for the campaign once set.

## 2. Probe before you build

Do this **once, before writing Gluon**, and put the result behind the dispatch gate. A session that
writes 300 lines of Gluon and then discovers the import fails has spent an iteration for nothing.

```bash
python -c "
import triton
print('triton', triton.__version__)
try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    print('gluon exports', sorted(getattr(gluon, '__all__', [])))
except Exception as e:
    print('GLUON UNAVAILABLE:', type(e).__name__, e); raise SystemExit(1)
try:
    from triton.experimental.gluon.language.amd import cdna4
    print('cdna4 ops', [n for n in dir(cdna4) if not n.startswith('_')])
except Exception as e:
    print('cdna4 unavailable:', e)
"
```

Then confirm the **arch**, because half this language is CDNA4-only:

```bash
rocminfo | grep -om1 'gfx[0-9a-f]*'
```

If Gluon does not import, or the arch does not carry the feature your plan depends on, **say so and
plan a different direction.** That is a complete, useful iteration result — not a failure.

## 3. Version traps

Gluon is under `triton.experimental` and is **not a stabilized API**. These are the ones that have
actually bitten:

- **`gluon.aggregate` is not in the released wheels.** It exists on `main` (re-exported from
  `triton.language.core._aggregate`) but not in the 3.7.0 release, whose `__all__` is exactly
  `["constexpr_function", "jit", "must_use_result", "nvidia", "amd"]`; a 3.6.0 build shows the same
  five names. Any code using `@gluon.aggregate` breaks — including Triton's own
  `tutorials/gluon/07-persistence.py`. **Do not build a design around `@gluon.aggregate` without
  checking `gluon.__all__` on the actual build.** This is the concrete reason the probe in § 2 prints
  `__all__` rather than just checking that the import succeeded.
- **Triton 3.7.1 fixed a missing fence** between a shared-memory store and an async
  `copy_local_to_global`: the async copy could read shared memory before the store completed,
  **silently producing wrong results**. On 3.7.0 with an async shared-memory pipeline you are exposed.
  If correctness is intermittent or shape-dependent, check the Triton version before you debug your
  own code.
- **AsyncCopy-by-default for gfx950/gfx1250 was enabled upstream and then reverted on `release/3.7.x`.**
  `main` and a 3.7.x wheel do not behave the same. This does not change what *your* explicit
  `async_copy` does, but it does mean a Triton-side comparison number may not have been measured under
  the pipeline you think.
- **The AMD tutorials pin an annotated Triton tag** (`gfx950-tutorial-v0.1` in the blog,
  `gfx950-tutorial-v0.2` in the repo) and assume **ROCm ≥ 7.0** with Triton built from source. Kernels
  copied from there may not compile against a stock wheel. Do not assume a tutorial kernel is a
  drop-in.
- The `gl.amd.*` surface moves, and **`cdna3` is materially thinner than `cdna4`** — no `async_copy`,
  no `mfma_scaled`. A plan whose second rung is async-copy-to-LDS is a CDNA4 plan. Read `dir(cdna4)`
  (or `dir(cdna3)`) on your build rather than the docs for `main`; see
  [`../../../API_docs/amd_targets.md`](../../../API_docs/amd_targets.md) for the observed split.

When one of these bites, **that is the iteration's finding.** Record the version, the symbol, and the
error in your report — it saves every later session the same discovery.

## 4. Measurement hygiene

### Environment variables change the generated code
`TRITON_ENABLE_LLIR_SCHED=1` and `TRITON_ENABLE_AMDGCN_AS=1` are not runtime tuning — they change the
instruction schedule and the register allocation. They are the difference between two of the rungs in
the AMD ladder. Inside a campaign there are exactly two honest ways to use them:

- **Make them travel with the candidate** — set them from the kernel module's own import path (e.g.
  `os.environ.setdefault(...)` before the first compile) so any measurement of that source includes
  them, and the committed kernel keeps behaving the way it was measured. This is usually right, because
  the flags are properties of the kernel design, not of the run.
- **Sweep them explicitly** as `FORGE_SWEEP_*` knobs when the question is whether they help. One data
  point per command, echoed, per `common_methodology/optimization/lever_cheap_sweeps.md`.

What is **not** honest is exporting them in your shell and then reporting the number as the kernel's.
The loop's own canonical measurement will not have them set, and the candidate will regress on the
measurement that decides.

### JIT cache
Gluon uses Triton's cache (`TRITON_CACHE_DIR`, else `$TRITON_HOME/.triton/cache`, else
`~/.triton/cache`) and re-keys on source, so an edit recompiles — you do **not** need to clear it after
an ordinary source change, and forge treats those variables as reserved so a sweep cannot move them.
Clear it only when you have changed something outside the source that affects codegen (a Triton
rebuild, a flag that is not part of the cache key).

### SNR is a pre-filter, not the gate
Gluon's two most common wrong-answer bugs are **silent**, and both clear a loose numeric check:

- a **layout mismatch** that reads the right memory in the wrong order;
- **scaled-MFMA scale packing**, whose order differs between `mfma_scaled_16x16x128`
  (`op_0, op_2, op_1, op_3`) and `mfma_scaled_32x32x64` (`op_0, op_1, op_2, op_3`);
- and inherited from Triton, the **fp8 FNUZ (gfx942) vs OCP (gfx950)** dialect mismatch.

Run the task's own `compile_command` and then its `correctness_command` yourself before you propose a
change. SNR ≥ 30 dB is a fast pre-filter; the task's tolerances are what decide, and they are not
forge's.

### Reference-check against the incumbent, not against a table
The ceilings in [`overview.md`](overview.md) are AMD-measured on gfx950 at large K. Your baseline is
the pristine measurement the loop took on this box. Never report a speedup against a published number.

## 5. A first-iteration plan that usually works

1. Probe the toolchain and the arch (§ 2). Record what you found.
2. Read the incumbent and find the **public entry** the driver calls. That signature is frozen.
3. Write `v0`: a **correct** Gluon kernel with explicit layouts, dispatched behind the gate, fallback
   intact. Expect it to be *slower* than the tuned Triton incumbent. That is a successful v0 — but note
   that the loop's KEEP gate will (correctly) reject it, so say clearly in your report that v0 is
   scaffolding and what the next rung is.
4. Only then start the ladder — buffer ops, then async-copy-to-LDS, then LDS layout — one rung per
   measurement.

If the session budget cannot reach at least the async-copy rung, **a Gluon direction is the wrong use
of this round**; the naive version will not beat a tuned Triton kernel and nothing will be kept. Say so
and pick a Triton-level direction instead.

## Cross-links
- When Gluon is the right call, and the full ladder: [`overview.md`](overview.md)
- The AMD ops the ladder uses: [`../../../API_docs/amd_targets.md`](../../../API_docs/amd_targets.md)
- Layouts and conversion costs: [`../../../API_docs/layouts.md`](../../../API_docs/layouts.md)
- Sweep contract: `common_methodology/optimization/lever_cheap_sweeps.md`
- Edit surface: `common_methodology/optimization/lever_edit_surface.md`
- Production dual-backend dispatch: `aiter/ops/triton/attention/pa_mqa_logits.py` (read the source;
  `framework/aiter/overall/dispatch_and_rebind.md` explains how aiter picks between the two paths)

## Sources
- `gluon.aggregate` missing from 3.7.0; the exact 3.7.0 `__all__`:
  https://github.com/triton-lang/triton/issues/10265
- 3.7.1 FenceAsync async-read-dependency correctness fix; AsyncCopy default enabled then reverted on
  release/3.7.x: https://github.com/triton-lang/triton/releases
- Pinned annotated tags, ROCm ≥ 7.0, build-from-source assumption:
  https://github.com/ROCm/gfx950-gluon-tutorials
- Production dual-backend (Gluon + Triton) dispatch in one file, Gluon path strictly more capable:
  https://github.com/ROCm/aiter
