# AGENTS.md — instructions for coding agents working in this repo

This is the RDNA fork of [AMD-AGI/Hyperloom](https://github.com/AMD-AGI/Hyperloom).
Its job is to make Radeon-class AMD GPUs usable with Hyperloom **and to be
honest about which ones are actually proven**.

If you are an agent operating here, your task is almost always one of two
things. Both end with **posting a report**, because a result nobody can see is
the same as no result.

1. **CONFIRM support** on hardware the user physically has → §1
2. **ADD support** for a gfx target that is not mapped yet → §2

and when either one goes wrong, **§3 is the playbook** for working out why.

---

## THE ONE RULE

**A claim is MEASURED or it is ASSUMED. Never publish an ASSUMED claim as support.**

- MEASURED = the command ran **on this machine, this session**, and you are
  pasting its real output.
- ASSUMED = theory, a spec sheet, a related arch, a previous session, "it should
  work because gfx1100 and gfx1102 are both RDNA3."

Assumptions are useful for deciding what to *test*. They never *close* a
question. A support matrix entry is a promise to a stranger whose hardware you
cannot see — if you did not run it, say "untested" and move on. That is a
perfectly good answer and costs nobody anything. A wrong "supported" costs
someone a day.

Corollaries that have already bitten this project:

- **Do not grade your own output.** Where a harness can judge a result, let it
  (§1.3 uses Hyperloom's own `is_valid_measurement()`, not your reading of the
  JSON). "The JSON looks right" is not a verdict.
- **A missing tool must fail loudly.** A silent empty result is
  indistinguishable from "the feature is absent" — which is the exact wrong
  answer from a probe whose whole job is establishing truth.
- **Verify the path you are testing is the live one.** From upstream
  [#1041](https://github.com/AMD-AGI/Hyperloom/issues/1041): compare the actual
  rendered source that was compiled, rather than inferring engagement from log
  markers. A knob that looks plausible but never reaches the GPU will otherwise
  "work".
- **Measured on A does not close B.** Same ISA family is not the same board.

---

## §1 — CONFIRM support on hardware you have

### 1.1 Run the probe

```bash
./examples/rdna/rdna-support-probe.sh            # identity + resolution only, no GPU work
./examples/rdna/rdna-support-probe.sh --bench    # adds a real benchmark + harness verdict
```

Use `--bench` whenever a model is available (`MODEL=/path/to/*.gguf` and a
llama.cpp build). Without a benchmark you can only confirm that the board is
*recognised*, which is the weaker half of the claim.

The probe prints a paste-ready markdown block and one of three verdicts:

| Verdict | Means |
|---|---|
| `CONFIRMED` | Board resolves **and** the runner produced a measurement Hyperloom's own validator accepted |
| `PARTIAL` | Board resolves, but no accepted measurement (no model, no llama.cpp, or the run failed) |
| `UNSUPPORTED` | The arch is not in `_GFX_TO_RUNNER` — go to §2 |

### 1.2 Do not edit the matrix from a PARTIAL

`docs/rdna-port/README.md`'s support matrix moves to "MEASURED" only on
`CONFIRMED`. A `PARTIAL` is still worth posting — it tells us the identity
plumbing works on your board — but it is not support.

### 1.3 Post it

Open an issue using the **RDNA hardware report** template and paste the probe
block verbatim. Do not summarise it, do not tidy the numbers, and do not omit
the failures: a report whose honest verdict is "this regressed" or "this did
not build" is worth more than a clean one, because it is the one nobody else
can produce.

State plainly what you did **not** test. Card count matters (a 4-GPU host is
not a validated 2-GPU host), and so does anything you skipped.

---

## §2 — ADD support for a new gfx target

The port for gfx1201 is the template — read that commit first. Adding a board
is small, because upstream derives the accepted-board set and the `rocm-smi`
product tags from **one table**.

### 2.1 The change

1. **`src/hyperloom/common/gpu_identity.py`** — one row:
   `"<board>": ("<gfx arch>", <CU count>)`.
   Read the CU count **off the board** (`amd-smi static -g 0`), never a spec
   sheet or a web search. This single row also makes the board CLI-accepted and
   autodetectable, because `_AMD_GPU_TYPES` and `_PRODUCT_TAGS` derive from it.
   The board key must be a substring of what `rocm-smi --showproductname`
   prints, or autodetect will not match it.
2. **`src/hyperloom/inference_optimizer/gpu_types.py`** — a `_GFX_TO_RUNNER`
   entry mapping the gfx arch to the runner label.
3. **A runner script** in `examples/rdna/`. `custom_r9700.sh` is llama.cpp-based
   and should port to any RDNA target with a rename and *its own* measured
   caveats. Keep the contract: read `$MODEL`/`$RESULT_DIR`/`$RESULT_FILENAME`,
   write an InferenceX-shaped `inferencex_result.json`, refuse to emit a
   zero-throughput "success".
4. **Tests** in `src/hyperloom/inference_optimizer/tests/`. Mirror
   `test_rdna4_r9700_support.py`, **including its negative tests**. If upstream
   asserts your arch is unmapped (it asserts this for gfx1100 in
   `test_profile_and_kernel_handlers.py`), flip that assertion into positive
   coverage in the same commit.
5. **The support matrix** in `docs/rdna-port/README.md`, plus any profiling
   caveats that differ from gfx1201's.

### 2.2 Map only what you ran

Do not add a `_GFX_TO_RUNNER` entry for a sibling chip you have not booted.
gfx1200 (Navi 44) is deliberately absent here for exactly this reason, and
there is a test locking that absence. Adding an untested arch is not
generosity — it is a support claim written by someone who cannot honour it.

### 2.3 Verify before you post

```bash
PY=<a python with pytest>
PYTHONPATH=src $PY -m pytest src/hyperloom/inference_optimizer/tests/ -q \
  -k "gpu or quant or preflight or parser or provenance"
./examples/rdna/rdna-support-probe.sh --bench
```

Both must pass, and the probe must say `CONFIRMED`, before the matrix changes.

---

## §3 — Playbook: when it does not work

Most "unsupported hardware" is not unsupported hardware. Work it in this order.

### 3.1 Exhaust the mundane before the exotic

When a counter reads zero, a feature reports unsupported, or a kernel
underperforms, check **in this order**, and stop at the first thing that
explains it:

1. **perf-level / power gating** (a zero counter is usually a gate, not silicon)
2. **env-var traps** (a stray `ROCR_VISIBLE_DEVICES`, `HSA_*`, `HIP_*` left set)
3. **the vendor tracker** — search ROCm/llama.cpp issues before theorising; the
   bug is often known, with a workaround
4. **driver / firmware / runtime versions** (and whether the one you *think* is
   loaded is the one actually loaded — check `ldd`, not your memory)
5. **build flags / target arch** (was it even compiled for this gfx?)
6. **only then** tile sizes, occupancy, ILP, kernel internals

Most "impossible" walls are a mundane switch. Reaching for the exotic
explanation first is the single most expensive habit in this work.

### 3.2 Isolate by layer, bottom-up

Name the layer the fault lives in, verify with **that layer's own ground-truth
tool**, and do not blame one layer for another's problem. Cross-layer
misattribution — blaming silicon for a compiler artifact, or the model for an
environment trap — sends you optimising the wrong thing entirely.

| Layer | Ground truth | Classic trap |
|---|---|---|
| L0 board / link / topology | `lspci -vv`, transfer benchmarks | Checking only the GPU's own link. Navi boards sit behind an on-board switch whose **upstream** port is where the motherboard downgrades you — the leaf can read Gen5 while the real path is Gen3 |
| L1 silicon / ISA | the assembler (`llvm-mc`) — it cannot lie | Assuming a marketing feature is present |
| L2 firmware / driver | `amd-smi`, `dmesg`, sysfs perf level | A counter reading 0 is a **gate**, not broken hardware |
| L3 ROCm runtime / env | `ldd`, `HIP_VISIBLE_DEVICES`, `LD_LIBRARY_PATH` | A stale env var causing silent CPU fallback; the loader picking a different ROCm than you meant |
| L4 compiler / toolchain | `--save-temps` + disassembly, target features | A builtin that **arity-checks before target-feature**, so an unavailable instruction looks available |
| L5 framework kernels | disassembly; correctness vs a CPU reference | "Instruction unused" concluded from a *sample*; the kernel may exist but be dormant or undispatched |
| L6 model / quant format | bit round-trip vs reference | Trusting a pack layout without a bit-identical proof |
| L7 serving | served tokens, VRAM, throughput | Blaming the model for an L2/L3 fault (garbled output, CPU fallback, a reaped process) |

### 3.3 Specific failures, and what they usually are

| Symptom | Look here first |
|---|---|
| Probe says `UNSUPPORTED` but the GPU works | The arch simply is not mapped — §2. Not a hardware problem |
| Board not detected, arch reads `unknown` | `rocm-smi`/`rocminfo` absent (a minimal runtime install omits them; the probe falls back to `amd-smi`), or the board key is not a substring of the product name |
| Runner exits 3, "llama-bench not found" | Working as designed — it refuses to report a zero rather than let a missing binary read as a slow GPU. Set `LLAMA_BENCH` / `LLAMA_CPP_DIR` |
| Benchmark runs, validator REJECTS | `output_throughput` is 0/absent, or a `quality_gate` failed. A failed gate makes a run unselectable **no matter how fast it was** — that is the gate working |
| Throughput looks impossibly good | You are probably measuring cache, not the real path. If a result exceeds the memory roofline, the working set fits in cache — enlarge it until it does not |
| A/B shows a win | Check the direction: `speedup = baseline ÷ candidate`. Inverted ratios have reported regressions as wins here before |
| A/B shows a huge win | Suspect the **baseline**, not the candidate. Compare against a competent baseline, not a strawman; a broken baseline (NaN/Inf, a pathological path) manufactures spectacular speedups |
| Numbers move run to run | Interleave A/B, discard warmup, take medians of ≥3, and re-measure cold. A number you did not measure *this run* does not exist |

### 3.4 Before you claim a fix

- It **compiled and ran on the target**, and you have the output.
- A **correctness gate passed** — versus a CPU reference or a quality metric.
  A wrong-but-fast kernel is worth zero.
- The path you changed is the one that actually **executed**. Verify by
  comparing the rendered source or the disassembly, not by trusting a log line.
- Say which claims are MEASURED and which are ASSUMED. Both belong in the
  report; only one belongs in the support matrix.

Negative results are first-class here. "This does not work on gfx1102, here is
the exact failure" is a real contribution, and often a more useful one than a
win, because nobody else can produce it without your hardware.

## Repo conventions

- **Branches.** `rdna` is the working branch. **`main` tracks upstream
  byte-for-byte — never commit to it.** PR branches for upstream are cut from
  `main` and cherry-pick only port commits, so they stay reviewable.
- **Never open a PR against AMD-AGI/Hyperloom without explicit human approval.**
  Same for posting on upstream issues. Draft it and ask.
- **Licence.** Upstream files stay MIT © AMD with their SPDX headers intact.
  New files carry their own SPDX header. This fork is not an AMD product and
  must not present itself as one.
- **Commits** explain *why*, and say which claims were measured.

## RDNA profiling traps

Do not report profiling numbers from an RDNA4 box without accounting for these
(all measured on gfx1201; re-verify on your own arch rather than assuming they
carry):

| Trap | Why it misleads |
|---|---|
| `--setperflevel high` | **Reduces** inference throughput ~10% vs `auto` while tightening variance — pinning clocks for "stable" absolutes makes the number worse |
| `profile_standard` | Un-gates the memory counters but throttles to ~1593 MHz vs ~2330 — it suppresses any benchmark running beside it. Prefer `profile_peak` |
| `GL2C` reading zero | Perf-level gating, not broken firmware |
| `FETCH_SIZE` and some derived PMC counters | Collect zeros — reads as "no traffic" rather than "unsupported" |
| PC sampling | `host_trap` only, min interval 512; `stochastic` is unsupported |
| No MFMA on RDNA | Anything reaching for CDNA matrix cores must branch on gfx arch, not on "is AMD" |
