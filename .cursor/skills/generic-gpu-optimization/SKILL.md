---
name: generic-gpu-optimization
description: |
  Autonomous DFS-guided performance optimization for arbitrary GPU codebases on AMD
  ROCm hardware (MI300X / MI325X / MI355X). Works on any repo that has a buildable
  GPU workload and a way to measure runtime — does NOT require an inference server,
  Primus/Megatron, or any specific framework. Auto-detects build system, benchmark
  harness, correctness tests, and kernel languages (HIP, Triton, CUDA->HIP, SYCL,
  PyTorch). Submits hot kernels to the GEAK MCP for AI-driven kernel rewrites.
globs:
  - "**/CMakeLists.txt"
  - "**/pyproject.toml"
  - "**/setup.py"
  - "**/build.sh"
  - "**/*.hip"
  - "**/*.cu"
  - "**/*.cuh"
  - "**/triton*"
---

# Generic GPU Optimization — DFS Orchestrator

## When to Use This Skill

Use this skill when the user points you at a repository containing GPU code and
asks you to make it faster, but the project is **not** a known inference server
(SGLang/vLLM) or training stack (Primus/Megatron). Examples:

- A C++/HIP library with Google Benchmark or Catch2 micro-benchmarks (e.g. hipRAFT,
  rocPRIM, rocSPARSE, custom kernel libraries)
- A standalone PyTorch script that runs a forward/backward loop
- A Triton kernel collection with a Python harness
- An HPC application with a `--bench` mode
- A research codebase with a `benchmarks/` or `bench/` directory

If the repo IS a SGLang/vLLM server use `inference-optimization`. If it is
Primus/Megatron training use `training-optimization`. Otherwise: this skill.

## Hard Constraints

1. **Correctness must not regress.** If the project ships tests (ctest, pytest,
   gtest, Catch2, unittest), they MUST pass after every kept change. If no tests
   exist, fall back to comparing benchmark numerical output (golden file) before
   accepting a patch.
2. **Build must remain reproducible.** Every change is recorded as either a
   compile-flag override, an environment variable, or a code patch. Nothing is
   accepted that cannot be re-applied to a clean checkout.
3. **One change per attempt.** Even though the agent can think in groups, each
   attempt isolates a single variable so attribution is clean.

## Architecture

```
SKILL.md (this file)        — DFS orchestrator
README.md                   — Usage guide and examples
actions/
  detect.md                 — Auto-discover build system, benchmarks, tests, kernels
  setup.md                  — Environment validation, GPU detection
  build.md                  — Build the project (CMake / pip / make / etc.)
  baseline.md               — Run the benchmark, record metric
  correctness.md            — Run the test suite, gate keep/revert
  profile.md                — Profile with the right tool for the stack
  compile-flags.md          — Try compiler / CMake flags
  env-vars.md               — Try ROCm runtime env vars
  kernel-opt.md             — Extract hot kernels and submit to GEAK
  patch.md                  — Apply / revert a code patch with correctness gate
  report.md                 — Produce final optimization report
scripts/
  common.sh                 — Shared helpers (metric extraction, kill, etc.)
  detect_project.sh         — Writes $RESULT_DIR/detected.env
  run_bench.sh              — Dispatches based on detected.env
  run_correctness.sh        — Runs detected test suite
  profile_rocm.sh           — rocprofv3 wrapper for native code
  profile_torch.sh          — torch.profiler wrapper for PyTorch code
kb/
  entries.jsonl             — Persistent knowledge base (seeded empty)
```

## DFS Search Tree

```
                        ┌──────────┐
                        │  SETUP   │   GPU type, ROCm version, Python env
                        └────┬─────┘
                             │
                        ┌────▼─────┐
                        │  DETECT  │   Build system, benchmarks, tests, kernel langs
                        └────┬─────┘
                             │
                        ┌────▼─────┐
                        │  BUILD   │   Initial clean build
                        └────┬─────┘
                             │
                        ┌────▼─────┐
                        │ BASELINE │   Run benchmark, record metric
                        └────┬─────┘
                             │
                        ┌────▼─────┐
                        │CORRECTNESS│  Confirm clean tests pass (sanity)
                        └────┬─────┘
                             │
                        ┌────▼─────┐
                        │ PROFILE  │   rocprofv3 OR torch.profiler
                        └────┬─────┘
                             │
              ┌──────────────▼──────────────┐
              │      HEURISTIC SCORING      │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────────────┐
              │    DFS over candidate actions       │
              │                                      │
              │ ┌──────────┐ ┌──────────┐ ┌───────┐ │
              │ │ ENV-VARS │ │COMPILE-  │ │KERNEL │ │
              │ │ (cheap)  │ │  FLAGS   │ │ -OPT  │ │
              │ └──────────┘ └──────────┘ └───────┘ │
              │                                      │
              │ Each: build → bench → correctness    │
              │       → keep or revert               │
              └──────────────┬──────────────────────┘
                             │
                      ┌──────▼──────┐
                      │   REPORT    │
                      └─────────────┘
```

## Heuristic Scoring

```
score = (expected_pct_reduction / cost_minutes)
        × (1 - correctness_risk)
        × (1 - build_risk)
        × target_gap_multiplier
```

### Initial Score Priors (set after DETECT)

The detect step writes `project_class` into `$RESULT_DIR/detected.env`. Choose
priors based on it:

| Action          | hip-cmake-bench | pytorch-script | triton-collection | hpc-app |
|-----------------|----------------:|---------------:|------------------:|--------:|
| env-vars        |               4 |              4 |                 3 |       4 |
| compile-flags   |               9 |              2 |                 3 |       8 |
| kernel-opt      |               8 |              5 |                 9 |       6 |
| torch.compile   |               0 |              7 |                 0 |       0 |
| precision-cast  |               2 |              5 |                 4 |       2 |

Re-score after every action using the same rules as `training-optimization/SKILL.md`
(success boosts similar actions ×1.5, failure penalizes ×0.5).

## State Schema

```python
state = {
    "repo_root": "",
    "project_class": "",          # hip-cmake-bench / pytorch-script / triton-collection / hpc-app / unknown
    "build_system": "",           # cmake / setup.py / make / pyproject / none
    "build_command": "",          # exact command resolved by detect
    "bench_command": "",          # exact command (may include filters/args)
    "bench_metric": "ms_per_iter",# or items_per_sec / GB_per_sec, etc.
    "test_command": "",           # ctest / pytest / etc., empty if no tests
    "correctness_mode": "tests",  # tests / golden-output / none
    "kernel_langs": [],           # subset of {hip, triton, cuda, sycl}
    "gpu_type": "",               # MI300X / MI325X / MI355X
    "gpu_arch": "",               # gfx942 / gfx950
    "rocm_version": "",

    "baseline_metric": 0.0,
    "current_metric": 0.0,
    "metric_lower_is_better": True,
    "cumulative_gain_pct": 0.0,

    "kept_env_vars": {},          # name -> value
    "kept_compile_flags": [],
    "kept_patches": [],           # list of git patches saved to $RESULT_DIR/patches/
    "kernel_candidates": [],      # populated by profile

    "action_stack": [],
    "completed_actions": [],
    "consecutive_discards": 0,
}
```

## Orchestrator Loop

```
PROCEDURE optimize(repo_root, [target_metric]):

  1. SETUP                → actions/setup.md
  2. DETECT               → actions/detect.md   (writes detected.env)
  3. BUILD (clean)        → actions/build.md
  4. BASELINE             → actions/baseline.md (runs detected bench, records metric)
  5. CORRECTNESS (sanity) → actions/correctness.md (must pass on clean build)
  6. PROFILE              → actions/profile.md (rocprof or torch.profiler)
  7. KB WARM-UP           → query kb/entries.jsonl for prior runs on this repo
  8. BUILD ACTION STACK   → score candidates per detected project_class

  9. DFS LOOP:
     WHILE action_stack not empty AND NOT stopping_criteria_met():
       a. Pop highest-scored action
       b. Execute it (env-var, compile-flag, kernel-opt, etc.)
       c. REBUILD if needed → actions/build.md
       d. CORRECTNESS GATE → actions/correctness.md
          - If tests fail: REVERT, mark INVALID, do not record metric
       e. MEASURE          → actions/baseline.md (re-run benchmark)
       f. KEEP/REVERT decision based on metric delta
       g. RE-SCORE remaining actions
       h. Push any new sub-actions discovered (e.g. profile reveals new hot kernel)

 10. REPORT               → actions/report.md
 11. KB INGEST            → write learnings to kb/entries.jsonl
```

## Stopping Criteria

Stop the DFS loop when ANY of these is true:

- Action stack is empty AND no new candidates exist
- 3 consecutive discards (likely plateaued)
- Wall-clock budget exceeded (`$TIME_BUDGET_MIN`, default 90)
- Cumulative gain >= 25% (good enough, write report)
- All remaining action scores < 1.0

## Correctness Gate Protocol (CRITICAL)

After EVERY change, before recording metric improvement:

1. Trigger `actions/correctness.md` with the detected `test_command`.
2. If `correctness_mode == "tests"`: tests must pass (exit 0, all pass).
3. If `correctness_mode == "golden-output"`: bench output must match the golden
   file within tolerance saved in `$RESULT_DIR/golden.json`.
4. If `correctness_mode == "none"`: WARN the user once, then proceed using only
   benchmark "did it run cleanly" as the gate. Annotate the report.

A change that improves the metric but fails correctness is ALWAYS reverted and
recorded as INVALID, regardless of speedup magnitude.

## Kernel Optimization Eligibility

Trigger `actions/kernel-opt.md` for a kernel when:

- It accounts for ≥ 3% of total GPU time in the profile
- Its source is locatable in the repo (not vendor BLAS / hipBLASLt / hipFFT)
- It is one of: HIP `__global__`, Triton `@triton.jit`, or torch.compile-generated
  Inductor Triton

Do NOT submit:
- hipBLASLt / hipBLAS / rocBLAS GEMMs (vendor-tuned)
- RCCL / NCCL communication kernels
- aiter / Composable Kernel attention (vendor-tuned)

## Notes for the Agent

- **Never rewrite the user's code path silently.** Every code patch must be saved
  to `$RESULT_DIR/patches/<NN>-<short-name>.patch` so it can be applied to a
  clean checkout.
- **Always rebuild from scratch when changing compile flags.** Set
  `CMAKE_BUILD_DIR=$RESULT_DIR/build-<attempt>` to avoid cache contamination.
- **Prefer env-vars first** — they are zero-cost to revert and often produce
  the largest single gains on AMD (e.g. `HSA_ENABLE_SDMA`,
  `GPU_MAX_HW_QUEUES`, `HIP_FORCE_DEV_KERNARG`).
- **If detect.md cannot find a benchmark or test command**, ask the user once.
  Do not invent commands.
