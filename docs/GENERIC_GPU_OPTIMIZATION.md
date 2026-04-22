# Generic GPU Optimization

Hyperloom's `generic-gpu-optimization` skill lets you point the agent at any
GPU codebase — not just inference servers (SGLang/vLLM) and not just
Primus/Megatron training. It auto-detects the project shape and runs the same
DFS Think → Try → Measure → Decide loop that powers the other skills.

## Design

The skill is built around three observations:

1. **The DFS orchestrator is framework-agnostic.** `SKILL.md` only needs three
   abstract operations: *build*, *measure*, *test*. Whether "build" means
   `cmake --build` or `pip install -e .` is just a string.
2. **Almost any GPU repo provides those three operations**, even if the
   commands vary. A heuristic detector is enough to find them in 90% of cases;
   for the rest the user supplies them once.
3. **GEAK and the OOB Agent MCPs don't care about the surrounding framework.**
   Given a kernel source file, target arch, and sample shapes, they produce a
   rewrite. The skill just needs to extract the kernel and splice the result back.

## What the skill detects

Run by `scripts/detect_project.sh`:

| Signal | Implies |
|---|---|
| `CMakeLists.txt` + `*.hip`/`*.cu` + `benchmark::benchmark` link | `hip-cmake-bench` |
| `pyproject.toml` + `import torch` + `bench*.py` | `pytorch-script` |
| `@triton.jit` + `bench*.py` or `benchmarks/` | `triton-collection` |
| `Makefile` + `*.hip` + `--bench`/`-b` flag in main | `hpc-app` |
| `CTestTestfile.cmake` | `tests` correctness mode + `ctest` |
| `tests/test_*.py` or pyproject pytest section | `tests` correctness mode + `pytest` |
| neither of the above | `golden-output` (capture first bench output) or `none` |

Detection writes everything to `$RESULT_DIR/detected.env`. Subsequent actions
just `source` it.

## Action repertoire

Five action types are scored by the heuristic and explored DFS-style:

| Action | Cost | Risk | Typical gain |
|---|---|---|---|
| `env-vars`      | seconds        | low (correctness can fail on fine-grain PCIe) | 1-8% |
| `compile-flags` | minutes (rebuild) | medium (`-ffast-math` can break things) | 2-15% |
| `kernel-opt` (GEAK) | 10-30 min/kernel | medium (output validated by tests) | 5-50% |
| `patch` (manual code change) | minutes | medium-high | varies |
| `torch.compile` (pytorch-script only) | minutes | low | 5-15% |

Initial priors are set by `project_class`; gains/losses re-score after each
attempt (success boosts similar actions ×1.5, failure penalizes ×0.5).

## Correctness gating

Every kept change must pass the project's own test suite. The skill never
invents tests — it only runs what's already there:

| Detected | Gate |
|---|---|
| `CTestTestfile.cmake` exists | `ctest --output-on-failure` |
| `tests/test_*.py` exists | `pytest -x` |
| neither | Capture baseline bench output as golden, compare numerical fields |
| nothing parseable | Warn user; accept on bench exit-code only |

The gate is mandatory. A 10x speedup that fails tests is reverted, recorded as
INVALID, and the agent moves on.

## Profiling

| Stack | Tool | Output |
|---|---|---|
| `pytorch-script` / `triton-collection` | `torch.profiler` (Chrome trace) | `profiles/torch_trace.json` |
| `hip-cmake-bench` / `hpc-app` | `rocprofv3` (preferred), `rocprof` v2 fallback | `profiles/rocprof.json` |

Both schemas are normalized to `{KernelName, DurationNs}` for the kernel
selector, so `kernel-opt.md` doesn't need to know which profiler ran.

## Bootstrap flow

```
[user] ./scripts/optimize_repo.sh /path/to/repo
       ├── copies skill to <repo>/.cursor/skills/generic-gpu-optimization/
       ├── copies mcp.json
       ├── copies .env.template
       └── runs detect_project.sh, prints summary

[user] cd /path/to/repo  &&  cp .env.template .env  &&  edit AK_YOUR_API_KEY

[user] open repo in Cursor; in chat:
       @.cursor/skills/generic-gpu-optimization/SKILL.md
       Optimize this repo for MI300X.

[agent] setup -> detect -> build -> baseline -> correctness -> profile
        -> DFS loop (env-vars, compile-flags, kernel-opt) -> report
```

## Reproducing a run

Every run produces:
- `$RESULT_DIR/results.tsv`
- `$RESULT_DIR/optimization_report.md`
- `$RESULT_DIR/kept_env.sh`
- `$RESULT_DIR/kept_cmake.txt`
- `$RESULT_DIR/patches/*.patch`

Reproduction on a clean checkout:
```bash
git checkout $BASELINE_SHA
git apply $RESULT_DIR/patches/*.patch
$(cat $RESULT_DIR/kept_cmake.txt | tr '\n' ' ')   # rebuild with kept flags
source $RESULT_DIR/kept_env.sh
$BENCH_COMMAND
```

## Limitations

- **No TraceLens for native code.** TraceLens consumes torch.profiler traces;
  it doesn't analyze rocprof JSON. For C++/HIP projects you get the kernel
  breakdown but not the full agentic analysis pass.
- **Multi-GPU correctness is project-defined.** The skill runs whatever the
  project's tests run; it doesn't synthesize distributed correctness checks.
- **Build isolation per attempt is per-CMake-cache.** If your project has
  side-effecting build artifacts outside the build dir (e.g. generated headers
  in `src/`), you'll need a manual `make clean` between attempts.
- **Kernel splicing is heuristic.** GEAK output replaces the function body
  textually. For complex multi-function kernels you may need to apply patches
  manually.

## Comparison with the other skills

| Concern | inference-optimization | training-optimization | generic-gpu-optimization |
|---|---|---|---|
| Server lifecycle | Yes (SGLang/vLLM) | No | No |
| YAML config parsing | No | Yes (Primus) | No |
| GBS gate | No | Yes (immutable global batch) | No (uses test suite) |
| Build system | n/a (pip) | n/a (pip) | Detected: cmake/pip/make |
| Profiler | torch.profiler + TraceLens | torch.profiler + TraceLens | torch.profiler OR rocprofv3 |
| GEAK kernel-opt | Yes | Yes | Yes |
| Initial priors | model-class table | model-class table | project-class table |
| Stopping criteria | identical | identical | identical |
