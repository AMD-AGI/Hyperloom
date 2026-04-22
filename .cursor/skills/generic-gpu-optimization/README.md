# Generic GPU Optimization Skill

A point-and-shoot performance optimization skill for any GPU codebase on AMD ROCm.
Unlike `inference-optimization` and `training-optimization`, this skill makes no
assumption about the framework. It auto-detects what your repo is and runs the
Hyperloom DFS optimization loop against it.

## What it can handle

| Project shape | Detected as | Build | Bench | Tests |
|---|---|---|---|---|
| C++/HIP library with CMake + Google Benchmark | `hip-cmake-bench` | `cmake --build` | `bench --benchmark_format=json` | `ctest` |
| C++/HIP library with CMake + Catch2 | `hip-cmake-bench` | `cmake --build` | parsed from `[[bench]]` tag | `ctest` |
| Standalone PyTorch script with `if __name__ == "__main__"` and timing | `pytorch-script` | `pip install -e .` (if pyproject) | direct `python` invocation | `pytest` (if `tests/`) |
| Triton kernel collection with `bench.py` / `benchmark.py` | `triton-collection` | `pip install -e .` | `python bench.py` | `pytest tests/` |
| HPC application with `--bench` mode | `hpc-app` | `make` or `cmake` | `./app --bench` | optional |

If detection comes back `unknown` the agent will ask you for the build, bench,
and (optional) test commands once, then proceed.

## Quick Start

### Bootstrap the skill into your target repo

From the Hyperloom root:

```bash
./scripts/optimize_repo.sh /path/to/your/gpu/repo
```

This copies the skill, the `mcp.json`, and an `.env.template` into
`<your-repo>/.cursor/`. Edit `<your-repo>/.env` to set `AK_YOUR_API_KEY`.

### Run from Cursor

Open `<your-repo>` in Cursor with the GEAK + OOB MCPs enabled, then:

```
@.cursor/skills/generic-gpu-optimization/SKILL.md
Optimize this repo for MI355X.
```

The agent runs setup → detect → build → baseline → correctness → profile → loop.

### Override what the agent picks up

```
@.cursor/skills/generic-gpu-optimization/SKILL.md
Optimize this repo.
BENCH_COMMAND: ./build/bench/MATRIX_BENCH --benchmark_filter=select_k --benchmark_format=json
TEST_COMMAND: ctest -R MATRIX_TEST --output-on-failure
GPU: MI300X
TIME_BUDGET_MIN: 60
Try at least env-vars and one kernel rewrite.
```

Anything you specify wins over auto-detection.

## Worked Example: hipRAFT

```
@.cursor/skills/generic-gpu-optimization/SKILL.md
Optimize hipRAFT select_k on MI300X.

REPO_ROOT: /home/gpinkert/projects/rocm-ds/hipRaft
BENCH_TARGET: select_k
GPU: MI300X (gfx942)
```

Detect will find `cpp/CMakeLists.txt`, `cpp/bench/prims/matrix/select_k.cu`,
`cpp/tests/`, and classify the project as `hip-cmake-bench`. It will build via
`./build.sh bench-prims`, run
`./cpp/build/release/bench/prims/MATRIX_BENCH --benchmark_filter=select_k
--benchmark_format=json` for the metric, and gate every change with
`ctest -R MATRIX_TEST`.

## What the agent tries

| Category | Examples |
|---|---|
| Env vars | `HSA_ENABLE_SDMA`, `GPU_MAX_HW_QUEUES`, `HIP_FORCE_DEV_KERNARG`, `HSA_FORCE_FINE_GRAIN_PCIE`, `HIP_LAUNCH_BLOCKING=0` |
| Compile flags | `-O3` vs `-O2`, `-ffast-math`, `--offload-arch=gfx950`, `-mllvm -amdgpu-early-inline-all`, `-DNDEBUG`, `-DCMAKE_HIP_ARCHITECTURES=…` |
| Kernel rewrites (GEAK) | Hot HIP `__global__` or Triton kernel sent to GEAK MCP, validated against tests |
| Code patches | Targeted edits suggested by profile (e.g. fuse two passes, change launch config) |
| torch.compile | If `pytorch-script` and the workload is amenable |

## Output

- `$RESULT_DIR/results.tsv` — every attempt with metric, status, description
- `$RESULT_DIR/optimization_report.md` — final report with what worked and why
- `$RESULT_DIR/patches/` — all kept code patches, applicable to clean checkout
- `$RESULT_DIR/kept_env.sh` — sourceable file with all kept env vars
- `$RESULT_DIR/kept_cmake.txt` — list of kept CMake/compile flag overrides

## Limitations

- TraceLens agentic analysis only runs if the project has a PyTorch trace. For
  C++/HIP projects the agent uses `rocprofv3` and parses kernel times directly.
- If your benchmark prints results in a non-standard format, you may need to
  point the agent at the metric line (`BENCH_METRIC_REGEX`).
- Multi-GPU correctness gating relies on the project's existing test suite; the
  skill does not invent distributed correctness tests.
