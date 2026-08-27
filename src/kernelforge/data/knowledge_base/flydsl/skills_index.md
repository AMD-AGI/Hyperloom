# FlyDSL `.claude/skills/` Index

> The upstream repo ships its own `.claude/skills/*/SKILL.md` pack — 16 skills
> covering kernel authoring, profiling, debugging, and workflow. Some are
> already distilled into this KB (`gemm_optimization.md`, `lds_optimization.md`,
> `pitfalls.md`, `kernel_patterns.md`); the rest are listed here for reference.
> Working copies live at `_tmp_flydsl_src/FlyDSL/.claude/skills/*/SKILL.md`.

## Authoring & Programming
| Skill | What it covers |
|---|---|
| `flydsl-kernel-authoring` | Comprehensive reference: layout algebra, tiled copy/MMA, buffer ops, range loops with `init=`, SmemAllocator, autotuning. **→ distilled in [dsl_api.md](dsl_api.md), [kernel_patterns.md](kernel_patterns.md)** |
| `flydsl-tile-programming` | Step-by-step wizard for new kernels: 8-step pattern recipe (elementwise / copy / GEMM / buffer-low-level). **→ distilled in [kernel_patterns.md](kernel_patterns.md)** |
| `port-to-layout-api` | Migrate raw `buffer_ops` code to the layout API (`make_buffer_tensor` + `logical_divide` + `copy_atom_call`) |
| `flydsl-internal-types-cleanup` | Replace direct `scf/arith/vector/memref` dialect calls with FlyDSL internal types (`fx.Int32`, `Vec`, etc.) while preserving correctness/perf |
| `add-target-atom-op` | Add a new target-specific MMA/Copy Op to a backend dialect (`lib/Dialect/FlyROCDL/<SUBTARGET>/`) |
| `prefetch-data-load` | Patterns for loop-carried prefetch and software pipelining |

## Optimization
| Skill | What it covers |
|---|---|
| `gemm-optimization` | Tiling, ping-pong LDS, XOR16 swizzle, prefetch pipeline, `hot_loop_scheduler`, MFMA inner loop, epilogue choice, VGPR budget, TFLOPS/bandwidth, ATT bottleneck matrix. **→ distilled in [gemm_optimization.md](gemm_optimization.md)** |
| `lds-optimization` | Bank-conflict diagnosis (gfx942/gfx950), XOR swizzle math, padding, write-read distance, CDNA4 `DS_READ_TR` transpose loads. **→ distilled in [lds_optimization.md](lds_optimization.md)** |

## Profiling & Trace Analysis
| Skill | What it covers |
|---|---|
| `capture-kernel-trace` | rocprofv3 ATT (Advanced Thread Trace) capture via Docker remote: input.yaml config, kernel name discovery |
| `kernel-trace-analysis` | Parse ATT traces with `hotspot_analyzer.py`; identify top-K stall hotspots; ships scripts in subskill `scripts/` |
| `bisect-perf-regression` | `git bisect` on perf: given known-good and known-bad commits, find the regression-introducing commit |

## Debug & Hygiene
| Skill | What it covers |
|---|---|
| `debug-flydsl-kernel` | Classify error (NaN / zeros / >50% / 1-5% / hang); cache invalidation, range vs range_constexpr, loop-carried state packing, buffer_load addressing, MFMA operand layout, LDS bank conflicts. **→ distilled in [pitfalls.md](pitfalls.md)** |
| `format-code` | Pre-commit cleanup: autoflake unused imports/vars, black for Python, clang-format Google style for C++ |
| `check-python-style` | Reproduce CI Python style check locally via `scripts/check_python_style.sh` (black 120-col + ruff `E/W/F/I`) |

## Infrastructure
| Skill | What it covers |
|---|---|
| `build-flydsl` | Build/install FlyDSL on a remote host or Docker container (LLVM + FlyDSL C++ + Python bindings, editable install) |
| `build-rocm-image` | SSH to remote and build Docker image with rocprofv3, vllm, aiter, FlyDSL, custom Triton |

## How to invoke (from the FlyDSL repo's `.claude` setup)

In a Claude Code session running inside the FlyDSL repo, these skills are
invocable as `/<skill-name>`. From this KB they are reference material — if
you need to run any of them, work from within the upstream repo or copy the
script bits referenced (e.g. `kernel-trace-analysis/scripts/hotspot_analyzer.py`).
