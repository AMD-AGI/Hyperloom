<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Triton to assembly: AttnRes score

This minimal example uses the existing `forge-loop --kernel-backend assembly`.
Its internal PORT phase replaces `kernel.py` with a standalone HIP launcher and
creates `kernel.s`. After host validation, the optimization loop can edit only
`kernel.s`. The independent FP64 oracle, launcher and attributed seed remain fixed.

Requirements: Linux, MI355X (`gfx950`), ROCm LLVM tools at
`/opt/rocm/llvm/bin`, ROCm PyTorch, Triton, and this Hyperloom revision installed.
Use a dedicated workspace and GPU. This example does not require vLLM or SGLang.

```bash
W=$(mktemp -d)
cp -a examples/triton2asm-attnres/. "$W/"
git -C "$W" init
git -C "$W" add .
git -C "$W" commit -m 'AttnRes source and independent driver'
python "$W/driver.py" --mode test
python "$W/driver.py" --mode bench
kernelforge forge-loop \
  --workspace "$W" --kernel "$W/kernel.py" --driver "$W/driver.py" \
  --kernel-backend assembly --gpu-target gfx950 \
  --program-md-file "$W/program.md" --max-hours 1 --git-branch attnres-asm
```

Configure your normal Forge agent provider and model before running the loop.
No additional command or new-file allowlist is needed. The PORT agent is told to
use `seed/score.s` and `seed/launcher.py`, so this example tests a known attributed
port, not autonomous discovery of Neha's implementation. It may be slower than
the original Triton. PORT acceptance requires correctness, full case coverage
and propagation of a deliberately injected assembler error; it requires no
speedup. Optimization uses Forge's normal repeated measurements and KEEP gate.

`forge_experiments/assembly_port/result.json` records the original and initial
ASM measurements, hashes, validation and port commit. Regular loop artifacts
record subsequent ASM candidates. The export base remains the original source
commit so that a clean checkout receives both the launcher and `.s`, even when
no later instruction edit is kept. Use the same command with `--resume` to
continue; changing the verified launcher requires a fresh campaign.

The optional GPU regression exercises real PORT validation without an LLM,
nondefault streams, graph replay with changed inputs, rejection of a no-op
assembly edit, build-error propagation and clean patch replay:

```bash
pytest -q src/kernelforge/tests/test_assembly_hip_gpu.py
```

## Contract and provenance

The fixed specialization is `T=64`, `NVB=8`, `H=7168`: BF16 prefix and bank,
FP32 weights, contiguous tensors, FP32 scores with 16 columns. Only the nine
active columns are written. For each vector `v`, the result is
`sum(v * weight) * rsqrt(mean(v * v) + 1e-6)`. The driver uses FP64 evaluation
with fixed `rtol=atol=3e-4`, checks untouched inputs/inactive columns, and tests
random, unit, near-zero and zero inputs. This is one specialization, not full
AttnRes or model-serving coverage.

The Triton source and ASM seed were extracted from
[AITER PR #4863](https://github.com/ROCm/aiter/pull/4863), commit
`1ddd136b0cf8ee9cd67afbb9b1e69202dc96ff68`:

- `op_tests/test_kimi_k3_attnres_asm.py`: `_score_kernel_ref`.
- `hsa/gfx950/attnres/kimik3_attnres_score_gfx950.s`: Neha's published score ASM.

The ASM instructions and metadata are unchanged. License/provenance comments
were added and the upstream headline performance comment removed to avoid
presenting it as a measurement of this example. The Forge launcher uses the
declared 128 bytes of static LDS, with zero additional dynamic LDS. The source
implements handwritten ASM replacing Triton; it is not a recovered Evolve agent
or a demonstration that Triton's compiler emitted this optimized file.

An earlier independent MI355X measurement found this score ASM at 5.302 us versus
2.892 us for Triton. That historical result is not a new run of this example.
See the [case card](../../src/kernelforge/data/local_knowledge/languages/assembly/cases/evolve_attnres_score_gfx950.md).
Measure again on the target machine; no performance or E2E gain is assumed.
