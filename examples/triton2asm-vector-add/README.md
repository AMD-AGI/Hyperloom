<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Minimal Triton compiler-to-assembly campaign

This FP32 vector addition starts from an ordinary Triton `kernel[grid](...)`
launch. Forge captures its AMD compiler `amdgcn` output, rebuilds the selected
`.s`, and retains Triton's argument marshaller, constexprs and current stream.
The source implementation remains the incumbent. No FlyDSL or handwritten
assembly is involved.

Use ROCm PyTorch, Triton with the AMD backend, and ROCm LLVM under
`$ROCM_PATH/llvm/bin` (default `/opt/rocm/llvm/bin`).

```bash
cp -a examples/triton2asm-vector-add /tmp/forge-triton-add
cd /tmp/forge-triton-add
git init
git add .
git commit -m 'Triton source and independent oracle'
kernelforge forge-loop \
  --workspace "$PWD" --kernel "$PWD/kernel.py" --driver "$PWD/driver.py" \
  --kernel-backend assembly --gpu-target gfx950 \
  --program-md-file "$PWD/program.md" --max-hours 1 --git-branch asm-triton-add
```

Preparation creates `kernel.s` and its specialization manifest, verifies an
assembler-error and a wrong-output control, and measures the unchanged source
and rebuilt assembly. Only `.s` is editable in the subsequent search. Correctness
requires exact addition, input preservation, nondefault-stream execution, graph
replay on changed inputs and repeated numerical evidence.

The same adapter accepts a direct Gluon JIT kernel. Autotuners and heuristics must
first be resolved into one fixed JIT specialization. A new dtype, constexpr or
compilation option requires a separate campaign. The finite GPU test covers both
frontends, instruction-error rejection and clean patch replay:

```bash
pytest -q src/kernelforge/tests/test_assembly_triton_gpu.py
```

This example verifies execution; it does not promise a performance improvement.
