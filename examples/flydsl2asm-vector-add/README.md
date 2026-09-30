<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Minimal FlyDSL compiler-to-assembly campaign

This example adds two FP32 vectors of length 4103, including a masked final
block. It starts with an ordinary FlyDSL implementation and one explicit compile
call. Forge captures that compiler's `.s`, reconnects it to the original launcher,
then allows only instruction edits. No handwritten seed is supplied.

Use ROCm PyTorch, FlyDSL with the CompiledFunction API and ISA dumping, and
`llvm-mc`/`ld.lld` under `$ROCM_PATH/llvm/bin` (default `/opt/rocm/llvm/bin`).
Set the target to the exact architecture of the GPU, including target features.

```bash
cp -a examples/flydsl2asm-vector-add /tmp/forge-vector-add
cd /tmp/forge-vector-add
git init
git add .
git commit -m 'vector-add source and independent driver'
python driver.py --mode test
kernelforge forge-loop \
  --workspace "$PWD" --kernel "$PWD/kernel.py" --driver "$PWD/driver.py" \
  --kernel-backend assembly --gpu-target gfx950 \
  --program-md-file "$PWD/program.md" --max-hours 1 --git-branch asm-vector-add
```

The host exports `kernel.s` and `kernel.s.json` and changes only the compile-call
binding in `kernel.py`. The driver tests exact results against PyTorch, varied
inputs, input preservation, a nondefault stream and graph replay. `config.yaml`
is the canonical compile/correctness gate. Benchmarks run compilation and loading
outside timing. Preparation verifies a deliberate assembler failure and rejects
a no-op assembly negative control, ensuring the oracle tests the returned
candidate beyond FlyDSL compilation warmup.

The unmodified source remains the baseline. The example does not promise a
speedup: vector addition may already be limited by memory or launch overhead.
If no instruction candidate passes KEEP, Forge selects the source commit.

Run `pytest -q src/kernelforge/tests/test_assembly_campaign_gpu.py` from the
Hyperloom checkout for a finite, LLM-free capture/rebuild test, a deliberate
add-to-subtract wrong-result test, and clean patch replay. The test requires
gfx950 and skips when the GPU dependencies are unavailable.
