<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# Risk lenses

Consulted from Step 7 of [`../SKILL.md`](../SKILL.md); each lens is one line in `risks.txt`. An
upstream PR is reviewed and measured on its author's platform. These are the ways a change that is
right there turns out wrong, slower or inert on a ROCm deployment, and the author's platform hides
them; what the change does, which paths reach it and whether it is in the tree are Steps 1 to 4.
`Seen in` cites the public upstream PR the lens was learned from.

## L1 — Parity with the CUDA path

**Look at** changes in a CUDA-only file, under a platform guard, or in code with a ROCm twin: a HIP
kernel, a ROCm attention or MoE backend, a `CustomOp`'s `forward_hip`, a platform class.
**RISK** when the CUDA side changes behaviour or an interface the twin also implements and the diff
leaves the twin alone, or a new platform check names CUDA and not ROCm: the ROCm path keeps the old
bug, diverges in output, or fails at import or dispatch. **CLEAR** when the twin changes in the same
diff, or the changed code has no ROCm counterpart (cite where the ROCm path goes instead).

**Seen in:** vllm-project/vllm#34543 -- #32993 added UVA CPU weight offloading behind a platform
check for CUDA and XPU only, so ROCm raised `ValueError` although its hipified kernel worked.
vllm-project/vllm#57425 -- the AMD twin of a `CustomOp` defined only `forward_native`; ROCm routes
an enabled op to `forward_hip`, which falls back to `forward_cuda`, so serving crashed at the op's
first forward.

## L2 — Weights and quantization

**Look at** weight loading, padding, layout transforms and shuffles done at load, scale handling,
and the quantization method the target's checkpoint selects. **RISK** when a load-time layout,
padding or scale change reaches that method, above all where the ROCm path pads, preshuffles or
re-lays out weights for its own kernels, or where a padding rule meets the target's sharding.
**CLEAR** when the changed method is not the one the checkpoint selects (cite the selection).

**Seen in:** vllm-project/vllm#56359 -- the ROCm branch of the MXFP4 MoE round-up padded each
tensor-parallel shard's intermediate size from 384 to 512: 33% more expert weights, about 30 GiB
per GPU at TP8, taken from the KV cache. With expert parallelism the shard was already aligned.

## L3 — Numeric formats and bounds

**Look at** FP8 dtypes, accumulation dtypes, epsilons, clamps, scale ranges, and the integer width
of offsets and indices. **RISK** when the change hardcodes an encoding the arch does not run
(gfx942 runs FP8 `e4m3fnuz`, gfx950 the OCP `e4m3fn`; their maximum values and NaN encodings
differ), or grows a shape until an offset overflows. **CLEAR** when the dtype comes from the
platform (`current_platform.fp8_dtype()` or the like) and the range holds on every arch, or the
diff handles the variant (cite where).

**Seen in:** vllm-project/vllm#46730 -- a fused RoPE and quantization kernel hardcoded Q to
`e4m3fn` while K took the platform type, `e4m3fnuz` on gfx942, so the FP8 logits kernel fell back
to a mixed-dtype path on every call; matching them made mean TTFT 2.6x to 7.4x shorter.
vllm-project/vllm#39953 -- block indices and slot offsets in Triton kernels had to become int64 to
stay clear of int32 overflow on large KV caches.

## L4 — Collectives and parallelism

**Look at** all-reduce, all-gather, reduce-scatter and all-to-all calls, custom communication
kernels and the checks that enable them, collectives under graph capture, and the parallel sizes
the change assumes. **RISK** when it changes which communication backend runs, or the count or
order of collectives per step, or assumes a topology, an interconnect, a library feature, a
capability ordering or a world size the target lacks. **CLEAR** when no collective or group
changes, or the change is backend-neutral and the target's parallel layout runs it unchanged.

**Seen in:** vllm-project/vllm#16010 -- custom all-reduce on ROCm was enabled by comparing device
capabilities, which on ROCm are not sequential; the check moved into the platform class, which
names the gfx94 architectures.

## L5 — Tests, and whether ROCm CI ran them

**Look at** the changed tests in `files.txt`, their skip conditions and platform markers, and
`checks.json`. **RISK** when the new tests skip on ROCm or are marked for one platform, or no ROCm
job ran at the head: the PR's green says nothing about the target. **CLEAR** when a ROCm job in
`checks.json` ran the changed tests at the head and passed (name the job).

**Seen in:** vllm-project/vllm#48654 -- a test #48390 added passed on CUDA, where the path it drives
resolves to `F.linear`, and failed on AMD CI, where the same path dispatches to a ROCm GEMM op with
no CPU kernel; the fix merged the day after #48390.

## L6 — Dependency pins

**Look at** the requirement, lock and build files the diff changes or relies on, version checks in
the code, and the versions the target's image pins: a ROCm build pins its own torch, triton and
kernel libraries, often in a file of its own. **RISK** when the change needs a version or package
the target's image lacks, or moves a pin on the CUDA side only. **CLEAR** when the target's image
carries what the change needs (cite the pin), or the diff moves the ROCm pin with the CUDA one.

**Seen in:** vllm-project/vllm#48155 -- torch 2.13.0 and triton 3.7.1 replaced the CUDA and CPU
pins while the ROCm build requirements and Dockerfile were left out of scope.
vllm-project/vllm#47072 -- a test dependency declared only in the CUDA test requirements failed
ROCm CI at collection with `ModuleNotFoundError`.
