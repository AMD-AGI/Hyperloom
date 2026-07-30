# Generic — Hardware / GPU-generation facts (all frameworks)

## MXFP4 native acceleration needs gfx950 (MI350X / MI355X); gfx942 emulates it
- kind: hint
- source: cph-perf-tuning:KNOWLEDGE.md#0.2.1
- impact: correctness/enablement (prevents OOM / no-benefit quantization)
- accuracy_risk: stripping baked-in mxfp4 -> 4x memory -> OOM
- domain_tags: framework

Native FP4 acceleration needs gfx950 (MI350X / MI355X). On gfx942 (MI300X / MI325X) MXFP4 runs in
dequant-to-BF16 EMULATION (Marlin/Triton MoE path), so electively quantizing a higher-precision model
to MXFP4 on gfx942 buys no real memory/throughput benefit. BUT models shipped natively in MXFP4 (e.g.
gpt-oss-120b) MUST keep their mxfp4 quantization on gfx942 — it is how the model fits; stripping it or
switching to BF16 4x's memory and OOMs. There is also a gfx942+AITER startup-crash footgun on the FP4
path — see the vLLM correctness flags.

## GPU generations and HBM
- kind: hint
- source: cph-perf-tuning:conf/gpu/*.yaml
- domain_tags: freeform

MI300X = gfx942, 192 GB, 8 GPUs, MXFP4 emulated. MI325X = gfx942, MXFP4 emulated. MI350X = gfx950,
256 GB, native FP4. MI355X = gfx950, 288 GB, native FP4. Rule: gfx942 = emulated FP4; gfx950 = native
FP4.

## Host GPU driver is NOT isolated by the container
- kind: hint
- source: cph-perf-tuning:SKILL.md#field-notes
- domain_tags: systems

Kernels run against the HOST amdgpu/KFD driver, not a container-private one. A host ROCm generation
that differs from the recorded run (e.g. 7.2.3 vs 6.4.1) can make ±10% reproduce impossible from
userspace. Quantify what a newer image recovers and attribute residual gap to the driver, separately
from tuning wins.
