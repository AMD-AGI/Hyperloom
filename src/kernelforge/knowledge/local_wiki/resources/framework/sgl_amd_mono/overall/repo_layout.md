---
title: sgl-amd-mono — packages, how a step reaches a launch, test tiers
kind: technique
gens: [gfx950]
updated: 2026-11-09
sources:
  - sgl-project/sgl-amd-mono@219f40be:README.md
  - sgl-project/sgl-amd-mono@219f40be:pyproject.toml
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/__init__.py
  - sgl-project/sglang@035670b6:python/sglang/kernels/plugin_kernels/amd/spec.py
---

# sgl-amd-mono — repo layout

## How SGLang loads it
- `pyproject.toml` declares a general plugin: entry point group `sglang.srt.plugins`,
  `sgl_amd_mono = "sgl_amd_mono:register_specs"`. SGLang runs it in every process.
- `sgl_amd_mono/__init__.py::SPECS` maps `(HF architecture, model type or None)` to a
  `"module:SPEC"` path; `register_specs()` registers each under backend `"amd"`. A spec module is
  imported only when a server runs this backend, so a model's imports cost nothing elsewhere.
- Serving flag: `--enable-amd-monokernel` (`--monokernel-backend amd` on SGLang before
  sgl-project/sglang#45389). The interface module moved in that PR too: specs import
  `sglang.kernels.plugin_kernels.amd.spec` first and fall back to `sglang.srt.monokernels.spec`.

## What a model package holds
```
sgl_amd_mono/
├── common/              # the device runtime every launch shares (overall/device_runtime.md)
└── <model>/
    ├── spec.py          # the MonoSpec and the forward that replaces the model's on routed steps
    ├── draft.py         # a draft model's spec, when the model has one
    ├── check.py         # SGLANG_AMD_MONO_CHECK: each layer against SGLang's own, in eager steps
    └── <isa>/           # launches, layouts and the launch runner, per ISA (e.g. deepseek_v41/gfx950/)
```
Some models keep their launch at the model root instead of under an ISA folder
(`minimax_m31/ffn.py`, `a16w4.py`, `a16w8.py`, `runner.py`). `qwen38_flash_next/README.md` walks
its kernel stage by stage; read a model's README or launch docstring before its code.

## How a served step reaches a launch
1. SGLang keys every decode and speculative-verify step by its CUDA-graph key (`ShapeKey`:
   `forward_mode`, padded `batch_size`, `tokens_per_request`; `rows = batch_size × tokens_per_request`).
2. A step whose rows are in the spec's `DeviceLimits.widths` for this ISA runs
   `BuiltMonokernels.forward(plan, ...)` in place of the model's forward; every other step runs
   SGLang's standard path. The decision uses the key alone, so every rank decides alike and a
   captured graph replays the decision made at capture.
3. `forward` runs SGLang's modules where the spec does not cover the layer (embedding, attention on
   some models, final norm, logits) and one launch per covered span.

## The three test tiers
| Tier | Runs on | What it pins |
|---|---|---|
| `test/unit` | CPU | each spec's fields; `test_contract.py` (every `SGLANG_READS` name against the installed SGLang); `test_source_digest.py` (launch cache keys) |
| `test/kernels` | GPU | each launch against SGLang's reference ops on random weights; also the check and timing tools (`test/kernels/<model>/check_*.py`, `time_layer(s).py`) |
| `test/e2e` | GPU, checkpoint | serving accuracy (chat GSM8K past the largest width), check mode against `NUMERICS`, and fault injection: a stalled rank stops the server without sending tokens |

Most kernel and e2e tests need the model's TP size in GPUs; e2e tests also need the checkpoint path
in an environment variable (e.g. `MINIMAX_M31_FLASH_MODEL_PATH`). `README.md` § Testing lists the
command and GPU count per model. Many kernel tests build from random weights and need no checkpoint.

## Conventions a change must keep
- Every module that reads SGLang internals declares the names in `SGLANG_READS` (see
  `overall/monokernel_interface.md`).
- Files adapted from ROCm/ATOM or vLLM name their origin and commit in the header.
- `pre-commit` formats the tree; docstrings and comments say only what the code cannot show.
- Debug and test knobs already exist (`SGLANG_AMD_MONO_CHECK`, `SGLANG_AMD_MONO_FAULT_INJECT`,
  `SGLANG_AMD_MONO_TIMELINE`, `SGLANG_AMD_MONO_CACHE_DIR`, `SGLANG_AMD_MONO_ROCM_LLVM`). A served
  launch does not keep A/B switches once a choice is measured: upstream dropped Kimi-K3.1's wide
  launch's debug checks and A/B switches in #114.
