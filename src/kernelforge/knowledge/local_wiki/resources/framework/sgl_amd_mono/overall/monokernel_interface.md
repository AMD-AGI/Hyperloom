---
title: the monokernel interface — MonoSpec, routing, numerics and refusal
kind: technique
gens: [gfx950]
updated: 2026-11-09
sources:
  - sgl-project/sglang@035670b6:python/sglang/kernels/plugin_kernels/amd/spec.py
  - sgl-project/sgl-amd-mono@219f40be:sgl_amd_mono/minimax_m31/spec.py
  - sgl-project/sgl-amd-mono@219f40be:test/unit/test_contract.py
  - https://github.com/sgl-project/sglang/issues/43949
---

# The monokernel interface (RFC sgl-project/sglang#43949, API version 1)

A spec is the contract between a model's launches and SGLang. In a forge campaign it is the task,
not the candidate: the launch and its forward change, the declared contract does not.

## `MonoSpec` fields
| Field | Meaning |
|---|---|
| `name` | the spec's name, e.g. `"minimax_m31_ffn_hip"` |
| `devices` | ISA → `DeviceLimits(widths, tp_sizes, resident_blocks)`: step rows served (ascending), TP sizes, and the launch grid (one workgroup a CU) |
| `features` | the `Feature`s the spec accepts; every other one (pipeline parallel, DP attention, EPLB, LoRA, torch compile, …) is refused by default |
| `requires` | `Requirements(dtypes, attention_backends, server_args, hf_config, env)`, checked alike for every spec at startup; an empty field accepts any value |
| `numerics` | `Numerics(kind, bounds)` against the standard path on the same GPU |
| `model_disable_reasons` | `(model, hf_config, copies) -> list[str]` after load |
| `build` | `(model, hf_config, BuildContext) -> BuiltMonokernels` |
| `api_version` | the `MONOKERNEL_API_VERSION` the spec is written against |
| `config_disable_reasons` | `(cfg, hf_config) -> list[str]` before load: what `requires` cannot state |
| `load_tensor` | `(copies, model, name, tensor)`: keep a checkpoint tensor before SGLang converts it |
| `tokens_per_request`, `unread_metadata` | draft block sizes; step metadata the forward never reads |

`BuildContext(isa, widths, deadline_s, copies)`: `deadline_s` is every wait's deadline, below
`--watchdog-timeout`; `copies` holds what `load_tensor` kept, and `build` adds its own.

## Numerics
- `NumericsKind.BITWISE` or `NumericsKind.TOLERANCE`. Bounds are named `<launch>.<output>.<metric>`
  with metric `max_abs`, `rel` or `cosine`; a `max_abs` bound of 0 requires bitwise equality, and a
  `TOLERANCE` spec without bounds is refused.
- The bounds come from check mode: `SGLANG_AMD_MONO_CHECK=1` runs SGLang's layer on a copy of each
  covered layer's input in eager steps and logs both outputs' differences. MiniMax-M3.1-Flash's bounds
  record two such runs (49,476 and 44,004 launches) next to the worst case they saw.
- Upstream perf PRs that change no value claim — and prove — bitwise equality with `main` at every
  width; that is a stronger result than staying inside `NUMERICS`, and the usual one for a hand-off or
  placement change.

## What `forward` must do
`BuiltMonokernels.forward(plan, *args, **kwargs)` takes the model forward's arguments and returns what
it returns. It writes the KV cache and auxiliary hidden states the standard path writes for every row,
padded rows included, and it is capturable in a CUDA graph: no host sync, shapes fixed by `plan.key`.

## Refuse, do not fall back
A configuration the launches cannot serve is refused before the server starts, with every reason
listed (`config_disable_reasons`, `model_disable_reasons`, `requires`, `features`). Inside a routed
step there is no fallback: a failed wait stops the launch on every rank (`overall/device_runtime.md`),
so no token of a failed step is returned. A candidate that adds a silent fallback, a host-side retry
or a host sync inside a routed step breaks the contract even when every test passes.

## `SGLANG_READS`
Every module that imports from `sglang` lists what it reads, by file under the `sglang` package:
```python
SGLANG_READS = {
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": ("MonoSpec", "Numerics", ...),
    "srt/models/minimax_m3.py": ("is_layer_sparse", "o_proj", ...),
}
```
A key names a file's locations newest first (`"new.py|old.py"`); the first one the installed SGLang has
is checked. `test/unit/test_contract.py` checks each name statically (AST, no import) against the
installed SGLang, so a rename fails on CPU instead of at a GPU server's startup. A candidate that reads a
new SGLang name declares it in the same change.
