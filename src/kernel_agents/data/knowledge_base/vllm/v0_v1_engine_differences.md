# V1 engine internals (V0 is gone on this checkout)

**Source checkout:** `vllm-amd` (vllm main,
May 2026). Title kept for legacy reasons — see below for the comparison
this file used to promise.

## TL;DR — V0 is a 7-line alias to V1

`vllm/engine/llm_engine.py` (complete file):

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine

LLMEngine = V1LLMEngine  # type: ignore
"""The `LLMEngine` class is an alias of [vllm.v1.engine.llm_engine.LLMEngine][]."""
```

Same shape for `vllm/engine/async_llm_engine.py` — aliases
`vllm.v1.engine.async_llm.AsyncLLM`. **Setting `VLLM_USE_V1=0` does
nothing** on this tree; there is no V0 code path to fall back to.

If older docs mention V0/V1 toggles, treat them as historical. Every
runtime fact below is V1.

## Where V1 things live

| Concern | File | Symbol / line |
| --- | --- | --- |
| Sync engine entry | `vllm/v1/engine/llm_engine.py` | `LLMEngine.start_profile` L327, `stop_profile` L330 |
| Async engine entry | `vllm/v1/engine/async_llm.py` | `AsyncLLM.start_profile` L876, `stop_profile` L882 |
| Engine core subprocess | `vllm/v1/engine/core.py` | `EngineCore.profile` L584 |
| GPU worker process | `vllm/v1/worker/gpu_worker.py` | `GPUWorker.profile` L874, lazy profiler factory L897-911 |
| Model runner | `vllm/v1/worker/gpu_model_runner.py` | 7174 LOC; `CudagraphDispatcher` plumbed in L787 |
| Cudagraph dispatcher | `vllm/v1/cudagraph_dispatcher.py` | `dispatch` L234, `initialize_cudagraph_keys` L165 |
| HTTP profile router | `vllm/entrypoints/serve/profile/api_router.py` | `/start_profile` L21, `/stop_profile` L29 |

Runtime topology: HTTP -> AsyncLLM (async front) -> EngineCoreClient
(IPC) -> EngineCore subprocess(es) -> GPUWorker (one OS process per TP
rank). The torch profiler **lives in each GPUWorker process**, not in
the engine — each worker writes
`dp{dp}_pp{pp}_tp{tp}_*.pt.trace.json.gz`.

## CUDAGraphMode (V1's cudagraph state machine)

Defined as an enum where the value is either a scalar or a 2-tuple
(`vllm/config/compilation.py:53-103`):

```python
class CUDAGraphMode(enum.Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = (FULL, NONE)
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)

    def decode_mode(self) -> "CUDAGraphMode":
        return CUDAGraphMode(self.value[0]) if self.separate_routine() else self
    def mixed_mode(self) -> "CUDAGraphMode":
        return CUDAGraphMode(self.value[1]) if self.separate_routine() else self
    def separate_routine(self) -> bool:
        return isinstance(self.value, tuple)
```

Key: 2-tuple modes mean "use `value[0]` for decode-only batches, `value[1]`
for mixed prefill+decode". The V1 production default is
`FULL_AND_PIECEWISE` — full graph for uniform decode, piecewise for
prefill (see `compilation.py:589` "(v1 default)").

The 5 modes (`vllm/config/compilation.py:583-606`):

| Mode | Meaning |
| --- | --- |
| `NONE` | Pure eager; every kernel launch is visible. |
| `PIECEWISE` | Capture inductor partitions only. |
| `FULL` | Single graph for the whole forward (no piecewise). |
| `FULL_DECODE_ONLY = (FULL, NONE)` | Full graph for uniform-decode batches; eager otherwise. |
| `FULL_AND_PIECEWISE = (FULL, PIECEWISE)` | V1 production default. |

`cudagraph_mode = None` at declaration (compilation.py:581) gets filled
in by `__post_init__` based on platform + backend support; it can be
forced to `NONE` for incompatible features (e.g. DeepEP high-throughput
disables capture at compilation.py:1187).

## Cudagraph dispatch logic

`CudagraphDispatcher.dispatch` (cudagraph_dispatcher.py:234-323)
returns `(CUDAGraphMode, BatchDescriptor)` per step:

```python
# Pseudo-code; see cudagraph_dispatcher.py:273-323
def dispatch(num_tokens, uniform_decode, has_lora, num_active_loras, ...):
    if not keys_initialized or cudagraph_mode == NONE or num_tokens > max_size:
        return NONE, BatchDescriptor(num_tokens)
    batch_desc = _create_padded_batch_descriptor(...)
    if FULL in allowed_modes and batch_desc in cudagraph_keys[FULL]:
        return FULL, batch_desc                              # uniform decode hit
    if PIECEWISE in allowed_modes:
        relaxed = replace(batch_desc, num_reqs=None, uniform=False)
        if relaxed in cudagraph_keys[PIECEWISE]:
            return PIECEWISE, relaxed                        # prefill or mixed
    return NONE, BatchDescriptor(num_tokens)                 # fallback eager
```

`uniform_decode` means every sequence in the batch contributes exactly
`uniform_decode_query_len` tokens (1 by default; >1 for spec decode).

**Profiling trap.** Trace interpretation depends on which mode ran:
- `FULL_AND_PIECEWISE` (default): per-kernel `ts` = real GPU compute.
  Rank-0 trace ~30-130 MB.
- `NONE` (cgnone): every launch shows 5-30 us injected overhead per
  kernel. Trace 600+ MB. The "AR is 77% of decode" claim came from a
  cgnone trace; real share is <5%. **Always state which mode produced
  a number.**

## Profiler plumbing

`/start_profile` HTTP route (`api_router.py:21-26`):

```python
@router.post("/start_profile")
async def start_profile(raw_request: Request):
    logger.info("Starting profiler...")
    await engine_client(raw_request).start_profile()
    logger.info("Profiler started.")
    return Response(status_code=200)
```

Chain: HTTP -> `AsyncLLM.start_profile` (async_llm.py:876)
-> `engine_core.profile_async(True)` -> each `GPUWorker.profile(True)`
(gpu_worker.py:874-920), which **lazily** builds either
`TorchProfilerWrapper` or `CudaProfilerWrapper` (gpu_worker.py:897-911,
from `vllm/profiler/wrapper.py`). Profiler **type** is validated at
worker init (gpu_worker.py:151):

```python
if self.profiler_config.profiler not in ("torch", "cuda", None):
    raise ValueError(f"Unknown profiler type: {self.profiler_config.profiler}")
```

So `--profiler-config.profiler=torch` must be set at server launch —
unknown values raise on import, not on request.

## ROCm-specific V1 quirks

- **Aux CUDA streams disabled on ROCm** in DeepSeek V4 to avoid hangs
  (deepseek_v4.py:1249-1253):
  ```python
  aux_stream_list = (
      None if current_platform.is_rocm()
      else [torch.cuda.Stream() for _ in range(3)]
  )
  ```
  `attn_gemm_parallel_execute` then runs serially on ROCm — the three
  light input GEMMs (compressor kv_score, indexer.weights_proj,
  indexer.compressor kv_score) lose overlap with `fused_wqa_wkv`.
- `PROFILER_WITH_STACK=False` recommended to avoid V1 trace-flush
  timeouts at TP=8.
- `cudagraph_mode` accepts the **string form** because
  `validate_cudagraph_mode_before` (compilation.py:838-845) is a
  `@field_validator(mode="before")` — config files can write
  `"FULL_AND_PIECEWISE"`.

## Per-TP-rank JIT note

V1 spawns one OS process per TP rank. Any
`torch.utils.cpp_extension.load(...)` runs **independently in each
worker** after model load, serially. The first inference call
post-restart can time out while workers still build. Always warm up
with a tiny bench between server-up and timed profile.

## Not investigated

- The pre-alias V0 code (no longer in the tree).
- LoRA cudagraph specialization detail
  (`captured_lora_counts`, cudagraph_dispatcher.py:283-300).
- `EngineCoreClient` IPC encoding (zmq layer; not on the hot path).
