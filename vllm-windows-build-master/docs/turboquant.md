# Multi-TurboQuant KV cache compression

vLLM v0.27.1 on Windows ships with integrated support for ten KV cache
compression methods: six local
[Multi-TurboQuant](https://github.com/aivrar/multi-turboquant)
methods plus four upstream TurboQuant variants. This page focuses on
the six local methods:

| Method | Bits | Family | Calibration | Notes |
|---|---|---|---|---|
| `isoquant3` | 3.25 | quaternion 4D rotation | none | golden-ratio quaternion, no setup |
| `isoquant4` | 4.25 | quaternion 4D rotation | none | higher quality, less compression |
| `planarquant3` | 3.25 | Givens 2D rotation | none | simplest transform |
| `planarquant4` | 4.25 | Givens 2D rotation | none | higher quality |
| `turboquant25` | 2.25 | WHT + MSE codebook + QJL residual | runtime | most compression, lossiest |
| `turboquant35` | 3.25 | WHT + MSE codebook + QJL residual | runtime | balanced |

The methods are dispatched by the `kv_cache_dtype` argument when
constructing an `LLM`:

```python
from vllm import LLM
llm = LLM(
    model="path/to/Qwen3-14B-AWQ",
    kv_cache_dtype="isoquant3",   # or any of the 6 methods
    ...
)
```

The methods can also be selected via the OpenAI server's
`--kv-cache-dtype` command-line flag.

---

## How the integration works

vLLM's KV cache is a paged memory pool of shape `[num_blocks, 2,
block_size, num_kv_heads, head_size]`. The standard layout uses fp16
or bf16 elements. For TQ:

1. **Storage type**: the dtype is changed to `torch.uint8`. The cache
   buffer is now half the size in bytes.
2. **Cache write** (`do_kv_cache_update`): K and V are passed through
   the chosen method's vectorised encoder, which produces packed bytes
   of shape `[num_tokens, num_kv_heads, packed_dim]`. The packed bytes
   are scattered into the first `packed_dim` columns of each cache slot
   (the remaining bytes are unused but kept so the cache shape stays
   compatible with the standard attention kernel).
3. **Attention forward**: instead of running the standard Triton
   attention kernel directly on the packed cache, vLLM:
   - Identifies the unique blocks referenced by the current batch's
     `block_table`
   - Decodes only those blocks back to fp16 K, V tensors via the
     method's decoder
   - Builds a *compact* fp16 cache containing only the active blocks
   - Remaps `block_table` indices to the compact cache
   - Runs the standard `unified_attention` kernel on the compact cache

The persistent cache stays small (the memory savings stick across the
lifetime of the engine); the temporary fp16 buffer for each forward
call only contains the blocks for the current batch.

The wiring lives in `vllm/v1/attention/ops/multi_turboquant_kv.py`
and `vllm/v1/attention/backends/triton_attn.py`.

---

## Memory savings

Measured on Qwen3-14B AWQ at `head_size=128`, `num_kv_heads=8`,
`block_size=16`, `gpu_memory_utilization=0.5` on a 24 GB RTX 3090:

| KV dtype | Bytes per slot | KV cache tokens | Concurrency @ 512 ctx |
|---|---|---|---|
| `auto` (fp16) | 256 | 16,336 | 31.91x |
| `isoquant3` | 128 (uint8) | **32,672** | **63.94x** |
| `isoquant4` | 128 (uint8) | 32,672 | 63.94x |
| `planarquant3` | 128 (uint8) | 32,672 | 63.94x |
| `planarquant4` | 128 (uint8) | 32,672 | 63.94x |
| `turboquant25` | 128 (uint8) | 32,672 | 63.94x |
| `turboquant35` | 128 (uint8) | 32,672 | 63.94x |

**That's a clean 2× capacity gain at the same `gpu_memory_utilization`.**

The actual `packed_dim` for each method is smaller still (54-70 bytes
out of the 128 we allocate per slot — the remaining bytes are wasted
but kept for kernel layout compatibility). A future revision could
shrink the cache further by overriding `AttentionSpec.real_page_size_bytes`
to use `packed_dim` directly, taking the savings to ~75-80%.

---

## Quality

Each method introduces a different quantization noise pattern, which
shows up as different token sampling at temperature=0:

```
FP16 baseline:  Paris. What is the capital of Italy? The capital of Italy is Rome.
isoquant3:      Paris, and the capital of Canada is Ottawa.
isoquant4:      Paris. The capital of Italy is Rome. The capital of Spain is Madrid.
planarquant3:   Paris. The capital of France is Paris. The capital of France is...
planarquant4:   Paris. What is the capital of Italy? The answer is Rome.
turboquant25:   Paris. The capital of France is Paris. So, the capital of France...
turboquant35:   Paris. The capital of Italy is Rome. The capital of Spain is Madrid.
```

In general:
- The **3-bit / 2-bit** variants have visible noise — outputs may
  diverge from the baseline early.
- The **4-bit** variants (`isoquant4`, `planarquant4`) stay much closer
  to the baseline. They are the quality-first choices among the local TQ
  methods, but their current unfused fallback is intended for offline or
  memory-constrained workloads rather than latency-sensitive serving.
- `turboquant35` uses calibrated outlier dimensions (currently the
  fixed "first N dims as outliers" — calibrated metadata via
  `multi_turboquant.calibration` would improve quality further).

---

## Throughput cost

There's no free lunch: the encode and decode paths run in
PyTorch-vectorised mode (no fused Triton kernel). On a 14B AWQ model
with the default test settings:

| KV dtype | tok/s (single prompt, 20 tok) |
|---|---|
| `auto` (fp16) | ~37 |
| `isoquant3` | ~0.12 |
| `turboquant35` | ~1.0 |

The slow path is the per-cache-write encode and per-attention-step
decode. A fused Triton kernel would close the gap significantly. The
current code is correct and ready to optimise.

**This means TQ in this state is good for: long-context offline
inference, large batches at low QPS, scenarios where you'd otherwise
have to swap to a smaller model.** It is *not* yet suitable for
latency-sensitive serving — wait for the kernel optimisation.

---

## Calibration (optional)

`turboquant25` and `turboquant35` support calibrated outlier dimension
selection for better quality. To generate calibration metadata for a
model:

```python
from multi_turboquant.calibration.generate_metadata import generate_turboquant_metadata
import json

meta = generate_turboquant_metadata(
    "path/to/model",
    recipe="turbo3",
    verbose=True,
)
with open("path/to/model/turboquant_kv.json", "w") as f:
    json.dump(meta, f)
```

The current Windows integration uses fixed "first N dims as outliers"
indices and ignores `turboquant_kv.json` metadata. Plugging the
calibrated indices into `_get_fixed_group_indices` in
`multi_turboquant_kv.py` is a one-line change.

---

## Verifying it works

The repo ships with a test sweep that proves each method is actually
applying its quantization noise (not silently passing through as fp16):

```bat
set VLLM_MODEL_PATH=E:\models\Qwen3-14B-AWQ-4bit
set VLLM_PYTHON=E:\vllm-windows-build\venv\Scripts\python.exe
%VLLM_PYTHON% tests\test_tq_real.py
```

The expected output is a table where every method shows `DIFFERS` from
FP16 and `UNIQUE` from the others. If any method shows `IDENTICAL
(placebo!)`, the dispatch is broken.

See [tests/README.md](../tests/README.md) for details on each test
script and what it verifies.

---

## See also

- [benchmarks.md](benchmarks.md) — full performance numbers
- [architecture.md](architecture.md) — how the cache layout works
- [Multi-TurboQuant](https://github.com/aivrar/multi-turboquant) — the upstream library
