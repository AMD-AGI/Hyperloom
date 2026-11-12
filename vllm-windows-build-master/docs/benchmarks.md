# Benchmarks

This page preserves the original v0.19.0/cu126 Multi-TurboQuant benchmark
run. It is useful for understanding the local TQ memory/throughput tradeoff,
but it is not a performance claim for current releases. The current 0.33.0
wheel has targeted RTX 3090 serving, KV-format, and CPU-restore checks in its
[build record](v0.33.0-build-candidate.md); the 0.27.1 record has API,
prefix-cache, CPU-LRU, filesystem-restore, and persistent-restart measurements.
No broad Multi-TurboQuant sweep was rerun. See the current
[install requirements](install.md).

All numbers below are from the same historical hardware run with
the same settings, varying only `kv_cache_dtype`. Source data is in
[tests/](../tests/).

## Hardware

| | |
|---|---|
| GPU | NVIDIA RTX 3090 (24 GB, SM 8.6) |
| CPU | (16-core, 64 GB RAM) |
| OS | Windows 10 Pro 22H2 |
| Driver | 591.86 |
| CUDA | 12.6 |
| PyTorch | 2.10.0+cu126 |
| vLLM | 0.19.0+cu126 |

## Model

`Qwen3-14B-abliterated-AWQ-4bit`

| | |
|---|---|
| Architecture | Qwen3ForCausalLM |
| Parameters | 14.8B |
| Quantization | AWQ-4bit (compressed-tensors / Marlin) |
| Layers | 40 |
| Hidden size | 5120 |
| Attention heads | 40 |
| KV heads | 8 |
| Head dim | 128 |
| Tokenizer vocab | 151,936 |
| Disk size | 9.4 GB |
| GPU weights | 9.36 GiB |

## Build

| Step | Time |
|---|---|
| CMake configure | ~3 min |
| Compile (140 targets, MAX_JOBS=4) | ~28 min |
| Total | ~31 min |
| Wheel size | 201 MB |

## Model loading

| | Time | Notes |
|---|---|---|
| Custom mmap reader (load weights) | **6.5 s** | numpy.memmap + chunked GPU streaming |
| Marlin layout repack + Triton compile | ~3 s | one-time per process |
| Total LLM() init | ~10 s | |
| Compare: original safetensors mmap | ~189 s | unworkable on systems with small pagefile |

The Windows-specific safetensors reader is **29× faster** than the
upstream mmap path on Windows. The original path also fails outright
when the pagefile is too small to satisfy fragmented allocations
mid-load — see [troubleshooting.md](troubleshooting.md#oserror-1455).

## KV cache compression

Configuration: `max_model_len=512`, `gpu_memory_utilization=0.5`,
`max_num_seqs=4`, `block_size=16`, single 24 GB RTX 3090.

| dtype | Bytes/slot | KV tokens | Concurrency @ 512 | Cache vs FP16 |
|---|---|---|---|---|
| `auto` (fp16) | 256 | 16,336 | 31.91× | 1.00× |
| `isoquant3` | 128 | **32,672** | **63.94×** | **2.00×** |
| `isoquant4` | 128 | 32,672 | 63.94× | 2.00× |
| `planarquant3` | 128 | 32,672 | 63.94× | 2.00× |
| `planarquant4` | 128 | 32,672 | 63.94× | 2.00× |
| `turboquant25` | 128 | 32,672 | 63.94× | 2.00× |
| `turboquant35` | 128 | 32,672 | 63.94× | 2.00× |

Each TQ method also produces a unique numerical signature in inference
output (verified by `tests/test_tq_real.py`):

```
auto         ' Paris. What is the capital of Italy? The capital of Italy is Rome.'
isoquant3    ' Paris, and the capital of Canada is Ottawa.'
isoquant4    ' Paris. The capital of Italy is Rome. The capital of Spain is Madrid.'
planarquant3 ' Paris. The capital of France is Paris. The capital of France is...'
planarquant4 ' Paris. What is the capital of Italy? The answer is Rome.'
turboquant25 ' Paris. The capital of France is Paris. So, the capital of France...'
turboquant35 ' Paris. The capital of Italy is Rome. The capital of Spain is Madrid.'
```

## Throughput

Single 24 GB RTX 3090, single prompt, 20 tokens, deterministic sampling
(`temperature=0.0, seed=0`).

| dtype | tok/s | Generation cost |
|---|---|---|
| `auto` (fp16) | ~37 | (baseline) |
| `isoquant3` | ~0.12 | encode + decode in PyTorch, no fused kernel |
| `isoquant4` | ~0.12 | same |
| `planarquant3` | ~0.12 | same |
| `planarquant4` | ~0.12 | same |
| `turboquant25` | ~1.0 | turbo path is more vectorised |
| `turboquant35` | ~1.0 | same |

**The compression methods are correct but not fast.** The encode and
decode currently run in PyTorch-vectorised mode without a fused Triton
kernel. They give you real memory savings (the cache is half the size
on disk and in VRAM) but cost ~30-300× in throughput.

This makes them suitable for:
- Offline / batch inference where memory is the bottleneck
- Long-context workloads where the KV cache dominates
- Validation that the methods work end-to-end

They're not yet suitable for:
- Latency-sensitive online serving
- High-QPS workloads

A fused Triton kernel for encode and decode would close most of this
gap. The wiring is in place — the optimization is the next milestone.

## Reproducing

```bat
set VLLM_MODEL_PATH=E:\models\Qwen3-14B-abliterated-AWQ-4bit
set VLLM_PYTHON=E:\vllm-windows-build\venv\Scripts\python.exe

REM Smoke test (~30 sec)
%VLLM_PYTHON% tests\test_v19.py

REM TQ correctness sweep (~15 min)
%VLLM_PYTHON% tests\test_tq_real.py

REM Full benchmark (~70 min)
%VLLM_PYTHON% tests\test_tq_thorough.py
```
