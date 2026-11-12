# run/ — bench & tooling drivers (native Windows + ROCm)

**Run everything from THIS directory** (not the repo root): the cloned `vllm/` directory at the
root would shadow the installed `vllm` package as a namespace package otherwise.

Install and first-run instructions live in [`../SETUP.md`](../SETUP.md); feature documentation
(KVarN, fp8 KV cache, CK FMHA prefill, native kernels) in the [root README](../README.md).

Historical milestone: first token **PASSED 2026-06-30** on RX 7900 XT (gfx1100) — believed
world-first for vLLM on native Windows + ROCm (OPT-125m, eager, `TRITON_ATTN`).

## Scripts

| script | what it does |
|---|---|
| `first_token.py` | smallest end-to-end smoke test (OPT-125m) |
| `bench.py` | decode tok/s + VRAM (`VLLM_BENCH_COMPILE`, `VLLM_BENCH_CGMODE`, `VLLM_BENCH_MODEL`, …) |
| `batch_sweep.py` | aggregate throughput vs concurrency |
| `precision_check.py` | generation-quality sanity across configs |
| `profile_decode.py` | decode profiling |
| `test_compile.py` | torch.compile / inductor smoke test |
| `chat_ui.py` | dependency-free browser chat on top of `vllm serve` — or just `localserve.ps1` at the repo root |
| `kv_bench.py` | KV-cache dtype bench: prefill/decode speed and output drift vs fp16 across context lengths (`VLLM_KV_*` env vars, see the file docstring) |
| `kv_scale_probe.py` | per-layer amplitudes, per-channel max\|K\| map, one-layer q/k/v dump, and fp8 KV offset/scale calibration (`VLLM_KV_PROBE_*`) |
| `kv_quant_schemes.py` | offline KV quantization-format comparison on a probe dump |
| `kv_pc_kernel_check.py` | direct bit-exact check of the fp8 per-channel-scale Triton kernels |
| `s5_bench/` | micro-benchmarks (flash/moe/sliding kernels, perf ablations), see `s5_bench/README.md` |

## The fp8 KV workflow on Qwen2.5 (issue #28)

fp8 KV with default scales garbles Qwen2.5 output (K carries a large per-channel offset). The
fix is a two-step calibration, both steps from `run/` with `HIP_VISIBLE_DEVICES` set if your
box has an iGPU:

```bash
# 1. calibrate once: writes offsets + per-layer + per-channel scales
VLLM_KV_CTXS=128,2048,8192 VLLM_KV_PROBE_MEAN=kv_offsets.pt python kv_scale_probe.py

# 2. run any fp8 workload with the offsets file
VLLM_KV_DTYPE=fp8 VLLM_WIN_KV_OFFSETS=kv_offsets.pt python kv_bench.py
```

Design, measurements and results: [`../kv-fp8-audit.md`](../kv-fp8-audit.md).

## Engine notes

- In-process engine only: `VLLM_ENABLE_V1_MULTIPROCESSING=0` (set by every script before
  importing vllm) — the plugin's config-time hooks need it.
- `VLLM_ATTENTION_BACKEND` is gone in this vLLM version — use the `attention_backend=` kwarg.
- vLLM pin: the **v0.19.1** tag (== commit `b1388b1`, reported by pip as `0.19.2.dev0`), which
  pins torch 2.10.
