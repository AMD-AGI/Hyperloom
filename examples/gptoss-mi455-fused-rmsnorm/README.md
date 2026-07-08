# Fused RMSNorm kernel demo — gpt-oss-120b on MI455 (gfx1250)

Reproduces a **kernel-level optimization** for `gpt-oss-120b` served with vLLM on a
pre-silicon **AMD MI455 / gfx1250** board: a **fused residual-add + RMSNorm Triton
kernel** that replaces the stock eager path.

## Why this kernel

On this image the `_C` `fused_add_rms_norm` op fails to load (ABI mismatch:
`undefined symbol getCurrentHIPStream`), so vLLM's `RMSNorm` falls back to
`forward_static`, where the residual add runs **unfused in fp32**:

```python
x = x.to(torch.float32)
x = x + residual          # <-- profiling: ~25% of GPU time, biggest single kernel
```

A torch-profiler trace of the baseline showed this single fp32 elementwise add is
the #1 GPU consumer, ahead of the rocBLAS dense GEMM (~18%) and the mxfp4 MoE
matmul (~19%). This demo fuses residual-add + RMSNorm into **one Triton kernel**
(bf16 I/O, fp32 accumulate), eliminating that add plus the separate up/down-casts
and reduce/rsqrt launches — no rebuild, bind-mounted like the mxfp4 patch.

## Results (measured)

| | value |
|---|---|
| Isolated op (N=2048, H=2880, bf16) | **8.4× faster** — 52 µs vs 438 µs |
| End-to-end throughput @ conc32/ISL1024/OSL128 | **~+10%** |
| Correctness | Paris / 391 / 101,103,107 (matches baseline), rel < 0.7% vs fp32 ref |

> Honest caveat: the board thermally drifts ~16% over a long session, so `run_demo.sh`
> measures baseline and fused **back-to-back** and warms first (cold Triton autotune
> is ~2.5× slower — always discard cold runs).

## Prerequisites

- Docker with `--device=/dev/kfd --device=/dev/dri` (ROCm) access to the gfx1250 board.
- The serving image: `registry-sc-harbor.amd.com/hotswap/dsv4-hotswap-overlay:gfx1250-deepseek-probe2`
- Model at `/home/yanyuqin/models/gpt-oss-120b` (override with `MODEL_HOST`).
- The required gfx1250 mxfp4 MoE fix at `/home/yanyuqin/tk-patch/mxfp4_utils.py`
  (override with `MXFP4_PATCH`) — this is the `block_k=256` kernel fix, unrelated to
  the RMSNorm work but needed for the model to serve at all on gfx1250.
- A free GPU card (default card `1`; card `0` is assumed to hold a production server).

## Run it

```bash
cd examples/gptoss-mi455-fused-rmsnorm

# Full A/B (launches baseline, then fused; correctness + warm + measure each):
./run_demo.sh

# Also run the fast isolated kernel microbench first (correctness + 8.4x op speedup,
# no full model load needed):
MICRO=1 ./run_demo.sh
```

Expected tail:

```
==================== RESULT (median out tok/s @ conc=32, ISL=1024, OSL=128) ====================
  baseline (stock RMSNorm)       ~490
  fused add+RMSNorm (demo)       ~540
  throughput delta               +10.x%
================================================================================================
```

### Knobs (env vars)

| var | default | meaning |
|---|---|---|
| `CARD` | `1` | physical GPU (ROCR index) to lease |
| `PORT` | `8001` | server port |
| `CONC` / `NREQ` | `32` / `64` | benchmark concurrency / request count |
| `ISL` / `OSL` | `1024` / `128` | input / output token lengths |
| `WARMUP` / `RUNS` | `3` / `4` | warm-up runs (discarded) / measured runs |
| `MODEL_HOST` | `/home/yanyuqin/models` | host dir containing `gpt-oss-120b/` |
| `MXFP4_PATCH` | `/home/yanyuqin/tk-patch/mxfp4_utils.py` | required gfx1250 MoE fix |

## Files

| file | what |
|---|---|
| `run_demo.sh` | orchestrator: baseline vs fused A/B (this is what you run) |
| `serve.sh` | launch one gptoss vLLM server; `MODE=baseline|fused` toggles the kernel |
| `layernorm_fused.py` | the patched vLLM `layernorm.py` with the fused Triton kernel (bind-mounted in `MODE=fused`) |
| `bench.py` | closed-loop throughput probe; unique prompt per request (defeats prefix-cache inflation) |
| `microbench.py` | isolated kernel: correctness vs fp32 reference + op-level speedup (fast) |
| `correctness.sh` | 3 spot-check prompts against a running server |

## Run individual pieces

```bash
# just the fused server, then poke it:
MODE=fused ./serve.sh
PORT=8001 ./correctness.sh
python3 bench.py 8001 fused 32 64 1024 128

# just the isolated kernel correctness + speedup (inside a gptoss container):
docker run -d --name m --network host --privileged --device=/dev/kfd --device=/dev/dri \
  -e ROCR_VISIBLE_DEVICES=1 -e HIP_VISIBLE_DEVICES=0 \
  registry-sc-harbor.amd.com/hotswap/dsv4-hotswap-overlay:gfx1250-deepseek-probe2 -lc 'sleep infinity'
# run from /root — the image's default cwd shadows `import triton` with its source tree
docker cp microbench.py m:/root/ && docker exec m bash -lc 'cd /root && python3 microbench.py'
docker rm -f m
```

## The one non-obvious bug (worth knowing)

The first version of the kernel produced **garbage output** despite passing an
isolated random-input test. Cause: the kernel hardcodes row stride =
`hidden_size`, but vLLM sometimes hands `RMSNorm` a **non-contiguous (sliced)**
tensor, and `reshape()` + `empty_like()` silently preserved the wrong stride. Fix
in `layernorm_fused.py`: force `.contiguous()` on the 2D views before the kernel.
The correctness gate (`correctness.sh`) catches this — always run it before trusting
a throughput number.

## Cleanup

```bash
docker rm -f gptoss-demo-baseline gptoss-demo-fused gptoss-demo-micro
```
