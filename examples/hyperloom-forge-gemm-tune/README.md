# Hyperloom demo — deterministic GEMM tuning with `forge`

This is a **real Hyperloom demo**: it runs an actual Hyperloom kernel-optimization
backend — **`forge` (`forge_gemm_tune`)** — end-to-end against `gpt-oss-120b` on
MI455 / gfx1250. Unlike the sibling `gptoss-mi455-fused-rmsnorm/` example (a
hand-written kernel patch), this drives a Hyperloom tool.

## Why forge runs here (and GEAK / the full coordinator don't)

`forge_gemm_tune` is **LLM-free** — "all tuning is exhaustive search via aiter CK
tuners or PyTorch TunableOp." So it needs **zero credentials**: no LLM gateway, no
`SAFE_API_KEY`, no `ANTHROPIC_API_KEY`. That's exactly why it's the one Hyperloom
optimization path that works on a credential-less box, whereas:

- `inference_optimizer optimize` (the Coordinator) needs the LLM gateway or a real
  Anthropic key + full `install.sh`;
- `geak` needs an LLM endpoint (a subscription OAuth token can't drive litellm).

## What the demo does

The canonical forge dense-GEMM pipeline, orchestrated by `run_forge_demo.sh`:

1. **RECORD** — serve gpt-oss-120b with `PYTORCH_TUNABLEOP_RECORD_UNTUNED=1` and
   drive traffic to capture the real dense GEMM shapes (attention QKV/O proj,
   gate/up, lm_head).
2. **TUNE** — `forge-gemm-tune run --tuner vllm_dense_tunableop` runs PyTorch
   TunableOp exhaustive search over those shapes → a tuned kernel-selection CSV.
3. **VALIDATE** — serve with the tuned CSV vs baseline, **back-to-back A/B**
   (warms first — cold Triton autotune is ~2.5× slower).

## Honest expected result (pre-silicon gfx1250)

forge finds faster **non-Default** hipBLASLt/rocBLAS kernels for many of the ~37
shapes, but **none clears its 3% improvement bar**, and the end-to-end A/B shows
**~no gain** — the dense GEMMs (~18% of GPU time) are already near-optimal on this
board. That is the correct outcome, and the point of the demo:

- it shows the **workflow** of running a Hyperloom backend end-to-end, and
- it shows a *deterministic* tuner honestly reporting "no headroom" instead of
  inventing one.

(For a path that *did* win on this board, see the fused-RMSNorm example — the
headroom here is in kernel **fusion**, not GEMM **selection**.)

## Prerequisites

- Docker with ROCm device access (`--device=/dev/kfd --device=/dev/dri`) to gfx1250.
- Image `registry-sc-harbor.amd.com/hotswap/dsv4-hotswap-overlay:gfx1250-deepseek-probe2`.
- Model at `/home/yanyuqin/models/gpt-oss-120b` (`MODEL_HOST`).
- The gfx1250 mxfp4 MoE fix at `/home/yanyuqin/tk-patch/mxfp4_utils.py` (`MXFP4_PATCH`).
- KernelForge checkout providing `forge_gemm_tune` at
  `/home/yanyuqin/hyperloom-run/deps/KernelForge/src/forge_gemm_tune` (`FORGE_SRC`).
- A free GPU card (default `1`).

## Run it

```bash
cd examples/hyperloom-forge-gemm-tune
./run_forge_demo.sh
```

Expected tail:

```
================= FORGE GEMM-TUNE RESULT (median out tok/s @ conc=32) =================
  baseline (TunableOp off)          ~514
  forge-tuned dense GEMMs           ~500
  throughput delta                  ~0% (within noise)
======================================================================================
```

### Knobs (env)

| var | default | meaning |
|---|---|---|
| `CARD` / `PORT` | `1` / `8001` | GPU card / server port |
| `CONC`/`NREQ`/`ISL`/`OSL` | `32`/`64`/`1024`/`128` | benchmark workload |
| `WARMUP`/`RUNS` | `3`/`4` | warm-up (discarded) / measured runs |
| `MODEL_HOST` | `/home/yanyuqin/models` | host dir with `gpt-oss-120b/` |
| `MXFP4_PATCH` | `/home/yanyuqin/tk-patch/mxfp4_utils.py` | required gfx1250 MoE fix |
| `FORGE_SRC` | `.../KernelForge/src/forge_gemm_tune` | forge source to `pip install -e` |
| `OUT_HOST` | `$MODEL_HOST/forge_demo_out` | artifacts (untuned/tuned CSVs, `tune/result.json`) |

## Files

| file | what |
|---|---|
| `run_forge_demo.sh` | orchestrator: control container → install forge → record → tune → A/B |
| `_run_vllm.sh` | in-container vLLM launcher honoring ambient `PYTORCH_TUNABLEOP_*` env (kills prior server + waits for the port to free, avoiding "false-ready") |
| `bench.py` | closed-loop throughput probe (unique prompt/request) |

## Inspect the forge output

```bash
cat  /home/yanyuqin/models/forge_demo_out/tune/result.json        # status, tuners_run, improved_shapes
head /home/yanyuqin/models/forge_demo_out/tune/tuners/vllm_dense_tunableop/tunableop_results.csv
```
The `Validator` header (`GCN_ARCH_NAME,gfx1250`, `HIPBLASLT_VERSION`, …) records the
exact backend the tuning is valid for.

## Cleanup

```bash
docker rm -f hl-forge-demo
```
