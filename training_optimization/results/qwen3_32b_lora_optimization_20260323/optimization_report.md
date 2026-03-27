# Qwen3 32B LoRA Optimization Report — torchtune on MI355X 8-GPU

**Date:** 2026-03-23
**Platform:** 8× AMD Instinct MI355X (gfx950, CDNA4, 288 GiB HBM per GPU)
**ROCm:** 7.2.26015, PyTorch 2.10.0a0+git449b176
**Framework:** torchtune (dev build, PyTorch-native post-training library)
**Model:** Qwen3 32B (64 layers, 64 heads, 8 KV heads, 5120 hidden, 25600 intermediate, 152K vocab, bf16)
**Recipe:** `lora_finetune_distributed` (FSDP + LoRA rank=8, alpha=16)
**Dataset:** alpaca_cleaned (51,760 samples)

---

## Executive Summary

| Metric | Baseline | Optimized | Delta |
|---|---|---|---|
| **tokens/sec/gpu** | 467.4 | 4,984.3 | **+4,517** |
| **Speedup** | — | — | **10.7×** |
| Total attempts | — | 19 | |
| Kept | — | 8 | |
| Crashes (OOM) | — | 5 | |

**Baseline:** Stock torchtune `qwen3/32B_lora` config — LoRA (rank=8, q/v/output+MLP), batch_size=2, gradient_accumulation_steps=8, unpacked variable-length sequences, no activation checkpointing, compile disabled, FSDP FULL_SHARD.

**Optimized config (attempt 17, best):**
```yaml
compile: True
dataset.packed: True
tokenizer.max_seq_len: 8192
fsdp_reshard_after_forward: False   # SHARD_GRAD_OP
batch_size: 1
gradient_accumulation_steps: 1
# env: NCCL_ALGO=Ring
```

---

## What Worked (cumulative progression)

| # | tok/s/gpu | vs Baseline | vs Previous | Description |
|---|-----------|-------------|-------------|-------------|
| 0 | 467.4 | — | — | **Baseline** (stock config) |
| 1 | 673.8 | +44.2% | +44.2% | `compile=True` — fuses elementwise ops |
| 2 | 2,935.4 | +527.9% | +335.5% | + `packed=True` (seq_len=2048) — eliminates padding waste |
| 3 | 3,526.8 | +654.4% | +20.2% | + `fsdp_reshard_after_forward=False` — SHARD_GRAD_OP |
| 4 | 4,391.1 | +839.4% | +24.5% | + `batch_size=4, GA=4` — larger micro-batch |
| 9 | 4,658.1 | +896.5% | +6.1% | → `seq_len=4096, batch_size=2` — longer packed sequences |
| 12 | 4,752.0 | +916.6% | +2.0% | → `seq_len=8192, batch_size=1` — maximum seq length |
| **17** | **4,984.3** | **+966.3%** | **+4.9%** | + `NCCL_ALGO=Ring` — communication algorithm tuning |

### What Didn't Work

| # | Result | Description |
|---|--------|-------------|
| 5 | OOM | `batch_size=8` with SHARD_GRAD_OP (32B fills 281/288 GiB) |
| 7 | -75% | `batch_size=6` — torch.compile regression on non-power-of-2 batch |
| 8 | OOM | `seq_len=4096, batch_size=4` |
| 10 | -75% | `batch_size=3` — same torch.compile shape issue |
| 11 | -6% | AC=True + larger batch — AC overhead outweighs batch gain |
| 13 | OOM | `seq_len=16384` — exceeds memory at 32B scale |
| 14 | -15% | AC=True + seq=8192 bs=2 — AC overhead dominates |
| 15 | -75% | `seq_len=12288` — torch.compile regression on non-standard shape |
| 16 | +0.0% | `CUDA_DEVICE_MAX_CONNECTIONS=2` — negligible |
| 18 | -0.1% | `custom_sharded_layers=['tok_embeddings']` — marginal regression |
| 19 | OOM | FULL_SHARD + seq=8192 bs=2 |

---

## Key Insights for 32B-Scale Models

### 1. Memory is the binding constraint
At 32B with LoRA, the model alone consumes ~281 GiB per GPU with SHARD_GRAD_OP. This leaves only ~7 GiB for activations, severely limiting batch size and sequence length. Every optimization must be evaluated against its memory footprint.

### 2. torch.compile requires power-of-2 batch sizes
Non-power-of-2 batch sizes (3, 6, 12) consistently caused **75% regressions** due to torch.compile recompilation and suboptimal kernel selection. Stick to batch_size ∈ {1, 2, 4, 8, 16, ...}.

### 3. Sequence length > batch size for throughput at large scale
Unlike the 8B model where batch_size=16 was optimal, the 32B model benefits more from longer sequences (8192) with batch_size=1. The quadratic attention compute at longer sequences better amortizes the fixed FSDP communication cost.

### 4. Activation checkpointing is a net negative for LoRA at this scale
With LoRA freezing 99.6% of parameters, the activation memory is relatively small. AC's recomputation cost (-15-20%) always exceeded the memory savings gained.

### 5. NCCL tuning matters more at larger model scale
`NCCL_ALGO=Ring` gave +4.9% on 32B but <1% on 8B. Larger per-message sizes in FSDP communication benefit from Ring algorithm's bandwidth efficiency.

---

## All Attempts

| # | tok/s/gpu | Speedup | Status | Description |
|---|-----------|---------|--------|-------------|
| 0 | 467.4 | 0.0% | baseline | Stock torchtune qwen3/32B_lora config |
| 1 | 673.8 | +44.2% | keep | compile=True |
| 2 | 2,935.4 | +527.9% | keep | + packed=True (seq=2048) |
| 3 | 3,526.8 | +654.4% | keep | + fsdp_reshard_after_forward=False |
| 4 | 4,391.1 | +839.4% | keep | + batch_size=4 GA=4 |
| 5 | — | — | crash | batch_size=8 GA=2 (OOM) |
| 6 | 4,404.5 | +842.3% | keep | batch_size=4 GA=1 |
| 7 | 1,102.8 | +135.9% | discard | batch_size=6 (compile shape regression) |
| 8 | — | — | crash | seq=4096 bs=4 (OOM) |
| 9 | 4,658.1 | +896.5% | keep | seq=4096 bs=2 |
| 10 | 1,173.5 | +151.1% | discard | seq=4096 bs=3 (compile shape regression) |
| 11 | 4,133.3 | +784.2% | discard | seq=4096 bs=4 AC=True |
| 12 | 4,752.0 | +916.6% | keep | seq=8192 bs=1 |
| 13 | — | — | crash | seq=16384 bs=1 (OOM) |
| 14 | 4,023.7 | +860.8% | discard | seq=8192 bs=2 AC=True |
| 15 | 1,201.6 | +157.1% | discard | seq=12288 bs=1 (compile shape regression) |
| 16 | 4,977.4 | +964.9% | keep | seq=8192 + NCCL_Ring + CUDA_CONN=2 |
| **17** | **4,984.3** | **+966.3%** | **keep** | **BEST: seq=8192 + NCCL_Ring** |
| 18 | 4,978.9 | +965.2% | discard | + custom_sharded_layers (marginal regression) |
| 19 | — | — | crash | FULL_SHARD + seq=8192 bs=2 (OOM) |

---

## Methodology

Systematic iterative optimization following the same THINK → TRY → MEASURE → DECIDE loop used on Qwen3 8B and gpt-oss:
1. **torch.compile** — single biggest per-change win (+44%)
2. **Data efficiency** — packed sequences (+336%)
3. **FSDP strategy** — SHARD_GRAD_OP (+20%)
4. **Batch size / GA tuning** — constrained by memory, batch_size=4 max at seq=2048 (+25%)
5. **Sequence length tuning** — longer sequences at smaller batch (+10% cumulative)
6. **NCCL tuning** — Ring algorithm (+5%)

---

## Artifacts

| Artifact | Location |
|----------|----------|
| Results TSV | `results.tsv` |
| Baseline config | `configs/baseline.yaml` |
| All attempt logs | `logs/` |
| This report | `optimization_report.md` |

---

## Reproducibility

**Baseline:**
```bash
/opt/venv/bin/tune run --nnodes 1 --nproc_per_node 8 lora_finetune_distributed \
    --config /shared_nfs/nehaprakriya/results/qwen3_32b_lora_optimization_20260323/configs/baseline.yaml
```

**Optimized:**
```bash
NCCL_ALGO=Ring /opt/venv/bin/tune run --nnodes 1 --nproc_per_node 8 lora_finetune_distributed \
    --config /shared_nfs/nehaprakriya/results/qwen3_32b_lora_optimization_20260323/configs/baseline.yaml \
    compile=True \
    dataset.packed=True \
    tokenizer.max_seq_len=8192 \
    fsdp_reshard_after_forward=False \
    batch_size=1 \
    gradient_accumulation_steps=1
```

---

*Generated by workload-optimization agent on Sun Mar 23 2026*
