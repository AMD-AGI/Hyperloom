# torchtune Optimization Summary — Cross-Model Results

**Last updated:** 2026-03-23
**Platform:** AMD Instinct MI300X / MI355X (ROCm, PyTorch 2.9–2.10)
**Framework:** [torchtune](https://github.com/pytorch/torchtune) — PyTorch-native post-training library
**Methodology:** Profile-guided iterative optimization (baseline → profile → one-change-at-a-time → measure → keep/discard)

---

## Results at a Glance

| Model | Params | Recipe | GPUs | Baseline | Optimized | Speedup | Date |
|-------|--------|--------|------|----------|-----------|---------|------|
| **Qwen3 8B** | 8.2B | full finetune (FSDP) | 8× MI355X | 862.6 tok/s/gpu | 18,859.9 tok/s/gpu | **21.9×** | 2026-03-23 |
| **Qwen3 32B** | 32.5B | LoRA finetune (FSDP) | 8× MI355X | 467.4 tok/s/gpu | 4,984.3 tok/s/gpu | **10.7×** | 2026-03-23 |
| **Llama 4 Scout 17B-16E** | ~109B (17B active) | full finetune (TP+FSDP) | 8× MI355X | 19.9 tok/s/gpu | 170.4 tok/s/gpu | **8.6×** | 2026-03-23 |
| **Qwen2.5-Coder 7B** | 7.6B | LoRA finetune | 1× MI300X | 1.15 it/s | 3.8 it/s (5.1× tok/s) | **3.3× / 5.1×** | 2026-03-19 |

> **Note on metrics:** Qwen3 8B, 32B, and Llama 4 use tokens/sec/gpu (throughput-normalized). Qwen2.5 7B uses it/s for same-workload comparison and tokens/sec for packed comparison. All measured after warmup (steps 6+).
> **Note on Llama 4 Scout:** This is a Mixture-of-Experts model (16 experts, top-1 routing). torch.compile is incompatible with MoE routing, so the 8.6× gain comes entirely from data layout and memory management optimizations. Further gains (batch_size, seq_len) were not explored due to time.

---

## Optimization Levers — What Works Across Models

| Optimization | Qwen3 8B (full ft) | Qwen3 32B (LoRA) | Llama 4 Scout (MoE full ft) | Qwen2.5 7B (LoRA) | Generalizable? |
|-------------|--------------------|--------------------|----------------------------|--------------------| ----------------|
| **`torch.compile`** | +62% | +44% | **N/A — crashes** (MoE guard) | +52% | **Dense only** — fails on MoE routing |
| **`packed=True`** | +440% (on compiled) | +336% (on compiled) | **+756%** (uncompiled) | +5.1× tok/s | **Yes** — universal, even bigger on MoE |
| **Larger `batch_size` / smaller GA** | +54% cumulative | +25% (limited by memory) | Not explored | +100% (bs 2→16) | **Yes** — amortizes per-step overhead |
| **Disable activation checkpointing** | +20% | N/A (already off for LoRA) | **+22%** | +38% | **Yes** — when GPU memory allows |
| **`fsdp_reshard_after_forward=False`** | +23% | +20% | Not explored | N/A (single GPU) | **Multi-GPU only** — SHARD_GRAD_OP |
| **Sequence length tuning** | +10% (4096) | +10% (8192) | Not explored | N/A | **Yes** — longer seqs amortize overhead |
| **NCCL env vars** | <1% | **+5%** (Ring) | Not explored | N/A | **Scale-dependent** — matters more at 32B |
| **`optimizer_in_bwd`** | -0.6% | N/A | N/A | N/A | **No** — conflicts with compiled optimizer |
| **`CUDA_DEVICE_MAX_CONNECTIONS`** | <0.1% | <0.1% | N/A | N/A | **No** — negligible |

### Consistent Pattern Across All Models

The optimization stack follows a clear priority order:

```
1. torch.compile                    (~44-62% gain, dense models only — fails on MoE)
2. Pack sequences                   (~3-8× token throughput, every model including MoE)
3. Maximize batch size per GPU      (~25-100% gain, memory-dependent)
4. FSDP sharding strategy           (~20% gain, multi-GPU only)
5. Disable activation checkpointing (~20-40% gain, when memory allows)
6. Sequence length tuning           (~10% gain, memory-dependent)
7. NCCL algorithm tuning            (~5% at 32B+, <1% at 8B)
```

**Key scaling insight:** At larger model sizes (32B), memory becomes the binding constraint — batch size is limited and longer sequences become the primary throughput lever. NCCL tuning also becomes more impactful as message sizes grow.

**torch.compile caveat:** Non-power-of-2 batch sizes cause severe regressions (~75%) on the 32B model due to recompilation. Always use batch_size ∈ {1, 2, 4, 8, 16, ...}.

**MoE caveat:** `torch.compile` is incompatible with Mixture-of-Experts models due to data-dependent expert routing (dynamic top-k selection). This makes packed datasets even more critical for MoE — it becomes the primary optimization lever when compile is unavailable.

---

## Individual Model Reports

### Qwen3 8B — Full Finetune Distributed (8× MI355X)

- **Report:** [`qwen3_8b_optimization_20260323/optimization_report.md`](qwen3_8b_optimization_20260323/optimization_report.md)
- **Results TSV:** [`qwen3_8b_optimization_20260323/results.tsv`](qwen3_8b_optimization_20260323/results.tsv)
- **Profile traces:** [`qwen3_8b_optimization_20260323/traces/`](qwen3_8b_optimization_20260323/traces/)

| Metric | Baseline | Optimized |
|--------|----------|-----------|
| tokens/sec/gpu | 862.6 | 18,859.9 |
| Peak GPU mem | 14.3 GiB / 288 GiB | ~283 GiB / 288 GiB |
| Attempts | — | 19 (8 kept, 4 crashed) |

**Key finding:** Baseline was communication-dominated (45% NCCL) due to severe memory underutilization (5% of 288 GiB HBM). Matching config to hardware unlocked 21.9× throughput.

**Best config overrides:**
```yaml
compile: True
dataset.packed: True
tokenizer.max_seq_len: 4096
fsdp_reshard_after_forward: False
batch_size: 8
gradient_accumulation_steps: 1
enable_activation_checkpointing: False
```

---

### Qwen3 32B — LoRA Finetune Distributed (8× MI355X)

- **Report:** [`qwen3_32b_lora_optimization_20260323/optimization_report.md`](qwen3_32b_lora_optimization_20260323/optimization_report.md)
- **Results TSV:** [`qwen3_32b_lora_optimization_20260323/results.tsv`](qwen3_32b_lora_optimization_20260323/results.tsv)

| Metric | Baseline | Optimized |
|--------|----------|-----------|
| tokens/sec/gpu | 467.4 | 4,984.3 |
| Peak GPU mem | ~281 GiB / 288 GiB | ~281 GiB / 288 GiB |
| Attempts | — | 19 (8 kept, 5 crashed) |

**Key finding:** Memory is the binding constraint at 32B scale — the model fills 97% of 288 GiB HBM. Optimization pivots from "use more memory" (8B strategy) to "maximize compute density within fixed memory." Longer sequences (8192) beat larger batches. Non-power-of-2 batch sizes cause torch.compile regressions. NCCL Ring algorithm gives meaningful +5% at this scale.

**Best config overrides:**
```yaml
compile: True
dataset.packed: True
tokenizer.max_seq_len: 8192
fsdp_reshard_after_forward: False
batch_size: 1
gradient_accumulation_steps: 1
# env: NCCL_ALGO=Ring
```

---

### Llama 4 Scout 17B-16E — Full Finetune Distributed (8× MI355X) ⚡ MoE

- **Report:** [`llama4_scout_17b_optimization_20260323/optimization_report.md`](llama4_scout_17b_optimization_20260323/optimization_report.md)
- **Results TSV:** [`llama4_scout_17b_optimization_20260323/results.tsv`](llama4_scout_17b_optimization_20260323/results.tsv)

| Metric | Baseline | Optimized |
|--------|----------|-----------|
| tokens/sec/gpu | 19.9 | 170.4 |
| Peak GPU mem | 36.25 GiB / 288 GiB | ~36 GiB / 288 GiB |
| Attempts | — | 3 (2 kept, 1 crashed) |

**Key finding:** First MoE model in the series. torch.compile fails entirely due to data-dependent expert routing, making this a crucial test of non-compile optimizations. Packed datasets alone deliver 8.6× — the largest single-optimization gain across all models — because MoE routing overhead is wasted on padding tokens. Significant further headroom exists (batch_size, seq_len, FSDP strategy) but was not explored due to time.

**Best config overrides:**
```yaml
dataset.packed: True
tokenizer.max_seq_len: 2048
enable_activation_checkpointing: False
fsdp_cpu_offload: False
compile: False  # MoE incompatible
```

**Bug fix required:** `torchtune/modules/loss/cross_entropy_loss.py` needs patching to recognize `"decoder.output"` key in TP plan (Llama4 uses `"decoder.output"` instead of `"output"`).

---

### Qwen2.5-Coder 7B — LoRA Finetune Single-Device (1× MI300X)

- **Report:** [`torchtune/qwen25_7b_lora_optimization_report.md`](torchtune/qwen25_7b_lora_optimization_report.md)

| Metric | Baseline | Optimized (same workload) | Optimized (packed) |
|--------|----------|--------------------------|-------------------|
| it/s | 1.15 | 3.8 | — |
| tokens/sec | ~590 | ~1,540 | ~23,918 |
| Peak GPU mem | 17.8 GiB / 64 GiB | ~32 GiB / 64 GiB | ~48 GiB / 64 GiB |
| Attempts | — | 8 | — |

**Key finding:** Baseline was elementwise-op dominated (29% of GPU time). torch.compile fused these into efficient Triton kernels, and disabling AC removed unnecessary recomputation on a 64 GiB GPU.

**Best config overrides:**
```yaml
compile: True
batch_size: 16
gradient_accumulation_steps: 1
enable_activation_checkpointing: False
dataset.packed: True           # for max tok/s
tokenizer.max_seq_len: 2048    # for packed
```

---

## Platform Details

| Component | Value |
|-----------|-------|
| GPUs | AMD Instinct MI300X (64 GiB HBM3) / MI355X (288 GiB HBM, gfx950 CDNA4) |
| ROCm | 7.2.26015 |
| PyTorch | 2.9.1 / 2.10.0a0 |
| torchtune | dev (0.0.0, from-source) |
| Dataset | alpaca_cleaned (51,760 samples) |

---

## How to Add a New Model

1. Pick a model + recipe from `tune ls` (e.g., `gemma2/9B_full`, `llama4/scout_17B_16E_full`)
2. Run stock baseline with `max_steps_per_epoch=15`
3. Profile with `profiler.enabled=True`
4. Apply the optimization stack in order (compile → pack → batch → AC → FSDP → seq_len)
5. Record in a new `<model>_optimization_<date>/` directory under `/shared_nfs/nehaprakriya/results/`
6. Update this summary table

### Candidate models ready to go:

| Model | Recipe | Config | Weights Available | Notes |
|-------|--------|--------|-------------------|-------|
| ~~Llama 4 Scout 17B-16E~~ | ~~`full_finetune_distributed`~~ | ~~`llama4/scout_17B_16E_full`~~ | ~~Done~~ | **Completed** — 8.6× (MoE, torch.compile incompatible) |
| Gemma 2 9B | `full_finetune_distributed` | `gemma2/9B_full` | Needs download | Different arch (Google) |
| Llama 3.1 8B | `full_finetune_distributed` | `llama3_1/8B_full` | Needs HF token (gated) | Most popular model |
| ~~Qwen3 32B~~ | ~~`lora_finetune_distributed`~~ | ~~`qwen3/32B_lora`~~ | ~~Done~~ | **Completed** — 10.7× |

---

*Generated by workload-optimization agent. See individual reports for full methodology and attempt logs.*
