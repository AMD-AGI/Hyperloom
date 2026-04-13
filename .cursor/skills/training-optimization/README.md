# Training Optimization Skill

This folder contains skills for automated GPU training optimization in Cursor. `@`-reference the skill files in your Cursor chat to use them.

| File | Description |
|------|-------------|
| `SKILL.md` | Main skill — iterative training optimization loop |
| `GEAK-KERNEL-OPTIMIZATION.md` | Optional extension — kernel-level optimization via GEAK |
| `README.md` | This file — usage guide and examples |

---

## What It Does

Give the agent a real distributed training workload (Primus/Megatron), and it will:
1. Run the training to establish a baseline (ms/iter from actual `torchrun`)
2. Profile with `torch.profiler` and analyze kernel breakdown
3. Enter a closed-loop optimization cycle: apply one config override or code patch → re-run training → measure ms/iter → keep or revert → repeat
4. Optionally send Triton kernels to GEAK for kernel-level optimization
5. Write a report with what worked, what didn't, and final speedup

## Architecture: How It All Fits Together

```
  ┌─────────────┐
  │ Setup Phase  │  Read config, understand torchrun command
  └──────┬──────┘
         ▼
  ┌──────────────┐
  │  Baseline    │  Run training, record ms/iter (iter 6–10)
  └──────┬───────┘
         ▼
  ┌───────────────────────┐
  │ Profile + TraceLens   │  torch.profiler trace → kernel breakdown
  │ Agentic Analysis      │  TraceLens CLI for automated analysis
  └──────┬────────────────┘
         ▼
  ┌──────────────────────────────────────────────┐
  │  DECISION POINT: Check for GEAK candidates   │
  │                                              │
  │  Scan top-20 kernels from profile:           │
  │  - Any custom Triton kernels > 3% GPU time?  │
  │  - Any custom HIP kernels > 3% GPU time?     │
  │  - Any unfused elementwise chains > 5%?      │
  │                                              │
  │  If YES → kick off GEAK (GEAK-KERNEL-        │
  │           OPTIMIZATION.md) IN PARALLEL        │
  │  If NO  → skip GEAK, continue code-only      │
  └──────┬───────────────────────────────────────┘
         ▼
  ┌──────────────────────────────────────────────────┐
  │              Optimization Loop                    │
  │                                                   │
  │  ┌───────────┐    ┌───────────┐    ┌──────────┐  │
  │  │ 1. THINK  │───▶│  2. TRY   │───▶│3. MEASURE│  │
  │  │ Pick idea │    │ Apply one │    │ ms/iter   │  │
  │  └───────────┘    │  change   │    └────┬─────┘  │
  │       ▲           └───────────┘         │        │
  │       │                                 ▼        │
  │       │                          ┌───────────┐   │
  │       │    keep or revert        │ 4. DECIDE │   │
  │       └──────────────────────────│  keep /   │   │
  │                                  │  revert   │   │
  │  Types of changes:               └───────────┘   │
  │  • Config override                               │
  │  • Code patch                                    │
  │  • GEAK kernel (when result arrives)             │
  │  • Environment variable                          │
  │                                                   │
  │  Stop when: plateau, time budget, 3 consecutive  │
  │  discards, or good enough (>5% speedup)          │
  └──────────────────────┬───────────────────────────┘
                         ▼
  ┌────────────────┐
  │  Final Report  │  Include GEAK results in "What Worked / Didn't"
  └────────────────┘
```

## Prerequisites

- **GPU**: AMD MI325X/MI355X (ROCm)
- **Training stack**: Primus/Megatron-LM installed in `/workspace/Primus/`
- **Container**: `rocm/primus-training-private` or equivalent with ROCm + PyTorch + Node.js
- **Config file**: YAML config for the model (e.g., `examples/megatron/configs/MI355X/gpt_oss_20B-BF16-pretrain.yaml`)
- **TraceLens CLI**: `pip install -e /hyperloom/TraceLens-internal` for automated profiling analysis
- **Optional MCP servers** in `.cursor/mcp.json`:
  - `geak-agent` — for kernel optimization (see `GEAK-KERNEL-OPTIMIZATION.md`)
- Environment: `GEAK_AUTH_KEY` and `LITELLM_API_KEY` set in `.env` (see `.env.template`)

## Quick Start

> **Tip:** Reference the skill file in your Cursor chat with `@.cursor/skills/training-optimization/SKILL.md` so the agent knows to follow it.

### Example: Optimize GPT-OSS 20B Training

Paste this into Cursor chat:

```
Optimize GPT-OSS 20B training on 8× MI355X using the training-optimization skill.

Container: rocm/primus-training-private:20260317_v26dot2_rc5
Config: examples/megatron/configs/MI355X/gpt_oss_20B-BF16-pretrain.yaml
Primus commit: e79d302c2f81db6416e5f2aff7254515e1d23dd0

Script: /root/example_workload.py
Hardware: 1× MI355X
Results: /tmp/gpt_oss_toy_results/

Try at least 8 optimization ideas. Write results to /shared_nfs/nehaprakriya/results/gpt_oss_primus/.
```

The agent will:
- Run baseline training (10 iterations, measure ms/iter from iter 6–10)
- Profile with PyTorch profiler, analyze kernel breakdown
- Try config overrides one at a time (`moe_permute_fusion`, `gradient_accumulation_fusion`, etc.)
- Try code patches (caching hot lookups, MoE dispatch changes, attention backend swaps)
- Keep improvements, revert regressions
- Stop after diminishing returns and write the optimization report

**Validated results** (GPT-OSS 20B on 8× MI355X): Baseline 13265.3 ms/iter → 13042.4 ms/iter (+1.68%) via config overrides + one code patch.

## What the Agent Tries

The agent picks from strategies like (non-exhaustive — it's creative):

| Category | Examples |
|----------|----------|
| Config overrides | `moe_permute_fusion=true`, `gradient_accumulation_fusion=true`, `moe_use_fused_router_with_aux_score=true` |
| Code patches | Cache `get_args()` in hot forward methods, MoE token dispatch rewrite |
| Attention backend | Toggle `use_sink_attention`, check `deterministic` flag |
| MoE dispatch | Fused permute kernels, pre-sorted tokens, batched operations |
| Environment | `PYTORCH_TUNABLEOP_ENABLED=1`, NCCL tuning vars |
| Kernel-level (GEAK) | Triton kernel tuning (block sizes, warps, pipelining) |

## Output

The agent produces:
- **`results.tsv`** — log of every attempt with ms/iter and keep/discard status
- **Optimization report** (`.md`) — executive summary, what worked, what didn't, kernel profile comparison, recommendations
- **Profiler traces** (`.json`) — Chrome traces in the training output directory

## Toy Example: GPT-OSS 20B MoE (Single-GPU Proxy)

This is a standalone single-GPU proxy of the GPT-OSS 20B MoE model for quick local testing or validating the GEAK kernel optimization flow. It approximates the real model's architecture (GQA attention via aiter, 8 experts with top-k=4, RMSNorm) but runs on a single GPU with fewer layers. Save as `example_workload.py`:

```python
#!/usr/bin/env python3
"""
GPT-OSS 20B MoE — single-GPU proxy for optimization testing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from aiter import flash_attn_func as aiter_fa

DEVICE = "cuda:0"
DTYPE = torch.bfloat16
HIDDEN = 2880
FFN_HIDDEN = 10944
NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
SEQ = 4096
BATCH = 2
NUM_LAYERS = 4
NUM_EXPERTS = 8
TOPK = 4
VOCAB = 100096

torch.cuda.set_device(0)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=DTYPE, device=DEVICE))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).to(DTYPE) * self.weight


class GQAAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_dim = NUM_HEADS * HEAD_DIM + 2 * NUM_KV_HEADS * HEAD_DIM
        self.qkv_proj = nn.Linear(HIDDEN, self.qkv_dim, bias=False, dtype=DTYPE, device=DEVICE)
        self.o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN, bias=False, dtype=DTYPE, device=DEVICE)
        self._q_end = NUM_HEADS * HEAD_DIM
        self._k_end = self._q_end + NUM_KV_HEADS * HEAD_DIM

    def forward(self, x):
        B, S, _ = x.shape
        qkv = self.qkv_proj(x)
        q = qkv[:, :, :self._q_end].view(B, S, NUM_HEADS, HEAD_DIM)
        k = qkv[:, :, self._q_end:self._k_end].view(B, S, NUM_KV_HEADS, HEAD_DIM)
        v = qkv[:, :, self._k_end:].view(B, S, NUM_KV_HEADS, HEAD_DIM)
        result = aiter_fa(q, k, v, causal=True, return_lse=True, deterministic=False)
        o = result[0].reshape(B, S, NUM_HEADS * HEAD_DIM)
        return self.o_proj(o)


class BatchedMoEFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(HIDDEN, NUM_EXPERTS, bias=False, dtype=DTYPE, device=DEVICE)
        self.gate_w = nn.Parameter(torch.randn(NUM_EXPERTS, HIDDEN, FFN_HIDDEN, dtype=DTYPE, device=DEVICE) * 0.01)
        self.up_w = nn.Parameter(torch.randn(NUM_EXPERTS, HIDDEN, FFN_HIDDEN, dtype=DTYPE, device=DEVICE) * 0.01)
        self.down_w = nn.Parameter(torch.randn(NUM_EXPERTS, FFN_HIDDEN, HIDDEN, dtype=DTYPE, device=DEVICE) * 0.01)
        self.shared_gate = nn.Linear(HIDDEN, FFN_HIDDEN, bias=False, dtype=DTYPE, device=DEVICE)
        self.shared_up = nn.Linear(HIDDEN, FFN_HIDDEN, bias=False, dtype=DTYPE, device=DEVICE)
        self.shared_down = nn.Linear(FFN_HIDDEN, HIDDEN, bias=False, dtype=DTYPE, device=DEVICE)

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(-1, D)
        M = x_flat.shape[0]
        logits = self.router(x_flat)
        scores = logits.float().softmax(dim=-1)
        topk_weights, topk_ids = torch.topk(scores, TOPK, dim=-1)
        topk_weights = topk_weights.to(DTYPE)
        flat_ids = topk_ids.view(-1)
        flat_weights = topk_weights.view(-1)
        flat_token_idx = torch.arange(M, device=DEVICE).unsqueeze(1).expand(-1, TOPK).reshape(-1)
        output = torch.zeros(M, D, dtype=DTYPE, device=DEVICE)
        for e in range(NUM_EXPERTS):
            mask = flat_ids == e
            if not mask.any():
                continue
            tok_idx = flat_token_idx[mask]
            w = flat_weights[mask].unsqueeze(1)
            tokens = x_flat[tok_idx]
            gate_out = F.silu(tokens @ self.gate_w[e])
            up_out = tokens @ self.up_w[e]
            expert_out = (gate_out * up_out) @ self.down_w[e]
            output.index_add_(0, tok_idx, w * expert_out)
        shared_gate = F.silu(self.shared_gate(x_flat))
        shared_up = self.shared_up(x_flat)
        shared_out = self.shared_down(shared_gate * shared_up)
        return (output + shared_out).view(B, S, D)


class GPTOSSBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_norm = RMSNorm(HIDDEN)
        self.attn = GQAAttention()
        self.ffn_norm = RMSNorm(HIDDEN)
        self.moe = BatchedMoEFFN()

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.moe(self.ffn_norm(x))
        return x


class GPTOSSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN, dtype=DTYPE, device=DEVICE)
        self.layers = nn.ModuleList([GPTOSSBlock() for _ in range(NUM_LAYERS)])
        self.norm = RMSNorm(HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False, dtype=DTYPE, device=DEVICE)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


def bench(warmup=5, iters=20):
    """Run fwd+bwd benchmark, return ms/iter."""
    model = GPTOSSModel()
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE)
    labels = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE)

    for _ in range(warmup):
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, VOCAB), labels.view(-1))
        loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, VOCAB), labels.view(-1))
        loss.backward()
        model.zero_grad()
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters
    print(f"ms_per_iter:{ms:.2f}")
    return ms


if __name__ == "__main__":
    bench()
```

Run it with:
```
python3 example_workload.py
```

**Expected results** (on MI355X): Baseline ~196 ms/iter → ~186 ms/iter (+4.9%) via torch.compile + fused QKV + pre-sorted MoE dispatch.

This toy example is useful for:
- Validating the GEAK kernel optimization flow on a single GPU
- Testing `torch.compile` and attention backend changes quickly
- Understanding the model architecture before running the full distributed training

For the **real optimization**, use the full Primus training stack as described below.

## Full Training Stack Example: GPT-OSS 20B on 8× MI355X

This is the real workflow — run inside a `rocm/primus-training-private` pod with 8 GPUs.

### 1. Setup

```bash
cd /workspace/Primus
git checkout <primus_commit>
git submodule update --init --recursive
pip install -r requirements.txt

# Compile C++ dataset helpers (required for mock data)
make -C third_party/Megatron-LM/megatron/core/datasets
```

### 2. Baseline run (10 iterations)

```bash
torchrun --nproc_per_node=8 --master_port=29500 \
  -m primus.cli.main train pretrain \
  --config examples/megatron/configs/MI355X/gpt_oss_20B-BF16-pretrain.yaml \
  profile=false use_pytorch_profiler=false \
  2>&1 | tee /tmp/baseline.log
```

Extract ms/iter from the log — average iterations 6–10 (skip warmup). For GPT-OSS 20B this was **13265.3 ms/iter**.

### 3. Profile baseline

```bash
torchrun --nproc_per_node=8 --master_port=29500 \
  -m primus.cli.main train pretrain \
  --config examples/megatron/configs/MI355X/gpt_oss_20B-BF16-pretrain.yaml \
  profile=true use_pytorch_profiler=true \
  profile_step_start=6 profile_step_end=7 \
  2>&1 | tee /tmp/profile.log
```

This produces a `.pt.trace.json` Chrome trace. Analyze the top-20 GPU kernels to identify where time is spent (GEMMs, attention, MoE dispatch, NCCL, etc.).

### 4. Optimization loop — apply one change at a time

Each attempt adds a config override or code patch on top of the running best. Example:

**Attempt 1** — enable fused MoE permutation:
```bash
torchrun --nproc_per_node=8 --master_port=29500 \
  -m primus.cli.main train pretrain \
  --config examples/megatron/configs/MI355X/gpt_oss_20B-BF16-pretrain.yaml \
  moe_permute_fusion=true \
  profile=false use_pytorch_profiler=false \
  2>&1 | tee /tmp/attempt1.log
```
Result: 13107.0 ms/iter → **+1.19% speedup** → KEEP

**Attempt 2** — add gradient accumulation fusion:
```bash
torchrun --nproc_per_node=8 --master_port=29500 \
  -m primus.cli.main train pretrain \
  --config examples/megatron/configs/MI355X/gpt_oss_20B-BF16-pretrain.yaml \
  moe_permute_fusion=true \
  gradient_accumulation_fusion=true \
  profile=false use_pytorch_profiler=false \
  2>&1 | tee /tmp/attempt2.log
```
Result: improved → KEEP. If it regressed → revert the new override and try something else.

Repeat for ~8 attempts: try an idea, measure, keep or revert, log to `results.tsv`.

### 5. Final optimized command (all kept overrides)

```bash
torchrun --nproc_per_node=8 --master_port=29500 \
  -m primus.cli.main train pretrain \
  --config examples/megatron/configs/MI355X/gpt_oss_20B-BF16-pretrain.yaml \
  moe_permute_fusion=true \
  moe_use_fused_router_with_aux_score=true \
  gradient_accumulation_fusion=true \
  profile=false use_pytorch_profiler=false \
  2>&1 | tee /tmp/final.log
```

Plus one code patch (cache `get_args()` in `PrimusTurboGroupedMLP.__init__`).

**Validated results:** 13265.3 → 13042.4 ms/iter (**+1.68% speedup**), 4 of 9 attempts kept.

### 6. Write report

The agent produces:
- `results.tsv` — every attempt with ms/iter and keep/discard
- `optimization_report.md` — executive summary, kernel profile comparison, recommendations

### Common pitfalls

- **Port conflicts**: if a previous run was killed, increment `--master_port` (29501, 29502, ...)
- **Mock data**: if `MockGPTDataset` fails, compile the C++ helpers (step 1)
- **TunableOp**: don't use during benchmarking — GEMM autotuning takes >30 min on MoE models
- **Sliding window**: Triton attention backend doesn't support `sink_sliding_window`, will crash

## Tips

- The agent stops automatically when it hits diminishing returns, crashes, or time budget
- You can interrupt anytime — it writes the report with whatever progress it has
- Config overrides are safest; code patches are more invasive but sometimes necessary
- The knowledge base in SKILL.md contains lessons from prior optimization runs — the agent uses these to avoid known pitfalls
