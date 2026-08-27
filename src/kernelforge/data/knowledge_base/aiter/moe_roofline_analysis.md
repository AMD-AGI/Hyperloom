# Fused MoE Roofline Analysis (bf16 act / blockscale-fp8 weights)

Extracted from `inference-testing` internal doc
(`docs/uplift_plan/roofline/fmoe_bf16_blockscale_fp8.md`, M. Hartikainen).
Target kernel: `aiter::fmoe_bf16_blockscaleFp8_g1u1_novs_silu_1tg_ps_32x256`.
Companion: [`moe_asm_kernel_patterns.md`](moe_asm_kernel_patterns.md).

## Scope and model

- Workload: DeepSeek-R1 FFN expert block on MI325X, TP=8.
- Model params after TP: `hidden=7168`, `gate_up_dim=512`, `down_dim=256`,
  `n_experts=256`, `top_k=8`.
- Dtypes: bf16 activations, fp8 (`Float8_e4m3fnuz`) weights, fp32 block scales.
- Block scale: **128×128** (confirmed from profiled `w2_scale` shape
  `[256, 56, 2]` → `7168/128=56, 256/128=2`). The `ps_32x256` in the kernel
  name is the **output tile** (32 tokens × 256 inter_dim), not the scale block.

## Headline finding: MoE is always memory-bound

Arithmetic intensity is **0.4–2.0 FLOPs/Byte** across all shapes. The kernel
never crosses the roofline ridge — weight HBM traffic dominates every regime.
This is the opposite intuition from dense GEMM.

## Counter-intuitive: efficiency DROPS with larger batches

Cache-thrashing dominates once weights stop fitting in L2.

| Batch | Time (μs) | BW (TB/s) | TFLOPs | Bytes (GB) | % peak BW |
|------:|----------:|----------:|-------:|-----------:|----------:|
| 480   | 353       | 4.05      | 120    | 1.43       | 67%       |
| 775   | 454       | 3.17      | 150    | 1.44       | 53%       |
| 1547  | 600       | 2.46      | 227    | 1.47       | 41%       |
| 2907  | 984       | 1.55      | 260    | 1.53       | 26%       |
| 5913  | 1677      | 0.99      | 311    | 1.65       | 16%       |
| 8613  | 2469      | 0.71      | 307    | 1.76       | 12%       |

Implied peak HBM BW on MI325X: ~6.0 TB/s.
Implied L2 working-set threshold: ~192 MB shared across CUs.

Root causes (stated in the doc):
1. L2 saturation — weights evicted before reuse
2. Intermediates grow with batch (5.9 MB → 106 MB)
3. Memory-controller congestion at large batches
4. Time scales **super-linearly**: 18× batch = 7× time

**Implication**: there is an optimal batch; past it, throughput collapses.

## Activated experts scaling (dominates bytes formula)

`activated_experts = min(batch × top_k, n_experts)`

| batch | activated | weight bytes (fp8) |
|------:|----------:|-------------------|
| 1     | 8         | 8/256 of all experts  |
| 16    | 128       | ½ of all experts      |
| 32+   | 256 (cap) | all experts loaded    |

Decode (`batch=1`) is efficient **per-token** because only 8/256 = 3% of
weights transfer. Above `batch=32` weight cost is fixed; only activations
grow — which is why BW plateaus then collapses.

## Memory and compute formulas

```
bytes_accessed =
    batch × hidden × 4                                   # hidden_states bf16 R+W
  + activated × gate_up_dim × hidden × 1                 # w1 fp8
  + activated × hidden × down_dim × 1                    # w2 fp8
  + activated × w1_scale_elts × 4                        # w1 scales fp32
  + activated × w2_scale_elts × 4                        # w2 scales fp32
  + batch × top_k × (gate_up_dim + down_dim) × 2         # intermediates bf16
  + batch × top_k × 8                                    # topk_weights + ids

total_flops =
    batch × top_k × hidden × gate_up_dim × 2             # gate+up
  + batch × top_k × down_dim × hidden × 2                # down
```

## Efficiency thresholds (per the doc)

- \>80% of roofline: good
- 60–80%: acceptable
- <60%: escalate (doc says "contact Amir's team")

batch=480 @ 67% sits right at the "acceptable" boundary; everything past
batch=1547 is below the escalation threshold.

## Worked example (batch=480)

```
activated_experts = min(480×8, 256) = 256      # all experts

hidden_states    = 480 × 7168 × 4    = 13.8 MB
w1               = 256 × 512 × 7168  = 939 MB
w2               = 256 × 7168 × 256  = 469 MB
w1_scale         = 256 × 4 × 56 × 4  = 0.23 MB
w2_scale         = 256 × 56 × 2 × 4  = 0.11 MB
intermediates    = 480 × 8 × 768 × 2 = 5.9 MB
topk             = 480 × 8 × 8       = 0.03 MB
total            = 1.43 GB

flops = 480 × 8 × 2 × 7168 × 768     = 42.3 GFLOPs
time  = 353 μs
BW    = 1.43 GB / 353 μs = 4.05 TB/s (matches profiled)
```

Weights = 1.41 GB of the 1.43 GB total — ~99%. **Weight transfer is
the only thing that matters** for this kernel on MI325X.

## Kernel name breakdown

`aiter::fmoe_bf16_blockscaleFp8_g1u1_novs_silu_1tg_ps_32x256`:

| Token | Meaning |
|---|---|
| `fmoe` | fused MoE |
| `bf16` | activation dtype in/out |
| `blockscaleFp8` | fp8 weights, block-wise fp32 scales |
| `g1u1` | gate:up fused 1:1 |
| `novs` | v-skip disabled |
| `silu` | gate · silu(up) |
| `1tg` | one thread group per expert tile |
| `ps_32x256` | persistent-scheduler; 32-token × 256-inter_dim output tile |

`g1u1` corresponds to the GEMM0+GEMM1 X-sharing described in the asm doc.

## Reproduction / measurement

Perfetto SQL query to extract profiled MoE calls is in Appendix H of the
source doc. Pattern: match `vllm::rocm_aiter_fused_moe` CPU slice to
`aiter::fmoe` GPU slice via timestamp overlap, pivot tensor arg dims.
Excel columns map to formulas in Appendix F.

## Takeaways for kernel optimization

1. **Any uplift work must cut weight HBM traffic** — FLOP-side optimizations
   have zero roofline headroom. Candidates: weight-compression past fp8 (fp6/fp4
   via `f8f6f4` MFMA, see [`../shared/gpu_arch_gfx950.md`](../shared/gpu_arch_gfx950.md)),
   better L2 residency (persistent kernels, expert-coresidency batching).
2. **Don't micro-optimize the high-batch path** — it is cache-thrashing, not
   kernel-inefficiency. Hunt at batch 480–1547 where the 40–67% regime lives.
3. **L2 ≈ 192 MB** is the binding budget on MI325X. 1.41 GB of weights is 7×
   larger, so weight reuse across tokens (not within an expert) is the only
   lever left.
4. The 67% peak BW at batch=480 suggests ~33% of HBM time is going to scale
   reads, intermediates, and scheduling overhead rather than weights. Worth
   measuring on MI350X (gfx950) where packed fp32 VOP3P and larger MFMA K
   change the compute/memory split only marginally (still memory-bound).
