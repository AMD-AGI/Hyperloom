---
title: fusion — cutting HBM round-trips and hiding work behind a donor
kind: lever
lever: fusion
gens: [gfx950]
bottleneck: bandwidth-bound; also launch-bound decode
updated: 2026-08-28
---

# Fusion

## Route here when
- Two or more **bandwidth-bound** ops are chained through HBM (producer writes, consumer reads).
- A GEMM or attention kernel has a cheap epilogue/prologue running as a separate pass.
- Decode is **launch-bound**: GPU-busy time is draining into a tail of tiny ops.

**Classify first** (`lever_bottleneck_class.md`). Fusion pays only when it changes the *binding*
bottleneck. Fusing two compute-bound kernels usually just makes a bigger compute-bound kernel.

## Two different payoffs — know which one you are chasing

| | Traffic fusion | Launch fusion |
|---|---|---|
| Removes | an HBM round-trip | a kernel launch |
| Signal | HBM bytes per token | `launch_bound_share` |
| Typical target | norm+quant, epilogue→GEMM | decode's tiny-op tail |
| Measured by | memory counters | trace with **CUDA graphs OFF** |

They need different measurements and different candidate selection. Do not use one signal to justify
the other.

## Why it works

- **Removes an HBM pass.** Two BW-bound elementwise ops chained through memory read-twice/write-twice;
  fused they read once, write once. On the binding roof that is up to **2×**.
- **Donor latency hiding.** A GEMM's MFMA pipeline has spare VALU and memory cycles. Folding bias /
  activation / quant into the epilogue — or dequant / norm into the prologue — costs close to zero
  additional time.
- **Overlaps comm with compute.** A fused collective+norm lets the collective progress while the
  norm's VALU work runs.

## High-value fusions

| Fusion | Donor | Payoff |
|---|---|---|
| **epilogue → GEMM** (bias, activation, scale, fp8 quant) | GEMM | free epilogue, no C round-trip; pair with `OPTIMIZE_EPILOGUE=1` |
| **prologue → GEMM** (dequant / norm of A) | GEMM | removes a pre-pass over activations |
| **norm + quant** | both BW-bound | one pass; writes quantized output + scale together |
| **residual add + RMSNorm** | BW-bound chain | the dominant serving form of norm |
| **rope + KV-cache write** | BW/latency-bound | apply rope and write paged KV in one pass |
| **collective + norm** (all-reduce / all-gather + RMSNorm) | comm/compute overlap | hides collective latency |
| **MoE routing + dispatch** | latency-bound | fewer launches, less traffic |

**Good donors**: GEMM and attention (deep pipelines, spare cycles) · a norm pass (absorbs residual add,
quant, scale compute) · a copy/cast pass (absorbs quant or layout shuffle).

## When NOT to fuse

| Situation | Why it backfires |
|---|---|
| It displaces a **faster library kernel** | a hand-fused GEMM+epilogue that loses to the tuned library GEMM plus a cheap separate epilogue is a regression. The live lever is the aiter DB (`lever_autotune.md`). |
| It blows the **register / LDS budget** | extra fused state drops occupancy below the latency-hiding threshold (`lever_occupancy.md`) |
| It **destroys reuse** | fusing a high-reuse op into a streaming one can force recompute or extra traffic — recompute AI and re-classify |
| The ops want **different tile/grid shapes** | one launch geometry penalizes both |
| It crosses a **numerics boundary** | fusing across a needed FP32 accumulate or rescale point (`lever_numerics.md`) |
| It changes the GEMM's **dispatch signature** | a mismatched `bias` defeats the aiter 10-tuple lookup → 0 engagement, and you lose the tuned kernel entirely |

That last one is subtle and expensive: fusing a bias into a GEMM changes the `bias` field of the
lookup key. If your tuned CSV was captured with the old signature, engagement drops to zero and the
"fused" kernel is now competing against an *untuned* baseline.

## Launch-bound decode — the other kind

Once GEMM and attention are tuned, serving decode leaves a long tail of tiny ops (copy, elementwise,
rmsnorm, rope, activation, reduce, sample), each paying a full launch. Here the arithmetic is on
**launch count**, not traffic, and the donor is the chain itself.

**Capture the trace with CUDA graphs OFF.** With graphs on, the launches you are counting are already
amortized and the tail disappears.

**`launch_bound_share`** = fraction of GPU-busy time in those tiny-op categories (everything that is
not GEMM / attention / MoE).
- Below **0.10** → compute or attention dominated; no decode fusion will pay. Candidate floor.
- Measured decode fusions landed in the **low single digits** of e2e gain while their graphs-off shares
  were **0.25–0.45** — roughly a **0.13 discount**, because graph replay already removes most launch
  overhead. Use that as the prior when nothing better is available; a memory-traffic signal, when
  present, is the more accurate channel.
- Predicted gain below **3%** → not worth an authoring campaign.

Both numbers **rank** candidates rather than veto them. `launch_bound_share` is a poor discriminator
alone: a low share with a strong identified chain beats a high share with nothing fusible in it.

## Verify

| Check | How | Pass |
|---|---|---|
| Traffic actually dropped | HBM bytes per token, before/after | measurably lower |
| Class changed | re-run `lever_bottleneck_class.md` | the binding roof moved — if it didn't, the fusion bought nothing |
| Still engaged | `grep -c 'is tuned on cu_num'` if a GEMM signature changed | > 0 |
| Numerics | oracle vs unfused reference | within tolerance (`lever_numerics.md`) |
| e2e | fused vs staged, median of ≥3 warm non-overlapping runs | outside the noise band |

## Expected magnitude
Two chained BW-bound elementwise ops → one: approaching **2×** on those ops. Epilogue into a GEMM:
the epilogue's cost approaches **zero**. Decode launch fusion with graphs on: **low single-digit e2e**
— real but small; budget the campaign accordingly.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Fused, no gain | wasn't the binding bottleneck | re-classify before fusing |
| Fused GEMM slower than library | displaced a tuned kernel | keep the library GEMM; fuse elsewhere |
| Big traffic win, small e2e win | that op wasn't Amdahl-dominant | profile for the dominant op first |
| Megakernel spills | over-fusion | split it; check `.vgpr_count` |
| Engagement dropped to 0 after fusing | changed the GEMM dispatch signature | re-capture and re-tune (`lever_autotune.md`) |
| Decode fusion predicted 30%, delivered 2% | measured `launch_bound_share` with graphs off, deployed with graphs on | apply the ~0.13 discount up front |

## Deeper
`languages/fusion/` (the decode fusion pattern cards and CUDA-graph authoring rules) ·
`lever_bottleneck_class.md` (**classify first, re-classify after**) · `lever_autotune.md` ·
`lever_numerics.md` · `lever_occupancy.md`
