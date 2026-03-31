# PRISM DFS Optimization — Complete Scoring Walkthrough

## Using GLM-5-FP8 on 8× AMD MI355X as a Concrete Example

**Document purpose:** Explain every detail of how the PRISM inference optimization skill constructs its search tree, assigns scores, picks branches, updates scores after results, backtracks, and discovers new branches. All formulas, all numbers, all decision points — traced through the real GLM-5 optimization that achieved +47.8%.

---

## Table of Contents

1. [The Core Idea](#1-the-core-idea)
2. [The Scoring Formula](#2-the-scoring-formula)
3. [Phase 0: Classification and Initial Score Assignment](#3-phase-0)
4. [Phase 1: Baseline + Profiling — Feeding the Heuristic](#4-phase-1)
5. [Phase 2: Building the Initial Action Stack](#5-phase-2)
6. [Phase 3: DFS Loop — Picking the First Branch](#6-phase-3)
7. [Phase 4: Score Updates After Backends](#7-phase-4)
8. [Phase 5: Combined Test — Super-Linear Synergy](#8-phase-5)
9. [Phase 6: Re-Scoring and Picking the Next Branch](#9-phase-6)
10. [Phase 7: GEMM Tuning — A New Branch Emerges](#10-phase-7)
11. [Phase 8: Kernel Optimization — LLM Proxy](#11-phase-8)
12. [Phase 9: Aggressive Exploration — Crashed Branches](#12-phase-9)
13. [Backtracking Mechanics](#13-backtracking)
14. [Adding New Branches at Runtime](#14-new-branches)
15. [Stopping Criteria and Transition to Sweep](#15-stopping)
16. [Full Score Evolution Table](#16-score-evolution)
17. [Comparison: GLM-5 vs Qwen3.5 Tree Shapes](#17-comparison)

---

## 1. The Core Idea <a name="1-the-core-idea"></a>

The PRISM inference optimization skill is a **single-agent sequential loop** that uses **depth-first search (DFS) with heuristic scoring** to decide which optimization action to try next.

It is NOT a fixed pipeline. The agent maintains a **priority stack** of candidate actions, each scored by a heuristic. After each action completes:

1. The action's result is measured (throughput change, crash, regression).
2. **All remaining candidates on the stack are re-scored** based on the new information.
3. The highest-scored candidate is popped next.
4. New sub-actions may be pushed onto the stack (discovered during execution).

This means the tree is **built dynamically** as the agent explores. The initial tree structure (CLASSIFY → BASELINE → PROFILE → {BACKENDS, PARAMS, KERNEL-OPT, SWEEP}) is a starting template, but the agent can add entirely new branches based on discoveries.

**Why DFS, not BFS or random?** DFS naturally explores the most promising branch deeply before backtracking. In inference optimization, this matters because:
- Optimizations compound (backend A + backend B may give 10× the sum of individual gains).
- Each action changes the performance landscape (a 20% speedup shifts which kernels are bottlenecks).
- The agent can stop early once it reaches diminishing returns on one branch.

---

## 2. The Scoring Formula <a name="2-the-scoring-formula"></a>

Every candidate action is scored by:

```
score = (expected_gain_per_gpu / cost_minutes)
        × (1 - accuracy_risk)
        × (1 - crash_risk)
        × target_gap_multiplier
```

### Component Breakdown

| Component | What It Measures | Range | Source |
|-----------|-----------------|-------|--------|
| `expected_gain_per_gpu` | How many tok/s/GPU this action might add | 0–100+ | KB lookup + model class priors + profiling data |
| `cost_minutes` | Estimated wall-clock time to execute | 2–120 min | Action-specific (restart + benchmark = ~5min, GEAK = ~15min) |
| `accuracy_risk` | Probability the optimization breaks numerical correctness | 0.0–1.0 | Action type (scheduling params = 0.0, kernel mods = 0.15, vendor kernels = 0.5) |
| `crash_risk` | Probability the server crashes | 0.0–1.0 | KB history + action type (vendor kernel mods = 0.5, scheduling = 0.05) |
| `target_gap_multiplier` | Urgency boost when chasing a competitor target | 1.0–2.0 | `1 + min(target_gap_pct, 100) / 100` |

### What This Formula Rewards

- **High expected gain, low cost** → high score (quick wins first)
- **Low risk** → multiplied score (safe actions preferred)
- **Large target gap** → everything boosted (more aggressive exploration)
- **An action with 20% expected gain, 5 min cost, 0 risk, 1.3× urgency:** `(20/5) × 1.0 × 1.0 × 1.3 = 5.2`
- **An action with 2% expected gain, 30 min cost, 0.5 crash risk:** `(2/30) × 1.0 × 0.5 = 0.033`

The first action is 150× more attractive than the second. This is how the agent avoids wasting time on low-probability experiments.

---

## 3. Phase 0: Classification and Initial Score Assignment <a name="3-phase-0"></a>

### What Happens

The agent reads `config.json` and classifies GLM-5-FP8:

```
Architecture: GlmMoeDsaForCausalLM
MoE: True (256 experts)
MLA: True (multi-latent attention, kv_lora_rank > 0)
NSA: True (native sparse attention — tilelang/aiter backends)
torch.compile: INCOMPATIBLE (NSA layers use custom CuteDSL/tilelang kernels)
```

**Classification result: `moe_mla_nsa`**

### Initial Score Priors

The classify action sets initial priors based on the model class. For `moe_mla_nsa`:

| Action | Initial Score | Rationale |
|--------|--------------|-----------|
| **Backends** | **10** | MoE+MLA+NSA models have multiple backend switches (attention, NSA prefill/decode, MoE runner, allreduce). Historical KB data shows backends are the #1 lever for this class. GLM-5 specifically has tilelang vs aiter for NSA, which is a known high-impact switch. |
| **Params** | 5 | Server params (decode-steps, mem-fraction, cuda-graph-bs) are moderate. They help but rarely exceed 5% individually. |
| **Kernel-opt (GEAK/LLM)** | 2 | Without torch.compile, there are no Inductor-generated Triton kernels for GEAK to optimize. Only framework-level Triton kernels are available, and most hot kernels are vendor CK/ASM code. Low expected surface area. |
| **torch.compile** | 0 | Incompatible with NSA architecture. Score is zero — this action will never be selected. |
| **Sweep** | 1 | Measurement-only action. Low priority until optimization is done. |

**These are not arbitrary numbers.** They come from the decision table in `classify.md`:

- Dense models: torch.compile=7, GEAK=8 (Inductor generates ~293 Triton kernels for GEAK)
- MoE+MLA: torch.compile=0 (MLA is incompatible), backends=9
- MoE+MLA+NSA: backends=10 (NSA adds extra backend switches beyond MLA), everything else low

### How Initial Scores Relate to the Formula

At this point, the full formula hasn't been applied yet — these are **base scores** that will be refined after profiling. Think of them as `expected_gain_per_gpu` estimates in tok/s units, where cost and risk will be factored in during stack construction (Phase 2).

---

## 4. Phase 1: Baseline + Profiling — Feeding the Heuristic <a name="4-phase-1"></a>

### Baseline

The agent launches the server with default config and measures:

```
Baseline: 1,379 tok/s (TP=4, CONC=64)
Per-GPU:  344.8 tok/s
TPOT:     87 ms
```

This number becomes `baseline_tput_per_gpu = 344.8`.

### Profiling (TraceLens Analysis)

The agent profiles the running server and gets kernel-level breakdown:

| Component | % GPU Time | Implication |
|-----------|-----------|-------------|
| AllReduce communication | 44.9% | 124 calls/step × ~200μs — latency-bound, not bandwidth-bound |
| GPU idle (scheduling gaps) | 49.0% | CPU scheduling between CUDA graph replays |
| MoE Router GEMM | 8.2% | Inside CUDA graph — potential GEMM tuning target |
| MoE Expert GEMM | 4.7% | Inside CUDA graph |
| Other compute | ~2% | Negligible |

### How Profiling Shapes Scores

The profile action updates the heuristic using these rules:

1. **>50% time in vendor kernels (Cijk_*, aiter::*):** "Boost backend exploration scores significantly." In GLM-5's case, aiter CK kernels and communication dominate. This confirms backends should be priority.

2. **GEAK candidate identification:** Any non-vendor kernel >3% GPU time is a candidate. For GLM-5:
   - MoE Router GEMM: 8.2% — but this is an aiter Triton GEMM, not a pure vendor kernel. It's a potential tuning target.
   - Most other hot kernels are vendor CK/ASM → not GEAK candidates.
   
3. **Per-candidate scoring formula from profile.md:**
   ```
   candidate_score = gpu_pct × expected_speedup_for_type / cost_minutes × (1 - accuracy_risk)
   ```
   For the MoE router GEMM (Triton template kernel):
   - `gpu_pct = 8.2`
   - `expected_speedup = 0.1` (template/GEMM kernels — Inductor autotuner usually near-optimal)
   - `cost_minutes = 15`
   - `accuracy_risk = 0.05`
   - `crash_risk = 0.5` (GEMM changes can crash)
   - Score: `8.2 × 0.1 / 15 × (1 - 0.05) × (1 - 0.5) = 0.026`
   
   This is very low — which is why GEMM tuning started with score=2 and wasn't prioritized initially.

### Target Analysis (NVIDIA B200)

The agent also reads NVIDIA B200 results: 660 tok/s/GPU. Our baseline is 344.8. Gap:

```
target_gap_pct = (660 - 344.8) / 660 × 100 = 47.8%
```

This is a **LARGE** gap (>30%), so `target_gap_multiplier = 1 + 47.8/100 = 1.478`.

All action scores get multiplied by 1.478. This makes the agent explore more aggressively.

---

## 5. Phase 2: Building the Initial Action Stack <a name="5-phase-2"></a>

After classification, baseline, and profiling, the agent constructs the action stack. Each candidate gets its full score computed:

### Full Score Calculation for Each Action

**Backends (base=10):**
```
expected_gain = 10 tok/s/GPU estimate (KB says backends typically +5–20% for MoE+MLA)
cost = 5 min per backend test, ~30 min for full exploration
accuracy_risk = 0.1 (backend switches change code paths)
crash_risk = 0.1 (most backends either work or crash immediately)
target_gap_multiplier = 1.478

score = (10 / 5) × (1 - 0.1) × (1 - 0.1) × 1.478
       = 2.0 × 0.81 × 1.478
       = 2.39
```

**Params (base=5):**
```
expected_gain = 5 tok/s/GPU estimate
cost = 5 min per param
accuracy_risk = 0.0 (scheduling params)
crash_risk = 0.05
target_gap_multiplier = 1.478

score = (5 / 5) × 1.0 × 0.95 × 1.478
       = 1.0 × 0.95 × 1.478
       = 1.404
```

**Kernel-opt (base=2):**
```
expected_gain = 2 tok/s/GPU (limited surface without torch.compile)
cost = 15 min (GEAK/LLM round-trip)
accuracy_risk = 0.15 (kernel modifications)
crash_risk = 0.2
target_gap_multiplier = 1.478

score = (2 / 15) × 0.85 × 0.8 × 1.478
       = 0.133 × 0.68 × 1.478
       = 0.134
```

**torch.compile (base=0):**
```
score = 0 (incompatible — never selected)
```

### The Priority Stack (Sorted High→Low)

```
STACK:
  1. backends     score=2.39  ← WILL BE POPPED FIRST
  2. params       score=1.40
  3. kernel-opt   score=0.13
  4. sweep        score=0.07
  5. torch.compile score=0.00
```

**The agent pops `backends` first** because it has the highest score. This is why backend exploration happened before param tuning or kernel optimization in the GLM-5 run.

---

## 6. Phase 3: DFS Loop — Picking the First Branch (Backends) <a name="6-phase-3"></a>

The agent pops `backends` from the stack and executes `actions/backends.md`.

### How Backends are Explored

Backends are tested in a **tiered** order (Tier 1 = highest expected per-switch impact):

**Tier 1 — Attention/Decode backend switches** (change actual GPU kernels):
- `--nsa-decode-backend aiter` (replace Triton decode attention with CK kernels)
- `--nsa-prefill-backend aiter` (replace tilelang prefill with CK)

**Tier 2 — Scheduling modes** (change batching behavior):
- `--enable-mixed-chunk` (mix prefill + decode in same batch)

**Tier 3 — Compute fusion** (fuse adjacent operations):
- `--enable-aiter-allreduce-fusion` (fuse allreduce + RMSNorm + quant)

**Tier 4 — MoE/GEMM backend switches**
- `SGLANG_ROCM_FUSED_DECODE_MLA=1` (fused MLA decode)

**Tier 5 — Communication**
- Custom allreduce, NCCL channel tuning

### Individual Backend Results

Each backend is tested independently against the baseline:

| Backend | Throughput | vs Baseline | Status |
|---------|-----------|-------------|--------|
| aiter NSA decode | 1,446 tok/s | +3.1% | **WINNER** (>1%) |
| mixed-chunk | 1,444 tok/s | +2.9% | **WINNER** |
| allreduce fusion | 1,407 tok/s | +0.3% | NEUTRAL (<1%) |
| fused MoE | 1,409 tok/s | +0.4% | NEUTRAL |
| FUSED_DECODE_MLA | 1,403 tok/s | +0.0% | NEUTRAL |

### The Combination Rule

**CRITICAL:** The skill mandates testing ALL winners combined, because:

> "Individual gains do NOT predict combined gains — switches affecting different pipeline stages produce super-linear synergy (validated: GLM-5 +3.1% + +2.9% → +16.2% combined)."

The agent identifies 2 clear winners (aiter decode +3.1%, mixed-chunk +2.9%) and one borderline (allreduce fusion +0.3%). It tests all three together.

---

## 7. Phase 4: Score Updates After Individual Backends <a name="7-phase-4"></a>

After each individual backend test, the heuristic updates:

### Update Rule: "Action succeeded (gain > 0%)"

> "Boost similar actions. E.g., if backends gained +5%, boost remaining untested backends by 1.5×. Boost combined_test score."

After aiter decode (+3.1%):
- Remaining backend candidates get 1.5× boost
- `combined_test` is pushed with boosted score

After mixed-chunk (+2.9%):
- Another 1.5× boost on remaining candidates
- `combined_test` score updated: `sum(individual_scores) × 1.5`

### Update Rule: "After 2+ backend wins"

> "Push combined_backends_test with score = sum(individual scores) × 1.5"

The combined test gets pushed onto the stack:
```
combined_test_score = (score_aiter + score_mixed_chunk + score_allreduce_fusion) × 1.5
                    = (3.1 + 2.9 + 0.3) × 1.5  (using gain% as proxy for score)
                    = 9.45
```

This is now the **highest-scored item on the stack**, so it's popped next.

### Update Rule: Individual backends <1% (the neutrals)

> "If individual backends all <1%: reduce remaining backend scores, boost param tuning."

FUSED_DECODE_MLA and fused MoE both showed 0%. This slightly reduces scores for similar untested backends.

---

## 8. Phase 5: Combined Test — Super-Linear Synergy <a name="8-phase-5"></a>

The agent tests all three winners together:

```
aiter decode + mixed-chunk + allreduce fusion = 1,981 tok/s (+41.2%)
```

Individual: +3.1% + +2.9% + +0.3% = +6.3% (additive prediction)
Combined: +41.2% (**6.5× the additive prediction**)

### Why Super-Linear?

The three optimizations attack different parts of the decode pipeline:

1. **aiter decode** makes attention kernels faster (compute reduction)
2. **mixed-chunk** eliminates GPU idle between prefill/decode phases (scheduling)
3. **allreduce fusion** hides communication behind compute (overlapping)

When all three are active simultaneously:
- Faster attention means each decode step finishes sooner → more decode steps per second
- Mixed-chunk keeps the GPU busy during what used to be idle → GPU utilization jumps
- Allreduce fusion overlaps communication with the now-faster compute → near-zero communication stalls

The idle time, communication overhead, and compute latency were all hiding each other. Fixing all three simultaneously exposes the full pipeline throughput.

### Score Update After Combined Success

This is a massive result (+41.2%). The update rules fire:

1. **Backends completed:** All tested. Push `re-profile` to discover new bottlenecks.
2. **Gain > 25%:** The stopping criterion for "cumulative gain > 25%" is met, but the agent doesn't stop because the target gap is still large (we had 1,981 tok/s vs B200's ~2,640 tok/s for 4 GPUs).
3. **Re-score everything:** With the new baseline of 1,981 tok/s, the expected gains from other actions shift. Kernel-opt on a kernel that was 8.2% of a slower baseline is now a smaller absolute gain.

### Updated Stack After Combined Test

```
STACK:
  1. re-profile   score=high (mandatory after backend wins)
  2. params       score=1.40 (unchanged — hasn't been tried yet)
  3. kernel-opt   score=0.13 (will be re-scored after re-profiling)
  4. sweep        score=0.07
```

---

## 9. Phase 6: Re-Scoring and Picking the Next Branch <a name="9-phase-6"></a>

### After Re-Profiling

The new profile (at 1,981 tok/s) shows the same basic picture:
- AllReduce still dominates (~45%)
- GPU idle still ~49%
- MoE Router GEMM still ~8%

The bottleneck hasn't shifted — it was communication-bound before and still is. This means:

### Re-Scoring All Candidates

**Params (score was 1.40):**
The expected gain is now relative to 1,981 tok/s, not 1,379. Scheduling params that might have given 5% on the old baseline now give 5% on a higher number — so the absolute gain is larger, but the percentage is the same. Score stays roughly the same.

**Kernel-opt (score was 0.13):**
Profile shows MoE Router GEMM still at 8.2% — but the baseline is now 1,981 tok/s, so optimizing that 8.2% could yield a larger absolute gain. However, the profiling update rule says:

> "Re-profiled candidates rescored from new gpu_pct"

The GPU% is unchanged, so the kernel-opt score doesn't change much. It stays low.

**GEMM Tuning (not yet on the stack):**
This is where things get interesting. The agent hasn't yet discovered the missing GEMM config. It will come later.

### Stack Order

```
STACK:
  1. params       score=1.40  ← POPPED NEXT
  2. kernel-opt   score=0.13
  3. sweep        score=0.07
```

### Params Exploration

The agent tests server parameters on top of the winning backend config:

| Parameter | Throughput | vs Backend Baseline | Status |
|-----------|-----------|-------------------|--------|
| decode-steps=8 | 1,981 tok/s | 0% | NEUTRAL — **discovered to be a NO-OP** |
| decode-steps=16 | 1,981 tok/s | 0% | NEUTRAL |
| mem-fraction=0.90 | 1,981 tok/s | <0.5% | NEUTRAL |
| schedule-conserv=0.3 | 1,981 tok/s | 0% | NEUTRAL |
| cuda-graph-max-bs=64 | 1,981 tok/s | 0% (already set) | KEEP — but critical (32 or 128 → -50%) |
| QR INT4 quantization | ~2,010 tok/s | +1–2% | **KEEP** |

### Score Update After Params

> "All params <1%: model is already well-tuned, proceed to kernel optimization or sweep."

The params action produced almost nothing. This triggers:
- Reduce remaining param-type scores by 0.5×
- Slightly boost kernel-opt and sweep scores (the "well-tuned" signal means compute is the remaining frontier)

Updated stack:
```
STACK:
  1. kernel-opt   score=0.13 × 1.2 (slight boost) = 0.16
  2. sweep        score=0.07 × 1.2 = 0.08
```

---

## 10. Phase 7: GEMM Tuning — A New Branch Emerges <a name="10-phase-7"></a>

### Discovery: The Missing Config

This is a critical example of **the agent adding new branches at runtime**.

During v7 (after a node regression to 1,704 tok/s from node state issues), the agent investigated WHY performance was lower. While examining the aiter GEMM tuning infrastructure, it discovered:

**The MoE router GEMM shape (N=256, K=6144) had no tuning config.** The aiter Triton GEMM library (`_gemm_a16_w16_atomic_kernel`) looks up tuning configs in JSON files at `/sgl-workspace/aiter/aiter/ops/triton/configs/gemm/`. For the specific shape `gfx950-GEMM-A16W16-ATOMIC-N=256-K=6144.json`, **no file existed**.

Without a config, the kernel uses a default tile configuration that leaves most of the 256 Compute Units idle.

### How the New Branch Gets Scored

The agent creates an ad-hoc action "GEMM Router Tuning" and scores it:

```
expected_gain = MoE Router is 8.2% of total GPU time.
                Router runs once per layer × 78 layers.
                With ksplit=24, each CU group processes one K-split.
                Expected speedup of the router GEMM: ~60% (from profiling the default vs tuned).
                E2E expected: 8.2% × 60% ≈ 5% of total throughput.

cost_minutes = 2 min (create JSON file + restart server)
accuracy_risk = 0.05 (GEMM tuning is numerically equivalent — same math, different tiling)
crash_risk = 0.1 (bad tile config could OOM or deadlock)
target_gap_multiplier = 1.478 (still chasing B200)

score = (5 / 2) × (1 - 0.05) × (1 - 0.1) × 1.478
       = 2.5 × 0.855 × 1.478
       = 3.16
```

This score of **3.16** is higher than anything else on the stack. It gets pushed and immediately popped.

### Result: +21.4%

The actual result was 1,704 → 2,070 tok/s (+21.4%) — massively exceeding the 5% estimate.

### Score Update After GEMM Success

This is the largest single optimization in the entire run. The update rules fire:

> "Kept kernel: boost similar kernel type scores"

- All GEMM-related tuning candidates get boosted. The agent checks if there are other missing GEMM configs.
- The agent tunes all 55 dense GEMM shapes → zero decode impact (bandwidth-bound at small M). Score for dense GEMM tuning drops.

> "Score jumped from 2→9" for the GEMM tuning category.

This is reflected in the tree: the GEMM node's score annotation changes from 2 to 9, showing how the heuristic adapted after the discovery.

---

## 11. Phase 8: Kernel Optimization — LLM Proxy <a name="11-phase-8"></a>

### Why LLM Proxy, Not GEAK?

The kernel-opt action has two backends: GEAK (remote GPU pod) and LLM proxy (Claude/GPT via PRISM gateway). For GLM-5:

- **GEAK is blocked** because torch.compile is incompatible with NSA. Without torch.compile, there are no Inductor-generated standalone Triton kernel files for GEAK to optimize.
- **LLM proxy can still optimize framework-level Triton kernels** that exist in the aiter/SGLang source code.

### What Was Optimized

The agent identified `_per_token_group_quant_8bit` — the FP8 quantization kernel called on every linear/MoE input. It's a Triton kernel in the aiter source. The LLM proxy (Claude Opus) generated an optimized version:

1. `eviction_policy="evict_first"` — streaming data should not pollute L2 cache
2. `tl.math.fast_dividef` — hardware reciprocal instead of full division
3. Pre-compute `y_abs = tl.abs(y)` — avoid recomputation in reduction
4. CSE on `y_off` — compute once instead of twice

### Scoring

```
expected_gain = FP8 quant kernel at ~1% GPU time, expected speedup ~50%
                → ~0.5% E2E
cost_minutes = 1 min (LLM proxy is fast, <30 seconds for simple kernels)
accuracy_risk = 0.15 (reduction kernel modification)
crash_risk = 0.1

score = (0.5 / 1) × 0.85 × 0.9 × 1.478
       = 0.5 × 0.765 × 1.478
       = 0.565
```

### Result: +0.5%

Measured E2E: +0.5%. Matched the estimate exactly. **KEEP.**

The accuracy gate passed (output text identical to reference). The kernel was patched into the framework source using AST-based replacement (Strategy B from `integrate.md`).

---

## 12. Phase 9: Aggressive Exploration — Crashed Branches <a name="12-phase-9"></a>

After the main optimization loop, the agent entered an "aggressive" phase trying to close the remaining ~23% gap with B200.

### fMoE Kernel Modifications

| Experiment | Result | Score After |
|-----------|--------|-------------|
| fMoE 2-stage CK path | -76% regression | crash_risk → 1.0, score → 0 |
| fMoE BLOCK_SIZE_M=64 | GPU crash | crash_risk → 1.0, score → 0 |
| fMoE BLOCK_SIZE_M=128 | GPU crash | crash_risk → 1.0, score → 0 |

After 2+ crashes on vendor-type kernels, the update rule fires:

> "After 2+ discards on vendor-type kernels: reduce all GEAK scores to near-zero"

This effectively kills the entire kernel modification branch for vendor kernels. The agent has learned that the 1-stage ASM kernel has hardcoded register allocation and cannot be modified externally.

### RCCL/NCCL Experiments

| Experiment | Result | Score After |
|-----------|--------|-------------|
| NCCL_PROTO=LL128 | RCCL crash | crash_risk → 1.0, score → 0 |
| RCCL_MSCCL_ENABLE=1 | Server crash | crash_risk → 1.0, score → 0 |

These also crash, confirming that communication optimization on this stack is hardware-limited.

### Score Impact of Crashes

The crash risk update is permanent for that specific action:

```
Before: RCCL tuning score = some_value
After crash: crash_risk = 1.0
            score = ... × (1 - 1.0) = 0
```

A crashed action can **never** be selected again (score = 0).

---

## 13. Backtracking Mechanics <a name="13-backtracking"></a>

Backtracking in PRISM's DFS is **implicit through re-scoring**, not explicit tree traversal.

### How It Works

1. Agent explores backends deeply → combined test → +41.2%.
2. Agent re-scores everything. Params now has the highest score (1.40 vs kernel-opt 0.13).
3. Agent "backtracks" from the backends branch and explores params.
4. Params yields <1% → re-score. Kernel-opt is now highest.
5. Agent "backtracks" from params and explores kernel-opt.

There is no explicit "undo" or "go up a level." The stack is a flat priority queue. Whichever action has the highest score after re-scoring gets popped next, regardless of which branch it belongs to.

### When Does Real Backtracking Happen?

Real backtracking (reverting changes) happens in two cases:

1. **Accuracy gate failure:** The optimization is immediately reverted. The agent restores `.bak` files and marks the action as FAIL. The score for that action and similar actions drops.

2. **Regression detected:** If throughput decreases after an action, the agent reverts:
   ```python
   actual_e2e = (new_tput - baseline_tput) / baseline_tput * 100
   if actual_e2e <= 0:
       revert_all_bak_files()  # Restore original code
       status = "REVERT"
   ```

### Example: A8W8 Triton for MLA GEMMs

The agent tried routing MLA attention GEMMs to Triton W8A8 kernels. Result: -5.1%.

1. Throughput dropped → **REVERT**.
2. Original CK kernels restored from `.bak` files.
3. Triton-for-MLA score reduced by 0.5× (failure dampening).
4. Similar actions (other Triton-for-CK substitutions) reduced by 0.7×.

---

## 14. Adding New Branches at Runtime <a name="14-new-branches"></a>

The skill explicitly states:

> "The agent is NOT limited to the pre-defined actions. If profiling reveals an unexpected bottleneck or a KB query suggests a novel technique, the agent can create ad-hoc actions and score them with the same heuristic."

### Examples from GLM-5

**1. GEMM Router Tuning (Phase 7)**

This branch did not exist in the initial tree. The agent discovered the missing config while investigating performance variance. It created a new action, scored it at 3.16 (higher than anything on the stack), and executed it immediately. Result: +21.4%.

**2. Combined Backend Test**

After 2+ individual backend wins, the rule automatically pushes a new action:

> "After 2+ backend wins: push combined_backends_test with score = sum(individual scores) × 1.5"

This is a **rule-based branch creation** — the tree grows automatically when certain conditions are met.

**3. Re-Profile After Keep**

After any kept optimization, a re-profile action is pushed:

> "After kernel opt kept: push re-profile + next-kernel with boosted score"

This discovers new GEAK candidates that may have emerged due to the changed performance landscape.

**4. FP8 Quant Kernel Optimization**

The agent discovered that `_per_token_group_quant_8bit` was a framework-level Triton kernel (not vendor) during source code exploration. It created an ad-hoc kernel-opt action for the LLM proxy to optimize it. This wasn't in any pre-defined list — the agent found it by grepping the aiter source for `@triton.jit`.

### How New Branches Get Scored

New branches use the same formula as everything else:

```
score = (expected_gain / cost) × (1 - accuracy_risk) × (1 - crash_risk) × target_gap_multiplier
```

The agent estimates `expected_gain` from:
- Profiling data (GPU time % of the target kernel)
- KB lookup (historical gains for similar actions on similar models)
- Architecture reasoning (e.g., "this GEMM runs 78 times per forward pass")

If the estimated score is higher than everything on the stack, the new branch is popped immediately. If lower, it waits its turn.

---

## 15. Stopping Criteria and Transition to Sweep <a name="15-stopping"></a>

The DFS loop stops when any of these conditions are met:

| Condition | Threshold | GLM-5 Status |
|-----------|----------|--------------|
| All action scores < 1.0 | All remaining candidates scored below 1.0 | **MET** — after aggressive phase, all remaining scores were near 0 |
| Cumulative gain > 25% | Total improvement exceeds 25% | **MET** — +47.8% |
| 5 consecutive discards | Five actions in a row with gain ≤ 0% | Nearly met — 4 consecutive neutrals/regressions in aggressive phase |
| Wall clock > 180 min | Total optimization time exceeds 3 hours | Not met — but across multiple sessions |
| Target exceeded (gap ≤ 0%) | Our throughput exceeds the target | Not met — still 23% behind B200 |
| 2+ server crashes | Two or more crashes in a session | MET during aggressive phase (fMoE + RCCL crashes) |

For GLM-5, the agent stopped because:
1. All action scores dropped below 1.0 (backends exhausted, params exhausted, kernel-opt marginal, GEMM tuning done, aggressive mods crashed).
2. Cumulative gain exceeded 25%.
3. Multiple crashes in the aggressive phase triggered the emergency stop criterion.

After stopping, the agent transitioned to:
- **Sweep:** Full ISL/OSL/CONC parameter sweep with the optimized config.
- **Report:** Generated optimization report with all history, Pareto curves, and B200 comparison.

---

## 16. Full Score Evolution Table <a name="16-score-evolution"></a>

This table traces the score of every action through the entire GLM-5 optimization:

| Phase | Event | backends | params | kernel-opt | GEMM tuning | sweep | Notes |
|-------|-------|----------|--------|------------|-------------|-------|-------|
| **0** | Initial priors (classify) | **10** | 5 | 2 | — | 1 | Based on moe_mla_nsa class |
| **1** | After profiling | **10** | 5 | 2 | — | 1 | Profile confirms communication-bound → backends stay high |
| **1** | Target gap applied (×1.478) | **14.8** | 7.4 | 3.0 | — | 1.5 | B200 gap boosts everything |
| **2** | Stack built (with cost/risk) | **2.39** | 1.40 | 0.13 | — | 0.07 | Full formula applied. Backends popped first. |
| **3a** | aiter decode wins (+3.1%) | 2.39→**3.6** | 1.40 | 0.13 | — | 0.07 | 1.5× boost on remaining backends |
| **3b** | mixed-chunk wins (+2.9%) | 3.6→**5.4** | 1.40 | 0.13 | — | 0.07 | Another 1.5× boost |
| **3c** | combined_test pushed | — | 1.40 | 0.13 | — | 0.07 | `combined = sum × 1.5 = 9.45` pushed |
| **3d** | Combined test wins (+41.2%) | **done** | 1.40 | 0.13 | — | 0.07 | Backends branch complete. Push re-profile. |
| **4** | Re-profile (same picture) | done | **1.40** | 0.13→0.16 | — | 0.08 | Kernel-opt slightly boosted |
| **5** | Params explored (<1%) | done | **done (→0.7)** | 0.16→0.19 | — | 0.10 | All params neutral → reduce params, slight boost kernel |
| **6** | GEMM config discovered | done | done | 0.19 | **3.16** (new!) | 0.10 | New branch pushed, immediately popped |
| **7** | GEMM tuning wins (+21.4%) | done | done | 0.19 | **done (9)** | 0.10 | Massive win. Boost similar GEMM actions. |
| **7b** | Dense GEMM tuning (0%) | done | done | 0.19 | sub-done | 0.10 | No effect → reduce dense GEMM score |
| **7c** | A8W8 Triton (-5.1%) | done | done | 0.19→0.13 | done | 0.10 | Regression → reduce kernel substitution scores |
| **8** | FP8 quant kernel (LLM, +0.5%) | done | done | **done (0.57)** | done | 0.10 | Small win. Score logged but action complete. |
| **9a** | fMoE 2-stage (-76%) | done | done | **→0** | done | 0.10 | Regression → reduce vendor kernel mod scores |
| **9b** | fMoE block_m=64 (CRASH) | done | done | **0** | done | 0.10 | Crash → crash_risk=1.0 on this type |
| **9c** | RCCL LL128 (CRASH) | done | done | 0 | done | 0.10 | 2+ crashes → emergency stop consideration |
| **10** | All scores < 1.0 | done | done | 0 | done | **0.10** | Stopping criterion met. Proceed to sweep. |

---

## 17. Comparison: GLM-5 vs Qwen3.5 Tree Shapes <a name="17-comparison"></a>

The two models show dramatically different tree shapes because classification routes them through different paths:

### GLM-5 (moe_mla_nsa)

```
Tree shape: WIDE then DEEP on backends branch

CLASSIFY → BASELINE → PROFILE →
   ├── BACKENDS (score 10) → aiter, mixed-chunk, allreduce → COMBINED (+41.2%) ← DEEP
   ├── PARAMS (score 5) → all neutral, cuda-graph critical
   ├── GEMM TUNING (score 2→9) → router ksplit=24 (+21.4%) ← NEW BRANCH
   ├── KERNEL-OPT (score 2) → LLM proxy FP8 quant (+0.5%)
   ├── fMoE mods → CRASH
   └── RCCL tuning → CRASH
```

The tree went deep into backends first (3 sub-branches + combined test), then wide across params/GEMM/kernel-opt. The GEMM branch was added at runtime.

### Qwen3.5 (moe_hybrid_attention)

```
Tree shape: NARROW — backends branch dies immediately

CLASSIFY → PATCHES → BASELINE →
   ├── BACKENDS (score 8→0) → aiter CRASH, alter CRASH ← BRANCH KILLED
   ├── PARAMS (score 7→8) → decode-8, mixed-chunk, mem 0.85 (+4.7%) ← WINNER
   ├── FP8 KV → WORSE
   ├── EP-8 → -2 to -6%
   ├── TP-4 DP-2 → WORSE
   ├── GEMM tuning → untested (amber)
   ├── kernel-opt → untested (amber)
   ├── torch.compile → untested (amber, score=0)
   └── MTP → untested (amber, score=0)
```

The backends branch crashed immediately (hybrid attention is incompatible), which triggered:

> "If backends produce crashes: crash_risk=1.0 for that backend. If no backends available: skip to params."

This collapsed the backends score from 8 to 0 and elevated params to the primary strategy. The tree is narrow because the most impactful branch (backends) was pruned by crashes, and several other branches remain untested.

### Key Structural Differences

| Aspect | GLM-5 | Qwen3.5 |
|--------|-------|---------|
| Deepest branch | Backends (4 levels deep: individual → combined → re-profile → next) | Params (2 levels: individual → combined) |
| Widest level | 6 branches from PROFILE (backends, params, GEMM, kernel, fMoE, RCCL) | 9 branches from BASELINE (backends, params, FP8, EP, TP-4, GEMM, kernel, compile, MTP) |
| New branches added | 2 (GEMM tuning, combined test) | 0 (no discoveries — backends crashed before exploration could deepen) |
| Crashed branches | 2 (fMoE, RCCL) | 3 (aiter, alter, backends category) |
| Untested branches | 1 (GEAK) | 4 (GEMM, kernel, compile, MTP) — significant upside remaining |
| Total improvement | +47.8% | +39.7% |
| Primary lever | Backends (3-way combo) | Params (scheduling) |
| Secondary lever | GEMM tuning (router ksplit) | Container/ROCm version advantage (baseline already +19.6%) |

---

## Appendix: The Complete Action Stack Over Time (GLM-5)

```
T=0  [backends=2.39, params=1.40, kernel-opt=0.13, sweep=0.07]
     POP → backends

T=1  [combined_test=9.45, params=1.40, kernel-opt=0.13, sweep=0.07]
     (after individual backend tests, combined pushed)
     POP → combined_test

T=2  [re-profile=HIGH, params=1.40, kernel-opt=0.16, sweep=0.08]
     (after +41.2% combined win)
     POP → re-profile

T=3  [params=1.40, kernel-opt=0.16, sweep=0.08]
     POP → params

T=4  [kernel-opt=0.19, sweep=0.10]
     (params done, all <1%, slight boost to remaining)
     — GEMM tuning discovered externally —
     PUSH → [GEMM=3.16, kernel-opt=0.19, sweep=0.10]
     POP → GEMM tuning

T=5  [dense-GEMM=0.5, A8W8=0.3, kernel-opt=0.19, sweep=0.10]
     (sub-branches from GEMM success)
     POP → dense-GEMM (0%)
     POP → A8W8 (-5.1%, REVERT)
     POP → kernel-opt

T=6  [FP8-quant-LLM=0.57, GEAK=0.01, sweep=0.10]
     POP → FP8-quant-LLM (+0.5%, KEEP)
     POP → GEAK (untested, score too low)

T=7  [fMoE-mods=0.3, RCCL=0.2, sweep=0.10]
     (aggressive phase)
     POP → fMoE-mods (CRASH, -76%)
     POP → fMoE-block-m (CRASH)
     POP → RCCL (CRASH)

T=8  [sweep=0.10]
     All remaining scores < 1.0. STOP.
     POP → sweep (measurement only)
     → REPORT
```

---

*This document describes the PRISM inference optimization skill's DFS orchestrator as implemented in `/shared_nfs/nehaprakriya/PRISM/.cursor/skills/inference-optimization/SKILL.md` and its action modules, traced through the GLM-5-FP8 optimization run of March 25–30, 2026.*
