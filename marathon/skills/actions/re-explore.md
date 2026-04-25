# Action: Re-Explore — Plateau Breaking

**DFS role:** Triggered when the DFS loop detects a plateau (3+ consecutive discards or
all scores < 2.0 with significant gap remaining). Injects novelty to escape local optima.

## When to Trigger

| Condition | Trigger |
|-----------|---------|
| 3+ consecutive discards | Automatic |
| All action scores < 2.0 AND target_gap_pct > 10% | Automatic |
| Same action pattern repeated 3x with same result | Loop detection |
| Agent explicitly recognizes plateau | Manual |

## Loop Detection

Detect when the agent is cycling through the same actions without progress.

```python
import hashlib

def compute_loop_signature(action_name, params, result_status):
    """Deterministic signature for detecting repeated action patterns."""
    sig_str = f"{action_name}|{sorted(params.items()) if params else ''}|{result_status}"
    return hashlib.md5(sig_str.encode()).hexdigest()[:12]

def detect_loop(state, window=5):
    """Check if recent actions form a repeating pattern."""
    recent = state["completed_actions"][-window:]
    signatures = [compute_loop_signature(a["action"], a.get("params", {}), a["status"])
                  for a in recent]

    # Check for exact repeats
    if len(signatures) >= 3 and len(set(signatures[-3:])) == 1:
        return True, f"Same action repeated 3x: {signatures[-1]}"

    # Check for alternating pattern (A-B-A-B)
    if len(signatures) >= 4:
        if signatures[-4] == signatures[-2] and signatures[-3] == signatures[-1]:
            return True, f"Alternating loop: {signatures[-2]}/{signatures[-1]}"

    # Store for future reference
    state["loop_signatures"] = signatures
    return False, None
```

## Procedure

### Step 1: Diagnose the plateau

```python
# What has been tried?
tested_actions = set(a["action"] for a in state["completed_actions"])
tested_strategies = set(state["strategies_tested"])
untested_strategies = set("ABCDEF") - tested_strategies

# What worked vs didn't?
gains = [(a["action"], a["gain_pct"]) for a in state["completed_actions"] if a["gain_pct"] > 0]
losses = [(a["action"], a["gain_pct"]) for a in state["completed_actions"] if a["gain_pct"] <= 0]

# Where is time being spent?
tier_breakdown = state.get("tier_breakdown", {})
largest_untouched_tier = max(
    ((tier, pct) for tier, pct in tier_breakdown.items()
     if tier not in ["T5_COMPILED"]),  # T5 can't be source-rewritten
    key=lambda x: x[1],
    default=("none", 0)
)
```

### Step 2: Generate novel actions (structured)

Based on the plateau diagnosis, inject new actions not yet tried.

```python
NOVEL_ACTIONS = [
    # Try untested strategies
    {"condition": "'C' not in tested_strategies and selective_compile_viable",
     "action": "kernel-opt-submit", "params": {"strategy": "C"},
     "score": 5, "description": "Selective compile → Inductor Triton for GEAK"},

    {"condition": "'D' not in tested_strategies and tier_breakdown.get('T2_AITER_CK', 0) > 20",
     "action": "call-stack-opt", "params": {"strategy": "D", "tier": "T2"},
     "score": 6, "description": "aiter dispatch patching for model-specific shapes"},

    {"condition": "'E' not in tested_strategies and tier_breakdown.get('T3_FRAMEWORK', 0) > 5",
     "action": "call-stack-opt", "params": {"strategy": "E", "tier": "T3"},
     "score": 5, "description": "Framework scheduling optimization"},

    {"condition": "'F' not in tested_strategies",
     "action": "call-stack-opt", "params": {"strategy": "F", "tier": "T1"},
     "score": 4, "description": "Kernel sequence fusion"},

    # Re-profile after accumulated changes
    {"condition": "len(gains) >= 2",
     "action": "re-profile",
     "score": 7, "description": "Re-profile to discover new bottlenecks after gains"},

    # Try different concurrency points
    {"condition": "True",
     "action": "params", "params": {"conc_multiplier": 2},
     "score": 3, "description": "Test at 2x concurrency"},

    # Try alternative backends not yet tested
    {"condition": "'claude' not in state.get('backend_wins', {})",
     "action": "kernel-opt-submit", "params": {"force_backend": "claude"},
     "score": 4, "description": "Try Claude backend (may find different optimizations)"},

    # Communication tuning if comm is significant
    {"condition": "tier_breakdown.get('T4_COMM', 0) > 15",
     "action": "nccl_tuning",
     "score": 5, "description": "NCCL/RCCL parameter tuning for communication bottleneck"},
]

for novel in NOVEL_ACTIONS:
    if eval(novel["condition"]):
        push_action(novel["action"], novel["score"], novel["params"])
        print(f"Re-explore injected: {novel['description']} (score={novel['score']})")
```

### Step 2b: Synthetic fallback injection (from failed actions)

When structured re-explore produces fewer than 3 actions, mine the
completed_actions history for retry opportunities with alternate strategies.

Walk through every DISCARD, error, or crash in completed_actions. For each
failed kernel, look up what strategy was used and push retries using the
alternate strategies from this map:

| Failed strategy | Alternatives to try |
|---|---|
| oob-rewrite | triton-rewrite, hip-kernel, register-constrained-rewrite |
| triton-rewrite | hip-kernel, oob-rewrite, selective-compile |
| hip-kernel | triton-rewrite, oob-rewrite |
| dispatch-fix | framework-rebuild, env-var-override |
| operator-tuning | manual-shape-config, alt-tuning-tool |
| framework-rebuild | patch-only, alt-compiler-flags |
| env-var-toggle | env-var-integer-sweep, env-var-combination |
| comm-optimization | alt-algorithm, topology-change |
| compiler-tuning | cache-clear-retry, alt-tiling, pinned-triton |

Skip alternatives already tried on that same kernel. Score retries at 4.5.

Then fill strategy gaps: compare the strategies you have actually tested
against the full design space (all the strategies in the table above, plus
kernel-fusion, selective-compile, memory-layout-opt, mixed-precision-expand,
kv-cache-layout, prefetch-opt). Push an exploratory action for each untested
strategy type with score 4.

Next, check the kernel_dispatch_map for any kernel with >=1% GPU time that
no completed action has touched. Push a deep-kernel-analysis action for each,
scored proportional to its GPU%.

Finally, if there are 2+ successful KEEPs, push a compound action that
combines the two highest-gain optimizations to test for superlinear effects.
Score it at 6.

### Step 2c: Self-reflection (last resort)

If Steps 2 + 2b together still produce fewer than 2 new actions, step back
and brainstorm genuinely novel ideas by considering dimensions that haven't
been explored yet:

- Memory alignment and tensor padding for hardware-optimal access patterns
- Kernel launch overhead reduction via graph capture or persistent kernels
- Mixed-precision expansion — can more operations run in fp8 or fp4?
- KV cache layout optimization and paged attention variants
- Prefetch and async copy overlap to hide data movement behind compute
- Hardware-specific features unique to the target GPU (gfx950 instructions, etc.)
- Custom fused operators (attention+norm, gate+up projection fusion)

For each idea, estimate expected gain, confidence, and risk. Push 1-3 novel
actions with score 5 each. These are speculative but keep the marathon alive
during deep plateaus.

### Step 3: Reset consecutive discard counter

After injecting novel actions, reset `state.consecutive_discards = 0` to give the
new actions a fair chance before the stopping criteria triggers again.

## Outputs
- New actions injected into `state.action_stack`
- Reset `state.consecutive_discards`
- Log of which novel actions were injected and why

## Accuracy Validation
N/A — re-explore only injects new actions, doesn't modify the model.

## Heuristic Update
- Novel actions start with their injected scores (not the default priors)
- After a successful re-explore action, boost the "re-explore" trigger threshold
  (require 4+ discards instead of 3 next time)
