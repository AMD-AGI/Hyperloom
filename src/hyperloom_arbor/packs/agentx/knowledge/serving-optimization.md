# Serving Optimization on MI355X

## SIKL decode profile — run this before every hypothesis

**Before proposing any change, run the SIKL decode profile to know where time is actually going.**

```bash
# SIKL requires MODEL_PATH and TP — always set them explicitly:
docker exec -e MODEL_PATH=/models/GLM-5.2-MXFP4 -e TP=4 \
  $CONTAINER bash /workspace/Arbor/packs/agentx/validator/sikl_profile_decode.sh
# Reads: /tmp/sikl_decode_result.json
# Shows: per-op GPU time breakdown for GLM-5.2 decode step
# Note: SIKL starts its own server instance, so run it BEFORE the bench server is up
```

**To validate server flags before committing:** run `docker exec $CONTAINER python3 -m sglang.launch_server --help 2>&1 | grep <flag>` to confirm a flag is valid in this build.

The profile tells you the top bottlenecks by GPU time — attention, MoE dispatch, topk, allreduce, etc. A change that doesn't address one of the top-3 ops is unlikely to move the AgentX number.

Run it again after a KEEP to see if the bottleneck shifted. The agent should maintain a running view of the current bottleneck breakdown, not rely on stale profile data from the start of the campaign.

## Tier ladder

1. **Config** — env vars, server flags, AITER switches. Exhaust in 2-3 iterations. These are warmup.
2. **Framework** — patches to vLLM/SGLang Python source: fused ops, scheduling, dispatch, batching, attention backends, KV cache policies.
3. **Kernel** — new or rewritten HIP/Triton/FlyDSL/CK kernels: MoE routing, attention, KV cache, GEMM, prefill/decode overlap.
4. **Feature** — port a whole NVIDIA feature: persistent kernels, flash-decode, chunked prefill improvements, speculative paths, fused GEMM+activation, EP overlap.

Config changes are one-time warmup. The session's value comes from code changes at framework/kernel/feature tier.

## Research protocol

Before proposing any code change in framework/kernel/feature tier:
1. Search for whether NVIDIA already solved it in TRT-LLM, CUTLASS, or the vLLM CUDA path
2. Check unmerged PRs in the relevant framework (SGLang, vLLM, AITER, CK)
3. Read the actual diffs, not summaries
4. Prioritize: proven speedup on NVIDIA + low AMD porting complexity = highest priority

## Depth expectations

Every code change must address a real hardware bottleneck — not a wrapper around an existing flag, not a one-line conditional. The right question: "what would it take to actually fix this bottleneck at the hardware level?"
