# General Kernel Tuning Rules

## Development Loop (MANDATORY ORDER)
1. READ current source and configuration
2. PREDICT PMC counter impact of proposed change
3. BUILD (clean stale artifacts first)
4. TEST correctness (SNR gate) — STOP if FAIL
5. BENCH wall-clock (30-iter median, in-context)
6. PROFILE PMC counters
7. ANALYZE: compare prediction vs reality
8. DECIDE next change based on data
9. LOG experiment: {config, SNR, wall_ms, PMC, decision}

## One Variable at a Time
Change only ONE configuration parameter per iteration.
Multiple simultaneous changes make it impossible to attribute
improvements or regressions to specific changes.

## Gate-Driven Development
- Define a target wall_ms BEFORE starting optimization
- Once the gate is met, STOP and report GREEN
- If 3 consecutive iterations show <2% improvement, the config has PLATEAUED
- At plateau: move to module-level optimization, not more tile tuning

## When to Stop Kernel Tuning
1. Gate is met → STOP
2. PMC shows >90% MFMA utilization → near hardware limits → STOP
3. 3 consecutive <2% improvements → PLATEAUED → STOP
4. Register pressure is at occupancy boundary → structural change needed

## Occupancy vs Register Reuse Tradeoff
- occupancy=2 at 256 VGPR often beats occupancy=1 at 384 VGPR
- But if the kernel is compute-bound (wait/MFMA < 5), higher register
  reuse (more VGPR, lower occupancy) can win
- Measure both configurations; do not assume

## Never Inherit Tuning Parameters
- Dense kernel tunings DO NOT transfer to sparse kernels
- Different attention patterns need independent tuning
- WAVES_PER_EU must be re-tuned per kernel (dense=3, sparse=2 in our experience)
- Always re-measure after parameter transplant

## Hybrid Strategy
When one backend plateaus, consider:
- FlyDSL for compute-heavy stages + Triton/CK for memory-bound stages
- This captured 80% of the win for SLA backward attention
- But beware instruction cache eviction at backend boundaries (~7us penalty)
