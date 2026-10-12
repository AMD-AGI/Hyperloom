---
title: measurement protocol — warmup, repeats, noise band, locked clocks, A/B
kind: measure
measure: protocol
gens: [gfx950]
updated: 2026-08-28
---

# Measurement protocol

**A number produced outside this protocol is not evidence.** Every claim in a campaign — baseline,
delta, regression — has to clear these rules or it does not count. This is the cheapest place to lose
a week: chasing a "win" that was clock drift.

## Route here when
- Establishing a baseline, before any optimization.
- Validating a change — before you believe it, and before you report it.
- A result looks too good, or refuses to reproduce.

## The rules

| Rule | Value | Why |
|---|---|---|
| **Warm** | discard cold runs | clocks ramp, caches fill, JIT/autotune resolves on first call |
| **Repeats** | **REPEATS=7** (minimum median-of-3) | single samples are dominated by DVFS position |
| **Report** | **median + spread**, never a single number | spread is how a reader judges the claim |
| **Noise band** | **~0.5% e2e** — a delta inside it is *not a result* | below this, clock and scheduling variance dominate |
| **Clocks** | locked, or at minimum monitored | see below |
| **A/B** | **same session, non-overlapping**, ref then candidate back-to-back | never compare across sessions/boxes/days |
| **Untraced** | time in a separate pass from counter collection | tracer and counter replay inflate timing |

## What you are fighting on Instinct

- **Peak ≠ sustained clock.** The boost ceiling is not what you run at under sustained AI load; the
  engine clock settles lower and is power/thermal-capped. MI355X (1400 W liquid) holds clock longer
  than MI350X (1000 W air) — the same kernel measures differently on the two SKUs at identical peak
  tables.
- **Per-XCD clock variance ~3–10%** across the 8 XCDs. Different launches land on different dies, so
  repeat-to-repeat spread partly reflects *which* XCDs the scheduler used.
- **DVFS ramp lag** — a short kernel can finish before the clock ramps. This is what warmup hides.

Net rule: **compute achieved TFLOP/s from measured time, never from an assumed clock.**

## The recipe

1. **Warm up.** Several untimed runs to ramp clocks, warm caches, resolve JIT/autotune. Discard.
2. **Time REPEATS=7.** Report median and spread.
3. **Apply the noise band.** ~0.5% e2e. Per-kernel microbench bands are tighter but never zero — quote
   the spread either way.
4. **Control clocks.** Pin a deterministic performance level with `rocm-smi` / `amd-smi` for kernel
   microbenchmarks. At minimum monitor with `amd-smi metric` (sclk / mclk / power / temp / throttle)
   during the run and **reject any A/B where the clock drifted between ref and candidate.**
5. **A/B in one session.** Ref then candidate, back-to-back, same process, same clocks.
6. **Use HIP graphs for launch-bound work** — replays a launch sequence with near-zero host overhead.
   Both a measurement tool (get the real GPU-bound time) and an optimization when the trace shows
   host-launch gaps.

## 2-launch A/B beats summed per-leg microbenchmarks

For an e2e serving change, run a **full ref launch vs a full candidate launch**. Do not sum per-kernel
microbenchmarks: that misses overlap, caching, and dispatch interactions, and routinely disagrees with
e2e in both directions.

Reference: the aiter GEMM DB tuning win (**+2.23% e2e**, Qwen3.5-27B / sglang 0.5.11, 1548.9 → 1583.5
tok/s) was validated by a same-session non-overlapping 2-launch A/B — not by per-kernel sums.

## Reporting format

```
<value> @ <hw>, ROCm <ver>, <lib>@<commit/ver>, <date>
```
e.g. `+2.23% e2e @ MI300X gfx942, sglang 0.5.11 / aiter, 2026-06-08`

Median of ≥3 (preferably 7) warm repeats, with spread. Never present theoretical peak as achievable.

## Prove the change is actually live

A measurement of the wrong binary is worse than no measurement. Before believing a delta:

- The kernel you edited **appears in the profiled dispatch list** (`measure_rocpc_workflow.md`).
- For an aiter DB tune: `grep -c 'is tuned on cu_num' server.log` **> 0** (`lever_autotune.md`).
- For a source edit: the ISA changed in the way you expected — a "win" whose ISA is byte-identical to
  the baseline is noise, every time.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Sub-0.5% "win" | inside the noise band | not a result; do not report it |
| Win doesn't reproduce | different session / clock state | same-session non-overlapping A/B |
| Timing inflated | measured a traced/profiled run | separate untraced timing pass |
| First run is an outlier | cold cache, unramped clock | warm up and discard |
| Per-leg sums say +15%, e2e says 0% | missed overlap and dispatch interaction | trust the 2-launch A/B |
| Huge spread across repeats | XCD clock variance, or background load | lock clocks; report the spread; re-run |
| Delta real but ISA unchanged | you measured something else | confirm the edit is live |

## Deeper
`measure_rocpc_workflow.md` (how to actually run the profiler) ·
`measure_triage.md` (what to do with the counters) ·
`hardware/mi350_clocks.md` (sustained-clock behaviour, SKU differences) ·
`lever_autotune.md` (engagement proof before A/B)
