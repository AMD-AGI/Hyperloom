---
title: Fusion validation harness — contract, JSON shape and the warm-up trap
kind: skill
scope: languages/fusion
updated: 2026-08-14
---

# The kernel-validation harness

The loop runs your harness and parses one JSON object from its stdout. If the
file is missing or the JSON is malformed, every validation attempt fails with
"harness not found" regardless of how good the kernel is.

## What it must do

Write a self-contained Python script at the path the task gives you. Guarded by
the fusion env flag(s), it must:

1. Import the fused module **and** the real eager op named by the reference hint.
2. Build representative decode tensors from the task's shapes.
3. Run fused against eager and compute per-shape parity: `snr_db` and
   `max_abs_err`.
4. Microbench both arms in microseconds (see the warm-up rule below).
5. Print, as the **last** stdout line and with nothing after it, one JSON object.

## JSON shape

```json
{"compiled": true, "is_triton": true, "error": "",
 "parity": [{"snr_db": 42.1, "max_abs_err": 3.2e-05, "label": "bs16_h4096"}],
 "eager_us": 118.4, "fused_us": 96.7,
 "skipped": false, "skip_reason": ""}
```

- Compile failure: `"compiled": false` and the real message in `"error"`.
- Microbench unavailable (hybrid/Mamba on ROCm): `"skipped": true` plus a
  `"skip_reason"`. Parity is still required.

Never hard-code a metric. Compute all of them live.

## The warm-up trap

Warm up **each arm** with at least **500 iterations before timing it**, then time
at least **200 iterations** and report the median.

This is not a detail to trim. Measured on this hardware, a 25-iteration warm-up
leaves the chip below its steady clock, and whichever arm is timed *second* comes
out about 3% slower from heat alone. That is exactly the size of the 1.03x
speedup gate, and it lands against the fused arm whenever eager is timed first.
A trimmed warm-up therefore manufactures a result of the same magnitude as the
effect being measured, in the direction that looks like failure.

## Gates

- Parity: SNR >= 30 dB.
- Speed: fused/eager >= 1.03x.
- Both must hold on the same run for the candidate to be kept.
