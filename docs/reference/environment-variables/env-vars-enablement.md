---
myst:
    html_meta:
        "description": "Enablement admission reference for Hyperloom: the --enablement CLI flag (off/launch/eval/all) for self-heal lanes and the fixed accuracy floor constant for enablement KEEP decisions."
        "keywords": "Hyperloom, enablement, --enablement, self-heal, baseline_failed, accuracy floor, DEFAULT_ENABLEMENT_ACCURACY_FLOOR, launch lane, eval lane"
---
# Hyperloom Enablement admission

Enablement is not configured through the environment. Both self-heal lanes
are admitted by the `--enablement {off,launch,eval,all}` CLI flag, which defaults
to `all`:

- `launch` — a baseline that cannot boot routes into patch authoring.
- `eval` — a baseline that boots and measures throughput but fails its accuracy
  eval (crashes, produces no result, or scores below the floor) routes into
  patch authoring. Single-node only; multi-node keeps the strict stop.
- `all` (default) — both lanes.
- `off` — neither lane engages, and a baseline that keeps failing terminates the
  run with `stop_reason='baseline_failed'` instead of opening an authoring loop.

The eval trigger and the enablement KEEP gate both judge a score against the
fixed constant `hyperloom.common.gain_math.DEFAULT_ENABLEMENT_ACCURACY_FLOOR` (`0.5`):
`score >= floor` passes, and a score that is not finite or not above zero fails.
The lanes differ only when a candidate's eval produced no score: an `eval`-lane
candidate then fails, while a `launch`-lane candidate that serves is kept as
provisional, with its correctness not verified.
