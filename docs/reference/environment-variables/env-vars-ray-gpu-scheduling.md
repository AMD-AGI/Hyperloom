---
myst:
    html_meta:
        "description": "Single-node Ray GPU scheduling variables for Hyperloom: in-flight GPU specialist limit, per-round wall-clock ceiling, and serving-priority flag to prevent research work from starving serving."
        "keywords": "Hyperloom, environment variables, Ray GPU scheduling, INFERENCE_OPTIMIZER_RAY_GPU_PENDING_LIMIT, INFERENCE_OPTIMIZER_RAY_ROUND_WAIT_SEC, INFERENCE_OPTIMIZER_RAY_SERVING_PRIORITY"
---
# Hyperloom Single-node Ray GPU scheduling

These variables tune the single-node Ray execution path (active when
`INFERENCE_OPTIMIZER_RAY_EXEC=1` and `--nodes=1`). They have no effect on
multi-node runs or when the Ray backend is disabled.

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_OPTIMIZER_RAY_GPU_PENDING_LIMIT` | `4` | Maximum number of GPU specialists that can be simultaneously in-flight (pending Ray scheduling + running) on the single-node Ray path. Ray still serializes execution on the physical GPU(s) using `num_gpus`; this limit caps how many actors can queue behind the current one. Floored at `1`. Reduce to `1` or `2` when GPU memory or per-process overhead is a concern (each queued actor holds a Ray worker slot even while it waits). |
| `INFERENCE_OPTIMIZER_RAY_ROUND_WAIT_SEC` | Unset (the round's own timeout + 3600s) | Wall-clock ceiling on blocking for one Ray round to return. Without it a round Ray never schedules parks the waiting thread forever, leaving its task `running` and its lane lease never released — the 2026-09-21 stall. Set a number of seconds to override; `<= 0` disables the ceiling, which is also what a round with no cap of its own gets. A non-finite value (`nan`, `inf`) is rejected, not tolerated, because `nan > 0` is false and would silently switch the ceiling off. On timeout the round is abandoned and its actor is kept alive on purpose, so its GPUs stay reserved rather than being handed back while its server tree may live. |
| `INFERENCE_OPTIMIZER_RAY_SERVING_PRIORITY` | On | When enabled (default), the dispatcher defers admitting new GPU research specialists while a serving benchmark holds the whole-machine `serving_slot`, preventing research work from starving serving. The slot is probed immediately before each specialist is admitted so a serving start that races the dispatch pass is caught. Set to `0`, `false`, `no`, or `off` to disable. |
