---
myst:
    html_meta:
        "description": "Single-node Ray execution variables for Hyperloom: Ray backend toggle, head readiness timeout, and driver connect probe timeout for the single-node Ray serving path."
        "keywords": "Hyperloom, environment variables, Ray execution, INFERENCE_OPTIMIZER_RAY_EXEC, HYPERLOOM_RAY_READY_TIMEOUT_SEC, HYPERLOOM_RAY_CONNECT_PROBE_TIMEOUT_SEC, single-node"
---
# Hyperloom Single-node Ray execution

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_OPTIMIZER_RAY_EXEC` | Unset (`on` for single-node) | Controls whether single-node serving benchmarks and `needs_gpu` specialists run through Ray actors. When unset, single-node runs are routed through Ray-managed leases while multi-node stays on the multi-node backend. Set to `0` / `false` / `no` / `off` to force the local subprocess path, or `1` / `true` / `yes` / `on` to force Ray. |
| `HYPERLOOM_RAY_READY_TIMEOUT_SEC` | `30` | How long, in seconds, a Ray head is given to answer `ray status` when a single probe (bounded by `HYPERLOOM_RAY_STATUS_TIMEOUT_SEC`, 5 s) timed out, and after `ray start --head` before the new head is reported unreachable. Each probe while waiting is bounded by the time left, so a healthy head that is slower than the single-probe bound still answers. A head that times out is not stopped or restarted: `ensure_ray_cluster()` raises once the time is up, and only a head that definitely is not there (`ray status` exits non-zero) is stopped and replaced. A non-positive or unparsable value keeps the default. |
| `HYPERLOOM_RAY_CONNECT_PROBE_TIMEOUT_SEC` | `60` | Bound, in seconds, on the driver connect (`ray.init`) a process tries in a child process before it connects to a running Ray head in-process: first when work is routed to Ray, and again before every in-process reconnect (after a Ray actor dies, after the process starts or restarts a local head, and before the retry after a version mismatch). In-process, Ray 2.44 waits for the raylet's registration reply with the GIL held and no timeout, so a head that never answers would freeze the whole process. When the probe times out (its process group is killed) or exits non-zero, the process does not connect, takes the local subprocess path from then on, as with `INFERENCE_OPTIMIZER_RAY_EXEC=0`, and logs one WARNING. The installer bounds its own `serving_slot` check of a running head by the same value and replaces a head that does not answer in time. A non-positive or unparsable value keeps the default. |
