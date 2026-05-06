# Claw Mode

## Mode Detection

```bash
if [ "${GEAK_LOCAL:-true}" != "true" ]; then
    MODE="claw"
fi
```

Claw mode dispatches every GPU-side command through `exec_on_gpu`, which
forwards to the Ray client at `RAY_HEAD_ADDRESS`. A single inference RayJob
is created up-front and reused for both server and benchmark phases.

## Iron Rules (claw-specific)

- **IR-10**: SaFE MCP only — `workload_create(kind="RayJob")`, `workload_get`,
  `workload_list`, `workload_stop`. NEVER `workload_delete`.
- **IR-12**: shared NFS is already mounted at `/shared_nfs` inside the Ray
  pod; do not remount.
- IR-4 / IR-5 / IR-8 still apply (kill stale server, safe pgrep-based
  process management, background runner for long sweeps).

## RayJob payload (single-node example)

```jsonc
{
  "display_name": "inference-sweep-<model_short>",
  "workspace_id": "<user_workspace_id>",
  "kind": "RayJob",
  "images": ["<KERNEL_OPT_IMAGE>"],
  "resources": [
    {"replica": 1, "cpu": "96", "gpu": "8", "memory": "1024Gi",
     "sharedMemory": "500Gi", "ephemeralStorage": "500Gi"}
  ],
  "env": { "RAY_JOB_ENTRYPOINT": "dGFpbCAtZiAvZGV2L251bGw=" },
  "is_tolerate_all": true,
  "ttl_seconds_after_finished": 600
}
```

`RAY_JOB_ENTRYPOINT` MUST be base64 of `tail -f /dev/null` to keep the
cluster alive while we drive it via `exec_on_gpu`.

## Wait for Ready

```python
import time
RAYJOB_ID = "<workload_id from workload_create>"
for _ in range(60):
    r = workload_get(workload_id=RAYJOB_ID)
    phase = r.get("status", {}).get("phase", "")
    if phase == "Running":
        break
    if phase in ("Failed", "Stopped"):
        raise RuntimeError(f"RayJob failed: {phase}")
    time.sleep(30)

pods = r["status"]["pods"]
head = next((p for p in pods if "head" in p["name"]), pods[0])
HEAD_IP = head["ip"]
RAY_HEAD_ADDRESS = f"ray://{HEAD_IP}:10001"
```

## Magpie installation on the RayJob head pod

Run once, immediately after `workload_get` reports `Running`:

```bash
exec_on_gpu '
set -e
candidates=(
    "${MAGPIE_PATH:-}"
    "/hyperloom/users/8cf535bc3ad11fa15e48157cf3b3f726/Magpie"
)
for c in "${candidates[@]}"; do
    if [ -n "$c" ] && [ -d "$c" ]; then MAGPIE_PATH="$c"; break; fi
done
if [ -z "${MAGPIE_PATH:-}" ]; then
    MAGPIE_PATH=$(ls -d /shared_nfs/*/Magpie 2>/dev/null | head -1)
fi
MAGPIE_PATH="${MAGPIE_PATH:-/shared_nfs/Magpie}"
export MAGPIE_PATH
echo "Resolved MAGPIE_PATH=$MAGPIE_PATH"
command -v magpie >/dev/null || pip install -e "$MAGPIE_PATH" 2>&1 | tail -5
python3 -c "from Magpie.modes.benchmark.config import SweepMatrix" \
    || { echo "ERROR: Magpie at $MAGPIE_PATH lacks sweep_matrix support" >&2; exit 1; }
'
```

## Per-action Claw overrides

### Setup (`actions/setup.md`)

After environment detection, create the RayJob using the payload above. Then
run the Magpie install snippet inside the head pod via `exec_on_gpu`. After
that, `actions/setup.md` Steps 2–5 still need to run inside the pod (they
read `MODEL` / `INFERENCEX_PATH` from the pod's filesystem, which is the
shared NFS mount).

The simplest pattern: export the same env vars on both Brain and the pod,
then call `exec_on_gpu` for any step that touches GPU state (the ROCm patch
in setup Step 5, for example).

### Sweep (`actions/sweep.md`)

Wrap the single `magpie benchmark` call:

```bash
exec_on_gpu "magpie benchmark \
    --benchmark-config $SWEEP_DIR/sweep_config.yaml \
    -o $SWEEP_DIR"
```

The YAML is generated on the Brain side (it lives on shared NFS under
`$RESULT_DIR`, which both Brain and the pod see). Magpie's `_execute_local_sweep`
runs entirely inside the pod.

### Long sweeps (IR-8)

Use `exec_on_gpu_bg` for sweeps expected to exceed the agent's foreground
budget (i.e. > ~30 minutes):

```bash
PID=$(exec_on_gpu_bg "
    magpie benchmark --benchmark-config $SWEEP_DIR/sweep_config.yaml \
        -o $SWEEP_DIR > $SWEEP_DIR/magpie.log 2>&1
")
echo "Background sweep PID on Ray head: $PID"
# Poll later:
exec_on_gpu "tail -50 $SWEEP_DIR/magpie.log"
```

## Cleanup

After the sweep:

```python
workload_stop(workload_id=RAYJOB_ID)
```

Do NOT use `workload_delete` (IR-10).
