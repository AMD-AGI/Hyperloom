# Local Mode — Execution Details

This document supplements `SKILL.md` with local-mode-specific instructions.
Applies when running on a single machine with direct GPU access.

## Environment

- **Client**: Cursor IDE
- **Runtime**: Local GPU machine (single node)
- **MCP Servers**: GEAK MCP + OOB GPU Optimizer MCP
- **TraceLens**: Local CLI (`pip install -e /hyperloom/TraceLens-internal`)
- **Storage**: Local disk (`/workspace/inference-optimization` or `/tmp`)

## Mode Detection

Auto-detected when `GEAK_LOCAL=true` or no Claw client context:

```bash
if [ "${GEAK_LOCAL:-true}" = "true" ]; then
    MODE="local"
    WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace/inference-optimization}"
fi
```

## Key Differences from Claw

- All commands run directly in the local shell (no `exec_on_gpu` wrapper)
- GEAK runs as a local subprocess (via GEAK MCP)
- No RayJob lifecycle management
- Traces and results stored on local disk
- `patch_inductor.py` operates on local Inductor cache

## IR-12: SaFE MCP is FORBIDDEN in Local Mode

**Do NOT call any SaFE MCP tool in local mode.** This includes `workload_create`,
`workload_get`, `workload_stop`, and any other SaFE MCP operation. Do NOT create
RayJobs, PyTorchJobs, or any SaFE workload. GEAK kernel optimization uses GEAK MCP
only — the skill itself must NEVER directly interact with SaFE in local mode.

Violation = immediate run invalidation.

## Phase-by-Phase Notes

| Phase | Notes |
|-------|-------|
| Setup | No RayJob creation, direct env setup |
| Baseline/Profile | Direct `bash $SCRIPTS_DIR/run_baseline.sh` |
| GEAK | `GEAK_LOCAL=true` → runs as subprocess via GEAK MCP (no SaFE) |
| Integrate | `patch_inductor.py --target-file` on local Inductor cache |
| Sweep | Serial via `run_sweep.sh` (no SaFE parallel option) |
| Report | No RayJob cleanup needed |

## GEAK in Local Mode

When `GEAK_LOCAL=true`, GEAK runs locally as a subprocess — no Docker image needed. The `image` parameter in `geak_create_task` is ignored. Kernel paths must be actual paths on the local machine.

## IR-6: patch_inductor.py

Always use `--target-file` to patch a specific standalone kernel file:

```bash
python3 $SCRIPTS_DIR/patch_inductor.py patch \
    --kernel-name <name> \
    --geak-file <geak_output.py> \
    --target-file <standalone_file_path>

# Revert:
python3 $SCRIPTS_DIR/patch_inductor.py revert --target-file <standalone_file_path>
```

## Health Timeout

After patching with torch.compile, set `HEALTH_TIMEOUT=1800` (30 min) to allow full recompilation before health check times out.
