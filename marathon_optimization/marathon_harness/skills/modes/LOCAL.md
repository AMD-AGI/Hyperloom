# Local Mode — Execution Details

This document supplements `SKILL.md` with local-mode-specific instructions.
Applies when running on a single machine with direct GPU access.

## Environment

- **Client**: Cursor IDE
- **Runtime**: Local GPU machine (single node)
- **MCP Servers**: GEAK MCP, OOB Agent MCP (`oob`), LLM Proxy
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
- GEAK runs as a local subprocess or remote SaFE PyTorchJob
- OOB agents (Codex/Claude) work identically in both modes — remote code gen, local verification
- No RayJob lifecycle management
- Traces and results stored on local disk
- `patch_inductor.py` operates on local Inductor cache

## Phase-by-Phase Notes

| Phase | Notes |
|-------|-------|
| Setup | No RayJob creation, direct env setup |
| Baseline/Profile | Direct `bash $SCRIPTS_DIR/run_baseline.sh` |
| GEAK | `GEAK_LOCAL=true` → runs as subprocess; otherwise remote SaFE PyTorchJob |
| OOB (Codex/Claude) | Remote MCP → code gen → **local GPU verification** (see below) |
| LLM Proxy | Direct API call → code gen → **local GPU verification** |
| Integrate | `patch_inductor.py --target-file` on local Inductor cache |
| Sweep | Serial via `run_sweep.sh` (no SaFE parallel option) |
| Report | No RayJob cleanup needed |

## GEAK in Local Mode

When `GEAK_LOCAL=true`, GEAK runs locally as a subprocess — no Docker image needed. The `image` parameter in `geak_create_task` is ignored. Kernel paths must be actual paths on the local machine.

## OOB Agents in Local Mode

OOB agents (Codex, Claude) work in local mode with no extra setup. The `oob`
MCP is a remote HTTP endpoint — the agent generates code on a remote pod (no GPU), and
the calling skill verifies locally on the machine's GPU.

**All four backends are available in local mode:**

| Backend | MCP | Code gen | Verification | Works locally? |
|---------|-----|----------|-------------|:-:|
| `geak` | GEAK MCP | Remote GPU pod | On-pod (GEAK verifies internally) | Yes |
| `codex` | OOB Agent MCP | Remote (no GPU) | **Local GPU** | Yes |
| `claude` | OOB Agent MCP | Remote (no GPU) | **Local GPU** | Yes |
| `llm` | Direct OpenAI API | Remote (no GPU) | **Local GPU** | Yes |

### Local verification loop (Codex/Claude/LLM)

These backends have no GPU — the calling skill runs the iterative refinement loop
on the local machine. Each iteration:

1. **Submit** kernel + prompt to remote agent via MCP
2. **Download** optimized kernel from agent output
3. **Verify locally** on local GPU:
   - **Compile check:** `exec(compile(code, "kernel.py", "exec"))`
   - **Correctness:** run original + optimized with trace-derived test inputs,
     `torch.allclose(orig_out, opt_out, atol=1e-2, rtol=1e-2)`
   - **Micro-benchmark:** time both kernels with trace-derived shapes
     (see `kernel-opt.md` "Test Harness Generation" for how to build test inputs)
4. **Feed results back** as context for next iteration

The local machine must have the inference server stopped or a separate GPU available
for micro-benchmarking. If the server is running, use E2E benchmark via `run_baseline.sh`
instead of isolated micro-benchmark.

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
