# Inference Optimization Skill

Remote-only LLM inference optimization on AMD Instinct GPUs. The skill creates a SaFE
RayJob, profiles the serving stack, submits kernel/code optimization work through OOB
Codex/Claude, validates changes with controlled benchmarks, and writes a final report.

This skill is intentionally **OOB-only** for kernel/code optimization. Do not use non-OOB
backends, direct LLM API calls, or ad-hoc optimization scripts.

| File | Description |
|------|-------------|
| `SKILL.md` | Main skill — Remote RayJob orchestration, OOB-only kernel/code optimization rules, scoring, accuracy gates |
| `SKILL_chinese.md` | Chinese translation of `SKILL.md` for human review |
| `modes/REMOTE.md` | Remote RayJob execution details, MCP lifecycle, Ray Dashboard REST patterns |
| `addendum.md` | Additional execution rules learned from prior remote sessions |
| `kernel-opt/codex.md` | Codex backend reference via `oob_ray_submit.py run -a codex` |
| `kernel-opt/claude.md` | Claude backend reference via `oob_ray_submit.py run -a claude` |
| `scripts/bootstrap.sh` | Installs/verifies OOB, TraceLens, CLI auth, and runtime env inside the RayJob |
| `scripts/oob_ray_submit.py` | Runs OOB tasks inside the RayJob Ray runtime with GPU isolation |
| `scripts/common.sh` | Shared serving helpers — kill_server, wait_for_health, filter_trace, check_gpu_memory |
| `scripts/run_baseline.sh` | Baseline/re-baseline benchmark runner |
| `scripts/run_profile.sh` | Profiling run against an already-running server |
| `scripts/run_sweep.sh` | CONC/ISL/OSL sweep runner |

## Output Directories

Runtime outputs live on the shared filesystem mounted by the RayJob:

```text
/wekafs/inference-optimization/
├── results/<timestamp>/     # Benchmark JSON, server logs, eval output, final report
├── traces/<timestamp>/      # Profiler traces for TraceLens / offline analysis
```

OOB outputs should be written to a pod-visible shared path such as:

```text
$RESULT_DIR/oob_<agent>_<kernel_name>/tasks/<user>/<task_id>/workspace/
```

Use the `.workspace` field from `oob_ray_submit.py run --json` instead of guessing paths.

## What It Does

1. Create one SaFE `RayJob` for the full optimization run.
2. Bootstrap the RayJob image so OOB, TraceLens, CLI auth, and helper scripts are available.
3. Classify the model and choose initial optimization priorities.
4. Run a controlled baseline benchmark.
5. Profile the serving engine with torch profiler and analyze trace data.
6. Identify OOB optimization candidates from hot kernels, framework source, scheduling code, or integration bottlenecks.
7. Submit candidates to OOB Codex/Claude via `oob_ray_submit.py run`.
8. Patch returned changes, re-baseline with the same benchmark configuration, and keep or revert.
9. Run a final parameter sweep for the optimized configuration.
10. Generate the report and ingest useful findings into the KB.

## Execution Model

The control plane is MCP + Ray Dashboard REST:

- Use SaFE MCP `workload_create(kind="RayJob")` to create the cluster.
- Use SaFE MCP `workload_get` / `workload_list` to inspect workload status.
- Use Ray Dashboard REST `POST http://<HEAD_IP>:8265/api/jobs/` to submit work inside the RayJob.
- Use REST `GET /api/jobs/<submission_id>` and `/api/jobs/<submission_id>/logs` to poll status and logs.
- Use SaFE MCP `workload_stop` to stop the RayJob at the end.

Do not drive the RayJob from the sandbox with Ray Client (`ray://<head>:10001`). If a step needs Ray APIs, submit a Python driver through Ray Dashboard REST and call `ray.init()` from inside the RayJob image.

## Prerequisites

- **GPU**: AMD Instinct MI355X / MI325X / MI300X.
- **Runtime**: SaFE RayJob with pod-visible shared filesystem.
- **Framework**: SGLang or vLLM installed in the RayJob image.
- **InferenceX**: Available from inside the RayJob, for example `/wekafs/InferenceX` or bootstrap-provided path.
- **Model**: Available from inside the RayJob, for example `/wekafs/models/<model>`.
- **OOB**: Installed by `scripts/bootstrap.sh` inside the RayJob.
- **TraceLens**: Installed by bootstrap or available on a shared path for trace analysis.

## Quick Start

### Full Remote Optimization

```text
@inference-optimization Optimize DeepSeek-R1-0528 inference on MI300X.
Mode: remote
Model: /wekafs/models/DeepSeek-R1-0528
InferenceX: /wekafs/InferenceX
KERNEL_OPT_BACKENDS: codex,claude

Create a SaFE RayJob, run baseline, profile, submit OOB kernel/code optimization
tasks, integrate validated changes, sweep CONC, and generate the report.
```

### Analysis Only

```text
@inference-optimization Profile DeepSeek-R1 inference bottlenecks.
Mode: remote
Model: /wekafs/models/DeepSeek-R1-0528

Run baseline + profiling + TraceLens/kernel analysis only. Do not submit OOB
optimization tasks and do not patch code.
```

### Sweep Only

```text
@inference-optimization Sweep DeepSeek-R1 inference across CONC=4,8,16,32,64.
Mode: remote
Model: /wekafs/models/DeepSeek-R1-0528
ISL/OSL: 1024/1024 and 8192/1024

Skip TraceLens and OOB optimization. Run controlled benchmark sweeps only.
```

## OOB Optimization Flow

OOB optimization is always mediated by `oob_ray_submit.py` inside the RayJob:

```bash
$OOB_RAY_CLI run \
  -a codex \
  -p "$PROMPT" \
  -f "$WORK_DIR/kernel.py" \
  -o "$WORK_DIR/oob_codex_${KERNEL_NAME}" \
  --max-turns 20 \
  --timeout 1200 \
  --no-live --json
```

For Claude, use `-a claude` and the constraints in `kernel-opt/claude.md`.

Every OOB task should:

- Receive the full source via `-f` or prompt content.
- Write a complete `optimized_kernel.py` or code patch in its workspace.
- Return JSON whose `.workspace` points to the workspace.
- Be verified before patching the serving stack.

## Validation Rules

- Re-baseline with the same model, TP, CONC, ISL, OSL, server args, and benchmark args.
- Use `run_baseline.sh` for both baseline and re-baseline.
- If a patch changes computation, run the GSM8K accuracy gate before KEEP.
- If throughput regresses, accuracy drops, or the server crashes, revert the patch.
- Treat gains from changed batching/concurrency as invalid kernel/code gains.

## Output

- **Benchmark JSON** — baseline, re-baseline, and sweep results.
- **`results.tsv`** — sweep table when `run_sweep.sh` is used.
- **Optimization report** — TraceLens analysis, OOB attempts, kept/reverted changes, sweep summary.
- **Profiler traces** — baseline and optimized traces when profiling is enabled.
- **OOB workspaces** — task trajectories, logs, and generated files.
- **Kernel/code backups** — originals needed for rollback.

## Tips

- Bootstrap OOB inside the RayJob before any kernel/code optimization.
- Use pod-visible paths (`/wekafs`, `/hyperloom/users/...`) in OOB prompts and `-f` arguments; client paths such as `/mnt/weka/...` are not valid inside RayJob workers.
- Do not call model APIs directly from the orchestrator as a shortcut; use OOB so the trajectory, workspace, and artifacts are reproducible.
- `torch.compile` can expose runtime-generated Triton kernels, but OOB can also work on framework source, scheduling logic, launch scripts, and integration code.
- TraceLens may fail on very large traces; if so, use filtered traces or kernel summaries and document the fallback.
- Always `unset PROFILE SGLANG_TORCH_PROFILER_DIR VLLM_TORCH_PROFILER_DIR` after profiling to avoid severe slowdown.
- Only patch standalone Inductor kernel files; graph module files contain multiple kernels and are unsafe to replace wholesale.
