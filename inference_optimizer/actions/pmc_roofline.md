# PMC Roofline

Run PMC/roofline profiling in a dedicated server process. Do not use this action
for the normal torch-profiler trace path. By default this action launches the
server under `rocprofv3` instead of using `rocprofv3 --attach`, so it can run in
containers that do not grant `CAP_SYS_PTRACE`.

## GPU Scheduling Contract

All GPU work for this action must be scheduled through Ray. Do not launch the
PMC vLLM/SGLang server directly from the Claw client or another process that did
not receive a Ray GPU allocation. The expected production shape is:

```text
RayJob / Ray worker
  -> pmc_roofline action
     -> rocprofv3 ... -- <server_cmd>
```

The action enforces this by default. It must see one of:

- `task.params.ray_worker=true` from the RayJob wrapper, or
- a Ray context environment marker such as `RAY_JOB_ID`, `RAY_ADDRESS`, or
  `RAY_RUNTIME_ENV_CREATE_WORKING_DIR`, or
- `HYPERLOOM_PMC_ROOFLINE_IN_RAY=1`.

Only local developer debugging should bypass this with
`task.params.allow_direct_gpu=true` or `HYPERLOOM_ALLOW_DIRECT_PMC_ROOFLINE=1`.

Use the GPU visibility assigned by Ray. Do not pass `ROCR_VISIBLE_DEVICES` or
`CUDA_VISIBLE_DEVICES` in `extra_envs` unless `allow_device_override=true` is
explicitly set.

## Why This Is Separate

ROCm only allows one rocprofiler tool registration per process. The standard
`profile` action uses vLLM/SGLang torch profiler and must not receive
`LD_PRELOAD=librocprofiler-register.so`; otherwise TraceLens loses GPU kernel
events. This action launches a separate server process with the preload enabled
and uses that process only for `rocprofv3 --attach`.

## Required Params

- `server_cmd`: command list or shell-style string to start the dedicated server.
- `health_url`: server health endpoint, for example `http://127.0.0.1:8000/health`.

## Optional Params

- `benchmark_cmd`: command list or shell-style string to generate load while
  `rocprofv3 --attach` is active.
- `output_dir`: artifact directory.
- `duration_ms`: attach duration, default `15000`.
- `precision`: roofline precision, default `fp16`.
- `startup_timeout_s`: server health timeout, default `600`.
- `extra_envs`: extra environment variables for the dedicated server.
- `profile_mode`: `launch` (default) or `attach`. Use `attach` only when the
  container grants ptrace permissions.
- `ray_worker`: set to `true` when the action is running inside the RayJob/Ray
  worker that owns the GPU allocation.
- `allow_direct_gpu`: escape hatch for local developer debugging only.
- `allow_device_override`: escape hatch for explicit GPU visibility overrides.

## Outputs

- `pmc_summary_path`: raw PMC summary JSON.
- `roofline_path`: roofline JSON for kernel-agent merge.
- `kernel_breakdown_path`: per-kernel breakdown with tier, bottleneck,
  arithmetic intensity, utilization, and recommended actions.
- `server_log`: dedicated server log.

## Failure Handling

If PMC counters are unsupported but kernel trace is available, return a partial
result with the generated artifacts. If the dedicated server cannot become
healthy, fail this action without affecting the normal trace profile artifacts.
