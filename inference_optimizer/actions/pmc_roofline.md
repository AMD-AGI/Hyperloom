# PMC Roofline

Run PMC/roofline profiling in a dedicated server process. Do not use this action
for the normal torch-profiler trace path.

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

## Outputs

- `pmc_summary_path`: raw PMC summary JSON.
- `roofline_path`: roofline JSON for kernel-agent merge.
- `server_log`: dedicated server log.

## Failure Handling

If PMC counters are unsupported but kernel trace is available, return a partial
result with the generated artifacts. If the dedicated server cannot become
healthy, fail this action without affecting the normal trace profile artifacts.
