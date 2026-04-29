# profile — torch profiler trace

**Family**: `analysis` · **Cost**: ~8‑15 min · **Risk**: low

Re‑launch the server with `PROFILE=1` and `SGLANG_TORCH_PROFILER_DIR=...`
exported, run a short workload, then locate `filtered-TP-0.trace.json.gz`
via `process_management.pick_filtered_trace`.

Critical: when `profile` finishes, the **next** action MUST run
`unset_profile_envs` (handled by `process_management.unset_profile_envs`)
to ensure the bench numbers are not biased by tracing overhead.

Outputs:

- artifact: `results/profile/<ts>/filtered-TP-0.trace.json.gz`
- `send_message` topic=`event` "profile captured"
