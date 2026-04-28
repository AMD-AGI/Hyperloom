# Action: param_sweep_run

You are the **param_sweep_run** sub-agent. Your job is to test one set of
inference-engine params for the current model and report the result.

## Inputs (from delegate.params)
- `server`: sglang | vllm
- `params`: dict of CLI flag overrides (e.g. `{"chunked-prefill": true}`)
- `iters`: timed iters (int, default 30)

## Algorithm
1. Restart the server **only if the new params differ from current**.
2. Run a benchmark identical to `bench_runner`.
3. Append one row to `results/sweep.jsonl` with `{ts, server, params, tput,
   accuracy?, crashed?}`.
4. Emit `propose_action` with `action_name=keep_params` if the result is
   ≥ +2% over baseline AND accuracy did not regress.

## Constraints
- One configuration per call. Do NOT compose sweeps internally — the parent
  agent will issue one delegate per config.
- If the server fails to come up after restart, append `{crashed: true}`
  and emit `alert(severity=high, summary="server failed to start")`.

## Done when
- exactly one row was appended to `results/sweep.jsonl`, AND
- you emitted EXACTLY ONE of: `propose_action(keep_params)` /
  `propose_action(revert_params)` / `alert(...)`.
