# Bench-enabled specialist registry

These bash scripts back the bench ids declared in
`inference_optimizer.orchestrator.specialist_bench.BENCH_REGISTRY`.
They are granted only to bench-enabled specialists (`mode=patch` &
`bench=true`) and are constrained to:

- run inside the specialist worktree (`cd $SPECIALIST_BENCH_WORKTREE`);
- write all output under `$SPECIALIST_BENCH_OUTPUT_DIR`;
- read optional tuning knobs from `$SPECIALIST_BENCH_PARAMS_JSON`;
- finish within the per-bench `wall_clock_sec` budget (runner timeout,
  capped by `MAX_BENCH_WALL_CLOCK_SEC`, is the source of truth);
- never start a server, never run Magpie, never write outside the
  worktree.

The placeholder scripts emit a small JSON marker so the runner / tool
plumbing can be smoke-tested end-to-end. Real probe implementations
land alongside the bench owners; adding a new `bench_id` requires
extending `BENCH_REGISTRY` (a code change, not a prompt change).

**Current status:** `BENCH_TOOL_ENABLED = True` in
`specialist_bench.py`; `BENCH_REGISTRY` is populated and `run_bench`
is advertised to bench-enabled specialists. The probe bodies here are
still lightweight placeholders — replace them with real per-`bench_id`
probes without changing the env contract.
