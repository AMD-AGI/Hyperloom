# dynamic_action sub-agent bench registry

These bash scripts back the bench ids declared in
`inference_optimizer.orchestrator.dynamic_action_tools.BENCH_REGISTRY`.
P3 §4.1.c constrains them to:

- run inside the sub-agent worktree (`cd $DYNAMIC_BENCH_WORKTREE`);
- write all output under `$DYNAMIC_BENCH_OUTPUT_DIR`;
- finish within the per-bench `wall_clock_sec` budget (runner timeout
  is the source of truth);
- never start a server, never run Magpie, never write outside the
  worktree.

The placeholder scripts emit a small JSON marker so the runner / tool
plumbing can be smoke-tested end-to-end. Real probe implementations
land alongside the bench owners; adding a new `bench_id` requires
extending `BENCH_REGISTRY` (a code change, not a prompt change).

**Current status:** `BENCH_TOOL_ENABLED_V1 = False` in
`dynamic_action_tools.py`. `BENCH_REGISTRY` is empty and the
sub-agent's tool surface does not advertise `run_bench`. The
placeholder scripts here remain as scaffolding; flip
`BENCH_TOOL_ENABLED_V1` to `True` together with real bench bodies
(and re-populate `BENCH_REGISTRY`).
