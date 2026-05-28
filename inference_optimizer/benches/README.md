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

P3 ships placeholder scripts that emit a small JSON marker so the
runner / tool plumbing can be smoke-tested end-to-end. Real probe
implementations land alongside the bench owners; adding a new
`bench_id` requires extending `BENCH_REGISTRY` (a code change, not a
prompt change).
