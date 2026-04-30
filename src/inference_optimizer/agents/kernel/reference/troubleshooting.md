# Troubleshooting — Kernel Agent

Lookup table for common kernel-opt failures + recovery actions.

| Symptom (from script log / RESPONSE prep) | Probable cause | Recovery |
|---|---|---|
| `geak_ray_submit.py` exits rc=124 | Per-task timeout (default 30 min) | RESPONSE `status=failed result.reason="geak_timeout"`; executor pivots |
| `oob_ray_submit.py` exits rc=1 with "OOB_API_KEY missing" | env not propagated through Ray worker | `alert{severity=high, summary="OOB credentials missing in Ray workers"}`; do NOT retry without operator action |
| `apply_patch.sh` aborts after `patch_inductor.py` rc=2 | IR-6 strict mode left on accidentally (`INFERENCE_OPTIMIZER_IR6_STRICT=1`) | RESPONSE `status=failed result.reason="ir6_strict_violation"`; recommend operator unset env or fix argv |
| Re-baseline reports `output_throughput=0` | Server didn't actually start (port busy / model OOM) | RESPONSE `status=failed result.reason="server_didnt_start"`; emit `alert{severity=high}`; do NOT retry without `state.json::current_action` clearing |
| Accuracy gate REVERT (drop > 1%) | Patch broke output correctness | RESPONSE `status=failed result.reverted=True result.accuracy=X`; record kernel name + KB ingest entry so future runs skip it |
| Inbox shows `delegate_dedup_to_terminal` for executor's pivot | Executor accidentally re-emit-ed an old delegate | Not your problem — executor's SKILL handles dedup. Continue normal request handling. |
| Long Ray job blocks turn → SDK timeout → restart loop | Expected for 30-min GEAK rounds | On restart, check `$SESSION_DIR/results/<task_id>/` for partial logs; resume parsing instead of re-submitting (idempotency: GEAK task IDs are persisted) |
| Two consecutive `select_kernels` requests with identical params | Executor rare bug; both will produce same output | OK to respond identically — request msg_ids differ so no idempotency hit; second response is a tiny waste |
| `state.current_action` shows `bench_*` when you want to apply_patch | Soft lane busy (executor benching) | Defer: emit `send_message{topic=heartbeat, body_md="deferring patch — executor active"}` and exit; executor will re-issue when current_action clears |
| Trace file >100 MB | Unfiltered trace (rare; raw rocprofv3 not supported by TraceLens) | RESPONSE `status=failed result.reason="raw_trace_unsupported"`; recommend executor re-run profile to produce filtered TP-0 |

## When to emit `alert` vs RESPONSE failure

- **alert** (`severity=high|critical`): infrastructure problem that
  requires operator attention (Ray cluster down, GPU OOM cluster, env
  misconfig). Watchdog will pick this up + cross-reference KB.
- **RESPONSE failure** (`status=failed`): expected per-task issue
  that the executor should react to (no candidates, timeout, accuracy
  revert). Just close the loop.

## When to update_persona

After a patch succeeds AND you learned something the next request
should know: append a one-liner to your persona via
`update_persona{body_md="..."}`. Examples:

- "Model X TP=8 with FP4 GEMM regresses in run_optimization (tried 3x);
  pivot to non-FP4 candidates"
- "GEAK consistently slower than codex for triton.jit kernels in this
  model class; bias backend selection toward codex"

Keep persona writes ≤200 chars to avoid premature distillation.
