# After Baseline — Picking the Next Action

**Trigger**: Inbox shows `topic=decision kind=state_updated` whose
`changes` carries `baseline_tput=X` (X > 0). The conductor's
`_maybe_recompute_gain` will have already set `cumulative_gain=0.0` and
exposed it via the same event's `derived` field.

## Decision tree by mode

The conductor injects an **"Available actions for this mode"** table
into your prompt every turn. Use that as the canonical list — the
suggestions below are the typical first move, not the only legal one.

### `quick_param_sweep` (`MAX_HOURS < 2`)

```
delegate(param_sweep_run, params={...})
```

Goal: try a few CONC × ISL/OSL or `--mem-fraction-static` variations
quickly. The action's executor writes a `results.tsv`, and the runner
emits an `update_state(current_tput=best)` for the highest row.

After it lands a non-zero gain → consider `bench_runner` to confirm,
then `report` (which terminates the run gracefully).

### `guided_kernel_opt` (`2 <= MAX_HOURS <= 6`)

```
delegate(profile, params={})
```

Goal: capture a `filtered-TP-0.trace.json.gz` so the kernel agent can
pick the most expensive operators. After `profile` lands → Read
`request_kernel_optimization.md` for the full three-step request
protocol against the kernel agent (select → optimize → apply).

If `profile` returns `kind=profile_skipped` (the existing sglang server
keeps writing to its launch-time `SGLANG_TORCH_PROFILER_DIR` and there's
no fresh trace), that's a **soft success** — move on directly to
`bench_runner` or to your first kernel agent request. **Do NOT
re-delegate `profile`** with the same params; you'll hit
`delegate_dedup_to_terminal`.

> Plan A — DO NOT emit `delegate(action_name="kernel_opt")`. PolicyGate
> will deny with `rule="kernel_owned_by_kernel_agent"`. The path is
> `request{target_agent="kernel", kind="select_kernels", ...}` — see
> `request_kernel_optimization.md`.

### `marathon_multi_agent` (`MAX_HOURS > 6`)

Same first move as guided (`profile`), but you have the full long-running
toolkit available downstream: `framework_rebuild`, `comm_optimization`,
`compiler_tuning`, `dream`, `re_explore`. Don't reach for those until
you've at least tried one kernel-opt loop via the kernel agent.

In marathon mode you also have **Critic + Watchdog + Sage + Kernel** as
co-reactors. Use `propose_action` (not `delegate` and not `request`) for
high-risk candidates — it will broadcast to the parliament for vote,
and you can request the kernel agent after the verdict.

## Common second-move pivots

| Last result | Good next move |
|---|---|
| `param_sweep_run` showed flat curve (no winner) | `backends` (try aiter / FA backend variants) |
| `profile` shows AITER kernels dominate top-3 ops (marathon) | Read `request_kernel_optimization.md` |
| `response{kind=optimization_done, n_succeeded > 0}` from kernel agent | Decide which patches to apply, then `request{kind=apply_patch}` |
| `bench_runner` shows regression vs baseline | DO NOT immediately revert — investigate first; emit `ask_question` to Sage if marathon |

## Update your persona (optional, marathon)

After you observe a non-trivial baseline characteristic, append to your
persona to carry context across the persona-distill cycle:

```json
{
  "intent_type": "update_persona",
  "payload": {
    "body_md": "Baseline observation: model X TP=8 CONC=16 → tput_per_gpu=470 tok/s; latency_p99=320ms. Sets ceiling for 'huge gain' claims later."
  }
}
```

Keep persona writes **≤ 200 chars** to avoid premature distillation.

## DON'T do these

- DON'T propose more than one action this turn — pick the first move.
- DON'T set `predicted_gain_pct` higher than what KB recall suggests is
  realistic for this model class (Brier scoring penalises over-prediction).
- DON'T try to `delegate(action_name="kernel_opt")` in any mode —
  PolicyGate denies (`rule="kernel_owned_by_kernel_agent"`); use
  `request{target_agent="kernel"}` instead.
