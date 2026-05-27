# Single-path roofline/profile analysis + +10% watermark refresh + EXPLORE caps

## Summary

Collapse the legacy roofline/profile bifurcation into one Coordinator-
owned analysis lifecycle: roofline (default) or profile
(`--no-enable-roofline`). The Coordinator auto-enqueues an analysis
task at the end of PRELUDE (after baseline) and at every +10%
validated-gain watermark crossing
(`cur_tput / last_roofline_tput >= 1.10`, where
`cur_tput = baseline_tput * (1 + cumulative_gain_validated / 100)`).
While an analysis task is in flight, `specialist` / `explore` /
`kernel_opt` / `integrate` / `deep_kernel_analysis` /
`operator_tuning` / `vendor_kernel_config` dispatches are blocked at
PolicyGate (`rule='wait_for_auto_roofline'`) so downstream actions
always read the freshest `analysis.md` / `last_profile_trace`. The
LLM cannot propose `roofline` or `profile` — both are denied with
`rule='analysis_action_not_llm_proposable'`.

Cleanups bundled into the same PR: drop the
`use_roofline_composite` / `deny_direct_profile` toggles, drop the
N9 policy rule, delete the EXPLORE/KERNEL entry-time auto-roofline
plus the auto_profile fallback, cap specialist parallelism +
proposal_set + explore grid widths, and rewrite the prompts/docs to
the single-path narrative.

## Highlights

- `_enqueue_internal_analysis_task` is the single internal-task
  constructor; the idempotency key is the kind-agnostic
  `internal-analysis-<reason>`.
- Module-level `_ROOFLINE_GATED_ACTIONS` is consulted in every
  dispatch path: `_handle_delegate`, `_handle_propose_action`, and
  the post-Critic `_materialize_approved_proposal`. Critic-approved
  proposals deferred by the gate are re-materialised on analysis
  completion via a small FIFO drain, so a watermark trip mid-Critic
  doesn't waste the round-trip.
- `--enable-roofline` (BooleanOptionalAction, default on) is the
  single mode-select; `_internal_analysis_kind()` picks
  `roofline` or `profile` based on the flag.
- `_register_executors` now registers both `roofline` and `profile`
  unconditionally so `--no-kernel` boots also dispatch the PRELUDE
  analysis task.

## Migration notes

- **Retired CLI flags hard-fail at argparse time.** Operator scripts
  invoking `--use-roofline-composite`, `--deny-direct-profile`, or
  `--force-roofline-after-baseline` (and their `--no-*` spellings)
  now exit 2 with a one-line message naming `--enable-roofline /
  --no-enable-roofline`. The PRELUDE-initial analysis is
  unconditional in the new design, so there is no replacement for
  `--force-roofline-after-baseline=no`.
- **No resume compatibility for the idempotency key rename.** The
  internal-task key migrated from `internal-roofline-<reason>` /
  `internal-profile-<reason>` to the kind-agnostic
  `internal-analysis-<reason>`. Sessions started on commits before
  this PR will enqueue at most one extra analysis task on the first
  resume tick after upgrade. No data loss; the extra task is gated
  by phase + watermark like any other.
- **`last_roofline_tput` field name kept for stability.** The field
  stores the validated-tput anchor at the time of the most recent
  successful analysis task; it is not the tput measured during the
  roofline run (those values diverge by design — see the watermark
  formula in `_current_tput_from_validated_gain`). Renaming was
  considered and deferred to avoid touching every prompt rendering
  site in this PR.
- **Unrelated diff.** `optimizer_runs/robustness_monitor.sh.example`
  carries a tweak unrelated to the analysis lifecycle; included
  because the example script was already part of the working tree
  before the rebase.
