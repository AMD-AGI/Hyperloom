# Actions — Subskill Index

This directory holds **task-specific playbooks** the Executor reads on
demand. SKILL.md links to these by trigger; this INDEX is a flat
lookup if you want to browse.

| File | Trigger | Mode |
|---|---|---|
| `first_turn.md` | Inbox shows `event{kind=run_started}` | all |
| `after_baseline.md` | Inbox shows `decision{kind=state_updated}` carrying `baseline_tput` | all |
| `retry_after_dedup.md` | Inbox shows `event{kind=delegate_dedup_to_terminal}` | all |
| `request_kernel_optimization.md` | guided/marathon AND `baseline_tput` already set AND no kernel work in flight | guided / marathon |
| `budget_aware_planning.md` | `time_left_minutes < cost_p75 × 1.25` for the action you wanted | all |

> Plan A: kernel-opt + integrate are owned by the **kernel agent**.
> Executor uses `request{target_agent="kernel", kind=...}` instead of
> `delegate(action_name="kernel_opt")` — see
> `request_kernel_optimization.md` for the full three-step protocol
> (select_kernels → run_optimization → apply_patch).

## Adding a new subskill

1. Drop a new `actions/<name>.md` here
2. Add a row to the table above
3. Add a row to the **Subskill index** in `../SKILL.md`
4. Trigger conditions should be **machine-checkable** (a specific event
   topic+kind, or a state field threshold). Vague triggers like "when
   you feel stuck" don't fire reliably.

## Anti-patterns

- **Don't** create a subskill that's only Read once at session start —
  put that content directly into SKILL.md instead.
- **Don't** create subskills with overlapping triggers — pick one or
  merge them.
- **Don't** put hard rules / non-negotiables in subskills — those go in
  `../reference/ir_rules.md` and SKILL.md's "Hard rules" section so they
  are always in context.
