<!-- when: a specialist has been dispatched and you need to intervene, redirect, or cancel it -->
<!-- phase: FRAMEWORK_AGENT -->
# Specialist rescue moves

Nothing in the prompt reports in-flight specialists: the prompt renders
between blocking actions, so a running specialist is exactly what you are
waiting on and never appears here. Never read silence as "nothing is
running". Two signals do reach you:

- **`specialist_progress` inbox observations** — pushed whenever a
  specialist rewrites its checkpoint, carrying `task_id`, `elapsed_sec`,
  summary, proposal count, findings and `residual_questions`. Sparse (often
  2-3 per specialist, the first lagging dispatch by minutes), so read each
  as a sample of work that has been running unobserved.
- **`get_running_tasks`** — the live view. Call it whenever a
  `specialist_progress` lands, before a phase change, and when a stretch of
  turns has passed with no specialist news.

Elapsed time alone decides nothing: an offline autotune legitimately runs
for an hour, a five-minute agent can already be wedged. Judge on what you
asked for, whether successive checkpoints advance or repeat, and what is
queued behind the lane or GPUs it holds.

## Moves for a single task

- `send_message{to='specialist:<task_id>', body_md}` — lands in its inbox
  and it acts without restarting. Prefer this when the agent works well but
  on the wrong question, or to answer its `residual_questions`.
- `extend_lease{task_id, extra_sec, reason}` — grows the lease TTL and the
  expiry of the lanes and GPUs the task holds, in bounded steps, and moves a
  running specialist's wall-clock kill deadline out by `extra_sec`, never
  past the phase end. The observation's `wall_budget_extension_sec` is how
  far the deadline has moved in total; `0` means the phase has no room left.
  An expired lease stops nothing by itself; extend a specialist whose work
  needs longer than its wall-clock budget.

There is no single-task cancel. The only cancel is `prune_branch`, and it
takes a whole family.

## Move for the queue

- `prune_branch{family, reason, scope='queued'}` — cancels every *queued*
  task of one family and leaves the family usable. Use it when a backlog
  outlived its purpose: several turns queued the same measurement before the
  first one returned, and the answer is now in hand. The default
  `scope='family'` instead retires the action for the rest of the run, so
  reach for it only when the family itself is a dead end. Queued baselines
  are drained automatically once `baseline_tput > 0` (a `baseline_drain`
  observation reports what was cancelled); this is the manual equivalent for
  any family.
