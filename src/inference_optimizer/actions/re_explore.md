# re_explore — break out of a local optimum

**Family**: `creative` · **Cost**: ~5‑12 min · **Risk**: zero (read‑only)

Triggered when the scheduler is about to declare `no_more_leverage`
prematurely. Critic reviews the action history and proposes a
`re_explore` plan that tries actions whose `diminishing` factor pushed
them out of contention.
