# Action: `re-explore` (STUB)

> Family: **creative** · marathon-only · accuracy_risk=0.0.

When the Scheduler signals `no_more_leverage` but time remains, this action
deliberately picks a previously-discarded family with `0.7×` discount
removed for one round.

## TODO (IMPL-CHECKLIST §4.40)

- [ ] Reset `_diminishing` counter for one chosen family
- [ ] Bound to 1 invocation per session (avoid livelock)
