# `dynamic_action` action playbook

## Purpose

Dispatch a multi-turn ReAct sub-agent on the **research_lane** to explore
one cross-domain patch combination that no single specialist (with its
own-domain prompt) could surface. dynamic_action is the supplementary
EXPLORE channel for organic synthesis across `>=2` specialist domains;
specialists remain the default path.

dynamic_action shares the `research_lane` physical channel with
specialists (`MAX_RESEARCH_LANE_CAPACITY=6`) but counts against an
independent round-cap so the two pools cannot starve each other.

## Phase / source

- Phase: `EXPLORE` only.
- Source role: `orchestration` only. Any other source is rejected with
  `dynamic_source_violation`.

PolicyGate's `_validate_dynamic_action_dispatch` enforces the §1.2 red
lines at dispatch time. See `dynamic_action.MD` §1.2 and
`action_dynamic_plan/P1_dispatch_skeleton.md` §4 for the canonical
rule set.

## Payload schema (closed)

| Key                     | Type   | Required | Description |
|-------------------------|--------|----------|-------------|
| `motivation_gap_text`   | string | yes      | Why specialists cannot cover this combination (audit only). |
| `scope_domains`         | list   | yes      | `>=2` registered specialist domain keys. |
| `side_effects_declared` | list   | yes      | Action categories the sub-agent expects to touch. |
| `budget_hint`           | string | no       | `low` / `medium` / `high`; default `medium`. |

The field set is intentionally closed; no `notes` / `extra` / free
extension slots. See P1 §3.

`side_effects_declared` may not contain any kernel-owned action, any
metric/accuracy_gate/server lifecycle category, or every entry in
`scope_domains` may not be `kernel`. PolicyGate denies with
`dynamic_side_effects_red_line` / `dynamic_kernel_only_disallowed`.

## EMIT format

```
delegate{
  action_name = 'dynamic_action',
  params = {
    motivation_gap_text   = '<why specialists cannot cover this>',
    scope_domains         = ['kv_cache_specialist', 'serving_specialist'],
    side_effects_declared = ['framework_source'],
    budget_hint           = 'medium',
  },
  idempotency_key = 'dynamic-<round>-<seq>',
}
```

## Capacity

| Knob                              | Default | Source       |
|-----------------------------------|---------|--------------|
| `MAX_DYNAMIC_PER_ROUND`           | 1       | P1 §4.3 + §10 |
| `MAX_DYNAMIC_SOURCED_VARIANTS`    | 1       | P0 Q3 → P1 §4.3 |
| `scope_domains` min length        | 2       | P1 §3 |
| `MAX_RESEARCH_LANE_CAPACITY`      | 6       | shared with specialists |

Failed dispatches do NOT consume `MAX_DYNAMIC_PER_ROUND` (see P1 §4.2):
PolicyGate rejects come back with a structured reason; only successful
dispatches bump the round counter.

## Output

P1 ships a stub executor (`StubDynamicActionExecutor`) that returns
`{proposal_set: [], dyn_id: <generated>, outcome: "stub_empty"}` and
appends one line to `dispatch_history.jsonl`. The empty `proposal_set`
flows through the specialist-equivalent empty path; nothing reaches
the critic or grid runner in P1.

The real multi-turn ReAct runner replaces this stub at P3.
