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

PolicyGate's `_validate_dynamic_action_dispatch` enforces the red-line
checks at dispatch time.

## Payload schema (closed)

| Key                     | Type   | Required | Description |
|-------------------------|--------|----------|-------------|
| `motivation_gap_text`   | string | yes      | Why specialists cannot cover this combination (audit only). |
| `scope_domains`         | list   | yes      | `>=2` registered specialist domain keys. |
| `side_effects_declared` | list   | yes      | Action categories the sub-agent expects to touch. |
| `budget_hint`           | string | no       | `low` / `medium` / `high`; default `medium`. |

The field set is intentionally closed; no `notes` / `extra` / free
extension slots.

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

| Knob                              | Default |
|-----------------------------------|---------|
| `MAX_DYNAMIC_PER_ROUND`           | 1       |
| `MAX_DYNAMIC_SOURCED_VARIANTS`    | 1       |
| `scope_domains` min length        | 2       |
| `MAX_RESEARCH_LANE_CAPACITY`      | 6 (shared with specialists) |

Failed dispatches do NOT consume `MAX_DYNAMIC_PER_ROUND`: PolicyGate
rejects come back with a structured reason; only successful dispatches
bump the round counter.

## Output

When no Claude backend is configured, the stub executor returns
`{proposal_set: [], dyn_id: <generated>, outcome: "stub_empty"}` and
appends one closed-schema `SUB_AGENT_DONE` row to
`dispatch_history.jsonl`. The empty `proposal_set` flows through the
specialist-equivalent empty path; nothing reaches the critic or grid
runner.

The multi-turn ReAct runner :class:`DynamicActionRunner` replaces the
stub when a backend is wired.
