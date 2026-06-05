# `dynamic_action` action playbook

## Purpose

Dispatch a multi-turn ReAct sub-agent on the **research_lane** to explore
a patch combination that a single specialist (with its own-domain prompt)
might not surface. dynamic_action is the supplementary EXPLORE channel for
organic synthesis across one or more specialist domains; specialists
remain the default path.

dynamic_action shares the `research_lane` physical channel with
specialists; per-round breadth is bounded by the research_lane / GPU pool
leases, not by a dispatch cap.

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
| `scope_domains`         | list   | yes      | `>=1` registered specialist domain key(s). |
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
| `scope_domains` min length        | 1       |
| `MAX_RESEARCH_LANE_CAPACITY`      | 2 × visible GPU (shared with specialists) |

There is no per-round dynamic dispatch cap: breadth is bounded by the
shared `research_lane` / GPU pool leases. `dynamic_action_round_count`
remains as telemetry and the `dyn-<round>-<seq>` id sequence; only
successful dispatches bump it.

## Output

When no Claude backend is configured, the stub executor returns
`{proposal_set: [], dyn_id: <generated>, outcome: "stub_empty"}` and
appends one closed-schema `SUB_AGENT_DONE` row to
`dispatch_history.jsonl`. The empty `proposal_set` flows through the
specialist-equivalent empty path; nothing reaches the critic or grid
runner.

The multi-turn ReAct runner :class:`DynamicActionRunner` replaces the
stub when a backend is wired.
