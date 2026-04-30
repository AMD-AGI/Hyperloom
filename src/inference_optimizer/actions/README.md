# Actions registry

Each action ships as a pair:

- `<name>.md`        — system-prompt body fed to the sub-agent (and Executor when proposing).
- `_meta/<name>.yaml`— machine-readable metadata loaded by `ActionRegistry`.

Metadata schema (DESIGN §12.2):

```yaml
name:              <action name>           # required
family:            prep|analysis|shallow|deep_kernel|long|creative|resilience
cost_minutes_p50:  <float>
cost_minutes_p75:  <float>
expected_gain_pct: [<low>, <high>]         # tuple
accuracy_risk:     <0.0..1.0>
crash_risk:        <0.0..1.0>
prerequisites:     [<action_name>, ...]
requires_lanes:    [server_lifecycle, workspace_mutation, benchmark_lane, profile_lane]
allowed_tools:     [Read, Bash, Edit, Glob, ...]
side_effects:      [workspace_write, server_restart, ...]
allowed_modes:     [quick_param_sweep, guided_kernel_opt, marathon_multi_agent]
preferred_backend: claude|codex
preferred_model:   claude-opus-4-7|gpt-5.4
max_turns:         <int>
lease_ttl_sec:     <int>
applicable_when:   [<predicate_id>, ...]   # consulted by Scheduler
```

## Action list (21 — see DESIGN §12.1)

| family       | quick | guided | marathon | actions |
| -----------  | :---: | :---:  | :---:    | ------------------------------------------------ |
| prep         |  ✓    |  ✓     |  ✓       | setup, classify, target_analysis, baseline, bench_runner |
| analysis     |  ✓    |  ✓     |  ✓       | profile                                         |
| shallow      |  ✓    |  ✓     |  ✓       | backends, params, sweep, param_sweep_run, report |
| deep_kernel  |  ✗    |  ✓     |  ✓       | kernel_opt, integrate                           |
| deep_kernel  |  ✗    |  ✗     |  ✓       | deep_kernel_analysis, operator_tuning, vendor_kernel_config |
| long         |  ✗    |  ✗     |  ✓       | comm_optimization, compiler_tuning              |
| creative     |  ✗    |  ✗     |  ✓       | dream, re_explore                               |
| resilience   |  ✗    |  ✗     |  ✓       | recover                                         |

## TODO (IMPL-CHECKLIST Phase 4 §4.22‒4.43)

- [x] Runtime action content lives in this package directory
- [ ] Calibrate `expected_gain_pct` p50 from sprint+marathon historical data
- [ ] Fill in `applicable_when` predicates referenced by Scheduler
