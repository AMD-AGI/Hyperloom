# P1_05 — 删/放宽探索广度上限

- **Phase**: P1 · **风险**: 中 · **依赖**: 无 · **后继**: P1_09, P2_16

## 目标

让模型"探得宽"。当前一堆 breadth cap 把每轮可并行/可组合的探索压得很紧,这些都是 STRATEGY,唯一该保留的硬上限是**资源派生**(GPU 数)。

## 当前上限(审计)

| 项 | 常量/位置 | 当前 | 性质 |
|---|---|---|---|
| 每轮 specialist 变体数 | `MAX_SPECIALIST_SOURCED_EXPLORE_VARIANTS=1`(policy 269);`_validate_explore_grid_size` 1623–1642(`explore_specialist_grid_max_one`,实际绑 `research_lane_capacity`) | ≤ research_lane_capacity | STRATEGY |
| 每轮 dynamic 变体数 | `MAX_DYNAMIC_SOURCED_VARIANTS=1`(policy 304);1651–1662 | ≤1 | STRATEGY |
| 每轮 dynamic 派发次数 | `MAX_DYNAMIC_PER_ROUND=1`(policy 303);2257–2272 | ≤1 | STRATEGY |
| dynamic 跨域要求 | `dynamic_scope_too_narrow` 2169–2180 | ≥2 域 | STRATEGY |
| specialist sub_kind | 1973–1983 | 必须在目录内 | STRATEGY |
| research_lane_capacity 默认 | CLI 默认 4(cli 4849–4862),clamp 到 `2×GPU`(policy 225–234) | 4 | STRATEGY(默认)/ INVARIANT(2×GPU 上限) |
| GPU specialist pool / max_turns≤16 | policy 2040–2060 / 2006–2015 | — | INVARIANT(资源,保留) |

## 改动清单(删除优先)

### 1. 删 specialist 每轮变体硬上限(`policy.py`)
- 删除 `_validate_explore_grid_size` 中 specialist 变体计数与 `explore_specialist_grid_max_one`(1623–1642)。并发由 **lane 租约 / GPU pool**(资源不变量)自然限制,无需在 grid 校验里再压一道。
- 删 `MAX_SPECIALIST_SOURCED_EXPLORE_VARIANTS`(269)、`_effective_specialist_grid_cap`(1587–1606)。

### 2. 删 dynamic 变体/每轮上限(`policy.py`)
- 删 `MAX_DYNAMIC_SOURCED_VARIANTS`(304)与 1651–1662 的 `dynamic_sourced_variant_cap_exceeded`。
- 删 `MAX_DYNAMIC_PER_ROUND`(303)与 2257–2272 的 `dynamic_round_cap_exhausted`。
- 删 `dynamic_action_round_count` 的"为了 cap"用途(`shared_state.py` 913, 3267–3279;`coordinator.py` 985/6562/6765/6767 的 reset/读取)——若该计数仅服务 round cap,整删;若 telemetry 有用,保留为中性计数但不参与 deny。

### 3. 放宽 dynamic 跨域要求(`policy.py`)
- `dynamic_scope_too_narrow`(2169–2180):从 ≥2 域放宽到 **≥1 域**(允许单域深挖也能用 dynamic_action),或整删该校验。**保留** `dynamic_kernel_only_disallowed`(2152–2168,红线)与 `dynamic_side_effects_red_line`(2243–2254,红线)。

### 4. 放宽 specialist sub_kind(`policy.py`)
- 删 1973–1983 的 sub_kind 目录限制(自由 sub_kind)。**保留** domain tag 校验(`specialist_unknown_domain` 1945–1955)与其余 schema/资源校验。

### 5. research_lane_capacity 默认(`cli.py` / 触类旁通)
- 提高 CLI 默认(4849–4862)到等于 ceiling(`2×GPU`),或直接让默认 = `research_lane_ceiling()`。**保留** clamp 到 `2×GPU`(资源不变量)。

### 6. 同步 prompt/文档(触类旁通)
- `prompt_builder._format_grid_injection_hint`(327–354)、`_format_emit_hint` 中 dynamic 段(302–319):删除"至多 1 个 specialist/dynamic 变体""每轮至多 1 次 dynamic""≥2 域"等已删约束的描述。
- `orchestration.md`(54, 79–101, 181–188 相关行)、`actions/specialist.md`(50–51)、`actions/dynamic_action.md`(12, 60–65)、`SKILL.md`(213):同步措辞。
- `actions/_meta/dynamic_action.yaml`(18,22 max_turns)、`specialist.yaml`(27 max_turns)放宽(见 P1_07/P1_08)。

## 连带测试

| 文件 | 函数 | 动作 |
|---|---|---|
| `test_explore_grid_limits.py` | `test_constant_is_one`(69)、`test_denies_two_specialist_variants*`(73–84)、`test_cap_tracks_*`(176–214) | **删除**(cap 没了);保留 GPU pool 资源测试(229–250) |
| `test_dynamic_action_dispatch.py` | `test_*_scope_domains_too_narrow_denied`(125)、`test_*_round_cap_exhausted_denied`(178) | 删除或改为"≥1 域允许 / 多次派发允许" |
| `test_dynamic_action_invariants.py` | `test_inv_dynamic_sourced_cap_*`(700)、`test_inv_scope_domains_dedup_*`(721)、`test_inv_round_cap_*`(752) | 删除策略 cap 用例;**保留**红线类(kernel-only / side-effects) |
| `test_per_domain_prompts.py` | `test_research_lane_capacity_is_core_state_field`(920) | 保留(core-state 不变量);默认值断言更新 |
| `test_specialist_concurrent_dispatch.py` | `test_cli_default_research_lane_capacity_is_4`(323)、`test_cli_clamps_*`(333) | 默认值断言更新;clamp 测试保留 |
| `test_dynamic_action_orchestration_prompt.py` | `test_*_emit_hint_includes_every_reason_code`(174)、`test_round_cap_value_visible_in_prompt`(349) | 删除已移除 reason-code/cap 的断言 |

## 验证
- 一个 EXPLORE round 可并行派发 >1 specialist(直到 lane/GPU 资源上限)、grid 可含多 specialist/dynamic 变体。
- 红线仍生效:dynamic 不能声明 kernel/metric/server/magpie 副作用;GPU specialist 超 pool 仍被拒。
- 烟测:research_lane_capacity 默认值生效,clamp 到 2×GPU。

## 回退
- 恢复常量与校验块、prompt 描述、测试。

## 残留风险
- 中。放宽并发后单 tick 可能派发更多 GPU specialist——由 **GPU pool / lane 租约**(保留)兜底,不会超物理资源。注意 `research_lane` 在 SQLite 的租约容量(coordinator 311–313)随默认值上调要同步。
