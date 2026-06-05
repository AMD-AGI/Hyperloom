# P1_01 — 边界所见即所得:`PHASE_LLM_PROPOSABLE_ACTIONS` + R1 单一拒绝源

- **Phase**: P1 · **风险**: 低 · **依赖**: 无 · **后继**: P1_09, P3_18

## 目标

让展示给 LLM 的"phase 允许动作集"**精确等于它真正能提的集合**。当前 `PHASE_ALLOWED_ACTIONS` 是**超集**:含 `roofline`/`profile`/`framework_pr`,但 LLM 一提就被另一条规则(`analysis_action_not_llm_proposable` / `framework_pr_action_not_llm_proposable`)拒,而不是 R1。模型看到的"可用集"是假的,只能靠经验绕开。

## 不变量 vs 策略判定

- `PHASE_ALLOWED_ACTIONS` 作为**内部校验词表**(闭枚举)= INVARIANT(保留)。
- 但"超集同时被用来渲染给 LLM"= 边界模糊,需拆分。`roofline`/`profile` 由 Coordinator 拥有调度(资源,INVARIANT,保留),但**对 LLM 不可提**这件事应通过"它根本不在你的集合里"来表达,而不是两条规则。

## 关键事实(来自审计)

- 内部 analysis enqueue 走 `tasks.create_or_return_existing(... source="coordinator_internal")`(`coordinator.py` 2673–2701),**根本不过 PolicyGate**。→ 因此 `PHASE_ALLOWED_ACTIONS` 含 `roofline`/`profile` **并非内部通道所必需**。
- `INTERNAL_ONLY_ACTION_NAMES`(`protocol/action_surfaces.py` 24–28)= {roofline, profile, replay_warm_recipe}。`framework_pr` 由 `framework_pr_action_not_llm_proposable` 单独挡。

## 改动清单

### 1. 新增 LLM 可提集(`phase_state.py`)
- 在 `PHASE_ALLOWED_ACTIONS`(78–152)之后新增:
  ```
  PHASE_LLM_PROPOSABLE_ACTIONS[phase] = PHASE_ALLOWED_ACTIONS[phase]
      - INTERNAL_ONLY_ACTION_NAMES - {"framework_pr"}
  ```
  并提供 `llm_proposable_actions_for(phase)`(对照现有 `allowed_actions_for` 167–172)。
- `is_action_allowed_in_phase`(155–164)保持基于 `PHASE_ALLOWED_ACTIONS`(内部校验仍用超集),**新增** `is_action_llm_proposable_in_phase(phase, action)` 基于新集合。

### 2. R1 成为唯一面向 LLM 的 phase 拒绝源(`policy.py`)
- `_validate_phase_action`(1288–1346)对 LLM 来源改用 `PHASE_LLM_PROPOSABLE_ACTIONS`(等价于把 internal-only/framework_pr 视为"不在集合内")。
- **删除优先**:`_validate_action_not_llm_proposable`(1219–1269)对 `roofline`/`profile`/`framework_pr` 的独立 reason-code 与 R1 重复 → 合并进 R1。保留对 `replay_warm_recipe` 的处理(若它不在任何 phase LLM 集中,自然被 R1 覆盖)。统一 reason-code 为 `phase_incompatible`,hint 文案说明"该动作由 Coordinator 自动管理,不可提"。
- 调用点同步:delegate 863–865、propose 985–987、request 1130。

### 3. 渲染改用真集合(触类旁通)
- `prompt_builder._section_phase_semantics`(149–168):渲染 `PHASE_LLM_PROPOSABLE_ACTIONS`,不再渲染超集;`roofline`/`profile`/`framework_pr` 行删除或移到一句"以下动作由 Coordinator 自动管理:…"。
- `shared_state.to_phase_status_summary`(3807–3895,`=== Phase ===` 的 `allowed_actions` 行)与 `coordinator.py` 4522–4531 注入点:`allowed_actions` 行改用 `llm_proposable_actions_for`。
- `critic_prompt_builder`(95–105)渲染 phase allowlist:同样改真集合(Critic 评审也应看到 LLM 真实可提集)。

## 连带测试(更新/删除)

| 文件 | 函数 | 动作 |
|---|---|---|
| `test_phase_state_machine.py` | `test_allowed_actions_disjoint_phases`(58)、`test_is_action_allowed_in_phase_handles_unknowns`(76)、`test_policy_gate_phase_strict_*`(263–322) | 新增 `PHASE_LLM_PROPOSABLE_ACTIONS` 断言;R1 测试改为对真集合 |
| `test_delegate_denial_loop.py` | `test_*_with_analysis_*_is_denied`(246–278) | reason-code 从 `analysis_action_not_llm_proposable` 改为 `phase_incompatible`(若合并),或保留旧名做兼容——二选一并锁定 |
| `test_delegate_denial_loop.py` | `test_phase_explore_allowlist_drops_legacy_actions`(297) | 改断言真集合 |
| `test_policy_atom_invariants.py` | `test_framework_pr_still_denied_*`(171) | 同上 reason-code |
| `test_action_catalogue.py` | `test_phase_allowlist_actions_are_live_registry_actions`(147)、`test_action_surface_sets_are_phase_aligned`(162) | 对齐新集合 |
| `test_prompt_builder.py` / `test_role_realignment.py` / `test_prompt_visibility.py` | phase contract 渲染相关 | 改断言真集合渲染 |
| `test_no_llm_propose_profile_hints.py` / `test_prompts_no_propose_roofline.py` | 全部 | 仍须通过(更强:真集合本就不含 profile/roofline) |

## 验证
- 单测:上述文件全绿。
- 不变量:内部 analysis enqueue 仍能在 PRELUDE/watermark 跑(它不过 PolicyGate,故不受影响)——加一条回归测试确认内部 enqueue 不被新集合影响。
- 烟测:启动一次 run,确认 PRELUDE 后 roofline/profile 仍自动触发。

## 回退
- 删除 `PHASE_LLM_PROPOSABLE_ACTIONS` 与 `is_action_llm_proposable_in_phase`,渲染/校验改回 `PHASE_ALLOWED_ACTIONS`,恢复 `_validate_action_not_llm_proposable` 独立分支。

## 残留风险
- 低。纯"展示=强制"对齐 + 合并两条等价拒绝。注意 reason-code 改名会影响依赖该字符串的下游(resume 重放、`test_resume_deferred_proposals` 用的是 `wait_for_auto_roofline` 而非本 code,故不受影响)。
