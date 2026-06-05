# Loosen Plan — 索引与总纲

> 本目录把 [`lossen_guardian_plan.MD`](../lossen_guardian_plan.MD) 的核心思想拆成**可独立交付、可回退**的步骤,每步一个 MD。
>
> **核心思想**:代码只做"不变量",不做"策略"。把策略性判断(下一步做什么、何时切 phase、选哪个 gap、何时重测、config vs code、探多宽多深)交还给 LLM。
>
> **本轮决策(已与用户确认)**:
> - **范围 = 全部**:`orchestrator/` 三件套 + dynamic_action/specialist/explore-grid 执行器 + `robustness-agent/` ActionLadder + critic 策略性 reject。
> - **策略默认 = 删除优先**:能删的代码/常量/分支直接删干净,只保留不变量;次选才是"降级为 advisory / 放宽上限"。
> - **交付 = 仅规格**:这些 MD 是实施规格,**暂不改源码**;逐步批准后再落地。

---

## 0. 判定准则(每个步骤都遵循)

| | 不变量 (INVARIANT) — 保留,确定性 | 策略 (STRATEGY) — 删除 / 降级 advisory / 放宽 |
|---|---|---|
| 定义 | 违反会破坏**安全 / 正确性 / 资源 / 可恢复性 / 产物契约 / 去重** | 关于"下一步做什么、做多久、走哪条路、探多宽"的**判断**,做错只是优化效率低 |
| 处理 | 原样保留 | 删除优先;无法整删时降级为不 deny 的 advisory,或放宽到资源上限 |

**删除优先**的含义:对每条 STRATEGY 守卫,首选是**删掉代码 + 删掉对应常量 + 删掉对应测试**;只有当它同时承担了某个不变量职责(如数据依赖、产物诚实性)时,才保留该不变量部分并把策略部分拆出来删。

---

## 1. 保留清单(任何步骤都不得删除的不变量)

| 不变量 | 位置 | 步骤需保护 |
|---|---|---|
| 角色→intent 权限矩阵 / source 校验 | `policy.py` 749–756, 2436–2568 | 全程 |
| 路径沙箱 | `policy.py` `_validate_payload_paths` 2518–2546 | 全程 |
| KB 单写者 | `policy.py` 1408–1419 | 全程 |
| `integrate_patch` 必须有 Critic verdict | `policy.py` 1825–1889 | P1_02 / P3_20 |
| sweep/conc_sweep 单例(防并发崩引擎) | `policy.py` 1730–1802 | 全程 |
| fp8_only / kill scope / specialist intent 面 / core-state 单写 | `policy.py` 1373–1382, 2436–2456, 2281–2434, 1084–1098 | 全程 |
| dynamic 红线副作用(kernel/metric/server/magpie) | `policy.py` 2243–2254 | P1_05 / P1_08 / P2_16 |
| 终止态→CLOSE / skip_to_close / baseline 门 / baseline_failure 3 次 abort | `phase_state.py` 955–984, 1002–1038 | 全程 |
| warm-replay 单飞 / IR-6 EXPLORE 硬 force-exit / phase 预算耗尽硬墙 | `phase_state.py` 988–999, 581–659, 1169–1175 等 | P3_17 / P3_22 |
| phase 链单调拓扑 + 优先级(abort>terminal>normal) | `phase_state.py` 1567–1577 | P3_18 |
| baseline 自环去重 + canonical_fingerprint 去重 | `coordinator.py` 4975–5046;`explore.py` 724–749;`_grid_runner.py` 58–74 | P2_10 / P2_15 |
| CLOSE 5 步顺序器 + 安全网 breakdown | `coordinator.py` 3291–3474 | 全程 |
| 资源:lane 租约 / GPU pool / 子进程 wall-clock kill / 每-variant timeout | `resource_lock.py`, `gpu_pool.py`, `specialist_subprocess.py`, `_grid_runner.py` 1380+ | P1_07 / P1_08 / P2_15 |
| task 状态机合法迁移 / objective.reached / MAX_HOURS 校验 | `task_registry.py` 39–46;`objective.py` | 全程 |

---

## 2. 步骤索引

### P1 — 透明化 + 放宽上限 + 廉价删除(低风险,先做)

| 步骤 | 标题 | 主改文件 | 风险 |
|---|---|---|---|
| [P1_01](P1_01_phase_llm_proposable_actions.md) | 边界所见即所得:`PHASE_LLM_PROPOSABLE_ACTIONS` + R1 单一拒绝源 | phase_state / prompt_builder / shared_state / policy | 低 |
| [P1_02](P1_02_remove_explore_grid_silent_reroute.md) | 删掉 explore-grid → propose_action 静默改写;config grid 不再走 Critic 硬门 | coordinator / policy | 中 |
| [P1_03](P1_03_delete_required_next_step.md) | 删除 `_required_next_step` 执行清单(纯事实由 SharedState 提供) | coordinator | 低 |
| [P1_04](P1_04_intervention_mix_dedupe_advisory.md) | intervention-mix 去重 + 删 ESCALATION 指令 + 删 prose MUST | coordinator / shared_state / orchestration.md | 低 |
| [P1_05](P1_05_loosen_exploration_breadth_caps.md) | 删/放宽探索广度 cap(specialist/dynamic 每轮上限、scope≥2、sub_kind) | policy / shared_state / prompt_builder / *.md | 中 |
| [P1_06](P1_06_delete_kernel_opt_minimum_and_web_phase_gate.md) | 删 `explore_attempts_minimum_before_kernel_opt` + Web 工具 phase 限制 | policy | 低 |
| [P1_07](P1_07_loosen_specialist_caps.md) | 放宽 specialist 上限(proposal_set 截断 3、max_turns 8、scorer 12) | specialist_runner / specialist_domains / proposal_scorer / yaml | 低 |
| [P1_08](P1_08_loosen_dynamic_action_react_caps.md) | 放宽 dynamic_action ReAct 上限 + 删机械字段校验(非红线) | dynamic_action_runner/proposal/seed_kit | 中 |
| [P1_09](P1_09_prompt_single_source_descriptive.md) | prompt 单一来源 + 命令式改描述式 + 同步已删守卫的措辞 | orchestration.md / prompt_builder | 中 |

### P2 — 顺序/分析 deny 降级或删除(中风险)

| 步骤 | 标题 | 主改文件 | 风险 |
|---|---|---|---|
| [P2_10](P2_10_downgrade_sequence_denial_action.md) | `_sequence_denial_for_action`:保 baseline 门,删其余顺序 deny | coordinator | 高 |
| [P2_11](P2_11_downgrade_sequence_denial_request.md) | `_sequence_denial_for_request`:保数据依赖,删其余;依赖改 handler 输入校验 | coordinator / kernel_request_handlers | 中 |
| [P2_12](P2_12_delete_roofline_pending_deny.md) | 删 `wait_for_auto_roofline` deny(资源靠 GPU 租约);保留内部 enqueue | coordinator | 中 |
| [P2_13](P2_13_steward_to_advisory_delete_depth_gate.md) | steward 降级 advisory + 删 depth_gate + 删 assess 节流;不再驱动 phase | coordinator / phase_state / shared_state | 高 |
| [P2_14](P2_14_breakdown_validated_flag.md) | 用 `validated=false` 标注替代 stack_rebench deny,保产物诚实 | breakdown / shared_state | 中 |
| [P2_15](P2_15_loosen_explore_keep_and_grid_filters.md) | explore KEEP 阈值参数化 + 删 roofline 硬门 + 删 grid 重排/经验跳过 | explore.py / _explore_roofline_filter.py / _grid_runner.py | 中 |
| [P2_16](P2_16_delete_dynamic_action_mechanical_critic_floor.md) | 删 dynamic_action 机械 critic floor(关键词 reject),交 LLM Critic | dynamic_action_critic/pipeline | 中 |

### P3 — phase 软化 + robustness/critic(高风险,触类旁通)

| 步骤 | 标题 | 主改文件 | 风险 |
|---|---|---|---|
| [P3_17](P3_17_plateau_judges_to_advisory.md) | plateau 判据全降级 advisory + 删 steward 内部 enqueue;phase 只靠硬墙/显式 hint | phase_state / coordinator | 高 |
| [P3_18](P3_18_phase_interleave_and_llm_advance.md) | 打破线性:EXPLORE↔KERNEL 交错 + Orchestration 可显式请求前进 | phase_state / policy / agent_role / coordinator | 最高 |
| [P3_19](P3_19_robustness_ladder_to_advisory.md) | robustness ActionLadder 自动 escalate/prune → alert+hint;kill 仅留资源 | robustness-agent/ / coordinator | 高 |
| [P3_20](P3_20_critic_strategy_vs_safety.md) | critic 只在安全维度 reject,策略维度降 `advise` | critic.md / critic_agent.py / action_registry | 中 |
| [P3_21](P3_21_action_registry_scoreboard_cleanup.md) | action_registry/YAML 隐藏 scoreboard 字段审计与清理 | action_registry / actions/_meta/*.yaml | 低 |
| [P3_22](P3_22_phase_budget_defaults_and_dead_knobs.md) | 调高/简化 phase 预算默认 + 删死旋钮 `steward_continuation_cap` | phase_state / cli | 低 |

---

## 3. 依赖关系(实施顺序约束)

```
P1_01 ──┬─> P1_09 (prompt 措辞依赖最终允许集)
P1_02 ──┘
P2_10 ─> P2_14 (删 stack_rebench deny 前先有 validated=false 标注)
P2_13 ─> P3_17 (steward 先降 advisory,再删内部 enqueue)
P3_17 ─> P3_18 (plateau 先软化,再做交错)
P1_05 ─> P2_16 (dynamic caps 放宽与机械 floor 删除相关)
P3_19 独立(robustness 包),但与 coordinator prune/escalate 处理(9252–9351)联动
```

**建议节奏**:P1 全做完并验证 → P2 顺序做(10→11→12→14→13→15→16)→ P3 谨慎做,P3_18 最后且单独 A/B。

---

## 4. 全局测试策略

调查已得到**每个被改行为对应的测试清单**(见各步骤 MD 的"连带测试"段)。原则:

- **deny 改 allow / 删守卫** → 对应"断言被 deny"的测试**删除或改写**为"断言被 allow / 守卫不存在"。
- **超集→真集合**(P1_01)→ 对齐类测试(`test_action_catalogue` 的 alignment、`test_prompt_*`)按新集合更新。
- **零测试行为**(`explore_attempts_minimum_before_kernel_opt`、`stack_rebench_required` 的 rule、`sub_kind`)→ 无需改测试。
- **新增**:对每个"降级为 advisory"的项,加 1 个"该 action 在该情形下**不再** deny"的回归测试,锁住放权不回退。
- **黄金契约**(README/pr.md):`session_breakdown.json` 形状、`stop_reason` 先于 breakdown(#427)、CLI help —— 每个 P2/P3 步骤必须跑这三项烟测。

高 blast-radius 测试文件(改动集中):`test_required_step_gates.py`、`test_critic_verdict_map.py`、`test_assess_remaining_gaps.py`、`test_phase_state_plateau.py`、`test_delegate_denial_loop.py`、`test_explore_grid_limits.py`、`test_prompt_builder.py`、`test_depth_gate.py`、`robustness-agent/tests/test_signals_progress.py`。

---

## 5. 全局回退策略

- 每步独立成 commit;高风险步(P2_10/13、P3_17/18/19)单独成 PR,便于 revert。
- 可配置项优先用**改默认值**实现放权(保留旋钮),这样回退只需翻默认;删除类改动在 MD 里标注"删除点",回退即还原删除块。
- P3_18(打破线性 phase)用 feature flag/env(如 `INFERENCE_OPTIMIZER_PHASE_INTERLEAVE`)包裹,默认关,验证后再翻默认。

---

## 6. 风险兜底(放权但不失控)

所有放权都由**保留的不变量**兜底:IR-6/预算硬墙防烧光预算;baseline+fingerprint 去重防刷屏;sweep 单例/GPU 租约防资源打挂;`integrate_patch` Critic 门+路径沙箱+dynamic 红线防越权;phase 单调拓扑+终止路由+CLOSE 顺序器+安全网 breakdown 防不可恢复;P2_14 的 `validated=false` 防产物失真。
