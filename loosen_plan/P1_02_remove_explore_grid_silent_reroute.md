# P1_02 — 删掉 explore-grid → propose_action 静默改写;config grid 不再走 Critic 硬门

- **Phase**: P1 · **风险**: 中 · **依赖**: 无 · **后继**: P1_09

## 目标

不要静默改写 LLM 的 intent 语义。当前 `delegate{action_name='explore', params={grid:[...]}}` 被**静默重路由**成 `propose_action`,强制走 Critic 的 per-variant `verdict_map` 评审才能 bench(`coordinator.py` 6292–6314)。这违反"所见即所得",且把"config/env grid 要不要先评审"这个**策略**判断变成了硬门。

## 不变量 vs 策略判定

- `integrate_patch`(源码 patch)必须过 Critic = **INVARIANT**(安全,保留,见 P3_20)。
- explore grid 是 **config/env 变体**(`extra_server_args` / `extra_envs`),不是源码 patch。对它做 per-variant Critic 预审 = **STRATEGY**(质量判断),且每个变体本来就会被 benchmark 实测、按 KEEP 阈值裁决——预审属于冗余防御。
- 静默改写 intent 类型 = 边界模糊,**删除**。

## 改动清单(删除优先)

### 1. 删除静默改写(`coordinator.py`)
- 删除 6292–6314 的 explore+grid → `_handle_propose_action` 重路由块。explore delegate 走正常 `_handle_delegate` → 直接 `tasks.create_or_return_existing` → `_pump_dispatcher_once`。

### 2. 处置 explore-grid 的 Critic 评审(二选一,推荐 A)
- **A(删除,推荐)**:explore config grid 不再经 Critic 预审;变体直接进执行器,由 benchmark + KEEP 阈值(见 P2_15,阈值参数化)裁决。删除/简化 `_handle_verdict_map`、`pending_proposals` 中针对 explore 的分支。
- **B(降级 advisory)**:保留 Critic 对 explore 变体的"建议",但**不阻断 bench**——verdict 仅作为 advisory 注入 prompt,执行器照跑全部变体。

> 选 A 与"删除优先"一致;选 B 保留更多评审信号但复杂度高。MD 默认按 A 写,落地时如需保留评审改 B。

### 3. EMIT hint 透明化(触类旁通)
- `prompt_builder._format_grid_injection_hint`(327–354)与 `orchestration.md`:明确"带 grid 的 explore 用 `delegate{action_name='explore', params={grid:...}}`,变体直接 bench",删除任何"会被改写/会进 Critic verdict_map"的旧描述。

## 连带测试(大面积)

`test_critic_verdict_map.py` 是本步最大 blast-radius:

| 函数 | 动作(按方案 A) |
|---|---|
| `test_delegate_explore_with_grid_routes_to_pending_proposals`(726) | **删除/反转**:断言改为"直接建 task,不进 pending_proposals" |
| `test_delegate_explore_with_empty_grid_does_not_route_to_critic`(764) | 删除(不再有 reroute 概念) |
| `test_delegate_non_explore_action_does_not_route_to_critic`(787) | 删除 |
| `test_verdict_map_*`(355–625,materialise/filter 等) | 若方案 A 删除 explore verdict_map:删除这些;若仍保留 verdict_map 给别处则保留 |
| envelope schema 的 verdict_map 解析(83–191) | 保留(schema 本身无害,除非彻底删 verdict_map) |
| `test_role_realignment.py::test_critic_prompt_includes_phase_review_contract`(83) | 同步 critic prompt 措辞 |

> 注意:`verdict_map` schema 与 Critic 的 phase-review 协议若别处仍用,需确认 explore 是唯一消费者再决定整删。审计显示 verdict_map 主要服务 explore-grid;若确认唯一,可整块删 verdict_map(schema + handler + 测试),进一步简化。

## 验证
- explore delegate(带 grid)端到端跑通:变体被 bench、KEEP 入 stack,无 Critic 预审阻断。
- `integrate_patch` 仍被 Critic 门拦(P3_20 不变)——回归确认源码 patch 安全门未受影响。
- 烟测:一次 EXPLORE 多变体 round 正常产出 winners_history。

## 回退
- 恢复 6292–6314 重路由块与 verdict_map handler/测试。

## 残留风险
- 中。去掉 explore 变体预审后,"明显坏的变体"也会消耗一次 bench。兜底:每-variant timeout(`_grid_runner` 1380+,保留)、explore overtime kill(P2_15 保留为资源超时)、fingerprint 去重(保留)。质量信号改由实测承担,更符合"以测量为准"。
