# §3.9 砍掉 scoreboard — 让 LLM 直接根据 gaps + KB 决策

## 1. 设计目标

把 v0.6 中"代码维护一份数值化 scoreboard, prompt 渲染 top-12, LLM 选
top-1 propose"的决策模型彻底移除, 决策权回到 LLM, 仅靠 *phase 允许
集合 + 当前 gaps + KB 子图 + specialist proposal_set* 让 LLM 决策。

成功标准:

- `orchestrator/scoring.py` 中所有评分类型 (base_score / score_mult /
  ucb_bonus / aging_bonus / cooldown_until_tick / streak / locked_reason)
  从 SharedState / prompt / breakdown 全部消失。
- LLM prompt 不再包含 "Action scores" 区块。
- v0.6 老 session 的 `state.json.action_scores` 在 resume 时被静默丢
  弃, 不影响新 session。
- 必要的"硬熔断"语义 (kernel_opt 同 kernel_id PARTIAL 上限 / specialist
  domain 反复空提议) 仍然以**事件触发**保留, 不重新引入"分值"。

## 2. 现状回顾

v0.6 评分体系大致由 `orchestrator/scoring.py` + `MARATHON_PRIORS` +
SharedState.action_scores 共同维护:

| 组件 | 内容 |
|---|---|
| `MARATHON_PRIORS[model_class][action]` | 1–10 的初始优先级表; 4 个 model_class |
| `compute_initial_priors_from_metadata` | 从 ActionMetadata 自动算 base_score |
| `apply_keep` / `apply_failure` / `apply_no_promote` | 流式更新 score_mult |
| streak 系统 (`STREAK_THRESHOLD=3`, `STREAK_PENALTY_MULT=0.85`) | 失败/空跑积累惩罚 |
| `cooldown_until_tick` (legacy) | 老的强制冷却 (现在已经改为软 cooldown 但字段仍在) |
| UCB-style + aging bonus | 反过度采样 / 反饥饿 |
| `effective_score` | 综合分; 渲染 top-12 进 prompt |
| `target_gap_multiplier` | 根据 Objective 调整 explore vs deep 的权重 |

为什么砍:

- v0.8 决策层是 phase + KB + LLM 三件套, 评分体系是冗余的"第二决策
  系统"。
- LLM 同时看代码评分 + KB 提示, 容易困惑 (TBO 经验已经验证: 评分让 LLM
  机械跟随 top-1, 失去探索性)。
- 评分代码和 prompt 渲染要双向同步演化; 砍掉后维护成本下降一档。

## 3. 不变量

继承 §3.1 三主轴 + 三不变量。本节额外:

### Inv-9.1 — 决策层无评分

任何系统侧字段都不应表达"action / variant 的优先级数值"。可以表达
"已发生的事实" (KEEP/REVERT 计数 / 累积增益 / streak 计数), 这些是
*事实层*, 仅供 LLM 阅读决策, 不参与代码侧排序。

### Inv-9.2 — 硬熔断仅作数据保护

保留下来的少量"看起来像 cooldown"的逻辑 (例如 kernel_opt PARTIAL ≤ 2,
specialist domain 连 3 轮空), 性质上是**事件触发的硬熔断 / 调度限速**,
不是"基于分值的排序"。它们的目的是防止 LLM 误用资源 (反复跑同一个
明显失败的 kernel), 而不是替 LLM 做"现在该选谁" 的决策。

## 4. 移除范围

### 4.1 SharedState 字段 (从 SharedState 完全删除)

- `action_scores: dict[action_name, ActionScore]`
- `last_action_score_snapshot` (v0.6 内某些路径偶有保存)
- `params_no_promote_streak` (评分流的副产物; 用 `explore_search` 里
  的 winners_history 长度差替代)
- 任何以 `score_*` / `priority_*` / `ucb_*` / `cooldown_until_*` 开
  头的字段

### 4.2 模块 (从 orchestrator 移除)

- `orchestrator/scoring.py` 整个文件
- `orchestrator/coordinator.py` 中 `_score_action_keep` /
  `_score_action_discard` / `_score_action_failure` /
  `_score_action_no_promote` / `_apply_action_score_update` 等私有
  方法
- `MARATHON_PRIORS` 表

### 4.3 prompt 字段

- "Action scores" 区块整个删除
- 各 action 的 `effective_score / cooldown_state / ema_gain / ucb_bonus`
  注解删除
- `target_gap_multiplier` 在 prompt 里的注释删除

### 4.4 breakdown / 报告

- `breakdown.attribution.score_view` (如有) 删除
- `report.md` 中"动作排序"段删除, 改为"phase 内尝试动作时序"叙述

### 4.5 CLI flag

- 与评分相关的 CLI flag (例如 `--scoreboard-top-k`, `--ucb-c`) 删除
- 不引入新的"评分调参"flag

## 5. 保留下来的硬熔断 / 软 cooldown

### 5.1 kernel_opt PARTIAL 上限

来自 v0.6 `_DEFAULT_KERNEL_OPT_MAX_PARTIAL = 2`。语义: 同一 `kernel_id +
patch_path` 的 `run_optimization` 累计 PARTIAL 出 2 次后, 该 kernel_id
自动被纳入 `rejected_kernel_partial_overflow` 集合, 后续 `select_kernels`
不再返回。这是**数据保护**: 防止 60–120 min 一次的昂贵动作反复挣扎。

是否仍叫"cooldown"? 不叫。它是 **事件触发的拒绝集**, 不带时间维度,
不带分值, 与评分体系无关。在 SharedState 里是一个 `set[str]` 字段。

### 5.2 specialist domain 软 cooldown

来自 §3.5 specialist 框架。语义: 同一 EXPLORE phase 内, 某 domain
连续 3 轮派出去都返回 empty proposal_set, 该 domain 在本 phase 剩余
时间内**不再被派**。phase 切换 (例如进 KERNEL 再回不来; 或下次 session
resume) 自动 reset。

字段: `SharedState.specialist_domain_empty_streak[domain] -> int`,
phase 退出时清零。这同样是 *事件触发*, 不带评分。

### 5.3 baseline_self_loop_threshold

来自 v0.6 `_BASELINE_SELF_LOOP_THRESHOLD = 2`。两次 baseline 失败 + 同
fingerprint 第三次 propose → PolicyGate 拒。v0.8 沿用不变, 这是
PolicyGate 层的反循环保护, 与评分无关。

### 5.4 全局 policy_loop 熔断

`policy_denial_streak ≥ 10` 同一 (action, rule) → stop_reason =
`policy_loop`。沿用 v0.6, 不变。

## 6. 决策原料的替代

砍掉评分后, LLM 决策依据来自以下 prompt 字段 (由 §3.3 / §3.6 提供):

| 字段 | 给 LLM 的信号 |
|---|---|
| `phase` | 当前在哪一阶段, 允许做哪些 action |
| `phase_allowed_actions` | 闭合枚举 |
| `phase_budget_remaining_pct` | "我还有多少时间" 的隐式紧迫感 |
| `gaps[]` | 当前未解决的瓶颈 (canonical_id + symptom + layer + 已尝试历史) |
| `warm_start_recipe_summary` | 上次 session 的 best_config / what_worked / what_failed |
| `warm_start_pitfalls` | 已知坑 |
| `kb_subgraph_per_gap` | 实时 KB traverse 子图 (specialist 装配时刷新) |
| `pr_feed` | 最近 PR 列表 |
| `last_action_failures` | 最近事实层失败 |
| `optimization_stack` | 当前累积的 KEEP 构成 |
| `cumulative_gain` | 已经走了多远 |
| `explore_search.winners_history` | 最近几轮的 winner 与 gain (取代评分中的 EMA) |
| `specialist_rounds[]` | 各 domain 上轮派发结果 |

**这些都是事实, 不是分数**。LLM 在 prompt 中 "DECISION FRAMEWORK" 段
看到的指引会从 v0.6 的"按 Action scores 选 top-1"改为 TBO 风格的
"基于 gaps 优先级 (你自己判断) 选下一步"。

## 7. resume 兼容

老 v0.6 session 的 `state.json.action_scores` 字段非空时, v0.8 启动
有两种行为, 由 CLI flag `--legacy-action-scores=drop|warn` 控制:

- `drop` (默认): 静默丢弃整个 action_scores 字典, 写一行
  `state.json migration: action_scores dropped (rows=N)` 日志。
- `warn`: 同 drop, 但额外打 warning 到 logs/cli.log + breakdown
  warnings 段。

不允许"保留 action_scores 但忽略它"模式, 因为 v0.8 prompt builder 不
会读这个字段, 保留只会让 state.json 变胖。

类似处理:

- `params_no_promote_streak`, `score_violation`, `cooldown_until_tick`
  等评分派生字段一律 drop。
- `MARATHON_PRIORS` 既不是 SharedState 字段也不是文件, 直接随代码删
  除即可。

## 8. prompt 改写要点

Orchestration prompt 的 "DECISION FRAMEWORK" 段 (现今由
`prompt_builder` 生成) 改写指导:

- 删除"Action scores top-12"区块;
- 删除"cooldown / locked"提示;
- 增加 "Phase awareness" 段: "你现在在 X phase, 允许做以下 action: …";
- 增加 "Gaps & KB" 段: 列当前 gaps + 每个 gap 对应的 KB 子图摘要;
- 决策指引改为 TBO 风格: "依据 gaps 的层 / 历史 / KB priors / specialist
  proposal_set, 你判断下一步该做什么; 系统不再给分数排序"。

Critic prompt 同步移除任何对 "score view" 的引用; 其评审依据继续
依赖 `judge_bundle` (KB priors / review_constraints), 与评分独立。

## 9. 实施步骤

1. **先冻结**: 一个 PR 把 `scoring.py` 中所有 *外部入口* 标 deprecated,
   保留实现; 同时让 prompt builder 不再读 action_scores (写 warning
   日志)。这个 PR 不动行为, 只切断耦合。
2. **prompt 改写**: 一个 PR 把 prompt_builder 的"Action scores" 区块
   删除, 替换为 phase / gaps / KB 段; 此时仍保留 SharedState.action_scores
   字段的写入, 但不再被读。
3. **删字段**: 一个 PR 删除 SharedState.action_scores 写入, 同时删
   `_score_action_*` 私有方法。resume 路径加迁移 (drop)。
4. **删模块**: 最后一个 PR 删除 `scoring.py`, `MARATHON_PRIORS`,
   相关测试。
5. **CLI flag**: 在 §1 的 PR 里加 `--legacy-action-scores`, 默认 drop。

分 4 个 PR 是为了让回退面始终很小; 第 1/2 PR 出问题可纯改 prompt 回滚,
第 3 PR 出问题改 SharedState 字段读路径回滚, 第 4 PR 是不可回滚的
"清理"。

## 10. 边界条件 / 失败模式

| 场景 | 行为 |
|---|---|
| 老 session resume 时 action_scores 字段 ≈ 100KB | drop, 不影响 |
| LLM 在 prompt 里看不到分数, 反复尝试低价值 action | 由 phase 允许集合 + KB negation 边过滤; explore_search.fingerprint 去重 |
| Critic 误以为评分还在 (老 prompt 习惯) | Critic prompt 已同步改写; 即使 Critic 误判, propose_action 还是要走标准评审 |
| 操作员监控脚本用过 action_scores | breakdown v2 不再有 score_view 段; 提供 §3.15 cheatsheet 翻译指南 |

## 11. 验收标准

- [ ] 一次新 session 启动后, state.json 中无 `action_scores` 字段。
- [ ] Orchestration prompt snapshot 中无 "Action scores" 区块。
- [ ] resume 一次 v0.6 session, drop 行为正确, breakdown.warnings 段
      显式列出 (warn 模式) 或 logs/cli.log 列出 (drop 模式)。
- [ ] kernel_opt 同 kernel_id 第 3 次 PARTIAL 仍被拒 (硬熔断保留)。
- [ ] specialist 同 domain 连 3 轮空 → 该 phase 内不再派 (软 cooldown
      保留)。
- [ ] 不再有任何 prompt 字段 / SharedState 字段 / breakdown 字段含
      "score" / "priority" / "ucb" / "ema" 这种评分语义。

## 12. 依赖与影响面

- **上游**: §3.1 (主轴 A), §3.3 (Orchestration 的决策原料), §3.6
  (KB 子图作为新的 prior)。
- **下游**:
  - §3.10 SharedState 字段表删项。
  - §3.11 PolicyGate 的 score 相关校验 (如果有) 删除。
  - §3.12 breakdown 段删除 score_view / attribution.score。
  - §3.13 milestone M2 的实施。
  - §3.15 cheatsheet 收录"v0.6 → v0.8 操作员翻译" 条目。

## 13. 哲学回引

本节是**主轴 A** 的核心剥离: 决策落 LLM, 评分代码退出舞台;
**Inv-9.1 决策层无评分** 是本节内部不变量, 也是 §3.10 / §3.12 删字
段的依据。
