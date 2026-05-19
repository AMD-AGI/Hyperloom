# §3.8 phase 状态机 + plateau 判定 + stop_reason 词表

## 1. 设计目标

把 §3.2 提出的 5 段固化管线落到一个**机器可执行**的状态机, 同时把
v0.6 散在多个 audit 字段里的"什么时候停"凝聚成一个**可枚举的判定
集合**。判定集合本身简单, 落地复杂的部分是 plateau — 它要在不引入
评分系统的前提下识别"这段已经没收益了"。

成功标准:

- `phase` 字段是有限状态; 转移条件机器可枚举。
- `plateau_explore` / `plateau_kernel` 信号有明确的输入信号 + 阈值
  + 触发动作。
- `stop_reason` 词表收敛 (闭合枚举), 每条都对应一个"做完之后下次能不
  能 resume / 操作员该不该介入" 的明确语义。

## 2. 现状回顾

v0.6 的状态机现状:

- `stop_reason` 在多处被写入 (`baseline_failed` / `target_reached` /
  `no_more_leverage` / `time_exhausted` / `max_ticks` / `policy_loop`),
  但词表是隐式的, 没有 ENUM 约束。
- 没有"phase"概念, 所有 action 同一时刻都可申请。
- "已经没收益了"靠 `params_no_promote_streak` / 各种隐式 cooldown 表
  达, 操作员需要把这些字段拼起来才能判断。
- 没有"刚才那段已经空跑很久"的中断信号 (LLM 自己感知, 系统不主动报)。

## 3. 不变量

继承 §3.1 + §3.2 的不变量。本节额外:

### Inv-8.1 — phase 转移由 Coordinator 唯一触发

LLM (任意角色) 不能直接写 SharedState.phase。Coordinator 在每 tick 末
扫退出条件 → 写 phase + phase_history。LLM 可通过
`escalate_strategy_change` *建议*跳 phase, 由 Coordinator 在下一 tick
扫描时与系统判定合并决策。

### Inv-8.2 — 每个退出条件机器可判定

退出条件必须是 `f(SharedState) -> bool` 的 pure function, 不依赖
LLM 的回复 / 用户指令 / 实时 IO。例外: 用户 SIGTERM / 时间超限两条
是事件触发, 不是状态查询, 但同样落到机器可判定的 trigger set。

### Inv-8.3 — stop_reason 词表闭合

任何写入 `state.stop_reason` 的字符串必须属于 §6 词表。PolicyGate
对 SharedState 写入侧加 enum 校验, 非词表值拒绝。

## 4. 转移图与触发条件

```
                    [enter session]
                          │
                          ▼
                ┌─────────────────────┐
                │      PRELUDE         │
                └────────┬────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
   prelude_done             prelude_baseline_failed
   →  EXPLORE               time_exhausted_during_prelude
                            prelude_policy_loop
                            user_stop_requested
                            cortex_t0_failed (新)
                            →  CLOSE
                ┌─────────────────────┐
                │       EXPLORE        │
                └────────┬────────────┘
                         │
            ┌────────────┼─────────────────────────┐
            ▼            ▼                         ▼
   plateau_explore     time_exhausted        robustness_escalated
   explore_phase_     target_reached         no_more_leverage
   budget_exhausted   user_stop_requested
            │            │                         │
   (kernel              CLOSE                     CLOSE
    enabled
    ? KERNEL : SWEEP)
                ┌─────────────────────┐
                │       KERNEL         │
                └────────┬────────────┘
                         │
            ┌────────────┼─────────────────────────┐
            ▼            ▼                         ▼
   plateau_kernel       time_exhausted        robustness_escalated
   kernel_phase_       target_reached         no_more_leverage
   budget_exhausted    user_stop_requested
   kernel_phase_                                   no_kernel_skipped
   aborted_no_trace
            │            │                         │
            ▼            ▼                         ▼
          SWEEP        CLOSE                     CLOSE
                ┌─────────────────────┐
                │       SWEEP          │
                └────────┬────────────┘
                         │
            ┌────────────┼─────────────────────────┐
            ▼            ▼                         ▼
   sweep_done          time_exhausted        robustness_escalated
   sweep_budget_       user_stop_requested
   exhausted
            │            │                         │
            └────────────┴─────────────────────────┘
                         ▼
                ┌─────────────────────┐
                │       CLOSE          │
                └────────┬────────────┘
                         │
                         ▼
                  [exit session]
```

(robustness 的 `recover` action 是 phase-orthogonal, 不出现在转移
图中。)

## 5. plateau 判定逻辑

plateau 不是"凭感觉", 而是用 SharedState 中**事实层数据**机器判定。

### 5.1 plateau_explore

输入信号 (全部都是事实层数据, 不依赖评分):

- `explore_search.winners_history`: 最近 K 轮的 KEEP 列表
- `specialist_rounds`: 最近 K 轮的 specialist 派发记录, 含每轮
  proposal_set 大小
- `last_action_failures`: 最近失败列表
- `cumulative_gain` 在 EXPLORE phase 内的增量

判定 (默认参数; 全部 CLI flag 可调):

```
ε_explore_keep_gain    = 0.5%  (累计增量阈值)
ε_explore_lookback     = 5     (轮数)
ε_explore_empty_streak = 3     (连续空 proposal_set 轮数)
```

**触发 plateau_explore** 当以下两个条件**同时**成立:

- 最近 `lookback` 轮 specialist 派发的 proposal_set 累积 KEEP 增益
  < `ε_explore_keep_gain`
- 最近 `empty_streak` 轮 proposal_set 全空 (specialist 都返 empty=true)

理由: 既要看"试了但没用"也要看"没新东西可试"。任一条单独不足以判
plateau。

### 5.2 plateau_kernel

输入信号:

- `kernel_opt_attempts`: 最近 K 个 kernel_id 的尝试结果
- `kernel_integrate_attempts`: 最近 K 个 integrate 的 KEEP/REVERT 结果
- `rejected_kernel_*`: 已拒的 kernel 集合

判定 (默认参数):

```
ε_kernel_revert_streak  = 3     (连续 REVERT 的 kernel_id 数)
ε_kernel_keep_gain      = 0.5%  (累计增量)
ε_kernel_lookback       = 5     (轮数)
```

**触发 plateau_kernel** 当以下任一成立:

- 最近 `revert_streak` 个 kernel_id 全部 REVERT 或 NEEDS_REVIEW;
- 最近 `lookback` 个 kernel_id 累积 KEEP 增益 < `ε_kernel_keep_gain`。

(plateau_kernel 的 OR 条件比 plateau_explore 的 AND 弱, 因为 kernel 单
次成本远高于 explore variant; 早收敛比晚收敛代价更小。)

### 5.3 phase budget 退出

每段 phase 设最大占用比例 (相对总 wall-clock budget); 超过即强制退出,
即使 plateau 信号未触发。

| Phase | 默认 budget % | 退出 reason |
|---|---|---|
| PRELUDE | 5% | (不强制 budget; baseline 失败 streak 已经覆盖) |
| EXPLORE | 60% | `explore_phase_budget_exhausted` |
| KERNEL | 25% | `kernel_phase_budget_exhausted` |
| SWEEP | 8% | `sweep_budget_exhausted` |
| CLOSE | 2% | (不强制 budget; CLOSE 步骤固定短) |

总和 < 100% 留 padding; budget % 仅作上限, 不要求"必须用完"。

### 5.4 全局停机

无论 phase, 命中即转 CLOSE, reason 取下表:

| 触发 | reason |
|---|---|
| 用户 SIGTERM / `inference_optimizer stop` | `user_stop_requested` |
| `cumulative_gain >= target_gain_pct` (target 是 gain_pct 时) | `target_reached` |
| `current_best.tput >= target_tput_per_gpu` (target 是 tput 时) | `target_reached` |
| wall-clock > `max_minutes` | `time_exhausted` |
| robustness 连续 high alert ≥ N (默认 5) | `robustness_escalated` |
| `policy_denial_streak ≥ 10` 同一 (action, rule) | `policy_loop` |
| `crash_count ≥ 5` | `crash_threshold_exceeded` (新增) |
| EXPLORE 的 specialist 全部 stale 或 KB 全不可达 | `no_more_leverage` |

`no_more_leverage` 是兜底语义: 系统已经没有任何方式产生新提议时使用,
具体触发条件留实施稿决定 (但必须机器可判定)。

## 6. stop_reason 词表 (闭合)

任何写入 `SharedState.stop_reason` 的值必须属于下表; PolicyGate 在
state 写入路径上加 ENUM 校验。

```
target_reached
no_more_leverage
time_exhausted
max_ticks
policy_loop
crash_threshold_exceeded   (v0.8 新增)
robustness_escalated       (v0.8 新增)
user_stop_requested        (v0.8 新增)
baseline_failed            (v0.6 沿用)
prelude_baseline_failed    (v0.8 新增, 与 baseline_failed 互斥, 表示 PRELUDE 内退出)
prelude_policy_loop        (v0.8 新增)
time_exhausted_during_prelude (v0.8 新增)
cortex_t0_failed           (v0.8 新增)
cortex_drain_failed        (v0.8 新增)
plateau_explore            (v0.8 新增, 仅在主动结束 EXPLORE 而非接转 KERNEL/SWEEP 时使用)
plateau_kernel             (v0.8 新增, 同上)
no_kernel_skipped          (v0.8 新增, --no-kernel 时跳 KERNEL 直转 SWEEP, 不算异常)
sweep_done                 (v0.8 新增, 正常 SWEEP 完成进 CLOSE)
```

注意: `plateau_*` 既出现在 phase 转移 reason 列表 (§3.2 §6), 也可能
作为 stop_reason 出现在 *正常退出* 时。两者关系:

- 正常: plateau_explore → 转 KERNEL, 不写 stop_reason。
- 异常: plateau_explore + KERNEL 已被禁用 (`--no-kernel`) → 转 SWEEP,
  不写 stop_reason。
- 操作员关心 stop_reason 仅作 *最终退出原因*; phase_history 记录中间
  转移。

## 7. 接口/契约

### 7.1 phase 退出条件函数集

概念上每个 phase 配 4 个判定:

- `enter_<phase>(state) -> bool` — 是否能进入该 phase
- `exit_normal_<phase>(state) -> Optional[reason]` — 是否应正常退出
  到下一 phase
- `exit_terminal_<phase>(state) -> Optional[stop_reason]` — 是否应直
  接进 CLOSE
- `abort_<phase>(state) -> Optional[stop_reason]` — 是否应紧急中止

每 tick 末, Coordinator 按 abort > exit_terminal > exit_normal 优先级
扫当前 phase。abort / terminal 写 stop_reason; normal 不写 stop_reason
(保留可 resume 性)。

### 7.2 plateau 判定函数

`compute_plateau_explore(state) -> bool`,
`compute_plateau_kernel(state) -> bool` 概念上是 pure function over
SharedState, 由 `exit_normal_<phase>` 调用。

### 7.3 escalate_strategy_change 与 phase 状态机的合流

LLM 通过 robustness 发出 `escalate_strategy_change` 时, 携带
`next_action_hint` 字段。Coordinator 在下 tick 扫退出条件时:

- 如果 hint = "skip_to_kernel" 且当前 EXPLORE → 视作 plateau_explore
  立即触发 (尊重 LLM 判断, 但仍走机器路径写 phase_history)。
- 如果 hint = "skip_to_close" → 视作 robustness_escalated 立即触发。
- 其他 hint → 仅写入日志, 不改 phase。

## 8. 实施步骤

1. **stop_reason ENUM 锁定**: 把 §6 词表列入一份独立 contract, 加进
   PolicyGate 校验。
2. **退出条件 unit 化**: 把每 phase 的 4 个判定写成 pure function
   (实施稿层面再具体), 在 Coordinator 每 tick 末扫描。
3. **plateau 参数化**: 默认参数表 + CLI flag (`--plateau-explore-keep-gain`
   等)。
4. **escalate hint 词表**: 与 robustness 角色统一约定 hint 词表 (
   skip_to_kernel / skip_to_close / extend_explore_budget / 等)。
5. **resume 兼容**: 老 session 的 stop_reason 如不在新词表内, resume
   时 strict 模式拒绝启动, lenient 模式映射到最接近的新词表项 (CLI
   flag `--legacy-stop-reason-mapping=strict|lenient`)。

## 9. 边界条件 / 失败模式

| 场景 | 行为 |
|---|---|
| 同一 tick 多个退出条件同时满足 | 优先级 abort > exit_terminal > exit_normal; 同优先级按 §6 词表中靠前的优先 |
| escalate hint 与机器判定不一致 (LLM 想 skip 但 plateau 没到) | LLM 优先, 但 phase_history 标 `evidence: llm_escalation`, breakdown 中可见 |
| robustness 连发 5 次 high alert 但 LLM 主动 stop_reason = target_reached | target_reached 优先 (业务目标达成), robustness alert 仅作 evidence |
| `no_more_leverage` 触发但用户尚未确认 | session 进 CLOSE, 写 stop_reason; 操作员 resume 时手工传 `--ignore-no-more-leverage` 强制重新 EXPLORE |
| plateau_kernel 触发但 `kernel_enabled=False` | 不可能 (KERNEL phase 在 no-kernel 模式下根本不进入) |
| time_exhausted 在 PRELUDE 命中 | 写 `time_exhausted_during_prelude` (与全局 time_exhausted 区分, 便于操作员定位) |

## 10. 验收标准

- [ ] phase_history 任一行 reason 都属于 §3.2 §6 词表 + 本节 §6 词表
      并集。
- [ ] stop_reason 写入路径有 ENUM 校验 (尝试写非法值 → log warning + 拒)。
- [ ] plateau_explore 在 KEEP 增益累积 ≥ 阈值时**不**触发 (即使
      proposal_set 空) — 验证 AND 语义。
- [ ] plateau_kernel 在 K 个 kernel REVERT 后**立即**触发 — 验证 OR
      语义。
- [ ] phase budget exhausted 触发可观测, breakdown 中显式标 `forced_by_budget`。
- [ ] `escalate_strategy_change{hint=skip_to_kernel}` 在 EXPLORE 内可
      触发提前转 phase。
- [ ] resume 老 session, 旧 stop_reason 映射或拒绝行为符合 CLI flag。

## 11. 依赖与影响面

- **上游**: §3.2 (phase 边界), §3.4 (winners_history 是 plateau 输入),
  §3.5 (specialist_rounds 是 plateau 输入)。
- **下游**:
  - §3.10 SharedState 加 `stop_reason` ENUM 校验, 加 plateau 参数字段。
  - §3.11 PolicyGate 加 stop_reason 写入校验。
  - §3.12 breakdown.phase_timeline 反序列化 phase_history 给可视化。
  - §3.13 milestone M2 (phase + 砍 scoreboard) + M7 (plateau 调参)。

## 12. 哲学回引

本节是**主轴 A** (流程固化由 Coordinator 强制) 的判定层落地; **Inv-1**
phase 字段属 CORE_STATE_FIELDS, 仅 Coordinator 写; **Inv-8.3** stop_reason
词表闭合, 防止 v0.6 那种"散落字符串"式的退出语义。
