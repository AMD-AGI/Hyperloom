# M2 — phase 状态机 + 砍 scoreboard

## 1. 设计目标

让 Coordinator 在管线层面持有 **phase** 状态机 (PRELUDE→EXPLORE→KERNEL→
SWEEP→CLOSE), 同时把 v0.6 的评分体系 (`scoring.py` / `MARATHON_PRIORS`
/ `action_scores`) 完整移除。

落地后用户应当看到: prompt 中 "Action scores" 区块消失, 取而代之是
"phase + gaps + warm_start" 三段; 一次冷启 session 在 breakdown 中可
见 5 段 phase 边界 timeline。

## 2. 范围

**包含**:

- SharedState 新增 `phase` / `phase_started_ts` / `phase_history[]` /
  `phase_budget_pct` 字段。
- Coordinator 每 tick 末扫 phase 退出条件 (§3.8 规则)。
- PolicyGate 加 R1 phase_incompatible 规则 + CORE_STATE_FIELDS 扩展。
- prompt_builder 改写: 删除 Action scores 区块, 加入 phase /
  phase_allowed_actions / phase_budget_remaining_pct / gaps /
  warm_start_recipe_summary 字段。
- 删除 `orchestrator/scoring.py` 整个文件 + `MARATHON_PRIORS` 表 +
  Coordinator 中 `_score_action_*` 方法。
- SharedState 加载时迁移老 `action_scores` 字段 (drop 默认)。
- breakdown.phase_timeline 重新定义为外层 phase 边界, 内嵌 actions[]。
- CLI flags: `--legacy-action-scores`, `--max-minutes-prelude /
  -explore / -kernel / -sweep / -close` (phase budget 百分比)。

**不包含**:

- explore 合并 (M3); 本里程碑 EXPLORE 阶段仍按 v0.6 backends/params 跑。
- specialist (M5)。
- plateau 阈值参数化 (M7); 本里程碑用固定默认值。

## 3. 与 M1 的关系

- M1 已提供 cortex_session_id 字段; 本里程碑 phase 状态机进入 PRELUDE
  时 T0 begin 已经被 M1 写好, 不重复。
- T2 在本里程碑里仍按 M1 的"propose_action 即写 hypothetical 边"逻辑;
  M3 后改成 multi-variant。

## 4. 概念交付物

| 交付物 | 说明 |
|---|---|
| 新 SharedState 字段 | phase / phase_started_ts / phase_history / phase_budget_pct |
| phase 转移机制 | Coordinator.tick 末尾扫 enter / exit_normal / exit_terminal / abort 4 类判定 |
| phase_history 写入 | 每次转移写一行, 包含 reason (§3.8 §6 词表) |
| PolicyGate R1 | phase_incompatible (§3.11 §4.1) |
| CORE_STATE_FIELDS 扩展 | + phase / phase_started_ts / phase_history |
| prompt_builder 改写 | 删 Action scores; 加 phase 字段 |
| 评分代码删除 | scoring.py / MARATHON_PRIORS / coordinator._score_action_* |
| SharedState 迁移 | drop / warn 模式 (CLI flag) |
| breakdown.phase_timeline 升级 | 外层 phase + 内嵌 actions; 兼容字段 action_timeline |
| CLI flags | 同上, 默认 phase budget 5/60/25/8/2 |

## 5. 状态机入门细节

### 5.1 enter / exit 判定

每个 phase 配 4 个 *pure function* (概念层):

- `enter_<P>(state) -> bool` — 是否能进入 P
- `exit_normal_<P>(state) -> Optional[reason]` — 是否应正常转下一 phase
- `exit_terminal_<P>(state) -> Optional[stop_reason]` — 是否应直接进
  CLOSE
- `abort_<P>(state) -> Optional[stop_reason]` — 是否紧急中止

每 tick 末扫 **abort > exit_terminal > exit_normal** 优先级, 命中即
转 phase 并写 phase_history。

### 5.2 PRELUDE 退出条件

- exit_normal: `baseline_tput > 0 ∧ stop_reason == None` → reason =
  `prelude_done`, target = EXPLORE。
- exit_terminal: `baseline_failure_streak >= 3` → stop_reason =
  `prelude_baseline_failed`, target = CLOSE。
- abort: 用户 SIGTERM / 超时 / cortex_t0_failed → 对应 stop_reason。

### 5.3 EXPLORE 退出条件 (M2 阶段保守)

M2 还没真正合并 explore, 也没 specialist; 此时 EXPLORE 退出条件简化为:

- exit_normal: `params_no_promote_streak >= 5 ∧ backends_no_promote_streak
  >= 5` → reason = `plateau_explore` (用现有 v0.6 字段近似)。
- 或 EXPLORE 用时 ≥ phase_budget_pct.explore × max_minutes → reason
  = `explore_phase_budget_exhausted`。
- exit_terminal: 同 PRELUDE 的全局停机集合。

注意: 这里用 `params_no_promote_streak` 等 v0.6 字段是 *暂时方案*,
M3 / M5 / M7 三步会逐步替换为 plateau_explore 真定义。本里程碑文档
里要在代码注释 + breakdown.warnings 中显式标 "phase plateau heuristic
provisional, see M7"。

### 5.4 KERNEL / SWEEP / CLOSE

参考 §3.2 §5 的退出条件, M2 阶段直接按 §3.2 落地。

## 6. prompt_builder 改写要点

`prompt_builder.build_orchestration_prompt` 在 M2 后必须:

- 不再读 `state.action_scores`。
- 注入 `phase` (字符串) / `phase_allowed_actions` (列表) /
  `phase_budget_remaining_pct` (浮点)。
- 注入 `gaps[]` (M2 阶段 gap 列表来自现有 last_action_failures +
  current_best 的简单推导, M5 后由 specialist 反馈细化)。
- 注入 `warm_start_recipe_summary` 文本 (M1 已写到 SharedState.warm_start_*,
  M2 在 prompt 中渲染)。

DECISION FRAMEWORK 段重写: 不再讲"按 score 选 top-1", 改讲"按 phase
允许集合内 LLM 自由判断, 优先解决 high-severity gap"。

## 7. 实施步骤 (PR 拆分)

| PR | 内容 |
|---|---|
| 1 | SharedState 新字段 + 迁移 (drop)；phase 推断 (resume 老 session) |
| 2 | Coordinator 每 tick 末扫 phase 退出条件 (4 个判定函数), 写 phase_history |
| 3 | PolicyGate R1 + CORE_STATE_FIELDS 扩展 |
| 4 | prompt_builder 改写 (删 Action scores 区块) |
| 5 | 删除 `scoring.py` / `MARATHON_PRIORS` / coordinator `_score_action_*` |
| 6 | breakdown.phase_timeline 升级 + action_timeline 兼容字段 |
| 7 | CLI flags + phase budget 默认值 |
| 8 | resume 兼容 (老 v0.6 session 处理) |

PR1–4 是 *破坏-但兼容* 的: 删除评分代码后, 老 prompt snapshot 里残
留"Action scores"区块也无影响 (snapshot 是只读的)。

## 8. 验收清单

- [ ] 冷启 session, breakdown.phase_timeline 显示 5 段 (`PRELUDE` /
      `EXPLORE` / `KERNEL` / `SWEEP` / `CLOSE`)。
- [ ] `state.json` 不再含 `action_scores` / `params_no_promote_streak`
      作 *prompt 输入* (字段如果保留, 也仅作内部计数, 不进 prompt)。
- [ ] Orchestration prompt snapshot 中无 "Action scores" 字样。
- [ ] PolicyGate 在 PRELUDE 阶段 propose `kernel_opt` 时返回
      `phase_incompatible`, hint 含 "you are in phase=PRELUDE"。
- [ ] resume v0.6 session, 旧 `action_scores` 字段被 drop, 推断 phase
      为 `EXPLORE`, 写一行 `resumed_from_v06_inferred` 到 phase_history。
- [ ] breakdown.warnings 列出 1 条 "v0.6 → v0.8 migration: action_scores
      dropped" 条目 (warn 模式)。

## 9. 风险与回退

主风险:

- **phase 退出条件用 v0.6 字段近似导致提前 plateau** (M2 阶段). 缓解:
  把阈值取保守 (proxy_streak >= 5), M7 时机替换为真正的 plateau 计算。
- **prompt 改写后 LLM 出现行为退化** (失去 scoreboard 后乱跑). 缓解:
  prompt 中加详细 phase + gaps + warm_start 段; 灰度时跑同一 model 的
  v0.6 / v0.8 双跑对比, 看 cumulative_gain。
- **delete scoring.py 后某处遗漏 import 报错**. 缓解: PR5 之前先在 PR1–4
  把所有 import 路径标 deprecated, 真删时已无引用。

回退:

- 单 PR 回退即可 (PR5 删除是不可回滚的, 但 PR1–4 都可回); 可灰度时
  先回退 PR4 / PR6 prompt 与 breakdown 改动, 让评分恢复。

## 10. 哲学回引

本里程碑是**主轴 A (流程固化优先于评分动态调度)** 的核心落地;
**Inv-1 (事实层单写者)** 通过 PolicyGate R1 + CORE_STATE_FIELDS 守住;
**Inv-9.1 (决策层无评分)** 通过删除 scoring.py 完成。
