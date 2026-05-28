# P8 — Resume 语义（重启时 Abandoned）

> 第八阶段的对外可见产物：Coordinator 重启时未完成的 dynamic action
> 不悬空——全部标 `ABANDONED`，artifact 保留，由 orchestration 在下一
> tick 自行决定是否重派；不引入"续跑"机制。
>
> 对应 dynamic_action.MD §3.9。

---

## 1. 目标

让 Coordinator 重启对 dynamic action 是**安全的**——重启不会让 dynamic
action 卡在某个非终态（leak resource lane、悬空 task queue 条目、
worktree 未清理、SharedState summary 永远不更新），但也不引入复杂的
续跑逻辑。

具体可观测的产物：

1. Coordinator 启动 hook 扫 `agents/orchestration/dynamic_actions/*/`，
   识别处于非终态的 dynamic action；
2. 每条非终态 dynamic action 转 `ABANDONED` 终态，写回 SharedState
   summary；
3. 所有相关 worktree（`runs/dynamic/<dyn_id>/worktree/`）清理；
4. 相关 git branch（`dynamic-<dyn_id>`）删除；
5. SharedState 中相关 summary 更新；
6. orchestration 在下一 tick 看到 `ABANDONED` 状态后可自主决定是否
   重新派发。

---

## 2. 触及的架构平面

| 平面 | 改动性质 | 中心思想 |
|---|---|---|
| Coordinator 启动 hook | 新加 dynamic action 扫描逻辑 | 与既有 specialist resume hook 平级新增 |
| Worktree 清理 | 复用既有 worktree 清理机制 | 仅扩展模式匹配的 branch 名 |
| SharedState summary 更新 | 沿用 P6 节点 | 重启 → ABANDONED 是 P6 §5 节点 D' |
| Resource lane 释放 | 沿用既有 lane 释放机制 | Coordinator 启动时 lane 状态本身就重置 |

---

## 3. "不续跑"原则

### 3.1 为什么不续跑

dynamic sub-agent 的 multi-turn ReAct 探索强依赖：

- **worktree 状态**：sub-agent 中途可能已经在 worktree 内做了一些试
  apply（apply_patch_in_worktree 工具），状态不可重建；
- **外部模型 session**：每轮 `claude` 子进程是无状态的（Q1 = (b)
  决策），但内部模型可能已积累了短期 context；重启会丢；
- **micro-bench 结果**：scratch 目录可能已经有部分 bench 结果，但
  没有元数据指明是哪轮哪次的（journal 是 markdown 文本，不是结构化
  状态）；
- **journal 解析复杂度**：续跑要求 runner 能把 journal 反序列化为
  状态机，对解析鲁棒性要求高，bug 面大。

跨进程恢复成本高 + 收益低（v1 还没证明 dynamic action 的命中率值得
投入续跑机制）→ **直接放弃续跑**。

### 3.2 替代策略

让 orchestration 在下一 tick 看到 ABANDONED 状态后**自主决定**是否
重派。这与 §1.7 "多次失败时让 orchestration 自主决定回退" 的设计哲学
一致——重启与失败在 orchestration 视角下都是"这条 dynamic action 没
拿到结果"，处理方式同源。

### 3.3 重派 = 新 dyn_id

如果 orchestration 决定重派一条同 motivation 的 dynamic action：

- 必须是新的 `dyn_id`（不能复用原 dyn_id）；
- 新 dispatch 走完整的 P1 PolicyGate / P2 装配 / P3 runner 全流程；
- 原 abandoned 的 artifact 保留为审计记录；不复用其 seed kit 或 journal。

---

## 4. 启动 hook 流程

### 4.1 触发时机

Coordinator 启动后、进入主循环之前，与既有 specialist resume hook
**同步**执行（也可以在同一个 hook 内统一处理；见 §4.4）。

### 4.2 扫描逻辑

```
对每个 $SESSION_DIR/agents/orchestration/dynamic_actions/<dyn_id>/:
    读 spec.json 与（如存在）telemetry.json
    从 SharedState.dynamic_actions[dyn_id].status 读当前状态
    if status 在非终态集合 {DISPATCHED, SUB_AGENT_RUNNING,
                            AWAITING_CRITIC, INTEGRATING}:
        → 转 ABANDONED
        → 更新 SharedState summary（status, last_outcome, updated_at）
        → 清理 worktree (if exists): runs/dynamic/<dyn_id>/worktree/
        → 删除 git branch dynamic-<dyn_id>
        → 在 dispatch_history.jsonl 追加一条 abandoned 记录
        → 如果还在 task queue 中，移除（既有 queue 清理机制）
    else:
        no-op（终态本就稳定）
```

### 4.3 SharedState 不一致的兜底

可能出现"artifact 目录存在但 SharedState summary 不存在"的 corner
case（如 P2 dispatch 写盘成功后、SharedState save 之前 Coordinator
崩溃）：

- 重建一条 summary，初始化为 `ABANDONED`；
- 该条 summary 用 spec.json 中的 dispatched_at / scope_domains /
  motivation_gap_short 填充；
- 视为正常 abandoned 处理。

反向 corner case："SharedState summary 存在但 artifact 目录不存在"
（异常磁盘删除等）：

- 把 summary 标 `ABANDONED` + 加特殊 flag `artifact_missing=true`；
- 在 alert / log 中记录此异常（说明有外部干预）；
- 不主动尝试重建 artifact。

### 4.4 与 specialist resume 的合并 vs 平级

specialist 的重启清理逻辑（见既有 worktree 孤儿 branch 清理）已经存
在。dynamic action 的清理可以：

- **(a) 合并**：在同一个 hook 函数里一并扫 specialist 与 dynamic 的
  worktree；
- **(b) 平级新建**：新建独立 hook，先扫 specialist 后扫 dynamic（或
  反之）。

**DEFAULT = (b) 平级新建**。理由：

- 合并会让 hook 函数职责膨胀，未来加新的 sub-agent 类型需要再次合并
  （耦合不断增长）；
- 平级新建让每条 sub-agent 类型的清理逻辑独立，回归测试可以分别
  enable / disable；
- 启动开销可忽略——扫 dynamic 目录 O(N) 其中 N = 历史 dynamic action
  总数（v1 上限：每 round 1 条 × round 数 ≈ 几十条）。

---

## 5. Worktree 与 branch 清理

### 5.1 worktree 清理

- 路径：`$SESSION_DIR/runs/dynamic/<dyn_id>/worktree/`；
- 操作：`git worktree remove --force` + 若失败回退到 `rm -rf`；
- 失败处理：log warning，不阻断重启；下次重启再清理。

### 5.2 branch 清理

- branch 名：`dynamic-<dyn_id>`；
- 操作：`git branch -D dynamic-<dyn_id>`；
- 不存在 → no-op；
- 失败 → log warning，不阻断。

### 5.3 不在此清理的内容

- **artifact 目录**：永远保留（审计需要）；
- **SharedState 中 KEPT 的 dynamic action 对应的 patch**：已经
  `_promote_to_shared_state`，与 specialist KEPT patch 等价处理（既有
  机制保证）；
- **scratch/ 子目录**：随 worktree 一起清理（位于 worktree 内）。

---

## 6. dispatch_history.jsonl 的 abandoned 记录

每个被标 abandoned 的 dynamic action 在
`agents/orchestration/dynamic_actions/<dyn_id>/dispatch_history.jsonl`
追加一条结构化记录：

| 字段 | 含义 |
|---|---|
| `event` | 固定 = `"abandoned_on_resume"` |
| `ts` | abandoned 时间 |
| `previous_status` | 重启前的 status |
| `coordinator_session_id` | 当前 Coordinator session 标识 |
| `worktree_cleanup_outcome` | success / partial / skipped |

### 中心思想

- abandoned 不是 silent 操作——必须有结构化记录可被事后审计；
- previous_status 帮助调试"是 sub-agent 跑到一半被打断了，还是 critic
  审查中被打断了"。

---

## 7. orchestration 视角下的 abandoned

orchestration 在下一 tick 通过 P6 注入的 summary 看到：

```
- dyn-3-1 [ABANDONED] scope=[kv_cache,scheduler]
  motivation: "trade-off between cache layout and scheduler ..."
  artifact: agents/orchestration/dynamic_actions/dyn-3-1/
```

orchestration 可以基于以下信息自主决定：

- 这条 motivation 是否仍然有效（roofline / profile 可能已变）；
- 是否值得重派（abandon 不是失败，但消耗了一次 round 机会的痕迹）；
- 是否切回 specialist 通路解决相同 motivation。

**关键**：prompt 不引导重派 / 不引导跳过——LLM 完全自主。

---

## 8. 边界情况

### 8.1 重启发生在 KEPT/REVERTED 之后、SharedState save 之前

可能存在"patch 已 apply 到 framework_source、optimization_stack 已
更新，但 SharedState.dynamic_actions[dyn_id].status 还是 INTEGRATING"
的 race。

处理：

- 启动 hook 看到 status = INTEGRATING → 标 ABANDONED；
- 但实际 patch 状态可能与 baseline 不一致（已 apply）；
- 这条由**既有的 SharedState resume + git status 校验机制**兜底（与
  specialist patch 同源问题，dynamic 不引入新的兜底逻辑）。

### 8.2 重启时 task queue 中还排着 dynamic action 的 task

- 启动时已经清理 task queue（既有逻辑）；
- summary status 已转 ABANDONED；
- 不会出现"重启后 task queue 重新派发同一个 dyn_id 任务"的情况。

### 8.3 重启时 worktree 处于 git apply -R 失败后的中间态

- 启动 hook 优先 `git worktree remove --force` 强制删除；
- 强制删除失败 → `rm -rf` 兜底；
- 不尝试"修复"中间态（dynamic action 不允许复杂 recovery）。

### 8.4 多次重启同一 dyn_id

- 第一次重启：status = SUB_AGENT_RUNNING → ABANDONED；
- 后续重启：status = ABANDONED（终态）→ no-op（§4.2 末分支）。

---

## 9. 依赖与前置条件

P8 依赖：

- P2 的 artifact 目录布局（决定扫描路径）；
- P6 的状态机（决定哪些状态是"非终态"）；
- 既有 worktree 清理机制（specialist resume 的实现可参考）。

P8 与 P5 / P6 / P7 之间无强依赖关系；可与它们并行实施。

---

## 10. 验收信号

| # | 测试场景 | 期望结果 |
|---|---|---|
| 1 | 重启时存在 1 个 SUB_AGENT_RUNNING 状态的 dyn_id | 转 ABANDONED；worktree 清理；branch 删除；dispatch_history.jsonl 追加 abandoned 记录 |
| 2 | 重启时存在多个非终态 dyn_id（如 1 个 DISPATCHED + 1 个 AWAITING_CRITIC + 1 个 INTEGRATING） | 全部转 ABANDONED；逐条独立处理 |
| 3 | 重启时只有终态 dyn_id（如 KEPT / TIMED_OUT） | 全部 no-op |
| 4 | artifact 目录存在但 SharedState summary 不存在 | 重建 summary，标 ABANDONED |
| 5 | SharedState summary 存在但 artifact 目录不存在 | 标 ABANDONED + flag artifact_missing=true |
| 6 | worktree 清理失败 | log warning；启动继续；下次重启再尝试 |
| 7 | 重启后 orchestration 派一条新的同 motivation 的 dynamic action | 新 dyn_id 创建（如 dyn-3-2 或 dyn-N+1-1）；P1 PolicyGate 通过；与原 dyn 独立 |
| 8 | 多次重启 | 终态 ABANDONED 不再变化 |
| 9 | 已转 ABANDONED 的 dyn_id 后续被重派（同 motivation） | 视为新 dyn_id，与原 ABANDONED 不互相影响 |

---

## 11. DEFAULT / 待 review

| # | 条目 | DEFAULT | 备注 |
|---|---|---|---|
| 1 | hook 与 specialist 是否合并 | 否（平级新建） | §4.4 |
| 2 | 是否在 abandoned 记录中包含 worktree cleanup outcome | 是 | §6 |
| 3 | abandoned 是否影响 round-cap 计数 | 不影响（重启时 round 概念已重置） | 待 review |
| 4 | 重启后是否提示 orchestration "你有 N 条 abandoned dynamic action" | 否（仅通过 summary 自然展示） | 与 §1.7 不诱导原则一致 |
| 5 | abandoned summary 是否在下一 round 仍可见于 prompt | 是（与其他终态同处理） | 与 P6 §7 一致 |
| 6 | 重启时是否清理 dispatch_history 中的临时缓存 | 否（仅追加 abandoned 记录） | 历史完整性 |

---

## 12. 与设计哲学的对应关系

| 设计要点 | 在 P8 的落点 |
|---|---|
| §1.7 "多次失败时让 orchestration 自主决定回退，不引入显式 cooldown / kill switch" | abandoned 不是 cooldown / kill switch，而是 orchestration 视角下的"这条没结果"信号；后续派不派由 LLM 自主 |
| §1.8 "跨 session 学习暂不处理" | abandoned 的 artifact 不影响 KB / 不写入跨 session 缓存 |
| 设计稿 §3.9 "不做续跑" | §3.1 / §3.2 完整论证 |
| §1.2 红线 "micro-bench 不进 promote 链路" | abandoned 时 worktree 销毁 = scratch/bench/ 销毁 = 任何 bench 数字都消失 |

P8 是重启场景下"系统不悬空 + 红线不被打破"的兜底——任何重启路径都
保持系统对 §1.2 红线的承诺。
