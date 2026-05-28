# Dynamic Action — 实施路径文档集

> 本目录是 `dynamic_action.MD` §3 实施路径的**展开**：把每个阶段的方向
> 性设计作为独立文档维护，便于在实施期一段一段稳推、独立 review、独立
> 回滚。
>
> 所有文档**只做方向上的设计**，不下沉到具体代码 diff、函数签名、JSON
> schema 字面量。详细接口字段（dynamic_action.MD §2.1–2.10）随各阶段
> 实施前再展开。

---

## 1. 阅读顺序

P0 → P1 → P2 → P3 → P4 → P5 是关键路径，按此顺序读、按此顺序实施。
P6 / P7 / P8 在 P5 完成后可并行；P9 测试用例随各阶段同步编写。

| 文档 | 阶段 | 主题 |
|---|---|---|
| `P0_decisions.md` | P0 | 锁定的 7 条设计决策（4 条 v1 lock-down + Q1/Q2/Q3） |
| `P1_dispatch_skeleton.md` | P1 | 合法派发 — Action 注册 + PolicyGate 骨架 |
| `P2_session_artifact_seed_kit.md` | P2 | Session 目录骨架 + Seed kit 装配 |
| `P3_subagent_runner.md` | P3 | Multi-turn ReAct sub-agent runner |
| `P4_critic_cross_domain.md` | P4 | Critic `patch_landing` + 跨域审查规则 |
| `P5_e2e_wiring.md` | P5 | 端到端 happy path 连线 |
| `P6_sharedstate_summary.md` | P6 | SharedState 聚合视图与状态机 |
| `P7_orchestration_prompt.md` | P7 | Orchestration prompt 入口声明 |
| `P8_resume_semantics.md` | P8 | Resume 语义（重启时 abandoned） |
| `P9_test_matrix.md` | P9 | 测试矩阵（单元 / 集成 / 不变量） |

每个 PX 文档的固定结构：

1. **目标** — 本阶段对外可见的产物边界
2. **触及的架构平面** — 改动覆盖哪些子系统
3. **中心思想** — 为什么这么设计
4. **关键设计点（方向级）** — 各子系统的设计要点（不写代码）
5. **依赖与前置条件** — 必须哪些阶段先完成
6. **验收信号** — 本阶段"完成"的可观测判据
7. **DEFAULT 项 / 待 review** — 我假设的默认值，等待用户确认

---

## 2. 已锁定的核心决策（来自 P0）

### v1 设计 lock-down（dynamic_action.MD §1.9）

| | 决策 | 选择 |
|---|---|---|
| D-A | Dispatch 通道 | 复用 `DELEGATE` + 新 `action_name="dynamic_action"`；不开新 IntentType |
| D-B | Critic 分类 | 复用 `patch_landing`，加 `cross_domain=true` flag |
| D-C | SharedState 聚合视图保护 | `dynamic_actions` 进受保护字段集，仅 Coordinator 写 |
| D-D | Sub-agent 形态 | 真正的 multi-turn ReAct，不走 single-shot 过渡 |

### 实施期补充决策（来自 Q1/Q2/Q3）

| | 决策 | 选择 |
|---|---|---|
| Q1 | Multi-turn 物理形态 | runner 多次起 `claude` sub-process，state 由 runner 在外部 journal 维护，sub-agent 是无状态的多次调用 |
| Q2 | micro-bench 边界 | 允许 ≤60s 简化 inference 路径；sub-agent 整体 wall-clock budget ≤ 15 分钟 |
| Q3 | proposal_set 上限 | = 1（一个 dynamic action 只产 1 个跨域组合 patch） |

---

## 3. DEFAULT 总清单（待 review）

下列条目是我在文档中预设的默认值，**未与用户 lock-down**。建议在 P1
代码动手前批量 review 这一节。每项注明所在文档章节。

| # | 条目 | DEFAULT | 出处 | 备注 |
|---|---|---|---|---|
| 1 | `dyn_id` 命名规则 | `dyn-<round>-<seq>`（例 `dyn-3-1`） | P2 §3 | 可读性优先；与 specialist `task_id` 风格保持差异以便目视区分 |
| 2 | artifact 根路径 | `$SESSION_DIR/agents/orchestration/dynamic_actions/<dyn_id>/` | P2 §2 | 设计稿 §1.5 已规定，沿用 |
| 3 | 资源 lane | 与 specialist 共享 `research_lane`；`MAX_RESEARCH_LANE_CAPACITY=6` 不变 | P1 §4 | 设计稿 §1.4 已规定 |
| 4 | round-cap | `MAX_DYNAMIC_PER_ROUND = 1` | P1 §4 | 设计稿 §1.4 已规定 |
| 5 | grid sourced cap | `MAX_DYNAMIC_SOURCED_VARIANTS = 1` | P1 §4 | 由 Q3 自然导出（proposal_set cap=1）|
| 6 | `scope_domains` 最小数量 | 硬约束 ≥ 2 | P1 §3 | "跨域"的字面要求 |
| 7 | seed kit 总 token 预算 | ≤ 8K tokens | P2 §4 | 防 prompt 体积膨胀 |
| 8 | seed kit 各类条数硬上限 | KB pitfalls ≤ 10、kept_patches ≤ 20、reverted_patches ≤ 10、profile slices ≤ 6 | P2 §4 | 经验值，待实际跑出 token 占比后调优 |
| 9 | sub-agent turn cap | ≤ 12 轮 LLM call | P3 §3 | Q2 budget ≤15 分钟下的合理上限 |
| 10 | sub-agent 整体 wall-clock | ≤ 15 分钟（Q2 决策） | P3 §3 | runner 用 wall-clock timer 强制终止 |
| 11 | 单次 bench wall-clock | ≤ 60s subprocess timeout（Q2 决策） | P3 §4 | runner 侧硬限 |
| 12 | 单轮 LLM token cap | input ≤ 32K, output ≤ 4K | P3 §3 | 与 specialist runner 持平起步 |
| 13 | 工具白名单（名词级） | `read_source` / `read_session_artifact` / `run_bench` / `emit_proposal` | P3 §4 | 4 类资源，prompt 不暴露 tool name |
| 14 | 回收路径白名单 | 仅 `proposal_set.json` + `sub_agent_journal.md` | P3 §5 | §1.2 红线的机械保障 |
| 15 | 状态机状态数 | 11 个状态（DISPATCHED → 多个分支终态），见 P6 §3 | P6 §3 | 待 review |
| 16 | prompt 注入截断 | 最近 5 条 dynamic_actions summary | P6 §5 | token 预算 |
| 17 | 重启时未完成的 dynamic action | 全部标 `ABANDONED`，artifact 保留 | P8 §3 | 设计稿 §3.9 已规定 |
| 18 | grid 派单顺序 | dynamic variant 与 specialist variant 同 round 内 dynamic 优先（因 cap=1，影响小） | P5 §5 | 待 review |
| 19 | 空 proposal_set 处理 | 与 specialist `empty=true` 同处理（status → COMPLETED_EMPTY） | P5 §6 | 沿用现有 specialist 路径 |

---

## 4. 文档维护守则

- 每个 PX 文档作为该阶段的**单一事实源**；与 `dynamic_action.MD` §3 的
  对应小节保持双向引用。
- 实施期发现某 PX 文档与代码现实不符 → 优先修文档（除非是显式设计变
  更，需走 dynamic_action.MD §3.11 的设计变更流程）。
- DEFAULT 一旦被确认 → 把对应条目从本 README §3 表格移除，并在原文档
  中标记为"已确认"。
- 新增的待办设计问题统一汇集到本 README §3，避免散落在各 PX 文档底部。
