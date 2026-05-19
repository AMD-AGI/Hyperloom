# §3.13 v0.6 → v0.8 里程碑 — M1 ~ M7

## 1. 设计目标

把 §3.1–§3.12 的所有概念变更切成 **7 个独立可发布的里程碑**, 每个里程
碑都是 *additive* (不破坏前一个里程碑的可用性), 可以单独评审 / 单独回
退。这是 v0.6 平稳过渡到 v0.8 的执行时间线。

每个里程碑的设计文件位于本目录下:

| 里程碑 | 主题 | 文件 |
|---|---|---|
| M1 | Cortex 接入 (纯写) | `M1_cortex_writes.md` |
| M2 | phase 状态机 + 砍 scoreboard | `M2_phase_and_scoreboard.md` |
| M3 | explore 合并 (backends + params) | `M3_explore_merge.md` |
| M4 | PR Monitor 接入 | `M4_pr_monitor.md` |
| M5 | Specialist 框架 (MVP) | `M5_specialist_mvp.md` |
| M6 | Specialist 多 domain + 并发 | `M6_specialist_concurrent.md` |
| M7 | SWEEP / KERNEL 收敛 + plateau 调参 | `M7_plateau_tuning.md` |

## 2. 里程碑序列原则

### 2.1 additive 原则

每个里程碑必须满足:

- **运行可性**: 该里程碑落地后, 一个 *fresh session* 仍能跑通完整
  baseline → explore → kernel → sweep → close (即使 specialist 没接
  入, 用 default_grid 兜底)。
- **resume 可性**: 落地后能 resume 任何更早里程碑产生的 session, 不
  报错 (字段缺失走默认 / 推断)。
- **回退可性**: 仅回退本里程碑的 PR 即可恢复上一里程碑的 session 行
  为, 不依赖 SharedState / Cortex 数据回滚。

### 2.2 依赖关系

```
M1 (Cortex 写入)               独立, 不依赖其它
M2 (phase + 砍 scoreboard)     独立 (不依赖 M1, 但实施先后建议 M1→M2)
M3 (explore 合并)              依赖 M2 (phase 状态机)
M4 (PR Monitor)                依赖 M1 (KB 数据模型)
M5 (Specialist MVP)            依赖 M2 + M4 (phase + KB + PR)
M6 (Specialist 并发)           依赖 M5 + M3 (specialist 实例 + explore 入口)
M7 (plateau 调参)              依赖 M2 + M3 + M5 + M6 (信号面齐全后再调参)
```

时间线建议: M1 / M2 并行 (两个独立 PR 链), M3 接 M2 后, M4 接 M1 后,
M5 等 M2+M4, M6 等 M5+M3, M7 收尾。

### 2.3 可灰度

每个里程碑都引入对应的 CLI flag, 默认行为对老 session **保持兼容**:

| 里程碑 | CLI flag | 默认 | 灰度方式 |
|---|---|---|---|
| M1 | `--no-cortex` | off (即 Cortex 启用) | `--no-cortex` 完全旁路 KB |
| M2 | `--legacy-action-scores` | drop | `warn` 模式仅 log, 不 drop |
| M3 | `--legacy-explore-split` | off | 老路径暂停可用, 留作回滚 |
| M4 | `--no-pr-monitor` | off (启用) | 关掉 PR feed |
| M5 | `--research-lane-capacity` | 1 (M5 阶段, M6 调到 6) | 0 = 关 specialist |
| M6 | 同 `--research-lane-capacity` | 6 | 调小 = 限并发 |
| M7 | `--plateau-explore-keep-gain`, `--plateau-kernel-revert-streak`, ... | 默认见 §3.8 | 灰度调阈值 |

### 2.4 验收串

每个里程碑落地都要跑一次"完整冷启动 + resume" 双场景烟测, 通过后才
进入下一里程碑。烟测是各 milestone MD 内 §"验收清单"的强制 1 条。

## 3. 里程碑总览表

| M | 主交付 | 主要触及文件域 | 验收信号 |
|---|---|---|---|
| M1 | Coordinator T0/T2/T3/T4 hook + NDJSON 兜底 + Cortex audit | `orchestrator/` + `runtime/cortex/` 子目录 + breakdown.kb_provenance | KB 中能看到本 session 的 hypothesize→verify→commit 链 |
| M2 | phase 字段 + phase_history + 退出条件函数 + 删除评分代码 | `orchestrator/coordinator.py`, `shared_state.py`, `policy.py`, `prompt_builder.py`, *删除* `scoring.py` | breakdown.phase_timeline 显示 5 段; state.json 无 action_scores |
| M3 | explore action + ledger + executor + validate_stack 内嵌 | `actions/_meta/explore.yaml`, 删除 backends/params yaml, 新 ledger | breakdown.capability_summary 只见 explore 行; resume v0.6 ledger 无损迁移 |
| M4 | KnowledgePlane 抽象 + PR Monitor REST 预热 + MCP 注入 | `orchestrator/knowledge_plane.py` (新), prompt_builder pr_feed 段 | specialist prompt 中 PR feed 段非空; pr_node 入 KB |
| M5 | specialist sub-agent runner + specialist_done intent + 1 类 specialist | `orchestrator/sub_agent_runner.py`, `intent_parser.py`, prompt_builder specialist 模板 | EXPLORE 内可见 1 个 specialist 跑完 + 提议进 explore |
| M6 | research_lane (capacity > 1) + 6 类 specialist + 并发 dispatcher | `orchestrator/resource_lock.py` schema, dispatcher 改 asyncio.gather | EXPLORE 内并发 ≥ 2 个 specialist; benchmark_lane 仍 ≤ 1 |
| M7 | plateau 阈值参数化 + escalate hint 词表 + 软退路径 | `coordinator.py`, `cli.py` flag, `phase_state_machine` 实现 | 阈值调小后 EXPLORE 提早收敛, breakdown.attribution.phase_breakdown 完整 |

## 4. 一份"操作员视角"的部署节奏建议

```
Week 0  M1 PR 链 + M2 PR 链  并行启动
Week 1  M1 落地, 跑 v0.6 流程 + 写 KB; M2 仍在 PR 评审
Week 2  M2 落地, 删除 scoreboard, 老 session resume drop_legacy
Week 3  M3 落地, EXPLORE 合并; M4 同时落地, PR feed 上线
Week 4  M5 落地 (capacity=1, 单 specialist); 灰度观察 LLM quota / Cortex 容量
Week 5  M6 提 capacity=6, 灰度 1 周; 观察 EXPLORE plateau 行为
Week 6  M7 调参 + 上线默认值; v0.8 GA
```

(纯估时, 真实节奏看每 PR 评审 + dogfood 反馈。)

## 5. 风险联动 (与 §3.14 关联)

每个里程碑各自带的主要风险, 在 §3.14 中重复登记 + 提探测 / 缓解 /
回滚路径; 本节的里程碑 MD 在自己 §"风险与回退" 中也会复述关键 1–3
条便于评审者上下文对齐。

## 6. 哲学回引

里程碑切分本身是 **主轴 A / B / C** 在执行层的风险隔离: 不要一次推
所有概念, 防止"v0.8 上线但某段不可用即整体回退"。每个 milestone 守
住 §3.1 三主轴 + 三不变量 + 各章节内部不变量, 不存在"我这一步先违反
不变量, 下一步再补回"的取巧路径。
