# M5 — Specialist 框架 (MVP)

## 1. 设计目标

引入第二种 sub-agent 形态 — LLM specialist, 但在 M5 阶段**只跑单
domain 单实例** (单 framework_specialist), capacity = 1, 验证 prompt
契约 / 退出协议 / stale 检测 / Cortex T2 multi-variant 写入路径。

落地后用户应当看到: EXPLORE phase 内, Orchestration 派发 1 个
specialist (framework_specialist), specialist 跑完返回 specialist_done,
其中 proposal_set 进入 explore round, 经 Critic 评审, explore executor
跑变 KEEP/REVERT。

## 2. 范围

**包含**:

- 新 IntentType `specialist_done` + PolicyGate R3 校验。
- PolicyGate R2 (specialist 派发来源 = orchestration only) +
  `delegate{action='specialist'}` 路径。
- 新 sub-agent runner 类型 SpecialistRunner (走 LLM backend); 与 M3
  ExploreExecutor 共存。
- specialist prompt 装配 (9 段固定, 见 §3.5 §6)。
- specialist 工作目录 `runs/specialist/<task_id>/` + heartbeat /
  transcript / done 文件协议。
- robustness specialist stale 检测 (新 tick step)。
- 仅落地 1 个 domain: `framework_specialist`; 关心仓库 = M4
  domain_repos 中 framework_specialist 行。
- research_lane 引入但 capacity = 1 (M5 阶段串行 specialist), M6 提
  到 6。
- T2 hypothesize 升级: 每个 variant 一条 hypothetical 边 (M3 的
  per-propose_action 简化版替换为 per-variant)。
- T3 verify 升级: 每个 variant KEEP/REVERT 各自 verify。
- breakdown.specialist_runs 段 collector。

**不包含**:

- 6 类 specialist (留 M6)。
- 并发 dispatcher (留 M6)。
- plateau 阈值参数化 (留 M7)。

## 3. 与 M1/M2/M3/M4 的关系

- M1 提供 KB 写通路; 本里程碑用其 hypothesize/verify 接口, 升级到
  per-variant 粒度。
- M2 提供 phase 状态机; specialist 派发只在 EXPLORE phase 允许 (R1)。
- M3 提供 explore action; specialist 输出的 proposal_set 进入 explore
  的 grid。
- M4 提供 PR feed + MCP 工具集 spec; 本里程碑启用工具白名单。

## 4. 概念交付物

| 交付物 | 说明 |
|---|---|
| IntentType.SPECIALIST_DONE | 新增 (intent_parser) + PolicyGate R3 |
| PolicyGate R2 | specialist_dispatch_source + 子规则 |
| SpecialistRunner (概念) | 走 LLM backend, 多 turn, 收尾 specialist_done; 生命周期文件管理 |
| 工具白名单 (specialist) | Read/Grep/Glob, Bash 白名单 (rocm-smi/pgrep/cat 只读), WebSearch/WebFetch (EXPLORE only), mcp__pr_monitor__*, mcp__cortex_kb__traverse / find_recipe / query (只读) |
| prompt 装配 9 段 | identity / hardware / gap / kb_subgraph / recipe / pr_feed / source_hint / output_protocol / iron_rules |
| 工作目录文件 | prompt.md / transcript.jsonl / heartbeat.json / tool_calls.jsonl / specialist_done.json |
| robustness stale 扫描 | 新 tick step, kill 阈值 = max_turns × per_turn_max_min × 1.5 |
| Cortex T2 升级 | per-variant hypothesize, kb_edge_id 写到 PendingProposal.kb_edge_ids[variant_name] |
| Cortex T3 升级 | per-variant ingest_attempt + verify |
| research_lane 引入 | capacity=1 (M5); 与 benchmark/profile/server 不冲突 |
| breakdown.specialist_runs | 见 §3.12 §4.3, M5 阶段每轮只有 1 个 specialist |

## 5. specialist 派发到结果的完整链路 (M5 layer)

```
[Orchestration tick]
  ├ 决定要派 1 个 framework_specialist 探 gap=<canonical>
  ├ emit delegate{
  │     action='specialist',
  │     params={
  │        domain='framework_specialist',
  │        gap=<canonical_id>,
  │        kb_subgraph=<traverse 结果, 由 Coordinator 装配前调>,
  │        pr_feed=<KnowledgePlane.pr_feed_warm 调用结果>,
  │        source_roots=<framework_source_roots>,
  │        max_turns=8
  │     },
  │     idempotency_key='spec-<round>-framework'
  │   }
  └ PolicyGate R2 通过, TaskRegistry 入队

[Coordinator dispatcher]
  ├ research_lane 容量 1, 申到 lane
  ├ SpecialistRunner.run_task(task)
  │  ├ 装配 9 段 prompt → 写 prompt.md
  │  ├ 启动 LLM backend, 多 turn, 工具调用; 每 turn 写 heartbeat
  │  ├ 最终一轮 emit specialist_done intent (内含 proposal_set)
  │  ├ 解析 specialist_done → SubAgentResult.result.proposal_set
  │  └ 标 task succeeded
  └ delegated_result event 写入 bus

[Orchestration 下一 tick]
  ├ 看到 specialist_round_summary (= [framework_specialist round=N
  │   proposals=K kept=0 (待 explore 跑)])
  ├ 选 top-M variant (M5 阶段 M = K, 全部接收)
  ├ emit propose_action{
  │     action_name='explore',
  │     payload={grid: [variants...], base_extra_args: ..., ...},
  │     idempotency_key='explore-round-<N>'
  │   }
  └ PolicyGate 通过 → PendingProposal 落入字典

[Coordinator T2 hook (升级版)]
  ├ for each variant in grid:
  │    propose-point optimization_node (HYPOTHESIZED)
  │    canonical=opt.session-{sid}.proposal-{msg_id}.variant-{name}
  │    hypothesize hypothetical edge issue_node→opt_node
  │      attrs={domain: 'framework_specialist', variant: ..., round: ...}
  │    PendingProposal.kb_edge_ids[variant.name] = tentative_edge_id

[Critic Review]
  ├ 收到一组 K variant + judge_bundle (含 KB priors 各 variant)
  ├ 返回 verdict map {variant_name → verdict}
  └ Coordinator 处理 verdict → 把 approved 的 variant 通过 delegate
    → ExploreExecutor

[ExploreExecutor 跑]
  ├ 串行 run, KEEP/REVERT, KEEP 后 stack rebench
  └ 每 variant 调 T3:
      KEEP → ingest_attempt PASS + verify edge=kb_edge_ids[variant]
              outcome=confirmed promote=EXPERIENTIAL
      REVERT → ingest_attempt FAIL + verify outcome=refuted
```

## 6. specialist 失败模式

| 场景 | 处理 |
|---|---|
| LLM token overflow / API 错 | SpecialistRunner 自检, task → failed; 合成 empty specialist_done 进 inbox |
| heartbeat stale > 阈值 | robustness kill_task → cancelled; 合成 empty specialist_done |
| LLM 试调禁用工具 (Edit / git apply) | 工具调用拒, 计入 transcript; 多次拒 (≥3) → task → failed |
| specialist_done payload 校验失败 | PolicyGate R3 拒, intent 不入 inbox; 视同 empty done |
| max_turns 用尽未 emit specialist_done | 最后一轮强制收 + 合成 empty done |

合成 empty specialist_done 的目的: Orchestration round 不会因 specialist
失败而无限等待。

## 7. 实施步骤 (PR 拆分)

| PR | 内容 |
|---|---|
| 1 | IntentType.SPECIALIST_DONE + PolicyGate R3 校验 (来源/schema) |
| 2 | PolicyGate R2 (delegate dispatch source) + R5 工具白名单 |
| 3 | research_lane 引入 (capacity=1), LANE_CONFLICTS 表更新 |
| 4 | SpecialistRunner 概念落地, prompt 装配 9 段, 工作目录文件协议 |
| 5 | robustness stale 扫描 tick step + 合成 empty done 路径 |
| 6 | Cortex T2 hypothesize 升级到 per-variant + kb_edge_ids 字段 |
| 7 | Cortex T3 verify 升级到 per-variant |
| 8 | breakdown.specialist_runs collector |
| 9 | CLI flag (--research-lane-capacity 默认 1, 后续 M6 提到 6) + framework_specialist 默认开启 |

## 8. 验收清单

- [ ] EXPLORE 内可见 1 个 specialist 任务, transcript 完整, heartbeat
      更新, 收尾 specialist_done。
- [ ] specialist 不能调 Edit/Write/git apply (工具调用记录中可见拒绝)。
- [ ] specialist 提议的 variant 进入 explore round, KEEP/REVERT 路径
      完整。
- [ ] 每个 variant 在 Cortex 中有独立的 hypothetical 边, KEEP→confirmed
      / REVERT→refuted。
- [ ] kill 一个 specialist (人工或自动 stale) 后, 主 reactor 收到
      empty specialist_done, EXPLORE round 推进。
- [ ] PolicyGate 拒绝 robustness/critic 派 specialist (R2)。
- [ ] PolicyGate 拒绝伪造 from_agent 的 specialist_done (R3)。
- [ ] breakdown.specialist_runs 见 1 行 round, domain=framework_specialist。

## 9. 风险与回退

主风险:

- **prompt 装配偏差导致 specialist 跑空**: 9 段中某段质量太差 (例如
  KB subgraph 取错锚点). 缓解: 灰度时 transcripts 全保留, 由人工评审
  几轮 specialist 输出, 调整 prompt 模板。
- **LLM quota / 速率限制**: M5 capacity=1, 按理影响有限; 但对长 session
  串行 N 轮 specialist 仍累积调用. 缓解: max_turns 默认 8, 每 turn
  设置 token cap。
- **Cortex T2 multi-variant 写入失败导致 KB 视图破碎**: NDJSON 兜底 +
  T3 走 propose-edge `late_verified` 兜底。

回退:

- `--research-lane-capacity 0` 即时 degrade — specialist 不派, EXPLORE
  退化到 M3 的 LLM-direct grid。
- 整 M5 回退: PR1–9 全 revert, T2 回到 per-propose_action 简化版,
  ExploreExecutor 不变。

## 10. 哲学回引

本里程碑是**主轴 C (执行体一分为二)** 的真正起点, 也是
**Inv-5.1 / Inv-5.2 / Inv-5.3 (specialist 不出 patch / 经中转 / 单退
出协议)** 的具体落地。**Inv-3 (serving GPU 单租户)** 通过 research_lane
与 benchmark/profile/server 不冲突保证。
