# §3.14 风险矩阵 — 探测 / 缓解 / 回滚

## 1. 设计目标

把 v0.8 演进过程中**已识别**的风险集中登记, 给每条风险一份完整的
"探测 → 缓解 → 回滚" 三段式应对预案。本节是**操作册**, 不是设计文档,
但每一条都回引相关 §3.x 章节, 便于 RFC 评审者交叉验证。

成功标准:

- 每条风险有: 风险描述 / 触发场景 / 探测信号 / 缓解措施 / 回滚路径 /
  关联里程碑 / 关联章节。
- 每条风险都有**至少一条机器可读的探测信号** (breakdown 字段 / log
  marker / metric)。
- 任何在 §3.13 milestone 中提到的"主风险"在本节有完整登记。

## 2. 风险矩阵

下文按风险类别分组。每条用 `R-<两位数字>` 编号, 在后续 v0.8 里程碑
PR / runbook / monitoring 中可索引。

### 2.1 R-01 — Cortex 服务不可达 (短暂)

| 维度 | 说明 |
|---|---|
| 触发场景 | kb-service 重启 / 网络抖动 / 跨集群 ingress 故障 |
| 影响范围 | T0 / T2 / T3 / T4; 全部 phase |
| 探测信号 | breakdown.kb_provenance.pending_at_close > 0; `.kb_pending.ndjson` 行数持续增长; robustness alert `cortex_kb_unreachable` |
| 缓解 | NDJSON 兜底 (T2/T3 异步, 不阻塞); flusher 重试; 主流程不被 Cortex 拖慢 |
| 回滚 | 不需要 — 等 Cortex 恢复后 flusher 自动 drain。极端可 `--no-cortex` 重启该 session, 但会丢本 session 的 KB 写 |
| 关联章节 | §3.6 / §3.13 M1 |

### 2.2 R-02 — Cortex 服务不可达 (长期)

| 维度 | 说明 |
|---|---|
| 触发场景 | kb-service 长时间宕机 (>1h) / 跨集群路由配置错误未恢复 |
| 影响范围 | NDJSON pending 累积; T4 commit 失败 |
| 探测信号 | NDJSON pending > 1000 行 / 30 min; T4 commit timeout |
| 缓解 | robustness HIGH alert; 操作员介入决定 (a) 等待 (b) `--no-cortex` 后续 session 暂时旁路 (c) 手工 dump pending 留存; 设置 max pending 上限 (默认 10000 行), 超过即触发紧急 stop_reason=`cortex_overloaded` |
| 回滚 | M1 整段回退到 v0.6 (no Cortex) — `--no-cortex` 即可; M5+ 阶段 specialist 失去 KB 子图后退到 default_grid |
| 关联章节 | §3.6 / §3.13 M1 |

### 2.3 R-03 — PR Monitor 跨集群不可达

| 维度 | 说明 |
|---|---|
| 触发场景 | primus-cortex 集群 ingress 缺失 (default 数据面 pod 跨集群拉) |
| 影响范围 | specialist prompt 中 pr_feed 段为空; pr_intel_specialist 退化 |
| 探测信号 | breakdown.warnings 含 `pr_monitor_unreachable`; specialist_runs.notes 含 `pr_feed=(unavailable)` |
| 缓解 | prompt 模板允许 pr_feed 为空; specialist 仅靠 KB + 本地源码继续工作; pr_intel_specialist 直接 `specialist_done{empty=true, reason='pr_monitor_unreachable'}` |
| 回滚 | `--no-pr-monitor` 即时关 PR feed 注入, specialist 工具白名单不含 mcp__pr_monitor__* |
| 关联章节 | §3.6 / §3.13 M4 |

### 2.4 R-04 — Specialist 输出 patch 与本地框架版本不匹配

| 维度 | 说明 |
|---|---|
| 触发场景 | specialist 引用某个 PR 还没 merge / 与当前 sglang/aiter 源码冲突, 或 specialist 误抄文档中的旧 flag |
| 影响范围 | EXPLORE / KERNEL 中 server 启动失败 / accuracy gate 失败 |
| 探测信号 | last_action_failures 中 error_class=`server_startup_failed` 频率上升; explore_search.rejected.reason='failed' 比例 |
| 缓解 | explore executor 内 server 启动失败立即 REVERT, 标 `failed_startup` 进 ledger; T3 verify refuted 自动 negation 边, KB 后续过滤 |
| 回滚 | specialist 输出本身可逆 (REVERT 即恢复); 不可逆破坏从未发生 (Inv-5.1: specialist 不出 patch) |
| 关联章节 | §3.4 / §3.5 / §3.13 M5 |

### 2.5 R-05 — research_lane 并发过高耗光 LLM 配额

| 维度 | 说明 |
|---|---|
| 触发场景 | M6 capacity=6, 6 个 specialist 同时跑 max_turns=8, LLM API 短时间内打高 QPS |
| 影响范围 | 主 4-agent loop 也调 LLM, 可能被节流; Critic 评审时延上升 |
| 探测信号 | LLM backend error 速率上升 (rate_limited / quota_exceeded); breakdown.specialist_runs.notes 含 `llm_throttled` |
| 缓解 | 全局 LLM 速率限流器 (在 ClaudeBackend / CodexBackend 层加 token bucket); per-specialist max_turns 缩短; --research-lane-capacity 灰度调小 (默认 6 → 3) |
| 回滚 | `--research-lane-capacity 1` 立即退到 M5 串行 specialist; `--research-lane-capacity 0` 完全 degrade 到 M3 |
| 关联章节 | §3.7 / §3.13 M6 |

### 2.6 R-06 — 砍掉 scoreboard 后 LLM 决策不稳定

| 维度 | 说明 |
|---|---|
| 触发场景 | M2 落地后, prompt 失去 scoreboard 提示, LLM 反复尝试已被 REVERT 的 variant 类别 / 错过 high-severity gap |
| 影响范围 | EXPLORE 阶段 cumulative_gain 增长缓慢, plateau 提前触发 |
| 探测信号 | explore_search.tested.outcome 中 SKIPPED (fingerprint dedup) 比例 > 30%; specialist_rounds.proposals_kept 持续 < 1 / round |
| 缓解 | (a) Cortex KB negation 边过滤已失败 variant (KB 子图天然不再返回这些); (b) explore_search.fingerprint dedup 已防止真正重复; (c) prompt 中 last_action_failures + winners_history 提供事实层 prior; (d) 灰度时跑 v0.6 vs v0.8 同 model 双跑对比 |
| 回滚 | 极端情况下可恢复 prompt 中的"phase 内 action 优先级"提示 (但不恢复评分代码) — 本质是让 LLM 看到一个**人工排序的**枚举, 而非数值 |
| 关联章节 | §3.9 / §3.13 M2 |

### 2.7 R-07 — Critic 评审吞吐成为瓶颈

| 维度 | 说明 |
|---|---|
| 触发场景 | M5/M6 后, specialist 大量提案, Critic 单次评一组 K 个 variant, 节流后导致 EXPLORE round 时长拉长 |
| 影响范围 | EXPLORE phase 用时 > budget, 提前触发 explore_phase_budget_exhausted |
| 探测信号 | breakdown.critic_robustness.review_latency_avg > 阈值; phase_timeline EXPLORE 段长度异常 |
| 缓解 | (a) 评审改批量 (一次 K 个 variant), 已经在 §3.3 §4.3 设计; (b) 限制 propose_action 携带 variant 数 (M6 默认 explore_round_batch_size=5); (c) Critic 不可达时 fallback `--critic-mock` (但失去 KB 评审, 仅做兜底) |
| 回滚 | Critic 性能问题独立于 v0.8 架构; 由 critic-agent 自身 RFC 解决。短期 fallback `--critic-mock` |
| 关联章节 | §3.3 / §3.5 / §3.13 M5 / M6 |

### 2.8 R-08 — 多 specialist 同时读 framework 源码触发 I/O 抖动

| 维度 | 说明 |
|---|---|
| 触发场景 | M6 capacity=6, 6 个 specialist 同时调 Read/Grep 大文件, wekafs 单挂载点压力 |
| 影响范围 | specialist 工具调用延迟; 不影响 server bench |
| 探测信号 | specialist transcript 中工具调用 latency 上升; wekafs 读 IOPS 监控 |
| 缓解 | source-mirrors 已是 wekafs 高并发友好; 必要时 specialist 工具集加 LRU cache (在 LLM backend 层做); per-specialist 工具调用速率限制 |
| 回滚 | --research-lane-capacity 调小; 不需要架构回滚 |
| 关联章节 | §3.6 / §3.7 |

### 2.9 R-09 — phase 退出条件 proxy 与真定义口径不一致

| 维度 | 说明 |
|---|---|
| 触发场景 | M2 上线但 M7 还没落地, 期间 EXPLORE 退出用 `params_no_promote_streak >= 5` proxy, 与 §3.8 真 plateau_explore 口径不同 |
| 影响范围 | 灰度期 EXPLORE 退出时机 / phase_history 记录的 reason 与 GA 后不一致 |
| 探测信号 | breakdown.warnings 含 `plateau_proxy_provisional` 条目; phase_history.reason 标 `plateau_explore_proxy` (非 v0.8 真值) |
| 缓解 | M2 文档明确该 proxy 是临时方案, M7 替换; 灰度文档说明 |
| 回滚 | 在 M7 PR 出问题时, 退到 M2 proxy |
| 关联章节 | §3.8 / §3.13 M2 / M7 |

### 2.10 R-10 — schema migration 失败导致 resume 拒绝

| 维度 | 说明 |
|---|---|
| 触发场景 | v0.6 → v0.8 迁移函数对某条字段处理报错 (例如 backends_search 中含未知 outcome 值) |
| 影响范围 | resume 失败, 进程退出 1 |
| 探测信号 | logs/cli.log 含 `migration error: ...`; state.json 不被覆写 |
| 缓解 | (a) `--migration-mode=lenient` 接受部分丢失 (类别 2 事实层除外); (b) `--reset-state` 完全从头跑 (但保留 Cortex KB) |
| 回滚 | 撤回 M2 / M3 / M5 中相关 PR; v0.6 reader 仍能读老 state.json |
| 关联章节 | §3.10 |

### 2.11 R-11 — phase 切换期间 in-flight 任务被孤立

| 维度 | 说明 |
|---|---|
| 触发场景 | EXPLORE 退出转 KERNEL 时, 仍有 specialist 在 research_lane 跑; 或 KEKERNEL 转 SWEEP 时仍有 kernel_opt 任务在跑 |
| 影响范围 | 任务结果错位 (上 phase 的 specialist done 进新 phase inbox 浪费 turn) |
| 探测信号 | phase_history 中 phase_started_ts 与 specialist completed_at 重叠; breakdown.specialist_runs 中 round_id 与 phase 不对齐 |
| 缓解 | phase 退出时 Coordinator 主动 kill_task 所有 in-flight 任务, 合成 empty done; phase_history.evidence 标 `inflight_kills=N` |
| 回滚 | 不需要架构回滚; 调试时把 phase 退出条件做更保守 (例如等所有 in-flight specialist 完成再触发 plateau_explore) |
| 关联章节 | §3.2 / §3.5 |

### 2.12 R-12 — Specialist 在 prompt 中泄漏认证 token

| 维度 | 说明 |
|---|---|
| 触发场景 | LLM 调 Bash 工具执行 `env` / `cat ~/.codex/auth.json` 等; transcript 写入 workspace |
| 影响范围 | transcript 落 wekafs, 含 SAFE_API_KEY / OPENAI_API_KEY 字面量 |
| 探测信号 | transcript 文件中 grep `(sk-|Bearer )` 命中 |
| 缓解 | tool 调用日志在写 workspace 前 redact (token 字段替换为 `[REDACTED]`); Bash 工具白名单显式排除 `env` / `cat ~/.codex/*` / `cat ~/.claude/*` |
| 回滚 | 工具白名单收紧不影响功能, 不需要架构回滚 |
| 关联章节 | §3.5 / §3.11 |

### 2.13 R-13 — Cortex `optimization_node` meta 注册尚未到位

| 维度 | 说明 |
|---|---|
| 触发场景 | `cortex-for-hyperloom-2026-05-18.md` §8 列出 — Cortex 一侧 schema 暂未为 optimization_node 注册 meta, attrs 走 jsonb |
| 影响范围 | 写入正常, 但跨 session 检索 / 派生 summary 可能少一些索引化能力 |
| 探测信号 | Cortex 一侧 RFC 跟踪 |
| 缓解 | 短期 attrs jsonb 即可工作 (canonical_id 已唯一); 长期 Cortex schema PR 推进 (haiskong@ 维护) |
| 回滚 | 不需要 |
| 关联章节 | §3.6 |

### 2.14 R-14 — explore stack rebench 把已 KEEP 的 variant 弹出导致 cumulative_gain 抖动

| 维度 | 说明 |
|---|---|
| 触发场景 | M3 引入 stack rebench, 但 stack rebench 受 noise 影响, 偶发性把刚 KEEP 的 variant 误判 keep_unstable_in_stack |
| 影响范围 | breakdown 中 cumulative_gain 时序看起来抖动; 操作员困惑 |
| 探测信号 | breakdown.attribution.phase_breakdown.explore.by_round 中频繁 keep_unstable_in_stack 行 |
| 缓解 | stack rebench 阈值默认保守 (0.5%); 必要时启用"双 sample 平均"再判定 (M3 内 PR5 实施时考虑) |
| 回滚 | stack rebench 是 explore executor 内的可选步骤, 可加 `--no-stack-rebench` flag 暂关; 改 keep_unstable_in_stack 阈值 |
| 关联章节 | §3.4 / §3.13 M3 |

### 2.15 R-15 — specialist domain 选择不均衡 (LLM 永远派 framework)

| 维度 | 说明 |
|---|---|
| 触发场景 | M6 后, Orchestration prompt 中 DOMAIN SELECTION GUIDE 不够强, LLM 习惯只派 framework_specialist, kernel/comm/compiler 鲜有派遣 |
| 影响范围 | EXPLORE 多样性下降; 跨 domain 协同探索能力 underutilized |
| 探测信号 | breakdown.specialist_runs.domain_breakdown 中 framework 占比 > 80%; 其它 domain 派出次数 ≤ 1 |
| 缓解 | (a) 强化 prompt 中的 domain selection 描述; (b) Coordinator 在 EXPLORE 内强制每 K 轮至少派 pr_intel_specialist 一次 (类似 M6 §5 描述); (c) 灰度时让 robustness 用 escalate_strategy_change `pause_specialist_<domain>` 强制轮换 |
| 回滚 | 不需要; prompt 调整即可 |
| 关联章节 | §3.5 / §3.13 M6 |

## 3. 风险与里程碑交叉表

```
                    M1   M2   M3   M4   M5   M6   M7
R-01 cortex 短不可达  ●                              
R-02 cortex 长不可达  ●                              
R-03 PR Monitor 不可达          ●                    
R-04 specialist patch 不匹配                  ●     
R-05 LLM 配额                              ●         
R-06 砍 scoreboard         ●                          
R-07 Critic 瓶颈                          ●  ●     
R-08 wekafs I/O 抖动                              ●  
R-09 plateau proxy        ●                       ●
R-10 schema migration     ●  ●                    
R-11 phase 切换孤立任务   ●                      
R-12 token 泄漏                            ●         
R-13 Cortex meta 未注册   ●                            
R-14 stack rebench 抖动         ●                    
R-15 domain 不均衡                              ●  
```

## 4. 监控建议

每个里程碑落地后, **必须**接入下列监控信号:

- breakdown.warnings 段长度 (上升即检查)
- breakdown.kb_provenance.pending_at_close (理论 = 0)
- breakdown.specialist_runs.parallelism / proposals_total / kept
- breakdown.attribution.phase_breakdown 完整性
- breakdown.telemetry.lane_timeline.research_lane.peak_holders
- robustness alert 总数 / 等级分布
- LLM backend rate-limit 计数

任一信号超过阈值, 触发对应风险条目的"缓解"流程。

## 5. 哲学回引

风险登记本身不是设计, 但每条风险的应对方案都尊重 §3.1 的三主轴 +
三不变量。例如 R-04 的核心缓解依赖 Inv-5.1 (specialist 不出 patch) —
没有 patch 自然就没"不匹配"破坏面。R-06 的应对依赖 Inv-9.1 (决策层
无评分) — 砍 scoreboard 是有意的, 不能"等不稳定就加回评分"。
