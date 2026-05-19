# M1 — Cortex KB 接入 (纯写)

## 1. 设计目标

为 v0.8 引入 Cortex KB 的**写**通路 (T0/T2/T3/T4 锚点) + NDJSON 兜底,
同时让现有 v0.6 行为完全不变。这是 v0.8 的 *基础层*: 之后所有 KB 相
关功能 (M4 PR 接入, M5 specialist 装配读 KB) 都依赖本里程碑落地的
KnowledgePlane 写入面 + audit 设施。

落地后用户应当看到: 一个**没有 specialist、没有 phase 状态机、没有
explore 合并**的 v0.6 风格 session, 在 EXPLORE / KERNEL 阶段每一次
KEEP/REVERT 都被 hypothesize / verify 镜像到 Cortex KB; 整段 session
有一个 cortex_session_id, T4 commit 后 KB 中能反查全部 attempts。

## 2. 范围

**包含**:

- KnowledgePlane facade 的*写*侧最小可用形态 (T0 begin /
  propose_point / hypothesize / ingest_attempt / verify / commit / abort)。
- T0 锚点: PRELUDE 入口写 cortex_session_id / mint workload_node。
- T2 锚点: `_handle_propose_action` 在 PolicyGate 通过 + PendingProposal
  落地的同一时机, 对每个 propose_action 创建 hypothetical 边 (单
  variant 简化版, 因为 EXPLORE 还没合并)。
- T3 锚点: `_promote_to_shared_state` (KEEP) / `_handle_unpromotable_result`
  (REVERT) 各自触发 ingest_attempt + verify。
- T4 锚点: Coordinator.stop() 末尾 + report 之前 commit。
- NDJSON 兜底: `<session_dir>/runtime/cortex/.kb_pending.ndjson` 写
  + flusher 子进程 (由 robustness 启 / 检活)。
- breakdown.kb_provenance 段 + warnings 段标记本里程碑产生的迁移点。

**不包含**:

- KB 读 (T1 traverse 等到 M4/M5 用)。
- PR Monitor。
- Specialist 框架。
- phase 状态机 (本里程碑仍按 v0.6 流程; T0 仅是"在 baseline 之前"
  调一次 begin, 不引入 phase 概念)。
- explore 合并。

## 3. 与现有代码的关系

- 不修改 `scoring.py` / `MARATHON_PRIORS` (留给 M2)。
- 不修改 `actions/_meta/` (留给 M3)。
- 不修改 `policy.py` 的角色矩阵; 仅扩 PolicyGate 的 CORE_STATE_FIELDS
  增加 `cortex_session_id` / `pending_kb_edges` (这两个字段属 Coordinator
  写, 防止 LLM update_state)。
- critic-agent 的 KB 通路保持现状; 后端共享同一 Cortex (URL 一致),
  写入的 `kb_writes` 与 Coordinator 的 `verify` 落同一图。

## 4. 概念交付物

| 交付物 | 描述 |
|---|---|
| KnowledgePlane.write facade | 单点封装 cortex-kb CLI 子进程 / HTTP 直连 / NDJSON enqueue 失败兜底 (一个 facade 类的概念, 不强制写代码) |
| 新 SharedState 字段 | `cortex_session_id`, `pending_kb_edges[]`, `cortex_session_summary` (留 T4 填) |
| 新 manifest 字段 | `stack_fingerprint` (rocm/aiter/sglang/vllm), 用于 T0 begin 的 attrs |
| 新文件路径 | `<session_dir>/runtime/cortex/{,.kb_sid,.kb_warm.json,.kb_pending.ndjson,.kb_flushed.ndjson,.kb_dead_letter.ndjson,.kb_audit.jsonl,.kb_flusher.pid}` |
| flusher 进程 | 由 robustness 启动 / 检活的常驻 daemon, 读 pending 队列 batch POST |
| breakdown.kb_provenance 段 | 见 §3.12 §4.4 |
| CLI flags | `--cortex-kb-url`, `--no-cortex`, `--cortex-strict-fingerprint` |

## 5. T0–T4 落点 (本里程碑层面)

注意: M1 还没 phase 状态机; T0 / T4 都用现有 Coordinator 入口/退出 hook。

### 5.1 T0 (一次性, 入口)

时机: `cli._run_optimize` 在 `make_session_dir()` + `manifest.write_manifest()`
之后, 在 Coordinator 实例化之前。

调用:

- `cortex-kb session begin --goal find_recommendation --task <text>
  --thinking-style recommendation --attrs {...}` 拿 sid。
- `propose-point workload_node` (canonical: `workload.<model>.<hw>`)。
- `find-recipe / traps` 暂时 *只调不存* (留 M5 prompt 装配时再读); 输出
  落到 `.kb_warm.json` 作 audit。
- 写 `SharedState.cortex_session_id = sid`。
- 写 `.kb_sid` 文件供 resume 用。

失败:

- HTTP 不通 → 默认 fail-fast (PRELUDE 失败, stop_reason=cortex_t0_failed)。
- 显式 `--no-cortex` 启动 → 跳过 T0 全部步骤, cortex_session_id = ''。

### 5.2 T2 (每次 propose_action 通过 PolicyGate)

时机: `Coordinator._handle_propose_action` 在 PendingProposal 入字典
之后。

调用 (per proposal):

- `propose-point optimization_node (HYPOTHESIZED)` canonical
  `opt.session-{sid}.proposal-{msg_id}` (由 propose_action 的
  msg_id 派生, 自然幂等)。
- `session hypothesize from gap_anchor to opt_canonical type=hypothetical`
  attrs={action: propose_action.action_name, predicted_gain_pct: ..., msg_id: ...}。
- 拿 tentative_edge_id 写到 `PendingProposal.kb_edge_id`。

注: M1 阶段还没 specialist, 所以一个 propose_action 只对应一个
opt_node + 一条 hypothetical 边 (M3 之后改成多 variant)。

`gap_anchor` canonical 的派生: 在 M1 阶段简化为
`workload_node.canonical_id` (即"为这个 workload 开的优化"); M5
specialist 接入后才有真正的 issue_node。

失败:

- 写 NDJSON pending; tentative_edge_id 留空; 后续 T3 直接走
  `propose-edge + 标 late_verified`。

### 5.3 T3 (KEEP / REVERT 时机)

时机: `_promote_to_shared_state` (succeeded + promotable) /
`_handle_unpromotable_result` (failed/non-promotable) 内, 紧跟现有
audit 写入之后。

调用:

- KEEP: `ingest-attempt outcome=PASS metrics={...}` +
  `verify edge=<tentative_edge_id> outcome=confirmed
  promote=EXPERIENTIAL`。
- REVERT: `ingest-attempt outcome=FAIL metrics={...}` +
  `verify edge=<tentative_edge_id> outcome=refuted` (自动 negation)。
- attempt canonical: `attempt.session-{sid}.task-{task_id}` 自然幂等。

失败:

- 写 NDJSON pending, 不阻塞主流程。
- tentative_edge_id 缺失时 (T2 失败的回退) 走 propose-edge 而非
  verify, 标 `late_verified`。

### 5.4 T4 (一次性, 退出)

时机: `Coordinator.stop()` 内, 任何 report 类输出之前; 也可由 `report`
action 在执行中显式调一次 (重复调用幂等, 不会双 commit)。

调用:

- 等 NDJSON drain 完成 (超时 60s)。
- `cortex-kb session commit --sid <sid>` 拿 promoted_edges /
  derived_summary_id; 写 `SharedState.cortex_session_summary`。
- 失败 → stop_reason=`cortex_drain_failed`, 进程退出非零, session_dir
  保留, resume 时再 drain + commit。

## 6. NDJSON 兜底协议

- 所有 KB 写入操作先写 `.kb_pending.ndjson` (append-only), 再异步
  POST。
- flusher 进程: 5s 或 50 行触发 batch; 每条带 `idempotency_key` 由
  Coordinator 派生 (canonical_id 已天然幂等, 这里 idempotency_key 主
  要给 HTTP 层重传去重)。
- 推送成功 → 行写 `.kb_flushed.ndjson`。
- 推送失败 (4xx 业务错) → 写 `.kb_dead_letter.ndjson` + alert (
  robustness HIGH severity)。
- 推送失败 (5xx 网络) → 留在 pending, 下次重试。
- T4 commit 之前必须 drain 干净。

## 7. resume 行为

resume 一个 v0.8 session:

- 读 `.kb_sid` 拿到 sid;
- 不重新 begin; 直接复用 sid。
- pending 文件如果非空, flusher 启动后第一时间 drain。
- 老 v0.6 session resume 时:
  - `.kb_sid` 不存在 → 视为新接入, 调一次 T0 begin (但不重新 mint
    workload_node, 已存在则去重)。
  - 老 SharedState 不含 cortex_session_id → 第一次 resume 后写入。

## 8. 实施步骤 (PR 拆分)

| PR | 内容 |
|---|---|
| 1 | 新增 `runtime/cortex/` 子目录创建; `.kb_pending.ndjson` 写入抽象 (无 flush, 仅 append); SharedState 加新字段; manifest 加 stack_fingerprint |
| 2 | KnowledgePlane facade write 侧 (调用 cortex-kb CLI 子进程, 失败转 NDJSON); T0 hook |
| 3 | T2 hook (在 propose_action 路径) + tentative_edge_id 持久化 |
| 4 | T3 hook (在 promote / unpromotable 路径) + verify 调用 |
| 5 | T4 hook (在 stop() 末尾) + drain 等待 + commit |
| 6 | flusher daemon (启动脚本 + robustness 检活) |
| 7 | breakdown.kb_provenance 段 collector + warnings 段 |
| 8 | CLI flag (`--no-cortex`, `--cortex-kb-url`, `--cortex-strict-fingerprint`) |
| 9 | resume 兼容 + 老 session 处理 |

每 PR 末尾跑一次"冷启 + KB 写串"小烟测, 验证当前 PR 范围内的写已落
KB / NDJSON。

## 9. 验收清单

- [ ] 一次冷启 v0.8 session, 完整跑 baseline + 1 个 backends propose +
      KEEP/REVERT + report, 在 Cortex 中能查到对应 session_id 的
      session_node + workload_node + opt_node + hypothetical edge +
      ingest_attempt + verify (confirmed 或 refuted) + commit。
- [ ] `--no-cortex` 启动后不写任何 KB, 老 v0.6 行为完全一致。
- [ ] Cortex 模拟不可达 (临时改 URL 到错误地址) 时, NDJSON pending
      增长, 主流程不挂; 切回正确 URL 后 flusher 自动 drain。
- [ ] T4 失败 (人工 kill kb-service) → stop_reason=cortex_drain_failed,
      进程非零退出, session_dir 保留; resume 后 commit 成功。
- [ ] breakdown.kb_provenance 段 cross-check Cortex 实际数据一致。
- [ ] resume 一次 v0.6 session, 第一次 resume 即被接入 KB, 后续 KEEP
      入 KB 正常。

## 10. 风险与回退

主风险:

- **flusher 进程挂掉但主进程还在写 pending** → pending 持续增长 →
  内存 / 磁盘耗尽。缓解: robustness 在 tick 中扫 pending 行数, 超阈值
  发 alert。
- **canonical_id 派生错误导致 KB 重复 mint** → 操作员看到 KB 膨胀。缓
  解: 派生公式上 unit 测试 + Cortex 一侧自然去重。
- **KB 写 Coordinator 集成 bug → 主流程被卡** → 严格走 NDJSON 写一定
  成功 + HTTP/CLI 失败永远只 enqueue, 主流程 never block on Cortex。

回退:

- 整 M1 回退 = 撤掉所有 9 个 PR; 还原 v0.6 行为。
- 局部回退: `--no-cortex` flag 即时 degrade。

## 11. 哲学回引

本里程碑是**主轴 B (知识外接 Cortex)** 的最早落点; **Inv-2 (写经
Coordinator 中转)** 在 KnowledgePlane facade 上具体实现; **Inv-6.3
(时序锚点固定)** 在 T0–T4 各唯一落点。
