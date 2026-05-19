# §3.6 知识平面 — Cortex KB + PR Monitor + 本地源码

## 1. 设计目标

把 v0.6 中"KB 只在 Critic 时点访问、PR 不存在、本地源码靠 LLM 自由读"
的零散现状, 收拢为一个**统一的知识平面 (KnowledgePlane)** 抽象。所
有需要外部知识的角色 / sub-agent 通过这个平面消费, 不直接持有底层
传输的细节。

成功标准:

- 三源 (Cortex KB / PR Monitor / 本地源码) 在 Coordinator 一侧呈现
  为 *单一 facade*; specialist / Critic / prompt 装配的调用方不感知
  后端传输。
- T0–T4 时序契约在新 phase 状态机 (§3.2) 中各有精确落点, 每个时点的
  调用是幂等的、可观测的、有 NDJSON 兜底。
- Cortex 短暂不可达不阻塞 EXPLORE / KERNEL phase 的主流程; 长期不可
  达走显式 fail-fast。
- PR Monitor 的两条接入面 (REST + MCP) 各得其用 — REST 给 Coordinator
  预热, MCP 给 specialist 自由查询。

## 2. 现状回顾

| 源 | v0.6 现状 | 痛点 |
|---|---|---|
| Cortex KB | 没接入。最近的近似品: `kb_digest` (固定 3 query pack 拉 14 行 markdown) + critic-agent 的 `KB_BASE_URL` (仅评审时点) | 没有跨 session 持久化; 没有 hypothesize-verify-commit 协议; Orchestration / specialist 看不到 KB |
| PR Monitor | 完全没接入 | 优化经验完全靠 LLM 自带先验 |
| 本地源码 | 通过 PolicyGate `framework_source_roots` 允许 source_file 引用 (`/sgl-workspace/{aiter,sglang,vllm}/`) | 只有 path 校验, 没有读 / 检索的统一语义 |

## 3. 不变量

继承 §3.1 三主轴 + 三不变量。本节额外引入:

### Inv-6.1 — 三源不重复存储

知识只在它的"权威源"上存:

- 跨 session 的优化经验 / 决策树 → Cortex KB
- 跨仓库的 PR 摘要 / 文件 patch → PR Monitor
- framework / aiter / vLLM / sglang 的实时源码 → 本地文件系统 (read-only mount)

**禁止**: 在 SharedState 里冗余存一份 KB 子图缓存; 在本地 wekafs 上
镜像 PR 摘要; 在 Cortex 中存 framework 源码片段。

### Inv-6.2 — 写经 Coordinator 中转 (重申 §3.1 Inv-2)

任何对 Cortex 的写入 (propose-point / propose-edge / hypothesize /
ingest-attempt / verify / commit) 都通过 Coordinator 内部一个唯一
的 KnowledgePlane.write 接口代发。LLM 角色 / specialist 不持有写
凭证。

### Inv-6.3 — 时序契约固定锚点

T0 / T1 / T2 / T3 / T4 五个锚点在 phase 状态机内有**且仅有**一处落
点, 不允许在多处重复触发同一锚点。例如 T2 hypothesize 仅在
`Coordinator._handle_propose_action` 处理 explore 提议时触发, 其它
地方 (例如 `_promote_to_shared_state`) 不能再做一次 hypothesize。

## 4. KnowledgePlane facade 概念

KnowledgePlane 对外暴露的 *能力* 列表 (具体接口签名留实施稿):

### 4.1 读 (任何角色, 由 Coordinator 在 prompt 装配时调用)

| 能力 | 后端 | 触发时机 |
|---|---|---|
| `find_recipe(workload, hw)` | Cortex `find-recipe` | T0 |
| `get_traps(symptom)` | Cortex `traps` | T0 |
| `traverse(start, steps, branches)` | Cortex `traverse` | T1 / specialist 装配 |
| `pr_feed_warm(domain)` | PR Monitor REST `pr_list` + 关键词过滤 | specialist 装配前预热 |
| `read_source_excerpt(path, line_range)` | 本地文件系统 (受 PolicyGate 校验) | 装配 prompt 时引用 (一般直接交给 specialist 工具) |

### 4.2 写 (仅 Coordinator)

| 能力 | 后端 | 触发时机 |
|---|---|---|
| `propose_point(canonical_id, kind, attrs, evidence)` | Cortex `propose-point` | T0 mint workload/gap; T2 mint optimization_node |
| `hypothesize(sid, from, to, edge_type, attrs, evidence)` | Cortex `session hypothesize` | T2 |
| `ingest_attempt(sid, iter, outcome, metrics, plan_edge, evidence)` | Cortex `ingest-attempt` | T3 (每 variant 跑完) |
| `verify(sid, edge, outcome, evidence, promote_authority)` | Cortex `session verify` | T3 (KEEP→confirmed, REVERT→refuted) |
| `commit(sid)` | Cortex `session commit` | T4 |
| `abort(sid)` | Cortex `session abort` | 异常退出 |

### 4.3 specialist 工具白名单 (specialist 自己调)

specialist 不通过 KnowledgePlane facade, 而是 **以 LLM 工具形式**直接
访问:

- Cortex MCP (只读 traverse / find-recipe / query)
- PR Monitor MCP (全 readonly tools)
- Read / Grep / Glob 工具到 framework_source_roots

specialist 不能调用 Cortex 的写端点 (PolicyGate 在工具白名单层面
即拒)。

## 5. 三源各自的细节

### 5.1 Cortex KB

**接入形态**:

- Coordinator 一侧: 通过 cortex-kb CLI 子进程 (推荐) 或 HTTP 直连
  (兜底); 失败走 NDJSON。
- specialist 一侧: 通过 Cortex MCP server (只读)。

**会话生命周期**:

- T0 `session begin` 必带 attrs:
  - `workload`, `hw`, `image_digest`, `stack_fingerprint`(rocm/aiter/
    sglang/vllm 版本), `marathon_dispatch_id` (来自 manifest.json), `phase` (初始 `PRELUDE`)。
- T2 hypothesize 必带 attrs: `phase`, `round_id`, `domain` (如来自
  specialist), `variant_canonical_fingerprint`。
- T3 ingest-attempt 必带 metrics dict: 至少 `output_throughput`,
  `gain_pct_vs_cb`, `accuracy` (如有 eval), `failure_class` (如失败)。
- T4 commit 必带 closing summary: 本 session 的 phase_history /
  optimization_stack snapshot / cumulative_gain。

**幂等性**:

- 所有 propose-point 用 deterministic canonical_id (例如
  `attempt.session-{sid}.iter#{K}`) → 重复写自动去重。
- session_id 跨重启 resume; pending NDJSON 自动追加。

**NDJSON 兜底**:

- 路径 `<session_dir>/runtime/cortex/.kb_pending.ndjson` (新增子目录,
  避免与现有 runtime 子目录冲突)。
- 推送成功 → 移到 `.kb_flushed.ndjson` (保留供调试)。
- 推送失败 → 留在 pending, 下个 5 秒/50 行 batch 再试。
- T4 commit 之前必须 drain 干净, 否则 commit 拒绝执行 (走 abort 路径
  + 写 stop_reason = `cortex_drain_failed`)。

### 5.2 PR Monitor

**接入形态**:

- Coordinator 一侧: REST (`http://primus-cortex-pr-api.primus-cortex.svc.cluster.local/v1`),
  用于 specialist 装配前的"预热": 拉关心仓库的近期 PR 摘要写入
  prompt。
- specialist 一侧: MCP (`http://primus-cortex-pr-api...svc/mcp/`),
  允许 specialist 自由调 `pr_get` / `pr_patches` / `pr_search` 等。

**配置维度**:

- 关心仓库列表按 domain 配置 (kernel: ROCm/aiter, triton-lang/triton;
  framework: sgl-project/sglang, ROCm/vllm; comm: ROCm/aiter (RCCL /
  QuickReduce), pytorch/pytorch (NCCL plugin); compiler: triton-lang/
  triton, pytorch/pytorch; system: ROCm/hip)。
- 预热摘要的"近期"窗口默认 30 天, CLI flag 可调。

**故障处理**:

- 预热请求失败 → prompt 中 pr_feed 段写 `(unavailable)`, specialist
  自己也无法调 MCP, 但任务不挂。
- specialist MCP 调用失败 → specialist 自行决定是否继续 (prompt 鼓
  励即使 PR feed 空也产出基于 KB 的提议)。

**与 Cortex 的关系**:

- specialist 引用某个 PR (在 specialist_done.proposal_set[i].pr_evidence
  里给 url + sha) → Coordinator 在 T2 hypothesize 时把 PR 作为
  `pr_node` mint, 同时把 hypothetical 边的 evidence 多挂一条
  `kind:url`。
- 同 url 的 pr_node 在 Cortex 中通过 canonical_id 自然去重 (canonical_id
  公式: `pr.<repo>#<number>` 或 `pr.<repo>@<sha>`)。

### 5.3 本地源码

**接入形态**:

- 仅 Read / Grep / Glob 工具, 不允许 Edit / Write (PolicyGate 在工
  具白名单层面就拦截)。
- 受 `framework_source_roots` 限制: `/sgl-workspace/{aiter,sglang,
  vllm}/`, 加上 `INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS` env 中
  追加的目录。

**约定**:

- specialist 读完后, 应当在 specialist_done.proposal_set[i].source_evidence
  里记录 `(path, line_range)` 引用, 而非粘贴大段源码。
- Coordinator 在 T2 hypothesize 时把这些引用作为 `kind:url` 或
  自定义 `kind:source` evidence 写入 Cortex (具体 evidence kind 与
  Cortex 一侧约定)。

## 6. T0–T4 在新 phase 状态机的精确落点

| Cortex 时机 | 触发位置 (phase + 内部 hook) | 什么内容 |
|---|---|---|
| **T0** | PRELUDE 进入后, manifest 写入完成, baseline 跑之前 | `session begin` + `find-recipe` + `traps` + 把 workload/hw mint 为 `workload_node` (canonical: `workload.<model>.<hw>`); 结果存 SharedState.warm_start_* |
| **T1** | EXPLORE 每轮 specialist 装配前 | per-gap `traverse` (3 步 / 4 分支), 5 min LRU 缓存; 结果直接拼进 specialist prompt 第 4 段 (KB subgraph) |
| **T2** | Orchestration 发出 `propose_action='explore'` 经 PolicyGate 通过, 落 PendingProposal 的同一时刻 | 对每个 variant: `propose-point optimization_node (HYPOTHESIZED)` + `hypothesize hypothetical edge from issue_node(gap) to optimization_node`; tentative_edge_id 落到 `PendingProposal.kb_edge_ids[variant_name]` |
| **T3** | explore executor 跑完一个 variant, 调 `_promote_to_shared_state` (KEEP) 或 `_handle_unpromotable_result` (REVERT) | KEEP → `ingest-attempt outcome=PASS` + `verify edge confirmed promote=EXPERIENTIAL`; REVERT → `ingest-attempt outcome=FAIL` + `verify edge refuted` (自动开 negation 边) |
| **T4** | CLOSE phase, NDJSON drain 之后, breakdown 写入之前 | `session commit`; 收到 `promoted_edges` / `derived_summary_id` 写入 SharedState.cortex_session_summary, 供 breakdown 引用 |

每个锚点的失败都不阻塞主流程 *直到一个明确的 boundary*:

- T0 失败 → fail-fast (PRELUDE 退出, stop_reason = `cortex_t0_failed`)。
  原因: 没有 warm_start 的 EXPLORE 跟瞎跑没区别。
- T1 失败 → 该 specialist 的 KB subgraph 段写 `(unavailable)`, 不挂任务。
- T2 失败 → 进 NDJSON pending, 不阻塞 propose_action; tentative_edge_id
  字段留空, 后续 T3 verify 改成 propose-edge + 标 `late_verified`。
- T3 失败 → 进 NDJSON pending, 不阻塞下一 variant。
- T4 失败 → CLOSE phase 退出码非零, 保留 session_dir, 操作员人工 commit
  或下次 resume 时由 Coordinator 自动重试 commit。

## 7. NDJSON 兜底协议

文件布局:

```
<session_dir>/runtime/cortex/
├── .kb_sid                  # T0 写一次, resume 用
├── .kb_warm.json            # T0 拉到的 warm_start 快照 (find-recipe + traps 输出)
├── .kb_pending.ndjson       # 异步队列, append-only
├── .kb_flushed.ndjson       # 推送成功后移过来, 调试用
└── .kb_flusher.pid          # flusher 进程的 pid 文件
```

每行 NDJSON entry 概念字段:

- `op`: `propose_point` / `hypothesize` / `ingest_attempt` / `verify` / `commit` / `abort`
- `payload`: 具体参数
- `created_at`: ISO 时间戳
- `idempotency_key`: 由调用方根据 op + 业务唯一键派生
- `attempts`: 已重试次数

flusher 进程 (由 robustness 拉起、检活):

- 每 5 秒或队列长度 ≥ 50 行, 触发一次 batch flush。
- 单条失败: HTTP 4xx (业务错误) → 移到 `.kb_dead_letter.ndjson` 并发
  alert; HTTP 5xx / 连接错误 → 留在 pending 重试。
- crash 重启: 读 .kb_pending.ndjson, 从头继续。

## 8. 与 critic-agent 现有 KB 通路的合流

v0.6 的 critic-agent 已经有一条 KB 通路 (`KB_BASE_URL` /
`CRITIC_KB_CLIENT_MODE`)。v0.8 的合流方式:

- **不替换**: critic-agent 继续走自己的 `prepare-review` /
  `commit-review` 协议, 不引入新链路。
- **后端共享**: critic-agent 的 `KB_BASE_URL` 指向同一个 Cortex
  kb-service (URL 与 v0.8 KnowledgePlane 一致); 这样 Critic 写入的
  `kb_writes` 与 Coordinator 写入的 `verify` 落在同一个图。
- **写权 boundary**: critic-agent 的 `commit-review` 仍然只写 verdict
  相关边; 不写 `optimization_node` (那是 Coordinator T2 的事)。

合流后 KB 视图统一, 无需关心"这条边是 Coordinator 写的还是 Critic
写的", 都通过 Cortex authority + provenance 字段区分。

## 9. 实施步骤

1. **抽象层**: 设计 KnowledgePlane facade 概念 (单点封装 Cortex CLI /
   HTTP / NDJSON 兜底); 不写代码, 但锁定能力清单 (§4.1 / §4.2)。
2. **NDJSON 协议**: 锁定 file 名 / op 词表 / idempotency_key 派生
   规则, 写入本节作为契约。
3. **flusher 进程**: 概念上是 robustness 的子任务, 与现有
   `robustness_monitor.sh.example` 同样的 setsid nohup 启动模式; pid
   文件 + 健康检查规约。
4. **CLI flag**: 至少新增 `--cortex-kb-url`, `--pr-monitor-url`,
   `--no-cortex` (degrade mode), `--cortex-stack-fingerprint-strict`
   (是否强制把 stack_fingerprint 一致作为 warm_start 过滤条件)。
5. **PolicyGate 工具白名单**: §3.11 详述; 本节锁定 specialist 可调的
   工具集 (Cortex MCP 只读 + PR Monitor MCP 全 readonly)。
6. **prompt 装配回引**: §3.3 / §3.5 已在 prompt 中指定 KB / PR / source
   字段, 本节负责保证装配数据来源正确。

## 10. 边界条件 / 失败模式

| 场景 | 行为 |
|---|---|
| Cortex 在 T0 不可达 | PRELUDE 失败退 CLOSE; stop_reason = `cortex_t0_failed`。NOT degrade-to-empty (那会让 EXPLORE 没有 warm_start) |
| `--no-cortex` 启动 | 跳过 T0/T1/T2/T3/T4 全部锚点; warm_start_* 字段为空; specialist 收到的 KB subgraph = (none); 这是 *显式 degrade*, 仅供调试 |
| T1 traverse 失败 | 该 specialist 的 prompt KB 段写 `(unavailable)`; specialist 仍按 PR + source 跑 |
| T2 失败 (Cortex 5xx 或 NDJSON 写失败) | NDJSON 写一定成功 (本地磁盘); HTTP 失败留 pending 重试; PendingProposal 没有 tentative_edge_id, 后续 T3 走 propose-edge 而非 verify |
| T3 失败 | 同上, NDJSON 兜底; ingest-attempt 进 pending, 主流程不阻塞 |
| T4 失败 (NDJSON drain 失败超时) | CLOSE 退出码非零; stop_reason = `cortex_drain_failed`; resume 时会自动重新 drain + commit |
| PR Monitor 跨集群不可达 | 预热段写 `(unavailable)`; specialist MCP 也不可达; specialist 不挂; pr_intel_specialist 直接报空 proposal_set |
| 本地源码挂载丢失 | specialist 工具调用 Read 失败; specialist 自动降级到只用 KB / PR; 不挂 |
| Cortex 长期不可达 (例如 24h) | NDJSON 持续累积; robustness 检测到 pending > 阈值 (例如 1000 行 / 1h) 发 high alert; 由人工决定是否中止 |

## 11. 验收标准

- [ ] T0 在 PRELUDE 入口被调用一次, 失败即 fail-fast (除 `--no-cortex`
      模式)。
- [ ] T1 traverse 调用频率 ≤ 1 次 / specialist 派发 / 5 min, LRU 命中
      率可观测。
- [ ] T2 在每个 propose_action='explore' 通过 PolicyGate 后被调用,
      tentative_edge_id 写入 PendingProposal。
- [ ] T3 在每个 variant KEEP/REVERT 时被调用, NDJSON pending 增长可
      观测。
- [ ] T4 在 CLOSE 内被调用一次, drain 后 pending 长度 = 0; commit 成功。
- [ ] critic-agent 的 `kb_writes` 与 Coordinator 的 `verify` 在同一个
      Cortex 图, breakdown.kb_provenance 中可同时看到两类条目。
- [ ] PR Monitor 预热在 specialist 装配前完成, prompt 中 pr_feed 段
      非空 (PR Monitor 可达时)。
- [ ] specialist 不能调 cortex-kb 写端点 (尝试 → 工具白名单拒)。

## 12. 依赖与影响面

- **上游**: §3.1 (主轴 B 知识外接), §3.2 (phase 锚点位置), §3.3
  (角色 prompt 注入), §3.5 (specialist 工具集)。
- **下游**:
  - §3.7 research_lane 容量影响 specialist 一次能并发多少, 也影响 PR
    Monitor 调用并发。
  - §3.9 砍 scoreboard 后, KB warm_start 是 LLM 的主要 prior。
  - §3.10 SharedState 新字段 `warm_start_recipe` / `warm_start_pitfalls`
    / `cortex_session_id` / `pending_kb_edges`。
  - §3.11 PolicyGate 工具白名单 + 写权 boundary。
  - §3.12 breakdown 中 `kb_provenance` 段。
  - §3.13 milestone M1 (Cortex 接入, 纯写) + M4 (PR Monitor 接入)。

## 13. 哲学回引

本节是**主轴 B** 的核心落地: 知识外接 Cortex, 跨 session 持久化;
**Inv-2** (写经 Coordinator 中转) 在 §4.2 显式落实;
**Inv-6.1 三源不重复存储**, **Inv-6.2 写经中转**, **Inv-6.3 时序契约
固定锚点** 是本节内部的强约束, 也是后续章节验收依据。
