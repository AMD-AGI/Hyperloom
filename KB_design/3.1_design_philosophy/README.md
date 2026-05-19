# §3.1 设计哲学 — 三主轴与不变量

## 1. 设计目标

为 v0.8 所有后续章节 (§3.2–§3.15) 给出**唯一一致的取舍标准**:
当两个设计选项冲突时, 应当依据本节的三主轴 + 不变量挑选, 而不是临时
讨价还价。本节本身**不规定具体机制**, 只锚定"为什么这么做"。

## 2. 现状回顾

v0.6 的设计哲学在三个维度上是有矛盾的, v0.8 的核心目标就是消除这些
矛盾:

| v0.6 立场 | v0.6 体现 | 矛盾点 |
|---|---|---|
| "由代码评分调度 action" | `MARATHON_PRIORS` + `scoring.py` + `streak` + `cooldown` + prompt 渲染 top-12 | 评分代码与 prompt 渲染要同步演化, LLM 又必须再次理解评分含义, 双层认知成本 |
| "知识只在 Critic 评审时点存在" | `CRITIC_KB_CLIENT_MODE` 仅 Critic 调用, KB 写权只在 Critic | session 销毁后 95% 的"刚学到的东西"丢失, 下一次 session 只能从 `MARATHON_PRIORS` 这种粗粒度 prior 起步 |
| "sub-agent 都是确定性 Python 执行体" | `SubAgentRunner.executor_registry` 全部是 `async def fn(ctx) -> dict` | 想加一个 "看 PR / 看源码 / 提议参数" 的能力时, 没有合适的承载层 — 写成 deterministic 又会和 Magpie wrapper 强耦合 |

v0.8 对这三处分别给出针对性的主轴。

## 3. 三主轴

### 3.1 主轴 A — 流程固化优先于评分动态调度

**断言**: 真实的优化收益结构是 *先填配置面 → 再填内核面 → 最后跨
workload 验证*。这个顺序是**领域事实**, 不是优化算法的输出。任何把
顺序当变量的设计 (评分 / cooldown / streak) 都在花算力解一个**已经
有答案**的问题。

**推论**:

- phase 之间的转移由 *Coordinator 强制*, 不由 LLM 自由选择 (LLM 只能
  在 phase 内决策)。
- 同一 phase 内, 哪个 action 优先 / 派多少个 specialist / 选哪个
  domain, **由 LLM 基于 gaps + KB 决定**, 系统不再做数值排序。
- 当 LLM 错判应当跳 phase 时, 显式走 robustness 的
  `escalate_strategy_change` 软退路径, 不是隐式跳转。

**反主轴 (拒绝的诱惑)**:

- "先固化, 再加一些 phase 内评分作 nudge" — 拒绝。半评分系统会让 LLM
  和系统的认知不一致再次出现。
- "保留 cooldown 防止反复尝试" — 用 KB negation edge 替代 (refuted →
  自动过滤), 不再用代码侧时钟。

### 3.2 主轴 B — 知识从单 session 抽出, 拉到 Cortex

**断言**: "知识"和"事实"应当分层。事实层 (本次 baseline 多少, 哪个
variant KEEP, 哪个 kernel REVERT) 留在 SharedState; 知识层 (这类
模型在这类 GPU 上, 这类 flag 通常有效, 那类 PR 已被验证) 必须跨
session 持久化, 否则系统的"长记忆"无从谈起。

**推论**:

- `state.json` 不再存"上次 session 学到的东西"。它只存本次 session
  的活动事实层。
- 跨 session 的全部知识走 Cortex KB (图 + edge authority + negation),
  以**幂等的 canonical_id** 作为去重锚点。
- session 启动 (T0) 必须从 Cortex 拉 warm_start 写到 SharedState 的
  专门字段; 这一步是新流程的"读起点", 与 baseline 一样 mandatory。
- session 收尾 (T4) 必须 commit 整段 hypothesize→verify 链, 否则本次
  session 的知识贡献为 0, 视为可重新跑。

**反主轴 (拒绝的诱惑)**:

- "把 KB 也复制到 session_dir 一份, 防止 Cortex 不可达" — 拒绝。NDJSON
  兜底已经覆盖短暂不可达; 若长时间不可达应直接错误退出, 而不是用陈旧
  快照欺骗 LLM。
- "在 SharedState 里也存一份冗余 best_config 副本" — 拒绝。冗余 =
  双源一致性问题; warm_start_recipe 字段是 T0 一次性快照, 之后**只读
  不写**, 不与 Cortex 双向同步。

### 3.3 主轴 C — 执行体一分为二: deterministic + LLM specialist

**断言**: "跑一个 benchmark / 应用一个 patch / 解析 trace" 这些有标准
答案的工作必须由确定性代码做; "看 KB / 读 PR / 翻框架源码 / 提出 6 个
候选 variant" 这些有判断的工作必须由 LLM 做。把它们硬塞进同一种
sub-agent 形态会两边都做不好。

**推论**:

- v0.6 的所有 deterministic executor (`baseline_executor` /
  `params_executor` / `kernel_*_handlers` / ...) 全部保留, 不动。
- 新增一类 LLM specialist sub-agent, **只做提案**, 不跑 E2E bench, 不
  应用 patch, 不抢 serving GPU。
- specialist 的输出是 *提案 + 引用证据*, 不是 patch 文件。Coordinator
  收到提案后, 走原 `propose_action → Critic Review → explore executor`
  的标准管线; specialist 的存在不增加新决策路径, 只增加**决策原料的
  来源**。
- 这是与 TBO 的关键差异: TBO specialist 会自己跑 micro-bench 并产出
  patch, v0.8 不允许。原因: Hyperloom 的 PolicyGate / Critic / 串行
  bench 闸是高价值组件, 不应被 specialist 旁路。

**反主轴 (拒绝的诱惑)**:

- "让 specialist 直接跑 micro-bench 加快收敛" — 拒绝。会引入"micro
  speedup ≠ E2E speedup" 这一 TBO 已经踩过的坑, 而我们没有 TBO 的
  shape_capture 校验链。
- "specialist 直接调 cortex-kb hypothesize 写 KB" — 拒绝。任何 KB 写
  都必须经过 Coordinator 中转, 这样 Critic Review 的"事前留痕"才完整。

## 4. 三条不变量 (跨章节强约束)

任何后续章节的设计如果违反下列三条之一, 都视为**有 bug**, 必须修改
方案:

### Inv-1 — 事实层单写者

`SharedState` 中的 *核心事实字段* (`baseline_tput` / `current_best` /
`cumulative_gain` / `optimization_stack` / `phase` / `stop_reason`)
**只能由 Coordinator 写**。LLM 角色一律通过 intent 申请, PolicyGate
拒绝越权写入。这继承自 v0.6 `CORE_STATE_FIELDS` 概念, v0.8 的扩展只
增加成员, 不放松写权。

### Inv-2 — 知识写经 Coordinator 中转

任何对 Cortex KB 的写 (propose-point / propose-edge / hypothesize /
ingest-attempt / verify / commit) 都由 *Coordinator* 代发, 调用方
仅以 intent 表达"要写什么"。LLM 角色 (包括 specialist) 不直接持有
Cortex CLI / HTTP 客户端的写权限。

理由: Critic Review 必须能看到完整的"事前 hypothesize", 跳过中转就
绕过了评审; 同时也方便加幂等去重 / 频率控制 / 失败重放。

### Inv-3 — serving GPU 单租户

任何 phase 任何时刻, 在 serving GPU 上跑 E2E bench 的进程**只有一
个**, 由 `benchmark_lane` (单容量) 强制。specialist / research 类
任务一律不进 serving GPU, 由新增的 `research_lane` 收容并允许并发。

理由: 数据正确性比并发收益高一个数量级; "并行探索 + 串行验证"是 TBO
已验证的有效形态, 不应在 v0.8 一开始就追求"并行验证"。

## 5. 决策原则 (设计冲突时的仲裁)

当后续章节 / 子文件夹中两位设计者对同一个点有不同主张时, 按下表
仲裁:

| 维度 | 偏向选择 | 拒绝选择 | 理由锚点 |
|---|---|---|---|
| 决策落在代码 vs 落在 LLM | 落 LLM | 落代码评分 | 主轴 A |
| 信息存 SharedState vs 存 Cortex | 事实存 State, 知识存 Cortex | 双源都存 | 主轴 B |
| 加新 deterministic executor vs 加新 specialist | 看本质工作: 有标准答案 → executor; 有判断 → specialist | 双形态混合 | 主轴 C |
| 增加 PolicyGate 规则 vs 让 LLM 自律 | 增加 PolicyGate | 仅靠 prompt | Inv-1/Inv-2 |
| 加新 lane vs 复用旧 lane | 优先看冲突语义; 不强行复用 | 把 research 塞进 benchmark_lane | Inv-3 |
| 引入冗余字段 / 缓存 | 拒绝 | 接受 | 主轴 B |
| 新功能默认开 vs 默认关 | 默认关 (CLI flag), 灰度开 | 默认开 | 风险面控制 |

## 6. 不变量的可观测性

每条不变量必须**可被 session_breakdown / 监控数据探测**, 否则我们
没办法证明它没被违反:

| 不变量 | 探测方式 |
|---|---|
| Inv-1 | breakdown 中的 `core_state_writes` 段 (新增) 必须只看到 `actor=coordinator` |
| Inv-2 | Cortex 一侧 propose/hypothesize 的 `provenance.generator` 字段必须只看到 `inference_optimizer.coordinator`, 不出现 specialist 子进程的 ID |
| Inv-3 | breakdown `phase_timeline` 段在任意时刻 `benchmark_lane.holders ≤ 1` |

§3.12 (observability) 章节将这三条探测落到具体段落 / 字段。

## 7. 实施步骤

本节是哲学层, 没有"实施步骤" — 它通过被引用而生效。具体落地步骤是:

1. **写入本文件**: 锁定三主轴 + 三不变量 + 决策仲裁表 (本节即是)。
2. **要求所有后续 §3.x 文件在文末显式回引**: 每份设计 MD 必须写明
   "本节遵守 Inv-X / 主轴 X" 或 "本节是主轴 X 的具体落地"。
3. **PR 评审清单**: 每个 v0.8 PR 评审时, reviewer 必须勾选三主轴
   是否被尊重 + 三不变量是否未被破坏。

## 8. 验收标准

- [ ] §3.2–§3.15 所有 README 文末都有"哲学回引"段, 显式引用 Inv-X /
      主轴 X。
- [ ] M1–M7 任一个里程碑落地后, §6 的三个观测点都给出**绿灯**数据。
- [ ] 出现一次"我们要破坏 Inv-X 一下" 的提议, 都必须先回到本文件
      讨论是否要修改不变量本身 — 不允许"局部破例"。

## 9. 依赖与影响面

- **上游依赖**: 无 (本节是根)。
- **下游影响**: 全部其他章节都从这里取仲裁标准。

## 10. 哲学回引

(本节自身即哲学源, 不需要回引。)
