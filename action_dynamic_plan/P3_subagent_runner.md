# P3 — Sub-agent Runner（Multi-turn ReAct）

> 第三阶段的对外可见产物：dynamic sub-agent 在隔离 worktree 内通过
> multi-turn ReAct 探索，最终产出一个跨域 patch；所有探索过程在
> journal 中可复检；探索过程的物理边界由 runner 侧硬编码，不由 LLM
> 在 prompt 里自由扩展。
>
> 对应 dynamic_action.MD §3.4。

---

## 1. 目标

把 P1/P2 的 stub executor 替换为真正的 dynamic sub-agent runner。本
阶段后，dynamic action 已经能在 worktree 内通过多轮 LLM call + 工具
执行产出 `proposal_set.json` 和 `sub_agent_journal.md`，但产出的
patch **尚未接入 Critic / integrate_patch**——那一步留给 P5。

具体可观测的产物：

1. dispatch → runner 启动 → multi-turn ReAct 循环 → emit_proposal →
   写 `proposal_set.json` + `sub_agent_journal.md`；
2. budget 三层（wall-clock / turn / token）任一耗尽 → runner 强制终止
   并记录终态；
3. 工具白名单的 4 类资源调用全部落在 journal 上，可独立审计；
4. 回收路径白名单生效——worktree 内除 `proposal_set.json` /
   `sub_agent_journal.md` 外的任何文件都不会被收回。

---

## 2. 触及的架构平面

| 平面 | 改动性质 | 中心思想 |
|---|---|---|
| Sub-agent runner | 新建 1 个独立 module，与 specialist runner 平级 | 复用底层任务派单 + worktree 隔离，但 prompt/budget/journal 各自独立 |
| Worktree 隔离 | 复用 specialist 的隔离机制 | 仅换 worktree base 路径与 branch 命名 |
| Prompt builder | 新建 1 个独立 module | 多轮 prompt 装配 + 工具调用约定 |
| 工具白名单 | 新建（runner 侧） | 名词级白名单：4 类资源 + 1 个终止信号 |
| Budget enforcement | 新建（runner 侧） | wall-clock + turn + token 三层硬限 |
| 回收路径白名单 | 新建（runner 侧） | 仅 2 个固定文件，机械保障 §1.2 红线 |
| Bench 注册表 | 新建 | runner 侧维护可执行 bench script 列表 + 各自 wall-clock 上限 |

---

## 3. Runner 形态（Q1 = (b)）

按 Q1 决策，runner 在外部 journal 文件中维护 multi-turn 状态；每轮
都启一个独立的 `claude` 子进程；sub-agent 是无状态的多次调用。

### 3.1 主循环骨架（概念伪代码）

```
journal = []          # 写到 sub_agent_journal.md
turn = 0
deadline = now() + 15min
while turn < TURN_CAP and now() < deadline:
    prompt = build_turn_prompt(seed_kit, journal, last_tool_result)
    out   = invoke_claude(prompt, token_caps)
    journal.append(out)
    action = parse_next_action(out)
    if action is emit_proposal:
        validate_proposal(action.payload)
        write_proposal_set(action.payload)
        return COMPLETED
    elif action is tool_call:
        result = run_tool(action.name, action.args)        # 受白名单约束
        last_tool_result = result
        journal.append(result_summary)
    else:
        return FAILED(reason="unparsable_output")
    turn += 1
return TIMED_OUT
```

### 3.2 设计要点

- **Runner 拼 prompt，不让 sub-agent 自由选 context**：每轮 prompt
  完全由 runner 构造（seed kit + journal 摘要 + 上轮 tool result），
  sub-agent 不能"我让你看 X"地操控 runner 给的 context。
- **解析层是窄接口**：runner 只识别两种动作 —— `emit_proposal` 与
  `tool_call`（≥1 类）；其他输出形态视为解析失败、终止当前 sub-process。
- **每轮独立 sub-process**：单个 claude 子进程的失败/超时只影响本轮，
  runner 仍可重试本轮（DEFAULT 不重试，详见 §10）。
- **状态恢复点 = journal**：journal 是唯一持久化的状态；中途崩溃后
  能否续跑，本质上看 journal 的完整性。v1 的策略是**不续跑**（见 P8），
  即使中途崩溃也直接 abandoned。
- **历史压缩**：journal 长度增长 → prompt token 上压；当 journal 摘要
  超过单轮输入 cap 的 70% 时，runner 进行**机械式截断**（保留最早 N
  轮 + 最近 N 轮，中间用占位符），不做 LLM 总结（避免引入新的 LLM 调
  用环节）。

### 3.3 Budget 三层

| 层 | DEFAULT | enforcement |
|---|---|---|
| wall-clock 总 budget | 15 分钟 | runner 在每轮开始前检查 deadline |
| turn cap | 12 轮 LLM call | runner 累加 turn 计数 |
| 单轮 token cap | input ≤ 32K, output ≤ 4K | invoke_claude 调用参数硬编码 |

任一层超出 → runner 强制终止当前 sub-process，状态置 `TIMED_OUT`。

### 3.4 与 specialist runner 的关系

dynamic runner 与 specialist runner **平级**（兄弟模块），不是子类。

- 共享：底层 SubAgentRunner 派单接口、worktree 隔离机制、研究 lane
  抢占机制、log 写入接口。
- 不共享：prompt 装配、turn loop、工具白名单、回收路径白名单、budget
  enforcement、journal 形态。

强行继承会让"specialist 是 single-shot、dynamic 是 multi-turn"这条
主要差异隐藏在 if/else 里，runner 演化耦合，未来想动 specialist 的
prompt 形态会牵连 dynamic。

---

## 4. 工具白名单（名词级，4 类资源 + 1 终止信号）

工具集合是 dynamic action 探索能力的**唯一物理边界**。设计原则：

- **名词级而非动词级**：白名单声明可访问的资源（"可读源码"/"可读
  artifact"/"可跑 bench"），而不是 LLM 工具调用名（API 命名）。这层
  映射由 runner 侧维护。
- **数量极少**：v1 仅 4 类资源 + 1 个终止信号，无未来扩展承诺。
- **每条都有运行时硬限**：不能仅靠 prompt 描述约束。

### 4.1 资源类（受控读 + 受控执行）

#### a. `read_source(path)` — 读源码

- 只能读 `INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS` 内的文件；
- 路径越界 → runner 返回 error 给 LLM（不抛终止），journal 记录；
- 单次返回内容 cap（DEFAULT 4K tokens），超长截断；
- 不允许通配符或目录列举（避免 sub-agent 枚举源码树）。

#### b. `read_session_artifact(path)` — 读 session artifact

- 白名单路径（DEFAULT）：
  - `$SESSION_DIR/runs/grid/.../result.json` 之类的 grid 历史结果；
  - `$SESSION_DIR/agents/orchestration/dynamic_actions/<dyn_id>/seed_kit.json`
    （即自己的 seed kit，但 sub-agent 已经在 prompt 里见到，只是允许
    它再 fetch 全文）；
  - 可选：自身 worktree 内的 `proposal_set.json`（用于自我检查）。
- 黑名单（必须显式 deny）：
  - **其他 dyn_id 的目录**——避免 dynamic 之间互相学习；
  - **agents/orchestration/inbox.jsonl / outbox.jsonl**——避免读
    orchestration 内部消息；
  - **agents/critic/** 目录——避免读其他 review 历史；
  - **runs/specialist/<task_id>/** 目录——specialist 工作目录隔离；
  - **任何 agents/specialists/<other_domain>/ 的输出**。

#### c. `run_bench(bench_id, params)` — 跑限时 micro-bench

- `bench_id` 必须在 runner 侧的 bench 注册表中（DEFAULT 见 §4.3）；
- 不允许 sub-agent 提供 bench 脚本路径或源码——只能选注册表中的；
- 单次 wall-clock ≤ 60s（subprocess timeout 强制），超时 → 返回
  `timed_out`，不计入 proposal 证据；
- 输出严格隔离到 worktree 内 `scratch/bench/<bench_id>/<call_id>/`，
  **runner 不收回此目录**（回收白名单只含 proposal_set.json + journal）。

#### d. `apply_patch_in_worktree(patch_text)` — 在 worktree 内试应用 patch

- 仅在自己的 worktree 内 `git apply`；
- 失败（hunk 冲突等）→ 返回 error 给 LLM，方便 sub-agent 调整 patch；
- 这一类是为了让 sub-agent 在 emit_proposal 之前能**自检 patch 是否
  apply-able**。不开这条 sub-agent 可能产出 apply 不上的 patch，浪费
  下游 critic + integrate_patch 资源。
- 注意：worktree 内的 apply 是**临时**的——runner 在终止时 reset
  worktree 到干净状态；最终的 patch 落地仍走下游 `integrate_patch`。

### 4.2 终止信号

#### `emit_proposal(patch_text, scope_domains, cross_domain_rationale, expected_qualitative_argument)`

- LLM 通过这个动作宣告"探索完成"，runner 解析后立即终止循环；
- 字段约束：
  - `patch_text`：unified diff 格式，单 patch；
  - `scope_domains`：必须是 P1 payload 中 `scope_domains` 的子集；
  - `cross_domain_rationale`：解释为什么这是跨域组合（每个 domain
    的改动 rationale + 跨 domain 耦合点）；
  - `expected_qualitative_argument`：定性论证（**不允许**任何 quantitative
    数字声称——见 §5）。
- 解析失败 → runner 返回 error 给 LLM，本轮算消耗 turn-cap 但不视
  作终止；继续下一轮（或耗尽 cap 后 FAILED）。

### 4.3 Bench 注册表（DEFAULT 起步集合）

runner 侧维护 bench 注册表：

| bench_id | 内容 | wall-clock 上限 |
|---|---|---|
| `kernel_attention_timing` | 单 attention layer 的 forward timing | 30s |
| `kernel_gemm_timing` | gemm op 的 timing 与 occupancy | 20s |
| `kernel_kvcache_layout` | KV cache 不同布局的 read/write throughput | 30s |
| `inference_short_prompt` | 极短 prompt（≤32 tokens）+ 极短 max_tokens（≤32）的端到端 latency | 60s |

**起步集合保守**：仅囊括最常用的几类。新增 bench 必须显式扩注册表
（即新增 bench 是设计变更，不允许 sub-agent 在 prompt 里临时定义）。

---

## 5. 输出 Schema（proposal_set.json）

dynamic action 输出与 specialist 同构，以最大化下游路径复用：

### 5.1 字段集（封闭）

- `proposal_set`：list，DEFAULT cap = 1（Q3 决策）。每项：
  - `name`：proposal 的简称（与 specialist proposal 同字段）；
  - `provenance`：固定字面量 `"dynamic"`（不可改写）；
  - `patch_text`：unified diff，单文件或多文件均可；
  - `scope_domains`：与 spec.json 中 `scope_domains` 一致或为子集；
  - `cross_domain_rationale`：跨域论证文本；
  - `expected_qualitative_argument`：定性预期（**不允许数字**）。
- `empty`：bool，proposal_set 为空时 = true（与 specialist empty 路径
  对齐）。
- `journal_path`：指向 `sub_agent_journal.md` 的路径。

### 5.2 显式禁止字段（runner 侧 reject）

- `expected_gain` 数字字段（与 specialist 不同——dynamic 不允许声明
  metric，§1.2 红线）；
- `bench_evidence` 字段（micro-bench 数字不进 promote 链路，§1.2 红线）；
- 任何 `confidence` / `score` / `rank` 数字字段（避免 sub-agent 用数
  字伪装 priority）；
- `force_provenance` 等覆写 provenance 的字段。

### 5.3 runner 侧 validation

LLM 输出 `emit_proposal` 后，runner 在写 proposal_set.json 前做以下
强制检查：

- 字段集严格匹配 §5.1（多余字段 → reject）；
- `provenance == "dynamic"`；
- `scope_domains ⊆ spec.json.payload.scope_domains`；
- `patch_text` 为合法 unified diff（语法层检查，不 apply）；
- `expected_qualitative_argument` 中通过简单正则筛查不含明显的数字
  声称（如 `\d+%`、`\d+\.\d+x`）——发现 → reject 或 warn（DEFAULT
  reject，详见 §10）；
- `cross_domain_rationale` 必须显式提及 `scope_domains` 列表中的每个
  domain（出现频次 ≥ 1）。

reject 的 sub-agent 输出 → runner 把 reject reason 回写给 LLM 让其
修正（仍占用 turn-cap 一轮）；连续 reject 超过 N 次（DEFAULT 2）→
runner 终止，状态置 `FAILED(reason="proposal_validation_failed")`。

---

## 6. 回收路径白名单（机械保障 §1.2 红线）

runner 在 sub-agent 终止后**只回收**两个固定文件到
`agents/orchestration/dynamic_actions/<dyn_id>/`：

- `proposal_set.json`（来自 worktree 根）；
- `sub_agent_journal.md`（runner 全程写入的 turn-by-turn 历史）。

worktree 内的其他任何文件（`scratch/bench/...` / `scratch/scribbles.txt`
/ 试 apply 的临时 commit / sub-agent 自己写的中间文件）**不**被回收。

### 中心思想

- **白名单硬编码在 runner 中**——sub-agent 不能通过 prompt 协商
  "请把我的 X.json 也带回去"；
- **回收完成后 worktree 立即销毁**（git worktree remove + branch
  delete）——避免后续阶段意外读到 sub-agent 的内部诊断文件；
- **bench 输出永远不进入 SharedState**——这是 §1.2 红线的物理基础。
  即使未来想"统计 bench 是否预测了 KEEP"等 ML-style 分析，也只能基于
  journal 文本（journal 中的 bench 调用会被记录），不能直接读 bench
  输出文件——bench 输出在销毁 worktree 时一并消失。

---

## 7. Worktree 隔离

复用 specialist 的 worktree 隔离机制，仅以下不同：

- worktree base：`$SESSION_DIR/runs/dynamic/<dyn_id>/worktree/`（与
  specialist 的 `runs/specialist/<task_id>/worktree/` 平级，根目录不同）；
- branch 名：`dynamic-<dyn_id>`（与 specialist 的 `specialist-<task_id>`
  风格对齐）；
- 销毁策略：sub-agent 终止后**立即**销毁 worktree（不保留供后续 review）；
  journal 与 proposal_set.json 已回收到 artifact 目录，不需要保留 worktree
  本身。

---

## 8. Sub-agent 失败/终态分类

runner 退出时必须给一个明确的终态标签，对应到 P6 的状态机：

| 终态 | 含义 | 触发条件 |
|---|---|---|
| `COMPLETED` | sub-agent emit 了合法 proposal | emit_proposal 通过 §5.3 validation |
| `COMPLETED_EMPTY` | sub-agent 主动 emit 空（即声称无可行方案） | emit_proposal 但 patch 为空 |
| `TIMED_OUT` | budget 耗尽（wall-clock / turn cap） | runner 主动 kill |
| `FAILED` | 解析失败 / 多次 proposal validation reject / 子进程崩溃 | runner 检测到不可恢复错误 |

每个终态附带结构化 reason（如 `timed_out_walclock` / `timed_out_turn_cap`
/ `proposal_validation_failed` / `subprocess_crashed`）。

`COMPLETED_EMPTY` 的引入：避免 sub-agent 被迫"硬挤一个 patch"。如果
跨域组合在 sub-agent 视角下不成立，empty 是合法终态——下游与
specialist `empty=true` 等价处理（详见 P5 §6）。

---

## 9. 依赖与前置条件

P3 必须在 P2 之后实施。P3 依赖：

- P2 的 `seed_kit.json`（runner 启动时读取）；
- P2 的 artifact 目录（journal 与 proposal_set 写入此目录）；
- P1 的 dispatch 通道；
- 既有 `SubAgentRunner` 派单链路、worktree 创建逻辑、研究 lane 抢占
  机制。

P3 的产物会被 P4 / P5 直接复用：

- P4 critic 读 `proposal_set.json` 做审查；
- P5 把 `proposal_set.json` 转成 grid variant 进入 `_grid_runner`。

---

## 10. 验收信号

| # | 测试场景 | 期望结果 |
|---|---|---|
| 1 | 正常派发 → sub-agent 在 ≤12 轮内 emit 合法 proposal | 终态 COMPLETED；proposal_set.json + journal.md 落盘；worktree 已销毁 |
| 2 | sub-agent 在 turn cap 内未 emit | 终态 TIMED_OUT(turn_cap)；proposal_set.json 为空 stub；journal 完整 |
| 3 | sub-agent 在 wall-clock 内未 emit | 终态 TIMED_OUT(walclock) |
| 4 | sub-agent emit 含 expected_gain 数字字段 | runner reject；记 journal；占用一轮；sub-agent 修正后通过 |
| 5 | sub-agent 多次 reject 累计 | 累计 ≥2 次 reject → 终态 FAILED |
| 6 | sub-agent 调 read_source 读 framework_source_roots 外文件 | 工具返回 error；journal 记录；sub-agent 可继续 |
| 7 | sub-agent 调 run_bench 用未注册的 bench_id | 工具返回 error |
| 8 | sub-agent 调 read_session_artifact 读其他 dyn_id 目录 | 工具返回 error（黑名单） |
| 9 | sub-agent 输出 emit 之外的非 tool_call 内容 | 第一次：runner 回错让其修正；连续 → FAILED |
| 10 | bench 单次超 60s | subprocess timeout；返回 timed_out；sub-agent 看到 |
| 11 | sub-agent emit 空 patch（声称无可行方案） | 终态 COMPLETED_EMPTY |
| 12 | 主动 abort（外部 kill）→ runner 接到信号 | 终态 ABANDONED；artifact 保留；worktree 销毁 |

---

## 11. DEFAULT / 待 review

| # | 条目 | DEFAULT | 备注 |
|---|---|---|---|
| 1 | wall-clock budget | 15 分钟 | Q2 决策 |
| 2 | turn cap | 12 轮 | README §3 #9 |
| 3 | 单轮 input token cap | 32K | README §3 #12 |
| 4 | 单轮 output token cap | 4K | README §3 #12 |
| 5 | journal 历史压缩阈值 | 占输入 cap 70% 触发机械截断 | 待 review |
| 6 | 历史截断策略 | 保留最早 2 轮 + 最近 4 轮 + 占位符 | 待 review |
| 7 | proposal validation 连续 reject 上限 | 2 | 待 review |
| 8 | bench 注册表起步集合 | 见 §4.3 | 待 review |
| 9 | bench 单次 wall-clock 上限 | 60s | Q2 决策 |
| 10 | apply_patch_in_worktree 是否开放 | 开放（自检 patch apply-able） | 待 review |
| 11 | `expected_qualitative_argument` 数字声称检测 | 简单正则筛查（reject） | 待 review，可能误伤合理表述如 "ratio of 1:1" |
| 12 | 子进程崩溃是否重试 | 不重试，直接 FAILED | 待 review |
| 13 | worktree 终态后是否保留 | 不保留（销毁） | 待 review |

---

## 12. 与 §1.2 红线的对应关系

| 红线 | 在 P3 的落点 |
|---|---|
| 跨 domain 的 KB + 全 profile + 全 roofline + 已尝试 patches 摘要（输入语料） | seed kit 已在 P2 完成；P3 runner 仅原样 reference |
| ReAct 多轮探索 + 工具调用 | §3 主循环 + §4 工具白名单 |
| 跑 micro-bench（仅作内部假设验证） | §4.1.c run_bench + §6 回收白名单 + §5.2 输出 schema 显式禁止 bench 数字字段 |
| 不能起独立 server / 跑 Magpie | bench 注册表中绝不包含 server 启动或 Magpie 调用 |
| 不能声明自己的 metric | §5.2 显式禁止 expected_gain 等数字字段 |
| 不能落 patch 不经 integrate_patch | runner 输出仅 proposal_set.json 这一个机械约束的 schema，patch 落地必须经下游 critic + integrate_patch；runner 自己不写 framework source |
| 不能动 SharedState 受保护字段 | runner 在 worktree 隔离内运行，看不到 SharedState 写接口 |
