# P9 — 测试矩阵

> 第九阶段的对外可见产物：以可重复的方式守住 §1.2 红线。本阶段的核
> 心产出不是"代码可以跑"，而是"哪怕实现期偏差、未来 refactor、新人接
> 手，红线都还在"。
>
> 对应 dynamic_action.MD §3.10。

---

## 1. 目标

为 dynamic action 建立三层测试金字塔：

1. **单元层**：每条 PolicyGate 规则、IR-4 provenance 规则、payload
   schema 规则、seed kit 装配规则、proposal validation 规则都有显式
   negative + positive test。
2. **集成层**：mocked sub-agent 输出固定 proposal_set，验证 critic →
   integrate_patch → grid 全链路；dynamic + specialist 在同一 EXPLORE
   round 共存且资源 lane 不冲突。
3. **不变量层**：以专门回归用例**机械化**地守住 §1.2 红线——这层是
   全套测试中最关键的，红线一旦在实现期被悄悄突破，这层应当 fail。

测试用例的编写**不滞后于** P1–P8 的实施：每完成一个阶段，对应单元层
和集成层用例同步入库；P5 之后做集成层与不变量层的补全。

---

## 2. 触及的架构平面

| 平面 | 改动性质 | 中心思想 |
|---|---|---|
| 单元测试 | 新建 testset，按 PX 阶段分组 | 每条规则一个 case |
| 集成测试 | 新建 testset，含 mocked sub-agent | 端到端 happy path + 主要失败模式 |
| 不变量测试 | 新建独立 testset | 专攻 §1.2 红线的机械证据 |
| Mocked sub-agent | 新建 test fixture | 跑出固定 proposal_set，绕过真 LLM |
| 既有回归测试 | 不动 specialist 既有用例 | 验证 dynamic 接入未破坏 specialist |

---

## 3. 单元层（按 PX 阶段索引）

### 3.1 P1 PolicyGate 规则

| 测试组 | 验证内容 | DEFAULT 用例数 |
|---|---|---|
| Phase 限制 | EXPLORE 通过 / PRELUDE / FRAMEWORK_PR / KERNEL / SWEEP / CLOSE 全拒 | 6 |
| Source 限制 | orchestration 通过 / specialist / critic / framework / external 全拒 | 5 |
| Payload schema 完整性 | 三必填字段缺失各 1 例 + 全空 | 4 |
| scope_domains 约束 | length=0 / length=1 / 含未注册 domain / 全 kernel / 合法 ≥2 个 | 5 |
| side_effects 红线 | 含 kernel_owned / metric / accuracy_gate / server / 合法 | 5 |
| budget_hint 取值 | low / medium / high 通过；invalid / null 行为 | 4 |
| Round-cap | round 内首条通过；超 cap 拒；拒绝不消耗 cap | 3 |
| reason code | 每个失败场景 reason code 命中既定枚举 | 与上述拒绝用例同源 |

合计：约 30+ 个用例，覆盖 P1 §4 全部校验项。

### 3.2 P1 IR-4 Provenance 白名单

| 测试组 | 验证内容 |
|---|---|
| 白名单接受 | `specialist:foo` / `default_grid` / `dynamic` 三类 stamp 通过 |
| 白名单拒绝 | `dynamic:kv_cache+scheduler` 复合形式 / `dyn` 缩写 / 空字符串 / 大小写变体 全拒 |
| MAX_DYNAMIC_SOURCED | round 内第二条 dynamic stamp variant 拒 |

### 3.3 P2 Seed kit 装配

| 测试组 | 验证内容 |
|---|---|
| 完整路径 | 完整 SharedState + roofline + profile → 各字段填充正确 |
| 降级路径 | 缺 roofline → roofline_summary 为空 + degraded 标记；缺 profile / 缺 KB / 缺 patches 历史 类似 |
| Token cap | 超 cap → fail-fast；正好达 cap → 通过 |
| 字段集封闭 | 装配输出 dict 字段集严格匹配 §5.1，未声明字段 → 抛错 |
| KB 检索筛选 | scope_domains 命中规则的 unit test |
| dyn_id 生成 | 同 round 多次 → seq 递增；冲突场景 → 抛错 |

### 3.4 P3 Sub-agent runner（mocked LLM）

| 测试组 | 验证内容 |
|---|---|
| 主循环正常退出 | mocked LLM 在第 N 轮 emit_proposal → COMPLETED |
| Turn cap | mocked LLM 持续 tool_call 不 emit → TIMED_OUT(turn_cap) |
| Wall-clock cap | mocked LLM 在轮内 sleep → TIMED_OUT(walclock) |
| Tool 白名单 | read_source 越界 → error；run_bench 未注册 bench_id → error；read_session_artifact 黑名单路径 → error |
| Bench timeout | mocked bench 长睡 → subprocess timeout 强制终止 |
| Proposal validation | 含 expected_gain → reject；scope_domains 越出 spec → reject；patch 不合法 unified diff → reject |
| 解析失败 | mocked LLM 输出非 emit / 非 tool_call 内容 → error；连续 N 次 → FAILED |
| 历史压缩 | journal 长度超阈值 → 触发机械截断；保留首 N + 末 N |

### 3.5 P3 输出 schema

| 测试组 | 验证内容 |
|---|---|
| 字段集 | proposal_set 字段集严格匹配 §5.1；多余字段 → reject |
| provenance 强制 | runner 输出的 provenance 必为字面量 `dynamic` |
| cap=1 | proposal_set.length > 1 → 截断或 reject |

### 3.6 P3 回收路径白名单

| 测试组 | 验证内容 |
|---|---|
| 白名单文件回收 | proposal_set.json + journal.md 正确回收 |
| 黑名单不回收 | sub-agent 在 worktree 内写 scratch/bench/X.json → 不出现在 artifact 目录 |
| Worktree 销毁 | 终态后 worktree 已删除；branch 已删除 |

### 3.7 P4 Critic 分类与规则

| 测试组 | 验证内容 |
|---|---|
| 分类 | provenance=`dynamic` 自动挂 `cross_domain=true`；specialist 不挂 |
| 规则 1 | rationale 漏一个 domain → REVISE / `cross_domain_rationale_incomplete` |
| 规则 2 | rationale 未提耦合点 → REVISE |
| 规则 3 | motivation 退化为 specialist combo → REJECT |
| patch_landing 不弱化 | 任一 patch_landing checklist 不过 → 取严 REJECT |
| specialist 不受影响 | specialist patch 走 critic，3 条规则不触发 |

### 3.8 P6 SharedState

| 测试组 | 验证内容 |
|---|---|
| 状态转移合法性 | 全 11 状态的合法转移路径都能跑通；非法转移（如 KEPT → INTEGRATING）抛错 |
| 写入原子性 | 单次写入 status / last_outcome / updated_at / verdict / cumulative_gain 一致 |
| 字段集封闭 | summary 包含未声明字段 → save fail-fast |
| Prompt 注入 | 5 条 / 6 条 / 0 条 时段落体积与内容符合 §7.2 |

### 3.9 P8 重启清理

| 测试组 | 验证内容 |
|---|---|
| 非终态全转 ABANDONED | 4 个非终态各 1 例 |
| 终态 no-op | 7 个终态各 1 例 |
| Corner: 仅 artifact 无 summary | 重建 summary 标 ABANDONED |
| Corner: 仅 summary 无 artifact | 标 ABANDONED + artifact_missing=true |
| Worktree 清理 | 正常清理 / 失败 log warning |
| dispatch_history.jsonl | abandoned 记录正确追加 |

---

## 4. 集成层

### 4.1 端到端 happy path

| 测试 | 步骤 | 验证 |
|---|---|---|
| `dynamic_e2e_happy` | dispatch → mocked runner emit 合法 patch → critic APPROVE → integrate apply → grid → KEEP | summary KEPT；optimization_stack 含 dynamic patch；ledger 有 source=dynamic 记录 |
| `dynamic_e2e_revert` | dispatch → ... → grid 测出 gain 不达标 → REVERT | summary REVERTED；optimization_stack 不含 dynamic patch；baseline 状态恢复 |
| `dynamic_e2e_accuracy_fail` | dispatch → ... → integrate apply → accuracy gate fail | summary REVERTED；patch 已 revert |
| `dynamic_e2e_critic_reject` | dispatch → ... → critic REJECT | summary CRITIC_REJECTED；不进 integrate |
| `dynamic_e2e_empty` | dispatch → mocked runner COMPLETED_EMPTY | summary COMPLETED_EMPTY；不进 critic |
| `dynamic_e2e_timed_out` | dispatch → mocked runner sleep → TIMED_OUT | summary TIMED_OUT |

### 4.2 共存与并发

| 测试 | 验证 |
|---|---|
| `dynamic_specialist_coexist` | 同 round 派 1 dynamic + 2 specialist；三者按 FIFO 派单；research_lane 利用率正常；最终 status 各自正确 |
| `dynamic_round_cap` | 同 round 派 2 dynamic（第二条应被拒） |
| `dynamic_lane_full` | research_lane 已满时派 dynamic → 等待；不影响其他 dispatch |
| `dynamic_round_advance` | round-1 派 1 dynamic（abandoned），round-2 仍可派 1 dynamic |

### 4.3 重启场景

| 测试 | 验证 |
|---|---|
| `dynamic_resume_subagent_running` | 模拟 sub-agent 跑到一半重启 → 转 ABANDONED；worktree / branch 清理 |
| `dynamic_resume_awaiting_critic` | 重启时 critic 未来得及审 → 转 ABANDONED |
| `dynamic_resume_integrating` | 重启时 patch 已 apply 但 grid 未跑完 → 转 ABANDONED；既有 patch 状态校验机制兜底 |
| `dynamic_resume_terminal_noop` | 重启时已是 KEPT/REVERTED → no-op |
| `dynamic_resume_then_redispatch` | 重启 abandoned 后 LLM 派同 motivation 新 dyn_id → 通过 |

---

## 5. 不变量层（§1.2 红线机械证据，最关键）

不变量测试是本阶段的**核心产出**——这层用例的存在让红线变成"任何
未来 refactor 不会悄悄破坏"的机械保障，而非依赖人审 PR。

### 5.1 红线列表与对应测试

#### 不变量 I-1：micro-bench 产出永不进 promote 链路

| 测试 | 构造 | 期望 |
|---|---|---|
| `inv_microbench_not_in_proposal` | mocked sub-agent 在输出中加 `expected_gain=2.5` 字段 | runner reject；最终落盘的 proposal_set.json 不含此字段 |
| `inv_microbench_not_in_sharedstate` | mocked sub-agent 在 worktree 内 scratch/bench/ 写带数字的输出 | KEPT 后扫 SharedState 全字段，不出现 bench 数字（包括 cumulative_gain 必须来自 grid runner，不来自 bench） |
| `inv_microbench_not_in_ledger` | 同上 | record_intervention 中无 bench 数字 |
| `inv_worktree_destroyed_with_bench_outputs` | sub-agent 跑 bench 后终止 | worktree（含 scratch/bench/）已物理销毁 |

#### 不变量 I-2：SharedState 受保护字段不可被 LLM 修改

| 测试 | 构造 | 期望 |
|---|---|---|
| `inv_dynamic_actions_write_protected_top` | LLM emit `UPDATE_STATE{dynamic_actions: ...}` | PolicyGate 拒，code `core_state_write_violation` |
| `inv_dynamic_actions_write_protected_inner` | LLM emit `UPDATE_STATE{dynamic_actions[dyn-3-1].cumulative_gain: 9.9}` | 拒 |
| `inv_dynamic_actions_no_create_via_llm` | LLM 试图通过 UPDATE_STATE 添加新 dyn_id | 拒 |
| `inv_dynamic_actions_no_delete_via_llm` | LLM 试图删除 dyn_id | 拒 |
| `inv_optimization_stack_unchanged` | 同 #1，附带验证 | optimization_stack / current_best / gaps 同样不可写 |

#### 不变量 I-3：Provenance 字面量为 `dynamic`

| 测试 | 构造 | 期望 |
|---|---|---|
| `inv_provenance_literal_runner` | mocked sub-agent 输出 `provenance="specialist:foo"` 的 dynamic patch | runner reject，proposal 不写盘 |
| `inv_provenance_literal_critic` | （越过 runner）伪造 provenance 进入 critic | critic 入口校验拒，alert |
| `inv_provenance_literal_grid` | （越过 critic）伪造 provenance 进入 grid | IR-4 校验拒 |
| `inv_provenance_no_compound` | 输出 `provenance="dynamic:foo+bar"` 复合形式 | 任一层拒 |

#### 不变量 I-4：dynamic action 不能动 kernel-owned actions

| 测试 | 构造 | 期望 |
|---|---|---|
| `inv_kernel_owned_dispatch` | dispatch 时 side_effects_declared 含 kernel_owned | P1 PolicyGate 拒 |
| `inv_kernel_only_scope` | scope_domains 全为 kernel | P1 拒 |
| `inv_dynamic_patch_not_modifying_kernel` | mocked sub-agent 输出实际改动 kernel 文件的 patch | critic patch_landing 拦截或 P3 输出 schema 拦截（取决于实施层） |

#### 不变量 I-5：dynamic action 不能起独立 server / 跑 Magpie

| 测试 | 构造 | 期望 |
|---|---|---|
| `inv_no_server_in_side_effects` | side_effects_declared 含 server | P1 拒 |
| `inv_no_magpie_in_bench_registry` | 验证 bench 注册表中不含任何 server / Magpie 类调用 | 静态校验 |

#### 不变量 I-6：dynamic action 不能声明自己的 metric

| 测试 | 构造 | 期望 |
|---|---|---|
| `inv_no_metric_in_side_effects` | side_effects_declared 含 metric / accuracy_gate | P1 拒 |
| `inv_no_quantitative_claim_in_proposal` | mocked sub-agent 在 expected_qualitative_argument 中含 `+5%` 字样 | runner reject |
| `inv_keep_decision_from_grid` | KEPT 后验证 cumulative_gain 来源 = grid runner 测得值 | 静态可追溯 |

#### 不变量 I-7：dynamic patch 必经 integrate_patch

| 测试 | 构造 | 期望 |
|---|---|---|
| `inv_no_direct_source_write` | mocked sub-agent 在 worktree 内 git commit 改动 framework_source 后退出 | runner 不回收此 commit；最终 framework_source 状态未变；后续走正规 critic + integrate_patch 路径 |
| `inv_optimization_stack_only_via_promote` | 验证 optimization_stack 中所有 dynamic 来源条目的 patch 都有对应 ledger record（来自 _promote_to_shared_state） | 100% 命中 |

#### 不变量 I-8：dynamic action 间不互相学习

| 测试 | 构造 | 期望 |
|---|---|---|
| `inv_dyn_cannot_read_other_dyn` | sub-agent 调 read_session_artifact 读其他 dyn_id 路径 | error |
| `inv_seed_kit_no_other_dyn` | seed kit 装配输出中绝不包含其他 dyn_id 的内容 | 静态校验 |

### 5.2 不变量测试的中心思想

- **每条不变量是独立用例**——失败时能精准定位被破坏的红线；
- **跨多层验证**——同一不变量在 dispatch / runner / critic / grid /
  promote 多层都有断言（多层防御）；
- **不依赖业务代码的"善意"**——构造的输入是恶意的（伪造 provenance、
  塞数字字段、写 server 类 side effect 等），用以模拟未来某次 refactor
  无意中开了口子；
- **静态可追溯**：尽量让断言基于"能从 artifact / SharedState / ledger
  反推"的信息，而非内部状态——这条让用例对实现重构鲁棒。

---

## 6. Mocked sub-agent fixture

集成层与不变量层依赖一组 mocked sub-agent，绕过真实 LLM 调用。

### 6.1 必备 mocks

- **`MockRunnerSuccess`**：固定输出一个合法 patch 与完整
  cross_domain_rationale；
- **`MockRunnerEmpty`**：emit_proposal 但 patch 为空（COMPLETED_EMPTY）；
- **`MockRunnerTimeoutTurn`**：持续 tool_call 直到 turn cap；
- **`MockRunnerTimeoutWalclock`**：在某轮 sleep 超过 wall-clock 上限；
- **`MockRunnerCrash`**：sub-process 中途崩溃；
- **`MockRunnerInvalidProposal`**：emit 不合法字段（数字 / scope 越
  界 / provenance 错） — 用于不变量测试。

### 6.2 fixture 设计原则

- **不调用真 LLM**——CI 中可重现；
- **不依赖网络**——独立运行；
- **每个 mock 是状态机**——按 turn 顺序产出预设输出；
- **统一接口**——同 P3 runner 的 invoke_claude 接口签名，便于替换。

---

## 7. 既有 specialist 回归

P1–P8 实施期间，既有 specialist 路径**不能被破坏**。回归用例包括：

- specialist dispatch / runner / critic / integrate_patch / grid /
  promote 全套既有用例继续通过；
- specialist 与 dynamic 同 round 共存场景（已含在集成层 §4.2）；
- specialist provenance 走 IR-4 校验仍正常通过；
- Critic 在非 dynamic patch 上 cross_domain flag 不挂。

---

## 8. CI 集成与门槛

### 8.1 入门级（每次 PR 必跑）

- 单元层 §3 全部用例；
- 集成层 §4.1 happy path 与 §4.1 主要失败路径（约 6 条）；
- 不变量层 I-1 / I-2 / I-3 / I-7（最核心 4 条）。

### 8.2 完整级（合并到 main / 每日运行）

- 上述全部 + 集成层 §4.2 / §4.3 + 不变量层全部 + specialist 回归。

### 8.3 失败处理原则

- 任何不变量层用例失败 → **block merge**（红线被破坏，必须立刻修）；
- 单元层失败 → block merge；
- 集成层失败 → 视情况处理（mocked sub-agent 引入的偶发性问题可重试，
  但累计 ≥3 次 → block）。

---

## 9. 依赖与前置条件

P9 与 P1–P8 同步推进：

- 每完成一个 PX，对应单元层用例同步入库；
- P5 完成后开始集成层补全；
- P5 完成后开始不变量层补全（不变量层需要端到端通路存在才能验证）。

---

## 10. 验收信号

| # | 测试场景 | 期望 |
|---|---|---|
| 1 | P9 用例总数 | 单元 ≥80 / 集成 ≥15 / 不变量 ≥25，合计 ≥120 |
| 2 | CI 入门级跑通 | 全部通过；耗时 ≤15 分钟 |
| 3 | CI 完整级跑通 | 全部通过；耗时 ≤45 分钟 |
| 4 | 故意破坏红线 | 任一不变量用例 fail，能精准定位红线 |
| 5 | specialist 回归 | 既有 specialist 用例 100% 通过 |
| 6 | mocked sub-agent fixture | 独立可调用；替换真 runner 时端到端跑通 |

---

## 11. DEFAULT / 待 review

| # | 条目 | DEFAULT | 备注 |
|---|---|---|---|
| 1 | 用例命名前缀 | `dynamic_*` / `inv_*` | 便于过滤 |
| 2 | 不变量测试目录是否独立 | 独立（如 `tests/invariants/dynamic_action/`） | 便于 CI 单独跑 |
| 3 | mocked sub-agent 实现位置 | `tests/fixtures/dynamic_action/` | 与产线代码隔离 |
| 4 | 是否包含 fuzz / property-based 测试 | 否（v1 暂不） | 静态用例已覆盖核心 |
| 5 | 是否覆盖性能回归测试 | 否（v1 暂不） | dynamic action 不引入性能敏感路径 |

---

## 12. 与 §1.2 / §3.10 的对应关系

| 设计哲学 | 在 P9 的落点 |
|---|---|
| §1.2 全部红线 | §5 不变量层每条对应一类红线；I-1 至 I-8 完整覆盖 |
| §3.10 "测试矩阵" | §3 单元 + §4 集成 + §5 不变量 三层金字塔 |
| §3.11 "任何阶段触发设计变更流程而非局部打补丁" | §5.2 不变量测试是"是否触发设计变更流程"的检测器——红线被破坏 → 测试失败 → 必须走变更流程 |

P9 是 dynamic action v1 的**自我保护机制**——红线、状态机、字段集都
被映射到可执行的检测点。任何未来对 dynamic action 的修改，都必须在
P9 测试集合面前接受检验。
