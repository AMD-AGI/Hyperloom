# 新增 PR (#288 roofline-v2 + #280 framework-agent) 评估报告

> Branch: feature/zhenggong/lossen_explore (51 commits ahead of origin/main, 134 commits behind)
> Fork point: e7c2bdb (PR #284 — Robustness Agent Update + Closing-Phase / Recover Enablement)
> Both PRs already merged into origin/main — they are part of the 134 commits we are currently behind.

## 0. 关键事实速览 (TL;DR)

| 项 | 数值 / 结论 |
|---|---|
| PR #288 base | main (xiaofei/roofline-v2 → main, merged 2026-05-23) |
| PR #288 体量 | +19,496 / -1,009, 100 文件 |
| PR #280 base | main (framework-agent-pr → main, merged 2026-05-23) |
| PR #280 体量 | +11,446 / -24, 76 文件 (其中绝大多数是新建的 framework-agent/ 子目录) |
| 与本 branch 的 fork 距离 | 134 commits (含 #288 + #280 + 多个其它合入) |
| 直接 git merge origin/main 冲突文件 | ~14 个 (见 §2) |
| 真正的"语义死结" | v0.6 vs v0.8 action-graph 分裂 (见 §2.2) |
| 直接 cherry-pick 可行性 | 否 (见 §2.3) |


---

## 1. 当前 branch 已经有的、与这两个 PR 重叠的能力

在评估冲突前,需要先看清当前 branch 已经走到了哪里。本 branch (`feature/zhenggong/lossen_explore`,
后续称 v0.8) 距离 fork 点 e7c2bdb 已经累积了 51 个 commit,关键的"已经做过的事"包括:

### 1.1 v0.6 → v0.8 action-graph 重写 (commit 76389d8 / a86bdf2)
- **物理删除**: `action_executors/{backends,params,validate_stack}.py`,`scoring.py`,
  `actions/_meta/{backends,params,validate_stack}.yaml`,`actions/validate_stack.md`,
  以及 6 个相关测试文件。
- 这三个 v0.6 的"action 入口"被合并进 v0.8 的 **`explore`** 单一 action(grid runner)。
- PolicyGate 用 `rule='action_deprecated'` 在 intent 边界拒掉旧名字,LLM 会收到结构化
  forward-pointer 错误,而不是 `phase_incompatible`。
- 这是 KB_design §3.4 / §3.13 M3 §PR7 + KB_gaps/Gap-10 落地的最终形态。

### 1.2 5-phase 状态机 (KB_design §3.2, M2)
- 引入 `phase_state.py`(1379 行,本 branch 独有),严格单调链:
  `PRELUDE → EXPLORE → KERNEL → SWEEP → CLOSE`。
- 每个 phase 有自己的 allowlist + budget(默认 PRELUDE 5% / EXPLORE 60% /
  KERNEL 25% / SWEEP 8% / CLOSE 2%)。
- PolicyGate R1 `phase_incompatible` 在 intent 边界拒绝跨 phase 提议。
- main 上 *没有* 这个文件 —— main 还是 v0.6 的"closing_phase" 单 bool 模型。

### 1.3 Specialist sub-agent (KB_design §3.5, M5 + PR-A2/A6/A9)
- `specialist_runner.py` + `specialist_subprocess.py`(per-task git worktree + claude
  subprocess + `--add-dir` scoping)。
- `SPECIALIST_DOMAINS`: `serving_specialist` / `kernel_switch_specialist` /
  `comm_specialist` / `compiler_specialist` / `system_specialist` /
  `pr_intel_specialist` / `session_steward_specialist` (7 个域,M5 active set 全部启用)。
- IR-4 EXPLORE specialist-first contract:`provenance='specialist:<domain>'` 或冷启动
  `provenance='default_grid'` 才能进 explore grid;PolicyGate 拒绝纯 `llm_direct`。
- IR-7 session_steward_specialist:plateau 时 Coordinator 自动派 steward 决定是否继续。
- 关键命名:本 branch 上 **`framework_specialist`** 已经被 1c28dd6 重命名为
  **`serving_specialist`**;**`kernel_specialist`** 重命名为 **`kernel_switch_specialist`**。

### 1.4 IR-6 EXPLORE force-exit 时间硬门 (e974567)
- 总剩余 wall-clock ≤ `--explore-force-exit-hours-remaining`(默认 3.0h)
  或 EXPLORE 剩余 budget% ≤ `--explore-force-exit-budget-pct`(默认 0.20)→ 强制退出。

### 1.5 Critic / Robustness 都已经 v0.8 化
- Critic: `--critic-agent` 默认,带 KB priors / per-session memory / `review_constraints`。
- Robustness: `--robustness-agent` 默认,带 RCA / scheduling-police。
- 两者在 origin/main 也都有,但具体 IO 协议差异较大。


---

## 2. 问题 1:这两个 PR 与本 branch 的冲突大吗?

**答案:文件级冲突约 14 个,可以机械修;真正棘手的是语义层冲突,会让 PR #288 的核心机制失效。**

### 2.1 文件级冲突清单(直接 `git merge origin/main` 试出)

```
CONFLICT (content):  inference_optimizer/cli.py
CONFLICT (content):  inference_optimizer/orchestrator/action_executors/__init__.py
CONFLICT (content):  inference_optimizer/orchestrator/action_executors/report.py
CONFLICT (content):  inference_optimizer/orchestrator/coordinator.py
CONFLICT (content):  inference_optimizer/orchestrator/shared_state.py
CONFLICT (content):  inference_optimizer/orchestrator/sub_agent_runner.py
CONFLICT (content):  inference_optimizer/orchestrator/system_prompts/critic.md
CONFLICT (content):  inference_optimizer/orchestrator/system_prompts/orchestration.md
CONFLICT (content):  inference_optimizer/orchestrator/system_prompts/prompt_builder.py
CONFLICT (content):  inference_optimizer/paths.py
CONFLICT (content):  inference_optimizer/session_paths.py
CONFLICT (content):  inference_optimizer/tests/test_p0_1_protocol.py
CONFLICT (content):  inference_optimizer/tests/test_p1_2_full_action_catalogue.py
CONFLICT (content):  inference_optimizer/tests/test_p5_decision.py
CONFLICT (content):  inference_optimizer/tests/test_prompt_assets.py
CONFLICT (content):  inference_optimizer/tests/test_required_step_gates.py

CONFLICT (modify/delete):  inference_optimizer/orchestrator/action_executors/backends.py        # HEAD: deleted, main: modified
CONFLICT (modify/delete):  inference_optimizer/orchestrator/action_executors/params.py          # HEAD: deleted, main: modified
CONFLICT (modify/delete):  inference_optimizer/orchestrator/action_executors/validate_stack.py  # HEAD: deleted, main: modified
CONFLICT (modify/delete):  inference_optimizer/orchestrator/scoring.py                          # HEAD: deleted, main: modified
CONFLICT (modify/delete):  inference_optimizer/tests/test_p3_search_space_expansion.py
CONFLICT (modify/delete):  inference_optimizer/tests/test_validate_stack.py
CONFLICT (modify/delete):  inference_optimizer/tests/test_validate_stack_gate_skip.py
```


### 2.2 真正的语义冲突(这才是核心)

#### A. v0.6 vs v0.8 action graph 完全错位

PR #288 的所有 LLM-side 决策规则都是建立在 v0.6 的 **`baseline → roofline → backends/params/comm_optimization → roofline → kernel_opt`** 这条 action 流水线上。
而本 branch 已经把 `backends` / `params` / `validate_stack` 从 action 表里物理删除,用一个 `explore` 复合 action 替代。

具体冲突点(读 `git show origin/main:inference_optimizer/orchestrator/system_prompts/orchestration.md`):

- main 的 `### Roofline-v2 action ordering (HARD RULES — PolicyGate enforced)` 整段 prompt 写的是
  "`params` (CUDA graph / torch_compile / decode steps / etc.)" + "`backends` (attention backend / sampling / MoE a2a / etc.)" + "PolicyGate rejects `request{kind="run_optimization"}` when `snapshot_id < 2` OR `backends_attempts < 1` OR `params_attempts < 1`"。
- 本 branch 的 EXPLORE 只有 `explore`、`specialist`、`integrate_patch`、`assess_remaining_gaps`、`profile`、`pmc_roofline`、`recover` 七个 action。`backends_attempts` / `params_attempts` 字段在 SharedState 里仍然在(Inv-10.1 v0.6 resume 兼容),但已经不再有任何写入路径。
- 后果:PR #288 的 sequence_denial 规则 `backends_attempts < 1` 会**永远成立**,把 kernel_opt 永远锁住。

#### B. coordinator.py 的 _is_promotable_result / _promote_to_shared_state 路径

本 branch 在 76389d8 把 `_promote_to_shared_state` 里 280 行的 `("backends","params","sweep")` 块裁成了 25 行的 sweep-only 分支。
PR #288 的 `coordinator.py` (+1179, -82) 在 v0.6 形态上重写了大量 backends/params 路径(record_partial, next_pending_keep_kernel_id, untried_hot_reusable_kernels gate 等)。
这些都需要重新映射到 `explore` 的 grid_runner 路径,不能机械三方合并。

#### C. action_executors 模块结构

- main 上 `action_executors/{backends,params,validate_stack}.py` 还在,而且 PR #288 还小幅修改了这三个文件(`+59` / `+57` / `+38` 行)。
- 本 branch 已经把它们删了。
- 如果直接 `git merge origin/main`,git 给出 `modify/delete` 冲突,保留三方都不正确(留 main 的版本 → `__init__.py` 找不到符号;删 → PR #288 的 `roofline.py` 引用三个文件中的常量也会断)。

#### D. PR-A9 explore_requires_specialist_provenance vs PR #288 的 first-explore priority

- 本 branch 的 IR-4(PR-A9):EXPLORE 必须 specialist-first,纯 `llm_direct` 拒。
- main 上 PR #288 + #280 引入了 "**`framework_pr` first-explore priority** rule"(主提示里强制要求 baseline 后的第一次 explore 必须是 `framework_pr`)。
- 如果要把 main 合进来,这两条规则要重新缝合:`framework_pr` 要么作为 specialist 的一种 provenance,要么作为冷启动豁免(类似 `default_grid`)。


### 2.3 直接 cherry-pick 的可行性

**结论:不行。** 两个 PR 的 base commit (`62f87fc`, `c794cdb`) 都是已经经过 main 的多次 merge 整理的合并提交,
其内部 commit 多数依赖于 v0.6 的 `backends.py` / `params.py` / `scoring.py` 这些已被本 branch 删除的文件。
单独 cherry-pick `roofline.py` 或 `framework_pr.py` 是可行的,但它们引用的 SharedState 字段
(`backends_attempts` / `last_backends` 等)在本 branch 已经不再被写入,导致功能在 v0.8 的 EXPLORE 路径里失效。

### 2.4 冲突量定性评估

| 维度 | 评估 |
|---|---|
| 文件级冲突 | **中** (~14 个 content + 7 modify/delete);单独看每个文件都能解 |
| 语义结构冲突 | **大** (整个 action 拓扑不同) |
| 测试覆盖冲突 | **中** (PR #288 的 ~15 个新 test_n*.py 文件需要全部重写到 explore-only 路径) |
| Prompt 冲突 | **大** (orchestration.md 的"Roofline-v2 action ordering"整段需要重写为 explore-friendly) |
| KB / cortex 冲突 | **小** (两边 KB 写入接口都已稳定) |
| 整体合入工作量预估 | **5-8 个工作日**:不是无脑 merge,需要"重新实现 PR #288 的能力,而不是合并它的 diff" |


---

## 3. 问题 2:这两个 PR 能给当前 branch 带来什么新东西?

### 3.1 PR #288 (roofline-v2) 带来的能力

> 标题: "Feature/xiaofei/roofline v2"。100 文件,+19,496/-1,009。

**核心机制 — 把 profile + trace_analyze 原子化成 `roofline` 复合 action。**

| 能力 | 来源 | 价值 |
|---|---|---|
| `roofline` composite action | `action_executors/roofline.py` (499 行) + `actions/roofline.md` | 把"profile → trace_analyze"的两步序列从 LLM 责任移到 executor,LLM 不再可能漏掉第二步 |
| TraceLens `analysis.md` 直注 prompt | `system_prompts/prompt_builder.py` (+434/-45) | LLM 直接读人类可读的 perf 报告,不再需要 LLM 二次解释结构化字典(TraceLens 团队"no second interpretation"硬约束) |
| `roofline_snapshot_id` 单调计数 | `roofline_snapshot.py` (294 行) | 让"快照是否过期"成为可判断的事实,不靠 LLM 估计 |
| 🔴/🟡/🟢 优先级标记驱动 prune_branch | orchestration.md +173 行 | LLM 看到 🔴 = 必须 kernel_opt;🟡 = 可选;🟢 = fusion 机会;不再"乱扎" |
| **N9 — 禁止直接 propose `profile`** | policy.py | 强制走 `roofline`,杜绝重复 profile 浪费 |
| **N19c gain-driven gate** | `test_n19c_gain_driven_gates.py` | `last_cheap_delta_gain < ε` 才解锁 kernel_opt;cheap 还在产生增益就不能跳到 deep |
| **N20 backends_subset / params_subset** | `test_n20_*.py` | LLM 可以提名"只跑 grid 的某一子集",不必每次跑全表 |
| **N22 keyword advisory** | analysis.md 关键词 → variant 建议 | analysis.md 文本被 NLP 提取成"应该试 X 这个 flag" |
| **N25/N26 steady-state chunk auto-retry** | `_AUTO_RETRY_WARNING_CODES` | TraceLens 切片选错 chunk 时,executor 单次自动用 alternate mode 重试 |
| **N27 roofline-failure fallback** | coordinator.py | 连续 2 次 roofline 失败 → 解锁 backends/params 旁路 |
| **N30 cheap-exhausted deep boost** | scoring 调整 | cheap 用尽时主动加 kernel_opt 的优先级 |
| **N31 final roofline & report comparison** | report.py | 报告里展示首尾两个 roofline snapshot 对比,直接看到"瓶颈是不是真的被解决了" |
| **GEAK 90min wrapper alignment** | kernel-agent | GEAK budget 真正生效 |
| **Multi-KEEP integrate queue** | `next_pending_keep_kernel_id` | 一次 batch 多个 KEEP 时不会被慢的 GEAK 阻塞 |
| **failure_count + max_failures retire** | shared_state | 失败一次的 ladder 退役,避免无限重试 |
| **`untried_hot_reusable_kernels` report gate** | coordinator | report 必须等所有 ≥3% 的热 kernel 都尝试过 |
| **`HYPERLOOM_TRUST_GEAK_CORRECTNESS=1` 默认** | coordinator | GEAK 不再动不动 NEEDS_REVIEW |
| **PR-F: prefer KEEP over higher-micro non-KEEP** | ladder selection | (is_keep, micro) 元组排序,真 KEEP 不再被 NEEDS_REVIEW@1.30x 抢走 |
| **task_groups 在 record_trace_analyze 持久化** | shared_state | 同一 AST function 的同组 kernel 不会被重复尝试 |

**简化版本:PR #288 ≈ 把 v0.6 的"LLM 凭意识序列化 profile 与 trace_analyze"换成确定性的 `roofline` action,
并补 N9~N38 一系列 PolicyGate 规则把多个长尾失败模式(GEAK 失活、重复 profile、kernel_opt 死循环、report 早出)关掉。**


### 3.2 PR #280 (framework-agent) 带来的能力

> 标题: "Framework agent pr"。76 文件,+11,446/-24。

**核心机制 — 引入一个独立的 sibling skill `framework-agent/`,作为 sglang/vllm 源码层 PR 探索工具,
并通过 `framework_pr` bandit arm 接入 inference_optimizer。**

| 能力 | 来源 | 价值 |
|---|---|---|
| **`framework-agent/` 独立 skill 包** | `framework-agent/src/framework_agent/` (~3000 行) | 一个独立可运行的 `fa` CLI,提供 `fa candidates / fa explore / fa kb / fa agent` 子命令 |
| **PR 候选发现** | `sources/primus_cortex.py` + `sources/github.py` | 从 AMD 内部 Primus Cortex KB + GitHub 双源拉 PR 候选,带 keyword 过滤 |
| **Per-candidate 隔离运行** | `isolation.py` | 每个候选 PR 在独立 worktree + venv 里 build + bench |
| **3-gate 决策** | `decision.py` | accuracy / throughput / regression 三道闸门 |
| **`framework_pr` bandit arm** | `inference_optimizer/orchestrator/action_executors/framework_pr.py` (535 行) | inference_optimizer 把 fa 的输出当成一个 bandit arm:auto-compose gap → fa candidates → git fetch + checkout → A/B 子 baseline → KEEP 或 rollback |
| **gap auto-composer** | `action_executors/_framework_gap_composer.py` (225 行) | 从 manifest workload + last_profile_kernel_breakdown 自动合成 `(gap, keywords)`;不再需要操作员手敲 `--framework-gap` 字符串(历史上多次因 typo 选错 PR) |
| **KB partition** | `framework-agent/kb/framework_optimization/` (PR-I 占位) | 框架优化经验进 KB,跨 session 复用 |
| **`fa agent` sibling-skill 协议**(PR-D+ 占位) | `runtime/cli.py` `prepare-task / commit-result` | 4 种 envelope (`OptimizeSuccess/Failure`, `IntegrateSuccess/Failure`),为后续把 framework agent 接入 5 角色架构做铺垫 |
| **`--no-framework` toggle** | inference_optimizer/cli.py | 与 `--no-kernel` 对称,运行时关掉 framework arm |
| **`framework_pr` first-explore priority rule** | orchestration.md | baseline 后第一次 explore 强制 `framework_pr`,直到 `runs >= 1` |
| **AST scanner 占位** | `framework-agent/src/.../ast_scanner.py`(PR-E 落地中) | 用 libcst 扫 sglang/vllm 源码,为 patch_proposer 提供 symbol 表 |

**简化版本:PR #280 ≈ "去 GitHub 上找 sglang/vllm 的最新 PR,自动 cherry-pick 进 worktree,A/B benchmark,
KEEP 就留下,DISCARD 就 rollback"。本质上是用 bandit arm 形态把社区上游优化"拉过来用",
不需要 LLM 自己读源码改代码。**

### 3.3 两个 PR 加在一起的协同效应

PR #288 + PR #280 在 main 上是**先合 #280 再合 #288**(`c794cdb` then `c6f0a71`),所以 #288 的
`62f87fc` "Merge origin/main into feature/xiaofei/roofline-v2 (incl PR #280 framework-agent)"
里已经把 framework_pr 整合进 roofline-v2 的 action 序列。具体协同点:

1. **`framework_pr` 在 EXPLORE 第一次跑** → 拉社区 PR → A/B → 拿到当前框架的"最新已知优化"
2. **第一个 `roofline` snapshot** 此时反映的是"上游 PR 已合"后的瓶颈分布,直接奠定后续 deep work 方向
3. **`backends/params` cheap exploration** 在 framework_pr 已收敛的基础上跑(N19c)
4. **`roofline` 第二次** 看 backends/params 是否真的把瓶颈消掉
5. **`kernel_opt`** 用第二个 snapshot 的 Top kernel 去做 deep optimization
6. **`roofline` 第三次** 在收尾阶段(N31)对比首末两次,作为 "整个 session 优化是否真的发生在了瓶颈上"的硬证据


---

## 4. 问题 3:让 Orchestration 根据 roofline 来 dispatch agent —— 正反两面评价

> 想法概述: Orchestration 在 dispatch specialist / explore 之前,先看 roofline snapshot 当前瓶颈分布,
> 对已经"顶到天花板"的方向(例如 compute % > 90% + 当前 kernel 已经达到 peak FLOPs 的 X%)
> 不再继续 dispatch,把 budget 让给真正还有空间的方向。

### 4.1 正面 (PRO) —— 这个想法非常符合 v0.8 的设计哲学

#### Pro-1. 与 IR-4 specialist-first contract 同源同构
本 branch 已经强制要求 EXPLORE 的每个 variant 必须可追溯到一个 specialist。把"roofline-aware dispatch"
做成 "specialist 在生成 proposal 前必须先读 roofline snapshot 才能 emit specialist_done" 是这条契约的
**自然加强**,不是新约束。具体落地点:`SpecialistPromptInputs.gap_evidence` 和
`build_specialist_prompts` 已经把 analysis_md 注入第 2 节,只缺一个明确的"如果该方向已经触顶,你要直接
emit empty specialist_done"指令。

#### Pro-2. 用确定性事实压缩 LLM 决策空间
当前 EXPLORE 的失败模式之一是 LLM 在 plateau 之后还在反复 re-explore 同一个已经无空间的方向(`Version1_report.MD` 第 186 行附近的失败案例)。
"roofline-gated dispatch"等于把这些信息从 LLM 的"自觉"变成 PolicyGate 的"强制",对应:
- 一个新 PolicyGate rule:`dispatch_against_saturated_layer`
- 它消费 `last_trace_analyze.workload_summary.compute_pct` 和 `category_data/<layer>_metrics.json`
- 拒绝 `delegate{action_name='specialist', params.domain='X'}`
  当 X 对应的层已经 ≥ 操作员定义的"饱和线"

#### Pro-3. 把 budget 倾向"还能赚的方向"
Inv-9 (KB_design §3.9) 已经退场了 scoreboard,Pro-3 等于把"动态优先级"重新引入,但不是历史式的成功率,
而是"理论上还剩多少头部空间"。这是**前瞻**信号,胜过 scoreboard 的**回顾**信号。

#### Pro-4. IR-6 force-exit 的更精细版本
IR-6 是粗粒度的"时间到了就强退";roofline-gated 是细粒度的"哪个方向到了就只屏蔽那个方向"。
两者正交叠加,可以让 EXPLORE budget 利用率明显提升。

#### Pro-5. 与 PR #288 的 prune_branch + 🔴/🟡/🟢 标记天然咬合
PR #288 已经把 analysis.md 的 🔴/🟡/🟢 标记暴露给 LLM,`prune_branch` 也已经存在。
roofline-gated dispatch ≈ **把 prune_branch 的部分语义自动化**,从 LLM 责任降到 Coordinator/PolicyGate 责任,
减少"LLM 看到 🔴 + 🟢 但没 prune 🟢"这种漏判。


### 4.2 反面 (CON) —— 风险与反例

#### Con-1. "顶到天花板"是个滑坡概念,容易过早收敛
Roofline 报告的"compute %"是**已发生**的瓶颈,不是**已被优化掉**的瓶颈。
GLM-5/MI355X 的真实案例:第 1 次 roofline 显示 compute 92% saturated,但实际上在 fp8 量化 + MoE expert
重排之后(framework 层动作),compute% 掉到 78%,新的瓶颈出现在 KV 拷贝。如果在第 1 次 snapshot 就
"屏蔽 framework 层 dispatch",你永远到不了第 2 个 snapshot —— 这是个**非凸**的优化空间。

#### Con-2. 跨层级的隐藏耦合
- 一个 kernel 看似"已达 peak FLOPs 的 95%",但其实下游 framework 用了一个错误的 attention backend
  (例如 prefill 在用 paged attention 而不是 flash attention),换了 backend 之后这个 kernel 根本不再
  是 hot kernel。
- "顶到天花板"是相对于**当前的 kernel set**,而 framework 切换会**整体替换 kernel set**。
- 如果在 kernel set 不稳定时就 gate 上层 dispatch,你会丢掉 backends 这条最大的杠杆。

#### Con-3. Roofline 信号本身是有 5–15% 不确定性的
- TraceLens 切片的 chunk 选择会显著影响 compute_pct(N25/N26 就是为了治这个)。
- pmc 计数器有 hardware-level noise(尤其是 MI355X 上 XCD 的资源分配)。
- 用一个 noisy signal 去做硬 gate 会导致**不可恢复的误屏蔽**:specialist 一旦被 PolicyGate 拒,
  这个 round 的 LLM 就无法翻案,只能等下一次 roofline。

#### Con-4. specialist 的价值就在于"挑战 roofline 的判断"
session_steward_specialist / kernel_switch_specialist 等域的设计目的之一是**质疑当前 snapshot**
(典型例子:specialist 通过读源码发现了 `cuda_graph_max_bs` 没开,虽然 roofline 说当前 launch latency
不是瓶颈,但这恰好是 launch latency 不是瓶颈的**原因**)。如果 specialist 的派发本身被 roofline gate 屏蔽,
就阉割了它的探索能力 —— 等于让 specialist 只能"附和" roofline,不能"反驳" roofline。

#### Con-5. 与 Critic 的角色重叠
Critic 在 v0.8 已经接管了"variant 是否合理"的判断(per-variant KB priors 检查)。
roofline-gated dispatch 把"方向是否合理"加在 Critic 之外,容易出现两个 review 层级互相打架:
Coordinator 屏蔽 → LLM 不发 → Critic 没东西 review → KB 信号永远更新不到 "方向 X 实际是死路"
这条事实层。

#### Con-6. 闭门优化 vs 上游同步矛盾
PR #280 framework_pr 的核心价值是"拉上游最新 PR";很多上游 PR 修复的恰恰是"看似已饱和"的方向
(例如 fmha_v3 的精度修复)。如果 roofline-gated dispatch 屏蔽了 attention 方向,
framework_pr 也无法跟进社区修复 —— 失去与社区版本同步的能力。


### 4.3 推荐落地形态(平衡 PRO/CON)

不要做成"硬 gate",做成"**软 advisory + 硬 retry budget**":

1. **软 advisory(对 LLM 提示,非 PolicyGate 拒绝)**
   - 在 specialist prompt 第 2 节、orchestration `=== Phase ===` block、以及 explore grid materialize 时
     注入 `saturated_layers=[X, Y]` 字段。
   - LLM 看到一个方向已饱和,**仍然可以 dispatch**,但需要在 `notes` 里说明为什么(挑战的依据)。
   - 这种"必须说理由"的软压力,比硬 gate 健康得多。

2. **硬 retry budget(配额式压制,而非完全屏蔽)**
   - 每个饱和方向给一个递减的尝试 budget:第一次允许全速 dispatch,第二次降到 50%,
     第三次降到 20%,第四次起 PolicyGate 拒。
   - 数值随每次新 roofline snapshot 重新评估 —— 一旦 snapshot 显示该方向饱和度
     < 阈值 - margin,budget 立即恢复。

3. **per-domain 配置**
   - 同一个"饱和"对不同 specialist 域意义不同。`compiler_specialist` 关心 register pressure,
     `system_specialist` 关心 launch latency,`comm_specialist` 关心 exposed_comm。
     需要每个 domain 各自的 `saturation_metric` 而不是统一的 compute_pct。

4. **落地路径建议**
   - 不要单独立项,**搭车 PR #288 的 N22(keyword advisory)**:N22 已经把 analysis.md 的关键词
     映射到 variant 建议,只需扩展为"关键词 → 饱和方向 → 建议方向"的双向映射。
   - 实现位置:`_analysis_keyword_map.py`(PR #288 已经引入),加一个 `saturation_advisory()` 函数。

5. **可观测性**
   - SharedState 加 `roofline_saturation_history: list[{snapshot_id, layer, saturation_pct}]`
   - report.py 渲染"哪些方向被屏蔽过、屏蔽多久、实际 KEEP 是否真的没在该方向出现"
   - 让我们能事后判断 saturation gate 是否真的提升了 budget 利用率,还是只是过度收敛


---

## 5. 问题 4:framework agent 应该落到哪个 phase?

### 5.1 候选 phase 评估

| Phase | 是否合适 | 理由 |
|---|---|---|
| PRELUDE | ❌ 否 | PRELUDE 只允许 `target_analysis` / `baseline` / `recover`。framework_pr 需要 `baseline_tput > 0` 才能 A/B,本身依赖 PRELUDE 的产出 |
| **EXPLORE** | ✅ **首选** | 与 specialist 同源,都是"探索性获取上游/源码层 IO"。PR #288 + #280 在 main 上就是放在 EXPLORE 里 |
| KERNEL | ❌ 否 | KERNEL 是 deep 阶段,允许 actions 强烈限定为 5 个 KERNEL_OWNED + profile/pmc_roofline。framework 改框架代码会扰动 kernel set,与 KERNEL 阶段"用稳定 kernel set 做 deep work"的语义冲突 |
| SWEEP | ❌ 否 | SWEEP 阶段是 workload grid 验证,不应有源码层动作 |
| CLOSE | ❌ 否 | 终结阶段 |

### 5.2 EXPLORE 内部应该作为什么形态?

有 3 种形态可选:

#### 形态 A — bandit arm(main 上的当前形态)
- `framework_pr` 是一个独立 action,与 `explore` / `specialist` 平级
- pros: 简单直接,与 main 一致
- cons: 与本 branch IR-4 specialist-first contract 冲突 —— `explore` 必须 specialist-derived,
  那 `framework_pr` 是 specialist 还是不是?如果不是,IR-4 就要开特例口子
- IR-4 妥协:把 `framework_pr` 加到 `provenance` 白名单里,允许 `provenance='framework_pr'` 类似
  `provenance='default_grid'` 的冷启动豁免

#### 形态 B — `framework_specialist` domain(本 branch 命名)
- 在 `SPECIALIST_DOMAINS` 里加一个第 8 个 domain `framework_pr_specialist`(或者把现有的
  `serving_specialist` 增强成会调用 fa)
- `serving_specialist` 已经在管 sglang/vllm 源码层,正好与 `framework_pr` 是同一个层
- pros: 完全融入 v0.8 specialist 框架,IR-4 自动满足,KB 路径自动统一,worktree 隔离自动生效
- cons: specialist 子进程模型(claude subprocess + worktree)和 fa 的 PR 候选 + git fetch 模型
  在执行机制上不同,需要桥接层

#### 形态 C — 混合(推荐)
- 把 fa 的 **PR 候选发现**(`fa candidates`)留在 inference_optimizer 一侧作为 specialist 的 KB priors
  数据源
- 把 **patch 的实际 apply**(`git fetch + checkout` 或 cherry-pick PR diff)交给 v0.8 已有的
  `integrate_patch` action(它本来就是干这个的:接收 specialist worktree 里产出的 patches,
  锁 serving lane,apply,restart,benchmark gate)
- specialist 写 `specialist_done.json` + `worktree/patches/<NN>_pr<id>.patch` 时,patch 来源标注
  `provenance='framework_pr'`,这样既走 specialist-first contract,又走 integrate_patch 的统一闸门

### 5.3 推荐落地

**EXPLORE phase + 形态 C(把 fa 当成 serving_specialist 的工具,patch 走 integrate_patch)**

实现要点:
1. SPECIALIST_DOMAINS 不动(`serving_specialist` 已经覆盖 sglang/vllm)
2. `serving_specialist` 的 default_tools 里加上一个内部 wrapper 调 `fa candidates`(只读,不需要写权限)
3. 让 specialist 把候选 PR 的 diff 写成 `worktree/patches/<NN>_pr<id>_<sha>.patch`
4. `integrate_patch` 已有的 KEEP/REVERT 逻辑天然支持
5. 副作用:不再需要单独的 `framework_pr` action,EXPLORE allowlist 不变,IR-4 自动满足

**Phase budget 考虑**:framework_pr 平均 12-20 分钟一次(YAML cost_minutes_p50=12 / p75=20),
EXPLORE 默认 budget = 60% * total。在 6h max-hours 的 session 里 EXPLORE = 3.6h,
够跑 10-18 次 framework_pr 候选(实际不会跑这么多 —— first-explore 跑 1 次,plateau 时再跑 1-2 次)。
所以 budget 影响很小。


---

## 6. 问题 5:roofline 应该落到哪个 phase?

### 6.1 候选 phase 评估

| Phase | 是否合适 | 理由 |
|---|---|---|
| PRELUDE | ❌ 否 | PRELUDE 只产 baseline_tput,roofline 需要 baseline 完成才能 profile,在 PRELUDE 内部嵌 roofline 把 PRELUDE 复杂化,违反"PRELUDE 只做 baseline"的简洁性 |
| **EXPLORE 入口必跑 1 次** | ✅ **首选** | EXPLORE 的所有 specialist / explore 都需要瓶颈信号才能合理派发 —— 没有第一次 roofline 等于盲跑 |
| **EXPLORE 内部按需跑** | ✅ **协同** | 当 cumulative_gain ≥ 3% 触发 / KEEP 后 kernel 分布漂移 / plateau 触发时,需要重 profile |
| **KERNEL 入口必跑 1 次** | ✅ **首选** | KERNEL 是 deep 阶段,必须看 EXPLORE 之后的 post-cheap kernel 分布,否则 deep 工作目标错(PR #288 §4 排序硬规则就是 "kernel_opt requires snapshot ≥ 2") |
| KERNEL 内部按需跑 | ⚠️ 谨慎 | KERNEL 阶段每次 roofline 8-15 分钟很贵;最多在 KEEP 大量 integrate 后跑一次确认 |
| SWEEP | ❌ 否 | SWEEP 是 workload grid,不需要 perf 报告 |
| CLOSE | ✅ **闭环用** | N31 final roofline:在 closing_phase 之前跑一次,与 baseline 时的 roofline 对比,作为"瓶颈是否真的被解决"的硬证据。如果 closing_phase 内部时间不够,可以略 |

### 6.2 推荐 roofline 时机表(全 phase)

```
                  ┌─────────────┐
PRELUDE        →  │ baseline    │  (no roofline)
                  └─────┬───────┘
                        ↓
                  ┌─────────────┐
EXPLORE entry  →  │ roofline #1 │  ← MANDATORY (snapshot_id=1)
                  └─────┬───────┘
                        ↓                ┌──────────────────────────────┐
                  ┌─────────────┐ ┌──→  │ specialist + explore +        │
EXPLORE loop   →  │  cheap loop ├─┤     │  framework_pr (form C)        │
                  └─────┬───────┘ └──→  │ → integrate_patch / KEEP      │
                        ↓                └──┬───────────────────────────┘
                        │                   ↓
                        │              cumulative_gain Δ ≥ 3% ?
                        │                   ↓ yes
                  ┌─────────────┐
                  │ roofline #N │  ← N9c gain-driven gate
                  └─────┬───────┘
                        ↓
                  ┌─────────────┐
KERNEL entry   →  │ roofline #M │  ← MANDATORY (PR#288 ordering rule)
                  └─────┬───────┘
                        ↓
                  ┌─────────────┐
KERNEL loop    →  │ kernel_opt  │  ← uses snapshot_id=M
                  └─────┬───────┘   only refresh if ≥3 reverts in a row
                        ↓
                  ┌─────────────┐
SWEEP          →  │   sweep     │  (no roofline)
                  └─────┬───────┘
                        ↓
                  ┌─────────────┐
CLOSE pre-rep  →  │ roofline #F │  ← N31 final comparison (skipable when budget tight)
                  └─────┬───────┘
                        ↓
                  ┌─────────────┐
CLOSE          →  │   report    │  renders before/after roofline diff
                  └─────────────┘
```

### 6.3 与本 branch 现状的差距

| 现状 (本 branch) | 目标 |
|---|---|
| EXPLORE allowlist 已经手 patch 了 `profile` / `pmc_roofline` 进去 (phase_state.py:130) | 应该改成 `roofline` 复合 action,把 profile + trace_analyze 原子化 |
| KERNEL allowlist 已含 `profile` / `pmc_roofline`,文档说"once per phase entry" | 改成 `roofline`,KERNEL entry 必跑 1 次 |
| 没有 `roofline_snapshot_id` 字段 | 引入后,sequence_denial / N9c / N19c / N31 才能成立 |
| 没有 N31 final roofline | CLOSE phase 加一个 "close_phase_pre_report_roofline" 钩子 |


---

## 7. 补充信息(我觉得对决策有帮助的)

### 7.1 main 与本 branch 的"哲学差异"小结

| 维度 | main (含 PR #288 + #280) | 本 branch (v0.8 lossen_explore) |
|---|---|---|
| Action 拓扑 | v0.6 多 action(backends/params/validate_stack/sweep/...) | v0.8 单一 `explore` + `specialist`(LLM 子代理) |
| LLM 责任 | 高(LLM 自己排序 profile → trace_analyze → backends → ...) | 低(executor 原子化复合,LLM 只挑方向) |
| Phase 结构 | 单 bool `closing_phase` | 5 阶段严格单调 |
| 子代理形态 | bandit arm | LLM-driven specialist subprocess |
| Provenance gate | 无 | IR-4(specialist-first) |
| 分支治理 | 由 LLM `prune_branch` 显式控制 | 由 PolicyGate / phase_allowlist 隐式控制 |
| 上游 PR 接入 | `framework_pr` bandit arm | (尚未接入) |
| Roofline | 一等公民复合 action | 仍是 `profile` + `pmc_roofline` 两个原子 action(未原子化) |

**结论:本 branch 的 specialist + phase 框架更"现代",但缺了 main 上的 roofline 一等公民和 framework_pr 接入这两块"成熟工业能力"。**
最优路径不是"用 main 覆盖本 branch"或"用本 branch 覆盖 main",而是
**把 main 的两个能力按 v0.8 形态重新组织**(见 §5.3 / §6.2)。

### 7.2 合入路线图建议(分 4 步)

> **强烈不建议**直接 `git merge origin/main`,会陷入"修冲突→修测试→修 prompt→再修测试"的两周泥潭。
> 推荐"按 PR 重新实现"的渐进路线。

**Step 1 (1-2 天) — 引入 `roofline` 复合 action(轻量版)**
- 仅 cherry-pick `action_executors/roofline.py` + `actions/roofline.md` + `actions/_meta/roofline.yaml`
- 在 phase_state.py 的 EXPLORE / KERNEL allowlist 加 `roofline`,删 `profile` / `pmc_roofline`
  (这两个变成 roofline 内部步骤,不再对 LLM 暴露)
- 引入 `roofline_snapshot_id` 字段到 SharedState
- 不引入 N9 / N19c / N20 / N22 / N25-31 这些规则(单独后续做)

**Step 2 (2-3 天) — 引入 framework_pr 作为 serving_specialist 的工具**
- cherry-pick `framework-agent/` 整个子目录(几乎全是新增,无冲突)
- 不要 cherry-pick `action_executors/framework_pr.py` —— 不需要单独 action
- 改 `serving_specialist` 的 default_tools,加 `Bash` allowed pattern `fa candidates ...` / `git fetch ...`
- specialist 产出的 PR diff 走现有 `integrate_patch` 路径
- 撤掉 PR #280 引入的 `framework_pr first-explore priority rule`(与 IR-4 冲突)

**Step 3 (1-2 天) — 引入关键 PolicyGate 规则**
- N9 ban direct `profile`(本 branch EXPLORE 的 profile/pmc_roofline 入口已经临时开放,N9 让它再次关上)
- N19c gain-driven kernel_opt unlock
- N31 closing-phase final roofline
- 不要 N20 backends_subset(本 branch 没有 backends)/ N22(本 branch 已经有 specialist KB priors,等价)

**Step 4 (1 天) — Roofline-aware dispatch (本 doc §4 的想法)**
- 实现"软 advisory"形态(不实现硬 gate)
- 在 specialist prompt + orchestration prompt + explore grid materialize 三处注入
  `saturated_layers` 提示
- 加 `roofline_saturation_history` 可观测字段

**总工作量**: ~6-8 个工作日,远低于直接 merge 的 ~10-14 天

### 7.3 风险点与回滚开关

每一步都加 env 开关,允许快速回滚:

| Step | 开关 | 默认 | 用途 |
|---|---|---|---|
| 1 | `INFERENCE_OPTIMIZER_USE_ROOFLINE_COMPOSITE` | `1` | `0` 则保持 profile + pmc_roofline 双 action |
| 2 | `INFERENCE_OPTIMIZER_FRAMEWORK_PR_ENABLED` | `1` | `0` 则 specialist 不调用 fa |
| 3a | `INFERENCE_OPTIMIZER_DENY_DIRECT_PROFILE` | `1` | `0` 则恢复 profile 直接 propose |
| 3b | `INFERENCE_OPTIMIZER_GAIN_DRIVEN_KERNEL_OPT` | `1` | `0` 则不查 cumulative_gain |
| 4 | `INFERENCE_OPTIMIZER_ROOFLINE_SATURATION_ADVISORY` | `0` | `1` 则启用饱和度提示(默认关,先观察一段时间再开) |


### 7.4 与现有 docs 的关联

本报告的判断与本 branch 已有的几份文档保持一致,可以作为后续工作的 cross-ref:

- `Agent-deligate-gap.MD` —— Arbor-into-Hyperloom 的 IR-4 / specialist-first 设计依据
- `KB_design.MD` —— v0.8 §3.2 phase / §3.4 explore-merge / §3.5 specialist
- `KB_design_gaps.MD` —— Gap-10 v0.6 backends/params/validate_stack 退场记录
- `Version1_report.MD` —— v0.8 落地 12h marathon 的复盘(含 plateau 之后 LLM 反复 explore 同方向的失败模式 → §4 PRO-2 论据)
- `优化报告_DeepSeek-R1-0528_MI300X_vLLM.md` —— 当前 branch 的真实运行报告

### 7.5 一句话决策

> **不直接合 main;按 v0.8 形态(specialist + phase + integrate_patch)重新引入 PR #288 的 `roofline` 复合 action 和 PR #280 的 fa 工具,
> 把 framework agent 嵌进 EXPLORE 的 serving_specialist,把 roofline 放在 EXPLORE 入口 + 内部按需 + KERNEL 入口 + CLOSE 收尾,
> Orchestration roofline-aware dispatch 做成软 advisory(可记录、可统计、不硬拒)。**

