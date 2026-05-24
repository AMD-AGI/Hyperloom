# Main 主动放弃清单

> 本 branch (`feature/zhenggong/lossen_explore`) 主动选择 **不要** `origin/main` (PR #288 + #280)
> 的哪些产物。F1/F2/F3 解冲突时直接选 "ours";后续 PR 描述里引用此清单解释每个选择。

## 类别 1: v0.6 残骸 (与 KB_design §3.4 / §3.9 / §3.13 冲突)

| 主动放弃 | 来源 PR | 不要的理由 | 替代方案 |
|---|---|---|---|
| `action_executors/backends.py` (main 改动) | #288 | v0.6 已退役 (KB_design §3.4) | F1: 功能并入 explore |
| `action_executors/params.py` (main 改动) | #288 | 同上 | F1: 同上 |
| `action_executors/validate_stack.py` (main 改动) | #288 | M3 已内联进 explore (§4.4) | M3 KEEP 后 explore 自动 rebench |
| `scoring.py` (main 改动) | #288 | scoreboard 已退场 (§3.9) | KB priors 在 Critic 侧 |
| `actions/_meta/{backends,params,validate_stack}.yaml` | #288 | v0.6 退役 | F1: 取 `roofline.yaml` 即可 |
| `actions/validate_stack.md` (main 改动) | #288 | 同上 | M3 内联 |
| `tests/test_p3_search_space_expansion.py` (main 改动) | #288 | 测试已删除 | F1: 用 `test_explore_*.py` 覆盖 |
| `tests/test_validate_stack.py` (main 改动) | #288 | 同上 | 同上 |
| `tests/test_validate_stack_gate_skip.py` (main 改动) | #288 | 同上 | 同上 |

## 类别 2: v0.6 sequence_denial (会永久锁死 v0.8)

main 上 PolicyGate 仍然有 `backends_attempts < 1` / `params_attempts < 1` → deny `kernel_opt`
之类的 sequence_denial 规则。在 v0.8 上 **没有写者** 给 `backends_attempts` / `params_attempts`,
denial 永真,合并后 `kernel_opt` 永久锁死 → 整个 KERNEL phase 跑不动。

| 主动放弃 | 行为 | F3 替代 |
|---|---|---|
| `backends_attempts < 1` sequence_denial | v0.8 永真 → 锁死 kernel_opt | F3: 重写为 `explore_attempts_succeeded < 1` |
| `params_attempts < 1` sequence_denial | 同上 | F3: 同上,合并 |
| `validate_stack_attempts < 1` sequence_denial | 同上 | M3 已内联,无需替代 |

## 类别 3: 与 IR-4 specialist-first 冲突的 prompt / action

| 主动放弃 | 来源 | 不要的理由 | 替代 |
|---|---|---|---|
| `framework_pr` 顶层 action | #280 | 违反 IR-4 (PR-A9):EXPLORE 必须 specialist-first | F2: 嵌入 `serving_specialist` 的 sub_kind |
| `framework_pr first-explore priority rule` | #280 `orchestration.md` | 同上 | F2: 走 specialist proposal 路径 |
| `--framework-gap` / `--framework-pr-discover` CLI 标志 | #280 `cli.py` | 暴露给操作员形成 v0.8 之外的"第二条路" | F2: 单一 `--framework-agent-enabled` 开关 |
| `action_executors/framework_pr.py` 独立 bandit arm | #280 | bandit 概念已退场 (§3.9 priors 走 KB) | F2: 不引入 |

## 类别 4: 已退场的 LLM 决策框架 (合并风险高)

| 主动放弃 | 来源 | 替代 |
|---|---|---|
| `orchestration.md` 中 "Action scores top-12 block" | (历史) | 已用 phase + gaps + KB priors 决策 (§3.9 Inv-9.1) |
| `orchestration.md` 中 "validate_stack is mandatory" 段 | (历史) | M3 已内联进 explore |
| 所有 `score_violation` rule_id | (历史) | scoreboard 退场后无意义 |

## 类别 5: 部分采纳

| 文件 | main 改动量 | 我们要的部分 | 我们不要的部分 |
|---|---|---|---|
| `system_prompts/prompt_builder.py` | 大 (+434/-45) | `_format_analysis_md_full` + section 5.5 注入 | v0.6 backends/params 字段渲染 |
| `system_prompts/orchestration.md` | 大 (+173/-5) | 🔴/🟡/🟢 标记说明 + analysis.md → action mapping | Action scores block / framework_pr first-explore / score_violation 段 |
| `coordinator.py` | 大 (+1179/-82) | record_partial / next_pending_keep_kernel_id / untried_hot_reusable_kernels gate | ~280 行的 v0.6 backends/params/validate_stack 路径 |
| `cli.py` | 中 (+306/-79) | (无;F2 自己加 `--framework-agent-enabled` toggle 即可) | `--framework-gap` / `--framework-pr-discover` |

## 类别 6: 完全采纳但形态调整

| 来源 | main 形态 | 本 branch 形态 |
|---|---|---|
| `roofline` 复合 action | 一等公民 action | F1: 同样一等公民,但 EXPLORE phase 通过 allowlist 接收 |
| `orchestrator/roofline_snapshot.py` | F0-7 cherry-pick | 不动,无 v0.6 依赖 |
| `orchestrator/_analysis_keyword_map.py` | F0-7 cherry-pick | 不动 |
| `framework-agent/` 子目录 | F0-7 cherry-pick | F2: 嵌入 serving_specialist 工具链 |
| N9 deny direct profile | PolicyGate 规则 | F3: 同样实现 (`n9_deny_direct_profile_when_composite_on`) |
| N19c gain-driven kernel_opt | PolicyGate 规则 | F3: 重写为 `explore_attempts_succeeded` 风格,适配 v0.8 |
| N31 final roofline | Coordinator auto-enqueue | F3: 同样实现 (CLOSE phase 入口触发) |

## 解冲突指引

**对 dry-run 列出的 16 个 content + 7 个 modify/delete 冲突**:

| 冲突文件 | 解法 |
|---|---|
| `action_executors/{backends,params,validate_stack}.py` | `git rm` (保持删除) |
| `scoring.py` | `git rm` |
| `tests/test_p3_search_space_expansion.py` | `git rm` |
| `tests/test_validate_stack*.py` | `git rm` |
| `cli.py` | "ours" + 单独 cherry-pick `--framework-agent-enabled` (F2) + roofline toggles (F0-10) |
| `coordinator.py` | "ours" + F1 加 `record_trace_analyze` 调用 + F3 加 N31 enqueue |
| `shared_state.py` | "ours" + F0-8/10 占位字段 + F1 `record_trace_analyze` 方法 |
| `system_prompts/{orchestration,critic,prompt_builder}.{md,py}` | "ours" + 手工 port `_format_analysis_md_full` + analysis.md→action 映射 (F1-4/5) |
| `action_executors/__init__.py` | "ours" + F1-3 注册 roofline executor |
| `action_executors/report.py` | 内容冲突,case-by-case;若 main 改动只是 v0.6 字段渲染则 "ours";若是 roofline 报告就接 |
| `sub_agent_runner.py` | "ours";F1/F2 不需要扩 lease 类型 |
| `paths.py` / `session_paths.py` | "ours";v0.6 路径常量已删 |
| 5 个测试文件 (test_p0_1_protocol / p1_2_full_action_catalogue / p5_decision / prompt_assets / required_step_gates) | 大概率 "ours"(本 branch 的 PolicyGate 规则集和 main 不同);case-by-case |
