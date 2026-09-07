# critic 字段统一方案

commit `da54496de`。待讨论文档。

你的判断（`critic_robustness` / `specialist_runs` 是无意义的重复字段）成立。critic 事实现在散在 7 个地方，下面是统一方案。

---

## 1. 现状：7 个 critic 数据源

| # | 位置 | 粒度 | 内容 |
|---|------|------|------|
| 1 | `critic_iterations` recorder section | **per-turn** | 14 个字段，含 `framework_reviews[]`、`kb_writes[]`、`kb_priors` |
| 2 | `robustness_signals` recorder section | per-signal | 4 个字段，**但无 writer**（见 §5） |
| 3 | `critic_robustness` legacy 段 | per-turn | `critic_iterations` 的**子集**（只 9 个字段，丢了 `iteration_id` / `phase` / `macro_cycle` / `framework_reviews` / `kb_writes` / `kb_priors`） |
| 4 | `critic-workdir/<turn>/*.json` | per-turn | `request` / `judge_bundle` / `review` / `emit` 四个工件 |
| 5 | `framework_agent.ext.critic_reviews[]` | **per-proposal** | 18 个字段，但只有 phase ∈ {FRAMEWORK_AGENT, EXPLORE} 的行 |
| 6 | v4 `operations` + `trace_events` | per-turn + per-kb-write | `kind=critic` / `kind=kb_write` |
| 7 | `critic-session-memory` | per-verdict | `mark_reviewed` / `append_decision` |

**核心矛盾**：#1 是 turn 粒度，#5 是 proposal 粒度。一次 critic turn 可以包含 N 个 `review_verdicts`。两套并存，计数口径必然对不上。

---

## 2. critic 活动的 5 种形态

| 形态 | 被评审对象 | 可能 verdict | 下游效果 |
|------|-----------|-------------|----------|
| **提案评审**（主路径） | bus 上 pending 的 proposals | `approve` / `reject` / `redirect` / `advise` / `needs_review`，外加 per-variant 的 `verdict_map` | materialize / deny / reauthor / explore grid 过滤 / bus 广播 / 可选写 KB |
| **决策评审** | 显式 decision 对象 | `adopt` / `reject` / `revise` / `needs_info` | 写 KB + session memory，无 intent_envelope。runtime 支持但 coordinator 当前不走这条 |
| **KB draft 提交** | KB draft 列表 | 无 verdict（批量写） | KB upsert |
| **framework candidate 门控** | 上游 PR / specialist 产物 | 同提案评审 | `critic_denied` 行 / 入队 author / reauthor 循环 |
| **integrate/patch 门控** | specialist_task_id 或 candidate_id | 只有 `approve` / `advise` 才放行 | PolicyGate 阻断 integrate_patch |

后三种其实是**提案评审的子集**，只是 subject 类型和下游不同。真正独立的只有决策评审和 KB draft。

---

## 3. 作用域问题

critic **每 tick 都跑**，与 orchestration / robustness 并列（`coordinator.py:842`），**不绑定任何单一 phase handler**。

- `phase` / `macro_cycle` 由 coordinator 在 run 前经 `set_trace_context` 注入 `request.context`
- **可以为空**：manifest 缺失时 `required_context` 非空，强制 `needs_review` + `critic_unavailable`

这直接说明一个问题：**critic 不该只嵌在 `framework_agent` 事件里**。KERNEL 等 phase 的 review 现在**确实产生了** `critic_iterations` 和 bus `review_verdict`，但因为 `critic_reviews.py:60-137` 只放行 FRAMEWORK_AGENT / EXPLORE 两个 phase，这些行**不进任何 timeline 投影**，等于丢了。

---

## 4. 统一方案

### 主粒度定为 per-proposal verdict

turn 只作为分组键。理由：verdict 才是有业务语义的单位，turn 顶层的 `verdict` / `topic` / `summary` 三个字段**在实际数据里基本恒空**（review 用的是 `review_verdicts[]` 数组，emit 没有顶层 ts/topic），`kb_writes_summary` 统计的正是这个恒空字段，所以常年是 `{total: 0}`。

### 建议的 `timeline[critic]` 事件形状

```
type: critic
kind: agent
status: succeeded | failed | degraded
start_time / end_time
ext:
  tick
  phase                              可为空
  macro_cycle                        可为空
  review_batch_id                    同一 turn 的分组键
  request_kind                       coordinator_inbox | decision_request | kb_draft | session_close
  verdicts[]:
    subject:
      kind                           proposal | decision | kb_draft
      proposal_msg_id
      candidate_id
      variant_name
      action_name
      decision_id
    verdict                          原始裁决
    effective_verdict                emit intent 生效后的裁决
    source
    reasoning
    confidence
    predicted_gain_pct
    risks[]
    required_evidence
    failure_reason_code
    advice_text
    alternative_action
    followup_task_ids[]
    persist_to_kb
    downstream:
      materialized                   是否落地成 variant/task
      framework_denied               是否写了 critic_denied
      patch_verdict_key              integrate_patch 门控用的键
      arm                            config | source
      target_action                  explore | integrate_patch | specialist
  kb:
    priors:                          configured / mode / scope_filter / limit / prior_count / requested / skipped_reason / referenced_in_verdict
    writes[]:                        trigger / target_proposal_msg_id / status / detail
  availability:
    required_context[]               缺失的 manifest 上下文
    unavailable_reason
  artifacts:
    request_path / judge_bundle_path / review_path / emit_path
  failure:
    error_class / error
```

### 录制时机

- `verdicts[]` 每条在 **critic 产出该 verdict 时**录一行
- `downstream.*` 在 **intent_router 处理该 verdict 时**回填（这是另一个时刻，需要用 `proposal_msg_id` 关联）
- `kb.writes[]` 在 **KB write 完成时**录
- 事件在 turn 结束时 finish

`downstream` 的回填是唯一需要跨时刻关联的地方。如果不想跨时刻回填，也可以把它拆成 action 事件上的一个 `critic_verdict_ref` 字段，反过来由动作侧引用 verdict。这块想听你意见。

---

## 5. 一个实际 bug：`robustness_signals` 无 writer

`collect_critic_robustness` 读 `robustness-workdir/*/signal.json` 和 `action.json`。

**全代码库没有任何地方写这两个文件。** robustness backend 只写 `request.json` 和 `emit.json`（`robustness/runtime/cli.py:31-40`）。

所以 `robustness_signals` 在真实 session 里 `signal` 和 `action` 都是空字符串，只有 `workdir` 和 `ts`。`close.robustness.signals` 同样受影响。

实际的 robustness ladder 结论落在另一个地方：`{session_dir}/agents/robustness/findings/{session_id}.jsonl`（`findings.py:6-13`），而且 critic 的 prepare-review 会把它作为 `robustness_priors` 读进去。

**两个选择**：补 `signal.json` / `action.json` 的 writer，或者改成读 findings jsonl。我倾向后者，因为 findings 已经是活的数据源。

---

## 6. robustness 应该独立，不并进 critic

robustness 与 critic 是**邻域关系**而非同类：

- robustness 产出 findings，critic 消费 findings 作为 priors
- robustness 的 intent 是 `prune_branch` / `escalate_strategy_change` / `alert` / `delegate(recover)`，词汇和生命周期完全不同
- robustness 还有一组 critic 健康信号（`signals/critic_health.py`）：`critic_kb_outage`、`critic_unavailable_streak`、`critic_prune_stuck`、`critic_runtime_stuck`——这是 robustness **监控** critic

建议：`timeline[critic]` 和 `close.robustness.findings[]` 两条独立流。

---

## 7. 两个不要塞进来的东西

**`proposal_scorer`** 不是 critic（`scoring/proposal_scorer.py:4-8`），是独立的 advisory scorer，只挂在 decision trace 的 `proposal_scores` 上。你已经决定不再记录 `decision_trace`，那这块跟着一起删。

**KernelForge 的 PlanCritic**（`kernelforge/orchestrator/plan_critic.py`）verdict 是 `ACCEPT` / `REVISE` / `REPLACE`，完全不在 inference_optimizer 的 critic 管线内。如果要记录，应该作为 kernel 事件下的 forge 子字段，不要塞进 critic 的 verdict 枚举。

---

## 8. 删除后的映射

| legacy | 去向 |
|--------|------|
| `critic_robustness.critic_iterations[]` | `timeline[critic].ext.verdicts[]`（升为 proposal 粒度） |
| `critic_robustness.robustness_signals[]` | `close.robustness.findings[]`（改数据源） |
| `critic_robustness.kb_writes_summary` | 拆成两个：verdict 计数 和 KB write 结果计数（现在这个名字统计的是恒空的顶层 verdict，是误名） |
| `framework_agent.ext.critic_reviews[]` | 保留 `[争]`。作为 framework 事件内的便利引用，还是完全移到 critic 事件？我建议只在 framework 事件里留 `critic_verdict_refs[]` 指向 critic 事件，正文只有一份 |
| `specialist_runs` | 见 `SBD_V6_EXPLORE_FIELDS.md` §3 |
