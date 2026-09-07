# Explore 字段审计：legacy vs V6

commit `da54496de`。待讨论文档。

范围：explore / framework config 搜索全过程的字段。legacy 侧是 `param_search`、`explore_search`、`capability_summary.explore`、`capability_summary.specialist`、`specialist_runs`、`optimizations` 的 explore 部分；V6 侧是 `timeline[framework_agent].ext`。

---

## 0. 先说三个死字段

审计中发现的既成事实，跟迁移无关，但会影响你怎么定字段：

| 字段 | 状态 |
|------|------|
| `params_search` ledger | **死**。全仓库无 orchestrator writer，collector 读空 dict 后 shape 成空壳。`collect_explore_search` 的注释直接写了 "unused ones shape to empty shells" |
| `backends_search` ledger | **死**。同上 |
| `explore_search.synergy_attempted` | **恒空**。executor 只做 pass-through，全 orchestrator 无 append writer |
| `state.discovered_flags` | **恒空**。AST 扫描从未实现，`explore.py:2021` 固定返回 `None` |
| `state.framework_config_exploration_results` | **死**。V6 有读取 fallback 路径，但无 writer |
| `explore_search.domains_round_summary` | **死**。pass-through only，无 writer |

三个 ledger 只有 `explore` 是活的。

---

## 1. V6 结构比 legacy 好在哪

这不是简单的字段搬家，V6 的 `framework_agent` 是重新组织过的：

| V6 增值 | legacy 对应 |
|---------|-------------|
| **双 arm 划分**：`config_arm`（配置搜索）与 `source_arm`（PR/patch 改源码） | legacy 混在一起，只能靠 `provenance` 区分 |
| **分角色 specialist**：config specialist、candidate discovery、authoring 三类分开 | legacy `specialist_runs` 一个平表 |
| **per-variant 的 accuracy 子块**：`required` / `reference` / `value` / `passed` | legacy 完全没有 |
| **per-variant 的 failure 子块**：`error_class` / `error_excerpt` | legacy 只有 `rejected[].reason` |
| **per-variant 的 artifacts**：`workspace` / `server_log_path` / `raw_result_path` | legacy 没有 |
| **内嵌 critic reviews** | legacy 的 critic 在另一个段 |
| **source integrate 的完整门控**：`gates.{accuracy_passed, keep_threshold_pct, switch_off_parity_passed}` | legacy 没有 |
| **exit / failure 与 phase 退出证据绑定** | legacy 没有 |

所以对 explore 这块，**方向应该是采用 V6 结构，然后把 legacy 里 V6 没有的补进去**，而不是反过来。

---

## 2. legacy 有、V6 缺，需要你决定留不留

| legacy 字段 | 含义 | V6 现状 | 建议 |
|-------------|------|---------|------|
| `explore.winners_history[]` | KEEP 历史的独立 append-only 表 | 无独立数组，信息散在 `rounds[].variants` | **保留**。plateau 判定的 lookback 现在靠窗口内 rows 重算，有独立表才能录准（见 §4） |
| `explore.no_promote_streak` | 连续未提升轮数 | 无直接字段。`config_arm.plateau.empty_streak` 语义不同（数的是 specialist 空轮，不是未提升轮） | **需明确**用哪个表达 explore plateau，两者不可混用 |
| `top_by_gain[]` | tested 按 gain 降序 top-20 | 无预计算 ranking | 可选。审计友好但能从 variants 排序得到 |
| `tested_count` | 全 session 累计测试数 | `plateau.tested_this_cycle` 只覆盖当前 macro_cycle | **保留**。粒度不同 |
| `capability_summary.explore.best_gain_pct` | 最佳增益 | 无 | 可选 |
| `capability_summary.explore.keep_unstable_count` | 因栈不稳被拒数 | 无 | 可选 |
| entry 的 `proposer` | 解析后的提案者 | 无（有 `provenance` 和 `scope`） | 可选 |
| entry 的 `operation_kind` | backend/param/env 分类 | 无，可从 `config_delta` 推 | 可选 |
| `specialist_runs[].parallelism` | 并行度 | 无 | 可选 |
| `specialist_runs[].confidence_avg` | 平均置信度 | 无 | 可选 |
| `specialist_runs[].transcripts[]` | 磁盘 transcript 引用 | 无 | **建议保留**，排查时要看原文 |
| `specialist_runs[].dispatched_at` / `completed_at` | 派发与完成时刻 | V6 用 operation ts，未投影 | **保留** |
| `specialist_runs[].domain_breakdown` | 按域拆分 | 无 | 见 §3 |
| `specialist_runs[].proposals_{total,kept,rejected,skipped}` | 提案计数 | `proposal_names` 长度；kept/rejected 需解析 proposal_set | 见 §3 |

---

## 3. specialist 提案明细：你已决定补实现

这块是审计表 A24 标记的最脏一处。当前 `_build_specialist_round_entry`（`phases/explore.py:1849`）**不填** `proposals_kept` / `proposals_rejected` / `proposals_skipped` / `domain_breakdown` / `confidence_avg`，于是 `capability_summary.specialist.by_specialist.{domain}` 在 `timeline.py:640` 按域数**平均分摊**——数字是编的。

补实现后，建议直接**逐条 proposal 录一行**，而不是录汇总计数：

| 字段 | 含义 |
|------|------|
| `proposal_msg_id` | 提案 ID |
| `round_id` / `task_id` | 归属轮次 |
| `domain` / `scope` | specialist 域与 dial 范围 |
| `confidence` | 置信度 |
| `status` | kept / rejected / skipped |
| `reason` | 状态原因 |
| `variant_name` / `fingerprint` | 落地成哪个 variant |

这样所有计数和 domain_breakdown 都在 close 时从这些行快照，不再有分摊猜测。

---

## 4. plateau 一族必须在判定时刻录快照

`config_arm.plateau` 和 `source_arm.plateau` 现在是导出时重算的：

| 字段 | 当前算法 | 问题 |
|------|----------|------|
| `recent_keep_gain_pct` | `winners_history[-lookback:]` 的 gain 求和 | session 后期 winners 变了，重算值可能与当时判定相反 |
| `empty_streak` | 倒序扫 `specialist_runs[status=empty]` | 同上 |
| `consecutive_no_keep` | 倒序扫 `framework_agent_phase_progress` | 同上 |
| `candidates_exhausted` | evidence 或 `framework_agent_phase_done` | 部分靠推断 |
| `triggered` | 上述值与阈值比较 | 结论可能与实际行为矛盾 |

这几个是**运行时用来决定要不要切 arm 的输入**。breakdown 可能显示「没触发 plateau」而实际切了 arm，这种矛盾很难排查。必须在判定那一刻把当时用的值和结论一起录下来。

---

## 5. `winners_history` 与 `optimizations.entries[]` 不是一回事

容易误合并，实际语义不同：

| 维度 | `winners_history` | `optimizations.entries[]` |
|------|-------------------|---------------------------|
| 粒度 | 每次 explore KEEP 一行（variant 级） | 每次可归因 adoption 一行（attempt 级） |
| 增益语义 | 相对**当轮 anchor** 的 local gain | 相对 **session baseline** 的 chain 贡献 |
| 覆盖范围 | 只有 explore KEEP | 所有 agent 的采纳（explore + framework + kernel + warm_replay） |
| 数据源 | `state.explore_search` | recorder 的 operations + adoptions |

同一次 explore KEEP 通常两边都出现，但字段集和增益定义不同，不能互相替代。

---

## 6. `capability_summary.explore` 不是纯 rollup

它是多源合成，这也是它口径混乱的原因：

| 字段 | 实际来源 |
|------|----------|
| `attempts` / `keeps` | `state.explore_attempts`（action audit） |
| `keeps`（可能被抬高） | 再叠加 `explore_search.accepted` 的长度 |
| `tested` | `len(explore_search.tested)` |
| `last_validated_gain_pct` | `state.cumulative_gain_validated`，**跟 explore 没关系** |

三个不同来源混在一个对象里。V6 里应该拆开：动作尝试计数归 action 事件，ledger 计数归 config_arm，session 累计增益归 outcome。

---

## 7. 一个合并决定要你定

`outcome.validation.attribution.by_source.framework_agent` **故意把 legacy 的 `explore` 桶和 `framework_agent` 桶合并了**（`v6.py:2149-2151`）。

合并后无法回答「配置搜索贡献了多少 vs 改源码贡献了多少」。V6 的 timeline 里两者是分开的（config_arm / source_arm），但归因汇总把它们合了。

要不要在 `by_source` 里也分开？
