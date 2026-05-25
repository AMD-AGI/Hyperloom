# Main → lossen_explore 语义合入计划

> 顶层索引。继 `plan_roofline_framework/` (F0..F3,已完成,tag `f3-done`)之后,
> 把 `origin/main` 上 134 个 commit 里**仍未吸收**的部分按语义增量合入本分支。

## 完成状态对账

| 阶段 | 内容 | tag | 状态 |
|---|---|---|---|
| **F0** | 集成冻结准备 + 文档 | `f0-done` | ✅ 已完成 |
| **F1** | Roofline 复合 action 移植(PR #288) | `f1-done` | ✅ 已完成 |
| **F2** | framework-agent 嵌入 serving_specialist(PR #280) | `f2-done` | ✅ 已完成 |
| **F3** | PolicyGate v0.8 形态 + Roofline soft advisory | `f3-done` | ✅ 已完成 |
| **M0** | 准备 + 低风险 PR + 11 个 kernel-opt 纯 fix | `m0-done` | 🔵 待执行 |
| **M1** | PR #239 multi-node 全量吸收(单节点 + 多节点双轨) | `m1-done` | 🔵 待执行 |
| **M2** | N17 + N23 per-model+per-launch session_dir 改造 | `m2-done` | 🔵 待执行 |
| **M3** | 纯 v0.8 兼容的 N 系列 follow-up 提取 | `m3-done` | 🔵 待执行 |

## 关键路径变量(执行前 export)

```bash
export FEAT_ROOT="/wekafs/zgong/Hyperloom-KB"
export FEAT_BRANCH="feature/zhenggong/lossen_explore"
export ROLLBACK_TAG="f3-done"          # M0 起点 = F3 完成点
export MAIN_REF="origin/main"          # 直接从 fetch 来的 main ref 取文件
                                        # (无需另外 clone 一份 main checkout)
```

> 与 F-phase plan 不同,M-phase 不再依赖 `$MAIN_ROOT` 独立 checkout。
> 所有从 main 取文件的操作都走 `git show origin/main:<path>` 或
> `git checkout origin/main -- <path>` —— 假设 `git fetch origin main`
> 已经跑过(每次 phase 开始时强制 re-fetch)。

## 合入语义(hybrid: per-commit + per-feature)

按 Q6 hybrid 决策:

| 类型 | 合入方式 | 标记 |
|---|---|---|
| 纯 bug fix,无 v0.6 / scoreboard 依赖,单文件低冲突 | `git cherry-pick <sha>` | **CP** |
| 跨 feature 的语义改造,涉及本分支已 ours-取舍 的文件 | "ours + 手工 port" | **MP** |
| 整个新增子树,无冲突,bulk add | `git checkout origin/main -- <path>` | **BA** (bulk add) |
| 明确 drop(违反 v0.8 不变量 / MUST_PRESERVE) | 不动 / `git rm` | **DR** |

每个 plan 文件里,**每一步**都标明 `CP / MP / BA / DR` 中的一个。

## 执行规则

1. **顺序执行**:M0 → M1 → M2 → M3,任何一步失败立即停下,不要并行。
2. **每个 commit 后跑测试**:
   ```bash
   pytest inference_optimizer/tests -q --tb=line 2>&1 | tail -5
   ```
   绿色测试数必须**单调不降**(基线 1624 passed / 1 skipped /
   见 `docs/test-baselines/known_failed_20260524.txt`)。
3. **每个 phase 完成后打 tag**:`m0-done` / `m1-done` / `m2-done` / `m3-done`。
4. **每个 phase 完成后跑 MUST_PRESERVE 校验**:
   ```bash
   bash docs/integration/check_preserved.sh   # rc=0 必须成立
   ```
5. **失败回滚**:见每个 plan 文件的"回滚"章节。

## 与 F-phase 的关系

F-phase 留下的"集成冻结"状态(`docs/integration/MUST_PRESERVE.md` + 
`docs/integration/MAIN_FEATURES_DROPPED.md`)在 M-phase 里仍然生效。
M-phase 不会:
- 重新引入 `backends.py` / `params.py` / `validate_stack.py` / `scoring.py`(MUST_PRESERVE §1)
- 接受任何 `framework_pr` 顶层 action / `--framework-gap` CLI 标志(MAIN_FEATURES_DROPPED §3)
- 接受 `backends_attempts` / `params_attempts` 字段写者(MAIN_FEATURES_DROPPED §2)

M-phase 结束后,可以发 PR 到 `origin/main`(或先合到上游 `lossen_explore`,
看再合 main 的时机)。M3-done 之后 SKILL.md 里的 "Integration Freeze
(active 2026-05-24 → tag f3-done)" 段落要更新为 "expired at tag m3-done"。

## 测试基线

| Phase | 期望 pass | 增量来源 |
|---|---|---|
| 起点(f3-done) | 1624 | F3-7 归档 |
| m0-done | 1624 + N(M0 新加的 11 个 kernel-opt fix 自带测试) | M0 |
| m1-done | m0 + ~25(multi-node + framework-pr-discover 删后剩余的) | M1 |
| m2-done | m1 + 8(N17 test_p6_session_layout) | M2 |
| m3-done | m2 + ~20(N5/N6/N9/N11/N12/N25/N26/N27/N32/N33/N34/N36/N38) | M3 |

任一 phase 完成后,`pytest inference_optimizer/tests -q --tb=line` 必须满足:
- pass >= 上一 phase pass + 该 phase 增量
- failed ⊆ `docs/test-baselines/known_failed_20260524.txt`

## 文件清单

| 文件 | 阶段 | 工作日 | 风险 |
|---|---|---|---|
| [M0_pre_merge_and_lowrisk.MD](./M0_pre_merge_and_lowrisk.MD) | M0 | 1 天 | 极低(全部加性 + 1-3 行单点改动) |
| [M1_multinode.MD](./M1_multinode.MD) | M1 | 2-3 天 | 中(新维度,但 multi_node/ 子树独立) |
| [M2_session_layout.MD](./M2_session_layout.MD) | M2 | 1-2 天 | 中高(SKILL.md / paths.py / 所有 IR doc 必改) |
| [M3_pure_n_extracts.MD](./M3_pure_n_extracts.MD) | M3 | 1-2 天 | 低(每个 N 都是隔离的语义增量) |

总计 ~5-8 个工作日。
