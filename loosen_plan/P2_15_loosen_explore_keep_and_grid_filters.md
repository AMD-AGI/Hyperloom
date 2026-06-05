# P2_15 — explore KEEP 阈值参数化 + 删 roofline 硬门 + 删 grid 重排/经验跳过

- **Phase**: P2 · **风险**: 中 · **依赖**: 无 · **后继**: 无

## 目标

explore 执行器与 grid runner 里有若干"代替 LLM 判断 win/loss、该不该试、按什么顺序试"的逻辑。保留**真去重 + 资源超时 + 真不兼容**,删除经验性策略过滤与强制排序。

## 逐项判定(审计)

| 项 | 位置 | 当前 | 性质 | 处理 |
|---|---|---|---|---|
| KEEP 阈值 | `explore.py` 86–94 (`DEFAULT_KEEP_THRESHOLD_PCT=1.0`,stack stable 0.5%) | 硬编码裁决 | STRATEGY | **参数化**(LLM 可在 grid params 指定 `keep_threshold_pct`),降级 advisory 默认 |
| overtime kill | `explore.py` 517–563, 828–924 (`explore_overtime_kill_ratio`) | 超时跳过变体 | **资源 INVARIANT** | 保留(资源超时,非策略 prune) |
| 内容指纹去重 | `explore.py` 724–749 | ledger + in-round dedup | INVARIANT | 保留 |
| roofline 硬门 | `explore.py` 753–781 + `_explore_roofline_filter.py` 116–220 | 所有方向 ≥80% 饱和则丢弃变体 | STRATEGY | **删除**(改 advisory "likely saturated" 标注,默认本就 off) |
| atom 默认 grid 种子 + model_class gating | `explore.py` 195–286 | 注入默认 grid | STRATEGY(种子)/ INVARIANT(model 不兼容) | 缩小种子或仅 LLM 提供;**保留** model 不兼容判定 |
| 指纹 dedup key | `_grid_runner.py` 58–74 | — | INVARIANT | 保留 |
| 多节点丢 `cuda_graph_max_bs<CONC` | `_grid_runner.py` 224–272 | 经验丢弃 | STRATEGY | **降级 advisory**(标注建议跳过,不强制) |
| `reorder_grid_for_multi_node` | `_grid_runner.py` 316–348 | 强制优先级排序 | STRATEGY | **删除/降级**(顺序由 LLM grid 顺序决定) |
| 兼容/skip-list/single-vs-multi 过滤 | `_grid_runner.py` 369–398, 533–643 | 混合 | MIXED | **保留真不兼容**(框架/硬件不支持),删经验性 skip |
| 每-variant timeout / 串行 | `_grid_runner.py` 156, 1380–1466 | 7800s | INVARIANT(资源) | 保留 |

## 改动清单(删除优先)

### 1. KEEP 阈值参数化(`explore.py`)
- `DEFAULT_KEEP_THRESHOLD_PCT` / stack_stable 阈值改为可被 grid params 覆盖(`keep_threshold_pct` / `stack_stable_threshold_pct`,EMIT hint 已暴露)。保留默认值作 advisory,但 LLM 指定优先。

### 2. 删 roofline 硬门(`explore.py` + `_explore_roofline_filter.py`)
- 删 `explore.py` 753–781 与 `_explore_roofline_filter.py` 116–220 的 **drop** 行为。若需提示饱和,改为在结果里**标注** `likely_saturated`(advisory),不丢弃变体。
- env `INFERENCE_OPTIMIZER_EXPLORE_ROOFLINE_HARD_GATE`(默认 off)与 CLI `--explore-roofline-hard-gate`:删除(死开关)。

### 3. 删/降 grid 重排与经验跳过(`_grid_runner.py`)
- 删 `reorder_grid_for_multi_node`(316–348):保持 LLM 给的 grid 顺序。
- 多节点 `cuda_graph_max_bs<CONC` 丢弃(224–272):降级为 advisory 标注。
- skip-list / single-vs-multi 经验过滤(369–398, 533–643):**保留真不兼容**(框架不支持的组合),删经验性"通常没用"的 skip。

### 4. 缩小默认 grid 种子(`explore.py` 195–286)
- atom/默认 grid 种子缩小或仅在 LLM 未提供 grid 时兜底;**保留** model_class 不兼容剔除(INVARIANT)。

## 连带测试
- explore 执行器测试:KEEP 阈值参数化、roofline 硬门删除(若有 `test_explore_roofline_filter` / explore keep 测试)。
- `_grid_runner` 测试:重排/经验跳过删除;**保留**去重、timeout、真不兼容用例。
- 以实际 grep `DEFAULT_KEEP_THRESHOLD_PCT` / `roofline_hard_gate` / `reorder_grid_for_multi_node` 的测试为准更新。

## 验证
- LLM 指定 `keep_threshold_pct` 生效;未指定用默认。
- 高饱和方向的变体仍会被 bench(不再静默丢弃),结果带 advisory 标注。
- grid 按 LLM 给定顺序执行;真不兼容组合仍被剔除并记录原因。
- 资源:overtime kill / per-variant timeout / 去重保留生效。

## 回退
- 恢复硬门、重排、经验跳过与默认阈值。

## 残留风险
- 中。删 roofline 硬门后,饱和方向也会消耗 bench 时间 —— 由 overtime kill / per-variant timeout / phase 预算兜底。缩小默认种子可能减少 cold-start 覆盖 —— 由 LLM 主动 grid + specialist 补足。
