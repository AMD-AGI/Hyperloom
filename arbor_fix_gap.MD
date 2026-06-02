# Arbor → Hyperloom 落地差距清单（fix-gap）

> 配套文档：决策见 `arbor_merge_points.MD`；概念实现见 `arbor_merge_schema.MD`；
> 分阶段设计见 `arbor_merge_details/P1..P7`。
>
> 本文件对照**当前代码**核对 P1–P7 的落地度，列出**仍存在的差距**与**简要解决方法**，
> 形式与上述文档一致。状态图例：
> - ✅ **已落地** = 代码已实现且有测试。
> - 🟡 **部分落地** = 主路径在，但设计某点尚缺。
> - 🔲 **未落地** = 设计点在代码中找不到对应实现。
>
> 分支：`feature/zhenggong/overhaul`。

---

## 0. HEADLINE HINT（实现纪律 —— 写代码时务必遵守）
1. **禁止溯源式注释**：注释只做中性、简洁的功能性描述，不写「本段来自 Arbor / item X / 步骤 Y」。
2. **不留计划痕迹**：代码/docstring/变量名不嵌入 "arbor_merge" / "schema step"；用领域语义命名。
3. **每项交付自带清理**：实现收益的同时下线被取代的旧机制，避免双轨。
4. **fail-soft 默认**：所有新读路径缺失时静默降级，绝不抛错中断主循环。

---

## 1. 总览：逐项落地度

| item | 阶段 | 落地度 | 残余差距摘要 |
|---|---|---|---|
| 1 | P1 动态 specialist + 多 tag + 并行 | 🟡 | GPU-类 specialist 无独立 lane 限流（仅 research_lane=2×GPU）；并行上限仅约束 CPU 侧 |
| 5 | P2 intervention_mix | ✅ | — |
| 3 | P3 research_scout | ✅ | 命中 hint 的 variant 以 **advisory 排序信号**（`priors_match_summary`）注入 prompt（守非门控边界） |
| 2 | P4 depth_tracker / depth_gate | ✅ | — |
| 4 | P5 kernel shape 校验 | ✅ | — |
| 6 | P6 competitor target gap | 🟡 | 「gap 方向」已并入 priors-match advisory 排序信号；inferencex auto-seed 回退未接（设计标可选） |
| 7 | P7 recipe 沉淀 + KB 角色 | ✅ | KEEP→hint 溯源优先用 `gap_canonical_id`（沉淀时盖章 stack entry），name 匹配仅兜底；可选 Cortex KB 写未做（设计标可选） |

清理清单 C1–C11 落地度见 §3。

---

## 2. 逐项差距与解决方法

### GAP-1（item 1 / P1）GPU-类 specialist 无独立并行限流
**现状**：`policy.detect_gpu_count()` + `research_lane_ceiling()=2×GPU` 已替换静态
`MAX_RESEARCH_LANE_CAPACITY=6`（C3 ✅），多 tag 校验 / 多 anchor prompt 装配 / breakdown 按 tag
归因 / 移除 provenance 硬门均已落地。但 `storage/schema.py::DEFAULT_LANE_CAPACITIES` 只有
`research_lane`（CPU 侧），**没有 `gpu_lane`**；P1 设计点 5 要求「GPU 类 specialist 受总 GPU 数约束、
与 CPU lane 分别计数、各自限流」（类比 Arbor GPU pool），目前 GPU 类任务并未按 GPU 总数单独限流。

**影响**：大规模并行下，benchmark/profile 等 GPU-占用型任务可能超订 GPU（依赖现有
`benchmark_lane=1`/`profile_lane=1` 的粗粒度串行，而非「按 GPU 数」的 pool）。

**解决方法（简要）**
- 在 `DEFAULT_LANE_CAPACITIES` 增 `gpu_lane`，容量在 Coordinator boot 时 `set_lane_capacity("gpu_lane", detect_gpu_count())`。
- GPU-占用型 task kind 的 `requires_lanes` 追加 `gpu_lane`（`_resolve_lanes`/`KNOWN_LANES` 同步）。
- CPU/research-lane 与 gpu_lane 分别计数、各自限流（lease 表已支持 per-lane capacity，改动小）。
- 验证：GPU 数=N 时并发 GPU 任务 ≤ N，同时 CPU specialist 仍可并行 2×N。

---

### GAP-2（item 3 / P3）✅ 已落地 — 命中 hint 的 variant 以 advisory 排序信号注入
**落地**：新增 `research_hints.match_variants_to_priors` / `priors_match_summary`，Coordinator 通过
`_priors_match_advisory_block` 把「最近 N 轮 proposal_set 中命中某条 hint（`what`/`domain_tags` token 重叠）」
的 variant 渲染进 orchestration prompt 的 `=== Priors-match (advisory ordering) ===` 段。守边界：仅作
**排序/优先级**提示，不进 Objective、不进 scoring、不构成 gate；无 hints / 无匹配时整段省略（fail-soft）。
有测试覆盖（关键词命中、空集省略）。

> 原始差距记录（供回溯）：
**现状**：research_scout 在 PRELUDE 强制派发、每 K 轮重派、产出 `research_hints.md` +
`competitor_target.json`、注入 gaps[] 种子、与 FRAMEWORK_PR 去重、喂 depth_tracker 计数——均已落地。
hints 也已注入 orchestration / specialist prompt（advisory）。但 P3 设计点 5「EXPLORE gap 评分读取
hints，对命中提示的 variant **加权**」在评分/排序代码里找不到对应实现：hints 目前**只影响 prompt**，
不进入任何 variant 评分。

**影响**：命中已证明高收益项的 variant 不会被更早/更高优先地尝试，弱化了 scout「便宜、命中率高」的收益。

**解决方法（简要）**
- 在现有 proposal/variant 评分处（ProposalScorer 或 explore 排序）增一个 **advisory 加权项**：
  variant 的 `name`/args/domain_tags 命中某条 hint（按 `what`/关键词匹配）时给一个小正权重。
- 守边界（Q6.1/§6 一致性）：加权仅作用于**排序/优先级**，不进入 Objective、不构成确定性 gate。
- fail-soft：无 hints 时权重为 0，排序退回现状。
- 验证：构造一条 hint 命中某 variant，确认该 variant 排序提前；hints 为空时排序不变。

---

### GAP-3（item 6 / P6）🟡 gap 方向已并入 advisory 排序信号；auto-seed 仍未接（可选）
**落地（评分加权部分）**：`primary_gap=='latency'` 时，`match_variants_to_priors` 对携带 latency-direction
关键词（mtp/speculative/decode/comm/overlap/allreduce/cuda_graph/fused…）的 variant 标 `aligns-with-latency-gap`，
并入 GAP-2 同一段 `Priors-match (advisory ordering)`。同样仅作排序提示、不进 Objective/scoring/gate。
**残余（auto-seed，可选/低优先）**：`inferencex_client` 仍休眠，scout 校订 auto-seed 回退未接线（设计标可选）。

> 原始差距记录（供回溯）：
**现状**：tpot 已串进实时状态（`VariantResult` + 4 处 best-value 快照 + derive 兜底）；
`competitor_target.json` 替代 InferenceX 实时拉取（C7 ✅，source=`llm_authored`）；
`load_competitor_target` / `gap_analysis` / `full_gap_summary` 已实现并 advisory 注入 prompt；
`--no-target-advisory` 已有。残余两点：
1. `arbor_merge_points.MD` §7「orch …读 competitor_target 注入 prompt / **给 gap 评分加权**」中的
   「评分加权」未落地（与 GAP-2 同类：当前只 prompt，不进评分）。
2. C8「`inferencex_client` 保留休眠作为可选 auto-seed」——客户端代码确实保留（dormant），但「scout 在其
   拉取结果上校订 competitor_target」的 auto-seed 回退路径**未接线**（设计本就标为可选）。

**解决方法（简要）**
- 评分加权：同 GAP-2，把 `primary_gap=='latency'` 等 gap 信号作为 advisory 权重接入排序（不 gate）。
- auto-seed（可选/低优先）：当 `--compare-against-gpu` 配置但无 `competitor_target.json` 时，可选地用
  `inferencex_client.fetch_rows` 拉一份草稿供 scout 校订；保持 fail-soft、默认关。
- 验证：高 tpot_ratio 场景下相关 variant 排序提前、Objective 不变；auto-seed 关时行为同现状。

---

### GAP-4（item 7 / P7）✅ 已落地 — KEEP→hint 溯源优先用 gap_canonical_id
**落地**：KEEP 沉淀 stack entry 时盖章 `gap_canonical_id`（explore 经 `_lift_to_current_best(..., gap_canonical_id=)`、
kernel-integrate 经 `_record_integrate_keep` 读 `result.gap_canonical_id`）。`_collect_attempt_provenance`
额外返回 `{gap_canonical_id: provenance}` 映射并按 `kernel_id` 兜底建索引；`_build_recipe_attrs_from_state`
解析 `source` 时**优先**用 entry 的 `gap_canonical_id`，name/kernel_id 匹配仅兜底——跨阶段命名不一致时
source/gain 仍能正确沉淀。有测试覆盖（kernel-integrate 名不匹配仍带 source）。

> 原始差距记录（供回溯）：
**现状**：`gaps[]` 新增 `provenance`（round-trip ✅）、`_collect_attempt_provenance` 把 KEEP attempt
映射回 hint source、`_build_recipe_attrs_from_state` 把 source 沉淀进 `what_worked`、REVERT 进
`what_failed`、scout 读 warm recipe 跳过已证明项、KB 角色降级（cold-start 优先 hints, C9 ✅）、
`--no-recipe-sediment` 已有。残余风险：溯源匹配键是
`what_worked[*].name == gap.attempts[*].variant_name`，而 `optimization_stack` 条目的 `name`
与 explore/integrate 写入 gap attempt 的 `variant_name` **并非保证同名**（kernel integrate 用
`kernel_id`、explore 用 variant_name），跨阶段优化的 source 可能漏标（沉淀为「无 source 的 what_worked」）。

**影响**：部分 KEEP 项虽源自 hint，但因命名不一致而未带上 `source`/`gain`，弱化「越用越聪明」的可审计性。
（fail-soft：漏标只是退化为无 source，不报错。）

**解决方法（简要）**
- 在 KEEP 写 stack 时同时记录其 `gap_canonical_id`（若该 explore/integrate 携带），
  `_collect_attempt_provenance` 改为优先用 `gap_canonical_id` 直接定位 gap 的 `provenance`，
  名匹配仅作兜底。
- 或在 stack entry 落地处统一 `name`（explore variant_name / kernel_id 归一），让匹配键一致。
- 验证：kernel-integrate 的 KEEP 若源自 hint，recipe `what_worked` 仍带 source。

---

### GAP-5（item 7 / P7，可选）可选 Cortex KB 沉淀未实现
**现状**：设计明确「`--cortex-kb-url` 配置时额外 best-effort commit 高置信 KEEP 为 KB 点」标为
**可选/后续**，因 v1 图谱写路径 `cortex_*` 是死代码（C10）。当前未实现，符合「可选」定位。

**解决方法（简要，低优先）**
- 若要做：在 finalize 处对高置信 KEEP 项 best-effort 调一个**新的**轻量 KB commit（不复活 v1 `cortex_*`），
  fail-soft、默认关、绝不阻塞主 recipe sink。
- 否则：保持现状（仅本地 recipe-snapshot sink），并把本项标注为「明确不做」。

---

## 3. 清理清单 C1–C11 落地核对

| # | 清理对象 | 设计动作 | 落地 |
|---|---------|---------|------|
| C1 | `explore_requires_specialist_provenance` 规则/常量/调用点 | 删除 | ✅ 仅余注释/prompt 文案提及 |
| C2 | specialist 闭集 domain 拒绝语义 | 改义为逐 tag 词表校验 | ✅ `KNOWLEDGE_DOMAIN_TAG_SET` 逐 tag 校验 |
| C3 | `MAX_RESEARCH_LANE_CAPACITY=6` 静态上限 | 替换为 2×GPU 动态上限 | ✅ `research_lane_ceiling()` |
| C4 | breakdown「必须 seed 全闭集 domain」 | 降级为可选展示 | ✅ tag 归因 + 单 tag 回退 |
| C5 | 「每 session 仅 1 次 steward continuation」上限 | 删除（depth_gate 取代） | ✅ |
| C6 | kernel「空 shape 也空跑 GEAK」隐式路径 | 断言前置堵死 | ✅ `_validate_kernel_shape_and_paths` |
| C7 | `target_analyzer` InferenceX 实时拉取 | 下线主路径 | ✅ 改读 `competitor_target.json` |
| C8 | `inferencex_client` | 保留休眠（可选 auto-seed） | 🟡 保留休眠 ✅；auto-seed 回退未接（见 GAP-3） |
| C9 | cold-start 直接回退 domain-focus | 改为优先 research_hints 填补 | ✅ `_section_kb_subgraph` 优先 hints |
| C10 | v1 图谱写路径 `cortex_*` 死代码 | 可选清除（低优先） | 🔲 未清除（设计标可选/独立 PR） |
| C11 | 单 `domain` 字段硬假设 | 改为 `tags[]`，`domain` 兼容别名 | ✅ `normalize_dispatch_tags` |

---

## 4. 跨项一致性复核（与 schema §6 对齐）

- **tag 词表单一来源**：`KNOWLEDGE_DOMAIN_TAG_SET`（含 `research_scout`）已统一，无漂移。✅
- **depth_tracker 计数齐套**：item 3/5 + `specialist_done.research` 写入均在；无供给维度动态 N/A 已实现。✅
- **advisory 不越界**：gap/hints/KB 不进 Objective/scoring/gate。「加权」以 **advisory 排序信号**
  （`Priors-match` prompt 段）落地（GAP-2/GAP-3），严格只作用排序/优先级，不进 Objective/scoring/gate。✅
- **IR-6 至上**：depth_gate 保留 IR-6 兜底。✅
- **resume 兼容**：新增 SharedState 字段（depth_tracker / tpot / tags / provenance /
  target_advisory_enabled / recipe_sediment_enabled）均 round-trip 且容缺。✅
- **breakdown 向后兼容**：单 `domain` 归档 session 走 tag/domain 回退读。✅

---

## 5. 建议处理顺序

1. **GAP-1**（GPU lane 限流）：独立、收益清晰、与并行主线强相关，优先。
2. **GAP-2 + GAP-3 评分加权**：同一类改动（advisory 权重接入排序），一并做，注意守非门控边界。
3. **GAP-4**（KEEP 溯源键归一）：小而稳，提升沉淀可审计性。
4. **GAP-5 / C10**（可选 KB 写 / 死代码清除）：低优先，可单独小 PR 或明确「不做」。

每项独立成 PR、独立可验证、独立可回滚；新增 advisory 信号缺失即视为该特性关闭（fail-soft）。
