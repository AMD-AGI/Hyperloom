# specialist-queue.plan.md

把 specialist 产出的、尚未被 benchmark 的可执行提案，接通到 explore 测试 lane。

本版为第二版，依据一次独立代码审核重写。第一版的渲染格式、指纹算法、过滤语义、注入位置四处核心设计均已修正；骨架决策（不新增 SharedState 字段、不加 DB 迁移、不恢复 PolicyGate cap、不自动 materialize、不新增 context 工具、三段 commit）不变。

## 1. 问题与证据

对 38 个本机会话（`coordinator.db` 的 `tasks` / `events` 表 + `state.json`）的统计：

| 指标 | 数值 |
|---|---|
| specialist 提案总数 | 801 |
| explore grid 变体总数 | 202 |
| 其中复用了某个提案 | 105 |
| Orchestration 自己现写的 | 97（48%） |
| 提案进过 grid 的比例 | **13.1%** |
| 从未被 benchmark 的提案 | **~696（86.9%）** |

按域拆开：

| 域 | 提案数 | 进 grid | 转化率 | 完整渲染进提示词 |
|---|---|---|---|---|
| research_scout | 186 | 84 | **45.2%** | 是（SEED 块） |
| comm | 5 | 2 | 40.0% | 否 |
| serving | 378 | 16 | **4.2%** | 否 |
| freeform | 59 | 2 | 3.4% | 否 |
| kernel_switch | 39 | 1 | 2.6% | 否 |
| static_recon | 66 | 0 | 0% | 否 |
| compiler | 6 | 0 | 0% | 否 |

`research_scout` 只贡献 23% 的提案，却占 grid 复用的 80%（84/105）。

### 1.1 机制

- 提案只存在于 `delegated_result` 消息的 `result.specialist_done.proposal_set`。
- `loop/coordinator.py:374-425` 的 `_format_inbox_event` 对 `delegated_result` 只渲染一行 header，`proposal_set` 完全不渲染。
- inbox 是游标式且有损的（`loop/conversation.py:742-746` 注释原文：`the inbox tail is lossy`）；耐久重投 `_augment_critic_inbox_with_pending` 只对 Critic 生效。
- 单节点没有自动物化：`phases/explore.py:1452` 第一行即 `if not is_multi_node(): return`。
- `state/_shared_state/render.py:562` 的 `to_prompt_summary` 不含任何 pending-proposal 字段；`last_specialist` 只存 `proposals_total` 计数。

对照组：patch 路径闭环。50 个产出 patch/artifact 的 specialist → 97 个 `integrate_patch` 任务（succeeded 90 / cancelled 4 / running 2 / queued 1），38 个会话中只有 1 个合成 proposal 未被 Critic 裁决。差别在于 patch 走 `pending_proposals` 耐久注册表 + Critic 强制裁决。

### 1.2 因果强度：一次对照检验

「渲染通道解释了转化率差距」是相关 + 机制，不是证明。可能的替代解释是 research_scout 的提案本身质量更高。

在已被 benchmark 的 specialist 提案中按来源域看 KEEP 率：

| 域 | 已测 | KEEP | KEEP 率 |
|---|---|---|---|
| research_scout | 57 | 4 | **7.0%** |
| serving | 9 | 2 | **22.2%** |
| comm / freeform / kernel_switch | 5 | 0 | 0% |
| 合计 | 71 | 6 | 8.5% |

`research_scout` 的命中率低于 `serving`，不高于。**「scout 提案质量更优」这一替代解释被数据反驳**；`serving` 的 n=9 过小，不足以反向断言其质量更优。诚实表述：没有证据表明质量差异能解释 10 倍的转化率差距，现有证据弱指向相反方向。上线后仍需按域 / gap severity / 提案类型分层复核（见 §7）。

### 1.3 本次改动不承诺提高 KEEP 率

已 benchmark 的 specialist 提案 KEEP 率为 8.5%（6/71），全局 explore KEEP 率为 8.7%（129/1487），两者基本相同。`benchmark_lane` 容量恒为 1、变体串行，单位时间可测量的变体数是固定的。把消费率从 13% 提到更高，改变的是**采样哪一批**，不是采样总量。

本次改动买到的是：specialist 支出不再大比例落空（P1-1 的成本面），以及候选覆盖面。KEEP 率只作为护栏指标 —— 若 grid 变宽而 KEEP 率下滑，说明在用稀缺串行 lane 换噪声。

## 2. 设计决策

| # | 决策 | 理由 / 取舍 |
|---|---|---|
| 1 | 只渲染，不自动物化 | `benchmark_lane` 容量恒为 1、串行、积压 6.6:1。自动灌入不产生容量。 |
| 2 | 队列渲染时派生，不新增 SharedState 字段 | `specialist_rounds`（已持久化，`_SPECIALIST_ROUNDS_CAP=200`）与 `explore_search` 已是权威来源。 |
| 3 | 排序 = gap severity → 轮次新旧 | `ensemble_scores` 在 27 个会话 221 个 round 中命中 0 次（`writeback.py:2229-2249` 需要 `_proposal_scorer`，实际从未配置）。self-reported `confidence` 与「自报数字会被剥离」的既有约定冲突。不复活打分器。 |
| 4 | 只收可执行提案（有 args / envs / 移除控制之一） | 带 patch 的已有闭环路径；纯 research 项没有可供重启应用的内容。 |
| 5 | 吸收 research-scout SEED 块的提案半段 | 否则 scout 提案渲染两遍。 |
| 6 | 已测指纹只做渲染侧过滤，不做准入门禁 | `actions/executors/explore.py:11-12` 明文写着历史结果 `is evidence, never an eligibility gate`。反转它需要独立改动。 |
| 7 | **只在 EXPLORE 阶段渲染**，其余阶段仍累积 | `explore` 仅在 EXPLORE 阶段 `allowed` 且 `llm_proposable`（其余五个阶段均为 False）。在无法消费的阶段渲染是纯噪声。 |
| 8 | 渲染器放 `state/_shared_state/render.py` | 三个输入（`specialist_rounds` / `explore_search` / `gaps`）与 base controls 投影（`stack_base_params(current_best)`）全在 SharedState 上。 |
| 9 | 上限 12 条 + 溢出计数行 | 见 §2.2 的实测深度。 |
| 10 | 只保留当前 macro_cycle 的提案 | 已三次确认。代价见 §6。 |
| 11 | 不新增 context pull 工具 | 溢出部分不可达，仅以计数形式可见。 |
| 12 | `prompt_builder.py` 与 `orchestration.md` 同步改，各两处 | 见 §3 Commit 3。 |
| 13 | 三段 commit | 第 2 段是可独立回退的核心。 |
| 14 | explore grid target 4 / max 6，只写提示词、不加门禁 | 见 §2.3。与 `proposal_set` 的 target 2 / max 4 配套。 |
| 15 | **提取共享 helper 统一「规范化 + 有效指纹 + controls」**，执行器一并改为调用它 | 见 §2.1。这是队列过滤与执行器身份计算保持同构的唯一保证。 |
| 16 | 未测量的 `tested` 条目（`FAILED` / `KILLED_OVERTIME`）按已测隐藏 | 已确认。代价见 §6。 |
| 17 | 渲染异常输出不可用标记，不静默消失 | 见 §2.4。 |

### 2.1 共享 helper：与执行器同构

`actions/executors/explore.py:989-1006` 是变体身份的权威实现：

```python
identity_remove_args = list(dict.fromkeys(base_remove_args + to_str_list(identity_controls.get("remove_args"))))
identity_unset_envs  = list(dict.fromkeys(base_unset_envs  + to_str_list(identity_controls.get("unset_envs"))))
if base_args_mode == "replace":
    identity_controls["args_mode"] = "replace"
fp = canonical_fingerprint(gv.extra_server_args, gv.extra_envs, **identity_controls)
```

即：**变体自身的 args/envs，加上「base 控制 ∪ 变体控制」的合并控制集**。渲染侧若不折入 base controls，指纹与 `tested` 的键不同源，过滤对带控制的条目完全失效。实测 `tested` 中 25/148（16.9%）带 `remove_args`、4 条 `args_mode=replace`，不是边缘情况。

base controls 的唯一投影点是 `state/shared_state.py:144` 的 `stack_base_params(current_best)`（`_STACK_BASE_FIELDS` 把 `current_best.remove_args` / `unset_envs` / `args_mode` 映射为 `base_remove_args` / `base_unset_envs` / `base_args_mode`）。渲染器直接调用该函数，不复制映射。

新建 `orchestrator/actions/executors/_proposal_identity.py`，导出三个纯函数：

- `is_executable(proposal) -> bool` —— 有 `extra_args`/`extra_server_args`、`extra_envs`、或移除类控制之一
- `normalize(proposal) -> dict` —— 统一 `extra_args`/`extra_server_args` 别名、列表参数经 `to_str_list`、`args_mode` 小写化
- `effective_fingerprint(proposal, *, base_remove_args, base_unset_envs, base_args_mode) -> str` —— 逐字复刻上述合并逻辑

消费方：渲染器、research-scout 去重、`_maybe_materialize_mn_explore` 的谓词，以及**执行器自身**。

执行器改为调用 helper 属于对热路径的机械重构，是本计划有意扩大的范围 —— 不这样做，两处逻辑会再次漂移，而漂移是静默的。为防止指纹变化导致既有 `tested` 台账在 resume 时失效，配一个 golden 测试：对一个覆盖矩阵（无控制 / 仅变体控制 / 仅 base 控制 / 两者叠加 / `args_mode=replace` / removal-only）断言指纹等于重构前捕获的硬编码值。

### 2.2 队列深度（应用全部过滤后实测）

过滤链：当前 cycle → 可执行 → 有效指纹去重 → 剔除 `tested`。

| | |
|---|---|
| 会话数 | 27（其中 3 个队列为空） |
| 深度 mean / median | 10.4 / **10** |
| p75 / p90 / max | 17 / 25 / 31 |
| 超过 12 条的会话 | **11 / 27** |

12 条上限覆盖中位数，在 41% 的会话触发截断。

### 2.3 grid 尺寸约束的由来

曾经存在一条门禁，在 2026-06-09 的 #486 中被删除。

删除前 `inference_optimizer/orchestrator/policy.py` 的 `_validate_explore_grid_size` 在 delegate 与 propose 两条路径上都执行（`:909` / `:1024`）：`specialist:*` 上限 `min(research_lane_ceiling(), research_lane_capacity)`、回落 `MAX_SPECIALIST_SOURCED_EXPLORE_VARIANTS = 1`（rule `explore_specialist_grid_max_one`）；`dynamic` 硬上限 1（rule `dynamic_sourced_variant_cap_exceeded`）；`llm_direct` / `default_grid` 不受限。

#486（`loosen_plan P1_05`）的删除理由是 breadth 应由资源导出的约束（`research_lane` / GPU pool 租约）承担。**该论证对 explore grid 不成立**：`explore.yaml` 的 `requires_lanes` 是 `server_lifecycle` + `benchmark_lane`，不含 `research_lane`；`benchmark_lane` 容量恒为 1、串行。

删除后唯一生效的是 `_grid_runner.py:1395-1407` 的会话截止跳过（`remaining_sec < variant_timeout_sec`，默认 7800s），属事后截断且单节点无排序，砍掉的是排在后面的而非价值最低的。审计集 `variants_per_round` 的 max = 20 即此形状的证据。

被删掉的那条 hint 原文是 `defer the runners-up beyond the cap to a subsequent explore round` —— 它预设了一个从未实现的队列。cap 与队列本应成对；先只做 cap、后又删 cap，两头落空。

只写提示词不恢复门禁：#486 是一次有意放宽，用门禁反转它需要独立改动（同决策 #6 原则）；且提示词约束已被证明有效 —— `proposal_set` 的 ceiling 在 #1056 加上 Section 1 提醒后，188 次运行 0 次超标。

物理依据：`benchmark_lane` 容量 1、单变体中位 13 分钟 —— 4 个变体约 1 小时，6 个约 1.3 小时。

### 2.4 渲染格式

每条一行，字段带前缀记号，空字段省略：

```
=== Untested proposals (current cycle) ===
<块头：排序依据、已排除的口径、ATOMIC 的处置要求>
• <name> [<domain>·<sev>] {ATOMIC} {+args=…} {+envs=k=v,…} {-args=…} {-envs=…} {mode=replace} why=<…>
(+N more not shown)
```

必须携带 `remove_args` / `unset_envs` / `args_mode` / `atomic`，理由是实测分布：

| 字段 | 命中 / 801 | 占比 |
|---|---|---|
| `atomic: true` | **309** | **38.6%** |
| `remove_args` | 101 | 12.6% |
| `unset_envs` | 7 | 0.9% |
| removal-only（无 args 无 envs） | 5 | 0.6% |

`atomic` 占 38.6%，而 `orchestration.md:228-230` 要求 `Dispatch it verbatim as one explore variant — never split, drop, or re-author`；行内不带该标记，Orchestration 会对近四成提案做重新推导，而 `atomic` 的定义正是拆开即失效的耦合集。

removal-only 只有 5 条但性质更坏：例如 `drop-prefix-caching`（`remove_args=['--enable-prefix-caching']`），若只渲染 args/envs 会输出两个空字段，看起来像空操作。

`why=` 取提案 `reason` 前 80 字符。每行总预算约 240 字符；12 行约 2.9k 字符（约 750 token），且只在 EXPLORE 阶段的每一轮支付。

### 2.5 异常与截断的可见性

- 渲染抛异常：输出一行不可用标记（含原因短语），不静默消失。全仓其它块的房规是 `except: log.exception(); block = ""`，此处有意偏离 —— 队列若无声消失，Orchestration 会以为队列为空。
- 溢出：输出 `(+N more not shown)`。
- 不输出 invalid 计数：无效提案是上游缺陷，把它的计数搬进提示词是把噪声当信号。
- 队列真为空：不渲染块（与「渲染失败」通过标记区分）。

## 3. 改动清单

### Commit 1 — `fix(specialist): tighten the proposal-set target to 2 and the ceiling to 4`

- `orchestrator/prompts/specialist_prompt_builder.py`
  - Section 1 IDENTITY（约 `:1033-1034`）：target 4 → 2，ceiling 6 → 4
  - Section 8 字段契约（约 `:2229-2237`）：同上
- `inference_optimizer/tests/test_critic_verdict_map.py`：断言跟随新措辞
- 冷启动分支（约 `:1509-1520`）的 `1–2 most conservative defaults` 不动，与 target 2 自洽

背景：以 2026-08-01 的 #1056 为界，之前 37 次运行 mean 5.35、11 次超过 cap（最多 12 个）；之后 188 次运行 0 次超标，但 `== 6` 占到 24.5% —— cap 从上限变成了锚点。

### Commit 2 — `feat(state): surface the untested specialist proposals as a ranked queue`

1. 新建 `orchestrator/actions/executors/_proposal_identity.py`（§2.1 的三个纯函数）
2. `actions/executors/explore.py:989-1006` 改为调用 `effective_fingerprint`；`phases/explore.py:1499-1501` 的谓词改为调用 `is_executable`
3. `state/_shared_state/render.py` 新增 `to_untested_proposals_summary(*, max_entries: int = 12) -> str`：
   - 取 `specialist_rounds` 中 `int(entry.get("cycle") or 0) == self.macro_cycle` 的轮次
   - 经 `is_executable` 过滤，`normalize` 规范化
   - 用 `stack_base_params(self.current_best)` 取 base controls，经 `effective_fingerprint` 计算指纹；剔除已在 `explore_search["tested"]` 中的，并做组内去重
   - 经 `gap_canonical_id` 关联 `gaps[]` 取 severity；gap 已被裁剪时 severity 记为未知并排在同级之后
   - 按 severity 降序、轮次新旧降序排序
   - 按 §2.4 渲染，超出 `max_entries` 输出溢出计数行
   - 空队列返回 `""`
4. `loop/conversation.py`：在 orchestration 分支、`push_full` 判断之外、**且仅当 `phase == "EXPLORE"`** 时接入 `=== Untested proposals (current cycle) ===`；渲染异常按 §2.5 输出标记
5. `loop/conversation.py:1188-1243` `_research_scout_seed_block`：删去 `Untested executable proposals:` 半段及其自建的指纹去重，保留 `Findings:` 与 `residual_questions`

`render.py` 的既有约定是方法内延迟导入（见 `:36` / `:106` / `:163` / `:332`），helper 与 `stack_base_params` 照此引入。

### Commit 3 — `feat(orchestration): point the grid author at the untested-proposal queue`

队列来源与 grid 尺寸都是「如何组一轮 grid」的指令，落在同一批文件同一段文案上，合并以免同处改两次。

- `orchestrator/prompts/prompt_builder.py:593-594` 第 (d) 条：现文案 `specialist proposal_set — when an explore round just finished, the proposal_set drives the next explore grid` 描述的机制并不存在。改为指向队列块。
- `orchestrator/prompts/prompt_builder.py:441-470` `_format_grid_injection_hint`：`explore` 分支补 target 4 / max 6，措辞对齐 Section 8 的形式（给目标值、给越过目标值的判据、说明代价）。
- `orchestrator/prompts/prompt_builder.py:689-717` `IDEA GENERATION` 块：该块列举五种造想法的方法却不给数量，是当前唯一系统性推高 grid 的地方；补一句收敛到 target 4。
- `orchestrator/prompts/orchestration.md:186-190` **Decision priority** 第 (b) 项：同样含 `specialist proposal_set` 的陈旧引用，改为指向队列块。
- `orchestrator/prompts/orchestration.md` EXPLORE 段：补 grid 的来源与尺寸，措辞与上述一致。

不改 `policy/gate.py`，不恢复 `_validate_explore_grid_size`。

## 4. 明确不做

- 不复活 `ensemble_scores` 打分器（`orchestration.md:232-238` 仍在向 LLM 广告该块，属另一条待办）
- 不反转 `explore.py:11-12` 的非门禁设计
- 不恢复 #486 删除的 `_validate_explore_grid_size`
- 不新增 context pull 工具
- 不新增单节点自动物化
- 不改动 patch / `integrate_patch` 路径
- 不在提示词中输出 invalid 提案计数

## 5. 边界语义

- `specialist_rounds` 条目缺 `cycle` 字段：读作 0。实测 221 个 round 缺失 0 个（`record_specialist_round` 在写入时 `setdefault`），此为纯防御。
- `gap_canonical_id` 为空、或对应 gap 已从 `gaps[]` 裁剪：severity 记为未知，排在同级之后，不丢弃该提案。
- `current_best` 为空或缺 `remove_args` / `unset_envs` / `args_mode`：`stack_base_params` 省略该键，helper 按空集处理，与执行器的 `to_str_list(params.get(...))` 行为一致。
- 指纹是 stack 相关的：base controls 变化会使同一提案得到不同指纹，因此 `tested` 过滤是近似判据而非精确判据。这是执行器既有语义，本计划不改变它。

## 6. 已知取舍

**cycle 过滤的代价（已三次确认）。** 提案的 cycle 分布：cycle 0 有 637 条（79.5%）、cycle 1 有 94 条、cycle 2 有 40 条、cycle 3+ 共 30 条。`research_scout` 30 轮中 29 轮、`static_recon` 24/24、`framework_rewrite` 11/11、`enablement` 8/8 只在 cycle 0 运行。27 个会话中 10 个最终 `macro_cycle > 0`。因此在这 10 个会话中，`cycle_reloop` 一旦触发，队列中约 80% 的内容（含 `research_scout` 的几乎全部产出，即转化率最高的来源）被丢弃，而丢弃原因恰是「尚未被测过」。独立审核将此列为首项阻塞；此处按决策执行并记录。

**未测量条目按已测隐藏的代价（已确认）。** `tested` 的 148 条中，26 条（17.6%）`tput` 为 None —— `FAILED`（reason `warmup_failed`，服务未起来）22 条、`KILLED_OVERTIME`（墙钟超时被杀）4 条。两者均非测量结果，按已测隐藏意味着约六分之一的候选未经 benchmark 即被永久退休，且在提示词中不可见。

**每轮渲染的代价。** 在 EXPLORE 阶段的每一轮重复出现并累积进对话历史。上限 12 条、cycle 过滤、阶段收窄三者共同约束体量，估算约 750 token/轮。

**溢出不可达。** 无 pull 工具时，超出 12 条的部分在该轮完全不可达，仅以计数形式可见；实测 11/27 会话会触发。

**grid 尺寸无强制。** target 4 / max 6 只是提示词约束，`_grid_runner` 的会话截止截断仍是唯一硬边界。若观测到 max 被系统性突破，恢复门禁是独立改动，且届时应与队列成对存在。

**执行器热路径重构。** Commit 2 改动 `explore.py` 的身份计算调用点。golden 指纹测试是防止 `tested` 台账在 resume 时失效的唯一保障。

## 7. 验收

### 单测

指纹与 helper：

- golden 指纹矩阵：无控制 / 仅变体控制 / 仅 base 控制 / 两者叠加 / `args_mode=replace` / removal-only，断言等于重构前捕获的硬编码值
- `is_executable`：`extra_args` 与 `extra_server_args` 两种别名均识别；removal-only 判为可执行；纯 research 项判为不可执行
- `effective_fingerprint` 与执行器身份计算在同一输入上逐位相同

队列渲染：

- cycle 过滤：非当前 cycle 的提案不出现；缺 `cycle` 字段读作 0
- tested 剔除：使用**有效指纹**（含 base controls）的条目被正确剔除；不含 base controls 的朴素指纹会漏剔，需有一条断言覆盖该回归
- 可执行过滤：纯 research 提案不出现
- 字段完整性：`remove_args` / `unset_envs` / `args_mode=replace` / `atomic` 全部出现在行内；removal-only 提案渲染出 `-args=` 而非空字段
- 排序：high severity 在 medium 之前；同 severity 下新轮次在前；gap 缺失的排在同级之后
- 截断：超过 12 条时渲染 12 条并输出溢出计数
- 空队列返回 `""`；渲染异常输出不可用标记而非空串
- 阶段收窄：EXPLORE 之外的阶段不注入该块（PRELUDE / FRAMEWORK_AGENT / KERNEL_AGENT / SWEEP / CLOSE 各一条）
- SEED 与 DELTA 两种轮次均注入
- scout 块：不再输出提案，`Findings` 与 `residual_questions` 保留

提示词：

- `_format_grid_injection_hint` 与 `IDEA GENERATION` 块同时出现 target 4 与 max 6，措辞一致
- `prompt_builder.py` 第 (d) 条与 `orchestration.md` Decision priority 第 (b) 项均不再含陈旧的 `proposal_set drives the next explore grid` 表述

### 上线后回归指标

- 提案 → grid 转化率，基线 13.1%
- **分层转化率**：按域 × gap severity × 提案类型（config-only / 含 removal / atomic）分别统计，用以复核 §1.2 的因果判断，而非只看全局数字。基线：非 `research_scout` 域 4.2%
- `proposal_set` 长度分布中 `== ceiling` 的占比，基线 24.5%（当时 ceiling = 6）
- `variants_per_round` 分布，基线 mean 4.34 / median 4 / max 20；关注 max 是否落回 6 以内，以及是否出现向 6 聚集的锚定效应
- grid 保真度：进入 grid 的 `atomic` 提案是否被逐字下发（未拆分、未改写）
- KEEP 率，基线 8.7%（全局）/ 8.5%（specialist 来源）。**作为护栏而非收益指标**：本次改动不承诺其提升，但若 grid 变宽而 KEEP 率下滑，说明在用串行 lane 换噪声
