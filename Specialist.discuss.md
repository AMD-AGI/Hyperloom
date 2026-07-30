<!-- SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# Specialist / Orchestration 上下文机制改造计划（已决策，最终版）

> 状态：**决策已定，正在实施。** 分支 `feat/zgong/explore-opt-11` 已创建/切换/push。
> 总目标：**删除冗余/封装代码，省本地实现。允许污染、允许高风险、允许删测试。**
>
> **最终范围：只做第 1、2 项；第 3 项放弃（见下）。**
> 1. **宁可信息污染，不可信息不全** → inbox 去掉尾窗截断，渲染全量。**但保留 Critic 的 `_augment_critic_inbox_with_pending`**（见 §1 纠正）。
> 2. **删除紧急 bypass 逻辑，让 Hyperloom 自由发挥** → 移除 98% emergency / anti-thrash floor / degenerate 强制 fallback 三个兜底分支。**保留正常 soft/hard 水位触发与 self-summary compaction 主体**（见 §2 收窄）。
>
> ### 第 3 项（用 `/compact` 替换 compaction）——放弃，无实施空间
> 调研发现 `orchestration_memory` 承载的 **`next_cycle_directive`**（跨 macro-cycle 战略指令：SWEEP→EXPLORE 回环时告诉下一轮攻哪个瓶颈、breadth vs depth、优先 specialist 域）由 self-summary 产出（`orchestration_memory.py:158,353`）、由 `explore.py:234` 每个新 cycle 读取注入。`/compact` 是 CLI 内部通用摘要，**不产出这个结构化 directive**。删掉 self-summary = 删掉跨 cycle 战略传递这一**正常运行时核心机制**，不是可选的"推理状态恢复"。因此第 3 项无实施空间——compaction 主体（self-summary + 结构化 memory + `next_cycle_directive`）**全部保留**，`orchestration_memory.py` **不删**。`/compact` 相关的 recovery / session 持久化 / 整文件删除等设计一并作废。

---

## 1. inbox 渲染全量（第 1 项：信息优先）

### 决策
去掉尾窗截断，渲染全部未读事件。接受 prompt 变大 / 成本上升。

### 改动点
- `loop/conversation.py:551` `rendered = list(msgs[-20:])` → `rendered = list(msgs)`。
- `_context_inbox_reader`（`:156`）的 `msgs[-40:]` → `msgs`（on-demand 拉取也给全量）；docstring 里 "last 40" 一并更新。

### 纠正：Critic 的 `_augment_critic_inbox_with_pending` **必须保留，不能删**
原计划以为全量渲染后补投冗余。**错。** cursor 是**每处理一个 intent 就 `_cursor_advance_to_latest` 跳到最新**（`intent_router.py:100`）。所以未裁决的 proposal 会被 cursor 跳过 → 不在 `replay_for(after_seq=cursor)` 结果里 → 全量渲染也捞不回。**只有** `_augment_critic_inbox_with_pending`（`:567-606`）从 durable `pending_proposals` 拉回。删它 = Critic 丢待裁决 proposal = 正是"信息不全"。**保留该方法及 `:555-556` 调用点。**

### 已知局限（接受）
去掉 `[-20:]` 只在**单 tick 内积压很多事件**时才有区别（cursor 尚未推进的那批）。跨 tick 已被 cursor 消费的历史，不因去尾窗而回来。要真正"全历史"需改 cursor 语义（不推进 / 从 seq=0 读），那会 token 爆炸且改动面大——**本次不做**，只去尾窗截断。第 1 项因此是"把单 tick 积压的信息全给出来"，不是"重放全历史"。

---

## 2. 删除紧急 bypass（第 2 项：让 Hyperloom 自由发挥）

### 收窄说明
第 3 项放弃后，compaction 主体（self-summary + 水位触发）**保留**。第 2 项只删**兜底/防抖分支**，不动正常触发。原计划"删水位判定/删常量/删 CheckpointPolicy"**作废**——那些是正常 compaction 必需的。

### 删除清单（`loop/maintenance.py`，`_maybe_checkpoint_orchestration`）
1. **anti-thrash floor + emergency ceiling**（`:200-222` 整块 `suppress_token_trigger` 逻辑）——含 `_checkpoint_min_tick_gap`、`emergency_ceiling`、`in_emergency`、`token_due`。删除后 token 水位到 soft/hard 就直接触发，不再抑制。
2. **degenerate 强制 fallback 路径**（`:286-289` 的 Path 2：`if degenerate and hard: parsed = deterministic_memory_fallback(...)`）——"近窗必压、用确定性 fallback"是兜底。删除后 degenerate 一律走 Path 1（跳过本轮、下轮再压）。
3. **`hard` 变量与 `is_hard_compaction` 调用**（`:199`）——`hard` 仅服务上述两个兜底分支；删兜底后 `hard` 无消费者，一并删。`:227`（`not hard`）、`:258`（`degenerate and not hard`→简化为 `degenerate`）、`:288`、`:334`（observation 里 `hard_compaction` 字段）同步清理。

### 保留（正常 compaction 主体）
- `should_checkpoint`（soft 水位 / phase / tick / minute / char cadence）——**全部保留**。
- soft 水位触发（`CheckpointPolicy.context_token_soft`）——保留。
- self-summary turn + `parse_checkpoint_reply` + Path 1（degenerate 跳过）+ Path 3（正常压缩）+ `build_memory_record` + reseed——保留。
- `next_cycle_directive` 全链路——保留。

### 连带清理（`state/orchestration_memory.py` + `loop/coordinator.py`）
- `is_hard_compaction`（`orchestration_memory.py:113-125`）删除（无消费者）。
- `context_token_hard` 字段（`:74`）删除；`CheckpointPolicy` 只留 soft。
- `DEFAULT_CONTEXT_TOKEN_HARD_FRACTION`（`:31`）+ `__all__` 导出（`:462`）删除。
- `deterministic_memory_fallback`（`:252`）+ `__all__` 导出（`:471`）删除（仅 Path 2 用）。
- `coordinator.py`：`_hard_frac` / `INFERENCE_OPTIMIZER_CTX_HARD_FRACTION` 读取（`:678-684`）、`context_token_hard=` 传参、`_checkpoint_min_tick_gap`（`:691`）删除。`_consec_degenerate_ckpt`（`:693`）**保留**（Path 1 仍用它记连续 degenerate 并升级 observation）。

### 保留 hard fraction 的语义决策
删掉 hard 水位后，token 到 soft 就压；不再有"soft 抑制、hard 才强压"的两级。soft 触发是正常 compaction，degenerate 时 Path 1 跳过、下轮再试——这就是"自由发挥"：不再有近窗强制兜底，若 compaction 连续 degenerate 且 token 持续涨，最坏撞 provider 上限报错（用户已接受）。

---

## 3. 执行顺序

1. **第 1 项**：`conversation.py` 去尾窗（`:551`、`:156`），保留 Critic augment。
2. **第 2 项**：
   a. `maintenance.py` 删 anti-thrash/emergency 块（`:200-222`）、Path 2、`hard` 变量与其消费点。
   b. `orchestration_memory.py` 删 `is_hard_compaction`、`context_token_hard`、`DEFAULT_CONTEXT_TOKEN_HARD_FRACTION`、`deterministic_memory_fallback` 及对应 `__all__`。
   c. `coordinator.py` 删 hard-fraction 装配与 `_checkpoint_min_tick_gap`。
3. 删对应测试（授权删测；先 grep 定位）。
4. `grep` 确认删除符号无残留引用 → `ruff check` → 相关 `pytest` 子树。

## 4. 验收标准
- inbox 对所有角色渲染单 tick 全量，无 `[-20:]`/`[-40:]` 尾窗。
- **Critic augment 保留**，待裁决 proposal 不丢。
- `_maybe_checkpoint_orchestration` 无 emergency ceiling / anti-thrash floor / degenerate 强制 fallback；soft 水位 + cadence 正常触发 compaction。
- `next_cycle_directive` 跨 cycle 传递不受影响。
- `grep` 确认 `is_hard_compaction` / `context_token_hard` / `DEFAULT_CONTEXT_TOKEN_HARD_FRACTION` / `deterministic_memory_fallback` / `emergency_ceiling` / `_checkpoint_min_tick_gap` 已无引用。
- 无死代码、无死符号、无被跳过的旧路径。

## 5. 残留风险（实施后仍存在，已知并接受）
- **R-A（第 1 项局限）**：去尾窗只解决单 tick 积压；跨 tick 被 cursor 消费的历史仍不重放。若后续发现关键信息仍丢，需另议 cursor 语义（本次不做）。
- **R-B（第 2 项失去兜底）**：删 emergency 后，soft 触发的 compaction 若连续 degenerate + token 持续涨 → 无近窗强制兜底 → 最坏撞 provider 上限硬报错。这是"自由发挥"的代价，用户已接受。
- **R-C（prompt 变大）**：全量 inbox + 保留 self-summary → 单 tick prompt 更大 → soft 水位更早触发 → compaction 更频繁（每次 ~一个 turn 开销）。属预期，非故障。
- **R-D（gate CORE_STATE_FIELDS）**：`gate.py:616-619` 仍保护 `orchestration_memory` 字段——第 3 项放弃后该字段保留，**无需改 gate**。（若误删字段会触发 gate 校验失败，本次不删故无此风险。）
