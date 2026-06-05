# P2_12 — 删 `wait_for_auto_roofline` deny(资源靠 GPU 租约);保留内部 enqueue

- **Phase**: P2 · **风险**: 中 · **依赖**: 无 · **后继**: 无

## 目标

analysis(roofline/profile)在飞时,当前**DENY** 所有 gated 动作(`specialist`/`explore`/kernel-owned),理由 `wait_for_auto_roofline`,让 LLM 干等。改为:允许 LLM 基于**当前(可能略旧的)** `analysis.md` 继续行动;真正的资源争用(并发 Magpie / GPU)由 **GPU 租约**兜底,不需要用 deny。

## 不变量 vs 策略判定

- roofline/profile 由 Coordinator 拥有调度 + 内部 enqueue = **INVARIANT**(资源,保留)。
- "analysis 在飞时不许动" = STRATEGY(怕用旧数据决策)。但用稍旧 analysis 决策只是效率问题;**资源冲突**由 GPU/lane 租约保证,无需额外 deny。删除。

## 改动清单(删除优先)

### 1. 删 pending deny(`coordinator.py`)
- 删 `_auto_roofline_pending_denial`(2711–2764)与 `_roofline_denial_for_action`(2766–2772)的 **deny** 行为;`_ROOFLINE_GATED_ACTIONS`(131–139)若仅服务此 deny 则删。
- 删调用点对应分支(`_handle_propose_action` / `_handle_delegate` / materialize 中引用 wait_for_auto_roofline 处)。

### 2. 处置 deferred-proposal 队列(`coordinator.py`)
- `_defer_approved_proposal_for_roofline`(2774–2809)与 materialize 处的 defer(6135–6142):删除"因 roofline pending 而延后"逻辑——proposal 直接 materialize。
- resume 端 deferred 队列重建(见测试 `test_resume_deferred_proposals`):随之简化/删除。

### 3. 保留(不动)
- 内部 analysis enqueue:`_maybe_enqueue_prelude_initial_analysis_after_baseline`(2599–2641)、`_maybe_enqueue_watermark_roofline`(1987–2020)、`_enqueue_internal_analysis_task`(2643–2709)、`auto_roofline_pending_task_id` 生命周期 —— **保留**(Coordinator 仍自动跑 analysis)。
- GPU/lane 租约(`resource_lock.py` / `gpu_pool.py`)= 资源不变量,**保留**;它们才是防并发 analysis 与其它 GPU 任务撞车的真实机制。

### 4. advisory 化(可选)
- 若希望提示"分析正在刷新",在 prompt 注入一行中性事实(`analysis refreshing, current snapshot may be stale`),不阻断动作。

## 连带测试

| 文件 | 函数 | 动作 |
|---|---|---|
| `test_propose_action_blocks_on_pending_roofline.py` | `test_propose_action_blocks_explore_while_roofline_pending`(104) | **删除/反转**:不再 block |
| `test_propose_action_blocks_on_pending_roofline.py` | `test_propose_action_passes_through_non_gated_action`(153) | 删除(无 gated 概念) |
| `test_specialist_block_on_watermark_roofline.py` | `test_pending_denial_blocks_every_gated_action`(305) | **删除** |
| `test_specialist_block_on_watermark_roofline.py` | `test_pending_denial_passes_through_when_no_task`(329)、`test_pending_denial_clears_field_*`(336–358) | 字段清理逻辑若保留(pending id 生命周期)则保留相应部分;否则删 |
| `test_materialize_blocks_on_pending_roofline.py` | 全部 | 删除/反转 |
| `test_resume_deferred_proposals.py` | 全部(127–199) | **删除**(deferred 队列没了);确认 resume 仍能正常重放普通 proposal |

## 验证
- analysis 在飞时,explore/specialist/kernel request 不再被拒;并发由 GPU 租约串行化(同一 GPU 上 analysis 与 bench 不会同时跑)。
- 内部 analysis 仍在 PRELUDE 与 +10% watermark 自动触发。
- resume 后无 deferred 队列依赖;普通 proposal 重放正常。
- 烟测:确认不会出现两个任务同时抢同一 GPU(租约日志)。

## 回退
- 恢复 deny、defer、deferred 队列与测试。

## 残留风险
- 中。删 deny 后,LLM 可能基于上一份 `analysis.md` 决策(数据稍旧)——这是可接受的效率折中,且 watermark 机制保证 +10% 后会刷新。**关键确认**:GPU 租约确实能阻止 analysis 任务与 bench 任务并发抢同一 GPU(否则需保留一个**资源级**串行,而非 strategy deny)。落地前验证租约覆盖 analysis 任务。
