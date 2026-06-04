# Phase E — 测试整理

## 目的

删除"功能增加过程中堆积的细粒度单测"(断言私有实现、随 Phase B 已失效的),合并小测试文件/小测试逻辑。
**护栏 keep-list(Phase 0)不在删除范围**——它们是功能等价判据,只可精简不可删。

> 放在最后:此时代码结构已定型,能准确判断哪些测试测的是"对外行为"(留)vs"私有实现"(删/合)。

## 原则

- **保留**:覆盖**对外行为/契约**的测试(护栏 + 同类)。
- **删除**:测私有 helper、内部中间状态、已被 Phase B 改没的实现细节、重复覆盖同一行为的冗余测试。
- **合并**:同一模块的多个小测试文件 → 一个;同一测试文件内的碎测试 → 参数化/合并。
- **功能等价 ≠ 单元等价**(主旨):不追求测试数量,追求"对外行为有护栏覆盖"。

## 步骤文件

- [`01_test_inventory.md`](01_test_inventory.md) — 盘点全量测试,分类(护栏/行为/私有/失效)。
- [`02_consolidate_and_delete.md`](02_consolidate_and_delete.md) — 合并与删除操作。
- [`conduct.md`](conduct.md) — 行为准则。

## 入口标准

- Phase D 出口达标(结构定型)。

## 出口标准

- [ ] 失效/私有实现测试已删。
- [ ] 小测试文件已合并(测试文件数显著下降)。
- [ ] 护栏 keep-list 仍全绿且仍存在。
- [ ] 剩余测试**全绿**(`pytest` 整体跑通)。
- [ ] 测试总 LOC 下降。

## 进度记录

### Step 01 — 盘点与分类(`test_classification.txt`)

- 交叉对照 Phase 0 keep-list + 扫描全部 176 个 `inference_optimizer/tests/` 的"对外可观测信号" token,逐文件打标。
- 结论:**GUARDRAIL 8(主包)+ BEHAVIOR ~168 + PRIVATE/STALE ≈ 0**。
  - STALE:grep 所有 A/B/D 删掉的特性(silent-tick / proxy / last_select_kernels 字段 / cortex_kb_constants /
    kb-flusher / IDLE_CLOSE_TICKS / no-more-leverage run-loop net / intent_parser 路径)——残留全是**活行为**
    (`params_no_promote_streak` 作为 LLM 可见 fact 保留;`skip_to_sweep`/`no_more_leverage` 仍是 phase_state 活路由)。
  - PRIVATE:套件早先已做过一轮(`a6b39547` 删重复 prompt asset 测试、`94a87099` 删私有 prompt formatter 测试),
    A/B 又把 proxy/silent-tick 单测随功能内联删除。无成片可删的 PRIVATE 文件。

### Step 02 — 合并(`2b51f769`)

- 唯一安全合并:`framework_paths` 三件套 → 一个 `test_framework_paths_units.py`
  (`test_apply_kernel_patch_roots.py` + `test_framework_source_roots.py` 折入,均测 `orchestrator.framework_paths`)。
- 覆盖不变(6 个测试函数全部保留,49 passed),测试文件数 **176 → 174**。
- 未做删除:按 conduct"删的只能是测私有实现/已失效的、禁止删后降低对外覆盖",套件已是行为导向,
  无可安全删除项;其余小文件各自单一主题,强行合并属"搬家"非降复杂度,故不做。

### 出口标准

- [x] 失效/私有实现测试已删(早先轮次 + A/B 内联已完成;本轮复核无残留)。
- [x] 安全的同模块小文件已合并(framework_paths 三合一)。
- [x] 护栏 keep-list 仍全绿且仍存在(356 passed 主包广扫含全部 8 个主包护栏)。
- [x] 剩余测试全绿。
- [x] 测试文件数下降(176→174)。

## 最终验收(主计划 §6,整个清理收尾)

- **总 LOC / 最大单文件**:较 Phase 0 显著下降——coordinator 12664→12119、cli ~6.5k→5436;
  god-module 拆为 coordinator_helpers / cli_executors / cli_kb / cli_backends / protocol.intent。
- **退役 action 引用 = 0**:Phase A 已清(`params_no_promote_streak`/`last_select_kernels` 是**活字段/迁移名单**,非退役 action)。
- **文件数净增 ≤ 个位数**:Phase B +4、Phase D +1−1、Phase E −2 → 合计 **+2**。
- **护栏全绿;CLI/产物金标准形状不变**:`golden_breakdown_keys` 逐键不变;CLI flag 面未动;envelope 形状不变;同版本 resume 对称。
- **循环引用 = 0**,依赖严格单向(Phase D 复核)。
- **净行数全程为负**:每个 commit `git diff --stat` 净删(拆分用 docstring 顺手压缩抵消搬运,合并去重)。
