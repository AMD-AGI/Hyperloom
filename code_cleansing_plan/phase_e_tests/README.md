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

测试文件数:273 → 264(-9)。范围内测试总 LOC:103493 → 103255(-238)。

入口校验:5 个子系统全量跑一遍——io 3620 / robustness 703 / critic 249 /
framework 191 全绿;kernel-agent 有 15 个**既有**失败(子进程工具依赖
GPU/aiter 源/磁盘 fixture,在本沙箱缺失),非回归、未触碰。故无 STALE 红灯
可删——Phase A–D 退役死代码时没留下孤儿测试。

执行(每项一个 commit):
- 分类盘点 `test_classification.txt`(GUARDRAIL/BEHAVIOR/PRIVATE/STALE)。
- 合并 `sources_local_probe` 家族 7→1(81 测试,全部同打 `local_probe`)。
- 合并 `local_health` 信号三件套 3→1(52 测试,共享 `_ctx`)。
- 合并 `SharedState` 单测三件套 3→1(34 测试),顺带退役 failure-log 里的
  `backends`/`params` 字符串标签。

出口校验:
- [x] 无 STALE/PRIVATE 红灯(suites 本就全绿,未盲删)。
- [x] 小文件已合并,测试文件数下降(-9)。
- [x] 护栏 keep-list 仍全绿且仍存在(main 307 / critic 21 / robustness 54 /
      kernel 5 / framework 4)。
- [x] 剩余测试全绿(io/robustness/critic/framework 全量重跑通过)。
- [x] 测试总 LOC 下降(-238),每个提交净行数为负。
- [x] 每个 §1 对外契约仍有 ≥1 护栏测试覆盖(未改动 keep-list)。

未合并(已评估,刻意保留):同一特性但 stub/helper 体不同会冲突的
(`framework_pr_discover_directed`/`_retry` 的 sync/async `_call_discover`)、
测不同子模块的(critic `web_tools_*` 各自 `_cfg`)、一文件一信号规则的
(`signals_*`)——强合并是改名搬家而非降复杂度,违背主旨。
