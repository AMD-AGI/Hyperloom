# Phase A — 死代码 / 退役代码清除

## 目的

把主计划 §8 退役登记表逐项删干净,同时删除"合并途中产生的废弃物"。
这是**纯减法**相位:几乎每个 commit 都应是大幅净删除。

> 顺序在逻辑合并(Phase B)之前:**先扔垃圾,再搬家**,避免把死代码一起合并进新结构。

## 步骤文件

- [`01_retired_actions.md`](01_retired_actions.md) — 退役 action 名(setup/classify/backends/params/validate_stack/select_kernels)。
- [`02_cortex_kb_remnants.md`](02_cortex_kb_remnants.md) — Cortex/T2-T3 KB / NDJSON flusher 残留。
- [`03_resume_migration_removal.md`](03_resume_migration_removal.md) — 跨版本 resume 迁移读取器(breakdown 兼容除外)。
- [`04_stubs_noops_shims.md`](04_stubs_noops_shims.md) — stub/no-op 执行器、proxy、payload 别名 shim。
- [`05_env_install_cleanup.md`](05_env_install_cleanup.md) — 退役 env / install flag / auth proxy。
- [`conduct.md`](conduct.md) — 本相位行为准则。

## 入口标准

- Phase 0 出口标准全满足(护栏全绿 + 基线已记录)。

## 出口标准

- [ ] §8 登记表每项:已删 或 已标注"保留+原因"。
- [ ] 退役 action 引用计数(Phase 0 第 5 项)归零(仅迁移/拒绝测试保留)。
- [ ] 护栏全绿;CLI flag 金标准 diff 仅显示"已登记的退役 flag"消失。
- [ ] 每个删除项一个独立 commit。

## 通用删除流程(每一项都走一遍)

1. **定位**:`rg` 全仓搜符号/字符串,列出所有 call-site。
2. **判活死**:确认无活引用(测试桩、迁移读取器、拒绝测试不算活引用)。
3. **删除**:删定义 + 所有死引用 + 相关注释 + 相关配置/yaml。
4. **跑护栏**:`pytest <keep-list>`;若有测试专测该退役项,一并删(记在 Phase E 也可,但既然现在删了功能,这里直接删更干净)。
5. **提交**:`Remove retired <X>`,确认 `git diff --stat` 净减。

## 进度记录

(完成后填:每项 commit hash + 净删行数)
