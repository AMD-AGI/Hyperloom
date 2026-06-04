# Phase 0 — 功能等价护栏 + 量化基线

## 目的

在任何"删代码 / 改代码"之前,建立两样东西:
1. **功能等价护栏**:一组粗粒度特征/契约测试 + CLI 冒烟,作为后续判断"功能是否一样"的**唯一客观判据**。
2. **量化基线**:记录当前 LOC / 文件数 / 最大文件 / 注释占比 / 退役引用计数,作为"严禁越清越多"的对照。

> 这是整套计划的安全网。**不做 Phase 0 就动后续相位 = 无安全网删代码**,禁止。

## 步骤文件

- [`01_baseline_metrics.md`](01_baseline_metrics.md) — 量化基线:测什么、怎么测、记到哪。
- [`02_guardrail_tests.md`](02_guardrail_tests.md) — 选定并固化护栏测试集(keep-list)。
- [`03_cli_e2e_smoke.md`](03_cli_e2e_smoke.md) — mock 全流程冒烟 + 采集"对外可观测金标准"。
- [`conduct.md`](conduct.md) — 本相位行为准则。

## 入口标准

- 在 `cleanup/zhenggong/safe-cleanup` 分支,工作区干净(`git status` 无未提交)。
- 已读 `../../code_cleansing.MD` §1(契约边界)、§2(护栏)、§6(度量)。

## 出口标准(全部满足才进 Phase A)

- [ ] §6 基线表已填实际数字(LOC、文件数、最大文件、注释占比、退役 action 引用计数)。
- [ ] 护栏测试集已选定并在文档中列明(keep-list),且**当前全绿**。
- [ ] mock 全流程冒烟跑通,金标准 stdout/产物形状已采集存档。
- [ ] 基线数字与护栏清单已提交(commit:`Add cleansing guardrails and baseline metrics`)。

## 进度记录

(完成后填写:起止 commit / 基线数字 / 护栏结果)
