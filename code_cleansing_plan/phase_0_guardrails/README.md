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

- [x] §6 基线表已填实际数字(LOC、文件数、最大文件、注释占比、退役 action 引用计数)。
- [x] 护栏测试集已选定并在文档中列明(keep-list),且**当前全绿**。
- [x] mock 全流程冒烟跑通,金标准 stdout/产物形状已采集存档(GPU 限制如实记录,见下)。
- [x] 基线数字与护栏清单已提交。

## 进度记录

**状态:Phase 0 完成 ✅(可进 Phase A)**

起始 HEAD:`14cceab6`("doc clean up")

提交序列:
1. `45fb9cf4` Add Phase 0 baseline metrics and cleansing plan docs
2. `cec2fb0e` Establish Phase 0 guardrail keep-list
3. `b9fa4df5` Capture Phase 0 golden CLI help and artifact key shapes

**基线数字(详见 `baseline_snapshot.txt`)**:
- 总 LOC:`inference_optimizer` 165563 / 5 子系统合计 232764。
- Python 文件数:`inference_optimizer` 344 / 5 子系统合计 570。
- 最大单文件:`coordinator.py` 12664、`cli.py` 6512、`shared_state.py` 4730、`collectors.py` 4391。
- 注释占比:io 10.8% / kernel 10.7% / critic 4.5% / robustness 7.8% / framework 4.5%。
- 退役 action 引用(RAW grep,未按 call-site 过滤):select_kernels 13 / validate_stack 107 / setup 39 / classify 47 / 'backends' 94 / 'params' 223。

**护栏结果(keep-list 17 文件,全绿)**:
- 主包子集:249 passed。
- critic 21 / robustness 54 / kernel 5 / framework 8,全 passed。
- keep-list 与逐子系统运行命令见 `guardrail_keep_list.txt`(跨包须分子系统跑,conftest 同名冲突)。

**金标准**:`golden_cli_help.txt`(CLI flag 面)+ `golden_{breakdown,state,manifest}_keys.txt`(产物键形状,真值由 breakdown 冒烟 fixture 生成,无需 GPU)。

**环境限制(如实记录)**:CLI 无 mock kernel/baseline workload flag,无法在无 GPU 下跑完整 `optimize` 到 CLOSE;故产物键形状改由 `build()` 跑 `test_breakdown_smoke` fixture 真实采集,§1 契约测试作为主要功能等价护栏。详见 `golden_smoke_notes.md`。

**工具备注**:本环境 shell PATH 无 `rg`,基线计数用 `grep` 采集;`vulture` 未安装,死代码探测留待 Phase A 逐 call-site 人工确认。
