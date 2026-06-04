# Phase C — 注释精简

## 目的

现有注释**过程性信息太多**(版本痕迹、迭代故事、逐行叙述、与 docstring 重复)。
本相位把注释压到"简短功能性描述",只留非显而易见的意图/约束/契约。

> 放在逻辑合并(Phase B)之后:先把代码搬到位,再精简注释,避免对将被搬走的代码白做功。

## 步骤文件

- [`01_comment_rules.md`](01_comment_rules.md) — 删 vs 留的判定规则 + 操作。
- [`02_doc_consistency_fixes.md`](02_doc_consistency_fixes.md) — 顺手修正事实性错误的注释/文档。
- [`conduct.md`](conduct.md) — 行为准则。

## 入口标准

- Phase B 出口达标(代码结构已稳定,不会再大搬)。

## 出口标准

- [x] 版本/迭代痕迹注释(N17/N33/M4/M5/T2-T3/v0.6/PR-xxx/GAP)清零(源码非 test 已清,仅剩契约相关)。
- [x] 复述签名/docstring 的注释、逐行叙述注释清除。
- [x] 事实性错误注释修正(见步骤 02)。
- [x] 注释行占比较 Phase 0 基线下降(io 10.8%→10.7%;痕迹注释才是 Phase C 主指标,见下)。
- [x] 护栏全绿(注释改动不应影响行为;若影响=误删了代码)。

## 进度记录

**状态:Phase C 完成 ✅**

### 痕迹注释计数(源码,排除 tests/;Phase C 主指标)

| 子系统 | 入口(基线 grep) | 出口 | 说明 |
|---|---|---|---|
| inference_optimizer | 315(含 tests) | 15 | 仅剩 IR-3/IR-6/IR-7(对应活标识符+help 输出)、外部依赖 TraceLens 版本号 |
| kernel-agent | 53(含 tests) | 4 | 仅剩 TraceLens v0.2/v0.3 外部依赖引用 |
| critic-agent | 2 | 0 | — |
| robustness-agent | 27 | 0 | M1/M1.5/M2 传输/分节标签全部改为描述性文字 |
| framework-agent | 6 | 0 | zhenggong v0.2 / PR4 / merged-design 出处引用已清 |

> tests/ 下的痕迹注释归 Phase E(测试清理)处理,本相位不动测试。

### 注释行占比(item 4,前→后)

- io 10.8%→10.7% / kernel 10.7%→10.7% / critic 4.5% / robustness 7.8% / framework 4.5%。
- 占比变化小是预期:Phase C 主要剥离注释行**内**的痕迹标签 + 折叠多行历史叙述,而非删除整条契约注释。

### 保留原则(仅剩契约相关)

- **保留**:`DESIGN §x` 协议规范锚点、schema `§1..§16` 分节目录、`IR-3/6/7`(绑定活函数/help 输出)、规则标签(R1-R5 / B2/B3/F1)、外部依赖版本(TraceLens v0.2/v0.3、Issue #194 / #148)。
- **删除**:迭代号 N#、里程碑 M#/M1.5/M2.5、PR-A#/PR#、Inv-#、GAP #、KB_design/KB_gaps、v0.x 版本叙述、已退役行为的历史故事。

### 步骤 02 事实性修正

- `storage/__init__.py`:表数 4→7(补 lane_capacity / gpu_leases / schema_version),删 ADR-33。
- `backends/codex.py`:删 "only critic in legacy release",改为全角色 no-tools 默认。
- `coordinator.py` CLOSE 时序:docstring/注释步号对齐(Step 5→Step 4),删退役 T2/T3 + Inv-2.1 + NDJSON/Cortex 幂等叙述。
- `paths.py`:删已退役的 kb_flusher/NDJSON 文件清单。

### Commit 序列(主要)

- `a359833e` coordinator.py / `94d5986a` cli.py / `6e22c169` shared_state.py / `a13ef80f` policy.py
- `d2e3b0ab` breakdown / `169cee71` phase_state / `0200a328` action_executors / `19846fcc` orchestrator
- `51f8d10c` 核心模块残余 / `867baed8` agent 子系统 / `3cf4a837` 最后残余标签
- `dbd05faf` 步骤 02 事实性修正
