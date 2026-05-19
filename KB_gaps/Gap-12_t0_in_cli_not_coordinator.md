# Gap-12 — T0 落在 CLI 不在 Coordinator

> 严重度: **P2 次要** (CLI 是唯一生产入口, 实际影响小)
> 主轴影响: **主轴 B (知识外接)**
> 体检报告: `../KB_design_gaps.MD` §6 Gap-12

## 1. 问题描述

KB_design §3.2 §5.1 暗示 T0 (Cortex `session begin` + find-recipe +
warm_start_recipe 写入) 在 PRELUDE phase 内由 Coordinator 跑.

实际: `cli.py::_bootstrap_cortex_kb` (~1748-1912) 在 Coordinator 构造
前跑. SDK / 测试 / 非 CLI 入口启动时, T0 不会跑 → warm_start 字段空.

CLI 是唯一生产入口, 实际影响小. 主要风险:
- SDK 直接 `Coordinator(...)` 启动 → 没 warm_start
- 集成测如果不走 cli 路径也没 warm_start
- 设计文档与实现位置不符, 阅读时困惑

## 2. 现状代码 trace

`cli.py:1748-1912` 实现 T0; `_run_optimize` 在 Coordinator 构造前
(~2197-2204) 调用. Coordinator 自身**无 T0 hook**.

## 3. 设计意图

§3.2 §5.1 把 T0 作为 PRELUDE phase 内动作之一. Coordinator 应当 own
phase 全部副作用 (主轴 A "phase 由 Coordinator 强制").

## 4. 根本原因

M1 PR 把 T0 放在 cli 内, 理由: T0 需要 cortex_kb_client 实例化, 而
client 又需要 args (URL / token / mode), 把这一切都放 Coordinator 构
造前更直观. 设计文档没明确"T0 写在哪里", 留了暗示.

## 5. 修复路径

### 选项 A — 接受现状, 文档对齐

最小成本: KB_design §3.2 §5.1 改为 "T0 在 cli boot 内, Coordinator
启动时已读到 warm_start_recipe". 不动代码.

### 选项 B — 把 T0 hook 移到 Coordinator

需要:
- Coordinator 构造时接收 cortex_client (已经在做)
- 加 `_cortex_t0_hook()` 方法, 在 `Coordinator.run()` 第一个 tick 跑
- cli 不再调 `_bootstrap_cortex_kb`, 只构造 client 传给 Coordinator
- SDK 启动也能享受 T0

工作量较大 (~100 行). 收益: SDK 启动正确性 + 设计一致.

### 推荐

选项 A (文档对齐). v0.8 GA 后真有 SDK 场景再做选项 B.

## 6. 验收口径

- [ ] (选项 A) KB_design §3.2 §5.1 文档明确 T0 在 cli boot 内
- [ ] (选项 B) Coordinator 单元测试 (无 cli) 启动时 warm_start_recipe
      非空

## 7. 风险 / 回退

- 选项 A 无风险, 仅文档改动
- 选项 B 涉及 Coordinator 构造时序改动, 可能影响 resume 路径; 回退 =
  把 _bootstrap_cortex_kb 放回 cli

## 8. 关联 gap

- 与 Gap-02 (KnowledgePlane bootstrap 位置) 类似, 选项 B 时一起重构
