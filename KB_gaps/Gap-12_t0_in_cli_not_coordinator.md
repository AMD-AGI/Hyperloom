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

### 实际落地 (混合 A + B-lite, 2026-05-20)

最终采用 **A + B-lite 混合方案**, 既对齐文档又给 SDK 一条 fallback,
但不做 cli 的大规模重构:

1. **共享 helper**: 把 cli 内 ~150 行 T0 ritual 抽到
   `inference_optimizer/orchestrator/cortex_t0.py::run_t0_anchor`.
   两条入口走同一个函数, 仅 `fail_fast` / `on_status` 不同.
2. **cli 仍是 canonical 入口**: `cli._bootstrap_cortex_kb` 改为薄
   wrapper, 调用 `run_t0_anchor(fail_fast=True, on_status=print)`,
   失败 `sys.exit(2)` 行为不变, 操作员看到的 boot banner 不变.
3. **Coordinator defensive fallback**: 新增
   `Coordinator._ensure_cortex_t0_anchored()`, 在 `__init__` 末尾
   (在 `_ensure_phase_initialised` 之后) 调用. 仅当
   `cortex_kb.enabled` 且 `cortex_session_id == ""` 时调
   `run_t0_anchor(fail_fast=False)`, 失败 warning 退化.
4. **文档对齐**: KB_design §3.2 §5.1 加 "T0 位置约定" 表,
   明确 cli / Coordinator 双入口的失败语义差异.

## 6. 验收口径

- [x] (选项 A) KB_design §3.2 §5.1 文档明确 T0 在 cli boot 内
      ("T0 位置约定" 表)
- [x] (选项 B-lite) Coordinator 单元测试 (无 cli) 启动时 warm_start_recipe
      非空 (`tests/test_v08_cortex_t0_anchor.py`)
- [x] cli 的 fail-fast (`sys.exit(2)` on Cortex 故障) 行为不变, 操作员
      boot banner 不变
- [x] `--no-cortex` 在两条入口都正确 no-op

## 7. 风险 / 回退

- 共享 helper 重构是 *additive*: cli 行为字节级不变.
- Coordinator fallback 在 cli 路径上 no-op (sid 已存在), 仅服务 SDK.
- 回退: 把 `_ensure_cortex_t0_anchored` 从 `__init__` 里删掉; cli +
  helper 仍工作.

## 8. 关联 gap

- 与 Gap-02 (KnowledgePlane bootstrap 位置) 类似, 后续若做 SDK 友好的
  全 KnowledgePlane fallback 可参考 Gap-12 同款 pattern.
