# 代码清理详细计划(分相位)

本文件夹是 [`../code_cleansing.MD`](../code_cleansing.MD)(主计划)的**逐相位展开**。
主计划定义"为什么/边界/验收";本文件夹定义"每一步怎么做/做与不做"。

> **中心主旨(始终遵守)**:大幅降低工程逻辑复杂度、降低维护难度。允许高风险清理与合并,
> 只需保证**代码修改后对外功能一样**(非单元级一致)。不要吝啬 cut off,
> **严禁清理后写得反而更多**。

## 相位顺序(有依赖,必须串行)

```
Phase 0(护栏+基线) → A(死代码) → B(逻辑合并) → C(注释) → D(文件/引用结构) → E(测试)
```

- **不可跳过 Phase 0**:它是后续所有"删 + 改"的安全网与度量基线。
- 每相位**出口标准**满足后才进下一相位(见各 `README.md`)。

| 相位 | 子文件夹 | 目标 | 风险 |
|---|---|---|---|
| 0 | [`phase_0_guardrails/`](phase_0_guardrails/) | 固化功能等价护栏 + 量化基线 | 低 |
| A | [`phase_a_dead_code/`](phase_a_dead_code/) | 删退役/死代码/迁移读取器 | 中 |
| B | [`phase_b_logic_merge/`](phase_b_logic_merge/) | 合并相似逻辑 + 拆瘦 god-module | 高 |
| C | [`phase_c_comments/`](phase_c_comments/) | 精简注释,只留功能性描述 | 低 |
| D | [`phase_d_structure/`](phase_d_structure/) | 降低 import / 文件结构复杂度 | 中 |
| E | [`phase_e_tests/`](phase_e_tests/) | 删细粒度单测、合并碎测试 | 中 |

## 已锁定的范围与边界(来自主计划 §10)

1. **范围** = `inference_optimizer/` + `kernel-agent/` + `critic-agent/` + `robustness-agent/` + `framework-agent/`。**`ci/` 不在本次范围**。
2. **外部契约**:对外字段/形状保持(`session_breakdown.json` 键、CLI flags、子进程 JSON 形状),**内部实现自由重写**。
3. **resume**:删所有**跨版本**迁移读取器与 sentinel;**唯一例外**=`breakdown/` 对旧 session 的兼容必须保留;同版本 `--resume`(崩溃恢复)作为核心功能保留。
4. **拆分**:接受拆 god-module,**单文件复杂度优先于文件数量**,但受"净行数不增 + 不引入循环引用"约束。

## 全局行为准则(每相位 `conduct.md` 在此基础上细化)

- **删除优先于重写**;改动若让净行数增加 → 放弃该项。
- **一类清理 = 一个 commit**,message 形如 `Remove retired <X>` / `Merge <A> into <B>` / `Compress <module> comments`。
- 每个 commit 前:跑护栏(Phase 0 选定集)+ `git diff --stat` 确认净减。
- 大块删除与逻辑合并**分开提交**(便于二分回退)。
- **不引入新抽象/新基类/新间接层**,除非它能净删除更多代码。
- 触碰 §1 契约边界时:对外字段/形状不变,改前在该相位记录里登记影响面。

## 进度记录约定

每相位完成后,在该相位 `README.md` 末尾记录:起止 commit、§6 度量前后快照、护栏结果。
