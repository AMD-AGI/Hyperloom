# KB_design — Hyperloom v0.8 详细设计索引

> 目标版本: `inference_optimizer` v0.8
> 父文档: `../KB_design.MD` 第 3 章 (概念设计)
> 本目录: 把第 3 章每节展开为可实施级别的逻辑设计 (无代码), 每节一个子文件夹.

## 阅读次序

```
基础理念    ┌─ 3.1_design_philosophy/         ── 不变量 + 决策原则
            │
管线骨架    ├─ 3.2_pipeline_phases/           ── 五段固化管线
            ├─ 3.3_role_realignment/          ── 4 角色重边界
            │
特性区      ├─ 3.4_explore_consolidation/     ── EXPLORE 合并
            ├─ 3.5_specialist_framework/      ── LLM specialist 框架
            ├─ 3.6_knowledge_plane/           ── Cortex KB + PR + 本地源码
            ├─ 3.7_resource_lanes/            ── research_lane + 并发
            │
横切层      ├─ 3.8_phase_state_machine/       ── phase 转移与 plateau
            ├─ 3.9_drop_scoreboard/           ── 砍 scoreboard
            ├─ 3.10_shared_state_evolution/   ── SharedState 加减
            ├─ 3.11_policy_gate_evolution/    ── PolicyGate 5 条新规则
            ├─ 3.12_observability/            ── breakdown v2 + 新段
            │
路线图      ├─ 3.13_milestones/               ── M1–M7 里程碑详细设计
            │
辅助参考    ├─ 3.14_risks/                    ── 风险矩阵
            └─ 3.15_v06_v08_cheatsheet/       ── 概念跃迁速查
```

建议第一次通读按上述纵向顺序; 实施时按 §3.13 的 M1→M7 里程碑横向推进,
每个里程碑回引相关章节。

## 文件命名约定

- 每个子文件夹一个 `README.md` 作为该节总览 + 自包含设计稿。
- 复杂节 (如 §3.13) 在 README 之外按"步骤"或"里程碑"再拆 MD,
  文件命名: `<step|milestone>_<slug>.md`。
- 所有文件**不含代码**, 只描述 *what / why / how (logical) / acceptance*。

## 设计文档统一模板

每份设计 MD 至少包含以下骨架 (允许根据节内容裁剪/扩展):

1. **设计目标** — 这节解决什么问题, 不解决什么问题。
2. **现状回顾** — v0.6 / v0.7 中的相关现状, 关键文件 / 概念引用。
3. **不变量** — 本节设计 *绝不能破坏* 的全局约束 (跨章节)。
4. **核心机制** — 概念层面的机制描述 (数据流, 控制流, 状态转移)。
5. **接口/契约** — 跨章节调用面 (输入 / 输出 / 失败模式), 不写函数签名。
6. **实施步骤** — 落地顺序, 每步独立可验证。
7. **边界条件 / 失败模式** — 各种 corner case 怎么处理。
8. **验收标准** — 客观可测的判定 (功能正确 + 可观测 + 可回退)。
9. **依赖与影响面** — 上游需要 / 下游被影响的章节列表。

## 与父文档的关系

本目录是**展开**, 不是**重述**。父文档 `KB_design.MD` 第 3 章给出 *概念
轮廓*, 本目录给出 *可拿去拆 PR 的逻辑设计*。如果两侧出现不一致, **以
本目录为准** (父文档应在更新 PR 中同步)。

## 不写代码的原则

> "代码设计" 留到每个里程碑 PR 的实施稿 (typically `docs/v0.8/M<N>/impl.md`)
> 中给出。本目录的所有 MD 只回答 *如果我是另一个工程师, 看完这份文档
> 能不能独立写出代码*; 如果还需要看到字段表 / 函数签名 / 测试用例
> 才能动手, 那是实施稿的事, 不是这里的事。
