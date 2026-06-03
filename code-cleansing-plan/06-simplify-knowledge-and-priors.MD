# Step 06: Simplify Knowledge And Priors

## 本计划的核心原则

这个代码清理是为了大幅度降低这个代码库的工程逻辑复杂度, 降低维护难度。清理只需要保证端到端功能一样, 不需要保证单元性的功能一致。当前代码库已经非常庞大, 所以不要吝啬 cut off 代码。

具体原则:

- 只保留当前端到端流程真实使用的知识系统。
- 旧 KB 协议不应继续污染主流程。
- Research priors 只能是 advisory, 不应变成第二套 gate。
- 知识写入失败不应影响优化主流程。

## 目标

当前系统里有多层知识相关概念:

- RecipeKB / recipe snapshot。
- warm start / warm replay。
- KnowledgePlane。
- PR feed。
- research scout hints。
- competitor target。
- old Cortex v1 graph KB。
- optimization journal。

这些概念有些是当前 live runtime 的核心辅助, 有些是历史迁移残留, 有些是只读 advisory。清理目标是让知识系统的职责清楚:

- 什么影响启动和 warm start。
- 什么影响 specialist prompt。
- 什么只进入 report。
- 什么已经退场。

## 知识系统职责划分

### Recipe snapshot

作用:

- 存储跨 session 的最佳配置、经验、pitfall、remaining gaps。
- 支撑 warm start。
- 支撑 CLOSE 后的经验沉淀。

原则:

- 作为当前 KB 主路径保留。
- shape 应尽量稳定、短小。
- 写入应 best-effort, 不阻断主流程。
- 不应和 old graph KB 混在一个抽象里。

### Warm start / warm replay

作用:

- 用历史 recipe 指导当前 run。
- 在满足条件时验证历史 best config。

原则:

- 保留端到端价值。
- 控制自动 replay 的复杂度。
- 不让 warm replay 变成绕过 baseline/explore 的隐藏路径。
- 失败或低 confidence 时应优雅降级。

### KnowledgePlane

作用:

- 为 Coordinator 和 specialist prompt 提供 PR feed / KB context。
- 作为 read facade, 不直接改变优化结果。

原则:

- 保留当前实际使用的 read path。
- 删除或隔离 old Cortex v1 graph methods。
- 不把未来工具或未实现 MCP surface 写进主 prompt。
- 不在 KnowledgePlane 里做 heavy runtime decision。

### Research scout hints

作用:

- 收集外部已验证先验。
- 给 Orchestration 和 specialists 提供 advisory context。

原则:

- read-only。
- 不 benchmark。
- 不 KEEP/REVERT。
- 不直接改 optimization stack。
- 必须有 source。

### Optimization journal

作用:

- 保存本 session 的 KEEP/REVERT 事实。
- 作为 KB write 的安全来源。

原则:

- 应与 runtime gate 解耦。
- 不应成为第二套 SharedState。
- 不应阻断 task promotion。

## 清理方向

### 删除 old KB path

如果当前 CLI 和 Coordinator 已经使用 RecipeKB, old graph KB 相关 path 应进入删除候选。

处理原则:

- 删除 old client 常量。
- 删除 old hypothesize/verify protocol。
- 删除 prompt 中旧 KB lifecycle。
- 删除只服务 old KB 的测试。
- 如果仍需读旧数据, 放到独立 migration/importer, 不进入 runtime。

### 简化 KB 写入

KB 写入应只在几个明确时机发生:

- terminal task result。
- KEEP / REVERT。
- CLOSE summary。
- warm replay result。

不要在 proposal 阶段建立复杂 graph edge。Proposal 是猜测, 不应带来持久知识负担。

### 简化 prompt 中的知识块

Prompt 中只需要展示:

- 当前 warm start recipe 的高价值摘要。
- 当前 gap 相关的 KB hints。
- research hints 的短摘要。
- PR feed 的最相关条目。

不要展示:

- KB 协议细节。
- old graph edge。
- 长 JSON。
- 过多历史 session 叙事。

### 收敛 specialist knowledge context

Specialist 的知识上下文应由一个清晰输入 bundle 生成:

- gap。
- domain tags。
- relevant recipe facts。
- relevant PRs。
- source roots。
- research hints。

不要让 specialist runner、prompt builder、KnowledgePlane 分别决定一部分知识结构。

## 测试调整

保留:

- warm start 加载和降级。
- warm replay 成功/失败的端到端结果。
- KB 写入 best-effort。
- research hints 需要 source。
- PR feed disabled 时 prompt 不暴露不可用工具。

删除或合并:

- old Cortex graph edge 测试。
- old hypothesize/verify 测试。
- 对 KB 内部 JSON 细节的过度断言。
- 对 prompt 中长知识块文案的测试。

## 风险

主要风险:

- 删除 old KB 后 warm start 缺数据。
- KnowledgePlane 简化后 specialist prompt 信息不足。
- research hints 被误当成 hard gate。
- KB 写入失败被错误升级为 runtime failure。

控制方式:

- 明确 KB read/write 调用点。
- 保持 no-KB degraded path。
- 保持 research hints advisory-only。
- 保留端到端 warm start smoke。

## 输出物

本阶段应产出:

- 当前知识系统职责图。
- old KB path 删除清单。
- RecipeKB canonical read/write 流程。
- research scout advisory 规则。
- specialist knowledge input bundle 设计。

## 完成标准

完成后应达到:

- 主流程只依赖当前 RecipeKB 和必要 PR feed。
- old KB 协议不再出现在 runtime prompt 和主代码路径。
- Research scout 仍提供先验, 但不会成为隐藏决策系统。
- KB 相关代码可以被新维护者独立理解。
