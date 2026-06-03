# Step 08: Clean Breakdown And Reporting Contract

## 本计划的核心原则

这个代码清理是为了大幅度降低这个代码库的工程逻辑复杂度, 降低维护难度。清理只需要保证端到端功能一样, 不需要保证单元性的功能一致。当前代码库已经非常庞大, 所以不要吝啬 cut off 代码。

具体原则:

- `session_breakdown.json` 是外部 contract, 不能像内部 helper 一样随意删除。
- 但外部 contract 不等于所有历史 collector 都要永久保留。
- 新 session 的 reporting path 应简单、稳定、可解释。
- archived session reader 应单独决策, 不应拖累 live producer。

## 目标

Breakdown 和 report 是 Hyperloom 对外展示优化结果的核心产物。它们被 dashboard、CI、离线分析、用户报告消费。当前 reporting 系统的复杂度主要来自:

- 新旧 schema 并存。
- 旧 action 名称兼容。
- 旧 kernel-agent artifact layout。
- 多个 collector 读同一类信息。
- 文档和代码版本漂移。
- report narrative 与 session_breakdown JSON 职责重叠。

本阶段目标是让 reporting 分为两层:

- live session producer: 只为当前 runtime 输出清晰稳定的 contract。
- archived session compatibility: 如果保留, 独立、可删除、少污染主路径。

## Contract 分层

### Core output contract

必须保留:

- session metadata。
- workload。
- baseline。
- final result。
- phase/action timeline。
- optimization stack。
- capability summary。
- kernel lifecycle。
- sweep / concurrency result。
- critic / robustness summary。
- warnings。
- source artifact references。

这些字段服务 dashboard 和用户判断优化是否成功。

### Optional diagnostic sections

可以保留, 但不应影响主流程:

- full specialist transcript。
- verbose kernel attempt details。
- raw telemetry path。
- PR feed details。
- internal policy denial history。

这些字段应该是 additive, 缺失时 downstream 能正常处理。

### Archived compatibility sections

只服务旧 session:

- legacy `backends` / `params` / `validate_stack` alias。
- legacy kernel-agent nested layout。
- old T2/T3 KB fields。
- old provenance strings。

这类字段要单独标记是否保留。不要让 live producer 为了旧字段继续写复杂 alias。

## Live Producer 清理

### 新 session 输出 canonical shape

新 session 不应继续写 retired action 的 primary fields。对于已经合并的功能:

- live primary section 应使用新名称。
- legacy alias 如果需要, 只作为 downstream transitional alias。
- alias 应有删除时间或删除条件。

### Collector 只读必要来源

Collector 应优先从稳定来源读取:

- `state.json`。
- `manifest.json`。
- task result。
- reports artifact。
- kernel-agent artifact。

不要从多处猜测同一个值。如果必须 fallback, fallback 应短且集中。

### Report 和 breakdown 分工

`session_breakdown.json`:

- 机器可读。
- 稳定字段。
- 少 narrative。
- 可用于 dashboard。

Markdown report:

- 人可读。
- 总结主要优化结果。
- 可以引用 breakdown 中的数据。
- 不应成为另一个事实来源。

## Archived Reader 策略

需要决定:

- 是否还要支持旧 session dashboard 展示?
- 是否还要支持旧 session offline analysis?
- 支持多老的版本?

如果不支持:

- 删除 archived reader。
- 删除 old alias。
- 删除 old fixture tests。
- docs 明确说明不支持旧 session。

如果支持:

- 把 archived reader 与 live producer 分离。
- 不允许 archived reader 的逻辑进入新写路径。
- 保留少量 fixture-based tests。

## Schema 文档清理

文档必须和代码一致:

- schema version。
- additive field policy。
- required vs optional field。
- missing data 表达方式。
- downstream consumer best practice。

删除:

- 已退场字段的详细介绍。
- 与当前 producer 不一致的版本示例。
- 旧流程 narrative。

保留:

- 真实 contract。
- 消费者如何容错。
- 如何定位 artifact。
- 哪些字段是 headline 指标。

## 测试调整

保留:

- live session breakdown smoke。
- schema version 和 headline field。
- missing data 不伪造。
- final throughput / gain / action path。
- kernel lifecycle summary。
- specialist/dynamic/action attribution, 如果对 dashboard 有价值。

删除或合并:

- 对旧 alias 的大量细节测试, 如果决定不支持。
- 对 collector fallback 顺序的过细测试。
- 对 markdown 具体句子的大量断言。
- 对 retired action section 的正向测试。

## 风险

主要风险:

- 删除字段导致 dashboard 失败。
- 文档和 schema 不一致。
- collector fallback 删除后旧 session 分析失败。
- report 和 breakdown 数据来源分裂。

控制方式:

- 与 downstream consumer 对齐必需字段。
- 建立 live schema smoke。
- archived reader 单独决策。
- report 从 breakdown 或同一事实源生成。

## 输出物

本阶段应产出:

- live breakdown schema 清单。
- archived compatibility 保留/删除清单。
- schema 文档更新。
- collector 拆分或简化方案。
- downstream contract test 清单。

## 完成标准

完成后应达到:

- 新 session 的 breakdown 简洁、稳定。
- 文档和 schema 一致。
- archived reader 不再污染 live producer。
- dashboard contract 清晰。
- reporting 代码不再承担所有历史版本复杂度。
