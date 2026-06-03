# Step 07: Clarify Integration Boundaries

## 本计划的核心原则

这个代码清理是为了大幅度降低这个代码库的工程逻辑复杂度, 降低维护难度。清理只需要保证端到端功能一样, 不需要保证单元性的功能一致。当前代码库已经非常庞大, 所以不要吝啬 cut off 代码。

具体原则:

- 外围系统边界要清楚, 主包不应复制外围系统内部逻辑。
- 保留实际 live bridge, 删除未实现或未接入的承诺。
- JSON/subprocess/artifact contract 比 Python import 更重要。
- 每个 integration 只保留一个清晰入口。

## 目标

`inference_optimizer` 不是单体系统, 它连接多个外围系统:

- kernel-agent。
- critic-agent。
- robustness-agent。
- framework-agent。
- Magpie。
- TraceLens。
- InferenceX。
- multi-node RayJob / SaFE。

当前复杂度的一部分来自边界不够清晰: 有些逻辑在主包和外围包各复制一份, 有些文档描述了未实现 future path, 有些旧 bridge 已经退场但仍有注释或测试残留。

本阶段目标是让每个 integration 有明确职责、入口和失败语义。

## Integration 分类

### Kernel integration

职责:

- TraceLens analysis。
- hot kernel candidate。
- GEAK/OOB/Cursor/Claude/Codex kernel optimization。
- kernel patch integrate/revert。

主包边界:

- 主包只负责发起 request、验证 shape/path、安全 gate、接收结果并更新 state。
- kernel-agent 工具负责具体优化。

保留:

- kernel request handler。
- kernel shape/path validation。
- apply/integrate safety gate。
- kernel attempt artifact reader。

删除或压缩:

- 未使用 request kind。
- 重复的 backend ordering 叙事。
- 与 kernel-agent 内部实现重复的逻辑。
- 旧 artifact layout, 如果不再支持历史 session。

### Critic integration

职责:

- review proposal。
- enforce approve standard。
- 输出 review verdict。

主包边界:

- 主包负责准备 proposal packet 和消费 verdict。
- critic-agent 负责 review bundle 和模型判断。

保留:

- prepare/commit JSON bridge。
- verdict envelope validation。
- missing context 降级。

删除或压缩:

- 未使用的 critic mode。
- old bare fallback, 如果产品不再支持。
- prompt 中过长的 approve 标准重复描述, 如果 critic-agent 已经拥有。

### Robustness integration

职责:

- health monitoring。
- RCA。
- recovery suggestion。
- scheduling police。

主包边界:

- 主包负责把 Coordinator prompt/state 发给 robustness runtime。
- robustness-agent 负责发出 valid intent。
- PolicyGate 负责最终拦截危险操作。

保留:

- intent envelope mirror。
- contract tests。
- robustness-only action boundary。
- multi-node mock downgrade 规则, 如果仍需要。

删除或压缩:

- legacy robustness mode, 如果不再支持。
- old provider 文档。
- 与 upstream PolicyGate 重复的大段解释。

### Framework integration

职责:

- upstream PR discovery。
- candidate diff apply。
- benchmark KEEP/REVERT。

主包边界:

- framework-agent 负责 discovery。
- 主包负责 Critic gate、apply、benchmark 和 rollback。

保留:

-实际 wired 的 discovery bridge。
- framework PR executor。
- repo map 的最小同步机制。

删除或修正:

- 未实现的 `fa agent` / 5th-role 描述。
- 未调用 wrapper。
- 旧 `fa explore` 与当前 FRAMEWORK_PR phase 的混淆说明。

### Benchmark integration

职责:

- Magpie benchmark。
- profile。
- accuracy gate。

主包边界:

- executors 负责 materialize config 和 parse result。
- Magpie 负责实际运行。

保留:

- benchmark config materialization。
- result parsing。
- timeout / kill / cleanup。

删除或压缩:

- 多处重复 benchmark 参数说明。
- old framework-specific args alias, 如果已经 canonical。
- placeholder bench registry, 如果未启用。

### Multi-node integration

职责:

- RayJob create/bootstrap。
- distributed serving。
- profile trace path。
- patch replay。

主包边界:

- CLI 可保留 multi-node provision。
- action executors 应只读取统一环境和 session artifact。

清理方向:

- 删除 backup 文件。
- 删除旧脚本路径。
- 压缩 SaFE/RayJob 过程性说明。
- 保留端到端 topology gate。

## Failure Semantics

每个 integration 都应明确失败语义:

- hard fail: 无法继续端到端优化。
- soft degrade: 功能缺失但主流程继续。
- needs manual review: 结果不确定, 需要人工判断。
- skip: 用户配置禁用或环境不支持。

不要让每个 integration 自己发明一套失败字段。主包应统一消费为 task result、policy denial、state warning 或 report warning。

## 测试调整

保留:

- subprocess JSON bridge contract。
- intent envelope validation。
- kernel handler path/shape safety。
- framework PR apply/revert behavior。
- robustness-only action guard。
- missing integration graceful degrade。

删除或合并:

- 外围工具内部实现测试, 如果不属于主包。
- 未实现 future integration 的测试。
- 对 subprocess temp file layout 的过细测试, 除非它是 contract。

## 风险

主要风险:

- 删除看似未引用的 subprocess bridge。
- 删除 artifact path 后 report/breakdown 失效。
- 外围包仍期待主包 mirror 某些常量。
- 文档保留未实现 feature, 误导 LLM 或维护者。

控制方式:

- 每个 integration 建一张 contract 卡片。
- 所有 bridge 都有最小 contract test。
- 删除 future path 前检查是否真的没有 runtime wiring。
- 主包只保留 integration 边界, 不复制内部实现。

## 输出物

本阶段应产出:

- integration boundary map。
- 每个 integration 的唯一入口说明。
- 删除未实现 bridge 的清单。
- 失败语义统一表。
- contract tests 精简方案。

本阶段的边界图见
`code-cleansing-plan/07-integrations/01-integration-boundary-map.MD`,
失败语义表见
`code-cleansing-plan/07-integrations/02-failure-semantics.MD`,
执行记录见
`code-cleansing-plan/07-integrations/03-execution-status.MD`。

## 完成标准

完成后应达到:

- `inference_optimizer` 与外围系统之间边界清楚。
- 未实现功能不再伪装成 live path。
- 外围系统内部复杂度不再泄漏到主包。
- 每个 integration 的失败行为可以被端到端流程理解。
