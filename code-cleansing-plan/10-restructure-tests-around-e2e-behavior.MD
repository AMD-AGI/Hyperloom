# Step 10: Restructure Tests Around E2E Behavior

## 本计划的核心原则

这个代码清理是为了大幅度降低这个代码库的工程逻辑复杂度, 降低维护难度。清理只需要保证端到端功能一样, 不需要保证单元性的功能一致。当前代码库已经非常庞大, 所以不要吝啬 cut off 代码。

具体原则:

- 测试应保护端到端行为和外部协议, 不应锁死内部 helper。
- 删除代码时同步删除只服务旧实现的测试。
- 保留少量高价值 contract tests。
- 测试数量下降是目标之一, 不是副作用。

## 目标

当前测试体系覆盖很多细碎 helper、历史 bug、迁移过程和中间实现细节。它在功能高速增加时有价值, 但在清理阶段会阻止内部结构简化。

本阶段目标:

- 把测试从“证明每个小 helper 没变”转为“证明端到端行为没变”。
- 删除 retired feature 的正向测试。
- 合并重复测试。
- 保留关键协议和安全 gate。
- 让测试失败能指向真实用户影响。

## 测试保留标准

### 必须保留

以下测试有长期价值:

- CLI smoke。
- phase transition。
- PolicyGate / intent envelope。
- action catalogue drift guard。
- executor wiring。
- session path / artifact layout。
- `state.json` core field persistence。
- baseline / explore / integrate / sweep / report flow。
- kernel request shape/path/safety gate。
- framework PR apply/revert。
- robustness/critic subprocess envelope contract。
- `session_breakdown.json` contract。

### 可以合并

以下测试可合并成更少文件:

- 同一个 executor 的多个小 helper 测试。
- 同一个 payload shape 的重复参数化测试。
- prompt 中多个相近 section 的文案测试。
- legacy migration 的相似 fixture。
- action metadata 的字段默认值测试。

### 可以删除

以下测试应删除:

- retired action 的正向执行测试。
- 只验证私有 helper 名称和内部 dict 的测试。
- 与更高层行为测试重复的低层测试。
- 只为某个 PR 临时 bug 编号存在的测试。
- 对注释或长 prompt 文案逐字断言的测试。
- 对旧兼容路径的测试, 如果产品决定不保留旧兼容。

## 建议测试结构

按行为域重组, 而不是按历史 PR 或 helper 文件重组。

建议测试组:

- `test_runtime_contracts`: intent, policy, action catalogue, phase allowlist。
- `test_cli_and_session`: CLI args, session layout, manifest/state basics。
- `test_prelude_flow`: target analysis, baseline, profile/roofline gate。
- `test_explore_flow`: explore grid, dedup, KEEP/REVERT, stack rebench。
- `test_framework_pr_flow`: discover, critic gate, apply, bench, revert。
- `test_specialist_dynamic_flow`: specialist output, dynamic action, integrate_patch handoff。
- `test_kernel_flow`: trace analyze, shape gate, run optimization, integrate。
- `test_sweep_and_close`: sweep, conc sweep, report, close sequencer。
- `test_breakdown_contract`: session_breakdown schema and headline fields。
- `test_compatibility`: only retained compatibility scenarios。

这样新维护者可以从测试文件名理解系统行为。

## 测试分层

### Tier 0: Contract tests

每次小改都应跑。

覆盖:

- action catalogue。
- phase allowlist。
- PolicyGate。
- intent parser。
- session path。

特点:

- 快。
- 少。
- 稳定。
- 不依赖 heavy subprocess。

### Tier 1: Runtime unit-flow tests

每个阶段完成后跑。

覆盖:

- Coordinator tick。
- executor result promotion。
- explore flow。
- kernel handler。
- framework PR executor。
- breakdown build。

特点:

- 用 fake executor / fake backend。
- 验证行为流, 不验证内部 helper。

### Tier 2: Package integration tests

大阶段完成后跑。

覆盖:

- inference_optimizer package。
- robustness-agent contract。
- critic-agent bridge, 如果环境允许。
- kernel-agent handler tests。

### Tier 3: Real smoke

最终验收时跑。

覆盖:

- no-kernel optimize。
- kernel-enabled smoke。
- resume。
- breakdown dump。

## 测试清理方法

### 先标记, 再删除

对测试文件先分类:

- contract。
- behavior。
- compatibility。
- implementation detail。
- retired feature。

然后删除 retired 和 implementation detail。

### 合并重复 fixture

很多测试的成本来自重复构造 session、state、task、variant。应统一高层 fixture, 但不要为了 fixture 复用引入复杂工厂。

原则:

- fixture 表达业务场景。
- 不暴露太多内部字段。
- 不为旧 helper shape 服务。

### 删除过细断言

测试应断言:

- 状态是否推进。
- action 是否被允许或拒绝。
- task 是否被创建。
- result 是否 promotion。
- artifact 是否可消费。

不应断言:

- 私有函数被调用几次。
- 内部 list 排序的非契约细节。
- prompt 完整文本。
- debug field。

## 风险

主要风险:

- 测试删太多导致协议漂移。
- 只保留大测试导致反馈太慢。
- 合并测试后定位困难。
- 删除 legacy test 后误删仍需保留的兼容。

控制方式:

- 保留 Tier 0 小型 contract tests。
- 每个外部协议至少一个 contract test。
- 兼容策略先决策, 再删测试。
- 测试命名按行为域, 提高定位能力。

## 输出物

本阶段应产出:

- 测试分类清单。
- 删除测试清单。
- 合并测试计划。
- 新测试分层说明。
- 最小必跑测试集。

## 完成标准

完成后应达到:

- 测试数量下降。
- 测试更关注端到端行为。
- 清理内部实现时不再被大量私有 helper 测试阻挡。
- 关键外部协议仍有明确护栏。
- 测试失败能更直接指向真实 runtime 风险。
