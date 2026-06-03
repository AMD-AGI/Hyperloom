# Step 11: Reorganize Files And Imports

## 本计划的核心原则

这个代码清理是为了大幅度降低这个代码库的工程逻辑复杂度, 降低维护难度。清理只需要保证端到端功能一样, 不需要保证单元性的功能一致。当前代码库已经非常庞大, 所以不要吝啬 cut off 代码。

具体原则:

- 文件结构整理应放在清理后期, 不应掩盖行为删除。
- 目录边界应表达责任边界。
- import 方向应单向, 避免 runtime 互相缠绕。
- 不为旧 import 路径长期保留复杂 shim。

## 目标

当 dead code、兼容路径、注释和测试都清理到一定程度后, 再整理文件结构。过早搬家会制造巨大 diff, 让真正的行为变化难以 review。后期搬家则可以把已经收敛好的职责放到更清楚的位置。

本阶段目标:

- 让目录结构反映系统职责。
- 降低跨模块 import 复杂度。
- 避免 executors 反向依赖 Coordinator。
- 避免 policy/phase 依赖 heavy runtime。
- 让外部 integrations 有明确边界。

## 目标结构原则

### Runtime core

放主循环和编排:

- Coordinator core。
- dispatcher。
- phase hooks。
- proposal flow。
- kernel flow。
- specialist flow。
- dynamic action flow。
- framework PR flow。
- fact write flow。

原则:

- runtime core 可以依赖 protocol, state, executors, integrations。
- runtime core 内部模块之间应按单向职责调用。
- 不要形成多个模块互相 import 的环。

### Protocol

放稳定协议:

- intent parser。
- policy。
- agent role。
- action metadata interface。

原则:

- 尽量纯逻辑。
- 不依赖 heavy executors。
- 不依赖 CLI。
- 可被外围 agent contract tests 引用。

### State

放状态模型:

- SharedState。
- state migration。
- state rendering。
- state schema constants。

原则:

- 写路径清晰。
- migration 与 live state 分离。
- prompt rendering 可单独测试。

### Executors

放确定性任务执行:

- baseline。
- explore。
- integrate patch。
- framework PR。
- roofline/profile。
- sweep/conc sweep。
- report/session breakdown。
- recover。

原则:

- executor 接收 task context。
- executor 不知道 Coordinator 内部。
- executor 只返回 result, 不直接改 core state。

### Integrations

放外部系统桥:

- kernel-agent。
- critic-agent。
- robustness-agent。
- framework-agent。
- Magpie helpers。
- InferenceX client。

原则:

- 每个 integration 只有一个主要入口。
- integration module 定义 request/response 边界。
- 不复制外围系统内部逻辑。

### Reporting

放 breakdown 和 report:

- schema。
- collectors。
- exporter。
- markdown reporters。

原则:

- live producer 和 archived reader 分离。
- reporting 不反向依赖 runtime internals。
- 尽量从 artifact 读事实。

## Import 方向

建议方向:

```text
cli
  -> runtime
      -> protocol
      -> state
      -> executors
      -> integrations
      -> reporting
```

禁止方向:

- executor import Coordinator。
- policy import executor。
- phase_state import Coordinator。
- state migration import CLI。
- integration import runtime private helper。
- tests 依赖过多 private module。

如果需要共享常量, 应放到 owner 模块或 protocol/state 层, 不应在两个 runtime 模块之间互相 import。

## 搬迁策略

### 后期执行

文件搬迁应在以下清理之后:

- dead code 已删除。
- action surface 已收敛。
- Coordinator 已拆出主要责任域。
- SharedState 已清理。
- tests 已按行为域重组。

否则搬迁会把旧复杂度搬到新目录里。

### 小步迁移

每次只迁移一个责任域:

- 先迁移 phase hooks。
- 再迁移 framework PR flow。
- 再迁移 specialist/dynamic flow。
- 再迁移 kernel flow。
- 再迁移 fact write。

每次迁移后:

- 替换 import。
- 删除旧 re-export, 如果没有外部 contract。
- 跑对应测试。

### 短期 shim 策略

如果存在外部 import contract, 可以短期保留 shim。但 shim 必须:

- 很薄。
- 有删除时间。
- 不包含业务逻辑。
- 不被新代码使用。

如果只是内部测试依赖旧路径, 应改测试, 不保留 shim。

## 文件命名原则

文件名表达业务职责, 不表达历史阶段:

推荐:

- `proposal_flow`
- `phase_hooks`
- `kernel_flow`
- `framework_pr_flow`
- `specialist_flow`
- `dynamic_action_flow`
- `fact_write`
- `state_migrations`

避免:

- `p1_fix`
- `gap_08`
- `legacy_new`
- `v06_compat`
- `misc`
- `utils` 过度泛化。

## 测试迁移

测试也应跟随行为域:

- runtime contract。
- phase and policy。
- explore flow。
- kernel flow。
- framework PR flow。
- specialist dynamic flow。
- reporting contract。

不要让测试目录镜像所有源码文件。测试目录应帮助理解行为, 而不是暴露内部文件结构。

## 风险

主要风险:

- 大规模 import 替换引入隐藏错误。
- circular import。
- 外部工具仍 import 旧路径。
- package data 或 entry point 漏改。
- tests 因路径迁移大量失败, 掩盖真实行为失败。

控制方式:

- 后期再搬。
- 每次只搬一个责任域。
- 先确认是否有外部 import contract。
- 用 search 检查旧路径引用。
- 每次迁移后跑最小测试。

## 输出物

本阶段应产出:

- 新目录结构设计。
- import direction rule。
- 责任域迁移计划。
- 旧路径 shim 清单。
- 搬迁后的测试布局。

## 完成标准

完成后应达到:

- 文件结构能反映系统职责。
- import 方向清楚。
- runtime core 更薄。
- executors 与 Coordinator 解耦。
- protocol/state/reporting/integrations 边界清晰。
- 新维护者能从目录结构理解系统。
