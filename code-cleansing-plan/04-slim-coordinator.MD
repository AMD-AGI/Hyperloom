# Step 04: Slim The Coordinator

## 本计划的核心原则

这个代码清理是为了大幅度降低这个代码库的工程逻辑复杂度, 降低维护难度。清理只需要保证端到端功能一样, 不需要保证单元性的功能一致。当前代码库已经非常庞大, 所以不要吝啬 cut off 代码。

具体原则:

- `Coordinator` 应表达主流程, 不应承载所有历史细节。
- 拆分以责任边界为准, 不以文件长度为准。
- 内部 helper 可以合并、删除、改名, 只要端到端行为不变。
- 先消除重复逻辑, 再做文件拆分。

## 目标

`Coordinator` 是当前系统最大的复杂度中心。它同时承担:

- agent reactor loop。
- intent routing。
- proposal / critic / materialization。
- phase transition side effects。
- task dispatch。
- framework PR pump。
- specialist lifecycle。
- dynamic action lifecycle。
- kernel request and promotion。
- roofline/warm replay/close sequence。
- KB and journal fact writes。

这些职责集中在一个文件里会导致:

- 新功能容易继续往同一个文件堆。
- 清理时难以确认修改范围。
- 注释越来越长, 因为局部代码需要解释太多全局背景。
- 测试更偏向私有 helper, 而不是行为契约。

本阶段目标是让 `Coordinator` 重新成为主流程编排器, 细节下沉到责任清晰的模块中。

## 先消重, 后拆分

拆大文件前必须先消重。否则只是把重复逻辑分散到多个文件。

优先消重方向:

- resume rebuild / drain / abandon dynamic action 的重复流程。
- backend health tracking 的重复初始化。
- roofline pending gate 的重复检查。
- FRAMEWORK_PR pump 的重复包装。
- prompt time budget block 的重复渲染。
- `extra_server_args` 兼容读取和合并入口。
- task result promotion 后的共通 hook。

消重标准:

- 不新增抽象层级过深的 helper。
- helper 名称表达业务含义, 不表达历史补丁名。
- helper 返回清晰结果, 不靠多个外部副作用隐式配合。

## 推荐责任拆分

### Coordinator core

保留:

- construction。
- `run()` / `tick()` 主循环。
- `_reactor_pass`。
- `_handle_intent` 分发入口。
- `_pump_dispatcher_once` 高层调度框架。
- phase advance 的主入口。

不保留:

- 每个 phase 的具体 enqueue 细节。
- 每种 proposal 的具体 materialization 细节。
- 每个 specialist/dynamic/kernel/framework PR 的生命周期细节。
- KB fact write 的长逻辑。

### Phase hooks

负责:

- phase entry side effects。
- prelude analysis enqueue。
- framework phase entry。
- kernel phase entry。
- sweep phase entry。
- close phase sequencer。

原则:

- phase module 不应该知道过多 agent backend 细节。
- 每个 phase hook 只表达进入该阶段应该自动发生什么。
- phase transition 判断仍归 `phase_state`。

### Proposal flow

负责:

- proposal pending lifecycle。
- Critic verdict handling。
- approve/reject/redirect/needs_review route。
- materialize approved proposal。
- verdict map for explore grid。

原则:

- Critic 是 gate, 不是 executor。
- materialization 只创建 task, 不直接执行 task。
- batch verdict 和 single verdict 可以共享结构。

### Framework PR flow

负责:

- candidate discovery。
- batch progress。
- candidate selection。
- critic pre-gate。
- framework_pr task enqueue。
- phase done / plateau / force exit markers。

原则:

- Framework PR 是 Coordinator-owned phase, 不应混在普通 LLM proposal flow 中。
- 只保留当前实际 wired 的 framework-agent 协议。
- 未实现的 future framework role 不进入主流程。

### Specialist flow

负责:

- specialist params warmup。
- KnowledgePlane PR/KB context selection。
- specialist result recording。
- proposal_set harvesting。
- steward verdict。
- research scout harvest。

原则:

- Specialist 只生成提案或研究信号。
- 真正 patch apply 只走 integrate_patch。
- research scout 是 read-only advisory, 不应和 mutation flow 混合。

### Dynamic action flow

负责:

- dynamic action dispatch spec。
- seed kit。
- runner result。
- mechanical critic pre-verdict。
- dynamic proposal materialization。
- integrate_patch result backfill。

原则:

- Dynamic action 是 EXPLORE 的补充通道, 不应成为第二套 patch integration system。
- 最终仍复用 Critic 和 integrate_patch。

### Kernel flow

负责:

- kernel request routing。
- programmatic handler result。
- trace analyze cache。
- kernel optimization record。
- integrate keep/revert。
- hot kernel gate。

原则:

- Kernel-owned action 不走 ordinary delegate。
- Kernel handler 是唯一 kernel-agent bridge。
- shape/path/safety gate 必须保留。

### Fact write flow

负责:

- optimization journal。
- RecipeKB write。
- KEEP / REVERT fact projection。
- close summary write。

原则:

- Fact write 不应该影响核心优化流程成功与否。
- KB failure 是降级, 不是 runtime abort。
- 所有 fact write 都应该从 terminal task result 触发。

## 拆分顺序

推荐顺序:

1. 消除明显重复逻辑。
2. 抽 phase hooks。
3. 抽 framework PR flow。
4. 抽 specialist flow。
5. 抽 dynamic action flow。
6. 抽 kernel flow。
7. 抽 fact write flow。
8. 再回头压缩 Coordinator 主文件。

每一步都应保持单一主题, 不混入测试重组或注释大清理。

## 测试策略

保留的测试关注:

- `run()` / `tick()` 主循环仍能推进。
- proposal 到 task materialization 不破。
- phase entry side effects 不破。
- framework PR phase 不破。
- specialist/dynamic action 最终仍接入 integrate_patch。
- kernel request handler 仍能被 inline route。
- terminal task result 仍能 promote 到 SharedState。

可删除或改写:

- 对 Coordinator 私有 helper 名字的测试。
- 对内部 pending dict 的过细断言。
- 对历史分支注释中提到的中间状态的测试。

## 风险

主要风险:

- 拆分后引入 circular import。
- 私有 helper 移动后丢失副作用。
- phase hooks 和 dispatcher 的调用顺序改变。
- task promotion 和 fact write 顺序改变。
- resume path 在主循环和 tick 测试中出现差异。

控制方式:

- 每次只移动一个责任域。
- 保持 public loop 调用顺序不变。
- 把副作用集中在明确的 result handler。
- 使用端到端 flow test, 不依赖 helper 名称。

## 输出物

本阶段应产出:

- 一个更短的 Coordinator core。
- 若干按责任划分的 runtime flow 模块。
- 简化后的主循环。
- 删除重复 helper 和重复注释。
- 更新后的行为测试。

## 完成标准

完成后应达到:

- 维护者能在 Coordinator 中看懂主流程。
- 具体业务域可以在对应模块中阅读。
- 新增一个 phase hook 或 action flow 不需要修改 12k 行文件。
- Coordinator 不再是所有历史兼容和 feature patch 的堆积点。
