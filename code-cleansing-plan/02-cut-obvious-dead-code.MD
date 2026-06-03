# Step 02: Cut Obvious Dead Code

## 本计划的核心原则

这个代码清理是为了大幅度降低这个代码库的工程逻辑复杂度, 降低维护难度。清理只需要保证端到端功能一样, 不需要保证单元性的功能一致。当前代码库已经非常庞大, 所以不要吝啬 cut off 代码。

具体原则:

- 先删除确定无端到端价值的代码, 不先做大规模搬家。
- 删除代码时以 live contract 为准, 不以历史注释或旧测试为准。
- 对明显退场的功能保持删除倾向, 不继续包兼容层。
- 删除后只需要保持端到端流程和外部契约, 不需要保留旧 helper 行为。

## 目标

本阶段目标是快速降低噪音, 删除没有争议的 dead code、backup artifacts、stale docs 和已退场路径。这个阶段不应该触碰最复杂的 runtime 编排, 也不应该做大量抽象重构。

优先删除:

- 明显 backup 文件。
- 不再存在文件的配置引用。
- 已经被 live action catalogue 排除的 retired 正向路径。
- 只在文档中过度承诺、但没有实现的 feature 描述。
- 开发过程中遗留的临时脚本和 demo。

## Dead Code 判断标准

一段代码满足以下任意条件, 可以进入删除候选:

- 不在 live action catalogue 中。
- 不被 CLI wiring、phase allowlist、PolicyGate、kernel request handler、session path、breakdown producer 使用。
- 只被 retired feature 的测试引用。
- 只在注释或文档中出现, 没有 runtime 入口。
- 只服务已经明确不再支持的旧兼容路径。
- 是 `.bak`, 临时输出, 旧生成物, 或人工复制的历史文件。

一段代码暂不删除的条件:

- 通过 subprocess、JSON、SQLite、artifact path 或 env 间接调用。
- 是 external contract 的 reader 或 writer。
- 是旧 session 兼容, 且产品尚未决定放弃旧 session。
- 是 agent envelope mirror 或 downstream dashboard contract。
- 是安全 gate, 即使调用路径不明显。

## 删除优先级

### 第一类: 文件级垃圾

这些文件通常可以最快处理:

- `.bak` 文件。
- 老的临时脚本。
- 不再链接的 demo。
- 空 scaffold。
- 已经不在 package data 或 CLI 中引用的样例。

处理方式:

- 先确认不在 live contract 内。
- 直接删除文件。
- 删除对应 stale test 或 doc reference。

### 第二类: 配置级垃圾

典型包括:

- coverage omit 中已经不存在的路径。
- pyproject 中旧 marker 或旧 testpath。
- package-data 中指向已删文件的条目。
- CI 文档里指向旧执行方式的描述。

处理方式:

- 删除配置项。
- 不为旧路径保留空文件。
- 不为旧路径新增 alias。

### 第三类: retired feature 残留

典型 retired 概念:

- old action: `setup`, `classify`, `backends`, `params`, `validate_stack`。
- old request: `select_kernels`。
- old module: scoring / scoreboard。
- old process: parliament, vote, objection。
- old KB protocol: v1 graph hypothesize / verify, 如果已经被 RecipeKB 替代。

处理方式:

- 删除正向执行路径。
- 删除 prompt 中鼓励使用它们的内容。
- 删除正向测试。
- 仅在必要 migration 层保留旧名字识别。

### 第四类: 未发布功能 scaffold

如果某功能只存在于 README/SKILL/注释中, 但没有实际 runtime wiring, 应避免继续维持“即将实现”的复杂描述。

处理方式:

- 要么删除文档承诺。
- 要么明确标记为 future, 不进入当前 runtime 复杂度。
- 不为未来功能保留当前主流程中的分支。

## 删除策略

### 小批量删除

每次删除应按主题进行:

- 一批 backup 文件。
- 一批 stale config。
- 一个 retired action。
- 一个 old KB path。
- 一个未实现 feature scaffold。

不要把文件搬迁、抽象重构和删除混在同一批, 否则很难判断行为变化来源。

### 删除同时删除测试负担

如果删除的是 dead path, 对应的正向测试也应该删除或改为 contract test。不要保留测试去证明 dead path 不再工作, 除非该拒绝行为本身是外部契约。

例如:

- 可以保留 “retired action 被拒绝” 的策略测试。
- 不应保留 “retired action 内部 helper 正常运行” 的测试。

### 删除同时压缩文档

删除代码后应同步删除:

- README 中的旧流程。
- SKILL 中的旧命令。
- system prompt 中的旧动作。
- 注释中的旧故事。

否则 LLM 或维护者会继续被旧路径误导。

## 风险

主要风险:

- import 图无法发现 subprocess 调用。
- artifact reader 看似无调用, 但 dashboard 依赖输出字段。
- old session migration 看似 dead, 但 resume 或 offline breakdown 仍使用。
- prompt 中出现 retired path, 导致 LLM 继续提出旧 action。

控制方式:

- 删除前检查 live contract。
- 对字符串协议做文本检索。
- 对 artifact path 做 session layout 检查。
- 对 downstream contract 做 schema 检查。
- 删除 prompt 中的旧行为引导。

## 输出物

本阶段应产出:

- 删除文件列表。
- 删除 retired concept 列表。
- 删除配置项列表。
- 保留但待决的 compat 列表。
- 被删除测试列表。

本阶段第一批低风险清理的执行记录见
`code-cleansing-plan/02-dead-code/01-execution-status.MD`。

## 完成标准

完成后应达到:

- 明显无用文件消失。
- retired action 的正向路径消失。
- stale config 不再引用不存在文件。
- 文档不再把未实现功能描述为已实现功能。
- 代码库第一层噪音明显下降。
