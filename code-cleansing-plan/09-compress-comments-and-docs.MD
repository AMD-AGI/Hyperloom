# Step 09: Compress Comments And Docs

## 本计划的核心原则

这个代码清理是为了大幅度降低这个代码库的工程逻辑复杂度, 降低维护难度。清理只需要保证端到端功能一样, 不需要保证单元性的功能一致。当前代码库已经非常庞大, 所以不要吝啬 cut off 代码。

具体原则:

- 注释服务当前维护, 不服务历史复盘。
- 文档描述当前系统, 不描述所有中间演进过程。
- 能从代码直接读出来的内容不需要注释重复。
- 长注释只有在解释外部契约或不变量时才值得保留。

## 目标

当前代码中有大量过程性注释: PR 编号、Gap 编号、Nxx 编号、历史 bug 叙述、曾经的实现方式、迁移原因、开发过程中的阶段说明。这些注释在当时有帮助, 但长期看会显著增加阅读成本。

本阶段目标:

- 保留当前行为需要的短注释。
- 删除过期历史叙事。
- 压缩重复 docstring。
- 修正文档和代码不一致的地方。
- 让 prompt / SKILL / README 不再误导 LLM 和维护者。

## 注释保留标准

### 必须保留

以下注释有长期价值:

- 外部协议说明。
- 数据格式约束。
- 安全 gate 的原因。
- 不明显的不变量。
- 兼容逻辑的删除条件。
- subprocess 或 artifact path 的边界。
- 为什么某个看似多余的 guard 必须存在。

### 应该删除

以下注释应删除:

- 与代码逐行重复的注释。
- PR / milestone / temporary branch 叙事。
- “之前为什么坏过”的长故事, 如果不再解释当前设计。
- 已经过期的设计状态。
- 对已删除模块的说明。
- 与函数 docstring 重复的块注释。

### 应该压缩

以下注释可以压缩成一两句:

- migration 背景。
- legacy alias 原因。
- multi-node 特殊处理。
- benchmark timeout 背景。
- robustness downgrade 背景。

压缩目标是回答:

- 当前为什么这样做?
- 删除条件是什么?

不需要回答:

- 当时哪个 PR 引入。
- 哪个临时 bug 编号。
- 过去尝试过哪些实现。

## 文档清理范围

### README

保留:

- 项目目的。
- 快速启动。
- 核心架构。
- 当前支持的 framework / GPU / mode。

删除或压缩:

- 过期 quickstart。
- 已删除 action。
- 未实现 future feature 的实现性描述。
- 重复环境变量表。

### SKILL

SKILL 会直接影响 agent 行为, 必须更严格。

保留:

- 当前启动规则。
- iron rules。
- session path。
- live action protocol。
- failure handling。

删除:

- retired modules。
- 旧 workflow。
- 过长历史叙事。
- 未实现工具的使用建议。

### System prompts

System prompt 应短而强, 不应包含大量历史设计说明。

保留:

- 当前可用 action。
- phase contract。
- intent output protocol。
- hard safety rules。

删除:

- old action 解释。
- old scoring narrative。
- 与 runtime 不一致的指令。
- 长篇背景故事。

### Code comments

代码注释应围绕不变量和边界。

每个核心文件应经过一次 comment pass:

- 删除重复说明。
- 压缩历史叙事。
- 修正 stale header。
- 保留少量关键 guard 注释。

## 注释风格

推荐风格:

- 一句说明当前 intent。
- 一句说明删除条件, 如果是 compat。
- 不引用 PR 编号, 除非该编号是外部文档唯一索引。
- 不使用长段落解释开发过程。

示例:

```text
Legacy state migration for archived sessions. Remove after old-session
resume support is dropped.
```

替代:

```text
PR-X / Gap-Y / N33 introduced this after a long sequence of failures...
```

## 执行顺序

推荐顺序:

1. 修正明显 stale 的文件头。
2. 清理 prompt 和 SKILL, 避免 LLM 继续遵循旧规则。
3. 清理 Coordinator / SharedState / Policy 的长注释。
4. 清理 action executor 中的过程叙事。
5. 清理 tests 中的过长 bug story。
6. 清理 README / docs 中的旧流程。

## 测试调整

注释清理本身不应需要大量测试。但如果 prompt 文案被测试锁死, 应把测试改成验证核心 contract, 而不是验证完整句子。

保留:

- prompt 包含 live action。
- prompt 不包含 retired action。
- prompt 包含必要 output protocol。

删除:

- 对大段 prompt 精确文本的断言。
- 对注释中历史编号的断言。
- 对 markdown 排版细节的断言。

## 风险

主要风险:

- 删除了真正解释安全 gate 的注释。
- prompt 压缩后 LLM 不再知道关键规则。
- README 删除过多, 新用户无法启动。
- compat 删除条件不清楚, 后续维护者误删。

控制方式:

- 注释压缩不和行为修改混在一起。
- Prompt 清理后跑 prompt visibility / action catalogue 相关测试。
- 对每个 compat block 保留一句删除条件。
- README/SKILL 保留当前可执行路径。

## 输出物

本阶段应产出:

- 精简后的 core file headers。
- 精简后的 SKILL 和 prompt。
- 删除 stale docs 的清单。
- 注释风格规则。

本阶段的执行记录、注释风格规则与 stale-doc 清单见
`code-cleansing-plan/09-comments/01-execution-status.MD`。

## 完成标准

完成后应达到:

- 注释明显变短。
- 文件头描述当前实现。
- Prompt 不再包含 retired action 或 stale workflow。
- 维护者读注释时看到的是当前不变量, 不是开发历史。
