# Hyperloom 项目简报

## 一、项目概述

**ROCm Hyperloom** 是一套面向 AMD GPU（ROCm）的**智能体式 LLM 优化系统**。它把"推理/训练调优"当作**搜索问题**——根据用户负载构建一棵带评分的候选优化树（换后端、改服务参数、GEMM/内核调优、并行配置等），然后**深度优先**探索；每一步在真实负载上**基准测试**，并在**正确性**约束下追求更高吞吐。

核心流程：理解负载与剖析 → **思考 → 实现 → 跑分 → 决策** 闭环 → 交付可合并的优化结果。

---

## 二、主要目录用途

| 目录 | 用途 |
|------|------|
| `.cursor/` | Cursor 侧配置：`mcp.json` 连接 TraceLens、GEAK、OOB 等 MCP；`skills/` 里是推理/训练优化的 Skill 指令与知识库 |
| `inference_optimization/` | 推理侧工作区，内含 `InferenceX/`：推理基准与相关工具、实验脚本、多硬件启动脚本 |
| `training_optimization/` | 训练侧工作区，内含 `turboquant/`：量化评估相关的库与脚本 |
| `dashboards/` | 交互式 HTML 看板（优化时间线、搜索树等），用于可视化优化过程与结果 |
| `slides/` | 架构图、示意图及生成图表用的脚本 |
| `docs/` | 深度文档与案例研究：优化循环原理、GLM-5/DeepSeek-R1 案例 |

---

## 三、关键组件解释

### GEMM（通用矩阵乘法）

不是外部工具，而是 GPU 计算中最基础的运算。LLM 的线性层、注意力投影、MoE 路由都归结为大量 GEMM。Hyperloom 通过 TraceLens 分析 GEMM 占 GPU 时间比例，然后：
- 发现缺失的 GEMM 调优配置（如 GLM-5 案例中 +21%）
- 调整 ksplit、tile 等参数充分利用 GPU
- 扫描 55+ 种 dense GEMM shape 自动调优

### TraceLens（性能剖析工具）

AMD 的智能体式性能分析服务，作为 MCP Server 接入。负责：
- 采集 GPU kernel 级 trace（依赖 Magpie）
- 给出每个 kernel 占 GPU 时间百分比
- 做 roofline 分析，量化与硬件峰值的差距
- 分析结果直接驱动搜索树评分

相当于 Hyperloom 的**"眼睛"**。

### Magpie（trace 采集框架）

AMD 的 GPU trace 收集工具，底层依赖 IntelliKit。作为 TraceLens 的**数据采集层**——TraceLens 做高层分析，Magpie 负责在 GPU 上抓取 kernel 执行 trace。也为 kernel 优化后的验证提供 profiling 支持。

### GEAK（AI 驱动的 kernel 优化服务）

AMD 的远程 kernel 优化平台，通过 MCP 接入。当 agent 发现热点 kernel（占 GPU 时间 >2%，有可修改源码）时，提交给 GEAK。GEAK 在**远程 GPU Pod** 上用 AI 生成优化 kernel，**自带验证环境**。适用于 Triton/HIP kernel，不适用于已手工优化的厂商 BLAS kernel。典型耗时 10–30 分钟/轮。

### OOB（Out-of-Band kernel 优化）

另一条 kernel 优化路径，通过 OOB Agent MCP 调用 OpenAI Codex 或 Claude Code 生成优化 kernel。与 GEAK 的关键区别：**OOB 没有自己的 GPU**，只生成代码，验证由 Hyperloom 本地完成。Codex 每轮 2–6 分钟（3 次迭代），Claude 每轮 3–15 分钟。

### 协同工作流

```
Magpie 采集 trace → TraceLens 分析瓶颈 → Hyperloom DFS 循环
  ├─ 配置调优（参数、后端切换）
  ├─ GEMM 调优（补配置、ksplit、auto-tune）
  ├─ GEAK（远程 GPU AI 优化 kernel）
  └─ OOB（Codex/Claude 生成 kernel → 本地验证）
每次改动后 benchmark + 正确性校验 → 保留或回滚
```

---

## 四、Prompt 与聊天管理

Hyperloom **没有传统意义上的聊天管理代码**。它不是独立运行的应用，而是寄生在 Cursor IDE / Claude Code / Claw 上的智能体系统。

### "Prompt"就是 `.cursor/skills/`

| 文件 | 角色 |
|------|------|
| `SKILL.md` | **主 prompt**：定义完整 DFS 优化协议（目标、铁律、评分函数、搜索树、循环流程、状态 schema、停止条件） |
| `actions/*.md` | **子 prompt**：每个 action 的独立指令模块（classify、baseline、profile、params、kernel-opt 等） |
| `kernel-opt/*.md` | Kernel 优化后端指令（geak.md、codex.md、claude.md、llm.md） |
| `kb/entries.jsonl` + 脚本 | RAG 知识库：action 前查询历史经验，action 后写回新发现 |
| `modes/LOCAL.md` / `CLAW.md` | 执行模式指令（本地 vs 云端） |

### "聊天管理器"就是 Cursor 本身

- 用户在 Cursor 聊天框 `@SKILL.md` 并描述工作负载
- Cursor 读取 SKILL.md 作为 agent 指令，自主执行
- `.cursor/mcp.json` 配置工具接口（TraceLens / GEAK / OOB）
- `.cursor/hooks/knowledge-sink.py` 是对话感知 hook——在对话结束时检测优化结果，自动提示 agent 写入知识库

---

## 五、云端 UI（PrimusClaw）

浏览器访问 `oci-slc.primus-safe.amd.com/hyperloom`，零配置使用。

### 三个标签页

| 标签 | 功能 | 等价于本地的什么 |
|------|------|-----------------|
| Hyperloom | 端到端自动优化（基线→分析→DFS循环→交付） | Cursor `@SKILL.md` 跑完整优化 |
| TraceLens | 只做性能分析（kernel 瓶颈、roofline 差距） | 只调 TraceLens MCP |
| GEAK | 只做 kernel 优化（提交热点 kernel 源码） | 只调 GEAK MCP |

### 比本地多了什么

- **沙箱隔离**：每个任务跑在独立容器，多节点通过 RayJob 分发
- **数据飞轮**：MinIO 存储 + Langfuse 可观测性 → 持续改进 agent
- **完整 MCP + Skills 支持**：与本地能力一致

---

## 六、MinIO 与 Langfuse

### MinIO（对象存储）

开源对象存储服务（类似 AWS S3）。在云端模式中充当持久化存储层——trace 文件、benchmark 结果、优化后的 kernel、配置快照都存在 MinIO，多次运行和多个沙箱之间通过 MinIO 共享数据。

### Langfuse（LLM 可观测性平台）

开源 LLM 观测平台。记录 agent 每一步行为：决策、MCP 调用输入输出、benchmark 结果、耗时、token 消耗。构成"数据飞轮"的可观测性一环，便于团队回溯和改进。

---

## 七、云端调用的 API

云端 PrimusClaw 用自己的 agent runtime 替代 Cursor，但读同一套 SKILL.md、调同一批 MCP 服务：

| API / 服务 | 用途 |
|------------|------|
| **LLM Proxy** | Agent 的推理能力（Claude/GPT），通过 AMD LLM Gateway |
| **SaFE API** | GPU 集群资源调度（RayJob 创建/管理） |
| **TraceLens MCP** | 性能剖析 |
| **GEAK MCP** | Kernel 级优化 |
| **OOB Agent MCP** | Codex/Claude kernel 优化 |

### 本地 vs 云端对比

| | 本地模式 | 云端模式（PrimusClaw） |
|---|---|---|
| Agent runtime | Cursor IDE | PrimusClaw 托管平台 |
| LLM 调用 | Cursor 内置 Claude | AMD LLM Gateway / PRISM Proxy |
| GPU | 本地 GPU | SaFE 集群（RayJob 自动分配） |
| MCP 调用 | 完全一样 | 完全一样 |
| SKILL.md | Cursor 读取 | PrimusClaw 加载 |
| 数据存储 | 本地磁盘 | MinIO + Shared NFS |
| 可观测性 | 无 | Langfuse 自动记录 |
