# ROCm Hyperloom

Hyperloom 是一个在 AMD GPU 上自动优化 LLM 推理的智能体系统。Hyperloom 将优化视为一个**搜索问题**：给定一个 workload，它会构建候选优化树，包括后端切换、服务参数、GEMM 调优、kernel 重写、并行配置等；然后根据预期收益和成本为候选项打分，并以深度优先方式探索，始终基于真实 workload 进行测量。你只需要提供 workload，智能体就会交付完整优化后的代码库：对照硬件峰值能力进行 profiling，识别瓶颈，并迭代重写代码以最大化 AMD GPU 上的吞吐，让团队获得可用于生产的优化代码。

<p align="center"><img width="600" alt="HyperLoom Architecture" src="slides/hyperloom_loop.png" /></p>

Block 1-3 - Workload 理解与 profiling：提交你的 workload 作为起点，让智能体理解代码库，使用 [TraceLens Agentic Analysis](https://github.com/AMD-AGI/TraceLens-internal/) 进行 profiling（依赖 [Magpie](https://github.com/AMD-AGI/Magpie) 收集 trace），捕获瓶颈和 roofline 目标。

Block 4 - 代码优化循环：这是 Hyperloom 的核心。智能体会构建一个带评分的候选树，包括配置覆盖、代码 patch、后端切换、kernel 重写等，并以深度优先方式一次探索一个变更：**Think → Implement → Benchmark → Decide**。每个结果都会重新为剩余候选树评分。

同时，热点 kernel 会通过外部后端异步优化，包括 [GEAK](https://github.com/AMD-AGI/GEAK/tree/main)，以及通过 Claude Code 和 OpenAI Codex 进行的 OOB kernel 优化，后者依赖 [Apex](https://github.com/AMD-AGI/Apex) 的 kernel 优化流程。Kernel profiling 和验证由 [Magpie](https://github.com/AMD-AGI/Magpie) 支撑，而 Magpie 的部分底层 GPU profiling 工具依赖 [IntelliKit](https://github.com/AMDResearch/intellikit)。

Block 5-6 - 验证交付：智能体在保持准确性的同时优化吞吐，每个变更在被接受前都会经过正确性门禁。循环结束后，智能体会打包优化后的代码，向你的 repo 提交 PR，并合入代码库，完成完整闭环。

### 了解更多

| | |
|---|---|
| **[优化循环如何工作](docs/HOW_THE_OPTIMIZATION_LOOP_WORKS.md)** | 评分启发式、优化栈机制、动态分支，以及自演进知识库 |
| **[GLM-5 — 发现人工难以察觉的优化](docs/CASE_STUDY_GLM5.md)** | 隐藏 GEMM 配置、跨 repo kernel patch、吞吐 +193% |
| **[DeepSeek-R1 — 新 workload 的快速扩展](docs/CASE_STUDY_DEEPSEEK_R1.md)** | 单次 session 内 7 个配置达到最优、MTP 调度修复、相对 B200 +97% |

---

## 前置条件

将你的 **[LLM Gateway](https://llm.amd.com/)** key 绑定到 **[Hyperloom](https://core42.primus-safe.amd.com/hyperloom/)**，以获得 `AK_YOUR_API_KEY`。Hyperloom UI 和本地优化流程都需要这个 key，它用于访问 TraceLens、GEAK 和 OOB 服务。

---

## 快速开始 — Hyperloom UI (PrimusClaw)

最快的开始方式是使用托管的 **AMD Hyperloom** Web 界面。该界面由 **[PrimusClaw](https://github.com/AMD-AGI/Primus-Claw)** 提供支持，是面向**大规模可达性**设计的在线托管模式。任何团队成员都可以通过浏览器发起优化，无需本地 GPU 设置或环境配置。

- **易于扩展** — 每个任务都运行在隔离的 sandbox 容器中（GPU 或 CPU）。单节点优化在 sandbox 内运行；多节点 workload 通过 RayJob 扩展到分布式 benchmark。
- **数据飞轮** — 每次优化运行都会通过 Minio 存储和 Langfuse 可观测性回流结果，形成闭环反馈，持续改进智能体的知识库和评分启发式。
- **完整 Skills 支持** — sandbox 按需加载优化 Skills，让智能体在云规模下具备相同的 profiling、kernel 重写和领域能力。

1. 访问 **[core42.primus-safe.amd.com/hyperloom](https://core42.primus-safe.amd.com/hyperloom/)**
2. 在首页选择 **Claw Agent** 或 **Get Started** 进入 PrimusClaw
   <p align="center"><img width="500" alt="Hyperloom Landing" src="slides/hyperloom_landing.png" /></p>
3. Hyperloom（标签页）：端到端模型性能优化
   <p align="center"><img width="500" alt="Hyperloom PrimusClaw UI" src="slides/hyperloom_claw_v2.png" /></p>
4. TraceLens-only（标签页）：模型性能/差距分析与桥接规划
   <p align="center"><img width="500" alt="TraceLens Config" src="slides/tracelens_quickstart.png" /></p>
5. GEAK-only：Kernel 优化
   <p align="center"><img width="500" alt="GEAK Config" src="slides/geak_quickstart.png" /></p>

---

## 快速开始 — Local Mode (Cursor)

### 环境准备

Local Mode 是在一台 AMD GPU 远程环境里运行 Hyperloom，然后用 Cursor 连接进去发起优化。你需要按顺序完成三件事：

1. 准备 GPU 环境。
2. 用 Cursor 连接到这个环境。
3. 在远程环境中 clone Hyperloom，并运行初始化脚本。

#### 1. 准备 GPU 环境

需要一台支持 MI300X 或 MI355X 的 AMD GPU 机器，并使用 SGLang 或 vLLM ROCm 推理镜像。可选镜像示例：

- SGLang MI300X: `lmsysorg/sglang:v0.5.11-rocm720-mi30x`
- SGLang MI355X: `lmsysorg/sglang:v0.5.11-rocm720-mi35x`
- vLLM MI300X: `vllm/vllm-openai-rocm:v0.19.0`
- vLLM MI355X: `vllm/vllm-openai-rocm:v0.19.0`

你可以二选一：

- **SaFE Authoring Pod**：在 [Primus-SaFE Authoring](https://core42.primus-safe.amd.com/authoring) 创建一个 Authoring Pod，选择上面的 SGLang 或 vLLM 镜像，等待 Pod 就绪。
- **自有 GPU 机器**：在你的 GPU 服务器上启动一个能访问 ROCm GPU 的长驻推理容器。下面示例中的容器名、workspace 路径、模型路径和镜像版本都可以按你的环境修改。

自有 GPU 机器的最小 Docker 示例：

```bash
docker run -d \
  --name hyperloom-local \
  --shm-size 64g \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  -v /path/to/workspace:/workspace \
  -v /path/to/models:/models \
  lmsysorg/sglang:v0.5.11-rocm720-mi30x \
  tail -f /dev/null
```

容器启动后，可以先确认容器可进入、GPU 可见：

```bash
docker exec -it hyperloom-local bash
rocm-smi
```

#### 2. 用 Cursor 连接运行环境

- **SaFE Authoring Pod**：Pod 就绪后，在 SaFE Authoring 页面查看连接方式，并按页面提示用 Cursor Remote SSH 连接 Pod。
- **自有 GPU 机器 + Docker**：先用 Cursor Remote SSH 连接运行 Docker 的服务器，再通过 Dev Containers / Attach to Running Container 选择 `hyperloom-local`，打开容器内的 `/workspace`。

> Tips：在 Cursor 中连接服务器上的 Docker 容器详细步骤：
>
> 1. 在 Cursor 中打开命令面板：`Ctrl+Shift+P`。
> 2. 搜索 `Remote-SSH: Connect to Host...`，连接运行 Docker 的服务器。
> 3. 在这个 SSH 远程窗口里打开 Extensions：`Ctrl+Shift+X`。
> 4. 搜索并安装 `Dev Containers`，确保它安装在当前远程环境中。
> 5. 再次打开命令面板：`Ctrl+Shift+P`。
> 6. 搜索 `Dev Containers: Attach to Running Container...`。
> 7. 选择 `hyperloom-local`（或你自己设置的容器名），打开容器内的 `/workspace`。

#### 3. 获取 Hyperloom 并初始化

进入远程环境后，先确保 GitHub 认证和 AMD-AGI 仓库权限可用；`local_setup.sh` 后续也会用这些权限自动 clone 依赖仓库。然后 clone Hyperloom：

```bash
git clone https://github.com/AMD-AGI/Hyperloom.git
cd Hyperloom
```

准备 Hyperloom 凭证：

```bash
cp .env.template .env
```

编辑 `.env`：

```env
SAFE_API_KEY=ak-your-safe-apikey
OPENAI_BASE_URL=https://core42.primus-safe.amd.com/api/v1/llm-proxy/v1
```

| 变量 | 说明 | 示例 |
|----------|-------------|---------|
| `SAFE_API_KEY` | LLM gateway 认证 key | `ak-your-safe-apikey` |
| `OPENAI_BASE_URL` | LLM gateway endpoint | `https://core42.primus-safe.amd.com/api/v1/llm-proxy/v1` |

然后运行 Local Mode bootstrap：

```bash
export USER_DATA_PATH=/path/to/hyperloom-run
bash inference_optimizer/scripts/local_setup.sh
```

`USER_DATA_PATH` 是 Hyperloom 的运行目录，用于保存依赖仓库、日志、状态和优化结果；它不是 Hyperloom 源码目录，可以按需改成其他有足够空间的位置。`local_setup.sh` 会自动推导 `REPO_ROOT`，克隆并接线 OOB、InferenceX、TraceLens 等依赖，并写入本地 env 文件。脚本完成后会打印：

- 需要在 Cursor 中打开的 Hyperloom workspace 路径。
- 需要复制到 Cursor Chat 的 prompt 模板。
- agent 启动前需要 source 的 env 文件路径。

示例输出片段：

```text
Open this folder in Cursor as the workspace:
  /path/to/Hyperloom

Before launching Hyperloom from Cursor Chat, ask the agent to source:
  source /workspace/hyperloom-run/runtime/local-setup.env.sh

Paste this into Cursor Chat and fill in your workload:

@inference_optimizer/SKILL.md

Optimize inference for this workload:
- Model: /path/to/your/model
- Framework: sglang
- GPU: MI300X
- TP: 8
- CONC: 64
- ISL: 1024
- OSL: 1024
- Precision: bf16
- Goal: improve throughput by at least 10%
- Budget: 24 hours

Before launch:
1. Source /path/to/hyperloom-run/runtime/local-setup.env.sh
2. Use USER_DATA_PATH=/path/to/hyperloom-run

Requirements:
1. Report the session ID, log path, PID, and initial health check result.
2. Monitor the process every 300s until the optimization is complete or failed.
```

按脚本输出操作即可；默认流程下，用户不需要手动配置 GEAK、OOB、InferenceX 或 TraceLens。

**可选（Cursor kernel-opt 后端）：**

| 变量 | 说明 | 示例 |
|----------|-------------|---------|
| `CURSOR_API_KEY` | OOB cursor 后端使用的 Cursor SDK key；它来自独立发行方（Cursor 账号，前缀 `crsr_...`）。未设置时，Hyperloom 会从默认后端选择中自动跳过 cursor，只使用 claude/codex/geak。 | `crsr_xxxxxxxxxxxx` |
| `CURSOR_DEFAULT_MODEL` | 覆盖默认 Cursor model id。 | `claude-opus-4-7`（默认） |

> `SAFE_API_KEY` 可从 [LLM Gateway](https://core42.primus-safe.amd.com/litellm-gateway) 获取。GEAK 和 OOB（claude/codex）的 API Key / Base URL 会自动继承 `SAFE_API_KEY` / `OPENAI_BASE_URL`。你可以把这些值放在 `$REPO_ROOT/.env` 中；不需要单独配置 GEAK、OOB、InferenceX 或 TraceLens。OOB **cursor** 后端是例外：它访问 Cursor 自己的 gateway，需要单独的 `CURSOR_API_KEY`。如果未设置 `CURSOR_API_KEY`，cursor 会从默认 kernel-opt 选择中静默跳过。

### 开始推理优化

环境准备完成后，按 `local_setup.sh` 的输出在 Cursor 中打开 Hyperloom workspace，并把脚本打印的 prompt 模板复制到 Cursor Chat。按需替换模型路径、框架、GPU 类型、预算及其他需要的参数后发送即可。

**恢复已有 session：**

示例 prompt：

```text
@inference_optimizer/SKILL.md

Resume the existing Hyperloom optimization session.

Requirements:
1. Launch `inference_optimizer optimize --resume`; do not start a new session.
2. Do not pass `--model`; read the model and workload from the saved manifest/state.
3. Before launching, verify `manifest.json` and `state.json` exist.
4. Report the log path, PID, initial health check result, current phase, cumulative gain, and best config.
5. Monitor the process every 300s until the optimization is complete or failed.
```

Prompt 参数参考：

| 字段 | 对应设置 | 说明 | 默认值 |
|---|---|---|---|
| `Model` | `--model`, `MODEL_PATH` | 模型路径。新运行必填；恢复已有 session 时忽略。 | 必填 |
| `Framework` | `--framework`, `FRAMEWORK` | 推理框架：`sglang` 或 `vllm`。一个 session 内不要混用框架。 | `sglang` |
| `GPU` | `--gpu-type`, `GPU_TYPE` | 目标 GPU 类型，例如 `MI300X`、`MI325X` 或 `MI355X`；也可以自动检测。 | 自动检测 |
| `Model class` | `--model-class` | 模型架构类型，用于 action 选择和评分。 | 未设置 |
| `Compare against GPU` | `--compare-against-gpu` | 可选外部参考 GPU，例如 `B200`。未设置时仍会继续优化，只是不拉取外部参考 baseline。 | 未设置 |
| `TP` | `TP` | Tensor parallel size。 | `1` |
| `CONC` | `CONC` | Benchmark concurrency。 | YAML 默认值，通常为 `8` |
| `ISL` | `--isl`, `ISL` | 输入序列长度。 | `256` |
| `OSL` | `--osl`, `OSL` | 输出序列长度。 | `256` |
| `Precision` | `--precision`, `PRECISION` | 模型精度，例如 `bf16`。 | `bf16` |
| `Goal` | `--target-gain`, `--target-tput`, `--target-baseline-dir` | 可选停止条件，例如目标吞吐提升比例。 | 未设置 |
| `Budget` | `--max-hours` | 最大优化时长。 | `2.0` 小时 |
| `Kernel optimization` | `--no-kernel` | 默认启用 kernel 优化；如果只想做参数/后端搜索，可以在 prompt 中说明跳过 kernel 优化。 | 启用 |
| `Resume` | `--resume` | 恢复已有 session；需要已有 `manifest.json` 和 `state.json`。 | 关闭 |

首次启动错误请参考 `inference_optimizer/SKILL.md` 的 §"Failure Handling"。

---

## 关键结果

### 推理优化 — InferenceX Challenge

Hyperloom 在 AMD Instinct MI355X 上为 [InferenceX](https://github.com/SemiAnalysisAI/InferenceX) benchmark 优化了 4 个旗舰模型，其中 3/4 个模型达到或超过 NVIDIA B200。

| Model | Best tok/s/GPU | vs MI355X Baseline | vs NVIDIA B200 |
|-------|---------------:|:------------------:|:--------------:|
| DeepSeek-R1-0528 (671B MoE) | **1,476** | — | **+97% ahead** |
| GLM-5-FP8 (756B MoE+NSA) | **509** | **+193%** | **+27% ahead** |
| Qwen3.5-397B (397B MoE) | **350** | **+40%** | **+2.5% ahead** |
| MiniMax-M2.5 (MoE 256E) | **2,276** | **+6.5%** | **+5.7% ahead** |
| gpt-oss-120b (120B MoE, mxfp4) | **11,643** | — | **+34% ahead** |

所有 benchmark：ISL=1024，OSL=1024，运行于 MI355X (gfx950)。“vs B200” 展示最佳 concurrency 点。完整 concurrency/ISL/OSL sweeps、patches、configs 和复现脚本请见：**[Agentic-InferenceX](https://github.com/AMD-AGI/Agentic-InferenceX)**。

## 详细 Skill 文档

本 repo 提供一个 skill：`inference_optimizer/`。它包含完整优化协议、示例，以及从历史运行中沉淀的经验知识库：

| 领域 | Skill | 说明 |
|--------|-------|-------------|
| **Inference** | [SKILL.md](inference_optimizer/SKILL.md) | 多智能体系统，CLI 驱动，全自动 |

Skill 文件是智能体的操作说明。它编码了完整优化方法：setup、profiling 协议、尝试什么、如何测量、何时停止，以及如何汇报。知识库章节会在运行期间随着新陷阱和验证结果持续更新。

---

## Repo 结构

```
Hyperloom/
├── inference_optimizer/                  # Inference optimization skill（唯一入口）
│   ├── SKILL.md                          # Skill spec（Cursor/Claw 入口）
│   ├── cli.py                            # CLI 入口：inference_optimizer optimize
│   ├── actions/_meta/                    # Action metadata 和调度策略
│   ├── baseline_comparison/              # InferenceX baseline comparison 和 target analysis
│   ├── orchestrator/                     # Coordinator + agent roles + action executors
│   │   ├── action_executors/             # baseline/profile/params/sweep 等 executor
│   │   ├── backends/                     # Claude/Codex/Critic backend adapters
│   │   └── system_prompts/               # Orchestration prompt construction
│   ├── scripts/                          # Install scripts、baseline/profile configs
│   └── tests/                            # Inference optimizer unit 和 regression tests
├── kernel-agent/                         # Kernel Agent toolkit（TraceLens/GEAK/OOB tools）
│   ├── SKILL.md                          # Kernel Agent operation spec
│   ├── tools/                            # TraceLens analysis、kernel optimization、patch apply
│   │   └── backends/                     # GEAK/OOB submission（Ray-scheduled）
│   ├── scripts/                          # Runtime setup scripts：install.sh、auth proxy 等
│   └── tests/                            # Kernel Agent tool tests
├── critic-agent/                         # Critic-agent subprocess runtime（proposal review）
├── robustness-agent/                     # Robustness-agent subprocess runtime（health/RCA）
├── ci/                                   # CI orchestration（PR submitter、AB test）
├── docs/                                 # Architecture docs、case studies 和 Mermaid diagrams
├── scripts/                              # Repo-level helper scripts
├── .env.template                         # Environment variables
└── README.md
```
