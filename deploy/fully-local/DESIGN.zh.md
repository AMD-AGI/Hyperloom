# Hyperloom 全本地模式设计文档

> 面向用户自有基础设施的本地节点支持。

## 1. 概述与动机

### 1.1 什么是全本地模式

全本地模式允许用户在**自有 GPU 基础设施**上运行完整的 Hyperloom 推理优化流程，无需依赖 AMD 托管的 PrimusClaw 沙箱或 Primus-SaFE 创作 Pod。

"全本地"意味着**本地 Agent + 本地 GPU** — Agent（Cursor IDE）在本地运行，基准测试在本地 GPU 上执行，**所有优化工具以容器内 CLI 的形式调用**（TraceLens、GEAK 通过 `geak_ray_submit.py` 经 Ray 调度、OOB 通过 `oob_ray_submit.py run` 经 Ray 调度）。

全本地模式支持**两种部署方式**：

| | 方式 A：预构建镜像 | 方式 B：BYOI（Bring Your Own Image） |
|--|--|--|
| 镜像来源 | Hyperloom 官方镜像（`Dockerfile.sglang` / `Dockerfile.vllm`） | 用户任意镜像（需满足最小依赖） |
| 依赖安装 | 构建时完成，开箱即用 | Agent SSH 连入后通过 `bootstrap.sh` 自动安装 |
| 适用场景 | 标准部署、快速上手 | 用户已有镜像、特殊驱动版本、内部镜像仓库 |
| entrypoint | Hyperloom `entrypoint.sh`（自动启动所有服务） | 用户自定义（Agent 负责引导） |

### 1.2 为什么需要全本地模式

| 场景 | PrimusClaw（云托管） | 全本地（用户自有基础设施） |
|------|---------------------|--------------------------|
| 用户有自己的 GPU 集群 | 数据上传到 AMD 云端 | 数据留在用户网络内 |
| 气隙网络/私有网络 | 需要公网连接 | 仅需 LLM API 出站连接 |
| 自定义镜像/驱动版本 | 受限于平台镜像 | 用户完全控制基础镜像 |
| 快速迭代与调试 | 任务排队、沙箱启动 | 持久化容器、SSH 直连、零排队 |
| 多节点/特殊拓扑 | 平台调度 | 用户自行编排 |
| 工具集成 | 远程 MCP 服务 | 本地 CLI |

### 1.3 与 PrimusClaw 的关系

```
┌──────────────────────────────────────────────────────────┐
│                  Hyperloom 优化引擎                        │
│           (Skills + 优化循环 + 知识库)                     │
├─────────────────────────┬────────────────────────────────┤
│   PrimusClaw 模式        │      全本地模式                  │
│   ────────────────       │    ──────────────────           │
│   Web UI 提交任务        │    Cursor SSH 连入容器            │
│   AMD 云端沙箱           │    用户自有 GPU 节点              │
│   远程 MCP 连接          │    容器内仅 CLI                  │
│   Minio + Langfuse       │    本地日志 + 文件系统            │
│   RayJob 分布式          │    Docker / K8s Pod             │
└─────────────────────────┴────────────────────────────────┘
```

两种模式共享相同的 Skill 文件、优化方法论和知识库。唯一区别在于**部署拓扑和资源归属权**。

---

## 2. 架构

### 2.1 高层架构

```
用户笔记本/工作站                      用户 GPU 节点
┌──────────────────┐                    ┌────────────────────────────────────┐
│                  │                    │  Hyperloom 容器                    │
│  Cursor IDE      │◄── SSH (22) ──────►│                                    │
│  + Agent         │                    │  容器内 CLI（无服务进程）：          │
│  + Skills        │                    │   • tracelens-* (离线分析)          │
│                  │                    │   • geak         (Ray 调度)         │
└──────────────────┘                    │   • oob run      (单任务子进程)     │
                                        │                                    │
                                        │  后台进程：                         │
                                        │   • sshd                    :22    │
                                        │   • Ray head + dashboard :6379/8265│
                                        │   • OOB auth-proxy (可选):4002     │
                                        │                                    │
                                        │  /opt/hyperloom/                   │
                                        │    ├── InferenceX/                 │
                                        │    └── .cursor/skills/             │
                                        │                                    │
                                        │  GPU: /dev/kfd + /dev/dri          │
                                        └────────────────────────────────────┘
                                                    │
                                                    ▼
                                          ┌──────────────────┐
                                          │ LLM API (出站)    │
                                          │ GEAK / OOB 后端   │
                                          └──────────────────┘
```

### 2.2 关键设计决策

**单一自包含容器**：所有优化工具、推理框架和基准测试脚本打包在一个容器中。
- 一条 `docker run` 命令即可部署
- 工具以直接子进程方式运行 — 无需服务发现、无 localhost RPC
- 简化 GPU 透传 — 单容器绑定 GPU 设备，避免多容器 GPU 共享的复杂性

**SSH 访问代替 Web UI**：用户通过 Cursor Remote SSH 连接。
- Cursor Agent 需要直接的文件系统访问来进行代码编辑和命令执行
- SSH 是 Cursor Remote 的原生协议 — 零额外适配
- 容器内 `/opt/hyperloom` 作为完整的 Cursor 工作区

**纯 CLI 工具，通过 Ray 调度**：TraceLens、GEAK 和 OOB 均由 Skill 以容器内 CLI 方式调用。
- Skill 直接执行 shell 命令：`tracelens-*`、`geak`（通过 `geak_ray_submit.py`）、`oob`（通过 `oob_ray_submit.py run`）
- GEAK 和 OOB 均使用本地 Ray 集群进行 GPU 分配（每 GPU 一个任务）
- OOB 的 `oob_ray_submit.py run` 将 `claude` / `codex` CLI 作为 Ray worker 子进程调度并阻塞至完成

### 2.3 BYOI（方式 B）架构

方式 B 的核心思路是：**将 Dockerfile 中的构建时安装，转化为 Agent 运行时的按需引导**。

```
用户笔记本/工作站                      用户 GPU 节点（任意镜像）
┌──────────────────┐                    ┌────────────────────────────────────┐
│                  │                    │  用户容器（自带推理框架 + ROCm）     │
│  Cursor IDE      │◄── SSH (22) ──────►│                                    │
│  + Agent         │                    │  Agent 首次连入时执行 bootstrap：   │
│  + Skills        │                    │   1. 探测已有组件                   │
│                  │                    │   2. 安装缺失依赖到 /opt/hyperloom  │
└──────────────────┘                    │   3. 启动 Ray + 配置环境变量        │
                                        │                                    │
                                        │  bootstrap 完成后，与方式 A 等价：  │
                                        │   • tracelens-* / geak / oob CLI   │
                                        │   • Ray head :6379                 │
                                        │   • /opt/hyperloom/ 工作区         │
                                        │                                    │
                                        │  GPU: /dev/kfd + /dev/dri          │
                                        └────────────────────────────────────┘
```

**关键原则**：
- **bootstrap 完成后，两种方式行为完全一致** — Skill 层无需区分方式 A/B
- **幂等**：`bootstrap.sh` 可重复执行，已安装的组件自动跳过
- **持久化**：所有组件安装到 `/opt/hyperloom`，容器重启不丢失（需配合卷挂载或持久化层）

---

## 3. 容器构建设计

> 本节仅适用于**方式 A（预构建镜像）**。方式 B 跳过本节，参见第 7 节。

### 3.1 多阶段构建流水线

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 0: hyperloom-src (FROM scratch)                           │
│                                                                 │
│ 将仓库产物打包为无运行时的镜像，下游 Dockerfile 无需本地构建      │
│ 上下文 — 只需 COPY --from。                                     │
│                                                                 │
│ 产物：OOB CLI / TraceLens / InferenceX / Skills / 脚本          │
└─────────────────────────────────────────────────────────────────┘
                              │
                    COPY --from=hyperloom-src
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
┌──────────────────────┐              ┌──────────────────────┐
│ Dockerfile.sglang     │              │ Dockerfile.vllm       │
│ BASE: sglang:v0.5.9   │              │ BASE: vllm:v0.17.0    │
│ FRAMEWORK=sglang      │              │ FRAMEWORK=vllm        │
└──────────────────────┘              └──────────────────────┘
```

### 3.2 分层策略

镜像层按**变更频率（低→高）**排列，以最大化构建缓存命中率：

| 层 | 内容 | 变更频率 |
|----|------|---------|
| Layer 1 | 系统依赖（openssh、Node.js 20、CA 证书） | 极低 |
| Layer 2 | GEAK + intellikit（git clone，固定分支） | 低 |
| Layer 3 | TraceLens + OOB Python/Node 依赖（`@anthropic-ai/claude-code`、`@openai/codex`） | 中 |
| Layer 4 | InferenceX、Skills、OOB 源码 + `pip install -e`（oob console script）、entrypoint | 高（轻量级，重建成本低） |

### 3.3 双框架支持

同一份 `hyperloom-src` 产物注入两个推理框架基础镜像（`sglang` / `vllm`）。两个 Dockerfile 结构完全相同 — 仅 `FROM` 镜像和 `FRAMEWORK` 环境变量不同。

---

## 4. 服务拓扑与生命周期

### 4.1 容器内进程

持久进程（Docker：`entrypoint.sh`；K8s：SSH 登录时 `/etc/profile.d/hyperloom.sh`）：

| 进程 | 端口 | 角色 |
|------|------|------|
| sshd | 22 | Cursor Remote SSH 入口 |
| Ray head + dashboard | 6379 / 8265 | GEAK GPU 任务调度 |
| OOB auth-proxy | 4002 | AMD LLM 网关的 Bearer 认证重写（仅当设置 `OOB_BASE_URL` 时） |

CLI 工具（无端口，由 Skill 按任务调用）：

| 工具 | 调用方式 |
|------|---------|
| TraceLens | `tracelens-*` console scripts（离线 trace 分析） |
| GEAK | `geak` CLI 通过 `geak_ray_submit.py` → Ray 调度 |
| OOB | `oob_ray_submit.py run -a {claude,codex} -p ... -f ...` — 通过 Ray 调度，生成 `claude` / `codex` 子进程并阻塞 |

### 4.2 生命周期

**Docker 模式**：`entrypoint.sh` 作为 PID 1 运行 — 配置 agent CLI 认证文件，启动 sshd / Ray /（可选）auth-proxy → 等待端口就绪（30 秒超时） → 进入 supervisor 循环（5 秒间隔，重启崩溃的 Ray 或 auth-proxy）。收到 SIGTERM/SIGINT 时，优雅地终止所有子进程。

**K8s 模式**：当 Pod CMD 被覆盖时，`/etc/profile.d/hyperloom.sh` 在 SSH 登录时运行（而非 PID 1）。它渲染 GEAK LiteLLM 配置，按需启动 Ray，当设置 `OOB_BASE_URL` 时启动本地 OOB auth-proxy 并将 `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` 重写为 `http://127.0.0.1:4002/...`。脚本通过检查 `ray status` 和 `:4002` 是否已在监听来避免重复启动后台服务。

---

## 5. 部署模型

### 5.1 Docker vs Kubernetes

| 维度 | Docker | Kubernetes |
|------|--------|------------|
| GPU 分配 | `--device=/dev/kfd --device=/dev/dri` | `amd.com/gpu` 设备插件 |
| 后台进程启动 | entrypoint.sh 直接管理 | autostart.sh 首次 SSH 登录时幂等启动 |
| 密钥管理 | `docker run -e` | K8s Secret |
| 网络暴露 | `-p 20022:22` 端口映射 | Service NodePort（仅端口 22） |
| 适用场景 | 单节点、快速验证 | 多用户、生产环境 |

参见 [README.md](README.md) 了解部署示例和完整参数参考。

---

## 6. 网络与安全

### 6.1 网络模型

```
外部 ──► SSH (:22) ──► 容器              （唯一入站点）
容器 ──► LLM API (GEAK/OOB 出站)        （唯一出站点）
内部：Ray (6379/8265) + 可选              （不对外暴露）
       OOB auth-proxy (4002)
无 MCP/REST 服务运行；工具均为纯 CLI。
```

| 端口 | 方向 | 对外暴露 |
|------|------|---------|
| 22 | 入站 | 是（必需） |
| 6379 / 8265 / 4002 | 内部 | 否 |

### 6.2 安全边界

- **数据留在用户网络内**：模型文件通过卷挂载；基准测试数据和优化结果存储在容器的本地文件系统
- **仅出站流量**：LLM API 调用（GEAK 内核优化、OOB `claude`/`codex` 子进程） — 可通过网络策略限制
- **无暴露的 RPC 接口**：不存在 MCP/REST 工具服务；仅 sshd 可从外部访问，Ray 和可选的 auth-proxy 绑定到 localhost
- **API Key 统一管理**：统一入口（`LLM_API_KEY` / `OOB_API_KEY`），启动脚本自动映射到各提供商的特定变量（`ANTHROPIC_API_KEY`、`OPENAI_API_KEY`、`AMD_LLM_API_KEY`），并在配置时通过本地 `:4002` 代理路由 OOB 流量

---

## 7. BYOI 模式设计（方式 B）

### 7.1 最小依赖要求

用户提供的镜像必须满足以下最小依赖：

| 组件 | 要求 | 说明 |
|------|------|------|
| Python | ≥ 3.10 | 基础运行时，无法自动安装 |
| GPU 驱动 | ROCm（`/dev/kfd` + `/dev/dri` 可用） | 内核级依赖，无法自动安装 |
| 推理框架 | sglang 或 vllm 已安装 | **必须自带，bootstrap 不负责安装** |
| 网络 | 可访问 PyPI + GitHub + npm registry | bootstrap 需要下载外部依赖 |
| WekaFS 挂载 | `$HYPERLOOM_BUNDLE` 路径已挂载且可读 | **必需** — OOB/TraceLens/InferenceX 等 Hyperloom 资源的来源 |

**可自动安装**（缺失时 bootstrap 会补装）：pip、git、curl、Node.js、Ray、GEAK、intellikit、TraceLens、OOB（含 npm CLI claude/codex）。

**必须由 Cursor 工作区提供**：Skills（用户用 Cursor 打开 `/opt/hyperloom` 时已自动加载）。

### 7.2 Bootstrap 流程

**前提**：用户已用自己的镜像启动容器，sshd 已运行，并通过 Cursor RemoteSSH 连入，Cursor 工作区为 `/opt/hyperloom`（其中已存在 `.cursor/skills/inference-optimization/` Skill 文件，**仅 Skills 而已**）。

**源码来源**（方式 B 与方式 A 的核心差异）：

| 组件 | 来源 | bootstrap 处理方式 |
|------|------|------------------|
| Skills | Cursor 工作区 `/opt/hyperloom/.cursor/skills/` | 已存在，不处理 |
| GEAK / intellikit | GitHub | git clone + `pip install -e` |
| Ray / click | PyPI | `pip install` |
| Node CLI（claude / codex） | npm | `npm install -g` |
| OOB / TraceLens | **WekaFS**（`$HYPERLOOM_BUNDLE/`） | `cp` 到 `/opt/hyperloom/` + `pip install -e` |
| InferenceX | **WekaFS** | 环境变量 `INFERENCEX_PATH` 直接指向（数据量大，不复制） |
| geak-litellm.yaml | **WekaFS** | 读取并渲染到 `/opt/hyperloom/geak-config/local.yaml` |

**WekaFS bundle 约定布局**（用户负责将 Hyperloom 资源放到 WekaFS 上一次，
所有 GPU 节点共享）：

```
$HYPERLOOM_BUNDLE/                # 默认 /wekafs/hyperloom-bundle/
├── OOB/                          # 完整 OOB 源码（含 oob_cli/ 子目录）
├── TraceLens-internal/           # TraceLens 源码
├── InferenceX/                   # InferenceX 数据 + 脚本
└── geak-litellm.yaml             # GEAK LiteLLM 模板
```

```
bootstrap.sh 执行流程
──────────────────────────────────────────────────────────

Step 1: 探测硬性依赖（缺失则报错退出，不尝试自动安装）
  ├── Python ≥ 3.10?
  ├── GPU? (rocm-smi)
  ├── 推理框架? (import sglang / vllm)
  └── WekaFS bundle 可访问? ($HYPERLOOM_BUNDLE/{OOB,TraceLens-internal,
                              InferenceX,geak-litellm.yaml} 全部存在)

Step 2: 软性系统依赖（缺什么装什么）
  ├── apt: pip, git, curl, gnupg, ca-certificates
  ├── Node.js 20 (nodesource，如缺失)
  ├── AMD CA 证书 (amd-root-ca.crt, amd-issuing-ca.crt，如未安装)
  └── mkdir -p /opt/hyperloom/geak-config /tmp/geak-data

Step 3: 安装外部依赖（GitHub + PyPI）
  ├── git clone GEAK → /opt/hyperloom/geak
  ├── git clone intellikit → /opt/hyperloom/intellikit (pinned SHA)
  ├── pip install -e /opt/hyperloom/geak
  ├── pip install -e /opt/hyperloom/intellikit/metrix/
  └── pip install ray "click<8.3"

Step 4: 从 WekaFS 复制并安装 Hyperloom 组件
  WekaFS 通常为只读共享盘，复制到 /opt/hyperloom 后再 pip install -e：
  ├── cp -r $HYPERLOOM_BUNDLE/TraceLens-internal → /opt/hyperloom/TraceLens
  │   └── pip install -e /opt/hyperloom/TraceLens
  ├── cp -r $HYPERLOOM_BUNDLE/OOB → /opt/hyperloom/OOB
  │   ├── pip install -r /opt/hyperloom/OOB/oob_cli/requirements.txt
  │   └── pip install -e /opt/hyperloom/OOB/oob_cli
  ├── certifi CA 证书注入（如安装了 AMD CA）
  └── npm install -g @anthropic-ai/claude-code @openai/codex@0.100.0

Step 5: 渲染配置 + Agent CLI 认证文件
  ├── 读取 $HYPERLOOM_BUNDLE/geak-litellm.yaml
  │   渲染到 /opt/hyperloom/geak-config/local.yaml (注入 model/key/url)
  ├── 写入 /root/.claude/config.json (如设置 OOB_API_KEY)
  └── 写入 /root/.codex/auth.json    (如设置 OOB_API_KEY)

Step 6: 启动后台服务 + 导出环境变量
  ├── ray start --head --num-gpus=$GPU_COUNT (如未运行)
  ├── OOB auth-proxy on :4002 (如设置 OOB_BASE_URL，复用 auth_proxy.py)
  ├── 写入 /etc/profile.d/hyperloom-env.sh
  │   ├── MODE=fully-local, FRAMEWORK=sglang/vllm
  │   ├── GEAK_CONFIG=/opt/hyperloom/geak-config/local.yaml
  │   ├── INFERENCEX_PATH=$HYPERLOOM_BUNDLE/InferenceX  ← 直接指向 WekaFS
  │   ├── SKILL_ROOT=/opt/hyperloom/.cursor/skills/inference-optimization
  │   ├── LLM_API_KEY → AMD_LLM_API_KEY 映射
  │   └── OOB_API_KEY → ANTHROPIC_API_KEY / OPENAI_API_KEY
  └── source 该文件以使当前 shell 立即生效
──────────────────────────────────────────────────────────
完成 → 工具链与方式 A 等价（geak/oob/tracelens 均可用，Ray 已就绪）
```

**与方式 A 的关键区别**：

| 方式 A 做的事 | 方式 B 处理方式 | 原因 |
|--------------|----------------|------|
| 安装 + 配置 sshd / 默认密码 | 不做 | 用户已通过 SSH 连入，sshd 由用户镜像负责 |
| `COPY --from=hyperloom-src /OOB`（构建时） | 改为运行时从 WekaFS `cp` | 构建时无源码，运行时从共享存储取 |
| `COPY --from=hyperloom-src /TraceLens`（构建时） | 改为运行时从 WekaFS `cp` | 同上 |
| `COPY --from=hyperloom-src /InferenceX`（构建时） | 通过 `INFERENCEX_PATH` 直接指向 WekaFS | 数据量大，无需复制 |
| `COPY --from=hyperloom-src /skills/...`（构建时） | 不做 | Skills 已在 Cursor 工作区中 |
| `COPY` entrypoint.sh / autostart.sh | 不做 | 方式 B 没有 PID 1 守护进程 |
| supervisor loop（5 秒重启崩溃服务） | 不做 | 服务崩溃由用户处理或重新跑 bootstrap |

### 7.3 幂等性设计

每个安装步骤都有**前置检查**，已完成的步骤自动跳过：

| 步骤 | 跳过条件 |
|------|---------|
| Step 1: WekaFS bundle | `$HYPERLOOM_BUNDLE/{OOB,TraceLens-internal,InferenceX,geak-litellm.yaml}` 全部存在 |
| Step 2: pip / git / curl | `command -v` 全部成功 |
| Step 2: Node.js 20 | `command -v node` 且版本 ≥ 20 |
| Step 2: AMD CA 证书 | `/usr/local/share/ca-certificates/amd-root-ca.crt` 存在 |
| Step 3: GEAK | `command -v geak` 成功 |
| Step 3: intellikit | `python -c "import metrix"` 成功 |
| Step 3: Ray | `command -v ray` 成功 |
| Step 4: TraceLens | `python -c "import TraceLens"` 成功 |
| Step 4: OOB CLI | `command -v oob` 成功 |
| Step 4: OOB Node CLI | `command -v claude` 且 `command -v codex` 成功 |
| Step 5: GEAK config | `/opt/hyperloom/geak-config/local.yaml` 存在 |
| Step 5: Claude auth | `/root/.claude/config.json` 存在 |
| Step 5: Codex auth | `/root/.codex/auth.json` 存在 |
| Step 6: Ray head | `ray status` 返回 0 |
| Step 6: OOB auth-proxy | `:4002` 端口已监听 |

这保证了：
- 第一次运行：完整安装（约 3-5 分钟，取决于网络）
- 后续运行：秒级跳过
- 部分失败后重新运行：只补装缺失部分

### 7.4 模式检测

Agent 在 Setup 阶段通过以下逻辑判断当前部署方式：

```bash
if [ -f /opt/entrypoint.sh ] && [ "${MODE:-}" = "local" ]; then
    # 方式 A：预构建镜像，entrypoint.sh 已完成所有初始化
    DEPLOY_METHOD="prebuilt"
else
    # 方式 B：BYOI，需要执行 bootstrap
    DEPLOY_METHOD="byoi"
    bash /opt/hyperloom/.cursor/skills/inference-optimization/scripts/bootstrap.sh
fi
# bootstrap 完成后，两种方式统一为 MODE=fully-local
```

### 7.5 方式 A vs 方式 B 对比

| 维度 | 方式 A（预构建） | 方式 B（BYOI） |
|------|-----------------|---------------|
| 镜像 | Hyperloom 官方镜像 | 用户任意镜像 |
| 首次启动耗时 | ~30 秒（服务启动） | 3-5 分钟（下载 + 安装） |
| 后续启动 | ~30 秒 | ~30 秒（幂等跳过） |
| 推理框架 | 镜像内置 | 用户自带 |
| 版本控制 | 随镜像 tag 锁定 | bootstrap 默认拉取最新，可通过环境变量锁定版本 |
| 离线部署 | 支持（镜像自包含） | 不支持（需联网安装） |
| entrypoint | Hyperloom `entrypoint.sh` | 用户自定义，Agent 调用 `bootstrap.sh` |

### 7.6 BYOI 环境变量

除方式 A 的所有环境变量外，BYOI 模式额外支持：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HYPERLOOM_BUNDLE` | `/wekafs/hyperloom-bundle` | **必需** — WekaFS 上的 Hyperloom 资源根目录（含 OOB/TraceLens-internal/InferenceX/geak-litellm.yaml） |
| `HYPERLOOM_ROOT` | `/opt/hyperloom` | 容器内 Hyperloom 组件安装根目录（OOB/TraceLens cp 目标） |
| `GEAK_REPO` | `https://github.com/AMD-AGI/GEAK.git` | GEAK 仓库地址 |
| `GEAK_BRANCH` | `main` | GEAK 仓库分支 |
| `INTELLIKIT_SHA` | `bcbfa0252df...` | intellikit 锁定的 commit SHA |
| `SKIP_BOOTSTRAP` | — | 设为 `1` 跳过 bootstrap（用户已手动安装） |

---

## 附录：文件清单

```
deploy/fully-local/
├── DESIGN.md                  # 本设计文档
├── README.md                  # 面向用户的快速入门指南
├── Dockerfile.hyperloom-src   # Stage 0: 仓库产物打包镜像（方式 A）
├── Dockerfile.sglang          # SGLang 基础镜像构建（方式 A）
├── Dockerfile.vllm            # vLLM 基础镜像构建（方式 A）
├── geak-litellm.yaml          # GEAK LiteLLM 模板（启动时渲染）
├── entrypoint.sh              # 容器入口：服务启动 + supervisor（方式 A）
└── hyperloom-autostart.sh     # K8s SSH 登录自动启动脚本

.cursor/skills/inference-optimization/
└── scripts/
    └── bootstrap.sh           # BYOI 环境引导脚本（方式 B）
```
