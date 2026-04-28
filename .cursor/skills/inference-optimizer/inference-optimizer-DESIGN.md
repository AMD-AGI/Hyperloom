# Inference Optimizer — Multi-Agent 自适应优化 Skill 设计方案

> **状态**: Final v0.4（统一代码 + Feature Flag 子集架构）  
> **作者**: 主 Agent + xiaofei 共同设计  
> **日期**: 2026-04-25  
> **占位 skill 名**: `inference-optimizer`  
> **目标读者**: 工程领导评审 + 落地实施 Agent  
> **v0.3 → v0.4 摘要**: 统一架构主线（marathon 全集 + quick/guided 子集）；吸收 sprint+marathon 漏补的 Iron Rules / 常量表 / Process Mgmt / Accuracy Gate / Initial Score Priors / 完整 20 action；KB 全 mode 启用；Watchdog 方案 A；删除 Roadmap 与 PoC

---

## 0. v0.1 → v0.2 变更日志

| 类别 | 变更 | 触发原因 |
|---|---|---|
| 阻塞修复 | 新增 Objective 抽象（§8）；§9/§10/§14 全部基于 Objective 重写 | 评审 #1：调度/早停只围绕 target_gain 写死，不支持多目标 |
| 阻塞修复 | §11 改为 OOB-only sub-agent；删除"Claw runSubagent HTTP 调用"假设 | 评审 #2：runSubagent 是 TS 内部函数，无 HTTP route |
| 阻塞修复 | §13 重写 + §13.5 新增：event log + 幂等键 + 任务状态机 | 评审 #3：原 checkpoint 是 best-effort，不是真可恢复 |
| 阻塞修复 | 新增 §10.5：结构化 Intent Transport（Claude tool_call / Codex JSON-only） | 评审 #4：自由文本解析 intent 太脆；Codex 角色不需要 tools |
| 阻塞修复 | §7 重写：早停 reason 分级 | 评审 #5：emergency 也跑 sweep 不合理 |
| 阻塞修复 | 新增 §3.5：资源锁模型（4 个 lane） | 评审 #6：共享 sandbox 缺锁 → 测量污染 |
| 拓扑 | §3 改为单 GPU sandbox 默认 + 多 sandbox TODO | xiaofei 决定：简单优先，扩展能力保留 |
| 高级 | §5.3 加 persona 蒸馏机制 | 评审：append-only 长跑会膨胀 |
| 高级 | §6/§17 明确 Brier 分期：P0 等权重 / P3 加权 | 评审：分期定义不一致 |
| 高级 | §16.5 加 PoC 退出标准 2 条（resume 幂等 / 重复 action 去重） | 评审：PoC 验收口径不全 |
| 高级 | §5.2 改：Critic 按运行时模式介入 KEEP/REVERT | 运行时分段后，quick mode 低开销优先；guided/marathon 默认 Critic |
| 高级 | §17 加：L4 第 2 次跑同模型族才 read | 评审：防 cold-start 单次坏经验污染 |
| 新增 ADR | ADR-13 ~ ADR-18，共 6 条 | 上述阻塞/高级修复对应的决策 |
| 新增 §25 | TODO 章节，明确标记延后/外部依赖项 | xiaofei 要求记 TODO 跟踪 |
| Open Q 收口 | §23 全部 7 个问题已答 | xiaofei + reviewer 答案合并 |

---

## 0.5 v0.2 → v0.3 交接版补充

| 类别 | 变更 | 触发原因 |
|---|---|---|
| 交接阻塞修复 | 明确 `MAX_HOURS` 必填；`TARGET_*` 只负责 Objective/早停 | 运行时 mode selection 不能依赖可选目标 |
| 交接阻塞修复 | 新增 §3.4 运行时三档：`<2h quick_param_sweep` / `2-6h guided_kernel_opt` / `>6h marathon_multi_agent` | xiaofei 原始分段设计需要显式落入方案 |
| 交接阻塞修复 | 新增 `PolicyGate`，所有 intent 先过 role/mode/action/state 校验 | 防止 Conductor 主循环绕过 mode/action gating |
| 交接阻塞修复 | A2A 明确 `event_log.jsonl` 是 source of truth，`asyncio.Queue` 仅内存缓存 | Resume 语义必须有唯一事实源 |
| 交接阻塞修复 | 任务状态机新增 `needs_manual_review` | evidence 不足的写副作用不能自动重放 |
| 交接收口 | Critic 改为按运行时模式介入：quick 默认不启，guided/marathon 默认介入 KEEP/REVERT | 避免短任务被审阅成本拖慢 |
| 交接收口 | 文档结构新增 `execution_mode.py` / `policy.py` / 对应测试 | 给实现者明确模块边界 |

---

## 0.6 v0.3 → v0.4 变更日志（按 xiaofei 5 轮反馈整理）

| 类别 | 变更 | 触发原因 |
|---|---|---|
| **架构主线** | 文档主线改为"统一代码库 + 三段 mode 是 Feature Flag 子集"；marathon 是全集，quick/guided 是 marathon 的子集，**共享同一套实现** | xiaofei #4：不要写两套实现，按 feature flag 决定启用哪些模块 |
| **删除 Roadmap + PoC** | 删除原 §18 PoC 整章 + 原 §19 Roadmap 整章；保留运行时条件（如 L4 warm-start）但不再用 Phase 表述 | xiaofei #5：先把架构和功能设计好，Roadmap 不着急 |
| **角色启用调整** | Critic 在 guided 也启用（轻量 review，防单 Agent 钻牛角尖） | guided 用 Critic 防上下文撑爆 + 防钻牛角尖 |
| **角色启用调整** | Watchdog guided **不常驻**，改为按需 ephemeral RCA（emergency 时让 Critic 兼任 no-tools RCA，写 `findings.jsonl`） | xiaofei 选择方案 A：保持"watchdog 是 multi-agent 标志" |
| **角色启用调整** | Sage 三段都启用，但 quick/guided 仅作 KB 查询服务（非常驻 reactor）；marathon 才作为常驻 reactor + Devil's advocate + strategic review | xiaofei：成本不高，所有 mode 都受益于 KB 召回 |
| **KB 全 mode 启用** | L4 KB read/write 在所有 mode 启用（沿用原 sprint 的 kb_query/kb_ingest 流程）；warm-start 条件保留（第 2+ 次同模型族才 read） | xiaofei：所有模式都有读写 KB 的权力 |
| **PolicyGate 修正** | quick mode Bash allowlist 拆分为"允许 server lifecycle / 禁 workspace_write"；server restart 是测参数/换 backend 的必备能力，不能禁 | xiaofei #1：肯定要允许服务重启，意义在于禁 workspace_write |
| **Iron Rules 强制** | 新增 §4.5 Iron Rules（IR-1 ~ IR-7），所有 mode 共同遵守 | xiaofei：强制；吸收 sprint+marathon 优点 |
| **新增硬资产** | §4.6 KERNEL_OPT 常量表 / §4.7 Process Management 规则 / §7.5 Accuracy Gate 协议 / §9 Initial Score Priors 表 | sprint+marathon 已有，v0.3 漏补 |
| **Action 体系完整化** | §12 补回完整 20 个 action（含 compiler-tuning / dream / re-explore / recover 等 marathon 专属）+ 每个 action 标 `allowed_modes` | marathon 漏补 |
| **可恢复语义收口** | (a) cursor 文件合并 `last_processed_seq + msg_id`，取消 `idempotency/` 目录，单文件原子写；(b) `dispatch_task` retry 前必须过 evidence-check，副作用 action 失败默认 `needs_manual_review`；(c) 明确 `tasks/<task_id>/state.json` 是 task lifecycle SoT | xiaofei #4#6#8 |
| **MessageBus 收口** | 明确 seq 由 MessageBus 内 `asyncio.Lock` 串行化 append | xiaofei #7 |
| **SubAgentRunner 接口** | 统一为 `run(task: DelegatedTask)`，删除悬空 `task_id`；按 `action.allowed_tools` 注入工具白名单 | xiaofei #5 |
| **Token 估算** | 拆三档分别给数（quick ≈0.5M / guided ≈3M / marathon ≈11.5M per 24h equivalent） | 按 mode 分 |
| **新增 ADR** | ADR-26 ~ ADR-32 共 7 条；ADR-9 标 `Superseded by ADR-14` | 文档卫生 |
| **新增 TODO** | T11 = "kernel-opt 只优化原生 kernel，不优化 torch.compile 后的 kernel（不好优化 + 损失精度风险大）" | xiaofei 新提 TODO |

---

## 1. TL;DR（一页汇报版）

把现有 `inference-optimization`（sprint，单 agent ≤3h）和 `marathon-inference-optimization`（长跑 24h+，tmux+claude CLI）合并成**一个**自适应、效果驱动、多 agent 协作的新 skill。

### 1.1 核心架构原则：**统一代码 + Feature Flag 子集**

> Marathon 是全集；quick 和 guided 是 marathon 的子集，**共享同一套实现**。  
> Conductor 启动时按 `MAX_HOURS` 设置一组 Feature Flag，关闭的功能在主循环里直接 skip，不创建对应 agent / 不注册对应 reactor。  
> 不写 "if mode == quick" 这种分支，全部靠 Feature Flag 配置 + action metadata 的 `allowed_modes` 驱动。

具体到三段 mode 的功能清单见 §3.4 Feature Flag 矩阵；一句话总结：

- **quick (<2h)** = 原 `inference-optimization` skill 去掉 `kernel-opt` / `integrate` + 跑在新框架上（享受 event log / lock / resume / accuracy gate / KB 等基础设施）
- **guided (2-6h)** = quick + 加 `kernel-opt` / `integrate` + 委托 sub-agent 并行 + Critic 轻量 review
- **marathon (>6h)** = guided + 真 Multi-Agent（常驻 Watchdog + 常驻 Sage + 议会/A2A + 完整 20 action + persona 蒸馏 + strategic review + cross-run synthesis）

### 1.2 核心创新 5 条

1. **统一入口 + 运行时分段**：用户必填 `MODEL_PATH + MAX_HOURS`，可选填 `TARGET_*`；skill 按 `MAX_HOURS` 自动选择三档运行时模式；同一套代码在不同模式下激活不同 Feature Flag。
2. **Objective 抽象 + Budget-Aware 调度器**：4 种目标（gain%/tput/baseline/time-only）走统一接口；progress_pressure 动态调度；达到目标立即早停。
3. **三段递进的 Agent 形态**：quick 单 Agent；guided 单 Agent + sub-agent + Critic；marathon 完整 Multi-Agent（Executor + Critic + Watchdog + Sage）。Watchdog/常驻 Sage 是 marathon 的标志特征；guided emergency 时用 ephemeral RCA（Critic 兼任）。
4. **4 层记忆模型 + 持久化语义**：L1 即时 / L2 session / L3 persona（marathon 才蒸馏） / L4 跨 run KB（**所有 mode 都读写**，warm-start 第 2+ 次同模型族才 read）；event log + 任务状态机保证 resume 不重不漏；副作用 action 失败默认 `needs_manual_review`。
5. **结构化 intent + 资源锁**：Claude 角色通过 `emit_intent` tool 发意图；Codex 角色不使用 tools，只输出 validated JSON intent；sandbox 内 4 个资源 lane 通过 durable file lease 实现互斥；PolicyGate 按 mode 校验所有 intent。

### 1.3 沿用 sprint + marathon 的硬资产

| 资产 | 来源 | 应用范围 |
|---|---|---|
| Iron Rules（IR-1 ~ IR-7） | sprint | 所有 mode 强制（§4.5） |
| KERNEL_OPT 常量表 | sprint | 所有 mode（§4.6） |
| Process Management 规则 | sprint+marathon | 所有 mode（§4.7） |
| Accuracy Gate 协议（GSM8K + 0.01 阈值） | sprint | 所有有 accuracy_risk 的 action（§7.5） |
| Initial Score Priors（model_class × action） | sprint | 调度器初始化（§9） |
| 20 个 action（11 浅 + 9 深） | sprint+marathon | 按 `allowed_modes` 启用（§12） |
| KB warm-up + ingest hook | sprint | 所有 mode（§6） |

### 1.4 载体

单 GPU sandbox（默认，简单优先）+ 共享 NFS（cross-restart 持久）。多 sandbox 拓扑作为扩展能力保留（见 §23 T1/T2）。

### 1.5 预期效果

- 用户体验：一个入口，3 行 env 起跑
- 效果驱动早停 → 用户实际等待时间显著下降
- 跨 run 记忆（KB 全 mode 启用） → 第 N 次跑同模型显著加快收敛
- guided 引入 Critic 防钻牛角尖；marathon 多模型协作决策可追溯、系统不脆

---

## 2. 背景与动机

### 2.1 现状两个 skill 的问题

| 问题 | sprint | marathon |
|---|---|---|
| 单 agent context 撑不住 24h | ✓ 是问题（≤3h 上限） | 用 tmux+claude CLI 解决 |
| 长跑没有效果驱动早停 | — | ✗ MAX_HOURS 是硬墙钟 |
| 协议被 skill 边界切开 | sprint 11 浅层 action | marathon 9 深层 action（不能用在短任务） |
| 单模型决策偏差 | ✗ Executor 自决 | ✗ 同 |
| KB 分裂 | sprint kb/ | marathon SPEC_ROOT/kb/，跨 run 知识不复用 |
| 长跑载体重 | — | tmux + npm install + base64 prompt + NFS file IPC |

### 2.2 核心痛点

> "用户要的是效果，不是时间。一个入口，能短能长，能并行能串行。同一个模型反复优化要越来越快。"

→ 4 个能力：自适应 / 效果驱动 / 真团队 / 长期记忆。

---

## 3. 整体架构

### 3.1 单 GPU sandbox 拓扑（marathon 全集形态）

下图是 **marathon 全功能启用** 时的拓扑。quick 和 guided 是这张图的子集（关闭部分 agent / 部分 sub-agent，但代码是同一份）。

```
                    用户 (Cursor / ClaudeCode CLI)
                              │ trigger skill (MODEL_PATH + MAX_HOURS + 可选 TARGET_*)
                              ↓
                          Claw Brain
                              │ create sandbox
                              ↓
╔═══════════════════ GPU Sandbox (单实例) ═══════════════════════════════════╗
║                                                                             ║
║  ┌─────────── Conductor (Python, 无 LLM, 永远存在) ───────────────┐         ║
║  │  主循环 / MessageBus / SharedState / ResourceLock /               │         ║
║  │  TaskRegistry / PolicyGate / Scheduler / Checkpoint /             │         ║
║  │  按 mode 启用对应 Feature Flag                                     │         ║
║  └────┬─────────────────────────────────────────────────────────────┘         ║
║       │                                                                       ║
║       │  ╔═══════ LLM Agent Pool (按 mode 启用对应 reactor) ═════╗           ║
║       ├──┤  Executor    Claude opus-4-7      —— 所有 mode        │           ║
║       │  │  Critic      Codex   gpt-5.4      —— guided/marathon  │           ║
║       │  │  Watchdog    Claude opus-4-7      —— marathon (常驻)  │           ║
║       │  │  Sage        Codex   gpt-5.4      —— marathon (常驻)  │           ║
║       │  │              (quick/guided 仅作 KB 查询服务，非 reactor)            ║
║       │  ╚═════════════════════════════════════════════════════════╝           ║
║       │                                                                       ║
║       │  ╔════ Ephemeral Sub-agent Pool (按 mode 启用 / 按需 spawn) ═╗      ║
║       └──┤  bench_runner    (benchmark_lane)                            │      ║
║          │  profile_runner  (profile_lane)                              │      ║
║          │  kernel_extract  (只读)                                      │      ║
║          │  geak_submitter  (外部 GEAK MCP)                             │      ║
║          │  patch_applier   (workspace_mutation + server_lifecycle)     │      ║
║          │  eval_runner     (benchmark_lane, 跑 GSM8K)                  │      ║
║          │  rca_runner      (按需 ephemeral RCA, guided emergency 才用) │      ║
║          │                                                               │      ║
║          │  实现：Conductor 直接 spawn OOB ClaudeBackend.run()             │      ║
║          │       / CodexBackend.run()，fresh context、跑完即销毁           │      ║
║          ╚═══════════════════╤═════════════════════════════════════════╝      ║
║                              ↓                                                ║
║                    推理 server (sglang/vllm)                                  ║
║                         (占 server_lifecycle lane)                            ║
╚══════════════════════════════╤════════════════════════════════════════════════╝
                               ↓
        ╔════════════════════════════════════════════════════════════╗
        ║   Shared NFS ($SESSION_DIR/)                                ║
        ║                                                             ║
        ║  - state.json                  (L2)                         ║
        ║  - event_log.jsonl             (L2, append-only)            ║
        ║  - cursors/<agent>.cursor      (含 last_processed_seq+id)   ║
        ║  - personas/<agent>.md         (L3, marathon 才蒸馏)        ║
        ║  - checkpoints/<ts>/           (resume point)               ║
        ║  - kb/entries.jsonl + insights.jsonl  (L4, 全 mode 读写)    ║
        ║  - results/<task_id>/          (sub-agent 输出)             ║
        ║  - tasks/<task_id>/state.json  (task SoT, lifecycle)        ║
        ║  - findings/<id>/              (RCA 结果, marathon+guided emergency) ║
        ║  - locks/<lane>.lock           (durable file lease)         ║
        ╚════════════════════════════════════════════════════════════╝
```

**核心设计点**：
- **同一份代码**：所有组件实现都在 `src/inference_optimizer/` 下；mode 只是 Feature Flag 配置
- **Conductor 永远存在**：Python、无 LLM；负责调度/锁/状态机/早停/checkpoint
- **LLM agent 按 mode 启用 reactor**：quick 只 Executor；guided 加 Critic；marathon 加 Watchdog 常驻 + Sage 常驻
- **Sage 在 quick/guided 仍可调**，但**不作为常驻 reactor**，只作为 KB 查询服务被 Executor 通过 Conductor 同步调用
- **跨 sandbox 重启的所有持久化数据落 NFS**
- **sub-agent 是 fresh OOB backend.run()**，不复用 Claw runSubagent
- **资源 lane 通过 Resource Lock Manager 协调**（§3.5）

### 3.2 多 sandbox 扩展方向（TODO，详见 §23）

未来可拓展到 CPU + GPU 分离：CPU sandbox 跑 Conductor + LLM agent + 思考型 sub-agent；GPU sandbox 跑推理 + benchmark。需要跟 sandbox 团队确认能力。

### 3.3 关键概念分离

| 概念 | 定义 |
|---|---|
| **Persistent Agent** | 长期"角色"（思考/决策/协商），通过 callable 多次唤醒；按 mode 启用对应 reactor |
| **KB Query Service** | Sage 在 quick/guided 的形态：非 reactor，由 Conductor 同步调用 `sage.recall(model, action)` 拿 KB 片段，结果注入下游 prompt |
| **Ephemeral Sub-agent** | 短期"动作"（具体执行），fresh OOB backend.run()，跑完销毁 |
| **Conductor** | 协议管理员（不是中央决策者），编排 message + 时钟 + 仲裁 + 早停 + 资源锁；按 mode 启用对应 Feature Flag |
| **Backend** | OOB 抽象，把 Claude/Codex 包成统一 `run(prompt, ...) → AgentResult` |
| **Resource Lane** | 互斥资源类别（server/workspace/benchmark/profile），sub-agent 必须先取 lease 才能动 |
| **Feature Flag** | mode → 一组开关；决定哪些 reactor 启动、哪些 action 允许、哪些定时任务跑、哪些资源初始化 |

### 3.4 运行时三段模式 — Feature Flag 矩阵（v0.4 主表）

> 这是 skill **运行时**的自适应分段。Conductor 在初始化时根据 `MAX_HOURS` 选择 `ExecutionMode`，再按下表设置 Feature Flag。所有 mode **共享同一套实现**，关闭的功能直接 skip 不创建对应组件。

#### 3.4.1 ExecutionMode 选择规则

```python
class ExecutionMode(Enum):
    QUICK_PARAM_SWEEP    = "quick_param_sweep"     # MAX_HOURS < 2
    GUIDED_KERNEL_OPT    = "guided_kernel_opt"     # 2 <= MAX_HOURS <= 6
    MARATHON_MULTI_AGENT = "marathon_multi_agent"  # MAX_HOURS > 6


def choose_execution_mode(env) -> ExecutionMode:
    max_hours = float(env["MAX_HOURS"])
    if max_hours < 2:
        return ExecutionMode.QUICK_PARAM_SWEEP
    if max_hours <= 6:
        return ExecutionMode.GUIDED_KERNEL_OPT
    return ExecutionMode.MARATHON_MULTI_AGENT
```

#### 3.4.2 Feature Flag 矩阵（共 34 项 feature）

##### 基础设施（所有 mode 都启用 = 共享代码的"地基"）

| # | Feature | quick | guided | marathon |
|---|---|:-:|:-:|:-:|
| F01 | Conductor 主循环 + MessageBus + SharedState | ✓ | ✓ | ✓ |
| F02 | ResourceLockManager（4 lane + durable file lease） | ✓ | ✓ | ✓ |
| F03 | TaskRegistry + DelegatedTask 状态机 | ✓ | ✓ | ✓ |
| F04 | PolicyGate（role / mode / action / state 校验） | ✓ | ✓ | ✓ |
| F05 | Event log + cursor + 幂等 + Checkpoint/Resume | ✓ | ✓ | ✓ |
| F06 | Intent Transport（emit_intent / validated_json） | ✓ | ✓ | ✓ |
| F07 | Accuracy Gate（GSM8K，§7.5） | ✓<sup>※1</sup> | ✓ | ✓ |

※1 quick mode 不跑 kernel-opt/integrate，accuracy_risk>0 的 action 仍然要过 gate（如 backends 切换）

##### LLM 角色启用

| # | Feature | quick | guided | marathon |
|---|---|:-:|:-:|:-:|
| F08 | Executor（Claude opus-4-7） | ✓ reactor | ✓ reactor | ✓ reactor |
| F09 | Critic（Codex gpt-5.4，no-tools） | ✗ | ✓ reactor（轻量 review） | ✓ reactor（完整 review + post-mortem） |
| F10 | Watchdog（Claude opus-4-7） | ✗ | ✗（emergency 时 ephemeral RCA via Critic） | ✓ 常驻 reactor |
| F11 | Sage（Codex gpt-5.4，no-tools） | ✓ KB 查询服务（非 reactor） | ✓ KB 查询服务（非 reactor） | ✓ 常驻 reactor + Devil's advocate |

##### Sub-agent 委托

| # | Feature | quick | guided | marathon |
|---|---|:-:|:-:|:-:|
| F12 | Sub-agent 委托能力（Executor 可 delegate） | ✗（Executor 直接执行） | ✓ | ✓ |
| F13 | 并行 sub-agent（受 lane 锁限流） | ✗ | ✓ | ✓ |

##### Action 范围（按 `action.allowed_modes` 自动 gating）

| # | Feature | quick | guided | marathon |
|---|---|:-:|:-:|:-:|
| F14 | 浅层 9 action：setup / classify / target-analysis / baseline / profile / backends / params / sweep / report | ✓ | ✓ | ✓ |
| F15 | kernel-opt + integrate（GEAK 调用 + Inductor patch） | ✗ | ✓ | ✓ |
| F16 | 深层 3 action：deep-kernel-analysis / operator-tuning / vendor-kernel-config | ✗ | ✗ | ✓ |
| F17 | 长跑 6 action：framework-rebuild / comm-optimization / compiler-tuning / dream / re-explore / recover | ✗ | ✗ | ✓ |

##### 协作模式（A2A）

| # | Feature | quick | guided | marathon |
|---|---|:-:|:-:|:-:|
| F18 | 委托模式（Executor → sub-agent → result） | ✗ | ✓ | ✓ |
| F19 | 流水线模式（多 sub-agent 串行编排） | ✗ | ✓ | ✓ |
| F20 | 议会模式（多 agent 投票决议） | ✗ | ✗ | ✓ |
| F21 | 事件驱动模式（Watchdog 监听 alert） | ✗ | ✗ | ✓ |

##### 4 层记忆 + KB（按 xiaofei #4：KB 全 mode 启用）

| # | Feature | quick | guided | marathon |
|---|---|:-:|:-:|:-:|
| F22 | L1 + L2 + L3（context / state / persona） | ✓ | ✓ | ✓ |
| F23 | Persona 自动蒸馏（每 4h / 8K token 触发） | ✗ | ✗ | ✓ |
| F24 | L4 KB **write**（每个 action 完成后 ingest，沿用 sprint kb_ingest.py） | ✓ | ✓ | ✓ |
| F25 | L4 KB **read**（warm-start，第 2+ 次同模型族） | ✓ | ✓ | ✓ |

##### 治理与早停

| # | Feature | quick | guided | marathon |
|---|---|:-:|:-:|:-:|
| F26 | 5 早停信号（target / time / leverage / brier<sup>※2</sup> / emergency） | ✓<sup>※3</sup> | ✓<sup>※3</sup> | ✓ |
| F27 | Strategic review（每 2h，Sage 主持） | ✗ | ✗ | ✓ |
| F28 | Cross-run synthesis（每 6h，Sage 写 insights） | ✗ | ✗ | ✓ |
| F29 | Brier 校准（critic 长期可信度加权） | ✗ | ✗ | 启用占位（数据成熟后开） |

※2 brier 信号需要 critic 历史预测数据，quick 没 critic 自动跳过  
※3 quick / guided 没常驻 watchdog → emergency 退化为 "ephemeral RCA via Critic"（guided）或"最小 crash report"（quick），见 §7.2

##### 沿用 sprint+marathon 硬资产

| # | Feature | quick | guided | marathon |
|---|---|:-:|:-:|:-:|
| F30 | Iron Rules（IR-1 ~ IR-7，§4.5） | ✓ 强制 | ✓ 强制 | ✓ 强制 |
| F31 | Initial Score Priors（model_class × action，§9） | ✓ | ✓ | ✓ |
| F32 | KERNEL_OPT 常量表（§4.6） | ✓（server 部分） | ✓ | ✓ |
| F33 | Process Management 规则（§4.7） | ✓ | ✓ | ✓ |
| F34 | Accuracy Gate 协议（§7.5） | ✓ | ✓ | ✓ |

#### 3.4.3 一句话总结

- **quick** = 原 sprint skill **去掉 kernel-opt/integrate** + 跑在新框架上
- **guided** = quick + 加 kernel-opt/integrate + 委托 sub-agent + Critic 轻量 review
- **marathon** = guided + 真 multi-agent + persona 蒸馏 + strategic review + cross-run synthesis

#### 3.4.4 Feature Flag 实现规范

```python
@dataclass
class FeatureFlags:
    """按 mode 决定的功能开关；Conductor 启动时根据 mode 实例化。"""

    enable_critic_reactor: bool        # F09
    enable_watchdog_reactor: bool      # F10
    enable_sage_reactor: bool          # F11 (常驻)
    enable_sage_query_service: bool    # F11 (KB 查询服务)
    enable_subagent_delegate: bool     # F12
    enable_persona_distill: bool       # F23
    enable_kb_read: bool               # F25 (warm-start 条件另判)
    enable_kb_write: bool              # F24
    enable_strategic_review: bool      # F27
    enable_cross_run_synthesis: bool   # F28
    enable_parliament: bool            # F20
    enable_event_driven_alert: bool    # F21


def build_feature_flags(mode: ExecutionMode) -> FeatureFlags:
    if mode == ExecutionMode.QUICK_PARAM_SWEEP:
        return FeatureFlags(
            enable_critic_reactor=False,
            enable_watchdog_reactor=False,
            enable_sage_reactor=False,
            enable_sage_query_service=True,
            enable_subagent_delegate=False,
            enable_persona_distill=False,
            enable_kb_read=True,
            enable_kb_write=True,
            enable_strategic_review=False,
            enable_cross_run_synthesis=False,
            enable_parliament=False,
            enable_event_driven_alert=False,
        )
    if mode == ExecutionMode.GUIDED_KERNEL_OPT:
        return FeatureFlags(
            enable_critic_reactor=True,
            enable_watchdog_reactor=False,
            enable_sage_reactor=False,
            enable_sage_query_service=True,
            enable_subagent_delegate=True,
            enable_persona_distill=False,
            enable_kb_read=True,
            enable_kb_write=True,
            enable_strategic_review=False,
            enable_cross_run_synthesis=False,
            enable_parliament=False,
            enable_event_driven_alert=False,
        )
    return FeatureFlags(  # MARATHON
        enable_critic_reactor=True,
        enable_watchdog_reactor=True,
        enable_sage_reactor=True,
        enable_sage_query_service=True,
        enable_subagent_delegate=True,
        enable_persona_distill=True,
        enable_kb_read=True,
        enable_kb_write=True,
        enable_strategic_review=True,
        enable_cross_run_synthesis=True,
        enable_parliament=True,
        enable_event_driven_alert=True,
    )
```

**Mode → action gating 是硬约束**：scheduler 不能通过高 pressure 绕过 mode 禁止项；例如 `<2h` 不跑 `kernel-opt`，也不允许 Executor delegate sub-agent，即使它的 score 更高。

---

## 3.5 资源锁模型

### 3.5.1 问题

所有 sub-agent 共享同一个 GPU sandbox 和 `/workspace`。如果不加锁：
- `bench_runner` 跑 benchmark 时另一个 sub `patch_applier` 改了 server 文件 → bench 结果污染
- 两个 `bench_runner` 同时跑 → GPU 被抢，吞吐数据失真
- `profile_runner` 在 profile 时 `bench_runner` 起 bench → profile trace 错乱

### 3.5.2 4 个资源 Lane

| Lane | 互斥粒度 | 持有者类型 | 典型 lease 时长 |
|---|---|---|---|
| `server_lifecycle` | 全局唯一持有者 | patch_applier / kernel-opt integrate / 重启 server 的 action | 30s ~ 10min |
| `workspace_mutation` | P0 全局独占；P1 可演进为多 reader/单 writer | 写 patch / 改 inductor cache / 改配置文件 | <30s |
| `benchmark_lane` | 全局唯一持有者 | bench_runner / sweep / eval_runner | 1 ~ 30min |
| `profile_lane` | 全局唯一持有者 | profile_runner | 1 ~ 5min |

### 3.5.3 跨 lane 互斥规则

```
benchmark_lane 持有时   → 禁止: patch_applier (server_lifecycle) 起新动作
                                profile_runner (profile_lane) 起
                          允许: 只读 sub-agent (kernel_extract 等)

profile_lane 持有时    → 禁止: bench / sweep / eval / patch
                          允许: 只读 sub-agent

server_lifecycle 持有时 → 禁止: bench / profile / eval (server 在重启不能用)
                          允许: 只读 sub-agent

workspace_mutation 持有 → 禁止: 任何 reader 读相关文件
                          (短锁，~30s)
```

### 3.5.4 实现：Durable Resource Lease（file lock / 可接入已有锁）

资源锁不能只用内存 `asyncio.Lock`。原因：
- Conductor 崩溃后，内存锁丢失，无法判断是否有未完成副作用任务。
- 未来如果 patch / bench / profile 被拆成独立进程，内存锁无法跨进程可见。
- 单 sandbox 里多个 worker 并发时，需要一个所有执行路径都能遵守的锁协议。

因此锁实现分两层：

1. `ResourceLockManager`：Conductor 内的统一仲裁入口，负责 action → lanes、权限检查、超时、死锁检测。
2. `ResourceLockBackend`：底层锁后端，默认用 durable file lease；如果仓库已有可靠锁代码，可替换接入。

```python
class ResourceLockBackend(Protocol):
    async def acquire_many(self, lanes: list[str], holder_id: str, ttl_sec: int) -> Lease:
        """Atomically acquire all lanes or acquire none."""
    async def heartbeat(self, lease: Lease) -> None:
        """Extend lease expiry while task is still alive."""
    async def release(self, lease: Lease) -> None:
        """Release only if holder_id still matches."""


class FileLeaseLockBackend:
    """P0 default: durable file lease under $SESSION_DIR/locks/."""
    def __init__(self, lock_dir):
        self.lock_dir = lock_dir

    async def acquire_many(self, lanes, holder_id, ttl_sec):
        # 1. Expand cross-lane conflicts into a canonical lane set.
        # 2. Sort lanes by fixed order to avoid deadlock.
        # 3. Create lease files with atomic create (O_CREAT|O_EXCL) or
        #    use repository-provided file-lock code if available.
        # 4. If any lane fails, release already acquired lanes and retry/backoff.
        # 5. If existing lease expired and holder heartbeat is stale, steal with
        #    recovery event recorded in event_log.
        ...
```

默认文件布局：

```
$SESSION_DIR/locks/
├── server_lifecycle.lock
├── workspace_mutation.lock
├── benchmark_lane.lock
└── profile_lane.lock
```

每个 lock file 记录：

```json
{
  "holder_id": "task-uuid",
  "action": "bench_runner",
  "lanes": ["benchmark_lane"],
  "acquired_at": "...",
  "expires_at": "...",
  "heartbeat_at": "...",
  "pid": 12345
}
```

### 3.5.5 原子 multi-lane lease 规则

- 所有 action 只能调用 `acquire_many(required_lanes)`，禁止嵌套获取单 lane。
- lane 获取顺序固定：`workspace_mutation` → `server_lifecycle` → `benchmark_lane` → `profile_lane`。
- 获取失败必须释放已获取 lane，并写入 `event_log`。
- lease 必须有 TTL 和 heartbeat；超时后不能静默抢锁，必须先写 `lease_expired` event。
- `workspace_mutation` P0 按全局独占锁处理；真正 reader/writer lock 延后，避免实现复杂度污染。
- 如果接入仓库已有锁实现，必须满足同样接口语义：atomic acquire-many、holder 校验、TTL/heartbeat、release 幂等。
- **acquire_many 失败策略**：非阻塞 + 指数退避（100ms → 1s → 5s），总等待上限 `action.lease_ttl × 2`；超过则任务转 `failed` 并把 action 回灌给调度器，不阻塞 reactor。

### 3.5.6 Conductor 调度时检查

调度器决定下一个 action 时，如果它需要的 lane 已被占，要么 wait，要么选另一个 action。这通过 action.metadata 里的 `requires_lanes` 字段声明。真正执行前仍必须调用 `acquire_many()`；调度检查只是优化，不能替代 lease。

### 3.5.7 File Lock 底层假设与替代实现

P0 默认 `FileLeaseLockBackend`，但它必须建立在明确的文件系统语义上。Design Gate 需要确认以下条件之一成立：

1. 目标 `$SESSION_DIR/locks` 所在文件系统支持跨进程原子 create（`O_CREAT|O_EXCL`）或等价的 atomic rename / link 语义。
2. 如果 NFS 语义无法保证上述原子性，则必须接入仓库/平台已有可靠锁实现，作为 `ResourceLockBackend` 的替代实现。

无论使用 file lock 还是已有锁代码，都必须满足同一语义契约：

- `acquire_many(lanes)` 要么拿到全部 lane，要么一个都不持有。
- lock holder 必须包含 `holder_id`、`task_id`、`lanes`、`expires_at`、`heartbeat_at`。
- stale lock 不能静默覆盖，必须写入 `lease_expired` / `lease_stolen` event。
- release 必须校验 holder_id，且重复 release 安全。
- Resume 时必须扫描 `$SESSION_DIR/locks`，把仍未过期的 lease 纳入 lock summary，防止恢复后立即污染 benchmark/profile。

---

## 4. 目标 / 非目标

### 4.1 目标 (核心)

- **G1**: 一个 skill 支持 `<2h` / `2-6h` / `>6h` 三档运行时执行模式，载体自动适配
- **G2**: 用户必须给 `MAX_HOURS`；`TARGET_*` 可选，用于 Objective 早停（基于 Objective 抽象）
- **G3**: 三段共享一套代码，按 Feature Flag 启用对应功能；marathon 是全集
- **G4**: 复用 OOB 的 `ClaudeBackend` / `CodexBackend` 抽象
- **G5**: ephemeral sub-agent 也用 OOB backend（**不依赖 Claw runSubagent**）
- **G6**: state.json + event_log.jsonl + personas/ + kb/ 完整 4 层记忆（L1-L4，KB 全 mode 启用）
- **G7**: Checkpoint + Resume + 幂等键 + 任务状态机 → 真可恢复语义
- **G8**: 结构化 Intent Transport（Claude tool_call + Codex JSON-only）+ 4 个资源 lane durable 互斥
- **G9**: 完整 20 个 action 接入；按 `allowed_modes` 自动 gating
- **G10**: Iron Rules + KERNEL_OPT 常量表 + Process Management + Accuracy Gate 全 mode 强制

### 4.2 目标 (扩展)

- **G11**: Brier 校准 → critic 长期可信度自动加权（数据成熟后启用）
- **G12**: 跨 run KB warm-start：第 2+ 次同模型族才 read

### 4.3 非目标

- **N1**: 不做 inference 引擎本身的修改
- **N2**: 不做 GEAK / Codex / Claude 内部 kernel 优化算法
- **N3**: 不做新的 LLM provider 集成（仅 Claude + Codex）
- **N4**: 不做前端 UI

---

## 4.5 Iron Rules（沿用 sprint，所有 mode 强制）

> 这些规则是 **跨 mode 硬约束**，违反任何一条 = 该次 run 无效。沿用原 `inference-optimization` skill 的 IR-1 ~ IR-7。

### IR-1: Submit ALL kernel candidates in parallel

`kernel-opt` action 必须把 `GEAK_TOP_CANDIDATES`（默认 5）个 candidate 同时提交给所有激活的 `KERNEL_OPT_BACKENDS`。**只交一个 candidate 或顺序交多个 backend = 违规**。

适用：guided / marathon（quick 不跑 kernel-opt，自动 N/A）。

### IR-2: NEVER modify kernel source before GEAK submission

提交的 kernel source **必须与 inductor cache 中提取的完全一致**。不允许：剥装饰器、改 strides、把 `@triton_heuristics` 替换成 `@triton.jit`、做任何"清理"编辑。GEAK 内部会处理 kernel 适配。

适用：guided / marathon（quick N/A）。

### IR-3: Integration is MANDATORY

GEAK 返回优化后的 kernel 后，必须执行 `integrate` action（patch → re-baseline → KEEP/REVERT 决策）。跳过 integrate = GEAK 结果未端到端验证 = 违规。`integrate` 必须用 `run_baseline.sh`（不是 `run_benchmark.sh`，后者不存在）。

适用：guided / marathon（quick N/A）。

### IR-4: Always kill_server + check_gpu_memory before server launch

每次启动 server 前必须先 kill 已存在的 server 进程，并验证 GPU 显存已释放。

适用：所有 mode 强制。这是 quick mode 也必须遵守的"server lifecycle 安全规则"。

### IR-5: Safe process management

**禁止使用 `pkill -f sglang`** —— 在 claw mode 下会 kill Ray worker。只允许：

```bash
kill $(pgrep -f 'python.*-m sglang.launch_server') 2>/dev/null
# 或 vLLM:
kill $(pgrep -f 'python.*-m vllm.entrypoints') 2>/dev/null
```

kill 与 relaunch 之间必须等 `SERVER_KILL_WAIT_S` 秒（默认 10s）。profiling 完后必须 `unset PROFILE SGLANG_TORCH_PROFILER_DIR`。

适用：所有 mode 强制。

### IR-6: Use `patch_inductor.py --target-file` for Inductor patching

必须用 `scripts/patch_inductor.py` + `--target-file`。`--cache-dir` 选项已移除。

**关键**：当 GEAK 改了 block size 或 warp count 时，必须同时传 `--best-config` 含更新的 tiling 参数；只 patch kernel `.py` 不更新 `.best_config` 会导致数值崩坏（输出乱码）。详见 `actions/integrate.md`。

适用：guided / marathon（quick N/A）。

### IR-7: NEVER modify GEAK configuration

GEAK 是外部服务 —— 视为**只读基础设施**。skill 不能修改 GEAK 任何配置文件、设置或参数（除了作为 `geak_create_task` 参数传入的）：

- **不**修改 GEAK server config / workspace 设置 / API 配置
- **不**写入或修改 GEAK config/settings 目录下的任何文件
- **不**在运行时改 `KERNEL_OPT_WORKSPACE` / `GEAK_STEP_LIMIT` 等常量
- **不**修改 GEAK MCP server 配置（`cursor_mcp_config.json` 等）
- **不**修改 GEAK 的测试数据 / 结果 / 配置文件

唯一允许的交互是通过 GEAK MCP tool call：`geak_get_model_config`（只读）/ `geak_create_task` / `geak_submit_task` / `geak_get_task` / `geak_get_outputs` / `geak_download_file` / `geak_list_tasks`。

**永远不要调 `geak_set_model_config` 改模型** —— LLM backend 由管理员预配置。

**例外 — tracing headers**：`kernel-opt` action 开始时，必须调一次 `geak_set_model_config` 注入 observability headers。先跑 `trace_action.py --component geak --action start` 记录时间并生成 config，再通过 MCP apply `extra_headers`。**不**修改 `model_class` / `model_name` / `api_base` / `api_key`。

适用：guided / marathon（quick N/A）。违反 = run 立即失效。

---

## 4.6 KERNEL_OPT 常量表（沿用 sprint，single source of truth）

所有 action 引用以下常量。**绝对不在运行时修改**。

| Constant | Value | 说明 |
|---|---|---|
| `KERNEL_OPT_BACKENDS` | `geak,codex` | 逗号分隔的激活 backend 列表，可被用户 override（任意组合：`geak` / `codex` / `claude` / `llm`） |
| `OOB_ROUND_ITERATIONS` | 3 | Codex/Claude round 数（submit → local benchmark → feedback → re-submit） |
| `KERNEL_OPT_IMAGE` | *(CI 或用户提供)* | kernel-opt 所有 backend 用的 framework image，每次 run 一个 |
| `KERNEL_OPT_WORKSPACE` | `control-plane-moe` | SaFE workspace（用户可 override） |
| `GEAK_STEP_LIMIT` | 100 | 每个 GEAK task 的最大 agent step |
| `GEAK_MAX_RETRIES` | 3 | 每个 kernel 的最大 submission 重试 |
| `GEAK_MAX_SUBMISSIONS` | 15 | 每次 run 的总 GEAK submission 预算 |
| `GEAK_TOP_CANDIDATES` | 5 | 提交的 top kernel candidate 数 |
| `GEAK_CONSECUTIVE_DISCARDS` | 5 | 连续这么多 discard 后停止 |
| `GEAK_WALL_CLOCK_MIN` | 120 | `kernel-opt` action 的最大 wall-clock minutes |
| `GEAK_POLL_INTERVAL_S` | 60 | GEAK task status 轮询间隔（秒） |
| `GEAK_POLL_TIMEOUT_MIN` | 15 | 单个 GEAK task 的最大轮询时间（分钟） |
| `MIN_GPU_PCT` | 3 | 作为 GEAK candidate 的最小 GPU 时间百分比 |
| `SERVER_KILL_WAIT_S` | 10 | server kill 与 relaunch 之间的等待秒数 |
| `FILTERED_TRACE_NAME` | `filtered-TP-0.trace.json.gz` | TraceLens 分析用的优选 trace 文件 |

**ALWAYS pass `KERNEL_OPT_IMAGE` to all kernel-opt backends**（包括 GEAK + OOB），无论 kernel 类型。对于源码在 image 里的 kernel（如 `/sgl-workspace/aiter/`），pod 用同一个 image。对于运行时生成的 kernel（如 `/tmp/torchinductor_root/` 来自 `torch.compile`），不在 prompt 里包含 `kernel_url` / `kernel_repo`；把文件复制到共享 NFS 或仅依赖 `files[].content`。

适用范围：guided 启用 kernel-opt 部分；marathon 全部启用；quick mode 仅引用 `SERVER_KILL_WAIT_S` 等 server 部分。

---

## 4.7 Process Management 规则（沿用 sprint+marathon）

- **永远 `export PATH="/opt/venv/bin:$PATH"`**：系统 python3 不带 sglang/vllm/numpy。每个 bash 命令必须先 prepend venv。失败模式：`ModuleNotFoundError: No module named 'sglang'`
- **永远不 `pkill -f "sglang.launch_server"` 在脚本内** —— 会 kill 脚本本身
- **永远等 `SERVER_KILL_WAIT_S` 秒**（默认 10）在 server kill 与 relaunch 之间
- **永远 `unset PROFILE SGLANG_TORCH_PROFILER_DIR`** 在 profiling 完成后
- **永远用 filtered trace** 给 TraceLens（raw 349MB / filtered 5MB）。TraceLens 不支持 `rocprofv3` 格式 —— 只支持 PyTorch Kineto
- **永远不 override 用户指定的 TP**：如果 prompt 说 TP=8 就用 TP=8，不要自动检测 GPU_COUNT 把它降到 TP=1（大模型 120B+ 单 GPU 跑不动）
- **vLLM flags 与 SGLang 不同**：常见错误 `--disable-log-requests` 不是 vLLM 有效 flag，用 `--disable-log-stats`
- **用 `run_baseline.sh` 而不是手动启 server**：脚本处理 server 启动 / health wait / benchmark / profiling 的测试过的序列；手动启会跳过 health check 撞 Exit 144（来自 stale 进程的 SIGTERM）

**适用范围**：所有 mode 强制。

---

## 5. Agent 角色与模型分配

### 5.1 Persistent Agent（marathon 全集 4 个 + Conductor，按 mode 启用）

| Agent | OOB Backend | 模型 | 主要职责 | LLM? | quick | guided | marathon |
|---|---|---|---|:-:|:-:|:-:|:-:|
| **Conductor** | — | Python | 主循环 + bus + state + 资源锁 + 议会主持 + 早停 + checkpoint + resume | ❌ | ✓ | ✓ | ✓ |
| **Executor** | `ClaudeBackend` | `claude-opus-4-7` | 提议 action、委托 sub-agent（quick 直接执行）、解读结果、写 prediction | ✓ | ✓ reactor | ✓ reactor | ✓ reactor |
| **Critic** | `CodexBackend` | `gpt-5.4` | review 提议、独立预测、本 run post-mortem；按 mode 介入；no-tools、JSON intent only；guided emergency 时兼任 ephemeral RCA | ✓ | ✗ | ✓ reactor（轻量 review） | ✓ reactor（完整 review） |
| **Watchdog** | `ClaudeBackend` | `claude-opus-4-7` | event_log RCA、crash 分析、健康监控（强工具使用） | ✓ | ✗ | ✗ | ✓ 常驻 reactor |
| **Sage** | `CodexBackend` | `gpt-5.4` | KB 主动维护、跨 run 召回、Devil's advocate、跨 run 教训沉淀；no-tools、JSON intent only | ✓ | ✓ KB 查询服务 | ✓ KB 查询服务 | ✓ 常驻 reactor + Devil's advocate |

#### 5.1.1 Tool Access 原则

Codex 角色（Critic / Sage）默认 **不使用 tools**。它们只消费 Conductor 放进 prompt 的证据、历史、diff、benchmark 摘要和 KB 片段，然后输出结构化 JSON intent。这样做有三个好处：

1. Critic / Sage 不直接读写 workspace，避免绕过资源锁和状态机。
2. 不依赖 Codex CLI 自定义 tool 能力，降低 Intent Transport 风险。
3. Codex 的角色边界更清晰：只做审阅、预测、归纳和反对意见，不执行副作用动作。

需要工具或 workspace 副作用的动作统一走 Claude-based Executor / Watchdog / ephemeral sub-agent，并由 Conductor 分配 action-scoped `allowed_tools` 和资源 lease。

#### 5.1.2 Sage 的两种形态（按 mode 切换）

| 形态 | 启用 mode | 实现 |
|---|---|---|
| **KB 查询服务**（非 reactor） | quick / guided | Conductor 在每个 action 执行前同步调用 `sage.recall(model, action) → str`；Sage backend 在该次调用内 fresh run，输出 markdown 片段，注入下游 Executor prompt；不参与议会、不维护 persona |
| **常驻 reactor + Devil's advocate** | marathon | Sage 作为持久 agent，订阅 `proposal` / `decision` topic，主持 strategic review（每 2h）+ cross-run synthesis（每 6h）+ 主动写 KB；可发 `objection` 触发议会 |

#### 5.1.3 Watchdog guided 缺席时的 emergency 处理

按 xiaofei 选择方案 A：guided 不常驻 Watchdog；emergency 触发时（crash_count >= 2），Conductor 启动 **ephemeral RCA task**：

```python
async def ephemeral_rca_via_critic(self) -> RCAFinding:
    """guided emergency 时，让 Critic 兼任 RCA。Critic 是 no-tools，
    所以传 event_log 摘要 + 当前 state + 最近 KEEP/REVERT 决策给它读，
    输出结构化 RCAFinding JSON。"""
    rca_prompt = self._compose_rca_prompt(
        event_log_tail=self.bus.tail(n=200),
        state_snapshot=self.state.summary(),
        recent_decisions=self.state.last_decisions(n=5),
    )
    result = await self.agents["critic"].backend.run(
        prompt=rca_prompt,
        system_prompt=self._load_rca_system_prompt(),
        max_turns=5,
        allowed_tools=[],  # no-tools
    )
    finding = parse_rca_finding(result.trajectory)
    write_finding(self.session_dir / "findings" / f"{ts}.json", finding)
    return finding
```

写入 `$SESSION_DIR/findings/<ts>.json`，包含进 final report。

### 5.2 Critic 介入策略（v0.4 调整：guided 启用）

| Mode | Critic 默认策略 |
|---|---|
| `quick_param_sweep` (`MAX_HOURS < 2`) | **不启用 Critic reactor**；emergency 时也不启动（quick 没有 KEEP/REVERT 副作用动作，crash 直接最小 crash report） |
| `guided_kernel_opt` (`2 <= MAX_HOURS <= 6`) | KEEP/REVERT 默认过 Critic（轻量 review）；emergency 时 Critic 兼任 ephemeral RCA |
| `marathon_multi_agent` (`MAX_HOURS > 6`) | KEEP/REVERT 默认过 Critic（完整 review）；常驻 Watchdog 处理 RCA；启用 strategic review/cross-run synthesis |

**采样降级**（仅当 token 成本告警时启用）：低风险 observation（accuracy_risk=0、单次 gain<2%、且 Brier 高 confidence）可采样 20% review。

理由：quick mode 是参数探索，单 Agent 直接执行，没有需要 review 的"副作用决策"；guided/marathon 涉及 kernel/server 副作用 + 委托 sub-agent，Critic 默认介入 KEEP/REVERT 防钻牛角尖。

### 5.3 Persona 蒸馏机制（仅 marathon 启用）

**问题**：L3 persona.md 如果 append-only，长跑会让 token 预算（§14）失真。quick/guided 时间窗口短，persona 不会膨胀到需要蒸馏；marathon 才需要。

**触发**：
- persona size > 8K tokens（硬触发）
- 每 4h（软触发）
- 每次 KEEP 决策后（机会触发）

**蒸馏过程**：
```python
async def distill_persona(agent_name):
    raw = read_persona(agent_name)
    if estimate_tokens(raw) < 4_000:
        return  # 不蒸馏

    # 让 agent 自己蒸馏（保留主观性）
    distilled = await self.agents[agent_name].backend.run(
        prompt=f"""Below is your accumulated persona/notes from this run.
Please rewrite it as a concise summary (max 2000 tokens) that preserves:
- Your stable opinions about this model class
- Lessons learned (what worked, what failed)
- Open questions you still have
- Stylistic markers that make you "you"

Drop: stale observations, redundant repetitions, ephemeral details.

ORIGINAL:
{raw}
""",
        ...
    )
    write_persona(agent_name, distilled.text)
    archive_old_persona(agent_name, raw)  # 留 backup 防误蒸
```

---

## 6. 4 层记忆模型

### 6.1 4 层定义（KB 全 mode 启用）

| 层 | 内容 | 载体 | 生命周期 | 维护者 | quick | guided | marathon |
|---|---|---|---|---|:-:|:-:|:-:|
| **L1 即时上下文** | 当前思考链 + 工具结果 | 单次 `backend.run()` 内 LLM context | run 内 | LLM 自然 | ✓ | ✓ | ✓ |
| **L2 本 session 工作记忆** | action_history / KEEP-REVERT / state / messages | `state.json` + `event_log.jsonl` (NFS) | session（含 resume） | Conductor | ✓ | ✓ | ✓ |
| **L3 本 run 人格** | "我是 X，对此 run 的累积观点" | `personas/<agent>.md` | run 持久；蒸馏仅 marathon | agent 自主 + 自动蒸馏（marathon） | ✓ Executor | ✓ Executor + Critic | ✓ 全部 + 蒸馏 |
| **L4 跨 run 长期记忆** | "这模型以前优化 N 次，结论 ..." | `kb/entries.jsonl` + `kb/insights.jsonl` + embeddings | 永久 | Sage 主动维护（marathon）；所有 mode 都 read/write | ✓ read+write | ✓ read+write | ✓ read+write+synthesis |

### 6.2 L4 KB 操作（沿用 sprint）

**Read（warm-start，所有 mode）**：每个 action 执行前查询 KB：
```bash
python3 $SKILL_ROOT/kb/kb_query.py "$MODEL_NAME $ACTION_NAME" --top-k 5 --compact
```

**warm-start 条件**：第 2+ 次同 model_family 才 read；第 1 次只 write 防 cold-start 单次坏经验污染。判定逻辑：`kb.count_entries(model_family) >= 1`。

**Write（所有 mode）**：每个 action 完成后 ingest：
```bash
python3 $SKILL_ROOT/kb/kb_ingest.py \
    --category $CATEGORY --model "$MODEL_NAME" \
    --action "$WHAT_WAS_DONE" --lesson "$KEY_TAKEAWAY" \
    --tags $TAGS --gain $GAIN --status $STATUS
```

**Sage 主动维护（仅 marathon）**：
- 每 6h cross-run synthesis：扫所有 entries，提炼 insights，写 `insights.jsonl`
- conflict detection：发现矛盾自动 flag 到 `kb/conflicts.jsonl`，等 Sage review

### 6.3 每次 `backend.run()` 的 prompt 拼接

```python
def compose_prompt(agent_name, msgs, state, objective, flags):
    return f"""
=== Your persona (L3, accumulated this run) ===
{kb.read_persona(agent_name)}

=== Cross-run knowledge (L4, warm-start 条件下 read) ===
{kb.recall_for_model(state.model_name, agent_name, top_k=5) if l4_warm_start_eligible(state) else "(L4 not yet warm for this model family)"}

=== Sage KB hint (quick/guided 也启用 Sage 查询服务) ===
{sage.recall(state.model_name, current_action) if flags.enable_sage_query_service and not flags.enable_sage_reactor else "(Sage not invoked)"}

=== Objective (统一抽象, §8) ===
type:           {objective.kind}              # gain_pct / tput / baseline / time_only
progress:       {objective.progress(state):.1%}
remaining_gap:  {objective.remaining_gap(state)}
pressure:       {scheduler.pressure(state):.2f}
time_left:      {state.time_left_min} min

=== Execution Mode (§3.4) ===
mode:           {state.execution_mode}
allowed_actions:{action_registry.allowed_for_mode(state.execution_mode)}
mode_rules:     {policy.mode_summary(state.execution_mode)}

=== Current session state (L2) ===
{state.summary()}

=== Active resource locks (§3.5) ===
{lock_mgr.summary()}

=== Messages for you to respond to (L2) ===
{format_messages(msgs)}

=== Your task ===
Respond through your configured Intent Transport (§10.5).
- Claude roles: call the `emit_intent` tool.
- Codex roles: return validated JSON only; no tools.
Free text is ignored.
"""
```

---

## 7. 早停机制

### 7.1 5 个早停信号（OR 关系）

```python
def should_stop_early(state, objective, flags) -> StopReason | None:
    # 1. 效果到达（核心）
    if objective.reached(state):
        return StopReason("target_reached", severity="success")

    # 2. 时间到
    if state.time_left_minutes <= 5:
        return StopReason("time_exhausted", severity="warning")

    # 3. 无优化空间
    if no_more_leverage(state):
        return StopReason("no_more_leverage", severity="warning")

    # 4. critic 长期不确定（边际收益低，仅 critic 启用时）
    if flags.enable_critic_reactor and critic_brier_plateau(state):
        return StopReason("brier_plateau", severity="info")

    # 5. 紧急
    if state.crash_count >= 2:
        return StopReason("emergency", severity="critical")

    return None
```

### 7.2 按 reason 分级的尾流（按 enable_watchdog / enable_sage_reactor 分支）

| Stop Reason | Severity | Sweep | Report | KB Synthesis | Final Checkpoint | RCA |
|---|---|---|---|---|---|---|
| `target_reached` | success | ✓ 完整 sweep | ✓ 完整报告 | ✓ if marathon (Sage 充分时间) | ✓ | — |
| `no_more_leverage` | warning | ✓ 完整 sweep | ✓ 完整报告 | ✓ if marathon (Sage ≤300s) | ✓ | — |
| `time_exhausted` | warning | ✗ 跳过 | ✓ Fast report (仅汇总) | ✓ if marathon (Sage ≤120s) | ✓ | — |
| `brier_plateau` | info | ✓ 完整 sweep | ✓ 完整报告 | ✓ if marathon (Sage ≤300s) | ✓ | — |
| `emergency` | critical | ✗ 禁止 | ✓ Crash report | ✗ 跳过 | ✓ 紧急 | marathon: 常驻 Watchdog 全力 RCA / guided: ephemeral RCA via Critic / quick: 最小 crash report |

```python
async def graceful_stop(self, reason: StopReason):
    self.state.set_stopping(reason)

    if reason.name == "emergency":
        # 不跑任何 benchmark 类动作，避免压垮已不稳定的环境
        await self.checkpoint(emergency=True)
        if self.flags.enable_watchdog_reactor:
            # marathon: 常驻 Watchdog
            await self.bus.send("watchdog", {"topic": "do_emergency_rca", "priority": 3})
            await self.bus.wait("watchdog", topic="rca_done", timeout=180)
        elif self.flags.enable_critic_reactor:
            # guided: Critic 兼任 ephemeral RCA
            finding = await self.ephemeral_rca_via_critic()
            self.state.attach_rca(finding)
        else:
            # quick: 仅最小 crash report（dump event_log tail + state）
            self.write_minimal_crash_report()
        await self.write_crash_report()
        return

    if reason.name == "time_exhausted":
        # 时间已到，只做最低限度
        await self.write_fast_report()
        if self.flags.enable_cross_run_synthesis:
            await self.bus.send("sage", {"topic": "synthesize_for_kb", "priority": 2})
            await self.bus.wait("sage", topic="synthesis_done", timeout=120)
        await self.checkpoint()
        return

    # success / warning / info：跑完整尾流
    await self.run_action("sweep")
    await self.run_action("report")
    if self.flags.enable_cross_run_synthesis:
        await self.bus.send("sage", {"topic": "synthesize_for_kb", "priority": 2})
        timeout = None if reason.name == "target_reached" else 300
        await self.bus.wait("sage", topic="synthesis_done", timeout=timeout)
    await self.checkpoint()
```

---

## 7.5 Accuracy Gate 协议（沿用 sprint，所有有 accuracy_risk 的 action 必跑）

> Baseline GSM8K accuracy 在 `baseline` action 中测量并存入 `state.baseline_accuracy`。每个后续 `accuracy_risk > 0` 的 action 必须在 KEEP 前过 accuracy gate。

### 7.5.1 哪些 action 触发 gate

| accuracy_risk | Actions | Gate required |
|:-:|---|:-:|
| 0.0 | server scheduling 参数（decode-steps / cuda-graph-max-bs / mem-fraction / chunked-prefill） | ❌ |
| 0.05–0.15 | kernel 修改（GEAK） / GEMM tuning | ✓ |
| 0.10 | backend 切换（aiter / alter / attention backends） | ✓ |
| 0.30 | 精度相关参数（kv-cache-dtype fp8 / 量化变更） | ✓ |

### 7.5.2 Gate 流程

对任何 `accuracy_risk > 0` 的 action，throughput benchmark 成功后：

1. **跑 GSM8K eval** 对运行中的 server 用 InferenceX 的 lm-evaluation-harness：
```bash
EVAL_TASK=gsm8k NUM_FEWSHOT=5 PORT=$PORT MODEL=$MODEL \
  RESULTS_DIR="$RESULT_DIR/eval_gsm8k_${ACTION_NAME}" \
  bash "$SKILL_ROOT/scripts/eval_accuracy.sh"
```

2. **提取 score** 从 eval summary：
```bash
new_accuracy=$(python3 -c "
import json, glob
f = sorted(glob.glob('$RESULT_DIR/eval_gsm8k_${ACTION_NAME}/eval_summary_gsm8k.json'))[-1]
d = json.load(open(f))
scores = list(d['scores'].values())[0]
print(scores.get('exact_match,strict-match', scores.get('exact_match,none', 0)))
")
```

3. **跟 baseline 比较**：
```
accuracy_drop = baseline_accuracy - new_accuracy
if accuracy_drop > accuracy_threshold (default 0.01 = 1 percentage point):
    REVERT immediately
    Log to KB: accuracy_risk=1.0 for this action+model
    Mark action as FAIL (accuracy degradation)
else:
    KEEP — accuracy within tolerance
```

### 7.5.3 Kernel-level 预检（可选，仅 GEAK / kernel 修改）

在完整 GSM8K eval 前，可以做一个快速 micro-benchmark sanity check 提前抓明显损坏：
```python
assert torch.allclose(original_output, optimized_output, atol=1e-3, rtol=1e-3)
```
这**不替代** GSM8K gate —— 只是 early-exit 优化。

### 7.5.4 跳过 gate 的 action

`setup` / `classify` / `profile` / `sweep` / `report` —— 这些是只读的，从不修改 serving computation path。纯 scheduling 参数（accuracy_risk=0.0）也跳过。

### 7.5.5 适用范围

所有 mode 强制。quick mode 不跑 kernel-opt，但若启用 `backends` 切换（accuracy_risk=0.1）仍需过 gate。

---

## 8. Objective 抽象

### 8.1 输入语义

`MAX_HOURS` 是必填输入，用于运行时分段（§3.4）和预算控制。`TARGET_*` 是可选输入，只用于 Objective/早停，不再承担 mode selection。

v0.1 调度器/早停/prompt 都直接依赖 `target_gain`，但现在用户可以给：
- `TARGET_GAIN_PCT=30`（基于 baseline 的相对增益）
- `TARGET_TPUT_PER_GPU=700`（绝对吞吐目标）
- `TARGET_DIR=/path/to/B200`（对标某个外部 baseline 目录）
- 都不给（仅 `MAX_HOURS=N`，时间驱动）

因此最终输入规则是：

- `MODEL_PATH` 必填。
- `MAX_HOURS` 必填，且必须 `>0`。
- `TARGET_GAIN_PCT` / `TARGET_TPUT_PER_GPU` / `TARGET_DIR` 可选，最多同时指定一个；不指定则进入 `TimeOnlyObjective`。

### 8.2 抽象接口

```python
class Objective(ABC):
    """统一目标接口，所有调度/早停/prompt 都通过此接口访问目标语义。"""

    @abstractmethod
    def kind(self) -> str:
        """gain_pct | tput | baseline | time_only"""

    @abstractmethod
    def progress(self, state: SharedState) -> float:
        """0.0 = 完全没动；1.0 = 已达成"""

    @abstractmethod
    def remaining_gap(self, state: SharedState) -> float:
        """剩余距离（百分比、绝对吞吐差、或 0 if time_only）"""

    @abstractmethod
    def reached(self, state: SharedState) -> bool:
        """是否已达成"""

    @abstractmethod
    def pressure_input(self, state: SharedState) -> float:
        """喂给 BudgetAwareScheduler.pressure() 的 progress_ratio
        time_only objective 应返回 0.0 (没目标，不施压)"""

    @abstractmethod
    def describe(self) -> str:
        """喂给 prompt 的人类可读描述"""
```

### 8.3 4 个具体实现

```python
class TargetGainObjective(Objective):
    def __init__(self, target_gain_pct: float):
        self.target = target_gain_pct

    def kind(self): return "gain_pct"
    def progress(self, state): return min(1.0, state.cumulative_gain / self.target)
    def remaining_gap(self, state): return max(0, self.target - state.cumulative_gain)
    def reached(self, state): return state.cumulative_gain >= self.target
    def pressure_input(self, state): return max(0, 1 - self.progress(state))
    def describe(self): return f"target: +{self.target}% gain over baseline"


class TargetTputObjective(Objective):
    def __init__(self, target_tput: float):
        self.target = target_tput

    def kind(self): return "tput"
    def progress(self, state):
        if state.baseline_tput is None: return 0.0
        gained = state.current_tput - state.baseline_tput
        needed = self.target - state.baseline_tput
        return min(1.0, gained / max(needed, 1e-6))
    def reached(self, state): return state.current_tput >= self.target
    def pressure_input(self, state): return max(0, 1 - self.progress(state))
    def describe(self): return f"target: {self.target} tok/s/GPU absolute"


class TargetBaselineObjective(Objective):
    """对标外部 baseline 目录（如 NVIDIA B200 数据）"""
    def __init__(self, target_dir: str):
        self.target_tput = parse_target_dir(target_dir)
        # 内部转化成 TargetTput
    # ... 同 TargetTput


class TimeOnlyObjective(Objective):
    """仅 MAX_HOURS，无效果目标"""
    def kind(self): return "time_only"
    def progress(self, state):
        return state.elapsed_minutes / state.max_minutes  # 时间进度
    def remaining_gap(self, state): return float('inf')
    def reached(self, state): return False  # 时间驱动永远不"reached"
    def pressure_input(self, state): return 0.0  # 不施加效果压力
    def describe(self): return f"time-only mode (no effect target), MAX_HOURS={state.max_hours}"
```

### 8.4 工厂

```python
def build_objective(env: dict) -> Objective:
    validate_required(env, ["MODEL_PATH", "MAX_HOURS"])
    validate_positive_float(env["MAX_HOURS"])
    validate_at_most_one(env, ["TARGET_GAIN_PCT", "TARGET_TPUT_PER_GPU", "TARGET_DIR"])

    if "TARGET_GAIN_PCT" in env:
        return TargetGainObjective(float(env["TARGET_GAIN_PCT"]))
    if "TARGET_TPUT_PER_GPU" in env:
        return TargetTputObjective(float(env["TARGET_TPUT_PER_GPU"]))
    if "TARGET_DIR" in env:
        return TargetBaselineObjective(env["TARGET_DIR"])
    return TimeOnlyObjective()
```

### 8.5 调度器/早停/prompt 全部基于此

```python
# 调度器
def pressure(self, state):
    pi = self.objective.pressure_input(state)
    ti = state.time_left / state.time_total
    return clip(pi / max(ti, 0.05), 0.5, 4.0) if pi > 0 else 1.0

# 早停
def should_stop_early(self, state):
    if self.objective.reached(state):
        return StopReason("target_reached", "success")
    # ...

# prompt
prompt += f"\n=== Objective ===\n{self.objective.describe()}\n"
prompt += f"progress: {self.objective.progress(state):.1%}\n"
```

---

## 9. Budget-Aware 调度器

### 9.1 评分公式（基于 Objective）

```python
score = base × pressure × mode_gate × depth_gate × diminishing × lane_available
```

| 因子 | 公式 | 作用 |
|---|---|---|
| `base` | `(expected_gain / cost_p75) × (1-acc_risk) × (1-crash_risk)` | 基础启发式 |
| `pressure` | 见 §8.5（基于 Objective.pressure_input） | 离目标远 + 时间少 → 倾向高 risk 高 gain |
| `mode_gate` | `1 if action.allowed_modes contains execution_mode else 0` | 运行时分段硬约束；`<2h` 禁止 `kernel-opt` |
| `depth_gate` | `1 if cost_p75 ≤ time_left × 0.8 else 0` | 防"半截饭" |
| `diminishing` | `0.7 ** count(completed, family=action.family)` | 防 DFS 一棵树吊死 |
| `lane_available` | `1 if all required lanes free else 0` | 资源锁过滤 |

### 9.2 Initial Score Priors（沿用 sprint，按 model class 设初值）

启动时按 `state.model_class` 给每个 action 设 base prior（KB warm-up 后再校准）：

| Action | Dense | MoE+MLA | MoE+SWA | MoE+MLA+NSA |
|---|:-:|:-:|:-:|:-:|
| backends | 3 | **9** | **8** | **10** |
| params | 5 | 6 | 7 | 5 |
| kernel-opt (GEAK) | **8** | 2 | 2 | 2 |
| torch.compile | **7** | 0 | 0 | 0 |
| sweep | 1 | 1 | 1 | 1 |

### 9.3 Score Update Rules（沿用 sprint）

每个 action 完成后：

1. **Action succeeded (gain > 0%)**: Boost similar actions. E.g., if `backends` gained +5%, boost remaining untested backends by 1.5×. Boost `combined_test` score.
2. **Action failed (gain ≤ 0%)**: Reduce similar actions by 0.5×.
3. **After 2+ backend wins**: Push `combined_backends_test` with score = sum(individual scores) × 1.5
4. **After all backends tested**: Push `re-profile`（discover new GEAK targets）
5. **After kernel opt kept**: Push `re-profile + next-kernel` with boosted score
6. **After kernel opt discarded**: Reduce remaining kernel scores by 0.7×
7. **When all action scores < 1.0**: Proceed to sweep → report

### 9.4 Brier 加权（数据成熟后启用）

```
默认: 所有 critic/agent 等权重
启用后: 每个 agent 维护历史 Brier score (prediction calibration)
        Brier 低（更准）的 agent 在议会投票里权重高
```

---

## 10. A2A 通信协议

### 10.1 Message Envelope（v0.4 加 seq 串行化语义）

`event_log.jsonl` 是 A2A Bus 的 source of truth；`asyncio.Queue` 只是运行时投递缓存。所有 message 必须先 append 到 event log 并获得全局递增 `seq`，再投递到内存 queue。Resume 时按 cursor 从 event log 重放，不能依赖 queue 状态。

**seq 分配语义**：MessageBus 内部用 `asyncio.Lock` 串行化 append + seq 分配：

```python
class MessageBus:
    def __init__(self, session_dir):
        self._append_lock = asyncio.Lock()
        self._next_seq = self._scan_max_seq(session_dir / "event_log.jsonl") + 1

    async def append_and_seq(self, msg: Message) -> int:
        """原子操作：分配 seq + append 到 event_log。"""
        async with self._append_lock:
            msg.seq = self._next_seq
            self._next_seq += 1
            line = json.dumps(asdict(msg))
            with open(self.event_log_path, "a") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            return msg.seq
```

```python
@dataclass
class Message:
    id: str               # uuid (idempotency key)
    from_agent: str
    to: str | list[str] | "*"
    topic: str            # 见白名单
    in_reply_to: str|None
    payload: dict
    priority: int         # 0=低 1=中 2=高 3=紧急
    timestamp: datetime
    seq: int              # 全局递增 sequence number, 用于 cursor 推进
```

### 10.2 Topic 白名单

`proposal` / `objection` / `question` / `answer` / `observation` / `event` / `decision` / `vote` / `vote_request` / `parliament_open` / `alert` / `historical_warning` / `reflection_tick` / `do_postmortem` / `do_strategic_review` / `do_emergency_rca` / `synthesize_for_kb` / `graceful_stop` / `heartbeat` / `delegated_result` / `intent_emitted` / `rca_done`

### 10.3 协议规则

1. 任何 message 必须有 `topic` + `priority` + `id`
2. `proposal/objection/decision` 必须带 `payload.reasoning`
3. `question` 60s 内必须有 `answer`
4. Rate limit：每 agent 10 msg/min（Sage 30）
5. `priority>=2` 立即触发，<2 batch
6. 接收方按 `id` 去重（幂等，通过 cursor 文件中的 `last_processed_msg_id`）

### 10.4 4 种协作模式

议会 / 流水线 / 事件驱动 / 委托 — 各 mode 启用范围按 §3.4 F18-F21。

---

## 10.5 结构化 Intent Transport

### 10.5.1 问题

让 Conductor 直接 `parse_intents(result.trajectory)` 解析自由文本太脆。Claude SDK 返回 message 对象流（带 ToolUseBlock / TextBlock），Codex CLI 返回 JSON event 流（type=item.completed/turn.completed），两种格式完全不同；agent 的"意图"埋在自由文本里 → 解析高度不可靠。

### 10.5.2 解决：统一 IntentEnvelope，按 backend 选择 transport

所有 agent 都必须输出同一个 `IntentEnvelope` schema，但不同 backend 使用不同 transport：

| Backend / 角色 | 是否使用 tools | Transport | 说明 |
|---|:-:|---|---|
| Claude Executor / Watchdog | ✓ | `tool_call: emit_intent` | 需要工具使用、RCA、委托动作编排 |
| Claude ephemeral sub-agent | ✓ | action-specific tools + `emit_intent` | 有副作用动作必须受资源锁和 allowed_tools 限制 |
| Codex Critic / Sage | ✗ | `validated_json_output` | no-tools；只审阅 Conductor 提供的证据并输出 JSON intent |

核心决策：**Codex 角色不需要 tools**。Critic / Sage 不直接读写 workspace、不跑命令、不调用 MCP；它们只消费 prompt 中的结构化 evidence、event log 摘要、benchmark 结果和 diff，然后输出 JSON。

### 10.5.3 统一 IntentEnvelope schema

```python
INTENT_ENVELOPE_SCHEMA = {
    "type": "object",
    "properties": {
        "intents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "intent_type": {
                        "type": "string",
                        "enum": [
                            "send_message", "delegate", "propose_action",
                            "objection", "vote", "update_state", "update_persona",
                            "answer", "ask_question", "alert"
                        ]
                    },
                    "payload": {
                        "type": "object",
                        "description": "Schema depends on intent_type."
                    }
                },
                "required": ["intent_type", "payload"],
                "additionalProperties": False
            }
        }
    },
    "required": ["intents"],
    "additionalProperties": False
}
```

### 10.5.4 Claude transport：`emit_intent` tool_call

```python
EMIT_INTENT_TOOL_SCHEMA = {
    "name": "emit_intent",
    "description": "The ONLY way to communicate decisions, messages, or actions to the system. Free-text responses are ignored.",
    "input_schema": {
        "type": "object",
        "properties": {
            "intent_type": {
                "type": "string",
                "enum": [
                    "send_message", "delegate", "propose_action",
                    "objection", "vote", "update_state", "update_persona",
                    "answer", "ask_question", "alert"
                ]
            },
            "payload": {
                "type": "object",
                "description": "Schema depends on intent_type, see system prompt for examples."
            }
        },
        "required": ["intent_type", "payload"]
    }
}
```

### 10.5.5 Codex transport：`validated_json_output`

Codex Critic / Sage 的 system prompt 强制：

```text
You have no tools. Do not ask to run commands, read files, or edit files.
Return exactly one JSON object matching INTENT_ENVELOPE_SCHEMA.
Do not include markdown, prose, code fences, or explanations outside JSON.
```

Conductor 对 Codex 输出执行：

1. 提取完整 JSON object。
2. 用 `INTENT_ENVELOPE_SCHEMA` 做 runtime validation。
3. 校验 `intent_type` 对应 payload 子 schema。
4. 校验角色权限：Critic / Sage 不允许 `delegate` 任何有副作用 action，不允许直接改核心 state。
5. 失败时最多发一次 repair prompt；仍失败则记录 `protocol_error`，交给 Watchdog（marathon）/ Critic 自审（guided）/ 直接 fail（quick）。

### 10.5.6 Conductor 解析侧

```python
def parse_intents(trajectory) -> list[Intent]:
    """按 backend transport 解析结构化 intent，忽略自由文本。"""
    intents = []
    for msg in trajectory:
        # Claude SDK: ToolUseBlock(name="emit_intent")
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if hasattr(block, "name") and block.name == "emit_intent":
                    intents.append(Intent.from_dict(block.input))
        # Codex CLI: final validated_json_output
        elif isinstance(msg, dict) and msg.get("type") == "item.completed":
            item = msg.get("item", {})
            if item.get("type") == "message" and item.get("role") == "assistant":
                envelope = parse_json_object(item.get("content", ""))
                validate(envelope, INTENT_ENVELOPE_SCHEMA)
                intents.extend(Intent.from_dict(x) for x in envelope["intents"])

    if not intents:
        raise NoIntentEmitted()

    return intents
```

### 10.5.7 Intent 权限边界

Agent 输出的是 intent，不是直接执行权。Conductor 必须按角色和状态机做二次校验：

- Codex Critic / Sage：允许 `objection` / `vote` / `answer` / `ask_question` / `alert` / `update_persona`；不允许发起有副作用的 `delegate`。
- Claude Executor：允许 `propose_action` / `delegate`，但必须经过 scheduler、resource lease 和 action allowed_tools 校验。
- 所有 agent：`update_state` 只能请求有限字段变更；核心字段（当前 best、KEEP/REVERT、task state、lock state）只能由 Conductor 的状态机 transition 写入。

### 10.5.8 Policy Gate（v0.4：quick mode 显式允许 server lifecycle）

所有 intent 在进入状态机或 task registry 前必须经过统一策略校验：

```python
class PolicyGate:
    def validate_intent(self, from_agent, intent, mode, action_registry, state):
        self._validate_role_permission(from_agent, intent)
        if intent.type == "delegate":
            action = action_registry.get(intent.task_kind)
            self._validate_mode_allowed(mode, action)
            self._validate_delegate_allowed(mode, from_agent, action)
            self._validate_side_effect_policy(action, state)
        if intent.type == "update_state":
            self._validate_state_transition(from_agent, intent.changes, state)

    def allowed_tools_for_agent(self, agent_name, mode):
        if agent_name in ["critic", "sage"]:
            return []  # Codex no-tools
        if mode == ExecutionMode.QUICK_PARAM_SWEEP:
            # quick: Executor 直接执行参数/后端探索 + server lifecycle，禁 workspace 写
            return ["Read", "Bash(quick_allowlist)", "emit_intent"]
        if agent_name == "executor":
            # guided/marathon: Executor 提议/委托/解读结果；写副作用走 sub-agent
            return ["Read", "emit_intent"]
        if agent_name == "watchdog":
            # Watchdog 可读日志、跑健康检查命令，但不直接写 workspace
            return ["Read", "Bash(health_allowlist)", "emit_intent"]
        return ["emit_intent"]
```

#### Quick mode Bash allowlist（v0.4 收口）

按 xiaofei #1 反馈：quick mode 必须允许 server restart（测参数/换 backend 必备），但禁 workspace_write。

**允许（quick_allowlist）**：
- server lifecycle 命令（kill / start / restart sglang / vllm；遵守 IR-4 / IR-5 / SERVER_KILL_WAIT_S）
- 只读检查：`ls` / `cat` / `head` / `tail` / `pgrep` / `ps` / `nvidia-smi` / `rocm-smi` / `df` / `du` / `which`
- benchmark / 参数 sweep 脚本：`bash $SKILL_ROOT/scripts/run_baseline.sh` / `bash $SKILL_ROOT/scripts/eval_accuracy.sh` / 等已注册脚本
- 运行时配置查询：`env` / `printenv`
- 推理 server 启动命令：`python -m sglang.launch_server ...` / `vllm serve ...`（带任意 flags 但 model 路径必须等于 `state.model_path`）
- Python 解释器只读模式：`python3 -c "..."` 仅做读取/计算，不写文件

**禁止（quick mode 硬规则）**：
- workspace 写：`Edit` 工具 / `rm` / `mv` / `cp` / `sed -i` / `awk -i inplace` / shell 重定向写文件（`>` / `>>` 写到 workspace 路径）
- git 写：`git commit` / `git add` / `git reset --hard` / `git push`
- patch 操作：`patch` / `git apply` / `scripts/patch_inductor.py`
- GEAK 调用：`geak_*` MCP（kernel-opt 类）
- kernel build：`pip install` / `python setup.py` / `make`

**意义**：quick mode 的承诺是"快速、低风险、可信回滚"。允许 server restart 是因为它**不留痕迹**（重启回到旧配置就行），但改代码会留痕迹，跟"<2h 拿到确定收益"目标矛盾。

#### 硬规则（PolicyGate 跨 mode 强制）

- `quick_param_sweep` 禁止任何 `delegate`，只允许 Executor 直接执行参数/后端探索动作。
- `quick_param_sweep` 禁止所有 `family=kernel_opt` 或 `side_effects` 包含 `workspace_write` 的 action。
- 持久 Executor / Watchdog 默认不拿 `Edit`；有副作用工具只通过 sub-agent 的 `action.allowed_tools` 注入。
- Codex Critic / Sage 永远不能 delegate 有副作用 action。
- `delegate` 必须满足 `action.allowed_modes` 包含当前 mode。
- 有副作用 action 必须声明 `requires_lanes`、`allowed_tools`、`side_effects`、`idempotency_key`。

---

## 11. Sub-agent 委托

### 11.1 决策：不依赖 Claw runSubagent

Conductor 直接 spawn OOB `ClaudeBackend.run()` / `CodexBackend.run()` 作为 sub-agent。理由：

- `runSubagent(opts)` 是 TS 内部函数，依赖 `parentSchemas/HandsClient/ToolRouter/onEvent` 等运行时对象，**没有 HTTP route 暴露**
- 从 Python 调用要么改 Brain（政治成本）要么用 TS 重写 Conductor（生态成本）

→ Conductor 直接 spawn OOB backend.run() 作为 sub-agent，工作量最小。

### 11.2 统一接口（v0.4 收口）

```python
@dataclass
class DelegatedTask:
    task_id: str            # uuid
    kind: str               # action.name (bench_runner / kernel_extract / patch_applier ...)
    params: dict            # action-specific
    idempotency_key: str    # 写入产物 / 外部请求 metadata 用
    requires_lanes: list[str]
    allowed_tools: list[str]
    side_effects: list[str]
    lease_ttl_sec: int
    state: str              # queued / running / succeeded / failed / cancelled / needs_manual_review
    attempts: int
    history: list[dict]


class SubAgentRunner:
    """Ephemeral sub-agent 统一入口。所有 action 都通过 run(task) 调用。"""

    def __init__(self, locks, workspace, action_registry, backends_pool):
        self.locks = locks
        self.workspace = workspace
        self.action_registry = action_registry
        self.backends_pool = backends_pool

    async def run(self, task: DelegatedTask) -> TaskResult:
        action = self.action_registry.get(task.kind)
        # 真正执行前必须按 PolicyGate 取交集校验 allowed_tools
        allowed_tools = self.policy.allowed_tools_for_action(self.mode, action)

        async with self.locks.acquire_many(
            lanes=action.requires_lanes,
            holder_id=task.task_id,
            ttl_sec=task.lease_ttl_sec,
        ) as lease:
            backend = self.backends_pool.pick(action.preferred_backend)
            try:
                task.transition("running", {"backend": backend.name, "lease": lease.id})
                result = await backend.run(
                    prompt=self._compose_prompt(task, action),
                    system_prompt=self._load_action_md(action.name),
                    cwd=self.workspace,
                    model=action.preferred_model,
                    max_turns=action.max_turns,
                    api_key=backend.api_key,
                    allowed_tools=allowed_tools,
                )
                parsed = self._parse_result(action, result.trajectory)
                task.transition("succeeded", parsed.evidence)
                return parsed
            except Exception as e:
                task.transition("failed", {"exception": str(e)})
                raise

    def _compose_prompt(self, task, action):
        # 注入 task.params + Iron Rules + KERNEL_OPT 常量 + KB hint
        ...

    def _parse_result(self, action, trajectory):
        # 按 action.result_schema 解析 + 校验
        ...
```

### 11.3 优势

| 维度 | 旧 (Claw runSubagent) | 新 (OOB-only) |
|---|---|---|
| 是否要改 Claw | 是（加 HTTP route 或政治成本） | **否** |
| Python 调用 | 不存在 | 直接 backend.run() |
| Sandbox 隔离 | Claw HandsClient | sub-agent 在 cwd 工作 + tempfile 隔离（OOB 已实现） |
| Keepalive | Claw 自动 5min | 不需要（短任务跑完即结束） |
| 事件流 | Claw event bubble | OOB trajectory（结构化 ToolUseBlock / JSON event） |
| 工作量 | 集成新 API ~1 周 | **沿用 OOB 现成代码** |

### 11.4 GPU 资源争抢（关键）

多个 sub-agent 在同一 GPU sandbox 同时跑 → 通过 §3.5 资源锁解决。

### 11.5 Codex sub-agent 限制

默认不使用 Codex 执行有工具或副作用的 ephemeral sub-agent。Codex 只用于 Critic / Sage 这类 no-tools 角色；如果未来需要 Codex sub-agent，必须先证明它的 transport 和 tool sandbox 能被 Conductor 约束。

---

## 12. Action 体系

### 12.1 完整 20 个 action（按 family 分组，按 `allowed_modes` 启用）

| Action | Family | Source | quick | guided | marathon | accuracy_risk |
|---|---|---|:-:|:-:|:-:|:-:|
| **setup** | prep | sprint | ✓ | ✓ | ✓ | 0.0 |
| **classify** | prep | sprint | ✓ | ✓ | ✓ | 0.0 |
| **target-analysis** | prep | sprint | ✓ | ✓ | ✓ | 0.0 |
| **baseline** | prep | sprint | ✓ | ✓ | ✓ | 0.0 |
| **profile** | analysis | sprint | ✓ | ✓ | ✓ | 0.0 |
| **backends** | shallow | sprint | ✓ | ✓ | ✓ | 0.10 |
| **params** | shallow | sprint | ✓ | ✓ | ✓ | 0.0 / 0.30<sup>※</sup> |
| **sweep** | shallow | sprint | ✓ | ✓ | ✓ | 0.0 |
| **report** | shallow | sprint | ✓ | ✓ | ✓ | 0.0 |
| **kernel-opt** | deep_kernel | sprint | ✗ | ✓ | ✓ | 0.05–0.15 |
| **integrate** | deep_kernel | sprint | ✗ | ✓ | ✓ | 0.15 |
| **deep-kernel-analysis** | deep_kernel | marathon | ✗ | ✗ | ✓ | 0.0 |
| **operator-tuning** | deep_kernel | marathon | ✗ | ✗ | ✓ | 0.10 |
| **vendor-kernel-config** | deep_kernel | sprint | ✗ | ✗ | ✓ | 0.10 |
| **framework-rebuild** | long | marathon | ✗ | ✗ | ✓ | 0.15 |
| **comm-optimization** | long | marathon | ✗ | ✗ | ✓ | 0.05 |
| **compiler-tuning** | long | marathon | ✗ | ✗ | ✓ | 0.05 |
| **dream** | creative | marathon | ✗ | ✗ | ✓ | 0.0 (生成假设) |
| **re-explore** | creative | marathon | ✗ | ✗ | ✓ | 0.0 (回溯重打分) |
| **recover** | resilience | marathon | ✗ | ✗ | ✓ | 0.0 (从 checkpoint 恢复) |

※ `params` 一般 0.0，但 `kv-cache-dtype fp8` / 量化变更属于 0.30

### 12.2 Action metadata 示例

```yaml
---
name: framework-rebuild
family: long
cost_minutes_p50: 60
cost_minutes_p75: 90
expected_gain_pct: 5-15
accuracy_risk: 0.15
crash_risk: 0.30
prerequisites: [profile, deep-kernel-analysis]
requires_lanes: [server_lifecycle, workspace_mutation]
allowed_tools: [Read, Bash, Edit]
side_effects: [workspace_write, server_restart]
allowed_modes: [marathon_multi_agent]
preferred_backend: claude
preferred_model: claude-opus-4-7
max_turns: 30
lease_ttl_sec: 5400
applicable_when:
  - kernel_dispatch_shows_aiter_dominance
  - cumulative_gain_plateau
---
```

调度器 `mode_gate` 因子读 `allowed_modes`，`lane_available` 因子读 `requires_lanes`。Sub-agent 启动时，Conductor 还必须按 `allowed_tools` 注入工具白名单；只读 action 不给 `Edit` / 写文件能力，有副作用 action 必须先拿到 durable lease。

### 12.3 dream / re-explore / recover 三个特殊 action（marathon 专属）

- **dream**: 让 Sage 生成"如果 X 是真的，会发生什么"假设；用于跳出 DFS 局部最优。无副作用，无 lane 需求。
- **re-explore**: 把已 discard 的 candidate 重新打分（基于新 KB 信息或新 baseline）；DFS 回溯。无副作用。
- **recover**: 从 checkpoint 恢复一个之前 crash 的子任务；走 §13 Resume 流程。

---

## 13. Checkpoint + Resume + Idempotency

### 13.1 旧设计的问题（v0.1）

asyncio.Queue 内存态 + "重放" 在以下点会出问题：
- 消息已发出未 ACK → 重启重发 → 接收方重复处理
- sub-agent 已执行未入账 → 重启重跑 → 重复改文件
- KEEP 已应用未 checkpoint → 重启回退到 propose 阶段 → 重复决策

### 13.2 v0.4 新设计：Event Log + Cursor 合一 + State Machine

**核心数据结构**（v0.4 收口：取消独立 idempotency 目录，合并到 cursor 文件）：

```
$SESSION_DIR/
├── state.json                       # 当前 snapshot（30min/KEEP 后写）
├── event_log.jsonl                  # append-only durable event log（A2A SoT）
├── cursors/
│   ├── executor.cursor              # {last_processed_seq, last_processed_msg_id, processed_at}
│   ├── critic.cursor                # 单文件原子写
│   ├── sage.cursor
│   └── watchdog.cursor
├── tasks/
│   └── <task_id>/state.json         # 委托任务状态机（task lifecycle SoT）
├── personas/<agent>.md
├── checkpoints/<ts>/                # 完整快照
├── findings/<ts>.json               # RCA 结果（marathon 常驻 watchdog / guided emergency ephemeral）
└── locks/<lane>.lock                # durable file lease
```

**SoT 划分**（v0.4 明确）：
- `event_log.jsonl` 是 A2A 消息 / 通知的 SoT
- `tasks/<task_id>/state.json` 是 task lifecycle 的 SoT
- 两者发生不一致时，**以 task state 为准**；event_log 中的 `delegated_result` 仅作为可重发通知

### 13.3 Message 处理：cursor 合并 idempotency（v0.4 简化）

```python
@dataclass
class CursorState:
    agent: str
    last_processed_seq: int
    last_processed_msg_id: str
    processed_at: str

async def process_message(self, agent_name, msg):
    cursor = self.bus.load_cursor(agent_name)

    # 幂等检查：seq 必须 > last_processed_seq
    if msg.seq <= cursor.last_processed_seq:
        return  # 已处理过，跳过

    # 处理
    await self._reactor_handle(agent_name, msg)

    # 单文件原子写（tmp + rename）：last_seq + last_id 一起更新
    new_cursor = CursorState(
        agent=agent_name,
        last_processed_seq=msg.seq,
        last_processed_msg_id=msg.id,
        processed_at=now_iso(),
    )
    atomic_write_json(self.bus.cursor_path(agent_name), asdict(new_cursor))
```

**好处**：
- 单文件 tmp+rename 原子写，trivial
- Resume 时按 `last_processed_seq + 1` 从 event_log 重放，天然不重不漏
- 重复检测只看 `seq <= last_processed_seq`，不需要单独的 marker 目录
- 失去"哪些 msg 处理过"的细粒度记录但不需要（seq 单调）

### 13.4 委托任务状态机（v0.4 加 evidence-check 优先 retry）

```python
TASK_STATES = [
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "needs_manual_review",  # evidence 不足或写副作用不安全重放时进入
]

class DelegatedTask:
    def __init__(self, task_id, kind, params):
        self.task_id = task_id
        self.kind = kind
        self.params = params
        self.state = "queued"
        self.attempts = 0
        self.history = []

    def transition(self, new_state, evidence):
        # 持久化到 tasks/<task_id>/state.json
        self.state = new_state
        self.history.append({"state": new_state, "ts": now_iso(), "evidence": evidence})
        atomic_write_json(f"tasks/{self.task_id}/state.json", asdict(self))


async def dispatch_task(task: DelegatedTask, action):
    """v0.4: retry 前必须按 evidence-check 决定。"""
    task.transition("running", {"sub_agent_pid": ...})
    try:
        result = await sub_agent_runner.run(task)
        task.transition("succeeded", result.evidence)
        return result
    except Exception as e:
        task.transition("failed", {"exception": str(e)})

        # v0.4 evidence-check 优先 retry
        if not action.side_effects:
            # 纯只读 action：可以自动重试
            if task.attempts < MAX_RETRY:
                task.attempts += 1
                return await dispatch_task(task, action)
            raise
        else:
            # 副作用 action：检查 evidence
            ev = evidence_check(task, action)
            if ev == "succeeded_recovered":
                task.transition("succeeded", {"recovered": True, **ev})
                return load_result_from_evidence(task)
            else:
                # evidence 不足 → needs_manual_review，不自动重跑
                task.transition("needs_manual_review", {
                    "reason": "side_effect_evidence_insufficient_for_safe_replay",
                    "evidence": ev,
                })
                raise NeedsManualReviewError(task.task_id)
```

### 13.5 Resume 流程（v0.4：先读 task state，再重放 event）

```python
async def resume_from_checkpoint(checkpoint_dir):
    # 1. 加载 state snapshot
    state = SharedState.load(f"{checkpoint_dir}/state.json")

    # 2. 加载所有 cursor（含 last_processed_seq + msg_id）
    cursors = load_cursors(f"{checkpoint_dir}/../cursors/")

    # 3. 加载 personas
    personas = load_personas(f"{checkpoint_dir}/../personas/")

    # 4. 检查所有 in-flight 任务（task state 是 SoT）
    for task_id in glob("tasks/*/state.json"):
        task = DelegatedTask.load(task_id)
        if task.state == "running":
            # 崩溃时正在跑的任务 → 按 §13.6 evidence-check 矩阵判定
            ev = evidence_check_matrix(task)
            if ev.verdict == "succeeded":
                task.transition("succeeded", {"recovered": True, **ev.details})
            elif ev.verdict == "safely_failed":
                task.transition("failed", {"reason": "crashed_mid_run, no side effects"})
                self.scheduler.requeue_action(task.params, attempts=task.attempts+1)
            else:
                task.transition("needs_manual_review", {
                    "reason": "evidence_insufficient_for_safe_replay",
                    "evidence": ev.details,
                })

    # 5. 重放 event_log 中 cursor 之后的事件给每个 agent
    for agent in self.agents:
        cursor = cursors[agent.name]
        for event in event_log.read_from(cursor.last_processed_seq + 1):
            # idempotency 已经在 process_message 内部按 cursor 检查
            await self.bus.send(agent.name, event.payload)

    return ResumeState(state=state, cursors=cursors, personas=personas)
```

### 13.6 副作用 Action 崩溃点恢复矩阵

所有有副作用 action 必须写入 `tasks/<task_id>/state.json`，并在 `history[]` 中记录 evidence。Resume 时不根据"上次跑到哪一行"猜测，而是按可验证 evidence 判定状态。

| Action 类型 | 副作用 | 崩溃点 | Resume 判定 | 恢复动作 |
|---|---|---|---|---|
| `bench_runner` / `eval_runner` | 生成结果文件，不改 workspace | benchmark 已跑完，`succeeded` 未写 | 检查 `results/<task_id>/metrics.json` + checksum + event_log 是否有 start event | 补写 `succeeded(recovered=true)`，重新发送 `delegated_result` |
| `profile_runner` | 生成 trace 文件，不改 workspace | trace 部分写入 | 检查 trace 完整性 marker；无 marker 视为未完成 | 标记 failed，清理 partial trace，允许重跑 |
| `patch_applier` / `integrate` | 改 workspace / server 配置 | patch 已应用，task state 未 commit | 检查 patch idempotency marker、git diff fingerprint、server config fingerprint | 若 fingerprint 匹配则补写 succeeded；否则进入 `needs_manual_review`，不自动二次 apply |
| `server_restart` | 重启推理 server | server 停在 unknown 状态 | 检查 pid/health endpoint/port owner + lease holder | Watchdog RCA（marathon）/ Critic ephemeral RCA（guided）；Conductor 只在确认无活跃 benchmark/profile 后重启 |
| `kernel_extract` / read-only | 只读输出 artifact | 输出缺失或 partial | 检查 artifact checksum | 缺失则安全重跑 |
| `geak_submitter` / 外部提交 | 外部系统可能已有请求 | 提交成功但本地未记录 | 使用外部 request id / idempotency key 查询 | 若外部已接受则补写 succeeded；否则按 retry policy 重试 |

规则：

- 每个有副作用 action 必须带 `task_id` / `idempotency_key`，并把该 key 写入产物或外部请求 metadata。
- `patch_applier` 这类 workspace 写操作必须先拿 `workspace_mutation` lease，并在 apply 前后记录 fingerprint。
- 任何 evidence 不足以证明 succeeded 的写操作，**不允许自动重放**，必须转 `needs_manual_review` 或让 Watchdog 出 RCA。
- `needs_manual_review` 是 terminal blocking state：Conductor 不自动 retry、不继续 apply 同类写操作；最终报告必须列出该 task 和人工处理建议。
- `delegated_result` 是可重发消息；真正幂等边界在 task state + evidence，而不是消息是否发过。

### 13.7 触发条件

- 每 30min 自动
- 每次 KEEP 决策后立即
- 每次 strategic review 后（仅 marathon）
- 收到 graceful_stop 信号
- crash 紧急

---

## 14. Hybrid 执行模式

```
事件驱动 (95%):
  inbox 空 → 静默不调 LLM
  有消息 → triage:
    priority>=2 → 立即调对应 agent.backend.run()
    priority=1  → batch (30s flush)
    priority=0  → batch (2min flush)

时钟驱动 (5%):
  每 5min   每 agent 主动 reflection
  每 30min  Critic mini post-mortem (guided/marathon)
  每 30min  Conductor 自动 checkpoint
  每 4h     Persona 自动蒸馏 (仅 marathon)
  每 2h     Sage strategic review (仅 marathon)
  每 6h     Sage cross-run synthesis (仅 marathon)
```

### Token 估算（按 mode 分档，v0.4 新增）

| Mode | Reactors | 估算 / 24h equivalent | 实际 wall-clock token |
|---|---|---|---|
| **quick** (<2h) | Executor 单 reactor + Sage 偶发查询 | 0.5M | ≈ 0.04M / 2h |
| **guided** (2-6h) | Executor + Critic + sub-agent + Sage 查询 | 3M | ≈ 0.75M / 6h |
| **marathon** (>6h) | Executor + Critic + Watchdog + Sage 常驻 + sub-agent + 蒸馏 + strategic review | 11.5M | 11.5M / 24h |

vs marathon skill 旧实现 50-100M / 24h，节省 4-9 倍。

---

## 15. Conductor 主循环骨架

```python
from oob.backends import ClaudeBackend, CodexBackend  # 复用 OOB

class Conductor:
    def __init__(self, session_dir, env):
        self.objective = build_objective(env)            # §8
        self.mode = choose_execution_mode(env)           # §3.4
        self.flags = build_feature_flags(self.mode)      # §3.4.4
        self.session_dir = session_dir

        # LLM agent pool (按 flags 启用对应 reactor)
        self.agents = {
            "executor": AgentRole("executor", ClaudeBackend(), "claude-opus-4-7"),
        }
        if self.flags.enable_critic_reactor:
            self.agents["critic"] = AgentRole("critic", CodexBackend(), "gpt-5.4")
        if self.flags.enable_watchdog_reactor:
            self.agents["watchdog"] = AgentRole("watchdog", ClaudeBackend(), "claude-opus-4-7")
        if self.flags.enable_sage_reactor:
            self.agents["sage"] = AgentRole("sage", CodexBackend(), "gpt-5.4")

        # 基础设施（所有 mode）
        self.bus       = MessageBus(session_dir)
        self.state     = SharedState(session_dir)
        self.scheduler = BudgetAwareScheduler(self.objective, self.mode, env)
        self.kb        = KnowledgeBase(session_dir)
        self.locks     = ResourceLockManager(FileLeaseLockBackend(session_dir / "locks"))
        self.tasks     = TaskRegistry(session_dir)
        self.policy    = PolicyGate(self.flags)
        self.actions   = ActionRegistry("actions/")
        self.sub       = SubAgentRunner(self.locks, self.workspace, self.actions, self.policy)

        # Sage KB 查询服务（quick / guided 用，marathon 由 reactor 提供同接口）
        self.sage_query = SageQueryService(CodexBackend(), self.kb) \
            if self.flags.enable_sage_query_service and not self.flags.enable_sage_reactor \
            else None

    async def run(self):
        # Resume 检查
        if self.session_dir.has_checkpoint():
            resume_state = await self.resume_from_checkpoint()
            self._apply_resume_state(resume_state)
        else:
            await self._init_session()

        await asyncio.gather(
            *(self._reactor(name) for name in self.agents),
            self._clock(),
            self._stopping_watcher(),
        )

        await self._graceful_stop(self.state.stop_reason)  # §7.2

    async def _reactor(self, agent_name):
        agent = self.agents[agent_name]
        while not self.state.should_stop():
            msgs = await self.bus.recv(agent_name, timeout=60)
            if not msgs:
                continue

            cursor = self.bus.load_cursor(agent_name)
            msgs = [m for m in msgs if m.seq > cursor.last_processed_seq]
            if not msgs:
                continue

            # 同步注入 Sage KB hint（quick/guided 走 query service）
            sage_hint = ""
            if self.sage_query and agent_name == "executor":
                sage_hint = await self.sage_query.recall(self.state.model_name, self.state.current_action)

            prompt = self._compose_prompt(agent_name, msgs, sage_hint=sage_hint)

            try:
                allowed_tools = self.policy.allowed_tools_for_agent(agent_name, self.mode)
                result = await agent.backend.run(
                    prompt=prompt,
                    system_prompt=agent.system_prompt,
                    cwd=self.state.cwd,
                    model=agent.model,
                    max_turns=10,
                    api_key=agent.api_key,
                    allowed_tools=allowed_tools,
                )
            except NoIntentEmitted:
                # agent 没通过对应 Intent Transport 输出，重试一次
                continue

            intents = parse_intents(result.trajectory)

            for intent in intents:
                await self._handle_intent(agent_name, intent)

            # 推进 cursor (单文件原子写, §13.3)
            for msg in msgs:
                self.bus.advance_cursor(agent_name, msg.seq, msg.id)

    async def _handle_intent(self, from_agent, intent):
        self.policy.validate_intent(from_agent, intent, self.mode, self.actions, self.state)

        if intent.type == "send_message":
            await self.bus.send(intent.to, intent.message)
        elif intent.type == "delegate":
            action = self.actions.get(intent.task_kind)
            task = self.tasks.create(action, intent.params)
            asyncio.create_task(self._dispatch_task_and_reply(task, action, from_agent))
        elif intent.type == "update_state":
            self.state.apply_validated_transition(from_agent, intent.changes)
        elif intent.type == "update_persona":
            self.kb.append_persona(from_agent, intent.note)
        elif intent.type == "propose_action":
            if self.flags.enable_parliament:
                await self.bus.broadcast({"topic": "proposal", **intent.payload})
            else:
                # quick mode 不 broadcast；guided 也不开议会，直接当作 Executor 自己的提议
                await self._record_proposal_for_self_review(from_agent, intent)
        elif intent.type == "objection":
            if self.flags.enable_parliament:
                await self._open_parliament(intent)
        elif intent.type == "vote":
            if self.flags.enable_parliament:
                await self._record_vote(from_agent, intent)

    async def _dispatch_task_and_reply(self, task, action, requester):
        try:
            result = await dispatch_task(task, action)  # §13.4 含 evidence-check
            await self.bus.send(requester, {
                "topic": "delegated_result",
                "payload": {"task_id": task.task_id, "result": result},
            })
        except NeedsManualReviewError as e:
            await self.bus.send(requester, {
                "topic": "delegated_result",
                "payload": {"task_id": task.task_id, "error": "needs_manual_review", "task_id_for_human": e.task_id},
                "priority": 3,
            })
        except Exception as e:
            await self.bus.send(requester, {
                "topic": "delegated_result",
                "payload": {"task_id": task.task_id, "error": str(e)},
            })
```

---

## 16. 用户接口

### 16.1 极简入口

```bash
@inference-optimizer

MODEL_PATH=/hyperloom/models/DeepSeek-R1-0528
MODEL_NAME=deepseek-ai/DeepSeek-R1-0528
TP=8 GPU_TYPE=MI355X FRAMEWORK=sglang

# MAX_HOURS 必填：用于运行时分段 + 预算控制
MAX_HOURS=24

# TARGET_* 可选：用于效果早停；最多同时指定一个
TARGET_GAIN_PCT=30                   # 或 TARGET_TPUT_PER_GPU=700
                                     # 或 TARGET_DIR=/path/to/B200_baseline
```

`MAX_HOURS` 同时决定运行时执行模式：

- `MAX_HOURS < 2`：`quick_param_sweep`，单 Agent 测参数/换 backend，不跑 `kernel-opt`，允许 server restart 但禁 workspace_write。
- `2 <= MAX_HOURS <= 6`：`guided_kernel_opt`，Agent + sub-agent，可跑 `kernel-opt`，启用 Critic 轻量 review。
- `MAX_HOURS > 6`：`marathon_multi_agent`，启用 multi-agent 长跑能力（Watchdog 常驻 + Sage 常驻 + 议会）。

### 16.2 三档 mode 入口示例

#### Quick（<2h）

```bash
@inference-optimizer

MODEL_PATH=/hyperloom/models/Qwen3-8B
MODEL_NAME=Qwen/Qwen3-8B
TP=8 GPU_TYPE=MI355X FRAMEWORK=sglang
MAX_HOURS=1.5
TARGET_GAIN_PCT=10
```

预期：单 Agent 试 backends + params，不跑 kernel-opt，目标 +10% 或 1.5h 早停。

#### Guided（2-6h）

```bash
@inference-optimizer

MODEL_PATH=/hyperloom/models/DeepSeek-R1-0528
MODEL_NAME=deepseek-ai/DeepSeek-R1-0528
TP=8 GPU_TYPE=MI355X FRAMEWORK=sglang
MAX_HOURS=4
TARGET_GAIN_PCT=20
```

预期：Executor 委托 sub-agent 跑 profile + kernel-opt + integrate，Critic 轻量 review KEEP/REVERT。

#### Marathon（>6h）

```bash
@inference-optimizer

MODEL_PATH=/hyperloom/models/DeepSeek-R1-0528
MODEL_NAME=deepseek-ai/DeepSeek-R1-0528
TP=8 GPU_TYPE=MI355X FRAMEWORK=sglang
MAX_HOURS=24
TARGET_GAIN_PCT=30
ENABLE_SAGE=1                # 默认启用 Sage 常驻 reactor
```

预期：完整 Multi-Agent，议会 + 跨 run KB synthesis + persona 蒸馏 + strategic review。

### 16.3 Marathon 兼容入口

保留一个 release 作为 thin shim，转发到新入口；下一个 release 删除。不长期维护两套心智。

---

## 17. 文件 / 目录结构

```
.cursor/skills/inference-optimizer/   # 入口与规则
├── SKILL.md
├── README.md
└── KNOWLEDGE-BASE.md

src/inference_optimizer/              # 实现独立目录（统一代码库）
├── orchestrator/
│   ├── conductor.py
│   ├── agent_role.py
│   ├── message_bus.py                # §10.1 含 seq 串行化
│   ├── shared_state.py
│   ├── scheduler.py
│   ├── score_priors.py               # §9.2 Initial Score Priors 表
│   ├── objective.py                  # §8
│   ├── execution_mode.py             # §3.4 ExecutionMode + FeatureFlags
│   ├── feature_flags.py              # §3.4.4 build_feature_flags
│   ├── policy.py                     # §10.5.8 PolicyGate (含 quick allowlist)
│   ├── intent_parser.py              # §10.5
│   ├── resource_lock.py              # §3.5
│   ├── task_registry.py              # §13.4
│   ├── sub_agent_runner.py           # §11
│   ├── persona.py                    # §5.3 蒸馏（仅 marathon）
│   ├── sage_query_service.py         # §5.1.2 KB 查询服务（quick/guided 用）
│   ├── kb.py                         # §6 全 mode 启用
│   ├── checkpoint.py                 # §13
│   ├── iron_rules.py                 # §4.5 跨 mode 强制
│   ├── kernel_opt_constants.py       # §4.6 single source of truth
│   ├── process_management.py         # §4.7 安全规则
│   ├── accuracy_gate.py              # §7.5 GSM8K + 0.01 阈值
│   └── system_prompts/
│       ├── executor.md
│       ├── critic.md
│       ├── sage.md
│       ├── watchdog.md
│       └── rca_critic.md             # guided emergency 时 Critic 兼任 RCA 用
├── actions/                          # 20 个合并 sprint+marathon
│   ├── setup.md / classify.md / target-analysis.md / baseline.md / profile.md
│   ├── backends.md / params.md / sweep.md / report.md
│   ├── kernel-opt.md / integrate.md
│   ├── deep-kernel-analysis.md / operator-tuning.md / vendor-kernel-config.md
│   ├── framework-rebuild.md / comm-optimization.md / compiler-tuning.md
│   ├── dream.md / re-explore.md / recover.md
│   └── _meta/                         # 每个 action 的 yaml metadata
├── kb/
│   ├── entries.jsonl
│   ├── insights.jsonl
│   ├── kb_query.py                    # 沿用 sprint
│   └── kb_ingest.py                   # 沿用 sprint
├── scripts/
│   ├── run_baseline.sh                # 沿用 sprint
│   ├── eval_accuracy.sh               # GSM8K eval
│   └── patch_inductor.py              # 沿用 sprint
└── tests/
    ├── test_objective.py
    ├── test_execution_mode.py
    ├── test_feature_flags.py
    ├── test_policy.py
    ├── test_scheduler.py
    ├── test_intent_parser.py
    ├── test_resource_lock.py
    ├── test_idempotency.py
    ├── test_resume.py
    ├── test_iron_rules.py
    ├── test_accuracy_gate.py
    └── e2e/
```

---

## 18. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| OOB Backend 在 GPU sandbox 内跑不通 | 低 | 高 | 实施第一件事就验证 |
| Codex JSON-only 输出不稳定 | 中 | 中 | Codex 角色 no-tools；使用 `validated_json_output` + runtime validate + 一次 repair；仍失败交 Watchdog (marathon) / Critic 自审 (guided) |
| critic 误判反而拖慢决策 | 低 | 中 | Brier 加权（数据成熟后）；可手动调整 critic 介入采样率；guided mode 默认轻量 review |
| cost_minutes 估算不准 | 中 | 中 | P75 + KB 历史校准 |
| pressure 过激进 → 频繁 crash | 中 | 中 | crash_count ≥ 2 紧急停 + crash_risk 因子压制 |
| 4 agent token 暴涨 | 低 | 中 | 事件驱动 + 弱模型分担，估 11.5M/24h marathon |
| KB 合并冲突 | 低 | 低 | conflicts.jsonl + Sage review |
| Sandbox 重启丢工作 | 中 | 高 | event_log + cursor 合一 + 状态机 (§13) |
| 多用户共用 KB 串数据 | 高 | 高 | KB 按 user_id + model 分区 |
| 资源锁死锁 | 中 | 高 | durable file lease + 原子 multi-lane lease + timeout + 调度器全局检测；死锁触发紧急 release + alert |
| 任务状态机崩在 transition 中间 | 低 | 高 | §13.6 evidence-check 矩阵；needs_manual_review 兜底 |
| Sage KB 查询服务（quick/guided）时延 | 中 | 低 | 异步预取 + 超时 30s 后空值 fallback |
| Watchdog guided 缺席 → emergency RCA 质量低 | 低 | 中 | Critic ephemeral RCA + event_log tail 摘要；marathon 才有完整 RCA |

---

## 19. Open Questions（v0.4 全部收口）

| # | 问题 | 决策 |
|---|---|---|
| 1 | Skill 名 | `inference-optimizer` |
| 2 | Conductor 落地位置 | 单 GPU sandbox（默认）；多 sandbox 扩展能力作为 TODO（§23） |
| 3 | Critic 介入策略 | quick 不启用；guided KEEP/REVERT 默认过 Critic（轻量）；marathon 完整 review；token 告警时降级采样 |
| 4 | marathon 兼容入口 | 保留 1 release thin shim，下 release 删除 |
| 5 | L4 启用阈值 | 第 2 次同模型族才 read；第 1 次只 write 防 cold-start 污染（所有 mode 都 read/write） |
| 6 | 新 skill 目录结构 | skill 入口在 `.cursor/skills/`，实现独立目录 `src/inference_optimizer/` |
| 7 | Watchdog guided 是否常驻 | 不常驻；emergency 时 Critic 兼任 ephemeral RCA（方案 A） |
| 8 | Sage 在 quick/guided 形态 | 非 reactor，作为 KB 查询服务由 Conductor 同步调用 |

---

## 20. 决策记录（ADR）

| ID | 决策 | 替代方案 | 理由 |
|---|---|---|---|
| ADR-1 | 合并 sprint + marathon 成新 skill | 保留两个独立 | KB 不分裂、深层 action 不被 skill 边界锁 |
| ADR-2 | 完整形态 4 个 persistent agent + Conductor，按 mode 启用 | 单 agent + ephemeral sub | 真团队（A2A、议会、Sage 长期记忆），但按 mode 子集启用 |
| ADR-3 | Sage 合并 Brainstormer + Historian | 6 个 agent | 跨 run 记忆 + 创意激发本质同源 |
| ADR-4 | Callable + 4 层记忆 | claude --continue 长进程 | 长期活着的"记忆优势"是幻觉 |
| ADR-5 | Executor/Watchdog = Claude opus-4-7 | 全 sonnet | 复杂工具使用 + RCA 需要强 reasoning |
| ADR-6 | Critic/Sage = Codex gpt-5.4 | 全 Claude | 异质模型视角、cost 低 |
| ADR-7 | Conductor 是 Python | LLM driven 调度 | 调度算法是数学，确定性 + 可单测 |
| ADR-8 | 复用 OOB backend 抽象 | 自己写 LLM client | 不重新造轮子 |
| ADR-9 | ~~复用 Claw runSubagent~~ **Superseded by ADR-14** | — | 见 ADR-14 |
| ADR-10 | 早停 5 信号 OR | 单一 MAX_HOURS | 真"效果驱动" |
| ADR-11 | L4 (跨 run KB) 全 mode 启用 | 仅 marathon | xiaofei #4：所有模式都受益于 KB |
| ADR-12 | A2A 自由通信 + 4 协议规则 | 中央化全过 Conductor | 真团队必须互相说话 |
| ADR-13 | Objective 抽象 (§8) | 调度公式直接绑 target_gain | 不支持 TARGET_TPUT/TARGET_DIR/MAX_HOURS-only |
| ADR-14 | OOB-only sub-agent，推翻 ADR-9 | Claw runSubagent / 改 Brain / TS rewrite | runSubagent 没有 HTTP route，方案 C 工作量最小 |
| ADR-15 | Event log + cursor 合一 + 状态机 (§13) | asyncio.Queue + 重放 | 原方案是 best-effort，崩溃在中间会重复/漏 |
| ADR-16 | 单 GPU sandbox 默认 + 多 sandbox TODO | 多 sandbox 一开始就上 | 简单优先；扩展能力保留待 sandbox 团队确认 |
| ADR-17 | 结构化 Intent Transport：Claude tool_call + Codex JSON-only (§10.5) | 自由文本 parse_intents / Codex 自定义 tool | Claude SDK / Codex CLI 输出格式不同；Codex 角色不需要 tools |
| ADR-18 | 资源锁模型 4 个 lane + durable file lease (§3.5) | 无锁 / 全局单锁 / 仅内存 asyncio.Lock | 共享 sandbox 必须显式资源模型，否则并行变测量污染；锁需跨进程/恢复可见 |
| ADR-19 | 早停 reason 分级尾流 (§7.2) | 所有 stop reason 跑同一尾流 | emergency 跑 sweep 可能压垮环境；time_exhausted 跑完整 sweep 超预算 |
| ADR-20 | Critic 在 guided 启用（轻量 review），marathon 完整 review | 仅 marathon 启用 Critic | guided 用 Critic 防钻牛角尖 + 防上下文撑爆 |
| ADR-21 | L4 第 2+ 次同模型族才 read | 第 1 次就 read | 第一次坏经验会污染 warm-start |
| ADR-22 | Persona 蒸馏机制（仅 marathon，§5.3） | 全 mode append-only | 长跑 token 预算变乐观；quick/guided 时间窗口短不需要 |
| ADR-23 | Brier 默认等权重，数据成熟后启用加权 | 各处定义不一致 | 实施分期统一 |
| ADR-24 | ~~PoC 前必须通过 Design Gate~~ **v0.4 删除（删 PoC 章节）** | — | xiaofei #5：完整设计先行，PoC 不着急 |
| ADR-25 | 运行时三档执行模式 (§3.4) | 按 Roadmap Phase 或统一 multi-agent 处理所有任务 | xiaofei 原始分段设计 |
| **ADR-26 (v0.4)** | **统一代码 + Feature Flag 子集架构** | 三套独立实现 / 复杂分支判断 | xiaofei #4：marathon 全集，quick/guided 是 marathon 的 feature flag 子集，共享同一套实现 |
| **ADR-27 (v0.4)** | **Watchdog guided 不常驻，emergency 时 Critic 兼任 ephemeral RCA**（方案 A） | guided 也常驻 Watchdog（方案 B） | 保持 watchdog 是 multi-agent 标志；guided 已有 Critic 双眼，节省 1M token / run |
| **ADR-28 (v0.4)** | **L4 KB 全 mode 启用读写** | 仅 marathon | xiaofei #4：所有模式都从 KB 受益（warm-start + ingest） |
| **ADR-29 (v0.4)** | **Sage 三段都启用，但 quick/guided 仅作 KB 查询服务（非 reactor）** | quick/guided 不启用 Sage / 全 mode 都常驻 | xiaofei：成本不高，让 quick/guided 也能用 Sage 召回 KB 知识；非 reactor 避免 token 浪费 |
| **ADR-30 (v0.4)** | **Iron Rules / KERNEL_OPT 常量表 / Process Mgmt / Accuracy Gate 全 mode 强制** | quick mode 简化版 | sprint+marathon 已有的硬资产，不要丢 |
| **ADR-31 (v0.4)** | **quick mode Bash allowlist 显式允许 server lifecycle，禁 workspace_write** | 全禁 / 全允许 | xiaofei #1：server restart 是测参数必备能力；workspace_write 才是不可信改动 |
| **ADR-32 (v0.4)** | **删除 Roadmap + PoC 章节** | 保留分阶段实施计划 | xiaofei #5：先把架构和功能设计好，实施推进暂不写 |

---

## 21. 与现有两个 skill 的对比

| 维度 | sprint | marathon | 新 skill |
|---|---|---|---|
| 用户认知负担 | 中 | 高 | **低**（一入口） |
| 短任务（<2h） | ✓ 但可能过度探索 | ✗ 大材小用 | ✓ 单 Agent 参数/后端探索，禁止 kernel-opt |
| 中任务（2-6h） | ✓ 但 sub-agent/恢复弱 | ✗ 偏重 | ✓ Agent + sub-agent + Critic 轻量 review |
| 长任务（>6h） | ✗ context 撑不住 | ✓ 但无 critic | ✓ multi-agent + critic + Watchdog + Sage |
| 效果驱动早停 | ✓ +25% 停 | ✗ | ✓ 用户自定义目标 + reason 分级 |
| 调度自适应 | 静态 score | 静态 score | **动态 pressure + lane gate + mode gate** |
| 多模型协同 | ✗ | 仅 kernel-opt | **按 mode 启用的异质多 agent** |
| 决策可追溯 | log 看 | log 看 | predictions + debates 入 KB |
| 长期校准 | 无 | 无 | Brier 长期跟踪（数据成熟后） |
| Sprint→Marathon 切换 | 手动 | 手动 | **state.json 无缝接续** |
| 跨 run "越来越快" | KB 单 skill | KB 单 skill | ✓ KB 全 mode 启用 |
| 载体复杂度 | 低 | 高 (tmux+CLI+NFS+npm) | 中 (Python + OOB + NFS) |
| 可恢复语义 | 无 | best-effort | **真可恢复 (event log + 状态机 + evidence)** |
| 资源争抢防护 | N/A | N/A | **4 个资源 lane + durable lease** |
| Iron Rules | ✓ | (沿用) | ✓ 全 mode 强制 |
| Token / 24h | N/A | 50-100M | **0.5M (quick) / 3M (guided) / 11.5M (marathon)** |

---

## 22. 评审请求

请按以下顺序评审 v0.4。评审目标是确认这一版完整设计是否成立。

1. **方向**：合并 + Multi-Agent + Budget-aware + Objective 抽象 + 单 sandbox + Feature Flag 子集架构是否认可？
2. **角色分工**：marathon 全集 4 agent + Conductor，Critic 在 guided 启用、Watchdog 仅 marathon 常驻、Sage 三段都用（quick/guided 仅 KB 查询服务）是否合理？
3. **记忆模型**：4 层 + 蒸馏（仅 marathon）+ L4 全 mode 启用 + warm-start 防 cold-start 是否完整？
4. **可恢复语义**：event log + cursor 合一 + 任务状态机 + evidence-check + needs_manual_review 是否够严密？
5. **资源锁**：4 个 lane + durable file lease / 现有锁接入是否覆盖所有冲突场景？
6. **Intent Transport**：Claude tool_call + Codex no-tools JSON-only 是否都可落地？
7. **运行时分段**：Feature Flag 矩阵 34 项 × 3 mode 是否符合预期？
8. **沿用资产**：Iron Rules / KERNEL_OPT 常量表 / Process Mgmt / Accuracy Gate / Initial Score Priors / 完整 20 action 是否完整吸收？
9. **PolicyGate**：quick mode Bash allowlist（允许 server lifecycle / 禁 workspace_write）是否合理？
10. **TODO**（§23）：哪些项需要立即跟进？

---

## 23. TODO 跟踪项

| ID | TODO | 触发动作 | 负责人 | 状态 |
|---|---|---|---|---|
| T1 | 跟 sandbox 团队确认是否能跨 sandbox 通信（CPU + GPU 分开） | 联系 sandbox 团队 | xiaofei | 未启动 |
| T2 | 默认单 GPU sandbox 起多个 GPU sandbox 用于并行 backend 测试（SaFE workload 动态创建套路） | T1 完成后 | TBD | 未启动 |
| T3 | Sage 跨 run KB 在多 user/多 session 共享时的隔离策略 | marathon 启用前 | TBD | 未启动 |
| T4 | 验证 CodexBackend no-tools + `validated_json_output` 的稳定性、repair 策略和 schema validation | 实施前 | 实施 agent | 未启动 |
| T5 | Marathon 兼容 thin shim 的具体实现（参数映射） | marathon 完整后 | TBD | 未启动 |
| T6 | KB schema (entries.jsonl + insights.jsonl + embeddings) 的具体 schema 设计（沿用 sprint，确认是否需扩展） | marathon 启用前 | TBD | 未启动 |
| T7 | 新 skill 名最终定稿 | 实施完成时 | xiaofei | 占位 `inference-optimizer` |
| T8 | 跨前端（Cursor / ClaudeCode）测试矩阵 | 实施末 | TBD | 未启动 |
| T9 | durable file lease / 现有锁代码接入、死锁检测算法（环路检测 / timeout 全局监控）和 multi-lane lease 原子获取语义 | 实施前 | TBD | 未启动 |
| T10 | 验证 §13.6 evidence-check 规则和崩溃点恢复矩阵是否覆盖所有 action | 实施前 | TBD | 未启动 |
| **T11 (v0.4)** | **kernel-opt 只优化原生 kernel，不优化 torch.compile 之后的 kernel**（理由：1) torch.compile 后的 kernel 不好优化，2) 损失精度的可能性比较大）；先记 TODO，不一定哪期做 | 待定 | xiaofei | 未启动 |

---

**End of Design v0.4 Final**
