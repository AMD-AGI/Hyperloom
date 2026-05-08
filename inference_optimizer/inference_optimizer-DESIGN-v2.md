# Inference Optimizer — Multi-Agent 自适应优化 Skill 设计方案

> **状态**: Final v0.6（v0.5 基础上统一为单一全模式 + 4 角色重组 + Critic 接管 KB / Sage 能力 + Robustness 接管 Robustness monitor/RCA/Handle + 删除议会模式 + orchestration 改名 orchestration + triage 改名 robustness）
> **作者**: 主 Agent + xiaofei 共同设计
> **日期**: 2026-04-30
> **占位 skill 名**: `inference_optimizer`
> **目标读者**: 工程领导评审 + 落地实施 Agent
> **基线**: 本文档替代 `inference_optimizer-DESIGN.md` v0.4 / `inference_optimizer-DESIGN-modified.md` v0.5;后两者作为历史参考保留
> **实施分工**: zhenggong 已不再继续开发;xiaofei 在当前 kernelAgent 分支上做整体落地（合并或参考 zhenggong 分支重写）;**P0 先跑通 Coordinator + Orchestration + KernelAgent 主链路**;Critic / Robustness 角色由其他人按本文档协议实现,在 P0 阶段先用 mock adapter,不得阻塞主链路
> **v0.5 → v0.6 摘要**: 删除 quick / guided / marathon 三档分段,统一为单一"全模式";orchestration 改名 orchestration、triage 改名 robustness(贴合 Hyperloom 优化栈架构图 6-agent 命名:4 layer experts = Orchestration/Framework/Kernel/Comm + 2 cross-layer = Critic/Robustness;v0.6 实现其中 4 个,Framework/Comm 见 §7.7 占位);critic 改 Codex gpt-5.4 + 接管 KB read/write + 接管 Sage 能力 + 负责 review 优化建议(approve/reject/redirect/advise),**不做 RCA**;robustness 正式确立为 Robustness monitor + RootCauseAnalysis + Handle 合并体并保留调度警察权限;删除议会(parliament / objection / vote)由 Critic 单 agent review 替代;framework + comm agent 不做;framework-rebuild action 已移除;multi-cli runtime 作为本地过渡方案保留,等待 claw 子 session 能力上线后退役

---

## 0. v0.5 → v0.6 变更日志


| 类别            | 变更                                                                                                                                                                                                                                                                                                                                           | 触发原因                                                                                                         |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **架构主线**      | 删除 `quick_param_sweep` / `guided_kernel_opt` / `marathon_multi_agent` 三档,统一为**单一全模式**;`MAX_HOURS` 不再用于 mode 选择,只用于早停预算 + 调度 pressure                                                                                                                                                                                                         | xiaofei #7:v0.5 实现里 guided 与 marathon 的 reactor roster 已经一致,只剩 checkpoint cadence + 少量 flag 差别;按"专注于全模式"原则统一 |
| **角色重组**      | `orchestration` → `orchestration`(改名 + 职责对齐架构图 Layer-1 expert)                                                                                                                                                                                                                                                                                    | xiaofei #排版2:贴合新架构图                                                                                          |
| **角色重组**      | `robustness monitor` + `sage 主动维护`中的 RCA/handle 相关职责合入 `robustness`;`robustness` 正式确立为 `Robustness monitor + RootCauseAnalysis + Handle` 合并体                                                                                                                                                                                                                     | xiaofei #2:robustness 本身就是这三件事的合并体;实现已经走这个方向                                                                 |
| **角色重组**      | `critic` 改回 Codex `gpt-5.4`(litellm 暂不支持 5.5),并接管:review 优化建议(approve/reject/redirect/advise) + KB read/write + KB 召回服务(原 Sage 在 quick/guided 的形态) + Devil's advocate(无议会);**Critic 不做 RCA,RCA 属于 Robustness**                                                                                                                               | xiaofei #3 #4:critic 用 codex 5.4;KB 逻辑放 critic;后续确认 Critic 不做 RCA                                            |
| **协作模式**      | 删除议会模式 + `objection` / `vote` / `parliament_open` / `vote_request` topic 与 intent;由 **Critic Review 协议**(approve / reject / redirect / advise 四种常规 verdict + P0/mock 用 `needs_review`)替代                                                                                                                                                     | xiaofei #5:不要议会,只 critic review                                                                              |
| **KB 归属**     | KB 最终由中心化共享存储收集和查询,按 `<model_family>/<model_name>` 分区,跨 sandbox/session/user 共享;**但中心化 KB 不属于 P0**,由 Critic owner 后续实现;P0 Critic mock 不依赖真实 KB                                                                                                                                                                                               | xiaofei #4 + 后续追问:KB 中心化、按模型分类、Critic 做;最新确认中心化 KB 先不做                                                       |
| **Action 体系** | `framework-rebuild` action 删除,总数 20 → 19;`deep-kernel-analysis` / `operator-tuning` / `vendor-kernel-config` 这 3 个深层 kernel action 与 `kernel-opt` / `integrate` 一并由 **Kernel agent owns**,Orchestration 通过 REQUEST 派发                                                                                                                        | zhenggong 已删 framework-rebuild;架构图里 Kernel 拥有这 5 个                                                           |
| **Agent 范围**  | 架构图里的 `framework agent` / `comm agent` 不做                                                                                                                                                                                                                                                                                                    | xiaofei:这两个先不做                                                                                               |
| **落地优先级**     | **P0 只要求 Coordinator + Orchestration + KernelAgent 跑通**;Critic / Robustness 先 mock,协议边界先留好,不阻塞主链路;后续由专人替换为真实实现                                                                                                                                                                                                                                 | xiaofei:critic 和 robustness 有专人做,先 mock;当前先把主链路跑通                                                            |
| **持久化**       | 沿用 v0.5 SQLite WAL 单库(4 表 leases/events/cursors/tasks);多文件 jsonl + file lock 方案彻底退役                                                                                                                                                                                                                                                          | xiaofei #1:SQLite 保留                                                                                         |
| **持久化生命周期**   | **显式契约:SQLite per-session,每次 run 在 `$SESSION_DIR/storage/coordinator.db` 新建独立 DB,session 结束即废弃**;**DB 直接落 NFS(WekaFS),不再用"本地盘 + backup"两层** → coordinator crash / sandbox 重新分配都自然恢复,不需要 restore 流程;跨 session 经验只走中心化 KB(ADR-42 修订)                                                                                                               | xiaofei:每次任务都是独立的,每个任务都新建一个;直接固化到 NFS 防 crash 丢失                                                             |
| **KB 形态**     | **明确为中心化共享存储,按 `<model_family>/<model_name>` 分区,Critic 是唯一 read/write/synthesis 入口**;具体载体(NFS 共享路径 vs HTTP service)由 T3 落地;v0.5 描述模糊,v0.6 写死(ADR-43)                                                                                                                                                                                         | xiaofei:KB 共享、按模型分类、中心化、Critic 做                                                                             |
| **传输模式**      | `MULTI_CLI`(基于 `claude --print --continue` + JSONL inbox/outbox)作为**本地运行过渡方案**保留,默认 `SINGLE_PROC`;后期 claw 提供"一 agent 一 sub-session"能力后退役                                                                                                                                                                                                     | xiaofei #6                                                                                                   |
| **文档卫生**      | 删除 `IMPLEMENTATION-CHECKLIST.md`(已删,不再维护);v0.4 / v0.5 文档作为历史保留                                                                                                                                                                                                                                                                               | xiaofei #8                                                                                                   |
| **新增 ADR**    | ADR-34(单一全模式)/ ADR-35(Critic 接管 KB + Sage 能力 + Review)/ ADR-36(Robustness = Robustness monitor + RCA + Handle)/ ADR-37(orchestration → orchestration 改名)/ ADR-38(删除议会,Critic Review 替代)/ ADR-39(multi-cli 作为本地过渡)/ ADR-40(framework/comm agent 不做)/ ADR-41(删除 IMPLEMENTATION-CHECKLIST)/ ADR-42(SQLite per-session)/ ADR-43(KB 中心化共享 + 按模型分区 + Critic 唯一入口) | 上述决策对应记录                                                                                                     |
| **保留**        | Iron Rules / KERNEL_OPT 常量表 / Process Management / Accuracy Gate / Initial Score Priors / Objective 抽象 / Budget-Aware 调度器 / Resource Lock 4-lane / Intent Transport / Plan A Kernel agent / SubAgentRunner                                                                                                                                   | 沿用 v0.4 / v0.5                                                                                               |


---

## 1. TL;DR(一页汇报版）

把现有 `inference-optimization`(sprint,单 agent ≤3h)和 `marathon-inference-optimization`(长跑 24h+,tmux+claude CLI)合并成**一个**自适应、效果驱动、多 agent 协作的新 skill。

### 1.1 核心架构原则:**单一全模式 + 4 个 Persistent Agent + Ephemeral Sub-agent 池**

> v0.4 ~ v0.5 的"统一代码 + Feature Flag 三档子集"在 v0.6 进一步收敛:**只有一个执行模式**。
> 所有 agent 全启用、所有 action 全可选;`MAX_HOURS` 只决定预算 + 调度 pressure,不再决定哪些 reactor 启动。
> 任务大小不同时,**按 Objective + 调度器 + 早停**自然区分,不再靠预设 mode 切换功能集。

具体到 4 个 persistent agent:

- **Orchestration agent**(原 Orchestration,Claude opus-4-7)— 提议 action / 委托 sub-agent / 解读结果 / 通过 REQUEST 调 Kernel agent / 接受 Critic review
- **Kernel agent**(Plan A,Claude opus-4-7,responder-only)— 独占 5 个 kernel 类 action(`kernel-opt` / `integrate` / `deep-kernel-analysis` / `operator-tuning` / `vendor-kernel-config`),只响应 Orchestration 的 REQUEST,不主动 propose / delegate
- **Critic agent**(Codex gpt-5.4,no-tools 原则 + KB 例外)— review Orchestration/Kernel 提出的优化建议(approve / reject / redirect / advise) + 给更高层建议 + KB read/write + 跨 run 召回 + Devil's advocate(无议会);**不做 RCA**;Brier-tracked
- **Robustness agent**(Claude opus-4-7,always-on)= **Robustness monitor + RootCauseAnalysis + Handle** 合并体 — event_log 监控 + 健康检查 + server lifecycle + accuracy gate exec + recovery + 调度警察 4 个 intent(`kill_task` / `force_dispatch` / `prune_branch` / `escalate_strategy_change`)
- **Coordinator**(Python,无 LLM)— 主循环 + bus + state + 资源锁 + REQUEST/RESPONSE 路由 + 早停 + checkpoint + resume

加上 **Ephemeral Sub-agent 池**(`bench_runner` / `profile_runner` / `kernel_extract` / `geak_submitter` / `patch_applier` / `eval_runner` / `rca_runner`),完成所有 GPU / workspace 副作用动作。

### 1.2 核心创新 5 条

1. **单一全模式 + Objective + Budget-Aware 调度**:用户必填 `MODEL_PATH + MAX_HOURS`,可选 `TARGET`_*;不再做 mode 分档,所有功能默认启用;调度器按 Objective 抽象 + 时间预算动态决定下一步,达到目标立即早停。
2. **4 个 persistent agent + Ephemeral Sub-agent 池**:Orchestration 提议 / Kernel 执行深层 kernel / Critic review + KB / Robustness 守护 + 干预 + handle;sub-agent 跑短任务即销毁。无议会、无投票,**Critic 一票 verdict 直接定方向**。
3. **4 层记忆模型 + Critic 主导 KB(非 P0)**:L1 即时 / L2 session(SQLite events) / L3 persona / L4 **中心化共享 KB**(按 `<model_family>/<model_name>` 分区,Critic 是唯一 read/write/synthesis 入口);中心化 KB 由 Critic owner 后续实现,P0 mock 不依赖真实 KB。
4. **可恢复语义(SQLite WAL 单库)**:`leases / events / cursors / tasks` 4 表合一,跨类型操作走单事务 `BEGIN IMMEDIATE`,resume 时事实源唯一;副作用 action 失败默认 `needs_manual_review`。
5. **结构化 intent + 资源锁 + Critic Review 闸门**:Claude 角色通过 `emit_intent` MCP tool 发意图;Codex Critic 输出 validated JSON;sandbox 内 4 个资源 lane 通过 SQLite 单事务原子互斥;PolicyGate 校验所有 intent;**带 `accuracy_risk > 0` 或重大方向变更的 proposal 必须先过 Critic Review**(approve / reject / redirect / advise)才能进入执行。

### 1.3 沿用 sprint + marathon 的硬资产(全部继承,无变更)


| 资产                                         | 来源              | 应用范围                                                      |
| ------------------------------------------ | --------------- | --------------------------------------------------------- |
| Iron Rules（IR-1 ~ IR-7）                    | sprint          | 全模式强制（§4）                                                 |
| KERNEL_OPT 常量表                             | sprint          | 全模式（§5）                                                   |
| Process Management 规则                      | sprint+marathon | 全模式（§6）                                                   |
| Accuracy Gate 协议（GSM8K + 0.01 阈值）          | sprint          | 所有有 accuracy_risk 的 action（§10）                           |
| Initial Score Priors（model_class × action） | sprint          | 调度器初始化（§12）                                               |
| 19 个 action（去掉 framework-rebuild）          | sprint+marathon | 按 applicable_when 过滤,kernel 类 5 个由 Kernel agent owns（§16） |
| KB warm-up + ingest 脚本                     | sprint          | 全模式;**入口在 Critic**（§8）                                    |


### 1.4 载体

单 GPU sandbox(默认,简单优先)+ NFS(WekaFS)上的 **per-session** SQLite WAL DB(每次 run 在 `$SESSION_DIR/storage/coordinator.db` 新建一个,session 结束即废弃,原子语义,任何 crash 都不丢失)+ NFS(personas / state snapshot / findings)。**中心化 KB 服务是 Critic 后续工作,不属于 P0 载体依赖**。多 sandbox 拓扑作为扩展能力保留(详见 §26)。Multi-CLI runtime 作为本地过渡方案,等 claw 提供子 session 能力后,改成"一 agent 一 sub-session"形态。

### 1.5 预期效果

- 用户体验:一个入口,3 行 env 起跑,不再需要选 mode
- 效果驱动早停 → 用户实际等待时间显著下降
- P1+ 跨 run 记忆(中心化 KB + Critic 召回) → 第 N 次跑同模型显著加快收敛(包括跨 user 共享经验)
- Critic 一票 verdict 防钻牛角尖 + KB 知识背书,避免重复历史教训
- Robustness always-on 守护,crash / stall / dead-end loop 自动干预

---

## 2. 背景与动机

### 2.1 现状两个 skill 的问题


| 问题                      | sprint              | marathon                                          |
| ----------------------- | ------------------- | ------------------------------------------------- |
| 单 agent context 撑不住 24h | ✓ 是问题(≤3h 上限)       | 用 tmux+claude CLI 解决                              |
| 长跑没有效果驱动早停              | —                   | ✗ MAX_HOURS 是硬墙钟                                  |
| 协议被 skill 边界切开          | sprint 11 浅层 action | marathon 9 深层 action(不能用在短任务)                     |
| 单模型决策偏差                 | ✗ Orchestration 自决       | ✗ 同                                               |
| KB 分裂                   | sprint kb/          | marathon SPEC_ROOT/kb/,跨 run 知识不复用                |
| 长跑载体重                   | —                   | tmux + npm install + base64 prompt + NFS file IPC |
| 角色边界不清                  | 单 agent             | 多 agent 但责任划分混乱                                   |


### 2.2 v0.4 → v0.6 路径上踩过的坑


| 阶段            | 设计                                                                                                                            | 落地结果                                                                        |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| v0.4          | 4 角色(orchestration/critic/robustness monitor/sage) + 三档 mode + 议会 + 多文件持久化 + KB 全 mode 启用                                                    | 角色边界过细,议会成本高,持久化容易半截饭                                                       |
| v0.5          | + SQLite WAL 单库 + Plan A Kernel agent                                                                                         | 持久化扎实了;但 robustness monitor 与 sage 的"常驻 vs 兼任"分歧没收口,critic 在 guided/marathon 介入策略也没收口 |
| v0.5 实现       | 实际删了 robustness monitor + sage,新增 robustness,critic 改成 Claude no-tools                                                                  | 文档与代码脱节;KB 设施在但全 mode flag 关闭;quick / guided / marathon 在实现里 roster 几乎相同    |
| **v0.6**(本文档) | **统一全模式 + 4 角色明确(Orchestration / Kernel / Critic / Robustness) + Critic owns KB + 删除议会 + Multi-CLI 作为过渡 + Framework/Comm 不做** | **目标:文档 = 代码,角色边界清晰,KB 真用上,议会成本归零**                                         |


### 2.3 核心痛点

> "用户要的是效果,不是时间。一个入口,功能全开,什么任务都能跑。同一个模型反复优化要越来越快。Critic 一票把关,Robustness 兜底救场。"

→ 5 个能力:**自适应 / 效果驱动 / 真团队 / 长期记忆 / 反方背书**。

---

## 3. 整体架构

### 3.1 单 GPU sandbox 拓扑(单一全模式形态)

下图是 v0.6 单一全模式下的拓扑。所有 4 个 persistent agent 全启用,所有 19 个 action 都可以被调度;**不再有"按 mode 关闭某些 reactor"的逻辑**。

```
                    用户 (Cursor / ClaudeCode CLI)
                              │ trigger skill (MODEL_PATH + MAX_HOURS + 可选 TARGET_*)
                              ↓
                          Claw Brain
                              │ create sandbox
                              ↓
╔═══════════════════ GPU Sandbox (单实例) ═══════════════════════════════════╗
║                                                                             ║
║  ┌───────── Coordinator (Python, 无 LLM, 永远存在) ───────────────────┐      ║
║  │  主循环 / MessageBus / SharedState / ResourceLockManager /         │      ║
║  │  TaskRegistry / PolicyGate / Scheduler / Checkpoint /              │      ║
║  │  REQUEST/RESPONSE 路由 (Plan A) / Critic Review 闸门 (§18)         │      ║
║  └────┬───────────────────────────────────────────────────────────────┘      ║
║       │                                                                       ║
║       │  ╔═══════ LLM Persistent Agent Pool (4 个全启用) ═══════════╗        ║
║       │  │  Orchestration   Claude opus-4-7  reactor               │        ║
║       │  │    └─ owns: setup/classify/target-analysis/baseline/    │        ║
║       │  │       profile/backends/params/sweep/report (9 actions)  │        ║
║       │  ├──┤  Kernel        Claude opus-4-7  reactor (responder)  │        ║
║       │  │    └─ owns: kernel-opt/integrate/deep-kernel-analysis/  │        ║
║       │  │       operator-tuning/vendor-kernel-config (5 actions)  │        ║
║       │  │  Critic         Codex   gpt-5.4   reactor (no-tools+KB)│        ║
║       │  │    └─ Review 优化建议 + KB read/write(后续) +            │        ║
║       │  │       Devil's advocate;不做 RCA                         │        ║
║       │  │  Robustness         Claude opus-4-7  reactor (always-on)   │        ║
║       │  │    └─ Robustness monitor + RCA + Handle:event_log monitor /       │        ║
║       │  │       crash signal / health check / server lifecycle /  │        ║
║       │  │       accuracy gate exec / recovery /                   │        ║
║       │  │       调度警察 4 intent(kill_task / force_dispatch /    │        ║
║       │  │       prune_branch / escalate_strategy_change)          │        ║
║       │  ╚════════════════════════════════════════════════════════╝        ║
║       │                                                                       ║
║       │  ╔════ Ephemeral Sub-agent Pool (按需 spawn,跑完销毁) ═════╗      ║
║       │  │  bench_runner    (benchmark_lane)                          │      ║
║       │  │  profile_runner  (profile_lane)                            │      ║
║       │  │  kernel_extract  (只读)                                    │      ║
║       │  │  geak_submitter  (外部 GEAK MCP,Kernel agent 调用)        │      ║
║       │  │  patch_applier   (workspace_mutation + server_lifecycle)   │      ║
║       │  │  eval_runner     (benchmark_lane,跑 GSM8K)                 │      ║
║       │  │  rca_runner      (按需,Robustness 调起)                        │      ║
║       │  │                                                             │      ║
║       │  │  实现:Coordinator 直接 spawn OOB ClaudeBackend.run()          │      ║
║       │  │       / CodexBackend.run(),fresh context、跑完即销毁        │      ║
║       │  ╚═══════════════════╤═════════════════════════════════════════╝      ║
║                              ↓                                                ║
║                    推理 server (sglang/vllm)                                  ║
║                         (占 server_lifecycle lane)                            ║
╚══════════════════════════════╤════════════════════════════════════════════════╝
                               ↓
        ╔════════════════════════════════════════════════════════════╗
        ║   Shared NFS (WekaFS) — 所有持久化资产都在这里                  ║
        ║                                                             ║
        ║  $SESSION_DIR/  (per-session,session 结束即废弃)             ║
        ║   - storage/coordinator.db        (SQLite WAL 单库,4 张表:    ║
        ║                                  leases / events / cursors / ║
        ║                                  tasks;直接落 NFS,无 backup ║
        ║                                  层,任何 crash 都不丢失)    ║
        ║   - state.json                  (L2 snapshot,辅助调试)      ║
        ║   - personas/<agent>.md         (L3)                        ║
        ║   - results/<task_id>/          (sub-agent 输出)             ║
        ║   - findings/<id>/              (RCA 结果)                   ║
        ║   - agents/<name>/inbox.jsonl|outbox.jsonl  (multi-cli)     ║
        ║                                                             ║
        ║  中心化 KB 服务 (跨 session/user,T13 后续落地,非 P0)         ║
        ║   - 按 <model_family>/<model_name> 分区                     ║
        ║   - Critic 是唯一 read/write 入口                           ║
        ╚════════════════════════════════════════════════════════════╝
```

**核心设计点**:

- **同一份代码,无 mode 分支**:不再写 `if mode == quick / guided / marathon`,所有 reactor / action / cadence 都常开
- **Coordinator 永远存在**:Python、无 LLM;负责调度 / 锁 / 状态机 / 早停 / checkpoint / Critic Review 闸门 / REQUEST/RESPONSE 路由
- **4 个 LLM agent 全启用**:Orchestration + Kernel + Critic + Robustness(注:架构图里 Framework / Comm 不做)
- **Sub-agent 是 fresh OOB backend.run()**:不复用 Claw runSubagent;跑完即销毁
- **资源 lane 通过 SQLite WAL 单事务原子互斥**:取代 v0.4 的多文件 `O_CREAT|O_EXCL` 方案
- **持久化分两类**:**SQLite 单库**(协调原语,要原子性)**per-session 直接落 NFS**(WekaFS,`$SESSION_DIR/storage/coordinator.db`),session 结束即废弃,任何 crash 都不丢失,无需 backup → restore 两层(前提:WekaFS 支持 SQLite WAL fcntl lock,落地前 self-test 验证,见 T20);**其它资产**(state snapshot / personas / findings)同 SESSION_DIR 直接落 NFS;**跨 session 长期记忆**走中心化 KB 服务,与 SESSION_DIR 解耦

### 3.2 多 sandbox 扩展方向(TODO,详见 §26)

未来可拓展到 CPU + GPU 分离:CPU sandbox 跑 Coordinator + LLM agent + 思考型 sub-agent;GPU sandbox 跑推理 + benchmark。需要跟 sandbox 团队确认能力。multi-cli runtime(§20)是这个方向的过渡形态。

### 3.3 关键概念分离


| 概念                      | 定义                                                                                                     |
| ----------------------- | ------------------------------------------------------------------------------------------------------ |
| **Persistent Agent**    | 长期"角色"(思考 / 决策 / 协商 / review),通过 callable 多次唤醒;v0.6 全启用,无 mode gating                                  |
| **Ephemeral Sub-agent** | 短期"动作"(具体执行),fresh OOB backend.run(),跑完销毁;由 Coordinator 调起                                               |
| **Coordinator**           | 协议管理员(不是中央决策者),编排 message + 时钟 + 仲裁 + 早停 + 资源锁 + REQUEST/RESPONSE 路由 + Critic Review 闸门                |
| **Backend**             | OOB 抽象,把 Claude / Codex 包成统一 `run(prompt, ...) → AgentResult`                                          |
| **Resource Lane**       | 互斥资源类别(server_lifecycle / workspace_mutation / benchmark_lane / profile_lane),sub-agent 必须先取 lease 才能动 |
| **Critic Review 闸门**    | 副作用 action 在执行前必须由 Critic 出 verdict(approve / reject / redirect / advise),由 Coordinator 拦截执行(§18)        |
| **Robustness 干预**       | always-on Robustness monitor,可发 4 个调度警察 intent 主动改变执行轨迹(§19)                                                     |
| **REQUEST/RESPONSE 路由** | Plan A 跨 agent RPC;Orchestration → Kernel agent 是当前唯一启用的 (source, target) 对                            |


### 3.4 单一全模式 vs 三档的演化论证

> 这是 v0.6 与 v0.4 / v0.5 最大的架构差。理由如下。

#### 3.4.1 v0.5 实现观察

v0.5 实施过程中,`guided` 与 `marathon` 的 reactor roster 已经收敛成同一个集合(`[orchestration, critic, kernel, robustness]`),区别只剩:

- `enable_strategic_review` / `enable_event_driven_alert` 两个 flag(其实 marathon 也没真用上)
- checkpoint cadence 略不同
- prompt 长度略不同

`quick` 模式删掉了 `kernel-opt` / `integrate` 但保留了 `delegate`,实质上跟 guided 的差别只是"action 子集"。

#### 3.4.2 三档的真实成本 vs 收益


| 成本项  | 实际付出                                                                     |
| ---- | ------------------------------------------------------------------------ |
| 测试矩阵 | 每个新功能要在 3 个 mode 下分别测,实际测了 ~600 项                                        |
| 文档心智 | 每条规则要标 "适用 mode",新读者要先记住 mode 矩阵                                         |
| 实现分支 | `feature_flags.py` + `roles_for_mode` + `allowed_modes` per action,3 套配置 |
| 维护成本 | 任何一项功能加减都要 review 3 套配置,经常忘改一处                                           |



| 收益项         | 实际拿到                                                                        |
| ----------- | --------------------------------------------------------------------------- |
| 短任务节省 token | quick 估 0.5M / 2h,但实际跑下来 Critic+Robustness 一开,跟 guided 差距 < 30%             |
| 短任务限制副作用    | quick 禁 workspace_write 是真有用,但这是 PolicyGate + ActionMetadata 的事,不需要 mode 来分 |
| 长任务多 agent  | 现在 4 agent 全开本来就能服务长任务,不需要"打开 marathon mode 才有"                             |


→ **三档分段的成本远大于收益**。把"短任务限制副作用"留给 PolicyGate + Action.applicable_when 表达;把"短任务省 token"留给 Objective + 调度器自然实现(任务小、调度器选低成本 action、Critic review 低风险 action 时降级为 sample 20%)。

#### 3.4.3 v0.6 替代方案

```python
# v0.6:不再有 ExecutionMode enum
# 取代:Objective + Scheduler + Critic Review 自然分流

def choose_runtime(env) -> RuntimeConfig:
    validate_required(env, ["MODEL_PATH", "MAX_HOURS"])
    return RuntimeConfig(
        objective=build_objective(env),         # §11
        max_hours=float(env["MAX_HOURS"]),
        agents_enabled=ALL_4_AGENTS,            # 永远全启用
        actions_allowed=ALL_19_ACTIONS,         # 永远全启用,applicable_when 过滤
    )
```

**MAX_HOURS 仍必填**,只是它的作用从"决定哪些 reactor 启动"退化为"决定预算 + 调度 pressure + 早停时钟"。

#### 3.4.4 短任务的实际行为(无需 mode)

用户给 `MAX_HOURS=1.5 + TARGET_GAIN_PCT=10`,系统的自然反应:

1. 调度器:`pressure = max(objective_progress_input, 1.0 - time_left/total)` ≈ 高;`depth_gate` 砍掉所有 `cost_p75 > 1.2h` 的 action,自然就排除了 `kernel-opt`(P75 = 120min)和长跑类
2. Critic Review:对 `accuracy_risk = 0` 的 action(参数 sweep / backends 切换)默认 approve 或 sampled 20%,不阻塞
3. Robustness:always-on,但没有 crash 就静默
4. 早停:达到 +10% 立即停

**等价于原 quick mode 行为,但不需要写 mode = quick 的分支代码**。

### 3.5 资源锁模型

#### 3.5.1 问题(同 v0.4 / v0.5)

所有 sub-agent 共享同一个 GPU sandbox 和 `/workspace`。如果不加锁:

- `bench_runner` 跑 benchmark 时另一个 sub `patch_applier` 改了 server 文件 → bench 结果污染
- 两个 `bench_runner` 同时跑 → GPU 被抢,吞吐数据失真
- `profile_runner` 在 profile 时 `bench_runner` 起 bench → profile trace 错乱

#### 3.5.2 4 个资源 Lane


| Lane                 | 互斥粒度    | 持有者类型                                                                         | 典型 lease 时长 |
| -------------------- | ------- | ----------------------------------------------------------------------------- | ----------- |
| `server_lifecycle`   | 全局唯一持有者 | patch_applier / kernel-opt integrate / 重启 server 的 action / Robustness handle | 30s ~ 10min |
| `workspace_mutation` | 全局独占    | 写 patch / 改 inductor cache / 改配置文件                                            | <30s        |
| `benchmark_lane`     | 全局唯一持有者 | bench_runner / sweep / eval_runner                                            | 1 ~ 30min   |
| `profile_lane`       | 全局唯一持有者 | profile_runner                                                                | 1 ~ 5min    |


#### 3.5.3 跨 lane 互斥规则

```
benchmark_lane 持有时   → 禁止: patch_applier (server_lifecycle) 起新动作
                                profile_runner (profile_lane) 起
                          允许: 只读 sub-agent (kernel_extract 等)

profile_lane 持有时    → 禁止: bench / sweep / eval / patch
                          允许: 只读 sub-agent

server_lifecycle 持有时 → 禁止: bench / profile / eval (server 在重启不能用)
                          允许: 只读 sub-agent

workspace_mutation 持有 → 禁止: 任何 reader 读相关文件
                          (短锁,~30s)
```

#### 3.5.4 实现:SQLite WAL 单库后端(沿用 v0.5 ADR-33)

```python
class SqliteLeaseBackend:
    """v0.5+ default: 4 类持久化合并到 $SESSION_DIR/storage/coordinator.db。"""

    async def acquire_many(self, lanes, holder_id, ttl_sec):
        # 1. Expand cross-lane conflicts into a canonical lane set.
        # 2. Sort lanes by fixed order (workspace_mutation → server_lifecycle
        #    → benchmark_lane → profile_lane) to avoid deadlock.
        # 3. BEGIN IMMEDIATE; INSERT INTO leases (4 行) — PRIMARY KEY 冲突
        #    任意一行触发 ROLLBACK,实现"all or nothing"。
        # 4. 失败时 release 已获取 lane(若任何插入成功) + 写 lease_blocked event,
        #    退避 + 重试。
        # 5. expired lease 不能静默覆盖,必须先 INSERT lease_expired event 再
        #    DELETE 旧 lease,事务内完成。
        ...
```

#### 3.5.5 原子 multi-lane lease 规则

- 所有 action 只能调用 `acquire_many(required_lanes)`,禁止嵌套获取单 lane
- lane 获取顺序固定:`workspace_mutation` → `server_lifecycle` → `benchmark_lane` → `profile_lane`
- 单 SQLite 事务保证 atomic acquire-many;失败回滚自动释放
- lease 必须有 TTL 和 heartbeat;超时后不能静默抢锁,必须先写 `lease_expired` event
- **acquire_many 失败策略**:非阻塞 + 指数退避(100ms → 1s → 5s),总等待上限 `action.lease_ttl × 2`;超过则任务转 `failed` 并把 action 回灌给调度器,不阻塞 reactor

#### 3.5.6 Coordinator 调度时检查

调度器决定下一个 action 时,如果它需要的 lane 已被占,要么 wait,要么选另一个 action。这通过 action.metadata 里的 `requires_lanes` 字段声明。真正执行前仍必须调用 `acquire_many()`;调度检查只是优化,不能替代 lease。

#### 3.5.7 部署 & 生命周期(per-session,直接落 NFS)

- **每次 run = 一个新 session = 一个独立 SQLite DB**:每次 `@inference_optimizer` 触发都新建 `$SESSION_DIR`,在其中初始化空 DB;不跨 session 复用、不跨 user 共享、不需要 schema migration(每次都是 fresh schema)
- **DB 直接落 NFS(WekaFS)**:`$SESSION_DIR/storage/coordinator.db`,与 personas / state / findings 同 SESSION_DIR;这样 coordinator 进程 crash / sandbox 重启 / sandbox 重新分配只要 `SESSION_DIR` 可访问就能直接重新打开,**不需要 backup → restore 两层**(ADR-42)
- **前提**:WekaFS 是企业级 POSIX 兼容文件系统,fcntl lock + SQLite WAL 模式应该可用;落地前必须跑 self-test 验证(T20):多进程并发写、断电恢复、wal/shm 文件交互;**self-test 失败的 fallback** 是回到"本地盘 DB + 30min NFS VACUUM INTO backup + restore"两层方案
- Resume 范围:**仅同一 session 内**有效——coordinator / sandbox 任何形式的 crash 重启,只要 `SESSION_DIR` 还在,DB 就还在,直接打开继续;**session 结束 = DB 废弃**,跨 session 一律不复用
- 长期记忆走 KB(L4):跨 session 的"同模型历史经验"目标上通过**中心化共享 KB 服务**(按 `<model_family>/<model_name>` 分区,Critic owns)传递,不依赖 SQLite;**该能力非 P0**,P0 可无真实 KB;这样可以让 SQLite 始终保持"小、快、可丢弃"

---

## 4. Iron Rules(沿用 sprint,全模式强制)

> 这些规则是**全模式硬约束**,违反任何一条 = 该次 run 无效。沿用原 `inference-optimization` skill 的 IR-1 ~ IR-7。v0.6 取消 mode 适用范围标注,统一全模式强制(N/A 由 action 是否触发决定)。

### IR-1: Submit ALL kernel candidates in parallel

`kernel-opt` action 必须把 `GEAK_TOP_CANDIDATES`(默认 5)个 candidate 同时提交给所有激活的 `KERNEL_OPT_BACKENDS`。**只交一个 candidate 或顺序交多个 backend = 违规**。**Kernel agent 主导,Orchestration 通过 REQUEST 触发。**

### IR-1a: Ray must advertise all visible GPUs

Kernel backends(OOB / GEAK / Claude / Codex)通过 Ray 提交 `num_gpus>=1` 的
worker task。启动 kernel-agent backend 前必须确认 Ray head 以当前 sandbox
可见 GPU 数启动:

```bash
ray stop --force || true
ray start --head --disable-usage-stats --num-gpus="$GPU_COUNT" --include-dashboard=false
ray status
```

`ray status` 必须显示 `0.0/<GPU_COUNT> GPU`。禁止用 `--num-gpus=0` 启动
Ray;否则 kernel backend 会长期 pending,表现为 optimizer 仍在跑但
`run_optimization` 不产出有效 attempt。pod / venv 重建后必须重新执行
Ray/Click pin + Ray 全 GPU 启动检查。

### IR-2: NEVER modify kernel source before GEAK submission

提交的 kernel source **必须与 inductor cache 中提取的完全一致**。不允许:剥装饰器、改 strides、把 `@triton_heuristics` 替换成 `@triton.jit`、做任何"清理"编辑。GEAK 内部会处理 kernel 适配。

### IR-3: Integration is MANDATORY

GEAK 返回优化后的 kernel 后,必须执行 `integrate` action(patch → re-baseline → KEEP/REVERT 决策)。跳过 integrate = GEAK 结果未端到端验证 = 违规。`integrate` 必须用 `run_baseline.sh`(不是 `run_benchmark.sh`,后者不存在)。**Kernel agent owns。**

### IR-4: Always kill_server + check_gpu_memory before server launch

每次启动 server 前必须先 kill 已存在的 server 进程,并验证 GPU 显存已释放。**Robustness handle 也要遵守**(它是 server lifecycle 的另一个合法源)。

### IR-5: Safe process management

**禁止使用 `pkill -f sglang`** —— 在 claw mode 下会 kill Ray worker。只允许:

```bash
kill $(pgrep -f 'python.*-m sglang.launch_server') 2>/dev/null
# 或 vLLM:
kill $(pgrep -f 'python.*-m vllm.entrypoints') 2>/dev/null
```

kill 与 relaunch 之间必须等 `SERVER_KILL_WAIT_S` 秒(默认 10s)。profiling 完后必须 `unset PROFILE SGLANG_TORCH_PROFILER_DIR`。

### IR-6: Use `patch_inductor.py --target-file` for Inductor patching

必须用 `scripts/patch_inductor.py` + `--target-file`。`--cache-dir` 选项已移除。

**关键**:当 GEAK 改了 block size 或 warp count 时,必须同时传 `--best-config` 含更新的 tiling 参数;只 patch kernel `.py` 不更新 `.best_config` 会导致数值崩坏(输出乱码)。详见 `actions/integrate.md`。**Kernel agent / patch_applier sub-agent 都受此约束。**

### IR-7: NEVER modify GEAK configuration

GEAK 是外部服务 —— 视为**只读基础设施**。skill 不能修改 GEAK 任何配置文件、设置或参数(除了作为 `geak_create_task` 参数传入的):

- **不**修改 GEAK server config / workspace 设置 / API 配置
- **不**写入或修改 GEAK config/settings 目录下的任何文件
- **不**在运行时改 `KERNEL_OPT_WORKSPACE` / `GEAK_STEP_LIMIT` 等常量
- **不**修改 GEAK MCP server 配置(`cursor_mcp_config.json` 等)
- **不**修改 GEAK 的测试数据 / 结果 / 配置文件

唯一允许的交互是通过 GEAK MCP tool call:`geak_get_model_config`(只读)/ `geak_create_task` / `geak_submit_task` / `geak_get_task` / `geak_get_outputs` / `geak_download_file` / `geak_list_tasks`。

**永远不要调 `geak_set_model_config` 改模型** —— LLM backend 由管理员预配置。

**例外 — tracing headers**:`kernel-opt` action 开始时,必须调一次 `geak_set_model_config` 注入 observability headers。先跑 `trace_action.py --component geak --action start` 记录时间并生成 config,再通过 MCP apply `extra_headers`。**不**修改 `model_class` / `model_name` / `api_base` / `api_key`。

违反 = run 立即失效。

---

## 5. KERNEL_OPT 常量表(沿用 sprint,single source of truth)

所有 action 引用以下常量。**绝对不在运行时修改**。


| Constant                    | Value                         | 说明                                                                         |
| --------------------------- | ----------------------------- | -------------------------------------------------------------------------- |
| `KERNEL_OPT_BACKENDS`       | `geak,codex`                  | 逗号分隔的激活 backend 列表,可被用户 override(任意组合:`geak` / `codex` / `claude` / `llm`) |
| `OOB_ROUND_ITERATIONS`      | 3                             | Codex/Claude round 数(submit → local benchmark → feedback → re-submit)      |
| `KERNEL_OPT_IMAGE`          | *(CI 或用户提供)*                  | kernel-opt 所有 backend 用的 framework image,每次 run 一个                         |
| `KERNEL_OPT_WORKSPACE`      | `control-plane-moe`           | SaFE workspace(用户可 override)                                               |
| `GEAK_STEP_LIMIT`           | 100                           | 每个 GEAK task 的最大 agent step                                                |
| `GEAK_MAX_RETRIES`          | 3                             | 每个 kernel 的最大 submission 重试                                                |
| `GEAK_MAX_SUBMISSIONS`      | 15                            | 每次 run 的总 GEAK submission 预算                                               |
| `GEAK_TOP_CANDIDATES`       | 5                             | 提交的 top kernel candidate 数                                                 |
| `GEAK_CONSECUTIVE_DISCARDS` | 5                             | 连续这么多 discard 后停止                                                          |
| `GEAK_WALL_CLOCK_MIN`       | 120                           | `kernel-opt` action 的最大 wall-clock minutes                                 |
| `GEAK_POLL_INTERVAL_S`      | 60                            | GEAK task status 轮询间隔(秒)                                                   |
| `GEAK_POLL_TIMEOUT_MIN`     | 15                            | 单个 GEAK task 的最大轮询时间(分钟)                                                   |
| `MIN_GPU_PCT`               | 3                             | 作为 GEAK candidate 的最小 GPU 时间百分比                                            |
| `SERVER_KILL_WAIT_S`        | 10                            | server kill 与 relaunch 之间的等待秒数                                             |
| `FILTERED_TRACE_NAME`       | `filtered-TP-0.trace.json.gz` | TraceLens 分析用的优选 trace 文件                                                  |


**ALWAYS pass `KERNEL_OPT_IMAGE` to all kernel-opt backends**(包括 GEAK + OOB),无论 kernel 类型。对于源码在 image 里的 kernel(如 `/sgl-workspace/aiter/`),pod 用同一个 image。对于运行时生成的 kernel(如 `/tmp/torchinductor_root/` 来自 `torch.compile`),不在 prompt 里包含 `kernel_url` / `kernel_repo`;把文件复制到共享 NFS 或仅依赖 `files[].content`。

适用范围:全模式;Kernel agent 是主消费者,但 `SERVER_KILL_WAIT_S` 等被 Robustness handle / Orchestration 也引用。

---

## 6. Process Management 规则(沿用 sprint+marathon)

- **永远 `export PATH="/opt/venv/bin:$PATH"`**:系统 python3 不带 sglang/vllm/numpy。每个 bash 命令必须先 prepend venv。失败模式:`ModuleNotFoundError: No module named 'sglang'`
- **永远不 `pkill -f "sglang.launch_server"` 在脚本内** —— 会 kill 脚本本身
- **永远等 `SERVER_KILL_WAIT_S` 秒**(默认 10)在 server kill 与 relaunch 之间
- **永远 `unset PROFILE SGLANG_TORCH_PROFILER_DIR`** 在 profiling 完成后
- **永远用 filtered trace** 给 TraceLens(raw 349MB / filtered 5MB)。TraceLens 不支持 `rocprofv3` 格式 —— 只支持 PyTorch Kineto
- **永远不 override 用户指定的 TP**:如果 prompt 说 TP=8 就用 TP=8,不要自动检测 GPU_COUNT 把它降到 TP=1(大模型 120B+ 单 GPU 跑不动)
- **vLLM flags 与 SGLang 不同**:常见错误 `--disable-log-requests` 不是 vLLM 有效 flag,用 `--disable-log-stats`
- **用 `run_baseline.sh` 而不是手动启 server**:脚本处理 server 启动 / health wait / benchmark / profiling 的测试过的序列;手动启会跳过 health check 撞 Exit 144(来自 stale 进程的 SIGTERM)

**适用范围**:全模式强制。

---

## 7. Agent 角色与模型分配

> 这是 v0.6 与 v0.4 / v0.5 最大的语义重组所在。下面 4 个角色全部常驻、全部启用,**没有 mode gating**;角色之间通过 Coordinator 中介通信(无议会、无投票),由 **Critic Review 协议(§18)** 决定方向、由 **Robustness 干预协议(§19)** 兜底救场。

### 7.1 Orchestration agent(原 Orchestration,Claude `claude-opus-4-7`)

> v0.5 名字是 `orchestration`;v0.6 改名 `orchestration` 以贴合架构图 Layer-1 expert 命名。**当前是开发阶段,本分支整体开发时一并完成代码 rename,不单独提 PR**。

#### 职责

- **提议 action**:基于当前 SharedState + Objective + Critic KB 召回 + Robustness 当前告警,选下一步要做的 action,通过 `propose_action` intent 发出
- **委托 sub-agent**:对自己 owns 的 9 个 action(setup / classify / target-analysis / baseline / profile / backends / params / sweep / report),通过 `delegate` intent 让 Coordinator spawn 对应 ActionOrchestration / ephemeral sub-agent
- **REQUEST Kernel agent**:对 5 个 kernel-owned action,通过 `request{target_agent="kernel", kind=...}` 派发,等待 `response`(Plan A,§13)
- **解读结果**:消费 `delegated_result` / `response`,更新 SharedState(`update_state`),写 prediction
- **接受 Critic Review**:Coordinator 把 Critic 的 verdict(approve / reject / redirect / advise)注入下一轮 prompt;Orchestration 必须按 verdict 调整 plan(reject 或 redirect 不能强行执行,详见 §18)
- **写 persona**:`update_persona` append-only 自己的 `personas/orchestration.md`

#### 不能做

- 不能 delegate kernel-owned 的 5 个 action(PolicyGate `kernel_owned_by_kernel_agent` 规则拒绝)
- 不能直接读写 KB(KB 是 Critic 的;Orchestration 通过 prompt 注入消费 KB 召回结果)
- 不能发 `kill_task` / `force_dispatch` / `prune_branch` / `escalate_strategy_change`(robustness-only)
- 不能 `update_state` 改核心字段(`current_best` / `stop_reason` 等,只 Coordinator 写)

#### 工具

- `emit_intent`(MCP)+ `Read` + 按 action 注入的 `Bash` / `Edit`(由 PolicyGate 按 action.allowed_tools 过滤)

### 7.2 Kernel agent(Plan A,Claude `claude-opus-4-7`,responder-only)

#### 职责(独占 5 个 action)

- `kernel-opt`:并行提交 5 个 candidate 给所有激活的 KERNEL_OPT_BACKENDS,按 IR-1 / IR-2
- `integrate`:patch → re-baseline → KEEP/REVERT,按 IR-3 / IR-6
- `deep-kernel-analysis`:从 trace 推 kernel 瓶颈、识别 fusion 候选 / tiling 候选
- `operator-tuning`:针对 GEMM / attention 等 op 的参数化调优
- `vendor-kernel-config`:配 aiter / alter 等 vendor backend 的参数

#### 触发模式(Plan A 关键)

- **只接受 `request{target_agent="kernel"}`** 触发;不主动 `propose_action` / `delegate`
- 处理完后必须发 `response{in_reply_to=<request_msg_id>, kind=<request_kind>, status, result}`
- Critic Review 仍然适用:在执行真正的副作用前(`integrate` 这种带 KEEP/REVERT 决策的),Coordinator 拦截走 §18 流程

#### 工具

- `emit_intent`(MCP)+ `Bash`(调 `geak_ray_submit.py` / `oob_ray_submit.py` / `patch_inductor.py` / `run_baseline.sh`)+ `Read` + `Edit`(写 patch)
- 拥有 server_lifecycle / workspace_mutation / benchmark_lane / profile_lane 全部 lease 申请权

#### 与 Critic / Robustness 交互

- Critic 可对 Kernel 的 `response` 做 review(approve / reject / redirect / advise),由 Coordinator 在写入 `decision` 前拦截
- Robustness 可对长跑 GEAK turn 发 `kill_task` 终止;可对反复失败发 `prune_branch{family="deep_kernel"}`

### 7.3 Critic agent(Codex `gpt-5.4`,no-tools 原则 + KB 例外)

> v0.6 关键扩张:Critic 接管原 Sage 的 KB 责任和 Devil's advocate,并负责 review Orchestration / Kernel 提出的优化建议,但**不开议会**(只发 `objection` 入 event log,不投票)。**Critic 不做 RCA;RCA 明确属于 Robustness。**
> Codex 暂用 `gpt-5.4`(litellm 当前不支持 5.5;5.5 可用后切换)。

#### 职责


| 子职责                     | 说明                                                                                                                                                                           |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Review 闸门**(§18)      | 对 Orchestration / Kernel agent 的优化建议 / 副作用 proposal(`accuracy_risk > 0` 或带方向变更)出 verdict:`approve` / `reject` / `redirect` / `advise`;`redirect` / `advise` 可给更高层方向建议,但不直接调度 |
| **独立预测 + Brier 校准**     | 对每个 proposal 输出 `predicted_gain_pct`,运行结束后跟实际比较,长期校准                                                                                                                         |
| **KB read**             | 用 `kb_query.py` 召回当前 model + action 的历史 entries,注入自己的 review prompt + 注入 Orchestration 下一轮 prompt;warm-start 第 2+ 次同 model_family 才 read                                     |
| **KB write**            | 每个 action 完成后用 `kb_ingest.py` 写一条 entry(model / action / lesson / gain / status / tags)                                                                                      |
| **Cross-run synthesis** | 每 6h 扫中心化 KB 内所有 entries 生成 insights;发现矛盾 flag 到 conflicts 表;入口由 `kb_ingest.py --synthesize` 包裹                                                                              |
| **Devil's advocate**    | 对低风险 proposal 也可主动发 `objection`(意见入 event log,**不触发议会、不投票**),由 Orchestration 在下一轮自决是否采纳;`objection` 仅作信号                                                                     |
| **Persona 维护**          | append-only 写自己的 `personas/critic.md`;每 4h 或 8K token 时自蒸馏                                                                                                                   |


#### Tool Access — Codex no-tools 原则在 KB 这一处有限放开

按 v0.4 §5.1.1,Codex 角色默认不使用 tools。v0.6 因为 KB 是 Critic 的核心责任,做**有限放开**:


| Tool                        | 是否允许                                                                                          | 理由                                                              |
| --------------------------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| `validated_json_output`     | ✓                                                                                             | Critic 的 intent transport                                       |
| `Read`(只读 NFS 路径)           | ✓                                                                                             | 必须能读 `state.json` / `event_log` / `personas/` / `kb/`           |
| `Bash` 限白名单                 | ✓ 仅 `python3 $SKILL_ROOT/kb/kb_query.py ...` 与 `python3 $SKILL_ROOT/kb/kb_ingest.py ...` 两条命令 | KB 读写需要执行脚本                                                     |
| 任何 `Edit` / `git` / 其它 Bash | ✗                                                                                             | 仍 no-tools 范畴,workspace 副作用走 Orchestration / Kernel / sub-agent |


PolicyGate 强制(见 §14.4):Critic 的 Bash allowlist 是**两条精确命令**(可带不同参数),其它一律拒。

#### 不能做

- 不能 `delegate` 任何 action(无 GPU 副作用权)
- 不能 `request`(no agent-to-agent RPC)
- 不能 `propose_action`(只 review 别人的 propose,不自己提议)
- 不能做 RCA;RCA / recovery / handle 全部属于 Robustness
- 不能改 SharedState 核心字段

### 7.4 Robustness agent(原 Triage,Claude `claude-opus-4-7`,always-on)

> v0.6 改名:`triage` → `robustness`,贴合架构图 cross-layer 命名;职责保持不变(Robustness monitor + RootCauseAnalysis + Handle 三合一)。代码层 rename 与 orchestration → orchestration 一并在本分支整体开发时完成,不单独提 PR。
> v0.6 正式确立:Robustness = **Robustness monitor + RootCauseAnalysis + Handle** 三合一。
> 不再有 v0.4 "marathon 才常驻 Robustness monitor,guided emergency 另走临时 RCA" 的复杂分支。Robustness always-on,所有 RCA / recovery / handle 均归 Robustness;Critic 不做 RCA。

#### 职责矩阵


| 子角色          | 子职责                                                                                                                                                                                                                                                                                                            | 触发                                                                                                 |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **Robustness monitor** | 监控 event_log 实时流;crash 信号检测;agent stall 检测;health check(server pid / port / GPU memory / lease holder); 写 `findings/alerts.jsonl`                                                                                                                                                                              | always-on tick(60s 默认) + 事件驱动                                                                      |
| **RCA**      | 深度根因分析:读 event_log tail + state snapshot + 最近 KEEP/REVERT;输出 RCAFinding 写 `findings/<ts>.json`;调起 `rca_runner` ephemeral sub-agent 跑深度分析                                                                                                                                                                       | crash_count >= 2 / repeated KEEP/REVERT bouncing                                                   |
| **Handle**   | server lifecycle 管理(kill_server / start / restart / health-wait,遵守 IR-4 / IR-5 / SERVER_KILL_WAIT_S);accuracy gate exec(`accuracy_risk > 0` 的 action 跑完后,Robustness 调起 `eval_runner` 跑 GSM8K,`compare_to_baseline`,FAIL 时通知 Coordinator `revert`);recovery(从 checkpoint / evidence-check 矩阵出发的 `recover` action) | 由 Orchestration / Kernel `delegate(server_lifecycle / accuracy_gate / recover)` 触发,或 Robustness 自发 |


#### 调度警察 4 intent(robustness-only,PolicyGate 严格限源)


| Intent                     | Payload                                        | 用途                                                                                                                                                       |
| -------------------------- | ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kill_task`                | `{task_id, reason, force?, scope="task"}`      | 终止 queued / running 的 delegate task;**scope 强制 "task"**,server / process kill 仍走 IR-5                                                                    |
| `force_dispatch`           | `{task_id, reason}`                            | 把 queued task 提到 dispatcher 队首;用于"高价值 bench 卡在低价值队列后"                                                                                                    |
| `prune_branch`             | `{family, reason}`                             | 砍掉某 family 的所有 queued task + 加 `state.pruned_families`,scheduler 下次 score 这个 family 时返回 0;用于"3+ 次同 family 失败"                                            |
| `escalate_strategy_change` | `{reason, next_action_hint, severity?="high"}` | 发 priority-0 alert(`kind=strategy_change`),Orchestration 下一 tick 必读;**非破坏性建议,不直接改 scheduler/prune/dispatch 状态**;用于"trace 显示完全不同的瓶颈,建议 Orchestration 换方向" |


PolicyGate 的 `ROBUSTNESS_ONLY_SOURCE_ALLOWLIST = {"robustness"}` 强制其它 agent 不能发上述 4 个 intent,加上 `KILL_TASK_ALLOWED_SCOPES = {"task"}` 双重保险。

#### 工具

- `emit_intent`(MCP)+ `Read`(任意 NFS 路径)+ `Bash` 受限 allowlist(read-only:`pgrep` / `ps` / `nvidia-smi` / `rocm-smi` / `df` / `du` / `ls` / `cat` / `head` / `tail` + server lifecycle 命令受 IR-4 / IR-5 约束)
- 不能 `Edit` 任何文件(workspace 副作用走 sub-agent)

#### 不能做

- 不能 `propose_action`(那是 Orchestration 的事;Robustness 通过 `escalate_strategy_change` 表达"换方向"建议)
- 不能 `delegate` kernel-owned action
- 不能改 SharedState 核心字段(`current_best` / `stop_reason` 等)

### 7.5 Coordinator(Python,无 LLM)

#### 职责

- 主循环 + asyncio reactor 调度
- MessageBus + SharedState + ResourceLockManager + TaskRegistry + PolicyGate + Scheduler + Checkpoint/Resume
- **REQUEST/RESPONSE 路由**(Plan A):Orchestration 发 `request` → Coordinator 投递到 Kernel agent inbox;Kernel 发 `response` → Coordinator 回投到 Orchestration inbox + 触发 Critic Review
- **Critic Review 闸门**(§18):副作用 proposal 必经 Critic verdict 才能进入执行;`reject` 拦截、`redirect` 替换、`advise` 注入下一轮 prompt、`approve` 放行
- **Robustness 干预执行**:`force_dispatch` 改 task `created_at`、`prune_branch` 写 `state.pruned_families` + `cancel_tasks_of_family`、`kill_task` 调 task lifecycle;`escalate_strategy_change` 只广播 priority-0 非破坏性建议,不直接改调度状态
- **早停 + checkpoint cadence**:见 §9 / §17
- **不做决策**:Coordinator 不替任何 LLM agent 做 action 选择 / verdict;它只是协议管理员

### 7.6 角色 × Intent 能力矩阵(PolicyGate 强制)


| intent_type                | Orchestration | Kernel | Critic | Robustness | 备注                                                                                                         |
| -------------------------- | ------------- | ------ | ------ | ---------- | ---------------------------------------------------------------------------------------------------------- |
| `propose_action`           | ✓             | ✗      | ✗      | ✗          | Critic 只 review 不提议;Robustness 用 escalate_strategy_change 表达                                               |
| `delegate`                 | ✓※1           | ✗      | ✗      | ✓※2        | ※1 不能 delegate kernel-owned 5 个;※2 仅可 delegate accuracy_gate / recover / server_lifecycle 这类 handle action |
| `request`                  | ✓※3           | ✗      | ✗      | ✗          | ※3 target_agent 限 `kernel`(REQUEST_ROUTING 表)                                                              |
| `response`                 | ✗             | ✓※4    | ✗      | ✗          | ※4 必带 `in_reply_to`                                                                                        |
| `update_state`             | ✓             | ✓※5    | ✗      | ✓          | ※5 仅写自己 action 产出的 metric 字段;CORE_STATE_FIELDS 只 Coordinator 写                                               |
| `update_persona`           | ✓             | ✓      | ✓      | ✓          | append-only 写自己的 `personas/<name>.md`                                                                      |
| `send_message`             | ✓             | ✓      | ✓      | ✓          | 任意 topic;不在白名单的软降级为 `observation`                                                                          |
| `ask_question`             | ✓             | ✓      | ✓      | ✓          | 通常给 Critic(KB 查询)                                                                                          |
| `answer`                   | ✓             | ✓      | ✓      | ✓          | 必须带 `in_reply_to`                                                                                          |
| `alert`                    | ✓             | ✓      | ✓      | ✓          | severity=high → priority=0;镜像到 `findings/alerts.jsonl`                                                     |
| `**review_verdict`**(§18)  | ✗             | ✗      | ✓※6    | ✗          | ※6 critic-only;`{target_proposal_msg_id, verdict, reasoning, kb_evidence?}`                                |
| `kill_task`                | ✗             | ✗      | ✗      | ✓※7        | ※7 scope="task" 强制                                                                                         |
| `force_dispatch`           | ✗             | ✗      | ✗      | ✓          | robustness-only                                                                                            |
| `prune_branch`             | ✗             | ✗      | ✗      | ✓          | robustness-only                                                                                            |
| `escalate_strategy_change` | ✗             | ✗      | ✗      | ✓          | robustness-only;触发 priority-0 broadcast;非破坏性建议,不直接改 scheduler/prune/dispatch 状态                            |
| ~~`objection`~~            | ✗             | ✗      | ✓      | ✗          | v0.6 改造:不触发议会、不投票,仅作信号入 event log;由 Orchestration 自决是否采纳;**保留为 Critic 的 Devil's advocate 出口**              |
| ~~`vote`~~                 | —             | —      | —      | —          | **删除**(无议会)                                                                                                |
| ~~`parliament_open`~~      | —             | —      | —      | —          | **删除**                                                                                                     |
| ~~`vote_request`~~         | —             | —      | —      | —          | **删除**                                                                                                     |


**Plan A 关键约束**(同 v0.5):

- 只有 `orchestration → kernel` 这对 (source, target) 允许 `request`(`REQUEST_ROUTING` 表)
- 只有 `kernel` 角色可发 `response`
- `orchestration.delegate(action_name in {kernel_opt, integrate, deep_kernel_analysis, operator_tuning, vendor_kernel_config})` 被拒(`rule="kernel_owned_by_kernel_agent"`)

### 7.7 Framework / Comm agent(架构图占位,本期不实现)

按 Hyperloom 优化栈完整命名,4 个 layer experts 应包括 Orchestration / Framework / Kernel / Comm,加 2 个 cross-layer 的 Critic / Robustness,共 6 个 agent。本期(v0.6)实现其中 4 个:Orchestration + Kernel(2 个 layer experts) + Critic + Robustness(2 个 cross-layer)。Framework agent(框架层专家,如 vLLM / SGLang 框架级调优)与 Comm agent(通信层专家,如 RCCL / NCCL 通信调优)在架构图中**占位但本期不开发**。

未来引入这两个 agent 时:

- 通过 PolicyGate 新增 `framework` / `comm` 的 source allowlist 与 intent 权限矩阵条目
- 通过 `REQUEST_ROUTING` 表新增 `orchestration → framework` / `orchestration → comm` 等合法 (source, target) 对
- 通过 `KERNEL_OWNED`_* 风格的 `FRAMEWORK_OWNED_ACTIONS` / `COMM_OWNED_ACTIONS` 划分 action ownership
- 不需要重构既有 4 agent 协议(SQLite events / Critic Review / Robustness 干预 / 资源锁 4 lane 全部沿用)

落地节奏跟 ADR-40 对齐:**v0.6 不做,纳入 v0.7+ 候选清单(详见 §26 TODO T?? 后续追加)**。

---

## 8. 4 层记忆模型 + KB(由 Critic 主导)

### 8.1 4 层定义


| 层                     | 内容                                        | 载体                                                                                         | 生命周期                     | 维护者                                                                                                                           | v0.6 启用     |
| --------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------ | ----------------------------------------------------------------------------------------------------------------------------- | ----------- |
| **L1 即时上下文**          | 当前思考链 + 工具结果                              | 单次 `backend.run()` 内 LLM context                                                           | run 内                    | LLM 自然                                                                                                                        | ✓           |
| **L2 本 session 工作记忆** | events / cursors / tasks / leases / state | SQLite WAL `storage/coordinator.db`(events / cursors / tasks / leases 4 表)+ NFS `state.json` | session(含 resume)        | Coordinator                                                                                                                     | ✓           |
| **L3 本 run 人格**       | "我是 X,对此 run 的累积观点"                       | NFS `personas/<agent>.md`                                                                  | run 持久;蒸馏每 4h 或 8K token | agent 自主 + 自动蒸馏                                                                                                               | ✓ 全 4 角色    |
| **L4 跨 run 长期记忆**     | "这模型以前优化 N 次,结论 ..."                      | **中心化共享 KB**(按 `<model_family>/<model_name>` 分区,具体载体 T13 后续落地:NFS 共享路径或 HTTP service)      | 永久,跨 session、跨 user      | **Critic 主导**(唯一 read/write/synthesis 入口,通过 `kb_query.py` / `kb_ingest.py` 包裹);其它 agent 不直接读写,只通过 prompt 注入消费;**P0 不依赖真实 KB** | P1+;P0 mock |


### 8.2 L4 KB 操作(中心化共享 + Critic owns,非 P0)

> **P0 说明**:中心化 KB 先不做,由 Critic owner 后续实现。P0 的 Critic mock 可以返回空 KB hint / 固定 mock evidence,不得阻塞 Coordinator + Orchestration + KernelAgent 主链路。
>
> **目标存储形态**:KB 是**中心化共享存储**,对所有 sandbox / session / user 可见;按 `<model_family>/<model_name>` 二级分区(如 `kb/deepseek/DeepSeek-R1-0528/entries.jsonl`、`kb/qwen/Qwen3-8B/entries.jsonl`)。具体载体由 T13 落地,候选两种:
>
> - **Option A — NFS 共享路径**:`/wekafs/kb/<model_family>/<model_name>/entries.jsonl`,append-only + 简单文件锁
> - **Option B — HTTP KB service**:Critic 通过 REST API 访问,服务侧统一并发控制 + embedding-based recall
>
> 两种载体下,Critic 都是**唯一 read/write/synthesis 入口**,通过 `kb_query.py` / `kb_ingest.py` 两个脚本包裹具体后端;其它 agent(Orchestration / Kernel / Robustness)**不直接读写 KB**,只通过 Coordinator 在 prompt 里注入 KB 召回结果消费(`=== Critic KB hint ===` 段)。这样写入并发只由 Critic 一家控制,简化竞争。

#### Read(warm-start,所有 agent prompt 都注入)

每个 action 执行前由 Critic 召回:

```bash
python3 $SKILL_ROOT/kb/kb_query.py "$MODEL_NAME $ACTION_NAME" --top-k 5 --compact
```

**warm-start 条件**:第 2+ 次同 model_family 才 read;第 1 次只 write 防 cold-start 单次坏经验污染。判定逻辑:`kb.count_entries(model_family) >= 1`。

召回结果有两个去向:

1. 注入 Critic 自己的 review prompt(让 verdict 带历史依据)
2. 注入 Orchestration / Kernel 下一轮 prompt 的 `=== KB hint ===` 段(让提议带历史依据)

#### Write(每个 action 完成后,由 Critic ingest)

```bash
python3 $SKILL_ROOT/kb/kb_ingest.py \
    --category $CATEGORY --model "$MODEL_NAME" \
    --action "$WHAT_WAS_DONE" --lesson "$KEY_TAKEAWAY" \
    --tags $TAGS --gain $GAIN --status $STATUS
```

触发时机:Orchestration / Kernel 的 `delegated_result` 或 `response` 进入 Coordinator 后,Coordinator 通知 Critic;Critic 在下一轮 reactor tick 写入。

#### Cross-run synthesis(每 6h,Critic 自发)

- 扫中心化 KB 所有 entries,提炼跨 model 共性 / 跨 action 共性,作为 insight 入库
- 检测矛盾(同 model + 同 action 不同 lesson)入 conflicts 表,等下一次同 model 跑时 Critic review 决定保留哪条
- 用 `validated_json_output` 输出 `kb_synthesis_done` event

### 8.3 每次 `backend.run()` 的 prompt 拼接(以 Orchestration 为例)

```python
def compose_prompt(agent_name, msgs, state, objective, kb_recall, latest_critic_verdict, latest_robustness_alert):
    return f"""
=== Your persona (L3, accumulated this run) ===
{kb.read_persona(agent_name)}

=== Critic KB hint (L4, by Critic warm-start) ===
{kb_recall or "(KB not warm for this model family)"}

=== Latest Critic verdict on your last proposal (§18) ===
{latest_critic_verdict or "(no recent verdict)"}

=== Latest Robustness alert (if any) ===
{latest_robustness_alert or "(robustness all clear)"}

=== Objective (统一抽象, §11) ===
type:           {objective.kind}              # gain_pct / tput / baseline / time_only
progress:       {objective.progress(state):.1%}
remaining_gap:  {objective.remaining_gap(state)}
pressure:       {scheduler.pressure(state):.2f}
time_left:      {state.time_left_min} min

=== Pruned families (Robustness prune_branch) ===
{state.pruned_families or "(none)"}

=== Current session state (L2) ===
{state.summary()}

=== Active resource locks ===
{lock_mgr.summary()}

=== Messages for you to respond to (L2) ===
{format_messages(msgs)}

=== Your task ===
Respond through your configured Intent Transport (§14).
- Claude roles: call the `emit_intent` MCP tool.
- Codex roles: return validated JSON only.
Free text is ignored.
"""
```

Critic / Robustness / Kernel 的 prompt 拼接结构同上,只在 KB 段、verdict 段、tools 段做角色特化。

---

## 9. 早停机制

### 9.1 5 个早停信号(OR 关系)

```python
def should_stop_early(state, objective, scheduler, brier_window, lock_summary, history) -> StopReason | None:
    # 1. 效果到达(核心)
    if objective.reached(state):
        return StopReason("target_reached", severity="success")

    # 2. 时间到
    if state.time_left_minutes <= 5:
        return StopReason("time_exhausted", severity="warning")

    # 3. 无优化空间
    if scheduler.no_more_leverage(state, history):
        return StopReason("no_more_leverage", severity="warning")

    # 4. critic 长期不确定(边际收益低)
    if critic_brier_plateau(brier_window):
        return StopReason("brier_plateau", severity="info")

    # 5. 紧急
    if state.crash_count >= 2:
        return StopReason("emergency", severity="critical")

    return None
```

> v0.6 与 v0.4 不同:由于 Critic 全启用,`brier_plateau` 信号不再有 mode gating。

### 9.2 按 reason 分级的尾流


| Stop Reason        | Severity | Sweep      | Report             | KB Synthesis   | Final Checkpoint | RCA                                                                                                      |
| ------------------ | -------- | ---------- | ------------------ | -------------- | ---------------- | -------------------------------------------------------------------------------------------------------- |
| `target_reached`   | success  | ✓ 完整 sweep | ✓ 完整报告             | ✓ Critic 充分时间  | ✓                | —                                                                                                        |
| `no_more_leverage` | warning  | ✓ 完整 sweep | ✓ 完整报告             | ✓ Critic ≤300s | ✓                | —                                                                                                        |
| `time_exhausted`   | warning  | ✗ 跳过       | ✓ Fast report(仅汇总) | ✓ Critic ≤120s | ✓                | —                                                                                                        |
| `brier_plateau`    | info     | ✓ 完整 sweep | ✓ 完整报告             | ✓ Critic ≤300s | ✓                | —                                                                                                        |
| `emergency`        | critical | ✗ 禁止       | ✓ Crash report     | ✗ 跳过           | ✓ 紧急             | **Robustness 全力 RCA;Robustness 不可用时记录 robustness_unavailable,等待 Robustness owner / 人工处理;Critic 不接管 RCA** |


```python
async def graceful_stop(self, reason: StopReason):
    self.state.set_stopping(reason)

    if reason.name == "emergency":
        await self.checkpoint(emergency=True)
        # Robustness 永远 always-on,所以走 robustness RCA;失败不兜底 Critic,避免角色越界
        try:
            await self.bus.send("robustness", {"topic": "do_emergency_rca", "priority": 3})
            await self.bus.wait("robustness", topic="rca_done", timeout=180)
        except (RobustnessUnavailable, TimeoutError):
            self.state.attach_rca({"status": "robustness_unavailable", "needs_manual_review": True})
        await self.write_crash_report()
        return

    if reason.name == "time_exhausted":
        await self.write_fast_report()
        await self.bus.send("critic", {"topic": "synthesize_for_kb", "priority": 2})
        await self.bus.wait("critic", topic="kb_synthesis_done", timeout=120)
        await self.checkpoint()
        return

    # success / warning / info: 跑完整尾流
    await self.run_action("sweep")
    await self.run_action("report")
    await self.bus.send("critic", {"topic": "synthesize_for_kb", "priority": 2})
    timeout = None if reason.name == "target_reached" else 300
    await self.bus.wait("critic", topic="kb_synthesis_done", timeout=timeout)
    await self.checkpoint()
```

---

## 10. Accuracy Gate 协议(沿用 sprint,所有 `accuracy_risk > 0` 必跑)

> v0.6 由 **Robustness handle 子角色**调起 `eval_runner` 跑;FAIL 时通知 Coordinator,Coordinator 把 task 标 `needs_revert` 并通知 Orchestration / Kernel。

### 10.1 哪些 action 触发 gate


| accuracy_risk | Actions                                                                                 | Gate required |
| ------------- | --------------------------------------------------------------------------------------- | ------------- |
| 0.0           | server scheduling 参数(decode-steps / cuda-graph-max-bs / mem-fraction / chunked-prefill) | ✗             |
| 0.05–0.15     | kernel 修改(GEAK)/ GEMM tuning                                                            | ✓             |
| 0.10          | backend 切换(aiter / alter / attention backends)                                          | ✓             |
| 0.30          | 精度相关参数(kv-cache-dtype fp8 / 量化变更)                                                       | ✓             |


### 10.2 Gate 流程(Robustness handle 主导)

对任何 `accuracy_risk > 0` 的 action,throughput benchmark 成功后:

1. Robustness 调起 `eval_runner` ephemeral sub-agent(占 benchmark_lane):
  ```bash
   EVAL_TASK=gsm8k NUM_FEWSHOT=5 PORT=$PORT MODEL=$MODEL \
     RESULTS_DIR="$RESULT_DIR/eval_gsm8k_${ACTION_NAME}" \
     bash "$SKILL_ROOT/scripts/eval_accuracy.sh"
  ```
2. 提取 score:
  ```bash
   new_accuracy=$(python3 -c "import json,glob; ...; print(scores.get('exact_match,strict-match', 0))")
  ```
3. 跟 baseline 比较:
  ```
   accuracy_drop = baseline_accuracy - new_accuracy
   if accuracy_drop > 0.01:
       通知 Coordinator: task → needs_revert
       Critic ingest KB: accuracy_risk=1.0 for this action+model
       Orchestration / Kernel 收到 revert request,执行 revert
   else:
       KEEP — accuracy within tolerance
  ```

### 10.3 Kernel-level 预检(可选)

在完整 GSM8K eval 前,Kernel agent 可做 micro-benchmark sanity check:`assert torch.allclose(original_output, optimized_output, atol=1e-3, rtol=1e-3)`。**不替代** GSM8K gate,只是 early-exit。

### 10.4 跳过 gate 的 action

`setup` / `classify` / `profile` / `sweep` / `report` 是只读;纯 scheduling 参数(`accuracy_risk = 0.0`)也跳过。

### 10.5 适用范围

全模式强制(无 mode gating);只看 `action.accuracy_risk` 字段。

---

## 11. Objective 抽象(沿用 v0.4)

### 11.1 输入语义

`MAX_HOURS` 必填(预算 + 调度 pressure + 早停时钟);`TARGET_`* 可选(用于 Objective 的"reached")。

- `TARGET_GAIN_PCT=30`(基于 baseline 的相对增益)
- `TARGET_TPUT_PER_GPU=700`(绝对吞吐目标)
- `TARGET_DIR=/path/to/B200`(对标某个外部 baseline 目录)
- 都不给(仅 `MAX_HOURS=N`,`TimeOnlyObjective`)

最多同时指定一个 TARGET_*。

### 11.2 抽象接口 + 4 个具体实现

```python
class Objective(ABC):
    @abstractmethod
    def kind(self) -> str: ...                            # gain_pct | tput | baseline | time_only
    @abstractmethod
    def progress(self, state) -> float: ...               # 0.0 ~ 1.0
    @abstractmethod
    def remaining_gap(self, state) -> float: ...
    @abstractmethod
    def reached(self, state) -> bool: ...
    @abstractmethod
    def pressure_input(self, state) -> float: ...         # 喂给 scheduler.pressure()
    @abstractmethod
    def describe(self) -> str: ...                        # 喂 prompt
```

4 个实现(`TargetGainObjective` / `TargetTputObjective` / `TargetBaselineObjective` / `TimeOnlyObjective`)与 v0.4 一致。

### 11.3 工厂

```python
def build_objective(env: dict) -> Objective:
    validate_required(env, ["MODEL_PATH", "MAX_HOURS"])
    validate_positive_float(env["MAX_HOURS"])
    validate_at_most_one(env, ["TARGET_GAIN_PCT", "TARGET_TPUT_PER_GPU", "TARGET_DIR"])
    if "TARGET_GAIN_PCT" in env: return TargetGainObjective(float(env["TARGET_GAIN_PCT"]))
    if "TARGET_TPUT_PER_GPU" in env: return TargetTputObjective(float(env["TARGET_TPUT_PER_GPU"]))
    if "TARGET_DIR" in env: return TargetBaselineObjective(env["TARGET_DIR"])
    return TimeOnlyObjective()
```

---

## 12. Budget-Aware 调度器

### 12.1 评分公式(v0.6 去掉 mode_gate,加 prune_gate)

```python
score = base × pressure × prune_gate × depth_gate × diminishing × lane_available × prior × adjustment
```


| 因子               | 公式                                                           | 作用                                                 |
| ---------------- | ------------------------------------------------------------ | -------------------------------------------------- |
| `base`           | `(expected_gain / cost_p75) × (1-acc_risk) × (1-crash_risk)` | 基础启发式                                              |
| `pressure`       | `max(objective.pressure_input(state), 1 - time_left/total)`  | 离目标远 + 时间少 → 倾向高 risk 高 gain                       |
| `prune_gate`     | `0 if action.family in state.pruned_families else 1`         | **新增**:Robustness `prune_branch` 后调度器永远不选这个 family |
| `depth_gate`     | `1 if cost_p75 ≤ time_left × 0.8 else 0`                     | 防"半截饭"                                             |
| `diminishing`    | `0.7 ** count(completed, family=action.family)`              | 防 DFS 一棵树吊死                                        |
| `lane_available` | `1 if all required lanes free else 0`                        | 资源锁过滤                                              |
| `prior`          | model_class × action 的初始分(§12.2)                             | 冷启动启发                                              |
| `adjustment`     | 7 条 update rule 累加(§12.3)                                    | 在线学习                                               |


> **没有 `mode_gate`**:v0.6 不再有 mode 概念,所有 action 默认可选;只靠 `applicable_when` 谓词 + Robustness `prune_branch` + Critic Review 自然过滤。

### 12.2 Initial Score Priors(沿用 sprint)


| Action           | Dense | MoE+MLA | MoE+SWA | MoE+MLA+NSA |
| ---------------- | ----- | ------- | ------- | ----------- |
| backends         | 3     | **9**   | **8**   | **10**      |
| params           | 5     | 6       | 7       | 5           |
| kernel-opt(GEAK) | **8** | 2       | 2       | 2           |
| torch.compile    | **7** | 0       | 0       | 0           |
| sweep            | 1     | 1       | 1       | 1           |


### 12.3 Score Update Rules(沿用 sprint)

每个 action 完成后:

1. **Action succeeded (gain > 0%)**: 同 family 其它 action × 1.5(cap 3.0);push `combined_test`
2. **Action failed (gain ≤ 0%)**: 同 family 其它 action × 0.5
3. **2+ backend wins**: push `combined_backends_test` score = sum × 1.5
4. **All backends tested**: push `re-profile`(发现新 GEAK 目标)
5. **Kernel opt kept**: push `re-profile + next-kernel` × boost
6. **Kernel opt discarded**: 剩余 kernel scores × 0.7
7. **All scores < 1.0**: proceed to sweep → report(进入 §9.2 success 尾流)

### 12.4 Brier 加权(数据成熟后启用)

```
默认: Critic 等权重
启用后: Critic 维护历史 Brier score (prediction calibration)
        Brier 低(更准)的 Critic 在 review verdict 里权重高
        — 但 v0.6 单 Critic 单 verdict,Brier 主要影响 Coordinator 是否启用
        sample-down(对低风险 proposal 降级为 sampled review)
```

---

## 13. A2A 通信协议

### 13.1 Message Envelope(SQLite events 表 + asyncio.Queue 投递缓存)

`events` 表是 A2A Bus 的 source of truth;`asyncio.Queue` 仅运行时投递缓存。所有 message 必须先 `INSERT INTO events` 拿到自增 `seq`,再投递到内存 queue。Resume 时按 cursor 从 events 表重放。

**seq 分配语义**:`events.seq INTEGER PRIMARY KEY AUTOINCREMENT` 由 SQLite 内部串行化,无需额外 lock。

```python
@dataclass
class Message:
    id: str               # uuid (idempotency key)
    from_agent: str
    to: str | list[str] | "*"
    topic: str            # 见白名单
    in_reply_to: str | None
    payload: dict
    priority: int         # 0=低 1=中 2=高 3=紧急
    timestamp: datetime
    seq: int              # 全局递增 sequence number, 用于 cursor 推进
```

### 13.2 Topic 白名单(v0.6 删除议会相关)

**保留**:
`proposal` / `review_verdict` / `request` / `response` / `question` / `answer` / `observation` / `event` / `decision` / `alert` / `historical_warning` / `reflection_tick` / `do_postmortem` / `do_strategic_review` / `do_emergency_rca` / `synthesize_for_kb` / `kb_synthesis_done` / `graceful_stop` / `heartbeat` / `delegated_result` / `intent_emitted` / `rca_done` / `kill` / `force_dispatch` / `prune_branch` / `strategy_change` / `policy_denied`

**删除**(v0.6,无议会):

- ~~`vote`~~
- ~~`vote_request`~~
- ~~`parliament_open`~~
- ~~`objection`~~ → 软降级:Critic 仍可发 `objection` intent,但 Coordinator 把它转成 `topic="advice"` 的 `send_message` 进 event log,不触发任何议会流程

### 13.3 协议规则

1. 任何 message 必须有 `topic` + `priority` + `id`
2. `proposal` / `review_verdict` / `decision` / `request` / `response` 必须带 `payload.reasoning`
3. `question` 60s 内必须有 `answer`
4. Rate limit:每 agent 10 msg/min(Critic 20,因 KB 召回 + review)
5. `priority>=2` 立即触发,<2 batch
6. 接收方按 `id` 去重(幂等,通过 cursor 文件中的 `last_processed_msg_id`)
7. **Critic Review 闸门**(§18):副作用 proposal 必须在 Coordinator 收到 `review_verdict` 后才能进入 task dispatch

### 13.4 4 种协作模式(v0.6,无议会)


| 模式                               | 描述                                                                                                          | 启用          |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------- | ----------- |
| **委托(Delegate)**                 | Orchestration / Robustness → SubAgentRunner → ActionOrchestration / ephemeral sub-agent                          | ✓ 全启用       |
| **流水线(Pipeline)**                | 多 sub-agent 串行编排(profile → kernel-opt → integrate)                                                          | ✓ 全启用       |
| **事件驱动(Alert)**                  | Robustness 监听 event_log → emit alert / kill_task / force_dispatch / prune_branch / escalate_strategy_change | ✓ always-on |
| **RPC(REQUEST/RESPONSE,Plan A)** | Orchestration → Kernel agent;Coordinator 路由 + Critic Review 闸门                                                | ✓ 全启用       |
| ~~**议会(Parliament)**~~           | ~~多 agent 投票~~                                                                                              | **删除**      |


---

## 14. 结构化 Intent Transport

### 14.1 IntentEnvelope schema

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
                            "request", "response",
                            "review_verdict",                       # NEW v0.6
                            "update_state", "update_persona",
                            "answer", "ask_question", "alert",
                            "kill_task",
                            "force_dispatch", "prune_branch",
                            "escalate_strategy_change",
                            # 保留作 Critic Devil's advocate 出口,但不再触发议会
                            "objection",
                        ]
                    },
                    "payload": {"type": "object"}
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

### 14.2 Claude transport:`emit_intent` MCP tool_call

- Orchestration / Kernel / Robustness 都用 Claude,通过 MCP 注册的 `emit_intent` 工具发 intent
- `mcp__inference_optimizer__emit_intent` 是唯一的 transport tool,自由文本被忽略
- Tool input_schema 与 `INTENT_ENVELOPE_SCHEMA` 等价

### 14.3 Codex transport:`validated_json_output`

Critic 的 system prompt 强制:

```text
You have limited tools (Read + 2 Bash commands for KB only).
Return exactly one JSON object matching INTENT_ENVELOPE_SCHEMA.
Do not include markdown, prose, code fences outside the validated_json_output fence,
or explanations outside JSON.
```

Coordinator 对 Critic 输出:

1. 提取完整 JSON object(支持 fenced `validated_json_output` / fenced `json` / 裸 JSON 三种)
2. 用 `INTENT_ENVELOPE_SCHEMA` 做 runtime validation
3. 校验 `intent_type` 对应 payload 子 schema
4. 校验角色权限(§7.6 矩阵)
5. 失败时最多发一次 repair prompt;仍失败记录 `protocol_error`,转 Robustness RCA(若 Robustness 不可用,记 alert 不阻塞主循环)

### 14.4 Coordinator 解析侧

```python
def parse_intents(trajectory) -> list[Intent]:
    intents = []
    for msg in trajectory:
        # Claude SDK: ToolUseBlock(name=qualified MCP tool name)
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if hasattr(block, "name") and block.name == EMIT_INTENT_TOOL_QUALIFIED:
                    intents.extend(Intent.from_envelope(block.input))
        # Codex: 取最后一段 fenced validated_json_output
        elif isinstance(msg, dict) and msg.get("type") == "item.completed":
            item = msg.get("item", {})
            if item.get("type") == "message" and item.get("role") == "assistant":
                envelope = parse_validated_json_output(item.get("content", ""))
                intents.extend(Intent.from_envelope(envelope))
    if not intents:
        raise NoIntentEmitted()
    return intents
```

### 14.5 PolicyGate(v0.6 完整规则)

```python
class PolicyGate:
    def validate_intent(self, from_agent, intent, state):
        role = self.role_registry[from_agent]
        # Layer 1: 角色 allowed_intents 矩阵(§7.6)
        self._validate_role_permission(role, intent)
        # Layer 2: per-intent specific
        if intent.type == IntentType.DELEGATE:
            action = self.action_registry.get(intent.task_kind)
            self._validate_kernel_owned(role, action)         # Plan A
            self._validate_side_effect_policy(action, state)
            self._validate_bash_allowlist_for_action(role, action)
        elif intent.type == IntentType.REQUEST:
            self._validate_request_routing(role, intent.payload)   # 限 orchestration→kernel
        elif intent.type == IntentType.RESPONSE:
            if role.name != "kernel":
                raise PolicyDenied("only kernel may emit response")
        elif intent.type == IntentType.UPDATE_STATE:
            self._validate_state_transition(role, intent.payload, state)
        elif intent.type in ROBUSTNESS_ONLY_INTENTS:               # kill_task / force_dispatch / prune_branch / escalate_strategy_change
            if role.name != "robustness":
                raise PolicyDenied(f"{intent.type.value} is robustness-only")
            self._validate_robustness_only_payload(intent)
        elif intent.type == IntentType.REVIEW_VERDICT:         # NEW v0.6
            if role.name != "critic":
                raise PolicyDenied("review_verdict is critic-only")
            self._validate_review_verdict_payload(intent)
        elif intent.type == IntentType.OBJECTION:
            if role.name != "critic":
                raise PolicyDenied("objection is critic-only (devil's advocate)")
            # 不触发议会,仅作信号

    def allowed_tools_for_agent(self, agent_name, action_or_none):
        role = self.role_registry[agent_name]
        if role.name == "critic":
            # Codex no-tools 原则 + KB 例外(§7.3)
            return ["Read", "Bash(kb_query.py|kb_ingest.py)"]
        if role.name == "robustness":
            # Robustness monitor 受限 Bash + Read + emit_intent
            return ["Read", "Bash(robustness_allowlist)", "emit_intent"]
        if role.name == "kernel":
            return ["Read", "Bash(kernel_allowlist)", "Edit", "emit_intent"]
        # orchestration:基础 + 按 action 注入
        return ["Read", "emit_intent"] + action_specific_tools(action_or_none)
```

#### Bash allowlist(全模式硬规则)


| 类别                                                          | Allow 条目                                                                                                                                                                                                         | 说明                                       |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| **read-only**(全 agent 可)                                    | `ls / cat / head / tail / pgrep / ps / nvidia-smi / rocm-smi / df / du / which / env / printenv`                                                                                                                 | 信息收集                                     |
| **server lifecycle**(robustness / kernel / orchestration 可) | `kill $(pgrep -f 'python.*-m sglang.launch_server')` / `kill $(pgrep -f 'python.*-m vllm.entrypoints')` / `python -m sglang.launch_server ...` / `vllm serve ...` / `bash $SKILL_ROOT/scripts/run_baseline.sh` 等 | 必须遵守 IR-4 / IR-5 / SERVER_KILL_WAIT_S    |
| **KB**(Critic only)                                         | `python3 $SKILL_ROOT/kb/kb_query.py ...` / `python3 $SKILL_ROOT/kb/kb_ingest.py ...`                                                                                                                             | 两条精确命令                                   |
| **kernel-opt**(kernel agent only)                           | `python3 $SKILL_ROOT/scripts/geak_ray_submit.py ...` / `python3 $SKILL_ROOT/scripts/oob_ray_submit.py ...` / `python3 $SKILL_ROOT/scripts/patch_inductor.py --target-file ...` 等                                 | IR-6 必须 `--target-file`,不能 `--cache-dir` |


#### Bash denylist(全模式硬规则,任何 agent 都禁)

- `pkill -f sglang` / `pkill -f vllm`(IR-5)
- `git commit` / `git push` / `git reset --hard`
- `patch`  / `git apply`(只能通过 `patch_inductor.py` 走 integrate action)
- `geak_set_model_config` 改 model(IR-7,只允许 tracing headers)
- `pip install` / `python setup.py` / `make`  / `cmake`  / `ninja`(framework 类构建,本来就是非目标)
- `rm -rf`
- `sudo`

---

## 15. Sub-agent 委托 + ActionOrchestration

### 15.1 决策:不依赖 Claw runSubagent(同 v0.4 ADR-14)

Coordinator 直接 spawn OOB `ClaudeBackend.run()` / `CodexBackend.run()` 作为 sub-agent。`runSubagent` 是 TS 内部函数,没 HTTP route 暴露,从 Python 调用要么改 Brain 要么 TS 重写,工作量都大。OOB 已现成。

### 15.2 两种 sub-agent 形态


| 形态                                 | 触发                                                                                               | 实现                                                                                                          |
| ---------------------------------- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| **ActionOrchestration**(Python 类,无 LLM) | Coordinator 调度 → SubAgentRunner.run(task) → 查 `EXECUTOR_REGISTRY[task.kind]` → 直接 subprocess shell | `BaselineOrchestration` / `BenchRunnerOrchestration` / `ProfileOrchestration` / `ParamSweepRunOrchestration` 等;无 LLM,纯 shell 包装 |
| **OOB sub-agent**(LLM)             | EXECUTOR_REGISTRY 没有命中 → fallback → spawn fresh `backend.run()`                                  | `kernel_extract` / `geak_submitter` / `patch_applier` / `eval_runner` / `rca_runner` 等                      |


### 15.3 统一接口

```python
@dataclass
class DelegatedTask:
    task_id: str            # uuid
    kind: str               # action.name
    params: dict
    idempotency_key: str
    requires_lanes: list[str]
    allowed_tools: list[str]
    side_effects: list[str]
    lease_ttl_sec: int
    state: str              # queued / running / succeeded / failed / cancelled / needs_manual_review
    attempts: int
    history: list[dict]


class SubAgentRunner:
    async def run(self, task: DelegatedTask) -> TaskResult:
        action = self.action_registry.get(task.kind)
        # PolicyGate 取交集校验 allowed_tools
        allowed_tools = self.policy.allowed_tools_for_action(action)

        async with self.locks.acquire_many(action.requires_lanes, task.task_id, task.lease_ttl_sec):
            # Path A: ActionOrchestration
            orchestration = EXECUTOR_REGISTRY.get(task.kind)
            if orchestration is not None:
                try:
                    result = await orchestration.run(OrchestrationContext(task, action, env=self.env))
                    self._publish_intents_via_sink(result.intents)
                    task.transition("succeeded", result.evidence)
                    return result
                except OrchestrationEnvError:
                    pass  # fallback to LLM
            # Path B: OOB backend
            backend = self.backends_pool.pick(action.preferred_backend)
            result = await backend.run(
                prompt=self._compose_prompt(task, action),
                system_prompt=self._load_action_md(action.name),
                cwd=self.workspace,
                model=action.preferred_model,
                max_turns=action.max_turns,
                allowed_tools=allowed_tools,
            )
            parsed = self._parse_result(action, result.trajectory)
            task.transition("succeeded", parsed.evidence)
            return parsed
```

### 15.4 GPU 资源争抢

通过 §3.5 资源锁解决。SubAgentRunner.run() 内部 `acquire_many` 失败会 retry/backoff;超过 `lease_ttl × 2` 转 failed,task 回灌给调度器,不阻塞 reactor。

---

## 16. Action 体系

### 16.0 两层命名:OptimizationAction vs TaskKind

> v0.6 文档里的 "action" 需要和 Gongzheng MVP 已跑通的 `actions/_meta/*.yaml` 结构对齐。MVP 里 `bench_runner` / `param_sweep_run` 是可执行 task kind,而 `kernel-opt` / `integrate` 在 Plan A 里更像 Kernel agent 的 request kind。为避免实现时混淆,本文档采用两层命名:


| 层级                          | 作用                                           | 例子                                                                                                                                 | 是否进入 Scheduler                           |
| --------------------------- | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| **OptimizationAction**      | 调度器评估的优化方向 / 策略节点                            | `baseline` / `profile` / `backends` / `params` / `deep-kernel-analysis` / `operator-tuning` / `vendor-kernel-config` / `recover`   | ✓                                        |
| **TaskKind / OrchestrationKind** | Coordinator 真正 dispatch 的执行任务                  | `bench_runner` / `param_sweep_run` / `profile_runner` / `eval_runner` / `patch_applier` / `geak_submitter` / `kernel_optimization` | ✗ 直接由 action 展开                          |
| **KernelRequestKind**       | Orchestration → Kernel agent 的 Plan A RPC 类型 | `select_kernels` / `run_optimization` / `apply_patch` / `integrate_result`                                                         | 通过 `request{target_agent="kernel"}` 间接触发 |


**兼容 Gongzheng MVP 的规则**:

- `bench_runner` / `param_sweep_run` 保留为 **TaskKind**,不计入 19 个 OptimizationAction,但必须在 `ActionOrchestrationRegistry` 中存在
- `kernel-opt` / `integrate` 在本文档中是 Kernel agent owns 的**优化阶段名**;落地时可映射到 KernelRequestKind(`run_optimization` / `apply_patch`)或作为 synthetic OptimizationAction 保留,但不能和 `bench_runner` 这类 TaskKind 混成一张表
- PolicyGate 的 `KERNEL_OWNED_ACTIONS` 应校验 OptimizationAction / KernelRequestKind 两层,避免 Orchestration 直接 dispatch kernel-owned 副作用 task
- P0 只要求 Coordinator + Orchestration + KernelAgent 主链路跑通:Orchestration 发 REQUEST,Kernel agent 返回 RESPONSE,Coordinator 能把结果入账并触发必要的 bench/evidence-check;Critic/Robustness mock 不阻塞

### 16.1 完整 19 个 OptimizationAction(v0.6,framework-rebuild 已删)


| Action                   | Family      | Owner                    | Source   | accuracy_risk |
| ------------------------ | ----------- | ------------------------ | -------- | ------------- |
| **setup**                | prep        | orchestration            | sprint   | 0.0           |
| **classify**             | prep        | orchestration            | sprint   | 0.0           |
| **target-analysis**      | prep        | orchestration            | sprint   | 0.0           |
| **baseline**             | prep        | orchestration            | sprint   | 0.0           |
| **profile**              | analysis    | orchestration            | sprint   | 0.0           |
| **backends**             | shallow     | orchestration            | sprint   | 0.10          |
| **params**               | shallow     | orchestration            | sprint   | 0.0 / 0.30※   |
| **sweep**                | shallow     | orchestration            | sprint   | 0.0           |
| **report**               | shallow     | orchestration            | sprint   | 0.0           |
| **kernel-opt**           | deep_kernel | **kernel**               | sprint   | 0.05–0.15     |
| **integrate**            | deep_kernel | **kernel**               | sprint   | 0.15          |
| **deep-kernel-analysis** | deep_kernel | **kernel**               | marathon | 0.0           |
| **operator-tuning**      | deep_kernel | **kernel**               | marathon | 0.10          |
| **vendor-kernel-config** | deep_kernel | **kernel**               | sprint   | 0.10          |
| **comm-optimization**    | long        | orchestration※2          | marathon | 0.05          |
| **compiler-tuning**      | long        | orchestration※2          | marathon | 0.05          |
| **dream**                | creative    | Orchestration / Critic※3 | marathon | 0.0           |
| **re-explore**           | creative    | orchestration            | marathon | 0.0           |
| **recover**              | resilience  | **robustness**※4         | marathon | 0.0           |


注:

- ※ `params` 一般 0.0,但 `kv-cache-dtype fp8` / 量化变更属于 0.30
- ※2 架构图里 `comm-optimization` 应由 Comm agent owns,但 v0.6 不做 Comm agent;`compiler-tuning` 同理(原 Framework agent owned)。**这两个 action 在 P0 禁用**,只作为 P1+ 粗粒度入口保留,避免干扰 Coordinator + Orchestration + KernelAgent 主链路验收
- ※3 `dream` 让 Critic 生成"如果 X 是真的,会发生什么"假设,由 Orchestration delegate
- ※4 `recover` 由 Robustness handle 主导

### 16.2 OptimizationAction metadata 示例(无 `allowed_modes` 字段了)

```yaml
---
name: kernel-opt
family: deep_kernel
owner: kernel
cost_minutes_p50: 60
cost_minutes_p75: 120
expected_gain_pct: [5, 25]
accuracy_risk: 0.10
crash_risk: 0.20
prerequisites: [profile, deep-kernel-analysis]
requires_lanes: [server_lifecycle, workspace_mutation, benchmark_lane]
allowed_tools: [Read, Bash, Edit]
side_effects: [workspace_write, server_restart]
preferred_backend: claude
preferred_model: claude-opus-4-7
max_turns: 30
lease_ttl_sec: 7200
dispatch_kind: request_kernel_agent
kernel_request_kind: run_optimization
applicable_when:
  - kernel_dispatch_shows_aiter_dominance
  - cumulative_gain_plateau
---
```

调度器 `prune_gate` 因子读 `family`,`lane_available` 因子读 `requires_lanes`,`depth_gate` 读 `cost_minutes_p75`。当 `dispatch_kind=request_kernel_agent` 时,Coordinator 不直接 spawn sub-agent,而是向 Kernel agent 发 `request{kind=kernel_request_kind}`;Kernel agent 内部再决定是否调 `geak_submitter` / `patch_applier` / `bench_runner` 等 TaskKind。Sub-agent 启动时,Coordinator 还必须按 `allowed_tools` 注入工具白名单;只读 action 不给 `Edit`。

### 16.3 dream / re-explore / recover 三个特殊 action

- **dream**:让 Critic 生成"如果 X 是真的,会发生什么"假设;用于跳出 DFS 局部最优。无副作用,无 lane 需求。Orchestration `delegate(dream)` → Coordinator 调起 Critic 走专门的 dream-mode prompt
- **re-explore**:把已 discard 的 candidate 重新打分(基于新 KB 信息或新 baseline);DFS 回溯。无副作用
- **recover**:Robustness 主导,从 checkpoint 恢复一个之前 crash 的子任务;走 §17 Resume 流程

### 16.4 不做的 action(架构图里有但 v0.6 不实现)

- ~~**framework-rebuild**~~:zhenggong 已删
- 任何 Framework agent / Comm agent 专属的更细分 action(如 PyTorch dispatch routing、AllReduce / Atl2All algorithm selection 等):**v0.6 不做**;`comm-optimization` 和 `compiler-tuning` 作为 P1+ 粗粒度入口保留,**P0 完全禁用**

### 16.5 P0 Action allowlist

P0 目标是先跑通 Coordinator + Orchestration + KernelAgent 主链路,因此 action 范围收窄:

- **允许**:`setup` / `classify` / `target-analysis` / `baseline` / `profile` / `backends` / `params` / `sweep` / `report`
- **允许(通过 KernelAgent request)**:`deep-kernel-analysis` / `kernel-opt` / `integrate` / `operator-tuning` / `vendor-kernel-config` 的最小闭环
- **禁用**:`comm-optimization` / `compiler-tuning`(无真实 Comm/Framework agent owner)
- **可选/后置**:`dream` / `re-explore` / `recover`(不作为 P0 验收条件)

---

## 17. Checkpoint + Resume + Idempotency

### 17.1 设计目标(同 v0.4)

- Resume 不重不漏(消息处理 / task 执行 / 副作用 apply 各自幂等)
- 副作用 action 失败时,evidence 不足以证明 succeeded 的写操作不允许自动重放
- Crash mid-transition 不会留下不一致 state
- **Resume 范围限同 session 内**(`SESSION_DIR` 不变);跨 session 一律启动新 DB,经验通过 KB(L4)传递(ADR-42)

### 17.2 v0.6 数据结构(SQLite per-session 直接落 NFS)

```
$SESSION_DIR/  ($SESSION_DIR 本身就在 NFS / WekaFS 上)
├── storage/
│   └── coordinator.db                 # SQLite WAL,4 表(per-session,直接落 NFS,无 backup 层)
│       ├── leases     (PK=lane)
│       ├── events     (PK=seq AUTOINCREMENT, msg_id UNIQUE)
│       ├── cursors    (PK=agent, last_processed_seq + last_processed_msg_id)
│       └── tasks      (PK=task_id, state machine)
├── state.json                       # 当前 snapshot(30min / KEEP 后写,辅助调试用,真值在 db)
├── personas/<agent>.md              # L3
├── results/<task_id>/               # sub-agent 输出 + idempotency marker
├── findings/                        # Robustness RCA / alerts
│   ├── alerts.jsonl
│   ├── kills.jsonl
│   └── <ts>.json
└── agents/<name>/                   # multi-cli 模式才有
    ├── inbox.jsonl
    └── outbox.jsonl
```

**SoT 划分**(v0.6 明确):

- `tasks` 表是 task lifecycle 的 SoT
- `events` 表是 A2A 消息 / 通知的 SoT
- `cursors` 表是每个 agent 处理进度的 SoT
- `leases` 表是当前持有的 lane lease 的 SoT
- 4 表跨类型一致性由 SQLite 单事务保证
- DB 直接落 NFS,任何 crash 后只要 SESSION_DIR 仍可访问,DB 即可重新打开继续(无 backup → restore 流程)

### 17.3 Message 处理:cursor 幂等

```python
@dataclass
class CursorState:
    agent: str
    last_processed_seq: int
    last_processed_msg_id: str
    processed_at: str

async def process_message(self, agent_name, msg):
    cursor = self.cursors.load(agent_name)
    if msg.seq <= cursor.last_processed_seq:
        return  # 已处理过
    await self._reactor_handle(agent_name, msg)
    self.cursors.upsert(agent_name, msg.seq, msg.id)  # 单 SQL UPSERT
```

### 17.4 委托任务状态机 + evidence-check 优先 retry

```python
TASK_STATES = ["queued", "running", "succeeded", "failed", "cancelled", "needs_manual_review"]

async def dispatch_task(task, action):
    task.transition("running")
    try:
        result = await sub_agent_runner.run(task)
        task.transition("succeeded", result.evidence)
        return result
    except Exception as e:
        task.transition("failed", {"exception": str(e)})
        if not action.side_effects:
            if task.attempts < MAX_RETRY:
                task.attempts += 1
                return await dispatch_task(task, action)
            raise
        else:
            ev = evidence_check(task, action)
            if ev == "succeeded_recovered":
                task.transition("succeeded", {"recovered": True, **ev})
                return load_result_from_evidence(task)
            else:
                task.transition("needs_manual_review", {"evidence": ev})
                raise NeedsManualReviewError(task.task_id)
```

### 17.5 Resume 流程(同 session 内,DB 直接落 NFS,无 restore)

> **范围**:Resume 仅在同一个 session 内有效(`SESSION_DIR` 不变)。跨 session 一律启动新 DB,不复用任何持久化协调状态;跨 session 的"经验"通过 KB(L4,中心化共享服务)传递,不通过 SQLite。

```python
async def resume_from_session(session_dir):
    # 1. DB 直接落 NFS,任何 crash 重启后 SESSION_DIR 仍可访问 → DB 直接打开,无 restore
    db = SqliteConnection(session_dir / "storage" / "coordinator.db")
    # 2. 加载 state snapshot(辅助调试用,真值在 db)
    state = SharedState.load(session_dir / "state.json")
    # 3. cursors / tasks / leases / events 都从 db 自动加载
    # 4. 检查 in-flight 任务(state="running"):evidence-check 矩阵决定续跑/重跑/人工
    for task in db.tasks.where("state=running"):
        ev = evidence_check_matrix(task)
        if ev.verdict == "succeeded":
            task.transition("succeeded", {"recovered": True, **ev.details})
        elif ev.verdict == "safely_failed":
            task.transition("failed", {"reason": "crashed_mid_run, no side effects"})
            scheduler.requeue_action(task.params, attempts=task.attempts+1)
        else:
            task.transition("needs_manual_review", {"evidence": ev.details})
    # 5. 重放 event 给每个 agent(从 cursor 之后)
    for agent in self.agents:
        cursor = db.cursors.load(agent.name)
        for event in db.events.where("seq > ?", cursor.last_processed_seq).order_by("seq"):
            await self.bus.send(agent.name, event.payload)
    return ResumeState(state=state)
```

### 17.6 副作用 Action 崩溃点恢复矩阵(沿用 v0.4 §13.6)


| Action 类型                      | 副作用                     | 崩溃点                           | Resume 判定                                                                  | 恢复动作                                                                |
| ------------------------------ | ----------------------- | ----------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `bench_runner` / `eval_runner` | 生成结果文件,不改 workspace     | benchmark 已跑完,`succeeded` 未写  | 检查 `results/<task_id>/metrics.json` + checksum                             | 补写 `succeeded(recovered=true)`,重发 `delegated_result`                |
| `profile_runner`               | 生成 trace 文件             | trace 部分写入                    | 检查 trace 完整性 marker                                                        | 标记 failed,清理 partial trace,允许重跑                                     |
| `patch_applier` / `integrate`  | 改 workspace / server 配置 | patch 已应用,task state 未 commit | 检查 patch idempotency marker、git diff fingerprint、server config fingerprint | fingerprint 匹配则补写 succeeded;否则 `needs_manual_review`,Robustness RCA |
| `server_restart`               | 重启推理 server             | server 停在 unknown 状态          | 检查 pid / health endpoint / port owner / lease holder                       | Robustness 接管 RCA + 安全 restart                                      |
| `kernel_extract` / read-only   | 只读输出 artifact           | 输出缺失或 partial                 | 检查 artifact checksum                                                       | 缺失则安全重跑                                                             |
| `geak_submitter` / 外部提交        | 外部系统可能已有请求              | 提交成功但本地未记录                    | 用外部 request id / idempotency key 查询                                        | 外部已接受则补写 succeeded;否则按 retry policy                                 |


### 17.7 触发条件

- 每 30min 自动
- 每次 KEEP 决策后立即
- 每次 strategic review 后(由 Critic / Robustness 触发)
- 收到 graceful_stop 信号
- crash 紧急

---

## 18. ★ Critic Review 协议(替代议会)

> v0.6 关键章节。议会模式删除后,Critic 是唯一的 review 角色;它通过 verdict 影响下游执行。真实 Critic 的 `reject` / `redirect` 必须由 KB 知识背书;P0 mock / timeout 产生的 `needs_review` 可以没有 KB evidence,但必须显式标 `source`。

### 18.1 哪些 intent 触发 review

> P0 阶段 Critic 由 mock adapter 代替,用于保证 Coordinator + Orchestration + KernelAgent 主链路先跑通。mock 的原则是**不阻塞主链路**,但不能让高风险副作用在没有明确 verdict 的情况下静默通过:低风险 action 可自动 `advise` 或 `approve(source="mock")`;高风险 action 要么走明确 mock verdict,要么标记 `needs_review` 并由当前 P0 流程选择跳过/延后,不能误报为已被真实 Critic 批准。


| 触发场景                                                                           | 是否需 verdict                                                       |
| ------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| Orchestration 发 `propose_action` 的 action 满足 `accuracy_risk > 0`               | ✓ 必须                                                              |
| Orchestration 发 `propose_action` 的 action 满足 `family in {deep_kernel, long}`   | ✓ 必须                                                              |
| Orchestration 发 `delegate` **直接绕过 propose**(适用 prep / analysis / shallow 类小动作) | ✗ Critic 不阻塞                                                      |
| Kernel 发 `response{kind=integrate, status=keep_proposed}`                      | ✓ 必须(KEEP/REVERT 决策)                                              |
| Robustness 发 `escalate_strategy_change`                                        | ✗ 不直接 review;它只是高优先级建议,Orchestration 后续 proposal 再走 Critic Review |
| Robustness 发 `prune_branch`                                                    | ✗ Critic 事后 review,不阻塞(Robustness 有调度警察权)                         |
| Robustness 发 `kill_task`                                                       | ✗ Critic 事后 review,不阻塞                                            |


> 设计原则:Critic 是"方向决策"的最后一道闸门,Robustness 是"现场救火"的执行者。Robustness 的紧急救火动作不被 Critic 阻塞;其中 `escalate_strategy_change` 只是非破坏性建议,不会直接改状态,因此不存在 rollback 问题。

### 18.1.1 P0 mock 策略(不阻塞主链路)

Critic / Robustness 由专人实现,在 xiaofei 当前分支的 P0 目标里先 mock:


| 场景                                                   | P0 mock 行为                                                                     | 说明                                        |
| ---------------------------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------- |
| `accuracy_risk = 0` 且无 workspace write               | 自动 `advise` 或 `approve(source="mock")`                                         | 让 baseline/profile/params/backends 主链路先跑通 |
| `accuracy_risk > 0` 或 `family in {deep_kernel,long}` | mock 可返回显式 `approve` 以跑通 e2e,但 verdict 必须标 `source="mock"`                     | 报告和 event log 里明确这不是最终 Critic 背书          |
| `integrate keep_proposed`                            | P0 可先只跑到 `needs_review` / `approve(source="mock")` 两种状态,不要求真实 Critic           | 避免把未审阅 kernel patch 误标为最终可用               |
| Critic mock timeout / 未返回 verdict                    | 低风险可 `advise`;高风险必须 `needs_review`,不能默认 `approve`                              | 防止静默放行高风险动作                               |
| Robustness mock                                      | 只发 heartbeat / basic alert;不主动 `kill_task` / `prune_branch` / `force_dispatch` | P0 先验证主链路,调度警察行为后续由 robustness owner 接入   |


P0 验收重点:Coordinator 能正确识别 mock verdict,并把 `source=mock` 写入 events/tasks/history;真实 Critic 接入后只替换 adapter,不改 Orchestration / KernelAgent 协议。

### 18.2 Verdict 集合

```python
class Verdict(str, Enum):
    APPROVE  = "approve"   # 同意,放行进入 task dispatch
    REJECT   = "reject"    # 否决,Coordinator 拦截不执行;Orchestration 必须重新提议
    REDIRECT = "redirect"  # 改方向:Critic 给出 alternative_action(必须是 ActionRegistry 已注册的);Coordinator 替换执行
    ADVISE   = "advise"    # 不阻塞放行,但注入 advice 到下一轮 prompt 让 Orchestration 注意
    NEEDS_REVIEW = "needs_review"  # P0/mock/timeout 场景:高风险 proposal 不放行,等待真实 Critic 或人工处理
```

#### 18.2.1 `approve`

- payload:`{target_proposal_msg_id, verdict="approve", reasoning, kb_evidence?}`
- Coordinator 行为:proposal 转成 task 进入 dispatch 队列;`reasoning` + `kb_evidence` 写入 task 的 `history`
- KB 加分:这条 proposal 完成后,Critic 的 Brier predictor 校准 +1 entry

#### 18.2.2 `reject`

- payload:`{target_proposal_msg_id, verdict="reject", reasoning, kb_evidence}`
- `**kb_evidence` 必填**:必须给出至少 1 条 KB entry id 或 insight id 作为否决依据(防止 Critic 拍脑袋)
- Coordinator 行为:proposal 不进入 task 队列;Orchestration inbox 收到 `review_rejected{proposal_msg_id, reasoning, kb_evidence}`
- Orchestration 必须在下一轮基于 reasoning + kb_evidence 重新提议(不能强行重发同样的 propose)
- 连续 3 次 reject 同一 family 触发 Robustness `escalate_strategy_change`

#### 18.2.3 `redirect`

- payload:`{target_proposal_msg_id, verdict="redirect", reasoning, kb_evidence, alternative_action: {name, params}}`
- `alternative_action.name` 必须在 ActionRegistry 中,且 owner 与原 proposal 同(避免 redirect 后跨 owner 派发)
- Coordinator 行为:用 `alternative_action` 创建 task 进入 dispatch;Orchestration inbox 收到 `review_redirected`
- 用例:Orchestration 提议 `params{kv_cache_dtype=fp8}`,Critic 召回 KB 知道这个模型 fp8 mismatch 严重 → redirect 到 `params{kv_cache_dtype=bf16, mem_fraction=0.85}`

#### 18.2.4 `advise`

- payload:`{target_proposal_msg_id, verdict="advise", reasoning, advice_text, kb_evidence?}`
- Coordinator 行为:proposal 正常进入 task dispatch;`advice_text` 写入 Orchestration 下一轮 prompt 的 `=== Critic advice ===` 段
- 用例:proposal 没问题,但 Critic 想提醒"这个 backend 上次 KEEP 边际很小,如果这次 gain < 3% 不要再深挖"

#### 18.2.5 `needs_review`

- payload:`{target_proposal_msg_id, verdict="needs_review", reasoning, source?="mock|timeout|critic_unavailable"}`
- Coordinator 行为:proposal **不进入 task dispatch**,task/proposal 标记为 `needs_review`;P0 阶段可选择跳过该高风险 action 或只跑到 `approve(source="mock")` 分支,但最终报告必须明确"未经过真实 Critic 审阅"
- 用例:Critic mock / timeout 时遇到 `integrate keep_proposed` 或 `accuracy_risk > 0` 的 proposal,不能静默 approve

### 18.3 KB 如何注入 review 决策

Critic 在 review tick 内执行:

```python
async def review_proposal(self, proposal_msg) -> Verdict:
    # 1. 召回 KB
    kb_entries = subprocess.run([
        "python3", f"{SKILL_ROOT}/kb/kb_query.py",
        f"{state.model_name} {proposal_msg.payload.action_name}",
        "--top-k", "5", "--compact",
    ], capture_output=True).stdout

    # 2. 召回 critic 自己的 persona(L3,本 run 累积)
    persona = read_persona("critic")

    # 3. 召回最近 5 条相关 decision
    recent = state.last_decisions(n=5)

    # 4. 调 backend.run() 出 verdict
    result = await self.backend.run(
        prompt=self._compose_review_prompt(proposal_msg, kb_entries, persona, recent),
        system_prompt=self._load_critic_review_prompt(),
        max_turns=3,
        allowed_tools=["Read", "Bash(kb_query.py|kb_ingest.py)"],
    )

    intents = parse_intents(result.trajectory)
    verdict_intent = next((i for i in intents if i.type == IntentType.REVIEW_VERDICT), None)
    if verdict_intent is None:
        # No verdict emitted:
        # - low risk: degrade to ADVISE so P0 can keep moving
        # - high risk: block as NEEDS_REVIEW; never silently approve
        if is_low_risk(proposal_msg.payload):
            return Verdict.ADVISE
        return Verdict.NEEDS_REVIEW
    return Verdict(verdict_intent.payload["verdict"])
```

### 18.4 Brier 校准

每个 `approve` / `redirect` 后,Critic 同时在 `review_verdict.payload` 里附带 `predicted_gain_pct`(数值预测)。运行结束后:

```python
brier = (predicted_gain_pct / 100 - actual_gain_pct / 100) ** 2
```

`brier` 滑动窗口 N=20,Critic 自己写 `personas/critic.md` 时附带 brier 历史。Coordinator 用 brier 决定:

- `brier < 0.05`(预测准):对低风险 proposal 启用 sample-down(默认 review,但 20% 概率跳过 → 直接 approve)节省 token
- `brier > 0.15`(预测差):停用 sample-down,所有 proposal 都过完整 review
- `brier_plateau` 信号(滑动窗口内 brier 标准差 < 0.02)是早停信号 4(§9.1)

### 18.5 Devil's advocate(`objection` intent 的新定位)

议会删除后,Critic 仍保留 `objection` intent 作为"反对意见"出口:

- 与 `review_verdict` 不同:`review_verdict` 是闸门(approve/reject/redirect/advise/needs_review);`objection` 是**主动 flag**,可以对当前已批准的 task / 已经 KEEP 的决策事后反对
- Coordinator 行为:`objection` 转成 `topic="advice", priority=1` 的 `send_message` 进 event log;Orchestration 在下一轮 prompt 看到;**不触发任何议会、不投票**
- 用例:Critic 看到 KEEP 后 cumulative_gain 涨了 +3%,但 KB 知道这模型在这个参数下半小时后会 OOM,主动发 objection 提醒

---

## 19. ★ Robustness 干预协议

### 19.1 always-on 健康监控

Robustness tick 默认 60s(`robustness_tick_s = 60.0`),每 tick 做:

1. 读 `events` 表 tail(过去 60s 的 events)
2. 检查 health signals:
  - `crash_count` 是否上升
  - 是否有 reactor stall(某 agent 超过 3min 没 process 新 message)
  - server pid / port / GPU memory 是否异常
  - lease 是否有 holder dead 但 lease 没 release
3. 命中任一 → emit `alert{severity, summary, detail}`(高 sev 优先级 0)
4. 如果 alert 不足以解决,继续走 §19.3 调度警察 intent

### 19.2 RCA 流程

触发条件(任一):

- `crash_count >= 2`
- repeated KEEP/REVERT bouncing(同一 action 在 30min 内 KEEP 后又 REVERT 2+ 次)
- `escalate_strategy_change` 自指(Robustness 自己的 escalate 没改善状况)

执行:

```python
async def do_rca(self):
    # Robustness 直接读
    event_tail = bus.tail(n=200)
    state_snap = state.summary()
    recent_decisions = state.last_decisions(n=5)
    recent_kb_entries = subprocess.run(["python3", "kb_query.py", state.model_name, "--top-k", "10"], ...)

    finding = await self.backend.run(
        prompt=self._compose_rca_prompt(event_tail, state_snap, recent_decisions, recent_kb_entries),
        system_prompt=self._load_rca_prompt(),
        max_turns=10,
    )
    finding_data = parse_rca_finding(finding.trajectory)
    write_finding(self.session_dir / "findings" / f"{ts}.json", finding_data)
    await self.bus.send("coordinator", {"topic": "rca_done", "finding": finding_data})

    # 根据 finding 决定后续动作
    if finding_data.action == "kill_task":
        emit_intent(IntentType.KILL_TASK, ...)
    elif finding_data.action == "prune_branch":
        emit_intent(IntentType.PRUNE_BRANCH, ...)
    elif finding_data.action == "strategy_change":
        emit_intent(IntentType.ESCALATE_STRATEGY_CHANGE, ...)
```

#### Robustness 不可用时的处理

- Robustness process / reactor 自身 hang / crash → Coordinator 探测到 5min 没 heartbeat → 写 `robustness_unavailable` alert,暂停新的高风险 action,已有低风险 task 可继续
- **不把 RCA 路由给 Critic**。Critic 不做 RCA,也不会临时拿 Bash 诊断权限;RCA 等 Robustness owner 接入或人工处理

### 19.3 调度警察 4 个 intent(详细 payload + Coordinator 行为)

#### 19.3.1 `kill_task`

- payload:`{task_id, reason, force?: bool, scope: "task"}`
- 限制:`scope` 强制 `"task"`,**不允许** `"process"` / `"server"`(IR-5 仍主导 server lifecycle;Robustness handle 子角色才能在 server lifecycle lane 内合法操作)
- Coordinator 行为:
  - task 状态是 `queued` → 直接转 `cancelled`,不进入 dispatch
  - task 状态是 `running` → 通知对应 sub-agent 中断(SIGTERM,5s 后 SIGKILL),task 转 `cancelled`,释放 lease
  - 镜像到 `findings/kills.jsonl`

#### 19.3.2 `force_dispatch`

- payload:`{task_id, reason}`
- Coordinator 行为:把 task `created_at` 改成 unix epoch(1970-01-01)→ dispatcher 按 `created_at` 升序选,自然排到队首
- 用例:high-value `bench_runner` 卡在 6 个低价值 proposal 后面,Robustness 判断"surface validation now"

#### 19.3.3 `prune_branch`

- payload:`{family, reason}`
- `family` 必须是 `prep` / `analysis` / `shallow` / `deep_kernel` / `long` / `creative` / `resilience` 之一
- Coordinator 行为:
  - 把 family 加到 `state.pruned_families`(set)
  - 把所有 `kind in actions_with_family(family)` 且 `state in {queued}` 的 task 转 `cancelled`
  - Scheduler 下次 `score_action` 时,`prune_gate = 0` 永不再选这个 family
  - 写 `findings/prunes.jsonl`
- 用例:3+ 次同 family 失败,Robustness 砍整族

#### 19.3.4 `escalate_strategy_change`

- payload:`{reason, next_action_hint, severity?: "high" | "medium"}`
- Coordinator 行为:
  - emit `topic="strategy_change", priority=0` 的 broadcast(所有 agent 都收到)
  - Orchestration 下一 tick 必须读这条,prompt 里高亮显示
  - **不直接修改 scheduler / pruned_families / dispatch queue**,因此不存在 rollback 问题
  - Orchestration 基于 `next_action_hint` 重新提出 proposal;该 proposal 再走 Critic Review(§18)
  - 如果 Critic reject,这条 strategy_change 仅作为历史 advice 留在 event log,无需回滚任何状态
- 用例:trace 显示 attention is bottleneck 但 Orchestration 还在调 prefill 参数 → Robustness escalate "建议考虑 attention backend";Orchestration 再决定是否提出对应 proposal

### 19.4 Handle 子角色

Robustness 还承担"执行手"角色,通过 `delegate` intent 调起 ephemeral sub-agent:


| Handle 任务              | 调起                                                                                       | Lane               |
| ---------------------- | ---------------------------------------------------------------------------------------- | ------------------ |
| **server lifecycle**   | `delegate(server_restart)` → spawn `patch_applier` 或直接 Bash(IR-4 / IR-5)                 | server_lifecycle   |
| **accuracy gate exec** | `delegate(eval_runner)` → spawn `eval_runner` 跑 GSM8K,FAIL → 通知 Coordinator `needs_revert` | benchmark_lane     |
| **recovery**           | `delegate(recover)` → SubAgentRunner 执行 §17.6 evidence-check 矩阵                          | 按对应 action 的 lanes |


注意:Robustness 不直接改 workspace,所有副作用走 sub-agent + 资源 lease。

---

## 20. Multi-CLI Runtime(本地过渡方案)

### 20.1 当前两种传输模式


| Mode                  | 实现                                                                                                                                                                    | 用途                                   |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `**SINGLE_PROC`**(默认) | 所有 agent 在 Coordinator 进程内跑 asyncio reactor,通过 `backend.run()` 直接调 SDK                                                                                                  | 单测 / 短任务 / 当前默认                      |
| `**MULTI_CLI`**       | 每个 agent 起独立 OS 进程,跑 `claude --print --continue` 或 `codex` CLI;进程间靠 `$SESSION_DIR/agents/<name>/inbox.jsonl` / `outbox.jsonl` 通信;`MultiCLIRouter` 桥接到 SQLite events 表 | **本地全功能跑**(沿用 marathon tmux 形态);崩溃隔离 |


### 20.2 v0.6 默认策略

- 默认 `--transport single-proc`
- `--transport multi-cli` 显式启用
- 长任务(`MAX_HOURS > 12`)启动时打印警告:"建议使用 `--transport multi-cli` 获得崩溃隔离"
- **没有 mode 自动切换**(避免增加心智负担)

### 20.3 未来:claw 子 session 形态(v0.7+ 目标)

按 xiaofei #6 的中长期规划:

- claw 提供"在主 sandbox 内创建子 sub-session"能力(目前未完成)
- 每个 persistent agent 绑定一个 claw sub-session,sub-session 内独立 process / 独立 conversation
- 取代 multi-cli 的"本地起 N 个 OS 进程"形态:稳定性更好、claw 平台原生管理、跨 sandbox 也能扩展
- 切换契约:agent 与 Coordinator 的协议(intent / inbox/outbox JSONL / cursor / SQLite events)**完全不变**,只是底层进程载体从 OS process 换成 claw sub-session

### 20.4 切换契约(三种形态共享同一个上层接口)

```
Coordinator 视角的 agent reactor:
    bus.recv(agent_name) → list[Message]
    bus.send(agent_name, msg)
    cursor.advance(agent_name, seq, msg_id)

底层实现可换:
    SINGLE_PROC: in-process asyncio.Queue
    MULTI_CLI:   inbox.jsonl + outbox.jsonl + Router 桥接
    CLAW_SUBSESSION: claw API 调用 + Router 桥接(同 multi-cli 协议)
```

**Multi-CLI 是过渡方案,不是终点**。文档明确这一点,避免后续把 multi-cli 当默认。

---

## 21. Coordinator 主循环骨架

```python
from oob.backends import ClaudeBackend, CodexBackend
from .agent_role import AgentRole, default_role_registry, roles_for_run

class Coordinator:
    def __init__(self, session_dir, env, transport_mode="single-proc"):
        self.objective = build_objective(env)         # §11
        self.session_dir = session_dir
        self.transport_mode = transport_mode

        # v0.6: 4 角色全启用,无 mode gating
        self.roles = roles_for_run()  # [orchestration, kernel, critic, robustness]
        self.agents = {
            "orchestration": AgentRole("orchestration", ClaudeBackend(),  "claude-opus-4-7"),
            "kernel":        AgentRole("kernel",        ClaudeBackend(),  "claude-opus-4-7"),
            "critic":        AgentRole("critic",        CodexBackend(),   "gpt-5.4"),
            "robustness":        AgentRole("robustness",        ClaudeBackend(),  "claude-opus-4-7"),
        }

        # 基础设施
        self.db        = SqliteConnection(session_dir / "storage" / "coordinator.db")
        self.bus       = MessageBus(self.db)
        self.state     = SharedState(session_dir)
        self.scheduler = BudgetAwareScheduler(self.objective, env, self.action_registry)
        self.kb_query_service = KbQueryService(...)   # 给所有 agent prompt 注入 KB hint 用,Critic 是真正的 read/write owner
        self.locks     = ResourceLockManager(SqliteLeaseBackend(self.db))
        self.tasks     = TaskRegistry(self.db)
        self.cursors   = CursorStore(self.db)
        self.policy    = PolicyGate(self.role_registry, self.action_registry)
        self.actions   = ActionRegistry("actions/")
        self.sub       = SubAgentRunner(self.locks, self.workspace, self.actions, self.policy,
                                        orchestration_registry=EXECUTOR_REGISTRY,
                                        intent_sink=self._orchestration_intent_sink)
        self.review_gate = CriticReviewGate(self.bus, self.agents["critic"], self.policy)  # §18

        # transport 选择
        if transport_mode == "multi-cli":
            self.launcher = MultiCLILauncher(...)
            self.router   = MultiCLIRouter(...)
        else:
            self.launcher = self.router = None

    async def run(self):
        if self.session_dir.has_checkpoint():
            await self.resume_from_checkpoint()
        else:
            await self._init_session()

        if self.transport_mode == "multi-cli":
            self._staged = self.launcher.launch_subprocess()

        await asyncio.gather(
            *(self._reactor(name) for name in self.agents),
            self._clock(),
            self._stopping_watcher(),
            self._dispatcher_loop(),
        )
        await self._graceful_stop(self.state.stop_reason)

    async def _reactor(self, agent_name):
        agent = self.agents[agent_name]
        while not self.state.should_stop():
            msgs = await self.bus.recv(agent_name, timeout=60)
            cursor = self.cursors.load(agent_name)
            msgs = [m for m in msgs if m.seq > cursor.last_processed_seq]
            if not msgs:
                continue
            kb_hint = await self.kb_query_service.recall(self.state.model_name, self.state.current_action) \
                      if agent_name in ("orchestration", "kernel") else None
            critic_verdict = self.state.last_review_verdict_for(agent_name) \
                             if agent_name == "orchestration" else None
            robustness_alert = self.state.last_robustness_alert() \
                           if agent_name == "orchestration" else None
            prompt = self._compose_prompt(agent_name, msgs, kb_hint, critic_verdict, robustness_alert)
            try:
                allowed_tools = self.policy.allowed_tools_for_agent(agent_name)
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
                continue

            for intent in parse_intents(result.trajectory):
                await self._handle_intent(agent_name, intent)

            for msg in msgs:
                self.cursors.upsert(agent_name, msg.seq, msg.id)

    async def _handle_intent(self, from_agent, intent):
        try:
            self.policy.validate_intent(from_agent, intent, self.state)
        except PolicyDenied as exc:
            await self._record_observation("coordinator", "policy_denied", {
                "agent": from_agent, "intent_type": intent.type.value,
                "rule": exc.rule, "reason": str(exc), "hint": exc.hint,
            })
            return

        if intent.type == IntentType.SEND_MESSAGE:
            await self.bus.send(intent.payload["to"], intent.payload)
        elif intent.type == IntentType.PROPOSE_ACTION:
            # §18 Critic Review 闸门:必要时拦截
            if self.review_gate.requires_verdict(intent.payload):
                verdict = await self.review_gate.review(from_agent, intent)
                await self.review_gate.apply(verdict, intent)
            else:
                await self._record_proposal(from_agent, intent)
        elif intent.type == IntentType.DELEGATE:
            action = self.actions.get(intent.payload["task_kind"])
            task = self.tasks.create(action, intent.payload["params"])
            # dispatcher_loop 会异步取走
        elif intent.type == IntentType.REQUEST:
            # Plan A: orchestration → kernel
            await self.bus.send("kernel", {"topic": "request", **intent.payload})
        elif intent.type == IntentType.RESPONSE:
            # kernel → orchestration;Critic Review 闸门可能拦截
            if self.review_gate.requires_verdict_for_response(intent.payload):
                verdict = await self.review_gate.review_response(from_agent, intent)
                await self.review_gate.apply_response_verdict(verdict, intent)
            else:
                await self.bus.send(intent.payload["target_agent"], intent.payload)
        elif intent.type == IntentType.REVIEW_VERDICT:
            await self.review_gate.record(from_agent, intent)
        elif intent.type == IntentType.UPDATE_STATE:
            self.state.apply_validated_transition(from_agent, intent.payload["changes"])
        elif intent.type == IntentType.UPDATE_PERSONA:
            self.kb.append_persona(from_agent, intent.payload["note"])
        elif intent.type == IntentType.KILL_TASK:
            await self._handle_kill_task(from_agent, intent.payload)
        elif intent.type == IntentType.FORCE_DISPATCH:
            await self._handle_force_dispatch(from_agent, intent.payload)
        elif intent.type == IntentType.PRUNE_BRANCH:
            await self._handle_prune_branch(from_agent, intent.payload)
        elif intent.type == IntentType.ESCALATE_STRATEGY_CHANGE:
            await self._handle_escalate_strategy_change(from_agent, intent.payload)
        elif intent.type == IntentType.OBJECTION:
            # 不触发议会,转 advice
            await self.bus.send(intent.payload.get("target_agent", "orchestration"),
                                {"topic": "advice", "priority": 1, **intent.payload})
        elif intent.type == IntentType.ALERT:
            await self._handle_alert(from_agent, intent.payload)
        # ... 其它
```

---

## 22. 用户接口

### 22.1 极简入口(单一全模式)

```bash
@inference_optimizer

MODEL_PATH=/hyperloom/models/DeepSeek-R1-0528
MODEL_NAME=deepseek-ai/DeepSeek-R1-0528
TP=8 GPU_TYPE=MI355X FRAMEWORK=sglang

# MAX_HOURS 必填:用于早停预算 + 调度 pressure
MAX_HOURS=24

# TARGET_* 可选:用于效果早停;最多同时指定一个
TARGET_GAIN_PCT=30                   # 或 TARGET_TPUT_PER_GPU=700
                                     # 或 TARGET_DIR=/path/to/B200_baseline
```

**与 v0.4 / v0.5 不同**:不再有 mode 自动选择;无论 `MAX_HOURS=0.5` 还是 `MAX_HOURS=48`,都启动同样的 4 个 agent + 19 个 action,差别由 Objective + 调度器 + Critic Review 自动产生。

### 22.2 短任务示例(等价原 quick mode)

```bash
@inference_optimizer

MODEL_PATH=/hyperloom/models/Qwen3-8B
MODEL_NAME=Qwen/Qwen3-8B
TP=8 GPU_TYPE=MI355X FRAMEWORK=sglang
MAX_HOURS=1.5
TARGET_GAIN_PCT=10
```

预期行为(无 mode 配置):

- Scheduler 因为 `time_left = 1.5h`,自动 prune 掉 cost_p75 > 1.2h 的 action(`kernel-opt` / `framework-rebuild` 已删 / `comm-optimization` 等)
- Critic Review 对 `accuracy_risk = 0` 的 backends / params 启用 sample-down(brier 好的话)
- Robustness always-on 但通常静默
- 达到 +10% 立即停

### 22.3 长任务示例

```bash
@inference_optimizer

MODEL_PATH=/hyperloom/models/DeepSeek-R1-0528
MODEL_NAME=deepseek-ai/DeepSeek-R1-0528
TP=8 GPU_TYPE=MI355X FRAMEWORK=sglang
MAX_HOURS=24
TARGET_GAIN_PCT=30
```

预期行为:

- 4 agent 全开,所有 19 action 可选
- Critic 主动每 6h cross-run synthesis
- Robustness 每 60s tick + 调度警察 4 intent 干预
- persona 每 4h / 8K token 自蒸馏

### 22.4 兼容老入口

保留一个 release 作为 thin shim:

- 老 `quick` / `guided` / `marathon` env 自动映射到 v0.6 单一入口(只取 `MAX_HOURS` + `TARGET_*`)
- 下一个 release 删除

### 22.5 CLI flag

```bash
python -m inference_optimizer \
  --model "$MODEL_PATH" \
  --max-hours "$MAX_HOURS" \
  --backend claude \                     # claude / codex / mock
  --transport single-proc \              # single-proc / multi-cli (默认 single-proc)
  --auto-install                         # 缺 Node / claude CLI 时自动 bootstrap
```

---

## 23. 文件 / 目录结构

```
.cursor/skills/inference_optimizer/   # 入口与规则
├── SKILL.md
├── README.md
└── KNOWLEDGE-BASE.md

src/inference_optimizer/              # 实现独立目录(统一代码库)
├── orchestrator/
│   ├── coordinator.py
│   ├── agent_role.py                 # 4 角色 default_registry
│   ├── message_bus.py                # SQLite events 表
│   ├── shared_state.py
│   ├── scheduler.py
│   ├── score_priors.py               # §12.2 Initial Score Priors
│   ├── objective.py                  # §11
│   ├── policy.py                     # §14.5 PolicyGate(含 KB allowlist / robustness_only / kernel_owned)
│   ├── intent_parser.py              # §14.1 INTENT_ENVELOPE_SCHEMA + REVIEW_VERDICT
│   ├── resource_lock.py              # §3.5 SQLite WAL backend
│   ├── task_registry.py              # §17.4 状态机
│   ├── cursor_store.py               # §17.3
│   ├── sub_agent_runner.py           # §15
│   ├── persona.py                    # 蒸馏(全 4 角色)
│   ├── kb.py                         # §8 KB 操作(由 Critic 触发)
│   ├── checkpoint.py                 # §17
│   ├── early_stop.py                 # §9
│   ├── accuracy_gate.py              # §10
│   ├── iron_rules.py                 # §4
│   ├── kernel_opt_constants.py       # §5
│   ├── process_management.py         # §6
│   ├── critic_review.py              # ★ §18 NEW v0.6
│   ├── robustness_intervention.py        # ★ §19 NEW v0.6
│   ├── action_registry.py            # actions/_meta/*.yaml 加载
│   ├── action_orchestrations/             # §15.2 ActionOrchestration
│   │   ├── base.py
│   │   ├── baseline.py
│   │   ├── bench_runner.py
│   │   ├── profile.py
│   │   └── param_sweep_run.py
│   ├── backends/
│   │   ├── base.py
│   │   ├── claude.py                 # ClaudeBackend + MCP emit_intent
│   │   ├── codex.py                  # CodexBackend + validated_json_output
│   │   ├── mcp_emit_intent.py
│   │   └── mock.py
│   ├── multi_cli/                    # §20 过渡方案
│   │   ├── launcher.py
│   │   ├── router.py
│   │   ├── envelope.py
│   │   ├── agent_card.py
│   │   └── codex_continuity.py
│   └── system_prompts/
│       ├── orchestration.md          # 改名(原 orchestration.md)
│       ├── kernel.md
│       ├── critic.md                 # 含 KB / Sage / Devil's advocate / Review 段;不含 RCA
│       ├── robustness.md                 # 含 Robustness monitor + RCA + Handle + 调度警察 4 intent 段
├── agents/                            # 每个 agent 独立 skill 目录(multi-cli 用)
│   ├── orchestration/
│   ├── kernel/                       # ★ xiaofei 当前分支正在做
│   ├── critic/
│   └── robustness/
├── actions/                          # 19 个 action(framework-rebuild 已删)
│   ├── setup.md / classify.md / target-analysis.md / baseline.md / profile.md
│   ├── backends.md / params.md / sweep.md / report.md
│   ├── kernel-opt.md / integrate.md
│   ├── deep-kernel-analysis.md / operator-tuning.md / vendor-kernel-config.md
│   ├── comm-optimization.md / compiler-tuning.md
│   ├── dream.md / re-explore.md / recover.md
│   └── _meta/                         # 每个 action 的 yaml metadata(无 allowed_modes)
├── kb/                                # KB client(后端是中心化共享存储,T13)
│   ├── kb_query.py                    # Critic 调,client 抽象,后端可切 jsonl/HTTP
│   ├── kb_ingest.py                   # Critic 调
│   ├── client/                        # 中心化 KB 后端 client 实现
│   │   ├── nfs_jsonl.py               # Option A 过渡:/wekafs/kb/<model_family>/<model_name>/*.jsonl
│   │   └── http.py                    # Option B 目标:HTTP service + REST + embedding
│   └── schema.py                      # entry / insight / conflict schema
├── scripts/
│   ├── run_baseline.sh
│   ├── eval_accuracy.sh
│   ├── patch_inductor.py
│   ├── geak_ray_submit.py
│   └── oob_ray_submit.py
├── storage/
│   ├── connection.py                  # SqliteConnection (WAL + BEGIN IMMEDIATE,DB 直接落 NFS)
│   ├── schema.py                      # 4 表 DDL
│   └── nfs_self_test.py               # T20:WekaFS WAL 模式 self-test(多进程并发写 / 断电恢复)
└── tests/
    ├── test_objective.py
    ├── test_feature_flags.py          # ← v0.6 删除(无 feature flags)
    ├── test_policy.py                 # 含 critic KB allowlist + robustness_only + kernel_owned
    ├── test_scheduler.py              # 含 prune_gate
    ├── test_intent_parser.py          # 含 REVIEW_VERDICT + 删除 vote / parliament_open
    ├── test_resource_lock.py
    ├── test_idempotency.py
    ├── test_resume.py
    ├── test_iron_rules.py
    ├── test_accuracy_gate.py
    ├── test_critic_review.py          # ★ NEW v0.6
    ├── test_robustness_intervention.py    # ★ NEW v0.6
    ├── test_kb_workflow.py            # ★ NEW v0.6 (Critic-driven)
    └── e2e/
```

> v0.6 删除项:`execution_mode.py` / `feature_flags.py` / 所有按 mode 分支的测试。

---

## 24. 风险与缓解


| 风险                                   | 概率  | 影响  | 缓解                                                                                                                                                                                                         |
| ------------------------------------ | --- | --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| OOB Backend 在 GPU sandbox 内跑不通       | 低   | 高   | 实施第一件事就验证                                                                                                                                                                                                  |
| Codex `gpt-5.4` 不稳定 / litellm 5.4 限速 | 中   | 中   | Critic 是 no-tools(KB 例外),输出 `validated_json_output`;1 轮 repair retry;最终失败 → 低风险转 advice,高风险 `needs_review`;RCA 不由 Critic 承担                                                                                |
| Critic 误判反而拖慢决策                      | 中   | 中   | Brier 校准 + sample-down(brier 好时低风险 proposal 20% 概率跳过 review);连续 reject 同 family 触发 Robustness escalate                                                                                                     |
| Robustness 漏报 / 误干预                  | 中   | 中   | 所有 `force_dispatch` / `prune_branch` / `escalate_strategy_change` 都写 findings/events 留痕;`escalate_strategy_change` 是非破坏性建议,不直接改状态                                                                          |
| KB 知识陈旧 / 误导                         | 中   | 中   | Critic cross-run synthesis 生成 insights;conflicts.jsonl 标注矛盾;新 entry 加权高于老 entry;warm-start 第 2+ 次同 model_family 才 read                                                                                     |
| cost_minutes 估算不准                    | 中   | 中   | P75 + KB 历史校准 + Critic Brier                                                                                                                                                                               |
| pressure 过激进 → 频繁 crash              | 中   | 中   | crash_count ≥ 2 紧急停 + crash_risk 因子压制 + Robustness 监控                                                                                                                                                      |
| 4 agent token 暴涨                     | 低   | 中   | 事件驱动 + Critic sample-down + Codex Critic 分担;估 ~12-15M / 24h(略高于 v0.4 marathon 11.5M,因 Critic 全启用 + KB 调用)                                                                                                  |
| KB 多 user 多 session 写入冲突 / 数据污染      | 中   | 中   | 中心化共享 + 按 `<model_family>/<model_name>` 分区 + Critic 是唯一写入口(简化并发) + ingest 时校验 model 字段一致性 + Option B(HTTP service)天然有服务侧串行化                                                                                |
| 资源锁死锁                                | 低   | 高   | SQLite 单事务 acquire-many + 按 lane 顺序 + timeout + 调度器全局检测                                                                                                                                                    |
| 任务状态机崩在 transition 中间                | 低   | 高   | §17.6 evidence-check 矩阵;needs_manual_review 兜底                                                                                                                                                             |
| Critic Bash allowlist 被绕过            | 低   | 中   | PolicyGate 字符串前缀严格匹配 + e2e 黑盒测试;违规 → `policy_denied` + 不 retry                                                                                                                                             |
| Robustness 自身 hang(单点失效)             | 低   | 高   | Coordinator 5min heartbeat 探测,写 `robustness_unavailable` alert,暂停新的高风险 action;不路由给 Critic,等待 Robustness owner / 人工处理                                                                                         |
| Multi-CLI 进程间消息丢失                    | 中   | 中   | inbox/outbox JSONL + cursor + Router seq 桥接;Resume 时按 SQLite events 回放                                                                                                                                     |
| Plan A REQUEST/RESPONSE 路由抖动         | 低   | 中   | request_msg_id 必填 + response.in_reply_to 必填 + 60s 超时则 Robustness 介入                                                                                                                                        |
| Orchestration 改名后 zhenggong 分支不一致    | 中   | 低   | 文档统一新名;开发阶段直接在本分支整体 rename(不单独 PR);zhenggong 分支不再继续开发                                                                                                                                                      |
| **WekaFS / NFS 上 SQLite WAL 不可靠**    | 中   | 高   | 落地前必跑 T20 self-test(多进程并发写 / 断电恢复 / wal+shm 文件交互);self-test 通过 → 直接落 NFS;失败 → fallback 回"本地盘 + 30min NFS VACUUM INTO backup + restore"两层方案;`SqliteConnection` 启动时记录 `journal_mode=WAL` 实际生效,失败立即 fail-fast |


---

## 25. 决策记录(ADR)

> 沿用 v0.4 ADR-1 ~ ADR-25(部分被替代);v0.5 新增 ADR-33 SQLite WAL;v0.6 新增 ADR-34 ~ ADR-43。

### 沿用(部分修订)


| ID     | 决策                                                                      | v0.6 状态                                                                        |
| ------ | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| ADR-1  | 合并 sprint + marathon 成新 skill                                           | 沿用                                                                             |
| ADR-2  | 完整形态 4 个 persistent agent + Coordinator                                   | 沿用,角色名重组(Orchestration/Kernel/Critic/Robustness)                               |
| ADR-3  | ~~Sage 合并 Brainstormer + Historian~~ → **Critic 接管 KB / Sage / Review** | **修订**(v0.6 ADR-35);RCA 改归 Robustness                                          |
| ADR-4  | Callable + 4 层记忆                                                        | 沿用                                                                             |
| ADR-5  | Orchestration / Kernel / Robustness = Claude `claude-opus-4-7`          | 沿用 + 改名(原 ADR-5 是 Orchestration/Robustness monitor)                                           |
| ADR-6  | Critic = Codex `gpt-5.4`(litellm 暂不支持 5.5)                              | 沿用 + model 显式标注                                                                |
| ADR-7  | Coordinator 是 Python                                                      | 沿用                                                                             |
| ADR-8  | 复用 OOB backend 抽象                                                       | 沿用                                                                             |
| ADR-9  | ~~复用 Claw runSubagent~~ Superseded by ADR-14                            | 已超越                                                                            |
| ADR-10 | 早停 5 信号 OR                                                              | 沿用                                                                             |
| ADR-11 | L4 跨 run KB 全 mode 启用                                                   | 沿用 + Critic owns + **中心化共享存储,按 `<model_family>/<model_name>` 分区**(v0.6 ADR-35) |
| ADR-12 | A2A 自由通信 + 4 协议规则                                                       | 沿用,**删除议会**(v0.6 ADR-38)                                                       |
| ADR-13 | Objective 抽象                                                            | 沿用                                                                             |
| ADR-14 | OOB-only sub-agent                                                      | 沿用                                                                             |
| ADR-15 | Event log + cursor + 状态机                                                | **超越**:由 ADR-33 SQLite WAL 单库统一持久化                                             |
| ADR-16 | 单 GPU sandbox 默认                                                        | 沿用                                                                             |
| ADR-17 | 结构化 Intent Transport(Claude tool_call + Codex JSON-only)                | 沿用,Codex 加 KB 例外(v0.6 ADR-35)                                                  |
| ADR-18 | 资源锁 4 lane                                                              | 沿用,后端改 SQLite(ADR-33)                                                          |
| ADR-19 | 早停 reason 分级尾流                                                          | 沿用                                                                             |
| ADR-20 | Critic 在 guided 启用                                                      | **修订**:无 mode 概念,Critic 全启用(v0.6 ADR-34)                                       |
| ADR-21 | L4 第 2+ 次同模型族才 read                                                     | 沿用                                                                             |
| ADR-22 | Persona 蒸馏机制                                                            | 沿用,全 4 角色启用(无 mode gating)                                                     |
| ADR-23 | Brier 默认等权重                                                             | 沿用 + 加 sample-down(v0.6 §18.4)                                                 |
| ADR-24 | ~~PoC 前必须通过 Design Gate~~ 已删                                            | 已删                                                                             |
| ADR-25 | 运行时三档执行模式                                                               | **超越**:v0.6 ADR-34 单一全模式                                                       |
| ADR-26 | 统一代码 + Feature Flag 子集                                                  | **超越**:v0.6 ADR-34 单一全模式后无需 feature flag                                       |
| ADR-27 | Robustness monitor guided 不常驻,曾考虑非 Robustness RCA fallback                        | **超越**:Robustness always-on(v0.6 ADR-36),Critic 不做 RCA                         |
| ADR-28 | L4 KB 全 mode 启用                                                         | 沿用                                                                             |
| ADR-29 | Sage 三段都启用                                                              | **超越**:Sage 角色删除,Critic 接管(v0.6 ADR-35)                                        |
| ADR-30 | Iron Rules 等硬资产全 mode 强制                                                | 沿用,改成"全模式强制"                                                                   |
| ADR-31 | quick mode Bash allowlist                                               | **超越**:无 mode,Bash allowlist 按 agent 角色定义(v0.6 §14.5)                          |
| ADR-32 | 删除 Roadmap + PoC 章节                                                     | 沿用                                                                             |
| ADR-33 | SQLite WAL 单库                                                           | 沿用(v0.5);**生命周期由 v0.6 ADR-42 细化为 per-session**                                 |


### 新增(v0.6)


| ID         | 决策                                                                                                                                                                                                                                                 | 替代方案                                          | 理由                                                                                                                    |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **ADR-34** | **单一全模式**(无 quick / guided / marathon)                                                                                                                                                                                                             | 三档分段 / 二档                                     | v0.5 实施观察:guided 与 marathon roster 已收敛;成本 > 收益;短任务限制由 Objective + 调度器 + Critic Review 自然实现                            |
| **ADR-35** | **Critic 接管 KB / Sage / Devil's advocate / Review 优化建议;不做 RCA;Codex no-tools 原则在 KB 处放开**(`Bash(kb_query.py / kb_ingest.py)`)                                                                                                                      | Sage 独立 / KB 独立服务 / 非 Robustness RCA fallback | 用户明确:Critic 负责 review 和更高层建议,不做 RCA                                                                                   |
| **ADR-36** | **Robustness = Robustness monitor + RCA + Handle 三合一,always-on,保留调度警察 4 intent**(`kill_task` / `force_dispatch` / `prune_branch` / `escalate_strategy_change`)                                                                                               | Robustness monitor 与 Sage 独立                            | xiaofei #2:robustness 本身就是这三件事的合并体                                                                                    |
| **ADR-37** | **orchestration → orchestration 改名**(贴合架构图 Layer-1 expert)                                                                                                                                                                                              | 保留 orchestration                                   | xiaofei:统一用新名                                                                                                         |
| **ADR-38** | **删除议会模式**(parliament / vote / vote_request / parliament_open),Critic Review 协议(approve / reject / redirect / advise)替代                                                                                                                            | 保留议会                                          | xiaofei #5:议会成本高,单 critic verdict 已经够用                                                                                |
| **ADR-39** | **Multi-CLI runtime 作为本地过渡方案保留**,默认 SINGLE_PROC,等 claw 子 session 能力上线后退役                                                                                                                                                                           | Multi-CLI 作为默认 / 立即删                          | xiaofei #6:本地全功能跑 + 平滑过渡                                                                                              |
| **ADR-40** | **Framework agent / Comm agent 不做**,架构图里的 framework-rebuild action 已删,`comm-optimization` / `compiler-tuning` 作为 P1+ 粗粒度入口保留,**P0 禁用**                                                                                                             | 完整 6 agent                                    | xiaofei:这两个先不做;P0 先跑通 Coordinator + Orchestration + KernelAgent 主链路                                                     |
| **ADR-41** | **删除 IMPLEMENTATION-CHECKLIST.md**                                                                                                                                                                                                                 | 维护 checklist                                  | xiaofei #8                                                                                                            |
| **ADR-42** | **SQLite per-session 直接落 NFS**:每次 run 在 `$SESSION_DIR/storage/coordinator.db` 新建独立 DB,session 结束即废弃,不跨 session/user 复用;DB 直接落 NFS(WekaFS)→ coordinator crash / sandbox 重新分配都自然恢复,**不再需要 backup → restore 两层**;跨 session 经验通过中心化 KB 服务传递                | 长期共享单库 / 跨 session 复用 / 本地盘 + NFS backup 两层   | 简化部署 + 无需 schema migration + 任何 crash 都不丢失 + KB 已覆盖跨 session 知识 + WekaFS 是企业级 POSIX 兼容,WAL 模式可用(需 self-test 验证,见 T20) |
| **ADR-43** | **KB 中心化共享存储 + 按模型分区 + Critic 唯一入口**:KB 是跨 sandbox/session/user 的中心化共享服务,按 `<model_family>/<model_name>` 二级分区;具体载体由 T13 后续落地(NFS 共享路径 / HTTP service);Critic 是唯一 read/write/synthesis 入口,通过 `kb_query.py` / `kb_ingest.py` 包裹后端;**P0 不依赖真实中心化 KB** | 各 session 本地 KB / 多 agent 直接读写 / P0 即实现中心化 KB | 用户明确要求中心化 + 按模型分类 + Critic 做;但中心化 KB 不是当前主链路阻塞项                                                                       |


---

## 26. TODO 跟踪项

### 26.0 P0 落地顺序(当前分支优先级)

当前分支不要求一次性完成 4 个真实 agent。为了复用 Gongzheng MVP 已跑通的 Coordinator/SQLite/Kernel RPC 底座,落地顺序固定为:

1. **先跑通主链路**:Coordinator + Orchestration + KernelAgent
  - Orchestration 能产生 proposal/request
  - Coordinator 能路由 `request{target_agent="kernel"}` / `response`
  - KernelAgent 能完成 `select_kernels` / `run_optimization` / `apply_patch` 的最小闭环
  - SQLite events/tasks/cursors/leases 能记录并支持同 session resume
  - P0 action 范围按 §16.5 allowlist;`comm-optimization` / `compiler-tuning` 禁用
2. **Critic 先 mock adapter**:
  - 低风险 proposal 返回 `advise` / `approve(source="mock")`
  - 高风险 proposal 返回显式 `approve(source="mock")` 或 `needs_review`,不得静默 approve
  - event/task history 必须记录 `source="mock"`
  - 中心化 KB 不做真实接入,只返回空/固定 mock KB hint
3. **Robustness 先 mock adapter**:
  - 只做 heartbeat / basic alert
  - 不主动 `kill_task` / `force_dispatch` / `prune_branch` / `escalate_strategy_change`
  - 调度警察能力等 robustness owner 接入后再启用
4. **真实 Critic / Robustness 接入后**只替换 adapter 和 prompt,不改 Coordinator / Orchestration / KernelAgent 的 wire protocol。


| ID             | TODO                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 负责人                         | 状态                       |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------- | ------------------------ |
| T1             | 跟 sandbox 团队确认是否能跨 sandbox 通信(CPU + GPU 分开)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | xiaofei                     | 未启动                      |
| T2             | 默认单 GPU sandbox 起多个 GPU sandbox 用于并行 backend 测试                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | TBD                         | 未启动                      |
| T3             | ~~KB 跨 user/多 session 隔离策略~~ → **合并到 T13**(中心化 KB 服务原生按 `<model_family>/<model_name>` 分区,user 维度由服务侧 ACL 决定;过渡期可加 user_id 前缀作 namespace)                                                                                                                                                                                                                                                                                                                                                                                                                                             | TBD                         | 合并                       |
| T4             | 验证 Codex no-tools + KB 例外 allowlist 的稳定性、repair 策略和 schema validation;测试 PolicyGate 字符串前缀匹配是否能挡住 `python3 kb_query.py && malicious_cmd`                                                                                                                                                                                                                                                                                                                                                                                                                                              | 实施 agent                    | 未启动                      |
| T5             | KB schema(entries.jsonl + insights.jsonl + conflicts.jsonl + embeddings)的具体 schema 设计;`kb_query` 是否引入 embedding-based recall                                                                                                                                                                                                                                                                                                                                                                                                                                                         | TBD                         | 未启动                      |
| T6             | 新 skill 名最终定稿(占位 `inference_optimizer`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | xiaofei                     | 占位                       |
| T7             | 跨前端(Cursor / ClaudeCode)测试矩阵                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | TBD                         | 未启动                      |
| T8             | SQLite WAL 死锁检测算法(环路检测 / timeout 全局监控)和 multi-lane lease 原子获取测试                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | TBD                         | 未启动                      |
| T9             | 验证 §17.6 evidence-check 规则和崩溃点恢复矩阵是否覆盖所有 19 action                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | TBD                         | 未启动                      |
| T10            | kernel-opt 只优化原生 kernel,不优化 torch.compile 之后的 kernel(理由:1) torch.compile 后的 kernel 不好优化,2) 损失精度的可能性比较大);先记 TODO,不一定哪期做                                                                                                                                                                                                                                                                                                                                                                                                                                                               | xiaofei                     | 未启动                      |
| **T0 (P0)**    | **跑通 Coordinator + Orchestration + KernelAgent 主链路**:基于 Gongzheng MVP 底座,先不等待真实 Critic/Robustness;Critic/Robustness 使用 mock adapter;验收:REQUEST/RESPONSE 路由、KernelAgent 最小闭环、SQLite 入账、同 session resume、final report 能输出                                                                                                                                                                                                                                                                                                                                                                | xiaofei(当前分支)               | 最高优先级                    |
| **T11 (v0.6)** | ~~orchestration → orchestration 代码 rename~~ → **细化到 T19**(具体文件清单)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | xiaofei(本分支整体开发)            | 合并到 T19                  |
| **T12 (v0.6)** | **Critic Review 协议(§18)落地**:`CriticReviewGate` 实现 + `REVIEW_VERDICT` intent + 5 verdict(`approve/reject/redirect/advise/needs_review`) + sample-down + Brier 集成;P0 阶段先接 mock adapter,不阻塞 T0                                                                                                                                                                                                                                                                                                                                                                                          | critic 负责人(其他人)+ 先 mock     | P0 用 mock                |
| **T13 (v0.6)** | **Critic 接管中心化 KB(非 P0)**:(a) `kb.py` 接通 Critic;Codex no-tools + KB 例外 Bash allowlist 落地;(b) **中心化 KB 后端选型**:Option A(`/wekafs/kb/<model_family>/<model_name>/entries.jsonl` + 文件锁,过渡)/ Option B(HTTP service + REST + embedding recall,目标);(c) `kb_query.py` / `kb_ingest.py` 抽象成 client,后端可切换;(d) schema:`{run_id, user_id, model_family, model_name, action, lesson, gain_pct, status, tags, predicted_gain_pct, ts}`;P0 使用空/固定 mock KB hint                                                                                                                                      | critic 负责人(其他人)+ 先 mock     | 后续,不阻塞 T0                |
| **T14 (v0.6)** | **Robustness always-on + RCA + Handle**:`robustness_intervention.py` + `do_rca` + 4 个调度警察 intent payload schema;P0 阶段只启 heartbeat/basic alert mock,不启调度警察动作,不阻塞 T0                                                                                                                                                                                                                                                                                                                                                                                                                   | robustness 负责人(其他人)+ 先 mock | 部分已实现(Phase G),P0 用 mock |
| **T15 (v0.6)** | **Kernel agent 5 个 action ownership 落地**:`policy.KERNEL_OWNED_ACTIONS` 把 deep-kernel-analysis / operator-tuning / vendor-kernel-config 加入(目前只有 kernel-opt / integrate);Kernel agent 的 action markdown 补 3 个                                                                                                                                                                                                                                                                                                                                                                          | xiaofei(kernelAgent 分支)     | 进行中                      |
| **T16 (v0.6)** | **Multi-CLI 默认关闭**:`cli.py` 默认 `--transport single-proc`;长任务 warning;claw 子 session 能力跟进                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | xiaofei                     | 未启动                      |
| **T17 (v0.6)** | **单一全模式兼容迁移**:第一步不直接删除 `execution_mode.py` / `feature_flags.py`,而是把所有 mode 映射到同一个 full runtime config,`allowed_modes` 字段先忽略但保留兼容;T0 e2e 跑通后再删除 mode/feature_flags 文件和对应测试                                                                                                                                                                                                                                                                                                                                                                                                            | 整体开发                        | T0 后执行                   |
| **T18 (v0.6)** | **议会相关代码彻底清理**:`vote` / `parliament_open` / `vote_request` / `_open_parliament` / `_record_vote`;`objection` 改成 `topic="advice"` 软降级                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 整体开发                        | 未启动                      |
| **T19 (v0.6)** | **orchestration → orchestration 代码 rename 文件清单**(本分支整体改,不单独 PR):(a) `src/inference_optimizer/orchestrator/system_prompts/orchestration.md` → `orchestration.md`;(b) `src/inference_optimizer/agents/orchestration/` → `agents/orchestration/`;(c) `agent_role.py` 里 `AgentName.EXECUTOR` → `ORCHESTRATION` + `default_role_registry` key + `roles_for_run` / `roles_for_mode` 引用;(d) `policy.py` 里 `EXECUTOR_INTENTS` / `KERNEL_OWNED_BY`_* 等常量;(e) `coordinator.py` / `agent_card.yaml` / 所有测试 / 所有 `system_prompts/*.md` 提及;(f) `multi_cli/*.py` agent name 字典;(g) docs:`docs/README.md` / 本设计文档已更新 | xiaofei(本分支整体开发)            | 未启动                      |
| **T20 (v0.6)** | **WekaFS SQLite WAL self-test**(ADR-42 前置):落地前必须跑通(a) 多进程并发 INSERT / SELECT(模拟 Coordinator + multi-cli agents 并发 events 写入);(b) `BEGIN IMMEDIATE` 跨表事务原子性;(c) 模拟 sandbox 重启:启动 → 写 → kill -9 → 重启 → 验证 wal/shm 自动 recover;(d) 长跑(2h+)无 lock 升级失败;失败则 fallback 回 "本地盘 + 30min NFS VACUUM INTO backup + restore"两层方案,且更新 §3.5.7 / §17 文档                                                                                                                                                                                                                                                 | xiaofei(本分支整体开发)            | 未启动                      |


---

**End of Design v0.6 Final**