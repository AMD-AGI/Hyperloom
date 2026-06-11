# Hyperloom Long-Run 补充改造方案：轨迹回顾 + profiler 结构化反馈

> 目标：在 long-run 优化主循环里补两块能力（对标 NVIDIA AVO, arXiv:2603.24517）：
> - **A. 回顾整条轨迹、在停滞时主动给新方向**（对标 AVO 的 self-supervision / supervisor）。
> - **B. 把 profiler 反馈结构化喂回 prompt**（对标 AVO "profiler 是灵魂"，按瓶颈聚焦 + 跨版本 delta）。
>
> **硬约束（本轮第一原则）：最小改动。最大限度复用已有设施，只做必要的新增。**
> 凡能复用现有 loader / advisory 块 / context tool / 映射表的，一律不新造。
>
> 适用范围：`inference_optimizer/orchestrator/`。
> 状态：设计待实施。与 `future_plan.MD` 互补，不冲突（本方案全部为 advisory，不动 phase 状态机 / 终态集合）。

---

## 0. 为什么这两块值得做（AVO 对照）

AVO 能 7 天无人干预持续超越 cuDNN / FA4，核心两点：

1. **supervisor 在停滞时回顾整条进化轨迹、重定向探索**——检测"卡死"与"原地打转"两种失败模式，触发后 review 全 trajectory 给出若干候选方向。
2. **agent 持续读 profiler 反馈定位瓶颈**——"对比多个历史实现的 profiling 特征"，每个增益都先 profile 出具体瓶颈再改，而非凭理论猜。

本项目现状：
- plateau 处理是**统计式、局部、被动**的（只报数字，不读历史、不给方向）。
- profiler 数据**已存在结构化形态**，但 prompt 里是整篇 `analysis_md` verbatim 注入（大、噪声多、无跨版本 delta、无瓶颈→lever 映射）。

这两点正是与 AVO 的差距，也是本方案要补的。

---

## 1. 复用清单（实施前先认清"已经有什么"）

> 本方案能"最小改动"的前提：下列设施全部已存在，**只读复用**或**小幅扩展**，不重写。

| 能力 | 已有设施 | 文件 / 符号 |
|---|---|---|
| 完整优化谱系（KEEP/REVERT/no_promote + gain_pct + kind + change + reason） | `Journal` | `orchestrator/optimization_journal.py::Journal.load_or_create` / `entries` |
| 探索 ledger（winners + rejected，带 gain） | `explore_search` | `shared_state.py`（`_format_explore_search`） |
| 结构化 gap（含 attempts 计数） | `gaps[]` | `shared_state.py::to_gaps_summary` |
| roofline 历史轨迹（compute/idle/comm pct、bound_kind、top bottleneck） | `roofline_snapshots` + parsers | `shared_state.py::roofline_snapshots`、`orchestrator/roofline_snapshot.py` |
| 当前主瓶颈 | `current_top_bottleneck()` | `shared_state.py` (~954) |
| plateau 检测（卡死 / 原地打转两种模式） | `compute_plateau_explore/kernel/framework_pr` | `orchestrator/phase_state.py` |
| advisory 块的标准写法 + 注入点 | 多个 `_*_advisory_block` | `coordinator.py::_compose_prompt` (~4207)；`research_hints.summarise_for_prompt` |
| 瓶颈方向 → specialist domain 映射 | `_BOTTLENECK_DOMAIN_HINTS` | `coordinator.py` (~5917) |
| **整篇 analysis_md 的 pull-on-demand 工具（已存在！）** | `show_analysis_md` context tool | `backends/mcp_context_tools.py` (~196) |
| profiler 结构化摘要源 | `last_trace_analyze`（hot_kernels/reusable）+ roofline 快照 | `shared_state.py` (~373) |
| 一次性子 agent 调用（LLM reviewer 复用） | specialist delegate 路径 | 现有 `specialist` action（不新增 agent 类型） |

**结论**：A 只需新增 1 个纯函数文件 + 1 个 advisory 块 + 复用 plateau 作触发；B 只需新增 1 个 digest 函数 + 改 1 处注入内容（pull 工具已存在）。无新增状态机、无新增终态、无新增 agent 类型。

---

## 2. 工作流 B：profiler 反馈结构化喂回 prompt（先做，风险最低）

> 先做 B 的原因：完全独立、零行为耦合、立刻提升每 tick 决策质量；A 的轨迹回顾也会引用 B 的结构化瓶颈。

### B1. 新增 `build_profiler_digest()`（结构化瓶颈摘要 + 跨版本 delta）

- **落点**：`orchestrator/roofline_snapshot.py` 内新增 `build_profiler_digest(snapshots, trace_analyze) -> str`。
  - 选此文件因为**解析基础设施已在此**（`_parse_executive_table` / `extract_workload_summary` 等），不另起文件。
- **输入**：`SharedState.roofline_snapshots`（取最近两张做 delta）+ `SharedState.last_trace_analyze`（hot_kernels / reusable ids）。**全部已有，只读。**
- **输出**（紧凑、瓶颈聚焦，带跨版本 delta）：
  ```
  === Profiler digest (structured) ===
  bound_kind=memory  compute=61%  idle=22%  comm=17%
  delta_vs_prev: compute +3.1pp  idle -2.4pp  comm -0.7pp   ← 上一改动的 profiling 影响
  top_bottlenecks:
    1. MoE_fused   28.8% gpu  (prev 31.2%, -2.4pp)  lever→ kernel_switch_specialist
    2. all_reduce  12.1% gpu  (prev 11.8%, +0.3pp)  lever→ comm_specialist
  reusable_native_kernel_ids=[k001,k004,...]
  ```
- **delta 是关键差异点**：逐项 vs 上一张快照的差值（pp = 百分点），让 LLM 看见"上轮改动把 compute bound 推高了 / 没动 idle"。这就是 AVO agent "对比 profiling 特征"的最小可用版。
- **最小化**：纯函数、无状态、fail-soft（解析不出返回 `""`，沿用本文件 best-effort 风格）。

### B2. 瓶颈 → lever 映射（提升复用，不新造）

- 把现有 `coordinator.py::_BOTTLENECK_DOMAIN_HINTS`（comm/compute/memory/idle/host_overhead → specialist domain）**移到模块级公共常量**（建议放 `roofline_snapshot.py` 或 `phase_state.py`），让 B1 digest 与 coordinator 的 `_bottleneck_redirect_advisory_block` 共用同一张表。
- **只搬不改语义**：移动后 coordinator 原引用改为 import；映射内容不变。

### B3. 默认注入瘦身：结构化 digest 常驻 + 整篇 md 转 pull-on-demand

- **现状**：`shared_state.py::_format_analysis_md_full`（~3214）把整篇 TraceLens md verbatim 注入每个 full SEED——long-run 下极耗 context。
- **改动（极小）**：
  1. `to_prompt_summary` 里把 `analysis_md=<整篇 verbatim>` 替换为 `profiler_digest=<B1 输出>`。
  2. 整篇 md 不再常驻；**pull 工具 `show_analysis_md` 已存在**，无需新增——只在 digest 末尾加一行提示 "整篇报告用 `show_analysis_md` 拉取"。
- **分两步降风险**：
  - **Step B3a（先）**：digest 与 verbatim md **并存**（digest 叠加在前），验证 digest 内容正确、LLM 会用。
  - **Step B3b（后）**：确认 B3a 有效后，再把 verbatim 撤成 pull-on-demand。
- **回归保护**：B3b 是行为改动（orchestration 默认看到的 context 变了），用开关控制（env gate，默认先维持 verbatim，灰度切换）。

---

## 3. 工作流 A：回顾整条轨迹、停滞时主动给新方向

### A1. 新增 `trajectory_reviewer.py`（确定性轨迹综合器，A2-a）

- **新文件**：`orchestrator/trajectory_reviewer.py`（唯一新增逻辑文件）。
- **唯一公共函数**：`build_trajectory_digest(session_dir, shared_state) -> str`（纯函数、只读、fail-soft、返回 markdown，空则 `""`）。
- **输入（全部已有，只读）**：
  - `Journal.load_or_create(session_dir, ...)` 全量 entries —— \(\mathcal{P}_t\) 谱系。
  - `shared_state.explore_search`（winners + rejected）。
  - `shared_state.roofline_snapshots`（历史轨迹）。
  - `shared_state.gaps[]`。
- **输出**（对标 AVO supervisor 的 "review trajectory → several candidate directions"）：
  ```
  === Trajectory review (advisory) ===
  progress: best_gain 3.2% → 3.3% over last 4 cycles (stalled 3 cycles)
  exhausted_directions (反复未过线，勿重复):
    - MoE kernel rewrite: 5 attempts, all REVERT (max gain 0.4%)
    - kv-cache-dtype fp8: 3 no_promote
  under-explored (高 bound 低投入):
    - comm bound 17% but only 1 attempt → try comm_specialist
  candidate_directions:
    1. <方向 + 依据(引用 journal/roofline)>
    2. ...
  (advisory only; does not gate phase advance)
  ```
- **A2-a 内部逻辑全是确定性聚合**（无 LLM）：
  1. 进度曲线：从 journal/`winners_history` 取 best-gain 序列 + 停滞 cycle 数。
  2. 枯竭方向：按 `kind`+`change` 聚类 journal 里反复 REVERT/no_promote 的簇。
  3. 欠探索方向：roofline 主 bound 方向 × 该方向历史尝试次数，挑"高 bound 低尝试"。
  4. 候选方向：由 2/3 + B2 的 lever 映射推导，每条带证据。
- **最小化**：单文件、单函数、纯聚合，可独立单测，零外部依赖。

### A2-b. 可选 LLM reviewer（复用 specialist，不新增 agent 类型）

- **要做**（用户确认）。但**复用现有 specialist 子 agent 调用路径**，不新建 agent 框架：
  - 触发时，以 A2-a 的 digest 作为上下文，走现有 specialist delegate 一次性调用（新增一个轻量 domain 名，如 `trajectory_reviewer`，或复用最贴近的现有 domain），让它读全轨迹产出更高质量方向。
  - 产出回填进同一个 `=== Trajectory review (advisory) ===` 块（替换/增强 A2-a 的 `candidate_directions`）。
- **gated + 默认 off**：env 开关（如 `INFERENCE_OPTIMIZER_TRAJECTORY_LLM_REVIEW`），默认关。先上 A2-a（零成本），再灰度开 A2-b。
- **节流**：只在 plateau 触发时调一次，避免侵占探索预算（= AVO 的 conditional intervention）。

### A3. 触发与注入（复用现有 plateau + advisory 注入点）

- **触发 = 复用现有 plateau 检测**：`_phase_state.compute_plateau_explore/kernel/framework_pr` 已能识别"卡死/原地打转"两种 AVO 失败模式。
  - 额外加一个**跨 cycle 停滞计数**触发（连续 M 个 macro-cycle 无 best 提升，M 默认 2）。该计数可由 A1 内部从 `winners_history` 推导，**不新增 SharedState 字段**。
- **注入点 = 复用 `_compose_prompt`**：在 `coordinator.py::_compose_prompt`（~4207 plateau_block 之后）新增一段调用 `trajectory_reviewer.build_trajectory_digest(...)`，与现有 advisory 块同样的 try/except fail-soft 包裹、同样的 `sections.append("=== ... ===")` 写法。
- **节流**：digest 仅在 plateau 活跃 / 停滞触发时重算并注入；非 plateau tick 不挂，省 token。
- **advisory 语义**：明确标注 "does not gate phase advance"，与 `future_plan.MD` I3 决策（plateau 非终态、纯 phase-exit reason）完全一致。

---

## 4. 实施顺序与依赖

1. **B2**（搬 `_BOTTLENECK_DOMAIN_HINTS` 到公共常量）—— 1 处移动，A/B 都依赖。
2. **B1**（`build_profiler_digest`）—— 纯函数 + 单测。
3. **B3a**（digest 叠加注入，verbatim 保留）—— 改 `to_prompt_summary` 1 处。
4. **A1 + A2-a**（`trajectory_reviewer.py` 确定性档）—— 新文件 + 单测。
5. **A3**（接 plateau 触发 + `_compose_prompt` 注入）—— coordinator 加 1 个 advisory 块。
6. **B3b**（verbatim md → pull-on-demand，env 灰度）—— 验证 B3a 后再做。
7. **A2-b**（LLM reviewer，复用 specialist，env 默认 off）—— 最后，按预算决定开关。

> 每步独立可测、可回滚。前 5 步全是叠加式（新增 advisory / 新增 digest），不改既有行为；只有 B3b 是默认行为变更，单独 gate。

---

## 5. 测试计划（要点）

- **B1 digest 单测**：给定两张 roofline 快照 + trace_analyze，断言 bound_kind / 各 pct / `delta_vs_prev`（pp 计算正确）/ top_bottlenecks lever 映射 / reusable ids；空输入 → `""`。
- **B2 映射搬迁**：断言 coordinator 与 roofline_snapshot 引用同一张表、内容不变（comm/compute/memory/idle/host_overhead 五项）。
- **B3a 注入**：`to_prompt_summary` 含 `profiler_digest=`；verbatim 仍在（B3a 阶段）。
- **B3b 灰度**：开关 on → 默认不注入整篇 md、含 "用 show_analysis_md 拉取" 提示；off → 维持现状（零回归）。
- **A1 确定性 digest**：构造 journal（5 次 REVERT 同一 kind）→ 命中 exhausted_directions；构造 roofline comm 高 + 尝试少 → 命中 under-explored；空 journal → `""`。
- **A3 触发**：plateau 未触发 → 不注入；触发 → 注入且块尾含 "does not gate"；断言不进 `STOP_REASON_VOCAB`、不改 phase。
- **A2-b gate**：env off → 不调 LLM（纯 A2-a）；env on + plateau → 调一次且结果回填同一块。
- **短 run 零回归**：`--max-hours 2`，plateau 基本不触发 → A 不注入；B3 维持 verbatim（B3b 默认 off）→ 与改造前一致。

---

## 6. 决策记录（取舍与拍板）

- **S1（先 B 后 A）**：B 独立零耦合且 A 引用 B 的结构化瓶颈，故 B 先行。
- **S2（最小新增）**：仅新增 1 个逻辑文件（`trajectory_reviewer.py`）+ 1 个 digest 纯函数（落在已有 `roofline_snapshot.py`）。其余全部复用：plateau 检测作触发、`_compose_prompt` 作注入、`show_analysis_md` 作 pull、specialist 路径作 LLM reviewer、`_BOTTLENECK_DOMAIN_HINTS` 作 lever 表。
- **S3（pull 工具不新造）**：`show_analysis_md` 已存在，B3 只改"默认注入什么"，不加新工具。
- **S4（A2-b 复用 specialist）**：LLM reviewer 不新建 agent 类型，复用 specialist delegate 一次性调用，env-gated 默认 off。
- **S5（全 advisory）**：A/B 所有输出均为 advisory，不 gate phase、不进终态集合，与 `future_plan.MD` I3 一致。
- **S6（不新增 SharedState 字段）**：停滞计数由 A1 从 `winners_history` 即时推导，避免扩状态/迁移成本（除非 A2-b 落地时确需缓存 reviewer 结果，届时再评估）。
- **S7（delta 只取最近两张快照）**：跨版本 delta 取 N 与 N-1，足够定位"上一改动的 profiling 影响"，不做全历史 diff（控复杂度）。

---

## 7. 受影响文件清单（落地索引）

| 项 | 主要文件 / 符号 | 改动性质 |
|---|---|---|
| B1 | `orchestrator/roofline_snapshot.py`: 新增 `build_profiler_digest()` | 新增纯函数 |
| B2 | `coordinator.py`: `_BOTTLENECK_DOMAIN_HINTS` 移至公共常量（建议 `roofline_snapshot.py`/`phase_state.py`）+ 原处改 import | 移动 |
| B3a | `shared_state.py`: `to_prompt_summary`（注入 `profiler_digest`）；`_format_analysis_md_full` 保留 | 改 1 处注入 |
| B3b | `shared_state.py`: `_format_analysis_md_full` 默认撤为 pull 提示 + env gate；`mcp_context_tools.py` 的 `show_analysis_md` **复用不改** | 行为变更（gated） |
| A1/A2-a | `orchestrator/trajectory_reviewer.py`: 新增 `build_trajectory_digest()` | 新增文件 |
| A3 | `coordinator.py`: `_compose_prompt`（~4207 后新增 advisory 块 + plateau/停滞触发） | 加 1 个 advisory 块 |
| A2-b | `coordinator.py`: 触发处复用 specialist delegate；env gate `INFERENCE_OPTIMIZER_TRAJECTORY_LLM_REVIEW` | 复用现有路径 |

---

## 8. 与 AVO 的对应关系（自检）

| AVO 机制 | 本方案对应 |
|---|---|
| supervisor 检测 stall / 原地打转 | A3 复用 plateau 检测 + 跨 cycle 停滞计数 |
| supervisor 回顾整条轨迹 → 候选方向 | A1 `build_trajectory_digest`（读 journal/explore/roofline 全量）+ A2-b LLM reviewer |
| agent 对比多版本 profiling 特征定位瓶颈 | B1 digest 的 `delta_vs_prev`（跨版本逐项 delta） |
| profiler 反馈驱动每个增益 | B1/B3 把瓶颈聚焦的结构化 profiler 常驻 prompt |
| conditional intervention（仅停滞时介入） | A3 节流 + A2-b 仅 plateau 调一次 |
