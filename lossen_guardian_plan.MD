# Loosen Guardian Plan — 把"过度防御"还给 LLM

> 目标:在**不破坏本项目目的**(自主优化推理、产出可恢复/可下游消费的产物)的前提下,
> **剂量地放权**,让 Orchestration LLM 尽可能多、尽可能深地探索。
>
> 核心准则:**代码只做"不变量",不做"策略"。**

---

## 0. 判定准则:什么是"不变量",什么是"策略"

| | 不变量 (Invariant)——代码保留,确定性 | 策略 (Strategy)——交还 LLM,或降级为纯 advisory |
|---|---|---|
| 定义 | 违反它会破坏**安全 / 正确性 / 资源 / 可恢复性 / 产物契约 / 去重**;与"优化得好不好"无关 | 关于"下一步做什么、做多久、走哪条路"的**判断**;做错了只是优化效率低,不会损坏系统 |
| 例子 | 路径沙箱、KB 单写者、kernel patch 必须过 Critic、sweep 单例(防并发崩引擎)、GPU 租约、baseline 门、终止态→CLOSE、幂等去重 | 何时切 phase、explore 探多少轮、先 config 还是先 code、何时重测、选哪个 gap、specialist 并行度 |
| 处理 | **保留** | **删除 / 降级为 advisory(只进 prompt,绝不 deny)/ 放宽上限** |

一句话:**凡是 deny 的地方,问一句"这条规则保护的是系统安全,还是我替模型做了决定?"——后者一律降级。**

---

## 1. 总览:保留 vs 放开

### 1.1 必须保留的不变量(不动)

| 机制 | 位置 | 理由 |
|---|---|---|
| 角色→intent 权限矩阵 | `policy.py` 749–756 | 身份/权限安全 |
| 路径沙箱(session_dir / framework 源根) | `policy.py` `_validate_payload_paths` 2518–2546 | 文件系统安全 |
| KB 单写者 | `policy.py` `_validate_no_kb_write_collision` 1408–1419 | 数据一致性 |
| `integrate_patch` 必须有 Critic verdict | `policy.py` 1825–1889 | patch 安全门 |
| `sweep`/`conc_sweep` 单例 | `policy.py` 1730–1747, 1788–1802 | 防并发把两个引擎都打挂(资源) |
| `fp8_only_action` | `policy.py` 1373–1382 | 精度正确性 |
| `kill_task` 仅 robustness + scope 限制 | `policy.py` 2436–2456 | 进程安全 |
| specialist intent 面(R3)/ payload schema | `policy.py` 2281–2434 | 结构正确性 |
| core-state 单写者 | `policy.py` `_validate_state_transition` 1084–1098 | 防篡改 |
| dynamic 红线副作用(kernel/metric/server/magpie) | `policy.py` 2243–2254 | 越权安全 |
| 终止态 → CLOSE / `skip_to_close` | `phase_state.py` `_global_terminal` 955–984 | 确定性收尾 |
| baseline 门(`baseline_tput>0` 才出 PRELUDE) | `phase_state.py` 1002–1020 | 没有 baseline 整个 mission 无意义 |
| `baseline_failure_streak>=3` → abort | `phase_state.py` 1023–1038 | 防无限失败 |
| warm-replay 在飞 / EXPLORE IR-6 硬 force-exit / phase 预算耗尽硬墙 | `phase_state.py` 988–999, 581–659, 1169–1175 等 | 资源 + 必须留时间写产物 |
| phase 链单调拓扑 + 优先级(abort>terminal>normal) | `phase_state.py` 1567–1577 | 可恢复性 / 审计 vocab |
| baseline 自环去重(完全相同 fingerprint) | `coordinator.py` `_baseline_self_loop_denial` 4975–5046 | 反 loop + 去重(用户明确要保留) |
| roofline/profile 由 Coordinator 拥有调度 | 内部 enqueue 通道 | 防并发 Magpie / GPU 争用(资源) |
| CLOSE 5 步顺序器 + 安全网 breakdown | `coordinator.py` `_on_enter_close` 3291–3474 | 产物契约 + 可恢复性 |
| 自动 analysis enqueue(PRELUDE/watermark)的**enqueue 本身** | `coordinator.py` 2599–2709, 1987–2020 | 资源调度(只去掉它的 gating 副作用,见 §3.4) |

### 1.2 要放开/降级的策略(本计划主体)

分 5 类,详见 §2–§6。

---

## 2. 类别 A:边界"所见即所得"(WYSIWYG)

> 问题:展示给 LLM 的 `allowed_actions` 是个**超集**,含 `profile`/`roofline`/`framework_pr`,
> 但 LLM 一提就被**另一条规则**(`analysis_action_not_llm_proposable` /
> `framework_pr_action_not_llm_proposable`)拒,而不是 R1。模型看到的"可用集"是假的。

| # | 机制 | 位置 | 当前 | 改动 |
|---|---|---|---|---|
| A1 | `PHASE_ALLOWED_ACTIONS` 含 `roofline`/`profile`(PRELUDE/FW_PR/EXPLORE/KERNEL) | `phase_state.py` 78–152 | 超集,既给内部 enqueue 用又展示给 LLM | 新增 `PHASE_LLM_PROPOSABLE_ACTIONS[phase] = PHASE_ALLOWED_ACTIONS[phase] - INTERNAL_ONLY_ACTION_NAMES`(后者已存在于 `action_surfaces.py` 24–28:roofline/profile/replay_warm_recipe + framework_pr)。内部 enqueue 走直连 `tasks.create_or_return_existing`(`coordinator.py` 2673–2701),**根本不过 PolicyGate**,所以超集对内部通道并非必需 —— 可安全收窄 LLM 视图。 |
| A2 | 静态 prompt 渲染 phase 允许集 | `prompt_builder.py` `_section_phase_semantics` 149–168 | 渲染 `PHASE_ALLOWED_ACTIONS` | 改渲染 `PHASE_LLM_PROPOSABLE_ACTIONS` |
| A3 | 每-tick `=== Phase-allowed actions ===` 块 | `shared_state.to_phase_status_summary`(被 `coordinator.py` 4522 调用) | 同样源自超集 | 同样改用 LLM 可提集 |
| A4 | R1 与 analysis 拒绝的二义性 | `policy.py` `_validate_phase_action` 1288–1346 + `_validate_action_not_llm_proposable` 1219–1269 | 两条规则、两套 reason-code | 让 R1 成为"不在你的集合里"的**唯一**面向 LLM 的拒绝来源(对照 `PHASE_LLM_PROPOSABLE_ACTIONS`)。`analysis_action_not_llm_proposable` 退化为冗余,可保留为更友好的报错文案,但 reason-code 与展示集保持一致。 |
| A5 | **不再静默改写 intent 类型** | `coordinator.py` 6292–6314 | `delegate{explore, grid}` 被静默重路由成 `propose_action` 走 Critic | 二选一:(a) 文档与 EMIT hint 直接告诉模型"带 grid 的 explore 用 `propose_action`",取消静默改写;或 (b) 保留 Critic 评审但**不改 intent 语义**——用一条内部"explore grid 需 Critic"标记,让 verdict_map 流程在 `delegate` 语义下完成。推荐 (a):最透明。 |

**风险**:低。纯属"让展示与强制一致"。

---

## 3. 类别 B:删除/降级"代替 LLM 思考"的层

### 3.1 `_required_next_step`(执行清单)
- 位置:`coordinator.py` 4808–4973,注入点 4612–4615。
- 现状:**ADVISE**(已不 deny)。但它用命令式("下一步必须做 X")替模型排好了序。
- 改动:**保留为纯事实**,去掉命令式措辞。把它从"Coordinator-enforced 清单"改名为 `=== 状态事实(供参考) ===`,只罗列客观事实(如"有 1 个未集成的 kernel KEEP""stack 有未验证 KEEP"),**不给出"你应该做 X"的结论**。结论交给模型。

### 3.2 Session Steward(最重要的一处放权)
- 位置:`coordinator.py` `_maybe_enqueue_steward` 3849–3869、`_enqueue_internal_steward_task` 3553–3641、`_route_steward_verdict` 8704–8903、depth-gate 改写 `_apply_depth_gate_to_verdict` 8502–8564;`phase_state.py` 配套 1110–1168、1179–1249。
- 现状:steward 的 `stop_session`/`advance_to_kernel` **直接 driving phase**(经 `pending_escalate_hint`,`coordinator.py` 8844–8863 → `phase_state.py` 1083–1097/1281–1285);并且 `_apply_depth_gate_to_verdict` 会**改写 steward 自己的 LLM 判断**。这是"代码替 LLM 做战略决定"的典型。
- 改动(剂量放权,二选一):
  - **温和版**:保留 steward 作为"第二意见",但其 recommendation **降级为 advisory** —— 注入 prompt 的 `=== Steward 建议(advisory) ===`,**不再写 `pending_escalate_hint`**、**不再驱动 phase**。删除 `_apply_depth_gate_to_verdict` 的改写(8502–8564)。phase 仍只由 §1.1 的硬信号 + LLM 显式 escalate hint 推进。
  - **激进版**:直接停用 steward 内部 enqueue(`_maybe_enqueue_steward`),由 LLM 在 plateau 事实面前自行决定继续/前进(配合 §4 plateau 降级)。
- 配套:EXPLORE plateau 不再"return None 等 steward"(`phase_state.py` 1152–1162),改为"plateau 是 advisory 事实,LLM 不显式前进就继续留在 EXPLORE 直到硬预算墙"。

### 3.3 priors-match / target-gap / proposal-scores 三个 advisory 块
- 位置:`coordinator.py` `_priors_match_advisory_block` 8481–8500、`_target_gap_advisory_block` 8374–8410、`to_proposal_scores_summary`(注入 4659–4690)。
- 现状:**已是纯 ADVISE,不 deny,不 gate**。符合"纯 advisory"目标。
- 改动:**保留**即可(它们不违反准则)。仅做两件小事:
  1. 确保措辞是"参考信号之一",无任何"应优先 X"的强制语气(现状基本满足)。
  2. **去重**:`intervention_mix` 块在一个 tick 内被注入**两次**(`coordinator.py` 4616–4623 与 4692–4701)——删掉其中一处。

### 3.4 ProposalScorer
- 位置:打分 `coordinator.py` 8206–8232,渲染 4667–4678。
- 现状:纯 ADVISE。保留。

**风险**:中(steward 那条)。steward 现在承担"防止 EXPLORE 无限探下去"的职责;降级后需依赖 §1.1 的硬预算墙(IR-6 / phase 预算耗尽)兜底——这些**保留**,所以不会失控。

---

## 4. 类别 C:把 phase / 顺序的"软"约束降级为 advisory

> 准则:phase **拓扑**和**硬预算墙**是不变量(保留);"什么时候算探够了"是策略(降级)。

| # | 机制 | 位置 | 当前 | 改动 | 风险 |
|---|---|---|---|---|---|
| C1 | EXPLORE plateau 判据 | `phase_state.py` `compute_plateau_explore` 749–850 | 触发后(经 steward)自动前进 | 计算结果**只作为事实注入 prompt**("系统认为已 plateau");是否前进交给 LLM(显式 hint)或硬预算墙。不再自动 driving。 | 中 |
| C2 | KERNEL plateau(3 REVERT / KEEP 增益<0.5%) | `phase_state.py` `compute_plateau_kernel` 853–951 | 自动 KERNEL→SWEEP | 同 C1:降级为 advisory 事实;硬墙(预算耗尽 1315–1322)兜底 | 中 |
| C3 | GEMM 完成即判 `plateau_kernel` 捷径 | `phase_state.py` 1290–1302 | 自动结束 KERNEL | 降级:GEMM 完成只是事实,不强制结束 | 低 |
| C4 | FRAMEWORK_PR plateau(3 批<1%)/ 0.6 wall | `phase_state.py` 1467–1525 | 自动退出 | plateau 降级为 advisory;0.6 wall(留 60% 给下游)作为**软**预算可保留但建议下调阈值/可配置 | 低 |
| C5 | `_sequence_denial_for_action` 中的策略性顺序 | `coordinator.py` 5139–5313 | **DENY** | 见下分项 | 高 |
| C6 | `_sequence_denial_for_request` | `coordinator.py` 5315–5368 | **DENY** | 见下分项 | 中 |
| C7 | `_auto_roofline_pending_denial` / `_roofline_denial_for_action` | `coordinator.py` 2711–2772 | **DENY**(`wait_for_auto_roofline`) | 降级为 advisory:analysis 在飞时**允许**模型基于旧 `analysis.md` 行动(资源争用由 GPU 租约兜底,不需要 deny)。 | 中 |
| C8 | `explore_attempts_minimum_before_kernel_opt` | `policy.py` 1075–1082 | **DENY**(没探过就不让 kernel_opt) | **删除**:让模型自行决定先 explore 还是直奔 kernel | 中(成本) |
| C9 | `tool_whitelist_phase`(Web 工具仅 EXPLORE) | `policy.py` 1460–1473 | **DENY** | 放宽:研究类工具在更多 phase 可用 | 低 |

#### C5 细分(`_sequence_denial_for_action` 5159–5312)
| 子规则 | 性质 | 改动 |
|---|---|---|
| `target_analysis` 最先 | 策略(且很廉价) | 降级为 advisory 或删除 |
| `baseline` 先于一切 | **不变量**(无 baseline 无意义) | **保留** |
| `last_profile_trace` required(除 profile 外都挡) | 策略("何时重测"应归 LLM) | 降级为 advisory:允许在无新 trace 时行动 |
| KEEP 后强制 `integrate`(只放行 integrate/report) | 策略 | 降级为 advisory("有未集成 KEEP"作为事实) |
| `report` 被未跑的热 kernel 挡(`hot_kernel_unfinished`) | 策略(质量门) | 降级为 advisory;若要保质量,改为在 report 里**标注**未尝试 kernel,而非 deny |
| `stack_rebench_required`(只放行 explore/baseline/report) | 策略,但关联"已验证增益"产物诚实性 | 降级为 advisory;`session_breakdown` 中对未验证 KEEP 显式标 `validated=false`,保持下游契约诚实 |

#### C6 细分(`_sequence_denial_for_request` 5315–5368)
| 子规则 | 性质 | 改动 |
|---|---|---|
| `baseline` 先于 kernel request | 不变量(需基准) | 保留 |
| `last_profile_trace` 先于 kernel request | 策略 | 降级 advisory |
| `trace_analyze` 先于其它 kernel kind | **数据依赖**(run_optimization 消费 trace_analyze 的 candidates_path) | **保留**,但改为由 kernel handler 做输入校验后优雅失败,而非前置硬 deny(等价但更"所见即所得") |
| `run_gemm_tuning` 先于 `run_optimization`(FP8) | 策略 | 降级 advisory |

---

## 5. 类别 D:放开探索广度的硬上限(让模型探得"多")

> 唯一应保留的硬上限是**资源派生**的(GPU 数);其余 breadth cap 是策略。

| # | 机制 | 位置 | 当前 | 改动 |
|---|---|---|---|---|
| D1 | `research_lane_capacity` 默认 4 | `shared_state` CORE_STATE_FIELDS;ceiling `research_lane_ceiling()` `policy.py` 225–234(`2×GPU`) | 默认 4,封顶 2×GPU | 提高默认值(如 = ceiling),把 **2×GPU 资源上限**作为唯一硬约束(资源不变量,保留) |
| D2 | `explore_specialist_grid_max_one`(每轮 specialist 变体数 ≤ research_lane_capacity) | `policy.py` `_validate_explore_grid_size` 1623–1642 | **DENY** | 上限直接绑到 GPU ceiling;移除"= research_lane_capacity"的二次收窄 |
| D3 | `MAX_DYNAMIC_SOURCED_VARIANTS=1` / `dynamic_sourced_variant_cap_exceeded` | `policy.py` 304, 1651–1662 | **DENY** | 提高或移除(保留红线副作用即可) |
| D4 | `MAX_DYNAMIC_PER_ROUND=1` / `dynamic_round_cap_exhausted` | `policy.py` 303, 2261–2272 | **DENY** | 提高/移除每轮 1 次的限制 |
| D5 | `dynamic_scope_too_narrow`(≥2 域) | `policy.py` 2169–2180 | **DENY** | 放宽到 ≥1 域(让单域深挖也能用 dynamic_action) |
| D6 | specialist `sub_kind` 受限于目录 | `policy.py` 1973–1983 | **DENY** | 放宽:允许自由 sub_kind(域 tag 仍校验) |
| D7 | `DEFAULT_PHASE_BUDGET_PCT`(EXPLORE 40% / KERNEL 35% …) | `phase_state.py` 278–293 | 软预算分配 | 改为可配置 + 调高 EXPLORE/KERNEL 比例,或让 LLM 通过 escalate hint 借调(已有 +5pp/80% 上限机制 395–445,可放宽) |

**保留**:GPU specialist pool 上限(`policy.py` 2040–2060)、`max_turns ≤ 16`(2006–2015)——资源不变量。

---

## 6. 类别 E:Prompt 单一来源 + 描述式而非命令式

> 问题:同一条策略在 **prose**(`orchestration.md` + `prompt_builder` DECISION FRAMEWORK)
> 和 **code** 里各写一遍(如"不要 config-only"既是 prose MUST,又是 `consecutive_config_only_rounds` 计数)。
> 既难维护又互相打架。

| # | 机制 | 位置 | 改动 |
|---|---|---|---|
| E1 | "Do not settle for config-only" 硬 MUST | `orchestration.md` 181–188 + `to_intervention_mix_summary` ESCALATION `shared_state.py` 3196–3252 | 二选一来源。建议:prose 去掉"MUST",code 侧保留**一个**计数事实作为 advisory(并按 §3.3 去掉重复注入)。 |
| E2 | DECISION FRAMEWORK 的命令式步骤(F1–F4、IDEA GENERATION 等) | `prompt_builder.py` 402–564 | 从"你必须按此顺序"改为"这些是可参考的启发法 + 客观事实";把"何时 stop / 何时 measure / 选哪个 gap"明确表述为**模型的决定** |
| E3 | `orchestration.md` 中大量"X 会被 deny"的描述 | `orchestration.md` 全文 | 随 §2–§5 的 deny 降级**同步更新**,删除已不存在的拒绝规则描述,避免 prompt 与运行时漂移 |
| E4 | phase 语义"你不能决定切 phase" | `orchestration.md` 16–22 / `prompt_builder` 122–131 | 若采纳 §7 P3(让 orchestration 可发前进 hint),同步改为"你可建议前进" |

---

## 7. 分阶段实施路线(剂量放权)

> 每阶段独立可交付、可回退;后阶段风险更高。

### P1 — 透明化 + 低风险放权(先做)
- A1–A5:WYSIWYG(allowed_actions 收窄到真集合;取消静默改写 intent)。
- B(3.1)`_required_next_step` 改纯事实;(3.3)去重 intervention_mix 注入。
- D1–D6:放开探索广度 cap(资源上限保留)。
- C8/C9:删 `explore_attempts_minimum_before_kernel_opt`、放宽 Web 工具 phase。
- E1–E3:prompt 单一来源 + 同步措辞。
- **验证**:跑最小测试集;跑一次 submit→optimize 烟测,确认 `session_breakdown.json` 形状不变(README/pr.md 的 golden 契约)。

### P2 — 把"软"顺序/分析门降级为 advisory(中风险)
- C5/C6 细分中的策略性顺序 → advisory(保留 baseline 门、trace_analyze→run_optimization 数据依赖)。
- C7:roofline-pending 由 deny 降级为 advisory(资源靠 GPU 租约)。
- B(3.2)**温和版**:steward 降级为 advisory,删 depth-gate 改写,不再 driving phase。
- **验证**:重点验 plateau 不再自动推进后,硬预算墙(IR-6 / phase 预算耗尽)仍能在预算内收尾;`stop_reason` 先于 breakdown 写(pr.md #427)。

### P3 — phase 软化 / 交错(高风险,可选)
- C1–C4:plateau 全面降级为 advisory。
- 让 Orchestration 可显式请求"前进 phase"(目前只能经 robustness 间接);或拓宽 `PHASE_LLM_PROPOSABLE_ACTIONS` 允许 EXPLORE↔KERNEL **交错**(EXPLORE 允许 kernel request、KERNEL 允许 explore),真正打破线性。
- B(3.2)**激进版**:停用 steward 内部 enqueue。
- **验证**:长时预算下的收敛性 A/B(交错 vs 线性),确认不退化、产物契约不破。

---

## 8. 风险与兜底

| 担忧 | 兜底(均为保留的不变量) |
|---|---|
| 放权后 EXPLORE/KERNEL 无限探,烧光预算 | IR-6 EXPLORE 硬 force-exit(`phase_state.py` 581–659)、各 phase 预算耗尽硬墙(1169–1175 等)、wall-clock deadline → closing → 自动 report,**全部保留** |
| 模型乱发已失败动作刷屏 | baseline 自环去重 + 幂等去重(canonical_fingerprint)**保留**;policy_denial streak 反 loop **保留** |
| 并发把引擎/资源打挂 | sweep 单例、GPU 租约、specialist pool 上限、warm-replay 单飞 **保留** |
| 未验证增益污染下游 | breakdown 对未验证 KEEP 显式 `validated=false`(替代 stack_rebench deny) |
| patch 安全 | `integrate_patch` Critic 门、路径沙箱、dynamic 红线 **保留** |
| 可恢复性 / 产物契约 | phase 单调拓扑、终止态路由、CLOSE 顺序器 + 安全网 breakdown **保留** |

---

## 9. 改动文件清单(速查)

| 文件 | 涉及条目 |
|---|---|
| `inference_optimizer/orchestrator/phase_state.py` | A1, C1–C4, D7(PHASE_ALLOWED_ACTIONS 拆分、plateau 降级、预算 pct) |
| `inference_optimizer/orchestrator/policy.py` | A4, C8, C9, D2–D6(R1 对齐、删 minimum-before-kernel、放宽 caps) |
| `inference_optimizer/orchestrator/coordinator.py` | A5, B(required_next_step / steward / 去重), C5–C7(序列与 roofline deny 降级) |
| `inference_optimizer/orchestrator/shared_state.py` | A3, D1(phase 块渲染、research_lane 默认、breakdown 标注) |
| `inference_optimizer/orchestrator/system_prompts/prompt_builder.py` | A2, E2(phase 语义渲染、DECISION FRAMEWORK 描述化) |
| `inference_optimizer/orchestrator/system_prompts/orchestration.md` | E1, E3, E4(去 MUST、同步措辞、phase 表述) |
| `inference_optimizer/protocol/action_surfaces.py` | A1(INTERNAL_ONLY_ACTION_NAMES 复用) |

---

## 10. 一句话总结

把所有 **DENY** 过一遍:**保护系统的留下,替模型做决定的降级为 advisory 或删除**;把所有
**breadth cap** 过一遍:**资源派生的留下,其余放开**;把所有 **prompt MUST** 过一遍:
**变成事实 + 目标,让模型自己推理下一步**。其余确定性骨架(可恢复性、产物契约、硬预算墙)
原样保留——它们正是"放权但不失控"的安全网。
