# Hyperloom Prompt 架构审计

评判三项判据：**固定的北极星** / **渐进式披露** / **特殊情况与主 prompt 分离**。

方法：不阅读 prompt 源文件了事，而是**执行仓库自己的 builder**（`build_orchestration_prompt`、
`build_specialist_prompts`、`_load_critic_prompt`、`default_role_registry()[...].load_system_prompt()`），
渲染出 60 份 artifact 到 `.prompt-audit/rendered/`，再对渲染结果做行数/字符/token 计量。
所有数字来自 `.prompt-audit/rendered/_manifest.json` 与 `SECTION_ORDER.md`，可用
`python /wekafs/zgong/Hyperloom-2/.prompt-audit/render_prompts.py` 复现。

> **数据完整性提示**：`.prompt-audit/recovered/{confirmed,refuted}.json` 中 `claim` 字段与
> `verdict` 字段是**错位 join** 的（第 i 条 claim 配到了另一条的 verdict）。本报告按内容重建了
> 正确的 1:1 配对：9 条 `refuted=false`（存活）+ 15 条 `refuted=true`（被证伪），双射完整。
> 下文第三、四节按重建后的配对陈述。

---

## 一、Prompt 是如何合成的

五个角色的合成机制**完全不同**，这本身就是后面三项判据得分分裂的根因：

| 角色 | 合成方式 | 模型实际收到 |
|---|---|---|
| orchestration | Python builder 拼 9 个 section，其中 1 个来自 .md | **835 行 / 47,838 字符 / ~11,960 tok** |
| specialist | 2129 行 builder → system/user 双半 | **342–449 行 / ~4.2k–6.4k tok**（16 个变体） |
| critic | 无 builder：system=.md 原文 + user=硬编码二元组拼接 | **608 行 / 28,325 静态字符 / ~7,087 tok** |
| robustness | role registry 读 .md → **被 backend 丢弃** | **0**（唯一入模型的是 4 行 `_SYSTEM_PROMPT`） |
| kernel_agent | CLI 常量 override 掉 .md | **26 行 / 1,168 字符 / 292 tok** |
| framework | 无 LLM role（是 CLI） | **0**（仅 enablement ladder 经 specialist 入模） |

### 1.1 Orchestration —— 唯一有真正 builder 的角色

入口 `src/hyperloom/orchestrator/prompts/prompt_builder.py::build_orchestration_prompt`，
关键字参数：`action_registry / enabled_actions / framework / kernel_enabled / explore_enabled /
framework_agent_phase_enabled / objective_kind / objective_value / max_minutes / macro_cycle /
cycle_directive / rules_fragment_path / framework_source_roots`。
先由 `_resolve_prompt_prelude` 把 action 名解析成 `ActionMetadata`、推导 `kernel_enabled`、
读入 rules .md，再构造 `list[list[str]]` 交给 `join_sections()`（行内 `\n`，section 间空行，
末尾 `.rstrip()` + 换行）。

配置：`kernel_enabled=True, explore_enabled=True, framework_agent_phase_enabled=True,
framework='sglang', gain_pct=15.0, max_minutes=480, macro_cycle=2, cycle_directive=<非空>`。

| # | Section | 来源 | 行数 | 门控 |
|---|---------|------|------|------|
| 1 | `## 1. MISSION` | computed（`_section_mission` 静态字面量，零参数） | 15 | 恒定 |
| 2 | `## 2. SESSION CONTEXT` | computed（插值 framework / kernel_enabled / explore_enabled / objective / max_minutes / source_roots） | 16 | 恒定 |
| 3 | `## 3. PIPELINE & TIME BUDGET` | computed（对 enabled actions 的 `ActionMetadata.typical_runtime_min` 按 phase 求和） | 22 | 恒定 |
| 4 | `## 3a. PHASE CONTRACT` | computed（`render_phase_proposable_bullets` × `PHASE_NAMES` × `llm_proposable_actions_for`） | 34 | 恒定（flag 关时该行标 `(DISABLED: --no-xxx — phase skipped)`） |
| 5 | `## 4. ACTIONS YOU MAY USE` | computed（ActionRegistry metadata：description/runtime/gain/risks/family + EMIT hint + grid hint） | 68 | 恒定（内容完全由 `enabled_actions` 决定） |
| 6 | `## 5. DECISION FRAMEWORK` | computed（静态字面量；**接收 `kernel_enabled` 但从不 branch**） | 157 | 恒定 |
| 7 | `## CYCLE DIRECTIVE` | computed（body 条件：LLM 撰写的 `cycle_directive` 非空则用之，否则用默认 breadth→depth 弧） | 7 | header 恒定，**body 条件** |
| 8 | `## 6. KERNEL-OPT REQUEST REFERENCE` | 模块级常量 `_KERNEL_OPT_PIPELINE_BODY.splitlines()` | 84 | **唯一结构性门控**：`kernel_enabled and any(a.name=='kernel_opt')` |
| 9 | `## 7. RULES & OUTPUT PROTOCOL` | **static-from-.md**：`orchestration.md` 逐字嵌入，作为**单个 list 元素** | 424 | 恒定（fragment 不可读时降级为一行占位符） |

827 section 行 + 8 个空分隔行 = **835 行**，与渲染文件精确吻合。

**两个结构性事实**：

1. **`build_orchestration_prompt` 没有 phase 参数** —— 合成出的 system prompt 是 **phase-invariant** 的。
   `SECTION_ORDER.md:5-6` 自己写明："There is **no phase argument** — the composed system prompt is
   phase-invariant; the live phase arrives per tick from the Coordinator."
   `diff` 证实 `orchestration.{PRELUDE,EXPLORE,KERNEL_AGENT,CLOSE}.md` 在 8 行前缀以下**逐字节相同**。
2. **9 个 section 只有 1 个（Sec6）被门控**。Sec7（424 行，占 51%）+ Sec5（157 行）
   = **581 行 / 70%** 的 always-on 预算完全无门控。Sec5 的 `_section_decision_framework(*, kernel_enabled: bool)`
   收下了参数却在函数体里从未引用 —— 一个**死掉的门控钩子**，`SECTION_ORDER.md:20` 诚实地记录了它。

关闭 `--no-kernel --no-explore --no-framework-agent` 后：
`orchestration.no-kernel-no-explore.md` = **717 行 / ~10,036 tok**（省 118 行）。
经生产 CLI 包装（`cli.__init__._build_orchestration_prompt` + 真实 `build_objective`）产出的
`orchestration.via-cli.md` 与直接调 builder **逐字节相同** —— CLI 层不额外添加任何内容。

### 1.2 每 tick 注入（不在 system prompt 里的那一半）

`loop/conversation.py::_compose_prompt` 每 tick 在 user 侧前置，顺序固定：

1. `SESSION_DIR=<path>`（所有角色）
2. `=== Phase ===` + `to_phase_status_summary()`（所有角色）—— phase/cycle/entered/budget/allowed，EXPLORE 多一行 `force_exit:`
3. 恢复的 `orchestration_memory` seed 块（仅 orchestration，仅 SEED 轮）
4. `=== Mission progress ===` + `to_mission_summary()`（orchestration）
5. cycle-strategy seed 块（仅 SEED）
6. `=== Time budget ===`（orchestration + robustness）
7. `=== Shared session state ===` + `=== Resource pools ===`（**仅 SEED**）
8. advisory/ledger 块：plateau、bottleneck redirect、acceptance threshold、target gap、priors match、current gaps、recent policy denials（**仅 SEED**）
9. inbox tail

SEED/DELTA 门在 `conversation.py:387`：
`push_full = not self._orchestration_conversational() or not self._orchestration_seeded`，
首轮完成后于 `coordinator.py:1799` 翻转。真实 73 轮生产 session 实测：
**31 个 SEED 轮中位 56,725 字符 vs 42 个 DELTA 轮中位 2,325 字符 —— 24× 压缩**（~14.2k → ~0.6k tok）。
会话连续性靠 Claude SDK session token（`claude.py:614-617` 抓取 → `:380-381` 存 → `:499-500` 以 `kwargs["resume"]` 回放）。

**但 system prompt 不在这个门控内**：`coordinator.py:1721` 无条件 `_load_system_prompt(agent_name)`，
`:1739` 无条件 `system_prompt=sys_prompt`，`claude.py:497-498` `if system_prompt: kwargs["system_prompt"] = system_prompt`
—— 包括 resume 轮。这是第三节 C2 的机械基础。

### 1.3 Specialist —— 2129 行 builder → ~400 行渲染

唯一公开入口 `specialist_prompt_builder.build_specialist_prompts`，纯函数：吃一个 frozen dataclass
`SpecialistPromptInputs`，吐 `(system_prompt, user_prompt)`。拆两半是为了让 backend 缓存 system 半。

- **system** = `_section_identity`（+ `_DOMAIN_FOCUS_TEMPLATES` 里按 domain 选中的 1 个 focus 块
  + 每个 `extra_focus_tags` 一个额外块 + `scope=='domains'` 时 `_cross_domain_block` /
  `scope=='freeform'` 时 `_freeform_block` + 有卡时 `_gpu_autonomy_block` + `auto_retry_reason` 时
  `_auto_retry_note_block`）→ `_section_output_protocol` → `_section_iron_rules`（按有无分卡 branch）。
- **user** = hardware → pd_disaggregation（非 PD 拆分则丢弃）→ execution_budget（`wall_budget_sec<=0` 丢弃）
  → gap → kb_subgraph（前序来源全空时走 COLD-START 分支）→ roofline → recipe → lessons → pitfalls
  → kg_recommended → kg_guided → pr_feed → source_hint → 可选 §10 NOTES。空 section 由 `_flatten` 丢弃。
  `enablement_specialist` 走一条**刻意更薄**的 user 路径（hardware / pd / budget / gap /
  `_section_enablement_playbook` / pr_feed / source_hint），13 个 `## ` section 砍到 7 个。

**门控量化**：约 **974 / 2129 行（46%）**的 builder 源码对任一次 dispatch 是被 gate 掉的
—— 10 个互斥 `_focus_*` 函数共 571 行（占文件 26.8%，**恰好 1 个会 fire**）、90 行 atom-only 分支
（sglang/vllm 上死代码）、7 个条件 section builder 共 279 行、cold-start directive 34 行。
单次 dispatch 执行 356–529 行，对 644 行并集；最大变体触达 82.1%，enablement 仅 55.3%。

**输出侧压缩只有 ~1.5–2×，不是 10×**：16 个渲染变体 342–449 行，对 669 行并集
—— 最大变体是全部可能文本的 67.1%，最小 48.1%。原因是约 130 行 always-on 内容
（Section 8 output protocol 渲染 82 行、Section 9 iron rules 46–48 行、13-repo allowlist）
统治每一份 prompt 且从不 gate。

**四个 dial 只有一个进 prompt**：`scope ∈ {domain, domains, freeform}` 在 `:854` 与 `:856` 被读；
`mode` / `bench` / `lane` 在 `:770-772` 声明，AST 扫描全 2129 行**零读取点**。它们经
`profile.py` / `dispatcher.py` 影响 Ray lease、pool 选择与 wall budget，**不影响模型读到的字**。
`bench` 有一条**间接**入 prompt 的路径：`bench=True → reserves_benchmark_lane → needs_gpu →
GPU lease → allocated_gpu_ids →` 门控 16 行 `_gpu_autonomy_block` 并翻转 Iron Rule 1 分支。

**system/user 缓存切分是反的**：system 半是**可变**的（217–324 行，随 domain/scope/GPU 变），
user 半在 16 个变体里有 12 个恒为 134 行 —— 与 `:6-9` 声明的缓存理由相悖。

### 1.4 Critic —— 无 builder，两段硬编码拼接

- **system** = `cli._load_critic_prompt()` == `(asset_system_prompts_dir()/'critic.md').read_text()`，
  **逐字原文，无包装** → 150 行 / 7,091 字符 / 1,773 tok。
- **user** = `roles/critic_agent.py::_reason` 拼三段：
  `_load_skill_preamble()`（**硬编码二元组**：`for rel in ("SKILL.md", "actions/review_coordinator_inbox.md"):`
  —— `critic_agent.py:1232`，各以 `==== <rel> ====` 前缀；SKILL.md 265 行 / 13,690 字符 +
  review_coordinator_inbox.md 139 行 / 5,471 字符）
  + `==== JUDGE BUNDLE ====` JSON
  + `_REVIEW_OUTPUT_INSTRUCTIONS` 常量（46 行 / 1,978 字符）
  → 458 行 / 21,990 字符 / 5,498 tok。

合计 **608 行 / 28,325 静态字符 / ~7,087 tok**，对渲染样例中 756 字符的 judge bundle
= **37.5 : 1** 静动比（真实 turn0 capture 的 bundle 是 2,811 字符 → ~10:1）。
消息数组经仓库自己的 `roles.base.build_chat_messages(system, user)` 产出。
**无 tool palette**：`critic_agent.py:542` `del tools, max_turns  # Critic is single-turn / no tool palette.`

### 1.5 Robustness —— 加载了、传下去了、然后被删掉

`default_role_registry()['robustness'].load_system_prompt()` 逐字读 `robustness.md`
= **148 行 / 10,059 字符 / 2,515 tok**，经 `coordinator.py:1721/1739` 传入 `backend.run(system_prompt=...)`。
两个 backend 都在函数体第一句丢弃：
`robustness_agent.py:162` `del system_prompt, tools, max_turns`；
`robustness_mock.py:53` `system_prompt (str | None): Unused; accepted for protocol parity.`
`cli/backends.py:233-243` 只能造出 `MockRobustnessBackend` 或 `RobustnessAgentBackend`
—— 全仓库**没有任何代码路径**为 robustness 构造 ClaudeBackend。
真正入模型的 robustness 文本只有 `decision/rca_engine.py:180` 的 4 行私有 `_SYSTEM_PROMPT`。
实测上下文：robustness 中位 **245 tok/轮** vs orchestration 中位 **190,051 tok/轮**，同 session 内 ~776× 落差。

### 1.6 Kernel_agent —— 常量遮蔽文件

`kernel_agent.md`（65 行 / 3,307 字符 / 827 tok，含 IR-1..IR-7）在正常运行中**从不加载**：
`cli/__init__.py:2095` `prompts["kernel_agent"] = args.kernel_prompt or _DEFAULT_KERNEL_PROMPT`
（由 `:2094` 的 `if not no_kernel:` 守卫）写入 `coordinator.system_prompt_overrides`，
而 `conversation.py:720-721` 先查 override 再查 role 文件：
`override = getattr(self, "system_prompt_overrides", {}).get(agent_name); if override is not None: return override`。
模型实收 **26 行 / 1,168 字符 / 292 tok**。

之所以能这么薄是**结构性**的而非编辑功劳：`intent_router.py:573-575`
`# Programmatic shortcut: run a registered kernel handler inline + emit RESPONSE without burning an LLM turn.`
—— `KERNEL_REQUEST_HANDLERS`（`request_handlers.py:6255-6262`）的 6 个 kind
（trace_analyze / run_gemm_tuning / run_fusion / run_optimization / integrate / apply_patch）
全部走 6,318 行确定性 Python，LLM 只负责发心跳。

### 1.7 Framework —— 4-role runtime 里没有它

`ls src/hyperloom/orchestrator/prompts/*.md` 恰好 4 个文件（critic / kernel_agent / orchestration / robustness）。
framework agent 是 CLI（`fa phase-discover` / `phase-audit`），由 `orchestrator/framework/client.py` 调起。
它的 4 个 stage prompts（359 行）+ 4 个 references（478 行）= **837 行没有任何 loader**。
唯一入模型的 framework 散文是 `build_enablement_ladder_book`（`enablement_ops.py:397`，40 行 / 6,492 字符），
经 `specialist_prompt_builder.py:1954-1983` 注入 enablement_specialist 的 **user** 半作 `## 1b. ENABLEMENT PLAYBOOK`。

**三个角色的死散文合计**：570（kernel SKILL）+ 252（robustness SKILL）+ 133（framework SKILL）
+ 101（framework AGENTS）+ 359（framework prompts/）+ 478（framework references/）
+ 65（kernel_agent.md，被遮蔽）+ 148（robustness.md，被丢弃）= **2,106 行**。
这三个角色的 LLM 实收：26 + 4 + 40 = **70 行**。

---

## 二、三项判据逐条评价

### 2.0 矩阵：每角色 × 每判据

| Agent | 北极星 | 渐进式披露 | 特殊情况分离 |
|---|---|---|---|
| **orchestration** | **达成** (8.5) | 部分达成 (6.5) | 部分达成 (4) |
| **specialist** | **未达成** (3.5) | 部分达成 (6) | 部分达成 (5) |
| **critic** | **未达成** (2.5) | **未达成** (2) | **未达成** (2) |
| **robustness** | **未达成** (1.5) | **未达成** (1) | **未达成** (2) |
| **kernel** | 部分达成 (5) | 部分达成 (6) | **达成** (8) |
| **framework** | **未达成** (3) | **未达成** (3) | 部分达成 (4) |
| **总体** | **部分达成 5.0** | **部分达成 5.5** | **部分达成 4.0** |

矩阵的形状本身就是结论：**三项判据在 orchestration 上都最好，在 critic/robustness 上都最差**，
而 kernel 在"分离"上拿最高分是因为它把例外**搬进了可执行代码**，不是搬进文档。
不要把六个角色平均成一个数 —— orchestration 的北极星 8.5 与 robustness 的 1.5 之间
没有共同的失败模式。

---

### 2.1 判据一：有固定的北极星 —— **部分达成，5/10**

**orchestration 达成（8.5），其余五个角色 3.5 或更低。** 系统级北极星存在且优秀，但它**只存在于一个角色的 prompt 里**。

**做到的部分（orchestration）**：北极星是一句可背诵的话，在 `orchestration.prompt.md:4-5`：

> "Your single most important goal is to maximise the run's **cumulative_gain**
> (percent over baseline_tput) within the wall-clock budget."

835 行里只有 **2 行**匹配 `/your (job|goal) is|single most important/` —— 不存在竞争性使命陈述。
它是**可度量**的：`:23` 插值出 `- objective : gain_pct=15.0`，`:166` 给出停止规则
（`if stop_reason is set OR cumulative_gain >= target_gain_pct`），
`render.py:132-133` 每 tick 推送 `gain : per-round-sum=X% validated=Y%`。

它是**不动**的，而且是结构性不动：`_section_mission()` 是零参数静态字面量
（`prompt_builder.py:54-77`），`build_orchestration_prompt` 无 phase 参数，
所以连每 macro-cycle 的 prompt 重建（`phases/explore.py:234-268`）都碰不到它。
`diff` 证实 PRELUDE 与 CLOSE 渲染在 `## 1. MISSION` 以下完全相同。
唯一可变的 CYCLE DIRECTIVE 明确标注 `(advisory — this macro-cycle's focus)`（`:319`），
限长 1500 字节，并被 `_DIRECTIVE_POLICY_BLACKLIST`（`'ignore phase'`/`'bypass policy'`/`'override policy'`，
`orchestration_memory.py:116-124`）筛查 —— LLM 撰写的 directive **无法重定义使命**。

它对自己度量的缺陷是**诚实**的：`:13-15` 写明 "sums of per-round gains still do NOT compose
linearly, so drive the loop until ``explore`` has produced at least one KEEP that survived
the stack rebench"，并同时渲染 raw 与 validated 两个数。一个带 raw/validated 分裂文档的单指标，
好过一个悄悄撒谎的单数字。

**决定性的失分证据（C8，见第三节）**：MISSION 这 13 行**不含任何正确性约束**。
`grep -n accuracy orchestration.prompt.md` 命中 4 行（95/124/128/130），
全部位于 action catalogue 的 payload schema 里，**MISSION 与 DECISION FRAMEWORK 内一处都没有**。
而真实 KEEP 规则是合取式：throughput 门 **AND** accuracy 门。
所以 prompt 声明的目标严格**弱于**系统实际执行的目标。
（注意：这一条被削弱但未被推翻 —— 反方证明 catalogue 里 `acc_risk=` 出现 **14 次**，
6 个 action 带非零 acc_risk，所以"agent 完全没有 accuracy 先验"是错的；
存活的是"MISSION 层不可见"这个更窄的版本。）

**第二条决定性证据（C5）**：`target_gap_pct` —— 运行时预计算的"距目标还差多少" —— 在真实运行中
**恒为 0.0**。`conversation.py:461` 把 `kind` 当属性读（`getattr(obj, "kind", "")`），
但 `Objective.kind` 是 `@abstractmethod`（`objective.py:39-40`），四个具体 objective 全部实现为**方法**；
`:463` 又读 `value`，而 `TargetGainObjective` 存的是 `target_gain_pct`。**双重 bug**：
即便修好 `kind()` 调用，`getattr(obj,'value',0.0)` 仍是 0.0。
后果：`explore.py:1112` 以 `target_gap > 0.0` 为门，`throughput_below_target` 这一行
**永远不进 `gaps[]`** —— 而 prompt 恰恰在 `:179` 告诉 agent 优先读 `=== Current gaps ===`；
`render.py:580` 每次 dump 都打印 `target_gap_pct=0.00`，读起来像"目标已达成"。

**其余五个角色为何未达成**：

- **specialist（3.5）**：不是一个目标，而是**四个互不相容的目标**，在 dispatch 时按 `domain` 选定并冻结。
  每个变体的 system 第 12 行都承诺 "probe the host via Bash, **author source patches into your
  isolated worktree**"，然后在同一份文件 66 行之后对两个 domain 撤销
  —— research_scout `:78` 与 static_recon `:67` 都写 "**Iron rule** — read-only. Never write a patch"。
  第三种把门换掉（enablement `:33-35` "The gate is RUNNABILITY ... not throughput"），
  第四种把使命交给运行时字符串（freeform `:31`）。
  **无一可度量**：16 份 specialist 渲染里 `grep -c cumulative_gain` = **0**。
- **critic（2.5）**：使命是**六项工作的编号列表**（`critic.user-wrapper.prompt.md:43-52`），
  其中五项 live path 从不触发（`critic_agent.py:557` 硬编码 `"kind": "coordinator_inbox"`）。
  更糟的是**两套独立来源的 verdict 分类法并存且互斥**（详见第四节 R12 —— 该条 severity 被证伪，
  但两套分类法同时到达模型这一事实经真实 turn0 capture 确认）。
- **robustness（1.5）**：目标句本身很干净（`:12-15` "detect failure modes *before* they cost a
  full session budget..."），但**没有任何模型读到它**。
- **kernel（5）**：override 的北极星单一明确（"responder-only ... emit ONE `response` intent in reply"），
  但**不可度量** —— `grep -niE 'gain|speedup|1\.05|1\.5'` 在两份 kernel prompt 上都是 0 命中。
  给 5 分而非 2 分，是因为 6 个 kind 全部绕过 LLM，目标可度量性几乎不影响行为。
- **framework（3）**：无 LLM role。唯一入模的 enablement ladder 自标
  "the gate here is RUNNABILITY ... not perf" —— 局部清晰，但**没有一句话解释 runnability 如何服务于
  cumulative_gain**（它是 baseline 存在的前提），所以是一个全局失锚的局部目标。

---

### 2.2 判据二：做到了渐进式披露 —— **部分达成，5.5/10**

**运行时那一层（L3 会话 + L2 工具 + 压缩）是真的、且做得好；静态那一层（L1 分段门控）几乎不存在。**

**做到的部分**：

- **L3（会话生命周期）真实存在**：持久 SDK session（`claude.py:296`
  `resume_session = self._session_id if self.conversational else None`）+ SEED/DELTA 门
  （`conversation.py:387`）+ 模型自撰 checkpoint。压缩触发是**五选一**
  （`orchestration_memory.py:76-109`）：context tokens ≥ 窗口 70%（200k 的 140k）、
  phase 边界、20 ticks、30 分钟、400k 字符。
  checkpoint prompt 本身教模型**不要压缩什么**：`orchestration_memory.py:146-149`
  "capture intent and rationale, not raw numbers you can re-pull from the context tools."
  劣化回复会**跳过压缩**而不是摧毁会话（`is_degenerate_checkpoint`，`:215-233`），
  连续 3 次才升级严重度；`build_memory_record` 累积去重 learnings（上限 50）并做 non-empty-wins
  前推，所以一次健忘的 checkpoint 不会清空在飞计划。真实 session：71 ticks / 29 次 checkpoint /
  50 条 learnings / 9,174 字符 → 渲染成 11,061 字符的工作记忆块。
- **L2（按需拉取）真实存在**：`CONTEXT_TOOL_SPECS` 12 个只读工具
  （`mcp_context_tools.py:235-337`），handler 返回真实字符串而非仅 ack，
  经 `gate.py:766-772` **仅授予 orchestration**（同时授予 `Read`/`WebSearch`/`WebFetch`）。
  多数 projection 自带上限（`to_warm_start_summary(max_lines=12)`、`to_gaps_summary(max_entries=10)`、
  `to_proposal_scores_summary(max_rounds=2)`、`to_policy_denial_summary(top_k=6)`、outcomes `LIMIT` 钳在 1..50）。
- **DELTA banner 是放对位置的指针**：`conversation.py:585-596` **恰好在**状态被扣留的那些轮
  内联列出十个拉取工具，静态 rules 再重复一次列表 —— agent 不会在薄轮里既不知为何薄、也不知去哪拉。
- **specialist 的 analysis.md 延迟是全仓最佳机制**：user prompt `:72` 同时给出路径、预期体积、
  **以及稳定的小节标题枚举** —— "All section headings are stable: ``## Executive Summary`` /
  ``## Top Operations`` / ..." —— 让 agent 可以 seek 而不是盲读 20 KB。

**决定性的失分证据**：

1. **L1 几乎不存在（C2）**：9 个 section 只有 1 个被门控。835 行 / 47,838 字符 / ~11,960 tok
   在**每一轮**重发（`coordinator.py:1721/1739` 无 seeded 门），其中 Sec7（424 行）+ Sec5（157 行）
   = **70%** 完全无门控。因为 prompt 是 phase-invariant 的，agent 在 PRELUDE 和 CLOSE 里
   照样背着 EXPLORE 专属内容（`### Watching a running specialist` 46 行、
   `### One specialist, four dials` 71 行）和 kernel payload 模板（84 行，
   在四个 PolicyGate 会直接拒绝这些请求的 phase 里）。
   Sec5 那个**收下却从不使用的 `kernel_enabled` 参数**是这个缺口的书面证据。
   （量级上要诚实：反方用真实遥测证明这不是"每轮 12k 的税" —— 见第四节 R5。存活的是
   L1 缺失这个**结构**事实，不是它的成本估算。）
2. **externalized 文档层不可达（C3）**：`inference_optimizer/references/`（10 文件 845 行）
   + `actions/*.md`（518 行）= **1,363 行零入站指针**。
   `grep -c 'references/' SKILL.md` = 0；渲染 prompt 里 `references/`、`actions/`、`SKILL.md`
   各出现 **0** 次。而 `registry.py:7` 与 `actions/__init__.py:10` **两处** docstring
   声称 markdown body "loaded lazily" —— `load()` 只 glob `meta_dir.glob("*.yaml")`，
   `ActionMetadata` 上根本没有 body 字段，`SubAgentRunner` 里连一个 `.md` token 都没有。
3. **critic 完全没有 L2（C4）**：`critic_agent.py:542` `del tools, max_turns`。
   loader 是硬编码二元组（`:1232`），10 个 skill 文件里 **8 个（893 行）永不加载**
   —— 47,119 字符 skill 语料的 **59% 不可达**。静动比 **37.5:1**。
   而 always-loaded 的 review 流程在 step 3 指示 agent 去读两个**它打不开的文件**。
4. **robustness 得 1 分**：148 行加载、快照、传输、然后被两个 backend 都删掉。这不是披露，是空转。
5. **framework 得 3 分**：837 行结构良好的 stage 文档零 loader；4 个 stage prompt 里
   只有 1 个（`filter_candidates.md:22`）指向自己的 reference，另外 **357 行 reference 在自己的文档集内部就是孤儿**。
   给 3 分而非 1 分，全靠 enablement ladder 那一个机制。

---

### 2.3 判据三：特殊情况与主 prompt 分离放入文档 —— **部分达成，4/10**

**这是三项里最低的一项，失败模式很具体：externalization 的"存"这一半建好了，"取"这一半没接线。
于是例外又被内联回主 prompt。**

三套文档语料同时存在且运行时全部不可达：
`inference_optimizer/references/` 845 行（零入站指针）、
`actions/*.md` 518 行（唯一指针是一条面向机器的 PolicyGate 拒绝串 `gate.py:1426`）、
critic 的 `references/` 511 行 + 4 个未路由 actions 382 行（被硬编码二元组排除，且无工具可拉）。

**没有可达的外部家，例外就回流到了主 prompt**：

- **orchestration**：**210 / 836 行 = 25.1%** 是例外/恢复块（非空行口径 194/705 = 27.5%）。
  逐块行区间：skip_to_close 豁免（84-90）、FAILURE RECOVERY（211-291）、
  KERNEL TARGETING 否定项（406-410）、specialist 救援动作（505-522）、
  SESSION_DIR 沙箱（642-661）、Hard rules（663-713）、
  "you cannot propose roofline/profile"（715-742）。
  标记密度：25 行含 never/NEVER、11 行含 do-NOT、5 行 MUST、2 行 EXCEPT。
  最大一块是 **FAILURE RECOVERY，81 行**，由 `prompt_builder.py:573` **无门控**地静态发出，
  而它自己第一句就把自己限定在一个多数 tick 都不存在的状态上：
  "When the inbox carries a fresh `delegated_result{state!='succeeded'}` ... do NOT re-propose
  the same action with the same params."
- **critic —— 全系统最差比例**：150 行 system prompt 里，cross-domain 块占 **90-150 行 = 61 行 = 41%**，
  而它的**第一句就宣告自己的条件性**：
  > "This block fires only when `judge_bundle.review_constraints.cross_domain == true` ...
  > For single-domain (`scope == "domain"`) or freeform ... skip this block entirely."（`critic.prompt.md:92-96`）

  运行时**已经算出了这个 boolean**：`has_cross_domain = any(_proposal_scope_literal(p) ==
  SCOPE_DOMAINS_LITERAL for p in proposals)` → `if not has_cross_domain: return` → `rc["cross_domain"] = True`
  （`critic_agent.py:251-258`）—— 它**只 gate 了 data，散文照发**。
  加上 deviate 块（67-75）与 Hard rules（76-89）：**84 / 151 行 = 55.6% 是例外**。
  真实 capture（`critic_turn0_prompt.txt`，PRELUDE，审 baseline+target_analysis）证实：
  bundle 的 `review_constraints` **完全没有 `cross_domain` 键**，模型确实收到了 61 行
  "第一句告诉你可以跳过这 61 行"的文本。
- **specialist（5，最好与最坏同时在场）**：坏的一面是 **9 个硬编码 `**Pitfalls**` 块**
  作为 Python 字符串字面量散在 `:105/:149/:241/:267/:318/:337/:371/:402/:437`，
  承载的正是 RecipeKB §5c 存在的目的（例："Raising `--max-num-seqs` past 512 on MI300X -> OOM
  on 671B MoE models"）；`NEVER_TOUCH` flag 集冻结在 prompt 文本里（`:125-127`）；
  一段 attention-backend 枚举紧跟在"去 `serve --help` 拿"之后内联（`:255-262`）。
  好的一面见第五节。
- **kernel 达成（8）**：唯一把例外放在**可执行代码**而非文档里的角色。
  26 行里 9 行是例外规则（比例 35% 很高，但基数是 26 行，绝对量 9 行），
  且两条都是不可延迟的安全不变量（native-only 规则、SESSION_DIR 路径白名单）。
  能这么薄是结构性的：6 个 kind 全在 Python 里执行，所以 570 行的
  `agents/kernel/SKILL.md` 是**不必要**，而不仅仅是**未加载**。
  扣分点：570 行 SKILL.md 与被遮蔽的 65 行 kernel_agent.md 仍在磁盘上且互不指向
  （`grep 'SKILL.md' kernel_agent.md` = 0），维护者无法判断哪份文档有效。
- **framework（4）**：文档集结构是全仓最好（4 stage prompt + 4 reference，
  每个 stage 带 Inputs / Tool-surface / Procedure / Failure-mode 且显式 Stage N-1 → N+1 链接），
  但运行时可达性近零。

**一个做对了的反例，值得记下**：retired-action 禁令放置正确 —— prompt 里只留一行事实
（`rendered:689` "The legacy `validate_stack` / `backends` / `params` action names are not in
any phase's proposable set (use `explore`)."），而 "do not re-introduce" 的政策
被正确地流放到 `SKILL.md:277`（launcher 面向，不进 agent 上下文）。团队知道这个区别。

---

## 三、确认的问题（经对抗验证存活）

以下 9 条都经过了一位**专职试图推翻它们**的怀疑者的独立复核（重跑 grep、重算行数、
执行真实代码路径、搜索缓解机制），结论均为 `refuted=false`。可以有把握地陈述。
按严重度排序；标注 severity 与判据归属。

### C1 —— `target_gap_pct` 恒为 0.0（high / 北极星）

**问题**：运行时预计算的"距目标还差多少"信号，在**每一次真实运行**中都被钉死为 0.0。

**证据**：`loop/conversation.py:460-469`：

```python
obj = getattr(self, "_current_objective", None)
obj_kind = getattr(obj, "kind", "") if obj is not None else ""
if obj_kind == "gain_pct":
    target_val = float(getattr(obj, "value", 0.0) or 0.0)
```

但 `Objective.kind` 是 `@abstractmethod`（`objective.py:39-40`），
`TargetGain/TargetTput/TargetBaseline/TimeOnly` 四个具体类**全部实现为方法**，且**都没有 `value` 字段**。
执行验证：`build_objective({'MAX_HOURS':'4','TARGET_GAIN_PCT':'15'})` 得到 `TargetGainObjective`，
`getattr(o,'kind','')` 返回 bound method，`== "gain_pct"` 为 False，走 else 分支。
**这是双重 bug**：即便修好 `kind()` 调用，`getattr(obj,'value',0.0)` 仍为 0.0
（真实字段名是 `target_gain_pct`），`max(0.0, 0.0-gain)` 依旧是 0.0。
全仓 `grep 'target_gap_pct\s*='` 确认 `conversation.py:464/469` 是**唯一的非测试写入点**，
没有任何反序列化或 fallback 路径能救它。
唯一覆盖它的测试 `test_coordinator_async_methods_coverage_unit.py:701-711` 用
`class _Obj: kind = "gain_pct"; value = 20.0` —— 一个**生产环境不存在的形状**，
测试通过并把 bug 锁死。

**影响**：两个消费者在每次 gain-targeted 运行中静默失效。
`explore.py:1112` 以 `target_gap > 0.0` 为门，所以最直接编码"我们离使命目标还差 N%"的
`throughput_below_target` 这一行**永远不进 `gaps[]`** —— 而 prompt `:179` 恰恰告诉 agent 先读 gaps。
`render.py:580` 每次 SharedState dump 都打印 `target_gap_pct=0.00`，语义上读作"目标已达成"。

**范围界定（怀疑者的修正，须保留）**：这**不是**运行时唯一的距离信号，原始表述过强。
`cli/__init__.py:167-190` 的 `_objective_summary_for_prompt` **正确地**调用 `objective.kind()`
并读 `target_gain_pct`，所以 system prompt 确实渲染出 `- objective : gain_pct=15.0`；
每 tick 的 `=== Mission progress ===` 渲染当前 `gain: per-round-sum=X% validated=Y%`；
`coordinator.py:1624` 的 `objective.reached()` 走正确的方法路径，**到点停止仍然有效**。
两个操作数都每 tick 到达模型，agent 可以自己算差值。死掉的是**预计算的派生值**和**由它构建的 gaps 行**。

**修法**：`conversation.py:461` 改为 `obj_kind = obj.kind() if obj is not None else ""`，
`:463` 改为 `float(getattr(obj, "target_gain_pct", 0.0) or 0.0)`。
更好的做法是在 `Objective` ABC（`objective.py:36`）上加 `def target(self) -> float`，
统一走 `obj.kind()` / `obj.target()`。
必须把测试里的 `_Obj` stub 换成真实的 `TargetGainObjective(20.0)`，否则测试无法失败。

---

### C2 —— system prompt 不在 SEED/DELTA 门控内（high / 渐进式披露）

**问题**：47,838 字符的 orchestration system prompt **逐字重发于每一轮**，
包括 resume 轮。SEED/DELTA 门只压缩 user 侧内容。

**证据**：`coordinator.py:1721` `sys_prompt = await self._load_system_prompt(agent_name)`，
`:1739` `system_prompt=sys_prompt` —— 路径上**没有任何 `_orchestration_seeded` 检查**
（该 flag 在 `:1799` 才被置 True，即本轮之后）。
`claude.py:497-498` `if system_prompt: kwargs["system_prompt"] = system_prompt` 每次调用都触发，
`resume_session_id` 在 `:499-500` 单独应用。
渲染实测：835 行 / 47,838 字符 / ~11,960 tok；
Sec7 RULES（`:412-835`）= 424 行 / 22,991 字符 / ~5.7k tok，
Sec5 DECISION（`:161-318`）= 158 行 / 9,247 字符 / ~2.3k tok —— 两块合计 **67.4%**，零门控。
`SECTION_ORDER.md:20` 逐字记录了那个死钩子：
`| 6 | 5. DECISION FRAMEWORK | computed-from-code (static literal list; takes kernel_enabled but does not currently branch on it) | 157 | always |`
`_section_decision_framework(*, kernel_enabled: bool)`（`prompt_builder.py:499`）
在函数体里**从未引用**该参数。
phase-invariance 经 `diff` 确认：PRELUDE 与 EXPLORE 渲染在前缀以下只差一个空行。

**影响（须按怀疑者的量级修正陈述）**：
结构事实成立 —— L1 分段门控实质缺失，PRELUDE/CLOSE 携带 ~117 行 EXPLORE 专属材料，
kernel payload 模板（84 行）穿过四个这些请求非法的 phase。
但**成本量级远小于直觉**：orchestration backend 是 conversational 的，
真实两个多 tick session 中，单轮 orchestration 总上下文**最小值就是 132,674 tok**，
中位数 156,538 / 245,105。system prompt（实测 11,776 与 16,122 tok）
是"曾经发生过的最薄的一轮"的 8–12%，**不是 floor**。
在真实 DELTA 轮上（session 20260724T025728Z，18 个 DELTA 轮），
中位切分是 fresh input 4 tok / cache_creation 2,310 tok / cache_read 149,707 tok
—— system prompt 位于 cache_read 内，按 0.1× 计费。整份 system prompt 在中位 DELTA 轮的
**边际成本约 1,178 token-equivalent，占该轮 ~17,900 的 6.6%**；
聚合口径下占两个 session orchestration 输入侧总量的 4.85% 与 8.92%。
另外，**phase-invariance 是刻意设计而非疏忽**（`SECTION_ORDER.md:5-6` 明写），
把 system prompt 做成 phase-conditional 会在每次 phase 切换时**击穿 prompt cache 前缀**。

**修法（据此收敛）**：不要为了省 token 去做 phase 门控 —— 那会用 ~1% 的上下文节省
换来重复的 cache-creation（1.25×）。
真正该做的是**删除**：把 Sec7 里与 Sec5 重复的 5 行（`636-640`）和 4 行硬规则（`710-713`）去掉，
把 `_section_decision_framework` 那个死参数要么用起来要么删掉。
如果确实要做 phase 门控，复用已有的 `_reseed_orch_prompt_for_cycle`
（`phases/explore.py:235-270`，它已经能在运行中替换 `system_prompt_overrides['orchestration']`），
且只在 macro-cycle 边界（本就要重建 cache）触发，不要在每次 phase 转换时触发。

---

### C3 —— 1,363 行 externalized 文档零入站指针 + 两处幽灵 loader（high / 渐进式披露）

**问题**：`inference_optimizer` 的 externalized 文档语料完全不可达，
且**两处 docstring 描述了一个从未被写出来的 lazy loader**。

**证据**：`wc -l src/hyperloom/inference_optimizer/references/*.md actions/*.md`
= **1,363 行 / 14 文件**（specialist.md 195、recover.md 116、integrate_patch.md 110、
roofline.md 97；references 侧 troubleshooting.md 103、operations.md 186、paths.md 94 等 845 行）。
全仓对各章文件名的 grep 在 `.prompt-audit` 之外零命中；
`grep -c 'references/' SKILL.md` = 0；渲染 prompt 中 `references/` / `actions/` / `SKILL.md`
各 **0** 次。
幽灵 loader 有**两处**（原始审计只报了一处）：
`actions/registry.py:7` "the markdown body at ``actions/<name>.md`` is loaded lazily"，
以及 `actions/__init__.py:10` "loaded lazily by SubAgentRunner when composing a sub-agent prompt"。
而 `load()`（`:272-303`）只 glob `meta_dir.glob("*.yaml")`（`meta_dir = actions_dir / "_meta"`，`:268`），
`ActionMetadata`（`:136-162`）**没有 body 字段**，`sub_agent_runner.py` 里
`.md` / `read_text` / `playbook` 三个 token 一个都没有。
唯一活的指针是 `gate.py:1426` 的拒绝提示串 `"...; see actions/integrate_patch.md"`。

**影响**：维护者在 `actions/specialist.md` 里补一条边界情况时，
**相信自己在改变 agent 行为，实际是在写 /dev/null**。
这也是 externalization"存了不取"的机制来源。
另需注意 `pyproject.toml:166,180` 把两棵树都作为 package-data 打包 —— 有披露意图，无投递路径。

**范围界定**：孤儿章节的内容并非一律独有。
kernel.md（36 行中 27 行）、troubleshooting.md（58 中 36）、cache.md（20 中 13）
大量重复进了 SKILL.md，对这些文件缺陷是**未同步的重复**而非知识丢失；
真正不可达的独有内容是 quantization.md（26 中 2）与 operations.md（94 中 28）。

**修法**：二选一。(a) 删掉 `references/` 与 `actions/*.md`，把有承载力的内容折进 builder；
(b) 接线：在 orchestration prompt 加一节 `## 8. ON-DEMAND REFERENCE INDEX`，
每份文档一行**绝对路径 + 五个词的"何时读"触发条件**（agent 已由 `gate.py:771` 持有 `Read`），
并加一个 `read_reference(name)` context tool 使路径不会腐烂。
无论选哪条，都必须修 `registry.py:7` 与 `actions/__init__.py:10` 这两句 docstring。
再加一个 pytest：存在 `<name>.md` 却无任何 reader 时失败。

---

### C4 —— critic 无任何拉取能力，却被指示去读它打不开的文件（high / 渐进式披露）

**问题**：always-loaded 的审查流程在**流程中段**指示 critic 去查阅两份文件，
而 critic 没有任何工具可以打开它们。

**证据**：`critic_agent.py:542` `del tools, max_turns  # Critic is single-turn / no tool palette.`
与此同时 `actions/review_coordinator_inbox.md`（每次都被 `:1232` 拼接）写着：

> "3. Cross-reference with [references/risk_rules.md](../references/risk_rules.md) for
> blocker / major / minor categorisation, [references/verdict_schema.md](../references/verdict_schema.md)
> for per-verdict required fields"

两份都不加载 —— loader 是硬编码二元组 `for rel in ("SKILL.md", "actions/review_coordinator_inbox.md"):`。
`risk_rules.md`（85 行）持有 blocker/major/minor 的分级定义，
而输出 schema 要求 `'risks': [{'severity': 'blocker|major|minor', ...}]`（`critic_agent.py:74`）。
实测静态负载 28,325 字符 : 756 字符 judge bundle = **37.5:1**。
真实 capture 确认 8/10 skill 文件（893 行）从不加载，占 skill 语料 59%。
另有腐烂证据：`critic.prompt.md:59` 引用 "the phase contract block in §5"、
`:140` 引用 "(§1.2 red lines)"，而 `grep -c '^## ' critic.prompt.md` = **0** —— prompt 没有编号小节。

**影响（须按怀疑者的修正收窄）**：
"agent 用一套它没见过的评级标准打分"这个描述**部分被削弱**：
severity 枚举词本身内联在 `critic_agent.py:74`；blocker 的**判据**在已投递的负载里
重复了三次（`SKILL.md:200` "Return `approve` only when all blocker risks are cleared:"
后跟六项清单，映射到 risk_rules.md 八条 blocker 中的六条；
`decision_reviewer.py:863-878` 还把 `approve_requires_by_class` 作为**机器可读**结构注入 bundle）。
更关键的是 **`risks[].severity` 不 gate 任何东西**：
`IntentType.REVIEW_VERDICT` 的必填字段元组只有 `("target_proposal_msg_id",)`（`intent.py:63`），
生产路径中 `risks` 只被用于打印计数（`coordinator.py:397`）、
字符串化进 framework notes、KB 行负载与 semantic_audit.md 渲染 —— **零处 `severity == "blocker"` 控制流**。
所以准确的影响是：**一条 always-loaded 的不可满足指令 + 死路由表 + 死链接**，
是真实的整洁性与可信度缺陷（模型被要求做一件物理上做不到的事），
**不是**"决定 patch 能否落地的门被架空"。

**修法**：(1) 把 risk_rules.md 的 severity 定义与 Benchmark Validity Checklist 内联进
`_REVIEW_OUTPUT_INSTRUCTIONS`（`critic_agent.py:59-106`），然后删掉那条指针。
(2) 删除 `SKILL.md:93-97` 四行不可达路由，或按 `request.kind` 门控整个 preamble。

---

### C5 —— critic 41% 的 system prompt 是自称条件性的死文本（high / 特殊情况分离）

**问题**：cross-domain 块每轮无条件发送，而**运行时早已算出它的触发条件**。

**证据**：`critic.prompt.md` 共 150 行，cross-domain 块占 `90-150` = **61 行 = 40.7%**
（按字符 2,991/7,091 = 42.2%）。它的开头自述：

> "This block fires only when `judge_bundle.review_constraints.cross_domain == true` ...
> For single-domain (`scope == "domain"`) or freeform (`scope == "freeform"`) specialist
> proposals skip this block entirely."（`:92-96`）

触发条件在 Python 里已经算好：`critic_agent.py:251-253` + `:258`，由 `:601` 无条件调用。
`diff` 证实渲染件与出厂源文件**完全一致**（非陈旧 artifact）。
决定性的是真实 capture：`critic_turn0_prompt.txt`（session 20260731T083332Z，Qwen3-30B-A3B，
PRELUDE，审 baseline + target_analysis）中该块出现在 `90-150` 行，
而内嵌 judge bundle 的 `review_constraints` **完全没有 `cross_domain` 键**。
加上 deviate 块与 Hard rules，例外合计 **84/151 = 55.6%**。
静动比独立复测 28,347 : 734 = **38.6:1**。

**加重情节**：`:259` 还设置了 `rc["cross_domain_rules"] = cross_domain_rule_descriptors()`，
发出带 `rule_id/description/failure_verdict/failure_reason_code` 的**结构化字典**
（`patch_safety.py:177-229`），与散文块的 strategy hints 1-3 重复
—— 即便在触发的那一次，散文也是冗余的。

**影响**：在压倒性常见的单域审查上，41% 的 critic system prompt 是模型必须读完再丢弃的死文本，
而它从这块里学到的第一件事就是"这块可以跳过"。

**修法**：约 6 行改动，复用已存在的分支 —— 在 `_maybe_inject_cross_domain_constraints`
（`critic_agent.py:245-258`）的 `if not has_cross_domain: return` 守卫之内，
把这 61 行追加到 **user prompt**；从 `critic.md` 删除 `90-150` 行。

---

### C6 —— specialist 在同一份 prompt 里既授予又撤销 patch 权限，且 `mode` dial 完全失效（high / 北极星）

**问题**：只读 specialist 收到一条明确的能力授予，66 行后又收到它的撤销，**没有优先级标记**；
而本应解决此冲突的 `mode` dial **对渲染文本零影响**。

**证据**：`research_scout_specialist.system.prompt.md:12`
= "probe the host via Bash, **author source patches into your isolated"；
同文件 `:78` = "**Iron rule** — read-only. Never write a patch, never launch a"。
`static_recon` 同样：授予在 `:12`，铁律在 `:67`。
**不是渲染 artifact** —— 绕过 `.prompt-audit/` 直接调 `build_specialist_prompts()` 复现：
授予由 `_section_identity`（`:851-852`）**无条件**发出，只读规则由
`_focus_research_scout_specialist:530` 与 `_focus_static_recon_specialist:627` 发出，
两者都经 `runner.py:872` / `:1044` 到达模型。
**失效性**：`grep -nE 'inp\.mode|inp\.bench|inp\.lane'` 在 2129 行 builder 上 **0 命中**；
实测 `mode='research'/bench=False/lane='cpu'` vs `mode='patch'/bench=True/lane='gpu'`
→ system 相同 True、user 相同 True、research 版本中含 "author source patches" 为 True。

**缓解机制搜索的结果是反向的**：
(a) Section 9 IRON RULES 站在**授予**一侧 —— 无 GPU 分配时规则 1 结尾是
"you propose what to try and optionally author patches"，规则 2 开头是
"**You MAY** produce changes for integration"，只读 sub-agent 读到的是**三条授予对一条禁令**。
(b) 唯一真实的门是 `runner.py:1370-1373` `readonly = params.get("readonly") or domain ==
"research_scout_specialist"`，**按 domain 名硬编码，不由 mode 驱动**。
(c) `prompt_builder.py:397-398` 在 orchestration 的 delegate emit hint 里广告了
`static_recon_specialist`，所以 LLM 发起的 dispatch **不带 readonly**，会拿到 worktree
和完整的 `DEFAULT_SPECIALIST_TOOLS`（Bash, Edit, Write, MultiEdit, Task）。
(d) `test_per_domain_prompts.py:151` 只断言 "Never write a patch" **存在**，
从不检查授予是否**缺席** —— 矛盾被测试锁死，而非被测试阻止。
(e) 生产路径 `phases/internal.py:48-62` 的 research-scout dispatch 设了 `readonly:True`
却不传 mode/lane，于是 `resolve_specialist_profile` 返回 `mode='patch', lane='gpu'`
—— 只读 scout 的 prompt 在 patch profile 下渲染。

**范围界定**：`bench` 并非完全失效，它有一条**间接**入 prompt 的路径
（`bench=True → reserves_benchmark_lane → needs_gpu → GPU lease → allocated_gpu_ids`
→ 门控 16 行 `_gpu_autonomy_block` 并翻转 Iron Rule 1）。**只有 `lane` 是纯死的。**

**修法**：复用 `:2063` 已验证的单行 `if` 模式。在 `_section_identity` 中把无条件的
"**author source patches into your isolated worktree**" 换成
`inp.mode == 'patch' and inp.domain.key not in _READ_ONLY_DOMAINS` 的分支；
只读分支输出 "read any code under the framework source roots, search GitHub, and probe the
host via Bash" 并去掉 worktree 从句。然后删掉现已冗余的只读铁律行。
加测试：`mode='research'` 及三个只读 domain 下断言 `'author source patches' not in system_prompt`。

---

### C7 —— robustness prompt：加载、传输、丢弃，且已与代码分叉（medium / 北极星 + 特殊情况分离）

**问题**：148 行 prompt 每 tick 从磁盘读出、传入 backend、在函数体第一句被删除；
其中约 35 行是内联的 if-X-then-emit-Y 例外规则，且已与执行中的 reactor 分叉。

**证据**：`robustness_agent.py:162` `del system_prompt, tools, max_turns` 是 `run()` 体的第一句；
`robustness_mock.py:53` "Unused; accepted for protocol parity."
`cli/backends.py:233-243` 只能产出这两个 backend —— **全仓无任何路径为 robustness 构造 ClaudeBackend**。
文件本身在 `:3-7` 承认了这一点，却同时断言
"it documents the same contract the subprocess reactor enforces in code so behaviour stays aligned across paths"。
**这个对齐断言是假的**：`grep -rn 'phase_budget_nearly_exhausted|conversation_no_progress|
specialist_stale|recipe_kb_pending_backlog' src/` **只命中这一个文件**（`:26/:34/:42/:51`）。
它还指示 agent 去读一个 Coordinator **刻意从不发出**的块：
`:30-34` 讲 `=== Specialist health ===`，而 `conversation.py:598` 写着
"NOTE: there is deliberately no '=== Specialist health ===' block"
—— 并且这个缺席被**两个测试锁定**（`test_role_realignment.py:351`、
`test_coordinator_async_batch2_unit.py:1323`，都断言 "Specialist health" 不在 prompt 中），
`conversation.py:598-609` 还有 12 行注释解释原因。
`:52` 的拼写错误 "The flusher daemon should be drainsing it" 至今存活，说明无人演练此文件。
另外三个它命名的配置键（`specialist_stale_sec`、`recipe_kb_pending_alert_threshold`、
`recipe_kb_pending_alert_window_sec`）在 `agents/robustness/config.py` 与 `cli/parser.py` 中**都不存在**，
其中一个还被描述为 "configurable via CLI"；
`## Tool access` 承诺 Read + 只读 Bash，而 `policy/gate.py:766-773` 给非 orchestration 角色的
工具列表恰好是 `["emit_intent"]`；`:90` 声称 "22 patterns"，实际 `local_probe.py:66` 有 27 个，
且常量在 `sources/local_probe.py` 而非它引用的 `signals/local_health.py`。

**影响**：零运行时模型成本（无人读），但维护成本高且是**认知陷阱** ——
维护者据以理解 robustness 行为的文档，描述了四个 reactor 无法产生的症状。

**范围界定（重要）**：这**不使 robustness 的北极星"不可证伪"**。
该角色的使命由代码密集地钉住：20 个 signal 模块、固定的 symptom→intent `ActionLadder`、
35 个文件的测试套件（`test_role_contract.py` 钉住 IntentType/PAYLOAD_REQUIRED，
`test_decision_action_ladder.py` 钉住 symptom→intent 对）。
文件的 symptom-families 表、intent 白名单、pipeline 描述与产物说明**都是对的**。
另外所谓"prune_branch 自相矛盾"是误读：`:26-29` 说的是 LLM 不要自己发，
`:106` 说的是确定性 ladder 会自动发 —— 一致，不矛盾。

**顺带发现的更强问题**：`conversation.py:621/:631` 确实为 robustness 渲染了
`=== Phase budget telemetry ===` 与 `=== Conversation progress ===`，
但 `role/prompt_inputs.py::_split_sections`（`:245-282`）只识别
shared_state / inbox / time_budget / kb 四个 header，**静默丢弃其余** ——
**活路径上的死遥测**，这比 prompt 分叉更值得修。

**修法**：优先 (a)：删除 `robustness.md` 及其 role 加载路径，
把仍然正确的部分（symptom-families 表 `:87-97`、capability boundaries `:113-118`）
移入 `src/hyperloom/agents/robustness/README.md` 作人类文档，
在 `robustness_agent.py` 留 5 行 docstring。
无论如何都应加一个 CI 检查：文件里每个 `summary='...'` 字面量必须在
`agents/robustness/signals/` 中解析得到。
并单独修 `_split_sections`，让那两个遥测块要么被消费、要么不要渲染。

---

### C8 —— kernel prompt 被 Python 常量遮蔽，两份都不含成功度量（medium / 北极星）

**问题**：文档化的 kernel prompt 在正常运行中从不加载；实际生效的是一个 26 行常量；
两者都没有说明"什么样的 kernel 算好"。

**证据**：`cli/__init__.py:2095`（由 `:2094` `if not no_kernel:` 守卫）
`prompts["kernel_agent"] = args.kernel_prompt or _DEFAULT_KERNEL_PROMPT`，写入 `:2096`；
`conversation.py:720-721` 在查 role 文件**之前**返回 override。
于是 65 行 / 827 tok 的 `kernel_agent.md`（含 IR-1..IR-7 与 phase-incompatible 回复契约）
在所有未加 `--no-kernel` 的运行中是死的，被 26 行 / 1,167 字符 / 292 tok 取代。
`grep -niE 'gain|speedup|1\.05|1\.5'` 在**两个文件上都返回退出码 1（零命中）**。
真实门槛写在 `agents/kernel/SKILL.md:396`（`**Target speedup**: >= 1.05x`）
—— 一个**没有任何 loader 读取**的文件，且它自己记录了一处刻意分叉：
"Prompt still tells agents to aim for `>= 1.50x` ... but the KEEP gate is 1.05x (issue #442)"。
**这句话现在处处为假**：全仓幸存的唯一 `1.50x` 字符串是
`parallel_e2e_runner.py:304` 的 argparse 帮助文本，从不发给模型。

**范围界定（怀疑者的修正，须保留）**：缺失成功度量**不是功能缺陷**。
kernel LLM 角色**结构上无法也无需自评**：`_KERNEL_INTENTS` 只允许 RESPONSE 与 UPDATE_STATE，
工具列表只有 `["emit_intent"]`（`gate.py:766-773`），6 个已注册 kind 全部程序化执行。
真实的门是确定性且基于测量的：`request_handlers.py:6083-6086`
`decision = KEEP if gain_pct > keep_threshold_pct`（默认 1.0，`:5894`），
microbench `KEEP_THRESHOLD = 1.10`（`kernel_optimization.py:3307`）。
在 kernel prompt 里加阈值会是**惰性文本**。
而且 override **并非**文件的精简子集：它额外加了 native-only 规则与 SESSION_DIR 路径契约
（`kernel_agent.md` 没有），并正确指示了该角色真正执行的心跳行为。
**正确的严重度是文档腐烂**（一个 65 行的死 prompt 文件 + 一句描述不存在指令的陈旧分叉说明），
不是运行时正确性缺口。

**修法**：删除 `cli/__init__.py:250` 的 `_DEFAULT_KERNEL_PROMPT` 让 role 加载 `kernel_agent.md`，
或把 override 独有的三条（native-only 规则、SESSION_DIR 契约、心跳）并入 `kernel_agent.md` 后删除 override。
加测试：默认 CLI 参数下断言 `system_prompt_overrides.get('kernel_agent') is None`。

---

### C9 —— catalogue 广告的 kernel kind 没有 handler（medium / 渐进式披露）

**问题**：orchestration action catalogue 让 agent 发出的若干 kernel request kind
在 handler 表里不存在，会落到那个 26 行、对这些 kind 毫无操作指引的 LLM 上。

**证据**：渲染 catalogue（`orchestration.prompt.md:144-149`）发出
`EMIT: REQUEST{target_agent='kernel_agent', kind='operator_tuning', ...}` 与 `kind='vendor_kernel_config'`。
`KERNEL_REQUEST_HANDLERS`（`request_handlers.py:6255-6262`）只有 6 个键
（trace_analyze / run_gemm_tuning / run_fusion / run_optimization / integrate / apply_patch）
—— 执行 `get_handler()` 对这两个 kind 均返回 `None`（`intent_router.py:574` 是查找点）。

**证据修正（两处，不改变结论）**：
(1) 覆盖这两个 kind 的文件**不是** `agents/kernel/SKILL.md`（对它们零命中），
而是 `orchestrator/prompts/kernel_agent.md:16-17` —— 该文件**确实有** Python reader
（`AgentRole.load_system_prompt`），但被 C8 的 CLI override 遮蔽了。
(2) 数量是 **3 个而非 2 个**：`deep_kernel_analysis`（prompt `:117`）
也经 `_format_emit_hint` fall-through 而无 handler。

**影响（有界）**：kernel 角色工具列表只有 `emit_intent`，
所有 SharedState 写回都在程序化分支内，`optimization_stack`/`cumulative_gain`
是该角色无法写入的 `CORE_STATE_FIELDS`，且默认 geak backend 顺序下
`_on_enter_kernel` 会吃掉整个 KERNEL phase。
但**反向也存在**：`dispatcher.py:1102-1106` 刻意豁免了无 handler 的 kind 的 baseline 前置门
—— 这些路径比有 handler 的路径**门更少**。

**修法**：要么在 `KERNEL_REQUEST_HANDLERS` 注册这三个 kind，
要么把它们从 `default_enabled_actions` 移除以停止广告。
若必须保持 LLM 驱动，则为这三个 kind 扩写 `_DEFAULT_KERNEL_PROMPT`，
并加一条启动断言：凡 EMIT hint 形如 `REQUEST{target_agent='kernel_agent'...}` 的 catalogue 条目，
必须有 handler 或有 prompt 覆盖。

---

## 四、被证伪的怀疑

以下都是**看起来像缺陷、实际不是**的说法。列出它们是为了防止过度纠正 ——
按这些说法去改，会把设计改坏。选 6 条最有教益的（共 15 条被证伪）。

### R1 —— "system prompt 每轮重发 = 每轮 12k token 的税"

**说法**：47.8k 字符 system prompt 让 DELTA 轮出现 20:1 的静动比，
always-on floor 是 ~11.9k tok，SEED/DELTA 压缩因此被抵消。

**为何失败**：分母算错，且被仓库自己的生产遥测推翻。
orchestration backend 是 conversational 的，每轮都带着 resume 的会话。
两个真实多 tick session 中，**单轮 orchestration 总上下文的最小观测值就是 132,674 tok**
（中位 156,538 / 245,105）—— system prompt 是"最薄的一轮"的 8–12%，**不是 floor**；
根本不存在 14k token 的轮次。
真实 DELTA 轮的中位切分是 fresh input **4 tok** / cache_creation 2,310 / cache_read 149,707
—— system prompt 在 cache_read 里，按 0.1× 计费，边际成本约占该轮的 6.6%。
聚合看，system prompt 占两个 session orchestration 输入侧的 4.85% / 8.92%，
而 orchestration 本身只是 session 开销的少数派（30B session：orchestration 7.76M vs specialist 81.6M）。
整份 prompt 约占 session token 总量的 **0.3–0.5%**。
更关键的是**该说法把缓解机制当成了缺陷**：phase-invariance 是刻意的
（`SECTION_ORDER.md:5-6` 明写），`conversation.py:370-382` 把变动的 `=== Phase ===`
放进每 tick 的 **user** 轮，正是为了让 system prompt 保持字节稳定、可缓存的前缀。
把它改成 phase-conditional 会在每次 phase 转换时击穿缓存前缀，
用 ~1% 的上下文节省换取按 1.25× 计费的重复 cache-creation。
"reseed 被 prompt 膨胀推动"也是反的：reseed 由 `CheckpointPolicy` 的 140,000 token
软阈值触发，实测触发时上下文中位 212,625 / 245,080 —— 那是会话历史撑破的，不是 11.8k 前缀。

**残留为真**：死掉的 `kernel_enabled` 参数（装饰性），PRELUDE/CLOSE 确实携带 ~117 行
EXPLORE 专属内容 —— 约 1.4k tok，缓存服务，是整洁性偏好而非数量级缺陷。
（这就是 C2 被收窄为"结构缺失"而非"成本缺陷"的原因。）

### R2 —— "MISSION 不提 accuracy，所以 agent 没有正确性先验"

**说法**：MISSION 13 行纯吞吐，agent 会按纯吞吐排序候选，
在从不合格的杠杆（激进量化、eager 关闭、投机解码）上烧 benchmark 周期。

**为何失败**：审计只 grep 了字面词 "accuracy"，漏了实际使用的缩写。
`grep -c "acc_risk=" orchestration.prompt.md` = **14** ——
catalogue 里**每一个** action 都在用于排序的同一行 cost/gain/risk 中渲染 accuracy 先验，
例如 "cost ~10min  gain 0-12%  acc_risk=0.10  crash_risk=0.15"，其中 6 个带非零 acc_risk。
被斥为"payload schema"的 `:95` 实为 catalogue 表头，明写条目携带
"phase / typical wall-clock / expected gain range / accuracy_risk / crash_risk"。
更糟的是，被引为"真实执行规则"的 `integrate_patch.md:73-75` 是**一个死文件**
（正是该审计自己控诉的错误：`registry.py:7` 声称 lazy load 但无 loader），
而且它引的数**是陈旧的**：live code 是 `_accuracy_gate.py:27` `ACCURACY_THRESHOLD = 0.05`
（5 个百分点），文档写的 ">1pp" **松了 5 倍** —— 照它去改会把一个错误的数字注入 prompt。
门也不是无条件合取：`explore.py:1311-1322` 仅在 `is_high_accuracy_risk()` 命中
**且** `baseline_accuracy > 0` 时才对 serving framework 启用 accuracy 门；
`integrate_patch.py:2677` 的 `acc_required` 默认 `fw_authored`（即默认 False）。
被称为"从不合格"的三类杠杆恰恰是 `_HIGH_RISK_CLI_PATTERNS`（`--kv-cache-dtype`、
`--enforce-eager`、`--compilation-config`、`--attention-backend`）——
**这个门存在的目的就是有条件地放它们进来**，候选生成器还主动标注
"--kv-cache-dtype fp8_e4m3 ... (gate accuracy!)"（`specialist_prompt_builder.py:123`）。
真实 capture `tick30_orch_prompt.txt:182-189` 显示 Research Scout 发现以
`{"accuracy_risk": "None (tiling config only; numerically identical)"}` 的形式动态到达。

**残留为真**：MISSION 那一段措辞确实是纯吞吐的（这就是 C8 存活的窄版本）。

### R3 —— "orchestration prompt 内部大量重复，两处决策排序互相矛盾"

**说法**：a/b/c/d 决策排序陈述两次且不一致，profile/roofline 禁令重复四次，约 30 行冗余。

**为何失败**：**两处排序并不冲突**。Sec7 是 Sec5 的**保序无损压缩**：
(a)=Sec5 a；(b) 合并 Sec5 的 b/c/d（同层级）；(c)=Sec5 e；(d)=Sec5 单独编号的第 6 项（phase budget）。
相对优先级完全一致，不存在模型"必须二选一"的层级。
而且 Sec5 本身就声明了子层级不具约束力（"These are reference heuristics and objective facts,
not a forced sequence" / "There is no system-side priority list"）。
"~30 行"高估了 **3–6 倍**：真正可删的是 Sec7 的 5 行（`636-640`）加 4 行硬规则（`710-713`），
即 836 行中的约 5–10 行（~1%）。要凑到 30 行必须把 Sec5 那 24 行 a-e 块算作冗余，
但该块**独有**地承载了 `=== Current gaps ===` 行 schema、
`last_action_failures + winners_history` 回退，以及"该段缺失意味着 baseline 未完成"的指示。
禁令实为 5 次（70/173/573/710/715）而非 4 次，但**只有 710 是纯重述**：
`:70` 解释为何这些名字不在每 phase 的 allowed 列表里，`:173` 教 watermark 刷新机制
（禁令只是以 "(see Hard rules)" 结尾的一个从句），`:573` 是限定目标列表的插入语，
`:715` 是一整段新分析生命周期正文的标题。这是刻意的路标，不是失控的漂移。
另外 PolicyGate R1 在 `gate.py:1202` 已硬性拒绝 profile/roofline 并向 inbox 发出纠正提示
—— prompt 的冗余是在为一个**已被强制执行的不变量**兜底，行为风险为零。

### R4 —— "1,363 行文档语料的边界情况被'重新内联'回了主 prompt"

**说法**：因为 externalized 的家不可达，`recover.md` 的内容被内联成 FAILURE RECOVERY，
`paths.md` 被内联成 SESSION_DIR 块。

**为何失败**：**因果链在文本上被推翻**。token 重叠测试显示
`recover.md` 中 `fingerprint`、`last_action_failures`、`baseline_failure_streak`、`RULE`、
`benchmark_script`、`no_report`、`subprocess_nonzero`、`policy_denial_streak`
**出现次数全为 0** —— 那 81 行 prompt 块的词汇它一个都没有。
`recover.md` 讲的是 GPU 显存泄漏清理执行器（rocm-smi 前后探针、SIGTERM→SIGKILL 属主阶梯、
result.json schema），并明写 "`recover` is **not** for recovering from a workload-level
KEEP/REVERT regression"，而且 recover 是 `ROBUSTNESS_DELEGATE_ONLY_ACTIONS` 成员，
PolicyGate 对 Orchestration 完全禁止。prompt 里的 FAILURE RECOVERY 讲的是提案指纹重试纪律
（RULE F1-F4）。**主题不同，只共享 "recovery" 这个英文词。**
`paths.md` vs SESSION_DIR 块同理：两边只共享 `SESSION_DIR` 这一个 token。
更根本的是**受众错配**：`SKILL.md` 开篇即 "You are the launcher and monitor."，
`README.md:60` 把 `references/` 描述为 "SKILL reference chapters"，
`operations.md` 标题是 "Launcher Operations"（Setup / Launch Flags / Smoke Test / Resume / Monitoring），
`paths.md` 教读者用 `jq -r .session_dir <launch-info-file>` 并重跑 install.sh。
**在环的 Orchestration LLM 这些都不做** —— 它从不跑 install.sh、从不选 session dir
（SESSION_DIR 每 tick 由 `conversation.py:369-370` 字面注入）。
所以这 1,363 行不是"模型被剥夺的恢复知识"，而是**面向另一个 agent 的另一层文档**。
仓库在受众**匹配**处证明了自己会接线：`critic_agent.py:1232` 确实加载 critic 的
`SKILL.md` + `actions/review_coordinator_inbox.md`。
另外"零入站指针"在字面上也有例外：`gate.py:1426` 的拒绝提示经 `writeback.py:190`
作为 policy_denied observation 推上 agent bus，是一个**真正到达模型**的指针。

**残留为真**：两处 docstring 的 loader 归属错误（C3 的窄核心）。
但注意：`specialist.md` 于 2026-08-01 在 `b05077444` 中与 12 个 .py 文件**同一 commit** 更新，
且 130 行实质内容与 builder **零逐字重叠** —— 它不是陈旧的重复品，
而是那套设计**唯一的散文描述**。

### R5 —— "specialist 的 mode/bench/lane 三个 dial 全是死的，造成实时权限矛盾"

**说法**：三个 dial 都不进 prompt，所以 `mode=research` 产出一份叫 agent 去写 patch 的 prompt。

**为何失败**：**`bench` 确实进 prompt**，只是经由分配而非直接读取：
`bench=True → SpecialistProfile.reserves_benchmark_lane（profile.py:82）→
params.setdefault("needs_gpu", True)（phases/explore.py:877-879）→ GPU lease →
extra_context["gpu_ids"]（dispatcher.py:513）→ allocated_gpu_ids（runner.py:558）`
→ 门控整个 16 行 `_gpu_autonomy_block`（builder `:858-859`）
**并翻转 Iron Rule 1** 在"你独占这些卡"与"你没有 GPU 分配……绝不碰 8888 端口"之间（`:1889-1905`）；
它还决定 wall budget（`dispatcher.py:714`，`60.0 if needs_gpu else 10.0`），而那是会渲染的。
审计用来证明"无影响"的那个 empty diff 是**测试夹具的产物**：
`render_prompts.py:250-258` 的 `COMMON` 里 `allocated_gpu_ids=(4, 5)`，
被比较的两侧都继承了它 —— 这个对比**恰好把真正起区分作用的变量held constant 了**。
审计自己的另一份夹具反而证明了相反结论：
`specialist.mode-research.lane-cpu.system.prompt.md` 对
`research_scout_specialist.system.prompt.md` 的 diff 有 **23 行差异**，整块 autonomy 消失、Iron Rule 1 改写。
只读指令也**确实是条件渲染的**：`_DOMAIN_FOCUS_TEMPLATES`（`:832-838`）
只为只读 domain 发出 "Iron rule — read-only" 块 —— domain 级的渐进式披露存在，审计报告为不存在。
数量也不对："3 of 10 domains" 实为 2（`pr_intel_specialist` 根本没有只读铁律）。

**残留为真**：**只有 `lane` 是纯死的**，Section 1 的笼统授予与 Sections 8/9 未按 domain 门控
—— 这正是 C6 存活的形态（C6 的定性是"语义缺陷/措辞"，不是"实时权限风险"：
Section 1 明确让位 "The hard capability boundary is fixed by Section 9 Iron Rules"，
且 section-8 契约对只读 agent 自我失效 —— "Empty list = no patches; downstream
integrate_patch action skips when empty"）。

### R6 —— "critic 收到两套互斥的 verdict 分类法，会橡皮图章放行未验证的源码补丁"

**说法**：`action_verdict_policy`（archival/exploration/promotion）与
`proposal_action_classes`（patch_landing/...）对 `integrate_patch` 给出相反指示，
后果是未经验证的补丁被盖章进 optimization_stack。

**为何失败**：文本核对全部属实（真实 turn0 capture 里两套分类法**确实同时到达模型**），
但**声称的影响不成立** —— 该条按北极星级严重度归档，而严重度完全依赖那个假影响。
**critic 的批准并不落地补丁**：`integrate_patch` 是确定性 Python 执行器
（`orchestrator/actions/executors/integrate_patch.py`），它 git-apply、重启 server、
跑吞吐 benchmark 加 accuracy 门，产出 kept/reverted，任何失败自动
`_revert_patches` / `git reset --hard`；`writeback._promote_integrate_patch`
只在 `status == "kept"` 时抬升 current_best。
**那个被指称"被豁免"的前后对比 benchmark，是在 verdict 下游由代码无条件执行的。**
critic 的门只授权"去跑这次测量"。
第二个角也被缓解：`INTEGRATE_PATCH_PERMISSIVE_VERDICTS = {approve, advise}`（`gate.py:285`），
`critic.prompt.md:145-150` 明确鼓励宽松使用 advise，
而 `needs_review` 会进入 `_maybe_reauthor_from_critic_feedback`（`intent_router.py:342`）的**重写循环**而非停摆。
"exploration" 这个归类还是**刻意的**：`test_critic_verdict_map.py:987` 点名了
该 fallback 所要防止的 N33/N35/N37 先有鸡还是先有蛋死锁。

**残留为真**：两套词汇同时在场是真实的整洁性/一致性问题（这支撑了 2.1 节对 critic 北极星的 2.5 分），
但不是"补丁把关被架空"。

---

## 五、做得好的地方

1. **Orchestration 的使命句是教科书级的**：一句话、一个指标、一个插值出来的数字目标、
   一个每 tick 推送的实时读数（`orchestration.prompt.md:4-5` + `:23` + `render.py:132-133`）。
   835 行里只有 2 行匹配目标陈述正则 —— **没有竞争性使命句**。
   而且不动性是**结构保证**的：`_section_mission()` 零参数、builder 无 phase 参数。

2. **唯一可变段被防劫持**：CYCLE DIRECTIVE 显式标注 `(advisory — ...)`、限长 1500 字节、
   并被 `_DIRECTIVE_POLICY_BLACKLIST`（`'ignore phase'`/`'bypass policy'`/`'override policy'`，
   `orchestration_memory.py:116-124`）筛查。LLM 自撰的方向不能重定义使命。

3. **对自身指标的诚实**：prompt `:13-15` 主动说明 per-round gain 的和**不线性可加**，
   并同时渲染 raw 与 validated 两个数。带文档化 raw/validated 分裂的单指标，
   工程上优于一个悄悄撒谎的单数字。

4. **Kernel 角色证明了最好的披露是消除**：6 个 request kind 由 6,318 行确定性 Python 处理
   （`intent_router.py:573` `# Programmatic shortcut: run a registered kernel handler inline
   + emit RESPONSE without burning an LLM turn.`），于是 LLM 契约只需 **26 行 / 292 tok**。
   对照那份本来会需要的 570 行 SKILL.md。**例外活在可执行代码里，这是分离的最强形式。**

5. **Enablement ladder 是全仓唯一在运行时解析的指针**：
   `framework/SKILL.md:114-117` 声明 `build_enablement_ladder_book` 为权威，
   而 `specialist_prompt_builder.py:1970` **真的 import 并调用了它**。
   文本由被引用的源生成，**因此不可能漂移**。
   它还只注入一个 domain 的 **user** 半，docstring 写明理由
   （`:1960-1961` "Kept in the user prompt so the cached system prompt stays task-independent."），
   并且是**置换**而非追加 —— user section 从 13 个换成 7 个，
   理由同样在代码里（`:2064-2067` "the perf context ... is noise when the server cannot boot"）。

6. **Specialist 的 analysis.md 延迟是全仓最佳的按需机制**：user prompt `:72`
   同时给出路径、预期体积、**以及稳定小节标题的枚举**
   （"All section headings are stable: ``## Executive Summary`` / ``## Top Operations`` / ..."），
   让 agent 可以定向 seek 而不是盲读 20 KB。这正是 orchestration 侧想做而没做成的东西。

7. **压缩机制在失败时不破坏状态**：劣化的 checkpoint 回复**跳过压缩**、保留在飞会话与既有记忆，
   只重置计数器以免风暴，连续 3 次才升级严重度；
   `build_memory_record` 对 learnings 累积去重（上限 50）并做 non-empty-wins 前推。
   checkpoint prompt 还教模型该存什么：**"capture intent and rationale, not raw numbers
   you can re-pull from the context tools."**

8. **有数据支撑的"负空间"设计**：`conversation.py:598-609` 用 12 行注释记录了一个
   **不加某个 prompt 块**的决定 —— "Measured over a full 11.6h session: 33 renders,
   0 of them overlapped a live specialist... A block that always reports 'none running'
   is worse than no block, because it manufactures a false belief."
   并由两个测试锁定。这种决策质量在仓库里很罕见，值得保护。

9. **Retired-action 禁令放置正确**：prompt 只留一行事实（`rendered:689`），
   "do not re-introduce" 的政策流放到 `SKILL.md:277`（launcher 面向）。团队清楚这个受众区别。

10. **RecipeKB 通道是过滤后注入而非仅仅延迟**：`_section_pitfalls` 在为空时渲染占位符，
    KB 条目携带 `severity=high, conf=0.80, observed=3, src=s-093` 这样的出处元数据；
    static-recon 清单（`knowledge/static_recon_checklist.py`，370 行）
    按 `(model_class, gpu_type, precision)` **先过滤再注入**（5 条 → 3 条，30 行 → 18 行）。

11. **审计基础设施本身**：60 份带 manifest（记录 lines/chars/tokens 与确切 builder 调用）
    的渲染 artifact，加上一份诚实到会记录自己死参数的 `SECTION_ORDER.md`
    （"takes kernel_enabled but does not currently branch on it"）
    —— 这是这三项判据**能够被审计**的前提。

---

## 六、建议的改动

按 (影响 / 成本) 排序。

| # | 文件 | 改动 | 收益 | 成本 |
|---|---|---|---|---|
| 1 | `loop/conversation.py:461,463` | `obj.kind()` + 读 `target_gain_pct`；`Objective` ABC 加 `target()`；把测试里的 `_Obj` stub 换成真 `TargetGainObjective(20.0)` | 修复 C1：`throughput_below_target` 重新进入 `gaps[]`，`target_gap_pct` 不再假报 0 | **2 行 + 1 测试** |
| 2 | `roles/critic_agent.py:245-258` + `prompts/critic.md` | 把 61 行 cross-domain 块移入已存在的 `if not has_cross_domain: return` 分支，追加到 **user** prompt；从 critic.md 删 `90-150` | 修复 C5：单域审查的 system prompt 减 41%，消除"第一句叫你跳过这 61 行"的自反文本 | **~6 行** |
| 3 | `prompts/prompt_builder.py:65-66` | MISSION 加一条约束从句："...subject to a hard constraint: no KEEP may drop task accuracy against the sealed baseline. A gain that fails the accuracy gate is worth zero." **不要写死数字**（live 值 `ACCURACY_THRESHOLD=0.05`，文档里的 1pp 已陈旧 5 倍） | 修复 C8：让声明目标与执行目标同构，在 agent 最先读的那一段 | **2 行** |
| 4 | `actions/registry.py:7`、`actions/__init__.py:10` | 删掉两句 "loaded lazily" docstring（或实现 `ActionMetadata.body_md`）；加 pytest：存在 `<name>.md` 而无 reader 则失败 | 修复 C3 的窄核心 —— 阻止维护者继续往 /dev/null 写边界情况 | **2 行 + 1 测试** |
| 5 | `cli/__init__.py:250,2095` + `prompts/kernel_agent.md` | 把 override 独有的三条（native-only、SESSION_DIR、心跳）并入 `kernel_agent.md`，删除 `_DEFAULT_KERNEL_PROMPT`；加测试断言默认参数下无 override | 修复 C8：消灭一份被遮蔽的 65 行死 prompt，让维护者要编辑的文件就是生效的文件 | **~30 行** |
| 6 | `prompts/specialist_prompt_builder.py`（`_section_identity`） | 按 `inp.mode == 'patch' and inp.domain.key not in _READ_ONLY_DOMAINS` 门控那句 "author source patches"（复用 `:2063` 已验证的单行 `if`）；加测试断言只读 domain 下该串缺席 | 修复 C6：消除同一份 prompt 内的授予/撤销矛盾。**注意同时修 `phases/internal.py:48-62`**（它设 `readonly:True` 却不传 mode，导致只读 scout 在 patch profile 下渲染） | **~15 行 + 1 测试** |
| 7 | `roles/critic_agent.py:59-106` + `actions/review_coordinator_inbox.md` | 把 `risk_rules.md` 的 severity 定义内联进 `_REVIEW_OUTPUT_INSTRUCTIONS`，删除那条不可满足的 step-3 指针与 `SKILL.md:93-97` 的四行死路由 | 修复 C4：消除一条 always-loaded 的、物理上无法执行的指令 | **~20 行** |
| 8 | `agents/robustness/role/prompt_inputs.py:245` | 让 `_split_sections` 消费 `=== Phase budget telemetry ===` 与 `=== Conversation progress ===`，或让 `conversation.py:621/631` 停止渲染它们 | 修复 C7 顺带发现的**活路径**问题：当前这两块每 tick 计算、渲染、传输、丢弃 | **~10 行** |
| 9 | `prompts/robustness.md` → `agents/robustness/README.md` | 删除该 prompt 与其 role 加载路径，保留仍正确的部分（symptom-families 表、capability boundaries）作人类文档；加 CI 检查：每个 `summary='...'` 必须在 `signals/` 中解析 | 修复 C7：消灭 148 行"加载后即删"且已分叉的文本，堵住认知陷阱 | **~1 小时** |
| 10 | `kernel/request_handlers.py:6255` 或 `default_enabled_actions` | 为 `operator_tuning` / `vendor_kernel_config` / `deep_kernel_analysis` 注册 handler，或停止在 catalogue 广告；加启动断言：凡 `REQUEST{target_agent='kernel_agent'...}` 的条目必须有 handler 或有 prompt 覆盖 | 修复 C9，并顺带堵上 `dispatcher.py:1102-1106` 对无 handler kind 豁免 baseline 前置门这个反向缺口 | **~1 小时** |
| 11 | `prompts/orchestration.md:636-640,710-713` | 删除与 Sec5 重复的 5 行排序与 4 行硬规则；把 `_section_decision_framework` 的 `kernel_enabled` 死参数要么用起来要么删掉 | 小幅整洁 + 消除 `SECTION_ORDER.md` 自己记录的死钩子 | **~10 行** |
| 12 | `agents/framework/prompts/{explore_prs,enrich_candidate,isolate_and_run}.md` | 各加一行 `Refer to references/<x>.md for ...`，形式照抄 `filter_candidates.md:22` | 让 357 行 reference 在自己的文档集内不再是孤儿（低优先：该文档集整体无 loader） | **3 行** |

**明确不建议做的三件事**（依据第四节）：

- ❌ **不要**给 `build_orchestration_prompt` 加 phase 参数去做分段门控。
  phase-invariance 是刻意设计，它让 system prompt 成为字节稳定的可缓存前缀；
  改动会用 ~1% 的上下文节省换来每次 phase 转换的 cache 击穿（1.25× 计费）。（R1）
- ❌ **不要**把 `integrate_patch.md` 的 ">1pp" 或 `actions/*.md` 里的 `keep_threshold_pct` 数值
  写进任何 prompt。这些文档值已陈旧（live accuracy 门是 5pp；
  KEEP 阈值是 `0.1 + 0.9/N` 的**衰减函数**，`machine_state.py:434-450`，静态文档无法表达）。（R2）
- ❌ **不要**为了消除"重复"而删 Sec5 的 24 行 a-e 块。它独有地承载 gaps 行 schema、
  `last_action_failures + winners_history` 回退和"该段缺失=baseline 未完成"的指示。（R3）
