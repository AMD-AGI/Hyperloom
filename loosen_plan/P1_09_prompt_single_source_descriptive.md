# P1_09 — prompt 单一来源 + 命令式改描述式 + 同步已删守卫措辞

- **Phase**: P1 · **风险**: 中 · **依赖**: P1_01..P1_08 · **后继**: 与 P2/P3 各步收尾时再扫一遍

## 目标

prompt 是 LLM 的唯一行为来源。当前 prose(`orchestration.md`)与 builder(`prompt_builder` DECISION FRAMEWORK)里大量**命令式 MUST**、**"X 会被 deny"的描述**与 code 双写,既漂移又压制自由度。本步把面向 Orchestration 的 prompt 整体从"命令"改为"事实 + 目标",并删除所有指向**已在 P1_01..P1_08 删除的守卫**的描述,确保 prompt 与运行时一致。

## 不变量 vs 策略判定

- prompt 是引导,不是不变量。命令式步骤、breadth-cap 描述、"必须先 X 再 Y"= STRATEGY 的 prose 镜像,**改描述式或删除**。
- 仍需保留的**事实性**说明:SESSION_DIR 路径契约、kind 必须是 5 种之一(数据/handler 契约)、不可伪造 trace_input(数据契约)、kernel-owned 经 request(角色契约)——这些是不变量的 prose 投影,保留。

## 改动清单

### 1. `prompt_builder.py` DECISION FRAMEWORK(402–564)改描述式
- 把 F1–F4 失败恢复、IDEA GENERATION、"Stop/Measure/Analysis/Phase"步骤里的 **MUST/必须按序** 语气改为"这些是可参考的启发法 + 客观事实";明确"下一步由你判断"。
- 删除引用已删 cap 的句子(specialist/dynamic ≤1、≥2 域、kernel_opt 前必须 explore 等,随 P1_05/P1_06)。
- 保留:baseline 先行(不变量)、kernel-owned 经 request(角色契约)、不可提 profile/roofline(随 P1_01 改为"由 Coordinator 自动管理")。

### 2. `prompt_builder._section_pipeline_and_budget`(219–253)/ `_section_phase_semantics`(149–168)
- phase 语义:保留"transitions 由 Coordinator 拥有"(若 P3_18 落地再改为"你可显式请求前进");删除已不存在的强制顺序描述。
- 与 P1_01 一致渲染 `PHASE_LLM_PROPOSABLE_ACTIONS`。

### 3. `orchestration.md`(rules fragment)全文梳理
- 删/改:
  - 16–22 "你不能决定切 phase"(P3_18 落地后改"可建议前进")。
  - 54–102 specialist 并发/`explore_specialist_grid_max_one`/proposal scores 命令式描述 → 随 P1_05 改事实式。
  - 152–208 Hard rules 里引用**已删 deny** 的条目(stack rebench required、validate_stack 系列、analysis-not-proposable 的双 reason-code 等)→ 删除或改为现状。
  - 181–188 config-only MUST → 删(P1_04 已处理 code 侧,这里删 prose)。
- 保留:SESSION_DIR 契约(131–151)、kind 5 选 1(154–159)、不可伪造 trace_input(160)、kernel-owned 不可 delegate(189–193)、emit_intent 输出协议(246–251)。

### 4. EMIT hints(`prompt_builder._format_emit_hint` 263–324)
- dynamic_action / specialist / explore 的 EMIT hint 删除已移除约束的 reason-code 列表与 cap 描述(随 P1_05/P1_08)。

## 连带测试

| 文件 | 函数 | 动作 |
|---|---|---|
| `test_prompt_builder.py` | `test_full_prompt_has_seven_sections`(110)、`test_*_phase_subheaders`(169)、`test_mission_*`(329)、`test_time_budget_*`(350)、`test_emit_hints_*`(244) | 章节结构基本不变;更新被改措辞的断言 |
| `test_role_realignment.py` | `test_orchestration_prompt_includes_phase_contract`(55)、`test_*_no_kernel_*`(72) | 同步 phase contract 措辞 |
| `test_orchestration_prompt_failure_policy.py` | `test_failure_recovery_*`(87,120) | F1–F4 改描述式后更新断言(或放宽为"包含恢复指引") |
| `test_drop_scoreboard.py` | `test_orchestration_prompt_has_no_scoreboard_block`(230) | 保持无 scoreboard;确认改写未引入 scoreboard 语 |
| `test_prompts_no_propose_roofline.py` | `test_no_prompt_instructs_llm_to_propose_analysis_action`(95) | 必须仍通过 |
| `test_dynamic_action_orchestration_prompt.py` | cap/iron-rule 可见性 | 同步 |

## 验证
- prompt 全文无引用已删守卫;无 config-only MUST;命令式步骤改描述式。
- `test_prompts_no_propose_roofline` / `test_drop_scoreboard` 仍绿。
- 人工通读 orchestration 完整 prompt 快照,确认"自由度 + 事实导向"。

## 回退
- 还原 prompt 文本与测试断言。

## 残留风险
- 中。prompt 措辞改动可能影响 LLM 行为分布——建议与 P2/P3 一起做一次小规模 A/B(同 workload,新旧 prompt 对比收敛性与产物形状)。本步应在每个 P2/P3 删除步落地后再回扫一次,保持 prompt 与运行时零漂移。
