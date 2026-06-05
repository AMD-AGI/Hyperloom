# P1_08 — 放宽 dynamic_action ReAct 上限 + 删机械字段校验(非红线)

- **Phase**: P1 · **风险**: 中 · **依赖**: P1_05 · **后继**: P2_16

## 目标

让 dynamic_action(跨域 ReAct 探索通道)探得更深、更自由。当前一堆 turn/拒绝/字符/seed-kit token 上限与机械字段校验把它压得很死,多数是 STRATEGY 或 anti-gaming 噪声。

## 当前限制(审计)

| 项 | 位置 | 当前 | 性质 |
|---|---|---|---|
| ReAct turn 上限 | `dynamic_action_runner.py` 72–73 (`DEFAULT_TURN_CAP=12`) + env/CLI | 12 | STRATEGY(可放) |
| wall-clock | 同上 (`DEFAULT_WALL_CLOCK_BUDGET_SEC≈900`) | 15min | STRATEGY/资源边界 |
| 连续坏 turn 中止 | 225–354 (`MAX_PROPOSAL_REJECTS=2`) | 2 次即中止 | STRATEGY |
| proposal_set 长度 | `dynamic_action_proposal.py` 256–259 (`MAX_PROPOSAL_SET_LEN=1`) | 1 | INVARIANT(integrate 契约)→ 保留 |
| 禁字段 + 数值声明正则 | 237–246, 359–464 | 禁 gain/score 字段 + 文本数值正则 reject | 混合 |
| scope_domains ⊆ spec | 398–413 | — | INVARIANT(保留) |
| patch 必须匹配 worktree diff | 424–437 | — | INVARIANT(保留) |
| 每域 substring 出现校验 | 439–452 | rationale 必须含每域关键词 | STRATEGY |
| motivation 短文本上限 | 213 (`MOTIVATION_GAP_SHORT_MAX_CHARS=200`) | 200 | STRATEGY(context 预算) |
| seed-kit token 上限 | `dynamic_action_seed_kit.py` 43–56,134–220 (`MAX_SEED_KIT_TOKENS=8000`,切片 6/20/10) | — | STRATEGY |
| seed-kit 过滤 kernel-only patch | 90–95, 151–175 | 丢弃 | STRATEGY |
| 工具 whitelist / 读上限 / bench wall 60s / deny segments | `dynamic_action_tools.py` 49–100, 336–354 | — | INVARIANT(保留) |
| bench tool 关闭 | `dynamic_action_tools.py` (`BENCH_TOOL_ENABLED_V1=False`) | 关 | STRATEGY(可开) |

## 改动清单(删除优先 / 放宽)

### 1. 放宽 turn / 拒绝中止(`dynamic_action_runner.py`)
- 提高 `DEFAULT_TURN_CAP`(12 → 更高)与 `DEFAULT_WALL_CLOCK_BUDGET_SEC`(已有 CLI/env `--dynamic-action-turn-cap` / `--dynamic-action-wall-clock-sec`,改默认即可)。
- `MAX_PROPOSAL_REJECTS`(2):提高或删除"连续坏 turn 即中止 run"的逻辑,改为**记日志继续**(让 ReAct 自行收敛),仅 wall-clock/turn 上限兜底。

### 2. 删机械字段校验(非红线)(`dynamic_action_proposal.py`)
- 删 359–464 中的**数值声明正则**(把它降为 advisory 警告,不 reject)与 439–452 的**每域 substring 校验**(交由 LLM Critic 判断深度)。
- **保留**:禁 gain/score 字段(anti-gaming,防伪造测量,保留)、`scope_domains ⊆ spec`(398–413)、patch 匹配 worktree(424–437)。

### 3. 放宽文本/seed-kit 上限(`dynamic_action_proposal.py` / `dynamic_action_seed_kit.py`)
- `MOTIVATION_GAP_SHORT_MAX_CHARS`(213)提高或按 token 预算比例化。
- `MAX_SEED_KIT_TOKENS` 与切片(43–56, 134–220)提高;seed-kit"过滤 kernel-only patch"(90–95, 151–175)降级为**advisory 标记**(带回但标注),不直接丢弃。
- `dynamic_action_resume.py` 255–258 的 motivation 截断:提高上限。

### 4. 保留的红线/资源(不动)
- 工具 whitelist、读上限、bench wall 60s、session deny segments(`dynamic_action_tools.py`)= INVARIANT。
- `MAX_PROPOSAL_SET_LEN=1`(proposal 256–259)= integrate 契约,保留(文档标注为不变量而非 breadth cap)。
- PolicyGate dynamic 红线(`policy.py` 2152–2168, 2243–2254)= 保留。

### 5. 同步 prompt(触类旁通)
- `system_prompts/dynamic_action_prompt_builder.py`(29–30, 158–214 "iron rules" / token caps):与放宽后的 runner 校验对齐,删除已不再强制的"iron rule"。
- `actions/dynamic_action.md`(60–65)、`actions/_meta/dynamic_action.yaml`(18,22):max_turns 同步。

## 连带测试
- `test_dynamic_action_dispatch.py` / `test_dynamic_action_invariants.py`:删除针对已删机械校验(数值正则、每域 substring)的用例;**保留**红线/schema/patch-match 用例。
- dynamic_action seed_kit / resume / proposal 的截断/token 测试:更新默认值或删除。
- `test_dynamic_action_orchestration_prompt.py`:同步 prompt 中可见的 cap 值/iron-rule 断言。

## 验证
- dynamic_action 可跑更多 turn、更长文本、更大 seed-kit;连续坏 turn 不再过早中止。
- 红线仍拦截:伪造 gain/score 字段、kernel-only 副作用、越权 patch。
- 烟测:一次 dynamic_action 派发完整跑完并产出可 integrate 的 proposal。

## 回退
- 还原各常量默认与删除的校验块、prompt iron-rules、测试。

## 残留风险
- 中。放宽后单次 dynamic_action 更耗时/耗 token——wall-clock/turn 上限 + lane 租约兜底。删数值正则后,LLM 可能在 rationale 里写非测量数字——由 Critic(P2_16/P3_20)与"禁 gain/score 字段"(保留)共同防伪。
