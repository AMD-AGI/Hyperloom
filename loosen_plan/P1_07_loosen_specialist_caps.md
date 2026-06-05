# P1_07 — 放宽 specialist 上限(proposal_set 截断 / max_turns / scorer)

- **Phase**: P1 · **风险**: 低 · **依赖**: 无 · **后继**: 无

## 目标

让 specialist"探得深"。当前 specialist 的产出与深度被几处硬上限压制,都是 STRATEGY。

## 当前上限(审计)

| 项 | 位置 | 当前 | 性质 |
|---|---|---|---|
| proposal_set 截断 | `specialist_runner.py` 511–515, 810–831 (`DEFAULT_SPECIALIST_MAX_PROPOSALS=3`) | 丢弃第 4+ 个提案 | STRATEGY |
| specialist max_turns 默认 | `specialist_domains.py` 278–285 (`DEFAULT_SPECIALIST_MAX_TURNS=8`,硬上限 16) | 8 | STRATEGY(默认)/ INVARIANT(16 上限) |
| proposal scorer 截断 | `proposal_scorer.py` 70–71, 310–311 (`_MAX_PROPOSALS_SCORED=12`) | 只打分前 12 | STRATEGY |
| 子进程 wall-clock kill | `specialist_subprocess.py` 94–95, 368–526 | = max_turns × per_turn | INVARIANT(防 runaway,保留) |
| CPU specialist 隐藏 GPU env | `specialist_subprocess.py` ~339 | — | INVARIANT(保留) |
| 工具 denylist(KB 写) | `specialist_runner.py` 134–137, 300–337 | — | INVARIANT(保留) |
| 空 domain → 空 done | `specialist_runner.py` 395–414 | 强制空退出 | STRATEGY(可降级) |

## 改动清单(删除优先 / 放宽)

### 1. 删/放宽 proposal_set 截断(`specialist_runner.py`)
- 删除 511–515 / 810–831 对 `proposal_set` 的硬截断(到 3),改为**不丢弃**(全部带回),或把截断数调到与 scorer 一致的较大值。截断属"代替 LLM 取舍",删除优先。
- `DEFAULT_SPECIALIST_MAX_PROPOSALS`(若整删截断则删常量)。

### 2. 提高 specialist max_turns 默认(`specialist_domains.py` + yaml)
- `DEFAULT_SPECIALIST_MAX_TURNS` 8 → 更高(如 12 或 16)。**保留**硬上限 16(`SPECIALIST_MAX_TURNS_HARD_CAP`,资源)。
- 同步 `actions/_meta/specialist.yaml`(27)max_turns 默认。

### 3. 提高 scorer 截断(`proposal_scorer.py`)
- `_MAX_PROPOSALS_SCORED` 12 → 与 specialist 产出上限匹配(或更高)。scorer 是 advisory(不 gate),截断只影响展示完整性。

### 4. 空 domain 处置(可选,降级)
- `specialist_runner.py` 395–414:未知 domain 当前强制空 `specialist_done`。改为返回**advisory 错误 payload**让 LLM 可重试/换 domain,而非静默空退出。**保留** domain tag 的 PolicyGate 校验(P1_05 已保留)。

## 连带测试

- 搜索 `DEFAULT_SPECIALIST_MAX_PROPOSALS` / `_MAX_PROPOSALS_SCORED` / `DEFAULT_SPECIALIST_MAX_TURNS` 的断言测试(specialist_runner / proposal_scorer / per_domain_prompts 系列)并更新默认值断言。
- `actions/_meta/specialist.yaml` 若被 `test_action_catalogue` / registry 测试读取,更新期望 max_turns。

> 审计未列出针对截断数的专门测试;以实际 grep 为准更新。

## 验证
- specialist 返回 >3 提案时不被截断(或截断阈值显著提高)。
- max_turns 默认提高后,specialist 子进程 wall-clock = max_turns × per_turn 相应放大,但仍被 kill 兜底。
- 烟测:一次 specialist 派发产出多提案,全部进入 explore grid 候选。

## 回退
- 还原常量默认与截断块。

## 残留风险
- 低。提高 turns/产出会增加单次 specialist 时间与 token——由子进程 wall-clock kill(保留)与 lane 租约兜底。scorer 截断放宽仅增展示,不影响裁决(advisory)。
