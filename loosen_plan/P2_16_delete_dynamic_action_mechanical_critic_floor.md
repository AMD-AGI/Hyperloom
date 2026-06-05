# P2_16 — 删 dynamic_action 机械 critic floor(关键词 reject),交 LLM Critic

- **Phase**: P2 · **风险**: 中 · **依赖**: P1_08 · **后继**: P3_20

## 目标

dynamic_action 有一层**机械 critic floor**:用关键词子串匹配判断"跨域耦合/副作用/动机覆盖",在 LLM Critic 之前就 reject/revise,而且"strictest wins"——机械层只能比 LLM 更严。这是典型的"代码替 Critic 做策略判断"。删除机械策略层,交给 LLM Critic;只保留 provenance/schema/patch-match 等不变量。

## 不变量 vs 策略判定

| 项 | 位置 | 性质 | 处理 |
|---|---|---|---|
| 机械跨域检查(关键词 substring:耦合/副作用/动机) | `dynamic_action_critic.py` 242–351 | STRATEGY | **删除** |
| "strictest wins" floor(机械只能收紧 LLM) | `dynamic_action_critic.py` 174–190 | STRATEGY | **删除**(策略规则部分);只保留 provenance/schema 守卫 |
| pipeline 机械 pre-verdict 阻断 + `revise≡reject`(无 redispatch) | `dynamic_action_pipeline.py` 232–270 | STRATEGY | **降级**:机械 block 改 advisory;`revise` 增加 redispatch 或交 LLM 决定 |
| provenance / schema / patch 匹配 worktree | (proposal 层,P1_08 保留) | INVARIANT | 保留 |

## 改动清单(删除优先)

### 1. 删机械跨域 reject(`dynamic_action_critic.py`)
- 删 242–351 的关键词 substring 跨域检查(耦合/副作用/动机覆盖)。这些判断交 LLM Critic 的语义评审。
- 删 174–190 的机械 floor "strictest wins" 中**针对策略规则**的部分;若该函数还合并了 schema/provenance 守卫(不变量),拆出保留不变量部分。

### 2. pipeline 机械 pre-verdict 降级(`dynamic_action_pipeline.py`)
- `compose_critic_verdict_envelope`(232–270):机械 pre-verdict 不再**阻断**,改为作为**advisory 注入**给 LLM Critic 的判据;最终 verdict 以 LLM Critic 为准。
- `revise≡reject`:要么实现"revise → redispatch 一次让 dynamic_action 修正",要么把 revise 当 advisory 让 LLM Critic 决定 approve/advise/reject。删除"revise 直接等同 reject 且无重试"的死路。

### 3. 与 critic prompt 对齐(触类旁通,→ P3_20)
- `critic.md` 79–141 的 cross-domain 规则当前**镜像**机械层;删机械层后,这些规则保留为 **LLM Critic 的判断指引**(prose),不再有 code 双写。P3_20 统一 critic 策略 vs 安全。
- `backends/critic_agent.py` 283–308 注入 `cross_domain` 约束:改为注入为**hints**(advisory),不作为上游硬 deny。

## 连带测试
- dynamic_action critic / pipeline 测试:删除机械 reject/floor 的用例;**保留** provenance/schema/patch-match 用例。
- 以实际 grep 该模块测试为准(审计未逐一列出 dynamic critic 测试函数)。

## 验证
- dynamic_action proposal 的跨域质量由 LLM Critic 评审(approve/advise/reject),无关键词机械 reject。
- 伪造/越权仍被不变量层拦(provenance、schema、patch 匹配 worktree、PolicyGate 红线)。
- `revise` 不再是死路。
- 烟测:一个跨域 dynamic_action proposal 经 LLM Critic 正常裁决并(若 approve)integrate。

## 回退
- 恢复机械 critic 层与 pipeline 阻断、测试。

## 残留风险
- 中。去掉机械关键词层后,质量完全依赖 LLM Critic——与项目"以 LLM 判断为主"的方向一致。安全维度(patch 匹配 worktree、红线副作用、accuracy gate)仍由不变量层 + Critic 安全规则(P3_20)兜底。
