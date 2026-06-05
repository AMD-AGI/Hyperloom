# P3_20 — critic 只在安全维度 reject,策略维度降 `advise`

- **Phase**: P3 · **风险**: 中 · **依赖**: P2_16 · **后继**: 无

## 目标

Critic 应只在**安全/正确性**维度 reject(阻断),在**策略**维度最多 `advise`(建议但不阻断)。当前 Critic(prompt + backend + fixtures)混合了两类:既 reject 不安全的(对),也 reject "phase 顺序不对/跨域覆盖不足/不是自然的下一步 TODO"(策略,应放权)。

## 安全 vs 策略切分(审计)

| Critic reject 理由 | 性质 | 处理 |
|---|---|---|
| benchmark 不可比 / 测量造假 | 安全 | **保留 reject** |
| accuracy gate 不过 | 安全 | **保留 reject** |
| 无 rollback / 不可回滚的危险 patch | 安全 | **保留 reject** |
| 与 robustness 冲突(已知崩溃模式) | 安全 | **保留 reject** |
| patch 不匹配 worktree / 越权路径 | 安全(不变量层已挡) | 保留 |
| phase sequencing 不对 | 策略 | **降 `advise`** |
| 跨域关键词覆盖不足 | 策略 | **降 `advise`**(P2_16 已删机械层) |
| "不是自然的下一步 TODO" | 策略 | **降 `advise`** |

## 改动清单

### 1. `critic.md`(系统 prompt)
- 10–19, 67–74(verdict classes:`archival` 总 approve、`exploration` vs `promotion`):**保留** `promotion`(高风险 patch)严格安全审查;`exploration`(config/探索)默认更宽松,策略问题给 `advise` 而非 `reject`。
- 32–54(phase-incompatible→reject 指引):与放宽后的 PolicyGate R1 对齐——Critic 不再因 phase 顺序 reject(phase 由 PolicyGate/LLM 管),策略问题降 advise。
- 79–141(cross-domain 规则):保留为 **LLM 判断指引**(语义评审),删除"必须覆盖每个域关键词否则 reject"的机械式表述(P2_16 已删 code 层)。

### 2. `backends/critic_agent.py`
- 283–308 注入 `cross_domain` 约束:作为 **hints** 注入(advisory),不作为上游硬 deny 的依据。
- 72–78 workdir keep 50:INVARIANT,保留。

### 3. `action_registry.py` verdict_class(触类旁通)
- 91–107, 119–141:`verdict_class` 驱动 Critic prompt 策略——**保留** `promotion` 严格;`exploration` 默认放宽(策略问题 advise)。确保 verdict_class 只影响"Critic 用哪套 prompt 指引",不引入隐藏硬 gate。

### 4. 不变量保护
- `integrate_patch` 必须有 Critic verdict(`policy.py` 1825–1889)= **保留**(这是 patch 安全门);本步只改 Critic **判什么**(安全 reject / 策略 advise),不改"patch 必须经 Critic"这件事。
- verdict ∈ {approve, advise} 才放行 integrate(1874–1889):保留;策略问题归 `advise`(放行),安全问题归 `reject`(阻断)。

## 连带测试

| 文件 | 动作 |
|---|---|
| `critic-agent/tests/review_verdict_cases.json`(108–355) | **安全** reject 用例保留;**策略** reject 用例改为 `advise` 期望 |
| `test_role_realignment.py`(critic prompt 契约 83) | 同步 critic.md 措辞 |
| `test_critic_verdict_map.py`(若 P1_02 保留了部分 verdict) | 与 P1_02 决定一致 |
| critic_agent backend 测试 | cross_domain 注入改 hints 的断言 |

## 验证
- Critic 对不安全提案(测量造假/accuracy 不过/危险 patch)仍 `reject`(阻断 integrate)。
- Critic 对策略性疑虑(phase/跨域/下一步)给 `advise`(放行,附建议)。
- `integrate_patch` 仍必须经 Critic;安全门不变。
- 烟测:一个"策略上不理想但安全"的 patch 经 Critic `advise` 后可 integrate;一个"测量不可比"的提案被 `reject`。

## 回退
- 恢复 critic.md / fixtures 的策略 reject 措辞与 backend 注入方式。

## 残留风险
- 中。放宽 Critic 策略 reject 后,更多"非最优但安全"的 patch 会进入 stack——由 benchmark 实测 + KEEP 阈值 + stack rebench(validated 标注 P2_14)裁决其去留。安全维度不受影响。
