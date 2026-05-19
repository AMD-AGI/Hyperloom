# Gap-03 — `IntentType.SPECIALIST_DONE` 在 Coordinator 路由表缺失

> 严重度: **P0 阻断** (设计目标不可达)
> 主轴影响: **主轴 C** — specialist 整链最后一公里断开
> 体检报告: `../KB_design_gaps.MD` §4 Gap-3

## 1. 问题描述

KB_design §3.5 §7 + §3.11 R3 定义: specialist 跑完一定通过
`specialist_done` intent 关闭 lifecycle. Coordinator 接收后应当:

1. 解析 `payload.proposal_set` (variant 列表)
2. 调 `SharedState.record_specialist_round(...)` 写 ledger
3. 更新 `specialist_domain_empty_streak` (空提议 streak)
4. 把 task 标 `succeeded` + emit `delegated_result` 事件
5. (M5 §5 step 6) 给每个 variant 触发 T2 hypothesize

实际: `coordinator.py::_handle_intent` (~1971-2009) 的 if/elif 链
**不包含 `IntentType.SPECIALIST_DONE`** 分支. 该 intent 走 `else` 仅记
入 observation, 无任何后续效果. PolicyGate R3 已经放行的 intent 实际
"过了海关但没出机场".

## 2. 现状代码 trace

### 2.1 路由表

`coordinator.py:1979-2008`:

```text
if it == IntentType.PROPOSE_ACTION:
    await self._handle_propose_action(source, intent)
elif it == IntentType.REVIEW_VERDICT:
    await self._handle_review_verdict(source, intent)
elif it == IntentType.DELEGATE:
    await self._handle_delegate(source, intent)
elif it == IntentType.REQUEST:
    await self._handle_request(source, intent)
elif it == IntentType.RESPONSE:
    await self._handle_response(source, intent)
elif it == IntentType.KILL_TASK:
    await self._handle_kill_task(source, intent)
elif it == IntentType.PRUNE_BRANCH:
    await self._handle_prune_branch(source, intent)
elif it == IntentType.FORCE_DISPATCH:
    await self._handle_force_dispatch(source, intent)
elif it == IntentType.ESCALATE_STRATEGY_CHANGE:
    await self._handle_escalate_strategy_change(source, intent)
elif it == IntentType.SEND_MESSAGE:
    await self._handle_send_message(source, intent)
elif it == IntentType.ALERT:
    await self._handle_alert(source, intent)
elif it == IntentType.UPDATE_STATE:
    await self._handle_update_state(source, intent)
else:
    # ASK_QUESTION / ANSWER / UPDATE_PERSONA — record for replay
    await self._record_observation(source, "observation", {...})
```

`IntentType.SPECIALIST_DONE` 落入 `else` 分支.

### 2.2 record_specialist_round 已实现

`shared_state.py:1953-1985`:

```text
def record_specialist_round(self, entry: dict[str, Any]) -> None:
    """Append one round summary to ``specialist_rounds``."""
    ...
    self.specialist_rounds.append(dict(entry))
    ...
```

写好的, 只是没人调.

### 2.3 PolicyGate R3 已实现

`policy.py:_validate_specialist_done` (~745-893): 校验 from_agent
prefix + gap/domain match + payload schema. **通过的 intent 进入路由
表后无人处理**.

### 2.4 任务生命周期

正常 `delegate` task 的生命周期:

```
delegate intent → _handle_delegate → TaskRegistry enqueue
                                     ↓
                         dispatcher → SubAgentRunner.run_task
                                     ↓
                                  task.result → delegated_result event
                                     ↓
                          Orchestration 下 tick 看到 delegated_result
```

Specialist task 的设计生命周期:

```
delegate{action='specialist'} → enqueue task
                              ↓
                   SpecialistRunner.run_task (LLM 多 turn)
                              ↓
                LLM emit specialist_done intent (在 prompt 中要求)
                              ↓
                _handle_specialist_done (尚不存在) → 解析 proposal_set
                              ↓
                    record_specialist_round + mark task succeeded
                              ↓
                  delegated_result(state='succeeded', result={...})
```

中间一环缺失.

## 3. 设计意图

- §3.5 §7 "specialist 退出协议": "specialist 任务唯一退出方式 = 一条
  `specialist_done` intent". 单退出协议 (Inv-5.3).
- §3.5 §10 "Coordinator 收到 specialist_done": 列出 5 个步骤
  (解析 / record / streak / mark succeeded / 触发 T2).
- §3.13 M5 §3 §5 step 6: per-variant T2 hypothesize 也是这里触发.
- §3.11 R3 校验 specialist_done 来源 + gap/domain match + schema, **设计
  明确"R3 通过即应被 Coordinator 处理"**.

## 4. 根本原因

`IntentType.SPECIALIST_DONE` 是 M5 PR1 加入的 (PolicyGate R3 + 解析).
PR1 重心放在"防止伪造 from_agent", 把 happy-path 处理留作 M5 PR4 (新
SpecialistRunner) 内容. PR4 实现了 SpecialistRunner 但没在 Coordinator
加 `_handle_specialist_done`, 推测原因:

1. M5 PR4 描述里写 "SpecialistRunner runs LLM, parses specialist_done,
   returns SubAgentResult". 设计上意图 *runner 内部已经把 proposal_set
   组装进 result*, Coordinator 走 `delegated_result` 标准路径就行.
2. 但 v0.8 的 IntentType.SPECIALIST_DONE 是 LLM emit 的 intent, 走标准
   intent 流水线 (intent_parser → PolicyGate → _handle_intent). 它和
   runner 内部 result 是**两条路径**, 设计文档没说清楚走哪条.
3. 实际 SpecialistRunner 的 LLM 会通过 `emit_intent` 工具发出
   `specialist_done`, 这个 intent 走标准 PolicyGate 路径**先**到达
   `_handle_intent`. 然后才轮到 runner 拿 transcript 转 result. 顺序
   颠倒, 路由表必须先处理.

## 5. 修复路径

### PR 5.1 — 路由表加分支

`coordinator.py:_handle_intent` 加:

```text
elif it == IntentType.SPECIALIST_DONE:
    await self._handle_specialist_done(source, intent)
```

### PR 5.2 — 新 `_handle_specialist_done` 方法

```text
async def _handle_specialist_done(self, source: str, intent: Intent) -> None:
    """KB_design §3.5 §10 — terminal intent of a specialist task."""
    payload = intent.payload
    task_id = source.removeprefix(SPECIALIST_FROM_AGENT_PREFIX)
    task = self.task_registry.get(task_id)
    if task is None:
        # PolicyGate R3 should have rejected, but defense in depth:
        log.warning("specialist_done from unknown task %s", task_id)
        return

    # 1. Record the round summary
    round_entry = self._build_specialist_round_entry(task, payload)
    self.shared_state.record_specialist_round(round_entry)

    # 2. Update domain empty streak
    domain = payload.get("domain", "")
    proposals = payload.get("proposal_set") or []
    streak_dict = self.shared_state.specialist_domain_empty_streak
    if not proposals:
        streak_dict[domain] = int(streak_dict.get(domain, 0)) + 1
    else:
        streak_dict.pop(domain, None)

    # 3. Materialize SubAgentResult for the dispatcher
    result = SubAgentResult(
        state="succeeded",
        result={
            "domain": domain,
            "gap_canonical_id": payload.get("gap_canonical_id"),
            "proposal_set": proposals,
            "confidence_avg": _compute_confidence_avg(proposals),
            "notes": payload.get("notes") or [],
        },
        evidence=...,
    )
    await self._finalize_specialist_task(task, result)

    # 4. (Gap-07 dependency) per-variant T2 hypothesize
    # Defer to Gap-07 implementation; placeholder here.
    if self.cortex_kb and self.knowledge_plane:
        for variant in proposals:
            await self._cortex_t2_hypothesize_variant(task, variant)

    # 5. Persist + observation event
    await self._record_observation(source, "observation", {
        "kind": "specialist_done",
        "task_id": task_id,
        "domain": domain,
        "proposals_total": len(proposals),
    })
    await self._persist_state()
```

### PR 5.3 — `_finalize_specialist_task` helper

把 task 标 `succeeded` + emit `delegated_result` 事件, 复用现有
`_promote_to_shared_state` 但绕过 explore promote 逻辑 (specialist
不是 grid run).

### PR 5.4 — synth empty done 路径对齐

`_handle_kill_task` 在 specialist task 被 kill 时, Coordinator 应当
*代发*一条合成的 empty `specialist_done` (KB_design §3.5 §13 / M5 §6
"合成 empty done"). 当前 SpecialistRunner 内部已有 synth empty 逻辑,
但 path 在 stale 检测 + kill_task 上不统一. PR 5.4 把这条路径也接到
`_handle_specialist_done` 入口, 保证*所有*终止都走相同 handler.

### PR 5.5 — 集成测

`tests/test_v08_m5_specialist_lifecycle.py` (新增):

- mock SpecialistRunner emit 一条假的 specialist_done intent
- 走 Coordinator.tick 路径
- 断言 specialist_rounds 长度 +1
- 断言 specialist_domain_empty_streak 在空 / 非空两种情况下正确更新
- 断言 delegated_result event 进入 Orchestration inbox

## 6. 验收口径

- [ ] specialist_done intent 进入 _handle_intent 后, specialist_rounds
      ledger 增长 1 行
- [ ] 空 proposal_set 累加 streak; 非空清零
- [ ] task 状态 → `succeeded`, delegated_result event 发出
- [ ] Orchestration 下 tick 看到 delegated_result + proposal_set 可用
- [ ] PolicyGate R3 拒绝 (伪造 from_agent / gap_mismatch / domain_mismatch /
      schema fail) 后 handler 不执行 (R3 优先级正确)

## 7. 风险 / 回退

- 实施面较小 (单方法 + 路由 1 行), 主要风险是 _handle_specialist_done
  和 _finalize_specialist_task 之间的事务性: 如果 record_specialist_round
  成功但 emit delegated_result 失败, 会有"ledger 记了但 task 不结束"
  的 inconsistency. 缓解: 用 try/finally, 失败时 task 仍标 failed (而
  非 succeeded).
- **回退**: SpecialistRunner 注册移除 → 整个路径不触发, 等同于
  `--research-lane-capacity 0`.

## 8. 关联 gap

- **必须先做**: Gap-01 (没有 SpecialistRunner 也就没有 intent 发出),
  Gap-02 (没有 plane 派不出来)
- **同时做**: Gap-07 (per-variant T2 在 `_handle_specialist_done` 内触发),
  Gap-09 (`gaps[]` 字段 — `_build_specialist_round_entry` 需要 gap_id
  从这里取)
