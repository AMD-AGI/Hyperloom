# Gap-01 — SpecialistRunner 未注册 → `delegate{action='specialist'}` 整链不通

> 严重度: **P0 阻断** (设计目标不可达)
> 主轴影响: **主轴 C (sub-agent 双形态)** — 0% 落地
> 体检报告: `../KB_design_gaps.MD` §4 Gap-1

## 1. 问题描述

KB_design §3.5 把 LLM-based specialist 定义为 v0.8 的核心新增 sub-agent
形态. EXPLORE phase 内 Orchestration 应当通过
`delegate{action='specialist', params={domain, gap, ...}}` 派 specialist;
SpecialistRunner 实例化 LLM backend, 多 turn 跑完, emit
`specialist_done` 关闭 lifecycle.

实际: SpecialistRunner 代码 + 测试都齐 (~700 行 + 40 个 test case 通过),
**但 cli `_register_executors` 不实例化它**, 也不把 `'specialist'` 路
由到任何 executor. 派发 `delegate{action='specialist'}` 进入 dispatcher
后, `SubAgentRunner.run_task` 找不到 executor, 报 `no_executor`, task
立即 fail.

## 2. 现状代码 trace

### 2.1 `cli.py:_register_executors` (~688–730)

```text
for kind, fn in _REAL_EXECUTORS_FULL.items():
    coordinator.sub.register_executor(kind, fn)

coordinator.sub.register_executor("target_analysis", TargetAnalysisExecutor(...))

if no_kernel:
    return

for kind, fn in _REAL_EXECUTORS_KERNEL_ONLY.items():
    coordinator.sub.register_executor(kind, fn)
for kind in _NOOP_KINDS_KERNEL_ONLY:
    coordinator.sub.register_executor(kind, _noop_prep)
```

`_REAL_EXECUTORS_FULL` (~625-647) 里的键: `baseline / backends / params /
explore / sweep / report / session_breakdown / validate_stack / recover`.
**没有 `specialist`**.

### 2.2 SpecialistRunner 自身

`inference_optimizer/orchestrator/specialist_runner.py`: 完整实现, 含
prompt 装配 (9 段) / heartbeat / transcript / tool whitelist / synth empty
done. 测试 `tests/test_v08_m5_specialist.py` 40/40 通过.

### 2.3 Coordinator 全文搜索

```
$ grep -n "SpecialistRunner\|specialist_runner\|specialist_factory" orchestrator/coordinator.py
(no matches)
```

Coordinator 完全不感知 SpecialistRunner 类型.

### 2.4 派发流终点

`SubAgentRunner.run_task` (`sub_agent_runner.py:113-124`) 检测到
`task.kind == 'specialist'` 且 `executor_registry` 中无对应 fn → raise
`ExecutorMissingError` → task 标 `failed`, 写 `last_action_failures`,
延伸为 PolicyGate R2 通过但实际 0 个 specialist 跑起来.

## 3. 设计意图 (引 KB_design)

- §3.5 §5 "specialist 派发到结果的完整链路":
  `[Coordinator dispatcher] → research_lane 容量 1, 申到 lane →
  SpecialistRunner.run_task(task) → ...`
- §3.13 M5 §3 §交付物表: "SpecialistRunner (概念) — 走 LLM backend,
  多 turn, 收尾 specialist_done"
- §3.13 M5 §8 验收清单第 1 条: "EXPLORE 内可见 1 个 specialist 任务,
  transcript 完整, heartbeat 更新, 收尾 specialist_done"

## 4. 根本原因

M5 PR 链拆得很细 (9 个 PR), PR 1-8 都把"代码 + 单元测试" 写好, **PR 9
("CLI flag 默认开 framework_specialist")**应当把 cli 注册接上, 但实际
合入的 PR 9 只加了 `--research-lane-capacity` flag, **漏了** "把
SpecialistRunner 注入 `_register_executors`" 这一步.

成因可归纳为:

1. M5 设计中 SpecialistRunner 概念归 §3.5 / 测试归 M5 §PR4, 但**接线**
   被分散在两处 (实例化 + 注册), 任何一处漏写都不会让现有测试变红 —
   因为单元测试都用 mock backend / mock runner, 不走 cli.
2. cli `_register_executors` 是 cli.py 的局部, M5 PR 没把它列入触及文
   件清单.
3. fresh-session 烟测 (M5 §8) 没自动化, 操作员肉眼验证时容易跳过
   "specialist 任务派出"这一步 (默认看 cumulative_gain 涨没涨).

## 5. 修复路径

### PR 5.1 — KnowledgePlane bootstrap 落地

依赖 **Gap-02** 必须先做. SpecialistRunner 的工具白名单 / pr_feed
是从 KnowledgePlane 取的, 没 plane 就没法 construct runner.

### PR 5.2 — SpecialistRunner 实例化 + 注册

在 cli.py:

```text
def _build_specialist_runner(args, plane, session_dir) -> SpecialistRunner:
    return SpecialistRunner(
        backend_factory=_make_backend_factory(args, role='specialist'),
        session_dir=session_dir,
        knowledge_plane=plane,
        default_tools=DEFAULT_SPECIALIST_TOOLS,   # from specialist_runner module
        default_max_turns=int(args.specialist_max_turns or 8),
        per_turn_max_seconds=float(args.specialist_per_turn_max_seconds or 600),
    )
```

`_register_executors` 末尾新增:

```text
if args.research_lane_capacity > 0:
    spec_runner = _build_specialist_runner(args, plane, session_dir)
    coordinator.sub.register_executor("specialist", spec_runner.run_task)
```

注意:

- `args.research_lane_capacity == 0` 时不注册, fresh session 走老
  M3 (LLM-direct grid).
- backend_factory 复用 `args.claude_model` / `args.specialist_model`
  (后者新加 CLI flag, 默认同 orchestration model).

### PR 5.3 — Coordinator 透传 KnowledgePlane

Coordinator `__init__` 加 `knowledge_plane: KnowledgePlane | None = None`
参数, cli 在构造时透传. SpecialistRunner 在 dispatch 时通过
`task.params['knowledge_plane_ref']` 拿到 plane (或 cli 直接把 plane 注
入 runner 构造, 后者更简洁).

### PR 5.4 — 派发前自动 warm pr_feed + kb_subgraph

`_handle_delegate` 内检测 `action == 'specialist'`:

```text
async def _handle_delegate(self, source, intent):
    ...
    if action_name == "specialist":
        if self.knowledge_plane is not None:
            params = dict(intent.payload.get("params") or {})
            domain = params.get("domain")
            gap_id = params.get("gap")
            if domain:
                params.setdefault("pr_feed",
                    await self.knowledge_plane.pr_feed_warm(domain=domain))
            if gap_id:
                params.setdefault("kb_subgraph",
                    await self.knowledge_plane.cortex_traverse(gap_id))
            intent.payload["params"] = params
    ...
```

### PR 5.5 — fresh-session 烟测 (集成测)

新增 `tests/test_v08_m5_specialist_integration.py`:

- 端到端 cli boot → fake Cortex / PR Monitor → 派 1 specialist → 验证
  `breakdown.specialist_runs` 至少 1 行
- 必须用真的 SubAgentRunner + 真的 SpecialistRunner, **不能 mock**.
  这是该 gap 没被测试发现的根本原因.

## 6. 验收口径

- [ ] fresh session 内 `breakdown.specialist_runs` ≥ 1 行 (KB_design
      M5 §8 第 1 条)
- [ ] 任务 transcript 文件 `runs/specialist/<task_id>/specialist_done.json`
      存在 + 非空
- [ ] `state.json.specialist_rounds` 非空, proposals_total ≥ 1
- [ ] M5 §8 全部 8 条 (specialist 不能调 Edit/Write/git apply, 提议进
      explore round, KB hypothetical edge per variant, kill 后合成 empty,
      PolicyGate R2 拒非 orchestration source, R3 拒伪造 from_agent)

## 7. 风险 / 回退

- **新增 LLM API 调用** → quota 风险, 见 §3.14 R-05. 缓解: M5 默认
  `--research-lane-capacity 1` (串行), 不是 M6 的 6.
- **token 泄漏** → §3.14 R-12. 工具白名单已禁 Edit/cat ~/.codex/*; tool
  调用前 redact 已落 (specialist_runner._safe_redact).
- **回退**: `--research-lane-capacity 0` 即时关 specialist, 退到 M3
  LLM-direct grid.

## 8. 关联 gap

- **必须先做**: Gap-02 (KnowledgePlane bootstrap), Gap-03
  (specialist_done 路由)
- **同时做**: Gap-09 (`gaps[]` 字段, specialist 派发依据)
- **解锁**: Gap-07, Gap-08 (per-variant T2/T3 — specialist 提议每 variant
  独立 edge)
