# Gap-06 — CLOSE phase 无统一 5 步顺序器, 三处零散触发

> 严重度: **P1 主要** (degraded behavior, 用户提早 Ctrl-C 时退化为 P0)
> 主轴影响: **主轴 A (流程固化)**
> 体检报告: `../KB_design_gaps.MD` §5 Gap-6

## 1. 问题描述

KB_design §3.2 §5.5 明确 CLOSE phase 内 **固定顺序** (不允许 LLM 跳序):

```
1. Coordinator 触发 `report` (生成 markdown / json 报告)
2. Coordinator 触发 `session_breakdown` (写 session_breakdown.json v2)
3. NDJSON flusher drain (等待异步 _enqueue 全部 POST)
4. Cortex `session commit`
5. 退出 reactor loop, 进程结束
```

实际三处零散触发, 无统一 sequencer:

- (1) `_enter_closing_phase` 在 **wall-clock deadline** 自动 enqueue report
  — 与 phase 切换无关
- (2) cli.py `finally` 兜底写 `session_breakdown.json` — 不绑定 phase=CLOSE
- (3) NDJSON drain: `Coordinator.stop()::_cortex_t4_hook` — 进程退出时一次性跑
- (4) `session commit`: 同 (3) 在 stop() 内, 与 report 之间无序保证
- (5) Exit: 自然结束

后果:
- fresh session 跑到 deadline 时, 三个产物大概都有, 但顺序不保证.
- 用户 Ctrl-C 提早终止时: (3) + (4) 在 stop() 内会跑, 但 (1) report
  没人发, (2) session_breakdown 因 cli.finally 救场可能仍写, 但内容缺
  最后一段.

## 2. 现状代码 trace

### 2.1 closing_phase 与 phase 状态机分离

`coordinator.py:1076-1135` `_enter_closing_phase`:

```text
def _enter_closing_phase(self) -> None:
    """Enter closing on wall-clock deadline (not phase=CLOSE)."""
    if self.shared_state.closing_phase:
        return
    self.shared_state.closing_phase = True
    self.shared_state.closing_started_unix = time.time()
    # Auto-enqueue report
    if not self.shared_state.closing_report_task_id:
        task = self._build_internal_report_task()
        ...
        self.shared_state.closing_report_task_id = task.task_id
```

触发条件: `elapsed >= max_minutes * 0.95` (经验值, wall-clock).

### 2.2 session_breakdown 走 cli finally

`cli.py:2465-2474`:

```text
finally:
    try:
        from .breakdown.exporter import write_breakdown_json
        write_breakdown_json(session_dir=session_dir)
    except Exception as e:
        log.exception("session_breakdown write failed: %s", e)
```

不挂在 phase=CLOSE 转入. 进程**任何**退出路径都触发 (包括 crash).

### 2.3 NDJSON drain + commit 在 stop() 内

`coordinator.py:549-598` `_cortex_t4_hook`:

```text
async def _cortex_t4_hook(self) -> None:
    """Drain pending NDJSON + Cortex session commit."""
    if not self.cortex_kb:
        return
    try:
        await self.cortex_kb.flush_pending()  # NDJSON drain
        outcome = await self.cortex_kb.session_commit(...)
        self.shared_state.cortex_session_summary = outcome.summary
    except Exception as e:
        log.warning("cortex T4 hook failed: %s", e)
```

被 `Coordinator.stop()` 调, 不在 phase=CLOSE 转入路径上.

### 2.4 phase_state.py CLOSE allowlist

`phase_state.py:112-114` 允许 `{report, session_breakdown, recover}`. 但
没人在 phase=CLOSE 内**实际**enqueue 这些 action.

## 3. 设计意图

§3.2 §5.5 强调:
- **固定顺序**, 不允许 LLM 跳序
- **每个动作都是原子** (失败时保留 session_dir 供人工检查, 而非自动重试)
- 操作员见到 `session_breakdown.json` 即可信 NDJSON 已 drain, Cortex 已 commit

设计目的:
- 给 Cortex T4 commit 留缓冲 (报告写完再 commit, 万一 commit 失败报告仍可读)
- 防止 Ctrl-C 时数据丢失

## 4. 根本原因

closing_phase 概念在 v0.7 时代由 wall-clock deadline 驱动, 与 v0.8 phase
状态机概念**没整合**. M2 PR 加 phase 字段时, `_enter_closing_phase` 没
被改造, `phase=CLOSE` 转入与 closing_phase 是两条平行路径.

设计意图本是"phase 状态机 = single source of truth", 但 closing_phase
还是另一条 truth.

## 5. 修复路径

### PR 5.1 — phase-entry hook 框架 (复用 Gap-04)

Gap-04 PR 5.1 引入的 `_on_phase_entered` 框架.

### PR 5.2 — `_on_enter_close` 5 步顺序器

```text
async def _on_enter_close(self, from_phase: str) -> None:
    """KB_design §3.2 §5.5 — CLOSE phase 5-step sequencer."""
    state = self.shared_state
    log.info("CLOSE entered (from=%s); starting 5-step close sequence", from_phase)

    # Step 1: Auto-enqueue report (if not already enqueued)
    if not state.closing_report_task_id:
        task = self._build_internal_report_task()
        await self.task_registry.enqueue(task)
        state.closing_report_task_id = task.task_id
        await self._wait_for_task(task.task_id, timeout=600)
    self._record_close_step("report", status="done")

    # Step 2: Auto-enqueue session_breakdown
    bd_task = self._build_internal_session_breakdown_task()
    await self.task_registry.enqueue(bd_task)
    await self._wait_for_task(bd_task.task_id, timeout=300)
    self._record_close_step("session_breakdown", status="done")

    # Step 3: NDJSON drain
    if self.cortex_kb:
        try:
            await asyncio.wait_for(self.cortex_kb.flush_pending(), timeout=300)
            self._record_close_step("ndjson_drain", status="done")
        except asyncio.TimeoutError:
            self._record_close_step("ndjson_drain", status="timeout")

    # Step 4: Cortex session commit
    if self.cortex_kb:
        try:
            outcome = await self.cortex_kb.session_commit(...)
            state.cortex_session_summary = outcome.summary
            self._record_close_step("cortex_commit", status="done")
        except Exception as e:
            self._record_close_step("cortex_commit", status="failed", error=str(e))

    # Step 5: Mark exit
    state.stop_reason = state.stop_reason or "close_sequence_done"
    self._record_close_step("exit", status="ready")
    log.info("CLOSE 5-step sequence complete")
```

### PR 5.3 — 协调 closing_phase 与 phase=CLOSE

`_enter_closing_phase` (wall-clock deadline 路径) 不再 enqueue report.
改为: 把 stop_reason 设为 `time_exhausted` 触发 phase=CLOSE 转入,
然后 `_on_enter_close` 接管 5 步.

```text
def _enter_closing_phase(self) -> None:
    """Wall-clock deadline → request CLOSE phase transition."""
    if self.shared_state.closing_phase:
        return
    self.shared_state.closing_phase = True
    self.shared_state.closing_started_unix = time.time()
    self.shared_state.set_stop_reason("time_exhausted")
    # phase transition will be picked up by _advance_phase_if_needed next tick
```

### PR 5.4 — cli finally 兜底改 *only on abort*

`cli.py finally` 改为:

```text
finally:
    # If CLOSE 5-step sequencer didn't run (crash / Ctrl-C),
    # try a best-effort breakdown dump.
    if not getattr(coordinator.shared_state, 'close_sequence_done', False):
        try:
            from .breakdown.exporter import write_breakdown_json
            write_breakdown_json(session_dir=session_dir)
        except Exception:
            log.exception("emergency breakdown write failed")
```

避免重复写 (CLOSE sequencer 已经写过则跳过).

### PR 5.5 — `_record_close_step` 记录到 phase_history evidence

```text
def _record_close_step(self, step: str, *, status: str, error: str = "") -> None:
    history = self.shared_state.phase_history
    if history and history[-1].get("to") == PHASE_CLOSE:
        evidence = history[-1].setdefault("evidence", {})
        evidence.setdefault("close_steps", []).append({
            "step": step, "status": status, "error": error,
            "ts": _now_iso(),
        })
```

breakdown 中 phase_history 段可看到 5 步状态.

### PR 5.6 — 测试

`tests/test_v08_close_phase_sequencer.py`:

- mock plateau_kernel → SWEEP → sweep_done → CLOSE
- 验证 5 步按顺序记录到 phase_history evidence
- 验证 NDJSON pending 在 step 3 后清空
- 验证 Cortex session_commit 在 step 4 后调用 1 次
- crash 场景: 模拟 step 2 失败, 验证 cli.finally 兜底仍写 breakdown

## 6. 验收口径

- [ ] fresh session 进入 CLOSE 后, breakdown 中 phase_history 末尾段
      evidence.close_steps 含 5 行, 各 status='done'
- [ ] session_breakdown.json 写入时间 *晚于* report.md
- [ ] NDJSON pending = 0 在 Cortex commit 之前
- [ ] Cortex session_commit 仅调一次 (不重复)
- [ ] 用户 Ctrl-C 中断 EXPLORE 时, cli.finally 救场写 breakdown,
      close_steps 中缺失的标 `status='aborted'`

## 7. 风险 / 回退

- **report task 失败**: step 2 仍跑 (session_breakdown 不依赖 report);
  step 4 commit 失败时 NDJSON pending 保留, 下次 resume 自动 drain
  (R-01 路径).
- **Ctrl-C 时序**: cli.finally 救场覆盖 CLOSE sequencer 的部分写入风险.
  解决: 在 sequencer step 2 完成后立刻设 `state.close_sequence_done=True`
  flag, cli.finally 据此 short-circuit.
- **回退**: 删除 `_on_enter_close` 即退到当前 (wall-clock deadline + cli
  finally 双路径). 三个 phase-entry hook 框架 (Gap-04/05/06) 不耦合,
  可独立回退.

## 8. 关联 gap

- **同框架**: Gap-04, Gap-05 — 三 phase-entry hook 共享调度
- **关联**: §3.14 R-01/R-02 (Cortex 不可达) — step 3/4 容错由 R-01 缓
  解路径覆盖
