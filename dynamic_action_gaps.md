# `dynamic_action` 实施 vs. 设计差距（gap）报告

> 基线：`dynamic_action.MD`（462 行设计稿，v1 lock-down P0 决策已合并）。
> 实施代码：commits `f42ba6a` → `79c5f72`（P0–P9 全部 10 个阶段已完成）。
> 测试基线：266 个 `test_dynamic_action_*` 用例（单元 183 + 集成 43 +
> 不变量 40），全部通过；2699 全仓 pytest 仅 1 个无关 grid_runner flake。
>
> 本文按**严重度**分组列出现状与设计之间的差距，每条均给出
> **机械检测点**与**修复思路**。"严重"指阻塞真实运行；"中等"指防御
> 层缺失但有上游兜底；"低"指语义对齐或可观察性差距；"文档/外观"
> 指代码到位但文档/CLI 没跟上。
>
> 修复优先级建议见 §8。

---

## 0. 总览

| 严重度 | 数量 | 影响 |
|---|---|---|
| Critical | 3 | 阻塞 dynamic_action 在真实 session 端到端跑通 |
| Medium | 4 | 防御层有缝（红线在上游兜底，但未机械锁死） |
| Low | 5 | 语义对齐 / 可观察性 / 一致性 |
| Doc | 3 | 代码到位但文档 / CLI / 测试矩阵未跟上 |

P1–P9 全部产线代码骨架到位；缺口集中在 **(a) 真实 bench 实现**、
**(b) 全 lifecycle 的 dispatch_history / telemetry 落盘**、**(c) IR-4
sourced-variant cap 的显式机械校验**。其余为可观察性与文档跟齐。

---

## 1. Critical — 阻塞真实 session 端到端

### G1. 全部 micro-bench 脚本均为 placeholder

**设计约定**：dynamic_action.MD §1.3 / P3 §4 工具白名单包含
"跑 micro-bench (仅作内部假设验证)"。`BENCH_REGISTRY`
（`dynamic_action_tools.py`）声明了 4 条 bench：

| bench_id | 设计意图 |
|---|---|
| `kernel_attention_timing` | 单层 attention forward 计时 |
| `kernel_gemm_timing` | GEMM 计时 + occupancy |
| `kernel_kvcache_layout` | KV cache layout 读写吞吐 |
| `inference_short_prompt` | 端到端短 prompt 延迟 |

**实施现状**（机械检测点）：

```4:5:inference_optimizer/benches/kernel_attention_timing.sh
# kernel_attention_timing.sh — placeholder body wired through _placeholder.sh until
# the real probe lands. See dynamic_action_tools.BENCH_REGISTRY.
exec env BENCH_ID="kernel_attention_timing" bash "$(dirname "$0")/_placeholder.sh" "$@"
```

四个脚本全部 4 行，全部 `exec` 到 `_placeholder.sh`；`_placeholder.sh`
只写一个 `result.json` 标记，**不做任何实际测量**。

**影响**：
- sub-agent 调 `run_bench` 永远只能得到 `{"status": "placeholder"}`，
  无法形成有意义的内部假设验证。
- §1.2 "micro-bench 仅作内部假设验证" 在物理层成立（因为根本没有
  数据回流），但 sub-agent 的探索能力大幅受限。

**修复思路**：
1. **短期（推荐）**：把 `run_bench` 从工具白名单**临时摘除**，prompt
   builder 同步删除 bench 描述段落。把 `BENCH_REGISTRY` 降级为
   `frozenset()`；invariant test I-5 自动接管（bench 注册表为空 →
   "no server/Magpie" 平凡满足）。
2. **中期**：为每条 bench 实现真实探针；最小可行版本是 wrap 现有
   `target_analysis` / `kernel_attention_layer_timing.py` 等 probe，
   通过 `DYNAMIC_BENCH_OUTPUT_DIR` 写入标准化 JSON。
3. **同步**：bench 注册表加 `min_runtime_sec` / `expected_output_keys`
   字段，runner 在 `run_bench` 返回前对输出 schema 做白名单校验，
   再次确认数字不会通过 stdout/stderr 泄漏到 journal。

---

### G2. `dispatch_history.jsonl` 仅写入资源回收 / stub 路径

**设计约定**：dynamic_action.MD §1.5 显式列出：

> `dispatch_history.jsonl  # 每次 grid run 的结果`

**实施现状**（机械检测点）：

唯一写盘的两个位置：

| 写位置 | event | 触发时机 |
|---|---|---|
| `action_executors/dynamic_action.py:65-68` | `stub_empty` | stub executor 跑（仅 P3 前 fallback） |
| `dynamic_action_resume.py:_append_abandoned_history` | `abandoned_on_resume` | P8 resume sweep |

**完全缺失的事件**：
- runner COMPLETED / TIMED_OUT / FAILED / COMPLETED_EMPTY 终态
- critic verdict（approve / reject / revise）
- integrate_patch KEPT / REVERTED / INTEGRATE_FAILED

**影响**：
- 一条 dynamic_action 从派发到落地中间所有事件**没有持久化时间线**；
  事后审计只能靠 SharedState in-memory 视图（session 内存丢失即丢）+
  artefact 文件的 mtime。
- §1.5 设计意图（dispatch_history 是 telemetry 聚合视图的源数据）落空。

**修复思路**：

1. 新建 `dynamic_action_history.py`，定义闭合事件类型：

   ```python
   class DispatchHistoryEvent(str, Enum):
       DISPATCHED = "dispatched"
       SUB_AGENT_DONE = "sub_agent_done"
       SUB_AGENT_TERMINATED = "sub_agent_terminated"   # TIMED_OUT/FAILED
       CRITIC_VERDICT = "critic_verdict"
       INTEGRATE_RESULT = "integrate_result"
       ABANDONED_ON_RESUME = "abandoned_on_resume"      # P8 已实现
   ```
   每个事件配一个闭合 `<EVENT>_FIELDS: frozenset[str]`（仿
   `ABANDONED_HISTORY_FIELDS`）。

2. 提供 `append_dispatch_history_row(session_dir, dyn_id, event, payload)`
   统一写入入口；schema 校验在写盘前完成。

3. 在三个 Coordinator 钩子里调用：
   - `_handle_dynamic_action_runner_result` — 写 `sub_agent_done` /
     `sub_agent_terminated`
   - `_mirror_critic_verdict_to_dynamic_action` — 写 `critic_verdict`
   - `_maybe_update_dynamic_action_after_integrate` — 写 `integrate_result`

4. 新增 invariant test `inv_dispatch_history_complete_lifecycle`：
   构造一条 KEPT 的 dyn_id，断言 dispatch_history.jsonl 至少包含
   `dispatched` / `sub_agent_done` / `critic_verdict` / `integrate_result`
   四个事件。

---

### G3. `telemetry.json` 完全缺失

**设计约定**：dynamic_action.MD §1.5：

> `telemetry.json  # 累计成功 / revert / 增益（聚合视图源）`

**实施现状**（机械检测点）：

```bash
$ rg -l 'telemetry\.json' inference_optimizer/orchestrator/
# (no orchestrator-side writer)
```

仅在 `session_paths.py` 的 docstring 中提到，**没有路径辅助函数**、
**没有任何写盘代码**、**没有任何读取代码**。

**影响**：
- 跨 session / 跨 round 的命运统计没有持久化产物。SharedState.
  `dynamic_actions` 提供的是 session-scoped in-memory 视图。
- 后续 retrospective（"过去 100 次 dynamic_action KEEP 率"）无法做。

**修复思路**：

1. 选项 A — **接受现状，更新设计文档**：v1 决策只通过 SharedState
   提供聚合视图，§1.5 的 `telemetry.json` 行从设计稿中移除并在
   gap 报告留 ADR（已在 §1.8 "跨 session 学习暂不处理" 暗含）。

2. 选项 B — **实现 telemetry.json**：
   - 新增 `dynamic_action_telemetry_path(session_dir, dyn_id)` 路径助手。
   - 在 `_maybe_update_dynamic_action_after_integrate` 末尾写入：

     ```python
     {
       "dyn_id": dyn_id,
       "rolled_up_at": iso_ts,
       "kept": 1 | 0,
       "reverted": 1 | 0,
       "integrate_failed": 1 | 0,
       "gain_pct": delta_pct,
       "round_index": round_index,
     }
     ```
     一 dyn_id 一文件（dyn_id 是一次性的），后续聚合脚本扫盘即可。
   - 闭合 schema + invariant test 同 G2。

推荐 **选项 B**（5% 成本，明确兑现设计承诺）。

---

## 2. Medium — 防御层缺失（红线在上游兜底但未机械锁死）

### G4. `MAX_DYNAMIC_SOURCED_VARIANTS = 1` 仅声明未强制

**设计约定**：dynamic_action.MD §1.4 + P1 §4.4：
> `dynamic` 计入 sourced variant 上限；每 round 最多 1 条
> dynamic-sourced variant。

**实施现状**（机械检测点）：

```258:259:inference_optimizer/orchestrator/policy.py
MAX_DYNAMIC_PER_ROUND: int = 1
MAX_DYNAMIC_SOURCED_VARIANTS: int = 1
```

```bash
$ rg 'MAX_DYNAMIC_SOURCED_VARIANTS' inference_optimizer/
# 只在 policy.py 声明 + tests/test_dynamic_action_invariants.py 断言 == 1
# 没有任何 enforcement 代码引用它
```

`_validate_explore_grid_size`（policy.py 1783）只校验
`provenance="specialist:*"` 的 grid 上限，**没有**为
`provenance="dynamic"` 单独计数。

**实际不会触发**的原因（兜底链）：
- `MAX_DYNAMIC_PER_ROUND = 1` 保证一 round 最多 1 个 dynamic_action dispatch。
- `MAX_PROPOSAL_SET_LEN = 1` 保证一 dispatch 最多 1 个 proposal。
- 故 max 1 dynamic-sourced variant per round 在算术上成立。

**风险**：未来若某条约束松动（例如 v2 允许 dispatch 2 个），IR-4 校验
仍会过，红线被悄悄突破。

**修复思路**：

1. 在 `_validate_explore_provenance_block` 旁边加
   `_validate_explore_dynamic_sourced_cap`：

   ```python
   def _validate_explore_dynamic_sourced_cap(
       self, payload: dict[str, Any],
   ) -> None:
       grid = (payload.get("params") or {}).get("grid") or []
       dyn_count = sum(
           1 for v in grid
           if isinstance(v, dict)
           and str(v.get("provenance") or "").strip() == "dynamic"
       )
       if dyn_count > MAX_DYNAMIC_SOURCED_VARIANTS:
           raise PolicyDenied(
               f"explore: {dyn_count} dynamic-sourced variants > "
               f"cap {MAX_DYNAMIC_SOURCED_VARIANTS}",
               rule="dynamic_sourced_variant_cap_exceeded",
           )
   ```
   挂在 `_validate_delegate` 的 explore 分支。

2. 在 `test_dynamic_action_invariants.py::TestInvariant_RoundCap`
   下加 `test_inv_dynamic_sourced_cap_enforced_at_explore_dispatch`
   构造一条 explore grid 含 2 个 `provenance="dynamic"` variant，
   断言 PolicyDenied with rule=`dynamic_sourced_variant_cap_exceeded`。

---

### G5. CLI 没有暴露 dynamic_action 运行时旋钮

**设计约定**：P3 §11 "budget 由 runner 配置硬编码上限"，未明确要求
operator 旋钮，但与 specialist 等姊妹通道对齐应有 CLI 覆盖。

**实施现状**（机械检测点）：

```1004:1015:inference_optimizer/cli.py
    model = (
        getattr(args, "dynamic_action_model", None)
        or getattr(args, "claude_model", "")
    ).strip()
    turn_cap = int(
        getattr(args, "dynamic_action_turn_cap", DEFAULT_TURN_CAP) or DEFAULT_TURN_CAP,
    )
    wall_clock = float(
        getattr(
            args, "dynamic_action_wall_clock_sec",
            DEFAULT_WALL_CLOCK_BUDGET_SEC,
        ) or DEFAULT_WALL_CLOCK_BUDGET_SEC,
    )
```

`getattr(args, "dynamic_action_*", default)` 总是返回 default —— argparser
**从未定义** `--dynamic-action-model` / `--dynamic-action-turn-cap` /
`--dynamic-action-wall-clock-sec`。operator 想跑短一点都没法。

**修复思路**：

在 argparser 注册（`cli.py` 的 `_build_parser` 区域）：

```python
ap.add_argument("--dynamic-action-model", default=None,
                help="Claude model id for the dynamic_action runner; "
                     "defaults to --claude-model.")
ap.add_argument("--dynamic-action-turn-cap", type=int, default=None,
                help=f"Max ReAct turns; default {DEFAULT_TURN_CAP}.")
ap.add_argument("--dynamic-action-wall-clock-sec", type=float, default=None,
                help=f"Wall-clock cap (sec); default "
                     f"{DEFAULT_WALL_CLOCK_BUDGET_SEC}.")
```

加一条 cli smoke test 确认参数解析。

---

### G6. `apply_patch_in_worktree` 落地真实 `git apply`（而非仅 `--check`）

**设计约定**：P3 §3 sub-agent 的探索过程在 worktree 内迭代；最终
proposal 的 `patch_text` 走 `integrate_patch`。

**实施现状**（机械检测点）：

```376:393:inference_optimizer/orchestrator/dynamic_action_tools.py
    # The --check pass succeeded; do the real apply so the sub-agent can
    # iterate. The runner will git reset --hard on termination.
    try:
        proc2 = subprocess.run(
            ["git", "apply", "-"],
            cwd=str(worktree),
            input=patch_text,
            text=True,
            capture_output=True,
            timeout=20.0,
            check=False,
        )
```

实际**真的应用** patch 到 worktree。`reset_worktree` 在 runner 终态前
回滚（`git reset --hard` + `git clean -fd`）。

**两个潜在问题**：

1. **`emit_proposal.patch_text` 语义模糊**：sub-agent 是否需要把
   "本次应用 + 之前增量" 合并成一个完整 diff？runner / proposal
   validator / pipeline 没有任何 "patch_text 必须是相对原始 HEAD
   的 cumulative diff" 的校验。一条 specialist 整合时会直接 `git apply`
   单一 patch_text。
2. **worktree 残留**：`reset_worktree` 先跑，再 `_teardown_worktree`
   销毁；若 reset 失败，残留在物理目录里。`_teardown_worktree` 应该
   不依赖 reset 状态强删（已基本如此，但缺断言）。

**修复思路**：

1. 在 `emit_proposal` 校验路径加 "若 worktree 内有未 commit 改动，
   要求 sub-agent 的 patch_text == `git diff HEAD`"。具体：
   - runner 在 emit_proposal 处理时先 `git diff HEAD --binary` 拿
     reference diff；
   - normalise 后与 proposal 的 patch_text 比较（行级而非字节级，
     容忍 unified diff context 行差异）；
   - 不匹配则 reject with reason `patch_text_not_cumulative_diff`。
2. `reset_worktree` 失败时升级为 log.warning + 触发
   `_teardown_worktree` 的强删路径（已是 `--force`，足够）。

---

### G7. `_load_dispatch_inputs` 在 JSON 损坏时静默失败

**实施现状**：

```429:436:inference_optimizer/orchestrator/dynamic_action_runner.py
        try:
            spec = json.loads(spec_text)
        except json.JSONDecodeError:
            return None, None
        try:
            seed = json.loads(seed_text)
        except json.JSONDecodeError:
            return None, None
```

OSError 路径有 `log.warning`；JSON parse 失败完全静默。

**修复思路**：

```python
try:
    spec = json.loads(spec_text)
except json.JSONDecodeError as exc:
    log.warning(
        "dynamic_action runner: spec.json parse failed at %s: %r",
        spec_path, exc,
    )
    return None, None
# same for seed_kit
```

零成本，纯可观察性提升。

---

## 3. Low — 语义对齐 / 一致性

### G8. `proposal_set` 长度防御缺失

**设计约定**：`MAX_PROPOSAL_SET_LEN = 1`（P3 §5.1 Q3）。

**实施现状**：runner 在 `emit_proposal` 后直接 break，所以最多 1。但
pipeline 侧 `read_runner_proposal_set` → `proposal = proposal_set[0]`
直接索引，没有断言 `len <= MAX_PROPOSAL_SET_LEN`。

**修复思路**：在 `_handle_dynamic_action_runner_result` 读完
`proposal_set` 后加：

```python
from .dynamic_action_proposal import MAX_PROPOSAL_SET_LEN
if len(proposal_set) > MAX_PROPOSAL_SET_LEN:
    log.warning(
        "dynamic_action: proposal_set len=%d > cap=%d for dyn_id=%s; "
        "truncating to first entry",
        len(proposal_set), MAX_PROPOSAL_SET_LEN, dyn_id,
    )
    proposal_set = proposal_set[:MAX_PROPOSAL_SET_LEN]
```

---

### G9. `cumulative_gain` 字段命名误导

**设计约定**：§1.5 / P6 §8 — 字段是 `cumulative_gain`。

**实施现状**：每 dyn_id 是一次性的（v1 没有"续跑"），实际只写一次
（integrate complete hook），值就是 integrate result 的 `delta_pct`。
"cumulative" 暗示跨多次累积，但语义上是单次。

**修复思路**：

- 选项 A：仅文档化 v1 语义（保留字段名，避免破坏 prompt 渲染）。
- 选项 B：v2 改名为 `gain_pct`；P6 §8 prompt projection 字段集同步
  更新；prompt renderer 兼容两种字段名一个版本期。

推荐 **选项 A** + 在 `SUMMARY_PROMPT_FIELDS` 旁边加 docstring
注明 v1 single-shot 语义。

---

### G10. degraded seed kit 无告警

**实施现状**：`SeedKitResult.degraded`（如缺 roofline / 缺 profile）
仅在 spec.json 落盘 `degraded_dispatch=True`，**无 log**。

**修复思路**：`_prepare_dynamic_action_dispatch` 在算完
`seed_kit_result` 后：

```python
if seed_kit_result.degraded:
    log.info(
        "dynamic_action: seed kit DEGRADED for dyn_id=%s "
        "(missing one or more of roofline/profile/kept_patches/pitfalls)",
        dyn_id,
    )
```

---

### G11. `revise` verdict 与 `reject` 在 v1 等价但未声明

**实施现状**：`compose_critic_verdict_envelope` 把 `revise` 映射到
`DynamicActionStatus.CRITIC_REJECTED`（同 reject）。
`critic.md` 写"v1 中 revise 等价于 reject"，但 docstring 不显眼。

**修复思路**：在 `compose_critic_verdict_envelope` 顶部 docstring 加：

```
v1 — REVISE collapses to CRITIC_REJECTED at the lifecycle layer; the
verdict label is preserved on critic_verdict.json envelope for audit
so a future v2 (sub-agent re-dispatch loop) can re-open the dispatch
without rewriting the envelope schema.
```

---

### G12. `spec.json["resource_lane"]` 硬编码字面量

**实施现状**：

```5813:5813:inference_optimizer/orchestrator/coordinator.py
            "resource_lane": "research_lane",
```

`ResourceLockManager` 若改 lane 命名，spec.json 与现实漂移。

**修复思路**：从 `resource_lock_manager` 或 policy 模块导入 lane name
常量；`spec["resource_lane"] = RESEARCH_LANE_NAME`。

---

### G13. `dispatch_history.jsonl` 仅 abandoned 行有闭合 schema

**实施现状**：`ABANDONED_HISTORY_FIELDS` 已 frozen；其他事件（待 G2 实现）
没有对应 frozen 字段集。

**修复思路**：作为 G2 的子任务一并落地：每个 `DispatchHistoryEvent`
配 `<EVENT>_FIELDS: frozenset[str]`，统一在 `append_dispatch_history_row`
校验。invariant test 检查每种 event row 的 key set 与对应 frozenset
精确相等。

---

## 4. Doc — 代码到位但文档 / CLI / 测试矩阵未跟上

### G14. 缺 bench 实现 → bench 集成测试缺失

**现状**：bench 工具的 tests/test_dynamic_action_tools.py 用 monkeypatch
注入假脚本。一旦真实 bench 落地（G1），需要 e2e fixture 测一次真实
subprocess 启动 + timeout + 输出回收的完整链路。

**修复思路**：与 G1 一并落地。

---

### G15. `dynamic_action.MD §2` 仍标"待补"

**现状**：

```196:209:dynamic_action.MD
## 2. 详细设计（待补）

下列子章节将在后续迭代中展开。**章节列表已按 §1.9 的 P0 决策更新**：

- 2.1 `delegate(action_name="dynamic_action")` 的 payload schema
- 2.2 PolicyGate 早期校验规则清单（含 IR-4 provenance 白名单扩展）
...
```

§2.1–2.10 在 `action_dynamic_plan/P1_*.md` 到 `P9_*.md` 中已全部落地，
但 `dynamic_action.MD` 本体未回填。

**修复思路**：要么把 §2 各小节回写（每段 5–10 行，引用对应 `PN.md`），
要么删除 §2 段落改为"详细设计见 `action_dynamic_plan/P1..P9.md`"。
推荐后者，避免双写。

---

### G16. orchestration prompt 与设计 §1.7 文案细微差异

**设计约定**（§1.7）：

> "如果你认为存在一组跨多个 domain 的 patch 组合，且任何单个
> specialist 在其 domain prompt 边界内都不可能提出来，可派发一条
> dynamic action。一般情况下应当依赖 specialist 体系；dynamic action
> 是补充通道，不是默认通道。"

**实施现状**（`prompt_builder._section_dynamic_action`）：英文版，
语义对齐，但用词与设计稿原文不完全一致；中文用户读 prompt 时可能
对照不上。

**修复思路**：保持英文（与全 prompt 一致），但在 docstring 引用
设计稿 §1.7 完整中文原句作为 source of truth。零代码改动。

---

## 5. 已确认 *不是* gap 的事项

避免后续重复审计，下面是看上去像 gap、实际已落地或被设计明确放弃
的项：

| 看似 | 实际 |
|---|---|
| 没有 `IntentType.PROPOSE_DYNAMIC_ACTION` | D-A 决策放弃，复用 DELEGATE |
| 没有专门 `cross_domain_proposal` review_kind | D-B 决策放弃，复用 `patch_landing` + cross_domain flag |
| 没有 trust tier / cooldown / kill switch | §1.8 显式放弃 |
| 没有跨 session 学习 / KB 回写 | §1.8 显式放弃 |
| 没有 prompt 反诱导段 | §1.8 显式放弃（机械约束足够） |
| Coordinator 不"续跑" 重启时未完成的 dispatch | §3.9 / P8 明确"统一标 abandoned" |
| `cross_domain_rules` 校验 LLM 侧未走 hard gate | §3.5 / P4 明确"review_constraints 是 prompt 检查项，不是硬 gate" |
| classifier 案例敏感（拒 `DYNAMIC`） | P9 §5 invariant I-3 修复，与 §1.2 字面量约定对齐 |

---

## 6. 测试矩阵覆盖映射

P9 §5 invariant test 已覆盖 §1.2 全部 8 条红线（I-1 → I-8）；
本 gap 报告涉及的 **机械约束未来回归** 也可挂回 invariant 层。建议
新增 invariant 与 gap 一对一：

| Gap | 配套 invariant test（修复时同步加） |
|---|---|
| G2 | `inv_dispatch_history_complete_lifecycle` |
| G3 | `inv_telemetry_present_on_terminal_state` |
| G4 | `inv_dynamic_sourced_cap_enforced_at_explore_dispatch` |
| G6 | `inv_proposal_patch_text_is_cumulative_diff` |
| G8 | `inv_proposal_set_truncated_to_cap` |
| G13 | `inv_dispatch_history_row_schema_closed`（每事件） |

---

## 7. 修复优先级建议

| 优先级 | gap | 理由 |
|---|---|---|
| **P0 — 必须**| G1, G2 | 真实 session 端到端跑通的前提；G1 决定 sub-agent 探索能力是否真实可用，G2 决定可观测性 / 审计是否可用 |
| **P1 — 强烈推荐** | G3, G4, G5 | G3 兑现设计承诺；G4 锁红线；G5 让 operator 能控制成本 |
| **P2 — 推荐** | G6, G10, G12 | 一致性与可观察性，零行为变更 |
| **P3 — 可选** | G7, G8, G9, G11, G13 | 纯文档 / docstring / 日志 |
| **P4 — 文档** | G14, G15, G16 | 与代码无关，纯文档跟齐 |

执行顺序：G1（含 G14）→ G2（含 G13）→ G3 → G4 → G5 → 其余按 P2/P3
顺序合并到下一个清理批次。每个 P0/P1 gap 落地都应配套 invariant
test（§6 表）。

---

## 8. 一句话总结

骨架与红线机制全部到位（P0–P9 10 个阶段、266 个测试、§1.2 8 条红线
机械化），但**真实可用性**还差三个关键件：**真实 bench 实现** /
**全 lifecycle dispatch_history 落盘** / **telemetry 兑现**。补齐
这 3 项后，dynamic_action 在真实 EXPLORE session 中可端到端跑通；
其余 13 项是中长期的可观察性 + 一致性清理。
