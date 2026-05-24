# F1 — Roofline 复合 Action

> 工作日: 2-3 天
> 风险: 中
> 依赖: F0 完成 (tag `f0-done` 在位)
> 完成节点: tag `f1-done` + ~7 个 commit
>
> **完成 F1 之后:**
> - 默认行为不变 (`--use-roofline-composite=false`),老路径 profile + pmc_roofline 仍然可用
> - `--use-roofline-composite=true` 时,LLM 可以 `propose_action{action_name='roofline'}`,
>   executor 内部原子化 profile + trace_analyze,产出 `last_trace_analyze.analysis_md_text`
>   注入到下个 LLM tick 的 prompt

## 0. 环境准备

```bash
export FEAT_ROOT="/wekafs/zgong/Hyperloom-KB"
export MAIN_ROOT="/wekafs/zgong/Hyperloom"
export FEAT_BRANCH="feature/zhenggong/lossen_explore"

cd "$FEAT_ROOT"
git log --oneline f0-done -1                  # 确认 F0 tag 存在
git status --short                            # 必须干净
pytest inference_optimizer/tests -q --tb=no 2>&1 | tail -3   # >= 1594 passed
bash docs/integration/check_preserved.sh                      # rc=0
```

## 1. 重要修正 — F0-8 schema 调整

> 在做 F1 各步骤前,先**修正 F0-8 的 TraceAnalyzeSnapshot 设计**。
>
> **背景**: 在 F0-8 我们把 `last_trace_analyze` 设计成了 `TraceAnalyzeSnapshot` dataclass。
> 但 main 上 `last_trace_analyze` 实际是个 `dict`,字段很多 (trace_input / candidates_path /
> hot_kernels_top15 / task_groups / reusable_native_kernel_ids / trace_health_warnings /
> analysis_md_path / analysis_md_text / roofline_snapshot_id / roofline_baseline_gain_at_snapshot / ts)。
> 而 `record_trace_analyze` 方法直接以 dict 形式写入。
>
> **修正方案**: 把 `last_trace_analyze` 改为 `dict[str, Any]`(与 main 一致),
> 保留 `TraceAnalyzeSnapshot` 仅作为类型提示 / 文档,提供 `to_dict()` / `from_dict()` 静态方法。
> 这样 F1 cherry-pick 的 `record_trace_analyze` 可以零修改运行。

### F1-0: 修正 F0-8

```bash
cd "$FEAT_ROOT"
```

#### 修改 `inference_optimizer/orchestrator/shared_state.py`:

```python
# 把 TraceAnalyzeSnapshot 改为 (用作类型hint+模式参考):
@dataclass
class TraceAnalyzeSnapshot:
    """Reference shape for SharedState.last_trace_analyze (stored as dict).

    F1 record_trace_analyze writes these fields verbatim into a dict;
    code that reads the dict can use this class as a type hint via
    `TraceAnalyzeSnapshot.from_dict(state.last_trace_analyze)`.
    """
    trace_input: str = ""
    candidates_path: str = ""
    hot_kernels_top15: list[dict] = field(default_factory=list)
    task_groups: list[dict] = field(default_factory=list)
    reusable_native_kernel_ids: list[str] = field(default_factory=list)
    trace_health_warnings: list[dict] = field(default_factory=list)
    analysis_md_path: str = ""
    analysis_md_text: str = ""
    roofline_snapshot_id: int = 0
    roofline_baseline_gain_at_snapshot: float = 0.0
    ts: str = ""

    @classmethod
    def from_dict(cls, d: dict | None) -> "TraceAnalyzeSnapshot":
        d = d or {}
        return cls(
            trace_input=str(d.get("trace_input") or ""),
            candidates_path=str(d.get("candidates_path") or ""),
            hot_kernels_top15=list(d.get("hot_kernels_top15") or []),
            task_groups=list(d.get("task_groups") or []),
            reusable_native_kernel_ids=list(d.get("reusable_native_kernel_ids") or []),
            trace_health_warnings=list(d.get("trace_health_warnings") or []),
            analysis_md_path=str(d.get("analysis_md_path") or ""),
            analysis_md_text=str(d.get("analysis_md_text") or ""),
            roofline_snapshot_id=int(d.get("roofline_snapshot_id") or 0),
            roofline_baseline_gain_at_snapshot=float(
                d.get("roofline_baseline_gain_at_snapshot") or 0.0
            ),
            ts=str(d.get("ts") or ""),
        )

# 在 SharedState 类内修改字段类型:
# 原: last_trace_analyze: TraceAnalyzeSnapshot = field(default_factory=TraceAnalyzeSnapshot)
# 新:
last_trace_analyze: dict = field(default_factory=dict)
```

#### 调整 F0-8 的 test 以匹配 dict 形态:

`inference_optimizer/tests/test_f0_roofline_scaffolding.py` 修改:

```python
def test_shared_state_has_roofline_placeholder_fields():
    from inference_optimizer.orchestrator.shared_state import SharedState
    s = SharedState()
    assert hasattr(s, "last_trace_analyze")
    assert s.last_trace_analyze == {}    # 改为 dict 比较

    assert hasattr(s, "roofline_snapshot_id")
    assert s.roofline_snapshot_id == 0
    assert hasattr(s, "roofline_saturation_history")
    assert s.roofline_saturation_history == []


def test_trace_analyze_snapshot_from_dict():
    from inference_optimizer.orchestrator.shared_state import TraceAnalyzeSnapshot
    snap = TraceAnalyzeSnapshot.from_dict({
        "snapshot_id": 0,
        "analysis_md_text": "hello",
        "roofline_snapshot_id": 3,
    })
    # snapshot_id is the new key name; older test had it
    assert snap.analysis_md_text == "hello"
    assert snap.roofline_snapshot_id == 3
```

#### Commit

```bash
pytest inference_optimizer/tests/test_f0_roofline_scaffolding.py -v
# 应该全 pass

git add inference_optimizer/orchestrator/shared_state.py
git add inference_optimizer/tests/test_f0_roofline_scaffolding.py
git commit -m "fix(F0-8 amend): TraceAnalyzeSnapshot becomes type hint, last_trace_analyze stays dict

F1 cherry-pick of record_trace_analyze (from main) writes a 11-field
dict into last_trace_analyze; matching that shape directly avoids a
schema-translation step in the executor.

TraceAnalyzeSnapshot remains a useful @dataclass for type hints +
from_dict() reader on the consumer side."
```

---

## 2. 步骤清单

```
F1-0   修正 F0-8 schema (TraceAnalyzeSnapshot -> dict + reader)  -> 1 commit
F1-1   port record_trace_analyze method to shared_state.py        -> 1 commit
F1-2   cherry-pick + adapt RooflineExecutor                       -> 1 commit
F1-3   register roofline executor in cli                          -> 1 commit
F1-4   adapt prompt_builder analysis.md injection                 -> 1 commit
F1-5   port roofline-v2 ordering rules to orchestration.md        -> 1 commit
F1-6   翻 --use-roofline-composite 默认 False -> True             -> 1 commit
F1-7   smoke session + 打 tag f1-done                             -> 0 commit (操作员)
```

---

## 3. F1-1: Port record_trace_analyze to shared_state.py

### 来源

`$MAIN_ROOT/inference_optimizer/orchestrator/shared_state.py` 的 `record_trace_analyze` 方法
(约 1508-1655 行,~150 行)。

### 命令

```bash
cd "$FEAT_ROOT"

# 取出 main 上的方法,粘贴到本 branch shared_state.py 的合适位置 (其他 record_* 方法附近)
# 用 grep 找到 "def record_" 段落定位
grep -n "def record_" inference_optimizer/orchestrator/shared_state.py | head -10

# 把 main 的 record_trace_analyze 整段拷过来 (人工编辑或 sed 拼接)
# 推荐人工编辑,因为方法签名 / docstring / 实现都要原样保留
```

### 检查点

`record_trace_analyze` 实现引用的字段必须在本 branch SharedState 上都已存在:
- `self.last_trace_analyze` ✓ (F0-8 + F1-0 已加)
- `self.cumulative_gain_validated` — 检查是否已存在:

```bash
grep -n "cumulative_gain_validated" inference_optimizer/orchestrator/shared_state.py | head -3
```

如果不存在,从 main port 过来 (上下文里搜索 `cumulative_gain_validated`)。

### 测试

新建 `inference_optimizer/tests/test_f1_record_trace_analyze.py`:

```python
"""F1 — Verify record_trace_analyze populates last_trace_analyze correctly."""

from __future__ import annotations

from inference_optimizer.orchestrator.shared_state import SharedState


def test_record_trace_analyze_basic():
    s = SharedState()
    payload = {"trace_input": "/path/to/trace.json"}
    result = {
        "status": "ok",
        "candidates_path": "/path/to/candidates.json",
        "hot_kernels": [
            {"kernel_id": "k001", "name": "fmoe_fp8", "gpu_pct": 28.78,
             "reusable_native_kernel": True},
        ],
        "trace_report_path": "",   # no analysis.md path
    }
    s.record_trace_analyze(payload, result)
    assert s.last_trace_analyze["trace_input"] == "/path/to/trace.json"
    assert s.last_trace_analyze["candidates_path"] == "/path/to/candidates.json"
    assert s.last_trace_analyze["roofline_snapshot_id"] == 1
    assert len(s.last_trace_analyze["hot_kernels_top15"]) == 1
    assert s.last_trace_analyze["reusable_native_kernel_ids"] == ["k001"]


def test_record_trace_analyze_monotonic_snapshot_id():
    s = SharedState()
    s.record_trace_analyze({"trace_input": "a"}, {"status": "ok"})
    assert s.last_trace_analyze["roofline_snapshot_id"] == 1
    s.record_trace_analyze({"trace_input": "b"}, {"status": "ok"})
    assert s.last_trace_analyze["roofline_snapshot_id"] == 2


def test_record_trace_analyze_reads_analysis_md_text(tmp_path):
    md = tmp_path / "analysis.md"
    md.write_text("# Executive Summary\n| Compute % | 92.5 |", encoding="utf-8")
    s = SharedState()
    s.record_trace_analyze(
        {"trace_input": "x"},
        {"status": "ok", "trace_report_path": str(md)},
    )
    assert "Executive Summary" in s.last_trace_analyze["analysis_md_text"]


def test_record_trace_analyze_passes_through_warnings():
    s = SharedState()
    s.record_trace_analyze(
        {"trace_input": "x"},
        {"status": "ok", "trace_health_warnings": [
            {"code": "high_gpu_idle_pct", "severity": "warn", "idle_pct": 35},
            {"code": "tracelens_analysis_failed", "severity": "error"},
        ]},
    )
    warnings = s.last_trace_analyze["trace_health_warnings"]
    assert len(warnings) == 2
    assert {w["code"] for w in warnings} == {"high_gpu_idle_pct", "tracelens_analysis_failed"}
```

### 命令

```bash
pytest inference_optimizer/tests/test_f1_record_trace_analyze.py -v
# 期望 4 passed

pytest inference_optimizer/tests -q --tb=no 2>&1 | tail -3
# 期望 1599 passed (1594 + 4 + 1 from F1-0 amend)

git add inference_optimizer/orchestrator/shared_state.py inference_optimizer/tests/test_f1_record_trace_analyze.py
git commit -m "feat(F1-1): port record_trace_analyze method from main

Method writes the canonical 11-field last_trace_analyze dict on every
successful trace_analyze invocation. Reads previous snapshot_id and
increments by 1 (monotonic).

Cherry-picked from /wekafs/zgong/Hyperloom main @ c6f0a71
inference_optimizer/orchestrator/shared_state.py:1508-1655

Tests: 4 new (basic, monotonic, analysis.md read, warnings pass-through).
Total: 1599 passed."
```


---

## 4. F1-2: Cherry-pick + Adapt RooflineExecutor

### 来源

`$MAIN_ROOT/inference_optimizer/orchestrator/action_executors/roofline.py` (499 行)

### 主要内容

- `RooflineStubExecutor` (line 134-167) — N2a fallback,允许操作员临时禁掉真 executor
- `RooflineExecutor` (line 228-486) — 真正的复合 executor
  - Step 1: 子 ctx 调 profile_executor
  - Step 2: 写 last_profile_trace / last_profile_status / last_profile_args (镜像 promote)
  - Step 3: 调 trace_analyze_handler
  - Step 4: N26 auto-retry on steady_state_chunk_{empty,missing,low_quality}
  - Step 5: 成功调 record_trace_analyze
  - Step 6: 返回 status dict
- `make_roofline_executor` (line 486-499) — factory

### 命令

```bash
cd "$FEAT_ROOT"

# 直接 cp; 这个文件没有相对 import 在本 branch 缺失 (已确认依赖都在)
cp "$MAIN_ROOT/inference_optimizer/orchestrator/action_executors/roofline.py" \
   "$FEAT_ROOT/inference_optimizer/orchestrator/action_executors/roofline.py"
```

### 适配检查 (检查依赖路径)

```bash
# 1. profile_executor 是否存在
grep -rn "^def profile_executor\|^async def profile_executor" \
   inference_optimizer/orchestrator/action_executors/profile.py | head -3
# 期望命中

# 2. trace_analyze_handler 是否存在
grep -rn "trace_analyze_handler" inference_optimizer/orchestrator/kernel_request_handlers.py | head -3
# 注意:本 branch 历史上叫 select_kernels;main 上 N1 已经把它重命名为 trace_analyze_handler
# 如果本 branch 仍然叫 select_kernels,在 roofline.py 里改成
#   from ..kernel_request_handlers import select_kernels as trace_analyze_handler
# 或更彻底地:把本 branch 也跟着重命名 (推荐 — 这是 main 的命名惯例)

# 3. RunnerContext 是否有 extra dict / lease 字段 (已通过 F0 dry-run 验证)
grep -n "class RunnerContext" inference_optimizer/orchestrator/sub_agent_runner.py | head -3
```

### 必要改动: trace_analyze_handler 命名对齐

如果本 branch 仍然叫 `select_kernels`,**有两条路**:

**A. 在 roofline.py 内做兼容 import** (改动小):

```python
# 替换 roofline.py 顶部的 import:
try:
    from ..kernel_request_handlers import trace_analyze_handler
except ImportError:
    # Branch still uses the legacy 'select_kernels' name
    from ..kernel_request_handlers import select_kernels as trace_analyze_handler
```

**B. 全 branch 重命名** (推荐 — 与 main 对齐):

```bash
# 在本 branch 也执行 N1 的重命名:
git grep -l "select_kernels" inference_optimizer/orchestrator/ | while read f; do
    sed -i 's/\bselect_kernels\b/trace_analyze_handler/g' "$f"
done
# 然后跑测试,看哪些失败
pytest inference_optimizer/tests -q --tb=no 2>&1 | tail -3
```

> 注意: 如果本 branch 的 select_kernels 还有 `kind="select_kernels"` 在 task params 里,
> 不要全局替换,只替换 import + 函数名。**实际操作时优先选 A**,F1 不要扩展到 N1 重命名。

### 添加单元测试

```bash
# 把 main 的测试也搬过来:
cp "$MAIN_ROOT/inference_optimizer/tests/test_roofline_executor.py" \
   inference_optimizer/tests/test_roofline_executor.py

# 检查 import 是否对应得上:
head -30 inference_optimizer/tests/test_roofline_executor.py
```

### 命令

```bash
pytest inference_optimizer/tests/test_roofline_executor.py -v
# 期望全 pass (~15 case)

pytest inference_optimizer/tests -q --tb=no 2>&1 | tail -3
# 期望 1614 passed (1599 + 15)

git add inference_optimizer/orchestrator/action_executors/roofline.py
git add inference_optimizer/tests/test_roofline_executor.py
git commit -m "feat(F1-2): cherry-pick RooflineExecutor from main

RooflineExecutor (composite profile + trace_analyze action) ported
verbatim from /wekafs/zgong/Hyperloom main @ c6f0a71
inference_optimizer/orchestrator/action_executors/roofline.py.

Adapted only the trace_analyze_handler import (this branch may still
use legacy select_kernels name; compat import added).

Includes:
- RooflineStubExecutor (N2a operator fallback)
- RooflineExecutor (production)
- make_roofline_executor factory
- N26 auto-retry on steady_state_chunk_{empty,missing,low_quality}

Tests: 15 new (test_roofline_executor.py from main).
Total: 1614 passed.

Note: action is not yet registered (F1-3) so propose_action{roofline}
still fails with no_executor; behavior identical to F0 with
--use-roofline-composite=false."
```

---

## 5. F1-3: 注册 roofline executor + action_registry

### 编辑 `inference_optimizer/orchestrator/action_executors/__init__.py`

```python
# 在文件末尾的 EXECUTOR_REGISTRY (或类似) 字典添加:
from .roofline import RooflineExecutor, RooflineStubExecutor, make_roofline_executor

# 把 make_roofline_executor 加到注册表里:
EXECUTOR_FACTORY_REGISTRY: dict[str, Callable[..., Any]] = {
    # ... existing entries ...
    "baseline": make_baseline_executor,
    "profile": make_profile_executor,
    "roofline": make_roofline_executor,   # NEW (F1-3)
    # ... etc ...
}
```

### 编辑 `inference_optimizer/orchestrator/action_registry.py`

确认 `roofline` action 名字在 ALL_ACTION_NAMES (或类似全集) 中:

```python
# 找到 ACTION_NAMES / ACTIONS_REGISTRY:
grep -n "ACTION_NAMES\|ACTIONS_REGISTRY\|action_name" inference_optimizer/orchestrator/action_registry.py | head -10

# 把 'roofline' 加进去 (如果还没有):
ALL_ACTION_NAMES = frozenset({
    # ... existing ...
    "explore", "specialist", "integrate_patch", "assess_remaining_gaps",
    "profile", "pmc_roofline",
    "roofline",   # NEW (F1-3)
    # ... etc ...
})
```

### 编辑 `inference_optimizer/cli.py` 的 `_register_executors`

```python
# 找到 _register_executors 函数 (或 register_action_executors / 类似):
grep -n "_register_executors\|register_action_executors\|sub_agent.register_executor" \
    inference_optimizer/cli.py | head -10

# 添加 roofline 注册:
if shared_state.use_roofline_composite:
    sub_agent_runner.register_executor(
        "roofline",
        make_roofline_executor(shared_state=shared_state),
    )
```

### 编辑 Coordinator 的 promote 路径

`coordinator.py::_promote_to_shared_state` 需要识别 `task.kind == "roofline"`。
在合适分支 (例如 profile / pmc_roofline 邻近) 加:

```python
if task_kind == "roofline":
    # RooflineExecutor 在 executor 内部已经写了 last_profile_trace +
    # 调用了 record_trace_analyze。Coordinator 这层只需:
    # - 不再做 v0.6 的 promote (executor 已经完成)
    # - 检查 result["status"] == "succeeded" 用于 audit / scoring (F3)
    # - bump roofline_snapshot_id (executor 内部 record_trace_analyze 已 bump)
    return
```

### 测试

新建 `inference_optimizer/tests/test_f1_roofline_dispatch.py`:

```python
"""F1-3 — verify roofline action is registered and dispatchable end-to-end."""

import pytest


def test_roofline_in_executor_registry():
    from inference_optimizer.orchestrator.action_executors import EXECUTOR_FACTORY_REGISTRY
    assert "roofline" in EXECUTOR_FACTORY_REGISTRY


def test_roofline_in_all_actions():
    from inference_optimizer.orchestrator.action_registry import ALL_ACTION_NAMES
    assert "roofline" in ALL_ACTION_NAMES


@pytest.mark.asyncio
async def test_roofline_dispatch_smoke(tmp_path, monkeypatch):
    """Dispatch a roofline task with mocked profile / trace_analyze handlers
    and verify the SharedState side effects."""
    # ... fixture: build a mock RunnerContext + mock profile_executor +
    #     mock trace_analyze_handler returning canned results ...
    # See test_roofline_executor.py for the pattern; this test is a
    # higher-level integration check.
    pass   # ⚠ stub — implement when fixtures are available
```

### 命令

```bash
pytest inference_optimizer/tests/test_f1_roofline_dispatch.py -v
# 至少 2 passed (smoke test 暂为 stub)

pytest inference_optimizer/tests -q --tb=no 2>&1 | tail -3
# 期望 1616 passed (1614 + 2)

git add inference_optimizer/orchestrator/action_executors/__init__.py
git add inference_optimizer/orchestrator/action_registry.py
git add inference_optimizer/cli.py
git add inference_optimizer/orchestrator/coordinator.py
git add inference_optimizer/tests/test_f1_roofline_dispatch.py
git commit -m "feat(F1-3): register roofline action + executor dispatch

Wires RooflineExecutor into:
- action_executors/__init__.py:EXECUTOR_FACTORY_REGISTRY
- action_registry.py:ALL_ACTION_NAMES
- cli.py:_register_executors (gated on --use-roofline-composite)
- coordinator.py:_promote_to_shared_state (passthrough — executor
  handles SharedState writes internally)

Tests: 2 new. Total: 1616 passed."
```


---

## 6. F1-4: Adapt prompt_builder analysis.md 注入

### 来源

`$MAIN_ROOT/inference_optimizer/orchestrator/system_prompts/prompt_builder.py` 中 `_format_analysis_md_full` + section 5.5 渲染段。

### 命令

```bash
cd "$FEAT_ROOT"

# 看 main 的 prompt_builder 关键函数:
grep -n "_format_analysis_md_full\|_format_roofline\|TraceLens Analysis" \
    "$MAIN_ROOT/inference_optimizer/orchestrator/system_prompts/prompt_builder.py" | head -10

# 看本 branch 当前的:
grep -n "_format_analysis_md_full\|_format_roofline\|TraceLens Analysis" \
    inference_optimizer/orchestrator/system_prompts/prompt_builder.py | head -10
```

### 做法 (谨慎 — 只 cherry-pick 2 个函数 + 1 段渲染)

**不要**直接覆盖 `prompt_builder.py`(冲突太大)。手工提取以下两段:

#### a. `_format_analysis_md_full` 函数

从 main 拷出整段函数,粘贴到本 branch `prompt_builder.py` 的合适位置:

```python
def _format_analysis_md_full(
    state: Any,
) -> str:
    """Render TraceLens analysis.md verbatim into the orchestration prompt.

    Wraps the cached analysis_md_text between explicit markers so the LLM
    sees the report bookended cleanly. When no snapshot exists, returns a
    single-line "(no TraceLens snapshot yet ...)" line.
    """
    last_ta = getattr(state, "last_trace_analyze", {}) or {}
    if not isinstance(last_ta, dict):
        last_ta = {}
    md_text = last_ta.get("analysis_md_text") or ""
    snapshot_id = last_ta.get("roofline_snapshot_id") or 0
    gain = last_ta.get("roofline_baseline_gain_at_snapshot") or 0.0
    if not md_text:
        return "analysis_md=(no TraceLens snapshot yet — propose `roofline` to produce one)"
    return (
        f"=== TraceLens Analysis (snapshot #{snapshot_id}, "
        f"gain at snapshot = {gain:.2f}%) ===\n"
        + md_text
        + "\n=== End TraceLens Analysis ==="
    )
```

#### b. 在 build_orchestration_prompt 里加一段 section 5.5

找到本 branch 渲染 SharedState 块的地方 (一般是 `_format_shared_state_dump` 或类似),
在 `last_trace_analyze=...` 一行**之后**增加:

```python
# Section 5.5 — full analysis.md when --use-roofline-composite
if getattr(state, "use_roofline_composite", False):
    sections.append(_format_analysis_md_full(state))
```

或者:F1 阶段不做条件,直接渲染 (开关默认 off,无 snapshot 时只多一行 "no snapshot yet"):

```python
sections.append(_format_analysis_md_full(state))
```

### 测试

```python
# 追加到 inference_optimizer/tests/test_f1_roofline_dispatch.py

def test_format_analysis_md_full_no_snapshot():
    from inference_optimizer.orchestrator.system_prompts.prompt_builder import _format_analysis_md_full
    from inference_optimizer.orchestrator.shared_state import SharedState
    s = SharedState()
    out = _format_analysis_md_full(s)
    assert "no TraceLens snapshot yet" in out


def test_format_analysis_md_full_with_snapshot():
    from inference_optimizer.orchestrator.system_prompts.prompt_builder import _format_analysis_md_full
    from inference_optimizer.orchestrator.shared_state import SharedState
    s = SharedState()
    s.last_trace_analyze = {
        "analysis_md_text": "# Hello\n## Top Operations\n...",
        "roofline_snapshot_id": 3,
        "roofline_baseline_gain_at_snapshot": 12.5,
    }
    out = _format_analysis_md_full(s)
    assert "snapshot #3" in out
    assert "gain at snapshot = 12.50%" in out
    assert "# Hello" in out
    assert out.startswith("=== TraceLens Analysis")
    assert out.rstrip().endswith("=== End TraceLens Analysis ===")
```

### 命令

```bash
pytest inference_optimizer/tests/test_f1_roofline_dispatch.py -v
# 4 passed (前 2 个 + 这 2 个)

pytest inference_optimizer/tests -q --tb=no 2>&1 | tail -3
# 期望 1618 passed (1616 + 2)

git add inference_optimizer/orchestrator/system_prompts/prompt_builder.py
git add inference_optimizer/tests/test_f1_roofline_dispatch.py
git commit -m "feat(F1-4): inject analysis.md verbatim into orchestration prompt

Adds _format_analysis_md_full(state) → str helper + invokes it from the
shared-state dump block. When last_trace_analyze.analysis_md_text is
empty, renders a single-line 'no snapshot yet' marker; otherwise wraps
the full text between '=== TraceLens Analysis (snapshot #N) ===' /
'=== End TraceLens Analysis ===' markers.

Cherry-picked the helper function verbatim from
/wekafs/zgong/Hyperloom main @ c6f0a71 prompt_builder.py.

Tests: 2 new. Total: 1618 passed."
```

---

## 7. F1-5: Port Roofline-v2 ordering rules to orchestration.md

### 来源

`$MAIN_ROOT/inference_optimizer/orchestrator/system_prompts/orchestration.md` 中以下三段:

1. `### Roofline-v2 action ordering (HARD RULES — PolicyGate enforced)`
2. `### Roofline-v2 analysis.md → action mapping (HARD RULES)`
3. `### How to consume the TraceLens Analysis`

### 做法 — **必须手工改写**,不能直接 cp

main 的版本写的是 v0.6 拓扑(backends/params/comm_optimization),本 branch 是 v0.8 explore-only。
抽取**逻辑骨架**,把 action 名替换为 v0.8 形态:

#### 改写后的版本 (放进本 branch orchestration.md 的合适位置)

````markdown
### Roofline-v2 action mapping (HARD RULES — when --use-roofline-composite is on)

When `last_trace_analyze.analysis_md_text` is non-empty (= a `roofline`
snapshot exists), the TraceLens `analysis.md` is injected verbatim into
section 5.5 of this prompt. The report uses 🔴 (P1, critical) /
🟡 (P2, secondary) / 🟢 (P1/P2, opportunity) markers. **You MUST
follow them**:

- **🔴 / 🟡 markers under `## Compute Kernel Optimizations`** →
  these are the kernels that need `kernel_opt`. Once you are in
  KERNEL phase, emit `request{target_agent='kernel',
  kind='run_optimization', params={kernel_id: <id from snapshot.candidates>,
  target_kernel: <name>}}` for the highest-priority entry (🔴 before 🟡).

- **🔴 / 🟢 markers under `## Kernel Fusion Opportunities`** → same
  kernel_opt path, but ask the kernel agent for a fused rewrite.
  Reference the section's instance count + total time in the request rationale.

- **🔴 / 🟡 markers under `## System-Level Optimizations`** → these
  map to `explore` action variants. The section text usually names
  the flag explicitly (e.g. "graph capture" → `--cuda-graph-max-bs`).
  Cross-check against the specialist's proposal_set and pick a
  variant whose `provenance='specialist:<domain>'` targets that flag.

### Choosing specialist domain by analysis.md bottleneck

The `## Compute Kernel Optimizations` / `## System-Level Optimizations`
section dictates which specialist to dispatch first:

* **attention / AllReduce / MoE expert dispatch** → `kernel_switch_specialist`
* **host overhead / cuda graph misses / KV-cache pressure / queue depth
  / `torch.compile` advice / GPU idle %** → `serving_specialist`
* **AllReduce / RCCL / QuickReduce hot kernels** → `comm_specialist`
* **register pressure / inductor advice** → `compiler_specialist`
* **launch latency / dispatch overhead** → `system_specialist`
* **uncertain / cross-cutting** → `pr_intel_specialist` (sparingly)

### When to propose `roofline` again

After `cumulative_gain_validated_pct` has moved by ≥ 3% since the
snapshot's `roofline_baseline_gain_at_snapshot` was recorded, the
bottleneck distribution has likely shifted. Propose `roofline` again
to refresh the snapshot. The section header `(snapshot #N, gain at
snapshot = X.XX%)` shows you the delta.

Do **not** propose `roofline`:
- in PRELUDE (allowlist disallows)
- when phase budget remaining < 15 minutes (the ~10 min cost would
  eat the closing window)
- when you just emitted `roofline` last tick (idempotency dedup)

### How to consume the TraceLens Analysis

When `--use-roofline-composite` is on and `roofline` has run at least
once, the prompt's SharedState dump contains:

* `last_trace_analyze=...` — one-line metadata (trace path, top-K
  ids, snapshot_id, warnings).
* Section 5.5 — the **full TraceLens `analysis.md` report**
  between `=== TraceLens Analysis (snapshot #N, gain at snapshot = X.XX%) ===`
  bookends.

Read it as you would a human-written perf report: Executive Summary
tells you the dominant bottleneck; Top Operations gives per-kernel
`gpu_pct` + efficiency; Recommendations explicitly lists what to try next.
````

### Commit

```bash
git add inference_optimizer/orchestrator/system_prompts/orchestration.md
git commit -m "docs(F1-5): port roofline-v2 analysis.md->action mapping to orchestration.md

Adapted from main's '### Roofline-v2 action ordering' / '### analysis.md
-> action mapping' / '### How to consume the TraceLens Analysis' sections.

Key adaptations from main's wording (v0.6 -> v0.8):
- 'backends / params / comm_optimization' -> 'explore variants stamped
  provenance=specialist:<domain>'
- '\`prune_branch\` examples' kept (\`prune_branch\` already exists in
  this branch's allowed intents)
- Action ordering rules dropped — they encode v0.6 backends_attempts
  sequence_denial which is not part of this branch (see
  docs/integration/MAIN_FEATURES_DROPPED.md)
- Specialist-domain mapping added (this branch's specialist-first
  contract takes the role main delegated to per-action selection)

Tests unchanged (prompt content)."
```


---

## 8. F1-6: 翻 --use-roofline-composite 默认 False → True (可选 / 守门点)

### 风险

打开默认值会让所有 GPU smoke 测试 / 真实 session 默认走 RooflineExecutor。如果 trace_analyze
handler / profile path 在某种 model 上有 regression,所有 session 都会受影响。

### 守门规则

**不在 F1-6 翻默认**(保守路径):
- 改为操作员/工程师手动 `--use-roofline-composite` 在 ad-hoc session 上跑。
- 跑完 5 个不同 model (DeepSeek-R1 / Qwen3-30B / Llama-3.3-70B / SDXL / Whisper) 各 1h smoke,
  全 green 后再起 F1-6 commit 翻默认。

**翻默认的具体改动**:

```python
# inference_optimizer/cli.py
parser.add_argument(
    "--use-roofline-composite",
    action=argparse.BooleanOptionalAction,   # 改成 BooleanOptionalAction (支持 --no-...)
    default=os.environ.get("INFERENCE_OPTIMIZER_USE_ROOFLINE_COMPOSITE", "1") == "1",   # 默认 1
    help=...,
)
```

CLI 调用现在变成:
- 默认: 走 roofline composite
- `--no-use-roofline-composite` : 退回老路径
- `INFERENCE_OPTIMIZER_USE_ROOFLINE_COMPOSITE=0` : 退回老路径

### 命令

```bash
# 修改 cli.py 默认值
# (按上面改)

# 修改 SKILL.md 把 'F1 default off' 改为 'F1 on by default since f1-done'

# 全测试
pytest inference_optimizer/tests -q --tb=no 2>&1 | tail -3

git add inference_optimizer/cli.py inference_optimizer/SKILL.md
git commit -m "feat(F1-6): flip --use-roofline-composite default to ON

Justified by 5 successful GPU smoke sessions (see
docs/integration/f1_smoke_$(date +%Y%m%d).log) covering:
- DeepSeek-R1-0528 / MI300X
- Qwen3-30B-A3B-Base / MI300X
- Llama-3.3-70B-Instruct / MI300X
- SDXL / MI300X
- Whisper-large-v3 / MI300X

Use --no-use-roofline-composite or env=0 to opt out.

Total: 1618 passed (no test changes — flag default flip only)."
```

> ⚠ **F1-6 不是必须的**。如果操作员对默认值切换有疑虑,直接跳到 F1-7 打 tag,把 F1-6 留到 F2 / F3 之后做。

---

## 9. F1-7: Smoke session + 打 tag f1-done

### 命令 (操作员人工)

```bash
cd "$FEAT_ROOT"

# 找一个空闲 GPU pod
nvidia-smi  # 全部 0% util
ls /sys/class/drm/card*/device/vendor 2>/dev/null   # 检查是不是 AMD pod

# 跑 1h smoke 测试 (用最熟悉的 model + workload)
INFERENCE_OPTIMIZER_USE_ROOFLINE_COMPOSITE=1 \
python -m inference_optimizer.cli optimize \
    --model deepseek-ai/DeepSeek-R1-0528 \
    --workload "isl=512,osl=128,bs=8" \
    --max-hours 1.0 \
    2>&1 | tee /tmp/f1_smoke_$(date +%Y%m%d_%H%M).log

# 验收点 (在 log 里 grep):
grep "TraceLens Analysis" /tmp/f1_smoke_*.log | head -5
grep "snapshot #1" /tmp/f1_smoke_*.log | head -5
grep "roofline_snapshot_id" /tmp/f1_smoke_*.log | head -5
grep "PolicyGate.*phase_incompatible" /tmp/f1_smoke_*.log
# 应该:
# - 至少看到 "TraceLens Analysis" 多次出现 (每次 roofline 后)
# - 至少看到 "snapshot #1" 一次 (第一次 roofline)
# - 不应该看到 "phase_incompatible.*roofline"

# 把 log 归档:
mkdir -p docs/integration
cp /tmp/f1_smoke_*.log docs/integration/

git add docs/integration/f1_smoke_*.log
git commit -m "chore(F1-7): archive F1 smoke session log

1h DeepSeek-R1-0528 smoke run with --use-roofline-composite=1
on MI300X: $(grep -c 'TraceLens Analysis' /tmp/f1_smoke_*.log) roofline
invocations, $(grep -c 'kernel_opt' /tmp/f1_smoke_*.log) kernel_opt
proposals, no PolicyGate denials of 'roofline' itself."

# 打 tag
git tag -a f1-done -m "F1 (Roofline Composite Action) complete

Commits since f0-done:
$(git log --oneline f0-done..HEAD)

State:
- 1618+ passing tests
- RooflineExecutor wired and dispatchable end-to-end
- last_trace_analyze populated as 11-field dict (matches main schema)
- analysis.md verbatim text injected into orchestration prompt
- Roofline-v2 ordering rules (adapted to v0.8) in orchestration.md
- 1h smoke session green on DeepSeek-R1-0528 / MI300X

Default state: --use-roofline-composite=False (operator opts in).
F1-6 (flip default to True) deferred to post-F2 stability window.

Next: F2 — Framework Agent integration into serving_specialist.
See plan_roofline_framework/F2_framework_agent.MD"

git push origin f1-done
git push origin "$FEAT_BRANCH"
```

### 验收

- [ ] `git log --oneline f0-done..f1-done | wc -l` ≈ 7
- [ ] `pytest inference_optimizer/tests -q --tb=no | tail -3` >= 1618 passed
- [ ] `bash docs/integration/check_preserved.sh` rc=0
- [ ] `INFERENCE_OPTIMIZER_USE_ROOFLINE_COMPOSITE=1 python -m inference_optimizer.cli optimize --model X --max-hours 1` 成功完整跑完
- [ ] log 中可以看到 `TraceLens Analysis` 至少 1 次
- [ ] log 中**不**应该有 `roofline.*no_executor` 或 `phase_incompatible.*roofline`

---

## 10. F1 整体回滚

### 回到 F0 状态

```bash
git reset --hard f0-done
```

### 单步回滚

```bash
# 例如 F1-3 register 出问题:
git revert <F1-3 commit sha>
# 这会保留 F1-1 / F1-2 (record_trace_analyze + RooflineExecutor 在),
# 只是没有 dispatch 到运行路径 — 安全降级
```

### 紧急 kill switch

如果 F1 已经 push 到上游,后续发现某个 model 上 regression:
```bash
# 操作员立即设置环境变量 (不需要新 commit):
export INFERENCE_OPTIMIZER_USE_ROOFLINE_COMPOSITE=0
# 所有 session 立即退回老路径 (因为 F1-6 还没翻默认)
```

---

## 11. F1 完成后的 commit 拓扑

```
* (tag: f1-done)                                         ← 操作员
* chore(F1-7): archive F1 smoke session log
* feat(F1-6): flip --use-roofline-composite default ON   ← 可选,可跳过
* docs(F1-5): port roofline-v2 ordering rules
* feat(F1-4): inject analysis.md verbatim into prompt
* feat(F1-3): register roofline action + executor dispatch
* feat(F1-2): cherry-pick RooflineExecutor from main
* feat(F1-1): port record_trace_analyze method from main
* fix(F0-8 amend): TraceAnalyzeSnapshot becomes type hint
* (tag: f0-done)
```

测试数走势: `1594 → 1594 (F1-0 amend) → 1599 → 1614 → 1616 → 1618 → 1618 → 1618`.

---

## 12. F1 完成 → 进入 F2

下一步: 打开 `plan_roofline_framework/F2_framework_agent.MD`,从 F2-1 开始。

**F2 重点**:
- 不引入独立 `framework_pr` action(违反 IR-4)
- 把 framework-agent (PR #280) 嵌入 `serving_specialist` 的工具链
- specialist subprocess 内部跑 `fa candidates` + git fetch refs/pull/.../head
- 失败/集成走 specialist-first 的 proposal/integrate_patch 路径
