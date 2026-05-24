# F2 — Framework Agent 接入 Serving Specialist

> 工作日: 2-3 天
> 风险: 中-高 (新增子进程类型 + 网络访问 + git fetch refs/pull/...)
> 依赖: F1 完成 (tag `f1-done` 在位)
> 完成节点: tag `f2-done` + ~6 个 commit
>
> **完成 F2 之后:**
> - 默认行为不变 (`--framework-agent-enabled=false`)
> - `--framework-agent-enabled=true` 时,LLM 在 EXPLORE 提
>   `dispatch_specialist{domain='serving_specialist', sub_kind='framework_pr_scout'}`
>   (新增子类型),subprocess 内可调 `fa candidates` 拿到 PR list,然后
>   通过 specialist proposal 走 integrate_patch 路径
>
> **关键设计**: PR #280 在 main 上引入了**独立的 `framework_pr` action**,
> 这违反 IR-4 (specialist-first)。本 F2 把 framework-agent 嵌入 specialist 工具链,
> 不引入新的 top-level action。

## 0. 环境准备

```bash
export FEAT_ROOT="/wekafs/zgong/Hyperloom-KB"
export MAIN_ROOT="/wekafs/zgong/Hyperloom"

cd "$FEAT_ROOT"
git log --oneline f1-done -1            # 确认 F1 tag 存在
git status --short                       # 必须干净
ls framework-agent/                      # F0-7 已经把目录拷过来了
test -f framework-agent/SKILL.md && echo OK
test -f framework-agent/scripts/install.sh && echo OK
```

## 1. 关键设计决策 (与 main 形态对比)

| 维度 | main (PR #280) | F2 (本 branch) |
|---|---|---|
| Top-level action | 新增 `framework_pr` action | **不新增**;走 specialist |
| Bandit arm | 5th arm in EXPLORE | **不引入** bandit 概念(已退场) |
| Specialist 角色 | `framework_specialist` (新) | 复用 `serving_specialist` |
| sub_kind | n/a | `framework_pr_scout` (新增) |
| CLI flag | `--framework-gap`, `--framework-pr-discover` | `--framework-agent-enabled` (单一开关) |
| Coordinator 改动 | 大 (~280 行 promote 路径) | 小 (~30 行: 新增 sub_kind dispatch) |
| Subprocess 桥 | `fa agent prepare-task` / `commit-result` | 同样,但走 SpecialistRunner 已有的 subprocess machinery |
| Integration 路径 | `framework_integrate_handler` (新) | 复用 `integrate_patch` action(已存在) |

---

## 2. 步骤清单

```
F2-1   framework-agent install + ensure_framework_agent       -> 1 commit
F2-2   serving_specialist 加 framework_pr_scout sub_kind 工具 -> 1 commit
F2-3   specialist subprocess sandbox 加 git fetch 网络白名单  -> 1 commit
F2-4   serving_specialist prompt 加 framework PR 段           -> 1 commit
F2-5   integrate_patch 接收 fa-produced patch + KB 写回        -> 1 commit
F2-6   smoke session + 打 tag f2-done                          -> 1 commit (操作员)
```

---

## 3. F2-1: framework-agent 安装

### 来源

- `$MAIN_ROOT/framework-agent/scripts/install.sh` (已通过 F0-7 cp 整个目录)
- `$MAIN_ROOT/inference_optimizer/scripts/install.sh` 中的 `ensure_framework_agent` 段

### 命令

```bash
cd "$FEAT_ROOT"

# 看 main 的 ensure_framework_agent 段在哪:
grep -n "ensure_framework_agent\|framework-agent" "$MAIN_ROOT/inference_optimizer/scripts/install.sh" | head -10

# 把这段 cherry-pick 到本 branch install.sh,但要避开 IR-3 的 cortex-kb 段:
grep -n "cortex-kb\|preflight_kb" inference_optimizer/scripts/install.sh | head -5

# 推荐做法: 在 install.sh 末尾追加一个新函数 + 一个 main 调用,中间不动既有代码
cat >> inference_optimizer/scripts/install.sh << 'INNER_EOF'

# ---- F2-1: framework-agent (PR #280 sibling skill) ----
ensure_framework_agent() {
    if [[ "${INFERENCE_OPTIMIZER_FRAMEWORK_AGENT_ENABLED:-0}" != "1" ]]; then
        echo "  framework-agent: not enabled (skip)"
        return 0
    fi

    local fa_dir="${INFERENCE_OPTIMIZER_REPO:-$(pwd)}/framework-agent"
    if [[ ! -d "$fa_dir" ]]; then
        echo "  framework-agent: directory missing at $fa_dir"
        return 1
    fi
    if [[ ! -f "$fa_dir/scripts/install.sh" ]]; then
        echo "  framework-agent: install.sh missing"
        return 1
    fi
    echo "  framework-agent: installing..."
    bash "$fa_dir/scripts/install.sh"
    echo "  framework-agent: OK"
}

ensure_framework_agent
INNER_EOF
```

### 验证

```bash
# Smoke install (不开 flag,应该 skip)
bash inference_optimizer/scripts/install.sh 2>&1 | grep -E "framework-agent|cortex-kb" | head

# 开 flag,实际安装
INFERENCE_OPTIMIZER_FRAMEWORK_AGENT_ENABLED=1 \
   bash inference_optimizer/scripts/install.sh 2>&1 | grep -E "framework-agent" | head

# 验证 fa CLI
which fa
fa --version || echo "fa not installed yet (expected if smoke install skipped)"
```

### Commit

```bash
git add inference_optimizer/scripts/install.sh
git commit -m "feat(F2-1): wire framework-agent install behind --framework-agent-enabled

Adds ensure_framework_agent() function to install.sh that runs
framework-agent/scripts/install.sh ONLY when env
INFERENCE_OPTIMIZER_FRAMEWORK_AGENT_ENABLED=1 is set (set by --framework-agent-enabled
F0-10 toggle).

Skipped by default — keeps F0/F1 install path identical to pre-F2."
```

---

## 4. F2-2: serving_specialist 加 framework_pr_scout sub_kind 工具

### 设计

`serving_specialist` 当前的 subprocess 工具集 (在 SpecialistDomain.tools 里) 大概是:
- read_artifact
- read_kb_section
- propose_variant
- (可能的) git ops

F2-2 加一个新的 sub_kind: `framework_pr_scout`,该 sub_kind 进程内允许:
- 调 `fa candidates --gap '<gap>' --backend sglang|vllm` (PR list 探索)
- 调 `git fetch <fork>/refs/pull/<N>/head:fa/<N>` (拿 PR diff)
- 写 proposal: `propose_variant` 携带 `provenance='specialist:serving:framework_pr',
  fa_pr_url=...`,并把 patch 路径放进 `proposal_extra.patch_path`

### 编辑 `inference_optimizer/orchestrator/specialist_domains.py`

```python
SERVING_SPECIALIST = SpecialistDomain(
    key="serving_specialist",
    name="Serving Specialist",
    available_in="M5",
    # ... existing fields ...
    sub_kinds=frozenset({
        "tune_serving_flags",          # existing
        "framework_pr_scout",          # NEW (F2-2)
    }),
    # ... rest ...
)
```

### 编辑 `inference_optimizer/orchestrator/specialist_subprocess.py`

找到 subprocess 启动逻辑 (传入工作目录 / env / 工具白名单的地方):

```python
def _build_subprocess_env(domain: SpecialistDomain, sub_kind: str, ...) -> dict:
    env = base_env.copy()

    if sub_kind == "framework_pr_scout":
        # F2-2: framework_pr_scout 需要 framework-agent 工具链
        if not shared_state.framework_agent_enabled:
            raise PolicyError(
                "specialist:framework_pr_scout requires --framework-agent-enabled"
            )
        env["INFERENCE_OPTIMIZER_FA_ENABLED"] = "1"
        env["FA_ALLOW_NETWORK_FETCH"] = "1"   # Allow `fa candidates` HTTP + git fetch
        env["FA_KB_PATH"] = str(_kb_root_for_specialist())

    return env


def _build_subprocess_argv(domain: SpecialistDomain, sub_kind: str, ...) -> list[str]:
    argv = base_argv.copy()
    if sub_kind == "framework_pr_scout":
        # 加 framework-agent prompt 段引用
        argv += [
            "--include-prompt-fragment",
            "framework-agent/prompts/serving_specialist_pr_scout.md",
        ]
    return argv
```

### 编辑 `inference_optimizer/orchestrator/coordinator.py`

dispatch_specialist 路径加 sub_kind 校验:

```python
def _validate_dispatch_specialist(intent: dict, ...) -> None:
    domain = intent.get("domain")
    sub_kind = intent.get("sub_kind", "default")

    domain_obj = get_specialist_domain(domain)
    if sub_kind not in domain_obj.sub_kinds:
        raise PolicyError(f"specialist {domain} does not support sub_kind={sub_kind}")
    # ... existing checks ...

    if sub_kind == "framework_pr_scout":
        if not shared_state.framework_agent_enabled:
            raise PolicyError(
                f"sub_kind=framework_pr_scout requires --framework-agent-enabled"
            )
```

### 测试

```bash
mkdir -p inference_optimizer/tests
cat > inference_optimizer/tests/test_f2_framework_pr_scout.py << 'TEST_EOF'
"""F2-2 — framework_pr_scout sub_kind dispatch validation."""

import pytest


def test_serving_specialist_has_framework_pr_scout_sub_kind():
    from inference_optimizer.orchestrator.specialist_domains import SERVING_SPECIALIST
    assert "framework_pr_scout" in SERVING_SPECIALIST.sub_kinds


def test_dispatch_framework_pr_scout_blocked_when_flag_off():
    from inference_optimizer.orchestrator.shared_state import SharedState
    from inference_optimizer.orchestrator.coordinator import _validate_dispatch_specialist
    s = SharedState()
    s.framework_agent_enabled = False
    intent = {
        "kind": "dispatch_specialist",
        "domain": "serving_specialist",
        "sub_kind": "framework_pr_scout",
    }
    with pytest.raises(Exception, match="framework-agent-enabled"):
        _validate_dispatch_specialist(intent, shared_state=s)


def test_dispatch_framework_pr_scout_allowed_when_flag_on():
    from inference_optimizer.orchestrator.shared_state import SharedState
    from inference_optimizer.orchestrator.coordinator import _validate_dispatch_specialist
    s = SharedState()
    s.framework_agent_enabled = True
    intent = {
        "kind": "dispatch_specialist",
        "domain": "serving_specialist",
        "sub_kind": "framework_pr_scout",
    }
    _validate_dispatch_specialist(intent, shared_state=s)   # no raise


def test_dispatch_unknown_sub_kind_blocked():
    from inference_optimizer.orchestrator.shared_state import SharedState
    from inference_optimizer.orchestrator.coordinator import _validate_dispatch_specialist
    s = SharedState()
    s.framework_agent_enabled = True
    intent = {
        "kind": "dispatch_specialist",
        "domain": "serving_specialist",
        "sub_kind": "totally_unknown",
    }
    with pytest.raises(Exception, match="sub_kind"):
        _validate_dispatch_specialist(intent, shared_state=s)
TEST_EOF
```

### 命令

```bash
pytest inference_optimizer/tests/test_f2_framework_pr_scout.py -v
# 期望 4 passed

pytest inference_optimizer/tests -q --tb=no 2>&1 | tail -3
# 期望 1622 passed (1618 + 4)

git add inference_optimizer/orchestrator/specialist_domains.py
git add inference_optimizer/orchestrator/specialist_subprocess.py
git add inference_optimizer/orchestrator/coordinator.py
git add inference_optimizer/tests/test_f2_framework_pr_scout.py
git commit -m "feat(F2-2): serving_specialist gains framework_pr_scout sub_kind

Adds 'framework_pr_scout' as a sub_kind of serving_specialist
(in addition to the existing 'tune_serving_flags' default).

When dispatched:
- Subprocess gets INFERENCE_OPTIMIZER_FA_ENABLED=1 and FA_ALLOW_NETWORK_FETCH=1
- Allows specialist to invoke 'fa candidates' (HTTP search via Primus
  Cortex + GitHub Search) and 'git fetch <fork>/refs/pull/<N>/head:fa/<N>'
- Specialist outputs proposals with provenance='specialist:serving:framework_pr'
  and proposal_extra.patch_path

Gated on shared_state.framework_agent_enabled (--framework-agent-enabled
flag from F0-10). PolicyGate rejects the dispatch when flag is off.

Tests: 4 new. Total: 1622 passed."
```


---

## 5. F2-3: Specialist subprocess sandbox 加 git fetch 网络白名单

### 背景

`SpecialistRunner` (本 branch PR-A2 引入) 跑 LLM 的 subprocess。subprocess 默认运行在
worktree 沙箱里:
- 受限文件系统 (`--add-dir` 列表)
- 默认不允许 HTTP 出网 (`--no-network` 或类似)

`framework_pr_scout` 需要:
1. HTTP: `fa candidates` 调 Primus Cortex API + GitHub Search REST
2. Git fetch: `git fetch <fork-remote>/refs/pull/<N>/head:fa/<N>`

### 设计

**只在 framework_pr_scout sub_kind 下**放开网络。

### 编辑 `inference_optimizer/orchestrator/specialist_subprocess.py`

```python
def _build_subprocess_argv_for_specialist(
    domain: SpecialistDomain,
    sub_kind: str,
    worktree: Path,
    ...
) -> list[str]:
    argv = ["claude", "--print", "--output-format", "stream-json", ...]

    # 默认 sandbox 配置
    argv += ["--add-dir", str(worktree)]
    argv += ["--add-dir", str(_artifacts_dir())]
    argv += ["--add-dir", str(_kb_dir())]

    # 默认无网,默认 deny tools
    argv += ["--deny-tool", "Bash:*curl*"]
    argv += ["--deny-tool", "Bash:*wget*"]
    argv += ["--deny-tool", "WebSearch"]
    argv += ["--deny-tool", "WebFetch"]

    if sub_kind == "framework_pr_scout":
        # F2-3: 放开 PR 探索所需工具
        argv += ["--add-dir", str(worktree / "framework-agent-cache")]
        # 解除 deny:
        argv = [a for a in argv if a not in ("Bash:*curl*", "Bash:*wget*", "WebFetch")]
        # 但仍然限制白名单:
        argv += ["--allow-tool", "Bash:fa *"]
        argv += ["--allow-tool", "Bash:git fetch *"]
        argv += ["--allow-tool", "Bash:git diff *"]
        argv += ["--allow-tool", "Bash:git log *"]

    # ... rest ...
    return argv
```

> 注意: 实际的 `--allow-tool` / `--deny-tool` 形式取决于你们 SpecialistRunner 用的 LLM CLI
> (Claude Code / Cursor CLI / Codex)。**这一段需要按本 branch 实际语法调整**。

### 编辑 `inference_optimizer/orchestrator/policy.py` 加 PolicyGate 规则

```python
@policy_rule
def specialist_framework_pr_scout_requires_flag(intent, state):
    """F2-3: dispatch_specialist{sub_kind=framework_pr_scout} requires
    --framework-agent-enabled. Mirrors coordinator-level check but at
    PolicyGate boundary so denials carry rule_id."""
    if intent.get("kind") != "dispatch_specialist":
        return ALLOW
    if intent.get("sub_kind") != "framework_pr_scout":
        return ALLOW
    if not state.framework_agent_enabled:
        return DENY(
            rule="specialist_framework_pr_scout_requires_flag",
            message=(
                "framework_pr_scout subprocess needs HTTP + git fetch "
                "network access; gated behind --framework-agent-enabled "
                "for security. Set the flag and retry."
            ),
            recovery_hint="prune_branch{reason='flag_off'}",
        )
    return ALLOW
```

### 测试

```python
# 追加到 inference_optimizer/tests/test_f2_framework_pr_scout.py

def test_subprocess_argv_includes_fa_when_framework_pr_scout(tmp_path):
    from inference_optimizer.orchestrator.specialist_subprocess import (
        _build_subprocess_argv_for_specialist,
    )
    from inference_optimizer.orchestrator.specialist_domains import SERVING_SPECIALIST
    argv = _build_subprocess_argv_for_specialist(
        SERVING_SPECIALIST, "framework_pr_scout", tmp_path,
    )
    argv_str = " ".join(argv)
    assert "Bash:fa *" in argv_str
    assert "Bash:git fetch *" in argv_str


def test_subprocess_argv_default_sub_kind_no_network(tmp_path):
    from inference_optimizer.orchestrator.specialist_subprocess import (
        _build_subprocess_argv_for_specialist,
    )
    from inference_optimizer.orchestrator.specialist_domains import SERVING_SPECIALIST
    argv = _build_subprocess_argv_for_specialist(
        SERVING_SPECIALIST, "tune_serving_flags", tmp_path,
    )
    argv_str = " ".join(argv)
    assert "WebFetch" in argv_str   # still deny-listed
    assert "fa *" not in argv_str   # not allow-listed
```

### 命令

```bash
pytest inference_optimizer/tests/test_f2_framework_pr_scout.py -v
# 期望 6 passed (4 + 2)

pytest inference_optimizer/tests -q --tb=no 2>&1 | tail -3
# 期望 1624 passed

git add inference_optimizer/orchestrator/specialist_subprocess.py
git add inference_optimizer/orchestrator/policy.py
git add inference_optimizer/tests/test_f2_framework_pr_scout.py
git commit -m "feat(F2-3): allow specialist subprocess net access for framework_pr_scout

Differential sandbox: framework_pr_scout sub_kind unlocks
- WebFetch / Bash:curl / Bash:wget
- Bash:fa * / Bash:git fetch *

while default sub_kind ('tune_serving_flags') keeps net+git deny-listed.

PolicyGate rule specialist_framework_pr_scout_requires_flag denies the
dispatch when --framework-agent-enabled is off (defense in depth — the
coordinator-level check already does this, but PolicyGate emits a
rule_id for audit).

Tests: 2 new. Total: 1624 passed."
```

---

## 6. F2-4: serving_specialist prompt 加 framework PR 段

### 来源

`$MAIN_ROOT/framework-agent/prompts/` 中的 prompt 模板 (PR #280 给 framework_pr arm 写的)。

### 命令

```bash
cd "$FEAT_ROOT"

# 看 main 框架 prompt 在哪:
ls "$MAIN_ROOT/framework-agent/prompts/" 2>/dev/null

# 通常会有 framework_pr_explore.md / serving_pr_scout.md / 类似的
# 把整个 prompts 目录拷过来 (F0-7 已经拷了 framework-agent/ 整个目录,这一步只验证)
ls framework-agent/prompts/
```

### 新建 `framework-agent/prompts/serving_specialist_pr_scout.md` (适配本 branch)

```markdown
# Serving Specialist — Framework PR Scout sub-kind

You are dispatched as `serving_specialist{sub_kind='framework_pr_scout'}`
in the EXPLORE phase. Your goal is to discover an upstream PR
(sglang/vllm) that addresses a known gap, fetch it, and emit a
`propose_variant` with the resulting patch.

## Workflow

1. **Read the gap**: the dispatch carries a `gap` field in
   `task.params`. Decide which upstream backend is more likely to
   carry a fix (sglang vs vllm) — usually the project we are running.

2. **Search**:
   ```bash
   fa candidates --gap "$GAP" --backend sglang --max 10
   ```
   Emits a JSON list of `{pr_url, title, sha, primus_score, gh_score}`.

3. **Pick**: highest primus_score with title matching the gap.
   Cross-check `gh_score >= 0.5` (avoids spam PRs).

4. **Fetch**:
   ```bash
   git fetch <fork-remote> refs/pull/<N>/head:fa/<N>
   git diff main...fa/<N> > /tmp/fa-<N>.patch
   ```

5. **Validate locally** (cheap):
   - `git apply --check /tmp/fa-<N>.patch` — must succeed
   - `wc -l` of patch — reject if > 5000 lines (too big for single proposal)

6. **Propose**:
   ```json
   {
     "kind": "propose_variant",
     "domain": "serving_specialist",
     "sub_kind": "framework_pr_scout",
     "provenance": "specialist:serving:framework_pr",
     "fa_pr_url": "https://github.com/sgl-project/sglang/pull/1234",
     "fa_pr_sha": "<sha>",
     "proposal_extra": {
       "patch_path": "/tmp/fa-<N>.patch",
       "primus_score": 0.85,
       "gh_score": 0.7
     },
     "rationale": "PR #1234 addresses <gap> by ...",
     "expected_gain_pct": <int 0-30>
   }
   ```

## Hard Rules

- You **MUST NOT** integrate the patch yourself; emit `propose_variant`
  and let Coordinator route to `integrate_patch` (which runs in a
  GPU lane with rebuild + smoke validation).
- You **MUST** include `fa_pr_url` and `fa_pr_sha` in the proposal —
  KB writeback (F2-5) keys on these.
- If `fa candidates` returns empty (no PR found), emit
  `propose_variant{provenance='specialist:serving:framework_pr',
   proposal_extra={empty_result: true}}` — do NOT propose with
  `expected_gain_pct=0` (that wastes a tick).
- Cap subprocess wall time at 8 minutes. If `fa candidates` is slow,
  abort the search and emit empty_result.
```

### 命令

```bash
git add framework-agent/prompts/serving_specialist_pr_scout.md
git commit -m "feat(F2-4): serving_specialist framework_pr_scout prompt

Hard-rule prompt for the new sub_kind. Restricts the specialist to
emitting propose_variant (never directly integrate); KB writeback
contract requires fa_pr_url + fa_pr_sha; soft 8-min cap on
fa candidates wall time.

No code change beyond the new prompt file."
```


---

## 7. F2-5: integrate_patch 接收 fa-produced patch + KB 写回

### 设计

`integrate_patch` action 已存在 (本 branch 用于 kernel proposal 集成)。F2-5 扩展它接收
specialist 的 `proposal_extra.patch_path`,且在成功后写一条 KB 记录到
`kb/framework_optimization/` 用于后续 session priors。

### 编辑 `inference_optimizer/orchestrator/action_executors/integrate_patch.py`

```python
async def integrate_patch_executor(ctx: RunnerContext) -> dict[str, Any]:
    params = ctx.task.params or {}
    patch_path = params.get("patch_path") or ""
    provenance = params.get("provenance") or ""

    if not patch_path:
        return _failed("no patch_path in params")

    # ---- Step 1: apply ----
    ok = await _git_apply(patch_path, worktree=ctx.worktree)
    if not ok:
        return _failed("git apply failed")

    # ---- Step 2: rebuild ----
    rebuild_ok = await _rebuild(worktree=ctx.worktree)
    if not rebuild_ok:
        await _git_revert(worktree=ctx.worktree)
        return _failed("rebuild failed; reverted")

    # ---- Step 3: smoke benchmark ----
    smoke_result = await _smoke_benchmark(worktree=ctx.worktree)
    if not smoke_result["ok"]:
        await _git_revert(worktree=ctx.worktree)
        return _failed(f"smoke failed: {smoke_result['error']}")

    # ---- Step 4 (NEW F2-5): KB writeback for framework_pr provenance ----
    if provenance.startswith("specialist:serving:framework_pr"):
        from ..kb_writeback import write_framework_pr_record
        await write_framework_pr_record(
            pr_url=params.get("fa_pr_url", ""),
            pr_sha=params.get("fa_pr_sha", ""),
            patch_path=patch_path,
            outcome="integrated",
            tps_delta_pct=smoke_result.get("delta_pct", 0.0),
            session_id=ctx.session_id,
        )

    return {"status": "succeeded", "smoke_result": smoke_result}
```

### 新建 `inference_optimizer/orchestrator/kb_writeback.py`

```python
"""F2-5 — KB writeback adapters for specialist outcomes.

Each function appends a YAML/JSONL record into kb/framework_optimization/
or kb/serving/ for use as priors in future sessions."""

from __future__ import annotations
import json
import time
from pathlib import Path

KB_ROOT = Path(__file__).parents[2] / "framework-agent" / "kb" / "framework_optimization"


async def write_framework_pr_record(
    *,
    pr_url: str,
    pr_sha: str,
    patch_path: str,
    outcome: str,           # "integrated" | "reverted_smoke_fail" | "rejected_apply_fail"
    tps_delta_pct: float,
    session_id: str,
) -> None:
    """Append a single JSONL record under kb/framework_optimization/lessons.jsonl"""
    KB_ROOT.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.time(),
        "session_id": session_id,
        "pr_url": pr_url,
        "pr_sha": pr_sha,
        "patch_path": str(patch_path),
        "outcome": outcome,
        "tps_delta_pct": float(tps_delta_pct),
    }
    with (KB_ROOT / "lessons.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
```

### 测试

```python
# inference_optimizer/tests/test_f2_kb_writeback.py
from __future__ import annotations
import json
import pytest
from pathlib import Path


@pytest.mark.asyncio
async def test_write_framework_pr_record(tmp_path, monkeypatch):
    from inference_optimizer.orchestrator import kb_writeback
    monkeypatch.setattr(kb_writeback, "KB_ROOT", tmp_path)
    await kb_writeback.write_framework_pr_record(
        pr_url="https://github.com/sgl-project/sglang/pull/1234",
        pr_sha="abc123",
        patch_path="/tmp/fa-1234.patch",
        outcome="integrated",
        tps_delta_pct=4.5,
        session_id="20260524T140800Z",
    )
    rec_file = tmp_path / "lessons.jsonl"
    assert rec_file.exists()
    line = rec_file.read_text().strip()
    record = json.loads(line)
    assert record["pr_url"] == "https://github.com/sgl-project/sglang/pull/1234"
    assert record["pr_sha"] == "abc123"
    assert record["outcome"] == "integrated"
    assert record["tps_delta_pct"] == 4.5


@pytest.mark.asyncio
async def test_write_framework_pr_record_appends(tmp_path, monkeypatch):
    """Ensure two writes do not clobber each other."""
    from inference_optimizer.orchestrator import kb_writeback
    monkeypatch.setattr(kb_writeback, "KB_ROOT", tmp_path)
    for sha in ("aaa", "bbb"):
        await kb_writeback.write_framework_pr_record(
            pr_url=f"https://github.com/x/y/pull/{sha}",
            pr_sha=sha, patch_path=f"/tmp/{sha}.patch",
            outcome="integrated", tps_delta_pct=1.0,
            session_id="s",
        )
    lines = (tmp_path / "lessons.jsonl").read_text().splitlines()
    assert len(lines) == 2
```

### 命令

```bash
pytest inference_optimizer/tests/test_f2_kb_writeback.py -v
# 期望 2 passed

pytest inference_optimizer/tests -q --tb=no 2>&1 | tail -3
# 期望 1626 passed (1624 + 2)

git add inference_optimizer/orchestrator/action_executors/integrate_patch.py
git add inference_optimizer/orchestrator/kb_writeback.py
git add inference_optimizer/tests/test_f2_kb_writeback.py
git commit -m "feat(F2-5): integrate_patch + KB writeback for framework_pr provenance

When integrate_patch succeeds with provenance starting
'specialist:serving:framework_pr', append a record to
framework-agent/kb/framework_optimization/lessons.jsonl carrying
pr_url, pr_sha, outcome, tps_delta_pct.

These records become priors for future sessions: 'PR X integrated
successfully on Y model — try again before scouting fresh'.

Tests: 2 new. Total: 1626 passed."
```

---

## 8. F2-6: Smoke session + 打 tag f2-done

### 命令 (操作员人工)

```bash
cd "$FEAT_ROOT"

# 找空闲 GPU
nvidia-smi

# 启用 framework-agent 跑 smoke
INFERENCE_OPTIMIZER_FRAMEWORK_AGENT_ENABLED=1 \
INFERENCE_OPTIMIZER_USE_ROOFLINE_COMPOSITE=1 \
python -m inference_optimizer.cli optimize \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --workload "isl=1024,osl=256,bs=4" \
    --max-hours 2.0 \
    2>&1 | tee /tmp/f2_smoke_$(date +%Y%m%d_%H%M).log

# 验收 grep:
grep "framework_pr_scout" /tmp/f2_smoke_*.log | head
grep "fa candidates" /tmp/f2_smoke_*.log | head
grep "specialist:serving:framework_pr" /tmp/f2_smoke_*.log | head
grep "framework-agent.*lessons.jsonl" /tmp/f2_smoke_*.log | head
# 应该:
# - 至少 1 次 framework_pr_scout 被 dispatch
# - fa candidates 至少跑一次 (返回 PR list)
# - 至少 1 个 propose_variant 携带 specialist:serving:framework_pr provenance
# - (如果该次 integrate 成功) lessons.jsonl 写入

# 归档 log + KB 内容
cp /tmp/f2_smoke_*.log docs/integration/

# 如果 lessons.jsonl 有内容,归档 (脱敏)
if [[ -s framework-agent/kb/framework_optimization/lessons.jsonl ]]; then
    cp framework-agent/kb/framework_optimization/lessons.jsonl \
       docs/integration/f2_smoke_kb_records_$(date +%Y%m%d).jsonl
fi

git add docs/integration/f2_smoke_*.log docs/integration/f2_smoke_kb_records_*.jsonl
git commit -m "chore(F2-6): archive F2 smoke session log + KB records

2h Llama-3.3-70B-Instruct smoke run with --framework-agent-enabled=1
+ --use-roofline-composite=1 on MI300X.

Outcome:
- $(grep -c framework_pr_scout /tmp/f2_smoke_*.log) framework_pr_scout dispatches
- $(grep -c 'specialist:serving:framework_pr' /tmp/f2_smoke_*.log) framework_pr proposals
- $(wc -l < framework-agent/kb/framework_optimization/lessons.jsonl 2>/dev/null || echo 0) KB records appended"

git tag -a f2-done -m "F2 (Framework Agent integration) complete

Commits since f1-done:
$(git log --oneline f1-done..HEAD)

State:
- 1626+ passing tests
- framework-agent installed lazily (only on --framework-agent-enabled)
- serving_specialist supports sub_kind=framework_pr_scout
- specialist subprocess differential sandbox (network for fa, deny for rest)
- integrate_patch handles fa-produced patches + KB writeback
- 2h Llama-3.3 smoke session green with both flags on

Default state:
- --framework-agent-enabled=False (operator opts in)
- --use-roofline-composite=False (operator opts in)

Iron rule preservation:
- IR-4 specialist-first preserved (NO new top-level framework_pr action)
- IR-3 cortex-kb preflight unchanged
- KB_design §3.4 / §3.9 / §3.13 specialist domain semantics preserved

Next: F3 — PolicyGate rules + roofline saturation soft advisory.
See plan_roofline_framework/F3_policygate_advisory.MD"

git push origin f2-done
git push origin "$FEAT_BRANCH"
```

### 验收

- [ ] `git log --oneline f1-done..f2-done | wc -l` ≈ 6
- [ ] `pytest inference_optimizer/tests -q --tb=no | tail -3` >= 1626 passed
- [ ] `bash docs/integration/check_preserved.sh` rc=0
- [ ] 2h smoke session log 含 `framework_pr_scout` dispatch ≥ 1 次
- [ ] log 中**不**应该有 `framework_pr` 作为 top-level action(违反设计)
- [ ] log 中**不**应该有 `bandit_arm.*framework` (我们没有引入 bandit arm 概念)

---

## 9. F2 整体回滚

### 回到 F1 状态

```bash
git reset --hard f1-done
```

### 单步回滚示例

```bash
# 例如 F2-3 沙箱配置出问题:
git revert <F2-3 commit sha>
# 系统回到: framework_pr_scout sub_kind 存在但 subprocess 无网络 →
# specialist 启动后 'fa candidates' 失败 → empty_result proposal → 干净降级
```

### 紧急 kill switch

```bash
# 不需要 commit:
export INFERENCE_OPTIMIZER_FRAMEWORK_AGENT_ENABLED=0
# 所有 framework_pr_scout dispatch 立即被 PolicyGate 拒
```

---

## 10. F2 完成后的 commit 拓扑

```
* (tag: f2-done)
* chore(F2-6): archive F2 smoke session log
* feat(F2-5): integrate_patch + KB writeback
* feat(F2-4): serving_specialist framework_pr_scout prompt
* feat(F2-3): allow specialist subprocess net access
* feat(F2-2): serving_specialist gains framework_pr_scout sub_kind
* feat(F2-1): wire framework-agent install
* (tag: f1-done)
```

测试数走势: `1618 → 1618 → 1622 → 1624 → 1624 → 1626 → 1626`.

---

## 11. F2 完成 → 进入 F3

下一步: 打开 `plan_roofline_framework/F3_policygate_advisory.MD`,从 F3-1 开始。

**F3 重点**:
- N9 deny direct profile (when `--use-roofline-composite=true`)
- N19c gain-driven kernel_opt lock
- N31 final roofline (CLOSE phase 自动 enqueue)
- Roofline saturation soft advisory (注入 prompt,不硬拒)
- Replace v0.6 sequence_denial 规则
