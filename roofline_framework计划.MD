# Roofline + Framework Agent 融合计划 (F0 → F3)

> Branch: `feature/zhenggong/lossen_explore` (51 commits ahead, 134 commits behind origin/main)
> 目标: 把 PR #288 (roofline-v2) + PR #280 (framework-agent) 的核心能力按 v0.8 形态重新组织进本 branch
> 原则: **不直接 `git merge origin/main`**;按"重新实现 + 测试驱动"分四个 PR 渐进推进
> 总工作量: ~6-8 工作日,远低于直接 merge (~10-14 天) 的成本

---

## 目录

- [§1 总体策略](#1-总体策略)
- [§2 F0 — Pre-merge 准备清单 (1 天)](#2-f0--pre-merge-准备清单-1-天)
- [§3 F1 — Roofline 复合 Action (2-3 天)](#3-f1--roofline-复合-action-2-3-天)
- [§4 F2 — Framework Agent 接入 Serving Specialist (2-3 天)](#4-f2--framework-agent-接入-serving-specialist-2-3-天)
- [§5 F3 — PolicyGate 关键规则 + Soft Advisory (1-2 天)](#5-f3--policygate-关键规则--soft-advisory-1-2-天)
- [§6 风险点 + 回滚开关](#6-风险点--回滚开关)
- [§7 必须保留 / 主动放弃 清单](#7-必须保留--主动放弃-清单)
- [§8 时间线 + 验收标准](#8-时间线--验收标准)


---

## 1. 总体策略

### 1.1 为什么不直接 git merge origin/main

134 个 commit 一次性合并会同时引入:

1. **语义改写**: PR #288 的所有规则(N9 / N19c / sequence_denial)都建立在 v0.6 的
   `backends_attempts` / `params_attempts` / `validate_stack_attempts` 字段上。本 branch 已经
   把这些字段对应的 action 物理删除(commit 76389d8),v0.6 字段在 SharedState 仍然存在但
   **没有任何写入路径**。后果:`backends_attempts < 1` 这条 sequence_denial 规则会**永远成立**,
   把 kernel_opt 永久锁死。
2. **文件级冲突**: 14 个 content + 7 个 modify/delete (见 §2.4 dry-run merge 输出)。
3. **测试错位**: PR #288 引入的 ~15 个 `test_n*.py` 全部针对 v0.6 路径,合并后大批失败;
   本 branch 的 ~30 个 specialist / phase_force_exit / assess_remaining_gaps 测试可能被改坏。

### 1.2 推荐路线: 4 个独立 PR

| PR | 内容 | 规模 | 依赖 | 风险 |
|---|---|---|---|---|
| **F0** | Pre-merge 整理 + 加性 scaffolding | 1 天 | 无 | 极低 |
| **F1** | `roofline` 复合 action 替代 profile + pmc_roofline | 2-3 天 | F0 | 中 |
| **F2** | `framework-agent/` 子目录 + serving_specialist 调用 fa | 2-3 天 | F1 | 中 |
| **F3** | N9 / N19c / N31 关键 PolicyGate 规则 + roofline-aware soft advisory | 1-2 天 | F1 + F2 | 低 |

每个 PR 必须独立通过 1584 个现有测试,任何一步出问题立即回滚到上一个 tag。

### 1.3 关键设计决策(必须先达成共识)

| 决策点 | 选择 | 替代方案 + 不选的理由 |
|---|---|---|
| Framework agent 形态 | **方案 C — 嵌入 serving_specialist** | A (独立 bandit arm) 与 IR-4 specialist-first 冲突;B (新增 framework_pr_specialist domain) 重复(serving_specialist 已覆盖 sglang/vllm) |
| Roofline 落地 phase | **EXPLORE 入口必跑 + 内部按需 + KERNEL 入口必跑 + CLOSE 收尾** | PRELUDE 内嵌违反"PRELUDE 只做 baseline"语义;SWEEP 不需要 perf 报告 |
| Roofline-aware dispatch | **软 advisory(prompt 注入)** | 硬 gate 风险:roofline 信号有 5-15% 噪声,会过早收敛,阉割 specialist 挑战 roofline 的能力 |
| 旧 v0.6 文件命运 | **保持删除** | 重新引入会破坏 KB_design §3.4 explore-merge 设计 |
| `framework_pr` action 命运 | **不引入独立 action**,patch 走 integrate_patch | 引入会要求 IR-4 开特例口子 |
| Scoreboard 命运 | **保持退场**(KB_design §3.9) | 重新引入与 KB_priors 角色重叠 |


---

## 2. F0 — Pre-merge 准备清单 (1 天)

F0 全部是**加性 + 文档**操作,完成后 git 历史里多出 ~10 个 commit,实际代码行为完全等同当前 HEAD。
作用是把后续 F1/F2/F3 的合并面缩小、回滚机制铺好、协作者/agent 的 freeze 协议立起来。

### 2.1 步骤 1 — 归档根目录 7 个未跟踪 .MD 文件

当前未跟踪文件(`git status` 输出):
```
?? Agent-deligate-gap.MD
?? KB_design_bugs.MD
?? Version1_report.MD
?? cortex-kb-http-branch-b-2026-05-20.md
?? 优化报告_DeepSeek-R1-0528_MI300X_SGLang.md
?? 优化报告_DeepSeek-R1-0528_MI300X_vLLM.md
?? 新增PR报告roofline_framework.MD
?? roofline_framework计划.MD     # 本文件
```

**操作命令**:
```bash
mkdir -p docs/notes_lossen_explore
mv Agent-deligate-gap.MD KB_design_bugs.MD Version1_report.MD \
   cortex-kb-http-branch-b-2026-05-20.md \
   "优化报告_DeepSeek-R1-0528_MI300X_SGLang.md" \
   "优化报告_DeepSeek-R1-0528_MI300X_vLLM.md" \
   "新增PR报告roofline_framework.MD" \
   "roofline_framework计划.MD" \
   docs/notes_lossen_explore/
git add docs/notes_lossen_explore/
git commit -m "docs: archive lossen_explore session notes + integration plan"
```

**验收**: `git status --short` 不再显示根目录有 `??` 的 .MD 文件。

### 2.2 步骤 2 — 打 tag,建立确定的回滚锚点

```bash
git tag -a pre-roofline-merge -m "Last commit on lossen_explore before main integration (134 commits behind origin/main; PR #288 + #280 not yet integrated)"
git push origin pre-roofline-merge
```

**作用**:
- 任何一步出问题: `git reset --hard pre-roofline-merge` 一秒回到原点
- 后续 PR 描述里直接引用 tag 作为 "before" 状态

**注意**: 这一步**需要操作员人工执行**(涉及 push 到远端)。

### 2.3 步骤 3 — 锁定测试绿色基线 (1584 passed)

PR-A11 的 commit (a1d2730) 自报 **1584 passed, 1 skipped (不算 4 个无关 pre-existing 失败)**。
F0 之前必须本地复现这个数字,作为 F1/F2/F3 每一步的对照。

```bash
cd /wekafs/zgong/Hyperloom-KB
pytest inference_optimizer/tests -q --tb=no 2>&1 | tee /tmp/baseline_pre_merge_pytest.log
tail -3 /tmp/baseline_pre_merge_pytest.log
# 期望末尾形如: "1584 passed, 1 skipped, 4 failed in XXX.XXs"

mkdir -p docs/test-baselines
cp /tmp/baseline_pre_merge_pytest.log docs/test-baselines/pre_roofline_merge_$(date +%Y%m%d).log
git add docs/test-baselines/
git commit -m "chore(tests): archive pre-merge regression baseline (1584 passed, $(date +%F))"
```

**红线**: 如果当下不是 1584 passed (或差距 > 5),**先停下来修绿,不要做合并**。
在红色基线上做合并,无法判断"哪些失败是合并引起的"。

### 2.4 步骤 4 — 跑 dry-run merge,归档全量冲突清单

```bash
git fetch origin main
DRY_BRANCH="dry-run-merge-$(date +%Y%m%d)"
git checkout -b "$DRY_BRANCH"

# 不 commit 不 push,只看冲突
git merge --no-commit --no-ff origin/main 2>&1 | tee /tmp/dryrun_merge.log
git status --short > /tmp/dryrun_status.log
git diff --name-only --diff-filter=U > /tmp/dryrun_unmerged.log
git diff --name-only --diff-filter=D > /tmp/dryrun_deleted.log

# 立即放弃 merge,回到 feature branch
git merge --abort
git checkout feature/zhenggong/lossen_explore
git branch -D "$DRY_BRANCH"

# 归档进 git
mkdir -p docs/integration
cp /tmp/dryrun_*.log docs/integration/
git add docs/integration/dryrun_*.log
git commit -m "chore(integration): archive dry-run merge conflict map ($(date +%F))"
```

**已知冲突清单**(见 `新增PR报告roofline_framework.MD` §2.1):
- 14 个 content 冲突
- 7 个 modify/delete 冲突 (`backends.py`, `params.py`, `validate_stack.py`, `scoring.py`,
  `test_p3_search_space_expansion.py`, `test_validate_stack.py`, `test_validate_stack_gate_skip.py`)

**用途**: F1/F2/F3 每次解冲突时,对照这份归档"我现在解的这个冲突,在 dry-run 里是不是也存在",
避免引入新冲突。


### 2.5 步骤 5 — 生成 MUST_PRESERVE.md (本 branch 不可丢失资产清单)

写入 `docs/integration/MUST_PRESERVE.md`,作为 merge 期间 LLM/operator 都要核对的清单。

**清单内容**:

| 类别 | 项 | 来源 commit | merge 期保护方法 |
|---|---|---|---|
| **物理删除** | `action_executors/{backends,params,validate_stack}.py` | 76389d8 | merge 时遇 `modify/delete` 冲突,选 HEAD (删除) |
| **物理删除** | `scoring.py` | a86bdf2 | 同上 |
| **物理删除** | 6 个 v0.6 测试文件 | 76389d8 | 同上 |
| **新文件** | `phase_state.py` (1379 行) | c02c90a | main 不存在,无冲突,自动保留 |
| **新文件** | `specialist_runner.py` / `specialist_subprocess.py` / `specialist_domains.py` | 6b20bf8 + PR-A2 系列 | 同上 |
| **重命名** | `framework_specialist → serving_specialist` | 1c28dd6 | 文本搜索全局确认无 framework_specialist 残留 |
| **重命名** | `kernel_specialist → kernel_switch_specialist` | 1c28dd6 | 同上 |
| **PolicyGate 规则** | `action_deprecated` | Gap-10 | 在 F0 step 9 写一个 `test_policy_action_deprecated_kept.py` 守住 |
| **PolicyGate 规则** | `explore_requires_specialist_provenance` (IR-4) | PR-A9 | 同上 |
| **PolicyGate 规则** | `assess_remaining_gaps_throttle` (IR-7) | e974567 | 同上 |
| **CLI flag** | `--explore-force-exit-hours-remaining` | e974567 | 同上 |
| **CLI flag** | `--explore-force-exit-budget-pct` | e974567 | 同上 |
| **CLI flag** | `--steward-disabled` | e974567 | 同上 |
| **SharedState 字段** | `last_remaining_gaps_assessment` | e974567 | 同上 |
| **SharedState 字段** | `remaining_gaps_assessments` (cap 10) | e974567 | 同上 |
| **SharedState 字段** | `steward_continuation_used` | e974567 | 同上 |
| **SharedState 字段** | `specialist_domain_empty_streak` | bab2930 | 同上 |
| **Iron Rule** | IR-1 / IR-2 / IR-3 / **IR-4** / IR-6 / **IR-7** | SKILL.md + commits | SKILL.md 整段不动 |
| **测试** | `test_phase_force_exit.py` (10 cases) | e974567 | 自动保留(main 无对应文件) |
| **测试** | `test_assess_remaining_gaps.py` (24 cases) | e974567 | 同上 |
| **测试** | `test_specialist_*.py` 全套 | 6b20bf8 + 系列 | 同上 |
| **Cortex KB** | HTTP transport + IR-3 soft-degrade preflight | d6036a2 | merge 时 cli.py 冲突,手工保留 |

**生成方式**:
```bash
# 这一份 MUST_PRESERVE.md 由我(LLM 助手)在 F0 阶段一并生成,
# 内容自动从上表 + git log 的 commit refs 拼出。
```

### 2.6 步骤 6 — 生成 MAIN_FEATURES_DROPPED.md (主动放弃清单)

写入 `docs/integration/MAIN_FEATURES_DROPPED.md`,记录我们**主动选择不要** main 的哪些产物。
作用: 解冲突时直接选"我们方"无需犹豫;PR 描述里引用此清单解释为什么不合某段。

| 主动放弃 | 来源 PR | 不要的理由 | 替代方案 |
|---|---|---|---|
| `action_executors/{backends,params,validate_stack}.py` (main 版本) | #288 | v0.6 已退役 | F1 把功能合并进 explore |
| `scoring.py` (main 版本) | #288 | scoreboard 退场 (KB_design §3.9) | KB priors 在 Critic 侧 |
| `system_prompts/orchestration.md` 的 `Action scores top-12 block` 段 | (历史) | scoreboard 退场后无意义 | 已用"phase + gaps + KB priors"决策 |
| `framework_pr first-explore priority rule` | #280 | 与 IR-4 specialist-first 冲突 | F2 把 fa 嵌入 serving_specialist |
| `backends_attempts < 1` sequence_denial | #288 | v0.8 没有 backends | F3 重写为基于 explore 的等价规则 |
| `params_attempts < 1` sequence_denial | #288 | 同上 | 同上 |
| 提到 `validate_stack` 必须性的 prompt 规则 | (历史) | M3 已经把它内联进 explore | KEEP 后 explore 自动 rebench |
| main 的 `system_prompts/prompt_builder.py` 全量 (+434/-45) | #288 | 本 branch 已大幅重写 | F1 仅 cherry-pick "analysis.md 直注 prompt"段 |
| `--framework-gap` / `--framework-pr-discover` CLI 旧接口 | #280 | 已被 fa 内部化 | F2 走 serving_specialist 调 fa |
| main 的 `coordinator.py` 全量 diff (+1179/-82) | #288 | v0.6 路径 | F1 + F3 按需 cherry-pick |


### 2.7 步骤 7 — 加性 Scaffolding: 从 main cherry-pick 纯新增文件

PR #288 / #280 中**完全新增、与本 branch 不冲突**的文件可以提前 cherry-pick,把后续 F1/F2 的合并面缩小。

**操作命令**:
```bash
# (a) PR #280 整个 framework-agent/ 子目录 (顶层新目录,本 branch 无对应)
git checkout origin/main -- framework-agent/
# 检查没动到本 branch 的其他文件:
git status --short

# (b) PR #288 文档类文件
git checkout origin/main -- inference_optimizer/actions/roofline.md
git checkout origin/main -- inference_optimizer/actions/_meta/roofline.yaml

# (c) PR #288 纯 parser 文件 (无 v0.6 依赖)
git checkout origin/main -- inference_optimizer/orchestrator/roofline_snapshot.py

# (d) PR #288 关键词字典 (纯数据)
git checkout origin/main -- inference_optimizer/orchestrator/_analysis_keyword_map.py

# (e) install.sh 的 framework-agent 安装步(可能需要手工合并)
git diff origin/main -- inference_optimizer/scripts/install.sh | head -100
# 如果只是追加 ensure_framework_agent 函数,直接接受 main:
git checkout origin/main -- inference_optimizer/scripts/install.sh
# 如果有冲突,手工把 framework-agent 段挑出来加到末尾

git add framework-agent/ inference_optimizer/
git commit -m "feat(integration F0a): import additive scaffolding from main

- framework-agent/ (entire new top-level directory from PR #280)
- inference_optimizer/orchestrator/roofline_snapshot.py (parser, no deps)
- inference_optimizer/orchestrator/_analysis_keyword_map.py (data dict)
- inference_optimizer/actions/roofline.md + _meta/roofline.yaml (docs)
- inference_optimizer/scripts/install.sh (framework-agent install hook)

These files have no v0.6 dependencies. They are placed onto the tree
ahead of the F1/F2 functional integration so the cherry-picks for
roofline executor / specialist wiring will see fewer real conflicts.
F1 will wire roofline_snapshot.py into the actual roofline executor;
F2 will wire framework-agent/ into serving_specialist."
```

**验证**:
```bash
pytest inference_optimizer/tests -q --tb=no 2>&1 | tail -3
# 应该仍然 1584 passed (新增的文件没有改任何运行路径)
```

**注意**:
- 这一步**不要** cherry-pick `action_executors/roofline.py` (引用 v0.6 backends 路径)
- 这一步**不要** cherry-pick `action_executors/framework_pr.py` (F2 直接放弃,改走 specialist)
- 这一步**不要** cherry-pick `cli.py` / `coordinator.py` / `shared_state.py` 任何改动

### 2.8 步骤 8 — 在 SharedState 预留 roofline 字段(纯加性)

F1 / F3 都需要新字段。F0 阶段先以"加性 + 默认空值"方式落地,F1 cherry-pick `roofline.py` 时
字段访问全部命中,实际冲突大幅缩小。

**编辑 `inference_optimizer/orchestrator/shared_state.py`**(伪代码,实际用 StrReplace):

```python
@dataclass
class TraceAnalyzeSnapshot:
    """F0 placeholder for roofline-v2 (PR #288) trace_analyze cache.

    F1 will populate this from action_executors/roofline.py;
    until then it stays at default (snapshot_id=0 = no snapshot yet).
    """
    snapshot_id: int = 0
    analysis_md_path: str = ""
    analysis_md_text: str = ""
    candidates_path: str = ""
    timestamp_unix: float = 0.0


@dataclass
class SharedState:
    # ... existing fields unchanged ...

    # F0 placeholder fields for PR #288 (roofline-v2). F1 wires writers.
    last_trace_analyze: TraceAnalyzeSnapshot = field(default_factory=TraceAnalyzeSnapshot)
    roofline_snapshot_id: int = 0   # monotonic, incremented in F1's RooflineExecutor
    roofline_saturation_history: list[dict] = field(default_factory=list)   # F3 populates
```

**配套测试**:
```python
# inference_optimizer/tests/test_f0_roofline_scaffolding.py (新增)
def test_shared_state_has_roofline_placeholder_fields():
    s = SharedState(...)
    assert hasattr(s, "last_trace_analyze")
    assert s.last_trace_analyze.snapshot_id == 0
    assert s.roofline_snapshot_id == 0
    assert s.roofline_saturation_history == []

def test_trace_analyze_snapshot_default():
    snap = TraceAnalyzeSnapshot()
    assert snap.snapshot_id == 0
    assert snap.analysis_md_text == ""
```

```bash
pytest inference_optimizer/tests/test_f0_roofline_scaffolding.py -v
# 期望 2 passed
pytest inference_optimizer/tests -q --tb=no 2>&1 | tail -3
# 期望 1586 passed (1584 + 2 new)
git add inference_optimizer/orchestrator/shared_state.py inference_optimizer/tests/test_f0_roofline_scaffolding.py
git commit -m "feat(F0): pre-add roofline placeholder fields on SharedState (default empty)"
```


### 2.9 步骤 9 — 在 phase_state.py EXPLORE allowlist 加 roofline 占位

```python
# inference_optimizer/orchestrator/phase_state.py
PHASE_ALLOWED_ACTIONS: dict[str, frozenset[str]] = {
    PHASE_PRELUDE: frozenset({
        "target_analysis", "baseline", "recover",
    }),
    PHASE_EXPLORE: frozenset({
        "explore", "specialist",
        "integrate_patch",
        "assess_remaining_gaps",
        "profile", "pmc_roofline",
        "roofline",   # ← F0 占位; F1 落 RooflineExecutor 后才生效
        "recover",
    }),
    PHASE_KERNEL: frozenset({
        "profile", "pmc_roofline",
        "roofline",   # ← F0 占位; F1 落 RooflineExecutor 后才生效
        "kernel_opt", "integrate", "deep_kernel_analysis",
        "operator_tuning", "vendor_kernel_config",
        "recover",
    }),
    PHASE_SWEEP: frozenset({
        "sweep", "recover",
    }),
    PHASE_CLOSE: frozenset({
        "report", "session_breakdown",
        "roofline",   # ← F0 占位; F3 N31 final comparison 启用
        "recover",
    }),
}
```

**配套测试**:
```python
# inference_optimizer/tests/test_f0_roofline_scaffolding.py 追加
def test_roofline_in_explore_allowlist():
    from inference_optimizer.orchestrator.phase_state import (
        PHASE_EXPLORE, is_action_allowed_in_phase,
    )
    assert is_action_allowed_in_phase("roofline", PHASE_EXPLORE)

def test_roofline_in_kernel_allowlist():
    from inference_optimizer.orchestrator.phase_state import (
        PHASE_KERNEL, is_action_allowed_in_phase,
    )
    assert is_action_allowed_in_phase("roofline", PHASE_KERNEL)

def test_roofline_in_close_allowlist():
    from inference_optimizer.orchestrator.phase_state import (
        PHASE_CLOSE, is_action_allowed_in_phase,
    )
    assert is_action_allowed_in_phase("roofline", PHASE_CLOSE)

def test_roofline_NOT_in_prelude_or_sweep():
    from inference_optimizer.orchestrator.phase_state import (
        PHASE_PRELUDE, PHASE_SWEEP, is_action_allowed_in_phase,
    )
    assert not is_action_allowed_in_phase("roofline", PHASE_PRELUDE)
    assert not is_action_allowed_in_phase("roofline", PHASE_SWEEP)
```

```bash
pytest inference_optimizer/tests/test_f0_roofline_scaffolding.py -v
# 期望 6 passed (2 from step 8 + 4 here)
git add inference_optimizer/orchestrator/phase_state.py inference_optimizer/tests/test_f0_roofline_scaffolding.py
git commit -m "feat(F0): add 'roofline' placeholder to EXPLORE/KERNEL/CLOSE allowlists"
```

**作用**: F1 cherry-pick `roofline.py` 时,phase 检查直接放行,不需要在 PR 里同时改 phase_state.py。

### 2.10 步骤 10 — 加 5 个 env 开关 scaffolding (默认全 off)

在 `inference_optimizer/cli.py` 的 argparse 处统一加入,**默认值都让行为等同当前 branch**(即未启用任何 main 的新行为)。

```python
# 找到 cli.py 里的 argparse 定义,在合适位置插入:

# F1 — Roofline composite action
parser.add_argument(
    "--use-roofline-composite",
    action="store_true",
    default=False,
    help="(F1) Enable RooflineExecutor (composite profile + trace_analyze). "
         "Default off — falls back to separate profile / pmc_roofline actions.",
)

# F2 — Framework agent inside serving_specialist
parser.add_argument(
    "--framework-agent-enabled",
    action="store_true",
    default=False,
    help="(F2) serving_specialist may invoke `fa candidates` and "
         "`git fetch refs/pull/...` to pull upstream sglang/vllm PRs as "
         "specialist patch sources. Patches still go through integrate_patch.",
)

# F3 — N9 deny direct profile
parser.add_argument(
    "--deny-direct-profile",
    action="store_true",
    default=False,
    help="(F3 / N9) PolicyGate denies propose_action{profile} once "
         "--use-roofline-composite is on. Forces the LLM to use the "
         "atomic 'roofline' action.",
)

# F3 — N19c gain-driven kernel_opt unlock
parser.add_argument(
    "--gain-driven-kernel-opt",
    action="store_true",
    default=False,
    help="(F3 / N19c) Lock kernel_opt until last_cheap_delta_gain < epsilon. "
         "Prevents premature deep work when cheap exploration is still earning gain.",
)

# F3 — Roofline-aware soft advisory (the §4 idea from the prior report)
parser.add_argument(
    "--roofline-saturation-advisory",
    action="store_true",
    default=False,
    help="(F3) Inject saturated_layers hint into specialist + orchestration "
         "prompts when a layer's roofline saturation exceeds threshold. "
         "Soft advisory only — never hard-rejects an LLM intent.",
)
```

**配套环境变量**(给生产 env 用,优先级低于 CLI):
```python
# 在 SharedState 里把这些开关读进 state:
self.use_roofline_composite = bool(
    args.use_roofline_composite
    or os.environ.get("INFERENCE_OPTIMIZER_USE_ROOFLINE_COMPOSITE") == "1"
)
self.framework_agent_enabled = bool(
    args.framework_agent_enabled
    or os.environ.get("INFERENCE_OPTIMIZER_FRAMEWORK_AGENT_ENABLED") == "1"
)
# ... 同样 4 个其他开关 ...
```

**配套测试**:
```python
def test_f0_toggles_default_off():
    args = make_default_args()
    assert args.use_roofline_composite is False
    assert args.framework_agent_enabled is False
    assert args.deny_direct_profile is False
    assert args.gain_driven_kernel_opt is False
    assert args.roofline_saturation_advisory is False
```

```bash
pytest inference_optimizer/tests/test_f0_roofline_scaffolding.py -v
# 期望 7 passed
git add inference_optimizer/cli.py inference_optimizer/tests/test_f0_roofline_scaffolding.py
git commit -m "feat(F0): add F1/F2/F3 env toggles scaffolding (default off)

This commit only adds the CLI flags + env var bindings. Subsequent F1/F2/F3
PRs will:

- F1: --use-roofline-composite default flips to True after RooflineExecutor lands
- F2: --framework-agent-enabled default flips to True after serving_specialist
      gains fa support
- F3: --deny-direct-profile / --gain-driven-kernel-opt / --roofline-saturation-advisory
      default flips per PR-by-PR risk assessment

Operators / CI can preserve old behaviour any time by passing the negative
form (--no-... in argparse, or unset env)."
```


### 2.11 步骤 11 — SKILL.md 加 Integration Freeze 段

在 `inference_optimizer/SKILL.md` 顶部 (在"Iron Rules"之后,"Setup"之前)插入一段"集成冻结期"声明,
让协作者 / Cursor agent / Claude 都看到这个分支当前处于受控集成阶段。

```markdown
## Integration Freeze (2026-05-24 → F3 done)

This branch is in a managed integration phase against origin/main
(roofline-v2 #288 + framework-agent #280). Until F3 lands:

### What you MUST NOT do

- **Do NOT** `git merge origin/main` directly. Use the F0/F1/F2/F3
  cherry-pick path documented in `docs/notes_lossen_explore/roofline_framework计划.MD`.
- **Do NOT** re-introduce `backends.py` / `params.py` / `validate_stack.py` /
  `scoring.py`. They are intentionally deleted (KB_design §3.4 / §3.9 /
  KB_gaps/Gap-10). See `docs/integration/MUST_PRESERVE.md`.
- **Do NOT** add a `framework_pr first-explore priority` rule to
  `system_prompts/orchestration.md`. It conflicts with IR-4 specialist-first
  contract. See `docs/integration/MAIN_FEATURES_DROPPED.md`.
- **Do NOT** add a sequence_denial that consumes `backends_attempts` or
  `params_attempts` fields. v0.8 has no writers for those fields; the
  denial would be permanently true and lock kernel_opt forever.

### What you MAY do

- New work that touches Coordinator / SharedState / PolicyGate must
  rebase atop the F0 scaffolding commits (placeholder `roofline_snapshot_id`,
  `TraceAnalyzeSnapshot`, EXPLORE/KERNEL/CLOSE allowlist `roofline` placeholder).
- Add new tests under `inference_optimizer/tests/test_f0_*.py` /
  `test_f1_*.py` / etc.
- Use the `--use-roofline-composite` / `--framework-agent-enabled` /
  `--deny-direct-profile` / `--gain-driven-kernel-opt` /
  `--roofline-saturation-advisory` toggles to gate F1/F2/F3 features.

### References

- `docs/notes_lossen_explore/新增PR报告roofline_framework.MD` —
  full PR #288 / #280 evaluation
- `docs/notes_lossen_explore/roofline_framework计划.MD` —
  this integration plan (F0 → F3)
- `docs/integration/MUST_PRESERVE.md` — v0.8 features that must
  survive integration
- `docs/integration/MAIN_FEATURES_DROPPED.md` — main features
  intentionally dropped
- `docs/integration/dryrun_*.log` — original conflict map snapshot
- Tag `pre-roofline-merge` — rollback anchor
```

```bash
git add inference_optimizer/SKILL.md
git commit -m "docs(SKILL): add Integration Freeze section for F0→F3 phase"
```

**这一步需要操作员人工 review SKILL.md 的措辞**(SKILL.md 是 LLM 硬契约,不能轻易改)。

### 2.12 F0 完成后的 commit 拓扑

```
* (F0 step 11) docs(SKILL): add Integration Freeze section
* (F0 step 10) feat(F0): add F1/F2/F3 env toggles scaffolding (default off)
* (F0 step 9)  feat(F0): add 'roofline' placeholder to EXPLORE/KERNEL/CLOSE allowlists
* (F0 step 8)  feat(F0): pre-add roofline placeholder fields on SharedState
* (F0 step 7)  feat(F0a): import additive scaffolding from main
* (F0 step 6)  docs(integration): MAIN_FEATURES_DROPPED.md
* (F0 step 5)  docs(integration): MUST_PRESERVE.md
* (F0 step 4)  chore(integration): archive dry-run merge conflict map
* (F0 step 3)  chore(tests): archive pre-merge regression baseline
* (F0 step 1)  docs: archive lossen_explore session notes
* HEAD before F0  (tag: pre-roofline-merge)
```

每个 commit 后跑一次 `pytest inference_optimizer/tests -q --tb=no | tail -3`,
绿色测试数必须**单调不降**(允许新增测试带来上升)。

### 2.13 F0 验收标准

- [ ] `git status --short` 干净 (无未跟踪 .MD)
- [ ] `git tag -l pre-roofline-merge` 显示 tag 存在,`git ls-remote --tags origin pre-roofline-merge` 远端也有
- [ ] `docs/test-baselines/pre_roofline_merge_*.log` 末尾显示 1584+ passed
- [ ] `docs/integration/dryrun_*.log` 三份文件齐全
- [ ] `docs/integration/MUST_PRESERVE.md` + `MAIN_FEATURES_DROPPED.md` 存在
- [ ] `framework-agent/` 子目录已落地 (`ls framework-agent/SKILL.md` 存在)
- [ ] `inference_optimizer/orchestrator/roofline_snapshot.py` 存在
- [ ] `inference_optimizer/actions/roofline.md` + `_meta/roofline.yaml` 存在
- [ ] `pytest inference_optimizer/tests -q --tb=no | tail -3` ≥ 1591 passed (1584 + 7 F0 新增)
- [ ] CLI: `python -m inference_optimizer.cli --help | grep -E 'use-roofline-composite|framework-agent-enabled|deny-direct-profile|gain-driven-kernel-opt|roofline-saturation-advisory'` 显示 5 个开关
- [ ] `python -m inference_optimizer.cli optimize --model X --max-hours 0.1 --dry-run`
      默认行为(无任何开关)与 `pre-roofline-merge` 完全一致
- [ ] `inference_optimizer/SKILL.md` 含 "Integration Freeze" 段落


---

## 3. F1 — Roofline 复合 Action (2-3 天)

### 3.1 目标

把当前 EXPLORE allowlist 临时开放的 `profile` + `pmc_roofline` **替换**成单一的 `roofline` 复合 action,
让 LLM 不再需要"先 propose profile → 等结果 → 再 propose trace_analyze"两步序列。
单 action 内部 executor 原子化执行 profile + trace_analyze,产出 `last_trace_analyze.analysis_md_text`
供下个 LLM tick 直接读。

### 3.2 步骤

**Step F1-1**: cherry-pick `inference_optimizer/orchestrator/action_executors/roofline.py`
- 从 main 取来,**手工修改**: 删掉所有对 v0.6 `last_backends` / `backends_attempts` 等字段的
  访问;改为只写 `last_trace_analyze` (F0 step 8 已预留)
- 保留 N25/N26 steady-state chunk auto-retry 逻辑(纯 executor 内部,与 v0.8 拓扑无冲突)

**Step F1-2**: 注册到 `action_executors/__init__.py` + `action_registry.py`
- 在 EXECUTOR_REGISTRY 加 `"roofline": make_roofline_executor`
- 在 ACTION_NAMES 集合加 `"roofline"`

**Step F1-3**: Coordinator `_promote_to_shared_state` 增加 roofline 分支
- 当 `task.kind == "roofline"` 且 `result["status"] == "succeeded"` 时,把 `result["analysis_md_text"]`
  写入 `state.last_trace_analyze.analysis_md_text`,并 `state.roofline_snapshot_id += 1`

**Step F1-4**: prompt_builder 注入 analysis.md 全文
- 仅 cherry-pick main 的 `prompt_builder.py` 中"`_format_analysis_md_full` 函数 + section 5.5"段落
- 不要 cherry-pick 其他段(其他段引用 v0.6 backends/params 字段)

**Step F1-5**: `system_prompts/orchestration.md` 局部更新
- 在"### Phase awareness"之后插入 main 的"### Roofline-v2 analysis.md → action mapping"段
- **手工删掉**该段中所有提到 `backends` / `params` / `validate_stack` 的子条目,
  改为指向 `explore` action + specialist-first 路径

**Step F1-6**: 翻 `--use-roofline-composite` 默认值 `False → True`
- 同步加一个 `--legacy-profile-actions` 开关让操作员可以回滚

**Step F1-7**: 测试
- cherry-pick `test_roofline_executor.py` (~394 行,纯 executor 单元测试,可用)
- cherry-pick `test_n10_roofline_promote_persistence.py` (~311 行)
- cherry-pick `test_n26_steady_state_auto_retry.py` (~490 行)
- **不要** cherry-pick `test_n19c_gain_driven_gates.py` / `test_n20_*.py` / `test_n22_*.py` (F3 才做)
- 写新测试 `test_f1_roofline_replaces_profile.py`: 当 `--use-roofline-composite` on 时,
  EXPLORE 阶段 propose `profile` 仍然被允许(F1 不引入 N9),但 `roofline` 优先

### 3.3 F1 验收

- [ ] `pytest inference_optimizer/tests -q --tb=no | tail -3` ≥ 1591 + ~30 F1 新增 = ~1621 passed
- [ ] 一个 smoke session 跑通: `--use-roofline-composite --max-hours 0.5`,
      manifest 显示至少 1 个 `roofline` task 成功
- [ ] `state.last_trace_analyze.analysis_md_text` 非空
- [ ] 关掉开关 (`--use-roofline-composite=false`) 行为完全等同 F0 状态

---

## 4. F2 — Framework Agent 接入 Serving Specialist (2-3 天)

### 4.1 目标

把 `framework-agent/` (F0 已落地) 接到 `serving_specialist` 域的工具链上。
不引入独立 `framework_pr` action,patch 走现有 `integrate_patch`。

### 4.2 步骤

**Step F2-1**: serving_specialist 工具白名单扩展
- 在 `specialist_runner.py` 的 `DEFAULT_SPECIALIST_TOOLS` 之外,引入
  per-domain override: `serving_specialist` 的工具表里加上 `mcp__framework_agent__candidates` /
  `mcp__framework_agent__enrich`(走 MCP),或者直接给 Bash 一个允许 pattern `^fa (candidates|enrich)\b`

**Step F2-2**: serving_specialist prompt 增强
- 在 `specialist_prompt_builder.py` 的 serving_specialist focus 模板里加一段:
  "When `framework_agent_enabled=true`, you may run `fa candidates --gap '<gap>' --keywords '<keywords>'`
  to enumerate upstream sglang/vllm PRs, then `git fetch origin pull/<N>/head` + `git apply` into
  `worktree/patches/<NN>_pr<id>_<sha>.patch`. Patches will be applied by integrate_patch."

**Step F2-3**: gap auto-composer
- cherry-pick `_framework_gap_composer.py` (~225 行) — 这个文件**完全独立**,无 v0.6 依赖
- 让 specialist runner 在 prepare 阶段调一次 `compose_gap()`,把结果放进
  `prompt_inputs.notes` 给 LLM 看

**Step F2-4**: install.sh 验证 framework-agent 可调用
- F0 step 7 已经 cherry-pick 了 install.sh;F2 跑一次 `bash install.sh --check-only`
- 期望 `fa --help` 在新 venv 里 rc=0

**Step F2-5**: 翻 `--framework-agent-enabled` 默认 `False → True`

**Step F2-6**: 测试
- cherry-pick `test_framework_pr_executor.py`(改造为 specialist-driven,~445 行 → 估计删一半)
- 写新测试 `test_f2_serving_specialist_calls_fa.py`:
  - mock fa subprocess,驱动 serving_specialist 走完一轮,断言 `worktree/patches/` 至少有 1 个 .patch
  - 断言 `specialist_done.json` 的 `proposal_set` 含 `provenance='framework_pr_via_specialist'`
- **不要** cherry-pick `test_framework_pr_discover.py`(~507 行,绑定 v0.6 bandit arm)

### 4.3 F2 验收

- [ ] `pytest -q --tb=no | tail -3` ≥ 1621 + ~25 F2 新增 = ~1646 passed
- [ ] `fa candidates --help` rc=0
- [ ] smoke: `--framework-agent-enabled --max-hours 0.5` 一个 session,
      manifest 显示至少一个 serving_specialist 任务调用了 fa
- [ ] `runs/specialist/<task_id>/worktree/patches/` 存在 .patch 文件


---

## 5. F3 — PolicyGate 关键规则 + Soft Advisory (1-2 天)

### 5.1 目标

把 PR #288 中真正高价值的 PolicyGate 规则按 v0.8 形态重写引入,并落地"roofline-aware soft advisory"
(本文件§4 在 `新增PR报告roofline_framework.MD` 中讨论的想法)。

### 5.2 N9 — 禁止直接 propose `profile`

**触发条件**: `state.use_roofline_composite is True`
**规则**: PolicyGate `direct_profile_denied`,`propose_action{action_name='profile'}` 直接拒,hint = "use roofline action"
**实现位置**: `inference_optimizer/orchestrator/policy.py`
**测试**: `test_f3_n9_direct_profile_denied.py`
**回滚**: `--deny-direct-profile=false`

### 5.3 N19c — Gain-driven kernel_opt unlock

**触发条件**: `state.gain_driven_kernel_opt is True` 且 `state.kernel_enabled is True`
**规则**: PolicyGate `kernel_opt_premature`,当 `state.last_cheap_delta_gain >= 0.3%` 时拒
`request{kind="run_optimization"}`,hint = "wait for cheap exploration to plateau"
**实现位置**: `policy.py` 新增规则;`shared_state.py` 加 `last_cheap_delta_gain` 字段(F1 explore promote 时写入)
**测试**: `test_f3_n19c_gain_driven_gate.py`
**回滚**: `--gain-driven-kernel-opt=false`

### 5.4 N31 — Closing-phase final roofline

**触发条件**: `state.use_roofline_composite is True` 且 `state.closing_phase = True` 且
`closing_deadline - now > 12 minutes`
**规则**: Coordinator 在 closing_phase 入口自动 enqueue 一次 `roofline` task (内部 dispatch,绕 PolicyGate)
**实现位置**: `coordinator.py::_enter_closing_phase` 末尾加 `_enqueue_internal_final_roofline_task`
**测试**: `test_f3_n31_final_roofline.py`
**回滚**: 由 `--use-roofline-composite=false` 自动屏蔽

### 5.5 Roofline-aware soft advisory (核心新增)

**触发条件**: `state.roofline_saturation_advisory is True`
**机制**:

1. **数据源**: F1 引入的 `roofline_snapshot.py::extract_workload_summary` 已经能解析
   `analysis.md` 的 Executive Summary,产出 `compute_pct` / `idle_pct` / `comm_pct`
2. **饱和判定**: 在 `coordinator.py` 加一个 `_compute_saturated_layers(state) -> list[str]` 方法
   - `compute_pct > 90%` AND 相关 kernel 已达 peak FLOPs > 85% → `compute` 饱和
   - `comm_pct > 25%` AND `comm_optimization` 已尝试 ≥ 2 次 → `comm` 饱和
   - `idle_pct > 30%` AND `params` (cuda graph / queue) 类 KEEP ≥ 2 → `host_overhead` 饱和
3. **写入 SharedState**: `state.roofline_saturation_history.append({snapshot_id, layer, saturation_pct, ts})`
4. **Prompt 注入** (3 处):
   - `specialist_prompt_builder.py` 的 section 2 (Bottleneck context) 加
     "saturated_layers=[X, Y]; consider whether your domain still has leverage"
   - `system_prompts/orchestration.md` 的 `=== Phase ===` block 加
     "saturated_layers (advisory)=[...]"
   - `explore` grid materialize 时,如果 grid 全部 variant 命中 `saturated_layers`,
     给 LLM 回一个 `policy_advice`(不是 `policy_denied`)
5. **per-domain 配置**: 每个 specialist domain 有自己的 `saturation_metric`
   - `serving_specialist` 关心 `comm_pct + idle_pct`(host overhead)
   - `kernel_switch_specialist` 关心 `compute_pct` + per-kernel `peak_pct`
   - `compiler_specialist` 关心 register pressure(从 GEAK 回写)
   - `system_specialist` 关心 launch latency
   - `comm_specialist` 关心 `exposed_comm_pct`

**实现位置**:
- `_analysis_keyword_map.py` (F0 已落地) 加 `SATURATION_KEYWORDS` 映射表
- `coordinator.py::_compute_saturated_layers()` 新增
- `specialist_prompt_builder.py` section 2 渲染加 saturated_layers
- `prompt_builder.py` `=== Phase ===` block 渲染加 saturated_layers
- 不引入 PolicyGate `delegate_against_saturated_layer` 拒绝规则(soft only)

**测试**: `test_f3_saturation_advisory.py`(覆盖 5 种饱和判定 + 3 种 prompt 注入点 + 2 种 per-domain 差异)

**回滚**: `--roofline-saturation-advisory=false`

### 5.6 F3 验收

- [ ] `pytest -q --tb=no | tail -3` ≥ 1646 + ~30 F3 新增 = ~1676 passed
- [ ] smoke: `--use-roofline-composite --gain-driven-kernel-opt --roofline-saturation-advisory --max-hours 1`
      跑通,report.py 渲染出 saturated_layers history
- [ ] 在 saturation 触发后,specialist prompt 里能 grep 到 `saturated_layers=`
- [ ] LLM 看到 saturated_layers 后,**仍然能 dispatch 该方向的 specialist**(soft only,无 policy_denied)
- [ ] `roofline_saturation_history` 在 state.json 里能持久化

---

## 6. 风险点 + 回滚开关

### 6.1 单点回滚

| 阶段 | 回滚命令 | 影响 |
|---|---|---|
| F0 任何步骤 | `git reset --hard pre-roofline-merge` | 完全回到本 branch 原状态 |
| F0 完成,F1 出问题 | `git revert <F1 commits>` | 保留 F0 scaffolding,F1 功能消失 |
| F1 上线异常 | `--use-roofline-composite=false` | 立即回到 profile + pmc_roofline 双 action |
| F2 上线异常 | `--framework-agent-enabled=false` | serving_specialist 不再调 fa |
| F3 N9 拒错 | `--deny-direct-profile=false` | LLM 又能直接 propose profile |
| F3 N19c 锁死 | `--gain-driven-kernel-opt=false` | kernel_opt 解锁回到原 sequence_denial |
| F3 advisory 误判 | `--roofline-saturation-advisory=false` | prompt 不再注入 saturated_layers |

### 6.2 Env var 全局回滚

每个开关都同时支持 env var,部署时不需要改 CLI:

```bash
export INFERENCE_OPTIMIZER_USE_ROOFLINE_COMPOSITE=0
export INFERENCE_OPTIMIZER_FRAMEWORK_AGENT_ENABLED=0
export INFERENCE_OPTIMIZER_DENY_DIRECT_PROFILE=0
export INFERENCE_OPTIMIZER_GAIN_DRIVEN_KERNEL_OPT=0
export INFERENCE_OPTIMIZER_ROOFLINE_SATURATION_ADVISORY=0
inference_optimizer optimize ...   # 行为完全等同 pre-roofline-merge
```

### 6.3 已识别风险点

| 风险 | 触发场景 | 缓解 |
|---|---|---|
| Saturation advisory 过早收敛 | roofline 信号噪声,误标 compute 饱和 | 默认 off,先观察 1-2 周再 default-on |
| serving_specialist 工具集过大 | 加了 fa 后 LLM 上下文膨胀 | per-domain 工具表,只 serving_specialist 见 fa |
| integrate_patch lane 锁竞争 | F2 后 specialist 产出 patch 增多 | TaskRegistry 已有 lane lease,无需新机制 |
| F1 翻默认值后 EXPLORE 入口太慢 | roofline 比 profile 多 ~30s | EXPLORE budget 默认 60% * total,影响 < 1% |
| KB 写入冲突 | F1 + F3 都往 KB 写 roofline 相关 entry | Critic 是唯一 KB 写者,本 branch 已有去重 |
| serving_specialist worktree 膨胀 | F2 cherry-pick 大 PR 占磁盘 | worktree 在 session_dir 下,session 结束自动清理 |
| Resume 兼容 | F0/F1/F3 加新字段后 resume 旧 manifest | 新字段全部 default_factory,Inv-10.1 自动满足 |


---

## 7. 必须保留 / 主动放弃 清单

详见 §2.5 (步骤 5) + §2.6 (步骤 6) + 单独的 `docs/integration/MUST_PRESERVE.md` /
`MAIN_FEATURES_DROPPED.md` (F0 阶段生成)。这里给一个一页式概览:

### 7.1 必须保留 (本 branch 资产)

```
v0.8 拓扑 ────┬── 物理删除 backends.py / params.py / validate_stack.py / scoring.py
            ├── phase_state.py 5 阶段状态机
            ├── specialist_runner.py + specialist_subprocess.py
            ├── 7 个 specialist domains (含 session_steward_specialist)
            ├── IR-4 specialist-first contract
            ├── IR-6 EXPLORE force-exit
            ├── IR-7 honest self-stop steward
            ├── PolicyGate action_deprecated 规则
            ├── PolicyGate explore_requires_specialist_provenance 规则
            ├── PolicyGate assess_remaining_gaps_throttle 规则
            ├── Cortex KB HTTP transport + IR-3 soft-degrade preflight
            └── 现有 1584 个测试全部 green
```

### 7.2 主动放弃 (main 资产)

```
v0.6 残骸 ────┬── action_executors/{backends,params,validate_stack}.py 的 main 改动
            ├── scoring.py 的 main 改动
            ├── framework_pr first-explore priority rule (orchestration.md)
            ├── backends_attempts < 1 / params_attempts < 1 sequence_denial
            ├── system_prompts/orchestration.md 的 Action scores top-12 块
            ├── system_prompts/orchestration.md 的"validate_stack 必须性"段
            ├── prompt_builder.py 全量 (+434/-45) — 仅取 analysis.md 注入段
            ├── --framework-gap / --framework-pr-discover CLI 旧接口
            ├── action_executors/framework_pr.py (独立 bandit arm)
            └── coordinator.py 全量 diff (+1179/-82) — 仅按需取
```

### 7.3 重新组织 (从 main 取,但按 v0.8 形态)

```
F1 ──┬── action_executors/roofline.py (改写: 删 v0.6 字段访问)
   ├── orchestrator/roofline_snapshot.py (F0 已 cherry-pick)
   ├── prompt_builder analysis.md 注入段
   └── orchestration.md analysis.md → action mapping (改写: backends/params → explore/specialist)

F2 ──┬── framework-agent/ 子目录 (F0 已 cherry-pick)
   ├── _framework_gap_composer.py (整 cherry-pick)
   └── serving_specialist 的工具白名单 + prompt 模板增强

F3 ──┬── N9 deny direct profile (PolicyGate 规则)
   ├── N19c gain-driven kernel_opt unlock
   ├── N31 closing-phase final roofline
   └── Roofline-aware soft advisory (新增,本 branch 自创)
```

---

## 8. 时间线 + 验收标准

### 8.1 推荐时间线 (顺序执行,不要并行)

| 阶段 | 工作日 | 完成节点 | 关键验收 |
|---|---|---|---|
| **F0** | 1 | tag `f0-done` | 1591+ tests green, 11 commits, 全部 docs 就位 |
| F1 准备 | 0.5 | branch `f1-roofline-composite` | F0 commits cherry-picked into PR branch |
| **F1 实施** | 2 | tag `f1-done` | 1621+ tests green, smoke session 跑通 |
| F2 准备 | 0.5 | branch `f2-framework-agent` | F1 commits + F0 framework-agent/ 已落地 |
| **F2 实施** | 2 | tag `f2-done` | 1646+ tests green, fa 真实跑通 1 个候选 |
| **F3 实施** | 1.5 | tag `f3-done` | 1676+ tests green, advisory 在 prompt 里可见 |
| **稳定期** | 1 | — | 默认开关全部 on 后跑 1 个 6h 真实 model session |

**总计**: ~7-8 个工作日(含 0.5 天的合并 + 验证 buffer)

### 8.2 全局验收(F3 完成后)

- [ ] `pytest inference_optimizer/tests` 1676+ passed
- [ ] 1 个完整 6h 真实 model session(GLM-5 / DeepSeek-R1 / R1-0528)跑通,
      report 显示首尾 roofline 对比,saturated_layers history 非空
- [ ] 默认所有 F1/F2/F3 开关 ON 时,IR-4/IR-6/IR-7 仍然全部生效
- [ ] 全部 5 个开关 OFF 时,行为完全等同 `pre-roofline-merge` tag
- [ ] origin/main 上 PR #288 / #280 关心的 N9 / N19c / N31 / framework_pr 都有等价覆盖
- [ ] `git log pre-roofline-merge..HEAD --oneline | wc -l` ≈ 30-40 commits(不是几百)
- [ ] PR 描述里给出 `MUST_PRESERVE.md` / `MAIN_FEATURES_DROPPED.md` 的 cross-ref

### 8.3 失败回退条件

如果在任意阶段满足下列**任何一条**,**立即停下来**:

1. `pytest` 数字下降(超过 5 个测试失败,且不是已知 pre-existing)
2. smoke session 出现 1584 marathon 中已经修复过的 regression
3. `git diff pre-roofline-merge..HEAD inference_optimizer/orchestrator/specialist_runner.py | wc -l > 50`
   (specialist runner 不应该被融合工作改动太多)
4. `git status` 出现"我以为应该被删但又出现了"的文件(scoring.py / backends.py / ...)
5. F2/F3 阶段出现 IR-4 / IR-7 行为偏离

**回退**:
- 单 PR 失败: `git revert` 该 PR 全部 commits,回到上一阶段 tag
- 多 PR 失败: `git reset --hard pre-roofline-merge`,然后**重做计划**

---

## 9. 立即可执行的下一步

由 LLM 助手立即可做(全部纯加性,低风险):

- [ ] **§2.1 步骤 1**: 整理 7 个根目录 .MD 进 `docs/notes_lossen_explore/`,生成 commit
- [ ] **§2.4 步骤 4**: 跑 dry-run merge,把 conflict map 归档进 `docs/integration/`
- [ ] **§2.5 步骤 5**: 生成 `docs/integration/MUST_PRESERVE.md`
- [ ] **§2.6 步骤 6**: 生成 `docs/integration/MAIN_FEATURES_DROPPED.md`
- [ ] **§2.8 步骤 8**: 修改 `shared_state.py` 加 `TraceAnalyzeSnapshot` + 3 个新字段 + 新测试
- [ ] **§2.9 步骤 9**: 修改 `phase_state.py` EXPLORE/KERNEL/CLOSE allowlist + 4 个新测试
- [ ] **§2.10 步骤 10**: 修改 `cli.py` 加 5 个 env 开关骨架 + 1 个新测试

由操作员人工执行(涉及远端 / 长任务 / 政策决定):

- [ ] **§2.2 步骤 2**: `git tag pre-roofline-merge` + `git push`
- [ ] **§2.3 步骤 3**: 跑完整 1584 测试集,确认绿色基线
- [ ] **§2.7 步骤 7**: cherry-pick `framework-agent/` (操作员先确认目录看起来正常)
- [ ] **§2.11 步骤 11**: 操作员 review 并采纳 SKILL.md 的 Integration Freeze 段措辞

---

## 附录 A — 命令速查

```bash
# 看当前分支差异
git fetch origin main
git log --oneline origin/main..HEAD | wc -l   # commits ahead
git log --oneline HEAD..origin/main | wc -l   # commits behind

# 找 fork point
git merge-base HEAD origin/main
# = e7c2bdb (PR #284 — Robustness Agent Update)

# Dry-run merge (不影响 working tree)
git checkout -b dry-run-temp
git merge --no-commit --no-ff origin/main
git status --short    # 看冲突文件
git merge --abort
git checkout feature/zhenggong/lossen_explore
git branch -D dry-run-temp

# 回到 pre-roofline-merge
git reset --hard pre-roofline-merge

# 跑测试基线
pytest inference_optimizer/tests -q --tb=no 2>&1 | tail -3

# 单独跑某一阶段的测试
pytest inference_optimizer/tests/test_f0_roofline_scaffolding.py -v   # F0
pytest inference_optimizer/tests/test_f1_*.py -v                       # F1
pytest inference_optimizer/tests/test_f2_*.py -v                       # F2
pytest inference_optimizer/tests/test_f3_*.py -v                       # F3
```

## 附录 B — 文件速查

| 文件 | 内容 |
|---|---|
| `docs/notes_lossen_explore/新增PR报告roofline_framework.MD` | PR #288 / #280 评估报告 (本文件的姊妹) |
| `docs/notes_lossen_explore/roofline_framework计划.MD` | 本计划 (F0 → F3) |
| `docs/integration/MUST_PRESERVE.md` | v0.8 必保留资产清单 (F0 step 5 生成) |
| `docs/integration/MAIN_FEATURES_DROPPED.md` | main 主动放弃清单 (F0 step 6 生成) |
| `docs/integration/dryrun_merge.log` | dry-run merge 输出 (F0 step 4) |
| `docs/integration/dryrun_status.log` | dry-run 后 git status (F0 step 4) |
| `docs/integration/dryrun_unmerged.log` | dry-run 冲突文件清单 (F0 step 4) |
| `docs/integration/dryrun_deleted.log` | dry-run modify/delete 文件清单 (F0 step 4) |
| `docs/test-baselines/pre_roofline_merge_*.log` | 1584 测试 baseline (F0 step 3) |

---

> **本计划生成时间**: 2026-05-24
> **基于**: `feature/zhenggong/lossen_explore` HEAD = a1d2730
>          (PR-A11 / Saturday-May-2026 marathon hotfixes 之后)
> **目标 main commit**: c6f0a71 (PR #288 merge into main)
> **作者**: 协助 zhenggong 的 Cursor agent
