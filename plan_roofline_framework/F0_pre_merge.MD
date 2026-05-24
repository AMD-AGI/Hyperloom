# F0 — Pre-merge 准备

> 工作日: 1 天
> 风险: 极低 (全部加性 + 文档)
> 依赖: 无
> 完成节点: tag `f0-done` + ~11 个 commit
>
> **完成 F0 之后,本 branch 实际行为完全等同当前 HEAD;F0 的全部工作是为 F1/F2/F3 铺路。**

## 0. 环境准备

```bash
export FEAT_ROOT="/wekafs/zgong/Hyperloom-KB"
export MAIN_ROOT="/wekafs/zgong/Hyperloom"
export FEAT_BRANCH="feature/zhenggong/lossen_explore"
export ROLLBACK_TAG="pre-roofline-merge"

cd "$FEAT_ROOT"
git status --short                    # 应该看到 6 个未跟踪 .MD
git branch -vv | grep "$FEAT_BRANCH"  # 应该看到 [origin/feature/zhenggong/lossen_explore]
git log --oneline HEAD -1             # 应该是 a1d2730 (Saturday-May-2026 marathon hotfixes)

cd "$MAIN_ROOT"
git branch --show-current             # 应该是 main
git log --oneline HEAD -1             # 应该是 c6f0a71 (PR #288 merge)
```

如果上面任何一项不符,**停下,不要继续**。

---

## 1. 步骤清单

```
F0-1   归档 7 个根目录 .MD                  -> 1 commit  (LLM)
F0-2   打 tag pre-roofline-merge            -> 0 commit  (操作员)
F0-3   锁定测试基线 1584 passed             -> 1 commit  (操作员 + LLM)
F0-4   dry-run merge 归档冲突清单           -> 1 commit  (LLM)
F0-5   生成 MUST_PRESERVE.md                -> 1 commit  (LLM)
F0-6   生成 MAIN_FEATURES_DROPPED.md        -> 1 commit  (LLM)
F0-7   cherry-pick 加性 scaffolding         -> 1 commit  (操作员 + LLM)
F0-8   SharedState 加 roofline 占位字段     -> 1 commit  (LLM)
F0-9   phase_state.py allowlist 加 roofline -> 1 commit  (LLM)
F0-10  cli.py 加 5 个 env 开关骨架          -> 1 commit  (LLM)
F0-11  SKILL.md 加 Integration Freeze 段    -> 1 commit  (操作员)
F0-12  打 tag f0-done                       -> 0 commit  (操作员)
```

每步完成后跑 `pytest inference_optimizer/tests -q --tb=no | tail -3`,数字必须不降。

---

## 2. F0-1: 归档 7 个根目录 .MD

### 现状
```
?? Agent-deligate-gap.MD
?? Version1_report.MD
?? cortex-kb-http-branch-b-2026-05-20.md           # 之前 commit 过,现在又出现?核对一下
?? roofline_framework计划.MD
?? 优化报告_DeepSeek-R1-0528_MI300X_SGLang.md
?? 优化报告_DeepSeek-R1-0528_MI300X_vLLM.md
?? 新增PR报告roofline_framework.MD
```

### 命令

```bash
cd "$FEAT_ROOT"

mkdir -p docs/notes_lossen_explore

# 用 git mv 比 mv 安全(如果有同名追踪文件会报错而不是覆盖)
mv Agent-deligate-gap.MD                    docs/notes_lossen_explore/
mv Version1_report.MD                       docs/notes_lossen_explore/
mv "优化报告_DeepSeek-R1-0528_MI300X_SGLang.md" docs/notes_lossen_explore/
mv "优化报告_DeepSeek-R1-0528_MI300X_vLLM.md"  docs/notes_lossen_explore/
mv "新增PR报告roofline_framework.MD"          docs/notes_lossen_explore/
mv "roofline_framework计划.MD"                docs/notes_lossen_explore/

# cortex-kb-http-branch-b-2026-05-20.md 单独处理:可能已经 commit 过
git ls-files | grep -i cortex-kb-http
# 如果 git ls-files 显示已经跟踪,就 git rm 旧路径再 mv;
# 如果没显示,直接 mv:
mv cortex-kb-http-branch-b-2026-05-20.md docs/notes_lossen_explore/ 2>/dev/null || true

git add docs/notes_lossen_explore/
git status --short
```

### Commit

```bash
git commit -m "docs: archive lossen_explore session notes + integration plan

Move 6-7 root-level .MD files into docs/notes_lossen_explore/ so that
F0+ integration work has a clean root and the notes are preserved
under git history.

Files moved:
- Agent-deligate-gap.MD
- Version1_report.MD
- cortex-kb-http-branch-b-2026-05-20.md  (if previously untracked)
- roofline_framework计划.MD
- 优化报告_DeepSeek-R1-0528_MI300X_SGLang.md
- 优化报告_DeepSeek-R1-0528_MI300X_vLLM.md
- 新增PR报告roofline_framework.MD"
```

### 验收

```bash
git status --short                              # 不再列出 ?? .MD
ls docs/notes_lossen_explore/ | wc -l           # >=6
pytest inference_optimizer/tests -q --tb=no 2>&1 | tail -3   # 数字不变
```

---

## 3. F0-2: 打 tag pre-roofline-merge

### 命令 (操作员人工)

```bash
cd "$FEAT_ROOT"

git tag -a pre-roofline-merge -m "Last commit on lossen_explore before main integration

State at this tag:
- 51 commits ahead of fork point e7c2bdb (PR #284 baseline)
- 134 commits behind origin/main (incl. PR #288 + #280 not yet integrated)
- v0.8 action graph (explore + specialist), no v0.6 backends/params/validate_stack
- IR-1..IR-7 iron rules, 7 specialist domains
- 1584 passing tests, 1 skipped, 4 pre-existing unrelated failures

Rollback any F0-F3 step with:  git reset --hard pre-roofline-merge"

git push origin pre-roofline-merge
```

### 验收

```bash
git tag -l pre-roofline-merge       # 显示 tag 名
git ls-remote --tags origin pre-roofline-merge   # 远端也有
```

---

## 4. F0-3: 锁定测试基线

### 命令 (操作员触发,LLM 归档)

```bash
cd "$FEAT_ROOT"

# 跑全量测试 (可能 10-20 分钟)
pytest inference_optimizer/tests -q --tb=no 2>&1 | tee /tmp/baseline_$(date +%Y%m%d).log

# 期望末尾形如:
# "1584 passed, 1 skipped, 4 failed in NNN.NNs"
# 4 failed 是 PR-A11 commit 已知的 pre-existing,可以忽略;但 4 个是哪些必须列出来。

tail -20 /tmp/baseline_$(date +%Y%m%d).log
grep -E "FAILED " /tmp/baseline_$(date +%Y%m%d).log > /tmp/known_failed_$(date +%Y%m%d).txt
cat /tmp/known_failed_$(date +%Y%m%d).txt

# 归档进 git
mkdir -p docs/test-baselines
cp /tmp/baseline_$(date +%Y%m%d).log    docs/test-baselines/pre_roofline_merge_$(date +%Y%m%d).log
cp /tmp/known_failed_$(date +%Y%m%d).txt docs/test-baselines/known_failed_$(date +%Y%m%d).txt

git add docs/test-baselines/
git commit -m "chore(tests): archive pre-merge regression baseline ($(date +%F))

Baseline: 1584 passed, 1 skipped, 4 pre-existing failures.
Pre-existing failures (must NOT count against F1/F2/F3 verdicts):
$(cat docs/test-baselines/known_failed_$(date +%Y%m%d).txt | sed 's/^/  /')

Each F1/F2/F3 step must produce a regression sweep with:
- pass count >= 1584
- failed set ⊂ known_failed list (no NEW failures)"
```

### 红线

如果 baseline 不是 1584 passed (差距 >5),**停下来修绿,不要继续 F0-4+**。
红色基线上的合并工作无法判断"哪些失败是合并引起的"。

---

## 5. F0-4: Dry-run merge 归档冲突清单

### 命令 (LLM 可做)

```bash
cd "$FEAT_ROOT"

git fetch origin main

# 临时分支跑 merge,只看冲突,不真合
DRY_BRANCH="dry-run-merge-$(date +%Y%m%d)"
git checkout -b "$DRY_BRANCH"

git merge --no-commit --no-ff origin/main 2>&1 | tee /tmp/dryrun_merge.log
git status --short                     > /tmp/dryrun_status.log
git diff --name-only --diff-filter=U   > /tmp/dryrun_unmerged.log
git diff --name-only --diff-filter=D   > /tmp/dryrun_deleted.log

# 立即放弃
git merge --abort
git checkout "$FEAT_BRANCH"
git branch -D "$DRY_BRANCH"

# 归档
mkdir -p docs/integration
cp /tmp/dryrun_*.log docs/integration/

git add docs/integration/dryrun_*.log
git commit -m "chore(integration): archive dry-run merge conflict map ($(date +%F))

Dry-run merge of origin/main into $FEAT_BRANCH produced:
- $(wc -l < docs/integration/dryrun_unmerged.log) unmerged files
- $(wc -l < docs/integration/dryrun_deleted.log) modify/delete conflicts

Used by F1/F2/F3 cherry-pick steps to verify each conflict resolved
matches the original conflict map (no NEW conflicts introduced)."
```

### 已知输出

(基于先前 dry-run, 参见 docs/notes_lossen_explore/新增PR报告roofline_framework.MD §2.1)

```
14 content conflicts: cli.py, action_executors/__init__.py, report.py,
                      coordinator.py, shared_state.py, sub_agent_runner.py,
                      system_prompts/{critic,orchestration}.md,
                      system_prompts/prompt_builder.py, paths.py, session_paths.py,
                      tests/test_p0_1_protocol.py, tests/test_p1_2_full_action_catalogue.py,
                      tests/test_p5_decision.py, tests/test_prompt_assets.py,
                      tests/test_required_step_gates.py

7 modify/delete: action_executors/{backends,params,validate_stack}.py,
                 scoring.py,
                 tests/test_p3_search_space_expansion.py,
                 tests/test_validate_stack.py,
                 tests/test_validate_stack_gate_skip.py
```


---

## 6. F0-5: 生成 docs/integration/MUST_PRESERVE.md

文件名: `docs/integration/MUST_PRESERVE.md`

### 内容

```markdown
# v0.8 资产保留清单

> 本 branch (lossen_explore) 在 fork 点 e7c2bdb 之后做的 51 个 commit 的关键产物。
> 任何 F1/F2/F3 cherry-pick 操作都必须确保下表所有项**未被破坏**。
> 验收脚本: docs/integration/check_preserved.sh

## 物理删除文件 (必须保持删除)

| 文件 | 来源 commit | 验证 |
|---|---|---|
| `inference_optimizer/orchestrator/action_executors/backends.py` | 76389d8 | `! test -f $FEAT_ROOT/inference_optimizer/orchestrator/action_executors/backends.py` |
| `inference_optimizer/orchestrator/action_executors/params.py` | 76389d8 | `! test -f $FEAT_ROOT/inference_optimizer/orchestrator/action_executors/params.py` |
| `inference_optimizer/orchestrator/action_executors/validate_stack.py` | 76389d8 | `! test -f $FEAT_ROOT/inference_optimizer/orchestrator/action_executors/validate_stack.py` |
| `inference_optimizer/orchestrator/scoring.py` | a86bdf2 | `! test -f $FEAT_ROOT/inference_optimizer/orchestrator/scoring.py` |
| `inference_optimizer/actions/_meta/{backends,params,validate_stack}.yaml` | 76389d8 | 三个文件不存在 |
| `inference_optimizer/actions/validate_stack.md` | 76389d8 | 不存在 |
| `inference_optimizer/tests/test_backends_search.py` | 76389d8 | 不存在 |
| `inference_optimizer/tests/test_params_search.py` | 76389d8 | 不存在 |
| `inference_optimizer/tests/test_p2_3_param_executors.py` | 76389d8 | 不存在 |
| `inference_optimizer/tests/test_validate_stack.py` | 76389d8 | 不存在 |
| `inference_optimizer/tests/test_p2_5_grid_promotion.py` | 76389d8 | 不存在 |
| `inference_optimizer/tests/test_p3_search_space_expansion.py` | 76389d8 | 不存在 |

## 必须存在的新文件 (本 branch 独有)

| 文件 | 来源 commit | 验证 |
|---|---|---|
| `inference_optimizer/orchestrator/phase_state.py` | c02c90a | `wc -l` ≥ 1300 |
| `inference_optimizer/orchestrator/specialist_runner.py` | 6b20bf8 | 含 `class SpecialistRunner:` |
| `inference_optimizer/orchestrator/specialist_subprocess.py` | PR-A2 | 含 `class SpecialistSubprocessDispatcher:` |
| `inference_optimizer/orchestrator/specialist_domains.py` | 6b20bf8 | 含 7 个 SpecialistDomain |
| `inference_optimizer/orchestrator/system_prompts/specialist_prompt_builder.py` | 6b20bf8 | 存在 |
| `inference_optimizer/orchestrator/system_prompts/critic_prompt_builder.py` | (后续) | 存在 |

## 必须存在的 Iron Rules (SKILL.md 段落)

- IR-1 GPU MUST be unoccupied before every launch
- IR-2 install.sh MUST succeed before every launch
- IR-3 KB + PR Monitor reachability (in-loop, soft degrade)
- IR-4 EXPLORE is specialist-first (PR-A9 Arbor-into-Hyperloom)
- IR-6 EXPLORE force-exit on low budget (e974567)
- IR-7 session_steward_specialist (Honest self-stop) (e974567)

验证: `grep -E "^### IR-[1-7]" inference_optimizer/SKILL.md | wc -l` 应 >= 6

## 必须存在的 PolicyGate 规则

| 规则 ID | 文件 | 来源 |
|---|---|---|
| `action_deprecated` | `policy.py` | KB_gaps/Gap-10 |
| `explore_requires_specialist_provenance` | `policy.py` | PR-A9 |
| `assess_remaining_gaps_throttle` | `policy.py` | e974567 (IR-7) |

验证: `grep -E "rule=['\"]?(action_deprecated|explore_requires_specialist_provenance|assess_remaining_gaps_throttle)" inference_optimizer/orchestrator/policy.py | wc -l` 应 >= 3

## 必须存在的 CLI flags

`python -m inference_optimizer.cli optimize --help | grep -E -- "--explore-force-exit-hours-remaining|--explore-force-exit-budget-pct|--steward-disabled|--no-kernel|--no-framework|--robustness-agent|--critic-agent" | wc -l` 应 >= 7

## 必须存在的 SharedState 字段

```python
# inference_optimizer/orchestrator/shared_state.py 必须含:
last_remaining_gaps_assessment: dict | None
remaining_gaps_assessments: list[dict]   # cap 10
steward_continuation_used: bool
specialist_domain_empty_streak: dict[str, int]
```

验证: `grep -E "(last_remaining_gaps_assessment|remaining_gaps_assessments|steward_continuation_used|specialist_domain_empty_streak)" inference_optimizer/orchestrator/shared_state.py | wc -l` 应 >= 4

## 必须存在的测试文件

| 文件 | 测试数 | 来源 |
|---|---|---|
| `inference_optimizer/tests/test_phase_force_exit.py` | 10 | e974567 |
| `inference_optimizer/tests/test_assess_remaining_gaps.py` | 24 | e974567 |
| `inference_optimizer/tests/test_specialist_runner.py` | (M5) | 6b20bf8 |
| `inference_optimizer/tests/test_specialist_subprocess.py` | (PR-A2) | PR-A2 |
| `inference_optimizer/tests/test_phase_state.py` | (M2) | c02c90a |

## Cortex KB 相关

- HTTP transport: d6036a2
- IR-3 soft-degrade preflight: d6036a2
- 验证: `grep "preflight_kb.sh" inference_optimizer/scripts/install.sh` 有命中
```

### 配套验收脚本 (可选,F0-5 的"赠品")

写入 `docs/integration/check_preserved.sh`:

```bash
#!/usr/bin/env bash
# F0/F1/F2/F3 每个 commit 后跑一遍,任何一项失败立即停下
set -euo pipefail
FEAT_ROOT="${FEAT_ROOT:-/wekafs/zgong/Hyperloom-KB}"
cd "$FEAT_ROOT"

echo "=== Deleted files (must stay deleted) ==="
for f in \
  inference_optimizer/orchestrator/action_executors/backends.py \
  inference_optimizer/orchestrator/action_executors/params.py \
  inference_optimizer/orchestrator/action_executors/validate_stack.py \
  inference_optimizer/orchestrator/scoring.py \
  inference_optimizer/actions/_meta/backends.yaml \
  inference_optimizer/actions/_meta/params.yaml \
  inference_optimizer/actions/_meta/validate_stack.yaml \
  inference_optimizer/actions/validate_stack.md \
  inference_optimizer/tests/test_backends_search.py \
  inference_optimizer/tests/test_params_search.py \
  inference_optimizer/tests/test_p2_3_param_executors.py \
  inference_optimizer/tests/test_validate_stack.py \
  inference_optimizer/tests/test_p2_5_grid_promotion.py \
  inference_optimizer/tests/test_p3_search_space_expansion.py; do
  if [[ -f "$f" ]]; then echo "FAIL: $f re-appeared"; exit 1; fi
done
echo "  OK ($(wc -l <<< $f) files confirmed absent)"

echo "=== Required files (must exist) ==="
for f in \
  inference_optimizer/orchestrator/phase_state.py \
  inference_optimizer/orchestrator/specialist_runner.py \
  inference_optimizer/orchestrator/specialist_subprocess.py \
  inference_optimizer/orchestrator/specialist_domains.py; do
  if [[ ! -f "$f" ]]; then echo "FAIL: $f missing"; exit 1; fi
done
echo "  OK"

echo "=== Iron rules in SKILL.md ==="
N=$(grep -cE "^### IR-[1-7]" inference_optimizer/SKILL.md || echo 0)
[[ "$N" -ge 6 ]] || { echo "FAIL: only $N IR rules found, expected >= 6"; exit 1; }
echo "  OK ($N IR rules)"

echo "=== PolicyGate rules ==="
N=$(grep -cE "rule=['\"]?(action_deprecated|explore_requires_specialist_provenance|assess_remaining_gaps_throttle)" inference_optimizer/orchestrator/policy.py || echo 0)
[[ "$N" -ge 3 ]] || { echo "FAIL: only $N policy rules found, expected >= 3"; exit 1; }
echo "  OK ($N policy rules)"

echo "=== ALL PRESERVED OK ==="
```

```bash
chmod +x docs/integration/check_preserved.sh
git add docs/integration/MUST_PRESERVE.md docs/integration/check_preserved.sh
git commit -m "docs(integration): MUST_PRESERVE.md + check_preserved.sh

Inventory of v0.8 assets that must survive integration. Run
docs/integration/check_preserved.sh after every cherry-pick to verify
no v0.8 assets got accidentally clobbered."
```

### 验收

```bash
bash docs/integration/check_preserved.sh    # rc=0
```


---

## 7. F0-6: 生成 docs/integration/MAIN_FEATURES_DROPPED.md

文件名: `docs/integration/MAIN_FEATURES_DROPPED.md`

### 内容

```markdown
# Main 主动放弃清单

> 本 branch 主动选择**不要** main (PR #288 / #280) 的哪些产物。
> F1/F2/F3 解冲突时直接选"我们方";PR 描述里引用此清单解释为什么不合某段。

## 类别 1: v0.6 残骸 (与 KB_design §3.4 / §3.9 冲突)

| 主动放弃 | 来源 PR | 不要的理由 | 替代方案 |
|---|---|---|---|
| `action_executors/backends.py` (main 改动) | #288 (+59 行) | v0.6 已退役 | F1: 功能合并进 explore |
| `action_executors/params.py` (main 改动) | #288 (+57 行) | v0.6 已退役 | F1: 同上 |
| `action_executors/validate_stack.py` (main 改动) | #288 (+38 行) | v0.6 已退役 | M3 KEEP 后 explore 自动 rebench |
| `scoring.py` (main 改动) | #288 (+136 行) | scoreboard 退场 (KB_design §3.9) | KB priors 在 Critic 侧 |
| `actions/_meta/{backends,params,validate_stack}.yaml` | #288 | 同上 | F1: 取 roofline.yaml 即可 |
| `actions/validate_stack.md` (main 改动) | #288 | 同上 | M3 内联 |
| 6 个 test_* (test_backends_search.py 等) | #288 | 测试已删除 | F1: 用 test_explore_*.py 覆盖 |

## 类别 2: v0.6 sequence_denial (会永久锁死 v0.8)

| 主动放弃 | 行为 | F3 替代 |
|---|---|---|
| `backends_attempts < 1` sequence_denial | v0.8 没有 backends,字段无写者,denial 永真 → kernel_opt 永久锁死 | F3: 重写为 `explore_attempts < 1` 等价规则 |
| `params_attempts < 1` sequence_denial | 同上 | F3: 同上 |
| `validate_stack_attempts < 1` sequence_denial | 同上 | (无需替代,M3 已内联) |

## 类别 3: 与 IR-4 specialist-first 冲突的 prompt 规则

| 主动放弃 | 来源 | 替代 |
|---|---|---|
| `framework_pr first-explore priority rule` | #280 orchestration.md | F2: 把 fa 嵌进 serving_specialist;IR-4 自动满足 |
| `--framework-gap` / `--framework-pr-discover` CLI | #280 cli.py | F2: 走 specialist 内部调 fa,不暴露给操作员 |
| `action_executors/framework_pr.py` (独立 bandit arm) | #280 | F2: 不引入独立 action |

## 类别 4: 已退场的 LLM 决策框架

| 主动放弃 | 来源 | 替代 |
|---|---|---|
| `system_prompts/orchestration.md` 的 `Action scores top-12 block` 整段 | (历史) | 已用 phase + gaps + KB priors 决策 (KB_design §3.9 Inv-9.1) |
| `system_prompts/orchestration.md` 的 `**\`validate_stack\` is mandatory**` 段 | (历史) | M3 已内联进 explore |
| 提到 `score_violation` 的所有 prompt 段落 | (历史) | scoreboard 退场后无意义 |

## 类别 5: 部分采纳

| 文件 | main 改动量 | 我们要的部分 | 我们不要的部分 |
|---|---|---|---|
| `system_prompts/prompt_builder.py` | +434 / -45 | analysis.md 全文注入逻辑 (`_format_analysis_md_full` + section 5.5) | v0.6 backends/params 字段渲染 |
| `system_prompts/orchestration.md` | +173 / -5 | 🔴/🟡/🟢 标记说明 + Roofline-v2 analysis.md → action mapping | Action scores block / framework_pr first-explore / score_violation |
| `coordinator.py` | +1179 / -82 | record_partial / next_pending_keep_kernel_id / untried_hot_reusable_kernels gate | 280 行的 v0.6 backends/params/validate_stack 路径 |
| `cli.py` | +306 / -79 | `--no-framework` toggle (改造形态) / `--use-roofline-composite` (新加) | `--framework-gap` / `--framework-pr-discover` |

## 类别 6: 完全采纳但形态调整

| 来源 | main 形态 | 本 branch 形态 (调整后) |
|---|---|---|
| `roofline` 复合 action | 一等公民 action | F1: 同样一等公民,但走 explore-merged 路径 |
| `roofline_snapshot.py` parser | F0-7 cherry-pick | 不动,已无 v0.6 依赖 |
| `_analysis_keyword_map.py` | F0-7 cherry-pick | 不动 |
| `framework-agent/` 子目录 | F0-7 cherry-pick | F2: 嵌入 serving_specialist 工具链 |
| N9 deny direct profile | PolicyGate 规则 | F3: 同样实现 |
| N19c gain-driven kernel_opt | PolicyGate 规则 | F3: 同样实现 |
| N31 final roofline | Coordinator 内部 enqueue | F3: 同样实现 |
```

### Commit

```bash
git add docs/integration/MAIN_FEATURES_DROPPED.md
git commit -m "docs(integration): MAIN_FEATURES_DROPPED.md

Inventory of main features intentionally NOT integrated, with replacement
strategies for each. F1/F2/F3 cherry-pick steps reference this list to
justify selecting 'ours' on every conflict involving these surfaces."
```

---

## 8. F0-7: Cherry-pick 加性 Scaffolding

> **关键步骤**: 这一步把 main 上的纯加性文件搬过来,让 F1/F2 实际工作时引用面已经齐了。
> 用本地 main checkout (`$MAIN_ROOT`) 作为来源。

### 命令

```bash
cd "$FEAT_ROOT"

# (a) PR #280 整个 framework-agent/ 子目录 (顶层新目录,本 branch 无对应)
echo "=== Step a: framework-agent/ 整体 cp ==="
cp -r "$MAIN_ROOT/framework-agent" .
ls framework-agent/                         # 应该看到 SKILL.md, src/, tests/, kb/, ...

# (b) PR #288 文档类文件
echo "=== Step b: roofline action docs ==="
cp "$MAIN_ROOT/inference_optimizer/actions/roofline.md"            inference_optimizer/actions/roofline.md
cp "$MAIN_ROOT/inference_optimizer/actions/_meta/roofline.yaml"    inference_optimizer/actions/_meta/roofline.yaml

# (c) PR #288 纯 parser (无 v0.6 依赖)
echo "=== Step c: roofline_snapshot.py + _analysis_keyword_map.py ==="
cp "$MAIN_ROOT/inference_optimizer/orchestrator/roofline_snapshot.py"      inference_optimizer/orchestrator/roofline_snapshot.py
cp "$MAIN_ROOT/inference_optimizer/orchestrator/_analysis_keyword_map.py"  inference_optimizer/orchestrator/_analysis_keyword_map.py

# (d) install.sh 的 framework-agent 安装步
# 不要直接覆盖,先 diff 一下 (可能含与 IR-3 cortex-kb 冲突的段):
echo "=== Step d: install.sh 差异检查 ==="
diff "$MAIN_ROOT/inference_optimizer/scripts/install.sh" \
     "$FEAT_ROOT/inference_optimizer/scripts/install.sh" | head -100

# 如果差异主要是末尾追加 ensure_framework_agent 函数,直接 patch 进去:
# 临时方案: 等 F2 阶段再处理 install.sh,F0 阶段不动
echo "  ⚠ install.sh 暂不动,F2 阶段集中处理"
```

### 验证 cherry-pick 没破坏现有运行路径

```bash
echo "=== Verify imports OK ==="
python -c "from inference_optimizer.orchestrator.roofline_snapshot import extract_workload_summary; print(extract_workload_summary.__module__)"
python -c "from inference_optimizer.orchestrator._analysis_keyword_map import *"
python -c "import sys; sys.path.insert(0, 'framework-agent/src'); import framework_agent; print(framework_agent.__file__)" 2>&1 | head -5
```

### 验证测试基线没变

```bash
pytest inference_optimizer/tests -q --tb=no 2>&1 | tail -3
# 期望 1584 passed (新增的文件没有改任何运行路径)
```

### Commit

```bash
git add framework-agent/
git add inference_optimizer/actions/roofline.md
git add inference_optimizer/actions/_meta/roofline.yaml
git add inference_optimizer/orchestrator/roofline_snapshot.py
git add inference_optimizer/orchestrator/_analysis_keyword_map.py

git commit -m "feat(F0-7): import additive scaffolding from main

Cherry-picked from /wekafs/zgong/Hyperloom (main @ c6f0a71):

- framework-agent/ (entire new top-level directory from PR #280)
- inference_optimizer/orchestrator/roofline_snapshot.py (parser, no deps)
- inference_optimizer/orchestrator/_analysis_keyword_map.py (data dict)
- inference_optimizer/actions/roofline.md + _meta/roofline.yaml (docs)

These files have no v0.6 dependencies and no callers on this branch yet.
F1 will wire roofline_snapshot.py into the actual roofline executor;
F2 will wire framework-agent/ into serving_specialist's tool chain.

install.sh is intentionally NOT touched in F0; F2 handles framework-agent
install hooks in one shot to avoid IR-3 cortex-kb interaction.

Verified: pytest inference_optimizer/tests -q == 1584 passed (unchanged)."
```

### 注意事项

- **不要**这一步 cherry-pick `action_executors/roofline.py` (引用 v0.6 backends 路径,F1 时改写)
- **不要**这一步 cherry-pick `action_executors/framework_pr.py` (F2 时直接放弃)
- **不要**这一步 cherry-pick `cli.py` / `coordinator.py` / `shared_state.py` 任何改动
- **不要**这一步动 `policy.py` / `phase_state.py` (F0-9 才动 phase_state.py)

### 失败回滚

```bash
git reset --hard HEAD~1     # 回到 F0-6
rm -rf framework-agent/
rm inference_optimizer/actions/roofline.md
rm inference_optimizer/actions/_meta/roofline.yaml
rm inference_optimizer/orchestrator/roofline_snapshot.py
rm inference_optimizer/orchestrator/_analysis_keyword_map.py
```


---

## 9. F0-8: SharedState 加 Roofline 占位字段

> 纯加性 schema 改动,F1 落 RooflineExecutor 时只需写值,不需要改 schema。

### 编辑 `inference_optimizer/orchestrator/shared_state.py`

#### 找到 SharedState dataclass 定义

```bash
grep -n "^class SharedState" inference_optimizer/orchestrator/shared_state.py
grep -n "^@dataclass" inference_optimizer/orchestrator/shared_state.py | head -10
```

#### 添加 TraceAnalyzeSnapshot dataclass

在文件中合适位置(SharedState 类定义**之前**,但和其他 @dataclass 同级):

```python
@dataclass
class TraceAnalyzeSnapshot:
    """F0 placeholder for roofline-v2 (PR #288) trace_analyze cache.

    F1 will populate this from action_executors/roofline.py;
    until then it stays at default values (snapshot_id=0 = no snapshot yet).

    Fields mirror the canonical shape used by main's RooflineExecutor:
    * snapshot_id — monotonic, incremented per successful roofline task
    * analysis_md_path — absolute path to TraceLens analysis.md on disk
    * analysis_md_text — verbatim text injected into orchestration prompt
    * candidates_path — kernel-candidates JSON path
    * timestamp_unix — when this snapshot was produced
    """
    snapshot_id: int = 0
    analysis_md_path: str = ""
    analysis_md_text: str = ""
    candidates_path: str = ""
    timestamp_unix: float = 0.0
```

#### 在 SharedState 内添加新字段 (放在 last_profile_trace 附近)

```python
@dataclass
class SharedState:
    # ... existing fields ...

    last_profile_trace: str = ""
    last_profile_status: str = ""
    last_profile_args: dict = field(default_factory=dict)

    # F0 placeholder fields for PR #288 (roofline-v2). F1 wires writers.
    last_trace_analyze: TraceAnalyzeSnapshot = field(default_factory=TraceAnalyzeSnapshot)
    roofline_snapshot_id: int = 0   # monotonic, incremented in F1's RooflineExecutor
    roofline_saturation_history: list[dict] = field(default_factory=list)   # F3 populates

    # ... rest of existing fields ...
```

#### 创建测试文件 `inference_optimizer/tests/test_f0_roofline_scaffolding.py`

```python
"""F0 — Verify roofline placeholder fields are present and default-empty.

These tests guarantee F0-8 scaffolding is in place. F1 will add behavior tests."""

from __future__ import annotations

import pytest


def test_trace_analyze_snapshot_default():
    from inference_optimizer.orchestrator.shared_state import TraceAnalyzeSnapshot
    snap = TraceAnalyzeSnapshot()
    assert snap.snapshot_id == 0
    assert snap.analysis_md_path == ""
    assert snap.analysis_md_text == ""
    assert snap.candidates_path == ""
    assert snap.timestamp_unix == 0.0


def test_shared_state_has_roofline_placeholder_fields():
    from inference_optimizer.orchestrator.shared_state import (
        SharedState, TraceAnalyzeSnapshot,
    )
    # Use the project's preferred SharedState constructor / factory
    s = SharedState()  # NOTE: adjust if SharedState requires args
    assert hasattr(s, "last_trace_analyze")
    assert isinstance(s.last_trace_analyze, TraceAnalyzeSnapshot)
    assert s.last_trace_analyze.snapshot_id == 0

    assert hasattr(s, "roofline_snapshot_id")
    assert s.roofline_snapshot_id == 0

    assert hasattr(s, "roofline_saturation_history")
    assert s.roofline_saturation_history == []


def test_roofline_fields_serialise_round_trip():
    """SharedState must persist + reload without losing the new fields.

    This catches the case where someone forgets to add the new fields to
    the manifest writer / reader path."""
    from inference_optimizer.orchestrator.shared_state import (
        SharedState, TraceAnalyzeSnapshot,
    )
    import json

    s = SharedState()
    s.last_trace_analyze = TraceAnalyzeSnapshot(
        snapshot_id=3, analysis_md_text="hello", timestamp_unix=12345.6,
    )
    s.roofline_snapshot_id = 3
    s.roofline_saturation_history.append({"snapshot_id": 3, "layer": "compute", "saturation_pct": 92})

    # If SharedState has a to_dict / from_dict pair, exercise it:
    if hasattr(s, "to_dict") and hasattr(SharedState, "from_dict"):
        blob = s.to_dict()
        # Make sure the round-trip carries our fields
        assert "last_trace_analyze" in blob or any("trace_analyze" in str(k) for k in blob)
        assert blob.get("roofline_snapshot_id", -1) == 3
        # NOTE: from_dict may not be perfectly symmetric; this is a soft check.
```

### 命令

```bash
# 运行新测试
pytest inference_optimizer/tests/test_f0_roofline_scaffolding.py -v

# 全量
pytest inference_optimizer/tests -q --tb=no 2>&1 | tail -3
# 期望 1587 passed (1584 + 3 new)

git add inference_optimizer/orchestrator/shared_state.py
git add inference_optimizer/tests/test_f0_roofline_scaffolding.py
git commit -m "feat(F0-8): pre-add roofline placeholder fields on SharedState

Adds TraceAnalyzeSnapshot dataclass + 3 new fields on SharedState (default
empty). F1's RooflineExecutor will populate them; until then this is a
pure schema addition, all existing tests unchanged.

New fields:
- last_trace_analyze: TraceAnalyzeSnapshot (default snapshot_id=0)
- roofline_snapshot_id: int (default 0)
- roofline_saturation_history: list[dict] (default empty)

Tests: 3 new (default values + serialise round-trip).
Total: 1587 passed (1584 + 3)."
```

### 注意事项

1. **必须** default_factory,不能用裸的 `[]` / `{}`(dataclass 限制)
2. **不要**改 SharedState 的 `__init__` 签名,只加字段
3. **如果**项目有 manifest writer (`inference_optimizer/manifest.py`),检查是否需要把新字段加进去
4. **如果**项目有 SQLite WAL 路径 (`inference_optimizer/storage/`),新字段一般作为 JSON blob 存在已有列里,不需要 migration


---

## 10. F0-9: phase_state.py allowlist 加 roofline 占位

### 编辑 `inference_optimizer/orchestrator/phase_state.py`

找到 `PHASE_ALLOWED_ACTIONS` 并修改:

```python
PHASE_ALLOWED_ACTIONS: dict[str, frozenset[str]] = {
    PHASE_PRELUDE: frozenset({
        "target_analysis", "baseline", "recover",
    }),
    PHASE_EXPLORE: frozenset({
        "explore", "specialist",
        "integrate_patch",
        "assess_remaining_gaps",
        # F0-9: 'roofline' placeholder. F1's RooflineExecutor takes effect
        # once --use-roofline-composite=True; until then 'profile' /
        # 'pmc_roofline' remain available for backwards compat.
        "profile", "pmc_roofline",
        "roofline",
        "recover",
    }),
    PHASE_KERNEL: frozenset({
        "profile", "pmc_roofline",
        "roofline",   # F0-9 placeholder
        "kernel_opt", "integrate", "deep_kernel_analysis",
        "operator_tuning", "vendor_kernel_config",
        "recover",
    }),
    PHASE_SWEEP: frozenset({
        "sweep", "recover",
    }),
    PHASE_CLOSE: frozenset({
        "report", "session_breakdown",
        "roofline",   # F0-9 placeholder; F3 N31 final comparison uses this
        "recover",
    }),
}
```

### 添加测试到 `inference_optimizer/tests/test_f0_roofline_scaffolding.py`

(在 F0-8 已建文件中追加)

```python
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


def test_legacy_profile_pmc_roofline_still_in_explore():
    """F0 must keep profile/pmc_roofline available — F3 may close them
    via --deny-direct-profile."""
    from inference_optimizer.orchestrator.phase_state import (
        PHASE_EXPLORE, is_action_allowed_in_phase,
    )
    assert is_action_allowed_in_phase("profile", PHASE_EXPLORE)
    assert is_action_allowed_in_phase("pmc_roofline", PHASE_EXPLORE)
```

### 命令

```bash
pytest inference_optimizer/tests/test_f0_roofline_scaffolding.py -v
# 期望 8 passed (3 from F0-8 + 5 here)

pytest inference_optimizer/tests -q --tb=no 2>&1 | tail -3
# 期望 1592 passed (1587 + 5 new)

git add inference_optimizer/orchestrator/phase_state.py
git add inference_optimizer/tests/test_f0_roofline_scaffolding.py
git commit -m "feat(F0-9): add 'roofline' placeholder to EXPLORE/KERNEL/CLOSE allowlists

PolicyGate R1 phase_incompatible will accept propose_action{roofline}
in EXPLORE / KERNEL / CLOSE phases. F1's RooflineExecutor registration
will activate the action; until then a propose lands in TaskRegistry
with no executor and fails fast (no harm, no behaviour drift).

profile / pmc_roofline remain in EXPLORE for legacy / fallback paths;
F3's --deny-direct-profile flag may close them once roofline composite
is the default.

Tests: 5 new (allowlist membership for each phase + legacy preservation).
Total: 1592 passed (1587 + 5)."
```

### 注意事项

- **不要**这一步动 `action_registry.py` (action 名字 → executor factory),F1 才接 executor
- **不要**这一步动 `policy.py` 的 sequence_denial 规则,F3 才动
- 如果有人在 F0 完成后 LLM 模型真的提了 `propose_action{roofline}`, TaskRegistry 会找不到 executor 并失败 —— 这是预期行为(F1 之前不应该被提)

---

## 11. F0-10: cli.py 加 5 个 env 开关骨架

### 编辑 `inference_optimizer/cli.py`

找到 `argparse` 定义,在已有 `--no-kernel` / `--no-framework` 附近插入:

```python
# === F1/F2/F3 Integration Flags (default off; F0 scaffolding) ===

parser.add_argument(
    "--use-roofline-composite",
    action="store_true",
    default=os.environ.get("INFERENCE_OPTIMIZER_USE_ROOFLINE_COMPOSITE") == "1",
    help=(
        "(F1) Enable RooflineExecutor (composite profile + trace_analyze "
        "as a single action). Default off — falls back to separate "
        "profile / pmc_roofline actions. Env var: "
        "INFERENCE_OPTIMIZER_USE_ROOFLINE_COMPOSITE=1"
    ),
)

parser.add_argument(
    "--framework-agent-enabled",
    action="store_true",
    default=os.environ.get("INFERENCE_OPTIMIZER_FRAMEWORK_AGENT_ENABLED") == "1",
    help=(
        "(F2) Allow serving_specialist to invoke `fa candidates` and "
        "git fetch refs/pull/...  Default off. Env var: "
        "INFERENCE_OPTIMIZER_FRAMEWORK_AGENT_ENABLED=1"
    ),
)

parser.add_argument(
    "--deny-direct-profile",
    action="store_true",
    default=os.environ.get("INFERENCE_OPTIMIZER_DENY_DIRECT_PROFILE") == "1",
    help=(
        "(F3 / N9) PolicyGate denies propose_action{profile} when "
        "--use-roofline-composite is on. Forces the LLM to use the "
        "atomic 'roofline' action. Default off."
    ),
)

parser.add_argument(
    "--gain-driven-kernel-opt",
    action="store_true",
    default=os.environ.get("INFERENCE_OPTIMIZER_GAIN_DRIVEN_KERNEL_OPT") == "1",
    help=(
        "(F3 / N19c) Lock kernel_opt until last_cheap_delta_gain < epsilon. "
        "Prevents premature deep work when cheap exploration is still "
        "earning gain. Default off."
    ),
)

parser.add_argument(
    "--roofline-saturation-advisory",
    action="store_true",
    default=os.environ.get("INFERENCE_OPTIMIZER_ROOFLINE_SATURATION_ADVISORY") == "1",
    help=(
        "(F3) Inject saturated_layers hint into specialist + orchestration "
        "prompts when a layer's roofline saturation exceeds threshold. "
        "Soft advisory only — never hard-rejects an LLM intent. Default off."
    ),
)
```

### 把开关读进 SharedState (在 cli.py boot 路径里)

找到 `cli.py` 的 SharedState 构造或 args → state 转换处,加:

```python
# Wire F1/F2/F3 toggles into SharedState so all subsystems see them
shared_state.use_roofline_composite       = bool(args.use_roofline_composite)
shared_state.framework_agent_enabled      = bool(args.framework_agent_enabled)
shared_state.deny_direct_profile          = bool(args.deny_direct_profile)
shared_state.gain_driven_kernel_opt       = bool(args.gain_driven_kernel_opt)
shared_state.roofline_saturation_advisory = bool(args.roofline_saturation_advisory)
```

并在 `shared_state.py` 的 SharedState dataclass 里补对应字段(F0-10 同 commit):

```python
# F1/F2/F3 toggles (cli.py wires from --flags; default off)
use_roofline_composite: bool = False
framework_agent_enabled: bool = False
deny_direct_profile: bool = False
gain_driven_kernel_opt: bool = False
roofline_saturation_advisory: bool = False
```

### 添加测试

继续追加到 `inference_optimizer/tests/test_f0_roofline_scaffolding.py`:

```python
def test_f0_toggles_default_off_on_shared_state():
    from inference_optimizer.orchestrator.shared_state import SharedState
    s = SharedState()
    assert s.use_roofline_composite is False
    assert s.framework_agent_enabled is False
    assert s.deny_direct_profile is False
    assert s.gain_driven_kernel_opt is False
    assert s.roofline_saturation_advisory is False


def test_f0_cli_toggles_present():
    """Verify cli.py exposes the 5 F1/F2/F3 toggles."""
    import subprocess
    result = subprocess.run(
        ["python", "-m", "inference_optimizer.cli", "optimize", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    out = result.stdout + result.stderr
    for flag in ("--use-roofline-composite", "--framework-agent-enabled",
                 "--deny-direct-profile", "--gain-driven-kernel-opt",
                 "--roofline-saturation-advisory"):
        assert flag in out, f"flag {flag} missing from cli --help"
```

### 命令

```bash
pytest inference_optimizer/tests/test_f0_roofline_scaffolding.py -v
# 期望 10 passed

pytest inference_optimizer/tests -q --tb=no 2>&1 | tail -3
# 期望 1594 passed (1592 + 2)

# Smoke check
python -m inference_optimizer.cli optimize --help 2>&1 | grep -E "use-roofline-composite|framework-agent-enabled|deny-direct-profile|gain-driven-kernel-opt|roofline-saturation-advisory"
# 期望 5 行命中

git add inference_optimizer/cli.py
git add inference_optimizer/orchestrator/shared_state.py
git add inference_optimizer/tests/test_f0_roofline_scaffolding.py
git commit -m "feat(F0-10): add F1/F2/F3 env toggles scaffolding (default off)

Adds 5 CLI flags + matching env vars + matching SharedState fields:

- --use-roofline-composite          (F1, env INFERENCE_OPTIMIZER_USE_ROOFLINE_COMPOSITE)
- --framework-agent-enabled         (F2, env INFERENCE_OPTIMIZER_FRAMEWORK_AGENT_ENABLED)
- --deny-direct-profile             (F3 N9, env INFERENCE_OPTIMIZER_DENY_DIRECT_PROFILE)
- --gain-driven-kernel-opt          (F3 N19c, env INFERENCE_OPTIMIZER_GAIN_DRIVEN_KERNEL_OPT)
- --roofline-saturation-advisory    (F3, env INFERENCE_OPTIMIZER_ROOFLINE_SATURATION_ADVISORY)

All default to False — F0 behaviour fully matches pre-roofline-merge tag.
F1/F2/F3 will flip defaults per-PR after acceptance testing.

Tests: 2 new. Total: 1594 passed."
```


---

## 12. F0-11: SKILL.md 加 Integration Freeze 段

> **操作员人工执行** — SKILL.md 是 LLM 的硬契约,改动需要操作员决定措辞。

### 在 `inference_optimizer/SKILL.md` 顶部添加段落

位置: 在 "## Iron Rules" 之后,"## Setup" 之前。

```markdown
## Integration Freeze (2026-05-24 → F3 done)

This branch is in a managed integration phase against origin/main
(roofline-v2 PR #288 + framework-agent PR #280). Until F3 lands:

### What you MUST NOT do

- **Do NOT** `git merge origin/main` directly. Use the F0/F1/F2/F3
  cherry-pick path documented in `docs/notes_lossen_explore/roofline_framework计划.MD`
  and `plan_roofline_framework/`.
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
  `TraceAnalyzeSnapshot`, EXPLORE/KERNEL/CLOSE allowlist `roofline` placeholder,
  5 F1/F2/F3 toggles).
- Add new tests under `inference_optimizer/tests/test_f0_*.py` /
  `test_f1_*.py` / etc.
- Use the toggles to gate F1/F2/F3 features:
  `--use-roofline-composite`, `--framework-agent-enabled`,
  `--deny-direct-profile`, `--gain-driven-kernel-opt`,
  `--roofline-saturation-advisory`.

### References

- `docs/notes_lossen_explore/新增PR报告roofline_framework.MD` —
  full PR #288 / #280 evaluation
- `docs/notes_lossen_explore/roofline_framework计划.MD` —
  high-level integration plan (F0 → F3)
- `plan_roofline_framework/F{0,1,2,3}_*.MD` — phase-specific
  step-by-step playbooks
- `docs/integration/MUST_PRESERVE.md` — v0.8 features that must
  survive integration
- `docs/integration/MAIN_FEATURES_DROPPED.md` — main features
  intentionally dropped
- `docs/integration/dryrun_*.log` — original conflict map snapshot
- `docs/test-baselines/pre_roofline_merge_*.log` — 1584 passing baseline
- Tag `pre-roofline-merge` — rollback anchor
- Tag `f{0,1,2,3}-done` — phase completion anchors
```

### Commit

```bash
git add inference_optimizer/SKILL.md
git commit -m "docs(SKILL): add Integration Freeze section for F0->F3 phase

This section makes the integration policy explicit to anyone reading
SKILL.md (humans and LLM agents alike). It lists the 4 'MUST NOT' rules
that would silently break v0.8 if violated, and points to the phase-level
playbooks and rollback anchors.

The freeze is removed when F3 lands and tag f3-done exists."
```

### 验收

```bash
grep -n "## Integration Freeze" inference_optimizer/SKILL.md   # 找到段落
pytest inference_optimizer/tests -q --tb=no 2>&1 | tail -3     # 数字不变 (SKILL.md 不影响代码)
```

---

## 13. F0-12: 打 tag f0-done

### 命令 (操作员人工)

```bash
cd "$FEAT_ROOT"

# 最后跑一遍 preserved check
bash docs/integration/check_preserved.sh
# 应该全 OK

# 最后跑一遍全量测试
pytest inference_optimizer/tests -q --tb=no 2>&1 | tee /tmp/f0_done_baseline.log | tail -3
# 应该 1594 passed

# 打 tag
git tag -a f0-done -m "F0 (Pre-merge preparation) complete

Commits since pre-roofline-merge:
$(git log --oneline pre-roofline-merge..HEAD)

State:
- 1594 passing tests (1584 + 10 F0 additions)
- framework-agent/ subdirectory imported (no v0.6 deps, not yet wired)
- roofline_snapshot.py + _analysis_keyword_map.py imported (parsers, not yet wired)
- SharedState gained TraceAnalyzeSnapshot + roofline_snapshot_id + roofline_saturation_history
- phase_state.py EXPLORE/KERNEL/CLOSE allowlists gained 'roofline' placeholder
- cli.py gained 5 F1/F2/F3 toggles (all default off)
- shared_state.py gained 5 toggle fields (all default False)
- SKILL.md gained Integration Freeze section
- docs/integration/{MUST_PRESERVE,MAIN_FEATURES_DROPPED}.md generated
- docs/integration/dryrun_*.log archived
- docs/test-baselines/pre_roofline_merge_$(date +%Y%m%d).log archived
- docs/notes_lossen_explore/ holds 7 .MD notes

Behavior with all toggles off: identical to tag pre-roofline-merge.

Next: F1 — Roofline Composite Action.
See plan_roofline_framework/F1_roofline_composite.MD"

git push origin f0-done
git push origin "$FEAT_BRANCH"
```

### 验收

- [ ] `git log --oneline pre-roofline-merge..f0-done | wc -l` == 11 (步骤 1, 3, 4, 5, 6, 7, 8, 9, 10, 11) (注:步骤 2/12 是 tag,不产生 commit)
- [ ] `git tag -l 'f0-done'` 有
- [ ] `git ls-remote --tags origin f0-done` 远端有
- [ ] `pytest inference_optimizer/tests -q --tb=no | tail -3` >= 1594 passed
- [ ] `bash docs/integration/check_preserved.sh` rc=0
- [ ] `python -m inference_optimizer.cli optimize --help | grep -c -E -- "--use-roofline-composite|--framework-agent-enabled|--deny-direct-profile|--gain-driven-kernel-opt|--roofline-saturation-advisory"` == 5
- [ ] **smoke**: `python -m inference_optimizer.cli optimize --model X --max-hours 0.1 --dry-run` (without any new toggle) 行为与 `pre-roofline-merge` 完全一致

---

## 14. F0 整体回滚

如果 F0 任何一步出问题:

```bash
# 软回滚 (保留所有 commits 但放弃 F0 进度,从 pre-roofline-merge 重头来)
git reset --hard pre-roofline-merge

# 或者具体某一步出问题:
git revert <bad commit sha>          # 推荐 — 保留历史可追溯
# OR
git reset --hard HEAD~1              # 直接丢弃,但要小心 push 过的 commit
```

### 远端清理 (如果已经 push)

```bash
# 把远端 lossen_explore 重置回 pre-roofline-merge
git push --force origin pre-roofline-merge:feature/zhenggong/lossen_explore
# 或更保守:
# 1. 创建新备份分支: git push origin HEAD:feature/zhenggong/lossen_explore_backup_$(date +%Y%m%d)
# 2. force push: git push --force origin feature/zhenggong/lossen_explore
```

---

## 15. F0 完成后的 commit 拓扑

```
* (tag: f0-done)                                        ← 操作员
* docs(SKILL): add Integration Freeze section           ← F0-11
* feat(F0-10): add F1/F2/F3 env toggles scaffolding     ← F0-10
* feat(F0-9): add 'roofline' placeholder to allowlists  ← F0-9
* feat(F0-8): pre-add roofline placeholder fields       ← F0-8
* feat(F0-7): import additive scaffolding from main     ← F0-7
* docs(integration): MAIN_FEATURES_DROPPED.md           ← F0-6
* docs(integration): MUST_PRESERVE.md + check script    ← F0-5
* chore(integration): archive dry-run merge map         ← F0-4
* chore(tests): archive pre-merge regression baseline   ← F0-3
* docs: archive lossen_explore session notes            ← F0-1
* (tag: pre-roofline-merge)                             ← F0-2 (操作员先打)
* a1d2730 fix: Saturday-May-2026 marathon hotfixes      (HEAD before F0)
```

每个 commit 都跑一次 `pytest`,数字单调不降:
`1584 → 1584 → 1584 → 1584 → 1584 → 1584 → 1584 → 1587 → 1592 → 1594 → 1594`.

---

## 16. F0 完成 → 进入 F1

下一步: 打开 `plan_roofline_framework/F1_roofline_composite.MD`,从 F1-1 开始。

**重要前提**:
- F0-12 (`f0-done` tag) 已经打上
- 1594+ 测试 green
- `bash docs/integration/check_preserved.sh` rc=0
- 操作员有 30 分钟 + 一份新鲜的 GPU pod 可以做 smoke 测试
