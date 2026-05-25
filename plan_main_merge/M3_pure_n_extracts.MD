# M3 — 纯 v0.8 兼容的 N 系列 follow-up 提取

> 工作日: 1-2 天
> 风险: 低(每个 N 都是隔离的语义增量,出错可单步回滚)
> 依赖: `m2-done` tag 存在
> 完成节点: tag `m3-done` + ~14 个 commit

## 0. 决策依据

(摘自 Q2 = `extract_pure`)

main 上 30 个 Nxx follow-up 中,提取**与 v0.6 / scoreboard / sequence_denial
无关**的纯改进。每个 N 都附"判定 + 理由 + 移植方式"。

## 1. 完整判定表(用户 review 此表后才执行)

| N | 描述 | 判定 | 理由 / 移植方式 |
|---|---|---|---|
| N5 | analysis.md verbatim 注入到 prompt | **ADOPT MP** | 与本分支 F1-4 重叠;主体已有,M3 只补 `73 lines orchestration.md` 的 guidance 段 |
| N6 | ClaudeBackend cache hit metric extraction | **ADOPT CP** | 纯 backend 改进,无 v0.6 依赖 |
| N9 | hard-block LLM direct propose `profile` | **ADOPT MP** | F3-1 已加 `n9_deny_direct_profile_when_composite_on` PolicyGate rule;M3 补 orchestration.md 文档段 |
| N10 | persist RooflineExecutor SharedState mutations | **ADOPT CP** | 纯 roofline executor 修复,与 F1-3 兼容 |
| N11 | strip base64 image data URLs from analysis.md | **ADOPT CP** | 纯 helper,无 v0.6 依赖,加 SharedState 字段无冲突 |
| N12 | orchestration.md hard rules for action ordering | **ADOPT MP** | 与 F1-5 重叠;M3 检查 hard rules 是否已完整 port |
| N13 | hard-enforce kernel_opt ordering | **DROP** | 形态依赖 scoreboard;F3 已用 v0.8 形态替代 |
| N14 | kernel_opt 前置升级 (2,2,3) | **DROP** | scoreboard 字段(已退场) |
| N15 | auto-source kernel-agent.env.sh | **DONE in M2-5** | M2 已 cherry-pick |
| N16 | InferenceX path preference | **DONE in M2-6** | M2 已 revert |
| N17/N23/N24 | per-launch session_dir | **DONE in M2** | M2 已 adopt |
| N18 / N18b | catalog surface roofline + bury pmc_roofline | **DROP** | 本分支已删 pmc_roofline (commit 2486a19) + 注册 roofline (F1) |
| N19c | gain-driven kernel_opt unlock | **DONE in F3-5** | 本分支 F3-5 用 `explore_attempts_succeeded` 形态实现 |
| N20-A | backends/params variant subset | **DROP** | v0.6 only(backends/params 已合入 explore) |
| N21 | flags-conditional roofline | **DONE in F3-5** | F3-5 同 commit 实现 |
| N22 | keyword → variant advisory | **DONE in F3-4** | F3-4 实现(orchestrator/_analysis_keyword_map.py) |
| N25 | tracelens --steady-state-mode + empty-chunk hard-fail | **ADOPT CP** | 纯 kernel-agent tracelens 改进,无 v0.6 依赖 |
| N26 | auto-retry trace_analyze on chunk warnings | **ADOPT MP** | 改 RooflineExecutor,本分支 F1-2 已 port 基础;M3 补 retry 逻辑 |
| N27 | roofline-failure fallback unlock for backends/params | **ADOPT MP + 改写** | 需要改写:fallback 解锁目标从 `backends/params` 改成 `explore` |
| N28 | lazy session_dir resolve for validate_stack | **DROP** | validate_stack 已删 |
| N29 | critic add validate_stack to approve list | **DROP** | validate_stack 已删 |
| N30 | cheap-exhausted deep boost | **DROP** | 依赖 scoreboard;v0.8 priors 走 KB |
| N31 主体 | auto-enqueue final roofline on CLOSE | **DONE in F3-3** | F3-3 已实现 |
| N31 report 部分 | roofline-comparison report section | **ADOPT MP** | 改 `action_executors/report.py` 加入 first-snapshot vs last-snapshot 对比段 |
| N32 | prompt actionability tags + N31 backfill | **ADOPT MP** | 纯 prompt 改进 + SharedState 字段;无 v0.6 依赖 |
| N33 | idle-tick early-close + critic archival exception | **ADOPT CP** | 纯 coordinator 改进;3 个文件清晰定位 |
| N34 | dispatcher resilience + report-success exits run loop | **ADOPT CP** | 纯 sub_agent_runner + coordinator 改进 |
| N35 | critic exploration-action carve-out | **DROP** | critic.md 加 validate_stack 到 approve list(v0.6) |
| N36 | TraceLens chunk-quality gate | **ADOPT CP** | 纯 kernel-agent tracelens 改进 |
| N37 | move validate_stack from gated to exploration carve-out | **DROP** | validate_stack 已删 |
| N38 | per-action verdict_class metadata + action_verdict_policy | **ADOPT MP** | 纯 critic 改进;改 action_registry + critic prompt;**确保不引用 backends/params/validate_stack 的 verdict class** |

### 总结

- **ADOPT × 14**:N5 / N6 / N9 / N10 / N11 / N12 / N25 / N26 / N27 / N31(report) / N32 / N33 / N34 / N36 / N38
- **DROP × 12**:N13 / N14 / N18 / N18b / N20-A / N28 / N29 / N30 / N35 / N37 + 已在其他 phase 处理的 N15/N16/N17/N19c/N21/N22/N23/N24/N31(主体)
- 形态调整 × 1:**N27**(fallback 目标改成 `explore`)

---

## 2. 环境准备

```bash
cd "$FEAT_ROOT"
git fetch origin main
git tag -l 'm2-done' | grep .             # 必须存在
git tag -l 'm3-done' | grep . && false    # 必须不存在
git status --short                        # 必须为空

pytest inference_optimizer/tests -q --tb=line 2>&1 | tee /tmp/m3_baseline.log | tail -5
```

## 3. 步骤清单

```
M3-1   N5/N12 — orchestration.md guidance 段补全     -> MP  (1 commit)
M3-2   N6 — ClaudeBackend cache metric                -> CP  (1 commit)
M3-3   N9 — orchestration.md hard-block direct profile -> MP  (1 commit)
M3-4   N10 — RooflineExecutor SharedState persist     -> CP  (1 commit)
M3-5   N11 — strip base64 from analysis.md            -> CP  (1 commit)
M3-6   N25 — tracelens --steady-state-mode            -> CP  (1 commit)
M3-7   N26 — trace_analyze auto-retry                 -> MP  (1 commit)
M3-8   N27 — roofline-failure fallback unlock explore -> MP  (1 commit; 改写目标)
M3-9   N31 — roofline comparison report section       -> MP  (1 commit)
M3-10  N32 — prompt actionability tags                -> MP  (1 commit)
M3-11  N33 — idle-tick early-close + critic archival  -> CP  (1 commit)
M3-12  N34 — dispatcher resilience + report exit      -> CP  (1 commit)
M3-13  N36 — TraceLens chunk-quality gate             -> CP  (1 commit)
M3-14  N38 — verdict_class metadata + policy lookup   -> MP  (1 commit; drop v0.6 actions)
M3-15  归档 N-extraction 决策表 + 更新 dropped 文档     -> 1 commit
M3-16  打 tag m3-done                                  -> 0 commit
```

总计 ~15 个 commit。

---

## M3-1: N5/N12 — orchestration.md guidance 段 [MP]

**main commit**: `f8ba71b feat(roofline-v2): analysis.md verbatim injection + orchestration guidance [N5]` +
`fd5f239 feat(roofline-v2): orchestration.md hard rules for action ordering + analysis.md mapping [N12]`

**MP 处理**:
1. 本分支 F1-4 已经把 `_format_analysis_md_full` 接进 prompt_builder。
2. F1-5 已经把 main 的"### Roofline composite action (HARD RULES)"段 port 过来。
3. 残缺的可能是 N5 / N12 在 orchestration.md 加的若干段(`### When to (re-)propose roofline` / `### How to consume the TraceLens analysis section` 等)。
4. 对比:
   ```bash
   diff <(git show origin/main:inference_optimizer/orchestrator/system_prompts/orchestration.md) \
        inference_optimizer/orchestrator/system_prompts/orchestration.md \
        | head -200
   ```
5. 把缺失的 guidance 段补进来,**仅限**与 roofline / analysis.md 相关、与本分支不冲突的段。

```bash
$EDITOR inference_optimizer/orchestrator/system_prompts/orchestration.md
pytest inference_optimizer/tests/test_prompt_assets.py inference_optimizer/tests/test_n12_orchestration_hard_rules.py -q
git add inference_optimizer/orchestrator/system_prompts/orchestration.md
git commit -m "docs(orchestration): N5+N12 analysis.md guidance + action-ordering hard rules"
```

---

## M3-2: N6 — ClaudeBackend cache hit metric [CP]

```bash
git cherry-pick a72e94f
```
**冲突预期**: 低。本分支 `claude.py` 与 main 在 cache metric 路径上无重叠。

**通过条件**: `pytest inference_optimizer/tests/test_claude_backend_cache_metric.py -q` 全绿;
pass +1。

---

## M3-3: N9 — hard-block LLM direct propose profile [MP]

本分支 F3-1 已经实现 `n9_deny_direct_profile_when_composite_on` PolicyGate rule。
M3 补的是:
- orchestration.md 的"You may NOT propose profile directly" 段
- 测试 `test_n9_profile_blocked_from_propose.py`

```bash
# orchestration.md 的段
git show origin/main:inference_optimizer/orchestrator/system_prompts/orchestration.md \
    | grep -A 30 "profile direct"
# 找到对应段,手工 port
$EDITOR inference_optimizer/orchestrator/system_prompts/orchestration.md

# 测试
git checkout origin/main -- inference_optimizer/tests/test_n9_profile_blocked_from_propose.py
pytest inference_optimizer/tests/test_n9_profile_blocked_from_propose.py -q

git add inference_optimizer/orchestrator/system_prompts/orchestration.md \
        inference_optimizer/tests/test_n9_profile_blocked_from_propose.py
git commit -m "feat(N9): hard-block LLM direct propose 'profile' (doc + test, policy done in F3-1)"
```

---

## M3-4: N10 — RooflineExecutor SharedState persist [CP]

```bash
git cherry-pick 92c4389
```
**冲突预期**: 低,主要改 `action_executors/roofline.py`(F1-3 已 port 主体,N10 是
后续修补)。

**通过条件**: `pytest inference_optimizer/tests/test_n10_roofline_promote_persistence.py -q` 全绿。

---

## M3-5: N11 — strip base64 from analysis.md [CP]

```bash
git cherry-pick cf23ea0
```
**冲突预期**: 低。

**通过条件**: `pytest inference_optimizer/tests/test_n11_strip_base64.py -q` 全绿。

---

## M3-6: N25 — tracelens --steady-state-mode [CP]

```bash
git cherry-pick 79ac6b2
```
**冲突预期**: 低,主要改 `kernel-agent/tools/tracelens_analysis.py`。**注意**:本分支 M1
吸收了 kernel-agent 多节点改动,如果 N25 与 M1 中改动重叠,需手工解决。

**通过条件**: `pytest kernel-agent/tools/test_n25_steady_state_mode.py -q` 全绿。

---

## M3-7: N26 — trace_analyze auto-retry on chunk warnings [MP]

```bash
git cherry-pick bee202a
```
**冲突预期**: 中。改 `action_executors/roofline.py`(F1-3 已 port 主体),需要把
auto-retry 逻辑加到现有 RooflineExecutor 上。

**MP 处理**: 
1. cherry-pick 若冲突,**保留**本分支的 F1-3 / F3-4 改动。
2. 把 main 的 retry-on-low-busy-chunks 逻辑 append 到现有 RooflineExecutor 的
   错误处理路径(应该是一个 `if chunk_quality_low: retry` 的判断)。

**通过条件**: `pytest inference_optimizer/tests/test_n26_steady_state_auto_retry.py -q` 全绿。

---

## M3-8: N27 — roofline-failure fallback unlock(改写) [MP]

main commit: `1358386 feat(coordinator): N27 roofline-failure fallback unlock for backends/params`

**问题**: main 的 N27 在 roofline 连续失败时,临时解锁 `backends` / `params` 让
LLM 不阻塞。本分支已无 `backends` / `params`,**需要改写**:fallback 目标改成
`explore`。

**MP 处理**:
1. **不要** 直接 cherry-pick。
2. 读 main commit:
   ```bash
   git show 1358386 -- inference_optimizer/orchestrator/coordinator.py
   ```
3. 找到 `if roofline_consecutive_failures >= N: unlock_actions(['backends', 'params'])`
   类似的判断。
4. 在本分支 coordinator.py 对应位置加同样的判断,但 unlock 目标改成:
   ```python
   if roofline_consecutive_failures >= N:
       # Port-from-main-N27 (改写): main 解锁 backends/params,
       # v0.8 没有这两个 action,改成临时允许 LLM 在 EXPLORE phase 提 explore
       # 作为兜底(本就允许,加这个判断纯粹是把 sequence_denial 临时降级)。
       shared_state.roofline_fallback_unlocked = True
   ```
5. 在 SharedState 加 `roofline_fallback_unlocked: bool = False` 字段。
6. PolicyGate 在 `n9_deny_direct_profile_when_composite_on` rule 里读这个字段,
   roofline_fallback_unlocked=True 时降级该 rule(否则 LLM 永远没法绕过 roofline)。
7. 加测试 `test_n27_roofline_fallback.py`(可以用 main 的版本作为起点,改字段名)。

```bash
$EDITOR inference_optimizer/orchestrator/coordinator.py
$EDITOR inference_optimizer/orchestrator/shared_state.py
$EDITOR inference_optimizer/orchestrator/policy.py
# 移植测试 + 改字段名
git show origin/main:inference_optimizer/tests/test_n27_roofline_fallback.py > /tmp/n27.py
# 手工编辑成 explore-fallback 形态,放入 tests/
$EDITOR /tmp/n27.py
cp /tmp/n27.py inference_optimizer/tests/test_n27_roofline_fallback.py
pytest inference_optimizer/tests/test_n27_roofline_fallback.py -q

git add inference_optimizer/orchestrator/coordinator.py \
        inference_optimizer/orchestrator/shared_state.py \
        inference_optimizer/orchestrator/policy.py \
        inference_optimizer/tests/test_n27_roofline_fallback.py
git commit -m "feat(coordinator): N27 roofline-failure fallback unlock (v0.8 form, port from 1358386)

main commit 1358386 unlocked 'backends'/'params' actions on roofline
failure streak; v0.8 has no such actions. Reimplemented as
SharedState.roofline_fallback_unlocked which downgrades the
n9_deny_direct_profile_when_composite_on PolicyGate rule, letting the
LLM bypass the composite-action gate when roofline keeps failing."
```

---

## M3-9: N31 — roofline comparison report section [MP]

main commit: `e0cc6ae feat(roofline): N30 cheap-exhausted deep boost + N31 final roofline & report comparison`

N31 在 main 是和 N30 同 commit;**N30 部分 drop**,只取 N31 的 report comparison 段。

**MP 处理**:
1. 找到 main 在 `action_executors/report.py` 里加的 roofline-comparison 渲染:
   ```bash
   git show e0cc6ae -- inference_optimizer/orchestrator/action_executors/report.py
   ```
2. 提取 "First snapshot vs Last snapshot" 表格生成逻辑,加到本分支
   `report.py`(F3-3 已经在 CLOSE 加了 final roofline,M3-9 把对比段加进 report)。
3. 测试 `test_n31_roofline_comparison.py` 可以 BA 进来。

```bash
$EDITOR inference_optimizer/orchestrator/action_executors/report.py
git checkout origin/main -- inference_optimizer/tests/test_n31_roofline_comparison.py
pytest inference_optimizer/tests/test_n31_roofline_comparison.py -q
git add inference_optimizer/orchestrator/action_executors/report.py \
        inference_optimizer/tests/test_n31_roofline_comparison.py
git commit -m "feat(report): N31 roofline first-vs-last snapshot comparison section (port from e0cc6ae)

main commit e0cc6ae bundled N30 cheap-exhausted deep boost (scoreboard,
drop) with N31 final roofline + report comparison. We adopt only the
report.py comparison section. N31 final-roofline auto-enqueue is
already in F3-3 (744fdc4)."
```

---

## M3-10: N32 — prompt actionability tags [MP]

```bash
git cherry-pick 8e69732
```

**冲突预期**: 中。该 commit 改了 cli.py + shared_state.py + action_executors/__init__.py。

**MP 处理**:
1. **cli.py 部分**:可能与 M0/M1 的 cli.py 改动冲突。保留本分支已有 args,
   把 N32 加的 actionability-tags args (`--actionability-tags-disabled` ?
   先看 diff 确认参数名)接进来。
2. **shared_state.py 部分**:N32 加了 actionability_tags 字段,无冲突。
3. **action_executors/__init__.py 部分**:加 6 行 import,无冲突。

```bash
# 若 cherry-pick 冲突
git status --short
# 解冲突 → 跑测试
pytest inference_optimizer/tests/test_n32_actionability_tags.py -q
git add ...
git commit ...  # 接收 cherry-pick 自带的 message
```

---

## M3-11: N33 — idle-tick early-close + critic archival [CP]

```bash
git cherry-pick 0e3dc14
```
**冲突预期**: 低,主要改 coordinator.py + shared_state.py + critic.md。

**通过条件**: `pytest inference_optimizer/tests/test_n33_idle_closing_and_critic_archival.py -q` 全绿。

---

## M3-12: N34 — dispatcher resilience + report-success exit [CP]

```bash
git cherry-pick 5d1edd0
```
**冲突预期**: 中。改 sub_agent_runner.py(本分支可能有 F2 改动)+ coordinator.py。

**MP 处理**: 若冲突,**保留**本分支的 F2 修改,把 N34 的 try/except dispatch +
`report_emitted` exit semantics 加在现有结构之上。

**通过条件**: `pytest inference_optimizer/tests/test_n34_dispatcher_resilience_and_report_stop.py -q` 全绿。

---

## M3-13: N36 — TraceLens chunk-quality gate [CP]

```bash
git cherry-pick 3cc13ed
```
**冲突预期**: 低。主要改 `kernel-agent/tools/tracelens_analysis.py`(M1 / M3-6 已吃下基础)。

**通过条件**: `pytest kernel-agent/tools/test_n36_chunk_quality_gate.py -q` 全绿。

---

## M3-14: N38 — verdict_class metadata + policy lookup [MP]

```bash
git cherry-pick 3622572
```
**冲突预期**: 中。该 commit 改了 `action_registry.py` + `critic_agent.py` + `critic.md` +
`cli.py`。其中 `action_registry.py` 加的 verdict_class 表里可能引用 `backends` /
`params` / `validate_stack` —— **必须 drop**。

**MP 处理**:
1. cherry-pick → 看冲突。
2. 编辑 `action_registry.py`:把 `ACTION_VERDICT_POLICY` 表里的 `backends` /
   `params` / `validate_stack` entries 删掉,只留我们实际有的 actions
   (`baseline` / `profile` / `explore` / `roofline` / `sweep` / `recover` /
   `report` / `session_breakdown` / `integrate_patch` / `pmc_roofline`(若有)/
   `assess_remaining_gaps` / `specialist`)。
3. 同样的检查 `critic.md` / `critic_agent.py` 里的引用。
4. 跑测试:`test_n38_action_verdict_class.py`。该测试如果检查具体 action 的
   verdict_class,可能需要按本分支的 action set 调整。

```bash
# 解冲突 + 跑测试
$EDITOR inference_optimizer/orchestrator/action_registry.py
pytest inference_optimizer/tests/test_n38_action_verdict_class.py -q

git add ...
git commit --amend --no-edit   # 保留 cherry-pick 的 message,但加上"drop v0.6 actions"备注
```

---

## M3-15: 归档 N-extraction 决策表 + 更新 dropped 文档 [1 commit]

```bash
# 更新 MAIN_FEATURES_DROPPED.md 加入 §7 "N-series drops"
$EDITOR docs/integration/MAIN_FEATURES_DROPPED.md
# 内容:把上面的"DROP × 12"表格作为 §7 加进去,引用 M3 这个 plan
git add docs/integration/MAIN_FEATURES_DROPPED.md
git commit -m "docs(integration): MAIN_FEATURES_DROPPED §7 — N-series drops audit trail

N13, N14, N18, N18b, N20-A, N28, N29, N30, N35, N37 dropped because
they depend on scoreboard / validate_stack / backends_attempts /
params_attempts — all retired by KB_design §3.4 / §3.9. See
plan_main_merge/M3_pure_n_extracts.MD for per-N rationale."
```

---

## M3-16: 打 tag m3-done

```bash
# 全量回归测试
pytest inference_optimizer/tests -q --tb=line 2>&1 | tee /tmp/m3_done.log | tail -5
bash docs/integration/check_preserved.sh && echo "PRESERVED OK"

# 验证 no-zombie:确保所有 N-drop 没有引入孤儿引用
grep -rE 'backends_attempts|params_attempts|validate_stack_attempts' \
    inference_optimizer/orchestrator/ kernel-agent/ robustness-agent/ \
    | grep -v '\.pyc' | grep -v 'tests/' | wc -l   # 必须 = 0

grep -rE 'framework_pr|--framework-gap|--framework-pr-discover' \
    inference_optimizer/cli.py inference_optimizer/SKILL.md \
    inference_optimizer/orchestrator/policy.py | wc -l   # 必须 = 0

# 升级 SKILL.md Integration Freeze 段
$EDITOR inference_optimizer/SKILL.md
# 把 "Integration Freeze (active 2026-05-24 → tag f3-done)" 改成
# "Main-merge complete (m3-done)"
git add inference_optimizer/SKILL.md
git commit -m "docs(SKILL): retire Integration Freeze block (replaced by m3-done milestone)"

# 锁定 m3 完成
git tag -a m3-done -m "M3 complete: pure N-series follow-ups extracted from main; main-merge done"
git log --oneline m2-done..m3-done | wc -l   # 应该是 ~16
```

**通过条件(M3 整体)**:
1. `pytest inference_optimizer/tests -q` pass >= m2 起点 + 20
2. `bash docs/integration/check_preserved.sh` rc=0
3. zero-zombie 检查通过:
   - 没有 `backends_attempts` / `params_attempts` / `validate_stack_attempts` 引用
   - 没有 `framework_pr` / `--framework-gap` 引用
4. SKILL.md Integration Freeze 段已退役

---

## 回滚

```bash
# 单 N 失败
git reset --hard HEAD^

# 整 phase 回滚
git reset --hard m2-done
git tag -d m3-done
```

---

## 后续:发 PR 到 main

M3-done 之后,可以选择:

### 选项 A — 整体 force-push lossen_explore + 单 PR 到 main

```bash
# 先 push lossen_explore
git push -u origin "$FEAT_BRANCH"
git push origin m0-done m1-done m2-done m3-done

# 发 PR
gh pr create --base main --head "$FEAT_BRANCH" \
    --title "M0-M3: bring main (PR #239 multi-node + N-series follow-ups) into lossen_explore" \
    --body "$(cat plan_main_merge/README.md)"
```

### 选项 B — 拆 4 个 PR(对齐 m{0,1,2,3}-done tag)

每个 phase 一个 PR,review 友好:

```bash
# M0 PR
git checkout -b merge/m0-low-risk m0-done
git push -u origin merge/m0-low-risk
gh pr create --base main --head merge/m0-low-risk \
    --title "M0: low-risk PRs + 11 kernel-opt fixes from main"

# 同理 M1 / M2 / M3
```

### 选项 C — 暂留本分支,等下一轮 lossen_explore 工作完成后一并合 main

(取决于团队节奏。文档化在 README 里)

---

## 跨 phase 验收

到 m3-done 时,做一次跨 phase 的最终 audit:

```bash
# 1. 测试基线对账
cat docs/test-baselines/m0_start_pass.txt
pytest inference_optimizer/tests -q --tb=line 2>&1 | grep -oE '[0-9]+ passed' | tail -1
# 期望:m3-done passes >= m0_start + 53

# 2. MUST_PRESERVE 七项全过
bash docs/integration/check_preserved.sh

# 3. zero-zombie 检查
bash plan_main_merge/scripts/check_no_zombies.sh   # M3-15 后会创建

# 4. 多 layout 兼容性 smoke
INFERENCE_OPTIMIZER_SESSION_LAYOUT=flat python -m inference_optimizer.cli --help > /dev/null
INFERENCE_OPTIMIZER_SESSION_LAYOUT=per_model_ts python -m inference_optimizer.cli --help > /dev/null

# 5. multi-node smoke
python -m inference_optimizer.multi_node.cli --help > /dev/null

# 6. main vs HEAD 的 PR-level diff(用于 PR 描述)
git log --oneline f3-done..m3-done | wc -l        # 总 commit 数
git diff --shortstat f3-done m3-done | tail -1     # +xxxx / -xxxx
```
