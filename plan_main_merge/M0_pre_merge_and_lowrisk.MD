# M0 — 准备 + 低风险 PR + kernel-opt 纯 fix

> 工作日: 1 天
> 风险: 极低(全部加性或单点 fix)
> 依赖: `f3-done` tag 存在,工作树干净
> 完成节点: tag `m0-done` + ~17 个 commit

## 0. 环境准备

```bash
export FEAT_ROOT="/wekafs/zgong/Hyperloom-KB"
export FEAT_BRANCH="feature/zhenggong/lossen_explore"
export ROLLBACK_TAG="f3-done"

cd "$FEAT_ROOT"
git fetch origin main                       # 必须每次 phase 开始重 fetch
git status --short                          # 必须为空
git log --oneline HEAD -1                   # 应当是 lossen_explore tip(8aeaa11 或之后)
git tag -l 'f3-done'                        # 必须存在
git tag -l 'm0-done'                        # 必须不存在
```

如果上面任何一项不符,**停下**。

## 1. 步骤清单(commit-level,每条标明 CP / MP / BA / DR)

```
M0-1   归档 dry-run + 基线 (建 m0-staging tag)            -> 0 commit  (操作员)
M0-2   PR #242 sqlite synchronous=FULL                     -> CP  (1 commit)
M0-3   PR #251 TraceLens v0.3.1 patch subdir resolver      -> CP  (1 commit)
M0-4   PR #286/#287 CI workflows                            -> BA  (2 commits)
M0-5   codeql 清理(2 commits)                              -> CP  (2 commits)
M0-6   kernel-opt fix #1: GEAK trust default               -> CP  (1 commit)
M0-7   kernel-opt fix #2: ladder KEEP-over-micro           -> CP  (1 commit)
M0-8   kernel-opt fix #3: opt-in GEAK trust status=complete -> CP  (1 commit)
M0-9   kernel-opt fix #4: filter rejected/exhausted batch  -> CP  (1 commit)
M0-10  kernel-opt fix #5: retire kernel after 1 fail       -> CP  (1 commit)
M0-11  kernel-opt fix #6: isolate batch asyncio.gather     -> CP  (1 commit)
M0-12  kernel-opt fix #7: GEAK 90min wrapper + multi-KEEP  -> MP  (1 commit; 跨多文件)
M0-13  kernel-opt fix #8: batch fanout 3→8 + candidates_path -> CP  (1 commit)
M0-14  kernel-opt fix #9: deny report while hot kernels    -> CP  (1 commit)
M0-15  kernel-opt fix #10: yield hot_kernel_unfinished N19c -> MP  (1 commit; N19c 我们没有 → drop N19c 部分)
M0-16  robustness silence no_levers_found while kernel_opt -> CP  (1 commit)
M0-17  shared-state propagate task_groups from trace_analyze -> CP  (1 commit)
M0-18  打 tag m0-done                                       -> 0 commit  (操作员)
```

总计 1 个准备 + 14-16 个 CP + 1-2 个 MP + 0 个 BA。

---

## M0-1: 归档当前状态(建 staging tag,不改代码)

操作员手动:

```bash
cd "$FEAT_ROOT"
git tag -a m0-staging -m "Start of M0 (pre-merge + low-risk batch)"
# 跑一次完整测试,锁定起点 pass 数
pytest inference_optimizer/tests -q --tb=line 2>&1 | tee /tmp/m0_baseline.log | tail -5
# 把 pass 数记录到 docs/test-baselines/m0_start_pass.txt
grep -oE '[0-9]+ passed' /tmp/m0_baseline.log | tail -1 > docs/test-baselines/m0_start_pass.txt
git add docs/test-baselines/m0_start_pass.txt
git commit -m "chore(M0): record baseline pass count before M0"
```

**通过条件**: `pytest` exit 0,pass >= 1624,failed ⊆ `known_failed_20260524.txt`。

---

## M0-2: PR #242 — sqlite synchronous=FULL [CP]

main commit: `7e7ba88 fix(storage): use synchronous=FULL to prevent coordinator.db corruption (#242)`

```bash
git cherry-pick 7e7ba88
```

**冲突预期**: 无。该 commit 只改 `inference_optimizer/storage/connection.py` 的 PRAGMA 语句。

**通过条件**: cherry-pick clean apply;`pytest inference_optimizer/tests/test_storage* -q` 通过(若存在)。

---

## M0-3: PR #251 — TraceLens v0.3.1 patch subdir [CP]

main commit: `9e86dbd merge(main): bring in TraceLens v0.3.1 per-version sglang patch subdir resolver (#251)`

> ⚠️ 这是个 merge commit。先用 `git log 9e86dbd^..9e86dbd --first-parent` 查
> 真正改的代码 commit(应该是 `_server_patcher.py` 的几行)。

```bash
git log --oneline 9e86dbd^2 -10              # 找到 PR 分支的实际 fix commit
# 若是简单 merge,直接 cherry-pick -m 1 9e86dbd
git cherry-pick -m 1 9e86dbd
# 若需要单独取代码 commit,改用:
# git show origin/main:<file> > <file> && git add <file> && git commit ...
```

**冲突预期**: `_server_patcher.py` 在 lossen_explore 改动较多,**可能需要 MP**。
冲突时:`git status` 显示 UU,手工把 main 加的 per-version 路径解析逻辑保留,
本分支已有的修改保留,**不要丢任何一边**。

**通过条件**: `pytest inference_optimizer/tests/test_server_patcher* -q` 通过。
SKILL.md 里已有的 `release/hyperloom_integration_v0.3.1` 引用与 main 一致。

---

## M0-4: PR #286/#287 — CI workflows [BA × 2]

CI 工作流文件本分支没碰过,直接覆盖。

```bash
# PR #286 (cd6d248) 新增 manual-poke workflow
git checkout origin/main -- .github/workflows/manual-poke-issues.yml
git add .github/workflows/manual-poke-issues.yml
git commit -m "ci: one-click scan & poke for stale-pipeline backfill (#286)"

# PR #287 (0b55fc5) 改 stale 跳过逻辑
# 找到改的 workflow 文件
git diff --name-only 0b55fc5^..0b55fc5 | grep '\.github'
# 然后逐个 checkout
git checkout origin/main -- .github/workflows/<file>
git commit -m "fix(manual-poke): skip issues that already have 'stale' label (#287)"
```

**冲突预期**: 无(本分支没改 `.github/workflows/`)。

**通过条件**: 仅 git diff 命中 `.github/workflows/`,Python 代码完全不动。

---

## M0-5: CodeQL 清理 [CP × 2]

```bash
git cherry-pick 7e0eed9          # chore(codeql): clean up unused imports / vars + unreachable code
git cherry-pick cb97e00          # chore(codeql): second pass — clean up remaining
```

**冲突预期**: 可能在我们已经修改过的文件里命中(unused import 被我们改成了 used)。
冲突时:**优先保留本分支的导入语句**,只接受 main 真的删除了一个我们也未使用的 import 的那一行。

**通过条件**: `python -m py_compile inference_optimizer/**/*.py` 全部成功。

---

## M0-6 ~ M0-15: kernel-opt 10 个 follow-up fix

按时间顺序逐个 cherry-pick,每个独立 commit + 单独跑测试。

### M0-6: GEAK trust default [CP]

```bash
git cherry-pick 81f4e06       # fix(kernel-opt): trust GEAK correctness by default; opt-out via env=0
```

主要改 `kernel-agent/tools/kernel_optimization.py` + `kernel-agent/tools/apply_kernel_patch.py`。
**冲突预期**: 中。本分支 kernel-agent 与 main 在 N1 (`select_kernels → trace_analyze`) 重命名
之后没保留 main 的下游修正,可能需要 MP。

**通过条件**: `pytest kernel-agent -q` 通过。

### M0-7: ladder best-selection KEEP-over-micro [CP]

```bash
git cherry-pick 2e06fc8
```
**冲突预期**: 低,主要改 `kernel-agent/tools/kernel_optimization.py`。

### M0-8: opt-in GEAK trust status=complete [CP]

```bash
git cherry-pick 1010c06
```
**冲突预期**: 低(M0-6 已经吃下基础)。

### M0-9: filter rejected/exhausted/in-flight from batch [CP]

```bash
git cherry-pick 7bba999
```
**冲突预期**: 低。

### M0-10: retire kernel after 1 failed ladder + report-gate helper [CP]

```bash
git cherry-pick 90b187a
```
**冲突预期**: 中。该 commit 加了 `report-gate helper`,可能命中本分支 F3-3 N31
auto-enqueue final roofline 的同一段代码。

**冲突解决方针**: 本分支 F3-3 已经在 CLOSE phase 加了 final roofline,
report-gate helper 可以保留(它检查 hot reusable kernels)。**两者不冲突,
都吸收**。

### M0-11: isolate batch sub-task exceptions [CP]

```bash
git cherry-pick 75c2b73
```
**冲突预期**: 低。纯 asyncio.gather → try/except per-task,定位明确。

### M0-12: GEAK 90min wrapper timeout + multi-KEEP integrate queue [MP]

```bash
git cherry-pick ce36e70
```
**冲突预期**: 高(8 files, 1110 insertions, 77 deletions)。本分支
`kernel_request_handlers.py` 与 main 在 multi-KEEP 整合队列那里大概率冲突。

**MP 处理**:
1. 先尝试 cherry-pick;若冲突进入第 2 步。
2. 把 main 的 90min wrapper 改动放在我们的 `kernel_request_handlers.py` 上,**保留
   本分支的 F1-2 `trace_analyze_handler` 别名**。
3. 把 multi-KEEP integrate queue 的状态字段加到本分支的 `shared_state.py`
   (找到 `pending_keeps` / `integrate_queue` 类似字段)。
4. 跑 `pytest inference_optimizer/tests/test_kernel_integrate_and_report.py -q`。

### M0-13: batch fanout 3→8 + auto-inject candidates_path [CP]

```bash
git cherry-pick 1cd9f7d
```
**冲突预期**: 低。

### M0-14: deny report while untried hot reusable kernels remain [CP]

```bash
git cherry-pick 0270b67
```
**冲突预期**: 中,与 F3-3 N31 auto-enqueue final roofline 重叠区域。

**冲突解决方针**: 加 deny-report 但**不要**引入对 `backends_attempts` /
`params_attempts` 的检查(MAIN_FEATURES_DROPPED §2)。只检查 hot
reusable kernels 的 untried 状态。

### M0-15: yield hot_kernel_unfinished to N19c [MP]

main commit: `c900791 fix(coordinator): yield hot_kernel_unfinished gate to N19c (no death-spiral)`

**问题**: N19c (`gain_driven_kernel_opt`) 在本分支是 F3-5 实现的,但
**形态不同** —— main 的 N19c 检查 `backends_attempts` / `params_attempts`,
F3-5 改成了 `explore_attempts_succeeded`。

**MP 处理**:
1. **不要** `git cherry-pick c900791` 直接执行。
2. 看 main commit diff:`git show c900791 -- inference_optimizer/orchestrator/coordinator.py`
3. 找到 yield 逻辑(应该是 `if hot_kernel_unfinished and not gain_driven_unlocked:`
   类似的判断)。
4. 在**本分支的 F3-5 N19c 实现**位置,加同样的 yield;字段名从
   `backends_attempts < 1` 改成 `explore_attempts_succeeded < 1`。
5. commit message: `fix(coordinator): yield hot_kernel_unfinished gate to F3-5 N19c (port from c900791)`

### M0-16: silence no_levers_found while kernel_opt in flight [CP]

```bash
git cherry-pick 1760ce0
```
**冲突预期**: 低,主要改 `robustness-agent/src/robustness_agent/signals/progress.py`。

### M0-17: shared-state propagate task_groups from trace_analyze [CP]

```bash
git cherry-pick 893bc6f
```
**冲突预期**: 中。本分支 F1-1 已经 port 了 `record_trace_analyze`,**main 的 follow-up
要加 task_groups 字段**。两者兼容。

---

## M0-18: 打 tag m0-done

```bash
# 验收
pytest inference_optimizer/tests -q --tb=line 2>&1 | tail -5
bash docs/integration/check_preserved.sh && echo "PRESERVED OK"

# 锁定 m0 完成
git tag -a m0-done -m "M0 complete: low-risk PRs + kernel-opt 10 fixes"
git log --oneline f3-done..m0-done | wc -l   # 应该是 17 ± 1
```

**通过条件(M0 整体)**:
1. `pytest inference_optimizer/tests -q` 通过,pass >= 1624(等于或大于起点;
   有些 kernel-opt fix 自带新测试)。
2. `bash docs/integration/check_preserved.sh` rc=0。
3. `python -m inference_optimizer.cli --help` 不报 import 错误。
4. 没有引入对 `backends_attempts` / `params_attempts` / `validate_stack_attempts`
   字段的读写(MUST_PRESERVE §4 + MAIN_FEATURES_DROPPED §2):
   ```bash
   grep -rE 'backends_attempts|params_attempts|validate_stack_attempts' \
       inference_optimizer/orchestrator/ kernel-agent/ robustness-agent/ \
       | grep -v '\.pyc' | grep -v 'test_' | wc -l   # 必须 = 0
   ```

---

## 回滚

任何一步失败,**优先单步回滚**:

```bash
# 单 commit 失败 → 直接 reset
git reset --hard HEAD^

# 整个 M0 失败 → 回到 f3-done
git reset --hard f3-done
git tag -d m0-staging m0-done 2>/dev/null
```

> 注意:M0 全部产物都在本地未 push,reset 安全;**不要** force-push
> 到 origin 直到 m0-done tag 被打出且测试通过。

---

## 后续

M0 完成后,跳到 [M1_multinode.MD](./M1_multinode.MD)。
