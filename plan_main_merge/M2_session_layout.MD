# M2 — N17 + N23 per-model+per-launch session_dir 改造

> 工作日: 1-2 天
> 风险: 中高(SKILL.md / paths.py / IR-1 / IR-2 + resume 协议全面改写)
> 依赖: `m1-done` tag 存在
> 完成节点: tag `m2-done` + ~8 个 commit

## 0. 决策依据

(摘自 Q3 = `adopt`)

把 `$USER_DATA_PATH` 从"扁平单 session"语义改成"workspace + per-launch
subdir"语义,以支持:
- 同一 pod 上多个 session 共存(避免互相覆盖 state.json)
- 每个 session 按模型分类(`$USER_DATA_PATH/<model>/<launch_ts>/`)
- `--resume-from <path>` 显式选择某 session
- 不传 `--resume-from` 时:自动取最新 launch(N17 默认)
- workspace-shared:`runtime/`, `logs/`(install.sh 的产物)
- per-session:`storage/`, `agents/`, `runs/`, `manifest.json`, `state.json` 等

## 1. 关联 main commit

| commit | 角色 |
|---|---|
| `df4f1ce` | N17 主体:`feat(paths): per-model + per-launch session_dir layout` |
| `c48f1fc` | N17 followup:`fix(paths): accept PathLike for model_name` |
| `f37b2d3` | N23:`fix(cli): --resume N17-aware + N24 kernel-agent env hard-fail` |
| `2549b8f` | N15:`fix(env): auto-source kernel-agent.env.sh in cli preflight + lazy kernel-agent root` |
| `71f7068` | revert N16:drop /wekafs/InferenceX preference(直接接 revert,跳 N16 本体) |

> 注意:N16 (`4da230d`) 已被 revert (`71f7068`),不要接 N16 本体。

## 2. 环境准备

```bash
cd "$FEAT_ROOT"
git fetch origin main
git log --oneline HEAD -1
git tag -l 'm1-done' | grep .             # 必须存在
git tag -l 'm2-done' | grep . && false    # 必须不存在
git status --short                        # 必须为空

pytest inference_optimizer/tests -q --tb=line 2>&1 | tee /tmp/m2_baseline.log | tail -5
```

## 3. 步骤清单

```
M2-1   port N17 主体 paths.py + cli.py + conftest.py   -> MP  (1 commit)
M2-2   port N17 followup PathLike for model_name      -> CP  (1 commit)
M2-3   port N23 --resume-from <path> + find_latest    -> MP  (1 commit)
M2-4   port N24 kernel-agent env hard-fail            -> MP  (1 commit, bundled with N23 in main)
M2-5   port N15 auto-source kernel-agent.env.sh       -> CP  (1 commit)
M2-6   revert N16 InferenceX preference               -> CP  (1 commit)
M2-7   rewrite SKILL.md session_dir + IR-1 + IR-2     -> MP  (1 commit)
M2-8   update test_p6_session_layout.py + 增量测试    -> BA  (1 commit)
M2-9   打 tag m2-done                                  -> 0 commit
```

总计 ~8 个 commit。

---

## M2-1: port N17 主体 [MP]

main commit: `df4f1ce feat(paths): per-model + per-launch session_dir layout [N17]`

**Files**:
- `inference_optimizer/paths.py` (+204 -43)
- `inference_optimizer/cli.py` (+12 -0,只在 make_session_dir 调用处)
- `inference_optimizer/tests/conftest.py` (+25,加 autouse fixture)

**MP 处理**:
1. 先看 main 的实现:
   ```bash
   git show origin/main:inference_optimizer/paths.py > /tmp/main_paths.py
   diff inference_optimizer/paths.py /tmp/main_paths.py | head -100
   ```
2. paths.py 的关键改动:
   - `_SESSION_SKELETON` → 拆分成 `_WORKSPACE_SKELETON` + `_SESSION_SKELETON`
   - 新增 `INFERENCE_OPTIMIZER_SESSION_LAYOUT` env (`flat` | `per_model_ts`),默认 `per_model_ts`
   - 新增 `INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR` env(进程级 pin)
   - `make_session_dir(model_name=None, layout=None)` 签名扩展
   - 新增 `find_latest_per_session_dir(workspace_root) -> Path | None`
   - 新增 `sanitize_model_basename(name) -> str`
3. 这些都是新增/扩展,与本分支 paths.py 不冲突(单看 diff)。直接覆盖 paths.py。
4. cli.py 的改动是把 `args.model` 透传到 `make_session_dir(model_name=args.model)`,
   定位明确,手工合并。
5. conftest.py 的 autouse fixture 是为了防止 `INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR`
   跨测试泄露,直接接进来。

```bash
# 1. paths.py 直接采纳 main(它的改动是结构性的,本分支没碰过相同区段)
cp /tmp/main_paths.py inference_optimizer/paths.py
# 2. 手工核对 cli.py 在 make_session_dir 调用处的改动
$EDITOR inference_optimizer/cli.py
# 3. tests/conftest.py 加 autouse fixture
$EDITOR inference_optimizer/tests/conftest.py

# smoke
python -c "from inference_optimizer.paths import make_session_dir, find_latest_per_session_dir, sanitize_model_basename"

git add inference_optimizer/paths.py inference_optimizer/cli.py \
        inference_optimizer/tests/conftest.py
git commit -m "feat(paths): N17 per-model + per-launch session_dir layout (port from df4f1ce)

Workspace-shared (one copy per \$USER_DATA_PATH):
  runtime/  runtime/source-mirrors/  runtime/geak-config/  logs/

Per-session (one copy per launch, keyed by model + UTC timestamp):
  storage/  personas/  checkpoints/  kb/  findings/  reports/
  agents/{orchestration,kernel,critic,robustness}/
  runs/{baseline,profile,backends,params,sweep,integrate,kernel_opt}/
  kernel-agent/  kernel-agent-workspace/  patches/  optimizer_runs/

New env:
  INFERENCE_OPTIMIZER_SESSION_LAYOUT=flat|per_model_ts (default per_model_ts)
  INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR=<resolved per-session subdir pin>

Backward compatibility:
  USER_DATA_PATH=<workspace_root> still valid; the layout knob makes
  the per-launch subdir resolution explicit. INFERENCE_OPTIMIZER_SESSION_LAYOUT=flat
  restores the pre-N17 behaviour for legacy automation."
```

**通过条件**: `python -c "import inference_optimizer.paths"` 成功;`test_p6_session_layout.py`
应该 fail(因为还没更新测试),M2-8 会修。

---

## M2-2: N17 followup PathLike for model_name [CP]

```bash
git cherry-pick c48f1fc
```
**冲突预期**: 无(M2-1 已 port 主体)。

---

## M2-3: port N23 --resume-from + find_latest [MP]

main commit: `f37b2d3 fix(cli): N23 --resume N17-aware + N24 kernel-agent env hard-fail`

**Files**:
- `inference_optimizer/cli.py` (resume 路径改造)
- `inference_optimizer/paths.py` (`find_latest_per_session_dir` —— M2-1 已加)
- `inference_optimizer/scripts/install.sh` (相关 env 默认值)

**MP 处理**:
1. cli.py 加 `--resume-from PATH`:
   ```python
   parser.add_argument(
       "--resume-from",
       type=Path,
       help="Explicit per-session subdir to resume from (descendant of $USER_DATA_PATH). "
            "Implies --resume. When omitted with --resume, auto-pick latest.",
   )
   ```
2. `_run_optimize` 在 resume 分支:
   ```python
   if args.resume:
       if args.resume_from:
           session_dir = args.resume_from.resolve()
           if not session_dir.is_relative_to(workspace_root):
               raise SystemExit(f"--resume-from must be descendant of {workspace_root}")
       else:
           session_dir = find_latest_per_session_dir(workspace_root) or workspace_root
       os.environ["INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR"] = str(session_dir)
   ```
3. **依赖** `manifest.json` / `state.json` 必须在解析后的 session_dir 下,
   否则报 N17-aware 错误信息。

```bash
$EDITOR inference_optimizer/cli.py
# smoke
python -m inference_optimizer.cli --help | grep resume   # 应该看到 --resume + --resume-from
pytest inference_optimizer/tests/test_n23_resume_per_session.py -q  # 这个测试 M2-8 才加

git add inference_optimizer/cli.py
git commit -m "feat(cli): N23 --resume-from <path> + N17-aware latest-session auto-pick (port from f37b2d3)"
```

---

## M2-4: port N24 kernel-agent env hard-fail [MP]

> N24 在 main 是和 N23 同 commit (`f37b2d3`),M2-3 已部分覆盖。这里单独走
> 一个 commit 把 install.sh / cli preflight 的"找不到 kernel-agent.env.sh 立即报错"
> 路径补齐。

```bash
git diff f3-done origin/main -- inference_optimizer/scripts/install.sh > /tmp/m2_install_diff.patch
git diff f3-done origin/main -- inference_optimizer/cli.py | grep -A 20 'kernel-agent.env.sh'
```

**MP 处理**:
1. `cli.py::_preflight()` 在第 1 步(re-export auth aliases)之前加 hard-fail:
   ```python
   kernel_env = os.environ.get("KERNEL_AGENT_ENV") or \
       f"{user_data_path}/runtime/kernel-agent.env.sh"
   if not Path(kernel_env).exists():
       log.error("KERNEL_AGENT_ENV not found: %s  — run install.sh first", kernel_env)
       sys.exit(2)
   ```
2. `install.sh` 在 `write_env_file` 之后 verify 文件存在。

```bash
$EDITOR inference_optimizer/cli.py
$EDITOR inference_optimizer/scripts/install.sh
git add inference_optimizer/cli.py inference_optimizer/scripts/install.sh
git commit -m "feat(cli/install): N24 hard-fail when kernel-agent.env.sh missing (port from f37b2d3 b)"
```

---

## M2-5: port N15 auto-source kernel-agent.env.sh [CP]

```bash
git cherry-pick 2549b8f
```

**冲突预期**: 中。本分支 `_preflight` 已经在 M2-4 加了 hard-fail。N15 是在
hard-fail 之前先尝试 auto-source。

**冲突解决**: 顺序是 `auto-source(N15) → 如果还没有 → hard-fail(N24)`。
冲突时把 N15 的 source 逻辑放在 N24 hard-fail 之前。

```bash
# 若 cherry-pick 冲突
pytest inference_optimizer/tests/test_preflight_auth_override.py -q
```

---

## M2-6: revert N16 InferenceX preference [CP]

```bash
git cherry-pick 71f7068
```

> 这是 main 自己的 revert,我们直接接 revert,不接 N16 本体。

**冲突预期**: 无(本分支没有 N16 改动可 revert,但 git 会创建一个 no-op revert
commit;若 cherry-pick 报"empty",用 `--allow-empty`):

```bash
git cherry-pick --allow-empty 71f7068 2>/dev/null || \
    git commit --allow-empty -m "revert(install): drop /wekafs/InferenceX preference (port from 71f7068)"
```

---

## M2-7: rewrite SKILL.md session_dir + IR-1 + IR-2 [MP]

> 这是 M2 的核心文档改造。`/workspace/hyperloom` 单 session 假设贯穿
> SKILL.md 多处,要逐处替换。

**改造点清单**:

1. SKILL.md 顶部"What This Skill Runs":
   ```text
   State lives in **one fixed session directory**...
   ```
   改成:
   ```text
   State lives in a workspace directory; each launch creates a
   per-model + per-launch subdirectory (N17). Default workspace is
   /workspace/hyperloom; override via $USER_DATA_PATH.
   ```

2. SKILL.md 目录树插图:把扁平结构改成 workspace/per-session 分层。

3. IR-1 描述:GPU 检查仍然适用,但增加一句:多 session 并存时,GPU 检查只
   针对当前 launch 的 workspace_root。

4. IR-2 描述:install.sh 现在写 workspace-shared `runtime/kernel-agent.env.sh`,
   多个 launch 共享。

5. `## Resume Existing Session`:增加 `--resume-from <path>` 子节。

6. `## Setup → Recovery`:
   ```text
   In sandboxes where /workspace/hyperloom is unwritable, override:
   export USER_DATA_PATH=...
   ```
   保留,但说明:`USER_DATA_PATH` 现在是 workspace_root,不再是单 session。

7. `Session dir resolution order` 表:重新对齐到 N17 的实际优先级:
   ```text
   1. $INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR (process pin, set by N17/N23)
   2. $USER_DATA_PATH = workspace_root → make_session_dir(model, layout) →
      workspace_root/<model>/<launch_ts>/
   3. Default workspace_root = /workspace/hyperloom
   ```

8. Path helpers 表:加 `find_latest_per_session_dir(workspace_root)`,
   加 `sanitize_model_basename(name)`。

9. "Migration Notes (upgrading from earlier Hyperloom releases)" 段:
   加新一条:
   ```text
   4. Per-launch session layout (N17/N23). Workspace_root is now $USER_DATA_PATH
      itself (NOT a single session). Each launch auto-creates
      workspace_root/<model>/<launch_ts>/. Use --resume-from <path> to
      target a specific session; without it, --resume auto-picks the latest.
      Set INFERENCE_OPTIMIZER_SESSION_LAYOUT=flat to restore the legacy
      single-session behaviour.
   ```

10. `Integration Freeze (active 2026-05-24 → tag f3-done)` 段:**不动**,等
    M3-done 之后再决定是否升级到 `m3-done`。

```bash
$EDITOR inference_optimizer/SKILL.md

# smoke:确保 SKILL.md 还能 parse(frontmatter / 章节)
head -15 inference_optimizer/SKILL.md          # frontmatter 不变
grep -cE "^### IR-[0-9]+" inference_optimizer/SKILL.md   # 应该 ≥ 6

git add inference_optimizer/SKILL.md
git commit -m "docs(SKILL): N17/N23 per-model+per-launch session_dir migration notes

Rewrites the 'What This Skill Runs' / 'Resume Existing Session' /
'Session dir resolution order' / 'Path helpers' / 'Migration Notes'
sections to reflect the workspace-vs-per-session split. IR-1 and IR-2
clarified for multi-session workspaces."
```

---

## M2-8: update test_p6_session_layout.py + N17/N23 测试 [BA]

```bash
# main 上 test_p6_session_layout.py 已经更新成 N17 测试
git checkout origin/main -- inference_optimizer/tests/test_p6_session_layout.py
# 顺带把 N23 的测试拉进来
git checkout origin/main -- inference_optimizer/tests/test_n23_resume_per_session.py
git checkout origin/main -- inference_optimizer/tests/test_n24_kernel_agent_env_hardfail.py

# 跑测试
pytest inference_optimizer/tests/test_p6_session_layout.py \
       inference_optimizer/tests/test_n23_resume_per_session.py \
       inference_optimizer/tests/test_n24_kernel_agent_env_hardfail.py -q

git add inference_optimizer/tests/test_p6_session_layout.py \
        inference_optimizer/tests/test_n23_resume_per_session.py \
        inference_optimizer/tests/test_n24_kernel_agent_env_hardfail.py
git commit -m "test(paths): N17/N23/N24 per-launch layout + resume-from + env-hardfail tests"
```

**通过条件**: 3 个测试文件全绿;pass 增量 +8。

---

## M2-9: 打 tag m2-done

```bash
# 全量回归测试
pytest inference_optimizer/tests -q --tb=line 2>&1 | tee /tmp/m2_done.log | tail -5
bash docs/integration/check_preserved.sh && echo "PRESERVED OK"

# 多 layout 兼容性测试
INFERENCE_OPTIMIZER_SESSION_LAYOUT=flat python -c "
from inference_optimizer.paths import make_session_dir
import os, tempfile
with tempfile.TemporaryDirectory() as d:
    os.environ['USER_DATA_PATH'] = d
    sd = make_session_dir()
    print('flat:', sd)
    assert str(sd) == d, sd
"

INFERENCE_OPTIMIZER_SESSION_LAYOUT=per_model_ts python -c "
from inference_optimizer.paths import make_session_dir
import os, tempfile
with tempfile.TemporaryDirectory() as d:
    os.environ['USER_DATA_PATH'] = d
    sd = make_session_dir(model_name='/wekafs/models/Qwen-7B')
    print('per_model_ts:', sd)
    assert 'Qwen-7B' in str(sd), sd
"

# 锁定 m2 完成
git tag -a m2-done -m "M2 complete: N17 + N23 per-model+per-launch session_dir adopted"
git log --oneline m1-done..m2-done | wc -l   # 应该是 ~8
```

**通过条件(M2 整体)**:
1. `pytest inference_optimizer/tests -q` pass >= m1 起点 + 8 (N17/N23/N24 tests)
2. `bash docs/integration/check_preserved.sh` rc=0
3. `--resume` + `--resume-from` smoke 都过
4. legacy `INFERENCE_OPTIMIZER_SESSION_LAYOUT=flat` 兼容性 smoke 过
5. SKILL.md 解析 OK,frontmatter 不变
6. 没有引入 `backends_attempts` / `params_attempts` / `framework_pr` 字段

---

## 回滚

```bash
# 单步
git reset --hard HEAD^

# 整 phase
git reset --hard m1-done
git tag -d m2-done
```

> 注意 M2 改了 paths.py 的语义,**未跑过 M2 测试就 git push 是危险的** ——
> 现存 session_dir 可能已经按 flat layout 创建,N17 的 auto-pick 在已存在的
> session 上会创建新的 per_model_ts 子目录,resume 协议会找不到 manifest.json。
> 操作员在生产环境第一次跑要么先 set `INFERENCE_OPTIMIZER_SESSION_LAYOUT=flat`,
> 要么 `--resume-from <existing flat dir>`。

---

## 后续

M2 完成后,跳到 [M3_pure_n_extracts.MD](./M3_pure_n_extracts.MD)。
