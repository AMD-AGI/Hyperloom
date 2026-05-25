# M1 — PR #239 Multi-node 全量吸收

> 工作日: 2-3 天
> 风险: 中(新维度,但 multi_node/ 子树独立,单节点回路不破)
> 依赖: `m0-done` tag 存在
> 完成节点: tag `m1-done` + ~18 个 commit
>
> **本分支正式支持 `--nodes >= 2` 多节点优化**。

## 0. 决策依据

(摘自 Q1 = `full`)

- 整个 `inference_optimizer/multi_node/` 子树吸收(15 个文件 + bootstrap scripts)
- `_multi_node_env.py` / `_multi_node_server_lifecycle.py` 在 action_executors 的钩子吸收
- `kernel-agent` 多节点改动(SKILL.md / install.sh / kernel_optimization.py /
  parallel_e2e_runner.py / apply_kernel_patch.py)吸收
- `robustness-agent` 多节点抑制(B1/B2 false positives + no_levers_found
  multi-node bumps + auto-downgrade to mock)吸收
- SKILL.md + cli.py 暴露 `--nodes` 参数 / `MULTI_NODE_STATE_FILE` env

## 1. 环境准备

```bash
export FEAT_ROOT="/wekafs/zgong/Hyperloom-KB"
export FEAT_BRANCH="feature/zhenggong/lossen_explore"
cd "$FEAT_ROOT"
git fetch origin main
git log --oneline HEAD -1                   # 应当是 m0-done 之后
git tag -l 'm0-done' | grep .               # 必须存在
git tag -l 'm1-done' | grep . && echo "ERROR m1-done already exists" && false
git status --short                          # 必须为空

# 锁定起点
pytest inference_optimizer/tests -q --tb=line 2>&1 | tee /tmp/m1_baseline.log | tail -5
```

## 2. PR #239 内容总览

```text
59 files / +12920 / -87
├── inference_optimizer/multi_node/                   (15 files, BA)
│   ├── SKILL.md __init__.py __main__.py cli.py
│   ├── _internal/{log,ray_dashboard,safe_client,workload_spec}.py
│   └── scripts/{bootstrap,kill_*,launch_*,kernel_*,apply_tracelens_patch}_multinode.{py,sh}
├── inference_optimizer/orchestrator/action_executors/
│   ├── _multi_node_env.py                            (新增, BA)
│   ├── _multi_node_server_lifecycle.py               (新增, BA)
│   ├── _grid_runner.py                               (改动, MP)
│   ├── baseline.py / profile.py                      (改动, MP)
├── inference_optimizer/cli.py + SKILL.md             (改动, MP)
├── inference_optimizer/scripts/install.sh            (改动, MP)
├── kernel-agent/                                     (改动, MP × 5 files)
└── robustness-agent/                                 (改动, MP × 5 files)
```

## 3. 步骤清单

```
M1-1   bulk-add multi_node/ 子树 (15 files)              -> BA  (1 commit)
M1-2   bulk-add _multi_node_env + _multi_node_server_lifecycle -> BA  (1 commit)
M1-3   port baseline.py + profile.py multi-node hook    -> MP  (1 commit)
M1-4   port _grid_runner.py multi-node hook             -> MP  (1 commit)
M1-5   port cli.py --nodes / -np 参数                    -> MP  (1 commit)
M1-6   port scripts/install.sh multi-node bootstrap     -> MP  (1 commit)
M1-7   bulk-add multi-node 测试                          -> BA  (1 commit)
M1-8   kernel-agent SKILL.md + install.sh               -> MP  (1 commit)
M1-9   kernel-agent kernel_optimization.py multi-node   -> MP  (1 commit)
M1-10  kernel-agent parallel_e2e_runner.py multi-node   -> MP  (1 commit)
M1-11  kernel-agent apply_kernel_patch.py multi-node    -> MP  (1 commit)
M1-12  kernel-agent ensure_auth_proxy.sh (remove?)      -> MP  (1 commit, 谨慎)
M1-13  robustness-agent prompt_inputs / signals fix     -> MP  (1 commit)
M1-14  robustness-agent runtime cli 自动降级 mock        -> CP  (1 commit)
M1-15  robustness-agent no_levers_found multi-node bump -> CP  (1 commit)
M1-16  SKILL.md multi-node 章节                          -> MP  (1 commit)
M1-17  drop main 残留的 multi_node/ .bak 文件             -> DR  (1 commit)
M1-18  打 tag m1-done                                    -> 0 commit
```

总计 ~17 个 commit。

---

## M1-1: bulk-add multi_node/ 子树 [BA]

```bash
# 完整子树拷贝(注意排除 .bak 文件)
git checkout origin/main -- \
    inference_optimizer/multi_node/SKILL.md \
    inference_optimizer/multi_node/__init__.py \
    inference_optimizer/multi_node/__main__.py \
    inference_optimizer/multi_node/cli.py \
    inference_optimizer/multi_node/_internal/__init__.py \
    inference_optimizer/multi_node/_internal/log.py \
    inference_optimizer/multi_node/_internal/ray_dashboard.py \
    inference_optimizer/multi_node/_internal/safe_client.py \
    inference_optimizer/multi_node/_internal/workload_spec.py \
    inference_optimizer/multi_node/scripts/apply_tracelens_patch_multinode.py \
    inference_optimizer/multi_node/scripts/bootstrap.sh \
    inference_optimizer/multi_node/scripts/kernel_bench_multinode.py \
    inference_optimizer/multi_node/scripts/kernel_patch_multinode.py \
    inference_optimizer/multi_node/scripts/kill_multinode.py \
    inference_optimizer/multi_node/scripts/kill_server.sh \
    inference_optimizer/multi_node/scripts/launch_multinode.py \
    inference_optimizer/multi_node/scripts/launch_router.py \
    inference_optimizer/multi_node/scripts/launch_server.sh

# 检查 import-level OK
python -c "import inference_optimizer.multi_node" 2>&1 | head

git add inference_optimizer/multi_node/
git commit -m "feat(multi-node): bulk-add inference_optimizer/multi_node/ subtree (PR #239 part 1)

15 files, ~3500 lines. Pure new subtree, no conflict with single-node.
Source: AMD-AGI/Hyperloom main PR #239 (feature/weilei/multi-node).

  - multi_node/cli.py: --nodes >= 2 entry point
  - multi_node/scripts/: bootstrap, launch, kill, kernel_bench helpers
  - multi_node/_internal/: ray_dashboard, safe_client, workload_spec

Single-node path unchanged; M1-3/M1-4 wire the hooks."
```

**冲突预期**: 无(全新文件)。

**通过条件**:
- `python -c "import inference_optimizer.multi_node"` 成功
- `python -c "from inference_optimizer.multi_node import cli"` 成功

---

## M1-2: bulk-add 两个 action_executors 钩子 [BA]

```bash
git checkout origin/main -- \
    inference_optimizer/orchestrator/action_executors/_multi_node_env.py \
    inference_optimizer/orchestrator/action_executors/_multi_node_server_lifecycle.py

# 检查 import OK + 依赖
python -c "from inference_optimizer.orchestrator.action_executors import _multi_node_env, _multi_node_server_lifecycle"

git add inference_optimizer/orchestrator/action_executors/_multi_node_env.py \
        inference_optimizer/orchestrator/action_executors/_multi_node_server_lifecycle.py
git commit -m "feat(multi-node): add _multi_node_env + _multi_node_server_lifecycle helpers"
```

**冲突预期**: 无。这两个文件本分支不存在。

---

## M1-3: port baseline.py + profile.py multi-node hook [MP]

```bash
# 看 main 的 diff
git diff f3-done origin/main -- \
    inference_optimizer/orchestrator/action_executors/baseline.py \
    > /tmp/m1_baseline_diff.patch
git diff f3-done origin/main -- \
    inference_optimizer/orchestrator/action_executors/profile.py \
    > /tmp/m1_profile_diff.patch
```

**MP 处理步骤**:
1. 阅读 `/tmp/m1_baseline_diff.patch`,找到 multi-node 接入点:
   通常是在 `_run_magpie` 之前加判断:
   ```python
   if task.params.get("nodes", 1) > 1:
       from ._multi_node_env import setup_multi_node_env
       env = setup_multi_node_env(env, task.params)
   ```
2. **不要** 接受 main 在该文件里对 v0.6 字段(`backends_attempts`...)的任何引用。
3. 同样改 `profile.py`(几乎是 baseline 的克隆,模式一致)。

```bash
# 手工编辑 baseline.py + profile.py 加入 hook
$EDITOR inference_optimizer/orchestrator/action_executors/baseline.py
$EDITOR inference_optimizer/orchestrator/action_executors/profile.py

# 跑测试
pytest inference_optimizer/tests/test_p2_2_profile_and_handlers.py -q

git add inference_optimizer/orchestrator/action_executors/baseline.py \
        inference_optimizer/orchestrator/action_executors/profile.py
git commit -m "feat(multi-node): wire _multi_node_env hook into baseline + profile

When task.params['nodes'] > 1, baseline/profile executors invoke
setup_multi_node_env() to populate RCCL_*, MULTI_NODE_STATE_FILE, and
ray_dashboard URLs into the subprocess env before launching Magpie.

Single-node path (nodes=1, default) is byte-identical to f3-done."
```

**通过条件**: `test_p2_2_profile_and_handlers.py` 全绿;`test_breakdown_smoke.py` 全绿。

---

## M1-4: port _grid_runner.py multi-node hook [MP]

```bash
git diff f3-done origin/main -- \
    inference_optimizer/orchestrator/action_executors/_grid_runner.py \
    > /tmp/m1_grid_runner_diff.patch
```

**MP 处理**: `_grid_runner._run_magpie` 是 explore variant 的实际执行函数,
multi-node hook 与 baseline/profile 完全对称。同样的:
- 加 `if nodes > 1: setup_multi_node_env(env, params)`
- **抛弃** main commit `101d07c` 的 `error_class through VariantResult`
  改动? **不抛弃** —— 这是 cosmetic gap fix,纯收益。

```bash
# 编辑
$EDITOR inference_optimizer/orchestrator/action_executors/_grid_runner.py

# 跑测试
pytest inference_optimizer/tests/test_grid_runner_abort_marker.py -q

git add inference_optimizer/orchestrator/action_executors/_grid_runner.py
git commit -m "feat(multi-node): wire _multi_node_env hook into _grid_runner + thread error_class"
```

---

## M1-5: port cli.py --nodes / -np 参数 [MP]

```bash
git diff f3-done origin/main -- inference_optimizer/cli.py > /tmp/m1_cli_diff.patch
```

**MP 处理**:
1. 找到 `argparse.add_argument` 段(`_build_optimize_parser` 函数)。
2. 加入:
   ```python
   parser.add_argument("--nodes", type=int, default=1,
                       help="Number of nodes (>=2 enables multi-node path)")
   parser.add_argument("--nodes-per-job", "-np", type=int, default=None,
                       help="Override nodes per job (default: same as --nodes)")
   ```
3. 在 `_run_optimize` 把 args.nodes 写到 SharedState 或 env(具体看 main 怎么做的):
   - 检查 `main:cli.py` 里搜 `args.nodes`
   - 跟着同样的位置加。
4. **拒绝** 接 `--framework-gap` / `--framework-pr-discover` / `--no-framework`
   CLI 标志(MAIN_FEATURES_DROPPED §3)。

```bash
$EDITOR inference_optimizer/cli.py
python -m inference_optimizer.cli --help | grep nodes   # 应该看到 --nodes
pytest inference_optimizer/tests/test_cli_robustness_options.py -q
git add inference_optimizer/cli.py
git commit -m "feat(multi-node): cli --nodes / -np args + propagate to SharedState"
```

**通过条件**: `python -m inference_optimizer.cli --help` 列出 `--nodes`;无 `--framework-gap`。

---

## M1-6: port scripts/install.sh multi-node bootstrap [MP]

```bash
git diff f3-done origin/main -- inference_optimizer/scripts/install.sh > /tmp/m1_install_diff.patch
```

**MP 处理**:
1. 阅读 diff,找到 multi-node 相关的 ensure_* 函数(应该是 `ensure_ray_multinode`
   或类似)。
2. 把它们加在本分支 install.sh 的对应位置。
3. **拒绝** main 里 `84efe14 cli: gate TP/CONC/EP env export behind nodes>=2`
   带来的破坏 —— 本分支单节点路径还需要 TP/CONC/EP env export。
   - 处理方式:gate 只在 `--nodes>=2` 时生效,nodes==1 走原逻辑。
4. **拒绝** main `fe6cbbd preflight: refuse to clone Magpie on operator-supplied $MAGPIE_DIR`
   ? 这其实是改进,**保留**。

```bash
$EDITOR inference_optimizer/scripts/install.sh
bash inference_optimizer/scripts/install.sh --check-only
git add inference_optimizer/scripts/install.sh
git commit -m "feat(multi-node): install.sh multi-node ray bootstrap (gated on --nodes>=2)"
```

---

## M1-7: bulk-add multi-node 测试 [BA]

```bash
git checkout origin/main -- \
    inference_optimizer/tests/test_multi_node_env_ray.py \
    inference_optimizer/tests/test_multi_node_scripts.py \
    inference_optimizer/tests/test_cli_robustness_options.py \
    inference_optimizer/tests/test_coordinator_failed_variants_audit.py \
    inference_optimizer/tests/test_grid_runner_abort_marker.py \
    inference_optimizer/tests/test_preflight_tracelens_cli_gate.py

pytest inference_optimizer/tests/test_multi_node_env_ray.py \
       inference_optimizer/tests/test_multi_node_scripts.py \
       inference_optimizer/tests/test_cli_robustness_options.py -q

git add inference_optimizer/tests/test_multi_node_*.py \
        inference_optimizer/tests/test_cli_robustness_options.py \
        inference_optimizer/tests/test_coordinator_failed_variants_audit.py \
        inference_optimizer/tests/test_grid_runner_abort_marker.py \
        inference_optimizer/tests/test_preflight_tracelens_cli_gate.py
git commit -m "test(multi-node): bulk-add multi-node + cli + grid-runner tests from PR #239"
```

**通过条件**: 6 个测试文件全绿(若不绿,先 fix MP 前几步);pass 增量 +25 左右。

---

## M1-8 ~ M1-12: kernel-agent multi-node 改动 [MP × 5]

> ⚠️ kernel-agent 在本分支 F1/F2 期间被改过(N1 重命名 + F2 framework_pr_scout
> sub_kind),冲突会比单节点核心多。

```bash
# 5 个文件:逐个 cherry-pick 或手工 port
git diff f3-done origin/main -- kernel-agent/ > /tmp/m1_kernel_agent_diff.patch
wc -l /tmp/m1_kernel_agent_diff.patch
```

### M1-8: kernel-agent SKILL.md + install.sh [MP]

```bash
$EDITOR kernel-agent/SKILL.md          # 加 multi-node 章节
$EDITOR kernel-agent/scripts/install.sh # 加 multi-node ray bootstrap chain
git add kernel-agent/SKILL.md kernel-agent/scripts/install.sh
git commit -m "feat(kernel-agent/multi-node): SKILL.md + install.sh multi-node integration"
```

### M1-9: kernel-agent kernel_optimization.py multi-node [MP]

加 `MULTI_NODE_STATE_FILE` env 读写 + 多节点 GEAK 调度(`bb61c15`)。

```bash
git diff f3-done origin/main -- kernel-agent/tools/kernel_optimization.py
# 手工 port
git add kernel-agent/tools/kernel_optimization.py
git commit -m "feat(kernel-agent/multi-node): honour MULTI_NODE_STATE_FILE in kernel_optimization"
```

### M1-10: kernel-agent parallel_e2e_runner.py multi-node [MP]

main 改动来自 PR #239 的多节点 e2e 路径。

```bash
git diff f3-done origin/main -- kernel-agent/tools/parallel_e2e_runner.py
# 手工 port
git add kernel-agent/tools/parallel_e2e_runner.py
git commit -m "feat(kernel-agent/multi-node): parallel_e2e_runner multi-node fan-out"
```

### M1-11: kernel-agent apply_kernel_patch.py multi-node [MP]

```bash
git diff f3-done origin/main -- kernel-agent/tools/apply_kernel_patch.py
git add kernel-agent/tools/apply_kernel_patch.py
git commit -m "feat(kernel-agent/multi-node): apply_kernel_patch sync across nodes via MULTI_NODE_STATE_FILE"
```

### M1-12: kernel-agent ensure_auth_proxy.sh — 是否 remove? [MP, 谨慎]

main commit: `90669d9 remove ensure_auth_proxy`

**决策**: 本分支 SKILL.md (IR-2) **明确依赖** `ensure_auth_proxy.sh` —— 
否则 Claude SDK 返回 401。**不 remove**。但是 main 把它的内容内联进了
`install.sh`,所以:
1. 检查 main 的 install.sh 是否已经包含 ensure_auth_proxy 的逻辑(应该是的)。
2. 若是 —— 保留本分支的 `ensure_auth_proxy.sh` 作为独立脚本(向后兼容),
   同时把 install.sh 里的 inline 逻辑也接进来(双保险)。
3. 跑 `bash kernel-agent/scripts/ensure_auth_proxy.sh` smoke test。

```bash
# 不删 ensure_auth_proxy.sh,但把 install.sh 里的内联调用接过来
$EDITOR kernel-agent/scripts/install.sh
$EDITOR kernel-agent/scripts/ensure_auth_proxy.sh  # 不动 / 微调
git add kernel-agent/scripts/install.sh kernel-agent/scripts/ensure_auth_proxy.sh
git commit -m "feat(kernel-agent/multi-node): retain ensure_auth_proxy.sh (IR-2) + inline in install.sh

main commit 90669d9 deleted ensure_auth_proxy.sh; we restore it as
the canonical IR-2 anchor (SKILL.md still cites it) and additionally
inline the same logic in install.sh for safety."
```

---

## M1-13 ~ M1-15: robustness-agent multi-node 抑制 [MP / CP]

### M1-13: robustness-agent prompt_inputs + signals [MP]

```bash
git diff f3-done origin/main -- robustness-agent/src/robustness_agent/role/prompt_inputs.py \
    robustness-agent/src/robustness_agent/signals/progress.py
# 手工 port —— 主要是加 multi-node 字段到 prompt_inputs,加 false positive 抑制到 signals
git add robustness-agent/src/robustness_agent/role/prompt_inputs.py \
        robustness-agent/src/robustness_agent/signals/progress.py \
        robustness-agent/tests/test_role_prompt_inputs.py \
        robustness-agent/tests/test_signals_progress.py
git commit -m "feat(robustness/multi-node): suppress B1/B2 false positives in cold-start multi-node"
```

### M1-14: robustness-agent auto-downgrade to mock [CP]

```bash
git cherry-pick 9dd9832       # robustness/multi-node: auto-downgrade --robustness-agent to mock
```

### M1-15: robustness no_levers_found multi-node bump [CP]

```bash
git cherry-pick 468504f       # robustness/no_levers_found: multi-node bumps elapsed floor 45→60 min
git cherry-pick 97318ee       # robustness/no_levers_found: defer until explore phase actually started
git cherry-pick 9e64f7e       # robustness: suppress two B-family false positives in multi-node cold start
```

---

## M1-16: SKILL.md multi-node 章节 [MP]

```bash
git diff f3-done origin/main -- inference_optimizer/SKILL.md > /tmp/m1_skill_diff.patch
```

**MP 处理**:
1. 找到 main 在 SKILL.md 里加的 `## Multi-Node Setup` / `### IR-X multi-node`
   章节。
2. 移植到本分支 SKILL.md:
   - 在 IR-1 之前加 `### IR-1.5 Multi-node nodes>=2 capacity check`(若 main 有)
   - 在 `## Setup` 后加 `## Multi-Node Setup` 章节
   - **不要** 引入 `--framework-gap` 相关的文档
3. 更新 `Integration Freeze` 段:把 multi-node 加入"已吸收"列表。

```bash
$EDITOR inference_optimizer/SKILL.md
git add inference_optimizer/SKILL.md
git commit -m "docs(SKILL/multi-node): document --nodes>=2 path + IR check (PR #239 part 16)"
```

---

## M1-17: drop main 残留的 .bak 文件 [DR]

```bash
# 这两个文件不应该出现在仓库里 —— 是 main 上 git 误提交的
rm -f inference_optimizer/multi_node/cli.py.bak.1778853677
rm -f inference_optimizer/multi_node/scripts/bootstrap.sh.bak.1778853855
git status --short | grep .bak.   # 应该为空
# 若 M1-1 已经 BA 进来了 .bak,这一步要把它们 git rm
git rm -f inference_optimizer/multi_node/*.bak.* 2>/dev/null || true
git rm -f inference_optimizer/multi_node/scripts/*.bak.* 2>/dev/null || true
git commit -m "chore(multi-node): drop accidentally committed .bak files" --allow-empty
```

> 若 git status 已经干净(M1-1 BA 时没有匹配 .bak),用 `--allow-empty`
> 让 commit 留个 audit 痕迹。

---

## M1-18: 打 tag m1-done

```bash
# 验收
pytest inference_optimizer/tests -q --tb=line 2>&1 | tee /tmp/m1_done.log | tail -5
bash docs/integration/check_preserved.sh && echo "PRESERVED OK"

# 多节点 smoke(若有 GPU 环境)
python -m inference_optimizer.multi_node.cli --help 2>&1 | head -10

# 锁定 m1 完成
git tag -a m1-done -m "M1 complete: PR #239 multi-node fully absorbed"
git log --oneline m0-done..m1-done | wc -l   # 应该是 ~17
```

**通过条件(M1 整体)**:
1. `pytest inference_optimizer/tests -q` pass >= m0 起点 + 25 (multi-node tests)
2. `bash docs/integration/check_preserved.sh` rc=0
3. `python -m inference_optimizer.cli --help | grep nodes` 命中
4. `python -m inference_optimizer.multi_node.cli --help` 不 import 错
5. 单节点旧路径回归测试 100% 通过(`test_p2_2_profile_and_handlers.py`,
   `test_breakdown_smoke.py`, `test_p0_5_e2e.py`)
6. **没有引入** `--framework-gap` / `--framework-pr-discover` / `framework_pr.py`
   ```bash
   git diff m0-done m1-done --name-only | grep framework_pr   # 必须为空
   grep -rE 'framework-gap|framework-pr-discover' inference_optimizer/cli.py \
       inference_optimizer/SKILL.md | wc -l                    # 必须 = 0
   ```
7. `multi_node/*.bak.*` 不在工作树:
   ```bash
   find inference_optimizer/multi_node -name '*.bak.*' | wc -l   # 必须 = 0
   ```

---

## 回滚

```bash
# 单步回滚
git reset --hard HEAD^

# 整 phase 回滚到 m0-done
git reset --hard m0-done
git tag -d m1-done 2>/dev/null
# 删 multi_node/ 子树
git clean -fdx inference_optimizer/multi_node/
```

---

## 后续

M1 完成后,跳到 [M2_session_layout.MD](./M2_session_layout.MD)。
