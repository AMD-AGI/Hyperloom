# Roofline + Framework Agent 融合执行计划

> 顶层索引文件。每个 phase 一个独立 .MD,可单独打开执行。

## 文件清单

| 文件 | 阶段 | 工作日 | 状态 |
|---|---|---|---|
| [F0_pre_merge.MD](./F0_pre_merge.MD) | Pre-merge 准备 | 1 天 | 待执行 |
| [F1_roofline_composite.MD](./F1_roofline_composite.MD) | Roofline 复合 Action | 2-3 天 | F0 完成后 |
| [F2_framework_agent.MD](./F2_framework_agent.MD) | Framework Agent 接入 serving_specialist | 2-3 天 | F1 完成后 |
| [F3_policygate_advisory.MD](./F3_policygate_advisory.MD) | PolicyGate 规则 + Soft Advisory | 1-2 天 | F2 完成后 |

## 关键路径变量

所有 plan 文件用以下变量,执行前在 shell 里 export:

```bash
export FEAT_ROOT="/wekafs/zgong/Hyperloom-KB"           # 本 branch (lossen_explore) 工作区
export MAIN_ROOT="/wekafs/zgong/Hyperloom"              # main 分支本地 checkout (PR #288 + #280 已 merged)
export FEAT_BRANCH="feature/zhenggong/lossen_explore"
export ROLLBACK_TAG="pre-roofline-merge"

cd "$FEAT_ROOT"
git remote -v                         # 确认 origin = AMD-AGI/Hyperloom
git branch --show-current             # 确认 = $FEAT_BRANCH
ls "$MAIN_ROOT/inference_optimizer/orchestrator/action_executors/roofline.py"   # main 上的 roofline executor
```

## Cherry-pick 来源

不同于"git fetch + cherry-pick from origin/main",我们使用本地 main checkout 作为来源,
有两个好处:

1. 不需要每次 `git fetch origin main`(可能很大)
2. 可以用 `cp` / `git --git-dir=.../Hyperloom/.git` 双重路径

**两种 cherry-pick 模式**:

### 模式 A: 跨 worktree 的 git checkout (推荐)
```bash
# 在 $FEAT_ROOT 工作区,从 $MAIN_ROOT 的 git 历史里取文件
cd "$FEAT_ROOT"
git --git-dir="$MAIN_ROOT/.git" --work-tree="$MAIN_ROOT" show main:inference_optimizer/orchestrator/action_executors/roofline.py \
    > inference_optimizer/orchestrator/action_executors/roofline.py
```

### 模式 B: 直接 cp (适合大目录)
```bash
# 整个目录
cp -r "$MAIN_ROOT/framework-agent" "$FEAT_ROOT/"
# 单文件
cp "$MAIN_ROOT/inference_optimizer/orchestrator/roofline_snapshot.py" \
   "$FEAT_ROOT/inference_optimizer/orchestrator/roofline_snapshot.py"
```

> 模式 A 的优势:可以从 main 任意 commit 取文件(`show <sha>:path`)。
> 模式 B 的优势:简单粗暴,适合纯新增目录。

## 执行规则

1. **顺序执行**: F0 → F1 → F2 → F3。任何一步失败立即停下,不要并行。
2. **每个 commit 后跑测试**: `pytest inference_optimizer/tests -q --tb=no | tail -3`,
   绿色测试数必须**单调不降**。
3. **每个 phase 结束打 tag**: `f0-done`, `f1-done`, `f2-done`, `f3-done`。
4. **失败回滚**: 见每个 plan 文件的 "回滚" 章节。

## 与上游的关系

- 当前上游 (origin) = `feature/zhenggong/lossen_explore` (同名,已经在前面 push 过)
- main = `origin/main`(只读,我们不直接 merge)
- 每个 phase 完成后,可以发一个 PR 到 origin/main(或直接 push 到上游 lossen_explore,先合到 lossen_explore 再决定何时合 main)
