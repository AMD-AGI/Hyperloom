# Phase D · 步骤 01 — 降低 import 复杂度,消除循环

## 测量

```bash
pip install pydeps 2>/dev/null
# 循环引用检测
python - <<'PY'
import ast, os, collections
root="inference_optimizer"
graph=collections.defaultdict(set)
for dp,_,fs in os.walk(root):
    if "__pycache__" in dp: continue
    for f in fs:
        if not f.endswith(".py"): continue
        mod=os.path.join(dp,f)
        try: tree=ast.parse(open(mod).read())
        except: continue
        for n in ast.walk(tree):
            if isinstance(n,ast.ImportFrom) and n.module and n.module.startswith("inference_optimizer"):
                graph[mod].add(n.module)
# 简易环检测
print("modules:",len(graph))
PY
```
> 也可 `pydeps inference_optimizer --max-bacon=0 --show-cycles`。重点产出:**循环引用清单** + **被引用最多的"上帝模块"**。

## 目标依赖层次(理想方向)

```
paths / session_paths / protocol(action_surfaces) / intent_parser   ← 底层,无内部依赖
        ↑
storage / shared_state / phase_state / policy                        ← 状态与规则层
        ↑
action_executors / backends / kernel_request_handlers / specialist   ← 执行层
        ↑
coordinator(+ 抽出的 coordinator_*)                                  ← 编排层
        ↑
cli(+ 抽出的 cli_*)                                                  ← 组合根(顶层)
```
原则:**只能向下依赖**。发现向上/横向循环 → 把共享的东西下沉到更底层模块,或用局部 import 打破(最后手段)。

## 操作

1. 跑检测,列出循环与跨层引用。
2. 对每个循环:找出共享符号 → 下沉到底层模块(如 paths/protocol)→ 双方都 import 底层。
3. 消除"函数内 import"中那些**仅为打破循环**的(下沉后可提到模块顶部);保留那些**仅为延迟重依赖**的(记录原因)。
4. 每次改完 `python -c "import inference_optimizer.cli"` 验证。

## 验收

- [ ] 循环引用数 = 0(或记录无法消除项 + 原因)。
- [ ] 关键 import 正常,护栏绿。
- [ ] commit:`Break import cycles by sinking shared symbols`。

## ⚠️ 注意

- 打破循环优先用"下沉共享符号",不要用大量"函数内 import"糊弄(那是把复杂度藏起来,不是降低)。
- Phase B 拆出的 `coordinator_*` / `cli_*` 必须严格单向依赖,本步复核。
