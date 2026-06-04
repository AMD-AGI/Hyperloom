# Phase D · 步骤 03 — 合并碎文件 / 整理目录

## 合并候选(识别标准)

满足以下任一的小文件,考虑并入邻近归属:
- 极小(< ~50 行)且**只被一处** import。
- 单一函数/常量,与某模块强相关却独立成文件。
- 同目录下职责重叠的多个小文件。

## 已知候选(核实后处理)

| 文件 | 现状 | 建议 |
|---|---|---|
| `inference_optimizer/protocol/__init__.py`(一行 stub) | 几乎空 | 方案 A 后承载或并入 |
| `compat/`(Phase A 删 payload_aliases 后可能空) | 若清空 | 整个目录删除 |
| `action_executors/dynamic_action.py`(stub,若 Phase A 删) | 可能空 | 删 |
| `breakdown/reporters/_renderers/*`(多个小 renderer) | 碎 | 评估合并为少数文件(保留插件注册若有外部扩展点) |
| `multi_node/_internal/*` 小 client | 若各自极小 | 评估合并 |
| `benches/`(`BENCH_TOOL_ENABLED_V1=False`,空注册表) | 占位 | 评估整体删除(确认无引用) |

## 操作

```bash
# 找"小且单引用"文件
for f in $(find inference_optimizer -name '*.py' -not -path '*/__pycache__/*'); do
  lines=$(wc -l < "$f")
  if [ "$lines" -lt 50 ]; then
    base=$(basename "$f" .py)
    refs=$(rg -l "import .*\b$base\b|from .*\b$base\b" inference_optimizer | grep -v "$f" | wc -l)
    echo "$lines lines, $refs refs: $f"
  fi
done | sort -n
```
对 "refs ≤ 1" 的小文件:并入唯一引用方或邻近模块,删原文件,改 import。

## 验收

- [ ] 碎文件并入合理归属;空目录(如清空的 `compat/`、`benches/`)删除。
- [ ] 文件数较 Phase B 末**下降**(抵消拆分增量)。
- [ ] 护栏绿;关键 import 正常。
- [ ] commit:`Consolidate small single-use modules`。

## ⚠️ 注意

- **不要**把不相关的小文件硬塞进一个大杂烩文件——那是把"文件多"换成"单文件复杂",违背主旨。合并只针对**强相关**的。
- `breakdown/reporters/` 若有**插件注册机制**供外部扩展,合并前确认无外部插件依赖该结构。
- `benches/` 删前确认 `dynamic_action_tools.BENCH_REGISTRY` 确实空且无引用。
