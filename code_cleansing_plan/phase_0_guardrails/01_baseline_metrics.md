# Phase 0 · 步骤 01 — 量化基线

## 目标

测出当前数字,填入主计划 `../../code_cleansing.MD` §6 表格,作为全程对照。

## 测量项与命令

> 范围 = 主包 + 4 个 agent(不含 `ci/`)。在仓库根执行。

### 1. 总 LOC(按子系统)

```bash
for d in inference_optimizer kernel-agent critic-agent robustness-agent framework-agent; do
  printf "%-20s " "$d"; find "$d" -name '*.py' -not -path '*/__pycache__/*' | xargs wc -l | tail -1
done
```

### 2. Python 文件数

```bash
find inference_optimizer kernel-agent critic-agent robustness-agent framework-agent \
  -name '*.py' -not -path '*/__pycache__/*' | wc -l
```

### 3. 最大单文件 Top 20

```bash
find inference_optimizer kernel-agent critic-agent robustness-agent framework-agent \
  -name '*.py' -not -path '*/__pycache__/*' -exec wc -l {} + | sort -rn | head -21
```

### 4. 注释行占比(粗测)

```bash
# 注释行(以 # 开头,去缩进)/ 总非空行
for d in inference_optimizer kernel-agent critic-agent robustness-agent framework-agent; do
  total=$(grep -rhn --include='*.py' '.' "$d" | wc -l)
  comm=$(grep -rhn --include='*.py' -E '^\s*#' "$d" | wc -l)
  echo "$d comments=$comm nonblank=$total"
done
```

### 5. 退役 action 引用计数(Phase A 的清零对象)

```bash
rg -c --stats -e '\bselect_kernels\b' -e '\bvalidate_stack\b' \
  -e "'backends'" -e '"backends"' -e "'params'" -e '"params'\" \
  -e '\bsetup\b' -e '\bclassify\b' \
  inference_optimizer kernel-agent critic-agent robustness-agent framework-agent \
  | tail -5
```
> `backends`/`params`/`setup`/`classify` 是普通词,需人工区分"动作名引用" vs "无关用法";只统计动作名相关 call-site。

### 6. 死代码探测(辅助,非权威)

```bash
pip install vulture 2>/dev/null
vulture inference_optimizer --min-confidence 80 | head -50
```
> vulture 会误报(动态注册的 executor / 反射调用)。只作线索,不作删除依据;删除依据是 call-site 人工确认。

## 产出

在 §6 表"当前(基线待测)"列填入数字。把上述命令的原始输出存到:

```
code_cleansing_plan/phase_0_guardrails/baseline_snapshot.txt
```

(每相位结束重测一次,追加到同文件,形成趋势。)

## 验收

- [ ] §6 五项指标有实际数字。
- [ ] `baseline_snapshot.txt` 已生成并提交。
