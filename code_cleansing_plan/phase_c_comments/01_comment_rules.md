# Phase C · 步骤 01 — 注释删/留规则与操作

## 删(DELETE)

1. **版本/迭代痕迹**:`N17` / `N33` / `M4` / `M5` / `T2` / `T3` / `v0.6` / `PR-327` / `GAP 1` / `Inv-5.3` 等过程编号与历史故事。
2. **复述签名/docstring 的注释**:注释内容 = 函数名/参数名/docstring 已说的。
3. **逐行叙述"做了什么"**:`# 自增计数器`、`# 导入模块`、`# 返回结果`、`# 调用 X` 这类。
4. **已退役行为的解释性长注释**:no-more-leverage 自动停、NDJSON drain、silent-tick close 的"为什么曾经这样"的叙述。
5. **被注释掉的旧代码块**(dead commented-out code)。
6. **TODO/FIXME 中已完成或已失效的**(确认后删;仍有效的保留并精简)。

## 留(KEEP,但可精简到一行)

1. **非显而易见的意图**:为什么用这个不那么直观的写法。
2. **约束/契约**:"此键被 claw-stats 消费,勿改名"、"此顺序不可换,因为 X 依赖 Y"。
3. **坑/反直觉**:"不要设 HIP_VISIBLE_DEVICES,会让 cuda.is_available() 返回 false"这类经验。
4. **外部依赖说明**:与 §1 契约边界相关的说明。

## 操作

### 定位痕迹注释

```bash
rg -n -e 'N[0-9]{1,2}\b' -e '\bM[0-9]\b' -e '\bT[234]\b' -e 'v0\.[0-9]' \
  -e 'PR-?[0-9]+' -e '\bGAP\b' -e 'Inv-' -e 'retired' -e 'legacy' \
  --glob '*.py' inference_optimizer kernel-agent critic-agent robustness-agent framework-agent
```

### 多段过程注释 → 一行功能描述

把"这段曾经是…后来改成…因为…"压成"做什么"的一行。docstring 已说明的,行内不再重复。

### 批量但谨慎

- 用 `StrReplace` 逐处改;**不要**用正则批量删行(会误删带 `#` 的代码或契约注释)。
- 每个文件改完扫一眼:有没有把"约束/坑"误删。

## 验收

- [ ] 痕迹注释 `rg` 计数清零或仅剩契约相关。
- [ ] 注释占比下降(对比 Phase 0 基线第 4 项)。
- [ ] 护栏全绿(注释改动若让护栏变红 = 误删了代码,回滚)。
- [ ] commit:`Compress <module> comments`(按模块/目录拆)。

## ⚠️ 注意

- **注释精简不应改变任何行为**。护栏若因此变红,说明误删了代码或文档字符串被破坏。
- docstring(三引号)与行内注释区别对待:docstring 是 API 文档,保留但可精简过程性内容;行内 `#` 是重点削减对象。
- 删 TODO 前确认它真的失效;有效的技术债 TODO 保留(精简成一行)。
