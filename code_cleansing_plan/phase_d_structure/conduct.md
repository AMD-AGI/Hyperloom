# Phase D · 行为准则

## 必须(DO)

- 移动文件用 `git mv` 语义(StrReplace 改 import),保持历史可追溯。
- 每次移动/合并后 `python -c "import inference_optimizer.cli"` + 护栏。
- 合并只针对**强相关 + 单引用**的小文件。
- 打破循环优先"下沉共享符号"到底层。

## 禁止(DON'T)

- 禁止把不相关文件塞进大杂烩(用"文件少"换"单文件复杂"=没降复杂度)。
- 禁止用大量"函数内 import"假装消除了循环。
- 禁止让 `protocol/` / `paths` 等底层模块 import 上层(制造循环)。
- 禁止改 §1 契约(envelope/CLI/schema)的字段与形状,只动文件位置与 import。

## 复杂度判据(本相位是否成功)

- 循环引用数:目标 0。
- 依赖层次:严格单向(底层→顶层 cli)。
- 文件数:Phase B 拆分 + 本相位合并后,**净增 ≤ 个位数**。
- 单文件最大行数:较 Phase 0 基线大幅下降。

四项同时改善才算 Phase D 成功;若文件数暴增,说明 Phase B 拆过头,回看权衡。
