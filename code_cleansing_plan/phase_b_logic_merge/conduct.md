# Phase B · 行为准则

## 必须(DO)

- **每抽出/合并一小块就跑护栏**,不要攒大改动一次测。
- 抽出 = **移动 + 顺手删**,原处只留薄调用;移动前后 `git diff --stat` 总行数应**下降或持平**。
- 拆分前画依赖方向,确保单向;拆分后 `import` 验证无循环。
- 合并前确认两处语义**完全一致**(超时/默认值/异常路径)。

## 禁止(DON'T)

- 禁止净行数上升的"重构"。一上升立即放弃该项,回到删除思路。
- 禁止为"优雅/DRY"引入新基类、新接口层、新跨包公共库——除非它净删更多代码。
- 禁止强拆高耦合段(如 coordinator intent handlers);拿不准就停在"文件内分区 + 注释分隔"。
- 禁止改 §1 对外契约的字段名/形状/CLI flag(只动内部实现)。

## 红线验证(每个 commit 必过)

1. 护栏 keep-list 全绿。
2. CLI `--help` 金标准、breakdown/state 顶层键金标准 **diff 为空**(退役项除外)。
3. 同版本 resume 往返验证(若动了 state 序列化)。
4. `git diff --stat` 净行数 ≤ 0。

## 风险排序(从高到低)

coordinator intent handlers > state 序列化键改名 > breakdown 内部去重 > cli parser 抽出 > 模块内 grid/result 去重。
越靠前越要小步 + 多测;越靠后越可成块做。
