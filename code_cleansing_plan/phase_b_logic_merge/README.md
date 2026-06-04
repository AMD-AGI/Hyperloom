# Phase B — 逻辑合并 + 拆瘦 god-module

## 目的

- 合并散落的相似逻辑(去重)。
- 拆瘦超大文件(`coordinator.py` 12k / `cli.py` 6.5k / `shared_state.py` 4.7k / `collectors.py` 4.4k),
  **单文件复杂度优先于文件数量**(主计划 §10.4)。

> 这是**最高风险**相位:动的是活逻辑。每一步都靠 Phase 0 护栏兜底,并严守"净行数不增"。

## 核心约束(本相位铁律)

1. **净行数不增**:拆分/合并若让总行数上升 → 放弃。拆分是"搬运 + 顺手删",不是复制。
2. **不引入循环引用**:拆出的模块只能单向依赖。拆前画一遍依赖方向。
3. **不引入新抽象**:除非该抽象能净删更多代码。禁止为"优雅"加基类/接口层。
4. **行为等价**:对外可观测(护栏 + 金标准)不变。

## 步骤文件

- [`01_coordinator_split.md`](01_coordinator_split.md) — 拆瘦 `coordinator.py`。
- [`02_cli_split.md`](02_cli_split.md) — 拆瘦 `cli.py`。
- [`03_state_breakdown_slim.md`](03_state_breakdown_slim.md) — `shared_state.py` / `collectors.py` 瘦身。
- [`04_cross_subsystem_dedup.md`](04_cross_subsystem_dedup.md) — 跨子系统/模块内重复去重。
- [`conduct.md`](conduct.md) — 行为准则。

## 入口标准

- Phase A 出口达标(死代码已清,净减已提交)。在干净的代码上合并,不带垃圾。

## 出口标准

- [ ] 4 个 god-module 行数显著下降(目标见各步骤)。
- [ ] 无循环引用(`01` 提供的检测脚本通过)。
- [ ] 护栏全绿;CLI/产物金标准形状不变。
- [ ] 总 LOC 较 Phase A 末再次下降(或持平,绝不上升)。

## 通用合并流程

1. 识别重复/可下沉逻辑(`rg` 找相似块、相同字符串、复制粘贴)。
2. 选定单一归属(已有的最合适模块,**不新建**除非必要)。
3. 搬运 + 删重复 + 改引用。
4. 跑护栏 + 量 LOC。
5. 提交:`Merge <A> into <B>` / `Extract <X> from coordinator`。

## 进度记录

(完成后填:各文件前后行数 / commit)
