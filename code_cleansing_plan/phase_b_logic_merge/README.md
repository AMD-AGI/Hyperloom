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

### 已完成(net LOC 下降,护栏全绿)

- **Step 01A — 删 coordinator 死分支**(`34a0603a`)
  - 删除 `consecutive_silent_ticks`(每 tick 自增却无人消费,N33 idle early-close 退役后已成死状态)、未被调用的
    `_resolve_silent_ticks_closing_threshold` + `INFERENCE_OPTIMIZER_IDLE_CLOSE_TICKS` env、no-more-leverage / N33
    残留注释与陈旧 docstring stop-signal 行;把私有 silent-tick 单测换成一条"idle run 到 max_ticks 不自闭"的对外行为测。
  - `coordinator.py` 12623 → 12559。
- **Step 01B — 抽出纯函数 helper**(`9b45529b` + 护栏补丁 `91a51fd1`)
  - 把 9 个无状态 helper(model_class 推断 / ISO 解析 / baseline workload+fingerprint / roofline watermark /
    extra-args merge+dedupe)及其常量搬到新模块 `orchestrator/coordinator_helpers.py`(单向依赖,不 import 回 coordinator),
    顺手压缩冗长 docstring;`coordinator.py` 通过 re-export 保持 call-site 与测试 import 不变。
  - `coordinator.py` 12559 → 12149(-410);新文件 348 行;**总净 -62**。

### 各 god-module 行数(Phase B 末)

| 文件 | Phase B 始 | Phase B 末 |
|---|---|---|
| `coordinator.py` | 12623 | 12149 |
| `coordinator_helpers.py`(新) | — | 348 |
| `cli.py` | 6368 | 6368 |
| `shared_state.py` | 4707 | 4703 |
| `collectors.py` | 4377 | 4377 |

### 故意暂缓(尊重铁律,非遗漏)

- **resume/CLOSE 段抽出(01B 余项)**:`_detect_resume_state` / `_on_enter_close` 等大量依赖 `self.*`,强拆需透传
  self/多参数 → 增复杂度且非净减,违背"净行数不增 + 不为拆而拆"。按 conduct 停在文件内。
- **Step 02 cli.py 拆分**:parser/executor/backend 抽出在铁律下至多净持平(搬运 +1 文件),收益≈复杂度下降但风险非零;
  `_RetiredFlag`(退役 flag 显式报错 + 迁移提示)是**活的输入处理**(且 `help=SUPPRESS`,不在 CLI 金标准里),按"拿不准就留"保留。
- **Step 03 shared_state/collectors 去重**:`session_breakdown.json` 顶层键 = §1 对外契约 + 旧 session 兼容唯一保留点;
  state.json 键 = 同版本 resume 契约。内部 `record_*` 合并风险高(超时/默认值差异),金标准 diff 必须为空,暂缓。
- **Step 04 跨子系统去重**:grid / workload-env 已收敛——`explore` / `sweep` / `baseline` 均走 `_grid_runner.run_grid` +
  `materialize_config_with_envs`,无重复可删;critic/robustness backend 的 subprocess 样板**语义不同**(三段 prepare/reason/commit
  + web tools vs 简单调用),按 conduct "语义不一致别强合"不合并;跨包 envelope/常量为**有意复制**保持包独立。

> 退出校验:主包护栏 keep-list 全绿(329 passed);`golden_breakdown_keys` 顶层键 diff 为空;CLI flag 面未动;每个 commit `git diff --stat` 净行数 ≤ 0。
