# P3_21 — action_registry / YAML 隐藏 scoreboard 字段审计与清理

- **Phase**: P3 · **风险**: 低 · **依赖**: 无 · **后继**: 无

## 目标

`action_registry.py` + `actions/_meta/*.yaml` 携带 `expected_gain_pct` / `typical_runtime_min` / `family` / `max_turns` 等字段。文档与 §3.9 反复强调"没有 scoreboard、由 LLM 判断",但这些字段是潜在的"隐藏打分板"(P1+ scheduler 旋钮)。本步**审计**它们是否真的只作 prompt-advisory,确保没有任何字段被用作**隐藏 gate / 优先级排序**;清理未使用的 scheduler 字段。

## 不变量 vs 策略判定

- registry metadata 用于 **prompt 渲染**(catalogue 显示 cost/gain/risk)= advisory,可接受(只要不 gate)。
- 任何把这些字段用作**自动选择/排序/冷却/优先级**的逻辑 = 隐藏 scoreboard = STRATEGY,删除。

## 改动清单(审计 + 清理)

### 1. 审计字段消费者(`action_registry.py`)
- 确认 `expected_gain_pct`(175–177)、`typical_runtime_min`(184–190)、`family`、`max_turns`、`pipeline_phase`、`verdict_class` 的**所有读取点**:
  - **允许**:prompt 渲染(`prompt_builder` 的 catalogue/ETA、critic verdict_class 选 prompt)。
  - **禁止**:任何 `sort by expected_gain` / `cooldown by family` / `priority` 的调度逻辑。
- 若发现禁止类用法,删除(放权,选择交 LLM)。

### 2. YAML 清理(`actions/_meta/*.yaml`)
- `explore.yaml`(5, 23, 27:`expected_gain_pct:[2,12]`, `max_turns:30`)、`specialist.yaml`(27, 78–82)、`dynamic_action.yaml`(18, 22)、`report.yaml`(19:`<3×typical_runtime_min` 自动 enqueue):
  - `expected_gain_pct`:若仅用于 prompt 显示,保留为"参考区间"并在 prompt 标注"先验,非测量";若未被任何 prompt 使用,删除(避免误导性隐藏先验)。
  - `report.yaml` 自动 enqueue 规则:**保留** deadline 自动 report(产物契约不变量),但把"<3×typical_runtime_min 提前 enqueue"的软触发标注为 hint(不强制 LLM)。
  - `max_turns`:与 P1_07/P1_08 调整后的默认一致。
  - `family` / `pipeline_phase`:文档标注为"非 gating metadata"。

### 3. 文档同步
- 在 registry/YAML 注释中明确:这些字段是 **prompt-advisory metadata**,**不参与任何自动调度/打分/优先级**。

## 连带测试

| 文件 | 动作 |
|---|---|
| `test_action_catalogue.py` | 字段存在性/对齐断言更新(若删字段) |
| `test_prompt_builder.py`(catalogue/ETA/gain 渲染 244, 350) | 与 YAML 改动同步 |
| `test_action_registry`(若有) | 消费者审计后更新 |

## 验证
- grep 确认 `expected_gain_pct` / `typical_runtime_min` 无任何排序/gate 消费者(只剩 prompt 渲染)。
- deadline 自动 report 仍生效(产物契约);提前 enqueue 降级为 hint。
- prompt catalogue 仍正确渲染(或按删字段调整)。

## 回退
- 恢复字段与渲染。

## 残留风险
- 低。本步主要是审计 + 文档化 + 清理死字段。唯一需谨慎的是 `report.yaml` 的 deadline 自动 enqueue —— 那是**产物契约不变量**,务必保留 deadline 兜底,只软化"提前 enqueue"。
