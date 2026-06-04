# Phase B · 步骤 03 — shared_state.py / collectors.py 瘦身

## shared_state.py(4730 行)

### 删

- Phase A 已删退役字段(`last_select_kernels`、`params_no_promote_streak` proxy、`_migrate_*`)。这里补:确认无残留 setter/getter/序列化分支。
- 命名收口后(`cortex_* → recipe_*`)确认序列化键:**写到 `state.json` 的键是同版本 resume 契约**,内部变量名可改,磁盘键名改动要确保 save/load 对称(护栏:同版本 resume 验证)。

### 合并

- 大量 `record_*` / `apply_*_update` 方法形态相似(读 → 改 dataclass → save)。可合并的:
  - 同类 ledger 的 append(`kernel_opt_attempts` / `kernel_integrate_attempts` 等)抽公共 `_append_attempt(ledger_name, entry)`。
  - 仅在能净删行时合并;若合并后更绕,保留。

### 目标

降到更易读;dataclass 字段分组用注释分区(不强拆 dataclass)。

## collectors.py(4391 行,breakdown 生成器)

### 约束(关键)

`session_breakdown.json` 是 **§1 对外契约 + 旧 session 兼容唯一保留点(§10.3)**。
**对外顶层键与形状必须不变**(对照 Phase 0 `golden_breakdown_keys.txt`)。本步只动**内部生成逻辑**。

### 删 / 合并

- 删退役键的**内部计算**逻辑,保留对外键(产空值/默认):`kb_edge_ids`(恒空)、退役 kernel `reason: "retired"` 分支等。
- 各 `collect_*` 函数形态相似(读 state → 拼 dict)。可下沉公共的"安全读取 + 默认值"helper,减少重复样板。
- v1/v2 别名(`param_search`≡`explore_search`、`phase_timeline`≡`action_timeline`):**保留产出**,但产出代码可收敛为"算一次 + 复制键",别算两遍。

### 目标

净删样板,对外 JSON **逐键不变**。

## 验收

- [ ] `shared_state` / `collectors` 行数下降,总 LOC 不增。
- [ ] 同版本 resume 验证通过(state 键对称)。
- [ ] breakdown smoke + schema 护栏全绿;`golden_breakdown_keys.txt` diff 为空(顶层键不变)。
- [ ] commit:`Slim shared_state ledgers` / `Dedupe breakdown collectors (keep external schema)`。

## ⚠️ 注意

- breakdown 是本仓**唯一**保留旧 session 兼容的地方——这里只做"内部去重",**不碰对外形状**。任何键消失都是回归。
- state.json 键改名 = 破坏同版本 resume;若做命名收口,save/load 必须同时改且加一次往返验证。
