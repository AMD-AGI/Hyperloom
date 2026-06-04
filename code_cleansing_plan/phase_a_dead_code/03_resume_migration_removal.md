# Phase A · 步骤 03 — 跨版本 resume 迁移读取器清除

## 决策依据(主计划 §10.3)

- **不再兼容旧版本(v0.6)session**:删所有跨版本迁移读取器与 sentinel。
- **同版本 `--resume`(崩溃恢复)= 核心功能,保留**(SKILL/robustness monitor 依赖)。
- **唯一例外**:`breakdown/` 对旧 session 的兼容(v1 别名键 + 旧 session 读取)**必须保留,不在本步删除范围**。

## 删除对象

| 对象 | 位置 |
|---|---|
| `infer_phase_from_state`(v0.6 相位推断) | `phase_state.py`(~1752) |
| `resumed_from_v06_inferred`(PHASE_EXIT_REASONS) | `phase_state.py`(~222–223) |
| `_migrate_legacy_extra_sglang_args_keys` 等 `_migrate_*` | `shared_state.py`(~211) |
| resume 路径"丢弃 stale key / 推断旧相位"的兼容分支 | `coordinator.py` resume 段(~878–1125)、`shared_state.load_or_init` |

### ❌ 不删(已确认保留)

- **`baseline_failed`** (`phase_state.py` ~211, 237):**已确认是现役失败标记**(当前逻辑的 baseline 失败路径仍用它),**不是** v0.6 sentinel。保留,勿删。
  - 删 `resumed_from_v06_inferred` 等 sentinel 时,注意**只删 v0.6 推断那一项**,`baseline_failed` 留在 `PHASE_EXIT_REASONS` / `STOP_REASON_VOCAB`。

## 操作

```bash
rg -n -e 'infer_phase_from_state' -e 'resumed_from_v06' -e '_migrate' -e 'v0\.6' -e 'v06' -e 'legacy' \
  inference_optimizer/orchestrator inference_optimizer/*.py
```

1. 删迁移函数及其调用点。
2. resume 入口改为"只接受同版本 state.json":缺字段/版本不符直接报错或当 fresh,不再尝试迁移推断。
3. **跳过 `breakdown/`**:该目录的旧 session 兼容保留。删迁移代码时若发现 breakdown 调用了某迁移 helper,保留该 helper(或在 breakdown 内联一份),不要连带删。

## 验收

- [ ] 迁移读取器/ v0.6 sentinel 全删(breakdown 内的除外)。
- [ ] 同版本 `--resume` 护栏(若有)仍绿;若无该护栏,手动验证:跑一个 mock session → kill → `--resume` 能续上。
- [ ] `breakdown/` 旧 session 兼容未被触碰(diff 确认)。
- [ ] commit:`Drop cross-version resume migration readers (keep breakdown back-compat)`。

## ⚠️ 注意(高风险点)

- **不要误删同版本 resume 逻辑**。区分:`_migrate_*` / `infer_*` / `v06` = 跨版本(删);`load_or_init` / `replay_for_resume` / `_detect_resume_state` 的主路径 = 同版本(留)。
- **`baseline_failed` 是现役失败标记,保留**(已确认)。删 sentinel 时精确到 `resumed_from_v06_inferred` / v0.6 推断,勿波及 `baseline_failed`。
- 删完务必验证一次真实的同版本 resume,这是本步唯一容易回归的地方。
