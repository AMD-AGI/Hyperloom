# Phase C · 步骤 02 — 事实性错误注释 / 文档修正

## 对象

代码已变、注释/文档没跟上的事实性错误(顺手改对,不展开重写)。

| 错误 | 位置 | 修正 |
|---|---|---|
| "4 tables" 实为 6 表 | `inference_optimizer/storage/__init__.py` 头注释 | 改成实际表数/表名 |
| "Codex backend — ships in follow-up" / "critic only" | `orchestrator/backends/codex.py`(~3, 20–21) | Codex 已存在且用于 kernel;改注释 |
| KEEP 阈值注释 0.2% vs 代码 `DEFAULT_KEEP_THRESHOLD_PCT=1.0` | `orchestrator/action_executors/explore.py`(~81–86) | 注释与常量对齐 |
| CLOSE "4-step" vs 实际 5(或删 NDJSON 后重新计数) | `coordinator.py` CLOSE 注释 | 按 Phase A 删除后的实际步数改 |
| framework-agent `agent/` 包描述但不存在 | `framework-agent/SKILL.md`(10–48) | Phase A 已处理;此处复核文档与实现一致 |
| `_grid_runner.py` "backends.py/params.py stay tiny" | 模块头 | Phase A 已删退役名;复核此注释已改 |

## 操作

```bash
rg -n -e '4 tables' -e 'ships in follow-up' -e 'critic only' -e '0\.2%' -e '4-step' \
  inference_optimizer
```
逐处用 `StrReplace` 修正为当前事实。

## 同步文档(范围内)

- `kernel-agent/SKILL.md` / `critic-agent/SKILL.md` / `robustness-agent/SKILL.md` / `framework-agent/SKILL.md`:复核与 Phase A 删除项一致(auth proxy、`--with-*`、未接线 CLI、`agent/` 包)。
- 主仓 `README.md` / `docs/`:**本次范围聚焦代码 + 各 agent SKILL**;主仓大文档若有矛盾,记录但不在本相位深改(避免越清越多)。

## 验收

- [ ] 上表事实性错误全部修正。
- [ ] 各 agent SKILL 与实现一致。
- [ ] commit:`Fix stale doc/comments to match current behavior`。

## ⚠️ 注意

- 只**改对**,不**扩写**。一行能说清的别写成一段(违背主旨)。
- 文档修正不碰 §1 契约描述的语义(如 session_breakdown schema 文档),除非确实写错。
