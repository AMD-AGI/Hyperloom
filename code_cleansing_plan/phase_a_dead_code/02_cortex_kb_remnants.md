# Phase A · 步骤 02 — Cortex / T2-T3 KB / NDJSON flusher 残留清除

## 背景

KB 体系几经演进:旧的 Cortex graph KB(T2 hypothesize / T3 verify / T4 drain)+ NDJSON flusher 守护进程已退役,
现役是 `recipe_kb/`(RecipeKB 本地快照 + Arbor 远端读)。残留的是命名漂移 + no-op + 死字段。

## 已知残留(逐个核实并清)

| 残留 | 位置 | 处理 |
|---|---|---|
| `kb_edge_ids`(恒空,T2 hypothesize 遗留) | `coordinator.py`(605–613)、`breakdown/schema.py`(670–673) | 删字段;breakdown 若需保留键给 v1 reader,产空值即可(见 §1/Phase B 边界) |
| `_cortex_t4_hook`(NDJSON drain,no-op) | `coordinator.py`(~1091) | 删 |
| `_on_enter_close` step3 NDJSON drain no-op | `coordinator.py`(~3822–3829) | 删该步,CLOSE 序列重新编号 |
| `_maybe_spawn_kb_flusher` / `_stop_kb_flusher`(写 `retired_v2_local_write`) | `cli.py`(~3463–3516) | 删整套 flusher 机制 + cortex pid 路径 |
| KnowledgePlane 的 Cortex v1 graph(`/v1/points`) | `recipe_kb/knowledge_plane.py`(19–22) | 仅保留 PR feed;删 graph 死路径 |
| 退役远端 URL / 常量 | `recipe_kb/recipe_snapshot_constants.py`(10–12, 26–31)、`cli.py`(~3338–3341) | 删 |
| 命名漂移 `cortex_kb`/`cortex_session_id`/`cortex_t0` | 全仓 | 重命名为 `recipe_*`(本次允许破坏 grep 稳定性) |
| `runtime/cortex/` skeleton | `paths.py`(121–127) | 确认无写入者后删 |

## 操作

```bash
rg -n -e 'cortex' -e 'kb_edge_ids' -e 'flusher' -e 'knowledge_plane' -e 'hypothesize' -e 'NDJSON' -e 'ndjson' \
  inference_optimizer
```

1. 先删 no-op / 死路径(flusher、t4_hook、close step3、graph)。
2. 再做命名收口 `cortex_* → recipe_*`(用 `rg` 列全,逐文件 StrReplace replace_all,注意别误改字符串里的外部契约键)。
3. breakdown 的 `kb_provenance.flusher_status` / `kb_edge_ids`:**对外键可保留产空值**(§1),但产出它的内部 flusher 代码删除。

## 验收

- [ ] `rg cortex` 仅剩(若有)breakdown 对外兼容键的字符串字面量。
- [ ] flusher / t4_hook / graph 死路径全删。
- [ ] 护栏(尤其 breakdown smoke + schema)全绿。
- [ ] commit:`Remove retired Cortex/NDJSON KB remnants` + `Rename cortex_* to recipe_*`。

## ⚠️ 注意

- `breakdown/` 的 `kb_provenance` 段是**对外契约**:删内部产生逻辑可以,但顶层键/形状若有 v1 reader 依赖则保留(产空值/默认值)。具体保留哪些键 → 对照 Phase 0 `golden_breakdown_keys.txt`。
- 命名收口波及 RecipeKB canonical id 的话要小心:**磁盘格式/canonical id 是 §1 契约**,字段内部变量名可改,但写到磁盘/远端的键名不可改。
