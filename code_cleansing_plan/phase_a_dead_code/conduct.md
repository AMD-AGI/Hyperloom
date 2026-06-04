# Phase A · 行为准则

## 必须(DO)

- 每删一项前先 `rg` 列全 call-site,确认无活引用再删。
- **连带删除**:定义 + 死引用 + 相关注释 + 相关 yaml/config + 专测它的测试。半删会留下更乱的悬空引用。
- 一类退役 = 一个 commit,message 明确(`Remove retired <X>`)。
- 每个 commit 后跑护栏 + `git diff --stat` 确认净减。

## 禁止(DON'T)

- 禁止在本相位做"顺手重构/合并"(那是 Phase B)。本相位**只删**。
- 禁止删 `breakdown/` 的旧 session 兼容(§10.3 唯一例外)。
- 禁止删同版本 `--resume` 主路径(只删跨版本迁移)。
- 禁止改 §1 对外契约的**字段名/形状**;只能删产生它的内部死代码,必要时产空值保形。

## 拿不准时

- 区分不出"死代码 vs 契约/现役"——**保留 + 在该步骤文件标注"待确认+原因"**,集中到相位末尾问。
- 宁可漏删一项留到下轮,也不要误删契约导致下游静默损坏。

## 高风险清单(删前必额外验证)

1. resume 迁移(步骤 03)→ 删后跑一次真实同版本 resume。
2. `payload_aliases`(步骤 04)→ 删前 `test_no_legacy_writer_sites.py` 必绿。
3. `install.sh` 改动(步骤 05)→ `--check-only` 验证。
4. breakdown 相关键 → 对照 Phase 0 `golden_breakdown_keys.txt`。
