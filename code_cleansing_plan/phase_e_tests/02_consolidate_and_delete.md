# Phase E · 步骤 02 — 合并与删除

## 顺序:先删 STALE/PRIVATE,再合并小文件

### 1. 删 STALE / PRIVATE

```bash
# 按 test_classification.txt 中标 STALE/PRIVATE 的逐个删
```
- 整文件 STALE → `Delete` 整个文件。
- 文件内部分函数 PRIVATE → `StrReplace` 删该测试函数。
- 删后 `pytest` 整体跑通(无 import 残留、无 fixture 悬空)。

### 2. 合并小测试文件

合并标准:同一被测模块、同一主题的多个小文件 → 一个。
- 例:`test_<module>_a.py` + `test_<module>_b.py` → `test_<module>.py`。
- 合并时去重(多个文件重复测同一行为,只留一份)。

### 3. 合并文件内碎测试

- 同一行为的多个近似用例 → `@pytest.mark.parametrize` 合并。
- 重复的 setup → 共享 fixture(但**不新增复杂 fixture 框架**)。

## 量化目标

- 测试文件数显著下降。
- 测试总 LOC 下降。
- **覆盖的对外行为不减少**(GUARDRAIL + BEHAVIOR 仍覆盖所有契约)。

## 验收

- [ ] STALE/PRIVATE 全删。
- [ ] 小文件已合并;参数化已应用。
- [ ] `pytest`(范围内全部)全绿。
- [ ] 护栏 keep-list 仍存在且全绿。
- [ ] 测试文件数 / LOC 较 Phase 0 下降。
- [ ] commit:`Delete stale/private tests` → `Consolidate small test files`。

## ⚠️ 注意

- **删测试 ≠ 降覆盖对外行为**。删的是"测实现"的,不是"测契约"的。删完确认每个 §1 契约仍有至少一个测试覆盖。
- 合并不要引入庞大共享 fixture/工具层(那是把测试复杂度搬家,不是降低)。
- 整体 `pytest` 必须绿;有测试因 Phase B/D 改动而合理失败 → 判断是"测试过时(改测试)"还是"真回归(改代码)",别盲目删红灯测试。
