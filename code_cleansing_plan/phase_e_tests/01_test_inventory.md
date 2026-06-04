# Phase E · 步骤 01 — 测试盘点与分类

## 盘点

```bash
find inference_optimizer kernel-agent critic-agent robustness-agent framework-agent \
  -name 'test_*.py' -not -path '*/__pycache__/*' -exec wc -l {} + | sort -rn
```

对照 Phase 0 的 `all_tests.txt` 与 `guardrail_keep_list.txt`。

## 四类标签(逐文件/逐测试函数打标)

| 标签 | 定义 | 处理 |
|---|---|---|
| **GUARDRAIL** | Phase 0 keep-list | 保留;只可精简不可删 |
| **BEHAVIOR** | 覆盖对外行为/契约,但不在 keep-list | 保留;可与 GUARDRAIL 合并 |
| **PRIVATE** | 断言私有 helper / 内部中间状态 / 实现细节 | **删**(主旨:不保证单元一致) |
| **STALE** | 测已删功能 / 已被 Phase B 改没的实现 | **删** |

## 已知线索

- Phase A 删功能时已顺手删的"专测退役项"测试 → 复核无残留。
- 近期 git log 已在删重复:`Remove duplicate prompt asset tests`、`Remove private prompt formatter tests` —— 延续此思路找同类。
- 大测试文件优先看:`test_explore_executor.py`(1341)、`test_profile_and_kernel_handlers.py`(3105)、`test_breakdown_smoke.py`(2765)、`test_server_patcher.py`(1426)、`test_critic_agent_backend.py`(1344)、`test_critic_verdict_map.py`(1241)、`test_roofline_executor.py`(1181)。
  - 其中测私有实现 / 重复覆盖的子用例 → 删;测对外行为的 → 留。

## 产出

```
code_cleansing_plan/phase_e_tests/test_classification.txt
```
每行:`<test 路径或 ::函数>  <标签>  <一句理由>`。

## 验收

- [ ] 全量测试已打标(GUARDRAIL/BEHAVIOR/PRIVATE/STALE)。
- [ ] `test_classification.txt` 生成提交。

## ⚠️ 注意

- 拿不准 PRIVATE vs BEHAVIOR → 看它断言的是**对外可观测**(产物键/CLI/退出码/envelope)还是**内部变量**。前者 BEHAVIOR,后者 PRIVATE。
- 不要把 GUARDRAIL 误标成 PRIVATE 删掉——keep-list 是红线。
