# Phase 0 · 步骤 02 — 选定并固化护栏测试集

## 目标

选出一组**粗粒度特征/契约测试**作为 keep-list。后续相位:
- keep-list 中的测试**必须始终全绿**,是"功能一样"的判据;
- keep-list **之外**的细粒度单测,Phase E 才允许删/合并。

## 选取原则

护栏只覆盖**对外可观测行为**,不覆盖内部实现:
1. **E2E / 全流程**:一个 session 从 PRELUDE 走到 CLOSE 的运行测试。
2. **契约**:envelope 校验、`session_breakdown.json` schema、payload 别名、各 agent envelope 平价、`test_no_legacy_writer_sites.py`。
3. **CLI 冒烟**:`--help` + mock 全流程(见步骤 03)。

## 候选 keep-list(待确认后固化)

> 先 `find` 全量测试,再从中圈定。下面是基于结构的初选,需逐个打开确认其确实是"行为级"而非"私有实现级"。

主包 `inference_optimizer/tests/`:
- `test_coordinator_runtime.py` — 协调器全流程行为
- `test_breakdown_smoke.py` — breakdown 产出冒烟
- `test_payload_aliases.py` / `test_back_compat_legacy_field_name.py` / `test_no_legacy_writer_sites.py` — 契约
- `test_roofline_ceiling.py`(若覆盖对外报告数值)— 待确认
- breakdown schema 相关测试

各 agent:
- `critic-agent` / `robustness-agent`:envelope 平价 / `tick` 契约测试
- `framework-agent`:`fa phase-discover` 输出 schema 测试

## 操作

1. 列出全量测试清单:

```bash
find inference_optimizer kernel-agent critic-agent robustness-agent framework-agent \
  -name 'test_*.py' -not -path '*/__pycache__/*' | sort > code_cleansing_plan/phase_0_guardrails/all_tests.txt
```

2. 逐个判定,把护栏测试写入:

```
code_cleansing_plan/phase_0_guardrails/guardrail_keep_list.txt
```

每行一个测试路径,行尾注明"为什么是护栏(覆盖哪个对外行为/契约)"。

3. 跑一遍护栏,确认**当前全绿**(绿色基线):

```bash
python -m pytest $(cat code_cleansing_plan/phase_0_guardrails/guardrail_keep_list.txt) -q
```

## 验收

- [ ] `all_tests.txt`、`guardrail_keep_list.txt` 生成并提交。
- [ ] keep-list 每项有"覆盖哪个对外行为"的理由。
- [ ] keep-list 当前全绿。

## ⚠️ 注意

- 护栏测试**本身也可精简**(Phase C/E),但**不可删**。
- 若某护栏测试断言的是私有实现细节(会被 Phase B 合理改动破坏),说明它不是合格护栏——换成断言对外行为的测试,或在该相位同步更新断言(但要记录原因)。
