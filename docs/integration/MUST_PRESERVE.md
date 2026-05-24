# v0.8 资产保留清单

> 本 branch (`feature/zhenggong/lossen_explore`) 在 fork 点 `e7c2bdb` 之后 commit 的关键产物。
> 任何 F1/F2/F3 cherry-pick 操作都必须确保下表所有项**未被破坏**。
>
> **验收脚本**: `bash docs/integration/check_preserved.sh`(rc=0 = OK)
>
> 验收基线锚点: tag `pre-roofline-merge` (commit `45dd767` parent) + 1624 passed / 5 known-failed
> (见 `docs/test-baselines/known_failed_20260524.txt`)

## 1. 物理删除文件 (必须保持删除)

`origin/main` 仍然修改这些文件;merge 时若它们重新出现,意味着 v0.6 残骸复活。

| 文件 | 不要它的原因 |
|---|---|
| `inference_optimizer/orchestrator/action_executors/backends.py` | v0.6 已退役 (KB_design §3.4) |
| `inference_optimizer/orchestrator/action_executors/params.py` | 同上 |
| `inference_optimizer/orchestrator/action_executors/validate_stack.py` | M3 已内联进 explore (KB_design §3.4 §4.4) |
| `inference_optimizer/orchestrator/scoring.py` | scoreboard 已退场 (KB_design §3.9) |
| `inference_optimizer/actions/_meta/backends.yaml` | v0.6 退役 |
| `inference_optimizer/actions/_meta/params.yaml` | 同上 |
| `inference_optimizer/actions/_meta/validate_stack.yaml` | 同上 |
| `inference_optimizer/actions/validate_stack.md` | 同上 |
| `inference_optimizer/tests/test_p3_search_space_expansion.py` | 配套删除 |
| `inference_optimizer/tests/test_validate_stack.py` | 同上 |
| `inference_optimizer/tests/test_validate_stack_gate_skip.py` | 同上 |

## 2. 必须存在的核心文件

| 文件 | 验证 |
|---|---|
| `inference_optimizer/orchestrator/phase_state.py` | `wc -l` ≥ 1300, 含 `PHASE_ALLOWED_ACTIONS` |
| `inference_optimizer/orchestrator/specialist_runner.py` | 含 `class SpecialistRunner` |
| `inference_optimizer/orchestrator/specialist_subprocess.py` | 含 `class SpecialistSubprocessDispatcher` (或同类) |
| `inference_optimizer/orchestrator/specialist_domains.py` | 含 `SERVING_SPECIALIST` 等域定义 |
| `inference_optimizer/orchestrator/system_prompts/specialist_prompt_builder.py` | 存在 |

## 3. 必须存在的 Iron Rules (`SKILL.md` 段落)

- IR-1 GPU MUST be unoccupied before every launch
- IR-2 install.sh MUST succeed before every launch
- IR-3 KB + PR Monitor reachability (in-loop, soft degrade)
- IR-4 EXPLORE is specialist-first (PR-A9 Arbor-into-Hyperloom)
- IR-6 EXPLORE HARD force-exit on low budget
- IR-7 Honest self-stop via session_steward_specialist

> 注: 当前 SKILL.md **没有** IR-5 (历史上为合并/重排)。

验收: `grep -cE "^### IR-[0-9]+" inference_optimizer/SKILL.md` 应 ≥ 6

## 4. 必须存在的 PolicyGate / Coordinator 规则

| Rule ID | 文件 | 来源 |
|---|---|---|
| `action_deprecated` | `orchestrator/policy.py` | KB_gaps/Gap-10 |
| `explore_requires_specialist_provenance` | `orchestrator/policy.py` | PR-A9 (IR-4) |
| `assess_remaining_gaps_throttle` | `orchestrator/coordinator.py` | IR-7 |

验收:
```bash
grep -E "rule=['\"]?action_deprecated|rule=['\"]?explore_requires_specialist_provenance" \
    inference_optimizer/orchestrator/policy.py | wc -l   # 应 ≥ 2
grep -E "rule=['\"]?assess_remaining_gaps_throttle" \
    inference_optimizer/orchestrator/coordinator.py | wc -l   # 应 ≥ 1
```

## 5. 必须存在的 SharedState 字段

```python
# 在 inference_optimizer/orchestrator/shared_state.py 的 SharedState 类内
last_remaining_gaps_assessment: dict[str, Any]
remaining_gaps_assessments: list[dict[str, Any]]   # cap 10
steward_continuation_used: bool
specialist_domain_empty_streak: dict[str, int]
```

验收: `grep -cE "(last_remaining_gaps_assessment|remaining_gaps_assessments|steward_continuation_used|specialist_domain_empty_streak)" inference_optimizer/orchestrator/shared_state.py` ≥ 4

## 6. 必须存在的测试文件 (v0.8 独有)

| 文件 |
|---|
| `inference_optimizer/tests/test_phase_force_exit.py` |
| `inference_optimizer/tests/test_assess_remaining_gaps.py` |
| `inference_optimizer/tests/test_specialist_subprocess.py` |
| `inference_optimizer/tests/test_v08_m2_phase_machine.py` |
| `inference_optimizer/tests/test_v08_m5_specialist.py` |

## 7. Cortex KB 集成

- HTTP transport 已合入
- IR-3 soft-degrade preflight 已合入
- 验收: `grep -E "preflight_kb|cortex_kb_url|degraded_kb" inference_optimizer/scripts/install.sh inference_optimizer/cli.py | head -5` 应有命中

## 8. 测试基线

```
Pass:    >= 1624
Skipped: 1
Failed:  subset of docs/test-baselines/known_failed_20260524.txt (currently 5)
```

任何一步 cherry-pick 之后,跑 `pytest inference_optimizer/tests -q --tb=line` 必须满足上面三条。
