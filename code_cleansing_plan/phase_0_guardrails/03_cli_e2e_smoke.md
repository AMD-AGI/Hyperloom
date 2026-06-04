# Phase 0 · 步骤 03 — CLI 全流程冒烟 + 金标准采集

## 目标

用 mock 后端跑一次完整 optimize 流程(无需真实 GPU),采集**对外可观测金标准**:
- CLI 退出码与关键 stdout 行(`Session dir : ...`、`Preflight diagnostics:` 等)
- 产物文件存在性与形状:`manifest.json` / `state.json` / `session_breakdown.json` 顶层键 / `reports/`

这些金标准在后续相位作为"功能一样"的对照(尤其 Phase B 大改后)。

## 操作

### 1. `--help` 冒烟(flag 面契约)

```bash
python -m inference_optimizer.cli --help > code_cleansing_plan/phase_0_guardrails/golden_cli_help.txt
python -m inference_optimizer.cli optimize --help >> code_cleansing_plan/phase_0_guardrails/golden_cli_help.txt
```
> 这是 §1 的 CLI flags 契约快照。Phase A/B/D 后 diff 此文件,**对外 flag 不应消失/改名**(退役 flag 除外,删除时在 Phase A 记录)。

### 2. mock 全流程

用 `--critic-mock --robustness-mock`(以及可用的 mock kernel / 不依赖真实 GPU 的最小 workload)跑通一个 session,直到 CLOSE。
> 具体命令依赖本仓 mock 能力,执行前确认 `MockCriticBackend` / `robustness_mock` / 是否有 baseline mock。若主包无法完全脱离 GPU,退而求其次:跑到尽可能靠后的相位并采集已产出的产物形状。

采集:
```bash
# session 目录下
jq 'keys' session_breakdown.json   > golden_breakdown_keys.txt
jq 'keys' state.json               > golden_state_keys.txt
jq 'keys' manifest.json            > golden_manifest_keys.txt
```

把上述金标准存入 `code_cleansing_plan/phase_0_guardrails/`。

### 3. 子进程 JSON 桥契约快照(可选但推荐)

对 critic / robustness / framework-agent 各跑一次最小 CLI,采集 envelope 顶层键:
```bash
# 例:framework-agent
fa phase-discover --request <最小请求.json> --out - | jq 'keys'
```

## 验收

- [ ] `golden_cli_help.txt` 生成。
- [ ] `golden_breakdown_keys.txt` / `golden_state_keys.txt` / `golden_manifest_keys.txt` 采集(或记录"主包无法脱离 GPU,采集到 X 相位为止")。
- [ ] 金标准全部提交。

## ⚠️ 注意

- 金标准对比的是**键/形状/退出码**,不是逐字节输出(数值会变)。
- 若环境无法跑 mock 全流程,**如实记录限制**,并把 §1 契约测试(步骤 02)作为主要护栏。不要伪造金标准。
