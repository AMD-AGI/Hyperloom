# Phase A · 步骤 05 — 退役 env / install flag / auth proxy 清除

## 删除对象

| 对象 | 位置 | 处理 |
|---|---|---|
| auth proxy(:4002)残留 | `kernel-agent/scripts/install.sh`(~236)、`kernel-agent/SKILL.md`(~110)、robustness `auth_proxy_unhealthy` signal | 删 proxy 安装 + 信号 + 文档 |
| `--with-*` install no-op flag | `kernel-agent/scripts/install.sh`、SKILL(58–59) | 删 no-op flag 解析 |
| `WORKSPACE_PATH` 退役语义 | `kernel-agent`(`ray_runtime.py` 73–76、`tracelens_analysis.py` ~2503–2558)、`critic-agent/cli.py` 注释 | 删 legacy fallback 分支;保留现役用途(critic skill 资产根) |
| `parallel_e2e_runner` legacy auto-baseline | `kernel-agent`(~343–393) | 删 |
| robustness `--mode legacy` loop | `robustness-agent`(SKILL 56–58、`conductor.py` `IntentEmitter` 121–140) | 删 deprecated DB writer 路径,统一 `runtime.cli tick` |
| framework-agent 未接线 CLI | `phase-fetch` / `phase-emit-proposal`(client 已移除,`framework-agent/runtime/cli.py` 仍在) | **删**。已确认 framework-agent 仅作 Hyperloom 子进程、不对外,Coordinator 只用 `phase-discover` → 删除 `phase-fetch` / `phase-emit-proposal` 子命令及其实现/handler |
| framework-agent 不存在的 `agent/` 包描述 | `framework-agent/SKILL.md`(10–48) | 删/改文档,使其匹配实际实现 |

## 操作

```bash
rg -n -e '4002' -e 'auth.proxy' -e 'with-' -e 'WORKSPACE_PATH' -e 'parallel_e2e_runner' \
  -e 'IntentEmitter' -e 'phase.fetch' -e 'phase.emit.proposal' \
  kernel-agent critic-agent robustness-agent framework-agent inference_optimizer
```

### framework-agent 不对外:审计所有 `fa` 子命令

framework-agent 仅作 Hyperloom 子进程,Coordinator 只通过 `framework_agent_client.py` 调用 **`fa phase-discover`**。
因此除"被 Coordinator 调用 / 被范围内测试或库 API 依赖"之外的 `fa` 子命令都是删除候选:

```bash
rg -n 'add_parser|subcommand|def cmd_' framework-agent/runtime/cli.py
# 交叉确认哪些被 inference_optimizer 调用
rg -n "fa " inference_optimizer/orchestrator/framework_agent_client.py
```

| `fa` 子命令 | 处理 |
|---|---|
| `phase-discover` | **保留**(Coordinator 唯一接线) |
| `phase-fetch` / `phase-emit-proposal` | **删**(未接线,不对外) |
| `schema` / `candidates` / `explore` / `kb` | **审计**:仅当被范围内测试 / `tools_api` 库路径依赖才留;否则删 |

## 验收

- [ ] auth proxy / `--with-*` / `parallel_e2e_runner` / `IntentEmitter` legacy 路径全删。
- [ ] `WORKSPACE_PATH` 仅保留现役用途。
- [ ] framework-agent 文档与实际实现一致。
- [ ] install.sh 仍能跑通(`--check-only` 验证)。
- [ ] commit:`Remove retired install/env paths (auth proxy, --with-*, legacy loops)`。

## ⚠️ 注意

- `install.sh` 是运维入口,改动后**务必** `bash inference_optimizer/scripts/install.sh --check-only` 验证不报缺失。
- `WORKSPACE_PATH` 在 critic-agent 仍是现役(skill 资产根),只删 kernel-agent 的退役 fallback,别一刀切。
- **framework-agent 不对外已确认**:`phase-fetch`/`phase-emit-proposal` 直接删,无需保留给独立用户。审计其余 `fa` 子命令时,唯一保留依据是"被 Coordinator 调用 / 被范围内测试或库 API 依赖";删子命令时连同其 handler、参数解析、相关文档(SKILL 描述)、专测它的测试一并删。
