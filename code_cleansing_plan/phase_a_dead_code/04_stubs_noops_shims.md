# Phase A · 步骤 04 — stub / no-op / proxy / 别名 shim 清除

## 删除对象

| 对象 | 位置 | 处理 |
|---|---|---|
| `RooflineStubExecutor` + `make_roofline_stub_executor` | `orchestrator/action_executors/roofline.py`(32–35, ~160) | 确认无人 wire → 删 |
| `dynamic_action` stub(无 `claude` 时写空 proposal_set) | `orchestrator/action_executors/dynamic_action.py` | 评估:若生产路径已是 `_build_dynamic_action_executor`,stub 仅为占位 → 删或并入 |
| `_REAL_EXECUTORS_KERNEL_ONLY`(空表) | `cli.py`(~1412) | 删空表 + 相关分支 |
| `compat/payload_aliases.py`(`extra_sglang_args→extra_server_args`) | `inference_optimizer/compat/`、`kernel-agent/tools/_payload_aliases.py` | **条件删**:先确认 `test_no_legacy_writer_sites.py` 全绿(无 legacy writer),再删读侧 shim + 两份副本 |
| `params.domain` 别名 | `policy.py`(~1933)、`coordinator.py`(~8024) | 统一为 specialist tag 后删别名 |
| `apply_patch` 别名(= `integrate`) | `kernel_request_handlers.py`(2360) | 评估是否有外部 caller;无则删别名,统一 `integrate` |
| `compute_peak_from_state`(向后兼容标量包装) | `orchestrator/roofline_ceiling.py`(708–715) | 改调用点用 `compute_roofline_breakdown_from_state` → 删包装 |
| 模块级 `HYPERLOOM_KERNEL_AGENT_ROOT` 快照 | `kernel_request_handlers.py`(75–80) | 内部统一用 lazy env 读取 → 删模块级快照(确认无外部 import 它) |
| `_RetiredFlag` roofline 拼写占位 | `cli.py`(105–135, ~6080–6103) | 确认无人传 → 删 |

## 操作

```bash
rg -n -e 'Stub' -e 'payload_aliases' -e 'extra_sglang_args' -e 'compute_peak_from_state' \
  -e '_RetiredFlag' -e 'apply_patch' -e 'params.*domain' -e 'KERNEL_ONLY' \
  inference_optimizer kernel-agent
```

逐项:确认 call-site → 改调用方用新接口 → 删 shim/stub → 跑护栏。

## 验收

- [ ] 上表每项已删或标"保留+原因"。
- [ ] `extra_sglang_args` 仅(若有)留在拒绝/迁移测试。
- [ ] 护栏全绿(payload 别名相关契约测试重点看)。
- [ ] commit:按类拆,如 `Remove stub/no-op executors`、`Remove extra_sglang_args compat shim`、`Inline kernel-agent root env read`。

## ⚠️ 注意

- `payload_aliases` 删前**必须** `test_no_legacy_writer_sites.py` 绿,否则会有写侧仍产 legacy key 而读侧已删 → 静默丢数据。
- `apply_patch` / `params.domain` 是否被**子进程 JSON 桥**或外部消费 → 查 `kernel.md` 提示与 critic/robustness 协议;若是对外形状,归 §1,保留键、只清内部别名分支。
