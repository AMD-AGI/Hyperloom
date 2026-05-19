# Gap-13 — Legacy stub yamls 未清理

> 严重度: **P2 次要** (catalogue 噪声)
> 主轴影响: **主轴 A (action 表面)**
> 体检报告: `../KB_design_gaps.MD` §6 Gap-13
> 关联死代码: `Dead-A.4`

## 1. 问题描述

KB_design §3.15 §2.3 把 `dream` / `re_explore` / `comm_optimization` /
`compiler_tuning` 列为 **removed**, 由 specialist domain 替代.

实际: 4 个 yaml 仍在 `inference_optimizer/actions/_meta/`. ActionRegistry
加载它们, prompt 中 action 枚举多 4 行噪声. cli `_REAL_EXECUTORS_FULL`
不含它们 (好), 但 `tests/test_p1_2_full_action_catalogue.py` 仍断言它
们在 registry — 测试反向锁住 noise.

## 2. 现状代码 trace

```
$ ls inference_optimizer/actions/_meta/{dream,re_explore,comm_optimization,compiler_tuning}.yaml
inference_optimizer/actions/_meta/comm_optimization.yaml
inference_optimizer/actions/_meta/compiler_tuning.yaml
inference_optimizer/actions/_meta/dream.yaml
inference_optimizer/actions/_meta/re_explore.yaml
```

- yaml 仍在 (ActionRegistry 加载)
- executor `.py` 文件不存在 (从未实现)
- `_REAL_EXECUTORS_FULL` 不含 (cli 不注册)
- `FULL_ENABLED_ACTIONS` / `NO_KERNEL_ENABLED_ACTIONS` 不含
- `tests/test_p1_2_full_action_catalogue.py` 仍断言它们在 registry +
  yaml 字段格式正确

后果:
- LLM 派发这些 action → `no_executor` fail
- breakdown.capability_summary 按 family 分类时可能把它们当 "not_attempted" action
- 操作员看 cli `--help` 不会看到它们 (FULL_ENABLED_ACTIONS 不含),
  但 ActionRegistry 加载后, debug log / breakdown 可能漏出来

## 3. 设计意图

§3.15 §2.3 速查表:

```
v0.6 action            | v0.8 行为
dream / re_explore /
comm_optimization /
compiler_tuning        | removed: 此前 stub executor, 没有真实执行体.
                       | v0.8 由 specialist 类型替代 (kernel/comm/compiler
                       | 等 domain)
```

设计目的: 把"提议性"的 dream / re_explore 工作搬到 LLM specialist,
deterministic executor 表保持精简.

## 4. 根本原因

§3.15 §2.3 写了 "removed", 但 PR 拆解中没有专门 PR 删 yaml + 重构测试.
M5 / M6 specialist 落地时, 大家专注 specialist 自身, 没顺手清理.

## 5. 修复路径

### PR 5.1 — 删除 4 个 yaml

```
inference_optimizer/actions/_meta/dream.yaml          → DELETE
inference_optimizer/actions/_meta/re_explore.yaml     → DELETE
inference_optimizer/actions/_meta/comm_optimization.yaml → DELETE
inference_optimizer/actions/_meta/compiler_tuning.yaml → DELETE
```

### PR 5.2 — 修复 `tests/test_p1_2_full_action_catalogue.py`

```text
# Remove these assertions:
assert "dream" in registry
assert "re_explore" in registry
assert "comm_optimization" in registry
assert "compiler_tuning" in registry

# Add: ActionRegistry should NOT contain removed actions
for removed in ("dream", "re_explore", "comm_optimization", "compiler_tuning"):
    assert registry.get(removed) is None, f"{removed} should be removed"
```

### PR 5.3 — breakdown 渲染清理

`breakdown/collectors.py` 中按 family 分类时如有遗漏 family
"dream/re_explore/..." 也一并清除. 由 `_action_family` (~2533) 检查
覆盖.

### PR 5.4 — `session_paths.py` workspace fallback

`session_paths.py:69-91` workspace fallback 列表含 dream/re_explore 目
录名. 删除 fallback. (低优先级, 仅历史 session_dir 可能有这些子目录)

## 6. 验收口径

- [ ] 4 个 yaml 文件物理删除
- [ ] ActionRegistry.list() 不含 dream/re_explore/comm_optimization/compiler_tuning
- [ ] `tests/test_p1_2_full_action_catalogue.py` 断言反向: 这些 action
      *不在* registry
- [ ] fresh session breakdown.capability_summary 无 dream/re_explore 等
      family 行

## 7. 风险 / 回退

- **回退**: 把 yaml 加回, 测试断言改回. 无功能影响.
- **resume**: 老 session_dir 中可能有 `dream/` 等子目录, fallback 路
  径不读这些目录后, 现有 artifact 仍存在不影响 session.

## 8. 关联 gap

- 关联 `Dead-A.4` (同一项, 不同视角)
- 无依赖, 可独立做
