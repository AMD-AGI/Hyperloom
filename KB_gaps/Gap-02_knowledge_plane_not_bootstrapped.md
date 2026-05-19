# Gap-02 — KnowledgePlane 在生产路径未 bootstrap → pr_feed / kb_subgraph 永空

> 严重度: **P0 阻断** (设计目标不可达)
> 主轴影响: **主轴 B (知识外接 Cortex)** — KB / PR 入口侧不通
> 体检报告: `../KB_design_gaps.MD` §4 Gap-2

## 1. 问题描述

KB_design §3.6 把 KnowledgePlane 定义为 "Cortex KB + PR Monitor + 本
地源码" 三件套的统一 facade. 启动时由 CLI 构造, 注入到 Coordinator,
specialist 派发前调 `plane.pr_feed_warm(domain)` + `plane.cortex_traverse(gap)`
把外部知识装进 task.params.

实际: `cli.py::_bootstrap_knowledge_plane` 函数已经写好 (~1918-1975),
**`_run_optimize` 从来不调用它**. Coordinator 构造参数无 `knowledge_plane`,
全文 0 处引用. 即使 Gap-01 把 SpecialistRunner 接上, 它拿到的 `params.pr_feed`
也是 `None`, kb_subgraph 也是 `None`.

## 2. 现状代码 trace

### 2.1 函数定义存在

`cli.py:1916-1975`:

```text
# v0.8 M4 — PR Monitor + KnowledgePlane wiring (KB_design §3.6 + §3.13 M4)
def _bootstrap_knowledge_plane(
    args, cortex_client, ...
) -> "KnowledgePlane":
    """Construct the :class:`KnowledgePlane` facade for one session."""
    from .orchestrator.knowledge_plane import KnowledgePlane
    ...
    return KnowledgePlane.from_clients(
        cortex_kb=cortex_client,
        pr_monitor=pr_client,
        domain_repos=load_domain_repos(),
        pr_feed_window_days=window_days,
        pr_monitor_mcp_url=pr_mcp_url,
    )
```

### 2.2 调用点搜索

```
$ grep -n "_bootstrap_knowledge_plane" cli.py
1918:def _bootstrap_knowledge_plane(
2736:# (the ``--no-pr-monitor`` flag description mentions it, but never calls)
```

只有定义 + help 文字提到 KnowledgePlane, **没有任何调用点**.

### 2.3 Coordinator 不感知 KnowledgePlane

```
$ grep -n "knowledge_plane\|KnowledgePlane" orchestrator/coordinator.py
(no matches)
```

Coordinator 构造函数签名无 `knowledge_plane` 参数; 全文 0 引用.

### 2.4 SpecialistRunner 期待 plane

`specialist_runner.py:222-272`:

```text
self.knowledge_plane = knowledge_plane
...
plane = self.knowledge_plane
if plane is not None:
    pr_enabled = bool(plane.pr_monitor_enabled)
    if not pr_enabled:
        tools = [t for t in tools if not t.startswith("mcp__pr_monitor__")]
    cortex_enabled = bool(plane.cortex_enabled)
    if not cortex_enabled:
        tools = [t for t in tools if not t.startswith("mcp__cortex_kb__")]
```

`plane is None` 时直接跳过, 默认全工具集都开 → 如果 PR Monitor 实际
不可达, LLM 会调 MCP 失败 (无法在派发前 strip 工具列表).

## 3. 设计意图

- §3.6 §5 "KnowledgePlane 接口":
  ```
  cortex_traverse(canonical_id) -> sub_graph
  cortex_recipe(model_class, gpu_type) -> warm_start_recipe
  pr_feed_warm(domain, window_days) -> List[PR]
  source_grep(domain, pattern) -> List[match]
  ```
- §3.13 M4 §6: "KnowledgePlane is the v0.8 abstraction operators expect
  to bootstrap once at CLI start."
- §3.5 §6 specialist prompt 9 段: 段 5 (warm_start_recipe), 段 6 (pr_feed),
  段 7 (source_hint) 都从 KnowledgePlane 取数据.

## 4. 根本原因

M4 PR 链拆解 (KB_design M4 §PR1–PR8):

| PR | 内容 | 落地? |
|---|---|---|
| PR1 | `knowledge_plane.py` 模块 + KnowledgePlane class | ✅ |
| PR2 | `pr_monitor.py` REST 客户端 | ✅ |
| PR3 | `_domain_repos.yaml` + loader | ✅ |
| PR4 | Cortex KB readonly methods | ✅ |
| PR5 | `_bootstrap_knowledge_plane` in cli.py | ✅ |
| PR6 | **`_run_optimize` 调 PR5 函数** + 注入 Coordinator | ❌ **未落** |
| PR7 | specialist_runner 接 plane | ✅ (参数已有) |
| PR8 | CLI flags `--no-pr-monitor` 等 | ✅ (flag 接 env, env 不影响行为) |

**PR6 在合入时被拆出**, 推测原因: 当时 Coordinator `__init__` 参数已
经很长, reviewer 建议 "下个 PR 再加 knowledge_plane 参数". 这个"下个
PR"后来没人补上.

## 5. 修复路径

### PR 5.1 — `_run_optimize` 调用 `_bootstrap_knowledge_plane`

在 cli.py `_run_optimize` 适当位置 (manifest 写完, Coordinator 构造前):

```text
# v0.8 M4 — KnowledgePlane facade
if not args.no_cortex:
    plane = _bootstrap_knowledge_plane(
        args=args,
        cortex_client=cortex_client,  # 既有的 T0 客户端
        ...
    )
else:
    plane = None
```

### PR 5.2 — Coordinator 加 `knowledge_plane` 构造参数

`Coordinator.__init__` 签名加:

```text
knowledge_plane: "KnowledgePlane | None" = None,
```

`self.knowledge_plane = knowledge_plane` 存为属性. 不写其他逻辑 (仅
passthrough), 由 Gap-01 PR 5.4 的 `_handle_delegate` hook 消费.

### PR 5.3 — Coordinator transparent pass-through 到 SpecialistRunner

如果按 Gap-01 PR 5.2 路径走, cli 在 `_register_executors` 时已经把
`plane` 直接装到 SpecialistRunner 构造参数里, Coordinator 只需保存
引用供 `_handle_delegate` 用 (Gap-01 PR 5.4).

### PR 5.4 — PR Monitor warmup audit

每次 specialist 派发 + 每次 EXPLORE entry 自动 warm pr_feed:

```text
async def _on_phase_enter_explore(self):
    if self.knowledge_plane is None:
        return
    try:
        # Warm cache for the next round of specialists
        await self.knowledge_plane.pr_feed_warm_all_domains(window_days=30)
    except Exception as e:
        log.warning("pr_feed_warm failed at EXPLORE entry: %s", e)
        # graceful degrade — specialists will see empty pr_feed
```

(实际可放在 `_advance_phase_if_needed` 内的 phase=EXPLORE 分支.)

### PR 5.5 — 测试

`tests/test_v08_m4_knowledge_plane_integration.py` (新增):

- mock cortex_client + mock pr_monitor
- 启动 cli 路径
- 断言 Coordinator.knowledge_plane is not None
- 断言 phase==EXPLORE 时调过 pr_feed_warm
- 断言 specialist task params 携带 pr_feed (≥ 0 行)

## 6. 验收口径

- [ ] fresh session breakdown 中 `kb_provenance` 段 + 应当看到 PR Monitor
      数据 (R-03 探测信号反转: `pr_monitor_unreachable` warning 应当
      *不出现*)
- [ ] `--no-pr-monitor` 关闭后, specialist transcript / breakdown.warnings
      含 `pr_monitor_unreachable`
- [ ] specialist task params 含 `pr_feed` 字段 (即使空数组)
- [ ] KnowledgePlane 单元测试 (已 落) + 新增集成测全通

## 7. 风险 / 回退

- **PR Monitor 跨集群不可达** (§3.14 R-03) → KnowledgePlane 自身已设
  计 graceful degrade (`pr_monitor_enabled=False`), specialist 派出去
  pr_feed 空数组, 不阻塞.
- **回退**: `--no-pr-monitor` env flag (M4 已有) 即时关 PR Monitor; 全
  套 KnowledgePlane bootstrap 失败时 `plane=None` fallback, specialist
  退化到"只看 KB + 源码".

## 8. 关联 gap

- **解锁**: Gap-01 (SpecialistRunner 注册必须依赖 plane), Gap-03
  (specialist_done 处理)
- **同时改**: Gap-16 (KnowledgePlane help 文字与行为不一致 — 修好这个
  gap 后 help 文字就准确了)
