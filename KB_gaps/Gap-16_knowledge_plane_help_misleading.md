# Gap-16 — KnowledgePlane help 文字提示其存在但路径不通

> 严重度: **P2 次要** (cosmetic, 文档 vs 行为不一致)
> 主轴影响: 文档准确性
> 体检报告: `../KB_design_gaps.MD` §6 Gap-16

## 1. 问题描述

`cli.py` `--no-pr-monitor` flag 的 help 文字 (~2736) 描述
"switches the KnowledgePlane.pr_feed_warm behavior". 但 KnowledgePlane
实际上从未被 bootstrap (Gap-02). 操作员看 help 会以为该 flag 有效,
其实 flag 没有任何运行时效果 (因为 plane=None).

类似情况:
- `--pr-monitor-url`
- `--pr-feed-window-days`
- `--research-lane-capacity` (在 Gap-01 之前同样)

这些 flag 把 args 写入 args dict, 但**没有任何代码读取 args dict 后做
事** (因为 KnowledgePlane / SpecialistRunner 没接入).

## 2. 现状代码 trace

`cli.py:~2736` (大致):

```text
opt.add_argument(
    "--no-pr-monitor",
    action="store_true",
    help="Disable PR Monitor MCP tools for specialists. "
         "Switches the KnowledgePlane.pr_feed_warm behavior to "
         "return empty PR feeds gracefully when the monitor is "
         "unreachable.",
)
```

读取:

```text
$ rg "args.no_pr_monitor|args\.pr_monitor_url" cli.py
(only definition lines; no readers feeding into runtime)
```

(更准确说: 这些 flag 的值会被存到 `args` 对象, 但 `_bootstrap_knowledge_plane`
没被调用 (Gap-02), `SpecialistRunner` 也没被注册 (Gap-01), 因此运行时
没有任何代码消费这些值.)

## 3. 设计意图

§3.13 M4 §6 + §3.15 §5.3 CLI flag 速查:
- `--no-pr-monitor` 应当 disable PR Monitor MCP
- `--pr-monitor-url` 应当配置 PR Monitor REST endpoint
- `--pr-feed-window-days` 应当配置 PR 列表回看窗口

设计目的: 操作员通过 CLI flag 控制 KnowledgePlane 行为, 不需要改代码 /
环境变量.

## 4. 根本原因

Gap-02 (KnowledgePlane 未 bootstrap) 的下游效应. 修复 Gap-02 时, 这些
flag 的 readers 自动建立, help 文字也自动准确.

## 5. 修复路径

### 选项 A — 与 Gap-02 同 PR 修复

修 Gap-02 时, `_bootstrap_knowledge_plane` 自然消费 `args.no_pr_monitor` /
`args.pr_monitor_url` / `args.pr_feed_window_days`. help 文字无需改.

### 选项 B — 临时 disclaimer (Gap-02 之前)

在 help 文字加 deprecation/preview marker:

```text
opt.add_argument(
    "--no-pr-monitor",
    ...
    help="[v0.8 preview — wires up when KnowledgePlane is bootstrapped, "
         "see Gap-02 in KB_design_gaps.MD] "
         "Disable PR Monitor MCP tools for specialists. ...",
)
```

让操作员看 `--help` 时立刻意识到这是 preview flag.

### 推荐

选项 A — 等 Gap-02 修复时一并解决.

## 6. 验收口径

- [x] Gap-02 修复后, `cli --no-pr-monitor` 启动时, breakdown.warnings
      含 `pr_monitor:disabled` 标记 (Gap-02 同链落地; 由
      `_bootstrap_knowledge_plane` 写
      `pr_monitor_status.json`, `collect_kb_provenance` 注入
      breakdown.warnings)
- [x] `cli --pr-monitor-url=http://...` 启动时, plane 实际连该 URL
      (`test_cli_args_round_trip_into_bootstrap_knowledge_plane` 端到端 pin)
- [x] `args.pr_feed_window_days` 进入 plane 实例化
      (`test_cli_pr_feed_window_days_override_reaches_namespace` +
      端到端 round-trip 测试)

## 7. 实际落地 (2026-05-20)

Gap-16 由 Gap-02 修复时同链解决, 本次仅补一组端到端 CLI 透传测试,
以防未来 rename `--pr-feed-window-days` / `dest=` 时静默失效:

1. `tests/test_v08_m4_knowledge_plane_integration.py` 新增 §4 段, 6
   个测试覆盖:
   - 默认值 + dest 名约定 (`test_cli_pr_monitor_flags_have_expected_dest_and_defaults`)
   - `--no-pr-monitor` → `pr_monitor_enabled=False`
   - `--pr-monitor-url`, `--pr-monitor-mcp-url`,
     `--pr-feed-window-days` 三个 override 各自 round-trip 到 args
     namespace
   - End-to-end: argparse → `_bootstrap_knowledge_plane` → URL 进入
     `PRMonitorClient.from_args`, window_days +
     mcp_url 进入 `KnowledgePlane.from_clients`.
2. help 文字本身在 Gap-02 落地时已同步更新, 无需再改.

## 8. 风险 / 回退

- 选项 A 跟 Gap-02 风险一致 (已无风险)
- **回退**: 删除新增测试 + 回退 Gap-02 commit, 不影响生产路径.

## 9. 关联 gap

- 完全依赖 Gap-02 (已闭环)
- 与 Gap-01 / Gap-03 同链 (specialist 链整体已落地)
