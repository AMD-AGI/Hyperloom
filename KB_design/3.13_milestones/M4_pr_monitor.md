# M4 — PR Monitor 接入

## 1. 设计目标

把 `primus-cortex-pr-api` (REST + MCP) 接入 v0.8 的知识平面, 让
specialist (M5 起) 和 Coordinator 都能消费跨仓库的 PR 摘要 + 文件
patch + commit 文件树。

落地后用户应当看到: prompt 中出现一个 `pr_feed` 段, 列举本次 session
关心仓库的近期 PR; specialist 工具集中包含 PR Monitor MCP; Cortex KB
中可见 `pr_node` 类型的 point + `evidence url:` 引用。

## 2. 范围

**包含**:

- Coordinator 一侧: REST 客户端封装 (供 KnowledgePlane.read 使用),
  调用面: `pr_feed_warm(domain) -> list[PR_summary]`。
- specialist 一侧: MCP server 注入到 LLM 后端, 工具白名单含
  `mcp__pr_monitor__*` 全套 readonly tools。
- 配置: 关心仓库列表按 domain 配置 (kernel / framework / comm /
  compiler / system 各自仓库)。
- prompt_builder: orchestration prompt 不直接含 PR feed (仅 specialist
  prompt 含); 但 Orchestration 看到的 specialist_round_summary 可以
  包含"specialist 引用了哪些 PR"摘要。
- 把 specialist 引用 PR 时 (`specialist_done.proposal_set[i].pr_evidence`),
  Coordinator 在 T2 hypothesize 时把 PR 同时 mint 为 `pr_node` 并挂
  evidence 到 hypothetical 边 — 此功能 *预定义但留 M5 启用*, 因为 M4
  阶段还没 specialist。
- breakdown.kb_provenance.points_created 中应见 pr_node 类型 (M5 后)。
- CLI: `--pr-monitor-url`, `--no-pr-monitor`, `--pr-feed-window-days`
  (默认 30)。

**不包含**:

- specialist 框架本身 (M5)。
- Cortex KB 中 pr_node 的入图 (M5 起 specialist 引用时才入)。
- pr_intel_specialist 角色细节 (M5/M6)。

## 3. 与 M1/M3 的关系

- M1 已建 KnowledgePlane facade; M4 在 facade 的 *读* 侧加 pr_feed_warm
  能力。
- M3 已合并 explore action; M4 不动 explore 内部, 仅在 specialist 接
  入 (M5) 后通过 specialist proposal 间接影响 explore 提议。

## 4. 概念交付物

| 交付物 | 说明 |
|---|---|
| KnowledgePlane.read 扩展 | `pr_feed_warm(domain) -> list[PR_summary]`, 内部走 REST `/v1/repos/{repo}/prs?...` 取过去 N 天的 PR (state=open + state=merged 各一批), 关键字过滤后返回 |
| specialist 工具集预声明 | 工具白名单 spec 中加入 mcp__pr_monitor__* (M5 启用时绑定) |
| domain → 仓库映射 | YAML 配置 (放在 `actions/_meta/` 或 `runtime/` 下), 默认: kernel: ROCm/aiter, triton-lang/triton; framework: sgl-project/sglang, ROCm/vllm; comm: ROCm/aiter (RCCL), pytorch/pytorch (NCCL); compiler: triton-lang/triton, pytorch/pytorch; system: ROCm/hip |
| pr_node mint 路径 | KnowledgePlane.write 中加 `mint_pr_node(repo, number, url, sha)` 能力, canonical_id `pr.<repo>#<number>` |
| breakdown 段升级 | kb_provenance.points_created[] 支持 `kind=pr_node` 行 (M5 起填充) |
| CLI flags | --pr-monitor-url / --no-pr-monitor / --pr-feed-window-days |

## 5. PR feed 协议

`pr_feed_warm(domain)` 内部调用:

- 从 domain → 仓库映射拿仓库列表。
- 每个仓库: REST `/v1/repos/{repo}/prs?state=all&limit=50&since=<since>`,
  默认 since = now - window_days。
- 客户端做关键词过滤 (KB warm_start.gaps 提到的关键字 + domain 默认
  关键词集; 关键词集暂定为一份静态字典)。
- 返回 list of PR_summary (number / title / url / labels / merged_at /
  body_snippet 字段, 与 PR Monitor REST 响应一致)。

返回结果**只供 prompt 装配使用**, 不持久化到 SharedState (避免冗余;
specialist 真要看, 调 MCP)。

## 6. specialist 工具集 (M5 启用时绑定)

specialist 启动时, 工具白名单 = 默认集 + Cortex MCP (只读) +
PR Monitor MCP (全 readonly). 具体 PR Monitor 工具:

```
mcp__pr_monitor__pr_repos_list
mcp__pr_monitor__pr_repo_stats
mcp__pr_monitor__pr_list
mcp__pr_monitor__pr_get
mcp__pr_monitor__pr_files
mcp__pr_monitor__pr_file_patch
mcp__pr_monitor__pr_patches
mcp__pr_monitor__pr_blob
mcp__pr_monitor__pr_commit_files
mcp__pr_monitor__pr_commit_file
mcp__pr_monitor__pr_pr_file_baseline
mcp__pr_monitor__pr_search
```

(对应 `primus-cortex-pr-monitor-access.md` 列出的 12 个 tool, 全部
readonly。)

## 7. domain → 仓库映射

放在 `actions/_meta/_domain_repos.yaml` (新文件, 仅元数据), 概念结构:

```yaml
kernel_specialist:
  repos:
    - ROCm/aiter
    - triton-lang/triton
    - ROCm/composable_kernel
  default_keywords: [kernel, gemm, moe, attention, fmoe, ck, triton]

framework_specialist:
  repos:
    - sgl-project/sglang
    - ROCm/vllm
  default_keywords: [vllm, sglang, scheduler, cuda_graph, kv_cache]

comm_specialist:
  repos:
    - ROCm/aiter
    - pytorch/pytorch
    - NVIDIA/nccl-tests
  default_keywords: [rccl, nccl, allreduce, quickreduce, collective]

compiler_specialist:
  repos:
    - triton-lang/triton
    - pytorch/pytorch
  default_keywords: [inductor, torch.compile, codegen, triton]

system_specialist:
  repos:
    - ROCm/hip
    - ROCm/ROCm
  default_keywords: [hip, rocm, driver, kfd, dispatch]

pr_intel_specialist:
  repos: '*'   # 所有 PR Monitor 已知仓库
  default_keywords: []
```

具体仓库列表与 PR Monitor 实际监控仓库 (见
`primus-cortex-pr-monitor-access.md` §"当前监控仓库") 对齐, 缺失仓库
(例如 ROCm/composable_kernel) 通过 PR Monitor 的"添加仓库"流程
(`primus-cortex-pr-monitor-access.md` §"维护配置") 单独提请。

## 8. 实施步骤 (PR 拆分)

| PR | 内容 |
|---|---|
| 1 | KnowledgePlane.read 接口扩展 + REST 客户端 (urllib stdlib) + 关键词过滤 |
| 2 | domain → 仓库 yaml 配置 + ActionRegistry 加载 |
| 3 | prompt_builder 中 specialist prompt 模板预留 pr_feed 段 (M5 才填充) |
| 4 | KnowledgePlane.write 加 mint_pr_node + canonical 派生 |
| 5 | breakdown.kb_provenance.points_created 支持 pr_node 行 |
| 6 | CLI flag (--pr-monitor-url, --no-pr-monitor, --pr-feed-window-days) |
| 7 | MCP server URL 配置注入到 specialist 工具集 spec (M5 启用时绑定) |

PR1–PR7 均不影响 v0.6 老路径; 即使 M4 落地后 M5 还没接入, EXPLORE
依然按 M3 的 LLM-driven grid 跑, 不依赖 PR feed。

## 9. 验收清单

- [ ] `--pr-monitor-url` 配置后, 调一次 KnowledgePlane.pr_feed_warm
      ('kernel_specialist') 返回非空 list (PR Monitor 可达时)。
- [ ] PR Monitor 不可达时 (人工 kill / URL 错), pr_feed_warm 返回空,
      breakdown.warnings 加一条 "pr_monitor_unreachable" 条目, 不影响
      主流程。
- [ ] specialist 工具集 spec 中可见 mcp__pr_monitor__* 工具 (M5 启用
      后才真生效, M4 仅 spec 落地)。
- [ ] domain → 仓库 yaml 缺仓库 (例如 unknown repo) 时, KnowledgePlane
      跳过该仓库 + warning, 不挂。
- [ ] CLI `--no-pr-monitor` 完全跳过 PR feed, specialist 工具白名单
      (M5) 中也不出现 mcp__pr_monitor__* (M5 验收)。

## 10. 风险与回退

主风险:

- **PR Monitor 跨集群不可达** (本 service 在 primus-cortex 集群, 而
  marathon pod 在 data-plane 集群). 缓解: M4 文档显式承诺 prompt 中
  PR feed 段允许为空; specialist (M5) 不强依赖。长期看由平台侧推 PR
  Monitor 跨集群 ingress (与 Cortex KB 同样问题, 同样平台 RFC)。
- **REST limit 触发**: PR Monitor 默认 limit=200/请求, 不应触发 GitHub
  rate limit; 但客户端要做 backoff retry。
- **关键词过滤偏差**: 关键词集与真实 gap 不匹配, specialist 看不到相
  关 PR. 缓解: M5 起 specialist 自己可调 mcp__pr_monitor__pr_search 自
  由检索, 关键词过滤只是预热。

回退:

- 整 M4 回退即可 (KnowledgePlane.read 扩展 + yaml + spec); v0.6 / M3
  路径完全不依赖 PR Monitor。

## 11. 哲学回引

本里程碑是**主轴 B (知识外接)** 的扩展: 不仅 Cortex KB, 还包含跨仓库
PR 历史; **Inv-6.1 (三源不重复存储)** 通过 "PR Monitor 一份权威源,
Hyperloom 不镜像" 守住; **Inv-6.2 (写经中转)** 在 mint_pr_node 路径
上保持 (LLM 不直接调 PR Monitor 的写端点 — PR Monitor 本身是 readonly)。
