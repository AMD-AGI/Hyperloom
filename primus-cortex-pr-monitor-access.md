# Primus Cortex PR Monitor 接入说明

更新时间：2026-05-07 19:05 UTC+8

## 服务地址

集群内地址：

```text
REST: http://primus-cortex-pr-api.primus-cortex.svc.cluster.local/v1
MCP:  http://primus-cortex-pr-api.primus-cortex.svc.cluster.local/mcp/
```

本地调试可以先端口转发：

```bash
kubectl --kubeconfig=/tmp/core42-kubeconfig.yaml -n primus-cortex \
  port-forward svc/primus-cortex-pr-api 8088:80
```

本地访问：

```text
REST:    http://127.0.0.1:8088/v1
Swagger: http://127.0.0.1:8088/v1/docs
MCP:     http://127.0.0.1:8088/mcp/
```

MCP URL 需要保留末尾 `/`。

## 当前部署状态

当前 API 镜像：

```text
harbor.core42.primus-safe.amd.com/primussafe/primus-cortex:pr-monitor-api-mcp-202605071058
```

组件状态：

- `primus-cortex-pr-api`: 2/2 Running
- `primus-cortex-pr-monitor`: 2/2 Running
- `primus-cortex-pr-monitor-cache`: RWX PVC, 50Gi, `wekafs-storage-csi`

当前监控仓库：

- `ROCm/aiter`
- `ROCm/FlyDSL`
- `ROCm/hip`
- `ROCm/vllm`
- `sgl-project/sglang`
- `triton-lang/triton`

`ROCm/sglang` 已停用，原因是 GitHub 返回 404，已替换为 `sgl-project/sglang`。

## REST API

健康检查：

```bash
curl -s http://127.0.0.1:8088/v1/healthz
```

列出仓库：

```bash
curl -s http://127.0.0.1:8088/v1/repos
```

仓库统计：

```bash
curl -s http://127.0.0.1:8088/v1/repos/ROCm/aiter/stats
```

PR 列表：

```bash
curl -s 'http://127.0.0.1:8088/v1/repos/ROCm/aiter/prs?state=open&limit=10'
```

支持参数：

- `state`: `open`, `closed`, `merged`, `all`
- `author`: GitHub login
- `label`: label name
- `file_path`: 精确文件路径
- `since`: ISO 时间
- `until`: ISO 时间
- `before`: 分页 cursor
- `limit`: 1 到 200

PR 详情：

```bash
curl -s http://127.0.0.1:8088/v1/repos/ROCm/aiter/prs/3067
```

PR 变更文件列表：

```bash
curl -s http://127.0.0.1:8088/v1/repos/ROCm/aiter/prs/3067/files
```

这个接口返回所有变更文件的列表和元数据，不包含 patch 全文。

单文件 patch：

```bash
curl -sG http://127.0.0.1:8088/v1/repos/ROCm/aiter/prs/3067/files/by-path \
  --data-urlencode 'path=aiter/jit/optCompilerConfig.json'
```

整个 PR 的所有 patch：

```bash
curl -s http://127.0.0.1:8088/v1/repos/ROCm/aiter/prs/3067/patches
```

PR baseline 文件内容：

```bash
curl -sG http://127.0.0.1:8088/v1/repos/ROCm/aiter/prs/3067/files/baseline \
  --data-urlencode 'path=aiter/jit/optCompilerConfig.json'
```

某个 commit 下的文件内容：

```bash
curl -sG http://127.0.0.1:8088/v1/repos/ROCm/aiter/commits/<sha>/files/by-path \
  --data-urlencode 'path=aiter/jit/optCompilerConfig.json'
```

按 sha256 读取 blob：

```bash
curl -s http://127.0.0.1:8088/v1/blobs/<sha256>
```

按 commit 读取文件树快照：

```bash
curl -s http://127.0.0.1:8088/v1/repos/ROCm/aiter/commits/<sha>/files
```

搜索 PR：

```bash
curl -s 'http://127.0.0.1:8088/v1/search/prs?q=kernel&repo=ROCm/aiter&limit=10'
```

## MCP 接入

Cursor 或其他 agent 的 MCP 配置：

```json
{
  "mcpServers": {
    "primus-cortex-pr": {
      "transport": "streamable-http",
      "url": "http://127.0.0.1:8088/mcp/"
    }
  }
}
```

集群内 agent 可以使用：

```json
{
  "mcpServers": {
    "primus-cortex-pr": {
      "transport": "streamable-http",
      "url": "http://primus-cortex-pr-api.primus-cortex.svc.cluster.local/mcp/"
    }
  }
}
```

可用 tools：

- `pr_repos_list()`: 列出仓库和同步状态
- `pr_repo_stats(repo)`: 仓库统计
- `pr_list(repo, state="all", author=None, label=None, file_path=None, since=None, until=None, before=None, limit=50)`: PR 列表
- `pr_get(repo, number)`: PR 详情，包含 metadata、commits、文件列表
- `pr_files(repo, number)`: PR 文件列表，不含 patch 全文
- `pr_file_patch(repo, number, file_path)`: 单文件 patch
- `pr_patches(repo, number)`: 整个 PR 的所有 patch
- `pr_blob(sha256)`: 文件内容
- `pr_commit_files(repo, sha)`: commit 文件树快照
- `pr_commit_file(repo, sha, file_path, max_bytes=8388608)`: commit 下某个文件内容
- `pr_pr_file_baseline(repo, number, file_path, max_bytes=8388608)`: PR baseline 文件内容
- `pr_search(query, repo=None, state="all", limit=20)`: 搜索 PR 标题和正文

## 分析 PR 的推荐流程

分析一个完整 PR 时，推荐先拿概览，再按需拿内容：

1. 调 `pr_get(repo, number)` 获取 PR 标题、正文、commits 和文件列表。
2. 调 `pr_patches(repo, number)` 获取所有变更文件的 patch。
3. 对关键文件，用 `pr_pr_file_baseline(repo, number, file_path)` 获取 baseline 内容。
4. 对关键文件的新版本内容，用文件列表里的 `head_content_sha256` 调 `pr_blob(sha256)`。
5. 如果需要看任意 commit 的文件内容，用 `pr_commit_file(repo, sha, file_path)`。

示例提示词：

```text
请分析 ROCm/aiter#3067：
1. 调用 pr_get 获取 PR 概览。
2. 调用 pr_patches 获取所有 patch。
3. 对核心文件调用 pr_pr_file_baseline 获取 baseline 内容。
4. 总结 PR 目的、主要改动、风险点和建议测试项。
```

## 维护配置

修改已有仓库的回顾窗口：

```sql
UPDATE pr_monitor_repo_state
SET backfill_days = 180, last_polled_at = NULL, last_etag = NULL
WHERE repo_name = 'ROCm/hip';
```

添加新仓库：

```sql
INSERT INTO repositories (name, url, local_path, provider_id, is_private)
VALUES ('owner/repo', 'https://github.com/owner/repo.git', '', 'github', false)
ON CONFLICT (name) DO NOTHING;

INSERT INTO pr_monitor_repo_state (repo_name, is_active, poll_interval_s, backfill_days)
VALUES ('owner/repo', TRUE, 300, 30)
ON CONFLICT (repo_name) DO UPDATE
SET is_active = TRUE,
    poll_interval_s = 300,
    backfill_days = 30,
    last_polled_at = NULL,
    last_etag = NULL;
```

暂停仓库：

```sql
UPDATE pr_monitor_repo_state
SET is_active = FALSE
WHERE repo_name = 'owner/repo';
```

恢复仓库：

```sql
UPDATE pr_monitor_repo_state
SET is_active = TRUE, last_polled_at = NULL
WHERE repo_name = 'owner/repo';
```
