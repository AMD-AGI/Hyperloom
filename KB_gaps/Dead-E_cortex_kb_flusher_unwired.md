# Dead-E — `scripts/cortex_kb_flusher.py` 单飞 (未被 cli 拉起)

> 风险等级: **MEDIUM-noise** (R-01/R-02 缓解能力缺失)
> 体检报告: `../KB_design_gaps.MD` §12.6
> 关联: §3.14 R-01 / R-02 (Cortex 不可达)

## 1. 问题描述

KB_design §3.6 把 Cortex KB flusher 描述为 production-required 后台
daemon: 当 Cortex 临时不可达, NDJSON pending 累积; 后台 flusher 持续重
试 drain. 主流程不被 Cortex 拖慢.

`inference_optimizer/scripts/cortex_kb_flusher.py` 存在 (~190 LoC), 完
整实现了 daemon 行为. **但 cli.py 全文不 import / spawn 它**.

后果:
- Cortex 临时不可达时, NDJSON 兜底正常 (Coordinator T2/T3 hook 同步 enqueue)
- **但没有后台 drain**, 等到下次 session 启动才有可能补送
- 不符合 §3.6 / §3.14 R-01/R-02 缓解方案 "flusher 重试 + 主流程不被
  Cortex 拖慢"

## 2. 详细位置

| 件 | 路径 | 状态 |
|---|---|---|
| daemon 实现 | `inference_optimizer/scripts/cortex_kb_flusher.py` | ~190 LoC, 完整 |
| 文档承诺 | `KB_design/3.6_knowledge_plane/README.md`, R-01 缓解段 | flusher 必须跑 |
| 实际 cli 接线 | `cli.py` 全文 | **未 spawn / import** |
| 路径常量 | `paths.py:~88` `.kb_flusher.pid` | 定义但无人写 |

验证:

```
$ rg "cortex_kb_flusher" inference_optimizer/cli.py
(no matches)

$ rg "cortex_kb_flusher" inference_optimizer/
inference_optimizer/scripts/cortex_kb_flusher.py:1
inference_optimizer/cortex_kb_client.py:13  (注释提到, 没调用)
inference_optimizer/paths.py:~88  (.kb_flusher.pid path)
inference_optimizer/SKILL.md:?   (文档提到)
```

## 3. 设计意图

§3.6 §10 / §3.14 R-01 / R-02 缓解:

> NDJSON 兜底 (T2/T3 异步, 不阻塞); **flusher 重试**; 主流程不被
> Cortex 拖慢.

设计假设: cli 启动时 spawn flusher daemon (后台进程), 整 session 期间
持续 drain `.kb_pending.ndjson`. 当 Cortex 恢复, pending 自动消化.

## 4. 根本原因

M1 PR 链中, flusher 是 §PR8 (R-01 缓解 daemon). 实际:
- PR8 把 flusher 代码合入 `scripts/` 目录
- cli 接线 (spawn daemon, manage pid) 留在 §PR9 "production polish"
- §PR9 没合 (类似 M5 PR9 的命运)

成因: M1 上线时 reviewer 担心 daemon 调试复杂, 决定先合 daemon code,
观察手动跑的效果. 后来没人补 cli spawn.

## 5. 修复路径

### PR 5.1 — cli boot 时 spawn flusher

`cli.py` 在 `_bootstrap_cortex_kb` 后:

```text
def _maybe_spawn_kb_flusher(session_dir: Path, args) -> subprocess.Popen | None:
    """v0.8 §3.6 — spawn cortex_kb_flusher daemon for this session."""
    if args.no_cortex:
        return None
    if args.no_kb_flusher:   # NEW CLI flag, default False
        return None
    pid_path = session_dir / "runtime" / "cortex" / ".kb_flusher.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    # If a flusher is already running for this session, don't spawn again
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text())
            os.kill(pid, 0)   # check alive
            log.info("kb_flusher already running pid=%d", pid)
            return None
        except (OSError, ValueError):
            pass
    cmd = [
        sys.executable, "-m", "inference_optimizer.scripts.cortex_kb_flusher",
        "--session-dir", str(session_dir),
        "--cortex-url", args.cortex_kb_url,
        ...
    ]
    proc = subprocess.Popen(cmd, ...)
    pid_path.write_text(str(proc.pid))
    return proc
```

### PR 5.2 — 进程清理

`cli.py:_run_optimize` finally:

```text
finally:
    if kb_flusher_proc is not None:
        try:
            kb_flusher_proc.send_signal(signal.SIGTERM)
            kb_flusher_proc.wait(timeout=10)
        except Exception as e:
            log.warning("kb_flusher graceful shutdown failed: %s", e)
            kb_flusher_proc.kill()
        # Clean up pid file
        pid_path.unlink(missing_ok=True)
```

### PR 5.3 — CLI flag

`--no-kb-flusher` (默认 false). 操作员排查 NDJSON 问题时可手动关闭
flusher.

### PR 5.4 — 测试

`tests/test_v08_kb_flusher_lifecycle.py`:

- mock cortex client 永远 502 (不可达)
- cli boot 后等待 30s
- 验证 .kb_pending.ndjson 持续被 flusher 尝试 drain (pid 文件存在)
- 模拟 cortex 恢复后, .kb_pending.ndjson 行数下降到 0
- cli stop 后 pid 文件清除

### PR 5.5 — 监控 + breakdown

breakdown.kb_provenance.flusher_status: { spawned, alive, last_drain_ts,
total_drained_count, last_error }.

## 6. 验收口径

- [ ] fresh session 启动后 `.kb_flusher.pid` 存在, ps 确认 daemon 跑
- [ ] Cortex 不可达模拟下, `.kb_pending.ndjson` 不阻塞主流程
- [ ] Cortex 恢复后, pending 文件被 drain (不需要重启 session)
- [ ] session stop 时 daemon 优雅退出, pid 文件清除
- [ ] `--no-kb-flusher` 启动时, daemon 不 spawn

## 7. 风险 / 回退

- **daemon 进程泄漏**: cli crash 时 daemon 可能 orphan. 缓解: pid 文件
  机制 + 启动时 detect & kill stale daemon (PR 5.1 已含).
- **flusher 性能 / 内存**: 单 session 累积大量 pending 时可能占内存.
  flusher 应当 batched drain (现有实现已支持).
- **回退**: `--no-kb-flusher` 即时关 daemon, 退到当前状态.

## 8. 关联

- §3.14 R-01 / R-02 (Cortex 不可达) — 主要缓解能力依赖此 gap
- Gap-02 (KnowledgePlane bootstrap) — 同期 M4/M5 接线可一并完成
