# Prompt templates for `@inference-optimizer`

Three ready-to-paste templates, ordered from cheapest validation to
real GPU optimisation.

---

## Template A — `mock` backend (~3 min, no API key, no GPU)

Verifies the pipeline end-to-end without spending a single LLM token.
Great for "did the last refactor break anything obvious".

```text
@inference-optimizer

我要做一次 dry-run，验证 Inference Optimizer skill 的端到端骨架（不调任何真实 LLM）。

环境：
MODEL_PATH=fake/test-model
MAX_HOURS=0.05
INFERENCE_OPTIMIZER_SESSION_ROOT=/tmp/io-test

请按以下步骤跑：

1. cd 到 src/ 目录，pip install 一次依赖：
   pip install -r inference_optimizer/requirements.txt

2. 启动 CLI（mock backend，~3 分钟跑完）：
   python -m inference_optimizer \
     --model "$MODEL_PATH" \
     --max-hours "$MAX_HOURS" \
     --backend mock \
     --reactor-tick-s 0.5 \
     --clock-tick-s 1.0 \
     --log-level INFO

3. 跑完后给我看：
   a. CLI 打印的 session_dir 路径（在 stderr banner 里）
   b. 跑 `python scripts/inspect_session.py <session_dir>` 的完整输出
   c. `cat <session_dir>/state.json` 的内容
```

预期：`reason=time_exhausted`，events 表里有 run_started /
reflection_tick / heartbeat / graceful_stop 几种 topic，每个目录还是空。

---
## Template A2 test agents
@inference-optimizer
我要测试 ClaudeBackend + 多 reactor（marathon mode 下的 4 个角色：executor + critic + watchdog + sage）。
ANTHROPIC_API_KEY=<a_k_->
ANTHROPIC_BASE_URL="https://oci-slc.primus-safe.amd.com/api/v1/llm-proxy"
MODEL_PATH=/hyperloom/models/Qwen3-30B-A3B
MAX_HOURS=7
INFERENCE_OPTIMIZER_SESSION_ROOT=/tmp/io-claude-test
我**不想**真的跑 7 小时。下面通过 reactor-tick / clock-tick 加速，只跑 0.1h（~6min），但 mode 还是会被 MAX_HOURS=7 选成 marathon_multi_agent，所以会同时起 4 个 reactor。
请这样跑：
1. 检查依赖：
   pip install -r src/inference_optimizer/requirements.txt
   node --version || echo "node not found, will need --auto-install"
   which claude || echo "claude CLI not found, will need --auto-install"
   python -c "import os; print('ANTHROPIC_API_KEY set:', bool(os.environ.get('ANTHROPIC_API_KEY')))"
2. 用 MAX_HOURS=7 选 marathon mode，但通过 timeout 在6 分钟后强制结束：
   cd src
   timeout 300 python -m inference_optimizer \
     --model "$MODEL_PATH" \
     --max-hours 7 \
     --backend claude \
     --auto-install \
     --reactor-tick-s 5.0 \
     --clock-tick-s 5.0 \
     --log-level INFO
   注意：上面 timeout 300 是为了避免真跑 7 小时。SIGTERM 会触发 graceful stop。
3. 跑完后给我看：
   a. session_dir 路径
   b. python ../scripts/inspect_session.py <session_dir>
   c. sqlite3 <session_dir>/storage/conductor.db \
        "SELECT from_agent, COUNT(*) FROM events GROUP BY from_agent"
        （应看到 conductor / clock / executor / critic / watchdog / sage 6 个 from_agent）
   d. ls -la <session_dir>/personas/ <session_dir>/findings/
   e. 如果有 policy_denied observation，捞出来：
      sqlite3 <session_dir>/storage/conductor.db \
        "SELECT payload FROM events WHERE topic='observation' 
         AND payload LIKE '%policy_denied%' LIMIT 5"
预期：
- bootstrap 成功（如果 --auto-install 触发，会装 Node + claude CLI 到 ~/.cache/inference-optimizer）
- 4 个 reactor 同时在跑，每隔 ~5s 各发一次 LLM 调用
- events 表 from_agent 分组里 4 个角色都出现
- 因为 sage / critic 是 Codex 角色但用了 ClaudeBackend（v0.7 还没接 per-role backend），可能产生一些 PolicyDenied —— 这是已知的，不是 bug
- timeout 触发后 reason=emergency 或 SIGTERM-induced 停止

---

## Template B — Local GPU + Codex (recommended for first real run)

最小化的本地 GPU 跑通示例。要求：

* AMD MI3xx / MI355X 或 NVIDIA H100 在线（`rocm-smi` / `nvidia-smi` 任一可用）
* `sglang` 或 `vllm` 已装到 `/opt/venv` 或当前 Python env
* InferenceX checkout 在 `$INFERENCEX_PATH` (含 `benchmarks/benchmark_lib.sh`)
* `OPENAI_API_KEY` 通过 env 注入（不要写进文件）

```text
@inference-optimizer

我要在本地 GPU 上跑 Inference Optimizer 的 quick mode（< 2h），
目标是把 baseline 测出来 + 跑一轮 param sweep + 看到 cumulative_gain。

环境（请在 shell 里 export，不要写进文件）：
MODEL_PATH=Qwen/Qwen3-8B
MAX_HOURS=1
TARGET_GAIN_PCT=10
INFERENCEX_PATH=/opt/InferenceX
INFERENCE_OPTIMIZER_SESSION_ROOT=/tmp/io-local-test

请按以下步骤跑：

1. 检查依赖：
   pip install -r src/inference_optimizer/requirements.txt
   python -c "import sglang, openai; print(sglang.__version__, openai.__version__)"
   rocm-smi -h >/dev/null 2>&1 || nvidia-smi -h >/dev/null 2>&1 || echo "no GPU SMI"
   ls -d "$INFERENCEX_PATH/benchmarks" || echo "InferenceX missing"
   echo "OPENAI_API_KEY set: $(test -n "$OPENAI_API_KEY" && echo yes || echo no)"

2. 启动（auto-probe 会检测 GPU 数量、自动选 sglang）：
   cd src
   python -m inference_optimizer \
     --model "$MODEL_PATH" \
     --max-hours "$MAX_HOURS" \
     --target-gain-pct "$TARGET_GAIN_PCT" \
     --inferencex-path "$INFERENCEX_PATH" \
     --backend codex \
     --codex-model gpt-5.4 \
     --reactor-tick-s 5.0 \
     --clock-tick-s 10.0 \
     --log-level INFO

3. 监控（另开 terminal）：
   bash src/inference_optimizer/scripts/monitor.sh --watch 5

4. 跑完后给我：
   a. session_dir + state.json (cumulative_gain 是否 > 0)
   b. python scripts/inspect_session.py <session_dir>
   c. ls -la <session_dir>/results/  (应有每个 task 的子目录)
   d. sqlite3 <session_dir>/storage/conductor.db \
        "SELECT task_id, kind, state, params FROM tasks ORDER BY created_at"
```

预期：
- banner 打印 `gpu count=N type=gfxXXX framework=sglang ...`
- `actions: 22 loaded` 出现
- LLM 第一轮发 `delegate(action_name=baseline)` → BaselineExecutor 真起 sglang 跑 benchmark
- state.json 中 `baseline_tput > 0`
- 后续 `delegate(bench_runner)` 或 `param_sweep_run` 让 cumulative_gain > 0%
- 一旦达到 `target_gain_pct=10`, `reason=target_reached`

---

## Template C — Marathon multi-reactor (only when you have 6h+)

```text
@inference-optimizer

我要测试 ClaudeBackend + 4 个 reactor (executor + critic + watchdog + sage)。
ANTHROPIC_API_KEY=<set in shell>
ANTHROPIC_BASE_URL="https://oci-slc.primus-safe.amd.com/api/v1/llm-proxy"
MODEL_PATH=/hyperloom/models/Qwen3-30B-A3B
MAX_HOURS=7
INFERENCEX_PATH=/opt/InferenceX
INFERENCE_OPTIMIZER_SESSION_ROOT=/tmp/io-marathon-test

通过 timeout 在 6 分钟后强制结束（不真跑 7 小时）：
   cd src
   timeout 360 python -m inference_optimizer \
     --model "$MODEL_PATH" --max-hours 7 \
     --inferencex-path "$INFERENCEX_PATH" \
     --backend claude --auto-install \
     --reactor-tick-s 5.0 --clock-tick-s 5.0 \
     --log-level INFO

跑完后给我：
   sqlite3 <session_dir>/storage/conductor.db \
     "SELECT from_agent, COUNT(*) FROM events GROUP BY from_agent"
   ls -la <session_dir>/personas/ <session_dir>/findings/
```

预期 4 个 reactor (executor / critic / watchdog / sage) 都出现在 events 表。
