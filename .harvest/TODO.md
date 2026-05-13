# Hyperloom CI / Inference Optimizer — 待办事项

> 滚动式 TODO，新加的放最上面。

---

## 🔴 P0 — agent chat summary ≠ ci_metrics.json，6/7 数据被 chat 误导

**症状**（2026-05-13 调研）：
对照 PrimusClaw chat 里 agent 报告的 final summary vs `/wekafs/users/<uid>/<dir>/ci_metrics.json` 的真实数字：

| Model | Chat 数（已发 luochen） | 真实 ci_metrics.json | 差距 |
|---|---|---|---|
| Mistral-7B-v0.1 | 4717 → 4751 +0.72% | **没 ci_metrics**，phase04 就停 | ❌ 没跑完 |
| Qwen2.5-Coder-14B-Instruct | 2818 → 2840 +0.80% | baseline 3573（无 optimized）| chat 用 eager baseline |
| Qwen2.5-Coder-14B-Instruct-AWQ | 559 → 1971 +252% | baseline 1448 → 1556 **+7.49%** | chat 严重虚胖 |
| Qwen3-14B-AWQ | 1420 → 1552 +9.26% | baseline 1581 → 1526 **-3.48%** | 实际倒挂 |
| DeepSeek-Coder-V2-Lite-Instruct | 4267 → 4464 +4.64% | baseline 4175 → 4256 **+1.94%** | 略夸 |
| DeepSeek-V2-Lite-Chat | 5434 → 5697 +4.85% | 全 phase 跑完，根目录无 ci_metrics（要查 phase10）| 待查 |
| Qwen2.5-Coder-32B-Instruct-AWQ | (没贴) | baseline 750 → 785 **+4.60%** | 真数据 |

**根因**：
- agent 在 chat 末尾的"final summary"是它自己写的话，会用 enforce-eager baseline + 不同 CONC 的 peak 来算 gain → 自我吹嘘
- agent 写到 ci_metrics.json 的数字是 V2 orchestrator 的 phase 1/6/8 measurement → 真实

**当前状态**：
- luochen 那 26 条已经入库，user 决定**不覆盖**（"我们没拿到数据怎么会贴给你"）→ 有真相后再人工调整
- 但下次 batch 必须直接拿 ci_metrics.json，不能再依赖 chat summary

**待办**：
- [ ] 让 GHA `collect_artifacts` 能拉到 agent 实际写的 ci_metrics.json（见下面 P0-2）
- [ ] 加一段 prompt prefix 强制 agent 在 phase 10 后 cp ci_metrics.json 到 $RESULT_DIR ← **已在 commit 里**
- [ ] 重跑 batch 验证

---

## 🔴 P0-2 — agent 写到 /wekafs/users/<uid>/<dir>/ci_metrics.json，SaFE 看不到 → artifact_count=0

**症状**：
- 7 个 task 里只有 1 个 artifact_count=2，其余 5 个 Succeeded 的也都 artifact_count=0
- 2 个 Failed 的（Mistral / 32B-AWQ）直接是"optimization report not found"
- → build_summary 表格全是 ×，luochen 推送内容空

**根因**：
- prompt 让 agent 写到 `$RESULT_DIR=/workspace/hyperloom/`
- agent 调用 V2 orchestrator，V2 自己用 `paths.py / session_paths.py` → 写到 `/wekafs/users/<uid>/<arbitrary_name>/`（每次目录名不一样）
- SaFE 只看 task 的 canonical RESULT_DIR → artifact API 返回空
- → CI collect_artifacts 拿不到任何数据

**修法（不动 V2 代码）**：
- 在 prompt prefix 里加显式步骤：phase 10 后 `ls -td /wekafs/users/$(id -u)*/* | head -1` 找 SESSION_DIR，`cp ci_metrics.json + optimization_report.md` 到 $RESULT_DIR
- ✅ 已在 commit `<待补>` 里

---

## 🟠 P1 — Baseline 用 `--enforce-eager` 导致 gain% 虚胖

**症状**：
- Run 25749785697 (2026-05-12 batch=10) 里 Qwen2.5-Coder-14B-Instruct-AWQ 报 **+252.5%** 的"优化"，其中 +240.7% 来自单纯关掉 `--enforce-eager`
- 真正的 skill / kernel / sweep 优化贡献只有 ~+3-4%
- 上次 batch 里 Qwen2.5-14B-Instruct-AWQ 的 +55.13% 极可能是同款问题

**根因**：
- `.claude/skills/inference-optimization/actions/baseline.md:51-53` 默认就给 baseline 加 `--enforce-eager`：
  ```bash
  export VLLM_EXTRA_ARGS="--max-model-len 4096 --enforce-eager"
  ```
- 自家 KB（`.claude/skills/inference-optimization/kb/entries.jsonl`）已经记录："enforce-eager = -85.8%, torch.compile + CUDA graphs is ESSENTIAL"
- KB 知道这是坑，baseline.md 仍把脚踩坑里 → 自我吹嘘式 baseline

**为什么当时这么写**：
- 防御性默认：eager 模式不会 hang / 编译超时，baseline 100% 跑得起来
- 跨版本稳定，便于复现
- (副作用) gain% 数字漂亮，对外汇报好看

**对外汇报风险**：
- 任何懂 vLLM 的人会问 "baseline 是不是带了 enforce-eager？" → 一问就漏
- 真实可用 gain 应该是 **+3-4%**（kernel + 参数 sweep）而不是 +252%

**待办**：
- [ ] 决定整改方向：
  - (A) 改 baseline 默认去掉 `--enforce-eager`（gain 数字会塌但是真功夫）
  - (B) baseline 跑两次（带 / 不带 eager），report 模板加双列
  - (C) 保持现状但 report 里加 footnote："primarily from enabling CUDA/HIP graph capture"
- [ ] 重跑校准：把这次 batch 里 Qwen2.5-Coder-14B-Instruct-AWQ + Qwen2.5-14B-Instruct-AWQ 两个 model 用「公平 baseline」重新跑一次，拿到真实 skill 增益数字
- [ ] luochen dashboard 里这两条记录要更新（或加 caveat）：
  - `manual-publish-25` (Qwen2.5-Coder-14B-Instruct-AWQ +252.50%)
  - `manual-publish-01` (Qwen2.5-14B-Instruct-AWQ +55.13%)

---

## 🟠 P1 — luochen results-service 两个 bug（已绕过，待他修）

### Bug 1: `_nullable_timestamp()` 返回类型错
- 路径：`/app/src/hyperloom_results/storage/postgres_store.py` 函数 `_nullable_timestamp`
- 现状：返回 `str`（ISO 时间串），但 asyncpg 期望 `datetime` 实例 for `timestamptz` 列
- 后果：caller 传 `submitted_at: "2026-05-13T..."` 就 `DataError` → HTTP 500
- 我们这边绕过：`manual_publish.py` 把 `submitted_at` 设成 `None`
- 修法（一处）：
  ```python
  def _nullable_timestamp(value):
      if not value:
          return None
      if isinstance(value, datetime):
          return value
      return datetime.fromisoformat(str(value))
  ```

### Bug 2: PG pod liveness probe 太严
- Pod：`primus-claw-dev/hyperloom-results-service-postgres-0`
- 现状：`pg_isready ... timed out after 1s` 反复失败 → kubelet 杀 → restart loop（restart_count=8 在 39h 内）
- Limits 1Gi mem / 500m CPU，PG 16 启动 redo recovery 时 1s 不够
- 后果：每次 PG 重启完，service 拿的旧 connection 立刻死 → `ConnectionDoesNotExistError`，间歇性 500
- 修法（建议）：
  ```yaml
  livenessProbe:
    timeoutSeconds: 5      # 1 → 5
    periodSeconds: 30
    failureThreshold: 5
  # 加一个 startupProbe 给 PG 启动期更长容忍
  ```

**待办**：
- [ ] 把这两点丢给 luochen，等他修
- [ ] 修了之后把 `manual_publish.py` 里 `submitted_at: None` 的 workaround 去掉

---

## 🟡 P2 — 3 个大模型 SaFE download Job 直接秒挂

Run 25749785697 里这 3 个 model 都没下完：
- `Qwen/Qwen3-Coder-Next` (159GB)
- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (63GB)
- `Qwen/Qwen2.5-Coder-32B-Instruct` (65GB)

错误：`Download failed after 1 attempts: Unknown error during download`（**几秒内** fail，不是 timeout）

我们这边 P2 把 wait_ready 加到 8h，但 SaFE 这边的 K8s download Job **本身**就秒挂，wait_ready 再长也没用。

**待办**：
- [ ] 找 SaFE 同事看 model-register download Job 的具体 error（K8s pod log）
- [ ] 跟 HuggingFace 大文件下载的 retry / multipart 配置确认（可能 NFS write 速度问题，或 HF API 大文件 token 问题）

---

## 🟢 P3 — 第 7 个 success model 数据没收回来

`Qwen/Qwen2.5-Coder-32B-Instruct-AWQ` GHA 显示 success 但你没贴最终数据，luochen dashboard 这边也没有它的记录。

**待办**：
- [ ] 从 SaFE artifact API 拉一下这个 task 的 `optimization_report.md` + `ci_metrics.json`，把数据补到 luochen
- [ ] 或者你直接贴一下数字，我帮 POST 一条

---

## 🔵 已完成（最近 24h）

- [x] SaFE PromptPrefix 字段 + BuildHyperloomPrompt 拼接 — commit `06d8692d`，user 已部署
- [x] Hyperloom CI P1 (SSE settle) + P2 (wait_ready 8h) — commit `be01672`
- [x] PromptPrefix wiring + 多行预防 instruction 默认值 — commit `bec66e8` + `4e997ec`
- [x] 重新触发 batch=10 → Run 25749785697 → 7/10 success（vs 上次 0/10）
- [x] 26 条数据手动 POST 到 luochen `/api/import` ✅ 全部入库
