# Hyperloom Inference Optimization CI/CD

自动化推理优化流水线：从 InferenceX 拉取配置和基准数据，通过 Claw 调度 Agent 执行 Hyperloom Skill，生成优化报告。

## 文件结构

```
ci/
├── orchestrator.py          # 编排主程序（入口）
├── claw_client.py           # Claw API 封装（session/message/SSE/files）
├── inferenceX_parser.py     # InferenceX 配置解析 + API 数据获取
├── report_generator.py      # 报告生成（markdown + JSON + GitHub Summary）
├── ci-config.yaml           # 模型列表 + 运行配置
├── prompt_template.md       # 发给 Claw Agent 的 prompt 模板
├── inferenceX_models.yaml   # InferenceX API model 名称映射表
├── claw-integration.md      # Claw 调用流程文档（给 Claw 团队对接）
├── test_claw_flow.py        # Claw API 端到端测试脚本
├── requirements.txt         # Python 依赖
└── README.md
```

## 快速开始

```bash
pip install -r requirements.txt

# Dry-run：只生成 prompt，不执行
HARBOR_PREFIX=harbor.oci-slc.primus-safe.amd.com/proxy \
GEAK_WORKSPACE=control-plane-sandbox \
  python orchestrator.py --dry-run

# 实际执行（单个模型）
HARBOR_PREFIX=harbor.oci-slc.primus-safe.amd.com/proxy \
GEAK_WORKSPACE=control-plane-sandbox \
CLAW_API_KEY=ak-xxx \
  python orchestrator.py --models qwen3.5-bf16-mi355x-sglang --output-dir ./results

# 执行全部模型
python orchestrator.py --trigger manual --output-dir ./results
```

## 环境变量

| 变量 | 必须 | 说明 |
|------|------|------|
| `HARBOR_PREFIX` | 是 | 镜像仓库前缀，如 `harbor.oci-slc.primus-safe.amd.com/proxy` |
| `GEAK_WORKSPACE` | 是 | GEAK 执行的 workspace，如 `control-plane-sandbox` |
| `CLAW_API_KEY` | 是 | SaFE API Key（`ak-` 前缀） |
| `WEBHOOK_URL` | 否 | 通知 webhook（Slack / Teams Incoming Webhook 均兼容） |

## CLI 参数

```
python orchestrator.py [OPTIONS]

--config PATH        指定 ci-config.yaml 路径（默认同目录下）
--models KEYS        逗号分隔的模型 key，只跑子集
--trigger TYPE       触发类型：manual / scheduled / inferenceX
--dry-run            只打印 prompt 不执行
--output-dir DIR     报告输出目录（默认 ci-output/）
```

## 添加模型

1. 查 `inferenceX_models.yaml` 确认 API model 名
2. 查 InferenceX `amd-master.yaml` 确认 key
3. 确认模型已下载到 `/hyperloom/models/`
4. 在 `ci-config.yaml` 的 `models` 下添加：

```yaml
- inferenceX_key: dsr1-fp8-mi355x-sglang       # amd-master.yaml 中的 key
  inferenceX_api_name: DeepSeek-R1-0528          # InferenceX API model 参数
  model_path_override: /hyperloom/models/xxx     # 本地模型路径
  optimization_depth: full                       # full / param-only / baseline-only
  kernel_opt_backends: geak, claude              # kernel 优化后端
  target_gpu: b200                               # 对比的竞品 GPU
```

## GitHub Actions

### 配置 Secrets

Settings → Secrets and variables → Actions → New repository secret：

| Secret | 值 |
|--------|---|
| `HARBOR_PREFIX` | `harbor.oci-slc.primus-safe.amd.com/proxy` |
| `GEAK_WORKSPACE` | `control-plane-sandbox` |
| `CLAW_API_KEY` | `ak-xxx` |
| `WEBHOOK_URL` | Teams/Slack Incoming Webhook URL（可选） |

### 触发方式

- **定时**：每周一 UTC 02:00
- **手动**：Actions → Run workflow → 可选填模型子集
- **InferenceX 变更**：定时任务中检测 main 分支新 commit

### 运行方式

每个模型一个独立 Job（matrix 策略），互不影响。一个失败不影响其他。

### 输出

- **GitHub Summary**：每个模型 Job 页面显示对比表格
- **Artifact `report-{model_key}`**：每个模型的 `optimization_report.md` + 摘要
- 报告保留 90 天

## Webhook 通知

支持 Slack 和 Teams 的 Incoming Webhook，每个模型完成后发送一条通知：

```
Hyperloom CI [Qwen3.5-397B-A17B]: completed | Gain: +3.0% | Trigger: manual
```

Teams 配置：Teams Channel → Connectors → Incoming Webhook → 复制 URL → 填入 `WEBHOOK_URL` Secret。
