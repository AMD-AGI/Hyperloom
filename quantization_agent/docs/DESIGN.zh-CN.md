# Hyperloom Quantization-Agent 架构设计说明书

## 0. 文档目的与背景

立项目标：*"将 Quark 作为一个 sub agent 嵌入至 hyperloom 中…该模块也可以独立完成模型的量化"*。

本 agent 是 **Hyperloom → Quark 的薄适配层（thin adapter）**：

- 唯一入口是协程 `quantize_via_prompt(prompt, ...)`，把自然语言 prompt 喂给 Claude Agent SDK session。
- session 加载 Quark 的三个 skills（`quark-torch-ptq` / `quark-torch-result-validator` / `quark-torch-llm-eval`）与 agent 自身的 `SKILL.md`（运行时合约）。
- 自然语言到 `quant_plan` 的翻译、默认值填充、CRITICAL STOP 的呈现都由 Quark skills 完成。
- agent 自身只做两件事：按 `SKILL.md` 应答 checkpoint，扫描 workspace 校验产物。

本文档记录这套设计的 **结构、合约与权衡**；运行时行为以 `../SKILL.md` 为准。

---

## 1. Hyperloom 使用 Quark 的流程图

```
  Hyperloom 侧                              Quark 侧（只读）
  ┌────────────────────────────┐            ┌────────────────────────────┐
  │  quantization-agent        │  prompt    │                            │
  │  （薄适配层）              │ ─────────▶ │  quark-torch-ptq                 │
  │                            │            │  quark-quantization-       │
  │  · 按 SKILL.md 应答        │ ◀──── ?    │       result-validator     │
  │  · _result_collector 收产物 │  answer ─▶ │  quark-torch-llm-eval            │
  │                            │            │                            │
  │                            │ ◀───────── │  artifacts + reports       │
  └────────────────────────────┘            └────────────────────────────┘
```

三个箭头分别表示：① 首条 prompt；② Quark 三个 CRITICAL STOP 与 SKILL.md 触发的 warning checkpoint（按 SKILL.md：自动 → 默认 → 升级操作者）；③ Quark 工作完成后落盘的产物与报告。Quark 始终只读，所有 workflow 逻辑由 Quark 仓库自身维护。

---

## 2. Agent 内部调用流程

```
   调用方
     │  prompt
     ▼
   quantize_via_prompt
     │
     ▼
   ┌──────────────────────────────────────────────────┐
   │  Claude Agent SDK session (cwd = workspace)      │
   │    loads  : quantization-agent/SKILL.md          │
   │             + quark-torch-ptq / validator / llm-eval   │
   │    runs   : Intake → Plan → Manifest →           │
   │             Execute → Validate → (Eval)          │
   │    answers: checkpoints per SKILL.md             │
   └──────────────────────────────────────────────────┘
     │  artifacts on disk
     ▼
   _result_collector             ──  artifact-presence-as-truth
     │
     ▼
   QuantSkillRunResult
```

确定性代码按职责拆成几个小模块，外加一份运行时合约：

| 模块 | 职责 |
|---|---|
| `__init__.py` | 公共 API：`quantize_via_prompt` / `Assessment` / `OutcomeId` / `QuantSkillRunResult` |
| `cli.py` | 独立 CLI 入口（`python -m quantization_agent.cli`），把 stdin/argv 翻译成 `quantize_via_prompt` 调用 |
| `_runner.py` | 单次 SDK session 驱动：装载 `SKILL.md`、注入 run context、把 sdk_error 收集为字符串而非抛出 |
| `_retry.py` | 多次尝试编排：bootstrap 检查、`_decide_next_step` 决策表、`requantize_attempts.txt` 计数器、操作者 y/n |
| `_assessment.py` | `classify_attempt`（disk → OutcomeId）+ `Assessment` 数据类 + `derive_status` |
| `_outcomes.py` | `OutcomeId` 枚举 + `AUTO_RECOVER` / `AUTO_FAIL` / `ASK` / `ASK_RETRYABLE` / `SUCCESS_TAGS` / `MUST_HAVE_RECOVERS_THAT_FAIL_WITHOUT_ARTIFACT` 类别集 |
| `_result_collector.py` | 单次磁盘扫描 → `CollectedArtifacts` typed snapshot |
| `_eval.py` | `eval_report.json` 阈值判定与解析 |
| `SKILL.md`（运行时合约） | SDK 加载；checkpoint 应答策略、阶段顺序、auto-recover 矩阵、retry 协议 |

判定不依赖 SDK 退出码，而是基于落盘产物，分两层：

1. **存在性**——合约 artifact 是否齐全（决定 `failed` vs `partial`/`success`）。
2. **内容解析**——`validation_report.md` 中各 step 的 `ok` / `FAIL` / `skipped`（决定 `partial` 是否升级为 `failed`，门控细则见 §5.4）。

---

## 3. Agent 输入输出规范

Agent 对外暴露唯一的协程入口 `quantize_via_prompt`。一次最小调用如下：

```python
from quantization_agent import quantize_via_prompt

result = await quantize_via_prompt(
    "把 Qwen/Qwen3-8B 进行 mxfp4 量化，self-attention 与 kv-cache 用 fp8，"
    "排除 lm_head，输出到 /scratch/qwen3-8b-mxfp4",
    workspace="/tmp/wks-xxx",
)
print(result.run_result.status, result.run_result.quantized_model_dir)
```

输入是**自然语言 prompt**。自然语言到 `quant_plan` 的映射、默认值填充都由 Quark `quark-torch-ptq` 在 Plan checkpoint 里完成；用户在 checkpoint 里确认或调整。

### 3.1 输入参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `prompt` | str | ✅ | 自然语言量化请求；至少包含模型 + 输出目录 + 大致量化方案 |
| `workspace` | str \| Path | ✅ | agent 写中间产物的工作目录；Quark workflow 的 cwd |
| `quark_root` | str \| Path \| None | — | 显式指定 Quark checkout；缺省按 `$QUARK_ROOT` → `/scratch/kewang/workspace/Quark` 自动发现 |
| `interactive` | bool \| None | — | 是否允许 stdin 升级（CRITICAL STOP 应答）。`None` = tty 自动判断 |
| `acceptable_eval_gap` | float \| None | — | 数值评测可接受 gap（默认 0.03）。超过即触发 warning checkpoint。也可在 prompt 里写"接受 5% gap"等价表达，优先级见 §5.2 |
| `max_requantize_attempts` | int | — | Ask 类失败（§A 中 #3/#6/#16/#26）重跑上限，默认 `1`。设 `0` 即 fail-fast，不重跑；调大允许 caller 让 agent 多试几次（适合 inference_optimizer 想做整体重试预算时覆写）。受 `<workspace>/requantize_attempts.txt` 持久化计数器约束 |

**prompt 示例**（含常见字段——用户只写他在意的部分即可，未写项由 Quark skill 填默认）：

```
把 /scratch/models/Qwen3-30B-A3B 进行 mxfp4 量化：
- self-attention 用 fp8 量化
- kv-cache 用 fp8、排除 lm_head
- 校准用 pileval、128 条、序列长度 512
- 导出 HF 格式，evaluation 可接受 5% gap
- 输出到 /scratch/qwen3-30b-mxfp4
```

Quark 在 Plan checkpoint 里把 prompt 翻译成 `quant_plan` 并请求用户确认。可用 scheme 词表见 §4。

### 3.2 返回结构

调用返回 `QuantSkillRunResult`，只暴露 3 个字段（其余审计信息走日志 / `<workspace>` 内固定命名的 artifact 文件，不进返回值）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `status` | str | `success` / `partial` / `failed`（语义见下表） |
| `quantized_model_dir` | Path \| None | `success` / `partial` 时指向量化结果目录；`failed` 时为 `None` |
| `assessment` | `Assessment` | 主结论 + 多次尝试的完整轨迹（结构见下） |

`Assessment` 把"这次调用最后落在哪一类、一共跑了几次、有没有靠重跑救回来、关键数值是多少"四件事合并成一个对象，避免调用方再去翻日志：

```python
@dataclass
class Assessment:
    final:     str | None              # 最终主结论 ID（§A 中的行名）；干净一次过为 None
    attempts:  list[str | None]        # 每次量化尝试的主结论 ID，按时间顺序；len = 实际跑过的次数
    recovered: bool                    # True 当且仅当：前面有尝试失败、但后续重跑成功（即 attempts 中既有失败 ID 又以 None/成功 tag 收尾）
    eval_gap:  float | None = None     # eval 阶段的 relative_gap 数值；仅当 final ∈ {eval_gap_exceeded, eval_gap_accepted} 时填充
```

字段语义：

- **`final`**：本次调用对外公布的"主结论"。`failed` / `partial` 时是根因（如 `exec_oom`、`checkpoint_aborted`、`eval_gap_exceeded`、`upstream_change_required`、`unclassified_failure`）；`success` 但"有故事"时是叙事 tag（如 `eval_gap_accepted`）；干净成功为 `None`。所有可能取值见附录 §A。
- **`attempts`**：每次完整流水线（Intake → Execute → Validate → Eval）的主结论 ID，最早一次在 `[0]`。`len(attempts) - 1` 即额外重跑次数（受 §3.1 `max_requantize_attempts` 上限约束）。
- **`recovered`**：诊断 → 修补 → 重跑机制是否真正起作用。常被调用方用作"该不该把这次结果列入'需要人复盘'"的过滤条件。
- **`eval_gap`**：把唯一一个目前会暴露的数值字段抬到结构里，免得调用方再去 parse `<workspace>/eval_report.json`。其它结论一律为 `None`。

典型场景：

| 场景 | `final` | `attempts` | `recovered` | `eval_gap` |
|---|---|---|---|---|
| 干净一次过 | `None` | `[None]` | `False` | `None` |
| 首次 OOM、重跑成功 | `None` | `["exec_oom", None]` | `True` | `None` |
| 两次都 OOM | `"exec_oom"` | `["exec_oom", "exec_oom"]` | `False` | `None` |
| 超阈值、非交互直接 partial | `"eval_gap_exceeded"` | `["eval_gap_exceeded"]` | `False` | `0.052` |
| 超阈值、操作者接受 | `"eval_gap_accepted"` | `["eval_gap_accepted"]` | `False` | `0.041` |

调用方若要 artifact 路径，直接按 §3.3 出站合约固定命名拼：`workspace / "validation_report.md"`、`workspace / "eval_report.json"` 等；存在性 `Path.exists()` 自验。这样路径模式归调用方自己掌握，agent 改产物布局不会破坏调用方的上报字段。

**`status` 语义**（精确分档见 §5.4）

| status | 含义 | 调用方处理 |
|---|---|---|
| `success` | 全部 artifact 齐全 + MUST-validate 全 ok | 把 `quantized_model_dir` 切给下游；`assessment.final` 若非空（如 `eval_gap_accepted`），记审计 |
| `partial` | 量化模型可加载，但审计 / eval 类工件缺失或有警告 | 看 `assessment.final` 决定是否接受 |
| `failed` | 模型缺失或 MUST-validate 失败 | 视为失败，`assessment.final` 即根因 |

### 3.3 内部合约（与 Quark 的接口）

agent 与 Quark 只有**一条入站合约 + 一条出站合约**：

- **Hyperloom → Quark**：把用户 prompt 作为 SDK session 的首条消息发出；Quark 的 3 个 CRITICAL STOP 由 SKILL.md 规定的 checkpoint 应答协议自动 / 默认 / 升级处理。可选择性写 `session_context.json` 作为审计 trace。
- **Quark → Hyperloom**：合约 artifact 列表为 `session_context.json` / `model_analysis.json` / `quant_plan.json` / `run_manifest.yaml` / `<output_dir>/{config.json, *.safetensors, tokenizer*}` / `validation_report.md`；§5.2 评估开启时再加 `eval_report.md`。由 `_result_collector` 扫描并据此判定 status。

---

## 4. Scheme 词表参考

prompt 是自由文本，Quark `quark-torch-ptq` 在 Plan checkpoint 中负责把它翻译为合法 `quant_plan` 并由用户确认。本节只列**写 prompt 时常用的术语**作为参考。**权威定义见 Quark `quark-llm-ptq-workflow/SKILL.md` 与 `quant_plan.schema.json`**。

### 4.1 常用 scheme 术语

| 术语 | 含义 | 适用模块 |
|---|---|---|
| `fp8` | 8-bit 浮点（E4M3） | weight / activation / kv_cache |
| `mxfp4` | OCP MX 4-bit 浮点 | weight / activation（主要用于 MoE/MLP） |
| `int8` / `int4` | 整数量化 | weight |
| `awq` / `gptq` / `smoothquant` | 量化算法（非 dtype） | 与 `int*`/`fp8` 组合使用 |
| `none` / 不写 | 该模块保持原 dtype（bf16/fp16） | 任意 |

### 4.2 常被引用的模块名

写 prompt 时可以直接说"self-attention"、"MoE expert"、"普通 MLP"、"kv-cache"、"lm_head"——Quark 会把它们对应到 `attention_scheme`、`layer_overrides[".*\.experts\..*"]`、`layer_overrides[".*\.mlp\.(?!experts\.|shared_expert).*"]`、`kv_cache_scheme`、`exclude_layers` 上。无需在 prompt 里写正则。

### 4.3 当前已知约束

- `kv_cache` 在 Quark 当前发布版的 `quark-llm-ptq-workflow` 示例与 `quantize_quark.py` 中只接受 `fp8` 或不启用；其他值会在 Quark Plan checkpoint 被拒。
- 若 prompt 中模块/scheme 组合 Quark 不支持，Plan checkpoint 会列出原因并要求修改。

### 4.4 默认值

用户在 prompt 没指定的字段（calibration 样本、序列长度、导出格式、evaluation_intent 等），由 Quark `quark-torch-ptq` skill 在 Plan checkpoint 内填默认并展示给用户确认。

---

## 5. 量化后的 Validation 与 Evaluation

本章四节按"检查什么 → 怎么传播 → 怎么门控"展开：

- §5.1 **结构性 validation**——4 个字节级 sanity check
- §5.2 **精度 evaluation**——源模型 vs 量化模型 PPL/accuracy 对比
- §5.3 **失败传播**——validation 结果如何变成 `status` 与进程退出
- §5.4 **门控策略 + 失败兜底**——artifact 分档、recovery 矩阵、retry 上限

### 5.1 结构性 Validation

Execute 完成后，SKILL.md 指示 SDK 按 **4 → 1 → 3 → 2**
顺序调用 Quark 的 `quark-torch-result-validator`，入参为
`source_model_dir = model_path` 与 `quantized_model_dir = output_dir`，
产物写到 `<workspace>/validation_report.md`。

| # | 子命令 | 判定依据 |
|---|---|---|
| 4 | `fuzzy` | 每个 pattern 的 `dtype_counts`；同一 pattern 出现混合 dtype 即有层未按计划量化 |
| 1 | `auxiliary` | tokenizer / generation_config 等辅助文件全部转移，无 missing / mismatched / extra |
| 3 | `config` | 剥掉 `quantization_config` 等键后 `config.json` 深度相等 |
| 2 | `md5` | `exclude_layers` 张量字节级 MD5 抽样比对 |

这些是 **结构 / 字节级 sanity check**，不覆盖数值精度（perplexity、accuracy）。

### 5.2 精度 Evaluation

§5.1 只覆盖结构；本节是**精度层**校验——直接对比原模型 vs 量化模型在同一组测试上的差距，gap 超阈值即提请用户确认。

**机制**

- SKILL.md 在 Validate 阶段后追加一段 prompt，调用 Quark `quark-torch-llm-eval` skill。
- Skill 使用**当前环境中已安装的 vLLM 或 SGLang** 作为推理后端（由 skill 内部按可用性选择，优先级见 `quark-torch-llm-eval/SKILL.md`），启动两个 offline engine（源模型 + 量化模型），跑同一组测试（默认 PPL on wikitext-2 + 小规模 gsm8k 抽样）。**Skill 不做任何 `pip install`**——后端缺失即报 `eval_env_unavailable`（见附录 §A.8），由调用方按 SHOULD-have 档处理。
- 产物写到 `<workspace>/eval_report.md` 与 `eval_report.json`，后者含 `source_score` / `quantized_score` / `relative_gap` 字段。

**Gap 阈值**

`acceptable_eval_gap` 按以下优先级解析（高到低）：

1. **Python 入参** — `quantize_via_prompt(..., acceptable_eval_gap=)` 显式传值
2. **Prompt 内声明** — 用户在 prompt 里写明"接受 5% gap"等
3. **SKILL.md 默认** — `0.03`（相对 gap 3%）

调用方传了 Python 入参就直接生效，不再看 prompt 与 SKILL.md。

**判定规则**

| 相对 gap | 操作者响应 | status | 备注 |
|---|---|---|---|
| `≤ 阈值` | — | `success` | 直接通过 |
| `> 阈值` | `y` | `success` | `assessment.final=eval_gap_accepted` |
| `> 阈值` | `n` | `failed` | 显式拒绝视为错误 |
| `> 阈值` | 非交互 | `partial` | `assessment.final=eval_gap_exceeded`，上游 gating 自决 |

gap 超阈值时 SKILL.md 把 gap / 单条 metric / 阈值打到 stderr 后再要求 y/n。

**Eval 阶段自身的失败**（eval 跑不起来，区别于"gap 超阈值"——完整枚举见附录 §A.8）：

| 故障 | 触发 | 处理 |
|---|---|---|
| `quantized_load_failed` | vLLM/SGLang 无法加载量化模型 | 升级为 MUST-validate `failed`（下游同样无法 serve） |
| `eval_oom` | source + quantized 同卡放不下 | 改串行加载；仍 OOM → `partial` + `assessment.final=eval_oom` |
| `eval_env_unavailable` | 环境中既无 vLLM 也无 SGLang，或测试集缺失 | `partial` + `assessment.final=eval_skipped`；不阻断 |

**Result collector 行为**

- `eval_report.md` / `eval_report.json` 归入 SHOULD-have（§5.4）——缺失只警告，不阻断。
- 调用方若需要原始 `relative_gap` 数值，可直接读 `assessment.eval_gap`（agent 已从 `<workspace>/eval_report.json` 抽出）；agent 在超阈值时把判定结果写入 `assessment.final`：非交互模式下为 `eval_gap_exceeded`（status=`partial`），操作者接受时为 `eval_gap_accepted`（status=`success`）。

### 5.3 失败传播：如何向 inference_optimizer 返回

§5.1 / §5.2 给出"是否通过";本节给出"未通过时这条信号怎么走出 agent"。校验结果经四层向上传播：

| 层 | 产出 | 给下一层的信号 |
|---|---|---|
| `quark-torch-result-validator` / `quark-torch-llm-eval` | `validation_report.md` / `eval_report.json` | 每步 `ok` / `FAIL` / `skipped`；`relative_gap` |
| `_result_collector.collect_artifacts` + `_assessment.classify_attempt` + `_assessment.derive_status` | `QuantSkillRunResult` | `status ∈ {success, partial, failed}` + `assessment.final`（按 §5.4 解析 validation_report 各 step 状态后归并到主结论 ID） |
| `quantization_request_handlers.py`（prelude 适配层） | payload dict | `success` → `status="ok"`；`partial` → 按 §5.4 四档分级 ok / failed；`failed` → `status="failed"` |
| `_run_quantization_prelude` | 进程退出 / `args.model` 改写 | `status != "ok"` 则 `SystemExit(3)`；否则继续，并把 `quantized_model_dir` 注入下游 |

适配层按 §5.4 四档分级决定 ok / failed：

| QuantRunResult.status | 触发条件 | 适配层 status | inference_optimizer 行为 |
|---|---|---|---|
| `success` | artifact 齐全 + MUST-validate 全 ok | `ok` | 继续；`args.model` ← `quantized_model_dir` |
| `partial`（SHOULD/NICE 缺） | 模型可加载、MUST-validate ok，仅审计/eval 缺 | `ok` | 继续；payload 标注 `quant_status="partial"` |
| `partial`（MUST-validate FAIL/SKIPPED） | md5 / config FAIL，或 STRICT 下 SKIPPED | `failed` | `SystemExit(3)` |
| `failed` | `run_manifest.yaml` 或量化权重缺 | `failed` | `SystemExit(3)`，stderr 打印 `assessment.final` |

早期失败（`workspace_unwritable` / `quark_root_missing` / `sdk_runtime_error` / `checkpoint_aborted` 等，完整枚举见附录 §A.1 / §A.9）同样走 `status="failed"` + `assessment.final` + `SystemExit(3)`。

> **`SystemExit(3)` 归属**：由 `_run_quantization_prelude`（CLI 适配层）抛出；直接调用 `quantize_via_prompt` 不退进程，只返回 `QuantSkillRunResult`，由调用方读 `status` 自行处置。这是 agent 独立于 inference_optimizer 可用的前提（立项目标）。

### 5.4 门控策略与失败兜底

按"下游是否真的需要"把 artifact 分四档,每档给出明确门控动作:

| 档次 | 内容 | 缺失后果 | 门控动作 |
|---|---|---|---|
| **MUST-have**(加载) | `config.json`、`*.safetensors`/`*.bin`（多分片时加 `model.safetensors.index.json`）、`tokenizer*` | vLLM 无法加载 | 缺即终止 |
| **MUST-validate**(正确性) | validation_report 中 Step 3 config + Step 2 md5 = ok | 模型能加载但结果错 | FAIL 终止;SKIPPED 默认终止,`HYPERLOOM_QUANT_STRICT_VALIDATION=0` 改为只警告 |
| **SHOULD-have**(可低成本修复) | validation_report 存在 + Step 1/4 ok;`model_analysis.json` / `quant_plan.json` / `run_manifest.yaml` | 违反硬契约但可秒级修复 | FAIL → 走兜底矩阵补文件 + 重跑 validator |
| **NICE-to-have** | `session_context.json` | 缺失无害 | 仅警告 |

> **档次语义**：分档判定的是"违反时走哪条修复路径"，不是"重不重要"。Step 1 auxiliary 同样 enforce 一条硬契约——**"源目录里有的非权重文件，量化目录必须有"**（覆盖 `special_tokens_map.json` / `generation_config.json` / `chat_template.jinja` 等"源有则必须有"的文件）。它留在 SHOULD-have 不是因为允许失败，而是因为修复路径是"从 source `cp` + 重跑 Step 1"（秒级），不值得 abort 已完成的量化。SKIPPED 才走警告路径。

**失败处置概览**——`_result_collector` 解析 artifact 后，SKILL.md 中的 checkpoint 协议按附录 §A 的分类驱动 LLM 处置：

- **Auto-recover**（§A.10 列出 13 条）：LLM 在 sandbox 内补文件 / 重跑 validator / 修 plan，**不打扰人**。
- **Auto-fail**（10 条）：环境硬错或语义违约——立即 `failed`，不重跑（重跑无意义）。
- **Ask**（6 条）：决策点。`interactive=True` 询问操作者；`interactive=False` 时其中 4 条（#3 / #6 / #16 / #26）按 SKILL.md `Recovery` 中的 fix hypothesis **自动重跑 1 次**，其余两条（#2 缺信息、#21 验收决策）立即定性。
- **兜底 #30 `unclassified_failure`**：不在已枚举行内的失败，agent 在运行时分析（log + SKILL.md Recovery + 阶段上下文），自动落到上面三类之一。详见 §A.9。

**重跑前必须先诊断**：所有进入重跑路径的行（Ask 类 + #30）在再次调用 quark-torch-ptq 前，必须先输出一份具体可执行的 fix hypothesis，并把补丁仅落在 `<workspace>` 及 agent 自身可控范围；**不得修改 `quark_root` 下任何文件**（Quark 视为不可变上游）。流程细节见 §A.10。

每条失败的具体兜底动作与重跑触发条件见附录 §A.1–§A.9。

**重跑量化上限**：由 §3.1 入参 `max_requantize_attempts`（默认 `1`）控制，仅对 Ask 类 #3/#6/#16/#26 生效，通过 `<workspace>/requantize_attempts.txt` 持久化计数器约束。

| 模式 | 重跑前是否询问 | 计数行为 |
|---|---|---|
| `interactive=False`（CI） | 跳过询问，到达上限即 `failed` | 计数器自增；达 `max_requantize_attempts` 即停 |
| `interactive=True`（人在场） | 重跑前在 stderr 打印 fix hypothesis 请操作者 `y/n` | 同上；操作者拒绝 → `SystemExit(3)` |

**Caller 覆写**：inference_optimizer 可以传 `max_requantize_attempts=0` 让 agent 立即定性（适合全局重试预算已用完的场景），也可调高让 agent 多试几次。**Auto-recover 类（13 条）的 LLM 内部修复不受此参数限制**——那些是同 SDK session 内秒级动作，不计入 Ask 类计数。

原则：盲目重跑只是复现同一次失败；"有 SKILL.md fix hypothesis 才重跑"把每次重跑绑定到明确的修复假设，CI 在无人值守下既不会被 30 分钟量化盲跑两遍，也不会因瞬态错误失掉值得救的产物。

---

## 6. 需要操作者确认的检查点（Checkpoint 汇总）

确定性边界压到最薄,代价是把"是否继续"的判断让给操作者。本节把**所有会向操作者要 y/n / 让其改 prompt / 让其确认退出的点**一次列清,便于运维侧排班、CI 侧关停。

| # | Checkpoint | 来源 | 触发条件 | 自动跳过 | 拒绝后果 |
|---|---|---|---|---|---|
| 1 | Intake CRITICAL STOP | Quark `quark-torch-ptq` | 解析完模型结构请求确认 | prompt 写"接受 Intake 默认" + `interactive=False` | `failed`,`assessment.final=checkpoint_aborted` |
| 2 | Plan CRITICAL STOP | Quark `quark-torch-ptq` | 自然语言翻译为 `quant_plan` 后请求确认 | prompt 写"按生成方案执行";`interactive=False` 自动接受 | 同上 |
| 3 | Manifest CRITICAL STOP | Quark `quark-torch-ptq` | 执行前展示 run_manifest | prompt 写"接受默认 manifest";`interactive=False` 自动接受 | 同上 |
| 4 | Eval gap warning(§5.2) | SKILL.md | 相对 gap > `acceptable_eval_gap`(默认 3%) | `acceptable_eval_gap` 调大或 prompt 内声明 | `n` → `failed`;非交互 → `partial` + `eval_gap_exceeded` |
| 5 | Requantize warning(§A.10) | SKILL.md | Ask 类 #3/#6/#16/#26 与兜底 #30 在**完成诊断 → 形成 fix hypothesis** 之后、重跑之前请求确认 | `interactive=False` 跳过确认直接由计数器自动重跑（受 §3.1 `max_requantize_attempts` 约束）；诊断给不出 hypothesis → 不重跑直接 `failed` | `n` → `SystemExit(3)`,根因+fix hypothesis 摘要到 stderr |

**关于 `interactive` 的语义**:`quantize_via_prompt(..., interactive=...)` 是**唯一**控制以上 1–5 是否允许向 stdin 升级的开关。Auto-fail 类失败（§A.10）直接退出，不经过 checkpoint。`None`(默认)按 tty 自动判断;`True` 强开 stdin;`False` 强制走 SKILL.md 里写的"非交互回退策略"(基本上是"按 prompt / 按默认值,否则拒绝")。

**MUST-validate SKIPPED 不是 checkpoint**——§5.4 的"终止 / 警告"分支在 _result_collector 时由 `HYPERLOOM_QUANT_STRICT_VALIDATION` 决定，没有运行时 y/n；CI 在部署时设一次即可。

**CI 推荐配方**:`interactive=False` + prompt 内显式写全 Intake/Plan/Manifest 想要的选择 + `acceptable_eval_gap` 设大或为 prompt 显式声明阈值 + `HYPERLOOM_QUANT_STRICT_VALIDATION` 保持默认。这样除了真实致命错误外不会停在任何 checkpoint 上等人。

---

## 7. Agent 定位与其他 Agent 的区别

quantization-agent 是 **reactor loop 启动前的一次性 prelude**：用户带 `--quantize "<prompt>"` 启动 CLI 时，由 `inference_optimizer/cli.py::_run_quantization_prelude` 调用一次 `quantize_via_prompt`，返回 `quantized_model_dir` 并写回 `args.model`，随后退出，不参与任何 tick。

```
   用户 CLI                quantization-agent         Coordinator        reactor loop
   --quantize "<prompt>" ──►   (一次性)         ──►   (启动)        ──►  每 tick 唤起
                              返回 quantized_model_dir                   orchestration / kernel
                              写回 args.model                            critic / robustness
```

与 reactor-loop agent 的对照：

| 维度 | Reactor-loop agent | quantization-agent |
|---|---|---|
| 例子 | `orchestration` / `kernel` / `critic` / `robustness` | 本 agent |
| 生命周期 | 每个 tick 被重新唤起 | reactor loop 启动前调用一次 |
| 输出方式 | `emit_intent` → PolicyGate → dispatch | 直接以 Python `dict` 返回调用方 |
| 注册进 `AgentRole` | ✅ | ❌ |

---

## 8. Quark 仓库只读性保证

Agent 严格视 Quark 仓库为只读：`quantization-agent/SKILL.md` 与 Quark 各 workflow SKILL.md 都禁止修改 `quark/`、`examples/`、`tools/`、`docs/`、`tests/`、`pyproject.toml`、`requirements.txt`。如确需自定义脚本（新旗标、新模板），agent 写入 `<workspace>/` 并由 `run_manifest.yaml` 引用，从而保留对干净 Quark 安装的复现性。

---

## 9. 参考资料

- `quantization_agent/SKILL.md` — **运行时合约**：checkpoint 应答协议、Quark skill 调用顺序、eval gap 阈值
- `quantization_agent/__init__.py` — 公共 API（`quantize_via_prompt` / `Assessment` / `OutcomeId` / `QuantSkillRunResult`）
- `quantization_agent/cli.py` — 独立 CLI（`python -m quantization_agent.cli`）
- `quantization_agent/_runner.py` — 单次 SDK session 驱动
- `quantization_agent/_retry.py` — 多次尝试编排 + 计数器 + 操作者 y/n
- `quantization_agent/_assessment.py` — `classify_attempt` / `build_assessment` / `derive_status`
- `quantization_agent/_outcomes.py` — `OutcomeId` 枚举与类别集
- `quantization_agent/_result_collector.py` — workspace artifact 扫描（确定性边界）
- `quantization_agent/_eval.py` — `eval_report.json` 阈值判定
- `inference_optimizer/orchestrator/quantization_request_handlers.py` — 程序化派发（`quantize_via_prompt` 适配层）
- `inference_optimizer/cli.py::_run_quantization_prelude` — `--quantize "<prompt>"` pre-hook
- Quark `quark-torch-ptq/SKILL.md` — 本 agent 驱动的 PTQ workflow 合约
- Quark `quark-torch-result-validator/SKILL.md` — §5.1 四项结构性检查的实现
- Quark `quark-torch-llm-eval/SKILL.md` — §5.2 精度评测中调用的 skill
- `kernel-agent/tools/tracelens_skill_runner.py` — 架构模板

---

## 附录 A. 异常处置矩阵（按阶段分组）

本附录把 quantization-agent 全流程可能出现的失败一次列清，给出每条的**类别**（Auto-recover / Auto-fail / Ask）、**具体说明**与**CI / 交互两种模式下的处置**。原则：

- **Auto-recover**：LLM 在 sandbox 内能自愈，无需打扰人。
- **Auto-fail**：修不了或重跑无意义，立即 `failed`。
- **Ask**：决策点。CI 默认重跑（前提：SKILL.md `Recovery` 中有 fix hypothesis 且 LLM 已写出具体修复动作）；交互模式向用户请示。重跑由 `<workspace>/requantize_attempts.txt` 持久化计数器约束，上限由 §3.1 入参 `max_requantize_attempts`（默认 `1`）控制，超限即 `failed`。
- **兜底行 #30**：未列入 1–29 的任何失败统一进入 #30，agent 运行时诊断后映射到上面三类之一。

**通用约束**：所有重跑前必须先诊断并给出**具体**的 fix hypothesis；**任何修复仅可改 `<workspace>` 内容，不得改 `quark_root` 下文件**（Quark 视为不可变上游，避免污染共享 checkout）。完整流程见 §A.10。

### A.1 阶段 1 · Pre（预检）

| # | 异常 | 类别 | 具体说明 | CI=False | interactive=True |
|---|------|------|---------|----------|------------------|
| 1 | `quark_root_missing` | Auto-fail | `quark_root` 参数对应路径不存在 / 不可读 / 非 git checkout；首次启动时探测一次 | 立即 **failed** | 立即 **failed** |
| 7 | `quark_skill_unavailable` | Auto-fail | `quark_root/.claude/skills/{quark-torch-ptq,validator,llm-eval}/SKILL.md` 任一缺失或注册失败 | 立即 **failed** | 立即 **failed** |
| 8 | `intent_parse_failed` | Auto-recover | LLM 读 prompt 后无法产出合法 `hyperloom_quant_intent`：缺 model_path / target_dtype / output_dir 必填字段，或字段类型错 | LLM 内部自纠 ≤2 次；仍失败 → **failed** | LLM 自纠 2 次失败后向用户请求补全 prompt |
| 23 | `workspace_unwritable` | Auto-fail | `workspace` 路径无法 mkdir 或目录存在但 `os.access(W_OK)=False`；Pre 阶段 touch 探测一次，避免量化跑 30 分钟才暴露 | 立即 **failed**（带具体 PermissionError） | 立即 **failed** |

### A.2 阶段 2 · Intake（quark-torch-ptq Step 1）

| # | 异常 | 类别 | 具体说明 | CI=False | interactive=True |
|---|------|------|---------|----------|------------------|
| 9 | `model_path_unreachable` | Auto-fail | quark-torch-ptq Step 1 尝试装载源模型时 `model_path` 不可达 / 不可读 / 不含 `config.json` | 立即 **failed** | 立即 **failed** |
| 10 | `analysis_artifact_invalid_or_missing` | Auto-recover | `model_analysis.json` 完全没产出、或产出后 schema 不合法 / JSON 损坏 | LLM 重跑 Step 1 → 继续 | 同左 |

### A.3 阶段 3 · Plan（quark-torch-ptq Step 2）

| # | 异常 | 类别 | 具体说明 | CI=False | interactive=True |
|---|------|------|---------|----------|------------------|
| 2 | `checkpoint_aborted` | Ask | Intake/Plan/Manifest 任一 CRITICAL STOP 触发；非交互模式下 prompt 又没写「接受默认」 | prompt 写默认 → 自动接受；否则 **failed**（缺信息，重跑无用） | 询问用户接受 / 改写 prompt |
| 11 | `plan_artifact_invalid_or_missing` | Auto-recover | `quant_plan.json` 缺失或字段 schema 不合法（scheme 不在白名单、layer_overrides 正则非法等） | LLM 重跑 Step 2 → 继续 | 同左 |

### A.4 阶段 4 · Manifest（quark-torch-ptq Step 3）

| # | 异常 | 类别 | 具体说明 | CI=False | interactive=True |
|---|------|------|---------|----------|------------------|
| 12 | `manifest_artifact_invalid_or_missing` | Auto-recover | `run_manifest.yaml` 缺失，或解析后缺 `outputs.quantized_model_dir` 字段 | LLM 重跑 Step 3 → 继续 | 同左 |

### A.5 阶段 5 · Exec（quark-torch-ptq Step 4a，PTQ 计算）

| # | 异常 | 类别 | 具体说明 | CI=False | interactive=True |
|---|------|------|---------|----------|------------------|
| 3 | `exec_oom` | Ask | quark-torch-ptq Step 4a 执行 PTQ 时 GPU OOM；根因：batch 太大 / seq_len 太长 / kv_cache 量化开关问题 | LLM 减 batch/seq_len + 自动重跑 1 次 → success / **failed** | 同左 + 重跑前请用户确认参数 |
| 4 | `exec_model_load_failed` | Auto-fail | Exec 阶段装载源模型失败（与 #9 不同：路径在但权重损坏 / dtype 不支持 / 缺 transformers 依赖） | 立即 **failed** | 立即 **failed** |
| 5 | `exec_calibration_data_missing` | Auto-fail | calibration 数据集（pileval / wikitext / 自定义）不可达，或样本数为 0 | 立即 **failed** | 立即 **failed** |

### A.6 阶段 6 · Export（quark-torch-ptq Step 4b，写盘）

| # | 异常 | 类别 | 具体说明 | CI=False | interactive=True |
|---|------|------|---------|----------|------------------|
| 6 | `export_crashed` | Ask | Step 4b 序列化 quantized state_dict → safetensors 时崩溃；常见：磁盘满 / NFS 抖 / 临时文件竞争 / config 写入冲突 | LLM 自动重跑 1 次 → success / **failed** | 同左 + 重跑前确认 |

### A.7 阶段 7 · Validate（quark-torch-result-validator）

| # | 异常 | 类别 | 具体说明 | CI=False | interactive=True |
|---|------|------|---------|----------|------------------|
| 13 | `validator_self_test_failed` | Auto-fail | `run_validation.py self-test` 退出非零；Quark 验证脚本损坏或导入错误 | 立即 **failed** | 立即 **failed** |
| 14 | `must_have_config_missing_or_invalid` | Auto-recover | `<quantized_dir>/config.json` 缺失，或 JSON 损坏，或 vLLM 加载所需的核心字段（`model_type` / `architectures`）缺 | LLM 从 source 拷贝 + 补 `quantization_config` → 重跑 Step 3 → 继续 | 同左 |
| 15 | `must_have_tokenizer_missing` | Auto-recover | `tokenizer.json` / `tokenizer_config.json` 等 tokenizer 必需文件缺失 | LLM 从 source `cp` tokenizer 全集 → 重跑 Step 1 → 继续 | 同左 |
| 16 | `must_have_weights_missing` | Ask | `<quantized_dir>` 下没有 `*.safetensors` / `*.bin`；Step 4 实际没产出权重 | LLM 自动重跑 1 次 → success / **failed** | 同左 + 重跑前确认 |
| 17 | `must_validate_config_mismatch` | Auto-recover | Step 3 `config` FAIL：剥离 `quantization_config` 后仍有非量化字段差异 | LLM 修复字段（白名单加白 / 业务字段从 source 覆写）→ 重跑 Step 3 → 继续 | 同左 |
| 18 | `must_validate_md5_mismatch` | Auto-fail | Step 2 `md5` FAIL：`exclude_layers` 中声明的张量字节级 MD5 与 source 不一致——**承诺不动却动了** | 立即 **failed**（语义违约） | 立即 **failed** |
| 19 | `should_have_aux_missing` | Auto-recover | Step 1 `auxiliary` FAIL：源目录有但 quantized 目录缺的非权重文件（`special_tokens_map.json` / `generation_config.json` / `chat_template.jinja` 等） | LLM `cp` 缺失文件 → 重跑 Step 1 → 继续 | 同左 |
| 20 | `nice_to_have_skipped` | Auto-recover | PyYAML 缺失导致 manifest 解析降级；或 `session_context.json` 缺失等不影响下游的工件 | 记 note → 继续 success | 同左 |
| 25 | `validation_report_absent` | Auto-recover | `validation_report.md` 完全没产出——validator 未被 SKILL.md 调用，或调用时整体崩溃；区别于"产出但某 step FAIL" | LLM 直接调一次 `quark-torch-result-validator` → 继续 | 同左 |
| 26 | `fuzzy_check_failed` | Ask | Step 4 `fuzzy` FAIL：同一 pattern 出现混合 dtype，或 safetensors header 异常；多数是 plan 写错（mixed precision），少数是文件损坏 | LLM 比对 `model_analysis.json` 修正 plan + 自动重跑 1 次 → success / **failed** | 同左 + 重跑前确认 |
| 27 | `must_validate_skipped` | Auto-recover | Step 2 md5 或 Step 3 config 因环境原因 SKIPPED（source 不可达、quant_config 无 exclude 等）——**与 FAIL 不同**：未证实 ≠ 已违约 | `HYPERLOOM_QUANT_STRICT_VALIDATION=1`（默认）→ **failed**；`=0` → partial + warning | 同左（一次性环境变量决定） |

### A.8 阶段 8 · Eval（quark-torch-llm-eval）

| # | 异常 | 类别 | 具体说明 | CI=False | interactive=True |
|---|------|------|---------|----------|------------------|
| 21 | `eval_gap_exceeded` | Ask | quark-torch-llm-eval 跑完后 `relative_gap > acceptable_eval_gap` —— 验收决策问题，非重跑可解 | 自动 → **partial** + `eval_gap_exceeded` | 询问用户接受 partial / 重新 plan |
| 22 | `eval_env_unavailable` | Auto-recover | 环境中既无 vLLM 也无 SGLang，或测试数据集不可达、或无 GPU；**skill 不会自行安装后端** | 跳过 eval → success + note `eval_skipped` | 同左 |
| 28 | `quantized_load_failed` | Auto-fail | Eval 阶段 vLLM/SGLang 尝试加载量化模型失败（区别于 #4 装载的是源模型）——下游推理同样无法 serve，等价于 MUST-validate 违约 | 升级为 MUST-validate **failed** | 同左 |
| 29 | `eval_oom` | Auto-recover | Eval 阶段 source + quantized 双 engine 同卡放不下；区别于 #3：发生在 vLLM/SGLang 推理而非 PTQ 算子计算 | 自动改串行加载；仍 OOM → **partial** + note `eval_oom` | 同左 |

### A.9 跨阶段

| # | 异常 | 类别 | 具体说明 | CI=False | interactive=True |
|---|------|------|---------|----------|------------------|
| 24 | `sdk_runtime_error` | Auto-fail | Claude Agent SDK session 本身抛异常（rate-limit / network / context overflow / authentication）；与 Quark / 验证逻辑无关 | 立即 **failed**，附 sdk traceback 摘要 | 立即 **failed** |
| 30 | `unclassified_failure` | Auto-recover\* | 任何不匹配第 1–29 行的失败：例如 Quark 升级引入的新异常类型、未识别的 SKILL.md 输出、陌生 traceback。Agent 必须先做诊断（读 stderr / 阶段上下文 / 对应 SKILL.md 的 `Recovery` 表），再**自动决定**按哪种已知类别处置：能在 workspace 内打补丁就走 Auto-recover；适合重跑且有 fix hypothesis 走 Ask（受 §3.1 `max_requantize_attempts` 约束）；否则走 Auto-fail。**全过程禁止修改 `quark_root` 下任何文件**——Quark 视为上游不可变。 | 诊断 → 可修则修补 workspace 后重跑；不可修 → **failed** + `assessment.final=unclassified_failure`（附诊断摘要） | 同左；操作者可在 stderr 上覆写自动选定的策略 |

### A.10 分布与重跑机制

> **正交关系**：§5.4 的"档次"（MUST-have / MUST-validate / SHOULD-have / NICE-to-have）描述**违反一条契约后产生的后果**；本附录的"类别"（Auto-recover / Auto-fail / Ask）描述**处置行为**。例如 #14 `must_have_config_missing` 档次 = MUST-have（没有就不能加载），类别 = Auto-recover（LLM 能补出来）；#18 `must_validate_md5_mismatch` 档次 = MUST-validate（语义违约），类别 = Auto-fail（修不了）。

**诊断 → 修复 → 重跑流程**：所有 Ask 类（#3 / #6 / #16 / #26）以及兜底行 #30 在执行任何重跑之前，必须先走一次显式诊断：

1. **诊断**：读失败阶段的 stderr、artifacts、对应 SKILL.md 的 `Recovery` 列，定位根因。
2. **形成 fix hypothesis**：必须是**具体、可执行**的修改（例如 `batch_size 32→16`、`exclude lm_head`、`扩大 calibration 样本到 256`、`改 prompt 加 "execute generated plan"`）。空泛的"再跑一次试试"不算。
3. **应用修复**：补丁仅落在 `<workspace>` 下（修 prompt / 修 `quant_plan.json` / 调环境变量）或 agent 自身可控的状态里。**严禁修改 `quark_root` 下任何文件**——Quark 被视为不可变上游；任何"需要改 Quark 才能修"的失败一律转 Auto-fail，`assessment.final=upstream_change_required`。
4. **重跑**：用补丁后的输入重新执行失败阶段；计数器递增 1。

诊断给不出 fix hypothesis 的，立即 `failed`——不做盲目重跑。

**类别分布**（共 30 行）：

| 类别 | 行数 | 行号 |
|------|------|------|
| Auto-recover | 13 | 8, 10, 11, 12, 14, 15, 17, 19, 20, 22, 25, 27, 29 |
| Auto-fail | 10 | 1, 4, 5, 7, 9, 13, 18, 23, 24, 28 |
| Ask | 6 | 2, 3, 6, 16, 21, 26 |
| Auto-recover\* (兜底，运行时分类) | 1 | 30 |

**CI 重跑触发行**（仅 Ask 类 #3 / #6 / #16 / #26）：

| # | fix hypothesis 来源 |
|---|---------------------|
| 3 `exec_oom` | quark-torch-ptq SKILL.md Recovery：减 batch / seq_len |
| 6 `export_crashed` | 经验：清 tmp + 同参重试（多为瞬态） |
| 16 `must_have_weights_missing` | 等同 #3 / #6 的合集（PTQ 没产出权重） |
| 26 `fuzzy_check_failed` | LLM 对照 `model_analysis.json` 修正 plan 的 layer_overrides |

**不自动重跑的 Ask 类**：

- **#2** `checkpoint_aborted` — 缺的是用户决策信息，重跑还是缺。
- **#21** `eval_gap_exceeded` — 验收决策，非重跑可解；同 plan 仍是同 gap。
