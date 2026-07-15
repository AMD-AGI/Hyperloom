# Hyperloom Env 治理表

治理原则：`baseline` 和 `explore` 是最基础路径；除这两步外，`framework`、`profile/roofline`、`kernel`、`sweep` 以及只读辅助步骤原则上都应允许显式跳过。跳过开关需要是显式配置，避免被旧 shell env 静默影响。

| 序号 | Env 名称 | 具体作用 | 默认状态 | 当前引用状态 | 修改建议 |
|---:|---|---|---|---|---|
| 1 | `INFERENCE_OPTIMIZER_CYCLIC_PHASES` | 控制 Coordinator 在长预算/无限预算下是否循环推进 phase，而不是一次性走完固定 phase。 | 开 | 当前运行时代码引用 | 删除。移除该 env 入口，固定由代码策略决定是否循环。 |
| 2 | `INFERENCE_OPTIMIZER_PHASE_INTERLEAVE` | 控制不同 phase 是否交错执行，影响 run 的调度顺序。 | 关 | 当前运行时代码引用 | 删除。移除该 env 入口，避免旧 shell 静默改变 phase 调度。 |
| 3 | `INFERENCE_OPTIMIZER_ENABLE_ROOFLINE` | 控制内部分析任务使用 `roofline` 还是普通 `profile`；开启时走 `profile + trace_analyze + analysis.md`，关闭时只采 trace，但不能完全跳过 profile。 | 开 | 当前运行时代码引用 | 删除。移除该 env 入口；新增完全跳过 profile 的显式开关，建议 `INFERENCE_OPTIMIZER_NO_PROFILE` / `--no-profile`，并要求只能配合 `no_kernel` 模式使用，避免无 profile 仍进入 kernel。 |
| 4 | `INFERENCE_OPTIMIZER_ENABLE_CONC_SWEEP` | 控制优化结束后是否做 baseline vs current_best 的并发 sweep。 | 开 | 当前运行时代码引用 | 删除。保留 `--enable-conc-sweep` / `--no-enable-conc-sweep` CLI，移除 env fallback。 |
| 5 | `INFERENCE_OPTIMIZER_RESEARCH_SCOUT` | 控制只读 `research_scout_specialist`：在 PRELUDE 和周期性 EXPLORE 中收集参考启动脚本、模型结构特征、跨框架经验、相关 PR 等先验，并写入 `research_hints.md`。 | 开 | 当前运行时代码引用 | 删除。保留 `--research-scout` / `--no-research-scout` CLI，移除 env fallback。 |
| 6 | `INFERENCE_OPTIMIZER_STATIC_RECON` | 控制只读 `static_recon_specialist`：扫描 framework 源码里因模型/GPU/precision 条件静默禁用快速路径的 predicate，产出 bridge candidate，不 benchmark、不 patch。 | 开 | 当前运行时代码引用 | 删除。保留 `--static-recon` / `--no-static-recon` CLI，移除 env fallback。 |
| 7 | `INFERENCE_OPTIMIZER_RECIPE_SEDIMENT` | 控制 KEEP/REVERT 结果是否沉淀进持久 recipe，供后续 warm start 避免重复尝试。 | 开 | 当前运行时代码引用 | 删除。保留 `--recipe-sediment` / `--no-recipe-sediment` CLI，移除 env fallback。 |
| 8 | `INFERENCE_OPTIMIZER_TARGET_ADVISORY` | 控制是否把外部目标差距 advisory 注入 orchestration/specialist prompt，只提供方向建议，不参与 gate。 | 开 | 当前运行时代码引用 | 删除。保留 `--target-advisory` / `--no-target-advisory` CLI，移除 env fallback。 |
| 9 | `INFERENCE_OPTIMIZER_LEGACY_ACTION_SCORES` | 控制读取旧 SharedState 时如何处理 legacy `action_scores` 字段。 | `drop` | 当前运行时代码引用 | 删除。移除该 env 入口，固定迁移行为；测试如需覆盖用直接构造输入。 |
| 10 | `INFERENCE_OPTIMIZER_MIGRATION_MODE` | 控制 SharedState schema migration 的 strict/lenient 行为。 | `strict` | 当前运行时代码引用 | 删除。移除该 env 入口，迁移失败策略固定为代码逻辑。 |
| 11 | `INFERENCE_OPTIMIZER_BREAKDOWN_INCLUDE_TRANSCRIPTS` | 控制 session breakdown 是否内联 specialist transcript body。 | 关 | 当前运行时代码引用 | 删除。移除该 env 入口，如仍需要只保留显式工具参数。 |
| 12 | `INFERENCE_OPTIMIZER_NO_FRAMEWORK` | 控制是否跳过 FRAMEWORK phase。 | 关 | 当前运行时代码引用 | 删除。保留 `--no-framework-agent` CLI 和 state 字段，移除 env fallback。 |
| 13 | `INFERENCE_OPTIMIZER_FRAMEWORK_CONFIG_EXPLORATION` | 控制 FRAMEWORK phase 里的 config exploration lane。 | 关 | 当前运行时代码引用 | 删除。移除该 env 入口，该 lane 是否启用由代码策略决定。 |
| 14 | `INFERENCE_OPTIMIZER_SWEEP_SKIP_WHEN_NO_GAIN` | 控制没有收益时是否跳过 sweep。 | 开 | 未发现当前引用 | 删除。清理表格/文档里的残留，不需要保留 env。 |
| 15 | `INFERENCE_OPTIMIZER_SATURATION_CONVERGENCE` | 历史上的饱和收敛判断开关/常量。 | N/A | 仅测试/死常量 | 删除。清理该 env 的测试或死代码痕迹。 |
| 16 | `INFERENCE_OPTIMIZER_FRAMEWORK_PLATEAU_STREAK` | 控制 framework plateau 连续轮数阈值。 | `3` | 当前运行时代码引用 | 删除。移除该 env 入口，阈值固化为代码常量。 |
| 17 | `INFERENCE_OPTIMIZER_ENABLEMENT_MAX_STALL` | 控制 enablement 路径最大停滞轮数。 | `3` | 当前运行时代码引用 | 删除。移除该 env 入口，阈值固化为代码常量。 |
| 18 | `INFERENCE_OPTIMIZER_DISPATCHER_POLL_SECONDS` | 控制 dispatcher polling 间隔。 | `10` | 当前运行时代码引用 | 删除。移除该 env 入口，避免部署环境改变调度节奏。 |
| 19 | `INFERENCE_OPTIMIZER_CHECKPOINT_MIN_TICK_GAP` | 控制 checkpoint 最小 tick 间隔。 | `3` | 当前运行时代码引用 | 删除。移除该 env 入口，改用代码常量。 |
| 20 | `INFERENCE_OPTIMIZER_RESUME_REVERIFY_BEST` | 历史上控制 resume 后是否重新验证当前 best；注意 `source="resume_reverify_best"` 仍是活的 revalidation 结果来源。 | 关 | 未发现当前 env 读取；`resume_reverify_best` source 仍被测试覆盖 | 删除 env 入口可以继续，但必须保留 `_promote_to_shared_state()` 对 `resume_reverify_best` 的清旗/水印兼容，不能把 source 分支当死代码删掉。 |
| 21 | `INFERENCE_OPTIMIZER_RESUME_DRIFT_FLOOR` | 控制 resume reverify 的 drift floor 阈值。 | `95.0` | 当前运行时代码引用 | 删除。移除该 env 入口，阈值固化为代码常量。 |
| 22 | `INFERENCE_OPTIMIZER_MIN_ENGAGED_GAIN_PCT` | 控制 engaged gain 最低收益阈值。 | `2.0` | 当前运行时代码引用 | 删除。移除该 env 入口，阈值固化为代码常量。 |
| 23 | `INFERENCE_OPTIMIZER_MEASUREMENT_DIVERGENCE_WARN_PCT` | 控制测量偏差告警阈值。 | `3.0` | 当前运行时代码引用 | 删除。移除该 env 入口，阈值固化为代码常量。 |
| 24 | `HYPERLOOM_ROOFLINE_WATERMARK_RATIO` | 控制 roofline watermark crossing 比例，影响何时重新触发分析。 | `1.10` | 当前运行时代码引用 | 删除。固化为代码常量或 roofline 内部配置，不保留 runtime env。 |
| 25 | `WARM_REPLAY_ADVISORY_CONFIDENCE` | 控制 warm replay 过滤 advisory-blocked patch 的置信度阈值。 | `0.75` | 当前运行时代码引用 | 删除。移除该 env，阈值固化为 warm replay 代码常量。 |
| 26 | `SAFE_API_KEY` | SaFE/LLM 网关主 key，并会 fan-out 到多个 provider alias。 | 未设置 | 当前运行时代码引用 | 保留。用于 secret 注入。 |
| 27 | `OPENAI_API_KEY` | OpenAI-compatible provider key。 | 未设置 | 当前运行时代码引用 | 保留。用于 provider secret 注入。 |
| 28 | `OPENAI_BASE_URL` | OpenAI-compatible LLM endpoint。 | 未设置 | 当前运行时代码引用 | 保留。用于 endpoint 注入。 |
| 29 | `OPENAI_CUSTOM_HEADERS` | OpenAI-compatible 请求额外 header，例如订阅 key。 | 未设置 | 当前运行时代码引用 | 保留。用于 gateway/header 注入。 |
| 30 | `ANTHROPIC_API_KEY` | Anthropic provider key。 | 未设置 | 当前运行时代码引用 | 保留。用于 provider secret 注入。 |
| 31 | `ANTHROPIC_AUTH_TOKEN` | Claude CLI 使用的 Anthropic auth token。 | 未设置 | 当前运行时代码引用 | 保留。用于 Claude CLI 认证。 |
| 32 | `ANTHROPIC_BASE_URL` | Anthropic/Claude endpoint。 | 未设置 | 当前运行时代码引用 | 保留。用于 endpoint 注入。 |
| 33 | `ANTHROPIC_CUSTOM_HEADERS` | Anthropic/Claude 请求额外 header。 | 未设置 | 当前运行时代码引用 | 保留。用于 gateway/header 注入。 |
| 34 | `GEAK_API_KEY` | GEAK 调用 LLM 的 key。 | 未设置 | 当前运行时代码引用 | 保留。用于 GEAK secret 注入。 |
| 35 | `GEAK_BASE_URL` | GEAK 调用 LLM 的 endpoint。 | 未设置 | 当前运行时代码引用 | 保留。用于 GEAK endpoint 注入。 |
| 36 | `LLM_API_KEY` | 通用 LLM key alias，供部分工具/子进程使用。 | 未设置 | 当前运行时代码引用 | 保留。用于兼容不同 LLM 客户端。 |
| 37 | `LLM_API_BASE` | 通用 LLM endpoint alias，供 GEAK/工具链使用。 | 未设置 | 当前运行时代码引用 | 保留。用于兼容不同 LLM 客户端。 |
| 38 | `AMD_LLM_API_KEY` | AMD LLM key alias。 | 未设置 | 当前运行时代码引用 | 保留。用于内部网关兼容。 |
| 39 | `KB_BASE_URL` | Critic live KB base URL。 | 未设置 | 当前运行时代码引用 | 保留。仅在 live KB 模式使用。 |
| 40 | `KB_SERVICE_TOKEN` | KB 服务 token。 | 未设置 | 当前运行时代码引用 | 保留。用于 KB secret 注入。 |
| 41 | `CORTEX_KB_URL` | Cortex KB assess endpoint，用于 per-proposal reasoning assess enrichment。 | 未设置 | 当前运行时代码引用 | 删除。保留 `--cortex-kb-url` CLI，endpoint 不再从 env fallback 读取。 |
| 42 | `GBRAIN_BASE_URL` | gbrain recipe KB 读侧 endpoint。 | 未设置 | 当前运行时代码引用 | 保留。用于 recipe KB endpoint 注入。 |
| 43 | `GBRAIN_TOKEN` | gbrain recipe KB token。 | 未设置 | 当前运行时代码引用 | 保留。用于 recipe KB secret 注入。 |
| 44 | `PRIMUS_CORTEX_PR_API` | Framework Agent 查询内部 Primus Cortex/PR Monitor PR 数据的 base URL；配置后 discovery 会搜索内部 PR + GitHub，未配置则 GitHub-only。 | 未设置 | 当前运行时代码引用 | 保留。作为内部 PR API canonical，`PR_MONITOR_URL` 收敛为 legacy alias。 |
| 45 | `PR_MONITOR_URL` | PR Monitor endpoint 的旧/并行命名。 | 内置默认 URL | 当前运行时代码引用 | 删除。保留 `--pr-monitor-url` CLI；内部服务名统一映射到 `PRIMUS_CORTEX_PR_API`。 |
| 46 | `SAFE_API_URL` | multi-node/Claw RayJob CLI 访问 SaFE API 的 endpoint。 | 内置 fallback/未设置 | 当前工具/CI 代码引用 | 保留。用于平台 endpoint 注入。 |
| 47 | `USER_DATA_PATH` | session 数据根目录，承载 logs/runs/mirrors/breakdown。 | `/workspace/hyperloom` 或路径推导 | 当前运行时代码引用 | 保留。作为 session 数据路径 canonical。 |
| 48 | `WORKSPACE_PATH` | legacy workspace/session fallback，也被部分 agent runtime 用作 skill asset root。 | 未设置/路径推导 | 当前运行时代码引用 | 改名。收敛到 `USER_DATA_PATH` 或 agent-specific root，保留短期兼容。 |
| 49 | `HYPERLOOM_RUNTIME_DIR` | 历史 runtime 根目录变量。 | N/A | 仅测试/文档引用 | 删除。清理文档/测试里的历史描述；当前运行时不作为主配置。 |
| 50 | `KERNEL_AGENT_ENV` | kernel agent env 文件路径，用于子进程加载 agent 环境。 | 未设置 | 当前运行时代码引用 | 保留。标记为平台/agent handoff。 |
| 51 | `HYPERLOOM_KERNEL_AGENT_ROOT` | kernel agent root handoff 路径。 | 未设置/路径推导 | 当前运行时代码引用 | 保留。标记 internal-only，不建议用户手动设置。 |
| 52 | `FRAMEWORK_AGENT_ROOT` | framework agent root fallback，用于解析 `fa` 工具。 | 未设置 | 当前运行时代码引用 | 保留。标记 internal-only；优先用明确的 `FA_BIN` 或安装路径。 |
| 53 | `TRACELENS_ROOT` | TraceLens checkout 路径，roofline/profile 分析依赖。 | 未设置/自动 clone | 当前运行时代码引用 | 保留。属于外部依赖路径。 |
| 54 | `MAGPIE_PATH` | Magpie checkout 路径，用于 benchmark wrapper。 | 未设置/自动 clone | 当前运行时代码引用 | 保留。属于外部依赖路径。 |
| 55 | `INFERENCEX_PATH` | InferenceX checkout 路径，用于 baseline/target analysis。 | 未设置/自动 clone | 当前运行时代码引用 | 保留。属于外部依赖路径。 |
| 56 | `ROCR_VISIBLE_DEVICES` | ROCm GPU 可见性 mask。 | 继承 | 当前运行时代码引用 | 保留。平台/集群注入。 |
| 57 | `HIP_VISIBLE_DEVICES` | HIP GPU 可见性 mask。 | 继承 | 当前运行时代码引用 | 保留。平台/集群注入。 |
| 58 | `CUDA_VISIBLE_DEVICES` | CUDA GPU 可见性 mask。 | 继承 | 当前运行时代码引用 | 保留。平台/集群注入。 |
| 59 | `NCCL_IB_HCA` | NCCL IB/RoCE 网卡选择。 | 继承 | 外部库继承，当前代码未读取 | 保留。平台/集群注入。 |
| 60 | `RAY_ADDRESS` | Ray 集群地址。 | 继承/未设置 | 当前运行时代码引用 | 保留。平台/集群注入。 |
| 61 | `MULTI_NODE_STATE_FILE` | multi-node 状态文件路径，保存 RayJob/Dynamo 状态。 | session 路径派生 | 当前运行时代码引用 | 保留。保持 session-scoped。 |
| 62 | `MODEL_PATH` | 待优化模型路径或 HF id。 | 未设置 | 当前运行时代码引用 | 删除。保留 `--model` CLI，移除 env fallback。 |
| 63 | `MODEL_CLASS` | 模型类别提示，用于 prompt、策略和模型特性判断。 | 未设置/自动推断 | 当前运行时代码引用 | 删除。保留 `--model-class` CLI 和自动推断，移除 env fallback。 |
| 64 | `FRAMEWORK` | serving framework，例如 sglang/vllm/atom/xdit。 | `sglang` | 当前运行时代码引用 | 删除。保留 `--framework` CLI，移除 env fallback。 |
| 65 | `FRAMEWORK_VERSION` | framework 版本。 | 未设置/自动探测 | 当前运行时代码引用 | 删除。保留 `--framework-version` CLI 和自动探测，移除 env fallback。 |
| 66 | `GPU_TYPE` | 当前 GPU 类型。 | 自动探测 | 当前运行时代码引用 | 删除。保留 `--gpu-type` CLI 和硬件探测，移除 env fallback。 |
| 67 | `TARGET_GPU_TYPE` | 目标 GPU 类型，给脚本渲染和 benchmark config 使用。 | 未设置/跟随 `GPU_TYPE` | 当前运行时代码引用 | 删除。由 `--gpu-type` / 探测结果写入 state，移除 env fallback。 |
| 68 | `PRECISION` | 模型精度，例如 bf16/fp8/mxfp4。 | `bf16` | 当前运行时代码引用 | 删除。保留 `--precision` CLI，移除 env fallback。 |
| 69 | `TP` | tensor parallel size。 | `1` | 当前运行时代码引用 | 删除。保留 `--tp` CLI，移除 env fallback。 |
| 70 | `EP` | expert parallel size。 | `1` | 当前运行时代码引用 | 删除。保留 `--ep` CLI，移除 env fallback。 |
| 71 | `CONC` | baseline benchmark concurrency。 | `8` | 当前运行时代码引用 | 删除。保留 `--conc` / `--conc-sweep-concs` CLI，移除 env fallback。 |
| 72 | `ISL` | input sequence length。 | `256` | 当前运行时代码引用 | 删除。保留 `--isl` CLI，移除 env fallback。 |
| 73 | `OSL` | output sequence length。 | `256` | 当前运行时代码引用 | 删除。保留 `--osl` CLI，移除 env fallback。 |
| 74 | `PROFILE_OSL` | profile 阶段使用的 OSL，可比正式 workload 更轻。 | 未设置，自动取 `min(OSL,1024)` | 当前运行时代码引用 | 删除。保留 `--profile-osl` CLI，移除 env fallback。 |
| 75 | `MAX_MODEL_LEN` | server max model length。 | 自动解析 | 当前运行时代码引用 | 删除。保留 `--max-model-len` CLI，移除 env fallback。 |
| 76 | `NODES` | multi-node 节点数旧输入。 | `1` | 当前运行时代码引用 | 删除。保留 `--nodes` CLI，移除 env fallback。 |
| 77 | `INFERENCE_OPTIMIZER_NODES` | multi-node 节点数的 inference optimizer 命名空间版本。 | `1` | 当前运行时代码引用 | 删除。保留 `--nodes` CLI，移除 env fallback。 |
| 78 | `INFERENCE_OPTIMIZER_GPUS_PER_NODE` | multi-node 每节点 GPU 数。 | `8` | 当前运行时代码引用 | 删除。保留 `--rayjob-gpus-per-node` CLI，移除 env fallback。 |
| 79 | `INFERENCE_OPTIMIZER_SERVER_ARGS` | 注入 server 额外启动参数。 | 空 | 当前运行时代码引用 | 删除。移除该 env 入口，避免旧 shell 参数污染 run。 |
| 80 | `SKIP_VARIANTS` | grid variant skip pattern。 | 空 | 当前运行时代码引用 | 删除。移除 env fallback，skip 规则只从显式 params/state 读取。 |
| 81 | `INFERENCE_OPTIMIZER_CONC_SWEEP_CONCS` | conc sweep 的并发列表。 | `1,2,4,8,16,32,64,128` | 当前运行时代码引用 | 删除。移除该 env 入口，并随 `ENABLE_CONC_SWEEP` 残留一起清理。 |
| 82 | `INFERENCE_OPTIMIZER_CONC_SWEEP_TIMEOUT_SEC` | conc sweep 单次运行超时。 | `1800` | 当前运行时代码引用 | 删除。移除该 env 入口，并随 `ENABLE_CONC_SWEEP` 残留一起清理。 |
| 83 | `INFERENCE_OPTIMIZER_CONC_SWEEP_TOTAL_BUDGET_SEC` | conc sweep 总预算。 | `9000` | 当前运行时代码引用 | 删除。移除该 env 入口，并随 `ENABLE_CONC_SWEEP` 残留一起清理。 |
| 84 | `INFERENCE_OPTIMIZER_CATALOG_PROBE_INSECURE` | 历史 catalog probe 跳过 TLS 校验开关。 | N/A | 未发现当前引用 | 删除。清理该 env 的文档痕迹。 |
| 85 | `HYPERLOOM_ALLOW_EMPTY_KERNEL_SHAPE` | 允许 kernel optimization 在没有 trace-anchored shape 时继续派发。 | 关 | 当前运行时代码引用 | 删除。保留 `--allow-empty-kernel-shape` CLI，移除 env fallback。 |
| 86 | `INFERENCE_OPTIMIZER_STRICT_PATHS` | 路径解析严格模式。 | 关 | 当前运行时代码引用 | 删除。收窄 debug 入口，改为代码默认或测试注入，不保留 runtime env。 |
| 87 | `INFERENCE_OPTIMIZER_STRICT_PHASE` | phase 严格模式。 | 关 | 当前运行时代码引用 | 删除。保留 `--strict-phase` / `--no-strict-phase` CLI，移除 env fallback。 |
| 88 | `INFERENCE_OPTIMIZER_LOCAL_INFERENCEX_ROOT` | 本地 InferenceX checkout override。 | 未设置 | 当前运行时代码引用 | 删除。与 `INFERENCEX_PATH` 语义重叠，开发 override 不进入 runtime env。 |
| 89 | `INFERENCE_OPTIMIZER_AITER_JIT_DIR` | AITER JIT cache 目录 override。 | 未设置 | 当前运行时代码引用 | 删除。改为代码默认或 installer 配置，不保留 runtime env。 |
| 90 | `INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC` | cold start 超时。 | 代码常量 | 当前运行时代码引用 | 删除。timeout 调参固化为代码常量或后续 CLI，不保留 env。 |
| 91 | `INFERENCE_OPTIMIZER_BASELINE_SERVER_READY_SEC` | baseline server ready 超时。 | 代码常量 | 当前运行时代码引用 | 删除。timeout 调参固化为代码常量或后续 CLI，不保留 env。 |
| 92 | `INFERENCE_OPTIMIZER_SERVER_DEAD_GRACE_SEC` | server dead grace 时间。 | `120` | 当前运行时代码引用 | 删除。timeout 调参固化为代码常量或后续 CLI，不保留 env。 |
| 93 | `INFERENCE_OPTIMIZER_DETOK_STALL_GRACE_SEC` | detok stall grace 时间。 | `1800` | 当前运行时代码引用 | 删除。timeout 调参固化为代码常量或后续 CLI，不保留 env。 |
| 94 | `INFERENCE_OPTIMIZER_RESCUE_PATHS` | artifact rescue 扫描路径。 | 未设置 | 当前运行时代码引用 | 删除。artifact rescue 作为 debug/tool 参数处理，不进入 runtime env。 |
| 95 | `INFERENCE_OPTIMIZER_LEAK_ROOTS` | artifact leak harvest 根目录。 | 未设置/代码 fallback | 当前运行时代码引用 | 删除。测试/debug 隔离不应作为 runtime env。 |
| 96 | `HYPERLOOM_RECOVER_ALLOW_GPU_RESET` | recovery 时允许执行 GPU reset。 | 关 | 当前运行时代码引用 | 保留。高风险逃生口，文档强调常规部署不要设置。 |
| 97 | `REFRESH_GOLDEN` | 刷新测试 golden 文件。 | 关 | 仅测试引用 | 删除。从 runtime env 表移除；只保留在测试说明中。 |
| 98 | `CRITIC_WEB_*` | Critic web tools 总开关、tool turn 上限等配置。 | 默认关 | 当前运行时代码引用 | 删除。从主 runtime env 表移出，迁到 Critic agent 专项文档。 |
| 99 | `WEB_SEARCH_*` | web search provider、结果上限、denylist、rate limit 等配置。 | 默认关/`disabled` | 当前运行时代码引用 | 删除。从主 runtime env 表移出，迁到 Critic/web tools 专项文档。 |
| 100 | `WEB_FETCH_*` | web fetch 开关、最大字节、输出长度、timeout、cache 等配置。 | 默认关 | 当前运行时代码引用 | 删除。从主 runtime env 表移出，迁到 Critic/web tools 专项文档。 |
| 101 | `TAVILY_API_KEY` | Tavily web search provider key。 | 未设置 | 当前运行时代码引用 | 保留。用于 secret 注入。 |
| 102 | `SERPER_API_KEY` | Serper web search provider key。 | 未设置 | 当前运行时代码引用 | 保留。用于 secret 注入。 |
| 103 | `BRAVE_API_KEY` | Brave web search provider key。 | 未设置 | 当前运行时代码引用 | 保留。用于 secret 注入。 |
| 104 | `HYPERLOOM_LANGFUSE_ENABLE` | Langfuse live trace master switch。 | 关 | 当前运行时代码引用 | 保留。作为 observability 显式开关。 |
| 105 | `LANGFUSE_*` | Langfuse host/public key/secret key 等 tracing 配置。 | 未设置 | 当前运行时代码引用 | 保留。用于 observability endpoint/secret 注入。 |
| 106 | `HYPERLOOM_REPORT_*` | report LLM backend、model、max tokens 等配置。 | 默认关/`none` | 当前运行时代码引用 | 删除。从主 runtime env 表移出；如仍需要，迁到 report 工具专项文档。 |
| 107 | `CRITIC_KB_*` | Critic KB client mode、prior limit、breaker threshold/cooldown 等配置。 | `inmemory`/默认限额 | 当前运行时代码引用 | 删除。从主 runtime env 表移出，迁到 Critic KB 专项文档。 |
| 108 | `KB_TIMEOUT_MS` | live KB HTTP timeout。 | `10000` | 当前运行时代码引用 | 删除。从主 runtime env 表移出，迁到 Critic live KB 专项文档。 |
| 109 | `KB_RETRY_MAX` | live KB retry 次数。 | `3` | 当前运行时代码引用 | 删除。从主 runtime env 表移出，迁到 Critic live KB 专项文档。 |
| 110 | `KB_DEAD_LETTER_DIR` | KB 写失败 dead-letter 目录。 | `/var/lib/critic-kb-dlq` | 当前运行时代码引用 | 删除。从主 runtime env 表移出，迁到 Critic live KB 专项文档。 |
| 111 | `MAGPIE_REPO`, `MAGPIE_REF` | installer 用于指定 Magpie repo/ref。 | repo 默认 + pinned SHA | 当前 installer 引用 | 删除。从主 runtime env 表移出，迁到 installer 专项文档。 |
| 112 | `INFERENCEX_REPO`, `INFERENCEX_REF` | installer 用于指定 InferenceX repo/ref。 | repo 默认 + pinned SHA | 当前运行时/installer 引用 | 删除。从主 runtime env 表移出，迁到 installer 专项文档。 |
| 113 | `GEAK_*` | GEAK installer/runtime passthrough，包括 key、base url、repo/ref、e2e runner、RAG 和 benchmark knobs。 | 多数未设置，部分有超时默认 | 当前运行时/installer 引用 | 删除。从主 runtime env 表移出，迁到 GEAK 专项文档。 |
| 114 | `FORGE_*` | Forge backend/checkout/GEMM KB 配置。 | 未设置/代码默认 | 当前运行时/installer 引用 | 删除。从主 runtime env 表移出，迁到 Forge 专项文档。 |
| 115 | `KERNEL_FORGE_*` | KernelForge checkout/backend 配置。 | 未设置/代码默认 | 当前运行时/installer 引用 | 删除。从主 runtime env 表移出，迁到 Forge 专项文档。 |
| 116 | `SGLANG_*` | SGLang bare-metal install/runtime 参数。 | 安装脚本默认 | 当前运行时/installer 引用 | 删除。从主 runtime env 表移出，迁到 bare-metal installer 文档。 |
| 117 | `AITER_*` | AITER checkout/build/JIT 参数。 | 未设置/自动选择 | 当前运行时/installer 引用 | 删除。从主 runtime env 表移出，迁到 bare-metal installer 文档。 |
| 118 | `VLLM_*` | vLLM bare-metal install/runtime 参数。 | 安装脚本默认 | 当前运行时/installer 引用 | 删除。从主 runtime env 表移出，迁到 bare-metal installer 文档。 |
| 119 | `ROCM_PATH`, `HIP_PATH` | ROCm/HIP install path。 | 继承/未设置 | 当前 installer 引用 | 删除。从主 runtime env 表移出，迁到 installer/platform 文档。 |
| 120 | `LD_LIBRARY_PATH` | 动态库搜索路径，子进程和 installer 都会继承。 | 继承 | 当前运行时/installer 引用 | 保留。不作为 Hyperloom 产品行为配置。 |
| 121 | `PYTORCH_ROCM_ARCH` | PyTorch ROCm arch build hint。 | 未设置 | 当前 installer 引用 | 删除。从主 runtime env 表移出，迁到 bare-metal/build 文档。 |
| 122 | `GITHUB_TOKEN`, `GH_TOKEN` | GitHub API/private checkout auth。 | 未设置 | 当前 CI/工具引用 | 删除。从主 runtime env 表移出；作为 CI/tool secret 单独记录。 |
| 123 | `HF_TOKEN` | HuggingFace auth token。 | 未设置 | 当前 CI 引用 | 删除。从主 runtime env 表移出；作为 CI/model-download secret 单独记录。 |
| 124 | `SAFE_OPTIMIZE_*` | CI/submit 优化参数。 | workflow/脚本默认 | 当前 CI 引用 | 删除。从主 runtime env 表移出，迁到 CI 文档。 |
| 125 | `KERNEL_OPT_BACKENDS` | kernel backend 旧 alias。 | 继承 canonical/代码默认 | 当前运行时代码引用 | 改名。统一到 `KERNEL_OPT_BACKEND_ORDER`，保留短期兼容读取后删除旧 alias。 |
| 126 | `HYPERLOOM_FRAMEWORK_SOURCE_ROOTS` | framework source roots 的旧/并行命名。 | N/A | 未发现当前引用 | 删除。清理该旧名，统一使用 `INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS` 或后续 state 字段。 |
| 127 | `SESSION_DIR`, `HYPERLOOM_SESSION_DIR` | session dir legacy/agent hint。 | 未设置 | 当前运行时代码引用 | 改名。统一到 `INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR`，保留短期兼容读取。 |
| 128 | `INFERENCE_OPTIMIZER_MN_BACKEND` | multi-node backend 默认值 fallback。 | `rayjob` | 当前 CLI fallback 引用 | 删除。保留 `--mn-backend` CLI，移除 env fallback。 |
| 129 | `INFERENCE_OPTIMIZER_RAYJOB_IMAGE` | multi-node RayJob image fallback。 | 未设置 | 当前 CLI/runtime fallback 引用 | 删除。优先 `--rayjob-image` 或 state；平台镜像注入另走部署配置。 |
| 130 | `PD_PREFILL_NODES`, `PD_DECODE_NODES` | PD disaggregated prefill/decode node 数 fallback。 | `0` | 当前 CLI fallback 引用 | 删除。保留 `--pd-prefill-nodes` / `--pd-decode-nodes` CLI，移除 env fallback。 |
| 131 | `PD_PREFILL_TP`, `PD_DECODE_TP` | PD disaggregated prefill/decode TP fallback。 | `0` | 当前 CLI fallback 引用 | 删除。保留 `--pd-prefill-tp` / `--pd-decode-tp` CLI，移除 env fallback。 |
| 132 | `PD_TRANSFER_BACKEND`, `PD_IB_DEVICE` | PD transfer backend 和 IB device fallback。 | 空 | 当前 CLI fallback 引用 | 删除。保留 `--pd-transfer-backend` / `--pd-ib-device` CLI，移除 env fallback。 |
| 133 | `PR_FEED_WINDOW_DAYS` | PR feed warmup look-back 天数 fallback。 | `30` | 当前 CLI fallback 引用 | 删除。保留 `--pr-feed-window-days` CLI，移除 env fallback。 |
| 134 | `INFERENCE_OPTIMIZER_RESEARCH_LANE_CAPACITY` | research lane specialist 并发数 fallback。 | policy ceiling | 当前 CLI fallback 引用 | 删除。保留 `--research-lane-capacity` CLI，移除 env fallback。 |
| 135 | `INFERENCE_OPTIMIZER_GPU_SPECIALIST_CAPACITY` | GPU specialist 可用 GPU 数 fallback。 | visible GPU count | 当前 CLI fallback 引用 | 删除。保留 `--gpu-specialist-capacity` CLI，移除 env fallback。 |
| 136 | `INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES` | GPU specialist device pool fallback。 | 未设置 | 当前 runtime 引用 | 删除。若需要固定 GPU 池，改为 CLI/state 字段，不再从 env 读。 |
| 137 | `INFERENCE_OPTIMIZER_SPECIALIST_MODEL` | specialist sub-agent model fallback。 | orchestration model | 当前 CLI fallback 引用 | 删除。保留 `--specialist-model` CLI，移除 env fallback。 |
| 138 | `INFERENCE_OPTIMIZER_SPECIALIST_MAX_TURNS` | specialist 最大 turn 数 fallback。 | 默认常量 | 当前 CLI fallback 引用 | 删除。保留 `--specialist-max-turns` CLI，移除 env fallback。 |
| 139 | `INFERENCE_OPTIMIZER_SPECIALIST_PER_TURN_MAX_SECONDS` | specialist per-turn timeout fallback。 | `600` | 当前 CLI fallback 引用 | 删除。保留 `--specialist-per-turn-max-seconds` CLI，移除 env fallback。 |
| 140 | `INFERENCE_OPTIMIZER_SPECIALIST_DISPATCH_MODE` | specialist 执行模式 fallback。 | `subprocess` | 当前 CLI fallback 引用 | 删除。保留 `--specialist-dispatch-mode` CLI，移除 env fallback。 |
| 141 | `INFERENCE_OPTIMIZER_SPECIALIST_MCP_CONFIG` | specialist subprocess MCP config fallback。 | 未设置 | 当前 CLI fallback 引用 | 删除。保留 `--specialist-mcp-config` CLI，移除 env fallback。 |
| 142 | `HYPERLOOM_SPECIALIST_KB_MCP_URL` | specialist 只读 KB MCP endpoint fallback。 | 未设置 | 当前 CLI fallback 引用 | 删除。保留 `--specialist-kb-mcp-url` CLI；`GBRAIN_BASE_URL` 仍作为服务 endpoint 保留。 |
| 143 | `HYPERLOOM_LOCAL_KB_ROOT` | local recipe KB root fallback。 | `USER_DATA_PATH/kb` | 当前 CLI fallback 引用 | 删除。保留 `--local-kb-root` CLI；`USER_DATA_PATH` 保留。 |
| 144 | `INFERENCE_OPTIMIZER_EXPLORE_OVERTIME_KILL_RATIO` | EXPLORE variant overtime kill ratio fallback。 | `2.0` | 当前 CLI fallback 引用 | 删除。保留 `--explore-overtime-kill-ratio` CLI，移除 env fallback。 |
| 145 | `INFERENCE_OPTIMIZER_EXPLORE_VARIANT_TIMEOUT_SEC` | EXPLORE variant hard timeout fallback。 | 自动推导 | 当前 CLI fallback 引用 | 删除。保留 `--explore-variant-timeout-sec` CLI，移除 env fallback。 |
| 146 | `INFERENCE_OPTIMIZER_EXPLORE_VARIANT_TIMEOUT_SAFETY_MARGIN` | EXPLORE timeout 自动推导安全边际 fallback。 | `0.5` | 当前 CLI fallback 引用 | 删除。保留 `--explore-variant-timeout-safety-margin` CLI，移除 env fallback。 |
| 147 | `ROBUSTNESS_SERVER_URL` | Robustness server URL fallback。 | 自动发现/未设置 | 当前 CLI/runtime fallback 引用 | 删除。保留 `--robustness-server-url` CLI；平台默认发现另走部署配置。 |
| 148 | `ROBUSTNESS_LLM_RCA_DISABLED` | Robustness LLM RCA kill switch。 | 未设置 | 当前 runtime 引用 | 删除。保留 `--robustness-llm-rca` / `--no-robustness-llm-rca` CLI，移除 env kill switch。 |
