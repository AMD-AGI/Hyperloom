# Hyperloom 全量 fallback cut-off 裁决计划(GPT)

> 输入依据:
> - `fallback-survey-all.json`:全量 **768** 条 fallback 编目。
> - `fallback-clean.plan.md`:53 条疑似不可达的对抗式复核,结论为 **39 确认不可达 / 14 仍可达**。
> - 本轮额外派发 12 个 sub agent,按 category、subsystem、测试、迁移、批次分别复核。
>
> 目标:最大化删除本地冗余/兼容/封装代码,允许高风险修改和删测试。但本计划仍区分"冗余 fallback"与"产品真实分支"。前者删除,后者若要节省代码则改为集中工具、assert 或硬失败,不能盲删。

---

## Merge `origin/main` 后的影响裁决

本分支已合入 `origin/main`。这次 main 改动很大,但对本计划的影响是"局部推进,总体结论仍成立":

| main 新状态 | 对本计划的影响 |
|---|---|
| `SharedState.from_dict` 新增 `_migrate_legacy_extra_sglang_args_keys`,递归迁移 `extra_sglang_args -> extra_server_args` 与 `candidate_extra_sglang_args -> candidate_extra_server_args`。 | `extra_sglang_args` 相关迁移门槛已**部分满足**。因此 `_payload_aliases.py:94` 的遗留读取更适合进入第一批删除;但外部/operator 旧 `kernel_candidates.json` 兼容窗口仍需 Batch 0 决策。 |
| `proposals.py` `_reserved` 仍只排除 `"framework_name"`,未排除 `"framework"`。 | `framework -> framework_name` 的 Batch E 门槛**未满足**;`schema.py:452`、`cortex_t0`、local search 相关 fallback 仍不能直接删。 |
| `result_recorder.py:875` 仍复制 `"args"`/`"envs"` 到 `best_config`。 | `args/envs -> extra_server_args/extra_envs` 的 write-side 复活机制仍存在;`cortex_t0.py:187/211`、`gbrain_ingest.py:205` 仍必须迁移后再删。 |
| `gate.py:530/2343` 的 Python <3.9 `except AttributeError` 仍存在。 | Batch 1 的 gate 版本守卫删除结论不变。 |
| `sessions.py:217/413` 的 PyYAML `ImportError` fallback 仍存在。 | Batch 1 的 PyYAML hard-dependency 删除结论不变。 |
| `cli/backends.py:232/242/258`、critic CLI `environment` 三臂、`decision_reviewer.py:1091`、`fetch_rows`、`_payload_aliases.py`、`result_recorder.py:843/852` 均仍在当前代码里。 | Batch 2 的主要删除目标仍有效。 |
| `gate.py:1644` 注释仍写"Older SharedState without the field",且 `SharedState.from_dict` 尚未对 `specialist_patch_verdicts: null` 做 coercion。 | B13 仍是 live fallback;删除前必须先做 null -> `{}` 迁移并修注释。 |
| `breakdown` 仍写/声明 `forge_invocations`,但 reporter/`SECTION_GROUPS` 仍未注册 forge renderer。 | `forge_invocations` 丢失问题仍存在,建议作为顺手功能修复。 |

因此,合入 main 后需要微调的是:

1. **把 `extra_sglang_args` 迁移视为已由 main 部分完成**,删除 shim 时只剩外部文件兼容决策。
2. **不要误以为 Batch E 已解锁**:main 尚未处理 `framework`、`args/envs`、`specialist_patch_verdicts=null`、provenance marker。
3. 实际动手前最好对当前 HEAD 重新跑一次结构化 fallback survey,因为 main 改动了约 95 个文件,部分 line number 已漂移。

---

## 0. 总裁决

全量 768 条不能等价理解为 768 处可删代码。粗略分层如下:

| 层级 | 数量/范围 | 裁决 |
|---|---:|---|
| 已确认不可达 | 39 条 + C1 若干子臂 | **优先删除**。同时删除只覆盖这些路径的测试。 |
| 仍可达但只是旧数据兼容 | 约 20-40 条 | 先做"迁移或放弃旧数据"决策,再删除 read-side fallback。 |
| 静默防御 catch-all | default/if-else/enum 中一批 | 不保留静默 fallback,改成 `assert`/`raise`。 |
| env/config/operator switch | 大量 reachable | **保留**,但可集中到 `common/env.py` 或配置解析层以减少重复代码。 |
| 外部系统/网络/磁盘/LLM/subprocess 容错 | 大量 reachable | **保留或收窄异常类型**,不能删成硬崩溃,否则产品行为变化过大。 |
| mock/dry-run/noop/heartbeat 协议 | 少量 reachable | 多数保留;只删除旧 mock downgrade 或测试 seam。 |

推荐总策略:

1. **立刻删除 39 confirmed-dead + C1 子臂**。
2. **用一次 schema/data policy 解锁 Batch E**:旧 session/recipe/db 要么迁移,要么明确不再支持 resume/warm-start。
3. **把 silent fallback 改成 invariant**:`dict.get(..., default)`、`.get(..., fallback)`、enum 尾部分支如果按当前类型系统不应发生,改为直接索引或 `raise AssertionError`。
4. **集中 env/default 逻辑**:不要删除产品默认值,删除重复解析代码。
5. **删测试的顺序必须跟代码删除同 PR**:先删代码分支,再删只覆盖该分支的测试,避免覆盖率倒退。

---

## 1. Category 级处理策略

### 1.1 `legacy_key`(88)

这是最大可删除来源。分三类处理:

| 子类 | 例子 | 裁决 |
|---|---|---|
| 写侧已彻底迁移且有 CI guard | `extra_sglang_args -> extra_server_args` | 立刻删 read shim。 |
| 无 in-repo producer 的低层 operator 格式 | critic CLI `environment`、review JSON `verdicts` | 立刻删 legacy arm;保留主路径与输入校验。 |
| 旧 `state.json`/`recipe.json`/remote row 可喂活 | `args/envs`、`framework/framework_name`、`delta_pct/gain_pct` | 先迁移或放弃旧数据,再删。 |

**立刻删的 legacy clusters:**

- `src/hyperloom/agents/kernel/tools/_payload_aliases.py:94`
  - 删除 `extra_sglang_args` 分支、`LEGACY_KEY`、deprecation warning。
  - 同步收窄 `test_no_legacy_writer_sites.py` allowlist。
- `src/hyperloom/agents/critic/runtime/cli.py:162/200/228`
  - `packet.get("context") or packet.get("environment") or {}` 改为 `packet.get("context") or {}`。
- `src/hyperloom/agents/critic/runtime/decision_reviewer.py:1091`
  - 删除 `review.get("verdicts")` fallback。
  - 保留后续 `not isinstance(verdicts_raw, list)` 校验。
- `src/hyperloom/orchestrator/knowledge/recipe_kb/dispatcher.py:200`
  - 删除 `_v2_to_arbor` 中 `labels.get("framework")` 臂。
- `src/hyperloom/orchestrator/knowledge/recipe_kb/gbrain_ingest.py:271`
  - 删除 `recipe_to_page` 中顶层 `recipe.get("framework")` 臂。
- `src/hyperloom/orchestrator/knowledge/recipe_kb/local_store.py:878`
  - 删除 `framework_name` key 分支里的 `payload.get("framework")` fallback。
  - 注意:local search 的 `else` 分支中 `payload.get("framework")` 仍是 live legacy matching;这属于迁移门槛,见 Batch E。
- `src/hyperloom/orchestrator/loop/result_recorder.py:843/852`
  - 删除 `e.get("source_file")` 与 sibling `e.get("ts")` 子臂。
- `src/hyperloom/agents/robustness/sources/local_probe.py:393`
  - 删除 legacy events schema 的第二条 SELECT。

**迁移后删的 legacy clusters:**

- `extra_args/args/envs -> extra_server_args/extra_envs`
  - 涉及 `cortex_t0.py:187/211`、`gbrain_ingest.py:205`、`prelude.py:102/291`、`explore.py:1298`、`framework.py:3855`、`writeback.py`、`shared_state.py` 等。
  - 前置:在 `SharedState.from_dict` 迁移 `current_best.args/envs`;在 `result_recorder.py:875` 停止复制 `"args"`/`"envs"`。
- `framework -> framework_name`
  - 涉及 `schema.py:452`、`cortex_t0`、local store search。
  - 前置:`proposals.py` `_reserved` 加 `"framework"`;`Recipe.from_dict` 规范化;本地 recipe 搜索改用 `framework_name`。

### 1.2 `default_when_missing`(286)

大多数是产品默认值,不能全删。处理策略:

| 模式 | 裁决 |
|---|---|
| dataclass 字段恒存在,`getattr(..., default)` 防御 | 改直接字段访问;保留 live 的空 dict/list 检查。 |
| 封闭集合 `.get(..., default)` | 改 direct index 或 assert。 |
| env var 未设置时默认 | 保留默认值,把重复解析集中到 `common/env.py`。 |
| 缺失 dict key -> `{}`/`[]` | 若来自网络/磁盘/LLM/subprocess,保留;若来自强类型内部对象,改 assert。 |
| 构造函数 `None -> default object` | 多数保留,这是 API contract。 |
| 旧 manifest/state/session 字段缺失 | 迁移或 abandon old sessions 后删除。 |

**立刻处理:**

- `src/hyperloom/agents/critic/runtime/decision_reviewer.py:1017`
  - `_CLASS_RANK.get(cls, ...)` 改 `_CLASS_RANK[cls]`。
- `src/hyperloom/orchestrator/loop/intent_router.py:292`
  - 删除 `getattr(pending, "task_id", None) or`。
- `src/hyperloom/orchestrator/prompts/prompt_builder.py:265`
  - `_filter_actions` 中 `registry.get(name)` + `if meta is None` 改为直接 lookup/raise。
- `src/hyperloom/orchestrator/state/shared_state.py:1902`
  - `hasattr` guard 改 assert 或直接访问。
- `src/hyperloom/agents/robustness/signals/local_health.py:407/450`
  - `getattr(..., None)` + `isinstance` 防御删掉,保留 `if not ray_info/fd_info`。
- `src/hyperloom/agents/robustness/sources/local_probe.py:2095`
  - `_EXTERNAL_MOUNT_ENVS` 的 `default_path=""` 没意义,可改为只读 env var;保留 unset 时 skip。

**集中化而非删除:**

- env numeric defaults:timeout、ttl、max_turns、threshold、capacity。
  - 用 `common/env.py` 的 `env_int/env_float/env_bool` 或补齐缺口。
  - 删除散落的 `try ValueError -> default` 样板。

### 1.3 `try_except`(237)

处理重点不是"全删 try/except",而是删除不可能异常、收窄大 catch、集中重复 safe-read。

| 模式 | 裁决 |
|---|---|
| 支持版本不可能触发 | 删除。 |
| hard dependency `ImportError` | 删除。 |
| dead function 内部 retry/decode | 删除整函数。 |
| 宽泛 `except Exception` 包住外部 IO/LLM/subprocess | 收窄异常或集中处理,不要全删。 |
| SQLite/offline archived DB schema | 若放弃旧 DB 或先 `ensure_schema`,再删 fallback。 |
| JSON/YAML/file read | 多数保留或集中到 `_safe_read_json/_safe_load_yaml`。 |

**立刻删:**

- `src/hyperloom/orchestrator/policy/gate.py:530` 与 `:2343`
  - Python <3.9 `Path.is_relative_to` guard。
- `src/hyperloom/inference_optimizer/breakdown/collectors/sessions.py:217` 与 `:413`
  - PyYAML hard dep,删除 `except ImportError`。
- `inference_optimizer/baseline_comparison/inferencex_client.py:199`
  - `fetch_rows` 无生产调用方,整函数和 re-export 删除。
- `src/hyperloom/agents/robustness/decision/rca_engine.py:585/588`
  - `_safe_extra_evidence` provider 从不注入;删除 try/except 与 non-list guard,或删除整个 injection seam。
- `src/hyperloom/agents/quantization/driver/runner.py:338`
  - `"env" in kwargs` 恒 True,删除 dead else `raise env_exc from exc`。

**先迁移/改造再删:**

- `src/hyperloom/inference_optimizer/breakdown/collectors/telemetry.py:282`
  - lane_capacity fallback 对 v1 archived DB live。
  - 若要删除:在 collector 打开 DB 后调用 `ensure_schema(conn)` 或拒绝 v1 DB。

**收窄而非删除:**

- `orchestrator/kernel/request_handlers.py`、`critic/runtime/kb_writer.py`、`cortex_t0.py`、`roles/codex.py` 等 broad `except Exception`。
  - 把"吞所有异常 -> 默认值"改为捕获具体 IO/API/timeout 错误。
  - 对内部编程错误不要 fallback,直接 raise。

### 1.4 `if_else_default`(99)

多数是 live product routing。删除策略只针对封闭枚举尾部和常量条件。

**立刻删/改 assert:**

- `src/hyperloom/agents/quantization/driver/retry.py:329`
  - `DEFAULT_QUARK_GIT_URL` 恒非空,删除 `else ""`。
- `src/hyperloom/agents/quantization/driver/retry.py:271`
  - `_decide_next_step` 的 `non_retryable_ask` tail 改 `raise AssertionError`。
- `src/hyperloom/agents/quantization/driver/assessment.py:524`
  - `derive_status` 尾部 `return "failed"` 改 `raise AssertionError/ValueError`。

**确认保留:**

- `src/hyperloom/orchestrator/phases/framework.py:3148`
  - 有 sub agent 建议删,但裁决为 **保留**。
  - 原因: `UPDATE_STATE` 可把 `framework_agent_authoring_enabled` 写 False,且该字段不在 `CORE_STATE_FIELDS`。
- `src/hyperloom/orchestrator/specialists/rebench.py:65/68`
  - 函数 live;`18888` floor 在默认 sysctl practically dead,但是 OS 配置防御,保留。

### 1.5 `sdk_graceful`(48),`resume_downgrade`(6),`mock_downgrade`(3)

**立即删或修注释:**

- `src/hyperloom/orchestrator/roles/claude.py:606`
  - 旧 SDK TypeError 降级代码已不存在;删 stale docstring。
- `src/hyperloom/orchestrator/roles/claude.py:442`
  - `_resume_downgraded` 恒 False。若确认无外部 `llm_calls.jsonl` 消费者依赖该 key,删除字段、metadata、trace schema。
- `src/hyperloom/orchestrator/loop/sub_agent_runner.py:10`
  - 修 docstring:未知 task kind 硬失败,没有 `backend.run()` fallback。
- `src/hyperloom/orchestrator/specialists/runner.py:375-388`
  - 两个 `except AttributeError` 都死,直接读 `KnowledgePlane` property。
- `src/hyperloom/inference_optimizer/cli/backends.py:232/242/258`
  - CLI 调用前已 `sys.exit(2)`;删除降级/ValueError guard 或改 assert。

**版本门槛后再删:**

- `enum.StrEnum` shim:只有 bump `requires-python >= 3.11` 后可删。
- Langfuse emitter shims:只有 bump optional `langfuse>=4` 后可删。
- `cwd/env` SDK constructor shims:确认 `claude-agent-sdk>=0.2.110` 覆盖 `cwd/env` 后再删。

**mock downgrade 多数保留:**

- multi-node robustness mock downgrade、`InMemoryKBClient`、roofline test double guard 当前都 live。若追求极端删测试 seam,需单独产品决策。

---

## 2. 迁移/放弃旧数据的总门槛

为了最大化删除 fallback,建议采用高风险策略:

> **不再支持 pre-cut schema 的 old sessions / old recipes / old coordinator DB 直接 resume。**

落地方式二选一:

1. **迁移路线**:提供一次性 migration/normalization,运行后删除 fallback。
2. **放弃路线**:引入 `MIN_SUPPORTED_SCHEMA_VERSION`,旧 `state.json`/`recipe.json`/`coordinator.db` 直接拒绝或当作缺失,不再兼容。

如果目标是最大化代码 cut-off,推荐采用"轻迁移 + 严格版本门槛":

### 2.1 `SharedState.from_dict`

新增规范化:

- `specialist_patch_verdicts: null/non-dict -> {}`。
- `current_best.args -> current_best.extra_server_args`。
- `current_best.envs -> current_best.extra_envs`。
- 对 dict/list typed fields 做最小 coercion,否则后续读路径不再 catch `AttributeError`。

解锁删除:

- `gate.py:1644` `except AttributeError`。
- `cortex_t0.py:187/211` `args/envs` fallback 的一部分。
- `gbrain_ingest.py:205` nested `envs` fallback。

### 2.2 `Recipe.from_dict` / local recipe rewrite

新增规范化或一次性 rewrite:

- 顶层 `framework -> framework_name`。
- `best_config.args -> extra_server_args`。
- `best_config.envs -> extra_envs`。
- 剥离 `_field_sources` / `_sources`。

还需写侧停止复活旧字段:

- `proposals.py` `_reserved` 加 `"framework"`。
- `result_recorder.py:875` copy loop 移除 `"args"`/`"envs"`。
- `cortex_t0.py` prior extras 排除 `_field_sources`/`_sources`。
- local search 从 `"framework"` label_match 迁到 `"framework_name"` 或同时迁移索引。

解锁删除:

- `schema.py:452`。
- `cortex_t0.py:78/139/187/211`。
- `gbrain_ingest.py:205` 的 `get("envs")` 臂。
- local store 里的 legacy framework matching。

### 2.3 `coordinator.db`

策略:

- 在 breakdown offline collector 打开 DB 后调用 `ensure_schema(conn)`,或拒绝 v1 DB。
- 若选择拒绝旧 DB,删除 `telemetry.py:282` lane_capacity fallback。
- `local_probe.py` legacy events SELECT 已确认死,可先删。

---

## 3. 批次计划

### Batch 0 — 政策决定(先写进 PR 描述)

1. 关闭 `extra_sglang_args` 一版兼容窗口。
2. 关闭 `fetch_rows` 作为外部 API 的承诺。
3. 决定是否删除 `llm_calls.jsonl.resume_downgraded` key。
4. 决定旧 session/recipe/db 是迁移还是直接 abandon。
5. 决定是否 bump Python 到 `>=3.11`。

### Batch 1 — 零风险/文档和版本守卫

删除/修改:

- `gate.py:530`, `gate.py:2343`。
- `sessions.py:217`, `sessions.py:413`。
- `sub_agent_runner.py` docstring。
- `canonical_id.py` stale docstring(同时修返回维度顺序)。
- `claude.py:594-596` stale docstring。
- `llm_trace.py:157-158` stale resume downgrade comment。
- `local_probe.py:2082` stale "defaults" docstring。
- `cli/__init__.py:24-25` 的 `shutil/subprocess` re-export 和误导注释。

测试:

- 通常无需删测试;若 docstring 测试存在,同步更新。

### Batch 2 — 已确认死的 compatibility surface

删除:

- `_payload_aliases.py:94` legacy branch。
- critic CLI `environment` 三臂。
- `decision_reviewer.py:1091` `verdicts` 臂。
- `inferencex_client.fetch_rows` 整函数 + `baseline_comparison/__init__.py` re-export。
- `backends.py:232/242/258`。
- `specialists/runner.py:375-388` 两个 `except AttributeError`。
- `result_recorder.py:843/852`。
- `intent_router.py:292` dead `getattr`。
- `local_probe.py:393` legacy SELECT。
- `rca_engine.py:585/588` provider try/except 与 non-list guard。
- `runner.py:338` dead else。

删除/改测试:

- `test_payload_aliases_shim.py` legacy/deprecation cases。
- `test_no_legacy_writer_sites.py` allowlist 收窄。
- `test_fetch_rows_*`。
- `test_build_backends_*_without_root / requires_root` 直调 guard tests。
- `test_local_probe_reads_legacy_events_schema`。
- critic packet `environment` fixture(若存在)。
- `extra_evidence_provider` 注入测试(若删除字段)。

### Batch 3 — invariant/raise 替代 silent fallback

改动:

- `_CLASS_RANK.get` -> direct index。
- `_filter_actions` unknown action silent skip -> direct lookup/raise。
- `record_action_attempt` missing attrs guard -> assert。
- `derive_status` tail -> raise。
- `_decide_next_step` tail -> raise。
- `build_assessment([])` guard -> raise/assert non-empty。
- `DEFAULT_QUARK_GIT_URL` ternary -> unconditional string。
- `local_health.py` direct field access + empty dict check。

测试:

- 删除"unknown action silently skipped"测试,改为 `raises`。
- `build_assessment([])` 测试改为 `raises` 或删。
- 增加 enum partition invariant tests:
  - `ASK == {checkpoint_aborted, eval_gap_exceeded} | ASK_RETRYABLE`
  - 所有 `OutcomeId` 都被 `derive_status` 覆盖。

### Batch 4 — 旧数据迁移/放弃后删除

先做:

- `SharedState.from_dict` 规范化。
- `Recipe.from_dict` 规范化。
- `proposals.py` `_reserved` 加 `"framework"`。
- `result_recorder.py:875` 移除 `"args"`/`"envs"`。
- `cortex_t0.py` extras 排除 `_field_sources`/`_sources`。
- 可选:一次性 `USER_DATA_PATH` recipe/session normalize 工具。
- 可选:`telemetry.py` 调用 `ensure_schema`。

再删:

- `schema.py:452`。
- `cortex_t0.py:78/139/187/211` provenance 和 `args/envs` fallback。
- `gbrain_ingest.py:205` `get("envs")` 臂,保留 flat-sibling env extraction。
- `gate.py:1644` `except AttributeError`。
- `telemetry.py:282` lane_capacity fallback。
- `prelude.py:102/291`, `explore.py:1298`, `framework.py:3855`, `writeback.py` 等 `extra_args/args/envs` legacy reads。

### Batch 5 — 只保留产品 fallback,集中化

保留但整理:

- env/config defaults -> `common/env.py`。
- JSON/YAML/file safe-read -> shared helper。
- network/subprocess/LLM parsing failures -> typed exception + structured error。
- mock/noop/heartbeat -> 明确命名为 protocol behavior,不要叫 fallback。
- external API shape normalization -> 单独 adapter 层。

---

## 4. 必须保留或暂不删的关键项

即使允许高风险,以下不建议删:

| 位置/模式 | 原因 |
|---|---|
| `framework.py:3148` authoring downgrade | live via `UPDATE_STATE`,不是死码。 |
| `gate.py:1644` | 迁移前 live via `specialist_patch_verdicts=null`。 |
| `result_collector.py:270` | workspace 被 LLM Bash 删除时防崩溃,应提升 confidence 到 high。 |
| `telemetry.py:282` | offline/v1 DB 支持未关闭前 live。 |
| `invocations.py:35` top-level `geak_invocations` | 这是 live primary;应删 nested `invocations` 旧形态,不是删 top-level。 |
| `gbrain_ingest.py:209-213` flat-sibling env extraction | documented FLAT shape,必须保留。 |
| `cortex_t0.py:211` non-Mapping env guard | malformed-data 防御,删除 legacy `envs` 时仍保留。 |
| `resource_lock.py/schema.py` DB migration | 旧 coordinator DB 支持未关闭前保留。 |
| auth/env var chains | 多部署环境真实需要。 |
| external API response shape variants | 上游 API 多形态,不是本地冗余。 |
| heartbeat/noop branches | Coordinator 协议要求。 |

---

## 5. Reporter/功能缺口顺手修

`fallback-clean.plan.md` 的 C3 发现一个非 fallback 但相关的功能缺口:

- `forge_invocations` 是 schema/exporter/recorder 的一等字段:
  - `schema.py`
  - `exporter.py`
  - `recorder/sections.py`
- 但 reporter 只注册了 `geak_invocations`,没有 `render_forge`,`SECTION_GROUPS` 也没列入。

建议在 Batch 2 或单独 PR:

1. `invocations.py:35` 删除 nested `breakdown.get("invocations")` dead lookup,改读 top-level。
2. 增加 `render_forge`。
3. `compose.py` 的 `SECTION_GROUPS` 加 `"forge_invocations"`。
4. 改写 `test_reporters_v1_1.py` 中 nested `{"invocations":{"geak":...}}` fixture。

---

## 6. 测试删除策略

### 6.1 直接删除

- `test_payload_aliases_shim.py` 中 legacy warning/legacy wins/legacy constant cases。
- `test_fetch_rows_*`。
- `test_v2_to_arbor_reads_legacy_framework_label`。
- `test_local_recipe_store.py` 中 direct `_matches_labels` legacy `framework_name` case。
- `test_local_probe_reads_legacy_events_schema`。
- `test_build_backends_anthropic_only_degrades_to_claude_without_root`。
- `test_build_backends_critic_agent_requires_root`。
- `test_build_backends_robustness_agent_requires_root`。
- `test_assessment_branches_unit.py` 中 `build_assessment([])` 的 fallback 成功用例。
- 任何 critic CLI 顶层 `environment` packet fixture。
- 若删除 `_resume_downgraded`,删除/改写 `test_llm_trace_unit.py` 中手造 `resume_downgraded=True`。

### 6.2 改写为 invariant tests

- `_review_constraints`:测试 `classify_proposal_action` 返回集合等于 `_CLASS_RANK` keys。
- `_filter_actions`:unknown action 改为 raises,不再 silently skipped。
- quantization `OutcomeId`:分区穷举测试。
- `SharedState.from_dict`:新增 null/dict coercion 测试。
- `Recipe.from_dict`:新增 legacy key normalization 测试。
- `result_recorder`:确认不再写 `"args"`/`"envs"`。
- `proposals.py`:确认 `"framework"` 被 reserved 掉。

### 6.3 保留

- `test_cid_to_path_components_rejects_legacy_6_segment`:这是正确行为测试,不是 fallback 测试。
- env/config default tests:保留,但改成测试 `common/env.py`。
- network/disk/JSON parsing error tests:保留或改为 typed exception。
- heartbeat/noop/mock protocol tests:保留。

---

## 7. 预期收益

保守估算:

| 来源 | 代码收益 |
|---|---|
| 39 confirmed-dead + C1 子臂 | 约 100-180 LOC fallback/注释/测试 seam |
| `fetch_rows` + tests | 约 100+ LOC(含测试) |
| env parsing 集中化 | 约 50-100 LOC 重复样板 |
| Batch E 迁移后 | 约 50-100 LOC read compatibility + tests |
| docstring/re-export/test seam | 约 50 LOC |

更重要的收益不是 LOC 本身,而是:

- 删除隐式 backward compatibility。
- 将未来 enum/action 漏配从 silent default 变成 loud failure。
- 收敛本地数据 schema,减少 `state.json`/`recipe.json` 任意形状在系统中传播。
- 让测试从"证明旧 fallback 还能用"转向"证明新 invariant 不会破"。

---

## 8. 推荐第一批 PR 范围

如果现在开始实际代码 cut,我建议第一批只做这些:

1. `gate.py` 两个 Python 版本 guard。
2. `sessions.py` 两个 PyYAML ImportError guard。
3. `sub_agent_runner.py`、`canonical_id.py`、`claude.py`、`llm_trace.py`、`local_probe.py` 的 stale comments/docstrings。
4. `cli/__init__.py` `shutil/subprocess` re-export。
5. `inferencex_client.fetch_rows` + tests。
6. critic `environment` 三臂 + `verdicts` 臂。
7. `_payload_aliases.py` legacy branch + tests + guard allowlist。
8. `result_recorder.py:843/852`。
9. `intent_router.py:292`。
10. `runner.py:338`。

这批不依赖大迁移,删除面明显,CI 风险低。

第二批再做 invariant raise 与 `backends.py` / `specialists/runner.py`。

第三批开启 `SharedState` / `Recipe` 迁移门槛,再删除 Batch E。
