# Hyperloom Fallback 逻辑普查 — 最终报告

> 生成方式:22→18 子系统并行发现(编目 **768** 处 fallback）→ 对每个"疑似死"分支派怀疑者 agent 全仓库对抗式反证 → 汇总。共 **72 个 agent**。
>
> **准确统计(以结构化数据为准,下方正文的 37/8 是汇总 agent 在被截断 JSON 上的口径,已在此修正):**
> - 编目 fallback 总数:**768**
> - 进入对抗式验证的疑似项:**53**
> - ✅ **确认不可达(可删除):39**(全部 confidence=high)
> - ⚠️ **验证者反证判定仍可达(不要删,需人工复核):14**

## 快速索引 A — 39 处确认不可达(可删)

| 位置 | 类别 | 符号 | 置信度 |
|---|---|---|---|
| `inference_optimizer/baseline_comparison/inferencex_client.py:199` | try_except | fetch_rows | high |
| `src/hyperloom/agents/critic/runtime/cli.py:162` | legacy_key | _cmd_list_priors | high |
| `src/hyperloom/agents/critic/runtime/cli.py:200` | legacy_key | _cmd_write_verdict | high |
| `src/hyperloom/agents/critic/runtime/cli.py:228` | legacy_key | _cmd_write_kb_drafts | high |
| `src/hyperloom/agents/critic/runtime/decision_reviewer.py:1091` | legacy_key | DecisionReviewer._commit_coordinator_inbox | high |
| `src/hyperloom/agents/critic/runtime/decision_reviewer.py:1017` | default_when_missing | DecisionReviewer._review_constraints | high |
| `src/hyperloom/agents/kernel/tools/_payload_aliases.py:94` | legacy_key | read_extra_server_args | high |
| `src/hyperloom/agents/quantization/driver/assessment.py:452` | default_when_missing | build_assessment | high |
| `src/hyperloom/agents/quantization/driver/assessment.py:524` | if_else_default | derive_status | high |
| `src/hyperloom/agents/quantization/driver/assessment.py:271` | if_else_default | _decide_next_step | high |
| `src/hyperloom/agents/quantization/driver/retry.py:329` | if_else_default | quantize_via_prompt | high |
| `src/hyperloom/agents/quantization/driver/runner.py:338` | try_except | run_one_attempt | high |
| `src/hyperloom/agents/robustness/decision/rca_engine.py:585` | try_except | _safe_extra_evidence | high |
| `src/hyperloom/agents/robustness/signals/local_health.py:407` | default_when_missing | _ray_head_symptoms / _fd_pressure_symptoms (getattr guards) | high |
| `src/hyperloom/agents/robustness/sources/local_probe.py:393` | legacy_key | _read_coordinator_events / _try_select | high |
| `src/hyperloom/agents/robustness/sources/local_probe.py:2095` | default_when_missing | _probe_external_mounts | high |
| `src/hyperloom/inference_optimizer/breakdown/collectors/sessions.py:217` | try_except | _load_yaml_dict_safe | high |
| `src/hyperloom/inference_optimizer/cli/__init__.py:24` | legacy_key | cli/__init__ shutil/subprocess re-export | high |
| `src/hyperloom/inference_optimizer/cli/__init__.py:172` | legacy_key | cli/__init__ back-compat re-export block (__all__ / private helper re-exports) | high |
| `src/hyperloom/inference_optimizer/cli/backends.py:232` | sdk_graceful | _build_backends | high |
| `src/hyperloom/inference_optimizer/cli/backends.py:242` | default_when_missing | _build_backends | high |
| `src/hyperloom/inference_optimizer/cli/backends.py:258` | default_when_missing | _build_backends | high |
| `src/hyperloom/orchestrator/actions/executors/baseline.py:1181` | legacy_key | BaselineExecutor._run_once | high |
| `src/hyperloom/orchestrator/actions/executors/profile.py:786` | legacy_key | ProfileExecutor (profile server-args merge) | high |
| `src/hyperloom/orchestrator/knowledge/recipe_kb/canonical_id.py:111` | legacy_key | cid_to_path_components | high |
| `src/hyperloom/orchestrator/knowledge/recipe_kb/dispatcher.py:200` | legacy_key | _v2_to_arbor | high |
| `src/hyperloom/orchestrator/knowledge/recipe_kb/dispatcher.py:433` | default_when_missing | RecipeKB._remote_label | high |
| `src/hyperloom/orchestrator/knowledge/recipe_kb/gbrain_ingest.py:271` | legacy_key | recipe_to_page | high |
| `src/hyperloom/orchestrator/knowledge/recipe_kb/local_store.py:878` | legacy_key | _labels_match_payload | high |
| `src/hyperloom/orchestrator/loop/intent_router.py:292` | default_when_missing | IntentRouter (specialist verdict recording) | high |
| `src/hyperloom/orchestrator/loop/result_recorder.py:843` | legacy_key | ResultRecorder._build_kernel_optimizations_from_state | high |
| `src/hyperloom/orchestrator/loop/sub_agent_runner.py:10` | sdk_graceful | SubAgentRunner.run_task | high |
| `src/hyperloom/orchestrator/policy/gate.py:530` | try_except | _resolved_within | high |
| `src/hyperloom/orchestrator/policy/gate.py:2343` | try_except | PolicyGate._resolves_within_session (session-dir path check) | high |
| `src/hyperloom/orchestrator/prompts/prompt_builder.py:265` | default_when_missing | _filter_actions | high |
| `src/hyperloom/orchestrator/roles/claude.py:442` | resume_downgrade | ClaudeBackend.run | high |
| `src/hyperloom/orchestrator/roles/claude.py:606` | sdk_graceful | ClaudeBackend._instantiate_options | high |
| `src/hyperloom/orchestrator/specialists/runner.py:375` | sdk_graceful | SpecialistRunner._resolve_tools | high |
| `src/hyperloom/orchestrator/state/shared_state.py:1902` | default_when_missing | SharedState.record_action_attempt (audit) | high |

### 按类别
| 类别 | 数量 |
|---|---|
| `legacy_key` | 15 |
| `default_when_missing` | 10 |
| `try_except` | 6 |
| `sdk_graceful` | 4 |
| `if_else_default` | 3 |
| `resume_downgrade` | 1 |

## 快速索引 B — 14 处存疑(验证者判定仍可达,**保留**)

| 位置 | 置信度 | 验证者为何认为仍可达(摘要) |
|---|---|---|
| `src/hyperloom/agents/kernel/tools/tracelens_skill_runner.py:939` | high | Line 939 (`resolved_model = resolved_model or str(globals().get("DEFAULT_MODEL", "claude-opus-4-8")).strip()`) executes on the LIVE non-OpenAI SDK path: run_tracelens_skill reaches |
| `src/hyperloom/inference_optimizer/breakdown/collectors/telemetry.py:282` | high | The breakdown collector _collect_lane_timeline (telemetry.py:270-286) opens coordinator.db with a bare sqlite3.connect(str(db_path)) at line 271 and NEVER calls ensure_schema. So w |
| `src/hyperloom/inference_optimizer/breakdown/reporters/_renderers/invocations.py:35` | high | The "legacy_key" fallback branch at invocations.py:35 (breakdown.get('geak_invocations')/'forge_invocations') is NOT dead — it is the LIVE primary production path, and the "dead" c |
| `src/hyperloom/inference_optimizer/cli/preflight.py:33` | high | The auditor mischaracterized preflight.py:23-29 as a dead "back-compat re-export shim." It is a LIVE functional import: preflight.py imports _is_stale_proxy_url, _resolve_llm_endpo |
| `src/hyperloom/orchestrator/knowledge/cortex_t0.py:78` | high | The claim mislabels the fallback as dead. The characterized fallback behavior is the `return ""` at src/hyperloom/orchestrator/knowledge/cortex_t0.py:103 (default_when_missing). It |
| `src/hyperloom/orchestrator/knowledge/recipe_kb/gbrain_ingest.py:205` | high | Live trigger exists. Path: result_recorder._build_recipe_attrs_from_state (src/hyperloom/orchestrator/loop/result_recorder.py:875) copies "envs" from current_best into best_config  |
| `src/hyperloom/orchestrator/knowledge/recipe_kb/schema.py:452` | high | The fallback (schema.py:452 `framework_name=str(d.get("framework_name") or d.get("framework") or "")`) is REACHABLE via a live write-path, not just legacy reads. Chain:  1) result_ |
| `src/hyperloom/orchestrator/phases/framework.py:3148` | high | The fallback at framework.py:3148 (`if want_author and not authoring_enabled: want_raw=True; want_author=False`) can still fire in a live run, no hand-edited state.json or resume r |
| `src/hyperloom/orchestrator/policy/gate.py:1644` | high | The `except AttributeError` at gate.py:1644 is broader than "SharedState lacks the method." It also catches an AttributeError raised INSIDE the method body. get_specialist_patch_ve |
| `src/hyperloom/orchestrator/specialists/rebench.py:66` | low | The enclosing function _pick_free_port is definitively on a LIVE call path, not dead. Live chain: `python -m hyperloom.orchestrator.specialists.rebench` is a real subprocess entry  |
| `src/hyperloom/agents/quantization/driver/result_collector.py:270` | medium | Live trigger path: quantize_via_prompt (retry.py:279) is the sole production entry (CLI cli.py:143 and orchestrator prelude quantization_request_handlers.py:72, both without runner |
| `src/hyperloom/orchestrator/knowledge/cortex_t0.py:139` | medium | Branch at cortex_t0.py:143-144 (and donor branch 139-142) still fires on LEGACY PERSISTED rows, refuting the writer-only dead-reason. Chain: (1) schema.Recipe.from_dict (schema.py: |
| `src/hyperloom/orchestrator/knowledge/cortex_t0.py:187` | medium | The fallback (`best_config.get("args")` / `best_config.get("envs")` at cortex_t0.py:187-188 in `_recipe_is_actionable`) CAN still fire in current code. Reachability chain, all veri |
| `src/hyperloom/orchestrator/knowledge/cortex_t0.py:211` | medium | The fallback (best_config['args']/['envs'] when canonical keys absent) is reachable via multiple live paths, refuting the 'only pre-rename rows' dead-reason.  (1) LIVE WRITER emits |

## 快速索引 C — 本轮补充审核(10 个 sub agent 逐项复核)

> 口径:本节是对原 39/14 主清单的补充,**不改动原统计**。新增内容多为同一 fallback 表达式内的"子臂"、平行守卫、陈旧注释或删除批次约束;除 C1 标注的少数平行守卫外,不应直接折算为新的整条 fallback 数量。

### C1. 新增/未单列的确认不可达子臂(可并入清理批次)

| 位置 | 结论 | 建议处理 |
|---|---|---|
| `src/hyperloom/inference_optimizer/breakdown/collectors/sessions.py:413` | `_read_invocation_envs` 内 `import yaml` 的 `except ImportError` 与 A17 同构不可达:PyYAML 是 hard dependency,collector 进程内运行。 | 作为 A17b 与 `sessions.py:217` 同批删除。 |
| `src/hyperloom/orchestrator/loop/result_recorder.py:852` | `e.get("last_ts") or e.get("ts") or ""` 中 `e.get("ts")` 子臂不可达;`kernel_opt_attempts` 唯一写入方只写 `entry["last_ts"]`,无历史写入 `entry["ts"]`。 | 与 A31 的 `source_file` 子臂同批清理。 |
| `src/hyperloom/agents/critic/runtime/cli.py:162/200/228` | 三处 `packet.get("context") or packet.get("environment") or {}` 中 `environment` 臂确认死;`or {}` 第三臂在 in-repo 自动路径中也无生产者触发。触发描述应写作 `context` 缺失或 falsy,不只是"无 key"。 | 删除 `environment` 臂时保留 `context` 主路径;若继续保留 operator 手写包兼容,需显式说明其范围。 |
| `src/hyperloom/agents/critic/runtime/decision_reviewer.py:1091-1093` | `review.get("verdicts")` 臂仍确认死;但 `if not isinstance(verdicts_raw, list)` 还有 live 直达路径(当 `review_verdicts` 存在但非 list)。 | 删除 legacy `verdicts` 臂时不要误删非 list 校验。触发应写作 `review_verdicts` 缺失或为 null。 |
| `src/hyperloom/agents/robustness/decision/rca_engine.py:585/588` | `_safe_extra_evidence` 的 `except Exception` 与 `not isinstance(items, list)` 是两个独立死子臂;根因都是生产构造从不注入 `extra_evidence_provider`。 | 若删除 `extra_evidence_provider` 字段,两臂可一起清理;否则保留测试用注入语义需另行决定。 |
| `src/hyperloom/agents/robustness/sources/local_probe.py:439-441` | `_try_select` 中"所有 SELECT 都失败后 log.debug + return []"在生产 live schema 下不可达;第一条 SELECT 恒成功。legacy-schema 单测只命中第二条 SELECT 成功,不覆盖 all-fail 终端分支。 | 可单独补测试或作为防御性终端保留;不要把它和 legacy SELECT 成功路径混为一谈。 |
| `src/hyperloom/orchestrator/specialists/runner.py:378/385` | `_resolve_tools` 有两处平行 `except AttributeError`(pr monitor 与 cortex),生产 `KnowledgePlane` 两个 property 都存在且不抛 AttributeError。 | A38 清理时两处一起删,不是单个 try/except。 |
| `src/hyperloom/agents/kernel/tools/tracelens_skill_runner.py:939` | 整体 fallback literal `"claude-opus-4-8"` 是 live;但 `globals().get("DEFAULT_MODEL", ...)` 找到真实 `DEFAULT_MODEL` 的内部子臂不可达,因为模块从未定义/导入该全局。 | 若保留 literal fallback,可简化为显式默认模型并修正注释,不要保留虚构 global lookup。 |
| `src/hyperloom/orchestrator/specialists/rebench.py:65/68` | `_pick_free_port()` 函数 live;但在默认 Linux `ip_local_port_range=32768-60999` 下 `bind(0)` 不会返回 8888,所以 line 65 false 分支与 line 68 `return 18888` 实际不可达。自定义 sysctl 可使其触发。 | 保留防御性代码;报告中需区分"函数 live"与"18888 floor 在标准部署 practically dead"。 |

### C2. 存疑/仍可达条目的子臂拆分(删除时必须保留的 live 部分)

| 位置 | 补充结论 | 删除约束 |
|---|---|---|
| `src/hyperloom/agents/robustness/signals/local_health.py:407/450` | `getattr(..., None)` default 与 `not isinstance(..., dict)` 死;但 `or not ray_info` / `or not fd_info` 空 dict 检查 live 且有测试覆盖。 | 安全清理形态是改为直接读字段后 `if not ray_info:` / `if not fd_info:`,不能删空 dict 检查。 |
| `src/hyperloom/orchestrator/knowledge/cortex_t0.py:78` | `_field_sources["best_config"]` 的 str/list 两臂与 `_sources[0]` 三个子臂在 fresh sessions 中死;但与 B7 一样可由 pre-#757 持久化 `recipe.json` 重新喂活。 | 需先在 `Recipe.from_dict`/local store 迁移中剥离 `_field_sources`/`_sources`,之后 `_row_best_config_source` 才可折叠为恒 `""`。 |
| `src/hyperloom/orchestrator/knowledge/cortex_t0.py:139-146` | donor 分支与 own-source 分支应分开看:两者 fresh 死,但 legacy sibling rows 经 `kb.search()` 可触发 donor provenance。 | 同 C2 上一条迁移完成后,整个 `_warm_recipe_source` 可折叠为 `return "gbrain" if _remote_is_gbrain(kb) else "cortex-kb"`。 |
| `src/hyperloom/orchestrator/knowledge/cortex_t0.py:187/211` | `args` 与 `envs` legacy 子臂触发条件不同:二者主要来自 legacy `state.json/current_best`;尾部 `or {}` 是 live 默认,不是 legacy fallback。`_config_replay_args_envs` 的 `not isinstance(envs, Mapping)` 是 malformed-data 防御,也应保留。 | 删除前需迁移 `current_best.args -> extra_server_args`、`current_best.envs -> extra_envs`,并从 `result_recorder.py:875` 复制列表移除 `"args"`/`"envs"`;保留 `or {}` 与 non-Mapping guard。 |
| `src/hyperloom/orchestrator/knowledge/recipe_kb/gbrain_ingest.py:205/209-213` | line 205 `best_config.get("envs")` 仅 legacy/持久化路径 live;但 flat-sibling env extraction(lines 209-213) 是当前 documented FLAT shape 处理,在无 nested env map 时 live。 | 未来只能删除 `get("envs")` legacy 臂,不能删除 flat-sibling else 分支;`"envs"` 仍应留在 `_NON_ENV_BEST_CONFIG_KEYS` 以避免被当成平铺 env。 |
| `src/hyperloom/orchestrator/knowledge/recipe_kb/local_store.py:878/881` | A29 的 `framework_name` 分支确认死;但 cortex_t0 传入 `"framework"` key 的 local search 会走 `else` 分支并直接 `payload.get("framework")`,这是 live legacy matching。 | 清理 on-disk 顶层 `"framework"` 前,必须先让 cortex_t0 local search 改用/匹配 `framework_name`,否则会破坏旧行查询。 |
| `src/hyperloom/orchestrator/phases/framework.py:3148/3163` | line 3148 降级 live,因 `framework_agent_authoring_enabled` 不在 `CORE_STATE_FIELDS`,UPDATE_STATE 可写 False。line 3163 的 `authoring_enabled` 在 line 3148 已翻转 `want_author=False` 后是冗余保险。 | 保留 3148;3163 可不动,但报告应称其为 harmless redundancy,不是死 fallback。 |
| `src/hyperloom/orchestrator/policy/gate.py:1644` | B13 live 结论正确;真实触发不是"SharedState 无字段",而是 `from_dict` 宽松反序列化出 `specialist_patch_verdicts=None`,方法体内 `None.get` 抛 AttributeError。 | 保留 `except AttributeError`,但修正 line 1645 注释。 |

### C3. 报告文字/分类/批次需要修正

| 项 | 修正 |
|---|---|
| A20 `src/hyperloom/inference_optimizer/cli/backends.py:232` | 类别不应是 `sdk_graceful`;这是 root 缺失的 `default_when_missing` 防御降级。分类表应保持 `sdk_graceful=4`,并把本项计入 `default_when_missing`。 |
| A19 `src/hyperloom/inference_optimizer/cli/__init__.py:172` | 该 re-export block 已由 `44a5b6a8` 删除;快速索引应标为"已完成/仅确认",而非仍待删除。 |
| A25 `src/hyperloom/orchestrator/knowledge/recipe_kb/canonical_id.py:111` | 类别应为 `stale_docstring`,不是 `legacy_key`;同一 docstring 还误写返回维度顺序(文档顺序与 line 124 解包顺序不同)。 |
| A28 `src/hyperloom/orchestrator/knowledge/recipe_kb/gbrain_ingest.py:271` | 原文称 `framework_name` 恒非空导致 `or` 短路不准确;`framework_name` 可为空。正确死因是 inline mirror kwargs 无顶层 `"framework"` 值可供该臂返回。 |
| A18 `src/hyperloom/inference_optimizer/cli/__init__.py:18-25` | 源码注释声称测试 patch `cli.shutil`/`cli.subprocess`,但当前测试都 patch 子模块;删除 re-export 时应一起删该误导注释。 |
| A36/A37 companion comments | `src/hyperloom/orchestrator/trace/llm_trace.py:157-158` 仍描述 resume= 被 SDK 拒绝后降级;`src/hyperloom/orchestrator/roles/claude.py:594-595` 仍说 unknown SDK builds degrade via `_instantiate_options`。两者均随 SDK floor 上抬变陈旧。 |
| B2 `src/hyperloom/agents/quantization/driver/result_collector.py:270` | 仍可达且应从 medium 提到 high:这是 `collect_artifacts` 中唯一会把缺失 workspace 转为 graceful result 的异常守卫;其它读取都用 `is_file`/`_read_*` 静默吸收缺失。 |
| B4 `src/hyperloom/inference_optimizer/breakdown/reporters/_renderers/invocations.py` | 另有非 fallback 缺口:`forge_invocations` 是 schema/exporter/recorder 的一等字段,但 reporter 只注册 `geak_invocations`,无 `render_forge` 且 `SECTION_GROUPS` 未列入,报告会静默丢 forge invocation 数据。 |

---

以下为详细论证(每条含触发条件、为何现在不再触发、验证者证据)。注意其中章节标题里的 37/8 计数请以上方 39/14 为准。

# FALLBACK 逻辑普查报告 — Hyperloom 代码库

## 1. 概览

本次普查对 Hyperloom 代码库中的 FALLBACK(回退)逻辑进行了系统编目,并对其中被标记为"疑似不可达(possibly-dead)"的子集执行了对抗式可达性复核(adversarial reachability verification)。

### 1.1 编目与复核总量

- **CONFIRMED 确认不可达(`stillReachable=false`,可删除):37 项**(来自 `confirmedDead`)。
- **复核后判定仍可达 / 需人工复核(`stillReachable=true`):8 项**(仅出现在 `allVerified` 中,详见第 3 节)。

> 说明:本报告依据数据中的两个数组。`confirmedDead` 全部为 `stillReachable=false`;`allVerified` 除包含这些确认死码外,还额外包含 8 项复核结论为"仍可达"的条目。数据末尾 `gate.py:1644` 的证明文本在输入中被截断,但其 verdict 字段(`stillReachable=true`,`confidence=high`)完整,已据此归类。

### 1.2 按子系统(unit)分布(确认不可达)

| 子系统 (unit) | 确认不可达数 |
|---|---|
| orch/actions | 2 |
| orch/loop | 3 |
| orch/knowledge | 6 |
| orch/roles | 2 |
| orch/state+policy+bus | 3 |
| orch/trace+specialists+framework+scoring+prompts | 2 |
| agents/kernel-tools | 1 |
| agents/robustness | 4 |
| agents/critic | 5 |
| agents/quantization | 5 |
| io/cli | 4 |
| io/breakdown | 1 |
| io/rest+common | 1 |

(合计 39;其中若干 io/cli 与部分条目在同一文件多行,详见第 2 节逐条列表。)

### 1.3 按类别(category)分布(确认不可达)

| 类别 (category) | 确认不可达数 | 典型代表 |
|---|---|---|
| `legacy_key`(遗留键回退) | 15 | `extra_sglang_args`、`framework` 键、`source_file`、`verdicts`、`environment` 等 |
| `default_when_missing`(缺失时默认) | 8 | `task_id` getattr、`_remote_label`、`record_action_attempt` 等 |
| `try_except` | 6 | `is_relative_to` (Py<3.9)、PyYAML ImportError、InferenceX 重试等 |
| `sdk_graceful`(SDK 优雅降级) | 5 | `_instantiate_options`、`SubAgentRunner`、`_resolve_tools`、backends critic 降级 |
| `if_else_default` | 3 | `DEFAULT_QUARK_GIT_URL`、`derive_status`、`_decide_next_step` |
| `resume_downgrade` | 1 | `_resume_downgraded` |

---

## 2. 确认不可达的 fallback(可删除)

以下均为验证者判定 `stillReachable=false` 且 `confidence=high` 的条目。按置信度均为 high,故按子系统聚合、以证据强度(是否有提交/静态守卫测试固化)从强到弱大致排序。

### 2.1 遗留键回退:`extra_sglang_args`(orch/actions)

**`src/hyperloom/orchestrator/actions/executors/baseline.py:1181` — `BaselineExecutor._run_once`(legacy_key)**
- 触发条件:Task 参数携带遗留键 `extra_sglang_args` 而非规范键 `extra_server_args`。
- 现状为何不再触发:提交 `21a40b6f`("Prune orchestrator dead code paths")删除了 `hyperloom.common.payload_aliases` 模块,并把 `read_extra_server_args(params)` 替换为 `str(params.get("extra_server_args") or "")`;回退分支被结构性移除,连读取遗留键的代码都不存在。同提交删除了 `_grid_base.py` 中 `GridVariant(extra_sglang_args=...)` 弃用别名。
- 验证者证据:`baseline.py` 中已无任何 `extra_sglang_args`/`read_extra_server_args`/`payload_aliases` 引用(grep 全空);`grep -rn extra_sglang_args src/`(排除 tests)仅命中 kernel-agent 独立 shim,orchestrator 从不导入;静态守卫测试 `test_no_legacy_writer_sites.py` 强制该字面量只允许出现在 3 文件白名单内。
- 置信度:high。

**`src/hyperloom/orchestrator/actions/executors/profile.py:786` — `ProfileExecutor`(profile server-args merge, legacy_key)**
- 触发条件:同上,Task 参数携带 `extra_sglang_args`。
- 现状为何不再触发:同提交 `21a40b6f` 删除了 `payload_aliases` 导入、替换为规范键读取,并删除了 `params.pop("extra_sglang_args", None)` 清理行;其依赖的 `hyperloom.common.payload_aliases` 模块本身在 `0f601f2c` 被删除(`ls` 失败)。
- 验证者证据:全库 grep `extra_sglang_args` 仅命中 3 个只读/测试文件(kernel-agent shim、其测试、静态守卫测试);无 src/ 代码写入该键;CI 守卫 `test_no_legacy_writer_sites.py` 防止未来出现写入方。
- 置信度:high。

**`src/hyperloom/agents/kernel/tools/_payload_aliases.py:94` — `read_extra_server_args`(agents/kernel-tools, legacy_key)**
- 触发条件:payload 缺规范键但含遗留 `extra_sglang_args`,分支发 `DeprecationWarning` 并返回强转后的遗留值。
- 现状为何不再触发:src/ 内无任何路径写入 `extra_sglang_args`;唯一生产调用方 `kernel_optimization.py:1486` 读取 `load_candidates()` 解析的 `kernel_candidates.json`,而所有候选构造方(`tracelens_analysis.py`、`tracelens_skill_runner.py`、`parallel_e2e_runner.py`)都不写该键;prune 提交 `0f601f2c` 移除了最后一个遗留读取方并删除 `common/payload_aliases.py`。
- 验证者证据:`git grep extra_sglang_args` 仅命中该 shim 及其两个测试;无动态键构造。仅剩触发者为外部/运维手写 payload 或过期磁盘上的 `kernel_candidates.json`(即 shim 自述的"一版只读兼容"),无任何库内代码路径能把遗留键放入 payload。
- 置信度:high。

### 2.2 遗留键回退:kernel_opt 与 loop(orch/loop)

**`src/hyperloom/orchestrator/loop/result_recorder.py:843` — `ResultRecorder._build_kernel_optimizations_from_state`(legacy_key)**
- 触发条件:`kernel_opt_attempts` 条目无 `last_source_file` 但含裸键 `source_file`。
- 现状为何不再触发:该 map 的唯一写入方 `_kernel_decisions.py:639`(entry 源自 line 512,源路径仅存于 `entry["last_source_file"] = source_file`,line 541),全树无任何 `entry["source_file"] = ...` 赋值。
- 验证者证据:`git log --all -G 'entry\["source_file"\]\s*='` 全空;`last_source_file` 追溯至提交 `52b829ad`,该 diff 显示此字段是新增的,裸 `source_file` 进入的是另一个 dict(`last_kernel_opt`)——**从未发生过重命名**,故连过期恢复会话也不会携带裸键。测试 `test_profile_and_kernel_handlers.py:3911` 断言回退到 `last_source_file`。
- 置信度:high。

**`src/hyperloom/orchestrator/loop/intent_router.py:292` — `IntentRouter`(specialist verdict recording, default_when_missing)**
- 触发条件:`getattr(pending, "task_id", None) or pa_params.get("task_id")`——主路径尝试从 `PendingProposal` 读取 `task_id` 属性。
- 现状为何不再触发:`PendingProposal`(`coordinator.py:445-455`)是仅有 7 个字段的 dataclass,无 `task_id`,无 `__getattr__` shim;`getattr` 恒返回 None,仅 `pa_params.get("task_id")` 回退有效。代码注释本身已注明"PendingProposal has no task_id field"。
- 验证者证据:`dataclasses.fields()` 确认无 `task_id`;所有构造点(`intent_router.py:176`、`resume.py:91`、`explore.py:1500/1617`、`framework.py:3085` 及测试)均不传 `task_id`;全库无 `setattr(pending`/`pending.task_id =`。task_id 实际存于 `payload["params"]`,正由回退分支读取。
- 置信度:high。

**`src/hyperloom/orchestrator/loop/sub_agent_runner.py:10` — `SubAgentRunner.run_task`(sdk_graceful,陈旧文档串)**
- 触发条件:模块 docstring 声称存在"LLM external sub-agent fallback (backend.run())"作为无确定性执行器时的替代分派路径。
- 现状为何不再触发:`run_task`(lines 175-274)只做一次 `executor_registry.get(task.kind)`(line 205),None 时硬失败返回 `SubAgentResult(state='failed', error='no runner registered')`(lines 206-220);文件内除 docstring 外无任何 `backend.run()`。docstring 为**陈旧描述**,回退已删除。
- 验证者证据:grep 确认 `sub_agent_runner.py` 内无 `backend` 属性/`.run(` 调用;真实 LLM 路径是**已注册**执行器(`cli/executors.py:260` register_executor("specialist", ...)),并非未注册 kind 的回退;历史提交 `356b68bd` 显示该路径从未真正实现。
- 置信度:high。

### 2.3 遗留键回退:recipe KB `framework` / canonical-id(orch/knowledge)

**`src/hyperloom/orchestrator/knowledge/recipe_kb/dispatcher.py:200` — `_v2_to_arbor`(legacy_key)**
- 触发条件:远端行 labels 缺 `framework_name` 但含遗留 `framework`。
- 现状为何不再触发:生产 `self.remote` 恒为 `GbrainRemoteRecipeClient`(`kb.py:124-128`);每次 gbrain 读取都经 `_page_to_recipe` 用 `C.canonical_labels(...)`(`recipe_snapshot_constants.py:173-181`)重建固定 7 键 labels,永不产出裸 `framework`;且 `_page_to_recipe`(line 498)在建 labels 前已将遗留页的 `attrs.get('framework')` 提升为 `framework_name`。
- 验证者证据:`cortex_t0.py` 里的 `framework` 键是查询 label_match(非存储 labels),`_labels_match`(line 610)会静默跳过;唯一触发者是单测 `test_v2_to_arbor_reads_legacy_framework_label` 手工注入。
- 置信度:high。

**`src/hyperloom/orchestrator/knowledge/recipe_kb/local_store.py:878` — `_matches_labels`(数据中 symbol 名 `_labels_match_payload` 不准, legacy_key)**
- 触发条件:磁盘 recipe.json 缺 `framework_name` 但含遗留 `framework`(search 读原始 JSON 不经 from_dict 规范化)。
- 现状为何不再触发:该分支在 `elif key == "framework_name":`(line 831)内,需生产调用方传入以 `"framework_name"` 为**键**的 label_match;而三个生产调用方(`cortex_t0.py:434/1123/1146`)构建的 label_match 均用遗留键 `"framework"`,路由到通用 `else` 分支。
- 验证者证据:唯一以 `framework_name` 为键的生产 dict(`dispatcher.py:83 _labels_from_canonical_id`)只喂给 `self.remote.search`,从不进本地 search;gbrain 客户端的 framework_name labels 走其自身独立 `_labels_match`。该分支仅被 `test_local_recipe_store.py:53` 直接调用触发。**结论比原判更强:生产中根本无人向本地 search 传入 `framework_name` 键。**
- 置信度:high。

**`src/hyperloom/orchestrator/knowledge/recipe_kb/gbrain_ingest.py:271` — `recipe_to_page`(legacy_key)**
- 触发条件:recipe 缺 `framework_name` 但含遗留 `framework`。
- 现状为何不再触发:两条生产入口——(1) 内联镜像 `GbrainMirroringRecipeKB.put_recipe`,其 kwargs 来自仅有的两个调用方(`proposals.py:255`、`cortex_t0.py:1056`)且**恒设** `framework_name`,`recipe.get("framework_name")` 恒非空使 `or` 短路;(2) 外部批量摄取的驱动(`ingest_local_to_gbrain`/`main`/`__main__`)已在提交 `21a40b6f` 删除(`grep -rn ingest_local_to_gbrain` 全空,无 console entry)。
- 验证者证据:唯一在 live code 产出裸 `framework` 的 `_collect_workload_tags`(`result_recorder.py:714`)只嵌在 `extras` 内,从不出现在顶层。
- 置信度:high。

**`src/hyperloom/orchestrator/knowledge/recipe_kb/canonical_id.py:111` — `cid_to_path_components`(legacy_key,陈旧文档串)**
- 触发条件:docstring(lines 86-88)声称接受并补齐遗留 6 段 id;实际代码严格要求恰好 8 段,否则抛异常。
- 现状为何不再触发:line 111 `if len(parts) != 1 + CANONICAL_ID_DIMENSIONS: raise`,其后无补齐代码,line 124 tuple 解包假定恰好 7 维。所谓补齐回退**从未存在**,docstring 陈旧。
- 验证者证据:测试 `test_cid_to_path_components_rejects_legacy_6_segment`(`test_local_recipe_store.py:102-105`)断言 6 段 id 抛 `InvalidCanonicalIdError`;唯一 id 生产者 `recipe_canonical_id` 恒发 8 段;git 历史(`7b0cd37`/`cbd35fa`)显示文件从引入严格守卫起就带此 docstring,补齐路径从未存在。
- 置信度:high。

**`src/hyperloom/orchestrator/knowledge/recipe_kb/dispatcher.py:433` — `RecipeKB._remote_label`(default_when_missing)**
- 触发条件:远端客户端类名不在 `_REMOTE_LABELS` map 中。
- 现状为何不再触发:能被接入的唯一远端客户端是 `GbrainRemoteRecipeClient`(`kb.py:124` `build_gbrain_remote_from_env`;Cortex 后端在 PR #757 移除),且已在 map 中映射为 `'gbrain'`。`.get()` 默认(返回类名)分支只对今天从不被构造的假想后端触发。
- 验证者证据:`RecipeKB` 为 dataclass(`remote: Any = None`),非测试构造点仅 `cli/kb.py:118/126/128`;`build_gbrain_remote_from_env` 只返回 `GbrainRemoteRecipeClient` 或 None;`GbrainMirroringRecipeKB` 经 `__getattr__` 委托,`remote` 类型不变;库内无其他 remote client 类。remote 为 None 时在 line 431 提前返回 `'none'`。
- 置信度:high。

### 2.4 SDK 优雅降级 / resume 降级(orch/roles)

**`src/hyperloom/orchestrator/roles/claude.py:442` — `ClaudeBackend.run`(resume_downgrade)**
- 触发条件:原本在 claude-agent-sdk 构建拒绝 `resume=` 时置 True,强制无状态重试,写入 `llm_calls.jsonl` 的 `resume_downgraded`。
- 现状为何不再触发:`_resume_downgraded`(line 211)为 `field(default=False, init=False)`,全库无任何置 True 的赋值;注释注明"Always False now";pin `claude-agent-sdk>=0.2.110`(装 0.2.116)使 `resume=` 无条件受支持,翻转分支已删除。该 key 仅作为稳定字段以常量 False 发出。
- 验证者证据:grep `_resume_downgraded`/`resume_downgraded` 无生产 True 赋值;`git show 13fba44a`(引入字段的提交)同时删除了触发块(旧 `except TypeError` 警告 + 置 True);现 `_instantiate_options`(lines 606-619)为裸 return 无 try/except;唯一 True 值在 `test_llm_trace_unit.py:27`(手造 metadata 测序列化)。
- 置信度:high。

**`src/hyperloom/orchestrator/roles/claude.py:606` — `ClaudeBackend._instantiate_options`(sdk_graceful)**
- 触发条件:旧 SDK 构建拒绝 `resume`/`effort`/`thinking` kwargs(TypeError)时,原会不带这些 kwargs 重试。
- 现状为何不再触发:方法体现为单行 `return self.sdk_options_cls(**kwargs)`,无 try/except;docstring(lines 608-611)明示 `>=0.2.110` floor 下无需兼容回退。`_apply_effort_options` docstring(line 596)对降级的引用已陈旧。
- 验证者证据:回退在 `13fba44a` 删除;pin `>=0.2.110`,装 0.2.116,并经 `inspect.signature` 及实调 `ClaudeAgentOptions(...)` 验证三个 kwargs 均可构造;5 个生产实例化点均不注入 `sdk_options_cls`;唯一自定义注入在 tests(断言新的传播行为)。另两处 TypeError 重试块(quantization/kernel)属不相关的 cwd/env kwargs。
- 置信度:high。

### 2.5 Python 版本守卫 `try_except`(orch/state+policy+bus)

**`src/hyperloom/orchestrator/policy/gate.py:530` — `_resolved_within`(try_except)**
- 触发条件:`Path` 无 `is_relative_to` 方法(Python < 3.9)时抛 `AttributeError`。
- 现状为何不再触发:`pyproject.toml` `requires-python = ">=3.10"`(line 10),ruff `target-version="py310"`(line 451);`is_relative_to` 自 3.9 起恒存在。作者已标注 `# pragma: no cover — Python <3.9`。
- 验证者证据:`v = Path(str(value)).resolve()` 用 stdlib `pathlib.Path`(line 11),无 backport/遮蔽/monkeypatch,运行时 3.12 上 `hasattr==True`;无测试删除/patch 该方法。
- 置信度:high。

**`src/hyperloom/orchestrator/policy/gate.py:2343` — `PolicyGate._path_under_session`(数据中 symbol 名 `_resolves_within_session` 略有出入, try_except)**
- 触发条件:同上(Python < 3.9)。
- 现状为何不再触发:与 gate.py:530 结构相同;`requires-python >=3.10` 保证 `is_relative_to` 存在。作者已标注 `# pragma: no cover`。
- 验证者证据:`v = Path(str(value)).resolve()`(line 2338),无 shim/monkeypatch;3.12 上 `hasattr==True`;两个调用方(lines 2426/2439)均传字符串。
- 置信度:high。

**`src/hyperloom/orchestrator/state/shared_state.py:1902` — `SharedState.record_action_attempt`(default_when_missing)**
- 触发条件:`_AUDIT_ACTIONS` 中某 action 的 `<action>_attempts` 或 `last_<action>` 属性缺失(hasattr 为 False)。
- 现状为何不再触发:`_AUDIT_ACTIONS` 五个 action(baseline/profile/sweep/explore/roofline)的两个字段均以 `field(default_factory=...)` 声明,dataclass 实例上恒存在,故 hasattr 恒 True,`return None` 分支死。仅当未来新增 action 而不加字段时才可能触发。
- 验证者证据:`SharedState()` 与 `SharedState.from_dict({})` 经验证对全部十个属性 hasattr 均 True;`SharedState` 为无 `__getattr__`/`__slots__`/自定义 `__init__` 的 dataclass;`from_dict` 经 `cls(**filtered)` 依赖生成的 `__init__` 补默认值;非审计 action 命中的是 line 1899 的另一 return。
- 置信度:high。

### 2.6 SDK 优雅降级 / 默认(orch/trace+specialists+framework+scoring+prompts)

**`src/hyperloom/orchestrator/specialists/runner.py:375` — `SpecialistRunner._resolve_tools`(sdk_graceful)**
- 触发条件:`plane.pr_monitor_enabled` / `plane.cortex_enabled` 抛 `AttributeError`(knowledge_plane 对象缺该属性)。
- 现状为何不再触发:生产 knowledge_plane 恒为 `KnowledgePlane`,两个属性均为不会抛异常的 `@property`(`knowledge_plane.py:80`/`:89`);plane 为 None 时 `if plane is not None:`(line 375)守卫会跳过整个 try/except。仅鸭子类型/mock plane 触发。
- 验证者证据:唯一生产构造在 `cli/executors.py:165/189`,plane 来自 `cli/__init__.py:1611/1790`(`None if not cortex_enabled else _bootstrap_knowledge_plane(...)`);`KnowledgePlane` 无子类/无 `__getattr__` 魔法;无 subprocess/entry-point 以鸭子类型 plane 构造 runner。触发对象仅存于测试。
- 置信度:high。

**`src/hyperloom/orchestrator/prompts/prompt_builder.py:265` — `_filter_actions`(default_when_missing)**
- 触发条件:`registry.get(name)` 返回 None(未知 action 名)。
- 现状为何不再触发:唯一生产路径中 `enabled_actions` 恒来自 `default_enabled_actions(...)`,后者只从封闭元组 `FULL_ENABLED_ACTIONS`(`action_surfaces.py:71`)**过滤**(只减不增),14 个名字均有 `actions/_meta/<name>.yaml`;`ActionRegistry.load()` 严格校验,缺失/畸形会抛 `ActionRegistryError` 而非静默省略。
- 验证者证据:单一调用链 `cli/__init__.py:2050 -> ...`;经验证 4 种 flag 组合下 14 个 name 的 `registry.get` 均非 None;包数据以 glob 打包 yaml(`pyproject.toml:123`);唯一触及 None 分支的是单测 `test_unknown_enabled_action_is_silently_skipped`。
- 置信度:high。

### 2.7 Robustness agent(agents/robustness)

**`src/hyperloom/agents/robustness/sources/local_probe.py:393` — `_read_coordinator_events / _try_select`(legacy_key)**
- 触发条件:第一条 SELECT(`seq AS id, from_agent AS agent, ...`)抛 `sqlite3.Error`,回退到第二条遗留 schema SELECT(`id, agent, timestamp AS ts`)。
- 现状为何不再触发:当前 Coordinator DB 的 events 表 schema 为 `seq/msg_id/from_agent/.../ts`(`schema.py:70-80`),第一条恒成功;第二条针对已不存在的遗留列,即便到达也会抛"no such column"返回 []。
- 验证者证据:经验证第一条 SELECT 成功、第二条抛"no such column: id";`db_path` 固定为 `<session_dir>/storage/coordinator.db`,不可重定向;全树 + git 全历史(`git log --all -S`)仅在测试 fixture 中出现遗留 schema 的 CREATE/INSERT;schema 迁移只动 `leases` 表从不动 `events`。
- 置信度:high。

**`src/hyperloom/agents/robustness/sources/local_probe.py:2095` — `_probe_external_mounts`(default_when_missing)**
- 触发条件:`os.environ.get(env_name, default_path)`——挂载环境变量未设。
- 现状为何不再触发:`_EXTERNAL_MOUNT_ENVS` 三项(TRACELENS_ROOT/TRACELENS_INTERNAL_ROOT/INFERENCEX_PATH)`default_path` 全为 `''`,紧接的 `if not path: continue`(lines 2097-2098)会跳过,故默认值分支永不产出探测行;唯一探测方式是环境变量显式设为非空。
- 验证者证据:该常量全库仅定义一次、无重赋值/monkeypatch;`git -S` 历史显示自 src-layout 起默认恒为 `''`;测试 `test_probe_external_mounts_skips_tracelens_root_when_unset` 断言未设时零行。docstring "falling back to their defaults" 为陈旧文字。
- 置信度:high。

**`src/hyperloom/agents/robustness/decision/rca_engine.py:585` — `_safe_extra_evidence`(try_except)**
- 触发条件:`extra_evidence_provider` 抛异常或返回非 list。
- 现状为何不再触发:`extra_evidence_provider`(dataclass 字段默认 None,line 214)在库内从未被赋值;工厂 `_build_rca_engine`(`factory.py:394-401`)构造时不传该参数,故 `if provider is None: return []`(line 581)恒短路,try/except 与非 list 回退不可达。
- 验证者证据:全库 grep 无 `.extra_evidence_provider =`/setattr/`dataclasses.replace`/`**kwargs` 注入;`build_reactor_components` 的 `rca=` 逃逸口从未被非测试调用方使用;唯一注入在测试辅助 `_engine(...)`。
- 置信度:high。

**`src/hyperloom/agents/robustness/signals/local_health.py:407` — `_ray_head_symptoms / _fd_pressure_symptoms`(default_when_missing,getattr 守卫)**
- 触发条件:`getattr(data, 'local_ray'/'local_fd', None)` 缺失/None。
- 现状为何不再触发:`SourceData`(`sources/base.py:76-77`)中 `local_ray`/`local_fd` 均以 `field(default_factory=dict)` 声明,任何真实实例上恒为 dict,故 getattr 的 None 默认与 `not isinstance(...,dict)` 分支死;仅 `or not ray_info` 的空 dict 检查有效。
- 验证者证据:`SourceData` 无 `__slots__`/`__getattr__`/子类;数据路径单一(`Reactor.tick -> DegradeRouter.collect`);所有 `SourceData(...)` 构造点要么省略两字段(默认 {})要么由 typed `dict[str,Any]` 探针填充,永不为 None;无 from_dict/replace/pickle 重建。
- 置信度:high。

### 2.8 Critic agent(agents/critic)

**`src/hyperloom/agents/critic/runtime/decision_reviewer.py:1091` — `DecisionReviewer._commit_coordinator_inbox`(legacy_key)**
- 触发条件:review dict 无 `review_verdicts`,回退读遗留 `verdicts`,仍非 list 则抛 `ReviewValidationError`。
- 现状为何不再触发:所有生产者均产 `review_verdicts`:`critic_agent.py` 的 OUTPUT_INSTRUCTIONS、regex `_BARE_JSON_RE`(line 109)、空默认 `{"review_verdicts": []}`(lines 648/1088)、`extract_first_json_with_key(..., "review_verdicts", ...)`(`jsonio.py:151`,其 `_qualifies` 拒绝无该顶层键的对象)。唯一触发是手造 `--review` 文件传入低层 CLI。
- 验证者证据:该 dict 在进程内构建并序列化(line 657)后再跑 `commit-review`,模型文本不会成为自由文件;robustness/critic_mock 走 `build_envelope_dict`/`Intent()` 不经此函数;git blame 显示双读自文件诞生(`db087a4d`)即为防御性。
- 置信度:high。

**`src/hyperloom/agents/critic/runtime/cli.py:162` — `_cmd_list_priors`(legacy_key)**
- 触发条件:packet 无 `context` 键,回退 `environment` 键,再到 `{}`。
- 现状为何不再触发:`environment` packet 键**在全库无任何生产者**;所有构造方只填 `context`。(注:验证者驳回了原判的一个子说法——`list-priors` 并非调用死码,它在 Critic SKILL bash 白名单 `AGENTS.md:81` 中可被 LLM 调用;但 agent/运维只产 `context`,故 `context` 主分支恒胜。)
- 验证者证据:全库(.py/.json/.yaml/.sh/.rst/.toml)`git grep '"environment"'` 仅命中 cli.py 三处读取点;`git log -S` 无该键历史写入。仅遗留 `environment` 分支无 live 触发。
- 置信度:high。

**`src/hyperloom/agents/critic/runtime/cli.py:200` — `_cmd_write_verdict`(legacy_key)**
- 触发条件:同上(packet 无 `context`,回退 `environment`)。
- 现状为何不再触发:无 in-repo 生产者写 `environment` 键;且 `write-verdict` 子命令无程序化调用方——orchestrator 真实 critic 后端只以硬编码 `prepare-review`/`commit-review` 调 `python -m ...cli`。`KBWriter.write_verdict` 方法有真实调用方但走进程内路径,不经此 CLI 行。
- 验证者证据:live packet schema(`decision_review_schema.md`/`verdict_schema.md`)与测试 fixture `_PACKET` 均用 `context`;`grep 'write-verdict'`(cli.py/tests/docs 外)全空。
- 置信度:high。

**`src/hyperloom/agents/critic/runtime/cli.py:228` — `_cmd_write_kb_drafts`(legacy_key)**
- 触发条件:同上。
- 现状为何不再触发:无顶层 `environment` 键的生产者(所有生产者用 `context`,连 `draft_kb.md` 也指示把 environment 元数据放进 `context` 字段)。`write-kb-drafts` 子命令仍可被 SKILL LLM 调用(非严格 CFG 死码),但 `environment` 臂只能由无 in-repo 调用方产出的输入格式选中。
- 验证者证据:全库(dict 字面量/JSON/.md/fixture/shell/cron)搜 `environment` 顶层键仅命中三个回退读点本身;测试 fixture `_PACKET` 用 `context`。仅遗留臂无 live 触发,只能靠手写/库外遗留 packet。
- 置信度:high。

**`src/hyperloom/agents/critic/runtime/decision_reviewer.py:1017` — `DecisionReviewer._review_constraints`(default_when_missing)**
- 触发条件:proposal action class 不在 `_CLASS_RANK` 中。
- 现状为何不再触发:`cls` 恒来自 `classify_proposal_action()`,其经 AST 提取 + 运行时探测(含 None/int/空/未知输入)只返回三个常量(patch_landing/framework_op/evidence_producer),而这三个都是 `_CLASS_RANK`(lines 177-181)的键,故 `.get` 默认永不选中。
- 验证者证据:`sys.settrace`/运行时 `default arm ever needed: False`;全库无对 `_CLASS_RANK`/常量的外部引用、monkeypatch、item-assignment、`__getattr__`/importlib/globals 变更、重复定义;模块级常量跨进程一致。
- 置信度:high。

### 2.9 Quantization agent(agents/quantization)

**`src/hyperloom/agents/quantization/driver/retry.py:329` — `quantize_via_prompt`(if_else_default)**
- 触发条件:`DEFAULT_QUARK_GIT_URL` 为假值(空串)时,三元不追加 clone 提示。
- 现状为何不再触发:该常量赋非空字面量 `"https://github.com/amd/Quark.git"`(line 59)且从不重赋,else 分支永不执行。
- 验证者证据:`grep -rn DEFAULT_QUARK_GIT_URL --include=*.py` 仅两处(赋值 + 读取);从不被导入他处/setattr/globals;无 `__getattr__` shim;测试只 monkeypatch `_ask_operator`。
- 置信度:high。

**`src/hyperloom/agents/quantization/driver/runner.py:338` — `run_one_attempt`(try_except)**
- 触发条件:第二次 options 构造因非 env 原因(`env` 不在 kwargs)抛 TypeError,则 `raise env_exc from exc`。
- 现状为何不再触发:`env` 在 line 313 经 `_quark_py310_compat_env(workspace)` 无条件加入 kwargs 且在 line 332 前从不被移除,故 `if "env" in kwargs` 恒 True,else(line 338)死;仅当未来使 env 可选才可能触发。
- 验证者证据:kwargs 为 fresh 局部 dict;lines 313–332 间仅有对 `model`/`cwd` 的增删,`grep` 确认从不删 `"env"`;`**kwargs` 展开不改调用方 dict。
- 置信度:high。

**`src/hyperloom/agents/quantization/driver/assessment.py:452` — `build_assessment`(default_when_missing)**
- 触发条件:`attempts` 列表为空。
- 现状为何不再触发:唯一生产调用方 `quantize_via_prompt`(`retry.py:393`)经 `while True:` 循环(line 342)至少执行一次并在 `break`(line 383)前无条件 `attempts_list.append(outcome)`(line 363),故列表恒非空;守卫为防御性,仅被 `[]` 直调单测触发。
- 验证者证据:循环头到 append 间无 continue/return/break;若 `run_attempt` 提前抛异常则异常传播、不达 `build_assessment`;bootstrap 快路径直接构造 Assessment 不调该函数;唯一 `[]` 调用在 `test_assessment_branches_unit.py:61`。
- 置信度:high。

**`src/hyperloom/agents/quantization/driver/assessment.py:524` — `derive_status`(if_else_default)**
- 触发条件:final outcome 落空所有显式分支(声称穷举)。
- 现状为何不再触发:前置分支覆盖了 `outcomes.py` 中每个 `OutcomeId`;末尾 `return "failed"` 是标注的防御性 catch-all,仅当新增 OutcomeId 未加入类别集时才触发。
- 验证者证据:脚本回放谓词得 `MISSING=set()`;`sys.settrace` 覆盖全部 33 种 final 值执行 line 524 计数 0;`.final` 仅来自 typed `classify_attempt`/`_decide_next_step`/bootstrap;唯一 raw→enum 路径 `_parse_blocked_outcome`(line 180)在 ValueError 时返回 None;仅两处生产 `Assessment(...)` 构造、无 from_dict 重建。
- 置信度:high。

**`src/hyperloom/agents/quantization/driver/retry.py:271`(数据 symbol 误标为 `assessment.py:271` `_decide_next_step`, if_else_default)**
- 触发条件:某 ASK 类 outcome 既非 checkpoint_aborted/eval_gap_exceeded 也不在 ASK_RETRYABLE 中,落到尾部 `non_retryable_ask`。
- 现状为何不再触发:`ASK − {checkpoint_aborted, eval_gap_exceeded} − ASK_RETRYABLE = ∅`,每个 enum 值恰落入一个已消费类别;`outcome` 唯一来源 `classify_attempt`(OutcomeId|None),`_parse_blocked_outcome` 对未知串返回 None;无动态 enum 扩展。仅当未来新增未归类 OutcomeId 才触发。
- 验证者证据:全 member + None 分区计算残集为空;私有函数单一调用方(`retry.py:367`);无测试命中该尾部。
- 置信度:high。

### 2.10 CLI backends 与 re-export(io/cli)

**`src/hyperloom/inference_optimizer/cli/backends.py:232` — `_build_backends`(sdk_graceful,critic 降级)**
- 触发条件:`critic_choice=='agent'` 且 `provider_anthropic_only` 且 `critic_agent_root is None` → 将 CriticAgentBackend 降级为普通 ClaudeBackend。
- 现状为何不再触发:唯一生产调用方(`cli/__init__.py:1916`)在调用前已通过 `_critic_agent_runtime_needed(critic_choice)`(即 =='agent')解析 root,并在 None 时 `sys.exit(2)`(lines 1872-1884);故运行到本函数时 `critic_agent_root` 恒非 None,line 202 分支恒胜。
- 验证者证据:`critic_choice` 域仅 `{"mock","agent"}`(`_resolve_critic_choice` 校验);`_critic_agent_runtime_needed` 与 provider 无关恒对 'agent' 返回 True;唯一触发在直调单测 `test_build_backends_anthropic_only_degrades_to_claude_without_root`。
- 置信度:high。

**`src/hyperloom/inference_optimizer/cli/backends.py:242` — `_build_backends`(default_when_missing)**
- 触发条件:`critic_choice=='agent'`(非 provider-only 分支)且 `critic_agent_root is None` → 抛 ValueError。
- 现状为何不再触发:防御性守卫;生产调用方已在 `_build_backends` 前 `sys.exit(2)`。
- 验证者证据:lines 1872-1884 预校验;lines 1863→1916 间无分支/无 `critic_choice` 重赋值/无跳过 1872 的早退;唯一触发为绕过 CLI 的直调单测。
- 置信度:high。

**`src/hyperloom/inference_optimizer/cli/backends.py:258` — `_build_backends`(default_when_missing)**
- 触发条件:`robustness_choice=='agent'` 且 `robustness_agent_root is None` → 抛 ValueError。
- 现状为何不再触发:生产调用方(`cli/__init__.py:1901-1913`)已解析 `robustness_agent_root` 并在 None 时 `sys.exit(2)`。
- 验证者证据:`robustness_choice` 仅在 line 1898 赋值一次、调用前不重赋;同一 root 值透传;唯一触发为单测 `test_build_backends_robustness_agent_requires_root`。
- 置信度:high。

**`src/hyperloom/inference_optimizer/cli/__init__.py:24` — `shutil/subprocess` re-export(legacy_key)**
- 触发条件:测试 patch `cli.shutil.which` / `cli.subprocess.run`——由 cli 包上再导出的 stdlib 模块属性承接。
- 现状为何不再触发:`.preflight` 拆分后真实用法已移出;`cli/__init__.py` 生产代码无任何 `shutil.`/`subprocess.` 用法(仅注释提及)。
- 验证者证据:全库 grep 显示**无**任何 patch 命中包级属性——所有相关 patch 均针对**子模块**(`preflight.shutil`/`cli_preflight.subprocess`/字符串 `"...cli.preflight.subprocess.run"`);导入包本体的测试从不引用 `cli.shutil`/`cli.subprocess`。`lean-3.MD:649` 标其为 `redundant_wrapper`。
- 置信度:high。

**`src/hyperloom/inference_optimizer/cli/__init__.py:172` — 向后兼容 re-export 块(`__all__` / ~60 私有 helper, legacy_key)**
- 触发条件:外部调用方在 helper 迁入子模块后仍经 `cli.<name>` 导入。
- 现状为何不再触发:该 re-export 块**已不存在**——提交 `44a5b6a8`("Prune inference optimizer compatibility surfaces",2026-07-14,本分支)已删除 178 行,`__all__` 现为 `["main"]`(line 92),line 172 现为无关的 `_objective_summary_for_prompt`。
- 验证者证据:运行时 `import ...cli` 验证所有被删别名 `hasattr` 为 False、`__all__ == ['main']`;所有 in-tree 导入方已迁移到具体子模块;看似命中的测试实际把 `cli` 绑到子模块;唯一残留提及在 `gbrain_remote_client.py:635` 的失效注释。
- 置信度:high。

### 2.11 breakdown 与 REST 客户端(io/breakdown, io/rest+common)

**`src/hyperloom/inference_optimizer/breakdown/collectors/sessions.py:217` — `_load_yaml_dict_safe`(try_except)**
- 触发条件:`import yaml` 抛 ImportError → 跳过 yaml framework_args 回退,返回 None。
- 现状为何不再触发:PyYAML 是硬核心依赖(`pyproject.toml:16` `PyYAML>=6.0`,另见 `ci/requirements.txt:1`);正确安装环境中 `import yaml` 恒成功(装 6.0.3)。
- 验证者证据:collector 在主包**进程内**运行(经 `session_breakdown.py:62` / `phases/close.py`),无 subprocess/隔离 venv 缺 PyYAML 的路径;src/ 下无 `yaml` 模块遮蔽;无 monkeypatch `__import__`。仅坏/半装环境可触发(非 live 代码触发)。
- 置信度:high。

**`inference_optimizer/baseline_comparison/inferencex_client.py:199` — `fetch_rows`(try_except)**
- 触发条件:InferenceX HTTP GET 抛 `InferenceXFetchError` → 线性退避重试后返回 `(None, last_err)`;非 list / 解码错误亦各有回退。
- 现状为何不再触发:`fetch_rows` **无生产调用方**。唯一消费者 `target_analyzer.analyze()` 已改为读取 LLM 生成的 `competitor_target.json`(`target_analyzer.py:299` `research_hints.load_competitor_target(...)`),不再做 live HTTP 拉取;`reference_script.py:81` 直接用兄弟函数 `_fetch_raw`。
- 验证者证据:全库(非测试)`fetch_rows` 仅命中 `__init__.py:32` 再导出与定义/docstring,无 `.fetch_rows(` 调用/import;`git log -S 'fetch_rows('` 对 `target_analyzer.py` 零命中;无 wildcard import/registry 字符串/getattr/entry-point 动态触及。
- 置信度:high。

---

## 3. 存疑(验证者认为仍可达 / 低置信)— 需人工复核

以下条目原被审计标为疑似死码,但对抗式复核判定 **`stillReachable=true`**,**不应删除**,请人工二次核对。

### 3.1 高置信"仍可达"(强反证,基本应保留)

**`src/hyperloom/orchestrator/phases/framework.py:3148` — `_materialize_framework_agent_candidate`(if_else_default,confidence=high)**
- 分歧点:审计认为 `framework_agent_authoring_enabled` 默认 True 且无 CLI/env 置 False,故回退(降级为 raw diff)不可达。
- 验证者反证:审计漏掉了动态 setattr 状态变更通道。`SharedState.apply_changes(changes, allow_core=False)`(`shared_state.py:1651-1688`)对任意非 core 字段 setattr;`framework_agent_authoring_enabled`(line 609)是 dataclass 字段且**不在** `CORE_STATE_FIELDS`;`UPDATE_STATE` intent 是可路由 intent(`intent.py:30`),经 `intent_router.py:_handle_update_state` 调 `apply_changes`;PolicyGate `_validate_update_state`(`gate.py:1015-1046`)只拒 core 键。故 orchestration agent 发 `{"changes":{"framework_agent_authoring_enabled":false}}` 即可在运行时翻转,下一个 author 路由候选即触发降级;且经 resume 的 `from_dict` 持久化。

**`src/hyperloom/orchestrator/knowledge/cortex_t0.py:78` — `_row_best_config_source`(default_when_missing,confidence=high)**
- 分歧点:审计称 `_field_sources`/`_sources` provenance marker 无写入方(Cortex 后端在 PR #757 移除),函数恒返回 `""`,故回退死。
- 验证者反证(**归类混淆**):被审计的 fallback 行为其实是 `return ""`(line 103);它不但可达,而且是当前代码**唯一**结果、每次调用都触发。经 `machine.py:72`/`kb.py:198` → `run_t0_anchor` → `_warm_recipe_source` → `_row_best_config_source` 的 live 路径,所有 row 都不带 marker,故 `return ""` 100% 命中。审计把"marker 分支死"(真)与"`return ""` 回退死"(假)混为一谈。**该回退不可删除。**

### 3.2 高置信"仍可达"— 经写路径 / 持久化(应保留)

**`src/hyperloom/orchestrator/knowledge/recipe_kb/schema.py:452` — `Recipe.from_dict`(legacy_key,confidence=high)**
- 分歧点:审计称无 live writer 把顶层 `framework` 写入 recipe dict,仅 pre-rename 旧行才有。
- 验证者反证:`proposals.py:214-240` 构建 `prior_extras` 的排除集含 `"framework_name"`(line 221)但**不含** `"framework"`(对比 `cortex_t0.py:1020` 正确排除了),故磁盘旧行的遗留 `framework` 幸存进 `extras` → `put_kwargs["extras"]`;`proposals.py:255` 的 `framework_name` 可为空串(SharedState.framework 默认 "")。`local_store.put_recipe` 经 setdefault splat(lines 558-563)把 `framework` 注入 payload_dict 后调 `from_dict`,此时 `framework_name` 假值 + `framework` 有值 → `or d.get("framework")` 触发。**当前写路径确会命中。**

**`src/hyperloom/orchestrator/knowledge/recipe_kb/gbrain_ingest.py:205` — `_best_config_split`(legacy_key,confidence=high)**
- 分歧点:审计称无 live writer 把裸 `envs` 嵌套 map 写入 best_config。
- 验证者反证:`result_recorder.py:875` 的 `_build_recipe_attrs_from_state` 显式把 `"envs"` 从 `current_best` 逐键复制进 best_config(`for key in ("extra_envs","args","envs",...)`);当 `current_best` 有嵌套 `envs` 而无 `extra_envs` 时正好命中 line 205 前置条件。该 best_config 经 `proposals._kb_amend_recipe` → `GbrainMirroringRecipeKB.put_recipe`(内联镜像,`cli/kb.py:134-141`,`RECIPE_KB_MIRROR_MODE=inline`)→ `mirror_recipe` → `_best_config_split`。`current_best` 跨 resume 与 state.json 保留任意键。审计只读到 line 945 未读到 line 875。

### 3.3 低置信(medium)"仍可达"— 依赖持久化旧数据(重点人工复核)

**`src/hyperloom/orchestrator/knowledge/cortex_t0.py:139` — `_warm_recipe_source`(if_else_default,confidence=medium)**
- 验证者结论:经 `schema.from_dict`/`to_dict` 把未知顶层键(含 `_field_sources`/`_sources`)桶入并回吐 `extras`,`LocalRecipeStore.get_recipe`(line 613)返回原始未过滤 dict,`run_t0_anchor` 的排除集(lines 1011-1038)不排除这些 marker → 会重新盖回磁盘。故 pre-#757 composite 后端写下、尚存于 `$USER_DATA_PATH` 的 recipe.json 在 resume/重跑时经 line 143-144 命中,绕过 line 146 回退。**PR #757(a2d36e3f)确删除了唯一写入方,但持久化数据使读取分支仍活。无迁移/清理脚本。**

**`src/hyperloom/orchestrator/knowledge/cortex_t0.py:187` — `_recipe_is_actionable`(legacy_key,confidence=medium)**
- 验证者结论:`result_recorder.py:875` 的按键复制循环会把裸 `args`/`envs` 从 `current_best` 带入 best_config(且可无 `extra_server_args`);`current_best` 经 `kernel.py:979`/`resume.py:308` 逐字复制、经 state.json 往返;resume `resume.py:312-318` 是显式"legacy sessions"恢复分支。故 resume pre-rename 会话会使每次 warm-start 读命中 `extra_server_args or args` 回退。审计的 grep("无 best_config 字面量含裸 args/envs")过窄——裸键是经 line 875 动态注入而非字面量。

**`src/hyperloom/orchestrator/knowledge/cortex_t0.py:211` — `_config_replay_args_envs`(legacy_key,confidence=medium)**
- 验证者结论:同 line 187 的写路径(`result_recorder.py:870-877` 逐字复制遗留键);此外 `dispatcher._v2_to_arbor`(line 207)对每个 remote row 逐字复制 best_config,幂等守卫(lines 176-177)让已是 arbor 形态的 gbrain row 原样透传,外部/旧客户端行的裸 `args`/`envs` 直达 `cortex_t0.py:519`。back-compat 读管线仍接线(`gbrain_ingest._best_config_split` line 205、`_NON_ENV_BEST_CONFIG_KEYS` line 153)。

### 3.4 高置信"仍可达"— 状态反序列化(应保留)

**`src/hyperloom/orchestrator/policy/gate.py:1644` — PolicyGate `integrate_patch` verdict 查询(sdk_graceful,confidence=high)**
- 分歧点:审计只驳回了"SharedState 缺方法"这一形态,认为回退仅对鸭子类型 stub 触发。
- 验证者反证:`except AttributeError` 也捕获**方法体内**抛出的 AttributeError。`get_specialist_patch_verdict`(`explore_state.py:508`)执行 `self.specialist_patch_verdicts.get(sid,"")`;该字段(`shared_state.py:733`)为普通 dataclass 字段,`from_dict`(line 920)`cls(**filtered)` 只按键名过滤、**不做值强转**。若持久化 state.json 中 `"specialist_patch_verdicts": null`,则实例上该字段为 None,`None.get(...)` 抛 AttributeError → 回退触发。这是生产路径(Coordinator resume 经 `load_or_init` → `from_dict`,`coordinator.py:695/756`),验证者已运行时复现。state.json 可被运维编辑并经宽松迁移消费。

> 数据末尾该条 proof 文本在输入中被截断,但 verdict(`stillReachable=true`,`confidence=high`)完整,已据此归类为"应保留"。

---

## 4. 按类别的洞察

### 4.1 遗留键回退(legacy_key)是最大且最一致的死码来源
- **`extra_sglang_args` → `extra_server_args` 重命名已彻底完成**(baseline.py:1181、profile.py:786、kernel-tools `_payload_aliases.py:94`)。写入侧被 `21a40b6f`/`0f601f2c` 移除,并由 CI 静态守卫 `test_no_legacy_writer_sites.py` 锁死——这是"回退可安全删除"的**最强模式**:有提交证据 + 有主动 CI 守卫防止回潮。
- **`framework` → `framework_name` 重命名则更微妙**:读取侧回退在 orch/knowledge 广泛分布(dispatcher.py:200、schema.py:452、local_store.py:878、gbrain_ingest.py:205/271、cortex_t0.py:187/211)。其中**一部分确已死**(dispatcher.py:200、local_store.py:878、gbrain_ingest.py:271——因规范化在读取前完成),但**另一部分仍活**(schema.py:452、gbrain_ingest.py:205、cortex_t0.py:187/211——因 `proposals.py` 排除集遗漏 `framework`,以及 `result_recorder.py:875` 逐字复制遗留键)。**系统性机会:统一 `proposals.py:221` 与 `cortex_t0.py:1020` 的排除集(前者补上 `framework`),并收敛 `result_recorder.py:870-877` 的按键复制列表,即可让这批读取回退真正变死后再删。**
- **critic CLI 的 `environment` 键**(cli.py:162/200/228)三处同源死码:键无任何生产者;可整批处理。

### 4.2 SDK 优雅降级(sdk_graceful)随 SDK floor 上抬而失效
- claude.py:442/606 的 resume/effort/thinking 降级因 `claude-agent-sdk>=0.2.110` pin 而不可满足,回退代码已在 `13fba44a` 删除,仅剩 `_resume_downgraded` 常量字段与陈旧注释/docstring。**清理机会:删除常量字段与 line 595/596 陈旧注释(若确认 llm_calls.jsonl 消费方不依赖该 key)。**

### 4.3 Python 版本守卫(try_except)已被 `requires-python>=3.10` 淘汰
- gate.py:530/2343 的 `is_relative_to` AttributeError 回退是同构死码,作者已标 `# pragma: no cover — Python <3.9`。可作为**零风险批次**一并删除。

### 4.4 "穷举分区"防御式默认(if_else_default / default_when_missing)
- quantization 的 `derive_status`(:524)、`_decide_next_step`(retry.py:271),critic 的 `_review_constraints`(:1017),prompt_builder 的 `_filter_actions`(:265),shared_state 的 `record_action_attempt`(:1902)——均为"枚举/封闭集完全覆盖 + 防御性 catch-all"。这些**仅在未来新增枚举成员/action 而漏配时才活**。删除会移除对未来疏漏的静默兜底,**建议保留或改为显式 assert/raise**,而非静默删除(风险取舍需人工定夺)。

### 4.5 陈旧 docstring 描述并不存在的回退
- sub_agent_runner.py:10(backend.run() 回退)、canonical_id.py:86-88(6 段 padding)——代码里根本无此分支,docstring 是唯一"证据"。**清理动作是修正文档,而非删代码。**

### 4.6 无调用方的整块死码
- inferencex_client.py:199 `fetch_rows` 整个函数(及其重试/解码/非 list 三个回退)因 `analyze()` 改读 `competitor_target.json` 而无生产调用方。
- cli/__init__.py:172 re-export 块**已被 `44a5b6a8` 删除**——本条实为"已完成的清理",报告中仅作确认。

### 4.7 依赖持久化旧数据的读取回退是最难判定的一类
- cortex_t0.py:139/187/211、schema.py:452、gate.py:1644 的共性:写入侧已收敛,但 **`from_dict`/`load_or_init` 对 state.json / recipe.json 只按键过滤、不做值规范化或迁移**,使得磁盘上的旧数据(或运维手改的 null 值)仍能在 resume 时喂活读取回退。**系统性根因与机会:在 `SharedState.from_dict` 与 `Recipe.from_dict` 增加针对性字段规范化/迁移(如 null→{}、`framework`→`framework_name`、剥离 `_field_sources`),之后这批读取回退才可安全删除。**

---

## 5. 建议的删除批次

按风险从低到高分组;每批标注耦合项(回退 + 其专属测试)。

### 批次 A(零风险)— Python 版本守卫死码
- `gate.py:530`、`gate.py:2343`(两处 `is_relative_to` AttributeError 回退)。
- 耦合:两者结构同构,可同一 PR;均已 `# pragma: no cover`,无专属测试断言其行为(现有测试仅走 happy path)。删除后应确认覆盖率不倒退。

### 批次 B(低风险)— 已有 CI 守卫锁死的 `extra_sglang_args` 遗留读取
- `baseline.py:1181`、`profile.py:786`(orchestrator 侧已无回退代码,主要是确认无残留);`agents/kernel/tools/_payload_aliases.py:94` 的 `read_extra_server_args` 遗留分支。
- 耦合:`test_payload_aliases_shim.py`(shim 测试)与静态守卫 `test_no_legacy_writer_sites.py`。删除 shim 的遗留读取分支需同步调整/移除 `test_payload_aliases_shim.py` 对该分支的断言;`test_no_legacy_writer_sites.py` 的白名单可相应收窄。注意 shim 自述"一版只读兼容",删除前确认对外部/磁盘旧 `kernel_candidates.json` 的兼容窗口已过。

### 批次 C(低风险)— 无调用方 / 已删块 / 陈旧文档
- `inferencex_client.py:199` `fetch_rows` 整函数 + `baseline_comparison/__init__.py:32` 再导出。耦合:其单测(数据未列名,需搜 `fetch_rows` 测试)与 `__all__` 条目一并移除。
- sub_agent_runner.py:10、canonical_id.py:86-88、claude.py:595/596 的陈旧注释/docstring:仅改文档,无代码/测试耦合。
- cli/__init__.py:172:已在 `44a5b6a8` 删除,仅需在报告/清单中标记完成。

### 批次 D(中风险)— 已确认死但涉及多文件遗留键读取
- orch/knowledge `framework` 读取回退中**已确认死**的:`dispatcher.py:200`、`local_store.py:878`、`gbrain_ingest.py:271`、`dispatcher.py:433`。
- 耦合专属测试:`test_v2_to_arbor_reads_legacy_framework_label`(dispatcher.py:200)、`test_local_recipe_store.py:53`(`_matches_labels`)、`test_cid_to_path_components_rejects_legacy_6_segment`(canonical_id.py:111,若一并处理)。删除回退需同步删除/改写这些直调单测,否则测试会失败。
- critic `environment` 三处(cli.py:162/200/228)可与本批同处理,但注意 `list-priors`/`write-kb-drafts` 子命令本身仍被 SKILL/LLM 可调用——**只删 `environment` 遗留臂,保留 `context` 主路径与 `or {}` 默认**。

### 批次 E(需先改造再删,或保留)— 依赖持久化旧数据 / 防御式穷举
- **暂不删**:第 3 节全部 `stillReachable=true` 条目(framework.py:3148、cortex_t0.py:78/139/187/211、schema.py:452、gbrain_ingest.py:205、gate.py:1644)。
- 前置改造(见 4.1 / 4.7):统一 `proposals.py:221` 排除集补 `framework`、对齐 `result_recorder.py:870-877` 复制列表、在 `SharedState.from_dict` / `Recipe.from_dict` 加字段规范化与迁移;完成并加迁移守卫测试后,schema.py:452、cortex_t0.py:187/211、gbrain_ingest.py:205 等读取回退方可进入删除候选。
- **建议保留(改 assert 而非删)**:防御式穷举默认 `derive_status:524`、`retry.py:271`、`_review_constraints:1017`、`_filter_actions:265`、`record_action_attempt:1902`——它们守护未来枚举/action 扩展时的疏漏,静默删除会降低健壮性。

> 交付说明:所有结论均引自输入数据中验证者的 `verdict` 字段与证据,未新增任何数据外发现。第 2 节 37 条为 `stillReachable=false`/`confidence=high`,可按批次 A–D 推进;第 3 节 8 条为 `stillReachable=true`,归入批次 E 并需人工复核。

---

# 附录:完整结构化数据(未截断)

> 源自工作流验证者 `verdict` 字段,file:line 与 proof 全文。


## 附录 A — 39 处确认不可达(完整证据)


### A1. `inference_optimizer/baseline_comparison/inferencex_client.py:199` — fetch_rows
- 类别: `try_except`  | stillReachable: **False** | 置信度: high
- deadReason: fetch_rows has NO production caller. The only consumer, target_analyzer.analyze(), was rewritten to read LLM-authored competitor_target.json via research_hints.load_competitor_target() instead of doing a live HTTP pull (git history shows analyze() has always used load_competitor_target in its current form, never fetch_rows). fetch_rows now appears only in the __init__.py re-export and in tests. reference_script.py uses the sibling _fetch_raw directly, not fetch_rows, so the retry/decode/non-list fallback branches inside fetch_rows can no longer fire in production.
- 验证者 proof: No production caller reaches fetch_rows, so its retry/decode/non-list fallback branches (inferencex_client.py:199+) cannot fire. Evidence: (1) grep for 'fetch_rows' across the whole repo (all extensions, non-test) yields only the __all__ re-export in baseline_comparison/__init__.py:32,45 and the definition/docstring/__all__ in inferencex_client.py:57,171,230 — no '.fetch_rows(' call and no 'import ... fetch_rows' outside tests. (2) The only documented consumer, target_analyzer.analyze(), reads research_hints.load_competitor_target(Path(session_dir)) at target_analyzer.py:299 and imports only DEFAULT_BASE_URL from inferencex_client (line 37); git log -S 'fetch_rows(' on target_analyzer.py returns 0 hits, so analyze() never called it. (3) reference_script.py:81 imports the sibling _fetch_raw directly ('from .baseline_comparison.inferencex_client import _fetch_raw'), not fetch_rows, so it bypasses the fetch_rows fallback entirely; its own errors are caught by the broad except at reference_script.py:83. (4) The target_analysis executor (orchestrator/actions/executors/target_analysis.py:22,161,184) imports and calls analyze directly, not the package __init__ and not fetch_rows. (5) No dynamic reach: no wildcard imports of baseline_comparison, no 'fetch_rows' string literal for registry/dispatch, no getattr resolving to it, no python -m/console-script entry point. InferenceXFetchError is still raised by _fetch_raw but its only live catcher is reference_script's broad handler; the specific except-InferenceXFetchError retry/backoff/(None,last_err) block lives solely inside the uncalled fetch_rows.

### A2. `src/hyperloom/agents/critic/runtime/cli.py:162` — _cmd_list_priors
- 类别: `legacy_key`  | stillReachable: **False** | 置信度: high
- deadReason: The 'environment' packet key has no producer anywhere in src/hyperloom (grep shows only the three cli.py read sites). These fallbacks live only in the low-level operator CLI commands (list-priors/write-verdict/write-kb-drafts), which the module docstring labels 'kept for backward compat / tooling' and which have zero programmatic callers in-repo. Only reachable via manual operator invocation with a legacy-shaped packet.
- 验证者 proof: The fallback under review is the `packet.get("environment")` legacy-key branch at cli.py:162 (the trailing `or {}` is a separate always-live default). For this branch to fire, a packet must have a truthy 'environment' key AND a falsy/absent 'context' key. No such packet is produced anywhere. Repo-wide `git grep '"environment"'` across *.py/*.json/*.yaml/*.yml/*.sh/*.rst/*.toml returns ONLY the three consumer sites in cli.py (162,200,228) — zero producers in src, tests, JSON fixtures, shell scripts, or agent-instruction markdown. `git log -S` for a historical write of the 'environment' key in critic paths (all branches) also returns nothing; it was a defensive alias that never had a live counterpart. I did refute one sub-claim of the auditor: `list-priors` is NOT invocation-dead — it is on the Critic SKILL bash allowlist (agents/critic/AGENTS.md:81), so the Critic LLM agent can invoke it live, and the orchestrator/agent constructs the request via `request.context` (roles/critic_agent.py:302,774). But that only ever populates 'context', never 'environment', so the primary `packet.get('context')` branch always wins whenever data is present, and neither the agent nor any operator tooling produces a packet with 'environment'. The decision_reviewer.py references to list_priors/write_verdict/write_kb_drafts are direct in-process KBWriter method calls, not CLI subcommands, so they never traverse line 162. Conclusion: the 'environment' legacy-key branch specifically has no live trigger; reaching it requires a hand-crafted legacy packet shape that nothing in the current codebase, tests, docs, or git history ever emits.

### A3. `src/hyperloom/agents/critic/runtime/cli.py:200` — _cmd_write_verdict
- 类别: `legacy_key`  | stillReachable: **False** | 置信度: high
- deadReason: Same as _cmd_list_priors: no in-repo producer writes the 'environment' key, and the write-verdict low-level CLI command has no programmatic caller (backward-compat tooling only).
- 验证者 proof: The fallback at src/hyperloom/agents/critic/runtime/cli.py:200 (packet.get("context") or packet.get("environment") or {}) fires only when an input packet.json has no 'context' key but does have an 'environment' key. Two independent conditions must both hold, and neither does in current code: (1) NO in-repo producer emits an 'environment' packet key. A repo-wide sweep across *.py/*.json/*.md and all file types finds "environment" only as the three READ sites in cli.py (lines 162, 200, 228) plus unrelated uses ("environment variable", os.environ, "required environment", KB "environment metadata" prose). The live packet schemas (references/decision_review_schema.md, verdict_schema.md) use 'context'; even the test fixture _PACKET (tests/test_cli.py:132) uses 'context'. git history shows the last occurrences were removed in the src-layout/legacy-prune refactors. (2) The 'write-verdict' subcommand has no programmatic caller. Its subparser (cli.py:340) dispatches to _cmd_write_verdict via argparse only. The orchestrator's real critic backend (roles/critic_agent.py) invokes python -m hyperloom.agents.critic.runtime.cli with only hardcoded phases "prepare-review" (line 617) and "commit-review" (line 666); _default_runtime_caller branches only on commit-review. No RuntimeCall / subprocess / dynamic phase anywhere in the repo passes "write-verdict". grep for 'write-verdict' outside cli.py/tests/docs returns nothing. The KBWriter.write_verdict method does have a real caller (decision_reviewer.py:1243,1330) but that path builds packet_context in-process and never touches this CLI fallback line. The only trigger is a human operator hand-authoring a packet.json with an 'environment' key and no 'context' key, then running the documented backward-compat CLI — not a live in-code path. Matches the sibling _cmd_list_priors case.

### A4. `src/hyperloom/agents/critic/runtime/cli.py:228` — _cmd_write_kb_drafts
- 类别: `legacy_key`  | stillReachable: **False** | 置信度: high
- deadReason: Same as above: 'environment' key unproduced in-repo and write-kb-drafts is unused backward-compat tooling.
- 验证者 proof: The fallback `packet.get("context") or packet.get("environment") or {}` (src/hyperloom/agents/critic/runtime/cli.py:228, in _cmd_write_kb_drafts) requires an input packet with a falsy/absent top-level `context` AND a truthy top-level `environment` key. I searched the entire repo (Python dict literals, JSON files, .md action/skill files, test fixtures, shell/cron) for anything producing a top-level `environment` key and found NOTHING except the three read-sites of the fallback itself (cli.py:162, 200, 228). Every actual producer uses `context`: orchestrator critic_agent.py:600 (`"context": dict(self._static_context)`), robustness roles (`"context": {...}`), in-process decision_reviewer.py write_kb_drafts calls (`packet_context=req.context`), and the LLM-facing actions/draft_kb.md which instructs the agent to put environment metadata INSIDE the `context` field (line 76) — the word "environment" at lines 17/85 is metadata content, not a packet key. The test fixture _PACKET also uses `context`. The write-kb-drafts subcommand IS still registered/invokable (parser cli.py:347) and is documented for the critic SKILL LLM, so the branch is not unreachable dead code in the strict CFG sense — but the `environment` arm can only be selected by an input format (top-level `environment` key) that no current in-repo caller, LLM instruction, test, or fixture produces. I could not construct any live in-repo trigger; the only path is a hand-authored/out-of-repo legacy packet, which is not current code.

### A5. `src/hyperloom/agents/critic/runtime/decision_reviewer.py:1091` — DecisionReviewer._commit_coordinator_inbox
- 类别: `legacy_key`  | stillReachable: **False** | 置信度: high
- deadReason: Every current producer of the review JSON emits 'review_verdicts', never a bare 'verdicts' key: critic_agent.py OUTPUT_INSTRUCTIONS (line ~63), its regex _BARE_JSON_RE keyed on 'review_verdicts' (line 109), its empty defaults {"review_verdicts": []} (lines 648, 1088), and the SKILL/action docs (actions/review_coordinator_inbox.md). The only way to hit the 'verdicts' branch is a hand-crafted review.json passed to the low-level `commit-review --review` CLI, which no in-repo caller does. Defensive alias, effectively dead under current producers.
- 验证者 proof: The fallback `review.get("verdicts")` fires only when `_commit_coordinator_inbox` gets a review dict missing `review_verdicts` but carrying bare `verdicts`. Every automated path that reaches this function cannot produce that shape: (1) The sole production caller, CriticAgentBackend in orchestrator/roles/critic_agent.py, builds the review dict in-process — either `{"review_verdicts": []}` (lines 648, 1088) or `_extract_review_json(text)` (line 1085), which calls extract_first_json_with_key(text, "review_verdicts", _BARE_JSON_RE) in common/jsonio.py:151; its `_qualifies` (line 179-180) rejects any parsed object that does not contain the top-level key `review_verdicts`. It then serializes THAT dict to review.json (line 657) and runs `commit-review --review`. So the on-disk review always has `review_verdicts`; the model text never becomes a free-form file. (2) The robustness agent (agents/robustness) and critic_mock build intent envelopes directly via build_envelope_dict / Intent(), never routing through commit_review/_commit_coordinator_inbox. (3) Decision-request and KB-draft kinds dispatch to _commit_decision_request/_commit_kb_draft, which use `verdict` singular, not this branch. The only way to inject `{"verdicts": [...]}` is the standalone CLI `commit-review --review <hand-crafted file>`, which no in-repo caller, action registry entry, resume path, or test harness performs (grep for bare `"verdicts": []` yields only an unrelated KB-assess response fixture and test assertions, no review.json producer). Doc references to `verdicts[].status` (review_coordinator_inbox.md:53) are proposal-evidence fields, not the output key (output shape at line 84 is review_verdicts). Git blame shows the dual-read was defensive from file inception (db087a4d), not a live legacy migration.

### A6. `src/hyperloom/agents/critic/runtime/decision_reviewer.py:1017` — DecisionReviewer._review_constraints
- 类别: `default_when_missing`  | stillReachable: **False** | 置信度: high
- deadReason: cls always comes from classify_proposal_action() which only ever returns one of the three constants (patch_landing/framework_op/evidence_producer), and all three are keys in _CLASS_RANK (lines 177-181). The .get(...) default can never be selected given the current classifier. Defensive only.
- 验证者 proof: The default arm at decision_reviewer.py:1017 (_CLASS_RANK.get(cls, _CLASS_RANK[ACTION_CLASS_EVIDENCE_PRODUCER])) is dead. `cls` has exactly one binding in the _review_constraints loop: line 1015 `cls = classify_proposal_action(p.action_name)` (verified by awk/grep — the only other `cls` is a separate comprehension-scoped var at line 1024 that never feeds line 1017). classify_proposal_action (lines 190-211) has five return statements; AST extraction and runtime probing both confirm it returns exactly the three string literals {evidence_producer, patch_landing, framework_op}, including for None/int/empty/whitespace/unknown inputs (the isinstance/empty guards route those to evidence_producer). _CLASS_RANK (lines 177-181) keys are exactly {framework_op, evidence_producer, patch_landing} — identical to the classifier's output set. So .get(cls, ...) always finds cls and the default is never selected. I independently ruled out every escape hatch: (1) no external references to classify_proposal_action / _CLASS_RANK / the ACTION_CLASS_* constants anywhere in the repo except the module and one test import (grep across all *.py); (2) no monkeypatch.setattr, no reassignment, no dict item-assignment of _CLASS_RANK or the constants (grep repo-wide incl. tests/conftest); (3) no __getattr__ shim, importlib, or globals() mutation in the module; (4) no duplicate/shadowing definitions (exactly 5 matching top-level defs/assigns); (5) module-level constants are identical across every process, so `python -m` subprocess/console-script invocations and mock/real role pairings cannot alter the return set. Runtime check printed `default arm ever needed: False`. Purely defensive code.

### A7. `src/hyperloom/agents/kernel/tools/_payload_aliases.py:94` — read_extra_server_args
- 类别: `legacy_key`  | stillReachable: **False** | 置信度: high
- deadReason: No code path in src/ writes the 'extra_sglang_args' key anymore. A dedicated CI static guard test (src/hyperloom/inference_optimizer/tests/test_no_legacy_writer_sites.py) asserts the literal 'extra_sglang_args' appears in NO git-tracked repo file except this shim and its own tests, so no in-repo writer can put the legacy key into any payload. Only a hypothetical external/operator-authored payload could still trigger it.
- 验证者 proof: I could NOT refute the dead claim for any in-repo trigger. The fallback at _payload_aliases.py:94 fires only when a payload dict contains key 'extra_sglang_args' but not 'extra_server_args'. Its single production caller is kernel_optimization.py:1486, read_extra_server_args(candidate), where `candidate` comes from load_candidates(candidates_path) (kernel_optimization.py:3344) — i.e., json.loads of a kernel_candidates.json file (kernel_optimization.py:112). I traced every in-repo writer of that candidate structure: the trace-analysis constructors (tracelens_analysis.py:4310 and :4396, tracelens_skill_runner.py:1400-1425) and parallel_e2e_runner.py:389 build candidate dicts that NEVER include 'extra_sglang_args' or 'extra_server_args' as a key. The orchestrator kernel handlers (orchestrator/kernel/*.py) only ever read/write the canonical 'extra_server_args' / 'candidate_extra_server_args'. Git history confirms the rename is complete on the write side: prune commit 0f601f2c removed the last legacy reader `or candidate.get("candidate_extra_sglang_args", "")` (kernel_optimization.py, old line 2784) and deleted common/payload_aliases.py. I searched for dynamic key construction (concatenation/f-string/computed keys yielding 'extra_sglang_args') across src/ and found none, so the guard's literal-only scan (test_no_legacy_writer_sites.py:27, which DOES scan .json/.yaml/.toml/etc, not just .py) has no live blind spot to exploit. `git grep extra_sglang_args` returns only the shim + its two tests. No action-registry string key, intent-payload builder, SharedState field, subprocess/CLI arg, console entry point, __getattr__ shim, config/env flag, mock/real role pairing, or LLM prompt template emits the legacy key. The ONLY surviving triggers are external: an operator passing --reuse-candidates-from / --candidates-path at a stale pre-rename kernel_candidates.json on the workspace disk (contents pass through load_candidates unfiltered into `candidate`), or an operator-authored payload with the legacy key. Both are exactly the 'hypothetical external/operator-authored payload' the auditor already conceded and match the shim's stated one-release read-only compat purpose. No in-repo code path can put the legacy key into any payload.

### A8. `src/hyperloom/agents/quantization/driver/assessment.py:452` — build_assessment
- 类别: `default_when_missing`  | stillReachable: **False** | 置信度: high
- deadReason: The only production caller (quantize_via_prompt) always appends at least one outcome to attempts_list before calling build_assessment: the while loop runs run_attempt+classify_attempt+append (retry.py:363) at least once and only breaks after. So attempts is never empty on that path; the guard is defensive and only hit by a direct unit call with [].
- 验证者 proof: The empty-attempts guard at assessment.py:452 has exactly one production caller of build_assessment: quantize_via_prompt at retry.py:393 (confirmed via repo-wide grep; all other references are unit tests). That function reaches line 393 only through a `while True:` loop (retry.py:342) that always executes at least once and unconditionally runs `attempts_list.append(outcome)` at line 363 before the sole `break` (line 383) — there is no continue/return/break between the loop head and the append, so attempts_list is guaranteed len>=1 at line 393. classify_attempt returns OutcomeId|None; even a None outcome is appended, so the list is non-empty (the guard tests emptiness, not None). Other exit paths cannot trigger it: if run_attempt raises before the append, the exception propagates out of quantize_via_prompt and never reaches build_assessment; the bootstrap fast-path (retry.py:324 _build_failed_bootstrap_result) constructs an Assessment directly and does not call build_assessment. No dynamic dispatch (__getattr__/importlib/string registry) reaches build_assessment. The live wiring is real (orchestrator quantization_request_handlers.py:72 and cli.py:143 both call quantize_via_prompt), but every live path arrives with a populated list. The only [] invocation is the standalone unit test test_assessment_branches_unit.py:61 (build_assessment([], ...)). The auditor's dead-reason is corroborated and I found no additional live trigger.

### A9. `src/hyperloom/agents/quantization/driver/assessment.py:524` — derive_status
- 类别: `if_else_default`  | stillReachable: **False** | 置信度: high
- deadReason: The preceding branches cover None, eval_gap_accepted, AUTO_FAIL, eval_gap_exceeded, must_validate_skipped, AUTO_RECOVER, ASK, and unclassified_failure — together these enumerate every OutcomeId in outcomes.py. The final `return 'failed'` is explicitly labeled 'Defensive — partition is exhaustive'. It only fires if a new OutcomeId is added without updating a category set.
- 验证者 proof: The defensive `return "failed"` at src/hyperloom/agents/quantization/driver/assessment.py:524 is unreachable in current code. (1) Enumeration: a script that replays the branch predicates over set(OutcomeId) plus None reports MISSING=set() — every member of the enum is covered by an earlier branch (None/eval_gap_accepted→success; AUTO_FAIL→failed; eval_gap_exceeded→partial; must_validate_skipped→failed/partial; AUTO_RECOVER→partial/failed; ASK or unclassified_failure→failed; note upstream_change_required∈AUTO_FAIL and eval_gap_accepted in the success branch, covering the two derived tags). (2) Runtime trace: sys.settrace over the REAL derive_status across all 33 possible final values (32 OutcomeIds + None) × strict/config/tokenizer permutations records line 524 executed 0 times. (3) Input surface is closed: assessment.final is only populated from classify_attempt (typed/returns OutcomeId|None, every return is a genuine member), from _decide_next_step.promote_to (OutcomeId|None), or from bootstrap outcomes in _build_failed_bootstrap_result (which sets status='failed' directly and does not even route through derive_status). The single raw-text→enum path, _parse_blocked_outcome at assessment.py:180, wraps OutcomeId(raw) in try/except and returns None on ValueError, so no arbitrary string can become final. (4) No alternate construction: repo-wide grep finds only two production Assessment(...) sites (assessment.py:466, retry.py:431), one production derive_status caller (retry.py:395), no from_dict/JSON deserializer that rebuilds .final, and no mock-role/subprocess/__getattr__ path injecting a value. The only way the fallback fires is a source edit adding a new OutcomeId to outcomes.py without adding it to a category set — an author-time condition, not a runtime/config/env/CLI/data trigger.

### A10. `src/hyperloom/agents/quantization/driver/assessment.py:271` — _decide_next_step
- 类别: `if_else_default`  | stillReachable: **False** | 置信度: high
- deadReason: ASK = {checkpoint_aborted, exec_oom, export_crashed, must_have_weights_missing, eval_gap_exceeded, fuzzy_check_failed}. The function handles checkpoint_aborted and eval_gap_exceeded explicitly, and ASK_RETRYABLE = {exec_oom, export_crashed, must_have_weights_missing, fuzzy_check_failed} covers the remaining four. So every ASK member is consumed before the tail `non_retryable_ask` return; the comment even states 'none currently'.
- 验证者 proof: The fallback tail is actually at src/hyperloom/agents/quantization/driver/retry.py:271 (return _RetryDecision(retry=False, note=f"non_retryable_ask:{outcome}")) inside _decide_next_step — the claim's path assessment.py:271 is misattributed, but I evaluated the real branch. To reach it, `outcome` must survive all prior branches (retry.py:226 None/SUCCESS_TAGS, 228 AUTO_FAIL, 230 AUTO_RECOVER, 236 checkpoint_aborted, 241 eval_gap_exceeded, 255 ASK_RETRYABLE|UNCLASSIFIED_FAILURE). I computed the full partition over every OutcomeId member plus None: the residual set that reaches the tail is empty, and ASK − {checkpoint_aborted, eval_gap_exceeded} − ASK_RETRYABLE = ∅. Per-member mapping confirms each enum value lands in exactly one consumed class; the two narrative tags are covered (eval_gap_accepted∈SUCCESS_TAGS, upstream_change_required∈AUTO_FAIL) and unclassified_failure is handled explicitly at line 255. The `outcome` argument has only one production source: classify_attempt (return type OutcomeId|None), whose every return is a literal enum member or None — all consumed. _parse_blocked_outcome does OutcomeId(raw) in a try/except that returns None for any unknown string, so it cannot inject a non-member. There is no dynamic enum extension (no aenum/extend_enum/_missing_), so OutcomeId cannot gain an uncovered member at runtime. _decide_next_step is private with a single caller (retry.py:367 in quantize_via_prompt); all external entry points (cli.py:143, orchestrator quantization_request_handlers.py:72, __init__.py) route through quantize_via_prompt, and promote_to never re-enters the decision. No test hits the non_retryable_ask tail. The only way to fire it is a FUTURE change adding a new OutcomeId not placed in any retry-class set — not reachable in current code.

### A11. `src/hyperloom/agents/quantization/driver/retry.py:329` — quantize_via_prompt
- 类别: `if_else_default`  | stillReachable: **False** | 置信度: high
- deadReason: DEFAULT_QUARK_GIT_URL is a module constant assigned a non-empty literal 'https://github.com/amd/Quark.git' (retry.py:59) and is never reassigned, so the ternary's else-branch (omit the clone-hint) can never execute.
- 验证者 proof: DEFAULT_QUARK_GIT_URL is assigned exactly once at src/hyperloom/agents/quantization/driver/retry.py:59 to the non-empty literal "https://github.com/amd/Quark.git". A repo-wide grep (`grep -rn DEFAULT_QUARK_GIT_URL --include=*.py`) returns only two hits: the assignment (line 59) and the read inside the ternary (line 329). It is never reassigned, never imported into another module (so it can't be shadowed/rebound elsewhere), never set via setattr/globals(), and there is no __getattr__ shim in the module/package. Tests monkeypatch only `_ask_operator`, never the URL constant. The ternary `(f", clone from {DEFAULT_QUARK_GIT_URL}" if DEFAULT_QUARK_GIT_URL else "")` therefore always evaluates the truthy branch; the else-branch (empty string, no clone hint) cannot execute. No config/env/CLI flag mutates this Python module constant, and no subprocess/agent/entry-point path changes its value. The auditor's dead-reason is correct.

### A12. `src/hyperloom/agents/quantization/driver/runner.py:338` — run_one_attempt
- 类别: `try_except`  | stillReachable: **False** | 置信度: high
- deadReason: env is added unconditionally at runner.py:313 (_quark_py310_compat_env always sets env in kwargs) and is never popped before this point, so 'env' in kwargs is always True when line 332 is evaluated; the else-branch `raise env_exc from exc` at line 338 is dead unless a future change makes env optional.
- 验证者 proof: Confirmed dead. In run_one_attempt (src/hyperloom/agents/quantization/driver/runner.py), kwargs is a fresh local dict (line 308) with "env" added unconditionally at line 313 via _quark_py310_compat_env(workspace), which always returns a dict[str,str] (line 148) and cannot omit the key. Between line 313 and the guard at line 332, the only kwargs mutations are: conditional add of "model" (315-316), add of "cwd" (317), and kwargs.pop("cwd", None) (324). A repo-file grep for `del kwargs|kwargs =|pop(|"env"|'env'` returns only lines 313 (set), 324 (pop cwd), 332 (guard) — nothing ever removes "env". sdk_options_cls(**kwargs) unpacks/copies kwargs into the callee, so no callee can mutate the caller's dict either. Thus when the innermost `except TypeError as env_exc` runs, `"env" in kwargs` is always True, the RuntimeError branch (333-337) always fires, and `raise env_exc from exc` at line 338 is unreachable. It could only become live if a future change made env conditional/poppable — exactly the caveat in the code comment and the auditor's report. I could not construct any live trigger.

### A13. `src/hyperloom/agents/robustness/decision/rca_engine.py:585` — _safe_extra_evidence
- 类别: `try_except`  | stillReachable: **False** | 置信度: high
- deadReason: LlmRcaEngine.extra_evidence_provider defaults to None and is never populated anywhere in-tree — factory._build_rca_engine constructs the engine without passing extra_evidence_provider, so the guard `if provider is None: return []` (line 581) short-circuits before the try/except can run. The except/non-list fallback branches are only reachable if a caller injects a provider, which no current code does.
- 验证者 proof: extra_evidence_provider (dataclass field default None at rca_engine.py:214) is only ever set via the LlmRcaEngine/AnthropicRcaEngine constructor. The sole production builder, factory._build_rca_engine (factory.py:394-401), constructs engine_cls(base_url, api_key, model, timeout_s, max_chars, throttle) and omits extra_evidence_provider, so it stays None. Repo-wide grep finds NO `.extra_evidence_provider =` assignment, no setattr, no dataclasses.replace, and no `**kwargs` splat into the engine. The one injection escape hatch — the `rca=` override param of build_reactor_components (factory.py:94,309-311) — is never exercised: all non-test callers (main.py:50 `build_reactor_components(config)`, runtime/cli.py:231 `build_reactor_components(config, session_id=session_id)`) pass no rca, and no test passes rca= either. No config key, env var, or CLI flag feeds a provider (all `evidence=` hits are the unrelated Symptom.evidence dict). Therefore `if provider is None: return []` (line 581) always short-circuits before the try (584) executes, making the except fallback (585-587) and non-list fallback (588-589) unreachable. The only place the provider is populated is the private test helper _engine(...) in test_decision_rca_engine.py:470, not any production path.

### A14. `src/hyperloom/agents/robustness/signals/local_health.py:407` — _ray_head_symptoms / _fd_pressure_symptoms (getattr guards)
- 类别: `default_when_missing`  | stillReachable: **False** | 置信度: high
- deadReason: SourceData is a dataclass that always defines local_ray and local_fd with default_factory=dict (sources/base.py:76-77), so getattr never hits its default and the attribute is always a dict. The `None` fallback branch of the getattr is dead for any real SourceData instance; only the emptiness check (`or not ray_info`) does real work.
- 验证者 proof: I could not find any live trigger for the specific fallback the auditor names (the getattr(..., None) default and the `not isinstance(..., dict)` branch at local_health.py:407-408 and :450-451). Evidence: (1) SourceData (sources/base.py:54-96) is a plain @dataclass with no __slots__ and no __getattr__ shim; local_ray/local_fd both use `field(default_factory=dict)` (:76-77), so getattr always resolves the real attribute and its None default is unreachable. (2) No subclass of SourceData exists anywhere in the repo. (3) Data path is single: Reactor.tick (reactor.py:110) calls self._router.collect(ctx); _router is strictly typed DegradeRouter (reactor.py:40, factory.py:197); collect()/_fetch_fallback always return a real SourceData. classify (classifier.py:142/161) -> evaluate_local_health_signals (local_health.py:111-119) -> _ray_head_dead_symptoms/_fd_pressure_symptoms are the ONLY callers, always with that SourceData. (4) Every SourceData(...) construction (base.py:260,267; factory.py:473; server_client.py:490; local_probe.py:330) either omits local_ray/local_fd (defaulting to {}) or sets them from _probe_ray_head/_sample_fd_usage, both of which are typed dict[str,Any] and every return path yields a dict — {} or a populated dict, never None (local_probe.py:989-1121). (5) No deserialization/from_dict/replace/**unpack/pickle/json reconstruction of SourceData exists (grep found none), so local_ray/local_fd cannot be smuggled in as None or a non-dict. (6) No mock or alternate classifier feeds a duck-typed `data`. The only live part of the guard is the `or not ray_info` / `or not fd_info` empty-dict check (fires when ray/fd probing is disabled or returns {}), which the auditor already concedes does real work. The None-default and non-dict-isinstance sub-branches are genuinely dead defensive code.

### A15. `src/hyperloom/agents/robustness/sources/local_probe.py:393` — _read_coordinator_events / _try_select
- 类别: `legacy_key`  | stillReachable: **False** | 置信度: high
- deadReason: The second SELECT targets a legacy events schema (columns id/agent/timestamp) that the current Coordinator DB no longer has. src/hyperloom/orchestrator/bus/storage/schema.py defines events as (seq PK AUTOINCREMENT, msg_id, from_agent, to_agent, topic, in_reply_to, payload, priority, ts) and message_bus.py inserts exactly those columns; there is no writer of an id/agent/timestamp schema anywhere in the tree. The FIRST candidate matches the live schema and always succeeds, so the fallback SELECT is never the one that returns rows. Even if reached it would raise sqlite3.Error (no such column) and yield [].
- 验证者 proof: The fallback (2nd) SELECT at src/hyperloom/agents/robustness/sources/local_probe.py:393 ('SELECT id, agent, topic, payload, timestamp AS ts FROM events ...') is only reached when the 1st candidate raises sqlite3.Error, because _try_select (lines 433-435) returns on the first statement that executes. The 1st candidate ('SELECT seq AS id, from_agent AS agent, topic, payload, ts FROM events ...') matches the live schema. I empirically confirmed against schema.py's events table (seq PK/msg_id/from_agent/to_agent/topic/in_reply_to/payload/priority/ts) that the 1st SELECT succeeds (rows returned) and the 2nd raises 'no such column: id'. Reachability chain: _read_coordinator_events has exactly one caller (local_probe.py:236) with db_path fixed to <session_dir>/storage/coordinator.db (config.py:80, local_probe.py:193) — it cannot be redirected to a foreign DB. The sole production writers of that DB (orchestrator/bus/message_bus.py:186, resource_lock.py:239/375/455, connection.py) all INSERT the current columns, and the table is always created by ensure_schema (schema.py:70-80, called via connection.py:88). I searched the WHOLE tree and full git history (git log --all -S) for any CREATE/INSERT of the legacy id/agent/timestamp/intent_type events schema: the ONLY matches are in test fixtures (robustness tests/test_sources_local_probe.py:62 _seed_coordinator_db(schema='legacy'), and the pre-refactor robustness-agent/tests copy). No production bus code ever used it; schema migrations touch only 'leases', never 'events', so even the oldest real coordinator.db has the seq-PK events shape and hits the 1st SELECT. The fallback returns rows only under the dedicated unit test test_local_probe_reads_legacy_events_schema (line 106); in production it can never be the branch that returns rows, and if reached would raise 'no such column' and yield [].

### A16. `src/hyperloom/agents/robustness/sources/local_probe.py:2095` — _probe_external_mounts
- 类别: `default_when_missing`  | stillReachable: **False** | 置信度: high
- deadReason: All three entries in _EXTERNAL_MOUNT_ENVS have default_path='' (TRACELENS_ROOT/TRACELENS_INTERNAL_ROOT/INFERENCEX_PATH), and the very next lines strip and `if not path: continue`. So the 'default' arg to get() can never contribute a probed path; the else-of-default is a no-op. The only way a mount is probed is if the env var is explicitly set to a non-empty path.
- 验证者 proof: The auditor's claim is correct; I could not refute it. The fallback in question is the default_path VALUE contributing a probe row via os.environ.get(env_name, default_path) at local_probe.py:2095. _EXTERNAL_MOUNT_ENVS (2068-2073) is a single module-level constant with all three defaults hardcoded to "" (TRACELENS_ROOT, TRACELENS_INTERNAL_ROOT, INFERENCEX_PATH). I verified: (1) grep shows the tuple is defined exactly once and never reassigned or monkeypatched anywhere in the repo (no `_EXTERNAL_MOUNT_ENVS =` or `monkeypatch.setattr(...)` matches); (2) git -S history shows these entries only ever existed with "" defaults since the src-layout promotion. Data flow: when an env var is unset, get() returns default_path="" → `"" or ""` → .strip()="" → `if not path: continue` (2097-2098) skips it, producing no row. When an env var IS set to a non-empty value, the row comes from the os.environ value (first arg), NOT the default arm. When set-but-empty/whitespace, .strip() zeroes it and it is skipped. Therefore the default-value arm can never contribute a probed path. The function itself is live (called via _probe_environment→asyncio.to_thread at 1975, used by LocalProbeSource), and test_probe_external_mounts_skips_tracelens_root_when_unset (test file 1359-1370) explicitly asserts zero rows when all vars are unset — codifying the dead default behavior. The docstring "falling back to their defaults" (2082) is stale prose; the code and the constant are definitive. No caller, action-registry key, subprocess -m invocation, entry point, __getattr__ shim, config/CLI flag, or resume/degradation path can make default_path non-empty.

### A17. `src/hyperloom/inference_optimizer/breakdown/collectors/sessions.py:217` — _load_yaml_dict_safe
- 类别: `try_except`  | stillReachable: **False** | 置信度: high
- deadReason: PyYAML is a hard core dependency: pyproject.toml line 16 'PyYAML>=6.0'. In any correctly-installed environment `import yaml` succeeds (verified: yaml 6.0.3 importable). Only triggerable under a broken/partial install.
- 验证者 proof: No in-code trigger removes PyYAML while still running this collector. (1) PyYAML is a MANDATORY core dependency: pyproject.toml:16 'PyYAML>=6.0' (project.dependencies, not optional), also pinned in ci/requirements.txt:1 and docs/sphinx/requirements.txt:169. import yaml succeeds (6.0.3). (2) yaml is imported only locally inside _load_yaml_dict_safe (line 216) and _read_invocation_envs (line 412); the except ImportError at line 217 is the sole guard, but PyYAML is always present. (3) The breakdown collector runs IN-PROCESS in the main hyperloom package: reached via orchestrator/actions/executors/session_breakdown.py:62 'from hyperloom.inference_optimizer.breakdown import build' and orchestrator/phases/close.py. There is NO subprocess or isolated-venv path that runs collectors.sessions under an env lacking PyYAML. The subprocess agents (critic/framework/kernel/robustness) install the same package with the same hard PyYAML dep. (4) No [project.scripts] console entry runs breakdown.collectors.sessions standalone in a stripped env. (5) No 'yaml' module/dir shadowing exists anywhere under src/. (6) No CLI flag, env var, config, mock/real role pairing, resume, or SDK-degradation path removes PyYAML. (7) No test monkeypatches __import__ to hit this branch, and there is no pragma-no-cover on it. The only way to fire the branch is a broken/partial install or a user-supplied yaml.py shadowing on sys.path from the launch cwd -- both environmental corruption, not a live code trigger. The auditor themselves concede 'only triggerable under a broken/partial install.' I could not find any live code path.

### A18. `src/hyperloom/inference_optimizer/cli/__init__.py:24` — cli/__init__ shutil/subprocess re-export
- 类别: `legacy_key`  | stillReachable: **False** | 置信度: high
- deadReason: The module comment states these are re-exported ONLY because tests patch cli.<module> singletons; the .preflight split moved the actual usage out. lean-3.MD (line 649) flags the cli/__init__ re-export hub as redundant_wrapper. Not exercised by any production code path in this package after the split — purely a test-patch-target shim.
- 验证者 proof: The re-export at src/hyperloom/inference_optimizer/cli/__init__.py:24-25 (`import shutil`/`import subprocess`, noqa F401) is genuinely dead as a live trigger. Independent verification:

1. No production code in cli/__init__.py uses either module. The ONLY occurrences of `shutil.`/`subprocess.` in the file are inside the explanatory comment on lines 20-21; the module body has zero real usage (grep confirmed). The `.preflight` split moved all real usage into cli/preflight.py, which imports shutil/subprocess itself (preflight.py:17-18) and calls them directly (e.g. shutil.which at preflight.py:418,464,949; subprocess.run throughout).

2. The comment claims tests patch `cli.shutil.which` / `cli.subprocess.run` / the string `"hyperloom.inference_optimizer.cli.subprocess.run"`. A repo-wide grep shows NO such patch exists. Every shutil/subprocess patch in the suite targets a SUBMODULE, not the package __init__:
   - test_preflight_tracelens_cli_gate.py:20,29,48,62 does `monkeypatch.setattr(cli.shutil, ...)` but line 9 binds `cli` via `from hyperloom.inference_optimizer.cli import preflight as cli` — so `cli.shutil` IS `preflight.shutil`, not the package attribute.
   - test_preflight_auth_override.py:49,60,540,565,1378,... patches `cli_preflight.shutil`/`cli_preflight.subprocess`.
   - test_inferencex_preflight_clone.py:44,67,82,102,119 patches the string `"hyperloom.inference_optimizer.cli.preflight.subprocess.run"` (note the `.preflight.` segment).
   - Other suites patch br.shutil, bl.shutil, rb.subprocess, mn.subprocess, isolation.shutil — all unrelated submodules.

3. Tests that DO import the package itself (`from hyperloom.inference_optimizer import cli`: test_cli_no_explore, test_cli_no_framework_resume, test_cli_atom_auto_tighten, test_phase_budget_pct_cli, test_startup_robustness, test_specialist_concurrent_dispatch) never reference `cli.shutil` or `cli.subprocess` (grep returned nothing).

4. parser.py:3-13 header documents that the split verified via repo-wide grep for `setattr(cli, "<name>"` and fully-qualified `"hyperloom.inference_optimizer.cli.<name>"` string patches — corroborating that the package-level attribute patch pattern the comment cites does not exist.

No caller, no string-path patch, no console-script/entry-point, no __getattr__ shim, no resume/SDK path resolves through cli.shutil/cli.subprocess. The trigger the comment protects against is not present anywhere in current code. Confirmed dead.

### A19. `src/hyperloom/inference_optimizer/cli/__init__.py:172` — cli/__init__ back-compat re-export block (__all__ / private helper re-exports)
- 类别: `legacy_key`  | stillReachable: **False** | 置信度: high
- deadReason: The tree-reform split relocated these helpers into submodules; the cli/__init__ re-export block only preserves the old import paths. lean-3.MD lines 320 and 645 explicitly tag this 'Backward-compat re-exports' block (~60 private helpers) as a legacy cut candidate — indicating no live in-tree importer depends on the aliased path.
- 验证者 proof: The back-compat re-export block the auditor describes (cli/__init__.py:172 with __all__ lines 178-218, ~60 aliased private helpers + `# noqa: F401 - re-exported for callers/tests`) NO LONGER EXISTS. Commit 44a5b6a8 "Prune inference optimizer compatibility surfaces" (2026-07-14, on branch clean/zgong/lean-3) already deleted 178 lines from cli/__init__.py, stripping the entire __all__ list and all aliased imports (_load_model_arch, _detect_unsupported_model, _build_recipe_kb_dispatcher, _NOOP_KINDS_KERNEL_ONLY, _validate_credentials, _resolve_llm_endpoints, etc.). Current file has `__all__ = ["main"]` at line 92; line 172 is now the unrelated _objective_summary_for_prompt.

Whole-repo search for the trigger (callers importing helpers via cli.<name>) found NO live path: (1) all in-tree importers migrated to concrete submodules — from ...cli import model_gate/bootstrap/executors/kb/backends/credentials/preflight/parser; (2) the apparent hits test_multimodal_gate.py (cli._detect_unsupported_model x25) and test_preflight_tracelens_cli_gate.py (cli.shutil) bind `cli` to a SUBMODULE (`import model_gate as cli` / `import preflight as cli`), resolving through the submodule, not the deleted __init__ block; (3) tests that bind the TOP-LEVEL cli package (test_cli_atom_auto_tighten, test_preflight_auth_override, test_startup_robustness, etc.) only access names still defined/imported in the current file (_build_parser, _CATALOG_RETRY_DELAYS_SEC, _build_specialist_executor, _register_executors, _preflight, _resolve_robustness_choice) — none reference a removed alias.

Runtime verification via `import hyperloom.inference_optimizer.cli as cli`: every removed alias -> hasattr False (REMOVED_STILL_PRESENT: []); every top-level-accessed name -> present (MISSING: []); __all__ == ['main']. No __getattr__ shim in cli/__init__. Console script `inference_optimizer = ...cli:main` needs only `main`. Only residual mention of a removed name outside tests is an inert docstring comment at gbrain_remote_client.py:635. Trigger cannot fire.

### A20. `src/hyperloom/inference_optimizer/cli/backends.py:232` — _build_backends
- 类别: `sdk_graceful`  | stillReachable: **False** | 置信度: high
- deadReason: The sole production caller (cli/__init__.py:1872-1885) resolves critic_agent_root via _resolve_critic_agent_root() whenever _critic_agent_runtime_needed(critic_choice) (i.e. critic_choice=='agent') and sys.exit(2)s if it is None BEFORE calling _build_backends. Therefore when this function runs with critic_choice=='agent', critic_agent_root is always non-None, so the earlier branch at line 202 (provider_anthropic_only AND critic_agent_root is not None) always wins and this elif can never be entered from production. Only reached by the direct unit test test_build_backends_anthropic_only_degrades_to_claude_without_root.
- 验证者 proof: The fallback elif at backends.py:232 requires (critic_choice != 'mock' i.e. =='agent') AND provider_anthropic_only AND critic_agent_root is None. I independently confirmed this is unreachable from production:

1. Single production caller: cli/__init__.py:1916 (_build_backends). grep -rn shows the only non-test importer is cli/__init__.py; recover.py/setup.py/quantization.py do not build backends. _build_backends has no __getattr__ shim, console-script, or subprocess invocation.

2. critic_choice domain is exactly {"mock","agent"}: _resolve_critic_choice (cli/__init__.py:989) validates against _VALID_CRITIC_BACKENDS=("mock","agent") and sys.exit(2)s otherwise. So reaching the elif chain past line 200 (critic_choice != 'mock') means critic_choice=='agent'.

3. Guard makes critic_agent_root non-None for 'agent' regardless of provider: cli/__init__.py:1872 `if _critic_agent_runtime_needed(critic_choice):` — and _critic_agent_runtime_needed (cli/__init__.py:721, in credentials.py originally) returns simply `critic_choice == "agent"`, provider-independent (docstring + test_startup_robustness.py:143-159 confirm both anthropic-only and openai-only return True). Inside that block, line 1873 critic_agent_root=_resolve_critic_agent_root(); line 1874 `if critic_agent_root is None: sys.exit(2)`. Within _run_optimize, critic_agent_root is only ever assigned None at init (1864) or the resolver result (1873); no reassignment to None after the guard, and _build_backends is called exactly once (1916).

Therefore whenever _build_backends runs with critic_choice=='agent', critic_agent_root is non-None, so line 202 (`elif provider_anthropic_only and critic_agent_root is not None`) always wins over the elif at 232. No provider config (Anthropic-only, DeepSeek-only, OpenAI-only) bypasses root resolution. No third critic_choice value exists. The ValueError at 242 (`else: critic_choice=='agent' requires critic_agent_root`) is likewise dead for the same reason. The only reachability is the direct unit test test_build_backends_anthropic_only_degrades_to_claude_without_root (test_cli_backends_unit.py:113), which calls _build_backends directly bypassing the CLI guard. Auditor's dead-reason is correct.

### A21. `src/hyperloom/inference_optimizer/cli/backends.py:242` — _build_backends
- 类别: `default_when_missing`  | stillReachable: **False** | 置信度: high
- deadReason: Defensive guard; production caller (cli/__init__.py:1874-1884) already sys.exit(2)s when critic_agent_root is None for critic_choice=='agent', so this ValueError can never be raised in production. It is a re-raise/guard (not a degraded substitute) but included for completeness.
- 验证者 proof: The ValueError at backends.py:242 fires only when _build_backends receives critic_choice=='agent' with critic_agent_root=None in a NON-anthropic-only config (anthropic-only would hit the degrade branch at line 232, not raise). _build_backends has exactly ONE production caller: cli/__init__.py:1916 inside _run_optimize (confirmed via grep — no __getattr__ shim, no console-script entry, no subprocess/python -m invocation, no second/resume call site; the only other caller is the unit test at test_cli_backends_unit.py:45). Before that single call, cli/__init__.py:1872 runs `if _critic_agent_runtime_needed(critic_choice):` which resolves critic_agent_root and sys.exit(2)s if None (lines 1873-1884). _critic_agent_runtime_needed is hardcoded `return critic_choice == 'agent'` (line 731) with no env/config toggle and no monkeypatch in production — so it returns True for exactly the value that reaches line 242's else-branch, guaranteeing root is resolved-or-exit(2) first. Between line 1863 (resolve critic_choice) and 1916 (the call) there is no branch, no reassignment of critic_choice, and no early return that skips 1872 while reaching 1916 (verified by awk over that range). I could not construct any live production trigger. The guard is intentionally kept and unit-tested, but the test invokes _build_backends directly, bypassing CLI pre-validation — a test-only reach, not a production path.

### A22. `src/hyperloom/inference_optimizer/cli/backends.py:258` — _build_backends
- 类别: `default_when_missing`  | stillReachable: **False** | 置信度: high
- deadReason: Defensive guard; production caller (cli/__init__.py:1901-1913) resolves robustness_agent_root and sys.exit(2)s when None for robustness_choice=='agent' before the single _build_backends call. Never raised in production.
- 验证者 proof: The ValueError at src/hyperloom/inference_optimizer/cli/backends.py:258 (robustness_choice=='agent' AND robustness_agent_root is None) cannot fire via any production path. _build_backends is imported only by cli/__init__.py (line 41) and has exactly ONE production call site (cli/__init__.py:1916), reached as straight-line code inside main() (no dynamic dispatch, no __getattr__ shim, no action-registry string key, no `python -m` subprocess invocation of this function — grep for dynamic/name-based calls found none). Immediately before that call, lines 1898-1913 execute unconditionally: robustness_choice = _resolve_robustness_choice(args) at 1898; then `if robustness_choice == 'agent':` (1901) resolves robustness_agent_root = _resolve_robustness_agent_root() (1902) and, if it is None, prints an error and sys.exit(2) at 1913. robustness_choice is assigned exactly once (line 1898) and is NOT reassigned before the call (confirmed via grep: only assignment sites are 1898, plus the keyword pass-through at 1925 and a read at 2115). The exact same local robustness_agent_root value is passed to _build_backends at 1926, so there is no TOCTOU gap. Therefore whenever robustness_choice=='agent' reaches _build_backends, robustness_agent_root is guaranteed non-None, and the branch is dead in production. The only live trigger is the intentional unit test test_build_backends_robustness_agent_requires_root (test_cli_backends_unit.py:183), which is test-only, not a production trigger path. Other repo hits for `_build_backends` are unrelated test-local helper functions of the same name with different signatures.

### A23. `src/hyperloom/orchestrator/actions/executors/baseline.py:1181` — BaselineExecutor._run_once
- 类别: `legacy_key`  | stillReachable: **False** | 置信度: high
- deadReason: Commit 21a40b6f ('Prune orchestrator dead code paths') deleted the hyperloom.common.payload_aliases module and replaced read_extra_server_args(params) with a bare params.get('extra_server_args'). The legacy fallback branch no longer exists in the actions tree, and no non-test code in src/ writes 'extra_sglang_args' anymore, so the legacy key is never produced upstream. The GridVariant(extra_sglang_args=...) deprecation alias was deleted in the same commit.
- 验证者 proof: The legacy-key fallback at baseline.py:1181 is genuinely dead. Independent verification: (1) The site is now `effective_extra_server_args = str(params.get("extra_server_args") or "")` with NO reference to `extra_sglang_args`, `read_extra_server_args`, or `payload_aliases` anywhere in baseline.py (grep returned NONE FOUND). The legacy read branch is structurally removed, not merely unreachable — so even a `params` dict containing `extra_sglang_args` would be silently ignored. (2) Commit 21a40b6f confirmed present; `git show` verifies removal of `from hyperloom.common.payload_aliases import read_extra_server_args` and the `read_extra_server_args(params)` call, plus deletion of the `GridVariant(extra_sglang_args=...)` deprecation-alias kwarg in _grid_base.py (current GridVariant only has `extra_server_args`). (3) No live upstream producer of the legacy key: `grep -rn extra_sglang_args src/` outside tests yields ONLY the kernel-agent standalone `_payload_aliases.py` shim (LEGACY_KEY + docstring), which is never imported by the orchestrator (import search empty) and never feeds BaselineExecutor. All other ~40 non-test extra_server_args references use the canonical key. (4) The only surviving reader with legacy fallback, `_payload_aliases.read_extra_server_args`, is imported solely in kernel_optimization.py (a kernel-agent subprocess), out of scope for baseline. (5) A static guard test (test_no_legacy_writer_sites.py) actively enforces the key appears nowhere outside a 3-file kernel-agent allowlist. No caller, action-registry key, subprocess invocation, CLI/config flag, or resume/degradation path can route the legacy key into a live read at this site.

### A24. `src/hyperloom/orchestrator/actions/executors/profile.py:786` — ProfileExecutor (profile server-args merge)
- 类别: `legacy_key`  | stillReachable: **False** | 置信度: high
- deadReason: Same commit 21a40b6f removed the payload_aliases import, replaced read_extra_server_args(params) with params.get('extra_server_args'), and deleted the 'params.pop("extra_sglang_args", None)' cleanup line. No upstream in src/ writes 'extra_sglang_args', so neither the fallback read nor the pop can ever have an effect.
- 验证者 proof: The ProfileExecutor fallback at src/hyperloom/orchestrator/actions/executors/profile.py:786 is dead. (1) Commit 21a40b6f removed `from hyperloom.common.payload_aliases import read_extra_server_args`, replaced `read_extra_server_args(params)` with `str(params.get("extra_server_args") or "")`, and deleted `params.pop("extra_sglang_args", None)`. The current line 786 reads only the canonical key. (2) The source module the fallback depended on, hyperloom.common.payload_aliases, was itself deleted in commit 0f601f2c — it no longer exists (`ls` fails). (3) A whole-repo grep for `extra_sglang_args` (all .py + config formats) finds it in only 3 read-only/test files: the kernel-agent standalone shim agents/kernel/tools/_payload_aliases.py, its test test_payload_aliases_shim.py, and the static-guard test test_no_legacy_writer_sites.py. NO src/ code writes the legacy key into any Intent.payload / Task.params. (4) A dedicated CI guard (inference_optimizer/tests/test_no_legacy_writer_sites.py) actively fails the build if the literal appears outside a 3-file allowlist, preventing any future writer. (5) The surviving read_extra_server_args lives only in the kernel-agent standalone shim, imported by bare module name inside remote-node kernel subprocess tools (kernel_optimization.py) — a different symbol in a different process; the orchestrator ProfileExecutor's import list contains no reference to it. (6) No dynamic/indirect key construction (string concat, getattr, config/env/CLI) produces `extra_sglang_args` in the orchestrator. With no writer and no reader of the legacy key in the profile path, neither the removed fallback read nor the removed pop can ever fire.

### A25. `src/hyperloom/orchestrator/knowledge/recipe_kb/canonical_id.py:111` — cid_to_path_components
- 类别: `legacy_key`  | stillReachable: **False** | 置信度: high
- deadReason: The advertised legacy-6-segment padding FALLBACK does not exist in the code: line 111 raises InvalidCanonicalIdError for any segment count != 8 (1 prefix + 7 dims). The docstring is stale — the padding path was removed when the id moved to a strict 7-tuple. There is no dead branch executing; rather the fallback the docstring describes has been deleted, so no 6-segment id can be accepted.
- 验证者 proof: The "legacy-6-segment padding fallback" described in the docstring (canonical_id.py:86-88) does not exist in code and cannot fire. cid_to_path_components has a single definition (canonical_id.py:79) with no aliases/__getattr__ shims overriding it (the only __getattr__ in the package, gbrain_ingest.py:432, is an unrelated stub-class accessor). Line 111 `if len(parts) != 1 + CANONICAL_ID_DIMENSIONS: raise` strictly rejects any count != 8; there is no len==6 branch, no default-slug padding, and the unpack at line 124 assumes exactly 7 dims. Both live callers unpack a 7-tuple and add no padding: dispatcher.py:103 propagates the exception, gbrain_remote_client.py:917 catches it and returns None (a remote miss, not padding). Test test_cid_to_path_components_rejects_legacy_6_segment (test_local_recipe_store.py:102-105) asserts `inference:model:hw:fw:ver:prec` RAISES InvalidCanonicalIdError, locking in rejection. The only id producer, recipe_canonical_id (recipe_snapshot_constants.py:120,138), always emits the 8-segment form, so no live source generates a 6-segment id to feed the supposed fallback. Git history (commit 7b0cd37/cbd35fa, which introduced the docstring wording) shows the file shipped with the strict `!= 1 + CANONICAL_ID_DIMENSIONS: raise` guard alongside the padding docstring — the padding path never existed here; the docstring is aspirational/stale. No caller, action-registry key, subprocess/console-script entry, config/env/CLI flag, or resume/degradation path re-creates the padding behavior.

### A26. `src/hyperloom/orchestrator/knowledge/recipe_kb/dispatcher.py:200` — _v2_to_arbor
- 类别: `legacy_key`  | stillReachable: **False** | 置信度: high
- deadReason: Current writers persist identity under `framework_name` exclusively (schema.to_dict line 349, local_store.py:539, gbrain_ingest recipe page attrs line 275, gbrain client label C.F_LABEL_FRAMEWORK_NAME). The legacy `framework` label can only appear on gbrain pages/rows authored before the framework_name rename; no live code path emits it. Reachable only for legacy external/mirror corpus data, not current-code output.
- 验证者 proof: The fallback (dispatcher.py:200) fires only when a row reaching _v2_to_arbor has labels with key 'framework' but not 'framework_name'. _v2_to_arbor has one caller, _normalize_remote_row (dispatcher.py:422), fed only by self.remote.get_recipe/search (dispatcher.py:720/743/767/854). In production self.remote is always GbrainRemoteRecipeClient (kb.py:124-128; remote_client.py confirms gbrain is the sole read backend; all other configs pass remote=None so no v2 rows flow). Every gbrain read returns rows from _get_page_recipe -> _page_to_recipe (gbrain_remote_client.py:471-554), which ALWAYS rebuilds labels via C.canonical_labels(...). canonical_labels (recipe_snapshot_constants.py:173-181) emits a fixed 7-key dict keyed on F_LABEL_FRAMEWORK_NAME='framework_name' and can never emit a bare 'framework' key. Decisively, _page_to_recipe already promotes any legacy page's attrs.get('framework') into framework_name at line 498 BEFORE building labels, so even a pre-rename gbrain page arrives at _v2_to_arbor with labels.framework_name set. The 'framework' keys in cortex_t0.py are label_match QUERY dicts, not stored/returned labels, and _labels_match (line 610) silently skips a 'framework' query key since recipe_labels only carries framework_name. The inline mirror (GbrainMirroringRecipeKB) wraps only the write path; ingest (gbrain_ingest.py:271) also normalizes framework->framework_name on write. The only thing exercising the fallback is the unit test test_v2_to_arbor_reads_legacy_framework_label, which hand-injects labels={'framework':...} directly into _v2_to_arbor and does not represent any live producer. No current-code path emits a recipe labels dict with a bare 'framework' key that would bypass _page_to_recipe's own promotion.

### A27. `src/hyperloom/orchestrator/knowledge/recipe_kb/dispatcher.py:433` — RecipeKB._remote_label
- 类别: `default_when_missing`  | stillReachable: **False** | 置信度: high
- deadReason: The only remote client that can be wired is GbrainRemoteRecipeClient (kb.py:124 build_gbrain_remote_from_env; Cortex backend removed in PR #757), and it IS in _REMOTE_LABELS -> 'gbrain'. The dict.get default (class name) branch only fires for a hypothetical future backend that is never constructed today.
- 验证者 proof: The .get() default branch at dispatcher.py:433-435 fires only when type(self.remote).__name__ is not in _REMOTE_LABELS (which contains exactly {'GbrainRemoteRecipeClient': 'gbrain'}). RecipeKB is a plain @dataclass (dispatcher.py:335-363) with remote: Any = None; remote is set only at construction — no dynamic `.remote =` assignment exists in non-test code. The only non-test construction sites are in inference_optimizer/cli/kb.py: line 118 (remote=None), line 126 (remote=None), and line 128 (remote=gbrain_remote). gbrain_remote comes solely from build_gbrain_remote_from_env() (gbrain_remote_client.py:1056), which returns either a GbrainRemoteRecipeClient (line 1077) or None — never any other type. The GbrainMirroringRecipeKB wrapper (gbrain_ingest.py:390) delegates via __getattr__ to the inner RecipeKB, so _remote_label runs against the real RecipeKB whose remote is still a GbrainRemoteRecipeClient — no new class name. No other remote client class exists in src (grep for Cortex/other RecipeClient classes and for GbrainRemoteRecipeClient subclasses in non-test code returned nothing); KGClient (kg_client.py) is an mcp-based knowledge-graph client, never placed in the remote slot. When remote is None the method returns 'none' at line 431 before reaching .get(); when remote is set it is always GbrainRemoteRecipeClient, whose exact class name (line 631) matches the dict key (line 79) -> 'gbrain'. The default branch is unreachable in current code.

### A28. `src/hyperloom/orchestrator/knowledge/recipe_kb/gbrain_ingest.py:271` — recipe_to_page
- 类别: `legacy_key`  | stillReachable: **False** | 置信度: high
- deadReason: Mirror reads the local recipe payload, which current writers stamp with framework_name only. Legacy `framework` present only on pre-rename rows.
- 验证者 proof: recipe_to_page (gbrain_ingest.py:271) has exactly two live entry paths, neither of which can present a payload with a top-level 'framework' but no 'framework_name':

(1) INLINE MIRROR (RECIPE_KB_MIRROR_MODE=inline, wired at cli/kb.py:134-141): GbrainMirroringRecipeKB.put_recipe(**kwargs) -> mirror_recipe(kwargs) -> recipe_to_page(kwargs). The kwargs are the exact put_recipe argv built by the only two live callers: proposals.py:255 ("framework_name": framework) and cortex_t0.py:1056 (framework_name=_framework or ""). Both ALWAYS set top-level framework_name. The only bare 'framework' key produced anywhere in live code is _collect_workload_tags (result_recorder.py:714, out["framework"]=framework), but it is (a) nested inside overrides["extras"] -> merged_extras -> put_kwargs["extras"] (proposals.py:241,285), never at the top level recipe_to_page reads, and (b) only added when `if framework:` is truthy (line 713) — which is exactly when framework_name is also truthy. So recipe.get("framework_name") is always non-empty and the `or recipe.get("framework")` branch short-circuits before evaluating. Recipe.to_dict (schema.py:349) also unconditionally emits framework_name, so no current writer can persist a framework_name-less row even after extras-splat.

(2) EXTERNAL BULK INGEST (default mode): the module docstring still references an out-of-band CronJob bulk ingest that reads raw on-disk recipe.json WITHOUT Recipe.from_dict normalization — that was the only path that could feed a genuine pre-rename legacy row (bare 'framework', no 'framework_name') into recipe_to_page. But its driver (def ingest_local_to_gbrain, def main, if __name__=="__main__", argparse/sys imports) was DELETED in commit 21a40b6f "Prune orchestrator dead code paths". `grep -rn ingest_local_to_gbrain` returns nothing (exit 1); there is no [project.scripts] console entry, no __main__, and no remaining reference anywhere in the repo.

No __getattr__ shim, env/CLI flag, subprocess agent invocation, mock/real role pairing, or resume/SDK-degradation path reconstructs the deleted bulk reader or flattens extras["framework"] to top level before the mirror runs. schema.py:422/452 and local_store.py:878 still support READING legacy 'framework' via Recipe.from_dict / label match, but those normalize to framework_name and never route back into recipe_to_page's raw top-level read.

### A29. `src/hyperloom/orchestrator/knowledge/recipe_kb/local_store.py:878` — _labels_match_payload
- 类别: `legacy_key`  | stillReachable: **False** | 置信度: high
- deadReason: Same as schema finding: current writes stamp framework_name; only pre-rename on-disk rows carry bare `framework`. No live writer produces it.
- 验证者 proof: The fallback at local_store.py:878 (`payload.get("framework_name") or payload.get("framework")`) lives inside the `elif key == "framework_name":` branch of `_matches_labels` (line 831; the claim's symbol name `_labels_match_payload` is inaccurate — the real name is `_matches_labels`). For that branch to execute, a production caller must pass a `label_match` dict whose KEY is `"framework_name"` into `_matches_labels`. Full call-graph trace shows no such caller: (1) `_matches_labels` is invoked only by `LocalRecipeStore.search` (local_store.py:714), which reads raw JSON via `_read_json` without `Recipe.from_dict` normalization (auditor's mechanic is correct). (2) `LocalRecipeStore.search` has exactly one production caller: `RecipeKB.search` (dispatcher.py:885), passing label_match through unchanged. (3) `RecipeKB.search` has exactly three production callers, all in cortex_t0.py (lines 434, 1123, 1146); all three build label_match with the LEGACY key `"framework"` (cortex_t0.py:412, 424, 1115, 1139), which routes to the generic `else` branch (line 881-884), NOT the framework_name branch. (4) The only production dict keyed `"framework_name"`, `_labels_from_canonical_id` (dispatcher.py:83), feeds only `self.remote.search` (dispatcher.py:743, 767) inside get_recipe — never `self.local.search`; get_recipe's local fall-through uses `local.get_recipe(canonical_id=...)`, a direct path read that never calls `_matches_labels`. (5) The gbrain remote client's framework_name labels (gbrain_remote_client.py:924, 953) route to its OWN separate `_labels_match` (line 557), not local_store's. No __getattr__ shim, entry point, subprocess agent, CLI/env flag, or resume/degradation path injects a framework_name-keyed local search. The branch is exercised only by the unit test test_local_recipe_store.py:53 calling `_matches_labels` directly. Even the auditor's stated dead-reason understates it: the branch is unreachable in production not merely because no writer emits bare `framework` on disk, but because no production caller ever passes the `framework_name` KEY into local search at all.

### A30. `src/hyperloom/orchestrator/loop/intent_router.py:292` — IntentRouter (specialist verdict recording)
- 类别: `default_when_missing`  | stillReachable: **False** | 置信度: high
- deadReason: PendingProposal (coordinator.py:446) is a dataclass with fields proposal_msg_id/from_agent/action_name/predicted_gain_pct/payload/decided/verdict — it has NO task_id field, so `getattr(pending, "task_id", None)` always returns None. The code comment itself states 'PendingProposal has no task_id field'. Only the `pa_params.get("task_id")` fallback is ever effective; the getattr primary is dead.
- 验证者 proof: The primary term `getattr(pending, "task_id", None)` at intent_router.py:292 always evaluates to None at runtime, so it never contributes a truthy value; only the `pa_params.get("task_id")` fallback is ever effective. Verified independently:

1. `pending` in `_handle_single_verdict` is always a `PendingProposal`. Its sole caller (`_handle_review_verdict`, intent_router.py:221-227) passes `pending = self.state.pending_proposals.get(target)` (line 195), and that dict only ever holds `PendingProposal` instances.

2. `PendingProposal` (coordinator.py:445-455) is a plain `@dataclass` with exactly 7 fields: proposal_msg_id, from_agent, action_name, predicted_gain_pct, payload, decided, verdict. Confirmed by executing dataclasses.fields(): no `task_id`. It has NO `__getattr__` shim (verified `hasattr(type, '__getattr__') is False`).

3. Every construction site was audited: intent_router.py:176, resume.py:91, explore.py:1500, explore.py:1617, framework.py:3085, plus all test fixtures — none passes `task_id`. The only branch that reaches line 292 is `action_name == "specialist"`; those PendingProposals carry task_id inside `payload["params"]` (proposals.py records it into params), which is exactly what the `pa_params.get("task_id")` fallback reads — never as an attribute.

4. Grepped the whole repo for `setattr(pending`, `pending.task_id =`, and `.task_id =` on pending-typed objects: none exist. Every `.task_id`/`task_id=` occurrence is on `Task` objects (result_recorder.py, task_registry Task, test stubs), not PendingProposal. No __slots__ that would matter — dynamic assignment is technically possible but no code performs it.

Since PendingProposal never has a task_id attribute in any production or test path, the getattr primary is always None and the fallback (`pa_params.get`) is the only live branch. The auditor's dead-reason is correct.

### A31. `src/hyperloom/orchestrator/loop/result_recorder.py:843` — ResultRecorder._build_kernel_optimizations_from_state
- 类别: `legacy_key`  | stillReachable: **False** | 置信度: high
- deadReason: Current code always writes the source path under `last_source_file` (orchestrator/kernel/_kernel_decisions.py:541 `entry["last_source_file"] = source_file`) and never writes a bare `source_file` key into kernel_opt_attempts entries. The `e.get("source_file")` arm can only fire on entries persisted before the `last_source_file` rename, i.e. resuming a stale pre-rename session.
- 验证者 proof: The `e.get("source_file")` arm at result_recorder.py:843 reads a bare `source_file` key on entries of `state.kernel_opt_attempts`. I traced every writer of that map: the ONLY writer is _kernel_decisions.py:639 `state.kernel_opt_attempts[kernel_id] = entry`, where `entry = dict(state.kernel_opt_attempts.get(kernel_id) or {})` (line 512) and the source is stored exclusively as `entry["last_source_file"] = source_file` (line 541). No `entry["source_file"] = ...` assignment exists anywhere in the tree (`grep '\["source_file"\]\s*=' orchestrator/` → zero hits on any attempts entry).

Git history refutes the auditor's escape hatch (resuming a "pre-rename" session). `git log --all -G 'entry\["source_file"\]\s*='` returns NOTHING across all history, while `entry["last_source_file"] = source_file` traces back to commit 52b829ad. The diff of 52b829ad shows `last_source_file` was ADDED fresh into the attempts entry (the bare `"source_file": source_file` in that same diff goes into the separate `last_kernel_opt` dict, not the attempts entry). So there was NO rename: `last_source_file` is the original and only name that field ever had in a kernel_opt_attempts entry. No historical Hyperloom version ever persisted a bare `source_file` key there, so even a stale resumed session cannot carry one.

The other `source_file` writes are all different dicts: _accuracy_gate.py (quality-gate result), attempt_summary.py / shared_state.py:2259 (kernel roofline summary), request_handlers.py (candidate/integrate-payload/last_kernel_opt), recipe_kb schema. None mutate a kernel_opt_attempts entry.

State load (SharedState.load_or_init → from_dict) treats kernel_opt_attempts as a plain `dict[str, Any]` field; migration drops unknown TOP-LEVEL keys and never creates or normalizes per-entry keys, so it can neither synthesize nor is needed to strip a bare source_file. All test/tool inline `kernel_opt_attempts = {...}` literals with `source_file` are on candidate/payload dicts, and test_profile_and_kernel_handlers.py:3911 explicitly asserts the fallback is to `last_source_file`. The `e.get("source_file")` arm is unreachable in current code and in every prior version.

### A32. `src/hyperloom/orchestrator/loop/sub_agent_runner.py:10` — SubAgentRunner.run_task
- 类别: `sdk_graceful`  | stillReachable: **False** | 置信度: high
- deadReason: The documented backend.run() fallback is not implemented: run_task looks up executor_registry.get(task.kind) (line 205) and on None returns SubAgentResult(state='failed', error='no runner registered') (lines 206-220). There is no backend.run() call anywhere in sub_agent_runner.py. The fallback described in the docstring has been removed from the code; unregistered kinds now hard-fail instead of degrading to the LLM.
- 验证者 proof: The docstring at sub_agent_runner.py:10 documents an "LLM external sub-agent fallback (backend.run())" as the alternate dispatch path when a kind has no deterministic executor. This fallback is NOT implemented in current code. run_task (lines 175-274) does exactly one registry lookup: runner = self.executor_registry.get(task.kind) at line 205 (the ONLY such lookup in the whole repo, and .get() has no default arg so unknown kinds -> None). On None it hard-fails: transitions to 'failed' with evidence reason='no_executor' and returns SubAgentResult(state='failed', error="no runner registered for kind=...") at lines 206-220. There is no backend.run() call, no `backend` attribute, and no `.run(` anywhere in sub_agent_runner.py except the docstring string on line 10 (verified by grep). No runtime wrapper/monkeypatch injects a backend fallback into run_task. The three real backend.run() sites (maintenance.py:249 checkpoint summary, coordinator.py:1971 orchestration turn, specialists/runner.py:835) are all reached via other call chains, never via run_task's None branch. The LLM path that does exist is reached as a REGISTERED executor: cli/executors.py:260 register_executor("specialist", specialist_executor) -> _build_specialist_executor._executor adapter -> SpecialistRunner.run(ctx) -> backend.run (specialists/runner.py:835). That fires only when task.kind == "specialist" AND that executor was registered — i.e. it is a normal registered executor, not a fallback for an unregistered kind. Git history confirms the path was never real: legacy commit 356b68bd ("remove legacy") shows the old sub_agent_runner.py carried the same stale docstring ("OOB sub-agent (LLM) — fallback path: spawn a fresh backend.run()") while its run_task already returned no_executor/"no runner registered" without ever calling backend.run(). Line 10 is a condensed restatement of that stale docstring. Unregistered kinds hard-fail; they never degrade to an LLM backend.run().

### A33. `src/hyperloom/orchestrator/policy/gate.py:530` — _resolved_within
- 类别: `try_except`  | stillReachable: **False** | 置信度: high
- deadReason: pyproject.toml sets requires-python = ">=3.10"; Path.is_relative_to was added in Python 3.9, so it always exists and the AttributeError branch can never fire. The author already marked it '# pragma: no cover — Python <3.9'.
- 验证者 proof: The except AttributeError branch at gate.py:530-535 fires only when v.is_relative_to(r) raises AttributeError, i.e. when Path lacks is_relative_to (Python <3.9). But: (1) pyproject.toml:10 sets requires-python = ">=3.10" and pyproject.toml:451 sets ruff target-version = "py310"; no lower-version build/install target exists. (2) v = Path(str(value)).resolve() where Path is the stdlib pathlib.Path (gate.py:11 `from pathlib import Path`), no pathlib2 backport, no shadowing, no monkeypatch. resolve() always returns a concrete PosixPath which has is_relative_to on 3.9+ (verified on the runtime Python 3.12: hasattr == True). (3) v cannot become a non-Path before line 529; and passing a Path arg to is_relative_to never raises AttributeError in 3.10+ (a bad arg raises TypeError/ValueError, uncaught here). (4) All callers (_path_in_source_allowlist gate.py:2361, _path_in_trace_allowlist gate.py:2373) pass ordinary path strings. No test deletes/monkeypatches is_relative_to to force the branch. Author already marked it '# pragma: no cover — Python <3.9'. The identical sibling guard at gate.py:2342-2348 is dead for the same reason.

### A34. `src/hyperloom/orchestrator/policy/gate.py:2343` — PolicyGate._resolves_within_session (session-dir path check)
- 类别: `try_except`  | stillReachable: **False** | 置信度: high
- deadReason: Identical to gate.py:530. requires-python >=3.10 guarantees Path.is_relative_to exists, so the AttributeError branch is dead. Marked '# pragma: no cover — Python <3.9' by the author.
- 验证者 proof: The `except AttributeError` at gate.py:2343 in PolicyGate._path_under_session only fires when `v.is_relative_to` does not exist as an attribute. `v` is always produced by `v = Path(str(value)).resolve()` (line 2338), where `Path` is the unmodified stdlib import `from pathlib import Path` (gate.py:11) — no shim, no monkeypatch, no __getattr__ interception anywhere in the repo. `Path(...).resolve()` always returns a concrete pathlib.Path (verified: type is pathlib.PosixPath), and `Path.is_relative_to` was added in Python 3.9. pyproject.toml pins `requires-python = ">=3.10"`, and the running interpreter is 3.12.13, where I confirmed `hasattr(Path('/tmp').resolve(),'is_relative_to')` is True. The only two callers (gate.py:2426 and 2439) pass plain string `node` values that flow through the same Path.resolve() coercion. No test monkeypatches or delattrs is_relative_to (searched src/hyperloom/); existing tests only exercise the happy paths. The branch is structurally identical to gate.py:530, both marked `# pragma: no cover — Python <3.9`. Note the claim's symbol name (_resolves_within_session) is slightly off — the real symbol is _path_under_session — but the code, trigger, and analysis match. No live trigger path exists on any supported interpreter.

### A35. `src/hyperloom/orchestrator/prompts/prompt_builder.py:265` — _filter_actions
- 类别: `default_when_missing`  | stillReachable: **False** | 置信度: high
- deadReason: The comment states the caller has already validated enabled action names; enabled_actions come from default_enabled_actions() filtering the closed FULL_ENABLED_ACTIONS set, all of which are registered. The skip only fires for an unregistered name that never reaches here in normal flow
- 验证者 proof: The branch at prompt_builder.py:267 (registry.get(name) returns None) has exactly one production reach path: _filter_actions <- _resolve_prompt_prelude <- build_orchestration_prompt <- _build_orchestration_prompt (cli/__init__.py:197) <- single call site cli/__init__.py:2050. That caller ALWAYS sets enabled_actions = default_enabled_actions(no_kernel=..., no_explore=...) (cli/__init__.py:223). default_enabled_actions() only FILTERS (subtracts from) the hardcoded closed tuple FULL_ENABLED_ACTIONS (action_surfaces.py:71); flags can only remove names, never add. No env/config/CLI/manifest/subprocess/console-script/mock-role/resume path injects arbitrary action-name strings into enabled_actions (grep confirmed the only non-def references are the single cli call site and the builder signature). Every one of the 14 FULL_ENABLED_ACTIONS names has a shipped actions/_meta/<name>.yaml (verified on disk: all 14 present; empirically registry.get(name) is non-None for all 4 flag combinations). Package-data ships actions/_meta/*.yaml as a glob (pyproject.toml:123), so no partial-wheel gap. ActionRegistry.load()/from_yaml_dict is strict: a missing/malformed/name-mismatched yaml RAISES ActionRegistryError (registry.py:190,296) rather than silently omitting an action, so a successful load guarantees all FULL actions are cached, and a broken one aborts before any prompt is built. The only code that reaches the None branch is the unit test test_unknown_enabled_action_is_silently_skipped (test_prompt_builder.py:240), which hand-crafts 'no_such_action' — not a live flow. I found no plausible live trigger.

### A36. `src/hyperloom/orchestrator/roles/claude.py:442` — ClaudeBackend.run
- 类别: `resume_downgrade`  | stillReachable: **False** | 置信度: high
- deadReason: The `_resume_downgraded` field (claude.py:211) is `field(default=False, init=False)` and NOTHING in the codebase ever assigns it True (grep for `_resume_downgraded =` returns only the field decl and the `= False` default). The code comment at line 210-211 states 'Always False now (resume is supported by the pinned SDK floor); retained as a stable llm_calls.jsonl key'. The pin `claude-agent-sdk>=0.2.110` (pyproject.toml:13; installed 0.2.116) makes `resume=` unconditionally supported, so the downgrade branch that would flip it True was deleted. The metadata key is emitted but the value is a constant.
- 验证者 proof: The fallback (resume-downgrade → set resume_downgraded=True) is dead. Independent verification: (1) Repo-wide grep for `_resume_downgraded`/`resume_downgraded` finds NO production assignment to True anywhere — only the field decl `_resume_downgraded: bool = field(default=False, init=False)` (claude.py:211) and the read `"resume_downgraded": self._resume_downgraded` (claude.py:442). No setattr/__dict__/subclass path. (2) `git show 13fba44a` (the commit that introduced the field) is the same commit that DELETED the trigger: the old `_instantiate_options` had `except TypeError as exc:` → warn "SDK ClaudeAgentOptions rejected resume=; falling back to stateless turn" and set the flag; that block was removed. Current `_instantiate_options` (claude.py:606-619) is a bare `return self.sdk_options_cls(**kwargs)` with NO try/except, docstring stating no compatibility fallback is needed. (3) The trigger exception is now uncatchable as a downgrade: nothing in `run`/`_build_options`/`_instantiate_options` catches a `resume=` kwarg rejection — a TypeError would propagate unhandled, never flipping the flag. `_instantiate_options` is not overridden (no ClaudeBackend subclass) or monkeypatched in production. (4) Pin `claude-agent-sdk>=0.2.110` (installed 0.2.116) makes `resume=` unconditionally supported, so the original condition is unsatisfiable regardless. The only True value in the tree is test_llm_trace_unit.py:27, a hand-built metadata dict fed straight to the llm_trace serializer to test key round-tripping; it never instantiates ClaudeBackend or runs the fallback path. The metadata key is emitted as a constant False.

### A37. `src/hyperloom/orchestrator/roles/claude.py:606` — ClaudeBackend._instantiate_options
- 类别: `sdk_graceful`  | stillReachable: **False** | 置信度: high
- deadReason: The method body is now just `return self.sdk_options_cls(**kwargs)` with no try/except. Its own docstring (lines 608-611) states: 'The resume / effort / thinking kwargs are all supported by the pinned claude-agent-sdk >= 0.2.110 floor, so no compatibility fallback is needed.' `_apply_effort_options` docstring (line 596) still references 'Unknown SDK builds that reject these kwargs degrade via _instantiate_options' but that degrade code has been removed — the comment is stale. Given the >=0.2.110 pin, a build that rejects these kwargs cannot be installed.
- 验证者 proof: The old fallback (except TypeError -> pop 'resume', warn, retry) in ClaudeBackend._instantiate_options was removed in commit 13fba44a; current body (claude.py:606-619) is a single `return self.sdk_options_cls(**kwargs)` with no try/except, so TypeError propagates. Trigger requires the SDK constructor to reject resume/effort/thinking. That cannot happen in production: (1) pyproject pins claude-agent-sdk>=0.2.110 (two sites) and pip cannot resolve a below-floor build; (2) installed build is 0.2.116 and I verified via inspect.signature AND by actually calling ClaudeAgentOptions(max_turns=4, resume='sess', effort='medium', thinking={'type':'adaptive'}) that all three kwargs construct without TypeError; (3) all 5 production ClaudeBackend instantiations (cli/executors.py:184, cli/backends.py:236/273/286/296) pass only model/max_turns/conversational and never inject sdk_options_cls, so at runtime sdk_options_cls is always resolved via _import_sdk() to the real pinned ClaudeAgentOptions. The only custom sdk_options_cls injections are in /tests/ (e.g. test_instantiate_options_typeerror_propagates, which asserts the NEW propagation behavior). Corroborating dead-ness: _resume_downgraded (line 211) is read into turn metadata (line 442) but is never assigned True anywhere — its only write site was inside the removed except block. The two surviving TypeError-retry blocks (quantization/driver/runner.py:319-338 and kernel/tools/tracelens_skill_runner.py:945-950) are unrelated: they degrade on cwd/env kwargs in different functions, not on resume/effort/thinking in ClaudeBackend. The only remaining trace is a stale docstring at line 595 ('degrade via _instantiate_options') — a comment, not executable code.

### A38. `src/hyperloom/orchestrator/specialists/runner.py:375` — SpecialistRunner._resolve_tools
- 类别: `sdk_graceful`  | stillReachable: **False** | 置信度: high
- deadReason: In production the knowledge_plane is a KnowledgePlane which defines both pr_monitor_enabled (knowledge_plane.py:80) and cortex_enabled (:89) as real properties that never raise; the AttributeError branch only fires for a duck-typed/mock plane lacking the attr
- 验证者 proof: The AttributeError fallback at runner.py:375-388 fires only if plane.pr_monitor_enabled / plane.cortex_enabled raises AttributeError. I independently traced every production path and cannot make that happen:

1. ONLY production construction of SpecialistRunner is in cli/executors.py:165 and :189 (inside _build_specialist_executor), both passing knowledge_plane=knowledge_plane. That variable is set in exactly two places (cli/__init__.py:1611 and :1790) to `None if not cortex_enabled else _bootstrap_knowledge_plane(...)`. No other assignment exists (grep across whole repo, tests excluded).

2. When knowledge_plane is None (--degraded-kb / cortex_enabled false), the `if plane is not None:` guard at runner.py:375 skips the entire try/except — branch never entered.

3. When it is a KnowledgePlane, both attributes are real @property (knowledge_plane.py:80 pr_monitor_enabled, :89 cortex_enabled). pr_monitor_enabled reads `self.pr_monitor is not None and self.pr_monitor.enabled`; pr_monitor is a dataclass field (None or PRMonitorClient), and PRMonitorClient.enabled is a plain class attribute always present (pr_monitor.py:32). cortex_enabled reads dataclass field self.cortex_kb_mcp_url (default ""). Neither property can raise AttributeError.

4. KnowledgePlane has no subclass, no __getattr__/__slots__/__getattribute__ magic, no alternate factory that omits these fields. _bootstrap_knowledge_plane always returns a fully-initialized KnowledgePlane.

5. Coordinator stores self.knowledge_plane (Any) but does NOT build the runner from it; the runner is built in the CLI and injected as specialist_executor, so no divergent duck-typed object reaches _resolve_tools.

6. No console-script/entry-point or `python -m` subprocess path constructs SpecialistRunner with a duck-typed plane; subprocess specialists get MCP config, not this object.

The only inputs that would trigger the branch are duck-typed/mock planes lacking the attribute, which exist solely in tests (test_specialist_integration.py even defines them as class attrs, so they don't raise either). The auditor's dead-reason is correct and confirmed by independent search.

### A39. `src/hyperloom/orchestrator/state/shared_state.py:1902` — SharedState.record_action_attempt (audit)
- 类别: `default_when_missing`  | stillReachable: **False** | 置信度: high
- deadReason: All five actions in _AUDIT_ACTIONS (baseline, profile, sweep, explore, roofline) have both dataclass fields declared with field(default_factory=...) on SharedState (baseline_attempts/last_baseline, profile_attempts/last_profile, sweep_attempts/last_sweep, explore_attempts/last_explore, roofline_attempts/last_roofline). Dataclass default_factory fields are always present on every instance, so hasattr is always True and the `return None` branch is dead. Only reachable if a future action is added to _AUDIT_ACTIONS without its fields.
- 验证者 proof: The line-1902 branch `if not hasattr(self, attempts_attr) or not hasattr(self, last_attr): return None` cannot fire in current code. All five actions in _AUDIT_ACTIONS (baseline/profile/sweep/explore/roofline) have BOTH backing dataclass fields declared as `field(default_factory=...)` with init=True (shared_state.py:665/686/687/691/693/694/695/697/699/701). SharedState (shared_state.py:352) is a plain @dataclass with NO __getattr__, NO __slots__, NO custom __init__/__post_init__, NO pickle/__setstate__ overrides, and its mixins (_RenderMixin, _ExploreStateMixin) add no __init__. The single deserialization path from_dict → `cls(**filtered)` (line 1062) filters raw to known fields and relies on the generated __init__, so any field absent from an old/ancient state.json still receives its default_factory value. Verified empirically: `SharedState()` and `SharedState.from_dict({})` (empty ancient state) both yield hasattr==True for all ten attributes (both attempts_attr and last_attr for every audited action). No delattr on these fields anywhere, no runtime mutation of _AUDIT_ACTIONS (the identically-named symbol in breakdown/collectors/timeline.py:35 is an unrelated local tuple), no object.__new__(SharedState), and no subclass of SharedState (SharedStateSnapshot in robustness is unrelated). Callers in writeback.py only pass audited action names; a non-audited name would hit the DIFFERENT return at line 1899, not 1902. The 1902 branch is only reachable if a future action is added to _AUDIT_ACTIONS without its two backing fields — confirming the original auditor's dead-reason.


## 附录 B — 14 处存疑 / 仍可达(完整反证)


### B1. `src/hyperloom/agents/kernel/tools/tracelens_skill_runner.py:939`
- stillReachable: **True** | 置信度: high
- 原审计 deadReason: The 'primary' branch (returning a mutated module-level DEFAULT_MODEL global) is dead: DEFAULT_MODEL is never defined nor imported anywhere in tracelens_skill_runner.py (only DEFAULT_CODEX_MODEL is imported at line 27). globals().get('DEFAULT_MODEL', ...) therefore ALWAYS returns the hardcoded fallback literal 'claude-opus-4-8' when reached. The guard comment describes protecting a global that does not exist.
- 验证者反证 proof: Line 939 (`resolved_model = resolved_model or str(globals().get("DEFAULT_MODEL", "claude-opus-4-8")).strip()`) executes on the LIVE non-OpenAI SDK path: run_tracelens_skill reaches it whenever _should_use_openai_tool_runner() is False (line 898), which is the default production Claude-SDK path. When `model` is unset/empty, `resolved_model` is empty at 897, so the right-hand side is evaluated and the literal "claude-opus-4-8" is returned and assigned into kwargs["model"] (line 940), used to build real SDK options. So the fallback LITERAL branch is reachable and exercised — confirmed by the dedicated passing test test_run_tracelens_skill_uses_fallback_model_when_default_missing (test_tracelens_agent_transcript.py:138-170), which asserts seen_model == "claude-opus-4-8". I independently confirmed the module never binds DEFAULT_MODEL: no `DEFAULT_MODEL =` assignment (grep + git log -S across all history), no import (only DEFAULT_CODEX_MODEL at line 27), no star-import, no __getattr__, no setattr(tlr, "DEFAULT_MODEL") anywhere, and runtime hasattr(tlr,'DEFAULT_MODEL') is False. Thus the ONLY thing that is dead is the alternate sub-branch (retrieving a populated module global) — the auditor is correct that globals().get can never find a real DEFAULT_MODEL and the guard comment protects a nonexistent global. But the auditor labeled the finding as the literal-yielding fallback, and that literal fallback DOES fire in current code, so the trigger it describes ("always yields the literal") is reachable, not dead.

### B2. `src/hyperloom/agents/quantization/driver/result_collector.py:270`
- stillReachable: **True** | 置信度: medium
- 原审计 deadReason: quantize_via_prompt creates workspace via mkdir(parents=True) at retry.py:315 and run_one_attempt also mkdirs (runner.py:274) before any collect_artifacts call, so by the time _scan_hypothesis_attempts runs the workspace always exists on the production path. Only a direct unit call on a nonexistent dir triggers it.
- 验证者反证 proof: Live trigger path: quantize_via_prompt (retry.py:279) is the sole production entry (CLI cli.py:143 and orchestrator prelude quantization_request_handlers.py:72, both without runner_fn). It mkdirs the workspace at retry.py:315 ONCE, before the loop. Inside the loop it calls run_one_attempt (runner.py:229), which mkdirs at runner.py:274 only at the START, then runs the REAL Claude SDK agent with DEFAULT_ALLOWED_TOOLS=[Read,Write,Edit,Bash] (runner.py:28,311) inside/around the workspace, and returns AttemptResult WITHOUT re-creating the workspace (runner.py:360). Back in retry.py:355, collect_artifacts(workspace_path) runs IMMEDIATELY with no re-mkdir. If the untrusted LLM agent deletes its own workspace dir during the Bash session (errant/cleanup rm -rf, mv over the dir, or a vanished symlink target since retry.py:315 does .resolve()), then collect_artifacts hits _scan_hypothesis_attempts at result_collector.py:351, whose workspace.iterdir() at :266 raises FileNotFoundError (verified: Path.iterdir on a missing dir raises FileNotFoundError). Notably this is the ONLY scan in collect_artifacts that raises on a missing dir — every other field uses .is_file()/_read_text/_read_json/_has_glob which tolerate absence — so this try/except (result_collector.py:270) is the actual guard keeping the retry driver from crashing with an unhandled exception when the agent nukes its workspace. The auditor's dead-reason only reasons about the mkdir BEFORE the loop and ignores that an arbitrary-Bash LLM agent runs between the mkdir and the scan; the workspace is not guaranteed to exist at scan time.

### B3. `src/hyperloom/inference_optimizer/breakdown/collectors/telemetry.py:282`
- stillReachable: **True** | 置信度: high
- 原审计 deadReason: lane_capacity is created with CREATE TABLE IF NOT EXISTS in the current bus schema (orchestrator/bus/storage/schema.py:63) and seeded on init (schema.py:203-212); v2 added it. Current-run coordinator.db always has the table. Trigger only fires on pre-v2 archived DBs.
- 验证者反证 proof: The breakdown collector _collect_lane_timeline (telemetry.py:270-286) opens coordinator.db with a bare sqlite3.connect(str(db_path)) at line 271 and NEVER calls ensure_schema. So whether `lane_capacity` exists depends solely on what wrote the file, not on anything at breakdown time. The auditor's dead-reason ("current-run DB always has the table because the coordinator ran ensure_schema") only covers live current-run DBs; it does not make the branch dead. Live triggers still fire:

(1) v1 archived DBs rendered offline. dump_session_breakdown.py is explicitly the "offline / historical / batch / WekaFS" entrypoint (docstring lines 6-33, incl. a bulk `for d in /wekafs/users/*/inference_optimizer-sessions/*` loop calling `build`->collect_telemetry->_collect_lane_timeline). schema.py:20 documents v1 as the shape WITHOUT lane_capacity (v2 added it), and _migrate_leases_v1_to_v2 exists precisely because v1 DBs are a real, supported on-disk shape. A v1 coordinator.db captured on WekaFS and never reopened by a v2+ coordinator has no lane_capacity table; running the CURRENT tool against it does `SELECT lane, capacity FROM lane_capacity` -> sqlite3.OperationalError (no such table) -> fallback to DEFAULT_LANE_CAPACITIES. The auditor concedes archived sessions render but dismisses this as "pre-v2 only" — that IS a live trigger, not a dead one; nothing rewrites the archived file unless a live v2+ coordinator reopens it.

(2) Empty/partial DB file. open_connection (connection.py:78-89) does sqlite3.connect (which CREATES the file) THEN ensure_schema. If the process is killed between those, or ensure_schema's BEGIN IMMEDIATE fails on WAL/-shm corruption (a failure mode the code itself documents at connection.py:26-31 for WekaFS/NFS mounts), the file exists with no lane_capacity table. The end-of-session cli finally-block breakdown or a later offline dump then passes the db_path.exists() guard (telemetry.py:266) and hits the missing-table fallback.

Both paths execute lines 282-286 in current code.

### B4. `src/hyperloom/inference_optimizer/breakdown/reporters/_renderers/invocations.py:35`
- stillReachable: **True** | 置信度: high
- 原审计 deadReason: The naming is inverted: nothing writes a nested top-level 'invocations' dict. exporter.py:550-551 writes top-level 'geak_invocations'/'forge_invocations', recorder/sections.py:32-33 declares them as top-level file-born sections, and schema.py:2187-2188 has only geak_invocations/forge_invocations (no 'invocations' TypedDict field). So the PRIMARY lookup (breakdown.get('invocations') or {}).get(invocations_key) is always falsy for any exporter/recorder-produced breakdown; the branch labelled 'legacy_key' is in fact the only live path.
- 验证者反证 proof: The "legacy_key" fallback branch at invocations.py:35 (breakdown.get('geak_invocations')/'forge_invocations') is NOT dead — it is the LIVE primary production path, and the "dead" claim is backwards. Data flow proof: (1) exporter.py:550-551 writes session_breakdown.json with TOP-LEVEL keys 'geak_invocations'/'forge_invocations' and never a nested 'invocations' dict (confirmed by repo-wide grep — the only '"invocations":{' occurrence anywhere is exporter.py:301, which is a _safe_collect NAME string, not a written key). (2) recorder/sections.py:32-33 and schema.py:2187-2188 also declare these as top-level. (3) The consumer tools/dump_session_report.py (console path: python -m hyperloom.inference_optimizer.tools.dump_session_report --input session_breakdown.json) does json.loads(input) at line 109 and passes the dict verbatim to reporters.render_session_report (compose.py:142), which runs render_geak(breakdown). (4) In _render_pair line 35, breakdown.get('invocations') is None for every exporter-produced breakdown -> (None or {}).get('geak') is None -> execution FALLS THROUGH to breakdown.get('geak_invocations') = the branch the claim labels 'dead'. So the fallback fires on every real kernel-agent session report. The nested-'invocations' PRIMARY lookup is what is effectively unused in production — the only site producing {'invocations':{'geak':...}} is the single unit test test_reporters_v1_1.py:174. The auditor's own dead-reason text even concludes 'the branch labelled legacy_key is in fact the only live path', contradicting the DEAD verdict being audited.

### B5. `src/hyperloom/inference_optimizer/cli/preflight.py:33`
- stillReachable: **True** | 置信度: high
- 原审计 deadReason: Back-compat re-export shim retained for callers/tests after the credentials split; lean-3.MD lists this preflight credentials re-export block (id at preflight.py:33) as a legacy cut candidate. No production module was found importing these names from preflight (they live in credentials now); only kept for import-path stability.
- 验证者反证 proof: The auditor mischaracterized preflight.py:23-29 as a dead "back-compat re-export shim." It is a LIVE functional import: preflight.py imports _is_stale_proxy_url, _resolve_llm_endpoints, _reset_claude_config_to_upstream from .credentials because _preflight() itself calls all three. Concrete live call chain: CLI `optimize` -> cli/__init__.py:1364 `resolved_urls = _preflight(args)` (imported at cli/__init__.py:117-118). Inside _preflight (preflight.py:687): line 751 `anthropic_url, openai_url = _resolve_llm_endpoints()` (unconditional, every preflight run); line 772 `_reset_claude_config_to_upstream(claude_primary_key, anthropic_url)` (runs whenever any LLM URL resolves, i.e. the normal launch path, guarded only by `if anthropic_url or openai_url:` at 752); line 796 `if _is_stale_proxy_url(current):` (fires when a GEAK_BASE_URL/LLM_API_BASE operator override differs from the resolved gateway). None of these are re-exports for external callers — grep shows credentials.py defines them and preflight.py consumes them directly in production code. The auditor's premise ('No production module imports these from preflight; only kept for import-path stability') is backwards: preflight is exactly the production module that imports and uses them. lean-3.MD line 624 flags this block as a cut candidate, but cutting the import would break _preflight() at runtime (NameError on first optimize launch). Trigger is fully live.

### B6. `src/hyperloom/orchestrator/knowledge/cortex_t0.py:78`
- stillReachable: **True** | 置信度: high
- 原审计 deadReason: The `_field_sources`/`_sources` provenance markers were only stamped by the composite/multi-source Cortex recipe KB backend, which was DROPPED in PR #757 (a2d36e3f / e4d620f7 'refactor(kb): drop Cortex recipe KB backend and substrate parity'). A repo-wide grep shows NO code writes `_field_sources` or `_sources` on any recipe row (only readers exist: this fn, _warm_recipe_source, and breakdown/collectors/telemetry.py). The sole remaining remote client, GbrainRemoteRecipeClient, never emits them, and the local store/schema do not either. Both isinstance() branches inside are therefore never taken; the fn always returns "".
- 验证者反证 proof: The claim mislabels the fallback as dead. The characterized fallback behavior is the `return ""` at src/hyperloom/orchestrator/knowledge/cortex_t0.py:103 (default_when_missing). It is not just reachable — it is the ONLY reachable outcome and fires on every invocation in current code. Live call path: orchestrator/phases/machine.py:72 (and inference_optimizer/cli/kb.py:198) call run_t0_anchor -> _build_warm_start_context -> _warm_recipe_source (cortex_t0.py:140,143) -> _row_best_config_source. The row arguments (warm_point identity row and config_donor) come from the RecipeKB dispatcher (kb.get_recipe/kb.search) or the KG donor generator. I independently verified none of these stamp _field_sources/_sources: gbrain _page_to_recipe (gbrain_remote_client.py:471) returns a fixed key set with no markers; dispatcher._v2_to_arbor's extra-key splat (dispatcher.py:239-242) only carries markers if the remote payload already has them, and the sole remote GbrainRemoteRecipeClient never emits them; local_store.py/schema.py/gbrain_ingest.py have zero writers; generate_warmstart_donor_graph_guided (kg_client.py:1333) returns a donor dict with no markers. Repo-wide grep confirms the only writers of these keys are three TEST files (test_cortex_t0_anchor.py, test_knowledge_plane.py); all non-test occurrences (this fn + breakdown/collectors/telemetry.py) are readers. So because no current code stamps the markers, both isinstance branches are skipped and `return ""` executes on every real warm-start anchor. The auditor's own dead-reason ('the fn always returns ""') actually proves the fallback always executes; they conflated 'the marker-present branches are dead' (true) with 'the return "" fallback is dead' (false). The fallback trigger (row missing provenance) fires on 100% of production rows.

### B7. `src/hyperloom/orchestrator/knowledge/cortex_t0.py:139`
- stillReachable: **True** | 置信度: medium
- 原审计 deadReason: Both the donor-provenance branch (139-142) and the own-source branch (143-145) depend on _row_best_config_source returning a non-empty tag, which can only happen when a row carries `_field_sources`/`_sources`. Those markers were only produced by the composite/Cortex KB backend removed in PR #757; no current code writes them (see finding at line 78). So both branches are dead and the fn ALWAYS reaches the final `return "gbrain" if _remote_is_gbrain(kb) else "cortex-kb"` fallback at line 146.
- 验证者反证 proof: Branch at cortex_t0.py:143-144 (and donor branch 139-142) still fires on LEGACY PERSISTED rows, refuting the writer-only dead-reason. Chain: (1) schema.Recipe.from_dict (schema.py:442) buckets any non-well-known top-level key — including _field_sources/_sources — into `extras`, and to_dict (schema.py:389-391) re-splats extras at top level, so these markers are PRESERVED verbatim across rewrites. (2) LocalRecipeStore.get_recipe (local_store.py:613) returns the RAW on-disk _read_json dict unfiltered; dispatcher local fall-through returns it verbatim (dispatcher.py ~809). (3) run_t0_anchor's prior_extras exclusion set (cortex_t0.py:1011-1038) does NOT exclude _field_sources/_sources, so put_recipe re-stamps them onto disk. Trigger: a recipe.json written by the pre-#757 composite backend still sitting under $USER_DATA_PATH (no migration/purge exists — grep found none) → on resume/re-run for that model, warm-start L1 kb.get_recipe (cortex_t0.py:1102) returns it as warm_point → _warm_recipe_source (line 1274) → _row_best_config_source(row) (line 143) reads _field_sources['best_config'] and returns it at line 144, bypassing the line-146 fallback. Confirmed PR #757 (a2d36e3f) deleted composite_remote.py (the sole writer, its _merge_recipe_rows stamped these at lines 203-239) and no current writer or Composite class exists — but persistence keeps the reader branch live.

### B8. `src/hyperloom/orchestrator/knowledge/cortex_t0.py:187`
- stillReachable: **True** | 置信度: medium
- 原审计 deadReason: No current writer emits best_config with bare `args`/`envs` keys. Local writes go through _build_recipe_payload/result_recorder using `extra_server_args`; gbrain pages are projected into best_config via _best_config_dict (gbrain_remote_client.py:425-432) which only sets `extra_server_args`/`extra_envs`. Repo-wide grep for best_config literals with bare args/envs (excluding extra_server_args/extra_envs) finds none. Trigger can only fire for recipe rows persisted before the canonical rename (very old on-disk/gbrain data), not from any live code path.
- 验证者反证 proof: The fallback (`best_config.get("args")` / `best_config.get("envs")` at cortex_t0.py:187-188 in `_recipe_is_actionable`) CAN still fire in current code. Reachability chain, all verified in live (non-test) code:

1) VERBATIM STORAGE — nothing normalizes best_config keys. schema.py `Recipe.from_dict`/`to_dict` do `best_config=dict(d.get("best_config") or {})` / `dict(self.best_config)` (lines 455, 352); LocalRecipeStore.put_recipe stores `"best_config": dict(best_config or {})` (local_store.py:542); get_recipe (613) and search (708-720) return the on-disk dict verbatim. So whatever key shape lands on disk is exactly what `_recipe_is_actionable` reads back via kb.get_recipe/kb.search at cortex_t0.py:1102/1123/1146 -> lines 1129,1152,1165.

2) LIVE PRODUCER THAT COPIES BARE KEYS — result_recorder.py `_build_recipe_attrs_from_state` lines 875-877: `for key in ("extra_envs", "args", "envs", "name", "tput", "accuracy"): if key in current_best: best_config[key] = current_best[key]`. This unconditionally copies bare `args`/`envs` from current_best into the produced best_config; it is written to the KB via overrides["best_config"]=attrs["best_config"] (result_recorder.py:1090) -> _kb_amend_recipe -> put_recipe (proposals.py:258-306). The mirror path's `_NON_ENV_BEST_CONFIG_KEYS` frozenset (gbrain_ingest.py:153-163) lists the SAME bare-key set {"envs","args","name","tput","accuracy"}, so the pipeline is designed to carry them — the fallback is not vestigial. Critically, if current_best has bare `args` but no `extra_server_args`, best_config gets `args` WITHOUT `extra_server_args` (lines 872-877), which is exactly the shape where line-187's `extra_server_args or args` fallback is the operative branch.

3) LIVE INGRESS FOR THE BARE-KEY current_best — all fresh writers use canonical keys (writeback.py:261/853/957/1090, kernel.py:1394/1578/1895, prelude.py:566), BUT two live paths copy the prior current_best verbatim and preserve pre-existing keys before .update()ing canonical ones: kernel.py:979 (`cb = dict(self.shared_state.current_best or {})`) and resume.py:308 (`new_cb = dict(cb)`). current_best is persisted in state.json (shared_state.py:1011) and rehydrated verbatim on resume. resume.py:312-318 is an explicit "Legacy sessions before the append-only stack existed" recovery branch. Thus resuming a pre-canonical-rename session (old state.json / old recipe.json with bare args/envs) loads that current_best verbatim, the line-875 loop copies the bare keys into best_config, CLOSE re-persists them, and every warm-start read then hits the args/envs fallback. The auditor's own "very old on-disk/gbrain data" case is reached THROUGH live code (resume -> _build_recipe_attrs_from_state -> put_recipe -> warm-start search/get_recipe -> _recipe_is_actionable), not dead code. The documented arbor-compat contract (schema.py: operator points ARBOR_RECIPES_DIR at the store; arbor's native Recipe uses bare config keys) is an additional live external ingress.

The auditor's grep ("no best_config literal with bare args/envs") is too narrow: the bare keys are never a literal — they enter best_config dynamically via the result_recorder.py:875 key-copy loop from current_best, which the grep would not catch.

### B9. `src/hyperloom/orchestrator/knowledge/cortex_t0.py:211`
- stillReachable: **True** | 置信度: medium
- 原审计 deadReason: Same as line-187 finding: no live writer produces best_config with bare `args`/`envs`; canonical shape is extra_server_args/extra_envs everywhere (local store + gbrain projection). Only pre-rename persisted rows could trigger it.
- 验证者反证 proof: The fallback (best_config['args']/['envs'] when canonical keys absent) is reachable via multiple live paths, refuting the 'only pre-rename rows' dead-reason.

(1) LIVE WRITER emits bare keys: result_recorder.py:_build_recipe_attrs_from_state (lines 870-877), the current KB recipe writer (called from writeback.py:744 and result_recorder.py:1010), copies legacy keys verbatim: `for key in ("extra_envs","args","envs",...): if key in current_best: best_config[key] = current_best[key]`. current_best is deserialized from persisted state.json on resume (shared_state.current_best default_factory=dict, restored from disk), so a resumed/legacy state carrying bare args/envs produces a persisted best_config with bare keys and no extra_server_args/extra_envs — exactly the trigger. This is not a dead pre-rename artifact; the loop explicitly enumerates 'args'/'envs' in current code.

(2) LIVE READ PASSTHROUGH: dispatcher._v2_to_arbor is called on EVERY remote row (dispatcher.py:422). Line 207 copies best_config verbatim from body.best_config (central kb-service — an external shared store not shape-controlled by this repo), and the idempotency guard (lines 176-177) passes already-arbor gbrain rows through untouched via dict(v2_payload). Any external/older-client row with bare args/envs flows straight into the donor row consumed by _config_replay_args_envs at cortex_t0.py:519.

(3) Back-compat read plumbing is still wired, not vestigial: gbrain_ingest._best_config_split (line 205) still falls back to best_config.get('envs'); _NON_ENV_BEST_CONFIG_KEYS (line 153) explicitly lists 'args'/'envs'. The sibling gate at line 187-188 uses the identical fallback to decide actionability of the very donor consumed here.

The auditor's claim that no live writer/producer yields bare args/envs is false: result_recorder.py:875 is a live producer, and _v2_to_arbor is a live verbatim conduit for external rows.

### B10. `src/hyperloom/orchestrator/knowledge/recipe_kb/gbrain_ingest.py:205`
- stillReachable: **True** | 置信度: high
- 原审计 deadReason: Local best_config (from coordinator _build_recipe_payload / result_recorder) uses nested `extra_envs`; no live writer emits a bare `envs` nested map into best_config. The `envs` legacy read and the flat-sibling fallback only apply to old/externally-authored flat-shape configs.
- 验证者反证 proof: Live trigger exists. Path: result_recorder._build_recipe_attrs_from_state (src/hyperloom/orchestrator/loop/result_recorder.py:875) copies "envs" from current_best into best_config verbatim: `for key in ("extra_envs","args","envs","name","tput","accuracy"): if key in current_best: best_config[key] = current_best[key]`. When current_best carries a nested "envs" dict but no "extra_envs", best_config ends up with a nested `envs` map and no `extra_envs` -> exactly the line 205 precondition. That best_config flows unchanged into put_recipe via _kb_best_config_overrides_for_keep -> proposals._kb_amend_recipe (proposals.py:258 overrides.get("best_config"), :306 cortex_kb.put_recipe) and via cortex_t0.py:1059 best_config=dict(live.get("best_config")). GbrainMirroringRecipeKB.put_recipe (gbrain_ingest.py:425-427) intercepts EVERY live put_recipe and calls mirror_recipe(kwargs) -> recipe_to_page -> _best_config_split. This wrapper is the live coordinator's cortex_kb: cli/kb.py:134-141 returns GbrainMirroringRecipeKB when RECIPE_KB_MIRROR_MODE=inline and GBRAIN_BASE_URL/GBRAIN_TOKEN are set; cli/__init__.py:2004 passes cortex_kb=cortex_client to the Coordinator. "envs" is a genuine nested-dict shape in the pipeline (_server_lifecycle.py:155 / _grid_server_args.py:746: bench.setdefault("envs", {})), and current_best preserves arbitrary keys across resume (resume.py:293 new_cb=dict(cb)) and state.json round-trips (current_best is an unfiltered dict field, shared_state.py:500). The legacy-key read pattern is NOT dead elsewhere either: cortex_t0.py:188/212 and prelude.py:103/292 all still do best_config.get("extra_envs") or best_config.get("envs") in current code. The auditor cited result_recorder.py:945 as a canonical writer but did not read up to line 875, where "envs" is explicitly enumerated among copied keys - directly contradicting the dead-reason "no live writer emits a bare envs nested map into best_config".

### B11. `src/hyperloom/orchestrator/knowledge/recipe_kb/schema.py:452`
- stillReachable: **True** | 置信度: high
- 原审计 deadReason: Current writers persist framework_name only (to_dict line 349, local_store line 539). The legacy `framework` top-level recipe key is only present on rows persisted before the framework_name rename; no live code path emits it into a recipe dict.
- 验证者反证 proof: The fallback (schema.py:452 `framework_name=str(d.get("framework_name") or d.get("framework") or "")`) is REACHABLE via a live write-path, not just legacy reads. Chain:

1) result_recorder.py:437/460 (live benchmark-result recording) calls Coordinator._kb_amend_recipe.
2) proposals.py:196 reads the LOCAL on-disk recipe into `live`. A row persisted before the framework_name rename carries top-level legacy `framework` (the population the auditor concedes exists on disk).
3) proposals.py:214-240 builds prior_extras by excluding a `_reserved` set. That set contains "framework_name" (line 221) but NOT "framework" — so legacy `framework` survives into prior_extras -> merged_extras -> put_kwargs["extras"] (line 285). (Contrast cortex_t0.py:1020, the sibling amend path, which DOES reserve "framework" and correctly drops it — proposals.py does not.)
4) proposals.py:255 sets framework_name=`framework` where framework=str(getattr(ss,"framework","") or "") (line 188). SharedState.framework defaults to "" (shared_state.py:377), so on any amend before framework identity is stamped (or a resume/mock session that never stamped it) this is falsy "".
5) local_store.put_recipe builds payload_dict with framework_name (falsy "") and then splats extras via setdefault (local_store.py:558-563): payload_dict has no "framework" key yet, so legacy `framework` from extras is injected into payload_dict.
6) local_store.py:565 calls Recipe.from_dict(payload_dict). With payload_dict["framework_name"]=="" (falsy) and payload_dict["framework"]==<legacy value>, the `or d.get("framework")` fallback FIRES, recovering the legacy value.

So current code both (a) re-emits the legacy `framework` key into a dict passed to from_dict and (b) can present an empty framework_name alongside it, exercising exactly the disputed branch. The auditor's claim that "no live code path emits it into a recipe dict" is refuted by proposals.py's _reserved set omitting "framework".

### B12. `src/hyperloom/orchestrator/phases/framework.py:3148`
- stillReachable: **True** | 置信度: high
- 原审计 deadReason: SharedState.framework_agent_authoring_enabled defaults to True (shared_state.py:609) and is NEVER assigned False anywhere in src/ (no CLI flag, no env toggle, no resume override). The `not authoring_enabled` condition can therefore never be True in a fresh run; only a resume loading a hand-edited/very-old state.json with the key set false could trip it.
- 验证者反证 proof: The fallback at framework.py:3148 (`if want_author and not authoring_enabled: want_raw=True; want_author=False`) can still fire in a live run, no hand-edited state.json or resume required. The auditor only searched for literal `framework_agent_authoring_enabled =` assignments and CLI/env toggles, and missed the generic dynamic-setattr state-mutation channel:

1. `SharedState.apply_changes(changes, allow_core=False)` (src/hyperloom/orchestrator/state/shared_state.py:1651-1688) loops over an arbitrary `changes` dict and does `setattr(self, key, value)` for ANY key that is a dataclass field and NOT in `CORE_STATE_FIELDS`.
2. `framework_agent_authoring_enabled` IS a dataclass field (shared_state.py:609, default True) but is NOT in `CORE_STATE_FIELDS` (verified the full frozenset, gate.py:549-663 — the field is absent).
3. The `UPDATE_STATE` intent is a live, routable intent (protocol/intent.py:30, IntentType.UPDATE_STATE = "update_state"; payload key ("changes",) at intent.py:57). It is routed by intent_router.py:99-100 → `_handle_update_state` (intent_router.py:949-972), which calls `self.shared_state.apply_changes(intent.payload["changes"], allow_core=False)` and then `save()`.
4. PolicyGate `_validate_update_state` (gate.py:1015-1046) ONLY rejects `changes` keys that intersect `CORE_STATE_FIELDS`; a non-privileged role emitting `{"changes": {"framework_agent_authoring_enabled": false}}` passes the gate.

So an orchestration agent emitting an UPDATE_STATE intent flips the flag to False at runtime; on the next Critic-approved FRAMEWORK candidate, `_materialize_framework_agent_candidate` reads `authoring_enabled = getattr(...)` (framework.py:3142) == False, and for an author-route candidate (audit_step == "author_via_specialist", so want_author=True) the line-3148 branch fires and downgrades to raw diff. Also note the resume path `from_dict` (shared_state.py:920-1062) does `cls(**filtered)` keeping this known field, so a saved-then-reloaded False (produced by the apply_changes path above) also persists across resume — reinforcing reachability beyond a single tick.

### B13. `src/hyperloom/orchestrator/policy/gate.py:1644`
- stillReachable: **True** | 置信度: high
- 原审计 deadReason: get_specialist_patch_verdict is defined on _ExploreStateMixin, and current SharedState is `class SharedState(_RenderMixin, _ExploreStateMixin)`, so every real SharedState instance (Coordinator sets self.shared_state = SharedState.load_or_init(...) and passes it to PolicyGate) always has the method. The AttributeError branch can only fire for a hand-rolled duck-typed stub, not the production path.
- 验证者反证 proof: The `except AttributeError` at gate.py:1644 is broader than "SharedState lacks the method." It also catches an AttributeError raised INSIDE the method body. get_specialist_patch_verdict (explore_state.py:508) does `return self.specialist_patch_verdicts.get(sid, "") or ""`. The field is a plain dataclass field `specialist_patch_verdicts: dict[str, str] = field(default_factory=dict)` (shared_state.py:733) with NO __post_init__ and NO type coercion anywhere. SharedState.from_dict (shared_state.py:920) builds the instance as `cls(**filtered)` where `filtered = {k:v for k,v in raw.items() if k in known}` — it filters by KEY NAME only and never coerces values. So if the persisted state.json contains `"specialist_patch_verdicts": null` (or any non-dict), the production instance is constructed with `self.specialist_patch_verdicts = None`, and `None.get(...)` raises AttributeError. This is the exact production path: Coordinator (coordinator.py:695) does `self.shared_state = SharedState.load_or_init(session_dir)` (which calls from_dict on an existing state.json during resume) and passes it to `PolicyGate(shared_state=self.shared_state)` (coordinator.py:756). On an integrate_patch delegate, _validate_integrate_patch_critic_gate calls `ss.get_specialist_patch_verdict(sid)` -> AttributeError -> fallback fires (verdict=""). I proved it at runtime: `SharedState.from_dict({'schema_version':999,'specialist_patch_verdicts':None})` yields a real SharedState whose get_specialist_patch_verdict raises `'NoneType' object has no attribute 'get'`. The auditor only refuted the method-missing form of the AttributeError (a duck-typed stub) and missed this in-method form, which real, legitimately-loaded SharedState instances produce from a null/non-dict persisted field (state.json is operator-editable and consumed via lenient migration paths).

### B14. `src/hyperloom/orchestrator/specialists/rebench.py:66`
- stillReachable: **True** | 置信度: low
- 原审计 deadReason: The OS ephemeral-port allocator returning 8888 eight times in a row is practically impossible; the fallthrough is a defensive floor, not a normally-triggerable path
- 验证者反证 proof: The enclosing function _pick_free_port is definitively on a LIVE call path, not dead. Live chain: `python -m hyperloom.orchestrator.specialists.rebench` is a real subprocess entry the specialist prompt builder instructs agents to invoke (src/hyperloom/orchestrator/prompts/specialist_prompt_builder.py:925). main() defaults --port to 0 (rebench.py:265) -> run_specialist_rebench(port=0) -> _resolve_port(0) (rebench.py:149,84) -> _pick_free_port(). Internal callers integrate_patch.py and explore.py also reach run_specialist_rebench. So the function executes on every default-port rebench.

The specific fallthrough at line 68 (return 18888) fires only if all 8 `bind(("127.0.0.1", 0))` calls return exactly 8888. Whether that is possible depends solely on the OS ephemeral-port allocator, which the program neither controls nor asserts. On this host the range is 32768-60999 (verified via /proc/sys/net/ipv4/ip_local_port_range and an empirical 20-bind sample: min 33029, max 58505, never 8888), so under DEFAULT sysctl the branch cannot fire from bind(0). HOWEVER net.ipv4.ip_local_port_range is an operator-/namespace-configurable sysctl: setting it to a narrow band that includes 8888 (e.g. `8888 8888`, or a broad `1024 65535` where 8888 is the only free port) makes every bind(0) return 8888. Critically, the socket `with` block closes at line 64 BEFORE the port comparison at line 65, so a rejected 8888 is released and re-offered on the next iteration -- all 8 iterations can return 8888. I reproduced the branch firing (returns 18888) when getsockname yields 8888 eight times. Nothing in the repo (install.sh, SKILL.md, any source) sets or validates ip_local_port_range, so the trigger condition is not asserted away anywhere. It is a live, conditionally-executed defensive floor whose trigger is reachable via an OS setting outside the code's control -- I cannot prove no trigger path exists.


---

# 附录 C — 全量 fallback 编目统计(共 768 条)

> 源自 `fallback-survey-all.json`(完整 768 条 fallback 位置的结构化数据)。

## 按可达性
| reachability | 数量 |
|---|---|
| reachable | 715 |
| likely_unreachable | 36 |
| unreachable | 17 |

> 注:likely_unreachable(36)+ unreachable(17)= 53,即进入对抗式验证的疑似项;验证后 **39 确认死 / 14 被反证仍可达**。

## 按类别
| category | 数量 |
|---|---|
| default_when_missing | 286 |
| try_except | 237 |
| if_else_default | 99 |
| legacy_key | 88 |
| sdk_graceful | 48 |
| resume_downgrade | 6 |
| mock_downgrade | 3 |
| other | 1 |

## 按子系统
| unit | 数量 |
|---|---|
| orch/kernel | 68 |
| agents/critic | 62 |
| agents/framework | 61 |
| orch/trace+specialists+framework+scoring+prompts | 59 |
| orch/knowledge | 58 |
| io/cli | 53 |
| io/multi_node | 51 |
| agents/robustness | 46 |
| orch/phases | 45 |
| io/rest+common | 39 |
| orch/roles | 37 |
| agents/quantization | 36 |
| orch/actions | 34 |
| io/breakdown | 33 |
| orch/state+policy+bus | 28 |
| agents/kernel-tools | 26 |
| orch/loop | 24 |
| agents/kernel-rest | 8 |
