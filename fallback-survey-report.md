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