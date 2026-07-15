# 已清理函数清单

本表汇总本轮 `lean-3` 裁剪中从产品代码删除或退役的函数、方法、类和整模块入口。函数名、类名、路径、环境变量、模式名和 API 名称保持原文，其余说明使用中文。测试函数不逐条列入；与被删符号强绑定的专属测试已同步删除或改写。

## 回归修正

- `ActionRegistry.names` 不能列为删除项：`_inline_action_whitelist()` 通过它枚举 lane-light 动作，缺失会让 inline 白名单退化成空集合。当前已恢复 `names()`，`by_family` / `default_verdict_class_for` 仍可按独立证据评估。
- `source="resume_reverify_best"` 不能随 env 清理一起删除：它仍是 `_promote_to_shared_state()` 的 revalidation source，必须像 `resume_stack_revalidate` 一样清 `resume_pending_revalidation`。
- `_geak_legacy_promote` / `_promote_geak_result` 的删除只在确认 `writeback.py` 不再调用后成立。若 `_DELEGATED` 或调用点残留，会触发 collaborator 委托递归；当前代码已增加 `Coordinator.__getattr__` stale-delegate 防护，过期委派会抛清晰 `AttributeError`。

| 删除的函数 / 类 / 方法 | 使用位置 | 为什么可以删除 | 删除后替代 / 影响 |
|---|---|---|---|
| `CortexKBClient` 及 `__init__`, `list`, `upsert`, `batch_insert`, `add_edges`, `_strip_kind`, `_point_to_entry`, `_post`, `_backoff_for`, `_normalise_value`, `_normalise_scope` | `src/hyperloom/agents/critic/runtime/cortex_kb_client.py`；只被独立 critic 的 `CRITIC_KB_CLIENT_MODE=cortex` 分支和专属测试使用 | 产品入口只支持 `inmemory` / `live`；本轮高风险清理退役独立 `cortex` 模式 | 删除 `cortex` 分支和测试；`live` 模式继续由 `HTTPKBClient` 承担 |
| `_validate_payload` | `agents/critic/runtime/intent_envelope.py` | 与 `hyperloom.inference_optimizer.protocol.intent.validate_envelope` 的权威校验重复 | critic 运行时改为调用共享协议校验器，只保留 critic role 白名单 |
| `KBWriter.force_kb_unreachable` | `agents/critic/runtime/kb_writer.py`；测试/故障注入面 | 生产写入路径不调用 | 删除测试专用故障注入；真实不可达仍由异常路径处理 |
| `MetricsRegistry.snapshot`, `MetricsRegistry.reset` | `agents/critic/runtime/metrics.py`；测试读取/清空指标 | 运行时只写指标，不读取快照或重置 | 删除专属测试；指标记录行为保留 |
| `SessionMemory.list_decisions`, `list_events`, `is_msg_already_reviewed`, `reviewed_verdict_for`, `_read_jsonl` | `agents/critic/runtime/session_memory.py`；测试或未使用读取接口 | critic runtime 实际只需要追加和当前 review 流读取 | 删除测试专用查询面；session memory 追加行为不变 |
| `WebFetchClient.call_count`, `WebSearchClient.call_count` | `agents/critic/runtime/web_tools/{fetch,search}_client.py` | 只写不读，生产不依赖调用计数 | 删除字段和测试断言；请求行为不变 |
| `contribute_to_kb_for_framework` | `agents/framework/kb.py` | 已被按 domain 组织的 `contribute_to_kb` 取代，无生产调用 | 调用方使用统一 domain KB 写入路径 |
| `load_kb_content` | `agents/framework/kb.py` | 旧 KB 文本加载入口无运行时入口 | 无替代；删除测试 |
| `_normalise_keywords` | `agents/framework/kb.py` | 只服务已删除的 leaderboard / associate 辅助面 | 随旧辅助面删除 |
| `associate_prs_to_gap`, `leaderboard_for_gap` | `agents/framework/kb.py` | 测试专用旧 PR ledger / leaderboard 面，当前 agent 不走该路径 | 删除旧 PR ledger 测试；保留当前 PR discovery / KB 路径 |
| `score_title_against_keywords` | `agents/framework/keywords.py` | 已被 `score_title_with_anti_signal` 取代 | 使用带反信号的标题打分逻辑 |
| `rank_titles` | `agents/framework/enablement_discovery.py` | 调用方已直接排序或使用新 ranking 逻辑 | 删除包装函数；排序行为内联 |
| `_http_get_text` | `agents/framework/sources/primus_cortex.py` | 无生产调用 | 无替代 |
| `parse_meta_frontmatter`, `parse_meta_title` | `agents/framework/pr_kb.py` | 只被 meta page 测试路径使用，当前 diff-fetch 不读取这些字段 | 删除 meta parse 测试；PR KB diff/index 路径保留 |
| `meta_slug` | `agents/framework/pr_kb_slug.py` | 旧 meta slug 测试路径专用 | 删除测试；`.git` URL slug 标准化改用正确包路径 |
| `geak_vs_forge_driver._load_fusion_cues`, `_build_workload_context`, `main` | `agents/kernel/tools/geak_vs_forge_driver.py` | 整脚本没有生产入口或 CLI 入口 | 删除整文件；GEAK / FORGE 当前后端不受影响 |
| `quantize_via_prompt_sync` | `agents/quantization/driver/retry.py` | CLI 使用异步 `quantize_via_prompt`，同步包装层多余 | `agents.quantization` 只导出异步 `quantize_via_prompt` |
| `BackendTurnResult` | `agents/robustness/role/envelope.py` | 上游镜像数据类，当前 intent 路径不消费 | 删除 role 导出和测试 |
| `Reactor.last_symptoms`, `_summarise` | `agents/robustness/role/reactor.py` | 每 tick 计算但无人读取 | 停止保存 last-symptoms 调试状态；reactor 输出不变 |
| `evaluate_aiter_jit_signals`, `evaluate_gpu_leak_signals`, `evaluate_progress_signals` | `agents/robustness/signals/{aiter_jit,gpu_leak,progress}.py` | classifier 直接调用 stateful detector 的 `.evaluate()` | 删除模块级转发函数；detector 类继续注册和运行 |
| `ProgressDetector.gain_history` | `agents/robustness/signals/progress.py` | 测试专用 property | 删除 property 和测试断言 |
| `_normalise_extra_server_args_key` | `agents/robustness/signals/repeated_payload.py` | `extra_sglang_args` 生产兼容层已退役 | repeated-payload 指纹只看 `extra_server_args` |
| `SourceData.merge_from`, `call_with_timeout` | `agents/robustness/sources/base.py` | 预留合并/超时辅助函数，生产未调用 | 删除辅助函数；source data 形状不变 |
| `_ProbeOutcome` | `agents/robustness/sources/local_probe.py` | local probe 不再返回该中间对象 | 删除数据类 |
| `RobustnessServerClient.health`, `list_sessions`, `get_session`, `get_session_metrics`, `list_cluster_pod_metric_categories` | `agents/robustness/sources/server_client.py` | 预留 REST 方法，实际 Source adapter 未调用 | 删除测试专用 REST 接口；保留当前 server metric 获取路径 |
| `DetectorStateView.slot_name`, `is_persistent` | `agents/robustness/state_store.py` | 测试专用 accessor，运行时不读取 | 删除 property 和测试 |
| `common.io.read_jsonl`, `common.io.tail_lines` | `common/io.py` | 仅测试使用；`read_jsonl` 与 `common.jsonio.read_jsonl` 重复 | JSONL 读取使用 `hyperloom.common.jsonio.read_jsonl`；tail 需求使用专用逻辑 |
| `common.payload_aliases._coerce_str`, `read_extra_server_args`, `migrate_legacy_key_in_place` | `common/payload_aliases.py` | `extra_sglang_args` 生产兼容已退役 | 删除整个 `common.payload_aliases`；生产只读写 `extra_server_args` |
| `common.timeutil.parse_iso_unix` | `common/timeutil.py` | 与局部 timestamp parser 重复，无生产调用 | 保留 `now_iso` / `iso_z` 等当前时间辅助函数 |
| `BaselineSummary.from_dict` | `baseline_comparison/types.py` | `ReportExecutor` 读取原始 JSON，不构造该 dataclass | 删除反序列化方法和测试 |
| `breakdown.exporter._load_json`, `_load_state`, `_load_manifest` | `breakdown/exporter.py` | 重复 `common.jsonio.read_json` 和 session path 权威函数 | exporter 直接使用规范 JSON 和 session path helper |
| `section_shape` | `breakdown/recorder/sections.py` | recorder 不查询 shape helper | 删除辅助函数和导出 |
| `renderer_names`, `fmt_int` | `breakdown/reporters/base.py` | renderer 注册和 compose 不需要这些公开 helper | 删除辅助函数 |
| `_CacheCounts`, `safe_load_state`, `flatten_discovered_flag_names`, `extract_proposed_flags`, `count_analysis_md_references`, `cache_hit_rate`, `aggregate_cache_tokens` | `inference_optimizer/experiments/_roofline_audit_common.py` | 开发用 roofline audit 公共模块，无产品入口 | 删除整模块；保留 `roofline_sweep.py` |
| `ab_torch_compile_kernels.*` 入口和辅助函数 | `experiments/ab_torch_compile_kernels.py` | 开发用 A/B 脚本，无文档、CLI 或产品引用 | 删除整脚本和专属测试 |
| `ab_torch_compile_magpie.*` 入口和辅助函数 | `experiments/ab_torch_compile_magpie.py` | 开发用 A/B 脚本，无文档、CLI 或产品引用 | 删除整脚本 |
| `AuditReport`, `extract`, `_format_*`, `render`, `_parse_args`, `main` | `experiments/audit_roofline_decisions.py` | roofline-v2 开发审计脚本，无产品入口 | 删除整脚本和测试 |
| `SessionMetrics`, `_parse_iso`, `_wall_clock_min`, `_count_*`, `extract`, `_format_action_seq`, `render`, `_exit_code_for_delta`, `_parse_args`, `main` | `experiments/verify_roofline_v2.py` | roofline-v2 开发验证脚本，无产品入口 | 删除整脚本 |
| `framework_registry.get`, `kind`, `supports_server_reuse`, `repo_url` | `framework_registry.py` | 零生产调用或测试专用公开面 | 保留 `names`, `is_scriptable`, primary metric 等当前接口 |
| `discover_worker_pods` | `multi_node/_internal/dynamo_support.py` | Dynamo 命令路径不调用 | 删除辅助函数和测试 |
| `multi_node.cli._b64`, `_wrap_for_dash` | `multi_node/cli.py` | `RayDashboardClient.submit_job` 已经内部包装 | 删除重复包装；提交行为由 client 统一处理 |
| `multi_node.cli._validate_extra_server_args`, `multi_node.commands.dynamo._validate_extra_server_args` | `multi_node/cli.py`, `multi_node/commands/dynamo.py` | 单行包装，重复 `validate_server_args` | 调用方直接使用 `validate_server_args` |
| `install_kernel_tools_on_pods_best_effort` | `multi_node/cli.py` | 测试专用 best-effort hook，当前脚本分发不走该函数 | 删除辅助函数和测试 |
| `multi_node.cli.__getattr__` | `multi_node/cli.py` | tree-reform 兼容垫片，旧导出名无生产调用 | 删除 `_RAYJOB_COMPAT_EXPORTS` / `_DYNAMO_COMPAT_EXPORTS` |
| `recipe_snapshot_constants.format_recipe_path` | `recipe_snapshot_constants.py` | 已死 endpoint-template formatter | 删除；保留仍使用的常量 |
| `asset_scripts_dir`, `asset_kernel_opt_dir`, `agent_session_dir`, `kernel_agent_root` | `session/paths.py` | session layout 不再使用这些测试专用叶子路径 | 删除 helper；保留当前 session path 权威接口 |
| `kernel_workspace`, `report_file`, `ext_trace_path`, `agent_log`, `optimizer_run_log`, `optimizer_run_pidfile`, `agent_inbox`, `agent_outbox`, `agent_persona`, `cortex_sid_file` | `session/session_paths.py` | SQLite bus 和当前 session layout 已取代这些旧文件路径 | 删除 helper；保留当前路径函数 |
| `GridVariant.__init__` 中的 `extra_sglang_args` 分支 | `actions/executors/_grid_base.py` | `extra_sglang_args` 兼容层已退役 | 构造函数只接受 `extra_server_args` |
| `_grid_runner.pick_winners` | `actions/executors/_grid_runner.py` | 没有生产调用方 | winner 选择由当前 grid / explore 流程处理 |
| `explore._gain_pct` | `actions/executors/explore.py` | 重复 `common.gain_math.gain_pct` | 直接使用规范 gain helper |
| `report._atomic_write_text` | `actions/executors/report.py` | 重复 `common.io.atomic_write_text` | 直接调用 common IO helper |
| `ActionRegistry.default_verdict_class_for`, `by_family` | `actions/registry.py` | 薄包装或测试专用公开面 | 使用 `ActionRegistry.all()` 和 metadata；`names()` 已恢复为 inline whitelist 的活 API |
| `CursorStore.is_already_processed` | `bus/cursor_store.py` | 测试专用 accessor | 当前 cursor 流使用 mark / read cursor |
| `pending_proposal_seqs` | `bus/db_maintenance.py` | 测试专用 SQL helper | 维护逻辑已内联或重构 |
| `MessageBus.count` | `bus/message_bus.py` | bus 消费者不按总数查询 | 删除方法 |
| `SqliteLeaseBackend.active_lanes`, `ResourceLockManager.active_lanes`, `ResourceLockManager.counters_snapshot` | `bus/resource_lock.py` | 测试专用 inspection 接口 | 运行时只需要 lease acquire / release |
| `T0Result` | `knowledge/cortex_t0.py` | 字段被调用方丢弃 | T0 anchor 流只保留副作用和状态 |
| `PRMonitorError` | `knowledge/pr_monitor.py` | REST 时代遗留 stub exception | PR monitor stub 只剩 MCP enablement gate |
| `RecipeKB.delete_recipe`, `get_history`, `list_recent`, `list_attempts`, `list_session_attempts` | `knowledge/recipe_kb/dispatcher.py` | 当前循环只使用 `search` / `put_recipe` / `append_attempt` / `close` | 删除 dispatcher 查询面 |
| `gbrain_ingest.ingest_local_to_gbrain`, `main` | `knowledge/recipe_kb/gbrain_ingest.py` | 未注册批量导入 CLI 入口 | 删除孤立 CLI；保留仍使用的 mirror helper |
| `GbrainRemoteRecipeClient.health`, `get_history`, `list_recent`, `list_attempts`, `list_session_attempts`, `session_summary` | `knowledge/recipe_kb/gbrain_remote_client.py` | 当前远端 client 只需要 search / upsert / attempt write 子集 | 删除远端查询面 |
| `Fact.loss`, `KGClient.retract_fact_safe` | `knowledge/recipe_kb/kg_client.py` | 测试专用 property 和未使用 retract 方法 | 删除 |
| `local_store._atomic_write_json` | `knowledge/recipe_kb/local_store.py` | 重复 common IO atomic JSON | 使用 `common.io.atomic_write_json` |
| `LocalRecipeStore._walk_cid_dirs`, `get_history`, `delete_recipe`, `list_all_live_recipes`, `list_recent`, `list_attempts`, `list_session_attempts`, `purge_recipe` | `knowledge/recipe_kb/local_store.py` | 旧本地读删接口；当前只需 get / search / put / append / close | 删除接口和测试 |
| `_normalise_findings`, `_normalise_failures`, `_normalise_gaps`, `_normalise_pitfalls` | `knowledge/recipe_kb/local_store.py` | 四个单调用包装函数 | 内联 `_normalise_str_dicts` 调用 |
| `research_hints._atomic_write` | `knowledge/research_hints.py` | 重复 common IO | 使用 `common.io.atomic_write_text` |
| `_merge_cumulative_extra_sglang_args` | `loop/coordinator_helpers.py` | 旧字段名相关 merge helper | 由 `_merge_cumulative_extra_server_args` 取代 |
| `DispatcherCollaborator._wait_for_task_terminal` | `loop/dispatcher.py` | CLOSE 重构后成为孤儿方法 | 删除方法和 `_DELEGATED` 条目 |
| `GatingCollaborator._target_analysis_baseline_exists`, `_kernel_opt_keep_pending` | `loop/gating.py` | 包装方法，只被测试调用 | 底层检查或直接逻辑保留 |
| `ProposalsCollaborator._resolve_issue_canonical` | `loop/proposals.py` | 未接入 proposal 流 | 删除辅助函数 |
| `_geak_legacy_promote`, `_promote_geak_result` | `phases/kernel.py` | 退役默认关闭的 GEAK provisional-first 路径 | GEAK 固定为 rebench-first：`_record_geak_candidate` 后由 `_promote_geak_from_candidate` 写入实测结果 |
| `PolicyGate.allowed_tools_for_action`, `validate_tool_invocation` | `policy/gate.py` | 旧 tool invocation API，无当前调用方 | 当前 policy gate 使用 action/path 校验路径 |
| `critic_prompt_builder.*`, `build_critic_prompt` | `orchestrator/prompts/critic_prompt_builder.py` | 运行时使用静态 `orchestrator/prompts/critic.md` | 删除整模块和专属测试 |
| `build_specialist_prompts_for_domain` | `prompts/specialist_prompt_builder.py` | 便捷包装函数，调用方使用更底层 prompt builder | 删除包装函数 |
| `ClaudeBackend.has_emit_intent_tool`, `has_context_tools`, `conversation_session_id` | `roles/claude.py` | 测试专用 accessor | 删除 property |
| `MockKernelBackend`, `__init__`, `run` | `roles/kernel_mock.py` | 生产 backend 构建逻辑从不选择 mock kernel role | 删除模块和导出；测试改用 `MockBackend` 或局部 responder |
| `source_snapshot._symlink_mirror`, nested `rec`, `materialize_source_layer` | `orchestrator/source_snapshot.py` | Python reader / materializer 只被测试使用 | GEAK 侧有独立 materialization |
| `SpecialistProfile.is_cross_domain` | `specialists/profile.py` | 测试专用 property | dispatch 使用显式 domain / kind 字段 |
| `_ExploreStateMixin.replace_gaps`, `get_intervention_mix`, nested `_ct`, `has_seen_pr_id` | `state/_shared_state/explore_state.py` | 测试专用或旧 summary helper | 当前 robustness tool 使用 `to_intervention_mix_summary`；PR tracking 直接用 ledger |
| `Objective.remaining_gap`, `Objective.pressure_input` 及具体实现 | `state/objective.py` | `scheduler.pressure()` 未使用；live loop 使用 `progress`, `reached`, `kind`, `describe` | 删除抽象方法和实现 |
| `_migrate_legacy_extra_sglang_args_keys` | `state/shared_state.py` | 旧 session 的 `extra_sglang_args` 恢复兼容已退役 | state 只读规范键 |
| `SharedState.kernel_patch_key` | `state/shared_state.py` | 包装 kernel patch identity | 调用方直接导入 kernel decision helper |
| `conversation_trace._validate_row`, `llm_trace._validate_row` | `trace/{conversation_trace,llm_trace}.py` | 私有 row 校验包装重复 | 内联校验逻辑 |
| `langfuse_mapping.span_key` | `trace/langfuse_mapping.py` | 测试专用 helper | 删除 |
| `request_handlers` 的 `_kernel_decisions` 重导出尾巴 | `orchestrator/kernel/request_handlers.py` | 兼容导入面，真实调用方已迁移 | 调用方直接使用 `orchestrator.kernel._kernel_decisions` |
| `request_handlers._HONEST_E2E_UMBRELLA_ENV`, `_FALSEY` | `orchestrator/kernel/request_handlers.py` | 死常量重导出 | 删除导入和导出 |
| CLI root 私有 helper 重导出块 | `inference_optimizer/cli/__init__.py` | 生产和测试调用方已迁移到具体 `cli.*` 模块 | 包根只暴露 `main`，新增 `cli/__main__.py` 支持 `python -m` |
| `extra_sglang_args` 读写兼容分支 | `result_recorder.py`, `prelude.py`, `writeback.py`, `resume.py`, `optimization_journal.py`, `cortex_t0.py`, `kernel/_kernel_decisions.py` 等 | 高风险退役旧字段名 | 只使用 `extra_server_args` / `candidate_extra_server_args`；唯一保留遗留岛是 standalone kernel-agent `_payload_aliases.py` |

## 验证快照

- `python -m ruff check src/hyperloom` 已通过。
- `python -m ruff check ci/test_ci_transform_v2.py ci/transform_to_session_summary_v2.py pyproject.toml` 已通过。
- `pytest -m "not critic_agent_e2e and not robustness_agent_e2e" -q` 已通过：`9985 passed, 5 skipped, 47 deselected, 12 subtests passed`。
- `mypy src/hyperloom` 未运行成功，因为当前环境没有安装 `mypy`。
