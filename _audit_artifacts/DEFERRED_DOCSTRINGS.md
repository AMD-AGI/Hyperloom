# Docstring Deferred-Work Tracker

> STATUS: COMPLETE. All deferred items in this tracker were documented in a
> later pass. A full-repo rescan (tests excluded) reports **0 remaining gaps**,
> and every edited package compiles cleanly. The file lists below are retained
> for historical reference only.


Generated from `docstring_inventory.json`. Root: `C:\Users\troosta\Hyperloom`

- Files with gaps: **276**
- Total gaps: **3163**
- By kind: `{'function': 3076, 'class': 81, 'module': 6}`

Test files were excluded from the scan.

## Pass 1 (this pass) - core inference_optimizer

| Status | Gaps | File |
|---|---|---|
| [ ] | 144 | `inference_optimizer\orchestrator\coordinator.py` |
| [ ] | 94 | `inference_optimizer\orchestrator\shared_state.py` |
| [ ] | 90 | `inference_optimizer\breakdown\collectors.py` |
| [ ] | 66 | `inference_optimizer\cli.py` |
| [ ] | 47 | `inference_optimizer\orchestrator\kernel_request_handlers.py` |
| [ ] | 45 | `inference_optimizer\session_paths.py` |
| [ ] | 41 | `inference_optimizer\orchestrator\policy.py` |
| [ ] | 37 | `inference_optimizer\orchestrator\phase_state.py` |
| [ ] | 35 | `inference_optimizer\orchestrator\objective.py` |
| [ ] | 35 | `inference_optimizer\orchestrator\action_executors\_grid_runner.py` |
| [ ] | 34 | `inference_optimizer\breakdown\schema.py` |
| [ ] | 19 | `inference_optimizer\paths.py` |

## Deferred to later passes

Grouped by top-level component, sorted by gap count within each.

### `inference_optimizer/` - 1036 gaps in 133 files

| Gaps | File |
|---|---|
| 53 | `inference_optimizer\multi_node\cli.py` |
| 35 | `inference_optimizer\recipe_kb\local_store.py` |
| 27 | `inference_optimizer\orchestrator\system_prompts\specialist_prompt_builder.py` |
| 23 | `inference_optimizer\orchestrator\resource_lock.py` |
| 21 | `inference_optimizer\orchestrator\specialist_runner.py` |
| 18 | `inference_optimizer\orchestrator\roofline_snapshot.py` |
| 18 | `inference_optimizer\orchestrator\action_executors\_server_patcher.py` |
| 18 | `inference_optimizer\recipe_kb\remote_client.py` |
| 16 | `inference_optimizer\multi_node\scripts\launch_multinode.py` |
| 16 | `inference_optimizer\orchestrator\roofline_ceiling.py` |
| 16 | `inference_optimizer\orchestrator\system_prompts\prompt_builder.py` |
| 15 | `inference_optimizer\orchestrator\knowledge_plane.py` |
| 15 | `inference_optimizer\orchestrator\task_registry.py` |
| 15 | `inference_optimizer\orchestrator\action_executors\profile.py` |
| 15 | `inference_optimizer\orchestrator\backends\critic_agent.py` |
| 15 | `inference_optimizer\scripts\verify_roofline_v2.py` |
| 15 | `inference_optimizer\storage\connection.py` |
| 14 | `inference_optimizer\manifest.py` |
| 14 | `inference_optimizer\multi_node\_internal\safe_client.py` |
| 14 | `inference_optimizer\orchestrator\message_bus.py` |
| 14 | `inference_optimizer\orchestrator\action_executors\benchmark_result.py` |
| 14 | `inference_optimizer\recipe_kb\dispatcher.py` |
| 13 | `inference_optimizer\orchestrator\dynamic_action_runner.py` |
| 13 | `inference_optimizer\orchestrator\framework_paths.py` |
| 13 | `inference_optimizer\orchestrator\optimization_journal.py` |
| 13 | `inference_optimizer\orchestrator\action_executors\explore.py` |
| 13 | `inference_optimizer\orchestrator\action_executors\integrate_patch.py` |
| 13 | `inference_optimizer\orchestrator\action_executors\_inferencex_patcher.py` |
| 13 | `inference_optimizer\orchestrator\backends\claude.py` |
| 13 | `inference_optimizer\scripts\audit_roofline_decisions.py` |
| 12 | `inference_optimizer\orchestrator\dynamic_action_resume.py` |
| 12 | `inference_optimizer\orchestrator\action_executors\recover.py` |
| 12 | `inference_optimizer\orchestrator\action_executors\report.py` |
| 12 | `inference_optimizer\orchestrator\action_executors\roofline.py` |
| 11 | `inference_optimizer\breakdown\reporters\cross_section.py` |
| 11 | `inference_optimizer\multi_node\_internal\ray_dashboard.py` |
| 11 | `inference_optimizer\orchestrator\action_executors\framework_pr.py` |
| 11 | `inference_optimizer\orchestrator\system_prompts\critic_prompt_builder.py` |
| 10 | `inference_optimizer\baseline_comparison\target_analyzer.py` |
| 10 | `inference_optimizer\orchestrator\cursor_store.py` |
| 10 | `inference_optimizer\orchestrator\dynamic_action_seed_kit.py` |
| 10 | `inference_optimizer\orchestrator\dynamic_action_tools.py` |
| 10 | `inference_optimizer\orchestrator\pr_monitor.py` |
| 10 | `inference_optimizer\orchestrator\specialist_subprocess.py` |
| 9 | `inference_optimizer\multi_node\scripts\apply_tracelens_patch_multinode.py` |
| 9 | `inference_optimizer\multi_node\scripts\kernel_patch_multinode.py` |
| 9 | `inference_optimizer\orchestrator\dynamic_action_critic.py` |
| 9 | `inference_optimizer\scripts\ab_torch_compile_kernels.py` |
| 8 | `inference_optimizer\breakdown\reporters\base.py` |
| 8 | `inference_optimizer\orchestrator\action_registry.py` |
| 8 | `inference_optimizer\orchestrator\dynamic_action_pipeline.py` |
| 8 | `inference_optimizer\orchestrator\action_executors\baseline.py` |
| 8 | `inference_optimizer\recipe_kb\schema.py` |
| 7 | `inference_optimizer\baseline_comparison\inferencex_client.py` |
| 7 | `inference_optimizer\multi_node\scripts\kernel_bench_multinode.py` |
| 7 | `inference_optimizer\orchestrator\dynamic_action_proposal.py` |
| 7 | `inference_optimizer\orchestrator\sub_agent_runner.py` |
| 7 | `inference_optimizer\orchestrator\action_executors\_multi_node_server_lifecycle.py` |
| 7 | `inference_optimizer\orchestrator\action_executors\_subprocess_kill.py` |
| 7 | `inference_optimizer\orchestrator\system_prompts\dynamic_action_prompt_builder.py` |
| 6 | `inference_optimizer\breakdown\exporter.py` |
| 6 | `inference_optimizer\orchestrator\action_executors\target_analysis.py` |
| 6 | `inference_optimizer\orchestrator\action_executors\_multi_node_env.py` |
| 6 | `inference_optimizer\orchestrator\backends\robustness_agent.py` |
| 6 | `inference_optimizer\scripts\ab_torch_compile_magpie.py` |
| 6 | `inference_optimizer\storage\schema.py` |
| 5 | `inference_optimizer\recipe_snapshot_constants.py` |
| 5 | `inference_optimizer\breakdown\reporters\compose.py` |
| 5 | `inference_optimizer\breakdown\reporters\_renderers\kernel_lifecycle.py` |
| 5 | `inference_optimizer\examples\p0_main_loop.py` |
| 5 | `inference_optimizer\multi_node\scripts\launch_router.py` |
| 5 | `inference_optimizer\orchestrator\framework_agent_client.py` |
| 5 | `inference_optimizer\orchestrator\intent_parser.py` |
| 5 | `inference_optimizer\orchestrator\action_executors\sweep.py` |
| 5 | `inference_optimizer\orchestrator\action_executors\_magpie_patcher.py` |
| 5 | `inference_optimizer\orchestrator\backends\mock_backend.py` |
| 4 | `inference_optimizer\baseline_comparison\types.py` |
| 4 | `inference_optimizer\breakdown\reporters\llm_client.py` |
| 4 | `inference_optimizer\breakdown\reporters\_renderers\kernel_profiling.py` |
| 4 | `inference_optimizer\compat\payload_aliases.py` |
| 4 | `inference_optimizer\examples\p1_5_real_claude_demo.py` |
| 4 | `inference_optimizer\examples\p1_6_e2e_baseline_demo.py` |
| 4 | `inference_optimizer\examples\p1_7_three_agents_e2e_demo.py` |
| 4 | `inference_optimizer\multi_node\_internal\log.py` |
| 4 | `inference_optimizer\multi_node\_internal\workload_spec.py` |
| 4 | `inference_optimizer\orchestrator\agent_role.py` |
| 4 | `inference_optimizer\orchestrator\dynamic_action_history.py` |
| 4 | `inference_optimizer\orchestrator\kb_writeback.py` |
| 4 | `inference_optimizer\orchestrator\action_executors\dynamic_action.py` |
| 4 | `inference_optimizer\orchestrator\action_executors\_framework_gap_composer.py` |
| 4 | `inference_optimizer\orchestrator\action_executors\_robustness_pulse.py` |
| 4 | `inference_optimizer\orchestrator\action_executors\_workload_envs.py` |
| 4 | `inference_optimizer\orchestrator\backends\mcp_emit_intent.py` |
| 4 | `inference_optimizer\recipe_kb\canonical_id.py` |
| 4 | `inference_optimizer\scripts\dump_session_breakdown.py` |
| 3 | `inference_optimizer\breakdown\reporters\llm_prompt.py` |
| 3 | `inference_optimizer\breakdown\reporters\_renderers\invocations.py` |
| 3 | `inference_optimizer\breakdown\reporters\_renderers\roofline.py` |
| 3 | `inference_optimizer\breakdown\reporters\_renderers\_invocation.py` |
| 3 | `inference_optimizer\examples\p2_full_optimize_demo.py` |
| 3 | `inference_optimizer\multi_node\scripts\kill_multinode.py` |
| 3 | `inference_optimizer\orchestrator\_analysis_keyword_map.py` |
| 3 | `inference_optimizer\orchestrator\action_executors\_accuracy_gate.py` |
| 3 | `inference_optimizer\orchestrator\backends\codex.py` |
| 3 | `inference_optimizer\scripts\dump_session_report.py` |
| 2 | `inference_optimizer\tracelens_md.py` |
| 2 | `inference_optimizer\breakdown\reporters\_renderers\data_provenance.py` |
| 2 | `inference_optimizer\breakdown\reporters\_renderers\decision_journal.py` |
| 2 | `inference_optimizer\breakdown\reporters\_renderers\kernel_decision_path.py` |
| 2 | `inference_optimizer\breakdown\reporters\_renderers\sweep.py` |
| 2 | `inference_optimizer\orchestrator\cortex_t0.py` |
| 2 | `inference_optimizer\orchestrator\specialist_domains.py` |
| 2 | `inference_optimizer\orchestrator\action_executors\session_breakdown.py` |
| 2 | `inference_optimizer\orchestrator\action_executors\_canonical_fingerprint.py` |
| 2 | `inference_optimizer\orchestrator\action_executors\_explore_roofline_filter.py` |
| 2 | `inference_optimizer\orchestrator\backends\base.py` |
| 2 | `inference_optimizer\orchestrator\backends\critic_mock.py` |
| 2 | `inference_optimizer\orchestrator\backends\kernel_mock.py` |
| 2 | `inference_optimizer\orchestrator\backends\robustness_mock.py` |
| 1 | `inference_optimizer\baseline_comparison\name_mapping.py` |
| 1 | `inference_optimizer\breakdown\claw_mirror.py` |
| 1 | `inference_optimizer\breakdown\reporters\_renderers\attribution.py` |
| 1 | `inference_optimizer\breakdown\reporters\_renderers\baseline.py` |
| 1 | `inference_optimizer\breakdown\reporters\_renderers\capability_summary.py` |
| 1 | `inference_optimizer\breakdown\reporters\_renderers\critic_robustness.py` |
| 1 | `inference_optimizer\breakdown\reporters\_renderers\final.py` |
| 1 | `inference_optimizer\breakdown\reporters\_renderers\param_search.py` |
| 1 | `inference_optimizer\breakdown\reporters\_renderers\phase_timeline.py` |
| 1 | `inference_optimizer\breakdown\reporters\_renderers\session.py` |
| 1 | `inference_optimizer\breakdown\reporters\_renderers\source_files.py` |
| 1 | `inference_optimizer\breakdown\reporters\_renderers\workload.py` |
| 1 | `inference_optimizer\orchestrator\specialist_mcp_config.py` |
| 1 | `inference_optimizer\scripts\event_counts.py` |

### `robustness-agent/` - 464 gaps in 55 files

| Gaps | File |
|---|---|
| 44 | `robustness-agent\src\robustness_agent\sources\local_probe.py` |
| 21 | `robustness-agent\src\robustness_agent\sources\server_client.py` |
| 20 | `robustness-agent\src\robustness_agent\decision\rca_engine.py` |
| 17 | `robustness-agent\src\robustness_agent\finalize\postmortem.py` |
| 16 | `robustness-agent\src\robustness_agent\conductor.py` |
| 15 | `robustness-agent\src\robustness_agent\signals\preflight.py` |
| 14 | `robustness-agent\src\robustness_agent\decision\action_ladder.py` |
| 14 | `robustness-agent\src\robustness_agent\decision\policy_aware.py` |
| 13 | `robustness-agent\src\robustness_agent\providers\local.py` |
| 13 | `robustness-agent\src\robustness_agent\providers\robust.py` |
| 13 | `robustness-agent\src\robustness_agent\role\prompt_inputs.py` |
| 13 | `robustness-agent\src\robustness_agent\sources\base.py` |
| 12 | `robustness-agent\src\robustness_agent\state_store.py` |
| 11 | `robustness-agent\src\robustness_agent\role\envelope.py` |
| 10 | `robustness-agent\src\robustness_agent\signals\decision_audit.py` |
| 9 | `robustness-agent\src\robustness_agent\providers\hybrid.py` |
| 9 | `robustness-agent\src\robustness_agent\signals\event.py` |
| 9 | `robustness-agent\src\robustness_agent\signals\gpu_leak.py` |
| 9 | `robustness-agent\src\robustness_agent\signals\local_health.py` |
| 9 | `robustness-agent\src\robustness_agent\signals\repeated_payload.py` |
| 8 | `robustness-agent\src\robustness_agent\main.py` |
| 8 | `robustness-agent\src\robustness_agent\role\reactor.py` |
| 8 | `robustness-agent\src\robustness_agent\runtime\cli.py` |
| 7 | `robustness-agent\src\robustness_agent\config.py` |
| 7 | `robustness-agent\src\robustness_agent\rca.py` |
| 7 | `robustness-agent\src\robustness_agent\findings\sink.py` |
| 7 | `robustness-agent\src\robustness_agent\monitors\gpu_monitor.py` |
| 7 | `robustness-agent\src\robustness_agent\monitors\process_monitor.py` |
| 7 | `robustness-agent\src\robustness_agent\signals\external_deps.py` |
| 7 | `robustness-agent\src\robustness_agent\signals\kernel_pipeline.py` |
| 7 | `robustness-agent\src\robustness_agent\signals\progress.py` |
| 7 | `robustness-agent\src\robustness_agent\signals\state_integrity.py` |
| 6 | `robustness-agent\src\robustness_agent\monitors\log_tailer.py` |
| 6 | `robustness-agent\src\robustness_agent\signals\aiter_jit.py` |
| 6 | `robustness-agent\src\robustness_agent\signals\budget.py` |
| 5 | `robustness-agent\src\robustness_agent\factory.py` |
| 5 | `robustness-agent\src\robustness_agent\checks\event_check.py` |
| 5 | `robustness-agent\src\robustness_agent\providers\base.py` |
| 5 | `robustness-agent\src\robustness_agent\signals\cluster_fault.py` |
| 5 | `robustness-agent\src\robustness_agent\signals\critic_health.py` |
| 5 | `robustness-agent\src\robustness_agent\sources\cluster_decoder.py` |
| 4 | `robustness-agent\src\robustness_agent\agent.py` |
| 4 | `robustness-agent\src\robustness_agent\models.py` |
| 4 | `robustness-agent\src\robustness_agent\monitors\server_health.py` |
| 4 | `robustness-agent\src\robustness_agent\signals\health.py` |
| 3 | `robustness-agent\src\robustness_agent\checks\disk_check.py` |
| 3 | `robustness-agent\src\robustness_agent\checks\stall_check.py` |
| 3 | `robustness-agent\src\robustness_agent\signals\classifier.py` |
| 3 | `robustness-agent\src\robustness_agent\signals\stall.py` |
| 3 | `robustness-agent\src\robustness_agent\signals\symptom.py` |
| 2 | `robustness-agent\src\robustness_agent\_payload_aliases.py` |
| 2 | `robustness-agent\src\robustness_agent\signals\crash.py` |
| 1 | `robustness-agent\src\robustness_agent\checks\__init__.py` |
| 1 | `robustness-agent\src\robustness_agent\monitors\__init__.py` |
| 1 | `robustness-agent\src\robustness_agent\providers\__init__.py` |

### `ci/` - 363 gaps in 21 files

| Gaps | File |
|---|---|
| 74 | `ci\optimize_submit.py` |
| 55 | `ci\import_session_breakdown.py` |
| 26 | `ci\pr_submitter.py` |
| 24 | `ci\generate_hf_matrix.py` |
| 20 | `ci\artifact_normalizer.py` |
| 16 | `ci\tiny_submit.py` |
| 15 | `ci\claw_client.py` |
| 15 | `ci\inferenceX_parser.py` |
| 14 | `ci\build_summary.py` |
| 13 | `ci\build_candidates.py` |
| 12 | `ci\orchestrator.py` |
| 12 | `ci\transform_to_session_summary_v2.py` |
| 11 | `ci\prewarm_models.py` |
| 11 | `ci\send_webhook.py` |
| 10 | `ci\report_generator.py` |
| 8 | `ci\build_production_hf_pool.py` |
| 8 | `ci\progress.py` |
| 7 | `ci\post_perf_runs.py` |
| 5 | `ci\publish_artifacts.py` |
| 4 | `ci\publish_results.py` |
| 3 | `ci\generate_matrix.py` |

### `kernel-agent/` - 276 gaps in 14 files

| Gaps | File |
|---|---|
| 74 | `kernel-agent\tools\tracelens_analysis.py` |
| 68 | `kernel-agent\tools\kernel_optimization.py` |
| 27 | `kernel-agent\tools\harness_generator.py` |
| 27 | `kernel-agent\tools\tracelens_skill_runner.py` |
| 25 | `kernel-agent\tools\apply_kernel_patch.py` |
| 11 | `kernel-agent\tools\parallel_e2e_runner.py` |
| 8 | `kernel-agent\tools\gemm_tuning.py` |
| 8 | `kernel-agent\tools\backends\geak_submit.py` |
| 8 | `kernel-agent\tools\backends\oob_submit.py` |
| 7 | `kernel-agent\skills\unittest\validate_harness.py` |
| 5 | `kernel-agent\tools\backends\ray_runtime.py` |
| 4 | `kernel-agent\tools\geak_prompt_patcher.py` |
| 2 | `kernel-agent\tools\_collective_names.py` |
| 2 | `kernel-agent\tools\_payload_aliases.py` |

### `critic-agent/` - 208 gaps in 21 files

| Gaps | File |
|---|---|
| 28 | `critic-agent\runtime\session_memory.py` |
| 20 | `critic-agent\runtime\decision_reviewer.py` |
| 17 | `critic-agent\runtime\in_memory_kb_client.py` |
| 16 | `critic-agent\runtime\cli.py` |
| 16 | `critic-agent\runtime\kb_writer.py` |
| 16 | `critic-agent\runtime\web_tools\fetch_client.py` |
| 11 | `critic-agent\runtime\kb_client.py` |
| 10 | `critic-agent\runtime\metrics.py` |
| 9 | `critic-agent\runtime\inbox_parser.py` |
| 9 | `critic-agent\runtime\intent_envelope.py` |
| 8 | `critic-agent\runtime\dead_letter.py` |
| 8 | `critic-agent\runtime\web_tools\config.py` |
| 8 | `critic-agent\runtime\web_tools\search_client.py` |
| 7 | `critic-agent\runtime\request_models.py` |
| 6 | `critic-agent\runtime\web_tools\providers.py` |
| 5 | `critic-agent\runtime\scope_builder.py` |
| 4 | `critic-agent\runtime\web_tools\tool_schemas.py` |
| 3 | `critic-agent\runtime\importance_mapping.py` |
| 3 | `critic-agent\runtime\slugify.py` |
| 2 | `critic-agent\runtime\category_mapping.py` |
| 2 | `critic-agent\runtime\web_tools\__init__.py` |

### `framework-agent/` - 126 gaps in 17 files

| Gaps | File |
|---|---|
| 26 | `framework-agent\src\framework_agent\explorer.py` |
| 18 | `framework-agent\src\framework_agent\kb.py` |
| 15 | `framework-agent\src\framework_agent\models.py` |
| 13 | `framework-agent\src\framework_agent\runtime\cli.py` |
| 12 | `framework-agent\src\framework_agent\sources\primus_cortex.py` |
| 10 | `framework-agent\src\framework_agent\isolation.py` |
| 7 | `framework-agent\src\framework_agent\sources\__init__.py` |
| 6 | `framework-agent\src\framework_agent\runtime\tools_api.py` |
| 5 | `framework-agent\src\framework_agent\logging_setup.py` |
| 3 | `framework-agent\src\framework_agent\keywords.py` |
| 2 | `framework-agent\src\framework_agent\decision.py` |
| 2 | `framework-agent\src\framework_agent\shell.py` |
| 2 | `framework-agent\src\framework_agent\sources\github.py` |
| 2 | `framework-agent\src\framework_agent\sources\_shared.py` |
| 1 | `framework-agent\examples\bench_rmsnorm.py` |
| 1 | `framework-agent\examples\check_accuracy_rmsnorm.py` |
| 1 | `framework-agent\src\framework_agent\repo_map.py` |

### `slides/` - 3 gaps in 3 files

| Gaps | File |
|---|---|
| 1 | `slides\plot_deepseek_r1.py` |
| 1 | `slides\plot_glm5_breakdown.py` |
| 1 | `slides\plot_loop_diagram.py` |
