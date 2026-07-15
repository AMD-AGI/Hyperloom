# Fallback 处理裁决计划 (fallback-judge)

> 目标:对 `fallback-survey-all.json` 编目的 **768 处 fallback** 逐项裁决,产出一份可执行的
> **代码 cutoff 计划**。允许高风险改动、允许删测试。裁决口径由 18 个只读 sub agent(每 unit 一个,
> 外加 `orch/phases` 因首个 agent 误计数而重跑一次)逐项读源码判定,并对若干跨轮冲突做了直接源码复核。
>
> 生成方式:768 项 → 18 个 unit 分片(`/tmp/fbunit/*.json`)→ 每 unit 一个 sub agent 按统一策略判定
> → 主 agent 复核冲突 + 汇总。**共 19 个 judge agent。**

---

## 0. 一句话结论(先读这个)

**768 处 fallback 里,约 ~670–680 处是"真·运行期韧性"必须保留(KEEP);真正可动的只有 ~90 处,
分布高度集中在几个遗留键簇 + 一批已死的防御性守卫。**

- 直接可删源代码:**约 300–500 行**(不含测试)。这远小于 `lean-3` 的 ~11k 行,原因是
  **`lean-3` 已经把"死函数/冗余封装"这条轴清理过了**;fallback 这条轴剩下的多是**表达式内的单臂**
  (`or legacy_key`、`except AttributeError`、`getattr(..., default)`),单点 1–15 行。
- 因此本计划的价值 **不是"砍大量行数",而是"消除遗留兼容面 + 让隐藏 bug 浮现 + 降分支复杂度"**。
  如果预期是"再砍一万行",fallback 这条轴给不了;想继续砍大体量,得回到 `lean-3` 的死子系统轴。
- 最高杠杆的单件事:**统一 `args/envs`→`extra_server_args/extra_envs` 与 `framework`→`framework_name`
  两个遗留键簇的写入方**(约 4–5 个写入点),一次改动即可让 ~12 个读取回退臂变死可删,横跨
  `orch/knowledge`、`orch/phases`、`orch/loop`、`orch/state`。

---

## 1. 处理策略:不同种类 fallback 怎么处理(回应需求 1)

### 1.1 四种裁决(verdict)

| verdict | 含义 | 何时用 |
|---|---|---|
| **CUT** | 直接删除该回退臂/整函数。live 路径行为不变。 | 分支不可达(死代码)、纯冗余封装、常量恒真/恒假的三元。 |
| **CUT+WRITER** | 遗留键读取回退,唯一触发源是遗留写入方或旧持久化数据。**先改写入方发规范键,再删读取臂**,接受 back-compat 破坏。 | `legacy_key` 且写入侧可收敛;旧 session/旧 recipe.json 兼容明确放弃。 |
| **SIMPLIFY** | 分支可达但冗余:收拢到规范 helper / 改严格访问 / 用 `raise` 替代静默默认。降 LOC/分支但不牺牲健壮性。 | `getattr(封闭字段,默认)`、封闭枚举的穷举 catch-all、单调用方薄封装、误导性注释。 |
| **KEEP** | 承重的运行期韧性。**不动**。 | 外部 I/O、子进程、网络、文件系统、解析外部/LLM/持久化数据、真正可选的 config/env、运行时可选的产品模式(mock 后端等)。 |

### 1.2 按类别的默认处理(768 项的类别分布见 §2)

| category | 计数 | 默认处理 | 关键判据 |
|---|---:|---|---|
| `legacy_key` | 88 | **CUT+WRITER** | 全库 grep 写入方;若唯一写入方是旧数据/旧写入器 → 改写入方后删读取臂。**例外**:该键仍是"当前双键协议"或读取的是**外部/无法控制的语料**(gbrain 中央库、litellm 返回、operator 部署 env)→ KEEP。 |
| `resume_downgrade` | 6 | **CUT** | `claude-agent-sdk>=0.2.110` 保证 `resume=`;降级分支不可满足。`_resume_downgraded` 是恒 False 字段。 |
| `sdk_graceful` | 48 | **CUT** 若 pin floor 保证该 kwarg/特性;否则 **KEEP** | Claude SDK(resume/effort/thinking)已由 pin 保证 → CUT;**Langfuse v2/v3/v4 API 漂移、openai 可选依赖、psutil/torch/rocprof 可选** → KEEP(未 pin 单一大版本)。 |
| `mock_downgrade` | 3 | **KEEP** 若运行时可选产品模式;否则 CUT | critic/robustness mock 由 CLI 选择;robustness `--nodes>=2` 自动降 mock 是真实产品行为 → KEEP。 |
| `try_except` | 237 | 细分 | Py<3.9/版本守卫=**CUT**;硬依赖 ImportError(PyYAML/openai/claude-sdk)=**CUT**;可选依赖 ImportError=**KEEP**;外部 I/O/子进程/网络/解析外部数据=**KEEP**;封闭输入的纯内部逻辑防御 except=**SIMPLIFY/CUT**。 |
| `if_else_default` | 99 | 若在**封闭枚举/集合**上可证穷举 → **SIMPLIFY**(改 `raise`/`assert`)或 CUT;否则 **KEEP** | 大量是"LLM 排序失败回退 index-0""git-tree vs wheel 结构分支""env 逃生舱"——这些不是穷举默认,是真实双路径 → KEEP。 |
| `default_when_missing` | 286 | 细分 | **总是存在的 dataclass 字段/属性**(`default_factory`,`from_dict` 走 `cls(**filtered)`)→ **SIMPLIFY**(直读)/CUT;`getattr` 守卫恒存在属性=**CUT**;`dict.get` 读**可选/外部/持久化**键=**KEEP**。 |
| `other` | 1 | 逐项 | — |

> **核心原则**:fallback ≠ 死代码。这条轴的绝大多数(尤其 `try_except`/`default_when_missing`)是
> Hyperloom 在**真实失败边界**(ROCm 缺失、子进程超时、LLM 乱序输出、跨 schema 归档 session、
> AMD 多网关 env)上的**有意防御**。KEEP 是默认,CUT 需要证明 live 路径不受影响。

---

## 2. 全局统计(768 项)

### 2.1 类别 × 可达性(编目态)

| category | reachable | likely_unreachable | unreachable | 合计 |
|---|---:|---:|---:|---:|
| default_when_missing | 273 | 9 | 4 | 286 |
| try_except | 229 | 5 | 3 | 237 |
| if_else_default | 94 | 2 | 3 | 99 |
| legacy_key | 67 | 18 | 3 | 88 |
| sdk_graceful | 43 | 2 | 3 | 48 |
| resume_downgrade | 5 | 0 | 1 | 6 |
| mock_downgrade | 3 | 0 | 0 | 3 |
| other | 1 | 0 | 0 | 1 |

### 2.2 裁决汇总(本轮 19 个 agent 判定后)

| verdict | 估计项数 | 说明 |
|---|---:|---|
| **KEEP** | ~675 | 运行期韧性,见 §6 归类。 |
| **CUT** | ~28 | 死分支/死函数/死守卫,机械删除。 |
| **CUT+WRITER** | ~22 | 遗留键读取臂,需先改 ~5 个写入点。 |
| **SIMPLIFY** | ~30 | getattr→直读、穷举→raise、封装收拢、注释订正。 |
| **DISPUTED/需签字** | ~6 | 见 §5(Tier 4)。 |

> 数字为"可动项",单点体量小;真实源代码删除量约 **300–500 行 + 相应测试**。

---

## 3. 分层执行批次

按风险从低到高。每个 Tier = 一个/一组提交,可独立回退。每批后跑
`ruff check . && pytest -m "not critic_agent_e2e and not robustness_agent_e2e"`。

### Tier 0 — 零风险机械删除(死守卫 / 常量 / 陈旧文档串)

无 live 路径依赖,不需要改写入方。

| 位置 | 类别 | 动作 | ~LOC |
|---|---|---|---|
| `orchestrator/policy/gate.py:530` `_resolved_within` | try_except | 删 `except AttributeError`(Py<3.9 `is_relative_to`)整块,保留 `return v==r or v.is_relative_to(r)` | 6 |
| `orchestrator/policy/gate.py:2343` `_path_under_session` | try_except | 同上(同构第二处) | 6 |
| `orchestrator/state/shared_state.py:1870` `record_action_attempt` | default_when_missing | 删 `if not hasattr(...): return None`——五个 `_AUDIT_ACTIONS` 字段均 `default_factory` 恒存在 | 3 |
| `orchestrator/roles/claude.py:211` `_resume_downgraded` 字段 + `:442` metadata 发射 | resume_downgrade | 删字段 + metadata 键;连带 `trace/llm_trace.py` 的 `_VALID_KEYS`/字段/注释/`to_row`/`from_metadata` + 测试断言 | 16 |
| `orchestrator/roles/claude.py:595` docstring | sdk_graceful | 删"degrade via `_instantiate_options`"陈旧句(该降级代码已在 13fba44a 删除) | 1 |
| `orchestrator/loop/sub_agent_runner.py:10` docstring | sdk_graceful | 删"LLM external sub-agent fallback (backend.run())"陈旧描述(代码中无此分支) | 3 |
| `inference_optimizer/breakdown/collectors/sessions.py:217` `_load_yaml_dict_safe` | try_except | 删 `except ImportError`(PyYAML 是硬依赖),保留 `OSError/YAMLError` 外层 | 6 |
| `inference_optimizer/breakdown/collectors/sessions.py:413` `_extract_invocation_env` | try_except | 同上(同构第二处) | 6 |
| `agents/quantization/driver/retry.py:329` `DEFAULT_QUARK_GIT_URL` 三元 else | if_else_default | 删 else 分支(常量恒真) | 1 |
| `agents/quantization/driver/runner.py:338` `else: raise env_exc` | try_except | 删(`env` 恒在 kwargs,line 313 无条件设置且从不 pop) | 2 |
| `orchestrator/kernel/request_handlers.py:3208` `removed_oob` 集合替换块 | default_when_missing | 删(无代码把 claude/codex/cursor 写入 backend order;未知 token 已由 `return []` 兜底) | 3 |
| `agents/critic/runtime/kb_client.py:279` `if payload else {}` | default_when_missing | 删死分支(上一行 `or "{}"` 已保证非空) | 1 |
| `agents/kernel/tools/tracelens_skill_runner.py:939` `globals().get("DEFAULT_MODEL", …)` | — | SIMPLIFY→`resolved_model = resolved_model or "claude-opus-4-8"`(模块从不定义 DEFAULT_MODEL,查全局恒失败) | 2 |
| `agents/kernel/skills/unittest/validate_harness.py:298` | default_when_missing | SIMPLIFY→`report['static'].get('ok', True)`(`--all` 下外层 `.get('static',{})` 不可达) | 1 |

**Tier 0 小计:约 60 LOC + 少量测试断言。** 无写入方迁移,风险最低。

### Tier 1 — 死整函数 / 死分支(有调用图证据,不需改写入方)

| 位置 | 动作 | ~LOC | 耦合 |
|---|---|---|---|
| `inference_optimizer/baseline_comparison/inferencex_client.py:171` `fetch_rows` | **CUT 整函数**(无生产调用方;`analyze()` 读 `competitor_target.json`)。删 `__all__` 与 `baseline_comparison/__init__.py` 再导出。**保留 `InferenceXFetchError` + `_fetch_raw`**(`reference_script` 用)。 | ~23 源 + ~99 测试 | 删 `test_fetch_rows_*`(两个测试文件) |
| `agents/robustness/decision/rca_engine.py:585` `_safe_extra_evidence` | **CUT** `except`+`not isinstance(items,list)` 两臂 + `extra_evidence_provider` 字段(工厂从不注入,`provider is None` 恒短路) | ~7 | 删注入该 provider 的测试 |
| `agents/robustness/sources/local_probe.py:393` 遗留 events SELECT | **CUT** 第二条 `SELECT id, agent, timestamp …`(现 schema 无此列,第一条恒成功) | ~5 | 删遗留 schema 测试 |
| `orchestrator/knowledge/cortex_t0.py` `_row_best_config_source` (`:78`) | **CUT** `_field_sources`/`_sources` 的 str/list/`[0]` 三个 provenance 臂 → 函数塌成 `return ""`(PR #757 删了唯一写入方 composite Cortex 后端) | ~15 | — |
| `orchestrator/knowledge/cortex_t0.py` `_warm_recipe_source` (`:139`) | **CUT** donor + own-source 两分支 → 塌成 `return "gbrain" if _remote_is_gbrain(kb) else "cortex-kb"`;`row/config_donor/config_donor_tier` 参数随之删 | ~10 | 依赖上一条 |
| `inference_optimizer/cli/__init__.py:24` `shutil`/`subprocess` 再导出 + 注释 | **CUT**(生产无 `cli.shutil.`/`cli.subprocess.` 用法;测试 patch 的是 `preflight.shutil`) | ~6 | 无(测试已指向子模块) |
| `inference_optimizer/cli/backends.py:232-239` critic anthropic-only 降级臂 | **CUT**(唯一调用方在调用前 `sys.exit(2)`,`critic_agent_root` 到此恒非 None) | ~5 | 无专属测试 |
| `inference_optimizer/cli/backends.py:241-242` + `:257-258` critic/robustness root 守卫 | **CUT**(同上,调用方预校验后 exit) | ~4 | 无专属测试 |
| `orchestrator/loop/result_recorder.py:843` `e.get("source_file")` 臂 | **CUT**(主 agent 直接复核:`kernel_opt_attempts` entry 只写 `last_source_file`;裸 `source_file` 只进 `last_kernel_opt`) | 1 | — |
| `orchestrator/loop/result_recorder.py:852` `e.get("ts")` 臂 | **CUT**(同上复核:entry 只写 `last_ts`;裸 `ts` 只进 `last_kernel_opt`/`history`。**订正**:本轮 loop agent 曾误判为 live,主 agent 读码确认为死) | 1 |  |

**Tier 1 小计:约 75 源 LOC + ~100 测试 LOC。**

### Tier 2 — 遗留键簇(CUT+WRITER;先改写入方,再删读取臂)★ 最高杠杆

这是本计划**唯一成规模**的部分。**必须先落写入方迁移,再删读取臂,否则破坏 live 写路径。**

#### 簇 A:`args/envs` → `extra_server_args/extra_envs`

**写入方迁移(先做):**
1. `orchestrator/loop/result_recorder.py:875` `_build_recipe_attrs_from_state` 的按键复制循环:从 `("extra_envs","args","envs","name","tput","accuracy")` 删掉 `"args"`、`"envs"`。
2. `orchestrator/knowledge/recipe_kb/gbrain_ingest.py:157-158` `_NON_ENV_BEST_CONFIG_KEYS`/copy keys:只写 `extra_envs`/`extra_server_args`。

**随后删除的读取臂(CUT):**
- `orchestrator/knowledge/cortex_t0.py:187` `_recipe_is_actionable` `or best_config.get("args")`
- `orchestrator/knowledge/cortex_t0.py:188/211-212` `_config_replay_args_envs` `or best_config.get("args"/"envs")`(保留尾部 `or {}` 与 `not isinstance(envs,Mapping)` 守卫)
- `orchestrator/knowledge/recipe_kb/gbrain_ingest.py:205` `_best_config_split` `nested = best_config.get("envs")`(**保留 flat-sibling else 分支**,那是外部 gbrain 页的平铺 env 处理,live)
- `orchestrator/phases/prelude.py:102` `row.get("args")`
- `orchestrator/phases/prelude.py:291-292` `best_config.get("args"/"envs")`(保留识别分支 `:287`)

> `orchestrator/state/shared_state.py:1092` `winners_history` 的 `entry.get("extra_args") or entry.get("extra_server_args")`:延后 CUT+WRITER——现写入方 `explore.py:1411` 已发 `extra_args`,`extra_server_args` 臂仅旧持久化 row 触发,声明旧 session EOL 后再删。

#### 簇 B:`framework` → `framework_name`

**写入方迁移(先做):**
1. `inference_optimizer/cli/kb.py:191` 把传入的 `"framework"` 键改成 `"framework_name"`(唯一还用旧键的 live 调用点;`phases/machine.py:61` 已用规范键)。
2. (如存在)`orchestrator/loop/proposals.py` 的 `_reserved` 集合补上 `"framework"`,使旧磁盘 row 的裸 `framework` 不再幸存进 `extras`(与 `cortex_t0.py:1020` 对齐)。

**随后删除的读取臂(CUT):**
- `orchestrator/knowledge/recipe_kb/dispatcher.py:200` `_v2_to_arbor` `or labels.get("framework")`
- `orchestrator/knowledge/recipe_kb/local_store.py:878` `_matches_labels` framework_name/framework
- `orchestrator/knowledge/recipe_kb/schema.py:452` `Recipe.from_dict` `or d.get("framework")`
- `orchestrator/knowledge/recipe_kb/gbrain_ingest.py:271` `recipe_to_page` `or recipe.get("framework")`
- `orchestrator/knowledge/cortex_t0.py:912` framework 读取臂

> **KEEP 例外**:`orchestrator/knowledge/recipe_kb/gbrain_remote_client.py:498`——读**外部中央 gbrain 语料**,其他 operator 写的 pre-rename 页仍会遇到,不可删。

#### 簇 C:critic `environment` 键 & `verdicts` 别名

- `agents/critic/runtime/cli.py:162/200/228` `or packet.get("environment")`:**CUT**(全库无生产者写 `environment` 顶层键)。保留 `context` 主路径与 `or {}`。
- `agents/critic/runtime/decision_reviewer.py:1091` `review.get("verdicts")`:**CUT**(所有生产者发 `review_verdicts`)。**保留** `:1092-1093` 非 list 校验(有独立 live 直达路径)。
- `agents/critic/runtime/decision_reviewer.py:1099,1159` `or advisory.get("text")`:**CUT**(无生产者,子分支)。

#### 簇 D:`extra_sglang_args` 最后遗留岛(kernel-agent shim)

- `agents/kernel/tools/_payload_aliases.py:94` `read_extra_server_args` LEGACY_KEY 分支:**CUT+WRITER**。删 `LEGACY_KEY`/`_DEPRECATION_MESSAGE`/`import warnings`/`if LEGACY_KEY in payload:` 块;`kernel_optimization.py:1484` 内联为 `candidate.get("extra_server_args","")`;删 `test_payload_aliases_shim.py` 两个断言;**删/清空 `test_no_legacy_writer_sites.py`**(静态守卫失去意义)。接受放弃外部/operator 手写 payload 的旧键兼容。

#### 簇 E:小遗留键(各自独立)

- `orchestrator/kernel/attempt_summary.py:304` `elapsed = a.get("elapsed_sec")`:**CUT+WRITER**(所有 live 写入方发 `elapsed_s`)。
- `agents/framework/models.py:500` `or raw.get("model")` + 写入方 `agents/framework/runtime/cli.py:438` 的冗余 `"model":` 键:**CUT+WRITER**。
- `agents/framework/explorer.py:577` & `tools_api.py:237` `_metric_float` 的 `"tput"` 键:**CUT**(framework agent 消费的 `benchmark.json` 无 tput;子分支)。

#### 簇 F:multi_node 遗留 `/tmp/multi_node_state.json`

- `inference_optimizer/multi_node/cli.py:163` `_load_state` 遗留读 + `multi_node/.../state_paths.py` 的 `resolve_state_file` tier-3 + `legacy_state_file()` + `bind_state_file_to_session` 迁移块 + `orchestrator/actions/executors/_multi_node_env.py:55` 的并行遗留读:**CUT+WRITER**(`bind_state_file_to_session` 一次性迁移已完成,orchestrated run 恒设 `$MULTI_NODE_STATE_FILE`)。tier-3 改 `raise RuntimeError(...)`。删 `test_state_paths.py::test_bind_state_file_migrates_legacy`。

**Tier 2 小计:约 120–160 源 LOC(读取臂 + 迁移块)+ 相关测试。写入方改动约 5–6 点。**

### Tier 3 — SIMPLIFY(降复杂度,不牺牲健壮性)

| 位置 | 动作 |
|---|---|
| `orchestrator/phases/machine.py:96/106/123` `_kernel_enabled`/`_explore_enabled`/`_advance_phase_if_needed` | `getattr(state, X, default)` → 直读 dataclass 字段(`kernel_enabled`/`explore_enabled`/`framework_agent_phase_enabled` 均为恒存在字段,getattr 默认值与字段默认一致,直读无行为变化) |
| `agents/robustness/signals/local_health.py:407/450` | 删 `getattr(...,None)`+`isinstance(...,dict)` 死臂,**保留** `if not ray_info/fd_info` 空 dict 检查(live 且有测试) |
| `agents/robustness/decision/rca_engine.py:291`、`role/reactor.py:168` | `getattr(self,'_current_tick_id',-1)`/`getattr(ctx.shared_state,'tick',0)` → dataclass 默认 + 直读 |
| `agents/quantization/driver/assessment.py:524` `derive_status` catch-all | 穷举 OutcomeId → 改 `raise AssertionError(f"unhandled outcome: {final}")`(fail-loud,防未来新增枚举漏配静默) |
| `agents/quantization/driver/retry.py:271` `_decide_next_step` 尾部 | ASK 分区穷举 → 改 `raise AssertionError(...)` |
| `agents/quantization/driver/assessment.py:452` `build_assessment` 空 attempts 守卫 | → `raise ValueError("attempts must be non-empty")`(生产恒非空;静默替换掩盖调用方 bug) |
| `agents/quantization/driver/result_collector.py:145` `_read_text` 双 except | 合并为 `except OSError`(`FileNotFoundError ⊂ OSError`) |
| `agents/critic/runtime/decision_reviewer.py:889` int env 解析、`kb_assess_client.py:83` float env 解析 | try/except → `int(os.environ.get(...) or 5)` / `float(... or DEFAULT)` |
| `agents/critic/runtime/decision_reviewer.py:1017` `_CLASS_RANK.get(cls, default)` | → `_CLASS_RANK[cls]`(封闭集,分类器只返回三键) |
| `orchestrator/prompts/prompt_builder.py:265` `_filter_actions` 未知名静默 skip | → `assert meta is not None`(enabled_actions 来自封闭 `FULL_ENABLED_ACTIONS`) |
| `orchestrator/knowledge/recipe_kb/dispatcher.py:433` `_remote_label` 默认返类名 | → fail-loud `raise AssertionError`(未知后端应在接线期暴露) |
| `orchestrator/knowledge/recipe_kb/canonical_id.py:86-88` docstring | 删"接受 6 段 legacy id"陈旧句 + 订正返回维度顺序(与 `:124` 解包不一致) |
| `orchestrator/roles/critic_agent.py:253-256` isinstance 重塑 | 收成 `dict(... or {})` 一行 |
| `orchestrator/policy/gate.py:192` `detect_gpu_count` 冗余 `except Exception` | 并入首个 except 元组(加 `subprocess.TimeoutExpired`),删裸 catch-all |
| `orchestrator/loop/intent_router.py:292` `getattr(pending,"task_id",None)` | 删死臂 → `pa_params.get("task_id")`(PendingProposal 无 task_id 字段) |
| `orchestrator/loop/coordinator.py:946` 宽 `except Exception` | 收窄为 `(FileNotFoundError, yaml.YAMLError)` |
| `agents/framework/gbrain_page_client.py:198`、`sources/__init__.py:274` | 冗余 `or ""` / `or ("open",)`(上游已保证非空)→ 去掉 |
| `orchestrator/actions/executors/_grid_server_args.py:48` 未知 framework 静默映射 SGLang | → `raise ValueError(f"unknown framework: {name!r}")` |
| `orchestrator/actions/executors/baseline.py:1503` `except: return True` (double_run) | → `return False` + 收窄 `except OSError`(`True` 会静默翻倍 bench 成本) |
| `orchestrator/actions/executors/baseline.py:866-910`、`_workload_envs.py:372` | 抽共享 `_load_shared_state()` helper;订正"legacy callers"误导注释 |
| `orchestrator/trace/langfuse_mapping.py:96` `langfuse_session_id` | 塌成 `correlation_seed` 别名(函数体无变换) |
| `inference_optimizer/multi_node/*` `_sanitize_extra_env` null 守卫、`_dynamo_all_gpu_ips` pd_mode `.get`、`main` 异常子串分类器 | 分别 1 行 null 守卫 / `.get` 默认 / 换 typed `MultiNodeConfigError` |
| `orchestrator/phases` 子分支:`machine_state.py:1434` 未知 stop_reason 透传、`kernel_work_pending` 双 `except Exception`(dataclass 字段恒存在)、`close.py:462` isinstance、`explore.py:1174` 重复 if 块合并 | SIMPLIFY |

**Tier 3 小计:约 60–100 LOC 净减 + 若干 fail-loud 化(把静默默认换成 `raise`/`assert`)。**

> **关于 fail-loud**:穷举 catch-all(quantization 的 `derive_status`/`_decide_next_step`、prompt_builder
> 的 `_filter_actions`)当前是"未来新增枚举/action 漏配时静默兜底"。用户要砍冗余——这里推荐
> **改 `raise` 而非直接删**:既去掉静默分支,又保留对未来疏漏的爆炸式暴露。若坚持纯删,风险是
> 未来疏漏变静默错误。

### Tier 4 — 需人工签字 / 有争议(不要盲删)

| 位置 | 问题 | 建议 |
|---|---|---|
| `agents/quantization/driver/result_collector.py:270` `_scan_hypothesis_attempts` `except FileNotFoundError` | **跨轮冲突**:本轮 quantization agent 判 SIMPLIFY(mkdir 保证目录存在);上一轮对抗式复核判 **LIVE**(LLM agent 持 Bash,可在 mkdir 与 scan 之间 `rm -rf` 自己的 workspace)。这是 `collect_artifacts` **唯一**把缺目录转成 graceful 结果的守卫。 | **KEEP**(保守;不可信 Bash 边界)。若签字接受"agent 删 workspace 则崩"再删。 |
| `orchestrator/phases/machine_state.py:366` `STOP_REASON_VOCAB` 遗留哨兵 | phases agent 判 KEEP(`coordinator.py:1910`/`writeback.py:432/444` 仍写这些值);`lean-3` 附录曾列为高风险 legacy。 | **KEEP**,除非确认无写入方且放弃旧 session 恢复。 |
| `io/breakdown` 的 `legacy_key`/`default_when_missing`(除已列的 PyYAML/invocations 外) | breakdown 是**离线历史渲染器**,读 WekaFS 上跨 2–3 代 schema 的归档 session;绝大多数遗留读取是**特性**(渲染旧 session)。 | **KEEP**。`telemetry.py:282` lane_capacity 回退明确 LIVE(v1 归档 DB 无该表)。 |
| `orchestrator/state/shared_state.py:1092` `winners_history` 的 `extra_server_args` 臂 | 依赖旧持久化 row。 | 延后:声明旧 session EOL 后并入簇 A。 |
| `specialists/subprocess_.py:378` `wall_budget_sec`、`:736` intent-envelope 解包 | trace agent 判 CUT+WRITER,但需先确认无写入方/无旧 session 产该形状(其 `log.info` 可作探针)。 | 先 grep/看 session 数据确认,再删。 |
| PR Monitor / Cortex KB 遗留 env 源(`PR_MONITOR_URL`/`CORTEX_KB_URL` 的 `or` 链) | `pr.md` 显示已在 55d229a1 收敛到 flag;若仍有 `or legacy_env`,属可删,但涉及部署契约。 | 按 `pr.md` 契约核对后 CUT。 |
| **非 fallback 但顺带发现的缺陷**:`inference_optimizer/breakdown/reporters/_renderers/invocations.py` `forge_invocations` 无 renderer 注册 | `forge_invocations` 是 schema/exporter/recorder 一等字段,但 reporter 只注册 `geak_invocations`,渲染期静默丢弃 forge 数据。 | **修 bug(补 `render_forge`)**,不是 fallback 删除项。 |

### Tier 5 — SDK 漂移(明确 KEEP,勿碰)

- `orchestrator/trace/langfuse_emitter.py` 全部 `sdk_graceful`(`:142/149/256/405` 等):Langfuse v2/v3/v4
  API 漂移兼容 + 可选依赖 `[trace]` 守卫。**未 pin 单一大版本前删除会在任何 langfuse 升级时静默断链。**
- `roles/claude.py` 之外的可选 SDK(openai proposal_scorer、psutil、torch/triton/rocprof)守卫:KEEP。

---

## 4. 逐 unit 可动项明细(768 项的 non-KEEP 抽取)

> KEEP 项不逐条列(占 ~88%,归类见 §6)。下表只列 CUT/CUT+WRITER/SIMPLIFY/DISPUTED。

| unit | 项数 | 可动 | 主要可动项 |
|---|---:|---:|---|
| orch/kernel | 68 | 2 | attempt_summary `elapsed_sec`(C+W)、request_handlers `removed_oob`(CUT) |
| agents/critic | 62 | 10 | cli `environment`×3、`verdicts`、`advisory.text`×2(CUT/C+W);int/float/`_CLASS_RANK`/dead-if(SIMPLIFY) |
| agents/framework | 61 | 5 | models `model`(C+W)、`tput`×2(CUT)、gbrain_page/pr_states(SIMPLIFY) |
| orch/trace+specialists+... | 59 | 5 | prompt_builder `_filter_actions`、subprocess_ `wall_budget`/envelope(C+W,Tier4)、langfuse_session_id(SIMPLIFY) |
| orch/knowledge | 58 | 14 | 簇 A + 簇 B 读取臂、provenance 死臂、`_remote_label`/canonical_id docstring(SIMPLIFY) |
| io/cli | 53 | 3 | cli `shutil/subprocess`、backends 死降级臂 + 死守卫(CUT) |
| io/multi_node | 51 | 5 | 簇 F 遗留 state(C+W)、3× SIMPLIFY |
| agents/robustness | 46 | 7 | rca `_safe_extra_evidence`、legacy SELECT、`_probe_external_mounts`(CUT);4× getattr(SIMPLIFY) |
| orch/phases | 45 | 6 | prelude args/envs×2(C+W,簇 A);machine `_kernel/_explore/_advance`(SIMPLIFY,getattr→直读)、explore 合并 |
| io/rest+common | 39 | 1 | inferencex `fetch_rows` 整函数(CUT) |
| orch/roles | 37 | 3 | `_resume_downgraded`(CUT,连带 llm_trace)、critic_agent reshape、claude docstring |
| agents/quantization | 36 | 7 | QUARK_URL/env-raise(CUT);derive_status/_decide_next_step/build_assessment/_read_text(SIMPLIFY);result_collector(DISPUTED) |
| orch/actions | 34 | 6 | profile(已删)、`_multi_node_env` 遗留 state(C+W,簇 F)、_grid_server_args/baseline double_run 等(SIMPLIFY) |
| io/breakdown | 33 | 3 | sessions PyYAML×2(CUT)、invocations 反转(SIMPLIFY);其余归档读取 KEEP |
| orch/state+policy+bus | 28 | 4 | gate is_relative_to×2、record_action_attempt hasattr(CUT);gate:192(SIMPLIFY) |
| agents/kernel-tools+rest | 34 | 3+ | `_payload_aliases` extra_sglang_args(C+W,簇 D)、tracelens DEFAULT_MODEL、validate_harness(SIMPLIFY) |
| orch/loop | 24 | 4 | sub_agent_runner docstring、result_recorder source_file+ts(CUT);intent_router/coordinator(SIMPLIFY) |

---

## 5. 跨轮冲突与主 agent 复核订正

本轮/上轮之间出现的分歧,已由主 agent 直接读源码裁定:

1. **`result_recorder.py:852` `e.get("ts")`**:本轮 orch/loop agent 判"live,勿删"(称 `_kernel_decisions.py:514/575` 写 `ts`)。**主 agent 复核订正为 CUT(死)**:`_kernel_decisions.py:508-575` 中,`ts` 只写入 `history` 子项(:514)与 `state.last_kernel_opt`(:575);`kernel_opt_attempts` 的 entry 只写 `last_ts`(:545)。`result_recorder.py:852` 读的是 attempts entry `e`,故 `e.get("ts")` 恒 None → 死。与 `:843` `source_file` 同理(entry 只写 `last_source_file`,:539)。
2. **`result_collector.py:270` FileNotFoundError**:本轮 quantization agent 判 SIMPLIFY(删)。**主 agent 保留上轮对抗式结论 KEEP**:LLM agent 持 Bash 可在 mkdir 与 scan 间删 workspace;这是 `collect_artifacts` 唯一 graceful 化缺目录的守卫。列入 Tier 4 待签字。
3. **`orch/phases` 首个 agent 误计数(报 34/实 45)且混入他 unit 文件**:已重跑,以重跑结果(39 KEEP / 4 SIMPLIFY / 2 CUT+WRITER)为准。
4. **`machine.py:123` `_advance_phase_if_needed`**:phases agent 曾报 "getattr 默认值 bug"(称字段默认 `False`)。**主 agent 复核订正:非 bug**——`SharedState.framework_agent_phase_enabled` 字段默认 `True`(`shared_state.py:548`),getattr 默认 `True` 与之一致(agent 误把 `machine_state.py:2094` 的**函数参数**默认 `False` 当成了字段默认)。仍作 SIMPLIFY(直读),但无行为变化。

---

## 6. 为什么 ~675 项 KEEP(归类)

绝大多数 fallback 是承重的,删除会改变真实工作负载行为:

- **外部 I/O / 子进程 / 网络**(最大类):ROCm `rocm-smi`、GEAK/FORGE/GEMM/TraceLens 子进程、Ray Dashboard/SSH/K8s、SaFE/gbrain/GitHub REST、SGLang server 启动 + bench。这些 `try/except` 守卫真实失败。
- **解析外部/LLM/持久化数据**:LLM 乱序/无 fence JSON、子进程 log、跨 schema 归档 session、operator 手写 recipe、litellm 计数器名。
- **可选依赖**:Langfuse `[trace]`、openai(proposal scorer)、psutil、torch/triton/rocprof——`ImportError`/`AttributeError` 守卫合理。
- **SDK 版本漂移**:Langfuse v2/v3/v4 API(未 pin 单一大版本)。
- **真正可选的 config/env**:operator 调参旋钮(超时、headroom、GPUs-per-node)、`--degraded-kb`/`--robustness-agent`/`--nodes` 拓扑开关。
- **运行时可选产品模式**:critic/robustness mock 由 CLI 选择;robustness `--nodes>=2` 自动降 mock。
- **当前双键协议(非遗留)**:`verdict`/`verdict_map`、`gain_pct`/`delta_pct`、`extra_args`(specialist 仍活写)、AMD 多网关 `ANTHROPIC_*`/`OPENAI_*`/`SAFE_API_KEY`——不同部署拓扑的活键,删任一臂会断某类部署。
- **离线历史渲染**:`io/breakdown` 读旧 schema 归档 session 是产品特性。

---

## 7. 测试删除 / 改写清单

| 测试 | 动作 | 关联 |
|---|---|---|
| `test_fetch_rows_*`(`test_inferencex_client_unit.py`、`test_baseline_comparison.py`) | 删(~99 LOC) | Tier 1 fetch_rows |
| `test_llm_trace_unit.py:27,42` resume_downgraded 断言 | 删 | Tier 0 resume_downgrade |
| `test_payload_aliases_shim.py`(legacy_key 两断言)+ `test_no_legacy_writer_sites.py`(整删/清空) | 删 | 簇 D |
| `test_cortex_t0_anchor.py:133-156` `test_t0_anchor_tolerates_legacy_framework_key` | 删 | 簇 B |
| `test_state_paths.py::test_bind_state_file_migrates_legacy` | 删 | 簇 F |
| robustness 遗留 events schema 测试、`extra_evidence_provider` 注入测试 | 删 | Tier 1 |
| robustness `local_ray=None` mock 数据 | 改为 `local_ray={}` | Tier 3 |
| `test_build_assessment_empty_and_bad_gap`(仅 `build_assessment([])` 子断言;`:63-70` 保留)、`test_scan_hypothesis_attempts_missing_dir` | 改/删 | Tier 3(注意 result_collector 若按 Tier 4 保留则不动) |
| 簇 A/B 中含裸 `args`/`envs`/`framework` 的 fixture(`test_coordinator_kb_writes.py`、`test_kg_warmstart_donor.py`、`test_shared_state_evolution.py` 等) | 改为规范键 | 簇 A/B |
| backends 死降级臂/死守卫的直调单测(如有) | 删 | Tier 1 |

> 覆盖率权威仍是 `.github/workflows/tests-coverage.yml`(`fail_under=90`)。删死码连带其专属测试对覆盖率
> 中性或正向;若某测试顺带覆盖共享 live 分支,迁移断言而非丢弃。

---

## 8. 风险与验证

- **执行顺序**:Tier 0 → 1 → 3(SIMPLIFY 中不依赖写入方的)→ 2(先写入方后读取臂)→ 4(逐项签字)。
  Tier 5 不动。每 Tier 一提交序列,可独立 `git revert`。
- **簇 A/B 铁律**:**先落写入方迁移并跑全测,再删读取臂。** 反序会破坏 live 写路径(会把裸键写盘,
  读侧却已删回退)。
- **back-compat 破坏(已接受,登记在案)**:只含 `extra_sglang_args`/裸 `args`/`framework` 的旧 session
  与旧 recipe.json 恢复时将丢对应载荷;`/tmp/multi_node_state.json` 旧文件不再迁移。与 `pr.md`
  的 "Intentional Compatibility Breaks" 一致口径。
- **fail-loud 取舍**:Tier 3 把若干穷举 catch-all 改 `raise`。若不希望运行期因未来枚举疏漏抛异常,
  改为 `log.warning` + 原默认;但不要保留"静默 + 无日志"。
- **逐符号复核**(每次删前重跑,树在变):
  ```bash
  rg -n '<symbol_or_key>' src docs *.sh pyproject.toml
  rg -n '"<string_key>"'        # 字符串键写入方
  ```
  命中只剩定义/`__all__`/测试才可删;命中入口点/MCP/`_DELEGATED`/`python -m`/SKILL.md 视为存活。
- **每 Tier 后**:`ruff check . && mypy src/hyperloom && pytest -m "not critic_agent_e2e and not robustness_agent_e2e"`。

---

## 9. 附:与 `lean-3` 的关系

`lean-3`(已执行,见 `cleaned_function.MD`/`pr.md`)砍的是**死函数 / 未用符号 / 冗余封装**这条轴
(~11k 行),并已顺带删掉一批 fallback(`extra_sglang_args` 主体、GEAK legacy promote、payload_aliases 等)。
本计划是**正交的 fallback 轴**,在 lean-3 之后的树上做,因此剩余可动量小且分散。两者不重叠的部分:
本计划新增的是 lean-3 未覆盖的**表达式内遗留臂**(簇 A/B/C/E)、**死防御守卫**(Py<3.9、PyYAML、
hasattr、resume_downgrade)、以及 **getattr/穷举 的 SIMPLIFY**。若追求继续大体量删除,应回到 lean-3
的死子系统轴(experiments/、KB 查询面等),而非 fallback 轴。
