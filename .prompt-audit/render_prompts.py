# Renders the REAL, fully-composed Hyperloom agent prompts to disk by calling
# the repo's own prompt-composition code. Nothing here hand-writes prompt text:
# every artifact is the return value of a repo builder or a verbatim read of a
# repo .md fragment / repo string constant.
#
#   python .prompt-audit/render_prompts.py
#
# Writes into .prompt-audit/rendered/.

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = Path(__file__).resolve().parent / "rendered"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Repo imports (the real composition code)
# ---------------------------------------------------------------------------
from hyperloom.orchestrator.actions.registry import ActionRegistry  # noqa: E402
from hyperloom.orchestrator.prompts.prompt_builder import (  # noqa: E402
    build_orchestration_prompt,
    default_enabled_actions,
    _section_mission,
    _section_session_context,
    _section_pipeline_and_budget,
    _section_phase_semantics,
    _section_action_catalogue,
    _section_decision_framework,
    _section_cycle_directive,
    _section_rules,
    _KERNEL_OPT_PIPELINE_BODY,
    _resolve_prompt_prelude,
)
from hyperloom.orchestrator.prompts.specialist_prompt_builder import (  # noqa: E402
    SpecialistPromptInputs,
    build_specialist_prompts,
)
from hyperloom.orchestrator.specialists.domains import (  # noqa: E402
    SPECIALIST_DOMAINS,
    FREEFORM_DOMAIN,
    get_domain,
)
from hyperloom.orchestrator.specialists.profile import (  # noqa: E402
    resolve_specialist_profile,
)
from hyperloom.orchestrator.specialists.leaf import (  # noqa: E402
    build_leaf_agents_json,
)
from hyperloom.orchestrator.state.shared_state import SharedState  # noqa: E402
from hyperloom.orchestrator.roles.agent_role import default_role_registry  # noqa: E402
from hyperloom.orchestrator.roles.base import build_chat_messages  # noqa: E402
from hyperloom.inference_optimizer.session.paths import asset_system_prompts_dir  # noqa: E402
from hyperloom.inference_optimizer.cli import (  # noqa: E402
    _build_orchestration_prompt as cli_build_orchestration_prompt,
    _load_critic_prompt,
    _DEFAULT_KERNEL_PROMPT,
)
from hyperloom.orchestrator.state.objective import build_objective  # noqa: E402
from hyperloom.orchestrator.roles.critic_agent import (  # noqa: E402
    _REVIEW_OUTPUT_INSTRUCTIONS,
)

MANIFEST: list[dict] = []


def dump(name: str, text: str, what: str, how: str) -> None:
    p = OUT / name
    p.write_text(text, encoding="utf-8")
    MANIFEST.append(
        {
            "path": str(p),
            "what": what,
            "howProduced": how,
            "lines": len(text.splitlines()),
            "chars": len(text),
            "approxTokens": round(len(text) / 4),
        }
    )
    print(f"  wrote {p.name:52s} {len(text.splitlines()):5d} lines  ~{len(text)//4:6d} tok")


# ===========================================================================
# 1. ORCHESTRATION
# ===========================================================================
print("[1] orchestration")
registry = ActionRegistry().load()
ENABLED_FULL = default_enabled_actions(no_kernel=False)

# Realistic mid-run config: kernel enabled, all default actions, macro_cycle=2,
# an LLM-authored cycle directive, real rules fragment, real source roots.
from hyperloom.orchestrator.framework.paths import resolve_source_file_allowlist  # noqa: E402

RULES_PATH = asset_system_prompts_dir() / "orchestration.md"
CYCLE_DIRECTIVE = (
    "Cycle 2 focus: go deeper on the MoE dispatch path. Spend the amplified "
    "specialist budget on a single long autotune loop over the fused-MoE "
    "Triton configs rather than another wide config sweep."
)

BASE_KW = dict(
    action_registry=registry,
    enabled_actions=ENABLED_FULL,
    framework="sglang",
    kernel_enabled=True,
    explore_enabled=True,
    framework_agent_phase_enabled=True,
    objective_kind="gain_pct",
    objective_value=15.0,
    max_minutes=480,
    macro_cycle=2,
    cycle_directive=CYCLE_DIRECTIVE,
    rules_fragment_path=RULES_PATH,
    framework_source_roots=resolve_source_file_allowlist(),
)

orch = build_orchestration_prompt(**BASE_KW)
dump(
    "orchestration.prompt.md",
    orch,
    "Full composed Orchestration system prompt (kernel on, all default actions, macro_cycle=2, gain_pct=15, 480min)",
    "prompt_builder.build_orchestration_prompt(action_registry=ActionRegistry().load(), "
    "enabled_actions=default_enabled_actions(no_kernel=False), framework='sglang', kernel_enabled=True, "
    "explore_enabled=True, framework_agent_phase_enabled=True, objective_kind='gain_pct', objective_value=15.0, "
    "max_minutes=480, macro_cycle=2, cycle_directive=<text>, rules_fragment_path=asset_system_prompts_dir()/"
    "'orchestration.md', framework_source_roots=resolve_source_file_allowlist())",
)

# --- Phase sensitivity probe -------------------------------------------------
# build_orchestration_prompt takes NO phase argument: the static system prompt is
# phase-INVARIANT. The live phase is injected per tick by the Coordinator via
# SharedState.to_phase_status_summary() (conversation.py _compose_prompt ->
# "=== Phase ===" block). So instead of faking a phase kwarg, we render the real
# per-tick phase block for each phase and concatenate it with the real system
# prompt, exactly as the Coordinator does.
PHASES = ("PRELUDE", "FRAMEWORK_AGENT", "EXPLORE", "KERNEL_AGENT", "SWEEP", "CLOSE")
phase_blocks: dict[str, str] = {}
for ph in PHASES:
    st = SharedState()
    st.phase = ph
    st.macro_cycle = 2
    st.max_minutes = 480
    phase_blocks[ph] = st.to_phase_status_summary()

for ph in ("PRELUDE", "EXPLORE", "KERNEL_AGENT", "CLOSE"):
    composed = (
        f"SESSION_DIR=/workspace/hyperloom/session-EXAMPLE\n"
        f"=== Phase ===\n{phase_blocks[ph]}\n\n"
        f"<<< ---- end of per-tick Coordinator prefix; system prompt below ---- >>>\n\n"
        + orch
    )
    dump(
        f"orchestration.{ph}.md",
        composed,
        f"Same system prompt + the REAL per-tick '=== Phase ===' block for {ph} "
        f"(the only phase-varying part; the system prompt itself is phase-invariant)",
        f"SharedState(phase={ph!r}, macro_cycle=2, max_minutes=480).to_phase_status_summary() "
        f"prepended to build_orchestration_prompt(...) output, mirroring "
        f"loop/conversation.py::_compose_prompt",
    )

# Variant: all three optional phases disabled (shows DISABLED annotations + trimmed catalogue).
orch_nk = build_orchestration_prompt(
    **{
        **BASE_KW,
        "enabled_actions": default_enabled_actions(no_kernel=True, no_explore=True),
        "kernel_enabled": False,
        "explore_enabled": False,
        "framework_agent_phase_enabled": False,
    }
)
dump(
    "orchestration.no-kernel-no-explore.md",
    orch_nk,
    "Orchestration prompt with --no-kernel --no-explore --no-framework-agent (DISABLED annotations, trimmed catalogue, KERNEL-OPT section dropped)",
    "build_orchestration_prompt(..., enabled_actions=default_enabled_actions(no_kernel=True, no_explore=True), "
    "kernel_enabled=False, explore_enabled=False, framework_agent_phase_enabled=False)",
)

# Variant: exactly what the CLI produces (proves the CLI wrapper adds nothing).
obj = build_objective({"MAX_HOURS": "8", "TARGET_GAIN_PCT": "15"})
orch_cli = cli_build_orchestration_prompt(
    no_kernel=False,
    no_explore=False,
    no_framework_agent=False,
    framework="sglang",
    objective=obj,
    max_minutes=480,
    macro_cycle=2,
    cycle_directive=CYCLE_DIRECTIVE,
    action_registry=registry,
)
dump(
    "orchestration.via-cli.md",
    orch_cli,
    "Orchestration prompt built through the production CLI entry point (cli.__init__._build_orchestration_prompt)",
    "hyperloom.inference_optimizer.cli._build_orchestration_prompt(no_kernel=False, no_explore=False, "
    "no_framework_agent=False, framework='sglang', objective=build_objective({'MAX_HOURS':'8',"
    "'TARGET_GAIN_PCT':'15'}), max_minutes=480, macro_cycle=2, cycle_directive=<text>)",
)

# ===========================================================================
# 2. SPECIALIST
# ===========================================================================
print("[2] specialist")

ROOFLINE_EVIDENCE = {
    "roofline_snapshot_id": 7,
    "executive_summary": {
        "compute_pct": 62.0,
        "idle_pct": 14.0,
        "comm_pct": 9.0,
        "top_bottleneck": "fused_moe",
    },
    "kernel_roofline_top15": [
        {
            "kernel_id": "k001",
            "name": "fused_moe_kernel",
            "gpu_pct": 31.4,
            "bound_type": "memory",
            "arithmetic_intensity": 8.2,
            "efficiency_percent": 41.0,
            "compute_utilization_pct": 38.0,
            "bandwidth_utilization_pct": 77.0,
            "recommended_actions": ["retune block sizes", "check split-k"],
        }
    ],
    "hot_kernels_top15": [
        {
            "kernel_id": "k002",
            "name": "paged_attention_v2",
            "gpu_pct": 18.9,
            "bottleneck": "memory",
            "source_file": "vllm/attention/ops/paged_attn.py",
        }
    ],
    "analysis_md_path": "/workspace/hyperloom/session-EXAMPLE/reports/analysis.md",
}

COMMON = dict(
    task_id="spec-0007",
    max_turns=40,
    gpu_type="MI300X",
    allocated_gpu_ids=(4, 5),
    tp=2,
    hbm_gb=192.0,
    peak_tflops=1307.4,
    arch_notes="MoE 8x22B, MLA attention, fp8 weights",
    target_gap_notes="Reference stack reports ~1.6x our current output throughput.",
    precision="fp8",
    conc=64,
    isl=1024,
    osl=1024,
    max_model_len=8192,
    framework="sglang",
    framework_version="0.4.6.post1",
    gap_canonical_id="gap.moe.dispatch.bandwidth_bound",
    gap_symptom="fused_moe_kernel is 31% of GPU time and bandwidth-bound at 77% BW util",
    gap_layer="kernel_agent",
    gap_evidence={"kernel_id": "k001", "gpu_pct": 31.4, "bw_util_pct": 77.0},
    roofline_evidence=ROOFLINE_EVIDENCE,
    warm_start_recipe={"best_config": {"--max-num-seqs": "256", "SGLANG_AITER_MOE": "1"}},
    warm_start_lessons=[
        {
            "confidence": 0.9,
            "attrs": {
                "statement": "enable AITER fused MoE on MI300X fp8",
                "measured_impact": {
                    "gain_pct": 6.4,
                    "throughput_after": 4211.0,
                    "stack_depth_at_apply": 2,
                    "measured_at": "2026-06-02T11:00:00",
                },
                "validated_count": 4,
                "source_session_ids": ["s-101", "s-118"],
                "framework_version": "0.4.5",
            },
        }
    ],
    warm_start_pitfalls=[
        {
            "confidence": 0.8,
            "attrs": {
                "description": "--enforce-eager regressed throughput 12% on every MI300X run",
                "severity": "high",
                "validated_count": 3,
                "source_session_id": "s-093",
            },
        }
    ],
    kg_recommended_knobs=[
        {"knob": "chunked_prefill_size", "expected_gain": 3.1, "confidence": 0.6, "source": "kg:arch_family"}
    ],
    kg_guided_knobs=[
        {
            "knob": "cuda_graph_max_bs",
            "args": "--cuda-graph-max-bs 256",
            "envs": {},
            "name": "cudagraph-256",
            "expected_gain": 2.4,
            "confidence": 0.55,
            "source": "journal:KNOB_IMPROVES",
        }
    ],
    pr_monitor_available=True,
    framework_source_roots=("/sgl-workspace/sglang", "/sgl-workspace/aiter"),
    source_hint_directories=("/sgl-workspace/sglang/python/sglang/srt/layers/moe",),
    model_info={"attention_type": "MLA", "is_moe": True, "quantization": "fp8", "has_shared_expert": True,
                "num_shared_experts": 1},
    workspace_path="/workspace/hyperloom/session-EXAMPLE/runs/specialist/spec-0007",
    notes="Round 3 residual: does AITER MoE respect --chunked-prefill-size?",
    wall_budget_sec=5400.0,
    started_at_iso="2026-08-03T12:00:00+00:00",
)


def spec_dump(slug: str, inp: SpecialistPromptInputs, what: str, how: str) -> None:
    system, user = build_specialist_prompts(inp)
    dump(f"specialist.{slug}.system.prompt.md", system, f"{what} — SYSTEM half", how + " -> [0] system")
    dump(f"specialist.{slug}.user.prompt.md", user, f"{what} — USER half", how + " -> [1] user")
    # The subprocess dispatcher collapses both halves into one prompt.md file.
    combined = "<!-- system_prompt -->\n" + system + "\n<!-- user_prompt -->\n" + user
    dump(
        f"specialist.{slug}.prompt.md",
        combined,
        f"{what} — combined prompt.md exactly as specialists/subprocess_.py writes it",
        how + "; combined with the literal wrapper from specialists/subprocess_.py "
        "(\"<!-- system_prompt -->\\n\" + system + \"\\n<!-- user_prompt -->\\n\" + user)",
    )


# --- scope=domain, mode=patch, bench=False, lane=gpu (the default dials) -----
# One artifact per catalogue domain, since the domain focus template is the
# single biggest source of variation in Section 1.
for dom in SPECIALIST_DOMAINS:
    prof = resolve_specialist_profile({"domain": dom.key})
    inp = SpecialistPromptInputs(
        domain=dom,
        scope=prof.scope,
        mode=prof.mode,
        bench=prof.bench,
        lane=prof.lane,
        **COMMON,
    )
    spec_dump(
        f"scope-domain.{dom.key}",
        inp,
        f"Specialist prompt, scope=domain mode={prof.mode} bench={prof.bench} lane={prof.lane}, domain={dom.key}",
        f"specialist_prompt_builder.build_specialist_prompts(SpecialistPromptInputs(domain=get_domain({dom.key!r}), "
        f"scope={prof.scope!r}, mode={prof.mode!r}, bench={prof.bench!r}, lane={prof.lane!r}, <rich realistic ctx>)) "
        f"with dials from specialists.profile.resolve_specialist_profile({{'domain': {dom.key!r}}})",
    )

# --- scope=freeform ---------------------------------------------------------
ff_params = {
    "task_description": (
        "Investigate why SGLang's DP-attention path is not engaging on this "
        "MI300X 8-GPU node despite --enable-dp-attention; find the predicate "
        "that disables it and author a bridge patch if one exists."
    ),
}
ff_prof = resolve_specialist_profile(ff_params)
ff = SpecialistPromptInputs(
    domain=FREEFORM_DOMAIN,
    scope=ff_prof.scope,
    mode=ff_prof.mode,
    bench=ff_prof.bench,
    lane=ff_prof.lane,
    task_description=ff_params["task_description"],
    **COMMON,
)
spec_dump(
    "scope-freeform",
    ff,
    f"Specialist prompt, scope=freeform mode={ff_prof.mode} bench={ff_prof.bench} lane={ff_prof.lane} "
    f"(synthetic FREEFORM_DOMAIN, no catalogue lock)",
    "build_specialist_prompts(SpecialistPromptInputs(domain=domains.FREEFORM_DOMAIN, scope/mode/bench/lane from "
    "resolve_specialist_profile({'task_description': ...}), task_description=<text>, <rich realistic ctx>))",
)

# --- scope=domains (cross-domain) -------------------------------------------
xd_params = {"tags": ["serving_specialist", "comm_specialist", "compiler_specialist"]}
xd_prof = resolve_specialist_profile(xd_params)
xd = SpecialistPromptInputs(
    domain=get_domain("serving_specialist"),
    scope=xd_prof.scope,
    mode=xd_prof.mode,
    bench=xd_prof.bench,
    lane=xd_prof.lane,
    extra_focus_tags=("communication", "compiler"),
    **COMMON,
)
spec_dump(
    "scope-domains.cross-domain",
    xd,
    f"Specialist prompt, scope=domains (cross-domain over serving+comm+compiler) mode={xd_prof.mode} "
    f"bench={xd_prof.bench} lane={xd_prof.lane}",
    "build_specialist_prompts(SpecialistPromptInputs(domain=get_domain('serving_specialist'), "
    "extra_focus_tags=('communication','compiler'), scope/mode/bench/lane from "
    "resolve_specialist_profile({'tags': ['serving_specialist','comm_specialist','compiler_specialist']})))",
)

# --- mode=research, lane=cpu, bench=False (read-only research dial) ---------
res_params = {"domain": "research_scout_specialist", "mode": "research", "lane": "cpu", "bench": False}
res_prof = resolve_specialist_profile(res_params)
res_common = {**COMMON, "allocated_gpu_ids": ()}  # research lane holds no cards
res = SpecialistPromptInputs(
    domain=get_domain("research_scout_specialist"),
    scope=res_prof.scope,
    mode=res_prof.mode,
    bench=res_prof.bench,
    lane=res_prof.lane,
    already_proven=[{"name": "SGLANG_AITER_MOE=1", "source": "warm recipe s-101"}],
    **res_common,
)
spec_dump(
    "mode-research.lane-cpu",
    res,
    f"Specialist prompt, mode=research lane=cpu bench=False, no GPU allocation "
    f"(exercises the no-GPU iron-rule branch and drops the on-GPU autonomy block)",
    "build_specialist_prompts(SpecialistPromptInputs(domain=get_domain('research_scout_specialist'), "
    "allocated_gpu_ids=(), scope/mode/bench/lane from resolve_specialist_profile("
    "{'domain':'research_scout_specialist','mode':'research','lane':'cpu','bench':False})))",
)

# --- mode=patch, bench=True, lane=gpu (bench-capable patch dial) ------------
bench_params = {"domain": "serving_specialist", "mode": "patch", "bench": True, "lane": "gpu"}
bench_prof = resolve_specialist_profile(bench_params)
bench_inp = SpecialistPromptInputs(
    domain=get_domain("serving_specialist"),
    scope=bench_prof.scope,
    mode=bench_prof.mode,
    bench=bench_prof.bench,
    lane=bench_prof.lane,
    **COMMON,
)
spec_dump(
    "mode-patch.bench-true.lane-gpu",
    bench_inp,
    f"Specialist prompt, mode=patch bench=True lane=gpu (reserves_benchmark_lane="
    f"{bench_prof.reserves_benchmark_lane})",
    "build_specialist_prompts(SpecialistPromptInputs(domain=get_domain('serving_specialist'), "
    "scope/mode/bench/lane from resolve_specialist_profile("
    "{'domain':'serving_specialist','mode':'patch','bench':True,'lane':'gpu'})))",
)

# --- cold-start (no priors at all) + auto-retry note ------------------------
cold = SpecialistPromptInputs(
    task_id="spec-0001",
    domain=get_domain("serving_specialist"),
    max_turns=40,
    gpu_type="MI300X",
    allocated_gpu_ids=(4, 5),
    tp=2,
    framework="sglang",
    gap_canonical_id="gap.cold",
    gap_symptom="baseline throughput below expectation, no priors yet",
    gap_layer="framework",
    workspace_path="/workspace/hyperloom/session-EXAMPLE/runs/specialist/spec-0001",
    auto_retry_reason="previous attempt timed out at the wall-clock budget",
    wall_budget_sec=3600.0,
    started_at_iso="2026-08-03T12:00:00+00:00",
)
spec_dump(
    "cold-start.auto-retry",
    cold,
    "Specialist prompt with NO priors (COLD-START MODE branch in §4) plus the auto-retry notice block",
    "build_specialist_prompts(SpecialistPromptInputs(domain=get_domain('serving_specialist'), "
    "kb_subgraph={}, warm_start_*={}/[], research_hints='', auto_retry_reason=<text>))",
)

# --- Leaf sub-agent prompt (a real repo constant, emitted via build_leaf_agents_json)
leaf_json = json.loads(build_leaf_agents_json())
leaf_name = next(iter(leaf_json))
dump(
    "specialist.leaf-subagent.prompt.md",
    leaf_json[leaf_name]["prompt"] + "\n",
    "Leaf sub-agent system prompt (what a specialist's Task(subagent_type='hyperloom-leaf') fan-out receives)",
    "json.loads(specialists.leaf.build_leaf_agents_json())['hyperloom-leaf']['prompt']",
)

# ===========================================================================
# 3. CRITIC / ROBUSTNESS / KERNEL_AGENT roles
# ===========================================================================
print("[3] critic / robustness / kernel_agent")

roles = default_role_registry()

# --- critic: raw .md fragment, loaded the same way the CLI does -------------
critic_md = _load_critic_prompt()
dump(
    "critic.prompt.md",
    critic_md,
    "Critic system prompt — the raw critic.md fragment, verbatim (no builder wraps it)",
    "hyperloom.inference_optimizer.cli._load_critic_prompt() "
    "== (asset_system_prompts_dir()/'critic.md').read_text()",
)

# The critic role's real turn is a chat-messages pair: this system prompt plus a
# user prompt assembled in roles/critic_agent.py::_reason as
#   <SKILL.md + review_coordinator_inbox.md preamble> + JUDGE BUNDLE + _REVIEW_OUTPUT_INSTRUCTIONS
# Render that wrapper with the REAL preamble files and a minimal realistic bundle.
critic_root = REPO / "src" / "hyperloom" / "agents" / "critic"
preamble_parts: list[str] = []
for rel in ("SKILL.md", "actions/review_coordinator_inbox.md"):
    p = critic_root / rel
    try:
        preamble_parts.append(f"==== {rel} ====\n{p.read_text(encoding='utf-8').strip()}")
    except OSError:
        pass
preamble = "\n\n".join(preamble_parts)

judge_bundle_view = {
    "kind": "review_coordinator_inbox",
    "session_id": "session-EXAMPLE",
    "decision_id": "dec-0042",
    "merged_context": {"phase": "EXPLORE", "baseline_tput": 3960.0, "cumulative_gain": 7.4},
    "missing_context": [],
    "required_context": [],
    "proposals": [
        {
            "msg_id": "m-0311",
            "action_name": "explore",
            "predicted_gain_pct": 3.0,
            "params": {"grid": [{"name": "cudagraph-256", "extra_args": "--cuda-graph-max-bs 256"}]},
        }
    ],
    "kb_priors_by_proposal": {"m-0311": []},
    "kb_priors_for_decision": [],
    "kb_read_skipped_reason": None,
    "review_constraints": {"known_actions": list(ENABLED_FULL)},
    "notes": [],
}
bundle_text = json.dumps(judge_bundle_view, ensure_ascii=False, separators=(",", ":"))
critic_user = (
    f"{preamble}\n\n"
    f"==== JUDGE BUNDLE ====\n{bundle_text}\n==== END JUDGE BUNDLE ====\n\n"
    f"{_REVIEW_OUTPUT_INSTRUCTIONS}"
)
dump(
    "critic.user-wrapper.prompt.md",
    critic_user,
    "Critic USER prompt wrapper as roles/critic_agent.py::_reason builds it: real SKILL.md + "
    "review_coordinator_inbox.md preamble + JUDGE BUNDLE json + _REVIEW_OUTPUT_INSTRUCTIONS",
    "Reproduced verbatim from CriticAgentBackend._reason: "
    "f'{self._load_skill_preamble()}\\n\\n==== JUDGE BUNDLE ====\\n{json bundle}\\n==== END JUDGE BUNDLE ====\\n\\n"
    "{_REVIEW_OUTPUT_INSTRUCTIONS}', with _load_skill_preamble() re-implemented over the real "
    "src/hyperloom/agents/critic/{SKILL.md,actions/review_coordinator_inbox.md}",
)

critic_msgs = build_chat_messages(critic_md, critic_user)
dump(
    "critic.chat-messages.json",
    json.dumps(critic_msgs, ensure_ascii=False, indent=2) + "\n",
    "The exact OpenAI-style messages array the Critic backend sends (system=critic.md, user=the wrapper above)",
    "roles.base.build_chat_messages(_load_critic_prompt(), <critic user wrapper>)",
)

# --- robustness -------------------------------------------------------------
rob_role = roles["robustness"]
rob_md = rob_role.load_system_prompt()
dump(
    "robustness.prompt.md",
    rob_md,
    "Robustness system prompt — the raw robustness.md fragment resolved through the role registry",
    "default_role_registry()['robustness'].load_system_prompt() "
    "(AgentRole.system_prompt_path == asset_system_prompts_dir()/'robustness.md')",
)

# --- kernel_agent -----------------------------------------------------------
kern_role = roles["kernel_agent"]
kern_md = kern_role.load_system_prompt()
dump(
    "kernel_agent.prompt.md",
    kern_md,
    "Kernel-agent system prompt — the raw kernel_agent.md fragment resolved through the role registry "
    "(what Coordinator._load_system_prompt returns when no CLI override is set)",
    "default_role_registry()['kernel_agent'].load_system_prompt() "
    "(AgentRole.system_prompt_path == asset_system_prompts_dir()/'kernel_agent.md')",
)
dump(
    "kernel_agent.cli-override.prompt.md",
    _DEFAULT_KERNEL_PROMPT + "\n",
    "Kernel-agent prompt the production CLI actually installs into "
    "coordinator.system_prompt_overrides['kernel_agent'] (a Python string constant, NOT kernel_agent.md) — "
    "this SHADOWS kernel_agent.md on every real run where --no-kernel is not set",
    "hyperloom.inference_optimizer.cli._DEFAULT_KERNEL_PROMPT (see cli/__init__.py "
    "prompts['kernel_agent'] = args.kernel_prompt or _DEFAULT_KERNEL_PROMPT)",
)

# ===========================================================================
# 4. SECTION_ORDER.md — derived by re-running each section builder separately
# ===========================================================================
print("[4] SECTION_ORDER.md")

actions, kernel_enabled, framework_norm, rules_md = _resolve_prompt_prelude(
    registry, ENABLED_FULL, "sglang", True, RULES_PATH
)

section_specs = [
    (
        "1. MISSION",
        "computed-from-code (static literal list in _section_mission)",
        _section_mission(),
        "always",
    ),
    (
        "2. SESSION CONTEXT",
        "computed-from-code (interpolates framework / kernel_enabled / explore_enabled / "
        "framework_agent_phase_enabled / objective / max_minutes / framework_source_roots)",
        _section_session_context(
            framework=framework_norm,
            kernel_enabled=kernel_enabled,
            objective_kind="gain_pct",
            objective_value=15.0,
            max_minutes=480,
            explore_enabled=True,
            framework_agent_phase_enabled=True,
            framework_source_roots=BASE_KW["framework_source_roots"],
        ),
        "always",
    ),
    (
        "3. PIPELINE & TIME BUDGET",
        "computed-from-code (per-phase ETA sums over ActionMetadata.typical_runtime_min of the enabled actions)",
        _section_pipeline_and_budget(actions, max_minutes=480),
        "always",
    ),
    (
        "3a. PHASE CONTRACT",
        "computed-from-code (phases.machine_state.render_phase_proposable_bullets over PHASE_NAMES x "
        "llm_proposable_actions_for)",
        _section_phase_semantics(kernel_enabled=True, explore_enabled=True, framework_agent_phase_enabled=True),
        "always (rows annotated '(DISABLED: --no-xxx — phase skipped)' when the matching flag is off)",
    ),
    (
        "4. ACTIONS YOU MAY USE",
        "computed-from-code (ActionRegistry metadata: description / runtime / gain / risks / family + EMIT hint "
        "+ grid-injection hint)",
        _section_action_catalogue(actions),
        "always (content depends entirely on enabled_actions)",
    ),
    (
        "5. DECISION FRAMEWORK",
        "computed-from-code (static literal list; takes kernel_enabled but does not currently branch on it)",
        _section_decision_framework(kernel_enabled=True),
        "always",
    ),
    (
        "CYCLE DIRECTIVE",
        "computed-from-code (conditional body: LLM-authored cycle_directive text when non-empty, else the "
        "standing breadth->depth default arc)",
        _section_cycle_directive(macro_cycle=2, cycle_directive=CYCLE_DIRECTIVE),
        "always emitted; BODY is conditional on cycle_directive.strip() being non-empty",
    ),
    (
        "6. KERNEL-OPT REQUEST REFERENCE",
        "computed-from-code (module-level string constant _KERNEL_OPT_PIPELINE_BODY, split to lines)",
        _KERNEL_OPT_PIPELINE_BODY.splitlines(),
        "CONDITIONAL: `kernel_enabled and any(a.name == 'kernel_opt' for a in actions)`",
    ),
    (
        "7. RULES & OUTPUT PROTOCOL",
        "static-from-.md (verbatim orchestration.md, wrapped in a '## 7. RULES & OUTPUT PROTOCOL' header; "
        "placeholder text substituted when the fragment is unreadable)",
        _section_rules(rules_md),
        "always (falls back to a one-line placeholder when rules_fragment_path is None/unreadable)",
    ),
]

def rendered_lines(lines: list[str]) -> int:
    """True rendered line count of a section as join_sections() will emit it.

    A section entry may itself be a multi-line string (``_section_rules`` embeds the
    whole orchestration.md fragment as ONE element), and a trailing "" element is a
    real blank line in the output — so count newlines rather than list length or
    ``splitlines()`` (which would swallow the trailing blank)."""
    text = "\n".join(lines)
    return text.count("\n") + 1 if text else 0


so: list[str] = [
    "# build_orchestration_prompt — emitted section order",
    "",
    "Source: `src/hyperloom/orchestrator/prompts/prompt_builder.py::build_orchestration_prompt`.",
    "Sections are joined by `join_sections()`: lines with `\\n`, sections with a blank line,",
    "rstripped, trailing newline. There is **no phase argument** — the composed system prompt",
    "is phase-invariant; the live phase arrives per tick from the Coordinator.",
    "",
    "Config used for the line counts below: kernel_enabled=True, explore_enabled=True,",
    "framework_agent_phase_enabled=True, enabled_actions=default_enabled_actions(no_kernel=False),",
    "framework='sglang', objective gain_pct=15.0, max_minutes=480, macro_cycle=2,",
    "cycle_directive=<non-empty>, rules_fragment_path=<real orchestration.md>.",
    "",
    "| # | Section | Origin | Lines | Gate |",
    "|---|---------|--------|-------|------|",
]
for i, (name, origin, lines, gate) in enumerate(section_specs, start=1):
    so.append(f"| {i} | {name} | {origin} | {rendered_lines(lines)} | {gate} |")

total_section_lines = sum(rendered_lines(l) for _, _, l, _ in section_specs)
so += [
    "",
    f"Sum of section line counts: {total_section_lines}.",
    f"Plus the {len(section_specs) - 1} blank separator lines `join_sections` inserts "
    f"between sections: {total_section_lines + len(section_specs) - 1}.",
    f"Rendered `orchestration.prompt.md`: {len(orch.splitlines())} lines "
    f"(the final `.rstrip()` drops the catalogue's trailing blank line).",
    "",
    "## Per-section detail",
    "",
]
for i, (name, origin, lines, gate) in enumerate(section_specs, start=1):
    so += [
        f"### {i}. {name}",
        "",
        f"- **origin**: {origin}",
        f"- **rendered lines**: {rendered_lines(lines)}"
        + ("  (the section is a single list element containing the whole multi-line fragment)"
           if len(lines) != rendered_lines(lines) else ""),
        f"- **gate**: {gate}",
        f"- **first line**: `{lines[0] if lines else '(empty)'}`",
        "",
    ]

so += [
    "## What is NOT in the system prompt (Coordinator injects per tick)",
    "",
    "`loop/conversation.py::_compose_prompt` prepends, on every tick, in this order:",
    "",
    "1. `SESSION_DIR=<path>` (all agents)",
    "2. `=== Phase ===` + `SharedState.to_phase_status_summary()` (all agents) — "
    "phase / cycle / entered / budget / allowed, plus a `force_exit:` line in EXPLORE",
    "3. recovered `orchestration_memory` seed block (orchestration, SEED turns only)",
    "4. `=== Mission progress ===` + `to_mission_summary()` (orchestration)",
    "5. cycle-strategy seed block (orchestration, SEED only)",
    "6. `=== Time budget ===` elapsed/remaining/budget/closing_phase (orchestration + robustness)",
    "7. `=== Shared session state ===` + `to_prompt_summary()` and `=== Resource pools ===` (SEED only)",
    "8. advisory/ledger blocks (plateau, bottleneck redirect, acceptance threshold, target gap, "
    "priors match, current gaps, recent policy denials) — SEED only",
    "9. inbox tail",
    "",
    "That is why `orchestration.PRELUDE.md` / `.EXPLORE.md` / `.KERNEL_AGENT.md` / `.CLOSE.md` in this",
    "directory differ ONLY in their `=== Phase ===` prefix: the system prompt below the marker is",
    "byte-identical across all four.",
    "",
    "### Per-phase `=== Phase ===` blocks (rendered from the real SharedState renderer)",
    "",
]
for ph in PHASES:
    so += ["```", f"# {ph}", phase_blocks[ph], "```", ""]

dump(
    "SECTION_ORDER.md",
    "\n".join(so),
    "Ordered section inventory for build_orchestration_prompt with origin / line count / gate per section",
    "Each _section_* helper in prompt_builder.py re-invoked individually with the same args "
    "build_orchestration_prompt passes them (via _resolve_prompt_prelude), then len() of the returned line list",
)

# ===========================================================================
# manifest
# ===========================================================================
(OUT / "_manifest.json").write_text(json.dumps(MANIFEST, indent=2) + "\n", encoding="utf-8")
print(f"\n{len(MANIFEST)} artifacts -> {OUT}")
