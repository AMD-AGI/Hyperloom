# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for Coordinator pure/sync helper methods."""

from __future__ import annotations

import pytest

from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.knowledge.recipe_journal import (
    workload_kb_tags,
)
from hyperloom.orchestrator.bus.gpu_pool import SpecialistGpuPool
from hyperloom.orchestrator.loop.dispatcher import HeldLeases

from .conftest import make_coordinator


@pytest.fixture
def coord(session_dir) -> Coordinator:
    return make_coordinator(session_dir)


# -- The specialist wall-clock deadline ------------------------------------
def test_specialist_wall_budget_base_no_macro_cycle(coord: Coordinator) -> None:
    # macro_cycle == 0 → base mode values (research 10min / patch 60min).
    coord.shared_state.macro_cycle = 0
    assert coord.dispatcher._specialist_wall_budget_sec(params={"mode": "research"}) == 10 * 60
    assert coord.dispatcher._specialist_wall_budget_sec(params={"mode": "patch"}) == 60 * 60


def test_specialist_wall_budget_macro_cycle_amplifies(coord: Coordinator) -> None:
    coord.shared_state.macro_cycle = 1
    assert coord.dispatcher._specialist_wall_budget_sec(params={"mode": "research"}) == 20 * 60
    assert coord.dispatcher._specialist_wall_budget_sec(params={"mode": "patch"}) == 120 * 60


def test_specialist_wall_budget_caps_at_4h(coord: Coordinator) -> None:
    coord.shared_state.macro_cycle = 10
    assert coord.dispatcher._specialist_wall_budget_sec(params={"mode": "patch"}) == 240 * 60
    assert coord.dispatcher._specialist_wall_budget_sec(params={"mode": "research"}) == 110 * 60


def test_bench_specialist_budget_covers_rebench_timeout(coord: Coordinator) -> None:
    """Bench-capable specialists receive enough time for their advertised rebench."""
    from hyperloom.orchestrator.bus.gpu_pool import GPU_LEASE_TTL_GRACE
    from hyperloom.orchestrator.actions.executors._subprocess_kill import resolve_benchmark_timeouts

    params = {"scope": "domain", "mode": "patch", "bench": True}
    budget = coord.dispatcher._specialist_wall_budget_sec(params=params)

    assert budget == max(60 * 60, resolve_benchmark_timeouts()[1] + 10 * 60)
    assert coord.dispatcher.gpu_lease_ttl_sec(params=params) == pytest.approx(
        int(budget * (1.0 + GPU_LEASE_TTL_GRACE)), abs=2
    )


def test_specialist_deadline_does_not_outlast_the_session(coord: Coordinator) -> None:
    """A profile floor cannot extend a finite session past its own budget."""
    coord.shared_state.max_minutes = 30
    coord.shared_state.begin_leg()

    deadline = coord.dispatcher._specialist_deadline(params={"scope": "domain", "mode": "patch", "bench": True})

    assert deadline.remaining() == pytest.approx(30 * 60, abs=2)


def test_a_spent_session_yields_an_expired_specialist_deadline(coord: Coordinator) -> None:
    """An exhausted budget must tighten the specialist bound, never remove it."""
    import time as _time

    coord.shared_state.max_minutes = 30
    coord.shared_state.begin_leg(now_unix=_time.time() - 3_600.0)

    ample = coord.dispatcher._specialist_deadline(params={"mode": "patch"})
    coord.shared_state.max_minutes = 240
    coord.shared_state.begin_leg()
    fresh = coord.dispatcher._specialist_deadline(params={"mode": "patch"})

    assert ample.expired()
    assert not fresh.expired()
    assert ample.remaining() < fresh.remaining()


def _phase_with_minutes_left(coord: Coordinator, monkeypatch: pytest.MonkeyPatch, phase: str, minutes: float) -> None:
    from hyperloom.orchestrator.phases import machine_state

    coord.shared_state.baseline_tput = 800.0
    coord.shared_state.phase = phase
    monkeypatch.setattr(machine_state, "phase_budget_remaining_seconds", lambda _state, **_kw: minutes * 60.0)


def test_a_specialist_dispatched_in_a_gated_phase_stops_at_the_phase_end(
    coord: Coordinator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither its dispatch deadline nor a later lease extension carries a specialist past its phase's end."""
    from types import SimpleNamespace

    from hyperloom.orchestrator.specialists import wall_budget

    _phase_with_minutes_left(coord, monkeypatch, "FRAMEWORK_AGENT", 15)
    task = SimpleNamespace(task_id="spec-phase-end", kind="specialist", params={"mode": "patch"})

    context = coord.dispatcher._dispatch_context(task)

    try:
        assert context["specialist_deadline"].remaining() == pytest.approx(15 * 60, abs=2)
        assert wall_budget.grant_wall_budget_extension(task.task_id, 600) == 0.0
    finally:
        wall_budget.limit_wall_budget_extension(task.task_id, None)
        wall_budget.clear_wall_budget_extension(task.task_id)


def test_a_specialist_is_admitted_when_its_shortest_run_fits_the_phase(
    coord: Coordinator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Admission asks for a mode's base budget, not the full wall budget the dispatch then tightens."""
    _phase_with_minutes_left(coord, monkeypatch, "FRAMEWORK_AGENT", 30)

    assert coord.dispatcher.admission_denial_for_action("specialist", params={"mode": "research"}) is None
    denied = coord.dispatcher.admission_denial_for_action("specialist", params={"mode": "patch"})
    assert denied is not None and denied.rule == "phase_time_budget"


# -- GPU lease TTL re-source + structured-finally release -------------------
def test_gpu_lease_ttl_grace_over_wall_budget(coord: Coordinator) -> None:
    # TTL = wall_budget × (1 + grace); lease must outlive the kill.
    from hyperloom.orchestrator.bus.gpu_pool import GPU_LEASE_TTL_GRACE

    coord.shared_state.macro_cycle = 0
    budget = coord.dispatcher._specialist_wall_budget_sec(params={"mode": "patch"})  # 3600
    ttl = int(budget * (1.0 + GPU_LEASE_TTL_GRACE))
    assert ttl == int(3600 * 1.1)
    assert ttl >= budget
    assert coord.dispatcher.gpu_lease_ttl_sec(params={"mode": "patch"}) == pytest.approx(ttl, abs=2)


def test_run_dispatched_releases_gpu_lease_on_exception(coord: Coordinator) -> None:
    import asyncio

    released: list[object] = []

    class _Task:
        task_id = "tg-boom"
        kind = "explore"
        requires_lanes: list = []

    async def _boom(task, *, prebound_lease=None, extra_context=None, release_resources=None):
        try:
            raise RuntimeError("subprocess crashed")
        finally:
            await release_resources()

    async def _fake_release(lease):
        released.append(lease)

    coord.sub.run_task = _boom
    coord.gpu_specialist_pool.release = _fake_release
    sentinel_lease = object()
    with pytest.raises(RuntimeError, match="subprocess crashed"):
        asyncio.run(
            coord.dispatcher.run_task_registered(
                _Task(), HeldLeases(prebound_lease=None, gpu_lease=sentinel_lease), extra_context={}
            )
        )
    # lease released via finally even though run_task raised.
    assert released == [sentinel_lease]


def test_run_dispatched_no_gpu_lease_is_noop(coord: Coordinator) -> None:
    import asyncio

    class _Task:
        task_id = "tc-cpu"
        kind = "report"
        requires_lanes: list = []

    async def _fake_run_task(task, *, prebound_lease=None, extra_context=None, release_resources=None):
        await release_resources()
        return "CPU"

    async def _run() -> object:
        coord.gpu_specialist_pool = SpecialistGpuPool(coord.db, gpu_ids=[0])
        held = await coord.gpu_specialist_pool.try_acquire(count=1, holder_id="other", task_id="other")
        assert held is not None
        out = await coord.dispatcher.run_task_registered(
            _Task(), HeldLeases(prebound_lease=None, gpu_lease=None), extra_context={}
        )
        # No GPU lease -> another holder's capacity is left alone.
        assert await coord.db.fetchone("SELECT 1 FROM gpu_leases WHERE holder_id='other'") is not None
        return out

    coord.sub.run_task = _fake_run_task
    assert asyncio.run(_run()) == "CPU"


# -- static / pure helpers -------------------------------------------------
def test_is_promotable_result(coord: Coordinator) -> None:
    assert coord.writeback.is_promotable_result("sweep", {"status": "succeeded"}) is True
    assert coord.writeback.is_promotable_result("sweep", {"status": "failed"}) is False
    assert coord.writeback.is_promotable_result("replay_warm_recipe", {"status": "failed"}) is True
    assert coord.writeback.is_promotable_result("explore", {"status": "ok"}) is True
    assert coord.writeback.is_promotable_result("explore", {"status": "failed"}) is False


def test_is_promotable_result_baseline_eval_failed(coord: Coordinator) -> None:
    measured = {"output_throughput": 1000.0, "completed_requests": 10}
    assert coord.writeback.is_promotable_result("baseline", measured) is True
    eval_failed = {**measured, "baseline_eval_failed": True}
    assert coord.writeback.is_promotable_result("baseline", eval_failed) is False
    # profile with the same key still promotes (blocker is baseline-only).
    assert coord.writeback.is_promotable_result("profile", eval_failed) is True


# -- phase / id helpers ----------------------------------------------------
def test_journal_entry_phase(coord: Coordinator) -> None:
    coord.shared_state.phase = ""
    assert coord.recipe_journal.journal_entry_phase() == "UNKNOWN"
    coord.shared_state.phase = "framework_agent"
    assert coord.recipe_journal.journal_entry_phase() == "FRAMEWORK_AGENT"


# -- watermark / tput projection ------------------------------------------
def test_current_tput_from_validated_gain(coord: Coordinator) -> None:
    coord.shared_state.baseline_tput = 0.0
    assert coord.phase_kernel.current_tput_from_validated_gain() == 0.0
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.cumulative_gain_validated = 10.0
    assert coord.phase_kernel.current_tput_from_validated_gain() == pytest.approx(110.0)


# -- gap extraction --------------------------------------------------------
def test_extract_gaps_from_attempts(coord: Coordinator) -> None:
    ss = coord.shared_state
    ss.baseline_tput = 100.0
    ss.last_action_failures = [
        {"action": "kernel_opt", "error_class": "oom", "variant_name": "v1"},
        {"action": "kernel_opt", "error_class": "oom", "variant_name": "v2"},
    ]
    gaps = coord.gap_refresh._extract_gaps_from_attempts(coord.recipe_journal.workload_canonical_id())
    # distinct variant_names produce separate gaps; each has one attempt
    fail_gaps = [g for g in gaps if "fail:kernel_opt:oom" in g["canonical_id"]]
    assert len(fail_gaps) == 2
    assert all(len(g["attempts"]) == 1 for g in fail_gaps)


def test_extract_gaps_symptom_uses_excerpt(coord: Coordinator) -> None:
    """Symptom is built from the first non-empty excerpt line when available."""
    ss = coord.shared_state
    ss.baseline_tput = 100.0
    ss.last_action_failures = [
        {
            "action": "explore",
            "error_class": "server_init_dead",
            "variant_name": "fp8_kv",
            "error_excerpt": "mla_gluon[bh16bn128] requires batch_size=1, got 512",
        },
    ]
    gaps = coord.gap_refresh._extract_gaps_from_attempts(coord.recipe_journal.workload_canonical_id())
    fail_gaps = [g for g in gaps if "fail:explore:server_init_dead" in g["canonical_id"]]
    assert fail_gaps
    assert "mla_gluon" in fail_gaps[0]["symptom"]


# -- advisory blocks (empty-guard paths) ----------------------------------
def test_no_plateau_advisory_outside_framework_agent(coord: Coordinator) -> None:
    assert coord.conversation.plateau_advisory_block() == ""


# -- specialist findings block --------------------------------------------
def _round(domain: str, finding: str, confidence, questions=()) -> dict:
    return {
        "domain": domain,
        "confidence": confidence,
        "new_findings": [finding],
        "residual_questions": list(questions),
    }


def _findings(coord: Coordinator) -> str:

    return coord.conversation._specialist_findings_block()


def test_specialist_findings_survive_a_non_numeric_confidence(coord: Coordinator) -> None:
    """``confidence`` is an audit field, so no value of it can drop the section.

    It reaches the row straight from the specialist's own JSON, and the schema
    invites a free-form self-assessment, so a string or a dict there must not
    cost every domain its findings.
    """
    coord.shared_state.specialist_rounds = [
        _round("serving_specialist", "kv cache is the bottleneck", "high"),
        _round("comm_specialist", "all_reduce dominates", {"level": "high"}),
        _round("kernel_specialist", "gemm is fine", 0.7, questions=["what about fp8?"]),
    ]

    block = _findings(coord)

    assert "kv cache is the bottleneck" in block
    assert "all_reduce dominates" in block
    assert "gemm is fine" in block
    assert "[kernel_specialist] what about fp8?" in block


def test_specialist_findings_are_ordered_newest_first(coord: Coordinator) -> None:
    coord.shared_state.specialist_rounds = [
        _round("serving_specialist", "older finding", 0.9),
        _round("comm_specialist", "newer finding", 0.1),
    ]

    block = _findings(coord)

    assert block.index("newer finding") < block.index("older finding")


def test_specialist_findings_render_a_harvested_finding_once(coord: Coordinator) -> None:
    """Harvesting copies a round's sourced findings into research hints; the block still shows each once."""
    from hyperloom.inference_optimizer.baseline_comparison import research_hints

    finding = {"what": "ngram spec decode stacks with expert parallel", "source": "measured r41"}
    coord.shared_state.specialist_rounds = [_round("serving_specialist", finding, 0.8)]
    research_hints.append_hints(coord.session_dir, [finding])

    assert _findings(coord).count("ngram spec decode stacks with expert parallel") == 1


def test_specialist_findings_keep_the_newest_within_their_budget(coord: Coordinator, monkeypatch) -> None:
    from hyperloom.orchestrator.loop import conversation

    monkeypatch.setattr(conversation, "_FINDINGS_PROMPT_CHARS", 200)
    coord.shared_state.specialist_rounds = [
        _round("serving_specialist", f"finding-{i:02d} " + "x" * 40, 0.5) for i in range(10)
    ]

    block = _findings(coord)

    assert "finding-09" in block
    assert "finding-00" not in block
    assert "older omitted)" in block


def test_specialist_findings_skip_rows_carrying_neither_findings_nor_questions(coord: Coordinator) -> None:
    coord.shared_state.specialist_rounds = [
        {"domain": "serving_specialist", "new_findings": [], "residual_questions": []},
        "not a dict",
    ]

    assert _findings(coord) == ""
    assert coord.conversation._recent_proposed_variants() == []


def test_priors_match_advisory_reads_the_proposals_of_both_recent_rounds(coord: Coordinator) -> None:
    import asyncio

    from hyperloom.inference_optimizer.baseline_comparison import research_hints

    research_hints.append_hints(coord.session_dir, [{"what": "cudagraph decode capture", "source": "measured r7"}])
    coord.shared_state.specialist_rounds = [
        {"proposal_set": [{"name": "graph_a", "description": "cudagraph decode"}]},
        {"proposal_set": [{"name": "graph_b", "description": "cudagraph decode"}, "not-a-dict"]},
    ]

    prompt = asyncio.run(coord.conversation.compose_prompt()).text

    assert "- graph_a: hint:cudagraph decode capture" in prompt
    assert "- graph_b: hint:cudagraph decode capture" in prompt


# -- warm recipe + workload tags ------------------------------------------
def test_warm_recipe_proven_items(coord: Coordinator) -> None:
    coord.shared_state.warm_start_recipe = {}
    assert coord.phase_prelude.warm_recipe_proven_items() == []
    coord.shared_state.warm_start_recipe = {
        "recipe": {
            "attrs": {
                "what_worked": [
                    {"name": "fp8", "source": "kb"},
                    {"name": "", "source": "skip"},
                    "not-a-dict",
                ]
            }
        },
    }
    out = coord.phase_prelude.warm_recipe_proven_items()
    assert out == [{"name": "fp8", "source": "kb"}]


def test_collect_workload_tags(coord: Coordinator, monkeypatch) -> None:
    """The whole tag dict, because these keys are the warm-start reader's filter."""
    monkeypatch.delenv("EP", raising=False)
    ss = coord.shared_state
    ss.framework = "sglang"
    ss.model_class = "moe"
    ss.model_name = "Qwen3-32B"
    ss.precision = "fp8"
    ss.tp = 8
    ss.ep = 4
    ss.conc = 64
    ss.isl = 1024
    ss.osl = 128
    ss.max_model_len = 8192
    ss.stack_fingerprint_meta = {
        "sglang": "0.4.6",
        "rocm": "6.3.1",
        "aiter": "0.1.2",
        "image_digest": "sha256:abc",
    }
    ss.baseline_workload_extra = {
        "max_running_requests": 256,
        "max_num_seqs": 128,
        # A disabled toggle is a tag, not an absence.
        "chunked_prefill_enabled": False,
        "enable_torch_compile": True,
        "quant_scheme": "  fp8_block  ",
        "workload_mode": "serving",
    }

    assert workload_kb_tags(coord.shared_state) == {
        "framework": "sglang",
        "model_class": "moe",
        "model_name": "Qwen3-32B",
        "precision": "fp8",
        "tp": 8,
        "ep": 4,
        "conc": 64,
        "isl": 1024,
        "osl": 128,
        "max_model_len": 8192,
        "framework_version": "0.4.6",
        "rocm_version": "6.3.1",
        "aiter_version": "0.1.2",
        "image_digest": "sha256:abc",
        "max_running_requests": 256,
        "max_num_seqs": 128,
        "chunked_prefill_enabled": False,
        "enable_torch_compile": True,
        "quant_scheme": "fp8_block",
        "workload_mode": "serving",
    }


def test_workload_tags_read_ep_from_the_env_only_without_a_shared_state_value(coord: Coordinator, monkeypatch) -> None:
    """``$EP`` is the legacy-SDK fallback."""
    monkeypatch.setenv("EP", "2")
    ss = coord.shared_state
    ss.ep = 8

    tags = workload_kb_tags(coord.shared_state)
    assert tags["ep"] == 8

    ss.ep = 0
    assert workload_kb_tags(coord.shared_state)["ep"] == 2


def test_an_unparseable_or_non_positive_parallelism_env_is_not_tagged(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.setenv("EP", "two")

    tags = workload_kb_tags(coord.shared_state)
    assert "ep" not in tags


def test_workload_tags_survive_a_resumed_null_fingerprint_meta(coord: Coordinator, monkeypatch) -> None:
    """``state.json`` is a system boundary: a null where a dict is declared must not raise here."""
    monkeypatch.delenv("EP", raising=False)
    ss = coord.shared_state
    ss.framework = "vllm"
    ss.model_class = ""
    ss.stack_fingerprint_meta = None
    ss.baseline_workload_extra = None

    assert workload_kb_tags(coord.shared_state) == {"framework": "vllm"}


def test_unknown_and_empty_fingerprint_entries_are_not_tagged(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.delenv("EP", raising=False)
    ss = coord.shared_state
    ss.framework = "vllm"
    ss.model_class = ""
    ss.stack_fingerprint_meta = {
        "vllm": "unknown",
        "rocm": "",
        "aiter": "unknown",
        "image_digest": "sha256:abc",
    }

    tags = workload_kb_tags(coord.shared_state)
    assert tags == {"framework": "vllm", "image_digest": "sha256:abc"}


# -- phase denial gate -----------------------------------------------------
def test_the_coordinator_revalidation_baseline_is_not_phase_denied(coord: Coordinator) -> None:
    """The revalidation pump prices its own action and never runs the phase gate."""
    coord.shared_state.phase = "ENABLEMENT"
    assert coord.dispatcher.time_budget_denial_for_action("baseline") is None
    assert coord.dispatcher.admission_denial_for_action("baseline") is not None


# -- sequence denial gates -------------------------------------------------
def test_admission_denial_for_request(coord: Coordinator) -> None:
    ss = coord.shared_state
    ss.stop_reason = ""
    ss.baseline_tput = 0.0
    # non-kernel target -> not gated
    assert coord.dispatcher.admission_denial_for_request("orchestration", "anything") is None
    # trace_analyze always allowed
    assert coord.dispatcher.admission_denial_for_request("kernel_agent", "trace_analyze") is None
    # unknown handler kind -> not gated
    assert coord.dispatcher.admission_denial_for_request("kernel_agent", "no_such_kind") is None


def test_a_kernel_request_the_phase_cannot_fit_is_refused(coord: Coordinator, monkeypatch: pytest.MonkeyPatch) -> None:
    """A request runs its handler inline, so admission is the only point the phase budget can refuse it."""
    _phase_with_minutes_left(coord, monkeypatch, "KERNEL_AGENT", 10)

    denied = coord.dispatcher.admission_denial_for_request("kernel_agent", "apply_patch")

    assert denied is not None and denied.rule == "phase_time_budget"
    assert "action='integrate'" in str(denied)


def test_gemm_tuning_required_before_kernel_opt(coord: Coordinator, monkeypatch) -> None:
    monkeypatch.delenv("INFERENCE_OPTIMIZER_SKIP_GEMM_TUNING", raising=False)
    ss = coord.shared_state
    ss.last_gemm_tuning = {}
    # Any precision on a supported framework is eligible.
    ss.framework = "sglang"
    ss.precision = "fp16"
    assert coord.phase_kernel._gemm_tuning_required_before_kernel_opt() is True
    ss.precision = "bf16"
    assert coord.phase_kernel._gemm_tuning_required_before_kernel_opt() is True
    # Unsupported framework -> not eligible.
    ss.framework = "trt-llm"
    assert coord.phase_kernel._gemm_tuning_required_before_kernel_opt() is False
    # Supported framework + terminal status -> not required.
    ss.framework = "sglang"
    ss.precision = "fp8"
    ss.last_gemm_tuning = {"status": "succeeded"}
    assert coord.phase_kernel._gemm_tuning_required_before_kernel_opt() is False


# -- framework candidate selection -------------------------------------
def test_discovery_drops_prs_already_batched(coord: Coordinator) -> None:
    from types import SimpleNamespace

    ss = coord.shared_state
    ss.framework_agent_batches = [{"candidates": [{"candidate_id": "https://x/pr/1"}, {"pr_url": "https://x/pr/2"}]}]
    discovery = SimpleNamespace(task_id="disc-1", params={"candidate_discovery": True})
    proposals = [{"pr_url": f"https://x/pr/{n}", "verdict": "worth_a_bench"} for n in (1, 2, 3)]

    coord.phase_framework.on_specialist_settled(discovery, {"proposal_set": proposals}, run_error="")

    assert [c["pr_url"] for c in ss.framework_agent_batches[-1]["candidates"]] == ["https://x/pr/3"]
