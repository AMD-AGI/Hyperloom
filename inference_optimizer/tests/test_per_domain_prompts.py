"""PR-A6 (Arbor-into-Hyperloom): per-domain specialist prompt templates.

The v0.8 M5 ``specialist_prompt_builder`` shipped one template
(``serving_specialist``) and let the other five domains fall back
to the generic identity. PR-A6 ports Arbor's "agent expertise" table
into per-domain focus templates so each specialist starts with the
right corner of the search space (KB anchor, source roots, winning
techniques, pitfalls) baked into Section 1 of the system prompt.

These tests pin three contracts:

* All six domains have a per-domain focus template.
* The rendered prompt for each domain mentions the domain's
  characteristic KB anchor + signature techniques (a coarse "have we
  actually customised this template?" check; not a snapshot diff).
* The M5 active set has been widened to cover all six domains, so
  ``SpecialistRunner`` no longer logs a "generic template" note for
  the kernel / comm / compiler / system / pr_intel domains.
"""

from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.specialist_domains import (
    SPECIALIST_DOMAIN_KEYS,
    SPECIALIST_DOMAINS,
    SPECIALIST_DOMAINS_M5,
    get_domain,
)
from inference_optimizer.orchestrator.system_prompts.specialist_prompt_builder import (
    _DOMAIN_FOCUS_TEMPLATES,
    SpecialistPromptInputs,
    build_specialist_prompts,
)


def _build(domain_key: str) -> str:
    domain = get_domain(domain_key)
    assert domain is not None, domain_key
    inp = SpecialistPromptInputs(
        task_id=f"task-{domain_key}",
        domain=domain,
        max_turns=4,
        gap_canonical_id=f"gap.{domain_key}.example",
        gap_symptom="example symptom",
        gap_layer=domain.layer,
        workspace_path=f"/tmp/test/{domain_key}",
    )
    system, user = build_specialist_prompts(inp)
    return system + "\n" + user


# ---------------------------------------------------------------------------
# 1. Coverage — every catalogue domain has a focus template
# ---------------------------------------------------------------------------
def test_every_domain_has_focus_template():
    for domain in SPECIALIST_DOMAINS:
        assert domain.key in _DOMAIN_FOCUS_TEMPLATES, (
            f"missing per-domain template for {domain.key!r}"
        )


def test_specialist_domains_m5_covers_all_six():
    """PR-A6 widened the M5 active set so SpecialistRunner no longer
    logs ``generic prompt template`` notes for M6-only domains."""
    assert SPECIALIST_DOMAINS_M5 == SPECIALIST_DOMAIN_KEYS
    assert len(SPECIALIST_DOMAINS_M5) == 6


# ---------------------------------------------------------------------------
# 2. Per-domain content checks — each template mentions its signature
# ---------------------------------------------------------------------------
def test_serving_specialist_mentions_scheduler_and_kv_cache():
    text = _build("serving_specialist")
    for marker in (
        "serving_specialist", "scheduler", "cuda_graph", "kv_cache",
        "max-num-seqs",
    ):
        assert marker.lower() in text.lower(), f"missing {marker!r}"


def test_kernel_switch_specialist_mentions_aiter_and_attention_backends():
    text = _build("kernel_switch_specialist")
    for marker in (
        "kernel_switch_specialist", "aiter", "ROCM_AITER_MLA", "TRITON_MLA",
        "CDNA3",
    ):
        assert marker.lower() in text.lower(), f"missing {marker!r}"


def test_comm_specialist_mentions_quickreduce_and_topology():
    text = _build("comm_specialist")
    for marker in (
        "comm_specialist", "QuickReduce", "allreduce", "RCCL",
        "NCCL_MIN_NCHANNELS",
    ):
        assert marker.lower() in text.lower(), f"missing {marker!r}"


def test_compiler_specialist_mentions_torch_compile_and_triton():
    text = _build("compiler_specialist")
    for marker in (
        "compiler_specialist", "torch.compile", "inductor", "triton",
        "AMDGCN", "num_warps",
    ):
        assert marker.lower() in text.lower(), f"missing {marker!r}"


def test_system_specialist_mentions_kfd_and_rocm_smi():
    text = _build("system_specialist")
    for marker in (
        "system_specialist", "KFD", "rocm-smi", "HSA_ENABLE_SDMA",
        "numactl",
    ):
        assert marker.lower() in text.lower(), f"missing {marker!r}"


def test_pr_intel_specialist_mentions_cross_repo_research():
    text = _build("pr_intel_specialist")
    for marker in (
        "pr_intel_specialist", "cross-repo", "mcp__pr_monitor",
        "ROCm/aiter", "do NOT propose source patches",
    ):
        assert marker.lower() in text.lower(), f"missing {marker!r}"


# ---------------------------------------------------------------------------
# 3. SpecialistRunner no longer marks any domain as "generic template"
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_runner_does_not_log_generic_template_for_any_domain(tmp_path):
    """Defensive: when the M5 active set covers a domain, the runner
    must NOT add a ``post-M5 ... generic prompt template`` note to its
    SpecialistRunResult.notes."""
    from inference_optimizer.orchestrator.backends.mock_backend import (
        MockBackend, MockTurn, ScriptedPlan,
    )
    from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
    from inference_optimizer.orchestrator.specialist_runner import SpecialistRunner
    from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext
    from inference_optimizer.orchestrator.task_registry import Task

    done = {
        "gap_canonical_id": "gap.x",
        "domain": "kernel_switch_specialist",
        "proposal_set": [],
        "empty": True,
        "summary": "test",
        "reason": "test",
        "confidence": 0.0,
        "new_findings": [],
        "residual_questions": [],
    }
    plan = ScriptedPlan(turns=[
        MockTurn(intents=[Intent(type=IntentType.SPECIALIST_DONE, payload=done)]),
    ])
    runner = SpecialistRunner(
        backend_factory=lambda d: MockBackend(plan, name="mock"),
        session_dir=tmp_path,
        default_max_turns=2,
    )
    task = Task(
        task_id="t-kernel",
        kind="specialist",
        state="queued",
        params={
            "domain": "kernel_switch_specialist",
            "gap_canonical_id": "gap.x",
            "max_turns": 2,
        },
        idempotency_key="t-kernel",
        requires_lanes=tuple(),
    )
    ctx = RunnerContext(task=task, lease=None, extra={})
    result = await runner.run(ctx)
    for note in result.notes or []:
        assert "generic prompt template" not in note, (
            f"PR-A6 should have widened SPECIALIST_DOMAINS_M5 to cover "
            f"kernel_switch_specialist; got note={note!r}"
        )
