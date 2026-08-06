# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the ``framework_rewrite_specialist`` domain and its dispatch.

Three things are pinned here:

* the domain exists in the catalogue and resolves to the ``framework`` KB anchor,
  so PolicyGate's anchor whitelist accepts a dispatch to it;
* the prompt carries the rewrite-pattern taxonomy and the switch-manifest
  contract as a *prior*, without naming any specific function — that split is
  what makes the capability transfer to another model or framework instead of
  being a one-off reproduction;
* the FRAMEWORK phase routes to it by framework *kind*, so a scriptable
  iterative pipeline is not dispatched to a serving domain whose hot path
  (scheduler, batching, KV-cache admission) does not exist there.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.prompts.specialist_prompt_builder import (
    SpecialistPromptInputs,
    _DOMAIN_FOCUS_TEMPLATES,
    _focus_framework_rewrite_specialist,
)
from hyperloom.orchestrator.specialists.domains import (
    SPECIALIST_DOMAIN_KEYS,
    get_domain,
    normalize_dispatch_tags,
)


DOMAIN_KEY = "framework_rewrite_specialist"


# --------------------------------------------------------------------------
# catalogue
# --------------------------------------------------------------------------


def test_domain_is_in_the_catalogue():
    """The domain is dispatchable and its focus template is registered."""
    assert DOMAIN_KEY in SPECIALIST_DOMAIN_KEYS
    assert DOMAIN_KEY in _DOMAIN_FOCUS_TEMPLATES


def test_domain_maps_to_the_framework_kb_anchor():
    """PolicyGate validates the KB anchor, so a wrong one denies every dispatch."""
    domain = get_domain(DOMAIN_KEY)
    assert domain is not None
    assert domain.kb_anchor == "framework"
    assert normalize_dispatch_tags({"domain": DOMAIN_KEY}) == ["framework"]


def test_domain_description_separates_it_from_the_serving_domain():
    """The description says what makes this domain distinct, not just that it exists."""
    domain = get_domain(DOMAIN_KEY)
    assert domain is not None
    text = domain.description.lower()
    assert "iterative" in text
    assert "serving_specialist" in text


# --------------------------------------------------------------------------
# prompt
# --------------------------------------------------------------------------


def _focus_text(framework: str = "worldplay") -> str:
    """Render the domain focus block as one string.

    Args:
        framework: Framework name carried into the prompt.

    Returns:
        The rendered focus block.
    """
    inputs = SpecialistPromptInputs(task_id="t-1", domain=get_domain(DOMAIN_KEY), framework=framework)
    return "\n".join(_focus_framework_rewrite_specialist(inputs))


@pytest.mark.parametrize(
    "category",
    [
        "memoize",
        "hoist",
        "host round-trip",
        "fuse",
        "vendor kernel",
        "resident",
        "no-op glue",
    ],
)
def test_prompt_carries_every_taxonomy_category(category):
    """All seven rewrite patterns land, because the taxonomy is the prior."""
    assert category in _focus_text()


def test_prompt_states_the_switch_manifest_contract():
    """The manifest fields are named, since Stage 3 parses exactly these keys."""
    text = _focus_text()
    for field in ("framework_switches", "switch", "category", "target", "evidence", "depends_on", "enables"):
        assert field in text


def test_prompt_explains_why_the_default_off_switch_is_enforced():
    """A rule with a stated reason survives an LLM's judgement; a bare rule may not."""
    text = _focus_text()
    assert "defaults OFF" in text or "default-off" in text
    assert "parity leg" in text
    assert "independently measurable" in text


def test_prompt_warns_about_allocator_address_reuse():
    """The pinning requirement is the difference between a cache and a silent bug.

    Under a caching allocator a freed tensor's address is handed to the next
    allocation, so a cache keyed on ``data_ptr`` alone returns a previous
    computation for a brand-new tensor — a wrong-answer bug a throughput
    benchmark accepts happily.
    """
    text = _focus_text()
    assert "caching" in text and "allocator" in text
    assert "Pin the source tensors" in text


def test_prompt_explains_the_enabler_stakes():
    """The enabler contract is stated with its consequence, not just its syntax."""
    text = _focus_text()
    assert "enabler" in text
    assert "standalone" in text


def test_prompt_forbids_content_hashing_for_cache_keys():
    """Hashing contents to build a key reintroduces the sync being removed."""
    assert "Do NOT hash tensor *contents*" in _focus_text()


def test_prompt_requires_a_fallback_path():
    """A guard plus fallback keeps an unexpected input slow rather than wrong."""
    text = _focus_text()
    assert "fallback" in text.lower()
    assert "correctness" in text


def test_prompt_names_the_active_framework():
    """The block is framework-aware so the specialist knows what it is reading."""
    assert "worldmirror" in _focus_text("worldmirror")


def test_prompt_names_no_specific_function():
    """The prior is the pattern vocabulary, not a list of answers.

    Seeding specific landing points would reproduce one known result and teach
    the system nothing transferable; the whole value of the taxonomy is that it
    applies to a pipeline nobody has looked at yet.
    """
    text = _focus_text()
    for leaked_answer in (
        "all_gather_object",
        "_prepare_apply_fns",
        "sequence_parallel_attention",
        "WORLDPLAY_",
        "hyvideo",
    ):
        assert leaked_answer not in text


# --------------------------------------------------------------------------
# reference document
# --------------------------------------------------------------------------


def _reference_path() -> Path:
    """Return the bundled rewrite-pattern reference path."""
    from hyperloom.inference_optimizer.session.paths import asset_root

    return asset_root() / "references" / "framework_rewrite_patterns.md"


def test_reference_document_is_bundled():
    """The long-form taxonomy ships so a specialist can read past the summary."""
    assert _reference_path().is_file()


def test_reference_document_covers_every_category_id():
    """Category ids match the aggregator's, so evidence maps onto the reference."""
    from hyperloom.orchestrator.actions.executors import _framework_rewrite_evidence as ev

    text = _reference_path().read_text(encoding="utf-8")
    for category in (
        ev.CATEGORY_MEMOIZE,
        ev.CATEGORY_HOIST,
        ev.CATEGORY_HOST_ROUND_TRIP,
        ev.CATEGORY_HOST_SYNC,
        ev.CATEGORY_FUSE_COLLECTIVES,
        ev.CATEGORY_DEVICE_RESIDENT,
    ):
        assert category in text


# --------------------------------------------------------------------------
# phase routing
# --------------------------------------------------------------------------


class _State:
    """Minimal SharedState stand-in for the domain router."""

    def __init__(self, framework: str, evidence: str = "") -> None:
        self.framework = framework
        self.last_framework_rewrite_evidence = evidence


class _Phase:
    """Bind the two FrameworkPhase helpers under test to a stub state."""

    def __init__(self, framework: str, evidence: str = "") -> None:
        from hyperloom.orchestrator.phases.framework import FrameworkPhase

        self.shared_state = _State(framework, evidence)
        self._authoring_specialist_domain = FrameworkPhase._authoring_specialist_domain.__get__(self)
        self._render_rewrite_evidence_for_prompt = FrameworkPhase._render_rewrite_evidence_for_prompt.__get__(self)


@pytest.mark.parametrize("framework", ["worldplay", "worldmirror", "xdit", "hunyuan_image3"])
def test_scriptable_frameworks_route_to_the_rewrite_domain(framework):
    """A server-less iterative pipeline gets the rewrite domain."""
    assert _Phase(framework)._authoring_specialist_domain() == DOMAIN_KEY


@pytest.mark.parametrize("framework", ["sglang", "vllm", "atom"])
def test_serving_frameworks_keep_the_serving_domain(framework):
    """Routing is additive: the serving path is untouched."""
    assert _Phase(framework)._authoring_specialist_domain() == "serving_specialist"


@pytest.mark.parametrize("framework", ["", "  ", "something-unregistered"])
def test_unknown_framework_falls_back_to_serving(framework):
    """An unresolvable framework keeps the historical default rather than guessing."""
    assert _Phase(framework)._authoring_specialist_domain() == "serving_specialist"


def test_evidence_block_renders_from_the_recorded_path(tmp_path):
    """The arm hands the specialist the measured evidence, not just a gap phrase."""
    from hyperloom.orchestrator.actions.executors import _framework_rewrite_evidence as ev

    document = ev.build_evidence(
        [
            {
                "schema": "hyperloom.host_probe/1",
                "rank": 0,
                "wall_seconds": 100.0,
                "roots": ["/src/pipeline/"],
                "host_calls": [
                    {
                        "api": "torch.distributed.all_gather_object",
                        "site": "comm.py:60:exchange",
                        "count": 5000,
                        "wall_s": 40.0,
                        "bytes": 0,
                        "shape_sigs": [],
                        "callers": [],
                    }
                ],
                "framework_calls": [],
                "truncated": {},
                "notes": [],
            }
        ]
    )
    path = tmp_path / ev.EVIDENCE_FILENAME
    path.write_text(json.dumps(document), encoding="utf-8")

    text = _Phase("worldplay", str(path))._render_rewrite_evidence_for_prompt()
    assert "HOST-SIDE REWRITE EVIDENCE" in text
    assert ev.CATEGORY_HOST_ROUND_TRIP in text
    assert "comm.py:60:exchange" in text


def test_evidence_block_is_empty_without_a_recorded_path():
    """The arm can run before any profile has landed; that is not an error."""
    assert _Phase("worldplay")._render_rewrite_evidence_for_prompt() == ""


def test_evidence_block_tolerates_an_unreadable_path(tmp_path):
    """A stale or corrupt path degrades to no block rather than wedging the pump."""
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert _Phase("worldplay", str(broken))._render_rewrite_evidence_for_prompt() == ""


# --------------------------------------------------------------------------
# dispatch payload
# --------------------------------------------------------------------------


class _Tasks:
    """Record ``create_or_return_existing`` calls."""

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    async def create_or_return_existing(self, **kwargs: Any):  # noqa: ANN401
        """Record the dispatch and return a fresh-task sentinel."""
        self.created.append(kwargs)
        from types import SimpleNamespace

        return SimpleNamespace(task_id=f"t-{len(self.created)}", state="queued"), False


class _DispatchStub:
    """Drive ``_enqueue_framework_agent_local_explore_specialist`` in isolation."""

    def __init__(self, tmp_path: Path, framework: str, evidence: str = "") -> None:
        from hyperloom.orchestrator.phases.framework import FrameworkPhase

        self.session_dir = tmp_path
        self.tasks = _Tasks()
        self.shared_state = _State(framework, evidence)
        self.shared_state.framework_agent_phase_progress = []
        self.shared_state.framework_agent_specialist_candidate_map = {}
        self.shared_state.save = lambda _dir: None
        for name in (
            "_authoring_specialist_domain",
            "_render_rewrite_evidence_for_prompt",
            "_enqueue_framework_agent_local_explore_specialist",
            "_next_local_explore_candidate_id",
        ):
            setattr(self, name, getattr(FrameworkPhase, name).__get__(self))
        # A staticmethod on the real class; binding it would pass ``self`` as the
        # candidate row.
        self._framework_candidate_key = FrameworkPhase._framework_candidate_key

    def _cycle_idem_suffix(self) -> str:
        """Macro-cycle 0, as the Coordinator would report it."""
        return ""

    def _render_framework_memory_for_prompt(self, _memory) -> str:  # noqa: ANN001
        """Suppress the working-memory block; not under test here."""
        return ""

    def _build_framework_working_memory(self) -> dict:
        """Suppress the working-memory block; not under test here."""
        return {}

    def _framework_gpu_params(self) -> dict:
        """Provide no GPU params; not under test here."""
        return {}

    def _framework_authoring_lanes_ttl(self, _params, *, base_ttl_sec: int) -> tuple[list[str], int]:  # noqa: ANN001
        """Provide fixed lanes/TTL; lane accounting is not under test here."""
        return [], base_ttl_sec

    async def _warm_specialist_params(self, _params) -> None:  # noqa: ANN001
        """Skip warm-start enrichment; not under test here."""
        return None


def _dispatch(tmp_path: Path, framework: str, evidence: str = "") -> dict[str, Any]:
    """Dispatch a local-explore specialist and return the task params.

    Args:
        tmp_path: Session directory.
        framework: Session framework.
        evidence: Recorded rewrite-evidence path.

    Returns:
        The params of the created specialist task.
    """
    import asyncio

    stub = _DispatchStub(tmp_path, framework, evidence)
    candidate = {
        "kind": "local_explore",
        "candidate_id": "local_explore:0",
        "framework": framework,
        "gap_description": "improve throughput",
        "gap_canonical_id": "local_explore",
    }
    asyncio.run(stub._enqueue_framework_agent_local_explore_specialist(candidate))
    assert stub.tasks.created, "no specialist was dispatched"
    return stub.tasks.created[0]["params"]


def test_scriptable_dispatch_demands_a_switch_manifest(tmp_path):
    """The scriptable arm's mandate is a patch *plus* a manifest, not either/or.

    Making the manifest optional would leave the whole attribution and
    composition mechanism dependent on an LLM choosing to opt into it.
    """
    params = _dispatch(tmp_path, "worldplay")
    assert params["domain"] == DOMAIN_KEY
    notes = params["notes"]
    assert "framework_switches" in notes
    assert "default-off environment switch" in notes
    assert "depends_on" in notes and "enables" in notes


def test_scriptable_dispatch_drops_the_serving_hot_path_language(tmp_path):
    """A pipeline with no scheduler must not be pointed at scheduling."""
    notes = _dispatch(tmp_path, "worldplay")["notes"]
    assert "KV-cache / scheduling" not in notes
    assert "iterative loop" in notes


def test_serving_dispatch_is_unchanged(tmp_path):
    """The serving arm keeps its domain, mandate and either/or deliverable."""
    params = _dispatch(tmp_path, "sglang")
    assert params["domain"] == "serving_specialist"
    notes = params["notes"]
    assert "MoE / FP8 / attention / GEMM / KV-cache / scheduling" in notes
    assert "framework_switches" not in notes


def test_scriptable_dispatch_inlines_the_measured_evidence(tmp_path):
    """When evidence exists it travels with the dispatch."""
    from hyperloom.orchestrator.actions.executors import _framework_rewrite_evidence as ev

    document = ev.build_evidence(
        [
            {
                "schema": "hyperloom.host_probe/1",
                "rank": 0,
                "wall_seconds": 100.0,
                "roots": [],
                "host_calls": [
                    {
                        "api": "torch.Tensor.item",
                        "site": "attn.py:12:unpad",
                        "count": 9000,
                        "wall_s": 12.0,
                        "bytes": 0,
                        "shape_sigs": [],
                        "callers": [],
                    }
                ],
                "framework_calls": [],
                "truncated": {},
                "notes": [],
            }
        ]
    )
    path = tmp_path / ev.EVIDENCE_FILENAME
    path.write_text(json.dumps(document), encoding="utf-8")
    notes = _dispatch(tmp_path, "worldplay", str(path))["notes"]
    assert "HOST-SIDE REWRITE EVIDENCE" in notes
    assert "attn.py:12:unpad" in notes


def test_scriptable_dispatch_without_evidence_says_how_to_look(tmp_path):
    """No evidence yet still yields an actionable instruction, not silence."""
    notes = _dispatch(tmp_path, "worldplay")["notes"]
    assert "No host-side rewrite evidence has been collected yet" in notes
    assert "can change across iterations" in notes


# --------------------------------------------------------------------------
# scriptable source-root registration
# --------------------------------------------------------------------------


def test_materialization_publishes_the_checkout_to_the_orchestrator_env(tmp_path, monkeypatch):
    """The resolved checkout must reach PolicyGate, not only the benchmark process.

    Writing it into the materialized YAML reaches the benchmark subprocess, but
    PolicyGate runs in the orchestrator, and it is the orchestrator's environment
    that ``resolve_source_file_allowlist`` reads. Without this the source root is
    missing and every authored patch against the framework's own code is rejected
    on a session that otherwise looks correctly configured.
    """
    from hyperloom.orchestrator.actions.executors._workload_envs import (
        _apply_worldplay_runtime_defaults,
    )

    checkout = tmp_path / "HY-WorldPlay"
    (checkout / "hyvideo").mkdir(parents=True)
    for var in ("WORLDPLAY_REPO_PATH", "WORLDPLAY_DIR", "WORLDPLAY_REPO"):
        monkeypatch.delenv(var, raising=False)

    bench = {"framework": "worldplay"}
    envs = {"WORLDPLAY_REPO_PATH": str(checkout)}
    _apply_worldplay_runtime_defaults(
        bench,
        envs,
        gpu_type="mi355x",
        explicit_benchmark_script=False,
    )

    import os

    assert os.environ["WORLDPLAY_REPO_PATH"] == str(checkout)
    assert os.environ["WORLDPLAY_DIR"] == str(checkout)

    from hyperloom.orchestrator.framework.paths import resolve_source_file_allowlist

    assert any("HY-WorldPlay" in root for root in resolve_source_file_allowlist())


def test_publishing_never_overrides_an_operator_value(monkeypatch, tmp_path):
    """An explicitly exported checkout wins; Hyperloom only fills a gap."""
    from hyperloom.orchestrator.actions.executors._workload_envs import (
        _publish_scriptable_repo_root,
    )

    monkeypatch.setenv("WORLDPLAY_REPO_PATH", "/operator/checkout")
    monkeypatch.delenv("WORLDPLAY_DIR", raising=False)
    _publish_scriptable_repo_root("worldplay", str(tmp_path / "cache" / "HY-WorldPlay"))

    import os

    assert os.environ["WORLDPLAY_REPO_PATH"] == "/operator/checkout"
    # The unset alias is still filled, so the two agree.
    assert os.environ["WORLDPLAY_DIR"].endswith("HY-WorldPlay")


def test_publishing_ignores_blank_input(monkeypatch):
    """A blank path would make the allowlist claim a root that does not exist."""
    from hyperloom.orchestrator.actions.executors._workload_envs import (
        _publish_scriptable_repo_root,
    )

    import os

    monkeypatch.delenv("WORLDPLAY_REPO_PATH", raising=False)
    _publish_scriptable_repo_root("worldplay", "   ")
    _publish_scriptable_repo_root("", "/some/path")
    assert "WORLDPLAY_REPO_PATH" not in os.environ


def test_frameworks_reference_documents_the_rewrite_path():
    """The launch reference has to explain the switch contract and its enforcement.

    An operator reading only this file needs to know that rewrites are default-off,
    that parity is checked, and why an unprofitable bundle is kept rather than
    reverted — otherwise ``kept_inert`` looks like a bug.
    """
    from hyperloom.inference_optimizer.session.paths import asset_root

    text = (asset_root() / "references" / "frameworks.md").read_text(encoding="utf-8")
    assert "framework_rewrite_specialist" in text
    assert "framework_switches" in text
    assert "switch-off parity" in text
    assert "kept inert" in text
    assert "framework_rewrite_patterns.md" in text
