"""Hermetic tests for standalone FlyDSL rewrite KB reuse."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelforge.config import Config
from kernelforge.config import KnowledgeConfig, KnowledgeStoreMode
from kernelforge.knowledge.kb_store.identity.kernel_recipe import (
    KernelRecipeIdentity,
    kernel_recipe_canonical_id,
)
from kernelforge.workflows.flydsl_rewrite import driver_contract, identity as rewrite_identity, kb
from kernelforge.knowledge.kb_store import record_store
from kernelforge.knowledge.kb_store.recipe import (
    KernelRecipeKB,
    candidate_session_id as build_candidate_session_id,
)
from kernelforge.workflows.flydsl_rewrite.identity import (
    framework_version,
    segment,
)
from kernelforge.workflows.flydsl_rewrite.measurement import FlydslMeasurement
from kernelforge.tests.knowledge.kb_store._in_memory_store import (
    InMemoryKBStore,
    _remote_config,
    _use_in_memory_kb_store,
)
from kernelforge.tests.workflows.flydsl_rewrite._rewrite_spec import _spec

VLLM_VERSION = framework_version("vllm")
SOFTMAX_IDENTITY = f"kernel:flydsl:softmax:vllm:{VLLM_VERSION}:flydsl:mi355x"


def _local_config(tmp_path, spec, **knowledge_kwargs):
    knowledge = KnowledgeConfig.from_env(
        {},
        mode="local",
        local_root=tmp_path / "local-knowledge",
        **knowledge_kwargs,
    )
    return knowledge, Config.from_env(
        workspace=spec.workspace,
        gpu_target="gfx950",
        gpu_type="mi355x",
        knowledge_config=knowledge,
        agent_precheck=False,
    )


# One scored case keeps a stubbed measurement equal to its own aggregate.
_CASE = "m_1"


def _passing_validation(monkeypatch, *, best_ms, snr_db=80.0):
    class Report:
        all_passed = True
        results = [type("Result", (), {"snr_db": snr_db})()]

    async def validation(**_kwargs):
        return Report()

    monkeypatch.setattr(kb, "run_validation_pipeline", validation)
    monkeypatch.setattr(
        kb.driver_contract,
        "preflight_candidate",
        lambda *args, **kwargs: driver_contract.PreflightReport(
            ok=True,
            timing_ms=best_ms,
            case_ms={_CASE: best_ms},
        ),
    )


def test_rewrite_warmstart_uses_fuzzy_identity_after_exact_miss(
    tmp_path,
    monkeypatch,
):
    _use_in_memory_kb_store(monkeypatch)
    spec, _driver = _spec(tmp_path)
    config = _remote_config(tmp_path)
    donor = KernelRecipeIdentity(
        producer="flydsl",
        kernel_name="softmax",
        framework="vllm",
        framework_version="1.0.0",
        backend="flydsl",
        gpu="mi300x",
    )
    KernelRecipeKB.open_identity(donor, config).write_candidate(
        {"flydsl_kernel": "kernel.py"},
        files={"kernel.py": spec.flydsl_kernel},
        speedup=2.0,
    )
    monkeypatch.setattr(
        rewrite_identity,
        "framework_version",
        lambda _framework: "2.0.0",
    )

    plan = kb._read_top_candidates(
        spec,
        config,
        top_k=3,
    )

    assert plan.read_reason == "hit"
    assert [candidate["canonical_id"] for candidate in plan.candidates] == [kernel_recipe_canonical_id(donor)]


def test_rewrite_warmstart_keeps_exact_identity_ahead_of_fuzzy_donor(
    tmp_path,
    monkeypatch,
):
    _use_in_memory_kb_store(monkeypatch)
    spec, _driver = _spec(tmp_path)
    config = _remote_config(tmp_path)
    exact_identity = KernelRecipeIdentity(
        producer="flydsl",
        kernel_name="softmax",
        framework="vllm",
        framework_version=VLLM_VERSION,
        backend="flydsl",
        gpu="mi355x",
    )
    fuzzy_identity = KernelRecipeIdentity(
        producer="flydsl",
        kernel_name="softmax",
        framework="vllm",
        framework_version="99.0.0",
        backend="flydsl",
        gpu="mi300x",
    )
    for identity, tag, speedup in (
        (exact_identity, "exact", 1.1),
        (fuzzy_identity, "fuzzy", 9.0),
    ):
        KernelRecipeKB.open_identity(identity, config).write_candidate(
            {"flydsl_kernel": "kernel.py", "tag": tag},
            files={"kernel.py": spec.flydsl_kernel},
            speedup=speedup,
        )

    plan = kb._read_top_candidates(
        spec,
        config,
        top_k=3,
    )

    assert [candidate["attrs"]["tag"] for candidate in plan.candidates] == ["exact"]


# --------------------------------------------------------------------------- # identity
# --------------------------------------------------------------------------- #
def test_a_rewrite_is_filed_under_the_flydsl_producer_identity(tmp_path, monkeypatch):
    store = _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)

    written = kb.write_flydsl_kb_solution(
        kb.RewriteKbTarget(spec, str(driver), _remote_config(tmp_path)), FlydslMeasurement(10.0, 5.0, 2, snr_db=80.0)
    )

    assert written["written"] is True
    assert written["canonical_id"] == SOFTMAX_IDENTITY
    assert list(store.knowledge) == [(SOFTMAX_IDENTITY, written["session_id"])]
    identity = store.knowledge[(SOFTMAX_IDENTITY, written["session_id"])]["identity"]
    assert identity == {
        "producer": "flydsl",
        "kernel_name": "softmax",
        "gpu": "mi355x",
        "framework": "vllm",
        "framework_version": VLLM_VERSION,
        "backend": "flydsl",
    }
    assert store.knowledge[(SOFTMAX_IDENTITY, written["session_id"])]["producer"] == "flydsl"


def test_a_namespaced_operator_name_stays_out_of_the_identifiers(
    tmp_path,
    monkeypatch,
):
    # A logical name carries separators the identity and the session id both use, so an unnormalized one would let the
    # operator re-partition either of them.
    store = _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    spec.op_name = "vllm::softmax"

    written = kb.write_flydsl_kb_solution(
        kb.RewriteKbTarget(spec, str(driver), _remote_config(tmp_path)), FlydslMeasurement(10.0, 5.0, 2, snr_db=80.0)
    )

    assert written["written"] is True
    assert "::" not in written["session_id"]
    assert written["canonical_id"].count(":") == SOFTMAX_IDENTITY.count(":")
    identity = store.knowledge[(written["canonical_id"], written["session_id"])]
    assert ":" not in identity["identity"]["kernel_name"]


def test_the_same_port_on_another_gpu_is_a_different_identity(tmp_path, monkeypatch):
    store = _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    knowledge = KnowledgeConfig.from_env(
        {},
        mode="remote",
        local_root=tmp_path / "knowledge",
        kb_store_url="http://in-memory",
        kb_store_token="token",
    )
    other_gpu = Config.from_env(
        workspace=str(tmp_path),
        gpu_target="gfx950",
        gpu_type="mi300x",
        knowledge_config=knowledge,
        agent_precheck=False,
    )

    kb.write_flydsl_kb_solution(
        kb.RewriteKbTarget(spec, str(driver), _remote_config(tmp_path)), FlydslMeasurement(10.0, 5.0, 2)
    )
    kb.write_flydsl_kb_solution(kb.RewriteKbTarget(spec, str(driver), other_gpu), FlydslMeasurement(10.0, 5.0, 2))

    assert sorted({identity for identity, _ in store.knowledge}) == [
        f"kernel:flydsl:softmax:vllm:{VLLM_VERSION}:flydsl:mi300x",
        SOFTMAX_IDENTITY,
    ]
    # Artifact keys are partitioned by session id alone, so two identities sharing one would put both ports on one
    # object and let the second overwrite the first.
    assert len({session for _, session in store.knowledge}) == 2


def test_gpu_target_does_not_change_the_recipe_identity(tmp_path, monkeypatch):
    store = _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    gfx950 = _remote_config(tmp_path)
    gfx942 = _remote_config(tmp_path)
    gfx942.gpu_target = "gfx942"

    first = kb.write_flydsl_kb_solution(
        kb.RewriteKbTarget(spec, str(driver), gfx950), FlydslMeasurement(10.0, 5.0, 2), session_key="a" * 40
    )
    second = kb.write_flydsl_kb_solution(
        kb.RewriteKbTarget(spec, str(driver), gfx942), FlydslMeasurement(10.0, 5.0, 2), session_key="b" * 40
    )

    assert first["canonical_id"] == second["canonical_id"] == SOFTMAX_IDENTITY
    assert {document["value"]["metric"]["gpu_arch"] for document in store.knowledge.values()} == {"gfx942", "gfx950"}


def test_a_session_id_stays_inside_the_length_the_store_allows():
    overlong = "a" * 200
    generated = build_candidate_session_id(
        f"kernel:flydsl:{overlong}:vllm:0.1:flydsl:mi355x",
        overlong,
        "b" * 40,
    )
    assert record_store.validate_session_id(generated) == generated


def test_a_session_id_is_stable_so_one_port_stays_one_candidate():
    first = build_candidate_session_id(SOFTMAX_IDENTITY, "softmax", "c" * 40)
    second = build_candidate_session_id(SOFTMAX_IDENTITY, "softmax", "c" * 40)
    assert first == second


# --------------------------------------------------------------------------- # round trip
# --------------------------------------------------------------------------- #
def test_a_recorded_port_is_materialized_and_revalidated(tmp_path, monkeypatch):
    _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    config = _remote_config(tmp_path)
    expected = Path(spec.flydsl_kernel).read_bytes()

    written = kb.write_flydsl_kb_solution(
        kb.RewriteKbTarget(spec, str(driver), config), FlydslMeasurement(10.0, 5.0, 2, snr_db=80.0)
    )
    assert written["written"] is True

    _passing_validation(monkeypatch, best_ms=12.0)
    Path(spec.flydsl_kernel).write_text("def skeleton():\n    pass\n")

    restored = asyncio.run(
        kb.try_flydsl_kb_warmstart(kb.RewriteKbTarget(spec, str(driver), config), source_case_ms={_CASE: 10.0})
    )

    assert restored.applied is True
    assert restored.best_ms == 12.0
    assert restored.solution_slug == f"{SOFTMAX_IDENTITY}/{written['session_id']}"
    assert Path(spec.flydsl_kernel).read_bytes() == expected


def test_warmstart_materializes_crlf_bytes_without_newline_conversion(
    tmp_path,
    monkeypatch,
):
    _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    config = _remote_config(tmp_path)
    artifact = b"import flydsl\r\ndef build_softmax_module(config):\r\n    return lambda inputs: inputs['x']\r\n"
    Path(spec.flydsl_kernel).write_bytes(artifact)
    written = kb.write_flydsl_kb_solution(
        kb.RewriteKbTarget(spec, str(driver), config), FlydslMeasurement(10.0, 5.0, 2)
    )
    assert written["written"] is True
    _passing_validation(monkeypatch, best_ms=5.0)
    Path(spec.flydsl_kernel).write_bytes(b"def skeleton():\n    pass\n")

    restored = asyncio.run(
        kb.try_flydsl_kb_warmstart(kb.RewriteKbTarget(spec, str(driver), config), source_case_ms={_CASE: 10.0})
    )

    assert restored.applied is True
    assert Path(spec.flydsl_kernel).read_bytes() == artifact


def test_reference_decoding_does_not_change_candidate_or_rollback_bytes(
    tmp_path,
    monkeypatch,
):
    _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    config = _remote_config(tmp_path)
    artifact = (
        b"import flydsl\r\ndef build_softmax_module(config):\r\n    return lambda inputs: inputs['x']  # \xff\r\n"
    )
    Path(spec.flydsl_kernel).write_bytes(artifact)
    written = kb.write_flydsl_kb_solution(
        kb.RewriteKbTarget(spec, str(driver), config), FlydslMeasurement(10.0, 5.0, 2)
    )
    assert written["written"] is True
    seed = b"def skeleton():\r\n    pass\r\n"
    Path(spec.flydsl_kernel).write_bytes(seed)
    attempted: list[bytes] = []

    def reject_port(candidate_spec):
        attempted.append(Path(candidate_spec.flydsl_kernel).read_bytes())
        return "unsupported_source_encoding"

    monkeypatch.setattr(kb, "check_flydsl_port", reject_port)

    restored = asyncio.run(
        kb.try_flydsl_kb_warmstart(kb.RewriteKbTarget(spec, str(driver), config), source_case_ms={_CASE: 10.0})
    )

    assert restored.applied is False
    assert attempted == [artifact]
    assert "\ufffd" in restored.reference_context
    assert Path(spec.flydsl_kernel).read_bytes() == seed


def test_the_ported_file_is_an_artifact_not_a_document_field(tmp_path, monkeypatch):
    store = _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    expected = Path(spec.flydsl_kernel).read_bytes()

    written = kb.write_flydsl_kb_solution(
        kb.RewriteKbTarget(spec, str(driver), _remote_config(tmp_path)), FlydslMeasurement(10.0, 5.0, 2)
    )

    key = (SOFTMAX_IDENTITY, written["session_id"])
    value = store.knowledge[key]["value"]
    assert value["flydsl_kernel"] == "kernel.py"
    assert store.files[key] == {"kernel.py": expected}
    assert not any("content" in name for name in value)


# --------------------------------------------------------------------------- # champion is a pointer, not a filter
# --------------------------------------------------------------------------- #
def test_a_correct_but_slower_port_is_recorded_without_being_promoted(
    tmp_path,
    monkeypatch,
):
    store = _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    config = _remote_config(tmp_path)

    written = kb.write_flydsl_kb_solution(
        kb.RewriteKbTarget(spec, str(driver), config), FlydslMeasurement(5.0, 10.0, 0.5)
    )

    assert written["written"] is True
    assert written["speedup"] == 0.5
    assert written["champion"] is False
    assert SOFTMAX_IDENTITY not in store.champions

    _passing_validation(monkeypatch, best_ms=10.0, snr_db=None)
    Path(spec.flydsl_kernel).write_text("def skeleton():\n    pass\n")
    restored = asyncio.run(
        kb.try_flydsl_kb_warmstart(kb.RewriteKbTarget(spec, str(driver), config), source_case_ms={_CASE: 5.0})
    )
    assert restored.applied is True


def test_a_weaker_later_port_does_not_take_the_champion_pointer(tmp_path, monkeypatch):
    store = _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    config = _remote_config(tmp_path)

    strong = kb.write_flydsl_kb_solution(kb.RewriteKbTarget(spec, str(driver), config), FlydslMeasurement(10.0, 2.0, 5))
    Path(spec.flydsl_kernel).write_text("import flydsl\nRANK = 2\n")
    weak = kb.write_flydsl_kb_solution(
        kb.RewriteKbTarget(spec, str(driver), config), FlydslMeasurement(10.0, 8.0, 1.25)
    )

    assert strong["champion"] is True
    assert weak["written"] is True
    assert weak["champion"] is False
    assert store.champions[SOFTMAX_IDENTITY]["session_id"] == strong["session_id"]
    assert len(store.knowledge) == 2


# --------------------------------------------------------------------------- # contract gates
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "edit",
    [
        pytest.param("driver", id="driver edited since the record was written"),
        pytest.param("source", id="source edited since the record was written"),
    ],
)
def test_a_port_is_admitted_on_what_it_is_not_on_which_revision_wrote_it(tmp_path, monkeypatch, edit):
    """The run re-validates and re-measures on today's files, so yesterday's hashes decide nothing."""
    _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    config = _remote_config(tmp_path)
    port = Path(spec.flydsl_kernel).read_text()
    written = kb.write_flydsl_kb_solution(
        kb.RewriteKbTarget(spec, str(driver), config), FlydslMeasurement(10.0, 5.0, 2)
    )
    assert written["written"] is True

    Path(spec.flydsl_kernel).write_text("def skeleton():\n    pass\n")
    if edit == "driver":
        driver.write_text("# stable rewrite driver contract, with a comment added\n")
    else:
        Path(spec.source_kernel).write_text(
            "import triton\n# a comment the port does not care about\n"
            "@triton.jit\ndef softmax_kernel(x):\n    return x\n"
        )
    _passing_validation(monkeypatch, best_ms=5.0)

    adopted = asyncio.run(
        kb.try_flydsl_kb_warmstart(kb.RewriteKbTarget(spec, str(driver), config), source_case_ms={_CASE: 10.0})
    )

    assert adopted.applied is True
    assert adopted.attempts[-1]["reason"] == "applied"
    assert Path(spec.flydsl_kernel).read_text() == port


def test_a_port_whose_implementation_signature_disagrees_is_measured_not_refused(tmp_path, monkeypatch):
    """A second kernel in the source moves the signature while leaving the recorded port perfectly usable."""
    _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    config = _remote_config(tmp_path)
    port = Path(spec.flydsl_kernel).read_text()
    assert (
        kb.write_flydsl_kb_solution(kb.RewriteKbTarget(spec, str(driver), config), FlydslMeasurement(10.0, 5.0, 2))[
            "written"
        ]
        is True
    )

    source = Path(spec.source_kernel)
    before = kb.resolve_identity(spec, gpu="mi355x", source_text=source.read_text())[2]
    source.write_text(source.read_text() + "\n@triton.jit\ndef unrelated_kernel(y):\n    return y\n")
    after = kb.resolve_identity(spec, gpu="mi355x", source_text=source.read_text())[2]
    # Guard the premise: this edit is exactly what used to retire the record.
    assert before != after
    Path(spec.flydsl_kernel).write_text("def skeleton():\n    pass\n")
    _passing_validation(monkeypatch, best_ms=5.0)

    adopted = asyncio.run(
        kb.try_flydsl_kb_warmstart(kb.RewriteKbTarget(spec, str(driver), config), source_case_ms={_CASE: 10.0})
    )

    assert adopted.applied is True
    assert Path(spec.flydsl_kernel).read_text() == port


def test_a_port_that_does_not_expose_the_builder_the_driver_imports_is_refused(tmp_path, monkeypatch):
    """The one thing measurement cannot settle cheaply: the driver would fail to import the entry point."""
    _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    config = _remote_config(tmp_path)
    assert (
        kb.write_flydsl_kb_solution(kb.RewriteKbTarget(spec, str(driver), config), FlydslMeasurement(10.0, 5.0, 2))[
            "written"
        ]
        is True
    )

    seed = "def skeleton():\n    pass\n"
    Path(spec.flydsl_kernel).write_text(seed)
    # Same logical operator, so the same identity, but `softmax_kernel` slugs to a different builder than `softmax`.
    renamed = replace(spec, op_name="softmax_kernel")
    assert renamed.builder_symbol != spec.builder_symbol

    restored = asyncio.run(
        kb.try_flydsl_kb_warmstart(kb.RewriteKbTarget(renamed, str(driver), config), source_case_ms={_CASE: 10.0})
    )

    assert restored.applied is False
    assert restored.attempts[-1]["reason"] == "builder_contract_changed"
    assert Path(spec.flydsl_kernel).read_text() == seed


def test_top_three_are_tried_and_failures_become_references(tmp_path, monkeypatch):
    _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    config = _remote_config(tmp_path)

    for rank, best_ms in ((1, 2.0), (2, 3.0), (3, 4.0)):
        Path(spec.flydsl_kernel).write_text(
            f"import flydsl\nRANK = {rank}\ndef build_softmax_module(config):\n    return lambda inputs: inputs['x']\n"
        )
        written = kb.write_flydsl_kb_solution(
            kb.RewriteKbTarget(spec, str(driver), config), FlydslMeasurement(10.0, best_ms, 10.0 / best_ms)
        )
        assert written["written"] is True

    class Report:
        def __init__(self, passed):
            self.all_passed = passed
            self.results = [type("Result", (), {"snr_db": 80.0})()]

    async def validation(**_kwargs):
        content = Path(spec.flydsl_kernel).read_text()
        return Report("RANK = 3" in content)

    monkeypatch.setattr(kb, "run_validation_pipeline", validation)
    monkeypatch.setattr(
        kb.driver_contract,
        "preflight_candidate",
        lambda *args, **kwargs: driver_contract.PreflightReport(
            ok=True,
            timing_ms=4.0,
            case_ms={_CASE: 4.0},
        ),
    )
    Path(spec.flydsl_kernel).write_text("def skeleton():\n    pass\n")

    restored = asyncio.run(
        kb.try_flydsl_kb_warmstart(kb.RewriteKbTarget(spec, str(driver), config), source_case_ms={_CASE: 10.0})
    )

    assert restored.applied is True
    assert [attempt["reason"] for attempt in restored.attempts] == [
        "correctness_failed",
        "correctness_failed",
        "applied",
    ]
    assert "Reference 1" in restored.reference_context
    assert "Reference 2" in restored.reference_context


def _publish_ranked_candidates(spec, driver, config, ranks):
    """Publish one candidate per (rank, claimed source_ms/best_ms) pair."""
    for rank, source_ms, best_ms in ranks:
        Path(spec.flydsl_kernel).write_text(
            f"import flydsl\nRANK = {rank}\ndef build_softmax_module(config):\n    return lambda inputs: inputs['x']\n"
        )
        written = kb.write_flydsl_kb_solution(
            kb.RewriteKbTarget(spec, str(driver), config),
            FlydslMeasurement(source_ms, best_ms, source_ms / best_ms),
            session_key=str(rank) * 40,
        )
        assert written["written"] is True


def _time_candidates_by_rank(monkeypatch, spec, timings):
    """Pass every candidate, timing each one by the RANK it declares."""

    class Report:
        all_passed = True
        results = [type("Result", (), {"snr_db": 80.0})()]

    async def validation(**_kwargs):
        return Report()

    def preflight(*_args, **_kwargs):
        content = Path(spec.flydsl_kernel).read_text()
        for rank, timing_ms in timings.items():
            if f"RANK = {rank}" in content:
                return driver_contract.PreflightReport(
                    ok=True,
                    timing_ms=timing_ms,
                    case_ms={_CASE: timing_ms},
                )
        raise AssertionError(f"unexpected candidate timed: {content!r}")

    monkeypatch.setattr(kb, "run_validation_pipeline", validation)
    monkeypatch.setattr(kb.driver_contract, "preflight_candidate", preflight)


def test_the_fastest_measured_candidate_wins_not_the_first_to_pass(tmp_path, monkeypatch):
    """Correctness admits a candidate; this task's own clock picks between them.

    A claim is computed over whatever cases produced it, so it does not order candidates for a task that scores
    different ones. Taking the first that merely passed let the best-claiming record win on an unreproduced number.
    """
    _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    config = _remote_config(tmp_path)

    # Claims rank 1 above 2 above 3; measurement reverses that order exactly.
    _publish_ranked_candidates(
        spec,
        driver,
        config,
        [(1, 10.0, 2.0), (2, 10.0, 3.0), (3, 10.0, 4.0)],
    )
    _time_candidates_by_rank(monkeypatch, spec, {1: 8.0, 2: 4.0, 3: 1.0})
    Path(spec.flydsl_kernel).write_text("def skeleton():\n    pass\n")

    restored = asyncio.run(
        kb.try_flydsl_kb_warmstart(kb.RewriteKbTarget(spec, str(driver), config), source_case_ms={_CASE: 10.0})
    )

    assert restored.applied is True
    assert restored.best_ms == 1.0
    assert "RANK = 3" in Path(spec.flydsl_kernel).read_text()
    assert [attempt["reason"] for attempt in restored.attempts] == [
        "outperformed_by_rank_3",
        "outperformed_by_rank_3",
        "applied",
    ]


def test_the_field_is_ordered_by_the_metric_the_run_is_graded_on(tmp_path, monkeypatch):
    """A shape sweep makes the two aggregations disagree, and only one of them is the score.

    Rank 1 wins every small shape and loses the large one; rank 2 is the reverse. Summed wall time is dominated by the
    large shape, so ordering on it adopts rank 2 -- while the equal-weight mean the arena grades on prefers rank 1.
    """
    _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    config = _remote_config(tmp_path)

    _publish_ranked_candidates(spec, driver, config, [(1, 10.0, 5.0), (2, 10.0, 5.0)])

    baseline = {"m_1": 1.0, "m_2": 1.0, "m_4096": 100.0}
    per_rank = {
        1: {"m_1": 0.25, "m_2": 0.25, "m_4096": 100.0},
        2: {"m_1": 1.0, "m_2": 1.0, "m_4096": 50.0},
    }

    class Report:
        all_passed = True
        results = [type("Result", (), {"snr_db": 80.0})()]

    async def validation(**_kwargs):
        return Report()

    def preflight(*_args, **_kwargs):
        content = Path(spec.flydsl_kernel).read_text()
        for rank, case_ms in per_rank.items():
            if f"RANK = {rank}" in content:
                return driver_contract.PreflightReport(
                    ok=True,
                    timing_ms=sum(case_ms.values()) / len(case_ms),
                    case_ms=dict(case_ms),
                )
        raise AssertionError(f"unexpected candidate timed: {content!r}")

    monkeypatch.setattr(kb, "run_validation_pipeline", validation)
    monkeypatch.setattr(kb.driver_contract, "preflight_candidate", preflight)
    Path(spec.flydsl_kernel).write_text("def skeleton():\n    pass\n")

    restored = asyncio.run(
        kb.try_flydsl_kb_warmstart(kb.RewriteKbTarget(spec, str(driver), config), source_case_ms=baseline)
    )

    assert restored.applied is True
    assert "RANK = 1" in Path(spec.flydsl_kernel).read_text()


def test_a_candidate_whose_benchmark_failed_is_not_adopted(tmp_path, monkeypatch):
    """Passing correctness says the kernel is right, not that its claimed timing can be reproduced."""
    _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    config = _remote_config(tmp_path)

    _publish_ranked_candidates(spec, driver, config, [(1, 10.0, 2.0)])

    class Report:
        all_passed = True
        results = [type("Result", (), {"snr_db": 80.0})()]

    async def validation(**_kwargs):
        return Report()

    monkeypatch.setattr(kb, "run_validation_pipeline", validation)
    monkeypatch.setattr(
        kb.driver_contract,
        "preflight_candidate",
        lambda *args, **kwargs: driver_contract.PreflightReport(
            ok=False,
            failure_class=driver_contract.CANDIDATE_MODE_FAILED,
            detail="the driver failed in --bench-mode",
        ),
    )
    seed = "def skeleton():\n    pass\n"
    Path(spec.flydsl_kernel).write_text(seed)

    restored = asyncio.run(
        kb.try_flydsl_kb_warmstart(kb.RewriteKbTarget(spec, str(driver), config), source_case_ms={_CASE: 10.0})
    )

    assert restored.applied is False
    assert Path(spec.flydsl_kernel).read_text() == seed
    assert [attempt["reason"] for attempt in restored.attempts] == ["unscorable_measurement"]


def test_a_candidate_whose_validation_crashes_becomes_a_reference(tmp_path, monkeypatch):
    """A trial that raises is rejected under the error's type and handed to the agent as a reference."""
    _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    config = _remote_config(tmp_path)

    _publish_ranked_candidates(spec, driver, config, [(1, 10.0, 2.0)])

    async def crashing_validation(**_kwargs):
        raise RuntimeError("driver crashed")

    def unexpected_benchmark(*args, **kwargs):
        raise AssertionError("a crashed trial must not be benchmarked")

    monkeypatch.setattr(kb, "run_validation_pipeline", crashing_validation)
    monkeypatch.setattr(kb.driver_contract, "preflight_candidate", unexpected_benchmark)
    seed = "def skeleton():\n    pass\n"
    Path(spec.flydsl_kernel).write_text(seed)

    restored = asyncio.run(
        kb.try_flydsl_kb_warmstart(kb.RewriteKbTarget(spec, str(driver), config), source_case_ms={_CASE: 10.0})
    )

    assert restored.applied is False
    assert restored.read_reason == "candidates_rejected"
    assert Path(spec.flydsl_kernel).read_text() == seed
    assert [attempt["reason"] for attempt in restored.attempts] == ["validation_error:RuntimeError"]
    assert "- Rejection reason: validation_error:RuntimeError" in restored.reference_context


def test_a_candidate_claiming_less_than_the_floor_is_never_tried(tmp_path, monkeypatch):
    """A port that lost badly costs a full trial and teaches nothing."""
    store = _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    config = _remote_config(tmp_path)

    # Rank 2 claims 0.1x -- two hundred times off the pace of rank 1's 2.0x.
    _publish_ranked_candidates(
        spec,
        driver,
        config,
        [(1, 10.0, 5.0), (2, 1.0, 10.0)],
    )
    _time_candidates_by_rank(monkeypatch, spec, {1: 5.0})
    Path(spec.flydsl_kernel).write_text("def skeleton():\n    pass\n")

    restored = asyncio.run(
        kb.try_flydsl_kb_warmstart(kb.RewriteKbTarget(spec, str(driver), config), source_case_ms={_CASE: 10.0})
    )

    assert restored.applied is True
    assert "RANK = 1" in Path(spec.flydsl_kernel).read_text()
    assert [attempt["reason"] for attempt in restored.attempts] == [
        "applied",
        "below_claim_floor",
    ]
    # Skipped whole: never downloaded, and never offered to the author either.
    assert len(store.downloads) == 1
    assert "Reference" not in restored.reference_context


def test_a_field_entirely_under_the_floor_leaves_the_workspace_alone(tmp_path, monkeypatch):
    """Nothing admissible means PORT runs, not that a bad seed is adopted."""
    _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    config = _remote_config(tmp_path)

    _publish_ranked_candidates(
        spec,
        driver,
        config,
        [(1, 1.0, 10.0), (2, 1.0, 20.0)],
    )
    _time_candidates_by_rank(monkeypatch, spec, {})
    seed = "def skeleton():\n    pass\n"
    Path(spec.flydsl_kernel).write_text(seed)

    restored = asyncio.run(
        kb.try_flydsl_kb_warmstart(kb.RewriteKbTarget(spec, str(driver), config), source_case_ms={_CASE: 10.0})
    )

    assert restored.applied is False
    assert restored.read_reason == "candidates_rejected"
    assert {attempt["reason"] for attempt in restored.attempts} == {"below_claim_floor"}
    assert Path(spec.flydsl_kernel).read_text() == seed


def test_one_trial_cannot_spend_the_whole_search_budget(tmp_path, monkeypatch):
    """The budget bounds the field, so it has to bound each trial in it.

    A stage left at its own ceiling outlives the budget it runs under, and the first candidate then consumes a field
    that was widened precisely so several could be measured.
    """
    _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    config = _remote_config(tmp_path)
    monkeypatch.setenv("FORGE_KB_WARMSTART_BUDGET_SEC", "60")

    _publish_ranked_candidates(spec, driver, config, [(1, 10.0, 5.0)])
    stage_timeouts: list[int] = []

    class Report:
        all_passed = True
        results = [type("Result", (), {"snr_db": 80.0})()]

    async def validation(**kwargs):
        stage_timeouts.append(kwargs["timeout_per_stage"])
        return Report()

    monkeypatch.setattr(kb, "run_validation_pipeline", validation)
    monkeypatch.setattr(
        kb.driver_contract,
        "preflight_candidate",
        lambda *_a, **_k: driver_contract.PreflightReport(ok=True, timing_ms=5.0),
    )
    Path(spec.flydsl_kernel).write_text("def skeleton():\n    pass\n")

    asyncio.run(kb.try_flydsl_kb_warmstart(kb.RewriteKbTarget(spec, str(driver), config), source_case_ms={_CASE: 10.0}))

    # Its own ceiling is 1800s; what it may actually take is whatever is left of the 60s budget.
    assert stage_timeouts and all(0 < timeout <= 60 for timeout in stage_timeouts)


# --------------------------------------------------------------------------- # local mode uses the same record layout
# --------------------------------------------------------------------------- #
def test_local_mode_stores_the_same_record_shape_on_disk(tmp_path, monkeypatch):
    spec, driver = _spec(tmp_path)
    expected = Path(spec.flydsl_kernel).read_bytes()
    knowledge, config = _local_config(
        tmp_path,
        spec,
        kb_store_url="https://ambient.invalid",
        kb_store_token="ambient-secret",
    )

    written = kb.write_flydsl_kb_solution(
        kb.RewriteKbTarget(spec, str(driver), config), FlydslMeasurement(10.0, 12.0, 0.833333)
    )

    assert written["written"] is True
    assert knowledge.kb_store_url == ""
    assert knowledge.kb_store_token == ""
    session_dir = knowledge.rewrite_root / Path(*SOFTMAX_IDENTITY.split(":")) / "sessions" / written["session_id"]
    document = json.loads((session_dir / "knowledge.json").read_text())
    assert document["value"]["flydsl_kernel"] == "kernel.py"
    assert (session_dir / "files" / "kernel.py").read_bytes() == expected

    _passing_validation(monkeypatch, best_ms=12.0)
    Path(spec.flydsl_kernel).write_text("def skeleton():\n    pass\n")
    restored = asyncio.run(
        kb.try_flydsl_kb_warmstart(kb.RewriteKbTarget(spec, str(driver), config), source_case_ms={_CASE: 10.0})
    )

    assert restored.applied is True
    assert Path(spec.flydsl_kernel).read_bytes() == expected


def test_local_mode_never_reaches_for_ambient_credentials(tmp_path, monkeypatch):
    spec, driver = _spec(tmp_path)
    monkeypatch.delenv("KNOWLEDGE_STORE_MODE", raising=False)
    monkeypatch.delenv("KNOWLEDGE_LOCAL_ROOT", raising=False)
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "user-data"))
    monkeypatch.setenv("KB_STORE_URL", "https://ambient-kb.invalid")
    monkeypatch.setenv("KB_STORE_TOKEN", "ambient-kb-secret")

    def unexpected_remote(*_args, **_kwargs):
        raise AssertionError("rewrite must not construct a remote client in local mode")

    monkeypatch.setattr(record_store, "KBStoreClient", unexpected_remote)
    config = Config.from_env(
        workspace=spec.workspace,
        gpu_target="gfx950",
        gpu_type="mi355x",
        agent_precheck=False,
    )

    written = kb.write_flydsl_kb_solution(
        kb.RewriteKbTarget(spec, str(driver), config), FlydslMeasurement(10.0, 12.0, 0.833333)
    )

    assert written["written"] is True
    assert config.knowledge_config.mode.value == "local"
    assert config.knowledge_config.kb_store_url == ""


# --------------------------------------------------------------------------- # configuration
# --------------------------------------------------------------------------- #
def test_config_defaults_and_normalizes_gpu_type_independently_from_target(
    monkeypatch,
    tmp_path,
):
    knowledge = KnowledgeConfig.from_env(
        {},
        mode="local",
        local_root=tmp_path / "knowledge",
    )
    monkeypatch.setenv("GPU_TYPE", "mi300x")

    from_environment = Config.from_env(
        gpu_target="gfx950",
        knowledge_config=knowledge,
        agent_precheck=False,
    )
    overridden = Config.from_env(
        gpu_target="gfx950",
        gpu_type="MI300X",
        knowledge_config=knowledge,
        agent_precheck=False,
    )

    assert (from_environment.gpu_type, from_environment.gpu_target) == (
        "mi355x",
        "gfx950",
    )
    assert overridden.gpu_type == "mi300x"


def test_missing_gpu_type_skips_rewrite_kb_reads_and_writes(tmp_path, monkeypatch):
    store = _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    config = _remote_config(tmp_path)
    config.gpu_type = ""

    write = kb.write_flydsl_kb_solution(kb.RewriteKbTarget(spec, str(driver), config), FlydslMeasurement(10.0, 5.0, 2))
    read = asyncio.run(
        kb.try_flydsl_kb_warmstart(kb.RewriteKbTarget(spec, str(driver), config), source_case_ms={_CASE: 10.0})
    )

    assert write == {"written": False, "reason": "missing_gpu_type"}
    assert read.read_reason == "missing_gpu_type"
    assert store.knowledge == {}
    assert store.downloads == []


def test_remote_rewrite_asks_for_the_credentials_it_will_actually_use():
    with pytest.raises(ValueError, match="KB_STORE_URL and KB_STORE_TOKEN"):
        KnowledgeConfig.from_env({"KNOWLEDGE_STORE_MODE": "remote", "KNOWLEDGE_LOCAL_ROOT": "/tmp/kf"})


def test_remote_accepts_kb_store_credentials():
    config = KnowledgeConfig.from_env(
        {
            "KNOWLEDGE_STORE_MODE": "remote",
            "KNOWLEDGE_LOCAL_ROOT": "/tmp/kf",
            "KB_STORE_URL": "http://kb",
            "KB_STORE_TOKEN": "tok",
        },
    )
    assert config.kb_store_url == "http://kb"


def test_an_unrenderable_segment_falls_back_to_a_readable_address():
    """A dimension that folds away must not silently become an empty address."""
    assert segment("", fallback="unknown") == "unknown"
    assert segment(":::", fallback="unknown") == "unknown"


def test_remote_without_kb_store_credentials_reads_as_a_cold_start(tmp_path):
    spec, driver = _spec(tmp_path)
    # Built directly: from_env refuses this combination, which is exactly how a misconfigured run is caught at
    # startup.
    knowledge = KnowledgeConfig(
        mode=KnowledgeStoreMode.REMOTE,
        local_root=tmp_path / "knowledge",
    )
    config = Config.from_env(
        workspace=spec.workspace,
        gpu_target="gfx950",
        gpu_type="mi355x",
        knowledge_config=knowledge,
        agent_precheck=False,
    )

    written = kb.write_flydsl_kb_solution(
        kb.RewriteKbTarget(spec, str(driver), config), FlydslMeasurement(10.0, 5.0, 2)
    )
    read = asyncio.run(
        kb.try_flydsl_kb_warmstart(kb.RewriteKbTarget(spec, str(driver), config), source_case_ms={_CASE: 10.0})
    )

    assert written == {"written": False, "reason": "not_configured"}
    assert read.applied is False
    assert read.read_reason == "not_configured"


def test_kb_store_rewrite_keeps_the_measurement_a_consumer_recorded(tmp_path):
    """The remote backend must preserve a measurement across a replacing write."""
    client = InMemoryKBStore()
    store = record_store.KBStoreRewriteRecords(client)
    source = tmp_path / "kernel.py"
    source.write_bytes(b"first")
    canonical_id = "kernel:flydsl:softmax:vllm:1.0:flydsl:mi355x"

    store.write(canonical_id, "s1", {"speedup": 9.0}, {"kernel.py": source})
    store.record_measured_speedup(canonical_id, "s1", 1.2)
    store.write(
        canonical_id,
        "s1",
        {"speedup": 9.0, "version": "second"},
        {"kernel.py": source},
    )

    knowledge = client.knowledge[(canonical_id, "s1")]
    assert knowledge[record_store.MEASURED_SPEEDUP_KEY] == 1.2
    assert knowledge["version"] == "second"
    assert store.candidates(canonical_id, limit=1)[0].measured_speedup == 1.2


_EXPECTED_STORE_FAILURES = [
    record_store.KBStoreError("link down"),
    record_store.RewriteRecordError("unusable canonical id"),
    OSError("disk gone"),
]


def _raiser(failure: Exception):
    def fail(*_args, **_kwargs):
        raise failure

    return fail


def _failing_read_target(tmp_path, monkeypatch):
    store = _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    monkeypatch.setattr(store, "get_top_sessions", _raiser(record_store.KBStoreError("link down")))
    return spec, kb.RewriteKbTarget(spec, str(driver), _remote_config(tmp_path))


def test_a_failing_store_read_propagates_from_the_read_plan(tmp_path, monkeypatch):
    spec, target = _failing_read_target(tmp_path, monkeypatch)

    with pytest.raises(record_store.KBStoreError, match="^link down$"):
        kb._read_top_candidates(spec, target.config, top_k=3)


def test_a_failing_store_read_cold_starts_through_the_rewrite_reader(tmp_path, monkeypatch):
    from kernelforge.workflows.flydsl_rewrite import runner

    spec, target = _failing_read_target(tmp_path, monkeypatch)
    seed = Path(spec.flydsl_kernel).read_bytes()

    read = runner._RewriteKb(target, True).read(source_case_ms={_CASE: 10.0}, search_stop_unix=0.0)

    assert (read.applied, read.read_reason, read.read_error) == (False, "read_error", "KBStoreError: link down")
    assert Path(spec.flydsl_kernel).read_bytes() == seed


def _artifact_read(_tmp_path, _monkeypatch, failure: Exception) -> bytes:
    plan = kb._ReadPlan(SimpleNamespace(read_bytes=_raiser(failure)), [], "hit")
    candidate = {"canonical_id": SOFTMAX_IDENTITY, "session_id": "s1", "attrs": {"flydsl_kernel": "kernel.py"}}
    return kb._candidate_content(plan, candidate)


def _publication(tmp_path, monkeypatch, failure: Exception) -> dict:
    store = _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    monkeypatch.setattr(store, "put_knowledge", _raiser(failure))
    target = kb.RewriteKbTarget(spec, str(driver), _remote_config(tmp_path))
    return kb.write_flydsl_kb_solution(target, FlydslMeasurement(10.0, 5.0, 2))


@pytest.mark.parametrize("site", [_artifact_read, _publication], ids=lambda site: site.__name__)
def test_an_unexpected_store_error_propagates(tmp_path, monkeypatch, site):
    with pytest.raises(TypeError, match="^store contract drifted$"):
        site(tmp_path, monkeypatch, TypeError("store contract drifted"))


@pytest.mark.parametrize("failure", _EXPECTED_STORE_FAILURES, ids=lambda failure: type(failure).__name__)
def test_an_unreadable_artifact_is_a_miss(tmp_path, monkeypatch, failure):
    assert _artifact_read(tmp_path, monkeypatch, failure) == b""


@pytest.mark.parametrize(
    "failure", [*_EXPECTED_STORE_FAILURES, ValueError("bad identity")], ids=lambda failure: type(failure).__name__
)
def test_a_refused_publication_is_reported_not_raised(tmp_path, monkeypatch, failure):
    written = _publication(tmp_path, monkeypatch, failure)

    assert written == {"written": False, "reason": f"{type(failure).__name__}: {failure}"}
