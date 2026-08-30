"""Hermetic tests for the rewrite agent's own KB facade."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernelforge.config import Config
from kernelforge.knowledge import experience_integration, experience_sink
from kernelforge.knowledge.experience_store import (
    REMOTE_BACKEND_KB_STORE,
    KnowledgeConfig,
    KnowledgeStoreMode,
)
from kernelforge.knowledge.kernel_identity import (
    KERNEL_CANONICAL_DIMENSIONS,
    KernelRecipeIdentity,
    kernel_recipe_canonical_id,
)
from kernelforge.rewrite_by_flydsl import agent_kb as agent_kb_module
from kernelforge.rewrite_by_flydsl import kb as flydsl_kb
from kernelforge.rewrite_by_flydsl.agent_kb import KernelRecipeKB
from kernelforge.rewrite_by_flydsl.identity import framework_version, resolve_identity
from kernelforge.rewrite_by_flydsl.record_store import KBStoreError, RewriteRecordError
from kernelforge.rewrite_by_flydsl.spec import RewriteSpec

from kernelforge.tests.test_rewrite_by_flydsl_kb import (
    InMemoryKBStore,
    _remote_config,
    _spec,
    _use_in_memory_kb_store,
)

VLLM_VERSION = framework_version("vllm")
SOFTMAX_IDENTITY = f"kernel:flydsl:softmax:vllm:{VLLM_VERSION}:flydsl:mi355x"

#: The cap :func:`sanitize_read_error` bounds a persisted store error at.
MAX_REASON_LENGTH = 240


def _remote_config_with_token(tmp_path, token: str) -> Config:
    """A KB Store run configuration whose credential is a recognizable string."""
    knowledge = KnowledgeConfig.from_env(
        {},
        mode="remote",
        local_root=tmp_path / "remote-knowledge",
        kb_store_url="http://in-memory",
        kb_store_token=token,
        remote_backend=REMOTE_BACKEND_KB_STORE,
    )
    return Config.from_env(
        workspace=str(tmp_path),
        gpu_target="gfx950",
        gpu_type="mi355x",
        knowledge_config=knowledge,
        agent_precheck=False,
    )


def _credentialed_store_error(token: str) -> KBStoreError:
    """A store failure whose text carries the credential and overruns the cap.

    The client authenticates with a bearer token and addresses the store by URL,
    so a transport error that quotes the request line carries the credential
    twice over, and an error body is as long as the service decides to make it.
    """
    return KBStoreError(
        f"PUT https://forge:{token}@kb.example/knowledge failed "
        f"(sent Bearer {token}); the store said {token} expired" + " and returned an unbounded body" * 20
    )


def _resolved_identity(
    spec: RewriteSpec,
    config: Config,
    *,
    producer: str = "flydsl",
    backend: str = "flydsl",
) -> KernelRecipeIdentity:
    identity, _canonical_id, _signature, _implementation = resolve_identity(
        spec,
        framework="vllm",
        gpu=str(config.gpu_type or "").strip(),
        source_text=Path(spec.source_kernel).read_text(
            encoding="utf-8",
            errors="replace",
        ),
        producer=producer,
        backend=backend,
    )
    return identity


def _kb(tmp_path, monkeypatch) -> tuple[KernelRecipeKB, InMemoryKBStore, RewriteSpec]:
    store = _use_in_memory_kb_store(monkeypatch)
    spec, _driver = _spec(tmp_path)
    config = _remote_config(tmp_path)
    identity = _resolved_identity(spec, config)
    return (
        KernelRecipeKB.open_identity(identity, config),
        store,
        spec,
    )


def test_a_run_without_a_configured_store_turns_every_call_into_a_no_op(
    tmp_path,
    monkeypatch,
):
    # Remote mode without KB Store credentials: the store layer reports it by
    # declining to build a backend at all.
    monkeypatch.setattr(agent_kb_module, "create_rewrite_record_store", lambda _: None)
    spec, _driver = _spec(tmp_path)
    config = _remote_config(tmp_path)
    identity = _resolved_identity(spec, config)

    kb = KernelRecipeKB.open_identity(identity, config)

    assert kb.active is False
    assert kb.reason == "not_configured"
    assert kb.read_best(tmp_path / "best") is None
    assert kb.read_top_n(tmp_path / "top") == []
    assert kb.prior_file("any", "kernel.py") == b""
    assert kb.write_candidate({"metric": {}}) == {
        "written": False,
        "reason": "not_configured",
    }


def test_a_rewrite_is_filed_under_its_producer_owned_identity(tmp_path, monkeypatch):
    kb, store, _spec_ = _kb(tmp_path, monkeypatch)

    outcome = kb.write_candidate({"rewrite_kind": "standalone_flydsl"}, speedup=2.0)

    assert kb.canonical_id == SOFTMAX_IDENTITY
    assert outcome["written"] is True
    assert outcome["canonical_id"] == SOFTMAX_IDENTITY
    document = store.knowledge[(SOFTMAX_IDENTITY, outcome["session_id"])]
    assert document["producer"] == "flydsl"
    assert document["value"] == {"rewrite_kind": "standalone_flydsl"}
    assert document["identity"]["producer"] == "flydsl"
    assert document["identity"]["gpu"] == "mi355x"
    assert document["speedup"] == 2.0


def test_sdk_opens_and_records_an_arbitrary_rewrite_backend(tmp_path, monkeypatch):
    store = _use_in_memory_kb_store(monkeypatch)
    identity = KernelRecipeIdentity(
        producer="flydsl",
        kernel_name="softmax",
        framework="vllm",
        framework_version=VLLM_VERSION,
        backend="triton",
        gpu="mi355x",
    )

    kb = KernelRecipeKB.open_identity(identity, _remote_config(tmp_path))
    outcome = kb.write_candidate({"rewrite_kind": "triton"}, speedup=1.5)

    canonical_id = f"kernel:flydsl:softmax:vllm:{VLLM_VERSION}:triton:mi355x"
    assert kb.canonical_id == canonical_id
    assert outcome["canonical_id"] == canonical_id
    document = store.knowledge[(canonical_id, outcome["session_id"])]
    assert document["producer"] == "flydsl"
    assert document["identity"]["backend"] == "triton"
    assert document["value"] == {"rewrite_kind": "triton"}


def test_resolved_identity_allows_an_explicit_backend(tmp_path, monkeypatch):
    _use_in_memory_kb_store(monkeypatch)
    spec, _driver = _spec(tmp_path)
    config = _remote_config(tmp_path)
    identity = _resolved_identity(spec, config, backend="triton")

    kb = KernelRecipeKB.open_identity(identity, config)

    assert kb.canonical_id == f"kernel:flydsl:softmax:vllm:{VLLM_VERSION}:triton:mi355x"


def test_recipe_identity_requires_a_supported_producer():
    assert KERNEL_CANONICAL_DIMENSIONS == (
        "producer",
        "kernel_name",
        "framework",
        "framework_version",
        "backend",
        "gpu",
    )
    hip = KernelRecipeIdentity(
        producer="forge-loop",
        kernel_name="softmax",
        framework="vllm",
        framework_version=VLLM_VERSION,
        backend="hip",
        gpu="mi355x",
    )
    assert kernel_recipe_canonical_id(hip) == (f"kernel:forge-loop:softmax:vllm:{VLLM_VERSION}:hip:mi355x")
    with pytest.raises(ValueError, match="producer must be one of"):
        KernelRecipeIdentity(
            producer="other",
            kernel_name="softmax",
            framework="vllm",
            framework_version=VLLM_VERSION,
            backend="flydsl",
            gpu="mi355x",
        )


def test_producers_have_independent_candidates_top1_and_champions(
    tmp_path,
    monkeypatch,
):
    store = _use_in_memory_kb_store(monkeypatch)
    spec, _driver = _spec(tmp_path)
    config = _remote_config(tmp_path)
    flydsl_identity = _resolved_identity(
        spec,
        config,
        producer="flydsl",
        backend="flydsl",
    )
    forge_loop_identity = _resolved_identity(
        spec,
        config,
        producer="forge-loop",
        backend="flydsl",
    )
    flydsl = KernelRecipeKB.open_identity(flydsl_identity, config)
    forge_loop = KernelRecipeKB.open_identity(forge_loop_identity, config)

    flydsl.write_candidate({"owner": "flydsl", "rank": 2}, speedup=1.5)
    flydsl_best = flydsl.write_candidate({"owner": "flydsl", "rank": 1}, speedup=2.0)
    forge_loop.write_candidate({"owner": "forge-loop", "rank": 2}, speedup=3.0)
    forge_loop_best = forge_loop.write_candidate({"owner": "forge-loop", "rank": 1}, speedup=4.0)

    assert flydsl.canonical_id != forge_loop.canonical_id
    assert flydsl.canonical_id == SOFTMAX_IDENTITY
    assert forge_loop.canonical_id == (f"kernel:forge-loop:softmax:vllm:{VLLM_VERSION}:flydsl:mi355x")
    flydsl_bundles = flydsl.read_top_n(tmp_path / "flydsl-priors", limit=2)
    forge_bundles = forge_loop.read_top_n(tmp_path / "forge-priors", limit=2)
    assert [item.value["owner"] for item in flydsl_bundles] == [
        "flydsl",
        "flydsl",
    ]
    assert [item.value["owner"] for item in forge_bundles] == [
        "forge-loop",
        "forge-loop",
    ]
    assert all(bundle.bundle_dir.parent.name == "flydsl-priors" for bundle in flydsl_bundles)
    assert all(bundle.bundle_dir.parent.name == "forge-priors" for bundle in forge_bundles)
    assert {json.loads(bundle.recipe_path.read_text(encoding="utf-8"))["producer"] for bundle in flydsl_bundles} == {
        "flydsl"
    }
    assert {json.loads(bundle.recipe_path.read_text(encoding="utf-8"))["producer"] for bundle in forge_bundles} == {
        "forge-loop"
    }
    assert flydsl.read_best(tmp_path / "flydsl-best").value == {
        "owner": "flydsl",
        "rank": 1,
    }
    assert forge_loop.read_best(tmp_path / "forge-best").value == {
        "owner": "forge-loop",
        "rank": 1,
    }
    assert store.champions[flydsl.canonical_id]["session_id"] == flydsl_best["session_id"]
    assert store.champions[forge_loop.canonical_id]["session_id"] == forge_loop_best["session_id"]
    assert store.knowledge[(forge_loop.canonical_id, forge_loop_best["session_id"])]["producer"] == "forge-loop"


def test_a_file_list_becomes_artifacts_named_after_the_files(tmp_path, monkeypatch):
    kb, store, spec = _kb(tmp_path, monkeypatch)
    artifact = b"line1\r\nline2\r\n\xff"
    Path(spec.flydsl_kernel).write_bytes(artifact)

    outcome = kb.write_candidate({"metric": {}}, files=[spec.flydsl_kernel], speedup=1.5)

    assert outcome["files"] == ["kernel.py"]
    stored = store.files[(SOFTMAX_IDENTITY, outcome["session_id"])]
    assert stored["kernel.py"] == artifact
    assert kb.prior_file(outcome["session_id"], "kernel.py") == artifact
    assert kb.prior_file(outcome["session_id"], "missing.py") == b""


def test_a_mapping_names_the_artifacts_when_the_names_matter(tmp_path, monkeypatch):
    kb, store, spec = _kb(tmp_path, monkeypatch)

    outcome = kb.write_candidate(
        {"metric": {}},
        files={"ported/kernel.py": Path(spec.flydsl_kernel)},
        speedup=1.5,
    )

    assert outcome["files"] == ["ported/kernel.py"]
    assert "ported/kernel.py" in store.files[(SOFTMAX_IDENTITY, outcome["session_id"])]


@pytest.mark.parametrize("path_kind", ["absolute", "traversal"])
def test_unsafe_mapping_path_is_rejected_before_digest_or_staging(tmp_path, monkeypatch, path_kind):
    kb, store, spec = _kb(tmp_path, monkeypatch)
    staging = tmp_path / "staging"
    canary = tmp_path / "canary.py"
    canary.write_bytes(b"do not overwrite")
    rel_path = str(canary) if path_kind == "absolute" else "../../canary.py"
    digest_called = False

    class FixedTemporaryDirectory:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            staging.mkdir(exist_ok=True)
            return str(staging)

        def __exit__(self, *_args):
            return False

    def tracked_digest(knowledge, files):
        nonlocal digest_called
        digest_called = True
        return "unexpected"

    monkeypatch.setattr(agent_kb_module.tempfile, "TemporaryDirectory", FixedTemporaryDirectory)
    monkeypatch.setattr(agent_kb_module, "_port_digest", tracked_digest)

    outcome = kb.write_candidate(
        {"tag": "unsafe"},
        files={rel_path: Path(spec.flydsl_kernel)},
        speedup=2.0,
    )

    assert outcome["written"] is False
    assert "unsafe artifact path" in outcome["reason"]
    assert digest_called is False
    assert canary.read_bytes() == b"do not overwrite"
    assert store.knowledge == {}


@pytest.mark.parametrize("rel_path", ["/tmp/outside.py", "../../outside.py"])
def test_stage_revalidates_mapping_paths_without_overwriting_canary(tmp_path, monkeypatch, rel_path):
    kb, _store, spec = _kb(tmp_path, monkeypatch)
    staging = tmp_path / "stage"
    staging.mkdir()
    canary = tmp_path / "outside.py"
    canary.write_bytes(b"canary")

    with pytest.raises(RewriteRecordError, match="unsafe artifact path"):
        kb._stage({rel_path: Path(spec.flydsl_kernel)}, staging)

    assert canary.read_bytes() == b"canary"


def test_the_agent_reads_back_the_best_port_recorded_for_its_identity(
    tmp_path,
    monkeypatch,
):
    kb, _store, _spec_ = _kb(tmp_path, monkeypatch)
    kb.write_candidate({"tag": "slow"}, speedup=1.2)
    kb.write_candidate({"tag": "fast"}, speedup=3.0)

    assert kb.read_best(tmp_path / "best").value == {"tag": "fast"}
    assert [prior.value["tag"] for prior in kb.read_top_n(tmp_path / "top")] == ["fast", "slow"]


def test_top_n_materializes_isolated_complete_bundles_and_only_downloads_limit(
    tmp_path,
    monkeypatch,
):
    kb, store, _spec_ = _kb(tmp_path, monkeypatch)
    written: dict[str, dict] = {}
    for tag, speedup in (("slow", 1.2), ("fast", 3.0), ("middle", 2.0)):
        source = tmp_path / f"{tag}.py"
        source.write_text(f"# {tag}\n", encoding="utf-8")
        written[tag] = kb.write_candidate(
            {"tag": tag, "producer": "business-conflict"},
            files={
                "kernel.py": source,
                f"only/{tag}.txt": source,
            },
            speedup=speedup,
        )
        store.knowledge[(SOFTMAX_IDENTITY, written[tag]["session_id"])].update(
            canonical_id="business-conflict",
            session_id="business-conflict",
            champion="business-conflict",
            is_champion="business-conflict",
        )

    destination = tmp_path / "top-three"
    bundles = kb.read_top_n(destination)

    assert [bundle.value["tag"] for bundle in bundles] == [
        "fast",
        "middle",
        "slow",
    ]
    assert len({bundle.bundle_dir for bundle in bundles}) == 3
    assert [session_id for _canonical_id, session_id in store.downloads] == [
        written["fast"]["session_id"],
        written["middle"]["session_id"],
        written["slow"]["session_id"],
    ]
    for bundle in bundles:
        tag = bundle.value["tag"]
        assert bundle.bundle_dir == destination / bundle.session_id
        assert bundle.recipe_path == bundle.bundle_dir / "recipe.json"
        assert bundle.files_dir == bundle.bundle_dir / "files"
        assert {path.name for path in bundle.bundle_dir.iterdir()} == {
            "recipe.json",
            "files",
        }
        assert (bundle.files_dir / "kernel.py").read_text(encoding="utf-8") == f"# {tag}\n"
        assert (bundle.files_dir / "only" / f"{tag}.txt").is_file()
        assert sorted(path.name for path in (bundle.files_dir / "only").iterdir()) == [f"{tag}.txt"]
        recipe = json.loads(bundle.recipe_path.read_text(encoding="utf-8"))
        assert recipe["canonical_id"] == SOFTMAX_IDENTITY
        assert recipe["session_id"] == bundle.session_id
        assert recipe["producer"] == "flydsl"
        assert recipe["identity"]["producer"] == "flydsl"
        assert recipe["speedup"] == bundle.speedup
        assert recipe["value"]["tag"] == tag
        assert recipe["is_champion"] is bundle.is_champion
        assert recipe["champion"] is bundle.is_champion
    assert [bundle.value["tag"] for bundle in bundles if bundle.is_champion] == ["fast"]

    store.downloads.clear()
    limited = kb.read_top_n(tmp_path / "top-two", limit=2)
    assert [bundle.value["tag"] for bundle in limited] == ["fast", "middle"]
    assert [session_id for _canonical_id, session_id in store.downloads] == [
        written["fast"]["session_id"],
        written["middle"]["session_id"],
    ]


def test_remote_top_n_ranks_beyond_twenty_recent_sessions(tmp_path, monkeypatch):
    kb, store, _spec_ = _kb(tmp_path, monkeypatch)
    best = kb.write_candidate({"tag": "old-best"}, speedup=10.0)
    for index in range(21):
        kb.write_candidate({"tag": f"recent-{index}"}, speedup=1.0 + index / 100)
    store.champions.clear()

    bundle = kb.read_best(tmp_path / "best")

    assert bundle is not None
    assert bundle.session_id == best["session_id"]
    assert bundle.value == {"tag": "old-best"}
    assert bundle.speedup == 10.0


def test_materialization_cleans_stale_candidate_files(tmp_path, monkeypatch):
    kb, _store, spec = _kb(tmp_path, monkeypatch)
    outcome = kb.write_candidate(
        {"tag": "clean"},
        files={"kernel.py": Path(spec.flydsl_kernel)},
        speedup=2.0,
    )
    destination = tmp_path / "bundles"
    first = kb.read_best(destination)
    assert first is not None
    stale = first.files_dir / "stale.txt"
    stale.write_text("old", encoding="utf-8")
    (first.bundle_dir / "obsolete.json").write_text("{}", encoding="utf-8")

    second = kb.read_best(destination)

    assert second is not None
    assert second.session_id == outcome["session_id"]
    assert not stale.exists()
    assert not (second.bundle_dir / "obsolete.json").exists()
    assert (second.files_dir / "kernel.py").is_file()


def test_remote_materialization_rejects_traversal_before_download(tmp_path, monkeypatch):
    kb, store, spec = _kb(tmp_path, monkeypatch)
    outcome = kb.write_candidate(
        {"tag": "unsafe"},
        files={"kernel.py": Path(spec.flydsl_kernel)},
        speedup=2.0,
    )
    files = store.files[(SOFTMAX_IDENTITY, outcome["session_id"])]
    files["../escape.py"] = b"escaped"
    destination = tmp_path / "unsafe-bundles"

    assert kb.read_best(destination) is None
    assert store.downloads == []
    assert not (destination / "escape.py").exists()
    assert "unsafe artifact path" in kb.reason


def test_remote_read_rejects_a_mismatched_session_envelope(tmp_path, monkeypatch):
    kb, store, _spec_ = _kb(tmp_path, monkeypatch)
    kb.write_candidate({"tag": "wrong-envelope"}, speedup=2.0)
    original_get_session = store.get_session

    def mismatched_get_session(canonical_id, session_id):
        envelope = original_get_session(canonical_id, session_id)
        assert envelope is not None
        envelope["canonical_id"] = "kernel:wrong"
        return envelope

    store.get_session = mismatched_get_session

    assert kb.read_best(tmp_path / "wrong-envelope") is None
    assert store.downloads == []
    assert "canonical id mismatch" in kb.reason


def test_a_cold_identity_reads_as_empty_rather_than_failing(tmp_path, monkeypatch):
    kb, _store, _spec_ = _kb(tmp_path, monkeypatch)

    assert kb.active is True
    assert kb.read_best(tmp_path / "best") is None
    assert kb.read_top_n(tmp_path / "top") == []


def test_a_port_that_loses_to_its_baseline_is_recorded_but_never_promoted(
    tmp_path,
    monkeypatch,
):
    kb, store, _spec_ = _kb(tmp_path, monkeypatch)

    outcome = kb.write_candidate({"tag": "correct-but-slower"}, speedup=0.8)

    assert outcome["written"] is True
    assert outcome["champion"] is False
    assert SOFTMAX_IDENTITY not in store.champions
    assert kb.read_best(tmp_path / "best").value == {"tag": "correct-but-slower"}


def test_the_champion_pointer_only_moves_for_a_port_that_beats_the_incumbent(
    tmp_path,
    monkeypatch,
):
    kb, store, _spec_ = _kb(tmp_path, monkeypatch)

    first = kb.write_candidate({"tag": "first"}, speedup=2.0)
    second = kb.write_candidate({"tag": "second"}, speedup=1.5)
    third = kb.write_candidate({"tag": "third"}, speedup=2.5)

    assert [first["champion"], second["champion"], third["champion"]] == [
        True,
        False,
        True,
    ]
    assert store.champions[SOFTMAX_IDENTITY]["session_id"] == third["session_id"]
    assert store.champions[SOFTMAX_IDENTITY]["value"] == 2.5


def test_recording_the_same_port_twice_updates_one_candidate(tmp_path, monkeypatch):
    kb, store, spec = _kb(tmp_path, monkeypatch)

    first = kb.write_candidate({"tag": "same"}, files=[spec.flydsl_kernel], speedup=2.0)
    second = kb.write_candidate({"tag": "same"}, files=[spec.flydsl_kernel], speedup=2.0)

    assert first["session_id"] == second["session_id"]
    assert len(store.knowledge) == 1


def test_a_different_gpu_is_a_different_identity(tmp_path, monkeypatch):
    kb, _store, spec = _kb(tmp_path, monkeypatch)
    other_config = _remote_config(tmp_path)
    other_config.gpu_type = "mi300x"
    other_identity = _resolved_identity(spec, other_config)

    other = KernelRecipeKB.open_identity(other_identity, other_config)

    assert other.canonical_id != kb.canonical_id
    assert other.canonical_id.endswith(":mi300x")


def test_records_survive_on_the_local_backend_too(tmp_path):
    spec, _driver = _spec(tmp_path)
    artifact = b"line1\r\nline2\r\n\xff"
    Path(spec.flydsl_kernel).write_bytes(artifact)
    knowledge = KnowledgeConfig.from_env(
        {},
        mode="local",
        local_root=tmp_path / "local-knowledge",
    )
    assert knowledge.mode is KnowledgeStoreMode.LOCAL
    config = Config.from_env(
        workspace=spec.workspace,
        gpu_target="gfx950",
        gpu_type="mi355x",
        knowledge_config=knowledge,
        agent_precheck=False,
    )
    identity = _resolved_identity(spec, config)
    kb = KernelRecipeKB.open_identity(identity, config)

    outcome = kb.write_candidate({"tag": "on-disk"}, files=[spec.flydsl_kernel], speedup=2.0)

    assert outcome["written"] is True
    bundle = kb.read_best(tmp_path / "best")
    assert bundle is not None
    assert bundle.value == {"tag": "on-disk"}
    assert bundle.bundle_dir == tmp_path / "best" / outcome["session_id"]
    assert bundle.recipe_path.name == "recipe.json"
    assert bundle.files_dir.name == "files"
    recipe = json.loads(bundle.recipe_path.read_text(encoding="utf-8"))
    assert recipe["canonical_id"] == SOFTMAX_IDENTITY
    assert recipe["session_id"] == outcome["session_id"]
    assert recipe["producer"] == "flydsl"
    assert recipe["champion"] is True
    assert (bundle.files_dir / "kernel.py").read_bytes() == artifact
    assert kb.prior_file(outcome["session_id"], "kernel.py") == artifact
    assert kb.prior_file(outcome["session_id"], "missing.py") == b""


def test_local_materialization_rejects_symlink_artifacts(tmp_path):
    spec, _driver = _spec(tmp_path)
    knowledge = KnowledgeConfig.from_env(
        {},
        mode="local",
        local_root=tmp_path / "local-knowledge",
    )
    config = Config.from_env(
        workspace=spec.workspace,
        gpu_target="gfx950",
        gpu_type="mi355x",
        knowledge_config=knowledge,
        agent_precheck=False,
    )
    identity = _resolved_identity(spec, config)
    kb = KernelRecipeKB.open_identity(identity, config)
    outcome = kb.write_candidate(
        {"tag": "unsafe-local"},
        files=[spec.flydsl_kernel],
        speedup=2.0,
    )
    artifact = (
        knowledge.rewrite_root
        / Path(*SOFTMAX_IDENTITY.split(":"))
        / "sessions"
        / outcome["session_id"]
        / "files"
        / "kernel.py"
    )
    outside = tmp_path / "outside.py"
    outside.write_text("do not copy\n", encoding="utf-8")
    artifact.unlink()
    artifact.symlink_to(outside)

    assert kb.read_best(tmp_path / "unsafe-local-bundles") is None
    assert "regular file" in kb.reason
    assert outside.read_text(encoding="utf-8") == "do not copy\n"


def test_a_knowledge_payload_that_is_not_a_mapping_is_refused(tmp_path, monkeypatch):
    kb, _store, _spec_ = _kb(tmp_path, monkeypatch)

    assert kb.write_candidate(["not", "a", "mapping"]) == {
        "written": False,
        "reason": "knowledge_not_a_mapping",
    }


def test_an_unreadable_artifact_fails_the_write_instead_of_recording_half_a_port(
    tmp_path,
    monkeypatch,
):
    """Nothing is recorded, and the reason says which artifact was missing.

    The refusal is persisted in the run's result JSON, so it has to name the
    failure well enough to act on -- the exception type and the artifact that
    could not be read -- while staying inside the cap that keeps one error out of
    the rest of the file. Redaction of a reason that does carry a credential is
    pinned by
    :func:`test_a_refused_candidate_write_redacts_and_bounds_the_store_error`,
    which drives the same handler with a store error instead of a missing file.
    """
    kb, store, _spec_ = _kb(tmp_path, monkeypatch)

    outcome = kb.write_candidate({"tag": "x"}, files=[tmp_path / "absent.py"], speedup=2.0)

    assert outcome["written"] is False
    assert outcome["reason"].startswith("FileNotFoundError: ")
    assert "absent.py" in outcome["reason"]
    assert len(outcome["reason"]) <= MAX_REASON_LENGTH
    assert store.knowledge == {}


def test_the_record_store_is_unused_when_the_facade_is_inactive(tmp_path):
    kb = KernelRecipeKB(None, reason="missing_gpu_type")

    assert kb.active is False
    assert kb.write_candidate({"tag": "x"})["reason"] == "missing_gpu_type"


# --------------------------------------------------------------------------- #
# What a refused write is allowed to say about it. Every reason below is
# persisted into the run's result JSON, so none of them may carry the bearer
# token the store client authenticates with, and none of them may grow to
# whatever length the service made its error body.
# --------------------------------------------------------------------------- #
def test_a_refused_measured_write_back_redacts_the_error_that_opened_the_store(
    tmp_path,
    monkeypatch,
):
    """Opening the record's address is part of the write-back's error surface.

    ``record_measured_speedup`` sanitizes what the amendment itself raises, but
    ``_record_measured_speedup`` wraps the whole chain, and building the store
    client happens first: ``open_canonical_id`` calls
    ``create_rewrite_record_store``, which lets anything that is not a
    ``KBStoreError`` out. The reason this handler builds travels through
    ``measured_writebacks`` and ``measured_writeback_failures`` into the run's
    result JSON, so it is sanitized and bounded here too.
    """
    token = "kb-store-secret-9f3c"

    def refuse_to_open(_config):
        raise _credentialed_store_error(token)

    monkeypatch.setattr(agent_kb_module, "create_rewrite_record_store", refuse_to_open)

    writeback = experience_integration._record_measured_speedup(
        _remote_config_with_token(tmp_path, token),
        {
            "solution_slug": f"{SOFTMAX_IDENTITY}/kda-attn-session",
            "kernel_slug": SOFTMAX_IDENTITY,
            "session_id": "kda-attn-session",
        },
        2.5,
        rank=1,
    )

    assert writeback["recorded"] is False
    assert token not in writeback["reason"]
    assert writeback["reason"].startswith("KBStoreError: PUT https://[REDACTED]@")
    assert "Bearer [REDACTED]" in writeback["reason"]
    assert "the store said [REDACTED] expired" in writeback["reason"]
    assert len(writeback["reason"]) == MAX_REASON_LENGTH


def test_a_refused_candidate_write_redacts_and_bounds_the_store_error(
    tmp_path,
    monkeypatch,
):
    """A refused ``write_candidate`` reports the store's own words.

    Its reason reaches the run's result JSON through two routes: the rewrite
    runner files it under ``kb_experience.write``, and ``write_run_experience``
    passes it straight back to the forge loop. Both persist it, so the store's
    exception is redacted and capped before it is handed back.
    """
    token = "kb-store-secret-9f3c"
    store = _use_in_memory_kb_store(monkeypatch)
    spec, _driver = _spec(tmp_path)
    config = _remote_config_with_token(tmp_path, token)
    kb = KernelRecipeKB.open_identity(_resolved_identity(spec, config), config)

    def refuse(*_args, **_kwargs):
        raise _credentialed_store_error(token)

    monkeypatch.setattr(store, "put_knowledge", refuse)

    outcome = kb.write_candidate({"tag": "refused"}, files=[spec.flydsl_kernel], speedup=2.0)

    assert outcome["written"] is False
    assert token not in outcome["reason"]
    assert outcome["reason"].startswith("KBStoreError: PUT https://[REDACTED]@")
    assert "Bearer [REDACTED]" in outcome["reason"]
    assert "the store said [REDACTED] expired" in outcome["reason"]
    assert len(outcome["reason"]) == MAX_REASON_LENGTH


def test_a_refused_flydsl_solution_write_redacts_and_bounds_the_store_error(
    tmp_path,
    monkeypatch,
):
    """The rewrite-owned publish path reports a refusal into the result JSON too.

    ``write_flydsl_kb_solution`` records a validated port directly rather than
    through the facade, and the rewrite runner prints its reason and files it
    under ``kb_experience.write``, so it is sanitized on the same terms.
    """
    token = "kb-store-secret-9f3c"
    store = _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)

    def refuse(*_args, **_kwargs):
        raise _credentialed_store_error(token)

    monkeypatch.setattr(store, "put_knowledge", refuse)

    outcome = flydsl_kb.write_flydsl_kb_solution(
        spec,
        str(driver),
        _remote_config_with_token(tmp_path, token),
        source_ms=2.0,
        flydsl_best_ms=1.0,
        framework="vllm",
    )

    assert outcome["written"] is False
    assert token not in outcome["reason"]
    assert outcome["reason"].startswith("KBStoreError: PUT https://[REDACTED]@")
    assert "Bearer [REDACTED]" in outcome["reason"]
    assert "the store said [REDACTED] expired" in outcome["reason"]
    assert len(outcome["reason"]) == MAX_REASON_LENGTH


def test_a_failed_run_experience_write_redacts_and_bounds_the_store_error(
    tmp_path,
    monkeypatch,
    caplog,
):
    """The forge loop's own mirror reports a store failure into the result JSON.

    ``write_run_experience`` guards the whole mirror so a KB write cannot break
    the loop, and ``write_experience_to_kb`` hands what it returns to the caller
    that files ``kb_experience.write``. Opening the facade is part of what the
    guard covers, so a store client that cannot be built raises through it. The
    warning logged beside the reason lands in the run's log file, so it may not
    keep what the reason had to give up either.
    """
    token = "kb-store-secret-9f3c"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    kernel = workspace / "kernel.py"
    kernel.write_text(
        "import triton\n\n\n@triton.jit\ndef my_kernel(x):\n    return x\n",
        encoding="utf-8",
    )

    def refuse_to_open(_config):
        raise _credentialed_store_error(token)

    monkeypatch.setattr(agent_kb_module, "create_rewrite_record_store", refuse_to_open)

    status = experience_sink.write_run_experience(
        config=_remote_config_with_token(tmp_path, token),
        workspace=str(workspace),
        kernel_path=str(kernel),
        kernel_source=kernel.read_text(encoding="utf-8"),
        kernel_backend="triton",
        gpu_target="gfx950",
        experiment_id="exp-1",
        baseline_wall_ms=10.0,
        best_wall_ms=5.0,
        mean_case_speedup=2.0,
        cumulative_diff="--- a/kernel.py\n+++ b/kernel.py\n",
        digest="iter 1 kept",
        framework="standalone",
        summary_override={
            "category": "gemm",
            "strategy": "vectorize loads",
            "recipe": "Use vectorized loads.",
            "lessons": "Alignment matters.",
        },
    )

    assert status["written"] is False
    assert token not in status["reason"]
    assert status["reason"].startswith("KBStoreError: PUT https://[REDACTED]@")
    assert "Bearer [REDACTED]" in status["reason"]
    assert "the store said [REDACTED] expired" in status["reason"]
    assert len(status["reason"]) == MAX_REASON_LENGTH
    assert token not in caplog.text
    assert status["reason"] in caplog.text
