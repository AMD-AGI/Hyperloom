"""Hermetic tests for standalone FlyDSL rewrite KB reuse."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from kernelforge.config import Config
from kernelforge.knowledge.experience_store import (
    REMOTE_BACKEND_KB_STORE,
    KnowledgeConfig,
    KnowledgeStoreMode,
)
from kernelforge.rewrite_by_flydsl import driver_contract, kb, record_store
from kernelforge.rewrite_by_flydsl.identity import (
    framework_version,
    segment,
    session_id,
)
from kernelforge.rewrite_by_flydsl.spec import RewriteSpec

VLLM_VERSION = framework_version("vllm")
SOFTMAX_IDENTITY = f"kernel:flydsl:softmax:vllm:{VLLM_VERSION}:flydsl:mi355x"


class InMemoryKBStore:
    """The subset of the KB Store surface the rewrite records use."""

    def __init__(self, *_args, **_kwargs):
        self.knowledge: dict[tuple[str, str], dict] = {}
        self.files: dict[tuple[str, str], dict[str, bytes]] = {}
        self.champions: dict[str, dict] = {}
        self.order: list[tuple[str, str]] = []
        self.downloads: list[tuple[str, str]] = []

    def get_rollup(self, canonical_id):
        sessions = [
            {"session_id": session_id, "updated_at": f"{index:04d}"}
            for index, (identity, session_id) in enumerate(self.order)
            if identity == canonical_id
        ]
        if not sessions and canonical_id not in self.champions:
            return None
        return {"sessions": sessions, "champion": self.champions.get(canonical_id, {})}

    def get_top_sessions(
        self,
        canonical_id,
        *,
        metric="speedup",
        limit=3,
        offset=0,
    ):
        champion_id = str(self.champions.get(canonical_id, {}).get("session_id") or "")
        ranked = []
        for index, (identity, candidate_session_id) in enumerate(self.order):
            if identity != canonical_id:
                continue
            score = self.knowledge[(identity, candidate_session_id)].get(metric)
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                continue
            ranked.append(
                {
                    "session_id": candidate_session_id,
                    "score": float(score),
                    "updated_at": f"{index:04d}",
                    "is_champion": candidate_session_id == champion_id,
                }
            )
        ranked.sort(
            key=lambda item: (
                -item["score"],
                -int(item["updated_at"]),
                item["session_id"],
            )
        )
        return {"sessions": ranked[offset : offset + limit]}

    def get_session(self, canonical_id, session_id):
        knowledge = self.knowledge.get((canonical_id, session_id))
        return (
            None
            if knowledge is None
            else {
                "canonical_id": canonical_id,
                "session_id": session_id,
                "knowledge": knowledge,
            }
        )

    def list_session_files(self, canonical_id, session_id, *, kind=""):
        del kind
        return {
            "files": [
                {
                    "path": rel_path,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size": len(content),
                    "download_url": f"memory://{rel_path}",
                }
                for rel_path, content in self.files.get((canonical_id, session_id), {}).items()
            ]
        }

    def put_knowledge(self, canonical_id, knowledge, *, session_id="", mode="merge"):
        # Mirrors the SDK: "merge" shallow-merges over the stored section and
        # "replace" overwrites it. Always replacing would let a caller that
        # relies on merge keeping the other fields pass here and lose them
        # against the real store.
        if mode not in ("merge", "replace"):
            raise record_store.KBStoreError(f"mode must be 'merge' or 'replace', got {mode!r}")
        key = (canonical_id, session_id)
        if key not in self.knowledge:
            self.order.append(key)
        if mode == "merge":
            merged = dict(self.knowledge.get(key) or {})
            merged.update(knowledge)
            self.knowledge[key] = merged
        else:
            self.knowledge[key] = dict(knowledge)
        return {"session_id": session_id, "mode": mode}

    def put_file(self, canonical_id, session_id, rel_path, local_path, *, kind="other", meta=None):
        self.files.setdefault((canonical_id, session_id), {})[rel_path] = Path(local_path).read_bytes()
        return f"kb://{canonical_id}/{session_id}/{rel_path}"

    def download_session(self, canonical_id, session_id, destination, *, include_values=True):
        del include_values
        self.downloads.append((canonical_id, session_id))
        root = Path(destination) / "files"
        root.mkdir(parents=True, exist_ok=True)
        for rel_path, content in self.files.get((canonical_id, session_id), {}).items():
            target = root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

    def set_champion(self, canonical_id, session_id, *, metric="throughput", value=0.0):
        self.champions[canonical_id] = {
            "session_id": session_id,
            "metric": metric,
            "value": value,
        }
        return {}


def _spec(tmp_path):
    workspace = tmp_path / "workspace"
    source = workspace / "vllm" / "softmax.py"
    source.parent.mkdir(parents=True)
    source.write_text("import triton\n@triton.jit\ndef softmax_kernel(x):\n    return x\n")
    kernel = workspace / "kernel.py"
    kernel.write_text("import flydsl\ndef build_softmax_module(config):\n    return lambda inputs: inputs['x']\n")
    driver = workspace / "driver.py"
    driver.write_text("# stable rewrite driver contract\n")
    return (
        RewriteSpec(
            op_name="softmax",
            source_kernel=str(source),
            target_functions=["softmax_kernel"],
            flydsl_kernel=str(kernel),
            workspace=str(workspace),
            snr_threshold=30.0,
        ),
        driver,
    )


def _remote_config(tmp_path):
    knowledge = KnowledgeConfig.from_env(
        {},
        mode="remote",
        local_root=tmp_path / "knowledge",
        kb_store_url="http://in-memory",
        kb_store_token="token",
        remote_backend=REMOTE_BACKEND_KB_STORE,
    )
    return Config.from_env(
        workspace=str(tmp_path),
        gpu_target="gfx950",
        gpu_type="mi355x",
        knowledge_config=knowledge,
        agent_precheck=False,
    )


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
        ),
    )


def _use_in_memory_kb_store(monkeypatch):
    store = InMemoryKBStore()
    monkeypatch.setattr(record_store, "KBStoreClient", lambda *a, **k: store)
    return store


# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #
def test_a_rewrite_is_filed_under_the_flydsl_producer_identity(tmp_path, monkeypatch):
    store = _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)

    written = kb.write_flydsl_kb_solution(
        spec,
        str(driver),
        _remote_config(tmp_path),
        source_ms=10.0,
        flydsl_best_ms=5.0,
        best_commit="a" * 40,
        framework="vllm",
        snr_db=80.0,
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
    # A logical name carries separators the identity and the session id both use,
    # so an unnormalized one would let the operator re-partition either of them.
    store = _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    spec.op_name = "vllm::softmax"

    written = kb.write_flydsl_kb_solution(
        spec,
        str(driver),
        _remote_config(tmp_path),
        source_ms=10.0,
        flydsl_best_ms=5.0,
        best_commit="a" * 40,
        framework="vllm",
        snr_db=80.0,
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
        remote_backend=REMOTE_BACKEND_KB_STORE,
    )
    other_gpu = Config.from_env(
        workspace=str(tmp_path),
        gpu_target="gfx950",
        gpu_type="mi300x",
        knowledge_config=knowledge,
        agent_precheck=False,
    )

    kb.write_flydsl_kb_solution(
        spec,
        str(driver),
        _remote_config(tmp_path),
        source_ms=10.0,
        flydsl_best_ms=5.0,
        best_commit="a" * 40,
        framework="vllm",
    )
    kb.write_flydsl_kb_solution(
        spec,
        str(driver),
        other_gpu,
        source_ms=10.0,
        flydsl_best_ms=5.0,
        best_commit="a" * 40,
        framework="vllm",
    )

    assert sorted({identity for identity, _ in store.knowledge}) == [
        f"kernel:flydsl:softmax:vllm:{VLLM_VERSION}:flydsl:mi300x",
        SOFTMAX_IDENTITY,
    ]
    # Artifact keys are partitioned by session id alone, so two identities
    # sharing one would put both ports on one object and let the second
    # overwrite the first.
    assert len({session for _, session in store.knowledge}) == 2


def test_gpu_target_does_not_change_the_recipe_identity(tmp_path, monkeypatch):
    store = _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    gfx950 = _remote_config(tmp_path)
    gfx942 = _remote_config(tmp_path)
    gfx942.gpu_target = "gfx942"

    first = kb.write_flydsl_kb_solution(
        spec,
        str(driver),
        gfx950,
        source_ms=10.0,
        flydsl_best_ms=5.0,
        best_commit="a" * 40,
        framework="vllm",
    )
    second = kb.write_flydsl_kb_solution(
        spec,
        str(driver),
        gfx942,
        source_ms=10.0,
        flydsl_best_ms=5.0,
        best_commit="b" * 40,
        framework="vllm",
    )

    assert first["canonical_id"] == second["canonical_id"] == SOFTMAX_IDENTITY
    assert {document["value"]["metric"]["gpu_arch"] for document in store.knowledge.values()} == {"gfx942", "gfx950"}


def test_a_session_id_stays_inside_the_length_the_store_allows():
    overlong = "a" * 200
    generated = session_id(
        f"kernel:flydsl:{overlong}:vllm:0.1:flydsl:mi355x",
        overlong,
        "b" * 40,
    )
    assert record_store.validate_session_id(generated) == generated


def test_a_session_id_is_stable_so_one_port_stays_one_candidate():
    first = session_id(SOFTMAX_IDENTITY, "softmax", "c" * 40)
    second = session_id(SOFTMAX_IDENTITY, "softmax", "c" * 40)
    assert first == second


# --------------------------------------------------------------------------- #
# round trip
# --------------------------------------------------------------------------- #
def test_a_recorded_port_is_materialized_and_revalidated(tmp_path, monkeypatch):
    _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    config = _remote_config(tmp_path)
    expected = Path(spec.flydsl_kernel).read_bytes()

    written = kb.write_flydsl_kb_solution(
        spec,
        str(driver),
        config,
        source_ms=10.0,
        flydsl_best_ms=5.0,
        best_commit="a" * 40,
        framework="vllm",
        snr_db=80.0,
    )
    assert written["written"] is True

    _passing_validation(monkeypatch, best_ms=12.0)
    Path(spec.flydsl_kernel).write_text("def skeleton():\n    pass\n")

    restored = asyncio.run(
        kb.try_flydsl_kb_warmstart(
            spec,
            str(driver),
            config,
            source_ms=10.0,
            framework="vllm",
        )
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
        spec,
        str(driver),
        config,
        source_ms=10.0,
        flydsl_best_ms=5.0,
        best_commit="a" * 40,
        framework="vllm",
    )
    assert written["written"] is True
    _passing_validation(monkeypatch, best_ms=5.0)
    Path(spec.flydsl_kernel).write_bytes(b"def skeleton():\n    pass\n")

    restored = asyncio.run(
        kb.try_flydsl_kb_warmstart(
            spec,
            str(driver),
            config,
            source_ms=10.0,
            framework="vllm",
        )
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
        spec,
        str(driver),
        config,
        source_ms=10.0,
        flydsl_best_ms=5.0,
        best_commit="b" * 40,
        framework="vllm",
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
        kb.try_flydsl_kb_warmstart(
            spec,
            str(driver),
            config,
            source_ms=10.0,
            framework="vllm",
        )
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
        spec,
        str(driver),
        _remote_config(tmp_path),
        source_ms=10.0,
        flydsl_best_ms=5.0,
        best_commit="a" * 40,
        framework="vllm",
    )

    key = (SOFTMAX_IDENTITY, written["session_id"])
    value = store.knowledge[key]["value"]
    assert value["flydsl_kernel"] == "kernel.py"
    assert store.files[key] == {"kernel.py": expected}
    assert not any("content" in name for name in value)


# --------------------------------------------------------------------------- #
# champion is a pointer, not a filter
# --------------------------------------------------------------------------- #
def test_a_correct_but_slower_port_is_recorded_without_being_promoted(
    tmp_path,
    monkeypatch,
):
    store = _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    config = _remote_config(tmp_path)

    rejected = kb.write_flydsl_kb_solution(
        spec,
        str(driver),
        config,
        source_ms=5.0,
        flydsl_best_ms=10.0,
        framework="vllm",
    )
    assert rejected == {"written": False, "reason": "no_improvement"}

    written = kb.write_flydsl_kb_solution(
        spec,
        str(driver),
        config,
        source_ms=5.0,
        flydsl_best_ms=10.0,
        best_commit="b" * 40,
        framework="vllm",
        allow_non_improving=True,
    )

    assert written["written"] is True
    assert written["speedup"] == 0.5
    assert written["champion"] is False
    assert SOFTMAX_IDENTITY not in store.champions

    _passing_validation(monkeypatch, best_ms=10.0, snr_db=None)
    Path(spec.flydsl_kernel).write_text("def skeleton():\n    pass\n")
    restored = asyncio.run(
        kb.try_flydsl_kb_warmstart(
            spec,
            str(driver),
            config,
            source_ms=5.0,
            framework="vllm",
        )
    )
    assert restored.applied is True


def test_a_weaker_later_port_does_not_take_the_champion_pointer(tmp_path, monkeypatch):
    store = _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    config = _remote_config(tmp_path)

    strong = kb.write_flydsl_kb_solution(
        spec,
        str(driver),
        config,
        source_ms=10.0,
        flydsl_best_ms=2.0,
        best_commit="1" * 40,
        framework="vllm",
    )
    Path(spec.flydsl_kernel).write_text("import flydsl\nRANK = 2\n")
    weak = kb.write_flydsl_kb_solution(
        spec,
        str(driver),
        config,
        source_ms=10.0,
        flydsl_best_ms=8.0,
        best_commit="2" * 40,
        framework="vllm",
    )

    assert strong["champion"] is True
    assert weak["written"] is True
    assert weak["champion"] is False
    assert store.champions[SOFTMAX_IDENTITY]["session_id"] == strong["session_id"]
    assert len(store.knowledge) == 2


# --------------------------------------------------------------------------- #
# contract gates
# --------------------------------------------------------------------------- #
def test_a_changed_driver_contract_is_rejected_and_the_seed_restored(
    tmp_path,
    monkeypatch,
):
    _use_in_memory_kb_store(monkeypatch)
    spec, driver = _spec(tmp_path)
    config = _remote_config(tmp_path)
    written = kb.write_flydsl_kb_solution(
        spec,
        str(driver),
        config,
        source_ms=10.0,
        flydsl_best_ms=5.0,
        framework="vllm",
    )
    assert written["written"] is True

    seed = "def skeleton():\n    pass\n"
    Path(spec.flydsl_kernel).write_text(seed)
    driver.write_text("# changed contract\n")

    restored = asyncio.run(
        kb.try_flydsl_kb_warmstart(
            spec,
            str(driver),
            config,
            source_ms=10.0,
            framework="vllm",
        )
    )

    assert restored.applied is False
    assert restored.attempts[-1]["reason"] == "driver_contract_changed"
    assert "Historical FlyDSL rewrite references" in restored.reference_context
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
            spec,
            str(driver),
            config,
            source_ms=10.0,
            flydsl_best_ms=best_ms,
            best_commit=str(rank) * 40,
            framework="vllm",
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
        lambda *args, **kwargs: driver_contract.PreflightReport(ok=True, timing_ms=4.0),
    )
    Path(spec.flydsl_kernel).write_text("def skeleton():\n    pass\n")

    restored = asyncio.run(
        kb.try_flydsl_kb_warmstart(
            spec,
            str(driver),
            config,
            source_ms=10.0,
            framework="vllm",
            top_k=3,
        )
    )

    assert restored.applied is True
    assert [attempt["reason"] for attempt in restored.attempts] == [
        "correctness_failed",
        "correctness_failed",
        "applied",
    ]
    assert "Reference 1" in restored.reference_context
    assert "Reference 2" in restored.reference_context


# --------------------------------------------------------------------------- #
# local mode uses the same record layout
# --------------------------------------------------------------------------- #
def test_local_mode_stores_the_same_record_shape_on_disk(tmp_path, monkeypatch):
    spec, driver = _spec(tmp_path)
    expected = Path(spec.flydsl_kernel).read_bytes()
    knowledge, config = _local_config(
        tmp_path,
        spec,
        gbrain_base_url="https://ambient.invalid",
        gbrain_token="ambient-secret",
    )

    written = kb.write_flydsl_kb_solution(
        spec,
        str(driver),
        config,
        source_ms=10.0,
        flydsl_best_ms=12.0,
        best_commit="d" * 40,
        framework="vllm",
        allow_non_improving=True,
    )

    assert written["written"] is True
    assert config.gbrain_url == ""
    assert config.gbrain_token == ""
    session_dir = knowledge.rewrite_root / Path(*SOFTMAX_IDENTITY.split(":")) / "sessions" / written["session_id"]
    document = json.loads((session_dir / "knowledge.json").read_text())
    assert document["value"]["flydsl_kernel"] == "kernel.py"
    assert (session_dir / "files" / "kernel.py").read_bytes() == expected

    _passing_validation(monkeypatch, best_ms=12.0)
    Path(spec.flydsl_kernel).write_text("def skeleton():\n    pass\n")
    restored = asyncio.run(
        kb.try_flydsl_kb_warmstart(
            spec,
            str(driver),
            config,
            source_ms=10.0,
            framework="vllm",
        )
    )

    assert restored.applied is True
    assert Path(spec.flydsl_kernel).read_bytes() == expected


def test_local_mode_never_reaches_for_ambient_credentials(tmp_path, monkeypatch):
    spec, driver = _spec(tmp_path)
    monkeypatch.delenv("KNOWLEDGE_STORE_MODE", raising=False)
    monkeypatch.delenv("KNOWLEDGE_LOCAL_ROOT", raising=False)
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "user-data"))
    monkeypatch.setenv("GBRAIN_BASE_URL", "https://ambient.invalid")
    monkeypatch.setenv("GBRAIN_TOKEN", "ambient-secret")
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
        spec,
        str(driver),
        config,
        source_ms=10.0,
        flydsl_best_ms=12.0,
        best_commit="e" * 40,
        framework="vllm",
        allow_non_improving=True,
    )

    assert written["written"] is True
    assert config.knowledge_config.mode.value == "local"
    assert config.knowledge_config.kb_store_url == ""
    assert config.gbrain_url == ""


# --------------------------------------------------------------------------- #
# configuration
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

    write = kb.write_flydsl_kb_solution(
        spec,
        str(driver),
        config,
        source_ms=10.0,
        flydsl_best_ms=5.0,
        framework="vllm",
    )
    read = asyncio.run(
        kb.try_flydsl_kb_warmstart(
            spec,
            str(driver),
            config,
            source_ms=10.0,
            framework="vllm",
        )
    )

    assert write == {"written": False, "reason": "missing_gpu_type"}
    assert read.read_reason == "missing_gpu_type"
    assert store.knowledge == {}
    assert store.downloads == []


def test_remote_rewrite_asks_for_the_credentials_it_will_actually_use():
    with pytest.raises(ValueError, match="KB_STORE_URL and KB_STORE_TOKEN"):
        KnowledgeConfig.from_env(
            {"KNOWLEDGE_STORE_MODE": "remote", "KNOWLEDGE_LOCAL_ROOT": "/tmp/kf"},
            remote_backend=REMOTE_BACKEND_KB_STORE,
        )


def test_remote_default_accepts_kb_store_without_gbrain():
    config = KnowledgeConfig.from_env(
        {
            "KNOWLEDGE_STORE_MODE": "remote",
            "KNOWLEDGE_LOCAL_ROOT": "/tmp/kf",
            "KB_STORE_URL": "http://kb",
            "KB_STORE_TOKEN": "tok",
        },
    )
    assert config.kb_store_url == "http://kb"
    assert config.gbrain_base_url == ""


def test_an_unrenderable_segment_falls_back_to_a_readable_address():
    """A dimension that folds away must not silently become an empty address."""
    assert segment("", fallback="unknown") == "unknown"
    assert segment(":::", fallback="unknown") == "unknown"


def test_kb_store_alone_activates_the_rewrite_path_without_gbrain():
    config = KnowledgeConfig.from_env(
        {
            "KNOWLEDGE_STORE_MODE": "remote",
            "KNOWLEDGE_LOCAL_ROOT": "/tmp/kf",
            "KB_STORE_URL": "http://kb",
            "KB_STORE_TOKEN": "tok",
        },
        remote_backend=REMOTE_BACKEND_KB_STORE,
    )

    assert config.kb_store_url == "http://kb"
    assert config.gbrain_base_url == ""


def test_gbrain_alone_leaves_the_rewrite_store_unconfigured():
    config = KnowledgeConfig.from_env(
        {
            "KNOWLEDGE_STORE_MODE": "remote",
            "KNOWLEDGE_LOCAL_ROOT": "/tmp/kf",
            "GBRAIN_BASE_URL": "http://gbrain",
            "GBRAIN_TOKEN": "tok",
        }
    )

    assert config.gbrain_base_url == "http://gbrain"
    assert config.kb_store_url == ""
    assert record_store.create_rewrite_record_store(config) is None


def test_rewrite_validates_its_kb_store_pair_without_using_gbrain():
    with pytest.raises(ValueError, match="KB_STORE_TOKEN"):
        KnowledgeConfig.from_env(
            {
                "KNOWLEDGE_STORE_MODE": "remote",
                "KNOWLEDGE_LOCAL_ROOT": "/tmp/kf",
                "GBRAIN_BASE_URL": "http://gbrain",
                "GBRAIN_TOKEN": "tok",
                "KB_STORE_URL": "http://kb",
            },
            remote_backend=REMOTE_BACKEND_KB_STORE,
        )


def test_remote_without_kb_store_credentials_reads_as_a_cold_start(tmp_path):
    spec, driver = _spec(tmp_path)
    # Built directly: from_env refuses this combination, which is exactly how a
    # misconfigured run is caught at startup. This covers the path that stays
    # reachable when a caller supplies its own configuration.
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
        spec,
        str(driver),
        config,
        source_ms=10.0,
        flydsl_best_ms=5.0,
        framework="vllm",
    )
    read = asyncio.run(
        kb.try_flydsl_kb_warmstart(
            spec,
            str(driver),
            config,
            source_ms=10.0,
            framework="vllm",
        )
    )

    assert written == {"written": False, "reason": "not_configured"}
    assert read.applied is False
    assert read.read_reason == "not_configured"


def test_kb_store_rewrite_keeps_the_measurement_a_consumer_recorded(tmp_path):
    """The remote backend must preserve a measurement across a replacing write.

    ``write`` replaces the session document so a rewrite cannot leave stale
    fields behind, but the measured value is the one field its producer never
    wrote: a consumer recorded it after running the candidate, and the ranking
    trusts it over the claim. Replacing it away would restore the inflated claim
    the measurement exists to correct.
    """
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
