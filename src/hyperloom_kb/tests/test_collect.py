from __future__ import annotations

import copy
import hashlib
import json
import threading
import urllib.error
from pathlib import Path
from typing import Any

import pytest
import yaml

from hyperloom_kb import (
    PACKAGED_DECLARATION,
    ConfigurationError,
    ExperienceDeclaration,
    ExperienceHTTPService,
    FieldDeclaration,
    HTTPServiceConfig,
    LLMQueryPlanner,
    NoOpExperienceKB,
    ObjectiveDeclaration,
    ObjectiveDirection,
    PlannerConfiguration,
    RemoteClient,
    RemoteConfig,
    RemoteExperienceKB,
    create_http_server,
    load_declaration,
)
from hyperloom_kb.collect import (
    MappingError,
    SourceDocumentError,
    collect,
    compile_mapping,
    load_mapping,
)
from hyperloom_kb.collect.cli import main as collect_main
from hyperloom_kb.collect.sensitive import find_sensitive

MAPPING = "hyperloom-sbd-v6"
TOKEN = "collect-test-token"
REASONING = "static_recon flags the hipBMM linear route as disabled on this build."
PATCH = "--- a/vllm/x.py\n+++ b/vllm/x.py\n@@ -1 +1 @@\n-a = 1\n+a = 2\n"


def _attempt(attempt_id: str, **fields: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "attempt_id": attempt_id,
        "arm": "config",
        "ts": "2026-09-21T22:52:56Z",
        "round_id": "explore-001",
        "task_id": "t-explore-7",
        "proposal_ref": "p-1",
        "variant_name": attempt_id,
        "outcome": "REVERT",
        "reason": "gain_below_threshold",
        "reasoning": REASONING,
        "reasoning_origin": "action_payload.reasoning",
        "measured_against": {
            "throughput": 4752.57,
            "extra_server_args": "--max-num-seqs 256",
            "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
            "remove_args": ["--b", "--a", "--a"],
            "unset_envs": [],
            "args_mode": None,
        },
        "config_delta": {"extra_envs": {"VLLM_ROCM_USE_AITER_LINEAR_HIPBMM": "1"}},
        "measurement": {"before_tput": 4752.57, "after_tput": 4745.71, "gain_pct": -0.144},
        "accuracy": {"required": None, "reference": None, "value": None, "passed": None},
        "failure": {"error_class": "", "error_excerpt": "", "attribution": ""},
        "gates": [{"gate": "keep_threshold", "passed": False, "observed": -0.144}],
    }
    row.update(fields)
    return row


def _sbd(*attempts: dict[str, Any], benchmark_mode: str = "throughput") -> dict[str, Any]:
    return {
        "metadata": {
            "session": {"session_id": "run-mistral-1"},
            "grading": {"benchmark_mode": benchmark_mode},
            "task_config": {
                "model_name": "Mistral-7B-Instruct-v0.3",
                "gpu_type": "mi300x",
                "framework_name": "vllm",
                "framework_version": "0.29.0",
                "precision": "bf16",
                "tp": 1,
                "ep": 2,
                "conc": 64,
                "compute_partition": {"mode": "cpx", "partitions": 4},
                "architecture": {
                    "model_type": "mistral",
                    "architectures": ["MistralForCausalLM"],
                },
            },
        },
        "timeline": [
            {"type": "phase", "id": "prelude:0:phase"},
            {
                "type": "framework_agent",
                "id": "framework_agent:0:framework",
                "start_time": "2026-09-21T22:10:22+00:00",
                "ext": {
                    "proposals": [
                        {
                            "proposal_id": "p-1",
                            "reasoning": "Proposal-level rationale for this configuration grid.",
                            "kb_read_id": "kb-read-1",
                            "rendered_refs": [{"id": "exp-000750a291015fe29cf702840d789dc7", "purpose": ""}],
                        },
                        {"proposal_id": "c-1", "title": "Enable the fused linear path"},
                    ],
                    "attempts": list(attempts),
                },
            },
        ],
    }


def _source_attempt(attempt_id: str = "source-1", **fields: Any) -> dict[str, Any]:
    row = _attempt(
        attempt_id,
        arm="source",
        outcome="KEEP",
        proposal_ref="c-1",
        candidate_id="c-1",
        reasoning_origin="action_params.reasoning",
        source_ref="https://github.com/vllm-project/vllm/pull/1",
        target_files=["vllm/x.py"],
        measurement={"before_tput": 4752.57, "after_tput": 4900.0, "gain_pct": 3.1},
        patch_material=[
            {
                "path": "patches/a.diff",
                "sha256": hashlib.sha256(PATCH.encode()).hexdigest(),
                "content": PATCH,
            }
        ],
    )
    row.pop("config_delta")
    row.update(fields)
    return row


def _dry(document: dict[str, Any]) -> dict[str, Any]:
    return collect(MAPPING, document, dry_run=True).to_dict()


def _collected(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["unit_id"]: row["experience"] for row in report["collected"]}


def _skipped(report: dict[str, Any]) -> dict[str, str]:
    return {row["unit_id"]: row["reason"] for row in report["skipped"]}


def test_packaged_mapping_produces_the_packaged_declaration() -> None:
    mapping = load_mapping(MAPPING)
    assert mapping.declaration.schema_ref == load_declaration(PACKAGED_DECLARATION).schema_ref
    assert mapping.producer.name == "hyperloom-framework"


def test_config_attempt_projects_a_complete_experience() -> None:
    report = _dry(_sbd(_attempt("revert-1")))
    assert report["counts"]["collected"] == 1
    experience = _collected(report)["revert-1"]

    assert experience["identity"] == {
        "model": "Mistral-7B-Instruct-v0.3",
        "gpu": "mi300x",
        "framework": "vllm",
        "framework_version": "0.29.0",
        "model_type": "mistral",
        "architecture": "MistralForCausalLM",
        "precision": "bf16",
        "tp": 1,
        "ep": 2,
        "conc": 64,
        "compute_partition_mode": "CPX",
        "partitions": 4,
    }
    baseline = {
        "args_mode": "append",
        "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
        "extra_server_args": "--max-num-seqs 256",
        "remove_args": ["--a", "--b"],
        "unset_envs": [],
    }
    canonical = json.dumps(baseline, sort_keys=True, separators=(",", ":"))
    assert experience["baseline_identity"] == {"baseline_fingerprint": hashlib.sha256(canonical.encode()).hexdigest()}
    assert experience["preconditions"][2] == f"materialized_baseline_configuration={canonical}"
    assert experience["preconditions"][0] == "measured_baseline_tput=4752.57"

    change = experience["change"]
    assert change["kind"] == "config_variant"
    assert json.loads(change["content"])["extra_envs"] == {"VLLM_ROCM_USE_AITER_LINEAR_HIPBMM": "1"}
    assert change["identity"]["change_fingerprint"] == hashlib.sha256(change["content"].encode()).hexdigest()
    assert experience["reasoning"] == REASONING
    assert experience["rendered_refs"] == [{"id": "exp-000750a291015fe29cf702840d789dc7", "purpose": "representative"}]
    assert experience["outcome"] == {
        "decision": "revert",
        "value": 4745.71,
        "constraints": [{"name": "keep_threshold", "passed": False, "value": -0.144}],
        "error_class": "",
    }
    assert experience["reflection"].startswith('Recorded outcome: {"after_tput":4745.71,')
    assert experience["provenance"]["extra"]["kb_read_id"] == "kb-read-1"
    assert experience["provenance"]["source_ref"] == "session:run-mistral-1:attempt:revert-1"


def test_projection_is_deterministic_across_runs() -> None:
    document = _sbd(_attempt("revert-1"), _source_attempt())
    first, second = _dry(document), _dry(copy.deepcopy(document))
    assert _collected(first) == _collected(second)
    ids = [row["experience_id"] for row in first["collected"]]
    assert len(set(ids)) == 2


def test_source_attempt_carries_patch_material_from_the_document() -> None:
    experience = _collected(_dry(_sbd(_source_attempt())))["source-1"]
    change = experience["change"]
    content = json.loads(change["content"])
    assert content["patches"] == [
        {
            "path": "patches/a.diff",
            "sha256": hashlib.sha256(PATCH.encode()).hexdigest(),
            "content": PATCH,
        }
    ]
    assert change["identity"]["change_fingerprint"] == hashlib.sha256(PATCH.encode()).hexdigest()
    assert change["summary"] == "Enable the fused linear path"
    assert change["resource_refs"] == ["patches/a.diff"]


def test_a_source_attempt_that_changed_only_configuration_is_a_config_experience() -> None:
    delta = {"extra_server_args": "--enable-fused-moe", "extra_envs": {"VLLM_FUSED_MOE": "1"}}
    attempt = _source_attempt("levers-1", patch_material=[], config_delta=delta, variant_name="fused-moe-routing")
    experience = _collected(_dry(_sbd(attempt)))["levers-1"]

    change = experience["change"]
    assert change["kind"] == "config_variant"
    assert change["identity"]["change_family"] == "config_variant"
    assert json.loads(change["content"]) == {**delta, "remove_args": [], "unset_envs": [], "args_mode": "append"}
    assert change["identity"]["change_fingerprint"] == hashlib.sha256(change["content"].encode()).hexdigest()
    assert change["summary"] == "fused-moe-routing"
    assert change["resource_refs"] == []
    assert experience["provenance"]["extra"]["arm"] == "source"


def test_multi_patch_fingerprint_hashes_the_patch_list() -> None:
    second = {"path": "patches/b.diff", "sha256": "b" * 64, "content": "+b\n"}
    attempt = _source_attempt()
    attempt["patch_material"] = [*attempt["patch_material"], second]
    change = _collected(_dry(_sbd(attempt)))["source-1"]["change"]
    patches = json.loads(change["content"])["patches"]
    canonical = json.dumps(patches, sort_keys=True, separators=(",", ":"))
    assert change["identity"]["change_fingerprint"] == hashlib.sha256(canonical.encode()).hexdigest()


def test_quality_gates_skip_with_a_reason() -> None:
    report = _dry(
        _sbd(
            _attempt("no-reasoning", reasoning="", reasoning_origin="", proposal_ref="none"),
            _attempt("generic", reasoning="Optimize.", proposal_ref="none"),
            _attempt("untraceable", reasoning_origin="post_action_result.reasoning", proposal_ref="x"),
            _attempt(
                "killed",
                outcome="KILLED_OVERTIME",
                measurement={"before_tput": 4752.57, "after_tput": None},
                failure={"error_class": "", "attribution": "unknown"},
            ),
            _attempt(
                "accuracy-failed",
                outcome="KEEP",
                accuracy={"required": True, "passed": False, "value": 0.4},
            ),
            _attempt("no-delta", config_delta={}),
            _source_attempt("no-patch", patch_material=[]),
        )
    )
    assert report["counts"]["collected"] == 0
    assert _skipped(report) == {
        "no-reasoning": "decision reasoning is missing or non-specific",
        "generic": "decision reasoning is missing or non-specific",
        "untraceable": "decision reasoning is not traceable to the action-time proposal",
        "killed": "failed attempt is not candidate-attributed",
        "accuracy-failed": "kept attempt did not pass required accuracy",
        "no-delta": "config attempt has no effective config_delta",
        "no-patch": "source attempt has no durable patch material",
    }


def test_reasoning_falls_back_to_the_proposal() -> None:
    experience = _collected(_dry(_sbd(_attempt("fallback", reasoning="", reasoning_origin=""))))["fallback"]
    assert experience["reasoning"] == "Proposal-level rationale for this configuration grid."
    assert experience["provenance"]["extra"]["reasoning_origin"] == "proposal.reasoning"


def test_candidate_caused_failure_becomes_a_failed_experience() -> None:
    attempt = _attempt(
        "failed",
        outcome="FAILED",
        measurement={"before_tput": 4752.57, "after_tput": None},
        failure={"error_class": "capability_unsupported", "attribution": "candidate_caused"},
        gates=[],
    )
    outcome = _collected(_dry(_sbd(attempt)))["failed"]["outcome"]
    assert outcome == {
        "decision": "failed",
        "value": None,
        "constraints": [],
        "error_class": "capability_unsupported",
    }


def test_accuracy_is_added_only_when_no_gate_ruled_on_it() -> None:
    attempt = _attempt(
        "accuracy",
        accuracy={"required": True, "reference": 0.498, "value": 0.51, "passed": True},
    )
    constraints = _collected(_dry(_sbd(attempt)))["accuracy"]["outcome"]["constraints"]
    assert constraints == [
        {"name": "keep_threshold", "passed": False, "value": -0.144},
        {"name": "accuracy", "passed": True, "value": 0.51},
    ]


def test_document_skip_rule_blocks_every_unit() -> None:
    report = _dry(_sbd(_attempt("a"), _attempt("b"), benchmark_mode="AgentX"))
    assert report["blocked_reason"] == "agentx_experience_identity_not_supported"
    assert set(_skipped(report).values()) == {"agentx_experience_identity_not_supported"}
    assert report["counts"]["units"] == 2


def test_credentials_are_never_collected() -> None:
    report = _dry(
        _sbd(
            _attempt("env-secret", config_delta={"extra_envs": {"HF_TOKEN": "hf-abc"}}),
            _attempt("flag-secret", config_delta={"extra_server_args": "--api-key abc123"}),
            _attempt("bearer", reasoning=f"{REASONING} Bearer abcdefghijklmnop"),
        )
    )
    skipped = _skipped(report)
    assert set(skipped) == {"env-secret", "flag-secret", "bearer"}
    assert all(reason.startswith("sensitive content in") for reason in skipped.values())


def test_prose_is_not_mistaken_for_a_credential_assignment() -> None:
    reasoning = "Decode is launch bound; cost per token: dominated by kernel launch overhead."
    assert find_sensitive({"reasoning": reasoning}) is None
    assert find_sensitive({"preconditions": [f"note={reasoning}"]}) is not None
    assert find_sensitive({"content": '{"TOKENIZERS_PARALLELISM":"false"}'}) is None


def test_an_attempts_experience_citations_reach_its_provenance() -> None:
    citation = {"id": "exp-00000000000000000000000000000001", "stance": "avoid", "claim": "It regressed decode."}
    report = _dry(_sbd(_attempt("revert-1", experience_citations=[citation]), _attempt("revert-2")))
    collected = _collected(report)
    assert collected["revert-1"]["provenance"]["extra"]["experience_citations"] == [citation]
    assert collected["revert-2"]["provenance"]["extra"]["experience_citations"] == []


def test_a_large_patch_is_collected_whole() -> None:
    huge = "+" * (3 * 1024 * 1024)
    attempt = _source_attempt(patch_material=[{"path": "patches/a.diff", "sha256": "a" * 64, "content": huge}])
    report = _dry(_sbd(attempt))
    assert report["skipped"] == []
    assert huge in _collected(report)["source-1"]["change"]["content"]


def test_evaluation_errors_skip_only_the_unit() -> None:
    report = _dry(
        _sbd(
            _attempt("bad-envs", config_delta={"extra_envs": ["not", "a", "map"]}),
            _attempt("good"),
        )
    )
    assert _skipped(report)["bad-envs"].startswith("mapping evaluation failed")
    assert set(_collected(report)) == {"good"}


class _UnusedPlanner:
    model = "unused-planner"

    def complete(self, **_kwargs: Any) -> str:
        raise AssertionError("collection never plans a read")


def test_publish_to_the_experience_service(tmp_path: Path) -> None:
    declaration = load_mapping(MAPPING).declaration
    app = ExperienceHTTPService(
        HTTPServiceConfig(tmp_path / "service", TOKEN),
        declaration,
        LLMQueryPlanner(_UnusedPlanner(), PlannerConfiguration.create("unused-planner")),
    )
    server = create_http_server(app, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    thread.start()
    try:
        host, port = server.server_address[:2]
        client = RemoteClient(RemoteConfig(f"http://{host!s}:{port}", TOKEN, spool_root=tmp_path / "spool"))
        kb = RemoteExperienceKB(client, declaration)
        document = _sbd(_attempt("revert-1"), _source_attempt())
        first = collect(MAPPING, document, kb=kb)
        second = collect(MAPPING, document, kb=kb)
        listed = client.list_experiences()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert [row.status for row in first.collected] == ["created", "created"]
    assert [row.status for row in second.collected] == ["unchanged", "unchanged"]
    assert {item["experience_id"] for item in listed.items} == {row.experience_id for row in first.collected}


def test_unavailable_service_spools_the_write(tmp_path: Path) -> None:
    def offline(*_args: Any, **_kwargs: Any) -> Any:
        raise urllib.error.URLError("offline")

    declaration = load_mapping(MAPPING).declaration
    client = RemoteClient(
        RemoteConfig("http://service.invalid", TOKEN, spool_root=tmp_path / "spool"),
        opener=offline,
    )
    report = collect(MAPPING, _sbd(_attempt("revert-1")), kb=RemoteExperienceKB(client, declaration))
    assert [row.status for row in report.collected] == ["spooled"]
    assert len(list((tmp_path / "spool").glob("spool-*.json"))) == 1


def test_unconfigured_target_writes_nothing(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    report = collect(MAPPING, _sbd(_attempt("revert-1")), kb=NoOpExperienceKB(), receipt=receipt)
    assert report.enabled is False
    assert report.collected == ()
    assert json.loads(receipt.read_text())["enabled"] is False


def test_declaration_mismatch_fails_before_any_write(tmp_path: Path) -> None:
    other = ExperienceDeclaration(
        identity=(FieldDeclaration("model", "Model."),),
        baseline_identity=(FieldDeclaration("config", "Config."),),
        change_identity=(FieldDeclaration("knob", "Knob."),),
        objectives=(ObjectiveDeclaration("e2e_throughput@v1", ObjectiveDirection.HIGHER_IS_BETTER, "T."),),
        decisions=("keep", "revert", "failed"),
    )
    requests: list[Any] = []

    def record(request: Any, **_kwargs: Any) -> Any:
        requests.append(request)
        raise urllib.error.URLError("no request should be sent")

    client = RemoteClient(RemoteConfig("http://service.invalid", TOKEN, spool_root=tmp_path / "spool"), opener=record)
    with pytest.raises(ConfigurationError, match="produces"):
        collect(MAPPING, _sbd(_attempt("revert-1")), kb=RemoteExperienceKB(client, other))
    assert requests == []
    assert not (tmp_path / "spool").exists()


def _mapping_document(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "format": "hyperloom-kb.collect.v1",
        "declaration": str(PACKAGED_DECLARATION),
        "producer": {"name": "demo", "version": "1"},
        "units": [{"each": "$doc.items", "as": "item"}],
        "experience": {
            "run_id": "run",
            "seq": 0,
            "completed_at": "2026-09-21T22:52:56Z",
            "identity": "$item.identity",
            "objective": "e2e_throughput@v1",
            "baseline_value": 1,
            "baseline_identity": {"object": {"baseline_fingerprint": "base"}},
            "reasoning": "reasoning text",
            "change": {"identity": "$item.change", "summary": "summary"},
            "outcome": {"decision": "keep", "value": 2},
            "reflection": "reflection",
        },
    }
    document.update(overrides)
    return document


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"format": "other"}, "mapping format"),
        ({"units": []}, "units must be a non-empty list"),
        ({"let": {"x": {"nonexistent": 1}}}, "exactly one built-in call"),
        ({"let": {"x": {"text": 1, "lower": 2}}}, "exactly one built-in call"),
        ({"let": {"x": "$missing.path"}}, "not bound here"),
        ({"let": {"x": {"eq": [1]}}}, "exactly 2 arguments"),
        ({"let": {"x": "$bad path"}}, "not a valid path"),
        ({"let": {"x": {"map": "$item", "table": {}, "extra": 1}}}, "does not accept extra"),
        ({"surprise": 1}, "unknown keys"),
        ({"experience": {"run_id": "r"}}, "is required"),
        ({"require": [{"check": True}]}, "reason must be a non-empty string"),
    ],
)
def test_malformed_mappings_fail_at_load(override: dict[str, Any], message: str) -> None:
    with pytest.raises(MappingError, match=message):
        compile_mapping(_mapping_document(**override))


def test_custom_mapping_collects_a_custom_document() -> None:
    mapping = compile_mapping(
        _mapping_document(
            experience={
                **_mapping_document()["experience"],
                "seq": {"hash48": ["$item.name"]},
                "reasoning": "Tune {$item.name} because the item asked for it.",
            }
        )
    )
    document = {
        "items": [
            {
                "name": "alpha",
                "identity": {
                    "model": "m",
                    "gpu": "g",
                    "framework": "f",
                    "model_type": "t",
                    "architecture": "a",
                    "framework_version": "1",
                    "precision": "bf16",
                },
                "change": {"change_family": "config_variant", "change_fingerprint": "abc"},
            }
        ]
    }
    report = collect(mapping, document, dry_run=True)
    assert report.skipped == ()
    assert report.collected[0].experience.reasoning == "Tune alpha because the item asked for it."


def test_mapping_file_resolves_its_declaration_relative_to_itself(tmp_path: Path) -> None:
    (tmp_path / "decl.yaml").write_text(PACKAGED_DECLARATION.read_text(encoding="utf-8"))
    (tmp_path / "mapping.yaml").write_text(yaml.safe_dump(_mapping_document(declaration="decl.yaml")), encoding="utf-8")
    assert load_mapping(tmp_path / "mapping.yaml").declaration.schema_ref.startswith("schema:")


def test_cli_dry_run_prints_the_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    document = tmp_path / "session_breakdown.json"
    document.write_text(json.dumps(_sbd(_attempt("revert-1"))), encoding="utf-8")
    receipt = tmp_path / "reports" / "collect.json"
    code = collect_main(["--mapping", MAPPING, "--document", str(document), "--dry-run", "--receipt", str(receipt)])
    printed = json.loads(capsys.readouterr().out)
    assert code == 0
    assert printed["counts"]["collected"] == 1
    assert json.loads(receipt.read_text()) == printed


def test_cli_rejects_a_missing_document(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = collect_main(["--mapping", MAPPING, "--document", str(tmp_path / "absent.json")])
    assert code == 2
    assert "cannot read source document" in capsys.readouterr().err


def test_document_must_match_the_unit_shape() -> None:
    with pytest.raises(SourceDocumentError, match="does not match mapping units"):
        collect(MAPPING, {"timeline": {"not": "a list"}}, dry_run=True)
