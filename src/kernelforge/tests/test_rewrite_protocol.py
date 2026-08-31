"""Hermetic tests for the framework apply-back producer contract.

No GPU, no LLM, no git: this pins the version handshake, the logical-name to
builder-symbol rule, and the apply-back manifest schema a consumer integrates
against.
"""

from __future__ import annotations

import json

import pytest

from kernelforge.rewrite_by_flydsl import protocol
from kernelforge.rewrite_by_flydsl.spec import RewriteSpec


def test_capabilities_report_the_supported_protocol():
    capabilities = protocol.capabilities()

    assert capabilities == {
        "rewrite_protocol_version": 2,
        "artifact_schema_versions": [2],
        "driver_contract_versions": [1],
        "frameworks": ["aiter", "vllm", "sglang"],
        "source_languages": ["triton", "hip", "cuda", "cpp"],
        "source_kinds": ["triton", "hip_cpp"],
        "result_sentinel": "__FORGE_RESULT__",
        "driver_preparation": True,
    }
    # A consumer parses this from stdout, so it must round-trip as plain JSON.
    assert json.loads(json.dumps(capabilities)) == capabilities


def test_no_source_without_readable_code_is_advertised_as_portable():
    """A prebuilt binary or hand-written ASM has nothing to port.

    A consumer reads these lists to decide whether to hand work over, so naming
    a source-less kind here would invite a campaign that cannot even start.
    """
    advertised = set(protocol.SUPPORTED_SOURCE_LANGUAGES) | set(protocol.SUPPORTED_SOURCE_KINDS)

    assert not advertised & {"asm", "aiter_asm", "prebuilt", "binary", "hsaco"}


def test_capabilities_are_independent_between_calls():
    first = protocol.capabilities()
    first["frameworks"].append("cuda")

    assert protocol.capabilities()["frameworks"] == ["aiter", "vllm", "sglang"]


def test_plain_identifier_names_keep_their_symbol():
    assert protocol.operator_slug("softmax") == "softmax"
    assert protocol.builder_symbol("softmax") == "build_softmax_module"
    assert protocol.builder_symbol("mxfp8_grouped_gemm") == "build_mxfp8_grouped_gemm_module"


@pytest.mark.parametrize(
    "logical_op_name",
    [
        "vllm::logical_op",
        "attention<128, fp16>",
        "aiter.fused_moe",
        "2fast",
        "class",
        "-",
        "x" * 80,
    ],
)
def test_awkward_names_produce_legal_stable_symbols(logical_op_name):
    symbol = protocol.builder_symbol(logical_op_name)

    assert symbol.isidentifier()
    assert symbol.isascii()
    assert symbol == protocol.builder_symbol(logical_op_name)
    assert symbol.startswith("build_") and symbol.endswith("_module")


def test_names_that_sanitize_alike_stay_distinct():
    first = protocol.builder_symbol("vllm::softmax")
    second = protocol.builder_symbol("vllm/softmax")

    assert first != second
    assert first.isidentifier() and second.isidentifier()


def test_an_empty_logical_name_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        protocol.operator_slug("   ")


def test_the_spec_derives_its_symbol_from_the_logical_name():
    spec = RewriteSpec(
        op_name="vllm::logical_op",
        source_kernel="/ws/op.py",
        target_functions=["op"],
    )

    assert spec.operator_slug == protocol.operator_slug("vllm::logical_op")
    assert spec.builder_symbol == f"build_{spec.operator_slug}_module"
    assert spec.builder_symbol.isidentifier()


@pytest.mark.parametrize(
    "path",
    [
        "forge_experiments",
        "forge_experiments/best_result.json",
        "forge_experiments/rewrite_applyback/result.json",
        ".forge_rewrite/abc123/kernel.py",
        ".forge_driver_1234.py",
        "framework/ops/.forge_driver_tmp",
        "nested/forge_experiments/events.jsonl",
    ],
)
def test_producer_owned_paths_are_recognized(path):
    assert protocol.is_producer_owned_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "framework/dispatch.py",
        "framework/forge_experiments_reader.py",
        "forge_experiments_notes.md",
        "docs/forge_rewrite.md",
        "framework/forge_driver.py",
        "",
    ],
)
def test_framework_owned_paths_are_left_alone(path):
    assert protocol.is_producer_owned_path(path) is False


def test_driver_environment_carries_the_producer_owned_facts():
    environment = protocol.driver_environment(
        source_kernel="/ws/softmax.py",
        candidate_kernel="/ws/.forge_rewrite/kernel.py",
        logical_op_name="vllm::softmax",
    )

    assert environment == {
        "KERNELFORGE_REWRITE_SOURCE_KERNEL": "/ws/softmax.py",
        "KERNELFORGE_REWRITE_CANDIDATE_KERNEL": "/ws/.forge_rewrite/kernel.py",
        "KERNELFORGE_REWRITE_BUILDER_SYMBOL": protocol.builder_symbol("vllm::softmax"),
        "KERNELFORGE_REWRITE_LOGICAL_OP": "vllm::softmax",
    }


def _manifest(**overrides) -> dict:
    payload = {
        "schema_version": 2,
        "artifact_kind": "framework_applyback",
        "validation_scope": "reference",
        "logical_op_name": "vllm::softmax",
        "operator_slug": protocol.operator_slug("vllm::softmax"),
        "builder_symbol": protocol.builder_symbol("vllm::softmax"),
        "source_entry": "softmax",
        "reference_correctness_passed": True,
        "reference_snr_db": 45.0,
        "integration_validation_required": True,
        "integration_validation_status": "pending",
        "base_commit": "b" * 40,
        "commit_hash": "a" * 40,
        "commit_ref": "refs/forge-rewrite/applyback/softmax-aaaaaaaaaaaa",
        "flydsl_best_commit": "c" * 40,
        "baseline_wall_ms": 2.0,
        "best_wall_ms": 1.0,
        "framework": "vllm",
        "changed_files": ["framework/dispatch.py"],
        "artifact_dir": "rewrite_applyback/best/iter_000",
        "patch_path": "rewrite_applyback/best/iter_000/forge.patch",
    }
    payload.update(overrides)
    return payload


def test_a_complete_manifest_validates():
    payload = _manifest()

    assert protocol.validate_applyback_manifest(payload) is payload
    # An unmeasured reference SNR is still publishable.
    assert protocol.validate_applyback_manifest(_manifest(reference_snr_db=None))


@pytest.mark.parametrize("field", sorted(protocol._REQUIRED_MANIFEST_FIELDS))
def test_every_contract_field_is_required(field):
    payload = _manifest()
    del payload[field]

    with pytest.raises(ValueError):
        protocol.validate_applyback_manifest(payload)


def test_an_unknown_schema_version_fails_fast():
    with pytest.raises(ValueError, match="unsupported apply-back manifest schema"):
        protocol.validate_applyback_manifest(_manifest(schema_version=3))
    with pytest.raises(ValueError, match="unsupported apply-back manifest schema"):
        protocol.validate_applyback_manifest(_manifest(schema_version="2"))


def test_manifest_rejects_an_unknown_framework():
    with pytest.raises(ValueError, match="unsupported apply-back framework"):
        protocol.validate_applyback_manifest(_manifest(framework="unknown"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("logical_op_name", 42),
        ("changed_files", "framework/dispatch.py"),
        ("reference_correctness_passed", "yes"),
        ("reference_snr_db", "45.0"),
        ("baseline_wall_ms", True),
        ("integration_validation_required", 1),
    ],
)
def test_a_mistyped_field_fails_fast(field, value):
    with pytest.raises(ValueError, match="wrong type"):
        protocol.validate_applyback_manifest(_manifest(**{field: value}))


def test_the_ambiguous_correctness_key_is_rejected():
    with pytest.raises(ValueError, match="ambiguous field"):
        protocol.validate_applyback_manifest(_manifest(correctness_passed=True))


def test_the_producer_may_not_claim_integration_passed():
    with pytest.raises(ValueError, match="may not publish integration"):
        protocol.validate_applyback_manifest(_manifest(integration_validation_status="passed"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"artifact_kind": "standalone_flydsl"},
        {"validation_scope": "integration"},
        {"commit_hash": ""},
        {"base_commit": ""},
        {"changed_files": []},
        {"changed_files": ["/etc/passwd"]},
        {"changed_files": ["../outside.py"]},
        {"changed_files": ["forge_experiments/best_result.json"]},
        {"changed_files": ["framework/op.py", ".forge_rewrite/id/kernel.py"]},
        {"artifact_dir": ""},
        {"artifact_dir": "/tmp/elsewhere"},
        {"patch_path": "rewrite_applyback/../../forge.patch"},
    ],
)
def test_a_manifest_that_breaks_a_hard_constraint_is_rejected(overrides):
    with pytest.raises(ValueError):
        protocol.validate_applyback_manifest(_manifest(**overrides))


def test_a_non_object_manifest_is_rejected():
    with pytest.raises(ValueError, match="must be a JSON object"):
        protocol.validate_applyback_manifest([])


def test_the_manifest_names_the_producer_owned_path_it_refuses():
    with pytest.raises(ValueError, match="producer-owned state"):
        protocol.validate_applyback_manifest(_manifest(changed_files=["forge_experiments/run_state.json"]))
    # The bundle's own paths live under the campaign root and stay publishable.
    assert protocol.validate_applyback_manifest(_manifest())["artifact_dir"] == ("rewrite_applyback/best/iter_000")


def test_applyback_contract_example_validates_both_documents():
    example = protocol.applyback_contract_example()

    assert protocol.validate_applyback_manifest(example["manifest"]) is example["manifest"]
    assert protocol.validate_applyback_outer_result(example["outer_result"]) is example["outer_result"]
    assert example["manifest"]["commit_hash"] == example["outer_result"]["best_commit"]
    assert example["manifest"]["framework"] == "vllm"


@pytest.mark.parametrize(
    "overrides",
    [
        {"success": False},
        {"applyback_required": False},
        {"applyback_ok": False},
        {"artifact_kind": "standalone_flydsl"},
        {"artifact_schema_version": 1},
        {"best_commit": ""},
        {"canonical_manifest": ""},
        {"temporary_paths": "../scratch"},
    ],
)
def test_outer_result_rejects_a_broken_contract(overrides):
    payload = protocol.applyback_contract_example()["outer_result"]
    payload.update(overrides)

    with pytest.raises(ValueError):
        protocol.validate_applyback_outer_result(payload)
