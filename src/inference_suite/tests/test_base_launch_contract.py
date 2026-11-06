# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of ``base_launch.json``: round trip, the digest rule, and the layers a base may not set."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from inference_suite.contracts import (
    BASE_LAUNCH_SCHEMA,
    BaseLaunch,
    BaseLaunchError,
    ClientLaunch,
    GoldenAcceptance,
    GoldenCurve,
    ImageFacts,
    LaunchPoint,
    LaunchQuery,
    LaunchSource,
    PointAlternative,
    Redacted,
    RunnerOwned,
    SourceApi,
    SourceKind,
    Topology,
    VersionBasis,
    Workload,
)
from inference_suite.golden import golden_owned_settings

_COMMIT = "a" * 40
_SHA = "b" * 64


def _golden(**changes) -> GoldenAcceptance:
    golden = GoldenAcceptance(
        value=3.48,
        method="match-expected",
        token_mode="real-draft-token",
        spec_method="EAGLE",
        draft_length=3,
        curve=GoldenCurve(
            repo="https://github.com/SemiAnalysisAI/InferenceX", commit=_COMMIT, path="al.json", sha256=_SHA
        ),
        draft_model="lmsys/draft",
        args=("--speculative-algorithm", "EAGLE", "--speculative-num-steps=3"),
        env={"SGLANG_SIMULATE_ACC_LEN": "3.48"},
    )
    return replace(golden, **changes)


def _base(**changes) -> BaseLaunch:
    base = BaseLaunch(
        query=LaunchQuery(
            model_id="zai-org/GLM-5.3",
            framework="sglang",
            board="mi355x",
            workload=Workload.AGENTX,
            concurrencies=(8, 64),
            tp=8,
        ),
        source=LaunchSource(
            kind=SourceKind.INFERENCEX,
            repo="https://github.com/SemiAnalysisAI/InferenceX",
            commit=_COMMIT,
            entry="glm5.3-fp8-mi355x-sglang-agentic",
            files={"amd-master.yaml": _SHA},
            tools={"srtctl": _COMMIT},
            api=SourceApi(url="https://recipes.example/x.json", response_sha256=_SHA, fetched_unix=1),
        ),
        point=LaunchPoint(
            conc=64, exact=True, alternatives=(PointAlternative(topology=Topology(tp=4), locator="tp4"),)
        ),
        image=ImageFacts(
            run="rocm/sgl-dev:local",
            recipe="lmsysorg/sglang:v0.5.23-rocm700-mi35x",
            framework_version="0.5.23",
            version_basis=VersionBasis.TAG,
        ),
        framework="sglang",
        entrypoint=("python3", "-m", "sglang.launch_server"),
        model_id="zai-org/GLM-5.3",
        topology=Topology(tp=8),
        args=("--tp", "8", '--json-model-override-args={"a": 1}', "--disable-radix-cache"),
        env={"SGLANG_USE_AITER": "1", "HF_TOKEN": Redacted(_SHA)},
        client=ClientLaunch(tool="aiperf", commit=_COMMIT, args=("--seed", "7"), env={"AIPERF_EXPERIMENTAL_FAST": "0"}),
        resolved_unix=1_700_000_000,
        golden_al=_golden(),
        runner_owned=RunnerOwned(args=("--host", "--port"), env=("HIP_VISIBLE_DEVICES",)),
        stripped=("--host",),
        warnings=("no exact point",),
    )
    return replace(base, **changes)


def test_round_trip_through_json_keeps_the_base_and_its_digest():
    base = _base()
    document = json.loads(json.dumps(base.to_document()))

    assert document["schema"] == BASE_LAUNCH_SCHEMA
    assert document["digest"].startswith("sha256:")
    assert document["env"]["HF_TOKEN"] == {"sha256": _SHA}
    assert BaseLaunch.from_document(document) == base
    assert BaseLaunch.from_document(document).to_document() == document


def test_round_trip_without_golden_or_image():
    base = _base(
        query=replace(_base().query, workload=Workload.FIXED),
        source=LaunchSource(kind=SourceKind.DEFAULT_TABLE),
        image=ImageFacts(run=None, recipe=None, framework_version=None, version_basis=VersionBasis.NOT_APPLICABLE),
        golden_al=None,
    )
    assert BaseLaunch.from_document(base.to_document()) == base


def test_a_document_whose_digest_does_not_match_is_refused():
    document = _base().to_document()
    document["args"] = [*document["args"], "--enable-foo"]
    with pytest.raises(BaseLaunchError, match="base_digest_mismatch"):
        BaseLaunch.from_document(document)


def test_unknown_keys_are_refused():
    document = _base().to_document()
    document["extra"] = 1
    with pytest.raises(BaseLaunchError, match="unknown key 'extra'"):
        BaseLaunch.from_document(document)
    nested = _base().to_document()
    nested["golden_al"]["drift"] = 0
    with pytest.raises(BaseLaunchError, match="unknown key 'drift'"):
        BaseLaunch.from_document(nested)


def test_digest_reads_flag_equals_value_and_json_spacing_as_one_launch():
    spaced = _base(args=("--tp=8", "--json-model-override-args", '{"a":1}', "--disable-radix-cache"))
    assert spaced.digest() == _base().digest()
    assert spaced.args == _base().args


def test_digest_keeps_argv_order():
    first = _base(args=("--tp", "4", "--tensor-parallel-size", "8"), topology=Topology(tp=8))
    second = _base(args=("--tensor-parallel-size", "8", "--tp", "4"), topology=Topology(tp=8))
    assert first.digest() != second.digest()


def test_digest_sorts_env_and_hashes_a_redacted_value():
    reordered = _base(env={"HF_TOKEN": Redacted(_SHA), "SGLANG_USE_AITER": "1"})
    assert reordered.digest() == _base().digest()
    assert _base(env={"SGLANG_USE_AITER": "1", "HF_TOKEN": Redacted("c" * 64)}).digest() != _base().digest()


def test_digest_leaves_out_provenance_and_covers_what_runs():
    base = _base()
    provenance = _base(
        source=replace(base.source, entry="another-entry"),
        point=replace(base.point, conc=128, exact=False),
        image=replace(base.image, run="rocm/other:latest"),
        query=replace(base.query, concurrencies=(128,)),
        runner_owned=RunnerOwned(args=("--host",)),
        stripped=(),
        warnings=(),
        resolved_unix=1,
    )
    assert provenance.digest() == base.digest()
    for changed in (
        _base(entrypoint=("sglang", "serve")),
        _base(model_id="zai-org/GLM-5.3-FP8"),
        _base(golden_al=_golden(value=3.5)),
        _base(client=replace(base.client, args=("--seed", "8"))),
        _base(env={"SGLANG_USE_AITER": "0", "HF_TOKEN": Redacted(_SHA)}),
    ):
        assert changed.digest() != base.digest()


@pytest.mark.parametrize(
    ("args", "env", "named"),
    [
        (("--speculative-num-steps", "3"), {}, "--speculative-num-steps"),
        (("--speculative-config", '{"method":"eagle"}'), {}, "--speculative-config.method"),
        (("--speculative-config.num_speculative_tokens", "3"), {}, "--speculative-config.num_speculative_tokens"),
        ((), {"SGLANG_SIMULATE_ACC_LEN": "3"}, "SGLANG_SIMULATE_ACC_LEN"),
    ],
)
def test_args_or_env_setting_a_golden_key_are_refused_when_a_golden_layer_exists(args, env, named):
    with pytest.raises(BaseLaunchError, match=f"golden-owned {named}"):
        _base(args=("--tp", "8", *args), env=env)


def test_a_base_without_golden_may_carry_its_own_draft():
    base = _base(golden_al=None, args=("--tp", "8", "--speculative-num-steps", "3"))
    assert golden_owned_settings([("speculative-num-steps", "3")], {}) == {"--speculative-num-steps": "3"}
    assert BaseLaunch.from_document(base.to_document()) == base


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"args": ("--tp", "8", "--port", "30000")}, "--port"),
        ({"args": ("--tp", "8", "--Port=30000")}, "--port"),
        ({"env": {"HIP_VISIBLE_DEVICES": "0"}}, "HIP_VISIBLE_DEVICES"),
        ({"golden_al": _golden(args=("--speculative-num-steps", "3", "--host", "0.0.0.0"))}, "--host"),
    ],
)
def test_args_or_env_setting_a_runner_owned_key_are_refused(changes, named):
    with pytest.raises(BaseLaunchError, match=f"runner-owned {named}"):
        _base(**changes)


def test_agentx_refuses_the_default_table_and_frameworks_without_agentx_recipes():
    with pytest.raises(BaseLaunchError, match="default table"):
        _base(
            source=LaunchSource(kind=SourceKind.DEFAULT_TABLE),
            image=ImageFacts(run=None, recipe=None, framework_version=None, version_basis=VersionBasis.NOT_APPLICABLE),
        )
    with pytest.raises(BaseLaunchError, match="AgentX base runs"):
        _base(framework="atom")


def test_an_operator_topology_must_be_the_one_launched():
    with pytest.raises(BaseLaunchError, match="asks for tp 8"):
        _base(topology=Topology(tp=4))


@pytest.mark.parametrize(
    ("recipe", "version"),
    [
        ("lmsysorg/sglang:v0.5.3rc0-rocm700-mi35x", "0.5.3rc0"),
        ("vllm/vllm-openai-rocm:v0.10.2rc1", "0.10.2rc1"),
        ("vllm/vllm-openai-rocm:v0.11.0b2", "0.11.0b2"),
        ("lmsysorg/sglang:v0.5.4a1", "0.5.4a1"),
        ("vllm/vllm-openai-rocm:v0.10.2.post1", "0.10.2.post1"),
        ("vllm/vllm-openai-rocm:v0.12.0.dev3", "0.12.0.dev3"),
        ("vllm/vllm-openai:v0.10.1.1", "0.10.1.1"),
    ],
)
def test_a_pre_release_or_post_release_tag_is_a_tag_version(recipe, version):
    image = ImageFacts(run=None, recipe=recipe, framework_version=version, version_basis=VersionBasis.TAG)
    assert BaseLaunch.from_document(_base(image=image).to_document()).image == image


def test_an_image_without_a_version_in_its_tag_assumes_the_latest_release():
    image = ImageFacts(
        run="rocm/vllm:local",
        recipe="vllm/vllm-openai-rocm:nightly",
        framework_version="0.11.2",
        version_basis=VersionBasis.ASSUMED_LATEST,
        assumed_release="v0.11.2",
    )
    base = _base(image=image)
    assert BaseLaunch.from_document(base.to_document()).image == image
    assert base.digest() == _base().digest()
    with pytest.raises(BaseLaunchError, match="names the release"):
        replace(image, assumed_release=None)
    with pytest.raises(BaseLaunchError, match="framework version must match"):
        replace(image, framework_version="nightly")


def test_version_basis_follows_whether_the_source_names_an_image():
    with pytest.raises(BaseLaunchError, match="not_applicable exactly when"):
        _base(
            image=ImageFacts(run=None, recipe=None, framework_version=None, version_basis=VersionBasis.NOT_APPLICABLE)
        )
    with pytest.raises(BaseLaunchError, match="no recipe framework version"):
        ImageFacts(run=None, recipe=None, framework_version="0.5.23", version_basis=VersionBasis.NOT_APPLICABLE)


def test_a_source_names_repo_and_commit_together():
    with pytest.raises(BaseLaunchError, match="together"):
        LaunchSource(kind=SourceKind.INFERENCEX, repo="https://github.com/x/y")


def test_a_wrong_schema_is_refused():
    document = copy.deepcopy(_base().to_document())
    document["schema"] = "hyperloom-base-launch/2"
    with pytest.raises(BaseLaunchError, match="schema"):
        BaseLaunch.from_document(document)
