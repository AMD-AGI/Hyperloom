# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pairing cookbook AgentX launches with InferenceX's golden curves, from answers recorded as data.

The launches come from the sglang and vLLM recipe fixtures, the curves and master configs from the InferenceX
fixtures. The GLM-5.3 answer is what InferenceX's ``golden_length`` returns for its request (at 90fa315 too); the
vLLM one only names a curve the main fixture carries, since the fixture lacks the curve InferenceX would select.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
from functools import cache
from pathlib import Path

import pytest
import yaml

from inference_suite.argv import argv_flag_occurrences, speculative_config
from inference_suite.contracts import LaunchQuery, Workload
from inference_suite.recipe.golden_curve import GoldenLength, GoldenRequest, GoldenTable, golden_requests, with_golden
from inference_suite.recipe.inferencex import GOLDEN_DIR
from inference_suite.recipe.model import RecipeSourceError, RepoBundle, SourceLaunch, adaptive_verification_reason
from inference_suite.recipe.sglang_cookbook import sglang_cookbook_launches
from inference_suite.recipe.vllm_recipes import VllmRecipesSnapshot, vllm_recipes_launches

_FIXTURES = Path(__file__).parent / "fixtures"
_GLM53_MXFP4 = "amd/GLM-5.3-Quark-MXFP4-AttnFP8"
_GLM53_CURVE = "glm-5.3-nvfp4__mtp__native__default-default"
_GLM53_REQUEST = GoldenRequest(
    "glm5.3", {"method": "eagle", "num_speculative_tokens": 5, "model": ""}, "thinking_on", "standard"
)
_KIMI31 = "moonshotai/Kimi-K3.1-Pro"
_KIMI31_CURVE = "kimi-k3.1-pro__dflash__kimi-k3.1-dflash2-preview__default-block"


def _load(*parts: str) -> dict:
    return json.loads(gzip.decompress(_FIXTURES.joinpath(*parts).read_bytes()))


@cache
def _inferencex(commit: str) -> RepoBundle:
    return RepoBundle.from_document(_load("inferencex", f"{commit}.json.gz")["snapshot"]["bundle"])


def _agentx(model: str, framework: str = "sglang") -> LaunchQuery:
    return LaunchQuery(model, framework, "mi355x", Workload.AGENTX, (8,))


def _glm53_launch() -> SourceLaunch:
    (launch,) = sglang_cookbook_launches(
        _agentx(_GLM53_MXFP4), RepoBundle.from_document(_load("sglang", "c843608d39.json.gz"))
    )
    return launch


def _kimi_vllm_launches() -> tuple[SourceLaunch, ...]:
    snapshot = VllmRecipesSnapshot.from_document(_load("vllm_recipes", "20261106.json.gz"))
    return vllm_recipes_launches(LaunchQuery("moonshotai/Kimi-K3", "vllm", "mi355x", Workload.AGENTX, (4,)), snapshot)


def _table(bundle: RepoBundle, request: GoldenRequest, answer: GoldenLength) -> GoldenTable:
    return GoldenTable(bundle, {request.key: answer})


def test_glm53_cookbook_launch_reads_3_48_from_the_recorded_table():
    bundle = _inferencex("a1e8f975")
    launch = _glm53_launch()
    assert golden_requests(_agentx(_GLM53_MXFP4), [launch, launch], bundle, None) == (_GLM53_REQUEST,)

    table = _table(bundle, _GLM53_REQUEST, GoldenLength(3.48, curve=_GLM53_CURVE))
    paired = with_golden(launch, _agentx(_GLM53_MXFP4), table, None)

    path = f"{GOLDEN_DIR}/{_GLM53_CURVE}.yaml"
    assert paired.golden.value == 3.48 == yaml.safe_load(bundle.text(path))["glm-5.3-nvfp4"]["thinking_on"][5]
    assert (paired.golden.curve.repo, paired.golden.curve.commit, paired.golden.curve.path) == (
        bundle.repo,
        bundle.commit,
        path,
    )
    assert paired.golden.curve.sha256 == bundle.sha256(path)
    assert paired.env == {
        **launch.env,
        "SGLANG_SIMULATE_ACC_LEN": "3.48",
        "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
        "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token",
    }
    assert paired.args == launch.args


def test_model_prefix_falls_back_to_the_hf_base_model_one_level_only():
    main = _inferencex("ce35bc935")
    launch = _glm53_launch()

    assert golden_requests(_agentx(_GLM53_MXFP4), [launch], main, "zai-org/GLM-5.3") == (_GLM53_REQUEST,)
    assert golden_requests(_agentx(_GLM53_MXFP4), [launch], main, None) == ()
    unpaired = with_golden(launch, _agentx(_GLM53_MXFP4), GoldenTable(main, {}), "org/unknown-base")
    assert unpaired.golden.value is None
    assert "no model prefix for amd/GLM-5.3-Quark-MXFP4-AttnFP8 or its base model org/unknown-base" in (
        unpaired.golden.reason
    )
    assert unpaired.env == launch.env


def test_a_draft_with_no_golden_curve_runs_with_a_null_golden_and_its_reason():
    bundle = _inferencex("a1e8f975")
    table = _table(bundle, _GLM53_REQUEST, GoldenLength(None, error="No committed golden curve glm-5.3.yaml"))
    paired = with_golden(_glm53_launch(), _agentx(_GLM53_MXFP4), table, None)

    assert paired.golden.value is None and paired.golden.curve is None
    assert paired.golden.reason == "InferenceX has no golden curve for it: No committed golden curve glm-5.3.yaml"
    assert "SGLANG_SIMULATE_ACC_LEN" not in paired.env


@pytest.mark.parametrize(
    "args",
    [
        ["--speculative-config.enable_adaptive_verification", "true"],
        ["--speculative-config", '{"enable_adaptive_verification": true}'],
    ],
)
def test_adaptive_verification_reads_the_flag_text_and_the_json_bool_alike(args):
    assert adaptive_verification_reason(speculative_config(argv_flag_occurrences(args))) is not None


def test_adaptive_verification_off_gives_no_reason():
    args = ["--speculative-config", '{"enable_adaptive_verification": false}']
    assert adaptive_verification_reason(speculative_config(argv_flag_occurrences(args))) is None


def test_vllm_adaptive_verification_gives_a_null_golden_and_asks_nothing():
    main = _inferencex("ce35bc935")
    launch = _kimi_vllm_launches()[1]
    adaptive = dataclasses.replace(
        launch, args=(*launch.args, "--speculative-config.enable_adaptive_verification", "true")
    )

    assert golden_requests(_agentx(_KIMI31, "vllm"), [adaptive], main, None) == ()
    paired = with_golden(adaptive, _agentx(_KIMI31, "vllm"), GoldenTable(main, {}), None)
    assert paired.golden.value is None
    assert "adaptive verification is on" in paired.golden.reason
    assert paired.args == adaptive.args


def test_vllm_launch_forces_the_golden_length_in_one_speculative_config():
    main = _inferencex("ce35bc935")
    launch = _kimi_vllm_launches()[1]
    (request,) = golden_requests(_agentx(_KIMI31, "vllm"), [launch], main, None)
    assert (request.model_prefix, request.spec["method"], request.spec["num_speculative_tokens"], request.verify) == (
        "kimik3.1",
        "dspark",
        5,
        "block",
    )

    paired = with_golden(
        launch, _agentx(_KIMI31, "vllm"), _table(main, request, GoldenLength(3.43, _KIMI31_CURVE)), None
    )

    occurrences = argv_flag_occurrences(paired.args)
    assert [name for name, _ in occurrences].count("speculative-config") == 1
    assert speculative_config(occurrences) == {
        **request.spec,
        "rejection_sample_method": "synthetic",
        "synthetic_acceptance_length": 3.43,
    }
    assert paired.golden.value == 3.43


def test_a_launch_that_drafts_nothing_is_left_as_it_is():
    launch = _kimi_vllm_launches()[0]
    assert with_golden(launch, _agentx(_KIMI31, "vllm"), GoldenTable(_inferencex("ce35bc935"), {}), None) is launch


def test_an_unrecorded_request_is_an_error_not_a_null_golden():
    with pytest.raises(RecipeSourceError, match="records no answer"):
        with_golden(_glm53_launch(), _agentx(_GLM53_MXFP4), GoldenTable(_inferencex("a1e8f975"), {}), None)


def test_table_round_trips_and_a_length_is_a_value_or_an_error():
    table = _table(_inferencex("a1e8f975"), _GLM53_REQUEST, GoldenLength(3.48, curve=_GLM53_CURVE))
    assert GoldenTable.from_document(json.loads(json.dumps(table.to_document()))) == table

    for value, curve, error in ((None, None, None), (3.48, None, None), (3.48, _GLM53_CURVE, "boom"), (0.0, "c", None)):
        with pytest.raises(RecipeSourceError):
            GoldenLength(value, curve, error)
