# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real-acceptance launch: the golden layer with only its acceptance simulation taken out.

It is the server a check of the model's own text runs on after an AgentX test's measured rounds: the same compose as
the measured launch, the draft kept, and neither sglang's ``SGLANG_SIMULATE_ACC_*`` environment nor vLLM's synthetic
rejection sampling inside ``--speculative-config``.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from inference_suite.compose import RoundFacts, compose
from inference_suite.contracts import ConfigError, GoldenUse, Lifecycle, LifecycleMode, ProfileConfig, WorkloadKind
from inference_suite.contracts.config import BaseRef
from inference_suite.contracts.launch import Launch, LaunchError, LayerSource, VersionCheck
from inference_suite.golden import real_acceptance
from inference_suite.tests.test_base_launch_contract import _base, _golden
from inference_suite.tests.test_config_contract import _config, _eval

_FACTS = RoundFacts(config_sha256="d" * 64, installed_version="0.5.23.post1", version_check=VersionCheck.PASSED)
_DRAFT = {"method": "eagle3", "model": "/models/draft", "num_speculative_tokens": 3}
_SIMULATION = {"rejection_sample_method": "synthetic", "synthetic_acceptance_length": 2.75}


def _check_round(base, golden: GoldenUse = GoldenUse.REAL_ACCEPTANCE, **changes):
    server = replace(_config().server, golden=golden, base=BaseRef(path="/b", digest=base.digest()))
    return _config(
        server=server,
        workload=WorkloadKind.NONE,
        agentx=None,
        eval=_eval(enabled=False),
        profile=ProfileConfig(False),
        lifecycle=Lifecycle(LifecycleMode.BOOT, "/pids"),
        **changes,
    )


def _launch(base, golden: GoldenUse = GoldenUse.REAL_ACCEPTANCE) -> Launch:
    launch = compose(_check_round(base, golden), base, _FACTS)
    assert isinstance(launch, Launch), launch
    return launch


def _vllm_base(spec: dict, *extra: str):
    golden = _golden(args=("--speculative-config", json.dumps(spec), *extra), env={})
    return _base(framework="vllm", entrypoint=("vllm", "serve"), args=("--tensor-parallel-size", "8"), golden_al=golden)


def _spec_config(launch: Launch) -> dict:
    (arg,) = [arg for arg in launch.args if arg.flag == "--speculative-config"]
    assert arg.source is LayerSource.GOLDEN
    return json.loads(arg.values[0])


def test_on_sglang_the_simulation_env_goes_and_the_speculative_flags_stay_as_the_golden_layer_set_them():
    base = _base()
    measured, check = _launch(base, GoldenUse.APPLY), _launch(base)

    assert check.argv == measured.argv, "the same compose: only the simulation differs"
    golden_flags = [(arg.flag, arg.values) for arg in check.args if arg.source is LayerSource.GOLDEN]
    assert golden_flags == [("--speculative-algorithm", ("EAGLE",)), ("--speculative-num-steps", ("3",))]
    assert "SGLANG_SIMULATE_ACC_LEN" in measured.env and "SGLANG_SIMULATE_ACC_LEN" not in check.env
    assert {k: v for k, v in measured.env.items() if not k.startswith("SGLANG_SIMULATE_ACC_")} == check.env


def test_on_vllm_the_synthetic_sampler_goes_from_the_speculative_config_and_the_draft_stays():
    base = _vllm_base({**_DRAFT, **_SIMULATION})
    measured, check = _launch(base, GoldenUse.APPLY), _launch(base)

    assert _spec_config(measured) == {**_DRAFT, **_SIMULATION}
    assert _spec_config(check) == _DRAFT
    others = [token for token in check.argv if "speculative" not in token and not token.startswith("{")]
    assert others == [t for t in measured.argv if "speculative" not in t and not t.startswith("{")]


def test_on_vllm_a_rejection_sampler_that_is_not_synthetic_is_the_drafts_own_and_stays():
    base = _vllm_base({**_DRAFT, "rejection_sample_method": "strict", "synthetic_acceptance_length": 2.75})

    assert _spec_config(_launch(base)) == {**_DRAFT, "rejection_sample_method": "strict"}


def test_per_key_speculative_config_flags_that_simulate_go_and_the_others_stay():
    args, env = real_acceptance(
        (
            "--speculative-config", json.dumps(_DRAFT),
            "--speculative-config.rejection_sample_method", "synthetic",
            "--speculative-config.synthetic-acceptance-length", "2.75",
            "--speculative-config.draft_tensor_parallel_size", "1",
        ),
        {"SGLANG_SIMULATE_ACC_METHOD": "match-expected", "SGLANG_OTHER": "1"},
    )  # fmt: skip

    assert args == (
        "--speculative-config", json.dumps(_DRAFT, separators=(",", ":")),
        "--speculative-config.draft_tensor_parallel_size", "1",
    )  # fmt: skip
    assert env == {"SGLANG_OTHER": "1"}


def test_the_launch_records_its_golden_use_and_forces_no_acceptance():
    launch = _launch(_base())
    document = json.loads(json.dumps(launch.to_document()))

    assert (document["golden_use"], document["forced_acceptance"], document["golden"]) == (
        "real_acceptance",
        False,
        None,
    )
    assert Launch.from_document(document) == launch


def test_a_base_without_a_golden_layer_has_no_real_acceptance_launch():
    base = _base(golden_al=None)

    with pytest.raises(ConfigError, match="golden real_acceptance needs a base with a golden layer"):
        _check_round(base).check_base(base)
    with pytest.raises(LaunchError, match="golden real_acceptance needs a base with a golden layer"):
        replace(_launch(_base()), base=base)


def test_a_real_acceptance_round_runs_no_workload():
    server = replace(_config().server, golden=GoldenUse.REAL_ACCEPTANCE)

    with pytest.raises(ConfigError, match="real_acceptance .* runs no workload"):
        _config(server=server, profile=ProfileConfig(False))
