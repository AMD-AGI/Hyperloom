# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of ``config.yaml``: round trip, unknown keys refused, golden exclude bound to a golden base, probe required."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from inference_suite.contracts import (
    CONFIG_SCHEMA,
    DEFAULT_PRE_READY_SILENCE_SEC,
    AgentxWorkload,
    BaseRef,
    ConfigError,
    Deadlines,
    EvalConfig,
    EvalProbe,
    FixedWorkload,
    GoldenUse,
    GpuTelemetry,
    LaunchDelta,
    Lifecycle,
    LifecycleMode,
    MlperfWorkload,
    ProfileConfig,
    RunnerLayer,
    RunPaths,
    ScriptableWorkload,
    ServerConfig,
    SuiteConfig,
    WorkloadKind,
)
from inference_suite.tests.test_base_launch_contract import _base

_SHA = "c" * 64
_FIXED = FixedWorkload(
    concurrency=64,
    isl=1024,
    osl=1024,
    random_range_ratio=0.8,
    num_prompts=640,
    num_warmups=128,
    seed=0,
    warmup_seed=1,
    tokenizer_mode="auto",
    trust_remote_code=True,
)
_AGENTX = AgentxWorkload(
    concurrency=8,
    dataset="agentx-v1",
    num_prompts=40,
    warmup_requests_per_lane=2,
    seed=42,
    warmup_seed=43,
    failed_request_threshold=0.1,
    client_env={"AIPERF_EXPERIMENTAL_FAST": "0"},
)

_MLPERF = MlperfWorkload(
    concurrency=16,
    trajectories=150,
    flow="smoke_test",
    model="glm",
    hardware="mi355x",
    dataset="/data/agentic_combined_v6.jsonl",
    tokenizer="/data/tokenizer",
    endpoints_dir="/opt/mlperf-endpoints",
)


def _eval(**changes) -> EvalConfig:
    probe = EvalProbe(enabled=True, min_samples=20, length_ratio=0.9)
    return replace(EvalConfig(enabled=True, num_fewshot=5, concurrency=32, probe=probe, limit=200), **changes)


def _config(base_digest: str | None = None, **changes) -> SuiteConfig:
    config = SuiteConfig(
        server=ServerConfig(
            base=BaseRef(path="/s/base_launch.json", digest=base_digest or _base().digest()),
            model_path="/models/GLM-5.3",
            served_model_name="glm",
            host="127.0.0.1",
            port=30000,
            gpus=(0, 1, 2, 3, 4, 5, 6, 7),
            framework_python="/opt/venv/bin/python",
            user=LaunchDelta(args_add=("--context-length=65536",), env_unset=("SGLANG_X",)),
            candidate=LaunchDelta(args_add=("--cuda-graph-max-bs", "64"), args_remove=("--disable-radix-cache",)),
            runner=RunnerLayer(
                env_prepend={"PYTHONPATH": "/patches"},
                env_set={"HYPERLOOM_REAP_MARK": "x"},
                args_floor={"--watchdog-timeout": 600},
            ),
            source_files={"patch.py": _SHA},
            absent_source_files=("gone.py",),
            env_inherit=("HF_HOME", "E2B_SANDBOX_ID"),
        ),
        workload=WorkloadKind.AGENTX,
        agentx=_AGENTX,
        eval=_eval(enabled=False),
        telemetry=GpuTelemetry(enabled=True, interval_sec=0.5),
        lifecycle=Lifecycle(mode=LifecycleMode.OWN),
        deadlines=Deadlines(boot_sec=1800, round_sec=3600, eval_sec=1200, silence_sec=600),
        paths=RunPaths(run_dir="/s/runs/r1", tool_cache="/s/tools", tool_manifest_sha256=_SHA),
        profile=ProfileConfig(enabled=True, agentx_open_after_sec=0, agentx_duration_sec=30),
    )
    return replace(config, **changes)


def _accuracy_round(**changes) -> SuiteConfig:
    server = replace(_config().server, golden=GoldenUse.EXCLUDE)
    return _config(
        server=server, workload=WorkloadKind.NONE, agentx=None, eval=_eval(), profile=ProfileConfig(False), **changes
    )


@pytest.mark.parametrize("config", [_config(), _accuracy_round()], ids=["agentx-profile", "accuracy-round"])
def test_round_trip_through_json(config):
    document = json.loads(json.dumps(config.to_document()))

    assert document["schema"] == CONFIG_SCHEMA
    assert document["eval"]["max_tokens_cap"] == 4096
    assert document["deadlines"]["pre_ready_silence_sec"] == 900
    assert SuiteConfig.from_document(document) == config
    assert SuiteConfig.from_document(document).to_document() == document


def test_round_trip_fixed_and_scriptable():
    fixed = _config(workload=WorkloadKind.FIXED, agentx=None, fixed=_FIXED, profile=ProfileConfig(False))
    script = ScriptableWorkload(params={"XDIT_NUM_STEPS": 28, "XDIT_GUIDANCE_SCALE": 4.0})
    scriptable = _config(workload=WorkloadKind.SCRIPTABLE, agentx=None, scriptable=script, profile=ProfileConfig(False))
    for config in (fixed, scriptable):
        assert SuiteConfig.from_document(json.loads(json.dumps(config.to_document()))) == config


def _mlperf_config(**changes) -> SuiteConfig:
    return _config(
        **{
            "workload": WorkloadKind.MLPERF,
            "agentx": None,
            "mlperf": _MLPERF,
            "profile": ProfileConfig(False),
            **changes,
        }
    )


def test_round_trip_mlperf():
    config = _mlperf_config()
    document = json.loads(json.dumps(config.to_document()))

    assert document["workload"] == {"kind": "mlperf"}
    assert document["mlperf"]["endpoints_dir"] == "/opt/mlperf-endpoints"
    assert SuiteConfig.from_document(document) == config
    document["mlperf"]["surprise"] = 1
    with pytest.raises(ConfigError, match="unknown key 'surprise'"):
        SuiteConfig.from_document(document)


def test_the_mlperf_harness_requests_the_model_the_server_serves():
    with pytest.raises(ConfigError, match="requests 'kimi-k3', the server serves 'glm'"):
        _mlperf_config(mlperf=replace(_MLPERF, model="kimi-k3"))


@pytest.mark.parametrize("name", ["flow", "model", "hardware", "dataset", "tokenizer", "endpoints_dir"])
def test_every_mlperf_knob_is_explicit(name):
    with pytest.raises(ConfigError, match=f"the MLPerf {name}"):
        replace(_MLPERF, **{name: ""})


def test_an_mlperf_round_is_not_profiled():
    with pytest.raises(ConfigError, match="a mlperf profile needs a fixed, AgentX or scriptable workload"):
        _mlperf_config(profile=ProfileConfig(enabled=True, agentx_open_after_sec=0, agentx_duration_sec=30))


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("server",),
        ("server", "user"),
        ("server", "user", "args"),
        ("server", "runner"),
        ("server", "env"),
        ("agentx",),
        ("profile",),
        ("eval",),
        ("eval", "probe"),
        ("telemetry",),
        ("telemetry", "gpu"),
        ("lifecycle",),
        ("deadlines",),
        ("paths",),
        ("workload",),
    ],
)
def test_unknown_key_is_refused_at_every_level(path):
    document = copy.deepcopy(_config().to_document())
    target = document
    for key in path:
        target = target[key]
    target["surprise"] = 1

    with pytest.raises(ConfigError, match="unknown key 'surprise'"):
        SuiteConfig.from_document(document)


def test_unknown_schema_is_refused():
    with pytest.raises(ConfigError, match="schema"):
        SuiteConfig.from_document({**_config().to_document(), "schema": "inference-suite-config/2"})


def test_golden_exclude_is_refused_on_a_base_without_golden():
    base = _base(golden_al=None)
    config = _accuracy_round()
    config = replace(config, server=replace(config.server, base=BaseRef(path="/b", digest=base.digest())))

    with pytest.raises(ConfigError, match="golden exclude needs a base with a golden layer"):
        config.check_base(base)
    _accuracy_round().check_base(_base())


def test_golden_exclude_is_only_the_accuracy_round():
    server = replace(_config().server, golden=GoldenUse.EXCLUDE)
    with pytest.raises(ConfigError, match="accuracy round"):
        _config(server=server, profile=ProfileConfig(False))


def test_base_with_another_digest_is_refused():
    with pytest.raises(ConfigError, match="base_digest_mismatch"):
        _config(base_digest="sha256:" + "0" * 64).check_base(_base())


def test_base_of_another_workload_is_refused():
    config = _config(workload=WorkloadKind.FIXED, agentx=None, fixed=_FIXED, profile=ProfileConfig(False))
    with pytest.raises(ConfigError, match="fixed round needs a fixed base"):
        config.check_base(_base())


@pytest.mark.parametrize("missing", ["min_samples", "length_ratio", "enabled"])
def test_eval_probe_fields_are_required(missing):
    document = _config().to_document()
    del document["eval"]["probe"][missing]
    with pytest.raises(ConfigError, match=f"the eval probe must set {missing}"):
        SuiteConfig.from_document(document)


@pytest.mark.parametrize(
    ("section", "missing", "match"),
    [
        ("deadlines", "boot_sec", "the boot_sec deadline"),
        ("deadlines", "silence_sec", "the silence_sec deadline"),
        ("profile", "enabled", "the profile's enabled"),
    ],
)
def test_required_deadline_and_profile_keys_are_config_errors(section, missing, match):
    document = _config().to_document()
    del document[section][missing]
    with pytest.raises(ConfigError, match=match):
        SuiteConfig.from_document(document)


def test_defaulted_deadline_keys_may_be_omitted():
    document = _config().to_document()
    del document["deadlines"]["pre_ready_silence_sec"]
    del document["deadlines"]["deadline_unix"]
    assert SuiteConfig.from_document(document).deadlines == replace(
        _config().deadlines, pre_ready_silence_sec=DEFAULT_PRE_READY_SILENCE_SEC, deadline_unix=None
    )


def test_eval_probe_is_required():
    document = _config().to_document()
    del document["eval"]["probe"]
    with pytest.raises(ConfigError, match="must set probe"):
        SuiteConfig.from_document(document)
    document["eval"] = None
    with pytest.raises(ConfigError, match="must set eval"):
        SuiteConfig.from_document(document)
    with pytest.raises(ConfigError, match="probe must be set"):
        EvalConfig(enabled=True, num_fewshot=5, concurrency=8, probe=None)


def test_workload_section_must_match_the_kind():
    with pytest.raises(ConfigError, match="sets \\['fixed'\\]"):
        _config(workload=WorkloadKind.FIXED, profile=ProfileConfig(False))
    with pytest.raises(ConfigError, match="boots a server or runs eval"):
        _config(workload=WorkloadKind.NONE, agentx=None, profile=ProfileConfig(False))
    _config(workload=WorkloadKind.NONE, agentx=None, profile=ProfileConfig(False), lifecycle=Lifecycle("boot", "/p"))


def test_profile_owns_its_server_and_names_its_window():
    with pytest.raises(ConfigError, match="owns its server"):
        _config(lifecycle=Lifecycle(LifecycleMode.REUSE, pid_dir="/p"))
    with pytest.raises(ConfigError, match="agentx profile needs agentx_duration_sec"):
        _config(profile=ProfileConfig(enabled=True, agentx_open_after_sec=0))
    with pytest.raises(ConfigError, match="disabled profile"):
        ProfileConfig(enabled=False, max_iters=3)


@pytest.mark.parametrize(("mode", "pid_dir"), [("boot", None), ("reuse", None), ("own", "/p"), ("external", "/p")])
def test_pid_dir_is_set_exactly_for_boot_and_reuse(mode, pid_dir):
    with pytest.raises(ConfigError, match="pid_dir"):
        Lifecycle(mode, pid_dir)


@pytest.mark.parametrize(
    ("client_env", "match"),
    [
        ({"AIPERF_EXPERIMENTAL_FAST": "1"}, "AIPERF_EXPERIMENTAL_FAST=0"),
        ({}, "AIPERF_EXPERIMENTAL_FAST=0"),
        ({"AIPERF_EXPERIMENTAL_FAST": "0", "HF_TOKEN": "x"}, "AIPERF_\\* only"),
    ],
)
def test_agentx_client_env_is_aiperf_only_with_fast_mode_off(client_env, match):
    with pytest.raises(ConfigError, match=match):
        replace(_AGENTX, client_env=client_env)


@pytest.mark.parametrize("name", ["XDIT_QUALITY_LPIPS_MAX", "XDIT_QUALITY_SSIM_MIN", "quality_threshold"])
def test_scriptable_params_carry_no_threshold(name):
    with pytest.raises(ConfigError, match="threshold"):
        ScriptableWorkload(params={name: 0.1})


@pytest.mark.parametrize(
    ("name", "match"), [("RESULT_DIR", "set by the round"), ("not-an-env", "environment variable name")]
)
def test_scriptable_params_are_env_names_the_round_does_not_set(name, match):
    with pytest.raises(ConfigError, match=match):
        ScriptableWorkload(params={name: 1})


def test_layers_refuse_contradictions():
    with pytest.raises(ConfigError, match="both sets and unsets"):
        LaunchDelta(env_set={"A": "1"}, env_unset=("A",))
    with pytest.raises(ConfigError, match="long flag"):
        LaunchDelta(args_remove=("tp",))
    with pytest.raises(ConfigError, match="both prepends to and sets"):
        RunnerLayer(env_prepend={"PATH": "/x"}, env_set={"PATH": "/y"})
    with pytest.raises(ConfigError, match="both present and absent"):
        replace(_config().server, source_files={"a.py": _SHA}, absent_source_files=("a.py",))


def test_added_args_are_normalized():
    assert LaunchDelta(args_add=("--tp=8",)).args_add == ("--tp", "8")


def test_only_reuse_and_external_rounds_find_their_server_running() -> None:
    assert {mode for mode in LifecycleMode if mode.finds_server} == {LifecycleMode.REUSE, LifecycleMode.EXTERNAL}
