# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of ``compose`` and ``launch.json``: layer order, flag sources, golden and runner keys, forced acceptance."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from inference_suite.compose import DEFAULT_SEED, GPU_MASK_ENV, Refusal, RoundFacts, compose, runner_owned
from inference_suite.contracts import FailedStep, GoldenUse, LaunchDelta, RunnerLayer
from inference_suite.contracts.launch import (
    LAUNCH_SCHEMA,
    ClientInvocation,
    EvalInvocation,
    Launch,
    LaunchError,
    LayerSource,
    Removal,
    VersionCheck,
)
from inference_suite.recipe.image_version import recipe_image_facts
from inference_suite.tests.test_base_launch_contract import _base
from inference_suite.tests.test_config_contract import _SHA, _accuracy_round, _config

_FACTS = RoundFacts(
    config_sha256="d" * 64,
    installed_version="0.5.23.post1",
    version_check=VersionCheck.PASSED,
    tool_pins={"inferencex": "e" * 40},
    client=ClientInvocation(argv=("python", "-m", "bench"), env={"AIPERF_X": "1"}, seed=0, warmup_seed=1),
    eval=EvalInvocation(
        argv=("python", "-I", "-c", "main()", "/src", "request.json"), definition_digest="sha256:" + _SHA
    ),
)


def _compose(base=None, **server_changes) -> Launch | Refusal:
    base = base or _base()
    server = _config().server
    server = replace(server, base=replace(server.base, digest=base.digest()), **server_changes)
    return compose(_config(server=server), base, _FACTS)


def _launch(base=None, **server_changes) -> Launch:
    launch = _compose(base, **server_changes)
    assert isinstance(launch, Launch), launch
    return launch


def _refusal(base=None, **server_changes) -> Refusal:
    refusal = _compose(base, **server_changes)
    assert isinstance(refusal, Refusal), refusal
    return refusal


def _sources(launch: Launch) -> list[tuple[str, tuple[str, ...], str, tuple[str, ...]]]:
    return [(arg.flag, arg.values, arg.source.value, tuple(s.value for s in arg.overrode)) for arg in launch.args]


def test_layers_apply_in_the_fixed_order():
    launch = _launch(
        user=LaunchDelta(args_add=("--chunked-prefill-size", "4096", "--tp", "4"), env_set={"SGLANG_USE_AITER": "0"}),
        candidate=LaunchDelta(
            args_add=("--chunked-prefill-size", "8192", "--watchdog-timeout", "300"),
            args_remove=("--disable-radix-cache",),
        ),
        runner=RunnerLayer(env_prepend={"PYTHONPATH": "/farm"}, args_floor={"--watchdog-timeout": 600}),
    )

    assert _sources(launch) == [
        ("--json-model-override-args", ('{"a":1}',), "base", ()),
        ("--tp", ("4",), "user", ("base",)),
        ("--chunked-prefill-size", ("8192",), "candidate", ("user",)),
        ("--speculative-algorithm", ("EAGLE",), "golden", ()),
        ("--speculative-num-steps", ("3",), "golden", ()),
        ("--model-path", ("/models/GLM-5.3",), "guard", ()),
        ("--host", ("127.0.0.1",), "guard", ()),
        ("--port", ("30000",), "guard", ()),
        ("--served-model-name", ("glm",), "guard", ()),
        ("--watchdog-timeout", ("600",), "runner", ("candidate",)),
        ("--enable-metrics", (), "runner", ()),
        ("--random-seed", (str(DEFAULT_SEED),), "runner", ()),
    ]
    assert launch.removed == (Removal("--disable-radix-cache", LayerSource.CANDIDATE),)
    assert launch.argv[:3] == ("/opt/venv/bin/python", "-m", "sglang.launch_server")
    assert launch.env["SGLANG_USE_AITER"] == "0"
    assert launch.env["PYTHONPATH"] == "/farm"
    assert launch.env[GPU_MASK_ENV] == "0,1,2,3,4,5,6,7"
    assert {entry.name: entry.source for entry in launch.env_sources}["SGLANG_SIMULATE_ACC_LEN"] is LayerSource.GOLDEN


def test_every_base_group_is_kept_overridden_or_removed():
    launch = _launch()
    removed = {removal.name for removal in launch.removed}
    accounted = {arg.flag for arg in launch.args if LayerSource.BASE in (arg.source, *arg.overrode)}

    assert {"--tp", "--json-model-override-args", "--disable-radix-cache"} <= accounted | removed


def test_a_removal_scoped_to_a_value_keeps_other_values():
    kept = _launch(candidate=LaunchDelta(args_remove=("--tp 4",)))
    dropped = _launch(candidate=LaunchDelta(args_remove=("--tp=8",)))

    assert ("--tp", ("8",), "base", ()) in _sources(kept)
    assert not kept.removed
    assert "--tp" not in dropped.argv
    assert dropped.removed == (Removal("--tp", LayerSource.CANDIDATE),)


def test_a_runner_floor_only_raises():
    launch = _launch(
        candidate=LaunchDelta(args_add=("--watchdog-timeout", "1200")),
        runner=RunnerLayer(args_floor={"--watchdog-timeout": 600}),
    )

    assert ("--watchdog-timeout", ("1200",), "candidate", ()) in _sources(launch)


def test_the_runner_prepends_to_a_path_the_layers_below_set():
    launch = _launch(
        user=LaunchDelta(env_set={"PYTHONPATH": "/user"}), runner=RunnerLayer(env_prepend={"PYTHONPATH": "/farm"})
    )

    entry = {entry.name: entry for entry in launch.env_sources}["PYTHONPATH"]
    assert (entry.value, entry.source, entry.overrode) == ("/farm:/user", LayerSource.RUNNER, (LayerSource.USER,))


@pytest.mark.parametrize(
    "delta, named",
    [
        (LaunchDelta(args_add=("--speculative-num-steps", "5")), "--speculative-num-steps"),
        (
            LaunchDelta(args_add=("--speculative-config", '{"enable_adaptive_verification": true}')),
            "--speculative-config.enable_adaptive_verification",
        ),
        (LaunchDelta(args_remove=("--speculative-algorithm",)), "--speculative-algorithm"),
        (LaunchDelta(env_set={"SGLANG_SIMULATE_ACC_LEN": "4"}), "SGLANG_SIMULATE_ACC_LEN"),
    ],
)
@pytest.mark.parametrize("layer", ["user", "candidate"])
def test_a_layer_touching_a_golden_key_is_refused(delta, named, layer):
    refusal = _refusal(**{layer: delta})

    assert refusal.step is FailedStep.GOLDEN_OWNED_TOUCHED
    assert refusal.source is LayerSource(layer)
    assert named in refusal.detail


def test_without_a_golden_layer_the_draft_is_open():
    launch = _launch(base=_base(golden_al=None), candidate=LaunchDelta(args_add=("--speculative-num-steps", "5")))

    assert ("--speculative-num-steps", ("5",), "candidate", ()) in _sources(launch)
    assert launch.forced_acceptance is False


@pytest.mark.parametrize(
    "delta, named",
    [
        (LaunchDelta(args_add=("--port", "1")), "--port"),
        (LaunchDelta(args_add=("--model-path", "/other")), "--model-path"),
        (LaunchDelta(env_set={GPU_MASK_ENV: "0"}), GPU_MASK_ENV),
        (LaunchDelta(args_add=("--enable-metrics",)), "--enable-metrics"),
        (LaunchDelta(args_remove=("--enable-metrics",)), "--enable-metrics"),
    ],
)
def test_a_candidate_setting_a_runner_owned_key_is_refused(delta, named):
    refusal = _refusal(candidate=delta)

    assert (refusal.step, refusal.source) == (FailedStep.CONFIG_INVALID, LayerSource.CANDIDATE)
    assert named in refusal.detail


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
def test_runner_owned_names_every_key_the_composer_refuses(framework):
    base = (
        _base()
        if framework == "sglang"
        else _base(framework=framework, entrypoint=("vllm", "serve"), args=("--tensor-parallel-size", "8"))
    )
    owned = runner_owned(base)
    config = _config()
    server = replace(config.server, base=replace(config.server.base, digest=base.digest()))

    for flag in owned.args:
        user = LaunchDelta(args_add=(flag,))
        refusal = compose(replace(config, server=replace(server, user=user)), base, _FACTS)
        assert isinstance(refusal, Refusal) and flag in refusal.detail, flag
    assert GPU_MASK_ENV in owned.env
    if framework == "sglang":
        assert "enable-metrics" in owned.flags()


def test_a_base_naming_a_guard_flag_is_refused():
    refusal = _refusal(base=_base(args=("--model-path", "/m")))

    assert (refusal.step, refusal.source) == (FailedStep.CONFIG_INVALID, LayerSource.BASE)


def test_a_base_setting_the_metrics_flag_is_refused():
    refusal = _refusal(base=_base(args=("--tp", "8", "--enable-metrics")))

    assert (refusal.step, refusal.source) == (FailedStep.CONFIG_INVALID, LayerSource.BASE)
    assert "--enable-metrics" in refusal.detail


def _kv_launch(framework: str, kv: bool) -> Launch:
    base = _base()
    if framework != "sglang":
        base = _base(framework=framework, entrypoint=("vllm", "serve"), args=("--tensor-parallel-size", "8"))
    config = _config()
    server = replace(config.server, base=replace(config.server.base, digest=base.digest()))
    launch = compose(replace(config, server=server, telemetry=replace(config.telemetry, kv=kv)), base, _FACTS)
    assert isinstance(launch, Launch), launch
    return launch


def test_kv_telemetry_has_the_runner_turn_on_the_sglang_metrics_endpoint():
    on, off = _kv_launch("sglang", kv=True), _kv_launch("sglang", kv=False)

    assert ("--enable-metrics", (), "runner", ()) in _sources(on)
    assert "--enable-metrics" in on.argv
    assert "--enable-metrics" not in off.argv


def test_vllm_serves_its_metrics_without_a_flag():
    assert _kv_launch("vllm", kv=True).argv == _kv_launch("vllm", kv=False).argv


def _seeds(launch: Launch) -> list[tuple[str, tuple[str, ...], str]]:
    return [(arg.flag, arg.values, arg.source.value) for arg in launch.args if "seed" in arg.flag]


@pytest.mark.parametrize(
    "framework, entrypoint, flag",
    [("sglang", ("python3", "-m", "sglang.launch_server"), "--random-seed"), ("vllm", ("vllm", "serve"), "--seed")],
)
def test_the_runner_pins_the_engine_seed_no_layer_names(framework, entrypoint, flag):
    base = _base(framework=framework, entrypoint=entrypoint, args=("--tensor-parallel-size", "8"))
    launch = _launch(base)

    assert _seeds(launch) == [(flag, (str(DEFAULT_SEED),), "runner")]
    assert launch.argv[launch.argv.index(flag) + 1] == str(DEFAULT_SEED)


@pytest.mark.parametrize(
    "layer",
    [
        {"base": _base(args=("--tp", "8", "--random-seed", "11"))},
        {"user": LaunchDelta(args_add=("--random-seed", "11"))},
        {"candidate": LaunchDelta(args_add=("--random_seed=11",))},
    ],
    ids=["base", "user", "candidate"],
)
def test_a_seed_a_layer_names_is_kept_and_the_runner_adds_none(layer):
    launch = _launch(**layer)

    assert [values for _, values, _ in _seeds(launch)] == [("11",)]
    assert "runner" not in {source for _, _, source in _seeds(launch)}
    assert str(DEFAULT_SEED) not in [launch.argv[i + 1] for i, token in enumerate(launch.argv) if "seed" in token]


def test_a_base_with_another_digest_is_refused():
    refusal = compose(_config(base_digest="sha256:" + "0" * 64), _base(), _FACTS)

    assert isinstance(refusal, Refusal)
    assert refusal.step is FailedStep.BASE_DIGEST_MISMATCH


def test_an_applied_golden_layer_forces_acceptance():
    launch = _launch()
    document = launch.to_document()

    assert document["forced_acceptance"] is True
    assert document["golden"] == _base().golden_al.to_document()


def test_the_accuracy_round_leaves_golden_out_and_forces_nothing():
    base = _base()
    config = _accuracy_round()
    config = replace(config, server=replace(config.server, base=replace(config.server.base, digest=base.digest())))
    launch = compose(config, base, replace(_FACTS, client=None))

    assert isinstance(launch, Launch)
    assert launch.golden_use is GoldenUse.EXCLUDE
    assert launch.forced_acceptance is False
    assert launch.to_document()["golden"] is None
    assert "--speculative-algorithm" not in launch.argv
    assert "SGLANG_SIMULATE_ACC_LEN" not in launch.env


@pytest.mark.parametrize(
    "framework, entrypoint, server",
    [
        ("sglang", ("python3", "-m", "sglang.launch_server"), ("-m", "sglang.launch_server")),
        ("vllm", ("vllm", "serve"), ("-m", "vllm.entrypoints.cli.main", "serve")),
    ],
)
def test_the_server_runs_under_the_interpreter_the_round_probes(framework, entrypoint, server):
    base = _base(framework=framework, entrypoint=entrypoint, args=("--tensor-parallel-size", "8"))
    launch = _launch(base, framework_python="/envs/serve/bin/python")

    assert launch.argv[: 1 + len(server)] == ("/envs/serve/bin/python", *server)
    assert launch.argv[0] == launch.framework.python
    assert launch.to_document()["argv"] == list(launch.argv)
    assert launch.base == base and launch.to_document()["base_digest"] == base.digest()


def test_a_base_whose_entrypoint_is_not_its_framework_server_is_refused():
    refusal = _refusal(base=_base(entrypoint=("/other/bin/python", "-m", "sglang.launch_server")))

    assert (refusal.step, refusal.source) == (FailedStep.CONFIG_INVALID, LayerSource.BASE)
    assert "server.framework_python" in refusal.detail


def test_a_launch_document_without_an_entrypoint_ran_the_base_entrypoint():
    launch = _launch()
    recorded = replace(launch, entrypoint=launch.base.entrypoint)
    document = recorded.to_document()
    del document["entrypoint"]

    again = Launch.from_document(json.loads(json.dumps(document)))
    assert again == recorded
    assert again.argv[:3] == ("python3", "-m", "sglang.launch_server")
    assert again.digest() == document["digest"]


def test_round_trip_through_json_keeps_the_launch_and_its_digest():
    launch = _launch()
    document = json.loads(json.dumps(launch.to_document()))

    again = Launch.from_document(document)
    assert again == launch
    assert again.digest() == launch.digest()
    assert document["schema"] == LAUNCH_SCHEMA
    assert document["base_digest"] == _base().digest()
    assert document["tools"]["staging_manifest_sha256"] == _SHA


@pytest.mark.parametrize(
    "key, value",
    [("argv", ["python3"]), ("forced_acceptance", False), ("env", {}), ("digest", "sha256:" + "0" * 64)],
)
def test_a_document_whose_derived_fields_disagree_is_refused(key, value):
    document = copy.deepcopy(_launch().to_document())
    document[key] = value

    with pytest.raises(LaunchError, match=key):
        Launch.from_document(document)


def test_unknown_keys_are_refused():
    document = copy.deepcopy(_launch().to_document())
    document["dropped"] = []

    with pytest.raises(LaunchError, match="unknown key"):
        Launch.from_document(document)


def test_the_digest_keeps_argv_order_and_covers_the_client():
    launch = _launch(candidate=LaunchDelta(args_add=("--a", "--b")))
    swapped = _launch(candidate=LaunchDelta(args_add=("--b", "--a")))

    assert launch.digest() != swapped.digest()
    assert launch.digest() != replace(launch, client=replace(launch.client, seed=5)).digest()
    assert launch.digest() == replace(launch, config_sha256="f" * 64).digest()


def test_the_version_check_follows_whether_the_base_names_a_recipe_version():
    with pytest.raises(LaunchError, match="not_applicable"):
        compose(_config(), _base(), replace(_FACTS, version_check=VersionCheck.NOT_APPLICABLE))


_ASSUMED_BASE = _base(image=recipe_image_facts("lmsysorg/sglang-rocm:muse-spark-open", "v0.5.23"))


def test_a_version_not_held_because_it_is_assumed_needs_a_base_whose_version_is_assumed():
    assumed = replace(_FACTS, version_check=VersionCheck.ASSUMED_NOT_HELD)
    launch = compose(_assumed_config(), _ASSUMED_BASE, assumed)
    assert isinstance(launch, Launch)
    assert launch.framework.recipe_min_version == "0.5.23"
    assert Launch.from_document(json.loads(json.dumps(launch.to_document()))) == launch
    with pytest.raises(LaunchError, match="assumed"):
        compose(_config(), _base(), assumed)


def test_a_launch_that_held_an_assumed_version_before_it_was_exempt_still_reads():
    launch = compose(_assumed_config(), _ASSUMED_BASE, _FACTS)
    assert isinstance(launch, Launch) and launch.framework.version_check is VersionCheck.PASSED
    assert Launch.from_document(json.loads(json.dumps(launch.to_document()))) == launch


def _assumed_config():
    server = _config().server
    return _config(server=replace(server, base=replace(server.base, digest=_ASSUMED_BASE.digest())))
