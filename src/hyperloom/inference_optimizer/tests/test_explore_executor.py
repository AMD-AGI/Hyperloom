# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ExploreExecutor and explore_search ledger tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.orchestrator.actions.executors.explore import ExploreExecutor
from hyperloom.inference_optimizer.canonical_fingerprint import (
    canonical_fingerprint,
)
from hyperloom.orchestrator.actions.executors._grid_runner import GridRecipe, GridVariant
from hyperloom.orchestrator.actions.executors._grid_variant_filter import apply_compatibility_filter
from hyperloom.orchestrator.actions.executors._grid_variant_filter import ServingStack
from hyperloom.orchestrator.actions.executors._launch_evidence import RequestedLaunch, build_launch_evidence
from hyperloom.orchestrator.actions.executors._grid_base import _atom_default_grid
from hyperloom.orchestrator.actions.executors.explore import (
    _default_grid_for_framework,
    filter_baseline_noop_variants,
    observed_launch_from_state,
    restates_the_baseline_launch,
)
from hyperloom.orchestrator.actions.executors._launch_restatement import Launch, launch_change, named_settings
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.bus.resource_lock import (
    SqliteLeaseBackend,
)
from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentRunner
from hyperloom.orchestrator.state.task_registry import TaskRegistry, TaskSpec
from hyperloom.orchestrator.bus.storage.connection import SqliteConnection

from .conftest import write_benchmark_yaml
from hyperloom.orchestrator.actions.executors._workload_envs import BenchmarkTarget, LaunchLayer


@pytest.fixture(autouse=True)
def _isolate_leak_root(tmp_path_factory, monkeypatch):
    sandbox = tmp_path_factory.mktemp("isolated_leak_root_m3")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))


#: What the report mapper stamps on every AgentX round it accepts as evidence.
_AGENTX_EVIDENCE = {"valid_measurement": True, "diagnostic_only": False, "workload_identity": "agentx-workload"}


def _fake_workspace(slot: Path, *, tput: float = 800.0, perf_axes: dict[str, float] | None = None) -> Path:
    workspace = slot / "benchmark_sglang_20260519_001122"
    workspace.mkdir(parents=True)
    (workspace / "benchmark_report.json").write_text(
        json.dumps(
            {
                "success": True,
                "framework": "sglang",
                "model": "/path/models/Qwen-Qwen3-8B",
                "throughput": {
                    "request_throughput": tput / 256,
                    "output_throughput": tput,
                    "total_token_throughput": (
                        tput * 2 if perf_axes is None else perf_axes.get("total_token_throughput")
                    ),
                    "completed_requests": 80,
                    "duration_seconds": 25.0,
                },
                "latency": {
                    "ttft": {"mean_ms": 140.0, "p99_ms": 160.0},
                    "e2el": {"mean_ms": 2500.0, "p99_ms": 2560.0},
                },
            }
        )
    )
    if perf_axes is not None:
        (workspace / "inferencex_result.json").write_text(json.dumps({"output_throughput": tput, **perf_axes}))
    return workspace


@pytest.fixture
def sub_agent_runner(tmp_path):
    db = SqliteConnection(tmp_path / "db.db")
    locks = SqliteLeaseBackend(db)
    tr = TaskRegistry(db)
    runner = SubAgentRunner(locks, tr)
    yield runner, tr, tmp_path
    db.close()


def test_canonical_fingerprint_collapses_renames():
    """Two variants with identical content collapse to the same fingerprint."""
    fp_a = canonical_fingerprint("--max-num-seqs 256", {})
    fp_b = canonical_fingerprint("--max-num-seqs 256", {})
    assert fp_a == fp_b

    # Argument order doesn't matter (sorted token tuple).
    fp_c = canonical_fingerprint("--max-num-seqs 256 --block-size 128", {})
    fp_d = canonical_fingerprint("--block-size 128 --max-num-seqs 256", {})
    assert fp_c == fp_d


def test_canonical_fingerprint_distinguishes_envs():
    fp_args = canonical_fingerprint("--max-num-seqs 256", {})
    fp_args_envs = canonical_fingerprint(
        "--max-num-seqs 256",
        {"VLLM_ROCM_USE_AITER": "1"},
    )
    assert fp_args != fp_args_envs


_CANDIDATE = "candidate"
# Names a flag the observed stack already holds at this value, so the argv half
# of the gate matches and the verdict turns on env alone.
_RESTATES_ARGV = "--max-running-requests 512"
# A knob a stack genuinely varies: the recipe reads it as
# ``${HICACHE_IO_BACKEND:-direct}``, so both values below are reachable.
_TUNED_ENV = "HICACHE_IO_BACKEND"
# The recipe's own env. The stack stamps it whether or not a variant names it,
# so it rides identically on both sides and the verdict turns on the delta.
_RECIPE_ENV = {"KV_OFFLOADING": "1"}
# The argv the base config carries before any variant overlays it.
_BASE_ARGS_ENV = {"EXTRA_SGLANG_ARGS": "--mem-fraction-static 0.9"}


def _stack_env(**changes: str) -> dict[str, str]:
    """The env the stack stamped: the recipe's, plus its own tuning."""
    return {**_RECIPE_ENV, **changes}


def _stack_config(**changes: object) -> dict[str, object]:
    """What the running engine resolved, typed as SGLang reports it."""
    return {"mem_fraction_static": 0.9, "max_running_requests": 512, **changes}


def _noop_filter_kwargs(tmp_path: Path, **overrides: object) -> dict:
    base_yaml = tmp_path / "base.yaml"
    base_yaml.write_text(
        yaml.safe_dump({"benchmark": {"framework": "sglang", "envs": {**_RECIPE_ENV, **_BASE_ARGS_ENV}}}),
        encoding="utf-8",
    )
    kwargs = {"base_yaml_path": base_yaml}
    kwargs.update(overrides)
    kwargs["recipe"] = GridRecipe(
        base_yaml_path=kwargs.pop("base_yaml_path"),
        base=LaunchLayer(extra_server_args="", extra_envs={}, remove_args=[], unset_envs=[], args_mode="append"),
        target=BenchmarkTarget(model_path=None, gpu_type=None, benchmark_script=None),
    )
    kwargs["observed"] = (
        kwargs.pop("observed_server_config", _stack_config()),
        kwargs.pop("observed_server_env", _stack_env()),
    )
    return kwargs


def _variant(args: str = _RESTATES_ARGV, **fields: object) -> GridVariant:
    return GridVariant(_CANDIDATE, extra_server_args=args, **fields)


# Each case is (id, variant, filter overrides). The verdict is the whole
# expectation: does the launch this variant would produce already match the
# stack the server is running?

# Everything the variant changes already holds in the running server, so the
# measurement would re-run the stack.
_ALREADY_THE_STACK = (
    (
        "same_argv_and_env",
        _variant(extra_envs={_TUNED_ENV: "kernel"}),
        {"observed_server_env": _stack_env(HICACHE_IO_BACKEND="kernel")},
    ),
    (
        # 3600 and 3600.0 are one value; the engine reports the float.
        "a_value_the_engine_reports_as_a_float",
        _variant("--watchdog-timeout 3600"),
        {"observed_server_config": _stack_config(watchdog_timeout=3600.0)},
    ),
)

# The variant changes something the stack does not hold, or the stack is not
# known to hold it: either way the variant runs.
_MUST_RUN = (
    ("different_argv_value", _variant("--max-running-requests 128"), {}),
    (
        "different_env_value",
        _variant(extra_envs={_TUNED_ENV: "direct"}),
        {"observed_server_env": _stack_env(HICACHE_IO_BACKEND="kernel")},
    ),
    (
        # chaojhou 1: parallelism is compared, not stripped.
        "different_parallelism_value",
        _variant("--dp-size 4"),
        {"observed_server_config": _stack_config(dp_size=8)},
    ),
    (
        # chaojhou 2: a list-valued flag is never judged against the merged config.
        "multi_value_flag_is_not_comparable",
        _variant("--cuda-graph-bs 1 2 4 8"),
        {"observed_server_config": _stack_config(cuda_graph_bs=[1, 2, 4, 8, 16])},
    ),
)


@pytest.mark.parametrize(
    ("variant", "overrides"),
    [case[1:] for case in _ALREADY_THE_STACK],
    ids=[case[0] for case in _ALREADY_THE_STACK],
)
def test_variant_that_is_already_the_stack_is_dropped(tmp_path, variant, overrides):
    """The server is already running this launch, so measuring it buys nothing."""
    kept, dropped = filter_baseline_noop_variants([variant], **_noop_filter_kwargs(tmp_path, **overrides))
    assert kept == []
    assert [name for name, _ in dropped] == [_CANDIDATE]
    assert "baseline noop" in dropped[0][1]


@pytest.mark.parametrize(
    ("variant", "overrides"),
    [case[1:] for case in _MUST_RUN],
    ids=[case[0] for case in _MUST_RUN],
)
def test_variant_that_is_not_known_to_be_the_stack_is_kept(tmp_path, variant, overrides):
    """Keep on a real difference, and keep on a guess: only a proven match may drop."""
    kept, dropped = filter_baseline_noop_variants([variant], **_noop_filter_kwargs(tmp_path, **overrides))
    assert [gv.name for gv in kept] == [_CANDIDATE]
    assert dropped == []


def test_a_removal_changes_the_launch_only_when_the_stack_passes_what_it_removes(tmp_path):
    """The materialized argv shows a removal; an unset also pops the process env, which no config shows."""
    grid = [
        GridVariant("removes-a-passed-arg", remove_args=["--mem-fraction-static"]),
        GridVariant("removes-an-absent-arg", remove_args=["--enable-dp-attention"]),
        GridVariant("unsets-an-env", unset_envs=["HIP_FORCE_DEV_KERNARG"]),
    ]
    kept, dropped = filter_baseline_noop_variants(grid, **_noop_filter_kwargs(tmp_path))
    assert [gv.name for gv in kept] == ["removes-a-passed-arg", "unsets-an-env"]
    assert dropped == [("removes-an-absent-arg", "launches the server the stack already runs (baseline noop)")]


def test_a_row_claiming_a_revalidation_provenance_is_still_judged(tmp_path):
    """Only a stack revalidation round is exempt; a grid row's own provenance cannot buy it a measurement."""
    variant = _variant(_RESTATES_ARGV)
    variant.provenance = "geak_revalidate"  # type: ignore[attr-defined]
    kept, dropped = filter_baseline_noop_variants([variant], **_noop_filter_kwargs(tmp_path))
    assert kept == []
    assert dropped == [(_CANDIDATE, "launches the server the stack already runs (baseline noop)")]


def _agentx_noop_filter_kwargs(tmp_path: Path) -> dict:
    workload = tmp_path / "agentx.yaml"
    workload.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "sglang",
                    "model": "Qwen/Qwen3-0.6B",
                    "agentx": {"enabled": True, "mode": "canonical"},
                    "envs": {"TP": "1", "EP_SIZE": "1", "CONC": "8"},
                }
            }
        ),
        encoding="utf-8",
    )
    kwargs = _noop_filter_kwargs(
        tmp_path, observed_server_config={"kv_cache_dtype": "fp8", "max_running_requests": 512}
    )
    kwargs["recipe"] = GridRecipe(
        base_yaml_path=workload,
        base=LaunchLayer(extra_server_args="--kv-cache-dtype fp8", extra_envs={"SGLANG_TEST": "1"}),
        target=BenchmarkTarget(model_path=None, gpu_type=None, benchmark_script=None),
    )
    return kwargs


def test_an_agentx_variant_is_judged_on_its_launch_overrides(tmp_path):
    """AgentX carries the server argv in ``launch_overrides``, not in ``benchmark.envs``."""
    grid = [
        GridVariant("restates-the-stack-arg", "--kv-cache-dtype fp8"),
        GridVariant("restates-the-engine-default", "--max-running-requests 512"),
        GridVariant("restates-the-stack-env", extra_envs={"SGLANG_TEST": "1"}),
        GridVariant("changes-an-arg", "--chunked-prefill-size 2048"),
        GridVariant("changes-an-env", extra_envs={"SGLANG_TEST": "0"}),
        GridVariant("removes-a-recipe-arg", remove_args=["--disable-radix-cache"]),
    ]
    kept, dropped = filter_baseline_noop_variants(grid, **_agentx_noop_filter_kwargs(tmp_path))
    assert [gv.name for gv in kept] == ["changes-an-arg", "changes-an-env", "removes-a-recipe-arg"]
    assert [name for name, _ in dropped] == [
        "restates-the-stack-arg",
        "restates-the-engine-default",
        "restates-the-stack-env",
    ]


def test_the_engines_own_record_is_what_makes_the_filter_fire(tmp_path):
    """End to end on the only launch account SGLang gives: the resolved settings it logs.

    No argv is echoed anywhere, so ``observed_server_launch_flags`` stays empty
    and every earlier form of this gate filtered nothing.
    """
    config = tmp_path / "stack.yaml"
    config.write_text(
        yaml.safe_dump({"benchmark": {"framework": "sglang", "envs": {**_RECIPE_ENV, **_BASE_ARGS_ENV}}}),
        encoding="utf-8",
    )
    log = tmp_path / "server.log"
    log.write_text(
        "[2026-09-21 21:14:33] Attention backend not explicitly specified.\n"
        "[2026-09-21 21:14:34] server_args="
        f"{ {'mem_fraction_static': 0.9, 'max_running_requests': 512, 'dp_size': 8}!r}\n",
        encoding="utf-8",
    )
    evidence = build_launch_evidence(RequestedLaunch(config, "sglang"), actual_server_log=str(log), slot=tmp_path)
    assert evidence["observed_server_launch_flags"] == ""
    assert evidence["observed_server_config"]["max_running_requests"] == 512
    # The env is stamped from the recipe; the argv key holds the launch command, not an env value.
    assert evidence["requested_server_env"] == _RECIPE_ENV

    # chaojhou 4: last_baseline is an attempt entry that never carries launch_evidence,
    # so a stack with no current best is unknown rather than read from it.
    no_current_best = SimpleNamespace(current_best_measurement={}, last_baseline={"launch_evidence": evidence})
    assert observed_launch_from_state(no_current_best) == ({}, {})

    state = SimpleNamespace(current_best_measurement={"launch_evidence": evidence})
    observed_config, observed_env = observed_launch_from_state(state)
    assert observed_env == _RECIPE_ENV
    kwargs = _noop_filter_kwargs(tmp_path, observed_server_config=observed_config, observed_server_env=observed_env)

    _, dropped = filter_baseline_noop_variants([_variant(_RESTATES_ARGV)], **kwargs)
    assert [name for name, _ in dropped] == [_CANDIDATE]

    # A different value runs, and so does a setting the engine never reported:
    # absence from the record is unknown, not equal.
    for args in ("--max-running-requests 128", "--stream-interval 20"):
        kept, none_dropped = filter_baseline_noop_variants([_variant(args)], **kwargs)
        assert [gv.name for gv in kept] == [_CANDIDATE], args
        assert none_dropped == []


def test_a_setting_the_engine_rewrote_is_not_already_in_effect(tmp_path):
    """DP attention divides chunked_prefill_size, so the resolved 8192 is not what ``--chunked-prefill-size 8192`` passes.

    Taken from a real SGLang log: 65536 was requested and 8192 resolved under
    dp_size=8. Passing 8192 would launch with 1024, a real change. The engine's
    own report covers it, and so does the DP-attention rule for a log that never
    carried the report.
    """
    config = tmp_path / "stack.yaml"
    config.write_text(
        yaml.safe_dump({"benchmark": {"framework": "sglang", "envs": {**_RECIPE_ENV, **_BASE_ARGS_ENV}}}),
        encoding="utf-8",
    )
    log = tmp_path / "server.log"
    record = {"enable_dp_attention": True, "dp_size": 8, "chunked_prefill_size": 8192, "max_running_requests": 512}
    log.write_text(
        "[2026-09-18 19:11:48] DP attention is enabled. chunked prefill size is adjusted from 65536 to 8192.\n"
        f"[2026-09-18 19:11:50] server_args={record!r}\n",
        encoding="utf-8",
    )
    evidence = build_launch_evidence(RequestedLaunch(config, "sglang"), actual_server_log=str(log), slot=tmp_path)
    assert evidence["engine_adjusted_settings"] == {"chunked_prefill_size": {"requested": "65536", "resolved": "8192"}}

    for engine_report in (evidence["engine_adjusted_settings"], {}):
        stack = {**evidence, "engine_adjusted_settings": engine_report}
        observed_config, observed_env = observed_launch_from_state(
            SimpleNamespace(current_best_measurement={"launch_evidence": stack})
        )
        kwargs = _noop_filter_kwargs(tmp_path, observed_server_config=observed_config, observed_server_env=observed_env)

        kept, dropped = filter_baseline_noop_variants([_variant("--chunked-prefill-size 8192")], **kwargs)
        assert [gv.name for gv in kept] == [_CANDIDATE]
        assert dropped == []

        # Only the rewritten settings are unknown: a plain restatement still drops.
        _, dropped = filter_baseline_noop_variants([_variant(_RESTATES_ARGV)], **kwargs)
        assert [name for name, _ in dropped] == [_CANDIDATE]


# The operator baseline's own argv, as a vLLM stack carries it: both flags a
# restating variant names are already set here.
_VLLM_BASE_ARGV = "--kv-cache-dtype fp8 --enable-prefix-caching --max-num-seqs 64"


def _vllm_noop_filter_kwargs(tmp_path: Path) -> dict:
    base_yaml = tmp_path / "vllm_base.yaml"
    base_yaml.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "vllm",
                    "envs": {"VLLM_ROCM_USE_AITER": "1", "EXTRA_VLLM_ARGS": _VLLM_BASE_ARGV},
                }
            }
        ),
        encoding="utf-8",
    )
    return _noop_filter_kwargs(
        tmp_path,
        base_yaml_path=base_yaml,
        observed_server_config={"kv_cache_dtype": "fp8", "enable_prefix_caching": True, "max_num_seqs": 64},
        observed_server_env={"VLLM_ROCM_USE_AITER": "1"},
    )


def test_a_variant_that_only_restates_the_baseline_launch_is_dropped(tmp_path):
    """A variant naming flags the baseline already sets launches the baseline again, and costs a full measurement.

    Its merged launch has no delta from the baseline, which is also what a change
    the merged config cannot express looks like; the variant's own request is
    what tells the two apart.
    """
    grid = [
        GridVariant("kv-cache-fp8", extra_server_args="--kv-cache-dtype fp8"),
        GridVariant("maxseqs-64", extra_server_args="--max-num-seqs=64"),
        GridVariant("aiter-on", extra_envs={"VLLM_ROCM_USE_AITER": "1"}),
        GridVariant("maxseqs-128", extra_server_args="--max-num-seqs 128"),
        GridVariant("restates-one-changes-one", extra_server_args="--kv-cache-dtype fp8 --block-size 128"),
    ]
    kept, dropped = filter_baseline_noop_variants(grid, **_vllm_noop_filter_kwargs(tmp_path))

    assert [name for name, _ in dropped] == ["kv-cache-fp8", "maxseqs-64", "aiter-on"]
    assert all("baseline noop" in reason for _, reason in dropped)
    assert [gv.name for gv in kept] == ["maxseqs-128", "restates-one-changes-one"]


def test_a_replacing_variant_is_not_a_restatement(tmp_path):
    """``args_mode=replace`` drops every baseline arg it does not name, which the merged-delta probe cannot show."""
    variant = GridVariant("replace-with-a-subset", extra_server_args="--kv-cache-dtype fp8", args_mode="replace")
    kept, dropped = filter_baseline_noop_variants([variant], **_vllm_noop_filter_kwargs(tmp_path))
    assert [gv.name for gv in kept] == ["replace-with-a-subset"]
    assert dropped == []


def test_restated_args_with_a_kernel_overlay_are_not_a_restatement(tmp_path):
    """The overlay changes the code launched; only the merged config shows it, so the variant runs."""
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    variant = GridVariant("kv-cache-fp8-with-kernel", extra_server_args="--kv-cache-dtype fp8")
    variant.overlay_pythonpath = str(overlay)  # type: ignore[attr-defined]
    kept, dropped = filter_baseline_noop_variants([variant], **_vllm_noop_filter_kwargs(tmp_path))
    assert [gv.name for gv in kept] == ["kv-cache-fp8-with-kernel"]
    assert dropped == []


def _vllm_baseline_state(tmp_path: Path, base_argv: str = _VLLM_BASE_ARGV, **envs: object) -> SimpleNamespace:
    config = tmp_path / "baseline.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "vllm",
                    "envs": {"VLLM_ROCM_USE_AITER": "1", **envs, "EXTRA_VLLM_ARGS": base_argv},
                }
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(baseline_config_path=str(config), current_best_measurement={})


def test_a_null_baseline_env_is_unset_not_the_string_none(tmp_path):
    """The launch leaves a null env unset, so a request for the string ``None`` sets something new."""
    state = _vllm_baseline_state(tmp_path, FOO=None)
    assert restates_the_baseline_launch(state, "", {"FOO": "None"}) is False
    assert restates_the_baseline_launch(state, "--kv-cache-dtype fp8", {}) is True
    missing = SimpleNamespace(baseline_config_path=str(tmp_path / "missing.yaml"), current_best_measurement={})
    assert restates_the_baseline_launch(missing, "--kv-cache-dtype fp8", {}) is False


@pytest.mark.parametrize(
    ("argv", "envs", "base_argv", "expected"),
    [
        pytest.param("--kv-cache-dtype fp8", {}, _VLLM_BASE_ARGV, True, id="restated_flag"),
        pytest.param("--enable-prefix-caching", {}, _VLLM_BASE_ARGV, True, id="restated_bare_flag"),
        pytest.param("", {"VLLM_ROCM_USE_AITER": "1"}, _VLLM_BASE_ARGV, True, id="restated_env"),
        pytest.param("", {}, _VLLM_BASE_ARGV, True, id="empty_request"),
        pytest.param("--kv-cache-dtype auto", {}, _VLLM_BASE_ARGV, False, id="different_value"),
        pytest.param("--block-size 128", {}, _VLLM_BASE_ARGV, False, id="setting_the_baseline_lacks"),
        pytest.param("", {"VLLM_ROCM_USE_AITER": "0"}, _VLLM_BASE_ARGV, False, id="different_env"),
        pytest.param("", {"NEW_ENV": "1"}, _VLLM_BASE_ARGV, False, id="env_the_baseline_lacks"),
        pytest.param("fp8 --kv-cache-dtype fp8", {}, _VLLM_BASE_ARGV, False, id="positional_token"),
        pytest.param("--kv-cache-dtype 'fp8", {}, _VLLM_BASE_ARGV, False, id="unparsable"),
        pytest.param("--no-enable-prefix-caching", {}, _VLLM_BASE_ARGV, False, id="negates_a_baseline_flag"),
        pytest.param(
            "--enable-prefix-caching",
            {},
            "--enable-prefix-caching --no-enable-prefix-caching",
            True,
            id="dedupe_keeps_the_negation_last",
        ),
        pytest.param("--cuda-graph-bs 1 2 4", {}, "--cuda-graph-bs 1 2 4", True, id="list_restated_in_full"),
        pytest.param("--cuda-graph-bs 1 2", {}, "--cuda-graph-bs 1 2 4", False, id="list_restated_in_part"),
    ],
)
def test_restates_the_baseline_launch(tmp_path, argv, envs, base_argv, expected):
    """Only a request that leaves the materialized launch as it was restates it; anything unreadable does not."""
    assert restates_the_baseline_launch(_vllm_baseline_state(tmp_path, base_argv), argv, envs) is expected


def test_the_baseline_observation_counts_only_for_the_config_it_was_made_on(tmp_path):
    """``--max-num-seqs 256`` is the observed engine default, but only on the launch that was observed."""
    state = _vllm_baseline_state(tmp_path)
    observed = {"launch_evidence": {"observed_server_config": {"max_num_seqs": 64, "block_size": 16}}}
    state.current_best_measurement = observed
    assert restates_the_baseline_launch(state, "--block-size 16", {}) is False
    observed["launch_evidence"]["materialized_config_path"] = state.baseline_config_path
    assert restates_the_baseline_launch(state, "--block-size 16", {}) is True


def test_named_settings_resolve_like_the_engine():
    """The last spelling of a boolean flag wins, negated or not; an argv it cannot account for raises."""
    assert named_settings("--enable-prefix-caching --no-enable-prefix-caching", "vllm") == {
        "enable_prefix_caching": ["false"]
    }
    assert named_settings("--no-enable-prefix-caching --enable-prefix-caching", "vllm") == {"enable_prefix_caching": []}
    for argv in ("fp8 --kv-cache-dtype fp8", "--kv-cache-dtype 'fp8"):
        with pytest.raises(ValueError):
            named_settings(argv, "vllm")


def test_launch_change_names_every_change_and_discounts_what_the_server_resolved():
    base = Launch({"enable_x": [], "kv_cache_dtype": ["fp8"]}, {"A": "1"}, {"unset_envs": []})
    launch = Launch({"enable_x": ["false"], "block_size": ["128"]}, {"B": "2"}, {"unset_envs": ["C"]})
    assert launch_change(base, launch, ({}, {})) == {
        "settings": {"enable_x": ["false"], "block_size": ["128"]},
        "removed": ["kv_cache_dtype"],
        "envs": {"B": "2"},
        "unset_envs": ["A"],
        "controls": ["unset_envs"],
    }
    same_server = Launch({**base.settings, "block_size": ["128"], "enable_y": ["false"]}, {"A": "1", "B": "2"}, {})
    observed = ({"block_size": 128, "no_enable_y": True}, {"B": "2"})
    assert launch_change(Launch(base.settings, base.envs, {}), same_server, observed) == {}


def test_record_explore_accepted_dedup_by_fingerprint():
    state = SharedState()
    variant = {
        "name": "vllm_kv_fp8",
        "fingerprint": "cc" * 8,
        "extra_server_args": "--kv-cache-dtype fp8",
        "extra_envs": {},
        "output_throughput": 1500.0,
        "gain_pct": 4.0,
        "provenance": "llm_direct",
    }
    state.record_explore_accepted(variant)
    state.record_explore_accepted(variant)
    assert len(state.explore_search["accepted"]) == 1


def test_apply_explore_search_update_preserves_accepted():
    state = SharedState()
    state.record_explore_accepted(
        {
            "name": "a",
            "extra_server_args": "--flag-a",
            "fingerprint": "aa" * 8,
            "gain_pct": 3.0,
        }
    )
    # Update arrives with tested/rejected but NOT accepted; the bucket survives the merge.
    state.apply_explore_search_update(
        {
            "tested": {
                "bb" * 8: {
                    "name": "b",
                    "extra_server_args": "--flag-b",
                    "extra_envs": {},
                    "outcome": "REVERT",
                }
            },
            "rejected": [
                {
                    "fingerprint": "bb" * 8,
                    "name": "b",
                    "reason": "not_keep",
                }
            ],
            "cursor": 1,
            "last_round": {"round_id": "explore-001"},
        }
    )
    assert len(state.explore_search["accepted"]) == 1
    assert state.explore_search["accepted"][0]["name"] == "a"
    assert "bb" * 8 in state.explore_search["tested"]


def _round_update(round_no: int, fingerprint: str, name: str) -> dict:
    """One round's worth of executor ledger writes."""
    return {
        "tested": {fingerprint: {"name": name, "outcome": "REVERT"}},
        "rejected": [{"fingerprint": fingerprint, "name": name, "reason": "not_keep"}],
        "last_round": {"round_id": f"explore-{round_no:03d}"},
    }


def test_apply_explore_search_update_accumulates_across_rounds():
    """The executor reports one round; the ledger has to remember the ones before it."""
    state = SharedState()
    state.apply_explore_search_update(_round_update(1, "aa" * 8, "a"))
    state.apply_explore_search_update(_round_update(2, "bb" * 8, "b"))

    search = state.explore_search
    assert set(search["tested"]) == {"aa" * 8, "bb" * 8}
    assert {r["fingerprint"] for r in search["rejected"]} == {"aa" * 8, "bb" * 8}


def test_apply_explore_search_update_remeasured_fingerprint_replaces_its_row():
    state = SharedState()
    state.apply_explore_search_update(_round_update(1, "aa" * 8, "a"))
    second = _round_update(2, "aa" * 8, "a")
    second["tested"]["aa" * 8]["outcome"] = "KEEP"
    state.apply_explore_search_update(second)

    assert state.explore_search["tested"]["aa" * 8]["outcome"] == "KEEP"
    assert len(state.explore_search["rejected"]) == 1


def test_apply_explore_search_update_advances_the_cursor_per_round():
    """The cursor is the round ordinal: round ids and idempotency keys derive from it."""
    state = SharedState()
    state.apply_explore_search_update(_round_update(1, "aa" * 8, "a"))
    assert state.explore_search["cursor"] == 1
    # A round that benched three variants still advances the ordinal by one.
    third = _round_update(2, "bb" * 8, "b")
    third["tested"].update({"cc" * 8: {"name": "c"}, "dd" * 8: {"name": "d"}})
    state.apply_explore_search_update(third)
    assert state.explore_search["cursor"] == 2


def test_warm_history_rejected_rows_survive_the_first_round():
    """Warm-start pre-fills ``rejected`` so the dedup gate denies re-tests; a round must not wipe it."""
    state = SharedState()
    state.explore_search = {
        "rejected": [{"fingerprint": "ee" * 8, "name": "warm", "reason": "warm_recipe_what_failed"}]
    }
    state.apply_explore_search_update(_round_update(1, "aa" * 8, "a"))
    assert {r["fingerprint"] for r in state.explore_search["rejected"]} == {"ee" * 8, "aa" * 8}


def test_observed_launch_after_drops_only_what_the_stack_changed():
    from hyperloom.orchestrator.actions.executors._launch_restatement import observed_launch_after

    opening = {
        "extra_server_args": "--a 1",
        "extra_envs": {"E1": "x"},
        "remove_args": [],
        "unset_envs": [],
        "args_mode": "append",
    }
    observed = ({"a": 1, "kv_cache_dtype": "fp8", "max_num_seqs": 64}, {"E1": "x", "E2": "y", "E3": "z"})

    def after(**moved):
        return observed_launch_after(*observed, opening, {**opening, **moved}, "sglang")

    assert after() == observed
    # A KEEP that adds an unrelated setting leaves the rest observed; one that sets or drops a named one does not.
    assert after(extra_server_args="--a 1 --max-num-seqs 256")[0] == {"a": 1, "kv_cache_dtype": "fp8"}
    assert after(extra_server_args="--a 1 --kv-cache-dtype auto")[0] == {"a": 1, "max_num_seqs": 64}
    assert after(extra_server_args="")[0] == {"kv_cache_dtype": "fp8", "max_num_seqs": 64}
    assert after(remove_args=["--kv-cache-dtype"])[0] == {"a": 1, "max_num_seqs": 64}
    # A removal the opening stack carried and a KEEP cleared exposes the setting again, so its observed value is stale too.
    cleared = {**opening, "remove_args": ["--kv-cache-dtype"], "unset_envs": ["E3"]}
    assert observed_launch_after(*observed, cleared, opening, "sglang") == (
        {"a": 1, "max_num_seqs": 64},
        {"E1": "x", "E2": "y"},
    )
    # A boolean flag and its negated spelling are one setting: switching either way leaves the observed value stale.
    flag_on = ({"a": 1, "enable_x": True}, {})
    assert observed_launch_after(*flag_on, opening, {**opening, "extra_server_args": "--a 1 --no-enable-x"}, "sglang")[
        0
    ] == {"a": 1}
    assert after(extra_envs={"E1": "x", "E2": "other"})[1] == {"E1": "x", "E3": "z"}
    assert after(extra_envs={})[1] == {"E2": "y", "E3": "z"}
    assert after(unset_envs=["E3"])[1] == {"E1": "x", "E2": "y"}
    # The mode is not a setting: the caller compares args as launched, so a switch to replace moves nothing by itself.
    assert after(args_mode="replace") == observed


@pytest.mark.asyncio
async def test_explore_executor_empty_grid_returns_failed(sub_agent_runner, tmp_path):
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    write_benchmark_yaml(base)
    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "base_tput": 800.0,
                "grid": [],
            },
            idempotency_key="ex-empty",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    res = await sub.run_task(task)
    assert res.result["status"] == "failed"
    assert res.result["error_class"] == "empty_grid"


def _names(variants):
    return [v.name for v in variants]


def test_atom_default_grid_mla_moe_model_emits_all_gated_variants():
    """MoE + MLA + MTP-capable model class (``moe_mla``) unlocks the full atom seed grid (>= 5 variants)."""
    grid = _atom_default_grid(
        model_class="moe_mla",
        conc=64,
    )
    names = _names(grid)
    assert len(grid) >= 5, f"too few variants: {names}"
    assert "atom_level_2" in names
    assert "atom_level_3" not in names
    assert "atom_prefix_cache" in names
    assert "atom_ep" in names, "MoE branch missing for moe_mla"
    assert "atom_dp_attn" in names, "MLA branch missing for moe_mla"
    assert "atom_mtp_3" in names, "MTP branch missing for moe_mla"
    assert "atom_mtp_1" in names
    assert "atom_cudagraph_bracket" in names


def test_atom_default_grid_dense_model_omits_moe_mla_mtp():
    """Dense model class must NOT emit MoE / MLA / MTP variants (would fail flag-compat or crash atom)."""
    grid = _atom_default_grid(
        model_class="dense",
        conc=8,
    )
    names = _names(grid)
    assert "atom_ep" not in names
    assert "atom_dp_attn" not in names
    assert "atom_mtp_3" not in names
    assert "atom_mtp_1" not in names
    assert "atom_level_2" in names
    assert "atom_level_3" not in names
    assert "atom_prefix_cache" in names


def test_atom_default_grid_fp8_model_emits_kv_fp8():
    grid = _atom_default_grid(
        model_class="moe_fp8",
        conc=16,
    )
    names = _names(grid)
    assert "atom_kv_fp8" in names
    assert "atom_ep" in names


def test_atom_default_grid_non_fp8_omits_kv_fp8():
    grid = _atom_default_grid(model_class="dense", conc=8)
    names = _names(grid)
    assert "atom_kv_fp8" not in names


def test_atom_default_grid_variants_have_unique_names():
    """Names must be unique within each model class's grid (the ledger keys by name per round)."""
    for mc in ("dense", "moe", "moe_mla", "moe_mla_nsa", "moe_fp8", ""):
        grid = _atom_default_grid(model_class=mc, conc=32)
        names = _names(grid)
        assert len(names) == len(set(names)), f"duplicate names in atom grid for model_class={mc!r}: {names}"


def test_atom_default_grid_names_use_atom_prefix():
    grid = _atom_default_grid(model_class="moe_mla", conc=64)
    for v in grid:
        assert v.name.startswith("atom_"), f"variant name does not start with 'atom_': {v.name!r}"


def test_atom_default_grid_variants_carry_default_grid_provenance():
    grid = _atom_default_grid(model_class="moe_mla", conc=64)
    for v in grid:
        assert getattr(v, "provenance", None) == "default_grid", (
            f"variant {v.name!r} provenance not set to default_grid"
        )


def test_atom_default_grid_conc_zero_omits_cudagraph_bracket():
    """When CONC is unavailable, the bracket variant is skipped."""
    grid = _atom_default_grid(model_class="dense", conc=0)
    assert "atom_cudagraph_bracket" not in _names(grid)


def test_default_grid_for_framework_atom_returns_seeded_grid():
    grid = _default_grid_for_framework(
        "atom",
        model_class="moe_mla",
        conc=64,
    )
    assert grid, "atom framework should produce a non-empty default grid"
    assert any(v.name == "atom_level_2" for v in grid)


@pytest.mark.parametrize("framework", ["sglang", "vllm", "", "unknown"])
def test_default_grid_for_framework_non_atom_returns_empty(framework):
    """Sglang/vllm rely on LLM-emitted variants; unknown frameworks also return ``[]`` (no crash)."""
    grid = _default_grid_for_framework(
        framework,
        model_class="moe_mla",
        conc=64,
    )
    assert grid == []


def _write_atom_baseline_yaml(path: Path) -> None:
    """Atom-flavoured base YAML for gap-G1 cold-start wiring tests."""
    cfg = {
        "benchmark": {
            "framework": "atom",
            "model": "/path/models/Qwen-Qwen3-32B",
            "precision": "fp8",
            "run_mode": "local",
            "envs": {"TP": 4, "CONC": 64, "ISL": 1024, "OSL": 1024},
            "benchmark_script": "atom_mi355x.sh",
            "timeout_seconds": 600,
            "profiler": {
                "torch_profiler": {"enabled": False},
                "system_profiler": {"enabled": False},
                "tracelens": {"enabled": False},
            },
            "gpu_selection": {"auto": False},
        },
    }
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)


@pytest.mark.asyncio
async def test_explore_executor_atom_empty_grid_seeds_default_grid(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
):
    """An empty grid on an atom session falls through to ``_default_grid_for_framework('atom', ...)`` rather than failing."""
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base_atom.yaml"
    _write_atom_baseline_yaml(base)

    # Sandbox MODEL_PATH so compatibility_filter doesn't auto-drop MoE/MLA variants.
    monkeypatch.setenv("MODEL_PATH", "/path/models/Qwen-Qwen3-32B")
    monkeypatch.setenv("FRAMEWORK", "atom")

    received_grid: list[list[str]] = []

    from hyperloom.orchestrator.actions.executors import (
        explore as explore_mod,
    )

    async def _capture_run_grid(**kwargs):
        received_grid.append([v.name for v in (kwargs.get("grid") or [])])
        return []

    monkeypatch.setattr(explore_mod, "run_grid", _capture_run_grid)

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "base_tput": 800.0,
                "grid": [],
                "model_class": "moe_mla",
            },
            idempotency_key="ex-atom-empty-seeded",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))

    await sub.run_task(task)

    assert received_grid, "run_grid was not invoked by the seed path"
    flat_names = [n for sub in received_grid for n in sub]
    assert all(n.startswith("atom_") for n in flat_names), (
        f"non-atom variants reached run_grid via the seed: {flat_names!r}"
    )
    assert "atom_level_2" in flat_names
    assert "atom_ep" in flat_names
    assert "atom_dp_attn" in flat_names


@pytest.mark.asyncio
async def test_explore_executor_sglang_empty_grid_still_fails_with_empty_grid(
    sub_agent_runner,
    tmp_path,
):
    """Inverse of the atom test: an empty grid on sglang/vllm still returns ``error_class='empty_grid'``."""
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base_sglang.yaml"
    write_benchmark_yaml(base)
    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "base_tput": 800.0,
                "grid": [],
            },
            idempotency_key="ex-sglang-empty-still-fails",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    res = await sub.run_task(task)
    assert res.result["status"] == "failed"
    assert res.result["error_class"] == "empty_grid"


def _lever_state() -> SharedState:
    """A session whose specialist has authored two dormant, dependency-chained rewrite levers."""
    from hyperloom.orchestrator.actions.executors import _framework_switch_manifest as manifest

    switches, problems = manifest.parse_manifest(
        [
            {"switch": "HL_HOIST", "category": "hoist_loop_invariant", "enables": ["HL_CACHE"]},
            {"switch": "HL_CACHE", "category": "memoize_invariant"},
        ]
    )
    assert problems == []
    state = SharedState(framework="sglang")
    assert state.record_authored_framework_levers(switches, default_on=False)
    return state


def test_a_leave_one_out_arm_that_regressed_attributes_its_lever():
    """Dropping a kept switch and getting slower is a REVERT, and it is the reading that says the switch pays."""
    from hyperloom.orchestrator.actions.executors.explore import _framework_lever_attributions, _outcome_row

    row = _outcome_row(
        {"name": "fwlever_drop_hl_a", "outcome": "REVERT", "gain_pct": None, "graded_gain_pct": -4.0}, "fp", "", "t"
    )
    seed = {"name": "fwlever_drop_hl_a", "framework_lever": "HL_A", "framework_lever_source": "leave_one_out"}

    attributions = _framework_lever_attributions([row], [seed])

    assert [(a["switch"], a["gain_pct"], a["outcome"]) for a in attributions] == [("HL_A", 4.0, "REVERT")]


def test_a_lever_attribution_arm_runs_after_the_candidates_that_could_keep():
    """Additive lever arms may KEEP and run first; leave-one-out arms only attribute and run last."""
    from hyperloom.orchestrator.actions.executors.explore import _lever_seeded_payload

    levers = [
        {"name": "fwlever_drop_hl_a", "framework_lever": "HL_A", "framework_lever_source": "leave_one_out"},
        {"name": "fwlever_hl_b", "framework_lever": "HL_B", "framework_lever_source": "additive"},
    ]

    payload = _lever_seeded_payload([{"name": "llm_proposal"}], levers)

    assert [v["name"] for v in payload] == ["fwlever_hl_b", "llm_proposal", "fwlever_drop_hl_a"]


def test_unified_attn_filter_matches_aiter_dist_version_short_sha(tmp_path):
    """The aiter fingerprint degrades to a git-describe dist version when ``AITER_COMMIT`` is unset; the same source
    tree must still be recognised, otherwise the filter fails open and the unsafe lever reaches the benchmark.
    """
    model = tmp_path / "Qwen3-14B-FP8"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "model_type": "qwen3",
                "torch_dtype": "bfloat16",
                "head_dim": 128,
                "num_attention_heads": 40,
                "num_key_value_heads": 8,
                "quantization_config": {"quant_method": "fp8"},
            }
        ),
        encoding="utf-8",
    )
    kept, dropped = apply_compatibility_filter(
        [GridVariant("unified", extra_envs={"SGLANG_USE_AITER_UNIFIED_ATTN": "1"})],
        serving=ServingStack(
            "sglang",
            str(model),
            gpu_type="mi355x",
            stack_fingerprint={
                "sglang": "0.5.20.dev20260920+gc610c40399",
                # The dist version install_baremetal.sh falls back to; embeds the short sha of the pinned commit.
                "aiter": "0.1.21.dev48+g4ad998328.d20260920",
                "rocm": "10.0.0",
            },
        ),
        base_server_args="--page-size 1 --kv-cache-dtype auto",
    )
    assert ([variant.name for variant in kept], dropped) == (
        [],
        [
            {
                "name": "unified",
                "source": "compatibility_filter",
                "reason": "SGLANG_USE_AITER_UNIFIED_ATTN=1 is unsafe on the exact ROCm 10 Qwen3-14B-FP8 stack",
            }
        ],
    )


def test_atom_default_grid_survives_compatibility_filter():
    """``apply_compatibility_filter`` drops no atom seed variant."""
    grid = _atom_default_grid(model_class="moe_mla", conc=64)
    kept, dropped = apply_compatibility_filter(grid, serving=ServingStack("atom", ""))
    assert kept == grid, f"compatibility filter dropped seed variants; dropped={dropped}"


def test_grid_variants_from_payload_coerces_list_extra_args():
    """A JSON-list ``extra_args`` must be space-joined into a shell-arg string, not stringified into a Python repr."""
    from hyperloom.orchestrator.actions.executors.explore import (
        _grid_variants_from_payload,
    )

    payload = [
        {"name": "list_args", "extra_args": ["--max-num-batched-tokens", "32768"]},
        {"name": "str_args", "extra_args": "--block-size 64"},
        {"name": "tuple_server_args", "extra_server_args": ("--distributed-executor-backend", "mp")},
    ]
    by_name = {v.name: v for v in _grid_variants_from_payload(payload)}

    assert by_name["list_args"].extra_server_args == "--max-num-batched-tokens 32768"
    assert "[" not in by_name["list_args"].extra_server_args
    assert by_name["str_args"].extra_server_args == "--block-size 64"
    assert by_name["tuple_server_args"].extra_server_args == "--distributed-executor-backend mp"


def test_grid_variants_from_payload_carries_removal_controls():
    from hyperloom.orchestrator.actions.executors.explore import (
        _grid_variants_from_payload,
    )

    payload = [
        {
            "name": "without_cache",
            "remove_args": "--enable-prefix-caching",
            "unset_envs": ["SGLANG_ENABLE_FOO"],
            "args_mode": "replace",
            "extra_args": "--max-num-seqs 256",
        }
    ]

    variant = _grid_variants_from_payload(payload)[0]
    assert variant.remove_args == ["--enable-prefix-caching"]
    assert variant.unset_envs == ["SGLANG_ENABLE_FOO"]
    assert variant.args_mode == "replace"
    assert variant.extra_server_args == "--max-num-seqs 256"


def test_grid_variants_preserve_authored_reason_as_experience_reasoning():
    from hyperloom.orchestrator.actions.executors.explore import (
        _grid_variants_from_payload,
    )

    variant = _grid_variants_from_payload(
        [
            {
                "name": "chunked",
                "extra_args": "--max-num-batched-tokens 8192",
                "reason": "A larger chunk should reduce scheduler dispatch overhead.",
                "provenance": "specialist:scheduler",
            }
        ]
    )[0]

    assert variant.note == "A larger chunk should reduce scheduler dispatch overhead."
    assert variant.reasoning_origin == "action_payload.reason"


def test_grid_variants_do_not_treat_provenance_as_reasoning():
    from hyperloom.orchestrator.actions.executors.explore import (
        _grid_variants_from_payload,
    )

    variant = _grid_variants_from_payload(
        [{"name": "unexplained", "extra_args": "--foo", "provenance": "specialist:scheduler"}]
    )[0]

    assert variant.note == ""
    assert variant.reasoning_origin == ""


def test_on_disk_stderr_tail_reads_benchmark_stderr_log(tmp_path):
    from hyperloom.orchestrator.actions.executors._grid_runner import (
        _on_disk_stderr_tail,
        _report_errors_summary,
    )

    (tmp_path / "benchmark_stderr.log").write_text("bench_fps.py: error: unrecognized arguments: --use_cache teacache")
    tail = _on_disk_stderr_tail(tmp_path)
    assert "unrecognized arguments" in tail
    # Empty dir → empty string (caller keeps its original blank error).
    assert _on_disk_stderr_tail(tmp_path / "nope") == ""
    assert _report_errors_summary(None) == ""
    assert _report_errors_summary({"errors": []}) == ""
    assert (
        _report_errors_summary({"errors": ["scriptable benchmark script not found for custom_mi355x.sh"]})
        == "scriptable benchmark script not found for custom_mi355x.sh"
    )


def test_measured_verdict_holds_only_for_the_same_workload_and_stack():
    from hyperloom.orchestrator.actions.executors.explore_prefilter.measured_variants import (
        MeasuredOn,
        grid_fingerprints,
        measured_verdict,
        stack_of,
    )

    params = {"base_tput": 800.0, "base_extra_args": "--a 1", "grid": [{"name": "v", "extra_args": "--b 2"}]}
    (fp,) = grid_fingerprints(params)
    stack = stack_of(params)
    variant = params["grid"][0]
    row = {
        "outcome": "REVERT",
        "workload_signature": "w1",
        "extra_server_args": "--b 2",
        "extra_envs": {},
        "measured_against": {"throughput": 800.0, "extra_server_args": "--a 1", "extra_envs": {}},
    }
    ledger = {"tested": {fp: row}}

    def verdict(ledger=ledger, variant=variant, stack=stack, signature="w1"):
        return measured_verdict(ledger, fp, variant=variant, now=MeasuredOn(stack, signature, 1.0))

    assert verdict() is not None
    assert verdict(signature="w2") is None
    assert verdict(stack=stack_of({**params, "base_extra_envs": {"E": "1"}})) is None
    assert verdict(ledger={"tested": {fp: {**row, "measured_against": None}}}) is None
    # The key was fingerprinted against its round's opening controls; a KEEP in that round can move them, so a row
    # under the same key whose own change differs (here: it removed nothing) does not speak for this variant.
    assert verdict(variant={**variant, "remove_args": ["--c"]}) is None
    # The bar decays per macro-cycle: a throughput-gate REVERT stands only while its gain is below the current bar.
    missed = {**row, "reason": "gain_below_threshold", "graded_gain_pct": 0.7}
    assert verdict(ledger={"tested": {fp: missed}}) is not None
    assert verdict(ledger={"tested": {fp: {**missed, "graded_gain_pct": 1.0}}}) is None
    # Accuracy and tail-guard REVERTs do not depend on the bar, whatever gain they measured.
    for reason in ("accuracy_drop", "accuracy_unavailable", "p90_regressed"):
        assert verdict(ledger={"tested": {fp: {**missed, "reason": reason, "graded_gain_pct": 5.0}}}) is not None
    # An overlay changes the launch without changing args or envs, so it has no comparable identity.
    assert grid_fingerprints(
        {**params, "grid": [{"name": "v", "extra_args": "--b 2", "overlay_pythonpath": "/o"}]}
    ) == [""]


@pytest.mark.parametrize("text", ["other: {}\n", "benchmark:\n  envs: x\n", "benchmark: [\n"])
def test_explore_refuses_a_round_whose_materialized_config_is_not_a_point(tmp_path, text):
    from hyperloom.orchestrator.actions.executors import explore

    cfg = tmp_path / "explore_base.with_envs.yaml"
    cfg.write_text(text, encoding="utf-8")

    with pytest.raises(explore._RoundRefused) as refused:
        explore._round_point(cfg)
    assert refused.value.result["error_class"] == "unreadable_config"
    assert str(cfg) in refused.value.result["error"]
