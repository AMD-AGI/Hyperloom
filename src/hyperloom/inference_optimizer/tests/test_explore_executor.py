# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ExploreExecutor and explore_search ledger tests."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from hyperloom.orchestrator.actions.executors.explore import ExploreExecutor
from hyperloom.inference_optimizer.canonical_fingerprint import (
    canonical_fingerprint,
)
from hyperloom.orchestrator.actions.executors._grid_runner import GridRecipe, GridVariant
from hyperloom.orchestrator.actions.executors._grid_variant_filter import apply_compatibility_filter
from hyperloom.orchestrator.actions.executors._subprocess_kill import (
    ORCHESTRATOR_CANCELLED_RETURNCODE,
    SESSION_TIME_EXHAUSTED_RETURNCODE,
)
from hyperloom.orchestrator.actions.stop_attribution import (
    ORCHESTRATOR_CANCELLED_CLASS,
    SESSION_TIME_EXHAUSTED_CLASS,
)
from hyperloom.orchestrator.actions.executors._grid_variant_filter import ServingStack
from hyperloom.orchestrator.actions.executors._inferencex_patcher import BENCH_SEED_ENV
from hyperloom.orchestrator.actions.executors._launch_evidence import RequestedLaunch, build_launch_evidence
from hyperloom.orchestrator.actions.executors.explore import (
    _atom_default_grid,
    _default_grid_for_framework,
    filter_baseline_noop_variants,
    observed_launch_from_state,
    restates_the_baseline_launch,
)
from hyperloom.orchestrator.actions.executors._launch_restatement import Launch, launch_change, named_settings
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.common.env import is_truthy
from hyperloom.common.perf_metric import perf_snapshot_from_mapping
from hyperloom.inference_optimizer.session.session_binding import session_scope
from hyperloom.orchestrator.bus.resource_lock import (
    SqliteLeaseBackend,
)
from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentRunner
from hyperloom.orchestrator.state.task_registry import TaskRegistry, TaskSpec
from hyperloom.orchestrator.bus.storage import SqliteConnection

from .conftest import make_coordinator
from hyperloom.orchestrator.actions.executors._workload_envs import BenchmarkTarget, LaunchLayer


def _eval_off(value: object) -> bool:
    return not is_truthy(value, default=True)


@pytest.fixture(autouse=True)
def _isolate_leak_root(tmp_path_factory, monkeypatch):
    sandbox = tmp_path_factory.mktemp("isolated_leak_root_m3")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))


def _force_cold_decision(monkeypatch) -> None:
    """Make server_lifecycle reuse ineligible, one of the two warm-decision preconditions."""
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.explore.resolve_lifecycle_params",
        lambda _config_path: {
            "eligible": False,
            "framework": "sglang",
            "port": 30000,
            "reason": "test: server_lifecycle reuse disabled",
        },
    )


def _write_baseline_yaml(path: Path) -> None:
    cfg = {
        "benchmark": {
            "framework": "sglang",
            "model": "/path/models/Qwen-Qwen3-8B",
            "precision": "bf16",
            "run_mode": "local",
            "envs": {"TP": 1, "CONC": 8, "ISL": 256, "OSL": 256},
            "benchmark_script": "sglang_mi300x.sh",
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


@pytest.mark.asyncio
async def test_explore_executor_keeps_and_reverts_per_variant(sub_agent_runner, tmp_path):
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-out"

    call_counter = {"i": 0}

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        slug = slot.parent.name + "/" + slot.name
        if "v00_v_keep" in slug:
            tput = 840.0  # +5% vs base 800
        elif "v01_v_revert" in slug:
            tput = 800.4  # +0.05% — below 1.0% threshold
        else:
            tput = 800.0
        _fake_workspace(slot, tput=tput)
        call_counter["i"] += 1
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    grid = [
        {
            "name": "v_keep",
            "extra_args": "--keep-flag",
            "extra_envs": {},
            "provenance": "default_grid",
        },
        {
            "name": "v_revert",
            "extra_args": "--revert-flag",
            "extra_envs": {},
            "provenance": "llm_direct",
        },
    ]
    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(output_dir),
                "base_tput": 800.0,
                "grid": grid,
            },
            idempotency_key="ex-1",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    assert res.state == "succeeded"
    out = res.result
    assert out["status"] == "succeeded"
    assert {w["name"] for w in out["winners"]} == {"v_keep"}
    assert {lr["name"] for lr in out["losers"]} == {"v_revert"}
    ledger = out["explore_search_update"]
    assert set(ledger["tested"].keys()) == {
        canonical_fingerprint("--keep-flag", {}),
        canonical_fingerprint("--revert-flag", {}),
    }
    outcomes = {v["name"]: v["outcome"] for v in ledger["tested"].values()}
    assert outcomes["v_keep"] == "KEEP"
    assert outcomes["v_revert"] == "REVERT"
    assert out["best_variant"]["name"] == "v_keep"
    assert out["best_gain_pct"] >= 4.0
    rejected_provenance = {r["provenance"] for r in ledger["rejected"]}
    assert rejected_provenance == {"llm_direct"}


async def _explore_on_an_observed_fp8_stack(
    sub_agent_runner, tmp_path, monkeypatch, grid, tputs, inherited="", **params
):
    """Run one explore round whose stack was observed launching with ``--kv-cache-dtype fp8``."""
    sub, tr, _ = sub_agent_runner
    _force_cold_decision(monkeypatch)
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    if inherited:
        cfg = yaml.safe_load(base.read_text())
        cfg["benchmark"]["envs"]["EXTRA_SGLANG_ARGS"] = inherited
        base.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.explore.observed_launch_from_state",
        lambda _state: ({"kv_cache_dtype": "fp8", "max_running_requests": 512}, {}),
    )
    launched: list[str] = []

    def _fake_run(cmd, *args, **kwargs):
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        name = next(n for n in tputs if n in slot.parent.name + "/" + slot.name)
        launched.append(name)
        _fake_workspace(slot, tput=tputs[name])
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(tmp_path / "out"),
                "base_tput": 800.0,
                "grid": grid,
                **params,
            },
            idempotency_key="ex-noop-stack",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)
    return res.result, launched


@pytest.mark.asyncio
async def test_a_stack_revalidation_runs_although_it_restates_the_stack(sub_agent_runner, tmp_path, monkeypatch):
    """A stack revalidation exists to re-measure the stack, so restating it is the point."""
    grid = [{"name": "stack_revalidate", "extra_args": "--kv-cache-dtype fp8", "provenance": "stack_revalidate"}]
    out, launched = await _explore_on_an_observed_fp8_stack(
        sub_agent_runner, tmp_path, monkeypatch, grid, {"stack_revalidate": 800.0}, source="stack_revalidate"
    )

    assert launched == ["stack_revalidate"]
    assert out["skipped_dup"] == []


@pytest.mark.asyncio
async def test_rows_that_launch_the_same_server_are_benched_once(sub_agent_runner, tmp_path, monkeypatch):
    """Restating the stack's fp8 beside the same change launches the server the first row already measured."""
    grid = [
        {"name": "v_seqs", "extra_args": "--max-num-seqs 256", "extra_envs": {}},
        {"name": "v_fp8_seqs", "extra_args": "--kv-cache-dtype fp8 --max-num-seqs 256", "extra_envs": {}},
    ]
    out, launched = await _explore_on_an_observed_fp8_stack(
        sub_agent_runner,
        tmp_path,
        monkeypatch,
        grid,
        {"v_seqs": 800.4, "v_fp8_seqs": 800.4},
        inherited="--kv-cache-dtype fp8",
    )

    assert launched == ["v_seqs"]
    assert [(sd["name"], sd["reason"]) for sd in out["skipped_dup"]] == [("v_fp8_seqs", "round_dup")]


@pytest.mark.asyncio
async def test_rows_answered_without_measuring_leave_the_proposal_queue(sub_agent_runner, tmp_path, monkeypatch):
    """Restatements and duplicates are decided, so the auto-bench queue moves on to the proposal behind them."""
    proposals = [
        {"name": "v_noop_a", "extra_args": "--max-running-requests 512"},
        {"name": "v_noop_b", "extra_args": "--kv-cache-dtype fp8"},
        {"name": "v_seqs", "extra_args": "--max-num-seqs 256"},
        {"name": "v_fp8_seqs", "extra_args": "--kv-cache-dtype fp8 --max-num-seqs 256"},
        {"name": "v_lpm", "extra_args": "--schedule-policy lpm"},
    ]
    out, launched = await _explore_on_an_observed_fp8_stack(
        sub_agent_runner,
        tmp_path,
        monkeypatch,
        [{**row, "extra_envs": {}} for row in proposals[:4]],
        {"v_seqs": 800.4, "v_fp8_seqs": 800.4},
        inherited="--kv-cache-dtype fp8",
    )
    assert launched == ["v_seqs"]
    tested = out["explore_search_update"]["tested"]
    seqs_fp = next(fp for fp, row in tested.items() if row["name"] == "v_seqs")
    skipped = {
        row["name"]: (row["outcome"], row["reason"], row.get("duplicate_of", ""))
        for row in tested.values()
        if row["outcome"] == "SKIPPED_DEDUP"
    }
    assert skipped == {
        "v_noop_a": ("SKIPPED_DEDUP", "baseline_noop", ""),
        "v_noop_b": ("SKIPPED_DEDUP", "baseline_noop", ""),
        "v_fp8_seqs": ("SKIPPED_DEDUP", "round_dup", seqs_fp),
    }

    state = SharedState()
    state.specialist_rounds = [{"task_id": "s1", "domain": "serving_specialist", "cycle": 0, "proposal_set": proposals}]
    state.apply_explore_search_update(out["explore_search_update"])
    assert [row["name"] for row in state.untested_proposal_rows()] == ["v_lpm"]


@pytest.mark.asyncio
async def test_a_variant_that_was_the_stack_runs_once_a_keep_has_moved_the_stack(
    sub_agent_runner, tmp_path, monkeypatch
):
    """``--kv-cache-dtype fp8`` restates the round's opening stack, but not the stack after ``auto`` KEEPs."""
    grid = [
        {"name": "v_auto", "extra_args": "--kv-cache-dtype auto", "extra_envs": {}},
        {"name": "v_fp8", "extra_args": "--kv-cache-dtype fp8", "extra_envs": {}},
    ]
    out, launched = await _explore_on_an_observed_fp8_stack(
        sub_agent_runner, tmp_path, monkeypatch, grid, {"v_auto": 840.0, "v_fp8": 900.0}
    )

    assert launched == ["v_auto", "v_fp8"]
    outcomes = {row["name"]: row["outcome"] for row in out["explore_search_update"]["tested"].values()}
    assert outcomes == {"v_auto": "KEEP", "v_fp8": "KEEP"}
    assert not [sd for sd in out["skipped_dup"] if sd.get("reason") == "baseline_noop"]


@pytest.mark.asyncio
async def test_a_variant_that_is_the_stack_is_skipped_while_the_stack_has_not_moved(
    sub_agent_runner, tmp_path, monkeypatch
):
    """Before any KEEP, and after a REVERT, the observed launch still describes the stack."""
    grid = [
        {"name": "v_noop_first", "extra_args": "--max-running-requests 512", "extra_envs": {}},
        {"name": "v_revert", "extra_args": "--revert-flag", "extra_envs": {}},
        {"name": "v_noop_after", "extra_args": "--kv-cache-dtype fp8", "extra_envs": {}},
    ]
    out, launched = await _explore_on_an_observed_fp8_stack(
        sub_agent_runner, tmp_path, monkeypatch, grid, {"v_noop_first": 900.0, "v_revert": 800.4, "v_noop_after": 900.0}
    )

    assert launched == ["v_revert"]
    noop = {sd["name"] for sd in out["skipped_dup"] if sd.get("reason") == "baseline_noop"}
    assert noop == {"v_noop_first", "v_noop_after"}
    outcomes = {row["name"]: row["outcome"] for row in out["explore_search_update"]["tested"].values()}
    assert outcomes == {"v_noop_first": "SKIPPED_DEDUP", "v_revert": "REVERT", "v_noop_after": "SKIPPED_DEDUP"}


@pytest.mark.asyncio
async def test_a_restatement_dropped_up_front_does_not_claim_a_later_identical_variant(
    sub_agent_runner, tmp_path, monkeypatch
):
    """``fp8`` then ``auto`` then ``fp8``: the first ``fp8`` can only meet the opening stack; the last meets ``auto``."""
    grid = [
        {"name": "v_fp8_first", "extra_args": "--kv-cache-dtype fp8", "extra_envs": {}},
        {"name": "v_auto", "extra_args": "--kv-cache-dtype auto", "extra_envs": {}},
        {"name": "v_fp8_later", "extra_args": "--kv-cache-dtype fp8", "extra_envs": {}},
    ]
    out, launched = await _explore_on_an_observed_fp8_stack(
        sub_agent_runner, tmp_path, monkeypatch, grid, {"v_fp8_first": 900.0, "v_auto": 840.0, "v_fp8_later": 900.0}
    )

    assert launched == ["v_auto", "v_fp8_later"]
    assert {sd["name"] for sd in out["skipped_dup"] if sd.get("reason") == "baseline_noop"} == {"v_fp8_first"}
    assert not [sd for sd in out["skipped_dup"] if sd.get("reason") == "round_dup"]


@pytest.mark.asyncio
async def test_a_restatement_that_repeats_a_launch_of_this_round_is_a_round_dup(
    sub_agent_runner, tmp_path, monkeypatch
):
    """On a moved stack a restatement runs once; the same launch later in the round is the usual in-round duplicate."""
    grid = [
        {"name": "v_auto", "extra_args": "--kv-cache-dtype auto", "extra_envs": {}},
        {"name": "v_fp8_a", "extra_args": "--kv-cache-dtype fp8", "extra_envs": {}},
        {"name": "v_fp8_b", "extra_args": "--kv-cache-dtype fp8", "extra_envs": {}},
    ]
    out, launched = await _explore_on_an_observed_fp8_stack(
        sub_agent_runner, tmp_path, monkeypatch, grid, {"v_auto": 840.0, "v_fp8_a": 900.0, "v_fp8_b": 900.0}
    )

    assert launched == ["v_auto", "v_fp8_a"]
    assert [sd["name"] for sd in out["skipped_dup"] if sd.get("reason") == "round_dup"] == ["v_fp8_b"]


@pytest.mark.asyncio
async def test_a_grid_that_only_restates_the_stack_is_exhausted_before_the_round(
    sub_agent_runner, tmp_path, monkeypatch
):
    """With nothing ahead of them that could KEEP, every restatement is dropped before a round is planned."""
    grid = [
        {"name": "v_noop_a", "extra_args": "--max-running-requests 512", "extra_envs": {}},
        {"name": "v_noop_b", "extra_args": "--kv-cache-dtype fp8", "extra_envs": {}},
    ]
    out, launched = await _explore_on_an_observed_fp8_stack(
        sub_agent_runner, tmp_path, monkeypatch, grid, {"v_noop_a": 900.0, "v_noop_b": 900.0}
    )

    assert launched == []
    assert out["explore_grid_exhausted"] is True
    assert {sd["name"] for sd in out["skipped_dup"] if sd.get("reason") == "baseline_noop"} == {"v_noop_a", "v_noop_b"}


@pytest.mark.asyncio
async def test_a_restatement_stays_skipped_after_a_keep_on_an_unrelated_setting(
    sub_agent_runner, tmp_path, monkeypatch
):
    """A KEEP of ``--max-num-seqs 256`` does not touch the observed ``--kv-cache-dtype fp8``, so fp8 is still the stack."""
    grid = [
        {"name": "v_seqs", "extra_args": "--max-num-seqs 256", "extra_envs": {}},
        {"name": "v_fp8", "extra_args": "--kv-cache-dtype fp8", "extra_envs": {}},
    ]
    out, launched = await _explore_on_an_observed_fp8_stack(
        sub_agent_runner, tmp_path, monkeypatch, grid, {"v_seqs": 840.0, "v_fp8": 900.0}
    )

    assert launched == ["v_seqs"]
    assert [v["name"] for v in out["winners"]] == ["v_seqs"]
    skipped = [(sd["name"], sd["reason"]) for sd in out["skipped_dup"]]
    assert skipped == [("v_fp8", "baseline_noop")]


@pytest.mark.asyncio
async def test_a_restatement_left_leading_by_the_skip_list_is_skipped_at_its_turn(
    sub_agent_runner, tmp_path, monkeypatch
):
    """The operator skip list removes ``auto`` after the leading drop, so ``fp8`` meets the unmoved stack and is skipped."""
    monkeypatch.setenv("SKIP_VARIANTS", "v_auto")
    grid = [
        {"name": "v_auto", "extra_args": "--kv-cache-dtype auto", "extra_envs": {}},
        {"name": "v_fp8", "extra_args": "--kv-cache-dtype fp8", "extra_envs": {}},
    ]
    out, launched = await _explore_on_an_observed_fp8_stack(
        sub_agent_runner, tmp_path, monkeypatch, grid, {"v_auto": 840.0, "v_fp8": 900.0}
    )

    assert launched == []
    assert {sd["name"] for sd in out["skipped_dup"] if sd.get("reason") == "baseline_noop"} == {"v_fp8"}


@pytest.mark.asyncio
async def test_a_restatement_stays_skipped_after_a_keep_that_switches_the_stack_to_replace_mode(
    sub_agent_runner, tmp_path, monkeypatch
):
    """A KEEP that removes a flag persists the stack's launched args in replace mode, which moves only that flag.

    ``fp8`` then launches the server ``v_drop`` just measured, so it is that round's duplicate.
    """
    grid = [
        {
            "name": "v_drop",
            "extra_args": "--chunked-prefill-size 4096",
            "remove_args": ["--stale-flag"],
            "extra_envs": {},
        },
        {"name": "v_fp8", "extra_args": "--kv-cache-dtype fp8", "extra_envs": {}},
    ]
    out, launched = await _explore_on_an_observed_fp8_stack(
        sub_agent_runner,
        tmp_path,
        monkeypatch,
        grid,
        {"v_drop": 840.0, "v_fp8": 900.0},
        inherited="--stale-flag 1 --kv-cache-dtype fp8",
    )

    assert launched == ["v_drop"]
    (keep,) = out["winners"]
    assert keep["args_mode"] == "replace"
    assert [(sd["name"], sd["reason"]) for sd in out["skipped_dup"]] == [("v_fp8", "round_dup")]


@pytest.mark.asyncio
async def test_a_restatement_runs_after_a_keep_in_replace_mode_that_removes_what_it_restates(
    sub_agent_runner, tmp_path, monkeypatch
):
    """The same switch to replace mode releases ``fp8`` when the KEEP is the one that removed ``--kv-cache-dtype``."""
    grid = [
        {"name": "v_drop", "extra_args": "", "remove_args": ["--kv-cache-dtype"], "extra_envs": {}},
        {"name": "v_fp8", "extra_args": "--kv-cache-dtype fp8", "extra_envs": {}},
    ]
    out, launched = await _explore_on_an_observed_fp8_stack(
        sub_agent_runner,
        tmp_path,
        monkeypatch,
        grid,
        {"v_drop": 840.0, "v_fp8": 900.0},
        inherited="--kv-cache-dtype fp8",
    )

    assert launched == ["v_drop", "v_fp8"]


@pytest.mark.asyncio
async def test_a_restatement_twin_that_was_only_skipped_has_not_claimed_the_launch(
    sub_agent_runner, tmp_path, monkeypatch
):
    """``fp8`` listed twice: the first meets the unmoved stack and is skipped; after ``auto`` KEEPs, the second runs."""
    grid = [
        {"name": "v_seqs", "extra_args": "--max-num-seqs 256", "extra_envs": {}},
        {"name": "v_fp8_a", "extra_args": "--kv-cache-dtype fp8", "extra_envs": {}},
        {"name": "v_auto", "extra_args": "--kv-cache-dtype auto", "extra_envs": {}},
        {"name": "v_fp8_b", "extra_args": "--kv-cache-dtype fp8", "extra_envs": {}},
    ]
    out, launched = await _explore_on_an_observed_fp8_stack(
        sub_agent_runner,
        tmp_path,
        monkeypatch,
        grid,
        {"v_seqs": 840.0, "v_fp8_a": 900.0, "v_auto": 880.0, "v_fp8_b": 920.0},
    )

    assert launched == ["v_seqs", "v_auto", "v_fp8_b"]
    assert [(sd["name"], sd["reason"]) for sd in out["skipped_dup"]] == [("v_fp8_a", "baseline_noop")]


@pytest.mark.asyncio
async def test_a_round_of_measured_and_restated_variants_is_answered_not_failed(
    sub_agent_runner, tmp_path, monkeypatch
):
    """Everything is either already measured or a restatement of the stack: nothing is left to run, so it succeeded."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    sub.shared_state = state
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.explore.observed_launch_from_state",
        lambda _state: ({"kv_cache_dtype": "fp8"}, {}),
    )
    first: list[str] = []
    await _run_explore_round(
        sub,
        tr,
        tmp_path,
        state,
        key="round-1",
        grid=[{"name": "max_running_48", "extra_args": "--max-running-requests 48"}],
        base_args="--base-flag 1",
        bench_calls=first,
    )
    assert len(first) == 1

    second: list[str] = []
    out = await _run_explore_round(
        sub,
        tr,
        tmp_path,
        state,
        key="round-2",
        grid=[
            {"name": "mrr_48", "extra_args": "--max-running-requests 48"},
            {"name": "v_fp8", "extra_args": "--kv-cache-dtype fp8"},
        ],
        base_args="--base-flag 1",
        bench_calls=second,
    )

    assert second == []
    assert {(sd["name"], sd["reason"]) for sd in out["skipped_dup"]} == {
        ("mrr_48", "already_measured"),
        ("v_fp8", "baseline_noop"),
    }
    assert out["status"] == "succeeded"


@pytest.mark.asyncio
async def test_a_skip_never_replaces_a_measured_ledger_row(sub_agent_runner, tmp_path, monkeypatch):
    """fp8 measured before the stack was observed running it; restating it later leaves the measurement in place."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    sub.shared_state = state
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.explore.observed_launch_from_state",
        lambda _state: (dict(observed), {}),
    )
    grid = [{"name": "v_fp8", "extra_args": "--kv-cache-dtype fp8"}]
    await _run_explore_round(sub, tr, tmp_path, state, key="round-1", grid=grid, base_args="", bench_calls=[])
    measured = dict(state.explore_search["tested"])
    assert [row["outcome"] for row in measured.values()] == ["REVERT"]

    observed["kv_cache_dtype"] = "fp8"
    out = await _run_explore_round(sub, tr, tmp_path, state, key="round-2", grid=grid, base_args="", bench_calls=[])

    assert [(sd["name"], sd["reason"]) for sd in out["skipped_dup"]] == [("v_fp8", "baseline_noop")]
    assert out["explore_search_update"]["tested"] == {}
    assert state.explore_search["tested"] == measured


@pytest.mark.asyncio
async def test_restatements_left_leading_by_a_filter_are_dropped_before_the_serving_lease(
    sub_agent_runner, tmp_path, monkeypatch
):
    """A variant the skip list removes does not keep the restatements behind it in the round."""
    monkeypatch.setenv("SKIP_VARIANTS", "v_skipped")
    leases: list[int] = []
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.explore.maybe_serving_lease",
        lambda **kw: leases.append(1),
    )
    grid = [
        {"name": "v_skipped", "extra_args": "--kv-cache-dtype auto", "extra_envs": {}},
        {"name": "v_noop_a", "extra_args": "--max-running-requests 512", "extra_envs": {}},
        {"name": "v_noop_b", "extra_args": "--kv-cache-dtype fp8", "extra_envs": {}},
    ]
    out, launched = await _explore_on_an_observed_fp8_stack(
        sub_agent_runner, tmp_path, monkeypatch, grid, {"v_skipped": 840.0, "v_noop_a": 900.0, "v_noop_b": 900.0}
    )

    assert launched == []
    assert leases == []
    assert out["explore_grid_exhausted"] is True
    assert {sd["name"] for sd in out["skipped_dup"] if sd.get("reason") == "baseline_noop"} == {"v_noop_a", "v_noop_b"}


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
async def test_actual_explore_axis_rejection_cannot_be_revived_by_geak_fallback(
    sub_agent_runner, tmp_path, monkeypatch
):
    from hyperloom.inference_optimizer.tests.test_geak_gain_alignment import _coord, _ok_result

    _force_cold_decision(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    sub, tr, _ = sub_agent_runner
    coord = _coord(tmp_path, baseline=100.0, best_tput=110.0)
    state = coord.shared_state
    state.framework = "sglang"
    state.benchmark_mode = "agentx"
    state.baseline_perf = {
        "total_throughput": 1000.0,
        "e2e_norm_intvty_p90": 100.0,
        "e2e_norm_intvty_p50": 100.0,
        "duration_seconds": 25.0,
        "request_error_rate": 0.0,
    }
    state.current_best.update(state.baseline_perf)
    state.geak_result = _ok_result(final=150.0)
    sub.shared_state = state
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    fingerprint = canonical_fingerprint("--test-flag", {})
    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(tmp_path / "axis-rejection"),
                "base_tput": 110.0,
                "grid": [{"name": "candidate", "extra_args": "--test-flag"}],
                "source": "stack_revalidate",
                "geak_fallback": True,
                "expected_cfg_hash": fingerprint,
            },
            idempotency_key="geak-axis-rejection",
        ),
    )
    state.geak_pending = {"status": "awaiting_rebench", "revalidation_task_id": task.task_id}
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))

    def fake_measure(cmd, *args, **kwargs):
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        _fake_workspace(
            slot,
            tput=132.0,
            perf_axes={
                "total_token_throughput": 900.0,
                "e2e_norm_intvty_p90": 50.0,
                "e2e_norm_intvty_p50": 50.0,
                "duration_seconds": 25.0,
                "request_error_rate": 0.0,
            },
        )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    async def must_not_replay(**kwargs):
        pytest.fail("native rejection must settle the candidate before any favorable fallback can run")

    coord.writeback.validate_geak_via_geak_harness = must_not_replay
    with patch("hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill", side_effect=fake_measure):
        produced = (await sub.run_task(task)).result
    rejection = produced["per_variant_outcomes"][0]
    assert rejection["reason"].startswith("p50_gain_below_threshold,p90_regressed (")
    assert any(gate["gate"] == "graded_axes" and gate["passed"] is False for gate in rejection["gates"])
    await coord.writeback.promote_to_shared_state("explore", produced, task=task)
    assert state.current_best["tput"] == 110.0
    assert state.geak_result["revalidation_status"] == "no_promote"
    assert state.geak_result["revalidation_error"] == rejection["reason"]
    assert not state.optimization_stack


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_from", ["candidate", "current_best", "baseline"])
@pytest.mark.parametrize("missing_axis", ["total", "intvty"])
@pytest.mark.parametrize("output", [20000.0, 180.0])
async def test_explore_missing_axes_fails_closed(
    sub_agent_runner, tmp_path, monkeypatch, missing_from, missing_axis, output
):
    """Incomplete AgentX evidence fails closed instead of KEEPing on output throughput."""
    _force_cold_decision(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    sub, tr, _ = sub_agent_runner
    state = SharedState(framework="sglang", benchmark_mode="agentx")
    state.baseline_tput = 200.0
    state.baseline_perf = {
        "output_throughput": 200.0,
        "total_token_throughput": 20000.0,
        "e2e_norm_intvty_p90": 300.0,
        "e2e_norm_intvty_p50": 300.0,
        "duration_seconds": 25.0,
        "request_error_rate": 0.0,
    }
    base_tput = state.baseline_tput
    if missing_from == "current_best":
        state.current_best = {
            "action": "explore",
            "tput": 250.0,
            "total_token_throughput": 25000.0,
            "e2e_norm_intvty_p90": 300.0,
            "e2e_norm_intvty_p50": 300.0,
            "duration_seconds": 25.0,
            "request_error_rate": 0.0,
        }
        base_tput = 250.0
    candidate_axes = {
        "input_throughput": 20000.0,
        "total_token_throughput": 40000.0,
        "e2e_norm_intvty_p90": 300.0,
        "e2e_norm_intvty_p50": 300.0,
        "duration_seconds": 25.0,
        "request_error_rate": 0.0,
    }
    incomplete = {
        "candidate": candidate_axes,
        "current_best": state.current_best,
        "baseline": state.baseline_perf,
    }[missing_from]
    missing_keys = (
        ("input_throughput", "total_token_throughput")
        if missing_axis == "total"
        else ("e2e_norm_intvty_p90", "e2e_norm_intvty_p50")
    )
    for key in missing_keys:
        incomplete.pop(key, None)
    baseline_before = dict(state.baseline_perf)
    best_before = dict(state.current_best)
    sub.shared_state = state
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)

    def _fake_run(cmd, *args, **kwargs):
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        _fake_workspace(slot, tput=output, perf_axes=candidate_axes)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(tmp_path / "explore-missing-axes"),
                "base_tput": base_tput,
                "grid": [{"name": "v_incomplete", "extra_args": "--incomplete-flag"}],
            },
            idempotency_key="ex-missing-axes",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    tested = out["explore_search_update"]["tested"][canonical_fingerprint("--incomplete-flag", {})]
    assert tested["status"] == "succeeded"
    assert tested["outcome"] == "FAILED"
    assert tested["graded_objective"] == "output_throughput"
    assert tested["tput"] == output
    assert tested["base_tput"] == base_tput
    assert tested["gain_pct"] is None
    assert out["winners"] == []
    assert out["best_variant"] is None
    assert out["output_throughput"] is None
    assert any(gate["gate"] == "graded_axes" and gate["passed"] is False for gate in tested["gates"])
    assert state.baseline_perf == baseline_before
    assert state.current_best == best_before
    # A variant that could not be graded on its axes is the one row the ledger may still call unmeasured.
    assert tested["graded_gain_pct"] is None
    assert "no_meas" in SharedState._format_variant_line(tested)


@pytest.mark.asyncio
async def test_explore_guard_revert_row_reads_as_measured(sub_agent_runner, tmp_path, monkeypatch):
    """A variant the tail guard refuses keeps its graded movement and the gate's reason on its ledger row.

    The median here clears the KEEP bar by a wide margin and the slow tail halves, so the verdict is REVERT on a
    fully measured variant. The prompt ledger must say so rather than render the row as never measured.
    """
    _force_cold_decision(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    sub, tr, _ = sub_agent_runner
    state = SharedState(framework="sglang", benchmark_mode="agentx")
    state.baseline_tput = 200.0
    state.baseline_perf = {
        "output_throughput": 200.0,
        "total_token_throughput": 20000.0,
        "e2e_norm_intvty_p90": 100.0,
        "e2e_norm_intvty_p50": 100.0,
        "duration_seconds": 25.0,
        "request_error_rate": 0.0,
    }
    sub.shared_state = state
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)

    def _fake_run(cmd, *args, **kwargs):
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        _fake_workspace(
            slot,
            tput=200.0,
            perf_axes={
                "total_token_throughput": 20000.0,
                "e2e_norm_intvty_p90": 50.0,
                "e2e_norm_intvty_p50": 110.0,
                "duration_seconds": 25.0,
                "request_error_rate": 0.0,
            },
        )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(tmp_path / "explore-guard-revert"),
                "base_tput": 200.0,
                "grid": [{"name": "v_tail_regress", "extra_args": "--tail-flag"}],
            },
            idempotency_key="ex-guard-revert",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    tested = res.result["explore_search_update"]["tested"][canonical_fingerprint("--tail-flag", {})]
    assert tested["outcome"] == "REVERT"
    assert tested["graded_objective"] == "e2e_norm_intvty_p50"
    # A refused variant still contributes no gain anywhere a gain is summed or ranked.
    assert tested["gain_pct"] is None
    assert tested["graded_gain_pct"] == pytest.approx(10.0)
    assert tested["e2e_norm_intvty_p50"] == pytest.approx(110.0)
    assert tested["reason"].startswith("p90_regressed (")
    line = SharedState._format_variant_line(tested)
    assert "no_meas" not in line
    assert "+10.00%" in line
    assert "p90_regressed" in line


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_axis", ["total", "intvty"])
@pytest.mark.parametrize("incomplete_output", [170.0, 20000.0])
@pytest.mark.parametrize(
    "next_intvty,next_total,intvty_outcome",
    [(363.0, 23000.0, "KEEP"), (313.0, 20000.0, "REVERT"), (335.0, 22000.0, "REVERT")],
)
async def test_explore_missing_axes_preserves_running_grading_anchor(
    sub_agent_runner, tmp_path, monkeypatch, missing_axis, incomplete_output, next_intvty, next_total, intvty_outcome
):
    """An incomparable variant fails closed and leaves the intvty anchor on the last KEEP."""
    _force_cold_decision(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    sub, tr, _ = sub_agent_runner
    state = SharedState(framework="sglang", benchmark_mode="agentx")
    state.baseline_tput = 200.0
    state.baseline_perf = {
        "output_throughput": 200.0,
        "total_token_throughput": 20000.0,
        "e2e_norm_intvty_p90": 300.0,
        "e2e_norm_intvty_p50": 300.0,
        "duration_seconds": 25.0,
        "request_error_rate": 0.0,
    }
    sub.shared_state = state
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    observed_args: dict[str, str] = {}

    def _fake_run(cmd, *args, **kwargs):
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        name = slot.parent.name
        config = yaml.safe_load(Path(cmd[cmd.index("--benchmark-config") + 1]).read_text())
        observed_args[name] = config["benchmark"]["envs"]["EXTRA_SGLANG_ARGS"]
        output, total, intvty = {
            "v00_v_good": (210.0, 22000.0, 330.0),
            "v01_v_incomplete": (incomplete_output, 40000.0, 360.0),
            "v02_v_next": (220.0, next_total, next_intvty),
        }[name]
        axes = {
            "input_throughput": total - output,
            "total_token_throughput": total,
            "e2e_norm_intvty_p90": intvty,
            "e2e_norm_intvty_p50": intvty,
            "duration_seconds": 25.0,
            "request_error_rate": 0.0,
        }
        if name == "v01_v_incomplete":
            for key in (
                ("input_throughput", "total_token_throughput")
                if missing_axis == "total"
                else ("e2e_norm_intvty_p90", "e2e_norm_intvty_p50")
            ):
                axes.pop(key)
        _fake_workspace(slot, tput=output, perf_axes=axes)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(tmp_path / "explore-anchor-sequence"),
                "base_tput": 200.0,
                "grid": [
                    {"name": "v_good", "extra_args": "--good-flag"},
                    {"name": "v_incomplete", "extra_args": "--incomplete-flag"},
                    {"name": "v_next", "extra_args": "--next-flag"},
                ],
            },
            idempotency_key="ex-anchor-sequence",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    tested = {row["name"]: row for row in out["explore_search_update"]["tested"].values()}
    assert out["status"] == "succeeded"
    assert len(tested) == 3
    assert tested["v_good"]["outcome"] == "KEEP"
    assert tested["v_good"]["graded_objective"] == "e2e_norm_intvty_p50"
    assert tested["v_good"]["gain_pct"] == pytest.approx(10.0)
    assert tested["v_good"]["tput"] == 210.0
    assert tested["v_incomplete"]["status"] == "succeeded"
    assert tested["v_incomplete"]["outcome"] == "FAILED"
    assert tested["v_incomplete"]["graded_objective"] == "output_throughput"
    assert tested["v_incomplete"]["base_tput"] == 210.0
    assert tested["v_next"]["base_tput"] == 210.0
    assert tested["v_next"]["graded_objective"] == "e2e_norm_intvty_p50"
    if intvty_outcome == "REVERT":
        assert tested["v_next"]["gain_pct"] is None
    else:
        assert tested["v_next"]["gain_pct"] == pytest.approx((next_intvty / 330.0 - 1.0) * 100.0)
    assert tested["v_next"]["outcome"] == intvty_outcome
    assert "--good-flag" in observed_args["v02_v_next"]
    assert "--incomplete-flag" not in observed_args["v02_v_next"]
    assert "--next-flag" in observed_args["v02_v_next"]
    expected_winners = ["v_good"] + (["v_next"] if intvty_outcome == "KEEP" else [])
    assert [row["name"] for row in out["winners"]] == expected_winners
    assert [row["variant_name"] for row in out["explore_search_update"]["winners_history"]] == expected_winners
    assert out["output_throughput"] == (220.0 if intvty_outcome == "KEEP" else 210.0)
    # Promotion re-grades a winner from its record alone; a record thinner than the measurement reads as
    # candidate_axes_missing (or as an incomparable round) and the KEEP is dropped.
    for row in (*out["winners"], tested["v_good"]):
        snapshot = perf_snapshot_from_mapping(row)
        assert snapshot is not None
        assert snapshot["duration_seconds"] == 25.0
        assert snapshot["request_error_rate"] == 0.0


@pytest.mark.asyncio
async def test_an_agentx_explore_keep_is_promoted_by_writeback(sub_agent_runner, session_dir, tmp_path, monkeypatch):
    """The winner explore writes is the record writeback re-grades, so the KEEP must survive the hand-off.

    Each side was only ever exercised against hand-built dicts; a winner record thinner than the measurement that
    graded it dropped every AgentX explore KEEP at promotion from #1652 until #1790.
    """
    _force_cold_decision(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    baseline_axes = {
        "output_throughput": 200.0,
        "total_token_throughput": 20000.0,
        "e2e_norm_intvty_p90": 300.0,
        "e2e_norm_intvty_p50": 300.0,
        "duration_seconds": 25.0,
        "request_error_rate": 0.0,
    }
    sub, tr, _ = sub_agent_runner
    state = SharedState(framework="sglang", benchmark_mode="agentx")
    state.baseline_tput = 200.0
    state.baseline_perf = dict(baseline_axes)
    sub.shared_state = state
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)

    def _fake_run(cmd, *args, **kwargs):
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        _fake_workspace(
            slot,
            tput=210.0,
            perf_axes={
                "input_throughput": 21790.0,
                "total_token_throughput": 22000.0,
                "e2e_norm_intvty_p90": 330.0,
                "e2e_norm_intvty_p50": 330.0,
                "duration_seconds": 25.0,
                "request_error_rate": 0.0,
            },
        )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(tmp_path / "explore-promote"),
                "base_tput": 200.0,
                "grid": [{"name": "v_good", "extra_args": "--good-flag"}],
            },
            idempotency_key="ex-promote",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)
    [winner] = res.result["winners"]

    with session_scope(session_dir):
        coord = make_coordinator(session_dir)
        coord.shared_state.benchmark_mode = "agentx"
        coord.shared_state.baseline_tput = 200.0
        coord.shared_state.baseline_perf = dict(baseline_axes)
        coord.shared_state.current_best = {
            "action": "baseline",
            "tput": 200.0,
            **baseline_axes,
            "extra_server_args": "",
            "extra_envs": {},
        }

        assert coord.writeback.lift_to_current_best("explore", float(winner["tput"]), dict(winner)) is True
        assert coord.shared_state.current_best["variant_name"] == "v_good"
        assert coord.shared_state.current_best["e2e_norm_intvty_p50"] == 330.0


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["explicit-output", "synthetic-legacy"])
@pytest.mark.parametrize("output,expected_outcome", [(220.0, "KEEP"), (180.0, "REVERT")])
async def test_explore_required_axes_output_modes_keep_legacy_behavior(
    sub_agent_runner, tmp_path, monkeypatch, mode, output, expected_outcome
):
    """Missing total/interactivity stays irrelevant when output grading is requested."""
    _force_cold_decision(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    sub, tr, _ = sub_agent_runner
    state = SharedState(framework="sglang", benchmark_mode="agentx" if mode == "explicit-output" else "")
    state.baseline_tput = 200.0
    if mode == "explicit-output":
        monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "output_throughput")
    sub.shared_state = state
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)

    def _fake_run(cmd, *args, **kwargs):
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        _fake_workspace(slot, tput=output, perf_axes={})
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(tmp_path / "explore-output-mode"),
                "base_tput": 200.0,
                "grid": [{"name": "v_output", "extra_args": "--output-flag"}],
            },
            idempotency_key="ex-output-mode",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    tested = out["explore_search_update"]["tested"][canonical_fingerprint("--output-flag", {})]
    assert out["status"] == "succeeded"
    assert tested["status"] == "succeeded"
    assert tested["outcome"] == expected_outcome
    assert tested["graded_objective"] == "output_throughput"
    if expected_outcome == "KEEP":
        assert tested["gain_pct"] == pytest.approx((output / 200.0 - 1.0) * 100.0)
    else:
        assert tested["gain_pct"] is None
        assert out["losers"][0]["reason"] == "gain_below_threshold"
    assert bool(out["winners"]) is (expected_outcome == "KEEP")
    assert out["output_throughput"] == (output if expected_outcome == "KEEP" else None)


@pytest.mark.asyncio
async def test_explore_serving_no_eval_reverts_without_stopping(sub_agent_runner, tmp_path):
    """A high-risk serving variant that clears throughput but yields no accuracy verdict used to skip the gate (throughput-only fallback)."""
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    state.baseline_accuracy = 0.80
    sub.shared_state = state

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-acc-revert"

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=840.0)  # +5% vs base 800 (clears throughput)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    grid = [
        {
            # High-risk (precision) flag -> serving accuracy gate applies.
            "name": "v_risky",
            "extra_args": "--kv-cache-dtype fp8",
            "extra_envs": {},
            "provenance": "llm_direct",
        }
    ]
    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(output_dir),
                "base_tput": 800.0,
                "accuracy_baseline": 0.80,
                "grid": grid,
            },
            idempotency_key="ex-acc-revert",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    assert out["status"] == "succeeded"
    fp = canonical_fingerprint("--kv-cache-dtype fp8", {})
    tested = out["explore_search_update"]["tested"][fp]
    assert tested["outcome"] == "REVERT"
    reasons = {lr["name"]: lr.get("reason") for lr in out["losers"]}
    assert reasons.get("v_risky") == "accuracy_unavailable"
    # Post-baseline accuracy failure reverts the variant but never halts the run.
    assert state.stop_reason == ""
    # The arc is carried gate by gate: it cleared the gain bar and died on
    # accuracy. The outcome alone cannot say which of the two ended it.
    row = next(v for v in out["per_variant_outcomes"] if v["variant_name"] == "v_risky")
    assert [(g["gate"], g["passed"]) for g in row["gates"]] == [
        ("keep_threshold", True),
        ("accuracy", False),
    ]
    assert row["gates"][1]["reason"] == "accuracy_unavailable"
    # Nothing was adopted, so nothing stands behind an adoption.
    assert row["validation_basis"] == ""


@pytest.mark.asyncio
async def test_explore_gates_a_variant_no_flag_catalogue_would_have_caught(sub_agent_runner, tmp_path):
    """The gate no longer asks which knobs look risky."""
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    state.baseline_accuracy = 0.80
    sub.shared_state = state

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=840.0)  # +5% vs base 800 (clears throughput)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    extra_args = '--online_quant_config {"global_quant_config":"ptpc_fp8"}'
    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(tmp_path / "explore-uncatalogued"),
                "base_tput": 800.0,
                "accuracy_baseline": 0.80,
                "grid": [
                    {
                        "name": "v_quant",
                        "extra_args": extra_args,
                        "extra_envs": {},
                        "provenance": "llm_direct",
                    }
                ],
            },
            idempotency_key="ex-uncatalogued-acc",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    tested = res.result["explore_search_update"]["tested"][canonical_fingerprint(extra_args, {})]
    assert tested["outcome"] == "REVERT"
    reasons = {lr["name"]: lr.get("reason") for lr in res.result["losers"]}
    assert reasons.get("v_quant") == "accuracy_unavailable"
    assert state.stop_reason == ""


@pytest.mark.asyncio
async def test_explore_accuracy_gate_falls_back_to_shared_state(sub_agent_runner, tmp_path):
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    state.baseline_accuracy = 0.80
    sub.shared_state = state

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=840.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(tmp_path / "explore-shared-acc"),
                "base_tput": 800.0,
                "grid": [
                    {
                        "name": "v_risky",
                        "extra_args": "--attention-backend ROCM_AITER_FA",
                        "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
                        "provenance": "llm_direct",
                    }
                ],
            },
            idempotency_key="ex-shared-acc",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    fp = canonical_fingerprint(
        "--attention-backend ROCM_AITER_FA",
        {"VLLM_ROCM_USE_AITER": "1"},
    )
    tested = res.result["explore_search_update"]["tested"][fp]
    assert tested["outcome"] == "REVERT"
    reasons = {lr["name"]: lr.get("reason") for lr in res.result["losers"]}
    assert reasons.get("v_risky") == "accuracy_unavailable"


@pytest.mark.asyncio
async def test_explore_executor_keep_persists_effective_removal_stack(sub_agent_runner, tmp_path):
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=900.0)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(tmp_path / "explore-out"),
                "base_tput": 800.0,
                "base_extra_args": "--bad-base 1 --keep-base 2",
                "grid": [
                    {
                        "name": "remove_bad_base",
                        "extra_args": "--variant 4",
                        "remove_args": ["--bad-base"],
                        "provenance": "llm_direct",
                    }
                ],
            },
            idempotency_key="ex-remove-keep",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    winner = res.result["winners"][0]
    assert winner["remove_args"] == ["--bad-base"]
    assert "--bad-base" not in winner["extra_server_args"]
    assert "--keep-base 2" in winner["extra_server_args"]
    assert "--variant 4" in winner["extra_server_args"]
    assert res.result["best_variant"]["extra_server_args"] == winner["extra_server_args"]


@pytest.mark.asyncio
async def test_explore_executor_recovers_base_tput_from_shared_state(
    sub_agent_runner,
    tmp_path,
):
    """Regression: when params omits ``base_tput``, the executor recovers it from SharedState (else real wins are discarded)."""
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    sub.shared_state = state

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-base-tput-recovery"

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=840.0)  # +5% vs recovered baseline 800
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    grid = [
        {
            "name": "v_keep",
            "extra_args": "--keep-flag",
            "extra_envs": {},
            "provenance": "default_grid",
        }
    ]
    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(output_dir),
                # base_tput intentionally omitted to exercise SharedState recovery.
                "grid": grid,
            },
            idempotency_key="ex-base-tput-recovery",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    assert out["status"] == "succeeded"
    assert {w["name"] for w in out["winners"]} == {"v_keep"}
    fp = canonical_fingerprint("--keep-flag", {})
    tested = out["explore_search_update"]["tested"][fp]
    assert tested["outcome"] == "KEEP"
    assert tested["base_tput"] == 800.0


@pytest.mark.asyncio
async def test_per_variant_rows_carry_the_verdicts_and_the_stack(
    sub_agent_runner,
    tmp_path,
):
    """The round's own verdicts travel out with its numbers.

    Write-back records these on the framework timeline verbatim, because it
    cannot honestly rebuild them: an outcome of ``REVERT`` does not name the
    gate that ended the arc, and the stack a variant launched on has already
    moved on to whatever KEEP'd after it.
    """
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    sub.shared_state = state

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-verdict-carry"

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        _fake_workspace(Path(cmd[out_idx + 1]), tput=840.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(output_dir),
                "base_tput": 800.0,
                "grid": [
                    {
                        "name": "v_keep",
                        "extra_args": "--keep-flag",
                        "extra_envs": {},
                        "remove_args": ["--old-flag"],
                        "unset_envs": ["OLD_ENV"],
                        "args_mode": "replace",
                        "provenance": "llm_direct",
                    }
                ],
            },
            idempotency_key="ex-verdict-carry",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    row = next(v for v in res.result["per_variant_outcomes"] if v["variant_name"] == "v_keep")
    assert row["outcome"] == "KEEP"
    # The gain bar ruled, and says what it ruled against.
    keep_gate = next(g for g in row["gates"] if g["gate"] == "keep_threshold")
    assert keep_gate["passed"] is True
    assert keep_gate["observed"] == pytest.approx(5.0, abs=0.01)
    # This session has no baseline accuracy, so nothing gated the change and
    # the KEEP rests on throughput alone. The accuracy gate is absent rather
    # than reported as having passed.
    assert [g["gate"] for g in row["gates"]] == ["keep_threshold"]
    assert row["validation_basis"] == "keep_verdict_unscored"
    assert row["variant"]["remove_args"] == ["--old-flag"]
    assert row["variant"]["unset_envs"] == ["OLD_ENV"]
    assert row["variant"]["args_mode"] == "replace"
    # The stack it launched on: the anchor plus a base config still empty,
    # since nothing has KEPT before it.
    assert row["measured_against"]["throughput"] == 800.0
    assert row["measured_against"]["extra_server_args"] == ""


@pytest.mark.asyncio
async def test_explore_executor_prefers_current_best_over_baseline_for_recovery(
    sub_agent_runner,
    tmp_path,
):
    """SharedState recovery prefers ``current_best.tput`` over ``baseline_tput``, so a +5%-vs-baseline variant REVERTs against the best."""
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    state.current_best = {"action": "explore", "tput": 900.0}
    sub.shared_state = state

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-cb-recovery"

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=840.0)  # +5% vs baseline, -6.7% vs best
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    grid = [
        {
            "name": "v_below_best",
            "extra_args": "--below-best-flag",
            "extra_envs": {},
            "provenance": "default_grid",
        }
    ]
    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(output_dir),
                "grid": grid,
            },
            idempotency_key="ex-cb-recovery",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    assert out["status"] == "succeeded"
    assert out["winners"] == []
    fp = canonical_fingerprint("--below-best-flag", {})
    tested = out["explore_search_update"]["tested"][fp]
    assert tested["base_tput"] == 900.0
    assert tested["outcome"] == "REVERT"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "expected_base_tput", "expected_outcome", "has_winner"),
    [
        (None, 2358.80, "REVERT", False),
        ("stack_revalidate", 2192.52, "KEEP", True),
    ],
)
async def test_explore_executor_supersedes_stale_params_base_tput(
    sub_agent_runner,
    tmp_path,
    source,
    expected_base_tput,
    expected_outcome,
    has_winner,
):
    """Use the live anchor except when revalidating the complete stack."""
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 2195.86
    state.current_best = {"action": "replay_warm_recipe", "tput": 2358.80}
    sub.shared_state = state

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-stale-anchor"

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=2355.46)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    grid = [
        {
            "name": "minimax-fused-swiglu+moe-combine",
            "extra_args": "--fused-flag",
            "extra_envs": {},
            "provenance": "specialist:research_scout",
        }
    ]
    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(output_dir),
                # Snapshotted when the task was queued, before the warm replay landed.
                "base_tput": 2192.52,
                "grid": grid,
                **({"source": source} if source else {}),
            },
            idempotency_key="ex-stale-anchor",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    assert out["status"] == "succeeded"
    assert bool(out["winners"]) is has_winner
    fp = canonical_fingerprint("--fused-flag", {})
    tested = out["explore_search_update"]["tested"][fp]
    assert tested["base_tput"] == expected_base_tput
    assert tested["outcome"] == expected_outcome


@pytest.mark.asyncio
async def test_explore_executor_takes_live_base_args_with_the_live_anchor(
    sub_agent_runner,
    tmp_path,
):
    """Superseding a stale ``base_tput`` also re-reads the args it was measured on."""
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    state.current_best = {
        "action": "explore",
        "tput": 1000.0,
        "extra_server_args": "--live-layer 1",
        "extra_envs": {"LIVE_ENV": "1"},
    }
    sub.shared_state = state

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        _fake_workspace(Path(cmd[out_idx + 1]), tput=1100.0)  # +10% vs the live 1000
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(tmp_path / "explore-live-base"),
                # Snapshotted together at dispatch, before the newer layer landed.
                "base_tput": 800.0,
                "base_extra_args": "--stale-layer 1",
                "grid": [
                    {
                        "name": "on_live_stack",
                        "extra_args": "--variant 2",
                        "extra_envs": {},
                        "provenance": "llm_direct",
                    }
                ],
            },
            idempotency_key="ex-live-base-args",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    fp = canonical_fingerprint("--variant 2", {})
    assert out["explore_search_update"]["tested"][fp]["base_tput"] == 1000.0
    winner = out["winners"][0]
    assert "--live-layer 1" in winner["extra_server_args"]
    assert "--variant 2" in winner["extra_server_args"]
    assert "--stale-layer" not in winner["extra_server_args"]
    assert winner["extra_envs"]["LIVE_ENV"] == "1"


@pytest.mark.asyncio
async def test_explore_executor_historical_fingerprint_reruns(sub_agent_runner, tmp_path, monkeypatch):
    """A variant in explore_search.tested runs again; only same-grid exact duplicates are collapsed."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-dedup"

    bench_calls: list[str] = []

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        bench_calls.append(slot.name)
        _fake_workspace(slot, tput=900.0)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    fp_dup = canonical_fingerprint("--dup-flag", {})
    grid = [
        # Historical REVERT in ledger — must now execute (no cross-round block).
        {"name": "v_prior_revert", "extra_args": "--dup-flag", "extra_envs": {}, "provenance": "llm_direct"},
        # New variant — runs.
        {"name": "v_fresh", "extra_args": "--fresh-flag", "extra_envs": {}, "provenance": "llm_direct"},
        # Exact same-grid duplicate of v_prior_revert — collapsed to round_dup.
        {"name": "v_same_again", "extra_args": "--dup-flag", "extra_envs": {}, "provenance": "llm_direct"},
    ]
    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(output_dir),
                "base_tput": 800.0,
                "grid": grid,
                "explore_search": {
                    "tested": {
                        fp_dup: {
                            "fingerprint": fp_dup,
                            "name": "previous_run_name",
                            "extra_server_args": "--dup-flag",
                            "extra_envs": {},
                            "outcome": "REVERT",
                        }
                    },
                    "rejected": [],
                    "accepted": [],
                },
            },
            idempotency_key="ex-dedup",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    # v_same_again is a same-grid round_dup; v_prior_revert runs (no cross-round block).
    assert {d["name"] for d in out["skipped_dup"]} == {"v_same_again"}
    assert {d["reason"] for d in out["skipped_dup"]} == {"round_dup"}
    # Both historical and fresh variants execute (two bench calls, not one).
    assert len(bench_calls) == 2
    # Historical REVERT fingerprint was not blocked — it ran and can win.
    assert "v_prior_revert" in {w["name"] for w in out["winners"]}


@pytest.mark.asyncio
async def test_explore_executor_defaults_to_warm_decision_matching_hot_baseline(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
):
    """Default EXPLORE measures hot decisions, matching default hot baseline."""
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-warm"

    bench_calls: list[str] = []
    round_seeds: dict[bool, str | None] = {}

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        bench_calls.append(str(slot))
        round_seeds["warmup_round" in str(slot)] = (kwargs.get("env") or {}).get(BENCH_SEED_ENV)
        if "warmup_round" in str(slot):
            (slot / "server.log").write_text(
                "[aiter] shape is M:1, N:2, K:3, not found tuned config in /tmp/a.csv, will use default config!\n",
                encoding="utf-8",
            )
        _fake_workspace(slot, tput=920.0)  # +15% vs 800 — KEEP and stable
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(output_dir),
                "base_tput": 800.0,
                "grid": [
                    {
                        "name": "warm_keep",
                        "extra_args": "--warm-flag",
                        "extra_envs": {},
                        "provenance": "llm_direct",
                    }
                ],
                "baseline_runtime_sec": 10.0,
                "baseline_warm_runtime_sec": 5.0,
            },
            idempotency_key="ex-warm",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    # warmup (discarded) + decision == 2 Magpie runs.
    assert len(bench_calls) == 2, bench_calls
    assert sum("warmup_round" in c for c in bench_calls) == 1
    # The decision round reuses the warmup's server, so the warmup must not send the decision round's prompts.
    assert round_seeds == {True: "1", False: None}
    assert {w["name"] for w in out["winners"]} == {"warm_keep"}
    warm_log = next(Path(c) for c in bench_calls if "warmup_round" in c) / "server.log"
    decision_slot = next(Path(c) for c in bench_calls if "warmup_round" not in c)
    report = json.loads((decision_slot / "runtime_findings.json").read_text(encoding="utf-8"))
    assert report["log_path"] == str(warm_log)
    assert [(f["rule_id"], f["status"]) for f in report["findings"] if f["status"] != "not_detected"] == [
        ("aiter.tuned_miss", "detected")
    ]


def _run_eval_of(cmd: list[str]) -> str:
    """Read RUN_EVAL out of the materialized YAML a Magpie call was handed."""
    cfg_idx = cmd.index("--benchmark-config")
    with Path(cmd[cfg_idx + 1]).open() as f:
        cfg = yaml.safe_load(f)
    return str(cfg["benchmark"]["envs"].get("RUN_EVAL", "")).strip().lower()


@pytest.mark.asyncio
async def test_explore_decision_round_skips_eval_warmup_keeps_it(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
):
    """The overtime deadline is anchored on a throughput-only baseline, so the rounds it gates must measure throughput only."""
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-noeval"

    seen: list[tuple[str, str]] = []

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        seen.append((str(slot), _run_eval_of(cmd)))
        _fake_workspace(slot, tput=920.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(output_dir),
                "base_tput": 800.0,
                "grid": [
                    {
                        "name": "noeval_keep",
                        "extra_args": "--warm-flag",
                        "extra_envs": {},
                        "provenance": "llm_direct",
                    }
                ],
                "baseline_runtime_sec": 10.0,
                "baseline_warm_runtime_sec": 5.0,
            },
            idempotency_key="ex-noeval",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        await sub.run_task(task)

    warmup = [ev for slot, ev in seen if "warmup_round" in slot]
    decision = [ev for slot, ev in seen if "warmup_round" not in slot]
    assert warmup and not _eval_off(warmup[0])
    assert decision and all(_eval_off(ev) for ev in decision)


@pytest.mark.asyncio
async def test_explore_cold_decision_keeps_eval(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
):
    """Without server_lifecycle reuse there is no warmup round whose eval the decision round could fall back on, so it must run its own accuracy gate."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-coldeval"

    seen: list[tuple[str, str]] = []

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        seen.append((str(slot), _run_eval_of(cmd)))
        _fake_workspace(slot, tput=920.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(output_dir),
                "base_tput": 800.0,
                "grid": [
                    {
                        "name": "cold_keep",
                        "extra_args": "--cold-flag",
                        "extra_envs": {},
                        "provenance": "llm_direct",
                    }
                ],
                "baseline_runtime_sec": 10.0,
            },
            idempotency_key="ex-coldeval",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        await sub.run_task(task)

    assert not [slot for slot, _ in seen if "warmup_round" in slot]
    decision = [ev for _slot, ev in seen]
    assert decision and not any(_eval_off(ev) for ev in decision)


@pytest.mark.asyncio
async def test_explore_decision_stays_cold_when_the_session_skips_the_double_run(
    sub_agent_runner,
    tmp_path,
):
    """A cold ``baseline_tput`` must be graded cold even when lifecycle reuse is available."""
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    sub.shared_state = state

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    seen: list[str] = []

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        seen.append(str(slot))
        _fake_workspace(slot, tput=920.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(tmp_path / "explore-singleround"),
                "base_tput": 800.0,
                "baseline_double_run": False,
                "grid": [{"name": "v", "extra_args": "--flag", "extra_envs": {}, "provenance": "llm_direct"}],
            },
            idempotency_key="ex-no-double-run",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        await sub.run_task(task)

    assert not [slot for slot in seen if "warmup_round" in slot]


@pytest.mark.asyncio
async def test_explore_executor_warm_decision_warmup_failure_marks_failed(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
):
    """A failed warmup round records the variant FAILED(reason=warmup_failed), no decision run."""
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-warmfail"

    calls: list[str] = []

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        calls.append(str(slot))
        # Warmup boot fails (nonzero) — no workspace written.
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="boom")

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(output_dir),
                "base_tput": 800.0,
                "grid": [
                    {
                        "name": "warmfail",
                        "extra_args": "--warmfail-flag",
                        "extra_envs": {},
                        "reasoning": "Test whether this launch flag enables the scheduler fast path.",
                        "provenance": "llm_direct",
                    }
                ],
            },
            idempotency_key="ex-warmfail",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    assert len(calls) == 1, calls
    assert out["winners"] == []
    fp = canonical_fingerprint("--warmfail-flag", {})
    te = out["explore_search_update"]["tested"][fp]
    assert te["outcome"] == "FAILED"
    assert te["reason"] == "warmup_failed"
    assert te["stage"] == "warmup"
    assert "workspace" in te
    pvo = [v for v in out["per_variant_outcomes"] if v["outcome"] == "FAILED"]
    assert pvo, "expected FAILED entry in per_variant_outcomes"
    assert pvo[0]["stage"] == "warmup"
    assert pvo[0]["failure_attribution"] == "harness"
    assert pvo[0]["variant"]["reasoning_origin"] == "action_payload.reasoning"
    assert "failure_id" in pvo[0]
    assert pvo[0]["failure_id"].startswith("fail.")


@pytest.mark.asyncio
async def test_explore_executor_overtime_disabled_when_ratio_zero(
    sub_agent_runner,
    tmp_path,
):
    """ratio<=0 disables the gate; executor must NOT pass ``soft_deadline_sec``."""
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-no-overtime"

    received_deadlines: list[float | None] = []

    def _fake_kill(cmd, *args, **kwargs):
        received_deadlines.append(kwargs["watch"].silence_timeout_sec)
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=820.0)  # +2.5% KEEP
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(output_dir),
                "base_tput": 800.0,
                "grid": [
                    {
                        "name": "fast_variant",
                        "extra_args": "--fast-flag",
                        "extra_envs": {},
                        "provenance": "default_grid",
                    }
                ],
                "baseline_runtime_sec": 10.0,
            },
            idempotency_key="ex-overtime-off",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_kill,
    ):
        res = await sub.run_task(task)

    assert received_deadlines, "no Magpie calls were made"
    assert all(d == 600 for d in received_deadlines)
    out = res.result
    assert out["status"] == "succeeded"


@pytest.mark.asyncio
async def test_explore_variant_cap_is_clamped_to_the_session_budget(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
):
    """A granted cap never exceeds what is left of the session."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    state.max_minutes = 3.0
    sub.shared_state = state
    # Read before the run: the budget only shrinks from here, so a cap granted later can only be smaller than what
    # this allows.
    usable_sec = state.session_budget_usable_sec()

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    granted: list[int] = []

    def _fake_run(cmd, *args, **kwargs):
        # Only benchmark rounds carry --output-dir; the interpreter probe does not, and it is module-memoized, so
        # counting it would make this order-dependent.
        if "--output-dir" not in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")
        granted.append(int(kwargs["watch"].timeout))
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        _fake_workspace(slot, tput=840.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(tmp_path / "explore-clamp"),
                "base_tput": 800.0,
                "grid": [
                    {
                        "name": "v_fits",
                        "extra_args": "--max-num-seqs 256",
                        "extra_envs": {},
                        "provenance": "default_grid",
                    }
                ],
                "baseline_runtime_sec": 20.0,
            },
            idempotency_key="ex-budget-clamp",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    assert res.result["status"] == "succeeded"
    assert granted, f"the variant should have been admitted (20s expected, ~{usable_sec:.0f}s left)"
    assert all(t == 7800 for t in granted)


@pytest.mark.asyncio
async def test_explore_skips_a_variant_the_budget_cannot_fit(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
):
    """Admission is judged on the expected runtime, and refused when it does not fit."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    state.max_minutes = 3.0  # ~60s usable
    sub.shared_state = state

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    granted: list[int] = []

    def _fake_run(cmd, *args, **kwargs):
        if "--output-dir" not in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")
        granted.append(int(kwargs["watch"].timeout))
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        _fake_workspace(slot, tput=840.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(tmp_path / "explore-nofit"),
                "base_tput": 800.0,
                "grid": [
                    {
                        "name": "v_too_long",
                        "extra_args": "--max-num-seqs 256",
                        "extra_envs": {},
                        "provenance": "default_grid",
                    }
                ],
                "baseline_runtime_sec": 600.0,
            },
            idempotency_key="ex-budget-nofit",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    assert granted == [], "a variant needing 600s must not start with ~60s left"
    # Measuring nothing because the budget ran out is not the same as variants failing, and it must not be reported as
    # a bare, unattributed failure.
    assert res.result["error_class"] == "session_time_exhausted"
    assert "1 variant(s) went unmeasured" in res.result["error"]
    # Untested variants stay out of the ledger so a resume can retry them.
    assert res.result["losers"] == []
    assert res.result["explore_search_update"]["tested"] == {}


async def _explore_under_phase_budget(sub_agent_runner, tmp_path, *, phase_left_sec: list[float], variants: int):
    """Run a cold explore of ``variants`` 20s variants while the phase reads ``phase_left_sec`` before each one."""
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    sub.shared_state = state
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    started: list[int] = []

    def _fake_run(cmd, *args, **kwargs):
        if "--output-dir" not in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")
        started.append(int(kwargs["watch"].timeout))
        _fake_workspace(Path(cmd[cmd.index("--output-dir") + 1]), tput=840.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    grid = [
        {"name": f"v{i}", "extra_args": f"--max-num-seqs {64 << i}", "extra_envs": {}, "provenance": "default_grid"}
        for i in range(variants)
    ]
    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(tmp_path / "explore-phase"),
                "base_tput": 800.0,
                "grid": grid,
                "baseline_runtime_sec": 20.0,
            },
            idempotency_key=f"ex-phase-{variants}",
        ),
    )
    remaining = iter(phase_left_sec)
    phase_deadline = SimpleNamespace(remaining=lambda: next(remaining))
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task, extra_context={"phase_deadline": phase_deadline})
    return res.result, started


@pytest.mark.asyncio
async def test_explore_finishes_the_variant_in_flight_and_starts_none_the_phase_cannot_fit(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
):
    """The phase budget is judged before a variant starts, never against the one being measured."""
    _force_cold_decision(monkeypatch)
    result, started = await _explore_under_phase_budget(
        sub_agent_runner, tmp_path, phase_left_sec=[25.0, 10.0], variants=2
    )

    assert len(started) == 1
    assert result["status"] == "succeeded"


@pytest.mark.asyncio
async def test_explore_that_the_phase_cannot_fit_is_attributed_to_the_phase(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
):
    """A round the phase budget left unmeasured must not read as variants failing, nor as the session running out."""
    _force_cold_decision(monkeypatch)
    result, started = await _explore_under_phase_budget(sub_agent_runner, tmp_path, phase_left_sec=[10.0], variants=1)

    assert started == []
    assert result["error_class"] == "phase_time_exhausted"
    assert result["losers"] == []


@pytest.mark.asyncio
async def test_explore_leaves_a_variant_the_run_reaped_out_of_the_ledger(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
):
    """The common case: the budget expires while a variant is running, not before it."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    state.max_minutes = 600.0  # admits both variants; the reap comes mid-round
    state.baseline_runtime_sec = 20.0
    sub.shared_state = state

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    ran: list[str] = []

    def _fake_run(cmd, *args, **kwargs):
        if "--output-dir" not in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        ran.append(slot.name)
        if "v_reaped" not in str(slot):
            _fake_workspace(slot, tput=840.0)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")
        # The session deadline elapsed while this round was running, so the reaper tore the tree down and named the
        # cause.
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=SESSION_TIME_EXHAUSTED_RETURNCODE,
            stdout="",
            stderr="reaped",
        )

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(tmp_path / "explore-reaped"),
                "base_tput": 800.0,
                "grid": [
                    {
                        "name": "v_measured",
                        "extra_args": "--max-num-seqs 256",
                        "extra_envs": {},
                        "provenance": "default_grid",
                    },
                    {
                        "name": "v_reaped",
                        "extra_args": "--max-num-seqs 512",
                        "extra_envs": {},
                        "provenance": "default_grid",
                    },
                ],
                "baseline_runtime_sec": 20.0,
            },
            idempotency_key="ex-budget-reaped",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    tested = res.result["explore_search_update"]["tested"]
    # A variant the run reaped was never measured; recording it keeps a resume from ever retrying it, and teaches the
    # KB a clock's verdict.
    assert [t["name"] for t in tested.values()] == ["v_measured"]
    assert [lr["name"] for lr in res.result["losers"]] == []


@pytest.mark.asyncio
async def test_explore_leaves_a_variant_out_when_the_run_reaped_its_grid_warmup(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
):
    """The stop has to survive ``run_grid``'s own discarded warmup round."""
    _force_cold_decision(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "1")
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._server_lifecycle.resolve_lifecycle_params",
        lambda _config_path: {
            "eligible": True,
            "framework": "sglang",
            "port": 30000,
            "reason": "",
        },
    )
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    state.max_minutes = 600.0
    state.baseline_runtime_sec = 20.0
    sub.shared_state = state

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    ran: list[str] = []

    def _fake_run(cmd, *args, **kwargs):
        if "--output-dir" not in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        ran.append(slot.name)
        if slot.name == "warmup_round":
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=SESSION_TIME_EXHAUSTED_RETURNCODE,
                stdout="",
                stderr="reaped",
            )
        _fake_workspace(slot, tput=840.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(tmp_path / "explore-warmup-reaped"),
                "base_tput": 800.0,
                "grid": [
                    {
                        "name": "v_warmup_reaped",
                        "extra_args": "--max-num-seqs 256",
                        "extra_envs": {},
                        "provenance": "default_grid",
                    }
                ],
                "baseline_runtime_sec": 20.0,
            },
            idempotency_key="ex-budget-warmup-reaped",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    assert ran == ["warmup_round"], f"the decision round ran after the warmup was reaped: {ran}"
    assert res.result["explore_search_update"]["tested"] == {}
    assert res.result["losers"] == []
    assert res.result["error_class"] == SESSION_TIME_EXHAUSTED_CLASS
    assert "1 variant(s) went unmeasured" in res.result["error"]


@pytest.mark.asyncio
async def test_explore_attributes_a_round_the_run_reaped_before_anything_measured(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
):
    """With nothing measured the round is ``failed``, and must say who stopped it."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    state.max_minutes = 600.0
    state.baseline_runtime_sec = 20.0
    sub.shared_state = state

    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)

    def _fake_run(cmd, *args, **kwargs):
        if "--output-dir" not in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=ORCHESTRATOR_CANCELLED_RETURNCODE,
            stdout="",
            stderr="cancelled",
        )

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(tmp_path / "explore-cancelled"),
                "base_tput": 800.0,
                "grid": [
                    {
                        "name": "v_cancelled",
                        "extra_args": "--max-num-seqs 256",
                        "extra_envs": {},
                        "provenance": "default_grid",
                    }
                ],
                "baseline_runtime_sec": 20.0,
            },
            idempotency_key="ex-budget-cancelled",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    assert res.result["status"] == "failed"
    assert res.result["error_class"] == ORCHESTRATOR_CANCELLED_CLASS
    assert res.result["explore_search_update"]["tested"] == {}


@pytest.mark.asyncio
async def test_explore_executor_empty_grid_returns_failed(sub_agent_runner, tmp_path):
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
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
    _write_baseline_yaml(base)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("with_llm_proposal", [False, True])
async def test_explore_executor_measures_and_attributes_authored_levers(
    sub_agent_runner,
    tmp_path,
    monkeypatch,
    with_llm_proposal,
):
    """Seeded levers reach the runner and return measured attribution, with or without LLM proposals."""
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base_lever_grid.yaml"
    _write_baseline_yaml(base)
    sub.shared_state = _lever_state()
    _force_cold_decision(monkeypatch)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
    measurements = iter((900.0, 1000.0, 1100.0))

    def _fake_run(cmd, *args, **kwargs):
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        _fake_workspace(slot, tput=next(measurements))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill", _fake_run)
    grid = [{"name": "llm_proposal", "extra_server_args": "--max-num-seqs 256"}] if with_llm_proposal else []

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "base_tput": 800.0,
                "grid": grid,
            },
            idempotency_key="ex-fwlever-seeded-with-llm-grid",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    res = await sub.run_task(task)

    assert res.result["status"] == "succeeded"
    names = [row["variant_name"] for row in res.result["per_variant_outcomes"]]
    assert names == ["fwlever_hl_hoist", "fwlever_hl_cache"] + (["llm_proposal"] if with_llm_proposal else [])
    attributions = res.result["framework_lever_attributions"]
    assert [row["switch"] for row in attributions] == ["HL_HOIST", "HL_CACHE"]
    assert [row["gain_pct"] for row in attributions] == pytest.approx([12.5, 11.1111])
    assert all(row["outcome"] == "KEEP" and row["source"] == "additive" for row in attributions)


@pytest.mark.asyncio
async def test_explore_rejects_unsafe_aiter_unified_attn_before_benchmark(
    sub_agent_runner,
    tmp_path,
):
    sub, tr, _ = sub_agent_runner
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
    base = tmp_path / "base_sglang.yaml"
    _write_baseline_yaml(base)
    config = yaml.safe_load(base.read_text(encoding="utf-8"))
    config["benchmark"]["model"] = str(model)
    config["benchmark"]["precision"] = "fp8"
    config["benchmark"]["envs"]["EXTRA_SGLANG_ARGS"] = "--page-size 1 --kv-cache-dtype auto"
    base.write_text(yaml.safe_dump(config), encoding="utf-8")

    state = SharedState()
    state.model_path = str(model)
    state.model_name = "Qwen3-14B-FP8"
    state.model_type = "qwen3"
    state.gpu_type = "mi355x"
    state.stack_fingerprint_meta = {
        "sglang": "0.5.20.dev20260920+gc610c40399",
        "aiter": "4ad99832823dde2315b361cbd3b54b1c5c12acd5",
        "rocm": "10.0.0",
    }
    sub.shared_state = state

    output_dir = tmp_path / "explore-unified-attn-filter"
    benchmarked: list[str] = []

    def _fake_run(cmd, *args, **kwargs):
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        benchmarked.append(slot.relative_to(output_dir).as_posix())
        _fake_workspace(slot, tput=800.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(output_dir),
                "base_tput": 800.0,
                "baseline_double_run": False,
                "grid": [
                    {
                        "name": "unified",
                        "extra_envs": {
                            "SGLANG_USE_AITER": "1",
                            "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
                        },
                    },
                    {
                        "name": "unified_combo",
                        "extra_args": "--chunked-prefill-size 32768",
                        "extra_envs": {
                            "SGLANG_USE_AITER": "1",
                            "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
                            "SGLANG_USE_AITER_FP8_PER_TOKEN": "1",
                        },
                    },
                    {"name": "master", "extra_envs": {"SGLANG_USE_AITER": "1"}},
                    {
                        "name": "per_token",
                        "extra_envs": {
                            "SGLANG_USE_AITER": "1",
                            "SGLANG_USE_AITER_FP8_PER_TOKEN": "1",
                        },
                    },
                    {
                        "name": "unified_page16",
                        "extra_args": "--page-size 16",
                        "extra_envs": {
                            "SGLANG_USE_AITER": "1",
                            "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
                        },
                    },
                ],
            },
            idempotency_key="ex-unified-attn-filter",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        result = await sub.run_task(task)

    assert benchmarked == [
        "v00_master/variant_00_master",
        "v01_per_token/variant_00_per_token",
        "v02_unified_page16/variant_00_unified_page16",
    ]
    assert result.result["skipped_dup"] == [
        {
            "name": "unified",
            "reason": "compatibility_filter",
            "detail": "SGLANG_USE_AITER_UNIFIED_ATTN=1 is unsafe on the exact ROCm 10 Qwen3-14B-FP8 stack",
        },
        {
            "name": "unified_combo",
            "reason": "compatibility_filter",
            "detail": "SGLANG_USE_AITER_UNIFIED_ATTN=1 is unsafe on the exact ROCm 10 Qwen3-14B-FP8 stack",
        },
    ]
    other_stack = dict(state.stack_fingerprint_meta)
    other_stack["aiter"] = "newer-aiter"
    other_variant = GridVariant(
        "other_stack_unified",
        extra_envs={"SGLANG_USE_AITER_UNIFIED_ATTN": "1"},
    )
    kept, dropped = apply_compatibility_filter(
        [other_variant],
        serving=ServingStack("sglang", str(model), gpu_type="mi355x", stack_fingerprint=other_stack),
        base_server_args="--page-size 1 --kv-cache-dtype auto",
    )
    assert ([variant.name for variant in kept], dropped) == (["other_stack_unified"], [])


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


@pytest.mark.asyncio
async def test_explore_executor_historical_failed_and_accepted_rerun(sub_agent_runner, tmp_path, monkeypatch):
    """FAILED and accepted historical fingerprints may rerun; rejected contains latest attempt."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)
    output_dir = tmp_path / "explore-rerun"

    def _fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        _fake_workspace(slot, tput=900.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    fp_failed = canonical_fingerprint("--prev-failed", {})
    fp_accepted = canonical_fingerprint("--prev-kept", {})
    grid = [
        {"name": "rerun_failed", "extra_args": "--prev-failed", "extra_envs": {}, "provenance": "llm_direct"},
        {"name": "rerun_kept", "extra_args": "--prev-kept", "extra_envs": {}, "provenance": "llm_direct"},
    ]
    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(output_dir),
                "base_tput": 800.0,
                "grid": grid,
                "explore_search": {
                    "tested": {
                        fp_failed: {
                            "fingerprint": fp_failed,
                            "name": "rerun_failed",
                            "extra_server_args": "--prev-failed",
                            "extra_envs": {},
                            "outcome": "FAILED",
                            "gain_pct": None,
                        },
                    },
                    "rejected": [],
                    "accepted": [
                        {
                            "fingerprint": fp_accepted,
                            "name": "rerun_kept",
                            "extra_server_args": "--prev-kept",
                            "extra_envs": {},
                            "outcome": "KEEP",
                            "gain_pct": 5.0,
                        }
                    ],
                },
            },
            idempotency_key="ex-rerun-all",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    # Neither historical FAILED nor accepted fingerprint is blocked.
    assert not any(d["reason"] == "ledger_dup" for d in out["skipped_dup"])
    # Both ran (first one wins; second measures same tput as updated base so it won't KEEP).
    tested = out["explore_search_update"]["tested"]
    assert fp_failed in tested
    # The latest result for fp_failed overwrites the FAILED entry.
    assert tested[fp_failed]["outcome"] in ("KEEP", "REVERT", "FAILED", "KILLED_OVERTIME")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "budget_ms,expected_outcome,expected_reason",
    [
        (250.0, "REVERT", "latency_budget_exceeded"),
        (5000.0, "KEEP", ""),
        (0.0, "KEEP", ""),
    ],
)
async def test_explore_refuses_an_over_budget_winner_in_the_round_that_measured_it(
    sub_agent_runner, tmp_path, monkeypatch, budget_ms, expected_outcome, expected_reason
):
    """``--max-latency-ms`` rides the verdict explore's ladder already reads.

    The variant gains throughput either way; only the SLA separates the cases.
    Refusing here rather than at promotion keeps an over-budget variant from
    being folded onto the stack and becoming the anchor the rest of the batch is
    graded against, and gives the ledger a latency reason for the REVERT.
    """
    _force_cold_decision(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    sub, tr, _ = sub_agent_runner
    state = SharedState(framework="sglang")
    state.baseline_tput = 200.0
    state.latency_budget_ms = budget_ms
    sub.shared_state = state
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)

    def _fake_run(cmd, *args, **kwargs):
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        # The harness workspace reports e2el mean 2500 ms, well over the 250 ms budget.
        _fake_workspace(slot, tput=20000.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(tmp_path / f"explore-latency-{budget_ms:g}"),
                "base_tput": 200.0,
                "grid": [{"name": "v_slow_but_fast", "extra_args": "--split 8"}],
                "variant_timeout_sec": 10,
            },
            idempotency_key=f"ex-latency-{budget_ms:g}",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    tested = out["explore_search_update"]["tested"][canonical_fingerprint("--split 8", {})]
    assert tested["status"] == "succeeded"
    assert tested["outcome"] == expected_outcome
    if expected_outcome == "REVERT":
        # Not "gain_below_threshold": the variant gained 100x. Naming the wrong
        # gate would send the search looking for throughput it already has.
        assert out["losers"][0]["reason"] == expected_reason
        assert out["winners"] == []
        gates = {g["gate"]: g for g in tested["gates"]}
        assert "latency_budget" in gates
        assert gates["latency_budget"]["passed"] is False
    else:
        assert [w["name"] for w in out["winners"]] == ["v_slow_but_fast"]


@pytest.mark.asyncio
async def test_an_explore_winner_inside_the_budget_is_promoted_by_writeback(sub_agent_runner, tmp_path, monkeypatch):
    """Explore's own result, not a winner shaped for the gate, must survive promotion.

    Promotion re-checks the budget against the winner row explore returns. A row
    without ``e2el_mean_ms`` reads as unmeasured there, so a variant that passed
    explore's in-round check would be refused and explore could never move
    ``current_best``.
    """
    from hyperloom.orchestrator.loop import writeback as wb
    from hyperloom.orchestrator.loop.coordinator import Coordinator

    _force_cold_decision(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    sub, tr, _ = sub_agent_runner
    state = SharedState(framework="sglang", model_path="/models/m", gpu_type="mi355x")
    state.baseline_tput = 200.0
    state.latency_budget_ms = 5000.0
    sub.shared_state = state
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)

    def _fake_run(cmd, *args, **kwargs):
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        # The harness workspace reports e2el mean 2500 ms, inside the 5000 ms budget.
        _fake_workspace(slot, tput=20000.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(tmp_path / "explore-latency-promote"),
                "base_tput": 200.0,
                "grid": [{"name": "v_in_budget", "extra_args": "--split 8"}],
                "variant_timeout_sec": 10,
            },
            idempotency_key="ex-latency-promote",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    assert [w["name"] for w in out["winners"]] == ["v_in_budget"]
    assert out["winners"][0]["e2el_mean_ms"] == 2500.0

    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = state
    await coord.writeback._promote_explore(out, None, wb._PromoteOutcome(verdict=wb.Verdict.RECORDED))

    assert state.current_best["variant_name"] == "v_in_budget"


async def _mlperf_round(
    sub_agent_runner, tmp_path, monkeypatch, *, candidate: dict, name: str
) -> tuple[dict, SharedState]:
    """Run one explore variant on the MLPerf backend against a smoke baseline."""
    _force_cold_decision(monkeypatch)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("HYPERLOOM_AGENTIC_BACKEND", "mlperf")
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    sub, tr, _ = sub_agent_runner
    state = SharedState(framework="sglang", benchmark_mode="agentx", agentx_backend="mlperf")
    state.baseline_tput = 200.0
    state.baseline_accuracy = 0.72
    sub.shared_state = state
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)

    def _fake_run(cmd, *args, **kwargs):
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        _fake_workspace(slot, tput=candidate["output_throughput"], perf_axes=candidate)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(tmp_path / f"explore-{name}"),
                "base_tput": 200.0,
                "grid": [{"name": name, "extra_args": f"--{name}"}],
            },
            idempotency_key=f"ex-{name}",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        out = (await sub.run_task(task)).result
    return out["explore_search_update"]["tested"][canonical_fingerprint(f"--{name}", {})], state


def _mlperf_smoke(**over) -> dict:
    """A mapped smoke round: 150 trajectories, baseline-level accuracy, no failures."""
    return {
        "output_throughput": 220.0,
        "duration": 3500.0,
        "request_error_rate": 0.0,
        "submission_valid": True,
        "accuracy_score": 0.72,
        "accuracy_missing_turns": 0,
        **over,
    }


@pytest.mark.asyncio
async def test_explore_mlperf_keeps_an_output_throughput_gain(sub_agent_runner, tmp_path, monkeypatch):
    """+10% output throughput at the baseline's accuracy is a KEEP, graded on output throughput."""
    tested, _ = await _mlperf_round(sub_agent_runner, tmp_path, monkeypatch, candidate=_mlperf_smoke(), name="faster")
    assert tested["outcome"] == "KEEP"
    assert tested["graded_objective"] == "output_throughput"
    assert tested["gain_pct"] == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_explore_mlperf_reverts_an_accuracy_regression(sub_agent_runner, tmp_path, monkeypatch):
    """The throughput gain is real; the inline accuracy drop against the smoke baseline decides."""
    tested, _ = await _mlperf_round(
        sub_agent_runner, tmp_path, monkeypatch, candidate=_mlperf_smoke(accuracy_score=0.60), name="lossy"
    )
    assert tested["outcome"] == "REVERT"
    gates = {gate["gate"]: gate for gate in tested["gates"]}
    assert gates["accuracy"]["passed"] is False
    assert gates["accuracy"]["observed"] == pytest.approx(0.60)
    assert gates["accuracy"]["threshold"] == pytest.approx(0.72)


@pytest.mark.asyncio
async def test_explore_mlperf_reverts_when_turns_went_unscored(sub_agent_runner, tmp_path, monkeypatch):
    tested, _ = await _mlperf_round(
        sub_agent_runner, tmp_path, monkeypatch, candidate=_mlperf_smoke(accuracy_missing_turns=3), name="unscored"
    )
    assert tested["outcome"] == "REVERT"
    accuracy = {gate["gate"]: gate for gate in tested["gates"]}["accuracy"]
    assert accuracy["reason"] == "accuracy_drop"
    assert accuracy["observed"] == 0.0, "an unscored turn scores the run 0.0 rather than skipping the gate"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("power_w", "expected_outcome"),
    [pytest.param(812.0, "REVERT", id="over-budget"), pytest.param(650.0, "KEEP", id="in-budget")],
)
async def test_explore_grades_the_rounds_own_gpu_power_against_the_budget(
    sub_agent_runner, tmp_path, monkeypatch, power_w, expected_outcome
):
    """Power comes from the round's own gpu_power.json through the real extraction; the total of the serving cards
    is graded against the budget, and a KEEP reaches current_best."""

    from hyperloom.orchestrator.loop import writeback as wb
    from hyperloom.orchestrator.loop.coordinator import Coordinator

    _force_cold_decision(monkeypatch)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    sub, tr, _ = sub_agent_runner
    state = SharedState(framework="sglang", model_path="/models/m", gpu_type="mi355x")
    state.baseline_tput = 200.0
    state.power_budget_w = 700.0
    sub.shared_state = state
    base = tmp_path / "base.yaml"
    _write_baseline_yaml(base)

    def _fake_run(cmd, *args, **kwargs):
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        workspace = _fake_workspace(slot, tput=20000.0)
        report_path = workspace / "benchmark_report.json"
        assert report_path.exists()
        half = power_w / 2
        (workspace / "gpu_power.json").write_text(
            json.dumps(
                {
                    "started_unix": time.time(),
                    "serving_gpus": [4, 5],
                    "avg_power_w": half,
                    "per_gpu": {"4": {"avg_power_w": half}, "5": {"avg_power_w": half}},
                }
            )
        )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(tmp_path / "explore-power"),
                "base_tput": 200.0,
                "grid": [{"name": "v_power", "extra_args": "--power-lever"}],
                "variant_timeout_sec": 10,
            },
            idempotency_key="ex-power",
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        res = await sub.run_task(task)

    out = res.result
    tested = out["explore_search_update"]["tested"][canonical_fingerprint("--power-lever", {})]
    assert tested["outcome"] == expected_outcome
    if expected_outcome == "REVERT":
        assert out["losers"][0]["reason"] == "power_budget_exceeded"
        gates = {g["gate"]: g for g in tested["gates"]}
        assert gates["power_budget"]["passed"] is False
        assert "latency_budget" not in gates
        return
    assert out["winners"][0]["gpu_power_by_gpu_w"] == {"4": pytest.approx(power_w / 2), "5": pytest.approx(power_w / 2)}
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = state
    await coord.writeback._promote_explore(out, None, wb._PromoteOutcome(verdict=wb.Verdict.RECORDED))
    assert state.current_best["variant_name"] == "v_power"


async def _run_explore_round(
    sub,
    tr,
    tmp_path,
    state,
    *,
    key: str,
    grid: list[dict],
    base_args: str,
    bench_calls,
    extra_params=None,
    keep="",
    fail="",
):
    """Run one explore round on ``state`` and fold its ledger update back, as write-back does.

    Every variant measures within noise of the 800 anchor (REVERT) except the one named ``keep``, which KEEPs, and
    the one named ``fail``, whose benchmark exits non-zero.
    """

    def _fake_run(cmd, *args, **kwargs):
        slot = Path(cmd[cmd.index("--output-dir") + 1])
        bench_calls.append(slot.name)
        if fail and slot.name.endswith(f"_{fail}"):
            return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="boom")
        _fake_workspace(slot, tput=900.0 if keep and slot.name.endswith(f"_{keep}") else 801.0)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    base = tmp_path / "base.yaml"
    if not base.exists():
        _write_baseline_yaml(base)
    task, _ = await tr.create_or_return_existing(
        TaskSpec(
            kind="explore",
            params={
                "config_path": str(base),
                "output_dir": str(tmp_path / key),
                "base_tput": 800.0,
                "base_extra_args": base_args,
                "grid": grid,
                **(extra_params or {}),
            },
            idempotency_key=key,
        ),
    )
    sub.register_executor("explore", ExploreExecutor(session_dir=tmp_path))
    with patch(
        "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
        side_effect=_fake_run,
    ):
        out = (await sub.run_task(task)).result
    state.apply_explore_search_update(out["explore_search_update"])
    return out


@pytest.mark.asyncio
async def test_a_variant_measured_on_the_same_stack_is_not_measured_again(sub_agent_runner, tmp_path, monkeypatch):
    """A later round skips a launch an earlier round already gave a verdict on; the new variant still runs."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    sub.shared_state = state
    first: list[str] = []
    await _run_explore_round(
        sub,
        tr,
        tmp_path,
        state,
        key="round-1",
        grid=[{"name": "max_running_48", "extra_args": "--max-running-requests 48"}],
        base_args="--base-flag 1",
        bench_calls=first,
    )
    assert len(first) == 1

    second: list[str] = []
    out = await _run_explore_round(
        sub,
        tr,
        tmp_path,
        state,
        key="round-2",
        grid=[
            # Same launch under another name.
            {"name": "mrr_48", "extra_args": "--max-running-requests 48"},
            {"name": "fresh", "extra_args": "--fresh-flag"},
        ],
        base_args="--base-flag 1",
        bench_calls=second,
    )
    assert len(second) == 1
    assert [v["name"] for v in out["losers"]] == ["fresh"]
    skipped = {d["name"]: d for d in out["skipped_dup"]}
    assert skipped["mrr_48"]["reason"] == "already_measured"
    assert "max_running_48" in skipped["mrr_48"]["detail"]
    assert out["status"] == "succeeded"

    # A round with nothing left to measure is answered from the ledger, not failed.
    out = await _run_explore_round(
        sub,
        tr,
        tmp_path,
        state,
        key="round-3",
        grid=[{"name": "mrr_48", "extra_args": "--max-running-requests 48"}],
        base_args="--base-flag 1",
        bench_calls=second,
    )
    assert len(second) == 1
    assert out["status"] == "succeeded"

    # A skipped variant does not make a round whose only new variant failed look successful.
    out = await _run_explore_round(
        sub,
        tr,
        tmp_path,
        state,
        key="round-4",
        grid=[
            {"name": "mrr_48", "extra_args": "--max-running-requests 48"},
            {"name": "broken", "extra_args": "--broken-flag"},
        ],
        base_args="--base-flag 1",
        bench_calls=second,
        fail="broken",
    )
    assert out["status"] == "failed"


@pytest.mark.asyncio
async def test_a_budget_stop_after_a_skipped_variant_is_still_reported(sub_agent_runner, tmp_path, monkeypatch):
    """Skipping a measured variant does not hide that the budget stopped the new one before it ran."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    sub.shared_state = state
    measured = {"name": "max_running_48", "extra_args": "--max-running-requests 48"}
    calls: list[str] = []
    await _run_explore_round(sub, tr, tmp_path, state, key="r1", grid=[measured], base_args="", bench_calls=calls)
    state.max_minutes = 3.0  # ~60s usable
    out = await _run_explore_round(
        sub,
        tr,
        tmp_path,
        state,
        key="r2",
        grid=[measured, {"name": "fresh", "extra_args": "--fresh-flag"}],
        base_args="",
        bench_calls=calls,
        extra_params={"baseline_runtime_sec": 600.0},
    )
    assert len(calls) == 1
    assert [d["reason"] for d in out["skipped_dup"]] == ["already_measured"]
    assert out["status"] == "failed"
    assert out["error_class"] == "session_time_exhausted"


@pytest.mark.asyncio
async def test_a_measured_variant_with_removals_is_not_measured_again(sub_agent_runner, tmp_path, monkeypatch):
    """A variant's own removals are part of the launch it measured, so the same removal is the same launch."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    sub.shared_state = state
    grid = [{"name": "drop_base", "extra_args": "--x 1", "remove_args": ["--base-flag"]}]
    calls: list[str] = []
    await _run_explore_round(
        sub, tr, tmp_path, state, key="r1", grid=grid, base_args="--base-flag 1", bench_calls=calls
    )
    out = await _run_explore_round(
        sub, tr, tmp_path, state, key="r2", grid=grid, base_args="--base-flag 1", bench_calls=calls
    )
    assert len(calls) == 1
    assert [d["reason"] for d in out["skipped_dup"]] == ["already_measured"]


@pytest.mark.asyncio
async def test_a_measured_variant_runs_after_a_keep_earlier_in_the_same_round(sub_agent_runner, tmp_path, monkeypatch):
    """A KEEP advances the stack mid-round, so a variant measured on the round's opening stack is measured again."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    sub.shared_state = state
    measured = {"name": "max_running_48", "extra_args": "--max-running-requests 48"}
    calls: list[str] = []
    await _run_explore_round(sub, tr, tmp_path, state, key="r1", grid=[measured], base_args="", bench_calls=calls)
    out = await _run_explore_round(
        sub,
        tr,
        tmp_path,
        state,
        key="r2",
        grid=[{"name": "winner", "extra_args": "--winner 1"}, measured],
        base_args="",
        bench_calls=calls,
        keep="winner",
    )
    assert [v["name"] for v in out["winners"]] == ["winner"]
    assert [v["name"] for v in out["losers"]] == ["max_running_48"]
    assert out["skipped_dup"] == []
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_a_measured_variant_runs_again_once_the_stack_moved(sub_agent_runner, tmp_path, monkeypatch):
    """The verdict belongs to the stack it was measured on; on a different base launch the variant is measured anew."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    sub.shared_state = state
    grid = [{"name": "max_running_48", "extra_args": "--max-running-requests 48"}]
    calls: list[str] = []
    await _run_explore_round(
        sub, tr, tmp_path, state, key="r1", grid=grid, base_args="--base-flag 1", bench_calls=calls
    )
    out = await _run_explore_round(
        sub, tr, tmp_path, state, key="r2", grid=grid, base_args="--base-flag 2", bench_calls=calls
    )
    assert len(calls) == 2
    assert out["skipped_dup"] == []


@pytest.mark.asyncio
async def test_a_throughput_revert_is_measured_again_once_the_bar_decays_below_its_gain(
    sub_agent_runner, tmp_path, monkeypatch
):
    """The round's variants measure +0.125%: refused under a 1.0% bar, worth another look under a 0.1% bar."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    sub.shared_state = state
    grid = [{"name": "max_running_48", "extra_args": "--max-running-requests 48"}]
    calls: list[str] = []
    await _run_explore_round(
        sub,
        tr,
        tmp_path,
        state,
        key="r1",
        grid=grid,
        base_args="",
        bench_calls=calls,
        extra_params={"keep_threshold_pct": 1.0},
    )
    (row,) = state.explore_search["tested"].values()
    assert row["outcome"] == "REVERT" and row["reason"] == "gain_below_threshold"
    assert row["graded_gain_pct"] == pytest.approx(0.125)

    out = await _run_explore_round(
        sub,
        tr,
        tmp_path,
        state,
        key="r2",
        grid=grid,
        base_args="",
        bench_calls=calls,
        extra_params={"keep_threshold_pct": 0.5},
    )
    assert len(calls) == 1
    assert [d["reason"] for d in out["skipped_dup"]] == ["already_measured"]

    out = await _run_explore_round(
        sub,
        tr,
        tmp_path,
        state,
        key="r3",
        grid=grid,
        base_args="",
        bench_calls=calls,
        extra_params={"keep_threshold_pct": 0.1},
    )
    assert len(calls) == 2
    assert out["skipped_dup"] == []


@pytest.mark.asyncio
async def test_an_accuracy_revert_stays_answered_when_the_bar_decays(sub_agent_runner, tmp_path, monkeypatch):
    """Only the throughput gate moves with the bar: a variant refused on accuracy is not re-measured for a lower bar."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    sub.shared_state = state
    grid = [{"name": "max_running_48", "extra_args": "--max-running-requests 48"}]
    calls: list[str] = []
    await _run_explore_round(
        sub,
        tr,
        tmp_path,
        state,
        key="r1",
        grid=grid,
        base_args="",
        bench_calls=calls,
        extra_params={"keep_threshold_pct": 0.1},
    )
    (row,) = state.explore_search["tested"].values()
    row.update(outcome="REVERT", reason="accuracy_drop", graded_gain_pct=5.0)
    out = await _run_explore_round(
        sub,
        tr,
        tmp_path,
        state,
        key="r2",
        grid=grid,
        base_args="",
        bench_calls=calls,
        extra_params={"keep_threshold_pct": 0.1},
    )
    assert len(calls) == 1
    assert [d["reason"] for d in out["skipped_dup"]] == ["already_measured"]


@pytest.mark.asyncio
async def test_a_failed_variant_is_not_treated_as_measured(sub_agent_runner, tmp_path, monkeypatch):
    """A FAILED row carries no verdict on the launch, so the variant may run again on the same stack."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    sub.shared_state = state
    grid = [{"name": "max_running_48", "extra_args": "--max-running-requests 48"}]
    calls: list[str] = []
    await _run_explore_round(sub, tr, tmp_path, state, key="r1", grid=grid, base_args="", bench_calls=calls)
    for row in state.explore_search["tested"].values():
        row["outcome"] = "FAILED"
    await _run_explore_round(sub, tr, tmp_path, state, key="r2", grid=grid, base_args="", bench_calls=calls)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_a_stack_revalidation_measures_again(sub_agent_runner, tmp_path, monkeypatch):
    """A revalidation exists to re-measure, so a verdict already in the ledger does not skip it."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    sub.shared_state = state
    grid = [{"name": "max_running_48", "extra_args": "--max-running-requests 48"}]
    calls: list[str] = []
    await _run_explore_round(sub, tr, tmp_path, state, key="r1", grid=grid, base_args="", bench_calls=calls)
    await _run_explore_round(
        sub,
        tr,
        tmp_path,
        state,
        key="r2",
        grid=grid,
        base_args="",
        bench_calls=calls,
        extra_params={"source": "stack_revalidate"},
    )
    assert len(calls) == 2


def test_measured_verdict_holds_only_for_the_same_workload_and_stack():
    from hyperloom.orchestrator.actions.executors.measured_variants import (
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


@pytest.mark.asyncio
async def test_a_kernel_overlay_variant_is_not_skipped_by_a_plain_one(sub_agent_runner, tmp_path, monkeypatch):
    """An overlay changes the launch without changing args or envs, so a measured plain twin does not skip it."""
    _force_cold_decision(monkeypatch)
    sub, tr, _ = sub_agent_runner
    state = SharedState()
    state.baseline_tput = 800.0
    sub.shared_state = state
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    calls: list[str] = []
    plain = {"name": "plain", "extra_args": "--max-running-requests 48"}
    await _run_explore_round(sub, tr, tmp_path, state, key="r1", grid=[plain], base_args="", bench_calls=calls)
    out = await _run_explore_round(
        sub,
        tr,
        tmp_path,
        state,
        key="r2",
        grid=[{**plain, "name": "with_overlay", "overlay_pythonpath": str(overlay)}],
        base_args="",
        bench_calls=calls,
    )
    assert len(calls) == 2
    assert out["skipped_dup"] == []
