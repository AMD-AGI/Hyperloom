# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Regression tests for GEAK same-harness revalidation dispatch (L1/L2)."""

from __future__ import annotations

import asyncio
import json
import subprocess
import shlex
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.actions.executors._grid_runner import GridVariant
from hyperloom.orchestrator.actions.executors._proposal_identity import effective_fingerprint
from hyperloom.orchestrator.phases import machine_state as ps
from hyperloom.orchestrator.state.shared_state import ESCALATE_HINT_SKIP_TO_SWEEP


@pytest.fixture
def coordinator(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    from hyperloom.inference_optimizer.session.paths import make_session_dir as _msd
    from hyperloom.orchestrator.loop.coordinator import Coordinator
    from hyperloom.orchestrator.roles import (
        MockBackend,
        MockCriticBackend,
        ScriptedPlan,
    )
    from .conftest import seed_target_analysis_marker

    sd = _msd()
    seed_target_analysis_marker(sd)
    backends = {
        "orchestration": MockBackend(ScriptedPlan(turns=[]), name="orchestration"),
        "critic": MockCriticBackend(),
    }
    return Coordinator(sd, backends=backends)


def _geak_rebench_params(**extra: object) -> dict:
    return {
        "source": "resume_stack_revalidate",
        "geak_fallback": True,
        "reason": "geak_e2e_win",
        **extra,
    }


def _arm_kernel_to_sweep(st) -> None:
    now = datetime.now(timezone.utc)
    st.phase = ps.PHASE_KERNEL_AGENT
    st.phase_started_ts = (now - timedelta(minutes=5)).isoformat()
    st.phase_started_unix = (now - timedelta(minutes=5)).timestamp()
    st.start_ts = (now - timedelta(minutes=10)).isoformat()
    st.max_minutes = 96 * 60
    st.kernel_optimizer = "geak"
    st.geak_result = {"status": "ok", "accepted_config": {"flags": "--foo", "env": ""}}
    st.geak_pending = {}
    st.set_pending_escalate_hint(ESCALATE_HINT_SKIP_TO_SWEEP)


@pytest.mark.asyncio
async def test_agentx_direct_dispatch_fallback_refuses_geak_replay(coordinator, tmp_path, monkeypatch) -> None:
    c = coordinator
    st = c.shared_state
    _arm_kernel_to_sweep(st)
    st.benchmark_mode = "agentx"
    st.resume_pending_revalidation = True
    st.baseline_tput = 100.0
    st.current_best = {"action": "explore", "tput": 120.0}
    st.cumulative_gain_validated = 20.0
    st.geak_result = {
        "status": "ok",
        "throughput_speedup": 2.0,
        "accepted_config": {},
        "final_overlay": str(tmp_path / "missing-overlay"),
    }
    geak_dir = c.session_dir / "geak"
    geak_dir.mkdir()
    (geak_dir / "result.json").write_text(json.dumps(st.geak_result), encoding="utf-8")
    before_best = dict(st.current_best)

    async def _must_not_launch(**_kwargs):
        raise AssertionError("Direct dispatch fallback must refuse canonical AgentX replay")

    monkeypatch.setattr("hyperloom.orchestrator.actions.executors._geak_sweep.sweep_via_geak", _must_not_launch)
    c.phase_kernel._record_geak_kernel_journey = lambda _result: None
    summary = c._geak_rebench_params(reason="unit")
    assert summary["fallback"] == "geak_harness"
    await c._run_geak_kernel_phase(from_phase="KERNEL")

    assert st.current_best == before_best
    assert st.cumulative_gain_validated == 20.0
    assert st.resume_pending_revalidation is True
    assert "revalidation_task_id" not in st.geak_pending
    assert st.geak_pending["revalidation_error"] == "geak_harness_unsupported_canonical_workload"
    assert not any(entry.get("action") == "geak_e2e" for entry in st.optimization_stack)


@pytest.mark.asyncio
@pytest.mark.parametrize("bench_client", ["auto", "native", "inferencex"])
@pytest.mark.asyncio
async def test_agentx_2b_dispatch_uses_canonical_recipe_not_geak_client(coordinator, bench_client) -> None:
    c = coordinator
    st = c.shared_state
    st.benchmark_mode = "agentx"
    st.baseline_config_path = "/run/canonical-agentx.yaml"
    st.baseline_tput = 100.0
    st.geak_result = {
        "status": "ok",
        "bench_client": bench_client,
        "accepted_config": {"flags": "--candidate", "env": ""},
    }

    params = c._geak_rebench_params(reason="unit")

    assert params.get("geak_fallback") is True
    assert params["config_path"] == st.baseline_config_path
    assert params["grid"][0]["extra_args"] == "--candidate"
    assert "bench_client" not in params


@pytest.mark.parametrize(
    ("remove_args", "unset_envs", "args_mode"),
    [
        (["--speculative-algorithm"], ["SGLANG_ENABLE_SPECULATIVE"], "replace"),
        ("--speculative-algorithm", "SGLANG_ENABLE_SPECULATIVE", " REPLACE "),
        ([], [], "replace"),
        (None, None, None),
    ],
    ids=["lists", "scalars", "replace_only", "absent"],
)
@pytest.mark.asyncio
async def test_geak_rebench_preserves_native_base_removal_controls(
    coordinator, remove_args, unset_envs, args_mode
) -> None:
    c = coordinator
    st = c.shared_state
    st.baseline_tput = 100.0
    st.current_best = {
        "action": "explore",
        "tput": 110.0,
        "extra_server_args": "--incumbent",
        "extra_envs": {"SGLANG_USE_AITER": "0"},
    }
    if args_mode is not None:
        st.current_best.update(remove_args=remove_args, unset_envs=unset_envs, args_mode=args_mode)
    st.geak_result = {}
    native = await c._enqueue_internal_stack_rebench(reason="resume")
    native_row = await c.tasks.get(str(native["task_id"]))
    base_keys = {"base_remove_args", "base_unset_envs", "base_args_mode"}
    native_controls = {key: value for key, value in native_row.params.items() if key in base_keys}
    expected = {"base_args_mode": "replace"} if args_mode is not None else {}
    if remove_args:
        expected.update(
            base_remove_args=["--speculative-algorithm"],
            base_unset_envs=["SGLANG_ENABLE_SPECULATIVE"],
        )
    assert native_controls == expected

    st.geak_result = {
        "status": "ok",
        "accepted_config": {
            "flags": "--fp8-gemm-backend aiter",
            "env": "SGLANG_USE_AITER=1",
        },
    }
    enqueued = c._geak_rebench_params(reason="geak_e2e_win")

    assert enqueued.get("geak_fallback") is True
    assert enqueued["grid"][0]["extra_args"] == "--fp8-gemm-backend aiter"
    assert enqueued["grid"][0]["extra_envs"] == {"SGLANG_USE_AITER": "1"}
    assert {key: value for key, value in enqueued.items() if key in base_keys} == native_controls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("current", "accepted", "expected_flags", "expected_env"),
    [
        (
            {
                "extra_server_args": "--mem-fraction-static 0.9",
                "extra_envs": {},
                "args_mode": "replace",
                "remove_args": ["--disable-radix-cache"],
                "unset_envs": ["SGLANG_AITER_MLA_PERSIST"],
            },
            {"flags": "--mem-fraction-static 0.9", "env_map": {}},
            "--mem-fraction-static 0.9",
            None,
        ),
        (
            {"extra_server_args": "--chunked-prefill-size 1024 --mem-fraction-static 0.8"},
            {"flags": "--mem-fraction-static 0.9", "env_map": {}},
            "--disable-radix-cache --mem-fraction-static 0.9 --chunked-prefill-size 1024",
            "1",
        ),
        (
            {"extra_server_args": "--chunked-prefill-size 1024", "extra_envs": {"SGLANG_USE_AITER": "1"}},
            {"flags": "--mem-fraction-static 0.9", "env_map": {}, "args_mode": "replace"},
            "--mem-fraction-static 0.9",
            "1",
        ),
        (
            {"extra_server_args": "--chunked-prefill-size 1024"},
            {"flags": "", "env_map": {}, "args_mode": "replace"},
            "",
            "1",
        ),
        (
            {"extra_server_args": "--chunked-prefill-size 1024", "extra_envs": {"SGLANG_AITER_MLA_PERSIST": "2"}},
            {"remove_args": ["--disable-radix-cache"], "unset_envs": ["SGLANG_AITER_MLA_PERSIST"]},
            "--mem-fraction-static 0.7 --chunked-prefill-size 1024",
            None,
        ),
        (
            {
                "extra_server_args": "--mem-fraction-static 0.8",
                "args_mode": "replace",
                "unset_envs": ["SGLANG_AITER_MLA_PERSIST"],
            },
            {"flags": "--mem-fraction-static 0.9", "env_map": {"SGLANG_AITER_MLA_PERSIST": "3"}},
            "--mem-fraction-static 0.9",
            "3",
        ),
        (
            {"extra_server_args": "--chunked-prefill-size 1024", "extra_envs": {"SGLANG_AITER_MLA_PERSIST": "2"}},
            {"unset_envs": ["SGLANG_AITER_MLA_PERSIST"]},
            "--disable-radix-cache --mem-fraction-static 0.7 --chunked-prefill-size 1024",
            None,
        ),
        (
            {
                "extra_server_args": "--mem-fraction-static 0.8",
                "args_mode": "replace",
                "extra_envs": {"SGLANG_AITER_MLA_PERSIST": "3"},
                "unset_envs": ["SGLANG_AITER_MLA_PERSIST"],
            },
            {"flags": "--mem-fraction-static 0.9", "env_map": {}},
            "--mem-fraction-static 0.9",
            "3",
        ),
    ],
    ids=[
        "legacy_retains_removals",
        "legacy_delta",
        "complete_flags",
        "empty_replacement",
        "removal_only",
        "readd_env",
        "unset_only",
        "retained_env_override",
    ],
)
async def test_geak_launch_controls_reach_materialized_rebench(
    coordinator, tmp_path, monkeypatch, current, accepted, expected_flags, expected_env
) -> None:
    import yaml

    from hyperloom.orchestrator.actions.executors import ExploreExecutor, explore
    from hyperloom.orchestrator.actions.executors._grid_runner import VariantResult, _build_variant_yaml
    from hyperloom.orchestrator.state.shared_state import SharedState

    baseline = tmp_path / "baseline.yaml"
    baseline.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "sglang",
                    "model": "/models/test",
                    "run_mode": "local",
                    "benchmark_script": "sglang_custom.sh",
                    "envs": {
                        "TP": "1",
                        "CONC": "8",
                        "ISL": "256",
                        "OSL": "256",
                        "EXTRA_SGLANG_ARGS": "--disable-radix-cache --mem-fraction-static 0.7",
                        "SGLANG_AITER_MLA_PERSIST": "1",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    state = coordinator.shared_state
    state.baseline_config_path = str(baseline)
    state.baseline_tput = 100.0
    state.baseline_double_run = False
    state.current_best = {"tput": 110.0, **current}
    state.geak_result = {"schema_version": 2, "status": "ok", "accepted_config": accepted}
    enqueued = coordinator._geak_rebench_params(reason="launch_controls_regression")
    task = await coordinator.tasks.create(kind="explore", params=enqueued, idempotency_key="geak-revalidate-c0")
    calls = []
    fingerprints = []
    original_fingerprint = explore.effective_fingerprint

    def observe_fingerprint(*args, **kwargs):
        fingerprint = original_fingerprint(*args, **kwargs)
        fingerprints.append(fingerprint)
        return fingerprint

    async def capture_grid(**kwargs):
        output = tmp_path / f"materialized_{len(calls)}"
        output.mkdir()
        path = _build_variant_yaml(
            kwargs["base_yaml_path"],
            kwargs["base_extra_args"],
            kwargs["grid"][0],
            output_subdir=output,
            model_path=kwargs["model_path"],
            gpu_type=kwargs["gpu_type"],
            benchmark_script=kwargs["benchmark_script"],
            base_args_mode=kwargs["base_args_mode"],
        )
        calls.append(yaml.safe_load(path.read_text())["benchmark"]["envs"])
        if len(calls) > 1:
            return []
        return [
            VariantResult(
                name="geak_revalidate",
                extra_server_args=kwargs["grid"][0].extra_server_args,
                extra_envs=dict(kwargs["grid"][0].extra_envs),
                status="succeeded",
                output_throughput=120.0,
            )
        ]

    monkeypatch.setattr(explore, "run_grid", capture_grid)
    monkeypatch.setattr(explore, "effective_fingerprint", observe_fingerprint)
    monkeypatch.setattr(explore, "maybe_serving_lease", lambda **_kwargs: None)
    monkeypatch.setattr(explore, "teardown_lifecycle_server", lambda **_kwargs: None)
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._grid_variant_filter._probe_server_help_text", lambda _framework: ""
    )
    result = await ExploreExecutor(session_dir=coordinator.session_dir)._run_explore(
        SimpleNamespace(task=task, extra={"shared_state": state})
    )

    assert len(calls) == 1
    assert fingerprints == [task.params["expected_cfg_hash"]]
    envs = calls[0]
    flags = shlex.split(envs.get("EXTRA_SGLANG_ARGS", ""))
    if "--watchdog-timeout" in flags:
        position = flags.index("--watchdog-timeout")
        del flags[position : position + 2]
    assert flags == shlex.split(expected_flags)
    assert envs.get("SGLANG_AITER_MLA_PERSIST") == expected_env
    if current.get("extra_envs", {}).get("SGLANG_USE_AITER"):
        assert envs["SGLANG_USE_AITER"] == "1"
    assert len(result["winners"]) == 1
    assert coordinator._promote_geak_from_candidate(
        state.geak_result, measured_tput=120.0, measurement_provenance=result["best_variant"], overlay_loaded=False
    )
    state.geak_result = {}
    state.save(coordinator.session_dir)
    coordinator.shared_state = SharedState.load_or_init(coordinator.session_dir)
    resumed = await coordinator._enqueue_internal_stack_rebench(reason="launch_controls_resume")
    resume_task = await coordinator.tasks.get(str(resumed["task_id"]))
    await ExploreExecutor(session_dir=coordinator.session_dir)._run_explore(
        SimpleNamespace(task=resume_task, extra={"shared_state": coordinator.shared_state})
    )
    assert len(calls) == 2
    resume_flags = shlex.split(calls[1].get("EXTRA_SGLANG_ARGS", ""))
    if "--watchdog-timeout" in resume_flags:
        position = resume_flags.index("--watchdog-timeout")
        del resume_flags[position : position + 2]
    assert resume_flags == flags
    assert calls[1].get("SGLANG_AITER_MLA_PERSIST") == expected_env


@pytest.mark.asyncio
@pytest.mark.parametrize("with_removal_controls", [False, True], ids=["plain", "removal_controls"])
async def test_expected_cfg_hash_matches_the_variant_the_executor_builds(
    coordinator, with_removal_controls: bool
) -> None:
    """The pinned hash must describe the config the grid executor actually runs."""
    c = coordinator
    st = c.shared_state
    st.baseline_tput = 100.0
    st.macro_cycle = 0
    st.geak_result = {
        "status": "ok",
        "accepted_config": {
            "flags": "--fp8-gemm-backend aiter",
            "env": "PATH=/opt/venv/bin:/usr/bin SGLANG_USE_AITER=1 TP=1",
        },
    }
    controls = (
        {
            "remove_args": ["--speculative-algorithm"],
            "unset_envs": ["SGLANG_ENABLE_SPECULATIVE"],
            "args_mode": "replace",
        }
        if with_removal_controls
        else {}
    )
    st.current_best = {"extra_server_args": "--incumbent", **controls}

    params = c._geak_rebench_params(reason="geak_e2e_win")
    task = await c.tasks.create(kind="explore", params=params, idempotency_key="geak-revalidate-c0")
    entry = task.params["grid"][0]
    ran = GridVariant(
        str(entry["name"]),
        str(entry["extra_args"]),
        dict(entry["extra_envs"]),
    )

    assert "PATH" not in ran.extra_envs
    assert ran.extra_envs == {"SGLANG_USE_AITER": "1", "TP": "1"}
    assert task.params["expected_cfg_hash"] == effective_fingerprint(
        ran.extra_server_args,
        ran.extra_envs,
        base_remove_args=controls.get("remove_args"),
        base_unset_envs=controls.get("unset_envs"),
        base_args_mode=controls.get("args_mode"),
    )


@pytest.mark.asyncio
async def test_structured_environment_alone_dispatches_geak_rebench(coordinator) -> None:
    st = coordinator.shared_state
    st.baseline_tput = 100.0
    st.geak_result = {"status": "ok", "accepted_config": {"env_map": {"SGLANG_USE_AITER": "1"}}}

    params = coordinator._geak_rebench_params(reason="geak_e2e_win")
    entry = params["grid"][0]
    ran = GridVariant(str(entry["name"]), str(entry["extra_args"]), dict(entry["extra_envs"]))
    assert params["geak_fallback"] is True
    assert ran.extra_envs == {"SGLANG_USE_AITER": "1"}
    assert params["expected_cfg_hash"] == ran.fingerprint


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_env", ["", "SGLANG_USE_AITER=1"])
async def test_empty_structured_environment_does_not_rebench_legacy_values(coordinator, legacy_env) -> None:
    st = coordinator.shared_state
    st.baseline_tput = 100.0
    st.geak_result = {"status": "ok", "accepted_config": {"env_map": {}, "env": legacy_env}}

    enqueued = coordinator._geak_rebench_params(reason="geak_e2e_win")
    assert enqueued == {"skipped": True, "reason": "geak_no_material"}
    assert not await coordinator.tasks.queued()


@pytest.mark.asyncio
@pytest.mark.parametrize("env_map", [None, [], {"SGLANG_USE_AITER": 1}, {"BAD-NAME": "1"}, {"VALID": "a\0b"}])
async def test_malformed_structured_environment_does_not_dispatch(coordinator, env_map) -> None:
    coordinator.shared_state.geak_result = {"status": "ok", "accepted_config": {"env_map": env_map}}
    result = coordinator._geak_rebench_params(reason="geak_e2e_win")
    assert result == {"skipped": True, "reason": "geak_invalid_config"}
    assert coordinator.shared_state.geak_result["revalidation_status"] == "no_promote"
    assert not coordinator.shared_state.geak_pending
    assert not await coordinator.tasks.queued()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "material",
    [{"final_patch": "/geak/final.patch"}, {"accepted_kernels": ["gemm"]}, {"accepted_heads": ["attention"]}],
)
async def test_artifact_only_result_requires_its_own_harness(coordinator, material) -> None:
    coordinator.shared_state.geak_result = {"status": "ok", **material}
    enqueued = coordinator._geak_rebench_params(reason="geak_e2e_win")
    assert enqueued == {"skipped": True, "reason": "geak_material_requires_harness", "fallback": "geak_harness"}
    assert not await coordinator.tasks.queued()


@pytest.mark.asyncio
async def test_recovered_empty_map_closes_without_fallback(coordinator, tmp_path, monkeypatch) -> None:
    c = coordinator
    _arm_kernel_to_sweep(c.shared_state)
    geak_dir = tmp_path / "geak"
    geak_dir.mkdir()
    result = {"status": "ok", "final_throughput_tok_s": 116.0, "accepted_config": {"env_map": {}}}
    (geak_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    c.shared_state.geak_result = {}

    async def _must_not_fallback(**_kwargs):
        pytest.fail("empty optimization must not launch a fallback")

    monkeypatch.setattr(c, "_validate_geak_via_geak_harness", _must_not_fallback)
    await c._run_geak_kernel_phase(from_phase="KERNEL")

    assert c.shared_state.geak_result["revalidation_status"] == "no_material"
    assert not c.shared_state.geak_pending
    assert not c.shared_state.resume_pending_revalidation
    assert not await c.tasks.queued()


def test_material_check_ignores_untrusted_env_names() -> None:
    """An untrusted key on one side only must not read as a config difference."""
    from hyperloom.orchestrator.loop.coordinator_helpers import _geak_result_has_material

    echoed = {
        "status": "ok",
        "accepted_config": {
            "flags": "--fp8-gemm-backend aiter",
            "env": "PATH=/opt/venv/bin SGLANG_USE_AITER=1",
        },
    }
    assert not _geak_result_has_material(
        echoed,
        prev_best_flags="--fp8-gemm-backend aiter",
        prev_best_envs={"SGLANG_USE_AITER": "1"},
    )
    # A real config delta still registers.
    assert _geak_result_has_material(
        echoed,
        prev_best_flags="--fp8-gemm-backend triton",
        prev_best_envs={"SGLANG_USE_AITER": "1"},
    )


@pytest.mark.parametrize(
    ("accepted", "previous", "expected"),
    [
        ({"flags": "", "args_mode": "replace"}, {"flags": "--disable-radix-cache"}, True),
        ({"flags": "", "args_mode": "replace"}, {"flags": "", "args_mode": "replace"}, False),
        (
            {"flags": "--mem-fraction-static 0.9", "args_mode": "replace"},
            {"flags": "--mem-fraction-static 0.9", "args_mode": "replace"},
            False,
        ),
        ({"flags": "--mem-fraction-static 0.9"}, {"flags": "--mem-fraction-static 0.9", "args_mode": "replace"}, False),
        ({"flags": "", "env_map": {}}, {"flags": "--disable-radix-cache", "args_mode": "replace"}, False),
        ({"unset_envs": ["SGLANG_AITER_MLA_PERSIST"]}, {"env_map": {"SGLANG_AITER_MLA_PERSIST": "1"}}, True),
    ],
)
def test_material_check_distinguishes_explicit_controls_from_empty_legacy(accepted, previous, expected):
    from hyperloom.orchestrator.loop.coordinator_helpers import _geak_result_has_material

    assert (
        _geak_result_has_material(
            {"status": "ok", "accepted_config": accepted},
            prev_best_flags=previous.get("flags", ""),
            prev_best_envs=previous.get("env_map", {}),
            prev_best_controls=previous,
        )
        is expected
    )


@pytest.mark.asyncio
async def test_resume_stack_revalidate_promotes_material_geak_candidate(coordinator) -> None:
    """The legacy source label must not suppress a proven GEAK product."""
    c = coordinator
    st = c.shared_state
    st.baseline_tput = 100.0
    st.current_best = {
        "action": "baseline",
        "tput": 110.0,
        "extra_server_args": "--incumbent",
        "extra_envs": {},
    }
    st.geak_result = {
        "status": "ok",
        "accepted_config": {"flags": "--candidate", "env": ""},
        "accepted_kernels": ["replacement_kernel"],
    }

    task = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key="geak-revalidate-c0",
        task_id="material-geak-rebench",
    )
    st.geak_pending = {"status": "awaiting_rebench", "revalidation_task_id": task.task_id}

    await c._promote_to_shared_state(
        task.kind,
        {
            "output_throughput": 120.0,
            "best_variant": {"fingerprint": "candidate-hash"},
            "winners": [],
        },
        task=task,
    )

    assert st.current_best["action"] == "geak_e2e"
    assert st.current_best["tput"] == pytest.approx(120.0)
    assert any(entry.get("action") == "geak_e2e" for entry in st.optimization_stack)
    assert st.geak_pending == {}


@pytest.mark.asyncio
async def test_resume_stack_revalidate_rejects_same_config_noise(coordinator) -> None:
    """A faster remeasure of unchanged config is not a GEAK optimization."""
    c = coordinator
    st = c.shared_state
    st.baseline_tput = 100.0
    st.current_best = {
        "action": "explore",
        "tput": 110.0,
        "extra_server_args": "--same-config",
        "extra_envs": {"SGLANG_USE_AITER": "1"},
    }
    st.geak_result = {
        "status": "ok",
        "accepted_config": {
            "flags": "--same-config",
            "env": "SGLANG_USE_AITER=1",
        },
    }

    task = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key="geak-revalidate-c0",
        task_id="same-config-geak-rebench",
    )
    st.geak_pending = {"status": "awaiting_rebench", "revalidation_task_id": task.task_id}

    await c._promote_to_shared_state(
        task.kind,
        {
            "output_throughput": 120.0,
            "best_variant": {"fingerprint": "same-config-hash"},
            "winners": [],
        },
        task=task,
    )

    assert st.current_best["action"] == "explore"
    assert st.current_best["tput"] == pytest.approx(110.0)
    assert not any(entry.get("action") == "geak_e2e" for entry in st.optimization_stack)
    assert st.geak_result["revalidation_status"] == "no_material"
    assert st.geak_pending == {}


@pytest.mark.asyncio
async def test_no_material_drop_does_not_claim_the_stack_was_revalidated(coordinator) -> None:
    """Dropping a candidate is not a revalidation of the stack behind it.

    ``resume_pending_revalidation`` means the accepted stack still owes a
    post-resume remeasure, and the contract everywhere else in this function is
    that only a reconciled watermark clears it. Scoped to the canonical
    workload; a replayable one keeps clearing the flag as it always has.
    """
    c = coordinator
    st = c.shared_state
    st.benchmark_mode = "agentx"
    st.baseline_tput = 100.0
    st.baseline_perf = {"total_throughput": 1000.0, "e2e_norm_intvty_p90": 30.0}
    st.current_best = {
        "action": "explore",
        "tput": 110.0,
        "total_throughput": 1100.0,
        "e2e_norm_intvty_p90": 33.0,
        "extra_server_args": "--same-config",
        "extra_envs": {"SGLANG_USE_AITER": "1"},
    }
    st.optimization_stack = [{"action": "explore", "tput": 110.0}]
    st.cumulative_gain_validated = 10.0
    st.cumulative_gain_validated_stack_len = 0
    st.resume_pending_revalidation = True
    st.geak_result = {
        "status": "ok",
        "accepted_config": {"flags": "--same-config", "env": "SGLANG_USE_AITER=1"},
    }

    task = await c.tasks.create(
        kind="explore",
        params=_geak_rebench_params(),
        idempotency_key="geak-revalidate-c0",
        task_id="no-material-watermark",
    )
    st.geak_pending = {"status": "awaiting_rebench", "revalidation_task_id": task.task_id}

    await c._promote_to_shared_state(
        task.kind,
        {
            "output_throughput": 120.0,
            "best_variant": {
                "fingerprint": "same-config-hash",
                "output_throughput": 120.0,
                "total_throughput": 1200.0,
                "e2e_norm_intvty_p90": 36.0,
            },
            "winners": [],
        },
        task=task,
    )

    assert st.geak_result["revalidation_status"] == "no_material"
    assert not (
        st.resume_pending_revalidation is False and st.cumulative_gain_validated_stack_len < len(st.optimization_stack)
    ), "no_material cleared the revalidation flag without reconciling the validation watermark"


def _render_final(
    geak_candidate: dict, *, rebench: dict | None = None, claim: dict | None = None
) -> tuple[list[str], list[str]]:
    from hyperloom.inference_optimizer.breakdown.reporters._renderers.final import render

    timeline = []
    if rebench is not None or claim is not None:
        timeline.append({"type": "kernel", "ext": {"geak": {"rebench": rebench or {}, "claim": claim or {}}}})
    section = render(
        {
            "outcome": {
                "final": {"throughput_tok_s_per_gpu": 140.0, "gain_pct": 0.0},
                "baseline": {"throughput_tok_s_per_gpu": 100.0},
                "validation": {},
            },
            "close": {"geak_candidate": geak_candidate},
            "timeline": timeline,
        }
    )
    return list(section.key_facts), list(section.warnings)


def test_final_report_surfaces_cancelled_geak_revalidation() -> None:
    """A measured candidate dropped for a missed rebench must be visible."""
    facts, warnings = _render_final(
        {
            "status": "rebench_cancelled",
            "revalidation_error": "close_sequence",
            "self_reported_gain_pct": 12.5,
        }
    )

    blob = " ".join(facts + warnings).lower()
    assert "rebench" in blob
    assert any("close_sequence" in w or "could not" in w.lower() for w in warnings)
    # The dropped candidate must not be presented as awaiting anything.
    assert not any("awaiting" in f.lower() for f in facts)


def test_final_report_surfaces_failed_geak_revalidation() -> None:
    facts, warnings = _render_final(
        {},
        rebench={"final_status": "failed", "final_error": "subprocess_nonzero"},
        claim={"self_reported_gain_pct": 3.2},
    )

    blob = " ".join(facts + warnings).lower()
    assert "dropped" in blob
    assert "subprocess_nonzero" in blob


def test_final_report_reads_the_last_geak_visit_not_an_earlier_failure() -> None:
    """A terminal status outlives the visit that stamped it.

    An earlier cycle's ``failed`` must not outrank a later cycle's rebench
    that concluded nothing about a candidate still in play.
    """
    from hyperloom.inference_optimizer.breakdown.reporters._renderers.final import render

    section = render(
        {
            "outcome": {
                "final": {"throughput_tok_s_per_gpu": 140.0, "gain_pct": 0.0},
                "baseline": {"throughput_tok_s_per_gpu": 100.0},
                "validation": {},
            },
            "close": {"geak_candidate": {}},
            "timeline": [
                {"type": "kernel", "ext": {"geak": {"rebench": {"final_status": "failed"}}}},
                {"type": "kernel", "ext": {"geak": {"rebench": {"final_status": "validated"}}}},
            ],
        }
    )

    assert not any("DROPPED" in fact for fact in section.key_facts)


def test_final_report_still_flags_awaiting_geak_revalidation() -> None:
    facts, warnings = _render_final({"status": "awaiting_rebench", "self_reported_gain_pct": 12.5})

    assert any("AWAITING" in f for f in facts)
    assert warnings


@pytest.mark.asyncio
async def test_crash_recovery_tombstones_no_promote_result(coordinator, tmp_path) -> None:
    """An adjudicated no_promote result must not be recovered and re-enqueued."""
    c = coordinator
    st = c.shared_state
    _arm_kernel_to_sweep(st)
    geak_dir = tmp_path / "geak"
    geak_dir.mkdir()
    result = {
        "status": "ok",
        "final_throughput_tok_s": 116.0,
        "accepted_config": {"flags": "--foo", "env": ""},
    }
    (geak_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    st.geak_result = {**result, "revalidation_status": "no_promote"}
    st.geak_pending = {}

    c.phase_kernel._record_geak_kernel_journey = lambda _result: None
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._kernel_agent_tool._kernel_agent_tool_path",
        lambda _name: (_ for _ in ()).throw(RuntimeError("runner should not run")),
    )
    try:
        await c._run_geak_kernel_phase(from_phase="KERNEL")
    finally:
        monkeypatch.undo()

    queued = await c.tasks.queued()
    assert not [task for task in queued if task.params.get("geak_fallback")]


def _arm_settled_candidate(coordinator, tmp_path) -> dict:
    """A session whose only GEAK product was already settled as incomparable."""
    st = coordinator.shared_state
    _arm_kernel_to_sweep(st)
    st.benchmark_mode = "agentx"
    st.baseline_tput = 100.0
    settled = {
        "status": "ok",
        "final_throughput_tok_s": 116.0,
        "accepted_config": {"flags": "--foo", "env": ""},
    }
    geak_dir = tmp_path / "geak"
    geak_dir.mkdir(exist_ok=True)
    (geak_dir / "result.json").write_text(json.dumps(settled), encoding="utf-8")
    st.geak_result = {
        **settled,
        "returncode": 0,
        "revalidation_status": "fallback_failed",
        "revalidation_error_class": "incomparable",
        "revalidation_error": "geak_harness_unsupported_canonical_workload",
    }
    st.geak_pending = {}
    coordinator.phase_kernel._record_geak_kernel_journey = lambda _result: None
    return settled


async def _assert_settled_candidate_survived(coordinator, tmp_path) -> None:
    st = coordinator.shared_state
    queued = await coordinator.tasks.queued()
    assert not [t for t in queued if t.params.get("geak_fallback")]
    assert st.geak_result["revalidation_error_class"] == "incomparable"
    assert st.geak_pending.get("status") != "awaiting_rebench"
    # The runner's file is left for the next run to overwrite.
    assert (tmp_path / "geak" / "result.json").is_file()


def _stub_geak_runner_call(monkeypatch, tmp_path, outcome) -> list[str]:
    """Replace only the GEAK runner's own thread call; every other one runs.

    ``asyncio.to_thread`` is shared, so the phase's bus and file work travels
    through it too — dispatching on the runner closure keeps those real.

    Returns the list the interceptions are recorded in, so a test can prove the
    phase reached the runner rather than returning at an earlier gate.
    """
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._kernel_agent_tool._kernel_agent_tool_path",
        lambda _name: tmp_path / "geak_runner.py",
    )
    original_to_thread = asyncio.to_thread
    calls: list[str] = []

    async def _to_thread(func, /, *args, **kwargs):
        qualname = getattr(func, "__qualname__", "")
        if qualname.endswith("_run_geak_kernel_phase.<locals>._run"):
            calls.append(qualname)
            return outcome()
        return await original_to_thread(func, *args, **kwargs)

    monkeypatch.setattr("hyperloom.orchestrator.phases.kernel.asyncio.to_thread", _to_thread)
    return calls


@pytest.mark.asyncio
async def test_failed_runner_does_not_replay_the_settled_result(coordinator, tmp_path, monkeypatch) -> None:
    """A runner that exits nonzero over the old file shipped no new product."""
    c = coordinator
    _arm_settled_candidate(c, tmp_path)
    calls = _stub_geak_runner_call(
        monkeypatch,
        tmp_path,
        lambda: subprocess.CompletedProcess(["geak_runner.py"], 1, "", "runner failed"),
    )

    await c._run_geak_kernel_phase(from_phase="KERNEL")

    assert len(calls) == 1
    await _assert_settled_candidate_survived(c, tmp_path)


@pytest.mark.asyncio
async def test_runner_timeout_does_not_replay_the_settled_result(coordinator, tmp_path, monkeypatch) -> None:
    """The SIGTERM grace read finds the old file, not a flushed new win."""
    c = coordinator

    def _timed_out():
        raise subprocess.TimeoutExpired(["geak_runner.py"], 1)

    _arm_settled_candidate(c, tmp_path)
    calls = _stub_geak_runner_call(monkeypatch, tmp_path, _timed_out)

    await c._run_geak_kernel_phase(from_phase="KERNEL")

    assert len(calls) == 1
    await _assert_settled_candidate_survived(c, tmp_path)


_SETTLED_GEAK_RESULT = {
    "status": "ok",
    "final_throughput_tok_s": 110.0,
    "accepted_config": {"flags": "--settled", "env": ""},
    "final_overlay": "/tmp/geak-overlay",
    "eval_dir": "e2e_cycle0",
}


@pytest.mark.parametrize(
    "fresh",
    [
        {
            "status": "ok",
            "final_throughput_tok_s": 130.0,
            "accepted_config": {"flags": "--fresh", "env": ""},
        },
        {
            **_SETTLED_GEAK_RESULT,
            "final_throughput_tok_s": 132.0,
            "eval_dir": "e2e_cycle1",
        },
    ],
    ids=["new_config", "same_config_new_evidence"],
)
@pytest.mark.asyncio
async def test_crash_recovery_still_promotes_new_evidence(coordinator, tmp_path, fresh: dict) -> None:
    """A settled verdict tombstones its own candidate, not the next one.

    The crash window this recovery exists for is exactly the one where the
    runner wrote a fresh ``result.json`` and the handback never landed, so the
    persisted verdict describes the previous candidate. A rerun that lands on
    the same flags is still a different run, so config alone cannot say the
    verdict already covers what is on disk.
    """
    c = coordinator
    st = c.shared_state
    _arm_kernel_to_sweep(st)
    st.benchmark_mode = "agentx"
    st.baseline_tput = 100.0
    geak_dir = tmp_path / "geak"
    geak_dir.mkdir()
    (geak_dir / "result.json").write_text(json.dumps(fresh), encoding="utf-8")
    st.geak_result = {**_SETTLED_GEAK_RESULT, "returncode": 0, "revalidation_status": "no_promote"}
    st.geak_pending = {}

    c.phase_kernel._record_geak_kernel_journey = lambda _result: None
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._kernel_agent_tool._kernel_agent_tool_path",
        lambda _name: (_ for _ in ()).throw(RuntimeError("runner should not run")),
    )
    try:
        await c._run_geak_kernel_phase(from_phase="KERNEL")
    finally:
        monkeypatch.undo()

    queued = [t for t in await c.tasks.queued() if t.params.get("geak_fallback")]
    assert len(queued) == 1
    assert queued[0].params["grid"][0]["extra_args"] == fresh["accepted_config"]["flags"]
    assert st.geak_result["final_throughput_tok_s"] == pytest.approx(fresh["final_throughput_tok_s"])


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["geak", "resume"])
@pytest.mark.parametrize(
    ("baseline_runtime", "legacy_timeout_override"),
    [(4140.0, 9000), (4140.0, 0), (0.0, 0)],
    ids=["retired_override", "measured_baseline", "default"],
)
@pytest.mark.parametrize("session_remaining_sec", [None, 20000.0], ids=["unbounded", "bounded"])
async def test_internal_stack_rebench_passes_runtime_budget_to_executor(
    coordinator,
    tmp_path,
    monkeypatch,
    source,
    baseline_runtime,
    legacy_timeout_override,
    session_remaining_sec,
) -> None:
    import time

    from hyperloom.orchestrator.actions.executors import ExploreExecutor

    baseline = tmp_path / "baseline.yaml"
    baseline.write_text(
        "benchmark:\n"
        "  framework: sglang\n"
        "  model: /models/test\n"
        "  run_mode: local\n"
        "  benchmark_script: sglang_mi300x.sh\n"
        "  envs: {TP: 1, CONC: 8, ISL: 256, OSL: 256}\n",
        encoding="utf-8",
    )
    state = coordinator.shared_state
    state.baseline_config_path = str(baseline)
    state.baseline_tput = 100.0
    state.baseline_runtime_sec = baseline_runtime
    state.explore_variant_timeout_sec_override = legacy_timeout_override
    state.explore_variant_timeout_safety_margin = 0.5
    state.explore_overtime_kill_ratio = 1.5
    state.baseline_double_run = False
    monkeypatch.setattr(state, "session_budget_usable_sec", lambda **_kwargs: session_remaining_sec)
    if source == "geak":
        state.geak_result = {"status": "ok", "accepted_config": {"flags": "--mem-fraction-static 0.9"}}
        params = coordinator._geak_rebench_params(reason="runtime_budget_regression")
        task = await coordinator.tasks.create(kind="explore", params=params, idempotency_key="geak-revalidate-c0")
    else:
        state.current_best = {"extra_server_args": "--mem-fraction-static 0.9"}
        enqueued = await coordinator._enqueue_internal_stack_rebench(reason="runtime_budget_regression")
        task = await coordinator.tasks.get(str(enqueued["task_id"]))
    calls = []

    async def capture_grid(**kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr("hyperloom.orchestrator.actions.executors.explore.run_grid", capture_grid)
    monkeypatch.setattr("hyperloom.orchestrator.actions.executors.explore.maybe_serving_lease", lambda **_kwargs: None)
    executor = ExploreExecutor(session_dir=coordinator.session_dir)
    started = time.monotonic()
    await executor._run_explore(SimpleNamespace(task=task, extra={"shared_state": state}))
    finished = time.monotonic()

    assert len(calls) == 1
    assert calls[0]["variant_expected_sec"] == (baseline_runtime or None)
    if session_remaining_sec is None:
        assert calls[0]["session_deadline_sec"] is None
    else:
        assert started + session_remaining_sec <= calls[0]["session_deadline_sec"] <= finished + session_remaining_sec
    assert task.params.get("baseline_runtime_sec", 0.0) == baseline_runtime
    for retired_param in ("variant_timeout_sec", "soft_deadline_sec", "overtime_kill_ratio"):
        assert retired_param not in calls[0]
        assert retired_param not in task.params


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["geak", "resume"])
@pytest.mark.parametrize("enablement_revalidation", [False, True])
async def test_internal_stack_rebench_preserves_baseline_script(
    coordinator, tmp_path, monkeypatch, source, enablement_revalidation
) -> None:
    import yaml

    from hyperloom.orchestrator.actions.executors import ExploreExecutor
    from hyperloom.orchestrator.actions.executors._grid_runner import _build_variant_yaml

    baseline = tmp_path / "baseline.yaml"
    baseline.write_text(
        "benchmark:\n  framework: sglang\n  model: /models/test\n  run_mode: local\n"
        "  benchmark_script: sglang_custom.sh\n  envs: {TP: 1, CONC: 8, ISL: 256, OSL: 256}\n",
        encoding="utf-8",
    )
    state = coordinator.shared_state
    state.baseline_tput = 0.0
    state.baseline_double_run = True
    for name, tput in [("sglang_custom.sh", 100.0), ("rejected_script.sh", 90.0)]:
        task = await coordinator.tasks.create(kind="baseline", params={"benchmark_script": name}, idempotency_key=name)
        await coordinator._promote_to_shared_state(
            "baseline", {"output_throughput": tput, "materialized_config": str(baseline)}, task=task
        )
    assert state.baseline_tput == 100.0
    assert state.last_baseline["extras"]["fingerprint"]["benchmark_script"] == "rejected_script.sh"
    if enablement_revalidation:
        task = await coordinator.tasks.create(
            kind="baseline",
            params={"reason": "enablement_eval_revalidation", "benchmark_script": "sglang_custom.sh"},
            idempotency_key="revalidate-baseline",
        )
        await coordinator._promote_to_shared_state(
            "baseline", {"output_throughput": 105.0, "materialized_config": str(baseline)}, task=task
        )
    state.save(coordinator.session_dir)
    coordinator.shared_state = state = type(state).load_or_init(coordinator.session_dir)
    assert state.baseline_benchmark_script == "sglang_custom.sh"
    if source == "geak":
        state.geak_result = {"status": "ok", "accepted_config": {"flags": "--mem-fraction-static 0.9"}}
    else:
        state.current_best = {"extra_server_args": "--mem-fraction-static 0.9"}
    monkeypatch.setenv("GPU_TYPE", "mi355x")
    if source == "geak":
        params = coordinator._geak_rebench_params(reason="script_regression")
        task = await coordinator.tasks.create(kind="explore", params=params, idempotency_key="geak-revalidate-c0")
    else:
        enqueued = await coordinator._enqueue_internal_stack_rebench(reason="script_regression")
        task = await coordinator.tasks.get(str(enqueued["task_id"]))
    calls = []

    async def capture_grid(**kwargs):
        output = tmp_path / f"variant_{len(calls)}"
        output.mkdir()
        variant_path = _build_variant_yaml(
            kwargs["base_yaml_path"],
            "",
            kwargs["grid"][0],
            output_subdir=output,
            gpu_type=kwargs["gpu_type"],
            benchmark_script=kwargs["benchmark_script"],
        )
        calls.append((kwargs, yaml.safe_load(variant_path.read_text())["benchmark"]))
        return []

    monkeypatch.setattr("hyperloom.orchestrator.actions.executors.explore.run_grid", capture_grid)
    monkeypatch.setattr("hyperloom.orchestrator.actions.executors.explore.maybe_serving_lease", lambda **_kwargs: None)
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.explore.teardown_lifecycle_server", lambda **_kwargs: None
    )
    await ExploreExecutor(session_dir=coordinator.session_dir)._run_explore(
        SimpleNamespace(task=task, extra={"shared_state": state})
    )

    assert len(calls) == 1
    assert calls[0][1]["benchmark_script"] == "sglang_custom.sh"
    assert calls[0][1]["runner_type"] == "mi355x"
    assert calls[0][0]["server_lifecycle"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["synthetic", "agentx"])
@pytest.mark.parametrize("settled", [False, True])
async def test_invalid_handoff_configuration_never_launches_geak(coordinator, monkeypatch, mode, settled):
    from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import ROUTE_GEAK, make_kernel_recorder
    from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events

    def invalid_spec():
        raise ValueError("unserializable launch configuration")

    def must_not_launch(*_args, **_kwargs):
        pytest.fail("Invalid launch configuration must stop before runner invocation")

    monkeypatch.setattr(coordinator, "build_env_spec", invalid_spec)
    monkeypatch.setattr("hyperloom.orchestrator.phases.kernel.subprocess.Popen", must_not_launch)
    coordinator.shared_state.benchmark_mode = mode
    previous = {"status": "ok", "revalidation_status": "no_promote", "accepted_config": {"flags": "--old"}}
    if settled:
        coordinator.shared_state.geak_result = dict(previous)
    recorder = make_kernel_recorder(macro_cycle=0, route=ROUTE_GEAK)
    assert recorder is not None
    recorder.begin()
    coordinator._kernel_timeline_recorder = recorder
    await coordinator._run_geak_kernel_phase(from_phase="EXPLORE")
    if settled:
        assert coordinator.shared_state.geak_result == previous
    else:
        assert coordinator.shared_state.geak_result["error_class"] == "invalid_env_spec"
    event = next(row for row in read_timeline_events(coordinator.session_dir) if row["type"] == "kernel")
    assert event["status"] == "failed"
    assert event["ext"]["outcome"]["error_class"] == "invalid_env_spec"
    assert not await coordinator.tasks.queued()


@pytest.mark.asyncio
@pytest.mark.parametrize("config", [{"env_map": None}, {"unset_envs": ["PYTHONPATH"]}])
async def test_invalid_config_with_real_artifact_is_rejected_before_rebench(coordinator, config):
    coordinator.shared_state.geak_result = {
        "status": "ok",
        "accepted_config": config,
        "accepted_kernels": ["real_kernel"],
    }
    result = coordinator._geak_rebench_params(reason="invalid_config")
    assert result == {"skipped": True, "reason": "geak_invalid_config"}
    assert coordinator.shared_state.geak_result["revalidation_status"] == "no_promote"
    assert "accepted_config" in coordinator.shared_state.geak_result["revalidation_error"]
    assert not await coordinator.tasks.queued()
