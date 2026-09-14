# Copyright Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Unit tests for the enablement_launch_only bench-only mode of IntegratePatchExecutor."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor
from hyperloom.orchestrator.bringup import observe_bringup, write_boot_observation
from hyperloom.orchestrator.rehearsal import boot_log_for
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.state.task_registry import Task


def _make_ctx(task_id: str, params: dict[str, Any], extra: dict | None = None) -> RunnerContext:
    task = Task(
        task_id=task_id,
        kind="integrate_patch",
        state="queued",
        params=params,
        idempotency_key=task_id,
        requires_lanes=tuple(),
    )
    return RunnerContext(task=task, lease=None, extra=extra or {})


def _runtime_override() -> dict[str, Any]:
    return {"path_prefix": "/tmp/venv/bin", "framework_python": "/tmp/venv/bin/python"}


def _params_base(session_dir: Path) -> dict[str, Any]:
    return {
        "enablement": True,
        "enablement_launch_only": True,
        "runtime_override": _runtime_override(),
        "config_path": str(session_dir / "bench.yaml"),
        "source": "coordinator_internal",
    }


def _write_minimal_config(path: Path) -> None:
    path.write_text("benchmark:\n  model: /tmp/m\n", encoding="utf-8")


def _booted_observation(session: Path) -> str:
    """Record the observation a bench that came up and served would leave.

    The gate reads the boot verdict off this artifact rather than off a
    throughput, so a bench stub that records none has not booted anything.
    """
    slot = session / "round"
    slot.mkdir(parents=True, exist_ok=True)
    verdict = observe_bringup(server_log=boot_log_for(None), server_elapsed_sec=5.0, session_dir=session)
    return write_boot_observation(verdict.observation, session_dir=session, output_dir=slot, attempt=0)


# ---------------------------------------------------------------------------
# _stage_resolve: launch-only bypasses specialist/Critic checks


@pytest.mark.asyncio
async def test_launch_only_skips_missing_specialist_task_id(tmp_path):
    """No specialist_task_id needed in launch-only mode."""
    session = tmp_path / "s"
    session.mkdir()
    _write_minimal_config(session / "bench.yaml")
    ex = IntegratePatchExecutor(session_dir=session)
    params = _params_base(session)

    bench_result = {
        "output_throughput": 100.0,
        "status": "succeeded",
        "boot_observation_path": _booted_observation(session),
    }
    gate_evidence = {"enablement_accuracy": None, "timed_out": False}

    with patch.object(ex, "_bench_patch", new=AsyncMock(return_value=(bench_result, gate_evidence))):
        res = await ex(_make_ctx("probe-1", params))

    assert res["status"] == "kept"
    assert res.get("enablement") is True


@pytest.mark.asyncio
async def test_launch_only_skips_critic_gate(tmp_path):
    """Critic verdict is not consulted in launch-only mode."""
    session = tmp_path / "s"
    session.mkdir()
    _write_minimal_config(session / "bench.yaml")
    ex = IntegratePatchExecutor(session_dir=session)
    params = _params_base(session)

    class _RejectAll:
        def get_specialist_patch_verdict(self, tid):
            return "reject"

    bench_result = {
        "output_throughput": 50.0,
        "status": "succeeded",
        "boot_observation_path": _booted_observation(session),
    }
    gate_evidence = {"enablement_accuracy": None, "timed_out": False}

    with patch.object(ex, "_bench_patch", new=AsyncMock(return_value=(bench_result, gate_evidence))):
        res = await ex(_make_ctx("probe-2", params, extra={"shared_state": _RejectAll()}))

    assert res["status"] == "kept"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("patches", ["candidate.patch"]),
        ("localization_candidate", {"kind": "pr_backport"}),
        ("runtime_candidate", {"kind": "runtime_candidate"}),
        ("artifacts", [{"source": "x", "target": "y"}]),
        ("config_changes", {"EXTRA_VLLM_ARGS": "--unsafe"}),
        ("enablement_setup_commands", ["pip install package"]),
    ],
)
async def test_launch_only_rejects_mutation_fields(tmp_path, field, value):
    session = tmp_path / "s"
    session.mkdir()
    _write_minimal_config(session / "bench.yaml")
    ex = IntegratePatchExecutor(session_dir=session)
    params = {**_params_base(session), field: value}

    res = await ex(_make_ctx("probe-mutation", params))

    assert res["status"] == "failed"
    assert res["error_class"] == "launch_only_mutation_forbidden"
    assert field in res["error"]


# _stage_apply: launch-only falls through to bench when no patches exist


@pytest.mark.asyncio
async def test_launch_only_does_not_return_no_patches(tmp_path):
    """enablement_launch_only suppresses the no_patches early return."""
    session = tmp_path / "s"
    session.mkdir()
    _write_minimal_config(session / "bench.yaml")
    ex = IntegratePatchExecutor(session_dir=session)
    params = _params_base(session)

    bench_result = {"output_throughput": 0.0, "status": "failed", "error": "cuda oom"}
    gate_evidence = {"enablement_accuracy": None, "timed_out": False}

    with patch.object(ex, "_bench_patch", new=AsyncMock(return_value=(bench_result, gate_evidence))):
        res = await ex(_make_ctx("probe-3", params))

    assert res["status"] != "no_patches", "launch-only must never return no_patches"
    assert res.get("enablement") is True


# Gate routing: kept / reverted / advanced


@pytest.mark.asyncio
async def test_launch_only_boot_success_returns_kept(tmp_path):
    session = tmp_path / "s"
    session.mkdir()
    _write_minimal_config(session / "bench.yaml")
    ex = IntegratePatchExecutor(session_dir=session)
    params = _params_base(session)

    bench_result = {
        "output_throughput": 200.0,
        "status": "succeeded",
        "boot_observation_path": _booted_observation(session),
    }
    gate_evidence = {"enablement_accuracy": None, "timed_out": False}

    with patch.object(ex, "_bench_patch", new=AsyncMock(return_value=(bench_result, gate_evidence))):
        res = await ex(_make_ctx("probe-4", params))

    assert res["status"] == "kept"
    assert res["runnable"] is True
    assert res["patches_applied"] == []


@pytest.mark.asyncio
async def test_launch_only_boot_fail_returns_reverted_or_advanced(tmp_path):
    """A non-booting result returns reverted (no progress) or advanced (new gap)."""
    session = tmp_path / "s"
    session.mkdir()
    _write_minimal_config(session / "bench.yaml")
    ex = IntegratePatchExecutor(session_dir=session)
    params = _params_base(session)

    bench_result = {"output_throughput": 0.0, "status": "failed", "error": "ImportError: no module"}
    gate_evidence = {"enablement_accuracy": None, "timed_out": False}

    with patch.object(ex, "_bench_patch", new=AsyncMock(return_value=(bench_result, gate_evidence))):
        res = await ex(_make_ctx("probe-5", params))

    assert res["status"] in ("reverted", "advanced")
    assert res.get("enablement") is True
    assert res["runnable"] is False


def _persist_observation(session: Path, slot: str, log_text: str) -> str:
    """Observe ``log_text`` as a server log and persist it the way a round does."""
    from hyperloom.orchestrator.bringup import observe_bringup, write_boot_observation

    out = session / slot
    out.mkdir(parents=True, exist_ok=True)
    verdict = observe_bringup(server_log=log_text, server_elapsed_sec=3.0, session_dir=session)
    return write_boot_observation(verdict.observation, session_dir=session, output_dir=out, attempt=0)


@pytest.mark.asyncio
async def test_launch_only_advances_when_the_boot_climbs_the_ladder(tmp_path):
    """A deeper wall in the persisted after-observation produces 'advanced'."""
    session = tmp_path / "s"
    session.mkdir()
    _write_minimal_config(session / "bench.yaml")
    ex = IntegratePatchExecutor(session_dir=session)

    before_path = _persist_observation(
        session,
        "before",
        'Traceback (most recent call last):\n  File "/x/ops.py", line 4, in init\n'
        "ImportError: cannot import name '_C' from 'vllm'\n",
    )
    after_path = _persist_observation(
        session,
        "after",
        'Traceback (most recent call last):\n  File "/x/attention.py", line 51, in forward\n'
        "NotImplementedError: paged attention v2 has no ROCm path yet\n",
    )
    params = {**_params_base(session), "enablement_before_observation_path": before_path}

    bench_result = {
        "output_throughput": 0.0,
        "status": "failed",
        "error": "wrapper: benchmark subprocess exited 1",
        "boot_observation_path": after_path,
    }
    gate_evidence = {"enablement_accuracy": None, "timed_out": False}

    with patch.object(ex, "_bench_patch", new=AsyncMock(return_value=(bench_result, gate_evidence))):
        res = await ex(_make_ctx("probe-6", params))

    assert res["status"] == "advanced"
    # Both halves are recorded, and neither degraded.
    assert res["before_observation"]["stage_failed"] == "IMPORT"
    assert res["after_observation"]["stage_failed"] == "ENGINE_INIT"
    assert res["before_observation"]["failure_digest"] != res["after_observation"]["failure_digest"]
    assert res["before_observation_degraded"] == ""
    assert res["after_observation_degraded"] == ""
    # The next round's before half.
    assert res["enablement_observation_path"] == after_path


@pytest.mark.asyncio
async def test_launch_only_does_not_advance_on_a_shallower_wall(tmp_path):
    """A boot that got less far is not progress, however different its failure is.

    The digest covers the failed stage, so a regression always carries a wall
    the session has not seen. A gate that asks only "did the failure change"
    therefore reads every regression as forward progress -- which is how a
    patch that broke the import gets kept as a base for the next round.
    """
    session = tmp_path / "s"
    session.mkdir()
    _write_minimal_config(session / "bench.yaml")
    ex = IntegratePatchExecutor(session_dir=session)

    before_path = _persist_observation(
        session,
        "before",
        'Traceback (most recent call last):\n  File "/x/loader.py", line 9, in load\n'
        "KeyError: 'model.layers.0.mlp.gate_up_proj.weight'\n",
    )
    after_path = _persist_observation(
        session,
        "after",
        'Traceback (most recent call last):\n  File "/x/ops.py", line 4, in init\n'
        "ImportError: cannot import name '_C' from 'vllm'\n",
    )
    params = {**_params_base(session), "enablement_before_observation_path": before_path}

    bench_result = {
        "output_throughput": 0.0,
        "status": "failed",
        "error": "wrapper: benchmark subprocess exited 1",
        "boot_observation_path": after_path,
    }
    gate_evidence = {"enablement_accuracy": None, "timed_out": False}

    with patch.object(ex, "_bench_patch", new=AsyncMock(return_value=(bench_result, gate_evidence))):
        res = await ex(_make_ctx("probe-6c", params))

    assert res["status"] == "reverted"
    assert res["before_observation"]["failure_digest"] != res["after_observation"]["failure_digest"]


@pytest.mark.asyncio
async def test_launch_only_names_a_missing_observation_instead_of_reclassifying(tmp_path):
    """With no persisted observation, the verdict says so rather than reading the wrapper."""
    session = tmp_path / "s"
    session.mkdir()
    _write_minimal_config(session / "bench.yaml")
    ex = IntegratePatchExecutor(session_dir=session)
    params = {**_params_base(session), "enablement_before_observation_path": str(session / "gone.json")}

    bench_result = {"output_throughput": 0.0, "status": "failed", "error": "ImportError: vllm._C not found"}
    gate_evidence = {"enablement_accuracy": None, "timed_out": False}

    with patch.object(ex, "_bench_patch", new=AsyncMock(return_value=(bench_result, gate_evidence))):
        res = await ex(_make_ctx("probe-6b", params))

    assert res["status"] == "reverted"
    assert res["before_observation_degraded"] == "observation_unreadable"
    assert res["after_observation_degraded"] == "no_observation_path"
    assert res["before_observation"] is None
    assert res["after_observation"] is None


@pytest.mark.asyncio
async def test_launch_only_runtime_override_passed_to_bench(tmp_path):
    """runtime_override from params is visible inside _bench_patch params."""
    session = tmp_path / "s"
    session.mkdir()
    _write_minimal_config(session / "bench.yaml")
    ex = IntegratePatchExecutor(session_dir=session)
    params = _params_base(session)

    captured: list[dict] = []

    async def _capture_bench(self_inner=None, **kwargs):
        captured.append(dict(kwargs.get("params", {})))
        return {"output_throughput": 50.0, "status": "succeeded"}, {"enablement_accuracy": None, "timed_out": False}

    with patch.object(IntegratePatchExecutor, "_bench_patch", new=_capture_bench):
        await ex(_make_ctx("probe-7", params))

    assert captured, "_bench_patch was not called"
    bench_params = captured[0]
    assert bench_params.get("runtime_override") == _runtime_override()


# ---------------------------------------------------------------------------
# A launch probe inherits what earlier enablement rounds established.
#
# ``lane._rearm_on_advanced`` accumulates every advance into
# ``accepted_config``; a launch that does not read it back walks into a wall an
# earlier round already cleared. Observed live: build launch probe ac82fc2b
# booted with an empty EXTRA_VLLM_ARGS and died on "DeepseekV4 only supports
# fp8 kv-cache format for now, got auto" while accepted_config held
# ``--kv-cache-dtype fp8``. This probe is emitted by
# ``EnablementBuild._enqueue_build_launch_probe``, which never passes through
# the specialist autosubmit bridge, so the inheritance has to live in the
# executor every path crosses.


class _Established:
    """A SharedState stub carrying only what the inheritance reads."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.enablement = SimpleNamespace(accepted_config=dict(cfg))

    def get_specialist_patch_verdict(self, tid):  # never consulted on this lane
        return "approve"


async def _capture_launch(ex, params, ctx_extra):
    """Run the executor and return what it handed the bench."""
    captured: dict[str, Any] = {}

    async def _spy(**kwargs):
        captured.update(kwargs)
        return (
            {
                "output_throughput": 100.0,
                "status": "succeeded",
                "boot_observation_path": kwargs["params"].get("_obs_path", ""),
            },
            {"enablement_accuracy": None, "timed_out": False},
        )

    with patch.object(ex, "_bench_patch", new=AsyncMock(side_effect=_spy)):
        res = await ex(_make_ctx("probe-inherit", params, extra=ctx_extra))
    return res, captured


@pytest.mark.asyncio
async def test_launch_probe_inherits_the_flag_earlier_rounds_established(tmp_path):
    """The ac82fc2b regression: a probe that restated nothing still launches with it."""
    session = tmp_path / "s"
    session.mkdir()
    _write_minimal_config(session / "bench.yaml")
    ex = IntegratePatchExecutor(session_dir=session)
    params = _params_base(session)
    params["_obs_path"] = _booted_observation(session)

    state = _Established(
        {
            "extra_server_args": "--kv-cache-dtype fp8",
            "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
            "args_mode": "append",
        }
    )
    res, captured = await _capture_launch(ex, params, {"shared_state": state})

    assert captured["extra_server_args_applied"] == "--kv-cache-dtype fp8"
    assert captured["extra_envs_applied"]["VLLM_ROCM_USE_AITER"] == "1"
    # The launch-only control flow is untouched: it still benches rather than
    # reporting no_patches.
    assert res["status"] != "no_patches"


@pytest.mark.asyncio
async def test_launch_probe_with_no_state_inherits_nothing(tmp_path):
    """The defensive read: a context carrying no SharedState must not abort the round."""
    session = tmp_path / "s"
    session.mkdir()
    _write_minimal_config(session / "bench.yaml")
    ex = IntegratePatchExecutor(session_dir=session)
    params = _params_base(session)
    params["_obs_path"] = _booted_observation(session)

    _res, captured = await _capture_launch(ex, params, {})
    assert captured["extra_server_args_applied"] == ""
    assert captured["extra_envs_applied"] == {}


@pytest.mark.asyncio
async def test_an_optimization_probe_inherits_nothing(tmp_path):
    """The rule is the enablement lane's; nothing leaks into an optimization round."""
    session = tmp_path / "s"
    session.mkdir()
    _write_minimal_config(session / "bench.yaml")
    ex = IntegratePatchExecutor(session_dir=session)
    params = _params_base(session)
    params["_obs_path"] = _booted_observation(session)
    params["enablement"] = False

    state = _Established({"extra_server_args": "--kv-cache-dtype fp8"})
    _res, captured = await _capture_launch(ex, params, {"shared_state": state})
    assert captured["extra_server_args_applied"] == ""


@pytest.mark.asyncio
async def test_an_unparseable_round_arg_still_keeps_the_inherited_flag(tmp_path):
    """A quoted value with whitespace must not cost the round its inherited flags.

    The deduper cannot parse such a string, and provenance cannot stand in for
    it: only the autosubmit bridge pre-merges the inheritance, while this
    executor also serves the build probe, the framework-config bridge, authored
    proposals and re-queued rows, any of which may carry args of their own.
    """
    session = tmp_path / "s"
    session.mkdir()
    _write_minimal_config(session / "bench.yaml")
    ex = IntegratePatchExecutor(session_dir=session)
    params = _params_base(session)
    params["_obs_path"] = _booted_observation(session)
    # Not bridge-shaped: its own arg, never merged with accepted_config, and
    # unparseable because the value carries whitespace.
    params["extra_server_args"] = '--tool-call-parser "my parser"'

    state = _Established({"extra_server_args": "--kv-cache-dtype fp8"})
    _res, captured = await _capture_launch(ex, params, {"shared_state": state})

    args = captured["extra_server_args_applied"]
    assert "--kv-cache-dtype fp8" in args, args
    assert "--tool-call-parser" in args, args
    assert args.count("--kv-cache-dtype") == 1, args


@pytest.mark.asyncio
async def test_an_unparseable_round_arg_that_restates_the_flag_does_not_double_it(tmp_path):
    """The bridge-shaped case of the same input: already inherited, so nothing is added."""
    session = tmp_path / "s"
    session.mkdir()
    _write_minimal_config(session / "bench.yaml")
    ex = IntegratePatchExecutor(session_dir=session)
    params = _params_base(session)
    params["_obs_path"] = _booted_observation(session)
    params["extra_server_args"] = '--kv-cache-dtype fp8 --tool-call-parser "my parser"'

    state = _Established({"extra_server_args": "--kv-cache-dtype fp8"})
    _res, captured = await _capture_launch(ex, params, {"shared_state": state})

    args = captured["extra_server_args_applied"]
    assert args.count("--kv-cache-dtype") == 1, args


@pytest.mark.asyncio
async def test_a_longer_flag_sharing_a_prefix_does_not_satisfy_the_inherited_one(tmp_path):
    """Option names are compared exactly: --foo is not satisfied by --foo-bar."""
    session = tmp_path / "s"
    session.mkdir()
    _write_minimal_config(session / "bench.yaml")
    ex = IntegratePatchExecutor(session_dir=session)
    params = _params_base(session)
    params["_obs_path"] = _booted_observation(session)
    # Unparseable (quoted whitespace) and carrying a flag whose name merely
    # starts with the inherited one.
    params["extra_server_args"] = '--kv-cache-dtype-override auto --tool-call-parser "my parser"'

    state = _Established({"extra_server_args": "--kv-cache-dtype fp8"})
    _res, captured = await _capture_launch(ex, params, {"shared_state": state})

    args = captured["extra_server_args_applied"]
    assert "--kv-cache-dtype fp8" in args, args
    assert "--kv-cache-dtype-override auto" in args, args


@pytest.mark.asyncio
async def test_the_inherited_name_inside_a_quoted_value_is_not_an_option(tmp_path):
    """A name that only appears inside a quoted value does not count as present."""
    session = tmp_path / "s"
    session.mkdir()
    _write_minimal_config(session / "bench.yaml")
    ex = IntegratePatchExecutor(session_dir=session)
    params = _params_base(session)
    params["_obs_path"] = _booted_observation(session)
    params["extra_server_args"] = '--tool-call-parser "uses --kv-cache-dtype internally"'

    state = _Established({"extra_server_args": "--kv-cache-dtype fp8"})
    _res, captured = await _capture_launch(ex, params, {"shared_state": state})

    args = captured["extra_server_args_applied"]
    assert args.startswith("--kv-cache-dtype fp8"), args


@pytest.mark.asyncio
async def test_an_inherited_quoted_value_survives_the_fallback_intact(tmp_path):
    """The inherited side may itself carry whitespace inside a quoted value.

    Splitting it on spaces would keep only the first word and emit a malformed
    option, so the fallback parses it quote-aware and re-serialises what it adds.
    """
    session = tmp_path / "s"
    session.mkdir()
    _write_minimal_config(session / "bench.yaml")
    ex = IntegratePatchExecutor(session_dir=session)
    params = _params_base(session)
    params["_obs_path"] = _booted_observation(session)
    # Unparseable, and it does NOT restate the inherited option.
    params["extra_server_args"] = '--reasoning-parser "some parser"'

    state = _Established({"extra_server_args": '--tool-call-parser "my parser" --kv-cache-dtype fp8'})
    _res, captured = await _capture_launch(ex, params, {"shared_state": state})

    args = captured["extra_server_args_applied"]
    # The inherited value must come back whole, not truncated to its first word.
    assert "parser" in args and '"my' not in args, args
    import shlex as _shlex

    tokens = _shlex.split(args)
    assert "my parser" in tokens, tokens
    assert "--tool-call-parser" in tokens, tokens
    assert "--kv-cache-dtype" in tokens and "fp8" in tokens, tokens
    assert "some parser" in tokens, tokens
