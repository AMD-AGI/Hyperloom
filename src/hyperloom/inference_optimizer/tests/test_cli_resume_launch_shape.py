# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Launch-shape persistence across ``--resume``."""

from __future__ import annotations

import argparse
import json
import os

from hyperloom.inference_optimizer.cli import _export_operator_launch_shape
from hyperloom.inference_optimizer.cli.bootstrap import parse_operator_extra_env
from hyperloom.orchestrator.state.shared_state import SharedState


def _ns(**kw) -> argparse.Namespace:
    return argparse.Namespace(**kw)


def _knob_args(**kw) -> argparse.Namespace:
    base = dict(isl=None, osl=None, conc=None, tp=None, ep=None, precision=None, model=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_pin_fills_a_workload_knob_the_flags_left_unset(monkeypatch):
    """A pinned knob is resolved on ``args``, which is what every later env projection writes.

    Resolving it here rather than letting the pin's own export survive is what keeps one answer: the projections at
    the end of the fresh branch write ``args`` unconditionally and would otherwise overwrite the pin.
    """
    from hyperloom.inference_optimizer.cli import _resolve_workload_knobs

    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", json.dumps({"ISL": "4096", "OSL": "512"}))
    args = _knob_args()

    _resolve_workload_knobs(args)

    assert (args.isl, args.osl) == (4096, 512)


def test_an_explicit_flag_outranks_a_pin_of_the_same_knob(monkeypatch):
    """`--isl` wins over a pinned `ISL`, the ladder `_resolve_run_max_model_len_inner` already uses for its own knob."""
    from hyperloom.inference_optimizer.cli import _resolve_workload_knobs

    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", json.dumps({"ISL": "4096"}))
    args = _knob_args(isl=2048)

    _resolve_workload_knobs(args)

    assert args.isl == 2048


def test_pin_outranks_persisted_state_for_a_workload_knob(monkeypatch):
    """A pin sits above the resumed session's recorded value, below an explicit flag."""
    from hyperloom.inference_optimizer.cli import _resolve_workload_knobs

    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", json.dumps({"CONC": "16"}))
    args = _knob_args()

    _resolve_workload_knobs(args, state=_ns(conc=4, isl=0, osl=0, tp=0, ep=0, precision=""))

    assert args.conc == 16


def test_a_ladder_resolved_pin_is_not_exported_but_stays_in_the_blob(monkeypatch):
    """A knob with its own ladder enters through it; exporting it too would beat an explicit flag on one branch.

    The blob keeps it, because the ladder is what reads it and ``state.json`` is what a later resume restores from.
    """
    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", "")
    monkeypatch.setenv("ISL", "")
    monkeypatch.setenv("HYPERLOOM_AGENTIC_BACKEND", "")

    _export_operator_launch_shape(server_args="", extra_env={"ISL": "4096", "HYPERLOOM_AGENTIC_BACKEND": "mlperf"})

    assert os.environ["ISL"] == ""
    assert os.environ["HYPERLOOM_AGENTIC_BACKEND"] == "mlperf"
    assert json.loads(os.environ["INFERENCE_OPTIMIZER_EXTRA_ENV"]) == {
        "ISL": "4096",
        "HYPERLOOM_AGENTIC_BACKEND": "mlperf",
    }


def test_the_unset_loop_leaves_a_ladder_resolved_projection_alone(monkeypatch):
    """The projection writes ISL; a later export must not clear it just because the blob also names it."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", json.dumps({"ISL": "4096"}))
    monkeypatch.setenv("ISL", "4096")

    _export_operator_launch_shape(server_args="", extra_env={})

    assert os.environ["ISL"] == "4096"


def test_a_pinned_tp_reaches_the_environment_on_the_fresh_branch(monkeypatch):
    """Mirrors the fresh branch: launch-shape export, ladder, then the TP/CONC/EP projection.

    The environment is what the server launches from while ``state.json`` records ``args``, so a split between the
    two runs one shape and reports another. The projection has to sit after the ladder for them to agree.
    """
    from hyperloom.inference_optimizer.cli import _export_workload_envs_for_optimize, _resolve_workload_knobs

    for name in ("TP", "EP", "CONC"):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", "")
    args = _knob_args()

    _export_operator_launch_shape(server_args="", extra_env={"TP": "4", "EP": "2", "CONC": "63"})
    _resolve_workload_knobs(args)
    _export_workload_envs_for_optimize(args, tp_resolved=int(args.tp), ep_resolved=int(args.ep))

    assert (args.tp, args.ep, args.conc) == (4, 2, 63)
    assert (os.environ["TP"], os.environ["EP"], os.environ["CONC"]) == ("4", "2", "63")


def test_every_tp_projection_sits_after_the_ladder_that_resolves_the_pin():
    """The ordering is the fix, and the test above cannot see it: it calls the two in the order it wants.

    ``_export_workload_envs_for_optimize`` is the only non-test writer of ``os.environ["TP"|"CONC"|"EP"]``. Called
    before ``_resolve_workload_knobs``, it publishes the flag-derived default over a pin and the run launches at a
    shape its own ``state.json`` does not record. Asserted against the source because that is where the defect lives.
    """
    import ast
    import inspect
    import textwrap

    from hyperloom.inference_optimizer.cli import _run_optimize

    tree = ast.parse(textwrap.dedent(inspect.getsource(_run_optimize)))
    calls = [
        (node.lineno, node.func.id)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("_resolve_workload_knobs", "_export_workload_envs_for_optimize")
    ]
    ladders = [line for line, name in calls if name == "_resolve_workload_knobs"]
    projections = [line for line, name in calls if name == "_export_workload_envs_for_optimize"]

    assert projections, "_run_optimize no longer projects TP/CONC/EP at all"
    assert ladders, "_run_optimize no longer resolves the workload knobs"
    assert min(projections) > min(ladders), (
        f"TP/CONC/EP projected at line {min(projections)} of _run_optimize, "
        f"before the ladder at line {min(ladders)} -- a pinned TP would not reach the environment"
    )


def test_a_repassed_max_model_len_pin_outranks_the_resumed_value():
    """A resume re-passing the pin to change MAX_MODEL_LEN gets the new value, not what the session recorded."""
    from hyperloom.inference_optimizer.cli import _resolve_resume_max_model_len

    state = _ns(max_model_len=32768)

    assert _resolve_resume_max_model_len(_knob_args(), {"MAX_MODEL_LEN": "65536"}, state) == 65536
    # An explicit flag still wins, and with neither the session's own value stands.
    assert _resolve_resume_max_model_len(_knob_args(max_model_len=8192), {"MAX_MODEL_LEN": "65536"}, state) == 8192
    assert _resolve_resume_max_model_len(_knob_args(), {}, state) == 32768
    assert _resolve_resume_max_model_len(_knob_args(), {"MAX_MODEL_LEN": "nope"}, state) == 32768


def test_a_malformed_pinned_knob_does_not_take_the_run_down(monkeypatch):
    """A pin that is not a positive integer falls through to the rest of the ladder rather than raising."""
    from hyperloom.inference_optimizer.cli import _resolve_workload_knobs
    from hyperloom.common.workload_defaults import DEFAULT_ISL

    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", json.dumps({"ISL": "not-a-number"}))
    args = _knob_args()

    _resolve_workload_knobs(args)

    assert args.isl == DEFAULT_ISL


def test_resume_restores_pins_before_the_agentx_staleness_guard(tmp_path, monkeypatch, request):
    """A session whose AgentX backend came from a pin resumes; the guard must not read an unpinned environment.

    The guard resolves the backend with a bare ``os.environ.get``, so restoring the persisted pins after it runs
    compares a session seeded as ``mlperf`` against an ambient ``aiperf`` and refuses the resume outright.
    """
    import asyncio

    import pytest

    from hyperloom.inference_optimizer.cli import _build_parser, _run_optimize
    from hyperloom.inference_optimizer.cli.bootstrap import AGENTX_MEASUREMENT_EPOCH

    cli_mod = "hyperloom.inference_optimizer.cli"

    # ``_run_optimize`` writes the environment directly -- SKIP_VARIANTS, PD_MODE, INFERENCE_OPTIMIZER_NODES and
    # more -- so monkeypatch has nothing recorded to undo for them. Driving the real entry point means restoring the
    # whole environment here, or a later test in the same worker grades on what this one left behind.
    saved_environ = dict(os.environ)
    request.addfinalizer(lambda: (os.environ.clear(), os.environ.update(saved_environ)))

    workspace = tmp_path / "sessions"
    resume_dir = workspace / "Qwen-Test" / "pinned-backend"
    model = tmp_path / "Qwen-Test"
    resume_dir.mkdir(parents=True, exist_ok=True)
    (resume_dir / "reports").mkdir(parents=True, exist_ok=True)
    SharedState(
        session_id="resume-pinned-backend",
        model_name=model.name,
        model_path=str(model),
        benchmark_mode="agentx",
        agentx_epoch=AGENTX_MEASUREMENT_EPOCH,
        agentx_backend="mlperf",
        operator_extra_env={"HYPERLOOM_AGENTIC_BACKEND": "mlperf"},
    ).save(resume_dir)
    (resume_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 4, "session_id": "resume-pinned-backend"}), encoding="utf-8"
    )

    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(resume_dir))
    monkeypatch.setenv("USER_DATA_PATH", str(workspace))
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    # The operator exports AgentX in the shell but not the backend: that came from --extra-env on the fresh launch.
    monkeypatch.delenv("HYPERLOOM_AGENTIC_BACKEND", raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", "")
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.setattr(
        f"{cli_mod}.clean_stale_aiter_locks",
        lambda: {"dir": "", "deleted": 0, "skipped_fresh": 0, "errors": 0},
    )
    monkeypatch.setattr(f"{cli_mod}._preflight", lambda args: ("", ""))

    def past_the_guard(*a, **kw):
        raise RuntimeError("resume cleared the staleness guard")

    # The first call after the guard: reaching it means the guard did not reject the session.
    monkeypatch.setattr(f"{cli_mod}.latency_budget_scope_error", past_the_guard)
    args = _build_parser().parse_args(["optimize", "--resume-from", str(resume_dir), "--critic-mock"])

    # A SystemExit instead means the guard compared the session's pinned backend against an unpinned environment.
    with pytest.raises(RuntimeError, match="resume cleared the staleness guard"):
        asyncio.run(_run_optimize(args))

    assert os.environ["HYPERLOOM_AGENTIC_BACKEND"] == "mlperf"


def test_parse_operator_extra_env_keeps_pairs_and_drops_junk():
    """``NAME=VALUE`` pins survive; entries without ``=`` or with a blank name do not."""
    args = _ns(extra_env=["SGLANG_USE_AITER=0", "EMPTY=", "novalue", "=blank"])
    assert parse_operator_extra_env(args) == {"SGLANG_USE_AITER": "0", "EMPTY": ""}


def test_parse_operator_extra_env_missing_attr_is_empty():
    """A namespace without the flag yields no pins rather than raising."""
    assert parse_operator_extra_env(_ns()) == {}


def test_export_operator_launch_shape_sets_env(monkeypatch):
    """Both handoff variables are projected for downstream in-process executors."""
    # setenv, not delenv: the helper writes os.environ directly, so monkeypatch has to have recorded the pre-test
    # value to undo the write on teardown.
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SERVER_ARGS", "")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", "")

    _export_operator_launch_shape(
        server_args="--max-num-seqs 512",
        extra_env={"SGLANG_USE_AITER": "0"},
    )

    assert os.environ["INFERENCE_OPTIMIZER_SERVER_ARGS"] == "--max-num-seqs 512"
    assert json.loads(os.environ["INFERENCE_OPTIMIZER_EXTRA_ENV"]) == {"SGLANG_USE_AITER": "0"}


def test_export_operator_launch_shape_clears_stale_values(monkeypatch):
    """Empty inputs clear the variables so a second session in the same shell can't inherit them."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SERVER_ARGS", "--stale")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", '{"STALE":"1"}')

    _export_operator_launch_shape(server_args="", extra_env={})

    assert "INFERENCE_OPTIMIZER_SERVER_ARGS" not in os.environ
    assert "INFERENCE_OPTIMIZER_EXTRA_ENV" not in os.environ


def test_export_operator_launch_shape_exports_each_pin(monkeypatch):
    """A pin reaches the process environment under its own name, not only inside the JSON blob.

    Every Hyperloom control variable is read with a bare ``os.environ.get``. A pin visible only as
    ``INFERENCE_OPTIMIZER_EXTRA_ENV`` is therefore invisible to all of them, which is how one reader can resolve a
    knob differently from the rest.
    """
    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", "")
    monkeypatch.setenv("HYPERLOOM_AGENTIC_BACKEND", "")

    _export_operator_launch_shape(server_args="", extra_env={"HYPERLOOM_AGENTIC_BACKEND": "mlperf"})

    assert os.environ["HYPERLOOM_AGENTIC_BACKEND"] == "mlperf"


def test_exported_pin_is_seen_by_the_bare_env_readers(monkeypatch):
    """The workload the switch selects and the axis the session seeds agree once the pin is global."""
    from hyperloom.common.agentx_workload import MLPERF_CLIENT_SCRIPT, agentx_client_script
    from hyperloom.common.perf_metric import GRADED_OUTPUT
    from hyperloom.inference_optimizer.cli.bootstrap import seed_grading

    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", "")
    monkeypatch.setenv("HYPERLOOM_AGENTIC_BACKEND", "")
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "")

    _export_operator_launch_shape(server_args="", extra_env={"HYPERLOOM_AGENTIC_BACKEND": "mlperf"})

    assert agentx_client_script() == MLPERF_CLIENT_SCRIPT
    # MLPerf publishes no per-request OSL/E2EL series, so grading it on interactivity would REVERT every round.
    assert seed_grading("sglang", "agentx")["objective"] == GRADED_OUTPUT


def test_export_operator_launch_shape_unsets_pins_dropped_on_resume(monkeypatch):
    """A resume that drops a pin must clear the name the previous launch exported."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", "")
    monkeypatch.setenv("STALE_PIN", "")
    monkeypatch.setenv("KEPT_PIN", "")

    _export_operator_launch_shape(server_args="", extra_env={"STALE_PIN": "1", "KEPT_PIN": "a"})
    _export_operator_launch_shape(server_args="", extra_env={"KEPT_PIN": "b"})

    assert "STALE_PIN" not in os.environ
    assert os.environ["KEPT_PIN"] == "b"


def test_launch_shape_survives_a_state_roundtrip():
    """The fields reach disk, which is what a resume reads them back from."""
    state = SharedState(
        session_id="s",
        operator_server_args="--max-num-seqs 512",
        operator_extra_env={"SGLANG_USE_AITER": "0"},
        nodes=4,
        warm_replay_enabled=False,
        warm_replay_min_confidence=0.55,
        bypass_scripts_dir="/scripts",
        framework_repo_path="/fw",
        benchmark_backend="bypass",
    )

    restored = SharedState.from_dict(state.to_dict())

    assert restored.operator_server_args == "--max-num-seqs 512"
    assert restored.operator_extra_env == {"SGLANG_USE_AITER": "0"}
    assert restored.nodes == 4
    assert restored.warm_replay_enabled is False
    assert restored.warm_replay_min_confidence == 0.55
    assert restored.bypass_scripts_dir == "/scripts"
    assert restored.framework_repo_path == "/fw"
    assert restored.benchmark_backend == "bypass"


def test_pre_existing_state_without_the_fields_loads_defaults():
    """Sessions created before these fields existed resume on the documented defaults, not a crash."""
    restored = SharedState.from_dict({"session_id": "old"})

    assert restored.operator_server_args == ""
    assert restored.operator_extra_env == {}
    assert restored.nodes == 1
    assert restored.warm_replay_enabled is True
    assert restored.warm_replay_min_confidence == 0.7
    assert restored.bypass_scripts_dir == ""
    assert restored.framework_repo_path == ""
    assert restored.benchmark_backend == ""


def test_restore_operator_paths_fills_env_from_state(monkeypatch):
    from hyperloom.inference_optimizer.cli import _restore_operator_supplied_paths_from_state

    monkeypatch.setenv("FRAMEWORK_REPO_PATH", "")
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", "")
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "")
    state = SharedState(
        session_id="s",
        framework_repo_path="/archived/fw",
        bypass_scripts_dir="/archived/scripts",
        benchmark_backend="bypass",
    )
    _restore_operator_supplied_paths_from_state(_ns(framework_path=None, benchmark_scripts_dir=None), state)
    assert os.environ["FRAMEWORK_REPO_PATH"] == "/archived/fw"
    assert os.environ["HYPERLOOM_BYPASS_SCRIPTS_DIR"] == "/archived/scripts"
    assert os.environ["HYPERLOOM_BENCHMARK_BACKEND"] == "bypass"


def test_restore_operator_paths_leaves_env_when_cli_repasses(monkeypatch):
    from hyperloom.inference_optimizer.cli import _restore_operator_supplied_paths_from_state

    monkeypatch.setenv("FRAMEWORK_REPO_PATH", "")
    monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", "")
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "")
    state = SharedState(
        session_id="s",
        framework_repo_path="/archived/fw",
        bypass_scripts_dir="/archived/scripts",
        benchmark_backend="bypass",
    )
    _restore_operator_supplied_paths_from_state(
        _ns(framework_path="/cli/fw", benchmark_scripts_dir="/cli/scripts"),
        state,
    )
    assert os.environ.get("FRAMEWORK_REPO_PATH", "") == ""
    assert os.environ.get("HYPERLOOM_BYPASS_SCRIPTS_DIR", "") == ""
    assert os.environ["HYPERLOOM_BENCHMARK_BACKEND"] == "bypass"
