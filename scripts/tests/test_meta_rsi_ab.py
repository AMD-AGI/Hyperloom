# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests for the A/B step: arm environment and arguments, snapshot and caches, the launch
and resume path, Ray reuse, leftover processes, outcome and rerun policy."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import pytest
from meta_rsi.rsi.config import Arm, parse_config
from meta_rsi.rsi.pipeline import RoundContext, StepFailed
from meta_rsi.rsi.state import RoundState
from meta_rsi.rsi.steps import ab

SCENARIO = ["--model", "/m/Qwen", "--framework", "vllm", "--tp", "2", "--max-hours", "24"]


def _ctx(raw: dict) -> RoundContext:
    cfg = parse_config(raw)
    return RoundContext(config=cfg, state=RoundState.load(cfg.round_dir), log=lambda _m: None, sleep=lambda _s: None)


def _session(udp, **state) -> None:
    sess = udp / "Qwen" / "20261001T000000Z-abc"
    sess.mkdir(parents=True)
    (sess / "state.json").write_text(json.dumps(state))


class RayProbe:
    """Stands in for the arm interpreter running ab.RAY_PROBE: one answer per call, modes recorded."""

    def __init__(self, *answers: str):
        self.answers, self.modes = list(answers), []

    def run(self, cmd, **_kwargs):
        self.modes.append(tuple(cmd[3:]))
        return subprocess.CompletedProcess(cmd, 0, f"{self.answers.pop(0)}\n", "")


class Exited:
    pid = 4242

    def poll(self) -> int:
        return 0


class TestArmSetup:
    def test_arguments_replace_the_budget_and_add_the_arm_models(self, rsi_config_dict, tmp_path):
        rsi_config_dict["ab"].update(hours=6, extra_args=["--no-warm-replay"])
        ctx = _ctx(rsi_config_dict)
        arm = Arm("C", "candidate", {"specialist": "glm-5-3"})
        args = ab.arm_args(ctx, arm, tmp_path / "C", SCENARIO)
        assert args.count("--max-hours") == 1 and args[args.index("--max-hours") + 1] == "6"
        assert "--no-warm-replay" in args
        assert args[args.index("--claude-model") + 1] == "claude-opus-5"
        assert args[args.index("--specialist-model") + 1] == "glm-5-3"
        assert args[-2:] == ["--launch-info-file", str(tmp_path / "C" / "optimizer_runs/launch.json")]

    def test_an_arm_without_overrides_passes_no_specialist_model(self, rsi_config_dict, tmp_path):
        args = ab.arm_args(_ctx(rsi_config_dict), Arm("A", "base"), tmp_path / "A", SCENARIO)
        assert "--specialist-model" not in args

    def test_the_arm_environment_drops_driver_settings_and_sets_the_arm(self, rsi_config_dict, tmp_path, monkeypatch):
        monkeypatch.setenv("HYPERLOOM_TRACE_ANALYSIS_ROUTE", "deterministic")
        monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-the-shell")
        creds = tmp_path / "creds.env"
        creds.write_text("export ANTHROPIC_API_KEY='from-the-file'\n")
        rsi_config_dict["agent"]["env_file"] = str(creds)
        rsi_config_dict["ab"]["env"] = {"HYPERLOOM_TRACE_ANALYSIS_ROUTE": "bypass"}
        ctx = _ctx(rsi_config_dict)
        env = ab.arm_env(ctx, Arm("B", "candidate", {"geak": "g"}), tmp_path / "B", tmp_path / "tree")
        assert env["HYPERLOOM_TRACE_ANALYSIS_ROUTE"] == "bypass"
        assert env["PYTHONPATH"] == str(tmp_path / "tree" / "src")
        assert env["ANTHROPIC_API_KEY"] == "from-the-file"
        assert env["USER_DATA_PATH"] == str(tmp_path / "B") and env["GEAK_CLAUDE_MODEL"] == "g"
        assert env["PATH"].startswith(str(ctx.config.ab.python.parent) + ":")

    def test_the_kernel_agent_env_is_rewritten_by_variable_name(self, tmp_path):
        template = "\n".join(
            [
                "export USER_DATA_PATH='/install'",
                "export HYPERLOOM_ROOT='/install/runtime/source-mirrors'",
                "export INFERENCEX_PATH='/opt/InferenceX'",
                "export PYTHONPATH='/install/repo'",
            ]
        )
        env = {
            "USER_DATA_PATH": "/arms/B",
            "HYPERLOOM_KERNEL_AGENT_ROOT": "/tree/src/hyperloom/agents/kernel",
            "KERNEL_AGENT_ROOT": "/tree/src/hyperloom/agents/kernel",
            "PYTHONPATH": "/tree/src",
            "GEAK_CLAUDE_MODEL": "claude-opus-5",
        }
        text = ab.kernel_agent_env_text(template, env, tmp_path / "python")
        assert "export USER_DATA_PATH='/arms/B'" in text
        assert "export HYPERLOOM_ROOT='/arms/B/runtime/source-mirrors'" in text
        assert "export INFERENCEX_PATH='/opt/InferenceX'" in text
        assert "export PYTHONPATH='/tree/src'" in text and "/install" not in text
        assert f"export MAGPIE_PYTHON='{tmp_path / 'python'}'" in text


class TestSnapshotAndCaches:
    def test_a_restore_puts_back_what_the_snapshot_recorded(self, tmp_path):
        framework = tmp_path / "site" / "vllm"
        framework.mkdir(parents=True)
        (framework / "ops.py").write_text("original\n")
        assert ab.take_snapshot(tmp_path / "snap", (framework,)) is True
        assert ab.take_snapshot(tmp_path / "snap", (framework,)) is False
        (framework / "ops.py").write_text("patched by a session\n")
        (framework / "extra.py").write_text("added\n")
        ab.restore_snapshot(tmp_path / "snap")
        assert (framework / "ops.py").read_text() == "original\n"
        assert not (framework / "extra.py").exists()
        assert not list(framework.parent.glob(".vllm.rsi-*"))

    def test_a_cache_that_cannot_be_removed_stops_the_step(self, tmp_path, monkeypatch):
        cache = tmp_path / "cache"
        cache.mkdir()

        def refuse(path):
            raise PermissionError(13, "Permission denied", str(path))

        monkeypatch.setattr(ab.shutil, "rmtree", refuse)
        with pytest.raises(StepFailed, match="cannot clear"):
            ab.clear_caches((cache, tmp_path / "absent"))


class TestRunArm:
    def test_a_fresh_arm_starts_on_cleared_caches_without_a_snapshot_and_its_leftovers_are_stopped(
        self, rsi_config_dict, tmp_path, monkeypatch
    ):
        cache = tmp_path / "triton-cache"
        (cache / "kernel").mkdir(parents=True)
        rsi_config_dict["ab"]["clear_caches"] = [str(cache)]
        ctx = _ctx(rsi_config_dict)
        ctx.runner = RayProbe("unreachable")
        calls = []

        def launch(c, arm, _scenario):
            calls.append(("launch", cache.exists()))
            _session(ab.arm_dir(c, arm.name), stop_reason="time_exhausted")
            return Exited()

        monkeypatch.setattr(ab, "wait_idle", lambda _c: calls.append("idle"))
        monkeypatch.setattr(ab, "launch", launch)
        monkeypatch.setattr(ab, "stop_arm_processes", lambda _c, udp: calls.append(("stop", udp.name)))
        rec = {"status": "pending", "attempt": 0}
        ab.run_arm(ctx, Arm("A", "base"), rec, SCENARIO)
        assert calls == ["idle", ("launch", False), ("stop", "A")]
        assert ctx.runner.modes == [()]
        assert rec["status"] == "finished" and rec["attempt"] == 1 and rec["pid"] == Exited.pid

    def test_a_reachable_ray_cluster_stops_the_launch(self, rsi_config_dict, monkeypatch):
        ctx = _ctx(rsi_config_dict)
        ctx.runner = RayProbe("reachable")
        monkeypatch.setattr(ab, "wait_idle", lambda _c: None)
        monkeypatch.setattr(ab, "launch", lambda *_a: pytest.fail("launched next to a reachable Ray cluster"))
        with pytest.raises(StepFailed, match="Ray cluster is reachable"):
            ab.run_arm(ctx, Arm("A", "base"), {"status": "pending", "attempt": 0}, SCENARIO)
        assert ctx.runner.modes == [()]
        assert not ab.arm_dir(ctx, "A").exists()

    @pytest.mark.parametrize(("after_stop", "launched"), [("unreachable", True), ("reachable", False)])
    def test_stop_ray_stops_the_cluster_and_launches_only_once_it_is_gone(
        self, rsi_config_dict, monkeypatch, after_stop, launched
    ):
        rsi_config_dict["ab"]["stop_ray"] = True
        ctx = _ctx(rsi_config_dict)
        ctx.runner = RayProbe("reachable", after_stop)
        started = []
        monkeypatch.setattr(ab, "wait_idle", lambda _c: None)
        monkeypatch.setattr(ab, "launch", lambda *_a: started.append(True) or Exited())
        monkeypatch.setattr(ab, "stop_arm_processes", lambda _c, _udp: None)
        rec = {"status": "pending", "attempt": 0}
        if launched:
            ab.run_arm(ctx, Arm("A", "base"), rec, SCENARIO)
        else:
            with pytest.raises(StepFailed, match="still reachable"):
                ab.run_arm(ctx, Arm("A", "base"), rec, SCENARIO)
        assert ctx.runner.modes == [(), ("stop",)]
        assert started == ([True] if launched else [])

    def test_a_resumed_arm_whose_optimizer_died_is_recorded_not_relaunched(self, rsi_config_dict, monkeypatch):
        ctx = _ctx(rsi_config_dict)
        _session(ab.arm_dir(ctx, "B"), stop_reason="time_exhausted")
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        stopped = []
        monkeypatch.setattr(ab, "launch", lambda *_a: pytest.fail("a dead arm was relaunched"))
        monkeypatch.setattr(ab, "stop_arm_processes", lambda _c, udp: stopped.append(udp.name))
        rec = {"status": "running", "attempt": 1, "pid": dead.pid}
        ab.run_arm(ctx, Arm("B", "candidate"), rec, SCENARIO)
        assert rec["status"] == "finished" and rec["attempt"] == 1 and stopped == ["B"]


class TestHostProcesses:
    def test_only_processes_carrying_the_arms_user_data_path_are_stopped(self, rsi_config_dict):
        ctx = _ctx(rsi_config_dict)
        ctx.sleep = time.sleep
        udp = ab.arm_dir(ctx, "A")
        sleeper = [sys.executable, "-c", "import time; time.sleep(120)"]
        mine = subprocess.Popen(sleeper, env={**os.environ, "USER_DATA_PATH": str(udp)})
        other = subprocess.Popen(sleeper, env={**os.environ, "USER_DATA_PATH": f"{udp}-other"})
        try:
            assert mine.pid in ab.arm_processes(udp) and other.pid not in ab.arm_processes(udp)
            ab.stop_arm_processes(ctx, udp)
            assert mine.wait(timeout=10) == -signal.SIGTERM
            assert other.poll() is None
        finally:
            for proc in (mine, other):
                proc.kill()
                proc.wait()

    def test_leftover_servers_fail_the_step_after_the_idle_wait(self, rsi_config_dict, monkeypatch):
        ctx = _ctx(rsi_config_dict)
        naps = []
        ctx.sleep = naps.append
        monkeypatch.setattr(ab, "busy_processes", lambda: ["101: vllm serve /m/Qwen"])
        with pytest.raises(StepFailed, match="vllm serve /m/Qwen"):
            ab.wait_idle(ctx)
        assert sum(naps) == ab.IDLE_WAIT_SEC

    def test_an_idle_host_does_not_wait(self, rsi_config_dict, monkeypatch):
        ctx = _ctx(rsi_config_dict)
        ctx.sleep = lambda _s: pytest.fail("waited on an idle host")
        monkeypatch.setattr(ab, "busy_processes", lambda: [])
        ab.wait_idle(ctx)


class TestArmOutcome:
    def test_a_stopped_session_is_finished_and_a_cut_one_is_interrupted(self, tmp_path):
        _session(tmp_path / "A", stop_reason="sweep_done", phase="CLOSE")
        _session(tmp_path / "B", stop_reason="", phase="KERNEL_AGENT")
        assert ab.arm_outcome(tmp_path / "A") == "finished"
        assert ab.arm_outcome(tmp_path / "B") == "interrupted"
        assert ab.arm_outcome(tmp_path / "missing") == "interrupted"

    def test_an_interrupted_arm_is_moved_aside_once_then_kept(self, rsi_config_dict):
        ctx = _ctx(rsi_config_dict)
        udp = ab.arm_dir(ctx, "B")
        _session(udp, phase="KERNEL_AGENT")
        rec = {"status": "interrupted", "attempt": 1}
        ab.settle(ctx, Arm("B", "candidate"), rec)
        assert rec["status"] == "pending" and (udp.parent / "B.interrupted-1").exists() and not udp.exists()
        rec = {"status": "interrupted", "attempt": ab.MAX_ARM_ATTEMPTS}
        ab.settle(ctx, Arm("B", "candidate"), rec)
        assert rec == {"status": "finished", "attempt": ab.MAX_ARM_ATTEMPTS, "partial": True}

    def test_keep_policy_compares_an_interrupted_arm_as_it_stands(self, rsi_config_dict):
        rsi_config_dict["ab"]["on_interrupt"] = "keep"
        rec = {"status": "interrupted", "attempt": 1}
        ab.settle(_ctx(rsi_config_dict), Arm("B", "candidate"), rec)
        assert rec["status"] == "finished" and rec["partial"] is True

    def test_a_finished_process_is_not_taken_for_a_live_optimizer(self, tmp_path):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        assert ab.optimizer_alive(proc.pid, tmp_path) is False


def test_a_rerun_refuses_to_overwrite_an_existing_arm_directory(rsi_config_dict):
    ctx = _ctx(rsi_config_dict)
    ab.arm_dir(ctx, "A").mkdir(parents=True)
    with pytest.raises(ab.StepFailed, match="already exists"):
        ab.launch(ctx, Arm("A", "base"), SCENARIO)
