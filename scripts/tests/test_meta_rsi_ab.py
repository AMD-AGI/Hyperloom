# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests for the A/B step: arm environment and arguments, snapshot, outcome and rerun policy."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from meta_rsi.rsi.config import Arm, parse_config
from meta_rsi.rsi.pipeline import RoundContext
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


class TestSnapshot:
    def test_a_restore_puts_back_what_the_snapshot_recorded_and_clears_caches(self, tmp_path):
        framework, cache = tmp_path / "site" / "vllm", tmp_path / "cache"
        framework.mkdir(parents=True)
        (framework / "ops.py").write_text("original\n")
        assert ab.take_snapshot(tmp_path / "snap", (framework,)) is True
        assert ab.take_snapshot(tmp_path / "snap", (framework,)) is False
        (framework / "ops.py").write_text("patched by a session\n")
        (framework / "extra.py").write_text("added\n")
        cache.mkdir()
        ab.restore_snapshot(tmp_path / "snap", (cache,))
        assert (framework / "ops.py").read_text() == "original\n"
        assert not (framework / "extra.py").exists() and not cache.exists()
        assert not list(framework.parent.glob(".vllm.rsi-*"))


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
