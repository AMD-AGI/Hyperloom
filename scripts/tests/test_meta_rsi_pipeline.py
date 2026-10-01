# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests for the rsi driver's configuration, state and pipeline."""

from __future__ import annotations

import copy

import pytest
import yaml
from meta_rsi.rsi.__main__ import main
from meta_rsi.rsi.config import AGENT_STEPS, ConfigError, parse_config
from meta_rsi.rsi.pipeline import RoundContext, Step, StepFailed, read_env_file, run_pipeline, select_steps
from meta_rsi.rsi.state import RoundState, round_lock
from meta_rsi.rsi.steps import STEPS


def _ctx(cfg_dict: dict) -> RoundContext:
    cfg = parse_config(cfg_dict)
    return RoundContext(config=cfg, state=RoundState.load(cfg.round_dir), log=lambda _m: None)


class TestConfig:
    def test_a_minimal_configuration_parses_without_machine_defaults(self, rsi_config_dict):
        cfg = parse_config(rsi_config_dict)
        assert cfg.data.models_dir is None and cfg.data.local_sessions == ()
        assert cfg.ab.snapshot_dir is None and cfg.router_log is None
        assert [a.name for a in cfg.ab.arms] == ["A", "B"]

    @pytest.mark.parametrize(
        "section, key",
        [("target", "repo"), ("data", "bundles_dir"), ("data", "recent_since"), ("agent", "model"), ("ab", "python")],
    )
    def test_a_missing_required_value_is_named(self, rsi_config_dict, section, key):
        del rsi_config_dict[section][key]
        with pytest.raises(ConfigError, match=f"{section}.{key}"):
            parse_config(rsi_config_dict)

    def test_arms_need_a_control_and_unique_names_and_known_trees(self, rsi_config_dict):
        for arms, message in (
            ([{"name": "A", "tree": "base"}], "at least two"),
            ([{"name": "A", "tree": "base"}, {"name": "A", "tree": "candidate"}], "unique"),
            ([{"name": "A", "tree": "base"}, {"name": "B", "tree": "main"}], "tree"),
            ([{"name": "A", "tree": "base"}, {"name": "B", "tree": "candidate", "models": {"critic": "x"}}], "models"),
        ):
            raw = copy.deepcopy(rsi_config_dict)
            raw["ab"]["arms"] = arms
            with pytest.raises(ConfigError, match=message):
                parse_config(raw)

    def test_budgets_may_only_name_agent_steps(self, rsi_config_dict):
        rsi_config_dict["agent"]["budget_usd"] = {"fetch": 5}
        with pytest.raises(ConfigError, match="unknown agent steps"):
            parse_config(rsi_config_dict)

    def test_a_snapshot_needs_both_a_directory_and_paths(self, rsi_config_dict, tmp_path):
        rsi_config_dict["ab"]["snapshot"] = {"dir": str(tmp_path / "snap")}
        with pytest.raises(ConfigError, match="snapshot"):
            parse_config(rsi_config_dict)

    def test_stopping_a_ray_cluster_is_opt_in(self, rsi_config_dict):
        assert parse_config(rsi_config_dict).ab.stop_ray is False

    def test_test_runs_are_bounded_by_a_positive_timeout(self, rsi_config_dict):
        assert parse_config(rsi_config_dict).checks.timeout_min > 0
        rsi_config_dict["checks"]["timeout_min"] = 0
        with pytest.raises(ConfigError, match="timeout_min"):
            parse_config(rsi_config_dict)


class TestState:
    def test_records_survive_a_reload(self, tmp_path):
        state = RoundState.load(tmp_path)
        state.step("fetch").start()
        state.step("fetch").finish({"recent_sessions": 3})
        state.agent_cost_usd = 1.25
        state.data["ab"] = {"A": {"status": "running", "pid": 7}}
        state.save()
        again = RoundState.load(tmp_path)
        assert again.step("fetch").status == "done" and again.step("fetch").outputs == {"recent_sessions": 3}
        assert again.agent_cost_usd == 1.25 and again.data["ab"]["A"]["pid"] == 7

    def test_a_second_driver_cannot_take_the_lock(self, tmp_path):
        with round_lock(tmp_path):
            with pytest.raises(SystemExit, match="another rsi driver"):
                with round_lock(tmp_path):
                    pass


class TestPipeline:
    def test_selection_follows_pipeline_order(self):
        def names(steps):
            return [s.name for s in steps]

        assert names(select_steps(STEPS, until="analyze")) == ["fetch", "analyze"]
        assert names(select_steps(STEPS, start="compare", until="report")) == ["compare", "diagnose", "report"]
        assert names(select_steps(STEPS, only=["report", "fetch"])) == ["fetch", "report"]
        with pytest.raises(SystemExit, match="unknown step"):
            select_steps(STEPS, only=["nope"])

    def test_agent_steps_match_the_configured_budget_keys(self):
        assert tuple(s.name for s in STEPS if s.kind == "agent") == AGENT_STEPS

    def test_finished_steps_are_skipped_and_a_failure_stops_the_run(self, rsi_config_dict):
        ctx, calls = _ctx(rsi_config_dict), []

        def ok(name):
            return lambda _ctx: calls.append(name) or {"ran": name}

        def broken(_ctx):
            raise StepFailed("no data")

        steps = [Step("one", "script", ok("one")), Step("two", "script", broken), Step("three", "script", ok("three"))]
        assert run_pipeline(ctx, steps) is False
        assert calls == ["one"]
        assert ctx.state.step("two").status == "failed" and ctx.state.step("two").error == "no data"
        steps[1] = Step("two", "script", ok("two"))
        assert run_pipeline(ctx, steps) is True
        assert calls == ["one", "two", "three"]
        assert RoundState.load(ctx.round_dir).step("one").attempts == 1

    def test_a_step_does_not_run_alone_before_its_needs(self, rsi_config_dict):
        ctx = _ctx(rsi_config_dict)
        steps = [Step("later", "script", lambda _c: {}, needs=("earlier",))]
        assert run_pipeline(ctx, steps) is False
        assert ctx.state.step("later").status == "pending"

    def test_an_unexpected_error_leaves_the_step_failed_not_running(self, rsi_config_dict):
        ctx = _ctx(rsi_config_dict)

        def crash(_ctx):
            raise KeyError("bug")

        with pytest.raises(KeyError):
            run_pipeline(ctx, [Step("x", "script", crash)])
        assert RoundState.load(ctx.round_dir).step("x").status == "failed"

    def test_a_step_reporting_skipped_is_recorded_as_skipped(self, rsi_config_dict):
        ctx = _ctx(rsi_config_dict)
        assert run_pipeline(ctx, [Step("x", "script", lambda _c: {"skipped": True})])
        assert ctx.state.step("x").status == "skipped"

    def test_the_agent_budget_caps_each_call_and_stops_when_spent(self, rsi_config_dict, tmp_path):
        from meta_rsi.rsi.agent import AgentResult, AgentSpec

        seen = []

        def agent(spec):
            seen.append(spec.budget_usd)
            return AgentResult(text="", is_error=False, turns=1, cost_usd=6.0, usage={})

        ctx = _ctx(rsi_config_dict)
        ctx.agent = agent
        spec = AgentSpec("x", "p", "s", tmp_path, ("Read",), "m", budget_usd=8.0, max_turns=5)
        ctx.run_agent(spec)
        ctx.run_agent(spec)
        assert seen == [8.0, 4.0]
        with pytest.raises(StepFailed, match="budget is spent"):
            ctx.run_agent(spec)


def test_env_files_resolve_references_like_the_shell_files_they_come_from(tmp_path, monkeypatch):
    monkeypatch.setenv("FROM_PROCESS", "proc")
    env_file = tmp_path / "creds.env"
    env_file.write_text(
        "# credentials\n"
        'export SAFE_API_KEY="secret"\n'
        'export ANTHROPIC_API_KEY="$SAFE_API_KEY"\n'
        "ANTHROPIC_AUTH_TOKEN='${SAFE_API_KEY}'\n"
        "OTHER=$FROM_PROCESS-$UNDEFINED_NAME\n"
    )
    values = read_env_file(env_file)
    assert values["ANTHROPIC_API_KEY"] == values["ANTHROPIC_AUTH_TOKEN"] == "secret"
    assert values["OTHER"] == "proc-$UNDEFINED_NAME"


def test_cli_init_and_status_report_the_round(rsi_config_dict, tmp_path, capsys):
    cfg_path = tmp_path / "round.yaml"
    cfg_path.write_text(yaml.safe_dump(rsi_config_dict))
    assert main(["init", "--config", str(cfg_path)]) == 0
    assert (tmp_path / "round" / "round.yaml").exists()
    assert main(["status", "--config", str(cfg_path)]) == 0
    out = capsys.readouterr().out
    assert "fetch" in out and "pending" in out and "agent spend: $0.00 of $10" in out


def test_cli_rejects_a_bad_configuration(tmp_path, capsys):
    cfg_path = tmp_path / "round.yaml"
    cfg_path.write_text("round_dir: /tmp/x\n")
    assert main(["status", "--config", str(cfg_path)]) == 2
    assert "config:" in capsys.readouterr().err
