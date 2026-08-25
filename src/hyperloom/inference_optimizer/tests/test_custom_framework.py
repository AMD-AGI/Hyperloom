# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``custom`` framework: an operator's workload, supplied at launch.

Every other framework is described by code and shipped assets. ``custom`` is
described by two directories the operator passes in, so these lock the seams
that carry them: the registry entry, the config the loop falls back on, the
entrypoint lookup, and the measurement contract that replaces the hand-written
blacklists the shipped frameworks rely on.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from hyperloom.inference_optimizer import framework_registry as fr


class TestRegistryEntry:
    def test_custom_is_a_scriptable_framework(self):
        assert fr.is_supported("custom")
        assert fr.is_scriptable("custom")

    def test_custom_declares_no_upstream_repo(self):
        """There is no upstream to mine PRs from; the code is the operator's."""
        assert fr.FRAMEWORKS["custom"].repo_url is None

    def test_custom_is_offered_by_the_cli(self):
        from hyperloom.inference_optimizer.cli import parser as cli_parser

        p = cli_parser._build_parser()
        ns = p.parse_args(["optimize", "--framework", "custom", "--model", "/m"])
        assert ns.framework == "custom"

    def test_shipped_frameworks_keep_their_units(self):
        """The new entry must not perturb how existing sessions report."""
        assert fr.throughput_unit("xdit") == "img/s"
        assert fr.throughput_unit("sglang") == "tok/s"


class TestConfigResolution:
    def test_baseline_and_profile_resolve_to_the_custom_yamls(self, monkeypatch):
        from hyperloom.orchestrator.actions.executors._workload_envs import default_baseline_config
        from hyperloom.orchestrator.actions.executors.profile import _default_profile_config

        monkeypatch.setenv("FRAMEWORK", "custom")
        assert default_baseline_config().name == "baseline_custom.yaml"
        assert _default_profile_config().name == "profile_custom.yaml"

    @pytest.mark.parametrize("stem", ["baseline", "profile"])
    def test_the_custom_configs_declare_the_scriptable_contract(self, stem):
        """A serving default here would boot a server for a server-less run."""
        from hyperloom.inference_optimizer.session.paths import asset_root

        cfg = yaml.safe_load((asset_root() / "assets" / "configs" / f"{stem}_custom.yaml").read_text())
        bench = cfg["benchmark"]
        assert bench["framework"] == "custom"
        assert bench["workload_kind"] == "scriptable"
        assert bench["server_lifecycle"]["enabled"] is False

    def test_the_custom_configs_pin_no_workload_of_their_own(self):
        """Hyperloom cannot know the operator's knobs, so it must invent none."""
        from hyperloom.inference_optimizer.session.paths import asset_root

        cfg = yaml.safe_load((asset_root() / "assets" / "configs" / "baseline_custom.yaml").read_text())
        assert set(cfg["benchmark"]["envs"]) == {"TP", "PATH"}


class TestEntrypointAndCheckout:
    def _bench(self) -> dict:
        return {"framework": "custom"}

    def test_the_lone_script_in_the_directory_is_the_entrypoint(self, tmp_path, monkeypatch):
        """What an operator with one script for one machine actually has."""
        from hyperloom.orchestrator.actions.executors import _workload_envs as we

        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "run_whatever.sh").write_text("#!/bin/sh\n")
        monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(scripts))

        bench, envs = self._bench(), {}
        we.apply_scriptable_runtime_defaults(bench, envs, gpu_type="mi355x", explicit_benchmark_script=False)
        assert bench["benchmark_script"] == str(scripts / "run_whatever.sh")

    def test_the_runner_suffixed_script_wins_over_a_sibling(self, tmp_path, monkeypatch):
        from hyperloom.orchestrator.actions.executors import _workload_envs as we

        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "custom_mi355x.sh").write_text("#!/bin/sh\n")
        (scripts / "other.sh").write_text("#!/bin/sh\n")
        monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(scripts))

        bench, envs = self._bench(), {}
        we.apply_scriptable_runtime_defaults(bench, envs, gpu_type="mi355x", explicit_benchmark_script=False)
        assert bench["benchmark_script"] == str(scripts / "custom_mi355x.sh")

    def test_an_ambiguous_directory_picks_nothing(self, tmp_path, monkeypatch):
        """Two unnamed scripts is a question, not a default; let it surface."""
        from hyperloom.orchestrator.actions.executors import _workload_envs as we

        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "a.sh").write_text("#!/bin/sh\n")
        (scripts / "b.sh").write_text("#!/bin/sh\n")
        monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(scripts))

        bench, envs = self._bench(), {}
        we.apply_scriptable_runtime_defaults(bench, envs, gpu_type="mi355x", explicit_benchmark_script=False)
        assert "benchmark_script" not in bench

    def test_an_explicit_script_is_never_overwritten(self, tmp_path, monkeypatch):
        from hyperloom.orchestrator.actions.executors import _workload_envs as we

        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "custom_mi355x.sh").write_text("#!/bin/sh\n")
        monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(scripts))

        bench = {"framework": "custom", "benchmark_script": "/chosen/by/operator.sh"}
        we.apply_scriptable_runtime_defaults(bench, {}, gpu_type="mi355x", explicit_benchmark_script=True)
        assert bench["benchmark_script"] == "/chosen/by/operator.sh"

    def test_the_checkout_reaches_the_orchestrator_env(self, tmp_path, monkeypatch):
        """PolicyGate reads os.environ; a path only in the YAML is invisible."""
        from hyperloom.orchestrator.actions.executors import _workload_envs as we

        repo = tmp_path / "my-framework"
        repo.mkdir()
        # setenv, not delenv: the helper under test writes os.environ directly,
        # and monkeypatch only restores keys it recorded. delenv on an absent key
        # records nothing, so the write would outlive the test and land in the
        # source-root allowlist every later test computes.
        monkeypatch.setenv("CUSTOM_REPO_PATH", "")
        monkeypatch.setenv("CUSTOM_DIR", "")
        monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(repo))

        bench, envs = self._bench(), {}
        we.apply_scriptable_runtime_defaults(bench, envs, gpu_type="mi355x", explicit_benchmark_script=False)
        assert envs["CUSTOM_REPO_PATH"] == str(repo)
        assert envs["CUSTOM_DIR"] == str(repo)
        assert os.environ.get("CUSTOM_REPO_PATH") == str(repo)
        # The generic name has to travel in the config, not just os.environ: a
        # Ray worker inherits the raylet's environment from whenever it booted,
        # and a stale checkout there wins over anything published afterwards.
        assert envs["FRAMEWORK_REPO_PATH"] == str(repo)

    def test_extra_env_pins_reach_the_benchmark_config(self, monkeypatch):
        """The operator's only channel for their own knobs has to land in envs.

        The CLI serializes ``--extra-env`` into a JSON env var that, on its own,
        only the grid filter reads. Left there the pins are neither delivered to
        the script nor visible to the measurement contract, which is read off
        the materialized config.
        """
        from hyperloom.orchestrator.actions.executors import _workload_envs as we

        monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", '{"MYFW_STEPS": "50", "MYFW_CKPT": "/w/x.pt"}')
        bench, envs = self._bench(), {}
        we.apply_scriptable_runtime_defaults(bench, envs, gpu_type="mi355x", explicit_benchmark_script=False)
        assert envs["MYFW_STEPS"] == "50"
        assert envs["MYFW_CKPT"] == "/w/x.pt"

    def test_a_config_value_outranks_an_extra_env_pin(self, monkeypatch):
        """setdefault, not overwrite: a variant's override still has to win."""
        from hyperloom.orchestrator.actions.executors import _workload_envs as we

        monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", '{"MYFW_STEPS": "50"}')
        bench, envs = self._bench(), {"MYFW_STEPS": "4"}
        we.apply_scriptable_runtime_defaults(bench, envs, gpu_type="mi355x", explicit_benchmark_script=False)
        assert envs["MYFW_STEPS"] == "4"

    def test_an_unparseable_pin_does_not_take_the_run_down(self, monkeypatch):
        from hyperloom.orchestrator.actions.executors import _workload_envs as we

        monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", "{not json")
        bench, envs = self._bench(), {}
        we.apply_scriptable_runtime_defaults(bench, envs, gpu_type="mi355x", explicit_benchmark_script=False)
        assert envs == {}

    def test_extra_env_pins_stay_out_of_other_frameworks(self, monkeypatch):
        from hyperloom.orchestrator.actions.executors import _workload_envs as we

        monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", '{"MYFW_STEPS": "50"}')
        envs: dict = {}
        we.apply_scriptable_runtime_defaults(
            {"framework": "xdit"}, envs, gpu_type="mi355x", explicit_benchmark_script=False
        )
        assert "MYFW_STEPS" not in envs

    def test_the_helpers_stay_inert_for_other_frameworks(self, tmp_path, monkeypatch):
        from hyperloom.orchestrator.actions.executors import _workload_envs as we

        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "custom_mi355x.sh").write_text("#!/bin/sh\n")
        monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(scripts))

        bench = {"framework": "xdit"}
        envs: dict = {}
        we.apply_scriptable_runtime_defaults(bench, envs, gpu_type="mi355x", explicit_benchmark_script=False)
        assert "CUSTOM_REPO_PATH" not in envs
        assert "custom_mi355x.sh" not in str(bench.get("benchmark_script") or "")


class TestMeasurementContract:
    """What replaces the hand-written blacklists for an operator's workload."""

    def _grid(self, payload):
        from hyperloom.orchestrator.actions.executors.explore import _grid_variants_from_payload

        return _grid_variants_from_payload(payload)

    def test_overwriting_a_pinned_env_is_dropped(self):
        """Same output, different measurement: the quality gate cannot see it."""
        from hyperloom.orchestrator.actions.executors.explore import filter_operator_pinned_envs

        grid = self._grid([{"name": "halve_repeats", "extra_envs": {"MYFW_REPEATS": "1"}}])
        kept, dropped = filter_operator_pinned_envs(grid, {"MYFW_REPEATS": 3})
        assert kept == []
        assert "MYFW_REPEATS" in dropped[0][1]

    def test_adding_an_unpinned_env_is_the_point_of_exploring(self):
        from hyperloom.orchestrator.actions.executors.explore import filter_operator_pinned_envs

        grid = self._grid([{"name": "new_knob", "extra_envs": {"MYFW_CACHE": "1"}}])
        kept, dropped = filter_operator_pinned_envs(grid, {"MYFW_REPEATS": 3})
        assert [v.name for v in kept] == ["new_knob"]
        assert dropped == []

    def test_the_key_match_ignores_case(self):
        from hyperloom.orchestrator.actions.executors.explore import filter_operator_pinned_envs

        grid = self._grid([{"name": "lower", "extra_envs": {"myfw_repeats": "1"}}])
        kept, _ = filter_operator_pinned_envs(grid, {"MYFW_REPEATS": 3})
        assert kept == []

    def test_a_baseline_that_pins_nothing_constrains_nothing(self):
        from hyperloom.orchestrator.actions.executors.explore import filter_operator_pinned_envs

        grid = self._grid([{"name": "anything", "extra_envs": {"MYFW_X": "1"}}])
        kept, dropped = filter_operator_pinned_envs(grid, {})
        assert len(kept) == 1 and dropped == []

    def test_a_config_replay_carries_the_pinned_values_by_construction(self):
        """It reproduces a measured config; reading that as an edit strands it."""
        from hyperloom.orchestrator.actions.executors.explore import filter_operator_pinned_envs

        grid = self._grid(
            [
                {
                    "name": "geak_revalidate",
                    "extra_envs": {"MYFW_REPEATS": "3"},
                    "provenance": "geak_revalidate",
                }
            ]
        )
        kept, dropped = filter_operator_pinned_envs(grid, {"MYFW_REPEATS": 3})
        assert [v.name for v in kept] == ["geak_revalidate"]
        assert dropped == []

    def test_a_shipped_framework_keeps_flipping_its_pinned_defaults(self):
        """A shipped baseline pins knobs at their off value for explore to flip.

        Reading a pin as a lock would drop exactly the A/B legs those pins were
        written to enable, so the guard has to stay off the shipped configs.
        """
        from hyperloom.inference_optimizer.session.paths import asset_root

        cfg = yaml.safe_load((asset_root() / "assets" / "configs" / "baseline_xdit.yaml").read_text())
        assert cfg["benchmark"]["envs"], "a shipped baseline pins envs explore may still flip"

        import inspect

        from hyperloom.orchestrator.actions.executors import explore as explore_mod

        lines = inspect.getsource(explore_mod).splitlines()
        call_at = next(i for i, line in enumerate(lines) if "filter_operator_pinned_envs(grid" in line)
        preceding = next(line for line in reversed(lines[:call_at]) if line.strip())
        assert 'framework == "custom"' in preceding, (
            "the pinned-env guard must stay scoped to custom; applying it to a shipped "
            "framework would drop the A/B legs its baseline pins exist to enable, and "
            f"it is now guarded by: {preceding.strip()!r}"
        )


class TestLaunchValidation:
    def _args(self, **kw):
        from argparse import Namespace

        return Namespace(framework_path=None, benchmark_scripts_dir=None, **kw)

    @pytest.fixture(autouse=True)
    def _bypass_backend(self, monkeypatch):
        # custom refuses to launch on any other backend; the path assertions in
        # this class are about the paths, so give them the one that is valid.
        monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "bypass")

    def test_custom_without_its_paths_exits_instead_of_running(self, monkeypatch):
        """Failing here beats failing at the first benchmark, half an hour in."""
        from hyperloom.inference_optimizer.cli import _apply_operator_supplied_paths

        monkeypatch.setenv("FRAMEWORK_REPO_PATH", "")
        monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", "")
        with pytest.raises(SystemExit) as exc:
            _apply_operator_supplied_paths(self._args(), "custom")
        assert exc.value.code == 2

    def test_a_path_that_is_not_a_directory_exits(self, monkeypatch, tmp_path):
        from hyperloom.inference_optimizer.cli import _apply_operator_supplied_paths

        monkeypatch.setenv("FRAMEWORK_REPO_PATH", "")
        monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", "")
        args = self._args()
        args.framework_path = str(tmp_path / "does-not-exist")
        with pytest.raises(SystemExit) as exc:
            _apply_operator_supplied_paths(args, "custom")
        assert exc.value.code == 2

    def test_the_flags_publish_the_env_the_executors_read(self, monkeypatch, tmp_path):
        from hyperloom.inference_optimizer.cli import _apply_operator_supplied_paths

        repo, scripts = tmp_path / "fw", tmp_path / "sc"
        repo.mkdir()
        scripts.mkdir()
        monkeypatch.setenv("FRAMEWORK_REPO_PATH", "")
        monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", "")
        args = self._args()
        args.framework_path = str(repo)
        args.benchmark_scripts_dir = str(scripts)

        _apply_operator_supplied_paths(args, "custom")
        assert os.environ["FRAMEWORK_REPO_PATH"] == str(repo.resolve())
        assert os.environ["HYPERLOOM_BYPASS_SCRIPTS_DIR"] == str(scripts.resolve())

    @pytest.mark.parametrize("backend", ["", "magpie", "MAGPIE", "  ", "bypasss", "none"])
    def test_custom_refuses_a_backend_that_cannot_run_the_script(self, monkeypatch, tmp_path, backend):
        """The default backend is Magpie, which cannot run an operator's script.

        Nothing downstream rejects the pairing, so an unset or wrong value used
        to be accepted and the run simply took the wrong executor.
        """
        from hyperloom.inference_optimizer.cli import _apply_operator_supplied_paths

        repo, scripts = tmp_path / "fw", tmp_path / "sc"
        repo.mkdir()
        scripts.mkdir()
        monkeypatch.setenv("FRAMEWORK_REPO_PATH", "")
        monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", "")
        monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", backend)
        args = self._args()
        args.framework_path = str(repo)
        args.benchmark_scripts_dir = str(scripts)

        with pytest.raises(SystemExit) as exc:
            _apply_operator_supplied_paths(args, "custom")
        assert exc.value.code == 2

    @pytest.mark.parametrize("backend", ["bypass", "  Bypass  ", "BYPASS"])
    def test_custom_accepts_bypass_however_it_is_spelled(self, monkeypatch, tmp_path, backend):
        """Normalisation matches install.sh's, so the two gates cannot disagree."""
        from hyperloom.inference_optimizer.cli import _apply_operator_supplied_paths

        repo, scripts = tmp_path / "fw", tmp_path / "sc"
        repo.mkdir()
        scripts.mkdir()
        monkeypatch.setenv("FRAMEWORK_REPO_PATH", "")
        monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", "")
        monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", backend)
        args = self._args()
        args.framework_path = str(repo)
        args.benchmark_scripts_dir = str(scripts)

        _apply_operator_supplied_paths(args, "custom")
        assert os.environ["FRAMEWORK_REPO_PATH"] == str(repo.resolve())

    def test_a_shipped_framework_keeps_its_backend_freedom(self, monkeypatch):
        """The gate is scoped to custom; xdit/sglang launches are untouched."""
        from hyperloom.inference_optimizer.cli import _apply_operator_supplied_paths

        monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "magpie")
        for framework in ("sglang", "vllm", "xdit"):
            _apply_operator_supplied_paths(self._args(), framework)

    def test_a_shipped_framework_needs_neither_flag(self, monkeypatch):
        from hyperloom.inference_optimizer.cli import _apply_operator_supplied_paths

        monkeypatch.setenv("FRAMEWORK_REPO_PATH", "")
        monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", "")
        _apply_operator_supplied_paths(self._args(), "xdit")

    def test_an_exported_env_outranks_the_flag(self, monkeypatch, tmp_path):
        """Matches every other path override in the loop."""
        from hyperloom.inference_optimizer.cli import _apply_operator_supplied_paths

        repo = tmp_path / "fw"
        repo.mkdir()
        monkeypatch.setenv("FRAMEWORK_REPO_PATH", "/already/exported")
        monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", "/already/exported")
        args = self._args()
        args.framework_path = str(repo)

        _apply_operator_supplied_paths(args, "custom")
        assert os.environ["FRAMEWORK_REPO_PATH"] == "/already/exported"


class TestEntrypointFailFast:
    def _args(self, **kw):
        from argparse import Namespace

        return Namespace(framework_path=None, benchmark_scripts_dir=None, **kw)

    @pytest.fixture(autouse=True)
    def _bypass_backend(self, monkeypatch):
        monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "bypass")
        monkeypatch.setenv("CUSTOM_REPO_PATH", "")
        monkeypatch.setenv("CUSTOM_DIR", "")

    def test_an_empty_scripts_dir_exits_listing_candidates(self, monkeypatch, tmp_path, capsys):
        """Failing here beats looping magpie_nonzero_invalid_measurement for hours."""
        from hyperloom.inference_optimizer.cli import _require_custom_entrypoint

        repo, scripts = tmp_path / "fw", tmp_path / "sc"
        repo.mkdir()
        scripts.mkdir()
        monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(repo))
        monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(scripts))
        monkeypatch.delenv("MAGPIE_PATH", raising=False)
        monkeypatch.delenv("INFERENCEX_PATH", raising=False)

        with pytest.raises(SystemExit) as exc:
            _require_custom_entrypoint("custom", gpu_type="mi355x")
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "custom_mi355x.sh" in err
        assert str(scripts / "custom_mi355x.sh") in err

    def test_a_resolved_script_passes(self, monkeypatch, tmp_path):
        from hyperloom.inference_optimizer.cli import _require_custom_entrypoint

        repo, scripts = tmp_path / "fw", tmp_path / "sc"
        repo.mkdir()
        scripts.mkdir()
        (scripts / "custom_mi355x.sh").write_text("#!/bin/sh\n")
        monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(repo))
        monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(scripts))

        _require_custom_entrypoint("custom", gpu_type="mi355x")

    def test_a_lone_operator_script_passes(self, monkeypatch, tmp_path):
        from hyperloom.inference_optimizer.cli import _require_custom_entrypoint

        repo, scripts = tmp_path / "fw", tmp_path / "sc"
        repo.mkdir()
        scripts.mkdir()
        (scripts / "run_whatever.sh").write_text("#!/bin/sh\n")
        monkeypatch.setenv("FRAMEWORK_REPO_PATH", str(repo))
        monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(scripts))

        _require_custom_entrypoint("custom", gpu_type="mi355x")

    def test_shipped_frameworks_are_not_gated(self):
        from hyperloom.inference_optimizer.cli import _require_custom_entrypoint

        _require_custom_entrypoint("sglang", gpu_type="mi300x")
        _require_custom_entrypoint("xdit", gpu_type="mi300x")


class TestResumeReappliesOperatorPaths:
    def _args(self, **kw):
        from argparse import Namespace

        return Namespace(framework_path=None, benchmark_scripts_dir=None, **kw)

    @pytest.fixture(autouse=True)
    def _bypass_backend(self, monkeypatch):
        monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "bypass")

    def test_repassing_the_flags_publishes_them(self, monkeypatch, tmp_path):
        from hyperloom.inference_optimizer.cli import _apply_operator_supplied_paths

        repo, scripts = tmp_path / "fw", tmp_path / "sc"
        repo.mkdir()
        scripts.mkdir()
        (scripts / "custom_mi355x.sh").write_text("#!/bin/sh\n")
        monkeypatch.setenv("FRAMEWORK_REPO_PATH", "")
        monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", "")
        args = self._args()
        args.framework_path = str(repo)
        args.benchmark_scripts_dir = str(scripts)

        _apply_operator_supplied_paths(args, "custom")
        assert os.environ["FRAMEWORK_REPO_PATH"] == str(repo.resolve())
        assert os.environ["HYPERLOOM_BYPASS_SCRIPTS_DIR"] == str(scripts.resolve())

    def test_archived_paths_fill_in_when_flags_are_omitted(self, monkeypatch, tmp_path):
        from hyperloom.inference_optimizer.cli import (
            _apply_operator_supplied_paths,
            _restore_operator_supplied_paths_from_state,
        )
        from hyperloom.orchestrator.state.shared_state import SharedState

        repo, scripts = tmp_path / "fw", tmp_path / "sc"
        repo.mkdir()
        scripts.mkdir()
        (scripts / "custom_mi355x.sh").write_text("#!/bin/sh\n")
        monkeypatch.setenv("FRAMEWORK_REPO_PATH", "")
        monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", "")
        state = SharedState(
            session_id="s",
            framework="custom",
            framework_repo_path=str(repo),
            bypass_scripts_dir=str(scripts),
            benchmark_backend="bypass",
        )

        _restore_operator_supplied_paths_from_state(self._args(), state)
        _apply_operator_supplied_paths(self._args(), "custom")
        assert os.environ["FRAMEWORK_REPO_PATH"] == str(repo)
        assert os.environ["HYPERLOOM_BYPASS_SCRIPTS_DIR"] == str(scripts)

    def test_a_repassed_flag_outranks_the_archive(self, monkeypatch, tmp_path):
        from hyperloom.inference_optimizer.cli import (
            _apply_operator_supplied_paths,
            _restore_operator_supplied_paths_from_state,
        )
        from hyperloom.orchestrator.state.shared_state import SharedState

        archived_scripts = tmp_path / "old"
        archived_scripts.mkdir()
        (archived_scripts / "custom_mi355x.sh").write_text("#!/bin/sh\n")
        new_repo, new_scripts = tmp_path / "fw", tmp_path / "sc"
        new_repo.mkdir()
        new_scripts.mkdir()
        (new_scripts / "custom_mi355x.sh").write_text("#!/bin/sh\n")
        monkeypatch.setenv("FRAMEWORK_REPO_PATH", "")
        monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", "")
        state = SharedState(
            session_id="s",
            framework="custom",
            framework_repo_path=str(tmp_path / "old-fw"),
            bypass_scripts_dir=str(archived_scripts),
            benchmark_backend="bypass",
        )
        args = self._args()
        args.framework_path = str(new_repo)
        args.benchmark_scripts_dir = str(new_scripts)

        _restore_operator_supplied_paths_from_state(args, state)
        _apply_operator_supplied_paths(args, "custom")
        assert os.environ["HYPERLOOM_BYPASS_SCRIPTS_DIR"] == str(new_scripts.resolve())
        assert os.environ["FRAMEWORK_REPO_PATH"] == str(new_repo.resolve())


class TestCustomAdapterMeasurement:
    """Handoff probe: custom/native adapter emits nonzero throughput, no harness error."""

    def test_custom_scriptable_run_emits_nonzero_throughput(self, tmp_path, monkeypatch):
        import yaml
        from hyperloom.orchestrator.actions.executors import bypass_runner

        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "custom_mi355x.sh").write_text(
            "#!/bin/bash\n"
            'cat > "$RESULT_DIR/$RESULT_FILENAME.json" <<JSON\n'
            '{"framework": "custom", "workload_kind": "scriptable", '
            '"throughput_unit": "fps", "output_throughput": 1.29, '
            '"quality_gate": {"passed": true}}\n'
            "JSON\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HYPERLOOM_BYPASS_SCRIPTS_DIR", str(scripts))
        monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "bypass")

        cfg = {
            "benchmark": {
                "framework": "custom",
                "model": "/models/custom",
                "runner_type": "mi355x",
                "run_mode": "local",
                "timeout_seconds": 60,
                "workload_kind": "scriptable",
                "envs": {"TP": 8},
            }
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        rc = bypass_runner.run_benchmark(cfg_path, tmp_path / "out")
        assert rc == 0
        ws = sorted((tmp_path / "out").glob("benchmark_custom_*"))[-1]
        rep = json.loads((ws / "benchmark_report.json").read_text(encoding="utf-8"))
        assert rep["success"] is True
        assert float(rep["throughput"]["output_throughput"]) == pytest.approx(1.29)

    def test_an_inert_env_variant_measures_without_a_harness_error(self, tmp_path, monkeypatch):
        """A default-off extra_env must still produce a valid measurement (probe 3)."""
        import asyncio
        import subprocess
        import sys
        from unittest.mock import patch

        from hyperloom.orchestrator.actions.executors._grid_runner import (
            GridVariant,
            run_grid,
        )

        base = tmp_path / "base.yaml"
        base.write_text(
            yaml.safe_dump(
                {
                    "benchmark": {
                        "framework": "custom",
                        "model": "/models/custom",
                        "workload_kind": "scriptable",
                        "run_mode": "local",
                        "envs": {"TP": 8, "HYWORLD_HOIST": "0"},
                        "timeout_seconds": 60,
                    }
                }
            )
        )

        def _ok(cmd, *a, **k):
            slot = Path(cmd[cmd.index("--output-dir") + 1])
            ws = slot / "benchmark_custom_20260101_000000"
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "benchmark_report.json").write_text(
                json.dumps(
                    {
                        "success": True,
                        "framework": "custom",
                        "throughput": {
                            "output_throughput": 1.29,
                            "completed_requests": 1,
                            "duration_seconds": 10.0,
                        },
                    }
                )
            )
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        monkeypatch.setenv("INFERENCE_OPTIMIZER_RUN_GRID_WARMUP", "0")
        with patch(
            "hyperloom.orchestrator.actions.executors._grid_runner.run_with_session_kill",
            side_effect=_ok,
        ):
            results = asyncio.run(
                run_grid(
                    base_yaml_path=base,
                    base_extra_args="",
                    grid=[
                        GridVariant(name="baseline"),
                        GridVariant(
                            name="inert-off",
                            extra_envs={"HYWORLD_HOIST": "0"},
                        ),
                    ],
                    output_root=tmp_path / "out",
                    magpie_python=sys.executable,
                    variant_timeout_sec=10,
                    gpu_type="mi355x",
                )
            )
        assert [r.status for r in results] == ["succeeded", "succeeded"]
        assert all(not r.error_class for r in results)
        assert not list((tmp_path / "out").glob("variant_*/abort_reason.json"))
