# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""`hyperloom script prepare` without a model: the parts that need no torch.

The end-to-end tests in ``test_script_workload.py`` skip without torch, which only
the ``script-harness`` CI job installs. Everything here is argument handling, config
layout and refusals, so it runs in the main matrix too.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

from hyperloom.common.optimization_scope import aoti_config
from hyperloom.inference_optimizer import script_workload as sw


def _args(tmp_path: Path, **overrides):
    script = overrides.pop("script", None) or tmp_path / "model_run.py"
    if not script.exists():
        script.write_text("print('hello')\n")
    argv = [
        "prepare",
        "--workspace",
        str(tmp_path / "ws"),
        "--script",
        str(script),
        "--cwd",
        str(tmp_path),
        "--python",
        sys.executable,
        "--error-limit",
        "0.01",
    ]
    for flag, value in overrides.items():
        argv += [f"--{flag.replace('_', '-')}", str(value)]
    args = sw._parser().parse_args(argv[1:])
    args.module = args.module or f"{args.script.stem}_opt"
    args.python = sw._interpreter(args.python)
    return args


# --- tolerance ------------------------------------------------------------------------


def test_a_fixed_limit_and_a_calibrated_one_are_different_shapes(tmp_path):
    fixed = sw._tolerance(_args(tmp_path))
    assert fixed == {"limit": 0.01}

    calibrated = _args(tmp_path)
    calibrated.error_limit = None
    calibrated.calibrate_autocast = "bfloat16"
    assert sw._tolerance(calibrated) == {"autocast": "bfloat16", "headroom": 1.15, "floor": 1e-4}


def test_the_summary_names_the_units_of_the_limits(tmp_path):
    report = {"calls": 2, "rel_l2_limits": [0.01, 0.01]}
    assert sw._recorded_summary(report, _args(tmp_path)) == "Recorded 2 calls; error limits [0.01, 0.01] (relative L2)"

    scored = _args(tmp_path, scorer=tmp_path / "scorer.py")
    assert sw._recorded_summary(report, scored).endswith("(in the units of scorer.py)")

    calibrated = _args(tmp_path)
    calibrated.error_limit, calibrated.calibrate_autocast = None, "float16"
    assert sw._recorded_summary(report, calibrated).endswith("(relative L2, calibrated from float16 autocast)")


def test_rel_l2_is_the_old_spelling_of_error_limit():
    """Kept so a command line written against the original flag still runs."""
    base = ["--workspace", "w", "--script", "s.py"]
    assert sw._parser().parse_args([*base, "--rel-l2", "0.5"]).error_limit == 0.5
    assert sw._parser().parse_args([*base, "--error-limit", "0.5"]).error_limit == 0.5


def test_a_limit_and_a_calibration_cannot_both_be_given():
    with pytest.raises(SystemExit):
        sw._parser().parse_args(
            ["--workspace", "w", "--script", "s.py", "--error-limit", "0.1", "--calibrate-autocast", "bfloat16"]
        )


# --- the frozen benchmark directory ---------------------------------------------------


def test_prepare_freezes_the_scorer_and_records_its_digest(tmp_path):
    scorer = tmp_path / "my_scorer.py"
    scorer.write_text("def score(candidate, reference):\n    return 0.0\n")
    benchmark = tmp_path / "bench"
    benchmark.mkdir()

    digest = sw._copy_scorer(benchmark, scorer)

    assert digest == hashlib.sha256(scorer.read_bytes()).hexdigest()
    assert (benchmark / "scorer.py").read_bytes() == scorer.read_bytes()


def test_no_scorer_records_no_digest(tmp_path):
    benchmark = tmp_path / "bench"
    benchmark.mkdir()

    assert sw._copy_scorer(benchmark, None) is None
    assert not (benchmark / "scorer.py").exists()


def test_the_benchmark_directory_carries_the_scope_config_and_entrypoint(tmp_path):
    args = _args(tmp_path)
    benchmark, module_file = tmp_path / "bench", tmp_path / "ws" / "optimization" / "model_run_opt.py"

    sw._write_benchmark(benchmark, module_file, args)

    config = json.loads((benchmark / "script_workload.json").read_text())
    assert config["script"] == str(args.script.resolve())
    assert config["script_sha256"] == hashlib.sha256(args.script.read_bytes()).hexdigest()
    assert config["tolerance"] == {"limit": 0.01}
    assert config["scorer_sha256"] is None

    scope = json.loads((benchmark / "optimization_scope.json").read_text())
    assert scope == {"schema_version": 1, "file": str(module_file), "function": "apply"}
    # No aoti block: the operator adds one by hand to ask for an export.
    assert "aoti" not in scope

    entrypoint = benchmark / "benchmark.sh"
    assert entrypoint.stat().st_mode & 0o111, "the entrypoint must be executable"
    assert "harness.py" in entrypoint.read_text()
    # The name carries no architecture, so the launcher resolves it through the single-.sh
    # fallback -- which returns nothing at all once a second one is there.
    assert list(benchmark.glob("*.sh")) == [entrypoint]


# --- import-order refusals ------------------------------------------------------------


def _order(**overrides) -> dict:
    base = {
        "torch_imported": False,
        "gpu_initialized": False,
        "compiler_imported": [],
        "module_site": "",
        "torch_site": "",
    }
    return {**base, **overrides}


def test_importing_the_module_after_the_gpu_is_initialized_is_refused(tmp_path):
    args = _args(tmp_path)
    order = _order(gpu_initialized=True, module_site="model_run.py:9")

    with pytest.raises(SystemExit, match=r"after the GPU was initialized"):
        sw._check_import_order(order, args)


def test_importing_the_module_after_the_compiler_is_refused(tmp_path):
    args = _args(tmp_path)
    order = _order(compiler_imported=["torch._dynamo"], module_site="model_run.py:3")

    with pytest.raises(SystemExit, match=r"torch\._dynamo"):
        sw._check_import_order(order, args)


def test_importing_the_module_after_torch_only_warns(tmp_path, capsys):
    args = _args(tmp_path)
    order = _order(torch_imported=True, module_site="model_run.py:3", torch_site="model_run.py:2")

    sw._check_import_order(order, args)

    err = capsys.readouterr().err
    assert "warning:" in err and "model_run.py:2" in err


def test_a_clean_import_order_says_nothing(tmp_path, capsys):
    sw._check_import_order(_order(), _args(tmp_path))
    assert capsys.readouterr().err == ""


# --- refusals before anything is written ----------------------------------------------


def test_preparing_an_existing_workspace_is_refused(tmp_path):
    args = _args(tmp_path)
    (args.workspace / "benchmark").mkdir(parents=True)

    with pytest.raises(SystemExit, match="already prepared"):
        sw.prepare(args)


@pytest.mark.parametrize("bad", ["not an identifier", "1nvalid", "model-run"])
def test_a_module_name_that_is_not_an_identifier_is_refused(tmp_path, bad):
    with pytest.raises(SystemExit, match="must be Python identifiers"):
        sw.main(
            [
                "prepare",
                "--workspace",
                str(tmp_path / "ws"),
                "--script",
                str(tmp_path / "s.py"),
                "--error-limit",
                "0.1",
                "--module",
                bad,
            ]
        )


def test_an_empty_module_name_falls_back_to_the_script_s_own(tmp_path):
    """``--module ''`` is not a refusal: the name is derived from the script instead."""
    script = tmp_path / "model_run.py"
    script.write_text("x = 1\n")
    args = sw._parser().parse_args(
        ["--workspace", str(tmp_path / "ws"), "--script", str(script), "--error-limit", "0.1", "--module", ""]
    )
    assert (args.module or f"{args.script.stem}_opt") == "model_run_opt"


def test_fewer_timed_passes_than_held_out_inputs_is_refused(tmp_path):
    script = tmp_path / "s.py"
    script.write_text("x = 1\n")
    with pytest.raises(SystemExit, match="--timed-passes must be at least --held-out"):
        sw.main(
            [
                "prepare",
                "--workspace",
                str(tmp_path / "ws"),
                "--script",
                str(script),
                "--error-limit",
                "0.1",
                "--held-out",
                "8",
                "--timed-passes",
                "2",
            ]
        )


def test_a_command_that_is_neither_prepare_nor_check_prints_usage(capsys):
    assert sw.main(["measure"]) == 2
    assert "usage: hyperloom script {prepare,check}" in capsys.readouterr().err


# --- partial-workspace cleanup --------------------------------------------------------


def test_a_failed_prepare_keeps_the_recording_log_and_removes_the_rest(tmp_path, capsys):
    workspace = tmp_path / "ws"
    benchmark = workspace / "benchmark"
    benchmark.mkdir(parents=True)
    (workspace / "optimization").mkdir()
    (benchmark / "record_stderr.log").write_text("ImportError: no module named lerobot\n")

    sw._discard_partial_workspace(workspace)

    assert not benchmark.exists() and not (workspace / "optimization").exists()
    kept = workspace / "record_stderr.log"
    assert "lerobot" in kept.read_text(), "the script's own error is what the operator needs"
    assert "record_stderr.log" in capsys.readouterr().err


# --- hyperloom script check -----------------------------------------------------------


def _prepared(tmp_path: Path) -> Path:
    """A workspace shaped like one prepare produced, without running the model."""
    workspace = tmp_path / "ws"
    optimization = workspace / "optimization"
    optimization.mkdir(parents=True)
    module_file = optimization / "model_run_opt.py"
    module_file.write_text("def apply(model):\n    return model\n")
    sw._write_benchmark(workspace / "benchmark", module_file, _args(tmp_path))
    return workspace


def test_check_passes_a_freshly_prepared_workspace(tmp_path, capsys):
    assert sw.check(_prepared(tmp_path)) == 0
    assert "ready to launch" in capsys.readouterr().out


def test_check_refuses_a_directory_that_was_never_prepared(tmp_path, capsys):
    (tmp_path / "ws").mkdir()

    assert sw.check(tmp_path / "ws") == 1
    assert "not a prepared workspace" in capsys.readouterr().err


def test_check_names_a_moved_script_and_its_recorded_key(tmp_path, capsys):
    """The failure a moved workspace otherwise shows as a bare 'No such file or directory'."""
    workspace = _prepared(tmp_path)
    (tmp_path / "model_run.py").unlink()

    assert sw.check(workspace) == 1
    err = capsys.readouterr().err
    assert "the model script is missing" in err and "model_run.py" in err


def test_check_reports_every_problem_in_one_run(tmp_path, capsys):
    """One run has to name them all: fixing one at a time costs a round trip each."""
    workspace = _prepared(tmp_path)
    (tmp_path / "model_run.py").unlink()
    (workspace / "benchmark" / "benchmark.sh").unlink()

    assert sw.check(workspace) == 1
    err = capsys.readouterr().err
    assert "the model script is missing" in err
    assert "no .sh entrypoint" in err
    assert "2 problem(s)" in err


def test_check_catches_a_script_edited_after_prepare_recorded_it(tmp_path, capsys):
    workspace = _prepared(tmp_path)
    (tmp_path / "model_run.py").write_text("print('edited')\n")

    assert sw.check(workspace) == 1
    assert "changed since prepare recorded it" in capsys.readouterr().err


def test_check_refuses_a_second_nameless_entrypoint(tmp_path, capsys):
    """Two .sh files with no board in either name resolve to nothing at launch."""
    workspace = _prepared(tmp_path)
    (workspace / "benchmark" / "other.sh").write_text("#!/bin/sh\n")

    assert sw.check(workspace) == 1
    assert "none names a board" in capsys.readouterr().err


def test_check_accepts_a_board_named_entrypoint_beside_the_prepared_one(tmp_path):
    """custom_<gpu-type>.sh is the name the launcher prefers, so it still resolves."""
    workspace = _prepared(tmp_path)
    (workspace / "benchmark" / "custom_mi355x.sh").write_text("#!/bin/sh\n")

    assert sw.check(workspace) == 0


def test_check_refuses_a_module_outside_the_workspace(tmp_path, capsys):
    workspace = _prepared(tmp_path)
    stray = tmp_path / "elsewhere" / "model_run_opt.py"
    stray.parent.mkdir()
    stray.write_text("def apply(model):\n    return model\n")
    scope = workspace / "benchmark" / "optimization_scope.json"
    scope.write_text(json.dumps({"schema_version": 1, "file": str(stray), "function": "apply"}))

    assert sw.check(workspace) == 1
    assert "outside this workspace" in capsys.readouterr().err


def test_check_is_reachable_as_a_subcommand(tmp_path):
    assert sw.main(["check", "--workspace", str(_prepared(tmp_path))]) == 0


# --- --aoti ---------------------------------------------------------------------------


def _module_file(tmp_path: Path) -> Path:
    """The editable module the scope names; aoti_config refuses a scope whose file is absent."""
    module_file = tmp_path / "ws" / "optimization" / "model_run_opt.py"
    module_file.parent.mkdir(parents=True, exist_ok=True)
    module_file.write_text("def apply(model):\n    return model\n")
    return module_file


def _scope_of(tmp_path: Path, aoti=None) -> dict:
    args = _args(tmp_path)
    args.aoti = aoti
    benchmark = tmp_path / "bench"
    sw._write_benchmark(benchmark, _module_file(tmp_path), args)
    return json.loads((benchmark / "optimization_scope.json").read_text())


def test_without_the_flag_the_scope_stays_at_schema_1(tmp_path):
    scope = _scope_of(tmp_path)
    assert scope["schema_version"] == 1 and "aoti" not in scope


def test_the_skeleton_declares_an_export_at_schema_2(tmp_path):
    scope = _scope_of(tmp_path, aoti=True)
    assert scope["schema_version"] == 2
    assert scope["aoti"]["enabled"] is True and scope["aoti"]["pin_all_others"] is True


def test_the_skeleton_blocks_a_launch_until_it_is_filled_in(tmp_path):
    """The placeholders are the point: a skeleton that validated would run an unchosen range."""
    args = _args(tmp_path)
    args.aoti = True
    benchmark = tmp_path / "bench"
    sw._write_benchmark(benchmark, _module_file(tmp_path), args)

    with pytest.raises(ValueError, match="input must be a non-empty string"):
        aoti_config(str(benchmark))


def test_a_skeleton_with_no_dynamic_dims_is_valid(tmp_path):
    """Replacing the example entry with [] is how an operator says every shape is fixed."""
    benchmark = tmp_path / "bench"
    args = _args(tmp_path)
    args.aoti = True
    sw._write_benchmark(benchmark, _module_file(tmp_path), args)
    scope_path = benchmark / "optimization_scope.json"
    scope = json.loads(scope_path.read_text())
    scope["aoti"]["dynamic_dims"] = []
    scope_path.write_text(json.dumps(scope))

    config = aoti_config(str(benchmark))
    assert config.enabled and config.dynamic_dims == ()


def _declaration(tmp_path: Path, **overrides) -> Path:
    block = {
        "enabled": True,
        "pin_all_others": True,
        "dynamic_dims": [{"input": "input_ids", "dim": 1, "min": 310, "max": 512, "derivation": "sequence length T"}],
        **overrides,
    }
    path = tmp_path / "aoti_scope.json"
    path.write_text(json.dumps(block))
    return path


def test_a_declaration_file_is_merged_and_the_version_bumped(tmp_path):
    scope = _scope_of(tmp_path, aoti=str(_declaration(tmp_path)))

    assert scope["schema_version"] == 2
    assert scope["aoti"]["dynamic_dims"][0]["input"] == "input_ids"


def test_a_declaration_file_is_pinned_by_digest(tmp_path):
    """The declaration the run used stays reproducible, like the script and the scorer."""
    declaration = _declaration(tmp_path)
    args = _args(tmp_path)
    args.aoti = str(declaration)
    benchmark = tmp_path / "bench"
    sw._write_benchmark(benchmark, _module_file(tmp_path), args)

    config = json.loads((benchmark / "script_workload.json").read_text())
    assert config["aoti_sha256"] == hashlib.sha256(declaration.read_bytes()).hexdigest()


def test_a_declaration_file_must_say_whether_it_is_enabled(tmp_path):
    """Defaulting it would mean 'export' here and 'inert' for the same block pasted in by hand."""
    block = json.loads(_declaration(tmp_path).read_text())
    del block["enabled"]
    path = tmp_path / "no_enabled.json"
    path.write_text(json.dumps(block))

    with pytest.raises(SystemExit, match="must say whether it is enabled"):
        sw._aoti_block(str(path))


def test_a_malformed_declaration_file_fails_at_prepare(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json")

    with pytest.raises(SystemExit, match="broken.json"):
        sw._aoti_block(str(path))


def test_a_declaration_file_holding_a_list_is_refused(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[]")

    with pytest.raises(SystemExit, match="JSON object"):
        sw._aoti_block(str(path))


def test_a_bad_field_in_a_declaration_file_fails_before_any_budget_is_spent(tmp_path):
    benchmark = tmp_path / "bench"
    args = _args(tmp_path)
    args.aoti = str(
        _declaration(tmp_path, dynamic_dims=[{"input": "x", "dim": 1, "min": 9, "max": 2, "derivation": "d"}])
    )
    sw._write_benchmark(benchmark, _module_file(tmp_path), args)

    with pytest.raises(SystemExit, match="requires min < max"):
        sw._validate_aoti(benchmark)


# --- the recorded-input listing -------------------------------------------------------


def test_the_listing_reports_a_varying_axis_as_observed(tmp_path):
    report = {"calls": 4, "inputs": [{"name": "input_ids", "shape": [1, 317], "varies": {"1": [310, 512]}}]}

    listing = sw._recorded_inputs(report, tmp_path)

    assert "Inputs recorded (4 calls)" in listing
    assert "input_ids" in listing and "axis 1: observed 310 to 512" in listing
    assert "script check" in listing


def test_the_listing_suggests_nothing_when_no_axis_varied(tmp_path):
    """prepare reports what it recorded; which axes are meant to vary is the operator's call."""
    report = {"calls": 4, "inputs": [{"name": "input_ids", "shape": [1, 317], "varies": {}}]}

    listing = sw._recorded_inputs(report, tmp_path)

    assert "Nothing varied across the recorded calls" in listing
    assert "317" in listing


def test_check_names_each_placeholder_the_skeleton_left(tmp_path, capsys):
    """The listing an operator works from after --aoti, rather than one field per launch attempt."""
    workspace = _prepared(tmp_path)
    scope_path = workspace / "benchmark" / "optimization_scope.json"
    scope = json.loads(scope_path.read_text())
    scope["schema_version"] = 2
    scope["aoti"] = dict(sw._AOTI_SKELETON)
    scope_path.write_text(json.dumps(scope))

    assert sw.check(workspace) == 1
    err = capsys.readouterr().err
    assert "dynamic_dims[0].input is still the placeholder" in err
    assert "dynamic_dims[0].derivation is still the placeholder" in err


def test_check_passes_a_filled_in_declaration(tmp_path):
    workspace = _prepared(tmp_path)
    scope_path = workspace / "benchmark" / "optimization_scope.json"
    scope = json.loads(scope_path.read_text())
    scope["schema_version"] = 2
    scope["aoti"] = json.loads(_declaration(tmp_path).read_text())
    scope_path.write_text(json.dumps(scope))

    assert sw.check(workspace) == 0
