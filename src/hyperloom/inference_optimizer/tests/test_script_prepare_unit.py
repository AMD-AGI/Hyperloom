# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""`hyperloom script prepare` without a model: the parts that need no torch.

The end-to-end tests in ``test_script_workload.py`` skip wherever torch is absent,
which is every CI runner -- torch is deliberately not a dependency, since a bare
specifier resolves to the CUDA wheel. Everything here is argument handling, config
layout and refusals, so it runs anywhere.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

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
    assert scope == {"schema_version": 2, "file": str(module_file), "function": "apply"}
    # No aoti block: the operator adds one by hand to ask for an export.
    assert "aoti" not in scope

    entrypoint = benchmark / "custom_gfx11.sh"
    assert entrypoint.stat().st_mode & 0o111, "the entrypoint must be executable"
    assert "harness.py" in entrypoint.read_text()


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


def test_a_command_that_is_not_prepare_prints_usage(capsys):
    assert sw.main(["measure"]) == 2
    assert "usage: hyperloom script prepare" in capsys.readouterr().err


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


# --- what the sandbox has to bind -----------------------------------------------------


def test_sandbox_roots_come_from_the_command_line(tmp_path):
    script = tmp_path / "model_run.py"
    script.write_text("x = 1\n")
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
        "0.1",
    ]

    roots = sw.sandbox_roots(argv)

    assert roots.writable == ((tmp_path / "ws").resolve(),)
    assert tmp_path.resolve() in roots.readonly


@pytest.mark.parametrize("argv", [["measure"], ["prepare", "--help"], ["prepare", "--nonsense"]])
def test_sandbox_roots_are_empty_when_there_is_nothing_to_derive(argv):
    roots = sw.sandbox_roots(argv)
    assert roots.writable == () and roots.readonly == ()


def test_benchmark_inputs_are_empty_without_a_config(tmp_path):
    assert sw.benchmark_inputs(tmp_path) == ()


def test_benchmark_inputs_name_the_script_its_cwd_and_its_interpreter(tmp_path):
    benchmark = tmp_path / "bench"
    benchmark.mkdir()
    (benchmark / "script_workload.json").write_text(
        json.dumps({"script": str(tmp_path / "src" / "model_run.py"), "cwd": str(tmp_path), "python": sys.executable})
    )

    inputs = sw.benchmark_inputs(benchmark)

    assert (tmp_path / "src") in inputs and tmp_path in inputs
