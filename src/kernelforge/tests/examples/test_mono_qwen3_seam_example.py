# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The monokernel example's checks that need no GPU: the contract on its editable sources, the
judge's bf16 steps, and the acceptance suite it declares."""

from __future__ import annotations

import importlib.util
import shlex
import subprocess
import sys
import textwrap

import pytest
import yaml

from kernelforge.config.resources import resource_path

EXAMPLE = resource_path("examples") / "mono-qwen3-seam-gfx942-forge-loop"
TASK = EXAMPLE / "task"


def _load(name: str, directory: str = "benchmark/forge_mono"):
    path = TASK / directory / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_mono_example_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --- contract.py: the static rules on qwen3_seam/ ---


def test_the_pristine_launch_meets_the_static_rules():
    assert _load("contract").source_violations() == []


def test_every_way_around_the_guard_is_named(tmp_path):
    package = tmp_path / "qwen3_seam"
    package.mkdir()
    package.joinpath("waits.py").write_text(
        textwrap.dedent(
            """
            from sgl_amd_mono.common.epoch import spin_until
            from sgl_amd_mono.common import sync
            from flydsl.expr import rocdl

            def poll(load):
                v = load()
                while v == 0:
                    rocdl.s_sleep(1)
                    v = load()
                return sync._spin(load, lambda w: w == 0)
            """
        )
    )

    found = _load("contract").source_violations(package)

    assert [line.split(": ", 1)[0] for line in found] == [
        "qwen3_seam/waits.py:2",
        "qwen3_seam/waits.py:8",
        "qwen3_seam/waits.py:9",
        "qwen3_seam/waits.py:11",
    ]


def test_a_while_in_any_added_module_is_refused(tmp_path):
    package = tmp_path / "qwen3_seam" / "stages"
    package.mkdir(parents=True)
    package.joinpath("reduce.py").write_text("def f(n):\n    while n:\n        n -= 1\n")

    assert _load("contract").source_violations(tmp_path / "qwen3_seam") == [
        "qwen3_seam/stages/reduce.py:2: a while loop; a wait must be a guarded Mailbox.poll"
    ]


# --- judge.py: res_out bit for bit, x within bf16 steps ---


def _bf16(torch, *bits: int):
    return torch.tensor(bits, dtype=torch.int16).view(torch.bfloat16)


def test_bf16_steps_count_across_zero():
    torch = pytest.importorskip("torch")
    judge = _load("judge")
    plus_zero, minus_zero = _bf16(torch, 0x0000), _bf16(torch, -0x8000)
    smallest, minus_smallest = _bf16(torch, 0x0001), _bf16(torch, -0x7FFF)
    one, after_one = _bf16(torch, 0x3F80), _bf16(torch, 0x3F81)

    assert int(judge.ulp_distance(plus_zero, minus_zero)) == 0
    assert int(judge.ulp_distance(smallest, minus_smallest)) == 2
    assert int(judge.ulp_distance(one, after_one)) == 1
    assert int(judge.ulp_distance(-one, -after_one)) == 1


def _answer(torch, rows: int = 4):
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(rows, 5120, generator=gen).to(torch.bfloat16)
    res_out = torch.randn(rows, 5120, generator=gen).to(torch.bfloat16)
    return x, res_out


def test_an_identical_answer_passes():
    torch = pytest.importorskip("torch")
    x, res_out = _answer(torch)

    passed, metrics = _load("judge").judge((x, res_out), (x.clone(), res_out.clone()), [0, 0, 0])

    assert passed
    assert metrics == {"res_exact": 1, "x_finite": 1, "x_ulp_max": 0, "x_ulp_frac": 0.0, "errors": "0-0-0"}


def test_one_bit_of_res_out_fails():
    torch = pytest.importorskip("torch")
    x, res_out = _answer(torch)
    off = res_out.clone()
    off.view(torch.int16)[1, 7] += 1

    passed, metrics = _load("judge").judge((x, off), (x, res_out), [0, 0, 0])

    assert not passed
    assert metrics["res_exact"] == 0


def test_x_passes_a_few_single_steps_and_fails_a_double_step():
    torch = pytest.importorskip("torch")
    judge = _load("judge")
    x, res_out = _answer(torch)
    stepped = x.clone()
    stepped.view(torch.int16)[0, :3] += 1
    doubled = x.clone()
    doubled.view(torch.int16)[0, 0] += 2

    assert judge.judge((stepped, res_out), (x, res_out), [0, 0, 0])[0]
    assert not judge.judge((doubled, res_out), (x, res_out), [0, 0, 0])[0]


def test_x_fails_when_too_many_elements_are_a_step_off():
    torch = pytest.importorskip("torch")
    judge = _load("judge")
    x, res_out = _answer(torch)
    drifted = x.clone()
    count = int(judge.X_ULP_FRAC_MAX * x.numel()) + 1
    drifted.view(torch.int16).view(-1)[:count] += 1

    passed, metrics = judge.judge((drifted, res_out), (x, res_out), [0, 0, 0])

    assert not passed
    assert metrics["x_ulp_max"] == 1


def test_a_set_error_word_fails():
    torch = pytest.importorskip("torch")
    x, res_out = _answer(torch)

    passed, metrics = _load("judge").judge((x, res_out), (x, res_out), [0, 1, 0])

    assert not passed
    assert metrics["errors"] == "0-1-0"


# --- test/ranks.py: the four ranks of a test, each its own process ---


def _fake_ranks(monkeypatch, tmp_path, body: str):
    """``ranks`` with a stand-in rank script, so each rank's exit and output are the test's to choose."""
    ranks = _load("ranks", "test")
    worker = tmp_path / "rank.py"
    worker.write_text("import os, sys, time\nrank = int(os.environ['RANK'])\n" + textwrap.dedent(body))
    monkeypatch.setattr(ranks, "WORKER", worker)
    return ranks


def test_each_rank_is_seen_with_its_own_exit_and_output(monkeypatch, tmp_path):
    ranks = _fake_ranks(
        monkeypatch,
        tmp_path,
        """
        print(f"LAUNCH {time.time()}", flush=True)
        print('RESULT {"rank": %d, "passed": true}' % rank, flush=True)
        if rank == 1:
            sys.exit(7)
        print("FINISHED", flush=True)
        """,
    )

    runs = ranks.launch_ranks("parity", timeout=60.0)

    assert [run.returncode for run in runs] == [0, 7, 0, 0]
    assert [run.finished for run in runs] == [True, False, True, True]
    assert [run.verdicts() for run in runs] == [[{"rank": r, "passed": True}] for r in range(ranks.TP)]
    assert all(run.launched is not None and run.exited >= run.launched for run in runs)


def test_a_rank_still_running_at_the_timeout_is_killed_and_named(monkeypatch, tmp_path):
    ranks = _fake_ranks(monkeypatch, tmp_path, "if rank == 3:\n    time.sleep(600)\n")

    with pytest.raises(AssertionError, match=r"rank\(s\) \[3\] still running after 2.0 s"):
        ranks.launch_ranks("fault", timeout=2.0)


# --- config.yaml and run_example.sh ---


def test_the_acceptance_suite_runs_the_task_tests():
    suite = yaml.safe_load(TASK.joinpath("config.yaml").read_text())

    for step in ("compile", "correctness"):
        commands = suite[f"{step}_command"]
        assert isinstance(commands, list) and commands
        assert isinstance(suite[f"{step}_timeout"], int) and suite[f"{step}_timeout"] > 0
        for command in commands:
            for word in shlex.split(command):
                if word.startswith("test/"):
                    assert TASK.joinpath(word).is_file(), f"{step}: {word} does not exist"
    named = " ".join(suite["compile_command"] + suite["correctness_command"])
    for test in sorted(TASK.joinpath("test").glob("test_*.py")):
        assert f"test/{test.name}" in named, f"{test.name} is in no acceptance command"


def test_the_protected_directories_take_the_singular_names():
    """The repository's pytest collects every ``src/**/tests`` (these tests need four GPUs), and its
    ``.gitignore`` drops every ``benchmarks/``; forge protects the singular names too."""
    for name in ("test", "benchmark"):
        assert TASK.joinpath(name).is_dir()
        assert not TASK.joinpath(f"{name}s").exists()


def test_run_example_parses():
    subprocess.run(["bash", "-n", str(EXAMPLE / "run_example.sh")], check=True)
