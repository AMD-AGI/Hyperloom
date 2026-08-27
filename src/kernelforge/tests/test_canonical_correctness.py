# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""The task's declared suite, judged the way the arena judges it."""

from __future__ import annotations

import asyncio
import sys
import textwrap

import pytest
import yaml

from kernelforge.loop.canonical_correctness import (
    ARENA_DEFAULT_COMPILE_TIMEOUT_SEC,
    ARENA_DEFAULT_CORRECTNESS_TIMEOUT_SEC,
    accept_candidate,
)


def _python(*commands: str) -> list[str]:
    return [f"{sys.executable} -c {command!r}" for command in commands]


def _config(workspace, *commands: str, **settings) -> None:
    """Declare a task whose compilation is a no-op and whose Step 2 is ``commands``.

    The arena fails a task that declares no ``compile_command`` at all, so every
    workspace a correctness-focused test builds still needs one that passes.
    """
    document = {
        "compile_command": _python("pass"),
        "correctness_command": _python(*commands),
        **settings,
    }
    workspace.joinpath("config.yaml").write_text(yaml.safe_dump(document))


def _run(workspace, *, timeout_cap_sec: int = 60):
    return asyncio.run(
        accept_candidate(
            str(workspace),
            timeout_cap_sec=timeout_cap_sec,
            candidate_label="test candidate",
        )
    )


def test_passing_suite_passes_the_gate(tmp_path):
    _config(tmp_path, "print('all cases PASS')")

    result = _run(tmp_path)

    assert result.passed is True
    assert result.unverified_reason == ""


def test_non_zero_exit_fails_the_gate(tmp_path):
    # The mla-decode shape: the task runner asserts its own tolerance and dies,
    # and the number it names is the only thing that says what to fix.
    _config(
        tmp_path,
        "raise AssertionError('normalized max err 0.02468 too high')",
    )

    result = _run(tmp_path)

    assert result.passed is False
    assert "exited 1" in result.detail
    assert "0.02468" in result.output


def test_reported_failure_in_output_fails_the_gate_despite_exit_zero(tmp_path):
    _config(tmp_path, "print('mla-decode-bs64-kv8192: FAILED')")

    result = _run(tmp_path)

    assert result.passed is False
    assert "reported failure" in result.detail
    assert "mla-decode-bs64-kv8192" in result.output


def test_a_pass_anywhere_in_the_output_is_not_a_reported_failure(tmp_path):
    _config(tmp_path, "print('failed: 0  passed: 4')")

    result = _run(tmp_path)

    assert result.passed is True


def test_every_declared_command_must_pass(tmp_path):
    _config(
        tmp_path,
        "print('all cases PASS')",
        "import sys; sys.exit(1)",
    )

    result = _run(tmp_path)

    assert result.passed is False
    assert "exited 1" in result.detail


def test_missing_config_leaves_the_candidate_unverified_rather_than_failed(tmp_path):
    result = _run(tmp_path)

    assert result.passed is True
    assert "ships no config.yaml" in result.unverified_reason


def test_config_without_a_correctness_command_fails_the_gate(tmp_path):
    # The ``compile_command`` is well-formed filler: without it the gate would
    # stop on Step 1's declaration and never reach the case this test names.
    tmp_path.joinpath("config.yaml").write_text('compile_command:\n  - "true"\n')

    result = _run(tmp_path)

    assert result.passed is False
    assert result.unverified_reason == ""
    assert "declares no 'correctness_command'" in result.detail


def test_unreadable_config_fails_the_gate(tmp_path):
    tmp_path.joinpath("config.yaml").write_text("correctness_command: [unterminated\n")

    result = _run(tmp_path)

    assert result.passed is False
    assert "could not be read" in result.detail


def test_a_bare_string_command_is_refused_rather_than_run_per_character(tmp_path):
    tmp_path.joinpath("config.yaml").write_text(
        'compile_command:\n  - "true"\ncorrectness_command: python3 task_runner.py\n'
    )

    result = _run(tmp_path)

    assert result.passed is False
    assert "list of shell command strings" in result.detail


def test_a_non_numeric_declared_timeout_fails_the_gate(tmp_path):
    _config(tmp_path, "pass", correctness_timeout="soon")

    result = _run(tmp_path)

    assert result.passed is False
    assert "not a number of seconds" in result.detail


def test_timeout_fails_the_gate_and_is_clamped_to_the_stage_ceiling(tmp_path):
    _config(
        tmp_path,
        "import time; time.sleep(30)",
        correctness_timeout=600,
    )

    result = _run(tmp_path, timeout_cap_sec=1)

    assert result.passed is False
    assert result.outcome == "timeout"
    assert "timed out after 1s" in result.detail


def test_declared_timeout_binds_when_it_is_below_the_stage_ceiling(tmp_path):
    _config(
        tmp_path,
        "import time; time.sleep(30)",
        correctness_timeout=1,
    )

    result = _run(tmp_path, timeout_cap_sec=600)

    assert result.passed is False
    assert "timed out after 1s" in result.detail


def test_arena_default_timeout_is_the_one_the_arena_applies():
    # AgentKernelArena src/evaluator.py::_DEFAULT_CORRECTNESS_TIMEOUT_S. A task
    # that declares nothing is judged under this, so forge must reproduce it.
    assert ARENA_DEFAULT_CORRECTNESS_TIMEOUT_SEC == 3600


def test_arena_default_compile_timeout_is_the_one_the_arena_applies():
    # AgentKernelArena src/evaluator.py::_DEFAULT_COMPILE_TIMEOUT_S. It is a
    # separate constant from the correctness one, read from a separate key.
    assert ARENA_DEFAULT_COMPILE_TIMEOUT_SEC == 3600


@pytest.mark.parametrize("declared", ["correctness_command: []", "correctness_command:"])
def test_an_empty_correctness_command_fails_the_gate(tmp_path, declared):
    tmp_path.joinpath("config.yaml").write_text(f'compile_command:\n  - "true"\n{declared}\n')

    result = _run(tmp_path)

    assert result.passed is False
    assert "declares no 'correctness_command'" in result.detail


# --- The arena's Step 1, which forge reaches before Step 2 -------------------


def _two_step_config(workspace, *, compile_: list[str], correctness: list[str], **settings) -> None:
    document = {
        "compile_command": compile_,
        "correctness_command": correctness,
        **settings,
    }
    workspace.joinpath("config.yaml").write_text(yaml.safe_dump(document))


def test_a_failing_compile_fails_the_gate_before_correctness_is_run(tmp_path):
    marker = tmp_path / "correctness_ran"
    _two_step_config(
        tmp_path,
        compile_=_python("raise SystemExit(2)"),
        correctness=_python(f"open({str(marker)!r}, 'w').close()"),
    )

    result = _run(tmp_path)

    assert result.passed is False
    assert "compilation" in result.detail
    assert "exited 2" in result.detail
    # The arena stops the whole evaluation at Step 1; a kernel that does not
    # build has nothing for Step 2 to measure.
    assert not marker.exists()


def test_a_passing_compile_followed_by_a_failing_correctness_names_step_two(tmp_path):
    marker = tmp_path / "compiled"
    _two_step_config(
        tmp_path,
        compile_=_python(f"open({str(marker)!r}, 'w').close()"),
        correctness=_python("raise AssertionError('rel_max 0.031 exceeds 0.02')"),
    )

    result = _run(tmp_path)

    assert marker.exists()
    assert result.passed is False
    assert result.detail.startswith("correctness:")
    assert "compilation" not in result.detail
    assert "0.031" in result.output


def test_the_two_steps_are_distinguishable_from_the_detail_alone(tmp_path):
    _two_step_config(
        tmp_path,
        compile_=_python("raise SystemExit(1)"),
        correctness=_python("raise SystemExit(1)"),
    )
    failed_compile = _run(tmp_path)

    _two_step_config(
        tmp_path,
        compile_=_python("pass"),
        correctness=_python("raise SystemExit(1)"),
    )
    failed_correctness = _run(tmp_path)

    assert failed_compile.detail.startswith("compilation:")
    assert failed_correctness.detail.startswith("correctness:")


def test_a_passing_gate_reports_both_steps(tmp_path):
    _two_step_config(
        tmp_path,
        compile_=_python("pass"),
        correctness=_python("print('all cases PASS')"),
    )

    result = _run(tmp_path)

    assert result.passed is True
    assert "compilation" in result.detail
    assert "correctness" in result.detail


def test_config_without_a_compile_command_fails_the_gate(tmp_path):
    # evaluate_compilation returns (False, "No compile_command specified") when
    # the key is absent -- a failure, not a skip, unlike anything else there.
    tmp_path.joinpath("config.yaml").write_text("correctness_command:\n  - true\n")

    result = _run(tmp_path)

    assert result.passed is False
    assert result.unverified_reason == ""
    assert "declares no 'compile_command'" in result.detail


def test_a_compile_command_that_prints_failure_but_exits_zero_passes(tmp_path):
    # evaluate_compilation judges by exit status alone; only evaluate_correctness
    # scans the text. A warning naming a failed probe must not reject the build.
    _two_step_config(
        tmp_path,
        compile_=_python("print('hipcc: note: fail-fast codegen path disabled')"),
        correctness=_python("print('all cases PASS')"),
    )

    result = _run(tmp_path)

    assert result.passed is True


def test_compile_timeout_is_read_from_its_own_key(tmp_path):
    _two_step_config(
        tmp_path,
        compile_=_python("import time; time.sleep(30)"),
        correctness=_python("pass"),
        compile_timeout=1,
        correctness_timeout=600,
    )

    result = _run(tmp_path, timeout_cap_sec=600)

    assert result.passed is False
    assert result.outcome == "timeout"
    assert result.detail.startswith("compilation:")
    assert "timed out after 1s" in result.detail


def test_a_generous_compile_timeout_does_not_relax_the_correctness_one(tmp_path):
    _two_step_config(
        tmp_path,
        compile_=_python("pass"),
        correctness=_python("import time; time.sleep(30)"),
        compile_timeout=600,
        correctness_timeout=1,
    )

    result = _run(tmp_path, timeout_cap_sec=600)

    assert result.passed is False
    assert result.detail.startswith("correctness:")
    assert "timed out after 1s" in result.detail


def test_compile_timeout_is_clamped_to_the_stage_ceiling(tmp_path):
    _two_step_config(
        tmp_path,
        compile_=_python("import time; time.sleep(30)"),
        correctness=_python("pass"),
        compile_timeout=600,
    )

    result = _run(tmp_path, timeout_cap_sec=1)

    assert result.passed is False
    assert result.outcome == "timeout"
    assert "compilation" in result.detail
    assert "timed out after 1s" in result.detail


def test_a_non_numeric_declared_compile_timeout_fails_the_gate(tmp_path):
    _two_step_config(
        tmp_path,
        compile_=_python("pass"),
        correctness=_python("pass"),
        compile_timeout="soon",
    )

    result = _run(tmp_path)

    assert result.passed is False
    assert "'compile_timeout'" in result.detail
    assert "not a number of seconds" in result.detail


def test_a_kernel_that_only_builds_at_the_full_shape_fails_the_gate(tmp_path):
    """The tilelang_dsa_sparse_mla_glm5 incident, reduced to a hermetic stub.

    The agent made the launch geometry sweepable and guarded it with an
    assertion that holds for every shape it measured. The task's compile step
    shrinks ``num_seqs`` to keep the smoke test cheap, ``inner_iter`` collapses
    to 1 there, and the assertion fires for every knob value -- which forge's
    thirteen iterations, all run at the full shape, never saw.
    """
    kernel = tmp_path / "kernel_stub.py"
    kernel.write_text(
        textwrap.dedent(
            """
            import sys

            num_seqs = int(sys.argv[1])
            block_per_cu, cu, ni = 2, 256, 64
            inner_iter = max(1, int(num_seqs * ni / (cu * block_per_cu)))
            assert inner_iter >= 2, (
                f"inner_iter=={inner_iter} flips _q_in_shared and blows up LDS; "
                f"reduce BLOCK_PER_CU (={block_per_cu})"
            )
            print("all cases PASS")
            """
        )
    )
    _two_step_config(
        tmp_path,
        # What scripts/task_runner.py compile does: shrink the case to num_seqs=2.
        compile_=[f"{sys.executable} {kernel} 2"],
        correctness=[f"{sys.executable} {kernel} 64"],
    )

    result = _run(tmp_path)

    assert result.passed is False
    assert result.detail.startswith("compilation:")
    assert "blows up LDS" in result.output
