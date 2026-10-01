# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic checks: the test suite on both trees, and the A/B comparison."""

from __future__ import annotations

import json
import re

from meta_rsi.compare_ab import compare as compare_arms
from meta_rsi.rsi.pipeline import RoundContext, StepFailed
from meta_rsi.rsi.steps.ab import arm_dir
from meta_rsi.rsi.steps.levers import ensure_worktrees

SHOWN_FAILURES = 20
SUMMARY_LINE = re.compile(r"^(?:FAILED|ERROR) (\S+)", re.M)
PYTEST_VERDICT_EXITS = (0, 1)  # pytest.ExitCode.OK and TESTS_FAILED; 2-5 mean the session never judged every test


def failed_tests(pytest_output: str) -> set[str]:
    """Node ids that pytest's short test summary (``-rfE``) lists as failed or errored."""
    return set(SUMMARY_LINE.findall(pytest_output))


def suite(ctx: RoundContext) -> dict:
    """Run the configured suite on base and candidate in the same environment; flag candidate-only failures."""
    ensure_worktrees(ctx)
    checks = ctx.config.checks
    failed = {}
    for which in ("base", "candidate"):
        tree = ctx.worktree(which)
        env = {**ctx.agent_env(), "PYTHONPATH": str(tree / "src")}
        cmd = [str(checks.python), "-m", "pytest", *checks.pytest_args, "-rfE"]
        log = ctx.path("logs", f"suite-{which}.log")
        proc = ctx.runner.run(cmd, cwd=tree, env=env, log=log, check=False, timeout=60 * checks.timeout_min)
        if proc.returncode not in PYTEST_VERDICT_EXITS:
            raise StepFailed(
                f"pytest exited {proc.returncode} on the {which} tree without judging every test "
                f"(interrupted collection, usage error or nothing collected); see {log}"
            )
        failed[which] = failed_tests(proc.stdout)
    new = sorted(failed["candidate"] - failed["base"])
    summary = {
        "base_failures": len(failed["base"]),
        "candidate_failures": len(failed["candidate"]),
        "candidate_only": new,
    }
    ctx.path("suite", "suite.json").write_text(json.dumps(summary, indent=1))
    if new and not checks.allow_new_failures:
        shown = "\n".join(new[:SHOWN_FAILURES])
        raise StepFailed(f"{len(new)} tests fail only on the candidate:\n{shown}")
    return summary


def compare(ctx: RoundContext) -> dict:
    """Compare every arm with the control (the first arm); the rule's outcome is advisory."""
    arms = ctx.config.ab.arms
    control, verdicts = arms[0], {}
    for arm in arms[1:]:
        result = compare_arms(arm_dir(ctx, control.name), arm_dir(ctx, arm.name), ctx.config.router_log)
        pair = f"{control.name}_vs_{arm.name}"
        ctx.path("results", f"{pair}.json").write_text(json.dumps(result, indent=1, default=str))
        verdicts[pair] = result["verdict"]
    return {"verdicts": verdicts}
