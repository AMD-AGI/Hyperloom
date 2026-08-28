# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Guards for how the pre-release gate releases its runner and its GPUs.

The gate owns the only self-hosted baremetal runner, so a run that will not finish
blocks every later run: while the concurrency group is held, GitHub keeps the newer run
at run-level ``pending`` with an EMPTY jobs array, so it has no job with which to
reclaim anything. Two consequences are pinned here.

A ``preempt`` job on a GitHub-hosted runner used to claim the reclaiming role. It could
never work -- created too late to matter, and unable to reach ``SAFE_API_BASE``, which
is an in-network NodePort: every observed run logged ``[preempt] could not list
workloads; skipping reclaim`` after a 30s curl timeout, having stopped nothing. Nothing
that talks to SaFE may run on a GitHub-hosted runner again.

Teardown instead relies on the old run leaving promptly: the poll fails fast on the
first FAIL (the gate is already decided), leaves still-running workloads up for post-
mortem, and sleeps in short slices so a cancel lands in seconds instead of at the end
of a full poll interval.

There is no way to unit-test the scheduling itself short of running the workflow; these
tests pin the invariants it depends on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_SELF_HOSTED_LABEL = "hyperloom-pre-e2e-baremetal"


def _find_github_dir() -> Path | None:
    """Locate .github/; None when running from an installed wheel."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".github"
        if (candidate / "workflows" / "pre-release-e2e-test.yml").is_file():
            return candidate
    return None


_GITHUB = _find_github_dir()

pytestmark = pytest.mark.skipif(
    _GITHUB is None,
    reason="pre-release gate guards need the source checkout (.github/)",
)


@pytest.fixture(scope="module")
def workflow() -> dict:
    assert _GITHUB is not None
    path = _GITHUB / "workflows" / "pre-release-e2e-test.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def poll_script() -> str:
    assert _GITHUB is not None
    return (_GITHUB / "scripts" / "pre-release-e2e-poll.sh").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def dispatch_script() -> str:
    assert _GITHUB is not None
    return (_GITHUB / "scripts" / "pre-release-e2e-dispatch.sh").read_text(encoding="utf-8")


def test_nothing_that_talks_to_safe_runs_on_a_github_hosted_runner(workflow: dict) -> None:
    """SAFE_API_BASE is an in-network NodePort; a hosted runner can only time out."""
    for name, job in workflow["jobs"].items():
        runs_on = job.get("runs-on")
        if runs_on == _SELF_HOSTED_LABEL:
            continue
        rendered = yaml.safe_dump(job)
        assert "SAFE_API" not in rendered, f"job {name} on {runs_on} reaches for the SaFE API"


def test_the_preempt_job_is_gone(workflow: dict) -> None:
    assert "preempt" not in workflow["jobs"]
    assert workflow["jobs"] == {k: v for k, v in workflow["jobs"].items() if k in {"resolve", "build", "run"}}
    assert workflow["jobs"]["resolve"].get("needs") is None


def test_the_reap_script_is_gone_and_unreferenced() -> None:
    assert _GITHUB is not None
    assert not (_GITHUB / "scripts" / "pre-release-e2e-reap.sh").exists()
    for wf in (_GITHUB / "workflows").glob("*.yml"):
        text = wf.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.lstrip().startswith("#"):
                continue  # the removal rationale is documented in comments
            assert "pre-release-e2e-reap.sh" not in line, f"{wf.name} still runs the reap script"


def test_poll_fails_fast_once_the_gate_is_lost(poll_script: str) -> None:
    """A decided gate must not keep the only runner busy for another 12h."""
    assert 'POLL_FAIL_FAST="${POLL_FAIL_FAST:-1}"' in poll_script
    assert 'if [ "$POLL_FAIL_FAST" = "1" ] && [ "$fail_seen" -eq 1 ]; then' in poll_script
    assert "LEAVE_RUNNING_FILE=" in poll_script
    assert 'VERDICT["$leg"]="SKIP|still running (gate failed; workload left alive)"' in poll_script
    assert "leave_running_wid" in poll_script
    # Every path that records a FAIL has to arm the flag, or fail-fast never triggers.
    assert poll_script.count("fail_seen=1") == 2
    assert poll_script.count('VERDICT["$leg"]="FAIL|') >= 2


def test_poll_sleeps_in_slices_so_a_cancel_lands_quickly(poll_script: str) -> None:
    assert 'POLL_SLEEP_SLICE_S="${POLL_SLEEP_SLICE_S:-5}"' in poll_script
    assert 'sleep "$POLL_SLEEP_SLICE_S"' in poll_script
    assert 'sleep "$POLL_INTERVAL_S"' not in poll_script


def test_abnormal_end_cleanup_respects_leave_running(workflow: dict) -> None:
    """Fail-fast may leave workloads up; cleanup must not stop those wids."""
    steps = workflow["jobs"]["run"]["steps"]
    cleanup = [s for s in steps if "cancelled()" in str(s.get("if", ""))]
    assert cleanup, "the run job lost its cancel/failure cleanup step"
    body = cleanup[0]["run"]
    assert "/stop" in body
    assert "leave_running" in body


def test_dispatch_version_tag_is_unique_per_run(dispatch_script: str) -> None:
    """Reap must distinguish repeated pushes that reuse the same CI_VERSION wheel."""
    assert (
        'VERSION_TAG="$(printf \'%s-%s\' "$CI_VERSION" "${GITHUB_RUN_ID:-local}" | sha1sum | cut -c1-6)"'
        in dispatch_script
    )


def test_poll_exits_when_a_newer_run_is_queued(poll_script: str, workflow: dict) -> None:
    """A pending successor cannot dispatch until this poll releases the runner."""
    assert "superseded_by_newer_run" in poll_script
    assert "mark_superseded_and_exit_poll" in poll_script
    assert "GATE: SUPERSEDED" in poll_script
    assert "dispatch reap will stop" in poll_script
    run_env = yaml.safe_dump(workflow["jobs"]["run"].get("env", {}))
    assert "HEAD_REF:" in run_env


def test_poll_passes_on_clean_terminal_stop_reason_not_gain(poll_script: str) -> None:
    """Gate PASS aligns with optimize CLI exit 0, not cumulative_gain vs TARGET_GAIN."""
    assert "is_clean_stop_reason" in poll_script
    assert "target_reached|global_converged|time_exhausted|max_ticks|sweep_done|conc_sweep_done" in poll_script
    assert "not used to judge PASS" in poll_script
    assert 'echo "PASS|stop=${stop} gain=${gain}%"' in poll_script
    assert "gain=${gain}% < ${TARGET_GAIN}" not in poll_script
    assert "reached target_gain=" not in poll_script
    assert "clean terminal stop_reason" in poll_script
