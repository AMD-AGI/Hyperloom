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

Teardown instead relies on the old run leaving promptly when superseded: the poll
sleeps in short slices so a cancel lands in seconds instead of at the end of a full
poll interval. After the gate is lost it keeps polling until every leg reports.

There is no way to unit-test the scheduling itself short of running the workflow; these
tests pin the invariants it depends on.
"""

from __future__ import annotations

import os
import re
import subprocess
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


@pytest.fixture(scope="module")
def bootstrap_script() -> str:
    assert _GITHUB is not None
    return (_GITHUB / "pre-release" / "bootstrap-pre-release.sh").read_text(encoding="utf-8")


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


def test_poll_keeps_polling_after_the_gate_is_lost(poll_script: str) -> None:
    """Each leg must reach a terminal verdict even after the gate is already FAIL."""
    assert "gate_fail_announced" in poll_script
    assert "Continuing to poll until every leg reaches a terminal verdict" in poll_script
    assert 'VERDICT["$leg"]="SKIP|still running (gate failed; workload left alive)"' not in poll_script
    assert "POLL_FAIL_FAST" not in poll_script
    assert poll_script.count("fail_seen=1") == 2
    assert poll_script.count('VERDICT["$leg"]="FAIL|') >= 2


def test_poll_reads_root_only_state_json_via_sudo(poll_script: str) -> None:
    """Pods write state.json mode 600; the runner user reads it with sudo -n."""
    assert "state_json_query" in poll_script
    assert "sudo -n jq" in poll_script
    assert "state.json not readable" in poll_script
    assert "__UNREADABLE__" in poll_script


def test_poll_report_icons_distinguish_skip_from_fail(poll_script: str) -> None:
    assert "verdict_icon" in poll_script
    assert "SKIP|PENDING" in poll_script


def test_poll_sleeps_in_slices_so_a_cancel_lands_quickly(poll_script: str) -> None:
    assert 'POLL_SLEEP_SLICE_S="${POLL_SLEEP_SLICE_S:-5}"' in poll_script
    assert 'sleep "$POLL_SLEEP_SLICE_S"' in poll_script
    assert 'sleep "$POLL_INTERVAL_S"' not in poll_script


def test_abnormal_end_cleanup_respects_leave_running(workflow: dict, poll_script: str) -> None:
    """Superseded runs may leave workloads up; cleanup must not stop those wids."""
    assert "LEAVE_RUNNING_FILE=" in poll_script
    assert "leave_running_wid" in poll_script
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


def test_docker_host_is_dispatched_before_baremetal(dispatch_script: str) -> None:
    """The 8-GPU docker host schedules slowly; queue it before the 1-GPU baremetal pods."""
    docker_pos = dispatch_script.index("queue it before the four 1-GPU baremetal pods")
    bare_pos = dispatch_script.index("# ---- baremetal legs: one non-privileged 1-GPU workload each")
    assert docker_pos < bare_pos


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
    assert "state.json stop_reason" in poll_script
    assert "reports/final.json missing" not in poll_script
    assert 'echo "PENDING|state.json stop_reason not set yet' in poll_script
    # A tolerance knob nothing reads reads as a criterion the gate enforces; it did not.
    assert "MAX_BOOT_FAILS=" not in poll_script


def test_every_runner_label_is_declared_for_actionlint(workflow: dict) -> None:
    """An undeclared self-hosted label fails pre-commit on work that never touched it."""
    assert _GITHUB is not None
    declared = yaml.safe_load((_GITHUB / "actionlint.yaml").read_text(encoding="utf-8"))
    labels = set(declared["self-hosted-runner"]["labels"])
    for job in workflow["jobs"].values():
        runs_on = job["runs-on"]
        if isinstance(runs_on, str) and "${{" not in runs_on:
            assert runs_on in labels, f"runs-on {runs_on!r} is not in .github/actionlint.yaml"


def test_the_first_setup_turn_tolerates_a_transient_failure(bootstrap_script: str) -> None:
    """A bare first call let one 429 kill a multi-hour leg; turn 2 was already guarded."""
    assert 'if ! agent_turn "$agent_log" --session-id "$uuid" < "$setup_prompt"; then' in bootstrap_script


def test_the_env_file_does_not_claim_to_stay_off_nfs(dispatch_script: str, bootstrap_script: str) -> None:
    """The key lands on the share; comments saying otherwise stop anyone from hardening it."""
    for script in (dispatch_script, bootstrap_script):
        assert "never written to NFS" not in script
        assert "never to NFS" not in script
        assert "only to pod-local" not in script


# ---- trigger classification: what a PR costs in GPU hours ----
# pyproject.toml must stay in on.pull_request.paths, or a PR that only bumps the version
# would never start the workflow. So the scope decision cannot infer "not a version bump,
# therefore CI logic changed" -- a dependency edit satisfies the path filter too.

_PYPROJECT = '[project]\nname = "x"\nversion = "{version}"\ndependencies = [{deps}]\n'
_CI_PATHS = (
    ".github/workflows/pre-release-e2e-test.yml",
    ".github/scripts/pre-release-e2e-poll.sh",
    ".github/pre-release/bootstrap-pre-release.sh",
)


def _decide_step_script(workflow: dict) -> str:
    steps = workflow["jobs"]["resolve"]["steps"]
    return next(s["run"] for s in steps if s.get("id") == "decide")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def _decide(workflow: dict, tmp_path: Path, *, edits: dict[str, str], event: str = "pull_request") -> dict[str, str]:
    """Commit a base tree, apply `edits`, then run the real decide step over the diff."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "pyproject.toml").write_text(_PYPROJECT.format(version="1.0.0", deps=""), encoding="utf-8")
    for rel in _CI_PATHS:
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    base_sha = _git(repo, "rev-parse", "HEAD")

    for rel, body in edits.items():
        (repo / rel).write_text(body, encoding="utf-8")
    if edits:
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "head")

    out = tmp_path / "gh_output"
    out.write_text("", encoding="utf-8")
    subprocess.run(
        ["bash", "-c", _decide_step_script(workflow)],
        cwd=repo,
        env={
            "PATH": os.environ["PATH"],
            "HOME": str(tmp_path),
            "EVENT": event,
            "REUSE_IN": "",
            "TASKS_IN": "",
            "BASE_SHA": base_sha,
            "BASE_REF": "main",
            "GITHUB_OUTPUT": str(out),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(line.split("=", 1) for line in out.read_text(encoding="utf-8").splitlines() if "=" in line)


def test_a_version_bump_runs_every_leg(workflow: dict, tmp_path: Path) -> None:
    got = _decide(workflow, tmp_path, edits={"pyproject.toml": _PYPROJECT.format(version="1.0.1", deps="")})
    assert got["run"] == "true"
    assert got["run_scope"] == "full"
    assert got["tasks"] == ""  # empty = all 8 in dispatch
    assert got["ci_version"].startswith("1.0.1.dev")


def test_touching_this_ci_runs_the_fast_legs(workflow: dict, tmp_path: Path) -> None:
    got = _decide(workflow, tmp_path, edits={".github/scripts/pre-release-e2e-poll.sh": "changed\n"})
    assert got["run"] == "true"
    assert got["run_scope"] == "scripts-only"
    assert got["tasks"].split(",") == [
        "baremetal-vllm-3h",
        "baremetal-sglang-3h",
        "docker-vllm-3h",
        "docker-sglang-3h",
    ]


def test_a_dependency_edit_costs_no_gpu_time(workflow: dict, tmp_path: Path) -> None:
    """A pyproject change that is not a version bump must not buy a 4-leg round."""
    got = _decide(workflow, tmp_path, edits={"pyproject.toml": _PYPROJECT.format(version="1.0.0", deps='"requests"')})
    assert got["run"] == "false"
    assert got["run_scope"] == "none"


def test_a_dependency_edit_alongside_a_ci_change_still_runs(workflow: dict, tmp_path: Path) -> None:
    got = _decide(
        workflow,
        tmp_path,
        edits={
            "pyproject.toml": _PYPROJECT.format(version="1.0.0", deps='"requests"'),
            ".github/pre-release/bootstrap-pre-release.sh": "changed\n",
        },
    )
    assert got["run"] == "true"
    assert got["run_scope"] == "scripts-only"


def test_a_manual_run_is_always_full_scope(workflow: dict, tmp_path: Path) -> None:
    got = _decide(workflow, tmp_path, edits={}, event="workflow_dispatch")
    assert got["run"] == "true"
    assert got["run_scope"] == "full"
    assert got["tasks"] == ""


# ---- an unreachable API must not read as "no phase yet" ----


def _workload_phase(poll_script: str, tmp_path: Path, api: str) -> tuple[str, str]:
    """Run the real workload_phase() against `api`; return (stdout, captured curl stderr)."""
    lines = poll_script.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("workload_phase() {"))
    end = next(i for i in range(start, len(lines)) if lines[i] == "}")
    fn = "\n".join(lines[start : end + 1])
    err_file = tmp_path / "api_err"
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f'set -euo pipefail\nAPI="$1"; API_ERR_FILE="$2"; tls=(); auth=()\n{fn}\nworkload_phase wid-1',
            "_",
            api,
            str(err_file),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout, err_file.read_text(encoding="utf-8") if err_file.is_file() else ""


def test_an_unreachable_api_is_not_mistaken_for_a_missing_phase(poll_script: str, tmp_path: Path) -> None:
    """`Unknown` is not terminal, so swallowing the error waited out the global timeout."""
    # Port 1 on loopback refuses instantly -- no network egress, no timeout to wait on.
    out, err = _workload_phase(poll_script, tmp_path, "https://127.0.0.1:1")
    assert out.startswith("__APIERR__"), out
    assert err.strip(), "curl's diagnosis must be kept, not sent to /dev/null"


def test_the_poll_gives_up_on_a_total_api_outage(poll_script: str) -> None:
    """Waiting out 14h holds 8 GPUs to learn nothing the first failed poll did not say."""
    assert 'API_FAIL_ABORT="${API_FAIL_ABORT:-10}"' in poll_script
    assert 'case "$wphase" in __APIERR__*) api_err=$(( api_err + 1 )) ;; esac' in poll_script
    # Only a TOTAL outage counts: one leg's workload going missing must not abort the run.
    assert '[ "$api_err" -gt 0 ] && [ "$api_err" -eq "$api_queried" ]' in poll_script
    assert 'VERDICT["$leg"]="FAIL|SaFE API unreachable' in poll_script
    assert "api_fail_streak=0" in poll_script


# ---- reusing a CI_VERSION must not let the previous run's artifacts pass the gate ----
# A reused CI_VERSION (workflow_dispatch reuse_ci_version, or a job re-run) puts this run
# on the paths a finished run already wrote. Verdicts are recorded once and never revisited
# and the loop breaks as soon as nothing is pending, so a single stale read on the first
# tick is enough to declare the whole gate PASS before a pod has booted.


def _leg_session_dir(poll_script: str, runs_dir: Path, leg: str, run_tag: str) -> str:
    """Run the real leg_session_dir() out of the poll script."""
    lines = poll_script.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("leg_session_dir() {"))
    end = next(i for i in range(start, len(lines)) if lines[i] == "}")
    fn = "\n".join(lines[start : end + 1])
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f'runs_dir="$1"; RUN_TAG="$2"\n{fn}\nleg_session_dir "$3"',
            "_",
            str(runs_dir),
            run_tag,
            leg,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def test_a_stale_session_pin_cannot_pass_the_gate(poll_script: str, tmp_path: Path) -> None:
    """A pin written by an earlier run on these paths must not resolve to a session dir."""
    leg = "baremetal-vllm-3h"
    session = tmp_path / leg / "session"
    finished = session / "Qwen3-8B" / "20260830T000000Z-deadbeef"
    finished.mkdir(parents=True)
    pin = session / ".session_dir"

    pin.write_text(f"{finished}\nold-run\n", encoding="utf-8")
    assert _leg_session_dir(poll_script, tmp_path, leg, "this-run") == ""

    # An untagged pin predates the stamping and carries no proof of ownership either.
    pin.write_text(f"{finished}\n", encoding="utf-8")
    assert _leg_session_dir(poll_script, tmp_path, leg, "this-run") == ""

    pin.write_text(f"{finished}\nthis-run\n", encoding="utf-8")
    assert _leg_session_dir(poll_script, tmp_path, leg, "this-run") == str(finished)


def test_the_run_tag_reaches_the_pod_and_the_poll(dispatch_script: str, bootstrap_script: str) -> None:
    """Dispatch is the single source of the tag: pods stamp it, the poll compares it."""
    assert "RUN_TAG: $rtag" in dispatch_script
    assert "RUN_TAG:$rtag" in dispatch_script
    assert '"${DISPATCH_MAP}.version_tag"' in dispatch_script
    assert "printf '%s\\n' \"${RUN_TAG:-}\"" in bootstrap_script
    assert bootstrap_script.count("pin_session_dir ") == 2
    assert 'echo "$real_sdir" > "${session}/.session_dir"' not in bootstrap_script


def test_bootstrap_rotates_the_agent_log_instead_of_appending(bootstrap_script: str) -> None:
    """The setup marker grep reads this file; a previous run's marker must not satisfy it."""
    assert ".prev-$(date -u +%Y%m%dT%H%M%SZ).log" in bootstrap_script
    assert 'grep -qiE "setup complete: ${run_mode}/${backend}" "$agent_log"' in bootstrap_script


def test_bootstrap_ignores_a_state_json_older_than_the_leg(bootstrap_script: str) -> None:
    """The wait loop picks the newest state.json under $session -- scope it to this run."""
    assert 'local leg_t0; leg_t0="$(date +%s)"' in bootstrap_script
    assert '-name state.json -newermt "@$leg_t0"' in bootstrap_script


# ---- layered timeouts: bootstrap total < SaFE pod timeout < poll global timeout ----


def _shell_default(script: str, name: str) -> int:
    match = re.search(rf"\$\{{{name}:-(\d+)\}}", script)
    assert match, f"{name} default not found"
    return int(match.group(1))


def test_pod_timeout_covers_the_whole_bootstrap_budget(
    dispatch_script: str, bootstrap_script: str, poll_script: str, workflow: dict
) -> None:
    """Setup is a separate budget; leaving it out of the pod cap gets legs killed mid-wait."""
    setup_s = _shell_default(bootstrap_script, "LEG_SETUP_DEADLINE_S")
    # The workflow env wins over the script default, so the effective value is the one
    # the ladder has to hold for.
    global_s = int(workflow["jobs"]["run"]["env"]["GLOBAL_TIMEOUT_S"])
    assert global_s == _shell_default(poll_script, "GLOBAL_TIMEOUT_S")
    assert global_s < int(workflow["jobs"]["run"]["timeout-minutes"]) * 60
    assert "local deadline_s=$(( hours * 3600 + 3600 ))" in bootstrap_script
    for hours, name in ((3, "DEADLINE_3H_S"), (12, "DEADLINE_12H_S")):
        pod_s = _shell_default(dispatch_script, name)
        bootstrap_s = setup_s + hours * 3600 + 3600
        assert bootstrap_s < pod_s, f"{name}={pod_s} is below the {bootstrap_s}s bootstrap budget"
        assert pod_s < global_s, f"{name}={pod_s} outlives the poll's {global_s}s global timeout"
