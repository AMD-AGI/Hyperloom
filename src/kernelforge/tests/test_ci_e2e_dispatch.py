"""Contract tests for .github/scripts/ci-e2e-dispatch.sh.

The script pushes PR source onto a shared mount over ssh and submits a workload
over curl. Both are stubbed here, so the invariants that matter can be asserted
offline, without a cluster:

  * a staged tree is unique per run, because two runs sharing a path let the
    faster pod delete the tree the slower one is still copying;
  * the stale-tree reap only ever targets directories this job created, because
    the staging root is shared with the template's own git-mode clones;
  * a configured-but-broken staging path fails loudly instead of falling back to
    a fetch the GPU node is not allowed to make;
  * a staged run does not ship GITHUB_TOKEN, since template.env is readable
    through the orchestration API.

These are exactly the properties that are invisible in a green run: a token that
leaks, or a reap that quietly widens its blast radius, still produces a passing
e2e.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from kernelforge.conftest import REPO_ROOT, requires_repo_root

_SCRIPT = (REPO_ROOT or Path()) / ".github" / "scripts" / "ci-e2e-dispatch.sh"

_REQUIRED_TOOLS = ("bash", "jq", "git", "tar", "base64", "install", "timeout", "mktemp")

# The dispatch script is repository infrastructure, not a packaged resource:
# under a wheel install there is no .github/ to point at, so skip rather than
# resolve to a path that does not exist and read as "nothing to check".
pytestmark = [
    requires_repo_root,
    pytest.mark.skipif(
        any(shutil.which(tool) is None for tool in _REQUIRED_TOOLS),
        reason="needs a POSIX toolchain (bash, jq, git, coreutils)",
    ),
]

_FAKE_SSH = """#!/usr/bin/env bash
# Record the invocation, drain any tarball on stdin, then honour SSH_EXIT.
{
  printf '=== ssh invocation ===\\n'
  for a in "$@"; do printf '%s\\n' "$a"; done
} >> "$SSH_LOG"
cat > /dev/null 2>/dev/null || true
exit "${SSH_EXIT:-0}"
"""

# curl is used for three different things; the stub branches on the URL so the
# script's own parsing (`-w '\\n%{http_code}'`, jq over the poll body) is exercised.
_FAKE_CURL = r"""#!/usr/bin/env python3
import os
import sys

args = sys.argv[1:]
joined = " ".join(args)

body = ""
for i, a in enumerate(args):
    if a == "-d" and i + 1 < len(args):
        body = args[i + 1]

url = ""
for a in args:
    if a.startswith("http://") or a.startswith("https://"):
        url = a

if "/statuses/" in url or "/issues/" in url:
    # post_status / report_upsert use -o /dev/null -w '%{http_code}'.
    sys.stdout.write("200")
    sys.exit(0)

if "/orchestration/workloads" in url:
    uid = "test-uid-0001"
    if url.rstrip("/").endswith("workloads") and "-X" in args and "POST" in args:
        with open(os.environ["SUBMIT_BODY"], "w") as fh:
            fh.write(body)
        # The script splits the last line off as the HTTP status.
        sys.stdout.write('{"uid":"%s"}\n201' % uid)
        sys.exit(0)
    if "-X" in args and "DELETE" in args:
        sys.exit(0)
    # Poll: report a terminal success straight away so the test does not sleep.
    sys.stdout.write(
        '{"orchestration":{"phase":"Succeeded","conditions":[]},"dispatches":[]}'
    )
    sys.exit(0)

sys.stdout.write("200")
"""


def _write_exe(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def harness(tmp_path):
    """A stubbed environment for one dispatch run."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh_log = tmp_path / "ssh.log"
    submit_body = tmp_path / "submit.json"
    _write_exe(bin_dir / "ssh", _FAKE_SSH)
    _write_exe(bin_dir / "curl", _FAKE_CURL)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (workspace / "kernel.py").write_text("x = 1\n", encoding="utf-8")
    # `git archive HEAD` is the real thing under test for what gets shipped, so
    # use a real repository rather than stubbing git.
    env_git = {**os.environ, "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig")}
    for cmd in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "config", "user.email", "t@example.com"],
        ["git", "config", "user.name", "t"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "-m", "seed"],
    ):
        subprocess.run(cmd, cwd=workspace, check=True, env=env_git)

    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    key = tmp_path / "stage-key"
    key.write_text("not-a-real-key\n", encoding="utf-8")

    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "SSH_LOG": str(ssh_log),
        "SUBMIT_BODY": str(submit_body),
        "E2E_API_BASE": "https://example.invalid/api",
        "E2E_API_KEY": "test-key",
        "E2E_INFRA_TYPE": "kubernetes",
        "HEAD_REF": "feature/test",
        "HEAD_SHA": "abcdef0123456789abcdef0123456789abcdef01",
        "GITHUB_WORKSPACE": str(workspace),
        "RUNNER_TEMP": str(runner_temp),
        "GITHUB_TOKEN": "ghs_sentinel_token_value",
        "PR_NUMBER": "42",
        "GITHUB_RUN_ID": "998877",
        "GITHUB_RUN_ATTEMPT": "3",
        "POLL_INTERVAL_S": "1",
        "CI_E2E_STAGE_SSH_KEY": str(key),
    }
    # Keep the commit-status and PR-comment paths switched off.
    for noisy in ("GH_STATUS_TOKEN", "GH_STATUS_REPO", "GH_STATUS_SHA", "GITHUB_STEP_SUMMARY"):
        env.pop(noisy, None)

    class Harness:
        def __init__(self):
            self.env = env
            self.ssh_log = ssh_log
            self.submit_body = submit_body
            self.key = key

        def run(self, **overrides):
            run_env = {**self.env, **{k: str(v) for k, v in overrides.items()}}
            return subprocess.run(
                ["bash", str(_SCRIPT)],
                env=run_env,
                capture_output=True,
                text=True,
                timeout=120,
            )

        def ssh_calls(self) -> str:
            return self.ssh_log.read_text(encoding="utf-8") if self.ssh_log.exists() else ""

        def ssh_invocations(self) -> list[str]:
            """One entry per ssh process, so a caller can tell them apart.

            The staging command already contains an `rm -rf` of the target, so
            asserting on the log as a whole cannot distinguish "cleaned up
            afterwards" from "never cleaned up at all".
            """
            raw = self.ssh_calls()
            return [part.strip() for part in raw.split("=== ssh invocation ===") if part.strip()]

        def payload(self) -> dict:
            return json.loads(self.submit_body.read_text(encoding="utf-8"))

    return Harness()


def _staged(harness, **overrides):
    return harness.run(
        CI_E2E_STAGE_HOST="ci@mount.invalid",
        CI_E2E_STAGE_ROOT="/mnt/shared/kernelforge-ci/src",
        **overrides,
    )


def test_staged_tree_is_unique_per_run_and_attempt(harness):
    """Two runs must never share a staging path (see the cp -a race)."""
    result = _staged(harness)

    assert result.returncode == 0, result.stderr
    src_dir = harness.payload()["template"]["params"]["KF_SOURCE_DIR"]
    assert src_dir.startswith("/mnt/shared/kernelforge-ci/src/")
    # sha keeps it readable, run id + attempt are what make it collision-free.
    assert "998877" in src_dir and src_dir.split("/")[-2].endswith("_998877_3")
    assert "abcdef012345" in src_dir


def test_staged_run_tells_the_workload_not_to_fetch(harness):
    result = _staged(harness)

    assert result.returncode == 0, result.stderr
    assert harness.payload()["template"]["params"]["KF_USE_GIT"] == "0"


def test_staged_run_does_not_ship_the_github_token(harness):
    """template.env is readable via the orchestration API; a staged pod never fetches."""
    result = _staged(harness)

    assert result.returncode == 0, result.stderr
    payload = harness.payload()
    assert payload["template"].get("env", {}) == {}
    assert "ghs_sentinel_token_value" not in json.dumps(payload)


def test_unstaged_run_still_ships_the_token_for_its_fetch(harness):
    """Without staging configured the pod does fetch, so it still needs the token."""
    result = harness.run()

    assert result.returncode == 0, result.stderr
    payload = harness.payload()
    assert payload["template"]["params"]["KF_USE_GIT"] == "1"
    assert payload["template"]["env"]["GITHUB_TOKEN"] == "ghs_sentinel_token_value"
    assert harness.ssh_calls() == ""


def test_reap_only_targets_directories_this_job_created(harness):
    """The staging root also holds the template's git-mode clones."""
    result = _staged(harness)

    assert result.returncode == 0, result.stderr
    calls = harness.ssh_calls()
    assert "-name" in calls and "pr_*" in calls, calls
    # An unfiltered sweep would delete other workloads' trees.
    assert "-mtime" in calls


def test_configured_staging_without_a_key_fails_loudly(harness):
    """Half-configured staging must not degrade into an impossible fetch."""
    harness.key.unlink()

    result = _staged(harness)

    assert result.returncode != 0
    assert not harness.submit_body.exists(), "no workload should be submitted"
    assert "no key at" in (result.stdout + result.stderr)


def test_staging_failure_does_not_fall_back_to_a_fetch(harness):
    """Falling back would reproduce the 403 this whole path exists to avoid."""
    result = _staged(harness, SSH_EXIT="255")

    assert result.returncode != 0
    assert not harness.submit_body.exists(), "no workload should be submitted"
    assert "not falling back" in (result.stdout + result.stderr)


def test_staged_tree_is_removed_once_the_run_is_terminal(harness):
    result = _staged(harness)

    assert result.returncode == 0, result.stderr
    src_dir = harness.payload()["template"]["params"]["KF_SOURCE_DIR"]
    stage_dir = src_dir.rsplit("/", 1)[0]

    calls = harness.ssh_invocations()
    # The staging call also removes the target first, so the cleanup has to be
    # identified as its own invocation rather than by searching the whole log.
    assert len(calls) >= 2, f"cleanup never ran as a separate ssh call: {calls}"
    last = calls[-1]
    assert f"rm -rf -- '{stage_dir}'" in last
    assert "unpack.py" not in last, "matched the staging call, not the cleanup"


def test_private_key_copy_does_not_outlive_the_run(harness):
    result = _staged(harness)

    assert result.returncode == 0, result.stderr
    leftovers = list(Path(harness.env["RUNNER_TEMP"]).glob("*stage_key*"))
    assert leftovers == [], f"key copy left behind: {leftovers}"
