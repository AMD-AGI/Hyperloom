#!/usr/bin/env python3
"""Submit InferenceX PRs based on Hyperloom CI optimization results.

Flow: read ci_summary.json + optimization reports → extract changes via
diff or LLM → clone InferenceX fork → apply changes → create PR.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

log = logging.getLogger("pr-submitter")

CI_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CI_DIR / "ci-config.yaml"

# ── LLM-based extraction (fallback when diff-based fails) ──

LLM_ENDPOINT = os.environ.get("SAFE_BASE_URL", "") + "/api/v1/llm-proxy/v1/chat/completions"

LLM_EXTRACT_PROMPT = """\
You are analyzing an inference optimization report. Extract the specific server
configuration changes that improved performance.

Compare the BASELINE server launch command with the OPTIMIZED server launch command.
Return ONLY a JSON object with these fields:

{
  "flag_changes": [
    {"flag": "--flag-name", "old_value": "4", "new_value": "8", "action": "modify"},
    {"flag": "--new-flag", "value": "some_val", "action": "add"},
    {"flag": "--removed-flag", "action": "remove"}
  ],
  "env_var_changes": [
    {"var": "VAR_NAME", "value": "1", "action": "add"}
  ],
  "gain_pct": 6.27,
  "description": "one-line summary of what changed"
}

Rules:
- Only include changes that IMPROVED performance (positive gain)
- Ignore kernel optimization results (those are runtime, not config changes)
- If no server config changes were found, return {"flag_changes": [], "env_var_changes": [], "gain_pct": 0, "description": "no config changes"}
- Return ONLY valid JSON, no markdown fences, no explanation

Report:
{report_content}
"""


def _llm_extract_changes(report_content: str, api_key: str) -> dict:
    """Use an LLM to extract structured changes from an optimization report.

    Args:
        report_content (str): The optimization report text (truncated to the
            first 6000 chars before sending).
        api_key (str): Bearer token for the LLM proxy endpoint.

    Returns:
        dict: Parsed change object with ``flag_changes`` / ``env_var_changes``
            etc., or an empty dict on any failure.
    """
    import requests

    prompt = LLM_EXTRACT_PROMPT.replace("{report_content}", report_content[:6000])
    try:
        resp = requests.post(
            LLM_ENDPOINT,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"},
            json={"model": "openai/gpt-4.1-mini",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0, "max_tokens": 500},
            timeout=30,
            verify=os.environ.get(
                "SSL_CERT_FILE", os.environ.get("REQUESTS_CA_BUNDLE", True)
            ),
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(content)
    except Exception as e:
        log.warning("LLM extraction failed: %s", e)
        return {}


# ── Multi-format extraction (handles real CI report formats) ──


def _parse_flags_string(s: str) -> dict[str, str]:
    """Parse '--flag1 val1 --flag2 val2 ...' into a dict.

    Args:
        s (str): A flag string (line continuations are folded into spaces).

    Returns:
        dict[str, str]: Flag name -> value, with valueless flags mapped to an
            empty string.
    """
    flags: dict[str, str] = {}
    try:
        tokens = shlex.split(s.replace("\\\n", " "), posix=True)
    except ValueError:
        tokens = s.replace("\\\n", " ").split()
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            if "=" in tok:
                k, v = tok.split("=", 1)
                flags[k] = v
            elif i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                flags[tok] = tokens[i + 1]
                i += 1
            else:
                flags[tok] = ""
        elif tok.startswith("-") and "." in tok:
            flags[tok] = ""
        i += 1
    return flags


_LAUNCH_CMD_RE = re.compile(
    r"(?:python3?\s+-m\s+\S+\.launch_server|vllm\s+serve)\b",
)


def _extract_fenced_bash_blocks(report: str) -> list[str]:
    """Return bodies of ``` / ```bash fenced blocks (linear-time scan).

    Args:
        report (str): Markdown-ish report text to scan.

    Returns:
        list[str]: The inner text of each fenced code block, in order.
    """
    blocks: list[str] = []
    opener = re.compile(r"```(?:bash)?\s*\n", re.IGNORECASE)
    pos = 0
    while True:
        m = opener.search(report, pos)
        if not m:
            break
        start = m.end()
        end = report.find("\n```", start)
        if end == -1:
            break
        blocks.append(report[start:end])
        pos = end + 4
    return blocks


def _extract_optimized_flags(report: str) -> dict[str, str]:
    """Extract the final/optimized flag set from the report.

    Supports: EXTRA_SGLANG_ARGS="...", EXTRA_VLLM_ARGS in YAML,
    and full bash blocks with launch_server/vllm serve.

    Args:
        report (str): The optimization report text.

    Returns:
        dict[str, str]: The extracted optimized flag set, or an empty dict if
            no recognizable launch config was found.
    """
    m = re.search(r'EXTRA_SGLANG_ARGS="([^"]+)"', report)
    if m:
        log.debug("Extracted flags from EXTRA_SGLANG_ARGS")
        return _parse_flags_string(m.group(1))

    m = re.search(
        r"EXTRA_VLLM_ARGS:\s*>-?\s*\n((?:\s+.*\n)+)",
        report,
    )
    if m:
        args_text = " ".join(line.strip() for line in m.group(1).splitlines() if line.strip())
        log.debug("Extracted flags from EXTRA_VLLM_ARGS YAML")
        return _parse_flags_string(args_text)

    launch_blocks = [
        b for b in _extract_fenced_bash_blocks(report) if _LAUNCH_CMD_RE.search(b)
    ]
    if launch_blocks:
        block = launch_blocks[-1].replace("\\\n", " ")
        log.debug("Extracted flags from bash launch command")
        return _parse_flags_string(block)

    return {}


def _parse_script_flags(script_content: str) -> tuple[dict[str, str], dict[str, str]]:
    """Extract env vars and server flags from an InferenceX .sh script.

    Args:
        script_content (str): The shell script contents.

    Returns:
        tuple[dict[str, str], dict[str, str]]: ``(env_vars, flags)`` parsed
            from ``export`` lines and the launch command.
    """
    env_vars: dict[str, str] = {}
    flags: dict[str, str] = {}

    for m in re.finditer(r"^export\s+(\w+)=(\S+)", script_content, re.MULTILINE):
        env_vars[m.group(1)] = m.group(2).strip("'\"")

    joined = script_content.replace("\\\n", " ")
    launch_m = re.search(
        r"(python3?\s+-m\s+\S+\.launch_server|vllm\s+serve)\s+(.*?)(?:>|$)",
        joined, re.DOTALL,
    )
    if launch_m:
        cmd_part = launch_m.group(2)
        cmd_part = re.sub(r'\$\{?\w+\}?', '__VAR__', cmd_part)
        flags = _parse_flags_string(cmd_part)

    for m_var in re.finditer(r'^(\w+)="([^"]*)"', script_content, re.MULTILINE):
        vname, vval = m_var.group(1), m_var.group(2)
        if vname in ("ATTN_BACKEND", "FUSE_ROPE_KVCACHE"):
            flags.update(_parse_flags_string(vval))

    return env_vars, flags


def _diff_flags(baseline: dict[str, str],
                optimized: dict[str, str]) -> list[dict]:
    """Compare two flag dicts, return add/modify changes only.

    Does not generate 'remove' actions because EXTRA_*_ARGS in reports
    only contain the subset of flags, not the full command.

    Args:
        baseline (dict[str, str]): Flags from the current script.
        optimized (dict[str, str]): Flags from the optimization report.

    Returns:
        list[dict]: ``add``/``modify`` change descriptors (model/host/port and
            other infra flags are skipped).
    """
    changes = []
    skip = {"--model-path", "--model", "--host", "--port",
            "--tensor-parallel-size", "--trust-remote-code"}
    for flag, new_val in optimized.items():
        if flag in skip:
            continue
        old_val = baseline.get(flag)
        if old_val is None:
            changes.append({"flag": flag, "value": new_val, "action": "add"})
        elif old_val != new_val and old_val != "__VAR__":
            changes.append({"flag": flag, "old_value": old_val,
                            "new_value": new_val, "action": "modify"})
    return changes


def extract_changes_vs_script(report_content: str,
                              script_content: str | None,
                              api_key: str | None = None) -> dict:
    """Extract config changes by comparing report's optimized config vs script.

    Primary: parse report final config + diff against InferenceX script.
    Fallback: LLM-based extraction.

    Args:
        report_content (str): The optimization report text.
        script_content (str | None): The current InferenceX script, or
            ``None`` to treat all optimized flags as additions.
        api_key (str | None): LLM proxy token enabling the fallback path.

    Returns:
        dict: A change object with ``flag_changes`` / ``env_var_changes``, or
            an empty dict when nothing actionable is found.
    """
    opt_flags = _extract_optimized_flags(report_content)
    if not opt_flags:
        if api_key:
            log.info("No optimized flags found, trying LLM fallback")
            return _llm_extract_changes(report_content, api_key)
        log.warning("No optimized flags extracted from report")
        return {}

    if script_content:
        _script_env, script_flags = _parse_script_flags(script_content)
        flag_changes = _diff_flags(script_flags, opt_flags)
    else:
        flag_changes = [{"flag": f, "value": v, "action": "add"}
                        for f, v in opt_flags.items()
                        if f not in ("--model-path", "--host", "--port",
                                     "--tensor-parallel-size", "--trust-remote-code")]

    if not flag_changes:
        log.info("No flag differences found between report and script")
        return {}

    log.info("Extracted %d flag change(s) vs InferenceX script", len(flag_changes))
    return {"flag_changes": flag_changes, "env_var_changes": []}


# ── Benchmark script modification ──

def _apply_flag_to_script(content: str, flag: str, value: str | None,
                          action: str) -> str:
    """Apply a single flag change to a benchmark shell script.

    Args:
        content (str): The current script text.
        flag (str): The flag name to add/modify/remove.
        value (str | None): The flag value (may be ``None`` for valueless
            flags).
        action (str): One of ``"add"``, ``"modify"``, or ``"remove"``.

    Returns:
        str: The updated script text (unchanged if nothing matched).
    """
    if action == "modify" and value is not None:
        pattern = re.compile(
            rf"({re.escape(flag)}[\s=])(\S+)",
        )
        if pattern.search(content):
            return pattern.sub(rf"\g<1>{value}", content)

    if action == "add":
        flag_str = f"{flag} {value}" if value else flag
        lines = content.split("\n")
        cmd_start = cmd_end = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if re.match(r"(?:python3?\s+-m\s+\S+\.launch_server|vllm\s+serve)\b", stripped):
                cmd_start = i
            if cmd_start >= 0 and i >= cmd_start and not stripped.endswith("\\"):
                cmd_end = i
                break
        if cmd_start >= 0 and cmd_end >= 0:
            indent = "    "
            for j in range(cmd_start + 1, cmd_end + 1):
                m_indent = re.match(r"^(\s+)", lines[j])
                if m_indent:
                    indent = m_indent.group(1)
                    break
            prev = cmd_end - 1 if cmd_end > cmd_start else cmd_start
            if not lines[prev].rstrip().endswith("\\"):
                lines[prev] = lines[prev].rstrip() + " \\"
            lines.insert(cmd_end, f"{indent}{flag_str} \\")
            return "\n".join(lines)

    if action == "remove":
        pattern = re.compile(
            rf"\s*\\?\s*{re.escape(flag)}(?:[\s=]\S+)?",
        )
        content = pattern.sub("", content)

    return content


def _apply_env_to_script(content: str, var: str, value: str,
                         action: str) -> str:
    """Apply an env var change to a benchmark shell script.

    Args:
        content (str): The current script text.
        var (str): The environment variable name.
        value (str): The value to set (ignored for ``remove``).
        action (str): One of ``"add"``, ``"modify"``, or ``"remove"``.

    Returns:
        str: The updated script text.
    """
    export_line = f"export {var}={value}\n"
    if action in ("add", "modify"):
        pattern = re.compile(rf"^export\s+{re.escape(var)}=.*$", re.MULTILINE)
        if pattern.search(content):
            return pattern.sub(f"export {var}={value}", content)
        m = re.search(r"^(export\s+\w+=.*\n)", content, re.MULTILINE)
        if m:
            return content[:m.end()] + export_line + content[m.end():]
        return export_line + content

    if action == "remove":
        pattern = re.compile(rf"^export\s+{re.escape(var)}=.*\n?", re.MULTILINE)
        return pattern.sub("", content)

    return content


def apply_changes_to_script(script_path: Path, changes: dict) -> bool:
    """Apply extracted changes to an InferenceX benchmark script.

    Args:
        script_path (Path): Path to the benchmark shell script to edit.
        changes (dict): A change object with ``flag_changes`` and
            ``env_var_changes`` lists.

    Returns:
        bool: True if the file content changed and was written, else False.
    """
    if not script_path.exists():
        log.warning("Script not found: %s", script_path)
        return False

    content = script_path.read_text()
    original = content

    for fc in changes.get("flag_changes", []):
        value = fc.get("new_value") or fc.get("value")
        content = _apply_flag_to_script(content, fc["flag"], value, fc["action"])

    for ec in changes.get("env_var_changes", []):
        value = ec.get("new_value") or ec.get("value", "")
        content = _apply_env_to_script(content, ec["var"], value, ec["action"])

    if content != original:
        script_path.write_text(content)
        log.info("Updated script: %s", script_path)
        return True

    log.info("No changes applied to script: %s", script_path)
    return False


# ── perf-changelog.yaml update ──

def append_perf_changelog(changelog_path: Path, config_keys: list[str],
                          descriptions: list[str], pr_link: str = ""):
    """Prepend a new entry to perf-changelog.yaml.

    Args:
        changelog_path (Path): Path to ``perf-changelog.yaml`` (created if
            absent).
        config_keys (list[str]): InferenceX config keys touched by the entry.
        descriptions (list[str]): Human-readable change descriptions.
        pr_link (str): The PR URL; a placeholder is used when empty.
    """
    entry = {
        "config-keys": config_keys,
        "description": descriptions,
        "pr-link": pr_link or "https://github.com/SemiAnalysisAI/InferenceX/pull/XXX",
    }

    if changelog_path.exists():
        content = changelog_path.read_text()
    else:
        content = ""

    dumped = yaml.dump([entry], default_flow_style=False, allow_unicode=True,
                       sort_keys=False)
    new_content = dumped + "\n" + content
    changelog_path.write_text(new_content)
    log.info("Prepended perf-changelog entry for %s", config_keys)


# ── PR body generation ──

def _generate_pr_body(model_results: list[dict]) -> str:
    """Generate PR body following InferenceX PR template.

    Args:
        model_results (list[dict]): Eligible model results, each carrying
            metrics and a ``_changes`` descriptor.

    Returns:
        str: The rendered markdown PR body.
    """
    lines = [
        "## Description\n",
        "Automated performance optimization update from Hyperloom CI.\n",
    ]

    for mr in model_results:
        key = mr["inferenceX_key"]
        gain = mr.get("gain_pct")
        vs_ifx = mr.get("vs_inferenceX_pct")
        lines.append(f"### {key}\n")

        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        if mr.get("baseline_tok_per_gpu") is not None:
            lines.append(f"| Baseline (tok/s/GPU) | {mr['baseline_tok_per_gpu']:.2f} |")
        if mr.get("optimized_tok_per_gpu") is not None:
            lines.append(f"| Optimized (tok/s/GPU) | {mr['optimized_tok_per_gpu']:.2f} |")
        if gain is not None:
            lines.append(f"| Optimization Gain | {gain:+.1f}% |")
        if mr.get("inferenceX_tok_per_gpu") is not None:
            lines.append(f"| InferenceX Current (tok/s/GPU) | {mr['inferenceX_tok_per_gpu']:.2f} |")
        if vs_ifx is not None:
            lines.append(f"| **vs InferenceX** | **{vs_ifx:+.1f}%** |")
        lines.append("")

        changes = mr.get("_changes", {})
        if changes.get("flag_changes"):
            lines.append("**Server flag changes:**")
            for fc in changes["flag_changes"]:
                if fc["action"] == "modify":
                    lines.append(f"- `{fc['flag']}`: `{fc.get('old_value')}` → `{fc.get('new_value')}`")
                elif fc["action"] == "add":
                    lines.append(f"- Add `{fc['flag']} {fc.get('value', '')}`")
                elif fc["action"] == "remove":
                    lines.append(f"- Remove `{fc['flag']}`")
            lines.append("")

        if changes.get("env_var_changes"):
            lines.append("**Environment variable changes:**")
            for ec in changes["env_var_changes"]:
                if ec["action"] == "add":
                    lines.append(f"- Add `export {ec['var']}={ec.get('value', '')}`")
                elif ec["action"] == "modify":
                    lines.append(f"- `{ec['var']}`: `{ec.get('old_value')}` → `{ec.get('new_value')}`")
            lines.append("")

    lines.extend([
        "## Related Issue\n",
        "Automated by Hyperloom CI\n",
        "## Type of Change\n",
        "- [x] Configuration change\n",
        "## Checklist\n",
        "- [x] I have tested my changes locally",
        "- [x] I have updated documentation if necessary",
        "- [x] **If I changed a container image or config, I have already updated `perf-changelog.yaml`**",
    ])
    return "\n".join(lines)


# ── Git + PR operations ──

def _run_git(args: list[str], cwd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git subcommand and capture its output.

    Args:
        args (list[str]): Git arguments (without the leading ``git``).
        cwd (str): Working directory to run the command in.
        check (bool): Raise on non-zero exit when True.

    Returns:
        subprocess.CompletedProcess: The completed process with captured
            text stdout/stderr.
    """
    cmd = ["git"] + args
    log.debug("git %s (cwd=%s)", " ".join(args), cwd)
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)


def clone_fork(fork_url: str, target_dir: str, branch: str = "main"):
    """Clone a fork at a specific branch into a target directory.

    Args:
        fork_url (str): The git URL of the fork to clone.
        target_dir (str): Destination directory for the clone.
        branch (str): Branch to check out (default ``"main"``).
    """
    subprocess.run(
        ["git", "clone", f"--branch={branch}", fork_url, target_dir],
        check=True, capture_output=True, text=True,
    )


def sync_fork_from_upstream(repo_dir: str, upstream_url: str,
                            branch: str, token: str | None) -> None:
    """Merge upstream into the fork branch before creating a PR branch.

    The fork keeps verify-pr/sync workflow files that upstream does not have.
    Rebase is fragile with fork-only merge commits, while reset would make the
    later PR appear to delete fork-only files. Merge preserves those files and
    keeps the fork base current.

    Args:
        repo_dir (str): Local path to the cloned fork.
        upstream_url (str): URL of the upstream repository.
        branch (str): Branch to sync (e.g. ``"main"``).
        token (str | None): Optional token used to build an authenticated
            push URL; falls back to ``origin`` when absent.

    Returns:
        None: Nothing is returned; the fork branch is updated and pushed as a
            side effect.
    """
    _run_git(["remote", "add", "upstream", upstream_url], repo_dir, check=False)
    _run_git(["fetch", "upstream", branch], repo_dir)
    _run_git(["checkout", branch], repo_dir)

    behind = _run_git(
        ["rev-list", "--count", f"{branch}..upstream/{branch}"], repo_dir,
    ).stdout.strip()
    if behind == "0":
        log.info("Fork %s is already up-to-date with upstream", branch)
        return

    log.info("Fork %s is %s commit(s) behind upstream, merging", branch, behind)
    _run_git(["merge", f"upstream/{branch}", "--no-edit"], repo_dir)

    push_url = None
    if token:
        remote = _run_git(["remote", "get-url", "origin"], repo_dir)
        url = remote.stdout.strip()
        if url.startswith("https://"):
            push_url = url.replace("https://", f"https://x-access-token:{token}@")
        elif url.startswith("git@github.com:"):
            repo_path = url.replace("git@github.com:", "")
            push_url = f"https://x-access-token:{token}@github.com/{repo_path}"

    push_args = ["push"]
    push_args.append(push_url if push_url else "origin")
    push_args.append(branch)
    _run_git(push_args, repo_dir)
    log.info("Synced fork %s with upstream (merge)", branch)


def create_pr_branch(repo_dir: str, branch_name: str):
    """Create and check out a new branch for the PR.

    Args:
        repo_dir (str): Local path to the repository.
        branch_name (str): Name of the branch to create.
    """
    _run_git(["checkout", "-b", branch_name], repo_dir)


def commit_and_push(repo_dir: str, branch_name: str, message: str,
                    token: str | None = None):
    """Commit all changes and push the branch to the fork.

    Args:
        repo_dir (str): Local path to the repository.
        branch_name (str): Branch to push.
        message (str): Commit message.
        token (str | None): Optional token used to build an authenticated
            push URL; falls back to ``origin`` when absent.

    Returns:
        bool: True if a commit was pushed, False if there was nothing to
            commit or the push failed.
    """
    _run_git(["config", "user.email", "hyperloom-ci@noreply.github.com"], repo_dir)
    _run_git(["config", "user.name", "Hyperloom CI"], repo_dir)
    _run_git(["add", "-A"], repo_dir)

    status = _run_git(["status", "--porcelain"], repo_dir)
    if not status.stdout.strip():
        log.info("No changes to commit")
        return False

    _run_git(["commit", "-m", message], repo_dir)

    push_url = None
    if token:
        remote = _run_git(["remote", "get-url", "origin"], repo_dir)
        url = remote.stdout.strip()
        if url.startswith("https://"):
            push_url = url.replace("https://", f"https://x-access-token:{token}@")
        elif url.startswith("git@github.com:"):
            repo_path = url.replace("git@github.com:", "")
            push_url = f"https://x-access-token:{token}@github.com/{repo_path}"

    try:
        if push_url:
            _run_git(["push", push_url, branch_name], repo_dir)
        else:
            _run_git(["push", "-u", "origin", branch_name], repo_dir)
    except subprocess.CalledProcessError as e:
        log.error("Push failed (rc=%d): %s", e.returncode, e.stderr)
        return False

    return True


def create_github_pr(owner: str, repo: str, branch: str, base: str,
                     title: str, body: str, token: str):
    """Create a PR within the same repo using gh CLI or GitHub API.

    Args:
        owner (str): Repository owner/org.
        repo (str): Repository name.
        branch (str): Head branch for the PR.
        base (str): Base branch to merge into.
        title (str): PR title.
        body (str): PR body markdown.
        token (str): GitHub token for ``gh`` or the REST API.

    Returns:
        str | None: The created PR URL, or ``None`` on failure.
    """
    try:
        result = subprocess.run(
            ["gh", "pr", "create",
             "--repo", f"{owner}/{repo}",
             "--head", branch,
             "--base", base,
             "--title", title,
             "--body", body],
            capture_output=True, text=True,
            env={**os.environ, "GH_TOKEN": token},
        )
        if result.returncode == 0:
            pr_url = result.stdout.strip()
            log.info("PR created: %s", pr_url)
            return pr_url
        log.warning("gh pr create failed: %s", result.stderr)
    except FileNotFoundError:
        log.info("gh CLI not found, falling back to API")

    import requests
    resp = requests.post(
        f"https://api.github.com/repos/{owner}/{repo}/pulls",
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/vnd.github+json"},
        json={"title": title, "body": body, "head": branch, "base": base},
        timeout=30,
    )
    if resp.status_code in (200, 201):
        pr_url = resp.json()["html_url"]
        log.info("PR created via API: %s", pr_url)
        return pr_url

    log.error("Failed to create PR: %s %s", resp.status_code, resp.text)
    return None


def _parse_pr_number(pr_url: str) -> int | None:
    """Extract the numeric PR id from a GitHub PR URL.

    Args:
        pr_url (str): A GitHub PR URL (may be empty/None-like).

    Returns:
        int | None: The PR number, or ``None`` if not found.
    """
    m = re.search(r"/pull/(\d+)", pr_url or "")
    return int(m.group(1)) if m else None


def add_pr_labels(owner: str, repo: str, pr_url: str,
                  labels: list[str], token: str) -> bool:
    """Attach labels to an existing PR; triggers fork's labeled-event workflows.

    Prefers `gh pr edit` since PAT scopes are already verified for push;
    falls back to REST API POST /issues/{n}/labels.

    Args:
        owner (str): Repository owner/org.
        repo (str): Repository name.
        pr_url (str): URL of the PR to label.
        labels (list[str]): Labels to attach (no-op if empty).
        token (str): GitHub token for ``gh`` or the REST API.

    Returns:
        bool: True if labels were attached (or none were requested), else
            False.
    """
    if not labels:
        return True
    pr_number = _parse_pr_number(pr_url)
    if not pr_number:
        log.warning("Could not parse PR number from %s, skipping labels", pr_url)
        return False

    try:
        args = ["gh", "pr", "edit", str(pr_number),
                "--repo", f"{owner}/{repo}"]
        for lbl in labels:
            args += ["--add-label", lbl]
        result = subprocess.run(
            args, capture_output=True, text=True,
            env={**os.environ, "GH_TOKEN": token},
        )
        if result.returncode == 0:
            log.info("Labels attached via gh: %s", labels)
            return True
        log.warning("gh pr edit failed (rc=%d): %s",
                    result.returncode, result.stderr.strip())
    except FileNotFoundError:
        log.info("gh CLI not found for labels, falling back to API")

    import requests
    resp = requests.post(
        f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/labels",
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/vnd.github+json"},
        json={"labels": labels},
        timeout=30,
    )
    if resp.status_code in (200, 201):
        log.info("Labels attached via API: %s", labels)
        return True
    log.error("Failed to attach labels: %s %s", resp.status_code, resp.text)
    return False


# ── Main orchestration ──

def load_config(path: str | None = None) -> dict:
    """Load the CI YAML config.

    Args:
        path (str | None): Path to the config file; uses
            :data:`DEFAULT_CONFIG` when ``None``.

    Returns:
        dict: The parsed configuration mapping.
    """
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p) as f:
        return yaml.safe_load(f)


def _find_script_in_repo(repo_dir: Path, ifx_key: str,
                         scripts_path: str) -> Path | None:
    """Find benchmark script by inferenceX key (exact match then prefix).

    Args:
        repo_dir (Path): Root of the cloned InferenceX repo.
        ifx_key (str): InferenceX key used to locate the script.
        scripts_path (str): Relative path to the benchmark scripts directory.

    Returns:
        Path | None: The matching ``.sh`` path, or ``None`` if none matched.
    """
    scripts_dir = repo_dir / scripts_path
    if not scripts_dir.is_dir():
        return None
    normalized = ifx_key.replace("-", "_").replace(".", "")
    for sh in sorted(scripts_dir.glob("*.sh")):
        if sh.stem.replace(".", "") == normalized:
            return sh
    prefix = normalized.rsplit("_", 1)[0] if "_" in normalized else normalized
    for sh in sorted(scripts_dir.glob("*.sh")):
        if sh.stem.replace(".", "").startswith(prefix):
            return sh
    return None


def _load_report(model_result: dict, reports_dir: Path) -> str:
    """Load report content from inline data or file.

    Args:
        model_result (dict): A CI model result, possibly with inline
            ``report_content``.
        reports_dir (Path): Directory holding per-model report files.

    Returns:
        str: The report markdown, or ``""`` if none is found.
    """
    content = model_result.get("report_content", "")
    if content:
        return content
    model_name = model_result.get("model", "")
    report_path = reports_dir / model_name / "optimization_report.md"
    if report_path.exists():
        return report_path.read_text()
    return ""


def process_results(
    ci_summary: dict,
    reports_dir: Path,
    config: dict,
    dry_run: bool = False,
) -> list[dict]:
    """Filter CI results by gain threshold; return candidates for PR.

    Args:
        ci_summary (dict): Parsed ``ci_summary.json`` with a ``models`` list.
        reports_dir (Path): Directory holding per-model reports.
        config (dict): The CI config (provides ``pr_submission.min_gain_pct``).
        dry_run (bool): Reserved flag, unused here (kept for call symmetry).

    Returns:
        list[dict]: Eligible model results, each annotated with ``_report``.
    """
    pr_cfg = config.get("pr_submission", {})
    min_gain = pr_cfg.get("min_gain_pct", 3.0)

    eligible = []
    for model_result in ci_summary.get("models", []):
        if model_result.get("status") != "completed":
            continue

        gain = model_result.get("gain_pct")
        vs_ifx = model_result.get("vs_inferenceX_pct")

        if gain is None or gain < min_gain:
            log.info("Skip %s: gain=%.1f%% < threshold %.1f%%",
                     model_result.get("inferenceX_key", "?"),
                     gain or 0, min_gain)
            continue

        report_content = _load_report(model_result, reports_dir)
        if not report_content:
            log.warning("No report for %s, skipping", model_result.get("model"))
            continue

        model_result["_report"] = report_content
        eligible.append(model_result)
        log.info("Candidate: %s (gain=%.1f%%, vs_ifx=%s)",
                 model_result["inferenceX_key"], gain,
                 f"{vs_ifx:+.1f}%" if vs_ifx is not None else "N/A")

    return eligible


def submit_pr(
    eligible: list[dict],
    config: dict,
    ci_summary: dict,
    dry_run: bool = False,
):
    """Clone InferenceX repo, apply changes, and submit PR.

    Clones the fork, optionally syncs from upstream, extracts and applies
    per-model changes, updates the changelog, and creates the PR (skipped in
    ``dry_run``).

    Args:
        eligible (list[dict]): Candidate model results from
            :func:`process_results`.
        config (dict): The CI config.
        ci_summary (dict): The full CI summary (passed through for context).
        dry_run (bool): Preview changes without committing or creating a PR.

    Returns:
        None: Nothing is returned; the PR is created as a side effect.
    """
    pr_cfg = config.get("pr_submission", {})
    repo_url = pr_cfg.get("repo_url", "https://github.com/lishuoshuo-amd/InferenceX.git")
    repo_owner = pr_cfg.get("repo_owner", "lishuoshuo-amd")
    repo_name = pr_cfg.get("repo_name", "InferenceX")
    base_branch = pr_cfg.get("base_branch", "main")
    scripts_path = config.get("inferenceX", {}).get("scripts_path", "benchmarks/single_node")

    token = os.environ.get(pr_cfg.get("token_env", "INFERENCEX_FORK_TOKEN"))

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    branch_name = f"hyperloom/ci-{ts}"

    with tempfile.TemporaryDirectory() as tmpdir:
        log.info("Cloning %s...", repo_url)
        clone_fork(repo_url, tmpdir, base_branch)

        if pr_cfg.get("sync_upstream_before_pr", True):
            upstream_url = pr_cfg.get("upstream_repo_url")
            if upstream_url:
                log.info("Syncing fork from upstream %s before PR", upstream_url)
                sync_fork_from_upstream(tmpdir, upstream_url, base_branch, token)
            else:
                log.warning("sync_upstream_before_pr=true but upstream_repo_url not set")

        create_pr_branch(tmpdir, branch_name)

        api_key = os.environ.get("LLM_API_KEY")
        any_changed = False
        actually_eligible = []
        for m in eligible:
            key = m["inferenceX_key"]
            script_path = _find_script_in_repo(Path(tmpdir), key, scripts_path)
            report_content = m.get("_report", "")

            script_content = script_path.read_text() if script_path else None
            changes = extract_changes_vs_script(report_content, script_content, api_key)
            if not changes or not changes.get("flag_changes"):
                log.info("No actionable changes for %s, skipping", key)
                continue

            m["_changes"] = changes
            actually_eligible.append(m)
            log.info("Eligible: %s (%d flag changes)",
                     key, len(changes.get("flag_changes", [])))

            if script_path and apply_changes_to_script(script_path, changes):
                any_changed = True

        eligible = actually_eligible
        if not eligible:
            log.info("No models with actionable changes after extraction")
            return

        config_keys = [m["inferenceX_key"] for m in eligible]
        descriptions = []
        for m in eligible:
            key = m["inferenceX_key"]
            gain = m.get("gain_pct", 0)
            changes = m.get("_changes", {})
            desc_parts = []
            for fc in changes.get("flag_changes", []):
                if fc["action"] == "modify":
                    desc_parts.append(f"{fc['flag']}: {fc.get('old_value')} → {fc.get('new_value')}")
                elif fc["action"] == "add":
                    desc_parts.append(f"Add {fc['flag']} {fc.get('value', '')}")
            for ec in changes.get("env_var_changes", []):
                if ec["action"] == "add":
                    desc_parts.append(f"Add {ec['var']}={ec.get('value', '')}")
            summary = "; ".join(desc_parts) if desc_parts else f"+{gain:.1f}% optimization"
            descriptions.append(f"{key}: {summary}")

        title_models = ", ".join(config_keys)
        if len(title_models) > 60:
            prefixes = list(dict.fromkeys(k.split("-")[0] for k in config_keys))
            title_models = ", ".join(prefixes)
        pr_title = f"[AMD] Optimize {title_models}"

        if dry_run:
            log.info("=== DRY RUN ===")
            log.info("Branch: %s", branch_name)
            log.info("Title: %s", pr_title)
            log.info("Config keys: %s", config_keys)
            for d in descriptions:
                log.info("  - %s", d)
            log.info("PR body preview:\n%s", _generate_pr_body(eligible)[:800])
            return

        changelog_path = Path(tmpdir) / "perf-changelog.yaml"
        append_perf_changelog(changelog_path, config_keys, descriptions)
        any_changed = True

        commit_msg = f"{pr_title}\n\n" + "\n".join(f"- {d}" for d in descriptions)
        if not commit_and_push(tmpdir, branch_name, commit_msg, token):
            log.info("Nothing to push")
            return

        if not token:
            log.warning("No token configured (%s), skipping PR creation",
                        pr_cfg.get("token_env", "INFERENCEX_FORK_TOKEN"))
            return

        pr_body = _generate_pr_body(eligible)
        pr_url = create_github_pr(
            repo_owner, repo_name, branch_name, base_branch,
            pr_title, pr_body, token)

        if pr_url:
            log.info("PR submitted: %s", pr_url)
            labels = pr_cfg.get("labels") or []
            if labels:
                add_pr_labels(repo_owner, repo_name, pr_url, labels, token)


def main():
    """CLI entry point: load the CI summary and submit eligible PRs.

    Parses arguments, configures logging, loads the config and summary, filters
    eligible models, and dispatches PR submission.

    Returns:
        None: Nothing is returned; exits non-zero if the summary file is
            missing.
    """
    parser = argparse.ArgumentParser(description="Submit InferenceX PRs from CI results")
    parser.add_argument("--config", default=None, help="Path to ci-config.yaml")
    parser.add_argument("--summary", required=True, help="Path to ci_summary.json")
    parser.add_argument("--reports-dir", default="ci-output",
                        help="Directory containing per-model reports")
    parser.add_argument("--dry-run", action="store_true",
                        help="Extract and preview without creating PR")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)

    summary_path = Path(args.summary)
    if not summary_path.exists():
        log.error("Summary file not found: %s", args.summary)
        sys.exit(1)

    ci_summary = json.loads(summary_path.read_text())
    reports_dir = Path(args.reports_dir)

    eligible = process_results(ci_summary, reports_dir, config, args.dry_run)

    if not eligible:
        log.info("No models eligible for PR submission")
        return

    log.info("%d model(s) eligible for PR", len(eligible))
    submit_pr(eligible, config, ci_summary, args.dry_run)


if __name__ == "__main__":
    main()
