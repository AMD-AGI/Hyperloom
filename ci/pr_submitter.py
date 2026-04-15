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
from typing import Any

import yaml

log = logging.getLogger("pr-submitter")

CI_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CI_DIR / "ci-config.yaml"

# ── LLM-based extraction (fallback when diff-based fails) ──

LLM_ENDPOINT = "https://oci-slc.primus-safe.amd.com/api/v1/llm-proxy/v1/chat/completions"

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
    """Use LLM to extract structured changes from optimization report."""
    import requests

    prompt = LLM_EXTRACT_PROMPT.format(report_content=report_content[:6000])
    try:
        resp = requests.post(
            LLM_ENDPOINT,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"},
            json={"model": "openai/gpt-4.1-mini",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0, "max_tokens": 500},
            timeout=30, verify=False,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(content)
    except Exception as e:
        log.warning("LLM extraction failed: %s", e)
        return {}


# ── Diff-based extraction (primary, no LLM cost) ──

_SERVER_CMD_RE = re.compile(
    r"```(?:bash)?\s*\n((?:export\s+\S+=\S+\n)*"
    r"(?:python3?\s+-m\s+\S+\.launch_server|vllm\s+serve)\b.*?)\n```",
    re.DOTALL,
)


def _parse_server_block(block: str) -> tuple[dict[str, str], dict[str, str]]:
    """Parse a server launch block into (env_vars, flags)."""
    env_vars: dict[str, str] = {}
    flags: dict[str, str] = {}

    joined = block.replace("\\\n", " ")
    for line in joined.splitlines():
        line = line.strip()
        m = re.match(r"export\s+(\w+)=(.+)", line)
        if m:
            env_vars[m.group(1)] = m.group(2).strip().strip("'\"")
            continue
        if not line or line.startswith("#"):
            continue
        tokens = shlex.split(line, posix=True)
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
            i += 1
    return env_vars, flags


def _diff_server_commands(report: str) -> dict:
    """Extract changes by diffing baseline vs optimized server commands."""
    blocks = _SERVER_CMD_RE.findall(report)
    if len(blocks) < 2:
        return {}

    baseline_env, baseline_flags = _parse_server_block(blocks[0])
    opt_env, opt_flags = _parse_server_block(blocks[-1])

    flag_changes = []
    for flag, new_val in opt_flags.items():
        old_val = baseline_flags.get(flag)
        if old_val is None:
            flag_changes.append({"flag": flag, "value": new_val, "action": "add"})
        elif old_val != new_val:
            flag_changes.append({"flag": flag, "old_value": old_val,
                                 "new_value": new_val, "action": "modify"})
    for flag in baseline_flags:
        if flag not in opt_flags:
            flag_changes.append({"flag": flag, "action": "remove"})

    env_changes = []
    for var, val in opt_env.items():
        if var not in baseline_env:
            env_changes.append({"var": var, "value": val, "action": "add"})
        elif baseline_env[var] != val:
            env_changes.append({"var": var, "old_value": baseline_env[var],
                                "new_value": val, "action": "modify"})
    for var in baseline_env:
        if var not in opt_env:
            env_changes.append({"var": var, "action": "remove"})

    if not flag_changes and not env_changes:
        return {}

    return {
        "flag_changes": flag_changes,
        "env_var_changes": env_changes,
        "baseline_command": blocks[0],
        "optimized_command": blocks[-1],
    }


def extract_changes(report_content: str, api_key: str | None = None) -> dict:
    """Extract config changes from an optimization report.

    Tries diff-based extraction first; falls back to LLM if diff yields nothing.
    """
    result = _diff_server_commands(report_content)
    if result and (result.get("flag_changes") or result.get("env_var_changes")):
        log.info("Extracted changes via command diff")
        return result

    if api_key:
        log.info("Diff extraction empty, trying LLM fallback")
        return _llm_extract_changes(report_content, api_key)

    log.warning("No changes extracted (no diff, no LLM key)")
    return {}


# ── Benchmark script modification ──

def _apply_flag_to_script(content: str, flag: str, value: str | None,
                          action: str) -> str:
    """Apply a single flag change to a benchmark shell script."""
    if action == "modify" and value is not None:
        pattern = re.compile(
            rf"({re.escape(flag)}[\s=])(\S+)",
        )
        if pattern.search(content):
            return pattern.sub(rf"\g<1>{value}", content)

    if action == "add" and value is not None:
        flag_str = f"{flag} {value}" if value else flag
        m = re.search(
            r"(python3?\s+-m\s+\S+\.launch_server\b.*?)(>.*$|\n)",
            content, re.DOTALL,
        )
        if m:
            insert_pos = m.end(1)
            indent = " " * 4
            content = (content[:insert_pos] +
                       f" \\\n{indent}{flag_str}" +
                       content[insert_pos:])
            return content

    if action == "remove":
        pattern = re.compile(
            rf"\s*\\?\s*{re.escape(flag)}(?:[\s=]\S+)?",
        )
        content = pattern.sub("", content)

    return content


def _apply_env_to_script(content: str, var: str, value: str,
                         action: str) -> str:
    """Apply an env var change to a benchmark shell script."""
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
    """Apply extracted changes to an InferenceX benchmark script."""
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
    """Prepend a new entry to perf-changelog.yaml."""
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
    """Generate PR body following InferenceX PR template."""
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
    cmd = ["git"] + args
    log.debug("git %s (cwd=%s)", " ".join(args), cwd)
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)


def clone_fork(fork_url: str, target_dir: str, branch: str = "main"):
    subprocess.run(
        ["git", "clone", "--depth=1", f"--branch={branch}", fork_url, target_dir],
        check=True, capture_output=True, text=True,
    )


def create_pr_branch(repo_dir: str, branch_name: str):
    _run_git(["checkout", "-b", branch_name], repo_dir)


def commit_and_push(repo_dir: str, branch_name: str, message: str,
                    token: str | None = None):
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
    """Create a PR within the same repo using gh CLI or GitHub API."""
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


# ── Main orchestration ──

def load_config(path: str | None = None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p) as f:
        return yaml.safe_load(f)


def _find_script_in_repo(repo_dir: Path, ifx_key: str,
                         scripts_path: str) -> Path | None:
    """Find benchmark script by inferenceX key (exact match then prefix)."""
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


def process_results(
    ci_summary: dict,
    reports_dir: Path,
    config: dict,
    dry_run: bool = False,
) -> list[dict]:
    """Process CI results and return models eligible for PR submission."""
    pr_cfg = config.get("pr_submission", {})
    min_gain = pr_cfg.get("min_gain_pct", 3.0)
    api_key = os.environ.get("LLM_API_KEY")

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

        report_content = model_result.get("report_content", "")
        if not report_content:
            model_name = model_result.get("model", "")
            report_path = reports_dir / model_name / "optimization_report.md"
            if report_path.exists():
                report_content = report_path.read_text()

        if not report_content:
            log.warning("No report for %s, skipping", model_result.get("model"))
            continue

        changes = extract_changes(report_content, api_key)
        if not changes or (not changes.get("flag_changes") and
                           not changes.get("env_var_changes")):
            log.info("No config changes found for %s", model_result.get("model"))
            continue

        model_result["_changes"] = changes
        eligible.append(model_result)
        log.info("Eligible: %s (gain=%.1f%%, vs_ifx=%s, %d flag changes, %d env changes)",
                 model_result["inferenceX_key"], gain,
                 f"{vs_ifx:+.1f}%" if vs_ifx is not None else "N/A",
                 len(changes.get("flag_changes", [])),
                 len(changes.get("env_var_changes", [])))

    return eligible


def submit_pr(
    eligible: list[dict],
    config: dict,
    ci_summary: dict,
    dry_run: bool = False,
):
    """Clone InferenceX repo, apply changes, and submit PR."""
    pr_cfg = config.get("pr_submission", {})
    repo_url = pr_cfg.get("repo_url", "https://github.com/lishuoshuo-amd/InferenceX.git")
    repo_owner = pr_cfg.get("repo_owner", "lishuoshuo-amd")
    repo_name = pr_cfg.get("repo_name", "InferenceX")
    base_branch = pr_cfg.get("base_branch", "main")
    scripts_path = config.get("inferenceX", {}).get("scripts_path", "benchmarks/single_node")

    token = os.environ.get(pr_cfg.get("token_env", "INFERENCEX_FORK_TOKEN"))

    ci_run_id = ci_summary.get("ci_run_id", "unknown")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    branch_name = f"hyperloom/ci-{ts}"

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
        title_models = f"{len(config_keys)} AMD models"
    pr_title = f"[Hyperloom] Optimize {title_models}"

    if dry_run:
        log.info("=== DRY RUN ===")
        log.info("Branch: %s", branch_name)
        log.info("Title: %s", pr_title)
        log.info("Config keys: %s", config_keys)
        for d in descriptions:
            log.info("  - %s", d)
        log.info("PR body preview:\n%s", _generate_pr_body(eligible)[:500])
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        log.info("Cloning %s...", repo_url)
        clone_fork(repo_url, tmpdir, base_branch)
        create_pr_branch(tmpdir, branch_name)

        any_changed = False
        for m in eligible:
            key = m["inferenceX_key"]
            script_path = _find_script_in_repo(Path(tmpdir), key, scripts_path)
            changes = m.get("_changes", {})

            if script_path and apply_changes_to_script(script_path, changes):
                any_changed = True

        changelog_path = Path(tmpdir) / "perf-changelog.yaml"
        append_perf_changelog(changelog_path, config_keys, descriptions)
        any_changed = True

        if not any_changed:
            log.info("No files changed, skipping PR")
            return

        commit_msg = f"[Hyperloom CI] {pr_title}\n\n" + "\n".join(f"- {d}" for d in descriptions)
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


def main():
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
