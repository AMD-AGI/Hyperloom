#!/usr/bin/env python3
"""CI orchestrator: parse config → create Claw sessions → monitor → report."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from claw_client import ClawClient
from inferenceX_parser import (
    fetch_benchmarks,
    find_benchmark,
    find_benchmark_script,
    format_benchmark_for_prompt,
    get_latest_commit,
    get_nfs_root,
    merge_model_config,
    resolve_var,
    synthesize_entry_from_ci_config,
)
from report_generator import (
    build_model_result,
    generate_github_summary,
    generate_json_summary,
    generate_markdown_report,
)

log = logging.getLogger("ci-orchestrator")

CI_DIR = Path(__file__).resolve().parent
PROMPT_TEMPLATE = (CI_DIR / "prompt_template.md").read_text()
PROMPT_TEMPLATE_PR = (CI_DIR / "prompt_template_pr.md").read_text()

# Two prompt paths exist:
#
#   1. prompt_template.md (default, scheduled / workflow_dispatch CI):
#      Agent cd's into /wekafs/HyperloomV2 (= the deployed Hyperloom snapshot
#      on the WekaFS mount) and runs the Python `inference_optimizer optimize`
#      CLI from there. This validates the *deployed* version, not main HEAD —
#      wekafs is hand-maintained and may lag main by 50+ commits with local
#      uncommitted edits. That is the *intended* behavior for schedule runs:
#      we want to know what the production-deployed Hyperloom does today.
#
#   2. prompt_template_pr.md (PR-approve trigger only):
#      Agent first git-clones the PR head commit into the sandbox at
#      /tmp/Hyperloom-pr, installs from there, and drives the skill out of
#      the cloned tree. This validates the *unmerged PR code*, not the
#      wekafs snapshot — required so reviewer-visible numbers actually
#      reflect what the PR proposes to ship. The GitHub token is formatted
#      into the prompt (Claw API has no env-injection field; ClawClient.
#      create_session/send_message bodies only accept name/agent_id/
#      sandbox_image/content/tools/pluginId/workspaceId). The workflow
#      passes ${{ github.token }} (per-run installation token,
#      contents:read on this repo only, lifetime ≤ workflow run) — chosen
#      over a user-created PAT because AMD-AGI org policy bans Classic
#      PATs and gates fine-grained PATs behind admin approval. The flag
#      `--gh-token-env` keeps it pluggable: any env-var holding a token
#      that can clone AMD-AGI/Hyperloom works.
#
# Both paths share the same plugin 4 (tools 3 + 85). The skill runtime IS
# the inference_optimizer Python package — there is no marketplace skill
# download or /workspace/.skills/ layout. ci-mix300 (tool 74) is
# intentionally NOT bundled in plugin 4.


def load_config(config_path: str | None = None) -> dict:
    path = Path(config_path) if config_path else CI_DIR / "ci-config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def render_prompt(merged: dict, *, pr_mode: bool = False,
                  git_ref: str | None = None,
                  gh_token: str | None = None) -> str:
    """Render the agent prompt for the inference_optimizer skill.

    Two paths:
      - pr_mode=False (default): use prompt_template.md, agent cd's into
        /wekafs/HyperloomV2 (deployed snapshot). Used by schedule /
        workflow_dispatch.
      - pr_mode=True: use prompt_template_pr.md, agent first git-clones
        the PR head commit into /tmp inside the sandbox, then installs +
        drives the skill from there. Required so reviewer-visible numbers
        on PR-approve trigger reflect the proposed code, not the
        wekafs-deployed snapshot. Requires both git_ref (PR head sha)
        and gh_token (any GitHub token that can clone AMD-AGI/Hyperloom;
        the workflow passes the per-run GITHUB_TOKEN).
    """
    isl, osl = merged["isl_osl_configs"][0]
    ifx_text = format_benchmark_for_prompt(
        merged["inferenceX_benchmarks"],
        merged["target_gpu"], isl, osl, merged["precision"],
        image=merged.get("image"),
        tp=merged.get("tp"),
        conc=merged.get("conc"),
    )

    # NOTE: SAFE_API_KEY / SAFE_BASE_URL are NOT injected into the prompt
    # body. The previous templates leaked the key by interpolating it into an
    # "Auth: SAFE_API_KEY={safe_api_key}" block; the agent already has those
    # values in its sandbox env (via kernel-agent.env.sh) and does not need
    # them rendered into the prompt text — that path landed the key into Claw
    # session history, Actions logs, and uploaded artifacts. Keep this comment
    # so a future contributor doesn't add the variable back without thinking.

    # Multi-node entries (nodes > 1) need an explicit "Task submission" block
    # describing the RayJob image / per-node resources / RDMA env so the agent
    # can spawn the SaFE RayJob with the right topology and bnxt_re tar package.
    # Single-node entries (nodes == 1, the default) get an empty section — the
    # Claw sandbox is the only execution context they need.
    nodes = int(merged.get("nodes", 1) or 1)
    if nodes > 1:
        rayjob_image = merged.get("rayjob_image", "") or merged.get("sandbox_image", "")
        multinode_section = (
            f"\nTask submission ({nodes}-node):\n"
            f"RayJob image: {rayjob_image}\n"
            f"RayJob resource per node: CPU=96, GPU=8, memory=1024Gi, ephemeralStorage=400Gi\n"
            f"RayJob node count: {nodes}\n"
            f"env:\n"
            f"- PATH_TO_BNXT_TAR_PACKAGE=/wekafs/primus/data/libbnxt/libbnxt_re-234.0.154.0.tar.gz\n"
        )
    else:
        multinode_section = ""

    common_fields = dict(
        model_hf=merged["model_hf"],
        model_path=merged["model_path"],
        framework=merged["framework"],
        precision=merged["precision"],
        isl=isl,
        osl=osl,
        conc=merged["conc"],
        tp=merged["tp"],
        ep=merged["ep"],
        gpu_type=merged["gpu_type"],
        gpu_type_lc=str(merged["gpu_type"]).lower(),
        target_gpu=merged["target_gpu"],
        inferenceX_data=ifx_text,
        oob_path=merged.get("oob_path", ""),
        tracelens_root=merged.get("tracelens_root", ""),
        # Multi-node / Hyperloom-skill knobs (defaults match legacy single-node CI).
        nodes=nodes,
        target_gain=merged.get("target_gain", 10),
        max_hours=merged.get("max_hours", 2),
        random_range_ratio=merged.get("random_range_ratio", 0.8),
        kernel_agent_build_geak_rag_index=merged.get(
            "kernel_agent_build_geak_rag_index", 0,
        ),
        multinode_section=multinode_section,
    )

    if pr_mode:
        if not git_ref:
            raise ValueError("pr_mode=True requires git_ref (PR head sha)")
        if not gh_token:
            raise ValueError(
                "pr_mode=True requires gh_token (any GitHub token that can "
                "clone AMD-AGI/Hyperloom — the workflow passes the per-run "
                "GITHUB_TOKEN via ${{ github.token }}). Set "
                "HYPERLOOM_PR_CI_GH_TOKEN env or pass --gh-token-env"
            )
        return PROMPT_TEMPLATE_PR.format(
            git_ref=git_ref,
            gh_token=gh_token,
            **common_fields,
        )

    return PROMPT_TEMPLATE.format(**common_fields)


_STREAM_TRUNCATION_MIN_TOOL_CALLS = 3  # fewer tool calls than this in <10min = likely truncation
_STREAM_TRUNCATION_MAX_ELAPSED_S = 600  # session completed in under this → suspect truncation
_MAX_RETRY_ATTEMPTS = 4  # 1 initial + up to 3 retries (covers transient Vertex overload bursts)
_OVERLOADED_BACKOFF_S = (180, 360, 720)  # 3min, 6min, 12min — exponential for Vertex overloads
_TRUNCATION_BACKOFF_S = 60                # 1min — fast retry, truncations are usually transient


def _is_stream_truncated(status: str, sse_events: list[dict], elapsed_s: float) -> bool:
    """Detect Vertex AI stream truncation: session 'completed' suspiciously fast with no work."""
    if status != "completed":
        return False
    if elapsed_s > _STREAM_TRUNCATION_MAX_ELAPSED_S:
        return False
    tool_calls = sum(1 for e in sse_events if e.get("type") == "toolUsed")
    return tool_calls < _STREAM_TRUNCATION_MIN_TOOL_CALLS


def _detect_vertex_overloaded(sse_events: list[dict]) -> bool:
    """Detect Anthropic Vertex `overloaded_error` in SSE event tail.

    Pattern observed in real failures:
      {"type":"error","error":{"details":null,"type":"overloaded_error",
       "message":"Overloaded"},"request_id":"req_vrtx_..."}

    The error is usually emitted within the last few events before the session
    terminates. Scan a window large enough to catch it but small enough to
    avoid false positives from `overloaded` appearing in user content.
    """
    # Inspect tail of the stream — last 50 events is enough for any real session.
    for evt in sse_events[-50:]:
        if not isinstance(evt, dict):
            continue
        # Direct error events
        evt_type = evt.get("type", "")
        if evt_type in ("error", "agent_error", "stream_error"):
            err = evt.get("error") if isinstance(evt.get("error"), dict) else evt
            err_type = (err.get("type") or "").lower() if isinstance(err, dict) else ""
            err_msg = (err.get("message") or "").lower() if isinstance(err, dict) else ""
            if "overloaded" in err_type or "overloaded" in err_msg:
                return True
        # Sometimes errors get embedded inside chatDelta/statusUpdate as the
        # provider response. Use a narrow JSON-string match keyed on req_vrtx_
        # so we don't false-positive on the literal word "overloaded" in chat
        # content.
        blob = json.dumps(evt, default=str)
        if "overloaded_error" in blob and "req_vrtx_" in blob:
            return True
    return False


def run_model(
    claw: ClawClient,
    merged: dict,
    nfs_base: str,
    sandbox_timeout: int,
    *,
    pr_mode: bool = False,
    git_ref: str | None = None,
    gh_token: str | None = None,
) -> dict:
    """Execute the full optimization flow for a single model.

    pr_mode + git_ref + gh_token are forwarded to render_prompt to switch
    between schedule path (/wekafs/HyperloomV2) and PR-approve path
    (sandbox-internal git clone of PR head). See render_prompt() docstring.
    """
    model_name = merged["model_hf"].split("/")[-1]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    result_dir = os.path.join(tempfile.gettempdir(), "hyperloom-ci", model_name, timestamp)

    log.info("═" * 60)
    log.info("Starting model: %s (%s)", model_name, merged["precision"])
    if pr_mode:
        log.info("PR-CI mode: validating commit %s (sandbox will git-clone PR head)",
                 git_ref[:12] if git_ref else "?")
    log.info("═" * 60)

    prompt = render_prompt(merged, pr_mode=pr_mode,
                           git_ref=git_ref, gh_token=gh_token)
    # NEVER log full prompt in pr_mode — it contains the GH token.
    log.info("Prompt length: %d chars (pr_mode=%s)", len(prompt), pr_mode)

    status = "failed"
    session_id = None

    last_failure_reason: str | None = None  # "overloaded" | "truncated" | None
    for attempt in range(1, _MAX_RETRY_ATTEMPTS + 1):
        if attempt > 1:
            # Pick backoff based on what killed the previous attempt.
            if last_failure_reason == "overloaded":
                idx = min(attempt - 2, len(_OVERLOADED_BACKOFF_S) - 1)
                wait_s = _OVERLOADED_BACKOFF_S[idx]
                log.warning(
                    "[%s] Vertex overloaded_error on attempt %d — waiting %ds (~%dmin) then retrying",
                    model_name, attempt - 1, wait_s, wait_s // 60,
                )
            else:
                wait_s = _TRUNCATION_BACKOFF_S
                log.warning(
                    "[%s] Stream truncation on attempt %d — waiting %ds then retrying",
                    model_name, attempt - 1, wait_s,
                )
            time.sleep(wait_s)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")

        # Step 1: Create session — pass sandbox_image to get GPU sandbox
        session_name = f"ci-{model_name}-{timestamp}"
        if attempt > 1:
            session_name = f"ci-{model_name}-{timestamp}-retry{attempt - 1}"
        sandbox_image = merged.get("sandbox_image", "")
        try:
            session = claw.create_session(session_name, sandbox_image=sandbox_image or None)
            session_id = session["session_id"]
        except Exception as e:
            log.error("Failed to create session for %s: %s", model_name, e)
            break

        log.info("Session created: %s (attempt %d)", session_id, attempt)

        # Step 2: Subscribe SSE in background thread, then send message
        sse_events: list[dict] = []
        status_holder = {"status": "running"}
        attempt_start = time.time()

        def _monitor():
            def _on_event(evt):
                sse_events.append(evt)
                evt_type = evt.get("type", "")
                if evt_type == "chatDelta" and evt.get("sender") == "assistant":
                    delta = evt.get("delta", {})
                    content = delta.get("content", "") if isinstance(delta, dict) else ""
                    if content and len(content) > 20:
                        log.info("[%s] Agent: ...%s", model_name, content[:150])
                elif evt_type == "toolUsed":
                    log.info("[%s] Tool: %s", model_name, evt.get("tool", "unknown"))

            status_holder["status"] = claw.monitor_session(
                session_id, timeout=sandbox_timeout, on_event=_on_event)

        monitor_thread = threading.Thread(target=_monitor, daemon=True)
        monitor_thread.start()
        time.sleep(1)

        try:
            # Per-entry plugin_id override (merged["claw_plugin_id"] defaults
            # to 4 for entries that don't set it; null in ci-config opts the
            # entry out of the Hyperloom plugin entirely — claw_client.
            # send_message will omit the pluginId field from the JSON body
            # when plugin_id is None).
            entry_plugin_id = merged.get("claw_plugin_id", 4)
            claw.send_message(session_id, prompt, plugin_id=entry_plugin_id)
            log.info(
                "Prompt sent to session %s (plugin_id=%s)",
                session_id, entry_plugin_id,
            )
        except Exception as e:
            log.error("Failed to send message to %s: %s", session_id, e)
            status_holder["status"] = "failed"

        monitor_thread.join(timeout=sandbox_timeout + 60)
        status = status_holder["status"]
        elapsed = time.time() - attempt_start

        if status == "running":
            status = "timeout"
            log.warning("Session %s still running after sandbox_timeout (%ds), marking as timeout",
                        session_id, sandbox_timeout)

        log.info("Session %s finished with status: %s (elapsed %.0fs, %d tool calls)",
                 session_id, status, elapsed, sum(1 for e in sse_events if e.get("type") == "toolUsed"))

        # Decide whether the failure looks like one we can recover from:
        #   - overloaded_error: upstream Vertex capacity blip; exponential backoff
        #   - stream truncation: session marked completed too quickly; short retry
        # Order matters — overloaded check is broader and may catch sessions that
        # report status="failed", not just status="completed".
        if _detect_vertex_overloaded(sse_events):
            last_failure_reason = "overloaded"
            log.warning(
                "[%s] Anthropic Vertex overloaded_error detected (status=%s, elapsed=%.0fs, %d tool calls)",
                model_name, status, elapsed,
                sum(1 for e in sse_events if e.get("type") == "toolUsed"),
            )
            if attempt < _MAX_RETRY_ATTEMPTS:
                continue
            log.error("[%s] Vertex overloaded on all %d attempts — giving up", model_name, _MAX_RETRY_ATTEMPTS)
            status = "failed"
        elif _is_stream_truncated(status, sse_events, elapsed):
            last_failure_reason = "truncated"
            log.warning(
                "[%s] Stream truncation suspected (completed in %.0fs with %d tool calls)",
                model_name, elapsed,
                sum(1 for e in sse_events if e.get("type") == "toolUsed"),
            )
            if attempt < _MAX_RETRY_ATTEMPTS:
                continue
            log.error(
                "[%s] Stream truncation on all %d attempts, giving up",
                model_name, _MAX_RETRY_ATTEMPTS,
            )
            status = "failed"
        break

    # Step 3: Download optimization report from Claw, with NFS fallback
    # No sleep here — sandbox is already gone after session ends; download immediately.
    report_content = None
    os.makedirs(result_dir, exist_ok=True)
    # All three suffixes are part of the canonical Required Artifacts
    # contract (see ci/prompt_prefix.txt and DEFAULT_ARTIFACT_PATTERNS /
    # _KEY_RESULT_SUFFIXES in optimize_submit.py). session_breakdown.json
    # is the V2 dashboard contract emitted by inference_optimizer/cli.py's
    # finally block (or, for non-cli pipelines, manually written by the
    # agent with the documented schema). claw-stats-service prefers
    # session_breakdown.json over ci_metrics.json for dashboard rollup, so
    # this list MUST stay in sync with optimize_submit.py — adding or
    # removing entries should happen in both places.
    download_suffixes = (
        "optimization_report.md",
        "ci_metrics.json",
        "session_breakdown.json",
    )

    # 3a. Try Claw file API (sandbox /workspace)
    try:
        if not session_id:
            raise RuntimeError("No session_id — session never created")
        files = claw.list_files(session_id)
        log.info("Session %s has %d files", session_id, len(files))
        for f in files:
            fpath = f.get("path") or f.get("Path") or ""
            if not fpath or not any(fpath.endswith(s) for s in download_suffixes):
                continue
            local = os.path.join(result_dir, os.path.basename(fpath))
            try:
                claw.download_file_to(session_id, fpath, local)
                if fpath.endswith("optimization_report.md"):
                    report_content = Path(local).read_text()
            except Exception as e:
                log.warning("Failed to download %s: %s", fpath, e)
    except Exception as e:
        log.warning("Failed to list/download files for %s: %s", session_id, e)

    # 3b. NFS fallback: scan NFS for results written by PyTorchJob/RayJob.
    if not report_content:
        nfs_root = get_nfs_root()
        nfs_scan_dirs = [
            f"{nfs_root}/hyperloom-results",
            f"{nfs_root}/results/ci",
            f"{nfs_root}/inference-optimization/results",
        ]
        model_short = model_name.lower().replace("-", "").replace("_", "")
        for scan_dir in nfs_scan_dirs:
            if not os.path.isdir(scan_dir):
                continue
            for entry in sorted(os.listdir(scan_dir), reverse=True):
                entry_clean = entry.lower().replace("-", "").replace("_", "")
                if model_short not in entry_clean:
                    continue
                candidate = os.path.join(scan_dir, entry)
                if not os.path.isdir(candidate):
                    continue
                for suffix in download_suffixes:
                    for root, _, fnames in os.walk(candidate):
                        for fn in fnames:
                            if fn.endswith(suffix.split("/")[-1]):
                                src = os.path.join(root, fn)
                                dst = os.path.join(result_dir, fn)
                                if not os.path.exists(dst):
                                    shutil.copy2(src, dst)
                                    log.info("NFS fallback: copied %s → %s", src, dst)
                                    if fn == "optimization_report.md":
                                        report_content = Path(dst).read_text()
                if report_content:
                    log.info("NFS fallback found results in %s", candidate)
                    break
            if report_content:
                break

    # Step 4: Build result
    ifx_ref = None
    if merged.get("inferenceX_benchmarks"):
        isl, osl = merged["isl_osl_configs"][0]
        ifx_ref = find_benchmark(
            merged["inferenceX_benchmarks"],
            merged["target_gpu"], isl, osl, merged["precision"],
            image=merged.get("image"),
            tp=merged.get("tp"),
            conc=merged.get("conc"))

    result = build_model_result(
        model_name, merged["inferenceX_key"], merged["image"],
        merged["precision"], status, timestamp, result_dir, ifx_ref)
    result["target_gpu"] = merged.get("target_gpu", "")
    if report_content and not result.get("report_content"):
        result["report_content"] = report_content
        result["report_exists"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description="Hyperloom CI/CD Orchestrator")
    parser.add_argument("--config", default=None, help="Path to ci-config.yaml")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model subset (matches per-entry `key` field, "
                             "fallback `inferenceX_key`)")
    parser.add_argument("--trigger", default="manual", help="Trigger type: scheduled/manual/inferenceX")
    parser.add_argument("--plugin-id", type=int, default=None,
                        help="Override claw.plugin_id from ci-config.yaml. Used by the "
                             "Inference A/B Test workflow to compare different Hyperloom-side "
                             "plugin builds (e.g. 4 = current wekafs inference_optimizer; "
                             "5 = an experimental successor with a different SKILL.md).")
    parser.add_argument("--pr-mode", action="store_true",
                        help="PR-CI mode: render prompt_template_pr.md so the agent "
                             "first git-clones --git-ref into /tmp/Hyperloom-pr inside "
                             "the sandbox, then installs + drives the skill from PR head "
                             "(NOT from /wekafs/HyperloomV2). Required so reviewer-visible "
                             "numbers reflect the proposed code, not the wekafs snapshot.")
    parser.add_argument("--git-ref", default=None,
                        help="In --pr-mode: PR head commit sha (or branch name) for the "
                             "agent to clone. Use the full sha — branch name on the PR "
                             "could move during review.")
    parser.add_argument("--gh-token-env", default="HYPERLOOM_PR_CI_GH_TOKEN",
                        help="In --pr-mode: env var holding a GitHub token that can "
                             "clone AMD-AGI/Hyperloom inside the sandbox. The workflow "
                             "passes ${{ github.token }} (per-run GITHUB_TOKEN) — "
                             "AMD-AGI org policy bans Classic PATs and gates "
                             "fine-grained PATs behind admin approval, so the built-in "
                             "GITHUB_TOKEN is the only viable choice. The token value "
                             "gets formatted into the prompt — Claw API has no "
                             "env-injection field — but GITHUB_TOKEN dies with "
                             "the workflow run, so leak blast radius is bounded.")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without executing")
    parser.add_argument("--output-dir", default="ci-output", help="Output directory for reports")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)
    claw_cfg = config["claw"]
    ifx_cfg = config["inferenceX"]
    defaults = config.get("defaults", {})
    results_cfg = config.get("results", {})

    harbor_prefix = resolve_var(ifx_cfg.get("harbor_prefix", ""))
    claw_cfg["endpoint"] = resolve_var(claw_cfg["endpoint"])
    results_cfg["nfs_base"] = resolve_var(results_cfg.get("nfs_base", ""))
    defaults["inferencex_path"] = resolve_var(defaults.get("inferencex_path", ""))
    defaults["oob_path"] = resolve_var(defaults.get("oob_path", ""))
    defaults["tracelens_root"] = resolve_var(defaults.get("tracelens_root", ""))
    for m in config.get("models", []):
        if "model_path_override" in m:
            m["model_path_override"] = resolve_var(m["model_path_override"])

    # Resolve which models to run
    # Each ci-config entry has either an explicit `key` (matrix display id) or
    # falls back to `inferenceX_key`. We filter by the effective key so multiple
    # entries can share the same `inferenceX_key` (e.g. single-node + multi-node
    # DSR1 both look up `dsr1-fp8-mi300x-sglang` in amd-master.yaml but expose
    # distinct keys to the matrix).
    def _entry_key(m: dict) -> str:
        return m.get("key") or m["inferenceX_key"]

    model_list = config.get("models", [])
    if args.models:
        selected = set(args.models.split(","))
        model_list = [m for m in model_list if _entry_key(m) in selected]
        if not model_list:
            log.error("No models matched: %s", args.models)
            sys.exit(1)

    log.info("Models to process: %s", [_entry_key(m) for m in model_list])

    # Fetch InferenceX config + benchmark scripts in a single shallow clone
    log.info("Fetching InferenceX config from main...")
    ifx_commit = get_latest_commit(ifx_cfg["repo"])
    scripts_path = ifx_cfg.get("scripts_path", "benchmarks/single_node")
    ifx_scripts: dict[str, str | None] = {}

    with tempfile.TemporaryDirectory() as _tmpdir:
        subprocess.run(
            ["git", "clone", "--depth=1", "--branch=main",
             ifx_cfg["repo"], _tmpdir],
            check=True, capture_output=True, text=True,
        )
        yaml_path = Path(_tmpdir) / ifx_cfg["config_path"]

        # Read-only image version report. We deliberately don't pass --update —
        # CI baselines must stay pinned to the InferenceX image tag (otherwise
        # tok/s comparisons across runs are apples-to-oranges). Operators who
        # need a different image override it explicitly via ci-config.yaml.
        check_script = CI_DIR.parent / "inference_optimization" / "InferenceX" / "utils" / "check_image_versions.py"
        if check_script.exists():
            cmd = [sys.executable, str(check_script), "--config-files", str(yaml_path)]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                                        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                if result.stdout:
                    log.info("Image version check:\n%s", result.stdout.strip())
                if result.returncode != 0 and result.stderr:
                    log.warning("Image version check stderr: %s", result.stderr.strip())
            except Exception as e:
                log.warning("Image version check failed (non-fatal): %s", e)
        else:
            log.warning("check_image_versions.py not found at %s, skipping", check_script)

        with open(yaml_path) as f:
            amd_master = yaml.safe_load(f)
        log.info("InferenceX commit: %s", ifx_commit[:7])

        ifx_script_contents: dict[str, str] = {}
        for model_cfg in model_list:
            ifx_key = model_cfg.get("inferenceX_key", "")
            if not ifx_key:
                # Self-contained entry (no upstream InferenceX baseline). The
                # agent constructs the server launch directly from the prompt.
                continue
            script = find_benchmark_script(_tmpdir, ifx_key, scripts_path)
            ifx_scripts[ifx_key] = script
            if script:
                log.info("Found benchmark script for %s: %s", ifx_key, script)
                script_path = Path(_tmpdir) / script
                ifx_script_contents[ifx_key] = script_path.read_text() if script_path.exists() else ""
            else:
                log.warning("No benchmark script found for %s", ifx_key)
                ifx_script_contents[ifx_key] = ""

    # Merge configs and fetch API data
    merged_models = []
    for model_cfg in model_list:
        ifx_key = model_cfg.get("inferenceX_key", "")
        entry_label = ifx_key or model_cfg.get("key", "<unnamed>")

        if ifx_key and ifx_key in amd_master:
            # Standard path: pull image/precision/framework/etc. from amd-master.yaml.
            ifx_entry = amd_master[ifx_key]
        elif model_cfg.get("model_hf") and model_cfg.get("image"):
            # Self-contained Hyperloom-internal entry (no InferenceX baseline).
            # Build a synthetic amd-master-style entry directly from ci-config
            # fields so the rest of the merge flow works unchanged.
            ifx_entry = synthesize_entry_from_ci_config(model_cfg)
            log.info(
                "Model %s: using self-contained ci-config entry "
                "(no amd-master.yaml lookup)",
                entry_label,
            )
        else:
            log.warning(
                "Model %s: inferenceX_key %r not in amd-master.yaml AND "
                "ci-config entry lacks model_hf/image — skipping",
                entry_label, ifx_key,
            )
            continue

        api_name = model_cfg.get("inferenceX_api_name", "")
        ifx_benchmarks = []
        if api_name:
            log.info("Fetching InferenceX benchmarks for %s...", api_name)
            try:
                ifx_benchmarks = fetch_benchmarks(api_name, ifx_cfg.get("api_url"))
                log.info("  Got %d benchmark entries", len(ifx_benchmarks))
            except Exception as e:
                log.warning("Failed to fetch benchmarks for %s: %s", api_name, e)

        merged = merge_model_config(
            model_cfg, ifx_entry, defaults, harbor_prefix, ifx_benchmarks)
        merged["benchmark_script"] = ifx_scripts.get(ifx_key)
        merged["benchmark_script_content"] = ifx_script_contents.get(ifx_key, "")
        if results_cfg.get("result_dir"):
            merged["result_dir"] = resolve_var(results_cfg["result_dir"])
        merged_models.append(merged)

    if not merged_models:
        log.error("No valid models to process")
        sys.exit(1)

    # PR-mode: resolve + validate token / git-ref upfront so we fail fast
    # before spinning up any sandbox.
    pr_mode = bool(args.pr_mode)
    git_ref: str | None = None
    gh_token: str | None = None
    if pr_mode:
        if not args.git_ref:
            log.error("--pr-mode requires --git-ref (PR head commit sha)")
            sys.exit(2)
        git_ref = args.git_ref
        gh_token = os.environ.get(args.gh_token_env)
        if not gh_token:
            log.error("--pr-mode requires env %s to hold a GitHub token (got empty/unset). "
                      "The workflow normally passes ${{ github.token }} here.",
                      args.gh_token_env)
            sys.exit(2)
        log.info("PR-CI mode enabled: git_ref=%s, gh_token=*** (from %s)",
                 git_ref[:12] if len(git_ref) > 12 else git_ref, args.gh_token_env)

    # Dry run: print prompts and exit. Mask the GH token in pr_mode so the
    # log doesn't leak it.
    if args.dry_run:
        for merged in merged_models:
            prompt = render_prompt(merged, pr_mode=pr_mode,
                                   git_ref=git_ref,
                                   gh_token=gh_token)
            if pr_mode and gh_token:
                prompt = prompt.replace(gh_token, "***GH_TOKEN_REDACTED***")
            print(f"\n{'=' * 60}")
            print(f"Model: {merged['model_hf']} ({merged['precision']}) "
                  f"[pr_mode={pr_mode}]")
            print(f"{'=' * 60}")
            print(prompt)
        sys.exit(0)

    if args.plugin_id is not None:
        claw_cfg["plugin_id"] = args.plugin_id
        log.info("Overriding Claw plugin_id from CLI: %s", claw_cfg["plugin_id"])

    claw = ClawClient.from_config(claw_cfg)
    nfs_base = results_cfg.get("nfs_base") or (get_nfs_root() + "/results/ci")
    sandbox_timeout = claw_cfg.get("sandbox_timeout", 14400)
    results = []

    for merged in merged_models:
        result = run_model(
            claw, merged, nfs_base, sandbox_timeout,
            pr_mode=pr_mode, git_ref=git_ref, gh_token=gh_token,
        )
        results.append(result)
        log.info("Result for %s: status=%s, gain=%s",
                 result["model"], result["status"],
                 f"{result['gain_pct']}%" if result.get("gain_pct") is not None else "N/A")

    # Generate reports — wrap in try/except so report-generation bugs never
    # discard the expensive LLM results that already completed successfully.
    ci_run_id = f"ci-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_summary = None
    try:
        md_report = generate_markdown_report(results, args.trigger, ifx_commit, ci_run_id)
        (out_dir / "ci_report.md").write_text(md_report)
    except Exception:
        log.exception("generate_markdown_report failed — writing raw results as fallback")
        (out_dir / "ci_report_raw.json").write_text(
            json.dumps(results, indent=2, default=str))

    try:
        json_summary = generate_json_summary(results, args.trigger, ifx_commit, ci_run_id)
        (out_dir / "ci_summary.json").write_text(json.dumps(json_summary, indent=2))
    except Exception:
        log.exception("generate_json_summary failed — writing raw results as fallback")
        (out_dir / "ci_summary_raw.json").write_text(
            json.dumps(results, indent=2, default=str))

    # Per-model report files (for individual artifact downloads)
    for r in results:
        try:
            model_dir = out_dir / r["model"]
            model_dir.mkdir(parents=True, exist_ok=True)
            report_content = r.get("report_content") or r.get("optimization_report")
            if report_content:
                (model_dir / "optimization_report.md").write_text(report_content)
        except Exception:
            log.exception("Failed to write per-model report for %s", r.get("model"))

    # GitHub Actions Job Summary
    try:
        github_summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
        gh_summary = generate_github_summary(results, args.trigger, ifx_commit)
        if github_summary_file:
            with open(github_summary_file, "a") as f:
                f.write(gh_summary)
            log.info("GitHub Summary written to $GITHUB_STEP_SUMMARY")
        else:
            (out_dir / "github_summary.md").write_text(gh_summary)
            log.info("GitHub Summary written to %s/github_summary.md (not in CI)", out_dir)
    except Exception:
        log.exception("GitHub summary generation failed")

    log.info("Reports written to %s", out_dir)

    if json_summary:
        log.info("Summary: %d models, %d completed, %d failed",
                 json_summary["stats"]["total"],
                 json_summary["stats"]["completed"],
                 json_summary["stats"]["failed"])

        if json_summary["stats"]["avg_gain_pct"] is not None:
            log.info("Average gain: %.1f%%", json_summary["stats"]["avg_gain_pct"])
    else:
        completed = sum(1 for r in results if r.get("status") == "completed")
        failed = sum(1 for r in results if r.get("status") == "failed")
        log.info("Summary (fallback): %d models, %d completed, %d failed",
                 len(results), completed, failed)

    # Webhook notification (Slack / Teams / custom)
    try:
        webhook_env = config.get("notification", {}).get("webhook_env")
        if webhook_env:
            webhook = os.environ.get(webhook_env)
            if webhook and json_summary:
                _send_webhook(webhook, json_summary)
    except Exception:
        log.exception("Webhook notification failed")

    # Exit non-zero only if ALL models failed (never discard partial success)
    completed_count = (json_summary["stats"]["completed"] if json_summary
                       else sum(1 for r in results if r.get("status") == "completed"))
    if completed_count == 0:
        timeout_count = sum(1 for r in results if r.get("status") == "timeout")
        failed_count = sum(1 for r in results if r.get("status") == "failed")
        log.error("All models failed! (failed=%d, timeout=%d)", failed_count, timeout_count)
        sys.exit(1)


def _send_webhook(webhook: str, summary: dict):
    """Send notification via webhook. Uses Adaptive Card for Teams, plain text fallback."""
    import requests as req
    r = summary["models"][0] if summary["models"] else {}
    model = r.get("model", "unknown")
    status = r.get("status", "unknown")
    precision = r.get("precision", "")
    trigger = summary.get("trigger", "manual")

    def _val(key, fmt=".2f"):
        v = r.get(key)
        return f"{v:{fmt}}" if v is not None else "N/A"

    def _row(label, value, color=None):
        items = [{"type": "TextBlock", "text": str(value)}]
        if color:
            items[0]["color"] = color
        return {"type": "TableRow", "cells": [
            {"type": "TableCell", "items": [{"type": "TextBlock", "text": label}]},
            {"type": "TableCell", "items": items},
        ]}

    gain = r.get("gain_pct")
    gain_color = "Good" if gain and gain > 0 else "Attention" if gain else None
    vs_ifx = r.get("vs_inferenceX_pct")
    vs_color = "Good" if vs_ifx and vs_ifx > 0 else "Attention" if vs_ifx else None

    rows = [
        {"type": "TableRow", "style": "accent", "cells": [
            {"type": "TableCell", "items": [{"type": "TextBlock", "text": "Metric", "weight": "Bolder"}]},
            {"type": "TableCell", "items": [{"type": "TextBlock", "text": "Value", "weight": "Bolder"}]},
        ]},
        _row("Baseline (output tok/s/GPU)", _val("baseline_tok_per_gpu")),
        _row("Optimized (output tok/s/GPU)", _val("optimized_tok_per_gpu")),
        _row("**Optimization Gain**", f"**{'--' if gain is not None and abs(gain) < 0.05 else f'{gain:+.1f}%' if gain is not None else 'N/A'}**", gain_color),
    ]
    if r.get("inferenceX_tok_per_gpu"):
        rows.append(_row("InferenceX (output tok/s/GPU)", _val("inferenceX_tok_per_gpu")))
        rows.append(_row("**vs InferenceX**",
                         f"**{'--' if vs_ifx is not None and abs(vs_ifx) < 0.05 else f'{vs_ifx:+.1f}%' if vs_ifx is not None else 'N/A'}**", vs_color))

    status_emoji = {"completed": "\u2705", "failed": "\u274c", "timeout": "\u23f1"}.get(status, "\u2753")

    card = {
        "type": "message",
        "attachments": [{
            "contentType": "application/vnd.microsoft.card.adaptive",
            "content": {
                "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                "type": "AdaptiveCard", "version": "1.4",
                "body": [
                    {"type": "TextBlock", "weight": "Bolder", "size": "Large",
                     "text": f"{status_emoji} Hyperloom CI — {model} ({precision})"},
                    {"type": "TextBlock", "isSubtle": True, "spacing": "None",
                     "text": f"Trigger: {trigger} | Image: {r.get('image', 'N/A')}"},
                    {"type": "Table", "columns": [{"width": 2}, {"width": 1}], "rows": rows},
                ],
            },
        }],
    }

    try:
        resp = req.post(webhook, json=card, timeout=10)
        if resp.status_code >= 300:
            req.post(webhook, json={
                "text": f"{status_emoji} Hyperloom CI [{model}]: {status} | Gain: {_val('gain_pct','+.1f')}%"
            }, timeout=10)
    except Exception as e:
        log.warning("Webhook notification failed: %s", e)


if __name__ == "__main__":
    main()
