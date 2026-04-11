#!/usr/bin/env python3
"""CI orchestrator: parse config → create Claw sessions → monitor → report."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
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
    merge_model_config,
    resolve_var,
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


def load_config(config_path: str | None = None) -> dict:
    path = Path(config_path) if config_path else CI_DIR / "ci-config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def render_prompt(merged: dict) -> str:
    isl, osl = merged["isl_osl_configs"][0]
    ifx_text = format_benchmark_for_prompt(
        merged["inferenceX_benchmarks"],
        merged["target_gpu"], isl, osl, merged["precision"],
        image=merged.get("image"),
    )
    script = merged.get("benchmark_script")
    if script:
        bss = (
            f"MANDATORY: Read the InferenceX benchmark script and replicate its server config:\n"
            f"  {merged['inferencex_path']}/{script}\n"
            f"Key env vars for this run: MODEL={merged['model_path']} TP={merged['tp']} "
            f"EP_SIZE={merged['ep']} CONC={merged['conc']} ISL={isl} OSL={osl} "
            f"MAX_MODEL_LEN=4096 RANDOM_RANGE_RATIO=0.8 "
            f"RESULT_FILENAME=baseline\n"
            f"DO NOT run the script directly (it depends on benchmark_lib.sh and hf download).\n"
            f"Instead: read the script, extract the server launch command, env vars, and "
            f"server flags, then use them to launch the server and run the benchmark yourself."
        )
    else:
        bss = "No InferenceX benchmark script found. Construct server launch manually."

    return PROMPT_TEMPLATE.format(
        model_hf=merged["model_hf"],
        mode=merged["mode"],
        model_path=merged["model_path"],
        framework=merged["framework"],
        precision=merged["precision"],
        isl=isl,
        osl=osl,
        conc=merged["conc"],
        tp=merged["tp"],
        ep=merged["ep"],
        gpu_type=merged["gpu_type"],
        inferencex_path=merged["inferencex_path"],
        sandbox_image=merged["sandbox_image"],
        kernel_opt_backends=merged["kernel_opt_backends"],
        kernel_opt_image=merged["kernel_opt_image"],
        kernel_opt_workspace=merged["kernel_opt_workspace"],
        geak_step_limit=merged["geak_step_limit"],
        min_kernels=merged["min_kernels"],
        result_dir=merged["result_dir"],
        target_gpu=merged["target_gpu"],
        inferenceX_data=ifx_text,
        runner=merged["runner"],
        benchmark_script_section=bss,
    )


def run_model(
    claw: ClawClient,
    merged: dict,
    nfs_base: str,
    sandbox_timeout: int,
) -> dict:
    """Execute the full optimization flow for a single model."""
    model_name = merged["model_hf"].split("/")[-1]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    result_dir = f"{nfs_base}/{model_name}/{timestamp}"

    log.info("═" * 60)
    log.info("Starting model: %s (%s)", model_name, merged["precision"])
    log.info("═" * 60)

    # Step 1: Create session
    session_name = f"ci-{model_name}-{timestamp}"
    try:
        session = claw.create_session(session_name)
        session_id = session["session_id"]
    except Exception as e:
        log.error("Failed to create session for %s: %s", model_name, e)
        return build_model_result(
            model_name, merged["inferenceX_key"], merged["image"],
            merged["precision"], "failed", timestamp, result_dir)

    log.info("Session created: %s", session_id)

    # Step 2: Subscribe SSE in background thread, then send message
    prompt = render_prompt(merged)
    log.info("Prompt length: %d chars", len(prompt))

    sse_events: list[dict] = []
    status_holder = {"status": "running"}

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

    # Send the prompt after SSE is connected
    try:
        claw.send_message(session_id, prompt)
        log.info("Prompt sent to session %s", session_id)
    except Exception as e:
        log.error("Failed to send message to %s: %s", session_id, e)
        status_holder["status"] = "failed"

    # Wait for completion
    monitor_thread.join(timeout=sandbox_timeout + 60)
    status = status_holder["status"]
    if status == "running":
        status = "timeout"
        log.warning("Session %s still running after sandbox_timeout (%ds), marking as timeout",
                     session_id, sandbox_timeout)
    log.info("Session %s finished with status: %s", session_id, status)

    # Step 3: Download optimization report from Claw
    report_content = None
    try:
        files = claw.list_files(session_id)
        log.info("Session %s has %d files", session_id, len(files))
        os.makedirs(result_dir, exist_ok=True)
        download_suffixes = ("optimization_report.md", "ci_metrics.json")
        for f in files:
            fpath = f["path"]
            if not any(fpath.endswith(s) for s in download_suffixes):
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

    # Step 4: Build result
    ifx_ref = None
    if merged.get("inferenceX_benchmarks"):
        isl, osl = merged["isl_osl_configs"][0]
        ifx_ref = find_benchmark(
            merged["inferenceX_benchmarks"],
            merged["target_gpu"], isl, osl, merged["precision"],
            image=merged.get("image"))

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
    parser.add_argument("--models", default=None, help="Comma-separated model subset (inferenceX_key)")
    parser.add_argument("--trigger", default="manual", help="Trigger type: scheduled/manual/inferenceX")
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

    # Resolve which models to run
    model_list = config.get("models", [])
    if args.models:
        selected = set(args.models.split(","))
        model_list = [m for m in model_list if m["inferenceX_key"] in selected]
        if not model_list:
            log.error("No models matched: %s", args.models)
            sys.exit(1)

    log.info("Models to process: %s", [m["inferenceX_key"] for m in model_list])

    # Fetch InferenceX config + benchmark scripts in a single shallow clone
    log.info("Fetching InferenceX config from main...")
    ifx_commit = get_latest_commit(ifx_cfg["repo"])
    scripts_path = ifx_cfg.get("scripts_path", "benchmarks/single_node")
    ifx_scripts: dict[str, str | None] = {}

    import tempfile as _tmpmod
    with _tmpmod.TemporaryDirectory() as _tmpdir:
        subprocess.run(
            ["git", "clone", "--depth=1", "--branch=main",
             ifx_cfg["repo"], _tmpdir],
            check=True, capture_output=True, text=True,
        )
        yaml_path = Path(_tmpdir) / ifx_cfg["config_path"]
        with open(yaml_path) as f:
            amd_master = yaml.safe_load(f)
        log.info("InferenceX commit: %s", ifx_commit[:7])

        for model_cfg in model_list:
            ifx_key = model_cfg["inferenceX_key"]
            script = find_benchmark_script(_tmpdir, ifx_key, scripts_path)
            ifx_scripts[ifx_key] = script
            if script:
                log.info("Found benchmark script for %s: %s", ifx_key, script)
            else:
                log.warning("No benchmark script found for %s", ifx_key)

    # Merge configs and fetch API data
    merged_models = []
    for model_cfg in model_list:
        ifx_key = model_cfg["inferenceX_key"]
        if ifx_key not in amd_master:
            log.warning("Model %s not found in amd-master.yaml, skipping", ifx_key)
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
            model_cfg, amd_master[ifx_key], defaults, harbor_prefix, ifx_benchmarks)
        merged["benchmark_script"] = ifx_scripts.get(ifx_key)
        merged_models.append(merged)

    if not merged_models:
        log.error("No valid models to process")
        sys.exit(1)

    # Dry run: print prompts and exit
    if args.dry_run:
        for merged in merged_models:
            prompt = render_prompt(merged)
            print(f"\n{'=' * 60}")
            print(f"Model: {merged['model_hf']} ({merged['precision']})")
            print(f"{'=' * 60}")
            print(prompt)
        sys.exit(0)

    # Execute
    claw = ClawClient.from_config(claw_cfg)
    nfs_base = results_cfg.get("nfs_base", "/hyperloom/results/ci")
    sandbox_timeout = claw_cfg.get("sandbox_timeout", 14400)
    results = []

    for merged in merged_models:
        result = run_model(claw, merged, nfs_base, sandbox_timeout)
        results.append(result)
        log.info("Result for %s: status=%s, gain=%s",
                 result["model"], result["status"],
                 f"{result['gain_pct']}%" if result.get("gain_pct") else "N/A")

    # Generate reports
    ci_run_id = f"ci-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_report = generate_markdown_report(results, args.trigger, ifx_commit, ci_run_id)
    json_summary = generate_json_summary(results, args.trigger, ifx_commit, ci_run_id)

    (out_dir / "ci_report.md").write_text(md_report)
    (out_dir / "ci_summary.json").write_text(json.dumps(json_summary, indent=2))

    # Per-model report files (for individual artifact downloads)
    for r in results:
        model_dir = out_dir / r["model"]
        model_dir.mkdir(parents=True, exist_ok=True)
        report_content = r.get("report_content") or r.get("optimization_report")
        if report_content:
            (model_dir / "optimization_report.md").write_text(report_content)

    # GitHub Actions Job Summary
    github_summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    gh_summary = generate_github_summary(results, args.trigger, ifx_commit)
    if github_summary_file:
        with open(github_summary_file, "a") as f:
            f.write(gh_summary)
        log.info("GitHub Summary written to $GITHUB_STEP_SUMMARY")
    else:
        (out_dir / "github_summary.md").write_text(gh_summary)
        log.info("GitHub Summary written to %s/github_summary.md (not in CI)", out_dir)

    log.info("Reports written to %s", out_dir)
    log.info("Summary: %d models, %d completed, %d failed",
             json_summary["stats"]["total"],
             json_summary["stats"]["completed"],
             json_summary["stats"]["failed"])

    if json_summary["stats"]["avg_gain_pct"] is not None:
        log.info("Average gain: %.1f%%", json_summary["stats"]["avg_gain_pct"])

    # Webhook notification (Slack / Teams / custom)
    webhook_env = config.get("notification", {}).get("webhook_env")
    if webhook_env:
        webhook = os.environ.get(webhook_env)
        if webhook:
            _send_webhook(webhook, json_summary)

    # Exit non-zero if no models completed (failed + timeout = all bad)
    if json_summary["stats"]["completed"] == 0:
        timeout_count = json_summary["stats"].get("timeout", 0)
        failed_count = json_summary["stats"].get("failed", 0)
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
        _row("**Optimization Gain**", f"**{'+' if gain and gain>0 else ''}{_val('gain_pct', '.1f')}%**", gain_color),
    ]
    if r.get("inferenceX_tok_per_gpu"):
        rows.append(_row("InferenceX (output tok/s/GPU)", _val("inferenceX_tok_per_gpu")))
        rows.append(_row("**vs InferenceX**",
                         f"**{'+' if vs_ifx and vs_ifx>0 else ''}{_val('vs_inferenceX_pct', '.1f')}%**", vs_color))

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
                "text": f"{status_emoji} Hyperloom CI [{model}]: {status} | Gain: {_val('gain_pct','.1f')}%"
            }, timeout=10)
    except Exception as e:
        log.warning("Webhook notification failed: %s", e)


if __name__ == "__main__":
    main()
