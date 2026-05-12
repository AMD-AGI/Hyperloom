#!/usr/bin/env python3
"""Temporary one-shot submitter for DeepSeek-R1 multi-node MoRI 1P1D.

Bypasses ci-config.yaml + optimize_submit.py + InferenceX lookup —
sends a hand-crafted multi-node prompt directly to a fresh Claw session.

This reproduces the path yunkai validated manually through the PrimusClaw
GUI, so we can re-run programmatically without recreating the prompt.

Migration: to switch from DSR1 to another multi-node model (GLM-5, Qwen3.5,
etc.), edit the MODEL_CFG dict below. The rest of the script is generic.

Usage (from repo root or ci/):
    export CLAW_API_KEY=ak-...                              # = SAFE_API_KEY
    export SAFE_BASE_URL=https://core42.example-internal-host.invalid
    export SANDBOX_WORKSPACE=core42-sandbox                 # or workspace UUID
    python3 ci/run_dsr1_multinode.py --dry-run              # preview prompt
    python3 ci/run_dsr1_multinode.py                        # submit, print IDs
    python3 ci/run_dsr1_multinode.py --monitor              # also stream SSE
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CI_DIR))

log = logging.getLogger("dsr1-multinode")


# ── Model config (swap this block to migrate to another model) ──
MODEL_CFG: dict = {
    "model_hf":            "deepseek-ai/DeepSeek-R1-0528",
    "model_path":          "/wekafs/models/DeepSeek-R1-0528",
    "framework":           "sglang",
    "precision":           "FP8",
    "isl":                 1024,
    "osl":                 256,
    "conc":                16,
    # tp is used only by the prompt template's tok_per_gpu formula:
    #   tok_per_gpu_X = X / {tp}
    # MoRI 1P1D spans 2 nodes × 8 GPU = 16 GPUs total → tp=16 here.
    "tp":                  16,
    "ep":                  1,
    "gpu_type":            "MI300X",
    "target_gpu":          "mi300x",
    "inferencex_path":     "/wekafs/InferenceX",
    "rayjob_image":        "harbor.core42.example-internal-host.invalid/custom/sync/sglang:202604290707",
    "kernel_opt_backends": "claude",
    "kernel_opt_image":    "harbor.core42.example-internal-host.invalid/proxy/lmsysorg/sglang-rocm:v0.5.10rc0-rocm720-mi35x-20260413",
    "min_kernels":         5,
    "result_dir":          "/workspace/hyperloom/",
}


def render_prompt(cfg: dict, safe_api_key: str, safe_base_url: str,
                  nfs_root: str) -> str:
    """Render prompt_template_remote.md with our hardcoded multi-node config.

    Skips InferenceX API lookup and benchmark_script clone — multi-node
    skill addendum on /wekafs is the source of truth for server flags.
    """
    tpl = (CI_DIR / "prompt_template_remote.md").read_text()

    inferenceX_data = (
        f"# No InferenceX baseline injected (one-shot multi-node submit). "
        f"Target: {cfg['target_gpu']} ISL={cfg['isl']}/OSL={cfg['osl']}/"
        f"{cfg['precision']}. Follow the multi-node skill addendum."
    )
    benchmark_script_section = (
        "No InferenceX benchmark script attached. The agent must construct "
        "the SGLang MoRI 1P1D launch commands per the multi-node skill "
        "addendum (prefill on head, decode on worker, mori transfer "
        "backend, mini_lb router)."
    )

    return tpl.format(
        model_hf=cfg["model_hf"],
        model_path=cfg["model_path"],
        framework=cfg["framework"],
        precision=cfg["precision"],
        isl=cfg["isl"],
        osl=cfg["osl"],
        conc=cfg["conc"],
        tp=cfg["tp"],
        ep=cfg["ep"],
        gpu_type=cfg["gpu_type"],
        target_gpu=cfg["target_gpu"],
        inferencex_path=cfg["inferencex_path"],
        rayjob_image=cfg["rayjob_image"],
        kernel_opt_backends=cfg["kernel_opt_backends"],
        kernel_opt_image=cfg["kernel_opt_image"],
        min_kernels=cfg["min_kernels"],
        result_dir=cfg["result_dir"],
        inferenceX_data=inferenceX_data,
        benchmark_script_section=benchmark_script_section,
        safe_api_key=safe_api_key,
        safe_base_url=safe_base_url,
        nfs_root=nfs_root,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="Render prompt to stdout, do not submit.")
    parser.add_argument("--monitor", action="store_true",
                        help="After submit, stream SSE until session ends.")
    parser.add_argument("--session-name", default="",
                        help="Override Claw session name "
                             "(default: dsr1-multinode-<UTC timestamp>).")
    parser.add_argument("--workspace", default="",
                        help="Sandbox workspace (default: $SANDBOX_WORKSPACE "
                             "or 'core42-sandbox').")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable DEBUG-level logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    safe_api_key = (os.environ.get("CLAW_API_KEY")
                    or os.environ.get("SAFE_API_KEY") or "")
    safe_base_url = (os.environ.get("SAFE_BASE_URL")
                     or "https://core42.example-internal-host.invalid").rstrip("/")
    sandbox_workspace = (args.workspace
                         or os.environ.get("SANDBOX_WORKSPACE")
                         or "core42-sandbox")
    nfs_root = os.environ.get("NFS_ROOT", "/wekafs")

    if not safe_api_key:
        log.error("missing CLAW_API_KEY / SAFE_API_KEY env var")
        return 2

    prompt = render_prompt(MODEL_CFG, safe_api_key, safe_base_url, nfs_root)
    log.info("Rendered prompt: %d chars", len(prompt))

    if args.dry_run:
        print(prompt)
        return 0

    # Defer ClawClient import — it pulls in `requests` + `sseclient` which
    # only matter for actually submitting / monitoring. --dry-run works
    # in a barebones venv this way.
    from claw_client import ClawClient  # noqa: PLC0415

    claw_endpoint = f"{safe_base_url}/claw-api/v1"
    claw = ClawClient(
        endpoint=claw_endpoint,
        api_key=safe_api_key,
        timeout=28800,
        agent_id="agent_default",
        sandbox_workspace=sandbox_workspace,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    model_short = MODEL_CFG["model_hf"].split("/")[-1].lower()
    session_name = args.session_name or f"dsr1-multinode-{model_short}-{ts}"

    log.info("Creating Claw session: %s", session_name)
    session = claw.create_session(session_name)
    session_id = session["session_id"]

    log.info("Sending prompt to session %s (workspace=%s, remote=True)",
             session_id, sandbox_workspace)
    claw.send_message(
        session_id=session_id,
        content=prompt,
        task_mode="agent",
        plugin_id=4,
        tools=[],  # remote mode: agent uses RayJob + Ray Dashboard REST
        resource=None,
    )

    chat_url = f"{safe_base_url}/playground/chat/{session_id}"
    log.info("=" * 64)
    log.info("Submitted. Session ID: %s", session_id)
    log.info("Claw chat URL:        %s", chat_url)
    log.info("=" * 64)

    if args.monitor:
        log.info("Monitoring session via SSE + polling until done ...")
        status = claw.monitor_session(session_id, timeout=28800)
        log.info("Final status: %s", status)
        return 0 if status == "completed" else 1

    log.info("Exit. Re-run with --monitor to stream events, or visit chat URL.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
