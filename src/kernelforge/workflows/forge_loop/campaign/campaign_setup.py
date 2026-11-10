# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Campaign initialization: resolve or create the immutable campaign config."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

from kernelforge.utils.git import ensure_commit_identity, git_checkout_branch
from kernelforge.workflows.forge_loop.campaign.campaign_config import (
    CampaignConfig,
    CampaignConfigStore,
    CampaignInputs,
    create_campaign_config,
    env_backend_override,
    resolve_kernel_backend_override,
    validate_pending_campaign_head,
)


@dataclass
class CampaignResolution:
    """The resolved campaign configuration and its save-deferred flag."""

    campaign: CampaignConfig
    program_text: str | None
    save_deferred: bool


def resolve_campaign(
    workspace_dir: str,
    inputs: CampaignInputs,
    *,
    resume: bool,
    prepare_task: bool,
) -> CampaignResolution:
    """Resolve or create the immutable campaign configuration."""
    workspace = Path(workspace_dir).resolve()
    campaign_store = CampaignConfigStore(str(workspace))

    # Before the resume branch: a resumed campaign commits into this workspace too.
    written_identity = ensure_commit_identity(workspace)
    if written_identity:
        print(f"  [git] workspace had no commit identity; set {written_identity}")

    campaign_inputs_supplied = any(
        (inputs.kernel, inputs.driver, inputs.source_files, inputs.program_md_file, inputs.operator_name)
    )
    if resume and campaign_store.exists():
        if campaign_inputs_supplied:
            raise ValueError(
                "campaign already has immutable configuration; resume with "
                "--workspace, --resume, and session options only"
            )
        return CampaignResolution(campaign=campaign_store.load(), program_text=None, save_deferred=False)

    state_path = campaign_store.root / "run_state.json"
    pending_retry = not resume and campaign_store.exists() and not state_path.is_file()
    # ``normalize_kernel_backend_name`` already returns the bare backend key, so this is the name the operator
    # asked for, spelled canonically.
    requested_backend = inputs.kernel_backend.strip() or env_backend_override()
    resolved_kernel_backend = resolve_kernel_backend_override(requested_backend) if requested_backend else ""
    if resolved_kernel_backend != requested_backend:
        print(
            f"Warning: Unknown kernel backend '{requested_backend}'; falling back to '{resolved_kernel_backend}'.",
            file=sys.stderr,
        )

    if not inputs.kernel or not inputs.driver:
        raise ValueError("fresh campaign requires --kernel and --driver")

    # Put a fresh campaign on its development branch BEFORE the immutable config snapshots the branch/base_commit.
    if inputs.git_branch:
        checkout_message = git_checkout_branch(str(workspace), inputs.git_branch)
        if checkout_message:
            print(f"  [git] {checkout_message}")

    existing_campaign: CampaignConfig | None = None
    base_commit = None
    if pending_retry:
        existing_campaign = campaign_store.load()
        validate_pending_campaign_head(str(workspace), existing_campaign.base_commit)
        inputs = replace(inputs, gpu_type=existing_campaign.gpu_type, producer=existing_campaign.producer)
        base_commit = existing_campaign.base_commit

    provisional_campaign = create_campaign_config(
        str(workspace),
        replace(inputs, kernel_backend=resolved_kernel_backend),
        base_commit=base_commit,
    )

    program_text: str | None = None
    if inputs.program_md_file:
        program_text = Path(inputs.program_md_file).expanduser().read_text(errors="replace")

    if existing_campaign is not None:
        if provisional_campaign != existing_campaign:
            raise ValueError("pending campaign configuration does not match retry inputs")
        return CampaignResolution(campaign=existing_campaign, program_text=program_text, save_deferred=False)
    # Defer the immutable save until AFTER task preparation when prep will run: prep may repair the driver
    # (changing its digest) and commit scaffolding (advancing HEAD).
    save_deferred = prepare_task and not resume
    if not save_deferred:
        campaign_store.save(provisional_campaign, program_md=program_text)
    return CampaignResolution(campaign=provisional_campaign, program_text=program_text, save_deferred=save_deferred)
