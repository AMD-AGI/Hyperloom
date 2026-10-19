# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Campaign initialization: resolve or create the immutable campaign config."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

from kernelforge.loop.knowledge_integration import git_checkout_branch
from kernelforge.llm.git import ensure_commit_identity
from kernelforge.loop.campaign_config import (
    CampaignConfig,
    CampaignConfigStore,
    CampaignInputs,
    create_campaign_config,
    env_backend_override,
    normalize_kernel_backend_name,
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
    campaign_root = campaign_store.root
    state_path = campaign_root / "run_state.json"

    # Before the resume branch: a resumed campaign commits into this workspace too.
    written_identity = ensure_commit_identity(workspace)
    if written_identity:
        print(f"  [git] workspace had no commit identity; set {written_identity}")

    kernel, driver, program_md_file, git_branch = (
        inputs.kernel,
        inputs.driver,
        inputs.program_md_file,
        inputs.git_branch,
    )
    campaign_inputs_supplied = any(
        value not in (None, "")
        for value in (kernel, driver, inputs.source_files, program_md_file, inputs.operator_name)
    )
    pending_retry = not resume and campaign_store.exists() and not state_path.is_file()

    program_text: str | None = None
    save_deferred = False

    if resume and campaign_store.exists():
        if campaign_inputs_supplied:
            raise ValueError(
                "campaign already has immutable configuration; resume with "
                "--workspace, --resume, and session options only"
            )
        campaign = campaign_store.load()
        return CampaignResolution(
            campaign=campaign,
            program_text=None,
            save_deferred=False,
        )

    resolved_kernel_backend = (inputs.kernel_backend or "").strip()
    if not resolved_kernel_backend:
        resolved_kernel_backend = env_backend_override()
    if resolved_kernel_backend:
        # ``normalize_kernel_backend_name`` already returns the bare backend key, so this is the name the operator
        # asked for, spelled canonically.
        normalized = normalize_kernel_backend_name(resolved_kernel_backend)
        resolved_kernel_backend = resolve_kernel_backend_override(resolved_kernel_backend)
        if resolved_kernel_backend != normalized:
            print(
                f"Warning: Unknown kernel backend '{normalized}'; falling back to '{resolved_kernel_backend}'.",
                file=sys.stderr,
            )

    if not kernel or not driver:
        raise ValueError("fresh campaign requires --kernel and --driver")

    # Put a fresh campaign on its development branch BEFORE the immutable config snapshots the branch/base_commit.
    if git_branch:
        checkout_message = git_checkout_branch(str(workspace), git_branch)
        if checkout_message:
            print(f"  [git] {checkout_message}")

    existing_campaign: CampaignConfig | None = campaign_store.load() if pending_retry else None
    if existing_campaign is not None:
        validate_pending_campaign_head(str(workspace), existing_campaign.base_commit)

    provisional_campaign = create_campaign_config(
        str(workspace),
        replace(
            inputs,
            kernel_backend=resolved_kernel_backend,
            gpu_type=(existing_campaign.gpu_type if existing_campaign is not None else inputs.gpu_type),
            producer=(existing_campaign.producer if existing_campaign is not None else inputs.producer),
        ),
        base_commit=(existing_campaign.base_commit if existing_campaign is not None else None),
    )

    if program_md_file:
        program_text = Path(program_md_file).expanduser().read_text(errors="replace")

    if existing_campaign is not None:
        if provisional_campaign != existing_campaign:
            raise ValueError("pending campaign configuration does not match retry inputs")
        campaign = existing_campaign
    else:
        campaign = provisional_campaign
        # Defer the immutable save until AFTER task preparation when prep will run: prep may repair the driver
        # (changing its digest) and commit scaffolding (advancing HEAD).
        if prepare_task and not resume:
            save_deferred = True
        else:
            campaign_store.save(campaign, program_md=program_text)

    return CampaignResolution(
        campaign=campaign,
        program_text=program_text,
        save_deferred=save_deferred,
    )
