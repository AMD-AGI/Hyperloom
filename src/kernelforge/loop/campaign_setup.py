# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Campaign initialization: resolve or create the immutable campaign config."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from kernelforge.knowledge.experience_integration import git_checkout_branch
from kernelforge.loop.campaign_config import (
    CampaignConfig,
    CampaignConfigStore,
    create_campaign_config,
    env_backend_override,
    normalize_kernel_backend_name,
    resolve_kernel_backend_override,
    validate_pending_campaign_head,
)


def parse_list(raw: str) -> list[str]:
    """Split a comma-or-newline-separated string into a stripped list."""
    if not raw:
        return []
    return [p.strip() for p in re.split(r"[,\n]", raw) if p.strip()]


@dataclass
class CampaignResolution:
    """The resolved campaign configuration and its save-deferred flag."""

    campaign: CampaignConfig
    program_text: str | None
    save_deferred: bool


def resolve_campaign(
    workspace_dir: str,
    *,
    resume: bool,
    prepare_task: bool,
    kernel: str,
    driver: str,
    source_files: str = "",
    program_md_file: str | None = None,
    target_functions: str = "",
    operator_name: str = "",
    producer: str = "",
    kernel_backend: str = "",
    git_branch: str = "",
    gpu_target: str = "",
    gpu_type: str | None = None,
    task_type: str = "",
    framework: str = "",
    snr_threshold: float,
    nproc_per_node: int = 1,
    bench_repeat: int = 1,
    commit_new_paths: list[str] | None = None,
) -> CampaignResolution:
    """Resolve or create the immutable campaign configuration.

    Checks out ``git_branch`` for a fresh campaign before the config snapshots
    HEAD. Raises OSError or ValueError; the CLI converts those to ClickException.
    """
    workspace = Path(workspace_dir).resolve()
    campaign_store = CampaignConfigStore(str(workspace))
    campaign_root = campaign_store.root
    state_path = campaign_root / "run_state.json"

    campaign_inputs_supplied = any(
        value not in (None, "") for value in (kernel, driver, source_files, program_md_file, operator_name)
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

    resolved_kernel_backend = (kernel_backend or "").strip()
    if not resolved_kernel_backend:
        resolved_kernel_backend = env_backend_override()
    if resolved_kernel_backend:
        # ``normalize_kernel_backend_name`` already returns the bare backend key, so
        # this is the name the operator asked for, spelled canonically.
        normalized = normalize_kernel_backend_name(resolved_kernel_backend)
        resolved_kernel_backend = resolve_kernel_backend_override(resolved_kernel_backend)
        if resolved_kernel_backend != normalized:
            print(
                f"Warning: Unknown kernel backend '{normalized}'; falling back to '{resolved_kernel_backend}'.",
                file=sys.stderr,
            )

    if not kernel or not driver:
        raise ValueError("fresh campaign requires --kernel and --driver")

    # Put a fresh campaign on its development branch BEFORE the immutable
    # config snapshots the branch/base_commit.
    if git_branch:
        checkout_message = git_checkout_branch(str(workspace), git_branch)
        if checkout_message:
            print(f"  [git] {checkout_message}")

    existing_campaign: CampaignConfig | None = campaign_store.load() if pending_retry else None
    if existing_campaign is not None:
        validate_pending_campaign_head(str(workspace), existing_campaign.base_commit)

    provisional_campaign = create_campaign_config(
        # Measurement semantics travel with the campaign: a resumed session must
        # not re-derive them from CLI defaults.
        nproc_per_node=nproc_per_node,
        bench_repeat=bench_repeat,
        # What a KEEP may ship beyond the tracked diff is part of the campaign,
        # not of one session's invocation.
        commit_new_paths=list(commit_new_paths or []),
        workspace_dir=str(workspace),
        kernel=kernel,
        driver=driver,
        source_files=parse_list(source_files),
        program_md_file=program_md_file,
        base_commit=(existing_campaign.base_commit if existing_campaign is not None else None),
        target_functions=(parse_list(target_functions) or None),
        snr_threshold=snr_threshold,
        gpu_target=gpu_target,
        gpu_type=(existing_campaign.gpu_type if existing_campaign is not None else gpu_type),
        git_branch=git_branch,
        kernel_backend=resolved_kernel_backend,
        task_type=task_type,
        framework=framework,
        operator_name=operator_name,
        producer=(existing_campaign.producer if existing_campaign is not None else producer),
    )

    if program_md_file:
        program_text = Path(program_md_file).expanduser().read_text(errors="replace")

    if existing_campaign is not None:
        if provisional_campaign != existing_campaign:
            raise ValueError("pending campaign configuration does not match retry inputs")
        campaign = existing_campaign
    else:
        campaign = provisional_campaign
        # Defer the immutable save until AFTER task preparation when prep will
        # run: prep may repair the driver (changing its digest) and commit
        # scaffolding (advancing HEAD).
        if prepare_task and not resume:
            save_deferred = True
        else:
            campaign_store.save(campaign, program_md=program_text)

    return CampaignResolution(
        campaign=campaign,
        program_text=program_text,
        save_deferred=save_deferred,
    )
