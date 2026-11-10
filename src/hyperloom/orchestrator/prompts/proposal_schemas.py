# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``proposal_set`` item schemas a specialist prompt shows in Section 8: an example item and its field contract."""

from __future__ import annotations

from typing import Any

from hyperloom.inference_optimizer import framework_registry

EXPLORE_VARIANT_EXAMPLE: dict[str, Any] = {
    "name": "<unique-in-round>",
    "extra_args": "--example-flag value",
    "extra_envs": {"EXAMPLE_ENV": "1"},
    "remove_args": ["--harmful-base-flag"],
    "unset_envs": ["HARMFUL_BASE_ENV"],
    "args_mode": "append",
    "reason": "why this might help the gap",
    "atomic": False,
    "kb_evidence": [],
    "pr_evidence": [],
    "source_evidence": [],
    "experience_citations": [],
}
EXPLORE_VARIANT_CONTRACT: tuple[str, ...] = (
    (
        "- ``proposal_set`` items reuse the explore variant schema: "
        "``extra_args`` / ``extra_envs`` add or override knobs, "
        "``remove_args`` removes inherited server flags before appending, "
        "``unset_envs`` removes inherited env vars before applying "
        "``extra_envs``, and ``args_mode='replace'`` runs without "
        "inherited server args. Use removal fields when a user/base "
        "knob may be harmful; do not simulate deletion by adding an "
        "unrelated flag."
    ),
    (
        "- ``atomic`` (bool, default false): set ``true`` when this "
        "proposal's ``extra_args`` / ``extra_envs`` are a **coupled set "
        "that only works together** and MUST be benched as one variant "
        "(e.g. enabling MTP/speculative decoding REQUIRES a paired "
        "``--gpu-memory-utilization`` reduction so the draft model has "
        "headroom — split them and each half OOMs or shows no gain). "
        "Orchestration is instructed to dispatch an ``atomic`` proposal "
        "verbatim, without splitting, dropping, or re-deriving its flags. "
        "Put every co-required flag in THIS one entry; do not scatter a "
        "coupling across several proposals."
    ),
)
# FrameworkPhase._candidates_from_discovery_proposals drops a candidate whose framework is not one of these exactly.
_REGISTERED_FRAMEWORKS = ", ".join(framework_registry.names())

# The keys FrameworkPhase._candidates_from_discovery_proposals reads.
UPSTREAM_CANDIDATE_EXAMPLE: dict[str, Any] = {
    "pr_url": "https://github.com/<owner>/<repo>/pull/<number>",
    "head_sha": "<PR head commit>",
    "repo": "<owner>/<repo>",
    "pr_number": 0,
    "title": "<PR title>",
    "diff_url": "<unified diff URL>",
    "framework": f"<one of: {_REGISTERED_FRAMEWORKS}>",
    "changed_files": [],
    "verdict": "worth_a_bench",
    "route": "author_via_specialist",
    "reason": "why this candidate addresses the gap",
}
UPSTREAM_CANDIDATE_CONTRACT: tuple[str, ...] = (
    (
        "- ``proposal_set`` items are upstream candidates, not explore "
        "variants: an item with neither ``pr_url`` nor ``head_sha`` is "
        "dropped. ``verdict`` is ``already_present``, ``not_applicable`` or "
        "``worth_a_bench``; ``route`` is ``direct_framework`` or "
        "``author_via_specialist``."
    ),
    (
        f"- ``framework`` is the registered name exactly ({_REGISTERED_FRAMEWORKS}); "
        "a candidate whose ``framework`` carries anything else -- a backend, "
        "platform or version note -- is dropped."
    ),
)
