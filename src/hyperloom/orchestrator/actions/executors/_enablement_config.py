# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The established enablement configuration a framework switch is read against, and the provenance stamped on it."""

from __future__ import annotations

import logging
import shlex
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import filter_untrusted_env_mapping, is_allowed_variant_env_key
from hyperloom.common.failure_signature import RESOURCE_CONSTRAINT, FailureSignature
from hyperloom.inference_optimizer.grid_server_args import (
    dedupe_extra_server_args,
    merge_server_args,
    tokenize_server_args_preserving_json,
)

from . import _framework_switch_manifest as _switch_manifest
from ._patch_sources import _acquiring_round, _done_field, _read_patch_texts

log = logging.getLogger(__name__)


# Two-sided band, in percent of the pre-patch base, that a switch-off parity leg
# must land inside. The rewrite workloads this gates measure with a run-to-run
# spread well under 1%, so a band this wide clears noise by a comfortable margin
# while still catching a patch that is not actually inert when disabled.
DEFAULT_SWITCH_OFF_PARITY_BAND_PCT = 2.0


def _enablement_gap(params: dict[str, Any]) -> FailureSignature | None:
    """The round's capability gap, when it calls for code the stack lacks.

    Read from the verdict the round was dispatched on; None for a round that
    acquires nothing and for a gap no code closes.
    """
    if not _acquiring_round(params):
        return None
    raw = params.get("enablement_failure_signature")
    if not isinstance(raw, dict) or not raw:
        return None
    signature = FailureSignature.from_dict(raw)
    return signature if signature.kind != RESOURCE_CONSTRAINT else None


def _established_enablement_config(params: dict[str, Any], shared_state: Any) -> tuple[str, dict[str, str]]:
    """What earlier enablement rounds established, for a round that restated nothing.

    ``enablement.lane._rearm_on_advanced`` accumulates every advance into
    ``state.enablement.accepted_config`` so a later kept round replays them --
    but the accumulation only ever reached the emitted recipe. A launch that
    does not read it back walks into a wall an earlier round already cleared:
    observed live, a build launch probe booted with an empty ``EXTRA_VLLM_ARGS``
    and died on ``AssertionError: DeepseekV4 only supports fp8 kv-cache format
    for now, got auto`` while ``accepted_config`` held ``--kv-cache-dtype fp8``.
    Every dispatcher that opens an enablement integrate_patch crosses this
    executor, so the inheritance lives here rather than in each emitter.

    Optimization rounds inherit nothing -- the rule is the enablement lane's.
    Returns ``("", {})`` when the round is not enablement or when nothing has
    been established yet.
    """
    if not bool(params.get("enablement")):
        return "", {}
    established = shared_state.enablement.accepted_config
    if not established:
        return "", {}
    args = str(established.get("extra_server_args") or "").strip()
    raw_envs = established.get("extra_envs")
    envs = {str(k): str(v) for k, v in raw_envs.items()} if isinstance(raw_envs, dict) else {}
    envs, dropped = filter_untrusted_env_mapping(envs, allow_predicate=is_allowed_variant_env_key)
    if dropped:
        log.warning(
            "integrate_patch: dropping unsafe inherited enablement env key(s): %s",
            ", ".join(sorted(dropped)),
        )
    return args, envs


def _merge_established_server_args(inherited_args: str, round_args: str) -> str:
    """Inherited first, this round last, deduped -- the shape the bridge uses."""
    if not inherited_args:
        return round_args
    if not round_args:
        return inherited_args
    merged = merge_server_args(inherited_args, round_args)
    if tokenize_server_args_preserving_json(merged) is not None:
        return dedupe_extra_server_args(merged)
    # The combined string carries a quoted value with embedded whitespace, which
    # the deduper cannot parse; it would hand back the concatenation with two
    # copies of every inherited flag, and a duplicate is what the server
    # hard-errors on. Provenance cannot decide this either: only the autosubmit
    # bridge pre-merges the inheritance, while this executor also serves rounds
    # opened by the build probe, the framework-config bridge, an authored
    # proposal and a re-queued row, any of which may carry args of its own. So
    # add back only the inherited flags this round does not already name,
    # decided on the inherited side, which comes from ``accepted_config`` and is
    # already deduped.
    try:
        round_tokens = shlex.split(round_args)
    except ValueError:
        # Unbalanced quoting: fall back to whitespace splitting, which can only
        # over-report option names, never miss one, so the comparison below
        # stays conservative about adding a flag back.
        round_tokens = round_args.split()
    # Exact option names, never a substring test: ``--foo`` is not satisfied by
    # ``--foo-bar``, and a name appearing inside a quoted value is not an option
    # at all once shlex has stripped the quotes.
    round_names = {token.split("=", 1)[0] for token in round_tokens if token.startswith("-")}
    try:
        # Quote-aware on this side too: ``accepted_config`` may itself hold a
        # value carrying whitespace, and splitting it on spaces would keep only
        # the first word of it and emit a malformed option.
        inherited_tokens = shlex.split(inherited_args)
    except ValueError:
        inherited_tokens = inherited_args.split()
    missing: list[str] = []
    index = 0
    while index < len(inherited_tokens):
        token = inherited_tokens[index]
        if not token.startswith("-"):
            index += 1
            continue
        name = token.split("=", 1)[0]
        takes_value = (
            "=" not in token and index + 1 < len(inherited_tokens) and not inherited_tokens[index + 1].startswith("-")
        )
        if name not in round_names:
            # shlex stripped the quoting when it parsed; restore a spelling that
            # survives the next parse rather than re-emitting a bare value.
            missing.append(shlex.quote(token))
            if takes_value:
                missing.append(shlex.quote(inherited_tokens[index + 1]))
        index += 2 if takes_value else 1
    if not missing:
        return round_args
    return " ".join([*missing, round_args]).strip()


def _parse_framework_switches(
    *,
    params: dict[str, Any],
    done_payload: dict[str, Any] | None,
    patch_paths: list[Path],
) -> list[dict[str, Any]]:
    """Read the framework-rewrite switch manifest this round benches with on.

    Looks in the task params first (an explicit dispatch), then in the
    specialist's done payload (the normal authoring path, possibly nested under
    ``payload``). Every rewrite in such a patch sits behind a switch that
    defaults OFF, so the applied patch is inert and benching it as-is would
    measure the baseline.

    Args:
        params: The integrate_patch task params.
        done_payload: The originating specialist's done payload, if any.
        patch_paths: The patches this round delivers.

    Returns:
        The switches to turn on; empty when no manifest was delivered or the
        manifest gates code no patch carries.
    """
    raw = params.get(_switch_manifest.MANIFEST_KEY) or _done_field(done_payload, _switch_manifest.MANIFEST_KEY)
    if not raw:
        return []
    # Env the benchmark already defines is reserved: a "switch" colliding with it
    # would be toggled by unrelated configuration rather than by the lever.
    reserved: set[str] = set()
    for source in (params.get("base_extra_envs"), params.get("extra_envs")):
        if isinstance(source, dict):
            reserved.update(str(k).strip().upper() for k in source)
    manifest, problems = _switch_manifest.parse_manifest(
        raw, reserved_env=reserved, patch_text="\n".join(_read_patch_texts(patch_paths))
    )
    if manifest and not patch_paths:
        # A manifest without a patch describes switches that gate code which was
        # never delivered. Setting them would be a no-op, and registering them as
        # levers would leave the ledger pointing at absent code.
        problems.append(
            f"discarded {len(manifest)} switch(es): the deliverable carries no patch, "
            f"so there is no rewrite for them to gate"
        )
        manifest = []
    if manifest:
        log.info(
            "integrate_patch: benching with %d framework rewrite switch(es) on\n%s",
            len(manifest),
            _switch_manifest.summarize(manifest, problems),
        )
    elif problems:
        log.warning(
            "integrate_patch: framework switch manifest unusable\n%s",
            _switch_manifest.summarize(manifest, problems),
        )
    return manifest


_FRAMEWORK_KB_PROVENANCE_PREFIX = "specialist:serving:framework"


def _stamp_framework_kb_provenance(
    done_payload: dict[str, Any] | None,
    *,
    params: dict[str, Any],
    shared_state: Any,
) -> None:
    """Ensure a FRAMEWORK-dispatched deliverable carries KB-writeback provenance.

    Stamps the ``specialist:serving:framework...`` provenance prefix (that
    :meth:`IntegratePatchExecutor._find_framework_proposal` requires) from the
    dispatch context, so same-framework deliverables reach ``lessons.jsonl``.

    Mutates ``done_payload["proposal_set"][0]`` in place; no-ops when this
    action was not dispatched from FRAMEWORK authoring, or when a proposal
    already carries a matching provenance.

    Args:
        done_payload: The specialist's parsed ``specialist_done.json`` (may
            be ``None``/malformed; no-ops in that case).
        params: The ``integrate_patch`` action's dispatch params (carries
            ``framework_agent_authoring`` / ``framework_agent_candidate_id``
            when this run came from FRAMEWORK_AGENT).
        shared_state: The run's ``SharedState`` (best-effort ``framework``
            read for the provenance suffix).
    """
    if not isinstance(done_payload, dict):
        return
    if not params.get("framework_agent_authoring"):
        return
    pr_url = str(params.get("framework_agent_candidate_id") or "").strip()
    if not pr_url:
        return
    proposals = done_payload.get("proposal_set")
    if not isinstance(proposals, list) or not proposals or not isinstance(proposals[0], dict):
        # No proposal_set entry to stamp; synthesize a minimal anchor.
        done_payload["proposal_set"] = [{}]
        proposals = done_payload["proposal_set"]
    target = proposals[0]
    existing = str(target.get("provenance") or "")
    if existing.startswith(_FRAMEWORK_KB_PROVENANCE_PREFIX):
        return  # cross-framework (or already-stamped) path already complies
    framework = str(getattr(shared_state, "framework", "") or "").strip().lower()
    target["provenance"] = (
        f"{_FRAMEWORK_KB_PROVENANCE_PREFIX}:{framework}" if framework else _FRAMEWORK_KB_PROVENANCE_PREFIX
    )
    target.setdefault("fa_pr_url", pr_url)
    target.setdefault("framework", framework)
