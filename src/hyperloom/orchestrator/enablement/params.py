# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Enablement authoring-specialist request construction."""

from __future__ import annotations

import glob
import http.client
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from hyperloom.agents.framework.sources.pr_monitor import PRMonitorError
from hyperloom.common.deadline import Deadline
from hyperloom.common.model_paths import resolve_local_model_dir
from hyperloom.orchestrator.lever import LEVER_ENABLEMENT

from ..bringup import recorded_verdict, session_root
from ..collaborator import CoordinatorCollaborator

import logging as _logging

log = _logging.getLogger(__name__)

#: How long a round may spend discovering bridging candidates before the tick
#: stops waiting; discovery then degrades to repos-only.
ENABLEMENT_PARAMS_BUDGET_SEC: float = 45.0


def _enablement_carrier_params(state: Any) -> dict[str, Any]:
    """eval-origin trigger context threaded to specialist/integrate/build tasks.

    Empty for boot-origin enablement so the boot path is unchanged.
    """
    origin = str(state.enablement.origin or "")
    if not origin:
        return {}
    out: dict[str, Any] = {
        "enablement_origin": origin,
        "enablement_accuracy_floor": float(state.enablement.accuracy_floor or 0.0),
    }
    cfg = str(state.enablement.probe_config_path or "")
    if cfg:
        out["enablement_probe_config_path"] = cfg
    return out


def _prior_progress_note(enablement: Any) -> str:
    """What earlier rounds already banked, so this round composes instead of redoing it."""
    prior_patches = [str(p) for p in (enablement.kept_patches or [])]
    prior_artifacts = list(enablement.kept_artifacts or [])
    base_setup = [str(c) for c in (enablement.setup_commands or [])]
    acc_envs = enablement.accepted_extra_envs
    acc_args = enablement.accepted_extra_server_args
    bits = []
    if prior_patches:
        bits.append(f"{len(prior_patches)} prior patch(es) already in the tree: {prior_patches}")
    if prior_artifacts:
        art_targets = [a["target"] for a in prior_artifacts[:4]]
        bits.append(f"{len(prior_artifacts)} prior artifact(s) already in the tree: {art_targets}")
    if base_setup:
        bits.append(f"{len(base_setup)} prior setup command(s): {base_setup}")
    if acc_envs or acc_args:
        cfg_bits = ([f"envs={acc_envs}"] if acc_envs else []) + ([f"args={acc_args!r}"] if acc_args else [])
        bits.append("accumulated config (re-applied on every launch): " + "; ".join(cfg_bits))
    if not bits:
        return ""
    return (
        "PRIOR ENABLEMENT PROGRESS: the following already cleared earlier "
        "boot crashes. Patches and artifacts are permanently in the tree; "
        "setup commands and config are re-run on every launch. Do NOT redo "
        "them; fix only the CURRENT (deeper) failure, composing on top. " + "; ".join(bits)
    )


def _apply_feedback_note(enablement: Any) -> str:
    """The conflicts that kept the prior round's diff out of the framework tree."""
    feedback = [fb for fb in (enablement.last_apply_feedback or []) if isinstance(fb, dict)]
    if not feedback:
        return ""
    from ..actions.executors._apply_feedback import ApplyFeedback

    blocks = [
        "APPLY FAILURE FEEDBACK: the prior round's patch never reached the "
        "framework tree. Re-ground the diff against the target file as it "
        "reads now \u2014 the errors below name the conflict."
    ]
    blocks.extend(ApplyFeedback.from_dict(fb).format_for_mandate() for fb in feedback[:5])
    return "\n\n".join(blocks)


def _build_failure_note(enablement: Any) -> str:
    """The last targeted build's verdict, so the round picks a different ref or scope."""
    failure = enablement.last_build_failure or {}
    fc = str(failure.get("failure_class") or "")
    fs = str(failure.get("failure_summary") or "")
    if not (fc or fs):
        return ""
    return (
        f"PREVIOUS TARGETED-BUILD ATTEMPT: failure_class={fc!r}"
        + (f"; {fs}" if fs else "")
        + "\nIf the build ran out of time (failure_class='timeout' or "
        "'preflight_budget'), request more build_budget_sec or a smaller "
        "component scope.  For compile/symbol defects, choose a different "
        "ref or narrow the build target."
    )


def _mandate_notes(enablement: Any, *, attempt: int) -> str:
    """The per-dispatch dynamic context the static playbook section cannot provide.

    Ordered most-corrective first, so a stalled round that also banked progress
    does not read as "all good so far, go deeper".
    """
    grounding_drops = [str(d) for d in (enablement.last_grounding_drop_reason or [])]
    setup_refusals = [str(c) for c in (enablement.last_setup_refusals or [])]
    inert_env_reads = [str(n) for n in (enablement.last_inert_round_env_reads or [])]
    blocks = [_build_failure_note(enablement)]
    if enablement.patches_span_multiple_roots:
        blocks.append(
            "CROSS-TREE PATCH SET: the prior round's patches targeted more "
            "than one framework source tree, which integrate_patch cannot "
            "apply as one set. Submit patches for a single tree per round."
        )
    if setup_refusals:
        blocks.append(
            "SETUP COMMAND REFUSED: setup_commands are replayed before every "
            "boot and are limited to package/tool installation, so the prior "
            "round was failed WITHOUT running any of them. Refused: "
            + "; ".join(setup_refusals[:4])
            + ". Re-author these as installer invocations, or carry the work "
            "in a patch instead."
        )
    if grounding_drops:
        blocks.append(
            "PATCH GROUNDING FAILURE: the patches the prior round submitted "
            "were dropped because their target file(s) do not exist in ANY "
            "allowlisted framework source tree. Targets: "
            + "; ".join(grounding_drops[:4])
            + ". Verify the target path with Glob/Read inside the worktree "
            "before authoring the diff."
        )
    if inert_env_reads:
        blocks.append(
            "POSSIBLE CAUSE OF THE UNCHANGED BOOT: the prior round reverted "
            "without serving, and its boot stopped at the same wall with the "
            "same failure as the boot before it, as though its change were not "
            "there; its diff reads environment variable(s) "
            + ", ".join(inert_env_reads[:4])
            + " that nothing set for that boot. Check whether the patched "
            "path is switched off by default. This is one lead among others, "
            "not a finding: a read may be an opt-out, a value the code needs "
            "or unrelated to the change."
        )
    blocks.append(_apply_feedback_note(enablement))
    if attempt:
        blocks.append(
            f"RETRY ({attempt} prior round(s) cleared nothing): the last enablement "
            f"patch for this failure never made the combo runnable \u2014 it failed to apply "
            f"or did not advance the boot. Try a DIFFERENT bridging approach / candidate than before."
        )
    blocks.append(_prior_progress_note(enablement))
    return "\n\n".join(block for block in blocks if block)


def _round_carriers(state: Any, signature: Any) -> dict[str, Any]:
    """What an integrate_patch needs to be graded as this enablement round."""
    return {
        "enablement": True,
        # The verdict this round was dispatched on. The prompt builder renders
        # it; nothing downstream re-classifies a log to recover it.
        "enablement_failure_signature": signature.to_dict(),
        # The before half of integrate_patch's gate.
        "enablement_before_observation_path": state.enablement.launch_observation_path,
        # Only prior rounds' *actually-applied* setup commands (recorded by the
        # specialist and replayed by integrate_patch) stack as a base. No install
        # command is ever auto-seeded here: an unpinned upgrade of the shared
        # serving venv is unsafe (CUDA-wheel clobber of ROCm vLLM/torch,
        # transformers-major skew) and environment/build acquisition is owned by
        # the isolated targeted-build path + the specialist's own setup_commands.
        "enablement_setup_commands": [str(c) for c in (state.enablement.setup_commands or [])],
        # eval-origin trigger context (empty for boot-origin enablement).
        **_enablement_carrier_params(state),
    }


def _layers_by_family(names: Iterable[str]) -> dict[str, set[int]]:
    """Group parameter names by family (layer indices folded to ``{N}``) with the layer indices each one carries."""
    families: dict[str, set[int]] = {}
    for name in names:
        layers = families.setdefault(re.sub(r"\.\d+\.", ".{N}.", name), set())
        index = re.search(r"\.(\d+)\.", name)
        if index:
            layers.add(int(index.group(1)))
    return families


def _checkpoint_weight_map(model_path: str) -> Any:
    """Read the ``weight_map`` of the checkpoint's sharded index, or ``None`` when there is none.

    ``model_path`` may be an HF repo id; resolve to the local
    weights dir so the sharded-index read works for repo-id launches.
    """
    model_dir = resolve_local_model_dir(model_path)
    idx_files = glob.glob(f"{model_dir}/*.index.json") if model_dir else []
    data = json.loads(Path(idx_files[0]).read_text(errors="replace")) if idx_files else {}
    return data.get("weight_map") if isinstance(data, dict) else None


class EnablementParams(CoordinatorCollaborator):
    """Builds the enablement authoring specialist's parameters."""

    def build_enablement_specialist_params(self, launch_log: str, *, attempt: int = 0) -> dict[str, Any] | None:
        """Build enablement-specialist params from a captured launch failure.

        Classifies the failure (advisory ``kind`` only — see Q1 hardening),
        plans bridging discovery, and runs a **best-effort** candidate-PR
        enumeration (network; fully exception-guarded, degrades to repos-only).
        The mandate itself is rendered downstream by
        ``_section_enablement_playbook`` from the structured ``enablement_*``
        params emitted here, so the prompt text is built once, at the point of
        use. Returns ``None`` **only** when the launch log is blank (nothing to
        act on); a non-blank log always yields params, even when it classifies
        as ``UNKNOWN`` — the LLM specialist repairs from the raw log so a
        brand-new gap type never wedges the run.

        After a round that bought no ground (``attempt > 0``) the ranked
        candidate list is *rotated* so a different bridging PR leads, and the
        notes flag the revert — steering the sub-agent toward a different bridge.

        Args:
            launch_log: Captured launch / traceback text.
            attempt: Consecutive rounds that cleared nothing; drives candidate
                rotation and a retry hint in the mandate. Zero after an advance,
                so a progressing bring-up is never told to change approach.

        Returns:
            dict | None: Specialist task params (tagged ``enablement``) or ``None``.
        """
        text = (launch_log or "").strip()
        if not text:
            return None
        from hyperloom.common.failure_signature import EnablementRequest
        from .mandate import build_search_plan
        from hyperloom.inference_optimizer import framework_registry

        state = self.shared_state
        framework = (getattr(state, "framework", "") or "").strip().lower()
        model = (getattr(state, "model_name", "") or "").strip()
        repo_url = framework_registry.repo_url(framework)

        # Dispatch a specialist for ANY non-blank launch log, even one that
        # classifies as ``UNKNOWN``: ``kind`` is advisory (routes bridge-repo
        # hints and labels the mandate), not a gate.
        verdict, loaded = recorded_verdict(
            state.enablement.launch_observation_path,
            wrapper_text=text,
            session_dir=session_root(self),
        )
        signature = verdict.signature
        if loaded.degraded:
            log.info(
                "ENABLEMENT: routing on a re-derived signature (%s); no boot observation was recorded",
                loaded.degraded,
            )
        req = EnablementRequest(
            framework=framework,
            model=model or "(target model)",
            repo_url=repo_url,
            gpu_type=(getattr(state, "gpu_type", "") or "").strip().lower(),
        )
        plan = build_search_plan(signature, framework_repo_url=repo_url, model=model)
        candidate_refs = self._discover_enablement_candidate_refs(
            req,
            plan,
            deadline=Deadline.after(ENABLEMENT_PARAMS_BUDGET_SEC),
        )
        # Lead with a different candidate each stalled round (deterministic left-rotation).
        if candidate_refs and attempt:
            n = len(candidate_refs)
            k = attempt % n
            candidate_refs = candidate_refs[k:] + candidate_refs[:k]
        # Nothing here writes session state: this runs on a worker thread the
        # tick may already have stopped waiting on. The refs travel back in the
        # returned params for the lane to record.
        source_context = self._read_enablement_source_context(signature)
        # For a weight-init failure, fold the checkpoint's ground-truth per-layer
        # weight inventory into the mandate so the loop self-corrects each retry.
        weight_facts = self._derive_checkpoint_weight_facts(text)
        if weight_facts:
            source_context = (weight_facts + "\n\n" + source_context) if source_context else weight_facts
        # §1b ENABLEMENT PLAYBOOK renders mandate.task_description via _section_enablement_playbook.
        # notes carries only per-dispatch dynamic context that §1b cannot provide.
        notes = _mandate_notes(state.enablement, attempt=attempt)
        gap_cid = f"gap.enablement.{signature.kind}"

        params_out: dict[str, Any] = {
            "domain": "enablement_specialist",
            "source_phase": "ENABLEMENT",
            "gap_canonical_id": gap_cid,
            "gap_symptom": (f"{framework or '?'} cannot launch {model or 'the target model'}: {signature.kind}"),
            "gap_layer": "framework",
            "gap_evidence": {"model": model, "failure_kind": signature.kind},
            "framework": framework,
            "lever_kind": LEVER_ENABLEMENT,
            "enablement_attempt": attempt,
            "enablement_failure_kind": signature.kind,
            "enablement_search_repos": list(plan.repos),
            "enablement_candidate_refs": list(candidate_refs),
            # Source lines near the offending site, plus (on a weight-init
            # failure) the checkpoint's per-layer weight inventory. Rendered
            # into the mandate by _section_enablement_playbook.
            "enablement_source_context": source_context,
            # Config accumulated by prior advanced rounds. The bench variant
            # layers this round's proposal on top, so a KEEP is graded on the
            # whole stack and its effective_config records the whole stack.
            "base_extra_envs": state.enablement.accepted_extra_envs,
            "base_extra_args": state.enablement.accepted_extra_server_args,
            "source": "coordinator_internal",
            "notes": notes,
            **_round_carriers(state, signature),
        }
        last_build_failure = state.enablement.last_build_failure or {}
        if last_build_failure:
            params_out["enablement_last_build_failure"] = dict(last_build_failure)
        return params_out

    def _read_enablement_source_context(self, signature: Any, *, window: int = 12) -> str:
        """Best-effort read a small source window near the offending site.

        Resolves ``signature.offending_file`` against the kernel search roots,
        then returns ``window`` lines centred on the first occurrence
        of ``offending_symbol`` (or the file head when the symbol is absent).
        Fully exception-guarded: any failure returns ``""`` so the mandate
        degrades to the no-context form (G is grounding, never a hard dependency).

        Delegates to :func:`~..actions.executors._apply_feedback.source_context_for_file`
        which is the shared file-resolve + window primitive.

        Args:
            signature: The classified :class:`FailureSignature`.
            window: Total number of lines to return around the hit.

        Returns:
            str: A ``file:line`` header + snippet, or ``""``.
        """
        offending_file = str(getattr(signature, "offending_file", "") or "").strip()
        if not offending_file:
            return ""
        symbol = str(getattr(signature, "offending_symbol", "") or "").strip()
        from ..actions.executors._apply_feedback import source_context_for_file
        from hyperloom.inference_optimizer.framework_paths import resolve_kernel_search_roots

        search_roots = [Path(str(r)) for r in resolve_kernel_search_roots()]
        return source_context_for_file(
            offending_file,
            symbol=symbol,
            window=window,
            search_roots=search_roots,
        )

    def _derive_checkpoint_weight_facts(self, launch_log: str) -> str:
        """Auto-derive ground-truth checkpoint-weight facts for a weight-init failure.

        On a weight-loading error (strict init / state_dict mismatch), parses the
        offending ``model...`` parameter names from the launch log and
        cross-references the model's ``*.index.json`` ``weight_map`` to report,
        per offending family, which layer indices carry that weight in the
        checkpoint and which do not. Returns a compact FACTS block appended to the
        enablement mandate. Fully exception-guarded: any failure returns ``""``.

        Args:
            launch_log: The captured launch / traceback text.

        Returns:
            str: A ``CHECKPOINT WEIGHT FACTS`` block, or ``""``.
        """
        text = launch_log or ""
        low = text.lower()
        # Offending parameter names in the traceback; parsed first so the
        # trigger is robust to a head-truncated launch log.
        offending = set(re.findall(r"['\"]((?:model|language_model|transformer)\.[\w.]+)['\"]", text))
        weighty = {o for o in offending if o.endswith((".weight", ".bias", "_scale"))}
        phrase_hit = (
            "not initialized from checkpoint" in low
            or "missing key" in low
            or "unexpected key" in low
            or "error(s) in loading state_dict" in low
        )
        # Fire when the log names weight-shaped offending params even if the
        # explanatory phrase was truncated off.
        if not offending or not (phrase_hit or weighty):
            return ""
        try:
            weight_map = _checkpoint_weight_map(self.shared_state.model_path)
        except (OSError, ValueError):
            log.debug("enablement: checkpoint weight-facts derivation failed", exc_info=True)
            return ""
        if not weight_map:
            return ""
        missing_by_family = _layers_by_family(offending)
        present_by_family = _layers_by_family(weight_map.keys())
        lines: list[str] = []
        for fam in sorted(missing_by_family):
            present_layers = sorted(present_by_family.get(fam, ()))
            missing_layers = sorted(missing_by_family[fam])
            if present_layers:
                lines.append(
                    f"- '{fam}': PRESENT in checkpoint for layers {present_layers}; "
                    f"MISSING (instantiated by model, absent from checkpoint) for layers "
                    f"{missing_layers}. To satisfy the strict init check, the missing layers "
                    f"must obtain this tensor from a present layer (copy/alias from the nearest "
                    f"preceding present layer) OR the model must not instantiate it there."
                )
            else:
                lines.append(
                    f"- '{fam}': NOT present in the checkpoint for ANY layer "
                    f"(missing for layers {missing_layers}). The checkpoint has no source for "
                    f"this tensor — the model should not require it (guard/skip its "
                    f"instantiation) rather than copy it."
                )
        header = (
            "CHECKPOINT WEIGHT FACTS (auto-derived from the model's "
            "safetensors index — GROUND TRUTH, prefer over assumptions). The "
            "boot failed on weight initialization; for each offending tensor "
            "family, here is exactly which layers carry it in the checkpoint:"
        )
        footer = (
            "IMPORTANT: verify the exact model class + its load_weights() entry "
            "point actually used for this architecture (grep the framework "
            "source for the architecture/model_type) and confirm the parameter-"
            "dict key naming (with/without a 'model.' prefix) at that scope "
            "BEFORE writing copy logic — a prior fix silently no-op'd because "
            "it edited the wrong loader / used mismatched key names, so the copy "
            "never executed and the SAME weights stayed uninitialized."
        )
        return header + "\n" + "\n".join(lines) + "\n" + footer

    def _discover_enablement_candidate_refs(
        self,
        req: Any,
        plan: Any,
        *,
        deadline: Deadline | None = None,
    ) -> tuple[str, ...]:
        """Best-effort enumerate + rank bridging PRs for an enablement failure.

        Enumerates candidate PRs across every repo in ``plan.repos`` (framework
        + opted-in ROCm/HIP/aiter bridge repos) via the ``sources`` layer, then
        ranks each :class:`hyperloom.agents.framework.models.Candidate` with
        ``score_enablement_title`` (per-Candidate so the ref/html_url is
        preserved) and returns the top ``req.max_search_candidates`` refs
        (``html_url`` preferred).

        Network + git. Enumeration is guarded per repo, so a repo that fails is
        skipped and the rest are still ranked; a run that collects nothing
        returns an empty tuple and the mandate falls back to repos-only.

        ``deadline`` bounds the worker thread, not the caller: enumeration stops
        between repos and ranks whatever was collected.

        Args:
            req: The :class:`hyperloom.common.failure_signature.EnablementRequest`.
            plan: The :class:`hyperloom.orchestrator.enablement.mandate.EnablementSearchPlan`.
            deadline: When to stop enumerating further repos.

        Returns:
            tuple[str, ...]: Ranked candidate refs (best first; possibly empty).
        """
        from .mandate import score_enablement_title
        from hyperloom.agents.framework.models import Candidate, CandidateSearchRequest, PRMonitorConfig
        from hyperloom.agents.framework.sources import enumerate_candidates
        from hyperloom.common.pr_monitor_urls import pr_monitor_base_url

        max_candidates = int(getattr(req, "max_search_candidates", 5) or 5)
        # The PR query service is co-hosted by KB Store.
        plane = self.knowledge_plane
        pr_enabled = bool(plane is not None and getattr(plane, "pr_monitor_enabled", False))
        pr_monitor_url = pr_monitor_base_url() if pr_enabled else ""
        pr_monitor = PRMonitorConfig(base_url=pr_monitor_url) if pr_monitor_url else None
        search_modes = ("pr_monitor", "github") if pr_monitor else ("github",)

        collected: list[Candidate] = []
        for repo in plan.repos:
            if deadline is not None and deadline.expired():
                log.info(
                    "enablement: candidate discovery stopped at its deadline with %d collected",
                    len(collected),
                )
                break
            try:
                search = CandidateSearchRequest(
                    repo_url=repo,
                    search_modes=search_modes,
                    keywords=tuple(plan.keywords),
                    max_search_candidates=max_candidates,
                    pr_monitor=pr_monitor,
                )
                collected.extend(enumerate_candidates(search))
            except (PRMonitorError, http.client.HTTPException):
                log.debug(
                    "enablement: candidate discovery failed for repo=%s",
                    repo,
                    exc_info=True,
                )
                continue

        if not collected:
            return ()
        ranked = sorted(
            collected,
            key=lambda c: score_enablement_title(getattr(c, "title", "") or "", plan),
            reverse=True,
        )
        refs: list[str] = []
        seen: set[str] = set()
        for cand in ranked:
            ref = str(getattr(cand, "html_url", "") or getattr(cand, "ref", "") or "").strip()
            if ref and ref not in seen:
                seen.add(ref)
                refs.append(ref)
            if len(refs) >= max_candidates:
                break
        return tuple(refs)
