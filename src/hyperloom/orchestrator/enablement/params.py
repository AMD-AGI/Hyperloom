# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Enablement authoring-specialist request construction."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.breakdown.agent_ownership import LEVER_ENABLEMENT

from ..collaborator import CoordinatorCollaborator

import logging as _logging

log = _logging.getLogger(__name__)


def _maybe_build_runtime_candidate(
    capability_gap: Any,
    *,
    framework: str,
    model: str,
    gpu_type: str,
) -> dict[str, Any] | None:
    """Build a serialized runtime-candidate stack action, or None.

    Returns None when the gap does not require code acquisition, the run is
    multi-node (single-node-only guard), or the framework adapter cannot produce
    an evidence-backed candidate. Fully exception-guarded.
    """
    if not getattr(capability_gap, "requires_code_acquisition", False):
        return None
    try:
        from ..actions.executors._multi_node_env import is_multi_node

        if is_multi_node():
            return None
        from ..framework.adapters import get_adapter

        adapter = get_adapter(framework)
        action = adapter.build_stack_action(capability_gap, framework=framework, model=model, gpu_type=gpu_type)
        if action is None:
            return None
        return action.to_state()
    except Exception:  # noqa: BLE001 — candidate construction is best-effort
        log.debug("enablement: runtime-candidate construction failed", exc_info=True)
        return None


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


def _maybe_build_localization_candidate(
    capability_gap: Any,
    *,
    framework: str,
    model: str,
    repo_url: str,
    candidate_refs: tuple[str, ...],
) -> dict[str, Any] | None:
    """Build a serialized localization stack action, or None.

    Returns None when the gap does not require code acquisition, the run is
    multi-node, there is no merged-PR candidate ref, or the framework adapter
    cannot localize. The compiled-closure gate runs later in the executor.
    """
    if not getattr(capability_gap, "requires_code_acquisition", False):
        return None
    ref = next((r for r in (candidate_refs or ()) if str(r).strip()), "")
    if not ref or not repo_url:
        return None
    try:
        from ..actions.executors._multi_node_env import is_multi_node

        if is_multi_node():
            return None
        from ..framework.adapters import get_adapter

        adapter = get_adapter(framework)
        action = adapter.build_localization_action(
            capability_gap, framework=framework, model=model, candidate_ref=ref, repo_url=repo_url
        )
        if action is None:
            return None
        return action.to_state()
    except Exception:  # noqa: BLE001 — candidate construction is best-effort
        log.debug("enablement: localization-candidate construction failed", exc_info=True)
        return None


class EnablementParams(CoordinatorCollaborator):
    """Builds the enablement authoring specialist's parameters."""

    def _build_enablement_specialist_params(self, launch_log: str, *, attempt: int = 0) -> dict[str, Any] | None:
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

        On a retry (``attempt > 0``) the ranked candidate list is *rotated* so a
        different bridging PR leads, and the notes flag that prior attempts
        reverted — steering the sub-agent toward a different bridge.

        Args:
            launch_log: Captured launch / traceback text.
            attempt: Zero-based dispatch index; drives candidate rotation and a
                retry hint in the mandate.

        Returns:
            dict | None: Specialist task params (tagged ``enablement`` +
            ``framework_agent_authoring``) or ``None``.
        """
        text = (launch_log or "").strip()
        if not text:
            return None
        from hyperloom.agents.framework.enablement import EnablementRequest, classify_failure
        from hyperloom.agents.framework.enablement_ops import build_search_plan
        from hyperloom.agents.framework.repo_map import repo_url_for_framework

        state = self.shared_state
        framework = (getattr(state, "framework", "") or "").strip().lower()
        model = (getattr(state, "model_name", "") or "").strip()
        repo_url = repo_url_for_framework(framework)

        # Dispatch a specialist for ANY non-blank launch log, even one that
        # classifies as ``UNKNOWN``: ``kind`` is advisory (routes bridge-repo
        # hints and labels the mandate), not a gate.
        signature = classify_failure(text)
        req = EnablementRequest(
            framework=framework,
            model=model or "(target model)",
            repo_url=repo_url,
            launch_log=text,
            gpu_type=(getattr(state, "gpu_type", "") or "").strip().lower(),
        )
        plan = build_search_plan(signature, framework_repo_url=repo_url, model=model)
        candidate_refs = self._discover_enablement_candidate_refs(req, plan)
        # Lead with a different candidate each attempt (deterministic left-rotation).
        if candidate_refs and attempt:
            n = len(candidate_refs)
            k = attempt % n
            candidate_refs = candidate_refs[k:] + candidate_refs[:k]
        # Persist so _maybe_escalate_to_targeted_build can pick the top candidate.
        state.enablement.candidate_refs = list(candidate_refs)
        source_context = self._read_enablement_source_context(signature)
        # For a weight-init failure, fold the checkpoint's ground-truth per-layer
        # weight inventory into the mandate so the loop self-corrects each retry.
        weight_facts = self._derive_checkpoint_weight_facts(text)
        if weight_facts:
            source_context = (weight_facts + "\n\n" + source_context) if source_context else weight_facts
        # Progressing patches from prior rounds, re-applied as a base before this
        # round's patch (serial-gap stacking); author a fix composing on top.
        base_patches = [str(p) for p in (state.enablement.kept_patches or [])]
        # Whole-file artifacts kept by prior rounds, re-installed before the
        # boot the same way patches are re-applied.
        base_artifacts = list(state.enablement.kept_artifacts or [])
        # Only prior rounds' *actually-applied* setup commands (recorded by the
        # specialist and replayed by integrate_patch) stack as a base. No install
        # command is ever auto-seeded here: an unpinned upgrade of the shared
        # serving venv is unsafe (CUDA-wheel clobber of ROCm vLLM/torch,
        # transformers-major skew) and environment/build acquisition is owned by
        # the isolated targeted-build path + the specialist's own setup_commands.
        base_setup = [str(c) for c in (state.enablement.setup_commands or [])]
        # §1b ENABLEMENT PLAYBOOK renders mandate.task_description via _section_enablement_playbook.
        # notes carries only per-dispatch dynamic context that §1b cannot provide.
        notes = ""
        grounding_drops = list(state.enablement.last_grounding_drop_reason or [])
        spanned_roots = bool(state.enablement.patches_span_multiple_roots)
        acc_envs = dict((state.enablement.accepted_config or {}).get("extra_envs") or {})
        acc_args = str((state.enablement.accepted_config or {}).get("extra_server_args") or "").strip()
        if base_patches or base_setup or base_artifacts or acc_envs or acc_args:
            progress_bits = []
            if base_patches:
                progress_bits.append(f"{len(base_patches)} prior patch(es): {base_patches}")
            if base_artifacts:
                art_targets = [a["target"] for a in base_artifacts[:4]]
                progress_bits.append(f"{len(base_artifacts)} prior artifact(s) installed: {art_targets}")
            if base_setup:
                progress_bits.append(f"{len(base_setup)} prior setup command(s): {base_setup}")
            if acc_envs or acc_args:
                cfg_bits = []
                if acc_envs:
                    cfg_bits.append(f"envs={acc_envs}")
                if acc_args:
                    cfg_bits.append(f"args={acc_args!r}")
                progress_bits.append("accumulated config from prior advanced rounds: " + "; ".join(cfg_bits))
            notes = (
                "STACKED ENABLEMENT (progress so far): the following already "
                "cleared earlier boot crashes and WILL be re-applied/re-run as a "
                "base before your changes — do NOT redo them; fix only the CURRENT "
                "(deeper) failure, composing on top. " + "; ".join(progress_bits)
            )
        elif attempt:
            notes = (
                f"RETRY (attempt {attempt + 1}): a previous enablement patch for this "
                f"failure was REVERTED (did not make the combo runnable). Try a DIFFERENT "
                f"bridging approach / candidate than before."
            )
        if grounding_drops:
            drop_note = (
                "PATCH GROUNDING FAILURE: the patches the prior round submitted "
                "were dropped because their target file(s) do not exist in ANY "
                "allowlisted framework source tree. Targets: "
                + "; ".join(grounding_drops[:4])
                + ". Verify the target path with Glob/Read inside the worktree "
                "before authoring the diff."
            )
            notes = (drop_note + "\n\n" + notes).strip() if notes else drop_note
        if spanned_roots:
            span_note = (
                "CROSS-TREE PATCH SET: the prior round's patches targeted more "
                "than one framework source tree, which integrate_patch cannot "
                "apply as one set. Submit patches for a single tree per round."
            )
            notes = (span_note + "\n\n" + notes).strip() if notes else span_note
        gap_cid = f"gap.enablement.{signature.kind}"
        from hyperloom.agents.framework.enablement import CapabilityGap

        capability_gap = CapabilityGap.from_signature(signature)

        # When the gap requires code acquisition (not a resource constraint) and
        # an adapter can build an evidence-backed candidate, attach a
        # ``runtime_candidate`` so integrate_patch provisions an attempt-scoped
        # runtime before booting. Skipped in multi-node mode.
        runtime_candidate = _maybe_build_runtime_candidate(
            capability_gap, framework=framework, model=model, gpu_type=req.gpu_type
        )
        # When a merged-PR candidate exists, attach a ``localization_candidate``
        # so integrate_patch localizes the closure into the source tree
        # (compiled closures defer to the targeted build at apply).
        localization_candidate = _maybe_build_localization_candidate(
            capability_gap,
            framework=framework,
            model=model,
            repo_url=repo_url,
            candidate_refs=tuple(candidate_refs),
        )

        params_out: dict[str, Any] = {
            "domain": "enablement_specialist",
            "source_phase": "ENABLEMENT",
            "gap_canonical_id": gap_cid,
            "gap_symptom": (f"{framework or '?'} cannot launch {model or 'the target model'}: {signature.kind}"),
            "gap_layer": "framework",
            "gap_evidence": {"model": model, "failure_kind": signature.kind},
            "framework": framework,
            # Enablement tag routes the integrate gate to runnable_decision.
            "framework_agent_authoring": True,
            "enablement": True,
            "lever_kind": LEVER_ENABLEMENT,
            "enablement_attempt": attempt,
            "enablement_failure_kind": signature.kind,
            "enablement_search_repos": list(plan.repos),
            # Pre-patch failure signature, replayed by integrate_patch.
            "enablement_before_signature": signature.to_dict(),
            # CapabilityGap projection: marks resource_constraint as not actionable.
            "enablement_capability_gap": capability_gap.to_dict(),
            "enablement_candidate_refs": list(candidate_refs),
            # Source lines near the offending site, plus (on a weight-init
            # failure) the checkpoint's per-layer weight inventory. Rendered
            # into the mandate by _section_enablement_playbook.
            "enablement_source_context": source_context,
            # Progressing patches from prior rounds, stacked as a base.
            "enablement_base_patches": base_patches,
            # Whole-file artifact records from prior rounds, re-installed before boot.
            "enablement_base_artifacts": base_artifacts,
            # Allowlisted setup commands from prior rounds, replayed before boot.
            "enablement_setup_commands": base_setup,
            # Config accumulated by prior advanced rounds. The bench variant
            # layers this round's proposal on top, so a KEEP is graded on the
            # whole stack and its effective_config records the whole stack.
            "base_extra_envs": acc_envs,
            "base_extra_args": acc_args,
            "launch_probe": req.launch_probe,
            "source": "coordinator_internal",
            "notes": notes,
            # Whole-machine GPU request. Empty on multi-node / no-GPU hosts.
            **self._framework_gpu_params(),
            # eval-origin trigger context (empty for boot-origin enablement).
            **_enablement_carrier_params(state),
        }
        if runtime_candidate is not None:
            params_out["runtime_candidate"] = runtime_candidate
        # Re-activate a prior KEEP'd attempt runtime so serial stacking runs on
        # the same runtime the last round promoted.
        kept_action = state.enablement.kept_stack_action
        if isinstance(kept_action, dict) and kept_action and "runtime_candidate" not in params_out:
            params_out["runtime_candidate"] = kept_action
        if localization_candidate is not None:
            params_out["localization_candidate"] = localization_candidate
        # Inject the last targeted-build failure into the mandate.
        last_build_failure = state.enablement.last_build_failure or {}
        if isinstance(last_build_failure, dict) and last_build_failure:
            fc = str(last_build_failure.get("failure_class") or "")
            fs = str(last_build_failure.get("failure_summary") or "")
            if fc or fs:
                build_note = (
                    f"PREVIOUS TARGETED-BUILD ATTEMPT: failure_class={fc!r}"
                    + (f"; {fs}" if fs else "")
                    + "\nIf the build ran out of time (failure_class='timeout' or "
                    "'preflight_budget'), request more build_budget_sec or a smaller "
                    "component scope.  For compile/symbol defects, choose a different "
                    "ref or narrow the build target."
                )
                notes = build_note + "\n\n" + notes
                params_out["notes"] = notes
            params_out["enablement_last_build_failure"] = dict(last_build_failure)
        return params_out

    def _read_enablement_source_context(self, signature: Any, *, window: int = 12) -> str:
        """Best-effort read a small source window near the offending site.

        Resolves ``signature.offending_file`` against the framework/ROCm source
        allowlist, then returns ``window`` lines centred on the first occurrence
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
        from ..framework.paths import resolve_source_file_allowlist

        search_roots = [Path(str(r)) for r in resolve_source_file_allowlist()]
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
        try:
            import glob as _glob
            import json as _json
            import re as _re
            from pathlib import Path as _Path

            # Offending parameter names in the traceback; parsed first so the
            # trigger is robust to a head-truncated launch log.
            offending = set(_re.findall(r"['\"]((?:model|language_model|transformer)\.[\w.]+)['\"]", text))
            weighty = {o for o in offending if o.endswith((".weight", ".bias", "_scale"))}
            phrase_hit = (
                "not initialized from checkpoint" in low
                or "missing key" in low
                or "unexpected key" in low
                or "error(s) in loading state_dict" in low
            )
            # Fire when the log names weight-shaped offending params even if the
            # explanatory phrase was truncated off.
            if not (phrase_hit or weighty):
                return ""
            if not offending:
                return ""
            model_path = str(getattr(self.shared_state, "model_path", "") or "").strip()
            if model_path:
                # ``model_path`` may be an HF repo id; resolve to the local
                # weights dir so the sharded-index read works for repo-id launches.
                from hyperloom.inference_optimizer.model_config_utils import (
                    resolve_local_model_dir,
                )

                _resolved = resolve_local_model_dir(model_path)
                if _resolved is not None:
                    model_path = str(_resolved)
            if not model_path or not _Path(model_path).is_dir():
                return ""
            # Load the checkpoint weight_map (sharded index) or list single-file keys.
            weight_map: dict[str, Any] = {}
            idx_files = _glob.glob(f"{model_path}/*.index.json")
            if idx_files:
                data = _json.loads(_Path(idx_files[0]).read_text(errors="replace"))
                weight_map = data.get("weight_map", {}) if isinstance(data, dict) else {}
            if not weight_map:
                return ""
            ckpt_keys = set(weight_map.keys())

            # Group offending names by a layer-index-stripped "family".
            def _family(name: str) -> str:
                return _re.sub(r"\.\d+\.", ".{N}.", name)

            def _layer_idx(name: str) -> int | None:
                m = _re.search(r"\.(\d+)\.", name)
                return int(m.group(1)) if m else None

            fams: dict[str, dict[str, Any]] = {}
            for nm in offending:
                fam = _family(nm)
                d = fams.setdefault(fam, {"missing_layers": set()})
                li = _layer_idx(nm)
                if li is not None:
                    d["missing_layers"].add(li)
            lines: list[str] = []
            for fam in sorted(fams):
                # Which layer indices for this family exist in the checkpoint?
                fam_re = _re.compile("^" + _re.escape(fam).replace(r"\{N\}", r"\d+") + "$")
                present_layers = sorted(
                    {li for k in ckpt_keys if fam_re.match(k) for li in [_layer_idx(k)] if li is not None}
                )
                missing_layers = sorted(fams[fam]["missing_layers"])
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
            if not lines:
                return ""
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
        except Exception:  # noqa: BLE001 — auto-facts are best-effort grounding
            log.debug("enablement: checkpoint weight-facts derivation failed", exc_info=True)
            return ""

    def _discover_enablement_candidate_refs(self, req: Any, plan: Any) -> tuple[str, ...]:
        """Best-effort enumerate + rank bridging PRs for an enablement failure.

        Enumerates candidate PRs across every repo in ``plan.repos`` (framework
        + opted-in ROCm/HIP/aiter bridge repos) via the ``sources`` layer, then
        ranks each :class:`framework_agent.models.Candidate` with
        ``score_enablement_title`` (per-Candidate so the ref/html_url is
        preserved) and returns the top ``req.max_search_candidates`` refs
        (``html_url`` preferred).

        Network + git; **fully exception-guarded**: any failure degrades to an
        empty tuple so the mandate falls back to repos-only.

        Args:
            req: The :class:`framework_agent.enablement.EnablementRequest`.
            plan: The :class:`framework_agent.enablement_ops.EnablementSearchPlan`.

        Returns:
            tuple[str, ...]: Ranked candidate refs (best first; possibly empty).
        """
        from hyperloom.agents.framework.enablement_ops import score_enablement_title
        from hyperloom.agents.framework.models import Candidate, ExploreRequest
        from hyperloom.agents.framework.sources import enumerate_candidates

        max_candidates = int(getattr(req, "max_search_candidates", 5) or 5)
        # Only search primus_cortex when its URL is configured.
        primus_url = str(os.environ.get("PRIMUS_CORTEX_PR_API") or "").strip()
        if primus_url:
            search_modes = ["primus_cortex", "github"]
            primus_block: dict[str, Any] = {"primus_cortex": {"base_url": primus_url}}
        else:
            search_modes = ["github"]
            primus_block = {}

        collected: list[Candidate] = []
        for repo in plan.repos:
            try:
                explore_req = ExploreRequest.from_dict(
                    {
                        "framework": getattr(req, "framework", "") or "sglang",
                        "repo_url": repo,
                        "work_dir": str(
                            getattr(req, "work_dir", None) or (Path(tempfile.gettempdir()) / "framework-agent")
                        ),
                        "baseline": {"throughput": 1.0},
                        "search_perf_prs": True,
                        "search_modes": search_modes,
                        "keywords": list(plan.keywords),
                        "pr_states": ["all"],
                        "max_search_candidates": max_candidates,
                        **primus_block,
                    }
                )
                collected.extend(enumerate_candidates(explore_req))
            except Exception:  # noqa: BLE001 — discovery is best-effort
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
