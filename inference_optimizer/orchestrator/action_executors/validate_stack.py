"""Real ``validate_stack`` ActionRunner — re-baseline with the full stack applied.

Phase 3 of the prompt-builder refactor. See
``inference_optimizer/actions/validate_stack.md`` for the full
playbook; in short:

* read ``SharedState.optimization_stack`` (the running list of KEEP'd
  modifications across backends / params / integrate rounds),
* concatenate every entry's ``extra_sglang_args`` and merge every
  entry's ``extra_envs`` (later entries override earlier ones — we
  follow the same precedence the live serving config uses),
* hand the combined args/envs to :class:`BaselineExecutor`'s rendering
  + subprocess machinery so we get an apples-to-apples Magpie
  benchmark on a fresh server,
* return a dict shaped like a baseline result so the Coordinator's
  delegated_result router can promote it into
  ``cumulative_gain_validated`` / ``cumulative_gain_validated_ts`` /
  ``cumulative_gain_validated_stack_len`` (see
  :meth:`Coordinator._promote_to_shared_state`).

The executor is **read-only** with respect to ``optimization_stack`` —
it never adds or removes entries. Mutation is the Coordinator's job
when KEEP/REVERT propagates from the explore / integrate paths.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from ..shared_state import SharedState
from ..sub_agent_runner import RunnerContext
from ._grid_runner import sanitize_script_name
from ._workload_envs import materialize_config_with_envs
from .baseline import BaselineExecutor


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (pure — independent of subprocess machinery for unit-testability)
# ---------------------------------------------------------------------------
def _entry_args(entry: dict[str, Any]) -> str:
    """Pick the args fragment recorded for one stack entry.

    Precedence mirrors :meth:`Coordinator._lift_to_current_best`:

    * ``candidate_extra_sglang_args`` — the per-round delta (preferred,
      so we don't double-count an arg that was already present on the
      previous current_best).
    * ``extra_sglang_args`` — the full args of the round's winner
      (used when the round seeded the stack from current_best).
    """
    if not isinstance(entry, dict):
        return ""
    candidate = str(entry.get("candidate_extra_sglang_args") or "").strip()
    if candidate:
        return candidate
    return str(entry.get("extra_sglang_args") or "").strip()


def _entry_envs(entry: dict[str, Any]) -> dict[str, str]:
    if not isinstance(entry, dict):
        return {}
    raw = entry.get("extra_envs") or {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        if k is None:
            continue
        out[str(k)] = "" if v is None else str(v)
    return out


def _entry_action(entry: dict[str, Any]) -> str:
    return str((entry or {}).get("action") or "").strip()


def _entry_variant(entry: dict[str, Any]) -> str:
    return str((entry or {}).get("variant_name") or "").strip()


def combine_optimization_stack(
    stack: list[dict[str, Any]],
    *,
    include_actions: list[str] | None = None,
    exclude_variants: list[str] | None = None,
) -> tuple[str, dict[str, str], list[dict[str, Any]]]:
    """Merge a list of optimization_stack entries into (args, envs, applied).

    * args        — concatenated ``extra_sglang_args`` (separator: single
                    space, dedup of trailing/leading whitespace per entry,
                    no further normalisation — sglang argv parsing wins
                    last-arg conflicts itself).
    * envs        — dict union, later entries override earlier ones.
    * applied     — the subset of input entries we actually merged, in
                    the order they were merged. Useful for the result
                    payload so the Coordinator + report can show which
                    entries contributed to the validated number.

    ``include_actions`` and ``exclude_variants`` filter the stack
    BEFORE merging. Both are matched verbatim — passing
    ``include_actions=['backends']`` keeps only the backends round
    KEEPs.
    """
    include_set = (
        {str(a).strip() for a in include_actions if str(a).strip()}
        if include_actions else None
    )
    exclude_set = (
        {str(v).strip() for v in exclude_variants if str(v).strip()}
        if exclude_variants else set()
    )
    args_parts: list[str] = []
    envs: dict[str, str] = {}
    applied: list[dict[str, Any]] = []
    for entry in stack or []:
        if not isinstance(entry, dict):
            continue
        action = _entry_action(entry)
        variant = _entry_variant(entry)
        if include_set is not None and action not in include_set:
            continue
        if variant and variant in exclude_set:
            continue
        args_fragment = _entry_args(entry)
        if args_fragment:
            args_parts.append(args_fragment)
        envs.update(_entry_envs(entry))
        applied.append({
            "action": action or "?",
            "variant_name": variant or "?",
            "args": args_fragment,
            "envs": _entry_envs(entry),
            "tput_at_keep": entry.get("tput"),
        })
    return " ".join(args_parts).strip(), envs, applied


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
class ValidateStackExecutor(BaselineExecutor):
    """Reuse baseline's subprocess + parsing machinery, but:

    1. read SharedState's ``optimization_stack`` to compose the args/envs
       (instead of running with no modifications), and
    2. enrich the returned dict with ``validated_stack_len`` /
       ``applied_args`` / ``applied_envs`` / ``applied_entries`` so the
       Coordinator can update ``cumulative_gain_validated*`` and the
       report can show *which* stack contributed to the number.

    All other knobs (timeout, workspace resolution, accuracy gate,
    materialised YAML reuse) come from the parent class unchanged.
    """

    def _resolve_workspace(self, ctx: RunnerContext, action: str) -> Path:
        # Use a dedicated subdir so validate_stack runs don't pollute the
        # baseline/ namespace and remain easy to find for post-mortem.
        return super()._resolve_workspace(ctx, "validate_stack")

    async def __call__(self, ctx: RunnerContext) -> dict[str, Any]:
        params = dict(ctx.task.params or {})
        # Resolve the optimization stack. Two sources, in priority order:
        # 1. Explicit ``stack`` passed via task.params (tests / Coordinator
        #    overrides).
        # 2. SharedState loaded from the ACTIVE session_dir.
        #
        # N28 fix: resolve the active session_dir lazily at every call
        # using ``ctx.extra["session_dir"]`` (injected by SubAgentRunner
        # at task dispatch time -- see sub_agent_runner.py line 127),
        # falling back to ``self.session_dir`` for direct-instantiation
        # tests, then to the env-driven ``_resolve_session_dir()``
        # global fallback.
        #
        # Why the lazy resolve matters: ``self.session_dir`` is captured
        # in ``BaselineExecutor.__init__`` at MODULE-IMPORT time (via
        # ``validate_stack_executor = ValidateStackExecutor()`` at the
        # bottom of this file), which happens BEFORE cli.py's
        # ``make_session_dir(model_name=...)`` creates the per-session
        # subdir and pins INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR. So
        # ``self.session_dir`` ends up pinned to the workspace root
        # (e.g. ``/wekafs/xiaofei/sessions``) instead of the per-session
        # subdir (e.g. ``/wekafs/.../<model>/<UTC ts>/``), and
        # ``SharedState.load_or_init`` reads the workspace root's
        # nonexistent state.json -> blank SharedState -> empty
        # optimization_stack -> bogus "stack_len=0" warning that
        # silently degrades validate_stack to a pure baseline run.
        # The empirical case that drove this fix: SOLAR-10.7B TP=1
        # session 20260521T062837Z, where decode_steps_16 promoted to
        # optimization_stack at 07:11:10 with +0.74% gain, then the
        # 07:13:36 validate_stack ignored the stack entirely and just
        # re-ran a baseline noise check (+0.33% noise, NOT the real
        # +0.74% validation the operator wanted).
        from ._grid_runner import _resolve_session_dir
        ctx_sd = (ctx.extra or {}).get("session_dir") if ctx.extra else None
        if ctx_sd:
            active_session_dir = Path(str(ctx_sd))
        elif self.session_dir:
            active_session_dir = Path(self.session_dir)
        else:
            active_session_dir = _resolve_session_dir()
        stack: list[dict[str, Any]] | None = params.pop("stack", None)
        if stack is None:
            try:
                state = SharedState.load_or_init(active_session_dir)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "validate_stack_executor: failed to load SharedState "
                    "from %s: %s",
                    active_session_dir, exc,
                )
                state = SharedState()
            stack = list(state.optimization_stack or [])

        include_actions = params.pop("include_actions", None) or None
        exclude_variants = params.pop("exclude_variants", None) or None
        combined_args, combined_envs, applied = combine_optimization_stack(
            stack,
            include_actions=include_actions,
            exclude_variants=exclude_variants,
        )
        validated_stack_len = len(stack)

        # Caller may pass extra_sglang_args / extra_envs themselves; we
        # APPEND theirs after the stack-derived ones so an ad-hoc test
        # override wins (last arg wins on conflict).
        caller_args = str(params.pop("extra_sglang_args", "") or "").strip()
        caller_envs = dict(params.pop("extra_envs", {}) or {})
        merged_args = " ".join(part for part in (combined_args, caller_args) if part)
        merged_envs = dict(combined_envs)
        merged_envs.update(caller_envs)

        if not merged_args and not merged_envs:
            log.warning(
                "validate_stack_executor: optimization_stack is empty "
                "(stack_len=%d). Falling through to a pure baseline run; "
                "the resulting cumulative_gain_validated will equal "
                "current baseline noise.", validated_stack_len,
            )

        # Prepare the materialised YAML manually so we control exactly
        # what gets written, then hand the path back to BaselineExecutor
        # via ``config_path`` and let the parent run the subprocess +
        # parse the report.
        config_path = Path(
            params.get("config_path")
            or self.default_config_path
            or self._resolve_default_config()
        )
        if not config_path.exists():
            return {
                "status": "failed",
                "error_class": "missing_config",
                "error": f"validate_stack config not found: {config_path}",
                "validated_stack_len": validated_stack_len,
                "applied_args": merged_args,
                "applied_envs": merged_envs,
                "applied_entries": applied,
            }
        output_dir = self._resolve_workspace(ctx, "validate_stack")
        output_dir.mkdir(parents=True, exist_ok=True)

        resolved_model = (
            str(params.get("model_path") or "").strip()
            or os.environ.get("MODEL_PATH", "").strip()
        )
        resolved_gpu = (
            str(params.get("gpu_type") or "").strip().lower()
            or os.environ.get("GPU_TYPE", "").strip().lower()
        )
        # Forward the Orchestration-supplied ``benchmark_script`` override
        # so validate_stack honors the same script-selection routing as
        # baseline (see SKILL.md "Magpie leak-path salvage"). If we
        # skipped this, validate_stack would silently fall back to
        # Magpie's runner_type-derived script (e.g. dsr1_fp8_mi300x.sh
        # with hardcoded ``--result-dir /workspace/``) every time, even
        # after Orchestration learned that ``sglang_mi300x.sh`` succeeds
        # for this model. ``params.result_dir`` does NOT need explicit
        # handling here — it flows through ``forwarded_params`` to
        # BaselineExecutor.__call__, which sets ``$RESULT_DIR``.
        try:
            override_script = sanitize_script_name(params.get("benchmark_script"))
        except ValueError as exc:
            return {
                "status": "failed",
                "error_class": "bad_param",
                "error": str(exc),
                "validated_stack_len": validated_stack_len,
                "applied_args": merged_args,
                "applied_envs": merged_envs,
                "applied_entries": applied,
            }
        materialised = materialize_config_with_envs(
            config_path,
            output_dir,
            extra_sglang_args=merged_args,
            extra_server_args=merged_args,
            extra_envs=merged_envs,
            model_path=resolved_model,
            gpu_type=resolved_gpu,
            benchmark_script=override_script,
            out_name="validate_stack_config.with_envs.yaml",
        )

        # Re-enter the parent with the materialised path baked into
        # task.params so its `__call__` skips re-materialising. We also
        # pin output_dir so the parent shares the workspace (timeout
        # logic + workspace post-mortem stay in one tree).
        forwarded_params = dict(params)
        forwarded_params["config_path"] = str(materialised)
        forwarded_params["output_dir"] = str(output_dir)
        # Single-node: pop merged args/envs. They're already baked into
        # the materialised YAML above; BaselineExecutor will call
        # ``materialize_config_with_envs`` again but its idempotent
        # "non-empty-or-skip" guard preserves the EXTRA_*_ARGS we just
        # wrote, so the launched Magpie subprocess sees the full stack.
        #
        # Multi-node: KEEP merged args/envs in ``task.params``. The
        # BaselineExecutor multi-node path forwards
        # ``params["extra_sglang_args"]`` straight into
        # ``restart_server_for_round`` -> ``cmd_restart_server`` which
        # boots sglang via the multi_node CLI **without** ever reading
        # the materialised YAML. Popping there silently launches sglang
        # with default args and drops the entire ``optimization_stack``,
        # making ``cumulative_gain_validated`` collapse to ~0% no matter
        # how many variants were KEPT (validate_stack would
        # re-baseline against a vanilla server). Forward merged values
        # so the restarted multi-node sglang sees every KEEP'd flag.
        from ._multi_node_env import is_multi_node
        if is_multi_node():
            forwarded_params["extra_sglang_args"] = merged_args
            if merged_envs:
                forwarded_params["extra_envs"] = merged_envs
            else:
                forwarded_params.pop("extra_envs", None)
        else:
            forwarded_params.pop("extra_sglang_args", None)
            forwarded_params.pop("extra_envs", None)
        ctx.task.params = forwarded_params  # type: ignore[assignment]
        result = await super().__call__(ctx)

        # Enrich with validate_stack-specific bookkeeping.
        if isinstance(result, dict):
            result.setdefault("validated_stack_len", validated_stack_len)
            result.setdefault("applied_args", merged_args)
            result.setdefault("applied_envs", merged_envs)
            result.setdefault("applied_entries", applied)
        return result


# Module-level callable so callers can do
# ``register_executor("validate_stack", validate_stack_executor)``.
validate_stack_executor = ValidateStackExecutor()


__all__ = [
    "ValidateStackExecutor",
    "combine_optimization_stack",
    "validate_stack_executor",
]
