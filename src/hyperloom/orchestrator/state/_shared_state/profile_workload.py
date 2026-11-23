# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The workload a profile trace was taken under, and whether the current workload still matches it."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, ClassVar
from hyperloom_common.coerce import first_text, to_int, to_str_list
from hyperloom.common.profile_args import sanitize_profile_server_args

log = logging.getLogger(__name__)


def _merged_profile_envs(current_best: Mapping[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    """The current best's env overrides, with the caller's base and variant envs layered on top."""
    raw_current_envs = current_best.get("extra_envs")
    merged_envs = dict(raw_current_envs) if isinstance(raw_current_envs, dict) else {}
    if isinstance(incoming.get("base_extra_envs"), dict):
        merged_envs.update(incoming["base_extra_envs"])
    if isinstance(incoming.get("extra_envs"), dict):
        merged_envs.update(incoming["extra_envs"])
    return merged_envs


def _normalize_envs(value: Any) -> dict[str, str]:
    """Coerce an env mapping to ``str -> str``; a resumed non-dict reads as empty."""
    return {str(k): str(v) for k, v in value.items()} if isinstance(value, dict) else {}


# Workload fields a profile task may override; the rest of the profile identity is read off the session.
PROFILE_OVERRIDABLE_KEYS: tuple[str, ...] = (
    "framework",
    "precision",
    "model_path",
    "tp",
    "conc",
    "isl",
    "osl",
    "max_model_len",
)


# ``base_*`` task param -> the ``current_best`` field it mirrors and its normalizer.
_STACK_BASE_FIELDS: tuple[tuple[str, str, Callable[[Any], Any]], ...] = (
    ("base_extra_args", "extra_server_args", lambda v: str(v or "").strip()),
    ("base_extra_envs", "extra_envs", _normalize_envs),
    ("base_remove_args", "remove_args", to_str_list),
    ("base_unset_envs", "unset_envs", to_str_list),
    ("base_args_mode", "args_mode", lambda v: "replace" if str(v or "").strip().lower() == "replace" else ""),
)


class _ProfileWorkloadMixin:
    """Records the workload of the last profile and tells whether its trace still describes the current one."""

    #: Fields of :meth:`profile_workload_context` that say *what was profiled*.
    #: The rest -- ``server_args``, ``extra_envs``, ``remove_args``,
    #: ``unset_envs``, ``args_mode`` -- say how the profile task was
    #: parameterized, and are only populated when the recorder had those params
    #: to hand. Two call sites record the same trace differently for that reason
    #: alone, so comparing them makes a perfectly fresh trace read as stale.
    #: ``serving_config`` is excluded here too: it has its own comparison, which
    #: comes from ``current_best`` on both sides and is therefore symmetric.
    #:
    #: ClassVar so it never serialises into state.json.
    PROFILE_WORKLOAD_IDENTITY_KEYS: ClassVar[tuple[str, ...]] = ("benchmark_mode", *PROFILE_OVERRIDABLE_KEYS)

    @classmethod
    def profile_workload_identity(cls, context: Any) -> dict[str, Any]:
        """Project a workload context down to what identifies the profiled run."""
        if not isinstance(context, Mapping):
            return {}
        return {key: context.get(key) for key in cls.PROFILE_WORKLOAD_IDENTITY_KEYS}

    def profile_workload_context(
        self,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the normalized workload and runtime identity for a profile trace."""
        from hyperloom.inference_optimizer.grid_server_args import normalize_server_args

        params = overrides if isinstance(overrides, dict) else {}
        workload = {
            name: params[name] if params.get(name) not in (None, "") else getattr(self, name, "")
            for name in PROFILE_OVERRIDABLE_KEYS
        }
        text = {name: str(workload[name] or "").strip() for name in ("framework", "precision", "model_path")}
        model_path = text["model_path"]
        if model_path:
            path = Path(model_path).expanduser()
            if path.exists():
                try:
                    model_path = str(path.resolve())
                except OSError:
                    log.debug(
                        "profile workload path resolution failed for %s; keeping the unresolved path",
                        path,
                        exc_info=True,
                    )
        runtime = {name: params[name] if name in params else params.get(base) for base, name, _ in _STACK_BASE_FIELDS}
        server_args = sanitize_profile_server_args(str(runtime["extra_server_args"] or "").strip())
        joined_args = normalize_server_args(server_args)
        serving_args = str(self.current_best.get("extra_server_args") or "").strip()
        serving_envs = {
            key: value
            for key, value in sorted(_normalize_envs(self.current_best.get("extra_envs")).items())
            if key.strip()
        }
        # benchmark_mode distinguishes AgentX from synthetic traces with the
        # same CONC/TP so a synthetic trace is never reused for an AgentX session.
        context: dict[str, Any] = {
            "benchmark_mode": str(self.benchmark_mode).strip().lower(),
            "framework": text["framework"].lower(),
            "precision": text["precision"].lower(),
            "model_path": model_path,
            **{name: to_int(workload[name], 0) for name in ("tp", "conc", "isl", "osl", "max_model_len")},
            "server_args": joined_args,
            "extra_envs": dict(sorted(_normalize_envs(runtime["extra_envs"]).items())),
            "remove_args": sorted(set(to_str_list(runtime["remove_args"]))),
            "unset_envs": sorted(set(to_str_list(runtime["unset_envs"]))),
            "args_mode": str(runtime["args_mode"] or "append").strip().lower(),
        }
        if any((serving_args, serving_envs)):
            context["serving_config"] = {"extra_server_args": serving_args, "extra_envs": serving_envs}
        return context

    def profile_trace_usable(self) -> bool:
        """Whether the latest profile left a trace; ``recovered`` was salvaged from a run that reported failure."""
        return self.last_profile_status in {"succeeded", "recovered"}

    def profile_trace_matches_workload(self) -> bool:
        """Whether the recorded profile trace is fresh for the current-best workload."""
        recorded = self.last_profile_workload
        if not self.profile_trace_usable() or not isinstance(recorded, dict) or not recorded:
            return False
        # Compare against the *current-best* runtime identity, not the bare context: last_profile_workload is recorded
        # with the real profile params (actual server_args / extra_envs), while profile_workload_context() with no
        # overrides reports server_args="" and skips the current_best runtime backfill, so any workload with server
        # args/extra envs would compare unequal and every fresh profile would be discarded as stale.
        return recorded == self.current_profile_workload_context()

    def record_profile_workload(
        self,
        params: dict[str, Any] | None = None,
        *,
        arm: str = "",
    ) -> dict[str, Any]:
        """Record the workload identity and originating arm for a profile trace."""
        self.last_profile_workload = self.profile_workload_context(params or {})
        if arm == "baseline":
            self.last_profile_workload_action = "baseline"
        else:
            self.last_profile_workload_action = str(self.current_best.get("action") or "")
        self.last_profile_args = str(self.last_profile_workload.get("server_args") or "")
        return self.last_profile_workload

    def current_profile_workload_context(
        self,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the profile identity for the active current-best runtime."""
        current_best = self.current_best
        incoming = overrides if isinstance(overrides, dict) else {}
        params: dict[str, Any] = {}
        runtime_keys = {
            "extra_server_args",
            "extra_envs",
            "remove_args",
            "unset_envs",
            "args_mode",
            "base_extra_args",
            "base_extra_envs",
            "base_remove_args",
            "base_unset_envs",
            "base_args_mode",
        }
        has_runtime_override = any(key in current_best or key in incoming for key in runtime_keys)
        recorded = self.last_profile_workload if isinstance(self.last_profile_workload, dict) else {}
        # A bare baseline/profile ``current_best`` carries no runtime fields, so the only record of what the server
        # actually ran with is the last profile.
        recorded_action = first_text(self.last_profile_workload_action)
        if (
            not has_runtime_override
            and current_best.get("action") in {"baseline", "profile"}
            and recorded
            # Legacy sessions predate the recorded action; keep reusing them rather than forcing a reprofile on every
            # run.
            and recorded_action in {"", "baseline", "profile"}
        ):
            params.update(
                {
                    "base_extra_args": recorded.get("server_args", ""),
                    "base_extra_envs": dict(recorded.get("extra_envs") or {}),
                    "base_remove_args": list(recorded.get("remove_args") or []),
                    "base_unset_envs": list(recorded.get("unset_envs") or []),
                    "base_args_mode": recorded.get("args_mode", "append"),
                }
            )
        for source, target in (
            ("extra_server_args", "base_extra_args"),
            ("remove_args", "base_remove_args"),
            ("unset_envs", "base_unset_envs"),
            ("args_mode", "base_args_mode"),
        ):
            if source in current_best:
                params[target] = current_best[source]
            if source in incoming:
                params[target] = incoming[source]
            if target in incoming:
                params[target] = incoming[target]

        merged_envs = _merged_profile_envs(current_best, incoming)
        if merged_envs:
            params["base_extra_envs"] = merged_envs

        for name in PROFILE_OVERRIDABLE_KEYS:
            if name in incoming:
                params[name] = incoming[name]
        return self.profile_workload_context(params)
