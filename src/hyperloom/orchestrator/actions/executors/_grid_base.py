# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared value types and helpers for the ``explore`` executor's grid runs."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from hyperloom.common.coerce import to_str_list
from hyperloom.common.env_safety import filter_untrusted_env_mapping, is_allowed_variant_env_key
from ._canonical_fingerprint import canonical_fingerprint

log = logging.getLogger(__name__)


# Content-based variant fingerprint (cross-action dedup ledger key).
def variant_fingerprint(
    extra_server_args: str | None,
    extra_envs: dict[str, Any] | None,
    *,
    remove_args: list[str] | tuple[str, ...] | set[str] | str | None = None,
    unset_envs: list[str] | tuple[str, ...] | set[str] | str | None = None,
    args_mode: str = "append",
    runtime_override: dict[str, Any] | None = None,
) -> str:
    """Stable content fingerprint for a (extra_server_args, extra_envs) pair."""
    return canonical_fingerprint(
        extra_server_args,
        extra_envs,
        remove_args=remove_args,
        unset_envs=unset_envs,
        args_mode=args_mode,
        runtime_override=runtime_override,
    )


# Single home for the grid-level defaults every graded executor shares.
DEFAULT_VARIANT_TIMEOUT_SEC = 7800  # 130 min; matches BASELINE_DEFAULT_TIMEOUT_SEC
# Per-variant KEEP threshold (gain-pct + accuracy gate); the grid noise floor.
DEFAULT_KEEP_THRESHOLD_PCT = 1.0


@dataclass
class GridVariant:
    """One row of the grid we're going to test."""

    name: str
    extra_server_args: str = ""
    extra_envs: dict[str, str] = field(default_factory=dict)
    remove_args: list[str] = field(default_factory=list)
    unset_envs: list[str] = field(default_factory=list)
    args_mode: str = "append"
    note: str = ""

    def __init__(
        self,
        name: str,
        extra_server_args: str = "",
        extra_envs: dict[str, str] | None = None,
        note: str = "",
        *,
        remove_args: list[str] | tuple[str, ...] | set[str] | str | None = None,
        unset_envs: list[str] | tuple[str, ...] | set[str] | str | None = None,
        args_mode: str = "append",
    ) -> None:
        """Initialize a grid variant descriptor."""
        self.name = name
        self.extra_server_args = extra_server_args
        self.extra_envs, dropped_envs = filter_untrusted_env_mapping(
            extra_envs,
            allow_predicate=is_allowed_variant_env_key,
        )
        if dropped_envs:
            log.warning("Variant %s: dropping unsafe extra_envs %s", name, ", ".join(sorted(dropped_envs)))
        self.remove_args = to_str_list(remove_args)
        self.unset_envs = to_str_list(unset_envs)
        mode = str(args_mode or "append").strip().lower()
        self.args_mode = mode if mode in {"append", "replace"} else "append"
        self.note = note
        # Optional runtime override; injected into materialized YAML benchmark.envs by _build_variant_yaml so the
        # server subprocess resolves the attempt runtime.
        self.runtime_override: dict[str, str] = {}

    @property
    def fingerprint(self) -> str:
        """Content fingerprint used as dedup-ledger key. See module doc."""
        return variant_fingerprint(
            self.extra_server_args,
            self.extra_envs,
            remove_args=self.remove_args,
            unset_envs=self.unset_envs,
            args_mode=self.args_mode,
            runtime_override=getattr(self, "runtime_override", None) or None,
        )


def coerce_extra_envs(value: Any) -> dict[str, str]:
    """Normalize Orchestration-supplied ``extra_envs`` to ``dict[str,str]``."""
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items() if k is not None}
    if isinstance(value, str):
        out: dict[str, str] = {}
        # Split on the first ``=`` only to preserve URL-style assignments like ``HF_ENDPOINT=https://...``.
        tokens = re.split(r"[\s;]+", value.strip())
        for tok in tokens:
            if not tok:
                continue
            if "=" not in tok:
                continue
            k, v = tok.split("=", 1)
            k = k.strip()
            if not k:
                continue
            out[k] = v.strip()
        return out
    if isinstance(value, (list, tuple)):
        out_l: dict[str, str] = {}
        for item in value:
            if isinstance(item, dict):
                # ``[{"FOO": "1"}, {"BAR": "2"}]`` — later entries win.
                for k, v in item.items():
                    if k is None:
                        continue
                    out_l[str(k)] = str(v)
                continue
            if not isinstance(item, str) or "=" not in item:
                continue
            k, v = item.split("=", 1)
            k = k.strip()
            if not k:
                continue
            out_l[k] = v.strip()
        return out_l
    return {}


@dataclass
class VariantResult:
    """One bench run's parsed result."""

    name: str
    extra_server_args: str
    extra_envs: dict[str, str]
    status: str
    output_throughput: float | None = None
    request_throughput: float | None = None
    total_token_throughput: float | None = None
    completed_requests: int | None = None
    duration_seconds: float | None = None
    ttft_mean_ms: float | None = None
    e2el_mean_ms: float | None = None
    tpot_mean_ms: float | None = None
    input_throughput: float | None = None
    tpot_p90_ms: float | None = None
    intvty_p90: float | None = None
    workspace: str | None = None
    report_path: str | None = None
    raw_result_path: str | None = None
    reported_success: bool | None = None
    returncode: int | None = None
    nonfatal_warnings: list[str] = field(default_factory=list)
    error: str | None = None
    # Short failure-classification tag; empty for successes.
    error_class: str = ""
    note: str = ""
    # Wall-clock seconds the Magpie subprocess consumed.
    runtime_sec: float | None = None
    # True iff reaped by the overtime soft deadline.
    killed_overtime: bool = False
    # Rough output tok/s salvaged from server.log on the killed_overtime path; informational only, never feeds winner
    # selection.
    estimated_output_throughput: float | None = None
    server_log_path: str | None = None
    launch_evidence: dict[str, Any] = field(default_factory=dict)
    launch_evidence_path: str | None = None

    @property
    def fingerprint(self) -> str:
        """Same fingerprint scheme as :class:`GridVariant`."""
        return canonical_fingerprint(self.extra_server_args, self.extra_envs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result to a plain JSON-friendly dict."""
        return {
            "name": self.name,
            "extra_server_args": self.extra_server_args,
            "extra_envs": self.extra_envs,
            "fingerprint": self.fingerprint,
            "status": self.status,
            "output_throughput": self.output_throughput,
            "request_throughput": self.request_throughput,
            "total_token_throughput": self.total_token_throughput,
            "completed_requests": self.completed_requests,
            "duration_seconds": self.duration_seconds,
            "ttft_mean_ms": self.ttft_mean_ms,
            "e2el_mean_ms": self.e2el_mean_ms,
            "tpot_mean_ms": self.tpot_mean_ms,
            "input_throughput": self.input_throughput,
            "tpot_p90_ms": self.tpot_p90_ms,
            "e2e_norm_intvty_p90": self.intvty_p90,
            "workspace": self.workspace,
            "report_path": self.report_path,
            "raw_result_path": self.raw_result_path,
            "reported_success": self.reported_success,
            "returncode": self.returncode,
            "nonfatal_warnings": self.nonfatal_warnings,
            "error": self.error,
            "error_class": self.error_class,
            "note": self.note,
            "runtime_sec": self.runtime_sec,
            "killed_overtime": self.killed_overtime,
            "estimated_output_throughput": self.estimated_output_throughput,
            "server_log_path": self.server_log_path,
            "launch_evidence": self.launch_evidence,
            "launch_evidence_path": self.launch_evidence_path,
        }
