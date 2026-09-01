# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared value types and helpers for the ``explore`` executor's grid runs.

Holds :class:`GridVariant` / :class:`VariantResult`, the content-fingerprint
delegate, ``extra_envs`` coercion, the Pareto filter, and the shared
per-variant timeout default. The runner that actually invokes Magpie and
parses ``benchmark_report.json`` lives in :mod:`._grid_runner`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from hyperloom.common.coerce import to_str_list
from hyperloom.common.env_safety import filter_untrusted_env_mapping, is_allowed_variant_env_key
from ._canonical_fingerprint import canonical_fingerprint

log = logging.getLogger(__name__)


# Content-based variant fingerprint (cross-action dedup ledger key). Delegates
# to :func:`canonical_fingerprint` (the single source of truth); both produce
# the identical 16-char content hash.
def variant_fingerprint(
    extra_server_args: str | None,
    extra_envs: dict[str, Any] | None,
    *,
    remove_args: list[str] | tuple[str, ...] | set[str] | str | None = None,
    unset_envs: list[str] | tuple[str, ...] | set[str] | str | None = None,
    args_mode: str = "append",
    runtime_override: dict[str, Any] | None = None,
) -> str:
    """Stable content fingerprint for a (extra_server_args, extra_envs) pair.

    Name and note are NOT inputs — variants with identical content but
    different names collapse to the same fingerprint. Delegates to
    :func:`canonical_fingerprint` so the two never drift.

    Args:
        extra_server_args (str | None): Backend server args for the variant.
        extra_envs (dict[str, Any] | None): Per-variant environment overrides.
        remove_args: Base/server args to remove before appending this variant.
        unset_envs: Inherited env names to unset before applying this variant.
        args_mode: ``"append"`` or ``"replace"``.
        runtime_override: Attempt runtime override; folded into the hash only
            when non-empty so plain variants keep their historical fingerprint.

    Returns:
        str: The 16-char content fingerprint of the pair.
    """
    return canonical_fingerprint(
        extra_server_args,
        extra_envs,
        remove_args=remove_args,
        unset_envs=unset_envs,
        args_mode=args_mode,
        runtime_override=runtime_override,
    )


_VARIANT_TIMEOUT_SEC_DEFAULT = 7800  # 130 min; matches BASELINE_DEFAULT_TIMEOUT_SEC


@dataclass
class GridVariant:
    """One row of the grid we're going to test.

    A single server-config candidate: the flags/env overrides to apply on top
    of the base Magpie config for one benchmark run.

    Attributes:
        name (str): Human-readable label for the variant.
        extra_server_args (str): Backend server args appended via
            ``EXTRA_{SGLANG,VLLM,ATOM}_ARGS``. Defaults to ``""``.
        extra_envs (dict[str, str]): Per-variant environment overrides, minus
            any name in ``BLOCKED_VARIANT_ENV_NAMES``. Defaults to an empty dict.
        remove_args (list[str]): Base/server flags to remove before appending
            this variant's args. Defaults to ``[]``.
        unset_envs (list[str]): Inherited environment keys to remove before
            applying ``extra_envs``. Defaults to ``[]``.
        args_mode (str): ``"append"`` (default) or ``"replace"``.
        note (str): Optional reason/category tag (e.g. ``multi_node_only_*``).
            Defaults to ``""``.
        runtime_override (dict[str, str]): Set on the instance only (not a
            constructor argument); injected into the materialized YAML's
            ``benchmark.envs`` by ``_build_variant_yaml`` and folded into
            :attr:`fingerprint`.
    """

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
        """Initialize a grid variant descriptor.

        Args:
            name: Variant name.
            extra_server_args: Extra server CLI args for this variant.
            extra_envs: Extra environment variables; unsafe names are dropped.
            note: Optional reason/category note.
            remove_args: Base/server args to remove before appending this
                variant's args.
            unset_envs: Inherited env names to remove before applying
                ``extra_envs``.
            args_mode: ``"append"`` or ``"replace"``.
        """
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
        # Optional runtime override; injected into materialized YAML benchmark.envs
        # by _build_variant_yaml so the server subprocess resolves the attempt runtime.
        self.runtime_override: dict[str, str] = {}

    @property
    def fingerprint(self) -> str:
        """Content fingerprint used as dedup-ledger key. See module doc.

        Returns:
            str: :func:`canonical_fingerprint` of this variant's
            ``extra_server_args`` and ``extra_envs``.
        """
        return variant_fingerprint(
            self.extra_server_args,
            self.extra_envs,
            remove_args=self.remove_args,
            unset_envs=self.unset_envs,
            args_mode=self.args_mode,
            runtime_override=getattr(self, "runtime_override", None) or None,
        )


def coerce_extra_envs(value: Any) -> dict[str, str]:
    """Normalize Orchestration-supplied ``extra_envs`` to ``dict[str,str]``.

    Accepts the three shapes the LLM emits — canonical dict, shell-style
    ``"FOO=1 BAR=2"`` string, and ``["FOO=1", "BAR=2"]`` token list — so
    downstream ``.items()`` callers never crash on a non-dict. Unknown shapes
    coerce to an empty dict.

    Args:
        value (Any): The Orchestration-supplied ``extra_envs`` in any of the
            accepted shapes (dict, shell-style string, or token list).

    Returns:
        dict[str, str]: The normalized env mapping; empty for unknown shapes.
    """
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items() if k is not None}
    if isinstance(value, str):
        out: dict[str, str] = {}
        # Split on the first ``=`` only to preserve URL-style assignments
        # like ``HF_ENDPOINT=https://...``.
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
    """One bench run's parsed result.

    The parsed outcome of a single variant's Magpie run: identity, status,
    headline throughput/latency metrics, artifact paths, and
    failure-classification metadata.

    Attributes:
        name (str): Variant label (mirrors :attr:`GridVariant.name`).
        extra_server_args (str): Server args used for this run.
        extra_envs (dict[str, str]): Env overrides used for this run.
        status (str): ``"succeeded"`` or ``"failed"``.
        output_throughput (float | None): Output tokens/sec, if measured.
        request_throughput (float | None): Requests/sec, if measured.
        total_token_throughput (float | None): Total tokens/sec, if measured.
        completed_requests (int | None): Number of completed requests.
        duration_seconds (float | None): Benchmark duration in seconds.
        ttft_mean_ms (float | None): Mean time-to-first-token (ms).
        e2el_mean_ms (float | None): Mean end-to-end latency (ms).
        tpot_mean_ms (float | None): Mean time-per-output-token (ms).
        input_throughput (float | None): Input tokens/sec (prefill), if measured.
        tpot_p90_ms (float | None): p90 inter-token latency (ms), if measured.
        intvty_p90 (float | None): p90 output tok/s/user (interactivity), if measured.
        workspace (str | None): Path to the located ``benchmark_*`` workspace.
        report_path (str | None): Path to ``benchmark_report.json`` if present.
        raw_result_path (str | None): Path to the raw result JSON, if salvaged.
        reported_success (bool | None): Magpie's own success flag, if known.
        returncode (int | None): Magpie subprocess return code.
        nonfatal_warnings (list[str]): Non-fatal warning tags (e.g. harvested
            leaked artifacts).
        error (str | None): Error summary for failed/nonzero runs.
        error_class (str): Short failure-classification tag (empty on success).
        note (str): Optional reason/category tag carried from the variant.
        runtime_sec (float | None): Wall-clock seconds the subprocess consumed.
        killed_overtime (bool): ``True`` iff reaped by the soft overtime
            deadline rather than crashing/timing-out/succeeding.
        estimated_output_throughput (float | None): Rough output tokens/sec
            estimated from the engine's periodic ``server.log`` throughput
            logs when the variant was killed before finishing. Informational
            only — never a real measurement and never used for winner
            selection; ``output_throughput`` stays ``None`` on the kill path.
        server_log_path (str | None): Absolute path to the variant's
            ``server.log`` when a server was launched; ``None`` for pre-launch
            failures where no server ran.
    """

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
    # Rough output tok/s salvaged from server.log on the killed_overtime path;
    # informational only, never feeds winner selection.
    estimated_output_throughput: float | None = None
    server_log_path: str | None = None

    @property
    def fingerprint(self) -> str:
        """Same fingerprint scheme as :class:`GridVariant`.

        Returns:
            str: :func:`canonical_fingerprint` of this result's
            ``extra_server_args`` and ``extra_envs``.
        """
        return canonical_fingerprint(self.extra_server_args, self.extra_envs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result to a plain JSON-friendly dict.

        Returns:
            dict[str, Any]: All result fields plus the computed
            ``fingerprint``, keyed by attribute name.
        """
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
            "intvty_p90": self.intvty_p90,
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
        }


def _numeric(value: Any) -> float | None:
    """Return a finite number, excluding bool (a subclass of int)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def pareto_front(
    entries: list[dict[str, Any]],
    *,
    latency_key: str = "e2el_mean_ms",
    x_key: str = "output_throughput",
    y_key: str | None = None,
    y_higher_is_better: bool = False,
) -> list[dict[str, Any]]:
    """Naive O(N²) Pareto on two axes.

    Default (flag off): maximize ``output_throughput``, minimize *latency_key*
    (native sweep: ``e2el_mean_ms``; GEAK sweep: ``ttft_mean_ms``).

    Composite Pareto: maximize ``total_token_throughput`` and maximize
    ``intvty_p90`` (pass ``y_higher_is_better=True``).

    Args:
        entries: Sweep result entries to filter.
        latency_key: Minimize axis when *y_key* is omitted (backward compatible).
        x_key: Maximize axis.
        y_key: Second axis; defaults to *latency_key*.
        y_higher_is_better: When True, maximize *y_key*; when False, minimize it.

    Returns:
        The non-dominated subset of succeeded entries that have both axes.
    """
    y = latency_key if y_key is None else y_key

    def _y_at_least(other: float, cand: float) -> bool:
        return other >= cand if y_higher_is_better else other <= cand

    def _y_strictly_better(other: float, cand: float) -> bool:
        return other > cand if y_higher_is_better else other < cand

    succ = [
        e
        for e in entries
        if e.get("status") == "succeeded"
        and _numeric(e.get(x_key)) is not None
        and _numeric(e.get(y)) is not None
    ]
    front: list[dict[str, Any]] = []
    for cand in succ:
        cand_x = _numeric(cand.get(x_key))
        cand_y = _numeric(cand.get(y))
        if cand_x is None or cand_y is None:
            continue
        dominated = False
        for other in succ:
            if other is cand:
                continue
            other_x = _numeric(other.get(x_key))
            other_y = _numeric(other.get(y))
            if other_x is None or other_y is None:
                continue
            if (
                other_x >= cand_x
                and _y_at_least(other_y, cand_y)
                and (other_x > cand_x or _y_strictly_better(other_y, cand_y))
            ):
                dominated = True
                break
        if not dominated:
            front.append(cand)
    return front


def select_sweep_pareto(
    entries: list[dict[str, Any]],
    *,
    framework: str | None = None,
    fallback_latency_key: str = "e2el_mean_ms",
) -> list[dict[str, Any]]:
    """Pareto for a sweep grid: composite axes when the flag is on, else tput/latency.

    Flag on: max total token throughput vs max p90 intvty. If no cell has both
    axes, fall back to output-tput vs *fallback_latency_key*.
    """
    from hyperloom.common.perf_metric import composite_grading_enabled

    if composite_grading_enabled(framework):
        front = pareto_front(
            entries,
            x_key="total_token_throughput",
            y_key="intvty_p90",
            y_higher_is_better=True,
        )
        if front:
            return front
    return pareto_front(entries, latency_key=fallback_latency_key)


def best_entry_for_each_conc(
    entries: list[dict[str, Any]],
    *,
    framework: str | None = None,
    state: Any = None,
    baseline_perf: Any = None,
) -> dict[str, dict[str, Any]]:
    """Best succeeded cell per concurrency.

    Flag on with a baseline triple: highest composite *S*. Cells missing a
    triple do not compete on *S*; if a conc has no scored cell, fall back to
    highest output throughput. Flag off: highest output throughput.
    """
    from hyperloom.common.perf_metric import (
        composite_grading_enabled,
        composite_score,
        perf_snapshot_from_mapping,
        resolve_baseline_perf,
    )

    baseline = baseline_perf or resolve_baseline_perf(state)
    use_composite = composite_grading_enabled(framework) and bool(baseline)
    scored: dict[str, tuple[float, dict[str, Any]]] = {}
    tput_best: dict[str, dict[str, Any]] = {}
    for e in entries:
        if e.get("status") != "succeeded":
            continue
        conc = str(e.get("conc"))
        tput = _numeric(e.get("output_throughput"))
        if tput is not None:
            cur = tput_best.get(conc)
            cur_tput = _numeric(cur.get("output_throughput")) if cur is not None else None
            if cur is None or cur_tput is None or tput > cur_tput:
                tput_best[conc] = e
        if use_composite and baseline is not None:
            snap = perf_snapshot_from_mapping(e)
            if snap is None:
                continue
            score = composite_score(snap, baseline)
            prev = scored.get(conc)
            if prev is None or score > prev[0]:
                scored[conc] = (score, e)
    if use_composite:
        out: dict[str, dict[str, Any]] = {}
        for conc in set(scored) | set(tput_best):
            out[conc] = scored[conc][1] if conc in scored else tput_best[conc]
        return out
    return tput_best
