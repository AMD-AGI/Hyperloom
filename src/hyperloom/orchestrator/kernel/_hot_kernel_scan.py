# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Rank a trace's hot kernels and ask the kernel-opt ledger which of them still owe an attempt."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, NamedTuple

from kernelforge.knowledge.kb_store.identity.implementation import normalize_operator_name

from hyperloom.common.coerce import to_float, to_int

from .patch_landing import IntegratedStack
from ..state.kernel_decision_settings import (
    _DEFAULT_HOT_KERNEL_GATE_TOP_N,
    effective_hot_kernel_gpu_pct,
    effective_hot_kernel_min_gpu_pct,
    resolve_hot_kernel_min_gpu_pct,
)


class TraceRank(NamedTuple):
    """TraceLens' order for one hot row.

    Rows TraceLens never ranked (bypass route, injected collectives) sort behind the ranked ones and are capped
    separately, so they cannot fall behind TraceLens' own cap.
    """

    unranked: bool
    pitem_rank: int
    negated_impact: float


class KernelIdentity(NamedTuple):
    """The underlying kernel behind a row, independent of the synthetic per-row kernel_id."""

    source_file: str
    operation: str
    gpu_pct: float


class HotKernelRow(NamedTuple):
    """One hot kernel the scan admitted, with the task group it is dispatched as part of."""

    rank: TraceRank
    kernel_id: str
    source_file: str
    members: list[str]
    group_key: str
    identity: KernelIdentity


@dataclass(frozen=True)
class TaskGroups:
    """A trace's ``task_groups`` metadata, indexed by the kernel ids it names."""

    members_of: dict[str, list[str]]
    group_key_of: dict[str, str]
    aliases_of: dict[str, set[str]]

    @classmethod
    def of(cls, task_groups: Any) -> TaskGroups:
        """Index one trace's task groups by member kernel id."""
        members_of: dict[str, list[str]] = {}
        group_key_of: dict[str, str] = {}
        aliases_of: dict[str, set[str]] = {}
        for group in task_groups:
            if not isinstance(group, dict):
                continue
            members = [str(member) for member in (group.get("kernel_ids") or []) if member]
            group_key = str(group.get("task_group_key") or "")
            aliases_of[group_key] = _group_key_aliases(group, group_key)
            for member in members:
                members_of[member] = members
                group_key_of[member] = group_key
        return cls(members_of=members_of, group_key_of=group_key_of, aliases_of=aliases_of)

    def members_for(self, kernel_id: str) -> list[str]:
        """The members this kernel is dispatched with; the kernel alone when the trace named no group."""
        return sorted(self.members_of.get(kernel_id) or [kernel_id])

    def group_key_for(self, kernel_id: str) -> str:
        """The task-group key this kernel belongs to, or ``""`` when the trace named none."""
        return self.group_key_of.get(kernel_id, "")

    def aliases_for(self, group_key: str) -> set[str]:
        """Every key this group has been written under, falling back to the key itself."""
        return self.aliases_of.get(group_key) or {group_key}


def _group_key_aliases(group: dict[str, Any], group_key: str) -> set[str]:
    """Every task-group key one group has been written under, current and legacy."""
    legacy = [str(alias) for alias in (group.get("legacy_task_group_keys") or []) if str(alias)]
    return {alias for alias in (group_key, *legacy) if alias}


class HotKernelGate(NamedTuple):
    """The impact floor and the rank cap a hot-kernel scan admits rows under."""

    min_gpu_pct: float
    top_n: int

    @classmethod
    def resolve(cls, min_gpu_pct: float | None, top_n: int | None) -> HotKernelGate:
        """Fill whichever of the two bounds the caller left to the environment."""
        if min_gpu_pct is None:
            min_gpu_pct = resolve_hot_kernel_min_gpu_pct()
        if top_n is None:
            top_n = to_int(os.environ.get("HYPERLOOM_KERNEL_OPT_GATE_TOP_N"), default=_DEFAULT_HOT_KERNEL_GATE_TOP_N)
        return cls(min_gpu_pct=min_gpu_pct, top_n=max(1, int(top_n)))

    def admits(self, hot_kernel: dict[str, Any]) -> bool:
        """Whether a hot row is dispatchable and hot enough to owe a ``kernel_opt`` attempt."""
        return (
            hot_kernel.get("reusable_native_kernel") is True
            and hot_kernel.get("shape_dispatchable") is not False
            and effective_hot_kernel_gpu_pct(hot_kernel)
            >= effective_hot_kernel_min_gpu_pct(hot_kernel, self.min_gpu_pct)
        )


def _hot_kernel_row(hot_kernel: Any, groups: TaskGroups, gate: HotKernelGate) -> HotKernelRow | None:
    """One admitted hot row, or ``None`` for a row this scan does not rank."""
    if not isinstance(hot_kernel, dict) or not gate.admits(hot_kernel):
        return None
    kernel_id = str(hot_kernel.get("kernel_id") or "")
    if not kernel_id:
        return None
    source_file = str(hot_kernel.get("source_file") or "")
    pitem_rank = int(hot_kernel.get("tracelens_pitem_rank") or 0)
    return HotKernelRow(
        rank=TraceRank(pitem_rank <= 0, pitem_rank, -float(hot_kernel.get("impact_score") or 0.0)),
        kernel_id=kernel_id,
        source_file=source_file,
        members=groups.members_for(kernel_id),
        group_key=groups.group_key_for(kernel_id),
        identity=KernelIdentity(
            source_file=source_file,
            operation=str(hot_kernel.get("name") or hot_kernel.get("operation") or ""),
            gpu_pct=to_float(hot_kernel.get("gpu_pct"), default=0.0),
        ),
    )


def _one_row_per_task(rows: list[HotKernelRow]) -> list[HotKernelRow]:
    """Keep the strongest row of each task group, and of each underlying kernel identity."""
    seen_groups: set[str | tuple[str, ...]] = set()
    seen_identities: set[KernelIdentity] = set()
    kept: list[HotKernelRow] = []
    for row in rows:
        dedup_key: str | tuple[str, ...] = row.group_key or tuple(row.members)
        if dedup_key in seen_groups:
            continue
        if row.identity.source_file and row.identity.operation:
            if row.identity in seen_identities:
                continue
            seen_identities.add(row.identity)
        seen_groups.add(dedup_key)
        kept.append(row)
    return kept


def ranked_hot_rows(hot: list[Any], groups: TaskGroups, gate: HotKernelGate) -> list[HotKernelRow]:
    """The admitted hot rows in TraceLens order, one per task, each half capped at the gate's ``top_n``."""
    admitted = [row for row in (_hot_kernel_row(k, groups, gate) for k in hot) if row is not None]
    deduped = _one_row_per_task(sorted(admitted, key=lambda row: row.rank))
    ranked = [row for row in deduped if not row.rank.unranked]
    unranked = [row for row in deduped if row.rank.unranked]
    return ranked[: gate.top_n] + unranked[: gate.top_n]


@dataclass(frozen=True)
class KernelAttemptLedger:
    """Everything a hot row is checked against before it is called untried, read once per scan."""

    attempts: dict[str, Any]
    rejected: set[str]
    integrated: IntegratedStack
    integrated_operators: set[str]
    groups: TaskGroups

    @classmethod
    def of(cls, state: Any, groups: TaskGroups, integrated_operators: set[str]) -> KernelAttemptLedger:
        """Read the stable ledger, the rejections and the landed stack for one scan."""
        return cls(
            attempts=state.kernel_opt_task_attempts or {},
            rejected=set(state.rejected_kernel_ids or []),
            integrated=IntegratedStack.of(state.optimization_stack),
            integrated_operators=integrated_operators,
            groups=groups,
        )

    def still_owes_attempt(self, row: HotKernelRow) -> bool:
        """Whether this hot row has a ``kernel_opt`` attempt still outstanding."""
        return not (self._whole_task_rejected(row) or self._already_landed(row) or self._already_attempted(row))

    def attempt_for(self, member_id: str) -> dict[str, Any]:
        """Ledger entry covering ``member_id``, tolerating synthetic-id churn."""
        for value in self.attempts.values():
            if not isinstance(value, dict):
                continue
            if member_id in {
                str(value.get("current_kernel_id") or ""),
                str(value.get("kernel_id") or ""),
                str(value.get("task_group_primary_kernel_id") or ""),
            }:
                return value
            for key in ("task_group_kernel_ids", "opfanout_collapsed_ids"):
                if member_id in {str(m) for m in (value.get(key) or []) if m}:
                    return value
        return {}

    def member_is_rejected(self, member_id: str) -> bool:
        """True when ``member_id`` (or its ledger twin) is out of play."""
        if member_id in self.rejected:
            return True
        attempt = self.attempt_for(member_id)
        if not attempt:
            return False
        if str(attempt.get("rejected_reason") or "").strip():
            return True
        return str(attempt.get("integration_status") or "").strip().lower() == "rejected"

    def matches_current_task(self, member_id: str, group_key: str, source: str) -> bool:
        """Whether the ledger's record of this member describes the task the trace is ranking now."""
        if group_key:
            return self._first_group_attempt(group_key) is not None
        attempt = self.attempt_for(member_id)
        if not attempt:
            return False
        recorded_source = str(attempt.get("last_source_file") or "")
        return not source or not recorded_source or source == recorded_source

    def _first_group_attempt(self, group_key: str) -> dict[str, Any] | None:
        """The first ledger entry written under this task group, by current key or legacy alias."""
        aliases = self.groups.aliases_for(group_key)
        return next(
            (
                attempt
                for attempt in self.attempts.values()
                if isinstance(attempt, dict)
                and (
                    str(attempt.get("stable_task_key") or "") in aliases
                    or str(attempt.get("task_group_key") or "") in aliases
                )
            ),
            None,
        )

    def _whole_task_rejected(self, row: HotKernelRow) -> bool:
        """Whether every member of this row's task is out of play under the task the trace is ranking now."""
        return bool(row.members) and all(
            self.member_is_rejected(member) and self.matches_current_task(member, row.group_key, row.source_file)
            for member in row.members
        )

    def _already_landed(self, row: HotKernelRow) -> bool:
        """Whether an integrating row already retired this kernel."""
        if self.integrated.covers(
            kernel_ids=row.members,
            task_group_key=row.group_key,
            source_file=row.source_file,
            task_group_aliases=self.groups.aliases_of.get(row.group_key),
        ):
            return True
        operation = row.identity.operation
        return bool(operation) and normalize_operator_name(operation) in self.integrated_operators

    def _already_attempted(self, row: HotKernelRow) -> bool:
        """Whether this row's task has already spent a ``kernel_opt`` attempt."""
        if row.group_key:
            attempt = self._first_group_attempt(row.group_key)
            return attempt is not None and int(attempt.get("attempts", 0)) > 0
        return any(
            self.matches_current_task(member, "", row.source_file)
            and int((self.attempt_for(member) or {}).get("attempts", 0)) > 0
            for member in row.members
        )


__all__ = [
    "HotKernelGate",
    "HotKernelRow",
    "KernelAttemptLedger",
    "KernelIdentity",
    "TaskGroups",
    "TraceRank",
    "ranked_hot_rows",
]
