# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How noisy hosted measurement is, from the shadow runs: ``consistency.jsonl``, read for the dispersion prior (Q1).

The hosted consistency check (``hyperloom.measure.consistency``) runs each checklist test on the local backend and in a
fresh hosted sandbox, N times each when the test sets ``repeats``; every test is one ``"type": "item"`` line. What it
says of the noise, each a median over the tests that have it:

* ``hosted_cv_pct``: the hosted side's ``cv_pct``, the sample standard deviation of its readings over their mean in
  percent, the same measure as a session's baseline dispersion; a test run once has none;
* ``local_cv_pct``: the same of the local side;
* ``offset_pct``: ``throughput_diff_signed_pct``, hosted minus local over local, of the tests both sides measured: how
  far hosted reads from local, systematically.

A line names its checklist test (``id``), not the model, TP or GPU it ran. ``labels`` maps a test id to those, and then
only the tests labelled like the session count; without ``labels`` every test counts, for any task.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.jsonio import read_jsonl
from hyperloom_multibranch.notebook.rsi.similarity import MODEL, model_name

ITEM = "item"


@dataclass(frozen=True)
class ShadowItem:
    """One checklist test of the shadow runs, reduced to its noise."""

    test_id: str
    hosted_cv_pct: float | None
    local_cv_pct: float | None
    offset_pct: float | None


@dataclass(frozen=True)
class ShadowNoise:
    """The noise of the shadow runs' tests that count: their medians, and which tests they are."""

    hosted_cv_pct: float | None
    local_cv_pct: float | None
    offset_pct: float | None
    test_ids: tuple[str, ...] = ()


def _percent(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def shadow_item(line: Mapping[str, Any]) -> ShadowItem:
    """One ``consistency.jsonl`` item line; a side or a difference it lacks is ``None``."""

    hosted, local = line.get("hosted") or {}, line.get("local") or {}
    return ShadowItem(
        test_id=str(line.get("id") or ""),
        hosted_cv_pct=_percent(hosted.get("cv_pct")),
        local_cv_pct=_percent(local.get("cv_pct")),
        offset_pct=_percent(line.get("throughput_diff_signed_pct")) if line.get("compared") else None,
    )


def read_shadow(path: Path) -> tuple[ShadowItem, ...]:
    """The item lines of the ``consistency.jsonl`` at ``path``; the summary and unreadable lines are left out."""

    lines = read_jsonl(path, skip_malformed=True, skip_non_dict=True)
    return tuple(shadow_item(line) for line in lines if line.get("type") == ITEM)


def _labelled_like(labels: Mapping[str, Any], *, model: str, tp: int, gpu: str) -> bool:
    return (
        model_name(str(labels.get(MODEL, ""))) == model_name(model)
        and str(labels.get("gpu", "")).lower() == gpu.lower()
        and labels.get("tp") == tp
    )


def _median(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def shadow_noise(
    items: Sequence[ShadowItem],
    *,
    labels: Mapping[str, Mapping[str, Any]] | None = None,
    model: str = "",
    tp: int = 0,
    gpu: str = "",
) -> ShadowNoise:
    """The noise of ``items``: with ``labels``, only of the tests labelled ``model``, ``tp`` and ``gpu``."""

    counted = [
        item
        for item in items
        if labels is None or _labelled_like(labels.get(item.test_id, {}), model=model, tp=tp, gpu=gpu)
    ]
    return ShadowNoise(
        hosted_cv_pct=_median([item.hosted_cv_pct for item in counted]),
        local_cv_pct=_median([item.local_cv_pct for item in counted]),
        offset_pct=_median([item.offset_pct for item in counted]),
        test_ids=tuple(item.test_id for item in counted),
    )


__all__ = ["ITEM", "ShadowItem", "ShadowNoise", "read_shadow", "shadow_item", "shadow_noise"]
