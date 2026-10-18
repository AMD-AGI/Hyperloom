# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shadow runs' noise, read off consistency.jsonl, and the dispersion prior it gives hosted measurement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hyperloom_multibranch.notebook.rsi.archive import MeasurementNoise
from hyperloom_multibranch.notebook.rsi.priors import HOSTED, DispersionTask, PriorSource, dispersion_prior
from hyperloom_multibranch.notebook.rsi.shadow import ShadowNoise, read_shadow, shadow_noise
from hyperloom_multibranch.notebook.rsi.tests.trees import TASK, tree

MODEL, GPU, TP = TASK.labels["model"], TASK.labels["gpu"], 8
PRIOR_TASK = DispersionTask(MODEL, TP, GPU)
LABELS = {"t1": {"model": MODEL, "tp": TP, "gpu": GPU}, "t2": {"model": "Qwen/Qwen3-8B", "tp": 1, "gpu": GPU}}


def _item(test_id: str, diff: float | None, hosted_cv: float | None, local_cv: float | None) -> dict[str, Any]:
    return {
        "type": "item",
        "id": test_id,
        "compared": diff is not None,
        "throughput_diff_signed_pct": diff,
        "hosted": {"cv_pct": hosted_cv, "output_throughput": 1000.0},
        "local": {"cv_pct": local_cv, "output_throughput": 1010.0},
    }


@pytest.fixture
def report(tmp_path: Path) -> Path:
    path = tmp_path / "consistency.jsonl"
    lines = [_item("t1", -1.0, 0.6, 0.4), _item("t2", -3.0, 1.0, 0.2), _item("t3", None, None, 0.3)]
    text = "\n".join([*(json.dumps(line) for line in lines), "{not json", json.dumps({"type": "summary"})])
    path.write_text(text + "\n", encoding="utf-8")
    return path


def test_the_items_give_the_median_noise_of_each_side_and_the_hosted_offset(report: Path) -> None:
    items = read_shadow(report)

    assert [item.test_id for item in items] == ["t1", "t2", "t3"]
    assert items[2].offset_pct is None and items[2].hosted_cv_pct is None
    assert shadow_noise(items) == ShadowNoise(0.8, 0.3, -2.0, ("t1", "t2", "t3"))


def test_with_labels_only_the_tests_labelled_like_the_session_count(report: Path) -> None:
    noise = shadow_noise(read_shadow(report), labels=LABELS, model=MODEL.lower(), tp=TP, gpu=GPU.lower())

    assert noise == ShadowNoise(0.6, 0.4, -1.0, ("t1",))
    assert shadow_noise(read_shadow(report), labels=LABELS, model=MODEL, tp=4, gpu=GPU).test_ids == ()


def test_without_a_matching_hosted_session_the_shadow_runs_give_the_hosted_prior(report: Path) -> None:
    noise = shadow_noise(read_shadow(report))
    archived = tree("hosted-a", noise=MeasurementNoise(1.4, None, 3, HOSTED))

    shadowed = dispersion_prior([], PRIOR_TASK, shadow=noise)
    measured = dispersion_prior([archived], PRIOR_TASK, shadow=noise)
    local = dispersion_prior([], PRIOR_TASK, environment="local", shadow=noise)

    assert (shadowed.source, shadowed.dispersion_pct, shadowed.samples, shadowed.offset_pct) == (
        PriorSource.SHADOW,
        0.8,
        3,
        -2.0,
    )
    assert (measured.source, measured.dispersion_pct, measured.offset_pct) == (PriorSource.ARCHIVE, 1.4, -2.0)
    assert local.source is PriorSource.GLOBAL
    assert dispersion_prior([], PRIOR_TASK).offset_pct is None
