# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for HY-WorldPlay frame extraction and run provenance.

Both cover the same class of failure: a benchmark that silently reinterprets its
own inputs, so the result looks like a measurement of the model when it is a
measurement of the harness.

Frame extraction feeds the perceptual quality gate. It used to infer the value
range of the clip, rescaling by ``(x + 1) / 2`` whenever the minimum pixel fell
below -0.01. One extreme pixel anywhere in 125 frames therefore rewrote all of
them, and because the reference and the candidate are extracted in separate
processes they could be rewritten differently — the gate would then report a
difference the model never produced. HY-WorldPlay's own ``save_video`` scales by
255 with no such guess, and that is the contract followed here.

The provenance map exists because a container drift went unnoticed for three
sessions: every leg drew saturated colour noise, and the result JSON recorded
torch, hip and python but not the packages that had actually moved.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

_BENCH_PATH = (
    Path(__file__).parents[1] / "assets" / "benchmark_scripts" / "bench_fps.py"
)


@pytest.fixture(scope="module")
def bench():
    """The loaded ``bench_fps`` module.

    It lives under ``assets/`` and is run by torchrun, so it is not importable by
    package path. Only ``torch`` is imported at module scope.
    """
    spec = importlib.util.spec_from_file_location("bench_fps", _BENCH_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _clip(frames: int = 8, value: float = 0.4) -> torch.Tensor:
    """A [1, C, F, H, W] clip of a constant value, the range save_video assumes."""
    return torch.full((1, 3, frames, 16, 16), float(value))


# --------------------------------------------------------------------------
# frame extraction
# --------------------------------------------------------------------------


def test_a_unit_range_clip_is_scaled_not_rescaled(bench):
    """[0,1] in, [0,255] out, following the pipeline's own writer."""
    frames, idx = bench._extract_frames_uint8(_clip(value=0.4), 4)

    assert frames.shape == (4, 16, 16, 3)
    assert idx == [0, 2, 5, 7]
    assert frames.min() == frames.max() == round(0.4 * 255)


def test_one_out_of_range_pixel_does_not_brighten_the_other_frames(bench):
    """The bug: a single excursion used to rewrite the whole clip.

    Inferring the range from ``arr.min()`` means the most extreme pixel anywhere
    in the clip decides how every frame is scaled. A diverging autoregressive
    rollout produces such a pixel readily, and the old mapping pushed a correctly
    exposed frame from 0.4 to 0.7 — visibly brighter, for a reason that has
    nothing to do with that frame.
    """
    clip = _clip(value=0.4)
    clip[0, 0, -1, 0, 0] = -0.5  # one pixel, last frame

    frames, _ = bench._extract_frames_uint8(clip, 8)

    assert frames[0].min() == frames[0].max() == round(0.4 * 255), (
        "an untouched frame was rescaled because a later frame went out of range"
    )


def test_a_reference_and_a_candidate_agree_wherever_their_content_agrees(bench):
    """Extraction must not depend on the clip it is applied to.

    The reference is written by one process and the candidate measured in
    another, so the float-to-uint8 mapping has to be the same for both or the two
    disagree about what a pixel means and the gate charges that to the patch.
    Only the pixel that genuinely differs may differ: under the old range guess
    this one excursion moved every other pixel from 102 to 178.
    """
    ref = _clip(value=0.4)
    cand = _clip(value=0.4)
    cand[0, 0, -1, 0, 0] = -0.5  # sampled, so a difference here is legitimate

    ref_frames, idx = bench._extract_frames_uint8(ref, 4)
    cand_frames, _ = bench._extract_frames_uint8(cand, 4)

    assert idx[-1] == 7
    differing = np.argwhere(ref_frames != cand_frames)
    assert differing.tolist() == [[3, 0, 0, 0]], (
        "only the one modified pixel may differ; anything else is the transform "
        "having been chosen from the clip's contents"
    )


def test_out_of_range_values_are_clipped_and_reported(bench, capsys):
    """Clipping keeps the metric well-defined; the warning keeps it honest."""
    clip = _clip(value=0.4)
    clip[0, :, 0] = 4.0

    frames, _ = bench._extract_frames_uint8(clip, 2)

    assert frames[0].min() == frames[0].max() == 255
    assert "outside the [0,1] range" in capsys.readouterr().out


def test_a_batched_and_an_unbatched_clip_extract_identically(bench):
    """[B,C,F,H,W] and [C,F,H,W] are both what callers hand us."""
    batched = _clip()
    unbatched = batched[0]

    a, ia = bench._extract_frames_uint8(batched, 4)
    b, ib = bench._extract_frames_uint8(unbatched, 4)

    assert np.array_equal(a, b) and ia == ib


# --------------------------------------------------------------------------
# run provenance
# --------------------------------------------------------------------------


def test_the_packages_that_decide_correctness_are_recorded(bench):
    """A result that cannot be attributed to a software stack is not evidence."""
    versions = bench._package_versions()

    for name in ("diffusers", "transformers", "numpy", "moviepy", "flash_attn"):
        assert name in versions, f"{name} decides output validity but is unrecorded"
    assert versions["numpy"] == np.__version__


def test_a_missing_package_is_recorded_as_absent_not_omitted(bench):
    """Absence is the datum: which attention kernel resolved depends on it."""
    versions = bench._package_versions()

    assert all(isinstance(v, str) and v for v in versions.values())
    assert set(versions) == set(bench._RECORDED_PACKAGES)
