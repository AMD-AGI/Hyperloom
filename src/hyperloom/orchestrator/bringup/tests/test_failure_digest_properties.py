# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Property tests for the failure-digest invariant.

The digest of a failure must be a function of the failure and nothing else: two
hosts, two sessions, two pids and two log volumes that hit the same wall have to
collapse to one key, or the bring-up path cannot tell "the same wall again" from
"a new, deeper wall". The converse carries equal weight -- a digest that is
stable but not discriminating collapses every failure into one key -- so both
directions are asserted here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from hypothesis import given, settings
from hypothesis import strategies as st

from hyperloom.common.bringup import (
    BootObservation,
    LadderStage,
    TerminalFrame,
    failure_digest,
    normalise_file_rel,
    render_excerpt,
)

# The capture window is anchored at the failure and spends a quarter of its
# width on preceding context. Sizing that lead to a whole number of filler lines
# is what makes "more log before the failure" a no-op: the window then always
# opens on the same line boundary, whatever came before it.
_FILLER_LINE = "preflight ok\n"
_LEAD_LINES = 5
_EXCERPT_WIDTH = 4 * _LEAD_LINES * len(_FILLER_LINE)

# The remaining three quarters must hold the whole failure line at its widest
# draw. If it did not, a longer pid or operand would push bytes past the window
# edge and the digest would move for a reason that is not the failure.
_USABLE = _EXCERPT_WIDTH - _LEAD_LINES * len(_FILLER_LINE)

_EXC_TYPES = ("ValueError", "RuntimeError", "ImportError", "OutOfMemoryError")
_MODULES = ("hyperloom.engine.loader", "sglang.srt.server", "torch.cuda.memory")
_REL_FILES = ("engine/loader.py", "srt/server.py", "cuda/memory.py")
_STAGES = (
    LadderStage.IMPORT,
    LadderStage.CONFIG_VALIDATE,
    LadderStage.WEIGHTS_LOADING,
    LadderStage.ENGINE_INIT,
)
_INSTALL_ROOTS = (
    "/opt/hyperloom/lib/python3.12/site-packages",
    "/home/dev/work/hyperloom/src",
    "/scratch/ci/venv-9f2/lib/python3.11/site-packages",
)


@dataclass(frozen=True)
class _Frame:
    """The part of an observation that identifies which failure it is."""

    exc_type: str
    module: str
    rel_file: str


@dataclass(frozen=True)
class _Noise:
    """The part that identifies which run observed it."""

    slug: str
    run_hex: str
    install_root: str
    pid: int
    clock: int
    want: int
    have: int
    filler_lines: int


@st.composite
def _frames(draw: st.DrawFn) -> _Frame:
    return _Frame(
        exc_type=draw(st.sampled_from(_EXC_TYPES)),
        module=draw(st.sampled_from(_MODULES)),
        rel_file=draw(st.sampled_from(_REL_FILES)),
    )


@st.composite
def _frame_pairs(draw: st.DrawFn) -> tuple[_Frame, _Frame]:
    """A frame and a copy of it that differs in exactly one identifying field."""
    left = draw(_frames())
    field = draw(st.sampled_from(("exc_type", "module", "rel_file")))
    pool = {"exc_type": _EXC_TYPES, "module": _MODULES, "rel_file": _REL_FILES}[field]
    other = draw(st.sampled_from([v for v in pool if v != getattr(left, field)]))
    return left, replace(left, **{field: other})


@st.composite
def _noises(draw: st.DrawFn) -> _Noise:
    return _Noise(
        slug=draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=3, max_size=10)),
        run_hex=draw(st.text(alphabet="0123456789abcdef", min_size=6, max_size=16)),
        install_root=draw(st.sampled_from(_INSTALL_ROOTS)),
        pid=draw(st.integers(min_value=2, max_value=999_999)),
        clock=draw(st.integers(min_value=0, max_value=999_999)),
        want=draw(st.integers(min_value=1, max_value=9_999_999)),
        have=draw(st.integers(min_value=0, max_value=9_999_999)),
        filler_lines=draw(st.integers(min_value=_LEAD_LINES, max_value=_LEAD_LINES + 8)),
    )


def _observe(stage: LadderStage, frame: _Frame, noise: _Noise) -> BootObservation:
    """Render one run's view of the failure ``frame`` under the noise of ``noise``."""
    session = f"/tmp/hyperloom-{noise.slug}/run-{noise.run_hex}"
    stamp = f"2026-09-{noise.clock % 28 + 1:02d}T{noise.clock % 24:02d}:{noise.clock % 60:02d}:{noise.clock % 59:02d}"
    failure = (
        f"[{stamp}] pid={noise.pid} {frame.exc_type}: {session}/weights/shard.bin "
        f"wants {noise.want} bytes, {noise.have} free\n"
    )
    assert len(failure) <= _USABLE
    log = _FILLER_LINE * noise.filler_lines + failure
    return BootObservation(
        producer="ladder",
        stage_reached=LadderStage.WEIGHTS_LOADING,
        stage_failed=stage,
        terminal_frame=TerminalFrame(
            exc_type=frame.exc_type,
            module=frame.module,
            file_rel=normalise_file_rel(f"{noise.install_root}/{frame.rel_file}", [noise.install_root]),
            line=412,
        ),
        matched_marker="weights.shard_too_large",
        excerpt=render_excerpt(
            log,
            anchor=len(_FILLER_LINE) * noise.filler_lines,
            width=_EXCERPT_WIDTH,
            stream="server_log",
            redact_roots=[session],
        ),
        evidence_ref=f"{session}/logs/server.log",
        server_elapsed_sec=noise.clock / 1000.0,
    )


@settings(max_examples=50, deadline=None)
@given(
    stage=st.sampled_from(_STAGES),
    frame=_frames(),
    noises=st.lists(_noises(), min_size=2, max_size=4),
)
def test_digest_is_invariant_under_everything_but_the_failure(
    stage: LadderStage, frame: _Frame, noises: list[_Noise]
) -> None:
    """One failure seen through different sessions, pids, clocks and log volumes."""
    observations = [_observe(stage, frame, noise) for noise in noises]

    assert len({failure_digest(o) for o in observations}) == 1

    # Guard against a vacuous pass: the noise has to be visible somewhere, just
    # not in the digest. The install root is excluded because path normalisation
    # erases it outright -- it is the one dimension that reaches no field.
    if len({replace(n, install_root="") for n in noises}) > 1:
        assert len({repr(o.to_dict()) for o in observations}) > 1


@settings(max_examples=50, deadline=None)
@given(stage=st.sampled_from(_STAGES), pair=_frame_pairs(), noise=_noises())
def test_digest_separates_different_terminal_frames(
    stage: LadderStage, pair: tuple[_Frame, _Frame], noise: _Noise
) -> None:
    """A different exception type, module or normalised file is a different failure."""
    left, right = pair

    assert failure_digest(_observe(stage, left, noise)) != failure_digest(_observe(stage, right, noise))
